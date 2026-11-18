from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import copy
import math
import random
import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
ArrayLike = Any
ActivationName = Literal['relu', 'tanh', 'elu', 'gelu', 'leaky_relu']
SolverName = Literal['adam', 'sgd', 'lbfgs']

def _set_random_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resolved = torch.device(device)
    if resolved.type == 'cuda' and (not torch.cuda.is_available()):
        warnings.warn('CUDA was requested but is unavailable; SRDO falls back to CPU.', RuntimeWarning)
        return torch.device('cpu')
    return resolved

def _to_numpy_2d(X: ArrayLike, name: str='X') -> np.ndarray:
    if hasattr(X, 'to_numpy'):
        array = np.asarray(X.to_numpy(), dtype=float)
    else:
        array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError(f'{name} must be two-dimensional; got shape={array.shape}.')
    if array.shape[0] < 2:
        raise ValueError(f'{name} must contain at least two observations.')
    if array.shape[1] < 1:
        raise ValueError(f'{name} must contain at least one feature.')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} contains NaN or Inf.')
    return array

def _activation(name: ActivationName) -> nn.Module:
    if name == 'relu':
        return nn.ReLU()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'elu':
        return nn.ELU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError('activation must be one of: relu, tanh, elu, gelu, leaky_relu.')

def _effective_sample_size(weights: np.ndarray, eps: float=1e-15) -> float:
    w = np.asarray(weights, dtype=float).ravel()
    return float(w.sum() ** 2 / (np.dot(w, w) + eps))

def _normalize_weights_with_max_share(raw_weights: np.ndarray, max_weight_share: Optional[float]) -> Tuple[np.ndarray, float, float]:
    raw = np.asarray(raw_weights, dtype=float).ravel()
    if raw.size == 0 or not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError('raw_weights must be finite and strictly positive.')
    capped = raw.copy()
    cap_threshold = float('nan')
    if max_weight_share is not None:
        share = float(max_weight_share)
        if not 0.0 < share <= 1.0:
            raise ValueError('max_weight_share must be in (0,1] or None.')
        if share < 1.0 / raw.size:
            raise ValueError('max_weight_share is infeasible for this sample size.')
        if float(np.max(raw) / np.sum(raw)) > share:
            (low, high) = (0.0, float(np.max(raw)))
            for _ in range(100):
                midpoint = 0.5 * (low + high)
                trial = np.minimum(raw, midpoint)
                trial_share = midpoint / float(np.sum(trial))
                if trial_share <= share:
                    low = midpoint
                else:
                    high = midpoint
            cap_threshold = low
            capped = np.minimum(raw, cap_threshold)
    normalization_constant = float(np.mean(capped))
    if normalization_constant <= 0.0 or not np.isfinite(normalization_constant):
        raise FloatingPointError('Weight normalization constant is invalid.')
    normalized = capped / normalization_constant
    if max_weight_share is not None:
        achieved_share = float(np.max(normalized) / np.sum(normalized))
        if achieved_share > float(max_weight_share) + 1e-12:
            raise FloatingPointError('Normalized weights violate max_weight_share.')
    return (normalized, cap_threshold, normalization_constant)

def _stratified_split(labels: np.ndarray, validation_fraction: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int).ravel()
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError('validation_fraction must be in (0, 0.5).')
    rng = np.random.default_rng(int(random_state))
    train_parts: List[np.ndarray] = []
    valid_parts: List[np.ndarray] = []
    for value in np.unique(labels):
        indices = np.flatnonzero(labels == value)
        if indices.size < 2:
            raise ValueError('Each density-classification class needs >= 2 samples.')
        indices = rng.permutation(indices)
        n_valid = int(round(validation_fraction * indices.size))
        n_valid = max(1, min(n_valid, indices.size - 1))
        valid_parts.append(indices[:n_valid])
        train_parts.append(indices[n_valid:])
    train_idx = rng.permutation(np.concatenate(train_parts))
    valid_idx = rng.permutation(np.concatenate(valid_parts))
    return (train_idx.astype(int), valid_idx.astype(int))

def column_wise_resampling(X: ArrayLike, *, sensitive_variables: Optional[Sequence[int]]=None, random_state: int=0) -> np.ndarray:
    X_array = _to_numpy_2d(X)
    (n, p) = X_array.shape
    rng = np.random.default_rng(int(random_state))
    if sensitive_variables is None:
        columns = np.arange(p, dtype=int)
    else:
        columns = np.unique(np.asarray(sensitive_variables, dtype=int))
        if columns.ndim != 1:
            raise ValueError('sensitive_variables must be one-dimensional.')
        if np.any(columns < 0) or np.any(columns >= p):
            raise ValueError('sensitive_variables contains an invalid column index.')
    resampled = X_array.copy()
    for column in columns:
        resampled[:, column] = X_array[rng.permutation(n), column]
    return resampled

class _DensityRatioMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_layer_sizes: Tuple[int, ...], activation: ActivationName, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError('input_dim must be positive.')
        if any((width <= 0 for width in hidden_layer_sizes)):
            raise ValueError('All hidden-layer widths must be positive.')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in [0, 1).')
        layers: List[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_layer_sizes:
            layers.append(nn.Linear(previous, int(width), bias=True))
            layers.append(_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            previous = int(width)
        layers.append(nn.Linear(previous, 1, bias=True))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X).squeeze(-1)

def _network_l2(model: nn.Module) -> torch.Tensor:
    terms: List[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            terms.append(torch.sum(module.weight.square()))
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(terms).sum()

def _full_objective(model: _DensityRatioMLP, X: torch.Tensor, y: torch.Tensor, l2: float) -> torch.Tensor:
    logits = model(X)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
    return bce + 0.5 * float(l2) * _network_l2(model)

def _predict_target_probability(model: _DensityRatioMLP, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    tensor = torch.as_tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=max(1, min(int(batch_size), len(tensor))), shuffle=False)
    parts: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            parts.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(parts).astype(float)

def _train_density_classifier(X: np.ndarray, labels: np.ndarray, *, hidden_layer_sizes: Tuple[int, ...], activation: ActivationName, dropout: float, solver: SolverName, learning_rate: float, l2: float, batch_size: int, max_epochs: int, tol: float, patience: int, early_stopping: bool, validation_fraction: float, gradient_clip: Optional[float], random_state: int, device: torch.device, verbose: bool, log_every: int) -> Tuple[_DensityRatioMLP, Dict[str, float]]:
    X = _to_numpy_2d(X)
    labels = np.asarray(labels, dtype=np.float32).ravel()
    if labels.shape != (X.shape[0],):
        raise ValueError('Density-classification labels have the wrong length.')
    if not np.all(np.isin(np.unique(labels), [0.0, 1.0])):
        raise ValueError('Density-classification labels must be 0/1.')
    if learning_rate <= 0 or l2 < 0 or max_epochs <= 0 or (tol <= 0):
        raise ValueError('Invalid optimizer hyperparameters.')
    if batch_size <= 0 or patience <= 0:
        raise ValueError('batch_size and patience must be positive.')
    if solver not in {'adam', 'sgd', 'lbfgs'}:
        raise ValueError("solver must be 'adam', 'sgd', or 'lbfgs'.")
    _set_random_seed(random_state)
    if early_stopping:
        (train_idx, valid_idx) = _stratified_split(labels.astype(int), validation_fraction, random_state)
    else:
        train_idx = np.arange(X.shape[0], dtype=int)
        valid_idx = np.array([], dtype=int)
    model = _DensityRatioMLP(input_dim=X.shape[1], hidden_layer_sizes=hidden_layer_sizes, activation=activation, dropout=dropout).to(device)
    X_train = torch.as_tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.as_tensor(labels[train_idx], dtype=torch.float32)
    train_dataset = TensorDataset(X_train, y_train)
    generator = torch.Generator()
    generator.manual_seed(int(random_state))
    train_loader = DataLoader(train_dataset, batch_size=min(int(batch_size), len(train_dataset)), shuffle=solver != 'lbfgs', generator=generator, drop_last=False)
    X_train_device = X_train.to(device)
    y_train_device = y_train.to(device)
    if valid_idx.size:
        X_valid_device = torch.as_tensor(X[valid_idx], dtype=torch.float32, device=device)
        y_valid_device = torch.as_tensor(labels[valid_idx], dtype=torch.float32, device=device)
    else:
        X_valid_device = None
        y_valid_device = None
    best_state = copy.deepcopy(model.state_dict())
    best_monitor = math.inf
    no_improvement = 0
    used_epochs = 0
    if solver == 'adam':
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    elif solver == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate), momentum=0.9)
    else:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=float(learning_rate), max_iter=int(max_epochs), tolerance_grad=float(tol), tolerance_change=float(tol), history_size=100, line_search_fn='strong_wolfe')
    if solver == 'lbfgs':

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            objective = _full_objective(model, X_train_device, y_train_device, l2)
            objective.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            return objective
        optimizer.step(closure)
        used_epochs = int(optimizer.state[next(iter(model.parameters()))].get('n_iter', max_epochs))
        best_state = copy.deepcopy(model.state_dict())
    else:
        for epoch in range(1, int(max_epochs) + 1):
            used_epochs = epoch
            model.train()
            for (X_batch, y_batch) in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                objective = _full_objective(model, X_batch, y_batch, l2)
                objective.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
                optimizer.step()
            model.eval()
            with torch.no_grad():
                train_loss = float(_full_objective(model, X_train_device, y_train_device, l2).cpu())
                if X_valid_device is not None:
                    valid_loss = float(_full_objective(model, X_valid_device, y_valid_device, l2).cpu())
                    monitor = valid_loss
                else:
                    valid_loss = float('nan')
                    monitor = train_loss
            improvement = best_monitor - monitor
            if improvement > float(tol):
                best_monitor = monitor
                best_state = copy.deepcopy(model.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1
            if verbose and (epoch == 1 or epoch % max(1, int(log_every)) == 0):
                message = f'SRDO epoch={epoch:4d} | train_loss={train_loss:.6f}'
                if np.isfinite(valid_loss):
                    message += f' | valid_loss={valid_loss:.6f}'
                print(message)
            if no_improvement >= int(patience):
                break
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_train_loss = float(_full_objective(model, X_train_device, y_train_device, l2).cpu())
        if X_valid_device is not None:
            final_valid_loss = float(_full_objective(model, X_valid_device, y_valid_device, l2).cpu())
        else:
            final_valid_loss = float('nan')
    return (model, {'n_epochs': float(used_epochs), 'training_loss': final_train_loss, 'validation_loss': final_valid_loss})

@dataclass
class SRDOResult:
    raw_density_ratio: np.ndarray
    normalized_weights: np.ndarray
    target_probability: np.ndarray
    resampled_X: np.ndarray
    normalization_constant: float
    diagnostics: Dict[str, float]

class SRDO:

    def __init__(self, *, hidden_layer_sizes: Sequence[int]=(100, 5), activation: ActivationName='relu', dropout: float=0.0, solver: SolverName='adam', learning_rate: float=0.001, l2: float=0.0001, batch_size: int=200, max_epochs: int=500, tol: float=1e-05, patience: int=30, early_stopping: bool=False, validation_fraction: float=0.1, gradient_clip: Optional[float]=5.0, probability_clip: float=1e-06, weight_clip_quantile: Optional[float]=None, max_weight_share: Optional[float]=None, sensitive_variables: Optional[Sequence[int]]=None, random_state: int=0, device: Optional[Union[str, torch.device]]=None, verbose: bool=False, log_every: int=50) -> None:
        hidden = tuple((int(value) for value in hidden_layer_sizes))
        if any((value <= 0 for value in hidden)):
            raise ValueError('hidden_layer_sizes must contain positive integers.')
        if not 0.0 < probability_clip < 0.5:
            raise ValueError('probability_clip must be in (0, 0.5).')
        if weight_clip_quantile is not None and (not 0.5 < weight_clip_quantile < 1.0):
            raise ValueError('weight_clip_quantile must be in (0.5, 1) or None.')
        if max_weight_share is not None and (not 0.0 < float(max_weight_share) <= 1.0):
            raise ValueError('max_weight_share must be in (0,1] or None.')
        self.hidden_layer_sizes = hidden
        self.activation = activation
        self.dropout = float(dropout)
        self.solver = solver
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.tol = float(tol)
        self.patience = int(patience)
        self.early_stopping = bool(early_stopping)
        self.validation_fraction = float(validation_fraction)
        self.gradient_clip = gradient_clip
        self.probability_clip = float(probability_clip)
        self.weight_clip_quantile = weight_clip_quantile
        self.max_weight_share = max_weight_share
        self.sensitive_variables = sensitive_variables
        self.random_state = int(random_state)
        self.device = _resolve_device(device)
        self.verbose = bool(verbose)
        self.log_every = int(log_every)
        self.model_: Optional[_DensityRatioMLP] = None
        self.result_: Optional[SRDOResult] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.zero_variance_mask_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def _prepare_fit_X(self, X: ArrayLike) -> np.ndarray:
        array = _to_numpy_2d(X)
        self.n_features_in_ = int(array.shape[1])
        mean = np.zeros(array.shape[1], dtype=float)
        scale = np.ones(array.shape[1], dtype=float)
        zero_variance = np.zeros(array.shape[1], dtype=bool)
        transformed = array.copy()
        self.mean_ = mean
        self.scale_ = scale
        self.zero_variance_mask_ = zero_variance
        return transformed

    def _prepare_predict_X(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        array = _to_numpy_2d(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {array.shape[1]} features, expected {self.n_features_in_}.')
        return array

    def fit(self, X: ArrayLike) -> 'SRDO':
        _set_random_seed(self.random_state)
        X_input_scale = self._prepare_fit_X(X)
        (n, _) = X_input_scale.shape
        X_tilde = column_wise_resampling(X_input_scale, sensitive_variables=self.sensitive_variables, random_state=self.random_state)
        classification_X = np.vstack([X_input_scale, X_tilde])
        labels = np.concatenate([np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)])
        if self.verbose:
            print(f'[*] SRDO start: n={n}, p={X_input_scale.shape[1]}, device={self.device}, solver={self.solver}')
        (model, training_info) = _train_density_classifier(classification_X, labels, hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, dropout=self.dropout, solver=self.solver, learning_rate=self.learning_rate, l2=self.l2, batch_size=self.batch_size, max_epochs=self.max_epochs, tol=self.tol, patience=self.patience, early_stopping=self.early_stopping, validation_fraction=self.validation_fraction, gradient_clip=self.gradient_clip, random_state=self.random_state, device=self.device, verbose=self.verbose, log_every=self.log_every)
        self.model_ = model
        q_all = _predict_target_probability(model, classification_X, self.device, self.batch_size)
        q_original = np.clip(q_all[:n], self.probability_clip, 1.0 - self.probability_clip)
        raw_ratio = q_original / (1.0 - q_original)
        if not np.all(np.isfinite(raw_ratio)) or np.any(raw_ratio <= 0.0):
            raise FloatingPointError('SRDO produced invalid density ratios.')
        clipped_ratio = raw_ratio.copy()
        clipping_threshold = float('nan')
        if self.weight_clip_quantile is not None:
            clipping_threshold = float(np.quantile(clipped_ratio, self.weight_clip_quantile))
            clipped_ratio = np.minimum(clipped_ratio, clipping_threshold)
        (normalized_weights, share_clip_threshold, ratio_mean) = _normalize_weights_with_max_share(clipped_ratio, self.max_weight_share)
        predicted_labels = (q_all >= 0.5).astype(int)
        classifier_accuracy = float(np.mean(predicted_labels == labels.astype(int)))
        ess = _effective_sample_size(normalized_weights)
        diagnostics = {**training_info, 'n_samples': float(n), 'n_features': float(X_input_scale.shape[1]), 'density_classifier_accuracy': classifier_accuracy, 'target_probability_min': float(np.min(q_original)), 'target_probability_mean': float(np.mean(q_original)), 'target_probability_max': float(np.max(q_original)), 'raw_ratio_min': float(np.min(raw_ratio)), 'raw_ratio_mean': float(np.mean(raw_ratio)), 'raw_ratio_max': float(np.max(raw_ratio)), 'weight_min': float(np.min(normalized_weights)), 'weight_mean': float(np.mean(normalized_weights)), 'weight_max': float(np.max(normalized_weights)), 'weight_sum': float(np.sum(normalized_weights)), 'ess': ess, 'ess_ratio': ess / n, 'probability_clip': self.probability_clip, 'weight_clip_threshold': clipping_threshold, 'max_weight_share': self.max_weight_share, 'max_share_clip_threshold': share_clip_threshold, 'achieved_max_weight_share': float(np.max(normalized_weights) / np.sum(normalized_weights)), 'device_is_cuda': float(self.device.type == 'cuda')}
        self.result_ = SRDOResult(raw_density_ratio=raw_ratio.astype(float), normalized_weights=normalized_weights.astype(float), target_probability=q_original.astype(float), resampled_X=X_tilde.astype(float), normalization_constant=float(ratio_mean), diagnostics=diagnostics)
        if self.verbose:
            print(f'[✓] SRDO finished | classifier_acc={classifier_accuracy:.4f} | W min/mean/max={normalized_weights.min():.4f}/{normalized_weights.mean():.4f}/{normalized_weights.max():.4f} | ESS={ess:.1f}/{n}')
        return self

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        return self.fit(X).get_weights()

    def _check_fitted(self) -> None:
        if self.model_ is None or self.result_ is None:
            raise ValueError('SRDO is not fitted; call fit(X) first.')

    def predict_target_probability(self, X: ArrayLike) -> np.ndarray:
        X_input_scale = self._prepare_predict_X(X)
        probability = _predict_target_probability(self.model_, X_input_scale, self.device, self.batch_size)
        return np.clip(probability, self.probability_clip, 1.0 - self.probability_clip)

    def predict_density_ratio(self, X: ArrayLike, normalize: bool=False) -> np.ndarray:
        self._check_fitted()
        q = self.predict_target_probability(X)
        ratio = q / (1.0 - q)
        if self.weight_clip_quantile is not None:
            threshold = float(self.result_.diagnostics['weight_clip_threshold'])
            ratio = np.minimum(ratio, threshold)
        if normalize:
            ratio = ratio / float(self.result_.normalization_constant)
        return ratio.astype(float)

    def get_weights(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.normalized_weights.copy()

    def get_raw_density_ratio(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.raw_density_ratio.copy()

    def get_target_probability(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.target_probability.copy()

    def get_resampled_data(self, original_scale: bool=False) -> np.ndarray:
        self._check_fitted()
        result = self.result_.resampled_X.copy()
        if original_scale:
            result = result * self.scale_ + self.mean_
        return result

    def get_effective_sample_size(self) -> float:
        self._check_fitted()
        return float(self.result_.diagnostics['ess'])

    def get_diagnostics(self) -> Dict[str, float]:
        self._check_fitted()
        return dict(self.result_.diagnostics)

def fit_srdo(X: ArrayLike, **kwargs: Any) -> SRDO:
    return SRDO(**kwargs).fit(X)

def estimate_srdo_weights(X: ArrayLike, *, return_model: bool=False, **kwargs: Any):
    fitted = fit_srdo(X, **kwargs)
    weights = fitted.get_weights()
    if return_model:
        return (weights, fitted)
    return weights
__all__ = ['SRDOResult', 'SRDO', 'column_wise_resampling', 'fit_srdo', 'estimate_srdo_weights']
