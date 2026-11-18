from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import math
import random
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
ArrayLike = Any
ActivationName = Literal['relu', 'tanh', 'elu', 'gelu']

def _set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _effective_sample_size(weights: np.ndarray, eps: float=1e-15) -> float:
    w = np.asarray(weights, dtype=float).ravel()
    return float(w.sum() ** 2 / (np.sum(w ** 2) + eps))

def _normalize_weights_with_max_share(raw_weights: np.ndarray, max_weight_share: Optional[float]) -> Tuple[np.ndarray, float]:
    raw = np.asarray(raw_weights, dtype=float).ravel()
    if raw.size == 0 or not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError('raw_weights must be finite and strictly positive.')
    capped = raw.copy()
    cap_threshold = float('nan')
    if max_weight_share is not None:
        share = float(max_weight_share)
        if not 0.0 < share <= 1.0:
            raise ValueError('max_weight_share must lie in (0,1] or be None.')
        if share < 1.0 / raw.size:
            raise ValueError('max_weight_share is infeasible for the current sample size.')
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
    mean_weight = float(np.mean(capped))
    if mean_weight <= 0.0 or not np.isfinite(mean_weight):
        raise FloatingPointError('Invalid mean of SRDO weights.')
    weights = capped / mean_weight
    if max_weight_share is not None:
        achieved_share = float(np.max(weights) / np.sum(weights))
        if achieved_share > float(max_weight_share) + 1e-12:
            raise FloatingPointError('Normalized weights violate max_weight_share.')
    return (weights, cap_threshold)

def _to_numpy_2d(X: Any, name: str='X') -> np.ndarray:
    if hasattr(X, 'to_numpy'):
        arr = np.asarray(X.to_numpy(), dtype=float)
    else:
        arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f'{name} must be a two-dimensional matrix; got shape={arr.shape}.')
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f'{name} must contain at least one sample and one variable.')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} contains NaN or Inf.')
    return arr

def _to_binary_y(y: Any, n: int) -> np.ndarray:
    if hasattr(y, 'to_numpy'):
        arr = np.asarray(y.to_numpy()).ravel()
    else:
        arr = np.asarray(y).ravel()
    if arr.shape != (n,):
        raise ValueError(f'Length of y must equal the sample size n={n}; got shape={arr.shape}.')
    if not np.all(np.isfinite(arr.astype(float))):
        raise ValueError('y contains NaN or Inf.')
    unique = np.unique(arr)
    if np.all(np.isin(unique, [-1, 1])):
        arr = (arr == 1).astype(int)
    elif np.all(np.isin(unique, [0, 1])):
        arr = arr.astype(int)
    else:
        raise ValueError(f'y must be coded as {{0,1}} or {{-1,+1}}; observed labels={unique.tolist()}.')
    if np.unique(arr).size != 2:
        raise ValueError('y must contain both classes.')
    return arr

def _activation_layer(name: ActivationName) -> nn.Module:
    if name == 'relu':
        return nn.ReLU()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'elu':
        return nn.ELU()
    if name == 'gelu':
        return nn.GELU()
    raise ValueError("activation must be 'relu', 'tanh', 'elu' or 'gelu'.")

def column_wise_resampling(X: np.ndarray, random_state: int=0) -> np.ndarray:
    X = _to_numpy_2d(X)
    rng = np.random.default_rng(int(random_state))
    (n, p) = X.shape
    X_independent = np.empty_like(X)
    for j in range(p):
        X_independent[:, j] = X[rng.permutation(n), j]
    return X_independent

class _SRDODensityNetwork(nn.Module):

    def __init__(self, input_dim: int, hidden_layer_sizes: Tuple[int, ...], activation: ActivationName='relu', dropout: float=0.0):
        super().__init__()
        if input_dim <= 0:
            raise ValueError('input_dim must be positive.')
        if not hidden_layer_sizes:
            raise ValueError('SRDO hidden_layer_sizes must not be empty.')
        if any((int(v) <= 0 for v in hidden_layer_sizes)):
            raise ValueError('Widths in SRDO hidden_layer_sizes must be positive integers.')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('srdo_dropout must lie in [0,1).')
        layers: List[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_layer_sizes:
            width = int(width)
            layers.append(nn.Linear(previous, width, bias=True))
            layers.append(_activation_layer(activation))
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            previous = width
        layers.append(nn.Linear(previous, 1, bias=True))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X).squeeze(-1)

@dataclass
class _TorchSRDOClassifier:
    model: _SRDODensityNetwork
    device: torch.device
    n_iter_: int
    training_loss_: float
    validation_loss_: float

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1], dtype=int)

    def predict_proba(self, X: np.ndarray, batch_size: int=4096) -> np.ndarray:
        X_arr = _to_numpy_2d(X)
        tensor = torch.as_tensor(X_arr, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=max(1, min(int(batch_size), len(tensor))), shuffle=False)
        p1_parts: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                p1 = torch.sigmoid(logits).detach().cpu().numpy()
                p1_parts.append(p1)
        p1_all = np.concatenate(p1_parts, axis=0).astype(float)
        return np.column_stack([1.0 - p1_all, p1_all])

def _stratified_validation_split(y: np.ndarray, validation_fraction: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int).ravel()
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError('validation_fraction must lie in (0,0.5).')
    rng = np.random.default_rng(int(random_state))
    train_parts: List[np.ndarray] = []
    valid_parts: List[np.ndarray] = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if idx.size < 2:
            raise ValueError('Early stopping requires at least 2 samples per class.')
        idx = rng.permutation(idx)
        n_valid = int(round(validation_fraction * idx.size))
        n_valid = max(1, min(n_valid, idx.size - 1))
        valid_parts.append(idx[:n_valid])
        train_parts.append(idx[n_valid:])
    train_idx = rng.permutation(np.concatenate(train_parts))
    valid_idx = rng.permutation(np.concatenate(valid_parts))
    return (train_idx.astype(int), valid_idx.astype(int))

def _srdo_l2_penalty(model: nn.Module) -> torch.Tensor:
    terms: List[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            terms.append(torch.sum(module.weight ** 2))
    if not terms:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(terms).sum()

def _evaluate_srdo_bce(model: _SRDODensityNetwork, X: torch.Tensor, y: torch.Tensor, *, alpha: float) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
        penalty = 0.5 * float(alpha) * _srdo_l2_penalty(model)
        loss = bce + penalty
    return float(loss.detach().cpu())

def _train_srdo_torch_classifier(X: np.ndarray, labels: np.ndarray, *, hidden_layer_sizes: Tuple[int, ...], activation: ActivationName, dropout: float, solver: Literal['adam', 'lbfgs', 'sgd'], alpha: float, learning_rate_init: float, max_iter: int, tol: float, batch_size: int, early_stopping: bool, validation_fraction: float, n_iter_no_change: int, gradient_clip: Optional[float], random_state: int, device: torch.device, verbose: bool) -> _TorchSRDOClassifier:
    X = _to_numpy_2d(X)
    labels = np.asarray(labels, dtype=np.float32).ravel()
    if labels.shape != (X.shape[0],):
        raise ValueError('Length of SRDO labels does not match X.')
    if not np.all(np.isin(np.unique(labels), [0.0, 1.0])):
        raise ValueError('SRDO labels must be 0/1.')
    if max_iter <= 0 or tol <= 0 or learning_rate_init <= 0:
        raise ValueError('SRDO max_iter, tol and learning_rate_init must be positive.')
    if alpha < 0:
        raise ValueError('srdo_alpha must be >= 0.')
    if batch_size <= 0:
        raise ValueError('srdo_batch_size must be positive.')
    if n_iter_no_change <= 0:
        raise ValueError('srdo_n_iter_no_change must be positive.')
    if solver not in {'adam', 'sgd', 'lbfgs'}:
        raise ValueError("srdo_solver must be 'adam', 'sgd' or 'lbfgs'.")
    _set_random_seed(int(random_state))
    if early_stopping:
        (train_idx, valid_idx) = _stratified_validation_split(labels.astype(int), validation_fraction=validation_fraction, random_state=random_state)
    else:
        train_idx = np.arange(X.shape[0], dtype=int)
        valid_idx = np.array([], dtype=int)
    X_train = torch.as_tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.as_tensor(labels[train_idx], dtype=torch.float32)
    train_dataset = TensorDataset(X_train, y_train)
    generator = torch.Generator()
    generator.manual_seed(int(random_state))
    train_loader = DataLoader(train_dataset, batch_size=min(int(batch_size), len(train_dataset)), shuffle=solver != 'lbfgs', generator=generator, drop_last=False)
    if valid_idx.size > 0:
        X_valid_dev = torch.as_tensor(X[valid_idx], dtype=torch.float32, device=device)
        y_valid_dev = torch.as_tensor(labels[valid_idx], dtype=torch.float32, device=device)
    else:
        X_valid_dev = None
        y_valid_dev = None
    model = _SRDODensityNetwork(input_dim=X.shape[1], hidden_layer_sizes=hidden_layer_sizes, activation=activation, dropout=dropout).to(device)
    if solver == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate_init), weight_decay=0.0)
    elif solver == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate_init), momentum=0.9, weight_decay=0.0)
    else:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=float(learning_rate_init), max_iter=20, history_size=50, line_search_fn='strong_wolfe')
    best_state = {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}
    best_monitor = np.inf
    no_improve = 0
    final_train_loss = np.inf
    final_valid_loss = np.nan
    used_epoch = 0
    if solver == 'lbfgs':
        X_full_dev = X_train.to(device)
        y_full_dev = y_train.to(device)
    for epoch in range(1, int(max_iter) + 1):
        used_epoch = epoch
        model.train()
        if solver == 'lbfgs':

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                logits = model(X_full_dev)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, y_full_dev)
                objective = bce + 0.5 * float(alpha) * _srdo_l2_penalty(model)
                objective.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
                return objective
            optimizer.step(closure)
        else:
            for (xb, yb) in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, yb)
                objective = bce + 0.5 * float(alpha) * _srdo_l2_penalty(model)
                objective.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
                optimizer.step()
        X_train_dev = X_train.to(device)
        y_train_dev = y_train.to(device)
        final_train_loss = _evaluate_srdo_bce(model, X_train_dev, y_train_dev, alpha=alpha)
        if X_valid_dev is not None:
            final_valid_loss = _evaluate_srdo_bce(model, X_valid_dev, y_valid_dev, alpha=alpha)
            monitor = final_valid_loss
        else:
            monitor = final_train_loss
        improvement = best_monitor - monitor
        if improvement > float(tol):
            best_monitor = monitor
            best_state = {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if verbose and (epoch == 1 or epoch % 50 == 0):
            msg = f'      SRDO-PyTorch epoch={epoch:4d} | train={final_train_loss:.6f}'
            if np.isfinite(final_valid_loss):
                msg += f' | valid={final_valid_loss:.6f}'
            print(msg)
        if no_improve >= int(n_iter_no_change):
            break
    model.load_state_dict(best_state)
    model.to(device)
    final_train_loss = _evaluate_srdo_bce(model, X_train.to(device), y_train.to(device), alpha=alpha)
    if X_valid_dev is not None:
        final_valid_loss = _evaluate_srdo_bce(model, X_valid_dev, y_valid_dev, alpha=alpha)
    return _TorchSRDOClassifier(model=model, device=device, n_iter_=int(used_epoch), training_loss_=float(final_train_loss), validation_loss_=float(final_valid_loss))

def estimate_srdo_weights(X: ArrayLike, *, hidden_layer_sizes: Sequence[int]=(100, 5), activation: ActivationName='relu', dropout: float=0.0, solver: Literal['adam', 'lbfgs', 'sgd']='adam', alpha: float=0.0001, learning_rate_init: float=0.001, max_iter: int=500, tol: float=0.0001, batch_size: int=200, early_stopping: bool=False, validation_fraction: float=0.1, n_iter_no_change: int=20, gradient_clip: Optional[float]=5.0, cross_fit_folds: int=1, probability_clip: float=1e-06, weight_clip_quantile: Optional[float]=None, max_weight_share: Optional[float]=None, random_state: int=0, device: Optional[Union[str, torch.device]]=None, verbose: bool=False) -> Tuple[np.ndarray, Dict[str, float], Union[_TorchSRDOClassifier, List[_TorchSRDOClassifier]]]:
    X_arr = _to_numpy_2d(X)
    (n, _) = X_arr.shape
    hidden_layer_sizes = tuple((int(v) for v in hidden_layer_sizes))
    if not hidden_layer_sizes or any((v <= 0 for v in hidden_layer_sizes)):
        raise ValueError('SRDO hidden_layer_sizes must be a non-empty sequence of positive integers.')
    if max_iter <= 0 or tol <= 0 or alpha < 0 or (learning_rate_init <= 0):
        raise ValueError('SRDO max_iter, tol and learning_rate_init must be positive, and alpha must be nonnegative.')
    if batch_size <= 0:
        raise ValueError('SRDO batch_size must be positive.')
    if cross_fit_folds < 1:
        raise ValueError('cross_fit_folds must be >= 1.')
    if not 0.0 < probability_clip < 0.5:
        raise ValueError('probability_clip must lie in (0,0.5).')
    if weight_clip_quantile is not None and (not 0.5 < weight_clip_quantile < 1.0):
        raise ValueError('weight_clip_quantile must lie in (0.5,1) or be None.')
    if max_weight_share is not None and (not 0.0 < float(max_weight_share) <= 1.0):
        raise ValueError('max_weight_share must lie in (0,1] or be None.')
    if device is None:
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch_device = torch.device(device)
        if torch_device.type == 'cuda' and (not torch.cuda.is_available()):
            warnings.warn('CUDA is unavailable; SRDO falls back to CPU.', RuntimeWarning)
            torch_device = torch.device('cpu')
    X_target = column_wise_resampling(X_arr, random_state=random_state)
    Z_data = np.vstack([X_arr, X_target])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    models: List[_TorchSRDOClassifier] = []
    if cross_fit_folds == 1:
        clf = _train_srdo_torch_classifier(Z_data, labels, hidden_layer_sizes=hidden_layer_sizes, activation=activation, dropout=dropout, solver=solver, alpha=alpha, learning_rate_init=learning_rate_init, max_iter=max_iter, tol=tol, batch_size=batch_size, early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, gradient_clip=gradient_clip, random_state=random_state, device=torch_device, verbose=verbose)
        p_source_all = clf.predict_proba(Z_data)[:, 1]
        p_source = p_source_all[:n]
        models.append(clf)
        density_classifier_accuracy = float(np.mean((p_source_all >= 0.5).astype(int) == labels))
        density_classifier_iterations = float(clf.n_iter_)
        density_classifier_train_loss = float(clf.training_loss_)
        density_classifier_valid_loss = float(clf.validation_loss_)
    else:
        class_counts = np.bincount(labels, minlength=2)
        if int(np.min(class_counts)) < cross_fit_folds:
            raise ValueError('The number of samples per class must be at least cross_fit_folds.')
        splitter = StratifiedKFold(n_splits=int(cross_fit_folds), shuffle=True, random_state=int(random_state))
        p_source_oof = np.empty(2 * n, dtype=float)
        iteration_list: List[float] = []
        train_loss_list: List[float] = []
        valid_loss_list: List[float] = []
        for (fold_id, (train_idx, valid_idx)) in enumerate(splitter.split(Z_data, labels)):
            clf = _train_srdo_torch_classifier(Z_data[train_idx], labels[train_idx], hidden_layer_sizes=hidden_layer_sizes, activation=activation, dropout=dropout, solver=solver, alpha=alpha, learning_rate_init=learning_rate_init, max_iter=max_iter, tol=tol, batch_size=batch_size, early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, gradient_clip=gradient_clip, random_state=random_state + fold_id + 1, device=torch_device, verbose=verbose)
            p_source_oof[valid_idx] = clf.predict_proba(Z_data[valid_idx])[:, 1]
            iteration_list.append(float(clf.n_iter_))
            train_loss_list.append(float(clf.training_loss_))
            valid_loss_list.append(float(clf.validation_loss_))
            models.append(clf)
        p_source = p_source_oof[:n]
        density_classifier_accuracy = float(np.mean((p_source_oof >= 0.5).astype(int) == labels))
        density_classifier_iterations = float(np.nanmean(iteration_list))
        density_classifier_train_loss = float(np.nanmean(train_loss_list))
        valid_arr = np.asarray(valid_loss_list, dtype=float)
        density_classifier_valid_loss = float(np.nanmean(valid_arr)) if np.any(np.isfinite(valid_arr)) else float('nan')
    p_source = np.clip(p_source, probability_clip, 1.0 - probability_clip)
    raw_weights = (1.0 - p_source) / p_source
    if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights <= 0):
        raise FloatingPointError('SRDO density-ratio weights contain non-finite or non-positive values.')
    clipped_weights = raw_weights.copy()
    clipping_threshold = float('nan')
    if weight_clip_quantile is not None:
        clipping_threshold = float(np.quantile(clipped_weights, weight_clip_quantile))
        clipped_weights = np.minimum(clipped_weights, clipping_threshold)
    (weights, share_clip_threshold) = _normalize_weights_with_max_share(clipped_weights, max_weight_share)
    ess = _effective_sample_size(weights)
    diagnostics = {'n_samples': float(n), 'density_classifier_accuracy': density_classifier_accuracy, 'density_classifier_mean_iterations': density_classifier_iterations, 'density_classifier_train_loss': density_classifier_train_loss, 'density_classifier_validation_loss': density_classifier_valid_loss, 'raw_weight_min': float(np.min(raw_weights)), 'raw_weight_mean': float(np.mean(raw_weights)), 'raw_weight_max': float(np.max(raw_weights)), 'weight_min': float(np.min(weights)), 'weight_mean': float(np.mean(weights)), 'weight_max': float(np.max(weights)), 'weight_sum': float(np.sum(weights)), 'ess': ess, 'ess_ratio': ess / n, 'probability_clip': probability_clip, 'weight_clip_threshold': clipping_threshold, 'max_weight_share': max_weight_share, 'max_share_clip_threshold': share_clip_threshold, 'achieved_max_weight_share': float(np.max(weights) / np.sum(weights)), 'cross_fit_folds': float(cross_fit_folds), 'device_is_cuda': float(torch_device.type == 'cuda')}
    return (weights.astype(float), diagnostics, models[0] if len(models) == 1 else models)

class _GaussianStochasticGate(nn.Module):

    def __init__(self, input_dim: int, sigma: float, mu_init: float=0.0):
        super().__init__()
        if input_dim <= 0:
            raise ValueError('input_dim must be positive.')
        if sigma <= 0:
            raise ValueError('gate sigma must be positive.')
        self.sigma = float(sigma)
        self.mu = nn.Parameter(torch.full((input_dim,), float(mu_init)))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.training:
            epsilon = torch.randn_like(self.mu)
            gates = torch.clamp(self.mu + self.sigma * epsilon, 0.0, 1.0)
        else:
            gates = torch.clamp(self.mu, 0.0, 1.0)
        return X * gates

    def inclusion_probabilities(self) -> torch.Tensor:
        z = self.mu / self.sigma
        return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

class _BinarySTGNetwork(nn.Module):

    def __init__(self, input_dim: int, hidden_layer_sizes: Tuple[int, ...], activation: ActivationName, dropout: float, gate_sigma: float, gate_mu_init: float):
        super().__init__()
        self.gate = _GaussianStochasticGate(input_dim=input_dim, sigma=gate_sigma, mu_init=gate_mu_init)
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must lie in [0,1).')
        layers: List[nn.Module] = []
        previous = input_dim
        for width in hidden_layer_sizes:
            width = int(width)
            if width <= 0:
                raise ValueError('Widths in selector_hidden_layer_sizes must be positive.')
            layers.append(nn.Linear(previous, width, bias=True))
            layers.append(_activation_layer(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 1, bias=True))
        self.predictor = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X: torch.Tensor, use_gate: bool=True) -> torch.Tensor:
        if use_gate:
            X = self.gate(X)
        return self.predictor(X).squeeze(-1)

def _weighted_binary_loss(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    individual = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
    return torch.mean(weights * individual)

def _train_binary_stg_selector(X: np.ndarray, y: np.ndarray, weights: np.ndarray, *, hidden_layer_sizes: Tuple[int, ...], activation: ActivationName, dropout: float, gate_sigma: float, gate_mu_init: float, lambda_stg: float, regularizer_normalization: Literal['sum', 'mean'], pretrain_epochs: int, max_epochs: int, min_epochs: int, batch_size: Optional[int], predictor_lr: float, gate_lr: float, weight_decay: float, tol: float, patience: int, gradient_clip: Optional[float], random_state: int, device: torch.device, verbose: bool, log_every: int) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]], _BinarySTGNetwork]:
    (n, p) = X.shape
    if lambda_stg < 0:
        raise ValueError('lambda_stg must be >= 0.')
    if pretrain_epochs < 0 or max_epochs <= 0 or min_epochs < 0:
        raise ValueError('pretrain_epochs/min_epochs must be nonnegative and max_epochs must be positive.')
    if min_epochs > max_epochs:
        raise ValueError('min_epochs cannot exceed max_epochs.')
    if predictor_lr <= 0 or gate_lr <= 0 or tol <= 0:
        raise ValueError('predictor_lr, gate_lr and tol must be positive.')
    if patience <= 0:
        raise ValueError('patience must be positive.')
    if regularizer_normalization not in {'sum', 'mean'}:
        raise ValueError("regularizer_normalization must be 'sum' or 'mean'.")
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    w_tensor = torch.as_tensor(weights, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
    if batch_size is None or batch_size <= 0:
        actual_batch_size = n
    else:
        actual_batch_size = min(int(batch_size), n)
    generator = torch.Generator()
    generator.manual_seed(int(random_state))
    loader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, generator=generator, drop_last=False)
    n_batches = max(1, len(loader))
    model = _BinarySTGNetwork(input_dim=p, hidden_layer_sizes=hidden_layer_sizes, activation=activation, dropout=dropout, gate_sigma=gate_sigma, gate_mu_init=gate_mu_init).to(device)
    if pretrain_epochs > 0:
        pre_optimizer = torch.optim.Adam(model.predictor.parameters(), lr=float(predictor_lr), weight_decay=float(weight_decay))
        model.train()
        for _ in range(pretrain_epochs):
            for (xb, yb, wb) in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                pre_optimizer.zero_grad(set_to_none=True)
                logits = model(xb, use_gate=False)
                loss = _weighted_binary_loss(logits, yb, wb)
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), float(gradient_clip))
                pre_optimizer.step()
    optimizer = torch.optim.Adam([{'params': model.predictor.parameters(), 'lr': float(predictor_lr), 'weight_decay': float(weight_decay)}, {'params': [model.gate.mu], 'lr': float(gate_lr), 'weight_decay': 0.0}])
    history: List[Dict[str, float]] = []
    previous_probs: Optional[np.ndarray] = None
    previous_objective = np.inf
    stable_count = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_data_loss = 0.0
        epoch_objective = 0.0
        for (xb, yb, wb) in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, use_gate=True)
            data_loss = _weighted_binary_loss(logits, yb, wb)
            inclusion = model.gate.inclusion_probabilities()
            if regularizer_normalization == 'mean':
                sparsity_penalty = torch.mean(inclusion)
            else:
                sparsity_penalty = torch.sum(inclusion)
            objective = data_loss + float(lambda_stg) * sparsity_penalty
            objective.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            epoch_data_loss += float(data_loss.detach().cpu()) / n_batches
            epoch_objective += float(objective.detach().cpu()) / n_batches
        with torch.no_grad():
            probs = model.gate.inclusion_probabilities().detach().cpu().numpy()
            if regularizer_normalization == 'mean':
                reg_value = float(np.mean(probs))
            else:
                reg_value = float(np.sum(probs))
        if previous_probs is None:
            max_gate_change = np.inf
        else:
            max_gate_change = float(np.max(np.abs(probs - previous_probs)))
        objective_change = np.inf if not np.isfinite(previous_objective) else abs(epoch_objective - previous_objective)
        history.append({'epoch': float(epoch), 'objective': float(epoch_objective), 'weighted_bce': float(epoch_data_loss), 'gate_regularizer': reg_value, 'max_gate_probability_change': float(max_gate_change), 'objective_change': float(objective_change), 'mean_gate_probability': float(np.mean(probs)), 'min_gate_probability': float(np.min(probs)), 'max_gate_probability': float(np.max(probs))})
        if verbose and (epoch == 1 or epoch % max(1, log_every) == 0):
            print(f'  SVI selector epoch={epoch:4d} | obj={epoch_objective:.6f} | BCE={epoch_data_loss:.6f} | reg={reg_value:.4f} | dGate={max_gate_change:.3e}')
        if epoch >= min_epochs and max_gate_change < tol and (objective_change < tol):
            stable_count += 1
        else:
            stable_count = 0
        if stable_count >= patience:
            if verbose:
                print(f'  SVI selector converged at epoch {epoch}: dGate={max_gate_change:.3e}, dObj={objective_change:.3e}')
            break
        previous_probs = probs.copy()
        previous_objective = float(epoch_objective)
    with torch.no_grad():
        gate_mu = model.gate.mu.detach().cpu().numpy().astype(float)
        gate_probs = model.gate.inclusion_probabilities().detach().cpu().numpy().astype(float)
    if not np.all(np.isfinite(gate_mu)) or not np.all(np.isfinite(gate_probs)):
        raise FloatingPointError('STG selector produced NaN/Inf.')
    return (gate_mu, gate_probs, history, model)

def _fit_final_logistic(X: np.ndarray, y: np.ndarray, *, sample_weight: Optional[np.ndarray], l2: float, max_iter: int, tol: float) -> Dict[str, Any]:
    (n, p) = X.shape
    if l2 < 0:
        raise ValueError('final_l2 must be >= 0.')
    if max_iter <= 0 or tol <= 0:
        raise ValueError('final_max_iter and final_tol must be positive.')
    if sample_weight is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float).ravel()
        if w.shape != (n,):
            raise ValueError('Length of final sample_weight must equal n.')
        if np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
            raise ValueError('final sample_weight must be finite and nonnegative with a positive sum.')
    w_sum = float(w.sum())

    def objective(theta: np.ndarray) -> float:
        intercept = theta[0]
        beta = theta[1:]
        eta = intercept + X @ beta
        nll = np.sum(w * (np.logaddexp(0.0, eta) - y * eta)) / w_sum
        penalty = 0.5 * float(l2) * float(np.dot(beta, beta))
        return float(nll + penalty)

    def gradient(theta: np.ndarray) -> np.ndarray:
        intercept = theta[0]
        beta = theta[1:]
        eta = intercept + X @ beta
        residual = expit(eta) - y
        grad_intercept = float(np.sum(w * residual) / w_sum)
        grad_beta = X.T @ (w * residual) / w_sum + float(l2) * beta
        return np.concatenate(([grad_intercept], grad_beta))
    positive_rate = float(np.clip(np.average(y, weights=w), 1e-06, 1.0 - 1e-06))
    theta0 = np.zeros(p + 1, dtype=float)
    theta0[0] = math.log(positive_rate / (1.0 - positive_rate))
    result = minimize(objective, theta0, jac=gradient, method='L-BFGS-B', options={'maxiter': int(max_iter), 'ftol': float(tol), 'gtol': float(tol)})
    if not result.success:
        warnings.warn('Final logistic L-BFGS-B did not report successful convergence: ' + str(result.message), RuntimeWarning)
    if not np.all(np.isfinite(result.x)):
        raise FloatingPointError('Final logistic parameters contain NaN/Inf.')
    return {'intercept': float(result.x[0]), 'beta': np.asarray(result.x[1:], dtype=float), 'success': bool(result.success), 'message': str(result.message), 'n_iter': float(getattr(result, 'nit', np.nan)), 'objective': float(result.fun)}

@dataclass
class SVILogisticResult:
    selected_features: np.ndarray
    selected_feature_names: Optional[List[str]]
    srdo_weights: np.ndarray
    gate_mu: np.ndarray
    gate_probabilities: np.ndarray
    final_intercept_input_scale: float
    final_beta_selected_input_scale: np.ndarray
    final_beta_full_input_scale: np.ndarray
    srdo_diagnostics: Dict[str, float]
    selector_history: List[Dict[str, float]] = field(default_factory=list)
    final_logistic_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def ess(self) -> float:
        return _effective_sample_size(self.srdo_weights)

class SVILogistic:

    def __init__(self, *, srdo_hidden_layer_sizes: Sequence[int]=(100, 5), srdo_activation: ActivationName='relu', srdo_dropout: float=0.0, srdo_solver: Literal['adam', 'lbfgs', 'sgd']='adam', srdo_alpha: float=0.0001, srdo_learning_rate_init: float=0.001, srdo_max_iter: int=500, srdo_tol: float=0.0001, srdo_batch_size: int=200, srdo_early_stopping: bool=False, srdo_validation_fraction: float=0.1, srdo_n_iter_no_change: int=20, srdo_gradient_clip: Optional[float]=5.0, srdo_cross_fit_folds: int=1, srdo_probability_clip: float=1e-06, srdo_weight_clip_quantile: Optional[float]=None, srdo_max_weight_share: Optional[float]=None, precomputed_srdo_weights: Optional[ArrayLike]=None, precomputed_srdo_diagnostics: Optional[Dict[str, Any]]=None, selector_hidden_layer_sizes: Sequence[int]=(64, 32), selector_activation: ActivationName='relu', selector_dropout: float=0.0, gate_sigma: float=0.5, gate_mu_init: float=0.0, lambda_stg: float=0.01, regularizer_normalization: Literal['sum', 'mean']='sum', selection_threshold: float=0.5, n_features_to_select: Optional[int]=None, min_features: int=1, selector_pretrain_epochs: int=100, selector_max_epochs: int=1000, selector_min_epochs: int=100, selector_batch_size: Optional[int]=256, selector_predictor_lr: float=0.001, selector_gate_lr: float=0.005, selector_weight_decay: float=0.0001, selector_tol: float=1e-05, selector_patience: int=30, selector_gradient_clip: Optional[float]=5.0, selector_log_every: int=100, final_use_srdo_weights: bool=False, final_l2: float=1e-08, final_max_iter: int=2000, final_tol: float=1e-09, random_state: int=2026, device: Optional[Union[str, torch.device]]=None, verbose: bool=True):
        if not 0.0 < selection_threshold < 1.0:
            raise ValueError('selection_threshold must lie in (0,1).')
        if min_features < 1:
            raise ValueError('min_features must be >= 1.')
        if n_features_to_select is not None and n_features_to_select < 1:
            raise ValueError('n_features_to_select must be a positive integer or None.')
        self.srdo_hidden_layer_sizes = tuple((int(v) for v in srdo_hidden_layer_sizes))
        self.srdo_activation = srdo_activation
        self.srdo_dropout = float(srdo_dropout)
        self.srdo_solver = srdo_solver
        self.srdo_alpha = float(srdo_alpha)
        self.srdo_learning_rate_init = float(srdo_learning_rate_init)
        self.srdo_max_iter = int(srdo_max_iter)
        self.srdo_tol = float(srdo_tol)
        self.srdo_batch_size = int(srdo_batch_size)
        self.srdo_early_stopping = bool(srdo_early_stopping)
        self.srdo_validation_fraction = float(srdo_validation_fraction)
        self.srdo_n_iter_no_change = int(srdo_n_iter_no_change)
        self.srdo_gradient_clip = srdo_gradient_clip
        self.srdo_cross_fit_folds = int(srdo_cross_fit_folds)
        self.srdo_probability_clip = float(srdo_probability_clip)
        self.srdo_weight_clip_quantile = srdo_weight_clip_quantile
        self.srdo_max_weight_share = srdo_max_weight_share
        self.precomputed_srdo_weights = None if precomputed_srdo_weights is None else np.asarray(precomputed_srdo_weights, dtype=float).ravel().copy()
        self.precomputed_srdo_diagnostics = {} if precomputed_srdo_diagnostics is None else dict(precomputed_srdo_diagnostics)
        self.selector_hidden_layer_sizes = tuple((int(v) for v in selector_hidden_layer_sizes))
        self.selector_activation = selector_activation
        self.selector_dropout = float(selector_dropout)
        self.gate_sigma = float(gate_sigma)
        self.gate_mu_init = float(gate_mu_init)
        self.lambda_stg = float(lambda_stg)
        self.regularizer_normalization = regularizer_normalization
        self.selection_threshold = float(selection_threshold)
        self.n_features_to_select = None if n_features_to_select is None else int(n_features_to_select)
        self.min_features = int(min_features)
        self.selector_pretrain_epochs = int(selector_pretrain_epochs)
        self.selector_max_epochs = int(selector_max_epochs)
        self.selector_min_epochs = int(selector_min_epochs)
        self.selector_batch_size = selector_batch_size
        self.selector_predictor_lr = float(selector_predictor_lr)
        self.selector_gate_lr = float(selector_gate_lr)
        self.selector_weight_decay = float(selector_weight_decay)
        self.selector_tol = float(selector_tol)
        self.selector_patience = int(selector_patience)
        self.selector_gradient_clip = selector_gradient_clip
        self.selector_log_every = int(selector_log_every)
        self.final_use_srdo_weights = bool(final_use_srdo_weights)
        self.final_l2 = float(final_l2)
        self.final_max_iter = int(final_max_iter)
        self.final_tol = float(final_tol)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            if self.device.type == 'cuda' and (not torch.cuda.is_available()):
                warnings.warn('CUDA is unavailable; falling back to CPU.', RuntimeWarning)
                self.device = torch.device('cpu')
        self.result_: Optional[SVILogisticResult] = None
        self.feature_names_: Optional[List[str]] = None
        self.X_mean_: Optional[np.ndarray] = None
        self.X_scale_: Optional[np.ndarray] = None
        self.zero_variance_mask_: Optional[np.ndarray] = None
        self.srdo_classifier_: Optional[Union[_TorchSRDOClassifier, List[_TorchSRDOClassifier]]] = None
        self.selector_network_: Optional[_BinarySTGNetwork] = None

    def _prepare_X_fit(self, X: Any) -> np.ndarray:
        if hasattr(X, 'columns') and hasattr(X, 'to_numpy'):
            self.feature_names_ = [str(c) for c in list(X.columns)]
        else:
            self.feature_names_ = None
        X_arr = _to_numpy_2d(X)
        p = X_arr.shape[1]
        mean = np.zeros(p, dtype=float)
        scale = np.ones(p, dtype=float)
        zero_mask = np.zeros(p, dtype=bool)
        Xs = X_arr.copy()
        self.X_mean_ = mean
        self.X_scale_ = scale
        self.zero_variance_mask_ = zero_mask
        return Xs

    def _prepare_X_predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X_arr = _to_numpy_2d(X)
        if X_arr.shape[1] != len(self.result_.gate_probabilities):
            raise ValueError('The number of variables in the new X does not match the training data.')
        return X_arr

    def fit(self, X: Any, y: Any, feature_names: Optional[Sequence[str]]=None) -> 'SVILogistic':
        _set_random_seed(self.random_state)
        Xs = self._prepare_X_fit(X)
        (n, p) = Xs.shape
        ys = _to_binary_y(y, n)
        if feature_names is not None:
            if len(feature_names) != p:
                raise ValueError('Length of feature_names must equal the number of variables p.')
            self.feature_names_ = [str(v) for v in feature_names]
        if self.n_features_to_select is not None and self.n_features_to_select > p:
            raise ValueError('n_features_to_select cannot exceed p.')
        if self.min_features > p:
            raise ValueError('min_features cannot exceed p.')
        if self.verbose:
            print(f'[*] SVI Logistic Start: n={n}, p={p}, device={self.device}, lambda_stg={self.lambda_stg}, threshold={self.selection_threshold}')
        if self.verbose:
            print('[1/3] Estimating SRDO density-ratio weights ...')
        if self.precomputed_srdo_weights is None:
            (weights, srdo_info, srdo_classifier) = estimate_srdo_weights(Xs, hidden_layer_sizes=self.srdo_hidden_layer_sizes, activation=self.srdo_activation, dropout=self.srdo_dropout, solver=self.srdo_solver, alpha=self.srdo_alpha, learning_rate_init=self.srdo_learning_rate_init, max_iter=self.srdo_max_iter, tol=self.srdo_tol, batch_size=self.srdo_batch_size, early_stopping=self.srdo_early_stopping, validation_fraction=self.srdo_validation_fraction, n_iter_no_change=self.srdo_n_iter_no_change, gradient_clip=self.srdo_gradient_clip, cross_fit_folds=self.srdo_cross_fit_folds, probability_clip=self.srdo_probability_clip, weight_clip_quantile=self.srdo_weight_clip_quantile, max_weight_share=self.srdo_max_weight_share, random_state=self.random_state, device=self.device, verbose=False)
        else:
            weights = self.precomputed_srdo_weights.copy()
            if weights.shape != (n,):
                raise ValueError('Length of precomputed_srdo_weights must equal the training sample size.')
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0) or float(np.mean(weights)) <= 0.0:
                raise ValueError('precomputed_srdo_weights must be finite and strictly positive with a positive mean.')
            weight_mean = float(np.mean(weights))
            if not np.isclose(weight_mean, 1.0, rtol=0.0, atol=1e-12):
                weights = weights / weight_mean
            achieved_share = float(np.max(weights) / np.sum(weights))
            if self.srdo_max_weight_share is not None and achieved_share > float(self.srdo_max_weight_share) + 1e-12:
                raise ValueError('precomputed_srdo_weights violate srdo_max_weight_share.')
            srdo_info = {**self.precomputed_srdo_diagnostics, 'n_samples': float(n), 'weight_min': float(np.min(weights)), 'weight_mean': float(np.mean(weights)), 'weight_max': float(np.max(weights)), 'weight_sum': float(np.sum(weights)), 'ess': float(_effective_sample_size(weights)), 'ess_ratio': float(_effective_sample_size(weights) / n), 'max_weight_share': self.srdo_max_weight_share, 'achieved_max_weight_share': achieved_share, 'weights_precomputed': 1.0}
            srdo_classifier = None
        self.srdo_classifier_ = srdo_classifier
        if self.verbose:
            print(f"      W min/mean/max = {weights.min():.4f}/{weights.mean():.4f}/{weights.max():.4f} | ESS={srdo_info['ess']:.1f}/{n}")
        if self.verbose:
            print('[2/3] Training weighted stochastic-gate selector ...')
        (gate_mu, gate_probs, history, selector_network) = _train_binary_stg_selector(Xs, ys, weights, hidden_layer_sizes=self.selector_hidden_layer_sizes, activation=self.selector_activation, dropout=self.selector_dropout, gate_sigma=self.gate_sigma, gate_mu_init=self.gate_mu_init, lambda_stg=self.lambda_stg, regularizer_normalization=self.regularizer_normalization, pretrain_epochs=self.selector_pretrain_epochs, max_epochs=self.selector_max_epochs, min_epochs=self.selector_min_epochs, batch_size=self.selector_batch_size, predictor_lr=self.selector_predictor_lr, gate_lr=self.selector_gate_lr, weight_decay=self.selector_weight_decay, tol=self.selector_tol, patience=self.selector_patience, gradient_clip=self.selector_gradient_clip, random_state=self.random_state, device=self.device, verbose=self.verbose, log_every=self.selector_log_every)
        self.selector_network_ = selector_network
        if self.n_features_to_select is not None:
            order = np.argsort(gate_probs)[::-1]
            selected = np.sort(order[:self.n_features_to_select])
        else:
            selected = np.flatnonzero(gate_probs >= self.selection_threshold)
        if selected.size < self.min_features:
            order = np.argsort(gate_probs)[::-1]
            selected = np.sort(order[:self.min_features])
            warnings.warn(f'Fewer than min_features variables passed selection_threshold; topped up to {self.min_features} by gate probability.', RuntimeWarning)
        selected_names = None
        if self.feature_names_ is not None:
            selected_names = [self.feature_names_[j] for j in selected]
        if self.verbose:
            top_idx = np.argsort(gate_probs)[::-1][:min(10, p)]
            top_pairs = [(int(j), round(float(gate_probs[j]), 4)) for j in top_idx]
            print(f'      selected={len(selected)}/{p} | Top gate probabilities={top_pairs}')
        if self.verbose:
            print('[3/3] Refitting final Logistic model on selected variables ...')
        final_weight = weights if self.final_use_srdo_weights else None
        final_info = _fit_final_logistic(Xs[:, selected], ys, sample_weight=final_weight, l2=self.final_l2, max_iter=self.final_max_iter, tol=self.final_tol)
        beta_selected = np.asarray(final_info['beta'], dtype=float)
        beta_full = np.zeros(p, dtype=float)
        beta_full[selected] = beta_selected
        self.result_ = SVILogisticResult(selected_features=selected.astype(int), selected_feature_names=selected_names, srdo_weights=weights.copy(), gate_mu=gate_mu.copy(), gate_probabilities=gate_probs.copy(), final_intercept_input_scale=float(final_info['intercept']), final_beta_selected_input_scale=beta_selected.copy(), final_beta_full_input_scale=beta_full, srdo_diagnostics=dict(srdo_info), selector_history=list(history), final_logistic_info={k: v for (k, v) in final_info.items() if k != 'beta'})
        if self.verbose:
            print(f"[✓] SVI Logistic finished | selected={len(selected)} | final_success={final_info['success']}")
        return self

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise ValueError('The model is not fitted; call fit(X,y) first.')

    def get_selected_features(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.selected_features.copy()

    def get_selected_feature_names(self) -> Optional[List[str]]:
        self._check_fitted()
        if self.result_.selected_feature_names is None:
            return None
        return list(self.result_.selected_feature_names)

    def get_selection_probabilities(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.gate_probabilities.copy()

    def get_gate_mu(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.gate_mu.copy()

    def get_weights(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.srdo_weights.copy()

    def get_effective_sample_size(self) -> float:
        self._check_fitted()
        return float(self.result_.ess)

    def get_srdo_diagnostics(self) -> Dict[str, float]:
        self._check_fitted()
        return dict(self.result_.srdo_diagnostics)

    def get_final_coefficients(self, original_scale: bool=False) -> np.ndarray:
        self._check_fitted()
        beta = self.result_.final_beta_full_input_scale.copy()
        if original_scale:
            beta = beta
        return beta

    def get_final_intercept(self, original_scale: bool=False) -> float:
        self._check_fitted()
        intercept = float(self.result_.final_intercept_input_scale)
        if not original_scale:
            return intercept
        beta_original = self.get_final_coefficients(original_scale=True)
        return float(intercept - np.dot(self.X_mean_, beta_original))

    def decision_function(self, X: Any) -> np.ndarray:
        self._check_fitted()
        Xs = self._prepare_X_predict(X)
        return self.result_.final_intercept_input_scale + Xs @ self.result_.final_beta_full_input_scale

    def predict_proba(self, X: Any) -> np.ndarray:
        p1 = expit(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any, threshold: float=0.5) -> np.ndarray:
        if not 0.0 < threshold < 1.0:
            raise ValueError('threshold must lie in (0,1).')
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def selection_summary(self, sort_by_probability: bool=True):
        self._check_fitted()
        probs = self.result_.gate_probabilities
        mu = self.result_.gate_mu
        beta_std = self.result_.final_beta_full_input_scale
        beta_orig = self.get_final_coefficients(original_scale=True)
        selected_set = set(self.result_.selected_features.tolist())
        rows: List[Dict[str, Any]] = []
        for j in range(len(probs)):
            rows.append({'feature_index': int(j), 'feature_name': None if self.feature_names_ is None else self.feature_names_[j], 'gate_mu': float(mu[j]), 'selection_probability': float(probs[j]), 'selected': bool(j in selected_set), 'final_logistic_beta_input_scale': float(beta_std[j]), 'final_logistic_beta_original_scale': float(beta_orig[j])})
        if sort_by_probability:
            rows.sort(key=lambda item: item['selection_probability'], reverse=True)
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def history_table(self):
        self._check_fitted()
        try:
            import pandas as pd
            return pd.DataFrame(self.result_.selector_history)
        except ImportError:
            return list(self.result_.selector_history)

def SVI_Logistic(X: Any, y: Any, *, feature_names: Optional[Sequence[str]]=None, srdo_hidden_layer_sizes: Sequence[int]=(100, 5), srdo_activation: ActivationName='relu', srdo_dropout: float=0.0, srdo_solver: Literal['adam', 'lbfgs', 'sgd']='adam', srdo_alpha: float=0.0001, srdo_learning_rate_init: float=0.001, srdo_max_iter: int=500, srdo_tol: float=0.0001, srdo_batch_size: int=200, srdo_early_stopping: bool=False, srdo_validation_fraction: float=0.1, srdo_n_iter_no_change: int=20, srdo_gradient_clip: Optional[float]=5.0, srdo_cross_fit_folds: int=1, srdo_probability_clip: float=1e-06, srdo_weight_clip_quantile: Optional[float]=None, srdo_max_weight_share: Optional[float]=None, precomputed_srdo_weights: Optional[ArrayLike]=None, precomputed_srdo_diagnostics: Optional[Dict[str, Any]]=None, selector_hidden_layer_sizes: Sequence[int]=(64, 32), selector_activation: ActivationName='relu', selector_dropout: float=0.0, gate_sigma: float=0.5, gate_mu_init: float=0.0, lambda_stg: float=0.01, regularizer_normalization: Literal['sum', 'mean']='sum', selection_threshold: float=0.5, n_features_to_select: Optional[int]=None, min_features: int=1, selector_pretrain_epochs: int=100, selector_max_epochs: int=1000, selector_min_epochs: int=100, selector_batch_size: Optional[int]=256, selector_predictor_lr: float=0.001, selector_gate_lr: float=0.005, selector_weight_decay: float=0.0001, selector_tol: float=1e-05, selector_patience: int=30, selector_gradient_clip: Optional[float]=5.0, selector_log_every: int=100, final_use_srdo_weights: bool=False, final_l2: float=1e-08, final_max_iter: int=2000, final_tol: float=1e-09, random_state: int=2026, device: Optional[Union[str, torch.device]]=None, verbose: bool=True) -> SVILogistic:
    model = SVILogistic(srdo_hidden_layer_sizes=srdo_hidden_layer_sizes, srdo_activation=srdo_activation, srdo_dropout=srdo_dropout, srdo_solver=srdo_solver, srdo_alpha=srdo_alpha, srdo_learning_rate_init=srdo_learning_rate_init, srdo_max_iter=srdo_max_iter, srdo_tol=srdo_tol, srdo_batch_size=srdo_batch_size, srdo_early_stopping=srdo_early_stopping, srdo_validation_fraction=srdo_validation_fraction, srdo_n_iter_no_change=srdo_n_iter_no_change, srdo_gradient_clip=srdo_gradient_clip, srdo_cross_fit_folds=srdo_cross_fit_folds, srdo_probability_clip=srdo_probability_clip, srdo_weight_clip_quantile=srdo_weight_clip_quantile, srdo_max_weight_share=srdo_max_weight_share, precomputed_srdo_weights=precomputed_srdo_weights, precomputed_srdo_diagnostics=precomputed_srdo_diagnostics, selector_hidden_layer_sizes=selector_hidden_layer_sizes, selector_activation=selector_activation, selector_dropout=selector_dropout, gate_sigma=gate_sigma, gate_mu_init=gate_mu_init, lambda_stg=lambda_stg, regularizer_normalization=regularizer_normalization, selection_threshold=selection_threshold, n_features_to_select=n_features_to_select, min_features=min_features, selector_pretrain_epochs=selector_pretrain_epochs, selector_max_epochs=selector_max_epochs, selector_min_epochs=selector_min_epochs, selector_batch_size=selector_batch_size, selector_predictor_lr=selector_predictor_lr, selector_gate_lr=selector_gate_lr, selector_weight_decay=selector_weight_decay, selector_tol=selector_tol, selector_patience=selector_patience, selector_gradient_clip=selector_gradient_clip, selector_log_every=selector_log_every, final_use_srdo_weights=final_use_srdo_weights, final_l2=final_l2, final_max_iter=final_max_iter, final_tol=final_tol, random_state=random_state, device=device, verbose=verbose)
    return model.fit(X, y, feature_names=feature_names)
__all__ = ['SVILogisticResult', 'SVILogistic', 'column_wise_resampling', 'estimate_srdo_weights', 'SVI_Logistic']
