from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import math
import random
import warnings
import numpy as np
import torch
from torch import nn
from .srdo_density import SRDO, _normalize_weights_with_max_share
ArrayLike = Any
ActivationName = Literal['relu', 'tanh', 'elu', 'gelu', 'leaky_relu']
LogisticOptimizer = Literal['lbfgs', 'adam']

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
    result = torch.device(device)
    if result.type == 'cuda' and (not torch.cuda.is_available()):
        warnings.warn('CUDA is unavailable; SRDO_Logistic falls back to CPU.', RuntimeWarning)
        result = torch.device('cpu')
    return result

def _to_numpy_2d(X: ArrayLike, name: str='X') -> np.ndarray:
    if hasattr(X, 'to_numpy'):
        arr = np.asarray(X.to_numpy(), dtype=float)
    else:
        arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f'{name} must be a two-dimensional matrix; got shape={arr.shape}.')
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError(f'{name} requires at least 2 samples and 1 variable.')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} contains NaN or Inf.')
    return arr

def _to_binary_y(y: ArrayLike, n: int) -> np.ndarray:
    if hasattr(y, 'to_numpy'):
        arr = np.asarray(y.to_numpy()).ravel()
    else:
        arr = np.asarray(y).ravel()
    if arr.shape != (n,):
        raise ValueError(f'Length of y must equal n={n}; got shape={arr.shape}.')
    if not np.all(np.isfinite(arr.astype(float))):
        raise ValueError('y contains NaN or Inf.')
    unique = np.unique(arr)
    if np.all(np.isin(unique, [0, 1])):
        result = arr.astype(int)
    elif np.all(np.isin(unique, [-1, 1])):
        result = (arr == 1).astype(int)
    else:
        raise ValueError(f'y must be coded as {{0,1}} or {{-1,+1}}; observed labels={unique.tolist()}.')
    if np.unique(result).size != 2:
        raise ValueError('y must contain both classes.')
    return result

def _effective_sample_size(weights: np.ndarray, eps: float=1e-15) -> float:
    w = np.asarray(weights, dtype=float).ravel()
    return float(w.sum() ** 2 / (np.sum(w ** 2) + eps))

def _sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result

class _TorchLogisticLayer(nn.Module):

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError('input_dim must be positive.')
        self.linear = nn.Linear(input_dim, 1, bias=True, dtype=torch.float64)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X).squeeze(-1)

@dataclass
class TorchLogisticFit:
    intercept: float
    beta: np.ndarray
    objective: float
    optimizer: str
    iterations: int
    converged: bool
    history: List[Dict[str, float]] = field(default_factory=list)

def _weighted_logistic_objective(model: _TorchLogisticLayer, X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    logits = model(X)
    individual = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
    weight_sum = torch.sum(weights)
    if not bool(torch.isfinite(weight_sum)) or float(weight_sum.detach().cpu()) <= 0.0:
        raise FloatingPointError('Sample weights of the weighted logistic model must have a finite positive sum.')
    return torch.sum(weights * individual) / weight_sum

def _fit_torch_logistic(X: np.ndarray, y: np.ndarray, *, sample_weight: Optional[np.ndarray], optimizer_name: LogisticOptimizer, learning_rate: float, max_iter: int, tolerance: float, patience: int, random_state: int, device: torch.device, verbose: bool, label: str) -> TorchLogisticFit:
    X_arr = _to_numpy_2d(X)
    y_arr = _to_binary_y(y, X_arr.shape[0]).astype(float)
    (n, p) = X_arr.shape
    if sample_weight is None:
        w_arr = np.ones(n, dtype=float)
    else:
        w_arr = np.asarray(sample_weight, dtype=float).ravel()
        if w_arr.shape != (n,):
            raise ValueError('Length of sample_weight must equal the sample size.')
        if np.any(w_arr < 0) or not np.all(np.isfinite(w_arr)) or w_arr.sum() <= 0:
            raise ValueError('sample_weight must be finite and nonnegative with a positive sum.')
    if learning_rate <= 0 or max_iter <= 0 or tolerance <= 0 or (patience <= 0):
        raise ValueError('learning_rate, max_iter, tolerance and patience must be positive.')
    _set_random_seed(random_state)
    X_t = torch.as_tensor(X_arr, dtype=torch.float64, device=device)
    y_t = torch.as_tensor(y_arr, dtype=torch.float64, device=device)
    w_t = torch.as_tensor(w_arr, dtype=torch.float64, device=device)
    model = _TorchLogisticLayer(p).to(device=device, dtype=torch.float64)
    positive_rate = float(np.clip(np.average(y_arr, weights=w_arr), 1e-06, 1.0 - 1e-06))
    with torch.no_grad():
        model.linear.bias.fill_(math.log(positive_rate / (1.0 - positive_rate)))
    history: List[Dict[str, float]] = []
    converged = False
    used_iterations = 0
    if optimizer_name == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=float(learning_rate), max_iter=int(max_iter), tolerance_grad=float(tolerance), tolerance_change=float(tolerance), history_size=100, line_search_fn='strong_wolfe')
        closure_calls = 0

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            closure_calls += 1
            optimizer.zero_grad(set_to_none=True)
            objective = _weighted_logistic_objective(model, X_t, y_t, w_t)
            objective.backward()
            return objective
        optimizer.step(closure)
        used_iterations = int(closure_calls)
        converged = True
        with torch.no_grad():
            final_objective = float(_weighted_logistic_objective(model, X_t, y_t, w_t).cpu())
        history.append({'iteration': float(used_iterations), 'objective': final_objective})
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        best_state = {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}
        best_objective = math.inf
        stable_count = 0
        previous_parameters: Optional[np.ndarray] = None
        for iteration in range(1, int(max_iter) + 1):
            used_iterations = iteration
            model.train()
            optimizer.zero_grad(set_to_none=True)
            objective = _weighted_logistic_objective(model, X_t, y_t, w_t)
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                current_objective = float(_weighted_logistic_objective(model, X_t, y_t, w_t).cpu())
                params = np.concatenate([model.linear.bias.detach().cpu().numpy().ravel(), model.linear.weight.detach().cpu().numpy().ravel()])
            if previous_parameters is None:
                parameter_change = math.inf
            else:
                parameter_change = float(np.max(np.abs(params - previous_parameters)))
            history.append({'iteration': float(iteration), 'objective': current_objective, 'parameter_change': float(parameter_change)})
            if best_objective - current_objective > tolerance:
                best_objective = current_objective
                best_state = {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}
                stable_count = 0
            elif parameter_change < tolerance:
                stable_count += 1
            else:
                stable_count = 0
            if verbose and (iteration == 1 or iteration % 200 == 0):
                print(f'{label} Adam iter={iteration:5d} | obj={current_objective:.8f} | dPar={parameter_change:.3e}')
            if stable_count >= int(patience):
                converged = True
                break
            previous_parameters = params.copy()
        model.load_state_dict(best_state)
        model.to(device)
        with torch.no_grad():
            final_objective = float(_weighted_logistic_objective(model, X_t, y_t, w_t).cpu())
    else:
        raise ValueError("optimizer_name must be 'lbfgs' or 'adam'.")
    with torch.no_grad():
        intercept = float(model.linear.bias.detach().cpu().numpy().ravel()[0])
        beta = model.linear.weight.detach().cpu().numpy().ravel().astype(float)
    if not np.isfinite(intercept) or not np.all(np.isfinite(beta)):
        raise FloatingPointError(f'{label} produced NaN/Inf parameters.')
    return TorchLogisticFit(intercept=intercept, beta=beta, objective=float(final_objective), optimizer=str(optimizer_name), iterations=int(used_iterations), converged=bool(converged), history=history)

@dataclass
class SRDOLogisticResult:
    initial_fit: TorchLogisticFit
    final_fit: TorchLogisticFit
    initial_probabilities: np.ndarray
    curvature: np.ndarray
    curvature_used: np.ndarray
    srdo_density_ratio: np.ndarray
    srdo_weights: np.ndarray
    corrected_weights: np.ndarray
    ignored_mask: np.ndarray
    diagnostics: Dict[str, float]

class SRDOLogistic:

    def __init__(self, *, srdo_hidden_layer_sizes: Sequence[int]=(100, 5), srdo_activation: ActivationName='relu', srdo_dropout: float=0.0, srdo_optimizer: Literal['adam', 'sgd', 'lbfgs']='adam', srdo_alpha: float=0.0001, srdo_learning_rate: float=0.001, srdo_max_epochs: int=500, srdo_tolerance: float=0.0001, srdo_batch_size: int=200, srdo_early_stopping: bool=False, srdo_validation_fraction: float=0.1, srdo_patience: int=20, srdo_gradient_clip: Optional[float]=5.0, srdo_probability_clip: float=1e-06, srdo_weight_clip_quantile: Optional[float]=None, srdo_sensitive_variables: Optional[Sequence[int]]=None, precomputed_srdo_weights: Optional[ArrayLike]=None, precomputed_srdo_density_ratio: Optional[ArrayLike]=None, precomputed_srdo_diagnostics: Optional[Dict[str, Any]]=None, logistic_optimizer: LogisticOptimizer='lbfgs', logistic_learning_rate: float=1.0, logistic_max_iter: int=500, logistic_tolerance: float=1e-08, logistic_patience: int=30, curvature_floor: float=1e-06, ignore_high_confidence_correct: bool=False, confidence_threshold: float=0.99, corrected_weight_clip_quantile: Optional[float]=None, max_weight_share: Optional[float]=None, random_state: int=2026, device: Optional[Union[str, torch.device]]=None, verbose: bool=True) -> None:
        self.srdo_hidden_layer_sizes = tuple((int(v) for v in srdo_hidden_layer_sizes))
        self.srdo_activation = srdo_activation
        self.srdo_dropout = float(srdo_dropout)
        self.srdo_optimizer = srdo_optimizer
        self.srdo_alpha = float(srdo_alpha)
        self.srdo_learning_rate = float(srdo_learning_rate)
        self.srdo_max_epochs = int(srdo_max_epochs)
        self.srdo_tolerance = float(srdo_tolerance)
        self.srdo_batch_size = int(srdo_batch_size)
        self.srdo_early_stopping = bool(srdo_early_stopping)
        self.srdo_validation_fraction = float(srdo_validation_fraction)
        self.srdo_patience = int(srdo_patience)
        self.srdo_gradient_clip = srdo_gradient_clip
        self.srdo_probability_clip = float(srdo_probability_clip)
        self.srdo_weight_clip_quantile = srdo_weight_clip_quantile
        self.srdo_sensitive_variables = srdo_sensitive_variables
        self.precomputed_srdo_weights = None if precomputed_srdo_weights is None else np.asarray(precomputed_srdo_weights, dtype=float).ravel().copy()
        self.precomputed_srdo_density_ratio = None if precomputed_srdo_density_ratio is None else np.asarray(precomputed_srdo_density_ratio, dtype=float).ravel().copy()
        self.precomputed_srdo_diagnostics = {} if precomputed_srdo_diagnostics is None else dict(precomputed_srdo_diagnostics)
        self.logistic_optimizer = logistic_optimizer
        self.logistic_learning_rate = float(logistic_learning_rate)
        self.logistic_max_iter = int(logistic_max_iter)
        self.logistic_tolerance = float(logistic_tolerance)
        self.logistic_patience = int(logistic_patience)
        self.curvature_floor = float(curvature_floor)
        self.ignore_high_confidence_correct = bool(ignore_high_confidence_correct)
        self.confidence_threshold = float(confidence_threshold)
        self.corrected_weight_clip_quantile = corrected_weight_clip_quantile
        self.max_weight_share = max_weight_share
        self.random_state = int(random_state)
        self.device = _resolve_device(device)
        self.verbose = bool(verbose)
        if self.curvature_floor <= 0 or self.curvature_floor > 0.25:
            raise ValueError('curvature_floor must lie in (0,0.25].')
        if not 0.5 < self.confidence_threshold < 1.0:
            raise ValueError('confidence_threshold must lie in (0.5,1).')
        if self.corrected_weight_clip_quantile is not None and (not 0.5 < float(self.corrected_weight_clip_quantile) < 1.0):
            raise ValueError('corrected_weight_clip_quantile must lie in (0.5,1) or be None.')
        if self.max_weight_share is not None and (not 0.0 < float(self.max_weight_share) <= 1.0):
            raise ValueError('max_weight_share must lie in (0,1] or be None.')
        self.result_: Optional[SRDOLogisticResult] = None
        self.srdo_: Optional[SRDO] = None
        self.feature_names_: Optional[List[str]] = None
        self.X_mean_: Optional[np.ndarray] = None
        self.X_scale_: Optional[np.ndarray] = None
        self.zero_variance_mask_: Optional[np.ndarray] = None

    def _prepare_fit_X(self, X: ArrayLike) -> np.ndarray:
        if hasattr(X, 'columns') and hasattr(X, 'to_numpy'):
            self.feature_names_ = [str(v) for v in list(X.columns)]
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

    def _prepare_predict_X(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        X_arr = _to_numpy_2d(X)
        if X_arr.shape[1] != len(self.result_.final_fit.beta):
            raise ValueError('The number of variables in the new X does not match the training data.')
        return X_arr

    def fit(self, X: ArrayLike, y: ArrayLike, feature_names: Optional[Sequence[str]]=None) -> 'SRDOLogistic':
        _set_random_seed(self.random_state)
        Xs = self._prepare_fit_X(X)
        (n, p) = Xs.shape
        ys = _to_binary_y(y, n)
        if feature_names is not None:
            if len(feature_names) != p:
                raise ValueError('Length of feature_names must equal the number of variables p.')
            self.feature_names_ = [str(v) for v in feature_names]
        if self.verbose:
            print(f'[*] SRDO_Logistic Start: n={n}, p={p}, device={self.device}, optimizer={self.logistic_optimizer}')
        if self.verbose:
            print('[1/4] Fitting approximate Logistic model ...')
        initial_fit = _fit_torch_logistic(Xs, ys, sample_weight=None, optimizer_name=self.logistic_optimizer, learning_rate=self.logistic_learning_rate, max_iter=self.logistic_max_iter, tolerance=self.logistic_tolerance, patience=self.logistic_patience, random_state=self.random_state, device=self.device, verbose=False, label='Initial Logistic')
        initial_eta = initial_fit.intercept + Xs @ initial_fit.beta
        initial_prob = _sigmoid_numpy(initial_eta)
        initial_prob = np.clip(initial_prob, self.srdo_probability_clip, 1.0 - self.srdo_probability_clip)
        if self.verbose:
            print('[2/4] Estimating SRDO density ratio with PyTorch MLP ...')
        if self.precomputed_srdo_weights is None:
            self.srdo_ = SRDO(hidden_layer_sizes=self.srdo_hidden_layer_sizes, activation=self.srdo_activation, dropout=self.srdo_dropout, solver=self.srdo_optimizer, learning_rate=self.srdo_learning_rate, l2=self.srdo_alpha, batch_size=self.srdo_batch_size, max_epochs=self.srdo_max_epochs, tol=self.srdo_tolerance, patience=self.srdo_patience, early_stopping=self.srdo_early_stopping, validation_fraction=self.srdo_validation_fraction, gradient_clip=self.srdo_gradient_clip, probability_clip=self.srdo_probability_clip, weight_clip_quantile=self.srdo_weight_clip_quantile, max_weight_share=self.max_weight_share, sensitive_variables=self.srdo_sensitive_variables, random_state=self.random_state, device=self.device, verbose=False).fit(Xs)
            srdo_ratio = self.srdo_.get_raw_density_ratio()
            srdo_weights = self.srdo_.get_weights()
            srdo_diag = self.srdo_.get_diagnostics()
            weights_precomputed = False
        else:
            srdo_weights = self.precomputed_srdo_weights.copy()
            if srdo_weights.shape != (n,):
                raise ValueError('Length of precomputed_srdo_weights must equal the training sample size.')
            if not np.all(np.isfinite(srdo_weights)) or np.any(srdo_weights <= 0.0) or float(np.mean(srdo_weights)) <= 0.0:
                raise ValueError('precomputed_srdo_weights must be finite and strictly positive with a positive mean.')
            srdo_weight_mean = float(np.mean(srdo_weights))
            if not np.isclose(srdo_weight_mean, 1.0, rtol=0.0, atol=1e-12):
                srdo_weights = srdo_weights / srdo_weight_mean
            achieved_share = float(np.max(srdo_weights) / np.sum(srdo_weights))
            if self.max_weight_share is not None and achieved_share > float(self.max_weight_share) + 1e-12:
                raise ValueError('precomputed_srdo_weights violate max_weight_share.')
            if self.precomputed_srdo_density_ratio is None:
                srdo_ratio = srdo_weights.copy()
            else:
                srdo_ratio = self.precomputed_srdo_density_ratio.copy()
                if srdo_ratio.shape != (n,):
                    raise ValueError('Length of precomputed_srdo_density_ratio must equal the training sample size.')
                if not np.all(np.isfinite(srdo_ratio)) or np.any(srdo_ratio <= 0.0):
                    raise ValueError('precomputed_srdo_density_ratio must be finite and strictly positive.')
            srdo_diag = {**self.precomputed_srdo_diagnostics, 'n_samples': float(n), 'ess': float(_effective_sample_size(srdo_weights)), 'ess_ratio': float(_effective_sample_size(srdo_weights) / n), 'achieved_max_weight_share': achieved_share, 'density_classifier_accuracy': float(self.precomputed_srdo_diagnostics.get('density_classifier_accuracy', float('nan'))), 'weights_precomputed': 1.0}
            weights_precomputed = True
        if self.verbose:
            print('[3/4] Applying Logistic curvature correction ...')
        curvature = initial_prob * (1.0 - initial_prob)
        curvature_used = np.maximum(curvature, self.curvature_floor)
        corrected_raw = srdo_ratio / curvature_used
        ignored_mask = np.zeros(n, dtype=bool)
        if self.ignore_high_confidence_correct:
            predicted = (initial_prob >= 0.5).astype(int)
            confidence = np.maximum(initial_prob, 1.0 - initial_prob)
            ignored_mask = (predicted == ys) & (confidence >= self.confidence_threshold)
            corrected_raw = corrected_raw.copy()
            corrected_raw[ignored_mask] = 0.0
            if corrected_raw.sum() <= 0 or np.sum(corrected_raw[ys == 0]) <= 0 or np.sum(corrected_raw[ys == 1]) <= 0:
                warnings.warn('Ignoring high-confidence samples made the weights degenerate; this step was skipped automatically.', RuntimeWarning)
                ignored_mask[:] = False
                corrected_raw = srdo_ratio / curvature_used
        corrected_clip_threshold = float('nan')
        if self.corrected_weight_clip_quantile is not None:
            positive_weights = corrected_raw[corrected_raw > 0]
            if positive_weights.size == 0:
                raise FloatingPointError('No positive weights remain after curvature correction.')
            corrected_clip_threshold = float(np.quantile(positive_weights, float(self.corrected_weight_clip_quantile)))
            corrected_raw = np.minimum(corrected_raw, corrected_clip_threshold)
        if not np.all(np.isfinite(corrected_raw)) or np.any(corrected_raw < 0) or corrected_raw.mean() <= 0:
            raise FloatingPointError('Curvature-corrected weights contain invalid values.')
        (corrected_weights, corrected_share_clip_threshold, _) = _normalize_weights_with_max_share(corrected_raw, self.max_weight_share)
        corrected_ess = _effective_sample_size(corrected_weights)
        if self.verbose:
            print('[4/4] Fitting final unpenalized weighted Logistic model ...')
        final_fit = _fit_torch_logistic(Xs, ys, sample_weight=corrected_weights, optimizer_name=self.logistic_optimizer, learning_rate=self.logistic_learning_rate, max_iter=self.logistic_max_iter, tolerance=self.logistic_tolerance, patience=self.logistic_patience, random_state=self.random_state + 1, device=self.device, verbose=False, label='Weighted Logistic')
        diagnostics = {'n_samples': float(n), 'n_features': float(p), 'logistic_penalty_l1': 0.0, 'logistic_penalty_l2': 0.0, 'initial_objective': float(initial_fit.objective), 'final_weighted_objective': float(final_fit.objective), 'initial_positive_probability_min': float(np.min(initial_prob)), 'initial_positive_probability_mean': float(np.mean(initial_prob)), 'initial_positive_probability_max': float(np.max(initial_prob)), 'curvature_min': float(np.min(curvature)), 'curvature_mean': float(np.mean(curvature)), 'curvature_max': float(np.max(curvature)), 'curvature_floor': float(self.curvature_floor), 'n_curvature_floored': float(np.sum(curvature < self.curvature_floor)), 'n_ignored_high_confidence_correct': float(np.sum(ignored_mask)), 'srdo_weight_min': float(np.min(srdo_weights)), 'srdo_weight_mean': float(np.mean(srdo_weights)), 'srdo_weight_max': float(np.max(srdo_weights)), 'srdo_ess': float(_effective_sample_size(srdo_weights)), 'srdo_ess_ratio': float(_effective_sample_size(srdo_weights) / n), 'corrected_weight_min': float(np.min(corrected_weights)), 'corrected_weight_mean': float(np.mean(corrected_weights)), 'corrected_weight_max': float(np.max(corrected_weights)), 'corrected_weight_sum': float(np.sum(corrected_weights)), 'corrected_ess': float(corrected_ess), 'corrected_ess_ratio': float(corrected_ess / n), 'srdo_probability_clip': self.srdo_probability_clip, 'corrected_weight_clip_threshold': corrected_clip_threshold, 'max_weight_share': self.max_weight_share, 'corrected_max_share_clip_threshold': corrected_share_clip_threshold, 'corrected_achieved_max_weight_share': float(np.max(corrected_weights) / np.sum(corrected_weights)), 'srdo_density_classifier_accuracy': float(srdo_diag['density_classifier_accuracy']), 'srdo_weights_precomputed': float(weights_precomputed), 'device_is_cuda': float(self.device.type == 'cuda')}
        self.result_ = SRDOLogisticResult(initial_fit=initial_fit, final_fit=final_fit, initial_probabilities=initial_prob.copy(), curvature=curvature.copy(), curvature_used=curvature_used.copy(), srdo_density_ratio=srdo_ratio.copy(), srdo_weights=srdo_weights.copy(), corrected_weights=corrected_weights.copy(), ignored_mask=ignored_mask.copy(), diagnostics=diagnostics)
        if self.verbose:
            print(f"[✓] SRDO_Logistic finished | SRDO ESS={diagnostics['srdo_ess']:.1f}/{n} | Corrected ESS={corrected_ess:.1f}/{n} | W min/mean/max={corrected_weights.min():.4f}/{corrected_weights.mean():.4f}/{corrected_weights.max():.4f}")
        return self

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise ValueError('SRDO_Logistic is not fitted; call fit(X,y) first.')

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        Xs = self._prepare_predict_X(X)
        return self.result_.final_fit.intercept + Xs @ self.result_.final_fit.beta

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        p1 = _sigmoid_numpy(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: ArrayLike, threshold: float=0.5) -> np.ndarray:
        if not 0.0 < threshold < 1.0:
            raise ValueError('threshold must lie in (0,1).')
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def get_weights(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.corrected_weights.copy()

    def get_srdo_weights(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.srdo_weights.copy()

    def get_srdo_density_ratio(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.srdo_density_ratio.copy()

    def get_curvature(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.curvature.copy()

    def get_initial_probabilities(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.initial_probabilities.copy()

    def get_effective_sample_size(self) -> float:
        self._check_fitted()
        return float(self.result_.diagnostics['corrected_ess'])

    def get_diagnostics(self) -> Dict[str, float]:
        self._check_fitted()
        return dict(self.result_.diagnostics)

    def get_coefficients(self, original_scale: bool=False) -> np.ndarray:
        self._check_fitted()
        beta = self.result_.final_fit.beta.copy()
        if original_scale:
            beta = beta
        return beta

    def get_intercept(self, original_scale: bool=False) -> float:
        self._check_fitted()
        intercept = float(self.result_.final_fit.intercept)
        if not original_scale:
            return intercept
        beta_original = self.get_coefficients(original_scale=True)
        return float(intercept - np.dot(self.X_mean_, beta_original))

    def get_initial_coefficients(self, original_scale: bool=False) -> np.ndarray:
        self._check_fitted()
        beta = self.result_.initial_fit.beta.copy()
        if original_scale:
            beta = beta
        return beta

    def coefficient_summary(self):
        self._check_fitted()
        initial_std = self.result_.initial_fit.beta
        final_std = self.result_.final_fit.beta
        initial_orig = initial_std
        final_orig = final_std
        rows: List[Dict[str, Any]] = []
        for j in range(len(final_std)):
            rows.append({'feature_index': int(j), 'feature_name': None if self.feature_names_ is None else self.feature_names_[j], 'initial_beta_input_scale': float(initial_std[j]), 'final_beta_input_scale': float(final_std[j]), 'initial_beta_original_scale': float(initial_orig[j]), 'final_beta_original_scale': float(final_orig[j])})
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

def fit_srdo_logistic(X: ArrayLike, y: ArrayLike, *, feature_names: Optional[Sequence[str]]=None, **kwargs: Any) -> SRDOLogistic:
    model = SRDOLogistic(**kwargs)
    return model.fit(X, y, feature_names=feature_names)
SRDO_Logistic = SRDOLogistic
__all__ = ['TorchLogisticFit', 'SRDOLogisticResult', 'SRDOLogistic', 'SRDO_Logistic', 'fit_srdo_logistic']
