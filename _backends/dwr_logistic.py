from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import math
import random
import warnings
import numpy as np
import torch
from torch import nn
from .dwr_weights import DWR
ArrayLike = Any
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
        warnings.warn('CUDA is unavailable; DWR_Logistic falls back to CPU.', RuntimeWarning)
        result = torch.device('cpu')
    return result

def _to_numpy_2d(X: ArrayLike, name: str='X') -> np.ndarray:
    if hasattr(X, 'to_numpy'):
        arr = np.asarray(X.to_numpy(), dtype=np.float64)
    else:
        arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f'{name} must be a two-dimensional matrix; got shape={arr.shape}.')
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f'{name} requires at least 2 samples and 2 variables.')
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
    w = np.asarray(weights, dtype=np.float64).ravel()
    return float(w.sum() ** 2 / (np.dot(w, w) + eps))

def _sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    positive = x >= 0.0
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
    gradient_inf: float
    history: List[Dict[str, float]] = field(default_factory=list)

def _weighted_logistic_objective(model: _TorchLogisticLayer, X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    logits = model(X)
    individual = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
    weight_sum = torch.sum(weights)
    if not bool(torch.isfinite(weight_sum)) or float(weight_sum.detach().cpu()) <= 0.0:
        raise FloatingPointError('Sample weights of the weighted logistic model must have a finite positive sum.')
    return torch.sum(weights * individual) / weight_sum

def _objective_and_gradient_inf(model: _TorchLogisticLayer, X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> Tuple[float, float]:
    model.zero_grad(set_to_none=True)
    objective = _weighted_logistic_objective(model, X, y, weights)
    objective.backward()
    gradients: List[np.ndarray] = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradients.append(parameter.grad.detach().cpu().numpy().ravel())
    gradient_inf = float(np.max(np.abs(np.concatenate(gradients)))) if gradients else 0.0
    return (float(objective.detach().cpu()), gradient_inf)

def _fit_torch_logistic(X: np.ndarray, y: np.ndarray, *, sample_weight: Optional[np.ndarray], optimizer_name: LogisticOptimizer, learning_rate: float, max_iter: int, tolerance: float, patience: int, random_state: int, device: torch.device, verbose: bool, label: str) -> TorchLogisticFit:
    X_arr = _to_numpy_2d(X)
    y_arr = _to_binary_y(y, X_arr.shape[0]).astype(np.float64)
    (n, p) = X_arr.shape
    if sample_weight is None:
        w_arr = np.ones(n, dtype=np.float64)
    else:
        w_arr = np.asarray(sample_weight, dtype=np.float64).ravel()
        if w_arr.shape != (n,):
            raise ValueError('Length of sample_weight must equal the sample size.')
        if np.any(w_arr < 0.0) or not np.all(np.isfinite(w_arr)) or w_arr.sum() <= 0.0:
            raise ValueError('sample_weight must be finite and nonnegative with a positive sum.')
        if np.sum(w_arr[y_arr == 0.0]) <= 0.0 or np.sum(w_arr[y_arr == 1.0]) <= 0.0:
            raise ValueError('Both classes must have positive total sample weight.')
    if learning_rate <= 0.0 or max_iter <= 0 or tolerance <= 0.0 or (patience <= 0):
        raise ValueError('learning_rate, max_iter, tolerance and patience must be positive.')
    if optimizer_name not in {'lbfgs', 'adam'}:
        raise ValueError("optimizer_name must be 'lbfgs' or 'adam'.")
    _set_random_seed(random_state)
    X_t = torch.as_tensor(X_arr, dtype=torch.float64, device=device)
    y_t = torch.as_tensor(y_arr, dtype=torch.float64, device=device)
    w_t = torch.as_tensor(w_arr, dtype=torch.float64, device=device)
    model = _TorchLogisticLayer(p).to(device=device, dtype=torch.float64)
    positive_rate = float(np.clip(np.average(y_arr, weights=w_arr), 1e-08, 1.0 - 1e-08))
    with torch.no_grad():
        model.linear.bias.fill_(math.log(positive_rate / (1.0 - positive_rate)))
    history: List[Dict[str, float]] = []
    used_iterations = 0
    converged = False
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
        (final_objective, gradient_inf) = _objective_and_gradient_inf(model, X_t, y_t, w_t)
        converged = bool(gradient_inf <= max(1e-06, math.sqrt(float(tolerance))))
        history.append({'iteration': float(used_iterations), 'objective': float(final_objective), 'gradient_inf': float(gradient_inf)})
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        best_state = {key: value.detach().cpu().clone() for (key, value) in model.state_dict().items()}
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
                parameters = np.concatenate([model.linear.bias.detach().cpu().numpy().ravel(), model.linear.weight.detach().cpu().numpy().ravel()])
            parameter_change = math.inf if previous_parameters is None else float(np.max(np.abs(parameters - previous_parameters)))
            history.append({'iteration': float(iteration), 'objective': current_objective, 'parameter_change': float(parameter_change)})
            if best_objective - current_objective > float(tolerance):
                best_objective = current_objective
                best_state = {key: value.detach().cpu().clone() for (key, value) in model.state_dict().items()}
                stable_count = 0
            elif parameter_change <= float(tolerance):
                stable_count += 1
            else:
                stable_count = 0
            if verbose and (iteration == 1 or iteration % 200 == 0):
                print(f'{label} Adam iter={iteration:5d} | obj={current_objective:.10f} | dPar={parameter_change:.3e}')
            if stable_count >= int(patience):
                converged = True
                break
            previous_parameters = parameters.copy()
        model.load_state_dict(best_state)
        model.to(device=device, dtype=torch.float64)
        (final_objective, gradient_inf) = _objective_and_gradient_inf(model, X_t, y_t, w_t)
        converged = bool(converged or gradient_inf <= max(1e-06, math.sqrt(float(tolerance))))
    with torch.no_grad():
        intercept = float(model.linear.bias.detach().cpu().numpy().ravel()[0])
        beta = model.linear.weight.detach().cpu().numpy().ravel().astype(np.float64)
    if not np.isfinite(intercept) or not np.all(np.isfinite(beta)):
        raise FloatingPointError(f'{label} produced NaN or Inf parameters.')
    return TorchLogisticFit(intercept=intercept, beta=beta, objective=float(final_objective), optimizer=str(optimizer_name), iterations=int(used_iterations), converged=bool(converged), gradient_inf=float(gradient_inf), history=history)

@dataclass
class DWRLogisticResult:
    initial_fit: TorchLogisticFit
    final_fit: TorchLogisticFit
    initial_probabilities: np.ndarray
    curvature: np.ndarray
    curvature_used: np.ndarray
    dwr_weights: np.ndarray
    corrected_weights: np.ndarray
    ignored_mask: np.ndarray
    diagnostics: Dict[str, float]

class DWRLogistic:

    def __init__(self, *, dwr_eta: float=1.0, dwr_max_iter: int=1000, dwr_tol: float=1e-07, dwr_min_iter: int=5, dwr_initial_step: float=1.0, dwr_backtracking_factor: float=0.5, dwr_max_backtracking: int=50, dwr_step_growth: float=1.2, dwr_log_every: int=50, logistic_optimizer: LogisticOptimizer='lbfgs', logistic_learning_rate: float=1.0, logistic_max_iter: int=500, logistic_tolerance: float=1e-08, logistic_patience: int=30, probability_clip: float=1e-08, curvature_floor: float=1e-06, ignore_high_confidence_correct: bool=False, confidence_threshold: float=0.99, corrected_weight_clip_quantile: Optional[float]=None, random_state: int=2026, device: Optional[Union[str, torch.device]]=None, verbose: bool=True) -> None:
        self.dwr_eta = float(dwr_eta)
        self.dwr_max_iter = int(dwr_max_iter)
        self.dwr_tol = float(dwr_tol)
        self.dwr_min_iter = int(dwr_min_iter)
        self.dwr_initial_step = float(dwr_initial_step)
        self.dwr_backtracking_factor = float(dwr_backtracking_factor)
        self.dwr_max_backtracking = int(dwr_max_backtracking)
        self.dwr_step_growth = float(dwr_step_growth)
        self.dwr_log_every = int(dwr_log_every)
        self.logistic_optimizer = logistic_optimizer
        self.logistic_learning_rate = float(logistic_learning_rate)
        self.logistic_max_iter = int(logistic_max_iter)
        self.logistic_tolerance = float(logistic_tolerance)
        self.logistic_patience = int(logistic_patience)
        self.probability_clip = float(probability_clip)
        self.curvature_floor = float(curvature_floor)
        self.ignore_high_confidence_correct = bool(ignore_high_confidence_correct)
        self.confidence_threshold = float(confidence_threshold)
        self.corrected_weight_clip_quantile = corrected_weight_clip_quantile
        self.random_state = int(random_state)
        self.device = _resolve_device(device)
        self.verbose = bool(verbose)
        if self.dwr_eta < 0.0 or not np.isfinite(self.dwr_eta):
            raise ValueError('dwr_eta must be a finite nonnegative number.')
        if not 0.0 < self.probability_clip < 0.5:
            raise ValueError('probability_clip must lie in (0,0.5).')
        if self.curvature_floor <= 0.0 or self.curvature_floor > 0.25:
            raise ValueError('curvature_floor must lie in (0,0.25].')
        if not 0.5 < self.confidence_threshold < 1.0:
            raise ValueError('confidence_threshold must lie in (0.5,1).')
        if self.corrected_weight_clip_quantile is not None and (not 0.5 < float(self.corrected_weight_clip_quantile) < 1.0):
            raise ValueError('corrected_weight_clip_quantile must lie in (0.5,1) or be None.')
        self.result_: Optional[DWRLogisticResult] = None
        self.dwr_: Optional[DWR] = None
        self.feature_names_: Optional[List[str]] = None
        self.X_mean_: Optional[np.ndarray] = None
        self.X_scale_: Optional[np.ndarray] = None
        self.zero_variance_mask_: Optional[np.ndarray] = None

    def _prepare_fit_X(self, X: ArrayLike) -> np.ndarray:
        if hasattr(X, 'columns') and hasattr(X, 'to_numpy'):
            self.feature_names_ = [str(value) for value in list(X.columns)]
        else:
            self.feature_names_ = None
        X_arr = _to_numpy_2d(X)
        p = X_arr.shape[1]
        mean = np.zeros(p, dtype=np.float64)
        scale = np.ones(p, dtype=np.float64)
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

    def fit(self, X: ArrayLike, y: ArrayLike, feature_names: Optional[Sequence[str]]=None) -> 'DWRLogistic':
        _set_random_seed(self.random_state)
        Xs = self._prepare_fit_X(X)
        (n, p) = Xs.shape
        ys = _to_binary_y(y, n)
        if feature_names is not None:
            if len(feature_names) != p:
                raise ValueError('Length of feature_names must equal the number of variables p.')
            self.feature_names_ = [str(value) for value in feature_names]
        if self.verbose:
            print(f'[*] DWR_Logistic Start: n={n}, p={p}, device={self.device}, dwr_eta={self.dwr_eta}, optimizer={self.logistic_optimizer}')
        if self.verbose:
            print('[1/4] Fitting approximate Logistic model ...')
        initial_fit = _fit_torch_logistic(Xs, ys, sample_weight=None, optimizer_name=self.logistic_optimizer, learning_rate=self.logistic_learning_rate, max_iter=self.logistic_max_iter, tolerance=self.logistic_tolerance, patience=self.logistic_patience, random_state=self.random_state, device=self.device, verbose=False, label='Initial Logistic')
        initial_eta = initial_fit.intercept + Xs @ initial_fit.beta
        initial_probabilities = np.clip(_sigmoid_numpy(initial_eta), self.probability_clip, 1.0 - self.probability_clip)
        if self.verbose:
            print('[2/4] Estimating DWR weights ...')
        self.dwr_ = DWR(eta=self.dwr_eta, max_iter=self.dwr_max_iter, tol=self.dwr_tol, min_iter=self.dwr_min_iter, initial_step=self.dwr_initial_step, backtracking_factor=self.dwr_backtracking_factor, max_backtracking=self.dwr_max_backtracking, step_growth=self.dwr_step_growth, verbose=False, log_every=self.dwr_log_every).fit(Xs)
        dwr_weights = self.dwr_.get_weights()
        if self.verbose:
            print('[3/4] Applying Logistic curvature correction ...')
        curvature = initial_probabilities * (1.0 - initial_probabilities)
        curvature_used = np.maximum(curvature, self.curvature_floor)
        corrected_raw = dwr_weights / curvature_used
        ignored_mask = np.zeros(n, dtype=bool)
        if self.ignore_high_confidence_correct:
            predicted = (initial_probabilities >= 0.5).astype(int)
            confidence = np.maximum(initial_probabilities, 1.0 - initial_probabilities)
            ignored_mask = (predicted == ys) & (confidence >= self.confidence_threshold)
            corrected_raw = corrected_raw.copy()
            corrected_raw[ignored_mask] = 0.0
            if corrected_raw.sum() <= 0.0 or np.sum(corrected_raw[ys == 0]) <= 0.0 or np.sum(corrected_raw[ys == 1]) <= 0.0:
                warnings.warn('Ignoring high-confidence samples made the weights degenerate; this step was skipped automatically.', RuntimeWarning)
                ignored_mask[:] = False
                corrected_raw = dwr_weights / curvature_used
        corrected_clip_threshold = float('nan')
        if self.corrected_weight_clip_quantile is not None:
            positive_weights = corrected_raw[corrected_raw > 0.0]
            if positive_weights.size == 0:
                raise FloatingPointError('No positive weights remain after curvature correction.')
            corrected_clip_threshold = float(np.quantile(positive_weights, float(self.corrected_weight_clip_quantile)))
            corrected_raw = np.minimum(corrected_raw, corrected_clip_threshold)
        if not np.all(np.isfinite(corrected_raw)) or np.any(corrected_raw < 0.0) or corrected_raw.mean() <= 0.0:
            raise FloatingPointError('Curvature-corrected weights contain invalid values.')
        if np.sum(corrected_raw[ys == 0]) <= 0.0 or np.sum(corrected_raw[ys == 1]) <= 0.0:
            raise FloatingPointError('After DWR curvature correction one class has zero total weight; increase dwr_eta or curvature_floor, or enable weight clipping.')
        corrected_weights = corrected_raw / float(np.mean(corrected_raw))
        corrected_ess = _effective_sample_size(corrected_weights)
        if self.verbose:
            print('[4/4] Fitting final unpenalized weighted Logistic model ...')
        final_fit = _fit_torch_logistic(Xs, ys, sample_weight=corrected_weights, optimizer_name=self.logistic_optimizer, learning_rate=self.logistic_learning_rate, max_iter=self.logistic_max_iter, tolerance=self.logistic_tolerance, patience=self.logistic_patience, random_state=self.random_state + 1, device=self.device, verbose=False, label='Weighted Logistic')
        dwr_diagnostics = self.dwr_.get_diagnostics()
        diagnostics = {'n_samples': float(n), 'n_features': float(p), 'logistic_penalty_l1': 0.0, 'logistic_penalty_l2': 0.0, 'initial_logistic_objective': float(initial_fit.objective), 'initial_logistic_gradient_inf': float(initial_fit.gradient_inf), 'final_weighted_logistic_objective': float(final_fit.objective), 'final_weighted_logistic_gradient_inf': float(final_fit.gradient_inf), 'initial_positive_probability_min': float(np.min(initial_probabilities)), 'initial_positive_probability_mean': float(np.mean(initial_probabilities)), 'initial_positive_probability_max': float(np.max(initial_probabilities)), 'curvature_min': float(np.min(curvature)), 'curvature_mean': float(np.mean(curvature)), 'curvature_max': float(np.max(curvature)), 'probability_clip': float(self.probability_clip), 'curvature_floor': float(self.curvature_floor), 'n_curvature_floored': float(np.sum(curvature < self.curvature_floor)), 'n_ignored_high_confidence_correct': float(np.sum(ignored_mask)), 'dwr_eta': float(self.dwr_eta), 'dwr_initial_objective': float(dwr_diagnostics['initial_objective']), 'dwr_final_objective': float(dwr_diagnostics['final_objective']), 'dwr_final_decorrelation_loss': float(dwr_diagnostics['final_decorrelation_loss']), 'dwr_final_regularization_loss': float(dwr_diagnostics['final_regularization_loss']), 'dwr_projected_gradient_inf': float(dwr_diagnostics['projected_gradient_inf']), 'dwr_converged': float(dwr_diagnostics['converged']), 'dwr_iterations': float(dwr_diagnostics['iterations']), 'dwr_weight_min': float(np.min(dwr_weights)), 'dwr_weight_mean': float(np.mean(dwr_weights)), 'dwr_weight_max': float(np.max(dwr_weights)), 'dwr_weight_sum': float(np.sum(dwr_weights)), 'dwr_ess': float(_effective_sample_size(dwr_weights)), 'dwr_ess_ratio': float(_effective_sample_size(dwr_weights) / n), 'corrected_weight_min': float(np.min(corrected_weights)), 'corrected_weight_mean': float(np.mean(corrected_weights)), 'corrected_weight_max': float(np.max(corrected_weights)), 'corrected_weight_sum': float(np.sum(corrected_weights)), 'corrected_ess': float(corrected_ess), 'corrected_ess_ratio': float(corrected_ess / n), 'corrected_weight_clip_threshold': corrected_clip_threshold, 'device_is_cuda': float(self.device.type == 'cuda')}
        self.result_ = DWRLogisticResult(initial_fit=initial_fit, final_fit=final_fit, initial_probabilities=initial_probabilities.copy(), curvature=curvature.copy(), curvature_used=curvature_used.copy(), dwr_weights=dwr_weights.copy(), corrected_weights=corrected_weights.copy(), ignored_mask=ignored_mask.copy(), diagnostics=diagnostics)
        if self.verbose:
            print(f"[✓] DWR_Logistic finished | DWR ESS={diagnostics['dwr_ess']:.1f}/{n} | Corrected ESS={corrected_ess:.1f}/{n} | W min/mean/max={corrected_weights.min():.4f}/{corrected_weights.mean():.4f}/{corrected_weights.max():.4f}")
        return self

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise ValueError('DWR_Logistic is not fitted; call fit(X,y) first.')

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

    def get_dwr_weights(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.dwr_weights.copy()

    def get_curvature(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.curvature.copy()

    def get_initial_probabilities(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.initial_probabilities.copy()

    def get_effective_sample_size(self) -> float:
        self._check_fitted()
        return float(self.result_.diagnostics['corrected_ess'])

    def get_dwr_effective_sample_size(self) -> float:
        self._check_fitted()
        return float(self.result_.diagnostics['dwr_ess'])

    def get_diagnostics(self) -> Dict[str, float]:
        self._check_fitted()
        return dict(self.result_.diagnostics)

    def get_dwr_diagnostics(self) -> Dict[str, float]:
        self._check_fitted()
        if self.dwr_ is None:
            raise RuntimeError('Internal error: missing DWR object.')
        return self.dwr_.get_diagnostics()

    def dwr_history_table(self):
        self._check_fitted()
        if self.dwr_ is None:
            raise RuntimeError('Internal error: missing DWR object.')
        return self.dwr_.history_table()

    def get_dwr_model(self) -> DWR:
        self._check_fitted()
        assert self.dwr_ is not None
        return self.dwr_

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

def fit_dwr_logistic(X: ArrayLike, y: ArrayLike, *, feature_names: Optional[Sequence[str]]=None, **kwargs: Any) -> DWRLogistic:
    model = DWRLogistic(**kwargs)
    return model.fit(X, y, feature_names=feature_names)
DWR_Logistic = DWRLogistic
__all__ = ['TorchLogisticFit', 'DWRLogisticResult', 'DWRLogistic', 'DWR_Logistic', 'fit_dwr_logistic']
