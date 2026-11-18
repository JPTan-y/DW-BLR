from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import math
import warnings
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None
ArrayLike = Any
PenaltyName = Literal['none', 'lasso', 'ridge', 'elastic_net', 'elasticnet', 'scad', 'mcp']
CVMetric = Literal['log_loss', 'auc', 'accuracy', 'brier']
NonconvexInit = Literal['lasso', 'ridge', 'zero']

def _to_numpy_2d(X: Any, name: str='X') -> np.ndarray:
    if hasattr(X, 'to_numpy'):
        arr = np.asarray(X.to_numpy(), dtype=np.float64)
    else:
        arr = np.asarray(X, dtype=np.float64)
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
    try:
        arr_float = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('y must be convertible to numeric 0/1 or -1/+1.') from exc
    if not np.all(np.isfinite(arr_float)):
        raise ValueError('y contains NaN or Inf.')
    unique = np.unique(arr_float)
    if unique.size != 2:
        raise ValueError(f'y must contain both classes; observed labels={unique.tolist()}.')
    if np.all(np.isin(unique, [0.0, 1.0])):
        return arr_float
    if np.all(np.isin(unique, [-1.0, 1.0])):
        return (arr_float == 1.0).astype(np.float64)
    raise ValueError(f'y must be coded as {{0,1}} or {{-1,+1}}; observed labels={unique.tolist()}.')

def _normalize_penalty_name(penalty: str) -> str:
    key = str(penalty).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {'none': 'none', 'unpenalized': 'none', 'no_penalty': 'none', 'l1': 'lasso', 'lasso': 'lasso', 'l2': 'ridge', 'ridge': 'ridge', 'elastic': 'elastic_net', 'elasticnet': 'elastic_net', 'elastic_net': 'elastic_net', 'enet': 'elastic_net', 'scad': 'scad', 'mcp': 'mcp'}
    if key not in aliases:
        raise ValueError("penalty must be 'lasso', 'ridge', 'elastic_net', 'scad', 'mcp' or 'none'.")
    return aliases[key]

def _as_positive_grid(value: Union[float, Sequence[float], np.ndarray], name: str, *, allow_zero: bool=True) -> Tuple[np.ndarray, bool]:
    if np.isscalar(value):
        grid = np.array([float(value)], dtype=np.float64)
    else:
        grid = np.asarray(list(value), dtype=np.float64).ravel()
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(f'{name} must be a number or a non-empty one-dimensional sequence.')
    if not np.all(np.isfinite(grid)):
        raise ValueError(f'{name} contains NaN or Inf.')
    if allow_zero:
        if np.any(grid < 0):
            raise ValueError(f'All values of {name} must be >= 0.')
    elif np.any(grid <= 0):
        raise ValueError(f'All values of {name} must be > 0.')
    grid = np.unique(grid)
    return (grid, bool(grid.size > 1))

def _as_ratio_grid(value: Union[float, Sequence[float], np.ndarray]) -> Tuple[np.ndarray, bool]:
    if np.isscalar(value):
        grid = np.array([float(value)], dtype=np.float64)
    else:
        grid = np.asarray(list(value), dtype=np.float64).ravel()
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError('l1_ratio must be a number or a non-empty one-dimensional sequence.')
    if not np.all(np.isfinite(grid)) or np.any(grid < 0) or np.any(grid > 1):
        raise ValueError('All l1_ratio values must lie in [0,1].')
    grid = np.unique(grid)
    return (grid, bool(grid.size > 1))

def _weighted_mean_logloss(y: np.ndarray, prob: np.ndarray) -> float:
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(prob) + (1.0 - y) * np.log1p(-prob)))

def _binary_auc(y: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y, dtype=int).ravel()
    prob = np.asarray(prob, dtype=np.float64).ravel()
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return float('nan')
    ranks = rankdata(prob, method='average')
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return float((rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def _metric_value(metric: str, y: np.ndarray, prob: np.ndarray) -> float:
    metric = str(metric).lower()
    if metric == 'log_loss':
        return _weighted_mean_logloss(y, prob)
    if metric == 'auc':
        return _binary_auc(y, prob)
    if metric == 'accuracy':
        return float(np.mean((prob >= 0.5).astype(int) == y.astype(int)))
    if metric == 'brier':
        return float(np.mean((np.asarray(prob) - np.asarray(y)) ** 2))
    raise ValueError("cv_metric must be 'log_loss', 'auc', 'accuracy' or 'brier'.")

def _metric_is_larger_better(metric: str) -> bool:
    return str(metric).lower() in {'auc', 'accuracy'}

def _soft_threshold(x: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

def _scad_derivative(abs_beta: np.ndarray, lam: float, a: float) -> np.ndarray:
    t = np.asarray(abs_beta, dtype=np.float64)
    if lam <= 0:
        return np.zeros_like(t)
    out = np.empty_like(t)
    region1 = t <= lam
    region2 = (t > lam) & (t <= a * lam)
    region3 = t > a * lam
    out[region1] = lam
    out[region2] = (a * lam - t[region2]) / (a - 1.0)
    out[region3] = 0.0
    return np.maximum(out, 0.0)

def _scad_penalty(abs_beta: np.ndarray, lam: float, a: float) -> np.ndarray:
    t = np.asarray(abs_beta, dtype=np.float64)
    if lam <= 0:
        return np.zeros_like(t)
    out = np.empty_like(t)
    region1 = t <= lam
    region2 = (t > lam) & (t <= a * lam)
    region3 = t > a * lam
    out[region1] = lam * t[region1]
    out[region2] = (-t[region2] ** 2 + 2.0 * a * lam * t[region2] - lam ** 2) / (2.0 * (a - 1.0))
    out[region3] = 0.5 * (a + 1.0) * lam ** 2
    return out

def _mcp_derivative(abs_beta: np.ndarray, lam: float, gamma: float) -> np.ndarray:
    t = np.asarray(abs_beta, dtype=np.float64)
    if lam <= 0:
        return np.zeros_like(t)
    return np.maximum(lam - t / gamma, 0.0)

def _mcp_penalty(abs_beta: np.ndarray, lam: float, gamma: float) -> np.ndarray:
    t = np.asarray(abs_beta, dtype=np.float64)
    if lam <= 0:
        return np.zeros_like(t)
    out = np.empty_like(t)
    region1 = t <= gamma * lam
    region2 = ~region1
    out[region1] = lam * t[region1] - t[region1] ** 2 / (2.0 * gamma)
    out[region2] = 0.5 * gamma * lam ** 2
    return out

@dataclass
class _OptimizationResult:
    intercept: float
    beta: np.ndarray
    objective: float
    data_loss: float
    penalty_value: float
    converged: bool
    n_iter: int
    stationarity_inf: float
    history: List[Dict[str, float]] = field(default_factory=list)
    message: str = ''

def _smooth_loss_grad(X: np.ndarray, y: np.ndarray, intercept: float, beta: np.ndarray, l2_strength: float) -> Tuple[float, float, np.ndarray]:
    eta = float(intercept) + X @ beta
    data_loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    smooth_loss = data_loss + 0.5 * float(l2_strength) * float(np.dot(beta, beta))
    residual = expit(eta) - y
    grad_intercept = float(np.mean(residual))
    grad_beta = X.T @ residual / float(X.shape[0]) + float(l2_strength) * beta
    return (smooth_loss, grad_intercept, grad_beta)

def _convex_objective_components(X: np.ndarray, y: np.ndarray, intercept: float, beta: np.ndarray, l1_weights: np.ndarray, l2_strength: float) -> Tuple[float, float, float, float]:
    eta = float(intercept) + X @ beta
    data_loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    l1_value = float(np.dot(l1_weights, np.abs(beta)))
    l2_value = 0.5 * float(l2_strength) * float(np.dot(beta, beta))
    return (data_loss + l1_value + l2_value, data_loss, l1_value, l2_value)

def _initial_step_upper_bound(X: np.ndarray, l2_strength: float) -> float:
    n = X.shape[0]
    fro_sq_augmented = float(n + np.sum(X ** 2))
    lipschitz_upper = 0.25 * fro_sq_augmented / float(n) + float(l2_strength)
    return 1.0 / max(lipschitz_upper, 1e-12)

def _proximal_mapping_inf(X: np.ndarray, y: np.ndarray, intercept: float, beta: np.ndarray, l1_weights: np.ndarray, l2_strength: float, step: float) -> float:
    (_, g0, gb) = _smooth_loss_grad(X, y, intercept, beta, l2_strength)
    prox_intercept = intercept - step * g0
    prox_beta = _soft_threshold(beta - step * gb, step * l1_weights)
    mapping_intercept = (intercept - prox_intercept) / step
    mapping_beta = (beta - prox_beta) / step
    return float(max(abs(mapping_intercept), np.max(np.abs(mapping_beta))))

def _convex_kkt_inf(X: np.ndarray, y: np.ndarray, intercept: float, beta: np.ndarray, l1_weights: np.ndarray, l2_strength: float, zero_threshold: float=1e-10) -> float:
    (_, grad0, grad_beta) = _smooth_loss_grad(X, y, intercept, beta, l2_strength)
    residual = np.zeros_like(beta)
    active = np.abs(beta) > zero_threshold
    residual[active] = np.abs(grad_beta[active] + l1_weights[active] * np.sign(beta[active]))
    residual[~active] = np.maximum(np.abs(grad_beta[~active]) - l1_weights[~active], 0.0)
    return float(max(abs(grad0), np.max(residual)))

def _solve_convex_logistic(X: np.ndarray, y: np.ndarray, *, l1_weights: np.ndarray, l2_strength: float, max_iter: int, tol: float, min_iter: int, initial_step: Optional[float], backtracking_factor: float, max_backtracking: int, step_growth: float, monotone_restart: bool, init_intercept: Optional[float]=None, init_beta: Optional[np.ndarray]=None, store_history: bool=True, verbose: bool=False, log_every: int=100) -> _OptimizationResult:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    (n, p) = X.shape
    l1_weights = np.asarray(l1_weights, dtype=np.float64).ravel()
    if l1_weights.shape != (p,):
        raise ValueError('Length of l1_weights must equal the number of variables p.')
    if np.any(l1_weights < 0) or not np.all(np.isfinite(l1_weights)):
        raise ValueError('l1_weights must be finite and nonnegative.')
    if l2_strength < 0 or not np.isfinite(l2_strength):
        raise ValueError('l2_strength must be a finite nonnegative number.')
    if max_iter <= 0 or min_iter < 0 or min_iter > max_iter:
        raise ValueError('max_iter must be > 0 with 0 <= min_iter <= max_iter.')
    if tol <= 0:
        raise ValueError('tol must be > 0.')
    if not 0.0 < backtracking_factor < 1.0:
        raise ValueError('backtracking_factor must lie in (0,1).')
    if max_backtracking <= 0:
        raise ValueError('max_backtracking must be > 0.')
    if step_growth < 1.0:
        raise ValueError('step_growth must be >= 1.')
    positive_rate = float(np.clip(np.mean(y), 1e-08, 1.0 - 1e-08))
    if init_intercept is None:
        intercept = math.log(positive_rate / (1.0 - positive_rate))
    else:
        intercept = float(init_intercept)
    if init_beta is None:
        beta = np.zeros(p, dtype=np.float64)
    else:
        beta = np.asarray(init_beta, dtype=np.float64).ravel().copy()
        if beta.shape != (p,):
            raise ValueError('Length of init_beta must equal the number of variables p.')
    if not np.isfinite(intercept) or not np.all(np.isfinite(beta)):
        raise ValueError('Initial parameters contain NaN/Inf.')
    current_intercept = intercept
    current_beta = beta.copy()
    extrap_intercept = current_intercept
    extrap_beta = current_beta.copy()
    t_fista = 1.0
    if initial_step is None:
        step = _initial_step_upper_bound(X, l2_strength)
        max_step = 1000000.0
    else:
        if initial_step <= 0 or not np.isfinite(initial_step):
            raise ValueError('initial_step must be a finite positive number or None.')
        step = float(initial_step)
        max_step = max(float(initial_step), 1e-12) * 1000000.0
    (current_obj, current_data, current_l1, current_l2) = _convex_objective_components(X, y, current_intercept, current_beta, l1_weights, l2_strength)
    best_obj = current_obj
    best_intercept = current_intercept
    best_beta = current_beta.copy()
    history: List[Dict[str, float]] = []
    converged = False
    stationarity = np.inf
    last_change = np.inf
    used_iter = 0
    message = 'Reached the maximum number of iterations.'

    def _prox_step(point_intercept: float, point_beta: np.ndarray, step_start: float) -> Tuple[float, np.ndarray, float, float, float, float, float]:
        (smooth_point, grad0, gradb) = _smooth_loss_grad(X, y, point_intercept, point_beta, l2_strength)
        local_step = float(step_start)
        candidate_intercept = point_intercept
        candidate_beta = point_beta.copy()
        candidate_smooth = smooth_point
        for _ in range(max_backtracking):
            candidate_intercept = point_intercept - local_step * grad0
            candidate_beta = _soft_threshold(point_beta - local_step * gradb, local_step * l1_weights)
            (candidate_smooth, _, _) = _smooth_loss_grad(X, y, candidate_intercept, candidate_beta, l2_strength)
            diff0 = candidate_intercept - point_intercept
            diffb = candidate_beta - point_beta
            quadratic_upper = smooth_point + grad0 * diff0 + float(np.dot(gradb, diffb)) + (diff0 ** 2 + float(np.dot(diffb, diffb))) / (2.0 * local_step)
            if np.isfinite(candidate_smooth) and candidate_smooth <= quadratic_upper + 1e-12:
                break
            local_step *= backtracking_factor
        else:
            raise FloatingPointError('FISTA backtracking line search found no step satisfying the majorization condition.')
        (total, data_loss, l1_value, l2_value) = _convex_objective_components(X, y, candidate_intercept, candidate_beta, l1_weights, l2_strength)
        return (float(candidate_intercept), candidate_beta, float(total), float(data_loss), float(l1_value), float(l2_value), float(local_step))
    for iteration in range(1, max_iter + 1):
        used_iter = iteration
        (candidate_intercept, candidate_beta, candidate_obj, candidate_data, candidate_l1, candidate_l2, used_step) = _prox_step(extrap_intercept, extrap_beta, step)
        if monotone_restart and candidate_obj > current_obj + 1e-12:
            extrap_intercept = current_intercept
            extrap_beta = current_beta.copy()
            t_fista = 1.0
            (candidate_intercept, candidate_beta, candidate_obj, candidate_data, candidate_l1, candidate_l2, used_step) = _prox_step(extrap_intercept, extrap_beta, used_step)
        if not (np.isfinite(candidate_obj) and np.isfinite(candidate_intercept) and np.all(np.isfinite(candidate_beta))):
            raise FloatingPointError('Logistic FISTA produced NaN/Inf.')
        last_change = float(max(abs(candidate_intercept - current_intercept), np.max(np.abs(candidate_beta - current_beta))))
        stationarity = _convex_kkt_inf(X, y, candidate_intercept, candidate_beta, l1_weights, l2_strength)
        if candidate_obj < best_obj:
            best_obj = float(candidate_obj)
            best_intercept = float(candidate_intercept)
            best_beta = candidate_beta.copy()
        if store_history:
            history.append({'iteration': float(iteration), 'objective': float(candidate_obj), 'data_loss': float(candidate_data), 'l1_penalty': float(candidate_l1), 'l2_penalty': float(candidate_l2), 'max_parameter_change': float(last_change), 'stationarity_inf': float(stationarity), 'step_size': float(used_step), 'n_nonzero': float(np.sum(np.abs(candidate_beta) > 1e-10))})
        if verbose and (iteration == 1 or iteration % max(1, log_every) == 0):
            print(f'      iter={iteration:5d} | obj={candidate_obj:.8f} | stationarity={stationarity:.3e} | d={last_change:.3e} | nz={int(np.sum(np.abs(candidate_beta) > 1e-10))}')
        if iteration >= min_iter and stationarity < tol:
            converged = True
            message = 'KKT stationarity reached tolerance.'
            current_intercept = candidate_intercept
            current_beta = candidate_beta.copy()
            current_obj = candidate_obj
            current_data = candidate_data
            current_l1 = candidate_l1
            current_l2 = candidate_l2
            break
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_fista ** 2))
        momentum = (t_fista - 1.0) / t_new
        new_extrap_intercept = candidate_intercept + momentum * (candidate_intercept - current_intercept)
        new_extrap_beta = candidate_beta + momentum * (candidate_beta - current_beta)
        current_intercept = candidate_intercept
        current_beta = candidate_beta.copy()
        current_obj = candidate_obj
        current_data = candidate_data
        current_l1 = candidate_l1
        current_l2 = candidate_l2
        extrap_intercept = float(new_extrap_intercept)
        extrap_beta = new_extrap_beta
        t_fista = t_new
        step = min(max_step, used_step * step_growth)
    (best_total, best_data, best_l1, best_l2) = _convex_objective_components(X, y, best_intercept, best_beta, l1_weights, l2_strength)
    final_stationarity = _convex_kkt_inf(X, y, best_intercept, best_beta, l1_weights, l2_strength)
    if final_stationarity < tol:
        converged = True
        message = 'KKT stationarity at the optimum reached tolerance.'
    return _OptimizationResult(intercept=float(best_intercept), beta=best_beta.astype(np.float64), objective=float(best_total), data_loss=float(best_data), penalty_value=float(best_l1 + best_l2), converged=bool(converged), n_iter=int(used_iter), stationarity_inf=float(final_stationarity), history=history, message=message)

def _nonconvex_objective_components(X: np.ndarray, y: np.ndarray, intercept: float, beta: np.ndarray, penalty: str, lam: float, scad_a: float, mcp_gamma: float) -> Tuple[float, float, float]:
    eta = float(intercept) + X @ beta
    data_loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    if penalty == 'scad':
        penalty_value = float(np.sum(_scad_penalty(np.abs(beta), lam, scad_a)))
    elif penalty == 'mcp':
        penalty_value = float(np.sum(_mcp_penalty(np.abs(beta), lam, mcp_gamma)))
    else:
        raise ValueError('The nonconvex objective supports only scad or mcp.')
    return (data_loss + penalty_value, data_loss, penalty_value)

def _nonconvex_stationarity_inf(X: np.ndarray, y: np.ndarray, intercept: float, beta: np.ndarray, penalty: str, lam: float, scad_a: float, mcp_gamma: float, zero_threshold: float) -> float:
    eta = float(intercept) + X @ beta
    residual = expit(eta) - y
    grad0 = float(np.mean(residual))
    grad = X.T @ residual / float(X.shape[0])
    if penalty == 'scad':
        derivative = _scad_derivative(np.abs(beta), lam, scad_a)
    else:
        derivative = _mcp_derivative(np.abs(beta), lam, mcp_gamma)
    residuals = np.zeros_like(beta)
    active = np.abs(beta) > zero_threshold
    residuals[active] = np.abs(grad[active] + derivative[active] * np.sign(beta[active]))
    residuals[~active] = np.maximum(np.abs(grad[~active]) - lam, 0.0)
    return float(max(abs(grad0), np.max(residuals)))

def _nonconvex_start(X: np.ndarray, y: np.ndarray, *, init_kind: str, lam: float, ridge_strength: float, solver_kwargs: Dict[str, Any]) -> Tuple[float, np.ndarray]:
    p = X.shape[1]
    positive_rate = float(np.clip(np.mean(y), 1e-08, 1.0 - 1e-08))
    base_intercept = math.log(positive_rate / (1.0 - positive_rate))
    if init_kind == 'zero':
        return (base_intercept, np.zeros(p, dtype=np.float64))
    if init_kind == 'lasso':
        result = _solve_convex_logistic(X, y, l1_weights=np.full(p, lam, dtype=np.float64), l2_strength=0.0, init_intercept=base_intercept, init_beta=np.zeros(p, dtype=np.float64), store_history=False, verbose=False, **solver_kwargs)
        return (result.intercept, result.beta)
    if init_kind == 'ridge':
        result = _solve_convex_logistic(X, y, l1_weights=np.zeros(p, dtype=np.float64), l2_strength=float(ridge_strength), init_intercept=base_intercept, init_beta=np.zeros(p, dtype=np.float64), store_history=False, verbose=False, **solver_kwargs)
        return (result.intercept, result.beta)
    raise ValueError("nonconvex_init must be 'lasso', 'ridge' or 'zero'.")

def _solve_nonconvex_logistic(X: np.ndarray, y: np.ndarray, *, penalty: str, lam: float, scad_a: float, mcp_gamma: float, nonconvex_init: str, nonconvex_multi_start: bool, nonconvex_init_ridge: float, lla_max_iter: int, lla_tol: float, lla_min_iter: int, lla_monotone: bool, lla_max_line_search: int, coef_zero_threshold: float, solver_kwargs: Dict[str, Any], store_history: bool, verbose: bool, log_every: int) -> _OptimizationResult:
    if penalty not in {'scad', 'mcp'}:
        raise ValueError('The nonconvex solver supports only SCAD or MCP.')
    if lam < 0:
        raise ValueError('lambda must be >= 0.')
    if scad_a <= 2:
        raise ValueError('SCAD parameter scad_a must be > 2; 3.7 is a common value.')
    if mcp_gamma <= 1:
        raise ValueError('MCP parameter mcp_gamma must be > 1; 3 is a common value.')
    if lla_max_iter <= 0 or lla_min_iter < 0 or lla_min_iter > lla_max_iter:
        raise ValueError('Invalid LLA iteration settings.')
    if lla_tol <= 0:
        raise ValueError('lla_tol must be > 0.')
    if lla_max_line_search <= 0:
        raise ValueError('lla_max_line_search must be > 0.')
    if lam == 0:
        convex = _solve_convex_logistic(X, y, l1_weights=np.zeros(X.shape[1], dtype=np.float64), l2_strength=0.0, store_history=store_history, verbose=verbose, **solver_kwargs)
        return convex
    starts = [str(nonconvex_init).lower()]
    if nonconvex_multi_start:
        for extra in ('lasso', 'ridge'):
            if extra not in starts:
                starts.append(extra)
    all_results: List[_OptimizationResult] = []
    for (start_id, init_kind) in enumerate(starts):
        (intercept, beta) = _nonconvex_start(X, y, init_kind=init_kind, lam=lam, ridge_strength=nonconvex_init_ridge, solver_kwargs=solver_kwargs)
        (objective, data_loss, penalty_value) = _nonconvex_objective_components(X, y, intercept, beta, penalty, lam, scad_a, mcp_gamma)
        history: List[Dict[str, float]] = []
        converged = False
        stationarity = _nonconvex_stationarity_inf(X, y, intercept, beta, penalty, lam, scad_a, mcp_gamma, coef_zero_threshold)
        used_iter = 0
        message = 'Reached the maximum number of LLA iterations.'
        for outer in range(1, lla_max_iter + 1):
            used_iter = outer
            if penalty == 'scad':
                local_weights = _scad_derivative(np.abs(beta), lam, scad_a)
            else:
                local_weights = _mcp_derivative(np.abs(beta), lam, mcp_gamma)
            candidate = _solve_convex_logistic(X, y, l1_weights=local_weights, l2_strength=0.0, init_intercept=intercept, init_beta=beta, store_history=False, verbose=False, **solver_kwargs)
            cand_intercept = candidate.intercept
            cand_beta = candidate.beta.copy()
            (cand_obj, cand_data, cand_penalty) = _nonconvex_objective_components(X, y, cand_intercept, cand_beta, penalty, lam, scad_a, mcp_gamma)
            line_fraction = 1.0
            if lla_monotone and cand_obj > objective + 1e-12:
                accepted = False
                for _ in range(lla_max_line_search):
                    line_fraction *= 0.5
                    trial_intercept = intercept + line_fraction * (cand_intercept - intercept)
                    trial_beta = beta + line_fraction * (cand_beta - beta)
                    (trial_obj, trial_data, trial_penalty) = _nonconvex_objective_components(X, y, trial_intercept, trial_beta, penalty, lam, scad_a, mcp_gamma)
                    if trial_obj <= objective + 1e-12:
                        cand_intercept = float(trial_intercept)
                        cand_beta = trial_beta
                        cand_obj = float(trial_obj)
                        cand_data = float(trial_data)
                        cand_penalty = float(trial_penalty)
                        accepted = True
                        break
                if not accepted:
                    cand_intercept = float(intercept)
                    cand_beta = beta.copy()
                    cand_obj = float(objective)
                    cand_data = float(data_loss)
                    cand_penalty = float(penalty_value)
                    line_fraction = 0.0
            max_change = float(max(abs(cand_intercept - intercept), np.max(np.abs(cand_beta - beta))))
            intercept = float(cand_intercept)
            beta = cand_beta.copy()
            objective = float(cand_obj)
            data_loss = float(cand_data)
            penalty_value = float(cand_penalty)
            stationarity = _nonconvex_stationarity_inf(X, y, intercept, beta, penalty, lam, scad_a, mcp_gamma, coef_zero_threshold)
            if store_history:
                history.append({'lla_iteration': float(outer), 'objective': float(objective), 'data_loss': float(data_loss), 'penalty_value': float(penalty_value), 'max_parameter_change': float(max_change), 'stationarity_inf': float(stationarity), 'line_search_fraction': float(line_fraction), 'n_nonzero': float(np.sum(np.abs(beta) > coef_zero_threshold)), 'start_id': float(start_id)})
            if verbose and (outer == 1 or outer % max(1, log_every) == 0):
                print(f'      LLA={outer:3d} | start={init_kind:5s} | obj={objective:.8f} | stationarity={stationarity:.3e} | d={max_change:.3e} | nz={int(np.sum(np.abs(beta) > coef_zero_threshold))}')
            if outer >= lla_min_iter and max_change < lla_tol and (stationarity < max(lla_tol, solver_kwargs['tol'] * 10.0)):
                converged = True
                message = 'LLA parameter change and nonconvex stationarity residual reached tolerance.'
                break
            if line_fraction == 0.0:
                message = 'LLA monotone safeguard found no descent step; returning the current stationary approximation.'
                break
        all_results.append(_OptimizationResult(intercept=float(intercept), beta=beta.astype(np.float64), objective=float(objective), data_loss=float(data_loss), penalty_value=float(penalty_value), converged=bool(converged), n_iter=int(used_iter), stationarity_inf=float(stationarity), history=history, message=f'init={init_kind}; {message}'))
    all_results.sort(key=lambda result: (result.objective, int(np.sum(np.abs(result.beta) > coef_zero_threshold))))
    return all_results[0]

@dataclass
class _CandidateFit:
    intercept_input_scale: float
    beta_input_scale: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    zero_variance_mask: np.ndarray
    optimization: _OptimizationResult

def _fit_candidate(X: np.ndarray, y: np.ndarray, *, penalty: str, lam: float, l1_ratio: float, scad_a: float, mcp_gamma: float, nonconvex_init: str, nonconvex_multi_start: bool, nonconvex_init_ridge: float, lla_max_iter: int, lla_tol: float, lla_min_iter: int, lla_monotone: bool, lla_max_line_search: int, coef_zero_threshold: float, solver_kwargs: Dict[str, Any], store_history: bool, verbose: bool, log_every: int) -> _CandidateFit:
    X = _to_numpy_2d(X)
    y = _to_binary_y(y, X.shape[0])
    p = X.shape[1]
    mean = np.zeros(p, dtype=np.float64)
    scale = np.ones(p, dtype=np.float64)
    zero_mask = np.zeros(p, dtype=bool)
    Xs = X.copy()
    if penalty == 'none':
        l1_weights = np.zeros(p, dtype=np.float64)
        l2_strength = 0.0
        optimization = _solve_convex_logistic(Xs, y, l1_weights=l1_weights, l2_strength=l2_strength, store_history=store_history, verbose=verbose, log_every=log_every, **solver_kwargs)
    elif penalty == 'lasso':
        optimization = _solve_convex_logistic(Xs, y, l1_weights=np.full(p, lam, dtype=np.float64), l2_strength=0.0, store_history=store_history, verbose=verbose, log_every=log_every, **solver_kwargs)
    elif penalty == 'ridge':
        optimization = _solve_convex_logistic(Xs, y, l1_weights=np.zeros(p, dtype=np.float64), l2_strength=lam, store_history=store_history, verbose=verbose, log_every=log_every, **solver_kwargs)
    elif penalty == 'elastic_net':
        optimization = _solve_convex_logistic(Xs, y, l1_weights=np.full(p, lam * l1_ratio, dtype=np.float64), l2_strength=lam * (1.0 - l1_ratio), store_history=store_history, verbose=verbose, log_every=log_every, **solver_kwargs)
    elif penalty in {'scad', 'mcp'}:
        optimization = _solve_nonconvex_logistic(Xs, y, penalty=penalty, lam=lam, scad_a=scad_a, mcp_gamma=mcp_gamma, nonconvex_init=nonconvex_init, nonconvex_multi_start=nonconvex_multi_start, nonconvex_init_ridge=nonconvex_init_ridge, lla_max_iter=lla_max_iter, lla_tol=lla_tol, lla_min_iter=lla_min_iter, lla_monotone=lla_monotone, lla_max_line_search=lla_max_line_search, coef_zero_threshold=coef_zero_threshold, solver_kwargs=solver_kwargs, store_history=store_history, verbose=verbose, log_every=log_every)
    else:
        raise RuntimeError('Unknown penalty.')
    return _CandidateFit(intercept_input_scale=float(optimization.intercept), beta_input_scale=optimization.beta.copy(), mean=mean, scale=scale, zero_variance_mask=zero_mask, optimization=optimization)

def _candidate_probabilities(fit: _CandidateFit, X: np.ndarray) -> np.ndarray:
    X = _to_numpy_2d(X)
    Xs = X
    return expit(fit.intercept_input_scale + Xs @ fit.beta_input_scale)

def _evaluate_cv_candidate(candidate_id: int, lam: float, l1_ratio: float, X: np.ndarray, y: np.ndarray, folds: List[Tuple[np.ndarray, np.ndarray]], *, penalty: str, cv_metric: str, scad_a: float, mcp_gamma: float, nonconvex_init: str, nonconvex_multi_start: bool, nonconvex_init_ridge: float, lla_max_iter: int, lla_tol: float, lla_min_iter: int, lla_monotone: bool, lla_max_line_search: int, coef_zero_threshold: float, solver_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    scores: List[float] = []
    nonzero_counts: List[float] = []
    converged_flags: List[float] = []
    stationarity_values: List[float] = []
    for (train_idx, valid_idx) in folds:
        fit = _fit_candidate(X[train_idx], y[train_idx], penalty=penalty, lam=lam, l1_ratio=l1_ratio, scad_a=scad_a, mcp_gamma=mcp_gamma, nonconvex_init=nonconvex_init, nonconvex_multi_start=nonconvex_multi_start, nonconvex_init_ridge=nonconvex_init_ridge, lla_max_iter=lla_max_iter, lla_tol=lla_tol, lla_min_iter=lla_min_iter, lla_monotone=lla_monotone, lla_max_line_search=lla_max_line_search, coef_zero_threshold=coef_zero_threshold, solver_kwargs=solver_kwargs, store_history=False, verbose=False, log_every=100)
        prob = _candidate_probabilities(fit, X[valid_idx])
        scores.append(_metric_value(cv_metric, y[valid_idx], prob))
        nonzero_counts.append(float(np.sum(np.abs(fit.beta_input_scale) > coef_zero_threshold)))
        converged_flags.append(float(fit.optimization.converged))
        stationarity_values.append(float(fit.optimization.stationarity_inf))
    score_array = np.asarray(scores, dtype=np.float64)
    return {'candidate_id': int(candidate_id), 'lambda': float(lam), 'l1_ratio': float(l1_ratio), 'mean_cv_score': float(np.nanmean(score_array)), 'std_cv_score': float(np.nanstd(score_array, ddof=1)) if len(score_array) > 1 else 0.0, 'mean_n_nonzero': float(np.mean(nonzero_counts)), 'all_folds_converged': bool(np.all(np.asarray(converged_flags) > 0.5)), 'mean_stationarity_inf': float(np.mean(stationarity_values)), 'fold_scores': score_array.tolist()}

@dataclass
class PenalizedLogisticResult:
    penalty: str
    best_lambda: float
    best_l1_ratio: float
    intercept_input_scale: float
    beta_input_scale: np.ndarray
    selected_features: np.ndarray
    selected_feature_names: Optional[List[str]]
    converged: bool
    n_iter: int
    objective: float
    data_loss: float
    penalty_value: float
    stationarity_inf: float
    optimization_message: str
    cv_performed: bool
    cv_metric: str
    cv_results: List[Dict[str, Any]] = field(default_factory=list)
    optimization_history: List[Dict[str, float]] = field(default_factory=list)

class PenalizedLogistic:

    def __init__(self, *, penalty: PenaltyName='lasso', penalty_strength: Union[float, Sequence[float], np.ndarray]=0.01, l1_ratio: Union[float, Sequence[float], np.ndarray]=0.5, scad_a: float=3.7, mcp_gamma: float=3.0, nonconvex_init: NonconvexInit='lasso', nonconvex_multi_start: bool=False, nonconvex_init_ridge: float=0.001, lla_max_iter: int=50, lla_tol: float=1e-06, lla_min_iter: int=2, lla_monotone: bool=True, lla_max_line_search: int=30, cv_folds: int=5, cv_metric: CVMetric='log_loss', cv_shuffle: bool=True, n_jobs: int=1, prefer_stronger_penalty_on_tie: bool=True, max_iter: int=5000, tol: float=1e-07, min_iter: int=5, initial_step: Optional[float]=None, backtracking_factor: float=0.5, max_backtracking: int=50, step_growth: float=1.2, monotone_restart: bool=True, coef_threshold: float=1e-08, random_state: int=2026, verbose: bool=True, log_every: int=100):
        self.penalty = _normalize_penalty_name(penalty)
        self.penalty_strength = penalty_strength
        self.l1_ratio = l1_ratio
        self.scad_a = float(scad_a)
        self.mcp_gamma = float(mcp_gamma)
        self.nonconvex_init = str(nonconvex_init).lower()
        self.nonconvex_multi_start = bool(nonconvex_multi_start)
        self.nonconvex_init_ridge = float(nonconvex_init_ridge)
        self.lla_max_iter = int(lla_max_iter)
        self.lla_tol = float(lla_tol)
        self.lla_min_iter = int(lla_min_iter)
        self.lla_monotone = bool(lla_monotone)
        self.lla_max_line_search = int(lla_max_line_search)
        self.cv_folds = int(cv_folds)
        self.cv_metric = str(cv_metric).lower()
        self.cv_shuffle = bool(cv_shuffle)
        self.n_jobs = int(n_jobs)
        self.prefer_stronger_penalty_on_tie = bool(prefer_stronger_penalty_on_tie)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_iter = int(min_iter)
        self.initial_step = initial_step
        self.backtracking_factor = float(backtracking_factor)
        self.max_backtracking = int(max_backtracking)
        self.step_growth = float(step_growth)
        self.monotone_restart = bool(monotone_restart)
        self.coef_threshold = float(coef_threshold)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.log_every = int(log_every)
        self.result_: Optional[PenalizedLogisticResult] = None
        self.feature_names_: Optional[List[str]] = None
        self.X_mean_: Optional[np.ndarray] = None
        self.X_scale_: Optional[np.ndarray] = None
        self.zero_variance_mask_: Optional[np.ndarray] = None
        self._validate_static_parameters()

    def _validate_static_parameters(self) -> None:
        if self.scad_a <= 2:
            raise ValueError('scad_a must be > 2; the classical default is 3.7.')
        if self.mcp_gamma <= 1:
            raise ValueError('mcp_gamma must be > 1; the common default is 3.')
        if self.nonconvex_init not in {'lasso', 'ridge', 'zero'}:
            raise ValueError("nonconvex_init must be 'lasso', 'ridge' or 'zero'.")
        if self.nonconvex_init_ridge < 0:
            raise ValueError('nonconvex_init_ridge must be >= 0.')
        if self.lla_max_iter <= 0 or not 0 <= self.lla_min_iter <= self.lla_max_iter:
            raise ValueError('Invalid LLA iteration settings.')
        if self.lla_tol <= 0 or self.lla_max_line_search <= 0:
            raise ValueError('lla_tol and lla_max_line_search must be positive.')
        if self.cv_metric not in {'log_loss', 'auc', 'accuracy', 'brier'}:
            raise ValueError('cv_metric must be log_loss, auc, accuracy or brier.')
        if self.cv_folds < 2:
            raise ValueError('cv_folds must be >= 2.')
        if self.n_jobs == 0:
            raise ValueError('n_jobs cannot be 0.')
        if self.max_iter <= 0 or not 0 <= self.min_iter <= self.max_iter:
            raise ValueError('Invalid max_iter/min_iter settings.')
        if self.tol <= 0:
            raise ValueError('tol must be > 0.')
        if self.initial_step is not None and self.initial_step <= 0:
            raise ValueError('initial_step must be positive or None.')
        if not 0 < self.backtracking_factor < 1:
            raise ValueError('backtracking_factor must lie in (0,1).')
        if self.max_backtracking <= 0 or self.step_growth < 1:
            raise ValueError('max_backtracking must be > 0 and step_growth must be >= 1.')
        if self.coef_threshold < 0:
            raise ValueError('coef_threshold must be >= 0.')
        if self.log_every <= 0:
            raise ValueError('log_every must be > 0.')

    def _solver_kwargs(self) -> Dict[str, Any]:
        return {'max_iter': self.max_iter, 'tol': self.tol, 'min_iter': self.min_iter, 'initial_step': self.initial_step, 'backtracking_factor': self.backtracking_factor, 'max_backtracking': self.max_backtracking, 'step_growth': self.step_growth, 'monotone_restart': self.monotone_restart}

    def fit(self, X: Any, y: Any, feature_names: Optional[Sequence[str]]=None) -> 'PenalizedLogistic':
        X_arr = _to_numpy_2d(X)
        y_arr = _to_binary_y(y, X_arr.shape[0])
        (n, p) = X_arr.shape
        if hasattr(X, 'columns') and hasattr(X, 'to_numpy'):
            inferred_names = [str(c) for c in list(X.columns)]
        else:
            inferred_names = None
        if feature_names is not None:
            if len(feature_names) != p:
                raise ValueError('Length of feature_names must equal the number of variables p.')
            self.feature_names_ = [str(v) for v in feature_names]
        else:
            self.feature_names_ = inferred_names
        (lambda_grid, lambda_multiple) = _as_positive_grid(self.penalty_strength, 'penalty_strength', allow_zero=True)
        if self.penalty == 'none':
            if np.any(lambda_grid != 0):
                warnings.warn("With penalty='none', penalty_strength is ignored and set to 0.", RuntimeWarning)
            lambda_grid = np.array([0.0], dtype=np.float64)
            lambda_multiple = False
        if self.penalty == 'elastic_net':
            (ratio_grid, ratio_multiple) = _as_ratio_grid(self.l1_ratio)
        else:
            ratio_grid = np.array([1.0 if self.penalty == 'lasso' else 0.0], dtype=np.float64)
            ratio_multiple = False
            if not np.isscalar(self.l1_ratio) or float(np.asarray(self.l1_ratio).ravel()[0]) != 0.5:
                warnings.warn("l1_ratio only applies to penalty='elastic_net'; it is ignored for the current method.", RuntimeWarning)
        perform_cv = bool(lambda_multiple or ratio_multiple)
        if perform_cv:
            class_counts = np.bincount(y_arr.astype(int), minlength=2)
            if int(np.min(class_counts)) < self.cv_folds:
                raise ValueError(f'Each class must have at least cv_folds={self.cv_folds} samples; class counts={class_counts.tolist()}.')
            splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=self.cv_shuffle, random_state=self.random_state if self.cv_shuffle else None)
            folds = [(train.copy(), valid.copy()) for (train, valid) in splitter.split(X_arr, y_arr)]
            candidates: List[Tuple[int, float, float]] = []
            cid = 0
            for lam in lambda_grid:
                for ratio in ratio_grid:
                    candidates.append((cid, float(lam), float(ratio)))
                    cid += 1
            if self.verbose:
                print(f'[*] Penalized Logistic CV: penalty={self.penalty}, n={n}, p={p}, candidates={len(candidates)}, folds={self.cv_folds}, metric={self.cv_metric}')
            evaluator_kwargs = dict(X=X_arr, y=y_arr, folds=folds, penalty=self.penalty, cv_metric=self.cv_metric, scad_a=self.scad_a, mcp_gamma=self.mcp_gamma, nonconvex_init=self.nonconvex_init, nonconvex_multi_start=self.nonconvex_multi_start, nonconvex_init_ridge=self.nonconvex_init_ridge, lla_max_iter=self.lla_max_iter, lla_tol=self.lla_tol, lla_min_iter=self.lla_min_iter, lla_monotone=self.lla_monotone, lla_max_line_search=self.lla_max_line_search, coef_zero_threshold=self.coef_threshold, solver_kwargs=self._solver_kwargs())
            if self.n_jobs == 1 or Parallel is None:
                cv_results = [_evaluate_cv_candidate(candidate_id=cid, lam=lam, l1_ratio=ratio, **evaluator_kwargs) for (cid, lam, ratio) in candidates]
            else:
                cv_results = Parallel(n_jobs=self.n_jobs)((delayed(_evaluate_cv_candidate)(candidate_id=cid, lam=lam, l1_ratio=ratio, **evaluator_kwargs) for (cid, lam, ratio) in candidates))
            larger_better = _metric_is_larger_better(self.cv_metric)

            def selection_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
                score_key = -row['mean_cv_score'] if larger_better else row['mean_cv_score']
                if self.prefer_stronger_penalty_on_tie:
                    lambda_tie = -row['lambda']
                    ratio_tie = -row['l1_ratio']
                else:
                    lambda_tie = row['lambda']
                    ratio_tie = row['l1_ratio']
                return (float(score_key), float(lambda_tie), float(ratio_tie))
            cv_results.sort(key=selection_key)
            best_row = cv_results[0]
            best_lambda = float(best_row['lambda'])
            best_ratio = float(best_row['l1_ratio'])
            ranked_ids = {int(row['candidate_id']): rank + 1 for (rank, row) in enumerate(cv_results)}
            cv_results.sort(key=lambda row: int(row['candidate_id']))
            for row in cv_results:
                row['rank'] = int(ranked_ids[int(row['candidate_id'])])
                row['metric'] = self.cv_metric
            if self.verbose:
                print(f'    Best lambda={best_lambda:.8g}' + (f', l1_ratio={best_ratio:.4f}' if self.penalty == 'elastic_net' else '') + f", mean CV {self.cv_metric}={best_row['mean_cv_score']:.6f}")
        else:
            best_lambda = float(lambda_grid[0])
            best_ratio = float(ratio_grid[0])
            cv_results = []
            if self.verbose:
                print(f'[*] Penalized Logistic direct fit: penalty={self.penalty}, n={n}, p={p}, lambda={best_lambda:.8g}' + (f', l1_ratio={best_ratio:.4f}' if self.penalty == 'elastic_net' else ''))
        final_fit = _fit_candidate(X_arr, y_arr, penalty=self.penalty, lam=best_lambda, l1_ratio=best_ratio, scad_a=self.scad_a, mcp_gamma=self.mcp_gamma, nonconvex_init=self.nonconvex_init, nonconvex_multi_start=self.nonconvex_multi_start, nonconvex_init_ridge=self.nonconvex_init_ridge, lla_max_iter=self.lla_max_iter, lla_tol=self.lla_tol, lla_min_iter=self.lla_min_iter, lla_monotone=self.lla_monotone, lla_max_line_search=self.lla_max_line_search, coef_zero_threshold=self.coef_threshold, solver_kwargs=self._solver_kwargs(), store_history=True, verbose=self.verbose, log_every=self.log_every)
        self.X_mean_ = final_fit.mean.copy()
        self.X_scale_ = final_fit.scale.copy()
        self.zero_variance_mask_ = final_fit.zero_variance_mask.copy()
        if np.any(self.zero_variance_mask_):
            warnings.warn(f'Detected {int(np.sum(self.zero_variance_mask_))} near-zero-variance variables; their standardization scale was set to 1.', RuntimeWarning)
        beta = final_fit.beta_input_scale.copy()
        selected = np.flatnonzero(np.abs(beta) > self.coef_threshold)
        selected_names = None
        if self.feature_names_ is not None:
            selected_names = [self.feature_names_[j] for j in selected]
        opt = final_fit.optimization
        self.result_ = PenalizedLogisticResult(penalty=self.penalty, best_lambda=float(best_lambda), best_l1_ratio=float(best_ratio), intercept_input_scale=float(final_fit.intercept_input_scale), beta_input_scale=beta, selected_features=selected.astype(int), selected_feature_names=selected_names, converged=bool(opt.converged), n_iter=int(opt.n_iter), objective=float(opt.objective), data_loss=float(opt.data_loss), penalty_value=float(opt.penalty_value), stationarity_inf=float(opt.stationarity_inf), optimization_message=str(opt.message), cv_performed=perform_cv, cv_metric=self.cv_metric, cv_results=cv_results, optimization_history=list(opt.history))
        if self.verbose:
            print(f'[✓] Finished | converged={opt.converged} | iter={opt.n_iter} | obj={opt.objective:.8f} | stationarity={opt.stationarity_inf:.3e} | selected={len(selected)}/{p}')
        if not opt.converged:
            warnings.warn(f'The final optimizer did not reach the strict stationarity tolerance within the iteration limit; stationarity_inf={opt.stationarity_inf:.3e}. Increase max_iter/lla_max_iter or relax tol.', RuntimeWarning)
        return self

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise ValueError('The model is not fitted; call fit(X,y) first.')

    def _prepare_X_predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X_arr = _to_numpy_2d(X)
        if X_arr.shape[1] != len(self.result_.beta_input_scale):
            raise ValueError('The number of variables in the new X does not match the training data.')
        return X_arr

    def decision_function(self, X: Any) -> np.ndarray:
        Xs = self._prepare_X_predict(X)
        return self.result_.intercept_input_scale + Xs @ self.result_.beta_input_scale

    def predict_proba(self, X: Any) -> np.ndarray:
        p1 = expit(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any, threshold: float=0.5) -> np.ndarray:
        if not 0.0 < threshold < 1.0:
            raise ValueError('threshold must lie in (0,1).')
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def get_coefficients(self, original_scale: bool=False) -> np.ndarray:
        self._check_fitted()
        beta = self.result_.beta_input_scale.copy()
        if original_scale:
            beta = beta
        return beta

    def get_intercept(self, original_scale: bool=False) -> float:
        self._check_fitted()
        intercept = float(self.result_.intercept_input_scale)
        if not original_scale:
            return intercept
        beta_original = self.get_coefficients(original_scale=True)
        return float(intercept - np.dot(self.X_mean_, beta_original))

    def get_selected_features(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.selected_features.copy()

    def get_selected_feature_names(self) -> Optional[List[str]]:
        self._check_fitted()
        if self.result_.selected_feature_names is None:
            return None
        return list(self.result_.selected_feature_names)

    def get_best_penalty_strength(self) -> float:
        self._check_fitted()
        return float(self.result_.best_lambda)

    def get_best_l1_ratio(self) -> float:
        self._check_fitted()
        return float(self.result_.best_l1_ratio)

    def get_diagnostics(self) -> Dict[str, Any]:
        self._check_fitted()
        return {'penalty': self.result_.penalty, 'best_lambda': float(self.result_.best_lambda), 'best_l1_ratio': float(self.result_.best_l1_ratio), 'cv_performed': bool(self.result_.cv_performed), 'cv_metric': self.result_.cv_metric, 'converged': bool(self.result_.converged), 'n_iter': int(self.result_.n_iter), 'objective': float(self.result_.objective), 'data_loss': float(self.result_.data_loss), 'penalty_value': float(self.result_.penalty_value), 'stationarity_inf': float(self.result_.stationarity_inf), 'optimization_message': self.result_.optimization_message, 'n_selected': int(len(self.result_.selected_features)), 'coef_threshold': float(self.coef_threshold), 'n_zero_variance_features': int(np.sum(self.zero_variance_mask_)), 'scad_a': float(self.scad_a), 'mcp_gamma': float(self.mcp_gamma)}

    def coefficient_summary(self, sort_by_abs: bool=True):
        self._check_fitted()
        beta_std = self.result_.beta_input_scale
        beta_orig = self.get_coefficients(original_scale=True)
        selected_set = set(self.result_.selected_features.tolist())
        rows: List[Dict[str, Any]] = []
        for j in range(len(beta_std)):
            rows.append({'feature_index': int(j), 'feature_name': None if self.feature_names_ is None else self.feature_names_[j], 'coefficient_input_scale': float(beta_std[j]), 'coefficient_original_scale': float(beta_orig[j]), 'abs_coefficient_input_scale': float(abs(beta_std[j])), 'selected': bool(j in selected_set)})
        if sort_by_abs:
            rows.sort(key=lambda row: row['abs_coefficient_input_scale'], reverse=True)
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def cv_results_table(self):
        self._check_fitted()
        rows = [dict(row) for row in self.result_.cv_results]
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def history_table(self):
        self._check_fitted()
        rows = [dict(row) for row in self.result_.optimization_history]
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

def Penalized_Logistic(X: Any, y: Any, *, penalty: PenaltyName='lasso', penalty_strength: Union[float, Sequence[float], np.ndarray]=0.01, l1_ratio: Union[float, Sequence[float], np.ndarray]=0.5, feature_names: Optional[Sequence[str]]=None, scad_a: float=3.7, mcp_gamma: float=3.0, nonconvex_init: NonconvexInit='lasso', nonconvex_multi_start: bool=False, nonconvex_init_ridge: float=0.001, lla_max_iter: int=50, lla_tol: float=1e-06, lla_min_iter: int=2, lla_monotone: bool=True, lla_max_line_search: int=30, cv_folds: int=5, cv_metric: CVMetric='log_loss', cv_shuffle: bool=True, n_jobs: int=1, prefer_stronger_penalty_on_tie: bool=True, max_iter: int=5000, tol: float=1e-07, min_iter: int=5, initial_step: Optional[float]=None, backtracking_factor: float=0.5, max_backtracking: int=50, step_growth: float=1.2, monotone_restart: bool=True, coef_threshold: float=1e-08, random_state: int=2026, verbose: bool=True, log_every: int=100) -> PenalizedLogistic:
    model = PenalizedLogistic(penalty=penalty, penalty_strength=penalty_strength, l1_ratio=l1_ratio, scad_a=scad_a, mcp_gamma=mcp_gamma, nonconvex_init=nonconvex_init, nonconvex_multi_start=nonconvex_multi_start, nonconvex_init_ridge=nonconvex_init_ridge, lla_max_iter=lla_max_iter, lla_tol=lla_tol, lla_min_iter=lla_min_iter, lla_monotone=lla_monotone, lla_max_line_search=lla_max_line_search, cv_folds=cv_folds, cv_metric=cv_metric, cv_shuffle=cv_shuffle, n_jobs=n_jobs, prefer_stronger_penalty_on_tie=prefer_stronger_penalty_on_tie, max_iter=max_iter, tol=tol, min_iter=min_iter, initial_step=initial_step, backtracking_factor=backtracking_factor, max_backtracking=max_backtracking, step_growth=step_growth, monotone_restart=monotone_restart, coef_threshold=coef_threshold, random_state=random_state, verbose=verbose, log_every=log_every)
    return model.fit(X, y, feature_names=feature_names)
fit_penalized_logistic = Penalized_Logistic

def check_logistic_gradient(X: np.ndarray, y: np.ndarray, *, l2_strength: float=0.3, random_state: int=2026, finite_diff_eps: float=1e-06) -> Dict[str, float]:
    X = _to_numpy_2d(X)
    y = _to_binary_y(y, X.shape[0])
    if l2_strength < 0:
        raise ValueError('l2_strength must be >= 0.')
    if finite_diff_eps <= 0:
        raise ValueError('finite_diff_eps must be > 0.')
    rng = np.random.default_rng(int(random_state))
    intercept = float(rng.normal(scale=0.3))
    beta = rng.normal(scale=0.3, size=X.shape[1])
    (_, g0, gb) = _smooth_loss_grad(X, y, intercept, beta, l2_strength)
    analytic = np.concatenate(([g0], gb))
    theta = np.concatenate(([intercept], beta))

    def objective(th: np.ndarray) -> float:
        b0 = float(th[0])
        b = th[1:]
        (value, _, _) = _smooth_loss_grad(X, y, b0, b, l2_strength)
        return float(value)
    numerical = np.zeros_like(theta)
    h = float(finite_diff_eps)
    for j in range(len(theta)):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += h
        minus[j] -= h
        numerical[j] = (objective(plus) - objective(minus)) / (2.0 * h)
    abs_error = float(np.max(np.abs(analytic - numerical)))
    rel_error = float(abs_error / (np.max(np.abs(numerical)) + 1e-12))
    return {'max_abs_error': abs_error, 'max_relative_error': rel_error}
__all__ = ['PenalizedLogisticResult', 'PenalizedLogistic', 'Penalized_Logistic', 'fit_penalized_logistic', 'check_logistic_gradient']
