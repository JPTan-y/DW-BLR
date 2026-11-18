from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings
import numpy as np
ArrayLike = Any

def _to_numpy_2d(X: ArrayLike, name: str='X') -> np.ndarray:
    if hasattr(X, 'to_numpy'):
        arr = np.asarray(X.to_numpy(), dtype=np.float64)
    else:
        arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f'{name} must be a two-dimensional matrix; got shape={arr.shape}.')
    if arr.shape[0] < 2:
        raise ValueError(f'{name} requires at least 2 samples.')
    if arr.shape[1] < 2:
        raise ValueError(f'DWR decorrelation requires at least 2 variables; got p={arr.shape[1]}.')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} contains NaN or Inf.')
    return np.ascontiguousarray(arr, dtype=np.float64)

def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64).ravel()
    denominator = float(np.dot(w, w))
    if denominator <= 0.0:
        return 0.0
    return float(w.sum() ** 2 / denominator)

def project_to_simplex(v: np.ndarray, target_sum: float) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError('The vector to project must not be empty.')
    if not np.all(np.isfinite(x)):
        raise ValueError('The vector to project contains NaN or Inf.')
    if not np.isfinite(target_sum) or target_sum <= 0.0:
        raise ValueError('target_sum must be a finite positive number.')
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - float(target_sum)
    indices = np.arange(1, x.size + 1, dtype=np.float64)
    condition = u - cssv / indices > 0.0
    if not np.any(condition):
        result = np.zeros_like(x)
        result[int(np.argmax(x))] = float(target_sum)
        return result
    rho = int(np.flatnonzero(condition)[-1])
    theta = cssv[rho] / float(rho + 1)
    result = np.maximum(x - theta, 0.0)
    residual = float(target_sum) - float(result.sum())
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, float(target_sum))
    if abs(residual) > tolerance:
        positive = np.flatnonzero(result > 0.0)
        if positive.size == 0:
            result[int(np.argmax(x))] = float(target_sum)
        else:
            index = int(positive[np.argmax(result[positive])])
            result[index] += residual
            if result[index] < 0.0:
                return project_to_simplex(result, target_sum)
    result[result < 0.0] = 0.0
    return result

def weighted_covariance(X: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_arr = _to_numpy_2d(X)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != (X_arr.shape[0],):
        raise ValueError('Length of weights must equal the sample size n.')
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError('weights must be finite and nonnegative.')
    weight_sum = float(w.sum())
    if weight_sum <= 0.0:
        raise ValueError('weights must have a positive sum.')
    q = w / weight_sum
    mean = q @ X_arr
    centered = X_arr - mean
    covariance = (centered * q[:, None]).T @ centered
    covariance = 0.5 * (covariance + covariance.T)
    return (mean, covariance)

@dataclass
class DWRObjectiveEvaluation:
    total: float
    decorrelation: float
    regularization: float
    gradient: Optional[np.ndarray]
    weighted_mean: np.ndarray
    weighted_covariance: np.ndarray

def evaluate_dwr_objective(X: np.ndarray, weights: np.ndarray, eta: float, *, compute_gradient: bool=True) -> DWRObjectiveEvaluation:
    X_arr = _to_numpy_2d(X)
    w = np.asarray(weights, dtype=np.float64).ravel()
    (n, p) = X_arr.shape
    if w.shape != (n,):
        raise ValueError('Length of weights must equal the sample size n.')
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError('weights must be finite and nonnegative.')
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError('eta must be a finite nonnegative number.')
    weight_sum = float(w.sum())
    if weight_sum <= 0.0:
        raise ValueError('weights must have a positive sum.')
    q = w / weight_sum
    mean = q @ X_arr
    centered = X_arr - mean
    covariance = (centered * q[:, None]).T @ centered
    covariance = 0.5 * (covariance + covariance.T)
    covariance_off = covariance.copy()
    np.fill_diagonal(covariance_off, 0.0)
    pair_count = float(p * (p - 1))
    offdiag_squared_norm = float(np.sum(covariance_off * covariance_off))
    decorrelation_loss = offdiag_squared_norm / pair_count
    regularization_loss = float(eta) * float(np.dot(w - 1.0, w - 1.0)) / float(n)
    total_loss = decorrelation_loss + regularization_loss
    gradient: Optional[np.ndarray]
    if compute_gradient:
        centered_times_cov = centered @ covariance_off
        quadratic_terms = np.sum(centered_times_cov * centered, axis=1)
        gradient_decorrelation = 2.0 / (pair_count * weight_sum) * (quadratic_terms - offdiag_squared_norm)
        gradient_regularization = 2.0 * float(eta) * (w - 1.0) / float(n)
        gradient = gradient_decorrelation + gradient_regularization
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError('DWR analytic gradient is NaN or Inf.')
    else:
        gradient = None
    if not np.isfinite(total_loss):
        raise FloatingPointError('DWR objective is NaN or Inf.')
    return DWRObjectiveEvaluation(total=float(total_loss), decorrelation=float(decorrelation_loss), regularization=float(regularization_loss), gradient=gradient, weighted_mean=mean, weighted_covariance=covariance)

@dataclass
class DWROptimizationResult:
    weights: np.ndarray
    objective: float
    decorrelation_loss: float
    regularization_loss: float
    projected_gradient_inf: float
    iterations: int
    converged: bool
    final_step_size: float
    history: List[Dict[str, float]] = field(default_factory=list)

class DWRWeightOptimizer:

    @staticmethod
    def _backtracking_step(X: np.ndarray, point: np.ndarray, evaluation: DWRObjectiveEvaluation, eta: float, target_sum: float, initial_step: float, backtracking_factor: float, max_backtracking: int) -> Tuple[np.ndarray, DWRObjectiveEvaluation, float, int]:
        if evaluation.gradient is None:
            raise RuntimeError('Internal error: backtracking line search requires a gradient.')
        step = float(initial_step)
        gradient = evaluation.gradient
        reference_loss = float(evaluation.total)
        for number in range(int(max_backtracking) + 1):
            candidate = project_to_simplex(point - step * gradient, target_sum)
            difference = candidate - point
            candidate_evaluation = evaluate_dwr_objective(X, candidate, eta, compute_gradient=False)
            quadratic_upper_bound = reference_loss + float(np.dot(gradient, difference)) + 0.5 / step * float(np.dot(difference, difference))
            numerical_slack = 1e-12 * max(1.0, abs(reference_loss))
            if candidate_evaluation.total <= quadratic_upper_bound + numerical_slack:
                return (candidate, candidate_evaluation, step, number)
            step *= float(backtracking_factor)
            if step <= np.finfo(np.float64).tiny:
                break
        raise RuntimeError('DWR backtracking line search found no step satisfying the descent bound; check the input scale or increase max_backtracking.')

    @classmethod
    def optimize(cls, X: np.ndarray, initial_weights: np.ndarray, *, eta: float, max_iter: int, tol: float, min_iter: int, initial_step: float, backtracking_factor: float, max_backtracking: int, step_growth: float, verbose: bool, log_every: int) -> DWROptimizationResult:
        X_arr = _to_numpy_2d(X)
        (n, _) = X_arr.shape
        if eta < 0.0 or not np.isfinite(eta):
            raise ValueError('eta must be a finite nonnegative number.')
        if max_iter <= 0 or min_iter < 0 or min_iter > max_iter:
            raise ValueError('max_iter must be positive with 0 <= min_iter <= max_iter.')
        if tol <= 0.0 or not np.isfinite(tol):
            raise ValueError('tol must be a finite positive number.')
        if initial_step <= 0.0 or not np.isfinite(initial_step):
            raise ValueError('initial_step must be a finite positive number.')
        if not 0.0 < backtracking_factor < 1.0:
            raise ValueError('backtracking_factor must lie in (0,1).')
        if max_backtracking < 1:
            raise ValueError('max_backtracking must be >= 1.')
        if step_growth < 1.0 or not np.isfinite(step_growth):
            raise ValueError('step_growth must be finite and >= 1.')
        if log_every <= 0:
            raise ValueError('log_every must be a positive integer.')
        w0 = np.asarray(initial_weights, dtype=np.float64).ravel()
        if w0.shape != (n,):
            raise ValueError('Length of initial_weights must equal the sample size n.')
        if not np.all(np.isfinite(w0)):
            raise ValueError('initial_weights contains NaN or Inf.')
        target_sum = float(n)
        weights = project_to_simplex(w0, target_sum)
        extrapolated = weights.copy()
        momentum_parameter = 1.0
        step_size = float(initial_step)
        current_evaluation = evaluate_dwr_objective(X_arr, weights, eta, compute_gradient=True)
        history: List[Dict[str, float]] = []
        converged = False
        final_projected_gradient_inf = np.inf
        used_iteration = 0
        for iteration in range(1, int(max_iter) + 1):
            used_iteration = iteration
            extrapolated_evaluation = evaluate_dwr_objective(X_arr, extrapolated, eta, compute_gradient=True)
            (candidate, candidate_eval_no_grad, accepted_step, n_backtracking) = cls._backtracking_step(X_arr, extrapolated, extrapolated_evaluation, eta, target_sum, step_size, backtracking_factor, max_backtracking)
            restarted = False
            monotone_slack = 1e-12 * max(1.0, abs(current_evaluation.total))
            if candidate_eval_no_grad.total > current_evaluation.total + monotone_slack:
                restarted = True
                momentum_parameter = 1.0
                extrapolated = weights.copy()
                (candidate, candidate_eval_no_grad, accepted_step, extra_backtracking) = cls._backtracking_step(X_arr, extrapolated, current_evaluation, eta, target_sum, min(step_size, accepted_step), backtracking_factor, max_backtracking)
                n_backtracking += extra_backtracking
            candidate_evaluation = evaluate_dwr_objective(X_arr, candidate, eta, compute_gradient=True)
            if candidate_evaluation.gradient is None:
                raise RuntimeError('Internal error: missing candidate gradient.')
            projected_once = project_to_simplex(candidate - accepted_step * candidate_evaluation.gradient, target_sum)
            gradient_mapping = (candidate - projected_once) / accepted_step
            final_projected_gradient_inf = float(np.max(np.abs(gradient_mapping)))
            relative_weight_change = float(np.linalg.norm(candidate - weights) / max(1.0, np.linalg.norm(weights)))
            relative_objective_change = float(abs(current_evaluation.total - candidate_evaluation.total) / max(1.0, abs(current_evaluation.total)))
            history.append({'iteration': float(iteration), 'objective': float(candidate_evaluation.total), 'decorrelation_loss': float(candidate_evaluation.decorrelation), 'regularization_loss': float(candidate_evaluation.regularization), 'step_size': float(accepted_step), 'backtracking_steps': float(n_backtracking), 'relative_weight_change': relative_weight_change, 'relative_objective_change': relative_objective_change, 'projected_gradient_inf': final_projected_gradient_inf, 'restarted': float(restarted), 'ess': _effective_sample_size(candidate)})
            if verbose and (iteration == 1 or iteration % int(log_every) == 0):
                print(f'  DWR iter={iteration:4d} | obj={candidate_evaluation.total:.8e} | decor={candidate_evaluation.decorrelation:.8e} | reg={candidate_evaluation.regularization:.8e} | pg_inf={final_projected_gradient_inf:.3e} | ESS={_effective_sample_size(candidate):.1f}/{n}')
            old_weights = weights
            weights = candidate
            current_evaluation = candidate_evaluation
            if iteration >= int(min_iter) and final_projected_gradient_inf <= float(tol):
                converged = True
                break
            if restarted:
                extrapolated = weights.copy()
                momentum_parameter = 1.0
            else:
                new_momentum_parameter = (1.0 + np.sqrt(1.0 + 4.0 * momentum_parameter ** 2)) / 2.0
                momentum = (momentum_parameter - 1.0) / new_momentum_parameter
                extrapolated = weights + momentum * (weights - old_weights)
                extrapolated = project_to_simplex(extrapolated, target_sum)
                momentum_parameter = new_momentum_parameter
            step_size = min(float(initial_step), float(accepted_step) * float(step_growth))
        weights = project_to_simplex(weights, target_sum)
        final_evaluation = evaluate_dwr_objective(X_arr, weights, eta, compute_gradient=True)
        if final_evaluation.gradient is None:
            raise RuntimeError('Internal error: missing final gradient.')
        projected_once = project_to_simplex(weights - step_size * final_evaluation.gradient, target_sum)
        final_projected_gradient_inf = float(np.max(np.abs((weights - projected_once) / step_size)))
        converged = bool(converged or final_projected_gradient_inf <= float(tol))
        return DWROptimizationResult(weights=weights, objective=float(final_evaluation.total), decorrelation_loss=float(final_evaluation.decorrelation), regularization_loss=float(final_evaluation.regularization), projected_gradient_inf=final_projected_gradient_inf, iterations=int(used_iteration), converged=converged, final_step_size=float(step_size), history=history)

class DWR:

    def __init__(self, *, eta: float=1.0, max_iter: int=1000, tol: float=1e-07, min_iter: int=5, initial_step: float=1.0, backtracking_factor: float=0.5, max_backtracking: int=50, step_growth: float=1.2, verbose: bool=True, log_every: int=50) -> None:
        self.eta = float(eta)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_iter = int(min_iter)
        self.initial_step = float(initial_step)
        self.backtracking_factor = float(backtracking_factor)
        self.max_backtracking = int(max_backtracking)
        self.step_growth = float(step_growth)
        self.verbose = bool(verbose)
        self.log_every = int(log_every)
        if self.eta < 0.0 or not np.isfinite(self.eta):
            raise ValueError('eta must be a finite nonnegative number.')
        self.weights_: Optional[np.ndarray] = None
        self.optimization_result_: Optional[DWROptimizationResult] = None
        self.X_mean_: Optional[np.ndarray] = None
        self.X_scale_: Optional[np.ndarray] = None
        self.zero_variance_mask_: Optional[np.ndarray] = None
        self.weighted_mean_: Optional[np.ndarray] = None
        self.weighted_covariance_: Optional[np.ndarray] = None
        self.unweighted_covariance_: Optional[np.ndarray] = None
        self.initial_objective_: Optional[float] = None
        self.initial_decorrelation_loss_: Optional[float] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: ArrayLike, initial_weights: Optional[Sequence[float]]=None) -> 'DWR':
        if hasattr(X, 'columns') and hasattr(X, 'to_numpy'):
            self.feature_names_ = [str(value) for value in list(X.columns)]
        else:
            self.feature_names_ = None
        X_arr = _to_numpy_2d(X)
        (n, p) = X_arr.shape
        mean = np.zeros(p, dtype=np.float64)
        scale = np.ones(p, dtype=np.float64)
        zero_mask = np.zeros(p, dtype=bool)
        X_used = X_arr.copy()
        self.X_mean_ = mean
        self.X_scale_ = scale
        self.zero_variance_mask_ = zero_mask
        if initial_weights is None:
            w0 = np.ones(n, dtype=np.float64)
        else:
            w0 = np.asarray(initial_weights, dtype=np.float64).ravel()
            if w0.shape != (n,):
                raise ValueError('Length of initial_weights must equal the sample size n.')
            if not np.all(np.isfinite(w0)):
                raise ValueError('initial_weights contains NaN or Inf.')
        initial_feasible = project_to_simplex(w0, float(n))
        initial_eval = evaluate_dwr_objective(X_used, initial_feasible, self.eta, compute_gradient=False)
        self.initial_objective_ = float(initial_eval.total)
        self.initial_decorrelation_loss_ = float(initial_eval.decorrelation)
        self.unweighted_covariance_ = initial_eval.weighted_covariance.copy()
        if self.verbose:
            print(f'[*] DWR Start: n={n}, p={p}, eta={self.eta}, standardize={False}')
            print(f'  Initial objective={initial_eval.total:.8e} | decorrelation={initial_eval.decorrelation:.8e}')
        result = DWRWeightOptimizer.optimize(X_used, initial_feasible, eta=self.eta, max_iter=self.max_iter, tol=self.tol, min_iter=self.min_iter, initial_step=self.initial_step, backtracking_factor=self.backtracking_factor, max_backtracking=self.max_backtracking, step_growth=self.step_growth, verbose=self.verbose, log_every=self.log_every)
        weights = result.weights.copy()
        if np.any(weights < -1e-12):
            raise FloatingPointError('DWR returned negative weights.')
        if not np.isclose(weights.sum(), n, rtol=0.0, atol=1e-08 * max(1, n)):
            raise FloatingPointError('Final DWR weights do not sum to n.')
        final_eval = evaluate_dwr_objective(X_used, weights, self.eta, compute_gradient=False)
        self.weights_ = weights
        self.optimization_result_ = result
        self.weighted_mean_ = final_eval.weighted_mean.copy()
        self.weighted_covariance_ = final_eval.weighted_covariance.copy()
        if self.verbose:
            print(f'[✓] DWR finished | converged={result.converged} | iterations={result.iterations} | objective={result.objective:.8e} | ESS={_effective_sample_size(weights):.1f}/{n} | W min/mean/max={weights.min():.4f}/{weights.mean():.4f}/{weights.max():.4f}')
        return self

    def _check_fitted(self) -> None:
        if self.weights_ is None or self.optimization_result_ is None:
            raise ValueError('DWR is not fitted; call fit(X) first.')

    def get_weights(self) -> np.ndarray:
        self._check_fitted()
        return self.weights_.copy()

    def get_loss(self) -> float:
        self._check_fitted()
        return float(self.optimization_result_.objective)

    def get_decorrelation_loss(self) -> float:
        self._check_fitted()
        return float(self.optimization_result_.decorrelation_loss)

    def get_regularization_loss(self) -> float:
        self._check_fitted()
        return float(self.optimization_result_.regularization_loss)

    def get_weighted_covariance(self) -> np.ndarray:
        self._check_fitted()
        return self.weighted_covariance_.copy()

    def get_unweighted_covariance(self) -> np.ndarray:
        self._check_fitted()
        return self.unweighted_covariance_.copy()

    def get_effective_sample_size(self) -> float:
        self._check_fitted()
        return _effective_sample_size(self.weights_)

    def get_diagnostics(self) -> Dict[str, float]:
        self._check_fitted()
        result = self.optimization_result_
        n = len(self.weights_)
        return {'n_samples': float(n), 'n_features': float(self.weighted_covariance_.shape[0]), 'eta': float(self.eta), 'initial_objective': float(self.initial_objective_), 'initial_decorrelation_loss': float(self.initial_decorrelation_loss_), 'final_objective': float(result.objective), 'final_decorrelation_loss': float(result.decorrelation_loss), 'final_regularization_loss': float(result.regularization_loss), 'projected_gradient_inf': float(result.projected_gradient_inf), 'iterations': float(result.iterations), 'converged': float(result.converged), 'final_step_size': float(result.final_step_size), 'weight_min': float(np.min(self.weights_)), 'weight_mean': float(np.mean(self.weights_)), 'weight_max': float(np.max(self.weights_)), 'weight_sum': float(np.sum(self.weights_)), 'ess': float(_effective_sample_size(self.weights_)), 'ess_ratio': float(_effective_sample_size(self.weights_) / n)}

    def history_table(self):
        self._check_fitted()
        history = list(self.optimization_result_.history)
        try:
            import pandas as pd
            return pd.DataFrame(history)
        except ImportError:
            return history

def fit_dwr(X: ArrayLike, **kwargs: Any) -> DWR:
    model = DWR(**kwargs)
    return model.fit(X)

def check_dwr_gradient(X: Optional[ArrayLike]=None, *, eta: float=0.7, random_state: int=2026, finite_difference_step: float=1e-06) -> Dict[str, float]:
    if finite_difference_step <= 0.0:
        raise ValueError('finite_difference_step must be positive.')
    rng = np.random.default_rng(int(random_state))
    if X is None:
        X_arr = rng.normal(size=(11, 4))
        X_arr[:, 1] = 0.55 * X_arr[:, 0] + 0.8 * X_arr[:, 1]
    else:
        X_arr = _to_numpy_2d(X)
    n = X_arr.shape[0]
    weights = rng.uniform(0.6, 1.4, size=n)
    weights = weights / weights.mean()
    evaluation = evaluate_dwr_objective(X_arr, weights, eta, compute_gradient=True)
    if evaluation.gradient is None:
        raise RuntimeError('Internal error: missing analytic gradient.')
    numerical = np.empty(n, dtype=np.float64)
    for index in range(n):
        h = float(finite_difference_step) * max(1.0, abs(weights[index]))
        if weights[index] <= h:
            h = 0.25 * weights[index]
        plus = weights.copy()
        minus = weights.copy()
        plus[index] += h
        minus[index] -= h
        loss_plus = evaluate_dwr_objective(X_arr, plus, eta, compute_gradient=False).total
        loss_minus = evaluate_dwr_objective(X_arr, minus, eta, compute_gradient=False).total
        numerical[index] = (loss_plus - loss_minus) / (2.0 * h)
    absolute_error = np.abs(evaluation.gradient - numerical)
    denominator = np.maximum(1.0, np.maximum(np.abs(evaluation.gradient), np.abs(numerical)))
    relative_error = absolute_error / denominator
    return {'max_absolute_error': float(np.max(absolute_error)), 'max_relative_error': float(np.max(relative_error)), 'mean_absolute_error': float(np.mean(absolute_error)), 'analytic_gradient_inf': float(np.max(np.abs(evaluation.gradient))), 'numerical_gradient_inf': float(np.max(np.abs(numerical)))}
__all__ = ['DWRObjectiveEvaluation', 'DWROptimizationResult', 'DWRWeightOptimizer', 'DWR', 'project_to_simplex', 'weighted_covariance', 'evaluate_dwr_objective', 'fit_dwr', 'check_dwr_gradient']
