from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import warnings
import numpy as np
from scipy.special import expit

@dataclass
class PGVariationalParams:
    alpha: np.ndarray
    mu: np.ndarray
    s2: np.ndarray

    def copy(self) -> 'PGVariationalParams':
        return PGVariationalParams(alpha=self.alpha.copy(), mu=self.mu.copy(), s2=self.s2.copy())

    @property
    def posterior_mean(self) -> np.ndarray:
        return self.alpha * self.mu

    @property
    def second_moment(self) -> np.ndarray:
        return self.alpha * (self.mu ** 2 + self.s2)

    @property
    def posterior_variance(self) -> np.ndarray:
        m = self.posterior_mean
        return np.maximum(self.second_moment - m ** 2, 0.0)

@dataclass
class InterceptVariationalParams:
    mu_b: float = 0.0
    v_b: float = 1.0

    def copy(self) -> 'InterceptVariationalParams':
        return InterceptVariationalParams(mu_b=float(self.mu_b), v_b=float(self.v_b))

    @property
    def sd_b(self) -> float:
        return float(np.sqrt(max(self.v_b, 0.0)))

@dataclass
class DWBLRHyperParams:
    pi0: float = 0.5
    sigma0_sq: float = 1.0

    def copy(self) -> 'DWBLRHyperParams':
        return DWBLRHyperParams(pi0=float(self.pi0), sigma0_sq=float(self.sigma0_sq))

@dataclass
class DWBLRResult:
    selected_features: np.ndarray
    selected_feature_names: Optional[List[str]]
    weights: np.ndarray
    alpha: np.ndarray
    mu: np.ndarray
    s2: np.ndarray
    intercept_mu: float
    intercept_var: float
    sigma_b_sq: float
    rho_bar: np.ndarray
    eta_bar: np.ndarray
    Lambda: np.ndarray
    hyper: DWBLRHyperParams
    feature_prior: np.ndarray
    feature_prior_update: str
    n_outer_iterations: int
    history: List[Dict[str, float]] = field(default_factory=list)

    @property
    def posterior_mean_beta(self) -> np.ndarray:
        return self.alpha * self.mu

    @property
    def posterior_variance_beta(self) -> np.ndarray:
        second = self.alpha * (self.mu ** 2 + self.s2)
        mean = self.alpha * self.mu
        return np.maximum(second - mean ** 2, 0.0)

    @property
    def intercept_sd(self) -> float:
        return float(np.sqrt(max(self.intercept_var, 0.0)))

def _project_to_simplex(v: np.ndarray, target_sum: float) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    if n == 0:
        return v.copy()
    if target_sum <= 0:
        return np.zeros_like(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    j = np.arange(1, n + 1, dtype=float)
    cond = u - (cssv - target_sum) / j > 0
    idx = np.where(cond)[0]
    if idx.size == 0:
        theta = (cssv[-1] - target_sum) / n
    else:
        rho = idx[-1]
        theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s > 0 and abs(s - target_sum) > 1e-10 * max(1.0, target_sum):
        w *= target_sum / s
    return w

def _effective_sample_size(w: np.ndarray, eps: float=1e-15) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (np.sum(w ** 2) + eps))

def _make_nu_schedule(T_max: int, nu_start: float, nu_end: float, schedule: Literal['geometric', 'linear', 'constant']='geometric') -> np.ndarray:
    if T_max <= 0:
        raise ValueError('T_max must be a positive integer.')
    if nu_start <= 0 or nu_end <= 0:
        raise ValueError('nu_start and nu_end must both be > 0.')
    if schedule != 'constant' and nu_start < nu_end:
        warnings.warn('nu^(t) should decrease monotonically, but nu_start < nu_end; the two values were swapped.', RuntimeWarning)
        (nu_start, nu_end) = (nu_end, nu_start)
    if T_max == 1:
        return np.array([float(nu_end)])
    if schedule == 'geometric':
        return np.geomspace(nu_start, nu_end, T_max)
    if schedule == 'linear':
        return np.linspace(nu_start, nu_end, T_max)
    if schedule == 'constant':
        return np.full(T_max, float(nu_start))
    raise ValueError("schedule must be 'geometric', 'linear' or 'constant'.")

class CurvatureDecorrelator:

    @staticmethod
    def _covariance_and_center(X: np.ndarray, W: np.ndarray, rho_bar: np.ndarray, covariance_mode: Literal['precision', 'normalized'], eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        (n, p) = X.shape
        if n == 0 or p == 0:
            raise ValueError('X must contain at least one sample and one feature.')
        v = np.maximum(W, 0.0) * np.maximum(rho_bar, 0.0)
        v_sum = float(np.sum(v))
        if v_sum <= eps:
            mu = np.mean(X, axis=0)
            Xc = X - mu
            return (np.zeros((p, p), dtype=float), Xc, v, max(v_sum, eps))
        mu = v @ X / v_sum
        Xc = X - mu
        if covariance_mode == 'normalized':
            q = v / v_sum
            C = (Xc * q[:, None]).T @ Xc
        elif covariance_mode == 'precision':
            C = (Xc * v[:, None]).T @ Xc / float(n)
        else:
            raise ValueError("covariance_mode must be 'normalized' or 'precision'.")
        C = 0.5 * (C + C.T)
        return (C, Xc, v, v_sum)

    @staticmethod
    def loss_and_grad(X: np.ndarray, Lambda: np.ndarray, W: np.ndarray, rho_bar: np.ndarray, gamma_w: float=1.0, decorrelation_scale: float=1.0, covariance_mode: Literal['precision', 'normalized']='normalized', eps: float=1e-12) -> Tuple[float, np.ndarray]:
        X = np.asarray(X, dtype=float)
        W = np.asarray(W, dtype=float).ravel()
        rho_bar = np.asarray(rho_bar, dtype=float).ravel()
        Lambda = np.asarray(Lambda, dtype=float)
        if X.ndim != 2:
            raise ValueError('X must be a two-dimensional matrix.')
        (n, p) = X.shape
        if n == 0 or p == 0:
            raise ValueError('X must contain at least one sample and one feature.')
        if W.shape != (n,):
            raise ValueError('Length of W must equal the sample size n.')
        if rho_bar.shape != (n,):
            raise ValueError('Length of rho_bar must equal the sample size n.')
        if Lambda.shape != (p, p):
            raise ValueError('Lambda must have shape (p,p).')
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(W)):
            raise ValueError('X or W contains NaN/Inf.')
        if not np.all(np.isfinite(rho_bar)) or not np.all(np.isfinite(Lambda)):
            raise ValueError('rho_bar or Lambda contains NaN/Inf.')
        if gamma_w < 0:
            raise ValueError('gamma_w must be >= 0.')
        if decorrelation_scale < 0:
            raise ValueError('decorrelation_scale must be >= 0.')
        (C, Xc, _, v_sum) = CurvatureDecorrelator._covariance_and_center(X=X, W=W, rho_bar=rho_bar, covariance_mode=covariance_mode, eps=eps)
        Lambda_off = Lambda.copy()
        np.fill_diagonal(Lambda_off, 0.0)
        Lambda2 = Lambda_off ** 2
        pair_count = p * (p - 1)
        pair_scale = 0.0 if pair_count == 0 else decorrelation_scale / float(pair_count)
        raw_pair_loss = float(np.sum(Lambda2 * C ** 2))
        loss_dec = pair_scale * raw_pair_loss
        loss_reg = gamma_w / float(n) * float(np.sum((W - 1.0) ** 2))
        loss = loss_dec + loss_reg
        if pair_count == 0 or decorrelation_scale == 0.0:
            grad_dec = np.zeros(n, dtype=float)
        else:
            G = Lambda2 * C
            np.fill_diagonal(G, 0.0)
            XG = Xc @ G
            quad = np.sum(XG * Xc, axis=1)
            if covariance_mode == 'normalized':
                gc = float(np.sum(G * C))
                grad_dec = 2.0 * pair_scale * (rho_bar / max(v_sum, eps)) * (quad - gc)
            else:
                grad_dec = 2.0 * pair_scale * (rho_bar / float(n)) * quad
        grad_reg = 2.0 * gamma_w / float(n) * (W - 1.0)
        grad = grad_dec + grad_reg
        return (float(loss), grad)

    @staticmethod
    def loss(X: np.ndarray, Lambda: np.ndarray, W: np.ndarray, rho_bar: np.ndarray, gamma_w: float=1.0, decorrelation_scale: float=1.0, covariance_mode: Literal['precision', 'normalized']='normalized', eps: float=1e-12) -> float:
        (f, _) = CurvatureDecorrelator.loss_and_grad(X=X, Lambda=Lambda, W=W, rho_bar=rho_bar, gamma_w=gamma_w, decorrelation_scale=decorrelation_scale, covariance_mode=covariance_mode, eps=eps)
        return f

    @staticmethod
    def loss_components(X: np.ndarray, Lambda: np.ndarray, W: np.ndarray, rho_bar: np.ndarray, gamma_w: float=1.0, decorrelation_scale: float=1.0, covariance_mode: Literal['precision', 'normalized']='normalized', eps: float=1e-12) -> Dict[str, float]:
        X = np.asarray(X, dtype=float)
        (n, p) = X.shape
        (C, _, _, _) = CurvatureDecorrelator._covariance_and_center(X, np.asarray(W, dtype=float), np.asarray(rho_bar, dtype=float), covariance_mode, eps)
        L = np.asarray(Lambda, dtype=float).copy()
        np.fill_diagonal(L, 0.0)
        pair_count = p * (p - 1)
        loss_dec = 0.0
        if pair_count > 0:
            loss_dec = decorrelation_scale / float(pair_count) * float(np.sum(L ** 2 * C ** 2))
        loss_reg = gamma_w / float(n) * float(np.sum((np.asarray(W) - 1.0) ** 2))
        return {'decorrelation_loss': float(loss_dec), 'weight_regularization_loss': float(loss_reg), 'total_loss': float(loss_dec + loss_reg)}

    @staticmethod
    def fista(X: np.ndarray, Lambda: np.ndarray, rho_bar: np.ndarray, W_init: Optional[np.ndarray]=None, gamma_w: float=1.0, decorrelation_scale: float=1.0, covariance_mode: Literal['precision', 'normalized']='normalized', max_iter: int=500, tol: float=1e-06, initial_lr: float=1.0, backtrack_factor: float=0.5, max_backtracking: int=30, monotone_restart: bool=True, eps: float=1e-12) -> Tuple[np.ndarray, Dict[str, float]]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError('X must be a two-dimensional matrix.')
        (n, _) = X.shape
        if n == 0:
            raise ValueError('X must contain at least one sample.')
        if gamma_w < 0:
            raise ValueError('gamma_w must be >= 0.')
        if decorrelation_scale < 0:
            raise ValueError('decorrelation_scale must be >= 0.')
        if max_iter <= 0:
            raise ValueError('max_iter must be > 0.')
        if tol <= 0 or initial_lr <= 0:
            raise ValueError('tol and initial_lr must be > 0.')
        if not 0.0 < backtrack_factor < 1.0:
            raise ValueError('backtrack_factor must lie in (0,1).')
        if max_backtracking <= 0:
            raise ValueError('max_backtracking must be > 0.')
        if W_init is None:
            W = np.ones(n, dtype=float)
        else:
            W_init_arr = np.asarray(W_init, dtype=float).ravel()
            if W_init_arr.shape != (n,):
                raise ValueError('Length of W_init must equal the sample size n.')
            W = _project_to_simplex(W_init_arr, float(n))
        Y = W.copy()
        t_fista = 1.0
        lr = float(initial_lr)
        f_W = CurvatureDecorrelator.loss(X, Lambda, W, rho_bar, gamma_w=gamma_w, decorrelation_scale=decorrelation_scale, covariance_mode=covariance_mode, eps=eps)
        best_W = W.copy()
        best_f = float(f_W)
        last_change = np.inf
        used_iter = 0

        def projected_step(point: np.ndarray, lr_start: float):
            (f_point, grad_point) = CurvatureDecorrelator.loss_and_grad(X, Lambda, point, rho_bar, gamma_w=gamma_w, decorrelation_scale=decorrelation_scale, covariance_mode=covariance_mode, eps=eps)
            local_lr = float(lr_start)
            candidate = point.copy()
            f_new = f_point
            for _ in range(max_backtracking):
                candidate = _project_to_simplex(point - local_lr * grad_point, target_sum=float(n))
                f_new = CurvatureDecorrelator.loss(X, Lambda, candidate, rho_bar, gamma_w=gamma_w, decorrelation_scale=decorrelation_scale, covariance_mode=covariance_mode, eps=eps)
                diff = candidate - point
                upper = f_point + float(np.dot(grad_point, diff)) + float(np.dot(diff, diff)) / (2.0 * local_lr)
                if np.isfinite(f_new) and f_new <= upper + 1e-12:
                    return (candidate, float(f_new), local_lr)
                local_lr *= backtrack_factor
            return (candidate, float(f_new), local_lr)
        for k in range(1, max_iter + 1):
            used_iter = k
            (W_new, f_new, lr_used) = projected_step(Y, lr)
            if monotone_restart and (not np.isfinite(f_new) or f_new > f_W + 1e-12):
                Y = W.copy()
                t_fista = 1.0
                (W_new, f_new, lr_used) = projected_step(Y, lr_used)
            if not np.isfinite(f_new):
                warnings.warn('Decorrelation-weight optimization produced a non-finite objective; returning the best weights found so far.', RuntimeWarning)
                break
            if f_new < best_f:
                best_f = float(f_new)
                best_W = W_new.copy()
            last_change = float(np.max(np.abs(W_new - W)))
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_fista * t_fista))
            momentum = (t_fista - 1.0) / t_new
            Y_new = W_new + momentum * (W_new - W)
            Y_new = _project_to_simplex(Y_new, float(n))
            W = W_new
            f_W = float(f_new)
            Y = Y_new
            t_fista = t_new
            lr = min(float(initial_lr), lr_used / backtrack_factor)
            if last_change < tol:
                break
        best_W = _project_to_simplex(best_W, float(n))
        components = CurvatureDecorrelator.loss_components(X, Lambda, best_W, rho_bar, gamma_w=gamma_w, decorrelation_scale=decorrelation_scale, covariance_mode=covariance_mode, eps=eps)
        info = {'loss': float(best_f), 'decorrelation_loss': components['decorrelation_loss'], 'weight_regularization_loss': components['weight_regularization_loss'], 'n_iter': float(used_iter), 'max_weight_change': float(last_change), 'last_lr': float(lr), 'ess': _effective_sample_size(best_W)}
        return (best_W, info)

class PGMFVI:

    @staticmethod
    def _update_rho(X: np.ndarray, r: np.ndarray, m: np.ndarray, t_second: np.ndarray, intercept_var: float, delta_c: float) -> Tuple[np.ndarray, np.ndarray]:
        beta_var = np.maximum(t_second - m ** 2, 0.0)
        c2 = r ** 2 + max(float(intercept_var), 0.0) + X ** 2 @ beta_var
        c2 = np.maximum(c2, 0.0)
        c = np.sqrt(c2)
        rho = np.empty_like(c)
        small = c < delta_c
        rho[small] = 0.25
        large = ~small
        if np.any(large):
            cl = c[large]
            rho[large] = np.tanh(cl / 2.0) / (2.0 * cl)
        rho = np.clip(rho, 0.0, 0.25)
        return (c, rho)

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray, W: np.ndarray, variational_init: Optional[PGVariationalParams]=None, intercept_init: Optional[InterceptVariationalParams]=None, hyper_init: Optional[DWBLRHyperParams]=None, prior_inclusion_prob: Optional[np.ndarray]=None, feature_prior_update: Literal['fixed', 'em']='fixed', feature_prior_damping: float=1.0, sigma_b_sq: float=100.0, max_iter: int=200, tol: float=1e-06, delta_c: float=1e-08, delta_h: float=1e-06, sigma0_sq_max: Optional[float]=None, verbose: bool=False, update_sigma0_sq: bool=True) -> Tuple[PGVariationalParams, InterceptVariationalParams, DWBLRHyperParams, np.ndarray, np.ndarray, Dict[str, Any]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        W = np.asarray(W, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError('X must be a two-dimensional matrix.')
        (n, p) = X.shape
        if n == 0 or p == 0:
            raise ValueError('X must contain at least one sample and one feature.')
        if y.shape != (n,):
            raise ValueError('Length of y must equal the sample size n.')
        if W.shape != (n,):
            raise ValueError('Length of W must equal the sample size n.')
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)) or (not np.all(np.isfinite(W))):
            raise ValueError('X, y or W contains NaN/Inf.')
        if np.any(W < -1e-12):
            raise ValueError('W must be nonnegative.')
        if sigma_b_sq <= 0 or not np.isfinite(sigma_b_sq):
            raise ValueError('sigma_b_sq must be a finite positive number.')
        if max_iter <= 0 or tol <= 0 or delta_c <= 0 or (delta_h <= 0):
            raise ValueError('max_iter must be > 0, and tol, delta_c and delta_h must be positive.')
        if feature_prior_update not in {'fixed', 'em'}:
            raise ValueError("feature_prior_update must be 'fixed' or 'em'.")
        if not 0.0 < feature_prior_damping <= 1.0:
            raise ValueError('feature_prior_damping must lie in (0,1].')
        kappa = y - 0.5
        if hyper_init is None:
            hyper = DWBLRHyperParams(pi0=0.5, sigma0_sq=1.0)
        else:
            hyper = hyper_init.copy()
        hyper.pi0 = float(np.clip(hyper.pi0, delta_h, 1.0 - delta_h))
        hyper.sigma0_sq = float(max(hyper.sigma0_sq, delta_h))
        if prior_inclusion_prob is None:
            feature_prior = None
            if feature_prior_update == 'em':
                raise ValueError("With feature_prior_update='em', a length-p prior_inclusion_prob must be supplied as the initial value of each pi_j.")
        else:
            feature_prior = np.asarray(prior_inclusion_prob, dtype=float).ravel()
            if feature_prior.shape != (p,):
                raise ValueError(f'Length of prior_inclusion_prob must equal p={p}.')
            if not np.all(np.isfinite(feature_prior)):
                raise ValueError('prior_inclusion_prob contains NaN or Inf.')
            if np.any(feature_prior <= 0.0) or np.any(feature_prior >= 1.0):
                raise ValueError('Every element of prior_inclusion_prob must lie in (0,1).')
            feature_prior = np.clip(feature_prior, delta_h, 1.0 - delta_h)
            hyper.pi0 = float(np.mean(feature_prior))
        if variational_init is None:
            alpha = np.full(p, hyper.pi0, dtype=float)
            mu = np.zeros(p, dtype=float)
            s2 = np.full(p, hyper.sigma0_sq, dtype=float)
        else:
            vp = variational_init.copy()
            if vp.alpha.shape != (p,) or vp.mu.shape != (p,) or vp.s2.shape != (p,):
                raise ValueError('variational_init does not match the current p.')
            alpha = np.clip(vp.alpha.astype(float), delta_h, 1.0 - delta_h)
            mu = vp.mu.astype(float).copy()
            s2 = np.maximum(vp.s2.astype(float), delta_h)
        if intercept_init is None:
            mu_b = 0.0
            v_b = float(sigma_b_sq)
        else:
            ip = intercept_init.copy()
            mu_b = float(ip.mu_b)
            v_b = float(max(ip.v_b, delta_h))
            if not np.isfinite(mu_b) or not np.isfinite(v_b):
                raise ValueError('intercept_init contains non-finite parameters.')
        m = alpha * mu
        t_second = alpha * (mu ** 2 + s2)
        r = mu_b + X @ m
        last_change = np.inf
        used_iter = 0
        rho_bar = np.full(n, 0.25, dtype=float)
        eta_bar = W * rho_bar
        for u in range(1, max_iter + 1):
            used_iter = u
            alpha_old = alpha.copy()
            mu_old = mu.copy()
            s2_old = s2.copy()
            mu_b_old_global = float(mu_b)
            v_b_old = float(v_b)
            pi0_old = float(hyper.pi0)
            sigma0_old = float(hyper.sigma0_sq)
            feature_prior_old = None if feature_prior is None else feature_prior.copy()
            (_, rho_bar) = PGMFVI._update_rho(X=X, r=r, m=m, t_second=t_second, intercept_var=v_b, delta_c=delta_c)
            eta_bar = W * rho_bar
            mu_b_before = float(mu_b)
            A_b = float(np.sum(eta_bar))
            residual_without_intercept = r - mu_b_before
            B_b = float(np.sum(W * kappa - eta_bar * residual_without_intercept))
            intercept_precision = A_b + 1.0 / sigma_b_sq
            v_b = 1.0 / max(intercept_precision, delta_h)
            mu_b = v_b * B_b
            r += mu_b - mu_b_before
            for j in range(p):
                xj = X[:, j]
                m_old_j = float(m[j])
                A_j = float(np.dot(eta_bar, xj ** 2))
                residual_without_j = r - xj * m_old_j
                B_j = float(np.dot(xj, W * kappa - eta_bar * residual_without_j))
                precision = A_j + 1.0 / max(hyper.sigma0_sq, delta_h)
                s2_j = 1.0 / max(precision, delta_h)
                mu_j = s2_j * B_j
                pi0_clip = float(feature_prior[j] if feature_prior is not None else np.clip(hyper.pi0, delta_h, 1.0 - delta_h))
                logit_prior = np.log(pi0_clip) - np.log1p(-pi0_clip)
                log_odds = logit_prior + 0.5 * np.log(max(s2_j, delta_h) / max(hyper.sigma0_sq, delta_h)) + mu_j ** 2 / (2.0 * max(s2_j, delta_h))
                alpha_j = float(np.clip(expit(log_odds), delta_h, 1.0 - delta_h))
                s2[j] = s2_j
                mu[j] = mu_j
                alpha[j] = alpha_j
                m_new_j = alpha_j * mu_j
                t_new_j = alpha_j * (mu_j ** 2 + s2_j)
                r += xj * (m_new_j - m_old_j)
                m[j] = m_new_j
                t_second[j] = t_new_j
            if feature_prior is not None and feature_prior_update == 'em':
                feature_prior_target = np.clip(alpha, delta_h, 1.0 - delta_h)
                feature_prior = np.clip((1.0 - feature_prior_damping) * feature_prior + feature_prior_damping * feature_prior_target, delta_h, 1.0 - delta_h)
            hyper.pi0 = float(np.mean(feature_prior)) if feature_prior is not None else float(np.clip(np.mean(alpha), delta_h, 1.0 - delta_h))
            denom = max(float(np.sum(alpha)), delta_h)
            sigma0_new = float(np.sum(alpha * (mu ** 2 + s2)) / denom)
            sigma0_new = max(sigma0_new, delta_h)
            if sigma0_sq_max is not None:
                if sigma0_sq_max <= 0:
                    raise ValueError('sigma0_sq_max must be positive or None.')
                sigma0_new = min(sigma0_new, float(sigma0_sq_max))
            if update_sigma0_sq:
                hyper.sigma0_sq = sigma0_new
            feature_prior_change = 0.0 if feature_prior is None or feature_prior_old is None else float(np.max(np.abs(feature_prior - feature_prior_old)))
            last_change = max(float(np.max(np.abs(alpha - alpha_old))), float(np.max(np.abs(mu - mu_old))), float(np.max(np.abs(s2 - s2_old))), abs(mu_b - mu_b_old_global), abs(v_b - v_b_old), abs(hyper.pi0 - pi0_old), abs(hyper.sigma0_sq - sigma0_old), feature_prior_change)
            if verbose and (u == 1 or u % 20 == 0):
                print(f'      PG-MFVI iter={u:3d} | change={last_change:.3e} | mu_b={mu_b:.4f} | v_b={v_b:.3e} | pi0={hyper.pi0:.3f} | sigma0^2={hyper.sigma0_sq:.3e}')
            if not np.isfinite(last_change):
                raise FloatingPointError('PG-MFVI produced a non-finite parameter change.')
            if last_change < tol:
                break
        m = alpha * mu
        t_second = alpha * (mu ** 2 + s2)
        r = mu_b + X @ m
        (c, rho_bar) = PGMFVI._update_rho(X=X, r=r, m=m, t_second=t_second, intercept_var=v_b, delta_c=delta_c)
        eta_bar = W * rho_bar
        if not (np.all(np.isfinite(alpha)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(s2)) and np.all(np.isfinite(rho_bar)) and np.isfinite(mu_b) and np.isfinite(v_b)):
            raise FloatingPointError('PG-MFVI-I final parameters contain NaN/Inf.')
        variational = PGVariationalParams(alpha=alpha, mu=mu, s2=s2)
        intercept = InterceptVariationalParams(mu_b=float(mu_b), v_b=float(v_b))
        info = {'n_iter': float(used_iter), 'max_param_change': float(last_change), 'pi0': float(hyper.pi0), 'sigma0_sq': float(hyper.sigma0_sq), 'intercept_mu': float(mu_b), 'intercept_var': float(v_b), 'c_mean': float(np.mean(c)), 'c_max': float(np.max(c)), 'feature_prior': None if feature_prior is None else feature_prior.copy(), 'feature_prior_update': feature_prior_update, 'feature_prior_change_inf': float(feature_prior_change)}
        return (variational, intercept, hyper, rho_bar, eta_bar, info)

def update_penalty_matrix(alpha: np.ndarray, nu: float, lambda_floor: float=0.0) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float).ravel()
    if nu <= 0:
        raise ValueError('nu must be > 0.')
    if not 0.0 <= lambda_floor < 1.0:
        raise ValueError('lambda_floor must lie in [0,1).')
    joint = np.outer(alpha, alpha)
    joint = np.clip(joint, 0.0, 1.0)
    Lambda = 1.0 - np.power(joint, nu)
    if lambda_floor > 0:
        Lambda = np.maximum(Lambda, lambda_floor)
    np.fill_diagonal(Lambda, 0.0)
    Lambda = 0.5 * (Lambda + Lambda.T)
    return Lambda

class DWBLRLogistic:

    def __init__(self, T_max: int=50, tau: float=0.5, outer_tol: float=0.0001, min_outer_iter: int=2, nu_start: float=4.0, nu_end: float=1.0, nu_schedule: Literal['geometric', 'linear', 'constant']='geometric', nu_values: Optional[Sequence[float]]=None, anneal_iters: int=20, gamma_w: float=1.0, decorrelation_scale: float=1.0, covariance_mode: Literal['precision', 'normalized']='normalized', use_curvature_decorrelation: bool=True, attacker_max_iter: int=500, attacker_tol: float=1e-06, attacker_initial_lr: float=1.0, attacker_backtrack_factor: float=0.5, attacker_max_backtracking: int=30, inner_max_iter: int=200, inner_tol: float=1e-06, delta_c: float=1e-08, delta_h: float=1e-06, pi0_init: float=0.5, sigma0_sq_init: float=1.0, alpha_init: Optional[Sequence[float]]=None, prior_inclusion_prob: Optional[Sequence[float]]=None, feature_prior_update: Literal['fixed', 'em', 'outer_map']='fixed', feature_prior_damping: float=1.0, sigma0_sq_max: Optional[float]=None, sigma_b_sq: float=100.0, warm_start_inner: bool=True, lambda_floor: float=0.0, use_selective_protection: bool=True, verbose: bool=True, feature_prior_map_strength: float=2.0, feature_prior_map_center: Optional[Sequence[float]]=None, feature_prior_warmup_outer: int=5):
        if T_max <= 0:
            raise ValueError('T_max must be > 0.')
        if not 0.0 < tau < 1.0:
            raise ValueError('tau must lie in (0,1).')
        if outer_tol <= 0 or inner_tol <= 0 or attacker_tol <= 0:
            raise ValueError('All convergence thresholds must be > 0.')
        if min_outer_iter < 1:
            raise ValueError('min_outer_iter must be >= 1.')
        if anneal_iters < 1:
            raise ValueError('anneal_iters must be >= 1.')
        if T_max < anneal_iters + 1:
            raise ValueError('T_max must be at least anneal_iters + 1 so that at least one final refinement round runs after annealing.')
        if nu_start <= 0 or nu_end <= 0:
            raise ValueError('nu_start and nu_end must both be > 0.')
        if not 0.0 < pi0_init < 1.0:
            raise ValueError('pi0_init must lie in (0,1).')
        if sigma0_sq_init <= 0 or sigma_b_sq <= 0:
            raise ValueError('sigma0_sq_init and sigma_b_sq must be > 0.')
        if gamma_w < 0 or decorrelation_scale < 0:
            raise ValueError('gamma_w and decorrelation_scale must be >= 0.')
        if covariance_mode not in {'normalized', 'precision'}:
            raise ValueError("covariance_mode must be 'normalized' or 'precision'.")
        if feature_prior_update not in {'fixed', 'em', 'outer_map'}:
            raise ValueError("feature_prior_update must be 'fixed', 'em' or 'outer_map'.")
        if not 0.0 < feature_prior_damping <= 1.0:
            raise ValueError('feature_prior_damping must lie in (0,1].')
        if feature_prior_update in {'em', 'outer_map'} and prior_inclusion_prob is None:
            raise ValueError('Per-variable em/outer_map mode requires prior_inclusion_prob.')
        if feature_prior_update == 'outer_map':
            if not np.isfinite(feature_prior_map_strength) or feature_prior_map_strength <= 0:
                raise ValueError('feature_prior_map_strength must be a finite positive number.')
            if not isinstance(feature_prior_warmup_outer, (int, np.integer)) or feature_prior_warmup_outer < 0 or feature_prior_warmup_outer >= T_max:
                raise ValueError('feature_prior_warmup_outer must be a nonnegative integer smaller than T_max.')
        self.T_max = int(T_max)
        self.tau = float(tau)
        self.outer_tol = float(outer_tol)
        self.min_outer_iter = int(min_outer_iter)
        self.nu_start = float(nu_start)
        self.nu_end = float(nu_end)
        self.nu_schedule = nu_schedule
        self.nu_values = None if nu_values is None else np.asarray(nu_values, dtype=float)
        self.anneal_iters = int(anneal_iters)
        self.gamma_w = float(gamma_w)
        self.decorrelation_scale = float(decorrelation_scale)
        self.covariance_mode = covariance_mode
        self.use_curvature_decorrelation = bool(use_curvature_decorrelation)
        self.attacker_max_iter = int(attacker_max_iter)
        self.attacker_tol = float(attacker_tol)
        self.attacker_initial_lr = float(attacker_initial_lr)
        self.attacker_backtrack_factor = float(attacker_backtrack_factor)
        self.attacker_max_backtracking = int(attacker_max_backtracking)
        self.inner_max_iter = int(inner_max_iter)
        self.inner_tol = float(inner_tol)
        self.delta_c = float(delta_c)
        self.delta_h = float(delta_h)
        self.pi0_init = float(pi0_init)
        self.sigma0_sq_init = float(sigma0_sq_init)
        self.alpha_init = None if alpha_init is None else np.asarray(alpha_init, dtype=float).ravel()
        self.prior_inclusion_prob = None if prior_inclusion_prob is None else np.asarray(prior_inclusion_prob, dtype=float).ravel()
        self.feature_prior_update = feature_prior_update
        self.feature_prior_damping = float(feature_prior_damping)
        self.feature_prior_map_strength = float(feature_prior_map_strength)
        self.feature_prior_map_center = None if feature_prior_map_center is None else np.asarray(feature_prior_map_center, dtype=float).ravel()
        self.feature_prior_warmup_outer = int(feature_prior_warmup_outer)
        self.sigma0_sq_max = sigma0_sq_max
        self.sigma_b_sq = float(sigma_b_sq)
        self.warm_start_inner = bool(warm_start_inner)
        self.lambda_floor = float(lambda_floor)
        self.use_selective_protection = bool(use_selective_protection)
        self.verbose = bool(verbose)
        self.result_: Optional[DWBLRResult] = None
        self.feature_names_: Optional[List[str]] = None
        self.X_mean_: Optional[np.ndarray] = None
        self.X_scale_: Optional[np.ndarray] = None
        self.zero_variance_mask_: Optional[np.ndarray] = None

    def _prepare_X_fit(self, X: Any) -> np.ndarray:
        if hasattr(X, 'columns') and hasattr(X, 'to_numpy'):
            self.feature_names_ = [str(c) for c in list(X.columns)]
            X_arr = np.asarray(X.to_numpy(), dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.feature_names_ = None
        if X_arr.ndim != 2:
            raise ValueError(f'X must be a two-dimensional matrix; got shape={X_arr.shape}.')
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError('X must contain at least one sample and one feature.')
        if not np.all(np.isfinite(X_arr)):
            raise ValueError('X contains NaN or Inf.')
        p = X_arr.shape[1]
        self.X_mean_ = np.zeros(p, dtype=float)
        self.X_scale_ = np.ones(p, dtype=float)
        self.zero_variance_mask_ = np.zeros(p, dtype=bool)
        return X_arr

    def _prepare_X_predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        if hasattr(X, 'to_numpy'):
            X_arr = np.asarray(X.to_numpy(), dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError('X must be a two-dimensional matrix.')
        if X_arr.shape[1] != len(self.result_.alpha):
            raise ValueError('The number of features in the new X does not match the training data.')
        if not np.all(np.isfinite(X_arr)):
            raise ValueError('X contains NaN or Inf.')
        return X_arr

    @staticmethod
    def _prepare_y(y: Any, n: int) -> np.ndarray:
        if hasattr(y, 'to_numpy'):
            y_arr = np.asarray(y.to_numpy(), dtype=float).ravel()
        else:
            y_arr = np.asarray(y, dtype=float).ravel()
        if y_arr.shape != (n,):
            raise ValueError('X and y have different numbers of samples.')
        if not np.all(np.isfinite(y_arr)):
            raise ValueError('y contains NaN or Inf.')
        unique = np.unique(y_arr)
        if not np.all(np.isin(unique, [0.0, 1.0])):
            raise ValueError(f'y must be 0/1; observed labels={unique.tolist()}.')
        if unique.size < 2:
            warnings.warn('y has only one class; variable selection is usually meaningless.', RuntimeWarning)
        return y_arr

    def fit(self, X: Any, y: Any, feature_names: Optional[Sequence[str]]=None) -> 'DWBLRLogistic':
        Xs = self._prepare_X_fit(X)
        (n, p) = Xs.shape
        ys = self._prepare_y(y, n)
        if feature_names is not None:
            if len(feature_names) != p:
                raise ValueError('Length of feature_names must equal p.')
            self.feature_names_ = [str(v) for v in feature_names]
        if self.nu_values is not None:
            if self.nu_values.ndim != 1 or len(self.nu_values) < self.anneal_iters:
                raise ValueError('nu_values must be one-dimensional with length at least anneal_iters.')
            anneal_seq = self.nu_values[:self.anneal_iters].astype(float)
            if not np.all(np.isfinite(anneal_seq)) or np.any(anneal_seq <= 0):
                raise ValueError('The annealing part of nu_values must consist of finite positive numbers.')
            final_nu = float(anneal_seq[-1])
            if not np.isclose(final_nu, self.nu_end, rtol=1e-10, atol=1e-12):
                warnings.warn('The anneal_iters-th value of the custom nu_values differs from nu_end; after annealing the last annealing value of nu_values is used.', RuntimeWarning)
        else:
            anneal_seq = _make_nu_schedule(self.anneal_iters, self.nu_start, self.nu_end, self.nu_schedule)
            anneal_seq[-1] = float(self.nu_end)
            final_nu = float(self.nu_end)
        nu_seq = np.full(self.T_max, final_nu, dtype=float)
        nu_seq[:self.anneal_iters] = anneal_seq
        W = np.ones(n, dtype=float)
        rho_bar = np.full(n, 0.25, dtype=float)
        eta_bar = W * rho_bar
        Lambda = np.ones((p, p), dtype=float) - np.eye(p, dtype=float)
        feature_prior_state: Optional[np.ndarray]
        if self.prior_inclusion_prob is None:
            feature_prior_state = None
        else:
            if self.prior_inclusion_prob.shape != (p,):
                raise ValueError(f'Length of prior_inclusion_prob must equal p={p}.')
            if not np.all(np.isfinite(self.prior_inclusion_prob)):
                raise ValueError('prior_inclusion_prob contains NaN or Inf.')
            if np.any(self.prior_inclusion_prob <= 0.0) or np.any(self.prior_inclusion_prob >= 1.0):
                raise ValueError('Every element of prior_inclusion_prob must lie in (0,1).')
            feature_prior_state = np.clip(self.prior_inclusion_prob.copy(), self.delta_h, 1.0 - self.delta_h)
        alpha_feedback_old = np.zeros(p, dtype=float)
        map_center = None
        if self.feature_prior_update == 'outer_map':
            map_center = feature_prior_state.copy() if self.feature_prior_map_center is None else self.feature_prior_map_center.copy()
            if map_center.shape != (p,) or not np.all(np.isfinite(map_center)) or np.any(map_center <= 0) or np.any(map_center >= 1):
                raise ValueError('feature_prior_map_center must be a finite length-p vector with elements in (0,1).')
        self.feature_prior_trace_ = []
        if self.alpha_init is None:
            initial_alpha = np.full(p, self.pi0_init, dtype=float) if feature_prior_state is None else feature_prior_state.copy()
        else:
            if self.alpha_init.shape != (p,):
                raise ValueError(f'Length of alpha_init must equal p={p}.')
            if not np.all(np.isfinite(self.alpha_init)):
                raise ValueError('alpha_init contains NaN or Inf.')
            if np.any(self.alpha_init <= 0.0) or np.any(self.alpha_init >= 1.0):
                raise ValueError('Every element of alpha_init must lie in (0,1).')
            initial_alpha = np.clip(self.alpha_init, self.delta_h, 1.0 - self.delta_h)
        variational: Optional[PGVariationalParams] = PGVariationalParams(alpha=initial_alpha, mu=np.zeros(p, dtype=float), s2=np.full(p, self.sigma0_sq_init, dtype=float))
        intercept: Optional[InterceptVariationalParams] = InterceptVariationalParams(mu_b=0.0, v_b=self.sigma_b_sq)
        hyper = DWBLRHyperParams(pi0=self.pi0_init, sigma0_sq=self.sigma0_sq_init)
        history: List[Dict[str, float]] = []
        used_outer = 0
        if self.verbose:
            print(f'[*] DW-BLR start: n={n}, p={p}, cov_mode={self.covariance_mode}, gamma_w={self.gamma_w}, sigma_b^2={self.sigma_b_sq}, tau={self.tau}, anneal_iters={self.anneal_iters}, final_nu={final_nu}')
        for t in range(self.T_max):
            used_outer = t + 1
            W_old = W.copy()
            rho_old = rho_bar.copy()
            intercept_mu_old = float(intercept.mu_b)
            feature_prior_old_outer = None if feature_prior_state is None else feature_prior_state.copy()
            decorrelation_rho = rho_bar if self.use_curvature_decorrelation else np.ones_like(rho_bar)
            (W_new, attacker_info) = CurvatureDecorrelator.fista(X=Xs, Lambda=Lambda, rho_bar=decorrelation_rho, W_init=W, gamma_w=self.gamma_w, decorrelation_scale=self.decorrelation_scale, covariance_mode=self.covariance_mode, max_iter=self.attacker_max_iter, tol=self.attacker_tol, initial_lr=self.attacker_initial_lr, backtrack_factor=self.attacker_backtrack_factor, max_backtracking=self.attacker_max_backtracking, monotone_restart=True)
            slope_init = variational if self.warm_start_inner else None
            intercept_init = intercept if self.warm_start_inner else None
            (variational_new, intercept_new, hyper_new, rho_new, eta_new, inner_info) = PGMFVI.fit(X=Xs, y=ys, W=W_new, variational_init=slope_init, intercept_init=intercept_init, hyper_init=hyper, prior_inclusion_prob=feature_prior_state, feature_prior_update='fixed' if self.feature_prior_update == 'outer_map' else self.feature_prior_update, feature_prior_damping=self.feature_prior_damping, sigma_b_sq=self.sigma_b_sq, max_iter=self.inner_max_iter, tol=self.inner_tol, delta_c=self.delta_c, delta_h=self.delta_h, sigma0_sq_max=self.sigma0_sq_max, verbose=False)
            alpha_new = variational_new.alpha
            returned_feature_prior = inner_info['feature_prior']
            feature_prior_new = None if returned_feature_prior is None else np.asarray(returned_feature_prior, dtype=float).copy()
            map_residual = 0.0
            if self.feature_prior_update == 'outer_map':
                kappa = self.feature_prior_map_strength
                target = (alpha_new + kappa * map_center) / (1.0 + kappa)
                update_prior = t >= self.feature_prior_warmup_outer
                map_residual = float(np.max(np.abs(target - feature_prior_state)))
                if update_prior:
                    feature_prior_new = np.clip((1.0 - self.feature_prior_damping) * feature_prior_state + self.feature_prior_damping * target, self.delta_h, 1.0 - self.delta_h)
                hyper_new.pi0 = float(np.mean(feature_prior_new))
                self.feature_prior_trace_.append({'outer_iter': t + 1, 'updated': update_prior, 'prior_used': feature_prior_state.tolist(), 'prior_returned_by_inner': np.asarray(returned_feature_prior).tolist(), 'pip': alpha_new.tolist(), 'map_target': target.tolist(), 'prior_next': feature_prior_new.tolist()})
            nu_t = float(nu_seq[t])
            if self.use_selective_protection:
                Lambda_new = update_penalty_matrix(alpha=alpha_new, nu=nu_t, lambda_floor=self.lambda_floor)
            else:
                Lambda_new = np.ones((p, p), dtype=float) - np.eye(p, dtype=float)
            selected_now = np.where(alpha_new > self.tau)[0]
            dW = float(np.max(np.abs(W_new - W_old)))
            dA = float(np.max(np.abs(alpha_new - alpha_feedback_old)))
            dB = abs(float(intercept_new.mu_b) - intercept_mu_old)
            dRho = float(np.max(np.abs(rho_new - rho_old)))
            dPi = 0.0 if feature_prior_new is None or feature_prior_old_outer is None else float(np.max(np.abs(feature_prior_new - feature_prior_old_outer)))
            dOuter = max(dW, dA, dB, dRho, dPi)
            if self.feature_prior_update == 'outer_map':
                dOuter = max(dOuter, map_residual)
            history.append({'outer_iter': float(t + 1), 'nu': nu_t, 'outer_change': float(dOuter), 'weight_change_inf': dW, 'alpha_change_inf': dA, 'intercept_change_abs': float(dB), 'rho_change_inf': dRho, 'feature_prior_change_inf': dPi, 'feature_prior_map_residual': map_residual, 'feature_prior_mean': float(hyper_new.pi0 if feature_prior_new is None else np.mean(feature_prior_new)), 'feature_prior_min': float(hyper_new.pi0 if feature_prior_new is None else np.min(feature_prior_new)), 'feature_prior_max': float(hyper_new.pi0 if feature_prior_new is None else np.max(feature_prior_new)), 'intercept_mu': float(intercept_new.mu_b), 'intercept_var': float(intercept_new.v_b), 'attacker_loss': float(attacker_info['loss']), 'decorrelation_loss': float(attacker_info['decorrelation_loss']), 'weight_regularization_loss': float(attacker_info['weight_regularization_loss']), 'attacker_iters': float(attacker_info['n_iter']), 'inner_iters': float(inner_info['n_iter']), 'inner_change': float(inner_info['max_param_change']), 'pi0': float(hyper_new.pi0), 'sigma0_sq': float(hyper_new.sigma0_sq), 'ess': float(attacker_info['ess']), 'n_selected': float(len(selected_now)), 'rho_mean': float(np.mean(rho_new)), 'rho_min': float(np.min(rho_new)), 'rho_max': float(np.max(rho_new)), 'annealing_complete': float(t + 1 >= self.anneal_iters), 'final_refinement': float(t + 1 > self.anneal_iters), 'early_stopping_allowed': float(t + 1 >= max(self.min_outer_iter, self.anneal_iters + 1, self.feature_prior_warmup_outer + 2 if self.feature_prior_update == 'outer_map' else 1) and (self.feature_prior_update != 'outer_map' or inner_info['max_param_change'] < self.inner_tol))})
            if self.verbose and (t == 0 or (t + 1) % 5 == 0):
                top_idx = np.argsort(alpha_new)[-min(5, p):][::-1].tolist()
                top_val = [round(float(alpha_new[i]), 4) for i in top_idx]
                print(f"  Outer {t + 1:3d} | nu={nu_t:.3f} | phase={('anneal' if t + 1 <= self.anneal_iters else 'refine')} | d={dOuter:.2e} | dW={dW:.1e} | dPIP={dA:.1e} | dB0={dB:.1e} | dRho={dRho:.1e} | b0={intercept_new.mu_b:.4f} | ESS={attacker_info['ess']:.1f}/{n} | selected={len(selected_now)} | TopPIP={list(zip(top_idx, top_val))}")
            W = W_new
            rho_bar = rho_new
            eta_bar = eta_new
            Lambda = Lambda_new
            variational = variational_new
            intercept = intercept_new
            hyper = hyper_new
            feature_prior_state = feature_prior_new
            alpha_feedback_old = alpha_new.copy()
            min_stop_iter = max(self.min_outer_iter, self.anneal_iters + 1)
            if self.feature_prior_update == 'outer_map':
                min_stop_iter = max(min_stop_iter, self.feature_prior_warmup_outer + 2)
            can_early_stop = t + 1 >= min_stop_iter
            if self.feature_prior_update == 'outer_map':
                can_early_stop = can_early_stop and inner_info['max_param_change'] < self.inner_tol
            if can_early_stop and dOuter < self.outer_tol:
                if self.verbose:
                    print(f'  ✓ Outer converged at iter {t + 1} after completed annealing: max(dW,dPIP,dB0,dRho)={dOuter:.3e}, nu={nu_t:.6g}')
                break
        if variational is None or intercept is None:
            raise RuntimeError('Invalid internal state: variational parameters were not created.')
        alpha = variational.alpha.copy()
        mu = variational.mu.copy()
        s2 = variational.s2.copy()
        selected = np.where(alpha > self.tau)[0]
        selected_names = None
        if self.feature_names_ is not None:
            selected_names = [self.feature_names_[i] for i in selected]
        self.result_ = DWBLRResult(selected_features=selected, selected_feature_names=selected_names, weights=W.copy(), alpha=alpha, mu=mu, s2=s2, intercept_mu=float(intercept.mu_b), intercept_var=float(intercept.v_b), sigma_b_sq=float(self.sigma_b_sq), rho_bar=rho_bar.copy(), eta_bar=eta_bar.copy(), Lambda=Lambda.copy(), hyper=hyper.copy(), feature_prior=np.full(p, hyper.pi0, dtype=float) if feature_prior_state is None else feature_prior_state.copy(), feature_prior_update='scalar_eb' if self.prior_inclusion_prob is None else self.feature_prior_update, n_outer_iterations=used_outer, history=history)
        return self

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise ValueError('The model is not fitted; call fit(X, y) first.')

    def get_selected_features(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.selected_features.copy()

    def get_selected_feature_names(self) -> Optional[List[str]]:
        self._check_fitted()
        if self.result_.selected_feature_names is None:
            return None
        return list(self.result_.selected_feature_names)

    def get_pip(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.alpha.copy()

    def get_feature_prior(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.feature_prior.copy()

    def get_weights(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.weights.copy()

    def get_curvature(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.rho_bar.copy()

    def get_effective_precision_weights(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.eta_bar.copy()

    def get_lambda(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.Lambda.copy()

    def get_posterior_mean_coef(self, original_scale: bool=False) -> np.ndarray:
        self._check_fitted()
        beta = self.result_.posterior_mean_beta.copy()
        if original_scale:
            beta = beta
        return beta

    def get_posterior_variance_coef(self, original_scale: bool=False) -> np.ndarray:
        self._check_fitted()
        var = self.result_.posterior_variance_beta.copy()
        if original_scale:
            var = var / self.X_scale_ ** 2
        return var

    def get_intercept(self, original_scale: bool=False) -> float:
        self._check_fitted()
        if not original_scale:
            return float(self.result_.intercept_mu)
        beta_orig = self.get_posterior_mean_coef(original_scale=True)
        return float(self.result_.intercept_mu - np.dot(self.X_mean_, beta_orig))

    def get_original_scale_intercept(self) -> float:
        return self.get_intercept(original_scale=True)

    def get_intercept_variance(self, original_scale: bool=False) -> float:
        self._check_fitted()
        if not original_scale:
            return float(self.result_.intercept_var)
        transform = self.X_mean_
        extra = float(np.sum(transform ** 2 * self.result_.posterior_variance_beta))
        return float(self.result_.intercept_var + extra)

    def get_intercept_posterior(self) -> Dict[str, float]:
        self._check_fitted()
        original_var = self.get_intercept_variance(original_scale=True)
        return {'posterior_mean': float(self.result_.intercept_mu), 'posterior_variance': float(self.result_.intercept_var), 'posterior_sd': float(self.result_.intercept_sd), 'prior_variance_sigma_b_sq': float(self.result_.sigma_b_sq), 'original_scale_posterior_mean': float(self.get_intercept(original_scale=True)), 'original_scale_posterior_variance_approx': float(original_var), 'original_scale_posterior_sd_approx': float(np.sqrt(max(original_var, 0.0)))}

    def decision_function(self, X: Any) -> np.ndarray:
        Xs = self._prepare_X_predict(X)
        beta = self.result_.posterior_mean_beta
        return self.result_.intercept_mu + Xs @ beta

    def predict_proba(self, X: Any) -> np.ndarray:
        p1 = expit(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any, threshold: float=0.5) -> np.ndarray:
        if not 0.0 < threshold < 1.0:
            raise ValueError('threshold must lie in (0,1).')
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def posterior_summary(self, sort_by_pip: bool=True):
        self._check_fitted()
        alpha = self.result_.alpha
        mu = self.result_.mu
        s2 = self.result_.s2
        beta_mean = alpha * mu
        beta_var = self.result_.posterior_variance_beta
        rows = []
        for j in range(len(alpha)):
            rows.append({'feature_index': int(j), 'feature_name': None if self.feature_names_ is None else self.feature_names_[j], 'PIP_alpha': float(alpha[j]), 'slab_mu': float(mu[j]), 'slab_sd': float(np.sqrt(max(s2[j], 0.0))), 'posterior_mean_beta': float(beta_mean[j]), 'posterior_sd_beta': float(np.sqrt(max(beta_var[j], 0.0))), 'selected': bool(alpha[j] > self.tau)})
        if sort_by_pip:
            rows.sort(key=lambda z: z['PIP_alpha'], reverse=True)
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def intercept_summary(self):
        self._check_fitted()
        original_var = self.get_intercept_variance(original_scale=True)
        row = {'term': 'intercept', 'posterior_mean_input_scale': float(self.result_.intercept_mu), 'posterior_variance_input_scale': float(self.result_.intercept_var), 'posterior_sd_input_scale': float(self.result_.intercept_sd), 'posterior_mean_original_scale': float(self.get_intercept(original_scale=True)), 'posterior_variance_original_scale_approx': float(original_var), 'posterior_sd_original_scale_approx': float(np.sqrt(max(original_var, 0.0))), 'prior_variance_sigma_b_sq': float(self.result_.sigma_b_sq), 'always_included': True}
        try:
            import pandas as pd
            return pd.DataFrame([row])
        except ImportError:
            return row

    def history_table(self):
        self._check_fitted()
        try:
            import pandas as pd
            return pd.DataFrame(self.result_.history)
        except ImportError:
            return list(self.result_.history)

def _positive_int(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {('nonnegative' if zero else 'positive')} integer")
    return int(value)

def _prob_vector(value, p, name, scalar_ok=False):
    arr = np.asarray(value, dtype=float)
    if scalar_ok and arr.ndim == 0:
        arr = np.full(p, float(arr))
    if arr.shape != (p,) or not np.isfinite(arr).all() or np.any((arr <= 0) | (arr >= 1)):
        raise ValueError(f'{name} must contain {p} finite probabilities strictly between 0 and 1')
    return arr.copy()

class DWBLRStagedLogistic(DWBLRLogistic):

    def __init__(self, *, warmup_sweeps=30, unprotected_rounds=5, prior_mode='feature', prior_update='outer_map', prior_damping=0.2, prior_update_scope='outer', record_inner_trace=False, prior_update_interval=1, prior_strength=1.0, prior_center=0.3, prior_clip=0.01, update_prior_unprotected=False, weight_damping=0.6, lambda_damping=0.6, smooth_protection_start=True, attacker_scale_step_by_n=True, slab_update='outer_eb', slab_damping=0.5, stable_rounds=5, stability_window=10, **kwargs):
        forbidden = {'feature_prior_update', 'feature_prior_damping', 'feature_prior_map_strength', 'feature_prior_map_center', 'feature_prior_warmup_outer'}.intersection(kwargs)
        if forbidden:
            raise ValueError(f'Use the explicit prior_mode/prior_update controls, not {sorted(forbidden)}')
        defaults = dict(T_max=1200, tau=0.5, outer_tol=1e-05, min_outer_iter=5, nu_start=3.0, nu_end=1.0, anneal_iters=20, lambda_floor=0.1, gamma_w=0.1, decorrelation_scale=1.0, attacker_max_iter=3500, attacker_tol=1e-06, inner_max_iter=200, inner_tol=1e-06, sigma0_sq_init=2.0, sigma_b_sq=10.0, pi0_init=0.5, verbose=False)
        defaults.update(kwargs)
        super().__init__(**defaults)
        if not self.use_selective_protection or not self.use_curvature_decorrelation:
            raise ValueError('DW-BLR requires curvature decorrelation and selective protection')
        if not self.warm_start_inner:
            raise ValueError('warm_start_inner must be True to carry the stage state forward')
        self.warmup_sweeps = _positive_int(warmup_sweeps, 'warmup_sweeps', True)
        self.unprotected_rounds = _positive_int(unprotected_rounds, 'unprotected_rounds', True)
        self.prior_update_interval = _positive_int(prior_update_interval, 'prior_update_interval')
        self.stable_rounds = _positive_int(stable_rounds, 'stable_rounds')
        self.stability_window = _positive_int(stability_window, 'stability_window')
        if prior_mode not in {'shared', 'feature'}:
            raise ValueError('prior_mode must be shared or feature')
        if prior_update not in {'fixed', 'outer_eb', 'outer_map'}:
            raise ValueError('prior_update must be fixed, outer_eb or outer_map')
        if prior_update_scope not in {'inner', 'outer'}:
            raise ValueError('prior_update_scope must be inner or outer')
        if prior_update_scope == 'inner' and prior_update != 'outer_eb':
            raise ValueError('Only EM supports inner updates; Beta-MAP/fixed use outer scope')
        if prior_update_scope == 'inner' and prior_update_interval != 1:
            raise ValueError('inner updates require prior_update_interval=1')
        if update_prior_unprotected:
            raise ValueError('All modes freeze inclusion priors during unprotected rounds')
        if slab_update not in {'fixed', 'outer_eb'}:
            raise ValueError('slab_update must be fixed or outer_eb')
        for (name, val) in (('prior_damping', prior_damping), ('weight_damping', weight_damping), ('lambda_damping', lambda_damping), ('slab_damping', slab_damping)):
            if not np.isfinite(val) or not 0 < val <= 1:
                raise ValueError(f'{name} must be in (0,1]')
            setattr(self, name, float(val))
        if not np.isfinite(prior_clip) or not self.delta_h <= prior_clip < 0.5:
            raise ValueError('prior_clip must be at least delta_h and less than 0.5')
        if not np.isfinite(prior_strength) or prior_strength < 0 or (prior_update == 'outer_map' and prior_strength <= 0):
            raise ValueError('outer_map requires finite prior_strength > 0')
        if prior_mode == 'shared' and self.prior_inclusion_prob is not None:
            raise ValueError('shared mode uses pi0_init; prior_inclusion_prob must be None')
        (self.prior_mode, self.prior_update) = (prior_mode, prior_update)
        self.prior_update_scope = prior_update_scope
        self.record_inner_trace = bool(record_inner_trace)
        (self.prior_strength, self.prior_center) = (float(prior_strength), prior_center)
        (self.prior_clip, self.update_prior_unprotected) = (float(prior_clip), bool(update_prior_unprotected))
        self.slab_update = slab_update
        self.attacker_scale_step_by_n = bool(attacker_scale_step_by_n)
        self.smooth_protection_start = bool(smooth_protection_start)

    def _prior_target(self, alpha, current):
        if self.prior_update == 'fixed':
            return current.copy()
        k = self.prior_strength if self.prior_update == 'outer_map' else 0.0
        if self.prior_mode == 'shared':
            value = (float(alpha.sum()) + k * self.prior_center_[0]) / (len(alpha) + k)
            target = np.full_like(alpha, value)
        else:
            target = (alpha + k * self.prior_center_) / (1.0 + k)
        return np.clip(target, self.prior_clip, 1.0 - self.prior_clip)

    def fit(self, X, y, feature_names=None):
        self.result_ = None
        self.converged_ = False
        self.stop_reason_ = 'maximum full-stage iterations reached'
        Xs = self._prepare_X_fit(X)
        (N, p) = Xs.shape
        ys = self._prepare_y(y, N)
        if feature_names is not None:
            if len(feature_names) != p:
                raise ValueError('feature_names length must match p')
            self.feature_names_ = list(map(str, feature_names))
        self.prior_center_ = _prob_vector(self.prior_center, p, 'prior_center', scalar_ok=True)
        if self.prior_mode == 'shared' and (not np.all(self.prior_center_ == self.prior_center_[0])):
            raise ValueError('shared mode requires a scalar or constant prior_center')
        initial_prior = np.full(p, self.pi0_init) if self.prior_inclusion_prob is None else _prob_vector(self.prior_inclusion_prob, p, 'prior_inclusion_prob')
        if np.any((initial_prior < self.prior_clip) | (initial_prior > 1.0 - self.prior_clip)):
            raise ValueError('initial priors must lie within [prior_clip, 1-prior_clip]')
        prior = initial_prior.copy()
        alpha = np.full(p, 0.5) if self.alpha_init is None else _prob_vector(self.alpha_init, p, 'alpha_init')
        vp = PGVariationalParams(alpha=alpha, mu=np.zeros(p), s2=np.full(p, self.sigma0_sq_init))
        ip = InterceptVariationalParams(mu_b=0.0, v_b=self.sigma_b_sq)
        hyper = DWBLRHyperParams(pi0=float(prior.mean()), sigma0_sq=self.sigma0_sq_init)
        W = np.ones(N)
        rho = np.full(N, 0.25)
        global_lambda = np.ones((p, p)) - np.eye(p)
        Lambda = global_lambda.copy()
        if self.nu_values is None:
            nu_seq = _make_nu_schedule(self.anneal_iters, self.nu_start, self.nu_end, self.nu_schedule)
            nu_seq[-1] = self.nu_end
        else:
            a = np.asarray(self.nu_values, dtype=float)
            if a.ndim != 1 or len(a) < self.anneal_iters or (not np.isfinite(a).all()) or np.any(a <= 0):
                raise ValueError('nu_values must contain at least anneal_iters positive finite entries')
            nu_seq = a[:self.anneal_iters]
        (self.history_, self.trace_, self.phase_snapshots_) = ([], [], {})

        def solve(max_iter, inner_em=False):

            def fixed_solve(v, b, h, pi, limit):
                return PGMFVI.fit(X=Xs, y=ys, W=W, variational_init=v, intercept_init=b, hyper_init=h, prior_inclusion_prob=pi, feature_prior_update='fixed', sigma_b_sq=self.sigma_b_sq, max_iter=limit, tol=self.inner_tol, delta_c=self.delta_c, delta_h=self.delta_h, sigma0_sq_max=self.sigma0_sq_max, update_sigma0_sq=False, verbose=False)
            if not inner_em:
                return fixed_solve(vp, ip, hyper, prior, max_iter)
            (local_v, local_b, local_h) = (vp, ip, hyper)
            local_prior = prior.copy()
            inner_trace = []
            for u in range(1, max_iter + 1):
                (v, b, h, rho_out, eta_out, info) = fixed_solve(local_v, local_b, local_h, local_prior, 1)
                target = self._prior_target(v.alpha, local_prior)
                residual = float(np.max(abs(target - local_prior)))
                next_prior = (1.0 - self.prior_damping) * local_prior + self.prior_damping * target
                change = max(float(info['max_param_change']), residual)
                if self.record_inner_trace:
                    inner_trace.append(dict(sweep=u, pip=v.alpha.copy(), prior_used=local_prior.copy(), prior_target=target.copy(), prior_next=next_prior.copy(), prior_target_residual=residual))
                info.update(n_iter=u, max_param_change=change, inner_prior_target_residual=residual, inner_prior_next=next_prior.copy(), inner_prior_trace=inner_trace)
                if change < self.inner_tol or u == max_iter:
                    return (v, b, h, rho_out, eta_out, info)
                (local_v, local_b) = (v, b)
                local_prior = next_prior
                local_h = DWBLRHyperParams(pi0=float(local_prior.mean()), sigma0_sq=h.sigma0_sq)

        def snapshot(name):
            self.phase_snapshots_[name] = dict(pip=vp.alpha.copy(), prior=prior.copy(), weights=W.copy(), rho=rho.copy(), Lambda=Lambda.copy(), sigma0_sq=hyper.sigma0_sq)
        snapshot('initial')
        for step in range(1, self.warmup_sweeps + 1):
            (vp, ip, hyper, rho, _, info) = solve(1)
            self.history_.append(dict(stage='warmup', stage_iter=step, inner_iters=1, inner_change=info['max_param_change'], ess=float(N), prior_updated=False, sigma0_sq=hyper.sigma0_sq, prior_min=float(prior.min()), prior_max=float(prior.max())))
            self.trace_.append(dict(stage='warmup', stage_iter=step, pip=vp.alpha.copy(), prior_used=prior.copy(), prior_next=prior.copy(), prior_target=prior.copy()))
        snapshot('warmup')
        streak = 0
        full_support = []
        total_rounds = self.unprotected_rounds + self.T_max
        for t in range(1, total_rounds + 1):
            protected = t > self.unprotected_rounds
            stage = 'full' if protected else 'unprotected'
            step = t - self.unprotected_rounds if protected else t
            nu = float(nu_seq[min(step - 1, len(nu_seq) - 1)]) if protected else float(nu_seq[0])
            if not protected:
                Lambda = global_lambda.copy()
            elif step == 1:
                initial_target = update_penalty_matrix(vp.alpha, nu, self.lambda_floor)
                Lambda = (1.0 - self.lambda_damping) * global_lambda + self.lambda_damping * initial_target if self.smooth_protection_start else initial_target
            (old_W, old_rho, old_prior) = (W.copy(), rho.copy(), prior.copy())
            (old_vp, old_ip, old_sigma) = (vp.copy(), ip.copy(), hyper.sigma0_sq)
            used_lambda = Lambda.copy()
            (W_target, wi) = CurvatureDecorrelator.fista(X=Xs, Lambda=used_lambda, rho_bar=rho, W_init=W, gamma_w=self.gamma_w, decorrelation_scale=self.decorrelation_scale, covariance_mode=self.covariance_mode, max_iter=self.attacker_max_iter, tol=self.attacker_tol, initial_lr=self.attacker_initial_lr * (N if self.attacker_scale_step_by_n else 1.0), backtrack_factor=self.attacker_backtrack_factor, max_backtracking=self.attacker_max_backtracking)
            weight_residual = float(np.max(abs(W_target - W)))
            W = (1.0 - self.weight_damping) * W + self.weight_damping * W_target
            inner_em = protected and self.prior_update_scope == 'inner'
            (vp, ip, hyper, rho, _, info) = solve(self.inner_max_iter, inner_em=inner_em)
            if inner_em:
                prior = np.asarray(info['feature_prior'], dtype=float).copy()
            elif not np.array_equal(info['feature_prior'], prior):
                raise RuntimeError('Inclusion prior changed inside a fixed-prior inner solve')
            active_prior = self.prior_update != 'fixed' and protected
            target = self._prior_target(vp.alpha, prior)
            prior_residual = float(np.max(abs(target - prior))) if active_prior else 0.0
            prior_update_due = active_prior and step % self.prior_update_interval == 0
            prior_next = (1.0 - self.prior_damping) * prior + self.prior_damping * target if prior_update_due else prior.copy()
            slab_target = old_sigma
            sigma_next = old_sigma
            slab_residual = 0.0
            active_slab = protected and self.slab_update == 'outer_eb'
            if active_slab:
                slab_target = float(np.sum(vp.alpha * (vp.mu ** 2 + vp.s2)) / max(vp.alpha.sum(), self.delta_h))
                slab_target = max(slab_target, self.delta_h)
                if self.sigma0_sq_max is not None:
                    slab_target = min(slab_target, self.sigma0_sq_max)
                slab_residual = abs(slab_target - old_sigma) / max(1.0, abs(old_sigma))
                sigma_next = (1.0 - self.slab_damping) * old_sigma + self.slab_damping * slab_target
            L_target = update_penalty_matrix(vp.alpha, nu, self.lambda_floor) if protected else global_lambda.copy()
            lambda_residual = float(np.max(abs(L_target - used_lambda))) if protected else 0.0
            Lambda = (1.0 - self.lambda_damping) * used_lambda + self.lambda_damping * L_target if protected else global_lambda.copy()
            support = vp.alpha > self.tau
            if protected:
                streak = streak + 1 if full_support and np.array_equal(support, full_support[-1]) else 1
                full_support.append(support.copy())
            changes = [weight_residual, prior_residual, slab_residual, lambda_residual, float(np.max(abs(prior - old_prior))), float(np.max(abs(vp.alpha - old_vp.alpha))), float(np.max(abs(vp.mu - old_vp.mu))), float(np.max(abs(vp.s2 - old_vp.s2))), abs(ip.mu_b - old_ip.mu_b), abs(ip.v_b - old_ip.v_b), float(np.max(abs(rho - old_rho)))]
            outer_residual = max(changes)
            (_, grad) = CurvatureDecorrelator.loss_and_grad(Xs, Lambda, W, rho, gamma_w=self.gamma_w, decorrelation_scale=self.decorrelation_scale, covariance_mode=self.covariance_mode)
            pg_residual = float(np.max(abs(W - _project_to_simplex(W - N * grad, float(N)))))
            row = dict(stage=stage, stage_iter=step, outer_iter=t, nu=nu, prior_update_scope=self.prior_update_scope, inner_prior_updates=int(info['n_iter']) if inner_em else 0, inner_prior_target_residual=float(info.get('inner_prior_target_residual', 0.0)), outer_change=outer_residual, weight_target_residual=weight_residual, weight_projected_gradient_residual=pg_residual, prior_target_residual=prior_residual, slab_target_residual=slab_residual, lambda_target_residual=lambda_residual, inner_iters=int(info['n_iter']), inner_change=float(info['max_param_change']), attacker_iters=int(wi['n_iter']), attacker_last_change=float(wi['max_weight_change']), prior_updated=bool(prior_update_due), prior_min=float(prior.min()), prior_max=float(prior.max()), prior_next_min=float(prior_next.min()), prior_next_max=float(prior_next.max()), sigma0_sq=old_sigma, sigma0_sq_next=sigma_next, slab_update_active=bool(active_slab), ess=float(W.sum() ** 2 / np.dot(W, W)), max_weight_share=float(W.max() / W.sum()), n_selected=int(support.sum()), support_streak=streak)
            self.history_.append(row)
            self.trace_.append(dict(stage=stage, stage_iter=step, pip=vp.alpha.copy(), prior_outer_start=old_prior.copy(), prior_used=prior.copy(), prior_next=prior_next.copy(), prior_target=target.copy(), inner_prior_trace=info.get('inner_prior_trace', []), lambda_used=used_lambda, lambda_target=L_target.copy(), rho_mean=float(rho.mean())))
            due = not active_prior or prior_update_due
            stop = protected and step >= max(self.min_outer_iter, self.anneal_iters + 1, self.stable_rounds) and due and (streak >= self.stable_rounds) and (outer_residual < self.outer_tol) and (info['max_param_change'] < self.inner_tol) and (pg_residual < self.outer_tol)
            (self.next_prior_, self.next_sigma0_sq_) = (prior_next.copy(), sigma_next)
            self.prior_used_ = prior.copy()
            if not protected and step == self.unprotected_rounds:
                snapshot('unprotected')
            if self.verbose and (step == 1 or step % 20 == 0 or stop):
                print(f"{stage} {step}: change={outer_residual:.3g}, PG={pg_residual:.3g}, selected={support.sum()}, ESS/N={row['ess'] / N:.3f}", flush=True)
            if stop:
                (self.converged_, self.stop_reason_) = (True, 'full-stage fixed-point criteria met')
                break
            if t < total_rounds:
                prior = prior_next
                hyper = DWBLRHyperParams(pi0=float(prior.mean()), sigma0_sq=sigma_next)
        if not self.unprotected_rounds:
            self.phase_snapshots_['unprotected'] = self.phase_snapshots_['warmup'].copy()
        snapshot('full')
        self.selection_history_ = np.asarray(full_support, dtype=bool)
        self.selection_frequency_ = self.selection_history_[-self.stability_window:].mean(axis=0)
        self.convergence_diagnostics_ = dict(self.history_[-1], converged=self.converged_, stop_reason=self.stop_reason_)
        selected = np.flatnonzero(vp.alpha > self.tau)
        self.result_ = DWBLRResult(selected_features=selected, selected_feature_names=None if self.feature_names_ is None else [self.feature_names_[j] for j in selected], weights=W.copy(), alpha=vp.alpha.copy(), mu=vp.mu.copy(), s2=vp.s2.copy(), intercept_mu=ip.mu_b, intercept_var=ip.v_b, sigma_b_sq=self.sigma_b_sq, rho_bar=rho.copy(), eta_bar=W * rho, Lambda=Lambda.copy(), hyper=hyper.copy(), feature_prior=prior.copy(), feature_prior_update=f'{self.prior_mode}_inner_eb' if self.prior_update_scope == 'inner' else f'{self.prior_mode}_{self.prior_update}', n_outer_iterations=t, history=self.history_)
        if not self.converged_:
            warnings.warn('DWBLR reached T_max before all fixed-point criteria were met; inspect convergence_diagnostics_', RuntimeWarning)
        return self

    def get_stable_selected_features(self, min_frequency=0.9):
        self._check_fitted()
        if not 0 < min_frequency <= 1:
            raise ValueError('min_frequency must be in (0,1]')
        return np.flatnonzero((self.get_pip() > self.tau) & (self.selection_frequency_ >= min_frequency))

def DWBLR(X: Any, y: Any, *, optimization_mode: str='feature_em', prior_update_scope: Optional[str]=None, feature_names: Optional[Sequence[str]]=None, **kwargs: Any) -> DWBLRStagedLogistic:
    if optimization_mode != 'feature_em':
        raise ValueError("optimization_mode must be 'feature_em'")
    scope = 'outer' if prior_update_scope is None else prior_update_scope
    if scope not in {'inner', 'outer'}:
        raise ValueError('prior_update_scope must be inner or outer')
    if 'prior_mode' in kwargs and kwargs.pop('prior_mode') != 'feature':
        raise ValueError('prior_mode conflicts with optimization_mode')
    if 'prior_update' in kwargs and kwargs.pop('prior_update') != 'outer_eb':
        raise ValueError('prior_update conflicts with optimization_mode')
    model = DWBLRStagedLogistic(prior_mode='feature', prior_update='outer_eb', prior_update_scope=scope, **kwargs)
    model.optimization_mode_ = 'feature_em'
    return model.fit(X, y, feature_names=feature_names)

def check_decorrelator_gradient(X: np.ndarray, Lambda: np.ndarray, W: np.ndarray, rho_bar: np.ndarray, *, gamma_w: float=1.0, decorrelation_scale: float=1.0, covariance_mode: Literal['precision', 'normalized']='normalized', finite_diff_eps: float=1e-06) -> Dict[str, float]:
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float).copy()
    rho_bar = np.asarray(rho_bar, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    (f0, g) = CurvatureDecorrelator.loss_and_grad(X, Lambda, W, rho_bar, gamma_w=gamma_w, decorrelation_scale=decorrelation_scale, covariance_mode=covariance_mode)
    g_fd = np.zeros_like(W)
    h = float(finite_diff_eps)
    if h <= 0:
        raise ValueError('finite_diff_eps must be > 0.')
    for i in range(len(W)):
        wp = W.copy()
        wm = W.copy()
        wp[i] += h
        wm[i] -= h
        if wm[i] <= 0:
            fp = CurvatureDecorrelator.loss(X, Lambda, wp, rho_bar, gamma_w=gamma_w, decorrelation_scale=decorrelation_scale, covariance_mode=covariance_mode)
            g_fd[i] = (fp - f0) / h
        else:
            fp = CurvatureDecorrelator.loss(X, Lambda, wp, rho_bar, gamma_w=gamma_w, decorrelation_scale=decorrelation_scale, covariance_mode=covariance_mode)
            fm = CurvatureDecorrelator.loss(X, Lambda, wm, rho_bar, gamma_w=gamma_w, decorrelation_scale=decorrelation_scale, covariance_mode=covariance_mode)
            g_fd[i] = (fp - fm) / (2.0 * h)
    abs_err = float(np.max(np.abs(g - g_fd)))
    rel_err = float(abs_err / (np.max(np.abs(g_fd)) + 1e-12))
    return {'loss': float(f0), 'max_abs_error': abs_err, 'max_relative_error': rel_err}
__all__ = ['PGVariationalParams', 'InterceptVariationalParams', 'DWBLRHyperParams', 'DWBLRResult', 'CurvatureDecorrelator', 'PGMFVI', 'DWBLRLogistic', 'DWBLR', 'update_penalty_matrix', 'check_decorrelator_gradient']
