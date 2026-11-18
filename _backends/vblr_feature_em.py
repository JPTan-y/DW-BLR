
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
from scipy.special import expit


@dataclass
class VBLRVariationalParams:
    alpha: np.ndarray
    mu: np.ndarray
    s2: np.ndarray

    def copy(self) -> "VBLRVariationalParams":
        return VBLRVariationalParams(
            alpha=self.alpha.copy(),
            mu=self.mu.copy(),
            s2=self.s2.copy(),
        )

    @property
    def posterior_mean(self) -> np.ndarray:
        return self.alpha * self.mu

    @property
    def second_moment(self) -> np.ndarray:
        return self.alpha * (self.mu ** 2 + self.s2)

    @property
    def posterior_variance(self) -> np.ndarray:
        mean = self.posterior_mean
        return np.maximum(self.second_moment - mean ** 2, 0.0)


@dataclass
class VBLRInterceptParams:
    mu_b: float
    v_b: float

    def copy(self) -> "VBLRInterceptParams":
        return VBLRInterceptParams(float(self.mu_b), float(self.v_b))

    @property
    def sd_b(self) -> float:
        return float(np.sqrt(max(self.v_b, 0.0)))


@dataclass
class VBLRHyperParams:
    pi0: float
    sigma0_sq: float
    feature_prior: Optional[np.ndarray] = None

    def copy(self) -> "VBLRHyperParams":
        return VBLRHyperParams(float(self.pi0), float(self.sigma0_sq),
                              None if self.feature_prior is None else self.feature_prior.copy())


@dataclass
class VBLRResult:
    selected_features: np.ndarray
    selected_feature_names: Optional[List[str]]

    alpha: np.ndarray
    mu: np.ndarray
    s2: np.ndarray

    intercept_mu: float
    intercept_var: float
    sigma_b_sq: float

    c: np.ndarray
    rho_bar: np.ndarray

    hyper: VBLRHyperParams
    n_iterations: int
    converged: bool
    final_change: float
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


def _to_numpy_2d(X: Any, name: str = "X") -> np.ndarray:
    if hasattr(X, "to_numpy"):
        arr = np.asarray(X.to_numpy(), dtype=np.float64)
    else:
        arr = np.asarray(X, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix; got shape={arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one variable.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr


def _to_binary_y(y: Any, n: int) -> np.ndarray:
    if hasattr(y, "to_numpy"):
        arr = np.asarray(y.to_numpy()).ravel()
    else:
        arr = np.asarray(y).ravel()

    if arr.shape != (n,):
        raise ValueError(f"Length of y must equal the sample size n={n}; got shape={arr.shape}.")

    try:
        arr_float = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must be convertible to numeric 0/1 or -1/+1 labels.") from exc

    if not np.all(np.isfinite(arr_float)):
        raise ValueError("y contains NaN or Inf.")

    unique = np.unique(arr_float)
    if np.all(np.isin(unique, [-1.0, 1.0])):
        arr_float = (arr_float == 1.0).astype(np.float64)
    elif np.all(np.isin(unique, [0.0, 1.0])):
        arr_float = arr_float.astype(np.float64)
    else:
        raise ValueError(
            "y must be coded as {0,1} or {-1,+1}; "
            f"observed labels={unique.tolist()}."
        )

    if np.unique(arr_float).size != 2:
        raise ValueError("y must contain both classes.")

    return arr_float


def _safe_standard_deviation(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scale = np.std(X, axis=0, ddof=0)
    zero_mask = scale < 1e-12
    scale = scale.copy()
    scale[zero_mask] = 1.0
    return scale, zero_mask


def _max_abs_change(new: np.ndarray, old: np.ndarray) -> float:
    if new.size == 0:
        return 0.0
    return float(np.max(np.abs(new - old)))


class PolyaGammaVariationalSolver:
    @staticmethod
    def update_pg_moments(
        X: np.ndarray,
        r: np.ndarray,
        m: np.ndarray,
        t_second: np.ndarray,
        intercept_var: float,
        delta_c: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        beta_var = np.maximum(t_second - m ** 2, 0.0)
        c2 = r ** 2 + max(float(intercept_var), 0.0) + (X ** 2) @ beta_var
        c2 = np.maximum(c2, 0.0)
        c = np.sqrt(c2)

        rho = np.empty_like(c)
        small = c < float(delta_c)
        rho[small] = 0.25

        large = ~small
        if np.any(large):
            cl = c[large]
            rho[large] = np.tanh(cl / 2.0) / (2.0 * cl)

        rho = np.clip(rho, 0.0, 0.25)
        return c, rho

    @staticmethod
    def fit(
        X: np.ndarray,
        y: np.ndarray,
        *,
        alpha_init: Optional[np.ndarray],
        prior_inclusion_prob: Optional[np.ndarray],
        feature_prior_update: str = "fixed",
        sigma_b_sq: float,
        pi0_init: float,
        sigma0_sq_init: float,
        sigma0_sq_max: Optional[float],
        empirical_bayes: bool,
        max_iter: int,
        tol: float,
        delta_c: float,
        delta_h: float,
        tau: float,
        verbose: bool,
        log_every: int,
    ) -> Tuple[
        VBLRVariationalParams,
        VBLRInterceptParams,
        VBLRHyperParams,
        np.ndarray,
        np.ndarray,
        List[Dict[str, float]],
        bool,
        float,
    ]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix.")
        n, p = X.shape
        if y.shape != (n,):
            raise ValueError("Length of y must equal the sample size n.")
        if sigma_b_sq <= 0 or not np.isfinite(sigma_b_sq):
            raise ValueError("sigma_b_sq must be a finite positive number.")
        if not (0.0 < pi0_init < 1.0):
            raise ValueError("pi0_init must lie in (0,1).")
        if sigma0_sq_init <= 0 or not np.isfinite(sigma0_sq_init):
            raise ValueError("sigma0_sq_init must be a finite positive number.")
        if sigma0_sq_max is not None:
            if sigma0_sq_max <= 0 or not np.isfinite(sigma0_sq_max):
                raise ValueError("sigma0_sq_max must be a finite positive number or None.")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0.")
        if tol <= 0 or delta_c <= 0 or delta_h <= 0:
            raise ValueError("tol, delta_c and delta_h must be positive.")
        if not (0.0 < delta_h < 0.5):
            raise ValueError("delta_h must lie in (0,0.5) so that the PIP clipping interval is valid.")
        if log_every <= 0:
            raise ValueError("log_every must be a positive integer.")

        kappa = y - 0.5

        hyper = VBLRHyperParams(
            pi0=float(np.clip(pi0_init, delta_h, 1.0 - delta_h)),
            sigma0_sq=float(max(sigma0_sq_init, delta_h)),
        )

        if feature_prior_update not in {"fixed", "em"}:
            raise ValueError('feature_prior_update must be fixed or em.')
        if feature_prior_update == "em" and not empirical_bayes:
            raise ValueError('Per-variable EM requires empirical_bayes=True.')
        if feature_prior_update == "em" and prior_inclusion_prob is None:
            prior_inclusion_prob = np.full(p, pi0_init, dtype=np.float64)
        if prior_inclusion_prob is None:
            feature_prior = None
        else:
            feature_prior = np.asarray(prior_inclusion_prob, dtype=np.float64).ravel()
            if feature_prior.shape != (p,):
                raise ValueError(f"Length of prior_inclusion_prob must equal p={p}.")
            if not np.all(np.isfinite(feature_prior)):
                raise ValueError("prior_inclusion_prob contains NaN or Inf.")
            if np.any(feature_prior <= 0.0) or np.any(feature_prior >= 1.0):
                raise ValueError("Every element of prior_inclusion_prob must lie in (0,1).")
            feature_prior = np.clip(feature_prior, delta_h, 1.0 - delta_h)
            hyper.pi0 = float(np.mean(feature_prior))

        if alpha_init is None:
            alpha = np.full(p, hyper.pi0, dtype=np.float64)
        else:
            alpha_arr = np.asarray(alpha_init, dtype=np.float64).ravel()
            if alpha_arr.shape != (p,):
                raise ValueError(f"Length of alpha_init must equal p={p}.")
            if not np.all(np.isfinite(alpha_arr)):
                raise ValueError("alpha_init contains NaN or Inf.")
            if np.any(alpha_arr <= 0.0) or np.any(alpha_arr >= 1.0):
                raise ValueError("Every element of alpha_init must lie in (0,1).")
            alpha = np.clip(alpha_arr, delta_h, 1.0 - delta_h)
        mu = np.zeros(p, dtype=np.float64)
        s2 = np.full(p, hyper.sigma0_sq, dtype=np.float64)

        mu_b = 0.0
        v_b = float(sigma_b_sq)

        m = alpha * mu
        t_second = alpha * (mu ** 2 + s2)
        r = mu_b + X @ m

        c = np.zeros(n, dtype=np.float64)
        rho_bar = np.full(n, 0.25, dtype=np.float64)

        history: List[Dict[str, float]] = []
        converged = False
        last_change = np.inf
        used_iter = 0

        for iteration in range(1, int(max_iter) + 1):
            used_iter = iteration

            alpha_old = alpha.copy()
            mu_old = mu.copy()
            s2_old = s2.copy()
            mu_b_old_global = float(mu_b)
            v_b_old = float(v_b)
            pi0_old = float(hyper.pi0)
            sigma0_old = float(hyper.sigma0_sq)
            feature_prior_old = None if feature_prior is None else feature_prior.copy()

            c, rho_bar = PolyaGammaVariationalSolver.update_pg_moments(
                X=X,
                r=r,
                m=m,
                t_second=t_second,
                intercept_var=v_b,
                delta_c=delta_c,
            )

            mu_b_before = float(mu_b)
            A_b = float(np.sum(rho_bar))
            residual_without_intercept = r - mu_b_before
            B_b = float(np.sum(kappa - rho_bar * residual_without_intercept))

            intercept_precision = A_b + 1.0 / float(sigma_b_sq)
            v_b = 1.0 / max(intercept_precision, delta_h)
            mu_b = v_b * B_b
            r += mu_b - mu_b_before

            for j in range(p):
                xj = X[:, j]
                m_old_j = float(m[j])

                A_j = float(np.dot(rho_bar, xj ** 2))
                residual_without_j = r - xj * m_old_j
                B_j = float(np.dot(xj, kappa - rho_bar * residual_without_j))

                slab_variance = max(float(hyper.sigma0_sq), delta_h)
                precision = A_j + 1.0 / slab_variance
                s2_j = 1.0 / max(precision, delta_h)
                mu_j = s2_j * B_j

                pi0_clip = float(
                    feature_prior[j]
                    if feature_prior is not None
                    else np.clip(hyper.pi0, delta_h, 1.0 - delta_h)
                )
                logit_prior = np.log(pi0_clip) - np.log1p(-pi0_clip)

                z_signal = mu_j / np.sqrt(max(s2_j, delta_h))
                signal_term = 0.5 * z_signal * z_signal
                if not np.isfinite(signal_term):
                    signal_term = 1e6

                log_odds = (
                    logit_prior
                    + 0.5 * np.log(max(s2_j, delta_h) / slab_variance)
                    + signal_term
                )
                alpha_j = float(expit(np.clip(log_odds, -700.0, 700.0)))
                alpha_j = float(np.clip(alpha_j, delta_h, 1.0 - delta_h))

                s2[j] = s2_j
                mu[j] = mu_j
                alpha[j] = alpha_j

                m_new_j = alpha_j * mu_j
                t_new_j = alpha_j * (mu_j ** 2 + s2_j)

                r += xj * (m_new_j - m_old_j)
                m[j] = m_new_j
                t_second[j] = t_new_j

            if empirical_bayes:
                if feature_prior_update == "em":
                    feature_prior = np.clip(alpha.copy(), delta_h, 1.0 - delta_h)
                pi0_new = (
                    float(np.mean(feature_prior))
                    if feature_prior is not None
                    else float(np.clip(np.mean(alpha), delta_h, 1.0 - delta_h))
                )
                denominator = max(float(np.sum(alpha)), delta_h)
                sigma0_new = float(np.sum(alpha * (mu ** 2 + s2)) / denominator)
                sigma0_new = max(sigma0_new, delta_h)
                if sigma0_sq_max is not None:
                    sigma0_new = min(sigma0_new, float(sigma0_sq_max))

                hyper.pi0 = pi0_new
                hyper.sigma0_sq = sigma0_new

            last_change = max(
                _max_abs_change(alpha, alpha_old),
                _max_abs_change(mu, mu_old),
                _max_abs_change(s2, s2_old),
                abs(float(mu_b) - mu_b_old_global),
                abs(float(v_b) - v_b_old),
                abs(float(hyper.pi0) - pi0_old),
                abs(float(hyper.sigma0_sq) - sigma0_old),
                0.0 if feature_prior is None else _max_abs_change(feature_prior, feature_prior_old),
            )

            selected_now = int(np.sum(alpha > tau))
            history.append(
                {
                    "iteration": float(iteration),
                    "max_parameter_change": float(last_change),
                    "intercept_mu": float(mu_b),
                    "intercept_var": float(v_b),
                    "pi0": float(hyper.pi0),
                    "sigma0_sq": float(hyper.sigma0_sq),
                    "mean_pip": float(np.mean(alpha)),
                    "min_pip": float(np.min(alpha)),
                    "max_pip": float(np.max(alpha)),
                    "n_selected": float(selected_now),
                    "feature_prior_min": float(hyper.pi0 if feature_prior is None else np.min(feature_prior)),
                    "feature_prior_max": float(hyper.pi0 if feature_prior is None else np.max(feature_prior)),
                    "feature_prior_max_change": float(0.0 if feature_prior is None else _max_abs_change(feature_prior, feature_prior_old)),
                    "mean_c": float(np.mean(c)),
                    "mean_rho": float(np.mean(rho_bar)),
                }
            )

            if verbose and (iteration == 1 or iteration % int(log_every) == 0):
                top_idx = np.argsort(alpha)[-min(5, p):][::-1]
                top_pip = [(int(j), round(float(alpha[j]), 4)) for j in top_idx]
                print(
                    f"  VBLR iter={iteration:4d} | change={last_change:.3e} | "
                    f"b0={mu_b:.4f} | pi0={hyper.pi0:.4f} | "
                    f"sigma0^2={hyper.sigma0_sq:.4e} | "
                    f"selected={selected_now} | TopPIP={top_pip}"
                )

            if not np.isfinite(last_change):
                raise FloatingPointError("VBLR variational update produced a non-finite parameter change.")

            if last_change < tol:
                converged = True
                break

        m = alpha * mu
        t_second = alpha * (mu ** 2 + s2)
        r = mu_b + X @ m
        c, rho_bar = PolyaGammaVariationalSolver.update_pg_moments(
            X=X,
            r=r,
            m=m,
            t_second=t_second,
            intercept_var=v_b,
            delta_c=delta_c,
        )

        finite_ok = (
            np.all(np.isfinite(alpha))
            and np.all(np.isfinite(mu))
            and np.all(np.isfinite(s2))
            and np.all(np.isfinite(c))
            and np.all(np.isfinite(rho_bar))
            and np.isfinite(mu_b)
            and np.isfinite(v_b)
            and np.isfinite(hyper.pi0)
            and np.isfinite(hyper.sigma0_sq)
        )
        if not finite_ok:
            raise FloatingPointError("VBLR final variational parameters contain NaN or Inf.")
        if np.any(s2 <= 0) or v_b <= 0:
            raise FloatingPointError("VBLR final posterior variances must be strictly positive.")
        if np.any(alpha <= 0) or np.any(alpha >= 1):
            raise FloatingPointError("VBLR final PIPs must lie in (0,1).")
        if np.any(rho_bar < 0) or np.any(rho_bar > 0.25 + 1e-12):
            raise FloatingPointError("VBLR PG first moment is outside the theoretical range [0,1/4].")

        hyper.feature_prior = None if feature_prior is None else feature_prior.copy()
        variational = VBLRVariationalParams(alpha=alpha, mu=mu, s2=s2)
        intercept = VBLRInterceptParams(mu_b=float(mu_b), v_b=float(v_b))

        return (
            variational,
            intercept,
            hyper,
            c,
            rho_bar,
            history,
            converged,
            float(last_change),
        )


class VBLRModel:
    def __init__(
        self,
        *,
        tau: float = 0.5,
        sigma_b_sq: float = 100.0,
        pi0_init: float = 0.5,
        sigma0_sq_init: float = 1.0,
        alpha_init: Optional[Sequence[float]] = None,
        prior_inclusion_prob: Optional[Sequence[float]] = None,
        feature_prior_update: str = "fixed",
        sigma0_sq_max: Optional[float] = None,
        empirical_bayes: bool = True,
        max_iter: int = 200,
        tol: float = 1e-6,
        delta_c: float = 1e-8,
        delta_h: float = 1e-6,
        standardize: bool = True,
        verbose: bool = True,
        log_every: int = 20,
    ):
        if not (0.0 < tau < 1.0):
            raise ValueError("tau must lie in (0,1).")
        if sigma_b_sq <= 0 or not np.isfinite(sigma_b_sq):
            raise ValueError("sigma_b_sq must be a finite positive number.")
        if not (0.0 < pi0_init < 1.0):
            raise ValueError("pi0_init must lie in (0,1).")
        if sigma0_sq_init <= 0 or not np.isfinite(sigma0_sq_init):
            raise ValueError("sigma0_sq_init must be a finite positive number.")
        if sigma0_sq_max is not None and (
            sigma0_sq_max <= 0 or not np.isfinite(sigma0_sq_max)
        ):
            raise ValueError("sigma0_sq_max must be a finite positive number or None.")
        if max_iter <= 0 or tol <= 0 or delta_c <= 0 or delta_h <= 0:
            raise ValueError("max_iter must be > 0; tol, delta_c and delta_h must be positive.")
        if not (0.0 < delta_h < 0.5):
            raise ValueError("delta_h must lie in (0,0.5) so that the PIP clipping interval is valid.")
        if log_every <= 0:
            raise ValueError("log_every must be a positive integer.")

        self.tau = float(tau)
        self.sigma_b_sq = float(sigma_b_sq)
        self.pi0_init = float(pi0_init)
        self.sigma0_sq_init = float(sigma0_sq_init)
        self.alpha_init = None if alpha_init is None else np.asarray(alpha_init, dtype=float).ravel()
        self.prior_inclusion_prob = (
            None
            if prior_inclusion_prob is None
            else np.asarray(prior_inclusion_prob, dtype=float).ravel()
        )
        self.sigma0_sq_max = sigma0_sq_max
        self.empirical_bayes = bool(empirical_bayes)
        if feature_prior_update not in {"fixed", "em"}:
            raise ValueError('feature_prior_update must be fixed or em.')
        if feature_prior_update == "em" and not self.empirical_bayes:
            raise ValueError('Per-variable EM requires empirical_bayes=True.')
        self.feature_prior_update = feature_prior_update
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.delta_c = float(delta_c)
        self.delta_h = float(delta_h)
        self.standardize = bool(standardize)
        self.verbose = bool(verbose)
        self.log_every = int(log_every)

        self.result_: Optional[VBLRResult] = None
        self.feature_names_: Optional[List[str]] = None
        self.X_mean_: Optional[np.ndarray] = None
        self.X_scale_: Optional[np.ndarray] = None
        self.zero_variance_mask_: Optional[np.ndarray] = None

    def _prepare_X_fit(self, X: Any) -> np.ndarray:
        if hasattr(X, "columns") and hasattr(X, "to_numpy"):
            self.feature_names_ = [str(c) for c in list(X.columns)]
        else:
            self.feature_names_ = None

        X_arr = _to_numpy_2d(X)
        p = X_arr.shape[1]

        if self.standardize:
            mean = np.mean(X_arr, axis=0)
            scale, zero_mask = _safe_standard_deviation(X_arr)
            if np.any(zero_mask):
                warnings.warn(
                    f"Detected {int(np.sum(zero_mask))} near-zero-variance variables; their scale is set to 1.",
                    RuntimeWarning,
                )
            Xs = (X_arr - mean) / scale
        else:
            mean = np.zeros(p, dtype=np.float64)
            scale = np.ones(p, dtype=np.float64)
            zero_mask = np.zeros(p, dtype=bool)
            Xs = X_arr.copy()

        self.X_mean_ = mean
        self.X_scale_ = scale
        self.zero_variance_mask_ = zero_mask
        return Xs

    def _prepare_X_predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X_arr = _to_numpy_2d(X)
        if X_arr.shape[1] != len(self.result_.alpha):
            raise ValueError("The number of features in the new X does not match the training data.")
        return (X_arr - self.X_mean_) / self.X_scale_

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "VBLRModel":
        Xs = self._prepare_X_fit(X)
        n, p = Xs.shape
        ys = _to_binary_y(y, n)

        if feature_names is not None:
            if len(feature_names) != p:
                raise ValueError("Length of feature_names must equal the number of features p.")
            self.feature_names_ = [str(value) for value in feature_names]

        if self.verbose:
            print(
                f"[*] VBLR Start: n={n}, p={p}, tau={self.tau}, "
                f"sigma_b^2={self.sigma_b_sq}, empirical_bayes={self.empirical_bayes}"
            )

        (
            variational,
            intercept,
            hyper,
            c,
            rho_bar,
            history,
            converged,
            final_change,
        ) = PolyaGammaVariationalSolver.fit(
            X=Xs,
            y=ys,
            alpha_init=self.alpha_init,
            prior_inclusion_prob=self.prior_inclusion_prob,
            feature_prior_update=self.feature_prior_update,
            sigma_b_sq=self.sigma_b_sq,
            pi0_init=self.pi0_init,
            sigma0_sq_init=self.sigma0_sq_init,
            sigma0_sq_max=self.sigma0_sq_max,
            empirical_bayes=self.empirical_bayes,
            max_iter=self.max_iter,
            tol=self.tol,
            delta_c=self.delta_c,
            delta_h=self.delta_h,
            tau=self.tau,
            verbose=self.verbose,
            log_every=self.log_every,
        )

        selected = np.flatnonzero(variational.alpha > self.tau)
        selected_names = None
        if self.feature_names_ is not None:
            selected_names = [self.feature_names_[j] for j in selected]

        self.result_ = VBLRResult(
            selected_features=selected.astype(int),
            selected_feature_names=selected_names,
            alpha=variational.alpha.copy(),
            mu=variational.mu.copy(),
            s2=variational.s2.copy(),
            intercept_mu=float(intercept.mu_b),
            intercept_var=float(intercept.v_b),
            sigma_b_sq=float(self.sigma_b_sq),
            c=c.copy(),
            rho_bar=rho_bar.copy(),
            hyper=hyper.copy(),
            n_iterations=len(history),
            converged=bool(converged),
            final_change=float(final_change),
            history=list(history),
        )

        if self.verbose:
            status = "converged" if converged else "reached max_iter"
            print(
                f"[✓] VBLR finished: {status} | iterations={len(history)} | "
                f"change={final_change:.3e} | selected={len(selected)}/{p}"
            )

        return self

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise ValueError("The model is not fitted; call fit(X,y) first.")


    def get_pip(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.alpha.copy()

    def get_selected_features(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.selected_features.copy()

    def get_selected_feature_names(self) -> Optional[List[str]]:
        self._check_fitted()
        if self.result_.selected_feature_names is None:
            return None
        return list(self.result_.selected_feature_names)

    def get_slab_mean(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.mu.copy()

    def get_slab_variance(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.s2.copy()

    def get_posterior_mean_coef(self, original_scale: bool = False) -> np.ndarray:
        self._check_fitted()
        beta = self.result_.posterior_mean_beta.copy()
        if original_scale:
            beta = beta / self.X_scale_
        return beta

    def get_posterior_variance_coef(self, original_scale: bool = False) -> np.ndarray:
        self._check_fitted()
        variance = self.result_.posterior_variance_beta.copy()
        if original_scale:
            variance = variance / (self.X_scale_ ** 2)
        return variance

    def get_intercept(self, original_scale: bool = False) -> float:
        self._check_fitted()
        if not original_scale:
            return float(self.result_.intercept_mu)
        beta_original = self.get_posterior_mean_coef(original_scale=True)
        return float(self.result_.intercept_mu - np.dot(self.X_mean_, beta_original))

    def get_intercept_variance(self, original_scale: bool = False) -> float:
        self._check_fitted()
        if not original_scale:
            return float(self.result_.intercept_var)

        transform = self.X_mean_ / self.X_scale_
        extra = float(
            np.sum((transform ** 2) * self.result_.posterior_variance_beta)
        )
        return float(self.result_.intercept_var + extra)

    def get_intercept_posterior(self) -> Dict[str, float]:
        self._check_fitted()
        original_variance = self.get_intercept_variance(original_scale=True)
        return {
            "posterior_mean": float(self.result_.intercept_mu),
            "posterior_variance": float(self.result_.intercept_var),
            "posterior_sd": float(self.result_.intercept_sd),
            "prior_variance_sigma_b_sq": float(self.result_.sigma_b_sq),
            "original_scale_posterior_mean": float(
                self.get_intercept(original_scale=True)
            ),
            "original_scale_posterior_variance_approx": float(original_variance),
            "original_scale_posterior_sd_approx": float(
                np.sqrt(max(original_variance, 0.0))
            ),
        }

    def get_hyperparameters(self) -> Dict[str, float]:
        self._check_fitted()
        return {
            "pi0": float(self.result_.hyper.pi0),
            "sigma0_sq": float(self.result_.hyper.sigma0_sq),
            "sigma_b_sq": float(self.result_.sigma_b_sq),
        }

    def get_feature_prior(self) -> np.ndarray:
        self._check_fitted()
        prior = self.result_.hyper.feature_prior
        return (np.full(len(self.result_.alpha), self.result_.hyper.pi0)
                if prior is None else prior.copy())

    def get_pg_c(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.c.copy()

    def get_pg_curvature(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.rho_bar.copy()

    def get_diagnostics(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "n_iterations": int(self.result_.n_iterations),
            "converged": bool(self.result_.converged),
            "final_max_parameter_change": float(self.result_.final_change),
            "n_selected": int(len(self.result_.selected_features)),
            "mean_pip": float(np.mean(self.result_.alpha)),
            "min_pip": float(np.min(self.result_.alpha)),
            "max_pip": float(np.max(self.result_.alpha)),
            "intercept_mean": float(self.result_.intercept_mu),
            "intercept_variance": float(self.result_.intercept_var),
            "pi0": float(self.result_.hyper.pi0),
            "sigma0_sq": float(self.result_.hyper.sigma0_sq),
            "pg_c_mean": float(np.mean(self.result_.c)),
            "pg_c_max": float(np.max(self.result_.c)),
            "pg_rho_mean": float(np.mean(self.result_.rho_bar)),
            "pg_rho_min": float(np.min(self.result_.rho_bar)),
            "pg_rho_max": float(np.max(self.result_.rho_bar)),
            "sample_weights_used": False,
            "feature_prior_update": self.feature_prior_update,
            "feature_prior_min": float(np.min(self.get_feature_prior())),
            "feature_prior_max": float(np.max(self.get_feature_prior())),
            "decorrelation_used": False,
            "outer_alternating_loop_used": False,
        }

    def get_convergence_info(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "n_iter": int(self.result_.n_iterations),
            "converged": bool(self.result_.converged),
            "max_param_change": float(self.result_.final_change),
        }


    def decision_function(self, X: Any) -> np.ndarray:
        Xs = self._prepare_X_predict(X)
        return self.result_.intercept_mu + Xs @ self.result_.posterior_mean_beta

    def predict_proba(self, X: Any) -> np.ndarray:
        probability_one = expit(self.decision_function(X))
        return np.column_stack([1.0 - probability_one, probability_one])

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must lie in (0,1).")
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


    def posterior_summary(self, sort_by_pip: bool = True):
        self._check_fitted()
        alpha = self.result_.alpha
        mu = self.result_.mu
        s2 = self.result_.s2
        beta_mean = self.result_.posterior_mean_beta
        beta_var = self.result_.posterior_variance_beta

        rows: List[Dict[str, Any]] = []
        for j in range(len(alpha)):
            rows.append(
                {
                    "feature_index": int(j),
                    "feature_name": (
                        None if self.feature_names_ is None else self.feature_names_[j]
                    ),
                    "PIP_alpha": float(alpha[j]),
                    "slab_mu": float(mu[j]),
                    "slab_sd": float(np.sqrt(max(s2[j], 0.0))),
                    "posterior_mean_beta": float(beta_mean[j]),
                    "posterior_sd_beta": float(np.sqrt(max(beta_var[j], 0.0))),
                    "selected": bool(alpha[j] > self.tau),
                }
            )

        if sort_by_pip:
            rows.sort(key=lambda item: item["PIP_alpha"], reverse=True)

        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def intercept_summary(self):
        self._check_fitted()
        original_variance = self.get_intercept_variance(original_scale=True)
        row = {
            "term": "intercept",
            "posterior_mean_standardized": float(self.result_.intercept_mu),
            "posterior_variance_standardized": float(self.result_.intercept_var),
            "posterior_sd_standardized": float(self.result_.intercept_sd),
            "posterior_mean_original_scale": float(
                self.get_intercept(original_scale=True)
            ),
            "posterior_variance_original_scale_approx": float(original_variance),
            "posterior_sd_original_scale_approx": float(
                np.sqrt(max(original_variance, 0.0))
            ),
            "prior_variance_sigma_b_sq": float(self.result_.sigma_b_sq),
            "always_included": True,
        }

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


def VBLR(
    X: Any,
    y: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    tau: float = 0.5,
    sigma_b_sq: float = 100.0,
    pi0_init: float = 0.5,
    sigma0_sq_init: float = 1.0,
    alpha_init: Optional[Sequence[float]] = None,
    prior_inclusion_prob: Optional[Sequence[float]] = None,
    feature_prior_update: str = "fixed",
    sigma0_sq_max: Optional[float] = None,
    empirical_bayes: bool = True,
    max_iter: int = 200,
    tol: float = 1e-6,
    delta_c: float = 1e-8,
    delta_h: float = 1e-6,
    standardize: bool = True,
    verbose: bool = True,
    log_every: int = 20,
) -> VBLRModel:
    model = VBLRModel(
        tau=tau,
        sigma_b_sq=sigma_b_sq,
        pi0_init=pi0_init,
        sigma0_sq_init=sigma0_sq_init,
        alpha_init=alpha_init,
        prior_inclusion_prob=prior_inclusion_prob,
        feature_prior_update=feature_prior_update,
        sigma0_sq_max=sigma0_sq_max,
        empirical_bayes=empirical_bayes,
        max_iter=max_iter,
        tol=tol,
        delta_c=delta_c,
        delta_h=delta_h,
        standardize=standardize,
        verbose=verbose,
        log_every=log_every,
    )
    return model.fit(X, y, feature_names=feature_names)


__all__ = [
    "VBLRVariationalParams",
    "VBLRInterceptParams",
    "VBLRHyperParams",
    "VBLRResult",
    "PolyaGammaVariationalSolver",
    "VBLRModel",
    "VBLR",
]


