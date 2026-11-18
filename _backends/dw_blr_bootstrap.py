from __future__ import annotations
import csv
import hashlib
import json
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from . import dw_blr_variational as _core
if Path(_core.__file__).resolve() != Path(__file__).with_name('dw_blr_variational.py').resolve():
    raise ImportError('Inconsistent dw_blr_variational module path; restart the Python kernel and import from this project directory.')
__all__ = ['DWBLREstimator', 'PilotConvergenceError', 'fit_unprotected_pilot', 'build_frequency_prior', 'make_patient_subsamples']

def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return int(value)

def _unit(value, name, *, include_zero=False, include_one=True):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if include_zero else value <= 0) or (value > 1 if include_one else value >= 1):
        raise ValueError(f'{name} is outside the allowed probability interval')
    return value

def _xy(X, y):
    (x_index, y_index) = (getattr(X, 'index', None), getattr(y, 'index', None))
    if hasattr(x_index, 'equals') and hasattr(y_index, 'equals'):
        if not x_index.equals(y_index):
            raise ValueError('pandas index/row order of X and y do not match; align them by sample_id first')
    x = np.asarray(X, dtype=float)
    yy = np.asarray(y, dtype=float)
    if yy.ndim == 2 and yy.shape[1] == 1:
        yy = yy[:, 0]
    if x.ndim != 2 or x.shape[1] < 2 or yy.ndim != 1 or (len(yy) != len(x)):
        raise ValueError('Requires an n×p X (p>=2) and a row-aligned one-dimensional y')
    if not np.isfinite(x).all() or not np.isfinite(yy).all():
        raise ValueError('X/y contains NaN or Inf')
    if set(np.unique(yy)) != {0.0, 1.0}:
        raise ValueError('y must contain both 0 and 1')
    if np.any(x.std(axis=0) < 1e-12):
        raise ValueError('Near-zero-variance genes are present; columns are not dropped silently and non-evaluable genes are not recorded as unselected')
    return (x.copy(), yy.copy())

def make_patient_subsamples(y, n_subsamples=20, fraction=0.8, stratify=True, random_state=20260914, *, replace=False):
    k = _integer(n_subsamples, 'n_subsamples')
    fraction = _unit(fraction, 'subsample_fraction')
    if not isinstance(stratify, (bool, np.bool_)):
        raise ValueError('subsample_stratify must be a boolean')
    if not isinstance(replace, (bool, np.bool_)):
        raise ValueError('replace must be a boolean')
    replace = bool(replace)
    random_state = _integer(random_state, 'random_state', 0)
    y = np.asarray(y)
    if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError('Sampling labels must be one-dimensional and contain both 0 and 1')
    n = len(y)
    m = int(np.floor(n * fraction))
    if m < 4:
        raise ValueError('Each subsample must contain at least 4 observations')
    groups = [np.flatnonzero(y == c) for c in (0, 1)]
    m1 = int(np.floor(m * len(groups[1]) / n + 0.5))
    m0 = m - m1
    if stratify and (min(m0, m1) < 1 or (not replace and (m1 > len(groups[1]) or m0 > len(groups[0])))):
        raise ValueError('This subsampling fraction cannot guarantee patients from both classes; increase the fraction')
    seeds = np.random.SeedSequence(random_state).spawn(k)
    draws = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if stratify:
            idx = np.concatenate([rng.choice(groups[0], m0, replace=replace), rng.choice(groups[1], m1, replace=replace)])
        else:
            idx = rng.choice(n, m, replace=replace)
        draws.append(np.sort(idx))
    return np.asarray(draws, dtype=int)

def _prior_vector(value, p, clip):
    prior = _core._prob_vector(value, p, 'prior_inclusion_prob', scalar_ok=True)
    if np.any((prior < clip) | (prior > 1 - clip)):
        raise ValueError('Priors must lie in [prior_clip, 1-prior_clip]')
    return prior

def build_frequency_prior(pip_matrix, *, prior_floor=0.1, tau=0.5, mix=0.8, clip=0.01):
    q = np.asarray(pip_matrix, dtype=float)
    if q.ndim != 2 or min(q.shape) == 0:
        raise ValueError('pip_matrix must be a non-empty K×p matrix')
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('The PIP matrix contains invalid or missing results; the frequency prior cannot be computed')
    tau = _unit(tau, 'frequency_tau', include_one=False)
    mix = _unit(mix, 'frequency_mix', include_zero=True)
    clip = _unit(clip, 'prior_clip', include_one=False)
    if clip >= 0.5:
        raise ValueError('prior_clip must be < 0.5')
    prior_floor = float(prior_floor)
    if not np.isfinite(prior_floor) or not clip <= prior_floor <= 1 - clip:
        raise ValueError('prior_floor must lie in [prior_clip, 1-prior_clip]')
    selected = q > tau
    counts = selected.sum(axis=0)
    frequency = counts / q.shape[0]
    prior = np.clip(prior_floor + mix * frequency, clip, 1 - clip)
    return dict(selected=selected, counts=counts, frequency=frequency, prior=prior, mean_pip=q.mean(axis=0), median_pip=np.median(q, axis=0))

@dataclass
class PilotResult:
    pip: np.ndarray
    mu: np.ndarray
    s2: np.ndarray
    weights: np.ndarray
    rho: np.ndarray
    prior: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    intercept_mu: float
    intercept_var: float
    slab_variance: float
    converged: bool
    n_rounds: int
    stop_reason: str
    history: list
    snapshots: dict

    @property
    def beta_original(self):
        return self.pip * self.mu / self.x_scale

def fit_unprotected_pilot(X, y, *, prior_inclusion_prob=None, warmup_sweeps=5, max_rounds=500, min_rounds=5, stable_rounds=5, outer_tol=1e-05, core_params=None):
    warmup_sweeps = _integer(warmup_sweeps, 'pilot_warmup_sweeps', 0)
    max_rounds = _integer(max_rounds, 'pilot_max_rounds')
    min_rounds = _integer(min_rounds, 'pilot_min_rounds')
    stable_rounds = _integer(stable_rounds, 'pilot_stable_rounds')
    if not np.isfinite(outer_tol) or outer_tol <= 0:
        raise ValueError('pilot_outer_tol must be a finite positive number')
    (x, yy) = _xy(X, y)
    params = deepcopy(core_params or {})
    for key in ('optimization_mode', 'prior_mode', 'prior_update', 'prior_update_scope', 'prior_inclusion_prob', 'warmup_sweeps', 'unprotected_rounds', 'slab_update'):
        params.pop(key, None)
    cfg = _core.DWBLRStagedLogistic(prior_mode='feature', prior_update='fixed', prior_update_scope='outer', prior_inclusion_prob=prior_inclusion_prob, warmup_sweeps=0, unprotected_rounds=0, slab_update='fixed', **params)
    xs = cfg._prepare_X_fit(x)
    (m, p) = xs.shape
    prior = _prior_vector(cfg.pi0_init if prior_inclusion_prob is None else prior_inclusion_prob, p, cfg.prior_clip)
    q0 = np.full(p, 0.5) if cfg.alpha_init is None else _core._prob_vector(cfg.alpha_init, p, 'alpha_init')
    vp = _core.PGVariationalParams(q0.copy(), np.zeros(p), np.full(p, cfg.sigma0_sq_init))
    ip = _core.InterceptVariationalParams(0.0, cfg.sigma_b_sq)
    hyper = _core.DWBLRHyperParams(float(prior.mean()), cfg.sigma0_sq_init)
    (w, rho) = (np.ones(m), np.full(m, 0.25))
    lam = np.ones((p, p)) - np.eye(p)
    (history, snapshots) = ([], {})

    def snapshot():
        return dict(pip=vp.alpha.copy(), prior=prior.copy(), weights=w.copy(), rho=rho.copy(), mu=vp.mu.copy(), s2=vp.s2.copy(), sigma0_sq=float(hyper.sigma0_sq))

    def solve(limit):
        result = _core.PGMFVI.fit(X=xs, y=yy, W=w, variational_init=vp, intercept_init=ip, hyper_init=hyper, prior_inclusion_prob=prior, feature_prior_update='fixed', update_sigma0_sq=False, sigma_b_sq=cfg.sigma_b_sq, max_iter=limit, tol=cfg.inner_tol, delta_c=cfg.delta_c, delta_h=cfg.delta_h, sigma0_sq_max=cfg.sigma0_sq_max, verbose=False)
        if not np.array_equal(result[-1]['feature_prior'], prior):
            raise RuntimeError('Submodel incorrectly updated the inclusion prior')
        if result[2].sigma0_sq != cfg.sigma0_sq_init:
            raise RuntimeError('Submodel incorrectly updated the slab prior variance')
        return result
    snapshots['initial'] = snapshot()
    for step in range(1, warmup_sweeps + 1):
        (vp, ip, hyper, rho, _, info) = solve(1)
        history.append(dict(stage='warmup', stage_iter=step, inner_change=info['max_param_change'], ESS=float(m), prior_updated=False, prior_min=float(prior.min()), prior_max=float(prior.max()), sigma0_sq=hyper.sigma0_sq))
    snapshots['warmup'] = snapshot()
    (streak, converged) = (0, False)
    for step in range(1, max_rounds + 1):
        (old_w, old_rho, old_vp, old_ip) = (w.copy(), rho.copy(), vp.copy(), ip.copy())
        (target, wi) = _core.CurvatureDecorrelator.fista(X=xs, Lambda=lam, rho_bar=rho, W_init=w, gamma_w=cfg.gamma_w, decorrelation_scale=cfg.decorrelation_scale, covariance_mode=cfg.covariance_mode, max_iter=cfg.attacker_max_iter, tol=cfg.attacker_tol, initial_lr=cfg.attacker_initial_lr * (m if cfg.attacker_scale_step_by_n else 1.0), backtrack_factor=cfg.attacker_backtrack_factor, max_backtracking=cfg.attacker_max_backtracking)
        weight_residual = float(np.max(abs(target - w)))
        w = (1 - cfg.weight_damping) * w + cfg.weight_damping * target
        if not np.isfinite(w).all() or w.min() < -1e-10 or (not np.isclose(w.sum(), m)):
            raise FloatingPointError('Submodel weights must be finite, nonnegative and sum to the sample size')
        (vp, ip, hyper, rho, _, info) = solve(cfg.inner_max_iter)
        (_, grad) = _core.CurvatureDecorrelator.loss_and_grad(xs, lam, w, rho, gamma_w=cfg.gamma_w, decorrelation_scale=cfg.decorrelation_scale, covariance_mode=cfg.covariance_mode)
        pg = float(np.max(abs(w - _core._project_to_simplex(w - m * grad, float(m)))))
        residual = max(weight_residual, float(np.max(abs(w - old_w))), float(np.max(abs(vp.alpha - old_vp.alpha))), float(np.max(abs(vp.mu - old_vp.mu))), float(np.max(abs(vp.s2 - old_vp.s2))), float(np.max(abs(rho - old_rho))), abs(ip.mu_b - old_ip.mu_b), abs(ip.v_b - old_ip.v_b))
        good = np.isfinite([pg, residual, info['max_param_change']]).all() and step >= min_rounds and (pg < outer_tol) and (residual < outer_tol) and (info['max_param_change'] < cfg.inner_tol)
        streak = streak + 1 if good else 0
        history.append(dict(stage='unprotected', stage_iter=step, outer_residual=residual, weight_target_residual=weight_residual, projected_gradient_residual=pg, inner_change=float(info['max_param_change']), inner_iters=int(info['n_iter']), attacker_iters=int(wi['n_iter']), consecutive_converged_checks=streak, ESS=float(w.sum() ** 2 / (w @ w)), max_weight_share=float(w.max() / w.sum()), prior_updated=False, prior_min=float(prior.min()), prior_max=float(prior.max()), sigma0_sq=hyper.sigma0_sq, lambda_offdiag=1.0))
        if cfg.verbose and (step == 1 or step % 20 == 0 or streak >= stable_rounds):
            print(f"  pilot {step}: residual={residual:.3g}, PG={pg:.3g}, ESS={history[-1]['ESS']:.2f}", flush=True)
        if streak >= stable_rounds:
            converged = True
            break
    snapshots['unprotected'] = snapshot()
    return PilotResult(vp.alpha.copy(), vp.mu.copy(), vp.s2.copy(), w.copy(), rho.copy(), prior.copy(), cfg.X_mean_.copy(), cfg.X_scale_.copy(), float(ip.mu_b), float(ip.v_b), float(hyper.sigma0_sq), converged, step, 'unprotected fixed-point criteria met' if converged else 'pilot_max_rounds reached', history, snapshots)

class PilotConvergenceError(RuntimeError):

    def __init__(self, message, model):
        super().__init__(message)
        self.model = model

def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_value, allow_nan=False), encoding='utf-8')

def _write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys((key for row in rows for key in row)))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def _array_hash(a):
    a = np.ascontiguousarray(a, dtype='<f8')
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()

class DWBLREstimator:

    def __init__(self, *, auto_prior=True, n_subsamples=20, subsample_fraction=0.8, subsample_stratify=True, subsample_replace=False, random_state=20260914, pilot_gamma_w=None, pilot_warmup_sweeps=5, pilot_max_rounds=500, pilot_min_rounds=5, pilot_stable_rounds=5, pilot_outer_tol=1e-05, frequency_tau=0.5, frequency_mix=0.8, prior_floor=0.1, optimization_mode='feature_em', prior_update_scope='outer', warmup_sweeps=5, unprotected_rounds=0, prior_inclusion_prob=None, output_dir=None, **kwargs):
        if optimization_mode != 'feature_em':
            raise ValueError("DW-BLR only supports optimization_mode='feature_em'")
        if prior_update_scope not in {'outer', 'inner'}:
            raise ValueError("prior_update_scope must be 'outer' or 'inner'")
        if not isinstance(auto_prior, (bool, np.bool_)):
            raise ValueError('auto_prior must be the boolean True or False')
        self.auto_prior = bool(auto_prior)
        self.optimization_mode = optimization_mode
        self.prior_update_scope = prior_update_scope
        self.n_subsamples = _integer(n_subsamples, 'n_subsamples')
        self.subsample_fraction = _unit(subsample_fraction, 'subsample_fraction')
        if not isinstance(subsample_stratify, (bool, np.bool_)):
            raise ValueError('subsample_stratify must be a boolean')
        self.subsample_stratify = bool(subsample_stratify)
        if not isinstance(subsample_replace, (bool, np.bool_)):
            raise ValueError('subsample_replace must be the boolean True or False')
        self.subsample_replace = bool(subsample_replace)
        self.random_state = _integer(random_state, 'random_state', 0)
        self.pilot_warmup_sweeps = _integer(pilot_warmup_sweeps, 'pilot_warmup_sweeps', 0)
        self.pilot_max_rounds = _integer(pilot_max_rounds, 'pilot_max_rounds')
        self.pilot_min_rounds = _integer(pilot_min_rounds, 'pilot_min_rounds')
        self.pilot_stable_rounds = _integer(pilot_stable_rounds, 'pilot_stable_rounds')
        if not np.isfinite(pilot_outer_tol) or pilot_outer_tol <= 0:
            raise ValueError('pilot_outer_tol must be a finite positive number')
        self.pilot_outer_tol = float(pilot_outer_tol)
        self.frequency_tau = _unit(frequency_tau, 'frequency_tau', include_one=False)
        self.frequency_mix = _unit(frequency_mix, 'frequency_mix', include_zero=True)
        self.prior_floor = float(prior_floor)
        self.prior_inclusion_prob = deepcopy(prior_inclusion_prob)
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.core_params = dict(gamma_w=0.1, pi0_init=0.5, tau=0.5, T_max=3000, prior_center=0.5, prior_damping=0.8, weight_damping=0.8, lambda_damping=0.8, prior_clip=0.01, warmup_sweeps=warmup_sweeps, unprotected_rounds=unprotected_rounds)
        self.core_params.update(deepcopy(kwargs))
        for reserved in ('prior_mode', 'prior_update', 'prior_strength'):
            if reserved in self.core_params:
                raise ValueError(f'DW-BLR does not accept the parameters {reserved}')
        cfg = _core.DWBLRStagedLogistic(prior_mode='feature', prior_update='outer_eb', prior_update_scope=prior_update_scope, prior_inclusion_prob=prior_inclusion_prob, **deepcopy(self.core_params))
        self.pilot_gamma_w = cfg.gamma_w if pilot_gamma_w is None else float(pilot_gamma_w)
        if isinstance(pilot_gamma_w, (bool, np.bool_)) or not np.isfinite(self.pilot_gamma_w) or self.pilot_gamma_w < 0:
            raise ValueError('pilot_gamma_w must be a finite nonnegative number, or None (inherit gamma_w)')
        if not np.isfinite(self.prior_floor) or not 0 <= self.prior_floor <= 1:
            raise ValueError('prior_floor must be a finite number in [0,1]')
        if self.auto_prior and (not cfg.prior_clip <= self.prior_floor <= 1 - cfg.prior_clip):
            raise ValueError('prior_floor must lie in [prior_clip, 1-prior_clip]')
        self.final_model_ = None
        self.status_ = 'not_fitted'

    def fit(self, X, y, feature_names=None, sample_ids=None):
        if self.status_ != 'not_fitted':
            raise RuntimeError('Create a new DWBLREstimator object for each independent fit')
        (x, yy) = _xy(X, y)
        (n, p) = x.shape
        columns = [str(v) for v in X.columns] if hasattr(X, 'columns') else None
        names = columns if feature_names is None and columns is not None else [f'X{j + 1}' for j in range(p)] if feature_names is None else [str(v) for v in feature_names]
        if len(names) != p or len(set(names)) != p:
            raise ValueError('feature_names must match the number of columns and be unique')
        if columns is not None and columns != names:
            raise ValueError('feature_names do not match the DataFrame column names/order')
        if sample_ids is None:
            index = getattr(X, 'index', None)
            sample_ids = list(index) if index is not None and (not callable(index)) else [f'row_{i + 1}' for i in range(n)]
        ids = np.asarray([str(v) for v in sample_ids])
        if ids.shape != (n,) or len(set(ids)) != n:
            raise ValueError('sample_ids must correspond row by row and be unique; subsampling is done at the patient level')
        base = _prior_vector(self.core_params['pi0_init'] if self.prior_inclusion_prob is None else self.prior_inclusion_prob, p, self.core_params['prior_clip'])
        draws = make_patient_subsamples(yy, self.n_subsamples, self.subsample_fraction, self.subsample_stratify, self.random_state, replace=self.subsample_replace) if self.auto_prior else np.empty((0, 0), dtype=int)
        (self.feature_names_, self.sample_ids_, self.y_train_) = (names, ids, yy)
        (self.base_prior_, self.subsample_indices_) = (base.copy(), draws)
        (self.pilot_results_, self.pilot_status_) = ([], [])
        self.n_valid_subsamples_ = 0
        self.valid_subsample_ids_ = np.empty(0, dtype=int)
        self.initial_prior_ = None if self.auto_prior else base.copy()
        self.prior_source_ = 'subsample_frequency' if self.auto_prior else 'provided'
        self.input_hashes_ = dict(X=_array_hash(x), y=_array_hash(yy))
        self.source_hashes_ = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), Path(_core.__file__))}
        self.status_ = 'fitting_pilots' if self.auto_prior else 'fitting_final'
        self._output_path = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            self._output_path = self.output_dir
            self._write_header(self._output_path)
        started = time.perf_counter()
        pilot_params = deepcopy(self.core_params)
        pilot_params['gamma_w'] = self.pilot_gamma_w
        for (k, idx) in enumerate(draws):
            t0 = time.perf_counter()
            (unique_idx, inverse, multiplicities) = np.unique(idx, return_inverse=True, return_counts=True)
            pilot = None
            try:
                pilot = fit_unprotected_pilot(x[idx], yy[idx], prior_inclusion_prob=base, warmup_sweeps=self.pilot_warmup_sweeps, max_rounds=self.pilot_max_rounds, min_rounds=self.pilot_min_rounds, stable_rounds=self.pilot_stable_rounds, outer_tol=self.pilot_outer_tol, core_params=pilot_params)
                if pilot.pip.shape != (p,) or not np.isfinite(pilot.pip).all() or np.any((pilot.pip < 0) | (pilot.pip > 1)):
                    raise FloatingPointError('Submodel returned invalid PIPs, which cannot be counted in the frequencies')
                patient_weights = np.bincount(inverse, weights=pilot.weights)
                status = dict(subsample=k + 1, converged=pilot.converged, n_rounds=pilot.n_rounds, reason=pilot.stop_reason, n_selected=int((pilot.pip > self.frequency_tau).sum()), ESS=float(pilot.weights.sum() ** 2 / (pilot.weights @ pilot.weights)), ESS_unique=float(patient_weights.sum() ** 2 / (patient_weights @ patient_weights)))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                pilot = None
                status = dict(subsample=k + 1, converged=False, n_rounds=None, reason=f'{type(exc).__name__}: {exc}', n_selected=None, ESS=None, ESS_unique=None)
            status.update(n_patients=len(idx), n_draws=len(idx), n_unique_patients=len(unique_idx), max_multiplicity=int(multiplicities.max()), subsample_replace=self.subsample_replace, pCR_1=int(yy[idx].sum()), gamma_w=self.pilot_gamma_w, seconds=time.perf_counter() - t0)
            self.pilot_results_.append(pilot)
            self.pilot_status_.append(status)
            if self._output_path is not None:
                self._write_pilot(self._output_path, k)
                _write_json(self._output_path / 'run_status.json', dict(status=self.status_, pilots=self.pilot_status_))
            if self.core_params.get('verbose', False):
                print(f'DW-BLR subsample {k + 1}/{self.n_subsamples}: {status}', flush=True)
        valid_indices = [i for (i, s) in enumerate(self.pilot_status_) if s['converged']]
        self.n_valid_subsamples_ = len(valid_indices)
        self.valid_subsample_ids_ = np.asarray(valid_indices, dtype=int) + 1
        if self.auto_prior and self.n_valid_subsamples_ == 0:
            self.status_ = 'pilot_failure'
            self.elapsed_seconds_ = time.perf_counter() - started
            if self._output_path is not None:
                self._write_summary(self._output_path)
            failed = [s['subsample'] for s in self.pilot_status_ if not s['converged']]
            raise PilotConvergenceError(f'All submodels {failed} failed the convergence/validity checks (K_eff=0), so the frequency prior is undefined. The final model was not fitted; failures are not counted as zero votes and the supplied prior is not used as a fallback.', self)
        if self.auto_prior:
            self.subsample_pip_matrix_ = np.stack([self.pilot_results_[i].pip for i in valid_indices])
            agg = build_frequency_prior(self.subsample_pip_matrix_, prior_floor=self.prior_floor, tau=self.frequency_tau, mix=self.frequency_mix, clip=self.core_params['prior_clip'])
            self.subsample_selection_matrix_ = agg['selected']
            self.subsample_selection_counts_ = agg['counts']
            self.subsample_selection_frequency_ = agg['frequency']
            self.frequency_prior_ = agg['prior']
            self.initial_prior_ = self.frequency_prior_.copy()
            if self.n_valid_subsamples_ < self.n_subsamples:
                failed = [s['subsample'] for s in self.pilot_status_ if not s['converged']]
                warnings.warn(f'Submodels {failed} did not converge or were invalid; frequencies use only the {self.n_valid_subsamples_}/{self.n_subsamples} converged submodels and the final fit continues. Failures are not counted as zero votes.', RuntimeWarning, stacklevel=2)
        self.status_ = 'fitting_final'
        if self._output_path is not None:
            if self.auto_prior:
                self._write_aggregation(self._output_path)
            self._write_initial_prior(self._output_path)
            _write_json(self._output_path / 'run_status.json', dict(status=self.status_))
        try:
            self.final_model_ = _core.DWBLR(x, yy, feature_names=names, optimization_mode='feature_em', prior_update_scope=self.prior_update_scope, prior_inclusion_prob=self.initial_prior_.copy(), **deepcopy(self.core_params))
        except Exception:
            self.status_ = 'final_error'
            self.elapsed_seconds_ = time.perf_counter() - started
            if self._output_path is not None:
                self._write_summary(self._output_path)
            raise
        self.status_ = 'complete' if self.final_model_.converged_ else 'final_not_converged'
        self.elapsed_seconds_ = time.perf_counter() - started
        if self._output_path is not None:
            self._write_final(self._output_path)
            self._write_summary(self._output_path)
        return self

    def __getattr__(self, name):
        final = self.__dict__.get('final_model_')
        if final is not None:
            return getattr(final, name)
        raise AttributeError(f'{name}: the final model has not been fitted')

    def _prediction_X(self, X):
        if self.final_model_ is None:
            raise ValueError('The final model has not been fitted')
        if hasattr(X, 'columns') and [str(v) for v in X.columns] != self.feature_names_:
            raise ValueError('Gene names or column order of the prediction data do not match; align explicitly to the training gene_order')
        return X

    def predict_proba(self, X):
        checked = self._prediction_X(X)
        return self.final_model_.predict_proba(checked)

    def decision_function(self, X):
        checked = self._prediction_X(X)
        return self.final_model_.decision_function(checked)

    def predict(self, X, threshold=0.5):
        checked = self._prediction_X(X)
        return self.final_model_.predict(checked, threshold=threshold)

    def get_frequency_prior(self):
        if not self.auto_prior:
            raise ValueError('auto_prior=False: no frequency prior was computed; use get_initial_prior() to inspect the supplied initial values')
        if 'frequency_prior_' not in self.__dict__:
            raise ValueError('No valid frequency prior has been generated yet')
        return self.frequency_prior_.copy()

    def get_initial_prior(self):
        if 'initial_prior_' not in self.__dict__ or self.status_ in {'not_fitted', 'fitting_pilots', 'pilot_failure'}:
            raise ValueError('No initial prior for the final model has been generated yet')
        return self.initial_prior_.copy()

    def get_subsample_selection_frequency(self):
        if not self.auto_prior:
            raise ValueError('auto_prior=False: no subsampling was run, so no subsample frequencies exist')
        if 'subsample_selection_frequency_' not in self.__dict__:
            raise ValueError('No valid subsample frequencies have been generated yet')
        return self.subsample_selection_frequency_.copy()

    def _write_header(self, root):
        config = {k: v for (k, v) in vars(self).items() if not k.endswith('_') and k not in ('_output_path', 'final_model_')}
        _write_json(root / 'config.json', config)
        _write_json(root / 'input_hashes.json', self.input_hashes_)
        _write_json(root / 'source_hashes.json', self.source_hashes_)
        if self.auto_prior:
            np.savez_compressed(root / 'subsample_indices.npz', indices=self.subsample_indices_)
        _write_csv(root / 'gene_order.csv', (dict(column=j + 1, gene=g) for (j, g) in enumerate(self.feature_names_)))
        _write_csv(root / 'training_sample_order.csv', (dict(row=i + 1, sample_id=s, pCR=int(self.y_train_[i])) for (i, s) in enumerate(self.sample_ids_)))
        code = root / 'source'
        code.mkdir()
        for path in (Path(__file__), Path(_core.__file__)):
            (code / path.name).write_bytes(path.read_bytes())

    def _write_pilot(self, root, k):
        folder = root / 'pilots' / f'subset_{k + 1:03d}'
        folder.mkdir(parents=True, exist_ok=False)
        (idx, r) = (self.subsample_indices_[k], self.pilot_results_[k])
        _write_json(folder / 'status.json', self.pilot_status_[k])
        _write_csv(folder / 'sample_ids.csv', (dict(source_row_1based=int(i) + 1, sample_id=self.sample_ids_[i], pCR=int(self.y_train_[i]), draw_index_1based=d + 1) for (d, i) in enumerate(idx)))
        if r is None:
            return
        _write_csv(folder / 'all_gene_PIP.csv', (dict(gene=g, PIP=float(r.pip[j]), selected=bool(r.pip[j] > self.frequency_tau), beta=float(r.beta_original[j]), prior=float(r.prior[j]), valid_for_voting=r.converged) for (j, g) in enumerate(self.feature_names_)))
        _write_csv(folder / 'history.csv', r.history)
        _write_csv(folder / 'weights.csv', (dict(sample_id=self.sample_ids_[i], weight=float(w), source_row_1based=int(i) + 1, draw_index_1based=d + 1) for (d, (i, w)) in enumerate(zip(idx, r.weights))))
        np.savez_compressed(folder / 'state.npz', pip=r.pip, mu=r.mu, s2=r.s2, weights=r.weights, rho=r.rho, prior=r.prior, X_mean=r.x_mean, X_scale=r.x_scale, intercept_mu=r.intercept_mu, intercept_var=r.intercept_var, slab_variance=r.slab_variance)

    def _write_aggregation(self, root):
        folder = root / 'aggregation'
        folder.mkdir(exist_ok=True)
        _write_csv(folder / 'gene_frequency_prior.csv', (dict(gene=g, K_requested=self.n_subsamples, K_valid=self.n_valid_subsamples_, selected_count=int(self.subsample_selection_counts_[j]), selection_frequency=float(self.subsample_selection_frequency_[j]), mean_PIP=float(self.subsample_pip_matrix_[:, j].mean()), median_PIP=float(np.median(self.subsample_pip_matrix_[:, j])), base_prior=float(self.base_prior_[j]), prior_floor=self.prior_floor, frequency_mix=self.frequency_mix, prior_inclusion_prob=float(self.frequency_prior_[j])) for (j, g) in enumerate(self.feature_names_)))
        _write_csv(folder / 'selection_matrix.csv', (dict(subsample=int(self.valid_subsample_ids_[k]), **{g: int(self.subsample_selection_matrix_[k, j]) for (j, g) in enumerate(self.feature_names_)}) for k in range(self.n_valid_subsamples_)))
        np.savez_compressed(folder / 'PIP_matrix.npz', PIP=self.subsample_pip_matrix_, subsample_ids=self.valid_subsample_ids_)

    def _write_initial_prior(self, root):
        _write_csv(root / 'initial_prior.csv', (dict(gene=g, prior_source=self.prior_source_, initial_prior=float(self.initial_prior_[j])) for (j, g) in enumerate(self.feature_names_)))

    def _write_final(self, root):
        folder = root / 'final_model'
        folder.mkdir(exist_ok=True)
        (f, r) = (self.final_model_, self.final_model_.result_)
        rows = [dict(gene=g, PIP=float(r.alpha[j]), selected=bool(r.alpha[j] > f.tau), beta=float(f.get_posterior_mean_coef(original_scale=True)[j]), initial_prior=float(self.initial_prior_[j]), prior_source=self.prior_source_, final_prior_used=float(r.feature_prior[j])) for (j, g) in enumerate(self.feature_names_)]
        _write_csv(folder / 'all_gene_PIP.csv', rows)
        with (folder / 'selected_genes.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows((row for row in rows if row['selected']))
        _write_csv(folder / 'history.csv', f.history_)
        _write_csv(folder / 'weights.csv', (dict(sample_id=s, weight=float(w)) for (s, w) in zip(self.sample_ids_, r.weights)))
        np.savez_compressed(folder / 'state.npz', pip=r.alpha, mu=r.mu, s2=r.s2, weights=r.weights, rho=r.rho_bar, Lambda=r.Lambda, prior=r.feature_prior, initial_prior=self.initial_prior_, X_mean=f.X_mean_, X_scale=f.X_scale_, intercept_mu=r.intercept_mu, intercept_var=r.intercept_var, slab_variance=r.hyper.sigma0_sq)
        _write_json(folder / 'diagnostics.json', f.convergence_diagnostics_)
        _write_json(folder / 'effective_config.json', {k: v for (k, v) in vars(f).items() if not k.endswith('_') and k != 'result_'})

    def _write_summary(self, root):
        _write_json(root / 'run_status.json', dict(status=self.status_, auto_prior=self.auto_prior, prior_source=self.prior_source_, subsample_replace=self.subsample_replace, n_subsamples=self.n_subsamples if self.auto_prior else 0, n_subsamples_configured=self.n_subsamples, n_attempted=len(self.pilot_status_), valid_subsample_ids=self.valid_subsample_ids_, n_failed=sum((not s['converged'] for s in self.pilot_status_)), n_valid=self.n_valid_subsamples_, pilots=self.pilot_status_, elapsed_seconds=self.elapsed_seconds_, final_converged=None if self.final_model_ is None else self.final_model_.converged_))
        _write_json(root / 'manifest.json', {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'manifest.json'})

    def save_results(self, output_dir):
        if self.status_ not in {'complete', 'final_not_converged', 'pilot_failure', 'final_error'}:
            raise ValueError('There is no completed/failed status to export')
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=False)
        self._write_header(root)
        for k in range(len(self.pilot_results_)):
            self._write_pilot(root, k)
        if 'frequency_prior_' in self.__dict__:
            self._write_aggregation(root)
        if self.status_ != 'pilot_failure':
            self._write_initial_prior(root)
        if self.final_model_ is not None:
            self._write_final(root)
        self._write_summary(root)
        return root
