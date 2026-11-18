from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import mannwhitneyu, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, auc, brier_score_loss,
                             log_loss, precision_recall_curve, roc_auc_score)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path, expected_n, expected_p):
    names = ["GSE25066", "GSE163882", "GSE41998", "GSE230881",
             "GSE140494", "GSE194040", "GSE231629"]
    cohorts = {}
    genes = None
    for i, name in enumerate(names):
        sheet = ("Train_" if i == 0 else "Test_") + name
        frame = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        if list(frame.columns[:4]) != ["row_index_1based", "sample_id", "pCR", "outcome"]:
            raise ValueError(f"Unexpected metadata columns: {sheet}")
        if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
            raise ValueError(f"Missing/duplicate sample IDs: {sheet}")
        observed = frame.columns[4:].astype(str).tolist()
        if genes is None:
            genes = observed
        if observed != genes or len(set(observed)) != len(observed):
            raise ValueError(f"Feature order mismatch: {sheet}")
        X = frame.iloc[:, 4:].to_numpy(dtype=float)
        y = frame.pCR.to_numpy(dtype=float)
        if not np.isfinite(X).all() or not np.isfinite(y).all() or set(y) != {0., 1.}:
            raise ValueError(f"Non-finite data or non-binary outcome: {sheet}")
        if not np.array_equal(frame.outcome.to_numpy(), np.where(y == 1, "pCR", "RD")):
            raise ValueError(f"Outcome annotation mismatch: {sheet}")
        cohorts[name] = dict(X=X, y=y.astype(int), sample_id=frame.sample_id.astype(str).to_numpy())
    if cohorts[names[0]]["X"].shape != (expected_n, expected_p):
        raise ValueError("Training shape does not match parameters.py")
    train_ids = set(cohorts[names[0]]["sample_id"])
    if any(train_ids.intersection(cohorts[name]["sample_id"]) for name in names[1:]):
        raise ValueError("Training/test sample ID overlap")
    return cohorts, genes


def training_scale(X):
    mean, scale = np.mean(X, axis=0), np.std(X, axis=0, ddof=0)
    if np.any(scale <= 0):
        raise ValueError("Constant training feature")
    return mean, scale


def ridge_fit(X, y, cfg, weights=None, topk=False):
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, p = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if w.shape != (n,) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Invalid sample weights")
    w = w / w.mean()
    lam = float(cfg["lambda_l2"])
    if cfg.get("intercept_penalized", False):
        raise ValueError("This solver uses an unpenalized intercept")
    A = np.column_stack([np.ones(n), X])
    penalty = np.r_[0., np.full(p, lam)]

    def objective(b):
        eta = A @ b
        return float(np.dot(w, np.logaddexp(0., eta) - y * eta) / n + .5 * np.dot(penalty, b * b))

    def gradient(b):
        return A.T @ (w * (expit(A @ b) - y)) / n + penalty * b

    def hessian(b):
        pr = expit(A @ b)
        return (A.T * (w * pr * (1. - pr))) @ A / n + np.diag(penalty)

    start = np.r_[logit(np.average(y, weights=w)), np.zeros(p)]
    opt = cfg["optimizer"]
    result = minimize(objective, start, jac=gradient, hess=hessian,
                      method=opt["method"], options={k: opt[k] for k in ("gtol", "maxiter")})
    grad = float(np.max(np.abs(gradient(result.x))))
    check = cfg["independent_check"]
    if topk:
        independent = LogisticRegression(C=1. / (n * lam), penalty=check["penalty"],
            solver=check["solver"].split("/")[-1], tol=check["tol"], max_iter=check["max_iter"])
        independent.fit(X, y, sample_weight=w)
        bcheck = np.r_[independent.intercept_, independent.coef_.ravel()]
        bound = cfg["gradient_inf_lt"]
        difference_bound = check["max_difference"]
    else:
        independent = minimize(objective, start, jac=gradient, method=check["method"],
            options={k: check[k] for k in ("gtol", "maxiter")})
        bcheck = independent.x
        bound = cfg["verification"]["gradient_inf_lt"]
        difference_bound = cfg["verification"]["coefficient_max_difference_lt"]
    difference = float(np.max(np.abs(result.x - bcheck)))
    if grad >= bound or difference >= difference_bound:
        raise RuntimeError(f"Ridge verification failed: gradient={grad}, coefficient difference={difference}")
    return dict(intercept=float(result.x[0]), beta=result.x[1:], selected=np.ones(p, bool),
                diagnostics=dict(converged=True, optimizer_success=bool(result.success),
                optimizer_message=str(result.message), gradient_inf=grad,
                independent_coefficient_max_difference=difference, iterations=result.nit,
                lambda_l2=lam))


def predict_linear(fit, Z):
    p = expit(float(fit["intercept"]) + np.asarray(Z) @ np.asarray(fit["beta"], float))
    if not np.isfinite(p).all():
        raise FloatingPointError("Non-finite predictions")
    return p


def scores(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    return dict(Log_loss=log_loss(y, p, labels=[0, 1]), Brier=brier_score_loss(y, p),
                AUROC=roc_auc_score(y, p), AUPRC=auc(recall, precision),
                AP=average_precision_score(y, p))


def find_rscript(config):
    candidates = [os.environ.get("RSCRIPT"), shutil.which("Rscript")]
    if sys.platform == "win32":
        import winreg
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hive, r"SOFTWARE\R-core\R") as key:
                    root = Path(winreg.QueryValueEx(key, "InstallPath")[0])
                    candidates += [str(root / "bin" / "Rscript.exe"), str(root / "bin" / "x64" / "Rscript.exe")]
            except OSError:
                pass
    versions = {("ncvreg" if m == "SCAD" else "glmnet"): c["package_version"] for m, c in config.items()}
    test = "; ".join(["stopifnot(requireNamespace('jsonlite',quietly=TRUE))"] +
        [f"stopifnot(requireNamespace('{p}',quietly=TRUE), packageVersion('{p}') == package_version('{v}'))" for p, v in versions.items()])
    errors = []
    for candidate in dict.fromkeys(v for v in candidates if v):
        process = subprocess.run([candidate, "--vanilla", "-e", test], capture_output=True, text=True, errors="replace")
        if process.returncode == 0:
            return candidate
        errors.append(f"{candidate}: {process.stderr.strip()}")
    raise RuntimeError("Rscript with jsonlite, glmnet 4.1-10 and ncvreg 3.16.0 is required. "
                       "Set RSCRIPT and, if needed, R_LIBS_USER.\n" + "\n".join(errors))


def fit_penalties(X, y, genes, configs, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    executable = find_rscript(configs)
    pd.DataFrame(X, columns=genes).to_csv(folder / "X.csv", index=False)
    pd.DataFrame({"pCR": y}).to_csv(folder / "y.csv", index=False)
    save_json(folder / "parameters.json", configs)
    with (folder / "R.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([executable, "--vanilla", str(Path(__file__).with_name("real_data_penalties.R")), str(folder)],
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError((folder / "R.log").read_text(encoding="utf-8", errors="replace"))
    return json.loads((folder / "results.json").read_text(encoding="utf-8"))


def _pop_standardize(cfg):
    cfg = dict(cfg)
    if not cfg.pop("standardize", True):
        raise ValueError("Real-data adapter expects training-only standardization")
    return cfg


def fit_full(method, Xraw, Z, y, genes, sample_ids, params, folder, shared):
    from _backends.dwr_weights import DWR
    from _backends.srdo_density import SRDO, _normalize_weights_with_max_share
    from _backends.svi_core import SVILogistic
    from _backends.vblr_feature_em import VBLRModel
    from function.DW_BLR import DW_BLRModel
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if method == "LR":
        fit = ridge_fit(Z, y, params._internal_weighted_fit())
        fit["diagnostics"]["backend"] = "ridge-logistic baseline"
    elif method in ("Lasso", "Elastic Net", "SCAD"):
        if "penalties" not in shared:
            configs = {m: params.penalized_parameters(m) for m in ("Lasso", "Elastic Net", "SCAD")}
            shared["penalties"] = fit_penalties(Z, y, genes, configs, folder.parent / "R_penalties")
        fit = shared["penalties"][method]
    elif method in ("DWR", "SRDO"):
        cfg = params.dwr_parameters() if method == "DWR" else params.srdo_parameters(len(y))
        pilot = ridge_fit(Z, y, cfg["pilot_fit"])
        pcfg = cfg["curvature"]
        pp = np.clip(predict_linear(pilot, Z), pcfg["probability_clip"], 1 - pcfg["probability_clip"])
        curvature = np.maximum(pp * (1 - pp), pcfg["curvature_floor"])
        if pcfg["ignore_high_confidence_correct"] or pcfg["corrected_weight_clip_quantile"] is not None:
            raise ValueError("Unsupported curvature options")
        if method == "DWR":
            learner = DWR(**_pop_standardize(cfg["weight_learning"])).fit(Z)
            base = learner.get_weights()
        else:
            learner = SRDO(**_pop_standardize(cfg["density"])).fit(Z)
            base = learner.get_raw_density_ratio()
            shared["density_weights"] = learner.get_weights()
            shared["density_diagnostics"] = learner.get_diagnostics()
            save_json(folder / "density.json", dict(weights=learner.get_weights(), diagnostics=learner.get_diagnostics()))
        corrected, _, _ = _normalize_weights_with_max_share(base / curvature, pcfg["final_max_weight_share"])
        fit = ridge_fit(Z, y, cfg["final_weighted_fit"], weights=corrected)
        fit["weights"] = corrected
        fit["base_weights"] = base
        fit["diagnostics"]["weight_learning"] = learner.get_diagnostics()
        fit["diagnostics"]["pilot"] = pilot["diagnostics"]
        if method == "DWR":
            learner.history_table().to_csv(folder / "weight_history.csv", index=False)
    elif method == "SVI":
        if "density_weights" not in shared:
            cached = json.loads((folder.parent / "SRDO" / "density.json").read_text(encoding="utf-8"))
            shared["density_weights"], shared["density_diagnostics"] = cached["weights"], cached["diagnostics"]
        cfg = _pop_standardize(params.svi_parameters(len(y)))
        cfg.update(precomputed_srdo_weights=np.asarray(shared["density_weights"]),
                   precomputed_srdo_diagnostics=shared["density_diagnostics"])
        model = SVILogistic(**cfg).fit(Z, y, feature_names=genes)
        history = model.history_table()
        history.to_csv(folder / "selector_history.csv", index=False)
        tail = history.tail(cfg["selector_patience"])
        stopped = bool(len(tail) == cfg["selector_patience"] and
            (tail.epoch >= cfg["selector_min_epochs"]).all() and
            (tail.max_gate_probability_change < cfg["selector_tol"]).all() and
            (tail.objective_change < cfg["selector_tol"]).all())
        selected = np.zeros(len(genes), bool)
        selected[model.get_selected_features()] = True
        fit = dict(beta=model.get_final_coefficients(), intercept=model.get_final_intercept(),
                   selected=selected, importance=model.get_selection_probabilities(), importance_type="gate_probability", weights=model.get_weights(),
                   diagnostics=dict(converged=bool(stopped and model.result_.final_logistic_info["success"]),
                       selector_stopping_criterion_met=stopped, selector_epochs=int(history.epoch.max()),
                       final_fit=model.result_.final_logistic_info))
        np.testing.assert_allclose(predict_linear(fit, Z), model.predict_proba(Z)[:, 1], atol=1e-12)
    elif method == "VBLR":
        model = VBLRModel(**params.vblr_parameters(len(genes))).fit(Xraw, y, feature_names=genes)
        fit = dict(beta=model.get_posterior_mean_coef(), intercept=model.get_intercept(),
                   importance=model.get_pip(), importance_type="PIP", selected=model.get_pip() > params.vblr_parameters(len(genes))["tau"],
                   diagnostics=model.get_diagnostics())
        np.testing.assert_allclose(predict_linear(fit, Z), model.predict_proba(Xraw)[:, 1], atol=1e-10)
    elif method == "DW-BLR":
        cfg = params.dw_blr_parameters(len(genes))
        if cfg["subsample_replace"] or cfg["tau"] != params.DWBLR_FINAL_PIP_THRESHOLD:
            raise ValueError("DW-BLR requires without-replacement subsampling and the configured final PIP threshold")
        cfg["output_dir"] = str(folder / "trace")
        model = DW_BLRModel(**cfg).fit(Z, y, feature_names=genes, sample_ids=sample_ids)
        fit = dict(beta=model.get_posterior_mean_coef(), intercept=model.get_intercept(),
                   importance=model.get_pip(), importance_type="PIP",
                   selected=model.get_pip() > params.DWBLR_FINAL_PIP_THRESHOLD,
                   weights=model.get_weights(), subsample_frequency=model.get_subsample_selection_frequency(),
                   initial_prior=model.get_initial_prior(), diagnostics=dict(converged=bool(model.final_model_.converged_),
                   status=model.status_, pilots=model.pilot_status_, n_valid_subsamples=model.n_valid_subsamples_))
        np.testing.assert_allclose(predict_linear(fit, Z), model.predict_proba(Z)[:, 1], atol=1e-12)
    else:
        raise ValueError(method)
    if "importance" not in fit:
        fit["importance"] = np.abs(np.asarray(fit["beta"], float))
        fit["importance_type"] = "absolute_standardized_coefficient"
    fit["diagnostics"]["elapsed_seconds"] = time.perf_counter() - started
    if not fit["diagnostics"].get("converged", False):
        warnings.warn(f"{method}: convergence criterion was not met; see saved diagnostics", RuntimeWarning)
    save_json(folder / "fit.json", fit)
    return fit


def bh_adjust(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid], kind="stable")]
    if len(order):
        out[order] = np.minimum(1., np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    return out


def univariable_effect(x, y):
    sd = np.std(x, ddof=0)
    if sd <= 0 or np.unique(y).size != 2:
        return dict(beta=np.nan, se=np.nan, lower=np.nan, upper=np.nan, p=np.nan, fit_ok=False)
    z = (x - np.mean(x)) / sd
    A = np.column_stack([np.ones(len(y)), z])
    def loss(b):
        eta = A @ b
        return np.sum(np.logaddexp(0, eta) - y * eta)
    def grad(b):
        return A.T @ (expit(A @ b) - y)
    def hess(b):
        pr = expit(A @ b)
        return (A.T * (pr * (1 - pr))) @ A
    r = minimize(loss, [logit(np.mean(y)), 0.], jac=grad, hess=hess,
                 method="trust-exact", options=dict(gtol=1e-8, maxiter=1000))
    H = hess(r.x)
    okay = bool(np.max(np.abs(grad(r.x))) < 1e-6 and np.linalg.cond(H) < 1e10 and np.max(np.abs(r.x)) < 25)
    if not okay:
        return dict(beta=np.nan, se=np.nan, lower=np.nan, upper=np.nan, p=np.nan, fit_ok=False)
    se = float(np.sqrt(np.linalg.inv(H)[1, 1]))
    b = float(r.x[1])
    return dict(beta=b, se=se, lower=b - norm.ppf(.975) * se,
                upper=b + norm.ppf(.975) * se, p=2 * norm.sf(abs(b / se)), fit_ok=True)


def gene_statistics(cohorts, genes, requested_genes):
    rows = []
    for cohort, data in cohorts.items():
        for gene in requested_genes:
            x, y = data["X"][:, genes.index(gene)], data["y"]
            rows.append(dict(cohort=cohort, gene=gene, n=len(y), pCR=int(y.sum()),
                **univariable_effect(x, y), rank_sum_p=float(mannwhitneyu(x[y == 1], x[y == 0],
                    alternative="two-sided", method="asymptotic").pvalue)))
    return pd.DataFrame(rows)
