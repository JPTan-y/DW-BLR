from copy import deepcopy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

if __package__:
    from .function.Lasso import Lasso
    from .function.ElasticNet import ElasticNet
    from .function.SCAD import SCAD
    from .function.SRDO import SRDO
    from .function.SVI import SVI
    from .function.DWR import DWR
    from .function.VBLR import VBLR
    from .function.DW_BLR import DW_BLR
    from .simulation.parameters import all_parameters, density_parameters
else:
    from function.Lasso import Lasso
    from function.ElasticNet import ElasticNet
    from function.SCAD import SCAD
    from function.SRDO import SRDO
    from function.SVI import SVI
    from function.DWR import DWR
    from function.VBLR import VBLR
    from function.DW_BLR import DW_BLR
    from simulation.parameters import all_parameters, density_parameters


METHODS = {
    "Lasso": Lasso, "Elastic Net": ElasticNet, "SCAD": SCAD,
    "SRDO": SRDO, "SVI": SVI, "DWR": DWR, "VBLR": VBLR, "DW-BLR": DW_BLR,
}
STANDARDIZE = {
    "Lasso": True, "Elastic Net": True, "SCAD": True,
    "SRDO": True, "SVI": True, "DWR": True, "VBLR": False, "DW-BLR": False,
}


class FittedMethod:
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = list(feature_names)

    def transform(self, X):
        if hasattr(X, "columns") and [str(v) for v in X.columns] != self.feature_names:
            raise ValueError("Test-data column names or order do not match the training data")
        values = np.asarray(X, dtype=float)
        return self.scaler.transform(values) if self.scaler is not None else values

    def predict_proba(self, X):
        return self.model.predict_proba(self.transform(X))

    def predict(self, X, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0,1)")
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def coefficients(self):
        if hasattr(self.model, "get_posterior_mean_coef"):
            beta = self.model.get_posterior_mean_coef()
        elif hasattr(self.model, "get_final_coefficients"):
            beta = self.model.get_final_coefficients()
        else:
            beta = self.model.get_coefficients()
        if hasattr(self.model, "get_final_intercept"):
            intercept = self.model.get_final_intercept()
        else:
            intercept = self.model.get_intercept()
        if self.scaler is not None:
            beta = beta / self.scaler.scale_
            intercept = intercept - self.scaler.mean_ @ beta
        return np.asarray(beta), float(intercept)


def fit_methods(X_train, y_train, *, methods=None, params=None,
                standardize=None, feature_names=None):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train).ravel()
    if X.ndim != 2 or len(y) != len(X) or not np.isfinite(X).all():
        raise ValueError("Invalid training-data format")
    if hasattr(X_train, "index") and hasattr(y_train, "index"):
        if (hasattr(X_train.index, "equals") and hasattr(y_train.index, "equals")
                and not X_train.index.equals(y_train.index)):
            raise ValueError("X_train and y_train indices do not match")
    n, p = X.shape
    names = ([str(v) for v in X_train.columns] if hasattr(X_train, "columns")
             else [f"X{j+1}" for j in range(p)])
    if feature_names is not None:
        supplied = [str(v) for v in feature_names]
        if hasattr(X_train, "columns") and supplied != names:
            raise ValueError("feature_names do not match the column order of the training data")
        names = supplied
    if len(names) != p or len(set(names)) != p:
        raise ValueError("feature_names must match the number of columns and be unique")
    requested = list(METHODS) if methods is None else list(methods)
    if len(set(requested)) != len(requested) or any(m not in METHODS for m in requested):
        raise ValueError("Invalid or duplicated method names")
    options = all_parameters(n, p)
    for method, update in (params or {}).items():
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        options[method].update(deepcopy(update))
    scaling = dict(STANDARDIZE)
    if isinstance(standardize, (bool, np.bool_)):
        scaling = dict.fromkeys(METHODS, bool(standardize))
    elif standardize is not None:
        if any(m not in METHODS or not isinstance(v, (bool, np.bool_)) for m,v in standardize.items()):
            raise ValueError("standardize must be a boolean or a mapping from method to boolean")
        scaling.update(standardize)
    scaler = StandardScaler().fit(X) if any(scaling[m] for m in requested) else None
    scaled_X = scaler.transform(X) if scaler is not None else None
    fitted = {}
    for method in requested:
        cfg = deepcopy(options[method])
        if method == "SVI" and "SRDO" in fitted and "precomputed_srdo_weights" not in cfg:
            density = density_parameters(n)
            srdo_cfg = {k: options["SRDO"][k] for k in density}
            svi_cfg = dict(density)
            svi_cfg.update(cfg.get("srdo_params") or {})
            if srdo_cfg == svi_cfg and scaling["SRDO"] == scaling["SVI"]:
                cfg["precomputed_srdo_weights"] = fitted["SRDO"].model.get_srdo_weights()
                cfg["precomputed_srdo_diagnostics"] = fitted["SRDO"].model.get_diagnostics()
        values = scaled_X if scaling[method] else X
        model = METHODS[method](values, y, feature_names=names, **cfg)
        fitted[method] = FittedMethod(model, scaler if scaling[method] else None, names)
    return fitted


def evaluate_methods(fitted, X_test, y_test):
    y = np.asarray(y_test).ravel()
    rows = []
    for method, fitted_model in fitted.items():
        probability = fitted_model.predict_proba(X_test)[:, 1]
        row = dict(method=method, log_loss=float(log_loss(y, probability, labels=[0, 1])),
                   AUROC=float(roc_auc_score(y, probability)),
                   Brier=float(brier_score_loss(y, probability)))
        if hasattr(fitted_model.model, "get_selected_features"):
            row["model_size"] = int(len(fitted_model.model.get_selected_features()))
        rows.append(row)
    return rows


def run_experiment(X_train, y_train, X_test, y_test, **kwargs):
    fitted = fit_methods(X_train, y_train, **kwargs)
    return fitted, evaluate_methods(fitted, X_test, y_test)
