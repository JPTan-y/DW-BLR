import numpy as np

def shape_of(X):
    array = np.asarray(X, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError('X must be an n-by-p numeric array')
    if not np.isfinite(array).all():
        raise ValueError('X contains NaN or Inf; handle missing values using training data first')
    return array.shape

def resolve_share(value, n):
    if isinstance(value, str):
        if value != '50/n':
            raise ValueError('max_weight_share string must be "50/n"')
        return min(1.0, 50.0 / n)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or (not 1 / n <= float(value) <= 1):
        raise ValueError("max_weight_share must be in [1/n,1], None, or '50/n'")
    return float(value)

def checked_weights(weights, n, share=None):
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Weights must be finite, strictly positive and aligned with training rows')
    w = w / w.mean()
    if share is not None and w.max() / w.sum() > share + 1e-12:
        raise ValueError('Precomputed density weights violate max_weight_share')
    return w

def entry(model_class, X, y, params, feature_names=None, **fit_kwargs):
    if 'standardize' in params:
        raise TypeError('Control standardization in experiment.py; method functions do not accept standardize')
    if (X is None) != (y is None):
        raise ValueError('Pass both X and y to fit, or neither to construct a model')
    model = model_class(**params)
    if X is None:
        if feature_names is not None or any((v is not None for v in fit_kwargs.values())):
            raise ValueError('Pass feature_names/sample_ids when calling fit(X,y)')
        return model
    return model.fit(X, y, feature_names=feature_names, **fit_kwargs)
