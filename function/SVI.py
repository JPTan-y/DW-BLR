from copy import deepcopy
import numpy as np
if __package__ and '.' in __package__:
    from .._backends.svi_core import SVILogistic as _Base
    from .._backends.srdo_density import SRDO as _Density
    from ..simulation.parameters import svi_parameters, density_parameters
    from .._common import entry, shape_of, resolve_share, checked_weights
else:
    from _backends.svi_core import SVILogistic as _Base
    from _backends.srdo_density import SRDO as _Density
    from simulation.parameters import svi_parameters, density_parameters
    from _common import entry, shape_of, resolve_share, checked_weights

class SVIModel(_Base):

    def __init__(self, *, C=0.03, penalty_dimension=30, lambda_stg=None, srdo_params=None, **kwargs):
        if isinstance(C, (bool, np.bool_)) or not np.isfinite(C) or C < 0:
            raise ValueError('C must be finite and nonnegative')
        if penalty_dimension is not None and (isinstance(penalty_dimension, (bool, np.bool_)) or int(penalty_dimension) != penalty_dimension or penalty_dimension < 2):
            raise ValueError('penalty_dimension must be >=2 or None (use actual p)')
        if lambda_stg is not None and (isinstance(lambda_stg, (bool, np.bool_)) or not np.isfinite(lambda_stg) or lambda_stg < 0):
            raise ValueError('lambda_stg must be nonnegative or None for ESS scaling')
        cfg = svi_parameters()
        for key in ('C', 'penalty_dimension', 'lambda_stg', 'srdo_params'):
            cfg.pop(key)
        cfg.update(deepcopy(kwargs))
        network_keys = [k for k in cfg if k.startswith('srdo_') and k != 'srdo_max_weight_share']
        if network_keys:
            raise ValueError('Configure the shared density estimator using srdo_params: ' + str(network_keys))
        self.C = float(C)
        self.penalty_dimension = penalty_dimension
        self._lambda_spec = lambda_stg
        self.srdo_params = density_parameters()
        self.srdo_params.update(deepcopy(srdo_params or {}))
        self._share_spec = cfg.pop('srdo_max_weight_share', self.srdo_params['max_weight_share'])
        self._provided_weights = deepcopy(cfg.get('precomputed_srdo_weights'))
        self._provided_diagnostics = deepcopy(cfg.get('precomputed_srdo_diagnostics', {}))
        self.density_model_ = None
        super().__init__(lambda_stg=0.0 if lambda_stg is None else float(lambda_stg), srdo_max_weight_share=None, **cfg)

    def fit(self, X, y, feature_names=None):
        (n, p) = shape_of(X)
        share = resolve_share(self._share_spec, n)
        if self._provided_weights is None:
            density_cfg = deepcopy(self.srdo_params)
            density_cfg['max_weight_share'] = share
            self.density_model_ = _Density(**density_cfg).fit(X)
            weights = self.density_model_.get_weights()
            diagnostics = self.density_model_.get_diagnostics()
        else:
            self.density_model_ = None
            weights = self._provided_weights
            diagnostics = self._provided_diagnostics or {}
        weights = checked_weights(weights, n, share)
        self.ESS_ = float(weights.sum() ** 2 / np.dot(weights, weights))
        dimension = p if self.penalty_dimension is None else int(self.penalty_dimension)
        self.lambda_stg = float(self._lambda_spec) if self._lambda_spec is not None else float(self.C * np.sqrt(np.log(dimension) / self.ESS_))
        self.lambda_stg_ = self.lambda_stg
        self.lambda_source_ = 'explicit' if self._lambda_spec is not None else 'C*sqrt(log(penalty_dimension)/ESS)'
        self.precomputed_srdo_weights = weights.copy()
        self.precomputed_srdo_diagnostics = dict(diagnostics)
        self.srdo_max_weight_share = share
        return super().fit(X, y, feature_names=feature_names)

def SVI(X=None, y=None, *, feature_names=None, **kwargs):
    return entry(SVIModel, X, y, kwargs, feature_names)
__all__ = ['SVI', 'SVIModel']
