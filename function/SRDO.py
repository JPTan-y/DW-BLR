from copy import deepcopy
if __package__ and '.' in __package__:
    from .._backends.srdo_logistic import SRDOLogistic as _Base
    from .._backends.srdo_density import SRDO as SRDODensity
    from ..simulation.parameters import srdo_parameters
    from .._common import entry, shape_of, resolve_share
else:
    from _backends.srdo_logistic import SRDOLogistic as _Base
    from _backends.srdo_density import SRDO as SRDODensity
    from simulation.parameters import srdo_parameters
    from _common import entry, shape_of, resolve_share
_DENSITY_MAP = dict(hidden_layer_sizes='srdo_hidden_layer_sizes', activation='srdo_activation', solver='srdo_optimizer', l2='srdo_alpha', dropout='srdo_dropout', learning_rate='srdo_learning_rate', batch_size='srdo_batch_size', max_epochs='srdo_max_epochs', early_stopping='srdo_early_stopping', validation_fraction='srdo_validation_fraction', patience='srdo_patience', tol='srdo_tolerance', gradient_clip='srdo_gradient_clip', probability_clip='srdo_probability_clip', weight_clip_quantile='srdo_weight_clip_quantile', sensitive_variables='srdo_sensitive_variables')

class SRDOModel(_Base):

    def __init__(self, **kwargs):
        cfg = srdo_parameters()
        cfg.update(deepcopy(kwargs))
        self._share_spec = cfg.pop('max_weight_share')
        for (public, internal) in _DENSITY_MAP.items():
            if internal in cfg:
                raise ValueError(f'Use the public density parameter {public!r}, not {internal!r}')
            cfg[internal] = cfg.pop(public)
        super().__init__(max_weight_share=None, **cfg)

    def fit(self, X, y, feature_names=None):
        (n, _) = shape_of(X)
        self.max_weight_share = resolve_share(self._share_spec, n)
        return super().fit(X, y, feature_names=feature_names)

def SRDO(X=None, y=None, *, feature_names=None, **kwargs):
    return entry(SRDOModel, X, y, kwargs, feature_names)
__all__ = ['SRDO', 'SRDOModel', 'SRDODensity']
