import numpy as np
if __package__ and '.' in __package__:
    from .._backends.vblr_core import VBLRModel as _Base
    from ..simulation.parameters import vblr_parameters
    from .._common import entry, shape_of
else:
    from _backends.vblr_core import VBLRModel as _Base
    from simulation.parameters import vblr_parameters
    from _common import entry, shape_of

class VBLRModel(_Base):

    def __init__(self, **kwargs):
        self._auto_initial_pip = 'alpha_init' not in kwargs
        cfg = vblr_parameters()
        cfg.update(kwargs)
        super().__init__(**cfg)

    def fit(self, X, y, feature_names=None):
        if self._auto_initial_pip:
            (_, p) = shape_of(X)
            self.alpha_init = np.full(p, 0.5)
        return super().fit(X, y, feature_names=feature_names)

def VBLR(X=None, y=None, *, feature_names=None, **kwargs):
    return entry(VBLRModel, X, y, kwargs, feature_names)
__all__ = ['VBLR', 'VBLRModel']
