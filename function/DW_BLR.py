if __package__ and '.' in __package__:
    from .._backends.dw_blr_bootstrap import DWBLREstimator as _Base, PilotConvergenceError
    from ..simulation.parameters import dw_blr_parameters
    from .._common import entry, shape_of
else:
    from _backends.dw_blr_bootstrap import DWBLREstimator as _Base, PilotConvergenceError
    from simulation.parameters import dw_blr_parameters
    from _common import entry, shape_of

class DW_BLRModel(_Base):

    def __init__(self, **kwargs):
        self._auto_initial_prior = 'prior_inclusion_prob' not in kwargs
        cfg = dw_blr_parameters()
        cfg.update(kwargs)
        super().__init__(**cfg)

    def fit(self, X, y, feature_names=None, sample_ids=None):
        if self._auto_initial_prior:
            (_, p) = shape_of(X)
            self.prior_inclusion_prob = [0.5] * p
        return super().fit(X, y, feature_names=feature_names, sample_ids=sample_ids)

def DW_BLR(X=None, y=None, *, feature_names=None, sample_ids=None, **kwargs):
    return entry(DW_BLRModel, X, y, kwargs, feature_names, sample_ids=sample_ids)
__all__ = ['DW_BLR', 'DW_BLRModel', 'PilotConvergenceError']
