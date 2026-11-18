if __package__ and '.' in __package__:
    from .._backends.dwr_logistic import DWRLogistic as _Base
    from ..simulation.parameters import dwr_parameters
    from .._common import entry
else:
    from _backends.dwr_logistic import DWRLogistic as _Base
    from simulation.parameters import dwr_parameters
    from _common import entry

class DWRModel(_Base):

    def __init__(self, **kwargs):
        cfg = dwr_parameters()
        cfg.update(kwargs)
        super().__init__(**cfg)

def DWR(X=None, y=None, *, feature_names=None, **kwargs):
    return entry(DWRModel, X, y, kwargs, feature_names)
__all__ = ['DWR', 'DWRModel']
