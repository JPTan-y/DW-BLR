if __package__ and '.' in __package__:
    from .._backends.penalized_logistic import PenalizedLogistic
    from ..simulation.parameters import penalized_parameters
    from .._common import entry
else:
    from _backends.penalized_logistic import PenalizedLogistic
    from simulation.parameters import penalized_parameters
    from _common import entry

class LassoModel(PenalizedLogistic):

    def __init__(self, *, lambda_=0.02, **kwargs):
        cfg = penalized_parameters('Lasso')
        cfg.pop('lambda_')
        cfg.update(kwargs)
        super().__init__(penalty='lasso', penalty_strength=lambda_, **cfg)

def Lasso(X=None, y=None, *, feature_names=None, **kwargs):
    return entry(LassoModel, X, y, kwargs, feature_names)
__all__ = ['Lasso', 'LassoModel']
