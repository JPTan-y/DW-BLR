if __package__ and '.' in __package__:
    from .._backends.penalized_logistic import PenalizedLogistic
    from ..simulation.parameters import penalized_parameters
    from .._common import entry
else:
    from _backends.penalized_logistic import PenalizedLogistic
    from simulation.parameters import penalized_parameters
    from _common import entry

class SCADModel(PenalizedLogistic):

    def __init__(self, *, lambda_=0.02, a=5.0, **kwargs):
        cfg = penalized_parameters('SCAD')
        cfg.pop('lambda_')
        cfg.pop('a')
        cfg.update(kwargs)
        super().__init__(penalty='scad', penalty_strength=lambda_, scad_a=a, **cfg)

def SCAD(X=None, y=None, *, feature_names=None, **kwargs):
    return entry(SCADModel, X, y, kwargs, feature_names)
__all__ = ['SCAD', 'SCADModel']
