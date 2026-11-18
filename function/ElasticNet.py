if __package__ and '.' in __package__:
    from .._backends.penalized_logistic import PenalizedLogistic
    from ..simulation.parameters import penalized_parameters
    from .._common import entry
else:
    from _backends.penalized_logistic import PenalizedLogistic
    from simulation.parameters import penalized_parameters
    from _common import entry

class ElasticNetModel(PenalizedLogistic):

    def __init__(self, *, lambda_=0.02, alpha=0.8, **kwargs):
        cfg = penalized_parameters('Elastic Net')
        cfg.pop('lambda_')
        cfg.pop('alpha')
        cfg.update(kwargs)
        super().__init__(penalty='elastic_net', penalty_strength=lambda_, l1_ratio=alpha, **cfg)

def ElasticNet(X=None, y=None, *, feature_names=None, **kwargs):
    return entry(ElasticNetModel, X, y, kwargs, feature_names)
__all__ = ['ElasticNet', 'ElasticNetModel']
