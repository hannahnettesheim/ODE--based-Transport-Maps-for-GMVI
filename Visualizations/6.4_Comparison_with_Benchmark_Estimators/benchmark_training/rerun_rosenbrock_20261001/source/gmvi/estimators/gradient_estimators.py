
from typing import Dict, Type

from .baseEstimator import GradientEstimator
from .score_function import ScoreFunctionEstimator
from .gumbel_softmax import GumbelSoftmaxEstimator
from .exact_marginalization import ExactMarginalizationEstimator
from .ode_transport import ODETransportEstimator


ESTIMATOR_REGISTRY: Dict[str, Type[GradientEstimator]] = {
    "score_function": ScoreFunctionEstimator,
    "gumbel_softmax": GumbelSoftmaxEstimator,
    "exact_marginalization": ExactMarginalizationEstimator,
    "ode_transport": ODETransportEstimator,
}


def make_estimator(name: str, **kwargs) -> GradientEstimator:
    try:
        estimator_cls = ESTIMATOR_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(ESTIMATOR_REGISTRY))
        raise ValueError(
            f"Unknown gradient estimator {name!r}. "
            f"Available estimators: {available}"
        )
    return estimator_cls(**kwargs)