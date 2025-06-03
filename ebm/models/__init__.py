"""Energy-based model implementations."""

from .base import (
    EnergyBasedModel, LatentVariableModel, AISInterpolator
)

# Import RBM models
from . import rbm
from .rbm import (
    RBMBase, RBMAISAdapter,
    BernoulliRBM, CenteredBernoulliRBM, SparseBernoulliRBM,
    GaussianBernoulliRBM, WhitenedGaussianRBM
)

__all__ = [
    # Base classes
    "EnergyBasedModel", "LatentVariableModel", "AISInterpolator",
    
    # RBM module
    "rbm",
    
    # RBM models
    "RBMBase", "RBMAISAdapter",
    "BernoulliRBM", "CenteredBernoulliRBM", "SparseBernoulliRBM",
    "GaussianBernoulliRBM", "WhitenedGaussianRBM",
]