"""Restricted Boltzmann Machine (RBM) module.

This package provides implementations of Restricted Boltzmann Machines
and related Energy-Based Models.
"""

from ebm.rbm.model import BaseRBM, BernoulliRBM, BernoulliRBMConfig, RBMConfig
from ebm.rbm.utils import shape_beta

__all__ = ["BaseRBM", "RBMConfig", "BernoulliRBMConfig", "BernoulliRBM", "shape_beta"]
