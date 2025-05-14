"""RBM sampler module.

This module provides base classes, types, and specific implementations for
RBM sampling algorithms. It includes the infrastructure for creating custom
samplers as well as ready-to-use sampling algorithms.
"""

from ebm.rbm.sampler.base import (
    # Main base classes and utilities
    BaseSamplerRBM,
    # Type definitions users might need
    BundledSamplerHook,
    RemovableHandle,
    SampleRBM,
    SamplerHook,
    SamplingStepBundle,
    TensorType,
    UnbundledSamplerHook,
)
from ebm.rbm.sampler.cd import CDSampler
from ebm.rbm.sampler.pcd import PCDSampler

__all__ = [
    # Base classes and utilities
    "BaseSamplerRBM",
    "SampleRBM",
    "RemovableHandle",
    # Type definitions
    "TensorType",
    "SamplingStepBundle",
    "UnbundledSamplerHook",
    "BundledSamplerHook",
    "SamplerHook",
    # Specific samplers
    "CDSampler",
    "PCDSampler",
]
