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
]
