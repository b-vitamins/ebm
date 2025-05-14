"""Test helper module for the EBM test suite.

This module provides utilities and mock implementations for testing RBM
models and samplers. All testing helpers are organized into specific
submodules for performance, RBM models, and samplers.
"""

from __future__ import annotations

# Performance utilities
from tests.helper.perf import (
    _time_function,
    run_perf,
    run_performance_cpu_test,
    run_performance_gpu_test,
)

# RBM utilities and mocks
from tests.helper.rbm import (
    BetaRecordingModel,
    BetaValidatingModel,
    MockRBM,
    exact_visible_dist,
    make_rbm,
    print_distribution_comparison,
    rbm_model_config,
    xavier_init,
)

# Sampler utilities
from tests.helper.sampler import (
    sampler_config_from_fixtures,
    sampler_model_config,
)

__all__ = [
    # Performance utilities
    "run_perf",
    "run_performance_cpu_test",
    "run_performance_gpu_test",
    "_time_function",
    # RBM utilities and mocks
    "MockRBM",
    "BetaRecordingModel",
    "BetaValidatingModel",
    "make_rbm",
    "xavier_init",
    "exact_visible_dist",
    "print_distribution_comparison",
    "rbm_model_config",
    # Sampler utilities
    "sampler_model_config",
    "sampler_config_from_fixtures",
]
