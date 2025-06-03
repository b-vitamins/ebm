"""Test fixtures package for the EBM test suite."""

from .configs import (
    default_rbm_config,
    gaussian_rbm_config,
    sampler_configs,
    small_rbm_config,
    training_config,
)
from .datasets import (
    make_structured_data,
    mini_mnist_dataset,
    synthetic_binary_data,
    synthetic_continuous_data,
)
from .mocks import (
    mock_callback,
    mock_data_loader,
    mock_gradient_estimator,
    mock_sampler,
)
from .models import (
    make_test_rbm,
    pretrained_rbm,
    simple_bernoulli_rbm,
    small_gaussian_rbm,
)

__all__ = [
    # From configs
    "default_rbm_config",
    "small_rbm_config",
    "gaussian_rbm_config",
    "training_config",
    "sampler_configs",
    # From datasets
    "synthetic_binary_data",
    "synthetic_continuous_data",
    "mini_mnist_dataset",
    "make_structured_data",
    # From models
    "simple_bernoulli_rbm",
    "small_gaussian_rbm",
    "pretrained_rbm",
    "make_test_rbm",
    # From mocks
    "mock_data_loader",
    "mock_gradient_estimator",
    "mock_callback",
    "mock_sampler",
]
