"""Configuration fixtures for testing."""

from pathlib import Path

import pytest

from ebm.core.config import (
    CDConfig,
    GaussianRBMConfig,
    GibbsConfig,
    OptimizerConfig,
    ParallelTemperingConfig,
    RBMConfig,
    TrainingConfig,
)


@pytest.fixture
def default_rbm_config():
    """Provide a default RBM configuration for testing."""
    return RBMConfig(
        visible_units=784,
        hidden_units=500,
        weight_init="xavier_normal",
        bias_init=0.0,
        use_bias=True,
        device="cpu",
        dtype="float32",
        seed=42,
    )


@pytest.fixture
def small_rbm_config():
    """Provide a small RBM configuration for fast testing."""
    return RBMConfig(
        visible_units=20,
        hidden_units=10,
        weight_init="normal",
        bias_init="zeros",
        use_bias=True,
        device="cpu",
        dtype="float32",
        seed=42,
        l2_weight=0.001,
    )


@pytest.fixture
def gaussian_rbm_config():
    """Provide a Gaussian RBM configuration."""
    return GaussianRBMConfig(
        visible_units=100,
        hidden_units=50,
        weight_init="xavier_uniform",
        bias_init=0.0,
        sigma=1.0,
        learn_sigma=True,
        device="cpu",
        dtype="float32",
        seed=42,
    )


@pytest.fixture
def training_config(tmp_path: Path):
    """Provide a training configuration."""
    return TrainingConfig(
        epochs=10,
        batch_size=32,
        optimizer=OptimizerConfig(
            name="sgd", lr=0.01, momentum=0.9, weight_decay=0.0001
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=5,
        log_every=10,
        eval_every=2,
        eval_samples=100,
        early_stopping=True,
        patience=3,
        min_delta=0.001,
        mixed_precision=False,
        compile_model=False,
        num_workers=0,
        pin_memory=False,
    )


@pytest.fixture
def sampler_configs():
    """Provide various sampler configurations."""
    return {
        "gibbs": GibbsConfig(num_steps=10, block_gibbs=True),
        "cd": CDConfig(num_steps=1, persistent=False),
        "pcd": CDConfig(num_steps=1, persistent=True, num_chains=100),
        "pt": ParallelTemperingConfig(
            num_temps=5, min_beta=0.5, max_beta=1.0, swap_every=1, num_steps=10
        ),
    }


@pytest.fixture
def config_variations():
    """Provide various configuration variations for parametrized testing."""
    return [
        # Different initialization strategies
        {"weight_init": "xavier_normal", "bias_init": 0.0},
        {"weight_init": "xavier_uniform", "bias_init": "normal"},
        {"weight_init": "kaiming_normal", "bias_init": 0.01},
        {"weight_init": "uniform", "bias_init": "zeros"},
        # Different architectures
        {"visible_units": 100, "hidden_units": 50},
        {"visible_units": 784, "hidden_units": 128},
        {"visible_units": 1024, "hidden_units": 2048},
        # Different regularization
        {"l2_weight": 0.0, "l1_weight": 0.0},
        {"l2_weight": 0.001, "l1_weight": 0.0},
        {"l2_weight": 0.0, "l1_weight": 0.001},
        {"l2_weight": 0.001, "l1_weight": 0.001},
    ]


@pytest.fixture
def invalid_configs():
    """Provide invalid configurations for error testing."""
    return [
        # Invalid dimensions
        {"visible_units": 0, "hidden_units": 100},
        {"visible_units": 100, "hidden_units": -1},
        # Invalid initialization
        {"weight_init": "invalid_method"},
        {"bias_init": "unknown_init"},
        # Invalid dtypes
        {"dtype": "float128"},
        {"dtype": "complex64"},
        # Invalid devices
        {"device": "tpu"},
        {"device": "cuda:99"},
    ]
