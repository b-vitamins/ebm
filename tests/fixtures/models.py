"""Model fixtures for testing."""

from pathlib import Path

import pytest
import torch

from ebm.core.config import GaussianRBMConfig, RBMConfig
from ebm.models.rbm import (
    BernoulliRBM,
    CenteredBernoulliRBM,
    GaussianBernoulliRBM,
)


@pytest.fixture
def simple_bernoulli_rbm(small_rbm_config):
    """Provide a simple Bernoulli RBM for testing."""
    model = BernoulliRBM(small_rbm_config)

    # Initialize with known weights for reproducible tests
    torch.manual_seed(42)
    with torch.no_grad():
        model.W.data = torch.randn_like(model.W) * 0.01
        model.vbias.data = torch.zeros_like(model.vbias)
        model.hbias.data = torch.zeros_like(model.hbias)

    return model


@pytest.fixture
def small_gaussian_rbm():
    """Provide a small Gaussian RBM for testing."""
    config = GaussianRBMConfig(
        visible_units=20,
        hidden_units=10,
        sigma=1.0,
        learn_sigma=True,
        device="cpu",
        dtype="float32",
        seed=42
    )
    return GaussianBernoulliRBM(config)


@pytest.fixture
def pretrained_rbm(simple_bernoulli_rbm, synthetic_binary_data):
    """Provide a pre-trained RBM for testing inference."""
    model = simple_bernoulli_rbm
    data = synthetic_binary_data["data"]

    # Simulate pre-training by setting meaningful weights
    with torch.no_grad():
        # Use PCA-like initialization
        data_centered = data - data.mean(dim=0)
        cov = data_centered.T @ data_centered / data.shape[0]
        eigvals, eigvecs = torch.linalg.eigh(cov)

        # Use top eigenvectors as weight initialization
        n_components = min(model.num_hidden, data.shape[1])
        top_eigvecs = eigvecs[:, -n_components:]

        model.W.data[:n_components] = top_eigvecs.T

        # Set biases based on data statistics
        model.vbias.data = torch.log(data.mean(dim=0).clamp(0.01, 0.99) /
                                     (1 - data.mean(dim=0).clamp(0.01, 0.99)))

    return model


@pytest.fixture
def make_test_rbm():
    """Factory fixture for creating test RBMs with various configurations."""
    def _make_test_rbm(
        visible_units: int = 100,
        hidden_units: int = 50,
        model_type: str = "bernoulli",
        **kwargs
    ):
        """Create a test RBM with specified configuration."""
        base_config = {
            "visible_units": visible_units,
            "hidden_units": hidden_units,
            "device": "cpu",
            "dtype": "float32",
            "seed": 42
        }
        base_config.update(kwargs)

        if model_type == "bernoulli":
            config = RBMConfig(**base_config)
            return BernoulliRBM(config)
        if model_type == "gaussian":
            config = GaussianRBMConfig(**base_config)
            return GaussianBernoulliRBM(config)
        if model_type == "centered":
            config = RBMConfig(**base_config, centered=True)
            return CenteredBernoulliRBM(config)
        raise ValueError(f"Unknown model type: {model_type}")

    return _make_test_rbm


@pytest.fixture
def model_comparison_suite(make_test_rbm):
    """Provide a suite of models for comparison testing."""
    return {
        "small_bernoulli": make_test_rbm(20, 10, "bernoulli"),
        "medium_bernoulli": make_test_rbm(100, 50, "bernoulli"),
        "large_bernoulli": make_test_rbm(784, 500, "bernoulli"),
        "gaussian": make_test_rbm(100, 50, "gaussian"),
        "centered": make_test_rbm(100, 50, "centered"),
    }


@pytest.fixture
def save_and_load_model(tmp_path: Path):
    """Fixture for testing model save/load functionality."""
    def _save_and_load(model, filename: str = "test_model.pt"):
        """Save a model and load it back."""
        save_path = tmp_path / filename

        # Save original state
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Save model
        model.save_checkpoint(save_path)

        # Create new model instance
        new_model = model.__class__(model.config)

        # Load checkpoint
        new_model.load_checkpoint(save_path)

        return new_model, original_state

    return _save_and_load


@pytest.fixture
def model_parameter_stats():
    """Fixture for computing model parameter statistics."""
    def _compute_stats(model):
        """Compute statistics for model parameters."""
        stats = {}

        for name, param in model.named_parameters():
            param_data = param.data
            stats[name] = {
                "shape": list(param_data.shape),
                "mean": param_data.mean().item(),
                "std": param_data.std().item(),
                "min": param_data.min().item(),
                "max": param_data.max().item(),
                "norm": param_data.norm().item(),
                "num_zeros": (param_data == 0).sum().item(),
                "num_params": param_data.numel(),
                "requires_grad": param.requires_grad
            }

            # Add gradient stats if available
            if param.grad is not None:
                grad_data = param.grad
                stats[name]["grad_mean"] = grad_data.mean().item()
                stats[name]["grad_std"] = grad_data.std().item()
                stats[name]["grad_norm"] = grad_data.norm().item()

        return stats

    return _compute_stats
