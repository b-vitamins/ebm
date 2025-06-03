"""Energy-Based Models (EBM) Library.

A modern PyTorch library for energy-based machine learning models,
with a focus on Restricted Boltzmann Machines (RBMs) and related models.
"""

__version__ = "0.1.0"
__author__ = "EBM Contributors"
__license__ = "MIT"

# Core functionality
from . import core, inference, models, sampling, training, utils

# Import key classes for convenience
from .core.config import (
    GaussianRBMConfig,
    ModelConfig,
    OptimizerConfig,
    RBMConfig,
    TrainingConfig,
)
from .core.device import DeviceManager, get_device, set_device
from .core.logging import logger, setup_logging
from .core.registry import models as model_registry
from .core.registry import samplers as sampler_registry

# Inference
from .inference.partition import AISEstimator, BridgeSampling

# Models
from .models.base import EnergyBasedModel, LatentVariableModel
from .models.rbm import (
    BernoulliRBM,
    CenteredBernoulliRBM,
    GaussianBernoulliRBM,
    SparseBernoulliRBM,
    WhitenedGaussianRBM,
)

# Samplers
from .sampling.base import GradientEstimator, Sampler
from .sampling.deterministic import MeanFieldSampler, TAPSampler
from .sampling.gradient import (
    ContrastiveDivergence,
    FastPersistentCD,
    PersistentContrastiveDivergence,
)
from .sampling.mcmc import AnnealedImportanceSampling, ParallelTempering
from .training.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    VisualizationCallback,
)
from .training.metrics import MetricsTracker, ModelEvaluator

# Training
from .training.trainer import Trainer

# Utilities
from .utils.data import (
    compute_data_statistics,
    create_data_loaders,
    get_fashion_mnist_datasets,
    get_mnist_datasets,
)
from .utils.visualization import (
    plot_energy_histogram,
    plot_training_curves,
    visualize_filters,
    visualize_samples,
)


# Convenience functions
def create_model(name: str, config: ModelConfig) -> EnergyBasedModel:
    """Create a model by name.

    Args:
        name: Model name (e.g., 'bernoulli_rbm', 'gaussian_rbm')
        config: Model configuration

    Returns:
        Instantiated model
    """
    return model_registry.create(name, config=config)


def create_sampler(name: str, **kwargs) -> Sampler:
    """Create a sampler by name.

    Args:
        name: Sampler name (e.g., 'cd', 'pcd', 'parallel_tempering')
        **kwargs: Sampler-specific arguments

    Returns:
        Instantiated sampler
    """
    return sampler_registry.create(name, **kwargs)


# Package metadata
__all__ = [
    # Version info
    "__version__", "__author__", "__license__",

    # Submodules
    "core", "models", "sampling", "training", "inference", "utils",

    # Core classes
    "ModelConfig", "TrainingConfig", "OptimizerConfig",
    "RBMConfig", "GaussianRBMConfig",
    "DeviceManager", "get_device", "set_device",
    "setup_logging", "logger",

    # Models
    "EnergyBasedModel", "LatentVariableModel",
    "BernoulliRBM", "CenteredBernoulliRBM", "SparseBernoulliRBM",
    "GaussianBernoulliRBM", "WhitenedGaussianRBM",

    # Samplers
    "Sampler", "GradientEstimator",
    "ContrastiveDivergence", "PersistentContrastiveDivergence",
    "FastPersistentCD", "ParallelTempering", "AnnealedImportanceSampling",
    "MeanFieldSampler", "TAPSampler",

    # Training
    "Trainer", "Callback", "CallbackList",
    "LoggingCallback", "CheckpointCallback", "EarlyStoppingCallback",
    "VisualizationCallback", "MetricsTracker", "ModelEvaluator",

    # Inference
    "AISEstimator", "BridgeSampling",

    # Utilities
    "get_mnist_datasets", "get_fashion_mnist_datasets",
    "create_data_loaders", "compute_data_statistics",
    "visualize_filters", "visualize_samples",
    "plot_training_curves", "plot_energy_histogram",

    # Convenience functions
    "create_model", "create_sampler",
]
