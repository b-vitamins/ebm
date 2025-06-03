"""Energy-Based Models (EBM) Library.

A modern PyTorch library for energy-based machine learning models,
with a focus on Restricted Boltzmann Machines (RBMs) and related models.
"""

__version__ = "0.1.0"
__author__ = "EBM Contributors"
__license__ = "MIT"

# Core functionality
from . import core
from . import models
from . import sampling
from . import training
from . import inference
from . import utils

# Import key classes for convenience
from .core.config import (
    ModelConfig, TrainingConfig, OptimizerConfig,
    RBMConfig, GaussianRBMConfig
)
from .core.device import DeviceManager, get_device, set_device
from .core.logging import setup_logging, logger
from .core.registry import models as model_registry, samplers as sampler_registry

# Models
from .models.base import EnergyBasedModel, LatentVariableModel
from .models.rbm import (
    BernoulliRBM, CenteredBernoulliRBM, SparseBernoulliRBM,
    GaussianBernoulliRBM, WhitenedGaussianRBM
)

# Samplers
from .sampling.base import Sampler, GradientEstimator
from .sampling.gradient import (
    ContrastiveDivergence, PersistentContrastiveDivergence,
    FastPersistentCD
)
from .sampling.mcmc import ParallelTempering, AnnealedImportanceSampling
from .sampling.deterministic import MeanFieldSampler, TAPSampler

# Training
from .training.trainer import Trainer
from .training.callbacks import (
    Callback, CallbackList, LoggingCallback, CheckpointCallback,
    EarlyStoppingCallback, VisualizationCallback
)
from .training.metrics import MetricsTracker, ModelEvaluator

# Inference
from .inference.partition import AISEstimator, BridgeSampling

# Utilities
from .utils.data import (
    get_mnist_datasets, get_fashion_mnist_datasets,
    create_data_loaders, compute_data_statistics
)
from .utils.visualization import (
    visualize_filters, visualize_samples,
    plot_training_curves, plot_energy_histogram
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