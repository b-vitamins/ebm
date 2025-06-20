"""Core functionality for the EBM library."""

from .config import (
    BaseConfig,
    CDConfig,
    GaussianRBMConfig,
    GibbsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelTemperingConfig,
    RBMConfig,
    SamplerConfig,
    TrainingConfig,
)
from .device import (
    DeviceInfo,
    DeviceManager,
    auto_device,
    get_device,
    get_device_manager,
    memory_efficient,
    set_device,
    to_device,
)
from .logging_utils import (
    LogConfig,
    LoggerMixin,
    MetricProcessor,
    debug,
    error,
    info,
    log_context,
    log_duration,
    log_function_call,
    logger,
    metrics,
    setup_logging,
    warning,
)
from .registry import (
    ModelRegistry,
    OptimizerRegistry,
    Registry,
    SamplerRegistry,
    TransformRegistry,
    discover_plugins,
    get_all_registries,
    models,
    optimizers,
    register_model,
    register_optimizer,
    register_sampler,
    register_transform,
    samplers,
    transforms,
)
from .types_ import (
    Callback,
    ChainState,
    Config,
    Device,
    DType,
    EnergyModel,
    GradientEstimator,
    InitMethod,
    InitStrategy,
    LatentModel,
    Sampler,
    SamplingMetadata,
    Shape,
    TensorLike,
    Transform,
)

__all__ = [
    # Config
    "BaseConfig",
    "CDConfig",
    "Callback",
    "ChainState",
    # Types
    "Config",
    "DType",
    "Device",
    "DeviceInfo",
    # Device
    "DeviceManager",
    "EnergyModel",
    "GaussianRBMConfig",
    "GibbsConfig",
    "GradientEstimator",
    "InitMethod",
    "InitStrategy",
    "LatentModel",
    # Logging
    "LogConfig",
    "LoggerMixin",
    "MetricProcessor",
    "ModelConfig",
    "ModelRegistry",
    "OptimizerConfig",
    "OptimizerRegistry",
    "ParallelTemperingConfig",
    "RBMConfig",
    # Registry
    "Registry",
    "Sampler",
    "SamplerConfig",
    "SamplerRegistry",
    "SamplingMetadata",
    "Shape",
    "TensorLike",
    "TrainingConfig",
    "Transform",
    "TransformRegistry",
    "auto_device",
    "debug",
    "discover_plugins",
    "error",
    "get_all_registries",
    "get_device",
    "get_device_manager",
    "info",
    "log_context",
    "log_duration",
    "log_function_call",
    "logger",
    "memory_efficient",
    "metrics",
    "models",
    "optimizers",
    "register_model",
    "register_optimizer",
    "register_sampler",
    "register_transform",
    "samplers",
    "set_device",
    "setup_logging",
    "to_device",
    "transforms",
    "warning",
]
