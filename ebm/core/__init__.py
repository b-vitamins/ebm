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
from .logging import (
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
from .types import (
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
    "BaseConfig", "ModelConfig", "OptimizerConfig", "TrainingConfig",
    "SamplerConfig", "GibbsConfig", "CDConfig", "ParallelTemperingConfig",
    "RBMConfig", "GaussianRBMConfig",

    # Device
    "DeviceManager", "DeviceInfo",
    "get_device_manager", "set_device", "get_device", "to_device",
    "auto_device", "memory_efficient",

    # Logging
    "LogConfig", "LoggerMixin", "MetricProcessor",
    "setup_logging", "logger",
    "log_context", "log_duration", "log_function_call",
    "debug", "info", "warning", "error", "metrics",

    # Registry
    "Registry", "ModelRegistry", "SamplerRegistry",
    "OptimizerRegistry", "TransformRegistry",
    "models", "samplers", "optimizers", "transforms",
    "register_model", "register_sampler", "register_optimizer", "register_transform",
    "discover_plugins", "get_all_registries",

    # Types
    "Config", "EnergyModel", "LatentModel", "Sampler", "GradientEstimator",
    "Callback", "Transform",
    "TensorLike", "Device", "DType", "Shape",
    "InitMethod", "InitStrategy",
    "SamplingMetadata", "ChainState",
]
