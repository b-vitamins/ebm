"""Core functionality for the EBM library."""

from .config import (
    BaseConfig, ModelConfig, OptimizerConfig, TrainingConfig,
    SamplerConfig, GibbsConfig, CDConfig, ParallelTemperingConfig,
    RBMConfig, GaussianRBMConfig
)
from .device import (
    DeviceManager, DeviceInfo,
    get_device_manager, set_device, get_device, to_device,
    auto_device, memory_efficient
)
from .logging import (
    LogConfig, LoggerMixin, MetricProcessor,
    setup_logging, logger,
    log_context, log_duration, log_function_call,
    debug, info, warning, error, metrics
)
from .registry import (
    Registry, ModelRegistry, SamplerRegistry,
    OptimizerRegistry, TransformRegistry,
    models, samplers, optimizers, transforms,
    register_model, register_sampler, register_optimizer, register_transform,
    discover_plugins, get_all_registries
)
from .types import (
    Config, EnergyModel, LatentModel, Sampler, GradientEstimator,
    Callback, Transform,
    TensorLike, Device, DType, Shape,
    InitMethod, InitStrategy,
    SamplingMetadata, ChainState
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