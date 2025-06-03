"""Configuration management using Pydantic for type safety and validation.

This module provides base configuration classes that leverage Pydantic's
powerful validation and serialization capabilities.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import torch
from pydantic import BaseModel, Field, validator

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel, ABC):
    """Base configuration class with common functionality.

    All configuration classes should inherit from this to get:
    - Automatic validation
    - JSON/YAML serialization
    - Immutability (frozen)
    - Type conversion
    """

    class Config:
        """Pydantic configuration."""

        frozen = True  # Make configs immutable
        use_enum_values = True
        arbitrary_types_allowed = True
        json_encoders: ClassVar[dict[type[Any], Any]] = {
            torch.dtype: lambda v: str(v).replace("torch.", ""),
            torch.device: str,
            Path: str,
        }

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent mutation of frozen models by raising AttributeError."""
        if self.__config__.frozen:
            raise AttributeError(f"{self.__class__.__name__} is immutable")
        super().__setattr__(name, value)

    @classmethod
    def from_dict(cls: type[T], config_dict: dict[str, Any]) -> T:
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_file(cls: type[T], path: str | Path) -> T:
        """Load configuration from JSON or YAML file."""
        import json

        path = Path(path)

        if path.suffix in {".yaml", ".yml"}:
            try:
                import yaml

                with path.open() as f:
                    data = yaml.safe_load(f)
            except ImportError as err:
                raise ImportError(
                    "PyYAML required for YAML config files"
                ) from err
        elif path.suffix == ".json":
            with path.open() as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file type: {path.suffix}")

        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Save configuration to file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in {".yaml", ".yml"}:
            try:
                import yaml

                with path.open("w") as f:
                    yaml.safe_dump(self.dict(), f, default_flow_style=False)
            except ImportError as err:
                raise ImportError(
                    "PyYAML required for YAML config files"
                ) from err
        else:
            with path.open("w") as f:
                json.dump(self.dict(), f, indent=2)

    def dict(self, **kwargs: Any) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        import json

        return json.loads(super().json(**kwargs))

    def with_updates(self: T, **kwargs: Any) -> T:
        """Create a new config with updated fields."""
        return self.__class__(**{**self.dict(), **kwargs})


class ModelConfig(BaseConfig):
    """Base configuration for all models."""

    # Device and precision settings
    device: str | None = Field(
        None, description="Device to use (cuda/cpu/auto)"
    )
    dtype: str = Field(
        "float32", description="Data type (float32/float16/bfloat16)"
    )

    # Random seed
    seed: int | None = Field(
        None, description="Random seed for reproducibility"
    )

    @validator("device")
    def validate_device(cls, v: str | None) -> str | None:  # noqa: N805
        """Validate and normalize device string."""
        if v is None or v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if v not in {"cuda", "cpu", "mps"} and not v.startswith("cuda:"):
            raise ValueError(f"Invalid device: {v}")
        return v

    @validator("dtype")
    def validate_dtype(cls, v: str) -> str:  # noqa: N805
        """Validate data type string."""
        valid_dtypes = {
            "float32",
            "float",
            "fp32",
            "float16",
            "half",
            "fp16",
            "bfloat16",
            "bf16",
            "float64",
            "double",
            "fp64",
        }
        if v not in valid_dtypes:
            raise ValueError(
                f"Invalid dtype: {v}. Must be one of {valid_dtypes}"
            )
        return v

    @property
    def torch_device(self) -> torch.device:
        """Get torch device object."""
        return torch.device(self.device or "cpu")

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get torch dtype object."""
        dtype_map = {
            "float32": torch.float32,
            "float": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float64": torch.float64,
            "double": torch.float64,
            "fp64": torch.float64,
        }
        return dtype_map[self.dtype]


class OptimizerConfig(BaseConfig):
    """Configuration for optimizers."""

    name: str = Field("adam", description="Optimizer name")
    lr: float = Field(1e-3, description="Learning rate")
    weight_decay: float = Field(0.0, description="Weight decay")

    # Adam-specific
    betas: tuple[float, float] = Field((0.9, 0.999), description="Adam betas")
    eps: float = Field(1e-8, description="Adam epsilon", gt=0)

    # SGD-specific
    momentum: float = Field(0.0, description="SGD momentum", ge=0)
    nesterov: bool = Field(False, description="Use Nesterov momentum")

    # Learning rate scheduling
    scheduler: str | None = Field(None, description="LR scheduler type")
    scheduler_params: dict[str, Any] = Field(
        {}, description="Scheduler parameters"
    )

    @validator("name")
    def validate_optimizer(cls, v: str) -> str:  # noqa: N805
        """Validate optimizer name."""
        valid = {"adam", "adamw", "sgd", "rmsprop", "lbfgs"}
        if v.lower() not in valid:
            raise ValueError(f"Unknown optimizer: {v}. Must be one of {valid}")
        return v.lower()

    @validator("lr")
    def validate_lr(cls, v: float) -> float:  # noqa: N805
        """Ensure learning rate is positive."""
        if v <= 0:
            raise ValueError("ensure this value is greater than 0")
        return v

    @validator("weight_decay")
    def validate_weight_decay(cls, v: float) -> float:  # noqa: N805
        """Ensure weight decay is non-negative."""
        if v < 0:
            raise ValueError("ensure this value is greater than or equal to 0")
        return v


class TrainingConfig(BaseConfig):
    """Configuration for training loop."""

    # Basic training parameters
    epochs: int = Field(100, description="Number of epochs")
    batch_size: int = Field(64, description="Batch size")
    eval_batch_size: int | None = Field(
        None, description="Evaluation batch size"
    )

    # Optimization
    optimizer: OptimizerConfig = Field(
        OptimizerConfig(), description="Optimizer config"
    )
    grad_clip: float | None = Field(
        None, description="Gradient clipping value", gt=0
    )

    # Checkpointing and logging
    checkpoint_dir: Path = Field(
        Path("checkpoints"), description="Checkpoint directory"
    )
    checkpoint_every: int = Field(
        10, description="Checkpoint frequency (epochs)", gt=0
    )
    log_every: int = Field(100, description="Logging frequency (steps)", gt=0)

    # Evaluation
    eval_every: int = Field(
        1, description="Evaluation frequency (epochs)", gt=0
    )
    eval_samples: int = Field(
        1000, description="Number of evaluation samples", gt=0
    )

    # Early stopping
    early_stopping: bool = Field(False, description="Enable early stopping")
    patience: int = Field(10, description="Early stopping patience", gt=0)
    min_delta: float = Field(
        1e-4, description="Minimum improvement for early stopping", gt=0
    )

    # Advanced features
    mixed_precision: bool = Field(
        False, description="Use automatic mixed precision"
    )
    compile_model: bool = Field(
        False, description="Use torch.compile (PyTorch 2.0+)"
    )
    num_workers: int = Field(0, description="DataLoader workers", ge=0)
    pin_memory: bool = Field(True, description="Pin memory for DataLoader")

    @validator("epochs", "batch_size")
    def validate_positive(cls, v: int) -> int:  # noqa: N805
        """Ensure integer fields expected to be positive are > 0."""
        if v <= 0:
            raise ValueError("ensure this value is greater than 0")
        return v

    @property
    def eval_batch_size_actual(self) -> int:
        """Get actual evaluation batch size."""
        return self.eval_batch_size or self.batch_size


class SamplerConfig(BaseConfig):
    """Base configuration for samplers."""

    num_steps: int = Field(1, description="Number of sampling steps", gt=0)


class GibbsConfig(SamplerConfig):
    """Configuration for Gibbs sampling."""

    block_gibbs: bool = Field(True, description="Use block Gibbs sampling")


class CDConfig(SamplerConfig):
    """Configuration for Contrastive Divergence."""

    persistent: bool = Field(False, description="Use persistent CD")
    num_chains: int | None = Field(
        None, description="Number of persistent chains"
    )


class ParallelTemperingConfig(SamplerConfig):
    """Configuration for Parallel Tempering."""

    num_temps: int = Field(10, description="Number of temperatures", gt=1)
    min_beta: float = Field(
        0.01, description="Minimum inverse temperature", gt=0, le=1
    )
    max_beta: float = Field(
        1.0, description="Maximum inverse temperature", gt=0, le=1
    )
    swap_every: int = Field(1, description="Swap frequency", gt=0)

    @validator("min_beta")
    def validate_beta_range(cls, v: float, values: dict[str, Any]) -> float:  # noqa: N805
        """Ensure min_beta < max_beta."""
        if "max_beta" in values and v >= values["max_beta"]:
            raise ValueError("min_beta must be less than max_beta")
        return v


# Model-specific configurations
class RBMConfig(ModelConfig):
    """Configuration for Restricted Boltzmann Machines."""

    visible_units: int = Field(..., description="Number of visible units")
    hidden_units: int = Field(..., description="Number of hidden units")

    # Initialization
    weight_init: str = Field(
        "xavier_normal", description="Weight initialization method"
    )
    bias_init: str | float = Field(0.0, description="Bias initialization")

    # Architecture variants
    use_bias: bool = Field(True, description="Use bias terms")
    centered: bool = Field(False, description="Use centered RBM variant")

    # Regularization
    l2_weight: float = Field(0.0, description="L2 weight regularization")
    l1_weight: float = Field(0.0, description="L1 weight regularization")

    @validator("visible_units", "hidden_units")
    def validate_units(cls, v: int) -> int:  # noqa: N805
        """Ensure unit counts are positive."""
        if v <= 0:
            raise ValueError("ensure this value is greater than 0")
        return v

    @validator("l2_weight", "l1_weight")
    def validate_regularization(cls, v: float) -> float:  # noqa: N805
        """Ensure regularization weights are non-negative."""
        if v < 0:
            raise ValueError("ensure this value is greater than or equal to 0")
        return v

    @validator("weight_init")
    def validate_init_method(cls, v: str) -> str:  # noqa: N805
        """Validate initialization method."""
        valid_methods = {
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "he_uniform",
            "he_normal",
            "normal",
            "uniform",
            "zeros",
            "ones",
        }
        if v not in valid_methods:
            raise ValueError(
                f"Unknown init method: {v}. Must be one of {valid_methods}"
            )
        return v


class GaussianRBMConfig(RBMConfig):
    """Configuration for Gaussian-Bernoulli RBM."""

    visible_type: str = Field("gaussian", description="Visible unit type")
    sigma: float = Field(
        1.0, description="Standard deviation for Gaussian units", gt=0
    )
    learn_sigma: bool = Field(False, description="Learn sigma during training")
