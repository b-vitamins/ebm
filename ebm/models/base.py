"""Base classes and utilities for energy-based models.

This module provides abstract base classes and common utilities for
implementing various energy-based models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ebm.core.config import ModelConfig
from ebm.core.device import DeviceManager
from ebm.core.logging_utils import LoggerMixin
from ebm.core.types_ import Device
from ebm.utils.tensor import ensure_tensor


class EnergyBasedModel(nn.Module, LoggerMixin, ABC):
    """Abstract base class for all energy-based models.

    This class provides the core interface and common functionality for
    energy-based models, including device management, logging, and
    serialization.
    """

    def __new__(cls, *_args: Any, **_kwargs: Any) -> EnergyBasedModel:
        """Create instance and initialize base classes."""
        obj = super().__new__(cls)
        nn.Module.__init__(obj)
        LoggerMixin.__init__(obj)
        return obj

    def __setattr__(self, name: str, value: object) -> None:
        """Allow assigning tensors to parameter attributes."""
        if (
            isinstance(value, torch.Tensor)
            and not isinstance(value, nn.Parameter)
            and isinstance(
                getattr(self, name, None),
                nn.Parameter,
            )
        ):
            value = nn.Parameter(value)
        super().__setattr__(name, value)

    def __init__(self, config: ModelConfig):
        """Initialize the model.

        Args:
            config: Model configuration
        """
        super().__init__()
        # Initialize logging via LoggerMixin
        LoggerMixin.__init__(self)
        self.config = config
        self._device_manager = DeviceManager(config.device)

        # Set random seed if specified
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.seed)

        # Initialize parameters (to be implemented by subclasses)
        self._build_model()

        # Move to specified device and dtype
        self.to(device=self.device, dtype=self.dtype)

        self.log_info(
            "Initialized model",
            model_type=self.__class__.__name__,
            device=str(self.device),
            dtype=str(self.dtype),
        )

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return self._device_manager.device

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        return self.config.torch_dtype

    def _build_model(self) -> None:
        """Build model architecture. Can be overridden by subclasses."""
        raise NotImplementedError

    def energy(
        self,
        x: Tensor,
        *,
        beta: Tensor | None = None,
        return_parts: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Compute energy of configurations.

        Args:
            x: Input configurations of shape (batch_size, *dims)
            beta: Optional inverse temperature for parallel tempering
            return_parts: If True, return dict with energy components

        Returns
        -------
            Energy values of shape (batch_size,) or dict of components
        """
        raise NotImplementedError

    def free_energy(self, v: Tensor, *, beta: Tensor | None = None) -> Tensor:
        """Compute free energy by marginalizing latent variables.

        Args:
            v: Visible configurations of shape (batch_size, *visible_dims)
            beta: Optional inverse temperature

        Returns
        -------
            Free energy values of shape (batch_size,)
        """
        raise NotImplementedError

    def log_probability(
        self, x: Tensor, *, log_z: float | None = None
    ) -> Tensor:
        """Compute log probability of configurations.

        Args:
            x: Input configurations
            log_z: Log partition function. If None, returns unnormalized log prob.

        Returns
        -------
            Log probabilities of shape (batch_size,)
        """
        # ``log_probability`` can be called either with concatenated visible and
        # hidden states or with only visible states.  In the latter case we
        # compute the free energy of the visible configuration.
        if hasattr(self, "num_visible") and x.shape[-1] == self.num_visible:
            energy = self.free_energy(x)
        else:
            energy = self.energy(x)
        log_prob = -energy

        if log_z is not None:
            log_prob = log_prob - log_z

        return log_prob

    def to_device(self, x: Tensor) -> Tensor:
        """Move tensor to model device with correct dtype."""
        return x.to(device=self.device, dtype=self.dtype)

    def prepare_input(self, x: Tensor) -> Tensor:
        """Prepare input tensor (ensure correct device/dtype)."""
        x = ensure_tensor(x)
        return self.to_device(x)

    def save_checkpoint(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            metadata: Optional metadata to include
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.dict(),
            "model_class": self.__class__.__name__,
        }

        if metadata:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, path)
        self.log_info(f"Saved checkpoint to {path}")

    def load_checkpoint(
        self,
        path: str | Path,
        strict: bool = True,
        map_location: Device | None = None,
    ) -> dict[str, Any]:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
            strict: Whether to strictly enforce state dict matching
            map_location: Device to map checkpoint to

        Returns
        -------
            Checkpoint metadata
        """
        path = Path(path)
        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(path, map_location=map_location)

        # Load state dict
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        self.log_info(f"Loaded checkpoint from {path}")
        return checkpoint.get("metadata", {})

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: Device | None = None,
        **config_overrides: Any,
    ) -> EnergyBasedModel:
        """Create model instance from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model on
            **config_overrides: Config parameters to override

        Returns
        -------
            Loaded model instance
        """
        # Load checkpoint to CPU first
        checkpoint = torch.load(path, map_location="cpu")

        # Reconstruct config
        config_dict = checkpoint["config"]
        if device is not None:
            config_dict["device"] = device
        config_dict.update(config_overrides)

        # Get config class from model class
        config_cls = cls.get_config_class()
        config = config_cls(**config_dict)

        # Create model
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    @classmethod
    def get_config_class(cls) -> type[ModelConfig]:
        """Get configuration class for this model."""
        raise NotImplementedError

    def reset_parameters(self) -> None:
        """Reset all model parameters to initial values."""
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module != self:
                module.reset_parameters()

    def parameter_summary(self) -> dict[str, Any]:
        """Get summary of model parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "model_size_mb": total_params
            * 4
            / (1024 * 1024),  # Assuming float32
        }

    def __repr__(self) -> str:
        """Return representation with configuration values."""
        config_str = ", ".join(
            f"{k}={v}" for k, v in self.config.dict().items()
        )
        return f"{self.__class__.__name__}({config_str})"


class LatentVariableModel(EnergyBasedModel, ABC):
    """Base class for models with explicit latent variables.

    This includes models like RBMs, VAEs, and other models that have
    a clear separation between visible and hidden/latent variables.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap subclass sampling methods to handle float betas."""
        super().__init_subclass__(**kwargs)

        def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(
                self: LatentVariableModel,
                *args: Any,
                beta: Tensor | float | None = None,
                **kw: Any,
            ) -> Tensor | tuple[Tensor, Tensor]:
                if beta is not None and not isinstance(beta, Tensor):
                    beta = torch.tensor(beta, device=args[0].device)
                return func(self, *args, beta=beta, **kw)

            return wrapper

        if hasattr(cls, "sample_hidden"):
            cls.sample_hidden = _wrap(cls.sample_hidden)
        if hasattr(cls, "sample_visible"):
            cls.sample_visible = _wrap(cls.sample_visible)

    @abstractmethod
    def sample_hidden(
        self,
        visible: Tensor,
        *,
        beta: Tensor | None = None,
        return_prob: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample hidden/latent units given visible units.

        Args:
            visible: Visible unit values
            beta: Optional inverse temperature
            return_prob: If True, also return probabilities

        Returns
        -------
            Sampled hidden states, optionally with probabilities
        """

    @abstractmethod
    def sample_visible(
        self,
        hidden: Tensor,
        *,
        beta: Tensor | None = None,
        return_prob: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample visible units given hidden units.

        Args:
            hidden: Hidden unit values
            beta: Optional inverse temperature
            return_prob: If True, also return probabilities

        Returns
        -------
            Sampled visible states, optionally with probabilities
        """

    def joint_energy(
        self, visible: Tensor, hidden: Tensor, *, beta: Tensor | None = None
    ) -> Tensor:
        """Compute joint energy of visible and hidden configurations.

        Args:
            visible: Visible unit values
            hidden: Hidden unit values
            beta: Optional inverse temperature

        Returns
        -------
            Joint energy values
        """
        # Default implementation concatenates and calls energy
        joint = torch.cat([visible, hidden], dim=-1)
        return self.energy(joint, beta=beta)

    def gibbs_step(
        self,
        visible: Tensor,
        *,
        beta: Tensor | None = None,
        start_from: str = "visible",
    ) -> tuple[Tensor, Tensor]:
        """Perform one step of Gibbs sampling.

        Args:
            visible: Initial visible state
            beta: Optional inverse temperature
            start_from: Whether to start from 'visible' or 'hidden'

        Returns
        -------
            New visible and hidden states
        """
        if start_from == "visible":
            hidden = self.sample_hidden(visible, beta=beta)
            visible = self.sample_visible(hidden, beta=beta)
        else:
            hidden = self.sample_hidden(visible, beta=beta)
            visible = self.sample_visible(hidden, beta=beta)
            hidden = self.sample_hidden(visible, beta=beta)

        return visible, hidden

    def reconstruct(
        self, visible: Tensor, *, num_steps: int = 1, beta: Tensor | None = None
    ) -> Tensor:
        """Reconstruct visible units through sampling.

        Args:
            visible: Input visible units
            num_steps: Number of Gibbs steps
            beta: Optional inverse temperature

        Returns
        -------
            Reconstructed visible units
        """
        v = visible
        for _ in range(num_steps):
            v, _ = self.gibbs_step(v, beta=beta)
        return v

    @abstractmethod
    def sample_fantasy_particles(
        self, num_samples: int, num_steps: int
    ) -> Tensor:
        """Generate samples via Gibbs sampling for diagnostics."""
        raise NotImplementedError


class AISInterpolator(nn.Module):
    """Mixin for models that support AIS interpolation.

    This mixin modifies model behavior to interpolate between a base
    distribution and the target distribution for Annealed Importance Sampling.
    """

    def __init__(self, base_model: EnergyBasedModel):
        """Initialize interpolator.

        Args:
            base_model: The model to wrap for AIS
        """
        super().__init__()
        self.base_model = base_model
        self._ais_beta = 1.0

    @property
    def ais_beta(self) -> float:
        """Current AIS interpolation parameter."""
        return self._ais_beta

    @ais_beta.setter
    def ais_beta(self, value: float) -> None:
        """Set AIS interpolation parameter."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"AIS beta must be in [0, 1], got {value}")
        self._ais_beta = value

    @abstractmethod
    def base_log_partition(self) -> float:
        """Compute log partition function of base distribution."""

    def interpolated_energy(
        self, x: Tensor, beta: float | None = None
    ) -> Tensor:
        """Compute interpolated energy for AIS.

        Args:
            x: Input configurations
            beta: AIS interpolation parameter (overrides self.ais_beta)

        Returns
        -------
            Interpolated energy values
        """
        if beta is None:
            beta = self.ais_beta

        # Default implementation - should be overridden for efficiency
        base_energy = self.base_energy(x)
        target_energy = self.base_model.energy(x)

        return (1 - beta) * base_energy + beta * target_energy

    @abstractmethod
    def base_energy(self, x: Tensor) -> Tensor:
        """Compute energy under base distribution."""
