"""Core type definitions and protocols for the EBM library.

This module defines the fundamental types and interfaces used throughout
the library, leveraging Python's Protocol feature for better flexibility
than traditional ABCs.
"""

from __future__ import annotations

from typing import (
    Any,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import torch
from torch import Tensor

# Type aliases for common patterns
TensorLike: TypeAlias = Tensor | list[float] | float
Device: TypeAlias = torch.device | str | None
DType: TypeAlias = torch.dtype | None
Shape: TypeAlias = torch.Size | tuple[int, ...] | list[int]

# Type variables for generics
T = TypeVar("T")
ModelT = TypeVar("ModelT", bound="EnergyModel")
SamplerT = TypeVar("SamplerT", bound="Sampler")
ConfigT = TypeVar("ConfigT", bound="Config")


@runtime_checkable
class Config(Protocol):
    """Protocol for configuration objects.

    All configuration classes should be immutable (frozen dataclasses or Pydantic models)
    and provide a method to convert to a dictionary for serialization.
    """

    def dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        ...

    def validate(self) -> None:
        """Validate configuration parameters."""
        ...


@runtime_checkable
class EnergyModel(Protocol):
    """Protocol for energy-based models.

    This defines the minimal interface that all energy-based models must implement.
    Models should be able to compute energies and related quantities.
    """

    @property
    def device(self) -> torch.device:
        """Device where model parameters are stored."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Data type of model parameters."""
        ...

    def energy(self, x: Tensor, *, beta: Tensor | None = None) -> Tensor:
        """Compute energy of configurations.

        Args:
            x: Input configurations of shape (batch_size, *dims)
            beta: Optional inverse temperature for parallel tempering

        Returns:
            Energy values of shape (batch_size,) or (batch_size, num_chains)
        """
        ...

    def free_energy(self, v: Tensor, *, beta: Tensor | None = None) -> Tensor:
        """Compute free energy by marginalizing latent variables.

        Args:
            v: Visible configurations of shape (batch_size, *visible_dims)
            beta: Optional inverse temperature

        Returns:
            Free energy values of shape (batch_size,) or (batch_size, num_chains)
        """
        ...


@runtime_checkable
class LatentModel(EnergyModel, Protocol):
    """Protocol for models with explicit latent variables (e.g., RBMs)."""

    def sample_hidden(
        self, visible: Tensor, *, beta: Tensor | None = None, return_prob: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample hidden units given visible units.

        Args:
            visible: Visible unit values
            beta: Optional inverse temperature
            return_prob: If True, also return probabilities

        Returns:
            Sampled hidden states, optionally with probabilities
        """
        ...

    def sample_visible(
        self, hidden: Tensor, *, beta: Tensor | None = None, return_prob: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample visible units given hidden units.

        Args:
            hidden: Hidden unit values
            beta: Optional inverse temperature
            return_prob: If True, also return probabilities

        Returns:
            Sampled visible states, optionally with probabilities
        """
        ...


@runtime_checkable
class Sampler(Protocol):
    """Protocol for sampling algorithms."""

    def sample(
        self, model: EnergyModel, init_state: Tensor, num_steps: int = 1, **kwargs: Any
    ) -> Tensor:
        """Generate samples from the model.

        Args:
            model: Energy model to sample from
            init_state: Initial state for the Markov chain
            num_steps: Number of sampling steps
            **kwargs: Additional sampler-specific arguments

        Returns:
            Samples of shape (batch_size, *dims)
        """
        ...

    def reset(self) -> None:
        """Reset any internal state of the sampler."""
        ...


@runtime_checkable
class GradientEstimator(Protocol):
    """Protocol for gradient estimation methods (e.g., CD, PCD)."""

    def estimate_gradient(
        self, model: EnergyModel, data: Tensor, **kwargs: Any
    ) -> dict[str, Tensor]:
        """Estimate gradients for model parameters.

        Args:
            model: Energy model to train
            data: Training data batch
            **kwargs: Estimator-specific arguments

        Returns:
            Dictionary mapping parameter names to gradients
        """
        ...


@runtime_checkable
class Callback(Protocol):
    """Protocol for training callbacks."""

    def on_epoch_start(self, trainer: Any, model: EnergyModel) -> None:
        """Called at the start of each epoch."""
        ...

    def on_epoch_end(
        self, trainer: Any, model: EnergyModel, metrics: dict[str, float]
    ) -> None:
        """Called at the end of each epoch."""
        ...

    def on_batch_start(self, trainer: Any, model: EnergyModel, batch: Tensor) -> None:
        """Called before processing each batch."""
        ...

    def on_batch_end(self, trainer: Any, model: EnergyModel, loss: float) -> None:
        """Called after processing each batch."""
        ...


@runtime_checkable
class Transform(Protocol):
    """Protocol for data transformations."""

    def __call__(self, x: Tensor) -> Tensor:
        """Apply transformation to input tensor."""
        ...

    def inverse(self, x: Tensor) -> Tensor:
        """Apply inverse transformation."""
        ...


# Specialized type hints for common patterns
EnergyFunction = TypeVar("EnergyFunction", bound=callable)
ActivationFunction: TypeAlias = callable[[Tensor], Tensor]
InitStrategy: TypeAlias = str | float | Tensor | callable


# Constants for common initialization strategies
class InitMethod:
    """Namespace for initialization method names."""

    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    NORMAL = "normal"
    UNIFORM = "uniform"
    ZEROS = "zeros"
    ONES = "ones"


# Sampling-related types
SamplingMetadata = dict[str, Any]
ChainState = Tensor | tuple[Tensor, ...]
