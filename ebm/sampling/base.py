"""Base classes and interfaces for sampling algorithms.

This module defines the abstract interfaces for samplers used in
training and inference of energy-based models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from ebm.core.logging import LoggerMixin
from ebm.core.types import ChainState
from ebm.models.base import EnergyBasedModel, LatentVariableModel


@dataclass
class SamplerState:
    """State maintained by a sampler across batches."""

    # Current chain states
    chains: ChainState | None = None

    # Number of steps taken
    num_steps: int = 0

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset sampler state."""
        self.chains = None
        self.num_steps = 0
        self.metadata.clear()


class Sampler(nn.Module, LoggerMixin, ABC):
    """Abstract base class for all samplers.

    Samplers are responsible for generating samples from energy-based models
    and estimating gradients for training.
    """

    def __init__(self, name: str | None = None):
        """Initialize sampler.

        Args:
            name: Optional name for the sampler
        """
        super().__init__()
        self.name = name or self.__class__.__name__
        self.state = SamplerState()

    @abstractmethod
    def sample(
        self,
        model: EnergyBasedModel,
        init_state: Tensor,
        num_steps: int = 1,
        **kwargs: Any,
    ) -> Tensor:
        """Generate samples from the model.

        Args:
            model: Energy model to sample from
            init_state: Initial state for the Markov chain
            num_steps: Number of sampling steps
            **kwargs: Additional sampler-specific arguments

        Returns
        -------
            Samples from the model
        """

    def reset(self) -> None:
        """Reset sampler state."""
        self.state.reset()
        self.log_debug("Reset sampler state")

    @property
    def num_steps_taken(self) -> int:
        """Total number of sampling steps taken."""
        return self.state.num_steps

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information about the sampler.

        Returns
        -------
            Dictionary of diagnostic metrics
        """
        return {
            "num_steps": self.num_steps_taken,
            "has_chains": self.state.chains is not None,
        }


class GibbsSampler(Sampler):
    """Base class for Gibbs sampling algorithms."""

    def __init__(self, name: str | None = None, block_gibbs: bool = True):
        """Initialize Gibbs sampler.

        Args:
            name: Optional name for the sampler
            block_gibbs: Whether to use block Gibbs sampling
        """
        super().__init__(name)
        self.block_gibbs = block_gibbs

    def gibbs_step(
        self,
        model: LatentVariableModel,
        visible: Tensor,
        *,
        beta: Tensor | None = None,
        start_from: str = "visible",
    ) -> tuple[Tensor, Tensor]:
        """Perform one Gibbs sampling step.

        Args:
            model: Latent variable model
            visible: Current visible state
            beta: Optional inverse temperature
            start_from: Whether to start from 'visible' or 'hidden'

        Returns
        -------
            New (visible, hidden) states
        """
        if start_from == "visible":
            # v -> h -> v
            hidden = model.sample_hidden(visible, beta=beta)
            visible = model.sample_visible(hidden, beta=beta)
        else:
            # v -> h -> v -> h
            hidden = model.sample_hidden(visible, beta=beta)
            visible = model.sample_visible(hidden, beta=beta)
            hidden = model.sample_hidden(visible, beta=beta)

        return visible, hidden

    def sample(
        self,
        model: EnergyBasedModel,
        init_state: Tensor,
        num_steps: int = 1,
        *,
        beta: Tensor | None = None,
        return_all: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Generate samples using Gibbs sampling.

        Args:
            model: Must be a LatentVariableModel
            init_state: Initial visible state
            num_steps: Number of Gibbs steps
            beta: Optional inverse temperature
            return_all: If True, return all intermediate states
            **kwargs: Additional arguments

        Returns
        -------
            Final samples (or all samples if return_all=True)
        """
        if not isinstance(model, LatentVariableModel):
            raise TypeError(
                f"Gibbs sampling requires LatentVariableModel, got {type(model)}"
            )

        visible = init_state
        states = [visible] if return_all else None

        for _ in range(num_steps):
            visible, _ = self.gibbs_step(model, visible, beta=beta)
            if return_all:
                states.append(visible)

        self.state.num_steps += num_steps

        if return_all:
            return torch.stack(states, dim=0)
        return visible


class MCMCSampler(Sampler):
    """Base class for Markov Chain Monte Carlo samplers."""

    def __init__(self, name: str | None = None, num_chains: int | None = None):
        """Initialize MCMC sampler.

        Args:
            name: Optional name
            num_chains: Number of parallel chains
        """
        super().__init__(name)
        self.num_chains = num_chains

    @abstractmethod
    def transition_kernel(
        self, model: EnergyBasedModel, state: Tensor, **kwargs: Any
    ) -> tuple[Tensor, dict[str, Any]]:
        """Apply one transition of the Markov chain.

        Args:
            model: Energy model
            state: Current state
            **kwargs: Kernel-specific arguments

        Returns
        -------
            New state and transition metadata
        """

    def sample(
        self,
        model: EnergyBasedModel,
        init_state: Tensor,
        num_steps: int = 1,
        *,
        thin: int = 1,
        burn_in: int = 0,
        return_all: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Generate samples using MCMC.

        Args:
            model: Energy model
            init_state: Initial state
            num_steps: Number of MCMC steps
            thin: Thinning factor (keep every thin-th sample)
            burn_in: Number of burn-in steps
            return_all: If True, return all kept samples
            **kwargs: Kernel-specific arguments

        Returns
        -------
            Samples from the chain
        """
        state = init_state
        samples = []

        # Burn-in phase
        for _ in range(burn_in):
            state, _ = self.transition_kernel(model, state, **kwargs)

        # Sampling phase
        for i in range(num_steps):
            state, metadata = self.transition_kernel(model, state, **kwargs)

            if i % thin == 0 and return_all:
                samples.append(state)

        self.state.num_steps += burn_in + num_steps

        if return_all:
            return torch.stack(samples, dim=0)
        return state


class GradientEstimator(nn.Module, LoggerMixin, ABC):
    """Base class for gradient estimation methods."""

    def __init__(self, sampler: Sampler):
        """Initialize gradient estimator.

        Args:
            sampler: Sampler to use for negative phase
        """
        super().__init__()
        self.sampler = sampler

    @abstractmethod
    def estimate_gradient(
        self, model: EnergyBasedModel, data: Tensor, **kwargs: Any
    ) -> dict[str, Tensor]:
        """Estimate gradients for model parameters.

        Args:
            model: Energy model to train
            data: Training data batch
            **kwargs: Estimator-specific arguments

        Returns
        -------
            Dictionary mapping parameter names to gradients
        """

    def compute_metrics(
        self, model: EnergyBasedModel, data: Tensor, samples: Tensor
    ) -> dict[str, float]:
        """Compute training metrics.

        Args:
            model: Energy model
            data: Data samples
            samples: Model samples

        Returns
        -------
            Dictionary of metrics
        """
        with torch.no_grad():
            # Energy statistics
            data_energy = model.free_energy(data).mean()
            sample_energy = model.free_energy(samples).mean()

            # Reconstruction error (if applicable)
            if isinstance(model, LatentVariableModel):
                recon = model.reconstruct(data, num_steps=1)
                recon_error = (data - recon).pow(2).mean()
            else:
                recon_error = 0.0

        return {
            "data_energy": float(data_energy),
            "sample_energy": float(sample_energy),
            "energy_gap": float(sample_energy - data_energy),
            "reconstruction_error": float(recon_error),
        }


class AnnealedSampler(Sampler):
    """Base class for samplers using annealing/tempering."""

    def __init__(
        self,
        name: str | None = None,
        num_temps: int = 10,
        min_beta: float = 0.01,
        max_beta: float = 1.0,
    ):
        """Initialize annealed sampler.

        Args:
            name: Optional name
            num_temps: Number of temperature levels
            min_beta: Minimum inverse temperature
            max_beta: Maximum inverse temperature
        """
        super().__init__(name)
        self.num_temps = num_temps
        self.min_beta = min_beta
        self.max_beta = max_beta

        # Create temperature schedule
        self._create_schedule()

    def _create_schedule(self) -> None:
        """Create temperature schedule."""
        # Geometric schedule often works better than linear
        log_betas = torch.linspace(
            torch.log(torch.tensor(self.min_beta)),
            torch.log(torch.tensor(self.max_beta)),
            self.num_temps,
        )
        self.register_buffer("betas", torch.exp(log_betas))

    @property
    def temperatures(self) -> Tensor:
        """Get temperature values (1/beta)."""
        return 1.0 / self.betas
