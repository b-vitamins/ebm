"""Contrastive Divergence and Persistent Contrastive Divergence samplers.

This module implements the standard gradient estimation methods for
training RBMs and other energy-based models.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ebm.core.registry import register_sampler
from ebm.models.base import EnergyBasedModel, LatentVariableModel
from ebm.utils.tensor import batch_outer_product

from .base import GibbsSampler, GradientEstimator


@register_sampler('cd', aliases=['contrastive_divergence'])
class ContrastiveDivergence(GradientEstimator):
    """Contrastive Divergence (CD-k) gradient estimator.

    CD approximates the model distribution by running k steps of Gibbs
    sampling starting from the data distribution.
    """

    def __init__(
        self,
        k: int = 1,
        *,
        persistent: bool = False,
        num_chains: int | None = None
    ):
        """Initialize CD sampler.

        Args:
            k: Number of Gibbs sampling steps
            persistent: If True, use persistent chains (PCD)
            num_chains: Number of persistent chains (for PCD)
        """
        sampler = CDSampler(k=k, persistent=persistent, num_chains=num_chains)
        super().__init__(sampler)
        self.k = k
        self.persistent = persistent
        self.num_chains = num_chains

    def estimate_gradient(
        self,
        model: EnergyBasedModel,
        data: Tensor,
        **kwargs: Any
    ) -> dict[str, Tensor]:
        """Estimate gradients using contrastive divergence.

        Args:
            model: Energy model (must be LatentVariableModel)
            data: Training data batch
            **kwargs: Additional arguments

        Returns
        -------
            Dictionary of parameter gradients
        """
        if not isinstance(model, LatentVariableModel):
            raise TypeError("CD requires a LatentVariableModel")

        # Positive phase: sample hidden given data
        h_data = model.sample_hidden(data, return_prob=True)[1]  # Use probabilities

        # Negative phase: run k steps of Gibbs sampling
        v_model = self.sampler.sample(model, data, num_steps=self.k)
        h_model = model.sample_hidden(v_model, return_prob=True)[1]

        # Compute gradients
        gradients = {}

        # Weight gradient: <v h^T>_data - <v h^T>_model
        pos_stats = batch_outer_product(h_data, data).mean(dim=0)
        neg_stats = batch_outer_product(h_model, v_model).mean(dim=0)
        gradients['W'] = pos_stats - neg_stats

        # Bias gradients
        if hasattr(model, 'vbias') and model.vbias.requires_grad:
            gradients['vbias'] = data.mean(dim=0) - v_model.mean(dim=0)

        if hasattr(model, 'hbias') and model.hbias.requires_grad:
            gradients['hbias'] = h_data.mean(dim=0) - h_model.mean(dim=0)

        # Store negative samples for metrics
        self.last_negative_samples = v_model.detach()

        return gradients

    def apply_gradients(
        self,
        model: nn.Module,
        gradients: dict[str, Tensor],
        lr: float = 0.01
    ) -> None:
        """Apply gradients to model parameters.

        Args:
            model: Model to update
            gradients: Gradient dictionary
            lr: Learning rate
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in gradients:
                    # Gradient ascent (maximize likelihood)
                    param.add_(gradients[name], alpha=lr)


class CDSampler(GibbsSampler):
    """Gibbs sampler for Contrastive Divergence."""

    def __init__(
        self,
        k: int = 1,
        *,
        persistent: bool = False,
        num_chains: int | None = None
    ):
        """Initialize CD sampler.

        Args:
            k: Number of Gibbs steps
            persistent: Use persistent chains
            num_chains: Number of persistent chains
        """
        super().__init__(name=f"CD-{k}" if not persistent else f"PCD-{k}")
        self.k = k
        self.persistent = persistent
        self.num_chains = num_chains

        if persistent:
            self.register_buffer('persistent_chains', None)

    def sample(
        self,
        model: EnergyBasedModel,
        init_state: Tensor,
        num_steps: int | None = None,
        **kwargs: Any
    ) -> Tensor:
        """Run CD sampling.

        Args:
            model: Energy model
            init_state: Initial state (data for CD, ignored for PCD)
            num_steps: Number of steps (defaults to k)
            **kwargs: Additional arguments

        Returns
        -------
            Negative samples
        """
        if num_steps is None:
            num_steps = self.k

        if self.persistent:
            # Use persistent chains
            if self.persistent_chains is None:
                # Initialize chains
                num_chains = self.num_chains or init_state.shape[0]
                self.persistent_chains = init_state[:num_chains].clone()

            # Start from persistent state
            state = self.persistent_chains
        else:
            # Start from data
            state = init_state

        # Run Gibbs sampling
        final_state = super().sample(
            model, state, num_steps=num_steps, **kwargs
        )

        if self.persistent:
            # Update persistent chains
            self.persistent_chains = final_state.detach()

        return final_state

    def reset(self) -> None:
        """Reset sampler state."""
        super().reset()
        if self.persistent:
            self.persistent_chains = None


@register_sampler('pcd', aliases=['persistent_cd'])
class PersistentContrastiveDivergence(ContrastiveDivergence):
    """Persistent Contrastive Divergence (PCD) gradient estimator.

    PCD maintains persistent Markov chains that are updated after each
    gradient step, providing better samples of the model distribution.
    """

    def __init__(
        self,
        k: int = 1,
        num_chains: int | None = None
    ):
        """Initialize PCD sampler.

        Args:
            k: Number of Gibbs steps per update
            num_chains: Number of persistent chains
        """
        super().__init__(k=k, persistent=True, num_chains=num_chains)


@register_sampler('fast_pcd', aliases=['fpcd'])
class FastPersistentCD(PersistentContrastiveDivergence):
    """Fast Persistent Contrastive Divergence with momentum.

    FPCD uses a modified parameter update that includes momentum terms
    to accelerate mixing of the persistent chains.
    """

    def __init__(
        self,
        k: int = 1,
        num_chains: int | None = None,
        momentum: float = 0.9,
        fast_weight_scale: float = 5.0
    ):
        """Initialize Fast PCD.

        Args:
            k: Number of Gibbs steps
            num_chains: Number of chains
            momentum: Momentum coefficient
            fast_weight_scale: Scale for fast weights
        """
        super().__init__(k=k, num_chains=num_chains)
        self.momentum = momentum
        self.fast_weight_scale = fast_weight_scale

        # Storage for parameter velocities
        self.velocities = {}

    def estimate_gradient(
        self,
        model: EnergyBasedModel,
        data: Tensor,
        **kwargs: Any
    ) -> dict[str, Tensor]:
        """Estimate gradients with fast weights.

        Args:
            model: Energy model
            data: Training data
            **kwargs: Additional arguments

        Returns
        -------
            Parameter gradients
        """
        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Apply fast weights for negative sampling
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.velocities:
                    # Apply momentum update
                    param.add_(self.velocities[name], alpha=self.fast_weight_scale)

        # Get gradients with fast weights
        gradients = super().estimate_gradient(model, data, **kwargs)

        # Restore original parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(original_params[name])

        # Update velocities
        for name, grad in gradients.items():
            if name not in self.velocities:
                self.velocities[name] = torch.zeros_like(grad)
            self.velocities[name].mul_(self.momentum).add_(grad, alpha=1-self.momentum)

        return gradients


@register_sampler('cd_with_decay')
class CDWithDecay(ContrastiveDivergence):
    """CD with decaying number of steps during training.

    Starts with more Gibbs steps early in training for better gradients,
    then reduces steps for faster training as the model improves.
    """

    def __init__(
        self,
        initial_k: int = 25,
        final_k: int = 1,
        decay_epochs: int = 10,
        persistent: bool = False
    ):
        """Initialize CD with decay.

        Args:
            initial_k: Initial number of Gibbs steps
            final_k: Final number of Gibbs steps
            decay_epochs: Number of epochs for decay
            persistent: Use persistent chains
        """
        super().__init__(k=initial_k, persistent=persistent)
        self.initial_k = initial_k
        self.final_k = final_k
        self.decay_epochs = decay_epochs
        self.current_epoch = 0

    def update_k(self, epoch: int) -> None:
        """Update k based on current epoch.

        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        if epoch >= self.decay_epochs:
            self.k = self.final_k
        else:
            # Linear decay
            decay_factor = epoch / self.decay_epochs
            self.k = int(self.initial_k + decay_factor * (self.final_k - self.initial_k))

        self.sampler.k = self.k
        self.log_info(f"Updated k to {self.k} at epoch {epoch}")
