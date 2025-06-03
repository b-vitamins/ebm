"""Gaussian-Bernoulli Restricted Boltzmann Machine implementation.

This module provides RBM implementations with Gaussian visible units and
Bernoulli hidden units, suitable for continuous-valued data.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ebm.core.config import GaussianRBMConfig
from ebm.core.registry import register_model
from ebm.utils.tensor import shape_for_broadcast

from .base import RBMBase


@register_model("gaussian_rbm", aliases=["grbm", "gbrbm"])
class GaussianBernoulliRBM(RBMBase):
    """Gaussian-Bernoulli Restricted Boltzmann Machine.

    This RBM variant has Gaussian (continuous) visible units and
    Bernoulli (binary) hidden units. It's suitable for modeling
    continuous data like images with real-valued pixels.
    """

    def __init__(self, config: GaussianRBMConfig):
        """Initialize Gaussian-Bernoulli RBM.

        Args:
            config: Configuration for Gaussian RBM
        """
        self.learn_sigma = config.learn_sigma
        super().__init__(config)

    @classmethod
    def get_config_class(cls) -> type[GaussianRBMConfig]:
        """Get configuration class for this model."""
        return GaussianRBMConfig

    def _build_model(self) -> None:
        """Build model parameters including variance terms."""
        super()._build_model()

        # Variance parameters for visible units
        if self.learn_sigma:
            # Learnable variance per visible unit
            self.log_sigma = nn.Parameter(
                torch.zeros(self.num_visible, dtype=self.dtype)
            )
        else:
            # Fixed variance
            sigma = getattr(self.config, "sigma", 1.0)
            self.register_buffer(
                "log_sigma",
                torch.full(
                    (self.num_visible,), math.log(sigma), dtype=self.dtype
                ),
            )

    @property
    def sigma(self) -> Tensor:
        """Get standard deviation of visible units."""
        return torch.exp(self.log_sigma)

    @property
    def sigma_sq(self) -> Tensor:
        """Get variance of visible units."""
        return torch.exp(2 * self.log_sigma)

    def hidden_activation(self, pre_activation: Tensor) -> Tensor:
        """Apply sigmoid activation to hidden pre-activations."""
        return torch.sigmoid(pre_activation)

    def visible_activation(self, pre_activation: Tensor) -> Tensor:
        """Identity activation for Gaussian visible units."""
        return pre_activation

    def _sample_from_prob(self, prob: Tensor) -> Tensor:
        """Sample from distribution based on unit type.

        This is called for hidden units (Bernoulli).
        Visible unit sampling is handled separately.
        """
        return torch.bernoulli(prob)

    def sample_visible(
        self,
        hidden: Tensor,
        *,
        beta: Tensor | None = None,
        return_prob: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample visible units from Gaussian distribution.

        Args:
            hidden: Hidden unit values
            beta: Optional inverse temperature
            return_prob: If True, return mean and samples

        Returns
        -------
            Sampled visible states (and mean if return_prob=True)
        """
        hidden = self.prepare_input(hidden)

        # Compute mean of Gaussian
        mean_v = F.linear(hidden, self.W.t(), self.vbias)

        # Apply temperature scaling to mean
        if beta is not None:
            beta = shape_for_broadcast(beta, mean_v.shape[:-1])
            mean_v = mean_v * beta
            # Temperature also affects variance
            sigma = self.sigma / torch.sqrt(beta)
        else:
            sigma = self.sigma

        # Sample from Gaussian
        noise = torch.randn_like(mean_v)
        v_sample = mean_v + sigma * noise

        if return_prob:
            return v_sample, mean_v
        return v_sample

    def joint_energy(
        self,
        visible: Tensor,
        hidden: Tensor,
        *,
        beta: Tensor | None = None,
        return_parts: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Compute joint energy with Gaussian visible units.

        Energy: E(v,h) = sum_i (v_i - a_i)^2 / (2*sigma_i^2) - sum_j b_j h_j
                         - sum_ij (v_i / sigma_i^2) W_ij h_j

        Args:
            visible: Visible unit values
            hidden: Hidden unit values
            beta: Optional inverse temperature
            return_parts: If True, return dict with energy components

        Returns
        -------
            Energy values or dict of components
        """
        visible = self.prepare_input(visible)
        hidden = self.prepare_input(hidden)

        # Normalize visible units by variance
        v_normalized = visible / self.sigma_sq

        # Energy components
        # Quadratic term for visible units
        v_quad = 0.5 * ((visible - self.vbias) ** 2 / self.sigma_sq).sum(dim=-1)

        # Linear term for hidden units
        h_linear = (hidden * self.hbias).sum(dim=-1)

        # Interaction term
        interaction = torch.einsum(
            "...v,...h->...", F.linear(v_normalized, self.W.t()), hidden
        )

        if return_parts:
            parts = {
                "visible_quadratic": v_quad,
                "hidden_linear": -h_linear,
                "interaction": -interaction,
            }
            if beta is not None:
                beta = shape_for_broadcast(beta, visible.shape[:-1])
                parts = {k: beta * v for k, v in parts.items()}
            parts["total"] = sum(parts.values())
            return parts

        # Total energy
        energy = v_quad - h_linear - interaction

        if beta is not None:
            beta = shape_for_broadcast(beta, energy.shape)
            energy = beta * energy

        return energy

    def free_energy(self, v: Tensor, *, beta: Tensor | None = None) -> Tensor:
        """Compute free energy for Gaussian visible units.

        Free energy: F(v) = sum_i (v_i - a_i)^2 / (2*sigma_i^2)
                           - sum_j log(1 + exp(c_j + sum_i W_ij v_i / sigma_i^2))

        Args:
            v: Visible unit values
            beta: Optional inverse temperature

        Returns
        -------
            Free energy values
        """
        v = self.prepare_input(v)

        # Quadratic term
        v_term = 0.5 * ((v - self.vbias) ** 2 / self.sigma_sq).sum(dim=-1)

        # Hidden term (log partition function of hidden units)
        v_normalized = v / self.sigma_sq
        pre_h = F.linear(v_normalized, self.W, self.hbias)

        # Apply temperature scaling
        if beta is not None:
            beta = shape_for_broadcast(beta, pre_h.shape[:-1])
            pre_h = beta * pre_h
            v_term = beta * v_term

        h_term = F.softplus(pre_h).sum(dim=-1)

        return v_term - h_term

    def score_matching_loss(self, v: Tensor, noise_std: float = 0.01) -> Tensor:
        """Compute denoising score matching loss.

        This provides an alternative training objective that doesn't
        require sampling from the model.

        Args:
            v: Clean visible data
            noise_std: Standard deviation of noise to add

        Returns
        -------
            Score matching loss value
        """
        v = self.prepare_input(v)

        # Add noise
        noise = torch.randn_like(v) * noise_std
        v_noisy = v + noise

        # Compute score of noisy data
        v_noisy.requires_grad_(True)
        energy = self.free_energy(v_noisy)

        # Get score (gradient of log probability)
        score = torch.autograd.grad(energy.sum(), v_noisy, create_graph=True)[0]

        # Denoising score matching loss
        target = -noise / (noise_std**2)
        return 0.5 * ((score - target) ** 2).sum(dim=-1).mean()

    def sample_fantasy_particles(
        self,
        num_samples: int,
        num_steps: int = 1000,
        *,
        init_from_data: Tensor | None = None,
        beta: Tensor | None = None,
        return_chain: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Generate samples using Gibbs sampling.

        Args:
            num_samples: Number of samples to generate
            num_steps: Number of Gibbs steps
            init_from_data: Optional data to initialize from
            beta: Optional inverse temperature
            return_chain: If True, return all intermediate states

        Returns
        -------
            Final samples (and chain if requested)
        """
        # Initialize visible units
        if init_from_data is not None:
            v = self.prepare_input(init_from_data[:num_samples])
        else:
            # Initialize from prior (Gaussian with learned mean/variance)
            v = self.vbias + self.sigma * torch.randn(
                num_samples,
                self.num_visible,
                device=self.device,
                dtype=self.dtype,
            )

        chain = [v.clone()] if return_chain else None

        # Run Gibbs sampling
        for _ in range(num_steps):
            # Sample hidden given visible
            h = self.sample_hidden(v, beta=beta)

            # Sample visible given hidden
            v = self.sample_visible(h, beta=beta)

            if return_chain:
                chain.append(v.clone())

        if return_chain:
            return v, chain
        return v


@register_model("gaussian_rbm_whitened", aliases=["grbm_w"])
class WhitenedGaussianRBM(GaussianBernoulliRBM):
    """Gaussian RBM with data whitening preprocessing.

    This variant automatically whitens the input data, which can improve
    optimization and allow using a fixed unit variance.
    """

    def __init__(self, config: GaussianRBMConfig):
        """Initialize whitened Gaussian RBM.

        Args:
            config: Configuration for Gaussian RBM
        """
        super().__init__(config)

        # Whitening parameters
        self.register_buffer("whitening_mean", None)
        self.register_buffer("whitening_std", None)
        self.fitted = False

    def fit_whitening(self, data_loader: torch.utils.data.DataLoader) -> None:
        """Fit whitening transformation to data.

        Args:
            data_loader: DataLoader providing training data
        """
        # Compute data statistics
        sum_x = torch.zeros(
            self.num_visible, device=self.device, dtype=self.dtype
        )
        sum_x_sq = torch.zeros(
            self.num_visible, device=self.device, dtype=self.dtype
        )
        count = 0

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, list | tuple):
                    batch = batch[0]
                batch = self.to_device(batch)

                sum_x += batch.sum(dim=0)
                sum_x_sq += (batch**2).sum(dim=0)
                count += batch.shape[0]

        # Compute mean and std
        mean = sum_x / count
        var = (sum_x_sq / count) - (mean**2)
        std = torch.sqrt(var + 1e-8)

        # Store whitening parameters
        self.whitening_mean = mean
        self.whitening_std = std
        self.fitted = True

        # Set visible bias to zero (data is centered)
        self.vbias.data.zero_()

        # Set variance to 1 (data is normalized)
        if not self.learn_sigma:
            self.log_sigma.data.zero_()

        self.log_info("Fitted whitening transformation")

    def whiten(self, v: Tensor) -> Tensor:
        """Apply whitening transformation.

        Args:
            v: Input data

        Returns
        -------
            Whitened data
        """
        if not self.fitted:
            return v
        return (v - self.whitening_mean) / self.whitening_std

    def unwhiten(self, v: Tensor) -> Tensor:
        """Inverse whitening transformation.

        Args:
            v: Whitened data

        Returns
        -------
            Original scale data
        """
        if not self.fitted:
            return v
        return v * self.whitening_std + self.whitening_mean

    def prepare_input(self, x: Tensor) -> Tensor:
        """Prepare input with whitening."""
        x = super().prepare_input(x)
        return self.whiten(x)

    def sample_fantasy_particles(
        self,
        num_samples: int,
        num_steps: int = 1000,
        *,
        init_from_data: Tensor | None = None,
        beta: Tensor | None = None,
        return_chain: bool = False,
        unwhiten_output: bool = True,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Generate samples with optional unwhitening.

        Args:
            num_samples: Number of samples
            num_steps: Number of Gibbs steps
            init_from_data: Optional initialization data
            beta: Optional inverse temperature
            return_chain: If True, return chain
            unwhiten_output: If True, unwhiten the output

        Returns
        -------
            Samples in original scale (if unwhiten_output=True)
        """
        result = super().sample_fantasy_particles(
            num_samples,
            num_steps,
            init_from_data=init_from_data,
            beta=beta,
            return_chain=return_chain,
        )

        if not unwhiten_output:
            return result

        if return_chain:
            samples, chain = result
            samples = self.unwhiten(samples)
            chain = [self.unwhiten(v) for v in chain]
            return samples, chain
        return self.unwhiten(result)
