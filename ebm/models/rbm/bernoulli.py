"""Bernoulli Restricted Boltzmann Machine implementations.

This module provides implementations of Bernoulli-Bernoulli RBMs (binary units)
and their variants, including centered RBMs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ebm.core.config import RBMConfig
from ebm.core.registry import register_model
from ebm.utils.tensor import shape_for_broadcast

from .base import RBMBase


@register_model("bernoulli_rbm", aliases=["brbm", "rbm"])
class BernoulliRBM(RBMBase):
    """Bernoulli-Bernoulli Restricted Boltzmann Machine.

    This is the standard RBM with binary visible and hidden units,
    using sigmoid activation functions for both layers.
    """

    def hidden_activation(self, pre_activation: Tensor) -> Tensor:
        """Apply sigmoid activation to hidden pre-activations."""
        return torch.sigmoid(pre_activation)

    def visible_activation(self, pre_activation: Tensor) -> Tensor:
        """Apply sigmoid activation to visible pre-activations."""
        return torch.sigmoid(pre_activation)

    def _sample_from_prob(self, prob: Tensor) -> Tensor:
        """Sample from Bernoulli distribution."""
        return torch.bernoulli(prob)

    def log_probability_ratio(
        self, v1: Tensor, v2: Tensor, *, beta: Tensor | None = None
    ) -> Tensor:
        """Compute log probability ratio log(p(v1)/p(v2)).

        This is more numerically stable than computing individual log probabilities.

        Args:
            v1: First set of visible configurations
            v2: Second set of visible configurations
            beta: Optional inverse temperature

        Returns
        -------
            Log probability ratios
        """
        v1 = self.prepare_input(v1)
        v2 = self.prepare_input(v2)

        # Difference in free energies gives log probability ratio
        f1 = self.free_energy(v1, beta=beta)
        f2 = self.free_energy(v2, beta=beta)

        return f2 - f1  # log(p(v1)/p(v2)) = F(v2) - F(v1)

    def score_function(
        self, v: Tensor, *, beta: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Compute score function (gradient of log probability).

        The score function is useful for score matching and other
        gradient-based methods.

        Args:
            v: Visible configurations
            beta: Optional inverse temperature

        Returns
        -------
            Dictionary with score w.r.t. each parameter
        """
        input_device = v.device
        v = self.prepare_input(v)
        v.requires_grad_(True)

        # Compute free energy with gradients
        f = self.free_energy(v, beta=beta)

        # Get gradients w.r.t. visible units
        score_v = torch.autograd.grad(
            f.sum(), v, create_graph=True, retain_graph=True
        )[0]

        # Get gradients w.r.t. parameters
        params = {"W": self.W, "vbias": self.vbias, "hbias": self.hbias}
        scores = {}

        for name, param in params.items():
            if param.requires_grad:
                grad = torch.autograd.grad(f.sum(), param, retain_graph=True)[0]
                scores[name] = -grad  # Negative because we want grad of log p

        scores["visible"] = -score_v.to(input_device)

        return scores


@register_model("centered_rbm", aliases=["crbm"])
class CenteredBernoulliRBM(BernoulliRBM):
    """Centered Bernoulli RBM with offset parameters.

    This variant uses centering to improve training stability by maintaining
    zero-mean activations. It introduces offset parameters for both visible
    and hidden units.
    """

    def __init__(self, config: RBMConfig):
        """Initialize centered RBM.

        Args:
            config: RBM configuration
        """
        super().__init__(config)
        self.centered = True

    def _build_model(self) -> None:
        """Build model parameters including offset terms."""
        # Weight and bias parameters
        self.W = nn.Parameter(
            torch.empty(self.num_hidden, self.num_visible, dtype=self.dtype)
        )

        if self.use_bias:
            self.vbias = nn.Parameter(
                torch.empty(self.num_visible, dtype=self.dtype)
            )
            self.hbias = nn.Parameter(
                torch.empty(self.num_hidden, dtype=self.dtype)
            )
        else:
            self.register_buffer(
                "vbias", torch.zeros(self.num_visible, dtype=self.dtype)
            )
            self.register_buffer(
                "hbias", torch.zeros(self.num_hidden, dtype=self.dtype)
            )

        # Offset parameters
        self.v_offset = nn.Parameter(
            torch.zeros(self.num_visible, dtype=self.dtype)
        )
        self.h_offset = nn.Parameter(
            torch.zeros(self.num_hidden, dtype=self.dtype)
        )

        # Initialize all parameters
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize parameters including offsets."""
        super()._initialize_parameters()

        # Initialize offsets to 0.5 (centering for binary units)
        nn.init.constant_(self.v_offset, 0.5)
        nn.init.constant_(self.h_offset, 0.5)

    def sample_hidden(
        self,
        visible: Tensor,
        *,
        beta: Tensor | None = None,
        return_prob: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample hidden units with centering.

        Args:
            visible: Visible unit values
            beta: Optional inverse temperature
            return_prob: If True, also return probabilities

        Returns
        -------
            Sampled hidden states, optionally with probabilities
        """
        visible = self.prepare_input(visible)

        # Center visible units
        v_centered = visible - self.v_offset

        # Compute pre-activation
        pre_h = F.linear(v_centered, self.W, self.hbias)

        # Apply temperature scaling
        if beta is not None:
            beta = shape_for_broadcast(beta, pre_h.shape[:-1])
            pre_h = beta * pre_h

        # Get probabilities
        prob_h = self.hidden_activation(pre_h)

        # Sample
        h_sample = self._sample_from_prob(prob_h)

        if return_prob:
            return h_sample, prob_h
        return h_sample

    def sample_visible(
        self,
        hidden: Tensor,
        *,
        beta: Tensor | None = None,
        return_prob: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample visible units with centering.

        Args:
            hidden: Hidden unit values
            beta: Optional inverse temperature
            return_prob: If True, also return probabilities

        Returns
        -------
            Sampled visible states, optionally with probabilities
        """
        hidden = self.prepare_input(hidden)

        # Center hidden units
        h_centered = hidden - self.h_offset

        # Compute pre-activation
        pre_v = F.linear(h_centered, self.W.t(), self.vbias)

        # Apply temperature scaling
        if beta is not None:
            beta = shape_for_broadcast(beta, pre_v.shape[:-1])
            pre_v = beta * pre_v

        # Get probabilities
        prob_v = self.visible_activation(pre_v)

        # Sample
        v_sample = self._sample_from_prob(prob_v)

        if return_prob:
            return v_sample, prob_v
        return v_sample

    def joint_energy(
        self,
        visible: Tensor,
        hidden: Tensor,
        *,
        beta: Tensor | None = None,
        return_parts: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Compute joint energy with centering.

        Energy: E(v,h) = -(h-λ)^T W (v-μ) - a^T v - b^T h

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

        # Center units
        v_centered = visible - self.v_offset
        h_centered = hidden - self.h_offset

        # Compute energy components
        interaction = torch.einsum(
            "...h,...v->...", h_centered, F.linear(v_centered, self.W)
        )
        v_bias_term = torch.einsum("...v,v->...", visible, self.vbias)
        h_bias_term = torch.einsum("...h,h->...", hidden, self.hbias)

        if return_parts:
            parts = {
                "interaction": -interaction,
                "visible_bias": -v_bias_term,
                "hidden_bias": -h_bias_term,
            }
            if beta is not None:
                beta = shape_for_broadcast(beta, visible.shape[:-1])
                parts = {k: beta * v for k, v in parts.items()}
            parts["total"] = sum(parts.values())
            return parts

        # Total energy
        energy = -(interaction + v_bias_term + h_bias_term)

        if beta is not None:
            beta = shape_for_broadcast(beta, energy.shape)
            energy = beta * energy

        return energy

    def free_energy(self, v: Tensor, *, beta: Tensor | None = None) -> Tensor:
        """Compute free energy with centering.

        Args:
            v: Visible unit values
            beta: Optional inverse temperature

        Returns
        -------
            Free energy values
        """
        v = self.prepare_input(v)
        v_centered = v - self.v_offset

        # Pre-activation of hidden units
        pre_h = F.linear(v_centered, self.W, self.hbias)

        # Apply temperature scaling if needed
        if beta is not None:
            beta = shape_for_broadcast(beta, pre_h.shape[:-1])
            pre_h = beta * pre_h
            v_bias_term = beta * torch.einsum("...v,v->...", v, self.vbias)
        else:
            v_bias_term = torch.einsum("...v,v->...", v, self.vbias)

        # Free energy computation
        hidden_term = F.softplus(pre_h).sum(dim=-1)
        return -v_bias_term - hidden_term

    def update_offsets(
        self, v_mean: Tensor, h_mean: Tensor, momentum: float = 0.9
    ) -> None:
        """Update offset parameters using exponential moving average.

        This should be called periodically during training to maintain
        centered activations.

        Args:
            v_mean: Mean visible activation
            h_mean: Mean hidden activation
            momentum: Momentum for moving average
        """
        with torch.no_grad():
            self.v_offset.mul_(momentum).add_(v_mean, alpha=1 - momentum)
            self.h_offset.mul_(momentum).add_(h_mean, alpha=1 - momentum)

    def init_from_data(self, data_loader: torch.utils.data.DataLoader) -> None:
        """Initialize parameters from data statistics.

        Args:
            data_loader: DataLoader providing training data
        """
        super().init_from_data(data_loader)

        # Also initialize v_offset to data mean
        with torch.no_grad():
            sum_v = torch.zeros(
                self.num_visible, device=self.device, dtype=self.dtype
            )
            count = 0

            for data_batch in data_loader:
                batch_tensor = (
                    data_batch[0]
                    if isinstance(data_batch, list | tuple)
                    else data_batch
                )
                batch_tensor = self.to_device(batch_tensor)
                sum_v += batch_tensor.sum(dim=0)
                count += batch_tensor.shape[0]

            mean_v = sum_v / count
            self.v_offset.copy_(mean_v)

            # Adjust visible bias to account for offset
            self.vbias.add_(F.linear(self.h_offset, self.W))


@register_model("sparse_rbm", aliases=["srbm"])
class SparseBernoulliRBM(BernoulliRBM):
    """Sparse Bernoulli RBM with sparsity constraints.

    This variant encourages sparse hidden representations through
    various sparsity penalties and constraints.
    """

    def __init__(self, config: RBMConfig):
        """Initialize sparse RBM.

        Args:
            config: RBM configuration with sparsity settings
        """
        super().__init__(config)

        # Sparsity parameters
        self.sparsity_target = getattr(config, "sparsity_target", 0.1)
        self.sparsity_weight = getattr(config, "sparsity_weight", 0.01)
        self.sparsity_damping = getattr(config, "sparsity_damping", 0.9)

        # Running average of hidden activations
        self.register_buffer(
            "hidden_mean",
            torch.ones(self.num_hidden, dtype=self.dtype)
            * self.sparsity_target,
        )

    def sparsity_penalty(self, h_prob: Tensor) -> Tensor:
        """Compute sparsity penalty using KL divergence.

        Args:
            h_prob: Hidden unit probabilities

        Returns
        -------
            Sparsity penalty value
        """
        # Update running mean
        batch_mean = h_prob.mean(dim=0)
        with torch.no_grad():
            self.hidden_mean.mul_(self.sparsity_damping).add_(
                batch_mean, alpha=1 - self.sparsity_damping
            )

        # KL divergence between target and actual sparsity
        eps = 1e-8
        kl = self.sparsity_target * torch.log(
            self.sparsity_target / (self.hidden_mean + eps) + eps
        ) + (1 - self.sparsity_target) * torch.log(
            (1 - self.sparsity_target) / (1 - self.hidden_mean + eps) + eps
        )

        return self.sparsity_weight * kl.sum()

    def sparse_sample_hidden(
        self,
        visible: Tensor,
        k: int = 10,
        *,
        beta: Tensor | None = None,
        return_prob: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample hidden units with top-k sparsity constraint.

        Args:
            visible: Visible unit values
            k: Number of hidden units to activate
            beta: Optional inverse temperature
            return_prob: If True, also return probabilities

        Returns
        -------
            Sparse hidden states, optionally with probabilities
        """
        # Get probabilities
        if return_prob:
            h_sample, h_prob = self.sample_hidden(
                visible, beta=beta, return_prob=True
            )
        else:
            h_prob = self.sample_hidden(visible, beta=beta, return_prob=True)[1]

        # Apply top-k sparsity
        topk_values, topk_indices = torch.topk(h_prob, k, dim=-1)

        # Create sparse samples
        h_sparse = torch.zeros_like(h_prob)
        h_sparse.scatter_(-1, topk_indices, 1.0)

        if return_prob:
            return h_sparse, h_prob
        return h_sparse
