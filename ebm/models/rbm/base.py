"""Base classes and utilities for Restricted Boltzmann Machines.

This module provides the abstract base class for RBM implementations and
common utilities shared across different RBM variants.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ebm.core.config import RBMConfig
from ebm.models.base import AISInterpolator, LatentVariableModel
from ebm.utils.initialization import Initializer
from ebm.utils.tensor import shape_for_broadcast


class RBMBase(LatentVariableModel):
    """Abstract base class for Restricted Boltzmann Machines.

    This class provides the common interface and functionality for all RBM
    variants, including parameter management, energy computation, and
    sampling methods.
    """

    def __init__(self, config: RBMConfig):
        """Initialize RBM base.

        Args:
            config: RBM configuration
        """
        self.num_visible = config.visible_units
        self.num_hidden = config.hidden_units
        self.use_bias = config.use_bias

        # Initialize base class
        super().__init__(config)

    def _build_model(self) -> None:
        """Build RBM parameters."""
        # Weight matrix
        self.W = nn.Parameter(
            torch.empty(self.num_hidden, self.num_visible, dtype=self.dtype)
        )

        # Bias terms (optional)
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

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize model parameters."""
        config = self.config

        # Initialize weights
        weight_init = Initializer(config.weight_init)
        weight_init(self.W)

        # Initialize biases
        if self.use_bias:
            bias_init = Initializer(config.bias_init)
            bias_init(self.vbias)
            bias_init(self.hbias)

    @classmethod
    def get_config_class(cls) -> type[RBMConfig]:
        """Get configuration class for this model."""
        return RBMConfig

    def energy(
        self,
        x: Tensor,
        *,
        beta: Tensor | None = None,
        return_parts: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Compute energy of configurations.

        For RBMs, this expects concatenated visible and hidden states.

        Args:
            x: Concatenated states of shape (batch_size, num_visible + num_hidden)
            beta: Optional inverse temperature
            return_parts: If True, return dict with energy components

        Returns
        -------
            Energy values or dict of components
        """
        # Split into visible and hidden
        v, h = self._split_visible_hidden(x)
        return self.joint_energy(v, h, beta=beta, return_parts=return_parts)

    def joint_energy(
        self,
        visible: Tensor,
        hidden: Tensor,
        *,
        beta: Tensor | None = None,
        return_parts: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Compute joint energy of visible and hidden configurations.

        Energy function: E(v,h) = -v^T W h - a^T v - b^T h

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

        # Compute energy components
        # Interaction term: h^T W v
        # Dot product between hidden units and their pre-activations. Using the
        # same label for both tensors ensures a proper element-wise product
        # followed by summation rather than a product of sums.
        interaction = torch.einsum(
            "...h,...h->...", hidden, F.linear(visible, self.W)
        )

        # Bias terms
        v_bias_term = torch.einsum("...v,v->...", visible, self.vbias)
        h_bias_term = torch.einsum("...h,h->...", hidden, self.hbias)

        if return_parts:
            parts = {
                "interaction": -interaction,
                "visible_bias": -v_bias_term,
                "hidden_bias": -h_bias_term,
            }
            if beta is not None:
                beta_t = torch.as_tensor(
                    beta, device=self.device, dtype=self.dtype
                )
                beta_view = shape_for_broadcast(
                    beta_t, visible.shape[:-1], dim=0
                )
                parts = {k: beta_view * v for k, v in parts.items()}
            parts["total"] = sum(parts.values())
            return parts

        # Total energy
        energy = -(interaction + v_bias_term + h_bias_term)

        if beta is not None:
            beta_t = torch.as_tensor(beta, device=self.device, dtype=self.dtype)
            beta_view = shape_for_broadcast(beta_t, energy.shape, dim=0)
            energy = beta_view * energy

        return energy

    def free_energy(self, v: Tensor, *, beta: Tensor | None = None) -> Tensor:
        """Compute free energy of visible configurations.

        Free energy: F(v) = -a^T v - sum_j log(1 + exp(W_j v + b_j))

        Args:
            v: Visible unit values
            beta: Optional inverse temperature

        Returns
        -------
            Free energy values
        """
        v = self.prepare_input(v)

        # Pre-activation of hidden units
        pre_h = F.linear(v, self.W, self.hbias)

        # Apply temperature scaling if needed
        if beta is not None:
            beta_t = torch.as_tensor(beta, device=self.device, dtype=self.dtype)
            pre_beta = shape_for_broadcast(beta_t, pre_h.shape)
            pre_h = pre_beta * pre_h
            bias_beta = shape_for_broadcast(beta_t, v.shape[:-1], dim=0)
            v_bias_term = bias_beta * torch.einsum("...v,v->...", v, self.vbias)
        else:
            v_bias_term = torch.einsum("...v,v->...", v, self.vbias)

        # Free energy computation
        hidden_term = F.softplus(pre_h).sum(dim=-1)
        return -v_bias_term - hidden_term

    @abstractmethod
    def hidden_activation(self, pre_activation: Tensor) -> Tensor:
        """Apply activation function to hidden pre-activations.

        Args:
            pre_activation: Pre-activation values

        Returns
        -------
            Activation probabilities
        """

    @abstractmethod
    def visible_activation(self, pre_activation: Tensor) -> Tensor:
        """Apply activation function to visible pre-activations.

        Args:
            pre_activation: Pre-activation values

        Returns
        -------
            Activation probabilities
        """

    def sample_hidden(
        self,
        visible: Tensor,
        *,
        beta: Tensor | None = None,
        return_prob: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample hidden units given visible units.

        Args:
            visible: Visible unit values
            beta: Optional inverse temperature
            return_prob: If True, also return probabilities

        Returns
        -------
            Sampled hidden states, optionally with probabilities
        """
        visible = self.prepare_input(visible)

        # Compute pre-activation
        pre_h = F.linear(visible, self.W, self.hbias)

        # Apply temperature scaling
        if beta is not None:
            beta = torch.as_tensor(beta, device=self.device, dtype=self.dtype)
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
        """Sample visible units given hidden units.

        Args:
            hidden: Hidden unit values
            beta: Optional inverse temperature
            return_prob: If True, also return probabilities

        Returns
        -------
            Sampled visible states, optionally with probabilities
        """
        hidden = self.prepare_input(hidden)

        # Compute pre-activation
        pre_v = F.linear(hidden, self.W.t(), self.vbias)

        # Apply temperature scaling
        if beta is not None:
            beta = torch.as_tensor(beta, device=self.device, dtype=self.dtype)
            beta = shape_for_broadcast(beta, pre_v.shape[:-1])
            pre_v = beta * pre_v

        # Get probabilities
        prob_v = self.visible_activation(pre_v)

        # Sample
        v_sample = self._sample_from_prob(prob_v)

        if return_prob:
            return v_sample, prob_v
        return v_sample

    @abstractmethod
    def _sample_from_prob(self, prob: Tensor) -> Tensor:
        """Sample from probability distribution.

        Args:
            prob: Probability values

        Returns
        -------
            Sampled values
        """

    def _split_visible_hidden(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Split concatenated state into visible and hidden parts.

        Args:
            x: Concatenated state

        Returns
        -------
            Tuple of (visible, hidden)
        """
        v = x[..., : self.num_visible]
        h = x[..., self.num_visible :]
        return v, h

    def init_from_data(self, data_loader: torch.utils.data.DataLoader) -> None:
        """Initialize biases from data statistics.

        Args:
            data_loader: DataLoader providing training data
        """
        if not self.use_bias:
            return

        # Compute data statistics
        sum_v = torch.zeros(
            self.num_visible, device=self.device, dtype=self.dtype
        )
        count = 0

        with torch.no_grad():
            for data_batch in data_loader:
                batch_tensor = (
                    data_batch[0]
                    if isinstance(data_batch, list | tuple)
                    else data_batch
                )
                batch_tensor = self.to_device(batch_tensor)
                sum_v += batch_tensor.sum(dim=0)
                count += batch_tensor.shape[0]

        # Set visible bias to match data statistics
        mean_v = sum_v / count
        mean_v = mean_v.clamp(0.01, 0.99)  # Avoid extreme values
        self.vbias.data = torch.log(mean_v / (1 - mean_v))

        self.log_info("Initialized biases from data statistics")

    def effective_energy(self, v: Tensor, h: Tensor) -> Tensor:
        """Compute effective energy for gradient computation.

        This is used internally for computing gradients and may include
        additional terms like regularization.

        Args:
            v: Visible states
            h: Hidden states

        Returns
        -------
            Effective energy values
        """
        base_energy = self.joint_energy(v, h)

        # Add regularization if configured
        if hasattr(self.config, "l2_weight") and self.config.l2_weight > 0:
            base_energy = (
                base_energy + 0.5 * self.config.l2_weight * (self.W**2).sum()
            )

        if hasattr(self.config, "l1_weight") and self.config.l1_weight > 0:
            base_energy = (
                base_energy + self.config.l1_weight * self.W.abs().sum()
            )

        return base_energy

    def ais_adapter(self) -> RBMAISAdapter:
        """Create AIS adapter for this RBM."""
        return RBMAISAdapter(self)


class RBMAISAdapter(AISInterpolator):
    """AIS adapter for RBM models."""

    def __init__(self, rbm: RBMBase):
        """Initialize adapter.

        Args:
            rbm: RBM model to adapt
        """
        super().__init__(rbm)
        self.rbm = rbm

        # Store base distribution parameters
        with torch.no_grad():
            # Base visible bias from data statistics
            self.base_vbias = self.rbm.vbias.clone()
            # Base hidden bias is zero (uniform distribution)
            self.base_hbias = torch.zeros_like(self.rbm.hbias)

    def base_log_partition(self) -> float:
        """Compute log partition function of base distribution.

        For RBMs, the base distribution is typically:
        - Visible units: Independent Bernoulli with data-driven probabilities
        - Hidden units: Independent Bernoulli with p=0.5
        """
        # Visible partition function
        log_z_v = F.softplus(self.base_vbias).sum()

        # Hidden partition function (2^num_hidden)
        log_z_h = self.rbm.num_hidden * torch.log(torch.tensor(2.0))

        return float((log_z_v + log_z_h).item())

    def base_energy(self, x: Tensor) -> Tensor:
        """Compute energy under base distribution.

        Args:
            x: Concatenated visible and hidden states

        Returns
        -------
            Base energy values
        """
        v, h = self.rbm._split_visible_hidden(x)

        # Base energy has no interaction term
        v_term = -torch.einsum("...v,v->...", v, self.base_vbias)
        h_term = -torch.einsum("...h,h->...", h, self.base_hbias)

        return v_term + h_term

    def interpolated_energy(
        self, x: Tensor, beta: float | None = None
    ) -> Tensor:
        """Compute interpolated energy for AIS.

        Args:
            x: Concatenated states
            beta: AIS interpolation parameter

        Returns
        -------
            Interpolated energy values
        """
        if beta is None:
            beta = self.ais_beta

        v, h = self.rbm._split_visible_hidden(x)

        # Interpolate biases
        v_bias = (1 - beta) * self.base_vbias + beta * self.rbm.vbias
        h_bias = (1 - beta) * self.base_hbias + beta * self.rbm.hbias

        # Compute energy with interpolated parameters
        interaction = torch.einsum("...h,...h->...", h, F.linear(v, self.rbm.W))
        v_bias_term = torch.einsum("...v,v->...", v, v_bias)
        h_bias_term = torch.einsum("...h,h->...", h, h_bias)

        # Only the interaction term is scaled by beta
        return -(beta * interaction + v_bias_term + h_bias_term)
