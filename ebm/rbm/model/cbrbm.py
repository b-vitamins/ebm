"""Centered Bernoulli-Bernoulli Restricted Boltzmann Machine (RBM).

This module implements a centered variant of the classic Bernoulli RBM following
Montavon & Müller (2012). The model maintains offset parameters for visible and
hidden units that center the activations by subtracting empirical means before
linear projections.

Centering does not change the distribution the RBM can represent but significantly
improves the conditioning of the optimization problem, leading to faster convergence
and better final models. This implementation builds upon the standard BernoulliRBM
and modifies only the components that depend on the offset parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812

from ebm.rbm.utils import shape_beta

from .base import OffsetInit, init_offset_tensor
from .brbm import BernoulliRBM, BernoulliRBMConfig


@dataclass(slots=True, frozen=True)
class CenteredBernoulliRBMConfig(BernoulliRBMConfig):
    """Configuration for Centered Bernoulli RBM models.

    This dataclass extends the BernoulliRBMConfig for centered RBMs by adding
    parameters for offset initialization.

    Parameters
    ----------
    v_off_init : OffsetInit, optional
        Initialization strategy for visible unit offsets, which can be:
        - A tensor with shape (visible,)
        - A float for constant initialization
        - None to initialize every element to 0.5 (default)
    h_off_init : OffsetInit, optional
        Initialization strategy for hidden unit offsets, which can be:
        - A tensor with shape (hidden,)
        - A float for constant initialization
        - None to initialize every element to 0.5 (default)

    Examples
    --------
    >>> config = CenteredBernoulliRBMConfig(
    ...     visible=784,  # For MNIST
    ...     hidden=500,
    ...     v_off_init=0.3,  # Initialize visible offsets to 0.3
    ...     h_off_init=0.5,  # Initialize hidden offsets to 0.5
    ...     dtype=torch.float32
    ... )
    """

    v_off_init: OffsetInit = None
    h_off_init: OffsetInit = None


class CenteredBernoulliRBM(BernoulliRBM):
    """Centered Bernoulli-Bernoulli Restricted Boltzmann Machine.

    This class implements a Centered RBM with binary visible and hidden units.
    It extends the standard BernoulliRBM by adding learnable offset parameters
    that center the activations, significantly improving training dynamics.

    Parameters
    ----------
    cfg : CenteredBernoulliRBMConfig
        Configuration for the Centered Bernoulli RBM.
    init_now : bool, optional
        Whether to initialize parameters immediately, by default True.

    Attributes
    ----------
    cfg : CenteredBernoulliRBMConfig
        Configuration parameters for this RBM.
    W : torch.nn.Parameter
        Weight matrix with shape (hidden, visible).
    vb : torch.nn.Parameter
        Visible bias vector with shape (visible,).
    hb : torch.nn.Parameter
        Hidden bias vector with shape (hidden,).
    v_off : torch.Tensor
        Visible unit offsets with shape (visible,).
    h_off : torch.Tensor
        Hidden unit offsets with shape (hidden,).

    Examples
    --------
    >>> config = CenteredBernoulliRBMConfig(visible=784, hidden=500)
    >>> model = CenteredBernoulliRBM(config)
    >>> # Forward pass with batch of 64 samples
    >>> v = torch.bernoulli(torch.rand(64, 784))
    >>> h_probs = model.prob_h_given_v(v)
    >>> h_samples = model.sample_h_given_v(v)
    """

    # Typed buffers for static analysis
    v_off: torch.Tensor
    h_off: torch.Tensor

    def __init__(self, cfg: CenteredBernoulliRBMConfig, *, init_now: bool = True) -> None:
        # Call parent constructor but defer initialization
        super().__init__(cast(BernoulliRBMConfig, cfg), init_now=False)

        # Store config
        self.cfg = cfg

        # Register offset buffers
        self.register_buffer(
            "v_off",
            init_offset_tensor(cfg.v_off_init, cfg.visible, device=cfg.device, dtype=cfg.dtype),
        )
        self.register_buffer(
            "h_off",
            init_offset_tensor(cfg.h_off_init, cfg.hidden, device=cfg.device, dtype=cfg.dtype),
        )

        # Initialize parameters if requested
        if init_now:
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset all model parameters with default initialization.

        This method initializes weights and biases using the parent class method,
        then initializes the offset parameters according to the configuration.
        """
        # Initialize weights and biases using parent method
        super().reset_parameters()

        # Cast self.cfg to CenteredBernoulliRBMConfig to tell mypy it has the needed attributes
        cfg = cast(CenteredBernoulliRBMConfig, self.cfg)

        # Initialize offset parameters
        self.v_off.copy_(
            init_offset_tensor(cfg.v_off_init, cfg.visible, device=cfg.device, dtype=cfg.dtype)
        )
        self.h_off.copy_(
            init_offset_tensor(cfg.h_off_init, cfg.hidden, device=cfg.device, dtype=cfg.dtype)
        )

    def preact_h(self, v: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate centered pre-activation values for hidden units.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit values with shape (B, ..., V) or (B, K, ..., V).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape compatible
            with result tensor, by default None.

        Returns
        -------
        torch.Tensor
            Pre-activation values for hidden units with shape (B, ..., H)
            or (B, K, ..., H).

        Notes
        -----
        This method implements the centered pre-activation formula:
        preact_h = W · (v - v_off) + hb
        """
        # Center visible units by subtracting offsets
        v_centered = v - self.v_off

        # Use parent class method with centered visible units
        return super().preact_h(v_centered, beta=beta)

    def preact_v(self, h: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate centered pre-activation values for visible units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden unit values with shape (B, ..., H) or (B, K, ..., H).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape compatible
            with result tensor, by default None.

        Returns
        -------
        torch.Tensor
            Pre-activation values for visible units with shape (B, ..., V)
            or (B, K, ..., V).

        Notes
        -----
        This method implements the centered pre-activation formula:
        preact_v = W^T · (h - h_off) + vb
        """
        # Center hidden units by subtracting offsets
        h_centered = h - self.h_off

        # Use parent class method with centered hidden units
        return super().preact_v(h_centered, beta=beta)

    def energy(
        self,
        v: torch.Tensor,
        h: torch.Tensor,
        *,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculate the energy of a joint configuration with centering.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit values with shape (B, ..., V) or (B, K, ..., V).
        h : torch.Tensor
            Hidden unit values with shape (B, ..., H) or (B, K, ..., H).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape compatible
            with result tensor, by default None.

        Returns
        -------
        torch.Tensor
            Energy values with shape (B, ...) or (B, K, ...).

        Notes
        -----
        The energy function with centered units is:
        E(v,h) = -(v-v_off)^T · W · (h-h_off) - vb^T · v - hb^T · h
        """
        # Center both visible and hidden units
        v_centered = v - self.v_off
        h_centered = h - self.h_off

        # Use parent class method with centered units
        return super().energy(v_centered, h_centered, beta=beta)

    def free_energy(
        self,
        v: torch.Tensor,
        *,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Free energy of visible configurations for a *centred* Bernoulli–Bernoulli RBM.

        Parameters
        ----------
        v : torch.Tensor
            Visible states, shape (B, ..., V)   *or*   (B, K, ..., V).
        beta : torch.Tensor or None, optional
            In Parallel Tempering, energy is scaled by β.  Accepts a scalar or any
            prefix of the output shape.  If None (default), β = 1.

        Returns
        -------
        torch.Tensor
            Free-energy values, shape (B, ...) or (B, K, ...).

        Notes
        -----
        Let
            v_c   = v − v_off                  (centred visibles)
            s     = W @ v_c  +  hb             (hidden pre-activations)
        Then the centred free energy is

            F(v) = − (v_c · vb)                       ⎫  visible term
                   − Σ_j softplus(s_j)                ⎬  hidden term
                   + Σ_j s_j · h_off_j                ⎭  offset correction

        With a tempering factor β this becomes

            F_β(v) = β [ −(v_c·vb) + Σ_j s_j·h_off_j ]  −  Σ_j softplus(β s_j)

        which is exactly what the code below computes.
        """
        # 1.  centre the visibles
        v_c = v - self.v_off  # shape (..., V)

        # 2.  hidden pre-activations γ = W v_c + b
        s = F.linear(v_c, self.w, self.hb)  # shape (..., H)

        # 3.  base terms (β = 1)
        visible_term = -(v_c * self.vb).sum(dim=-1)  # −v_c·vb
        offset_term = (s * self.h_off).sum(dim=-1)  # Σ s·h_off
        hidden_term = -F.softplus(s).sum(dim=-1)  # −Σ log(1+e^s)

        if beta is None:
            return visible_term + offset_term + hidden_term

        # 4.  β ≠ 1  →  reshape/broadcast β to match the output shape
        β = shape_beta(beta, visible_term)

        # 4a.  scale the two linear terms
        visible_term = visible_term * β  # β (−v_c·vb)
        offset_term = (s * β * self.h_off).sum(dim=-1)  # β Σ s·h_off

        # 4b.  hidden softplus gets β inside its argument
        hidden_term = -F.softplus(s * β).sum(dim=-1)  # −Σ log(1+e^{βs})

        return visible_term + offset_term + hidden_term

    @torch.no_grad()
    def init_vb_from_means(self, means: torch.Tensor) -> None:
        """Initialize visible biases and offsets from data means.

        Parameters
        ----------
        means : torch.Tensor
            Mean values of visible units from the training data.

        Notes
        -----
        This method:
        1. Sets visible biases to match data marginals (as in standard RBM)
        2. Sets visible offsets to the actual data means

        Setting offsets equal to means ensures that centered visible units
        have zero mean, which improves optimization.
        """
        # Clamp means to avoid numerical issues
        clamped = means.clamp(1e-7, 1 - 1e-7)

        # Convert means to logits for biases
        logits = torch.logit(clamped)

        # Update biases (same as parent class)
        self.vb.copy_(logits)
        self.base_rate_vb.copy_(logits)

        # Set offsets to data means
        self.v_off.copy_(clamped)
