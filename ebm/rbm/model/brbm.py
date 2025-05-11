"""Implementation of Bernoulli-Bernoulli Restricted Boltzmann Machine (RBM).

This module provides the implementation of a Bernoulli-Bernoulli RBM,
which uses binary visible and hidden units with Bernoulli distributions.

The code follows the interface defined in base.py and uses PyTorch's
weight matrix convention of (out_dim, in_dim) for the weight matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812

from ebm.rbm.utils import shape_beta

from .base import BaseRBM, RBMConfig


@dataclass(slots=True, frozen=True)
class BernoulliRBMConfig(RBMConfig):
    """Configuration for Bernoulli-Bernoulli RBM models.

    This dataclass extends the base RBMConfig for Bernoulli RBMs
    with binary visible and hidden units. Both visible and hidden
    units follow Bernoulli distributions with sigmoid activation
    functions by default.

    Examples
    --------
    >>> config = BernoulliConfig(
    ...     visible=784,  # For MNIST
    ...     hidden=500,
    ...     dtype=torch.float32
    ... )
    """

    pass


class BernoulliRBM(BaseRBM):
    """Bernoulli-Bernoulli Restricted Boltzmann Machine.

    This class implements a Restricted Boltzmann Machine with binary
    visible and hidden units following Bernoulli distributions. This
    is the most common form of RBM, suitable for binary data or data
    that can be binarized (e.g., MNIST digits).

    Parameters
    ----------
    cfg : BernoulliConfig
        Configuration for the Bernoulli RBM.

    Attributes
    ----------
    cfg : BernoulliConfig
        Configuration parameters for this RBM.
    W : torch.nn.Parameter
        Weight matrix with shape (hidden, visible) following PyTorch convention.
    vb : torch.nn.Parameter
        Visible bias vector with shape (visible,).
    hb : torch.nn.Parameter
        Hidden bias vector with shape (hidden,).

    Examples
    --------
    >>> config = BernoulliConfig(visible=784, hidden=500)
    >>> model = BernoulliRBM(config)
    >>> # Forward pass with batch of 64 samples
    >>> v = torch.bernoulli(torch.rand(64, 784))
    >>> h_probs = model.prob_h_given_v(v)
    >>> h_samples = model.sample_h_given_v(v)
    """

    w: torch.nn.Parameter
    vb: torch.nn.Parameter
    hb: torch.nn.Parameter

    def __init__(self, cfg: BernoulliRBMConfig) -> None:
        """Initialize the Bernoulli RBM model.

        Parameters
        ----------
        cfg : BernoulliConfig
            Configuration for the Bernoulli RBM.
        """
        super().__init__()
        self.cfg = cfg

        self.w = torch.nn.Parameter(
            torch.empty(cfg.hidden, cfg.visible, dtype=cfg.dtype, device=cfg.device)
        )
        self.vb = torch.nn.Parameter(torch.empty(cfg.visible, dtype=cfg.dtype, device=cfg.device))
        self.hb = torch.nn.Parameter(torch.empty(cfg.hidden, dtype=cfg.dtype, device=cfg.device))

        self.register_buffer(
            "base_rate_vb", torch.zeros(cfg.visible, dtype=cfg.dtype, device=cfg.device)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset model parameters with default initialization.

        For weights: normal distribution with std=0.01
        For biases: initialized to zeros
        """
        torch.nn.init.normal_(self.w, std=0.01)
        torch.nn.init.zeros_(self.vb)
        torch.nn.init.zeros_(self.hb)

    def preact_h(self, v: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate pre-activation values for hidden units.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit values with shape (B, ..., V) or (B, K, ..., V).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape compatible
            with result tensor. If None, standard behavior without temperature
            scaling is used.

        Returns
        -------
        torch.Tensor
            Pre-activation values for hidden units with shape (B, ..., H)
            or (B, K, ..., H).
        """
        pre_h = F.linear(v, self.w, self.hb)
        β = shape_beta(beta, pre_h)
        return pre_h if β is None else pre_h * β

    def preact_v(self, h: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate pre-activation values for visible units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden unit values with shape (B, ..., H) or (B, K, ..., H).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape compatible
            with result tensor. If None, standard behavior without temperature
            scaling is used.

        Returns
        -------
        torch.Tensor
            Pre-activation values for visible units with shape (B, ..., V)
            or (B, K, ..., V).
        """
        pre_v = F.linear(h, self.w.t(), self.vb)
        β = shape_beta(beta, pre_v)
        return pre_v if β is None else pre_v * β

    def prob_h_given_v(self, v: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate conditional probabilities of hidden units given visible units.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit values with shape (B, ..., V) or (B, K, ..., V).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape compatible
            with v. If None, standard behavior without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Conditional probabilities of hidden units with shape matching
            the pre-activation tensor.
        """
        return self.cfg.h_act(self.preact_h(v, beta=beta))

    def prob_v_given_h(self, h: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate conditional probabilities of visible units given hidden units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden unit values with shape (B, ..., H) or (B, K, ..., H).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape compatible
            with h. If None, standard behavior without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Conditional probabilities of visible units with shape matching
            the pre-activation tensor.
        """
        return self.cfg.v_act(self.preact_v(h, beta=beta))

    def sample_h_given_v(
        self, v: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample hidden units given visible units.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit values with shape (B, ..., V) or (B, K, ..., V).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape compatible
            with v. If None, standard behavior without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Sampled binary hidden unit values with the same shape as the
            conditional probability tensor.
        """
        return torch.bernoulli(self.prob_h_given_v(v, beta=beta))

    def sample_v_given_h(
        self, h: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample visible units given hidden units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden unit values with shape (B, ..., H) or (B, K, ..., H).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape compatible
            with h. If None, standard behavior without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Sampled binary visible unit values with the same shape as the
            conditional probability tensor.
        """
        return torch.bernoulli(self.prob_v_given_h(h, beta=beta))

    def energy(
        self,
        v: torch.Tensor,
        h: torch.Tensor,
        *,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculate the energy of a joint configuration.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit values with shape (B, ..., V) or (B, K, ..., V).
        h : torch.Tensor
            Hidden unit values with shape (B, ..., H) or (B, K, ..., H).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape compatible
            with result tensor. If None, standard behavior without temperature
            scaling is used.

        Returns
        -------
        torch.Tensor
            Energy values with shape (B, ...) or (B, K, ...).
        """
        # Calculate energy terms using PyTorch convention
        interaction = (h * F.linear(v, self.w)).sum(-1)
        energy = -(v * self.vb).sum(-1) - (h * self.hb).sum(-1) - interaction

        # Apply temperature scaling if beta is provided
        β = shape_beta(beta, energy)
        return energy if β is None else energy * β

    def free_energy(self, v: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate the free energy of visible configurations.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit values with shape (B, ..., V) or (B, K, ..., V).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape compatible
            with result tensor. If None, standard behavior without temperature
            scaling is used.

        Returns
        -------
        torch.Tensor
            Free energy values with shape (B, ...) or (B, K, ...).
        """
        # Calculate visible bias term
        visible_term = -(v * self.vb).sum(-1)

        if beta is None:
            # Standard case without temperature scaling
            s = self.preact_h(v)
            hidden_term = -F.softplus(s).sum(-1)
            return visible_term + hidden_term
        else:
            # With temperature scaling
            β = shape_beta(beta, visible_term)
            scaled_visible_term = visible_term * β

            # Apply beta to pre-activation directly
            s = self.preact_h(v, beta=beta)
            hidden_term = -F.softplus(s).sum(-1)
            return scaled_visible_term + hidden_term

    @torch.no_grad()
    def init_vb_from_means(self, means: torch.Tensor) -> None:
        """Initialize visible biases from data means.

        Parameters
        ----------
        means : torch.Tensor
            Mean values of visible units from the training data.

        Notes
        -----
        This internal method sets the visible biases and base rates to match
        the marginal statistics of the training data. It converts the mean
        values to logits after clamping them to avoid numerical issues.
        """
        clamped = means.clamp(1e-3, 1 - 1e-3)
        logits = torch.logit(clamped)
        self.vb.copy_(logits)
        self.base_rate_vb.copy_(logits)
