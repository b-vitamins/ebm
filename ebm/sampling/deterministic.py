"""Deterministic approximation methods for sampling.

This module implements mean-field methods including the TAP
(Thouless-Anderson-Palmer) approximation which provides
deterministic updates instead of stochastic sampling.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from ebm.core.registry import register_sampler
from ebm.models.base import EnergyBasedModel, LatentVariableModel

from .base import GradientEstimator, Sampler


@register_sampler("mean_field", aliases=["mf", "naive_mf"])
class MeanFieldSampler(Sampler):
    """Naive mean-field approximation sampler.

    This implements the simplest mean-field approximation where
    we iteratively update the mean activations of each layer.
    """

    def __init__(
        self, num_iter: int = 10, damping: float = 0.0, tol: float = 1e-4
    ):
        """Initialize mean-field sampler.

        Args:
            num_iter: Maximum number of iterations
            damping: Damping factor for updates (0 = no damping)
            tol: Convergence tolerance
        """
        super().__init__(name="MeanField")
        self.num_iter = num_iter
        self.damping = damping
        self.tol = tol

    def sample(  # noqa: C901
        self,
        model: EnergyBasedModel,
        init_state: Tensor,
        num_steps: int | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Run mean-field approximation.

        Args:
            model: Energy model (must be LatentVariableModel)
            init_state: Initial visible state
            num_steps: Number of iterations (overrides num_iter)
            **kwargs: Additional arguments

        Returns
        -------
            Mean-field approximation of visible units
        """
        if not isinstance(model, LatentVariableModel):
            raise TypeError("Mean-field requires LatentVariableModel")

        num_iter = num_steps or self.num_iter

        # Initialize mean activations
        m_v = init_state.clamp(0, 1)  # Ensure valid probabilities
        m_h = model.sample_hidden(m_v, return_prob=True)[1]

        # Iterate mean-field updates
        for i in range(num_iter):
            m_v_old = m_v

            # Update visible given hidden means
            pre_v = F.linear(m_h, model.W.t(), model.vbias)
            m_v_new = torch.sigmoid(pre_v)

            # Apply damping
            if self.damping > 0:
                m_v = (1 - self.damping) * m_v_new + self.damping * m_v
            else:
                m_v = m_v_new

            # Update hidden given visible means
            pre_h = F.linear(m_v, model.W, model.hbias)
            m_h = torch.sigmoid(pre_h)

            # Check convergence
            if (m_v - m_v_old).abs().max() < self.tol:
                self.log_debug(f"Mean-field converged at iteration {i + 1}")
                break

        self.state.num_steps += i + 1

        # Return binary samples based on mean activations
        return (m_v > 0.5).to(m_v.dtype)


@register_sampler("tap", aliases=["thouless_anderson_palmer"])
class TAPSampler(Sampler):
    """Thouless-Anderson-Palmer mean-field approximation.

    TAP includes second-order corrections to the naive mean-field
    approximation, providing better accuracy especially for models
    with strong interactions.
    """

    def __init__(
        self,
        num_iter: int = 20,
        damping: float = 0.5,
        tol: float = 1e-3,
        order: str = "tap2",
        adaptive_damping: bool = True,
    ):
        """Initialize TAP sampler.

        Args:
            num_iter: Maximum iterations
            damping: Initial damping factor
            tol: Convergence tolerance
            order: TAP order ('naive', 'tap2', 'tap3')
            adaptive_damping: Whether to adapt damping
        """
        super().__init__(name=f"TAP-{order}")
        self.num_iter = num_iter
        self.damping = damping
        self.tol = tol
        self.order = order
        self.adaptive_damping = adaptive_damping

        # Cache for weight powers
        self.register_buffer("W2", None)
        self.register_buffer("W3", None)

    def _cache_weight_powers(self, model: LatentVariableModel) -> None:
        """Cache powers of weight matrix for efficiency."""
        if self.W2 is None or self.W2.shape != model.W.shape:
            self.W2 = model.W**2
            if self.order == "tap3":
                self.W3 = model.W**3

    def sample(  # noqa: C901
        self,
        model: EnergyBasedModel,
        init_state: Tensor,
        num_steps: int | None = None,
        return_magnetizations: bool = False,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Run TAP approximation.

        Args:
            model: Energy model
            init_state: Initial state
            num_steps: Number of iterations
            return_magnetizations: If True, return mean activations
            **kwargs: Additional arguments

        Returns
        -------
            Samples (and magnetizations if requested)
        """
        if not isinstance(model, LatentVariableModel):
            raise TypeError("TAP requires LatentVariableModel")

        self._cache_weight_powers(model)
        num_iter = num_steps or self.num_iter

        # Initialize magnetizations
        m_v = init_state.clamp(0, 1)
        m_h = model.sample_hidden(m_v, return_prob=True)[1]

        # Adaptive damping state
        damping = self.damping
        prev_error = float("inf")

        # TAP iterations
        for i in range(num_iter):
            m_v_old = m_v.clone()
            m_h_old = m_h.clone()

            # TAP equations for visible units
            field_v = F.linear(m_h, model.W.t(), model.vbias)

            if self.order != "naive":
                # Second-order TAP correction
                correction_v = F.linear(m_h * (1 - m_h), self.W2.t()) * (
                    0.5 - m_v
                )
                field_v += correction_v

                if self.order == "tap3":
                    # Third-order correction
                    h2_var = m_h * (1 - m_h)
                    v2_var = m_v * (1 - m_v)
                    correction3_v = F.linear(
                        h2_var * (1 - 2 * m_h), self.W3.t()
                    ) * (1 / 3 - 2 * v2_var)
                    field_v += correction3_v

            m_v_new = torch.sigmoid(field_v)

            # TAP equations for hidden units
            field_h = F.linear(m_v, model.W, model.hbias)

            if self.order != "naive":
                # Second-order TAP correction
                correction_h = F.linear(m_v * (1 - m_v), self.W2) * (0.5 - m_h)
                field_h += correction_h

                if self.order == "tap3":
                    # Third-order correction
                    v2_var = m_v * (1 - m_v)
                    h2_var = m_h * (1 - m_h)
                    correction3_h = F.linear(
                        v2_var * (1 - 2 * m_v), self.W3
                    ) * (1 / 3 - 2 * h2_var)
                    field_h += correction3_h

            m_h_new = torch.sigmoid(field_h)

            # Apply damping
            m_v = (1 - damping) * m_v_new + damping * m_v_old
            m_h = (1 - damping) * m_h_new + damping * m_h_old

            # Check convergence
            error = torch.max(
                (m_v - m_v_old).abs().max(), (m_h - m_h_old).abs().max()
            ).item()

            if error < self.tol:
                self.log_debug(f"TAP converged at iteration {i + 1}")
                break

            # Adaptive damping
            if self.adaptive_damping and i > 0:
                if error > prev_error:
                    # Increase damping if error increased
                    damping = min(0.9, damping * 1.1)
                else:
                    # Decrease damping if error decreased
                    damping = max(0.1, damping * 0.95)

            prev_error = error

        self.state.num_steps += i + 1

        # Sample from magnetizations
        v_sample = (m_v > torch.rand_like(m_v)).to(m_v.dtype)

        if return_magnetizations:
            return v_sample, m_v, m_h
        return v_sample


@register_sampler("tap_gradient", aliases=["tap_cd"])
class TAPGradientEstimator(GradientEstimator):
    """Gradient estimation using TAP approximation."""

    def __init__(
        self, num_iter: int = 20, damping: float = 0.5, order: str = "tap2"
    ):
        """Initialize TAP gradient estimator.

        Args:
            num_iter: TAP iterations
            damping: Damping factor
            order: TAP order
        """
        sampler = TAPSampler(num_iter=num_iter, damping=damping, order=order)
        super().__init__(sampler)

    def estimate_gradient(
        self, model: EnergyBasedModel, data: Tensor, **kwargs: Any
    ) -> dict[str, Tensor]:
        """Estimate gradients using TAP.

        Args:
            model: Energy model
            data: Training data
            **kwargs: Additional arguments

        Returns
        -------
            Parameter gradients
        """
        if not isinstance(model, LatentVariableModel):
            raise TypeError("TAP requires LatentVariableModel")

        # Get magnetizations for positive phase
        _, m_v_data, m_h_data = self.sampler.sample(
            model, data, return_magnetizations=True
        )

        # Get magnetizations for negative phase
        # Initialize from uniform distribution for negative phase
        v_init = torch.rand_like(data)
        _, m_v_model, m_h_model = self.sampler.sample(
            model, v_init, return_magnetizations=True
        )

        # Compute gradients using magnetizations
        from ebm.utils.tensor import batch_outer_product

        gradients = {}

        # Weight gradient
        pos_corr = batch_outer_product(m_h_data, m_v_data).mean(dim=0)
        neg_corr = batch_outer_product(m_h_model, m_v_model).mean(dim=0)
        gradients["W"] = pos_corr - neg_corr

        # Bias gradients
        if hasattr(model, "vbias") and model.vbias.requires_grad:
            gradients["vbias"] = m_v_data.mean(dim=0) - m_v_model.mean(dim=0)

        if hasattr(model, "hbias") and model.hbias.requires_grad:
            gradients["hbias"] = m_h_data.mean(dim=0) - m_h_model.mean(dim=0)

        return gradients


@register_sampler("belief_propagation", aliases=["bp"])
class BeliefPropagationSampler(Sampler):
    """Belief Propagation for tree-structured or loopy graphs.

    This is primarily useful for sparse RBMs or when the weight
    matrix has special structure.
    """

    def __init__(
        self, num_iter: int = 50, damping: float = 0.5, tol: float = 1e-4
    ):
        """Initialize BP sampler.

        Args:
            num_iter: Maximum iterations
            damping: Message damping
            tol: Convergence tolerance
        """
        super().__init__(name="BeliefPropagation")
        self.num_iter = num_iter
        self.damping = damping
        self.tol = tol

    def sample(
        self,
        model: EnergyBasedModel,
        init_state: Tensor,
        num_steps: int | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Run belief propagation.

        Note: This is a simplified implementation for bipartite RBMs.
        For general graphs, a more sophisticated message-passing
        scheme would be needed.

        Args:
            model: Energy model
            init_state: Initial state
            num_steps: Number of iterations
            **kwargs: Additional arguments

        Returns
        -------
            Approximate samples
        """
        if not isinstance(model, LatentVariableModel):
            raise TypeError("BP requires LatentVariableModel")

        num_iter = num_steps or self.num_iter
        batch_size = init_state.shape[0]

        # Initialize messages (log-space for stability)
        # Messages from visible to hidden
        msg_v_to_h = torch.zeros(
            batch_size,
            model.num_visible,
            model.num_hidden,
            device=init_state.device,
        )
        # Messages from hidden to visible
        msg_h_to_v = torch.zeros(
            batch_size,
            model.num_hidden,
            model.num_visible,
            device=init_state.device,
        )

        # BP iterations
        for i in range(num_iter):
            old_msg_v_to_h = msg_v_to_h.clone()

            # Update messages from visible to hidden
            # Aggregate messages from other hidden units
            field_v = model.vbias.unsqueeze(0).unsqueeze(-1)
            field_v = field_v + msg_h_to_v.sum(dim=1, keepdim=True)

            # Compute new messages
            new_msg_v_to_h = model.W.unsqueeze(0) * torch.tanh(field_v / 2)

            # Damping
            msg_v_to_h = (
                1 - self.damping
            ) * new_msg_v_to_h + self.damping * msg_v_to_h

            # Update messages from hidden to visible
            field_h = model.hbias.unsqueeze(0).unsqueeze(-1)
            field_h = field_h + msg_v_to_h.sum(dim=1, keepdim=True).transpose(
                -2, -1
            )

            new_msg_h_to_v = model.W.t().unsqueeze(0) * torch.tanh(field_h / 2)
            msg_h_to_v = (
                1 - self.damping
            ) * new_msg_h_to_v + self.damping * msg_h_to_v

            # Check convergence
            if (msg_v_to_h - old_msg_v_to_h).abs().max() < self.tol:
                self.log_debug(f"BP converged at iteration {i + 1}")
                break

        # Compute marginals
        # Visible marginals
        field_v_final = model.vbias.unsqueeze(0) + msg_h_to_v.sum(dim=1)
        prob_v = torch.sigmoid(field_v_final)

        # Sample
        v_sample = (prob_v > torch.rand_like(prob_v)).to(prob_v.dtype)

        self.state.num_steps += i + 1
        return v_sample
