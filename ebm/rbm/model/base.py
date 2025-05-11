"""Base classes for Restricted Boltzmann Machine implementations.

This module defines the foundational interfaces and configurations for
Restricted Boltzmann Machine (RBM) models, including the base abstract classes
that all RBM variants must implement and adapters for Annealed Importance Sampling.

RBMs are energy-based probabilistic models consisting of visible and hidden units
that can learn complex probability distributions. They serve as building blocks
for deep generative models and feature extractors in machine learning applications.

The module provides flexible initialization utilities, standardized configuration,
and a comprehensive abstract interface for implementing various RBM variants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

# Type aliases for initialization parameters
WeightInit = torch.Tensor | float | None
BiasInit = torch.Tensor | float | None


def init_weight_tensor(
    init: WeightInit,
    visible_size: int,
    hidden_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Initialize a weight tensor based on the initialization strategy.

    Parameters
    ----------
    init : WeightInit
        Initialization strategy, which can be:
        - A tensor to use directly (must match hidden_size x visible_size)
        - A float specifying the standard deviation for normal initialization
        - None to use default standard deviation: sqrt(1/sqrt(visible_size * hidden_size))
    visible_size : int
        Number of visible units
    hidden_size : int
        Number of hidden units
    device : torch.device or str or None, optional
        Device to place the tensor on, by default None
    dtype : torch.dtype or None, optional
        Data type of the tensor, by default None

    Returns
    -------
    torch.Tensor
        The initialized weight tensor with shape (hidden_size, visible_size)

    Raises
    ------
    ValueError
        If a tensor is provided with incorrect shape
    """
    if isinstance(init, torch.Tensor):
        if init.shape != (hidden_size, visible_size):
            raise ValueError(
                f"Weight tensor shape mismatch: expected ({hidden_size}, "
                f"{visible_size}), got {init.shape}"
            )
        return init.to(device=device, dtype=dtype)

    # Create empty tensor
    weights = torch.empty((hidden_size, visible_size), device=device, dtype=dtype)

    # Determine standard deviation
    if init is None:
        # Default: use sqrt(1/sqrt(visible * hidden))
        std = 1.0 / (visible_size * hidden_size) ** 0.25
    else:
        # User-specified standard deviation
        std = float(init)

    # Initialize with normal distribution
    nn.init.normal_(weights, mean=0.0, std=std)
    return weights


def init_bias_tensor(
    init: BiasInit,
    size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Initialize a bias tensor based on the initialization strategy.

    Parameters
    ----------
    init : BiasInit
        Initialization strategy, which can be:
        - A tensor to use directly (must match size)
        - A float for constant initialization
        - None for default initialization to 0.0
    size : int
        Size of the bias vector
    device : torch.device or str or None, optional
        Device to place the tensor on, by default None
    dtype : torch.dtype or None, optional
        Data type of the tensor, by default None

    Returns
    -------
    torch.Tensor
        The initialized bias tensor with shape (size,)

    Raises
    ------
    ValueError
        If a tensor is provided with incorrect shape
    """
    if isinstance(init, torch.Tensor):
        if init.shape != (size,):
            raise ValueError(f"Bias tensor shape mismatch: expected ({size},), got {init.shape}")
        return init.to(device=device, dtype=dtype)

    # Create bias tensor
    bias = torch.empty(size, device=device, dtype=dtype)

    # Initialize with constant value (0.0 if init is None)
    value = 0.0 if init is None else float(init)
    nn.init.constant_(bias, value)
    return bias


@dataclass(slots=True, frozen=True)
class RBMConfig:
    """Configuration for Restricted Boltzmann Machine models.

    This dataclass defines the architecture and initialization parameters
    for RBM models, providing a standardized way to configure various RBM
    implementations.

    Parameters
    ----------
    visible : int
        Number of visible units in the RBM.
    hidden : int
        Number of hidden units in the RBM.
    w_init : WeightInit, optional
        Weight initialization, which can be:
        - A weight tensor with shape (hidden, visible)
        - A float specifying standard deviation for normal initialization
        - None to use default std=sqrt(1/sqrt(visible*hidden)), by default None
    vb_init : BiasInit, optional
        Visible bias initialization, which can be:
        - A bias tensor with shape (visible,)
        - A float for constant initialization
        - None for default 0.0 initialization, by default None
    hb_init : BiasInit, optional
        Hidden bias initialization, which can be:
        - A bias tensor with shape (hidden,)
        - A float for constant initialization
        - None for default 0.0 initialization, by default None
    v_act : Callable[[torch.Tensor], torch.Tensor], optional
        Activation function for visible units, by default torch.sigmoid.
    h_act : Callable[[torch.Tensor], torch.Tensor], optional
        Activation function for hidden units, by default torch.sigmoid.
    dtype : torch.dtype or None, optional
        Data type for model parameters, by default None.
    device : torch.device or str or None, optional
        Device to use for model parameters, by default None.

    Examples
    --------
    >>> config = RBMConfig(
    ...     visible=784,
    ...     hidden=500,
    ...     w_init=0.01,  # Initialize weights with standard deviation 0.01
    ...     vb_init=-4.0,  # Initialize visible biases with constant -4.0
    ...     dtype=torch.float32,
    ...     device="cuda" if torch.cuda.is_available() else "cpu"
    ... )
    """

    visible: int
    hidden: int
    w_init: WeightInit = None
    vb_init: BiasInit = None
    hb_init: BiasInit = None
    v_act: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid
    h_act: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid
    dtype: torch.dtype | None = None
    device: torch.device | str | None = None


class BaseRBM(nn.Module, ABC):
    """Abstract interface of an RBM variant.

    This class defines the interface that all RBM variants must implement.
    It extends PyTorch's nn.Module with RBM-specific methods and provides
    support for Parallel Tempering through an optional `beta` parameter.

    Attributes
    ----------
    cfg : RBMConfig
        Configuration parameters for the RBM.

    Notes
    -----
    The optional `beta` parameter in methods is used for Parallel Tempering,
    which enables running multiple replicas at different temperatures.

    Tensor shapes and broadcasting for Parallel Tempering:

    1. Standard operation (no beta parameter):
       - Input tensors v: (B, V) and h: (B, H)
       - Parameter tensors W: (H, V), vb: (V,), hb: (H,)
       - Output tensors follow the same batch dimension

    2. Parallel Tempering (with beta parameter):
       - Input tensors v: (B, K, V) and h: (B, K, H), where K is replica dimension
       - Beta tensor: (1, K, 1) for batch-independent temperature scaling
       - Implementations should take care to properly broadcast the parameters
       - Output tensors maintain the batch and replica dimensions

    When implementing methods for parallel tempering, ensure that parameter
    scaling and tensor operations maintain proper broadcasting across all dimensions.
    """

    cfg: RBMConfig

    @abstractmethod
    def preact_h(self, v: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate pre-activation values for hidden units.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit values with shape (B, V) or (B, K, V).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape (1, K, 1)
            where K is the number of replicas. If None, standard behavior
            without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Pre-activation values for hidden units with shape (B, H) if beta is None
            or (B, K, H) if beta is provided.

        Notes
        -----
        This method computes the linear pre-activation of hidden units before
        the activation function is applied. For a standard RBM, this is W·v + hb.

        When beta is provided, implementations must ensure proper broadcasting:
        - For weights: scaled_W = beta * W  # beta: (1,K,1), W: (H,V)
        - For hidden bias: scaled_hb = beta * hb  # beta: (1,K,1), hb: (H,)

        Care should be taken with tensor operations to maintain correct dimensions.
        """
        ...

    @abstractmethod
    def preact_v(self, h: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate pre-activation values for visible units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden unit values with shape (B, H) or (B, K, H).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape (1, K, 1)
            where K is the number of replicas. If None, standard behavior
            without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Pre-activation values for visible units with shape (B, V) if beta is None
            or (B, K, V) if beta is provided.

        Notes
        -----
        This method computes the linear pre-activation of visible units before
        the activation function is applied. For a standard RBM, this is
        W^T·h + vb.

        When beta is provided, implementations must ensure proper broadcasting:
        - For weights: scaled_W = beta * W  # beta: (1,K,1), W: (H,V)
        - For visible bias: scaled_vb = beta * vb  # beta: (1,K,1), vb: (V,)

        Matrix transposition must be handled carefully when using batched operations.
        """
        ...

    @abstractmethod
    def prob_h_given_v(self, v: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate conditional probabilities of hidden units given visible units.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit values with shape (B, V) or (B, K, V).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape (1, K, 1).
            If None, standard behavior without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Conditional probabilities of hidden units with shape (B, H) if beta is None
            or (B, K, H) if beta is provided.

        Notes
        -----
        For binary RBMs, this typically applies the sigmoid function to the
        pre-activation values from preact_h. For other unit types, different
        activation functions may be used as specified in the RBM configuration.

        This method should typically call preact_h and then apply the hidden
        activation function from the configuration:

        ```python
        def prob_h_given_v(self, v, beta=None):
            return self.cfg.h_act(self.preact_h(v, beta=beta))
        ```
        """
        ...

    @abstractmethod
    def prob_v_given_h(self, h: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate conditional probabilities of visible units given hidden units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden unit values with shape (B, H) or (B, K, H).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape (1, K, 1).
            If None, standard behavior without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Conditional probabilities of visible units with shape (B, V) if beta is None
            or (B, K, V) if beta is provided.

        Notes
        -----
        For binary RBMs, this typically applies the sigmoid function to the
        pre-activation values from preact_v. For other unit types, different
        activation functions may be used as specified in the RBM configuration.

        This method should typically call preact_v and then apply the visible
        activation function from the configuration:

        ```python
        def prob_v_given_h(self, h, beta=None):
            return self.cfg.v_act(self.preact_v(h, beta=beta))
        ```
        """
        ...

    @abstractmethod
    def sample_h_given_v(
        self, v: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample hidden units given visible units.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit values with shape (B, V) or (B, K, V).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape (1, K, 1).
            If None, standard behavior without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Sampled hidden unit values with shape (B, H) if beta is None
            or (B, K, H) if beta is provided.

        Notes
        -----
        This method samples from the conditional distribution P(h|v).
        For binary units, this typically involves sampling from a Bernoulli
        distribution with probabilities from prob_h_given_v.

        A common implementation for binary units:

        ```python
        def sample_h_given_v(self, v, beta=None):
            p_h = self.prob_h_given_v(v, beta=beta)
            return torch.bernoulli(p_h)
        ```
        """
        ...

    @abstractmethod
    def sample_v_given_h(
        self, h: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample visible units given hidden units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden unit values with shape (B, H) or (B, K, H).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape (1, K, 1).
            If None, standard behavior without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Sampled visible unit values with shape (B, V) if beta is None
            or (B, K, V) if beta is provided.

        Notes
        -----
        This method samples from the conditional distribution P(v|h).
        For binary units, this typically involves sampling from a Bernoulli
        distribution with probabilities from prob_v_given_h.

        A common implementation for binary units:

        ```python
        def sample_v_given_h(self, h, beta=None):
            p_v = self.prob_v_given_h(h, beta=beta)
            return torch.bernoulli(p_v)
        ```
        """
        ...

    @abstractmethod
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
            Visible unit values with shape (B, V) or (B, K, V).
        h : torch.Tensor
            Hidden unit values with shape (B, H) or (B, K, H).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape (1, K, 1).
            If None, standard behavior without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Energy values with shape (B,) if beta is None or (B, K) if beta is provided.

        Notes
        -----
        This method computes the energy function E(v,h) for the RBM. The energy
        function defines the joint probability distribution through the relation
        P(v,h) ∝ exp(-E(v,h)). For a standard binary RBM, this is typically
        E(v,h) = -h^T·W·v - vb^T·v - hb^T·h.

        For the parallel tempering case, all parameters should be scaled by beta:
        ```
        scaled_W = beta * W
        scaled_vb = beta * vb
        scaled_hb = beta * hb
        ```

        Care must be taken to ensure proper broadcasting when computing energy terms:
        ```
        term1 = -torch.sum(h * torch.matmul(scaled_W, v), dim=-1)
        term2 = -torch.sum(v * scaled_vb, dim=-1)
        term3 = -torch.sum(h * scaled_hb, dim=-1)
        ```
        """
        ...

    @abstractmethod
    def free_energy(self, v: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate the free energy of visible configurations.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit values with shape (B, V) or (B, K, V).
        beta : torch.Tensor or None, optional
            Temperature parameter for Parallel Tempering with shape (1, K, 1).
            If None, standard behavior without temperature scaling is used.

        Returns
        -------
        torch.Tensor
            Free energy values with shape (B,) if beta is None or (B, K) if beta is provided.

        Notes
        -----
        The free energy F(v) is defined as -log(sum_h exp(-E(v,h))). It
        represents the effective energy of a visible configuration after
        marginalizing out the hidden units. For training and evaluation of RBMs,
        the free energy is often more useful than the energy itself.

        For a standard binary RBM, the free energy can be computed as:
        F(v) = -vb^T·v - sum_i log(1 + exp(W_i·v + hb_i))

        For the parallel tempering case, all parameters should be scaled by beta:
        ```
        scaled_W = beta * W
        scaled_vb = beta * vb
        scaled_hb = beta * hb
        ```

        And the free energy computation should use these scaled parameters.
        """
        ...
