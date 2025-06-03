"""Parameter initialization strategies for neural networks and EBMs.

This module provides flexible initialization methods for model parameters,
supporting various strategies from simple constant initialization to
sophisticated schemes like Xavier and Kaiming initialization.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
from torch import Tensor, nn

from ebm.core.types import InitStrategy


class InitMethod(str, Enum):
    """Enumeration of initialization methods."""

    # Basic methods
    ZEROS = "zeros"
    ONES = "ones"
    CONSTANT = "constant"
    NORMAL = "normal"
    UNIFORM = "uniform"

    # Advanced methods
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    HE_UNIFORM = "he_uniform"  # Alias for kaiming_uniform
    HE_NORMAL = "he_normal"  # Alias for kaiming_normal

    # Special methods
    ORTHOGONAL = "orthogonal"
    SPARSE = "sparse"
    EYE = "eye"
    DIRAC = "dirac"


class Initializer:
    """Flexible parameter initializer supporting various strategies."""

    def __init__(self, method: InitStrategy, **kwargs: Any):
        """Initialize the initializer.

        Args:
            method: Initialization method (string, callable, or tensor)
            **kwargs: Additional arguments for the initialization method
        """
        self.method = method
        self.kwargs = kwargs
        self._init_fn = self._resolve_init_fn()

    def _resolve_init_fn(self) -> Callable[[Tensor], None]:  # noqa: C901
        """Resolve initialization method to a callable."""
        # Handle direct callables
        if callable(self.method):
            return self.method

        # Handle tensor initialization (copy from existing tensor)
        if isinstance(self.method, Tensor):

            def copy_init(tensor: Tensor) -> None:
                if tensor.shape != self.method.shape:
                    raise ValueError(
                        f"Shape mismatch: {tensor.shape} vs {self.method.shape}"
                    )
                with torch.no_grad():
                    tensor.copy_(self.method)

            return copy_init

        # Handle constant initialization
        if isinstance(self.method, int | float):
            return lambda t: nn.init.constant_(t, self.method)

        # Handle string methods
        if isinstance(self.method, str):
            method = self.method.lower()

            # Basic methods
            if method in {InitMethod.ZEROS, "zero"}:
                return nn.init.zeros_
            if method in {InitMethod.ONES, "one"}:
                return nn.init.ones_
            if method == InitMethod.CONSTANT:
                val = self.kwargs.get("val", 0.0)
                return lambda t: nn.init.constant_(t, val)

            # Normal distribution
            if method == InitMethod.NORMAL:
                mean = self.kwargs.get("mean", 0.0)
                std = self.kwargs.get("std", 0.01)
                return lambda t: nn.init.normal_(t, mean=mean, std=std)

            # Uniform distribution
            if method == InitMethod.UNIFORM:
                a = self.kwargs.get("a", -0.1)
                b = self.kwargs.get("b", 0.1)
                return lambda t: nn.init.uniform_(t, a=a, b=b)

            # Xavier/Glorot initialization
            if method == InitMethod.XAVIER_UNIFORM:
                gain = self.kwargs.get("gain", 1.0)
                return lambda t: nn.init.xavier_uniform_(t, gain=gain)
            if method == InitMethod.XAVIER_NORMAL:
                gain = self.kwargs.get("gain", 1.0)
                return lambda t: nn.init.xavier_normal_(t, gain=gain)

            # Kaiming/He initialization
            if method in {InitMethod.KAIMING_UNIFORM, InitMethod.HE_UNIFORM}:
                a = self.kwargs.get("a", 0)
                mode = self.kwargs.get("mode", "fan_in")
                nonlinearity = self.kwargs.get("nonlinearity", "leaky_relu")
                return lambda t: nn.init.kaiming_uniform_(
                    t, a=a, mode=mode, nonlinearity=nonlinearity
                )
            if method in {InitMethod.KAIMING_NORMAL, InitMethod.HE_NORMAL}:
                a = self.kwargs.get("a", 0)
                mode = self.kwargs.get("mode", "fan_in")
                nonlinearity = self.kwargs.get("nonlinearity", "leaky_relu")
                return lambda t: nn.init.kaiming_normal_(
                    t, a=a, mode=mode, nonlinearity=nonlinearity
                )

            # Special methods
            if method == InitMethod.ORTHOGONAL:
                gain = self.kwargs.get("gain", 1.0)
                return lambda t: nn.init.orthogonal_(t, gain=gain)

            if method == InitMethod.SPARSE:
                sparsity = self.kwargs.get("sparsity", 0.1)
                std = self.kwargs.get("std", 0.01)
                return lambda t: nn.init.sparse_(t, sparsity=sparsity, std=std)

            if method == InitMethod.EYE:
                return self._eye_init

            if method == InitMethod.DIRAC:
                groups = self.kwargs.get("groups", 1)
                return lambda t: nn.init.dirac_(t, groups=groups)

            raise ValueError(f"Unknown initialization method: {method}")

        raise TypeError(
            f"Invalid initialization method type: {type(self.method)}"
        )

    def _eye_init(self, tensor: Tensor) -> None:
        """Initialize as identity matrix (or as close as possible)."""
        with torch.no_grad():
            tensor.zero_()
            if tensor.dim() == 2:
                # Regular matrix
                n = min(tensor.shape)
                tensor[:n, :n] = torch.eye(
                    n, device=tensor.device, dtype=tensor.dtype
                )
            else:
                # Higher dimensional tensor - initialize as identity along last 2 dims
                n = min(tensor.shape[-2:])
                eye = torch.eye(n, device=tensor.device, dtype=tensor.dtype)
                tensor[..., :n, :n] = eye

    def __call__(self, tensor: Tensor) -> None:
        """Apply initialization to tensor.

        Args:
            tensor: Tensor to initialize (modified in-place)
        """
        self._init_fn(tensor)

    def create_parameter(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        requires_grad: bool = True,
    ) -> nn.Parameter:
        """Create and initialize a parameter.

        Args:
            shape: Shape of the parameter
            dtype: Data type
            device: Device
            requires_grad: Whether parameter requires gradients

        Returns
        -------
            Initialized parameter
        """
        tensor = torch.empty(shape, dtype=dtype, device=device)
        self(tensor)
        return nn.Parameter(tensor, requires_grad=requires_grad)

    def create_buffer(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        """Create and initialize a buffer (non-parameter tensor).

        Args:
            shape: Shape of the buffer
            dtype: Data type
            device: Device

        Returns
        -------
            Initialized buffer
        """
        tensor = torch.empty(shape, dtype=dtype, device=device)
        self(tensor)
        return tensor


def get_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    """Calculate fan_in and fan_out for a tensor.

    Args:
        tensor: Tensor to analyze

    Returns
    -------
        Tuple of (fan_in, fan_out)
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1

    if dimensions > 2:
        # For convolutional layers
        for s in tensor.shape[2:]:
            receptive_field_size *= s

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    """Calculate the recommended gain value for the given nonlinearity function.

    Args:
        nonlinearity: Name of the nonlinearity function
        param: Optional parameter for the nonlinearity function

    Returns
    -------
        Recommended gain value
    """
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]

    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    if nonlinearity == "tanh":
        return 5.0 / 3
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        negative_slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1 + negative_slope**2))
    if nonlinearity == "selu":
        return 3.0 / 4
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def initialize_module(
    module: nn.Module,
    weight_init: InitStrategy | None = None,
    bias_init: InitStrategy | None = None,
    **init_kwargs: Any,
) -> None:
    """Initialize all parameters in a module.

    Args:
        module: Module to initialize
        weight_init: Weight initialization strategy
        bias_init: Bias initialization strategy
        **init_kwargs: Additional arguments for initializers
    """
    # Create initializers
    weight_initializer = (
        Initializer(weight_init, **init_kwargs) if weight_init else None
    )
    bias_initializer = (
        Initializer(bias_init, **init_kwargs) if bias_init else None
    )

    # Apply to all submodules
    for name, param in module.named_parameters():
        if "weight" in name and weight_initializer:
            weight_initializer(param.data)
        elif "bias" in name and bias_initializer:
            bias_initializer(param.data)


# Convenience functions for common initialization patterns
def uniform_init(low: float = -0.1, high: float = 0.1) -> Initializer:
    """Create uniform initializer."""
    return Initializer("uniform", a=low, b=high)


def normal_init(mean: float = 0.0, std: float = 0.01) -> Initializer:
    """Create normal initializer."""
    return Initializer("normal", mean=mean, std=std)


def xavier_init(uniform: bool = True, gain: float = 1.0) -> Initializer:
    """Create Xavier/Glorot initializer."""
    method = "xavier_uniform" if uniform else "xavier_normal"
    return Initializer(method, gain=gain)


def kaiming_init(
    uniform: bool = True,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Initializer:
    """Create Kaiming/He initializer."""
    method = "kaiming_uniform" if uniform else "kaiming_normal"
    return Initializer(method, a=a, mode=mode, nonlinearity=nonlinearity)


def init_from_data_statistics(
    data_mean: Tensor | None = None,
    data_std: Tensor | None = None,
    scale: float = 1.0,
) -> Callable[[Tensor], None]:
    """Create initializer based on data statistics.

    This is particularly useful for initializing visible biases in RBMs
    to match the data distribution.

    Args:
        data_mean: Mean of the data
        data_std: Standard deviation of the data
        scale: Scaling factor

    Returns
    -------
        Initialization function
    """

    def init_fn(tensor: Tensor) -> None:
        with torch.no_grad():
            if (
                data_mean is not None
                and tensor.shape[-1] == data_mean.shape[-1]
            ):
                # Initialize to scaled data mean
                tensor.copy_(data_mean * scale)
            elif data_std is not None and len(tensor.shape) >= 2:
                # Initialize weights based on data variance
                fan_in, fan_out = get_fan_in_and_fan_out(tensor)
                std = data_std.mean() * scale / math.sqrt(fan_in)
                tensor.normal_(0, std)
            else:
                # Fallback to standard normal
                tensor.normal_(0, 0.01 * scale)

    return init_fn
