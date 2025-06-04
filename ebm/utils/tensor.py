"""Tensor manipulation and utility functions.

This module provides efficient tensor operations commonly used in
energy-based models, with support for batching, broadcasting, and
device management.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import torch
from torch import Tensor

from ebm.core.types_ import Device, DType, Shape, TensorLike

MAT_DIM_2 = 2
MAT_DIM_3 = 3


def ensure_tensor(
    x: TensorLike, dtype: DType | None = None, device: Device | None = None
) -> Tensor:
    """Convert input to tensor with specified dtype and device.

    Args:
        x: Input data (tensor, list, or scalar)
        dtype: Target data type
        device: Target device

    Returns
    -------
        Tensor with specified properties
    """
    if isinstance(x, Tensor):
        if dtype is not None or device is not None:
            return x.to(dtype=dtype, device=device)
        return x

    # Convert to tensor first
    if isinstance(x, list | tuple):
        tensor = torch.tensor(x)
    elif isinstance(x, int | float):
        tensor = torch.tensor([x])
    else:
        # Handle numpy arrays properly
        import numpy as np

        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x).float()  # Default to float for numpy
        else:
            tensor = torch.as_tensor(x)

    # Then move to correct dtype/device
    if dtype is not None or device is not None:
        tensor = tensor.to(dtype=dtype, device=device)

    return tensor


def safe_log(x: Tensor, eps: float = 1e-10) -> Tensor:
    """Compute log with numerical stability.

    Args:
        x: Input tensor
        eps: Small constant for numerical stability

    Returns
    -------
        log(x + eps)
    """
    # Ensure x is at least eps (handles negative values)
    x_safe = torch.maximum(x, torch.tensor(eps, device=x.device, dtype=x.dtype))
    return torch.log(x_safe)


def safe_sqrt(x: Tensor, eps: float = 1e-10) -> Tensor:
    """Compute sqrt with numerical stability.

    Args:
        x: Input tensor
        eps: Small constant for numerical stability

    Returns
    -------
        sqrt(x + eps)
    """
    # Ensure x is at least eps (handles negative values)
    x_safe = torch.maximum(x, torch.tensor(eps, device=x.device, dtype=x.dtype))
    return torch.sqrt(x_safe)


def log_sum_exp(
    x: Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
) -> Tensor:
    """Compute log(sum(exp(x))) in a numerically stable way.

    Args:
        x: Input tensor
        dim: Dimension(s) to reduce
        keepdim: Whether to keep reduced dimensions

    Returns
    -------
        Log sum exp result
    """
    if dim is None:
        # Reduce over all dimensions
        return torch.logsumexp(x.flatten(), dim=0, keepdim=keepdim)
    return torch.logsumexp(x, dim=dim, keepdim=keepdim)


def batch_outer_product(a: Tensor, b: Tensor) -> Tensor:
    """Compute batch outer product efficiently.

    Args:
        a: Tensor of shape (batch_size, n)
        b: Tensor of shape (batch_size, m)

    Returns
    -------
        Outer products of shape (batch_size, n, m)
    """
    return torch.einsum("bi,bj->bij", a, b)


def batch_quadratic_form(x: Tensor, matrix: Tensor) -> Tensor:
    """Compute x^T A x for batched inputs.

    Args:
        x: Tensor of shape (batch_size, n)
        matrix: Matrix of shape (n, n) or (batch_size, n, n)

    Returns
    -------
        Quadratic form values of shape (batch_size,)
    """
    if matrix.dim() == MAT_DIM_2:
        # Single matrix for all batch elements
        return torch.einsum("bi,ij,bj->b", x, matrix, x)
    # Different matrix for each batch element
    return torch.einsum("bi,bij,bj->b", x, matrix, x)


def batch_mv(matrix: Tensor, x: Tensor) -> Tensor:
    """Batched matrix-vector multiplication.

    Args:
        matrix: Matrix of shape (n, m) or (batch_size, n, m)
        x: Vector of shape (m,) or (batch_size, m)

    Returns
    -------
        Result of shape (n,) or (batch_size, n)
    """
    if matrix.dim() == MAT_DIM_2 and x.dim() == 1:
        return torch.mv(matrix, x)
    if matrix.dim() == MAT_DIM_2 and x.dim() == MAT_DIM_2:
        return torch.einsum("nm,bm->bn", matrix, x)
    if matrix.dim() == MAT_DIM_3 and x.dim() == 1:
        return torch.einsum("bnm,m->bn", matrix, x)
    return torch.einsum("bnm,bm->bn", matrix, x)


def shape_for_broadcast(
    tensor: Tensor, target_shape: Shape, dim: int | None = None
) -> Tensor:
    """Reshape tensor for broadcasting with target shape.

    Args:
        tensor: Tensor to reshape
        target_shape: Target shape for broadcasting
        dim: If specified, which dimension to preserve

    Returns
    -------
        Reshaped tensor
    """
    # Handle scalar
    if tensor.dim() == 0:
        return tensor

    # Special case for 1D tensors matching first dimension
    if (
        tensor.dim() == 1
        and len(target_shape) == 1
        and tensor.shape[0] == target_shape[0]
    ):
        return tensor.reshape(*target_shape)

    if tensor.shape == target_shape:
        return tensor

    # Build new shape
    new_shape = [1] * len(target_shape)

    if dim is not None:
        # Preserve specified dimension
        if dim < len(new_shape):
            new_shape[dim] = tensor.shape[0]
    elif tensor.dim() == 1 and tensor.shape[0] in target_shape:
        # For 1D tensors try to match the dimension with the same size
        new_shape[target_shape.index(tensor.shape[0])] = tensor.shape[0]
    else:
        # Match dimensions from the end
        offset = len(target_shape) - tensor.dim()
        for i, size in enumerate(tensor.shape):
            idx = offset + i
            if 0 <= idx < len(target_shape):
                new_shape[idx] = size

    return tensor.reshape(new_shape)


def expand_dims_like(tensor: Tensor, reference: Tensor) -> Tensor:
    """Expand tensor dimensions to match reference tensor.

    Args:
        tensor: Tensor to expand
        reference: Reference tensor for shape

    Returns
    -------
        Expanded tensor
    """
    while tensor.dim() < reference.dim():
        tensor = tensor.unsqueeze(-1)
    return tensor


def masked_fill_inf(
    tensor: Tensor, mask: Tensor, value: float = float("-inf")
) -> Tensor:
    """Fill masked positions with specified value (default -inf).

    Args:
        tensor: Input tensor
        mask: Boolean mask (True = fill)
        value: Value to fill

    Returns
    -------
        Filled tensor
    """
    return tensor.masked_fill(mask, value)


def create_causal_mask(size: int, device: Device | None = None) -> Tensor:
    """Create a causal (lower triangular) mask.

    Args:
        size: Size of the square mask
        device: Device to create mask on

    Returns
    -------
        Boolean mask of shape (size, size)
    """
    return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))


def create_padding_mask(
    lengths: Tensor, max_length: int | None = None, device: Device | None = None
) -> Tensor:
    """Create padding mask from sequence lengths.

    Args:
        lengths: Tensor of sequence lengths
        max_length: Maximum sequence length
        device: Device to create mask on

    Returns
    -------
        Boolean mask of shape (batch_size, max_length)
    """
    if max_length is None:
        max_length = int(lengths.max().item())

    batch_size = lengths.shape[0]
    mask = torch.arange(max_length, device=device or lengths.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1)
    return mask < lengths.unsqueeze(1)


@overload
def split_tensor(
    tensor: Tensor,
    split_size_or_sizes: int | None = None,
    *,
    split_size: int | None = None,
    split_sizes: Sequence[int] | None = None,
    dim: int = 0,
) -> list[Tensor]: ...


@overload
def split_tensor(
    tensor: Tensor,
    split_size_or_sizes: Sequence[int] | None = None,
    *,
    split_size: int | None = None,
    split_sizes: Sequence[int] | None = None,
    dim: int = 0,
) -> list[Tensor]: ...


def split_tensor(
    tensor: Tensor,
    split_size_or_sizes: int | Sequence[int] | None = None,
    *,
    split_size: int | None = None,
    split_sizes: Sequence[int] | None = None,
    dim: int = 0,
) -> list[Tensor]:
    """Split tensor along dimension.

    Args:
        tensor: Tensor to split
        split_size_or_sizes: Size of each chunk or list of sizes
        dim: Dimension to split along

    Returns
    -------
        List of tensor chunks
    """
    if split_size_or_sizes is not None:
        if isinstance(split_size_or_sizes, int):
            return list(torch.split(tensor, split_size_or_sizes, dim=dim))
        return list(torch.split(tensor, split_size_or_sizes, dim=dim))

    if split_size is not None:
        return list(torch.split(tensor, split_size, dim=dim))

    if split_sizes is not None:
        return list(torch.split(tensor, split_sizes, dim=dim))

    raise ValueError("split_size or split_sizes must be provided")


def concat_tensors(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors along dimension.

    Args:
        tensors: Sequence of tensors
        dim: Dimension to concatenate along

    Returns
    -------
        Concatenated tensor
    """
    return torch.cat(tensors, dim=dim)


def stack_tensors(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Stack tensors along new dimension.

    Args:
        tensors: Sequence of tensors
        dim: Dimension to stack along

    Returns
    -------
        Stacked tensor
    """
    return torch.stack(tensors, dim=dim)


class TensorStatistics:
    """Helper class for computing running statistics of tensors."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self.mean_val = None
        self.m2 = None
        self.sum = None
        self.sum_sq = None
        self.min = None
        self.max = None

    def update(self, tensor: Tensor) -> None:
        """Update statistics with new tensor.

        Args:
            tensor: New tensor to include in statistics
        """
        tensor = tensor.detach().to(dtype=torch.float64)

        for value in tensor.view(-1):
            val = float(value)
            if self.count == 0:
                self.mean_val = val
                self.sum = val
                self.sum_sq = val * val
                self.m2 = 0.0
                self.min = value
                self.max = value
                self.count = 1
                continue

            self.count += 1
            delta = val - self.mean_val
            self.mean_val += delta / self.count
            delta2 = val - self.mean_val
            self.m2 += delta * delta2
            self.sum += val
            self.sum_sq += val * val
            self.min = torch.minimum(self.min, value)
            self.max = torch.maximum(self.max, value)

    @property
    def mean(self) -> float | None:
        """Get mean value."""
        if self.count == 0:
            return None
        return float(self.mean_val)

    @property
    def std(self) -> float | None:
        """Get standard deviation."""
        if self.count == 0:
            return None
        variance = self.m2 / self.count
        variance = max(0.0, float(variance))
        return variance**0.5

    @property
    def min_value(self) -> float | None:
        """Get minimum value."""
        return self.min.item() if self.min is not None else None

    @property
    def max_value(self) -> float | None:
        """Get maximum value."""
        return self.max.item() if self.max is not None else None

    def summary(self) -> dict[str, float]:
        """Get summary statistics."""
        return {
            "count": self.count,
            "mean": self.mean or 0.0,
            "std": self.std or 0.0,
            "min": self.min_value or 0.0,
            "max": self.max_value or 0.0,
        }
