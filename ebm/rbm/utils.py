"""Utilities for Restricted Boltzmann Machine (RBM) models."""

from __future__ import annotations

import torch


def shape_beta(beta: float | torch.Tensor | None, ref: torch.Tensor) -> float | torch.Tensor | None:
    """Broadcast beta to match the replica dimension of ref.

    Parameters
    ----------
    beta : float, torch.Tensor, or None
        The beta value to broadcast. Can be a scalar, tensor, or None.
    ref : torch.Tensor
        The reference tensor with the target shape.

    Returns
    -------
    float, torch.Tensor, or None
        The appropriately shaped beta value, or None if input beta was None.

    Notes
    -----
    - If beta is None, None is returned (early exit).
    - If beta is a scalar or 0-D tensor, it is returned unchanged (fast path).
    - Otherwise, beta is reshaped to broadcast correctly against the reference tensor,
      preserving the replica dimension.
    - The result will have the same dtype and device as ref.
    """
    if beta is None:
        return None

    if not torch.is_tensor(beta):
        return beta

    if beta.ndim == 0:
        return beta

    needed_dims = ref.ndim - 2  # skip (B, K)
    view = beta.view(1, -1, *([1] * needed_dims))
    return view.to(dtype=ref.dtype, device=ref.device)
