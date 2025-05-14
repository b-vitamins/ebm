"""
Contrastive Divergence (CD-k) sampler module for Restricted Boltzmann Machines (RBMs).

This module provides a class (`CDSampler`) implementing the k-step Gibbs sampling
procedure commonly used in training RBMs via Contrastive Divergence, as introduced by
Hinton (2002). It includes minimal-overhead memory management for large models.

Classes
-------
CDSampler
    Implements k-step Gibbs sampling for RBMs with adaptive memory management.

References
----------
Hinton, G.E. (2002). "Training Products of Experts by Minimizing Contrastive Divergence."
Neural Computation, 14(8), 1771-1800.

"""

from typing import Any

import torch

from ebm.rbm.model.base import BaseRBM
from ebm.rbm.sampler.base import BaseSamplerRBM, TensorType, _WrappedHookSig


class CDSampler(BaseSamplerRBM):
    """
    Contrastive Divergence sampler (CD-k) for Restricted Boltzmann Machines.

    Implements the k-step Gibbs sampling procedure as introduced by Hinton (2002).
    Starting from an initial visible state, performs alternating draws from hidden
    and visible states to generate samples from the negative phase.

    This implementation includes minimal-overhead memory management that activates
    only when needed to prevent OOM errors.

    Parameters
    ----------
    model : BaseRBM
        The RBM model from which to sample.
    k : int, default=1
        The number of Gibbs sampling steps to perform (must be >= 1).
    enable_memory_safe : bool, default=True
        Enable adaptive memory management for large batches.
    memory_threshold : float, default=0.9
        GPU memory usage threshold for triggering memory management.

    Attributes
    ----------
    k : int
        Number of Gibbs sampling steps.

    Examples
    --------
    >>> sampler = CDSampler(model=my_rbm, k=5)
    >>> v_sample = sampler.sample(v0)

    """

    def __init__(
        self,
        model: BaseRBM,
        k: int = 1,
        enable_memory_safe: bool = True,
        memory_threshold: float = 0.9,
    ) -> None:
        super().__init__(
            model, enable_memory_safe=enable_memory_safe, memory_threshold=memory_threshold
        )
        # Add type checking for k
        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k < 1:
            raise ValueError(f"Number of sampling steps k must be at least 1, got {k}")
        self.k = k

    @torch.inference_mode()
    def _sample(
        self,
        v0: TensorType,
        beta: TensorType | None = None,
        hook_fn: _WrappedHookSig | None = None,
    ) -> tuple[TensorType, TensorType]:
        """
        Perform k-step Gibbs sampling starting from an initial visible state.

        Memory management is applied only when GPU memory pressure is detected,
        minimizing performance overhead for normal operation.

        Parameters
        ----------
        v0 : TensorType
            The initial visible state tensor from which sampling begins.
        beta : Optional[TensorType], default=None
            Inverse temperature parameter (if applicable).
        hook_fn : Optional[Callable], default=None
            A callable hook function that receives (step, v_k, h_k, beta)
            at each sampling iteration.

        Returns
        -------
        Tuple[TensorType, TensorType]
            The final visible and hidden states after k Gibbs sampling steps.
        """
        sample_h = self.model.sample_h_given_v
        sample_v = self.model.sample_v_given_h
        b = beta  # local alias for efficiency

        vk = v0
        hk = sample_h(v0, beta=b)

        # Check memory pressure only for large tensors
        check_memory = self.enable_memory_safe and vk.numel() > 1e6
        under_pressure = False
        high_water_mark = None

        for step in range(self.k):
            # Standard Gibbs sampling
            vk = sample_v(hk, beta=b)
            hk = sample_h(vk, beta=b)

            # Memory management only when needed
            if check_memory and step == 0:
                under_pressure = self._check_memory_pressure()
                if under_pressure and torch.cuda.is_available():
                    total_mem = torch.cuda.get_device_properties(vk.device).total_memory
                    high_water_mark = self.memory_threshold * total_mem
            if (
                under_pressure
                and step < self.k - 1
                and (step + 1) % 5 == 0
                and torch.cuda.memory_stats()["active_bytes.all.current"] > high_water_mark
            ):
                torch.cuda.empty_cache()

            if hook_fn is not None:
                hook_fn((step, vk, hk, b))

        return vk, hk

    def get_config(self) -> dict[str, Any]:
        """
        Get configuration dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            JSON-serializable configuration dictionary.
        """
        return {"k": self.k}

    @classmethod
    def from_config(cls, config: dict[str, Any], model: BaseRBM) -> "CDSampler":
        """
        Instantiate a CDSampler from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary as returned by `get_config()`.
        model : BaseRBM
            RBM model instance.

        Returns
        -------
        CDSampler
            An initialized CDSampler instance.
        """
        return cls(model=model, k=config.get("k", 1))

    @property
    def num_steps(self) -> int:
        """
        Number of Gibbs sampling steps.

        Returns
        -------
        int
            Number of Gibbs steps (k).
        """
        return self.k

    def __repr__(self) -> str:
        """Return string representation of the CDSampler."""
        return f"CDSampler(k={self.k})"
