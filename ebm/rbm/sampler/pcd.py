"""
Persistent Contrastive Divergence (PCD) sampler module for Restricted Boltzmann Machines.

This module implements PCD sampling, which maintains persistent Markov chains that are
not reset between parameter updates. This allows better exploration of the model's
distribution compared to standard Contrastive Divergence.

Classes
-------
PCDSampler
    Implements persistent k-step Gibbs sampling for RBMs.

References
----------
Tieleman, T. (2008). "Training restricted Boltzmann machines using approximations to
the likelihood gradient." ICML '08: Proceedings of the 25th international conference
on Machine learning, pp. 1064-1071.
"""

from typing import Any, cast

import torch

from ebm.rbm.model.base import BaseRBM
from ebm.rbm.sampler.base import TensorType, _WrappedHookSig
from ebm.rbm.sampler.cd import CDSampler


class PCDSampler(CDSampler):
    """
    Persistent Contrastive Divergence sampler (PCD) for Restricted Boltzmann Machines.

    Implements PCD sampling which maintains persistent chains that evolve across
    sampling steps, unlike standard CD which resets chains from data. This leads
    to better mixing and exploration of the model's distribution.

    Parameters
    ----------
    model : BaseRBM
        The RBM model from which to sample.
    k : int, default=1
        The number of Gibbs sampling steps to perform (must be >= 1).
    num_chains : int, default=100
        Number of persistent chains to maintain.
    init_method : {"uniform", "data", "model"}, default="uniform"
        Initialization method for the persistent chains:
        - "uniform": Random uniform initialization
        - "data": Initialize from the first batch of data
        - "model": Initialize by sampling from the model
    enable_memory_safe : bool, default=True
        Enable adaptive memory management for large batches.
    memory_threshold : float, default=0.9
        GPU memory usage threshold for triggering memory management.

    Attributes
    ----------
    num_chains : int
        Number of persistent chains.
    init_method : str
        Initialization method for chains.
    persistent_chains : torch.Tensor | None
        The persistent chain states with shape (num_chains, ..., visible_dim).
        The ... allows for arbitrary dimensions including replicas for parallel tempering.
    chains_initialized : bool
        Whether the persistent chains have been initialized.

    Examples
    --------
    >>> sampler = PCDSampler(model=my_rbm, k=1, num_chains=100)
    >>> # First call initializes chains
    >>> v_sample = sampler.sample(v_data)
    >>> # Subsequent calls continue from persistent state
    >>> v_sample2 = sampler.sample(v_data2)

    Notes
    -----
    The persistent chains maintain shape (num_chains, ..., visible_dim) where
    ... can include replica dimensions for parallel tempering. The sampler
    handles arbitrary dimensional patterns between batch and visible dimensions.
    """

    def __init__(
        self,
        model: BaseRBM,
        k: int = 1,
        num_chains: int = 100,
        init_method: str = "uniform",
        enable_memory_safe: bool = True,
        memory_threshold: float = 0.9,
    ) -> None:
        """Initialize the PCD sampler.

        Parameters
        ----------
        model : BaseRBM
            The RBM model to sample from.
        k : int, default=1
            Number of Gibbs sampling steps.
        num_chains : int, default=100
            Number of persistent chains to maintain.
        init_method : str, default="uniform"
            Chain initialization method.
        enable_memory_safe : bool, default=True
            Enable memory safety features.
        memory_threshold : float, default=0.9
            GPU memory usage threshold.
        """
        super().__init__(
            model, k=k, enable_memory_safe=enable_memory_safe, memory_threshold=memory_threshold
        )

        # Validate parameters
        if num_chains < 1:
            raise ValueError(f"Number of chains must be at least 1, got {num_chains}")
        if init_method not in ["uniform", "data", "model"]:
            raise ValueError(
                f"init_method must be one of ['uniform', 'data', 'model'], got {init_method}"
            )

        self.num_chains = num_chains
        self.init_method = init_method
        self.persistent_chains: torch.Tensor | None = None
        self.chains_initialized = False

    def _initialize_chains(self, v0: TensorType, beta: TensorType | None = None) -> None:
        """Initialize persistent chains based on the first data batch.

        Parameters
        ----------
        v0 : TensorType
            Initial visible state used to determine chain shape and initialization.
            Shape: (batch_size, ..., visible_dim) where ... can be replica dimensions.
        beta : TensorType | None
            Temperature parameter for initialization (only used for "model" init).
        """
        # Extract shape pattern: we need (num_chains, ..., visible_dim)
        # where ... are any dimensions between batch and visible
        if v0.ndim < 2:
            raise ValueError(f"Input must have at least 2 dimensions, got {v0.ndim}")

        # Create shape for persistent chains
        chain_shape = list(v0.shape)
        chain_shape[0] = max(self.num_chains, v0.shape[0])

        device = v0.device
        dtype = v0.dtype

        if self.init_method == "uniform":
            # Random uniform initialization
            self.persistent_chains = torch.rand(chain_shape, device=device, dtype=dtype)
        elif self.init_method == "data":
            # Initialize from data by repeating/cycling through v0
            # Do NOT run sampling steps - just copy the data
            if v0.shape[0] >= self.num_chains:
                # Take first num_chains samples
                self.persistent_chains = v0[: self.num_chains].clone()
            else:
                # Repeat data to fill chains
                repeats = (self.num_chains + v0.shape[0] - 1) // v0.shape[0]
                repeated = v0.repeat(repeats, *([1] * (v0.ndim - 1)))
                self.persistent_chains = repeated[: self.num_chains].clone()
        elif self.init_method == "model":
            # Initialize by sampling from the model
            # Start with random states and run k Gibbs steps
            random_v = torch.rand(chain_shape, device=device, dtype=dtype)
            with torch.no_grad():
                h_sample = self.model.sample_h_given_v(random_v, beta=beta)
                for _ in range(self.k):
                    v_sample = self.model.sample_v_given_h(h_sample, beta=beta)
                    h_sample = self.model.sample_h_given_v(v_sample, beta=beta)
                self.persistent_chains = v_sample

        self.chains_initialized = True

    @torch.no_grad()
    def _sample(
        self,
        v0: TensorType,
        beta: TensorType | None = None,
        hook_fn: _WrappedHookSig | None = None,
    ) -> tuple[TensorType, TensorType]:
        """
        Perform k-step Gibbs sampling using persistent chains.

        Unlike standard CD, this continues from the persistent chain states
        rather than starting from the data. The chains are updated in-place.

        Parameters
        ----------
        v0 : TensorType
            The data batch (used for shape inference on first call).
            Shape: (batch_size, ..., visible_dim)
        beta : Optional[TensorType], default=None
            Inverse temperature parameter for parallel tempering.
        hook_fn : Optional[Callable], default=None
            Hook function called at each sampling iteration.

        Returns
        -------
        Tuple[TensorType, TensorType]
            The sampled visible and hidden states from the persistent chains.
            Both have shape (num_chains, ..., dimensionality).
        """
        # Initialize chains on first call
        if not self.chains_initialized:
            self._initialize_chains(v0, beta=beta)
            # For data initialization, we need to return the chains immediately
            # without running sampling steps, to match the expected behavior
            if self.init_method == "data":
                # Just return the initialized chains without sampling
                chains = cast(TensorType, self.persistent_chains)
                hk = self.model.sample_h_given_v(chains, beta=beta)
                return chains, hk

        # Make sure the stored chains match the requested layout
        chains = cast(TensorType, self.persistent_chains)
        expected_shape = (self.num_chains, *v0.shape[1:])
        if chains.shape != expected_shape:
            # Rebuild chains so subsequent operations broadcast correctly
            self._initialize_chains(v0, beta=beta)
            chains = cast(TensorType, self.persistent_chains)

        # Start from persistent chains
        vk: TensorType = chains
        sample_h = self.model.sample_h_given_v
        sample_v = self.model.sample_v_given_h
        b = beta

        # Check memory pressure only for large tensors
        check_memory = self.enable_memory_safe and vk.numel() > 1e6
        under_pressure = False
        high_water_mark = None

        # Run k steps of Gibbs sampling
        for step in range(self.k):
            hk = sample_h(vk, beta=b)
            vk = sample_v(hk, beta=b)

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

        # Update persistent chains
        self.persistent_chains = vk.detach()

        # Return the state of the chains
        return vk, hk

    def reset_chains(self) -> None:
        """Reset the persistent chains to uninitialized state.

        This forces re-initialization on the next sampling call.
        Useful when the model parameters have changed significantly.
        """
        self.persistent_chains = None
        self.chains_initialized = False

    def get_chains(self) -> torch.Tensor | None:
        """Get the current state of persistent chains.

        Returns
        -------
        torch.Tensor | None
            Current chain states or None if not initialized.
        """
        return self.persistent_chains

    def set_chains(self, chains: torch.Tensor) -> None:
        """Set the persistent chains to specific values.

        Parameters
        ----------
        chains : torch.Tensor
            New chain states with shape (num_chains, ..., visible_dim).
        """
        if chains.shape[0] != self.num_chains:
            raise ValueError(f"First dimension must be {self.num_chains}, got {chains.shape[0]}")
        self.persistent_chains = chains.detach()
        self.chains_initialized = True

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary including PCD-specific parameters.
        """
        config = super().get_config()
        config.update(
            {
                "num_chains": self.num_chains,
                "init_method": self.init_method,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any], model: BaseRBM) -> "PCDSampler":
        """Create a PCDSampler from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary as returned by `get_config()`.
        model : BaseRBM
            RBM model instance.

        Returns
        -------
        PCDSampler
            An initialized PCDSampler instance.
        """
        return cls(
            model=model,
            k=config.get("k", 1),
            num_chains=config.get("num_chains", 100),
            init_method=config.get("init_method", "uniform"),
        )

    def __repr__(self) -> str:
        """Return string representation of the PCDSampler."""
        return (
            f"PCDSampler(k={self.k}, num_chains={self.num_chains}, "
            f"init_method='{self.init_method}', initialized={self.chains_initialized})"
        )
