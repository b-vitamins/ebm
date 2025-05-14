"""Base classes for RBM samplers with optimized PyTorch-style hooks.

This module provides the abstract base class for all RBM sampling algorithms,
handling common functionality like hook management and metadata tracking while
allowing specific samplers to focus on their core algorithms.

Memory management is applied only when necessary to minimize performance impact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import singledispatch
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import DTypeLike, NDArray

from ebm.rbm.model.base import BaseRBM

# Type definitions
TensorType = torch.Tensor
# Internal hook wrapper signature (bundled)
SamplingStepBundle = tuple[int, TensorType, TensorType, TensorType | None]
_WrappedHookSig = Callable[[SamplingStepBundle], None]
# User-facing hook signatures
UnbundledSamplerHook = Callable[
    ["BaseSamplerRBM", int, TensorType, TensorType, TensorType | None], None
]
BundledSamplerHook = Callable[["BaseSamplerRBM", SamplingStepBundle], None]
SamplerHook = UnbundledSamplerHook | BundledSamplerHook


class _HookEntry(NamedTuple):
    """Internal storage for registered hooks with pre-computed dispatch style.

    Attributes
    ----------
    fn : SamplerHook
        The hook function to be called.
    style : Literal["unbundled", "bundled"]
        The calling convention for the hook.
    """

    fn: SamplerHook
    style: Literal["unbundled", "bundled"]


class SampleRBM:
    """Result of RBM sampling that acts like a tensor but carries metadata.

    This class wraps the sampled tensor and delegates tensor operations to it,
    while also providing access to optional metadata when requested. It provides
    seamless integration with PyTorch's tensor operations while preserving
    sampling-specific metadata like initial states and intermediate chains.

    Memory optimization: Uses lazy evaluation for chain states to avoid storing
    all intermediate states when not needed.

    Parameters
    ----------
    tensor : TensorType
        The final sampled visible states after k steps.
    initial_state : TensorType, optional
        The initial visible state (v0) if tracked.
    final_hidden : TensorType, optional
        The final hidden state if tracked.
    intermediate_states : list[TensorType], optional
        List of intermediate visible states if tracked.
    lazy_chain_getter : Callable[[], list[TensorType]], optional
        Function to lazily retrieve chain states when needed.

    Attributes
    ----------
    _tensor : TensorType
        The underlying sampled visible states after k steps.
    initial_state : TensorType | None
        The initial visible state (v0) if tracked.
    final_hidden : TensorType | None
        The final hidden state if tracked.
    _intermediate_states : list[TensorType] | None
        Cached intermediate states.
    _lazy_chain_getter : Callable | None
        Function to retrieve chain states lazily.

    Notes
    -----
    - Binary operations (+, *, etc.) return raw tensors, losing metadata
    - For operations preserving metadata, use the underlying tensor explicitly
    - CUDA tensors are automatically moved to CPU for numpy conversion
    - The class uses __slots__ for memory efficiency
    - Chain states are loaded lazily to save memory

    Examples
    --------
    >>> sampled = sampler.sample(v0, return_hidden=True, track_chains=True)
    >>> print(sampled.shape)  # Tensor-like behavior
    torch.Size([32, 784])
    >>> if sampled.has_hidden:
    ...     hidden = sampled.final_hidden
    >>> numpy_array = np.array(sampled)  # Automatic CPU conversion
    """

    __slots__ = (
        "_tensor",
        "initial_state",
        "final_hidden",
        "_intermediate_states",
        "_lazy_chain_getter",
    )

    def __init__(
        self,
        tensor: TensorType,
        initial_state: TensorType | None = None,
        final_hidden: TensorType | None = None,
        intermediate_states: list[TensorType] | None = None,
        lazy_chain_getter: Callable[[], list[TensorType]] | None = None,
    ) -> None:
        """Initialize a SampleRBM with tensor data and optional metadata.

        Parameters
        ----------
        tensor : TensorType
            The final sampled visible states.
        initial_state : TensorType | None, optional
            The initial visible state if tracked.
        final_hidden : TensorType | None, optional
            The final hidden state if tracked.
        intermediate_states : list[TensorType] | None, optional
            Intermediate states if chain tracking was enabled.
        lazy_chain_getter : Callable | None, optional
            Function to lazily load chain states.
        """
        self._tensor = tensor
        self.initial_state = initial_state
        self.final_hidden = final_hidden
        self._intermediate_states = intermediate_states
        self._lazy_chain_getter = lazy_chain_getter

    # Properties for convenient metadata access
    @property
    def intermediate_states(self) -> list[TensorType] | None:
        """Get intermediate states, loading lazily if needed."""
        if self._intermediate_states is None and self._lazy_chain_getter is not None:
            self._intermediate_states = self._lazy_chain_getter()
        return self._intermediate_states

    @property
    def has_initial_state(self) -> bool:
        """Check if initial state metadata is available."""
        return self.initial_state is not None

    @property
    def has_hidden(self) -> bool:
        """Check if final hidden state metadata is available."""
        return self.final_hidden is not None

    @property
    def has_chain(self) -> bool:
        """Check if intermediate chain states are available."""
        return self._intermediate_states is not None or self._lazy_chain_getter is not None

    # Delegate tensor operations to the underlying tensor
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying tensor.

        Parameters
        ----------
        name : str
            The attribute name to access.

        Returns
        -------
        Any
            The attribute value from the underlying tensor.

        Raises
        ------
        AttributeError
            If the attribute doesn't exist on the underlying tensor.
        """
        try:
            return getattr(self._tensor, name)
        except AttributeError as e:
            # Provide clearer error message
            raise AttributeError(
                f"'SampleRBM' object has no attribute '{name}' (delegated to underlying tensor)"
            ) from e

    def __getitem__(self, key: Any) -> Any:
        """Support indexing like a tensor.

        Parameters
        ----------
        key : Any
            The index or slice to access.

        Returns
        -------
        Any
            The indexed value from the underlying tensor.
        """
        return self._tensor[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Support item assignment like a tensor.

        Parameters
        ----------
        key : Any
            The index or slice to assign to.
        value : Any
            The value to assign.
        """
        self._tensor[key] = value

    def __iter__(self) -> Generator[Any, None, None]:
        """Iterate over the first dimension of the underlying tensor.

        This makes SampleRBM behave like a tensor when used in iterations.

        Yields
        ------
        Any
            Elements from the first dimension of the tensor.
        """
        yield from self._tensor

    def __len__(self) -> int:
        """Return the size of the leading dimension (number of chains)."""
        return self._tensor.shape[0]

    def __repr__(self) -> str:
        """Return string representation showing the tensor."""
        return repr(self._tensor)

    def __str__(self) -> str:
        """Convert to string representation of the tensor."""
        return str(self._tensor)

    # Tensor protocol methods
    def __array__(
        self,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,
    ) -> NDArray[Any]:
        """Convert to numpy array (handles CUDA tensors).

        Compatible with NumPy 2.0+ array protocol.

        Parameters
        ----------
        dtype : data-type, optional
            The desired data-type for the array.
        copy : bool, optional
            If True, always return a copy. Cannot be False for PyTorch tensors.

        Returns
        -------
        np.ndarray
            NumPy array with the tensor data.
        """
        base = self._tensor.detach().cpu().resolve_conj().resolve_neg().numpy()
        arr = np.asarray(base, dtype=dtype)
        return arr.copy() if copy else arr

    @staticmethod
    def _unpack(x: Any) -> Any:
        """Recursively unpack SampleRBM objects from nested structures.

        Uses singledispatch for clean type handling.

        Parameters
        ----------
        x : Any
            The object to unpack, potentially containing nested SampleRBM instances.

        Returns
        -------
        Any
            The unpacked structure with SampleRBM instances replaced by tensors.
        """
        return _unpack_impl(x)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Support torch functions with full nested structure handling.

        This method enables SampleRBM to work seamlessly with PyTorch functions
        by recursively unwrapping SampleRBM instances to their underlying tensors.

        Parameters
        ----------
        func : Callable
            The PyTorch function to call.
        types : tuple[type, ...]
            Types that triggered this function call.
        args : tuple
            Positional arguments for the function.
        kwargs : dict, optional
            Keyword arguments for the function.

        Returns
        -------
        Any
            The result of calling func with unwrapped arguments.

        Notes
        -----
        This is a custom implementation that directly unwraps SampleRBM
        objects. It doesn't use torch.overrides.handle_torch_function,
        which may have future optimizations. This approach was chosen for
        simplicity and to maintain full control over the unwrapping behavior.
        """
        if kwargs is None:
            kwargs = {}

        # Recursively unpack all SampleRBM objects
        args = tuple(cls._unpack(arg) for arg in args)
        kwargs = {k: cls._unpack(v) for k, v in kwargs.items()}

        return func(*args, **kwargs)

    # Common tensor operations (return raw tensors, metadata is lost)
    def __add__(self, other: TensorType | SampleRBM) -> TensorType:
        """Add tensors or SampleRBM objects element-wise."""
        return self._tensor + (other._tensor if isinstance(other, SampleRBM) else other)

    def __radd__(self, other: TensorType | SampleRBM) -> TensorType:
        """Right-hand addition for SampleRBM objects."""
        return (other._tensor if isinstance(other, SampleRBM) else other) + self._tensor

    def __mul__(self, other: TensorType | SampleRBM | float) -> TensorType:
        """Multiply tensors or SampleRBM objects element-wise."""
        return self._tensor * (other._tensor if isinstance(other, SampleRBM) else other)

    def __rmul__(self, other: TensorType | SampleRBM | float) -> TensorType:
        """Right-hand multiplication for SampleRBM objects."""
        return (other._tensor if isinstance(other, SampleRBM) else other) * self._tensor

    def __matmul__(self, other: TensorType | SampleRBM) -> TensorType:
        """Perform matrix multiplication with tensor or SampleRBM."""
        return self._tensor @ (other._tensor if isinstance(other, SampleRBM) else other)

    def __rmatmul__(self, other: TensorType | SampleRBM) -> TensorType:
        """Right-hand matrix multiplication for SampleRBM objects."""
        return (other._tensor if isinstance(other, SampleRBM) else other) @ self._tensor

    def __truediv__(self, other: TensorType | SampleRBM | float) -> TensorType:
        """Divide tensors or SampleRBM objects element-wise."""
        return self._tensor / (other._tensor if isinstance(other, SampleRBM) else other)

    def __rtruediv__(self, other: TensorType | SampleRBM | float) -> TensorType:
        """Right-hand division for SampleRBM objects."""
        return (other._tensor if isinstance(other, SampleRBM) else other) / self._tensor

    def __sub__(self, other: TensorType | SampleRBM) -> TensorType:
        """Subtract tensors or SampleRBM objects element-wise."""
        return self._tensor - (other._tensor if isinstance(other, SampleRBM) else other)

    def __rsub__(self, other: TensorType | SampleRBM) -> TensorType:
        """Right-hand subtraction for SampleRBM objects."""
        return (other._tensor if isinstance(other, SampleRBM) else other) - self._tensor

    # Comparison operations
    def __eq__(self, other: TensorType | SampleRBM) -> TensorType:  # type: ignore[override]
        """Compare tensors or SampleRBM objects for equality element-wise."""
        return self._tensor == (other._tensor if isinstance(other, SampleRBM) else other)

    def __ne__(self, other: TensorType | SampleRBM) -> TensorType:  # type: ignore[override]
        """Compare tensors or SampleRBM objects for inequality element-wise."""
        return self._tensor != (other._tensor if isinstance(other, SampleRBM) else other)

    def __lt__(self, other: TensorType | SampleRBM) -> TensorType:
        """Compare tensors or SampleRBM objects element-wise (less than)."""
        return self._tensor < (other._tensor if isinstance(other, SampleRBM) else other)

    def __le__(self, other: TensorType | SampleRBM) -> TensorType:
        """Compare tensors or SampleRBM objects element-wise (less than or equal)."""
        return self._tensor <= (other._tensor if isinstance(other, SampleRBM) else other)

    def __gt__(self, other: TensorType | SampleRBM) -> TensorType:
        """Compare tensors or SampleRBM objects element-wise (greater than)."""
        return self._tensor > (other._tensor if isinstance(other, SampleRBM) else other)

    def __ge__(self, other: TensorType | SampleRBM) -> TensorType:
        """Compare tensors or SampleRBM objects element-wise (greater than or equal)."""
        return self._tensor >= (other._tensor if isinstance(other, SampleRBM) else other)

    # Shape and properties
    @property
    def shape(self) -> torch.Size:
        """Get the shape of the underlying tensor."""
        return self._tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the underlying tensor."""
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the underlying tensor."""
        return self._tensor.device

    @property
    def samples(self) -> TensorType:
        """Get the underlying tensor of samples."""
        return self._tensor

    def to_tensor(self) -> TensorType:
        """Explicitly convert to a regular tensor.

        Returns
        -------
        TensorType
            The underlying tensor without metadata.
        """
        return self._tensor

    def has_metadata(self, *attrs: str) -> bool:
        """Check if specific metadata attributes are available.

        Parameters
        ----------
        *attrs : str
            The metadata attribute names to check.

        Returns
        -------
        bool
            True if all specified attributes exist and are not None.

        Examples
        --------
        >>> sample = sampler.sample(v0, return_hidden=True)
        >>> sample.has_metadata('final_hidden')
        True
        >>> sample.has_metadata('intermediate_states')
        False
        """
        for attr in attrs:
            if attr == "initial_state":
                if self.initial_state is None:
                    return False
            elif attr == "final_hidden":
                if self.final_hidden is None:
                    return False
            elif attr == "intermediate_states":
                if self.intermediate_states is None:
                    return False
            else:
                return False
        return True


# Singledispatch implementation for unpacking
@singledispatch
def _unpack_impl(x: Any) -> Any:
    """Has default unpacking behavior."""
    return x


@_unpack_impl.register(SampleRBM)
def _(x: SampleRBM) -> TensorType:
    """Unpack SampleRBM to its underlying tensor."""
    return x._tensor


@_unpack_impl.register(dict)
def _(x: dict[Any, Any]) -> dict[Any, Any]:
    """Unpack dictionary recursively."""
    return {k: _unpack_impl(v) for k, v in x.items()}


@_unpack_impl.register(list)
def _(x: list[Any]) -> list[Any]:
    """Unpack list recursively."""
    return [_unpack_impl(item) for item in x]


@_unpack_impl.register(tuple)
def _(x: tuple[Any, ...]) -> tuple[Any, ...]:
    """Unpack tuple recursively."""
    return tuple(_unpack_impl(item) for item in x)


@_unpack_impl.register(set)
def _(x: set[Any]) -> set[Any]:
    """Unpack set recursively."""
    return {_unpack_impl(item) for item in x}


class RemovableHandle:
    """Handle for removing registered hooks, mimicking PyTorch's behavior.

    This class implements safe double-removal (silently ignores) and works
    correctly when hooks remove themselves during iteration. It follows the
    same patterns as PyTorch's hook handles.

    Parameters
    ----------
    hooks_dict : dict[int, _HookEntry]
        The dictionary containing hooks, keyed by unique integers.
    key : int
        The unique key for the hook in the dictionary.

    Attributes
    ----------
    hooks_dict : dict[int, _HookEntry]
        Direct reference to the hooks dictionary.
    key : int
        The unique key for this hook.
    _removed : bool
        Flag indicating if the hook has been removed.

    Examples
    --------
    >>> sampler = ConcreteSampler(model)
    >>> handle = sampler.register_sampling_hook(my_hook)
    >>> # ... use the sampler ...
    >>> handle.remove()  # Remove the hook
    >>> handle.remove()  # Safe to call again (no-op)
    """

    def __init__(self, hooks_dict: dict[int, _HookEntry], key: int) -> None:
        """Initialize a RemovableHandle.

        Parameters
        ----------
        hooks_dict : dict[int, _HookEntry]
            The dictionary containing registered hooks.
        key : int
            The unique key identifying the hook to be removed.
        """
        self.hooks_dict = hooks_dict  # Direct reference, not weak reference
        self.key = key
        self._removed = False

    def remove(self) -> None:
        """Remove the hook from the dictionary.

        This method silently ignores double-removal attempts, following
        PyTorch conventions. Once a hook is removed, calling remove()
        again has no effect.
        """
        if self._removed:
            return  # PyTorch style: silent on double-remove

        if self.key in self.hooks_dict:
            del self.hooks_dict[self.key]
        self._removed = True

    def __enter__(self) -> RemovableHandle:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - removes the hook."""
        self.remove()


class BaseSamplerRBM(nn.Module, ABC):
    """Abstract base class for RBM samplers with memory-aware operations.

    This class provides common infrastructure for all RBM sampling algorithms,
    including hook management and metadata tracking, while delegating the actual
    sampling implementation to subclasses. It follows PyTorch module conventions
    and integrates seamlessly with the PyTorch ecosystem.

    Memory management is applied lazily only when needed to minimize performance impact.

    Parameters
    ----------
    model : BaseRBM
        The RBM model to sample from.
    enable_memory_safe : bool, default=True
        Whether to enable automatic memory management for large batches.
    memory_threshold : float, default=0.9
        GPU memory usage threshold for triggering memory management.
    batch_split_threshold : int, default=10000
        Batch size threshold for automatic splitting.

    Attributes
    ----------
    model : BaseRBM
        The RBM model to sample from.
    _sampling_hooks : dict[int, _HookEntry]
        Dictionary of registered sampling hooks.
    _hook_counter : int
        Counter for generating unique hook keys.
    enable_memory_safe : bool
        Whether memory safety features are enabled.
    memory_threshold : float
        GPU memory threshold for safety features.
    batch_split_threshold : int
        Batch size threshold for splitting.

    Notes
    -----
    The hook system follows PyTorch conventions:

    - Hooks run with gradients enabled (if not disabled globally)
    - Exceptions in hooks bubble up (not swallowed)
    - Safe for hooks to remove themselves during iteration
    - Double-removal of handles is silently ignored
    - Explicit registration methods for different hook styles
    - Not thread-safe: Hook mutations should be synchronized if used
      in multi-threaded contexts (same as PyTorch)

    Examples
    --------
    >>> class CDSampler(BaseSamplerRBM):
    ...     def _sample(self, v0, beta=None, hook_fn=None):
    ...         # Implement Contrastive Divergence sampling
    ...         pass
    >>>
    >>> sampler = CDSampler(model)
    >>> samples = sampler(initial_visible_states)
    """

    model: BaseRBM
    _sampling_hooks: dict[int, _HookEntry]
    _hook_counter: int

    def __init__(
        self,
        model: BaseRBM,
        enable_memory_safe: bool = True,
        memory_threshold: float = 0.9,
        batch_split_threshold: int = 10000,
    ) -> None:
        """Initialize the base sampler.

        Parameters
        ----------
        model : BaseRBM
            The RBM model to sample from.
        enable_memory_safe : bool, default=True
            Enable memory safety features.
        memory_threshold : float, default=0.9
            GPU memory usage threshold.
        batch_split_threshold : int, default=10000
            Batch size threshold for splitting.
        """
        super().__init__()
        self.model = model
        self.enable_memory_safe = enable_memory_safe
        self.memory_threshold = memory_threshold
        self.batch_split_threshold = batch_split_threshold

        # Use dict with unique keys for safe mutation during iteration
        self._sampling_hooks = {}
        self._hook_counter = 0

    def _check_memory_pressure(self) -> bool:
        """Check if GPU memory pressure is high.

        Returns
        -------
        bool
            True if memory usage exceeds threshold.
        """
        if not self.enable_memory_safe or not torch.cuda.is_available():
            return False

        try:
            torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory

            # Use reserved memory for more accurate pressure detection
            usage_ratio = reserved / total
            return usage_ratio > self.memory_threshold
        except Exception:
            return False

    def _adaptive_batch_split(
        self,
        v0: TensorType,
        beta: TensorType | None = None,
        hook_fn: _WrappedHookSig | None = None,
    ) -> tuple[TensorType, TensorType]:
        """Adaptively split batches only when needed.

        This method checks for memory pressure and batch size before
        deciding whether to split the batch processing.
        """
        batch_size = v0.size(0)

        # Fast path: Small batch or memory safety disabled
        if not self.enable_memory_safe or batch_size < self.batch_split_threshold:
            return self._sample(v0, beta, hook_fn)

        # Check memory pressure
        if not self._check_memory_pressure():
            return self._sample(v0, beta, hook_fn)

        # Memory pressure detected - split batch
        split_size = max(batch_size // 2, 100)  # At least 100 samples per split

        v_results = []
        h_results = []

        for i in range(0, batch_size, split_size):
            end_idx = min(i + split_size, batch_size)
            v_batch = v0[i:end_idx]
            vk, hk = self._sample(v_batch, beta, hook_fn)

            v_results.append(vk)
            h_results.append(hk)

            # Only clear cache if still under pressure
            if torch.cuda.is_available() and self._check_memory_pressure():
                torch.cuda.empty_cache()

        # Concatenate results
        return torch.cat(v_results, dim=0), torch.cat(h_results, dim=0)

    def sample(
        self,
        v0: TensorType,
        beta: TensorType | None = None,
        return_hidden: bool = False,
        track_chains: bool = False,
    ) -> SampleRBM:
        """Perform sampling starting from v0.

        This method handles the common infrastructure and delegates to
        _sample for the actual sampling algorithm. Hooks run with gradients
        enabled, while the core sampling runs with gradients disabled.

        Memory management is applied adaptively based on batch size and
        GPU memory pressure to minimize performance impact.

        Parameters
        ----------
        v0 : TensorType
            Initial visible state with shape (batch_size, visible_dim)
            or (batch_size, num_replicas, visible_dim) when using parallel tempering.
        beta : TensorType, optional
            Inverse temperature for sampling. Can be a scalar or
            tensor with shape (num_replicas,) for parallel tempering.
        return_hidden : bool, default=False
            If True, includes the final hidden state in result.
        track_chains : bool, default=False
            If True, tracks all intermediate states.
            Warning: This clones states at each step, which can use
            significant memory for long chains or large models.

        Returns
        -------
        SampleRBM
            Result object that acts like a tensor but may contain metadata.

        Notes
        -----
        - The input tensor v0 is never modified during sampling.
        - When using parallel tempering with track_chains=True, the tracked
          states will include the replica dimension, capturing replica-permuted
          states rather than physical chain trajectories.
        - States passed to hooks are already detached; re-enable requires_grad
          if gradients are needed in hooks.

        Examples
        --------
        >>> v0 = torch.randn(32, 784)  # 32 samples, 784 visible units
        >>> result = sampler.sample(v0, return_hidden=True)
        >>> print(result.shape)
        torch.Size([32, 784])
        >>> if result.has_hidden:
        ...     h_final = result.final_hidden
        """
        # Fast path when no observation is needed
        if not return_hidden and not track_chains and not self._sampling_hooks:
            # Directly call the implementation without any overhead
            with torch.no_grad():  # Only disable gradients for sampling
                vk, _ = self._adaptive_batch_split(v0, beta)
            return SampleRBM(vk)

        # Full path with potential observation
        initial_state = v0 if (return_hidden or track_chains) else None
        intermediate_states: list[TensorType] | None = [] if track_chains else None

        # Perform the actual sampling (gradients disabled only for model operations)
        vk, hk = self._adaptive_batch_split(
            v0,
            beta,
            hook_fn=self._make_hook_wrapper(intermediate_states)
            if self._sampling_hooks or track_chains
            else None,
        )

        # Use lazy chain getter for memory efficiency
        lazy_chain_getter = None
        if track_chains and intermediate_states:
            # Only store chains lazily if they're large
            total_elements = sum(t.numel() for t in intermediate_states)
            if total_elements > 1e6:  # More than 1M elements
                chains = intermediate_states

                def lazy_chain_getter() -> list[TensorType]:
                    return chains

                intermediate_states = None

        return SampleRBM(
            tensor=vk,
            initial_state=initial_state,
            final_hidden=hk if return_hidden else None,
            intermediate_states=intermediate_states,
            lazy_chain_getter=lazy_chain_getter,
        )

    @abstractmethod
    def _sample(
        self,
        v0: TensorType,
        beta: TensorType | None = None,
        hook_fn: _WrappedHookSig | None = None,
    ) -> tuple[TensorType, TensorType]:
        """Perform the actual sampling algorithm.

        This method must be implemented by subclasses to define the specific
        sampling algorithm (e.g., CD, PCD, etc.).

        Parameters
        ----------
        v0 : TensorType
            Initial visible state.
        beta : TensorType, optional
            Inverse temperature.
        hook_fn : _WrappedHookSig, optional
            Function to call at each step with signature
            hook_fn((step, vk, hk, beta))

        Returns
        -------
        tuple[TensorType, TensorType]
            Tuple of (final_visible_state, final_hidden_state)
        """
        ...

    def _make_hook_wrapper(self, intermediate_states: list[TensorType] | None) -> _WrappedHookSig:
        """Create a hook wrapper that handles both user hooks and chain tracking.

        Parameters
        ----------
        intermediate_states : list[TensorType] | None
            List to append intermediate states to, if tracking chains.

        Returns
        -------
        _WrappedHookSig
            The wrapper function that calls all registered hooks.
        """

        def hook_wrapper(bundle: SamplingStepBundle) -> None:
            step, vk, hk, beta = bundle

            # Track intermediate states if requested (detach to avoid gradient accumulation)
            if intermediate_states is not None:
                intermediate_states.append(vk.detach().clone())

            # Call user hooks (iterate over a copy to handle self-removal)
            for entry in tuple(self._sampling_hooks.values()):
                if entry.style == "unbundled":
                    # Cast to proper type for unbundled hooks
                    unbundled_hook = cast(UnbundledSamplerHook, entry.fn)
                    unbundled_hook(self, step, vk, hk, beta)
                else:  # bundled
                    # Cast to proper type for bundled hooks
                    bundled_hook = cast(BundledSamplerHook, entry.fn)
                    bundled_hook(self, bundle)

        return hook_wrapper

    def register_sampling_hook(self, hook: UnbundledSamplerHook) -> RemovableHandle:
        """Register a hook with PyTorch-style unbundled signature.

        The hook should have signature:
            hook(sampler, step, v_current, h_current, beta)

        The hook runs with gradients enabled (unless globally disabled) and
        exceptions are not caught (they bubble up to the caller).

        Parameters
        ----------
        hook : UnbundledSamplerHook
            Callable to be invoked at each sampling step.

        Returns
        -------
        RemovableHandle
            Handle that can be used to remove the hook.

        Examples
        --------
        >>> def monitor_hook(sampler, step, v, h, beta):
        ...     print(f"Step {step}: v_norm={v.norm():.3f}")
        >>>
        >>> handle = sampler.register_sampling_hook(monitor_hook)
        >>> samples = sampler.sample(v0)
        >>> handle.remove()
        """
        key = self._hook_counter
        self._hook_counter += 1
        self._sampling_hooks[key] = _HookEntry(hook, "unbundled")
        return RemovableHandle(self._sampling_hooks, key)

    def register_sampling_hook_bundled(self, hook: BundledSamplerHook) -> RemovableHandle:
        """Register a hook with bundled signature.

        The hook should have signature:
            hook(sampler, (step, v_current, h_current, beta))

        The hook runs with gradients enabled (unless globally disabled) and
        exceptions are not caught (they bubble up to the caller).

        Parameters
        ----------
        hook : BundledSamplerHook
            Callable to be invoked at each sampling step.

        Returns
        -------
        RemovableHandle
            Handle that can be used to remove the hook.

        Examples
        --------
        >>> def bundled_hook(sampler, bundle):
        ...     step, v, h, beta = bundle
        ...     print(f"Step {step}")
        >>>
        >>> handle = sampler.register_sampling_hook_bundled(bundled_hook)
        >>> samples = sampler.sample(v0)
        >>> handle.remove()
        """
        key = self._hook_counter
        self._hook_counter += 1
        self._sampling_hooks[key] = _HookEntry(hook, "bundled")
        return RemovableHandle(self._sampling_hooks, key)

    @contextmanager
    def temporary_hook(
        self, hook: SamplerHook, style: Literal["unbundled", "bundled"] = "unbundled"
    ) -> Generator[RemovableHandle, None, None]:
        """Context manager for temporary hook registration.

        This is a convenient way to register a hook that will be automatically
        removed when the context exits.

        Parameters
        ----------
        hook : SamplerHook
            The hook function to register temporarily.
        style : {"unbundled", "bundled"}, default="unbundled"
            The hook style to use.

        Yields
        ------
        RemovableHandle
            Handle to the registered hook.

        Examples
        --------
        >>> def monitor_hook(sampler, step, v, h, beta):
        ...     print(f"Step {step}")
        >>>
        >>> sampler = CDSampler(model)
        >>> with sampler.temporary_hook(monitor_hook) as handle:
        ...     samples = sampler.sample(v0)
        >>> # Hook is automatically removed here
        """
        if style == "unbundled":
            handle = self.register_sampling_hook(cast(UnbundledSamplerHook, hook))
        else:
            handle = self.register_sampling_hook_bundled(cast(BundledSamplerHook, hook))
        try:
            yield handle
        finally:
            handle.remove()

    def forward(
        self,
        v0: TensorType,
        beta: TensorType | None = None,
    ) -> SampleRBM:
        """Alias for sample() to make the sampler callable.

        This allows the sampler to be used like a function:
        ``samples = sampler(v0)``

        Parameters
        ----------
        v0 : TensorType
            Initial visible state.
        beta : TensorType, optional
            Inverse temperature.

        Returns
        -------
        SampleRBM
            Sampled states, potentially with metadata.
        """
        return self.sample(v0, beta=beta)

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Get configuration parameters for serialization.

        This should be implemented by subclasses to return their specific
        configuration parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing configuration parameters.
        """
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation of the sampler."""
        ...
