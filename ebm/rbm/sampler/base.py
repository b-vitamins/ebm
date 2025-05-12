"""Base classes for RBM samplers with optimized PyTorch-style hooks.

This module provides the abstract base class for all RBM sampling algorithms,
handling common functionality like hook management and metadata tracking while
allowing specific samplers to focus on their core algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import torch
import torch.nn as nn

from ebm.rbm.model.base import BaseRBM

if TYPE_CHECKING:
    import numpy.typing as npt

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

    Attributes
    ----------
    _tensor : TensorType
        The underlying sampled visible states after k steps.
    initial_state : TensorType | None
        The initial visible state (v0) if tracked.
    final_hidden : TensorType | None
        The final hidden state if tracked.
    intermediate_states : list[TensorType] | None
        List of intermediate visible states if tracked.

    Notes
    -----
    - Binary operations (+, *, etc.) return raw tensors, losing metadata
    - For operations preserving metadata, use the underlying tensor explicitly
    - CUDA tensors are automatically moved to CPU for numpy conversion
    - The class uses __slots__ for memory efficiency

    Examples
    --------
    >>> sampled = sampler.sample(v0, return_hidden=True, track_chains=True)
    >>> print(sampled.shape)  # Tensor-like behavior
    torch.Size([32, 784])
    >>> if sampled.has_metadata('final_hidden'):
    ...     hidden = sampled.final_hidden
    >>> numpy_array = np.array(sampled)  # Automatic CPU conversion
    """

    __slots__ = ("_tensor", "initial_state", "final_hidden", "intermediate_states")

    def __init__(
        self,
        tensor: TensorType,
        initial_state: TensorType | None = None,
        final_hidden: TensorType | None = None,
        intermediate_states: list[TensorType] | None = None,
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
        """
        self._tensor = tensor
        self.initial_state = initial_state
        self.final_hidden = final_hidden
        self.intermediate_states = intermediate_states

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

    def __repr__(self) -> str:
        """Return string representation showing the tensor."""
        return repr(self._tensor)

    def __str__(self) -> str:
        """Convert to string representation of the tensor."""
        return str(self._tensor)

    # Tensor protocol methods
    def __array__(self) -> npt.NDArray[Any]:
        """Convert to numpy array (handles CUDA tensors).

        Returns
        -------
        np.ndarray
            NumPy array with the tensor data, moved to CPU if necessary.
        """
        return self._tensor.detach().cpu().numpy()

    @staticmethod
    def _unpack(x: Any) -> Any:
        """Recursively unpack SampleRBM objects from nested structures.

        Parameters
        ----------
        x : Any
            The object to unpack, potentially containing nested SampleRBM instances.

        Returns
        -------
        Any
            The unpacked structure with SampleRBM instances replaced by tensors.
        """
        if isinstance(x, SampleRBM):
            return x._tensor
        if isinstance(x, dict):
            return {k: SampleRBM._unpack(v) for k, v in x.items()}
        if isinstance(x, list | tuple):
            return type(x)(SampleRBM._unpack(t) for t in x)
        if isinstance(x, set):
            return {SampleRBM._unpack(t) for t in x}
        # Strings and bytes are sequences but should not be unpacked
        if isinstance(x, str | bytes):
            return x
        # Try to iterate and unpack if it's iterable
        try:
            # Try to preserve the original type if possible
            return type(x)(SampleRBM._unpack(t) for t in x)
        except (TypeError, ValueError):
            # Not iterable or can't construct, return as-is
            return x

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
            # Check if attribute exists directly on the object
            # Only check __slots__ attributes, not delegated tensor attributes
            if attr not in self.__slots__:
                return False
            if getattr(self, attr, None) is None:
                return False
        return True


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
    """Abstract base class for RBM samplers.

    This class provides common infrastructure for all RBM sampling algorithms,
    including hook management and metadata tracking, while delegating the actual
    sampling implementation to subclasses. It follows PyTorch module conventions
    and integrates seamlessly with the PyTorch ecosystem.

    Parameters
    ----------
    model : BaseRBM
        The RBM model to sample from.

    Attributes
    ----------
    model : BaseRBM
        The RBM model to sample from.
    _sampling_hooks : dict[int, _HookEntry]
        Dictionary of registered sampling hooks.
    _hook_counter : int
        Counter for generating unique hook keys.

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

    def __init__(self, model: BaseRBM) -> None:
        """Initialize the base sampler.

        Parameters
        ----------
        model : BaseRBM
            The RBM model to sample from.
        """
        super().__init__()
        self.model = model

        # Use dict with unique keys for safe mutation during iteration
        self._sampling_hooks = {}
        self._hook_counter = 0

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
        >>> if result.has_metadata('final_hidden'):
        ...     h_final = result.final_hidden
        """
        # Fast path when no observation is needed
        if not return_hidden and not track_chains and not self._sampling_hooks:
            # Directly call the implementation without any overhead
            with torch.no_grad():  # Only disable gradients for sampling
                vk, _ = self._sample(v0, beta)
            return SampleRBM(vk)

        # Full path with potential observation
        initial_state = v0 if (return_hidden or track_chains) else None
        intermediate_states: list[TensorType] | None = [] if track_chains else None

        # Perform the actual sampling (gradients disabled only for model operations)
        vk, hk = self._sample(
            v0,
            beta,
            hook_fn=self._make_hook_wrapper(intermediate_states)
            if self._sampling_hooks or track_chains
            else None,
        )

        return SampleRBM(
            tensor=vk,
            initial_state=initial_state,
            final_hidden=hk if return_hidden else None,
            intermediate_states=intermediate_states,
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
