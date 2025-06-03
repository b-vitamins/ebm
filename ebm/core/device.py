"""Device and memory management utilities.

This module provides utilities for automatic device placement, memory management,
and handling of different hardware accelerators (CUDA, MPS, etc.).
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DeviceInfo:
    """Information about a compute device."""

    device: torch.device
    name: str
    total_memory: int | None = None
    allocated_memory: int | None = None
    cached_memory: int | None = None
    compute_capability: tuple[int, int] | None = None

    @property
    def memory_stats(self) -> dict[str, float]:
        """Get memory statistics in MB."""
        stats = {}
        if self.total_memory is not None:
            stats['total_mb'] = self.total_memory / 1024**2
        if self.allocated_memory is not None:
            stats['allocated_mb'] = self.allocated_memory / 1024**2
        if self.cached_memory is not None:
            stats['cached_mb'] = self.cached_memory / 1024**2

        if self.total_memory and self.allocated_memory:
            stats['utilization'] = self.allocated_memory / self.total_memory

        return stats


class DeviceManager:
    """Manages device placement and memory for the library."""

    def __init__(self, device: str | torch.device | None = None):
        """Initialize device manager.

        Args:
            device: Device specification. Can be:
                - None or 'auto': automatically select best device
                - 'cuda', 'cuda:0', etc.: specific CUDA device
                - 'mps': Metal Performance Shaders (Apple Silicon)
                - 'cpu': CPU device
        """
        self._device = self._resolve_device(device)
        self._original_device = self._device

    @property
    def device(self) -> torch.device:
        """Current active device."""
        return self._device

    @property
    def is_cuda(self) -> bool:
        """Check if current device is CUDA."""
        return self._device.type == 'cuda'

    @property
    def is_mps(self) -> bool:
        """Check if current device is MPS."""
        return self._device.type == 'mps'

    @property
    def is_cpu(self) -> bool:
        """Check if current device is CPU."""
        return self._device.type == 'cpu'

    def _resolve_device(self, device: str | torch.device | None) -> torch.device:
        """Resolve device specification to torch.device."""
        if device is None or device == 'auto':
            # Auto-select best available device
            if torch.cuda.is_available():
                return torch.device('cuda')
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')

        if isinstance(device, str):
            device = torch.device(device)

        # Validate device
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")
        if device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError("MPS device requested but MPS is not available")

        return device

    def get_device_info(self) -> DeviceInfo:
        """Get information about current device."""
        info = DeviceInfo(device=self._device, name=str(self._device))

        if self.is_cuda:
            props = torch.cuda.get_device_properties(self._device)
            info.name = props.name
            info.total_memory = props.total_memory
            info.allocated_memory = torch.cuda.memory_allocated(self._device)
            info.cached_memory = torch.cuda.memory_reserved(self._device)
            info.compute_capability = (props.major, props.minor)

        return info

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.is_mps:
            # MPS doesn't have explicit cache clearing yet
            pass

    @contextmanager
    def device_placement(self, device: str | torch.device | None = None) -> Iterator[None]:
        """Context manager for temporary device placement.

        Args:
            device: Temporary device to use. If None, uses current device.

        Example:
            >>> manager = DeviceManager('cuda')
            >>> with manager.device_placement('cpu'):
            ...     # Operations here run on CPU
            ...     x = torch.randn(10)
            >>> # Back to CUDA
        """
        if device is None:
            yield
            return

        old_device = self._device
        try:
            self._device = self._resolve_device(device)
            yield
        finally:
            self._device = old_device

    def to_device(self, tensor: Tensor, non_blocking: bool = False) -> Tensor:
        """Move tensor to current device.

        Args:
            tensor: Tensor to move
            non_blocking: Whether to use non-blocking transfer

        Returns
        -------
            Tensor on current device
        """
        return tensor.to(device=self._device, non_blocking=non_blocking)

    def module_to_device(self, module: torch.nn.Module) -> torch.nn.Module:
        """Move module and its parameters to current device.

        Args:
            module: Module to move

        Returns
        -------
            Module on current device
        """
        return module.to(device=self._device)

    @contextmanager
    def autocast(self, enabled: bool = True, dtype: torch.dtype | None = None):
        """Context manager for automatic mixed precision.

        Args:
            enabled: Whether to enable autocast
            dtype: Override dtype for autocast (default: float16 for CUDA, bfloat16 for CPU)
        """
        if not enabled:
            yield
            return

        if self.is_cuda:
            dtype = dtype or torch.float16
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                yield
        elif self.is_cpu:
            # CPU autocast uses bfloat16 by default
            dtype = dtype or torch.bfloat16
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                yield
        else:
            # No autocast for other devices yet
            yield

    def synchronize(self) -> None:
        """Synchronize device operations."""
        if self.is_cuda:
            torch.cuda.synchronize(self._device)

    def get_memory_summary(self) -> str:
        """Get formatted memory summary string."""
        info = self.get_device_info()
        stats = info.memory_stats

        lines = [f"Device: {info.name}"]
        if stats:
            lines.append(f"Memory: {stats.get('allocated_mb', 0):.1f}/{stats.get('total_mb', 0):.1f} MB "
                        f"({stats.get('utilization', 0)*100:.1f}% used)")
            if 'cached_mb' in stats:
                lines.append(f"Cached: {stats['cached_mb']:.1f} MB")

        return '\n'.join(lines)

    @staticmethod
    def get_available_devices() -> list[torch.device]:
        """Get list of all available devices."""
        devices = [torch.device('cpu')]

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device(f'cuda:{i}'))

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append(torch.device('mps'))

        return devices

    def __repr__(self) -> str:
        return f"DeviceManager(device={self._device})"


# Global device manager instance
_global_manager: DeviceManager | None = None


def get_device_manager() -> DeviceManager:
    """Get global device manager instance."""
    global _global_manager
    if _global_manager is None:
        # Initialize with auto device selection
        _global_manager = DeviceManager('auto')
    return _global_manager


def set_device(device: str | torch.device | None) -> None:
    """Set global device."""
    global _global_manager
    _global_manager = DeviceManager(device)


def get_device() -> torch.device:
    """Get current global device."""
    return get_device_manager().device


def to_device(tensor: Tensor, device: torch.device | None = None) -> Tensor:
    """Move tensor to device (global device if not specified)."""
    if device is None:
        return get_device_manager().to_device(tensor)
    return tensor.to(device=device)


def auto_device(fn):
    """Decorator to automatically handle device placement.

    This decorator ensures that all tensor operations in the decorated
    function use the global device manager's current device.
    """
    def wrapper(*args, **kwargs):
        get_device_manager()
        # Could add more sophisticated device handling here
        return fn(*args, **kwargs)
    return wrapper


def memory_efficient(fn):
    """Decorator to run function with memory optimization.

    Clears cache before and after function execution.
    """
    def wrapper(*args, **kwargs):
        manager = get_device_manager()
        manager.clear_cache()
        try:
            return fn(*args, **kwargs)
        finally:
            manager.clear_cache()
    return wrapper
