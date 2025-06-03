"""Unit tests for device management."""

from unittest.mock import patch

import pytest
import torch

from ebm.core.device import (
    DeviceInfo,
    DeviceManager,
    auto_device,
    get_device,
    get_device_manager,
    memory_efficient,
    set_device,
    to_device,
)


class TestDeviceInfo:
    """Test DeviceInfo dataclass."""

    def test_device_info_creation(self) -> None:
        """Test creating device info."""
        info = DeviceInfo(
            device=torch.device("cuda:0"),
            name="NVIDIA GeForce RTX 3090",
            total_memory=25_000_000_000,
            allocated_memory=5_000_000_000,
            cached_memory=1_000_000_000,
            compute_capability=(8, 6),
        )

        assert info.device.type == "cuda"
        assert info.device.index == 0
        assert info.name == "NVIDIA GeForce RTX 3090"
        assert info.compute_capability == (8, 6)

    def test_memory_stats(self) -> None:
        """Test memory statistics calculation."""
        info = DeviceInfo(
            device=torch.device("cuda"),
            name="Test GPU",
            total_memory=8_000_000_000,  # 8GB
            allocated_memory=2_000_000_000,  # 2GB
            cached_memory=500_000_000,  # 500MB
        )

        stats = info.memory_stats

        assert stats["total_mb"] == pytest.approx(8000 / 1.024, rel=0.01)
        assert stats["allocated_mb"] == pytest.approx(2000 / 1.024, rel=0.01)
        assert stats["cached_mb"] == pytest.approx(500 / 1.024, rel=0.01)
        assert stats["utilization"] == pytest.approx(0.25, rel=0.01)

    def test_memory_stats_none_values(self) -> None:
        """Test memory stats with None values."""
        info = DeviceInfo(device=torch.device("cpu"), name="CPU")

        stats = info.memory_stats
        assert stats == {}


class TestDeviceManager:
    """Test DeviceManager class."""

    def test_cpu_device(self) -> None:
        """Test CPU device initialization."""
        manager = DeviceManager("cpu")

        assert manager.device.type == "cpu"
        assert manager.is_cpu
        assert not manager.is_cuda
        assert not manager.is_mps

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self) -> None:
        """Test CUDA device initialization."""
        manager = DeviceManager("cuda")

        assert manager.device.type == "cuda"
        assert manager.is_cuda
        assert not manager.is_cpu
        assert not manager.is_mps

    def test_auto_device_cpu_only(self) -> None:
        """Test auto device selection when only CPU available."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = DeviceManager("auto")
            assert manager.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_auto_device_with_cuda(self) -> None:
        """Test auto device selection with CUDA available."""
        manager = DeviceManager("auto")
        assert manager.device.type == "cuda"

    def test_invalid_device(self) -> None:
        """Test invalid device specification."""
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(
                RuntimeError,
                match="CUDA device requested but CUDA is not available",
            ):
                DeviceManager("cuda")

    def test_device_from_torch_device(self) -> None:
        """Test initialization with torch.device object."""
        device = torch.device("cpu")
        manager = DeviceManager(device)

        assert manager.device == device
        assert manager.device.type == "cpu"

    def test_specific_cuda_device(self) -> None:
        """Test specific CUDA device selection."""
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            manager = DeviceManager("cuda:1")
            assert manager.device.type == "cuda"
            assert manager.device.index == 1

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_get_device_info_cuda(self) -> None:
        """Test getting CUDA device info."""
        manager = DeviceManager("cuda")
        info = manager.get_device_info()

        assert info.device.type == "cuda"
        assert info.name != ""
        assert info.total_memory is not None
        assert info.total_memory > 0
        assert info.compute_capability is not None

    def test_get_device_info_cpu(self) -> None:
        """Test getting CPU device info."""
        manager = DeviceManager("cpu")
        info = manager.get_device_info()

        assert info.device.type == "cpu"
        assert info.name == "cpu"
        assert info.total_memory is None

    def test_to_device(self) -> None:
        """Test moving tensor to device."""
        manager = DeviceManager("cpu")
        tensor = torch.randn(10, 10)

        moved = manager.to_device(tensor)
        assert moved.device.type == "cpu"

        # Test non-blocking
        moved_nb = manager.to_device(tensor, non_blocking=True)
        assert moved_nb.device.type == "cpu"

    def test_module_to_device(self) -> None:
        """Test moving module to device."""
        manager = DeviceManager("cpu")
        module = torch.nn.Linear(10, 5)

        moved = manager.module_to_device(module)
        assert next(moved.parameters()).device.type == "cpu"

    def test_device_placement_context(self) -> None:
        """Test device placement context manager."""
        manager = DeviceManager("cpu")

        # Test temporary device placement
        with manager.device_placement("cpu"):
            assert manager.device.type == "cpu"

        # Should return to original device
        assert manager.device.type == "cpu"

        # Test None device (no-op)
        original = manager.device
        with manager.device_placement(None):
            assert manager.device == original

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_clear_cache_cuda(self) -> None:
        """Test GPU cache clearing."""
        manager = DeviceManager("cuda")

        # Allocate some memory
        tensor = torch.randn(1000, 1000, device="cuda")
        del tensor

        # Clear cache
        manager.clear_cache()

        # Should not raise any errors
        assert True

    def test_clear_cache_cpu(self) -> None:
        """Test cache clearing on CPU (should be no-op)."""
        manager = DeviceManager("cpu")
        manager.clear_cache()

        # Should not raise any errors
        assert True

    def test_synchronize(self) -> None:
        """Test device synchronization."""
        manager = DeviceManager("cpu")
        manager.synchronize()  # Should be no-op for CPU

        if torch.cuda.is_available():
            cuda_manager = DeviceManager("cuda")
            cuda_manager.synchronize()  # Should sync CUDA

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_autocast_cuda(self) -> None:
        """Test autocast context for CUDA."""
        manager = DeviceManager("cuda")

        # Test with autocast enabled
        with manager.autocast(enabled=True):
            x = torch.randn(10, 10, device="cuda")
            y = torch.randn(10, 10, device="cuda")
            z = torch.matmul(x, y)
            # In autocast, operations may use float16
            assert z.dtype in (torch.float16, torch.float32)

        # Test with autocast disabled
        with manager.autocast(enabled=False):
            x = torch.randn(10, 10, device="cuda", dtype=torch.float32)
            y = torch.randn(10, 10, device="cuda", dtype=torch.float32)
            z = torch.matmul(x, y)
            assert z.dtype == torch.float32

    def test_autocast_cpu(self) -> None:
        """Test autocast context for CPU."""
        if not hasattr(torch.cpu.amp, "autocast"):
            pytest.skip("CPU autocast not available in this PyTorch version")

        manager = DeviceManager("cpu")

        with manager.autocast(enabled=True):
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.matmul(x, y)
            # CPU autocast typically uses bfloat16
            assert z.dtype in (torch.bfloat16, torch.float32)

    def test_get_memory_summary(self) -> None:
        """Test memory summary string generation."""
        manager = DeviceManager("cpu")
        summary = manager.get_memory_summary()

        assert "Device: cpu" in summary

        if torch.cuda.is_available():
            cuda_manager = DeviceManager("cuda")
            cuda_summary = cuda_manager.get_memory_summary()
            assert "Device:" in cuda_summary
            assert "Memory:" in cuda_summary

    def test_get_available_devices(self) -> None:
        """Test getting list of available devices."""
        devices = DeviceManager.get_available_devices()

        # CPU should always be available
        assert any(d.type == "cpu" for d in devices)

        if torch.cuda.is_available():
            # Should have at least one CUDA device
            assert any(d.type == "cuda" for d in devices)

            # Number of CUDA devices should match
            cuda_devices = [d for d in devices if d.type == "cuda"]
            assert len(cuda_devices) == torch.cuda.device_count()

    def test_repr(self) -> None:
        """Test string representation."""
        manager = DeviceManager("cpu")
        repr_str = repr(manager)

        assert "DeviceManager" in repr_str
        assert "cpu" in repr_str


class TestGlobalDeviceManagement:
    """Test global device management functions."""

    def test_get_device_manager(self) -> None:
        """Test getting global device manager."""
        manager1 = get_device_manager()
        manager2 = get_device_manager()

        # Should return same instance
        assert manager1 is manager2

    def test_set_device(self) -> None:
        """Test setting global device."""
        original_device = get_device()

        try:
            set_device("cpu")
            assert get_device().type == "cpu"

            if torch.cuda.is_available():
                set_device("cuda")
                assert get_device().type == "cuda"
        finally:
            # Restore original device
            set_device(original_device)

    def test_to_device_global(self) -> None:
        """Test global to_device function."""
        tensor = torch.randn(5, 5)

        # Move using global device
        moved = to_device(tensor)
        assert moved.device == get_device()

        # Move to specific device
        cpu_tensor = to_device(tensor, torch.device("cpu"))
        assert cpu_tensor.device.type == "cpu"


class TestDecorators:
    """Test device-related decorators."""

    def test_auto_device_decorator(self) -> None:
        """Test auto_device decorator."""

        @auto_device
        def dummy_function(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        tensor = torch.randn(5, 5)
        result = dummy_function(tensor)

        assert torch.allclose(result, tensor * 2)

    def test_memory_efficient_decorator(self) -> None:
        """Test memory_efficient decorator."""
        clear_cache_called = []

        # Mock clear_cache
        original_clear_cache = DeviceManager.clear_cache

        def mock_clear_cache(self) -> None:
            clear_cache_called.append(True)
            original_clear_cache(self)

        with patch.object(DeviceManager, "clear_cache", mock_clear_cache):

            @memory_efficient
            def dummy_function():
                return torch.randn(100, 100)

            result = dummy_function()

            # Clear cache should be called twice (before and after)
            assert len(clear_cache_called) == 2
            assert result.shape == (100, 100)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_device(self) -> None:
        """Test None device handling."""
        manager = DeviceManager(None)
        # Should default to auto
        assert manager.device.type in ("cpu", "cuda", "mps")

    def test_invalid_cuda_index(self) -> None:
        """Test invalid CUDA device index."""
        with pytest.raises(RuntimeError):
            DeviceManager("cuda:999")

    def test_device_switching(self) -> None:
        """Test switching between devices."""
        manager = DeviceManager("cpu")

        # Create tensor on CPU
        cpu_tensor = torch.randn(5, 5)
        assert manager.to_device(cpu_tensor).device.type == "cpu"

        if torch.cuda.is_available():
            # Temporarily switch to CUDA
            with manager.device_placement("cuda"):
                cuda_tensor = manager.to_device(cpu_tensor)
                assert cuda_tensor.device.type == "cuda"

            # Back to CPU
            assert manager.device.type == "cpu"
