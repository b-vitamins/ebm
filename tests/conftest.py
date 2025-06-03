"""Global pytest configuration and fixtures for the EBM test suite.

This module provides:
- Test configuration and setup
- Shared fixtures used across test modules
- Custom pytest markers
- Test utilities
"""

import os
import sys
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import torch
from _pytest.config import Config
from _pytest.python import Metafunc

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure random seeds for reproducibility
SEED = 42


def pytest_configure(config: Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", "property: marks property-based tests"
    )


def pytest_collection_modifyitems(config: Config, items: list) -> None:
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        if "property" in str(item.fspath):
            item.add_marker(pytest.mark.property)
        if "functional" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session", autouse=True)
def configure_testing_environment() -> None:
    """Configure the testing environment."""
    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Configure PyTorch for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables
    os.environ["PYTHONHASHSEED"] = str(SEED)


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the default device for testing."""
    if torch.cuda.is_available() and not os.environ.get("FORCE_CPU_TESTS"):
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def dtype() -> torch.dtype:
    """Get the default dtype for testing."""
    return torch.float32


@pytest.fixture
def random_state() -> Generator[np.random.RandomState, None, None]:
    """Provide a seeded random state for tests."""
    state = np.random.RandomState(SEED)
    yield state


@pytest.fixture
def torch_random_state() -> Generator[torch.Generator, None, None]:
    """Provide a seeded PyTorch random generator."""
    generator = torch.Generator()
    generator.manual_seed(SEED)
    yield generator


@pytest.fixture(autouse=True)
def cleanup_gpu_memory() -> Generator[None, None, None]:
    """Clean up GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def assert_tensors_equal():
    """Fixture providing tensor comparison utility."""
    def _assert_tensors_equal(
        actual: torch.Tensor,
        expected: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        msg: str = ""
    ) -> None:
        """Assert that two tensors are equal within tolerance."""
        assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
        assert actual.dtype == expected.dtype, f"Dtype mismatch: {actual.dtype} vs {expected.dtype}"
        assert actual.device.type == expected.device.type, f"Device mismatch: {actual.device} vs {expected.device}"

        if actual.dtype in (torch.float16, torch.float32, torch.float64):
            assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
                f"Tensor values not close. {msg}\n"
                f"Max diff: {(actual - expected).abs().max():.6e}\n"
                f"Mean diff: {(actual - expected).abs().mean():.6e}"
            )
        else:
            assert torch.equal(actual, expected), f"Tensor values not equal. {msg}"

    return _assert_tensors_equal


@pytest.fixture
def assert_gradients_valid():
    """Fixture providing gradient validation utility."""
    def _assert_gradients_valid(
        model: torch.nn.Module,
        check_nan: bool = True,
        check_inf: bool = True,
        max_norm: float | None = None
    ) -> None:
        """Assert that model gradients are valid."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad

                if check_nan:
                    assert not torch.isnan(grad).any(), f"NaN gradient in {name}"

                if check_inf:
                    assert not torch.isinf(grad).any(), f"Inf gradient in {name}"

                if max_norm is not None:
                    grad_norm = grad.norm().item()
                    assert grad_norm <= max_norm, (
                        f"Gradient norm too large in {name}: {grad_norm:.4f} > {max_norm}"
                    )

    return _assert_gradients_valid


@pytest.fixture
def benchmark_wrapper(benchmark):
    """Enhanced benchmark fixture with automatic GPU synchronization."""
    def _benchmark(func, *args, **kwargs):
        def wrapped():
            result = func(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return result

        return benchmark(wrapped)

    return _benchmark


# Parametrization helpers
def pytest_generate_tests(metafunc: Metafunc) -> None:
    """Generate parametrized tests based on markers."""
    # Parametrize over devices if requested
    if "parametrize_device" in metafunc.fixturenames:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        metafunc.parametrize("parametrize_device", devices)

    # Parametrize over dtypes if requested
    if "parametrize_dtype" in metafunc.fixturenames:
        dtypes = [torch.float32, torch.float64]
        if torch.cuda.is_available():
            dtypes.append(torch.float16)
        metafunc.parametrize("parametrize_dtype", dtypes)

    # Parametrize over batch sizes if requested
    if "parametrize_batch_size" in metafunc.fixturenames:
        batch_sizes = [1, 16, 64, 128]
        metafunc.parametrize("parametrize_batch_size", batch_sizes)


# Test data generation utilities
@pytest.fixture
def make_random_tensor():
    """Factory fixture for creating random tensors."""
    def _make_random_tensor(
        shape: tuple,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        low: float = 0.0,
        high: float = 1.0,
        binary: bool = False
    ) -> torch.Tensor:
        """Create a random tensor with specified properties."""
        if binary:
            tensor = torch.rand(shape, dtype=dtype, device=device)
            return (tensor > 0.5).to(dtype)
        else:
            tensor = torch.rand(shape, dtype=dtype, device=device)
            return low + (high - low) * tensor

    return _make_random_tensor


# Performance testing utilities
class PerformanceMonitor:
    """Monitor performance metrics during tests."""

    def __init__(self):
        self.metrics = {}

    def record(self, name: str, value: float) -> None:
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_stats(self, name: str) -> dict:
        """Get statistics for a metric."""
        values = self.metrics.get(name, [])
        if not values:
            return {}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values)
        }


@pytest.fixture
def performance_monitor():
    """Provide a performance monitoring utility."""
    return PerformanceMonitor()
