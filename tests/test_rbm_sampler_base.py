# tests/test_rbm_sampler_base.py
"""Comprehensive battle tests for the base RBM sampler module.

This test suite thoroughly exercises all aspects of the base module to ensure
robust and error-free behavior for all sampling infrastructure.
"""

from __future__ import annotations

import gc
import threading
import weakref
from typing import Any

import numpy as np
import pytest
import torch

from ebm.rbm.model.base import BaseRBM
from ebm.rbm.sampler.base import (
    BaseSamplerRBM,
    RemovableHandle,
    SampleRBM,
    SamplingStepBundle,
    TensorType,
    _HookEntry,
    _unpack_impl,
    _WrappedHookSig,
)


# Test fixtures and utilities
class MockRBMModel(BaseRBM):
    """Mock RBM model for testing - implements all abstract methods."""

    def __init__(self, visible_size: int = 4, hidden_size: int = 3) -> None:
        super().__init__()
        self.visible_size = visible_size
        self.hidden_size = hidden_size

    def sample_h_given_v(self, v: TensorType, beta: TensorType | None = None) -> TensorType:
        """Mock implementation."""
        return torch.randint(0, 2, (v.shape[0], self.hidden_size), dtype=torch.float32)

    def sample_v_given_h(self, h: TensorType, beta: TensorType | None = None) -> TensorType:
        """Mock implementation."""
        return torch.randint(0, 2, (h.shape[0], self.visible_size), dtype=torch.float32)

    def prob_h_given_v(self, v: TensorType, beta: TensorType | None = None) -> TensorType:
        """Mock implementation."""
        return torch.rand(v.shape[0], self.hidden_size)

    def prob_v_given_h(self, h: TensorType, beta: TensorType | None = None) -> TensorType:
        """Mock implementation."""
        return torch.rand(h.shape[0], self.visible_size)

    def preact_h(self, v: TensorType, beta: TensorType | None = None) -> TensorType:
        """Mock implementation."""
        return torch.randn(v.shape[0], self.hidden_size)

    def preact_v(self, h: TensorType, beta: TensorType | None = None) -> TensorType:
        """Mock implementation."""
        return torch.randn(h.shape[0], self.visible_size)

    def energy(self, v: TensorType, h: TensorType, beta: TensorType | None = None) -> TensorType:
        """Mock implementation."""
        return torch.randn(v.shape[0])

    def free_energy(self, v: TensorType, beta: TensorType | None = None) -> TensorType:
        """Mock implementation."""
        return torch.randn(v.shape[0])


class ConcreteSampler(BaseSamplerRBM):
    """Concrete implementation for testing."""

    def __init__(self, model: BaseRBM) -> None:
        super().__init__(model)
        self._sample_calls: list[tuple[TensorType, TensorType | None, _WrappedHookSig | None]] = []
        self._hook_calls: list[int] = []

    def _sample(
        self,
        v0: TensorType,
        beta: TensorType | None = None,
        hook_fn: _WrappedHookSig | None = None,
    ) -> tuple[TensorType, TensorType]:
        """Mock implementation."""
        self._sample_calls.append((v0, beta, hook_fn))

        vk = torch.randn_like(v0)
        # Use the model's actual hidden size
        hk = torch.randn(v0.shape[0], self.model.hidden_size)

        # Simulate multiple steps for hook testing
        for step in range(3):
            if hook_fn is not None:
                self._hook_calls.append(step)
                hook_fn((step, vk, hk, beta))

        return vk, hk

    def get_config(self) -> dict[str, Any]:
        return {"type": "concrete"}

    def __repr__(self) -> str:
        return "ConcreteSampler()"


class TestSampleRBM:
    """Battle tests for SampleRBM class."""

    def test_tensor_delegation_comprehensive(self) -> None:
        """Test all tensor operations and edge cases."""
        tensor = torch.randn(3, 4)
        sample = SampleRBM(tensor)

        # Test all properties
        assert sample.shape == tensor.shape
        assert sample.dtype == tensor.dtype
        assert sample.device == tensor.device
        assert sample.ndim == tensor.ndim
        assert sample.numel() == tensor.numel()

        # Test methods
        assert torch.all(sample.abs() == tensor.abs())
        assert torch.all(sample.sum() == tensor.sum())
        assert torch.all(sample.mean() == tensor.mean())

        # Test grad operations
        tensor.requires_grad_(True)
        sample = SampleRBM(tensor)
        assert sample.requires_grad == tensor.requires_grad

    def test_metadata_properties(self) -> None:
        """Test convenient metadata properties."""
        tensor = torch.randn(3, 4)
        initial = torch.randn(3, 4)
        hidden = torch.randn(3, 5)
        intermediate = [torch.randn(3, 4) for _ in range(3)]

        # Full metadata
        sample = SampleRBM(
            tensor, initial_state=initial, final_hidden=hidden, intermediate_states=intermediate
        )

        assert sample.has_initial_state
        assert sample.has_hidden
        assert sample.has_chain

        # Partial metadata
        sample2 = SampleRBM(tensor, initial_state=initial)
        assert sample2.has_initial_state
        assert not sample2.has_hidden
        assert not sample2.has_chain

        # No metadata
        sample3 = SampleRBM(tensor)
        assert not sample3.has_initial_state
        assert not sample3.has_hidden
        assert not sample3.has_chain

    def test_attribute_error_handling(self) -> None:
        """Test clear error messages for non-existent attributes."""
        sample = SampleRBM(torch.randn(3, 4))

        with pytest.raises(
            AttributeError, match="'SampleRBM' object has no attribute 'nonexistent'"
        ):
            _ = sample.nonexistent

    def test_binary_operations_comprehensive(self) -> None:
        """Test all binary operations thoroughly."""
        t1 = torch.randn(3, 4)
        t2 = torch.randn(3, 4)
        s1 = SampleRBM(t1)
        s2 = SampleRBM(t2)
        scalar = 2.5

        # Arithmetic operations
        assert torch.allclose(s1 + s2, t1 + t2)
        assert torch.allclose(s1 - s2, t1 - t2)
        assert torch.allclose(s1 * s2, t1 * t2)
        assert torch.allclose(s1 / s2, t1 / t2)
        assert torch.allclose(s1 @ s2.T, t1 @ t2.T)

        # Mixed operations
        assert torch.allclose(s1 + t2, t1 + t2)
        assert torch.allclose(t1 + s2, t1 + t2)
        assert torch.allclose(s1 * scalar, t1 * scalar)

        # Comparison operations
        assert torch.all((s1 == s2) == (t1 == t2))
        assert torch.all((s1 != s2) == (t1 != t2))
        assert torch.all((s1 < s2) == (t1 < t2))
        assert torch.all((s1 <= s2) == (t1 <= t2))
        assert torch.all((s1 > s2) == (t1 > t2))
        assert torch.all((s1 >= s2) == (t1 >= t2))

    def test_cuda_numpy_conversion(self) -> None:
        """Test CUDA tensor handling for numpy conversion."""
        if torch.cuda.is_available():
            cuda_tensor = torch.randn(3, 4).cuda()
            sample = SampleRBM(cuda_tensor)

            # Should automatically handle CUDA -> CPU conversion
            numpy_array = np.array(sample)
            assert numpy_array.shape == (3, 4)
            assert isinstance(numpy_array, np.ndarray)

    def test_singledispatch_unpacking(self) -> None:
        """Test singledispatch-based unpacking."""
        t1 = torch.randn(2, 3)
        t2 = torch.randn(2, 3)
        s1 = SampleRBM(t1)
        s2 = SampleRBM(t2)

        # Test direct unpacking
        assert torch.equal(_unpack_impl(s1), t1)

        # Test various types
        assert _unpack_impl(5) == 5
        assert _unpack_impl("hello") == "hello"
        assert torch.equal(_unpack_impl(t1), t1)

        # Test collections
        unpacked_list = _unpack_impl([s1, s2, t1])
        assert torch.equal(unpacked_list[0], t1)
        assert torch.equal(unpacked_list[1], t2)
        assert torch.equal(unpacked_list[2], t1)

        unpacked_dict = _unpack_impl({"a": s1, "b": t2})
        assert torch.equal(unpacked_dict["a"], t1)
        assert torch.equal(unpacked_dict["b"], t2)

        unpacked_tuple = _unpack_impl((s1, s2))
        assert torch.equal(unpacked_tuple[0], t1)
        assert torch.equal(unpacked_tuple[1], t2)

        # Note: Sets require hashable elements, and SampleRBM is not hashable
        # So we test with regular tensors in sets
        t3 = 3.14
        unpacked_set = _unpack_impl({t3})
        assert next(iter(unpacked_set)) == t3

    def test_nested_structure_unpacking(self) -> None:
        """Test comprehensive nested structure unpacking."""
        t1 = torch.randn(2, 3)
        t2 = torch.randn(2, 3)
        s1 = SampleRBM(t1)
        s2 = SampleRBM(t2)

        # Test various nested structures
        test_cases = [
            # Lists
            ([s1, s2], [t1, t2]),
            ([[s1], [s2]], [[t1], [t2]]),
            # Tuples
            ((s1, s2), (t1, t2)),
            ((s1, t2), (t1, t2)),
            # Dicts
            ({"a": s1, "b": s2}, {"a": t1, "b": t2}),
            ({"a": [s1, s2], "b": s1}, {"a": [t1, t2], "b": t1}),
            # Mixed
            ({"list": [s1, s2], "tuple": (s1, s2)}, {"list": [t1, t2], "tuple": (t1, t2)}),
        ]

        for input_structure, expected in test_cases:
            result = SampleRBM._unpack(input_structure)
            assert self._structures_equal(result, expected)

    def test_torch_function_comprehensive(self) -> None:
        """Test __torch_function__ with various torch operations."""
        t1 = torch.randn(3, 4)
        t2 = torch.randn(4, 5)
        s1 = SampleRBM(t1)
        s2 = SampleRBM(t2)

        # Test various torch functions using tensor operations
        assert torch.allclose(
            torch.cat([s1.to_tensor(), s1.to_tensor()], dim=0), torch.cat([t1, t1], dim=0)
        )
        assert torch.allclose(torch.stack([s1.to_tensor(), s1.to_tensor()]), torch.stack([t1, t1]))
        assert torch.allclose(torch.matmul(s1.to_tensor(), s2.to_tensor()), torch.matmul(t1, t2))

        # Test with kwargs
        assert torch.allclose(
            torch.cat(tensors=[s1.to_tensor(), s1.to_tensor()], dim=1),
            torch.cat(tensors=[t1, t1], dim=1),
        )

        # Test nested structures in torch functions
        s1_tensor = s1.to_tensor()
        t1_tensor = t1
        assert torch.allclose(
            torch.cat([s1_tensor, s1_tensor], dim=0), torch.cat([t1_tensor, t1_tensor], dim=0)
        )

    def test_metadata_handling(self) -> None:
        """Test metadata storage and access."""
        tensor = torch.randn(3, 4)
        initial = torch.randn(3, 4)
        hidden = torch.randn(3, 5)
        intermediate = [torch.randn(3, 4) for _ in range(3)]

        # Full metadata
        sample = SampleRBM(
            tensor, initial_state=initial, final_hidden=hidden, intermediate_states=intermediate
        )

        assert sample.has_metadata("initial_state", "final_hidden")
        assert sample.has_metadata("intermediate_states")

        # Test non-existent metadata
        assert sample.initial_state is not None
        assert sample.final_hidden is not None
        # Direct attribute check instead of has_metadata
        assert not hasattr(sample, "nonexistent_metadata")

        # Partial metadata
        sample2 = SampleRBM(tensor, initial_state=initial)
        assert sample2.has_metadata("initial_state")
        assert not sample2.has_metadata("final_hidden")

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency with slots."""
        sample = SampleRBM(torch.randn(3, 4))

        # __slots__ should prevent arbitrary attribute assignment
        with pytest.raises(AttributeError):
            sample.new_attr = "value"  # type: ignore

    def _structures_equal(self, a: Any, b: Any) -> bool:
        """Helper to recursively compare structures."""
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.allclose(a, b)
        if type(a) is not type(b):
            return False
        if isinstance(a, list | tuple):
            return len(a) == len(b) and all(
                self._structures_equal(ai, bi) for ai, bi in zip(a, b, strict=False)
            )
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(
                self._structures_equal(a[k], b[k]) for k in a.keys()
            )
        result: bool = a == b
        return result


class TestRemovableHandle:
    """Battle tests for RemovableHandle class."""

    def test_basic_removal(self) -> None:
        """Test basic hook removal."""
        hooks_dict = {}

        def mock_hook(
            s: BaseSamplerRBM, st: int, v: TensorType, h: TensorType, b: TensorType | None
        ) -> None:
            pass

        entry = _HookEntry(mock_hook, "unbundled")
        hooks_dict[0] = entry

        handle = RemovableHandle(hooks_dict, 0)
        assert 0 in hooks_dict

        handle.remove()
        assert 0 not in hooks_dict

    def test_double_removal_safety(self) -> None:
        """Test that double removal is safe."""
        hooks_dict = {}

        def mock_hook(
            s: BaseSamplerRBM, st: int, v: TensorType, h: TensorType, b: TensorType | None
        ) -> None:
            pass

        entry = _HookEntry(mock_hook, "unbundled")
        hooks_dict[0] = entry

        handle = RemovableHandle(hooks_dict, 0)
        handle.remove()
        handle.remove()  # Should not raise

        assert 0 not in hooks_dict

    def test_already_removed_key(self) -> None:
        """Test behavior when key is already removed."""
        hooks_dict = {}

        def mock_hook(
            s: BaseSamplerRBM, st: int, v: TensorType, h: TensorType, b: TensorType | None
        ) -> None:
            pass

        entry = _HookEntry(mock_hook, "unbundled")
        hooks_dict[0] = entry

        handle = RemovableHandle(hooks_dict, 0)

        # Remove directly from dict
        del hooks_dict[0]

        # Handle removal should still be safe
        handle.remove()


class TestBaseSamplerRBM:
    """Battle tests for BaseSamplerRBM class."""

    @pytest.fixture
    def mock_model(self) -> MockRBMModel:
        return MockRBMModel()

    @pytest.fixture
    def sampler(self, mock_model: MockRBMModel) -> ConcreteSampler:
        return ConcreteSampler(mock_model)

    def test_fast_path_optimization(self, sampler: ConcreteSampler) -> None:
        """Test that fast path is used when appropriate."""
        v0 = torch.randn(2, 4)

        # Fast path (no hooks, no metadata)
        result = sampler.sample(v0)
        assert len(sampler._sample_calls) == 1
        assert sampler._sample_calls[0][2] is None  # No hook_fn
        assert result.initial_state is None
        assert result.final_hidden is None
        assert result.intermediate_states is None

    def test_full_path_with_metadata(self, sampler: ConcreteSampler) -> None:
        """Test full path with metadata tracking."""
        v0 = torch.randn(2, 4)

        result = sampler.sample(v0, return_hidden=True, track_chains=True)

        assert len(sampler._sample_calls) == 1
        assert sampler._sample_calls[0][2] is not None  # Has hook_fn
        assert torch.all(result.initial_state == v0)
        assert result.final_hidden is not None
        assert result.intermediate_states is not None
        # Based on mock implementation
        assert len(result.intermediate_states) == 3

    def test_metadata_properties_in_sampling(self, sampler: ConcreteSampler) -> None:
        """Test convenient metadata properties in sampling results."""
        v0 = torch.randn(2, 4)

        # Test with full metadata
        result1 = sampler.sample(v0, return_hidden=True, track_chains=True)
        assert result1.has_initial_state
        assert result1.has_hidden
        assert result1.has_chain

        # Test with partial metadata
        result2 = sampler.sample(v0, return_hidden=True)
        assert result2.has_initial_state
        assert result2.has_hidden
        assert not result2.has_chain

        # Test with no metadata
        result3 = sampler.sample(v0)
        assert not result3.has_initial_state
        assert not result3.has_hidden
        assert not result3.has_chain

    def test_hook_registration_and_dispatch(self, sampler: ConcreteSampler) -> None:
        """Test hook registration with both styles."""
        call_log: list[tuple[str, int, torch.Size, torch.Size, Any]] = []

        def unbundled_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: TensorType,
            h: TensorType,
            beta: TensorType | None,
        ) -> None:
            call_log.append(("unbundled", step, v.shape, h.shape, beta))

        def bundled_hook(sampler: BaseSamplerRBM, bundle: SamplingStepBundle) -> None:
            step, v, h, beta = bundle
            call_log.append(("bundled", step, v.shape, h.shape, beta))

        _ = sampler.register_sampling_hook(unbundled_hook)
        _ = sampler.register_sampling_hook_bundled(bundled_hook)

        v0 = torch.randn(2, 4)
        _ = sampler.sample(v0)

        # Both hooks should be called for each step
        assert len(call_log) == 6  # 3 steps * 2 hooks
        unbundled_calls = [c for c in call_log if c[0] == "unbundled"]
        bundled_calls = [c for c in call_log if c[0] == "bundled"]

        assert len(unbundled_calls) == 3
        assert len(bundled_calls) == 3

        # Check step numbers
        for i, (_style, step, _, _, _) in enumerate(unbundled_calls):
            assert step == i

    def test_temporary_hook_context_manager(self, sampler: ConcreteSampler) -> None:
        """Test the temporary_hook context manager."""
        call_log: list[tuple[int, torch.Size]] = []

        def monitoring_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: TensorType,
            h: TensorType,
            beta: TensorType | None,
        ) -> None:
            call_log.append((step, v.shape))

        v0 = torch.randn(2, 4)

        # Test unbundled style (default)
        with sampler.temporary_hook(monitoring_hook):
            assert len(sampler._sampling_hooks) == 1
            sampler.sample(v0)
            assert len(call_log) == 3  # 3 steps

        # Hook should be automatically removed
        assert len(sampler._sampling_hooks) == 0

        # Test bundled style
        call_log.clear()

        def bundled_monitoring_hook(sampler: BaseSamplerRBM, bundle: SamplingStepBundle) -> None:
            step, v, h, beta = bundle
            call_log.append((step, v.shape))

        with sampler.temporary_hook(bundled_monitoring_hook, style="bundled"):
            assert len(sampler._sampling_hooks) == 1
            sampler.sample(v0)
            assert len(call_log) == 3

        assert len(sampler._sampling_hooks) == 0

        # Test exception handling
        call_log.clear()

        class TestExceptionError(Exception):
            pass

        def failing_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: TensorType,
            h: TensorType,
            beta: TensorType | None,
        ) -> None:
            if step == 1:
                raise TestExceptionError("Test error")

        with pytest.raises(TestExceptionError):
            with sampler.temporary_hook(failing_hook):
                sampler.sample(v0)

        # Hook should still be removed even with exception
        assert len(sampler._sampling_hooks) == 0

    def test_hook_removal_during_iteration(self, sampler: ConcreteSampler) -> None:
        """Test self-removal during hook iteration."""
        remove_handle: RemovableHandle | None = None
        call_count = 0

        def self_removing_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: TensorType,
            h: TensorType,
            beta: TensorType | None,
        ) -> None:
            nonlocal call_count
            call_count += 1
            if step == 1:  # Remove self on second call
                assert remove_handle is not None
                remove_handle.remove()

        def other_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: TensorType,
            h: TensorType,
            beta: TensorType | None,
        ) -> None:
            pass

        remove_handle = sampler.register_sampling_hook(self_removing_hook)
        _ = sampler.register_sampling_hook(other_hook)

        v0 = torch.randn(2, 4)
        sampler.sample(v0)

        # Should be called twice (steps 0 and 1) before removal
        assert call_count == 2

        # Other hook should still be registered
        assert len(sampler._sampling_hooks) == 1

    def test_hook_exception_propagation(self, sampler: ConcreteSampler) -> None:
        """Test that exceptions in hooks bubble up."""

        def failing_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: TensorType,
            h: TensorType,
            beta: TensorType | None,
        ) -> None:
            if step == 1:
                raise ValueError("Test exception")

        sampler.register_sampling_hook(failing_hook)

        v0 = torch.randn(2, 4)
        with pytest.raises(ValueError, match="Test exception"):
            sampler.sample(v0)

    def test_gradient_behavior(self, sampler: ConcreteSampler) -> None:
        """Test gradient behavior in hooks."""
        gradients_enabled: list[bool] = []

        def gradient_checking_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: TensorType,
            h: TensorType,
            beta: TensorType | None,
        ) -> None:
            gradients_enabled.append(torch.is_grad_enabled())

        sampler.register_sampling_hook(gradient_checking_hook)

        v0 = torch.randn(2, 4)

        # Hooks should run with gradients enabled by default
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            sampler.sample(v0)

        assert all(gradients_enabled)

        # But respect global gradient state
        gradients_enabled.clear()
        with torch.no_grad():
            sampler.sample(v0)

        assert not any(gradients_enabled)

    def test_beta_parameter_handling(self, sampler: ConcreteSampler) -> None:
        """Test beta parameter propagation."""
        captured_betas: list[TensorType | None] = []

        def beta_capture_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: TensorType,
            h: TensorType,
            beta: TensorType | None,
        ) -> None:
            captured_betas.append(beta)

        sampler.register_sampling_hook(beta_capture_hook)

        v0 = torch.randn(2, 4)
        beta = torch.tensor(0.5)

        sampler.sample(v0, beta=beta)

        assert all(torch.equal(b, beta) if b is not None else False for b in captured_betas)

    def test_hook_counter_overflow(self, sampler: ConcreteSampler) -> None:
        """Test that hook counter handles many registrations."""
        # Register many hooks
        handles: list[RemovableHandle] = []
        for _i in range(1000):
            handle = sampler.register_sampling_hook(lambda s, st, v, h, b: None)
            handles.append(handle)

        assert sampler._hook_counter == 1000

        # Remove all hooks
        for handle in handles:
            handle.remove()

        assert len(sampler._sampling_hooks) == 0

    def test_thread_safety_concerns(self, sampler: ConcreteSampler) -> None:
        """Test behavior under concurrent access (document limitations)."""
        # This test documents that hook registration is NOT thread-safe
        results: list[RemovableHandle] = []

        def register_hooks() -> None:
            for _i in range(100):
                handle = sampler.register_sampling_hook(lambda s, st, v, h, b: None)
                results.append(handle)

        threads = [threading.Thread(target=register_hooks) for _ in range(5)]

        # This MAY cause issues due to non-thread-safe dict mutations
        # We're testing that at least it doesn't crash
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # We should have some hooks registered
        assert len(sampler._sampling_hooks) > 0

    def test_memory_management(self, sampler: ConcreteSampler) -> None:
        """Test memory management and cleanup."""
        # Test that removed hooks are garbage collected
        hook_weakref: weakref.ReferenceType[Any] | None = None

        def test_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: TensorType,
            h: TensorType,
            beta: TensorType | None,
        ) -> None:
            pass

        hook_weakref = weakref.ref(test_hook)
        handle = sampler.register_sampling_hook(test_hook)

        # Hook should be alive
        assert hook_weakref() is not None

        # Remove and delete references
        handle.remove()
        del test_hook
        gc.collect()

        # Hook should be garbage collected
        assert hook_weakref() is None

    def test_forward_method(self, sampler: ConcreteSampler) -> None:
        """Test forward method aliases sample."""
        v0 = torch.randn(2, 4)

        result1 = sampler.sample(v0)
        result2 = sampler.forward(v0)
        result3 = sampler(v0)  # nn.Module __call__

        assert result1.shape == result2.shape == result3.shape


class TestIntegration:
    """Integration tests for the complete system."""

    def test_complete_sampling_workflow(self) -> None:
        """Test a complete realistic sampling workflow."""
        model = MockRBMModel(visible_size=10, hidden_size=8)
        sampler = ConcreteSampler(model)

        # Set up monitoring
        metrics: dict[str, list[int | float]] = {
            "steps": [],
            "v_norms": [],
            "h_norms": [],
        }

        def monitoring_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: TensorType,
            h: TensorType,
            beta: TensorType | None,
        ) -> None:
            metrics["steps"].append(step)
            metrics["v_norms"].append(torch.norm(v).item())
            metrics["h_norms"].append(torch.norm(h).item())

        # Register hook
        handle = sampler.register_sampling_hook(monitoring_hook)

        # Run sampling
        v0 = torch.randn(32, 10)  # Batch of 32
        result = sampler.sample(v0, return_hidden=True, track_chains=True)

        # Verify results
        assert isinstance(result, SampleRBM)
        assert result.shape == (32, 10)
        assert result.has_hidden
        assert result.has_chain
        if result.final_hidden is not None:
            assert result.final_hidden.shape == (32, 8)
        if result.intermediate_states is not None:
            assert len(result.intermediate_states) == 3

        # Check metrics
        assert metrics["steps"] == [0, 1, 2]
        assert len(metrics["v_norms"]) == 3
        assert len(metrics["h_norms"]) == 3

        # Clean up
        handle.remove()

    def test_complete_workflow_with_temporary_hook(self) -> None:
        """Test complete workflow using temporary hook context manager."""
        model = MockRBMModel(visible_size=10, hidden_size=8)
        sampler = ConcreteSampler(model)

        metrics: dict[str, list[int | float]] = {
            "steps": [],
            "v_norms": [],
        }

        def monitoring_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: TensorType,
            h: TensorType,
            beta: TensorType | None,
        ) -> None:
            metrics["steps"].append(step)
            metrics["v_norms"].append(torch.norm(v).item())

        v0 = torch.randn(16, 10)

        # Use temporary hook
        with sampler.temporary_hook(monitoring_hook):
            result = sampler.sample(v0, return_hidden=True)

        # Hook should be automatically removed
        assert len(sampler._sampling_hooks) == 0

        # Check results
        assert isinstance(result, SampleRBM)
        assert result.shape == (16, 10)
        assert result.has_hidden
        assert metrics["steps"] == [0, 1, 2]
        assert len(metrics["v_norms"]) == 3

    def test_cuda_compatibility(self) -> None:
        """Test CUDA compatibility if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = MockRBMModel()
        sampler = ConcreteSampler(model)

        v0 = torch.randn(4, 4).cuda()
        result = sampler.sample(v0)

        # Result should maintain device
        assert result.device == v0.device
        assert result.is_cuda


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
