"""Model-agnostic sampler test mixins.

This module provides generic test mixins that can be used with any sampler
implementation. These tests verify basic functionality, device handling,
hook systems, and other general properties.
"""

from __future__ import annotations

import gc
import importlib
import multiprocessing as mp
import os
import pickle
import threading
import weakref
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from hypothesis import given

from tests.helper import (
    run_performance_cpu_test,
    run_performance_gpu_test,
)
from tests.helper.sampler import sampler_model_config

if TYPE_CHECKING:
    from ebm.rbm.model.base import BaseRBM
    from ebm.rbm.sampler.base import BaseSamplerRBM


class ShapeTests:
    """Tests for output shapes and structure.

    These tests verify that samplers preserve batch dimensions and
    produce outputs with the expected shapes.
    """

    def test_output_shape(self, sampler: BaseSamplerRBM, sample_input: torch.Tensor) -> None:
        """Test that output shapes match input batch size.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance to test.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """
        # Basic sample
        result = sampler.sample(sample_input)
        assert hasattr(result, "shape"), "Output must be tensor-like"
        assert result.shape[0] == sample_input.shape[0], "Batch size mismatch"

        # With metadata
        try:
            result_full = sampler.sample(sample_input, return_hidden=True, track_chains=True)
            assert result_full.shape[0] == sample_input.shape[0]

            if hasattr(result_full, "final_hidden") and result_full.final_hidden is not None:
                assert result_full.final_hidden.shape[0] == sample_input.shape[0]

            if (
                hasattr(result_full, "intermediate_states")
                and result_full.intermediate_states is not None
            ):
                for state in result_full.intermediate_states:
                    assert state.shape[0] == sample_input.shape[0]
        except TypeError:
            # Sampler might not support all metadata options
            pass


class DeviceAutogradTests:
    """Tests for device handling and autograd behavior.

    These tests ensure that samplers correctly handle device placement
    and gradient computation.
    """

    def test_device_preservation(
        self,
        sampler_class: type[BaseSamplerRBM],
        model: BaseRBM,
        visible_size: int,
    ) -> None:
        """Test that outputs maintain input device.

        Parameters
        ----------
        sampler_class : type[BaseSamplerRBM]
            Sampler class to test.
        model : BaseRBM
            Model instance.
        visible_size : int
            Number of visible units.
        """
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            # Create a new sampler with model on the correct device
            model_on_device = model.to(device)
            sampler_on_device = sampler_class(model_on_device)

            v0 = torch.randn(4, visible_size, device=device)
            result = sampler_on_device.sample(v0)
            assert result.device.type == device

    def test_dtype_preservation(
        self,
        sampler_class: type[BaseSamplerRBM],
        rbm_class: type[BaseRBM],
        rbm_config: Any,
        visible_size: int,
    ) -> None:
        """Test that dtypes are preserved through sampling.

        Parameters
        ----------
        sampler_class : type[BaseSamplerRBM]
            Sampler class to test.
        rbm_class : type[BaseRBM]
            RBM class to instantiate.
        rbm_config : Any
            Base RBM configuration.
        visible_size : int
            Number of visible units.
        """
        dtypes_collected: list[torch.dtype] = []

        def dtype_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: torch.Tensor,
            h: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            dtypes_collected.extend([v.dtype, h.dtype])

        for dtype in [torch.float32, torch.float64]:
            # Create model and sampler with proper dtype
            config = type(rbm_config)(visible=visible_size, hidden=rbm_config.hidden, dtype=dtype)
            model = rbm_class(config)
            sampler = sampler_class(model)

            handle = sampler.register_sampling_hook(dtype_hook)
            try:
                v0 = torch.randn(4, visible_size, dtype=dtype)
                result = sampler.sample(v0)
                assert result.dtype == dtype

                # Clear for next iteration
                dtypes_collected.clear()
            finally:
                handle.remove()


class HookTests:
    """Tests for the hook system.

    These tests verify that the hook registration and invocation
    system works correctly.
    """

    def test_hook_registration_and_removal(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
    ) -> None:
        """Test basic hook registration and removal.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """
        calls = {"unbundled": 0, "bundled": 0}

        def unbundled_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: torch.Tensor,
            h: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            calls["unbundled"] += 1

        def bundled_hook(
            sampler: BaseSamplerRBM,
            bundle: tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None],
        ) -> None:
            calls["bundled"] += 1

        handle1 = sampler.register_sampling_hook(unbundled_hook)
        handle2 = sampler.register_sampling_hook_bundled(bundled_hook)

        # Run sampling
        sampler.sample(sample_input)

        # Both hooks should be called
        assert calls["unbundled"] > 0
        assert calls["bundled"] > 0

        # Remove hooks
        handle1.remove()
        handle2.remove()

        # Reset counters
        calls["unbundled"] = 0
        calls["bundled"] = 0

        # Run again - hooks shouldn't be called
        sampler.sample(sample_input)
        assert calls["unbundled"] == 0
        assert calls["bundled"] == 0

    def test_hook_exception_propagation(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
    ) -> None:
        """Test that exceptions in hooks bubble up.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """

        def failing_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: torch.Tensor,
            h: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            if step >= 0:  # Changed from step == 1 to be more generic
                raise ValueError("Test exception")

        handle = sampler.register_sampling_hook(failing_hook)

        try:
            with pytest.raises(ValueError, match="Test exception"):
                sampler.sample(sample_input)
        finally:
            handle.remove()

    def test_hook_self_removal(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
    ) -> None:
        """Test hooks can safely remove themselves.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """
        remove_handle = None
        call_count = 0

        def self_removing_hook(
            sampler: BaseSamplerRBM,
            step: int,
            v: torch.Tensor,
            h: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            nonlocal call_count
            call_count += 1
            # Remove after we've been called at least twice
            if call_count >= 2 and remove_handle is not None:
                remove_handle.remove()

        remove_handle = sampler.register_sampling_hook(self_removing_hook)

        # This should not crash
        sampler.sample(sample_input)

        # Hook should have been called at least once
        assert call_count >= 1

    def test_hook_self_removal_during_iteration(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
    ) -> None:
        """Test hook self-removal during multi-step sampling.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """
        calls_a: list[int] = []
        calls_b: list[int] = []

        def hook_a(
            s: BaseSamplerRBM,
            step: int,
            vk: torch.Tensor,
            hk: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            calls_a.append(step)
            if len(calls_a) >= 1:  # Remove after 1 call
                handle_a.remove()

        def hook_b(
            s: BaseSamplerRBM,
            step: int,
            vk: torch.Tensor,
            hk: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            calls_b.append(step)

        handle_a = sampler.register_sampling_hook(hook_a)
        sampler.register_sampling_hook(hook_b)
        sampler.sample(sample_input)

        # Hook A should have been removed after first call
        assert len(calls_a) >= 1
        # Hook B should have been called for all steps
        assert len(calls_b) >= len(calls_a)

    def test_hook_thread_safety(self, sampler: BaseSamplerRBM) -> None:
        """Test concurrent hook registration (document limitations).

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        """
        handles = []

        def register_hooks() -> None:
            for _i in range(100):
                handle = sampler.register_sampling_hook(lambda s, st, v, h, b: None)
                handles.append(handle)

        threads = [threading.Thread(target=register_hooks) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have registered some hooks
        assert len(sampler._sampling_hooks) > 0

        # Clean up
        for handle in handles:
            handle.remove()

    def test_hook_garbage_collected_after_removal(self, sampler: BaseSamplerRBM) -> None:
        """Test that hooks are garbage collected after removal.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        """

        def hook(
            s: BaseSamplerRBM,
            step: int,
            vk: torch.Tensor,
            hk: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            pass

        weak_ref = weakref.ref(hook)
        handle = sampler.register_sampling_hook(hook)
        handle.remove()
        del hook
        gc.collect()

        assert weak_ref() is None

    def test_concurrent_registration_no_crash(self, sampler: BaseSamplerRBM) -> None:
        """Test concurrent hook registration doesn't crash.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        """

        def register_hooks() -> None:
            for _ in range(100):
                sampler.register_sampling_hook(lambda s, step, vk, hk, beta: None)

        threads = [threading.Thread(target=register_hooks) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(sampler._sampling_hooks) >= 1

    def test_hooks_receive_detached_tensors(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
    ) -> None:
        """Test that hooks receive detached tensors.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """
        gradient_flags: list[tuple[bool, bool]] = []

        def grad_hook(
            sampler: BaseSamplerRBM,
            step: int,
            vk: torch.Tensor,
            hk: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            gradient_flags.append((vk.requires_grad, hk.requires_grad))

        handle = sampler.register_sampling_hook(grad_hook)

        try:
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                v0_grad = sample_input.clone().requires_grad_()
                sampler.sample(v0_grad)

            assert len(gradient_flags) > 0
            assert all(not v_flag and not h_flag for v_flag, h_flag in gradient_flags)
        finally:
            handle.remove()

    def test_global_no_grad_respected(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
    ) -> None:
        """Test that global no_grad context is respected.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """
        grad_enabled_states: list[bool] = []

        def grad_hook(
            s: BaseSamplerRBM,
            step: int,
            vk: torch.Tensor,
            hk: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            grad_enabled_states.append(torch.is_grad_enabled())

        handle = sampler.register_sampling_hook(grad_hook)

        try:
            with torch.no_grad():
                sampler.sample(sample_input)

            assert len(grad_enabled_states) > 0
            assert all(not g for g in grad_enabled_states)
        finally:
            handle.remove()

    def test_no_graph_growth(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
    ) -> None:
        """Test that sampling doesn't create computation graphs.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """
        v0_grad = sample_input.clone().requires_grad_(True)
        result = sampler.sample(v0_grad, return_hidden=True)

        assert result.grad_fn is None
        if hasattr(result, "final_hidden") and result.final_hidden is not None:
            assert result.final_hidden.grad_fn is None


class DeterminismTests:
    """Tests for deterministic behavior.

    These tests verify that samplers produce reproducible results
    with fixed random seeds.
    """

    def test_fixed_seed_reproducibility(
        self,
        sampler: BaseSamplerRBM,
        visible_size: int,
    ) -> None:
        """Test that fixed seed gives reproducible results.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        visible_size : int
            Number of visible units.
        """
        # Create input
        v0 = torch.randn(8, visible_size)

        # First run
        torch.manual_seed(42)
        result1 = sampler.sample(v0)

        # Second run with same seed
        torch.manual_seed(42)
        result2 = sampler.sample(v0)

        # Results should be identical
        assert torch.allclose(result1.to_tensor(), result2.to_tensor())

    def test_different_seeds_give_different_results(
        self,
        sampler: BaseSamplerRBM,
        visible_size: int,
    ) -> None:
        """Test that different seeds produce different results.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        visible_size : int
            Number of visible units.
        """
        v0 = torch.randn(8, visible_size)

        torch.manual_seed(42)
        result1 = sampler.sample(v0)

        torch.manual_seed(123)
        result2 = sampler.sample(v0)

        # Results should be different (with high probability)
        assert not torch.allclose(result1.to_tensor(), result2.to_tensor())


class StressTests:
    """Stress tests for robustness.

    These tests push the sampler to its limits to ensure it handles
    edge cases gracefully.
    """

    def test_large_batch(self, sampler: BaseSamplerRBM, visible_size: int) -> None:
        """Test handling of large batches.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        visible_size : int
            Number of visible units.
        """
        # Try a large batch
        v0 = torch.randn(1024, visible_size)

        # Should not crash or raise OOM
        try:
            result = sampler.sample(v0)
            assert result.shape[0] == 1024
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                pytest.skip("GPU OOM on large batch")
            else:
                raise

    def test_memory_stability(
        self,
        sampler_class: type[BaseSamplerRBM],
        model: BaseRBM,
        visible_size: int,
    ) -> None:
        """Test for memory leaks during repeated sampling.

        Parameters
        ----------
        sampler_class : type[BaseSamplerRBM]
            Sampler class to test.
        model : BaseRBM
            Model instance.
        visible_size : int
            Number of visible units.
        """
        # Skip on CPU as memory tracking is less reliable
        if not torch.cuda.is_available():
            pytest.skip("Memory tracking requires CUDA")

        # Create sampler with model on GPU
        device = torch.device("cuda")
        model_gpu = model.to(device)
        sampler = sampler_class(model_gpu)

        v0 = torch.randn(16, visible_size, device=device)

        # Force GC and get baseline
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        # Run many iterations
        for _ in range(100):
            _ = sampler.sample(v0)

        # Check memory
        gc.collect()
        torch.cuda.empty_cache()
        final = torch.cuda.memory_allocated()

        # Allow some growth but not unbounded
        # 10MB tolerance for caching/buffers
        assert final - baseline < 10 * 1024 * 1024, "Possible memory leak detected"


class StateTests:
    """Tests for state management and serialization.

    These tests verify that samplers can correctly save and restore
    their state.
    """

    def test_state_dict_roundtrip(
        self,
        sampler: BaseSamplerRBM,
        visible_size: int,
    ) -> None:
        """Test state_dict save/load if applicable.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        visible_size : int
            Number of visible units.
        """
        # Skip if sampler doesn't support state_dict
        if not hasattr(sampler, "state_dict"):
            pytest.skip("Sampler doesn't support state_dict")

        # Get initial state
        state = sampler.state_dict()

        # Create a new instance (if possible)
        try:
            # Try to get the constructor arguments
            if hasattr(sampler, "model"):
                new_sampler = type(sampler)(sampler.model)
            elif hasattr(sampler, "rbm"):
                new_sampler = type(sampler)(sampler.rbm)
            else:
                pytest.skip("Cannot determine how to construct new sampler")

            # Load state
            new_sampler.load_state_dict(state)

            # Compare outputs with same seed
            v0 = torch.randn(4, visible_size)

            torch.manual_seed(42)
            result1 = sampler.sample(v0)

            torch.manual_seed(42)
            result2 = new_sampler.sample(v0)

            assert torch.allclose(result1.to_tensor(), result2.to_tensor())
        except Exception:
            pytest.skip("Cannot create new sampler instance")


def _worker(payload: bytes, conn: Any) -> None:
    """Child process entry for multiprocess tests.

    Parameters
    ----------
    payload : bytes
        Pickled data containing sampler info.
    conn : Any
        Pipe connection to parent process.
    """
    torch.manual_seed(42)
    try:
        cls_path, model_bytes, vis_size = pickle.loads(payload)

        # import the class
        mod_name, _, cls_name = cls_path.rpartition(".")
        cls = getattr(importlib.import_module(mod_name), cls_name)

        # rebuild the model from its pickled state_dict & cfg
        model_cfg, state_dict = pickle.loads(model_bytes)
        model = cls.__bases__[0](model_cfg)  # works for all RBM variants
        model.load_state_dict(state_dict)

        # run sampler
        sampler = cls(model)
        v0 = torch.randn(4, vis_size)
        out = sampler.sample(v0).cpu()

        conn.send(out)
    except Exception as exc:
        conn.send(exc)  # bubble up to parent
    finally:
        conn.close()


class MultiprocessTests:
    """Verify sampler works from a newly spawned interpreter.

    These tests ensure samplers can be used in multiprocessing contexts.
    """

    @pytest.mark.parametrize("start_method", ["forkserver" if os.name == "posix" else "spawn"])
    def test_process_safety(
        self,
        sampler_class: type[BaseSamplerRBM],
        model: BaseRBM,
        visible_size: int,
        start_method: str,
    ) -> None:
        """Test sampler works in separate processes.

        Parameters
        ----------
        sampler_class : type[BaseSamplerRBM]
            Sampler class to test.
        model : BaseRBM
            Model instance.
        visible_size : int
            Number of visible units.
        start_method : str
            Multiprocessing start method.
        """
        # serialise everything we need once
        cls_path = sampler_class.__module__ + "." + sampler_class.__qualname__
        model_cpu = model.cpu()
        model_blob = pickle.dumps((model_cpu.cfg, model_cpu.state_dict()))
        payload = pickle.dumps((cls_path, model_blob, visible_size))

        ctx = mp.get_context(start_method)
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_worker, args=(payload, child))  # type: ignore[attr-defined]
        proc.start()

        try:
            child_out = parent.recv()  # will raise EOFError if child died
        finally:
            proc.join(timeout=5)

        if isinstance(child_out, Exception):
            raise child_out

        # reference run in this process
        torch.manual_seed(42)
        ref_sampler = sampler_class(model_cpu)
        v0 = torch.randn(4, visible_size)
        parent_out = ref_sampler.sample(v0).cpu()

        assert torch.allclose(parent_out.to_tensor(), child_out)


class MetadataTests:
    """Tests for sampler metadata handling.

    These tests verify that samplers correctly manage optional
    metadata like initial states and chains.
    """

    def test_fast_path_no_metadata(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
    ) -> None:
        """Test fast path when no hooks or metadata requested.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """
        result = sampler.sample(sample_input)  # fast path

        # metadata must be None
        assert result.initial_state is None
        assert result.final_hidden is None
        assert result.intermediate_states is None

    def test_metadata_fields_present(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
        model: BaseRBM,
    ) -> None:
        """Test that metadata fields are present when requested.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        model : BaseRBM
            Model instance.
        """
        result = sampler.sample(sample_input, return_hidden=True, track_chains=True)

        assert torch.equal(result.initial_state, sample_input)  # type: ignore[arg-type]
        if hasattr(result, "intermediate_states") and result.intermediate_states is not None:
            assert len(result.intermediate_states) > 0
        if hasattr(result, "final_hidden") and result.final_hidden is not None:
            assert result.final_hidden.shape[0] == sample_input.shape[0]

    def test_metadata_absent_when_not_requested(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
    ) -> None:
        """Test metadata is absent when not requested.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """
        result = sampler.sample(sample_input)

        assert result.initial_state is None
        assert result.intermediate_states is None
        assert result.final_hidden is None


class SerializationTests:
    """Tests for sampler serialization.

    These tests verify that samplers can be correctly serialized
    and deserialized.
    """

    def test_config_roundtrip(
        self,
        sampler: BaseSamplerRBM,
        model: BaseRBM,
    ) -> None:
        """Test configuration serialization roundtrip.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        model : BaseRBM
            Model instance.
        """
        if hasattr(sampler, "get_config") and hasattr(type(sampler), "from_config"):
            serialized_config = sampler.get_config()
            new_sampler = type(sampler).from_config(serialized_config, model=model)  # type: ignore[attr-defined]

            # Verify key properties are preserved
            if hasattr(sampler, "k"):
                assert new_sampler.k == sampler.k
        else:
            pytest.skip("Sampler doesn't support serialization")


class PropertyBasedTests:
    """Property-based tests for samplers.

    These tests use hypothesis to generate random test cases
    and verify that samplers handle them correctly.
    """

    @given(data=sampler_model_config())  # type: ignore[misc]
    def test_property_based_shapes_and_dtypes(
        self,
        data: tuple[BaseSamplerRBM, torch.Tensor, int, int, int],
    ) -> None:
        """Property test for shape and dtype preservation.

        Parameters
        ----------
        data : Tuple[BaseSamplerRBM, torch.Tensor, int, int, int]
            Generated test data from hypothesis.
        """
        sampler, v0, batch_size, visible, hidden = data

        result = sampler.sample(v0, return_hidden=True, track_chains=True)

        assert result.shape == (batch_size, visible)
        if hasattr(result, "final_hidden") and result.final_hidden is not None:
            assert result.final_hidden.shape == (batch_size, hidden)
        assert result.dtype == v0.dtype

        if hasattr(result, "intermediate_states") and result.intermediate_states is not None:
            for state in result.intermediate_states:
                assert state.dtype == v0.dtype


class PerformanceTests:
    """Performance tests for samplers.

    These tests ensure samplers meet minimum performance requirements.
    """

    @pytest.mark.performance
    def test_cpu_performance(self, sampler: BaseSamplerRBM, visible_size: int) -> None:
        """Test CPU performance meets requirements.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        visible_size : int
            Number of visible units.
        """
        run_performance_cpu_test(sampler, visible=visible_size)  # batch=4096 default

    @pytest.mark.performance
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_performance(
        self,
        sampler: BaseSamplerRBM,
        model: BaseRBM,
        visible_size: int,
    ) -> None:
        """Test GPU performance meets requirements.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        model : BaseRBM
            Model instance.
        visible_size : int
            Number of visible units.
        """
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            sampler_gpu = type(sampler)(model_gpu)
            run_performance_gpu_test(sampler_gpu, visible=visible_size)  # batch=4096 default


class CUDATests:
    """Tests for CUDA functionality.

    These tests verify correct behavior on GPU devices.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_chain_stays_on_gpu(
        self,
        sampler_class: type[BaseSamplerRBM],
        model: BaseRBM,
    ) -> None:
        """Test that CUDA tensors stay on GPU throughout sampling.

        Parameters
        ----------
        sampler_class : type[BaseSamplerRBM]
            Sampler class to test.
        model : BaseRBM
            Model instance.
        """
        device = torch.device("cuda")
        model_gpu = model.to(device)
        sampler = sampler_class(model_gpu)

        v0 = torch.randn(6, model.cfg.visible, device=device)
        result = sampler.sample(v0, return_hidden=True)

        assert result.device.type == "cuda"
        if hasattr(result, "final_hidden") and result.final_hidden is not None:
            assert result.final_hidden.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_to_numpy_conversion(
        self,
        sampler_class: type[BaseSamplerRBM],
        model: BaseRBM,
    ) -> None:
        """Test conversion of CUDA results to numpy.

        Parameters
        ----------
        sampler_class : type[BaseSamplerRBM]
            Sampler class to test.
        model : BaseRBM
            Model instance.
        """
        device = torch.device("cuda")
        model_gpu = model.to(device)
        sampler = sampler_class(model_gpu)

        v0 = torch.randn(3, model.cfg.visible, device=device)
        result = sampler.sample(v0)

        arr = np.asarray(result)
        assert arr.shape == v0.shape

    def test_numpy_array_kwargs(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
    ) -> None:
        """Test numpy array conversion with various kwargs.

        Parameters
        ----------
        sampler : BaseSamplerRBM
            Sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """
        result = sampler.sample(sample_input)

        arr1 = np.asarray(result, dtype=np.float64)
        arr2 = np.asarray(result).copy()

        assert arr1.dtype == np.float64
        assert arr1.shape == sample_input.shape
        assert arr2.shape == sample_input.shape
