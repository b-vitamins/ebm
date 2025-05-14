"""Comprehensive tests for the Persistent Contrastive Divergence (PCD) sampler.

This module provides a comprehensive test suite for the PCD sampler,
including both PCD-specific tests and generic sampler tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from ebm.rbm.model import (
    BernoulliRBM,
    BernoulliRBMConfig,
    CenteredBernoulliRBM,
    CenteredBernoulliRBMConfig,
)
from ebm.rbm.model.base import BaseRBM, RBMConfig
from ebm.rbm.sampler.pcd import PCDSampler
from tests.helper import MockRBM
from tests.rbm_sampler_assertions import (
    BetaTests,
    CUDATests,
    DeterminismTests,
    DeviceAutogradTests,
    HookTests,
    MetadataTests,
    PerformanceTests,
    PropertyBasedTests,
    SerializationTests,
    ShapeTests,
    StateTests,
    StatisticalTests,
    StressTests,
)

if TYPE_CHECKING:
    from ebm.rbm.sampler.base import BaseSamplerRBM


# RBM models and their configs for parametrization
RBM_MODELS: list[tuple[type[BaseRBM], type[RBMConfig]]] = [
    (MockRBM, RBMConfig),
    (BernoulliRBM, BernoulliRBMConfig),
    (CenteredBernoulliRBM, CenteredBernoulliRBMConfig),
]


@pytest.fixture(params=RBM_MODELS)
def rbm_model_and_config(
    request: pytest.FixtureRequest,
) -> tuple[type[BaseRBM], type[RBMConfig]]:
    """Provide RBM class and config class pairs."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def rbm_class(
    rbm_model_and_config: tuple[type[BaseRBM], type[RBMConfig]],
) -> type[BaseRBM]:
    """Extract RBM class from parametrized fixture."""
    return rbm_model_and_config[0]


@pytest.fixture
def rbm_config_class(
    rbm_model_and_config: tuple[type[BaseRBM], type[RBMConfig]],
) -> type[RBMConfig]:
    """Extract RBM config class from parametrized fixture."""
    return rbm_model_and_config[1]


@pytest.fixture
def rbm_config(rbm_config_class: type[RBMConfig]) -> RBMConfig:
    """Create RBM config instance."""
    return rbm_config_class(visible=10, hidden=8)


@pytest.fixture
def sampler_class() -> type[PCDSampler]:
    """Provide the PCD sampler class."""
    return PCDSampler


@pytest.fixture
def visible_size() -> int:
    """Standard visible dimension for tests."""
    return 10


@pytest.fixture
def hidden_size() -> int:
    """Standard hidden dimension for tests."""
    return 8


@pytest.fixture
def model(rbm_class: type[BaseRBM], rbm_config: RBMConfig) -> BaseRBM:
    """Create a test model using parametrized RBM class."""
    return rbm_class(rbm_config)


@pytest.fixture
def sampler(model: BaseRBM) -> PCDSampler:
    """Create a PCD sampler with default parameters."""
    return PCDSampler(model, k=1, num_chains=20)


@pytest.fixture
def sample_input(visible_size: int) -> torch.Tensor:
    """Standard input for sampling tests."""
    return torch.randn(16, visible_size)


class TestPCDSpecific:
    """Tests specific to PCD sampler implementation.

    These tests verify PCD-specific functionality like persistent chains,
    initialization methods, and chain management.
    """

    def test_num_chains_validation(self, model: BaseRBM) -> None:
        """Test that num_chains must be at least 1."""
        with pytest.raises(ValueError, match="at least 1"):
            PCDSampler(model=model, num_chains=0)

        with pytest.raises(ValueError, match="at least 1"):
            PCDSampler(model=model, num_chains=-5)

    def test_init_method_validation(self, model: BaseRBM) -> None:
        """Test that init_method must be valid."""
        with pytest.raises(ValueError, match="init_method must be one of"):
            PCDSampler(model=model, init_method="invalid")

    def test_default_parameters(self, model: BaseRBM) -> None:
        """Test default parameter values."""
        sampler = PCDSampler(model=model)
        assert sampler.k == 1
        assert sampler.num_chains == 100
        assert sampler.init_method == "uniform"
        assert sampler.persistent_chains is None
        assert not sampler.chains_initialized

    def test_repr_contains_parameters(self, model: BaseRBM) -> None:
        """Test string representation includes all parameters."""
        sampler = PCDSampler(model=model, k=5, num_chains=50, init_method="data")
        repr_str = repr(sampler)
        assert "k=5" in repr_str
        assert "num_chains=50" in repr_str
        assert "init_method='data'" in repr_str
        assert "initialized=False" in repr_str

        # Initialize chains
        v0 = torch.randn(10, model.cfg.visible)
        sampler.sample(v0)
        repr_str = repr(sampler)
        assert "initialized=True" in repr_str

    def test_chains_initialization_uniform(self, model: BaseRBM) -> None:
        """Test uniform chain initialization."""
        sampler = PCDSampler(model=model, num_chains=30, init_method="uniform")
        v0 = torch.randn(10, model.cfg.visible)

        # First sample should initialize chains
        assert sampler.persistent_chains is None
        _ = sampler.sample(v0)

        assert sampler.chains_initialized
        assert sampler.persistent_chains is not None
        assert sampler.persistent_chains.shape == (30, model.cfg.visible)

        # Check values are in [0, 1] for uniform initialization
        assert torch.all((sampler.persistent_chains >= 0) & (sampler.persistent_chains <= 1))

    def test_chains_initialization_data(self, model: BaseRBM) -> None:
        """Test data-based chain initialization."""
        sampler = PCDSampler(model=model, num_chains=20, init_method="data", k=1)

        # Test with more data than chains
        torch.manual_seed(42)  # Fix random seed for reproducibility
        v0_large = torch.randn(50, model.cfg.visible)

        # Initialize chains - should NOT run any sampling steps
        _ = sampler._sample(v0_large)

        # After just initialization, chains should contain exact data
        assert sampler.persistent_chains is not None
        assert sampler.persistent_chains.shape == (20, model.cfg.visible)
        # First 20 samples should match exactly (no sampling steps run)
        assert torch.allclose(sampler.persistent_chains, v0_large[:20], atol=1e-6)

        # Reset and test with less data than chains
        sampler.reset_chains()
        v0_small = torch.randn(5, model.cfg.visible)
        _ = sampler._sample(v0_small)

        assert sampler.persistent_chains is not None
        assert sampler.persistent_chains.shape == (20, model.cfg.visible)
        # Data should be repeated
        for i in range(20):
            assert torch.allclose(sampler.persistent_chains[i], v0_small[i % 5], atol=1e-6)

    def test_chains_initialization_model(self, model: BaseRBM) -> None:
        """Test model-based chain initialization."""
        sampler = PCDSampler(model=model, num_chains=25, init_method="model", k=3)
        v0 = torch.randn(10, model.cfg.visible)

        # First sample should initialize chains by sampling from model
        _ = sampler.sample(v0)

        assert sampler.chains_initialized
        assert sampler.persistent_chains is not None
        assert sampler.persistent_chains.shape == (25, model.cfg.visible)

    def test_persistent_chains_evolution(self, model: BaseRBM) -> None:
        """Test that chains evolve across multiple sampling calls."""
        sampler = PCDSampler(model=model, num_chains=15, k=1)
        v0 = torch.randn(10, model.cfg.visible)

        # Initialize chains
        _ = sampler.sample(v0)
        initial_chains = sampler.persistent_chains.clone()

        # Sample again - chains should evolve
        _ = sampler.sample(v0)
        evolved_chains = sampler.persistent_chains

        # Chains should have changed
        assert not torch.allclose(initial_chains, evolved_chains)

        # Multiple iterations should continue evolving
        for _ in range(5):
            prev_chains = sampler.persistent_chains.clone()
            _ = sampler.sample(v0)
            assert not torch.allclose(prev_chains, sampler.persistent_chains)

    def test_chains_reset(self, model: BaseRBM) -> None:
        """Test chain reset functionality."""
        sampler = PCDSampler(model=model, num_chains=10)
        v0 = torch.randn(5, model.cfg.visible)

        # Initialize and evolve chains
        _ = sampler.sample(v0)
        assert sampler.chains_initialized
        assert sampler.persistent_chains is not None

        # Reset chains
        sampler.reset_chains()
        assert not sampler.chains_initialized
        assert sampler.persistent_chains is None

        # Next sample should reinitialize
        _ = sampler.sample(v0)
        assert sampler.chains_initialized
        assert sampler.persistent_chains is not None

    def test_get_and_set_chains(self, model: BaseRBM) -> None:
        """Test getting and setting chain states."""
        sampler = PCDSampler(model=model, num_chains=20)

        # Get chains before initialization
        assert sampler.get_chains() is None

        # Initialize chains
        v0 = torch.randn(10, model.cfg.visible)
        _ = sampler.sample(v0)

        # Get chains
        chains = sampler.get_chains()
        assert chains is not None
        assert chains.shape == (20, model.cfg.visible)

        # Set chains to custom values
        custom_chains = torch.ones(20, model.cfg.visible) * 0.5
        sampler.set_chains(custom_chains)

        assert sampler.chains_initialized
        assert torch.allclose(sampler.persistent_chains, custom_chains)

        # Test shape validation
        with pytest.raises(ValueError, match="First dimension must be"):
            sampler.set_chains(torch.randn(30, model.cfg.visible))

    def test_parallel_tempering_shapes(self) -> None:
        """Test PCD with parallel tempering preserves shapes correctly."""
        cfg = RBMConfig(visible=6, hidden=4)
        model = MockRBM(cfg)
        sampler = PCDSampler(model, k=1, num_chains=10)

        num_replicas = 4
        batch_size = 8
        betas = torch.tensor([0.1, 0.5, 0.8, 1.0])

        # Input with replica dimension
        v0 = torch.randn(batch_size, num_replicas, cfg.visible)
        result = sampler.sample(v0, beta=betas)

        # Check result shape matches num_chains, not batch_size
        assert result.shape == (10, num_replicas, cfg.visible)

        # Check persistent chains shape
        assert sampler.persistent_chains is not None
        assert sampler.persistent_chains.shape == (10, num_replicas, cfg.visible)

    def test_device_consistency(self, model: BaseRBM) -> None:
        """Test that chains maintain correct device placement."""
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            sampler = PCDSampler(model_gpu, num_chains=15)

            v0 = torch.randn(10, model.cfg.visible).cuda()
            _ = sampler.sample(v0)

            assert sampler.persistent_chains.device.type == "cuda"

            # Chains should stay on GPU across samples
            _ = sampler.sample(v0)
            assert sampler.persistent_chains.device.type == "cuda"

    def test_dtype_consistency(self, rbm_class: type[BaseRBM], rbm_config: RBMConfig) -> None:
        """Test that chains maintain correct dtype."""
        for dtype in [torch.float32, torch.float64]:
            config = type(rbm_config)(
                visible=rbm_config.visible, hidden=rbm_config.hidden, dtype=dtype
            )
            model = rbm_class(config)
            sampler = PCDSampler(model, num_chains=10)

            v0 = torch.randn(5, model.cfg.visible, dtype=dtype)
            _ = sampler.sample(v0)

            assert sampler.persistent_chains.dtype == dtype

    def test_memory_efficiency_with_large_chains(self, model: BaseRBM) -> None:
        """Test memory management with large chain counts."""
        sampler = PCDSampler(model, num_chains=1000, enable_memory_safe=True)
        v0 = torch.randn(5, model.cfg.visible)

        # Should not crash with large number of chains
        _ = sampler.sample(v0)
        assert sampler.persistent_chains.shape[0] == 1000

    def test_hook_sees_chain_evolution(self, model: BaseRBM) -> None:
        """Test that hooks can observe chain evolution."""
        sampler = PCDSampler(model, k=3, num_chains=10)
        v0 = torch.randn(5, model.cfg.visible)

        chain_states: list[torch.Tensor] = []

        def capture_hook(
            s: BaseSamplerRBM,
            step: int,
            vk: torch.Tensor,
            hk: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            chain_states.append(vk.clone())

        handle = sampler.register_sampling_hook(capture_hook)
        try:
            _ = sampler.sample(v0)

            # Should have captured k states
            assert len(chain_states) == 3

            # Each state should have num_chains samples
            for state in chain_states:
                assert state.shape[0] == 10

            # States should evolve
            for i in range(len(chain_states) - 1):
                assert not torch.allclose(chain_states[i], chain_states[i + 1])
        finally:
            handle.remove()

    def test_chains_independent_of_data_batch_size(self, model: BaseRBM) -> None:
        """Test that chain count is independent of data batch size."""
        sampler = PCDSampler(model, num_chains=25)

        # Initialize with small batch
        v0_small = torch.randn(5, model.cfg.visible)
        result_small = sampler.sample(v0_small)
        assert result_small.shape[0] == 25

        # Sample with large batch - output should still be num_chains
        v0_large = torch.randn(100, model.cfg.visible)
        result_large = sampler.sample(v0_large)
        assert result_large.shape[0] == 25

    def test_gradient_behavior_with_persistent_chains(self, model: BaseRBM) -> None:
        """Test that persistent chains are properly detached from computation graph."""
        sampler = PCDSampler(model, num_chains=10)

        # Initialize chains
        v0 = torch.randn(5, model.cfg.visible)
        _ = sampler.sample(v0)

        # Chains should not have gradient
        assert not sampler.persistent_chains.requires_grad
        assert sampler.persistent_chains.grad_fn is None

        # Even with gradient-enabled input
        v0_grad = v0.clone().requires_grad_(True)
        result = sampler.sample(v0_grad)

        assert not sampler.persistent_chains.requires_grad
        assert result.grad_fn is None

    def test_chains_initialization_with_varying_dimensions(self, model: BaseRBM) -> None:
        """Test chain initialization with different dimensional patterns."""
        sampler = PCDSampler(model, num_chains=15)

        # Test 2D input
        v0_2d = torch.randn(10, model.cfg.visible)
        _ = sampler.sample(v0_2d)
        assert sampler.persistent_chains.shape == (15, model.cfg.visible)

        # Reset and test 3D input (with replica dimension)
        sampler.reset_chains()
        v0_3d = torch.randn(10, 4, model.cfg.visible)  # batch, replicas, visible
        _ = sampler.sample(v0_3d)
        assert sampler.persistent_chains.shape == (15, 4, model.cfg.visible)

        # Reset and test 4D input
        sampler.reset_chains()
        v0_4d = torch.randn(10, 2, 3, model.cfg.visible)
        _ = sampler.sample(v0_4d)
        assert sampler.persistent_chains.shape == (15, 2, 3, model.cfg.visible)


class TestPCDSamplerHooks(HookTests):
    """PCD sampler hook tests."""

    def test_hook_observes_persistent_chains(
        self,
        sampler: PCDSampler,
        sample_input: torch.Tensor,
    ) -> None:
        """Test that hooks observe the persistent chains, not the input data."""
        observed_shapes: list[torch.Size] = []

        def shape_hook(
            s: BaseSamplerRBM,
            step: int,
            vk: torch.Tensor,
            hk: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            observed_shapes.append(vk.shape)

        handle = sampler.register_sampling_hook(shape_hook)
        try:
            # Sample with different batch sizes
            v1 = torch.randn(10, sampler.model.cfg.visible)
            _ = sampler.sample(v1)

            v2 = torch.randn(50, sampler.model.cfg.visible)
            _ = sampler.sample(v2)

            # All observed shapes should have num_chains as first dimension
            for shape in observed_shapes:
                assert shape[0] == sampler.num_chains
        finally:
            handle.remove()


class TestPCDSamplerPerformance(PerformanceTests):
    """PCD-sampler performance tests."""

    pass


# Test classes inheriting from generic mixins
class TestPCDSamplerShape(ShapeTests):
    """Shape tests for PCD sampler.

    Note: PCD always returns num_chains samples regardless of input batch size.
    """

    @pytest.mark.fast
    def test_output_shape(self, sampler: PCDSampler, sample_input: torch.Tensor) -> None:
        """Override to test PCD-specific behavior."""
        # Basic sample - PCD returns num_chains, not batch_size
        result = sampler.sample(sample_input)
        assert hasattr(result, "shape"), "Output must be tensor-like"
        assert result.shape[0] == sampler.num_chains, "Should return num_chains samples"

        # With metadata
        try:
            result_full = sampler.sample(sample_input, return_hidden=True, track_chains=True)
            assert result_full.shape[0] == sampler.num_chains

            if hasattr(result_full, "final_hidden") and result_full.final_hidden is not None:
                assert result_full.final_hidden.shape[0] == sampler.num_chains

            if (
                hasattr(result_full, "intermediate_states")
                and result_full.intermediate_states is not None
            ):
                for state in result_full.intermediate_states:
                    assert state.shape[0] == sampler.num_chains
        except TypeError:
            pass


class TestPCDSamplerDeviceAutograd(DeviceAutogradTests):
    """Device and autograd tests for PCD sampler."""

    pass


class TestPCDSamplerDeterminism(DeterminismTests):
    """Determinism tests for PCD sampler.

    Note: PCD has different determinism properties than CD because it maintains
    persistent chains across calls.
    """

    @pytest.mark.fast
    def test_fixed_seed_reproducibility(
        self,
        sampler: BaseSamplerRBM,
        visible_size: int,
    ) -> None:
        """Override determinism test for PCD's unique behavior.

        PCD maintains persistent chains, so we need to test determinism
        differently - two fresh samplers with the same seed should produce
        the same sequence of outputs when initialized with the same method.
        """
        # Create fresh samplers for each test
        model = sampler.model

        # Create input with fixed seed
        torch.manual_seed(42)
        v0 = torch.randn(8, visible_size)

        # First sampler - use data initialization for predictable behavior
        torch.manual_seed(42)
        sampler1 = PCDSampler(model, num_chains=10, init_method="data")
        result1_a = sampler1.sample(v0)
        result1_b = sampler1.sample(v0)  # Second call continues from chains

        # Second sampler with same seed and same initialization
        torch.manual_seed(42)
        sampler2 = PCDSampler(model, num_chains=10, init_method="data")
        result2_a = sampler2.sample(v0)
        result2_b = sampler2.sample(v0)  # Second call continues from chains

        # First calls should match
        assert torch.allclose(result1_a.to_tensor(), result2_a.to_tensor())
        # Second calls should match
        assert torch.allclose(result1_b.to_tensor(), result2_b.to_tensor())
        # But first and second calls should differ (chains evolved)
        assert not torch.allclose(result1_a.to_tensor(), result1_b.to_tensor())


class TestPCDSamplerStress(StressTests):
    """Stress tests for PCD sampler."""

    pass


class TestPCDSamplerState(StateTests):
    """State management tests for PCD sampler."""

    pass


class TestPCDSamplerMetadata(MetadataTests):
    """Metadata tests for PCD sampler."""

    @pytest.mark.fast
    def test_metadata_fields_present(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
        model: BaseRBM,
    ) -> None:
        """Override to test PCD-specific metadata behavior."""
        result = sampler.sample(sample_input, return_hidden=True, track_chains=True)

        # PCD doesn't track initial state from data
        assert result.initial_state is None or torch.equal(result.initial_state, sample_input)

        if hasattr(result, "intermediate_states") and result.intermediate_states is not None:
            assert len(result.intermediate_states) > 0
            # PCD returns num_chains samples
            if isinstance(sampler, PCDSampler):
                for state in result.intermediate_states:
                    assert state.shape[0] == sampler.num_chains

        if hasattr(result, "final_hidden") and result.final_hidden is not None:
            # PCD returns num_chains samples
            if isinstance(sampler, PCDSampler):
                assert result.final_hidden.shape[0] == sampler.num_chains
            else:
                assert result.final_hidden.shape[0] == sample_input.shape[0]


class TestPCDSamplerPropertyBased(PropertyBasedTests):
    """Property-based tests for PCD sampler."""

    pass


class TestPCDSamplerSerialization(SerializationTests):
    """Serialization tests for PCD sampler."""

    @pytest.mark.fast
    def test_config_roundtrip(
        self,
        sampler: PCDSampler,
        model: BaseRBM,
    ) -> None:
        """Test configuration serialization roundtrip including PCD-specific fields."""
        config = sampler.get_config()
        new_sampler = PCDSampler.from_config(config, model=model)

        # Verify all properties are preserved
        assert new_sampler.k == sampler.k
        assert new_sampler.num_chains == sampler.num_chains
        assert new_sampler.init_method == sampler.init_method


class TestPCDSamplerCUDA(CUDATests):
    """CUDA tests for PCD sampler."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.fast
    def test_cuda_to_numpy_conversion(
        self,
        sampler_class: type[BaseSamplerRBM],
        model: BaseRBM,
    ) -> None:
        """Override to handle PCD's different output shape."""
        device = torch.device("cuda")
        model_gpu = model.to(device)
        sampler = sampler_class(model_gpu)

        v0 = torch.randn(3, model.cfg.visible, device=device)
        result = sampler.sample(v0)

        arr = np.asarray(result)
        # PCD returns num_chains, not batch_size
        if isinstance(sampler, PCDSampler):
            assert arr.shape[0] == sampler.num_chains
        else:
            assert arr.shape == v0.shape

    @pytest.mark.fast
    def test_numpy_array_kwargs(
        self,
        sampler: BaseSamplerRBM,
        sample_input: torch.Tensor,
    ) -> None:
        """Override to handle PCD's different output shape."""
        result = sampler.sample(sample_input)

        arr1 = np.asarray(result, dtype=np.float64)
        arr2 = np.asarray(result).copy()

        assert arr1.dtype == np.float64
        # PCD returns num_chains, not batch_size
        if isinstance(sampler, PCDSampler):
            assert arr1.shape[0] == sampler.num_chains
            assert arr2.shape[0] == sampler.num_chains
        else:
            assert arr1.shape == sample_input.shape
            assert arr2.shape == sample_input.shape


class TestPCDSamplerBeta(BetaTests):
    """Beta parameter tests for PCD sampler."""

    @pytest.mark.fast
    def test_beta_shape_broadcasting(
        self,
        sampler_class: type[BaseSamplerRBM],
        visible_size: int,
        hidden_size: int,
    ) -> None:
        """Override to handle PCD's output shape behavior."""
        cfg = RBMConfig(visible=visible_size, hidden=hidden_size)
        model = MockRBM(cfg)
        sampler = sampler_class(model)

        # PCD specific parameters
        num_chains = sampler.num_chains

        # Test various beta shapes
        batch_size = 4
        num_replicas = 3

        test_cases: list[tuple[tuple[int, ...], torch.Tensor, tuple[int, ...], str]] = [
            # (v0_shape, beta, expected_output_shape, description)
            (
                (batch_size, visible_size),
                torch.tensor(0.5),
                (num_chains, visible_size),
                "scalar beta",
            ),
            (
                (batch_size, num_replicas, visible_size),
                torch.tensor([0.3, 0.6, 1.0]),
                (num_chains, num_replicas, visible_size),
                "vector beta",
            ),
            (
                (batch_size, num_replicas, visible_size),
                torch.tensor([0.3, 0.6, 1.0]).unsqueeze(0).unsqueeze(-1),
                (num_chains, num_replicas, visible_size),
                "shaped beta",
            ),
        ]

        for v0_shape, beta, expected_shape, desc in test_cases:
            v0 = torch.randn(v0_shape)
            result = sampler.sample(v0, beta=beta)
            assert result.shape == expected_shape, f"Shape mismatch for {desc}"

    @pytest.mark.fast
    def test_beta_result_shapes(
        self,
        sampler_class: type[BaseSamplerRBM],
        rbm_config: RBMConfig,
        rbm_class: type[BaseRBM],
    ) -> None:
        """Override to handle PCD's output shape behavior."""
        visible, hidden = 5, 4
        batch_size, replicas = 2, 4

        config = type(rbm_config)(visible=visible, hidden=hidden)
        model = rbm_class(config)
        sampler = sampler_class(model)

        beta = torch.linspace(0.1, 1.0, steps=replicas)
        v0 = torch.randn(batch_size, replicas, visible)
        result = sampler.sample(v0, beta=beta, return_hidden=True)

        # PCD returns num_chains samples, not batch_size
        assert result.shape == (sampler.num_chains, replicas, visible)
        if hasattr(result, "final_hidden") and result.final_hidden is not None:
            assert result.final_hidden.shape == (sampler.num_chains, replicas, hidden)


class TestPCDSamplerStatistical(StatisticalTests):
    """Statistical tests for PCD sampler."""

    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
