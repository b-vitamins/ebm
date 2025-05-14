"""Comprehensive tests for the Contrastive Divergence (CD) sampler.

This module provides a comprehensive test suite for the CD sampler,
including both CD-specific tests and generic sampler tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
import torch

from ebm.rbm.model import (
    BernoulliRBM,
    BernoulliRBMConfig,
    CenteredBernoulliRBM,
    CenteredBernoulliRBMConfig,
)
from ebm.rbm.model.base import BaseRBM, RBMConfig
from ebm.rbm.sampler.cd import CDSampler
from tests.helper import (
    MockRBM,
)
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
    """Provide RBM class and config class pairs.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest request object containing test parameters.

    Returns
    -------
    tuple[type[BaseRBM], type[RBMConfig]]
        RBM class and its corresponding config class.
    """
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def rbm_class(
    rbm_model_and_config: tuple[type[BaseRBM], type[RBMConfig]],
) -> type[BaseRBM]:
    """Extract RBM class from parametrized fixture.

    Parameters
    ----------
    rbm_model_and_config : tuple[type[BaseRBM], type[RBMConfig]]
        Tuple of RBM class and config class.

    Returns
    -------
    type[BaseRBM]
        RBM class to use for testing.
    """
    return rbm_model_and_config[0]


@pytest.fixture
def rbm_config_class(
    rbm_model_and_config: tuple[type[BaseRBM], type[RBMConfig]],
) -> type[RBMConfig]:
    """Extract RBM config class from parametrized fixture.

    Parameters
    ----------
    rbm_model_and_config : tuple[type[BaseRBM], type[RBMConfig]]
        Tuple of RBM class and config class.

    Returns
    -------
    type[RBMConfig]
        RBM config class to use for testing.
    """
    return rbm_model_and_config[1]


@pytest.fixture
def rbm_config(rbm_config_class: type[RBMConfig]) -> RBMConfig:
    """Create RBM config instance.

    Parameters
    ----------
    rbm_config_class : type[RBMConfig]
        RBM config class to instantiate.

    Returns
    -------
    RBMConfig
        Configuration instance with standard test dimensions.
    """
    return rbm_config_class(visible=10, hidden=8)


@pytest.fixture
def sampler_class() -> type[CDSampler]:
    """Provide the CD sampler class.

    Returns
    -------
    type[CDSampler]
        CD sampler class.
    """
    return CDSampler


@pytest.fixture
def visible_size() -> int:
    """Standard visible dimension for tests.

    Returns
    -------
    int
        Number of visible units.
    """
    return 10


@pytest.fixture
def hidden_size() -> int:
    """Standard hidden dimension for tests.

    Returns
    -------
    int
        Number of hidden units.
    """
    return 8


@pytest.fixture
def model(rbm_class: type[BaseRBM], rbm_config: RBMConfig) -> BaseRBM:
    """Create a test model using parametrized RBM class.

    Parameters
    ----------
    rbm_class : type[BaseRBM]
        RBM class to instantiate.
    rbm_config : RBMConfig
        Configuration for the RBM.

    Returns
    -------
    BaseRBM
        Instantiated RBM model.
    """
    return rbm_class(rbm_config)


@pytest.fixture
def sampler(model: BaseRBM) -> CDSampler:
    """Create a CD sampler with default k=1.

    Parameters
    ----------
    model : BaseRBM
        RBM model to use with the sampler.

    Returns
    -------
    CDSampler
        CD sampler instance.
    """
    return CDSampler(model, k=1)


@pytest.fixture
def sample_input(visible_size: int) -> torch.Tensor:
    """Standard input for sampling tests.

    Parameters
    ----------
    visible_size : int
        Number of visible units.

    Returns
    -------
    torch.Tensor
        Random input tensor for testing.
    """
    return torch.randn(16, visible_size)


class TestCDSpecific:
    """Tests specific to CD sampler implementation.

    These tests verify CD-specific functionality like the k parameter
    and reconstruction quality.
    """

    def test_k_validation(self, model: BaseRBM) -> None:
        """Test that k must be at least 1.

        Parameters
        ----------
        model : BaseRBM
            RBM model instance.
        """
        with pytest.raises(ValueError, match="at least 1"):
            CDSampler(model=model, k=0)

        with pytest.raises(ValueError, match="at least 1"):
            CDSampler(model=model, k=-5)

    def test_default_k(self, model: BaseRBM) -> None:
        """Test default k value.

        Parameters
        ----------
        model : BaseRBM
            RBM model instance.
        """
        sampler = CDSampler(model=model)
        assert sampler.k == 1

    def test_repr_contains_k(self, model: BaseRBM) -> None:
        """Test string representation includes k value.

        Parameters
        ----------
        model : BaseRBM
            RBM model instance.
        """
        sampler = CDSampler(model=model, k=7)
        assert "k=7" in repr(sampler)

    def test_k_parameter_effects(self, model: BaseRBM) -> None:
        """Test different k values produce expected behavior.

        Parameters
        ----------
        model : BaseRBM
            RBM model instance.
        """
        v0 = torch.randn(4, model.cfg.visible)

        def count_hook_factory(
            counts: list[int],
        ) -> Callable[[BaseSamplerRBM, int, torch.Tensor, torch.Tensor, torch.Tensor | None], None]:
            """Create a hook that appends steps to the given list."""

            def hook(
                sampler: BaseSamplerRBM,
                step: int,
                v: torch.Tensor,
                h: torch.Tensor,
                beta: torch.Tensor | None,
            ) -> None:
                counts.append(step)

            return hook

        for k in [1, 3, 5, 10]:
            sampler = CDSampler(model, k=k)
            assert sampler.k == k

            step_counts: list[int] = []
            handle = sampler.register_sampling_hook(count_hook_factory(step_counts))
            try:
                sampler.sample(v0)
                assert len(step_counts) == k
                assert step_counts == list(range(k))
            finally:
                handle.remove()

    def test_reconstruction_improves_with_k(self, model: BaseRBM) -> None:
        """Test reconstruction quality improves with larger k.

        Parameters
        ----------
        model : BaseRBM
            RBM model instance.
        """
        torch.manual_seed(42)
        batch_size = 32
        v_data = torch.zeros(batch_size, model.cfg.visible)
        v_data[:, : model.cfg.visible // 2] = 1.0

        reconstruction_errors = {}

        for k in [1, 3, 5, 10]:
            sampler = CDSampler(model, k=k)
            v_recon = sampler.sample(v_data)
            error = (v_data - v_recon).pow(2).mean().item()
            reconstruction_errors[k] = error

        errors_list = [reconstruction_errors[k] for k in [1, 3, 5, 10]]

        # Allow some tolerance since reconstruction isn't guaranteed to be monotonic
        for i in range(len(errors_list) - 1):
            assert errors_list[i + 1] <= errors_list[i] * 1.2

    def test_parallel_tempering_shapes(self) -> None:
        """Test CD with parallel tempering preserves shapes."""
        cfg = RBMConfig(visible=6, hidden=4)
        model = MockRBM(cfg)
        sampler = CDSampler(model, k=3)

        num_replicas = 4
        batch_size = 2
        betas = torch.tensor([0.1, 0.5, 0.8, 1.0])

        v0 = torch.randn(batch_size, num_replicas, cfg.visible)
        result = sampler.sample(v0, beta=betas)

        assert result.shape == (batch_size, num_replicas, cfg.visible)

    def test_k_mutation(self, model: BaseRBM) -> None:
        """Test that k can be modified after construction.

        Parameters
        ----------
        model : BaseRBM
            RBM model instance.
        """
        sampler = CDSampler(model, k=3)
        sampler.k = 9

        v0 = torch.randn(5, model.cfg.visible)
        steps: list[int] = []

        handle = sampler.register_sampling_hook(lambda s, step, vk, hk, beta: steps.append(step))
        try:
            sampler.sample(v0)
            assert steps == list(range(9))
        finally:
            handle.remove()

        assert "k=9" in repr(sampler)


class TestCDSamplerHooks(HookTests):
    """CD sampler hook tests with k-specific additions."""

    def test_exact_step_count_default(
        self,
        sampler: CDSampler,
        sample_input: torch.Tensor,
    ) -> None:
        """Test exact step count for default k=1.

        Parameters
        ----------
        sampler : CDSampler
            CD sampler instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        """
        steps: list[int] = []

        def step_hook(
            sampler: BaseSamplerRBM,
            step: int,
            vk: torch.Tensor,
            hk: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            steps.append(step)

        handle = sampler.register_sampling_hook(step_hook)
        try:
            sampler.sample(sample_input)
            assert steps == [0]
        finally:
            handle.remove()

    @pytest.mark.parametrize("k", [1, 2, 4, 10])
    def test_exact_step_count_k(
        self,
        model: BaseRBM,
        sample_input: torch.Tensor,
        k: int,
    ) -> None:
        """Test exact step count for various k values.

        Parameters
        ----------
        model : BaseRBM
            RBM model instance.
        sample_input : torch.Tensor
            Input tensor for sampling.
        k : int
            Number of CD steps.
        """
        sampler = CDSampler(model, k=k)
        steps: list[int] = []

        def step_hook(
            s: BaseSamplerRBM,
            step: int,
            vk: torch.Tensor,
            hk: torch.Tensor,
            beta: torch.Tensor | None,
        ) -> None:
            steps.append(step)

        handle = sampler.register_sampling_hook(step_hook)
        try:
            sampler.sample(sample_input)
            assert steps == list(range(k))
        finally:
            handle.remove()


class TestCDSamplerPerformance(PerformanceTests):
    """CD-sampler performance tests.

    These tests verify performance characteristics specific to
    the CD sampler.
    """

    pass


# Test classes inheriting from generic mixins
class TestCDSamplerShape(ShapeTests):
    """Shape tests for CD sampler."""

    pass


class TestCDSamplerDeviceAutograd(DeviceAutogradTests):
    """Device and autograd tests for CD sampler."""

    pass


class TestCDSamplerDeterminism(DeterminismTests):
    """Determinism tests for CD sampler."""

    pass


class TestCDSamplerStress(StressTests):
    """Stress tests for CD sampler."""

    pass


class TestCDSamplerState(StateTests):
    """State management tests for CD sampler."""

    pass


class TestCDSamplerMetadata(MetadataTests):
    """Metadata tests for CD sampler."""

    pass


class TestCDSamplerPropertyBased(PropertyBasedTests):
    """Property-based tests for CD sampler."""

    pass


class TestCDSamplerSerialization(SerializationTests):
    """Serialization tests for CD sampler."""

    pass


class TestCDSamplerCUDA(CUDATests):
    """CUDA tests for CD sampler."""

    pass


class TestCDSamplerBeta(BetaTests):
    """Beta parameter tests for CD sampler."""

    pass


class TestCDSamplerStatistical(StatisticalTests):
    """Statistical tests for CD sampler."""

    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
