"""Unit tests for base sampling classes."""

from typing import Any
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

from ebm.models.base import EnergyBasedModel, LatentVariableModel
from ebm.sampling.base import (
    AnnealedSampler,
    GibbsSampler,
    GradientEstimator,
    MCMCSampler,
    Sampler,
    SamplerState,
)


class ConcreteSampler(Sampler):
    """Concrete sampler for testing."""

    def sample(
        self,
        model: EnergyBasedModel,
        init_state: Tensor,
        num_steps: int = 1,
        **kwargs: Any
    ) -> Tensor:
        # Simple implementation: add noise
        state = init_state
        for _ in range(num_steps):
            state = state + torch.randn_like(state) * 0.1
        self.state.num_steps += num_steps
        return state


class ConcreteGradientEstimator(GradientEstimator):
    """Concrete gradient estimator for testing."""

    def estimate_gradient(
        self,
        model: EnergyBasedModel,
        data: Tensor,
        **kwargs: Any
    ) -> dict[str, Tensor]:
        # Return mock gradients
        return {
            "param1": torch.randn(10, 10),
            "param2": torch.randn(10)
        }


class TestSamplerState:
    """Test SamplerState dataclass."""

    def test_initialization(self) -> None:
        """Test sampler state initialization."""
        state = SamplerState()

        assert state.chains is None
        assert state.num_steps == 0
        assert isinstance(state.metadata, dict)
        assert len(state.metadata) == 0

    def test_reset(self) -> None:
        """Test state reset."""
        state = SamplerState()

        # Modify state
        state.chains = torch.randn(10, 5)
        state.num_steps = 100
        state.metadata["key"] = "value"

        # Reset
        state.reset()

        assert state.chains is None
        assert state.num_steps == 0
        assert len(state.metadata) == 0


class TestSampler:
    """Test base Sampler class."""

    def test_initialization(self) -> None:
        """Test sampler initialization."""
        sampler = ConcreteSampler()

        assert sampler.name == "ConcreteSampler"
        assert isinstance(sampler.state, SamplerState)
        assert sampler.num_steps_taken == 0

        # With custom name
        sampler_named = ConcreteSampler(name="CustomSampler")
        assert sampler_named.name == "CustomSampler"

    def test_sample_method(self) -> None:
        """Test sampling functionality."""
        sampler = ConcreteSampler()
        model = Mock(spec=EnergyBasedModel)

        init_state = torch.randn(10, 5)
        samples = sampler.sample(model, init_state, num_steps=5)

        assert samples.shape == init_state.shape
        assert sampler.num_steps_taken == 5

        # Sample again
        sampler.sample(model, init_state, num_steps=3)
        assert sampler.num_steps_taken == 8

    def test_reset(self) -> None:
        """Test sampler reset."""
        sampler = ConcreteSampler()
        model = Mock(spec=EnergyBasedModel)

        # Run some sampling
        init_state = torch.randn(10, 5)
        sampler.sample(model, init_state, num_steps=10)

        assert sampler.num_steps_taken == 10

        # Reset
        sampler.reset()
        assert sampler.num_steps_taken == 0

    def test_get_diagnostics(self) -> None:
        """Test diagnostic information."""
        sampler = ConcreteSampler()

        diagnostics = sampler.get_diagnostics()
        assert "num_steps" in diagnostics
        assert "has_chains" in diagnostics
        assert diagnostics["num_steps"] == 0
        assert diagnostics["has_chains"] is False

        # After sampling
        model = Mock(spec=EnergyBasedModel)
        sampler.sample(model, torch.randn(5, 3), num_steps=10)

        diagnostics = sampler.get_diagnostics()
        assert diagnostics["num_steps"] == 10


class TestGibbsSampler:
    """Test GibbsSampler base class."""

    def test_initialization(self) -> None:
        """Test Gibbs sampler initialization."""
        sampler = GibbsSampler()

        assert sampler.block_gibbs is True
        assert sampler.name == "GibbsSampler"

        # Without block Gibbs
        sampler_no_block = GibbsSampler(block_gibbs=False)
        assert sampler_no_block.block_gibbs is False

    def test_gibbs_step(self) -> None:
        """Test single Gibbs step."""
        sampler = GibbsSampler()

        # Mock latent model
        model = Mock(spec=LatentVariableModel)
        model.sample_hidden.return_value = torch.rand(5, 10)
        model.sample_visible.return_value = torch.rand(5, 20)

        visible = torch.rand(5, 20)

        # Gibbs step starting from visible
        v_new, h_new = sampler.gibbs_step(model, visible)

        assert model.sample_hidden.called
        assert model.sample_visible.called
        assert v_new.shape == (5, 20)
        assert h_new.shape == (5, 10)

        # Starting from hidden
        model.reset_mock()
        v_new2, h_new2 = sampler.gibbs_step(model, visible, start_from='hidden')

        # Should call sample_hidden twice
        assert model.sample_hidden.call_count == 2
        assert model.sample_visible.call_count == 1

    def test_gibbs_sampling(self) -> None:
        """Test multi-step Gibbs sampling."""
        sampler = GibbsSampler()

        # Mock model
        model = Mock(spec=LatentVariableModel)
        counter = 0

        def mock_sample_hidden(v, **kwargs):
            nonlocal counter
            counter += 1
            return torch.rand_like(v) + counter * 0.01

        def mock_sample_visible(h, **kwargs):
            return torch.rand_like(h) + counter * 0.01

        model.sample_hidden.side_effect = mock_sample_hidden
        model.sample_visible.side_effect = mock_sample_visible

        init_state = torch.rand(10, 20)

        # Single step
        samples = sampler.sample(model, init_state, num_steps=1)
        assert samples.shape == init_state.shape
        assert sampler.num_steps_taken == 1

        # Multiple steps
        sampler.sample(model, init_state, num_steps=5)
        assert sampler.num_steps_taken == 6

        # Return all states
        all_samples = sampler.sample(
            model, init_state, num_steps=3, return_all=True
        )
        assert all_samples.shape == (4, 10, 20)  # init + 3 steps

    def test_invalid_model_type(self) -> None:
        """Test error on invalid model type."""
        sampler = GibbsSampler()

        # Non-latent model
        model = Mock(spec=EnergyBasedModel)
        init_state = torch.rand(5, 10)

        with pytest.raises(TypeError, match="Gibbs sampling requires LatentVariableModel"):
            sampler.sample(model, init_state)


class TestMCMCSampler:
    """Test MCMCSampler base class."""

    class ConcreteMCMC(MCMCSampler):
        """Concrete MCMC sampler for testing."""

        def transition_kernel(
            self,
            model: EnergyBasedModel,
            state: Tensor,
            **kwargs: Any
        ) -> tuple[Tensor, dict[str, Any]]:
            # Simple random walk
            proposal = state + torch.randn_like(state) * 0.1

            # Always accept
            metadata = {"accepted": True, "step_size": 0.1}
            return proposal, metadata

    def test_initialization(self) -> None:
        """Test MCMC sampler initialization."""
        sampler = self.ConcreteMCMC()
        assert sampler.num_chains is None

        sampler_chains = self.ConcreteMCMC(num_chains=10)
        assert sampler_chains.num_chains == 10

    def test_basic_sampling(self) -> None:
        """Test basic MCMC sampling."""
        sampler = self.ConcreteMCMC()
        model = Mock(spec=EnergyBasedModel)

        init_state = torch.randn(5, 10)
        samples = sampler.sample(model, init_state, num_steps=10)

        assert samples.shape == init_state.shape
        assert sampler.num_steps_taken == 10

    def test_burn_in(self) -> None:
        """Test burn-in functionality."""
        sampler = self.ConcreteMCMC()
        model = Mock(spec=EnergyBasedModel)

        init_state = torch.zeros(5, 10)

        # With burn-in
        samples = sampler.sample(
            model, init_state,
            num_steps=10,
            burn_in=5
        )

        # Should have done 15 total steps
        assert sampler.num_steps_taken == 15

        # Final state should be different from init
        assert not torch.allclose(samples, init_state)

    def test_thinning(self) -> None:
        """Test thinning functionality."""
        sampler = self.ConcreteMCMC()
        model = Mock(spec=EnergyBasedModel)

        init_state = torch.randn(5, 10)

        # Without return_all, thinning doesn't affect final sample
        samples = sampler.sample(
            model, init_state,
            num_steps=10,
            thin=2
        )
        assert samples.shape == init_state.shape

        # With return_all and thinning
        all_samples = sampler.sample(
            model, init_state,
            num_steps=10,
            thin=2,
            return_all=True
        )
        # Should keep every 2nd sample: 10 steps / 2 = 5 samples
        assert all_samples.shape == (5, 5, 10)

    def test_combined_options(self) -> None:
        """Test burn-in + thinning + return_all."""
        sampler = self.ConcreteMCMC()
        model = Mock(spec=EnergyBasedModel)

        init_state = torch.randn(3, 8)

        all_samples = sampler.sample(
            model, init_state,
            num_steps=20,
            burn_in=10,
            thin=5,
            return_all=True
        )

        # 20 steps, thin by 5 = 4 kept samples
        assert all_samples.shape == (4, 3, 8)
        assert sampler.num_steps_taken == 30  # 10 burn-in + 20 sampling


class TestGradientEstimator:
    """Test GradientEstimator base class."""

    def test_initialization(self) -> None:
        """Test gradient estimator initialization."""
        mock_sampler = Mock(spec=Sampler)
        estimator = ConcreteGradientEstimator(mock_sampler)

        assert estimator.sampler is mock_sampler

    def test_estimate_gradient(self) -> None:
        """Test gradient estimation."""
        mock_sampler = Mock(spec=Sampler)
        estimator = ConcreteGradientEstimator(mock_sampler)

        model = Mock(spec=EnergyBasedModel)
        data = torch.randn(32, 10)

        gradients = estimator.estimate_gradient(model, data)

        assert isinstance(gradients, dict)
        assert "param1" in gradients
        assert "param2" in gradients
        assert gradients["param1"].shape == (10, 10)
        assert gradients["param2"].shape == (10,)

    def test_compute_metrics(self) -> None:
        """Test metric computation."""
        mock_sampler = Mock(spec=Sampler)
        estimator = ConcreteGradientEstimator(mock_sampler)

        # Mock model
        model = Mock(spec=EnergyBasedModel)
        model.free_energy.side_effect = lambda x: torch.randn(x.shape[0])

        data = torch.randn(32, 10)
        samples = torch.randn(32, 10)

        metrics = estimator.compute_metrics(model, data, samples)

        assert "data_energy" in metrics
        assert "sample_energy" in metrics
        assert "energy_gap" in metrics
        assert "reconstruction_error" in metrics

        # All should be floats
        for value in metrics.values():
            assert isinstance(value, float)

    def test_compute_metrics_with_latent_model(self) -> None:
        """Test metrics with latent variable model."""
        mock_sampler = Mock(spec=Sampler)
        estimator = ConcreteGradientEstimator(mock_sampler)

        # Mock latent model
        model = Mock(spec=LatentVariableModel)
        model.free_energy.return_value = torch.tensor([1.0, 2.0, 3.0])
        model.reconstruct.return_value = torch.randn(3, 10)

        data = torch.randn(3, 10)
        samples = torch.randn(3, 10)

        metrics = estimator.compute_metrics(model, data, samples)

        # Should compute reconstruction error
        assert model.reconstruct.called
        assert metrics["reconstruction_error"] > 0


class TestAnnealedSampler:
    """Test AnnealedSampler base class."""

    class ConcreteAnnealed(AnnealedSampler):
        """Concrete annealed sampler for testing."""

        def sample(
            self,
            model: EnergyBasedModel,
            init_state: Tensor,
            num_steps: int = 1,
            **kwargs: Any
        ) -> Tensor:
            # Simple implementation
            return init_state + torch.randn_like(init_state) * 0.1

    def test_initialization(self) -> None:
        """Test annealed sampler initialization."""
        sampler = self.ConcreteAnnealed(
            num_temps=5,
            min_beta=0.1,
            max_beta=1.0
        )

        assert sampler.num_temps == 5
        assert sampler.min_beta == 0.1
        assert sampler.max_beta == 1.0
        assert hasattr(sampler, 'betas')
        assert len(sampler.betas) == 5

    def test_temperature_schedule(self) -> None:
        """Test temperature schedule creation."""
        sampler = self.ConcreteAnnealed(
            num_temps=10,
            min_beta=0.01,
            max_beta=1.0
        )

        # Check betas are in increasing order
        betas = sampler.betas
        assert len(betas) == 10
        assert betas[0] >= 0.01
        assert betas[-1] <= 1.0
        assert torch.all(betas[1:] >= betas[:-1])

        # Check geometric spacing
        log_betas = torch.log(betas)
        diffs = log_betas[1:] - log_betas[:-1]
        assert torch.allclose(diffs, diffs[0], rtol=1e-5)

    def test_temperatures_property(self) -> None:
        """Test temperature computation."""
        sampler = self.ConcreteAnnealed(
            num_temps=3,
            min_beta=0.5,
            max_beta=2.0
        )

        temps = sampler.temperatures

        # Temperatures should be inverse of betas
        expected = 1.0 / sampler.betas
        assert torch.allclose(temps, expected)

        # Check ordering (decreasing)
        assert torch.all(temps[1:] <= temps[:-1])


class TestEdgeCases:
    """Test edge cases for sampling classes."""

    def test_empty_state_handling(self) -> None:
        """Test handling of empty states."""
        sampler = ConcreteSampler()
        model = Mock(spec=EnergyBasedModel)

        # Empty batch
        empty_state = torch.empty(0, 10)
        samples = sampler.sample(model, empty_state)
        assert samples.shape == (0, 10)

    def test_single_sample(self) -> None:
        """Test single sample handling."""
        sampler = GibbsSampler()

        model = Mock(spec=LatentVariableModel)
        model.sample_hidden.return_value = torch.rand(1, 10)
        model.sample_visible.return_value = torch.rand(1, 20)

        single_state = torch.rand(1, 20)
        samples = sampler.sample(model, single_state)
        assert samples.shape == (1, 20)

    def test_large_batch(self) -> None:
        """Test large batch handling."""
        sampler = ConcreteSampler()
        model = Mock(spec=EnergyBasedModel)

        # Large batch
        large_state = torch.randn(10000, 100)
        samples = sampler.sample(model, large_state, num_steps=1)

        assert samples.shape == large_state.shape
        assert torch.all(torch.isfinite(samples))

    def test_gradient_estimator_without_sampler(self) -> None:
        """Test gradient estimator edge cases."""
        # Sampler that returns None
        mock_sampler = Mock(spec=Sampler)
        mock_sampler.sample.return_value = None

        estimator = ConcreteGradientEstimator(mock_sampler)

        # Should still return gradients
        model = Mock(spec=EnergyBasedModel)
        data = torch.randn(10, 5)

        gradients = estimator.estimate_gradient(model, data)
        assert isinstance(gradients, dict)
