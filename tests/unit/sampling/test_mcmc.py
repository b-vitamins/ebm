"""Unit tests for advanced MCMC sampling methods."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from ebm.models.base import EnergyBasedModel, LatentVariableModel
from ebm.sampling.mcmc import (
    AnnealedImportanceSampling,
    ParallelTempering,
    PTGradientEstimator,
)


class MockLatentModel(LatentVariableModel):
    """Mock model for testing."""

    def __init__(self, n_visible: int = 10, n_hidden: int = 5) -> None:
        self.num_visible = n_visible
        self.num_hidden = n_hidden
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.vbias = nn.Parameter(torch.zeros(n_visible))
        self.hbias = nn.Parameter(torch.zeros(n_hidden))

    def sample_hidden(
        self,
        visible: torch.Tensor,
        *,
        beta: torch.Tensor | None = None,
        return_prob: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units given visible layer."""
        pre_h = visible @ self.W.T + self.hbias
        if beta is not None:
            if beta.dim() == 0:
                pre_h = pre_h * beta
            else:
                pre_h = pre_h * beta.unsqueeze(-1)
        prob_h = torch.sigmoid(pre_h)
        sample_h = torch.bernoulli(prob_h)

        if return_prob:
            return sample_h, prob_h
        return sample_h

    def sample_visible(
        self,
        hidden: torch.Tensor,
        *,
        beta: torch.Tensor | None = None,
        return_prob: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sample visible units given hidden layer."""
        pre_v = hidden @ self.W + self.vbias
        if beta is not None:
            if beta.dim() == 0:
                pre_v = pre_v * beta
            else:
                pre_v = pre_v * beta.unsqueeze(-1)
        prob_v = torch.sigmoid(pre_v)
        sample_v = torch.bernoulli(prob_v)

        if return_prob:
            return sample_v, prob_v
        return sample_v

    def free_energy(
        self, v: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute free energy of visible samples."""
        pre_h = v @ self.W.T + self.hbias
        if beta is not None:
            if beta.dim() == 0:
                pre_h = pre_h * beta
                v_term = beta * (v @ self.vbias)
            else:
                pre_h = pre_h * beta.unsqueeze(-1)
                v_term = (v @ self.vbias) * beta
        else:
            v_term = v @ self.vbias
        h_term = torch.nn.functional.softplus(pre_h).sum(dim=-1)
        return -v_term - h_term

    def energy(
        self,
        x: torch.Tensor,
        *,
        beta: torch.Tensor | None = None,
        return_parts: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute joint energy of visible and hidden layers."""
        v = x[..., : self.num_visible]
        h = x[..., self.num_visible :]

        interaction = -torch.einsum("...i,...j,ji->...", h, v, self.W)
        v_term = -(v @ self.vbias)
        h_term = -(h @ self.hbias)

        energy = interaction + v_term + h_term

        if beta is not None:
            energy = energy * beta

        if return_parts:
            return {
                "interaction": interaction,
                "visible": v_term,
                "hidden": h_term,
                "total": energy,
            }
        return energy

    @property
    def device(self) -> torch.device:
        """Return tensor device."""
        return self.W.device

    @property
    def dtype(self) -> torch.dtype:
        """Return tensor dtype."""
        return self.W.dtype


class TestParallelTempering:
    """Test ParallelTempering sampler."""

    def test_initialization(self) -> None:
        """Test PT initialization."""
        pt = ParallelTempering(
            num_temps=5,
            min_beta=0.1,
            max_beta=1.0,
            swap_every=2,
            num_chains=10,
            adaptive=True,
        )

        assert pt.num_temps == 5
        assert pt.min_beta == 0.1
        assert pt.max_beta == 1.0
        assert pt.swap_every == 2
        assert pt.num_chains == 10
        assert pt.adaptive is True

        # Check temperature schedule
        assert len(pt.betas) == 5
        assert pt.betas[0] >= 0.1
        assert pt.betas[-1] <= 1.0

        # Check swap statistics buffers
        assert pt.swap_attempts.shape == (4,)  # num_temps - 1
        assert pt.swap_accepts.shape == (4,)
        assert torch.all(pt.swap_attempts == 0)
        assert torch.all(pt.swap_accepts == 0)

    def test_chain_initialization(self) -> None:
        """Test chain initialization."""
        pt = ParallelTempering(num_temps=3, num_chains=5)
        model = MockLatentModel(n_visible=8, n_hidden=4)

        # Initialize chains
        pt.init_chains(model, batch_size=10, state_shape=(8,))

        assert pt.chains is not None
        assert pt.chains.shape == (
            5,
            3,
            8,
        )  # num_chains x num_temps x visible_size
        assert pt.chain_temps.shape == (5, 3)

        # Check temperature assignments
        expected_temps = torch.arange(3).unsqueeze(0).expand(5, -1)
        assert torch.equal(pt.chain_temps, expected_temps)

    def test_gibbs_step_all_temps(self) -> None:
        """Test Gibbs step at all temperatures."""
        pt = ParallelTempering(num_temps=3, min_beta=0.5, max_beta=1.0)
        model = MockLatentModel()

        # Initialize
        pt.init_chains(model, batch_size=2, state_shape=(10,))

        # Store initial state
        initial_chains = pt.chains.clone()

        # Run Gibbs step
        pt._gibbs_step_all_temps(model)

        # Chains should have changed
        assert not torch.allclose(pt.chains, initial_chains)

        # Check that different temperatures produce different results
        # (Higher beta = lower temperature = less random)
        # This is statistical, so we just check shapes for now
        assert pt.chains.shape == initial_chains.shape

    def test_swap_attempts(self) -> None:
        """Test state swapping between temperatures."""
        pt = ParallelTempering(num_temps=3, swap_every=1)
        model = MockLatentModel(n_visible=5, n_hidden=3)

        # Initialize with known states
        pt.init_chains(model, batch_size=2, state_shape=(5,))

        # Set specific chain values for testing
        with torch.no_grad():
            pt.chains[0, 0] = torch.ones(5) * 0.0  # Chain 0, temp 0
            pt.chains[0, 1] = torch.ones(5) * 1.0  # Chain 0, temp 1

        # Mock energy calculations to force acceptance
        def mock_free_energy(v: torch.Tensor) -> torch.Tensor:
            # Make energy of state 1.0 lower than state 0.0
            return torch.where(
                v.mean() > 0.5, torch.tensor(-10.0), torch.tensor(0.0)
            )

        model.free_energy = mock_free_energy

        # Attempt swaps
        pt._attempt_swaps(model)

        # Check swap statistics updated
        assert pt.swap_attempts.sum() > 0

        # With our mock energies, swaps should be accepted
        # The exact behavior depends on random acceptance

    def test_sampling(self) -> None:
        """Test full PT sampling."""
        pt = ParallelTempering(
            num_temps=3, min_beta=0.5, max_beta=1.0, swap_every=2, num_chains=4
        )
        model = MockLatentModel()

        init_state = torch.rand(2, 10)  # 2 samples requested

        # Run sampling
        samples = pt.sample(model, init_state, num_steps=5)

        assert samples.shape == init_state.shape
        assert pt.state.num_steps == 5

        # Should have attempted swaps
        if pt.swap_every <= 5:
            assert pt.swap_attempts.sum() > 0

    def test_swap_rates(self) -> None:
        """Test swap rate calculation."""
        pt = ParallelTempering(num_temps=4)

        # Set some swap statistics
        pt.swap_attempts = torch.tensor([100.0, 100.0, 100.0])
        pt.swap_accepts = torch.tensor([30.0, 45.0, 20.0])

        rates = pt.swap_rates

        assert torch.allclose(rates, torch.tensor([0.3, 0.45, 0.2]))

    def test_adaptive_temperatures(self) -> None:
        """Test adaptive temperature adjustment."""
        pt = ParallelTempering(
            num_temps=3, min_beta=0.1, max_beta=1.0, adaptive=True
        )

        # Set swap rates
        pt.swap_attempts = torch.tensor([100.0, 100.0])
        pt.swap_accepts = torch.tensor([10.0, 80.0])  # Too low and too high

        # Store original betas
        pt.betas.clone()

        # Adapt temperatures
        pt.adapt_temperatures(target_rate=0.3)

        # Beta[0] should decrease (rates too low)
        # Beta[1] should increase (rates too high)
        # But changes should maintain ordering
        assert torch.all(pt.betas[1:] >= pt.betas[:-1])

    def test_invalid_model_type(self) -> None:
        """Test error on non-latent model."""
        pt = ParallelTempering()
        model = Mock(spec=EnergyBasedModel)
        init_state = torch.rand(5, 10)

        with pytest.raises(
            TypeError, match="PT requires a LatentVariableModel"
        ):
            pt.sample(model, init_state)

    def test_registry_registration(self) -> None:
        """Test PT registration in registry."""
        from ebm.core.registry import samplers

        assert "parallel_tempering" in samplers
        assert "pt" in samplers
        assert "replica_exchange" in samplers
        assert samplers.get("parallel_tempering") is ParallelTempering


class TestPTGradientEstimator:
    """Test PT-based gradient estimation."""

    def test_initialization(self) -> None:
        """Test PT gradient estimator initialization."""
        pt_grad = PTGradientEstimator(num_temps=5, k=1, swap_every=2)

        assert isinstance(pt_grad.sampler, ParallelTempering)
        assert pt_grad.k == 1
        assert pt_grad.sampler.num_temps == 5
        assert pt_grad.sampler.swap_every == 2

    def test_gradient_estimation(self) -> None:
        """Test gradient estimation using PT."""
        pt_grad = PTGradientEstimator(num_temps=3, k=2)
        model = MockLatentModel()
        data = torch.rand(10, 10)

        gradients = pt_grad.estimate_gradient(model, data)

        assert "W" in gradients
        assert "vbias" in gradients
        assert "hbias" in gradients

        assert gradients["W"].shape == model.W.shape
        assert gradients["vbias"].shape == model.vbias.shape
        assert gradients["hbias"].shape == model.hbias.shape

    def test_invalid_model(self) -> None:
        """Test error on non-latent model."""
        pt_grad = PTGradientEstimator()
        model = Mock(spec=EnergyBasedModel)
        data = torch.rand(5, 10)

        with pytest.raises(
            TypeError,
            match="PT gradient estimation requires LatentVariableModel",
        ):
            pt_grad.estimate_gradient(model, data)


class TestAnnealedImportanceSampling:
    """Test AIS implementation."""

    def test_initialization(self) -> None:
        """Test AIS initialization."""
        ais = AnnealedImportanceSampling(num_temps=100, num_chains=50, k=1)

        assert ais.num_temps == 100
        assert ais.num_chains == 50
        assert ais.k == 1
        assert ais.min_beta == 0.0
        assert ais.max_beta == 1.0

        # Check beta schedule
        assert len(ais.betas) == 100
        assert ais.betas[0] == 0.0
        assert ais.betas[-1] == 1.0

    def test_log_partition_estimation(self) -> None:
        """Test partition function estimation."""
        ais = AnnealedImportanceSampling(num_temps=10, num_chains=20, k=1)

        model = MockLatentModel(n_visible=3, n_hidden=2)

        # Base partition function for independent Bernoulli
        base_log_z = (model.num_visible + model.num_hidden) * np.log(2)

        # Estimate partition function
        log_z_est = ais.estimate_log_partition(model, base_log_z)

        assert isinstance(log_z_est, float)
        assert np.isfinite(log_z_est)

        # For a small model, should be in reasonable range
        # Can't test exact value due to stochasticity
        assert log_z_est > 0  # Should be positive for typical RBMs

    def test_log_partition_with_bounds(self) -> None:
        """Test partition function estimation with confidence bounds."""
        ais = AnnealedImportanceSampling(num_temps=50, num_chains=100, k=1)

        model = MockLatentModel(n_visible=5, n_hidden=3)
        base_log_z = (model.num_visible + model.num_hidden) * np.log(2)

        # Get estimate with bounds
        log_z, ci_lower, ci_upper = ais.estimate_log_partition(
            model, base_log_z, return_bounds=True
        )

        assert isinstance(log_z, float)
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)

        # Check ordering
        assert ci_lower <= log_z <= ci_upper

        # Confidence interval should be reasonable
        assert (ci_upper - ci_lower) < 10.0  # Not too wide

    def test_importance_weights(self) -> None:
        """Test importance weight computation."""
        ais = AnnealedImportanceSampling(num_temps=5, num_chains=10, k=1)

        model = MockLatentModel(n_visible=4, n_hidden=2)

        # Manually run AIS to check weights
        device = model.device

        # Initialize
        h_init = torch.rand(10, 2, device=device).round()
        v = model.sample_visible(h_init, beta=0.0)

        log_w = torch.zeros(10, device=device)

        # Run through temperatures
        for i in range(len(ais.betas)):
            beta = ais.betas[i]

            if i > 0:
                prev_beta = ais.betas[i - 1]
                # Weights should increase
                log_w.clone()
                log_w += (prev_beta - beta) * model.free_energy(v)
                # Free energy is negative, so log_w should generally increase
                # (but not always due to stochasticity)

            # Gibbs step
            h = model.sample_hidden(v, beta=beta)
            v = model.sample_visible(h, beta=beta)

        # Final weights should be finite
        assert torch.all(torch.isfinite(log_w))

    def test_effective_sample_size(self) -> None:
        """Test ESS calculation in AIS."""
        ais = AnnealedImportanceSampling(num_temps=20, num_chains=50, k=1)

        model = MockLatentModel()
        base_log_z = model.num_visible * np.log(2)

        # Get bounds (which includes ESS calculation)
        log_z, ci_lower, ci_upper = ais.estimate_log_partition(
            model, base_log_z, return_bounds=True
        )

        # Check that bounds are reasonable
        # With good ESS, bounds should be relatively tight
        width = ci_upper - ci_lower
        assert width < 2.0  # Reasonable for small model


class TestEdgeCases:
    """Test edge cases for MCMC methods."""

    def test_single_temperature(self) -> None:
        """Test PT with single temperature."""
        pt = ParallelTempering(num_temps=1)
        model = MockLatentModel()

        init_state = torch.rand(5, 10)
        samples = pt.sample(model, init_state, num_steps=5)

        # Should work but no swaps possible
        assert samples.shape == init_state.shape
        assert torch.all(pt.swap_attempts == 0)

    def test_extreme_temperatures(self) -> None:
        """Test PT with extreme temperature range."""
        pt = ParallelTempering(
            num_temps=3,
            min_beta=0.001,  # Very high temperature
            max_beta=100.0,  # Very low temperature
        )
        model = MockLatentModel()

        pt.init_chains(model, batch_size=2, state_shape=(10,))

        # Sample
        samples = pt.sample(model, torch.rand(2, 10), num_steps=10)

        # Should handle extreme temperatures
        assert torch.all(torch.isfinite(samples))

    def test_empty_batch_pt(self) -> None:
        """Test PT with empty batch."""
        pt = ParallelTempering()
        model = MockLatentModel()

        torch.empty(0, 10)

        # Initialize chains with empty batch
        pt.init_chains(model, batch_size=0, state_shape=(10,))
        assert pt.chains.shape[0] == pt.num_chains or 0

    def test_ais_with_zero_chains(self) -> None:
        """Test AIS with invalid number of chains."""
        with pytest.raises(ValueError, match="num_chains > 0"):
            # Should validate num_chains > 0
            AnnealedImportanceSampling(num_chains=0)

    def test_numerical_stability_ais(self) -> None:
        """Test AIS numerical stability."""
        ais = AnnealedImportanceSampling(num_temps=100, num_chains=10)

        # Model with extreme parameters
        model = MockLatentModel()
        with torch.no_grad():
            model.W.data = torch.randn_like(model.W) * 10  # Large weights
            model.vbias.data = torch.randn_like(model.vbias) * 5
            model.hbias.data = torch.randn_like(model.hbias) * 5

        base_log_z = 0.0  # Simple base

        # Should handle extreme energies
        log_z = ais.estimate_log_partition(model, base_log_z)
        assert np.isfinite(log_z)
