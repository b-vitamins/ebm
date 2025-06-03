"""Unit tests for partition function estimation."""

import math
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from ebm.inference.partition import (
    AISEstimator,
    BridgeSampling,
    PartitionFunctionEstimator,
    RatioEstimator,
    SimpleIS,
)
from ebm.models.base import EnergyBasedModel, LatentVariableModel


class MockRBM(LatentVariableModel):
    """Mock RBM for testing partition function estimation."""

    def __init__(self, n_visible=5, n_hidden=3) -> None:
        self.num_visible = n_visible
        self.num_hidden = n_hidden
        self.W = torch.nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.vbias = torch.nn.Parameter(torch.zeros(n_visible))
        self.hbias = torch.nn.Parameter(torch.zeros(n_hidden))
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def free_energy(self, v, *, beta=None):
        pre_h = v @ self.W.T + self.hbias
        if beta is not None:
            pre_h = pre_h * beta
            v_term = beta * (v @ self.vbias)
        else:
            v_term = v @ self.vbias
        h_term = torch.nn.functional.softplus(pre_h).sum(dim=-1)
        return -v_term - h_term

    def sample_hidden(self, visible, *, beta=None, return_prob=False):
        pre_h = visible @ self.W.T + self.hbias
        if beta is not None:
            pre_h = pre_h * beta
        prob_h = torch.sigmoid(pre_h)
        sample_h = torch.bernoulli(prob_h)

        if return_prob:
            return sample_h, prob_h
        return sample_h

    def sample_visible(self, hidden, *, beta=None, return_prob=False):
        pre_v = hidden @ self.W + self.vbias
        if beta is not None:
            pre_v = pre_v * beta
        prob_v = torch.sigmoid(pre_v)
        sample_v = torch.bernoulli(prob_v)

        if return_prob:
            return sample_v, prob_v
        return sample_v

    def sample_fantasy_particles(self, num_samples, num_steps):
        v = torch.rand(num_samples, self.num_visible).round()
        for _ in range(num_steps):
            h = self.sample_hidden(v)
            v = self.sample_visible(h)
        return v

    def ais_adapter(self):
        """Create AIS adapter."""
        from ebm.models.rbm.base import RBMAISAdapter

        return RBMAISAdapter(self)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def energy(self, x, *, beta=None, return_parts=False):
        v = x[:, : self.num_visible]
        h = x[:, self.num_visible :]

        interaction = -torch.einsum("bi,bj,ji->b", h, v, self.W)
        v_term = -(v @ self.vbias)
        h_term = -(h @ self.hbias)

        energy = interaction + v_term + h_term
        if beta is not None:
            energy = energy * beta

        return energy


class TestPartitionFunctionEstimator:
    """Test base PartitionFunctionEstimator class."""

    def test_initialization(self) -> None:
        """Test base estimator initialization."""
        model = MockRBM()
        estimator = PartitionFunctionEstimator(model)

        assert estimator.model is model

    def test_abstract_estimate(self) -> None:
        """Test that estimate method must be implemented."""
        model = MockRBM()
        estimator = PartitionFunctionEstimator(model)

        with pytest.raises(NotImplementedError):
            estimator.estimate()


class TestAISEstimator:
    """Test AISEstimator class."""

    def test_initialization(self) -> None:
        """Test AIS estimator initialization."""
        model = MockRBM()
        estimator = AISEstimator(model=model, num_temps=100, num_chains=50)

        assert estimator.model is model
        assert estimator.num_temps == 100
        assert estimator.num_chains == 50
        assert len(estimator.betas) == 100
        assert estimator.betas[0] == 0.0
        assert estimator.betas[-1] == 1.0

    def test_invalid_model_type(self) -> None:
        """Test error on non-latent model."""
        model = Mock(spec=EnergyBasedModel)
        estimator = AISEstimator(model)

        with pytest.raises(TypeError, match="AIS requires a LatentVariableModel"):
            estimator.estimate()

    def test_basic_estimation(self) -> None:
        """Test basic partition function estimation."""
        model = MockRBM(n_visible=3, n_hidden=2)

        # Small weights for stability
        with torch.no_grad():
            model.W.data *= 0.01

        estimator = AISEstimator(model=model, num_temps=50, num_chains=20)

        # Base partition function
        base_log_z = (model.num_visible + model.num_hidden) * math.log(2)

        log_z = estimator.estimate(base_log_z=base_log_z, show_progress=False)

        assert isinstance(log_z, float)
        assert np.isfinite(log_z)

        # For very small weights, should be close to base
        assert abs(log_z - base_log_z) < 2.0

    def test_estimation_with_diagnostics(self) -> None:
        """Test estimation with diagnostic information."""
        model = MockRBM(n_visible=4, n_hidden=3)

        estimator = AISEstimator(model=model, num_temps=20, num_chains=10)

        log_z, diagnostics = estimator.estimate(
            return_diagnostics=True, show_progress=False
        )

        assert isinstance(log_z, float)
        assert isinstance(diagnostics, dict)

        # Check diagnostic keys
        assert "log_Z" in diagnostics
        assert "log_Z_std" in diagnostics
        assert "log_Z_trajectory" in diagnostics
        assert "effective_sample_size" in diagnostics
        assert "log_weights" in diagnostics
        assert "final_weights" in diagnostics

        # Check values
        assert diagnostics["log_Z"] == log_z
        assert diagnostics["log_Z_std"] > 0
        assert len(diagnostics["log_weights"]) == 10
        assert diagnostics["effective_sample_size"] > 0
        assert diagnostics["effective_sample_size"] <= 10

    def test_custom_base_partition(self) -> None:
        """Test with custom base partition function."""
        model = MockRBM()

        # Mock AIS adapter
        mock_adapter = Mock()
        mock_adapter.base_log_partition.return_value = 5.0
        model.ais_adapter = Mock(return_value=mock_adapter)

        estimator = AISEstimator(model, num_temps=10, num_chains=5)

        estimator.estimate(show_progress=False)

        # Should use adapter's base partition
        mock_adapter.base_log_partition.assert_called_once()

    def test_temperature_schedule(self) -> None:
        """Test that temperature schedule is used correctly."""
        model = MockRBM(n_visible=2, n_hidden=2)

        estimator = AISEstimator(model=model, num_temps=5, num_chains=1)

        # Check beta schedule
        assert torch.allclose(
            estimator.betas, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        )

        # Run estimation and check that betas are used
        with patch.object(
            model, "sample_hidden", wraps=model.sample_hidden
        ) as mock_hidden:
            with patch.object(model, "sample_visible", wraps=model.sample_visible):
                estimator.estimate(show_progress=False)

                # Check that different betas were used
                beta_calls = []
                for call in mock_hidden.call_args_list:
                    if "beta" in call[1]:
                        beta_calls.append(call[1]["beta"])

                # Should have calls with different beta values
                unique_betas = {float(b) for b in beta_calls if b is not None}
                assert len(unique_betas) > 1


class TestBridgeSampling:
    """Test BridgeSampling class."""

    def test_initialization(self) -> None:
        """Test bridge sampling initialization."""
        model1 = MockRBM(n_visible=5, n_hidden=3)
        model2 = MockRBM(n_visible=5, n_hidden=3)

        estimator = BridgeSampling(model1=model1, model2=model2, num_samples=100)

        assert estimator.model is model1
        assert estimator.model2 is model2
        assert estimator.num_samples == 100

    def test_basic_ratio_estimation(self) -> None:
        """Test basic partition function ratio estimation."""
        # Create two similar models
        model1 = MockRBM(n_visible=4, n_hidden=2)
        model2 = MockRBM(n_visible=4, n_hidden=2)

        # Make model2 slightly different
        with torch.no_grad():
            model1.W.data = torch.randn_like(model1.W) * 0.01
            model2.W.data = model1.W.data * 1.1  # 10% larger weights

        estimator = BridgeSampling(model1=model1, model2=model2, num_samples=50)

        log_ratio, std_err = estimator.estimate(tol=1e-4, max_iter=100)

        assert isinstance(log_ratio, float)
        assert isinstance(std_err, float)
        assert np.isfinite(log_ratio)
        assert std_err > 0

        # For slightly different models, ratio should be small
        assert abs(log_ratio) < 5.0

    def test_convergence_detection(self) -> None:
        """Test bridge sampling convergence."""
        model1 = MockRBM(n_visible=3, n_hidden=2)
        model2 = MockRBM(n_visible=3, n_hidden=2)

        # Very similar models for fast convergence
        with torch.no_grad():
            model1.W.data = torch.randn_like(model1.W) * 0.001
            model2.W.data = model1.W.data * 1.01

        estimator = BridgeSampling(model1, model2, num_samples=100)

        # Track iterations

        # Mock log_debug to track convergence
        with patch.object(estimator, "log_debug") as mock_log:
            log_ratio, _ = estimator.estimate(tol=1e-6, max_iter=1000)

            # Should have converged
            assert mock_log.called
            convergence_msg = mock_log.call_args[0][0]
            assert "converged" in convergence_msg

    def test_sample_generation(self) -> None:
        """Test that models can generate samples."""
        model1 = Mock(spec=LatentVariableModel)
        model2 = Mock(spec=LatentVariableModel)

        # Mock sample generation
        model1.sample_fantasy_particles.return_value = torch.randn(50, 10)
        model2.sample_fantasy_particles.return_value = torch.randn(50, 10)

        # Mock energy functions
        model1.free_energy.return_value = torch.randn(50)
        model2.free_energy.return_value = torch.randn(50)

        estimator = BridgeSampling(model1, model2, num_samples=50)

        with patch.object(estimator, "log_info"):
            log_ratio, std_err = estimator.estimate()

        # Check sample generation was called
        model1.sample_fantasy_particles.assert_called_once_with(
            num_samples=50, num_steps=10000
        )
        model2.sample_fantasy_particles.assert_called_once()

    def test_missing_sample_method(self) -> None:
        """Test error when model lacks sampling method."""
        model1 = MockRBM()
        model2 = Mock(spec=EnergyBasedModel)  # No sample_fantasy_particles

        estimator = BridgeSampling(model1, model2)

        with pytest.raises(NotImplementedError):
            estimator.estimate()


class TestSimpleIS:
    """Test SimpleIS class."""

    def test_initialization(self) -> None:
        """Test simple IS initialization."""
        model = MockRBM()

        estimator = SimpleIS(model=model, proposal="uniform", num_samples=1000)

        assert estimator.model is model
        assert estimator.proposal == "uniform"
        assert estimator.num_samples == 1000

    def test_uniform_proposal(self) -> None:
        """Test estimation with uniform proposal."""
        model = MockRBM(n_visible=3, n_hidden=2)

        # Small weights for reasonable estimates
        with torch.no_grad():
            model.W.data *= 0.01

        estimator = SimpleIS(model=model, proposal="uniform", num_samples=500)

        log_z, std_err = estimator.estimate()

        assert isinstance(log_z, float)
        assert isinstance(std_err, float)
        assert np.isfinite(log_z)
        assert std_err > 0

        # For small RBM, partition function should be reasonable
        assert -10 < log_z < 20

    def test_data_proposal(self) -> None:
        """Test estimation with data-based proposal."""
        model = MockRBM(n_visible=5, n_hidden=3)

        # Create simple data loader
        data = torch.rand(200, 5).round()
        dataset = torch.utils.data.TensorDataset(data)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=50)

        estimator = SimpleIS(model=model, proposal="data", num_samples=100)

        log_z, std_err = estimator.estimate(data_loader=data_loader)

        assert isinstance(log_z, float)
        assert isinstance(std_err, float)
        assert np.isfinite(log_z)

    def test_invalid_proposal(self) -> None:
        """Test error on invalid proposal type."""
        model = MockRBM()
        estimator = SimpleIS(model, proposal="invalid")

        with pytest.raises(ValueError, match="Unknown proposal"):
            estimator.estimate()

    def test_effective_sample_size(self) -> None:
        """Test ESS calculation in importance sampling."""
        model = MockRBM(n_visible=4, n_hidden=2)

        # Set specific weights for predictable behavior
        with torch.no_grad():
            model.W.data = torch.zeros_like(model.W)
            model.vbias.data = torch.ones(4) * 0.1
            model.hbias.data = torch.zeros(2)

        estimator = SimpleIS(model, num_samples=100)

        log_z, std_err = estimator.estimate()

        # With uniform weights and simple model, ESS should be reasonable
        # Can't directly access ESS, but std_err reflects it
        assert 0 < std_err < 10


class TestRatioEstimator:
    """Test RatioEstimator class."""

    def test_initialization(self) -> None:
        """Test ratio estimator initialization."""
        models = [MockRBM(), MockRBM(), MockRBM()]

        estimator = RatioEstimator(models=models, method="bridge")

        assert estimator.models == models
        assert estimator.method == "bridge"
        assert estimator.model is models[0]

    def test_pairwise_ratios(self) -> None:
        """Test estimation of all pairwise ratios."""
        # Create three models with different scales
        model1 = MockRBM(n_visible=3, n_hidden=2)
        model2 = MockRBM(n_visible=3, n_hidden=2)
        model3 = MockRBM(n_visible=3, n_hidden=2)

        with torch.no_grad():
            model1.W.data = torch.randn_like(model1.W) * 0.01
            model2.W.data = model1.W.data * 1.5
            model3.W.data = model1.W.data * 2.0

        estimator = RatioEstimator(models=[model1, model2, model3], method="bridge")

        # Mock BridgeSampling to avoid actual computation
        with patch("ebm.inference.partition.BridgeSampling") as mock_bridge:
            # Set up mock returns
            mock_instance = Mock()
            mock_instance.estimate.side_effect = [
                (0.5, 0.1),  # log(Z2/Z1)
                (1.0, 0.15),  # log(Z3/Z1)
            ]
            mock_bridge.return_value = mock_instance

            ratios = estimator.estimate_all_ratios(reference_idx=0)

        # Check structure
        assert isinstance(ratios, dict)

        # Should have identity ratios
        assert ratios[(0, 0)] == (0.0, 0.0)

        # Should have computed ratios
        assert (1, 0) in ratios
        assert (2, 0) in ratios
        assert (0, 1) in ratios
        assert (0, 2) in ratios

        # Check symmetry
        log_10, se_10 = ratios[(1, 0)]
        log_01, se_01 = ratios[(0, 1)]
        assert abs(log_10 + log_01) < 1e-6

    def test_transitivity(self) -> None:
        """Test transitivity in ratio computation."""
        models = [MockRBM() for _ in range(4)]

        estimator = RatioEstimator(models, method="bridge")

        # Mock direct ratios
        with patch("ebm.inference.partition.BridgeSampling") as mock_bridge:
            mock_instance = Mock()
            mock_instance.estimate.side_effect = [
                (1.0, 0.1),  # log(Z1/Z0)
                (2.0, 0.1),  # log(Z2/Z0)
                (3.0, 0.1),  # log(Z3/Z0)
            ]
            mock_bridge.return_value = mock_instance

            ratios = estimator.estimate_all_ratios(reference_idx=0)

        # Check transitivity: log(Z2/Z1) = log(Z2/Z0) - log(Z1/Z0)
        log_21_direct = ratios[(2, 1)][0] if (2, 1) in ratios else None
        if log_21_direct is None:
            # Computed via transitivity
            log_20 = ratios[(2, 0)][0]
            log_10 = ratios[(1, 0)][0]
            log_21_computed = log_20 - log_10
            assert abs(log_21_computed - 1.0) < 1e-6  # 2.0 - 1.0

    def test_invalid_method(self) -> None:
        """Test error on invalid estimation method."""
        models = [MockRBM()]

        with pytest.raises(ValueError):
            RatioEstimator(models, method="invalid")


class TestEdgeCases:
    """Test edge cases for partition function estimation."""

    def test_empty_model(self) -> None:
        """Test with model having zero dimensions."""
        model = MockRBM(n_visible=0, n_hidden=0)

        estimator = AISEstimator(model, num_temps=10, num_chains=5)

        # Should handle gracefully
        with pytest.raises(Exception):  # noqa: B017 - error type not specified
            estimator.estimate(show_progress=False)

    def test_extreme_temperatures(self) -> None:
        """Test AIS with extreme temperature range."""
        model = MockRBM()

        # Very wide temperature range
        estimator = AISEstimator(model, num_temps=100, num_chains=10)
        estimator.betas = torch.logspace(-5, 2, 100)  # 0.00001 to 100

        # Should still work
        log_z = estimator.estimate(show_progress=False)
        assert np.isfinite(log_z)

    def test_numerical_stability(self) -> None:
        """Test numerical stability with extreme weights."""
        model = MockRBM()

        # Large weights
        with torch.no_grad():
            model.W.data = torch.randn_like(model.W) * 10
            model.vbias.data = torch.randn_like(model.vbias) * 5
            model.hbias.data = torch.randn_like(model.hbias) * 5

        estimator = SimpleIS(model, num_samples=100)

        # Should handle large energies
        log_z, std_err = estimator.estimate()
        assert np.isfinite(log_z) or np.isinf(
            log_z
        )  # May overflow but shouldn't be NaN

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self) -> None:
        """Test estimation with CUDA models."""
        model = MockRBM()

        # Move to CUDA
        model._device = torch.device("cuda")
        model.W = model.W.cuda()
        model.vbias = model.vbias.cuda()
        model.hbias = model.hbias.cuda()

        estimator = AISEstimator(model, num_temps=10, num_chains=5)

        # Should handle CUDA properly
        log_z = estimator.estimate(show_progress=False)
        assert isinstance(log_z, float)
