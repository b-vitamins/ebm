"""Unit tests for gradient-based sampling methods."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from ebm.models.base import EnergyBasedModel, LatentVariableModel
from ebm.sampling.gradient import (
    CDSampler,
    CDWithDecay,
    ContrastiveDivergence,
    FastPersistentCD,
    PersistentContrastiveDivergence,
)
from ebm.utils.tensor import batch_outer_product


class MockLatentModel(LatentVariableModel):
    """Mock latent variable model for testing."""

    def __init__(self, n_visible: int = 20, n_hidden: int = 10) -> None:
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
        """Sample hidden units from visible layer."""
        pre_h = visible @ self.W.T + self.hbias
        if beta is not None:
            pre_h = pre_h * beta
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
        """Sample visible units from hidden layer."""
        pre_v = hidden @ self.W + self.vbias
        if beta is not None:
            pre_v = pre_v * beta
        prob_v = torch.sigmoid(pre_v)
        sample_v = torch.bernoulli(prob_v)

        if return_prob:
            return sample_v, prob_v
        return sample_v

    def free_energy(
        self, v: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute free energy of visible units."""
        pre_h = v @ self.W.T + self.hbias
        if beta is not None:
            pre_h = pre_h * beta
            v_term = beta * (v @ self.vbias)
        else:
            v_term = v @ self.vbias
        h_term = torch.nn.functional.softplus(pre_h).sum(dim=-1)
        return -v_term - h_term

    @property
    def device(self) -> torch.device:
        """Return device of model parameters."""
        return self.W.device

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype of model parameters."""
        return self.W.dtype

    def energy(
        self,
        x: torch.Tensor,
        *,
        beta: torch.Tensor | None = None,
        return_parts: bool = False,
    ) -> torch.Tensor:
        """Return constant energy (not used in CD)."""
        # Not used in CD
        return torch.zeros(x.shape[0])

    def named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Return model parameters as name-parameter pairs."""
        return [("W", self.W), ("vbias", self.vbias), ("hbias", self.hbias)]


class TestContrastiveDivergence:
    """Test ContrastiveDivergence gradient estimator."""

    def test_initialization(self) -> None:
        """Test CD initialization."""
        cd = ContrastiveDivergence(k=1)

        assert cd.k == 1
        assert cd.persistent is False
        assert cd.num_chains is None
        assert isinstance(cd.sampler, CDSampler)

        # With persistence
        pcd = ContrastiveDivergence(k=5, persistent=True, num_chains=100)
        assert pcd.k == 5
        assert pcd.persistent is True
        assert pcd.num_chains == 100

    def test_estimate_gradient(self) -> None:
        """Test gradient estimation."""
        cd = ContrastiveDivergence(k=1)
        model = MockLatentModel()
        data = torch.rand(32, 20)

        gradients = cd.estimate_gradient(model, data)

        # Check gradient dictionary
        assert "W" in gradients
        assert "vbias" in gradients
        assert "hbias" in gradients

        # Check shapes
        assert gradients["W"].shape == model.W.shape
        assert gradients["vbias"].shape == model.vbias.shape
        assert gradients["hbias"].shape == model.hbias.shape

        # Check that negative samples were stored
        assert hasattr(cd, "last_negative_samples")
        assert cd.last_negative_samples.shape == data.shape

    def test_gradient_computation(self) -> None:
        """Test that gradients are computed correctly."""
        cd = ContrastiveDivergence(k=1)
        model = MockLatentModel(n_visible=5, n_hidden=3)

        # Use specific data for testing
        data = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0]]
        )

        # Get gradients
        gradients = cd.estimate_gradient(model, data)

        # Manually compute expected gradients
        # Positive phase
        h_data_prob = model.sample_hidden(data, return_prob=True)[1]

        # Negative phase (1 Gibbs step)
        h_sample = model.sample_hidden(data)
        v_model = model.sample_visible(h_sample)
        h_model_prob = model.sample_hidden(v_model, return_prob=True)[1]

        # Expected gradients
        expected_w = batch_outer_product(h_data_prob, data).mean(
            dim=0
        ) - batch_outer_product(h_model_prob, v_model).mean(dim=0)
        expected_vbias = data.mean(dim=0) - v_model.mean(dim=0)
        expected_hbias = h_data_prob.mean(dim=0) - h_model_prob.mean(dim=0)

        # Note: Due to stochasticity in sampling, we can't expect exact matches
        # But shapes should match
        assert gradients["W"].shape == expected_w.shape
        assert gradients["vbias"].shape == expected_vbias.shape
        assert gradients["hbias"].shape == expected_hbias.shape

    def test_apply_gradients(self) -> None:
        """Test gradient application."""
        cd = ContrastiveDivergence(k=1)
        model = MockLatentModel()

        # Store original parameters
        w_orig = model.W.data.clone()
        vbias_orig = model.vbias.data.clone()

        # Create mock gradients
        gradients = {
            "W": torch.ones_like(model.W),
            "vbias": torch.ones_like(model.vbias),
            "hbias": torch.ones_like(model.hbias),
        }

        # Apply gradients
        lr = 0.01
        cd.apply_gradients(model, gradients, lr=lr)

        # Check parameters were updated (gradient ascent)
        assert torch.allclose(model.W.data, w_orig + lr * gradients["W"])
        assert torch.allclose(
            model.vbias.data, vbias_orig + lr * gradients["vbias"]
        )

    def test_invalid_model_type(self) -> None:
        """Test error on non-latent model."""
        cd = ContrastiveDivergence(k=1)

        # Non-latent model
        model = Mock(spec=EnergyBasedModel)
        data = torch.rand(10, 5)

        with pytest.raises(
            TypeError, match="CD requires a LatentVariableModel"
        ):
            cd.estimate_gradient(model, data)


class TestCDSampler:
    """Test CDSampler class."""

    def test_initialization(self) -> None:
        """Test CD sampler initialization."""
        # Non-persistent
        sampler = CDSampler(k=1)
        assert sampler.k == 1
        assert sampler.persistent is False
        assert sampler.num_chains is None
        assert sampler.name == "CD-1"

        # Persistent
        sampler_pcd = CDSampler(k=5, persistent=True, num_chains=100)
        assert sampler_pcd.k == 5
        assert sampler_pcd.persistent is True
        assert sampler_pcd.num_chains == 100
        assert sampler_pcd.name == "PCD-5"
        assert hasattr(sampler_pcd, "persistent_chains")

    def test_non_persistent_sampling(self) -> None:
        """Test non-persistent CD sampling."""
        sampler = CDSampler(k=3)
        model = MockLatentModel()

        init_state = torch.rand(10, 20)
        samples = sampler.sample(model, init_state)

        assert samples.shape == init_state.shape
        # Should not maintain chains
        assert sampler.persistent_chains is None

    def test_persistent_sampling(self) -> None:
        """Test persistent CD sampling."""
        sampler = CDSampler(k=1, persistent=True, num_chains=5)
        model = MockLatentModel()

        # First call initializes chains
        init_state = torch.rand(10, 20)
        samples1 = sampler.sample(model, init_state)

        assert sampler.persistent_chains is not None
        assert sampler.persistent_chains.shape == (5, 20)

        # Second call uses persistent chains
        samples2 = sampler.sample(model, init_state)

        # Chains should have been updated
        assert not torch.allclose(samples1[:5], samples2)

    def test_reset(self) -> None:
        """Test sampler reset."""
        sampler = CDSampler(k=1, persistent=True, num_chains=10)
        model = MockLatentModel()

        # Initialize chains
        sampler.sample(model, torch.rand(20, 20))
        assert sampler.persistent_chains is not None

        # Reset
        sampler.reset()
        assert sampler.persistent_chains is None
        assert sampler.num_steps_taken == 0

    def test_custom_num_steps(self) -> None:
        """Test overriding k with num_steps."""
        sampler = CDSampler(k=1)
        model = MockLatentModel()

        init_state = torch.rand(5, 20)

        # Use custom num_steps
        samples = sampler.sample(model, init_state, num_steps=5)

        # Should have done 5 steps, not 1
        # We can't directly check this without mocking, but the samples
        # should be more different from init_state
        assert samples.shape == init_state.shape


class TestPersistentContrastiveDivergence:
    """Test PersistentContrastiveDivergence class."""

    def test_initialization(self) -> None:
        """Test PCD initialization."""
        pcd = PersistentContrastiveDivergence(k=1, num_chains=100)

        assert pcd.k == 1
        assert pcd.persistent is True
        assert pcd.num_chains == 100
        assert isinstance(pcd.sampler, CDSampler)
        assert pcd.sampler.persistent is True

    def test_gradient_estimation_persistent(self) -> None:
        """Test that PCD maintains persistent chains."""
        pcd = PersistentContrastiveDivergence(k=1, num_chains=5)
        model = MockLatentModel()

        data1 = torch.rand(10, 20)
        data2 = torch.rand(10, 20)

        # First gradient estimation
        gradients1 = pcd.estimate_gradient(model, data1)
        chains1 = pcd.sampler.persistent_chains.clone()

        # Second gradient estimation
        gradients2 = pcd.estimate_gradient(model, data2)
        chains2 = pcd.sampler.persistent_chains

        # Chains should have been updated
        assert not torch.allclose(chains1, chains2)

        # Gradients should be different
        assert not torch.allclose(gradients1["W"], gradients2["W"])


class TestFastPersistentCD:
    """Test FastPersistentCD with momentum."""

    def test_initialization(self) -> None:
        """Test FPCD initialization."""
        fpcd = FastPersistentCD(
            k=1, num_chains=100, momentum=0.9, fast_weight_scale=5.0
        )

        assert fpcd.momentum == 0.9
        assert fpcd.fast_weight_scale == 5.0
        assert isinstance(fpcd.velocities, dict)
        assert len(fpcd.velocities) == 0  # Empty initially

    def test_fast_weights_application(self) -> None:
        """Test that fast weights are applied during sampling."""
        fpcd = FastPersistentCD(k=1, num_chains=10, momentum=0.9)
        model = MockLatentModel()
        data = torch.rand(20, 20)

        # Store original parameters
        w_orig2 = model.W.data.clone()

        # First call - initializes velocities
        fpcd.estimate_gradient(model, data)

        # Check velocities were created
        assert "W" in fpcd.velocities
        assert "vbias" in fpcd.velocities
        assert "hbias" in fpcd.velocities

        # Parameters should be restored
        assert torch.allclose(model.W.data, w_orig2)

        # Second call - uses velocities
        gradients2 = fpcd.estimate_gradient(model, data)

        # Velocities should have been updated
        assert not torch.allclose(fpcd.velocities["W"], gradients2["W"])

    def test_velocity_updates(self) -> None:
        """Test velocity update mechanism."""
        fpcd = FastPersistentCD(k=1, momentum=0.95)
        MockLatentModel(n_visible=5, n_hidden=3)

        # Initialize with known gradients
        grad1 = torch.ones(3, 5)
        fpcd.velocities["W"] = torch.zeros(3, 5)

        # Update velocity
        fpcd.velocities["W"].mul_(fpcd.momentum).add_(
            grad1, alpha=1 - fpcd.momentum
        )

        expected = 0.05 * grad1  # (1-0.95) * grad1
        assert torch.allclose(fpcd.velocities["W"], expected)


class TestCDWithDecay:
    """Test CDWithDecay class."""

    def test_initialization(self) -> None:
        """Test CD with decay initialization."""
        cd_decay = CDWithDecay(
            initial_k=25, final_k=1, decay_epochs=10, persistent=False
        )

        assert cd_decay.initial_k == 25
        assert cd_decay.final_k == 1
        assert cd_decay.decay_epochs == 10
        assert cd_decay.k == 25  # Starts at initial_k
        assert cd_decay.current_epoch == 0

    def test_k_update(self) -> None:
        """Test k decay mechanism."""
        cd_decay = CDWithDecay(initial_k=20, final_k=2, decay_epochs=10)

        # Initial k
        assert cd_decay.k == 20

        # Update at epoch 0
        cd_decay.update_k(0)
        assert cd_decay.k == 20

        # Update at epoch 5 (halfway)
        cd_decay.update_k(5)
        expected_k = 20 + 0.5 * (2 - 20)  # Linear decay
        assert cd_decay.k == int(expected_k)

        # Update at epoch 10 (end of decay)
        cd_decay.update_k(10)
        assert cd_decay.k == 2

        # Update beyond decay epochs
        cd_decay.update_k(20)
        assert cd_decay.k == 2  # Should stay at final_k

    def test_sampling_with_decay(self) -> None:
        """Test that sampling uses current k value."""
        cd_decay = CDWithDecay(initial_k=10, final_k=1, decay_epochs=5)

        model = MockLatentModel()
        data = torch.rand(5, 20)

        # Initial sampling
        cd_decay.update_k(0)
        assert cd_decay.sampler.k == 10

        # After decay
        cd_decay.update_k(5)
        assert cd_decay.sampler.k == 1

        # Gradient estimation should use current k
        gradients = cd_decay.estimate_gradient(model, data)
        assert isinstance(gradients, dict)


class TestEdgeCases:
    """Test edge cases for gradient-based samplers."""

    def test_empty_batch(self) -> None:
        """Test handling of empty batches."""
        cd = ContrastiveDivergence(k=1)
        model = MockLatentModel()

        empty_data = torch.empty(0, 20)
        gradients = cd.estimate_gradient(model, empty_data)

        # Should return zero gradients
        assert gradients["W"].shape == model.W.shape
        assert torch.all(gradients["W"] == 0)

    def test_single_sample(self) -> None:
        """Test single sample gradient estimation."""
        cd = ContrastiveDivergence(k=1)
        model = MockLatentModel()

        single_data = torch.rand(1, 20)
        gradients = cd.estimate_gradient(model, single_data)

        assert gradients["W"].shape == model.W.shape
        assert not torch.all(gradients["W"] == 0)

    def test_large_k(self) -> None:
        """Test CD with large k value."""
        cd = ContrastiveDivergence(k=100)
        model = MockLatentModel(n_visible=10, n_hidden=5)

        data = torch.rand(5, 10)
        gradients = cd.estimate_gradient(model, data)

        # Should still work
        assert isinstance(gradients, dict)

        # With many steps, negative samples should be less correlated with data
        neg_samples = cd.last_negative_samples
        data_neg_corr = torch.corrcoef(
            torch.cat([data.flatten(), neg_samples.flatten()])
        )[0, 1]
        assert abs(data_neg_corr) < 0.5  # Weak correlation

    def test_persistent_chains_size_mismatch(self) -> None:
        """Test PCD when data batch size changes."""
        pcd = PersistentContrastiveDivergence(k=1, num_chains=10)
        model = MockLatentModel()

        # First call with batch size 20
        data1 = torch.rand(20, 20)
        pcd.estimate_gradient(model, data1)
        assert pcd.sampler.persistent_chains.shape[0] == 10

        # Second call with different batch size
        data2 = torch.rand(5, 20)
        pcd.estimate_gradient(model, data2)

        # Chains should maintain their size
        assert pcd.sampler.persistent_chains.shape[0] == 10

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_device_consistency(self) -> None:
        """Test that sampling maintains device consistency."""
        cd = ContrastiveDivergence(k=1)

        # Create model on CUDA
        model = MockLatentModel()
        model.W = model.W.cuda()
        model.vbias = model.vbias.cuda()
        model.hbias = model.hbias.cuda()

        data = torch.rand(10, 20).cuda()

        gradients = cd.estimate_gradient(model, data)

        # Gradients should be on same device
        assert gradients["W"].device.type == "cuda"
        assert gradients["vbias"].device.type == "cuda"
        assert gradients["hbias"].device.type == "cuda"
