"""Unit tests for Gaussian RBM implementations."""

import torch
import torch.nn as nn

from ebm.core.config import GaussianRBMConfig
from ebm.models.rbm.gaussian import GaussianBernoulliRBM, WhitenedGaussianRBM


class TestGaussianBernoulliRBM:
    """Test Gaussian-Bernoulli RBM."""

    def test_initialization(self, gaussian_rbm_config):
        """Test Gaussian RBM initialization."""
        rbm = GaussianBernoulliRBM(gaussian_rbm_config)

        assert isinstance(rbm, GaussianBernoulliRBM)
        assert rbm.num_visible == 100
        assert rbm.num_hidden == 50
        assert rbm.learn_sigma is True

        # Check sigma parameter
        assert hasattr(rbm, 'log_sigma')
        assert isinstance(rbm.log_sigma, nn.Parameter)
        assert rbm.log_sigma.shape == (100,)

    def test_fixed_sigma(self):
        """Test Gaussian RBM with fixed sigma."""
        config = GaussianRBMConfig(
            visible_units=20,
            hidden_units=10,
            sigma=2.0,
            learn_sigma=False
        )
        rbm = GaussianBernoulliRBM(config)

        # log_sigma should be a buffer, not parameter
        assert not isinstance(rbm.log_sigma, nn.Parameter)
        assert torch.allclose(rbm.log_sigma, torch.log(torch.tensor(2.0)))
        assert torch.allclose(rbm.sigma, torch.tensor(2.0))

    def test_sigma_properties(self):
        """Test sigma and sigma_sq properties."""
        config = GaussianRBMConfig(
            visible_units=10,
            hidden_units=5,
            sigma=1.5,
            learn_sigma=True
        )
        rbm = GaussianBernoulliRBM(config)

        # Set log_sigma
        with torch.no_grad():
            rbm.log_sigma.data = torch.log(torch.tensor([1.0, 2.0, 0.5] + [1.5] * 7))

        sigma = rbm.sigma
        sigma_sq = rbm.sigma_sq

        assert torch.allclose(sigma[:3], torch.tensor([1.0, 2.0, 0.5]))
        assert torch.allclose(sigma_sq[:3], torch.tensor([1.0, 4.0, 0.25]))

    def test_activation_functions(self):
        """Test activation functions for Gaussian RBM."""
        config = GaussianRBMConfig(visible_units=10, hidden_units=5)
        rbm = GaussianBernoulliRBM(config)

        pre_activation = torch.randn(5, 10)

        # Hidden activation should be sigmoid
        h_activation = rbm.hidden_activation(pre_activation)
        assert torch.allclose(h_activation, torch.sigmoid(pre_activation))

        # Visible activation should be identity
        v_activation = rbm.visible_activation(pre_activation)
        assert torch.allclose(v_activation, pre_activation)

    def test_sample_visible_gaussian(self):
        """Test Gaussian visible unit sampling."""
        config = GaussianRBMConfig(
            visible_units=20,
            hidden_units=10,
            sigma=0.5,
            learn_sigma=False,
            seed=42
        )
        rbm = GaussianBernoulliRBM(config)

        h = torch.rand(32, 10).round()

        # Sample visible units
        v_sample, v_mean = rbm.sample_visible(h, return_prob=True)

        assert v_sample.shape == (32, 20)
        assert v_mean.shape == (32, 20)

        # Mean should be deterministic given h
        v_mean2 = rbm.sample_visible(h, return_prob=True)[1]
        assert torch.allclose(v_mean, v_mean2)

        # Samples should be different (stochastic)
        v_sample2 = rbm.sample_visible(h)
        assert not torch.allclose(v_sample, v_sample2)

        # Check that samples are distributed around mean
        n_samples = 1000
        samples = torch.stack([rbm.sample_visible(h[0:1]) for _ in range(n_samples)]).squeeze(1)
        empirical_mean = samples.mean(dim=0)
        empirical_std = samples.std(dim=0)

        assert torch.allclose(empirical_mean, v_mean[0], atol=0.1)
        assert torch.allclose(empirical_std, torch.full((20,), 0.5), atol=0.1)

    def test_sample_visible_with_temperature(self):
        """Test temperature effects on visible sampling."""
        config = GaussianRBMConfig(
            visible_units=10,
            hidden_units=5,
            sigma=1.0,
            learn_sigma=False
        )
        rbm = GaussianBernoulliRBM(config)

        h = torch.ones(10, 5)

        # High temperature (low beta) - more variance
        rbm.sample_visible(h, beta=torch.tensor(0.1))

        # Low temperature (high beta) - less variance
        rbm.sample_visible(h, beta=torch.tensor(10.0))

        # Sample many times to check variance
        samples_high = torch.stack([
            rbm.sample_visible(h[0:1], beta=torch.tensor(0.1))
            for _ in range(100)
        ]).squeeze(1)
        samples_low = torch.stack([
            rbm.sample_visible(h[0:1], beta=torch.tensor(10.0))
            for _ in range(100)
        ]).squeeze(1)

        # High temperature should have higher variance
        var_high = samples_high.var(dim=0).mean()
        var_low = samples_low.var(dim=0).mean()
        assert var_high > var_low

    def test_joint_energy_gaussian(self):
        """Test joint energy computation for Gaussian units."""
        config = GaussianRBMConfig(
            visible_units=3,
            hidden_units=2,
            sigma=0.5,
            learn_sigma=False
        )
        rbm = GaussianBernoulliRBM(config)

        # Set known parameters
        with torch.no_grad():
            rbm.W.data = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            rbm.vbias.data = torch.tensor([0.1, 0.2, 0.3])
            rbm.hbias.data = torch.tensor([0.1, 0.2])

        v = torch.tensor([[1.0, 0.5, -0.5]])
        h = torch.tensor([[1.0, 0.0]])

        energy = rbm.joint_energy(v, h)

        # Manually compute
        sigma_sq = 0.25
        v_normalized = v / sigma_sq

        quadratic = 0.5 * ((v - rbm.vbias) ** 2 / sigma_sq).sum()
        h_linear = (h * rbm.hbias).sum()
        interaction = (h @ rbm.W @ v_normalized.T).item()

        expected = quadratic - h_linear - interaction
        assert torch.allclose(energy, torch.tensor([expected]), atol=1e-5)

        # Test with parts
        parts = rbm.joint_energy(v, h, return_parts=True)
        assert "visible_quadratic" in parts
        assert "hidden_linear" in parts
        assert "interaction" in parts
        assert torch.allclose(parts["total"], energy)

    def test_free_energy_gaussian(self):
        """Test free energy for Gaussian units."""
        config = GaussianRBMConfig(
            visible_units=5,
            hidden_units=3,
            sigma=1.0,
            learn_sigma=False
        )
        rbm = GaussianBernoulliRBM(config)

        v = torch.randn(10, 5)

        free_energy = rbm.free_energy(v)
        assert free_energy.shape == (10,)
        assert torch.all(torch.isfinite(free_energy))

        # Compare with brute force for small model
        config_small = GaussianRBMConfig(
            visible_units=3,
            hidden_units=2,
            sigma=1.0,
            learn_sigma=False
        )
        rbm_small = GaussianBernoulliRBM(config_small)

        v_small = torch.randn(2, 3)
        all_h = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

        for i in range(v_small.shape[0]):
            v_i = v_small[i:i+1].expand(4, -1)
            energies = torch.stack([
                rbm_small.joint_energy(v_i[j:j+1], all_h[j:j+1])
                for j in range(4)
            ]).squeeze()
            free_energy_exact = -torch.logsumexp(-energies, dim=0)
            free_energy_computed = rbm_small.free_energy(v_small[i:i+1])
            assert torch.allclose(free_energy_computed, free_energy_exact, atol=1e-4)

    def test_score_matching_loss(self):
        """Test denoising score matching loss."""
        config = GaussianRBMConfig(
            visible_units=20,
            hidden_units=10,
            sigma=1.0,
            learn_sigma=False
        )
        rbm = GaussianBernoulliRBM(config)

        v = torch.randn(32, 20)
        noise_std = 0.1

        loss = rbm.score_matching_loss(v, noise_std=noise_std)

        # Loss should be scalar and positive
        assert loss.dim() == 0
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_fantasy_particles(self):
        """Test fantasy particle generation."""
        config = GaussianRBMConfig(
            visible_units=10,
            hidden_units=5,
            sigma=1.0,
            learn_sigma=False,
            seed=42
        )
        rbm = GaussianBernoulliRBM(config)

        # Generate samples
        samples = rbm.sample_fantasy_particles(
            num_samples=20,
            num_steps=100
        )

        assert samples.shape == (20, 10)
        assert torch.all(torch.isfinite(samples))

        # Test with initialization from data
        init_data = torch.randn(30, 10)
        samples_init = rbm.sample_fantasy_particles(
            num_samples=20,
            num_steps=100,
            init_from_data=init_data
        )

        assert samples_init.shape == (20, 10)

        # Test returning chain
        final, chain = rbm.sample_fantasy_particles(
            num_samples=5,
            num_steps=10,
            return_chain=True
        )

        assert final.shape == (5, 10)
        assert len(chain) == 11  # Initial + 10 steps
        assert all(s.shape == (5, 10) for s in chain)

    def test_registry_registration(self):
        """Test Gaussian RBM registration."""
        from ebm.core.registry import models

        assert "gaussian_rbm" in models
        assert "grbm" in models
        assert "gbrbm" in models
        assert models.get("gaussian_rbm") is GaussianBernoulliRBM


class TestWhitenedGaussianRBM:
    """Test whitened Gaussian RBM."""

    def test_initialization(self):
        """Test whitened RBM initialization."""
        config = GaussianRBMConfig(
            visible_units=20,
            hidden_units=10,
            sigma=1.0,
            learn_sigma=False
        )
        rbm = WhitenedGaussianRBM(config)

        assert hasattr(rbm, 'whitening_mean')
        assert hasattr(rbm, 'whitening_std')
        assert rbm.fitted is False

        # Whitening parameters should be None initially
        assert rbm.whitening_mean is None
        assert rbm.whitening_std is None

    def test_fit_whitening(self, synthetic_continuous_data, make_data_loader):
        """Test fitting whitening transformation."""
        config = GaussianRBMConfig(
            visible_units=50,
            hidden_units=25,
            sigma=1.0,
            learn_sigma=False
        )
        rbm = WhitenedGaussianRBM(config)

        data_loader = make_data_loader(
            synthetic_continuous_data["dataset"],
            batch_size=100
        )

        # Fit whitening
        rbm.fit_whitening(data_loader)

        assert rbm.fitted is True
        assert rbm.whitening_mean is not None
        assert rbm.whitening_std is not None

        # Check that parameters match data statistics
        data = synthetic_continuous_data["data"]
        expected_mean = data.mean(dim=0)
        expected_std = data.std(dim=0)

        assert torch.allclose(rbm.whitening_mean, expected_mean, atol=0.01)
        assert torch.allclose(rbm.whitening_std, expected_std, atol=0.01)

        # Visible bias should be reset
        assert torch.allclose(rbm.vbias, torch.zeros_like(rbm.vbias))

        # If not learning sigma, should be set to 1
        if not rbm.learn_sigma:
            assert torch.allclose(rbm.sigma, torch.ones_like(rbm.sigma))

    def test_whitening_transform(self):
        """Test whitening and unwhitening."""
        config = GaussianRBMConfig(
            visible_units=10,
            hidden_units=5,
            sigma=1.0,
            learn_sigma=False
        )
        rbm = WhitenedGaussianRBM(config)

        # Set whitening parameters manually
        rbm.whitening_mean = torch.tensor([1.0, 2.0, 3.0] + [0.0] * 7)
        rbm.whitening_std = torch.tensor([2.0, 0.5, 1.0] + [1.0] * 7)
        rbm.fitted = True

        # Test data
        v = torch.tensor([[2.0, 2.5, 4.0] + [0.0] * 7])

        # Whiten
        v_white = rbm.whiten(v)
        expected = torch.tensor([[0.5, 1.0, 1.0] + [0.0] * 7])
        assert torch.allclose(v_white, expected)

        # Unwhiten
        v_recovered = rbm.unwhiten(v_white)
        assert torch.allclose(v_recovered, v)

    def test_prepare_input_whitening(self):
        """Test that prepare_input applies whitening."""
        config = GaussianRBMConfig(visible_units=5, hidden_units=3)
        rbm = WhitenedGaussianRBM(config)

        # Before fitting, should not change input
        v = torch.randn(10, 5)
        v_prepared = rbm.prepare_input(v)
        assert torch.allclose(v_prepared, v)

        # After fitting, should whiten
        rbm.whitening_mean = torch.zeros(5)
        rbm.whitening_std = torch.ones(5) * 2.0
        rbm.fitted = True

        v_prepared = rbm.prepare_input(v)
        assert torch.allclose(v_prepared, v / 2.0)

    def test_fantasy_particles_whitened(self):
        """Test fantasy particle generation with unwhitening."""
        config = GaussianRBMConfig(
            visible_units=10,
            hidden_units=5,
            sigma=1.0,
            learn_sigma=False,
            seed=42
        )
        rbm = WhitenedGaussianRBM(config)

        # Set whitening parameters
        rbm.whitening_mean = torch.ones(10) * 5.0
        rbm.whitening_std = torch.ones(10) * 2.0
        rbm.fitted = True

        # Generate samples without unwhitening
        samples_white = rbm.sample_fantasy_particles(
            num_samples=100,
            num_steps=500,
            unwhiten_output=False
        )

        # Should be centered around 0 (whitened space)
        assert abs(samples_white.mean()) < 0.5

        # Generate samples with unwhitening
        samples = rbm.sample_fantasy_particles(
            num_samples=100,
            num_steps=500,
            unwhiten_output=True
        )

        # Should be in original scale
        assert abs(samples.mean() - 5.0) < 1.0
        assert abs(samples.std() - 2.0) < 0.5

    def test_registry_registration(self):
        """Test whitened Gaussian RBM registration."""
        from ebm.core.registry import models

        assert "gaussian_rbm_whitened" in models
        assert "grbm_w" in models
        assert models.get("gaussian_rbm_whitened") is WhitenedGaussianRBM


class TestGaussianRBMProperties:
    """Test mathematical properties of Gaussian RBMs."""

    def test_energy_continuous_visible(self):
        """Test that energy handles continuous visible units correctly."""
        config = GaussianRBMConfig(
            visible_units=5,
            hidden_units=3,
            sigma=1.0,
            learn_sigma=False
        )
        rbm = GaussianBernoulliRBM(config)

        # Continuous visible units
        v = torch.randn(10, 5) * 2.0
        h = torch.rand(10, 3).round()

        energy = rbm.joint_energy(v, h)

        # Energy should be finite
        assert torch.all(torch.isfinite(energy))

        # Energy should increase with distance from bias
        v_near_bias = rbm.vbias.unsqueeze(0).expand(10, -1) + torch.randn(10, 5) * 0.1
        v_far_from_bias = rbm.vbias.unsqueeze(0).expand(10, -1) + torch.randn(10, 5) * 5.0

        energy_near = rbm.joint_energy(v_near_bias, h)
        energy_far = rbm.joint_energy(v_far_from_bias, h)

        assert energy_near.mean() < energy_far.mean()

    def test_gradient_consistency(self):
        """Test gradient consistency for Gaussian units."""
        config = GaussianRBMConfig(
            visible_units=5,
            hidden_units=3,
            sigma=0.5,
            learn_sigma=True
        )
        rbm = GaussianBernoulliRBM(config)

        v = torch.randn(10, 5, requires_grad=True)
        h = torch.rand(10, 3).round()

        # Compute energy
        energy = rbm.joint_energy(v, h).sum()
        energy.backward()

        # Numerical gradient for v
        eps = 1e-4
        v_grad_numerical = torch.zeros_like(v)

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_plus = v.clone()
                v_plus[i, j] += eps
                e_plus = rbm.joint_energy(v_plus, h)[i]

                v_minus = v.clone()
                v_minus[i, j] -= eps
                e_minus = rbm.joint_energy(v_minus, h)[i]

                v_grad_numerical[i, j] = (e_plus - e_minus) / (2 * eps)

        assert torch.allclose(v.grad, v_grad_numerical, atol=1e-3)

    def test_partition_independence(self):
        """Test that visible partition function depends on sigma."""
        config1 = GaussianRBMConfig(
            visible_units=3,
            hidden_units=2,
            sigma=1.0,
            learn_sigma=False
        )
        config2 = GaussianRBMConfig(
            visible_units=3,
            hidden_units=2,
            sigma=2.0,
            learn_sigma=False
        )

        rbm1 = GaussianBernoulliRBM(config1)
        rbm2 = GaussianBernoulliRBM(config2)

        # Set same weights and biases
        with torch.no_grad():
            weights = torch.randn(2, 3)
            vbias = torch.randn(3)
            hbias = torch.randn(2)

            rbm1.W.data = weights.clone()
            rbm1.vbias.data = vbias.clone()
            rbm1.hbias.data = hbias.clone()

            rbm2.W.data = weights.clone()
            rbm2.vbias.data = vbias.clone()
            rbm2.hbias.data = hbias.clone()

        # Free energies should be different due to different sigma
        v = torch.randn(5, 3)
        f1 = rbm1.free_energy(v)
        f2 = rbm2.free_energy(v)

        assert not torch.allclose(f1, f2)

    def test_learned_sigma_gradient(self):
        """Test that sigma can be learned via gradients."""
        config = GaussianRBMConfig(
            visible_units=10,
            hidden_units=5,
            sigma=1.0,
            learn_sigma=True
        )
        rbm = GaussianBernoulliRBM(config)

        # Create synthetic data with known variance
        true_sigma = 2.0
        v = torch.randn(100, 10) * true_sigma

        # Compute loss (negative log-likelihood proxy)
        free_energy = rbm.free_energy(v).mean()
        free_energy.backward()

        # log_sigma should have gradients
        assert rbm.log_sigma.grad is not None
        assert not torch.all(rbm.log_sigma.grad == 0)
