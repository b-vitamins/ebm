"""Unit tests for RBM base class."""

import torch
import torch.nn as nn
from torch import Tensor

from ebm.core.config import RBMConfig
from ebm.models.rbm.base import RBMAISAdapter, RBMBase


class ConcreteRBM(RBMBase):
    """Concrete RBM for testing."""

    def hidden_activation(self, pre_activation: Tensor) -> Tensor:
        return torch.sigmoid(pre_activation)

    def visible_activation(self, pre_activation: Tensor) -> Tensor:
        return torch.sigmoid(pre_activation)

    def _sample_from_prob(self, prob: Tensor) -> Tensor:
        return torch.bernoulli(prob)


class TestRBMBase:
    """Test RBMBase class."""

    def test_initialization(self, small_rbm_config):
        """Test RBM initialization."""
        rbm = ConcreteRBM(small_rbm_config)

        assert rbm.num_visible == 20
        assert rbm.num_hidden == 10
        assert rbm.use_bias is True

        # Check parameters exist
        assert hasattr(rbm, 'W')
        assert hasattr(rbm, 'vbias')
        assert hasattr(rbm, 'hbias')

        # Check shapes
        assert rbm.W.shape == (10, 20)
        assert rbm.vbias.shape == (20,)
        assert rbm.hbias.shape == (10,)

    def test_no_bias_initialization(self):
        """Test RBM without bias terms."""
        config = RBMConfig(
            visible_units=20,
            hidden_units=10,
            use_bias=False
        )
        rbm = ConcreteRBM(config)

        # Biases should be buffers, not parameters
        assert not isinstance(rbm.vbias, nn.Parameter)
        assert not isinstance(rbm.hbias, nn.Parameter)
        assert torch.all(rbm.vbias == 0)
        assert torch.all(rbm.hbias == 0)

    def test_parameter_initialization(self):
        """Test different parameter initialization methods."""
        # Xavier initialization
        config = RBMConfig(
            visible_units=100,
            hidden_units=50,
            weight_init="xavier_normal",
            bias_init=0.01
        )
        rbm = ConcreteRBM(config)

        # Weights should have appropriate scale
        weight_std = rbm.W.std().item()
        expected_std = (2.0 / (100 + 50)) ** 0.5
        assert abs(weight_std - expected_std) < 0.1

        # Biases should be initialized to 0.01
        assert torch.allclose(rbm.vbias, torch.full_like(rbm.vbias, 0.01))
        assert torch.allclose(rbm.hbias, torch.full_like(rbm.hbias, 0.01))

    def test_energy_computation(self):
        """Test energy computation."""
        config = RBMConfig(visible_units=5, hidden_units=3)
        rbm = ConcreteRBM(config)

        # Set known weights for testing
        with torch.no_grad():
            rbm.W.data = torch.tensor([
                [1.0, 0.0, -1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, -1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]
            ])
            rbm.vbias.data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
            rbm.hbias.data = torch.tensor([0.1, 0.2, 0.3])

        # Test with concatenated state
        v = torch.tensor([[1., 0., 1., 0., 1.]])
        h = torch.tensor([[1., 0., 1.]])
        x = torch.cat([v, h], dim=-1)

        energy = rbm.energy(x)

        # Manually compute expected energy
        interaction = -(h @ rbm.W @ v.T).item()
        v_bias = -(v @ rbm.vbias).item()
        h_bias = -(h @ rbm.hbias).item()
        expected = interaction + v_bias + h_bias

        assert torch.allclose(energy, torch.tensor([expected]))

    def test_joint_energy(self):
        """Test joint energy computation."""
        config = RBMConfig(visible_units=5, hidden_units=3)
        rbm = ConcreteRBM(config)

        v = torch.rand(10, 5)
        h = torch.rand(10, 3)

        # Basic joint energy
        energy = rbm.joint_energy(v, h)
        assert energy.shape == (10,)

        # With temperature
        beta = torch.rand(10)
        energy_beta = rbm.joint_energy(v, h, beta=beta)
        assert torch.allclose(energy_beta, energy * beta)

        # With parts
        parts = rbm.joint_energy(v, h, return_parts=True)
        assert isinstance(parts, dict)
        assert "interaction" in parts
        assert "visible_bias" in parts
        assert "hidden_bias" in parts
        assert "total" in parts

        # Check parts sum to total
        total = parts["interaction"] + parts["visible_bias"] + parts["hidden_bias"]
        assert torch.allclose(total, parts["total"])

    def test_free_energy(self):
        """Test free energy computation."""
        config = RBMConfig(visible_units=5, hidden_units=3)
        rbm = ConcreteRBM(config)

        v = torch.rand(10, 5)

        # Basic free energy
        free_energy = rbm.free_energy(v)
        assert free_energy.shape == (10,)

        # With temperature
        beta = 0.5
        rbm.free_energy(v, beta=beta)

        # Free energy should integrate out hidden units correctly
        # For small RBM, we can check by brute force
        # F(v) = -log(sum_h exp(-E(v,h)))
        all_h = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)

        for i in range(v.shape[0]):
            v_i = v[i:i+1].expand(8, -1)
            energies = rbm.joint_energy(v_i, all_h)
            free_energy_exact = -torch.logsumexp(-energies, dim=0)
            assert torch.allclose(free_energy[i], free_energy_exact, atol=1e-5)

    def test_sampling_hidden(self):
        """Test hidden unit sampling."""
        config = RBMConfig(visible_units=20, hidden_units=10, seed=42)
        rbm = ConcreteRBM(config)

        v = torch.rand(32, 20)

        # Sample without probabilities
        h = rbm.sample_hidden(v)
        assert h.shape == (32, 10)
        assert torch.all((h == 0) | (h == 1))

        # Sample with probabilities
        h_sample, h_prob = rbm.sample_hidden(v, return_prob=True)
        assert h_sample.shape == (32, 10)
        assert h_prob.shape == (32, 10)
        assert torch.all((h_prob >= 0) & (h_prob <= 1))

        # Check sampling is stochastic
        h1 = rbm.sample_hidden(v)
        h2 = rbm.sample_hidden(v)
        assert not torch.allclose(h1, h2)

    def test_sampling_visible(self):
        """Test visible unit sampling."""
        config = RBMConfig(visible_units=20, hidden_units=10, seed=42)
        rbm = ConcreteRBM(config)

        h = torch.rand(32, 10)

        # Sample without probabilities
        v = rbm.sample_visible(h)
        assert v.shape == (32, 20)
        assert torch.all((v == 0) | (v == 1))

        # Sample with probabilities
        v_sample, v_prob = rbm.sample_visible(h, return_prob=True)
        assert v_sample.shape == (32, 20)
        assert v_prob.shape == (32, 20)
        assert torch.all((v_prob >= 0) & (v_prob <= 1))

    def test_temperature_scaling(self):
        """Test temperature scaling in sampling."""
        config = RBMConfig(visible_units=20, hidden_units=10, seed=42)
        rbm = ConcreteRBM(config)

        v = torch.rand(100, 20)

        # Low temperature (high beta) should give more deterministic results
        _, h_prob_low_temp = rbm.sample_hidden(v, beta=10.0, return_prob=True)
        _, h_prob_high_temp = rbm.sample_hidden(v, beta=0.1, return_prob=True)

        # Check that low temperature probabilities are more extreme
        low_temp_entropy = -(h_prob_low_temp * torch.log(h_prob_low_temp + 1e-8) +
                            (1 - h_prob_low_temp) * torch.log(1 - h_prob_low_temp + 1e-8)).mean()
        high_temp_entropy = -(h_prob_high_temp * torch.log(h_prob_high_temp + 1e-8) +
                             (1 - h_prob_high_temp) * torch.log(1 - h_prob_high_temp + 1e-8)).mean()

        assert low_temp_entropy < high_temp_entropy

    def test_split_visible_hidden(self):
        """Test splitting concatenated states."""
        config = RBMConfig(visible_units=5, hidden_units=3)
        rbm = ConcreteRBM(config)

        x = torch.randn(10, 8)  # 5 visible + 3 hidden
        v, h = rbm._split_visible_hidden(x)

        assert v.shape == (10, 5)
        assert h.shape == (10, 3)
        assert torch.allclose(v, x[:, :5])
        assert torch.allclose(h, x[:, 5:])

    def test_init_from_data(self, synthetic_binary_data, make_data_loader):
        """Test initialization from data statistics."""
        config = RBMConfig(visible_units=100, hidden_units=50)
        rbm = ConcreteRBM(config)

        # Create data loader
        data_loader = make_data_loader(synthetic_binary_data["dataset"], batch_size=50)

        # Initialize from data
        rbm.init_from_data(data_loader)

        # Check visible bias is set based on data statistics
        data = synthetic_binary_data["data"]
        data_mean = data.mean(dim=0)
        expected_bias = torch.log(data_mean.clamp(0.01, 0.99) / (1 - data_mean.clamp(0.01, 0.99)))

        assert torch.allclose(rbm.vbias, expected_bias, atol=0.1)

    def test_effective_energy(self):
        """Test effective energy with regularization."""
        config = RBMConfig(
            visible_units=5,
            hidden_units=3,
            l2_weight=0.1,
            l1_weight=0.05
        )
        rbm = ConcreteRBM(config)

        v = torch.rand(10, 5)
        h = torch.rand(10, 3)

        # Compute effective energy
        eff_energy = rbm.effective_energy(v, h)
        base_energy = rbm.joint_energy(v, h)

        # Should include regularization
        l2_reg = 0.5 * 0.1 * (rbm.W ** 2).sum()
        l1_reg = 0.05 * rbm.W.abs().sum()
        expected = base_energy + l2_reg + l1_reg

        assert torch.allclose(eff_energy, expected)

    def test_ais_adapter_creation(self):
        """Test AIS adapter creation."""
        config = RBMConfig(visible_units=5, hidden_units=3)
        rbm = ConcreteRBM(config)

        adapter = rbm.ais_adapter()
        assert isinstance(adapter, RBMAISAdapter)
        assert adapter.rbm is rbm


class TestRBMAISAdapter:
    """Test RBM AIS adapter."""

    def test_initialization(self, simple_bernoulli_rbm):
        """Test AIS adapter initialization."""
        adapter = RBMAISAdapter(simple_bernoulli_rbm)

        assert adapter.rbm is simple_bernoulli_rbm
        assert adapter.base_model is simple_bernoulli_rbm

        # Check base parameters are copied
        assert torch.allclose(adapter.base_vbias, simple_bernoulli_rbm.vbias)
        assert torch.allclose(adapter.base_hbias, torch.zeros_like(simple_bernoulli_rbm.hbias))

    def test_base_log_partition(self, simple_bernoulli_rbm):
        """Test base partition function calculation."""
        adapter = RBMAISAdapter(simple_bernoulli_rbm)

        log_z_base = adapter.base_log_partition()

        # For independent Bernoulli units:
        # Z = prod_i (1 + exp(a_i)) * 2^num_hidden
        expected_log_z_v = torch.nn.functional.softplus(adapter.base_vbias).sum().item()
        expected_log_z_h = simple_bernoulli_rbm.num_hidden * torch.log(torch.tensor(2.0)).item()
        expected = expected_log_z_v + expected_log_z_h

        assert abs(log_z_base - expected) < 1e-5

    def test_base_energy(self, simple_bernoulli_rbm):
        """Test base energy computation."""
        adapter = RBMAISAdapter(simple_bernoulli_rbm)

        # Create test states
        v = torch.rand(5, simple_bernoulli_rbm.num_visible).round()
        h = torch.rand(5, simple_bernoulli_rbm.num_hidden).round()
        x = torch.cat([v, h], dim=-1)

        base_energy = adapter.base_energy(x)

        # Base energy has no interaction term
        expected = -(v @ adapter.base_vbias + h @ adapter.base_hbias)
        assert torch.allclose(base_energy, expected)

    def test_interpolated_energy(self, simple_bernoulli_rbm):
        """Test interpolated energy for AIS."""
        adapter = RBMAISAdapter(simple_bernoulli_rbm)

        v = torch.rand(5, simple_bernoulli_rbm.num_visible).round()
        h = torch.rand(5, simple_bernoulli_rbm.num_hidden).round()
        x = torch.cat([v, h], dim=-1)

        # At beta=0, should be base energy
        energy_0 = adapter.interpolated_energy(x, beta=0.0)
        base_energy = adapter.base_energy(x)
        assert torch.allclose(energy_0, base_energy)

        # At beta=1, should be full model energy
        energy_1 = adapter.interpolated_energy(x, beta=1.0)
        full_energy = simple_bernoulli_rbm.energy(x)
        assert torch.allclose(energy_1, full_energy)

        # At intermediate beta
        beta = 0.5
        energy_mid = adapter.interpolated_energy(x, beta=beta)

        # Manually compute interpolated energy
        interaction = torch.einsum('...h,...v->...', h, torch.nn.functional.linear(v, simple_bernoulli_rbm.W))
        v_bias_interp = (1 - beta) * adapter.base_vbias + beta * simple_bernoulli_rbm.vbias
        h_bias_interp = (1 - beta) * adapter.base_hbias + beta * simple_bernoulli_rbm.hbias
        expected = -(beta * interaction + v @ v_bias_interp + h @ h_bias_interp)

        assert torch.allclose(energy_mid, expected, atol=1e-5)

    def test_ais_beta_property(self, simple_bernoulli_rbm):
        """Test AIS beta property usage."""
        adapter = RBMAISAdapter(simple_bernoulli_rbm)

        x = torch.randn(5, simple_bernoulli_rbm.num_visible + simple_bernoulli_rbm.num_hidden).round()

        # Set beta and use default
        adapter.ais_beta = 0.7
        energy = adapter.interpolated_energy(x)  # Should use ais_beta
        energy_explicit = adapter.interpolated_energy(x, beta=0.7)

        assert torch.allclose(energy, energy_explicit)


class TestRBMEdgeCases:
    """Test edge cases for RBM."""

    def test_empty_batch(self, simple_bernoulli_rbm):
        """Test handling of empty batches."""
        v = torch.empty(0, simple_bernoulli_rbm.num_visible)
        h = torch.empty(0, simple_bernoulli_rbm.num_hidden)

        energy = simple_bernoulli_rbm.joint_energy(v, h)
        assert energy.shape == (0,)

        free_energy = simple_bernoulli_rbm.free_energy(v)
        assert free_energy.shape == (0,)

    def test_single_sample(self, simple_bernoulli_rbm):
        """Test single sample handling."""
        v = torch.rand(1, simple_bernoulli_rbm.num_visible)

        h = simple_bernoulli_rbm.sample_hidden(v)
        assert h.shape == (1, simple_bernoulli_rbm.num_hidden)

        free_energy = simple_bernoulli_rbm.free_energy(v)
        assert free_energy.shape == (1,)

    def test_large_batch(self, simple_bernoulli_rbm):
        """Test large batch handling."""
        # Large batch size
        v = torch.rand(10000, simple_bernoulli_rbm.num_visible)

        # Should handle without issues
        h = simple_bernoulli_rbm.sample_hidden(v)
        assert h.shape == (10000, simple_bernoulli_rbm.num_hidden)

        # Free energy computation should be stable
        free_energy = simple_bernoulli_rbm.free_energy(v)
        assert not torch.any(torch.isnan(free_energy))
        assert not torch.any(torch.isinf(free_energy))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        config = RBMConfig(visible_units=10, hidden_units=5)
        rbm = ConcreteRBM(config)

        # Set extreme weights
        with torch.no_grad():
            rbm.W.data = torch.randn_like(rbm.W) * 100
            rbm.vbias.data = torch.randn_like(rbm.vbias) * 50
            rbm.hbias.data = torch.randn_like(rbm.hbias) * 50

        v = torch.rand(10, 10)

        # Free energy should still be computable
        free_energy = rbm.free_energy(v)
        assert not torch.any(torch.isnan(free_energy))

        # Sampling probabilities should be in [0, 1]
        _, h_prob = rbm.sample_hidden(v, return_prob=True)
        assert torch.all((h_prob >= 0) & (h_prob <= 1))
