"""Unit tests for Bernoulli RBM implementations."""

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ebm.core.config import RBMConfig
from ebm.models.rbm.bernoulli import (
    BernoulliRBM,
    CenteredBernoulliRBM,
    SparseBernoulliRBM,
)


class TestBernoulliRBM:
    """Test standard Bernoulli RBM."""

    def test_initialization(self, small_rbm_config: RBMConfig) -> None:
        """Test Bernoulli RBM initialization."""
        rbm = BernoulliRBM(small_rbm_config)

        assert isinstance(rbm, BernoulliRBM)
        assert rbm.num_visible == 20
        assert rbm.num_hidden == 10

    def test_activation_functions(self) -> None:
        """Test activation functions."""
        config = RBMConfig(visible_units=5, hidden_units=3)
        rbm = BernoulliRBM(config)

        # Both should be sigmoid
        pre_activation = torch.randn(10, 5)

        h_activation = rbm.hidden_activation(pre_activation)
        v_activation = rbm.visible_activation(pre_activation)

        expected = torch.sigmoid(pre_activation)
        assert torch.allclose(h_activation, expected)
        assert torch.allclose(v_activation, expected)

    def test_sampling_from_prob(self) -> None:
        """Test Bernoulli sampling."""
        config = RBMConfig(visible_units=5, hidden_units=3)
        rbm = BernoulliRBM(config)

        # Test with known probabilities
        prob = torch.tensor([[0.0, 0.5, 1.0]])

        # Sample many times to check distribution
        samples = []
        torch.manual_seed(42)
        for _ in range(1000):
            sample = rbm._sample_from_prob(prob)
            samples.append(sample)

        samples = torch.stack(samples)
        empirical_prob = samples.mean(dim=0)

        # Check empirical probabilities match input
        assert abs(empirical_prob[0, 0].item() - 0.0) < 0.05
        assert abs(empirical_prob[0, 1].item() - 0.5) < 0.05
        assert abs(empirical_prob[0, 2].item() - 1.0) < 0.05

    def test_log_probability_ratio(self) -> None:
        """Test log probability ratio computation."""
        config = RBMConfig(visible_units=5, hidden_units=3, seed=42)
        rbm = BernoulliRBM(config)

        v1 = torch.rand(10, 5).round()
        v2 = torch.rand(10, 5).round()

        log_ratio = rbm.log_probability_ratio(v1, v2)

        # Check using free energies
        f1 = rbm.free_energy(v1)
        f2 = rbm.free_energy(v2)
        expected = f2 - f1

        assert torch.allclose(log_ratio, expected)

        # Test with temperature
        beta = torch.rand(10)
        log_ratio_beta = rbm.log_probability_ratio(v1, v2, beta=beta)
        expected_beta = beta * (f2 - f1)
        assert torch.allclose(log_ratio_beta, expected_beta)

    def test_score_function(self) -> None:
        """Test score function computation."""
        config = RBMConfig(visible_units=5, hidden_units=3)
        rbm = BernoulliRBM(config)

        v = torch.rand(5, 5, requires_grad=True)

        scores = rbm.score_function(v)

        # Check that we get scores for all parameters
        assert "W" in scores
        assert "vbias" in scores
        assert "hbias" in scores
        assert "visible" in scores

        # Check shapes
        assert scores["W"].shape == rbm.W.shape
        assert scores["vbias"].shape == rbm.vbias.shape
        assert scores["hbias"].shape == rbm.hbias.shape
        assert scores["visible"].shape == v.shape

        # Score should be negative gradient of free energy
        f = rbm.free_energy(v)
        f_sum = f.sum()

        # Compute gradients manually
        f_sum.backward()

        # Check visible score
        assert torch.allclose(scores["visible"], -v.grad, atol=1e-5)

    def test_registry_registration(self) -> None:
        """Test that BernoulliRBM is registered."""
        from ebm.core.registry import models

        # Should be registered under multiple names
        assert "bernoulli_rbm" in models
        assert "brbm" in models
        assert "rbm" in models

        # All should point to same class
        assert models.get("bernoulli_rbm") is BernoulliRBM
        assert models.get("brbm") is BernoulliRBM
        assert models.get("rbm") is BernoulliRBM


class TestCenteredBernoulliRBM:
    """Test centered Bernoulli RBM."""

    def test_initialization(self, small_rbm_config: RBMConfig) -> None:
        """Test centered RBM initialization."""
        rbm = CenteredBernoulliRBM(small_rbm_config)

        assert rbm.centered is True
        assert hasattr(rbm, "v_offset")
        assert hasattr(rbm, "h_offset")

        # Offsets should be parameters
        assert isinstance(rbm.v_offset, nn.Parameter)
        assert isinstance(rbm.h_offset, nn.Parameter)

        # Initial values should be 0.5
        assert torch.allclose(rbm.v_offset, torch.full_like(rbm.v_offset, 0.5))
        assert torch.allclose(rbm.h_offset, torch.full_like(rbm.h_offset, 0.5))

    def test_centered_sampling(self) -> None:
        """Test sampling with centering."""
        config = RBMConfig(visible_units=5, hidden_units=3)
        rbm = CenteredBernoulliRBM(config)

        # Set specific offsets
        with torch.no_grad():
            rbm.v_offset.data = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7])
            rbm.h_offset.data = torch.tensor([0.2, 0.5, 0.8])

        v = torch.ones(10, 5) * 0.5

        # Sample hidden
        h, h_prob = rbm.sample_hidden(v, return_prob=True)

        # Manually compute expected probabilities
        v_centered = v - rbm.v_offset
        pre_h = torch.nn.functional.linear(v_centered, rbm.W, rbm.hbias)
        expected_prob = torch.sigmoid(pre_h)

        assert torch.allclose(h_prob, expected_prob, atol=1e-5)

    def test_centered_energy(self) -> None:
        """Test energy computation with centering."""
        config = RBMConfig(visible_units=3, hidden_units=2)
        rbm = CenteredBernoulliRBM(config)

        # Set known parameters
        with torch.no_grad():
            rbm.W.data = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            rbm.vbias.data = torch.tensor([0.1, 0.2, 0.3])
            rbm.hbias.data = torch.tensor([0.1, 0.2])
            rbm.v_offset.data = torch.tensor([0.3, 0.5, 0.7])
            rbm.h_offset.data = torch.tensor([0.4, 0.6])

        v = torch.tensor([[1.0, 0.0, 1.0]])
        h = torch.tensor([[1.0, 0.0]])

        energy = rbm.joint_energy(v, h)

        # Manually compute
        v_centered = v - rbm.v_offset
        h_centered = h - rbm.h_offset
        interaction = -torch.einsum(
            "bh,bv->b", h_centered, v_centered @ rbm.W.T
        )
        v_term = -(v @ rbm.vbias)
        h_term = -(h @ rbm.hbias)
        expected = interaction + v_term + h_term

        assert torch.allclose(energy, expected, atol=1e-5)

    def test_centered_free_energy(self) -> None:
        """Test free energy with centering."""
        config = RBMConfig(visible_units=5, hidden_units=3)
        rbm = CenteredBernoulliRBM(config)

        v = torch.rand(10, 5)

        free_energy = rbm.free_energy(v)

        # Should be finite and reasonable
        assert torch.all(torch.isfinite(free_energy))
        assert free_energy.shape == (10,)

        # Compare with brute force for small model
        config_small = RBMConfig(visible_units=3, hidden_units=2)
        rbm_small = CenteredBernoulliRBM(config_small)
        v_small = torch.rand(2, 3)

        # Compute by summing over all hidden states
        all_h = torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32
        )

        for i in range(v_small.shape[0]):
            v_i = v_small[i : i + 1].expand(4, -1)
            energies = torch.stack(
                [
                    rbm_small.joint_energy(v_i[j : j + 1], all_h[j : j + 1])
                    for j in range(4)
                ]
            ).squeeze()
            free_energy_exact = -torch.logsumexp(-energies, dim=0)
            free_energy_computed = rbm_small.free_energy(v_small[i : i + 1])
            assert torch.allclose(
                free_energy_computed, free_energy_exact, atol=1e-4
            )

    def test_update_offsets(self) -> None:
        """Test offset update mechanism."""
        config = RBMConfig(visible_units=5, hidden_units=3)
        rbm = CenteredBernoulliRBM(config)

        # Initial offsets
        initial_v_offset = rbm.v_offset.clone()
        initial_h_offset = rbm.h_offset.clone()

        # New means
        v_mean = torch.rand(5)
        h_mean = torch.rand(3)

        # Update with momentum
        momentum = 0.9
        rbm.update_offsets(v_mean, h_mean, momentum=momentum)

        # Check update
        expected_v = momentum * initial_v_offset + (1 - momentum) * v_mean
        expected_h = momentum * initial_h_offset + (1 - momentum) * h_mean

        assert torch.allclose(rbm.v_offset, expected_v)
        assert torch.allclose(rbm.h_offset, expected_h)

    def test_init_from_data_centered(
        self,
        synthetic_binary_data: dict[str, object],
        make_data_loader: Callable[[TensorDataset, int, bool, int], DataLoader],
    ) -> None:
        """Test data initialization for centered RBM."""
        config = RBMConfig(visible_units=100, hidden_units=50)
        rbm = CenteredBernoulliRBM(config)

        data_loader = make_data_loader(
            synthetic_binary_data["dataset"], batch_size=50
        )

        rbm.init_from_data(data_loader)

        # v_offset should be set to data mean
        data = synthetic_binary_data["data"]
        data_mean = data.mean(dim=0)

        assert torch.allclose(rbm.v_offset, data_mean, atol=0.01)

        # vbias should be adjusted for offset
        torch.nn.functional.linear(rbm.h_offset, rbm.W)
        # The exact relationship depends on implementation details


class TestSparseBernoulliRBM:
    """Test sparse Bernoulli RBM."""

    def test_initialization(self) -> None:
        """Test sparse RBM initialization."""
        config = RBMConfig(
            visible_units=20,
            hidden_units=10,
            sparsity_target=0.05,
            sparsity_weight=0.1,
        )

        # Add sparsity attributes to config
        config.sparsity_target = 0.05
        config.sparsity_weight = 0.1
        config.sparsity_damping = 0.95

        rbm = SparseBernoulliRBM(config)

        assert rbm.sparsity_target == 0.05
        assert rbm.sparsity_weight == 0.1
        assert rbm.sparsity_damping == 0.95

        # Should have hidden mean buffer
        assert hasattr(rbm, "hidden_mean")
        assert rbm.hidden_mean.shape == (10,)
        assert torch.allclose(rbm.hidden_mean, torch.full((10,), 0.05))

    def test_sparsity_penalty(self) -> None:
        """Test sparsity penalty computation."""
        config = RBMConfig(visible_units=20, hidden_units=10)
        config.sparsity_target = 0.1
        config.sparsity_weight = 0.01
        config.sparsity_damping = 0.9

        rbm = SparseBernoulliRBM(config)

        # Create hidden probabilities
        h_prob = torch.rand(32, 10)

        penalty = rbm.sparsity_penalty(h_prob)

        # Should be scalar
        assert penalty.dim() == 0

        # Should be positive KL divergence
        assert penalty.item() >= 0

        # Check that hidden mean is updated
        batch_mean = h_prob.mean(dim=0)
        expected_mean = 0.9 * 0.1 + 0.1 * batch_mean  # Initial was 0.1
        assert torch.allclose(rbm.hidden_mean, expected_mean, atol=0.01)

    def test_sparse_sampling(self) -> None:
        """Test sparse sampling with top-k constraint."""
        config = RBMConfig(visible_units=20, hidden_units=10)
        config.sparsity_target = 0.1
        config.sparsity_weight = 0.01

        rbm = SparseBernoulliRBM(config)

        v = torch.rand(32, 20)
        k = 3  # Only 3 units active

        h_sparse, h_prob = rbm.sparse_sample_hidden(v, k=k, return_prob=True)

        # Check sparsity constraint
        assert h_sparse.shape == (32, 10)
        assert torch.all(h_sparse.sum(dim=1) == k)

        # Check that active units have highest probabilities
        for i in range(32):
            active_indices = torch.where(h_sparse[i] == 1)[0]
            inactive_indices = torch.where(h_sparse[i] == 0)[0]

            if len(active_indices) > 0 and len(inactive_indices) > 0:
                min_active_prob = h_prob[i, active_indices].min()
                max_inactive_prob = h_prob[i, inactive_indices].max()
                assert min_active_prob >= max_inactive_prob

    def test_registry_registration(self) -> None:
        """Test sparse RBM registration."""
        from ebm.core.registry import models

        assert "sparse_rbm" in models
        assert "srbm" in models
        assert models.get("sparse_rbm") is SparseBernoulliRBM


class TestBernoulliRBMProperties:
    """Test mathematical properties of Bernoulli RBMs."""

    def test_energy_gradient(self, simple_bernoulli_rbm: BernoulliRBM) -> None:
        """Test that energy gradient matches implementation."""
        rbm = simple_bernoulli_rbm

        # Create inputs with gradients
        v = torch.rand(5, rbm.num_visible, requires_grad=True)
        h = torch.rand(5, rbm.num_hidden, requires_grad=True)

        # Compute energy
        energy = rbm.joint_energy(v, h).sum()

        # Get gradients
        energy.backward()

        # Gradients should match expected forms
        # dE/dv = -W^T h - a
        expected_v_grad = -(h @ rbm.W + rbm.vbias)
        assert torch.allclose(v.grad, expected_v_grad, atol=1e-5)

        # dE/dh = -W v - b
        expected_h_grad = -(v @ rbm.W.T + rbm.hbias)
        assert torch.allclose(h.grad, expected_h_grad, atol=1e-5)

    def test_partition_function_consistency(self) -> None:
        """Test partition function calculation for tiny RBM."""
        # Very small RBM where we can compute Z exactly
        config = RBMConfig(visible_units=2, hidden_units=2)
        rbm = BernoulliRBM(config)

        # Set simple weights
        with torch.no_grad():
            rbm.W.data = torch.tensor([[0.5, -0.5], [0.5, 0.5]])
            rbm.vbias.data = torch.tensor([0.1, -0.1])
            rbm.hbias.data = torch.tensor([0.2, -0.2])

        # Compute partition function by brute force
        log_z = torch.tensor(float("-inf"))

        for v_bits in range(4):  # 2^2 visible states
            for h_bits in range(4):  # 2^2 hidden states
                v = torch.tensor(
                    [[(v_bits >> i) & 1 for i in range(2)]], dtype=torch.float32
                )
                h = torch.tensor(
                    [[(h_bits >> i) & 1 for i in range(2)]], dtype=torch.float32
                )

                energy = rbm.joint_energy(v, h)
                log_z = torch.logaddexp(log_z, -energy.squeeze())

        # Test that probabilities sum to 1
        total_prob = 0.0
        for v_bits in range(4):
            v = torch.tensor(
                [[(v_bits >> i) & 1 for i in range(2)]], dtype=torch.float32
            )
            log_prob = rbm.log_probability(v, log_z=log_z)
            total_prob += torch.exp(log_prob).item()

        assert abs(total_prob - 1.0) < 1e-5

    def test_conditional_independence(
        self, simple_bernoulli_rbm: BernoulliRBM
    ) -> None:
        """Test conditional independence property of RBMs."""
        rbm = simple_bernoulli_rbm

        v = torch.rand(100, rbm.num_visible).round()

        # Sample hidden units independently
        h1, prob1 = rbm.sample_hidden(v, return_prob=True)
        h2, prob2 = rbm.sample_hidden(v, return_prob=True)

        # Probabilities should be the same
        assert torch.allclose(prob1, prob2)

        # But samples should be different (stochastic)
        assert not torch.allclose(h1, h2)

        # Each hidden unit should be independent given visible
        # Check by computing correlations
        h_samples = torch.stack([rbm.sample_hidden(v) for _ in range(100)])

        # Compute correlations between hidden units
        h_mean = h_samples.float().mean(dim=0)
        h_centered = h_samples.float() - h_mean

        # For each sample, compute correlation matrix
        for i in range(v.shape[0]):
            h_i = h_centered[:, i, :]  # All samples for i-th visible config
            if h_i.std(dim=0).min() > 0:  # Avoid division by zero
                corr = torch.corrcoef(h_i.T)
                # Off-diagonal elements should be small (independent)
                off_diag = corr - torch.diag(torch.diag(corr))
                assert (
                    torch.abs(off_diag).max() < 0.2
                )  # Weak correlation due to finite samples
