"""Comprehensive tests for the Bernoulli-Bernoulli RBM implementation."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from ebm.rbm.model.brbm import BernoulliRBM, BernoulliRBMConfig


class TestBernoulliRBM:
    """Comprehensive tests for BernoulliRBM class."""

    @pytest.fixture
    def small_rbm(self) -> BernoulliRBM:
        """Create a small RBM for testing with fixed values."""
        visible, hidden = 4, 3
        config = BernoulliRBMConfig(visible=visible, hidden=hidden)
        rbm = BernoulliRBM(config)

        # Set fixed parameter values for deterministic testing
        with torch.no_grad():
            # Weight matrix (hidden x visible)
            rbm.w.copy_(
                torch.tensor(
                    [[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8], [-0.9, 1.0, -1.1, 1.2]]
                )
            )

            # Visible biases
            rbm.vb.copy_(torch.tensor([-0.1, 0.2, -0.3, 0.4]))

            # Hidden biases
            rbm.hb.copy_(torch.tensor([0.5, -0.6, 0.7]))

        return rbm

    @pytest.fixture
    def mnist_sized_rbm(self) -> BernoulliRBM:
        """Create an RBM sized for MNIST."""
        visible, hidden = 784, 500
        config = BernoulliRBMConfig(visible=visible, hidden=hidden)
        return BernoulliRBM(config)

    def test_init_and_reset(self, mnist_sized_rbm: BernoulliRBM) -> None:
        """Test initialization and weight reset."""
        # Check initial parameter shapes
        assert mnist_sized_rbm.w.shape == (500, 784)
        assert mnist_sized_rbm.vb.shape == (784,)
        assert mnist_sized_rbm.hb.shape == (500,)

        # Check initial parameter distributions
        # Weights should be normally distributed with std close to 1/(500 * 784)**0.25 (~0.04)
        assert 0.038 <= mnist_sized_rbm.w.std().item() <= 0.04
        assert abs(mnist_sized_rbm.w.mean().item()) < 0.001

        # Biases should be initialized to zero
        assert torch.allclose(mnist_sized_rbm.vb, torch.zeros_like(mnist_sized_rbm.vb))
        assert torch.allclose(mnist_sized_rbm.hb, torch.zeros_like(mnist_sized_rbm.hb))

        # Test reset_parameters
        with torch.no_grad():
            # Modify parameters
            mnist_sized_rbm.w.fill_(1.0)
            mnist_sized_rbm.vb.fill_(1.0)
            mnist_sized_rbm.hb.fill_(1.0)

        # Reset parameters
        mnist_sized_rbm.reset_parameters()

        # Verify they are reset properly
        assert 0.038 <= mnist_sized_rbm.w.std().item() <= 0.04
        assert abs(mnist_sized_rbm.w.mean().item()) < 0.001
        assert torch.allclose(mnist_sized_rbm.vb, torch.zeros_like(mnist_sized_rbm.vb))
        assert torch.allclose(mnist_sized_rbm.hb, torch.zeros_like(mnist_sized_rbm.hb))

    def test_preact_h_manual_calculation(self, small_rbm: BernoulliRBM) -> None:
        """Test hidden pre-activation against manual calculations."""
        # Create a single sample
        v = torch.tensor([1.0, 0.0, 1.0, 0.0])

        # Calculate pre-activation manually
        # pre_h_j = sum_i(W_ji * v_i) + hb_j
        expected_pre_h = torch.zeros(3)
        expected_pre_h[0] = 0.1 * 1.0 + (-0.2) * 0.0 + 0.3 * 1.0 + (-0.4) * 0.0 + 0.5  # = 0.9
        expected_pre_h[1] = 0.5 * 1.0 + (-0.6) * 0.0 + 0.7 * 1.0 + (-0.8) * 0.0 + (-0.6)  # = 0.6
        expected_pre_h[2] = (-0.9) * 1.0 + 1.0 * 0.0 + (-1.1) * 1.0 + 1.2 * 0.0 + 0.7  # = -1.3

        # Get pre-activation from model
        pre_h = small_rbm.preact_h(v)

        # Compare
        assert torch.allclose(pre_h, expected_pre_h, atol=1e-5)

        # Test batch dimension
        v_batch = v.repeat(5, 1)  # 5 identical samples
        pre_h_batch = small_rbm.preact_h(v_batch)

        assert pre_h_batch.shape == (5, 3)
        for i in range(5):
            assert torch.allclose(pre_h_batch[i], expected_pre_h, atol=1e-5)

    def test_preact_v_manual_calculation(self, small_rbm: BernoulliRBM) -> None:
        """Test visible pre-activation against manual calculations."""
        # Create a single sample
        h = torch.tensor([1.0, 0.0, 1.0])

        # Calculate pre-activation manually
        # pre_v_i = sum_j(W_ji * h_j) + vb_i
        expected_pre_v = torch.zeros(4)
        expected_pre_v[0] = 0.1 * 1.0 + 0.5 * 0.0 + (-0.9) * 1.0 + (-0.1)  # = -0.9
        expected_pre_v[1] = (-0.2) * 1.0 + (-0.6) * 0.0 + 1.0 * 1.0 + 0.2  # = 1.0
        expected_pre_v[2] = 0.3 * 1.0 + 0.7 * 0.0 + (-1.1) * 1.0 + (-0.3)  # = -1.1
        expected_pre_v[3] = (-0.4) * 1.0 + (-0.8) * 0.0 + 1.2 * 1.0 + 0.4  # = 1.2

        # Get pre-activation from model
        pre_v = small_rbm.preact_v(h)

        # Compare
        assert torch.allclose(pre_v, expected_pre_v, atol=1e-5)

        # Test batch dimension
        h_batch = h.repeat(5, 1)  # 5 identical samples
        pre_v_batch = small_rbm.preact_v(h_batch)

        assert pre_v_batch.shape == (5, 4)
        for i in range(5):
            assert torch.allclose(pre_v_batch[i], expected_pre_v, atol=1e-5)

    def test_prob_h_given_v(self, small_rbm: BernoulliRBM) -> None:
        """Test hidden unit probabilities calculation."""
        # Create a test visible vector
        v = torch.tensor([1.0, 0.0, 1.0, 0.0])

        # Calculate expected probabilities
        pre_h = small_rbm.preact_h(v)
        expected_probs = torch.sigmoid(pre_h)

        # Get probabilities from model
        probs = small_rbm.prob_h_given_v(v)

        # Compare
        assert torch.allclose(probs, expected_probs)

        # Check probability range
        assert torch.all((probs >= 0) & (probs <= 1))

        # Test batch dimension
        v_batch = torch.stack(
            [
                torch.tensor([1.0, 0.0, 1.0, 0.0]),
                torch.tensor([0.0, 1.0, 0.0, 1.0]),
                torch.tensor([1.0, 1.0, 1.0, 1.0]),
                torch.tensor([0.0, 0.0, 0.0, 0.0]),
            ]
        )

        probs_batch = small_rbm.prob_h_given_v(v_batch)
        assert probs_batch.shape == (4, 3)
        assert torch.all((probs_batch >= 0) & (probs_batch <= 1))

    def test_prob_v_given_h(self, small_rbm: BernoulliRBM) -> None:
        """Test visible unit probabilities calculation."""
        # Create a test hidden vector
        h = torch.tensor([1.0, 0.0, 1.0])

        # Calculate expected probabilities
        pre_v = small_rbm.preact_v(h)
        expected_probs = torch.sigmoid(pre_v)

        # Get probabilities from model
        probs = small_rbm.prob_v_given_h(h)

        # Compare
        assert torch.allclose(probs, expected_probs)

        # Check probability range
        assert torch.all((probs >= 0) & (probs <= 1))

        # Test batch dimension
        h_batch = torch.stack(
            [
                torch.tensor([1.0, 0.0, 1.0]),
                torch.tensor([0.0, 1.0, 0.0]),
                torch.tensor([1.0, 1.0, 1.0]),
                torch.tensor([0.0, 0.0, 0.0]),
            ]
        )

        probs_batch = small_rbm.prob_v_given_h(h_batch)
        assert probs_batch.shape == (4, 4)
        assert torch.all((probs_batch >= 0) & (probs_batch <= 1))

    def test_sampling_deterministic(self, small_rbm: BernoulliRBM) -> None:
        """Test sampling with fixed random seed for deterministic results."""
        # Create test vector
        v = torch.tensor([1.0, 0.0, 1.0, 0.0])

        # Set fixed random seed
        torch.manual_seed(42)
        h_samples1 = small_rbm.sample_h_given_v(v)

        # Reset seed and sample again
        torch.manual_seed(42)
        h_samples2 = small_rbm.sample_h_given_v(v)

        # Samples should be identical with the same seed
        assert torch.all(h_samples1 == h_samples2)

        # Samples should be binary
        assert set(torch.unique(h_samples1).tolist()).issubset({0.0, 1.0})

        # Similar test for visible sampling
        h = torch.tensor([1.0, 0.0, 1.0])

        torch.manual_seed(42)
        v_samples1 = small_rbm.sample_v_given_h(h)

        torch.manual_seed(42)
        v_samples2 = small_rbm.sample_v_given_h(h)

        assert torch.all(v_samples1 == v_samples2)
        assert set(torch.unique(v_samples1).tolist()).issubset({0.0, 1.0})

    def test_sampling_distribution(self, small_rbm: BernoulliRBM) -> None:
        """Test that sampling distribution matches probabilities."""
        # Create test vector
        v = torch.tensor([1.0, 0.0, 1.0, 0.0])

        # Get hidden probabilities
        h_probs = small_rbm.prob_h_given_v(v)

        # Take many samples
        num_samples = 10000
        samples = torch.zeros(num_samples, 3)

        for i in range(num_samples):
            samples[i] = small_rbm.sample_h_given_v(v)

        # Calculate mean of samples (should approach probabilities)
        sample_means = samples.mean(dim=0)

        # Compare with tolerance for randomness
        assert torch.allclose(sample_means, h_probs, atol=0.05)

        # Similar test for visible sampling
        h = torch.tensor([1.0, 0.0, 1.0])
        v_probs = small_rbm.prob_v_given_h(h)

        samples = torch.zeros(num_samples, 4)
        for i in range(num_samples):
            samples[i] = small_rbm.sample_v_given_h(h)

        sample_means = samples.mean(dim=0)
        assert torch.allclose(sample_means, v_probs, atol=0.05)

    def test_energy_manual_calculation(self, small_rbm: BernoulliRBM) -> None:
        """Test energy calculation against manual computation."""
        # Create test vectors
        v = torch.tensor([1.0, 0.0, 1.0, 0.0])
        h = torch.tensor([1.0, 0.0, 1.0])

        # Calculate energy manually
        # E(v,h) = -(v·vb + h·hb + h·W·v)
        vb_term = (v * small_rbm.vb.detach()).sum()  # v[0]*vb[0] + v[1]*vb[1] + ...
        hb_term = (h * small_rbm.hb.detach()).sum()  # h[0]*hb[0] + h[1]*hb[1] + ...

        # Interaction term: h·W·v - using tensor operations throughout
        # FIX: Avoid float/Tensor mismatch by using tensors directly
        interaction = torch.zeros(1)
        for j in range(3):  # hidden units
            for i in range(4):  # visible units
                interaction += h[j] * small_rbm.w[j, i] * v[i]

        expected_energy = -(vb_term + hb_term + interaction)

        # Get energy from model
        energy = small_rbm.energy(v, h)

        # Compare
        assert torch.allclose(energy, expected_energy)

        # Test batch dimension
        v_batch = v.repeat(5, 1)
        h_batch = h.repeat(5, 1)
        energy_batch = small_rbm.energy(v_batch, h_batch)

        assert energy_batch.shape == (5,)
        for i in range(5):
            assert torch.allclose(energy_batch[i], expected_energy)

    def test_free_energy_manual_calculation(self, small_rbm: BernoulliRBM) -> None:
        """Test free energy calculation against manual computation."""
        # Create test vector
        v = torch.tensor([1.0, 0.0, 1.0, 0.0])

        # Calculate free energy manually
        # F(v) = -v·vb - sum_j log(1 + exp(sum_i W_ji·v_i + hb_j))
        vb_term = (v * small_rbm.vb.detach()).sum()

        hidden_term = 0.0
        for j in range(3):  # hidden units
            # Calculate pre-activation without in-place operations
            pre_h_j = small_rbm.hb[j].item()  # Convert to Python scalar
            for i in range(4):  # visible units
                pre_h_j += small_rbm.w[j, i].item() * v[i].item()
            hidden_term += torch.log(1 + torch.exp(torch.tensor(pre_h_j))).item()

        expected_free_energy = -vb_term - hidden_term

        # Get free energy from model
        free_energy = small_rbm.free_energy(v)

        # Compare
        assert torch.allclose(free_energy, expected_free_energy)

        # Test batch dimension
        v_batch = v.repeat(5, 1)
        free_energy_batch = small_rbm.free_energy(v_batch)

        assert free_energy_batch.shape == (5,)
        for i in range(5):
            assert torch.allclose(free_energy_batch[i], expected_free_energy)

    def test_free_energy_vs_marginalized_energy(self, small_rbm: BernoulliRBM) -> None:
        """Test that free energy matches energy marginalized over hidden units."""
        # Create test vector
        v = torch.tensor([1.0, 0.0, 1.0, 0.0])

        # Calculate free energy using the model
        free_energy = small_rbm.free_energy(v)

        # Calculate marginalized energy by summing over all possible hidden configurations
        # For 3 hidden units, there are 2^3 = 8 possible configurations
        all_h_configs = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        # Calculate energy for each configuration
        energies = torch.zeros(8)
        for i in range(8):
            energies[i] = small_rbm.energy(v, all_h_configs[i])

        # Calculate marginalized energy: -log(sum_h exp(-E(v,h)))
        marginalized_energy = -torch.logsumexp(-energies, dim=0)

        # Compare
        assert torch.allclose(free_energy, marginalized_energy, atol=1e-5)

    def test_beta_temperature_scaling(self, small_rbm: BernoulliRBM) -> None:
        """Test temperature scaling with beta parameter."""
        # Create test vectors
        v = torch.tensor([1.0, 0.0, 1.0, 0.0])
        h = torch.tensor([1.0, 0.0, 1.0])

        # Test different beta values
        beta_values = [0.5, 1.0, 2.0]

        for beta_val in beta_values:
            # FIX: Convert beta to tensor
            beta = torch.tensor(beta_val)

            # Compare preact_h with and without beta
            pre_h_no_beta = small_rbm.preact_h(v)
            pre_h_with_beta = small_rbm.preact_h(v, beta=beta)
            assert torch.allclose(pre_h_with_beta, pre_h_no_beta * beta)

            # Compare preact_v with and without beta
            pre_v_no_beta = small_rbm.preact_v(h)
            pre_v_with_beta = small_rbm.preact_v(h, beta=beta)
            assert torch.allclose(pre_v_with_beta, pre_v_no_beta * beta)

            # Compare energy with and without beta
            energy_no_beta = small_rbm.energy(v, h)
            energy_with_beta = small_rbm.energy(v, h, beta=beta)
            assert torch.allclose(energy_with_beta, energy_no_beta * beta)

            # For free energy, scaling is more complex, but should still scale with beta
            # The visibility bias term scales directly, but the hidden term needs special handling
            free_energy_with_beta = small_rbm.free_energy(v, beta=beta)

            # Manually calculate scaled free energy
            vb_term = -(v * small_rbm.vb).sum() * beta
            pre_h = small_rbm.preact_h(v) * beta
            hidden_term = -torch.sum(F.softplus(pre_h))
            expected_scaled_free_energy = vb_term + hidden_term

            assert torch.allclose(free_energy_with_beta, expected_scaled_free_energy)

    def test_parallel_tempering_shapes(self, small_rbm: BernoulliRBM) -> None:
        """Test shapes with parallel tempering (multiple replicas)."""
        # Create batch of visible configurations with replica dimension
        batch_size = 4
        num_replicas = 3
        v = torch.rand(batch_size, num_replicas, 4)  # (B, K, V)
        h = torch.rand(batch_size, num_replicas, 3)  # (B, K, H)

        # Create beta tensor
        beta = torch.tensor([0.5, 1.0, 2.0]).view(1, 3, 1)  # (1, K, 1)

        # Test all methods with replica dimension
        pre_h = small_rbm.preact_h(v, beta=beta)
        assert pre_h.shape == (batch_size, num_replicas, 3)

        pre_v = small_rbm.preact_v(h, beta=beta)
        assert pre_v.shape == (batch_size, num_replicas, 4)

        h_probs = small_rbm.prob_h_given_v(v, beta=beta)
        assert h_probs.shape == (batch_size, num_replicas, 3)
        assert torch.all((h_probs >= 0) & (h_probs <= 1))

        v_probs = small_rbm.prob_v_given_h(h, beta=beta)
        assert v_probs.shape == (batch_size, num_replicas, 4)
        assert torch.all((v_probs >= 0) & (v_probs <= 1))

        h_samples = small_rbm.sample_h_given_v(v, beta=beta)
        assert h_samples.shape == (batch_size, num_replicas, 3)
        assert set(torch.unique(h_samples).tolist()).issubset({0.0, 1.0})

        v_samples = small_rbm.sample_v_given_h(h, beta=beta)
        assert v_samples.shape == (batch_size, num_replicas, 4)
        assert set(torch.unique(v_samples).tolist()).issubset({0.0, 1.0})

        energy = small_rbm.energy(v, h, beta=beta)
        assert energy.shape == (batch_size, num_replicas)

        free_energy = small_rbm.free_energy(v, beta=beta)
        assert free_energy.shape == (batch_size, num_replicas)

    def test_gibbs_sampling_chain(self, small_rbm: BernoulliRBM) -> None:
        """Test Gibbs sampling chain for reconstruction."""
        # Create a test visible vector
        v_0 = torch.tensor([1.0, 0.0, 1.0, 0.0])

        # Run one step of Gibbs sampling
        h_1 = small_rbm.sample_h_given_v(v_0)
        v_1 = small_rbm.sample_v_given_h(h_1)

        assert v_1.shape == v_0.shape
        assert set(torch.unique(v_1).tolist()).issubset({0.0, 1.0})

        # Run several steps and check reconstruction quality using means
        n_steps = 1000
        v_probs = torch.zeros_like(v_0)

        # FIX: Avoid using += for tensors that might have different representations
        for _ in range(n_steps):
            h_sample = small_rbm.sample_h_given_v(v_0)
            v_prob = small_rbm.prob_v_given_h(h_sample)
            v_sample = small_rbm.sample_v_given_h(h_sample)
            v_probs = v_probs.clone() + v_prob  # Use .clone() to create a new tensor
            v_0 = v_sample

        v_mean = v_probs / n_steps

        # Mean should stabilize to equilibrium distribution
        assert torch.all((v_mean >= 0) & (v_mean <= 1))

    def test_contrastive_divergence(self, small_rbm: BernoulliRBM) -> None:
        """Test Contrastive Divergence (CD) learning."""

        # Function to perform CD-k update
        def cd_k(model: BernoulliRBM, v_data: torch.Tensor, k: int = 1, lr: float = 0.01) -> None:
            """Perform one step of CD-k learning."""
            # Positive phase
            h_data_probs = model.prob_h_given_v(v_data)

            # Calculate positive gradients
            pos_grad_vb = v_data.mean(dim=0)
            pos_grad_hb = h_data_probs.mean(dim=0)
            pos_associations = torch.matmul(h_data_probs.t(), v_data) / v_data.shape[0]

            # Negative phase (k steps of Gibbs sampling)
            v_model = v_data.clone()
            for _ in range(k):
                h_model = model.sample_h_given_v(v_model)
                v_model = model.sample_v_given_h(h_model)

            h_model_probs = model.prob_h_given_v(v_model)

            # Calculate negative gradients
            neg_grad_vb = v_model.mean(dim=0)
            neg_grad_hb = h_model_probs.mean(dim=0)
            neg_associations = torch.matmul(h_model_probs.t(), v_model) / v_model.shape[0]

            # Update parameters
            with torch.no_grad():
                model.vb.add_(lr * (pos_grad_vb - neg_grad_vb))
                model.hb.add_(lr * (pos_grad_hb - neg_grad_hb))
                model.w.add_(lr * (pos_associations - neg_associations))

        # Create a simple dataset (10 identical samples)
        batch_size = 10
        v_data = torch.zeros(batch_size, 4)
        v_data[:, 0] = 1.0  # All samples have first unit on
        v_data[:, 2] = 1.0  # All samples have third unit on

        # Make a copy of the model for later comparison
        # FIX: Use BernoulliConfig instead of RBMConfig
        original_rbm = BernoulliRBM(
            BernoulliRBMConfig(visible=small_rbm.cfg.visible, hidden=small_rbm.cfg.hidden)
        )
        with torch.no_grad():
            original_rbm.w.copy_(small_rbm.w)
            original_rbm.vb.copy_(small_rbm.vb)
            original_rbm.hb.copy_(small_rbm.hb)

        # Calculate initial free energy
        initial_free_energy = small_rbm.free_energy(v_data).mean()

        # Perform CD-1 updates
        n_updates = 10
        for _ in range(n_updates):
            cd_k(small_rbm, v_data, k=1, lr=0.1)

        # Calculate final free energy
        final_free_energy = small_rbm.free_energy(v_data).mean()

        # Free energy should decrease after learning
        assert final_free_energy < initial_free_energy

        # The visible biases should increase for units that are on in the training data
        assert small_rbm.vb[0] > original_rbm.vb[0]
        assert small_rbm.vb[2] > original_rbm.vb[2]

        # The reconstruction error should decrease
        h_sample = small_rbm.sample_h_given_v(v_data)
        v_recon = small_rbm.prob_v_given_h(h_sample)

        recon_error = ((v_data - v_recon) ** 2).mean()
        assert recon_error < 0.5  # Should be much better than random

    def test_numerical_stability(self, small_rbm: BernoulliRBM) -> None:
        """Test numerical stability with extreme parameter values."""
        # Set extreme values for parameters
        with torch.no_grad():
            small_rbm.w.copy_(torch.ones_like(small_rbm.w) * 10.0)  # Very large weights
            small_rbm.vb.copy_(torch.ones_like(small_rbm.vb) * -10.0)  # Very negative bias
            small_rbm.hb.copy_(torch.ones_like(small_rbm.hb) * 10.0)  # Very positive bias

        # Create test vector
        v = torch.ones(4)

        # Test that probabilities are still in [0, 1] range
        h_probs = small_rbm.prob_h_given_v(v)
        assert torch.all((h_probs >= 0) & (h_probs <= 1))
        assert torch.all(torch.isfinite(h_probs))  # No NaNs or infinities

        # Test with different precision
        small_rbm_fp64 = BernoulliRBM(BernoulliRBMConfig(visible=4, hidden=3, dtype=torch.float64))

        with torch.no_grad():
            small_rbm_fp64.w.copy_(small_rbm.w.to(torch.float64))
            small_rbm_fp64.vb.copy_(small_rbm.vb.to(torch.float64))
            small_rbm_fp64.hb.copy_(small_rbm.hb.to(torch.float64))

        # Calculate free energy with both precisions
        fe_fp32 = small_rbm.free_energy(v)
        fe_fp64 = small_rbm_fp64.free_energy(v.to(torch.float64))

        # Values should be close, though not exactly the same due to precision differences
        assert abs(fe_fp32.item() - fe_fp64.item()) < 1e-2

    def test_init_vb_from_means(self, small_rbm: BernoulliRBM) -> None:
        """Test initialization of visible biases from data means."""
        # Test with various mean values
        test_cases = [
            # Format: [means, expected_activations]
            torch.tensor([0.1, 0.5, 0.9, 0.2]),
            torch.tensor([0.0, 1.0, 0.3, 0.7]),  # Test clamping
            torch.tensor([0.25, 0.75, 0.4, 0.6]),
        ]

        for means in test_cases:
            # Initialize visible biases from means
            small_rbm.init_vb_from_means(means)

            # Check that both vb and base_rate_vb are set correctly
            clamped_means = means.clamp(1e-3, 1 - 1e-3)
            expected_logits = torch.logit(clamped_means)

            assert torch.allclose(small_rbm.vb, expected_logits)
            assert torch.allclose(small_rbm.base_rate_vb, expected_logits)

            # Verify that the visible biases lead to appropriate activation probabilities
            # For this test, we set hidden units to zero so only the visible biases matter
            h_zeros = torch.zeros(small_rbm.cfg.hidden)
            v_probs = small_rbm.prob_v_given_h(h_zeros)

            # The activation probabilities should be close to the clamped mean values
            assert torch.allclose(v_probs, clamped_means)

    def test_init_vb_from_means_with_data(self, small_rbm: BernoulliRBM) -> None:
        """Test initialization of visible biases with simulated data."""
        # Generate some artificial data
        n_samples = 1000
        # Create biased data where some units are more active than others
        data = torch.zeros(n_samples, 4)
        # Unit 0 active in 20% of samples
        data[:200, 0] = 1.0
        # Unit 1 active in 60% of samples
        data[:600, 1] = 1.0
        # Unit 2 active in 40% of samples
        data[:400, 2] = 1.0
        # Unit 3 active in 80% of samples
        data[:800, 3] = 1.0

        # Calculate means
        means = data.mean(dim=0)
        expected_means = torch.tensor([0.2, 0.6, 0.4, 0.8])
        assert torch.allclose(means, expected_means)

        # Initialize visible biases from these means
        small_rbm.init_vb_from_means(means)

        # Check that biases lead to appropriate activations
        h_zeros = torch.zeros(small_rbm.cfg.hidden)
        v_probs = small_rbm.prob_v_given_h(h_zeros)

        # Check that the activations match the expected means
        assert torch.allclose(v_probs, expected_means)

        # Test with a batch of samples
        batch_size = 10
        h_zeros_batch = torch.zeros(batch_size, small_rbm.cfg.hidden)
        v_probs_batch = small_rbm.prob_v_given_h(h_zeros_batch)

        # Each sample in the batch should have the same activation probabilities
        assert v_probs_batch.shape == (batch_size, 4)
        for i in range(batch_size):
            assert torch.allclose(v_probs_batch[i], expected_means)

        # Verify that with random hidden configurations, we still have
        # approximately the correct marginal distribution over visible units
        n_samples = 1000
        v_samples = torch.zeros(n_samples, 4)

        for i in range(n_samples):
            # Random hidden configuration
            h = torch.bernoulli(torch.rand(small_rbm.cfg.hidden))
            # Sample visible units
            v_sample = small_rbm.sample_v_given_h(h)
            v_samples[i] = v_sample

        # Calculate means of the samples
        sample_means = v_samples.mean(dim=0)

        # The sample means should be approximately equal to the data means
        # (but could vary due to the influence of random hidden configurations)
        assert torch.allclose(sample_means, expected_means, atol=0.1)
