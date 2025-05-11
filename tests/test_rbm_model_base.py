from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn as nn

from ebm.rbm.model.base import (
    RBMBase,
    RBMConfig,
    init_bias_tensor,
    init_weight_tensor,
)


class TestInitWeightTensor:
    """Tests for weight tensor initialization function."""

    def test_init_with_none(self) -> None:
        """Test default initialization (None)."""
        visible_size, hidden_size = 10, 5
        weights = init_weight_tensor(None, visible_size, hidden_size)

        # Check shape follows PyTorch convention (out_features, in_features)
        assert weights.shape == (hidden_size, visible_size)

        # Check statistical properties (approximately)
        # Increase threshold for mean to account for random variation
        assert torch.abs(weights.mean()) < 0.15
        expected_std = 1.0 / (visible_size * hidden_size) ** 0.25
        assert torch.abs(weights.std() - expected_std) < 0.1

    def test_init_with_float(self) -> None:
        """Test initialization with custom standard deviation."""
        visible_size, hidden_size = 8, 4
        std_dev = 0.01
        weights = init_weight_tensor(std_dev, visible_size, hidden_size)

        assert weights.shape == (hidden_size, visible_size)
        assert torch.abs(weights.mean()) < 0.1
        assert torch.abs(weights.std() - std_dev) < 0.01

    def test_init_with_tensor(self) -> None:
        """Test initialization with provided tensor."""
        visible_size, hidden_size = 6, 3
        input_tensor = torch.randn(hidden_size, visible_size)
        weights = init_weight_tensor(input_tensor, visible_size, hidden_size)

        assert weights.shape == (hidden_size, visible_size)
        assert torch.all(torch.eq(weights, input_tensor))

    def test_init_with_incorrect_tensor_shape(self) -> None:
        """Test error is raised when tensor has wrong shape."""
        visible_size, hidden_size = 6, 3
        # Wrong shape: transposed from what it should be
        input_tensor = torch.randn(visible_size, hidden_size)

        with pytest.raises(ValueError, match="Weight tensor shape mismatch"):
            init_weight_tensor(input_tensor, visible_size, hidden_size)

    def test_init_with_device_and_dtype(self) -> None:
        """Test device and dtype parameters."""
        visible_size, hidden_size = 4, 2
        device = torch.device("cpu")
        dtype = torch.float64

        weights = init_weight_tensor(None, visible_size, hidden_size, device=device, dtype=dtype)

        assert weights.device == device
        assert weights.dtype == dtype


class TestInitBiasTensor:
    """Tests for bias tensor initialization function."""

    def test_init_with_none(self) -> None:
        """Test default initialization (None)."""
        size = 5
        bias = init_bias_tensor(None, size)

        assert bias.shape == (size,)
        assert torch.all(torch.eq(bias, torch.zeros(size)))

    def test_init_with_float(self) -> None:
        """Test initialization with custom constant."""
        size = 4
        constant = -0.5
        bias = init_bias_tensor(constant, size)

        assert bias.shape == (size,)
        assert torch.all(torch.eq(bias, torch.full((size,), constant)))

    def test_init_with_tensor(self) -> None:
        """Test initialization with provided tensor."""
        size = 6
        input_tensor = torch.randn(size)
        bias = init_bias_tensor(input_tensor, size)

        assert bias.shape == (size,)
        assert torch.all(torch.eq(bias, input_tensor))

    def test_init_with_incorrect_tensor_shape(self) -> None:
        """Test error is raised when tensor has wrong shape."""
        size = 6
        input_tensor = torch.randn(size + 1)  # Wrong size

        with pytest.raises(ValueError, match="Bias tensor shape mismatch"):
            init_bias_tensor(input_tensor, size)

    def test_init_with_device_and_dtype(self) -> None:
        """Test device and dtype parameters."""
        size = 4
        device = torch.device("cpu")
        dtype = torch.float64

        bias = init_bias_tensor(None, size, device=device, dtype=dtype)

        assert bias.device == device
        assert bias.dtype == dtype


class TestRBMConfig:
    """Tests for RBMConfig dataclass."""

    def test_default_config(self) -> None:
        """Test creation with minimal required parameters."""
        visible, hidden = 28 * 28, 100
        config = RBMConfig(visible=visible, hidden=hidden)

        assert config.visible == visible
        assert config.hidden == hidden
        assert config.w_init is None
        assert config.vb_init is None
        assert config.hb_init is None
        assert callable(config.v_act)
        assert callable(config.h_act)
        assert config.dtype is None
        assert config.device is None

    def test_custom_config(self) -> None:
        """Test creation with custom parameters."""
        visible, hidden = 784, 500
        w_init = 0.01
        vb_init = -4.0
        hb_init = 0.0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        config = RBMConfig(
            visible=visible,
            hidden=hidden,
            w_init=w_init,
            vb_init=vb_init,
            hb_init=hb_init,
            device=device,
            dtype=dtype,
        )

        assert config.visible == visible
        assert config.hidden == hidden
        assert config.w_init == w_init
        assert config.vb_init == vb_init
        assert config.hb_init == hb_init
        assert config.device == device
        assert config.dtype == dtype

    def test_immutability(self) -> None:
        """Test that the dataclass is immutable."""
        config = RBMConfig(visible=784, hidden=500)

        with pytest.raises(FrozenInstanceError):
            # This will raise FrozenInstanceError because RBMConfig is frozen
            config.visible = 1000  # type: ignore


# Create a simple concrete RBM implementation for testing
class SimpleBinaryRBM(RBMBase):
    """A minimal implementation of binary RBM for testing."""

    def __init__(self, config: RBMConfig) -> None:
        super().__init__()
        self.cfg = config

        # Initialize parameters
        self.W = nn.Parameter(
            init_weight_tensor(
                config.w_init,
                config.visible,
                config.hidden,
                device=config.device,
                dtype=config.dtype,
            )
        )
        self.vb = nn.Parameter(
            init_bias_tensor(
                config.vb_init, config.visible, device=config.device, dtype=config.dtype
            )
        )
        self.hb = nn.Parameter(
            init_bias_tensor(
                config.hb_init, config.hidden, device=config.device, dtype=config.dtype
            )
        )

    def preact_h(self, v: torch.Tensor, beta: torch.Tensor | None = None) -> torch.Tensor:
        # W has shape (H, V)
        if beta is None:
            return torch.matmul(v, self.W.t()) + self.hb
        else:
            # For parallel tempering:
            # v has shape (batch_size, num_replicas, visible_size)
            # W has shape (hidden_size, visible_size)
            # We need to handle the replica dimension correctly

            # First compute regular matmul for each replica
            # Reshape v to combine batch and replica dims
            batch_size, num_replicas, visible_size = v.shape
            v_reshaped = v.reshape(-1, visible_size)  # (batch_size*num_replicas, visible_size)

            # Regular matmul
            preact = torch.matmul(v_reshaped, self.W.t())  # (batch_size*num_replicas, hidden_size)

            # Reshape back to separate batch and replica dims
            preact = preact.reshape(batch_size, num_replicas, self.cfg.hidden)

            # Apply temperature scaling
            scaled_preact = beta * preact
            scaled_hb = beta * self.hb.view(1, 1, -1)  # (1, 1, hidden_size)

            return scaled_preact + scaled_hb

    def preact_v(self, h: torch.Tensor, beta: torch.Tensor | None = None) -> torch.Tensor:
        # W has shape (H, V)
        if beta is None:
            return torch.matmul(h, self.W) + self.vb
        else:
            # For parallel tempering:
            # h has shape (batch_size, num_replicas, hidden_size)
            # W has shape (hidden_size, visible_size)
            # We need to handle the replica dimension correctly

            # First compute regular matmul for each replica
            # Reshape h to combine batch and replica dims
            batch_size, num_replicas, hidden_size = h.shape
            h_reshaped = h.reshape(-1, hidden_size)  # (batch_size*num_replicas, hidden_size)

            # Regular matmul
            preact = torch.matmul(h_reshaped, self.W)  # (batch_size*num_replicas, visible_size)

            # Reshape back to separate batch and replica dims
            preact = preact.reshape(batch_size, num_replicas, self.cfg.visible)

            # Apply temperature scaling
            scaled_preact = beta * preact
            scaled_vb = beta * self.vb.view(1, 1, -1)  # (1, 1, visible_size)

            return scaled_preact + scaled_vb

    def prob_h_given_v(self, v: torch.Tensor, beta: torch.Tensor | None = None) -> torch.Tensor:
        return self.cfg.h_act(self.preact_h(v, beta=beta))

    def prob_v_given_h(self, h: torch.Tensor, beta: torch.Tensor | None = None) -> torch.Tensor:
        return self.cfg.v_act(self.preact_v(h, beta=beta))

    def sample_h_given_v(self, v: torch.Tensor, beta: torch.Tensor | None = None) -> torch.Tensor:
        p_h = self.prob_h_given_v(v, beta=beta)
        return torch.bernoulli(p_h)

    def sample_v_given_h(self, h: torch.Tensor, beta: torch.Tensor | None = None) -> torch.Tensor:
        p_v = self.prob_v_given_h(h, beta=beta)
        return torch.bernoulli(p_v)

    def energy(
        self, v: torch.Tensor, h: torch.Tensor, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        if beta is None:
            # E(v,h) = -h^T·W·v - vb^T·v - hb^T·h
            term1 = -torch.sum(torch.matmul(h, self.W) * v, dim=-1)
            term2 = -torch.sum(v * self.vb, dim=-1)
            term3 = -torch.sum(h * self.hb, dim=-1)
            return term1 + term2 + term3
        else:
            # For parallel tempering with shapes:
            # v: (batch_size, num_replicas, visible_size)
            # h: (batch_size, num_replicas, hidden_size)
            # beta: (1, num_replicas, 1)

            batch_size, num_replicas = v.shape[0], v.shape[1]

            # Reshape to handle batch and replica dimensions
            v_flat = v.reshape(-1, self.cfg.visible)  # (batch_size*num_replicas, visible_size)
            h_flat = h.reshape(-1, self.cfg.hidden)  # (batch_size*num_replicas, hidden_size)

            # Calculate the base energy terms
            term1 = -torch.sum(torch.matmul(h_flat, self.W) * v_flat, dim=-1)
            term2 = -torch.sum(v_flat * self.vb, dim=-1)
            term3 = -torch.sum(h_flat * self.hb, dim=-1)

            # Combine terms and reshape
            energy = (term1 + term2 + term3).reshape(batch_size, num_replicas)

            # Apply temperature scaling
            return beta.squeeze(-1) * energy

    def free_energy(self, v: torch.Tensor, beta: torch.Tensor | None = None) -> torch.Tensor:
        if beta is None:
            # F(v) = -vb^T·v - sum_i log(1 + exp(W_i·v + hb_i))
            vb_term = -torch.sum(v * self.vb, dim=-1)
            wx_b = torch.matmul(v, self.W.t()) + self.hb
            hidden_term = -torch.sum(torch.log(1 + torch.exp(wx_b)), dim=-1)
            return vb_term + hidden_term
        else:
            # For parallel tempering with shapes:
            # v: (batch_size, num_replicas, visible_size)
            # beta: (1, num_replicas, 1)

            batch_size, num_replicas = v.shape[0], v.shape[1]

            # Reshape to handle batch and replica dimensions
            v_flat = v.reshape(-1, self.cfg.visible)  # (batch_size*num_replicas, visible_size)

            # Calculate pre-activation for hidden units
            wx_b = (
                torch.matmul(v_flat, self.W.t()) + self.hb
            )  # (batch_size*num_replicas, hidden_size)

            # Calculate visible bias term
            vb_term = -torch.sum(v_flat * self.vb, dim=-1)  # (batch_size*num_replicas,)

            # Calculate hidden term
            hidden_term = -torch.sum(
                torch.log(1 + torch.exp(wx_b)), dim=-1
            )  # (batch_size*num_replicas,)

            # Combine and reshape
            free_energy = (vb_term + hidden_term).reshape(batch_size, num_replicas)

            # Apply temperature scaling
            return beta.squeeze(-1) * free_energy


class TestSimpleBinaryRBM:
    """Tests for simple binary RBM implementation."""

    @pytest.fixture
    def rbm(self) -> SimpleBinaryRBM:
        """Create a simple binary RBM for testing."""
        config = RBMConfig(visible=4, hidden=3, w_init=0.1)
        return SimpleBinaryRBM(config)

    def test_shape_convention(self, rbm: SimpleBinaryRBM) -> None:
        """Test weight matrix follows PyTorch convention."""
        assert rbm.W.shape == (rbm.cfg.hidden, rbm.cfg.visible)

    def test_preact_h(self, rbm: SimpleBinaryRBM) -> None:
        """Test hidden pre-activation calculation."""
        batch_size = 2
        v = torch.rand(batch_size, rbm.cfg.visible)

        result = rbm.preact_h(v)

        assert result.shape == (batch_size, rbm.cfg.hidden)

        # Manual calculation to verify
        expected = torch.matmul(v, rbm.W.t()) + rbm.hb
        assert torch.allclose(result, expected)

    def test_preact_v(self, rbm: SimpleBinaryRBM) -> None:
        """Test visible pre-activation calculation."""
        batch_size = 2
        h = torch.rand(batch_size, rbm.cfg.hidden)

        result = rbm.preact_v(h)

        assert result.shape == (batch_size, rbm.cfg.visible)

        # Manual calculation to verify
        expected = torch.matmul(h, rbm.W) + rbm.vb
        assert torch.allclose(result, expected)

    def test_prob_h_given_v(self, rbm: SimpleBinaryRBM) -> None:
        """Test hidden unit probabilities calculation."""
        batch_size = 2
        v = torch.rand(batch_size, rbm.cfg.visible)

        result = rbm.prob_h_given_v(v)

        assert result.shape == (batch_size, rbm.cfg.hidden)
        assert torch.all((result >= 0) & (result <= 1))

    def test_prob_v_given_h(self, rbm: SimpleBinaryRBM) -> None:
        """Test visible unit probabilities calculation."""
        batch_size = 2
        h = torch.rand(batch_size, rbm.cfg.hidden)

        result = rbm.prob_v_given_h(h)

        assert result.shape == (batch_size, rbm.cfg.visible)
        assert torch.all((result >= 0) & (result <= 1))

    def test_sampling(self, rbm: SimpleBinaryRBM) -> None:
        """Test sampling methods."""
        batch_size = 2
        v = torch.rand(batch_size, rbm.cfg.visible)

        # Test h sampling
        h_sample = rbm.sample_h_given_v(v)
        assert h_sample.shape == (batch_size, rbm.cfg.hidden)
        assert set(torch.unique(h_sample).tolist()).issubset({0.0, 1.0})

        # Test v sampling
        v_sample = rbm.sample_v_given_h(h_sample)
        assert v_sample.shape == (batch_size, rbm.cfg.visible)
        assert set(torch.unique(v_sample).tolist()).issubset({0.0, 1.0})

    def test_energy(self, rbm: SimpleBinaryRBM) -> None:
        """Test energy calculation."""
        batch_size = 2
        v = torch.rand(batch_size, rbm.cfg.visible)
        h = torch.rand(batch_size, rbm.cfg.hidden)

        energy = rbm.energy(v, h)

        assert energy.shape == (batch_size,)

        # Manual calculation to verify
        term1 = -torch.sum(torch.matmul(h, rbm.W) * v, dim=1)
        term2 = -torch.sum(v * rbm.vb, dim=1)
        term3 = -torch.sum(h * rbm.hb, dim=1)
        expected = term1 + term2 + term3
        assert torch.allclose(energy, expected)

    def test_free_energy(self, rbm: SimpleBinaryRBM) -> None:
        """Test free energy calculation."""
        batch_size = 2
        v = torch.rand(batch_size, rbm.cfg.visible)

        free_energy = rbm.free_energy(v)

        assert free_energy.shape == (batch_size,)

    def test_parallel_tempering(self, rbm: SimpleBinaryRBM) -> None:
        """Test parallel tempering with beta parameter."""
        batch_size = 2
        num_replicas = 3
        v = torch.rand(batch_size, num_replicas, rbm.cfg.visible)
        h = torch.rand(batch_size, num_replicas, rbm.cfg.hidden)
        beta = torch.tensor([0.5, 1.0, 2.0]).view(1, 3, 1)

        # Test preactivations
        preact_h = rbm.preact_h(v, beta=beta)
        assert preact_h.shape == (batch_size, num_replicas, rbm.cfg.hidden)

        preact_v = rbm.preact_v(h, beta=beta)
        assert preact_v.shape == (batch_size, num_replicas, rbm.cfg.visible)

        # Test probabilities
        prob_h = rbm.prob_h_given_v(v, beta=beta)
        assert prob_h.shape == (batch_size, num_replicas, rbm.cfg.hidden)

        prob_v = rbm.prob_v_given_h(h, beta=beta)
        assert prob_v.shape == (batch_size, num_replicas, rbm.cfg.visible)

        # Test energy
        energy = rbm.energy(v, h, beta=beta)
        assert energy.shape == (batch_size, num_replicas)

        # Test free energy
        free_energy = rbm.free_energy(v, beta=beta)
        assert free_energy.shape == (batch_size, num_replicas)
