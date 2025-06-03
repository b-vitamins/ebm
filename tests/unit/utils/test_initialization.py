"""Unit tests for parameter initialization utilities."""

import math

import pytest
import torch
import torch.nn as nn

from ebm.utils.initialization import (
    Initializer,
    InitMethod,
    calculate_gain,
    get_fan_in_and_fan_out,
    init_from_data_statistics,
    initialize_module,
    kaiming_init,
    normal_init,
    uniform_init,
    xavier_init,
)


class TestInitMethod:
    """Test InitMethod enum."""

    def test_init_method_values(self):
        """Test that init methods have correct string values."""
        assert InitMethod.ZEROS == "zeros"
        assert InitMethod.ONES == "ones"
        assert InitMethod.CONSTANT == "constant"
        assert InitMethod.NORMAL == "normal"
        assert InitMethod.UNIFORM == "uniform"
        assert InitMethod.XAVIER_UNIFORM == "xavier_uniform"
        assert InitMethod.XAVIER_NORMAL == "xavier_normal"
        assert InitMethod.KAIMING_UNIFORM == "kaiming_uniform"
        assert InitMethod.KAIMING_NORMAL == "kaiming_normal"
        assert InitMethod.HE_UNIFORM == "he_uniform"
        assert InitMethod.HE_NORMAL == "he_normal"
        assert InitMethod.ORTHOGONAL == "orthogonal"
        assert InitMethod.SPARSE == "sparse"
        assert InitMethod.EYE == "eye"
        assert InitMethod.DIRAC == "dirac"


class TestInitializer:
    """Test Initializer class."""

    def test_basic_initialization(self):
        """Test basic initializer creation."""
        # String method
        init = Initializer("zeros")
        tensor = torch.randn(5, 5)
        init(tensor)
        assert torch.all(tensor == 0)

        # Callable method
        def custom_init(t):
            t.fill_(42.0)

        init = Initializer(custom_init)
        tensor = torch.randn(3, 3)
        init(tensor)
        assert torch.all(tensor == 42.0)

        # Tensor method (copy)
        source = torch.randn(2, 3)
        init = Initializer(source)
        tensor = torch.zeros(2, 3)
        init(tensor)
        assert torch.equal(tensor, source)

        # Constant method
        init = Initializer(3.14)
        tensor = torch.zeros(4, 4)
        init(tensor)
        assert torch.all(tensor == 3.14)

    def test_zeros_ones_constant(self):
        """Test basic initialization methods."""
        # Zeros
        init = Initializer("zeros")
        tensor = torch.randn(5, 5)
        init(tensor)
        assert torch.all(tensor == 0)

        # Ones
        init = Initializer("ones")
        tensor = torch.randn(3, 4)
        init(tensor)
        assert torch.all(tensor == 1)

        # Constant
        init = Initializer("constant", val=2.5)
        tensor = torch.randn(2, 2)
        init(tensor)
        assert torch.all(tensor == 2.5)

    def test_normal_uniform(self):
        """Test normal and uniform initialization."""
        # Normal
        init = Initializer("normal", mean=0.0, std=0.1)
        tensor = torch.zeros(1000, 100)
        init(tensor)

        # Check statistics
        assert abs(tensor.mean().item()) < 0.01
        assert abs(tensor.std().item() - 0.1) < 0.01

        # Uniform
        init = Initializer("uniform", a=-0.5, b=0.5)
        tensor = torch.zeros(1000, 100)
        init(tensor)

        # Check range and statistics
        assert tensor.min() >= -0.5
        assert tensor.max() <= 0.5
        assert abs(tensor.mean().item()) < 0.01

    def test_xavier_initialization(self):
        """Test Xavier/Glorot initialization."""
        # Xavier uniform
        init = Initializer("xavier_uniform", gain=1.0)
        tensor = torch.zeros(50, 100)
        init(tensor)

        # Check bound
        fan_in, fan_out = 100, 50
        expected_bound = math.sqrt(6.0 / (fan_in + fan_out))
        assert tensor.min() >= -expected_bound * 1.01
        assert tensor.max() <= expected_bound * 1.01

        # Xavier normal
        init = Initializer("xavier_normal", gain=2.0)
        tensor = torch.zeros(100, 100)
        init(tensor)

        # Check std
        fan_in, fan_out = 100, 100
        expected_std = 2.0 * math.sqrt(2.0 / (fan_in + fan_out))
        assert abs(tensor.std().item() - expected_std) < 0.1

    def test_kaiming_initialization(self):
        """Test Kaiming/He initialization."""
        # Kaiming uniform
        init = Initializer("kaiming_uniform", a=0, mode='fan_in', nonlinearity='relu')
        tensor = torch.zeros(50, 100)
        init(tensor)

        # Check bound
        fan_in = 100
        gain = math.sqrt(2.0)  # ReLU gain
        expected_bound = gain * math.sqrt(3.0 / fan_in)
        assert tensor.min() >= -expected_bound * 1.01
        assert tensor.max() <= expected_bound * 1.01

        # Kaiming normal
        init = Initializer("kaiming_normal", a=0.1, mode='fan_out', nonlinearity='leaky_relu')
        tensor = torch.zeros(200, 100)
        init(tensor)

        # Check approximate std
        assert tensor.std().item() > 0

    def test_he_initialization_aliases(self):
        """Test He initialization aliases."""
        # He uniform (alias for kaiming_uniform)
        init_he = Initializer("he_uniform")
        init_kaiming = Initializer("kaiming_uniform")

        torch.manual_seed(42)
        tensor1 = torch.zeros(10, 10)
        init_he(tensor1)

        torch.manual_seed(42)
        tensor2 = torch.zeros(10, 10)
        init_kaiming(tensor2)

        assert torch.equal(tensor1, tensor2)

    def test_orthogonal_initialization(self):
        """Test orthogonal initialization."""
        init = Initializer("orthogonal", gain=1.0)

        # Square matrix
        tensor = torch.zeros(50, 50)
        init(tensor)

        # Check orthogonality
        eye = tensor @ tensor.T
        expected = torch.eye(50)
        assert torch.allclose(eye, expected, atol=1e-5)

        # Non-square matrix
        tensor = torch.zeros(30, 50)
        init(tensor)

        # Check semi-orthogonality
        eye = tensor @ tensor.T
        assert torch.allclose(eye, torch.eye(30), atol=1e-5)

    def test_sparse_initialization(self):
        """Test sparse initialization."""
        init = Initializer("sparse", sparsity=0.1, std=0.01)
        tensor = torch.zeros(100, 100)
        init(tensor)

        # Check sparsity
        num_nonzero = (tensor != 0).sum().item()
        expected_nonzero = 0.1 * 100  # sparsity * num_columns
        assert abs(num_nonzero / 100 - expected_nonzero) < 5  # Allow some variance

        # Check that non-zero values have correct std
        nonzero_values = tensor[tensor != 0]
        if len(nonzero_values) > 0:
            assert abs(nonzero_values.std().item() - 0.01) < 0.005

    def test_eye_initialization(self):
        """Test identity matrix initialization."""
        init = Initializer("eye")

        # Square matrix
        tensor = torch.zeros(5, 5)
        init(tensor)
        assert torch.equal(tensor, torch.eye(5))

        # Non-square matrix (tall)
        tensor = torch.zeros(6, 4)
        init(tensor)
        expected = torch.zeros(6, 4)
        expected[:4, :4] = torch.eye(4)
        assert torch.equal(tensor, expected)

        # Non-square matrix (wide)
        tensor = torch.zeros(3, 5)
        init(tensor)
        expected = torch.zeros(3, 5)
        expected[:3, :3] = torch.eye(3)
        assert torch.equal(tensor, expected)

        # Higher dimensional
        tensor = torch.zeros(2, 3, 4, 4)
        init(tensor)
        assert torch.equal(tensor[0, 0], torch.eye(4))
        assert torch.equal(tensor[1, 2], torch.eye(4))

    def test_dirac_initialization(self):
        """Test Dirac delta initialization."""
        # Only works for 3+ dimensional tensors
        init = Initializer("dirac", groups=1)

        # Conv2d-like tensor
        tensor = torch.zeros(8, 4, 3, 3)
        init(tensor)

        # Check that it's initialized properly
        # Should have spikes in the center
        assert tensor[:, :, 1, 1].sum() > 0

    def test_invalid_method(self):
        """Test error on invalid initialization method."""
        with pytest.raises(ValueError, match="Unknown initialization method"):
            Initializer("invalid_method")

        with pytest.raises(TypeError, match="Invalid initialization method type"):
            Initializer([1, 2, 3])  # Invalid type

    def test_shape_mismatch(self):
        """Test error on shape mismatch for tensor copy."""
        source = torch.randn(3, 4)
        init = Initializer(source)

        tensor = torch.zeros(5, 5)  # Different shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            init(tensor)

    def test_create_parameter(self):
        """Test parameter creation."""
        init = Initializer("xavier_normal")

        param = init.create_parameter(
            shape=(10, 20),
            dtype=torch.float32,
            device="cpu",
            requires_grad=True
        )

        assert isinstance(param, nn.Parameter)
        assert param.shape == (10, 20)
        assert param.dtype == torch.float32
        assert param.requires_grad

        # Check initialization was applied
        assert param.std().item() > 0

    def test_create_buffer(self):
        """Test buffer creation."""
        init = Initializer("ones")

        buffer = init.create_buffer(
            shape=(5, 5),
            dtype=torch.float64,
            device="cpu"
        )

        assert isinstance(buffer, torch.Tensor)
        assert not isinstance(buffer, nn.Parameter)
        assert buffer.shape == (5, 5)
        assert buffer.dtype == torch.float64
        assert torch.all(buffer == 1)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_fan_in_and_fan_out(self):
        """Test fan_in and fan_out calculation."""
        # 2D tensor (linear layer)
        tensor = torch.randn(20, 30)
        fan_in, fan_out = get_fan_in_and_fan_out(tensor)
        assert fan_in == 30
        assert fan_out == 20

        # 4D tensor (conv layer)
        tensor = torch.randn(64, 32, 3, 3)  # out_channels, in_channels, h, w
        fan_in, fan_out = get_fan_in_and_fan_out(tensor)
        assert fan_in == 32 * 3 * 3  # in_channels * kernel_size
        assert fan_out == 64 * 3 * 3  # out_channels * kernel_size

        # 1D tensor should raise error
        tensor = torch.randn(10)
        with pytest.raises(ValueError, match="fewer than 2 dimensions"):
            get_fan_in_and_fan_out(tensor)

    def test_calculate_gain(self):
        """Test gain calculation for different nonlinearities."""
        # Linear functions
        assert calculate_gain('linear') == 1
        assert calculate_gain('conv2d') == 1
        assert calculate_gain('sigmoid') == 1

        # Tanh
        assert calculate_gain('tanh') == pytest.approx(5.0 / 3)

        # ReLU
        assert calculate_gain('relu') == pytest.approx(math.sqrt(2.0))

        # Leaky ReLU
        assert calculate_gain('leaky_relu') == pytest.approx(math.sqrt(2.0 / 1.01))
        assert calculate_gain('leaky_relu', 0.2) == pytest.approx(math.sqrt(2.0 / 1.04))

        # SELU
        assert calculate_gain('selu') == pytest.approx(0.75)

        # Unknown nonlinearity
        with pytest.raises(ValueError, match="Unsupported nonlinearity"):
            calculate_gain('unknown')

    def test_initialize_module(self):
        """Test module initialization."""
        # Create a simple module
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(10, 20))
                self.weight2 = nn.Parameter(torch.randn(5, 10))
                self.bias1 = nn.Parameter(torch.randn(10))
                self.bias2 = nn.Parameter(torch.randn(5))

        module = TestModule()

        # Initialize with specific strategies
        initialize_module(
            module,
            weight_init="xavier_uniform",
            bias_init="zeros"
        )

        # Check biases are zero
        assert torch.all(module.bias1 == 0)
        assert torch.all(module.bias2 == 0)

        # Check weights are initialized (not zero)
        assert module.weight1.abs().mean() > 0
        assert module.weight2.abs().mean() > 0

    def test_convenience_functions(self):
        """Test convenience initialization functions."""
        # Uniform init
        init = uniform_init(low=-0.1, high=0.1)
        tensor = torch.zeros(100, 100)
        init(tensor)
        assert tensor.min() >= -0.1
        assert tensor.max() <= 0.1

        # Normal init
        init = normal_init(mean=0.0, std=0.02)
        tensor = torch.zeros(1000, 100)
        init(tensor)
        assert abs(tensor.mean().item()) < 0.01
        assert abs(tensor.std().item() - 0.02) < 0.01

        # Xavier init
        init = xavier_init(uniform=True, gain=2.0)
        assert isinstance(init, Initializer)
        assert init.kwargs["gain"] == 2.0

        # Kaiming init
        init = kaiming_init(uniform=False, a=0.1, mode='fan_out')
        assert isinstance(init, Initializer)
        assert init.kwargs["a"] == 0.1
        assert init.kwargs["mode"] == 'fan_out'

    def test_init_from_data_statistics(self):
        """Test initialization from data statistics."""
        # Mock data statistics
        data_mean = torch.tensor([0.2, 0.8, 0.5])
        data_std = torch.tensor([0.1, 0.2, 0.15])

        init_fn = init_from_data_statistics(
            data_mean=data_mean,
            data_std=data_std,
            scale=1.0
        )

        # Test on matching size tensor (for bias)
        bias = torch.zeros(3)
        init_fn(bias)
        assert torch.equal(bias, data_mean)

        # Test on weight matrix
        torch.manual_seed(42)
        weight = torch.zeros(5, 3)
        init_fn(weight)

        # Should be initialized based on data variance
        assert weight.std() > 0

        # Test with scale
        init_fn_scaled = init_from_data_statistics(
            data_mean=data_mean,
            scale=0.5
        )

        bias_scaled = torch.zeros(3)
        init_fn_scaled(bias_scaled)
        assert torch.allclose(bias_scaled, data_mean * 0.5)

        # Test fallback when dimensions don't match
        init_fn_fallback = init_from_data_statistics()
        tensor = torch.zeros(10)
        init_fn_fallback(tensor)
        assert tensor.std() > 0  # Should use default initialization


class TestEdgeCases:
    """Test edge cases for initialization."""

    def test_empty_tensor(self):
        """Test initialization of empty tensors."""
        init = Initializer("normal")
        tensor = torch.empty(0, 5)

        # Should not crash
        init(tensor)
        assert tensor.shape == (0, 5)

    def test_scalar_tensor(self):
        """Test initialization of scalar tensors."""
        init = Initializer("constant", val=3.14)
        tensor = torch.tensor(0.0)

        init(tensor)
        assert tensor.item() == 3.14

    def test_very_large_tensor(self):
        """Test initialization of very large tensors."""
        # Should handle large tensors efficiently
        init = Initializer("normal", std=0.01)
        tensor = torch.zeros(1000, 1000)

        init(tensor)

        # Check basic statistics
        assert abs(tensor.mean().item()) < 0.001
        assert abs(tensor.std().item() - 0.01) < 0.001

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_initialization(self):
        """Test initialization on CUDA tensors."""
        init = Initializer("xavier_normal")

        tensor = torch.zeros(10, 10, device="cuda")
        init(tensor)

        assert tensor.device.type == "cuda"
        assert tensor.std() > 0

        # Create parameter on CUDA
        param = init.create_parameter((5, 5), device="cuda")
        assert param.device.type == "cuda"
