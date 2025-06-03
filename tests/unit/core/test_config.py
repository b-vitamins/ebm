"""Unit tests for configuration management."""

import json
from pathlib import Path

import pytest
import torch

from ebm.core.config import (
    GaussianRBMConfig,
    ModelConfig,
    OptimizerConfig,
    RBMConfig,
    TrainingConfig,
)


class TestBaseConfig:
    """Test the base configuration class."""

    def test_immutability(self):
        """Test that configs are immutable."""
        config = ModelConfig(device="cpu", dtype="float32")

        with pytest.raises(AttributeError):
            config.device = "cuda"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {"device": "cuda", "dtype": "float16", "seed": 123}
        config = ModelConfig.from_dict(data)

        assert config.device == "cuda"
        assert config.dtype == "float16"
        assert config.seed == 123

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ModelConfig(device="cpu", dtype="float32", seed=42)
        data = config.dict()

        assert isinstance(data, dict)
        assert data["device"] == "cpu"
        assert data["dtype"] == "float32"
        assert data["seed"] == 42

    def test_with_updates(self):
        """Test creating updated config."""
        config1 = ModelConfig(device="cpu", dtype="float32")
        config2 = config1.with_updates(device="cuda")

        # Original unchanged
        assert config1.device == "cpu"
        # New config updated
        assert config2.device == "cuda"
        assert config2.dtype == "float32"

    def test_save_load_json(self, tmp_path: Path):
        """Test saving and loading config as JSON."""
        config = RBMConfig(
            visible_units=784,
            hidden_units=500,
            weight_init="xavier_normal"
        )

        # Save
        save_path = tmp_path / "config.json"
        config.save(save_path)

        # Load
        loaded = RBMConfig.from_file(save_path)

        assert loaded.visible_units == config.visible_units
        assert loaded.hidden_units == config.hidden_units
        assert loaded.weight_init == config.weight_init

    def test_save_load_yaml(self, tmp_path: Path):
        """Test saving and loading config as YAML."""
        pytest.importorskip("yaml")

        config = TrainingConfig(
            epochs=100,
            batch_size=64,
            optimizer=OptimizerConfig(name="adam", lr=0.001)
        )

        # Save
        save_path = tmp_path / "config.yaml"
        config.save(save_path)

        # Load
        loaded = TrainingConfig.from_file(save_path)

        assert loaded.epochs == config.epochs
        assert loaded.batch_size == config.batch_size
        assert loaded.optimizer.name == config.optimizer.name
        assert loaded.optimizer.lr == config.optimizer.lr


class TestModelConfig:
    """Test model configuration."""

    def test_device_validation(self):
        """Test device string validation."""
        # Valid devices
        config = ModelConfig(device="cpu")
        assert config.device == "cpu"

        config = ModelConfig(device="cuda")
        assert config.device == "cuda" if torch.cuda.is_available() else "cpu"

        config = ModelConfig(device="auto")
        assert config.device in ["cpu", "cuda"]

        # Invalid device
        with pytest.raises(ValueError):
            ModelConfig(device="invalid")

    def test_dtype_validation(self):
        """Test data type validation."""
        # Valid dtypes
        valid_dtypes = ["float32", "float16", "bfloat16", "float64"]
        for dtype in valid_dtypes:
            config = ModelConfig(dtype=dtype)
            assert config.dtype == dtype

        # Invalid dtype
        with pytest.raises(ValueError):
            ModelConfig(dtype="int32")

    def test_torch_device_property(self):
        """Test torch device property."""
        config = ModelConfig(device="cpu")
        assert isinstance(config.torch_device, torch.device)
        assert config.torch_device.type == "cpu"

    def test_torch_dtype_property(self):
        """Test torch dtype property."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "float64": torch.float64,
        }

        for str_dtype, torch_dtype in dtype_map.items():
            config = ModelConfig(dtype=str_dtype)
            assert config.torch_dtype == torch_dtype


class TestOptimizerConfig:
    """Test optimizer configuration."""

    def test_optimizer_validation(self):
        """Test optimizer name validation."""
        valid_names = ["adam", "adamw", "sgd", "rmsprop", "lbfgs"]

        for name in valid_names:
            config = OptimizerConfig(name=name)
            assert config.name == name

        with pytest.raises(ValueError):
            OptimizerConfig(name="invalid_optimizer")

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Learning rate must be positive
        with pytest.raises(ValueError):
            OptimizerConfig(lr=-0.01)

        # Weight decay must be non-negative
        with pytest.raises(ValueError):
            OptimizerConfig(weight_decay=-0.001)

        # Valid parameters
        config = OptimizerConfig(lr=0.001, weight_decay=0.01)
        assert config.lr == 0.001
        assert config.weight_decay == 0.01

    def test_scheduler_config(self):
        """Test scheduler configuration."""
        config = OptimizerConfig(
            scheduler="cosine",
            scheduler_params={"eta_min": 0.0001}
        )

        assert config.scheduler == "cosine"
        assert config.scheduler_params["eta_min"] == 0.0001


class TestTrainingConfig:
    """Test training configuration."""

    def test_validation_constraints(self):
        """Test validation of training parameters."""
        # Epochs must be positive
        with pytest.raises(ValueError):
            TrainingConfig(epochs=0)

        # Batch size must be positive
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=-1)

        # Valid config
        config = TrainingConfig(epochs=10, batch_size=32)
        assert config.epochs == 10
        assert config.batch_size == 32

    def test_eval_batch_size_property(self):
        """Test eval batch size property."""
        # Default to training batch size
        config = TrainingConfig(batch_size=32)
        assert config.eval_batch_size_actual == 32

        # Use specified eval batch size
        config = TrainingConfig(batch_size=32, eval_batch_size=64)
        assert config.eval_batch_size_actual == 64

    def test_checkpoint_dir_path(self):
        """Test checkpoint directory handling."""
        config = TrainingConfig(checkpoint_dir="./checkpoints")
        assert isinstance(config.checkpoint_dir, Path)
        assert config.checkpoint_dir == Path("./checkpoints")


class TestRBMConfig:
    """Test RBM configuration."""

    def test_dimension_validation(self):
        """Test dimension validation."""
        # Valid dimensions
        config = RBMConfig(visible_units=784, hidden_units=500)
        assert config.visible_units == 784
        assert config.hidden_units == 500

        # Invalid dimensions
        with pytest.raises(ValueError):
            RBMConfig(visible_units=0, hidden_units=100)

        with pytest.raises(ValueError):
            RBMConfig(visible_units=100, hidden_units=-1)

    def test_init_method_validation(self):
        """Test initialization method validation."""
        valid_methods = [
            "xavier_uniform", "xavier_normal",
            "kaiming_uniform", "kaiming_normal",
            "normal", "uniform", "zeros", "ones"
        ]

        for method in valid_methods:
            config = RBMConfig(
                visible_units=100,
                hidden_units=50,
                weight_init=method
            )
            assert config.weight_init == method

        with pytest.raises(ValueError):
            RBMConfig(
                visible_units=100,
                hidden_units=50,
                weight_init="invalid_method"
            )

    def test_regularization_params(self):
        """Test regularization parameters."""
        config = RBMConfig(
            visible_units=100,
            hidden_units=50,
            l2_weight=0.001,
            l1_weight=0.0001
        )

        assert config.l2_weight == 0.001
        assert config.l1_weight == 0.0001

        # Must be non-negative
        with pytest.raises(ValueError):
            RBMConfig(
                visible_units=100,
                hidden_units=50,
                l2_weight=-0.001
            )


class TestGaussianRBMConfig:
    """Test Gaussian RBM configuration."""

    def test_gaussian_specific_params(self):
        """Test Gaussian-specific parameters."""
        config = GaussianRBMConfig(
            visible_units=100,
            hidden_units=50,
            sigma=2.0,
            learn_sigma=True
        )

        assert config.visible_type == "gaussian"
        assert config.sigma == 2.0
        assert config.learn_sigma is True

        # Sigma must be positive
        with pytest.raises(ValueError):
            GaussianRBMConfig(
                visible_units=100,
                hidden_units=50,
                sigma=-1.0
            )

    def test_inheritance(self):
        """Test that GaussianRBMConfig inherits from RBMConfig."""
        config = GaussianRBMConfig(
            visible_units=100,
            hidden_units=50,
            weight_init="xavier_normal"
        )

        # Should have all RBMConfig attributes
        assert hasattr(config, "visible_units")
        assert hasattr(config, "hidden_units")
        assert hasattr(config, "weight_init")
        assert hasattr(config, "use_bias")


class TestConfigSerialization:
    """Test configuration serialization edge cases."""

    def test_torch_type_serialization(self):
        """Test serialization of torch types."""
        config = ModelConfig(device="cuda:0", dtype="float32")
        data = config.dict()

        # Check that torch types are properly serialized
        assert isinstance(data["device"], str)
        assert isinstance(data["dtype"], str)

        # Test JSON serialization
        json_str = json.dumps(data)
        loaded = json.loads(json_str)

        assert loaded["device"] == "cuda:0"
        assert loaded["dtype"] == "float32"

    def test_nested_config_serialization(self):
        """Test serialization of nested configs."""
        config = TrainingConfig(
            epochs=10,
            optimizer=OptimizerConfig(
                name="adam",
                lr=0.001,
                scheduler="cosine"
            )
        )

        data = config.dict()
        assert isinstance(data["optimizer"], dict)
        assert data["optimizer"]["name"] == "adam"

        # Test round-trip
        loaded = TrainingConfig.from_dict(data)
        assert loaded.optimizer.name == "adam"
        assert loaded.optimizer.lr == 0.001

    def test_path_serialization(self, tmp_path: Path):
        """Test Path object serialization."""
        config = TrainingConfig(checkpoint_dir=tmp_path / "checkpoints")
        data = config.dict()

        # Path should be serialized as string
        assert isinstance(data["checkpoint_dir"], str)

        # Test round-trip
        loaded = TrainingConfig.from_dict(data)
        assert isinstance(loaded.checkpoint_dir, Path)
