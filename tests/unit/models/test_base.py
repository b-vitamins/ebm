"""Unit tests for base model classes."""

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from ebm.core.config import ModelConfig
from ebm.models.base import (
    AISInterpolator,
    EnergyBasedModel,
    LatentVariableModel,
)


class ConcreteEnergyModel(EnergyBasedModel):
    """Concrete implementation for testing."""

    def _build_model(self) -> None:
        self.weight = nn.Parameter(torch.randn(10, 10))
        self.bias = nn.Parameter(torch.zeros(10))

    def energy(self, x: Tensor, *, beta=None, return_parts=False):
        """Compute the model energy for a batch."""
        x.shape[0]
        # Simple quadratic energy
        energy = 0.5 * (x @ self.weight @ x.T).diagonal() + (x @ self.bias)

        if beta is not None:
            energy = energy * beta

        if return_parts:
            return {
                "quadratic": 0.5 * (x @ self.weight @ x.T).diagonal(),
                "linear": x @ self.bias,
                "total": energy,
            }
        return energy

    def free_energy(self, v: Tensor, *, beta=None):
        """Return free energy, delegating to ``energy``."""
        # For testing, just use energy
        return self.energy(v, beta=beta)

    @classmethod
    def get_config_class(cls):
        """Return the config class used for this model."""
        return ModelConfig


class ConcreteLatentModel(LatentVariableModel):
    """Concrete latent variable model for testing."""

    def _build_model(self) -> None:
        self.W = nn.Parameter(torch.randn(20, 10) * 0.01)
        self.vbias = nn.Parameter(torch.zeros(10))
        self.hbias = nn.Parameter(torch.zeros(20))

    def energy(self, x: Tensor, *, beta=None, return_parts=False):
        """Compute energy for visible and hidden units."""
        # Split into visible and hidden
        v = x[..., :10]
        h = x[..., 10:]

        interaction = -torch.einsum("...i,...j,ij->...", v, h, self.W.T)
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

    def free_energy(self, v: Tensor, *, beta=None):
        """Compute free energy of visible units."""
        pre_h = v @ self.W.T + self.hbias
        if beta is not None:
            pre_h = pre_h * beta
            v_term = -(v @ self.vbias) * beta
        else:
            v_term = -(v @ self.vbias)

        h_term = -torch.logsumexp(
            torch.stack([torch.zeros_like(pre_h), pre_h], dim=-1), dim=-1
        ).sum(-1)
        return v_term + h_term

    def sample_hidden(self, visible: Tensor, *, beta=None, return_prob=False):
        """Sample hidden units from visibles."""
        pre_h = visible @ self.W.T + self.hbias
        if beta is not None:
            pre_h = pre_h * beta
        prob_h = torch.sigmoid(pre_h)
        sample_h = torch.bernoulli(prob_h)

        if return_prob:
            return sample_h, prob_h
        return sample_h

    def sample_visible(self, hidden: Tensor, *, beta=None, return_prob=False):
        """Sample visible units from hiddens."""
        pre_v = hidden @ self.W + self.vbias
        if beta is not None:
            pre_v = pre_v * beta
        prob_v = torch.sigmoid(pre_v)
        sample_v = torch.bernoulli(prob_v)

        if return_prob:
            return sample_v, prob_v
        return sample_v

    @classmethod
    def get_config_class(cls):
        """Return the configuration class for the model."""
        return ModelConfig


class TestEnergyBasedModel:
    """Test EnergyBasedModel base class."""

    def test_initialization(self) -> None:
        """Test model initialization."""
        config = ModelConfig(device="cpu", dtype="float32", seed=42)
        model = ConcreteEnergyModel(config)

        assert model.config == config
        assert model.device == torch.device("cpu")
        assert model.dtype == torch.float32
        assert hasattr(model, "weight")
        assert hasattr(model, "bias")

    def test_device_management(self) -> None:
        """Test device management."""
        config = ModelConfig(device="cpu")
        model = ConcreteEnergyModel(config)

        # Check device
        assert model.device == torch.device("cpu")
        assert model._device_manager.device == torch.device("cpu")

        # Check parameters are on correct device
        for param in model.parameters():
            assert param.device == torch.device("cpu")

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_initialization(self) -> None:
        """Test CUDA initialization."""
        config = ModelConfig(device="cuda", dtype="float16")
        model = ConcreteEnergyModel(config)

        assert model.device.type == "cuda"
        assert model.dtype == torch.float16

        # Parameters should be on CUDA with correct dtype
        for param in model.parameters():
            assert param.device.type == "cuda"
            assert param.dtype == torch.float16

    def test_prepare_input(self) -> None:
        """Test input preparation."""
        config = ModelConfig(device="cpu", dtype="float32")
        model = ConcreteEnergyModel(config)

        # Test with numpy array
        import numpy as np

        np_array = np.random.randn(5, 10)
        tensor = model.prepare_input(np_array)
        assert isinstance(tensor, Tensor)
        assert tensor.device == model.device
        assert tensor.dtype == model.dtype

        # Test with wrong device tensor
        if torch.cuda.is_available():
            cuda_tensor = torch.randn(5, 10, device="cuda")
            cpu_tensor = model.prepare_input(cuda_tensor)
            assert cpu_tensor.device == torch.device("cpu")

    def test_energy_computation(self) -> None:
        """Test energy computation."""
        config = ModelConfig(device="cpu", dtype="float32", seed=42)
        model = ConcreteEnergyModel(config)

        x = torch.randn(10, 10)

        # Basic energy
        energy = model.energy(x)
        assert energy.shape == (10,)
        assert energy.dtype == torch.float32

        # Energy with beta
        beta = torch.tensor(0.5)
        energy_beta = model.energy(x, beta=beta)
        assert torch.allclose(energy_beta, energy * 0.5)

        # Energy with parts
        parts = model.energy(x, return_parts=True)
        assert isinstance(parts, dict)
        assert "quadratic" in parts
        assert "linear" in parts
        assert "total" in parts
        assert torch.allclose(parts["total"], energy)

    def test_log_probability(self) -> None:
        """Test log probability computation."""
        config = ModelConfig(device="cpu", dtype="float32")
        model = ConcreteEnergyModel(config)

        x = torch.randn(5, 10)

        # Without normalization
        log_prob_unnorm = model.log_probability(x)
        energy = model.energy(x)
        assert torch.allclose(log_prob_unnorm, -energy)

        # With normalization
        log_z = 10.0
        log_prob = model.log_probability(x, log_z=log_z)
        assert torch.allclose(log_prob, -energy - log_z)

    def test_save_load_checkpoint(self, tmp_path: Path) -> None:
        """Test checkpoint save/load."""
        config = ModelConfig(device="cpu", dtype="float32", seed=42)
        model = ConcreteEnergyModel(config)

        # Modify parameters
        with torch.no_grad():
            model.weight.data = torch.randn_like(model.weight)
            model.bias.data = torch.ones_like(model.bias)

        # Save checkpoint
        checkpoint_path = tmp_path / "model.pt"
        metadata = {"epoch": 10, "loss": 0.5}
        model.save_checkpoint(checkpoint_path, metadata=metadata)

        assert checkpoint_path.exists()

        # Load checkpoint
        new_model = ConcreteEnergyModel(config)
        loaded_metadata = new_model.load_checkpoint(checkpoint_path)

        # Check parameters match
        assert torch.allclose(new_model.weight, model.weight)
        assert torch.allclose(new_model.bias, model.bias)
        assert loaded_metadata == metadata

    def test_from_checkpoint(self, tmp_path: Path) -> None:
        """Test loading model from checkpoint."""
        config = ModelConfig(device="cpu", dtype="float32", seed=42)
        model = ConcreteEnergyModel(config)

        # Save
        checkpoint_path = tmp_path / "model.pt"
        model.save_checkpoint(checkpoint_path)

        # Load from checkpoint
        loaded_model = ConcreteEnergyModel.from_checkpoint(checkpoint_path)

        assert isinstance(loaded_model, ConcreteEnergyModel)
        assert torch.allclose(loaded_model.weight, model.weight)
        assert loaded_model.config.device == config.device

    def test_reset_parameters(self) -> None:
        """Test parameter reset."""
        config = ModelConfig(device="cpu", dtype="float32")
        model = ConcreteEnergyModel(config)

        # Store original parameters
        model.weight.data.clone()

        # Modify parameters
        with torch.no_grad():
            model.weight.data = torch.ones_like(model.weight)

        # Reset (note: base implementation doesn't change values)
        model.reset_parameters()

        # For modules with reset_parameters method, they would be reset
        # Our simple model doesn't have this, so parameters stay the same
        assert torch.allclose(model.weight, torch.ones_like(model.weight))

    def test_parameter_summary(self) -> None:
        """Test parameter summary."""
        config = ModelConfig(device="cpu", dtype="float32")
        model = ConcreteEnergyModel(config)

        summary = model.parameter_summary()

        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "non_trainable_parameters" in summary
        assert "model_size_mb" in summary

        # Check counts
        total = sum(p.numel() for p in model.parameters())
        assert summary["total_parameters"] == total
        assert summary["trainable_parameters"] == total  # All trainable
        assert summary["non_trainable_parameters"] == 0

    def test_repr(self) -> None:
        """Test string representation."""
        config = ModelConfig(device="cpu", dtype="float32", seed=42)
        model = ConcreteEnergyModel(config)

        repr_str = repr(model)
        assert "ConcreteEnergyModel" in repr_str
        assert "device" in repr_str
        assert "dtype" in repr_str


class TestLatentVariableModel:
    """Test LatentVariableModel base class."""

    def test_initialization(self) -> None:
        """Test latent model initialization."""
        config = ModelConfig(device="cpu", dtype="float32")
        model = ConcreteLatentModel(config)

        assert hasattr(model, "W")
        assert hasattr(model, "vbias")
        assert hasattr(model, "hbias")

    def test_sampling_methods(self) -> None:
        """Test sampling visible and hidden units."""
        config = ModelConfig(device="cpu", dtype="float32", seed=42)
        model = ConcreteLatentModel(config)

        # Sample hidden given visible
        v = torch.rand(5, 10)
        h = model.sample_hidden(v)
        assert h.shape == (5, 20)
        assert torch.all((h == 0) | (h == 1))

        # Sample with probabilities
        h_sample, h_prob = model.sample_hidden(v, return_prob=True)
        assert h_sample.shape == (5, 20)
        assert h_prob.shape == (5, 20)
        assert torch.all((h_prob >= 0) & (h_prob <= 1))

        # Sample visible given hidden
        v_new = model.sample_visible(h)
        assert v_new.shape == (5, 10)
        assert torch.all((v_new == 0) | (v_new == 1))

    def test_sampling_with_beta(self) -> None:
        """Test sampling with temperature parameter."""
        config = ModelConfig(device="cpu", dtype="float32", seed=42)
        model = ConcreteLatentModel(config)

        v = torch.rand(5, 10)

        # Different temperatures should give different results
        model.sample_hidden(v, beta=torch.tensor(0.5))
        model.sample_hidden(v, beta=torch.tensor(2.0))

        # Higher beta (lower temperature) should give more deterministic results
        _, prob1 = model.sample_hidden(
            v, beta=torch.tensor(0.1), return_prob=True
        )
        _, prob2 = model.sample_hidden(
            v, beta=torch.tensor(10.0), return_prob=True
        )

        # Check that high beta makes probabilities more extreme
        entropy1 = -(
            prob1 * torch.log(prob1 + 1e-8)
            + (1 - prob1) * torch.log(1 - prob1 + 1e-8)
        ).mean()
        entropy2 = -(
            prob2 * torch.log(prob2 + 1e-8)
            + (1 - prob2) * torch.log(1 - prob2 + 1e-8)
        ).mean()
        assert entropy2 < entropy1  # Lower entropy for higher beta

    def test_joint_energy(self) -> None:
        """Test joint energy computation."""
        config = ModelConfig(device="cpu", dtype="float32")
        model = ConcreteLatentModel(config)

        v = torch.rand(5, 10)
        h = torch.rand(5, 20)

        # Default joint energy
        energy = model.joint_energy(v, h)
        assert energy.shape == (5,)

        # Joint energy from concatenated state
        x = torch.cat([v, h], dim=-1)
        energy_concat = model.energy(x)
        assert torch.allclose(energy, energy_concat)

    def test_gibbs_step(self) -> None:
        """Test Gibbs sampling step."""
        config = ModelConfig(device="cpu", dtype="float32", seed=42)
        model = ConcreteLatentModel(config)

        v_init = torch.rand(10, 10)

        # One Gibbs step starting from visible
        v_new, h_new = model.gibbs_step(v_init)
        assert v_new.shape == v_init.shape
        assert h_new.shape == (10, 20)

        # Starting from hidden
        v_new2, h_new2 = model.gibbs_step(v_init, start_from="hidden")
        assert v_new2.shape == v_init.shape
        assert h_new2.shape == (10, 20)

        # Results should be different
        assert not torch.allclose(h_new, h_new2)

    def test_reconstruct(self) -> None:
        """Test reconstruction."""
        config = ModelConfig(device="cpu", dtype="float32", seed=42)
        model = ConcreteLatentModel(config)

        v = torch.rand(5, 10)

        # Single step reconstruction
        v_recon = model.reconstruct(v, num_steps=1)
        assert v_recon.shape == v.shape

        # Multi-step reconstruction
        v_recon_multi = model.reconstruct(v, num_steps=5)
        assert v_recon_multi.shape == v.shape

        # Should be different from original
        assert not torch.allclose(v, v_recon)


class TestAISInterpolator:
    """Test AIS interpolator mixin."""

    def test_ais_interpolator_basic(self) -> None:
        """Test basic AIS interpolator functionality."""
        config = ModelConfig(device="cpu", dtype="float32")
        base_model = ConcreteEnergyModel(config)

        interpolator = AISInterpolator(base_model)

        assert interpolator.base_model is base_model
        assert interpolator.ais_beta == 1.0

    def test_ais_beta_setter(self) -> None:
        """Test AIS beta parameter setting."""
        config = ModelConfig(device="cpu", dtype="float32")
        base_model = ConcreteEnergyModel(config)
        interpolator = AISInterpolator(base_model)

        # Valid beta
        interpolator.ais_beta = 0.5
        assert interpolator.ais_beta == 0.5

        # Invalid beta
        with pytest.raises(ValueError, match="AIS beta must be in"):
            interpolator.ais_beta = -0.1

        with pytest.raises(ValueError, match="AIS beta must be in"):
            interpolator.ais_beta = 1.5

    def test_interpolated_energy_default(self) -> None:
        """Test default interpolated energy computation."""

        # Need to create a concrete interpolator for testing
        class TestInterpolator(AISInterpolator):
            def base_log_partition(self) -> float:
                return 0.0

            def base_energy(self, x):
                # Simple base energy (uniform distribution)
                return torch.zeros(x.shape[0])

        config = ModelConfig(device="cpu", dtype="float32")
        base_model = ConcreteEnergyModel(config)
        interpolator = TestInterpolator(base_model)

        x = torch.randn(5, 10)

        # At beta=0, should get base energy
        interpolator.ais_beta = 0.0
        energy = interpolator.interpolated_energy(x)
        assert torch.allclose(energy, torch.zeros(5))

        # At beta=1, should get target energy
        interpolator.ais_beta = 1.0
        energy = interpolator.interpolated_energy(x)
        target_energy = base_model.energy(x)
        assert torch.allclose(energy, target_energy)

        # At intermediate beta
        interpolator.ais_beta = 0.5
        energy = interpolator.interpolated_energy(x)
        expected = 0.5 * torch.zeros(5) + 0.5 * target_energy
        assert torch.allclose(energy, expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_abstract_method_enforcement(self) -> None:
        """Test that abstract methods must be implemented."""
        config = ModelConfig(device="cpu", dtype="float32")

        # Try to create model without implementing abstract methods
        class IncompleteModel(EnergyBasedModel):
            def _build_model(self) -> None:
                pass

            def energy(self, x, *, beta=None, return_parts=False) -> None:
                """Unimplemented energy function."""
                pass

            # Missing: free_energy, get_config_class

        # Should still instantiate (Python doesn't enforce at runtime)
        # but calling missing methods should fail
        model = IncompleteModel(config)

        with pytest.raises(NotImplementedError):
            model.free_energy(torch.randn(5, 10))

    def test_device_dtype_consistency(self) -> None:
        """Test device and dtype consistency."""
        config = ModelConfig(device="cpu", dtype="float64")
        model = ConcreteEnergyModel(config)

        # All parameters should have consistent dtype
        for param in model.parameters():
            assert param.dtype == torch.float64

        # Input preparation should convert dtype
        x_float32 = torch.randn(5, 10, dtype=torch.float32)
        x_prepared = model.prepare_input(x_float32)
        assert x_prepared.dtype == torch.float64

    def test_empty_batch(self) -> None:
        """Test handling of empty batches."""
        config = ModelConfig(device="cpu", dtype="float32")
        model = ConcreteEnergyModel(config)

        # Empty batch
        x = torch.randn(0, 10)
        energy = model.energy(x)
        assert energy.shape == (0,)
        assert energy.numel() == 0

    def test_single_sample(self) -> None:
        """Test handling of single samples."""
        config = ModelConfig(device="cpu", dtype="float32")
        model = ConcreteEnergyModel(config)

        # Single sample
        x = torch.randn(1, 10)
        energy = model.energy(x)
        assert energy.shape == (1,)

        # 1D input should work after prepare_input
        x_1d = torch.randn(10)
        x_prepared = model.prepare_input(x_1d)
        assert x_prepared.shape == (10,)  # prepare_input doesn't add batch dim
