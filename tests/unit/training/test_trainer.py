"""Unit tests for the Trainer class."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ebm.core.config import ModelConfig, OptimizerConfig, TrainingConfig
from ebm.models.base import EnergyBasedModel
from ebm.sampling.base import GradientEstimator
from ebm.training.callbacks import Callback, CallbackList
from ebm.training.trainer import Trainer


class MockModel(EnergyBasedModel):
    """Mock model for testing."""

    def __init__(self, n_features: int = 10) -> None:
        self.n_features = n_features
        self.param1 = nn.Parameter(torch.randn(n_features, n_features))
        self.param2 = nn.Parameter(torch.randn(n_features))
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def _build_model(self) -> None:
        pass

    def energy(
        self,
        x: torch.Tensor,
        *,
        _beta: float | None = None,
        _return_parts: bool = False,
    ) -> torch.Tensor:
        """Compute a simple quadratic energy."""
        return torch.sum(x**2, dim=-1)

    def free_energy(
        self, v: torch.Tensor, *, beta: float | None = None
    ) -> torch.Tensor:
        """Return free energy equal to energy."""
        return self.energy(v, beta=beta)

    @property
    def device(self) -> torch.device:
        """Return the device used for parameters."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype used for parameters."""
        return self._dtype

    def parameters(self) -> list[nn.Parameter]:
        """Return model parameters."""
        return [self.param1, self.param2]

    def named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Return named parameters."""
        return [("param1", self.param1), ("param2", self.param2)]

    def state_dict(self) -> dict[str, nn.Parameter]:
        """Return state dictionary of model."""
        return {"param1": self.param1, "param2": self.param2}

    def load_state_dict(self, state_dict: dict[str, nn.Parameter]) -> None:
        """Load parameters from a state dictionary."""
        self.param1.data = state_dict["param1"].data
        self.param2.data = state_dict["param2"].data

    def init_from_data(self, data_loader: DataLoader) -> None:
        """Mock data initialization."""

    @classmethod
    def get_config_class(cls) -> type[ModelConfig]:
        """Return the configuration class for this mock model."""
        return ModelConfig


class TestTrainer:
    """Test Trainer class."""

    @pytest.fixture
    def training_config(self, tmp_path: Path) -> TrainingConfig:
        """Create training configuration."""
        return TrainingConfig(
            epochs=5,
            batch_size=32,
            optimizer=OptimizerConfig(name="sgd", lr=0.01, momentum=0.9),
            checkpoint_dir=tmp_path / "checkpoints",
            checkpoint_every=2,
            log_every=10,
            eval_every=1,
            early_stopping=False,
            mixed_precision=False,
            compile_model=False,
        )

    @pytest.fixture
    def mock_gradient_estimator(self) -> GradientEstimator:
        """Create mock gradient estimator."""
        estimator = Mock(spec=GradientEstimator)
        estimator.estimate_gradient.return_value = {
            "param1": torch.randn(10, 10) * 0.01,
            "param2": torch.randn(10) * 0.01,
        }
        estimator.compute_metrics.return_value = {
            "energy_gap": 1.0,
            "reconstruction_error": 0.1,
            "data_energy": -10.0,
            "sample_energy": -9.0,
        }
        estimator.last_negative_samples = torch.randn(32, 10)
        return estimator

    @pytest.fixture
    def simple_data_loader(self) -> DataLoader:
        """Create simple data loader."""
        data = torch.randn(100, 10)
        dataset = TensorDataset(data)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def test_initialization(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
    ) -> None:
        """Test trainer initialization."""
        model = MockModel()
        trainer = Trainer(
            model=model,
            config=training_config,
            gradient_estimator=mock_gradient_estimator,
        )

        assert trainer.model is model
        assert trainer.config is training_config
        assert trainer.gradient_estimator is mock_gradient_estimator
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_metric == float("inf")

        # Check optimizer creation
        assert isinstance(trainer.optimizer, torch.optim.SGD)
        assert trainer.optimizer.param_groups[0]["lr"] == 0.01
        assert trainer.optimizer.param_groups[0]["momentum"] == 0.9

    def test_optimizer_creation(self, training_config: TrainingConfig) -> None:
        """Test different optimizer configurations."""
        model = MockModel()
        estimator = Mock(spec=GradientEstimator)

        # Test Adam
        training_config.optimizer.name = "adam"
        training_config.optimizer.betas = (0.9, 0.999)
        trainer = Trainer(model, training_config, estimator)
        assert isinstance(trainer.optimizer, torch.optim.Adam)

        # Test AdamW
        training_config.optimizer.name = "adamw"
        trainer = Trainer(model, training_config, estimator)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

        # Test RMSprop
        training_config.optimizer.name = "rmsprop"
        trainer = Trainer(model, training_config, estimator)
        assert isinstance(trainer.optimizer, torch.optim.RMSprop)

    def test_scheduler_creation(self, training_config: TrainingConfig) -> None:
        """Test learning rate scheduler creation."""
        model = MockModel()
        estimator = Mock(spec=GradientEstimator)

        # Step scheduler
        training_config.optimizer.scheduler = "step"
        training_config.optimizer.scheduler_params = {
            "step_size": 10,
            "gamma": 0.1,
        }
        trainer = Trainer(model, training_config, estimator)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)

        # Cosine scheduler
        training_config.optimizer.scheduler = "cosine"
        trainer = Trainer(model, training_config, estimator)
        assert isinstance(
            trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
        )

        # No scheduler
        training_config.optimizer.scheduler = None
        trainer = Trainer(model, training_config, estimator)
        assert trainer.scheduler is None

    def test_callback_setup(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
    ) -> None:
        """Test callback initialization."""
        model = MockModel()

        # Custom callback
        custom_callback = Mock(spec=Callback)

        trainer = Trainer(
            model=model,
            config=training_config,
            gradient_estimator=mock_gradient_estimator,
            callbacks=[custom_callback],
        )

        # Should have default callbacks plus custom
        assert isinstance(trainer.callbacks, CallbackList)
        assert custom_callback in trainer.callbacks.callbacks

        # Check default callbacks exist
        callback_types = [
            type(cb).__name__ for cb in trainer.callbacks.callbacks
        ]
        assert "MetricsCallback" in callback_types
        assert "LoggingCallback" in callback_types
        assert "CheckpointCallback" in callback_types  # checkpoint_every > 0

    def test_training_step(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
    ) -> None:
        """Test single training step."""
        model = MockModel()
        trainer = Trainer(model, training_config, mock_gradient_estimator)

        data = torch.randn(32, 10)
        loss, metrics = trainer._training_step(data)

        # Check gradient estimation was called
        mock_gradient_estimator.estimate_gradient.assert_called_once_with(
            model, data
        )

        # Check gradients were applied
        assert model.param1.grad is not None
        assert model.param2.grad is not None

        # Check metrics
        assert isinstance(loss, float)
        assert "grad_norm" in metrics
        assert metrics["grad_norm"] > 0

    def test_gradient_clipping(
        self, mock_gradient_estimator: GradientEstimator
    ) -> None:
        """Test gradient clipping."""
        model = MockModel()
        config = TrainingConfig(
            epochs=1,
            batch_size=32,
            optimizer=OptimizerConfig(name="sgd", lr=0.01),
            grad_clip=1.0,
        )

        trainer = Trainer(model, config, mock_gradient_estimator)

        # Set large gradients
        mock_gradient_estimator.estimate_gradient.return_value = {
            "param1": torch.ones(10, 10) * 100,  # Large gradient
            "param2": torch.ones(10) * 100,
        }

        data = torch.randn(32, 10)
        trainer._training_step(data)

        # Check gradients were clipped
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm**0.5

        assert total_norm <= 1.0 * 1.1  # Allow small numerical error

    def test_train_epoch(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
        simple_data_loader: DataLoader,
    ) -> None:
        """Test training for one epoch."""
        model = MockModel()
        trainer = Trainer(model, training_config, mock_gradient_estimator)

        # Run one epoch
        metrics = trainer._train_epoch(simple_data_loader)

        # Check metrics
        assert "epoch_time" in metrics
        assert "lr" in metrics
        assert metrics["lr"] == 0.01

        # Check that multiple batches were processed
        expected_batches = len(simple_data_loader)
        assert (
            mock_gradient_estimator.estimate_gradient.call_count
            == expected_batches
        )

        # Check global step increased
        assert trainer.global_step == expected_batches

    def test_validation(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
        simple_data_loader: DataLoader,
    ) -> None:
        """Test validation."""
        model = MockModel()

        # Add reconstruct method for validation
        model.reconstruct = Mock(return_value=torch.randn(32, 10))

        trainer = Trainer(model, training_config, mock_gradient_estimator)

        # Run validation
        val_metrics = trainer._validate(simple_data_loader)

        assert "val_reconstruction_error" in val_metrics
        assert "val_free_energy" in val_metrics

        # Should be in eval mode
        assert model.reconstruct.called

    def test_fit_method(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
        simple_data_loader: DataLoader,
    ) -> None:
        """Test complete training loop."""
        model = MockModel()
        trainer = Trainer(model, training_config, mock_gradient_estimator)

        # Mock callbacks to track calls
        trainer.callbacks.on_train_begin = Mock()
        trainer.callbacks.on_train_end = Mock()
        trainer.callbacks.on_epoch_start = Mock()
        trainer.callbacks.on_epoch_end = Mock()

        # Run training
        result = trainer.fit(
            train_loader=simple_data_loader,
            num_epochs=2,  # Override config
        )

        # Check callbacks were called
        trainer.callbacks.on_train_begin.assert_called_once()
        assert trainer.callbacks.on_epoch_start.call_count == 2
        assert trainer.callbacks.on_epoch_end.call_count == 2

        # Check result
        assert "history" in result
        assert "final_metrics" in result
        assert "best_metric" in result
        assert len(result["history"]["train"]) == 2

    def test_early_stopping(
        self,
        mock_gradient_estimator: GradientEstimator,
        simple_data_loader: DataLoader,
    ) -> None:
        """Test early stopping functionality."""
        model = MockModel()
        config = TrainingConfig(
            epochs=10,
            batch_size=32,
            optimizer=OptimizerConfig(name="sgd", lr=0.01),
            early_stopping=True,
            patience=2,
        )

        trainer = Trainer(model, config, mock_gradient_estimator)

        # Mock early stopping trigger
        trainer.callbacks.should_stop = False

        def stop_after_3_epochs(*_args: Any, **_kwargs: Any) -> None:
            if trainer.current_epoch >= 2:
                trainer.callbacks._should_stop = True

        trainer.callbacks.on_epoch_end = Mock(side_effect=stop_after_3_epochs)

        # Run training
        result = trainer.fit(simple_data_loader)

        # Should stop early
        assert len(result["history"]["train"]) == 3

    def test_checkpoint_save_load(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
        tmp_path: Path,
    ) -> None:
        """Test checkpoint saving and loading."""
        model = MockModel()
        trainer = Trainer(model, training_config, mock_gradient_estimator)

        # Modify state
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.best_metric = 0.5

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        saved_path = trainer.save_checkpoint(checkpoint_path)

        assert saved_path.exists()

        # Create new trainer and load
        new_model = MockModel()
        new_trainer = Trainer(
            new_model, training_config, mock_gradient_estimator
        )

        new_trainer.load_checkpoint(saved_path)

        # Check state restored
        assert new_trainer.current_epoch == 5
        assert new_trainer.global_step == 100
        assert new_trainer.best_metric == 0.5

        # Check model parameters
        assert torch.allclose(new_model.param1, model.param1)
        assert torch.allclose(new_model.param2, model.param2)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_model_compilation(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
    ) -> None:
        """Test model compilation with torch.compile."""
        model = MockModel()
        training_config.compile_model = True

        with patch("torch.compile") as mock_compile:
            mock_compile.return_value = model

            Trainer(model, training_config, mock_gradient_estimator)

            mock_compile.assert_called_once_with(model)

    def test_mixed_precision(
        self, mock_gradient_estimator: GradientEstimator
    ) -> None:
        """Test mixed precision training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")

        model = MockModel()
        config = TrainingConfig(
            epochs=1,
            batch_size=32,
            optimizer=OptimizerConfig(name="sgd", lr=0.01),
            mixed_precision=True,
        )

        trainer = Trainer(model, config, mock_gradient_estimator)

        # Check scaler created
        assert trainer.scaler is not None
        assert isinstance(trainer.scaler, torch.cuda.amp.GradScaler)

    def test_interrupt_handling(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
        simple_data_loader: DataLoader,
    ) -> None:
        """Test handling of keyboard interrupt."""
        model = MockModel()
        trainer = Trainer(model, training_config, mock_gradient_estimator)

        # Mock keyboard interrupt after first epoch
        original_train_epoch = trainer._train_epoch
        call_count = 0

        def interrupt_after_one(*args: Any, **kwargs: Any) -> dict[str, float]:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise KeyboardInterrupt()
            return original_train_epoch(*args, **kwargs)

        trainer._train_epoch = Mock(side_effect=interrupt_after_one)

        # Run training
        with patch("ebm.training.trainer.logger") as mock_logger:
            trainer.fit(simple_data_loader)

            # Check warning logged
            mock_logger.warning.assert_called_with(
                "Training interrupted by user"
            )

        # Should have completed one epoch
        assert call_count == 2


class TestTrainerEdgeCases:
    """Test edge cases for Trainer."""

    def test_empty_data_loader(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
    ) -> None:
        """Test training with empty data loader."""
        model = MockModel()
        trainer = Trainer(model, training_config, mock_gradient_estimator)

        # Empty data loader
        empty_dataset = TensorDataset(torch.empty(0, 10))
        empty_loader = DataLoader(empty_dataset, batch_size=32)

        # Should handle gracefully
        trainer._train_epoch(empty_loader)
        assert trainer.global_step == 0

    def test_single_batch(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
    ) -> None:
        """Test training with single batch."""
        model = MockModel()
        trainer = Trainer(model, training_config, mock_gradient_estimator)

        # Single batch
        data = torch.randn(10, 10)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=32)

        trainer._train_epoch(loader)
        assert trainer.global_step == 1

    def test_different_batch_formats(
        self,
        training_config: TrainingConfig,
        mock_gradient_estimator: GradientEstimator,
    ) -> None:
        """Test handling different batch formats."""
        model = MockModel()
        trainer = Trainer(model, training_config, mock_gradient_estimator)

        # Test with (data, label) format
        data = torch.randn(32, 10)
        labels = torch.randint(0, 2, (32,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=32)

        # Should extract just data
        next(iter(loader))
        loss, metrics = trainer._training_step(data)  # Pass data directly

        assert isinstance(loss, float)
        mock_gradient_estimator.estimate_gradient.assert_called()

    def test_scheduler_with_validation_metric(
        self,
        mock_gradient_estimator: GradientEstimator,
        simple_data_loader: DataLoader,
    ) -> None:
        """Test ReduceLROnPlateau scheduler."""
        model = MockModel()
        config = TrainingConfig(
            epochs=3,
            batch_size=32,
            optimizer=OptimizerConfig(
                name="sgd",
                lr=0.1,
                scheduler="reduce_on_plateau",
                scheduler_params={"factor": 0.5, "patience": 1},
            ),
            eval_every=1,
        )

        trainer = Trainer(model, config, mock_gradient_estimator)

        # Mock validation metrics
        trainer._validate = Mock(return_value={"loss": 1.0})

        # Run training
        trainer.fit(simple_data_loader, simple_data_loader)

        # Scheduler should have been called
        assert isinstance(
            trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )
