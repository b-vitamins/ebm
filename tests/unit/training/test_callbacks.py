"""Unit tests for training callbacks."""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from ebm.models.base import EnergyBasedModel
from ebm.training.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
    LoggingCallback,
    MetricsCallback,
    VisualizationCallback,
    WarmupCallback,
)


class MockTrainer:
    """Mock trainer for testing callbacks."""

    def __init__(self) -> None:
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.optimizer = Mock()
        self.optimizer.param_groups = [{"lr": 0.01}]
        self.callbacks = Mock()
        self.callbacks._should_stop = False

    def save_checkpoint(self, path: Path) -> Path:
        """Mock checkpoint saving."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": self.current_epoch}, path)
        return path


class MockModel(EnergyBasedModel):
    """Mock model for testing."""

    def __init__(self) -> None:
        self.W = torch.randn(10, 20)
        self.vbias = torch.randn(20)
        self.hbias = torch.randn(10)

    def named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        """Return parameters with names for the mock model."""
        return [
            ("W", torch.nn.Parameter(self.W)),
            ("vbias", torch.nn.Parameter(self.vbias)),
            ("hbias", torch.nn.Parameter(self.hbias)),
        ]

    def parameters(self) -> list[torch.nn.Parameter]:
        """Return list of model parameters."""
        return [p for _, p in self.named_parameters()]

    def sample_fantasy_particles(
        self, num_samples: int, _num_steps: int
    ) -> torch.Tensor:
        """Generate random fantasy particles."""
        return torch.randn(num_samples, 20)

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model."""
        return torch.float32

    def energy(
        self,
        x: torch.Tensor,
        *,
        _beta: float | None = None,
        _return_parts: bool = False,
    ) -> torch.Tensor:
        """Return zero energy for all inputs."""
        return torch.zeros(x.shape[0])

    def free_energy(
        self, v: torch.Tensor, *, _beta: float | None = None
    ) -> torch.Tensor:
        """Return zero free energy for all inputs."""
        return torch.zeros(v.shape[0])


class TestCallback:
    """Test base Callback class."""

    def test_default_methods(self) -> None:
        """Test that default callback methods do nothing."""
        callback = Callback()
        trainer = MockTrainer()
        model = MockModel()

        # All methods should be no-ops by default
        callback.on_train_begin(trainer)
        callback.on_train_end(trainer)
        callback.on_epoch_start(trainer, model)
        callback.on_epoch_end(trainer, model, {"loss": 0.5})
        callback.on_batch_start(trainer, model, torch.randn(32, 10))
        callback.on_batch_end(trainer, model, 0.5)
        callback.on_validation_start(trainer, model)
        callback.on_validation_end(trainer, model, {"val_loss": 0.4})

        # Should not raise any errors


class TestCallbackList:
    """Test CallbackList container."""

    def test_initialization(self) -> None:
        """Test callback list initialization."""
        cb1 = Mock(spec=Callback)
        cb2 = Mock(spec=Callback)

        callback_list = CallbackList([cb1, cb2])

        assert len(callback_list.callbacks) == 2
        assert cb1 in callback_list.callbacks
        assert cb2 in callback_list.callbacks
        assert callback_list.should_stop is False

    def test_delegation(self) -> None:
        """Test that methods are delegated to all callbacks."""
        cb1 = Mock(spec=Callback)
        cb2 = Mock(spec=Callback)

        callback_list = CallbackList([cb1, cb2])
        trainer = MockTrainer()
        model = MockModel()

        # Call a method
        callback_list.on_epoch_start(trainer, model)

        # Both callbacks should be called
        cb1.on_epoch_start.assert_called_once_with(trainer, model)
        cb2.on_epoch_start.assert_called_once_with(trainer, model)

    def test_stop_training(self) -> None:
        """Test stop training flag."""
        callback_list = CallbackList([])

        assert callback_list.should_stop is False

        callback_list.stop_training()

        assert callback_list.should_stop is True

    def test_missing_method_handling(self) -> None:
        """Test handling of callbacks without all methods."""

        # Callback with only some methods
        class PartialCallback:
            def on_epoch_start(
                self, _trainer: MockTrainer, _model: MockModel
            ) -> None:
                self.called = True

        partial_cb = PartialCallback()
        full_cb = Mock(spec=Callback)

        callback_list = CallbackList([partial_cb, full_cb])
        trainer = MockTrainer()
        model = MockModel()

        # Should handle missing methods gracefully
        callback_list.on_epoch_start(trainer, model)
        callback_list.on_epoch_end(trainer, model, {})

        assert partial_cb.called
        full_cb.on_epoch_start.assert_called_once()
        full_cb.on_epoch_end.assert_called_once()


class TestLoggingCallback:
    """Test LoggingCallback."""

    def test_initialization(self) -> None:
        """Test logging callback initialization."""
        callback = LoggingCallback(
            log_every=50, log_gradients=True, log_weights=True
        )

        assert callback.log_every == 50
        assert callback.log_gradients is True
        assert callback.log_weights is True
        assert callback.step_count == 0
        assert callback.epoch_start_time is None

    @patch("ebm.training.callbacks.logger")
    def test_epoch_logging(self, mock_logger: Mock) -> None:
        """Test epoch start/end logging."""
        callback = LoggingCallback()
        trainer = MockTrainer()
        trainer.current_epoch = 1
        model = MockModel()

        # Start epoch
        callback.on_epoch_start(trainer, model)
        mock_logger.info.assert_called_with("Starting epoch 1", lr=0.01)
        assert callback.epoch_start_time is not None

        # End epoch
        time.sleep(0.1)  # Ensure some time passes
        metrics = {"loss": 0.5, "accuracy": 0.95}
        callback.on_epoch_end(trainer, model, metrics)

        # Check logging
        assert mock_logger.info.call_count == 2
        end_call = mock_logger.info.call_args_list[1]
        assert "Epoch 1 completed" in end_call[0][0]
        assert "loss=0.5000, accuracy=0.9500" in end_call[1]["metrics"]

    @patch("ebm.training.callbacks.logger")
    def test_batch_logging(self, mock_logger: Mock) -> None:
        """Test batch logging."""
        callback = LoggingCallback(log_every=2)
        trainer = MockTrainer()
        model = MockModel()

        # First batch - no log
        callback.on_batch_end(trainer, model, 0.5)
        assert callback.step_count == 1
        mock_logger.debug.assert_not_called()

        # Second batch - should log
        trainer.global_step = 2
        callback.on_batch_end(trainer, model, 0.4)
        assert callback.step_count == 2

        mock_logger.debug.assert_called_once()
        log_call = mock_logger.debug.call_args
        assert log_call[0][0] == "Training step"
        assert log_call[1]["step"] == 2
        assert log_call[1]["loss"] == 0.4

    @patch("ebm.training.callbacks.logger")
    def test_gradient_logging(self, mock_logger: Mock) -> None:
        """Test gradient statistics logging."""
        callback = LoggingCallback(log_every=1, log_gradients=True)
        trainer = MockTrainer()
        model = MockModel()

        # Set gradients
        for _name, param in model.named_parameters():
            param.grad = torch.randn_like(param) * 0.01

        trainer.global_step = 1
        callback.on_batch_end(trainer, model, 0.5)

        log_call = mock_logger.debug.call_args
        assert "grad_norm_mean" in log_call[1]
        assert "grad_norm_max" in log_call[1]
        assert log_call[1]["grad_norm_mean"] > 0

    @patch("ebm.training.callbacks.logger")
    def test_weight_logging(self, mock_logger: Mock) -> None:
        """Test weight statistics logging."""
        callback = LoggingCallback(log_every=1, log_weights=True)
        trainer = MockTrainer()
        model = MockModel()

        trainer.global_step = 1
        callback.on_batch_end(trainer, model, 0.5)

        log_call = mock_logger.debug.call_args
        assert "weight_norm_mean" in log_call[1]
        assert log_call[1]["weight_norm_mean"] > 0


class TestMetricsCallback:
    """Test MetricsCallback."""

    def test_initialization(self) -> None:
        """Test metrics callback initialization."""
        callback = MetricsCallback()

        assert isinstance(callback.train_metrics, list)
        assert isinstance(callback.val_metrics, list)
        assert len(callback.train_metrics) == 0
        assert len(callback.val_metrics) == 0

    def test_metrics_storage(self) -> None:
        """Test metrics storage."""
        callback = MetricsCallback()
        trainer = MockTrainer()
        trainer.current_epoch = 1
        model = MockModel()

        # Store training metrics
        train_metrics = {"loss": 0.5, "accuracy": 0.9}
        callback.on_epoch_end(trainer, model, train_metrics)

        assert len(callback.train_metrics) == 1
        assert callback.train_metrics[0]["loss"] == 0.5
        assert callback.train_metrics[0]["epoch"] == 1
        assert "timestamp" in callback.train_metrics[0]

        # Store validation metrics
        val_metrics = {"val_loss": 0.4, "val_accuracy": 0.92}
        callback.on_validation_end(trainer, model, val_metrics)

        assert len(callback.val_metrics) == 1
        assert callback.val_metrics[0]["val_loss"] == 0.4
        assert callback.val_metrics[0]["epoch"] == 1

    def test_save_metrics(self, tmp_path: Path) -> None:
        """Test saving metrics to file."""
        save_path = tmp_path / "metrics.json"
        callback = MetricsCallback(save_path=save_path)
        trainer = MockTrainer()
        model = MockModel()

        # Add some metrics
        callback.on_epoch_end(trainer, model, {"loss": 0.5})
        callback.on_validation_end(trainer, model, {"val_loss": 0.4})

        # Check file saved
        assert save_path.exists()

        # Load and verify
        with save_path.open() as f:
            data = json.load(f)

        assert "train" in data
        assert "val" in data
        assert len(data["train"]) == 1
        assert len(data["val"]) == 1
        assert data["train"][0]["loss"] == 0.5
        assert data["val"][0]["val_loss"] == 0.4


class TestCheckpointCallback:
    """Test CheckpointCallback."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test checkpoint callback initialization."""
        callback = CheckpointCallback(
            checkpoint_dir=tmp_path,
            save_every=5,
            save_best=True,
            monitor="val_loss",
            mode="min",
        )

        assert callback.checkpoint_dir == tmp_path
        assert callback.save_every == 5
        assert callback.save_best is True
        assert callback.monitor == "val_loss"
        assert callback.mode == "min"
        assert callback.best_value == float("inf")

    def test_regular_checkpointing(self, tmp_path: Path) -> None:
        """Test regular epoch checkpointing."""
        callback = CheckpointCallback(checkpoint_dir=tmp_path, save_every=2)
        trainer = MockTrainer()
        model = MockModel()

        # Epoch 0 - should save
        trainer.current_epoch = 0
        callback.on_epoch_end(trainer, model, {})
        assert (tmp_path / "checkpoint_epoch_0.pt").exists()

        # Epoch 1 - should not save
        trainer.current_epoch = 1
        callback.on_epoch_end(trainer, model, {})
        assert not (tmp_path / "checkpoint_epoch_1.pt").exists()

        # Epoch 2 - should save
        trainer.current_epoch = 2
        callback.on_epoch_end(trainer, model, {})
        assert (tmp_path / "checkpoint_epoch_2.pt").exists()

    @patch("ebm.training.callbacks.logger")
    def test_best_model_saving(self, mock_logger: Mock, tmp_path: Path) -> None:
        """Test best model checkpointing."""
        callback = CheckpointCallback(
            checkpoint_dir=tmp_path,
            save_every=10,  # High value to avoid regular saves
            save_best=True,
            monitor="loss",
            mode="min",
        )
        trainer = MockTrainer()
        model = MockModel()

        # First epoch - best so far
        trainer.current_epoch = 1
        callback.on_epoch_end(trainer, model, {"loss": 0.5})
        assert (tmp_path / "best_model.pt").exists()
        assert callback.best_value == 0.5

        # Second epoch - worse
        trainer.current_epoch = 2
        callback.on_epoch_end(trainer, model, {"loss": 0.6})
        # Best value shouldn't change
        assert callback.best_value == 0.5

        # Third epoch - better
        trainer.current_epoch = 3
        callback.on_epoch_end(trainer, model, {"loss": 0.4})
        assert callback.best_value == 0.4

        # Check logging
        assert mock_logger.info.call_count == 2  # Called for epochs 1 and 3

    def test_mode_max(self, tmp_path: Path) -> None:
        """Test monitoring with mode='max'."""
        callback = CheckpointCallback(
            checkpoint_dir=tmp_path,
            save_best=True,
            monitor="accuracy",
            mode="max",
        )
        trainer = MockTrainer()
        model = MockModel()

        assert callback.best_value == float("-inf")

        # Should save when metric increases
        callback.on_epoch_end(trainer, model, {"accuracy": 0.8})
        assert callback.best_value == 0.8

        callback.on_epoch_end(trainer, model, {"accuracy": 0.9})
        assert callback.best_value == 0.9

        callback.on_epoch_end(trainer, model, {"accuracy": 0.85})
        assert callback.best_value == 0.9  # Should not update


class TestEarlyStoppingCallback:
    """Test EarlyStoppingCallback."""

    def test_initialization(self) -> None:
        """Test early stopping initialization."""
        callback = EarlyStoppingCallback(
            patience=5, min_delta=0.001, monitor="val_loss", mode="min"
        )

        assert callback.patience == 5
        assert callback.min_delta == 0.001
        assert callback.monitor == "val_loss"
        assert callback.mode == "min"
        assert callback.best_value == float("inf")
        assert callback.patience_counter == 0
        assert callback.trainer is None

    def test_improvement_detection(self) -> None:
        """Test improvement detection."""
        callback = EarlyStoppingCallback(
            patience=3, min_delta=0.01, monitor="loss", mode="min"
        )
        trainer = MockTrainer()
        model = MockModel()

        callback.on_train_begin(trainer)

        # Initial value
        callback.on_epoch_end(trainer, model, {"loss": 1.0})
        assert callback.best_value == 1.0
        assert callback.patience_counter == 0

        # Improvement
        callback.on_epoch_end(trainer, model, {"loss": 0.95})
        assert callback.best_value == 0.95
        assert callback.patience_counter == 0

        # No improvement (within min_delta)
        callback.on_epoch_end(trainer, model, {"loss": 0.945})
        assert callback.best_value == 0.95
        assert callback.patience_counter == 1

        # No improvement
        callback.on_epoch_end(trainer, model, {"loss": 0.96})
        assert callback.patience_counter == 2

    @patch("ebm.training.callbacks.logger")
    def test_early_stopping_trigger(self, mock_logger: Mock) -> None:
        """Test early stopping trigger."""
        callback = EarlyStoppingCallback(patience=2, monitor="loss", mode="min")
        trainer = MockTrainer()
        model = MockModel()

        callback.on_train_begin(trainer)

        # Good epochs
        callback.on_epoch_end(trainer, model, {"loss": 1.0})
        callback.on_epoch_end(trainer, model, {"loss": 0.9})

        # Bad epochs
        callback.on_epoch_end(trainer, model, {"loss": 0.91})
        callback.on_epoch_end(trainer, model, {"loss": 0.92})

        # Should not trigger yet
        assert not trainer.callbacks._should_stop

        # One more bad epoch - should trigger
        callback.on_epoch_end(trainer, model, {"loss": 0.93})

        # Check stop signal
        trainer.callbacks.stop_training.assert_called_once()

        # Check logging
        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args
        assert "Early stopping triggered" in log_call[0][0]

    def test_missing_monitor_metric(self) -> None:
        """Test behavior when monitor metric is missing."""
        callback = EarlyStoppingCallback(monitor="val_loss")
        trainer = MockTrainer()
        model = MockModel()

        # Metric not in results
        callback.on_epoch_end(trainer, model, {"loss": 0.5})

        # Should not update counters
        assert callback.patience_counter == 0
        assert callback.best_value == float("inf")


class TestVisualizationCallback:
    """Test VisualizationCallback."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test visualization callback initialization."""
        callback = VisualizationCallback(
            visualize_every=5, num_samples=32, save_dir=tmp_path
        )

        assert callback.visualize_every == 5
        assert callback.num_samples == 32
        assert callback.save_dir == tmp_path
        assert tmp_path.exists()

    @patch("ebm.utils.visualization.visualize_samples")
    @patch("ebm.utils.visualization.visualize_filters")
    def test_visualization_generation(
        self,
        mock_vis_filters: Mock,
        mock_vis_samples: Mock,
        tmp_path: Path,
    ) -> None:
        """Test visualization generation."""
        callback = VisualizationCallback(
            visualize_every=2, num_samples=16, save_dir=tmp_path
        )
        trainer = MockTrainer()
        model = MockModel()

        # Epoch 0 - should visualize
        trainer.current_epoch = 0
        callback.on_epoch_end(trainer, model, {})

        mock_vis_samples.assert_called_once()
        mock_vis_filters.assert_called_once()

        # Check save paths
        samples_path = tmp_path / "samples_epoch_0.png"
        filters_path = tmp_path / "filters_epoch_0.png"

        samples_call = mock_vis_samples.call_args
        assert samples_call[1]["save_path"] == samples_path

        filters_call = mock_vis_filters.call_args
        assert filters_call[1]["save_path"] == filters_path

        # Epoch 1 - should not visualize
        mock_vis_samples.reset_mock()
        mock_vis_filters.reset_mock()

        trainer.current_epoch = 1
        callback.on_epoch_end(trainer, model, {})

        mock_vis_samples.assert_not_called()
        mock_vis_filters.assert_not_called()


class TestLearningRateSchedulerCallback:
    """Test LearningRateSchedulerCallback."""

    def test_initialization(self) -> None:
        """Test LR scheduler callback initialization."""

        def schedule_fn(epoch: int, _step: int) -> float:
            return 0.1 * (0.9**epoch)

        callback = LearningRateSchedulerCallback(
            schedule_fn=schedule_fn, update_every="epoch"
        )

        assert callback.schedule_fn is schedule_fn
        assert callback.update_every == "epoch"

    def test_epoch_update(self) -> None:
        """Test LR update at epoch start."""

        def schedule_fn(epoch: int, _step: int) -> float:
            return 0.1 * (0.5**epoch)

        callback = LearningRateSchedulerCallback(
            schedule_fn=schedule_fn, update_every="epoch"
        )
        trainer = MockTrainer()
        model = MockModel()

        # Epoch 0
        trainer.current_epoch = 0
        callback.on_epoch_start(trainer, model)
        assert trainer.optimizer.param_groups[0]["lr"] == 0.1

        # Epoch 1
        trainer.current_epoch = 1
        callback.on_epoch_start(trainer, model)
        assert trainer.optimizer.param_groups[0]["lr"] == 0.05

        # Epoch 2
        trainer.current_epoch = 2
        callback.on_epoch_start(trainer, model)
        assert trainer.optimizer.param_groups[0]["lr"] == 0.025

    def test_step_update(self) -> None:
        """Test LR update at batch start."""

        def schedule_fn(_epoch: int, _step: int) -> float:
            return 0.1 / (1 + 0.01 * _step)

        callback = LearningRateSchedulerCallback(
            schedule_fn=schedule_fn, update_every="step"
        )
        trainer = MockTrainer()
        model = MockModel()
        batch = torch.randn(32, 10)

        # Step 0
        trainer.global_step = 0
        callback.on_batch_start(trainer, model, batch)
        assert trainer.optimizer.param_groups[0]["lr"] == 0.1

        # Step 10
        trainer.global_step = 10
        callback.on_batch_start(trainer, model, batch)
        assert abs(trainer.optimizer.param_groups[0]["lr"] - 0.0909) < 0.001

        # Step 100
        trainer.global_step = 100
        callback.on_batch_start(trainer, model, batch)
        assert trainer.optimizer.param_groups[0]["lr"] == 0.05


class TestWarmupCallback:
    """Test WarmupCallback."""

    def test_initialization(self) -> None:
        """Test warmup callback initialization."""
        callback = WarmupCallback(warmup_steps=100, start_lr=1e-6, end_lr=1e-3)

        assert callback.warmup_steps == 100
        assert callback.start_lr == 1e-6
        assert callback.end_lr == 1e-3

    def test_linear_warmup(self) -> None:
        """Test linear warmup schedule."""
        callback = WarmupCallback(warmup_steps=10, start_lr=0.0, end_lr=0.1)
        trainer = MockTrainer()
        model = MockModel()
        batch = torch.randn(32, 10)

        # Step 0
        trainer.global_step = 0
        callback.on_batch_start(trainer, model, batch)
        assert trainer.optimizer.param_groups[0]["lr"] == 0.0

        # Step 5 (halfway)
        trainer.global_step = 5
        callback.on_batch_start(trainer, model, batch)
        assert abs(trainer.optimizer.param_groups[0]["lr"] - 0.05) < 0.001

        # Step 9 (almost done)
        trainer.global_step = 9
        callback.on_batch_start(trainer, model, batch)
        assert abs(trainer.optimizer.param_groups[0]["lr"] - 0.09) < 0.001

        # Step 10 (after warmup)
        trainer.global_step = 10
        trainer.optimizer.param_groups[0]["lr"] = 0.2  # Set different value
        callback.on_batch_start(trainer, model, batch)
        # Should not change after warmup
        assert trainer.optimizer.param_groups[0]["lr"] == 0.2
