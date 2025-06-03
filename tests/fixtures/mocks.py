"""Mock objects for testing."""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from ebm.models.base import EnergyBasedModel
from ebm.sampling.base import GradientEstimator, Sampler
from ebm.training.callbacks import Callback
from ebm.training.trainer import Trainer


@pytest.fixture
def mock_data_loader() -> Callable[[int, int, int], DataLoader]:
    """Provide a mock data loader for testing."""

    def _make_mock_loader(
        n_batches: int = 10, batch_size: int = 32, n_features: int = 100
    ) -> DataLoader:
        """Create a mock data loader."""
        # Create synthetic data
        data = torch.rand(n_batches * batch_size, n_features)
        dataset = TensorDataset(data)

        # Create loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Add length property
        loader.__len__ = lambda: n_batches

        return loader

    return _make_mock_loader


@pytest.fixture
def mock_gradient_estimator() -> type[GradientEstimator]:
    """Provide a mock gradient estimator."""

    class MockGradientEstimator(GradientEstimator):
        def __init__(
            self, return_gradients: dict[str, torch.Tensor] | None = None
        ):
            # Create a mock sampler
            mock_sampler = Mock(spec=Sampler)
            super().__init__(mock_sampler)

            self.return_gradients = return_gradients or {}
            self.call_count = 0
            self.last_model = None
            self.last_data = None
            self.last_negative_samples = None

        def estimate_gradient(
            self, model: EnergyBasedModel, data: torch.Tensor, **kwargs: Any
        ) -> dict[str, torch.Tensor]:
            """Return mock gradients."""
            self.call_count += 1
            self.last_model = model
            self.last_data = data

            # Generate mock gradients if not provided
            if not self.return_gradients:
                gradients = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        gradients[name] = torch.randn_like(param) * 0.01
                return gradients

            return self.return_gradients

        def compute_metrics(
            self,
            model: EnergyBasedModel,
            data: torch.Tensor,
            samples: torch.Tensor,
        ) -> dict[str, float]:
            """Return mock metrics."""
            return {
                "energy_gap": 1.0,
                "reconstruction_error": 0.1,
                "data_energy": -10.0,
                "sample_energy": -9.0,
            }

    return MockGradientEstimator


@pytest.fixture
def mock_callback() -> type[Callback]:
    """Provide a mock callback for testing."""

    class MockCallback(Callback):
        def __init__(self) -> None:
            self.events = []
            self.epoch_metrics = []
            self.should_stop = False

        def on_train_begin(self, trainer: Trainer) -> None:
            self.events.append("train_begin")

        def on_train_end(self, trainer: Trainer) -> None:
            self.events.append("train_end")

        def on_epoch_start(
            self, trainer: Trainer, model: EnergyBasedModel
        ) -> None:
            self.events.append(f"epoch_start_{trainer.current_epoch}")

        def on_epoch_end(
            self,
            trainer: Trainer,
            model: EnergyBasedModel,
            metrics: dict[str, float],
        ) -> None:
            self.events.append(f"epoch_end_{trainer.current_epoch}")
            self.epoch_metrics.append(metrics)

            # Optionally trigger early stopping
            if self.should_stop:
                trainer.callbacks.stop_training()

        def on_batch_start(
            self, trainer: Trainer, model: EnergyBasedModel, batch: torch.Tensor
        ) -> None:
            self.events.append("batch_start")

        def on_batch_end(
            self, trainer: Trainer, model: EnergyBasedModel, loss: float
        ) -> None:
            self.events.append("batch_end")

    return MockCallback


@pytest.fixture
def mock_sampler() -> type[Sampler]:
    """Provide a mock sampler for testing."""

    class MockSampler(Sampler):
        def __init__(self, return_same: bool = False):
            super().__init__(name="MockSampler")
            self.return_same = return_same
            self.sample_count = 0
            self.last_init_state = None

        def sample(
            self,
            model: EnergyBasedModel,
            init_state: torch.Tensor,
            num_steps: int = 1,
            **kwargs: Any,
        ) -> torch.Tensor:
            """Return mock samples."""
            self.sample_count += 1
            self.last_init_state = init_state

            if self.return_same:
                return init_state.clone()
            # Return slightly perturbed version
            noise = torch.randn_like(init_state) * 0.1
            samples = torch.clamp(init_state + noise, 0, 1)
            return (samples > 0.5).to(init_state.dtype)

    return MockSampler


@pytest.fixture
def mock_model() -> Callable[[int, int, str], EnergyBasedModel]:
    """Provide a mock energy-based model."""

    def _make_mock_model(
        n_visible: int = 100, n_hidden: int = 50, device: str = "cpu"
    ) -> EnergyBasedModel:
        """Create a mock EBM."""
        model = MagicMock(spec=EnergyBasedModel)

        # Set basic properties
        model.num_visible = n_visible
        model.num_hidden = n_hidden
        model.device = torch.device(device)
        model.dtype = torch.float32

        # Mock methods
        model.free_energy.side_effect = lambda x: torch.randn(x.shape[0])
        model.energy.side_effect = lambda x: torch.randn(x.shape[0])

        # Mock parameters
        model.parameters.return_value = [
            torch.randn(n_hidden, n_visible, requires_grad=True),
            torch.randn(n_visible, requires_grad=True),
            torch.randn(n_hidden, requires_grad=True),
        ]

        model.named_parameters.return_value = [
            ("W", model.parameters.return_value[0]),
            ("vbias", model.parameters.return_value[1]),
            ("hbias", model.parameters.return_value[2]),
        ]

        return model

    return _make_mock_model


@pytest.fixture
def mock_optimizer() -> Callable[[float], Optimizer]:
    """Provide a mock optimizer."""

    def _make_mock_optimizer(lr: float = 0.01) -> Optimizer:
        """Create a mock optimizer."""
        optimizer = MagicMock()
        optimizer.param_groups = [{"lr": lr}]
        optimizer.state = {}
        optimizer.step_count = 0

        def step() -> None:
            optimizer.step_count += 1

        optimizer.step.side_effect = step

        return optimizer

    return _make_mock_optimizer


@pytest.fixture
def mock_scheduler() -> Callable[[Optimizer], object]:
    """Provide a mock learning rate scheduler."""

    def _make_mock_scheduler(optimizer: Optimizer) -> object:
        """Create a mock scheduler."""
        scheduler = MagicMock()
        scheduler.optimizer = optimizer
        scheduler.step_count = 0

        def step(metric: float | None = None) -> None:
            scheduler.step_count += 1
            # Simulate learning rate decay
            for group in optimizer.param_groups:
                group["lr"] *= 0.95

        scheduler.step.side_effect = step
        scheduler.get_last_lr.return_value = [optimizer.param_groups[0]["lr"]]

        return scheduler

    return _make_mock_scheduler
