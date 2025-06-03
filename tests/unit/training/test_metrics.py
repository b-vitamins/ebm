"""Unit tests for metrics tracking and evaluation."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from ebm.models.base import EnergyBasedModel, LatentVariableModel
from ebm.training.metrics import (
    MetricsTracker,
    MetricValue,
    ModelEvaluator,
    TrainingDynamicsAnalyzer,
)


class MockModel(LatentVariableModel):
    """Mock model for testing metrics."""

    def __init__(self, n_visible=10, n_hidden=5) -> None:
        self.num_visible = n_visible
        self.num_hidden = n_hidden
        self.W = torch.nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.vbias = torch.nn.Parameter(torch.zeros(n_visible))
        self.hbias = torch.nn.Parameter(torch.zeros(n_hidden))

    def free_energy(self, v, *, beta=None):
        # Simple quadratic energy
        return 0.5 * (v**2).sum(dim=-1)

    def energy(self, x, *, beta=None, return_parts=False):
        return self.free_energy(x[:, : self.num_visible], beta=beta)

    def reconstruct(self, v, num_steps=1):
        # Simple reconstruction with noise
        return v + torch.randn_like(v) * 0.1

    def sample_fantasy_particles(self, num_samples, num_steps):
        return torch.randn(num_samples, self.num_visible)

    def log_probability(self, v, log_z=0.0):
        return -self.free_energy(v) - log_z

    def sample_hidden(self, visible, *, beta=None, return_prob=False):
        prob = torch.sigmoid(visible @ self.W.T + self.hbias)
        if return_prob:
            return torch.bernoulli(prob), prob
        return torch.bernoulli(prob)

    def sample_visible(self, hidden, *, beta=None, return_prob=False):
        prob = torch.sigmoid(hidden @ self.W + self.vbias)
        if return_prob:
            return torch.bernoulli(prob), prob
        return torch.bernoulli(prob)

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32


class TestMetricValue:
    """Test MetricValue dataclass."""

    def test_initialization(self) -> None:
        """Test metric value initialization."""
        metric = MetricValue()

        assert metric.current == 0.0
        assert metric.mean == 0.0
        assert metric.std == 0.0
        assert metric.min == float("inf")
        assert metric.max == float("-inf")
        assert metric.count == 0

    def test_single_update(self) -> None:
        """Test updating with single value."""
        metric = MetricValue()

        metric.update(5.0)

        assert metric.current == 5.0
        assert metric.mean == 5.0
        assert metric.std == 0.0
        assert metric.min == 5.0
        assert metric.max == 5.0
        assert metric.count == 1

    def test_multiple_updates(self) -> None:
        """Test updating with multiple values."""
        metric = MetricValue()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            metric.update(v)

        assert metric.current == 5.0
        assert metric.mean == 3.0
        assert abs(metric.std - np.std(values, ddof=0)) < 0.001
        assert metric.min == 1.0
        assert metric.max == 5.0
        assert metric.count == 5

    def test_running_statistics(self) -> None:
        """Test running mean and std computation."""
        metric = MetricValue()

        # Generate random values
        np.random.seed(42)
        values = np.random.randn(100)

        for v in values:
            metric.update(float(v))

        # Check statistics
        assert abs(metric.mean - np.mean(values)) < 0.001
        assert abs(metric.std - np.std(values)) < 0.001
        assert abs(metric.min - np.min(values)) < 0.001
        assert abs(metric.max - np.max(values)) < 0.001


class TestMetricsTracker:
    """Test MetricsTracker class."""

    def test_initialization(self) -> None:
        """Test metrics tracker initialization."""
        tracker = MetricsTracker(window_size=50)

        assert tracker.window_size == 50
        assert len(tracker.metrics) == 0
        assert len(tracker.history) == 0

    def test_update(self) -> None:
        """Test updating metrics."""
        tracker = MetricsTracker()

        # Update with metrics
        metrics = {"loss": 0.5, "accuracy": 0.9, "lr": 0.01}
        tracker.update(metrics)

        assert "loss" in tracker.metrics
        assert "accuracy" in tracker.metrics
        assert "lr" in tracker.metrics

        assert tracker.metrics["loss"].current == 0.5
        assert tracker.metrics["accuracy"].current == 0.9

        # Check history
        assert len(tracker.history["loss"]) == 1
        assert tracker.history["loss"][0] == 0.5

    def test_window_size_limit(self) -> None:
        """Test that history respects window size."""
        tracker = MetricsTracker(window_size=5)

        # Add more than window size
        for i in range(10):
            tracker.update({"loss": float(i)})

        # Should only keep last 5
        assert len(tracker.history["loss"]) == 5
        assert list(tracker.history["loss"]) == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_get_current(self) -> None:
        """Test getting current metric values."""
        tracker = MetricsTracker()

        tracker.update({"loss": 0.5, "accuracy": 0.9})
        tracker.update({"loss": 0.4, "accuracy": 0.92})

        current = tracker.get_current()

        assert current["loss"] == 0.4
        assert current["accuracy"] == 0.92

    def test_get_average(self) -> None:
        """Test getting average values."""
        tracker = MetricsTracker()

        # Add some values
        for i in range(5):
            tracker.update({"loss": float(i)})

        # Full average
        avg = tracker.get_average()
        assert avg["loss"] == 2.0  # (0+1+2+3+4)/5

        # Windowed average
        avg_window = tracker.get_average(window=3)
        assert avg_window["loss"] == 3.0  # (2+3+4)/3

    def test_get_stats(self) -> None:
        """Test getting full statistics."""
        tracker = MetricsTracker()

        for i in range(10):
            tracker.update({"loss": float(i), "accuracy": 0.8 + 0.01 * i})

        stats = tracker.get_stats()

        assert "loss" in stats
        assert "accuracy" in stats

        loss_stats = stats["loss"]
        assert loss_stats["current"] == 9.0
        assert loss_stats["mean"] == 4.5
        assert loss_stats["min"] == 0.0
        assert loss_stats["max"] == 9.0
        assert loss_stats["count"] == 10

    def test_reset(self) -> None:
        """Test resetting tracker."""
        tracker = MetricsTracker()

        tracker.update({"loss": 0.5})
        assert len(tracker.metrics) > 0

        tracker.reset()

        assert len(tracker.metrics) == 0
        assert len(tracker.history) == 0

    def test_compute(self) -> None:
        """Test computing final values."""
        tracker = MetricsTracker()

        for i in range(5):
            tracker.update({"loss": float(i), "accuracy": 0.8 + 0.02 * i})

        final = tracker.compute()

        assert final["loss"] == 2.0
        assert final["accuracy"] == pytest.approx(0.84)


class TestModelEvaluator:
    """Test ModelEvaluator class."""

    def test_initialization(self) -> None:
        """Test evaluator initialization."""
        model = MockModel()
        evaluator = ModelEvaluator(model)

        assert evaluator.model is model

    def test_reconstruction_error(self) -> None:
        """Test reconstruction error computation."""
        model = MockModel()
        evaluator = ModelEvaluator(model)

        data = torch.randn(10, 10)

        # MSE error
        error_mse = evaluator.reconstruction_error(data, num_steps=1, metric="mse")
        assert error_mse.shape == (10,)
        assert torch.all(error_mse >= 0)

        # MAE error
        error_mae = evaluator.reconstruction_error(data, num_steps=1, metric="mae")
        assert error_mae.shape == (10,)
        assert torch.all(error_mae >= 0)

        # Invalid metric
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluator.reconstruction_error(data, metric="invalid")

    def test_reconstruction_error_non_latent_model(self) -> None:
        """Test error when model is not LatentVariableModel."""
        model = Mock(spec=EnergyBasedModel)
        evaluator = ModelEvaluator(model)

        data = torch.randn(10, 10)

        with pytest.raises(
            TypeError, match="Reconstruction requires LatentVariableModel"
        ):
            evaluator.reconstruction_error(data)

    def test_log_likelihood_with_partition(self) -> None:
        """Test log-likelihood with known partition function."""
        model = MockModel()
        evaluator = ModelEvaluator(model)

        data = torch.randn(5, 10)
        log_z = 10.0

        log_prob, std_err = evaluator.log_likelihood(data, log_z=log_z)

        assert log_prob.shape == (5,)
        assert std_err is None

        # Check computation
        expected = -model.free_energy(data) - log_z
        assert torch.allclose(log_prob, expected)

    def test_log_likelihood_importance_sampling(self) -> None:
        """Test log-likelihood estimation with importance sampling."""
        model = MockModel()
        evaluator = ModelEvaluator(model)

        data = torch.randn(5, 10)

        log_prob, std_err = evaluator.log_likelihood(data, num_samples=50)

        assert log_prob.shape == (5,)
        assert std_err is not None
        assert std_err.shape == ()

    def test_energy_gap(self) -> None:
        """Test energy gap computation."""
        model = MockModel()
        evaluator = ModelEvaluator(model)

        data = torch.randn(20, 10)

        gap_stats = evaluator.energy_gap(data, num_model_samples=30)

        assert "data_energy_mean" in gap_stats
        assert "data_energy_std" in gap_stats
        assert "model_energy_mean" in gap_stats
        assert "model_energy_std" in gap_stats
        assert "energy_gap" in gap_stats

        # Energy gap should be positive for good models
        # (model samples have higher energy than data)
        assert isinstance(gap_stats["energy_gap"], float)

    def test_sample_quality_metrics(self) -> None:
        """Test sample quality metric computation."""
        model = MockModel()
        evaluator = ModelEvaluator(model)

        real_data = torch.randn(100, 10)
        generated_data = torch.randn(100, 10)

        metrics = evaluator.sample_quality_metrics(real_data, generated_data)

        assert "mean_error" in metrics
        assert "std_error" in metrics
        assert "corr_error" in metrics
        assert "discriminability" in metrics

        # All metrics should be non-negative
        for value in metrics.values():
            assert value >= 0

        # Discriminability should be in [0, 1]
        assert 0 <= metrics["discriminability"] <= 1

    def test_sample_quality_high_dim(self) -> None:
        """Test sample quality metrics with high dimensional data."""
        model = MockModel()
        evaluator = ModelEvaluator(model)

        # High dimensional data (correlation matrix would be too large)
        real_data = torch.randn(50, 200)
        generated_data = torch.randn(50, 200)

        metrics = evaluator.sample_quality_metrics(real_data, generated_data)

        # Should still compute basic metrics
        assert "mean_error" in metrics
        assert "std_error" in metrics
        # Correlation might be skipped for efficiency
        assert "discriminability" in metrics


class TestTrainingDynamicsAnalyzer:
    """Test TrainingDynamicsAnalyzer class."""

    def test_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = TrainingDynamicsAnalyzer(window_size=50)

        assert analyzer.window_size == 50
        assert len(analyzer.history) == 0

    def test_update(self) -> None:
        """Test updating with metrics."""
        analyzer = TrainingDynamicsAnalyzer()

        analyzer.update({"loss": 1.0, "accuracy": 0.8})
        analyzer.update({"loss": 0.9, "accuracy": 0.82})

        assert len(analyzer.history["loss"]) == 2
        assert len(analyzer.history["accuracy"]) == 2
        assert analyzer.history["loss"] == [1.0, 0.9]

    def test_convergence_rate(self) -> None:
        """Test convergence rate estimation."""
        analyzer = TrainingDynamicsAnalyzer(window_size=20)

        # Exponential decay
        for i in range(20):
            loss = 1.0 * np.exp(-0.1 * i)
            analyzer.update({"loss": loss})

        rate = analyzer.convergence_rate("loss")

        assert rate is not None
        assert rate < 0  # Negative rate indicates convergence
        assert abs(rate - (-0.1)) < 0.01  # Should be close to -0.1

    def test_convergence_rate_insufficient_data(self) -> None:
        """Test convergence rate with insufficient data."""
        analyzer = TrainingDynamicsAnalyzer()

        # Too few points
        analyzer.update({"loss": 1.0})
        analyzer.update({"loss": 0.9})

        rate = analyzer.convergence_rate("loss")
        assert rate is None

    def test_convergence_rate_non_positive(self) -> None:
        """Test convergence rate with non-positive values."""
        analyzer = TrainingDynamicsAnalyzer()

        # Include zero/negative values
        for i in range(15):
            analyzer.update({"loss": 1.0 - 0.2 * i})  # Goes negative

        rate = analyzer.convergence_rate("loss")
        assert rate is None

    def test_oscillation_score(self) -> None:
        """Test oscillation detection."""
        analyzer = TrainingDynamicsAnalyzer()

        # Oscillating values
        for i in range(20):
            loss = 1.0 + 0.1 * ((-1) ** i)
            analyzer.update({"loss": loss})

        score = analyzer.oscillation_score("loss")

        assert score is not None
        assert 0 <= score <= 1
        assert score > 0.8  # High oscillation

        # Non-oscillating values
        analyzer2 = TrainingDynamicsAnalyzer()
        for i in range(20):
            analyzer2.update({"accuracy": 0.8 + 0.01 * i})

        score2 = analyzer2.oscillation_score("accuracy")
        assert score2 < 0.2  # Low oscillation

    def test_plateau_detection(self) -> None:
        """Test plateau detection."""
        analyzer = TrainingDynamicsAnalyzer()

        # Initial descent
        for i in range(10):
            analyzer.update({"loss": 1.0 - 0.1 * i})

        # Plateau
        for _ in range(15):
            analyzer.update({"loss": 0.1 + np.random.randn() * 0.0001})

        is_plateau, steps = analyzer.plateau_detection("loss", threshold=0.001)

        assert is_plateau is True
        assert steps >= 10
        assert steps <= 15

    def test_plateau_detection_no_plateau(self) -> None:
        """Test plateau detection when no plateau exists."""
        analyzer = TrainingDynamicsAnalyzer()

        # Continuous improvement
        for i in range(20):
            analyzer.update({"loss": 1.0 - 0.05 * i})

        is_plateau, steps = analyzer.plateau_detection("loss", threshold=0.01)

        assert is_plateau is False
        assert steps is None

    def test_get_summary(self) -> None:
        """Test getting training dynamics summary."""
        analyzer = TrainingDynamicsAnalyzer()

        # Add some data with different patterns
        for i in range(30):
            # Converging loss
            loss = 1.0 * np.exp(-0.1 * i) + np.random.randn() * 0.01
            # Oscillating accuracy
            acc = 0.8 + 0.1 * ((-1) ** i) * np.exp(-0.05 * i)
            # Plateaued metric
            plateau = 0.5 + np.random.randn() * 0.0001

            analyzer.update({"loss": loss, "accuracy": acc, "plateau_metric": plateau})

        summary = analyzer.get_summary()

        assert "loss" in summary
        assert "accuracy" in summary
        assert "plateau_metric" in summary

        # Check loss summary
        loss_summary = summary["loss"]
        assert loss_summary["final_value"] < 0.1
        assert loss_summary["convergence_rate"] < 0
        assert loss_summary["oscillation_score"] < 0.2
        assert loss_summary["is_plateau"] is False
        assert loss_summary["total_steps"] == 30

        # Check accuracy summary
        acc_summary = summary["accuracy"]
        assert acc_summary["oscillation_score"] > 0.5

        # Check plateau metric summary
        plateau_summary = summary["plateau_metric"]
        assert plateau_summary["is_plateau"] is True
