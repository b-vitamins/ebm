"""Metrics tracking and evaluation for energy-based models.

This module provides utilities for tracking training metrics,
computing model evaluation metrics, and analyzing training dynamics.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ebm.models.base import EnergyBasedModel, LatentVariableModel


@dataclass
class MetricValue:
    """Container for a metric value with statistics."""

    current: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    count: int = 0

    def update(self, value: float) -> None:
        """Update metric with new value."""
        self.current = value
        self.count += 1

        # Update running statistics
        if self.count == 1:
            self.mean = value
            self.min = value
            self.max = value
        else:
            delta = value - self.mean
            self.mean += delta / self.count
            self.std = np.sqrt(
                ((self.count - 1) * self.std**2 + delta * (value - self.mean))
                / self.count
            )
            self.min = min(self.min, value)
            self.max = max(self.max, value)


class MetricsTracker:
    """Tracks metrics during training."""

    def __init__(self, window_size: int = 100):
        """Initialize metrics tracker.

        Args:
            window_size: Size of sliding window for running averages
        """
        self.window_size = window_size
        self.metrics: dict[str, MetricValue] = defaultdict(MetricValue)
        self.history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def update(self, metrics: dict[str, float]) -> None:
        """Update metrics with new values.

        Args:
            metrics: Dictionary of metric values
        """
        for name, value in metrics.items():
            self.metrics[name].update(value)
            self.history[name].append(value)

    def get_current(self) -> dict[str, float]:
        """Get current metric values."""
        return {name: metric.current for name, metric in self.metrics.items()}

    def get_average(self, window: int | None = None) -> dict[str, float]:
        """Get average metric values over window.

        Args:
            window: Window size (uses full history if None)

        Returns
        -------
            Dictionary of average values
        """
        result = {}
        for name, hist in self.history.items():
            if hist:
                if window is None:
                    result[name] = np.mean(hist)
                else:
                    result[name] = np.mean(list(hist)[-window:])
        return result

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get full statistics for all metrics."""
        stats = {}
        for name, metric in self.metrics.items():
            stats[name] = {
                "current": metric.current,
                "mean": metric.mean,
                "std": metric.std,
                "min": metric.min,
                "max": metric.max,
                "count": metric.count,
            }
        return stats

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.history.clear()

    def compute(self) -> dict[str, float]:
        """Compute final metric values (averages)."""
        return self.get_average()


class ModelEvaluator:
    """Evaluates energy-based models on various metrics."""

    def __init__(self, model: EnergyBasedModel):
        """Initialize evaluator.

        Args:
            model: Model to evaluate
        """
        self.model = model

    @torch.no_grad()
    def reconstruction_error(
        self, data: Tensor, num_steps: int = 1, metric: str = "mse"
    ) -> Tensor:
        """Compute reconstruction error.

        Args:
            data: Input data
            num_steps: Number of reconstruction steps
            metric: Error metric ('mse' or 'mae')

        Returns
        -------
            Per-sample reconstruction errors
        """
        if not isinstance(self.model, LatentVariableModel):
            raise TypeError("Reconstruction requires LatentVariableModel")

        # Reconstruct data
        recon = self.model.reconstruct(data, num_steps=num_steps)

        # Compute error
        if metric == "mse":
            error = (data - recon).pow(2).mean(dim=tuple(range(1, data.dim())))
        elif metric == "mae":
            error = (data - recon).abs().mean(dim=tuple(range(1, data.dim())))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return error

    @torch.no_grad()
    def log_likelihood(
        self, data: Tensor, log_z: float | None = None, num_samples: int = 100
    ) -> tuple[Tensor, Tensor | None]:
        """Estimate log-likelihood of data.

        Args:
            data: Input data
            log_z: Log partition function (if known)
            num_samples: Number of importance samples (if log_z unknown)

        Returns
        -------
            Log-likelihood values and optional standard errors
        """
        # If log_z is provided, exact computation
        if log_z is not None:
            log_prob = self.model.log_probability(data, log_z=log_z)
            return log_prob, None

        # Otherwise, use importance sampling estimate
        # This is a simplified version - proper implementation would use AIS
        free_energy = self.model.free_energy(data)

        # Generate importance samples
        if hasattr(self.model, "sample_fantasy_particles"):
            samples = self.model.sample_fantasy_particles(
                num_samples=num_samples, num_steps=1000
            )
            sample_energies = self.model.free_energy(samples)

            # Simple importance sampling estimate
            log_z_estimate = torch.logsumexp(-sample_energies, dim=0) - np.log(
                num_samples
            )
            log_prob = -free_energy - log_z_estimate

            # Estimate standard error
            weights = torch.exp(-sample_energies - sample_energies.max())
            weights = weights / weights.sum()
            se = torch.sqrt(weights.var() / (1 / (weights**2).sum()))

            return log_prob, se
        # Fallback: just return negative free energy
        return -free_energy, None

    @torch.no_grad()
    def energy_gap(
        self, data: Tensor, num_model_samples: int = 100
    ) -> dict[str, float]:
        """Compute energy gap between data and model samples.

        Args:
            data: Real data samples
            num_model_samples: Number of model samples to generate

        Returns
        -------
            Dictionary with energy statistics
        """
        # Data energy
        data_energy = self.model.free_energy(data)

        # Model energy
        if hasattr(self.model, "sample_fantasy_particles"):
            model_samples = self.model.sample_fantasy_particles(
                num_samples=num_model_samples, num_steps=1000
            )
            model_energy = self.model.free_energy(model_samples)
        else:
            # Fallback for models without fantasy sampling
            model_energy = data_energy  # Placeholder

        return {
            "data_energy_mean": data_energy.mean().item(),
            "data_energy_std": data_energy.std().item(),
            "model_energy_mean": model_energy.mean().item(),
            "model_energy_std": model_energy.std().item(),
            "energy_gap": (model_energy.mean() - data_energy.mean()).item(),
        }

    @torch.no_grad()
    def sample_quality_metrics(
        self, real_data: Tensor, generated_data: Tensor
    ) -> dict[str, float]:
        """Compute sample quality metrics.

        Args:
            real_data: Real data samples
            generated_data: Generated samples

        Returns
        -------
            Dictionary of quality metrics
        """
        metrics = {}

        # Basic statistics comparison
        real_mean = real_data.mean(dim=0)
        gen_mean = generated_data.mean(dim=0)
        real_std = real_data.std(dim=0)
        gen_std = generated_data.std(dim=0)

        metrics["mean_error"] = (real_mean - gen_mean).abs().mean().item()
        metrics["std_error"] = (real_std - gen_std).abs().mean().item()

        # Correlation comparison
        if real_data.dim() == 2:
            real_corr = torch.corrcoef(real_data.T)
            gen_corr = torch.corrcoef(generated_data.T)
            metrics["corr_error"] = (real_corr - gen_corr).abs().mean().item()

        # Simple discriminability test
        # Train a linear classifier to distinguish real from generated
        from torch.nn import functional as F  # noqa: N812

        labels = torch.cat(
            [
                torch.ones(real_data.shape[0]),
                torch.zeros(generated_data.shape[0]),
            ]
        )
        data = torch.cat([real_data, generated_data])

        # Shuffle
        perm = torch.randperm(labels.shape[0])
        data = data[perm]
        labels = labels[perm]

        # Simple logistic regression
        w = torch.zeros(data.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([w, b], lr=0.1, max_iter=100)

        def closure():
            optimizer.zero_grad()
            logits = data @ w + b
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Compute accuracy
        with torch.no_grad():
            logits = data @ w + b
            preds = (logits > 0).float()
            accuracy = (preds == labels).float().mean().item()

        # If samples are good, accuracy should be close to 0.5 (random)
        metrics["discriminability"] = abs(accuracy - 0.5) * 2

        return metrics


class TrainingDynamicsAnalyzer:
    """Analyzes training dynamics and convergence."""

    def __init__(self, window_size: int = 100):
        """Initialize analyzer.

        Args:
            window_size: Window for computing statistics
        """
        self.window_size = window_size
        self.history: dict[str, list[float]] = defaultdict(list)

    def update(self, metrics: dict[str, float]) -> None:
        """Update with new metrics."""
        for name, value in metrics.items():
            self.history[name].append(value)

    def convergence_rate(self, metric: str) -> float | None:
        """Estimate convergence rate of a metric.

        Args:
            metric: Metric name

        Returns
        -------
            Convergence rate (negative = converging)
        """
        if metric not in self.history or len(self.history[metric]) < 10:
            return None

        values = self.history[metric][-self.window_size :]

        # Fit exponential decay: y = a * exp(b * x)
        # log(y) = log(a) + b * x
        x = np.arange(len(values))
        y = np.array(values)

        # Only fit if values are positive
        if np.all(y > 0):
            log_y = np.log(y)
            # Linear regression in log space
            design = np.vstack([x, np.ones(len(x))]).T
            b, log_a = np.linalg.lstsq(design, log_y, rcond=None)[0]
            return b

        return None

    def oscillation_score(self, metric: str) -> float | None:
        """Compute oscillation score for a metric.

        Args:
            metric: Metric name

        Returns
        -------
            Oscillation score (0 = no oscillation, 1 = high oscillation)
        """
        if metric not in self.history or len(self.history[metric]) < 10:
            return None

        values = np.array(self.history[metric][-self.window_size :])

        # Compute successive differences
        diffs = np.diff(values)

        # Count sign changes
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

        # Normalize by number of possible sign changes
        max_changes = len(diffs) - 1
        if max_changes > 0:
            return sign_changes / max_changes

        return 0.0

    def plateau_detection(
        self, metric: str, threshold: float = 1e-4
    ) -> tuple[bool, int | None]:
        """Detect if metric has plateaued.

        Args:
            metric: Metric name
            threshold: Change threshold for plateau detection

        Returns
        -------
            (is_plateau, steps_in_plateau)
        """
        if metric not in self.history or len(self.history[metric]) < 10:
            return False, None

        values = self.history[metric][-self.window_size :]

        # Check variance in recent values
        recent_std = (
            np.std(values[-20:]) if len(values) >= 20 else np.std(values)
        )

        if recent_std < threshold:
            # Find when plateau started
            for i in range(len(values) - 1, -1, -1):
                if abs(values[i] - values[-1]) > threshold * 10:
                    return True, len(values) - i - 1
            return True, len(values)

        return False, None

    def get_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of training dynamics."""
        summary = {}

        for metric in self.history:
            summary[metric] = {
                "final_value": self.history[metric][-1]
                if self.history[metric]
                else None,
                "convergence_rate": self.convergence_rate(metric),
                "oscillation_score": self.oscillation_score(metric),
                "is_plateau": self.plateau_detection(metric)[0],
                "total_steps": len(self.history[metric]),
            }

        return summary
