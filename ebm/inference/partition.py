"""Partition function estimation for energy-based models.

This module provides methods for estimating the intractable partition
function (normalization constant) of energy-based models, which is
crucial for computing exact likelihoods.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from ebm.core.logging_utils import LoggerMixin
from ebm.models.base import EnergyBasedModel, LatentVariableModel


class PartitionFunctionEstimator(nn.Module, LoggerMixin):
    """Base class for partition function estimation."""

    def __init__(self, model: EnergyBasedModel):
        """Initialize estimator.

        Args:
            model: Energy-based model
        """
        nn.Module.__init__(self)
        LoggerMixin.__init__(self)
        self.model = model

    def estimate(self, **kwargs: Any) -> float | tuple[float, float, float]:
        """Estimate log partition function.

        Returns
        -------
            Log Z estimate (and optionally confidence bounds)
        """
        raise NotImplementedError


class AISEstimator(PartitionFunctionEstimator):
    """Annealed Importance Sampling estimator for partition function.

    This is the gold standard method for estimating partition functions
    in RBMs and similar models.
    """

    def __init__(
        self,
        model: EnergyBasedModel,
        num_temps: int = 10000,
        num_chains: int = 100,
    ):
        """Initialize AIS estimator.

        Args:
            model: Energy-based model (must support AIS)
            num_temps: Number of intermediate temperatures
            num_chains: Number of independent AIS runs
        """
        super().__init__(model)
        self.num_temps = num_temps
        self.num_chains = num_chains

        # Determine device for temperature schedule
        device = getattr(model, "device", torch.device("cpu"))
        if not isinstance(device, torch.device):
            try:
                device = torch.device(device)
            except (TypeError, ValueError, RuntimeError):
                device = torch.device("cpu")

        # Create temperature schedule
        self.betas = torch.linspace(0, 1, num_temps, device=device)

    def estimate(  # noqa: C901 - complexity OK for tests
        self,
        base_log_z: float | None = None,
        return_diagnostics: bool = False,
        show_progress: bool = True,
    ) -> float | tuple[float, dict[str, Any]]:
        """Estimate log partition function using AIS.

        Args:
            base_log_z: Log partition of base distribution (auto-computed if None)
            return_diagnostics: Whether to return diagnostic information
            show_progress: Whether to show progress bar

        Returns
        -------
            Log partition estimate (and diagnostics if requested)
        """
        if not isinstance(self.model, LatentVariableModel):
            raise TypeError("AIS requires a LatentVariableModel")

        if (
            getattr(self.model, "num_visible", 0) == 0
            or getattr(self.model, "num_hidden", 0) == 0
        ):
            raise ValueError("Model dimensions must be positive")

        device = self.model.device

        # Get base partition function
        if base_log_z is None:
            if hasattr(self.model, "ais_adapter"):
                adapter = self.model.ais_adapter()
                base_log_z = adapter.base_log_partition()
            else:
                # Default: assume uniform distribution
                base_log_z = self.model.num_visible * math.log(2)

        self.log_debug(f"Base log Z: {base_log_z:.2f}")

        # Initialize chains at base distribution
        if hasattr(self.model, "num_hidden"):
            # For RBMs
            h_init = torch.randint(
                2,
                (self.num_chains, self.model.num_hidden),
                device=device,
                dtype=self.model.dtype,
            )
            v = self.model.sample_visible(h_init, beta=torch.tensor(0.0))
        else:
            # Generic initialization
            v = torch.rand(
                self.num_chains,
                self.model.num_visible,
                device=device,
                dtype=self.model.dtype,
            ).round()

        # Run AIS
        log_weights = torch.zeros(self.num_chains, device=device)
        log_z_k = []  # Track evolution of estimate

        pbar = tqdm(self.betas, desc="AIS", disable=not show_progress)

        for i, beta in enumerate(pbar):
            # Update importance weights
            if i > 0:
                prev_beta = self.betas[i - 1]
                delta_beta = beta - prev_beta

                # Compute energy difference
                with torch.no_grad():
                    free_energy = self.model.free_energy(v)
                    log_weights += delta_beta * free_energy

            # Gibbs sampling at current temperature
            if beta > 0:  # Skip at beta=0 (base distribution)
                with torch.no_grad():
                    h = self.model.sample_hidden(v, beta=beta)
                    v = self.model.sample_visible(h, beta=beta)

            # Track intermediate estimates
            if i % 100 == 0:
                log_z_est = (
                    base_log_z
                    + torch.logsumexp(log_weights, 0)
                    - math.log(self.num_chains)
                )
                log_z_k.append(log_z_est.item())
                pbar.set_postfix({"log_Z": f"{log_z_est.item():.2f}"})

        # Final estimate
        log_z = (
            base_log_z
            + torch.logsumexp(log_weights, 0)
            - math.log(self.num_chains)
        )
        log_z = log_z.item()

        if return_diagnostics:
            # Compute diagnostics
            weights = torch.exp(log_weights - log_weights.max())
            weights = weights / weights.sum()

            # Effective sample size
            ess = 1.0 / (weights**2).sum().item()

            # Standard error estimate
            se = torch.sqrt(weights.var() / ess).item()

            diagnostics = {
                "log_Z": log_z,
                "log_Z_std": se,
                "log_Z_trajectory": log_z_k,
                "effective_sample_size": ess,
                "log_weights": log_weights.cpu().numpy(),
                "final_weights": weights.cpu().numpy(),
            }

            self.log_info(
                "AIS completed",
                log_Z=f"{log_z:.2f}",
                std_error=f"{se:.3f}",
                ESS=f"{ess:.1f}",
            )

            return log_z, diagnostics

        return log_z


class BridgeSampling(PartitionFunctionEstimator):
    """Bridge sampling estimator for comparing two models.

    This estimates the ratio of partition functions between two models,
    which is useful for model comparison.
    """

    def __init__(
        self,
        model1: EnergyBasedModel,
        model2: EnergyBasedModel,
        num_samples: int = 10000,
    ):
        """Initialize bridge sampling estimator.

        Args:
            model1: First model
            model2: Second model
            num_samples: Number of samples from each model
        """
        super().__init__(model1)
        self.model2 = model2
        self.num_samples = num_samples

    def estimate(
        self, tol: float = 1e-6, max_iter: int = 1000
    ) -> tuple[float, float]:
        """Estimate log ratio of partition functions.

        Args:
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns
        -------
            (log(Z2/Z1), standard error)
        """
        # Generate samples from both models
        self.log_info("Generating samples from model 1...")
        samples1 = self._generate_samples(self.model, self.num_samples)

        self.log_info("Generating samples from model 2...")
        samples2 = self._generate_samples(self.model2, self.num_samples)

        # Compute energies under both models
        with torch.no_grad():
            # Energies of samples1 under both models
            e11 = self.model.free_energy(samples1)
            e12 = self.model2.free_energy(samples1)

            # Energies of samples2 under both models
            e21 = self.model.free_energy(samples2)
            e22 = self.model2.free_energy(samples2)

        # Bridge sampling iteration
        log_r = torch.tensor(0.0)  # Initial guess

        for i in range(max_iter):
            # Compute bridge function values
            f1 = 1 / (1 + torch.exp(e12 - e11 - log_r))
            f2 = 1 / (1 + torch.exp(e21 - e22 + log_r))

            # Update estimate
            log_r_new = torch.log(f2.mean() / f1.mean())

            # Check convergence
            if torch.abs(log_r_new - log_r) < tol:
                log_r = log_r_new
                self.log_debug(f"Bridge sampling converged at iteration {i}")
                break

            log_r = log_r_new
        else:
            self.log_debug(
                f"Bridge sampling converged at iteration {max_iter}",
            )

        # Estimate standard error using delta method
        var_f1 = f1.var() / (f1.mean() ** 2 * len(f1))
        var_f2 = f2.var() / (f2.mean() ** 2 * len(f2))
        se = torch.sqrt(var_f1 + var_f2).item()

        return log_r.item(), se

    def _generate_samples(
        self, model: EnergyBasedModel, num_samples: int
    ) -> Tensor:
        """Generate samples from a model."""
        if hasattr(model, "sample_fantasy_particles"):
            return model.sample_fantasy_particles(
                num_samples=num_samples, num_steps=10000
            )
        raise NotImplementedError(
            "Model must implement sample_fantasy_particles"
        )


class SimpleIS(PartitionFunctionEstimator):
    """Simple importance sampling estimator.

    This is less accurate than AIS but can be useful for quick estimates
    or when AIS is too expensive.
    """

    def __init__(
        self,
        model: EnergyBasedModel,
        proposal: str = "uniform",
        num_samples: int = 10000,
    ):
        """Initialize simple IS estimator.

        Args:
            model: Energy model
            proposal: Proposal distribution ('uniform' or 'data')
            num_samples: Number of importance samples
        """
        super().__init__(model)
        self.proposal = proposal
        self.num_samples = num_samples

    def estimate(
        self, data_loader: torch.utils.data.DataLoader | None = None
    ) -> tuple[float, float]:
        """Estimate log partition function.

        Args:
            data_loader: Data loader for data-based proposal

        Returns
        -------
            (log Z estimate, standard error)
        """
        device = self.model.device

        if self.proposal == "uniform":
            # Uniform proposal
            samples = torch.rand(
                self.num_samples, self.model.num_visible, device=device
            ).round()
            log_q = -self.model.num_visible * math.log(2)  # log(1/2^d)

        elif self.proposal == "data" and data_loader is not None:
            # Use data distribution as proposal
            samples = []
            for batch_data in data_loader:
                data_tensor = (
                    batch_data[0]
                    if isinstance(batch_data, list | tuple)
                    else batch_data
                )
                samples.append(data_tensor)
                if len(samples) * data_tensor.shape[0] >= self.num_samples:
                    break

            samples = torch.cat(samples, 0)[: self.num_samples].to(device)

            # Estimate data entropy (rough approximation)
            data_mean = samples.mean(0).clamp(1e-6, 1 - 1e-6)
            entropy = -(
                data_mean * torch.log(data_mean)
                + (1 - data_mean) * torch.log(1 - data_mean)
            ).sum()
            log_q = -entropy

        else:
            raise ValueError(f"Unknown proposal: {self.proposal}")

        # Compute importance weights
        with torch.no_grad():
            log_p_unnorm = -self.model.free_energy(samples)
            log_weights = log_p_unnorm - log_q

        # Estimate partition function
        log_z = torch.logsumexp(log_weights, 0) - math.log(self.num_samples)

        # Estimate standard error
        weights = torch.exp(log_weights - log_weights.max())
        weights = weights / weights.sum()
        ess = 1.0 / (weights**2).sum()
        se = torch.sqrt(weights.var() / ess).item()

        return log_z.item(), se


class RatioEstimator(PartitionFunctionEstimator):
    """Estimates ratio of partition functions for model selection.

    This is useful for comparing models or computing Bayes factors.
    """

    def __init__(self, models: list[EnergyBasedModel], method: str = "bridge"):
        """Initialize ratio estimator.

        Args:
            models: List of models to compare
            method: Estimation method ('bridge' or 'thermodynamic')
        """
        super().__init__(models[0])
        self.models = models
        if method not in {"bridge", "thermodynamic"}:
            raise ValueError(f"invalid method: {method}")
        self.method = method

    def estimate_all_ratios(
        self, reference_idx: int = 0, **kwargs: Any
    ) -> dict[tuple[int, int], tuple[float, float]]:
        """Estimate all pairwise log ratios.

        Args:
            reference_idx: Index of reference model
            **kwargs: Arguments for estimation method

        Returns
        -------
            Dictionary mapping (i,j) to (log(Zi/Zj), std_error)
        """
        n_models = len(self.models)
        ratios = {}

        # Estimate ratios with respect to reference
        for i in range(n_models):
            if i == reference_idx:
                ratios[(i, i)] = (0.0, 0.0)
                continue

            if self.method == "bridge":
                estimator = BridgeSampling(
                    self.models[reference_idx], self.models[i]
                )
                log_ratio, se = estimator.estimate(**kwargs)
                ratios[(i, reference_idx)] = (-log_ratio, se)
                ratios[(reference_idx, i)] = (log_ratio, se)

        # Use transitivity to get other ratios
        for i in range(n_models):
            for j in range(n_models):
                if (
                    (i, j) not in ratios
                    and i != j
                    and (i, reference_idx) in ratios
                    and (reference_idx, j) in ratios
                ):
                    log_ir, se_ir = ratios[(i, reference_idx)]
                    log_rj, se_rj = ratios[(reference_idx, j)]
                    ratios[(i, j)] = (
                        log_ir + log_rj,
                        (se_ir**2 + se_rj**2) ** 0.5,
                    )

        return ratios
