"""Advanced MCMC sampling algorithms including Parallel Tempering.

This module implements sophisticated MCMC methods that improve mixing
and exploration of the energy landscape.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ebm.core.registry import register_sampler
from ebm.models.base import EnergyBasedModel, LatentVariableModel
from ebm.utils.tensor import log_sum_exp

from .base import AnnealedSampler, GradientEstimator

PROB_THRESHOLD = 0.5


@register_sampler("parallel_tempering", aliases=["pt", "replica_exchange"])
class ParallelTempering(AnnealedSampler):
    """Parallel Tempering (Replica Exchange) sampler.

    PT runs multiple chains at different temperatures and exchanges
    states between adjacent temperature levels to improve mixing.
    """

    def __init__(
        self,
        num_temps: int = 10,
        min_beta: float = 0.1,
        max_beta: float = 1.0,
        swap_every: int = 1,
        num_chains: int | None = None,
        adaptive: bool = False,
    ):
        """Initialize Parallel Tempering sampler.

        Args:
            num_temps: Number of temperature levels
            min_beta: Minimum inverse temperature
            max_beta: Maximum inverse temperature
            swap_every: Frequency of swap attempts
            num_chains: Number of independent chains
            adaptive: Whether to adapt temperatures
        """
        super().__init__(
            name="ParallelTempering",
            num_temps=num_temps,
            min_beta=min_beta,
            max_beta=max_beta,
        )
        self.swap_every = swap_every
        self.num_chains = num_chains
        self.adaptive = adaptive

        # Initialize swap statistics
        self.register_buffer("swap_attempts", torch.zeros(num_temps - 1))
        self.register_buffer("swap_accepts", torch.zeros(num_temps - 1))

        # Chains have shape (num_chains, num_temps, *state_shape)
        self.register_buffer("chains", None)

        # Track which chain is at which temperature
        self.register_buffer("chain_temps", None)

    def init_chains(
        self,
        model: LatentVariableModel,
        batch_size: int,
        state_shape: tuple[int, ...],
    ) -> None:
        """Initialize chains at all temperatures.

        Args:
            model: Energy model
            batch_size: Number of chains
            state_shape: Shape of each state
        """
        num_chains = self.num_chains or batch_size
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Initialize visible states randomly
        v_init = torch.rand(
            num_chains, self.num_temps, *state_shape, device=device, dtype=dtype
        )

        # For binary units, make them actually binary
        if hasattr(model, "_sample_from_prob"):
            v_init = (v_init > PROB_THRESHOLD).to(dtype)

        self.chains = v_init
        self.chain_temps = (
            torch.arange(self.num_temps, device=device)
            .unsqueeze(0)
            .expand(num_chains, -1)
        )

        # Run a few Gibbs steps at each temperature to equilibrate
        with torch.no_grad():
            for t in range(self.num_temps):
                beta = self.betas[t]
                v_t = self.chains[:, t]

                for _ in range(10):  # Equilibration steps
                    h_t = model.sample_hidden(v_t, beta=beta)
                    v_t = model.sample_visible(h_t, beta=beta)

                self.chains[:, t] = v_t

    def sample(
        self,
        model: EnergyBasedModel,
        init_state: Tensor,
        num_steps: int = 1,
        **_kwargs: Any,
    ) -> Tensor:
        """Run parallel tempering sampling.

        Args:
            model: Energy model (must be LatentVariableModel)
            init_state: Initial state (used for shape/device info)
            num_steps: Number of PT steps
            **kwargs: Additional arguments

        Returns
        -------
            Samples from the target distribution (beta=1)
        """
        if not isinstance(model, LatentVariableModel):
            raise TypeError("PT requires a LatentVariableModel")

        # Initialize chains if needed
        if self.chains is None:
            self.init_chains(model, init_state.shape[0], init_state.shape[1:])

        # Run PT steps
        for step in range(num_steps):
            # Gibbs sampling at each temperature
            self._gibbs_step_all_temps(model)

            # Attempt swaps
            if step % self.swap_every == 0:
                self._attempt_swaps(model)

        self.state.num_steps += num_steps

        # Return samples from target temperature (beta=1)
        # Find which chains are at target temp
        target_idx = self.chain_temps == self.num_temps - 1
        return self.chains[target_idx][: init_state.shape[0]]

    def _gibbs_step_all_temps(self, model: LatentVariableModel) -> None:
        """Run one Gibbs step at all temperatures in parallel."""
        batch_size, num_temps = self.chains.shape[:2]

        # Reshape for batch processing
        v_all = self.chains.reshape(batch_size * num_temps, -1)

        # Create beta vector for all chains
        beta_all = self.betas[self.chain_temps.reshape(-1)]

        # Gibbs step
        h_all = model.sample_hidden(v_all, beta=beta_all)
        v_all = model.sample_visible(h_all, beta=beta_all)

        # Reshape back
        self.chains = v_all.reshape(batch_size, num_temps, -1)

    def _attempt_swaps(self, model: EnergyBasedModel) -> None:
        """Attempt swaps between adjacent temperature levels."""
        batch_size = self.chains.shape[0]

        # Randomly choose even or odd pairs
        if torch.rand(1).item() < PROB_THRESHOLD:
            # Even pairs: (0,1), (2,3), ...
            pairs = torch.arange(0, self.num_temps - 1, 2)
        else:
            # Odd pairs: (1,2), (3,4), ...
            pairs = torch.arange(1, self.num_temps - 1, 2)

        for i in pairs:
            if i + 1 >= self.num_temps:
                continue

            # Get chains at adjacent temperatures
            chain_i = self.chains[:, i]
            chain_j = self.chains[:, i + 1]

            # Compute energies
            energy_i = model.free_energy(chain_i)
            energy_j = model.free_energy(chain_j)

            # Compute swap acceptance probability
            beta_i = self.betas[i]
            beta_j = self.betas[i + 1]
            delta = (beta_j - beta_i) * (energy_i - energy_j)

            # Accept/reject swaps
            accept = torch.rand(batch_size, device=delta.device) < torch.exp(
                delta
            )

            # Perform swaps
            if accept.any():
                temp = self.chains[:, i].clone()
                self.chains[accept, i] = self.chains[accept, i + 1]
                self.chains[accept, i + 1] = temp[accept]

                # Update temperature assignments
                temp_idx = self.chain_temps[:, i].clone()
                self.chain_temps[accept, i] = self.chain_temps[accept, i + 1]
                self.chain_temps[accept, i + 1] = temp_idx[accept]

            # Update statistics
            self.swap_attempts[i] += batch_size
            self.swap_accepts[i] += accept.sum()

    @property
    def swap_rates(self) -> Tensor:
        """Get acceptance rates for each temperature pair."""
        return self.swap_accepts / (self.swap_attempts + 1e-8)

    def adapt_temperatures(self, target_rate: float = 0.3) -> None:
        """Adapt temperature schedule based on swap rates.

        Args:
            target_rate: Target swap acceptance rate
        """
        if not self.adaptive:
            return

        rates = self.swap_rates

        # Adjust temperatures to achieve target swap rate
        # If rate too low, move temperatures closer
        # If rate too high, move temperatures apart
        with torch.no_grad():
            for i in range(self.num_temps - 1):
                if rates[i] < target_rate - 0.1:
                    # Decrease temperature gap
                    factor = 0.95
                elif rates[i] > target_rate + 0.1:
                    # Increase temperature gap
                    factor = 1.05
                else:
                    continue

                # Adjust beta while maintaining order
                min_beta = self.betas[i - 1] * 1.01 if i > 0 else self.min_beta

                if i < self.num_temps - 2:
                    max_beta = self.betas[i + 2] * 0.99
                else:
                    max_beta = self.max_beta

                new_beta = self.betas[i] * factor
                self.betas[i] = torch.clamp(new_beta, min_beta, max_beta)

        self.log_debug(f"Adapted temperatures, swap rates: {rates}")


class PTGradientEstimator(GradientEstimator):
    """Gradient estimator using Parallel Tempering."""

    def __init__(
        self,
        num_temps: int = 10,
        k: int = 1,
        swap_every: int = 1,
        **pt_kwargs: Any,
    ):
        """Initialize PT gradient estimator.

        Args:
            num_temps: Number of temperatures
            k: Gibbs steps between gradient updates
            swap_every: Swap frequency
            **pt_kwargs: Additional PT arguments
        """
        sampler = ParallelTempering(
            num_temps=num_temps, swap_every=swap_every, **pt_kwargs
        )
        super().__init__(sampler)
        self.k = k

    def estimate_gradient(
        self, model: EnergyBasedModel, data: Tensor, **_kwargs: Any
    ) -> dict[str, Tensor]:
        """Estimate gradients using PT samples.

        Args:
            model: Energy model
            data: Training data
            **kwargs: Additional arguments

        Returns
        -------
            Parameter gradients
        """
        if not isinstance(model, LatentVariableModel):
            raise TypeError(
                "PT gradient estimation requires LatentVariableModel"
            )

        # Positive phase
        h_data = model.sample_hidden(data, return_prob=True)[1]

        # Negative phase using PT
        v_model = self.sampler.sample(model, data, num_steps=self.k)
        h_model = model.sample_hidden(v_model, return_prob=True)[1]

        # Compute gradients
        from ebm.utils.tensor import batch_outer_product

        gradients = {}
        pos_stats = batch_outer_product(h_data, data).mean(dim=0)
        neg_stats = batch_outer_product(h_model, v_model).mean(dim=0)
        gradients["W"] = pos_stats - neg_stats

        if hasattr(model, "vbias") and model.vbias.requires_grad:
            gradients["vbias"] = data.mean(dim=0) - v_model.mean(dim=0)

        if hasattr(model, "hbias") and model.hbias.requires_grad:
            gradients["hbias"] = h_data.mean(dim=0) - h_model.mean(dim=0)

        return gradients


@register_sampler("ais", aliases=["annealed_importance_sampling"])
class AnnealedImportanceSampling(AnnealedSampler):
    """Annealed Importance Sampling for partition function estimation.

    AIS provides an unbiased estimate of the partition function by
    using a sequence of intermediate distributions.
    """

    def __init__(
        self, num_temps: int = 1000, num_chains: int = 100, k: int = 1
    ):
        """Initialize AIS.

        Args:
            num_temps: Number of intermediate distributions
            num_chains: Number of independent AIS runs
            k: Gibbs steps at each temperature
        """
        super().__init__(
            name="AIS", num_temps=num_temps, min_beta=0.0, max_beta=1.0
        )
        self.num_chains = num_chains
        self.k = k

    def estimate_log_partition(
        self,
        model: LatentVariableModel,
        base_log_z: float,
        return_bounds: bool = False,
    ) -> float | tuple[float, float, float]:
        """Estimate log partition function.

        Args:
            model: Energy model with AIS support
            base_log_z: Log partition function of base distribution
            return_bounds: If True, return confidence bounds

        Returns
        -------
            Log partition estimate (and bounds if requested)
        """
        device = next(model.parameters()).device

        # Initialize at base distribution (beta=0)
        # For RBMs, this is typically independent Bernoulli units
        h_init = torch.rand(
            self.num_chains, model.num_hidden, device=device
        ).round()

        # Sample initial visible units from base distribution
        v = model.sample_visible(h_init, beta=0.0)

        # Initialize importance weights
        log_w = torch.zeros(self.num_chains, device=device)

        # Run AIS
        for i in range(self.num_temps):
            beta = self.betas[i]

            # Add contribution to importance weight
            if i > 0:
                prev_beta = self.betas[i - 1]
                log_w += (prev_beta - beta) * model.free_energy(v)

            # Run k Gibbs steps at current temperature
            for _ in range(self.k):
                h = model.sample_hidden(v, beta=beta)
                v = model.sample_visible(h, beta=beta)

        # Final contribution
        log_w += self.betas[-1] * model.free_energy(v)

        # Compute estimate
        log_z_estimate = (
            base_log_z
            + log_sum_exp(log_w)
            - torch.log(torch.tensor(self.num_chains))
        )

        if return_bounds:
            # Compute confidence bounds using empirical variance
            weights = torch.exp(log_w - log_w.max())
            weights = weights / weights.sum()

            # Effective sample size
            ess = 1.0 / (weights**2).sum()

            # Standard error
            se = torch.sqrt(weights.var() / ess)

            # 95% confidence interval
            ci_lower = log_z_estimate - 1.96 * se
            ci_upper = log_z_estimate + 1.96 * se

            return float(log_z_estimate), float(ci_lower), float(ci_upper)

        return float(log_z_estimate)
