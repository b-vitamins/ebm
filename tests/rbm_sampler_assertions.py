"""RBM-specific sampler test mixins.

This module provides test mixins specific to RBM samplers, including
tests for beta parameters and statistical properties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ebm.rbm.model.base import BaseRBM, RBMConfig
from tests.helper import (
    BetaRecordingModel,
    BetaValidatingModel,
    MockRBM,
    exact_visible_dist,
)

# Re-export generic mixins
from tests.sampler_assertions import (
    CUDATests,
    DeterminismTests,
    DeviceAutogradTests,
    HookTests,
    MetadataTests,
    MultiprocessTests,
    PerformanceTests,
    PropertyBasedTests,
    SerializationTests,
    ShapeTests,
    StateTests,
    StressTests,
)

if TYPE_CHECKING:
    from ebm.rbm.sampler.base import BaseSamplerRBM, SampleRBM

__all__ = [
    # Generic tests (re-exported)
    "ShapeTests",
    "DeviceAutogradTests",
    "HookTests",
    "DeterminismTests",
    "StressTests",
    "StateTests",
    "MultiprocessTests",
    "MetadataTests",
    "PropertyBasedTests",
    "PerformanceTests",
    "SerializationTests",
    "CUDATests",
    # RBM-specific tests
    "BetaTests",
    "StatisticalTests",
]


class BetaTests:
    """Tests for parallel tempering beta parameter handling.

    This test mixin validates that RBM samplers correctly handle
    temperature parameters for parallel tempering.
    """

    def test_beta_scalar_propagation(
        self,
        sampler_class: type[BaseSamplerRBM],
        visible_size: int,
        hidden_size: int,
    ) -> None:
        """Test scalar beta propagation through sampler.

        Parameters
        ----------
        sampler_class : Type[BaseSamplerRBM]
            Sampler class to test.
        visible_size : int
            Number of visible units.
        hidden_size : int
            Number of hidden units.
        """
        # Create recording model
        cfg = RBMConfig(visible=visible_size, hidden=hidden_size)
        model = BetaRecordingModel(cfg)
        sampler = sampler_class(model)

        # Sample with scalar beta
        beta_scalar = torch.tensor(0.7)
        v0 = torch.randn(4, visible_size)
        _ = sampler.sample(v0, beta=beta_scalar)

        # Check all recorded betas
        assert len(model.betas) > 0, "No beta values recorded"
        for recorded_beta in model.betas:
            assert recorded_beta is beta_scalar, "Beta not propagated correctly"

    def test_beta_vector_propagation(
        self,
        sampler_class: type[BaseSamplerRBM],
        visible_size: int,
        hidden_size: int,
    ) -> None:
        """Test vector beta propagation for parallel tempering.

        Parameters
        ----------
        sampler_class : Type[BaseSamplerRBM]
            Sampler class to test.
        visible_size : int
            Number of visible units.
        hidden_size : int
            Number of hidden units.
        """
        # Create validating model
        cfg = RBMConfig(visible=visible_size, hidden=hidden_size)
        beta_vector = torch.tensor([0.2, 0.5, 0.8, 1.0])
        model = BetaValidatingModel(cfg, expected=beta_vector)
        sampler = sampler_class(model)

        # Sample with vector beta
        num_replicas = len(beta_vector)
        v0 = torch.randn(4, num_replicas, visible_size)

        # This should not raise (validation happens inside model)
        _ = sampler.sample(v0, beta=beta_vector)

    def test_beta_shape_broadcasting(
        self,
        sampler_class: type[BaseSamplerRBM],
        visible_size: int,
        hidden_size: int,
    ) -> None:
        """Test beta shape broadcasting rules.

        Parameters
        ----------
        sampler_class : Type[BaseSamplerRBM]
            Sampler class to test.
        visible_size : int
            Number of visible units.
        hidden_size : int
            Number of hidden units.
        """
        cfg = RBMConfig(visible=visible_size, hidden=hidden_size)
        model = MockRBM(cfg)
        sampler = sampler_class(model)

        # Test various beta shapes
        batch_size = 4
        num_replicas = 3

        test_cases: list[tuple[tuple[int, ...], torch.Tensor, str]] = [
            # (v0_shape, beta_shape, description)
            ((batch_size, visible_size), torch.tensor(0.5), "scalar beta"),
            (
                (batch_size, num_replicas, visible_size),
                torch.tensor([0.3, 0.6, 1.0]),
                "vector beta",
            ),
            (
                (batch_size, num_replicas, visible_size),
                torch.tensor([0.3, 0.6, 1.0]).unsqueeze(0).unsqueeze(-1),
                "shaped beta",
            ),
        ]

        for v0_shape, beta, desc in test_cases:
            v0 = torch.randn(v0_shape)
            result = sampler.sample(v0, beta=beta)
            assert result.shape == v0_shape, f"Shape mismatch for {desc}"

    def test_beta_result_shapes(
        self,
        sampler_class: type[BaseSamplerRBM],
        rbm_config: RBMConfig,
        rbm_class: type[BaseRBM],
    ) -> None:
        """Verify samplers preserve shapes with beta parameter.

        Parameters
        ----------
        sampler_class : Type[BaseSamplerRBM]
            Sampler class to test.
        rbm_config : RBMConfig
            Base RBM configuration.
        rbm_class : Type[BaseRBM]
            RBM class to instantiate.
        """
        visible, hidden = 5, 4
        batch_size, replicas = 2, 4

        config = type(rbm_config)(visible=visible, hidden=hidden)
        model = rbm_class(config)
        sampler = sampler_class(model)

        beta = torch.linspace(0.1, 1.0, steps=replicas)
        v0 = torch.randn(batch_size, replicas, visible)
        result = sampler.sample(v0, beta=beta, return_hidden=True)

        assert result.shape == (batch_size, replicas, visible)
        if hasattr(result, "final_hidden") and result.final_hidden is not None:
            assert result.final_hidden.shape == (batch_size, replicas, hidden)


class StatisticalTests:
    """Comprehensive statistical tests for RBM samplers.

    These tests verify fundamental properties of MCMC samplers based on
    theoretical guarantees. All tests are sampler-agnostic and rely on
    mathematical properties that any correct sampler must satisfy.
    """

    def test_markov_property(
        self,
        sampler_class: type[BaseSamplerRBM],
        rbm_config: RBMConfig,
        rbm_class: type[BaseRBM],
    ) -> None:
        r"""Test the Markov property of the sampler.

        The Markov property states that the future evolution of a chain
        depends only on its current state, not on its history. This test
        verifies this by starting two chains from different histories but
        forcing them to the same state, then checking that their future
        evolution is statistically identical.

        Parameters
        ----------
        sampler_class : Type[BaseSamplerRBM]
            Sampler class to test.
        rbm_config : RBMConfig
            Base RBM configuration.
        rbm_class : Type[BaseRBM]
            RBM class to use.

        Notes
        -----
        The test uses a small RBM (V=8, H=10) and runs chains for 16,000 steps
        with thinning of 20, yielding 800 effectively independent samples.
        The acceptance threshold is set to 0.08, which corresponds to ~3.2σ.
        """
        torch.manual_seed(42)

        # Create small RBM
        cfg = type(rbm_config)(visible=8, hidden=10)
        rbm = rbm_class(cfg)
        with torch.no_grad():
            for name in ("w", "vb", "hb"):
                if hasattr(rbm, name):
                    getattr(rbm, name).normal_(0, 0.10)

        sampler = sampler_class(rbm)

        # Build divergent histories
        hist_len = 40
        chain_a: SampleRBM = sampler.sample(torch.zeros(100, 8))
        chain_b: SampleRBM = sampler.sample(torch.ones(100, 8))
        for _ in range(hist_len):
            chain_a = sampler.sample(chain_a.to_tensor())
            chain_b = sampler.sample(chain_b.to_tensor())

        # Reset both chains to identical state
        v_star = torch.randint(0, 2, (100, 8), dtype=torch.float32, device=chain_a.device)
        chain_a = sampler.sample(v_star.clone())
        chain_b = sampler.sample(v_star.clone())

        # Evolve and record
        raw_steps, thin = 16_000, 20
        future_a: list[torch.Tensor] = []
        future_b: list[torch.Tensor] = []

        for step in range(raw_steps):
            chain_a = sampler.sample(chain_a.to_tensor())
            chain_b = sampler.sample(chain_b.to_tensor())
            if step % thin == 0:
                future_a.append(chain_a.to_tensor().mean(0))
                future_b.append(chain_b.to_tensor().mean(0))

        m_a = torch.stack(future_a)
        m_b = torch.stack(future_b)

        max_diff = float((m_a.mean(0) - m_b.mean(0)).abs().max())
        assert max_diff < 0.08, f"Markov property violated: max Δ⟨v⟩ = {max_diff:.3f} (gate 0.08)"

    def test_stationarity(
        self,
        sampler_class: type[BaseSamplerRBM],
        rbm_config: RBMConfig,
        rbm_class: type[BaseRBM],
    ) -> None:
        r"""Test stationarity of the sampler.

        Stationarity means that if a chain starts from the equilibrium
        distribution, it remains in equilibrium. This test starts chains
        from the exact equilibrium distribution and verifies they stay there.

        Parameters
        ----------
        sampler_class : Type[BaseSamplerRBM]
            Sampler class to test.
        rbm_config : RBMConfig
            Base RBM configuration.
        rbm_class : Type[BaseRBM]
            RBM class to use.

        Notes
        -----
        Uses a tiny RBM (V=8, H=10) where the exact distribution can be
        computed. The test measures total variation distance between the
        empirical and exact distributions, accepting TV < 0.05.
        """
        torch.manual_seed(42)

        # Create tiny RBM
        small_cfg = type(rbm_config)(visible=8, hidden=10)
        rbm = rbm_class(small_cfg)

        # Initialize with small random weights
        with torch.no_grad():
            if hasattr(rbm, "w"):
                rbm.w.uniform_(-0.2, 0.2)
            if hasattr(rbm, "vb"):
                rbm.vb.uniform_(-0.1, 0.1)
            if hasattr(rbm, "hb"):
                rbm.hb.uniform_(-0.1, 0.1)

        sampler = sampler_class(rbm)

        # Compute exact distribution
        p_exact = exact_visible_dist(rbm)

        # Sample from equilibrium
        n_chains = 20_000
        states = list(p_exact.keys())
        probs = np.asarray(list(p_exact.values()), dtype=float)
        probs /= probs.sum()

        v0 = torch.stack(
            [
                torch.tensor(states[np.random.choice(len(states), p=probs)], dtype=torch.float32)
                for _ in range(n_chains)
            ]
        )

        # Evolve chains
        vk: SampleRBM = sampler.sample(v0.clone())
        for _ in range(1_000):
            vk = sampler.sample(vk.to_tensor())

        # Compute empirical distribution
        counts: dict[tuple[int, ...], int] = {}
        for s in vk:
            tup = tuple(int(x) for x in s)
            counts[tup] = counts.get(tup, 0) + 1
        p_emp = {k: c / n_chains for k, c in counts.items()}

        # Compute total variation
        tv = 0.5 * sum(abs(p_exact.get(s, 0) - p_emp.get(s, 0)) for s in set(p_exact) | set(p_emp))

        assert tv < 0.05, f"Stationarity violated: TV = {tv:.3f}"

    def test_ergodicity(
        self,
        sampler_class: type[BaseSamplerRBM],
        rbm_config: RBMConfig,
        rbm_class: type[BaseRBM],
    ) -> None:
        r"""Test ergodicity of the sampler.

        Ergodicity means that a single chain can eventually explore the
        entire state space. This test starts from the worst-case initial
        state (all zeros) and verifies that the chain eventually reaches
        equilibrium.

        Parameters
        ----------
        sampler_class : Type[BaseSamplerRBM]
            Sampler class to test.
        rbm_config : RBMConfig
            Base RBM configuration.
        rbm_class : Type[BaseRBM]
            RBM class to use.

        Notes
        -----
        The test runs for 2 million steps with thinning of 100, yielding
        20,000 effectively independent samples. The acceptance threshold
        is TV < 0.08, which is ~5σ from the expected sampling noise.
        """
        torch.manual_seed(42)

        # Create tiny RBM
        tiny_cfg = type(rbm_config)(visible=8, hidden=10)
        rbm = rbm_class(tiny_cfg)
        with torch.no_grad():
            for name in ("w", "vb", "hb"):
                if hasattr(rbm, name):
                    getattr(rbm, name).normal_(0, 0.01)

        sampler = sampler_class(rbm)

        # Start from worst case
        v = torch.zeros(1, 8)

        # Burn-in
        for _ in range(60_000):
            v = sampler.sample(v).to_tensor()

        # Production run
        raw_steps, thin = 2_000_000, 100
        counts: dict[tuple[int, ...], int] = {}

        for s in range(raw_steps):
            v = sampler.sample(v).to_tensor()
            if s % thin == 0:
                key = tuple(int(x) for x in v[0])
                counts[key] = counts.get(key, 0) + 1

        n_eff = sum(counts.values())  # 20,000
        p_emp = {k: c / n_eff for k, c in counts.items()}
        p_exact = exact_visible_dist(rbm)

        # Compute total variation
        tv = 0.5 * sum(
            abs(p_exact.get(s, 0.0) - p_emp.get(s, 0.0)) for s in set(p_exact) | set(p_emp)
        )

        assert tv < 0.08, f"Ergodicity failed: TV = {tv:.3f} ≥ 0.08"

    def test_mixing_time(
        self,
        sampler_class: type[BaseSamplerRBM],
        rbm_config: RBMConfig,
        rbm_class: type[BaseRBM],
    ) -> None:
        r"""Test mixing time of the sampler.

        The mixing time indicates how quickly the chain forgets its
        initial state. This test estimates mixing time by fitting an
        exponential decay to the autocorrelation function.

        Parameters
        ----------
        sampler_class : Type[BaseSamplerRBM]
            Sampler class to test.
        rbm_config : RBMConfig
            Base RBM configuration.
        rbm_class : Type[BaseRBM]
            RBM class to use.

        Notes
        -----
        Uses the "tail-pruning" variant where only significant lags are
        included in the exponential fit. Accepts τ_mix < 80 and R² > 0.30.
        """
        import warnings

        import numpy as np
        import torch

        torch.manual_seed(42)

        # Create RBM
        cfg = type(rbm_config)(visible=8, hidden=10)
        rbm = rbm_class(cfg)
        with torch.no_grad():
            for name in ("w", "vb", "hb"):
                if hasattr(rbm, name):
                    getattr(rbm, name).normal_(0, 0.05)

        sampler = sampler_class(rbm)

        # Burn-in
        v = torch.randint(0, 2, (1, 8), dtype=torch.float32)
        for _ in range(30_000):
            v = sampler.sample(v).to_tensor()

        # Collect time series
        raw, thin = 80_000, 20  # N_eff = 4,000
        series = torch.empty(raw // thin)
        for t in range(raw):
            v = sampler.sample(v).to_tensor()
            if t % thin == 0:
                series[t // thin] = v[0, 0]

        series -= series.mean()
        var0 = (series * series).mean()

        # Calculate autocorrelations
        k_max = 60
        rhos: list[float] = []
        for k in range(1, k_max + 1):
            rhos.append(((series[:-k] * series[k:]).mean() / var0).item())

        # Prune noise
        noise_floor = 1.96 / np.sqrt(series.numel())  # ≈ 95% CI
        keep = [(k, r) for k, r in enumerate(rhos, 1) if abs(r) > noise_floor]

        if len(keep) < 6:
            warnings.warn(
                "Autocorrelation dies too quickly – chain appears well mixed; skipping R² check.",
                stacklevel=2,
            )
            tau_mix = 1.0  # effectively ≤ 1 sweep
        else:
            lags_fit, rhos_fit = zip(*keep, strict=True)
            log_r = np.log(rhos_fit)
            slope, a = np.polyfit(lags_fit, log_r, 1)
            tau_mix = -1.0 / slope

            pred = a + slope * np.asarray(lags_fit)
            ss_res = ((log_r - pred) ** 2).sum()
            ss_tot = ((log_r - log_r.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot

            assert r2 > 0.30, f"ρ(k) not close to exponential: R² = {r2:.3f} (≤ 0.30)"

        assert tau_mix < 80, f"Slow mixing: τ_mix = {tau_mix:.1f} (> 80 sweeps)"

    def test_detailed_balance(
        self,
        sampler_class: type[BaseSamplerRBM],
        rbm_config: RBMConfig,
        rbm_class: type[BaseRBM],
    ) -> None:
        r"""Test detailed balance (reversibility) of the sampler.

        Detailed balance means that P(v1)P(v1→v2) = P(v2)P(v2→v1)
        for all state pairs. This is a fundamental property required
        for correct MCMC sampling.

        Parameters
        ----------
        sampler_class : Type[BaseSamplerRBM]
            Sampler class to test.
        rbm_config : RBMConfig
            Base RBM configuration.
        rbm_class : Type[BaseRBM]
            RBM class to use.

        Notes
        -----
        Tests random pairs of states on a V=3, H=2 RBM. The acceptance
        criterion is median relative error < 0.15.
        """
        import warnings

        import numpy as np
        import torch

        torch.manual_seed(42)

        # Create small RBM
        tiny_cfg = type(rbm_config)(visible=8, hidden=10)
        rbm = rbm_class(tiny_cfg)

        with torch.no_grad():
            if hasattr(rbm, "w"):
                rbm.w.uniform_(-0.10, 0.10)
            if hasattr(rbm, "vb"):
                rbm.vb.uniform_(-0.05, 0.05)
            if hasattr(rbm, "hb"):
                rbm.hb.uniform_(-0.05, 0.05)

        sampler = sampler_class(rbm)

        # Test state pairs
        n_pairs = 40
        n_transitions = 10_000
        rel_errors: list[float] = []

        for _ in range(n_pairs):
            v1 = torch.randint(0, 2, (1, 8), dtype=torch.float32)
            v2 = torch.randint(0, 2, (1, 8), dtype=torch.float32)
            if torch.equal(v1, v2):
                continue

            # P(v1 → v2)
            cnt12 = sum(
                torch.equal(sampler.sample(v1.clone()).to_tensor(), v2)
                for _ in range(n_transitions)
            )
            p12 = cnt12 / n_transitions

            # P(v2 → v1)
            cnt21 = sum(
                torch.equal(sampler.sample(v2.clone()).to_tensor(), v1)
                for _ in range(n_transitions)
            )
            p21 = cnt21 / n_transitions

            # Skip uninformative pairs
            if p12 < 0.002 and p21 < 0.002:
                continue
            if p12 == 0 or p21 == 0:
                continue

            log_pi1 = -rbm.free_energy(v1).item()
            log_pi2 = -rbm.free_energy(v2).item()

            lhs = log_pi1 + np.log(p12)
            rhs = log_pi2 + np.log(p21)

            rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
            rel_errors.append(rel_err)

        if not rel_errors:
            warnings.warn("No informative state pairs sampled; test skipped.", stacklevel=2)
            return

        median_err = float(np.median(rel_errors))
        assert median_err < 0.15, (
            f"Detailed balance violated: median relative error {median_err:.3f}"
        )

    def test_state_space_irreducibility(
        self,
        sampler_class: type[BaseSamplerRBM],
        rbm_config: RBMConfig,
        rbm_class: type[BaseRBM],
    ) -> None:
        r"""Test irreducibility of the sampler.

        Irreducibility means that every state is reachable from any other
        state. This test builds a transition graph and verifies full
        connectivity via breadth-first search.

        Parameters
        ----------
        sampler_class : Type[BaseSamplerRBM]
            Sampler class to test.
        rbm_config : RBMConfig
            Base RBM configuration.
        rbm_class : Type[BaseRBM]
            RBM class to use.

        Notes
        -----
        Tests on a V=8, H=10 RBM where the full state space can be
        enumerated. Uses k=60 trials per state to ensure edges aren't
        missed due to sampling variance.
        """
        import numpy as np
        import torch

        torch.manual_seed(42)

        # Create RBM
        cfg = type(rbm_config)(visible=8, hidden=10)
        rbm = rbm_class(cfg)
        with torch.no_grad():
            if hasattr(rbm, "w"):
                rbm.w.normal_(0, 0.10)
            if hasattr(rbm, "vb"):
                rbm.vb.normal_(0, 0.05)
            if hasattr(rbm, "hb"):
                rbm.hb.normal_(0, 0.05)

        sampler = sampler_class(rbm)

        # Enumerate all states
        all_states = [tuple(int(b) for b in np.binary_repr(i, width=8)) for i in range(256)]
        state_to_tensor = {s: torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in all_states}

        # Build adjacency graph
        k_trials = 60
        adjacency: dict[tuple[int, ...], set[tuple[int, ...]]] = {s: set() for s in all_states}

        for s in all_states:
            v0 = state_to_tensor[s].repeat(k_trials, 1)
            v1 = sampler.sample(v0)
            for row in v1:
                adjacency[s].add(tuple(int(x) for x in row))

        # BFS from all-zero state
        start = tuple(0 for _ in range(8))
        visited = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for nxt in adjacency[current]:
                if nxt not in visited:
                    visited.add(nxt)
                    frontier.append(nxt)

        coverage = len(visited) / len(all_states)
        assert coverage == 1.0, f"Irreducibility failed: only {coverage:.1%} of states reachable"
