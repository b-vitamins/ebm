"""RBM test utilities and mock implementations.

This module provides mock RBM implementations, utilities for RBM testing,
and hypothesis strategies for property-based testing of RBM models.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, overload

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from hypothesis import strategies as st
from hypothesis.strategies import composite

from ebm.rbm.model.base import BaseRBM, RBMConfig
from ebm.rbm.utils import shape_beta

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn


class MockRBM(BaseRBM):
    """Minimal Bernoulli-Bernoulli RBM for testing.

    This implementation is statistically correct and suitable for
    unit tests and property-based testing.

    Parameters
    ----------
    cfg : RBMConfig
        Configuration for the mock RBM.
    """

    def __init__(self, cfg: RBMConfig) -> None:
        """Initialize the mock RBM.

        Parameters
        ----------
        cfg : RBMConfig
            Configuration specifying visible/hidden dimensions and dtype/device.
        """
        super().__init__()
        self.cfg = cfg

        # Initialize with proper device and dtype from config
        device = cfg.device if cfg.device is not None else torch.device("cpu")
        dtype = cfg.dtype if cfg.dtype is not None else torch.float32

        self.w = nn.Parameter(
            torch.randn(cfg.hidden, cfg.visible, device=device, dtype=dtype) * 0.01
        )
        self.vb = nn.Parameter(torch.zeros(cfg.visible, device=device, dtype=dtype))
        self.hb = nn.Parameter(torch.zeros(cfg.hidden, device=device, dtype=dtype))

    @overload
    def to(
        self,
        device: str | torch.device | int | None = ...,
        dtype: torch.dtype | None = ...,
        non_blocking: bool = ...,
    ) -> MockRBM: ...
    @overload
    def to(self, dtype: torch.dtype, non_blocking: bool = ...) -> MockRBM: ...
    @overload
    def to(self, tensor: torch.Tensor, non_blocking: bool = ...) -> MockRBM: ...

    def to(self, *args: Any, **kwargs: Any) -> MockRBM:
        """Move model to specified device/dtype.

        Parameters
        ----------
        *args : Any
            Positional arguments matching PyTorch's Module.to signature.
        **kwargs : Any
            Keyword arguments matching PyTorch's Module.to signature.

        Returns
        -------
        MockRBM
            Self reference after moving parameters.
        """
        # Call parent's to method
        super().to(*args, **kwargs)

        # Extract device and dtype based on the overload
        device = None
        dtype = None

        if args:
            if isinstance(args[0], str | torch.device | int):
                device = args[0]
                if len(args) > 1 and isinstance(args[1], torch.dtype):
                    dtype = args[1]
            elif isinstance(args[0], torch.dtype):
                dtype = args[0]
            elif isinstance(args[0], torch.Tensor):
                device = args[0].device
                dtype = args[0].dtype

        # Check kwargs
        device = kwargs.get("device", device)
        dtype = kwargs.get("dtype", dtype)

        # Update config to reflect new device/dtype
        if device is not None:
            device = torch.device(device) if isinstance(device, str | int) else device
            self.cfg = RBMConfig(
                visible=self.cfg.visible,
                hidden=self.cfg.hidden,
                device=device,
                dtype=self.cfg.dtype,
            )
        if dtype is not None:
            self.cfg = RBMConfig(
                visible=self.cfg.visible,
                hidden=self.cfg.hidden,
                device=self.cfg.device,
                dtype=dtype,
            )
        return self

    def preact_h(self, v: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate hidden unit pre-activation.

        Parameters
        ----------
        v : torch.Tensor
            Visible units with shape (..., visible).
        beta : torch.Tensor, optional
            Temperature parameter for parallel tempering.

        Returns
        -------
        torch.Tensor
            Pre-activation values with shape (..., hidden).
        """
        pre = F.linear(v, self.w, self.hb)
        β = shape_beta(beta, pre)
        return pre if β is None else pre * β

    def preact_v(self, h: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate visible unit pre-activation.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units with shape (..., hidden).
        beta : torch.Tensor, optional
            Temperature parameter for parallel tempering.

        Returns
        -------
        torch.Tensor
            Pre-activation values with shape (..., visible).
        """
        pre = F.linear(h, self.w.t(), self.vb)
        β = shape_beta(beta, pre)
        return pre if β is None else pre * β

    def prob_h_given_v(self, v: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate P(h|v) probabilities.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        beta : torch.Tensor, optional
            Temperature parameter.

        Returns
        -------
        torch.Tensor
            Hidden unit probabilities.
        """
        return torch.sigmoid(self.preact_h(v, beta=beta))

    def prob_v_given_h(self, h: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate P(v|h) probabilities.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units.
        beta : torch.Tensor, optional
            Temperature parameter.

        Returns
        -------
        torch.Tensor
            Visible unit probabilities.
        """
        return torch.sigmoid(self.preact_v(h, beta=beta))

    def sample_h_given_v(
        self, v: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample hidden units given visible units.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        beta : torch.Tensor, optional
            Temperature parameter.

        Returns
        -------
        torch.Tensor
            Sampled hidden units.
        """
        probs = self.prob_h_given_v(v, beta=beta)
        return torch.bernoulli(probs).to(v.dtype)

    def sample_v_given_h(
        self, h: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample visible units given hidden units.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units.
        beta : torch.Tensor, optional
            Temperature parameter.

        Returns
        -------
        torch.Tensor
            Sampled visible units.
        """
        probs = self.prob_v_given_h(h, beta=beta)
        return torch.bernoulli(probs).to(h.dtype)

    def energy(
        self, v: torch.Tensor, h: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Calculate joint energy E(v,h).

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        h : torch.Tensor
            Hidden units.
        beta : torch.Tensor, optional
            Temperature parameter.

        Returns
        -------
        torch.Tensor
            Energy values.
        """
        e = -torch.einsum("...v,...h,hv->...", v, h, self.w)
        e -= torch.einsum("...v,v->...", v, self.vb)
        e -= torch.einsum("...h,h->...", h, self.hb)
        β = shape_beta(beta, e)
        return e if β is None else e * β

    def free_energy(self, v: torch.Tensor, *, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate free energy F(v).

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        beta : torch.Tensor, optional
            Temperature parameter.

        Returns
        -------
        torch.Tensor
            Free energy values.
        """
        vb_term = -torch.einsum("...v,v->...", v, self.vb)
        hidden_term = -torch.sum(F.softplus(self.preact_h(v, beta=beta)), dim=-1)

        if beta is None:
            return vb_term + hidden_term
        else:
            β = shape_beta(beta, vb_term)
            return (vb_term * β if β is not None else vb_term) + hidden_term


class BetaRecordingModel(MockRBM):
    """Mock RBM that records all beta values passed to sampling methods.

    This is useful for testing parallel tempering implementations.

    Parameters
    ----------
    cfg : RBMConfig
        Configuration for the model.
    """

    def __init__(self, cfg: RBMConfig) -> None:
        """Initialize the beta recording model.

        Parameters
        ----------
        cfg : RBMConfig
            Configuration for the model.
        """
        super().__init__(cfg)
        self.betas: list[torch.Tensor | None] = []

    def sample_h_given_v(
        self, v: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample hidden units and record beta.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        beta : torch.Tensor, optional
            Temperature parameter.

        Returns
        -------
        torch.Tensor
            Sampled hidden units.
        """
        self.betas.append(beta)
        return super().sample_h_given_v(v, beta=beta)

    def sample_v_given_h(
        self, h: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample visible units and record beta.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units.
        beta : torch.Tensor, optional
            Temperature parameter.

        Returns
        -------
        torch.Tensor
            Sampled visible units.
        """
        self.betas.append(beta)
        return super().sample_v_given_h(h, beta=beta)


class BetaValidatingModel(MockRBM):
    """Mock RBM that validates beta matches expected value.

    This is useful for testing that beta values are correctly propagated.

    Parameters
    ----------
    cfg : RBMConfig
        Configuration for the model.
    expected : torch.Tensor
        Expected beta value to validate against.
    """

    def __init__(self, cfg: RBMConfig, expected: torch.Tensor) -> None:
        """Initialize the beta validating model.

        Parameters
        ----------
        cfg : RBMConfig
            Configuration for the model.
        expected : torch.Tensor
            Expected beta value.
        """
        super().__init__(cfg)
        self.expected = expected

    def _validate_beta(self, beta: torch.Tensor | None) -> None:
        """Validate that beta matches expected value.

        Parameters
        ----------
        beta : torch.Tensor, optional
            Beta value to validate.

        Raises
        ------
        AssertionError
            If beta doesn't match expected value.
        """
        assert beta is not None, "Expected beta but got None"
        assert torch.equal(beta, self.expected), (
            f"Beta mismatch: got {beta}, expected {self.expected}"
        )

    def sample_h_given_v(
        self, v: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample hidden units with beta validation.

        Parameters
        ----------
        v : torch.Tensor
            Visible units.
        beta : torch.Tensor, optional
            Temperature parameter.

        Returns
        -------
        torch.Tensor
            Sampled hidden units.
        """
        self._validate_beta(beta)
        return super().sample_h_given_v(v, beta=beta)

    def sample_v_given_h(
        self, h: torch.Tensor, *, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample visible units with beta validation.

        Parameters
        ----------
        h : torch.Tensor
            Hidden units.
        beta : torch.Tensor, optional
            Temperature parameter.

        Returns
        -------
        torch.Tensor
            Sampled visible units.
        """
        self._validate_beta(beta)
        return super().sample_v_given_h(h, beta=beta)


def xavier_init(rbm: BaseRBM) -> None:
    """Apply Xavier/Glorot initialization to RBM parameters.

    Parameters
    ----------
    rbm : BaseRBM
        RBM model to initialize.
    """
    fan_in = rbm.cfg.visible
    fan_out = rbm.cfg.hidden
    std = math.sqrt(2.0 / (fan_in + fan_out))

    with torch.no_grad():
        if hasattr(rbm, "w"):
            rbm.w.normal_(0.0, std)
        if hasattr(rbm, "vb"):
            rbm.vb.zero_()
        if hasattr(rbm, "hb"):
            rbm.hb.zero_()


def make_rbm(
    model_cls: type[MockRBM] = MockRBM,
    visible: int = 4,
    hidden: int = 3,
    *,
    weight_scale: float = 0.01,
    use_xavier: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> MockRBM:
    """Create an RBM with controlled initialization.

    Parameters
    ----------
    model_cls : Type[MockRBM], optional
        Model class to instantiate, by default MockRBM.
    visible : int, optional
        Number of visible units, by default 4.
    hidden : int, optional
        Number of hidden units, by default 3.
    weight_scale : float, optional
        Scale for weight initialization, by default 0.01.
    use_xavier : bool, optional
        Whether to use Xavier initialization, by default False.
    dtype : torch.dtype, optional
        Data type, by default torch.float32.
    device : torch.device, optional
        Device for the model.

    Returns
    -------
    MockRBM
        Initialized RBM model.
    """
    cfg = RBMConfig(visible=visible, hidden=hidden, dtype=dtype, device=device)
    rbm = model_cls(cfg)

    with torch.no_grad():
        if use_xavier:
            xavier_init(rbm)
        else:
            rbm.w.normal_(0.0, weight_scale)
            # Keep biases at zero for testing
            rbm.vb.zero_()
            rbm.hb.zero_()

    return rbm


def exact_visible_dist(rbm: BaseRBM) -> dict[tuple[int, ...], float]:
    """Compute exact P(v) by brute force enumeration.

    Warning: Only feasible for visible_size <= 10!

    Parameters
    ----------
    rbm : BaseRBM
        RBM model to analyze.

    Returns
    -------
    Dict[Tuple[int, ...], float]
        Dictionary mapping visible states to probabilities.

    Raises
    ------
    ValueError
        If visible dimension is too large (>10).
    """
    v_dim = rbm.cfg.visible
    if v_dim > 10:
        raise ValueError(f"Cannot compute exact distribution for V={v_dim} > 10")

    # Generate all possible visible states
    num_states = 2**v_dim
    device = next(rbm.parameters()).device
    dtype = next(rbm.parameters()).dtype

    states = torch.stack(
        [
            torch.tensor([(i >> j) & 1 for j in range(v_dim)], dtype=dtype, device=device)
            for i in range(num_states)
        ]
    )

    # Compute probabilities
    with torch.no_grad():
        log_probs = -rbm.free_energy(states)
        log_probs -= torch.logsumexp(log_probs, 0)  # normalize
        probs = log_probs.exp()

    # Convert to dictionary
    return {
        tuple(int(x) for x in states[i].cpu().tolist()): float(probs[i].cpu().item())
        for i in range(num_states)
    }


def print_distribution_comparison(
    p_exact: dict[tuple[int, ...], float],
    p_empirical: dict[tuple[int, ...], float],
) -> None:
    """Pretty-print comparison of two probability distributions.

    Parameters
    ----------
    p_exact : Dict[Tuple[int, ...], float]
        Exact probability distribution.
    p_empirical : Dict[Tuple[int, ...], float]
        Empirical probability distribution.
    """
    print("\nState      Exact    Empirical  |Δ|")
    print("-" * 35)

    all_states = sorted(set(p_exact.keys()) | set(p_empirical.keys()))

    for state in all_states:
        exact = p_exact.get(state, 0.0)
        empirical = p_empirical.get(state, 0.0)
        diff = abs(exact - empirical)
        print(f"{state!s:<10} {exact:.4f}  {empirical:.4f}    {diff:.4f}")


@composite  # type: ignore[misc]
def rbm_model_config(draw: DrawFn) -> tuple[MockRBM, torch.Tensor, int, int, int]:
    """Generate random RBM configurations for property testing.

    Parameters
    ----------
    draw : DrawFn
        Hypothesis draw function.

    Returns
    -------
    Tuple[MockRBM, torch.Tensor, int, int, int]
        Tuple of (model, v0, batch_size, visible, hidden).
    """
    visible = draw(st.integers(min_value=3, max_value=64))
    hidden = draw(st.integers(min_value=3, max_value=64))
    batch_size = draw(st.integers(min_value=1, max_value=32))
    dtype = draw(st.sampled_from([torch.float32, torch.float64]))

    model = MockRBM(RBMConfig(visible=visible, hidden=hidden, dtype=dtype))
    v0 = torch.randn(batch_size, visible, dtype=dtype)

    return model, v0, batch_size, visible, hidden
