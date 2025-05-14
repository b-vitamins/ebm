"""Sampler test utilities and hypothesis strategies.

This module provides utilities for testing RBM samplers, including
hypothesis strategies for property-based testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from hypothesis import strategies as st
from hypothesis.strategies import composite

from ebm.rbm.model.base import RBMConfig
from ebm.rbm.sampler.cd import CDSampler
from tests.helper.rbm import MockRBM

if TYPE_CHECKING:
    from hypothesis.strategies._internal.core import DrawFn

    from ebm.rbm.sampler.base import BaseSamplerRBM


@composite  # type: ignore[misc]
def sampler_model_config(
    draw: DrawFn,
) -> tuple[BaseSamplerRBM, torch.Tensor, int, int, int]:
    """Generate random sampler configurations for property testing.

    Parameters
    ----------
    draw : DrawFn
        Hypothesis draw function.

    Returns
    -------
    Tuple[BaseSamplerRBM, torch.Tensor, int, int, int]
        Tuple of (sampler, v0, batch_size, visible, hidden).
    """
    # Size parameters
    visible = draw(st.integers(min_value=3, max_value=64))
    hidden = draw(st.integers(min_value=3, max_value=64))
    batch_size = draw(st.integers(min_value=1, max_value=32))
    dtype = draw(st.sampled_from([torch.float32, torch.float64]))

    # Create model
    config = RBMConfig(visible=visible, hidden=hidden, dtype=dtype)
    model = MockRBM(config)

    # Choose sampler type (for now just CD, but could be extended)
    sampler_type = draw(st.sampled_from([CDSampler]))

    # Create sampler with optional parameters
    if sampler_type == CDSampler:
        k = draw(st.integers(min_value=1, max_value=10))
        sampler = CDSampler(model, k=k)
    else:
        sampler = sampler_type(model)

    # Generate input data
    v0 = torch.randn(batch_size, visible, dtype=dtype)

    return sampler, v0, batch_size, visible, hidden


@composite  # type: ignore[misc]
def sampler_config_from_fixtures(
    draw: DrawFn,
) -> dict[str, torch.Tensor | int | torch.dtype]:
    """Generate sampler configuration for fixture-based testing.

    This version is designed to work with the fixture-based testing approach.

    Parameters
    ----------
    draw : DrawFn
        Hypothesis draw function.

    Returns
    -------
    Dict[str, Union[torch.Tensor, int, torch.dtype]]
        Configuration dictionary with keys: v0, batch_size, visible, hidden, dtype.
    """
    visible = draw(st.integers(min_value=3, max_value=64))
    hidden = draw(st.integers(min_value=3, max_value=64))
    batch_size = draw(st.integers(min_value=1, max_value=32))
    dtype = draw(st.sampled_from([torch.float32, torch.float64]))

    v0 = torch.randn(batch_size, visible, dtype=dtype)

    # Return config data that tests can use with fixtures
    return {
        "v0": v0,
        "batch_size": batch_size,
        "visible": visible,
        "hidden": hidden,
        "dtype": dtype,
    }
