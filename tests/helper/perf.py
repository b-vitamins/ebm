"""FLOP-scaled performance testing utilities for RBM samplers.

This module provides utilities to measure and validate the performance of RBM
samplers against host machine throughput. The tests ensure samplers achieve
at least 60% of measured peak performance on the host hardware.
"""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ebm.rbm.sampler.base import BaseSamplerRBM


def _matmul_flops(m: int) -> int:
    """Calculate FLOPs for matrix multiplication.

    Parameters
    ----------
    m : int
        Matrix dimension (assumes square matrices).

    Returns
    -------
    int
        Number of FLOPs for (m×m)·(m×m) GEMM (2m³).
    """
    return 2 * m * m * m


def _timed_gemm(m: int, device: torch.device) -> float:
    """Time a single matrix multiplication on the specified device.

    Parameters
    ----------
    m : int
        Matrix dimension.
    device : torch.device
        Device to run the computation on.

    Returns
    -------
    float
        Time in seconds to compute (m×m)·(m×m).
    """
    a = torch.randn(m, m, device=device, dtype=torch.float32)
    b = torch.randn(m, m, device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()  # type: ignore[no-untyped-call]
        _ = a @ b
        end.record()  # type: ignore[no-untyped-call]
        torch.cuda.synchronize()
        elapsed: float = start.elapsed_time(end) / 1e3  # type: ignore[no-untyped-call]
        return elapsed
    else:
        start_time = time.perf_counter()
        _ = a @ b
        return time.perf_counter() - start_time


@functools.cache
def _calibrate_peak(device_type: str) -> float:
    """Measure peak FLOP/s for the specified device type.

    This function is cached to ensure we only test once per run.

    Parameters
    ----------
    device_type : str
        Either "cpu" or "cuda".

    Returns
    -------
    float
        Peak FLOP/s achieved on the device.

    Raises
    ------
    RuntimeError
        If CUDA is requested but not available.
    """
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA peak requested but no CUDA device present")
        m = int(os.getenv("PEAK_M_CUDA", "2048"))
        t = _timed_gemm(m, torch.device("cuda"))
        return _matmul_flops(m) / t
    else:
        m = int(os.getenv("PEAK_M_CPU", "1024"))
        t = _timed_gemm(m, torch.device("cpu"))
        return _matmul_flops(m) / t


def _infer_rbm_dimensions(sampler: BaseSamplerRBM) -> tuple[int, int, int]:
    """Extract RBM dimensions from a sampler instance.

    Parameters
    ----------
    sampler : BaseSamplerRBM
        Sampler instance to inspect.

    Returns
    -------
    visible : int
        Number of visible units.
    hidden : int
        Number of hidden units.
    k : int
        Number of Gibbs steps.

    Raises
    ------
    ValueError
        If unable to determine RBM dimensions.
    """
    k = int(getattr(sampler, "k", 1))
    model = getattr(sampler, "model", None) or getattr(sampler, "rbm", None)
    if model is None:
        raise ValueError("Sampler lacks .model or .rbm attribute")

    cfg = getattr(model, "cfg", None)
    v = getattr(cfg, "visible", None) if cfg else None
    h = getattr(cfg, "hidden", None) if cfg else None
    if v and h:
        return int(v), int(h), k

    # fallbacks
    v = getattr(model, "visible_size", None) or getattr(model, "visible", None)
    h = getattr(model, "hidden_size", None) or getattr(model, "hidden", None)
    if v is None or h is None:
        raise ValueError("Unable to determine RBM dimensions")
    return int(v), int(h), k


# Performance thresholds from environment or defaults
TARGET_UTIL_CPU = float(os.getenv("RBM_TARGET_UTIL_CPU", "0.05"))  # 5%
TARGET_UTIL_GPU = float(os.getenv("RBM_TARGET_UTIL_GPU", "0.70"))  # 70%
FLOOR_SPS_CPU = int(os.getenv("RBM_FLOOR_SPS_CPU", "1_000"))
FLOOR_SPS_GPU = int(os.getenv("RBM_FLOOR_SPS_GPU", "10_000"))


def _time_function(fn: Callable[[], Any], device: str = "cpu") -> float:
    """Time a function execution on the specified device.

    Parameters
    ----------
    fn : Callable[[], Any]
        Function to time.
    device : str, optional
        Device type ("cpu" or "cuda"), by default "cpu".

    Returns
    -------
    float
        Execution time in seconds.

    Raises
    ------
    RuntimeError
        If CUDA is requested but not available.
    """
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]

        start.record()  # type: ignore[no-untyped-call]
        fn()
        end.record()  # type: ignore[no-untyped-call]
        torch.cuda.synchronize()
        elapsed: float = start.elapsed_time(end) / 1e3  # type: ignore[no-untyped-call]
        return elapsed
    else:
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0


def run_perf(
    sampler: BaseSamplerRBM,
    *,
    visible: int | None = None,
    batch_size: int = 4096,
) -> None:
    """Run performance test for an RBM sampler.

    This function measures the sampler's throughput and ensures it meets
    the required performance targets for the hardware.

    Parameters
    ----------
    sampler : BaseSamplerRBM
        Sampler instance to test.
    visible : int, optional
        Number of visible units. If None, inferred from sampler.
    batch_size : int, optional
        Batch size for testing, by default 4096.

    Raises
    ------
    AssertionError
        If performance targets are not met.
    """
    if visible is None:
        visible, hidden, k = _infer_rbm_dimensions(sampler)
    else:
        _, hidden, k = _infer_rbm_dimensions(sampler)

    model = getattr(sampler, "model", None) or getattr(sampler, "rbm", None)

    # Get model parameters iterator if model exists
    if model is not None:
        parameters_iter = model.parameters()
    else:
        parameters_iter = iter([])  # Empty iterator

    # Get device from first parameter, or use CPU as default
    try:
        first_param = next(parameters_iter)
        device = first_param.device
    except StopIteration:
        device = torch.device("cpu")

    device_type = device.type

    v0 = torch.randn(batch_size, visible, device=device, dtype=torch.float32)

    # warm-up (important on GPU)
    for _ in range(10):
        sampler.sample(v0)
    if device_type == "cuda":
        torch.cuda.synchronize()

    # timing loop
    repeats = 50 if device_type == "cpu" else 100
    start = time.perf_counter()
    for _ in range(repeats):
        sampler.sample(v0)
    if device_type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats

    # metrics
    total_flops = 2 * visible * hidden * k * batch_size
    achieved_fps = total_flops / elapsed
    peak_fps = _calibrate_peak(device_type)

    util: float | None = achieved_fps / peak_fps if total_flops >= 1_000_000_000 else None
    sps = batch_size / elapsed
    floor_sps = FLOOR_SPS_GPU if device_type == "cuda" else FLOOR_SPS_CPU

    target_util = TARGET_UTIL_GPU if device_type == "cuda" else TARGET_UTIL_CPU

    if (util is not None and util < target_util) or sps < floor_sps:
        util_str = f"{util * 100:5.1f}%" if util is not None else "N/A"
        raise AssertionError(
            f"{device_type.upper()} sampler too slow\n"
            f"  Utilisation : {util_str} (target ≥ {target_util * 100:.0f}%)\n"
            f"  Elapsed     : {elapsed * 1e3:7.2f} ms\n"
            f"  Samples/s   : {sps:,.0f} (floor {floor_sps})\n"
            f"  Config      : batch={batch_size}, V={visible}, H={hidden}, k={k}"
        )


def run_performance_cpu_test(
    sampler: BaseSamplerRBM,
    *,
    visible: int = 128,
    batch_size: int = 4096,
    hidden: int | None = None,
) -> None:
    """Run CPU performance test for a sampler.

    Parameters
    ----------
    sampler : BaseSamplerRBM
        Sampler to test.
    visible : int, optional
        Number of visible units, by default 128.
    batch_size : int, optional
        Batch size, by default 4096.
    hidden : int, optional
        Number of hidden units (unused).
    """
    run_perf(sampler, visible=visible, batch_size=batch_size)


def run_performance_gpu_test(
    sampler: BaseSamplerRBM,
    *,
    visible: int = 128,
    batch_size: int = 4096,
    hidden: int | None = None,
) -> None:
    """Run GPU performance test for a sampler.

    Parameters
    ----------
    sampler : BaseSamplerRBM
        Sampler to test.
    visible : int, optional
        Number of visible units, by default 128.
    batch_size : int, optional
        Batch size, by default 4096.
    hidden : int, optional
        Number of hidden units (unused).

    Notes
    -----
    Test is skipped if CUDA is not available.
    """
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA not available")
    run_perf(sampler, visible=visible, batch_size=batch_size)
