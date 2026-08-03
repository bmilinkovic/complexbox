#!/usr/bin/env python3
"""Quick, deterministic NumPy-versus-Torch CPU benchmark.

Run from any directory with::

    python tools/benchmark_torch.py

The benchmark uses public ``complexbox`` APIs, includes NumPy/Torch transfer
costs, performs exact-shape warm-ups, prints results, and creates no result
files.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complexbox.mvgc import var2trfun  # noqa: E402
from complexbox.ssdi import opt_gd_dds_mruns, opt_gd_ddx_mruns  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _random_bases(rng: np.random.Generator, n: int, m: int, runs: int) -> np.ndarray:
    bases = np.empty((n, m, runs))
    for run in range(runs):
        bases[:, :, run] = np.linalg.svd(rng.standard_normal((n, m)), full_matrices=False)[0]
    return bases


def _time_calls(fn: Callable[[], Any], repeats: int) -> tuple[Any, list[float]]:
    result = None
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        elapsed.append(time.perf_counter() - start)
    return result, elapsed


def _projector_error(left: np.ndarray, right: np.ndarray) -> float:
    left_projector = np.einsum("imr,jmr->ijr", left, left)
    right_projector = np.einsum("imr,jmr->ijr", right, right)
    return float(np.max(np.abs(left_projector - right_projector)))


def _ssdi_parity(numpy_result, torch_result) -> tuple[str, bool]:
    objective_error = float(np.max(np.abs(numpy_result[0] - torch_result[0])))
    projector_error = _projector_error(numpy_result[1], torch_result[1])
    convergence_equal = numpy_result[2] == torch_result[2]
    ok = objective_error < 1e-10 and projector_error < 1e-9 and convergence_equal
    text = (
        f"DD={objective_error:.2e}, P={projector_error:.2e}, "
        f"conv={'yes' if convergence_equal else 'NO'}"
    )
    return text, ok


def _array_parity(numpy_result, torch_result) -> tuple[str, bool]:
    error = float(np.max(np.abs(numpy_result - torch_result)))
    return f"max={error:.2e}", error < 1e-10


def _benchmark_case(
    name: str,
    dimensions: str,
    numpy_fn: Callable[[], Any],
    torch_fn: Callable[[], Any],
    parity_fn: Callable[[Any, Any], tuple[str, bool]],
    repeats: int,
) -> dict[str, Any]:
    # Keep each backend's warm-up adjacent to its measured calls. This avoids
    # counting allocator/thread-pool setup and reduces cross-BLAS variability.
    numpy_fn()
    numpy_result, numpy_times = _time_calls(numpy_fn, repeats)
    for _ in range(3):
        torch_fn()
    torch_result, torch_times = _time_calls(torch_fn, repeats)

    numpy_median = statistics.median(numpy_times)
    torch_median = statistics.median(torch_times)
    parity, parity_ok = parity_fn(numpy_result, torch_result)
    return {
        "name": name,
        "dimensions": dimensions,
        "numpy": numpy_median,
        "torch": torch_median,
        "speedup": numpy_median / torch_median,
        "parity": parity,
        "parity_ok": parity_ok,
        "numpy_range": (min(numpy_times), max(numpy_times)),
        "torch_range": (min(torch_times), max(torch_times)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=3,
        help="timed calls per backend; median is reported (default: 3)",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        parser.error("PyTorch is not installed; install the Torch optional dependency")

    rng = np.random.default_rng(20260803)
    results = []

    n, m, runs, lags, maxiters = 20, 5, 64, 24, 10
    proxy_bases = _random_bases(rng, n, m, runs)
    cak = 0.05 * rng.standard_normal((n, n, lags))
    proxy_options = dict(
        maxiters=maxiters,
        variant=2,
        gdsig0=1e-3,
        tol=(-1.0, -1.0, -1.0),
        history=False,
    )
    results.append(
        _benchmark_case(
            "SSDI proxy",
            f"n={n}, m={m}, R={runs}, lags={lags}, iters={maxiters}",
            lambda: opt_gd_ddx_mruns(cak, proxy_bases, backend="numpy", **proxy_options),
            lambda: opt_gd_ddx_mruns(
                cak,
                proxy_bases,
                backend="torch",
                device="cpu",
                run_chunk_size=32,
                lag_chunk_size=8,
                **proxy_options,
            ),
            _ssdi_parity,
            args.repeats,
        )
    )

    n, m, runs, frequencies, maxiters = 12, 4, 32, 129, 7
    spectral_bases = _random_bases(rng, n, m, runs)
    transfer = np.empty((n, n, frequencies), dtype=np.complex128)
    for frequency in range(frequencies):
        perturbation = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        transfer[:, :, frequency] = np.eye(n) + 0.02 * perturbation
    spectral_options = dict(
        maxiters=maxiters,
        variant=2,
        gdsig0=1e-3,
        tol=(-1.0, -1.0, -1.0),
        history=False,
    )
    results.append(
        _benchmark_case(
            "SSDI spectral",
            f"n={n}, m={m}, R={runs}, F={frequencies}, iters={maxiters}",
            lambda: opt_gd_dds_mruns(transfer, spectral_bases, backend="numpy", **spectral_options),
            lambda: opt_gd_dds_mruns(
                transfer,
                spectral_bases,
                backend="torch",
                device="cpu",
                run_chunk_size=16,
                frequency_chunk_size=64,
                **spectral_options,
            ),
            _ssdi_parity,
            args.repeats,
        )
    )

    n, order, fres = 24, 8, 2048
    coefficients = 0.01 * rng.standard_normal((n, n, order))
    results.append(
        _benchmark_case(
            "MVGC var2trfun",
            f"n={n}, p={order}, F={fres + 1}",
            lambda: var2trfun(coefficients, fres, backend="numpy"),
            lambda: var2trfun(
                coefficients,
                fres,
                backend="torch",
                device="cpu",
                dtype="float64",
                batch_size=128,
            ),
            _array_parity,
            args.repeats,
        )
    )

    print(
        f"complexbox Torch CPU benchmark | {platform.machine()} | "
        f"logical CPUs={os.cpu_count()} | NumPy={np.__version__} | "
        f"Torch={torch.__version__} | Torch threads={torch.get_num_threads()}"
    )
    print(f"Median of {args.repeats} timed calls after exact-shape warm-up; seconds")
    print()
    print(f"{'workload':<17} {'NumPy':>9} {'Torch':>9} {'speedup':>9}  parity")
    print("-" * 78)
    for result in results:
        print(
            f"{result['name']:<17} {result['numpy']:>9.4f} "
            f"{result['torch']:>9.4f} {result['speedup']:>8.2f}x  "
            f"{result['parity']}"
        )
        print(f"  {result['dimensions']}")

    if not all(result["parity_ok"] for result in results):
        print("\nERROR: at least one workload failed float64 parity", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
