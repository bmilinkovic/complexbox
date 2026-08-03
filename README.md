# ComplexBox

Hi users! `ComplexBox` is a toolkit for complexity science and emergence wizardry. It brings together multivariate time-series analysis using complexity, criticality, and emergence measures. Currently it holds analyses for dynamical independence, Granger causality and causal emergence, which includes $\Phi$-ID based measures. I attempted to port and extend three MATLAB toolboxes into one NumPy/SciPy API with a torch extension. This was a serious slog and I am not a software engineer, so any feedback is much appreciated.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/license-GPL%20v3-blue.svg)](LICENSE)

The project is currently in beta. There are optional Torch backends which accelerate regular,
batchable kernels without replacing SciPy's DARE, Lyapunov, statistical, or spectral-factorisation routines. This is thanks to the correct prompting to my awesome ChatBOT helpers. 

| Source toolbox | Module | Scope |
|---|---|---|
| [MVGC2](https://github.com/lcbarnett/MVGC2) | `complexbox.mvgc` | VAR and innovations-form state-space Granger causality |
| [SSDI-1](https://github.com/lcbarnett/ssdi) | `complexbox.ssdi` | Dynamical-dependence optimisation on the Grassmannian |
| [ELPH](https://gitlab.com/concog/elph) | `complexbox.elph` | PhiID, entropy, and complexity |

## Installation

This is currently under construction, so, until a package release is published, install from a local clone:

```bash
python -m pip install -e .
```

Optional extras are deliberately separate:

```bash
python -m pip install -e ".[plot]"   # Matplotlib examples
python -m pip install -e ".[fast]"   # Numba kernels
python -m pip install -e ".[torch]"  # batched Torch acceleration
python -m pip install -e ".[dev]"    # tests, docs, and development tools
```

## The SSDI spectral definitions

For an innovations-form state-space transfer function

$
H(z)=I+C(zI-A)^{-1}K, \qquad S(\omega)=H(\omega)H(\omega)^*,
$

and an orthonormal column basis $L\in\mathbb{R}^{n\times m}$, the broadband
objective used is Eq. (24) of Barnett and Seth (2023), written in the package convention as

$
D(L)=\frac{1}{\pi}\int_0^\pi
\log\det\!\left(L^\mathsf{T}S(\omega)L\right)d\omega.
$

`ssdi.trfun2dd` retains that MATLAB API exactly. Its returned frequency samples
are half-log-determinant quadrature terms. The paper's Eq. (25) curve is

$
f_L(\omega)=\log
\frac{\det\!\left(L^\mathsf{T}HH^*L\right)}
{\det\!\left[(L^\mathsf{T}HL)(L^\mathsf{T}HL)^*\right]},
$

available through `ssdi.trfun2dd_pointwise`; Eq. (26) band averages are
available through `ssdi.trfun2dd_band` and `ssdi.trfun2dd_bandgrad`.

Stability is based on eigenvalue radii, never the largest singular value:

$
\rho_A=\max|\operatorname{eig}(A)|<1,\qquad
\rho_B=\max|\operatorname{eig}(A-KC)|<1.
$

`mvgc.ss2fres` uses both radii in its fast power-of-two estimate. Its default
mode, like MVGC2, increases the frequency resolution until the integrated
log-spectrum identity reaches tolerance. VAR stability uses the eigenvalue
radius of the companion matrix through `mvgc.specnorm`.

## Quick starts

### Granger causality from a fitted VAR

```python
import numpy as np
from complexbox import mvgc

rng = np.random.default_rng(0)
A = mvgc.var_rand(5, 3, rho=0.95, rng=rng)
V = mvgc.corr_rand(5, rng=rng)
X, _ = mvgc.var_to_tsdata(A, V, m=2_000, rng=rng)

fit = mvgc.tsdata_to_var(X, p=3, regmode="LWR")
F = mvgc.var_to_pwcgc(fit.A, fit.V)
print(F)  # F[i, j] is causality j -> i; the diagonal is NaN
```

### Multi-restart SSDI optimisation

```python
from complexbox import mvgc, ssdi

A, C, K, rho_b = mvgc.iss_rand(n=9, m=18, rhoa=0.9, rng=rng)
CAK = ssdi.iss2cak(A, C, K)
L0 = ssdi.rand_orthonormal(n=9, m=3, runs=64, rng=rng)

dds, bases, convergence, histories = ssdi.opt_gd_ddx_mruns(
    CAK,
    L0,
    variant=2,
    maxiters=10_000,
)
print(dds[0], convergence[0], rho_b)
```

With the Torch extra installed, the same high-level call can batch restarts (this is super sexy!):

```python
dds, bases, convergence, histories = ssdi.opt_gd_ddx_mruns(
    CAK,
    L0,
    backend="torch",
    device="cpu",  # or an explicitly available CUDA device
    run_chunk_size=32,
    lag_chunk_size=16,
)
```

Torch defaults to float64/complex128 parity mode. It never silently changes the requested device or dtype.

### Pointwise and band-limited spectral DD

```python
fres, integration_error = mvgc.ss2fres(A, C, K, np.eye(9), return_error=True)
H = mvgc.ss2trfun(A, C, K, fres)
omega = np.linspace(0.0, np.pi, fres + 1)

f_omega = ssdi.trfun2dd_pointwise(bases[:, :, 0], H)
alpha_dd, alpha_samples = ssdi.trfun2dd_band(
    bases[:, :, 0], H, omega, band=(0.08 * np.pi, 0.13 * np.pi)
)
```

### Gaussian PhiID and emergence

```python
from complexbox import elph

X_micro = rng.standard_normal((4, 2_000))
phi = elph.phi_id_full(X_micro, tau=1, measure="MMI")
V_macro = X_micro.mean(axis=0)
psi = elph.emergence_psi(X_micro, V_macro, tau=1)

print(phi.synergy, psi.value)
```

## Validation

The repository includes a small deterministic MATLAB R2023b fixture generated
from the original MVGC2 and SSDI-1 code. It checks both spectral radii,
frequency resolution, causal emergence, broadband DD samples and gradient,
Grassmann tangency, and optimiser history. Additional tests cover finite
differences, NumPy/Torch parity, chunking, and per-restart early-stop freezing.

```bash
python -m pytest
ruff check src tests
ruff format --check src tests
```

To regenerate the committed core fixture, put MVGC2 and SSDI-1 on the MATLAB
path and run:

```matlab
cd tools/matlab_fixtures
generate_ssdi_core_reference
```

The larger optional fixture generator is `generate_all_fixtures.m`; large
`.mat` outputs remain ignored.

## Capacity benchmark

Run the deterministic benchmark (it creates no result files) with:

```bash
python tools/benchmark_torch.py
```

On the audited Apple arm64 CPU (2026-08-03, Torch 2.8, NumPy 2.0,
float64/complex128), one median-of-three warmed run produced:

| Workload | NumPy | Torch CPU | Speedup | Maximum parity error |
|---|---:|---:|---:|---:|
| SSDI proxy, 64 restarts | 6.05 s | 0.072 s | 84x | `3.6e-15` |
| SSDI spectral, 32 restarts × 129 bins | 0.65 s | 0.082 s | 8.0x | `3.6e-15` |
| MVGC VAR transfer, 2049 bins | 0.064 s | 0.040 s | 1.6x | `1.7e-15` |

These are capacity-oriented batch measurements, not universal speed promises.
Small one-off calls may not amortise Torch setup, CPU timings depend on thread
and chunk settings, and no CUDA or MPS hardware was available for this audit.

## Conventions and limits

- Time series have shape `(variables, samples)` or `(variables, samples, trials)`.
- VAR arrays have shape `(n, n, lags)`; lag 1 is `A[:, :, 0]`.
- Innovations-form state space uses `(A, C, K, V)` with state dimension `r`
  and observation dimension `n`.
- NumPy/SciPy remains authoritative for DARE/Lyapunov solvers, Wilson spectral
  factorisation, statistics, clustering, LWR/subspace identification, and
  discrete-information algorithms.
- Gaussian PhiID uses exhaustive bipartition search for more than two
  variables, so it is intended for modest dimensions.
- Discrete PhiID currently implements the documented MMI plug-in path for two
  variables.

See [the MATLAB/Python mapping](docs/mapping.md) for the current API coverage.

## License and citation

`complexbox` is distributed under GPL-3.0-only. The complete license is in
[LICENSE](LICENSE), upstream attribution is in [NOTICE](NOTICE), and retained
MIT/BSD notices are in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

If you do use this toolbox please cite it:

```bibtex
@software{complexbox,
  author = {Milinkovic, Borjan},
  title = {ComplexBox: A toolkit for complexity science and emergence wizardry},
  year = {2026},
  note = {ComplexBox: A toolkit for complexity science and emergence wizardry},
  url = {https://github.com/borjan/code/complexbox}
}
```



