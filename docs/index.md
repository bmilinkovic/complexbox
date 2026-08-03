# complexbox

`complexbox` is a beta Python toolkit for multivariate time-series analysis,
dynamical independence, and information emergence. It brings selected algorithms
from three related MATLAB toolboxes into one NumPy/SciPy API while retaining their
matrix orientation and numerical conventions:

- [MVGC2](https://github.com/lcbarnett/MVGC2) for multivariate Granger causality;
- [SSDI-1](https://github.com/lcbarnett/ssdi) for dynamical-dependence optimisation;
- [ELPH](https://gitlab.com/concog/elph) for PhiID, emergence, entropy, and
  complexity measures.

The NumPy/SciPy implementation is the scientific reference. Optional Torch paths
accelerate kernels that batch cleanly; they do not replace the SciPy DARE,
Lyapunov, spectral-factorisation, statistical, clustering, or identification
routines.

## Quick links

- [Project README](https://github.com/bmilinkovic/complexbox#readme)
- [MATLAB/Python mapping](mapping.md)
- [Contributing](https://github.com/bmilinkovic/complexbox/blob/main/CONTRIBUTING.md)
- [MVGC2 tutorial notebook](https://github.com/bmilinkovic/complexbox/blob/main/examples/mvgc_tutorial.ipynb)
- [SSDI-1 tutorial notebook](https://github.com/bmilinkovic/complexbox/blob/main/examples/ssdi_tutorial.ipynb)

## Install from source

No package-index release is assumed. From a local source checkout:

```bash
cd complexbox
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install only the extras needed for a workflow:

```bash
python -m pip install -e ".[plot]"   # Matplotlib examples
python -m pip install -e ".[fast]"   # Numba kernels
python -m pip install -e ".[torch]"  # batched Torch kernels
python -m pip install -e ".[dev]"    # tests, documentation, and tooling
```

## Module overview

### `complexbox.mvgc`

VAR and innovations-form state-space estimation; conversions among VAR,
autocovariance, CPSD, and state-space forms; time- and frequency-domain Granger
causality; model-order selection; and statistical utilities. Torch dispatch via
`backend="torch"` is available for VAR/state-space transfer functions, inverse
transfer functions, and CPSDs.

### `complexbox.ssdi`

CAK proxy and spectral dynamical-dependence objectives and gradients,
Grassmannian optimisation, multi-restart optimisation, subspace geometry,
clustering, beta statistics, network generators, and perfect-DD utilities.
The multi-restart proxy and spectral optimisers accept `backend="torch"` and
explicit run, lag, or frequency chunk sizes.

### `complexbox.elph`

Gaussian and discrete PhiID paths, Psi/Delta/Gamma emergence measures,
state-space entropy rate, LZ76 complexity, Gaussian transfer entropy, and graph
utilities. Optional Torch functions accelerate batched Gaussian MI, local MI,
and Gaussian emergence calculations.

## SSDI spectral definitions

For an innovations-form transfer function

\[
H(z)=I+C(zI-A)^{-1}K, \qquad S(\omega)=H(\omega)H(\omega)^*,
\]

and an orthonormal column basis \(L\in\mathbb{R}^{n\times m}\), MATLAB SSDI-1's
broadband objective is Eq. (24) of Barnett and Seth (2023), expressed in the
package convention as

\[
D(L)=\frac{1}{\pi}\int_0^\pi
\log\det\!\left(L^\mathsf{T}S(\omega)L\right)d\omega.
\]

`ssdi.trfun2dd` and `ssdi.trfun2ddgrad` preserve this MATLAB behavior. In
particular, the array returned by `trfun2dd` contains half-log-determinant
trapezoid terms; those values are not the paper's pointwise spectral DD.

The paper's Eq. (25) pointwise curve is

\[
f_L(\omega)=\log
\frac{\det\!\left(L^\mathsf{T}HH^*L\right)}
{\det\!\left[(L^\mathsf{T}HL)(L^\mathsf{T}HL)^*\right]},
\]

implemented by `ssdi.trfun2dd_pointwise`. Equation (26) band averages and their
Grassmann gradient are implemented by `ssdi.trfun2dd_band` and
`ssdi.trfun2dd_bandgrad`.

Model stability and fast frequency-resolution estimates use eigenvalue spectral
radii, not largest singular values. State-space resolution uses both
\(\rho(A)\) and \(\rho(A-KC)\); VAR stability uses the companion-matrix radius.

## Validation and current parity boundary

Run the complete suite with:

```bash
python -m pytest
ruff check src tests
ruff format --check src tests
```

The source tree includes three versioned MATLAB reference fixtures:

- `mvgc_basic.mat` checks selected VAR/autocovariance/state-space conversions,
  pairwise-conditional GC, and analytical statistics;
- `ssdi_basic.mat` checks selected proxy, exact, spectral, gradient, and proxy
  optimisation paths;
- `ssdi_core_reference.mat` checks spectral radii, frequency resolution, causal
  emergence, MATLAB broadband spectral samples and gradient, and a deterministic
  spectral optimiser history.

`pytest -m fixture` selects the marked MVGC and SSDI fixture tests. The additional
core SSDI checks are unmarked and can be run directly with
`pytest tests/ssdi/test_matlab_parity.py`. The optional `elph_basic.mat` export is
produced by the larger MATLAB generator, but there is no ELPH fixture assertion
for it yet.

These fixtures establish parity only for the named paths and inputs; they are not
a blanket proof that every mapped function is bitwise identical to MATLAB.
SciPy substitutions, finite-lag approximations, stochastic simulations, and
optimisers without a fixed initial state should be compared with appropriate
tolerances. ELPH's Python-only replacements for JIDT-dependent behavior are
algorithmic approximations in some cases: Gaussian PhiID uses an exhaustive MIB
search at higher dimensions, and discrete PhiID is currently limited to the MMI
plug-in path for two variables.

Torch tests compare the accelerated results with the NumPy reference in
float64/complex128 on CPU, including chunking and optimiser lifecycle behavior.
Accelerator availability is platform-dependent and is not an independent MATLAB
parity claim. Device and dtype requests are explicit and never silently fall
back.

## Citation

If you use `complexbox` in academic work, cite the relevant upstream toolbox and
methods paper. For SSDI, cite Barnett and Seth, *Physical Review E* 108, 014304
(2023), <https://doi.org/10.1103/PhysRevE.108.014304>.
