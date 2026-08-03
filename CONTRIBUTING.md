# Contributing to complexbox

Contributions are welcome, including bug reports, numerical-parity fixes,
documentation, tests, function ports, and tutorials. The project is in beta, so
claims about MATLAB parity should be tied to a formula, source routine, or
reproducible reference fixture.

## Development setup

Clone the source from the repository location supplied by the maintainer, then
install the editable development environment:

```bash
git clone <repository-url>
cd complexbox
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

Torch is optional and is not part of the `dev` extra:

```bash
python -m pip install -e ".[dev,torch]"
```

## Tests

Plain `pytest` collects the complete suite, including tests marked `fixture`:

```bash
python -m pytest
python -m pytest -m "not fixture"              # exclude marked fixture tests
python -m pytest -m fixture                    # marked MATLAB fixture tests
python -m pytest tests/ssdi/test_matlab_parity.py
python -m pytest --cov=complexbox
```

The repository versions `mvgc_basic.mat`, `ssdi_basic.mat`, and
`ssdi_core_reference.mat`. The first two drive the tests selected by
`-m fixture`; the core SSDI fixture drives the separately named, unmarked parity
module. A missing marked fixture is skipped by `tests/conftest.py`, while the core
fixture is expected to be present.

Regenerate the fixed core fixture with MVGC2 and SSDI-1 on the MATLAB path:

```matlab
cd tools/matlab_fixtures
generate_ssdi_core_reference
```

`generate_all_fixtures.m` produces the larger MVGC, SSDI, and optional ELPH
exports. The generated `elph_basic.mat` is ignored by version control and is not
currently consumed by an ELPH fixture test. Do not describe an exported file as a
parity test until Python assertions actually load it.

## Lint and format

Ruff is the formatter and linter used by the project:

```bash
ruff check src tests
ruff check src tests --fix
ruff format src tests
ruff format --check src tests
```

Run the non-mutating `check` commands before submitting a change. Source lines
use the configured 100-character limit.

## Adding or changing a function

The repository structure follows the upstream MATLAB toolboxes where practical.
Use this checklist:

1. Identify the MATLAB routine, paper equation, or explicitly Python-native
   behavior being implemented.
2. Put the implementation in the appropriate module under `src/complexbox/`.
3. Preserve documented orientation and indexing conventions, or document the
   deliberate difference.
4. Add type hints and a NumPy-style docstring that names the upstream routine or
   equation.
5. Add self-consistency, shape, invalid-input, and numerical-edge-case tests.
6. If MATLAB reference output is available, add assertions marked
   `@pytest.mark.fixture` or extend the deterministic core parity module.
7. Update `docs/mapping.md` with the tested level of parity rather than a broader
   claim.

## SSDI spectral guardrail

Do not conflate the MATLAB SSDI-1 broadband objective with the pointwise spectral
definition in the paper. For an orthonormal column basis \(L\), the legacy
MATLAB-parity path is Eq. (24):

\[
D(L)=\frac{1}{\pi}\int_0^\pi
\log\det\!\left[L^\mathsf{T}H(\omega)H(\omega)^*L\right]d\omega.
\]

`trfun2dd` implements that objective and returns half-log-determinant quadrature
terms. The paper's Eq. (25) ratio is implemented separately by
`trfun2dd_pointwise`; Eq. (26) band integration and its gradient are implemented
by `trfun2dd_band` and `trfun2dd_bandgrad`. Changes to any of these routines need
finite-difference gradient tests and, where applicable, MATLAB fixture coverage.

Frequency-resolution and stability code must use eigenvalue spectral radii.
For innovations-form state space, test both \(\rho(A)\) and \(\rho(A-KC)\); for
VAR models, test the companion matrix.

## Torch contributions

Torch is an optional accelerator, not the numerical authority. New Torch paths
should follow the established contract:

- import Torch lazily so `import complexbox` works without the extra;
- preserve the public NumPy input/output shapes and return NumPy values;
- use float64/complex128 as the default parity mode;
- expose device, dtype, and useful chunk-size controls explicitly;
- raise on unavailable devices or unsupported dtypes instead of silently
  changing them;
- compare CPU float64 results with the NumPy implementation, including gradients,
  chunking, and optimiser histories where relevant;
- make accelerator-specific tests conditional on actual device availability.

Torch parity means agreement with the tested NumPy path. It does not create a
second independent MATLAB oracle, and not every SciPy-based routine is suitable
for a Torch rewrite.

## Numerical-parity language

Use the narrowest accurate description in code comments, documentation, and
change proposals:

- **machine precision**: a deterministic MATLAB fixture or analytical reference
  verifies the named inputs at approximately `1e-10` or tighter;
- **statistical parity**: seeded simulations or estimators agree within stated
  sampling uncertainty, not sample-for-sample across different RNGs;
- **algorithmic parity**: the method and converged result agree within stated
  tolerances, but SciPy solvers, linkage choices, stopping rules, or random starts
  may prevent bitwise-identical trajectories;
- **NumPy/Torch parity**: the optional backend agrees with the package's NumPy
  reference for the tested dtype and device.

Current fixtures cover selected MVGC and SSDI paths, not every entry in the
mapping table. ELPH has important known boundaries: some JIDT-dependent behavior
is replaced by Python algorithms, Gaussian higher-dimensional MIB selection is
exhaustive and intended for modest dimensions, and discrete PhiID currently
implements only the two-variable MMI plug-in path. State such limitations rather
than presenting the package as mathematically exact across all three upstream
toolboxes.

## Style and conventions

- Prefer NumPy/SciPy primitives when they preserve the required numerical
  behavior.
- Document lag indexing, broadcast shapes, Hermitian assumptions, and matrix
  orientation when they are not obvious.
- Time series are variables by samples, VAR lag 1 is `A[:, :, 0]`, and causality
  entry `F[i, j]` means `j` to `i`.
- Avoid unrelated formatting or generated-file churn in a focused change.

## Licensing

`complexbox` is GPL-3.0-only, following the upstream licensing obligations.
Contributions are accepted under the same terms.
