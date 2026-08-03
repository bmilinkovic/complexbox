# MATLAB ↔ Python function mapping

This page is an API and evidence map, not a blanket claim that every port is
numerically identical to MATLAB. Status applies only to the functions in that
row and to the current automated tests.

## Evidence labels

- **MATLAB fixture**: compared directly with deterministic output saved from
  the named MATLAB toolbox. The stated tolerance is the assertion tolerance,
  not a promise for every input or platform.
- **Torch↔NumPy**: the optional Torch path is compared with the NumPy reference
  path. This does not independently establish MATLAB parity.
- **Self-check**: covered by an identity, round trip, invariant, analytic case,
  or finite-difference test, but not by an external MATLAB fixture.
- **Ported, unverified**: a source-level port exists, but no direct MATLAB
  fixture currently protects that mapping.
- **Approximation**: an intentional Python substitute differs from an upstream
  dependency or estimator.
- **Statistical**: validation is distributional or Monte Carlo; individual
  random draws are not expected to match.
- **Not ported**: no equivalent is provided.

Direct fixture assertions currently live in `tests/mvgc/test_fixtures.py`,
`tests/ssdi/test_fixtures.py`, and `tests/ssdi/test_matlab_parity.py`. The
fixture generator can also export `elph_basic.mat`, but that file is neither
versioned nor consumed by a direct ELPH parity test, so no ELPH row is labeled
**MATLAB fixture**.

## MVGC2 → `complexbox.mvgc`

| MATLAB / MVGC2 | Python | Evidence |
|---|---|---|
| `tsdata_to_var` | `mvgc.tsdata_to_var` | Ported, unverified |
| `var_to_tsdata` | `mvgc.var_to_tsdata` | Statistical; simulation-shape and recovery self-checks |
| `var_to_ss` | `mvgc.var_to_ss` | MATLAB fixture (≤ `1e-12`) |
| `var_to_autocov` | `mvgc.var_to_autocov` | MATLAB fixture (≤ `1e-12`) |
| `autocov_to_var` | `mvgc.autocov_to_var` | MATLAB fixture (≤ `1e-10`) |
| `ss_to_autocov`; covariance/autocovariance conversions | Same names under `mvgc` | Ported, unverified |
| `var2fres`; spectral integration check | `mvgc.var2fres`; `mvgc.var_check_fres` | MATLAB fixture for fast/adaptive resolution and integration error (≤ `1e-14`) |
| `ss2fres`; spectral integration check | `mvgc.ss2fres`; `mvgc.ss_check_fres` | MATLAB fixture for fast/adaptive resolution and integration error (≤ `1e-14`) |
| `specnorm` | `mvgc.specnorm` | MATLAB fixture for `A` and `A-KC` spectral radii (≤ `1e-14`) |
| `var2trfun`, `var2itrfun`, `ss2trfun`, `ss2itrfun` | Same names under `mvgc` | Self-check; optional Torch↔NumPy coverage |
| `var_to_cpsd`, `ss_to_cpsd` | Same names under `mvgc` | Self-check; optional Torch↔NumPy coverage |
| `tsdata_to_cpsd` | `mvgc.tsdata_to_cpsd` | Approximation using SciPy Welch/CSD |
| `tsdata_to_varmo`, `tsdata_to_ssmo`, `tsdata_to_ss` | Same names under `mvgc` | Ported, unverified; stochastic estimation where applicable |
| `var_to_pwcgc` | `mvgc.var_to_pwcgc` | MATLAB fixture on off-diagonal values (≤ `1e-10`) |
| VAR/SS time-domain MVGC and pairwise GC | `var_to_mvgc`, `ss_to_mvgc`, `ss_to_pwcgc` | Ported, unverified |
| VAR/SS spectral MVGC and pairwise GC | `*_to_smvgc`, `*_to_spwcgc` | Ported, unverified |
| Autocovariance/CPSD GC paths | `autocov_to_*gc`, `cpsd_to_*gc` | Ported, unverified; finite-lag or Wilson-factorisation error may apply |
| In/out, causal-density, and groupwise GC | `*_to_iogc`, `*_to_cggc`, `*_to_gwcgc`, `*_to_gwcggc`, `*_to_gwiogc` and spectral forms | Ported/adapted, unverified; the SS `gwcggc`/`gwiogc` functions are Python analogues of MVGC2's VAR routines because matching SS files are not present upstream |
| Covariance and CPSD mutual-information families | `cov_to_*mi`, `cpsd_to_*mi` | Ported, unverified; MVGC2's doubled-MI convention is retained where documented |
| `cpsd_specfac` | `mvgc.cpsd_specfac` | Self-check of Wilson factorisation; no direct MATLAB fixture |
| `mvgc_pval`, `mvgc_cval` | Same names under `mvgc` | MATLAB fixture for F/LR p-values and F critical value (≤ `1e-10`) |
| Remaining distribution, information-criterion, diagnostic, and nonparametric helpers | Same or documented names under `mvgc` | Ported, unverified |
| `var_rand`, `corr_rand`, `iss_rand` | Same names under `mvgc`, with `rng=` | Statistical and structural self-checks; raw draws differ by RNG |
| `parcov`, `logdet`, `dlyap_aitr`, `dlyap`, `mdare`, `demean`, `symmetrise`, grouping helpers | Same names under `mvgc` | Self-check or ported, unverified; no direct fixture unless listed above |

### Frequency-resolution criterion

`var_check_fres` and `ss_check_fres` evaluate the MVGC2 spectral
log-determinant integration identity. Adaptive `var2fres`/`ss2fres` select the
first configured power-of-two grid within tolerance. Fast `ss2fres` uses

\[
\rho=\max\{\rho(A),\rho(A-KC)\},
\]

where `rho` is the maximum eigenvalue modulus (spectral radius), not the
largest singular value. The one-argument `ss2fres(A)` compatibility form can
only use `rho(A)` and therefore cannot perform the full innovations-form
check.

## SSDI-1 → `complexbox.ssdi`

For identity innovations covariance and a column-basis projection `L`, the
broadband objective implemented by `trfun2dd` is paper Eq. 24:

\[
D(L)=\frac{1}{\pi}\int_0^\pi
\log\det\!\left[L^\mathsf{T}H(\omega)H(\omega)^*L\right]d\omega.
\]

The paper's pointwise spectral DD (Eq. 25) and normalized band objective
(Eq. 26) are

\[
f_L(\omega)=\log\frac{\det[L^\mathsf{T}HH^*L]}
{\det[(L^\mathsf{T}HL)(L^\mathsf{T}HL)^*]},\qquad
D_{[\omega_1,\omega_2]}(L)=\frac{1}{\omega_2-\omega_1}
\int_{\omega_1}^{\omega_2}f_L(\omega)d\omega.
\]

`trfun2dd(L, H)` returns `(D, d)`, where `d` contains the legacy MATLAB
half-log-determinant quadrature samples. Those samples are not Eq. 25's
`f_L(omega)`; use `trfun2dd_pointwise` for the pointwise quantity.

| MATLAB / paper | Python | Evidence |
|---|---|---|
| `iss2cak` | `ssdi.iss2cak` | MATLAB fixture (≤ `2e-13`) |
| `cak2ddx` | `ssdi.cak2ddx` | MATLAB fixture (≤ `2e-13`) |
| `cak2ddxgrad` | `ssdi.cak2ddxgrad` | MATLAB fixture (≤ `2e-12`) |
| `iss2dd` | `ssdi.iss2dd` | MATLAB fixture (≤ `2e-12`) |
| `iss2ce_precomp`, `iss2ce` | Same names under `ssdi` | MATLAB fixture with and without precompute (≤ `2e-13`) |
| Eq. 24 `trfun2dd`, Appendix-D gradient | `ssdi.trfun2dd`, `ssdi.trfun2ddgrad` | MATLAB fixture: objective/samples ≤ `2e-14`, gradient ≤ `2e-13` |
| Eq. 25 pointwise spectral DD | `ssdi.trfun2dd_pointwise` | Self-check against Eq. 26/full-band integration; no MATLAB fixture |
| Eq. 26 band DD and gradient | `ssdi.trfun2dd_band`, `ssdi.trfun2dd_bandgrad` | Self-check by broadband identity and finite differences; no MATLAB fixture |
| `opt_gd2_ddx` | `ssdi.opt_gd2_ddx` | MATLAB fixture for objective, iterations, and final subspace |
| `opt_gd2_dds` | `ssdi.opt_gd2_dds` | MATLAB fixture for objective, convergence state, history, and final subspace |
| `opt_gd1_ddx`, `opt_gd1_dds` | Same names under `ssdi` | Ported, unverified |
| `opt_gd_ddx_mruns`, `opt_gd_dds_mruns` | Same names under `ssdi` | Self-check; optional Torch↔NumPy coverage |
| Orthonormal/subspace transforms and Grassmann geometry | `orthonormalise`, `transform_*`, `L2Q`, `Q2L`, `gmetric`, `gmetrics*`, `subspace*`, `plucker` | Self-check of geometry, round trips, and tangent projections |
| `habeta`, `habetax`, HAXA/Beta statistics | Same or documented names under `ssdi` | Statistical or closed-form self-check; HAXA uses MATLAB principal-angle/complement reflection semantics; no direct MATLAB fixture |
| `Lcluster`, `Lhcluster` | Same names under `ssdi` | Ported, unverified; SciPy linkage is used for hierarchy |
| `iss_perfect_dd`, `dds_check` | Same names under `ssdi` | Self-check, including zero-DD constructions |
| `tnet9a` … `tnet9d`, `tneter`, `tnetmod` | Same names under `ssdi` | Structural self-check; random generators remain statistical |
| Python extension | `ssdi.tnet243` | Self-check of the documented 2+4+3 topology; no MATLAB counterpart claimed |

## ELPH → `complexbox.elph`

| MATLAB / ELPH | Python | Evidence |
|---|---|---|
| `PhiIDFull` | `elph.phi_id_full` | Approximation for `D > 2`: exhaustive Python bipartition search replaces JIDT; atom-sum self-check |
| `PhiIDFullDiscrete` | `elph.phi_id_full_discrete` | Approximation: documented two-variable MMI plug-in path only |
| `EmergencePsi` | `elph.emergence_psi` | IID-micro self-check; no direct MATLAB fixture |
| `EmergenceDelta`, `EmergenceGamma` | `elph.emergence_delta`, `elph.emergence_gamma` | Ported, unverified independently; optional Torch↔NumPy coverage |
| `GaussianMI` and local Gaussian MI dependency | `elph.gaussian_mi`, `elph.gaussian_local_mi` | Analytic/self-consistency checks; optional Torch↔NumPy coverage |
| `DiscreteMI` | `elph.discrete_mi` | Ported, unverified |
| `LZ76` | `elph.lz76` | Self-check on known sequences; no direct MATLAB fixture |
| LZ76 entropy-rate normalisation | `elph.lz76_entropy_rate` | Ported, unverified |
| `StateSpaceEntropyRate` | `elph.state_space_entropy_rate` | Ported, unverified |
| `TransferEntropy` | `elph.transfer_entropy_gaussian` | Approximation: closed-form Gaussian replacement for JIDT |
| `CovarianceSelectionModel` | `elph.covariance_selection_model` | Ported, unverified |
| `MaximalCliques` | `elph.maximal_cliques` | Structural self-check; no direct MATLAB fixture |
| `QuasiBayesMI` | `elph.quasi_bayes_mi` | Approximation: Miller–Madow substitute for the NSB estimator |
| `isdiscrete`, `discretize_oct` | `elph.isdiscrete`, `elph.discretize_quantile` | Ported, unverified |
| `CTWEntropyRate` | Not ported | External Java VMM dependency |
| `NBSmultistats`, `CompositeNBStest` | Not ported | External NBS dependency |

## Optional Torch coverage

Torch is lazy-loaded and NumPy arrays remain the public input/output boundary.
Requested devices or dtypes fail explicitly rather than silently falling back.

| Module | Accelerated entry points | Current evidence |
|---|---|---|
| MVGC | `var2trfun`, `var2itrfun`, `ss2trfun`, `ss2itrfun`, `var_to_cpsd`, `ss_to_cpsd` with `backend="torch"`, `device=`, `dtype=`, and `batch_size=` | Torch↔NumPy on CPU in float64, frequency chunking, public dispatch, and float32 tolerance checks |
| SSDI | Public `opt_gd_ddx_mruns` and `opt_gd_dds_mruns` with `backend="torch"`; internal batched value/gradient kernels in `ssdi._torch` | Torch↔NumPy on CPU in float64/complex128 for proxy/spectral values, gradients, histories, final subspaces, and chunking |
| ELPH | `gaussian_mi_torch`, `gaussian_local_mi_torch`, `emergence_psi_torch`, `emergence_delta_torch`, `emergence_gamma_torch` | Torch↔NumPy on CPU in float64, batching/broadcasting/chunking, local outputs, and float32 tolerance checks |

These checks establish CPU implementation parity, not a CUDA/MPS validation
result. The reproducible CPU workloads and machine-specific timing caveats are
recorded in `tools/benchmark_torch.py` and the project README.

## Calling-convention notes

- **Data orientation**: time-series data is `(variables, observations)` or
  `(variables, observations, trials)`, matching the MATLAB toolboxes.
- **Causality direction**: in pairwise GC matrices, `F[i, j]` is causality from
  `j` to `i`, matching MVGC2.
- **Random seeds**: MATLAB's default generator is MT19937, while
  `numpy.random.default_rng` uses PCG64. The same integer seed therefore does
  not produce the same samples. Cross-language comparisons should save the
  generated model/data arrays or explicitly use a compatible generator.
- **Frequency grid**: one-sided spectral arrays use
  `omega = np.linspace(0, np.pi, fres + 1)`, so a resolution of `fres` produces
  `fres + 1` bins including both endpoints.
