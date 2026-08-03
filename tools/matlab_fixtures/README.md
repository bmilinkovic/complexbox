# MATLAB reference fixtures

This directory holds the MATLAB script that produces ground-truth `.mat`
files used by the `complexbox` parity tests.

## Prerequisites

- MATLAB R2019b or newer
- [MVGC2](https://github.com/lcbarnett/MVGC2) on the path
- [SSDI-1](https://github.com/lcbarnett/ssdi) on the path
- [ELPH](https://gitlab.com/concog/elph) on the path (optional; PhiID
  fixtures depend on it)

## Regenerate fixtures

```matlab
>> cd path/to/complexbox/tools/matlab_fixtures
>> generate_all_fixtures
```

Outputs land in `complexbox/tests/fixtures/`. Re-run whenever the source
toolboxes are updated.

## Running the parity tests against fixtures

```bash
pytest -m fixture
```

If the `.mat` files are missing, tests marked `@pytest.mark.fixture` are
skipped automatically.
