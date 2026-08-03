"""complexbox: Python adaptations of MVGC2, SSDI-1, and ELPH.

This package provides three submodules:

- :mod:`complexbox.mvgc`: Multivariate Granger causality for VAR and innovations-form
  state-space models (port of MVGC2 by Lionel Barnett).
- :mod:`complexbox.ssdi`: Dynamical dependence / state-space causal emergence
  optimisation over the Grassmannian (port of SSDI-1).
- :mod:`complexbox.elph`: Integrated information decomposition (PhiID), causal
  emergence (Psi/Delta/Gamma), and entropy-rate complexity measures
  (port of ELPH by Mediano, Rosas, et al.).

All three submodules use a common data convention: time series are NumPy arrays
of shape ``(n, m)`` for a single trial or ``(n, m, N)`` for ``N`` trials, where
``n`` is the number of variables and ``m`` is the number of time samples.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("complexbox")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"

from complexbox import elph, mvgc, ssdi

__all__ = ["__version__", "mvgc", "ssdi", "elph"]
