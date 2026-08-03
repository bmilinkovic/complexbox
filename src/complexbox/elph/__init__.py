"""``complexbox.elph`` — Integrated information decomposition, causal emergence,
and entropy-rate complexity.

Python adaptations of the ELPH MATLAB toolbox (Mediano, Rosas et al.), with
documented pure-Python substitutes for unavailable Java dependencies.
"""

from __future__ import annotations

from . import emergence, entropy_rate, phi_id, utils
from ._mi import (
    discrete_entropy,
    discrete_mi,
    gaussian_entropy,
    gaussian_local_mi,
    gaussian_mi,
)
from ._torch import (
    emergence_delta_torch,
    emergence_gamma_torch,
    emergence_psi_torch,
    gaussian_local_mi_torch,
    gaussian_mi_torch,
)
from .emergence import (
    EmergenceResult,
    emergence_delta,
    emergence_gamma,
    emergence_psi,
)
from .entropy_rate import lz76, lz76_entropy_rate, state_space_entropy_rate
from .phi_id import (
    ATOM_NAMES,
    PhiIDResult,
    phi_id_full,
    phi_id_full_discrete,
)
from .utils import (
    covariance_selection_model,
    discretize_quantile,
    isdiscrete,
    maximal_cliques,
    quasi_bayes_mi,
    transfer_entropy_gaussian,
)

__all__ = [
    "phi_id",
    "emergence",
    "entropy_rate",
    "utils",
    "PhiIDResult",
    "ATOM_NAMES",
    "phi_id_full",
    "phi_id_full_discrete",
    "EmergenceResult",
    "emergence_psi",
    "emergence_delta",
    "emergence_gamma",
    "emergence_psi_torch",
    "emergence_delta_torch",
    "emergence_gamma_torch",
    "lz76",
    "lz76_entropy_rate",
    "state_space_entropy_rate",
    "gaussian_mi",
    "gaussian_entropy",
    "gaussian_local_mi",
    "gaussian_mi_torch",
    "gaussian_local_mi_torch",
    "discrete_mi",
    "discrete_entropy",
    "isdiscrete",
    "discretize_quantile",
    "maximal_cliques",
    "covariance_selection_model",
    "quasi_bayes_mi",
    "transfer_entropy_gaussian",
]
