"""High-Dimensional Difference-in-Differences estimation.

This package implements the semiparametric HD-DID estimator of
Ning, Peng, and Tao (2020), providing:

- Cross-fitted nuisance estimation with propensity trimming
- Doubly-robust score construction (Eq. 2.5/2.7)
- Partially-linear sieve regression (Eq. 3.1)
- Debiased parametric inference via sparse directions (Eq. 4.1-4.2)
- Nonparametric uniform confidence bands (Eq. 4.3)

Reference: Ning, Peng, and Tao (2020), arXiv preprint arXiv:2009.03151.
"""

from importlib import metadata as importlib_metadata

from .basis import bspline_sieve_basis, polynomial_sieve_basis, suggest_basis_degree, trigonometric_sieve_basis
from .estimation import (
    Eq31ProjectionRankError,
    Eq31SolverConvergenceError,
    EstimationPayload,
    estimate_eq31_mainline,
)
from .estimator import HDDIDFit, fit_hddid
from .inference import (
    InferenceComputationError,
    InvalidInferenceInputError,
    MissingEvaluationGridError,
    NonparametricInferencePayload,
    NonpositiveVarianceError,
    ParametricInferencePayload,
    SingularCovarianceError,
    SparseDirectionInfeasibleError,
    diagnose_nonparametric_omega_f,
    estimate_nonparametric_inference,
    estimate_parametric_inference,
    solve_eq43_projection_matrix,
    solve_eq42_sparse_direction,
)
from .nuisance import CrossfitNuisanceEstimator, NuisancePayload
from .results import (
    ConfidenceInterval,
    FoldDiagnostics,
    HDDIDResult,
    ResultDiagnostics,
    UniformBand,
)
from .score import ScorePayload, build_score_payload
from .splitting import CrossfitFold, CrossfitPlan, make_crossfit_splits

try:
    _pkg_version = importlib_metadata.version("hddid")
except importlib_metadata.PackageNotFoundError:
    _pkg_version = "0.1.0"

# Prevent stale editable-install placeholder from overriding source version
if _pkg_version == "0.0.0":
    __version__ = "0.1.0"
else:
    __version__ = _pkg_version

__all__ = [
    "__version__",
    "bspline_sieve_basis",
    "polynomial_sieve_basis",
    "suggest_basis_degree",
    "trigonometric_sieve_basis",
    "CrossfitFold",
    "CrossfitPlan",
    "make_crossfit_splits",
    "CrossfitNuisanceEstimator",
    "NuisancePayload",
    "ScorePayload",
    "build_score_payload",
    "Eq31ProjectionRankError",
    "Eq31SolverConvergenceError",
    "EstimationPayload",
    "estimate_eq31_mainline",
    "HDDIDFit",
    "fit_hddid",
    "InferenceComputationError",
    "InvalidInferenceInputError",
    "MissingEvaluationGridError",
    "SingularCovarianceError",
    "SparseDirectionInfeasibleError",
    "NonpositiveVarianceError",
    "ParametricInferencePayload",
    "NonparametricInferencePayload",
    "solve_eq42_sparse_direction",
    "solve_eq43_projection_matrix",
    "diagnose_nonparametric_omega_f",
    "estimate_parametric_inference",
    "estimate_nonparametric_inference",
    "ConfidenceInterval",
    "UniformBand",
    "FoldDiagnostics",
    "HDDIDResult",
    "ResultDiagnostics",
]
