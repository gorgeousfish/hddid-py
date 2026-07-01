"""Backward-compatible aggregation layer for inference submodules.

This module re-exports all public names from the inference submodules
so that existing code using ``from hddid.inference import ...`` continues
to work without modification.
"""
from __future__ import annotations

import sys
from functools import wraps

import numpy as np  # noqa: F401 – re-exported for backward compat (tests monkeypatch np.linalg)
from statistics import NormalDist  # noqa: F401 – re-exported for backward compat

from ._inference_common import (
    _coerce_finite_numeric,  # noqa: F401 – private, re-exported for backward compat
    InferenceComputationError,
    InvalidInferenceInputError,
    MissingEvaluationGridError,
    NonpositiveVarianceError,
    SingularCovarianceError,
    SparseDirectionInfeasibleError,
)
from .inference_parametric import (
    ParametricInferencePayload,
    estimate_parametric_inference,
    solve_eq42_sparse_direction,
)
from .inference_bootstrap import (
    _matrix_square_root_psd,  # noqa: F401 – private, re-exported for backward compat
    compute_uniform_band_critical_value,
)
from .inference_nonparametric import (
    NonparametricInferencePayload,
    _orthogonal_score_omega_f_hat,  # noqa: F401 – private, re-exported for monkeypatch compat
    diagnose_nonparametric_omega_f,
    estimate_nonparametric_inference as _estimate_nonparametric_inference_impl,
    solve_eq43_projection_matrix,
)

# ---------------------------------------------------------------------------
# Monkeypatch-propagation wrapper for backward compatibility.
# Existing tests monkeypatch private helpers on *this* module and then call
# estimate_nonparametric_inference via this module.  In the original monolithic
# inference.py, both the helper and the caller shared the same module globals,
# so the monkeypatch was visible to the caller.  After the split, the caller
# lives in inference_nonparametric.py with its own globals.  The wrapper below
# temporarily propagates any overridden names to the submodule during the call.
# ---------------------------------------------------------------------------
_NONPARAM_PATCHABLE_NAMES = (
    "solve_eq43_projection_matrix",
    "_orthogonal_score_omega_f_hat",
)


@wraps(_estimate_nonparametric_inference_impl)
def estimate_nonparametric_inference(*args, **kwargs):
    import hddid.inference_nonparametric as _impl_mod

    _this = sys.modules[__name__]
    _patches: dict[str, object] = {}
    for name in _NONPARAM_PATCHABLE_NAMES:
        our = _this.__dict__.get(name)
        if our is None:
            continue
        theirs = getattr(_impl_mod, name, None)
        if our is not theirs:
            _patches[name] = theirs
            setattr(_impl_mod, name, our)
    try:
        return _estimate_nonparametric_inference_impl(*args, **kwargs)
    finally:
        for name, orig in _patches.items():
            setattr(_impl_mod, name, orig)


__all__ = [
    "InferenceComputationError",
    "InvalidInferenceInputError",
    "MissingEvaluationGridError",
    "NonpositiveVarianceError",
    "SingularCovarianceError",
    "SparseDirectionInfeasibleError",
    "ParametricInferencePayload",
    "NonparametricInferencePayload",
    "solve_eq42_sparse_direction",
    "solve_eq43_projection_matrix",
    "diagnose_nonparametric_omega_f",
    "estimate_parametric_inference",
    "estimate_nonparametric_inference",
]
