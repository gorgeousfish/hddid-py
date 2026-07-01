from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from .inputs import ValidatedHDDIDData
from .nuisance import NuisancePayload
from .results import FoldDiagnostics, normalize_oracle_lane


def _contains_boolean_or_string_alias(value: Any) -> bool:
    """Check whether ``value`` contains booleans or strings (recursively)."""
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_ or value.dtype.kind in {"S", "U"}:
            return True
        if value.dtype == object:
            return any(
                _contains_boolean_or_string_alias(item) for item in value.flat
            )
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean_or_string_alias(item) for item in value)
    return False


def _coerce_vector(name: str, values: np.ndarray) -> np.ndarray:
    """Coerce input to a 1-D float array, rejecting booleans and strings."""
    if _contains_boolean_or_string_alias(values):
        raise ValueError(f"{name} must be numeric, not boolean or string")
    raw_array = np.asarray(values)
    if _contains_boolean_or_string_alias(raw_array):
        raise ValueError(f"{name} must be numeric, not boolean or string")
    array = raw_array.astype(float)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    raise ValueError(f"{name} must be one-dimensional")


def _coerce_matrix(name: str, values: np.ndarray) -> np.ndarray:
    """Coerce input to a 2-D float array, rejecting booleans and strings."""
    if _contains_boolean_or_string_alias(values):
        raise ValueError(f"{name} must be numeric, not boolean or string")
    raw_array = np.asarray(values)
    if _contains_boolean_or_string_alias(raw_array):
        raise ValueError(f"{name} must be numeric, not boolean or string")
    array = raw_array.astype(float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    return array


def _require_finite(name: str, values: np.ndarray) -> None:
    """Raise ValueError if any element of ``values`` is non-finite."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")


def _coerce_basis_degree(basis_family: str, basis_degree: int) -> int:
    """Validate basis degree given the basis family."""
    if isinstance(basis_degree, bool) or not isinstance(basis_degree, Integral):
        raise ValueError("basis_degree must be an integer")
    degree = int(basis_degree)
    if degree < 0:
        raise ValueError("basis_degree must be non-negative")
    if basis_family == "trigonometric" and degree < 1:
        raise ValueError("basis_degree must be positive for trigonometric basis")
    return degree


def _coerce_mask(values: np.ndarray) -> np.ndarray:
    """Coerce input to a 1-D boolean array."""
    raw_array = np.asarray(values)
    if raw_array.dtype != np.bool_:
        raise ValueError("valid_mask must contain only boolean values")
    array = raw_array.astype(bool, copy=False)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    raise ValueError("valid_mask must be one-dimensional")


def _coerce_fold_ids(values: np.ndarray) -> np.ndarray:
    """Coerce input to a 1-D array of positive integers."""
    raw_array = np.asarray(values)
    if raw_array.dtype == np.bool_:
        raise ValueError("fold_ids must contain positive integer values")
    if np.issubdtype(raw_array.dtype, np.integer):
        array = raw_array.astype(int, copy=False)
    elif raw_array.dtype == object:
        flat_values = raw_array.reshape(-1)
        if any(isinstance(value, bool) for value in flat_values):
            raise ValueError("fold_ids must contain positive integer values")
        if not all(isinstance(value, Integral) for value in flat_values):
            raise ValueError("fold_ids must contain positive integer values")
        array = raw_array.astype(int, copy=False)
    else:
        raise ValueError("fold_ids must contain positive integer values")
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 1:
        result = array
    elif array.ndim == 2 and array.shape[1] == 1:
        result = array[:, 0]
    else:
        raise ValueError("fold_ids must be one-dimensional")
    if np.any(result <= 0):
        raise ValueError("fold_ids must contain positive integer values")
    return result


def _coerce_bool(name: str, value: bool) -> bool:
    """Validate that ``value`` is a boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _drop_intercept_if_present(basis_matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    """Remove the leading intercept column if it is all ones.

    Returns
    -------
    basis_without_intercept : ndarray
        Basis matrix with intercept removed (or unchanged if no intercept).
    intercept_dropped : bool
        Whether an intercept column was found and removed.
    """
    if basis_matrix.shape[1] == 0:
        return basis_matrix.copy(), False
    if np.allclose(basis_matrix[:, 0], 1.0):
        return basis_matrix[:, 1:], True
    return basis_matrix.copy(), False


def _require_close(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> None:
    """Raise ValueError if ``actual`` is not close to ``expected``."""
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(f"{name} must match the Eq. (3.1) basis construction")


@dataclass(slots=True)
class ScorePayload:
    """Doubly-robust influence-function score for Eq. (3.1) estimation.

    Assembles the score S_hat = rho * (DeltaY - (1-pi)*phi1 - pi*phi0)
    from cross-fitted nuisance estimates and organizes the basis matrices
    needed for the partially-linear sieve regression.

    Attributes
    ----------
    delta_y : ndarray of float, shape (n,)
        First-differenced outcome Y1 - Y0.
    s_hat : ndarray of float, shape (n,)
        Full-sample doubly-robust score.
    s_hat_valid : ndarray of float, shape (n_valid,)
        Score restricted to propensity-trimmed observations.
    valid_mask : ndarray of bool, shape (n,)
        Boolean mask identifying trimmed-valid observations.
    fold_ids : ndarray of int, shape (n,)
        Cross-fitting fold assignments.
    fold_diagnostics : list of FoldDiagnostics
        Per-fold trimming diagnostics.
    basis_family : str
        Sieve basis family.
    basis_degree : int
        Sieve truncation parameter.
    oracle_lane : str
        Computational lane for R-parity verification.
    basis_matrix : ndarray of float, shape (n, L)
        Full-sample sieve basis matrix.
    x_valid : ndarray of float, shape (n_valid, p)
        Covariates restricted to valid observations.
    basis_valid_full : ndarray of float, shape (n_valid, L)
        Basis matrix restricted to valid observations.
    basis_design_valid : ndarray of float, shape (n_valid, L')
        Basis design matrix (intercept dropped if present).
    evaluation_basis : ndarray of float, shape (G, L)
        Basis evaluated at the z0 grid for f(z0) prediction.
    pi_hat : ndarray of float, shape (n,)
        Propensity scores from nuisance estimation.
    phi0_hat : ndarray of float, shape (n,)
        Conditional mean estimates for controls.
    phi1_hat : ndarray of float, shape (n,)
        Conditional mean estimates for treated.
    rho_hat : ndarray of float, shape (n,)
        Propensity weights.
    intercept_dropped_for_design : bool
        Whether the leading intercept column was removed from
        basis_valid_full to form basis_design_valid.
    """

    delta_y: np.ndarray
    s_hat: np.ndarray
    s_hat_valid: np.ndarray
    valid_mask: np.ndarray
    fold_ids: np.ndarray
    fold_diagnostics: list[FoldDiagnostics]
    basis_family: str
    basis_degree: int
    oracle_lane: str
    basis_matrix: np.ndarray
    x_valid: np.ndarray
    basis_valid_full: np.ndarray
    basis_design_valid: np.ndarray
    evaluation_basis: np.ndarray
    pi_hat: np.ndarray
    phi0_hat: np.ndarray
    phi1_hat: np.ndarray
    rho_hat: np.ndarray
    intercept_dropped_for_design: bool

    def __post_init__(self) -> None:
        self.delta_y = _coerce_vector("delta_y", self.delta_y)
        self.s_hat = _coerce_vector("s_hat", self.s_hat)
        self.s_hat_valid = _coerce_vector("s_hat_valid", self.s_hat_valid)
        self.valid_mask = _coerce_mask(self.valid_mask)
        self.fold_ids = _coerce_fold_ids(self.fold_ids)
        self.basis_matrix = _coerce_matrix("basis_matrix", self.basis_matrix)
        self.x_valid = _coerce_matrix("x_valid", self.x_valid)
        self.basis_valid_full = _coerce_matrix(
            "basis_valid_full", self.basis_valid_full
        )
        self.basis_design_valid = _coerce_matrix(
            "basis_design_valid", self.basis_design_valid
        )
        self.evaluation_basis = _coerce_matrix(
            "evaluation_basis", self.evaluation_basis
        )
        self.pi_hat = _coerce_vector("pi_hat", self.pi_hat)
        self.phi0_hat = _coerce_vector("phi0_hat", self.phi0_hat)
        self.phi1_hat = _coerce_vector("phi1_hat", self.phi1_hat)
        self.rho_hat = _coerce_vector("rho_hat", self.rho_hat)
        self.basis_family = str(self.basis_family).strip().lower()
        self.basis_degree = _coerce_basis_degree(
            self.basis_family,
            self.basis_degree,
        )
        self.oracle_lane = normalize_oracle_lane(
            basis_family=self.basis_family,
            oracle_lane=self.oracle_lane,
        )
        self.intercept_dropped_for_design = _coerce_bool(
            "intercept_dropped_for_design",
            self.intercept_dropped_for_design,
        )

        n_obs = self.delta_y.shape[0]
        for name, values in (
            ("delta_y", self.delta_y),
            ("s_hat", self.s_hat),
            ("s_hat_valid", self.s_hat_valid),
            ("basis_matrix", self.basis_matrix),
            ("x_valid", self.x_valid),
            ("basis_valid_full", self.basis_valid_full),
            ("basis_design_valid", self.basis_design_valid),
            ("evaluation_basis", self.evaluation_basis),
            ("pi_hat", self.pi_hat),
            ("phi0_hat", self.phi0_hat),
            ("phi1_hat", self.phi1_hat),
            ("rho_hat", self.rho_hat),
        ):
            _require_finite(name, values)

        for name, candidate in (
            ("s_hat", self.s_hat),
            ("valid_mask", self.valid_mask),
            ("fold_ids", self.fold_ids),
            ("pi_hat", self.pi_hat),
            ("phi0_hat", self.phi0_hat),
            ("phi1_hat", self.phi1_hat),
            ("rho_hat", self.rho_hat),
        ):
            if candidate.shape[0] != n_obs:
                raise ValueError(f"{name} must align with delta_y")
        if self.basis_matrix.shape[0] != n_obs:
            raise ValueError("basis_matrix must align with delta_y")
        if np.any((self.pi_hat <= 0.0) | (self.pi_hat >= 1.0)):
            raise ValueError("pi_hat must lie strictly between 0 and 1")
        expected_s_hat = self.rho_hat * (
            self.delta_y
            - (1.0 - self.pi_hat) * self.phi1_hat
            - self.pi_hat * self.phi0_hat
        )
        if not np.allclose(self.s_hat, expected_s_hat, rtol=1e-10, atol=1e-10):
            raise ValueError("s_hat must match the Eq. (3.1) score formula")

        n_valid = int(np.sum(self.valid_mask))
        if n_valid <= 0:
            raise ValueError("ScorePayload requires at least one valid observation")
        if self.s_hat_valid.shape[0] != n_valid:
            raise ValueError("s_hat_valid must align with valid_mask")
        if not np.allclose(
            self.s_hat_valid,
            self.s_hat[self.valid_mask],
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError("s_hat_valid must match s_hat on valid_mask")
        if self.x_valid.shape[0] != n_valid:
            raise ValueError("x_valid must align with valid_mask")
        if self.basis_valid_full.shape[0] != n_valid:
            raise ValueError("basis_valid_full must align with valid_mask")
        if self.basis_design_valid.shape[0] != n_valid:
            raise ValueError("basis_design_valid must align with valid_mask")
        if self.basis_design_valid.shape[1] > self.basis_valid_full.shape[1]:
            raise ValueError(
                "basis_design_valid cannot have more columns than basis_valid_full"
            )
        if self.evaluation_basis.shape[0] == 0:
            raise ValueError("evaluation_basis must contain at least one grid point")
        expected_basis_valid_full = self.basis_matrix[self.valid_mask]
        _require_close(
            "basis_valid_full",
            self.basis_valid_full,
            expected_basis_valid_full,
        )
        expected_basis_design_valid, expected_intercept_dropped = (
            _drop_intercept_if_present(expected_basis_valid_full)
        )
        _require_close(
            "basis_design_valid",
            self.basis_design_valid,
            expected_basis_design_valid,
        )
        if self.intercept_dropped_for_design != expected_intercept_dropped:
            raise ValueError(
                "intercept_dropped_for_design must match the Eq. (3.1) basis construction"
            )


def build_score_payload(
    data: ValidatedHDDIDData,
    nuisance_payload: NuisancePayload,
) -> ScorePayload:
    """Construct the doubly-robust score for Eq. (3.1) estimation.

    Assembles the influence-function-based score S_hat from the
    estimated nuisance parameters (propensity score, conditional
    outcome means) and the observed outcomes, following Eq. (2.5)
    and (2.7) of the paper.

    The score is defined as:

        S_hat_i = rho_i * (Delta Y_i - (1 - pi_i) phi1_i - pi_i phi0_i)

    where rho_i = (D_i - pi_i) / [pi_i (1 - pi_i)] is the
    propensity weight.

    Parameters
    ----------
    data : ValidatedHDDIDData
        Validated input data containing outcomes, treatment, covariates,
        the nonparametric variable z, evaluation grid z0, and basis
        configuration.
    nuisance_payload : NuisancePayload
        Cross-fitted nuisance estimates including pi_hat (propensity
        scores), phi0_hat and phi1_hat (conditional means), rho_hat
        (propensity weights), valid_mask, and fold assignments.

    Returns
    -------
    ScorePayload
        Contains the full-sample and valid-sample scores (s_hat,
        s_hat_valid), basis matrices for the sieve projection,
        the evaluation basis for computing f(z0), and all nuisance
        components needed downstream.

    Raises
    ------
    ValueError
        If nuisance_payload dimensions do not align with data, or
        if basis_family / basis_degree disagree between inputs.

    Notes
    -----
    Implements the doubly-robust score from Eq. (2.5) and (2.7) of
    Ning, Peng, and Tao (2020). The propensity trimming mask is
    inherited from nuisance_payload.valid_mask.

    Reference: Ning, Peng, and Tao (2020), arXiv preprint arXiv:2009.03151.
    """
    if nuisance_payload.pi_hat.shape[0] != data.n_obs:
        raise ValueError("nuisance payload must align with validated inputs")
    if nuisance_payload.basis_family != data.basis_family:
        raise ValueError("basis_family must agree between inputs and nuisance payload")
    if nuisance_payload.basis_degree != data.basis_degree:
        raise ValueError("basis_degree must agree between inputs and nuisance payload")
    expected_rho_hat = (
        np.asarray(data.treat, dtype=float) - nuisance_payload.pi_hat
    ) / (nuisance_payload.pi_hat * (1.0 - nuisance_payload.pi_hat))
    if not np.allclose(
        nuisance_payload.rho_hat,
        expected_rho_hat,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "rho_hat must match the Eq. (2.5) propensity score weight"
        )

    delta_y = np.asarray(data.y1 - data.y0, dtype=float)
    s_hat = nuisance_payload.rho_hat * (
        delta_y
        - (1.0 - nuisance_payload.pi_hat) * nuisance_payload.phi1_hat
        - nuisance_payload.pi_hat * nuisance_payload.phi0_hat
    )

    valid_mask = np.asarray(nuisance_payload.valid_mask, dtype=bool)
    basis_matrix = data.build_basis_matrix()
    basis_valid_full = np.asarray(basis_matrix[valid_mask], dtype=float)
    basis_design_valid, intercept_dropped = _drop_intercept_if_present(basis_valid_full)

    return ScorePayload(
        delta_y=delta_y,
        s_hat=s_hat,
        s_hat_valid=s_hat[valid_mask],
        valid_mask=valid_mask,
        fold_ids=nuisance_payload.fold_ids,
        fold_diagnostics=list(nuisance_payload.fold_diagnostics),
        basis_family=data.basis_family,
        basis_degree=data.basis_degree,
        oracle_lane=nuisance_payload.oracle_lane,
        basis_matrix=basis_matrix,
        x_valid=np.asarray(data.x[valid_mask], dtype=float),
        basis_valid_full=basis_valid_full,
        basis_design_valid=basis_design_valid,
        evaluation_basis=data.build_evaluation_basis(),
        pi_hat=nuisance_payload.pi_hat,
        phi0_hat=nuisance_payload.phi0_hat,
        phi1_hat=nuisance_payload.phi1_hat,
        rho_hat=nuisance_payload.rho_hat,
        intercept_dropped_for_design=intercept_dropped,
    )
