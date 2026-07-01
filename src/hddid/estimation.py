from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from .results import HDDIDResult, ResultDiagnostics
from .score import ScorePayload


class Eq31SolverConvergenceError(RuntimeError):
    """Raised when the Eq. (3.1) numerical solver fails to converge."""

    def __init__(self, message: str, *, metadata: dict[str, object]) -> None:
        super().__init__(message)
        self.metadata = dict(metadata)


class Eq31ProjectionRankError(RuntimeError):
    """Raised when an Eq. (3.1) projection design is not identified."""

    def __init__(self, message: str, *, metadata: dict[str, object]) -> None:
        super().__init__(message)
        self.metadata = dict(metadata)


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


def _coerce_nonnegative_finite(name: str, value: float) -> float:
    """Validate and return a non-negative finite scalar."""
    if _contains_boolean_or_string_alias(value):
        raise ValueError(f"{name} must be numeric")
    raw_value = np.asarray(value)
    if raw_value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    numeric = float(raw_value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _coerce_positive_finite(name: str, value: float) -> float:
    """Validate and return a strictly positive finite scalar."""
    if _contains_boolean_or_string_alias(value):
        raise ValueError(f"{name} must be numeric")
    raw_value = np.asarray(value)
    if raw_value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    numeric = float(raw_value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _coerce_positive_integer(name: str, value: int) -> int:
    """Validate and return a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _coerce_nonnegative_integer(name: str, value: int) -> int:
    """Validate and return a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _soft_threshold(value: float, threshold: float) -> float:
    """Apply the soft-thresholding (shrinkage) operator.

    Computes S(value, threshold) = sign(value) * max(|value| - threshold, 0),
    the proximal operator of the L1 penalty used in coordinate-descent
    Lasso.  This operator arises from the KKT condition of the Eq. (8)
    penalized least-squares problem.

    Parameters
    ----------
    value : float
        Input scalar (typically a coordinate-wise correlation).
    threshold : float
        Non-negative shrinkage parameter (typically lambda/2).

    Returns
    -------
    float
        Soft-thresholded value.

    Notes
    -----
    The soft-thresholding operator is the proximal map of the L1 norm:
        S(x, t) = argmin_y { (1/2)(x - y)^2 + t|y| }
    See Eq. (8) in Ning, Peng & Tao (2020), arXiv preprint arXiv:2009.03151.
    """
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _project_onto_basis(
    basis_matrix: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project ``values`` onto a sieve basis via ordinary least squares.

    Computes the Frisch-Waugh-Lovell (FWL) projection of ``values``
    onto the column space of ``basis_matrix``.  This is used to partial
    out the nonparametric sieve component from both the score S_hat and
    the covariates X before solving for the parametric coefficient beta
    in the Eq. (8) partially-linear sieve regression.

    The projection is:
        fitted = B (B'B)^{-1} B' values

    where ``B`` is the basis matrix and the coefficients are obtained
    via least-squares (``np.linalg.lstsq`` for numerical stability).

    Parameters
    ----------
    basis_matrix : ndarray of shape (n, L)
        Sieve basis matrix B (e.g., polynomial or trigonometric basis
        evaluated at the nonparametric variable Z).  Must have full
        column rank for a unique projection.
    values : ndarray of shape (n,) or (n, k)
        Values to project.  Can be a vector (e.g., S_hat) or a matrix
        (e.g., covariates X).

    Returns
    -------
    coefficients : ndarray of shape (L,) or (L, k)
        Least-squares regression coefficients of ``values`` on
        ``basis_matrix``.
    fitted : ndarray of shape (n,) or (n, k)
        Fitted projection ``basis_matrix @ coefficients``.

    Raises
    ------
    UserWarning
        If the design matrix is ill-conditioned (condition number > 1e12)
        or rank-deficient.

    Notes
    -----
    Implements the Frisch-Waugh partial-out step of Eq. (8) in
    Ning, Peng & Tao (2020), arXiv preprint arXiv:2009.03151.  The residualized
    quantities ``values - fitted`` are used in the subsequent
    coordinate-descent Lasso for beta.
    """
    design = np.asarray(basis_matrix, dtype=float)
    cond_number = np.linalg.cond(design)
    if cond_number > 1e12:
        warnings.warn(
            f"Design matrix condition number {cond_number:.2e} exceeds threshold "
            f"1e12; projection results may be numerically unstable.",
            UserWarning,
            stacklevel=2,
        )
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(
        design, values, rcond=None
    )
    n_rows, n_cols = design.shape
    effective_rank = int(rank)
    min_dim = min(n_rows, n_cols)
    if effective_rank < min_dim:
        warnings.warn(
            f"Least-squares design is rank deficient (rank={effective_rank}, "
            f"min(m,n)={min_dim}, condition_number={cond_number:.2e}); "
            f"returned coefficients may not be unique.",
            UserWarning,
            stacklevel=2,
        )
    fitted = basis_matrix @ coefficients
    return coefficients, fitted


def _assert_full_column_rank_for_eq31_projection(
    basis_matrix: np.ndarray,
) -> None:
    """Verify that the Eq. (8) sieve basis has full column rank."""
    shape = tuple(int(dimension) for dimension in np.asarray(basis_matrix).shape)
    rank = int(np.linalg.matrix_rank(np.asarray(basis_matrix, dtype=float)))
    n_columns = shape[1] if len(shape) == 2 else 0
    if rank < n_columns:
        raise Eq31ProjectionRankError(
            "Eq. (3.1) sieve projection requires full column rank",
            metadata={
                "matrix_name": "basis_valid_full",
                "matrix_shape": shape,
                "matrix_rank": rank,
                "required_rank": n_columns,
            },
        )


def _assert_full_column_rank_for_unpenalized_eq31_beta(
    projection_x_valid: np.ndarray,
) -> None:
    """Verify that the residualized design has full column rank for OLS."""
    shape = tuple(
        int(dimension) for dimension in np.asarray(projection_x_valid).shape
    )
    rank = int(np.linalg.matrix_rank(np.asarray(projection_x_valid, dtype=float)))
    n_columns = shape[1] if len(shape) == 2 else 0
    if rank < n_columns:
        raise Eq31ProjectionRankError(
            "Eq. (3.1) unpenalized beta block requires full column rank",
            metadata={
                "matrix_name": "projection_x_valid",
                "matrix_shape": shape,
                "matrix_rank": rank,
                "required_rank": n_columns,
            },
        )


def _sklearn_available() -> bool:
    """Return True if scikit-learn is installed."""
    try:
        import sklearn.linear_model  # noqa: F401
    except ImportError:
        return False
    return True


def _solve_beta_block_sklearn(
    projection_x_valid: np.ndarray,
    s_tilde_valid: np.ndarray,
    *,
    penalty_lambda: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Solve the beta block of Eq(3.1) using sklearn's Lasso.

    This is an alternative to the native coordinate descent that leverages
    sklearn's optimized Cython implementation for faster convergence.

    The penalty_lambda maps to sklearn's alpha as:
        sklearn_alpha = penalty_lambda / 2
    because the native path minimizes: (1/(2n))||y - Xw||^2 + (penalty_lambda/2)||w||_1
    and sklearn minimizes:             (1/(2n))||y - Xw||^2 + alpha * ||w||_1
    """
    try:
        from sklearn.linear_model import Lasso
    except ImportError:
        raise ImportError(
            "solver='sklearn' requires scikit-learn. "
            "Install with: pip install scikit-learn"
        ) from None

    n_obs, n_features = projection_x_valid.shape
    if n_features == 0:
        return np.zeros(0, dtype=float), {
            "solver": "sklearn-empty-beta-block",
            "coordinate_descent_iterations": 0,
            "coordinate_descent_converged": True,
        }

    if penalty_lambda == 0.0:
        _assert_full_column_rank_for_unpenalized_eq31_beta(projection_x_valid)
        beta_hat, *_ = np.linalg.lstsq(
            projection_x_valid, s_tilde_valid, rcond=None
        )
        return np.asarray(beta_hat, dtype=float), {
            "solver": "sklearn-residualized-ols",
            "coordinate_descent_iterations": 0,
            "coordinate_descent_converged": True,
        }

    # Convert penalty: builtin minimizes (1/(2n))||y-Xb||^2 + (lambda/2)||b||_1
    # sklearn minimizes (1/(2n))||y-Xb||^2 + alpha*||b||_1
    sklearn_alpha = penalty_lambda / 2.0

    model = Lasso(
        alpha=sklearn_alpha,
        fit_intercept=False,  # Already handled by projection
        max_iter=max_iter,
        tol=tol,
        warm_start=False,
    )
    model.fit(projection_x_valid, s_tilde_valid)
    beta_hat = model.coef_.astype(float)

    return beta_hat, {
        "solver": "sklearn-lasso",
        "coordinate_descent_iterations": int(model.n_iter_),
        "coordinate_descent_converged": model.n_iter_ < max_iter,
        "coordinate_descent_threshold": 0.5 * penalty_lambda,
    }


def _solve_beta_block(
    projection_x_valid: np.ndarray,
    s_tilde_valid: np.ndarray,
    *,
    penalty_lambda: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Solve the beta sub-problem of Eq. (8) via coordinate-descent Lasso.

    Minimizes the penalized objective:

        (1/(2n)) || S_tilde - X_tilde beta ||^2 + (lambda/2) ||beta||_1

    where ``X_tilde`` is the Frisch-Waugh residualized design matrix and
    ``S_tilde`` is the residualized score.  When ``penalty_lambda == 0``,
    the problem reduces to ordinary least squares (requiring full column
    rank of the design).

    The coordinate-descent algorithm updates each beta_j cyclically:
        beta_j <- S(<X_j, r_j> / n, lambda/2) / (||X_j||^2 / n)
    where r_j is the partial residual and S(·,·) is the soft-thresholding
    operator.

    Parameters
    ----------
    projection_x_valid : ndarray of shape (n, p)
        Frisch-Waugh residualized covariates (X projected orthogonal
        to the sieve basis).  Must be finite.
    s_tilde_valid : ndarray of shape (n,)
        Frisch-Waugh residualized doubly-robust score.  Must be finite.
    penalty_lambda : float
        L1 penalty parameter lambda >= 0.  When 0, OLS is used.
    max_iter : int
        Maximum number of coordinate-descent sweep iterations.
    tol : float
        Convergence tolerance on the maximum coordinate update.

    Returns
    -------
    beta_hat : ndarray of shape (p,)
        Estimated parametric coefficients.
    metadata : dict
        Solver diagnostics including:
        - ``solver``: solver identifier string
        - ``coordinate_descent_iterations``: number of sweeps performed
        - ``coordinate_descent_converged``: whether convergence was reached
        - ``coordinate_descent_threshold``: effective shrinkage lambda/2

    Raises
    ------
    Eq31ProjectionRankError
        If ``penalty_lambda == 0`` and the design matrix lacks full
        column rank.

    Notes
    -----
    Implements the beta sub-problem of Eq. (8) in Ning, Peng & Tao
    (2020), arXiv preprint arXiv:2009.03151.  The penalty form ``(lambda/2)||beta||_1``
    matches the paper's objective
    ``E_n[(S_i - X_i'beta - f_n(Z_i))^2] + lambda||beta||_1``
    after the Frisch-Waugh partial-out of f_n.
    """
    n_obs, n_features = projection_x_valid.shape
    if n_features == 0:
        return np.zeros(0, dtype=float), {
            "solver": "empty-beta-block",
            "coordinate_descent_iterations": 0,
            "coordinate_descent_converged": True,
        }

    if penalty_lambda == 0.0:
        _assert_full_column_rank_for_unpenalized_eq31_beta(projection_x_valid)
        beta_hat, *_ = np.linalg.lstsq(projection_x_valid, s_tilde_valid, rcond=None)
        return np.asarray(beta_hat, dtype=float), {
            "solver": "residualized-ols",
            "coordinate_descent_iterations": 0,
            "coordinate_descent_converged": True,
        }

    beta_hat = np.zeros(n_features, dtype=float)
    residual = np.asarray(s_tilde_valid, dtype=float).copy()
    column_second_moment = np.mean(projection_x_valid**2, axis=0)
    coordinate_threshold = 0.5 * penalty_lambda
    converged = False
    iterations = 0

    for iteration in range(1, max_iter + 1):
        max_update = 0.0
        for column in range(n_features):
            if column_second_moment[column] <= np.finfo(float).eps:
                continue
            partial_residual = (
                residual + projection_x_valid[:, column] * beta_hat[column]
            )
            correlation = float(
                np.mean(projection_x_valid[:, column] * partial_residual)
            )
            updated = (
                _soft_threshold(correlation, coordinate_threshold)
                / column_second_moment[column]
            )
            residual = partial_residual - projection_x_valid[:, column] * updated
            max_update = max(max_update, abs(updated - beta_hat[column]))
            beta_hat[column] = updated
        iterations = iteration
        if max_update <= tol:
            converged = True
            break

    return beta_hat, {
        "solver": "coordinate-descent-lasso",
        "coordinate_descent_iterations": iterations,
        "coordinate_descent_converged": converged,
        "coordinate_descent_threshold": coordinate_threshold,
    }


def _lasso_kkt_violation_max(
    projection_x_valid: np.ndarray,
    s_tilde_valid: np.ndarray,
    beta_hat: np.ndarray,
    *,
    coordinate_threshold: float,
) -> float:
    """Compute the maximum KKT condition violation for the Lasso problem.

    Checks the stationarity conditions of the Eq. (8) Lasso sub-problem:

    - For active coordinates (beta_j != 0):
        |X_j'(S - X*beta)/n - threshold * sign(beta_j)| should be ~0
    - For inactive coordinates (beta_j == 0):
        |X_j'(S - X*beta)/n| should be <= threshold

    Parameters
    ----------
    projection_x_valid : ndarray of shape (n, p)
        Frisch-Waugh residualized covariates.
    s_tilde_valid : ndarray of shape (n,)
        Frisch-Waugh residualized score.
    beta_hat : ndarray of shape (p,)
        Estimated coefficients from coordinate descent.
    coordinate_threshold : float
        Effective shrinkage threshold (lambda/2).

    Returns
    -------
    float
        Maximum KKT violation across all coordinates.  A value of 0
        indicates full optimality; values within ``tol`` indicate
        acceptable convergence.

    Notes
    -----
    References: Eq. (8) KKT conditions in Ning, Peng & Tao (2020),
    arXiv preprint arXiv:2009.03151.
    """
    if beta_hat.shape[0] == 0:
        return 0.0
    residual = np.asarray(s_tilde_valid, dtype=float) - (
        np.asarray(projection_x_valid, dtype=float) @ np.asarray(beta_hat, dtype=float)
    )
    correlations = np.mean(
        np.asarray(projection_x_valid, dtype=float) * residual[:, None],
        axis=0,
    )
    violations = []
    for coefficient, correlation in zip(beta_hat, correlations):
        if abs(float(coefficient)) <= 1e-12:
            violations.append(max(abs(float(correlation)) - coordinate_threshold, 0.0))
        else:
            violations.append(
                abs(
                    float(correlation)
                    - coordinate_threshold * np.sign(float(coefficient))
                )
            )
    return float(max(violations)) if violations else 0.0


def _aggregate_result_diagnostics(
    score_payload: ScorePayload,
    optimization_metadata: dict[str, object],
) -> ResultDiagnostics:
    """Aggregate per-fold diagnostics into a single ResultDiagnostics object."""
    n_holdout_raw = 0
    n_trimmed_propensity = 0
    n_valid_holdout = 0
    trim_lowers: list[float] = []
    trim_uppers: list[float] = []

    for diagnostic in score_payload.fold_diagnostics:
        if diagnostic.n_holdout_raw is not None:
            n_holdout_raw += int(diagnostic.n_holdout_raw)
        if diagnostic.n_trimmed_propensity is not None:
            n_trimmed_propensity += int(diagnostic.n_trimmed_propensity)
        if diagnostic.n_valid_holdout is not None:
            n_valid_holdout += int(diagnostic.n_valid_holdout)
        if diagnostic.trim_lower is not None:
            trim_lowers.append(float(diagnostic.trim_lower))
        if diagnostic.trim_upper is not None:
            trim_uppers.append(float(diagnostic.trim_upper))

    if n_holdout_raw == 0:
        n_holdout_raw = int(score_payload.valid_mask.shape[0])
    if n_valid_holdout == 0:
        n_valid_holdout = int(np.sum(score_payload.valid_mask))
    if n_trimmed_propensity == 0:
        n_trimmed_propensity = int(np.sum(~score_payload.valid_mask))

    return ResultDiagnostics(
        basis_family=score_payload.basis_family,
        basis_degree=score_payload.basis_degree,
        oracle_lane=score_payload.oracle_lane,
        fold_diagnostics=list(score_payload.fold_diagnostics),
        n_holdout_raw=n_holdout_raw,
        n_trimmed_propensity=n_trimmed_propensity,
        n_valid_holdout=n_valid_holdout,
        trim_lower=min(trim_lowers) if trim_lowers else None,
        trim_upper=max(trim_uppers) if trim_uppers else None,
        optimization_metadata=dict(optimization_metadata),
    )


@dataclass(slots=True)
class EstimationPayload:
    """Second-stage Eq. (3.1) partially-linear sieve regression output.

    Contains the estimated parametric coefficients beta, sieve
    coefficients gamma, fitted nonparametric function values, and
    residuals from S_hat = X'beta + f(Z) + epsilon.

    Attributes
    ----------
    beta_hat : ndarray of float, shape (p,)
        Estimated high-dimensional parametric coefficients.
    gamma_hat : ndarray of float, shape (L,)
        Estimated sieve basis coefficients for f(z).
    fitted_f_valid : ndarray of float, shape (n_valid,)
        Fitted nonparametric component f_hat on valid observations.
    second_stage_prediction_valid : ndarray of float, shape (n_valid,)
        Full second-stage prediction X'beta + f_hat on valid observations.
    residual_valid : ndarray of float, shape (n_valid,)
        Second-stage residuals S_hat - X'beta - f_hat.
    projection_x_valid : ndarray of float, shape (n_valid, p)
        Covariates projected orthogonal to the sieve basis
        (Frisch-Waugh residualized design).
    f_hat_at_z0 : ndarray of float, shape (G,)
        Estimated f(z) evaluated at the z0 grid points.
    optimization_metadata : dict
        Solver diagnostics including convergence status, penalty,
        and dimension information.
    """

    beta_hat: np.ndarray
    gamma_hat: np.ndarray
    fitted_f_valid: np.ndarray
    second_stage_prediction_valid: np.ndarray
    residual_valid: np.ndarray
    projection_x_valid: np.ndarray
    f_hat_at_z0: np.ndarray
    optimization_metadata: dict[str, object]

    def __post_init__(self) -> None:
        self.beta_hat = _coerce_vector("beta_hat", self.beta_hat)
        self.gamma_hat = _coerce_vector("gamma_hat", self.gamma_hat)
        self.fitted_f_valid = _coerce_vector("fitted_f_valid", self.fitted_f_valid)
        self.second_stage_prediction_valid = _coerce_vector(
            "second_stage_prediction_valid",
            self.second_stage_prediction_valid,
        )
        self.residual_valid = _coerce_vector("residual_valid", self.residual_valid)
        self.projection_x_valid = _coerce_matrix(
            "projection_x_valid", self.projection_x_valid
        )
        self.f_hat_at_z0 = _coerce_vector("f_hat_at_z0", self.f_hat_at_z0)
        self.optimization_metadata = dict(self.optimization_metadata)

        for name, values in (
            ("beta_hat", self.beta_hat),
            ("gamma_hat", self.gamma_hat),
            ("fitted_f_valid", self.fitted_f_valid),
            ("second_stage_prediction_valid", self.second_stage_prediction_valid),
            ("residual_valid", self.residual_valid),
            ("projection_x_valid", self.projection_x_valid),
            ("f_hat_at_z0", self.f_hat_at_z0),
        ):
            _require_finite(name, values)

        n_valid = self.fitted_f_valid.shape[0]
        if self.second_stage_prediction_valid.shape[0] != n_valid:
            raise ValueError(
                "second_stage_prediction_valid must align with fitted_f_valid"
            )
        if self.residual_valid.shape[0] != n_valid:
            raise ValueError("residual_valid must align with fitted_f_valid")
        if self.projection_x_valid.shape[0] != n_valid:
            raise ValueError("projection_x_valid must align with fitted_f_valid")

        n_valid_metadata = self.optimization_metadata.get("n_valid_obs")
        if n_valid_metadata is None:
            raise ValueError("n_valid_obs must be provided for EstimationPayload")
        n_valid_value = _coerce_positive_integer("n_valid_obs", n_valid_metadata)
        if n_valid_value != n_valid:
            raise ValueError("n_valid_obs must equal fitted_f_valid length")

        beta_dimension_metadata = self.optimization_metadata.get("beta_dimension")
        if beta_dimension_metadata is None:
            raise ValueError("beta_dimension must be provided for EstimationPayload")
        beta_dimension = _coerce_nonnegative_integer(
            "beta_dimension",
            beta_dimension_metadata,
        )
        if self.beta_hat.shape[0] != beta_dimension:
            raise ValueError("beta_hat must align with beta_dimension")
        if self.projection_x_valid.shape[1] != beta_dimension:
            raise ValueError("projection_x_valid must align with beta_dimension")
        if self.f_hat_at_z0.shape[0] == 0:
            raise ValueError("f_hat_at_z0 must contain at least one evaluation point")

        basis_dimension_metadata = self.optimization_metadata.get(
            "basis_dimension_full"
        )
        if basis_dimension_metadata is None:
            raise ValueError(
                "basis_dimension_full must be provided for EstimationPayload"
            )
        basis_dimension = _coerce_positive_integer(
            "basis_dimension_full",
            basis_dimension_metadata,
        )
        if self.gamma_hat.shape[0] != basis_dimension:
            raise ValueError("gamma_hat must align with basis_dimension_full")


def estimate_eq31_mainline(
    score_payload: ScorePayload,
    *,
    penalty_lambda: float = 0.0,
    max_iter: int = 1_000,
    tol: float = 1e-10,
    solver: str = "sklearn",
) -> tuple[EstimationPayload, HDDIDResult]:
    """Solve the Eq. (3.1) partially-linear sieve regression.

    Estimates beta and the nonparametric function f(z) from the
    second-stage model:

        S_hat = X' beta + f(Z) + residual

    The procedure projects out the sieve basis from both S_hat and X
    (Frisch-Waugh), then solves for beta via (optionally penalized)
    least squares. The sieve coefficients gamma are recovered by
    projecting (S_hat - X beta) onto the basis.

    Parameters
    ----------
    score_payload : ScorePayload
        Doubly-robust score payload from :func:`build_score_payload`,
        containing S_hat, the valid-sample mask, basis matrices, and
        the covariate matrix restricted to valid observations.
    penalty_lambda : float, default 0.0
        L1 penalty on beta. When 0.0, ordinary least squares is used
        (requires full column rank in the projected design). When
        positive, coordinate-descent Lasso is applied.
    max_iter : int, default 1000
        Maximum iterations for coordinate descent.
    tol : float, default 1e-10
        Convergence tolerance for the maximum coordinate update.
    solver : str, default "sklearn"
        Solver backend. "sklearn" delegates to scikit-learn's Lasso
        implementation; "builtin" or "native" use the pure-NumPy
        coordinate-descent implementation. If "sklearn" is requested
        but scikit-learn is not installed, a warning is issued and the
        native solver is used as a fallback.

    Returns
    -------
    estimation_payload : EstimationPayload
        Contains beta_hat, gamma_hat, fitted values, residuals, and
        solver metadata.
    result : HDDIDResult
        Result object populated with parametric and nonparametric
        point estimates (no inference yet).

    Raises
    ------
    ValueError
        If penalty_lambda is negative, tol is non-positive, or
        solver is not recognized.
    Eq31ProjectionRankError
        If the sieve basis or the projected design matrix lacks
        full column rank (when penalty_lambda == 0).
    Eq31SolverConvergenceError
        If coordinate descent fails to converge and KKT conditions
        are not satisfied.

    Notes
    -----
    Implements Eq. (3.1) from Ning, Peng, and Tao (2020).
    Reference: Ning, Peng, and Tao (2020), arXiv preprint arXiv:2009.03151.
    """
    penalty_value = _coerce_nonnegative_finite("penalty_lambda", penalty_lambda)
    max_iter_value = _coerce_positive_integer("max_iter", max_iter)
    tol_value = _coerce_positive_finite("tol", tol)

    native_solvers = frozenset({"builtin", "native"})
    sklearn_solvers = frozenset({"sklearn"})
    if solver not in native_solvers | sklearn_solvers:
        raise ValueError(
            f"solver must be 'sklearn', 'builtin', or 'native', got {solver!r}"
        )

    use_sklearn = solver in sklearn_solvers
    if use_sklearn and not _sklearn_available():
        warnings.warn(
            "solver='sklearn' requested but scikit-learn is not installed; "
            "falling back to the native coordinate-descent solver.",
            UserWarning,
            stacklevel=2,
        )
        use_sklearn = False

    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    x_valid = np.asarray(score_payload.x_valid, dtype=float)
    s_hat_valid = np.asarray(score_payload.s_hat_valid, dtype=float)
    _assert_full_column_rank_for_eq31_projection(basis_valid_full)

    _, projected_x_valid = _project_onto_basis(basis_valid_full, x_valid)
    projection_x_valid = x_valid - projected_x_valid
    _, projected_s_valid = _project_onto_basis(basis_valid_full, s_hat_valid)
    s_tilde_valid = s_hat_valid - projected_s_valid

    solve_fn = (
        _solve_beta_block_sklearn if use_sklearn else _solve_beta_block
    )
    beta_hat, solver_metadata = solve_fn(
        projection_x_valid,
        s_tilde_valid,
        penalty_lambda=penalty_value,
        max_iter=max_iter_value,
        tol=tol_value,
    )
    if solver_metadata["solver"] == "coordinate-descent-lasso":
        coordinate_threshold = float(solver_metadata["coordinate_descent_threshold"])
        kkt_violation_max = _lasso_kkt_violation_max(
            projection_x_valid,
            s_tilde_valid,
            beta_hat,
            coordinate_threshold=coordinate_threshold,
        )
        kkt_tolerance = max(tol_value, 1e-8)
        kkt_satisfied = kkt_violation_max <= kkt_tolerance
        solver_metadata = {
            **solver_metadata,
            "coordinate_descent_kkt_violation_max": kkt_violation_max,
            "coordinate_descent_kkt_tolerance": kkt_tolerance,
            "coordinate_descent_kkt_satisfied": kkt_satisfied,
            "coordinate_descent_convergence_criterion": (
                "coordinate-update"
                if bool(solver_metadata["coordinate_descent_converged"])
                else "kkt-optimality"
                if kkt_satisfied
                else "not-converged"
            ),
            "coordinate_descent_converged": bool(
                solver_metadata["coordinate_descent_converged"]
            )
            or kkt_satisfied,
        }

    if solver_metadata["solver"] == "coordinate-descent-lasso" and not bool(
        solver_metadata["coordinate_descent_converged"]
    ):
        failure_metadata: dict[str, object] = {
            **solver_metadata,
            "penalty_lambda": penalty_value,
            "max_iter": max_iter_value,
            "tol": tol_value,
            "n_valid_obs": int(s_hat_valid.shape[0]),
            "beta_dimension": int(x_valid.shape[1]),
        }
        raise Eq31SolverConvergenceError(
            "Eq. (3.1) coordinate-descent Lasso failed to converge",
            metadata=failure_metadata,
        )
    gamma_hat, fitted_f_valid = _project_onto_basis(
        basis_valid_full,
        s_hat_valid - x_valid @ beta_hat,
    )
    second_stage_prediction_valid = x_valid @ beta_hat + fitted_f_valid
    residual_valid = s_hat_valid - second_stage_prediction_valid
    f_hat_at_z0 = np.asarray(score_payload.evaluation_basis @ gamma_hat, dtype=float)

    optimization_metadata = {
        **solver_metadata,
        "penalty_target": "beta-only",
        "penalty_lambda": penalty_value,
        "basis_intercept_dropped": bool(score_payload.intercept_dropped_for_design),
        "n_valid_obs": int(s_hat_valid.shape[0]),
        "beta_dimension": int(x_valid.shape[1]),
        "basis_dimension_full": int(basis_valid_full.shape[1]),
        "basis_dimension_design": int(score_payload.basis_design_valid.shape[1]),
    }

    estimation_payload = EstimationPayload(
        beta_hat=beta_hat,
        gamma_hat=gamma_hat,
        fitted_f_valid=fitted_f_valid,
        second_stage_prediction_valid=second_stage_prediction_valid,
        residual_valid=residual_valid,
        projection_x_valid=projection_x_valid,
        f_hat_at_z0=f_hat_at_z0,
        optimization_metadata=optimization_metadata,
    )
    result = HDDIDResult(
        parametric_estimates={"beta_hat": estimation_payload.beta_hat},
        nonparametric_estimates={
            "gamma_hat": estimation_payload.gamma_hat,
            "f_hat_at_z0": estimation_payload.f_hat_at_z0,
        },
        standard_errors={},
        intervals={},
        diagnostics=_aggregate_result_diagnostics(score_payload, optimization_metadata),
    )
    return estimation_payload, result
