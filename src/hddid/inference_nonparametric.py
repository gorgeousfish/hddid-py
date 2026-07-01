from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import ClassVar

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_diag

from .estimation import EstimationPayload, _aggregate_result_diagnostics
from .results import ConfidenceInterval, HDDIDResult, UniformBand
from .score import ScorePayload

from ._inference_common import (
    InvalidInferenceInputError,
    MissingEvaluationGridError,
    NonpositiveVarianceError,
    SingularCovarianceError,
    SparseDirectionInfeasibleError,
    _INFERENCE_EPS,
    _PSD_EIGENVALUE_TOL,
    _assert_full_rank,
    _checked_solve,
    _coerce_basis_degree,
    _coerce_finite_numeric,
    _coerce_matrix,
    _coerce_nonnegative_finite,
    _coerce_optional_nonnegative_integer,
    _coerce_positive_integer,
    _coerce_uniform_band_random_state,
    _coerce_vector,
    _matrix_rank,
    _matrix_shape,
    _nonpositive_value_metadata,
    _raise_inference_error,
    _require_interval_finite,
    _require_interval_level,
    _require_interval_matches_center_scale,
    _validate_alpha,
)
from .inference_parametric import (
    _assert_estimation_payload_matches_score,
    _base_result,
    _raise_payload_mismatch,
)
from .inference_bootstrap import (
    _bootstrap_suprema_summary,
    _matrix_square_root_psd,
    _validate_uniform_band_bootstrap_summary,
    compute_uniform_band_critical_value,
)


def _max_eq43_constraint_violation(
    sigma_x_hat: np.ndarray,
    cross_row: np.ndarray,
    m_row: np.ndarray,
    *,
    lambda_double_prime: float,
) -> float:
    """Compute the maximum constraint violation for the Eq. (12) projection.

    The Eq. (12) constraint is:
        ||m' E_n(X_i X_i') - E_n(psi_j X_i')||_inf <= lambda''

    Parameters
    ----------
    sigma_x_hat : ndarray of shape (p, p)
        Estimated covariate covariance E_n(X_i X_i').
    cross_row : ndarray of shape (p,)
        Cross-moment E_n(psi_j X_i') for basis function j.
    m_row : ndarray of shape (p,)
        Candidate projection vector for basis function j.
    lambda_double_prime : float
        Relaxation parameter lambda''.

    Returns
    -------
    float
        Maximum violation ``max(0, ||m' Sigma_X - cross||_inf - lambda'')``.
        Zero indicates feasibility.
    """
    residual = m_row @ sigma_x_hat - cross_row
    if residual.size == 0:
        return 0.0
    return max(0.0, float(np.max(np.abs(residual)) - lambda_double_prime))


# Estimated non-zero entries threshold for the block-diagonal LP batch.
# Each Eq. (4.3) row contributes a (2p, 2p) dense block, i.e. 4p^2 nonzeros.
# Above this threshold we fall back to row-by-row solving to avoid excessive
# memory use.
_LP_BATCH_NNZ_THRESHOLD = 10_000_000


def _solve_eq43_batch_row(
    sigma_matrix: np.ndarray,
    cross_matrix: np.ndarray,
    *,
    lambda_double_prime: float,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    """Solve all Eq. (4.3) projection rows as one block-diagonal LP.

    Each row of ``cross_matrix`` corresponds to an independent LP with ``2p``
    variables and ``2p`` constraints.  Because the row blocks are disjoint,
    merging them into a single LP with a block-diagonal constraint matrix is
    mathematically equivalent to solving the rows separately: the objective
    and every constraint are identical, only concatenated.

    Parameters
    ----------
    sigma_matrix : ndarray of shape (p, p)
        Validated covariate covariance matrix.
    cross_matrix : ndarray of shape (L, p)
        Validated cross-moment matrix.
    lambda_double_prime : float
        Validated relaxation parameter.

    Returns
    -------
    m_rows : ndarray of shape (L, p)
        Optimal sparse projection matrix.
    row_metadata : list[dict]
        Per-row solver diagnostics.
    batch_metadata : dict
        Diagnostics for the combined solve, including ``linprog_status``,
        ``linprog_message`` and ``batch_solved``.
    """
    x_dimension = int(sigma_matrix.shape[0])
    basis_dimension = int(cross_matrix.shape[0])
    var_per_row = 2 * x_dimension
    constraint_per_row = 2 * x_dimension

    objective = np.ones(var_per_row * basis_dimension, dtype=float)
    constraint_blocks: list[np.ndarray] = []
    constraint_bounds_list: list[np.ndarray] = []

    for row_index in range(basis_dimension):
        cross_row = cross_matrix[row_index, :]
        block_rows = np.empty((constraint_per_row, var_per_row), dtype=float)
        block_bounds = np.empty(constraint_per_row, dtype=float)
        for i in range(x_dimension):
            sigma_column = sigma_matrix[:, i]
            positive_row = np.concatenate([sigma_column, -sigma_column])
            block_rows[2 * i, :] = positive_row
            block_bounds[2 * i] = float(cross_row[i] + lambda_double_prime)
            block_rows[2 * i + 1, :] = -positive_row
            block_bounds[2 * i + 1] = float(-cross_row[i] + lambda_double_prime)
        constraint_blocks.append(block_rows)
        constraint_bounds_list.append(block_bounds)

    result = linprog(
        objective,
        A_ub=block_diag(constraint_blocks, format="coo"),
        b_ub=np.concatenate(constraint_bounds_list),
        bounds=[(0.0, None)] * (var_per_row * basis_dimension),
        method="highs",
    )

    solver_status = int(result.status)
    solver_message = str(result.message)
    batch_metadata: dict[str, object] = {
        "linprog_status": solver_status,
        "linprog_message": solver_message,
        "batch_solved": result.success,
    }

    if not result.success:
        nan_array = np.full((basis_dimension, x_dimension), np.nan, dtype=float)
        row_metadata = [
            {
                "solver": "linear-programming-l1",
                "linprog_status": solver_status,
                "linprog_message": solver_message,
                "selected_threshold": None,
                "l1_norm": float("nan"),
                "constraint_violation": float("inf"),
                "feasible": False,
                "row_index": row_index,
            }
            for row_index in range(basis_dimension)
        ]
        return nan_array, row_metadata, batch_metadata

    m_rows = np.empty((basis_dimension, x_dimension), dtype=float)
    row_metadata: list[dict[str, object]] = []
    for row_index in range(basis_dimension):
        x_start = row_index * var_per_row
        x_end = x_start + var_per_row
        x_row = result.x[x_start:x_end]
        m_row = np.asarray(
            x_row[:x_dimension] - x_row[x_dimension:],
            dtype=float,
        )
        m_rows[row_index] = m_row
        best_l1 = float(np.sum(np.abs(m_row)))
        best_violation = _max_eq43_constraint_violation(
            sigma_matrix,
            cross_matrix[row_index, :],
            m_row,
            lambda_double_prime=lambda_double_prime,
        )
        row_metadata.append(
            {
                "solver": "linear-programming-l1",
                "linprog_status": solver_status,
                "linprog_message": solver_message,
                "selected_threshold": None,
                "l1_norm": best_l1,
                "constraint_violation": best_violation,
                "feasible": best_violation <= 1e-10,
                "row_index": row_index,
            }
        )

    # Lightweight KKT sanity check: the concatenated L1 objective must equal
    # the sum of per-row L1 norms (no extraction round-off).
    total_l1 = float(np.sum(np.abs(m_rows)))
    if not np.isclose(result.fun, total_l1, rtol=1e-10, atol=1e-10):
        raise RuntimeError(
            "Eq. (4.3) batch LP objective does not match the sum of "
            "per-row L1 norms; extraction may be incorrect"
        )

    return m_rows, row_metadata, batch_metadata


def _solve_eq43_rows_row(
    sigma_matrix: np.ndarray,
    cross_matrix: np.ndarray,
    *,
    lambda_double_prime: float,
    max_threshold_steps: int,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    """Solve Eq. (4.3) projection rows row-by-row (original fallback path)."""
    basis_dimension = int(cross_matrix.shape[0])
    x_dimension = int(sigma_matrix.shape[0])
    m_rows = np.zeros((basis_dimension, x_dimension), dtype=float)
    row_metadata: list[dict[str, object]] = []
    for row_index, cross_row in enumerate(cross_matrix):
        m_row, metadata = _solve_eq43_single_row(
            sigma_matrix,
            np.asarray(cross_row, dtype=float),
            lambda_double_prime=lambda_double_prime,
            max_threshold_steps=max_threshold_steps,
        )
        m_rows[row_index] = m_row
        row_metadata.append({"row_index": row_index, **metadata})
    fallback_metadata: dict[str, object] = {
        "batch_solved": False,
        "linprog_status": None,
        "linprog_message": None,
    }
    return m_rows, row_metadata, fallback_metadata


def _solve_eq43_single_row(
    sigma_x_hat: np.ndarray,
    cross_row: np.ndarray,
    *,
    lambda_double_prime: float,
    max_threshold_steps: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve a single Eq. (12) projection row via linear programming.

    For a single basis function row ``cross_row``, finds the L1-minimal
    projection vector ``m`` satisfying the Dantzig-type constraint:
        ||m' Sigma_X - cross||_inf <= lambda''

    Parameters
    ----------
    sigma_x_hat : ndarray of shape (p, p)
        Estimated covariate covariance.
    cross_row : ndarray of shape (p,)
        Cross-moment vector for the current basis function.
    lambda_double_prime : float
        Relaxation parameter lambda''.
    max_threshold_steps : int
        Reserved for alternative solvers (unused by the LP path).

    Returns
    -------
    m_row : ndarray of shape (p,)
        Optimal sparse projection vector.
    metadata : dict
        Per-row solver diagnostics.
    """
    x_dimension = int(sigma_x_hat.shape[0])
    if x_dimension == 0:
        best_m = np.zeros(0, dtype=float)
        best_l1 = 0.0
        best_violation = 0.0
        solver_status = 0
        solver_message = "empty-x-dimension"
    else:
        objective = np.ones(2 * x_dimension, dtype=float)
        constraint_rows = []
        constraint_bounds = []
        for column_index in range(x_dimension):
            sigma_column = sigma_x_hat[:, column_index]
            positive_row = np.concatenate([sigma_column, -sigma_column])
            constraint_rows.append(positive_row)
            constraint_bounds.append(
                float(cross_row[column_index] + lambda_double_prime)
            )
            constraint_rows.append(-positive_row)
            constraint_bounds.append(
                float(-cross_row[column_index] + lambda_double_prime)
            )

        result = linprog(
            objective,
            A_ub=np.asarray(constraint_rows, dtype=float),
            b_ub=np.asarray(constraint_bounds, dtype=float),
            bounds=[(0.0, None)] * (2 * x_dimension),
            method="highs",
        )
        solver_status = int(result.status)
        solver_message = str(result.message)
        if result.success:
            best_m = np.asarray(
                result.x[:x_dimension] - result.x[x_dimension:],
                dtype=float,
            )
            best_l1 = float(result.fun)
            best_violation = _max_eq43_constraint_violation(
                sigma_x_hat,
                cross_row,
                best_m,
                lambda_double_prime=lambda_double_prime,
            )
        else:
            best_m = np.full(x_dimension, np.nan, dtype=float)
            best_l1 = float("nan")
            best_violation = float("inf")

    row_metadata = {
        "solver": "linear-programming-l1",
        "linprog_status": solver_status,
        "linprog_message": solver_message,
        "selected_threshold": None,
        "l1_norm": best_l1,
        "constraint_violation": best_violation,
        "feasible": best_violation <= 1e-10,
    }
    return np.asarray(best_m, dtype=float), row_metadata


def solve_eq43_projection_matrix(
    *,
    sigma_x_hat: np.ndarray,
    cross_moment_hat: np.ndarray,
    lambda_double_prime: float,
    max_threshold_steps: int = 25,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve the Eq. (4.3) sparse projection matrix optimization.

    For each row of cross_moment_hat, finds a projection vector m
    that minimizes ||m||_1 subject to the constraint
    ||m' Sigma_X - cross_j||_inf <= lambda''.

    The projection matrix M is used to orthogonalize the sieve basis
    against high-dimensional covariates for nonparametric inference.

    Parameters
    ----------
    sigma_x_hat : ndarray of shape (p, p)
        Estimated covariance of the covariates X'X / n.
        Must be symmetric and positive semidefinite.
    cross_moment_hat : ndarray of shape (L, p)
        Cross-moment matrix psi'X / n, where psi is the sieve basis
        (L = basis dimension, p = covariate dimension).
    lambda_double_prime : float
        Relaxation parameter controlling the constraint tolerance.
    max_threshold_steps : int, default 25
        Maximum threshold search steps (metadata for alternative
        solvers).

    Returns
    -------
    m_hat : ndarray of shape (L, p)
        Optimal sparse projection matrix, one row per basis function.
    metadata : dict
        Solver diagnostics including feasibility status, constraint
        violations, and per-row metadata.

    Raises
    ------
    ValueError
        If sigma_x_hat is not square, not symmetric, or not positive
        semidefinite; if cross_moment_hat has incompatible dimensions.
    SparseDirectionInfeasibleError
        If the linear program fails to find a feasible solution for
        any basis row.

    Notes
    -----
    Implements Eq. (4.3) from Ning, Peng, and Tao (2020).
    The optimization is solved row-by-row via linear programming
    (HiGHS). The resulting M orthogonalizes the basis against X:
    psi_tilde = psi - M X, enabling valid nonparametric inference
    in the presence of high-dimensional confounders.

    Reference: Ning, Peng, and Tao (2020), arXiv preprint arXiv:2009.03151.
    """
    sigma_matrix = _coerce_matrix("sigma_x_hat", sigma_x_hat)
    if sigma_matrix.shape[0] != sigma_matrix.shape[1]:
        raise ValueError("sigma_x_hat must be square")
    if not np.allclose(
        sigma_matrix,
        sigma_matrix.T,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("sigma_x_hat must be symmetric")
    if _minimum_symmetric_eigenvalue(sigma_matrix) < -1e-10:
        raise ValueError("sigma_x_hat must be positive semidefinite")
    cross_matrix = _coerce_matrix("cross_moment_hat", cross_moment_hat)
    if cross_matrix.shape[1] != sigma_matrix.shape[0]:
        raise ValueError("cross_moment_hat must align with sigma_x_hat")
    if cross_matrix.shape[0] == 0:
        raise ValueError(
            "cross_moment_hat must contain at least one Eq. (4.3) basis target"
        )
    lambda_value = _coerce_nonnegative_finite(
        "lambda_double_prime",
        lambda_double_prime,
    )
    max_threshold_steps_value = _coerce_positive_integer(
        "max_threshold_steps",
        max_threshold_steps,
    )

    basis_dimension = int(cross_matrix.shape[0])
    x_dimension = int(sigma_matrix.shape[0])
    # Each row adds a (2p, 2p) dense block => 4 * L * p^2 nonzeros.
    estimated_batch_nnz = 4 * basis_dimension * x_dimension * x_dimension

    m_rows = np.zeros_like(cross_matrix, dtype=float)
    row_metadata: list[dict[str, object]] = []
    use_batch = (
        estimated_batch_nnz <= _LP_BATCH_NNZ_THRESHOLD
        and basis_dimension > 0
        and x_dimension > 0
    )
    batch_failed = False

    if use_batch:
        try:
            m_rows, batch_row_metadata, _ = _solve_eq43_batch_row(
                sigma_matrix,
                cross_matrix,
                lambda_double_prime=lambda_value,
            )
            if all(bool(m["feasible"]) for m in batch_row_metadata):
                row_metadata = batch_row_metadata
            else:
                batch_failed = True
        except Exception:
            batch_failed = True

    if batch_failed or not row_metadata:
        m_rows, row_metadata, _ = _solve_eq43_rows_row(
            sigma_matrix,
            cross_matrix,
            lambda_double_prime=lambda_value,
            max_threshold_steps=max_threshold_steps_value,
        )

    for row_index, metadata in enumerate(row_metadata):
        if not bool(metadata["feasible"]):
            _raise_inference_error(
                SparseDirectionInfeasibleError,
                "Eq. (4.3) projection solver failed to satisfy the constraint",
                failure_kind="eq43-infeasible",
                target_kind="eq43-projection",
                row_index=row_index,
                x_dimension=int(sigma_matrix.shape[0]),
                basis_dimension_full=int(cross_matrix.shape[0]),
                constraint_violation=float(metadata["constraint_violation"]),
                lambda_double_prime=lambda_value,
                linprog_status=int(metadata["linprog_status"]),
                linprog_message=str(metadata["linprog_message"]),
            )

    constraint_violation_max = (
        max(float(metadata["constraint_violation"]) for metadata in row_metadata)
        if row_metadata
        else 0.0
    )
    metadata = {
        "solver": "eq43-linear-programming-l1",
        "lambda_double_prime": lambda_value,
        "max_threshold_steps": max_threshold_steps_value,
        "basis_dimension_full": int(cross_matrix.shape[0]),
        "x_dimension": int(sigma_matrix.shape[0]),
        "constraint_violation_max": constraint_violation_max,
        "feasible": constraint_violation_max <= 1e-10,
        "row_metadata": row_metadata,
    }
    return m_rows, metadata


def _minimum_symmetric_eigenvalue(matrix: np.ndarray) -> float:
    """Return the minimum eigenvalue of the symmetric part of ``matrix``.

    Parameters
    ----------
    matrix : ndarray of shape (p, p)
        Input matrix (will be symmetrized internally).

    Returns
    -------
    float
        Minimum eigenvalue of ``(matrix + matrix') / 2``.
        Returns ``inf`` for empty matrices.
    """
    symmetric = 0.5 * (
        np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T
    )
    if symmetric.size == 0:
        return float("inf")
    return float(np.min(np.linalg.eigvalsh(symmetric)))


def _symmetric_eigenvalue_diagnostic(
    matrix: np.ndarray,
    *,
    prefix: str,
) -> dict[str, object]:
    """Return eigenvalue diagnostics for a symmetrized matrix.

    Parameters
    ----------
    matrix : ndarray of shape (p, p)
        Matrix to analyze.
    prefix : str
        Prefix for output dictionary keys.

    Returns
    -------
    dict
        Contains ``{prefix}_min_eigenvalue``, ``{prefix}_max_eigenvalue``,
        ``{prefix}_negative_eigenvalue_count``, and
        ``{prefix}_negative_eigenvalue_mass``.
    """
    symmetric = 0.5 * (
        np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T
    )
    if symmetric.size == 0:
        return {
            f"{prefix}_min_eigenvalue": float("inf"),
            f"{prefix}_max_eigenvalue": float("inf"),
            f"{prefix}_negative_eigenvalue_count": 0,
            f"{prefix}_negative_eigenvalue_mass": 0.0,
        }
    eigenvalues = np.linalg.eigvalsh(symmetric)
    negative_eigenvalues = eigenvalues[eigenvalues < -_PSD_EIGENVALUE_TOL]
    return {
        f"{prefix}_min_eigenvalue": float(np.min(eigenvalues)),
        f"{prefix}_max_eigenvalue": float(np.max(eigenvalues)),
        f"{prefix}_negative_eigenvalue_count": int(negative_eigenvalues.size),
        f"{prefix}_negative_eigenvalue_mass": max(
            0.0,
            float(-np.sum(negative_eigenvalues)),
        ),
    }


def _solve_sigma_f_left(sigma_f_hat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``sigma_f_hat @ x = rhs`` via checked linear solve.

    Computes ``x = Sigma_f^{-1} rhs``, used in the nonparametric
    debiasing step ``bar_gamma = gamma - Sigma_f^{-1} score_moment``.

    Parameters
    ----------
    sigma_f_hat : ndarray of shape (L, L)
        Covariance of the orthogonalized basis (must be full rank).
    rhs : ndarray of shape (L,) or (L, k)
        Right-hand side vector or matrix.

    Returns
    -------
    ndarray
        Solution ``x`` of ``sigma_f_hat @ x = rhs``.
    """
    return _checked_solve(
        np.asarray(sigma_f_hat, dtype=float),
        np.asarray(rhs, dtype=float),
    )


def _sandwich_solve_sigma_f(
    sigma_f_hat: np.ndarray,
    omega_f_hat: np.ndarray,
) -> np.ndarray:
    """Compute the sandwich matrix V_f = Sigma_f^{-1} Omega_f Sigma_f^{-1}.

    This is the asymptotic variance matrix for the nonparametric sieve
    coefficients in Theorem 3, defined as:
        V_f = Sigma_f^{-1} Omega_f Sigma_f^{-1}

    where ``Sigma_f = E[(psi - M*X) psi']`` is the covariance of the
    orthogonalized basis and ``Omega_f = E[sigma_i^2 psi psi'] -
    M E[sigma_i^2 X X'] M'`` is the long-run variance.

    Parameters
    ----------
    sigma_f_hat : ndarray of shape (L, L)
        Estimated basis covariance (must be full rank).
    omega_f_hat : ndarray of shape (L, L)
        Estimated long-run variance (must be PSD).

    Returns
    -------
    ndarray of shape (L, L)
        Sandwich matrix ``Sigma_f^{-1} Omega_f Sigma_f^{-1}``.

    Notes
    -----
    References: Theorem 3 and surrounding definitions in
    Ning, Peng & Tao (2020), arXiv preprint arXiv:2009.03151.
    """
    left_solved = _solve_sigma_f_left(sigma_f_hat, omega_f_hat)
    return _checked_solve(
        np.asarray(sigma_f_hat, dtype=float),
        left_solved.T,
    ).T


def _paper_difference_omega_f_hat(
    *,
    basis_valid_full: np.ndarray,
    x_valid: np.ndarray,
    residual_valid: np.ndarray,
    m_hat: np.ndarray,
    n_valid: int,
) -> np.ndarray:
    """Compute Omega_f using the paper-difference formula.

    Implements the Omega_f estimator from Section 4 of the paper:

        Omega_f = E_n[sigma_i^2 psi psi'] - M_hat E_n[sigma_i^2 X X'] M_hat'

    where ``sigma_i`` is the residual and ``psi`` is the sieve basis.
    This is the primary (non-fallback) strategy for computing the
    nonparametric long-run variance.

    Parameters
    ----------
    basis_valid_full : ndarray of shape (n, L)
        Sieve basis evaluated at valid observations.
    x_valid : ndarray of shape (n, p)
        Covariates at valid observations.
    residual_valid : ndarray of shape (n,)
        Second-stage residuals epsilon_i.
    m_hat : ndarray of shape (L, p)
        Sparse projection matrix from Eq. (12).
    n_valid : int
        Number of valid observations.

    Returns
    -------
    ndarray of shape (L, L)
        Estimated Omega_f matrix.

    Notes
    -----
    References: Eq. (12) and Theorem 3 in Ning, Peng & Tao (2020),
    arXiv preprint arXiv:2009.03151.
    """
    weighted_basis = (
        np.asarray(basis_valid_full, dtype=float)
        * np.asarray(
            residual_valid,
            dtype=float,
        )[:, None]
    )
    weighted_x = (
        np.asarray(x_valid, dtype=float)
        * np.asarray(
            residual_valid,
            dtype=float,
        )[:, None]
    )
    return (
        weighted_basis.T @ weighted_basis / n_valid
        - m_hat @ (weighted_x.T @ weighted_x / n_valid) @ m_hat.T
    )


def _orthogonal_score_omega_f_hat(
    *,
    orthogonal_basis_valid: np.ndarray,
    residual_valid: np.ndarray,
    n_valid: int,
) -> np.ndarray:
    """Compute Omega_f using the orthogonalized-score formula.

    Computes ``E_n[sigma_i^2 psi_tilde psi_tilde']`` where
    ``psi_tilde = psi - M*X`` is the orthogonalized basis.

    Parameters
    ----------
    orthogonal_basis_valid : ndarray of shape (n, L)
        Orthogonalized basis ``psi - M*X``.
    residual_valid : ndarray of shape (n,)
        Second-stage residuals.
    n_valid : int
        Number of valid observations.

    Returns
    -------
    ndarray of shape (L, L)
        Orthogonalized-score Omega_f estimate.
    """
    weighted_orthogonal_basis = (
        np.asarray(
            orthogonal_basis_valid,
            dtype=float,
        )
        * np.asarray(residual_valid, dtype=float)[:, None]
    )
    return weighted_orthogonal_basis.T @ weighted_orthogonal_basis / n_valid


def _max_abs_or_zero(values: np.ndarray) -> float:
    """Return ``max(|values|)`` or 0.0 for empty arrays."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def _omega_f_paper_difference_component_metadata(
    *,
    basis_valid_full: np.ndarray,
    x_valid: np.ndarray,
    residual_valid: np.ndarray,
    m_hat: np.ndarray,
    sigma_x_hat: np.ndarray,
    cross_moment_hat: np.ndarray,
    omega_f_hat: np.ndarray,
    orthogonal_basis_valid: np.ndarray,
    n_valid: int,
) -> dict[str, object]:
    weighted_basis = (
        np.asarray(basis_valid_full, dtype=float)
        * np.asarray(residual_valid, dtype=float)[:, None]
    )
    weighted_x = (
        np.asarray(x_valid, dtype=float)
        * np.asarray(residual_valid, dtype=float)[:, None]
    )
    weighted_basis_covariance = weighted_basis.T @ weighted_basis / n_valid
    weighted_x_covariance = weighted_x.T @ weighted_x / n_valid
    weighted_x_projection_covariance = (
        m_hat @ weighted_x_covariance @ np.asarray(m_hat, dtype=float).T
    )
    m_matrix = np.asarray(m_hat, dtype=float)
    sigma_x_matrix = np.asarray(sigma_x_hat, dtype=float)
    cross_matrix = np.asarray(cross_moment_hat, dtype=float)
    residual_weighted_cross_moment = weighted_basis.T @ weighted_x / n_valid
    unweighted_cross_gap = cross_matrix - m_matrix @ sigma_x_matrix
    residual_weighted_cross_gap = (
        residual_weighted_cross_moment - m_matrix @ weighted_x_covariance
    )
    residual_weighted_cross_correction = (
        residual_weighted_cross_gap @ m_matrix.T
        + m_matrix @ residual_weighted_cross_gap.T
    )
    orthogonal_score_omega_f = _orthogonal_score_omega_f_hat(
        orthogonal_basis_valid=orthogonal_basis_valid,
        residual_valid=residual_valid,
        n_valid=n_valid,
    )
    paper_minus_orthogonal_score = np.asarray(omega_f_hat, dtype=float) - (
        orthogonal_score_omega_f
    )
    return {
        **_symmetric_eigenvalue_diagnostic(
            omega_f_hat,
            prefix="omega_f_paper_difference",
        ),
        **_symmetric_eigenvalue_diagnostic(
            orthogonal_score_omega_f,
            prefix="omega_f_orthogonal_score",
        ),
        **_symmetric_eigenvalue_diagnostic(
            residual_weighted_cross_correction,
            prefix="omega_f_residual_weighted_cross_correction",
        ),
        "omega_f_basis_component_min_eigenvalue": _minimum_symmetric_eigenvalue(
            weighted_basis_covariance
        ),
        "omega_f_x_component_min_eigenvalue": _minimum_symmetric_eigenvalue(
            weighted_x_projection_covariance
        ),
        "omega_f_basis_component_trace": float(np.trace(weighted_basis_covariance)),
        "omega_f_x_component_trace": float(
            np.trace(weighted_x_projection_covariance)
        ),
        "omega_f_paper_difference_trace": float(np.trace(omega_f_hat)),
        "omega_f_orthogonal_score_trace": float(np.trace(orthogonal_score_omega_f)),
        "omega_f_paper_difference_vs_orthogonal_score_max_abs": _max_abs_or_zero(
            paper_minus_orthogonal_score
        ),
        "omega_f_residual_weighted_cross_correction_trace": float(
            np.trace(residual_weighted_cross_correction)
        ),
        "omega_f_residual_weighted_cross_correction_max_abs": _max_abs_or_zero(
            residual_weighted_cross_correction
        ),
        "omega_f_paper_vs_orthogonal_reconstruction_max_abs": _max_abs_or_zero(
            paper_minus_orthogonal_score - residual_weighted_cross_correction
        ),
        "omega_f_unweighted_cross_identity_max_abs": _max_abs_or_zero(
            unweighted_cross_gap
        ),
        "omega_f_residual_weighted_cross_identity_max_abs": _max_abs_or_zero(
            residual_weighted_cross_gap
        ),
    }


def _require_omega_f_strategy_metadata(metadata: dict[str, object]) -> None:
    strategy = metadata.get("omega_f_strategy")
    selected_strategy = metadata.get("omega_f_selected_strategy")
    primary_strategy = metadata.get("omega_f_primary_strategy")
    primary_psd = metadata.get("omega_f_primary_psd")
    fallback_attempted = metadata.get("omega_f_fallback_attempted")
    fallback_used = metadata.get("omega_f_fallback_used")

    if strategy != "paper-difference":
        raise ValueError("omega_f_strategy metadata must equal paper-difference")
    if selected_strategy != strategy:
        raise ValueError("omega_f_selected_strategy metadata must match omega_f_strategy")
    if primary_strategy != "paper-difference":
        raise ValueError("omega_f_primary_strategy metadata must equal paper-difference")
    if not isinstance(primary_psd, bool):
        raise ValueError("omega_f_primary_psd metadata must be boolean")
    if not isinstance(fallback_attempted, bool):
        raise ValueError("omega_f_fallback_attempted metadata must be boolean")
    if not isinstance(fallback_used, bool):
        raise ValueError("omega_f_fallback_used metadata must be boolean")

    if strategy == "paper-difference":
        if fallback_attempted or fallback_used:
            raise ValueError("paper-difference omega_f metadata cannot mark fallback usage")
        if primary_psd is not True:
            raise ValueError("paper-difference omega_f metadata requires primary PSD evidence")
        if metadata.get("omega_f_fallback_min_eigenvalue") is not None:
            raise ValueError(
                "paper-difference omega_f metadata cannot carry fallback eigenvalue evidence"
            )


def _require_omega_f_selected_eigenvalue_metadata(
    metadata: dict[str, object],
    *,
    selected_min_eigenvalue: float,
) -> None:
    metadata_selected_min_eigenvalue = metadata.get("omega_f_selected_min_eigenvalue")
    if metadata_selected_min_eigenvalue is None:
        raise ValueError("omega_f_selected_min_eigenvalue metadata must be provided")
    if not np.isclose(
        float(metadata_selected_min_eigenvalue),
        float(selected_min_eigenvalue),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(
            "omega_f_selected_min_eigenvalue metadata must match omega_f_hat"
        )

    primary_min_eigenvalue = metadata.get("omega_f_primary_min_eigenvalue")
    primary_psd = metadata.get("omega_f_primary_psd")
    if primary_min_eigenvalue is None:
        raise ValueError("omega_f_primary_min_eigenvalue metadata must be provided")
    primary_min_eigenvalue_value = float(primary_min_eigenvalue)
    if not np.isfinite(primary_min_eigenvalue_value):
        raise ValueError("omega_f_primary_min_eigenvalue metadata must be finite")
    if bool(primary_psd) != (primary_min_eigenvalue_value >= -1e-10):
        raise ValueError(
            "omega_f_primary_psd metadata must match primary eigenvalue evidence"
        )


def _sigma_z_squared(
    *,
    evaluation_basis: np.ndarray,
    v_f_hat: np.ndarray,
) -> np.ndarray:
    """Compute pointwise variance sigma_z^2 at the evaluation grid.

    Implements the pointwise variance from Theorem 3:
        sigma_z^2(z) = psi(z)' V_f psi(z)

    where ``V_f = Sigma_f^{-1} Omega_f Sigma_f^{-1}`` is the sandwich
    matrix and ``psi(z)`` is the sieve basis evaluated at grid point z.

    Parameters
    ----------
    evaluation_basis : ndarray of shape (G, L)
        Sieve basis evaluated at the z0 grid points.
    v_f_hat : ndarray of shape (L, L)
        Sandwich matrix V_f = Sigma_f^{-1} Omega_f Sigma_f^{-1}.

    Returns
    -------
    ndarray of shape (G,)
        Pointwise variance ``psi(z)' V_f psi(z)`` at each grid point.

    Notes
    -----
    References: Theorem 3 in Ning, Peng & Tao (2020), arXiv preprint arXiv:2009.03151.
    The asymptotic result is
    ``sqrt(n) sigma_z^{-1/2} (f_bar(z) - f_0(z)) ->_d N(0, 1)``.
    """
    return np.einsum(
        "ij,jk,ik->i",
        np.asarray(evaluation_basis, dtype=float),
        np.asarray(v_f_hat, dtype=float),
        np.asarray(evaluation_basis, dtype=float),
    )


@dataclass(slots=True)
class NonparametricInferencePayload:
    """Inference output for the nonparametric function f(z), Eq. (4.3).

    Contains the debiased sieve estimates bar_gamma, pointwise confidence
    intervals, and uniform confidence bands for f evaluated at the z0
    grid, using the Eq. (4.3) sparse projection M to orthogonalize the
    basis against high-dimensional covariates.

    Attributes
    ----------
    m_hat : ndarray of float, shape (L, p)
        Sparse projection matrix from Eq. (4.3) optimization.
    sigma_x_hat : ndarray of float, shape (p, p)
        Estimated covariate covariance X'X / n.
    cross_moment_hat : ndarray of float, shape (L, p)
        Cross-moment psi'X / n between basis and covariates.
    orthogonal_basis_valid : ndarray of float, shape (n_valid, L)
        Orthogonalized basis psi_tilde = psi - M X on valid obs.
    sigma_f_hat : ndarray of float, shape (L, L)
        Covariance of the orthogonalized basis.
    omega_f_hat : ndarray of float, shape (L, L)
        Long-run variance for the nonparametric score.
    v_f_hat : ndarray of float, shape (L, L)
        Sandwich matrix Sigma_f^{-1} Omega_f Sigma_f^{-1}.
    score_moment : ndarray of float, shape (L,)
        Sample mean of the orthogonalized basis score.
    bar_gamma_hat : ndarray of float, shape (L,)
        Debiased sieve coefficients gamma - Sigma_f^{-1} score.
    evaluation_basis : ndarray of float, shape (G, L)
        Basis evaluated at the z0 grid.
    bar_f_at_z0 : ndarray of float, shape (G,)
        Debiased f estimates at the z0 grid.
    sigma_z_hat : ndarray of float, shape (G,)
        Pointwise standard errors at the z0 grid.
    covariance_at_grid : ndarray of float, shape (G, G)
        Joint covariance of f estimates at the z0 grid.
    uniform_standardization : ndarray of float, shape (G,)
        Standardization factors sqrt(diag(covariance_at_grid)).
    pointwise_confidence_interval : ConfidenceInterval
        Pointwise confidence intervals for f(z0).
    uniform_band : UniformBand
        Simultaneous confidence band over the z0 grid.
    alpha : float
        Significance level.
    basis_family : str
        Sieve basis family.
    basis_degree : int
        Sieve truncation parameter.
    oracle_lane : str
        Computational lane identifier.
    optimization_metadata : dict
        Solver diagnostics for the Eq. (4.3) projection problem.
    """

    m_hat: np.ndarray
    sigma_x_hat: np.ndarray
    cross_moment_hat: np.ndarray
    orthogonal_basis_valid: np.ndarray
    sigma_f_hat: np.ndarray
    omega_f_hat: np.ndarray
    v_f_hat: np.ndarray
    score_moment: np.ndarray
    bar_gamma_hat: np.ndarray
    evaluation_basis: np.ndarray
    bar_f_at_z0: np.ndarray
    sigma_z_hat: np.ndarray
    covariance_at_grid: np.ndarray
    uniform_standardization: np.ndarray
    pointwise_confidence_interval: ConfidenceInterval
    uniform_band: UniformBand
    alpha: float
    basis_family: str
    basis_degree: int
    oracle_lane: str
    optimization_metadata: dict[str, object]

    _validate_optimality: ClassVar[bool] = False

    def __post_init__(self) -> None:
        self.m_hat = _coerce_matrix("m_hat", self.m_hat)
        self.sigma_x_hat = _coerce_matrix("sigma_x_hat", self.sigma_x_hat)
        self.cross_moment_hat = _coerce_matrix(
            "cross_moment_hat", self.cross_moment_hat
        )
        self.orthogonal_basis_valid = _coerce_matrix(
            "orthogonal_basis_valid",
            self.orthogonal_basis_valid,
        )
        self.sigma_f_hat = _coerce_matrix("sigma_f_hat", self.sigma_f_hat)
        self.omega_f_hat = _coerce_matrix("omega_f_hat", self.omega_f_hat)
        self.v_f_hat = _coerce_matrix("v_f_hat", self.v_f_hat)
        self.score_moment = _coerce_vector("score_moment", self.score_moment)
        self.bar_gamma_hat = _coerce_vector("bar_gamma_hat", self.bar_gamma_hat)
        self.evaluation_basis = _coerce_matrix(
            "evaluation_basis", self.evaluation_basis
        )
        self.bar_f_at_z0 = _coerce_vector("bar_f_at_z0", self.bar_f_at_z0)
        self.sigma_z_hat = _coerce_vector("sigma_z_hat", self.sigma_z_hat)
        self.covariance_at_grid = _coerce_matrix(
            "covariance_at_grid", self.covariance_at_grid
        )
        self.uniform_standardization = _coerce_vector(
            "uniform_standardization", self.uniform_standardization
        )
        self.alpha = _validate_alpha(self.alpha)
        self.basis_family = str(self.basis_family).strip().lower()
        self.basis_degree = _coerce_basis_degree(
            self.basis_family,
            self.basis_degree,
        )
        self.oracle_lane = str(self.oracle_lane)
        self.optimization_metadata = dict(self.optimization_metadata)

        _require_interval_finite(
            "pointwise_confidence_interval",
            self.pointwise_confidence_interval,
        )
        _require_interval_finite("uniform_band", self.uniform_band)
        _require_interval_level(
            "pointwise_confidence_interval",
            self.pointwise_confidence_interval,
            self.alpha,
        )
        _require_interval_level("uniform_band", self.uniform_band, self.alpha)

        basis_dimension = self.m_hat.shape[0]
        x_dimension = self.m_hat.shape[1]
        if self.sigma_x_hat.shape != (x_dimension, x_dimension):
            raise ValueError("sigma_x_hat must align with m_hat")
        if not np.allclose(
            self.sigma_x_hat,
            self.sigma_x_hat.T,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("sigma_x_hat must be symmetric")
        if _minimum_symmetric_eigenvalue(self.sigma_x_hat) < -1e-10:
            raise ValueError("sigma_x_hat must be positive semidefinite")
        if self.cross_moment_hat.shape != self.m_hat.shape:
            raise ValueError("cross_moment_hat must align with m_hat")
        if self.orthogonal_basis_valid.shape[1] != basis_dimension:
            raise ValueError("orthogonal_basis_valid must align with basis dimension")
        n_valid_obs_for_shape = self.optimization_metadata.get("n_valid_obs")
        if n_valid_obs_for_shape is not None:
            n_valid_shape_value = _coerce_positive_integer(
                "n_valid_obs",
                n_valid_obs_for_shape,
            )
            if self.orthogonal_basis_valid.shape[0] != n_valid_shape_value:
                raise ValueError("orthogonal_basis_valid must align with n_valid_obs")
        lambda_double_prime = self.optimization_metadata.get("lambda_double_prime")
        if lambda_double_prime is not None:
            lambda_double_prime_value = _coerce_nonnegative_finite(
                "lambda_double_prime",
                lambda_double_prime,
            )
            eq43_residual = self.m_hat @ self.sigma_x_hat - self.cross_moment_hat
            eq43_violation = (
                0.0
                if eq43_residual.size == 0
                else max(
                    0.0,
                    float(np.max(np.abs(eq43_residual)) - lambda_double_prime_value),
                )
            )
            if eq43_violation > 1e-10:
                raise ValueError(
                    "m_hat must satisfy the Eq. (4.3) projection constraint"
                )
            max_threshold_steps = self.optimization_metadata.get(
                "max_threshold_steps",
                25,
            )
            max_threshold_steps_value = _coerce_positive_integer(
                "max_threshold_steps",
                max_threshold_steps,
            )
            if self._validate_optimality:
                expected_m_hat, _ = solve_eq43_projection_matrix(
                    sigma_x_hat=self.sigma_x_hat,
                    cross_moment_hat=self.cross_moment_hat,
                    lambda_double_prime=lambda_double_prime_value,
                    max_threshold_steps=max_threshold_steps_value,
                )
                expected_l1 = np.sum(np.abs(expected_m_hat), axis=1)
                actual_l1 = np.sum(np.abs(self.m_hat), axis=1)
                if not np.allclose(
                    actual_l1,
                    expected_l1,
                    atol=1e-12,
                    rtol=1e-10,
                ):
                    raise ValueError(
                        "m_hat must attain the Eq. (4.3) projection L1 optimum"
                    )
        if self.sigma_f_hat.shape != (basis_dimension, basis_dimension):
            raise ValueError("sigma_f_hat must align with basis dimension")
        if n_valid_obs_for_shape is None:
            raise ValueError("n_valid_obs must be provided to validate sigma_f_hat")
        n_valid_shape_value = _coerce_positive_integer(
            "n_valid_obs",
            n_valid_obs_for_shape,
        )
        implied_orthogonal_cross_moment = (
            self.cross_moment_hat - self.m_hat @ self.sigma_x_hat
        )
        expected_sigma_f_hat = (
            self.orthogonal_basis_valid.T @ self.orthogonal_basis_valid
        ) / n_valid_shape_value + implied_orthogonal_cross_moment @ self.m_hat.T
        if not np.allclose(
            self.sigma_f_hat,
            expected_sigma_f_hat,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "sigma_f_hat must equal the covariance implied by "
                "orthogonal_basis_valid"
            )
        if self.omega_f_hat.shape != (basis_dimension, basis_dimension):
            raise ValueError("omega_f_hat must align with basis dimension")
        _require_omega_f_strategy_metadata(self.optimization_metadata)
        if not np.allclose(
            self.omega_f_hat,
            self.omega_f_hat.T,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("omega_f_hat must be symmetric")
        if _minimum_symmetric_eigenvalue(self.omega_f_hat) < -1e-10:
            raise ValueError("omega_f_hat must be positive semidefinite")
        _require_omega_f_selected_eigenvalue_metadata(
            self.optimization_metadata,
            selected_min_eigenvalue=_minimum_symmetric_eigenvalue(self.omega_f_hat),
        )
        if self.v_f_hat.shape != (basis_dimension, basis_dimension):
            raise ValueError("v_f_hat must align with basis dimension")
        if not np.allclose(self.v_f_hat, self.v_f_hat.T, atol=1e-10, rtol=1e-10):
            raise ValueError("v_f_hat must be symmetric")
        if _minimum_symmetric_eigenvalue(self.v_f_hat) < -1e-10:
            raise ValueError("v_f_hat must be positive semidefinite")
        sigma_f_singular_values = np.linalg.svd(self.sigma_f_hat, compute_uv=False)
        if (
            sigma_f_singular_values.size == 0
            or float(np.min(sigma_f_singular_values)) <= _INFERENCE_EPS
        ):
            raise ValueError("sigma_f_hat must be full rank")
        expected_v_f_hat = _sandwich_solve_sigma_f(
            self.sigma_f_hat,
            self.omega_f_hat,
        )
        if not np.allclose(self.v_f_hat, expected_v_f_hat, atol=1e-12, rtol=1e-10):
            raise ValueError(
                "v_f_hat must equal the sigma_f_hat linear-solve sandwich"
            )
        if self.score_moment.shape[0] != basis_dimension:
            raise ValueError("score_moment must align with basis dimension")
        if self.bar_gamma_hat.shape[0] != basis_dimension:
            raise ValueError("bar_gamma_hat must align with basis dimension")
        gamma_hat_metadata = self.optimization_metadata.get("gamma_hat")
        if gamma_hat_metadata is None:
            raise ValueError("gamma_hat metadata must be provided to validate bar_gamma_hat")
        gamma_hat = _coerce_vector("gamma_hat", gamma_hat_metadata)
        if gamma_hat.shape[0] != basis_dimension:
            raise ValueError("gamma_hat metadata must align with basis dimension")
        expected_bar_gamma_hat = gamma_hat - _solve_sigma_f_left(
            self.sigma_f_hat,
            self.score_moment,
        )
        if not np.allclose(
            self.bar_gamma_hat,
            expected_bar_gamma_hat,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "bar_gamma_hat must equal gamma_hat minus the sigma_f_hat linear solve of score_moment"
            )
        evaluation_grid_size = self.evaluation_basis.shape[0]
        if self.evaluation_basis.shape[1] != basis_dimension:
            raise ValueError("evaluation_basis must align with basis dimension")
        if self.bar_f_at_z0.shape[0] != evaluation_grid_size:
            raise ValueError("bar_f_at_z0 must align with evaluation_basis")
        if evaluation_grid_size <= 0:
            raise ValueError("evaluation_basis must contain at least one grid point")
        expected_bar_f_at_z0 = self.evaluation_basis @ self.bar_gamma_hat
        if not np.allclose(
            self.bar_f_at_z0,
            expected_bar_f_at_z0,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError("bar_f_at_z0 must equal evaluation_basis @ bar_gamma_hat")
        if self.sigma_z_hat.shape[0] != evaluation_grid_size:
            raise ValueError("sigma_z_hat must align with evaluation_basis")
        if self.covariance_at_grid.shape != (
            evaluation_grid_size,
            evaluation_grid_size,
        ):
            raise ValueError("covariance_at_grid must align with evaluation_basis")
        if not np.allclose(
            self.covariance_at_grid,
            self.covariance_at_grid.T,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("covariance_at_grid must be symmetric")
        if self.uniform_standardization.shape[0] != evaluation_grid_size:
            raise ValueError("uniform_standardization must align with evaluation_basis")
        if np.any(self.uniform_standardization <= np.finfo(float).eps):
            raise ValueError("uniform_standardization must be strictly positive")
        if np.any(self.sigma_z_hat <= np.finfo(float).eps):
            raise ValueError("sigma_z_hat must be strictly positive")
        covariance_grid_diagonal = np.diag(self.covariance_at_grid)
        if np.any(covariance_grid_diagonal <= np.finfo(float).eps):
            raise ValueError("covariance_at_grid diagonal must be strictly positive")
        if _minimum_symmetric_eigenvalue(self.covariance_at_grid) < -1e-10:
            raise ValueError("covariance_at_grid must be positive semidefinite")
        n_valid_obs = self.optimization_metadata.get("n_valid_obs")
        if n_valid_obs is None:
            raise ValueError("n_valid_obs must be provided to validate covariance_at_grid")
        n_valid_obs_value = _coerce_positive_integer("n_valid_obs", n_valid_obs)
        expected_covariance_at_grid = (
            self.evaluation_basis @ self.v_f_hat @ self.evaluation_basis.T
        ) / n_valid_obs_value
        if not np.allclose(
            self.covariance_at_grid,
            expected_covariance_at_grid,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "covariance_at_grid must equal "
                "evaluation_basis @ v_f_hat @ evaluation_basis.T / n_valid"
            )
        expected_standardization = np.sqrt(covariance_grid_diagonal)
        if not np.allclose(
            self.uniform_standardization,
            expected_standardization,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError(
                "uniform_standardization must equal sqrt(diag(covariance_at_grid))"
            )
        if not np.allclose(
            self.sigma_z_hat,
            expected_standardization,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("sigma_z_hat must equal sqrt(diag(covariance_at_grid))")
        z_critical = NormalDist().inv_cdf(1.0 - self.alpha / 2.0)
        _require_interval_matches_center_scale(
            "pointwise_confidence_interval",
            self.pointwise_confidence_interval,
            center=self.bar_f_at_z0,
            scale=self.sigma_z_hat,
            multiplier=z_critical,
        )
        if self.uniform_band.critical_value is None:
            raise ValueError("uniform_band.critical_value must be provided")
        if self.uniform_band.n_boot is None:
            raise ValueError("uniform_band.n_boot must be provided")
        if self.uniform_band.random_state is None:
            raise ValueError("uniform_band.random_state must be provided")
        if self._validate_optimality:
            covariance_sqrt = _matrix_square_root_psd(self.covariance_at_grid)
            rng = np.random.default_rng(int(self.uniform_band.random_state))
            gaussian_draws = (
                rng.standard_normal(
                    size=(int(self.uniform_band.n_boot), evaluation_grid_size)
                )
                @ covariance_sqrt.T
            )
            simulated_suprema = np.max(
                np.abs(gaussian_draws / self.uniform_standardization[None, :]),
                axis=1,
            )
            expected_uniform_critical_value = float(
                np.quantile(simulated_suprema, 1.0 - self.alpha)
            )
            expected_uniform_summary = _bootstrap_suprema_summary(
                simulated_suprema,
                alpha=self.alpha,
            )
            if not np.isclose(
                float(self.uniform_band.critical_value),
                expected_uniform_critical_value,
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    "uniform_band.critical_value must match the finite-grid "
                    "Gaussian bootstrap for covariance_at_grid"
                )
        uniform_band_metadata = self.optimization_metadata.get("uniform_band")
        if not isinstance(uniform_band_metadata, dict):
            raise ValueError("uniform_band metadata must be provided")
        metadata_critical_value = uniform_band_metadata.get("critical_value")
        metadata_n_boot = uniform_band_metadata.get("n_boot")
        metadata_random_state = uniform_band_metadata.get("random_state")
        if metadata_critical_value is None:
            raise ValueError("uniform_band metadata critical_value must be provided")
        if metadata_n_boot is None:
            raise ValueError("uniform_band metadata n_boot must be provided")
        if metadata_random_state is None:
            raise ValueError("uniform_band metadata random_state must be provided")
        metadata_critical_value_value = _coerce_finite_numeric(
            "uniform_band metadata critical_value",
            metadata_critical_value,
        )
        metadata_n_boot_value = _coerce_positive_integer(
            "uniform_band metadata n_boot",
            metadata_n_boot,
        )
        metadata_random_state_value = _coerce_optional_nonnegative_integer(
            "uniform_band metadata random_state",
            metadata_random_state,
        )
        if not np.isclose(
            metadata_critical_value_value,
            float(self.uniform_band.critical_value),
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "uniform_band metadata critical_value must match uniform_band"
            )
        if metadata_n_boot_value != int(self.uniform_band.n_boot):
            raise ValueError("uniform_band metadata n_boot must match uniform_band")
        if metadata_random_state_value != int(self.uniform_band.random_state):
            raise ValueError(
                "uniform_band metadata random_state must match uniform_band"
            )
        if self._validate_optimality:
            _validate_uniform_band_bootstrap_summary(
                uniform_band_metadata,
                expected_uniform_summary,
            )
        _require_interval_matches_center_scale(
            "uniform_band",
            self.uniform_band,
            center=self.bar_f_at_z0,
            scale=self.sigma_z_hat,
            multiplier=float(self.uniform_band.critical_value),
        )



def diagnose_nonparametric_omega_f(
    score_payload: ScorePayload,
    estimation_payload: EstimationPayload,
    nonparametric_payload: NonparametricInferencePayload,
) -> dict[str, object]:
    """Return paper-difference Omega_f component diagnostics for a valid payload."""

    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    x_valid = np.asarray(score_payload.x_valid, dtype=float)
    residual_valid = np.asarray(estimation_payload.residual_valid, dtype=float)
    n_valid = int(basis_valid_full.shape[0])
    target_metadata = {
        "oracle_lane": score_payload.oracle_lane,
        "basis_family": score_payload.basis_family,
        "basis_degree": int(score_payload.basis_degree),
        "n_valid_obs": n_valid,
    }
    if n_valid <= 0:
        _raise_inference_error(
            InvalidInferenceInputError,
            "nonparametric omega_f diagnostic requires at least one valid observation",
            failure_kind="invalid-input",
            valid_sample_size=n_valid,
            basis_valid_full_shape=_matrix_shape(basis_valid_full),
            target_kind="nonparametric-omega-f-diagnostic",
            **target_metadata,
        )
    _assert_estimation_payload_matches_score(
        score_payload,
        estimation_payload,
        target_kind="nonparametric-omega-f-diagnostic",
        check_projection_x=False,
        **target_metadata,
    )
    if x_valid.shape[0] != n_valid:
        _raise_inference_error(
            InvalidInferenceInputError,
            "x_valid must align with basis_valid_full",
            failure_kind="invalid-input",
            x_valid_rows=int(x_valid.shape[0]),
            basis_rows=n_valid,
            target_kind="nonparametric-omega-f-diagnostic",
            **target_metadata,
        )
    if residual_valid.shape[0] != n_valid:
        _raise_inference_error(
            InvalidInferenceInputError,
            "residual_valid must align with basis_valid_full",
            failure_kind="invalid-input",
            residual_length=int(residual_valid.shape[0]),
            basis_rows=n_valid,
            target_kind="nonparametric-omega-f-diagnostic",
            **target_metadata,
        )

    m_hat = np.asarray(nonparametric_payload.m_hat, dtype=float)
    sigma_x_hat = x_valid.T @ x_valid / n_valid
    cross_moment_hat = basis_valid_full.T @ x_valid / n_valid
    orthogonal_basis_valid = basis_valid_full - x_valid @ m_hat.T
    sigma_f_hat = orthogonal_basis_valid.T @ basis_valid_full / n_valid
    omega_f_hat = _paper_difference_omega_f_hat(
        basis_valid_full=basis_valid_full,
        x_valid=x_valid,
        residual_valid=residual_valid,
        m_hat=m_hat,
        n_valid=n_valid,
    )
    mismatch_metadata = {
        **target_metadata,
    }
    _raise_payload_mismatch(
        "sigma_x_hat",
        np.asarray(nonparametric_payload.sigma_x_hat, dtype=float),
        sigma_x_hat,
        target_kind="nonparametric-omega-f-diagnostic",
        **mismatch_metadata,
    )
    _raise_payload_mismatch(
        "cross_moment_hat",
        np.asarray(nonparametric_payload.cross_moment_hat, dtype=float),
        cross_moment_hat,
        target_kind="nonparametric-omega-f-diagnostic",
        **mismatch_metadata,
    )
    _raise_payload_mismatch(
        "orthogonal_basis_valid",
        np.asarray(nonparametric_payload.orthogonal_basis_valid, dtype=float),
        orthogonal_basis_valid,
        target_kind="nonparametric-omega-f-diagnostic",
        **mismatch_metadata,
    )
    _raise_payload_mismatch(
        "sigma_f_hat",
        np.asarray(nonparametric_payload.sigma_f_hat, dtype=float),
        sigma_f_hat,
        target_kind="nonparametric-omega-f-diagnostic",
        **mismatch_metadata,
    )
    _raise_payload_mismatch(
        "omega_f_hat",
        np.asarray(nonparametric_payload.omega_f_hat, dtype=float),
        omega_f_hat,
        target_kind="nonparametric-omega-f-diagnostic",
        **mismatch_metadata,
    )

    metadata = dict(nonparametric_payload.optimization_metadata)
    _require_omega_f_strategy_metadata(metadata)
    _require_omega_f_selected_eigenvalue_metadata(
        metadata,
        selected_min_eigenvalue=_minimum_symmetric_eigenvalue(omega_f_hat),
    )

    sigma_f_singular_values = np.linalg.svd(sigma_f_hat, compute_uv=False)
    sigma_f_min_singular_value = float(np.min(sigma_f_singular_values))
    sigma_f_max_singular_value = float(np.max(sigma_f_singular_values))
    sigma_f_condition_number = float(
        sigma_f_max_singular_value / sigma_f_min_singular_value
        if sigma_f_min_singular_value > 0.0
        else np.inf
    )
    return {
        "target_kind": "nonparametric-omega-f-diagnostic",
        "matrix_name": "omega_f_hat",
        "matrix_shape": _matrix_shape(nonparametric_payload.omega_f_hat),
        "basis_dimension_full": int(basis_valid_full.shape[1]),
        "x_dimension": int(x_valid.shape[1]),
        "n_valid_obs": n_valid,
        "sigma_x_min_eigenvalue": _minimum_symmetric_eigenvalue(sigma_x_hat),
        "sigma_x_rank": _matrix_rank(sigma_x_hat),
        "sigma_f_min_singular_value": sigma_f_min_singular_value,
        "sigma_f_max_singular_value": sigma_f_max_singular_value,
        "sigma_f_condition_number": sigma_f_condition_number,
        "eq43_solver": metadata.get("solver"),
        "eq43_feasible": bool(metadata.get("feasible", False)),
        "eq43_constraint_violation_max": float(
            metadata.get("constraint_violation_max", np.nan)
        ),
        "lambda_double_prime": float(metadata.get("lambda_double_prime", np.nan)),
        "omega_f_strategy": metadata.get("omega_f_strategy"),
        "omega_f_selected_strategy": metadata.get("omega_f_selected_strategy"),
        "omega_f_primary_strategy": metadata.get("omega_f_primary_strategy"),
        "omega_f_primary_psd": bool(metadata.get("omega_f_primary_psd", False)),
        "omega_f_fallback_attempted": bool(
            metadata.get("omega_f_fallback_attempted", False)
        ),
        "omega_f_fallback_used": bool(metadata.get("omega_f_fallback_used", False)),
        "omega_f_fallback_min_eigenvalue": metadata.get(
            "omega_f_fallback_min_eigenvalue"
        ),
        "omega_f_primary_min_eigenvalue": _minimum_symmetric_eigenvalue(omega_f_hat),
        "omega_f_selected_min_eigenvalue": _minimum_symmetric_eigenvalue(omega_f_hat),
        "paper_object": "Omega_f",
        "paper_formula": "E[sigma_i^2 psi psi'] - M E[sigma_i^2 X X'] M'",
        **_omega_f_paper_difference_component_metadata(
            basis_valid_full=basis_valid_full,
            x_valid=x_valid,
            residual_valid=residual_valid,
            m_hat=m_hat,
            sigma_x_hat=sigma_x_hat,
            cross_moment_hat=cross_moment_hat,
            omega_f_hat=omega_f_hat,
            orthogonal_basis_valid=orthogonal_basis_valid,
            n_valid=n_valid,
        ),
    }


def estimate_nonparametric_inference(
    score_payload: ScorePayload,
    estimation_payload: EstimationPayload,
    *,
    result: HDDIDResult | None = None,
    alpha: float = 0.05,
    lambda_double_prime: float = 0.0,
    max_threshold_steps: int = 25,
    n_boot: int = 1_000,
    random_state: int | None = None,
    allow_omega_f_fallback: bool = False,
) -> tuple[NonparametricInferencePayload, HDDIDResult]:
    """Perform debiased inference on the nonparametric function f(z).

    Constructs pointwise confidence intervals and a simultaneous
    confidence band for the nonparametric component f(z) using the
    debiasing approach of Theorem 3.

    The procedure:

    1. Solves the Eq. (12) projection matrix M to orthogonalize the
       sieve basis against high-dimensional covariates.
    2. Computes the orthogonalized basis ``psi_tilde = psi - M*X``.
    3. Estimates ``Sigma_f = E_n[psi_tilde psi']`` and
       ``Omega_f = E_n[sigma_i^2 psi psi'] - M E_n[sigma_i^2 X X'] M'``.
    4. Forms the sandwich ``V_f = Sigma_f^{-1} Omega_f Sigma_f^{-1}``.
    5. Debiases: ``bar_gamma = gamma - Sigma_f^{-1} score_moment``.
    6. Computes pointwise CI
       ``bar_f(z) +/- z_{1-alpha/2} * sqrt(psi(z)' V_f psi(z) / n)``.
    7. Computes uniform band via Gaussian bootstrap over the grid.

    Parameters
    ----------
    score_payload : ScorePayload
        Doubly-robust score payload containing basis matrices and
        valid-sample covariates.
    estimation_payload : EstimationPayload
        Output from :func:`estimate_eq31_mainline` containing
        gamma_hat, residuals, and beta_hat.
    result : HDDIDResult or None, default None
        Existing result to augment with inference outputs.
    alpha : float, default 0.05
        Significance level for confidence intervals and uniform band.
    lambda_double_prime : float, default 0.0
        Relaxation parameter for the Eq. (12) constraint.
    max_threshold_steps : int, default 25
        Maximum threshold steps for the projection solver.
    n_boot : int, default 1000
        Number of bootstrap replications for the uniform band.
    random_state : int or None, default None
        Random seed for the Gaussian bootstrap.
    allow_omega_f_fallback : bool, default False
        Whether to allow fallback when Omega_f is not PSD.
        Currently must be False (paper-difference is required).

    Returns
    -------
    payload : NonparametricInferencePayload
        Contains bar_gamma_hat, bar_f_at_z0, pointwise CI, uniform band,
        and all intermediate quantities (M, Sigma_f, Omega_f, V_f).
    result : HDDIDResult
        Updated result with nonparametric standard errors and
        confidence intervals.

    Raises
    ------
    InvalidInferenceInputError
        If payloads are inconsistent or dimensions mismatch.
    SparseDirectionInfeasibleError
        If the Eq. (12) projection solver is infeasible.
    SingularCovarianceError
        If Sigma_f_hat is singular.
    NonpositiveVarianceError
        If Omega_f or the pointwise variance is non-positive.
    MissingEvaluationGridError
        If the evaluation grid is empty.

    Notes
    -----
    Implements Eq. (12) and Theorem 3 from Ning, Peng & Tao (2020).
    The asymptotic normality result is
    ``sqrt(n) sigma_z^{-1/2} (f_bar(z) - f_0(z)) ->_d N(0, 1)``.

    Reference: Ning, Peng, and Tao (2020), arXiv preprint arXiv:2009.03151.
    """
    if not isinstance(allow_omega_f_fallback, bool):
        raise ValueError("allow_omega_f_fallback must be a boolean")
    if allow_omega_f_fallback:
        raise ValueError(
            "allow_omega_f_fallback is not supported; "
            "paper-difference omega_f is required"
        )
    alpha_value = _validate_alpha(alpha)
    n_boot_value = _coerce_positive_integer("n_boot", n_boot)
    rng_seed = _coerce_uniform_band_random_state(random_state)

    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    evaluation_basis = np.asarray(score_payload.evaluation_basis, dtype=float)
    x_valid = np.asarray(score_payload.x_valid, dtype=float)
    residual_valid = np.asarray(estimation_payload.residual_valid, dtype=float)
    gamma_hat = np.asarray(estimation_payload.gamma_hat, dtype=float)
    n_valid = int(basis_valid_full.shape[0])
    target_metadata = {
        "target_kind": "nonparametric",
        "oracle_lane": score_payload.oracle_lane,
        "basis_family": score_payload.basis_family,
        "basis_degree": int(score_payload.basis_degree),
        "n_valid_obs": n_valid,
    }
    if n_valid <= 0:
        _raise_inference_error(
            InvalidInferenceInputError,
            "nonparametric inference requires at least one valid observation",
            failure_kind="invalid-input",
            valid_sample_size=n_valid,
            basis_valid_full_shape=_matrix_shape(basis_valid_full),
            **target_metadata,
        )
    if x_valid.shape[0] != n_valid:
        _raise_inference_error(
            InvalidInferenceInputError,
            "x_valid must align with basis_valid_full",
            failure_kind="invalid-input",
            x_valid_rows=int(x_valid.shape[0]),
            basis_rows=n_valid,
            **target_metadata,
        )
    if residual_valid.shape[0] != n_valid:
        _raise_inference_error(
            InvalidInferenceInputError,
            "residual_valid must align with basis_valid_full",
            failure_kind="invalid-input",
            residual_length=int(residual_valid.shape[0]),
            basis_rows=n_valid,
            **target_metadata,
        )
    if gamma_hat.shape[0] != basis_valid_full.shape[1]:
        _raise_inference_error(
            InvalidInferenceInputError,
            "gamma_hat must align with basis_valid_full",
            failure_kind="invalid-input",
            gamma_dimension=int(gamma_hat.shape[0]),
            basis_dimension_full=int(basis_valid_full.shape[1]),
            **target_metadata,
        )
    if evaluation_basis.shape[1] != basis_valid_full.shape[1]:
        _raise_inference_error(
            InvalidInferenceInputError,
            "evaluation_basis must align with basis_valid_full",
            failure_kind="invalid-input",
            grid_dimension=int(evaluation_basis.shape[1]),
            basis_dimension_full=int(basis_valid_full.shape[1]),
            **target_metadata,
        )
    if evaluation_basis.shape[0] <= 0:
        _raise_inference_error(
            MissingEvaluationGridError,
            "uniform band requires an explicit non-empty evaluation grid",
            failure_kind="missing-evaluation-grid",
            grid_size=int(evaluation_basis.shape[0]),
            basis_dimension_full=int(basis_valid_full.shape[1]),
            **target_metadata,
        )
    _assert_estimation_payload_matches_score(
        score_payload,
        estimation_payload,
        target_kind="nonparametric",
        check_projection_x=False,
        oracle_lane=score_payload.oracle_lane,
        basis_family=score_payload.basis_family,
        basis_degree=int(score_payload.basis_degree),
        n_valid_obs=n_valid,
    )

    sigma_x_hat = x_valid.T @ x_valid / n_valid
    sigma_x_min_eigenvalue = _minimum_symmetric_eigenvalue(sigma_x_hat)
    cross_moment_hat = basis_valid_full.T @ x_valid / n_valid
    try:
        m_hat, eq43_metadata = solve_eq43_projection_matrix(
            sigma_x_hat=sigma_x_hat,
            cross_moment_hat=cross_moment_hat,
            lambda_double_prime=lambda_double_prime,
            max_threshold_steps=max_threshold_steps,
        )
    except SparseDirectionInfeasibleError as exc:
        _raise_inference_error(
            SparseDirectionInfeasibleError,
            str(exc),
            failure_kind="eq43-infeasible",
            x_dimension=int(x_valid.shape[1]),
            basis_dimension_full=int(basis_valid_full.shape[1]),
            grid_size=int(evaluation_basis.shape[0]),
            constraint_violation_max=float(
                exc.metadata.get(
                    "constraint_violation",
                    exc.metadata.get("constraint_violation_max", np.inf),
                )
            ),
            lambda_double_prime=float(
                exc.metadata.get(
                    "lambda_double_prime",
                    _coerce_nonnegative_finite(
                        "lambda_double_prime",
                        lambda_double_prime,
                    ),
                )
            ),
            row_index=exc.metadata.get("row_index"),
            linprog_status=exc.metadata.get("linprog_status"),
            linprog_message=exc.metadata.get("linprog_message"),
            solver_target_kind=exc.metadata.get("target_kind"),
            **target_metadata,
        )
    if not bool(eq43_metadata["feasible"]):
        _raise_inference_error(
            SparseDirectionInfeasibleError,
            "Eq. (4.3) projection solver failed to satisfy the constraint",
            failure_kind="eq43-infeasible",
            x_dimension=int(x_valid.shape[1]),
            basis_dimension_full=int(basis_valid_full.shape[1]),
            grid_size=int(evaluation_basis.shape[0]),
            constraint_violation_max=float(eq43_metadata["constraint_violation_max"]),
            lambda_double_prime=float(eq43_metadata["lambda_double_prime"]),
            **target_metadata,
        )

    orthogonal_basis_valid = basis_valid_full - x_valid @ m_hat.T
    sigma_f_hat = orthogonal_basis_valid.T @ basis_valid_full / n_valid
    sigma_f_singular_values = np.linalg.svd(sigma_f_hat, compute_uv=False)
    sigma_f_min_singular_value = float(np.min(sigma_f_singular_values))
    sigma_f_max_singular_value = float(np.max(sigma_f_singular_values))
    sigma_f_condition_number = float(
        sigma_f_max_singular_value / sigma_f_min_singular_value
        if sigma_f_min_singular_value > 0.0
        else np.inf
    )
    if sigma_f_min_singular_value <= _INFERENCE_EPS:
        _raise_inference_error(
            SingularCovarianceError,
            "sigma_f_hat must be full rank for nonparametric inference",
            failure_kind="singular-covariance",
            matrix_name="sigma_f_hat",
            matrix_shape=_matrix_shape(sigma_f_hat),
            matrix_rank=_matrix_rank(sigma_f_hat),
            sigma_f_min_singular_value=sigma_f_min_singular_value,
            sigma_f_max_singular_value=sigma_f_max_singular_value,
            sigma_f_condition_number=sigma_f_condition_number,
            x_dimension=int(x_valid.shape[1]),
            basis_dimension_full=int(basis_valid_full.shape[1]),
            grid_size=int(evaluation_basis.shape[0]),
            **target_metadata,
        )
    _assert_full_rank(
        sigma_f_hat,
        matrix_name="sigma_f_hat",
        sigma_f_min_singular_value=sigma_f_min_singular_value,
        sigma_f_max_singular_value=sigma_f_max_singular_value,
        sigma_f_condition_number=sigma_f_condition_number,
        x_dimension=int(x_valid.shape[1]),
        basis_dimension_full=int(basis_valid_full.shape[1]),
        grid_size=int(evaluation_basis.shape[0]),
        **target_metadata,
    )
    score_moment = np.mean(residual_valid[:, None] * orthogonal_basis_valid, axis=0)
    bar_gamma_hat = gamma_hat - _solve_sigma_f_left(sigma_f_hat, score_moment)
    omega_f_hat_primary = _paper_difference_omega_f_hat(
        basis_valid_full=basis_valid_full,
        x_valid=x_valid,
        residual_valid=residual_valid,
        m_hat=m_hat,
        n_valid=n_valid,
    )
    omega_f_primary_min_eigenvalue = _minimum_symmetric_eigenvalue(omega_f_hat_primary)
    omega_f_strategy = "paper-difference"
    omega_f_fallback_attempted = False
    omega_f_fallback_min_eigenvalue: float | None = None
    omega_f_hat = np.asarray(omega_f_hat_primary, dtype=float)
    v_f_hat = _sandwich_solve_sigma_f(sigma_f_hat, omega_f_hat)
    sigma_z_squared = _sigma_z_squared(
        evaluation_basis=evaluation_basis,
        v_f_hat=v_f_hat,
    )
    primary_omega_is_psd = omega_f_primary_min_eigenvalue >= -1e-10
    omega_f_selected_min_eigenvalue = _minimum_symmetric_eigenvalue(omega_f_hat)
    omega_f_fallback_used = False
    omega_f_invalidity_metadata = {
        "basis_dimension_full": int(basis_valid_full.shape[1]),
        "x_dimension": int(x_valid.shape[1]),
        "evaluation_grid_size": int(evaluation_basis.shape[0]),
        "sigma_x_min_eigenvalue": sigma_x_min_eigenvalue,
        "sigma_x_rank": _matrix_rank(sigma_x_hat),
        "sigma_f_min_singular_value": sigma_f_min_singular_value,
        "sigma_f_max_singular_value": sigma_f_max_singular_value,
        "sigma_f_condition_number": sigma_f_condition_number,
        "eq43_solver": str(eq43_metadata["solver"]),
        "eq43_feasible": bool(eq43_metadata["feasible"]),
        "eq43_constraint_violation_max": float(
            eq43_metadata["constraint_violation_max"]
        ),
        "lambda_double_prime": float(eq43_metadata["lambda_double_prime"]),
        "paper_object": "Omega_f",
        "paper_formula": "E[sigma_i^2 psi psi'] - M E[sigma_i^2 X X'] M'",
    }
    omega_f_component_metadata: dict[str, object] | None = None

    def _omega_f_component_metadata() -> dict[str, object]:
        nonlocal omega_f_component_metadata
        if omega_f_component_metadata is None:
            omega_f_component_metadata = _omega_f_paper_difference_component_metadata(
                basis_valid_full=basis_valid_full,
                x_valid=x_valid,
                residual_valid=residual_valid,
                m_hat=m_hat,
                sigma_x_hat=sigma_x_hat,
                cross_moment_hat=cross_moment_hat,
                omega_f_hat=omega_f_hat,
                orthogonal_basis_valid=orthogonal_basis_valid,
                n_valid=n_valid,
            )
        return omega_f_component_metadata

    if omega_f_selected_min_eigenvalue < -1e-10:
        _raise_inference_error(
            NonpositiveVarianceError,
            "selected omega_f_hat must be positive semidefinite",
            failure_kind="nonpositive-variance",
            matrix_name="omega_f_hat",
            matrix_shape=_matrix_shape(omega_f_hat),
            grid_size=int(evaluation_basis.shape[0]),
            omega_f_strategy=omega_f_strategy,
            omega_f_selected_strategy=omega_f_strategy,
            omega_f_primary_strategy="paper-difference",
            omega_f_primary_psd=primary_omega_is_psd,
            omega_f_primary_min_eigenvalue=omega_f_primary_min_eigenvalue,
            omega_f_fallback_attempted=omega_f_fallback_attempted,
            omega_f_fallback_used=omega_f_fallback_used,
            omega_f_fallback_min_eigenvalue=omega_f_fallback_min_eigenvalue,
            omega_f_selected_min_eigenvalue=omega_f_selected_min_eigenvalue,
            **omega_f_invalidity_metadata,
            **_omega_f_component_metadata(),
            **target_metadata,
        )

    if np.any(sigma_z_squared < -1e-10):
        _raise_inference_error(
            NonpositiveVarianceError,
            "estimated sigma_z squared must be non-negative",
            failure_kind="nonpositive-variance",
            matrix_name="sigma_z_squared",
            matrix_shape=_matrix_shape(sigma_z_squared),
            grid_size=int(evaluation_basis.shape[0]),
            omega_f_strategy=omega_f_strategy,
            omega_f_selected_strategy=omega_f_strategy,
            omega_f_primary_strategy="paper-difference",
            omega_f_primary_psd=primary_omega_is_psd,
            omega_f_primary_min_eigenvalue=omega_f_primary_min_eigenvalue,
            omega_f_fallback_attempted=omega_f_fallback_attempted,
            omega_f_fallback_used=omega_f_fallback_used,
            omega_f_fallback_min_eigenvalue=omega_f_fallback_min_eigenvalue,
            omega_f_selected_min_eigenvalue=omega_f_selected_min_eigenvalue,
            **omega_f_invalidity_metadata,
            **_omega_f_component_metadata(),
            **_nonpositive_value_metadata(
                sigma_z_squared,
                cutoff=-1e-10,
            ),
            **target_metadata,
        )
    if np.any(sigma_z_squared <= _INFERENCE_EPS):
        _raise_inference_error(
            NonpositiveVarianceError,
            "nonparametric inference requires strictly positive pointwise variance",
            failure_kind="nonpositive-variance",
            matrix_name="sigma_z_squared",
            matrix_shape=_matrix_shape(sigma_z_squared),
            grid_size=int(evaluation_basis.shape[0]),
            omega_f_strategy=omega_f_strategy,
            omega_f_selected_strategy=omega_f_strategy,
            omega_f_primary_strategy="paper-difference",
            omega_f_primary_psd=primary_omega_is_psd,
            omega_f_primary_min_eigenvalue=omega_f_primary_min_eigenvalue,
            omega_f_fallback_attempted=omega_f_fallback_attempted,
            omega_f_fallback_used=omega_f_fallback_used,
            omega_f_fallback_min_eigenvalue=omega_f_fallback_min_eigenvalue,
            omega_f_selected_min_eigenvalue=omega_f_selected_min_eigenvalue,
            **omega_f_invalidity_metadata,
            **_omega_f_component_metadata(),
            **_nonpositive_value_metadata(
                sigma_z_squared,
                cutoff=_INFERENCE_EPS,
            ),
            **target_metadata,
        )
    sigma_z_squared = np.maximum(sigma_z_squared, 0.0)
    sigma_z_hat = np.sqrt(sigma_z_squared / n_valid)
    bar_f_at_z0 = evaluation_basis @ bar_gamma_hat

    z_critical = NormalDist().inv_cdf(1.0 - alpha_value / 2.0)
    pointwise_confidence_interval = ConfidenceInterval(
        lower=bar_f_at_z0 - z_critical * sigma_z_hat,
        upper=bar_f_at_z0 + z_critical * sigma_z_hat,
        level=1.0 - alpha_value,
    )

    covariance_at_grid = evaluation_basis @ v_f_hat @ evaluation_basis.T / n_valid
    covariance_at_grid_min_eigenvalue = _minimum_symmetric_eigenvalue(
        covariance_at_grid
    )
    if covariance_at_grid_min_eigenvalue < -1e-10:
        _raise_inference_error(
            NonpositiveVarianceError,
            "uniform band covariance must be positive semidefinite",
            failure_kind="nonpositive-variance",
            matrix_name="covariance_at_grid",
            matrix_shape=_matrix_shape(covariance_at_grid),
            grid_size=int(evaluation_basis.shape[0]),
            covariance_at_grid_min_eigenvalue=covariance_at_grid_min_eigenvalue,
            omega_f_strategy=omega_f_strategy,
            omega_f_selected_strategy=omega_f_strategy,
            omega_f_primary_strategy="paper-difference",
            omega_f_primary_psd=primary_omega_is_psd,
            omega_f_primary_min_eigenvalue=omega_f_primary_min_eigenvalue,
            omega_f_fallback_attempted=omega_f_fallback_attempted,
            omega_f_fallback_used=omega_f_fallback_used,
            omega_f_fallback_min_eigenvalue=omega_f_fallback_min_eigenvalue,
            omega_f_selected_min_eigenvalue=omega_f_selected_min_eigenvalue,
            **omega_f_invalidity_metadata,
            **_omega_f_component_metadata(),
            **target_metadata,
        )
    standardization = np.sqrt(np.maximum(np.diag(covariance_at_grid), 0.0))
    if np.any(standardization <= np.finfo(float).eps):
        _raise_inference_error(
            NonpositiveVarianceError,
            "uniform band requires strictly positive pointwise variance",
            failure_kind="nonpositive-variance",
            matrix_name="covariance_at_grid",
            matrix_shape=_matrix_shape(covariance_at_grid),
            grid_size=int(evaluation_basis.shape[0]),
            omega_f_strategy=omega_f_strategy,
            omega_f_selected_strategy=omega_f_strategy,
            omega_f_primary_strategy="paper-difference",
            omega_f_primary_psd=primary_omega_is_psd,
            omega_f_primary_min_eigenvalue=omega_f_primary_min_eigenvalue,
            omega_f_fallback_attempted=omega_f_fallback_attempted,
            omega_f_fallback_used=omega_f_fallback_used,
            omega_f_fallback_min_eigenvalue=omega_f_fallback_min_eigenvalue,
            omega_f_selected_min_eigenvalue=omega_f_selected_min_eigenvalue,
            **omega_f_invalidity_metadata,
            **_omega_f_component_metadata(),
            **_nonpositive_value_metadata(
                standardization,
                cutoff=np.finfo(float).eps,
            ),
            **target_metadata,
        )
    uniform_critical_value, simulated_suprema, uniform_bootstrap_summary = compute_uniform_band_critical_value(
        covariance_at_grid=covariance_at_grid,
        standardization=standardization,
        alpha=alpha_value,
        n_boot=n_boot_value,
        random_state=rng_seed,
    )
    uniform_band = UniformBand(
        lower=bar_f_at_z0 - uniform_critical_value * sigma_z_hat,
        upper=bar_f_at_z0 + uniform_critical_value * sigma_z_hat,
        level=1.0 - alpha_value,
        critical_value=uniform_critical_value,
        n_boot=n_boot_value,
        random_state=rng_seed,
    )

    inference_metadata = {
        **eq43_metadata,
        "target_kind": "nonparametric",
        "n_valid_obs": n_valid,
        "basis_dimension_full": int(basis_valid_full.shape[1]),
        "x_dimension": int(x_valid.shape[1]),
        "evaluation_grid_size": int(evaluation_basis.shape[0]),
        "gamma_hat": gamma_hat,
        "sigma_x_min_eigenvalue": sigma_x_min_eigenvalue,
        "sigma_x_rank": _matrix_rank(sigma_x_hat),
        "sigma_f_min_singular_value": sigma_f_min_singular_value,
        "sigma_f_max_singular_value": sigma_f_max_singular_value,
        "sigma_f_condition_number": sigma_f_condition_number,
        "grid_size": int(evaluation_basis.shape[0]),
        "variance_positive": bool(np.all(sigma_z_squared > 0.0)),
        "oracle_lane": score_payload.oracle_lane,
        "omega_f_strategy": omega_f_strategy,
        "omega_f_selected_strategy": omega_f_strategy,
        "omega_f_primary_strategy": "paper-difference",
        "omega_f_primary_psd": primary_omega_is_psd,
        "omega_f_primary_min_eigenvalue": omega_f_primary_min_eigenvalue,
        "omega_f_fallback_attempted": omega_f_fallback_attempted,
        "omega_f_fallback_used": omega_f_fallback_used,
        "omega_f_fallback_min_eigenvalue": omega_f_fallback_min_eigenvalue,
        "omega_f_selected_min_eigenvalue": omega_f_selected_min_eigenvalue,
        "covariance_at_grid_min_eigenvalue": covariance_at_grid_min_eigenvalue,
        "bootstrap_seed": rng_seed,
        "uniform_band": {
            "simulation": "gaussian-finite-grid",
            "covariance_source": (
                "evaluation_basis @ v_f_hat @ evaluation_basis.T / n_valid"
            ),
            "standardization_source": "sqrt(diag(covariance_at_grid))",
            "critical_value": uniform_critical_value,
            "n_boot": n_boot_value,
            "random_state": rng_seed,
            **uniform_bootstrap_summary,
        },
    }

    payload = NonparametricInferencePayload(
        m_hat=m_hat,
        sigma_x_hat=sigma_x_hat,
        cross_moment_hat=cross_moment_hat,
        orthogonal_basis_valid=orthogonal_basis_valid,
        sigma_f_hat=sigma_f_hat,
        omega_f_hat=omega_f_hat,
        v_f_hat=v_f_hat,
        score_moment=score_moment,
        bar_gamma_hat=bar_gamma_hat,
        evaluation_basis=evaluation_basis,
        bar_f_at_z0=bar_f_at_z0,
        sigma_z_hat=sigma_z_hat,
        covariance_at_grid=covariance_at_grid,
        uniform_standardization=standardization,
        pointwise_confidence_interval=pointwise_confidence_interval,
        uniform_band=uniform_band,
        alpha=alpha_value,
        basis_family=score_payload.basis_family,
        basis_degree=score_payload.basis_degree,
        oracle_lane=score_payload.oracle_lane,
        optimization_metadata=inference_metadata,
    )

    updated_result = _base_result(score_payload, estimation_payload, result)
    updated_result.nonparametric_estimates = dict(
        updated_result.nonparametric_estimates
    )
    updated_result.standard_errors = dict(updated_result.standard_errors)
    updated_result.intervals = dict(updated_result.intervals)

    updated_result.nonparametric_estimates["bar_gamma_hat"] = np.asarray(
        payload.bar_gamma_hat,
        dtype=float,
    )
    updated_result.nonparametric_estimates["bar_f_at_z0"] = np.asarray(
        payload.bar_f_at_z0,
        dtype=float,
    )
    updated_result.standard_errors["f_se"] = np.asarray(
        payload.sigma_z_hat, dtype=float
    )
    updated_result.intervals["nonparametric_ci"] = payload.pointwise_confidence_interval
    updated_result.intervals["uniform_band"] = payload.uniform_band

    if updated_result.diagnostics is None:
        updated_result.diagnostics = _aggregate_result_diagnostics(
            score_payload,
            {
                **dict(estimation_payload.optimization_metadata),
                **inference_metadata,
            },
        )
    else:
        updated_result.diagnostics.optimization_metadata = {
            **dict(updated_result.diagnostics.optimization_metadata),
            **inference_metadata,
        }

    return payload, updated_result
