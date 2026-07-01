from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from statistics import NormalDist
from typing import ClassVar

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_diag

from .estimation import EstimationPayload, _aggregate_result_diagnostics
from .results import ConfidenceInterval, HDDIDResult
from .score import ScorePayload

from ._inference_common import (
    InferenceComputationError,
    InvalidInferenceInputError,
    NonpositiveVarianceError,
    SingularCovarianceError,
    SparseDirectionInfeasibleError,
    _INFERENCE_EPS,
    _assert_full_rank,
    _coerce_basis_degree,
    _coerce_matrix,
    _coerce_nonnegative_finite,
    _coerce_positive_integer,
    _coerce_vector,
    _coerce_xi,
    _matrix_rank,
    _matrix_shape,
    _nonpositive_value_metadata,
    _raise_inference_error,
    _require_finite,
    _require_interval_finite,
    _require_interval_level,
    _require_interval_matches_center_scale,
    _validate_alpha,
)


def _minimum_symmetric_eigenvalue(matrix: np.ndarray) -> float:
    """Return the minimum eigenvalue of the symmetric part of ``matrix``.

    Parameters
    ----------
    matrix : ndarray of shape (p, p)
        Input matrix (need not be symmetric; will be symmetrized).

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


def _max_constraint_violation(
    sigma_tilde_x_hat: np.ndarray,
    xi_row: np.ndarray,
    w_row: np.ndarray,
    *,
    lambda_prime: float,
) -> float:
    """Compute the maximum L-infinity constraint violation for Eq. (11).

    The Eq. (11) Dantzig selector constraint is:
        ||xi + Sigma_tilde_X @ w||_inf <= lambda'

    Parameters
    ----------
    sigma_tilde_x_hat : ndarray of shape (p, p)
        Estimated covariance of residualized covariates.
    xi_row : ndarray of shape (p,)
        Target vector for the current direction.
    w_row : ndarray of shape (p,)
        Candidate sparse direction.
    lambda_prime : float
        Relaxation parameter.

    Returns
    -------
    float
        Maximum violation ``max(0, ||xi + Sigma @ w||_inf - lambda')``.
        Zero indicates feasibility.
    """
    residual = xi_row + sigma_tilde_x_hat @ w_row
    if residual.size == 0:
        return 0.0
    return max(0.0, float(np.max(np.abs(residual)) - lambda_prime))


# Estimated non-zero entries threshold for the block-diagonal LP batch.
# Each Eq. (4.2) row contributes a (2p, 2p) dense block, i.e. 4p^2 nonzeros.
# Above this threshold we fall back to row-by-row solving to avoid excessive
# memory use.
_LP_BATCH_NNZ_THRESHOLD = 10_000_000


def _solve_eq42_batch_direction(
    sigma_matrix: np.ndarray,
    xi_matrix: np.ndarray,
    *,
    lambda_prime: float,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    """Solve all Eq. (4.2) directions as one block-diagonal LP.

    Each row of ``xi_matrix`` corresponds to an independent LP with ``2p``
    variables and ``2p`` constraints.  Because the row blocks are disjoint,
    merging them into a single LP with a block-diagonal constraint matrix is
    mathematically equivalent to solving the rows separately: the objective
    and every constraint are identical, only concatenated.

    Parameters
    ----------
    sigma_matrix : ndarray of shape (p, p)
        Validated covariance matrix.
    xi_matrix : ndarray of shape (k, p)
        Validated target matrix.
    lambda_prime : float
        Validated relaxation parameter.

    Returns
    -------
    w_rows : ndarray of shape (k, p)
        Optimal sparse direction matrix.
    row_metadata : list[dict]
        Per-row solver diagnostics.
    batch_metadata : dict
        Diagnostics for the combined solve, including ``linprog_status``,
        ``linprog_message`` and ``batch_solved``.
    """
    x_dimension = int(sigma_matrix.shape[0])
    xi_count = int(xi_matrix.shape[0])
    var_per_row = 2 * x_dimension
    constraint_per_row = 2 * x_dimension

    objective = np.ones(var_per_row * xi_count, dtype=float)
    constraint_blocks: list[np.ndarray] = []
    constraint_bounds_list: list[np.ndarray] = []

    for row_index in range(xi_count):
        xi_row = xi_matrix[row_index, :]
        block_rows = np.empty((constraint_per_row, var_per_row), dtype=float)
        block_bounds = np.empty(constraint_per_row, dtype=float)
        for i in range(x_dimension):
            sigma_row = sigma_matrix[i, :]
            positive_row = np.concatenate([sigma_row, -sigma_row])
            block_rows[2 * i, :] = positive_row
            block_bounds[2 * i] = float(lambda_prime - xi_row[i])
            block_rows[2 * i + 1, :] = -positive_row
            block_bounds[2 * i + 1] = float(lambda_prime + xi_row[i])
        constraint_blocks.append(block_rows)
        constraint_bounds_list.append(block_bounds)

    result = linprog(
        objective,
        A_ub=block_diag(constraint_blocks, format="coo"),
        b_ub=np.concatenate(constraint_bounds_list),
        bounds=[(0.0, None)] * (var_per_row * xi_count),
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
        nan_array = np.full((xi_count, x_dimension), np.nan, dtype=float)
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
            for row_index in range(xi_count)
        ]
        return nan_array, row_metadata, batch_metadata

    w_rows = np.empty((xi_count, x_dimension), dtype=float)
    row_metadata: list[dict[str, object]] = []
    for row_index in range(xi_count):
        x_start = row_index * var_per_row
        x_end = x_start + var_per_row
        x_row = result.x[x_start:x_end]
        w_row = np.asarray(
            x_row[:x_dimension] - x_row[x_dimension:],
            dtype=float,
        )
        w_rows[row_index] = w_row
        best_l1 = float(np.sum(np.abs(w_row)))
        best_violation = _max_constraint_violation(
            sigma_matrix,
            xi_matrix[row_index, :],
            w_row,
            lambda_prime=lambda_prime,
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
    total_l1 = float(np.sum(np.abs(w_rows)))
    if not np.isclose(result.fun, total_l1, rtol=1e-10, atol=1e-10):
        raise RuntimeError(
            "Eq. (4.2) batch LP objective does not match the sum of "
            "per-row L1 norms; extraction may be incorrect"
        )

    return w_rows, row_metadata, batch_metadata


def _solve_eq42_rows_direction(
    sigma_matrix: np.ndarray,
    xi_matrix: np.ndarray,
    *,
    lambda_prime: float,
    max_threshold_steps: int,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    """Solve Eq. (4.2) directions row-by-row (original fallback path)."""
    xi_count = int(xi_matrix.shape[0])
    x_dimension = int(sigma_matrix.shape[0])
    w_rows = np.zeros((xi_count, x_dimension), dtype=float)
    row_metadata: list[dict[str, object]] = []
    for row_index, xi_row in enumerate(xi_matrix):
        w_row, metadata = _solve_eq42_single_direction(
            sigma_matrix,
            np.asarray(xi_row, dtype=float),
            lambda_prime=lambda_prime,
            max_threshold_steps=max_threshold_steps,
        )
        w_rows[row_index] = w_row
        row_metadata.append({"row_index": row_index, **metadata})
    fallback_metadata: dict[str, object] = {
        "batch_solved": False,
        "linprog_status": None,
        "linprog_message": None,
    }
    return w_rows, row_metadata, fallback_metadata


def _solve_eq42_single_direction(
    sigma_tilde_x_hat: np.ndarray,
    xi_row: np.ndarray,
    *,
    lambda_prime: float,
    max_threshold_steps: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve a single Eq. (11) Dantzig direction via linear programming.

    For a single target row ``xi_row``, finds the L1-minimal direction
    ``w`` satisfying the Dantzig selector constraint.

    Parameters
    ----------
    sigma_tilde_x_hat : ndarray of shape (p, p)
        Covariance of residualized covariates.
    xi_row : ndarray of shape (p,)
        Target vector.
    lambda_prime : float
        Relaxation parameter.
    max_threshold_steps : int
        Reserved for alternative solvers (unused by the LP path).

    Returns
    -------
    w_row : ndarray of shape (p,)
        Optimal sparse direction.
    metadata : dict
        Per-row solver diagnostics.
    """
    x_dimension = int(sigma_tilde_x_hat.shape[0])
    objective = np.ones(2 * x_dimension, dtype=float)
    constraint_rows = []
    constraint_bounds = []
    for row_index in range(x_dimension):
        sigma_row = sigma_tilde_x_hat[row_index, :]
        positive_row = np.concatenate([sigma_row, -sigma_row])
        constraint_rows.append(positive_row)
        constraint_bounds.append(float(lambda_prime - xi_row[row_index]))
        constraint_rows.append(-positive_row)
        constraint_bounds.append(float(lambda_prime + xi_row[row_index]))

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
        best_w = np.asarray(
            result.x[:x_dimension] - result.x[x_dimension:],
            dtype=float,
        )
        best_l1 = float(result.fun)
        best_violation = _max_constraint_violation(
            sigma_tilde_x_hat,
            xi_row,
            best_w,
            lambda_prime=lambda_prime,
        )
    else:
        best_w = np.full(x_dimension, np.nan, dtype=float)
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
    return np.asarray(best_w, dtype=float), row_metadata


def solve_eq42_sparse_direction(
    *,
    sigma_tilde_x_hat: np.ndarray,
    xi: np.ndarray,
    lambda_prime: float,
    max_threshold_steps: int = 25,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve the Eq. (4.2) sparse direction optimization.

    For each row of xi, finds a direction vector w that minimizes
    ||w||_1 subject to the constraint
    ||xi_j + w_j' Sigma_tilde_X||_inf <= lambda'.

    This direction is used to debias the parametric estimate in
    the high-dimensional setting where direct inversion of
    Sigma_tilde_X is infeasible.

    Parameters
    ----------
    sigma_tilde_x_hat : ndarray of shape (p, p)
        Estimated covariance of the projected (residualized) covariates.
        Must be symmetric and positive semidefinite.
    xi : ndarray of shape (k, p) or (p, p)
        Target matrix specifying which linear functionals of beta to
        perform inference on. When None is passed to
        :func:`estimate_parametric_inference`, defaults to the identity
        matrix (componentwise inference).
    lambda_prime : float
        Relaxation parameter controlling the constraint tolerance.
        Typically set to 0 for exact solutions.
    max_threshold_steps : int, default 25
        Maximum threshold search steps (reserved for alternative
        solvers; the LP solver uses this as metadata only).

    Returns
    -------
    w_hat : ndarray of shape (k, p)
        Optimal sparse direction matrix, one row per xi target.
    metadata : dict
        Solver diagnostics including feasibility status, constraint
        violations, and per-row metadata.

    Raises
    ------
    ValueError
        If sigma_tilde_x_hat is not square, not symmetric, or not
        positive semidefinite; if xi has incompatible dimensions.
    SparseDirectionInfeasibleError
        If the linear program fails to find a feasible solution for
        any target row.

    Notes
    -----
    Implements Eq. (4.2) from Ning, Peng, and Tao (2020).
    The optimization is solved via linear programming (HiGHS).
    Reference: Ning, Peng, and Tao (2020), arXiv preprint arXiv:2009.03151.
    """
    sigma_matrix = _coerce_matrix("sigma_tilde_x_hat", sigma_tilde_x_hat)
    if sigma_matrix.shape[0] != sigma_matrix.shape[1]:
        raise ValueError("sigma_tilde_x_hat must be square")
    if not np.allclose(
        sigma_matrix,
        sigma_matrix.T,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("sigma_tilde_x_hat must be symmetric")
    if _minimum_symmetric_eigenvalue(sigma_matrix) < -1e-10:
        raise ValueError("sigma_tilde_x_hat must be positive semidefinite")
    lambda_prime_value = _coerce_nonnegative_finite("lambda_prime", lambda_prime)
    max_threshold_steps_value = _coerce_positive_integer(
        "max_threshold_steps",
        max_threshold_steps,
    )

    xi_matrix = _coerce_xi(xi, sigma_matrix.shape[0])
    if sigma_matrix.shape[0] == 0:
        raise ValueError("sigma_tilde_x_hat must have positive Eq. (4.2) dimension")
    if xi_matrix.shape[0] == 0:
        raise ValueError("xi must contain at least one parametric target")
    xi_count = int(xi_matrix.shape[0])
    x_dimension = int(sigma_matrix.shape[0])
    # Each row adds a (2p, 2p) dense block => 4 * k * p^2 nonzeros.
    estimated_batch_nnz = 4 * xi_count * x_dimension * x_dimension

    w_rows = np.zeros_like(xi_matrix, dtype=float)
    row_metadata: list[dict[str, object]] = []
    use_batch = (
        estimated_batch_nnz <= _LP_BATCH_NNZ_THRESHOLD
        and xi_count > 0
        and x_dimension > 0
    )
    batch_failed = False

    if use_batch:
        try:
            w_rows, batch_row_metadata, _ = _solve_eq42_batch_direction(
                sigma_matrix,
                xi_matrix,
                lambda_prime=lambda_prime_value,
            )
            if all(bool(m["feasible"]) for m in batch_row_metadata):
                row_metadata = batch_row_metadata
            else:
                batch_failed = True
        except Exception:
            batch_failed = True

    if batch_failed or not row_metadata:
        w_rows, row_metadata, _ = _solve_eq42_rows_direction(
            sigma_matrix,
            xi_matrix,
            lambda_prime=lambda_prime_value,
            max_threshold_steps=max_threshold_steps_value,
        )

    for row_index, metadata in enumerate(row_metadata):
        if not bool(metadata["feasible"]):
            _raise_inference_error(
                SparseDirectionInfeasibleError,
                "Eq. (4.2) sparse direction solver failed to satisfy the constraint",
                failure_kind="eq42-infeasible",
                target_kind="eq42-sparse-direction",
                row_index=row_index,
                xi_count=int(xi_matrix.shape[0]),
                sigma_tilde_x_dimension=int(sigma_matrix.shape[0]),
                constraint_violation=float(metadata["constraint_violation"]),
                lambda_prime=lambda_prime_value,
                linprog_status=int(metadata["linprog_status"]),
                linprog_message=str(metadata["linprog_message"]),
            )

    constraint_violation_max = (
        max(float(metadata["constraint_violation"]) for metadata in row_metadata)
        if row_metadata
        else 0.0
    )
    metadata = {
        "solver": "eq42-linear-programming-l1",
        "lambda_prime": lambda_prime_value,
        "max_threshold_steps": max_threshold_steps_value,
        "xi_count": int(xi_matrix.shape[0]),
        "sigma_tilde_x_dimension": int(sigma_matrix.shape[0]),
        "constraint_violation_max": constraint_violation_max,
        "feasible": constraint_violation_max <= 1e-10,
        "row_metadata": row_metadata,
    }
    return w_rows, metadata


@dataclass(slots=True)
class ParametricInferencePayload:
    """Debiased inference output for the parametric component beta, Eq. (4.1).

    Contains the debiased point estimates t_hat, asymptotic standard errors,
    and confidence intervals for linear functionals xi'beta, constructed
    via the Eq. (4.2) sparse direction w.

    Attributes
    ----------
    xi : ndarray of float, shape (k, p)
        Target matrix specifying linear functionals of beta.
    t_hat : ndarray of float, shape (k,)
        Debiased point estimates xi'beta - w' score_moment.
    w_hat : ndarray of float, shape (k, p)
        Sparse debiasing direction from Eq. (4.2) optimization.
    sigma_tilde_x_hat : ndarray of float, shape (p, p)
        Estimated covariance of the projected covariates.
    omega_beta_hat : ndarray of float, shape (p, p)
        Long-run variance matrix for the parametric score.
    score_moment : ndarray of float, shape (p,)
        Sample mean of the residualized score epsilon * X_tilde.
    asymptotic_variance_hat : ndarray of float, shape (k,)
        Estimated asymptotic variance w' Omega_beta w per target.
    standard_errors : ndarray of float, shape (k,)
        Standard errors sqrt(asymptotic_variance / n).
    confidence_interval : ConfidenceInterval
        Two-sided confidence intervals for each target.
    alpha : float
        Significance level for the confidence intervals.
    basis_family : str
        Sieve basis family.
    basis_degree : int
        Sieve truncation parameter.
    oracle_lane : str
        Computational lane identifier.
    optimization_metadata : dict
        Solver diagnostics for the Eq. (4.2) direction problem.
    """

    xi: np.ndarray
    t_hat: np.ndarray
    w_hat: np.ndarray
    sigma_tilde_x_hat: np.ndarray
    omega_beta_hat: np.ndarray
    score_moment: np.ndarray
    asymptotic_variance_hat: np.ndarray
    standard_errors: np.ndarray
    confidence_interval: ConfidenceInterval
    alpha: float
    basis_family: str
    basis_degree: int
    oracle_lane: str
    optimization_metadata: dict[str, object]

    _validate_optimality: ClassVar[bool] = False

    def __post_init__(self) -> None:
        self.xi = _coerce_matrix("xi", self.xi)
        self.t_hat = _coerce_vector("t_hat", self.t_hat)
        self.w_hat = _coerce_matrix("w_hat", self.w_hat)
        self.sigma_tilde_x_hat = _coerce_matrix(
            "sigma_tilde_x_hat", self.sigma_tilde_x_hat
        )
        self.omega_beta_hat = _coerce_matrix("omega_beta_hat", self.omega_beta_hat)
        self.score_moment = _coerce_vector("score_moment", self.score_moment)
        self.asymptotic_variance_hat = _coerce_vector(
            "asymptotic_variance_hat",
            self.asymptotic_variance_hat,
        )
        self.standard_errors = _coerce_vector("standard_errors", self.standard_errors)
        self.alpha = _validate_alpha(self.alpha)
        self.basis_family = str(self.basis_family).strip().lower()
        self.basis_degree = _coerce_basis_degree(
            self.basis_family,
            self.basis_degree,
        )
        self.oracle_lane = str(self.oracle_lane)
        self.optimization_metadata = dict(self.optimization_metadata)

        _require_interval_finite(
            "confidence_interval",
            self.confidence_interval,
        )
        _require_interval_level(
            "confidence_interval",
            self.confidence_interval,
            self.alpha,
        )
        _require_finite("asymptotic_variance_hat", self.asymptotic_variance_hat)
        _require_finite("standard_errors", self.standard_errors)

        if self.xi.shape != self.w_hat.shape:
            raise ValueError("w_hat must align with xi")
        beta_dimension = self.xi.shape[1]
        if self.sigma_tilde_x_hat.shape[0] != self.sigma_tilde_x_hat.shape[1]:
            raise ValueError("sigma_tilde_x_hat must be square")
        if self.sigma_tilde_x_hat.shape != (beta_dimension, beta_dimension):
            raise ValueError("sigma_tilde_x_hat must align with xi columns")
        if not np.allclose(
            self.sigma_tilde_x_hat,
            self.sigma_tilde_x_hat.T,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("sigma_tilde_x_hat must be symmetric")
        if _minimum_symmetric_eigenvalue(self.sigma_tilde_x_hat) < -1e-10:
            raise ValueError("sigma_tilde_x_hat must be positive semidefinite")
        if self.omega_beta_hat.shape != self.sigma_tilde_x_hat.shape:
            raise ValueError("omega_beta_hat must align with sigma_tilde_x_hat")
        if not np.allclose(
            self.omega_beta_hat,
            self.omega_beta_hat.T,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("omega_beta_hat must be symmetric")
        if _minimum_symmetric_eigenvalue(self.omega_beta_hat) < -1e-10:
            raise ValueError("omega_beta_hat must be positive semidefinite")
        if self.score_moment.shape[0] != beta_dimension:
            raise ValueError("score_moment must align with xi columns")
        xi_count = self.xi.shape[0]
        if xi_count == 0:
            raise ValueError("xi must contain at least one parametric target")
        beta_hat_metadata = self.optimization_metadata.get("beta_hat")
        if beta_hat_metadata is None:
            raise ValueError("beta_hat metadata must be provided to validate t_hat")
        beta_hat = _coerce_vector("beta_hat", beta_hat_metadata)
        if beta_hat.shape[0] != beta_dimension:
            raise ValueError("beta_hat metadata must align with xi columns")
        expected_t_hat = self.xi @ beta_hat - np.einsum(
            "ij,j->i",
            self.w_hat,
            self.score_moment,
        )
        if not np.allclose(self.t_hat, expected_t_hat, atol=1e-12, rtol=1e-10):
            raise ValueError(
                "t_hat must equal xi @ beta_hat - w_hat @ score_moment"
            )
        lambda_prime = self.optimization_metadata.get("lambda_prime")
        if lambda_prime is not None:
            lambda_prime_value = _coerce_nonnegative_finite(
                "lambda_prime",
                lambda_prime,
            )
            eq42_residual = self.xi + self.w_hat @ self.sigma_tilde_x_hat
            eq42_violation = (
                0.0
                if eq42_residual.size == 0
                else max(
                    0.0,
                    float(np.max(np.abs(eq42_residual)) - lambda_prime_value),
                )
            )
            if eq42_violation > 1e-10:
                raise ValueError(
                    "w_hat must satisfy the Eq. (4.2) sparse direction constraint"
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
                expected_w_hat, _ = solve_eq42_sparse_direction(
                    sigma_tilde_x_hat=self.sigma_tilde_x_hat,
                    xi=self.xi,
                    lambda_prime=lambda_prime_value,
                    max_threshold_steps=max_threshold_steps_value,
                )
                expected_l1 = np.sum(np.abs(expected_w_hat), axis=1)
                actual_l1 = np.sum(np.abs(self.w_hat), axis=1)
                if not np.allclose(
                    actual_l1,
                    expected_l1,
                    atol=1e-12,
                    rtol=1e-10,
                ):
                    raise ValueError(
                        "w_hat must attain the Eq. (4.2) sparse direction L1 optimum"
                    )
        if self.t_hat.shape[0] != xi_count:
            raise ValueError("t_hat must align with xi")
        if self.asymptotic_variance_hat.shape[0] != xi_count:
            raise ValueError("asymptotic_variance_hat must align with xi")
        if self.standard_errors.shape[0] != xi_count:
            raise ValueError("standard_errors must align with xi")
        if np.any(self.asymptotic_variance_hat <= _INFERENCE_EPS):
            raise ValueError("asymptotic_variance_hat must be strictly positive")
        expected_asymptotic_variance_hat = np.einsum(
            "ij,jk,ik->i",
            self.w_hat,
            self.omega_beta_hat,
            self.w_hat,
        )
        if not np.allclose(
            self.asymptotic_variance_hat,
            expected_asymptotic_variance_hat,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "asymptotic_variance_hat must equal "
                "w_hat @ omega_beta_hat @ w_hat"
            )
        if np.any(self.standard_errors <= _INFERENCE_EPS):
            raise ValueError("standard_errors must be strictly positive")
        n_valid_obs = self.optimization_metadata.get("n_valid_obs")
        if n_valid_obs is None:
            raise ValueError("n_valid_obs must be provided to validate standard_errors")
        n_valid_value = _coerce_positive_integer("n_valid_obs", n_valid_obs)
        expected_standard_errors = np.sqrt(
            self.asymptotic_variance_hat / n_valid_value
        )
        if not np.allclose(
            self.standard_errors,
            expected_standard_errors,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "standard_errors must equal "
                "sqrt(asymptotic_variance_hat / n_valid_obs)"
            )
        z_critical = NormalDist().inv_cdf(1.0 - self.alpha / 2.0)
        _require_interval_matches_center_scale(
            "confidence_interval",
            self.confidence_interval,
            center=self.t_hat,
            scale=self.standard_errors,
            multiplier=z_critical,
        )


def _componentwise_default(xi_matrix: np.ndarray) -> bool:
    """Check whether ``xi_matrix`` is the identity (componentwise inference)."""
    beta_dimension = xi_matrix.shape[1]
    return xi_matrix.shape == (beta_dimension, beta_dimension) and np.allclose(
        xi_matrix,
        np.eye(beta_dimension, dtype=float),
        atol=1e-12,
        rtol=0.0,
    )


def _base_result(
    score_payload: ScorePayload,
    estimation_payload: EstimationPayload,
    result: HDDIDResult | None,
) -> HDDIDResult:
    """Return an HDDIDResult with estimation-level estimates populated."""
    if result is not None:
        if not isinstance(result, HDDIDResult):
            raise ValueError("result must be an HDDIDResult")
        result_copy = deepcopy(result)
        return HDDIDResult(
            parametric_estimates=result_copy.parametric_estimates,
            nonparametric_estimates=result_copy.nonparametric_estimates,
            standard_errors=result_copy.standard_errors,
            intervals=result_copy.intervals,
            diagnostics=result_copy.diagnostics,
        )

    return HDDIDResult(
        parametric_estimates={
            "beta_hat": np.asarray(estimation_payload.beta_hat, dtype=float)
        },
        nonparametric_estimates={
            "gamma_hat": np.asarray(estimation_payload.gamma_hat, dtype=float),
            "f_hat_at_z0": np.asarray(estimation_payload.f_hat_at_z0, dtype=float),
        },
        standard_errors={},
        intervals={},
        diagnostics=_aggregate_result_diagnostics(
            score_payload,
            dict(estimation_payload.optimization_metadata),
        ),
    )


def _assert_estimation_payload_matches_score(
    score_payload: ScorePayload,
    estimation_payload: EstimationPayload,
    *,
    target_kind: str,
    check_projection_x: bool,
    **metadata: object,
) -> None:
    """Validate that estimation_payload is consistent with score_payload."""
    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    evaluation_basis = np.asarray(score_payload.evaluation_basis, dtype=float)
    x_valid = np.asarray(score_payload.x_valid, dtype=float)
    s_hat_valid = np.asarray(score_payload.s_hat_valid, dtype=float)
    beta_hat = np.asarray(estimation_payload.beta_hat, dtype=float)
    gamma_hat = np.asarray(estimation_payload.gamma_hat, dtype=float)
    fitted_f_valid = np.asarray(estimation_payload.fitted_f_valid, dtype=float)
    second_stage_prediction_valid = np.asarray(
        estimation_payload.second_stage_prediction_valid,
        dtype=float,
    )
    residual_valid = np.asarray(estimation_payload.residual_valid, dtype=float)
    projection_x_valid = np.asarray(estimation_payload.projection_x_valid, dtype=float)
    f_hat_at_z0 = np.asarray(estimation_payload.f_hat_at_z0, dtype=float)

    expected_shapes = {
        "beta_hat": (x_valid.shape[1],),
        "gamma_hat": (basis_valid_full.shape[1],),
        "fitted_f_valid": (basis_valid_full.shape[0],),
        "second_stage_prediction_valid": (basis_valid_full.shape[0],),
        "residual_valid": (basis_valid_full.shape[0],),
        "f_hat_at_z0": (evaluation_basis.shape[0],),
    }
    if check_projection_x:
        expected_shapes["projection_x_valid"] = x_valid.shape
    actual_shapes = {
        "beta_hat": beta_hat.shape,
        "gamma_hat": gamma_hat.shape,
        "fitted_f_valid": fitted_f_valid.shape,
        "second_stage_prediction_valid": second_stage_prediction_valid.shape,
        "residual_valid": residual_valid.shape,
        "f_hat_at_z0": f_hat_at_z0.shape,
    }
    if check_projection_x:
        actual_shapes["projection_x_valid"] = projection_x_valid.shape
    for field_name, expected_shape in expected_shapes.items():
        if actual_shapes[field_name] != expected_shape:
            _raise_inference_error(
                InvalidInferenceInputError,
                f"{field_name} must align with the current score payload",
                failure_kind="invalid-input",
                field_name=field_name,
                expected_shape=expected_shape,
                actual_shape=actual_shapes[field_name],
                target_kind=target_kind,
                **metadata,
            )

    expected_fitted_f_valid = basis_valid_full @ gamma_hat
    expected_second_stage_prediction_valid = x_valid @ beta_hat + fitted_f_valid
    expected_residual_valid = s_hat_valid - second_stage_prediction_valid
    expected_f_hat_at_z0 = evaluation_basis @ gamma_hat

    checks = [
        ("fitted_f_valid", fitted_f_valid, expected_fitted_f_valid),
        (
            "second_stage_prediction_valid",
            second_stage_prediction_valid,
            expected_second_stage_prediction_valid,
        ),
        ("residual_valid", residual_valid, expected_residual_valid),
        ("f_hat_at_z0", f_hat_at_z0, expected_f_hat_at_z0),
    ]
    if check_projection_x:
        projected_x_valid = basis_valid_full @ np.linalg.lstsq(
            basis_valid_full,
            x_valid,
            rcond=None,
        )[0]
        expected_projection_x_valid = x_valid - projected_x_valid
        checks.insert(
            0,
            ("projection_x_valid", projection_x_valid, expected_projection_x_valid),
        )
    for field_name, actual, expected in checks:
        if not np.allclose(actual, expected, rtol=1e-9, atol=1e-9):
            max_abs_diff = float(np.max(np.abs(actual - expected)))
            _raise_inference_error(
                InvalidInferenceInputError,
                f"{field_name} does not match the current score payload",
                failure_kind="payload-score-mismatch",
                field_name=field_name,
                max_abs_diff=max_abs_diff,
                target_kind=target_kind,
                **metadata,
            )


def _raise_payload_mismatch(
    field_name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    target_kind: str,
    **metadata: object,
) -> None:
    """Raise an InvalidInferenceInputError if actual and expected differ."""
    if actual.shape != expected.shape:
        _raise_inference_error(
            InvalidInferenceInputError,
            f"{field_name} must align with the current score and inference payloads",
            failure_kind="invalid-input",
            field_name=field_name,
            expected_shape=_matrix_shape(expected),
            actual_shape=_matrix_shape(actual),
            target_kind=target_kind,
            **metadata,
        )
    if not np.allclose(actual, expected, atol=1e-9, rtol=1e-9):
        _raise_inference_error(
            InvalidInferenceInputError,
            f"{field_name} does not match the current score and inference payloads",
            failure_kind="payload-score-mismatch",
            field_name=field_name,
            max_abs_diff=float(np.max(np.abs(actual - expected))),
            target_kind=target_kind,
            **metadata,
        )


def estimate_parametric_inference(
    score_payload: ScorePayload,
    estimation_payload: EstimationPayload,
    *,
    result: HDDIDResult | None = None,
    xi: np.ndarray | None = None,
    alpha: float = 0.05,
    lambda_prime: float = 0.0,
    max_threshold_steps: int = 25,
) -> tuple[ParametricInferencePayload, HDDIDResult]:
    """Perform debiased inference on the parametric component beta.

    Constructs asymptotically normal confidence intervals for linear
    functionals xi' beta using the debiasing approach of Eq. (4.1).
    The debiasing direction w is obtained by solving the Eq. (4.2)
    sparse optimization problem.

    Parameters
    ----------
    score_payload : ScorePayload
        Doubly-robust score payload.
    estimation_payload : EstimationPayload
        Output from :func:`estimate_eq31_mainline` containing
        beta_hat, residuals, and the projected covariate matrix.
    result : HDDIDResult or None, default None
        Existing result object to augment with inference outputs.
        When None, a fresh result is created.
    xi : ndarray of shape (k, p) or None, default None
        Target matrix for inference. When None, defaults to I_p
        (componentwise inference on each element of beta).
    alpha : float, default 0.05
        Significance level for the two-sided confidence intervals.
    lambda_prime : float, default 0.0
        Relaxation parameter for the Eq. (4.2) constraint.
    max_threshold_steps : int, default 25
        Maximum threshold steps passed to the sparse direction solver.

    Returns
    -------
    payload : ParametricInferencePayload
        Contains t_hat (debiased estimates), standard errors,
        confidence intervals, and all intermediate quantities.
    result : HDDIDResult
        Updated result with parametric standard errors and
        confidence intervals.

    Raises
    ------
    InvalidInferenceInputError
        If payloads are inconsistent or dimension mismatches occur.
    NonpositiveVarianceError
        If the estimated asymptotic variance is non-positive.
    SparseDirectionInfeasibleError
        If the Eq. (4.2) solver cannot find a feasible direction.

    Notes
    -----
    Implements Eq. (4.1) and (4.2) from Ning, Peng, and Tao (2020).
    The debiased estimator is:

        t_hat_j = xi_j' beta_hat - w_j' (1/n) sum_i [epsilon_i X_tilde_i]

    with asymptotic variance w_j' Omega_beta w_j / n.
    Reference: Ning, Peng, and Tao (2020), arXiv preprint arXiv:2009.03151.
    """
    alpha_value = _validate_alpha(alpha)

    projection_x_valid = np.asarray(estimation_payload.projection_x_valid, dtype=float)
    residual_valid = np.asarray(estimation_payload.residual_valid, dtype=float)
    beta_hat = np.asarray(estimation_payload.beta_hat, dtype=float)
    n_valid = int(projection_x_valid.shape[0])
    target_metadata = {
        "target_kind": "parametric",
        "oracle_lane": score_payload.oracle_lane,
        "basis_family": score_payload.basis_family,
        "basis_degree": int(score_payload.basis_degree),
        "n_valid_obs": n_valid,
    }
    if n_valid <= 0:
        _raise_inference_error(
            InvalidInferenceInputError,
            "parametric inference requires at least one valid observation",
            failure_kind="invalid-input",
            valid_sample_size=n_valid,
            projection_shape=_matrix_shape(projection_x_valid),
            **target_metadata,
        )
    _assert_estimation_payload_matches_score(
        score_payload,
        estimation_payload,
        target_kind="parametric",
        check_projection_x=True,
        oracle_lane=score_payload.oracle_lane,
        basis_family=score_payload.basis_family,
        basis_degree=int(score_payload.basis_degree),
        n_valid_obs=n_valid,
    )

    if residual_valid.shape[0] != n_valid:
        _raise_inference_error(
            InvalidInferenceInputError,
            "residual_valid must align with projection_x_valid",
            failure_kind="invalid-input",
            residual_length=int(residual_valid.shape[0]),
            projection_rows=n_valid,
            **target_metadata,
        )
    if beta_hat.shape[0] != projection_x_valid.shape[1]:
        _raise_inference_error(
            InvalidInferenceInputError,
            "beta_hat must align with projection_x_valid columns",
            failure_kind="invalid-input",
            beta_dimension=int(beta_hat.shape[0]),
            projection_dimension=int(projection_x_valid.shape[1]),
            **target_metadata,
        )

    if xi is None and beta_hat.shape[0] == 0:
        _raise_inference_error(
            InvalidInferenceInputError,
            "parametric inference requires a non-empty beta target",
            failure_kind="missing-parametric-target",
            beta_dimension=0,
            projection_dimension=int(projection_x_valid.shape[1]),
            **target_metadata,
        )

    xi_matrix = _coerce_xi(xi, beta_hat.shape[0])
    if xi_matrix.shape[0] == 0:
        _raise_inference_error(
            InvalidInferenceInputError,
            "parametric inference requires at least one xi target",
            failure_kind="missing-parametric-target",
            beta_dimension=int(beta_hat.shape[0]),
            xi_count=0,
            **target_metadata,
        )
    sigma_tilde_x_hat = projection_x_valid.T @ projection_x_valid / n_valid
    sigma_tilde_x_min_eigenvalue = _minimum_symmetric_eigenvalue(
        sigma_tilde_x_hat
    )
    sigma_tilde_x_rank = _matrix_rank(sigma_tilde_x_hat)
    score_moment = np.mean(residual_valid[:, None] * projection_x_valid, axis=0)
    omega_beta_hat = (
        (projection_x_valid * residual_valid[:, None]).T
        @ (projection_x_valid * residual_valid[:, None])
        / n_valid
    )
    omega_beta_min_eigenvalue = _minimum_symmetric_eigenvalue(omega_beta_hat)
    if omega_beta_min_eigenvalue < -1e-10:
        _raise_inference_error(
            NonpositiveVarianceError,
            "omega_beta_hat must be positive semidefinite for parametric inference",
            failure_kind="nonpositive-variance",
            matrix_name="omega_beta_hat",
            matrix_shape=_matrix_shape(omega_beta_hat),
            xi_count=int(xi_matrix.shape[0]),
            sigma_tilde_x_min_eigenvalue=sigma_tilde_x_min_eigenvalue,
            omega_beta_min_eigenvalue=omega_beta_min_eigenvalue,
            **target_metadata,
        )

    try:
        w_hat, eq42_metadata = solve_eq42_sparse_direction(
            sigma_tilde_x_hat=sigma_tilde_x_hat,
            xi=xi_matrix,
            lambda_prime=lambda_prime,
            max_threshold_steps=max_threshold_steps,
        )
    except SparseDirectionInfeasibleError as exc:
        _raise_inference_error(
            SparseDirectionInfeasibleError,
            str(exc),
            failure_kind="eq42-infeasible",
            xi_count=int(xi_matrix.shape[0]),
            sigma_tilde_x_dimension=int(sigma_tilde_x_hat.shape[0]),
            constraint_violation_max=float(
                exc.metadata.get(
                    "constraint_violation",
                    exc.metadata.get("constraint_violation_max", np.inf),
                )
            ),
            lambda_prime=float(
                exc.metadata.get(
                    "lambda_prime",
                    _coerce_nonnegative_finite("lambda_prime", lambda_prime),
                )
            ),
            row_index=exc.metadata.get("row_index"),
            linprog_status=exc.metadata.get("linprog_status"),
            linprog_message=exc.metadata.get("linprog_message"),
            solver_target_kind=exc.metadata.get("target_kind"),
            **target_metadata,
        )
    if not bool(eq42_metadata["feasible"]):
        _raise_inference_error(
            SparseDirectionInfeasibleError,
            "Eq. (4.2) sparse direction solver failed to satisfy the constraint",
            failure_kind="eq42-infeasible",
            xi_count=int(xi_matrix.shape[0]),
            sigma_tilde_x_dimension=int(sigma_tilde_x_hat.shape[0]),
            constraint_violation_max=float(eq42_metadata["constraint_violation_max"]),
            lambda_prime=float(eq42_metadata["lambda_prime"]),
            **target_metadata,
        )

    t_hat = xi_matrix @ beta_hat - np.einsum("ij,j->i", w_hat, score_moment)
    asymptotic_variance_hat = np.einsum("ij,jk,ik->i", w_hat, omega_beta_hat, w_hat)
    if np.any(asymptotic_variance_hat < -1e-10):
        _raise_inference_error(
            NonpositiveVarianceError,
            "estimated asymptotic variance must be non-negative",
            failure_kind="nonpositive-variance",
            matrix_name="asymptotic_variance_hat",
            matrix_shape=_matrix_shape(asymptotic_variance_hat),
            xi_count=int(xi_matrix.shape[0]),
            sigma_tilde_x_min_eigenvalue=sigma_tilde_x_min_eigenvalue,
            omega_beta_min_eigenvalue=omega_beta_min_eigenvalue,
            **target_metadata,
        )
    if np.any(asymptotic_variance_hat <= _INFERENCE_EPS):
        _raise_inference_error(
            NonpositiveVarianceError,
            "parametric inference requires strictly positive asymptotic variance",
            failure_kind="nonpositive-variance",
            matrix_name="asymptotic_variance_hat",
            matrix_shape=_matrix_shape(asymptotic_variance_hat),
            xi_count=int(xi_matrix.shape[0]),
            sigma_tilde_x_min_eigenvalue=sigma_tilde_x_min_eigenvalue,
            omega_beta_min_eigenvalue=omega_beta_min_eigenvalue,
            **_nonpositive_value_metadata(
                asymptotic_variance_hat,
                cutoff=_INFERENCE_EPS,
            ),
            **target_metadata,
        )
    standard_errors = np.sqrt(asymptotic_variance_hat / n_valid)
    z_critical = NormalDist().inv_cdf(1.0 - alpha_value / 2.0)
    confidence_interval = ConfidenceInterval(
        lower=t_hat - z_critical * standard_errors,
        upper=t_hat + z_critical * standard_errors,
        level=1.0 - alpha_value,
    )

    inference_metadata = {
        **eq42_metadata,
        "target_kind": "parametric",
        "xi_count": int(xi_matrix.shape[0]),
        "sigma_tilde_x_dimension": int(sigma_tilde_x_hat.shape[0]),
        "omega_beta_dimension": int(omega_beta_hat.shape[0]),
        "sigma_tilde_x_min_eigenvalue": sigma_tilde_x_min_eigenvalue,
        "sigma_tilde_x_rank": sigma_tilde_x_rank,
        "omega_beta_min_eigenvalue": omega_beta_min_eigenvalue,
        "beta_hat": np.asarray(beta_hat, dtype=float).copy(),
        "n_valid_obs": n_valid,
        "variance_positive": bool(np.all(asymptotic_variance_hat > 0.0)),
        "oracle_lane": score_payload.oracle_lane,
        "grid_size": 0,
        "bootstrap_seed": None,
    }

    payload = ParametricInferencePayload(
        xi=xi_matrix,
        t_hat=t_hat,
        w_hat=w_hat,
        sigma_tilde_x_hat=sigma_tilde_x_hat,
        omega_beta_hat=omega_beta_hat,
        score_moment=score_moment,
        asymptotic_variance_hat=asymptotic_variance_hat,
        standard_errors=standard_errors,
        confidence_interval=confidence_interval,
        alpha=alpha_value,
        basis_family=score_payload.basis_family,
        basis_degree=score_payload.basis_degree,
        oracle_lane=score_payload.oracle_lane,
        optimization_metadata=inference_metadata,
    )

    updated_result = _base_result(score_payload, estimation_payload, result)
    updated_result.parametric_estimates = dict(updated_result.parametric_estimates)
    updated_result.standard_errors = dict(updated_result.standard_errors)
    updated_result.intervals = dict(updated_result.intervals)

    updated_result.parametric_estimates["t_hat"] = np.asarray(
        payload.t_hat, dtype=float
    )
    if _componentwise_default(xi_matrix):
        updated_result.standard_errors["beta_se"] = np.asarray(
            payload.standard_errors, dtype=float
        )
    else:
        updated_result.standard_errors["parametric_se"] = np.asarray(
            payload.standard_errors,
            dtype=float,
        )
    updated_result.intervals["parametric_ci"] = payload.confidence_interval

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
