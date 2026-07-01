from __future__ import annotations

import warnings
from numbers import Integral

import numpy as np

from .results import ConfidenceInterval, UniformBand

_INFERENCE_EPS = float(np.finfo(float).eps)
_DEFAULT_UNIFORM_BAND_RANDOM_STATE = 0
_PSD_EIGENVALUE_TOL = 1e-10


class InferenceComputationError(RuntimeError):
    """Base error for inference computation failures."""
    def __init__(
        self,
        message: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.metadata = dict(metadata or {})
        super().__init__(message)


class InvalidInferenceInputError(InferenceComputationError):
    """Raised when inference inputs are invalid or inconsistent."""
    pass


class MissingEvaluationGridError(InferenceComputationError):
    """Raised when a non-empty evaluation grid is required but missing."""
    pass


class SingularCovarianceError(InferenceComputationError):
    """Raised when a covariance matrix is singular or rank-deficient."""
    pass


class SparseDirectionInfeasibleError(InferenceComputationError):
    """Raised when the Eq. (11) or Eq. (12) LP is infeasible."""
    pass


class NonpositiveVarianceError(InferenceComputationError):
    """Raised when the estimated variance is non-positive."""
    pass


def _matrix_shape(values: np.ndarray) -> tuple[int, ...]:
    """Return the shape tuple of ``values`` as plain Python ints."""
    return tuple(int(dimension) for dimension in np.asarray(values).shape)


def _matrix_rank(values: np.ndarray) -> int:
    """Return the numerical rank of ``values``."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0
    return int(np.linalg.matrix_rank(array))


def _checked_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    threshold: float = 1e10,
) -> np.ndarray:
    """Solve a linear system with a condition-number warning.

    This is a thin wrapper around :func:`numpy.linalg.solve` that emits a
    warning when the coefficient matrix is numerically ill-conditioned.
    It does not change the mathematical solution and intentionally keeps the
    default solver path so that well-conditioned inputs remain bit-for-bit
    identical to a direct call.
    """
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float)
    if a.size > 0:
        cond_number = float(np.linalg.cond(a))
        if cond_number > threshold:
            warnings.warn(
                f"Matrix condition number {cond_number:.2e} exceeds threshold "
                f"{threshold:.0e}; linear-system solution may be numerically "
                f"unstable.",
                UserWarning,
                stacklevel=2,
            )
    return np.linalg.solve(a, b)


def _require_finite(name: str, values: np.ndarray) -> None:
    """Raise ValueError if any element of ``values`` is non-finite."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")


def _contains_boolean_value(values: object) -> bool:
    """Check whether ``values`` contains booleans (recursively)."""
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        if values.dtype == np.bool_:
            return True
        if values.dtype == object:
            return any(_contains_boolean_value(item) for item in values.flat)
        return False
    if isinstance(values, (list, tuple)):
        return any(_contains_boolean_value(item) for item in values)
    return False


def _contains_string_value(values: object) -> bool:
    """Check whether ``values`` contains strings (recursively)."""
    if isinstance(values, (str, bytes, np.str_, np.bytes_)):
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.str_) or np.issubdtype(
            values.dtype,
            np.bytes_,
        ):
            return True
        if values.dtype == object:
            return any(_contains_string_value(item) for item in values.flat)
        return False
    if isinstance(values, (list, tuple)):
        return any(_contains_string_value(item) for item in values)
    return False


def _require_numeric_not_boolean(name: str, values: object) -> None:
    """Raise ValueError if ``values`` contains booleans or strings."""
    if _contains_boolean_value(values):
        raise ValueError(f"{name} must be numeric, not boolean")
    if _contains_string_value(values):
        raise ValueError(f"{name} must be numeric, not string")


def _coerce_raw_numeric_array(name: str, values: object) -> np.ndarray:
    """Coerce ``values`` to a float ndarray, rejecting booleans and strings."""
    _require_numeric_not_boolean(name, values)
    raw_array = np.asarray(values)
    _require_numeric_not_boolean(name, raw_array)
    return raw_array.astype(float)


def _coerce_nonnegative_finite(name: str, value: float) -> float:
    """Validate and return a non-negative finite scalar."""
    if _contains_boolean_value(value):
        raise ValueError(f"{name} must be numeric")
    if _contains_string_value(value):
        raise ValueError(f"{name} must be numeric, not string")
    raw_value = np.asarray(value)
    if raw_value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    numeric = float(raw_value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _coerce_finite_numeric(name: str, value: float) -> float:
    """Validate and return a finite scalar."""
    if _contains_boolean_value(value):
        raise ValueError(f"{name} must be numeric, not boolean")
    if _contains_string_value(value):
        raise ValueError(f"{name} must be numeric, not string")
    raw_value = np.asarray(value)
    if raw_value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    numeric = float(raw_value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _coerce_positive_integer(name: str, value: int) -> int:
    """Validate and return a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _coerce_optional_nonnegative_integer(name: str, value: int | None) -> int | None:
    """Validate and return a non-negative integer or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _coerce_uniform_band_random_state(value: int | None) -> int:
    """Resolve random_state for uniform band, defaulting to 0 if None."""
    seed = _coerce_optional_nonnegative_integer("random_state", value)
    if seed is None:
        return _DEFAULT_UNIFORM_BAND_RANDOM_STATE
    return seed


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


def _require_interval_finite(
    name: str,
    interval: ConfidenceInterval,
) -> None:
    """Validate that a confidence interval contains finite bounds."""
    _require_finite(f"{name}.lower", np.asarray(interval.lower, dtype=float))
    _require_finite(f"{name}.upper", np.asarray(interval.upper, dtype=float))
    if interval.level is not None and not np.isfinite(float(interval.level)):
        raise ValueError(f"{name}.level must be finite")
    if isinstance(interval, UniformBand) and interval.critical_value is not None:
        if not np.isfinite(float(interval.critical_value)):
            raise ValueError(f"{name}.critical_value must be finite")


def _validate_alpha(alpha: float) -> float:
    """Validate and return a significance level in (0, 1)."""
    if _contains_boolean_value(alpha):
        raise ValueError("alpha must be numeric")
    if _contains_string_value(alpha):
        raise ValueError("alpha must be numeric, not string")
    raw_alpha = np.asarray(alpha)
    if raw_alpha.ndim != 0:
        raise ValueError("alpha must be a scalar")
    alpha_value = float(raw_alpha)
    if not np.isfinite(alpha_value):
        raise ValueError("alpha must be finite")
    if not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return alpha_value


def _require_interval_level(name: str, interval: ConfidenceInterval, alpha: float) -> None:
    """Validate that ``interval.level`` equals ``1 - alpha``."""
    expected_level = 1.0 - alpha
    if interval.level is None:
        raise ValueError(f"{name}.level must be provided")
    if not np.isclose(float(interval.level), expected_level, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name}.level must equal 1 - alpha")


def _require_interval_matches_center_scale(
    name: str,
    interval: ConfidenceInterval,
    *,
    center: np.ndarray,
    scale: np.ndarray,
    multiplier: float,
) -> None:
    """Validate that interval bounds equal center +/- multiplier * scale."""
    lower = np.asarray(interval.lower, dtype=float)
    upper = np.asarray(interval.upper, dtype=float)
    if lower.shape != center.shape or upper.shape != center.shape:
        raise ValueError(f"{name} bounds must align with the inference target")
    expected_lower = center - multiplier * scale
    expected_upper = center + multiplier * scale
    if not np.allclose(lower, expected_lower, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name}.lower must equal center - critical_value * scale")
    if not np.allclose(upper, expected_upper, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name}.upper must equal center + critical_value * scale")


def _raise_inference_error(
    error_type: type[InferenceComputationError],
    message: str,
    **metadata: object,
) -> None:
    """Raise an ``InferenceComputationError`` subclass with metadata."""
    raise error_type(message, metadata=metadata)


def _nonpositive_value_metadata(
    values: np.ndarray,
    *,
    cutoff: float,
) -> dict[str, object]:
    """Return diagnostic metadata for nonpositive entries in ``values``."""
    array = np.asarray(values, dtype=float)
    flat = array.reshape(-1)
    violating = np.flatnonzero(flat <= float(cutoff))
    return {
        "minimum_value": float(np.min(flat)),
        "maximum_value": float(np.max(flat)),
        "nonpositive_entry_count": int(violating.size),
        "nonpositive_entry_indices": tuple(int(index) for index in violating.tolist()),
    }


def _assert_full_rank(
    matrix: np.ndarray,
    *,
    matrix_name: str,
    target_kind: str,
    **metadata: object,
) -> None:
    """Raise SingularCovarianceError if ``matrix`` is rank-deficient."""
    shape = _matrix_shape(matrix)
    rank = _matrix_rank(matrix)
    if len(shape) == 2 and shape[0] == shape[1] and rank < shape[0]:
        _raise_inference_error(
            SingularCovarianceError,
            f"{matrix_name} must be full rank for {target_kind} inference",
            target_kind=target_kind,
            failure_kind="singular-covariance",
            matrix_name=matrix_name,
            matrix_shape=shape,
            matrix_rank=rank,
            **metadata,
        )


def _coerce_vector(name: str, values: np.ndarray) -> np.ndarray:
    """Coerce input to a 1-D finite float array."""
    array = _coerce_raw_numeric_array(name, values)
    if array.ndim == 0:
        vector = array.reshape(1)
        _require_finite(name, vector)
        return vector
    if array.ndim == 1:
        _require_finite(name, array)
        return array
    if array.ndim == 2 and array.shape[1] == 1:
        vector = array[:, 0]
        _require_finite(name, vector)
        return vector
    raise ValueError(f"{name} must be one-dimensional")


def _coerce_matrix(name: str, values: np.ndarray) -> np.ndarray:
    """Coerce input to a 2-D finite float array."""
    array = _coerce_raw_numeric_array(name, values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    _require_finite(name, array)
    return array


def _coerce_xi(xi: np.ndarray | None, beta_dimension: int) -> np.ndarray:
    """Coerce ``xi`` to a 2-D target matrix for inference.

    When ``xi`` is None, defaults to the identity matrix (componentwise
    inference on each element of beta).
    """
    if xi is None:
        return np.eye(beta_dimension, dtype=float)

    array = _coerce_raw_numeric_array("xi", xi)
    if array.ndim == 1:
        matrix = array.reshape(1, -1)
    elif array.ndim == 2:
        matrix = array
    else:
        raise ValueError("xi must be a vector or a matrix")

    if matrix.shape[1] != beta_dimension:
        raise ValueError("xi must align with beta_hat dimension")
    _require_finite("xi", matrix)
    return matrix
