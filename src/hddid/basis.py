from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import numpy as np


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
            return any(_contains_boolean_or_string_alias(item) for item in value.flat)
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean_or_string_alias(item) for item in value)
    return False


def _coerce_univariate_array(z: np.ndarray) -> np.ndarray:
    """Coerce input to a 1-D finite float array."""
    if _contains_boolean_or_string_alias(z):
        raise ValueError("z must be numeric, not boolean or string")
    raw_values = np.asarray(z)
    if _contains_boolean_or_string_alias(raw_values):
        raise ValueError("z must be numeric, not boolean or string")
    values = raw_values.astype(float)
    if values.ndim == 0:
        column = values.reshape(1)
    elif values.ndim == 1:
        column = values
    elif values.ndim == 2 and values.shape[1] == 1:
        column = values[:, 0]
    else:
        raise ValueError("z must be one-dimensional")
    if not np.all(np.isfinite(column)):
        raise ValueError("z must contain only finite values")
    if column.shape[0] == 0:
        raise ValueError("z must contain at least one coordinate")
    return column


def _require_integer_degree(
    degree: int, *, minimum: int, message: str
) -> int:
    """Validate that ``degree`` is an integer >= ``minimum``."""
    if isinstance(degree, bool) or not isinstance(degree, Integral):
        raise ValueError("degree must be an integer")
    degree_value = int(degree)
    if degree_value < minimum:
        raise ValueError(message)
    return degree_value


def polynomial_sieve_basis(z: np.ndarray, degree: int) -> np.ndarray:
    """Construct a one-dimensional polynomial sieve basis with an intercept.

    Builds the matrix [1, z, z^2, ..., z^degree] for use as the
    nonparametric sieve in the partially-linear model Eq. (3.1).

    Parameters
    ----------
    z : ndarray of float, shape (n,)
        Scalar nonparametric variable values. Must be finite.
    degree : int
        Maximum polynomial power (>= 0). The resulting basis has
        degree + 1 columns including the intercept.

    Returns
    -------
    ndarray of float, shape (n, degree + 1)
        Polynomial basis matrix with column j equal to z^j.

    Raises
    ------
    ValueError
        If z is not one-dimensional, contains non-finite values,
        or degree is not a non-negative integer.
    """
    degree_value = _require_integer_degree(
        degree,
        minimum=0,
        message="degree must be non-negative",
    )

    column = _coerce_univariate_array(z)

    with np.errstate(over="ignore", invalid="ignore"):
        basis = np.column_stack([column**power for power in range(degree_value + 1)])
    if not np.all(np.isfinite(basis)):
        raise ValueError("polynomial basis must contain only finite values")
    return basis


def trigonometric_sieve_basis(z: np.ndarray, degree: int) -> np.ndarray:
    """Construct a one-dimensional trigonometric sieve basis with an intercept.

    Builds the matrix [1, cos(2*pi*z), sin(2*pi*z), ...,
    cos(2*degree*pi*z), sin(2*degree*pi*z)] for use as the
    nonparametric sieve in Eq. (3.1).

    Parameters
    ----------
    z : ndarray of float, shape (n,)
        Scalar nonparametric variable values. Must be finite.
    degree : int
        Number of harmonic pairs (>= 1). The resulting basis has
        2*degree + 1 columns including the intercept.

    Returns
    -------
    ndarray of float, shape (n, 2*degree + 1)
        Trigonometric basis matrix.

    Raises
    ------
    ValueError
        If z is not one-dimensional, contains non-finite values,
        or degree is not a positive integer.
    """
    degree_value = _require_integer_degree(
        degree,
        minimum=1,
        message="degree must be at least 1",
    )

    column = _coerce_univariate_array(z)
    basis = [np.ones_like(column)]
    with np.errstate(over="ignore", invalid="ignore"):
        for harmonic in range(1, degree_value + 1):
            angle = 2.0 * harmonic * np.pi * column
            basis.extend([np.cos(angle), np.sin(angle)])
    matrix = np.column_stack(basis)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("trigonometric basis must contain only finite values")
    return matrix


def bspline_sieve_basis(z: np.ndarray, degree: int) -> np.ndarray:
    """Construct a B-spline sieve basis matrix.

    B-splines have the optimal Bernstein constant O(1), superior to
    polynomial O(k^{d_z/2}) and trigonometric O(k^{d_z}) bases
    Ning, Peng, and Tao (2020).

    Parameters
    ----------
    z : ndarray, shape (n,)
        Scalar variable values.
    degree : int
        Number of interior knots (>= 1). Cubic B-splines (order 4) are used.
        Total basis dimension = degree + 4.

    Returns
    -------
    basis_matrix : ndarray, shape (n, degree + 4)
        B-spline basis functions evaluated at z.
        Includes a leading constant column for compatibility.
        One B-spline column is dropped to break the partition-of-unity
        constraint and ensure full column rank.
    """
    from scipy.interpolate import BSpline as _BSpline

    z_arr = _coerce_univariate_array(z)
    degree_value = _require_integer_degree(
        degree,
        minimum=1,
        message="degree must be at least 1 for bspline basis",
    )

    n = len(z_arr)
    spline_order = 4  # cubic
    n_interior = degree_value

    # Quantile-based interior knots
    quantiles = np.linspace(0, 1, n_interior + 2)[1:-1]
    interior_knots = np.quantile(z_arr, quantiles)

    # Full knot vector with repeated boundary knots
    z_min = float(z_arr.min())
    z_max = float(z_arr.max())
    # Handle degenerate case: all values identical (e.g. single evaluation point)
    if z_max - z_min < 1e-14:
        z_min = z_min - 1.0
        z_max = z_max + 1.0
        interior_knots = np.linspace(z_min, z_max, n_interior + 2)[1:-1]
    knots = np.concatenate([
        np.full(spline_order, z_min),
        interior_knots,
        np.full(spline_order, z_max),
    ])

    n_basis = len(knots) - spline_order  # = n_interior + spline_order
    basis = np.empty((n, n_basis), dtype=float)
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        # Use natural polynomial extrapolation outside the knot range rather
        # than zero-filling.  In-range evaluations remain bit-for-bit identical
        # to extrapolate=False because the flag only affects out-of-range points.
        spl = _BSpline(knots, coeffs, spline_order - 1, extrapolate=True)
        basis[:, i] = spl(z_arr)

    # B-splines satisfy partition of unity: columns sum to 1.
    # Drop the first B-spline to break this linear dependency,
    # then prepend a constant column for compatibility.
    # Result: full column rank matrix with degree + 4 columns.
    return np.column_stack([np.ones(n), basis[:, 1:]])


def build_sieve_basis(z: np.ndarray, basis_family: str, degree: int) -> np.ndarray:
    """Dispatch to the requested sieve basis family.

    Parameters
    ----------
    z : ndarray of float, shape (n,)
        Scalar variable values.
    basis_family : str
        One of ``'polynomial'``, ``'trigonometric'``, or ``'bspline'``.
    degree : int
        Basis degree parameter (meaning depends on family).

    Returns
    -------
    ndarray of float
        Sieve basis matrix with an intercept column.

    Raises
    ------
    ValueError
        If ``basis_family`` is not recognized.
    """
    family = str(basis_family).strip().lower()

    if family == "polynomial":
        return polynomial_sieve_basis(z, degree=degree)
    if family == "trigonometric":
        return trigonometric_sieve_basis(z, degree=degree)
    if family == "bspline":
        return bspline_sieve_basis(z, degree=degree)
    raise ValueError(
        "basis_family must be one of {'polynomial', 'trigonometric', 'bspline'}"
    )


def suggest_basis_degree(
    n: int,
    p: int,
    *,
    basis_family: str = "polynomial",
    method: str = "conservative",
) -> int:
    """Suggest a basis function degree based on sample size and dimension.

    Implements the truncation parameter selection heuristic from
    Ning, Peng, and Tao (2020), Assumption 6 and surrounding discussion.

    The conservative rule uses k_n = floor(c * log(n)) with c chosen to
    satisfy the theoretical constraint k_n = o(sqrt(n) / sqrt(log(p))).

    Parameters
    ----------
    n : int
        Sample size (number of observations).
    p : int
        Number of high-dimensional covariates.
    basis_family : str, default "polynomial"
        Basis function family ("polynomial" or "trigonometric").
        Affects the Bernstein constant growth rate.
    method : str, default "conservative"
        Selection method. Currently supported:

        - "conservative": Uses k_n = floor(c * log(n)) with c calibrated
          to the theoretical upper bound.
        - "upper_bound": Returns the theoretical maximum k_n satisfying
          Assumption 6 constraints.

    Returns
    -------
    int
        Recommended basis degree (>= 1).

    Raises
    ------
    ValueError
        If n <= 0, p <= 0, basis_family is invalid, or method is invalid.

    Notes
    -----
    The theoretical constraint from the paper is:

        k_n = o(sqrt(n) / sqrt(log(p)))

    For the conservative rule, we use c = 1.5 (middle of the paper's
    suggested range [1, 3]) and then clip to the theoretical upper bound.

    For polynomial basis, the effective constraint is tighter due to the
    Bernstein constant growth O(k^{d_z/2}) with d_z = 1.

    Examples
    --------
    >>> suggest_basis_degree(500, 100)
    8
    >>> suggest_basis_degree(1000, 200, basis_family="trigonometric")
    4
    """
    # --- Input validation ---
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not isinstance(p, int) or isinstance(p, bool) or p <= 0:
        raise ValueError(f"p must be a positive integer, got {p!r}")
    if basis_family not in ("polynomial", "trigonometric"):
        raise ValueError(
            f"Invalid basis_family {basis_family!r}. "
            f"Valid options: 'polynomial', 'trigonometric'"
        )
    if method not in ("conservative", "upper_bound"):
        raise ValueError(
            f"Invalid method {method!r}. "
            f"Valid options: 'conservative', 'upper_bound'"
        )

    # --- Theoretical upper bound ---
    # k_n < sqrt(n) / sqrt(log(p))  [general constraint]
    log_p = math.log(max(p, 2))  # guard against log(1) = 0
    theoretical_upper = math.sqrt(n) / math.sqrt(log_p)

    # --- Basis-family-specific effective bound ---
    # From Assumption 6: k_n * xi_0^2(k_n) * log(p) / n = O(1)
    #   polynomial (d_z=1): xi_0 = O(k^{1/2}) → k * k * log(p)/n < C
    #                       → k < sqrt(n / log(p))
    #   trigonometric (d_z=1): xi_0 = O(k) → k * k^2 * log(p)/n < C
    #                          → k < (n / log(p))^{1/3}
    if basis_family == "polynomial":
        effective_upper = math.sqrt(n / log_p)
    else:  # trigonometric
        effective_upper = (n / log_p) ** (1.0 / 3.0)

    # Take the tighter of the two bounds
    upper = min(theoretical_upper, effective_upper)

    # --- Compute degree ---
    if method == "upper_bound":
        # Apply safety factor 0.8 to stay within the o(·) regime
        result = max(1, int(math.floor(upper * 0.8)))
    else:  # conservative
        # Conservative rule: k_n = floor(c * log(n)), c = 1.5
        c = 1.5
        conservative = math.floor(c * math.log(n))
        # Clip to the safe upper bound
        result = max(1, min(conservative, int(math.floor(upper * 0.8))))

    return result
