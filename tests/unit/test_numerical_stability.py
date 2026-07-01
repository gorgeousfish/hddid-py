from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.interpolate import BSpline as _BSpline

from hddid._inference_common import _checked_solve
from hddid.basis import bspline_sieve_basis
from hddid.estimation import _project_onto_basis
from hddid.inference_nonparametric import (
    _sandwich_solve_sigma_f,
    _solve_sigma_f_left,
)


def _hilbert_matrix(n: int) -> np.ndarray:
    """Return a classic ill-conditioned symmetric matrix."""
    indices = np.arange(1, n + 1)
    return 1.0 / (indices[:, None] + indices[None, :] - 1.0)


def test_checked_solve_matches_numpy_on_well_conditioned_system() -> None:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((5, 5))
    a = a.T @ a + np.eye(5)  # well-conditioned SPD
    b = rng.standard_normal(5)

    expected = np.linalg.solve(a, b)
    result = _checked_solve(a, b)

    np.testing.assert_array_equal(result, expected)


def test_checked_solve_warns_on_ill_conditioned_matrix() -> None:
    a = _hilbert_matrix(12)
    b = np.ones(a.shape[0])

    with pytest.warns(UserWarning, match="condition number"):
        _checked_solve(a, b)


def test_project_onto_basis_unchanged_for_well_conditioned_design() -> None:
    rng = np.random.default_rng(0)
    n = 50
    basis = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    values = rng.standard_normal(n)

    coefficients, fitted = _project_onto_basis(basis, values)

    expected_coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
    expected_fitted = basis @ expected_coefficients

    np.testing.assert_array_equal(coefficients, expected_coefficients)
    np.testing.assert_array_equal(fitted, expected_fitted)


def test_project_onto_basis_warns_on_ill_conditioned_design() -> None:
    # A 12x12 Hilbert matrix has a condition number ~1e15.
    basis = _hilbert_matrix(12)
    values = np.ones(basis.shape[0])

    with pytest.warns(UserWarning, match="condition number"):
        _project_onto_basis(basis, values)


def test_project_onto_basis_warns_on_rank_deficient_design() -> None:
    n = 10
    basis = np.column_stack([np.ones(n), np.ones(n), np.linspace(0, 1, n)])
    values = np.ones(n)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _project_onto_basis(basis, values)

    messages = [str(w.message) for w in recorded]
    assert any("rank deficient" in msg for msg in messages)


def test_bspline_basis_unchanged_within_knot_range() -> None:
    rng = np.random.default_rng(1)
    z = np.sort(rng.uniform(0, 1, 50))
    degree = 3

    basis = bspline_sieve_basis(z, degree=degree)

    # Manual reference using extrapolate=False for in-range points.
    z_min = float(z.min())
    z_max = float(z.max())
    n_interior = degree
    spline_order = 4
    quantiles = np.linspace(0, 1, n_interior + 2)[1:-1]
    interior_knots = np.quantile(z, quantiles)
    knots = np.concatenate([
        np.full(spline_order, z_min),
        interior_knots,
        np.full(spline_order, z_max),
    ])
    n_basis = len(knots) - spline_order
    reference = np.empty((len(z), n_basis), dtype=float)
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spl = _BSpline(knots, coeffs, spline_order - 1, extrapolate=False)
        reference[:, i] = np.nan_to_num(spl(z), nan=0.0)
    expected = np.column_stack([np.ones(len(z)), reference[:, 1:]])

    np.testing.assert_array_equal(basis, expected)


def test_bspline_basis_extrapolates_instead_of_zeroing() -> None:
    rng = np.random.default_rng(2)
    z_in = np.sort(rng.uniform(0, 1, 30))
    degree = 3

    # Fit knots to the in-range sample, then evaluate well outside.
    basis_inside = bspline_sieve_basis(z_in, degree=degree)

    z_out = np.array([-0.5, -0.25, 1.25, 1.5])
    basis_out = bspline_sieve_basis(z_out, degree=degree)

    # The old zero-fill behaviour would give all zeros outside the range.
    # Natural extrapolation should produce non-zero values for most columns.
    assert not np.all(basis_out == 0.0)
    assert basis_out.shape[1] == basis_inside.shape[1]
    # The leading constant column is preserved for compatibility.
    np.testing.assert_array_equal(basis_out[:, 0], np.ones(len(z_out)))


def test_solve_sigma_f_left_matches_numpy_and_warns_when_ill_conditioned() -> None:
    sigma_f = np.array([[1.0, 0.99], [0.99, 1.0]])
    rhs = np.array([[1.0], [2.0]])

    expected = np.linalg.solve(sigma_f, rhs)
    result = _solve_sigma_f_left(sigma_f, rhs)

    np.testing.assert_array_equal(result, expected)
    assert not np.any(np.isnan(result))

    ill = _hilbert_matrix(10)
    with pytest.warns(UserWarning, match="condition number"):
        _solve_sigma_f_left(ill, np.ones(ill.shape[0]))


def test_sandwich_solve_sigma_f_matches_reference() -> None:
    rng = np.random.default_rng(3)
    dim = 4
    sigma_f = rng.standard_normal((dim, dim))
    sigma_f = sigma_f.T @ sigma_f + np.eye(dim)
    omega_f = rng.standard_normal((dim, dim))

    result = _sandwich_solve_sigma_f(sigma_f, omega_f)
    left = np.linalg.solve(sigma_f, omega_f)
    expected = np.linalg.solve(sigma_f, left.T).T

    np.testing.assert_array_equal(result, expected)
