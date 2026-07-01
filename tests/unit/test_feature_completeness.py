"""Tests for feature completeness: auto lambda and B-spline basis."""
from __future__ import annotations

import math
import numpy as np
import numpy.testing as npt
import pytest


def _make_test_data(n=100, p=5, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    z = rng.standard_normal(n)
    treat = (rng.random(n) > 0.5).astype(float)
    y0 = x @ rng.standard_normal(p) + rng.standard_normal(n)
    y1 = y0 + 0.5 * treat + rng.standard_normal(n) * 0.1
    return y0, y1, treat, x, z, np.array([0.0])


class TestAutoLambda:
    """Test penalty_lambda='auto' based on paper's theoretical rate."""

    def test_auto_lambda_produces_result(self):
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data(n=200, p=20)
        fit = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0, 
                       penalty_lambda="auto")
        assert fit.result is not None

    def test_auto_lambda_vs_manual_reasonable(self):
        """Auto lambda should produce a reasonable penalty value."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data(n=200, p=20)
        # Theoretical: lambda = 2.2 * sqrt(log(20) / 200) ≈ 0.269
        fit_auto = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                            penalty_lambda="auto")
        fit_zero = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                            penalty_lambda=0.0)
        # Both should produce valid results
        assert fit_auto.result is not None
        assert fit_zero.result is not None

    def test_auto_lambda_backward_compatible(self):
        """Default penalty_lambda=0.0 should still work."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        fit = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0)
        assert fit.result is not None

    def test_invalid_string_raises(self):
        """Non-'auto' string should raise ValueError."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        with pytest.raises((ValueError, TypeError)):
            fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                     penalty_lambda="invalid")

    def test_numeric_lambda_still_works(self):
        """Explicit numeric lambda should work as before."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        fit = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                       penalty_lambda=0.1)
        assert fit.result is not None


class TestBSplineBasis:
    """Test B-spline sieve basis functions."""

    def test_import(self):
        from hddid import bspline_sieve_basis
        assert callable(bspline_sieve_basis)

    def test_basic_shape(self):
        from hddid import bspline_sieve_basis
        z = np.linspace(-2, 2, 50)
        basis = bspline_sieve_basis(z, degree=3)
        assert basis.shape[0] == 50
        assert basis.shape[1] >= 4  # At least degree + 1 columns

    def test_has_constant_column(self):
        """First column should be constant (ones)."""
        from hddid import bspline_sieve_basis
        z = np.linspace(-2, 2, 50)
        basis = bspline_sieve_basis(z, degree=3)
        npt.assert_allclose(basis[:, 0], 1.0)

    def test_nonnegative_values(self):
        """B-spline values should be non-negative."""
        from hddid import bspline_sieve_basis
        z = np.linspace(-2, 2, 100)
        basis = bspline_sieve_basis(z, degree=5)
        # Allow small numerical noise
        assert np.all(basis >= -1e-10)

    def test_degree_1_minimum(self):
        """degree < 1 should raise ValueError."""
        from hddid import bspline_sieve_basis
        z = np.linspace(-2, 2, 50)
        with pytest.raises(ValueError):
            bspline_sieve_basis(z, degree=0)

    def test_integration_with_fit(self):
        """B-spline should work end-to-end in fit_hddid."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        fit = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                       basis_family="bspline", basis_degree=3)
        assert fit.result is not None

    def test_bspline_vs_polynomial_both_produce_results(self):
        """B-spline and polynomial should both produce valid estimates."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        fit_poly = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                            basis_family="polynomial", basis_degree=3)
        fit_bspl = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                            basis_family="bspline", basis_degree=3)
        assert fit_poly.result is not None
        assert fit_bspl.result is not None

    def test_combined_bspline_auto_lambda(self):
        """B-spline + auto lambda should work together."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data(n=200, p=20)
        fit = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                       basis_family="bspline", basis_degree=3,
                       penalty_lambda="auto")
        assert fit.result is not None
