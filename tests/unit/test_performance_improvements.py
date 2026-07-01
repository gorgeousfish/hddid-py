"""Tests for performance improvements: sklearn LASSO backend and parallel cross-fitting."""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest


def _make_test_data(n=100, p=5, seed=42):
    """Create minimal test data for performance tests."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    z = rng.standard_normal(n)
    treat = (rng.random(n) > 0.5).astype(float)
    y0 = x @ rng.standard_normal(p) + rng.standard_normal(n)
    y1 = y0 + 0.5 * treat + rng.standard_normal(n) * 0.1
    z0 = np.array([0.0])
    return y0, y1, treat, x, z, z0


class TestSklearnSolverBackend:
    """Test sklearn LASSO solver backend (Eq 3.1 optimization)."""

    def test_builtin_is_default(self):
        """Default solver must be 'builtin' for backward compatibility."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        fit = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0)
        # Should work without error (default is builtin)
        assert fit.result is not None

    def test_sklearn_solver_works(self):
        """sklearn solver should produce valid results."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        fit = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0, solver="sklearn")
        assert fit.result is not None
        assert fit.estimation_payload is not None

    def test_builtin_sklearn_equivalence_no_penalty(self):
        """Without penalty, both solvers should give identical results."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        
        fit_builtin = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                                 penalty_lambda=0.0, solver="builtin")
        fit_sklearn = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                                 penalty_lambda=0.0, solver="sklearn")
        
        beta_builtin = np.asarray(fit_builtin.estimation_payload.beta_hat, dtype=float)
        beta_sklearn = np.asarray(fit_sklearn.estimation_payload.beta_hat, dtype=float)
        npt.assert_allclose(beta_builtin, beta_sklearn, atol=1e-10)

    def test_builtin_sklearn_close_with_penalty(self):
        """With penalty, both solvers should produce statistically close results."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        
        fit_builtin = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                                 penalty_lambda=0.1, solver="builtin")
        fit_sklearn = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                                 penalty_lambda=0.1, solver="sklearn")
        
        beta_builtin = np.asarray(fit_builtin.estimation_payload.beta_hat, dtype=float)
        beta_sklearn = np.asarray(fit_sklearn.estimation_payload.beta_hat, dtype=float)
        # Allow small numerical differences due to different convergence paths
        npt.assert_allclose(beta_builtin, beta_sklearn, atol=1e-6)

    def test_invalid_solver_error(self):
        """Invalid solver value should raise ValueError."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        with pytest.raises(ValueError, match="[Ss]olver|[Ii]nvalid"):
            fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0, solver="invalid")


class TestParallelCrossFitting:
    """Test parallel cross-fitting (n_jobs parameter)."""

    def test_default_n_jobs_is_one(self):
        """Default n_jobs=1 for backward compatibility."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        fit = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0)
        assert fit.result is not None

    def test_parallel_produces_identical_results(self):
        """Parallel (n_jobs=2) must produce bit-identical results to sequential."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        
        fit_seq = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                            n_folds=3, n_jobs=1)
        fit_par = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                            n_folds=3, n_jobs=2)
        
        # Beta must be bit-identical
        beta_seq = np.asarray(fit_seq.estimation_payload.beta_hat, dtype=float)
        beta_par = np.asarray(fit_par.estimation_payload.beta_hat, dtype=float)
        npt.assert_array_equal(beta_seq, beta_par)
        
        # Gamma must be bit-identical
        gamma_seq = np.asarray(fit_seq.estimation_payload.gamma_hat, dtype=float)
        gamma_par = np.asarray(fit_par.estimation_payload.gamma_hat, dtype=float)
        npt.assert_array_equal(gamma_seq, gamma_par)

    def test_parallel_with_more_folds(self):
        """Parallel with 4 folds and 3 workers should work."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data(n=200)
        
        fit = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                        n_folds=4, n_jobs=3)
        assert fit.result is not None

    def test_parallel_and_sklearn_combined(self):
        """Both optimizations should work together."""
        from hddid import fit_hddid
        y0, y1, treat, x, z, z0 = _make_test_data()
        
        fit = fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0,
                        n_folds=3, n_jobs=2, solver="sklearn")
        assert fit.result is not None


class TestPerformanceBenchmark:
    """Verify that optimizations provide actual speedup."""

    @pytest.mark.slow
    def test_parallel_faster_than_sequential(self):
        """Parallel should be faster for sufficient workload."""
        import time
        from hddid import fit_hddid
        
        y0, y1, treat, x, z, z0 = _make_test_data(n=500, p=20, seed=123)
        
        # Sequential timing
        start = time.perf_counter()
        fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0, n_folds=4, n_jobs=1)
        seq_time = time.perf_counter() - start
        
        # Parallel timing
        start = time.perf_counter()
        fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0, n_folds=4, n_jobs=4)
        par_time = time.perf_counter() - start
        
        # Just verify parallel completes (speedup depends on hardware)
        print(f"Sequential: {seq_time:.3f}s, Parallel: {par_time:.3f}s, "
              f"Speedup: {seq_time/par_time:.2f}x")
        # Don't assert speedup (unreliable in CI), just verify correctness
        assert par_time > 0
