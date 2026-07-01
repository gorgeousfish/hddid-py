from __future__ import annotations

import importlib
from unittest.mock import patch

import numpy as np
import pytest


def _load_score_module():
    try:
        return importlib.import_module("hddid.score")
    except ModuleNotFoundError as exc:
        pytest.fail(f"hddid.score is missing: {exc}")


def _load_estimation_module():
    try:
        return importlib.import_module("hddid.estimation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"hddid.estimation is missing: {exc}")


def _make_synthetic_score_payload(
    *,
    n_obs: int = 100,
    n_features: int = 10,
    basis_degree: int = 3,
    rng: np.random.Generator,
) -> object:
    """Build a ScorePayload for a simple penalized Eq. (3.1) problem."""
    score_module = _load_score_module()

    x = rng.standard_normal((n_obs, n_features))
    z = rng.uniform(0.0, 1.0, n_obs)
    basis = np.column_stack([z**degree for degree in range(basis_degree + 1)])

    # Sparse true beta so Lasso shrinks some coefficients exactly to zero.
    beta_true = np.zeros(n_features, dtype=float)
    beta_true[:3] = np.array([1.5, -0.8, 0.6], dtype=float)
    gamma_true = np.array([0.3, -0.2, 0.1, -0.05], dtype=float)
    s_hat_valid = x @ beta_true + basis @ gamma_true + 0.1 * rng.standard_normal(n_obs)

    return score_module.ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=np.ones(n_obs, dtype=bool),
        fold_ids=np.ones(n_obs, dtype=int),
        fold_diagnostics=[],
        basis_family="polynomial",
        basis_degree=basis_degree,
        oracle_lane="r-parity-polynomial",
        basis_matrix=basis,
        x_valid=x,
        basis_valid_full=basis,
        basis_design_valid=basis[:, 1:],
        evaluation_basis=np.column_stack(
            [np.linspace(0.0, 1.0, 5) ** degree for degree in range(basis_degree + 1)]
        ),
        pi_hat=np.full(n_obs, 0.5, dtype=float),
        phi0_hat=np.zeros(n_obs, dtype=float),
        phi1_hat=np.zeros(n_obs, dtype=float),
        rho_hat=np.ones(n_obs, dtype=float),
        intercept_dropped_for_design=True,
    )


def test_estimate_eq31_mainline_default_solver_is_sklearn(
    eq31_manual_slice: dict[str, object],
) -> None:
    """After the default change, estimate_eq31_mainline uses sklearn."""
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()

    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    estimation_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.2,
    )

    solver_name = estimation_payload.optimization_metadata["solver"]
    assert solver_name in ("sklearn-lasso", "sklearn-residualized-ols")


def test_native_and_sklearn_solvers_are_numerically_equivalent() -> None:
    """Both backends must converge to the same KKT solution."""
    estimation_module = _load_estimation_module()
    rng = np.random.default_rng(42)
    score_payload = _make_synthetic_score_payload(n_obs=100, n_features=10, rng=rng)

    native_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.3,
        max_iter=10_000,
        tol=1e-10,
        solver="builtin",
    )
    sklearn_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.3,
        max_iter=10_000,
        tol=1e-10,
        solver="sklearn",
    )

    max_beta_diff = float(np.max(np.abs(native_payload.beta_hat - sklearn_payload.beta_hat)))
    assert max_beta_diff < 1e-6, (
        f"max|beta_native - beta_sklearn| = {max_beta_diff:.3e} >= 1e-6"
    )

    np.testing.assert_allclose(
        native_payload.gamma_hat,
        sklearn_payload.gamma_hat,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        native_payload.second_stage_prediction_valid,
        sklearn_payload.second_stage_prediction_valid,
        atol=1e-5,
    )


def test_native_alias_matches_builtin() -> None:
    """'native' must behave identically to 'builtin'."""
    estimation_module = _load_estimation_module()
    rng = np.random.default_rng(123)
    score_payload = _make_synthetic_score_payload(n_obs=80, n_features=6, rng=rng)

    builtin_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.25,
        solver="builtin",
    )
    native_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.25,
        solver="native",
    )

    np.testing.assert_allclose(builtin_payload.beta_hat, native_payload.beta_hat, atol=1e-12)
    np.testing.assert_allclose(builtin_payload.gamma_hat, native_payload.gamma_hat, atol=1e-12)


def test_zero_penalty_gives_identical_ols_path() -> None:
    """When lambda=0 both solders reduce to residualized OLS."""
    estimation_module = _load_estimation_module()
    rng = np.random.default_rng(7)
    score_payload = _make_synthetic_score_payload(n_obs=60, n_features=5, rng=rng)

    native_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
        solver="builtin",
    )
    sklearn_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
        solver="sklearn",
    )

    np.testing.assert_allclose(
        native_payload.beta_hat,
        sklearn_payload.beta_hat,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        native_payload.gamma_hat,
        sklearn_payload.gamma_hat,
        atol=1e-10,
    )


def test_sklearn_uninstalled_falls_back_to_native_with_warning() -> None:
    """If scikit-learn is unavailable, solver='sklearn' warns and falls back."""
    estimation_module = _load_estimation_module()
    rng = np.random.default_rng(99)
    score_payload = _make_synthetic_score_payload(n_obs=70, n_features=4, rng=rng)

    with patch.dict("sys.modules", {"sklearn.linear_model": None}):
        with pytest.warns(UserWarning, match="scikit-learn is not installed"):
            fallback_payload, _ = estimation_module.estimate_eq31_mainline(
                score_payload,
                penalty_lambda=0.2,
                solver="sklearn",
            )

    # After fallback, the solver metadata should be native.
    assert fallback_payload.optimization_metadata["solver"] in (
        "coordinate-descent-lasso",
        "residualized-ols",
        "empty-beta-block",
    )
