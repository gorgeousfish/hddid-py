from __future__ import annotations

from dataclasses import replace
import importlib
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pytest


def _load_inference_module():
    try:
        return importlib.import_module("hddid.inference")
    except ModuleNotFoundError as exc:
        pytest.fail(f"hddid.inference is missing: {exc}")


def _load_results_module():
    try:
        return importlib.import_module("hddid.results")
    except ModuleNotFoundError as exc:
        pytest.fail(f"hddid.results is missing: {exc}")


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


def test_solve_eq43_projection_matrix_rejects_boolean_cross_moments() -> None:
    inference_module = _load_inference_module()

    with pytest.raises(
        ValueError,
        match="cross_moment_hat must be numeric, not boolean",
    ):
        inference_module.solve_eq43_projection_matrix(
            sigma_x_hat=np.eye(2, dtype=float),
            cross_moment_hat=np.array([[True, False]]),
            lambda_double_prime=0.0,
        )


def test_solve_eq43_projection_matrix_rejects_string_numeric_aliases() -> None:
    inference_module = _load_inference_module()

    with pytest.raises(
        ValueError,
        match="cross_moment_hat must be numeric, not string",
    ):
        inference_module.solve_eq43_projection_matrix(
            sigma_x_hat=np.eye(2, dtype=float),
            cross_moment_hat=np.array([["1.0", "0.0"]]),
            lambda_double_prime=0.0,
        )

    with pytest.raises(
        ValueError,
        match="cross_moment_hat must be numeric, not string",
    ):
        inference_module.solve_eq43_projection_matrix(
            sigma_x_hat=np.eye(2, dtype=float),
            cross_moment_hat=np.array([[b"1.0", b"0.0"]]),
            lambda_double_prime=0.0,
        )

    with pytest.raises(
        ValueError,
        match="lambda_double_prime must be numeric, not string",
    ):
        inference_module.solve_eq43_projection_matrix(
            sigma_x_hat=np.eye(2, dtype=float),
            cross_moment_hat=np.array([[1.0, 0.0]]),
            lambda_double_prime="0.0",
        )

    with pytest.raises(
        ValueError,
        match="lambda_double_prime must be numeric, not string",
    ):
        inference_module.solve_eq43_projection_matrix(
            sigma_x_hat=np.eye(2, dtype=float),
            cross_moment_hat=np.array([[1.0, 0.0]]),
            lambda_double_prime=b"0.0",
        )

    with pytest.raises(
        ValueError,
        match="lambda_double_prime must be numeric, not string",
    ):
        inference_module.solve_eq43_projection_matrix(
            sigma_x_hat=np.eye(2, dtype=float),
            cross_moment_hat=np.array([[1.0, 0.0]]),
            lambda_double_prime=np.array("0.0", dtype=object),
        )


def test_solve_eq43_projection_matrix_minimizes_l1_objective() -> None:
    inference_module = _load_inference_module()

    sigma_x_hat = np.array(
        [
            [2.73711252, 0.61361018],
            [0.61361018, 0.27289310],
        ],
        dtype=float,
    )
    cross_moment_hat = np.array([[0.92023090, 0.57710379]], dtype=float)

    m_hat, metadata = inference_module.solve_eq43_projection_matrix(
        sigma_x_hat=sigma_x_hat,
        cross_moment_hat=cross_moment_hat,
        lambda_double_prime=0.4616724990135282,
    )

    np.testing.assert_allclose(
        m_hat,
        np.array([[0.188118278915, 0.0]], dtype=float),
        atol=1e-10,
        rtol=0.0,
    )
    assert np.linalg.norm(m_hat[0], ord=1) == pytest.approx(
        0.188118278915,
        abs=1e-10,
    )
    assert metadata["solver"] == "eq43-linear-programming-l1"
    assert metadata["feasible"] is True
    assert metadata["constraint_violation_max"] == pytest.approx(0.0, abs=1e-10)


def test_nonparametric_payload_accepts_nonunique_eq43_l1_optimum() -> None:
    inference_module = _load_inference_module()
    results_module = _load_results_module()

    m_hat = np.array([[0.5, 0.5]], dtype=float)
    sigma_x_hat = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
    cross_moment_hat = np.array([[1.0, 1.0]], dtype=float)
    orthogonal_basis_valid = np.array([[1.0], [2.0], [3.0]], dtype=float)
    sigma_f_hat = orthogonal_basis_valid.T @ orthogonal_basis_valid / 3
    omega_f_hat = np.array([[1.0]], dtype=float)
    sigma_f_inverse = np.linalg.inv(sigma_f_hat)
    v_f_hat = sigma_f_inverse @ omega_f_hat @ sigma_f_inverse.T
    score_moment = np.zeros(1, dtype=float)
    gamma_hat = np.array([2.0], dtype=float)
    bar_gamma_hat = gamma_hat - sigma_f_inverse @ score_moment
    evaluation_basis = np.array([[1.0], [2.0]], dtype=float)
    bar_f_at_z0 = evaluation_basis @ bar_gamma_hat
    covariance_at_grid = evaluation_basis @ v_f_hat @ evaluation_basis.T / 3
    sigma_z_hat = np.sqrt(np.diag(covariance_at_grid))
    alpha = 0.1
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    n_boot = 32
    random_state = 11
    covariance_sqrt = inference_module._matrix_square_root_psd(covariance_at_grid)
    gaussian_draws = (
        np.random.default_rng(random_state).standard_normal(
            size=(n_boot, evaluation_basis.shape[0])
        )
        @ covariance_sqrt.T
    )
    simulated_suprema = np.max(np.abs(gaussian_draws / sigma_z_hat[None, :]), axis=1)
    uniform_critical = float(np.quantile(simulated_suprema, 1.0 - alpha))
    uniform_summary = {
        "quantile_level": 1.0 - alpha,
        "simulated_suprema_min": float(np.min(simulated_suprema)),
        "simulated_suprema_max": float(np.max(simulated_suprema)),
        "simulated_suprema_mean": float(np.mean(simulated_suprema)),
        "simulated_suprema_std": float(np.std(simulated_suprema)),
    }

    payload = inference_module.NonparametricInferencePayload(
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
        uniform_standardization=sigma_z_hat,
        pointwise_confidence_interval=results_module.ConfidenceInterval(
            lower=bar_f_at_z0 - z_critical * sigma_z_hat,
            upper=bar_f_at_z0 + z_critical * sigma_z_hat,
            level=1.0 - alpha,
        ),
        uniform_band=results_module.UniformBand(
            lower=bar_f_at_z0 - uniform_critical * sigma_z_hat,
            upper=bar_f_at_z0 + uniform_critical * sigma_z_hat,
            level=1.0 - alpha,
            critical_value=uniform_critical,
            n_boot=n_boot,
            random_state=random_state,
        ),
        alpha=alpha,
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        optimization_metadata={
            "lambda_double_prime": 0.0,
            "max_threshold_steps": 25,
            "n_valid_obs": 3,
            "gamma_hat": gamma_hat,
            "omega_f_strategy": "paper-difference",
            "omega_f_selected_strategy": "paper-difference",
            "omega_f_primary_strategy": "paper-difference",
            "omega_f_primary_psd": True,
            "omega_f_primary_min_eigenvalue": 1.0,
            "omega_f_fallback_attempted": False,
            "omega_f_fallback_used": False,
            "omega_f_fallback_min_eigenvalue": None,
            "omega_f_selected_min_eigenvalue": 1.0,
            "uniform_band": {
                "critical_value": uniform_critical,
                "n_boot": n_boot,
                "random_state": random_state,
                **uniform_summary,
            },
        },
    )

    solver_m_hat, solver_metadata = inference_module.solve_eq43_projection_matrix(
        sigma_x_hat=sigma_x_hat,
        cross_moment_hat=cross_moment_hat,
        lambda_double_prime=0.0,
    )
    assert solver_metadata["feasible"] is True
    assert np.linalg.norm(payload.m_hat[0], ord=1) == pytest.approx(
        np.linalg.norm(solver_m_hat[0], ord=1),
        abs=1e-12,
    )
    np.testing.assert_allclose(
        payload.m_hat @ payload.sigma_x_hat - payload.cross_moment_hat,
        np.zeros_like(payload.cross_moment_hat),
        atol=1e-12,
        rtol=0.0,
    )

    nonoptimal_m_hat = np.array([[0.5, 0.5]], dtype=float)
    nonoptimal_cross_moment_hat = np.zeros((1, 2), dtype=float)
    nonoptimal_implied_cross = (
        nonoptimal_cross_moment_hat - nonoptimal_m_hat @ sigma_x_hat
    )
    nonoptimal_sigma_f_hat = (
        orthogonal_basis_valid.T @ orthogonal_basis_valid / 3
        + nonoptimal_implied_cross @ nonoptimal_m_hat.T
    )
    nonoptimal_sigma_f_inverse = np.linalg.inv(nonoptimal_sigma_f_hat)
    nonoptimal_v_f_hat = (
        nonoptimal_sigma_f_inverse @ omega_f_hat @ nonoptimal_sigma_f_inverse.T
    )
    nonoptimal_bar_gamma_hat = gamma_hat - nonoptimal_sigma_f_inverse @ score_moment
    nonoptimal_bar_f_at_z0 = evaluation_basis @ nonoptimal_bar_gamma_hat
    nonoptimal_covariance_at_grid = (
        evaluation_basis @ nonoptimal_v_f_hat @ evaluation_basis.T / 3
    )
    nonoptimal_sigma_z_hat = np.sqrt(np.diag(nonoptimal_covariance_at_grid))
    nonoptimal_covariance_sqrt = inference_module._matrix_square_root_psd(
        nonoptimal_covariance_at_grid
    )
    nonoptimal_gaussian_draws = (
        np.random.default_rng(random_state).standard_normal(
            size=(n_boot, evaluation_basis.shape[0])
        )
        @ nonoptimal_covariance_sqrt.T
    )
    nonoptimal_uniform_critical = float(
        np.quantile(
            np.max(
                np.abs(
                    nonoptimal_gaussian_draws / nonoptimal_sigma_z_hat[None, :]
                ),
                axis=1,
            ),
            1.0 - alpha,
        )
    )

    inference_module.NonparametricInferencePayload._validate_optimality = True
    try:
        with pytest.raises(
            ValueError,
            match=r"m_hat must attain the Eq\. \(4\.3\) projection L1 optimum",
        ):
            inference_module.NonparametricInferencePayload(
            m_hat=nonoptimal_m_hat,
            sigma_x_hat=sigma_x_hat,
            cross_moment_hat=nonoptimal_cross_moment_hat,
            orthogonal_basis_valid=orthogonal_basis_valid,
            sigma_f_hat=nonoptimal_sigma_f_hat,
            omega_f_hat=omega_f_hat,
            v_f_hat=nonoptimal_v_f_hat,
            score_moment=score_moment,
            bar_gamma_hat=nonoptimal_bar_gamma_hat,
            evaluation_basis=evaluation_basis,
            bar_f_at_z0=nonoptimal_bar_f_at_z0,
            sigma_z_hat=nonoptimal_sigma_z_hat,
            covariance_at_grid=nonoptimal_covariance_at_grid,
            uniform_standardization=nonoptimal_sigma_z_hat,
            pointwise_confidence_interval=results_module.ConfidenceInterval(
                lower=nonoptimal_bar_f_at_z0 - z_critical * nonoptimal_sigma_z_hat,
                upper=nonoptimal_bar_f_at_z0 + z_critical * nonoptimal_sigma_z_hat,
                level=1.0 - alpha,
            ),
            uniform_band=results_module.UniformBand(
                lower=(
                    nonoptimal_bar_f_at_z0
                    - nonoptimal_uniform_critical * nonoptimal_sigma_z_hat
                ),
                upper=(
                    nonoptimal_bar_f_at_z0
                    + nonoptimal_uniform_critical * nonoptimal_sigma_z_hat
                ),
                level=1.0 - alpha,
                critical_value=nonoptimal_uniform_critical,
                n_boot=n_boot,
                random_state=random_state,
            ),
            alpha=alpha,
            basis_family="polynomial",
            basis_degree=1,
            oracle_lane="r-parity-polynomial",
            optimization_metadata={
                "lambda_double_prime": 1.0,
                "max_threshold_steps": 25,
                "n_valid_obs": 3,
                "gamma_hat": gamma_hat,
                "omega_f_strategy": "paper-difference",
                "omega_f_selected_strategy": "paper-difference",
                "omega_f_primary_strategy": "paper-difference",
                "omega_f_primary_psd": True,
                "omega_f_primary_min_eigenvalue": 1.0,
                "omega_f_fallback_attempted": False,
                "omega_f_fallback_used": False,
                "omega_f_fallback_min_eigenvalue": None,
                "omega_f_selected_min_eigenvalue": 1.0,
                "uniform_band": {
                    "critical_value": nonoptimal_uniform_critical,
                    "n_boot": n_boot,
                    "random_state": random_state,
                },
            },
            )
    finally:
        inference_module.NonparametricInferencePayload._validate_optimality = False


def test_estimate_nonparametric_inference_matches_manual_eq43_slice(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()
    assert hasattr(inference_module, "NonparametricInferencePayload")
    assert hasattr(inference_module, "estimate_nonparametric_inference")

    payload, updated_result = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice[
            "lambda_double_prime"
        ],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )

    np.testing.assert_allclose(
        payload.m_hat,
        eq43_nonparametric_reference_slice["m_hat"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        payload.sigma_f_hat,
        eq43_nonparametric_reference_slice["sigma_f_hat"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        payload.omega_f_hat,
        eq43_nonparametric_reference_slice["omega_f_hat"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        payload.bar_gamma_hat,
        eq43_nonparametric_reference_slice["bar_gamma_hat"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        payload.bar_f_at_z0,
        eq43_nonparametric_reference_slice["bar_f_at_z0"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        payload.sigma_z_hat**2 * eq43_nonparametric_reference_slice["n_valid"],
        eq43_nonparametric_reference_slice["sigma_z_squared_hat"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        updated_result.standard_errors["f_se"],
        eq43_nonparametric_reference_slice["pointwise_standard_errors"],
        atol=1e-12,
        rtol=0.0,
    )


def test_estimate_nonparametric_inference_allows_empty_x_block() -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    inference_module = _load_inference_module()

    s_hat_valid = np.array([1.0, 2.0, 1.2, 3.0, 2.2], dtype=float)
    z_values = np.linspace(0.0, 1.0, s_hat_valid.shape[0])
    basis = np.column_stack([np.ones(s_hat_valid.shape[0]), z_values])
    score_payload = score_module.ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=np.ones(s_hat_valid.shape[0], dtype=bool),
        fold_ids=np.ones(s_hat_valid.shape[0], dtype=int),
        fold_diagnostics=[],
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        basis_matrix=basis,
        x_valid=np.empty((s_hat_valid.shape[0], 0), dtype=float),
        basis_valid_full=basis,
        basis_design_valid=basis[:, 1:],
        evaluation_basis=np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]]),
        pi_hat=np.full(s_hat_valid.shape[0], 0.5, dtype=float),
        phi0_hat=np.zeros(s_hat_valid.shape[0], dtype=float),
        phi1_hat=np.zeros(s_hat_valid.shape[0], dtype=float),
        rho_hat=np.ones(s_hat_valid.shape[0], dtype=float),
        intercept_dropped_for_design=True,
    )
    estimation_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
    )

    payload, updated_result = inference_module.estimate_nonparametric_inference(
        score_payload,
        estimation_payload,
        alpha=0.1,
        n_boot=64,
        random_state=123,
    )

    assert payload.m_hat.shape == (2, 0)
    assert payload.sigma_x_hat.shape == (0, 0)
    assert payload.optimization_metadata["x_dimension"] == 0
    assert payload.optimization_metadata["sigma_x_min_eigenvalue"] == np.inf
    np.testing.assert_allclose(
        payload.bar_f_at_z0,
        np.array([1.2, 1.88, 2.56]),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        payload.sigma_z_hat,
        np.array([0.268208873828, 0.240997925302, 0.405087644838]),
        atol=1e-12,
        rtol=0.0,
    )
    assert payload.uniform_band.critical_value == pytest.approx(1.947776335706)
    assert updated_result.standard_errors["f_se"].shape == (3,)


def test_estimate_nonparametric_inference_allows_high_dimensional_singular_sigma_x() -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    inference_module = _load_inference_module()

    x_valid = np.array(
        [
            [0.12573022, -0.13210486, 0.64042265, 0.10490012, -0.53566937],
            [-0.62327446, 0.04132598, -2.32503077, -0.21879166, -1.24591095],
            [-0.12853466, 1.36646347, -0.66519467, 0.35151007, 0.90347018],
            [-1.00961818, -0.20917557, -0.15922501, 0.54084558, 0.21465912],
            [-1.25906553, 1.51392377, 1.34587542, 0.78131140, 0.26445563],
            [0.35738041, -1.20831863, -0.00445413, 0.65647494, -1.28836146],
            [-0.43643525, -1.16980191, 1.73936788, -0.49591073, 0.32896963],
            [0.05202897, 0.68368619, 1.00396158, -0.61790704, 1.82201136],
        ],
        dtype=float,
    )
    x_valid = np.column_stack([x_valid, x_valid])
    z_values = np.linspace(0.0, 1.0, x_valid.shape[0])
    basis = np.column_stack([np.ones_like(z_values), z_values, z_values**2])
    beta_hat = np.zeros(x_valid.shape[1], dtype=float)
    gamma_hat = np.array([0.2, 0.5, -0.1], dtype=float)
    residual_valid = np.full(x_valid.shape[0], 0.1, dtype=float)
    fitted_f_valid = basis @ gamma_hat
    s_hat_valid = fitted_f_valid + residual_valid
    evaluation_basis = np.array([[1.0, 0.2, 0.04], [1.0, 0.8, 0.64]], dtype=float)
    score_payload = score_module.ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=np.ones(s_hat_valid.shape[0], dtype=bool),
        fold_ids=np.ones(s_hat_valid.shape[0], dtype=int),
        fold_diagnostics=[],
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        basis_matrix=basis,
        x_valid=x_valid,
        basis_valid_full=basis,
        basis_design_valid=basis[:, 1:],
        evaluation_basis=evaluation_basis,
        pi_hat=np.full(s_hat_valid.shape[0], 0.5, dtype=float),
        phi0_hat=np.zeros(s_hat_valid.shape[0], dtype=float),
        phi1_hat=np.zeros(s_hat_valid.shape[0], dtype=float),
        rho_hat=np.ones(s_hat_valid.shape[0], dtype=float),
        intercept_dropped_for_design=True,
    )
    estimation_payload = estimation_module.EstimationPayload(
        beta_hat=beta_hat,
        gamma_hat=gamma_hat,
        fitted_f_valid=fitted_f_valid,
        second_stage_prediction_valid=fitted_f_valid,
        residual_valid=residual_valid,
        projection_x_valid=x_valid,
        f_hat_at_z0=evaluation_basis @ gamma_hat,
        optimization_metadata={
            "solver": "manual-high-dimensional-singular-sigma-x",
            "n_valid_obs": int(s_hat_valid.shape[0]),
            "beta_dimension": int(x_valid.shape[1]),
            "basis_dimension_full": int(basis.shape[1]),
        },
    )

    payload, updated_result = inference_module.estimate_nonparametric_inference(
        score_payload,
        estimation_payload,
        alpha=0.1,
        n_boot=32,
        random_state=7,
    )

    assert payload.sigma_x_hat.shape == (10, 10)
    assert payload.optimization_metadata["x_dimension"] == 10
    assert payload.optimization_metadata["sigma_x_rank"] == 5
    assert payload.optimization_metadata["sigma_x_min_eigenvalue"] == pytest.approx(0.0)
    assert payload.optimization_metadata["constraint_violation_max"] == pytest.approx(0.0)
    assert payload.optimization_metadata["omega_f_selected_strategy"] == (
        "paper-difference"
    )
    assert payload.optimization_metadata["omega_f_selected_min_eigenvalue"] > 0.0
    assert np.all(payload.sigma_z_hat > 0.0)
    assert updated_result.standard_errors["f_se"].shape == (2,)


def test_estimate_nonparametric_inference_rejects_nearly_singular_sigma_f_before_inversion() -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    inference_module = _load_inference_module()

    n_valid = 3
    basis_scale = 1e-12
    basis = np.full((n_valid, 1), basis_scale, dtype=float)
    x_valid = np.empty((n_valid, 0), dtype=float)
    s_hat_valid = np.array([1.0, -1.0, 1.0], dtype=float)
    gamma_hat = np.array([0.0], dtype=float)
    fitted_f_valid = basis @ gamma_hat
    score_payload = score_module.ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=np.ones(n_valid, dtype=bool),
        fold_ids=np.ones(n_valid, dtype=int),
        fold_diagnostics=[],
        basis_family="polynomial",
        basis_degree=0,
        oracle_lane="r-parity-polynomial",
        basis_matrix=basis,
        x_valid=x_valid,
        basis_valid_full=basis,
        basis_design_valid=basis,
        evaluation_basis=np.array([[basis_scale]], dtype=float),
        pi_hat=np.full(n_valid, 0.5, dtype=float),
        phi0_hat=np.zeros(n_valid, dtype=float),
        phi1_hat=np.zeros(n_valid, dtype=float),
        rho_hat=np.ones(n_valid, dtype=float),
        intercept_dropped_for_design=False,
    )
    estimation_payload = estimation_module.EstimationPayload(
        beta_hat=np.empty(0, dtype=float),
        gamma_hat=gamma_hat,
        fitted_f_valid=fitted_f_valid,
        second_stage_prediction_valid=fitted_f_valid,
        residual_valid=s_hat_valid,
        projection_x_valid=x_valid,
        f_hat_at_z0=np.array([0.0], dtype=float),
        optimization_metadata={
            "solver": "manual-nearly-singular-sigma-f",
            "n_valid_obs": n_valid,
            "beta_dimension": 0,
            "basis_dimension_full": 1,
        },
    )

    with pytest.raises(
        inference_module.SingularCovarianceError,
        match="sigma_f_hat must be full rank for nonparametric inference",
    ) as exc_info:
        inference_module.estimate_nonparametric_inference(
            score_payload,
            estimation_payload,
            n_boot=8,
            random_state=0,
        )

    metadata = exc_info.value.metadata
    assert metadata["failure_kind"] == "singular-covariance"
    assert metadata["matrix_name"] == "sigma_f_hat"
    assert metadata["sigma_f_min_singular_value"] <= np.finfo(float).eps
    assert metadata["basis_dimension_full"] == 1
    assert metadata["x_dimension"] == 0


def test_estimate_nonparametric_inference_rejects_alpha_string_alias(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    with pytest.raises(ValueError, match="alpha must be numeric, not string"):
        inference_module.estimate_nonparametric_inference(
            eq43_nonparametric_reference_slice["score_payload"],
            eq43_nonparametric_reference_slice["estimation_payload"],
            alpha="0.05",
            lambda_double_prime=eq43_nonparametric_reference_slice[
                "lambda_double_prime"
            ],
            n_boot=eq43_nonparametric_reference_slice["n_boot"],
            random_state=eq43_nonparametric_reference_slice["random_state"],
        )

    with pytest.raises(ValueError, match="alpha must be numeric, not string"):
        inference_module.estimate_nonparametric_inference(
            eq43_nonparametric_reference_slice["score_payload"],
            eq43_nonparametric_reference_slice["estimation_payload"],
            alpha=np.array("0.05", dtype=object),
            lambda_double_prime=eq43_nonparametric_reference_slice[
                "lambda_double_prime"
            ],
            n_boot=eq43_nonparametric_reference_slice["n_boot"],
            random_state=eq43_nonparametric_reference_slice["random_state"],
        )

    with pytest.raises(ValueError, match="alpha must be numeric"):
        inference_module.estimate_nonparametric_inference(
            eq43_nonparametric_reference_slice["score_payload"],
            eq43_nonparametric_reference_slice["estimation_payload"],
            alpha=True,
            lambda_double_prime=eq43_nonparametric_reference_slice[
                "lambda_double_prime"
            ],
            n_boot=eq43_nonparametric_reference_slice["n_boot"],
            random_state=eq43_nonparametric_reference_slice["random_state"],
        )

    for alpha in ([0.05], np.array([0.05])):
        with pytest.raises(ValueError, match="alpha must be a scalar"):
            inference_module.estimate_nonparametric_inference(
                eq43_nonparametric_reference_slice["score_payload"],
                eq43_nonparametric_reference_slice["estimation_payload"],
                alpha=alpha,
                lambda_double_prime=eq43_nonparametric_reference_slice[
                    "lambda_double_prime"
                ],
                n_boot=eq43_nonparametric_reference_slice["n_boot"],
                random_state=eq43_nonparametric_reference_slice["random_state"],
            )


def test_estimate_nonparametric_inference_preserves_explicit_grid_and_oracle_lane(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    payload, updated_result = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice["lambda_double_prime"],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )

    assert payload.evaluation_basis.shape == (3, 3)
    assert payload.bar_f_at_z0.shape == (3,)
    assert payload.oracle_lane == "r-parity-polynomial"
    assert payload.basis_family == "polynomial"
    assert payload.optimization_metadata["evaluation_grid_size"] == 3
    assert payload.optimization_metadata["basis_dimension_full"] == 3
    assert updated_result.diagnostics is not None
    assert updated_result.diagnostics.oracle_lane == "r-parity-polynomial"


def test_nonparametric_inference_uses_full_basis_not_eq31_design_block(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    payload, _ = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice[
            "lambda_double_prime"
        ],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )
    score_payload = eq43_nonparametric_reference_slice["score_payload"]
    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    basis_design_valid = np.asarray(score_payload.basis_design_valid, dtype=float)
    x_valid = np.asarray(score_payload.x_valid, dtype=float)
    n_valid = eq43_nonparametric_reference_slice["n_valid"]

    full_cross_moment = basis_valid_full.T @ x_valid / n_valid
    design_cross_moment = basis_design_valid.T @ x_valid / n_valid

    assert payload.optimization_metadata["basis_dimension_full"] == (
        basis_valid_full.shape[1]
    )
    assert basis_valid_full.shape[1] == basis_design_valid.shape[1] + 1
    assert payload.m_hat.shape[0] == basis_valid_full.shape[1]
    assert payload.cross_moment_hat.shape[0] == basis_valid_full.shape[1]
    np.testing.assert_allclose(
        payload.cross_moment_hat,
        full_cross_moment,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        payload.cross_moment_hat[1:],
        design_cross_moment,
        atol=1e-12,
        rtol=0.0,
    )


def test_estimate_nonparametric_inference_fills_pointwise_ci_and_uniform_band(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    payload, updated_result = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice["lambda_double_prime"],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )

    nonparametric_ci = updated_result.intervals["nonparametric_ci"]
    uniform_band = updated_result.intervals["uniform_band"]

    np.testing.assert_allclose(
        nonparametric_ci.lower,
        eq43_nonparametric_reference_slice["pointwise_ci_lower"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        nonparametric_ci.upper,
        eq43_nonparametric_reference_slice["pointwise_ci_upper"],
        atol=1e-12,
        rtol=0.0,
    )
    assert np.all(nonparametric_ci.lower <= nonparametric_ci.upper)
    assert np.all(uniform_band.lower <= nonparametric_ci.lower)
    assert np.all(uniform_band.upper >= nonparametric_ci.upper)
    assert getattr(uniform_band, "critical_value") == pytest.approx(
        eq43_nonparametric_reference_slice["uniform_critical_value"],
    )
    assert (
        getattr(uniform_band, "n_boot") == eq43_nonparametric_reference_slice["n_boot"]
    )
    assert (
        getattr(uniform_band, "random_state")
        == eq43_nonparametric_reference_slice["random_state"]
    )
    assert (
        payload.optimization_metadata["uniform_band"]["n_boot"]
        == eq43_nonparametric_reference_slice["n_boot"]
    )
    uniform_metadata = payload.optimization_metadata["uniform_band"]
    assert uniform_metadata["quantile_level"] == pytest.approx(
        1.0 - eq43_nonparametric_reference_slice["alpha"]
    )
    assert uniform_metadata["simulated_suprema_min"] >= 0.0
    assert uniform_metadata["simulated_suprema_min"] <= uniform_metadata[
        "simulated_suprema_mean"
    ]
    assert uniform_metadata["simulated_suprema_mean"] <= uniform_metadata[
        "simulated_suprema_max"
    ]
    assert uniform_metadata["simulated_suprema_std"] > 0.0
    markdown = updated_result.to_markdown(style="legacy", missing_value="")
    estimation_payload = eq43_nonparametric_reference_slice["estimation_payload"]
    for index, (f_hat, bar_f, standard_error, lower, upper) in enumerate(
        zip(
            estimation_payload.f_hat_at_z0,
            payload.bar_f_at_z0,
            payload.sigma_z_hat,
            payload.pointwise_confidence_interval.lower,
            payload.pointwise_confidence_interval.upper,
            strict=True,
        )
    ):
        assert (
            f"| Nonparametric | f_hat_at_z0 | {index} | {f_hat:.4f} |  |  |"
            in markdown
        )
        assert (
            f"| Nonparametric | bar_f_at_z0 | {index} | {bar_f:.4f} | "
            f"{standard_error:.4f} | [{lower:.4f}, {upper:.4f}] "
            f"({(1.0 - payload.alpha) * 100.0:.1f}%) |"
            in markdown
        )


def test_estimate_nonparametric_inference_uses_linear_solves_not_explicit_inverse(
    eq43_nonparametric_reference_slice: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_module = _load_inference_module()

    def reject_explicit_inverse(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("nonparametric inference must not call np.linalg.inv")

    monkeypatch.setattr(inference_module.np.linalg, "inv", reject_explicit_inverse)

    payload, _ = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice["lambda_double_prime"],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )

    np.testing.assert_allclose(
        payload.bar_gamma_hat,
        eq43_nonparametric_reference_slice["bar_gamma_hat"],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        payload.v_f_hat,
        eq43_nonparametric_reference_slice["v_f_hat"],
        atol=1e-12,
        rtol=0.0,
    )


def test_matrix_square_root_psd_uses_explicit_symmetric_part() -> None:
    inference_module = _load_inference_module()
    covariance = np.array(
        [
            [1.0, 0.25 + 5e-13],
            [0.25 - 5e-13, 0.5],
        ],
        dtype=float,
    )

    covariance_sqrt = inference_module._matrix_square_root_psd(covariance)
    symmetric_covariance = 0.5 * (covariance + covariance.T)

    np.testing.assert_allclose(
        covariance_sqrt @ covariance_sqrt.T,
        symmetric_covariance,
        atol=1e-12,
        rtol=0.0,
    )


def test_matrix_square_root_psd_rejects_indefinite_input() -> None:
    inference_module = _load_inference_module()
    covariance = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="matrix square root requires positive semidefinite input",
    ):
        inference_module._matrix_square_root_psd(covariance)


def test_estimate_nonparametric_inference_default_random_state_is_reproducible(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    first_payload, first_result = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice["lambda_double_prime"],
        n_boot=64,
    )
    second_payload, second_result = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice["lambda_double_prime"],
        n_boot=64,
    )

    assert first_payload.uniform_band.random_state == 0
    assert first_payload.optimization_metadata["bootstrap_seed"] == 0
    assert first_payload.optimization_metadata["uniform_band"]["random_state"] == 0
    assert first_payload.uniform_band.critical_value == pytest.approx(
        second_payload.uniform_band.critical_value,
        abs=0.0,
        rel=0.0,
    )
    np.testing.assert_allclose(
        first_result.intervals["uniform_band"].lower,
        second_result.intervals["uniform_band"].lower,
        atol=0.0,
        rtol=0.0,
    )


def test_estimate_nonparametric_inference_exposes_uniform_process_objects(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    payload, _ = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice["lambda_double_prime"],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )

    expected_covariance_at_grid = (
        eq43_nonparametric_reference_slice["evaluation_basis"]
        @ eq43_nonparametric_reference_slice["v_f_hat"]
        @ eq43_nonparametric_reference_slice["evaluation_basis"].T
        / eq43_nonparametric_reference_slice["n_valid"]
    )
    expected_standardization = np.sqrt(np.diag(expected_covariance_at_grid))

    np.testing.assert_allclose(
        payload.covariance_at_grid,
        expected_covariance_at_grid,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        payload.uniform_standardization,
        expected_standardization,
        atol=1e-12,
        rtol=0.0,
    )
    assert payload.optimization_metadata["uniform_band"][
        "covariance_source"
    ] == "evaluation_basis @ v_f_hat @ evaluation_basis.T / n_valid"
    assert payload.optimization_metadata["uniform_band"][
        "standardization_source"
    ] == "sqrt(diag(covariance_at_grid))"
    assert payload.optimization_metadata["sigma_x_min_eigenvalue"] == pytest.approx(
        float(np.min(np.linalg.eigvalsh(payload.sigma_x_hat)))
    )
    assert payload.optimization_metadata["sigma_x_min_eigenvalue"] > 0.0
    sigma_f_singular_values = np.linalg.svd(payload.sigma_f_hat, compute_uv=False)
    assert payload.optimization_metadata["sigma_f_min_singular_value"] == (
        pytest.approx(float(np.min(sigma_f_singular_values)))
    )
    assert payload.optimization_metadata["sigma_f_condition_number"] == (
        pytest.approx(
            float(
                np.max(sigma_f_singular_values)
                / np.min(sigma_f_singular_values)
            )
        )
    )
    assert payload.optimization_metadata["sigma_f_min_singular_value"] > 0.0
    assert payload.optimization_metadata["omega_f_selected_strategy"] == (
        "paper-difference"
    )
    assert payload.optimization_metadata["omega_f_primary_psd"] is True
    assert payload.optimization_metadata["omega_f_fallback_attempted"] is False
    assert payload.optimization_metadata["omega_f_fallback_used"] is False
    assert payload.optimization_metadata["omega_f_fallback_min_eigenvalue"] is None
    assert payload.optimization_metadata["omega_f_selected_min_eigenvalue"] == (
        pytest.approx(payload.optimization_metadata["omega_f_primary_min_eigenvalue"])
    )


def test_nonparametric_inference_omega_f_uses_paper_difference_formula(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    payload, _ = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice["lambda_double_prime"],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )
    score_payload = eq43_nonparametric_reference_slice["score_payload"]
    estimation_payload = eq43_nonparametric_reference_slice["estimation_payload"]
    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    x_valid = np.asarray(score_payload.x_valid, dtype=float)
    residual_valid = np.asarray(estimation_payload.residual_valid, dtype=float)
    n_valid = int(residual_valid.shape[0])

    weighted_basis = basis_valid_full * residual_valid[:, None]
    weighted_x = x_valid * residual_valid[:, None]
    r_snapshot_omega_f = (
        weighted_basis.T @ weighted_basis
        - payload.m_hat @ weighted_x.T @ weighted_x @ payload.m_hat.T
    ) / n_valid
    orthogonal_score_omega_f = (
        (payload.orthogonal_basis_valid * residual_valid[:, None]).T
        @ (payload.orthogonal_basis_valid * residual_valid[:, None])
        / n_valid
    )
    residual_weighted_cross_moment = weighted_basis.T @ weighted_x / n_valid
    residual_weighted_x_covariance = weighted_x.T @ weighted_x / n_valid
    residual_weighted_cross_gap = (
        residual_weighted_cross_moment
        - payload.m_hat @ residual_weighted_x_covariance
    )
    residual_weighted_cross_correction = (
        residual_weighted_cross_gap @ payload.m_hat.T
        + payload.m_hat @ residual_weighted_cross_gap.T
    )

    np.testing.assert_allclose(
        payload.omega_f_hat,
        r_snapshot_omega_f,
        atol=1e-12,
        rtol=0.0,
    )
    assert (
        np.max(np.abs(payload.omega_f_hat - orthogonal_score_omega_f)) > 1e-3
    )
    diagnostics = inference_module.diagnose_nonparametric_omega_f(
        score_payload,
        estimation_payload,
        payload,
    )
    assert diagnostics["target_kind"] == "nonparametric-omega-f-diagnostic"
    assert diagnostics["matrix_name"] == "omega_f_hat"
    assert diagnostics["matrix_shape"] == payload.omega_f_hat.shape
    assert diagnostics["omega_f_primary_min_eigenvalue"] == pytest.approx(
        float(np.min(np.linalg.eigvalsh(payload.omega_f_hat)))
    )
    assert diagnostics["omega_f_paper_difference_trace"] == pytest.approx(
        float(np.trace(payload.omega_f_hat))
    )
    assert diagnostics["omega_f_unweighted_cross_identity_max_abs"] <= 1e-10
    assert diagnostics["omega_f_residual_weighted_cross_identity_max_abs"] > 0.0
    assert diagnostics["omega_f_paper_difference_vs_orthogonal_score_max_abs"] == (
        pytest.approx(
            float(np.max(np.abs(payload.omega_f_hat - orthogonal_score_omega_f)))
        )
    )
    assert diagnostics["omega_f_residual_weighted_cross_correction_max_abs"] == (
        pytest.approx(float(np.max(np.abs(residual_weighted_cross_correction))))
    )
    assert diagnostics["omega_f_residual_weighted_cross_correction_trace"] == (
        pytest.approx(float(np.trace(residual_weighted_cross_correction)))
    )
    assert diagnostics[
        "omega_f_residual_weighted_cross_correction_min_eigenvalue"
    ] == pytest.approx(
        float(np.min(np.linalg.eigvalsh(residual_weighted_cross_correction)))
    )
    assert diagnostics["omega_f_paper_vs_orthogonal_reconstruction_max_abs"] <= 1e-8
    np.testing.assert_allclose(
        payload.omega_f_hat - orthogonal_score_omega_f,
        residual_weighted_cross_correction,
        atol=1e-9,
        rtol=1e-12,
    )


def test_nonparametric_omega_f_diagnostic_rejects_empty_valid_sample() -> None:
    inference_module = _load_inference_module()
    score_payload = SimpleNamespace(
        basis_valid_full=np.empty((0, 1)),
        x_valid=np.empty((0, 0)),
        evaluation_basis=np.ones((1, 1)),
        s_hat_valid=np.empty(0),
        oracle_lane="aipw",
        basis_family="power",
        basis_degree=1,
    )
    estimation_payload = SimpleNamespace(
        beta_hat=np.empty(0),
        gamma_hat=np.zeros(1),
        fitted_f_valid=np.empty(0),
        second_stage_prediction_valid=np.empty(0),
        residual_valid=np.empty(0),
        projection_x_valid=np.empty((0, 0)),
        f_hat_at_z0=np.zeros(1),
        optimization_metadata={"n_valid_obs": 0},
    )
    nonparametric_payload = SimpleNamespace(m_hat=np.empty((1, 0)))

    with pytest.raises(inference_module.InvalidInferenceInputError) as excinfo:
        inference_module.diagnose_nonparametric_omega_f(
            score_payload,
            estimation_payload,
            nonparametric_payload,
        )

    assert excinfo.value.metadata["target_kind"] == (
        "nonparametric-omega-f-diagnostic"
    )
    assert excinfo.value.metadata["failure_kind"] == "invalid-input"
    assert excinfo.value.metadata["n_valid_obs"] == 0
    assert excinfo.value.metadata["valid_sample_size"] == 0


def test_nonparametric_omega_f_diagnostic_rejects_stale_payload_pair(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    payload, _ = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice["lambda_double_prime"],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )
    stale_estimation_payload = replace(
        eq43_nonparametric_reference_slice["estimation_payload"],
        residual_valid=np.asarray(
            eq43_nonparametric_reference_slice["estimation_payload"].residual_valid,
            dtype=float,
        )
        + 0.125,
    )

    with pytest.raises(inference_module.InvalidInferenceInputError) as excinfo:
        inference_module.diagnose_nonparametric_omega_f(
            eq43_nonparametric_reference_slice["score_payload"],
            stale_estimation_payload,
            payload,
        )

    assert excinfo.value.metadata["target_kind"] == (
        "nonparametric-omega-f-diagnostic"
    )
    assert excinfo.value.metadata["failure_kind"] == "payload-score-mismatch"
    assert excinfo.value.metadata["field_name"] == "residual_valid"


def test_nonparametric_omega_f_diagnostic_rejects_stale_sigma_f_payload(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    payload, _ = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice["lambda_double_prime"],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )
    stale_payload = replace(payload)
    stale_payload.sigma_f_hat = np.asarray(payload.sigma_f_hat, dtype=float).copy()
    stale_payload.sigma_f_hat[0, 0] += 0.125

    with pytest.raises(inference_module.InvalidInferenceInputError) as excinfo:
        inference_module.diagnose_nonparametric_omega_f(
            eq43_nonparametric_reference_slice["score_payload"],
            eq43_nonparametric_reference_slice["estimation_payload"],
            stale_payload,
        )

    assert excinfo.value.metadata["target_kind"] == (
        "nonparametric-omega-f-diagnostic"
    )
    assert excinfo.value.metadata["failure_kind"] == "payload-score-mismatch"
    assert excinfo.value.metadata["field_name"] == "sigma_f_hat"


def test_estimate_nonparametric_inference_uses_symmetric_sandwich_for_regularized_eq43(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    payload, _ = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=0.01,
        n_boot=32,
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )

    sigma_f_inverse = np.linalg.inv(payload.sigma_f_hat)
    expected_v_f_hat = sigma_f_inverse @ payload.omega_f_hat @ sigma_f_inverse.T
    legacy_no_transpose = sigma_f_inverse @ payload.omega_f_hat @ sigma_f_inverse

    np.testing.assert_allclose(payload.v_f_hat, expected_v_f_hat, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(payload.v_f_hat, payload.v_f_hat.T, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        payload.covariance_at_grid,
        payload.covariance_at_grid.T,
        atol=1e-12,
        rtol=0.0,
    )
    assert np.max(np.abs(legacy_no_transpose - expected_v_f_hat)) > 1e-8


def test_nonparametric_payload_accepts_alternate_equal_l1_eq43_projection() -> None:
    inference_module = _load_inference_module()
    results_module = _load_results_module()

    sigma_x_hat = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    cross_moment_hat = np.array([[1.0, 1.0]], dtype=float)
    m_hat = np.array([[0.5, 0.5]], dtype=float)
    orthogonal_basis_valid = np.array([[1.0], [2.0]], dtype=float)
    n_valid = 2
    sigma_f_hat = orthogonal_basis_valid.T @ orthogonal_basis_valid / n_valid
    omega_f_hat = np.array([[1.0]], dtype=float)
    sigma_f_inverse = np.linalg.inv(sigma_f_hat)
    v_f_hat = sigma_f_inverse @ omega_f_hat @ sigma_f_inverse.T
    gamma_hat = np.array([0.4], dtype=float)
    score_moment = np.zeros(1, dtype=float)
    bar_gamma_hat = gamma_hat - sigma_f_inverse @ score_moment
    evaluation_basis = np.array([[1.0], [2.0]], dtype=float)
    bar_f_at_z0 = evaluation_basis @ bar_gamma_hat
    covariance_at_grid = evaluation_basis @ v_f_hat @ evaluation_basis.T / n_valid
    sigma_z_hat = np.sqrt(np.diag(covariance_at_grid))
    alpha = 0.1
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    n_boot = 64
    random_state = 11
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_at_grid)
    covariance_sqrt = eigenvectors @ np.diag(
        np.sqrt(np.clip(eigenvalues, 0.0, None))
    )
    rng = np.random.default_rng(random_state)
    gaussian_draws = (
        rng.standard_normal(size=(n_boot, evaluation_basis.shape[0]))
        @ covariance_sqrt.T
    )
    simulated_suprema = np.max(
        np.abs(gaussian_draws / sigma_z_hat[None, :]),
        axis=1,
    )
    uniform_critical_value = float(np.quantile(simulated_suprema, 1.0 - alpha))
    uniform_summary = {
        "quantile_level": 1.0 - alpha,
        "simulated_suprema_min": float(np.min(simulated_suprema)),
        "simulated_suprema_max": float(np.max(simulated_suprema)),
        "simulated_suprema_mean": float(np.mean(simulated_suprema)),
        "simulated_suprema_std": float(np.std(simulated_suprema)),
    }

    solver_m_hat, solver_metadata = inference_module.solve_eq43_projection_matrix(
        sigma_x_hat=sigma_x_hat,
        cross_moment_hat=cross_moment_hat,
        lambda_double_prime=0.0,
    )
    assert solver_metadata["feasible"] is True
    assert np.linalg.norm(m_hat[0], ord=1) == pytest.approx(
        np.linalg.norm(solver_m_hat[0], ord=1),
        abs=1e-12,
    )
    np.testing.assert_allclose(
        m_hat @ sigma_x_hat - cross_moment_hat,
        np.zeros_like(cross_moment_hat),
        atol=1e-12,
        rtol=0.0,
    )

    payload = inference_module.NonparametricInferencePayload(
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
        uniform_standardization=sigma_z_hat,
        pointwise_confidence_interval=results_module.ConfidenceInterval(
            lower=bar_f_at_z0 - z_critical * sigma_z_hat,
            upper=bar_f_at_z0 + z_critical * sigma_z_hat,
            level=1.0 - alpha,
        ),
        uniform_band=results_module.UniformBand(
            lower=bar_f_at_z0 - uniform_critical_value * sigma_z_hat,
            upper=bar_f_at_z0 + uniform_critical_value * sigma_z_hat,
            level=1.0 - alpha,
            critical_value=uniform_critical_value,
            n_boot=n_boot,
            random_state=random_state,
        ),
        alpha=alpha,
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        optimization_metadata={
            "lambda_double_prime": 0.0,
            "max_threshold_steps": 25,
            "n_valid_obs": n_valid,
            "gamma_hat": gamma_hat,
            "omega_f_strategy": "paper-difference",
            "omega_f_selected_strategy": "paper-difference",
            "omega_f_primary_strategy": "paper-difference",
            "omega_f_primary_psd": True,
            "omega_f_primary_min_eigenvalue": 1.0,
            "omega_f_fallback_attempted": False,
            "omega_f_fallback_used": False,
            "omega_f_fallback_min_eigenvalue": None,
            "omega_f_selected_min_eigenvalue": 1.0,
            "uniform_band": {
                "critical_value": uniform_critical_value,
                "n_boot": n_boot,
                "random_state": random_state,
                **uniform_summary,
            },
        },
    )

    np.testing.assert_allclose(payload.m_hat, m_hat, atol=0.0, rtol=0.0)


def test_estimate_nonparametric_inference_uniform_band_is_reproducible(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    first_payload, first_result = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice["lambda_double_prime"],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )
    second_payload, second_result = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice["lambda_double_prime"],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )

    first_band = first_result.intervals["uniform_band"]
    second_band = second_result.intervals["uniform_band"]

    np.testing.assert_allclose(
        first_payload.bar_gamma_hat, second_payload.bar_gamma_hat, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(first_band.lower, second_band.lower, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first_band.upper, second_band.upper, atol=0.0, rtol=0.0)
    assert getattr(first_band, "critical_value") == pytest.approx(
        getattr(second_band, "critical_value")
    )


def test_nonparametric_inference_payload_rejects_nonfinite_direct_outputs(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()
    results_module = _load_results_module()

    payload, _ = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice[
            "lambda_double_prime"
        ],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )

    with pytest.raises(ValueError, match="sigma_z_hat must contain only finite values"):
        replace(payload, sigma_z_hat=np.full_like(payload.sigma_z_hat, np.inf))

    with pytest.raises(ValueError, match="alpha must be numeric, not string"):
        replace(payload, alpha="0.05")

    with pytest.raises(ValueError, match="alpha must be numeric, not string"):
        replace(payload, alpha=np.array("0.05", dtype=object))

    with pytest.raises(ValueError, match="alpha must be numeric"):
        replace(payload, alpha=True)

    with pytest.raises(
        ValueError,
        match=r"sigma_z_hat must equal sqrt\(diag\(covariance_at_grid\)\)",
    ):
        replace(payload, sigma_z_hat=payload.sigma_z_hat * 1.1)

    with pytest.raises(
        ValueError,
        match="evaluation_basis must contain at least one grid point",
    ):
        replace(
            payload,
            evaluation_basis=np.empty((0, payload.evaluation_basis.shape[1])),
            bar_f_at_z0=np.empty(0),
            sigma_z_hat=np.empty(0),
            covariance_at_grid=np.empty((0, 0)),
            uniform_standardization=np.empty(0),
            pointwise_confidence_interval=results_module.ConfidenceInterval(
                lower=np.empty(0),
                upper=np.empty(0),
                level=1.0 - payload.alpha,
            ),
            uniform_band=results_module.UniformBand(
                lower=np.empty(0),
                upper=np.empty(0),
                level=1.0 - payload.alpha,
                critical_value=payload.uniform_band.critical_value,
                n_boot=payload.uniform_band.n_boot,
                random_state=payload.uniform_band.random_state,
            ),
        )

    stale_pointwise_lower = np.asarray(
        payload.pointwise_confidence_interval.lower,
        dtype=float,
    ).copy()
    stale_pointwise_lower[0] -= 0.01
    with pytest.raises(
        ValueError,
        match=r"pointwise_confidence_interval\.lower must equal "
        r"center - critical_value \* scale",
    ):
        replace(
            payload,
            pointwise_confidence_interval=results_module.ConfidenceInterval(
                lower=stale_pointwise_lower,
                upper=np.asarray(payload.pointwise_confidence_interval.upper, dtype=float),
                level=payload.pointwise_confidence_interval.level,
            ),
        )

    with pytest.raises(
        ValueError,
        match="uniform_band.upper must contain only finite values",
    ):
        replace(
            payload,
            uniform_band=results_module.UniformBand(
                lower=np.asarray(payload.uniform_band.lower, dtype=float),
                upper=np.full_like(payload.bar_f_at_z0, np.inf),
                level=payload.uniform_band.level,
                critical_value=payload.uniform_band.critical_value,
                n_boot=payload.uniform_band.n_boot,
                random_state=payload.uniform_band.random_state,
            ),
        )

    stale_uniform_upper = np.asarray(payload.uniform_band.upper, dtype=float).copy()
    stale_uniform_upper[0] += 0.01
    with pytest.raises(
        ValueError,
        match=r"uniform_band\.upper must equal center \+ critical_value \* scale",
    ):
        replace(
            payload,
            uniform_band=results_module.UniformBand(
                lower=np.asarray(payload.uniform_band.lower, dtype=float),
                upper=stale_uniform_upper,
                level=payload.uniform_band.level,
                critical_value=payload.uniform_band.critical_value,
                n_boot=payload.uniform_band.n_boot,
                random_state=payload.uniform_band.random_state,
            ),
        )

    stale_uniform_critical_value = float(payload.uniform_band.critical_value) * 1.05
    inference_module.NonparametricInferencePayload._validate_optimality = True
    try:
        with pytest.raises(
            ValueError,
            match=(
                "uniform_band.critical_value must match the finite-grid "
                "Gaussian bootstrap for covariance_at_grid"
            ),
        ):
            replace(
                payload,
                uniform_band=results_module.UniformBand(
                    lower=payload.bar_f_at_z0
                    - stale_uniform_critical_value * payload.sigma_z_hat,
                    upper=payload.bar_f_at_z0
                    + stale_uniform_critical_value * payload.sigma_z_hat,
                    level=payload.uniform_band.level,
                    critical_value=stale_uniform_critical_value,
                    n_boot=payload.uniform_band.n_boot,
                    random_state=payload.uniform_band.random_state,
                ),
            )
    finally:
        inference_module.NonparametricInferencePayload._validate_optimality = False

    stale_uniform_metadata = dict(payload.optimization_metadata)
    stale_uniform_metadata["uniform_band"] = dict(
        payload.optimization_metadata["uniform_band"]
    )
    stale_uniform_metadata["uniform_band"]["critical_value"] = (
        float(payload.uniform_band.critical_value) * 1.05
    )
    with pytest.raises(
        ValueError,
        match="uniform_band metadata critical_value must match uniform_band",
    ):
        replace(payload, optimization_metadata=stale_uniform_metadata)

    inference_module.NonparametricInferencePayload._validate_optimality = True
    try:
        missing_suprema_metadata = dict(payload.optimization_metadata)
        missing_suprema_metadata["uniform_band"] = dict(
            payload.optimization_metadata["uniform_band"]
        )
        del missing_suprema_metadata["uniform_band"]["simulated_suprema_mean"]
        with pytest.raises(
            ValueError,
            match="uniform_band metadata simulated_suprema_mean must be provided",
        ):
            replace(payload, optimization_metadata=missing_suprema_metadata)
    finally:
        inference_module.NonparametricInferencePayload._validate_optimality = False

    for invalid_critical_value, message in (
        ("1.0", "uniform_band metadata critical_value must be numeric, not string"),
        (b"1.0", "uniform_band metadata critical_value must be numeric, not string"),
        (True, "uniform_band metadata critical_value must be numeric, not boolean"),
        (np.nan, "uniform_band metadata critical_value must be finite"),
    ):
        invalid_uniform_metadata = dict(payload.optimization_metadata)
        invalid_uniform_metadata["uniform_band"] = dict(
            payload.optimization_metadata["uniform_band"]
        )
        invalid_uniform_metadata["uniform_band"]["critical_value"] = (
            invalid_critical_value
        )
        with pytest.raises(ValueError, match=message):
            replace(payload, optimization_metadata=invalid_uniform_metadata)

    inference_module.NonparametricInferencePayload._validate_optimality = True
    try:
        stale_suprema_metadata = dict(payload.optimization_metadata)
        stale_suprema_metadata["uniform_band"] = dict(
            payload.optimization_metadata["uniform_band"]
        )
        stale_suprema_metadata["uniform_band"]["simulated_suprema_mean"] = (
            float(stale_suprema_metadata["uniform_band"]["simulated_suprema_mean"]) * 1.05
        )
        with pytest.raises(
            ValueError,
            match="uniform_band metadata simulated_suprema_mean must match current bootstrap draws",
        ):
            replace(payload, optimization_metadata=stale_suprema_metadata)
    finally:
        inference_module.NonparametricInferencePayload._validate_optimality = False

    fractional_n_boot_metadata = dict(payload.optimization_metadata)
    fractional_n_boot_metadata["uniform_band"] = dict(
        payload.optimization_metadata["uniform_band"]
    )
    fractional_n_boot_metadata["uniform_band"]["n_boot"] = (
        float(payload.uniform_band.n_boot) + 0.5
    )
    with pytest.raises(
        ValueError,
        match="uniform_band metadata n_boot must be an integer",
    ):
        replace(payload, optimization_metadata=fractional_n_boot_metadata)

    fractional_random_state_metadata = dict(payload.optimization_metadata)
    fractional_random_state_metadata["uniform_band"] = dict(
        payload.optimization_metadata["uniform_band"]
    )
    fractional_random_state_metadata["uniform_band"]["random_state"] = (
        float(payload.uniform_band.random_state) + 0.5
    )
    with pytest.raises(
        ValueError,
        match="uniform_band metadata random_state must be an integer",
    ):
        replace(payload, optimization_metadata=fractional_random_state_metadata)

    stale_bar_f_at_z0 = np.asarray(payload.bar_f_at_z0, dtype=float).copy()
    stale_bar_f_at_z0[0] += 0.01
    pointwise_half_width = (
        np.asarray(payload.pointwise_confidence_interval.upper, dtype=float)
        - payload.bar_f_at_z0
    )
    uniform_half_width = (
        np.asarray(payload.uniform_band.upper, dtype=float) - payload.bar_f_at_z0
    )
    with pytest.raises(
        ValueError,
        match="bar_f_at_z0 must equal evaluation_basis @ bar_gamma_hat",
    ):
        replace(
            payload,
            bar_f_at_z0=stale_bar_f_at_z0,
            pointwise_confidence_interval=results_module.ConfidenceInterval(
                lower=stale_bar_f_at_z0 - pointwise_half_width,
                upper=stale_bar_f_at_z0 + pointwise_half_width,
                level=payload.pointwise_confidence_interval.level,
            ),
            uniform_band=results_module.UniformBand(
                lower=stale_bar_f_at_z0 - uniform_half_width,
                upper=stale_bar_f_at_z0 + uniform_half_width,
                level=payload.uniform_band.level,
                critical_value=payload.uniform_band.critical_value,
                n_boot=payload.uniform_band.n_boot,
                random_state=payload.uniform_band.random_state,
            ),
        )

    stale_bar_gamma_hat = np.asarray(payload.bar_gamma_hat, dtype=float).copy()
    stale_bar_gamma_hat[0] += 0.01
    with pytest.raises(
        ValueError,
        match="bar_gamma_hat must equal gamma_hat minus the sigma_f_hat linear solve of score_moment",
    ):
        replace(payload, bar_gamma_hat=stale_bar_gamma_hat)

    stale_score_moment = np.asarray(payload.score_moment, dtype=float).copy()
    stale_score_moment[0] += 0.01
    with pytest.raises(
        ValueError,
        match="bar_gamma_hat must equal gamma_hat minus the sigma_f_hat linear solve of score_moment",
    ):
        replace(payload, score_moment=stale_score_moment)

    with pytest.raises(
        ValueError,
        match="uniform_standardization must be strictly positive",
    ):
        replace(
            payload,
            uniform_standardization=np.zeros_like(payload.sigma_z_hat),
        )

    asymmetric_sigma_x_hat = np.asarray(payload.sigma_x_hat, dtype=float).copy()
    asymmetric_sigma_x_hat[0, 1] += 1e-4
    with pytest.raises(ValueError, match="sigma_x_hat must be symmetric"):
        replace(payload, sigma_x_hat=asymmetric_sigma_x_hat)

    indefinite_sigma_x_hat = np.asarray(payload.sigma_x_hat, dtype=float).copy()
    indefinite_sigma_x_hat[0, 0] = -1e-6
    with pytest.raises(ValueError, match="sigma_x_hat must be positive semidefinite"):
        replace(payload, sigma_x_hat=indefinite_sigma_x_hat)

    stale_m_hat = np.asarray(payload.m_hat, dtype=float).copy()
    stale_m_hat[0, 0] += 1e-4
    with pytest.raises(
        ValueError,
        match=r"m_hat must satisfy the Eq\. \(4\.3\) projection constraint",
    ):
        replace(payload, m_hat=stale_m_hat)

    fractional_solver_metadata = dict(payload.optimization_metadata)
    fractional_solver_metadata["max_threshold_steps"] = (
        float(payload.optimization_metadata["max_threshold_steps"]) + 0.5
    )
    with pytest.raises(ValueError, match="max_threshold_steps must be an integer"):
        replace(payload, optimization_metadata=fractional_solver_metadata)

    boolean_lambda_metadata = dict(payload.optimization_metadata)
    boolean_lambda_metadata["lambda_double_prime"] = True
    with pytest.raises(ValueError, match="lambda_double_prime must be numeric"):
        replace(payload, optimization_metadata=boolean_lambda_metadata)

    string_lambda_metadata = dict(payload.optimization_metadata)
    string_lambda_metadata["lambda_double_prime"] = "0.0"
    with pytest.raises(
        ValueError,
        match="lambda_double_prime must be numeric, not string",
    ):
        replace(payload, optimization_metadata=string_lambda_metadata)

    with pytest.raises(ValueError, match="basis_degree must be an integer"):
        replace(payload, basis_degree=float(payload.basis_degree) + 0.5)

    with pytest.raises(
        ValueError,
        match="orthogonal_basis_valid must align with basis dimension",
    ):
        replace(
            payload,
            orthogonal_basis_valid=payload.orthogonal_basis_valid[:, :-1],
        )

    with pytest.raises(
        ValueError,
        match="orthogonal_basis_valid must align with n_valid_obs",
    ):
        replace(
            payload,
            orthogonal_basis_valid=payload.orthogonal_basis_valid[:-1, :],
        )

    stale_orthogonal_basis_valid = np.asarray(
        payload.orthogonal_basis_valid,
        dtype=float,
    ).copy()
    stale_orthogonal_basis_valid[0, 0] += 1e-4
    with pytest.raises(
        ValueError,
        match=(
            "sigma_f_hat must equal the covariance implied by "
            "orthogonal_basis_valid"
        ),
    ):
        replace(payload, orthogonal_basis_valid=stale_orthogonal_basis_valid)

    asymmetric_omega_f_hat = np.asarray(payload.omega_f_hat, dtype=float).copy()
    asymmetric_omega_f_hat[0, 1] += 1e-4
    with pytest.raises(ValueError, match="omega_f_hat must be symmetric"):
        replace(payload, omega_f_hat=asymmetric_omega_f_hat)

    indefinite_omega_f_hat = np.asarray(payload.omega_f_hat, dtype=float).copy()
    indefinite_omega_f_hat[0, 0] = -1e-6
    with pytest.raises(ValueError, match="omega_f_hat must be positive semidefinite"):
        replace(payload, omega_f_hat=indefinite_omega_f_hat)

    asymmetric_v_f_hat = np.asarray(payload.v_f_hat, dtype=float).copy()
    asymmetric_v_f_hat[0, 1] += 1e-4
    with pytest.raises(ValueError, match="v_f_hat must be symmetric"):
        replace(payload, v_f_hat=asymmetric_v_f_hat)

    indefinite_v_f_hat = np.asarray(payload.v_f_hat, dtype=float).copy()
    indefinite_v_f_hat[0, 0] = -1e-6
    with pytest.raises(ValueError, match="v_f_hat must be positive semidefinite"):
        replace(payload, v_f_hat=indefinite_v_f_hat)

    stale_v_f_hat = np.asarray(payload.v_f_hat, dtype=float) + np.eye(
        payload.v_f_hat.shape[0],
        dtype=float,
    ) * 1e-6
    with pytest.raises(
        ValueError,
        match="v_f_hat must equal the sigma_f_hat linear-solve sandwich",
    ):
        replace(payload, v_f_hat=stale_v_f_hat)


def test_nonparametric_inference_payload_rejects_negative_covariance_grid_variance(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    payload, _ = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=eq43_nonparametric_reference_slice[
            "lambda_double_prime"
        ],
        n_boot=eq43_nonparametric_reference_slice["n_boot"],
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )
    covariance_at_grid = np.asarray(payload.covariance_at_grid, dtype=float).copy()
    covariance_at_grid[1, 1] = -1e-8

    with pytest.raises(
        ValueError,
        match="covariance_at_grid diagonal must be strictly positive",
    ):
        replace(payload, covariance_at_grid=covariance_at_grid)

    asymmetric_covariance_at_grid = np.asarray(
        payload.covariance_at_grid, dtype=float
    ).copy()
    asymmetric_covariance_at_grid[0, 1] += 1e-4
    with pytest.raises(ValueError, match="covariance_at_grid must be symmetric"):
        replace(payload, covariance_at_grid=asymmetric_covariance_at_grid)

    stale_covariance_at_grid = np.asarray(payload.covariance_at_grid, dtype=float).copy()
    stale_covariance_at_grid[0, 1] += 1e-6
    stale_covariance_at_grid[1, 0] += 1e-6
    with pytest.raises(
        ValueError,
        match=(
            "covariance_at_grid must equal "
            r"evaluation_basis @ v_f_hat @ evaluation_basis\.T / n_valid"
        ),
    ):
        replace(payload, covariance_at_grid=stale_covariance_at_grid)

    metadata_without_n_valid = dict(payload.optimization_metadata)
    metadata_without_n_valid.pop("n_valid_obs")
    with pytest.raises(
        ValueError,
        match="n_valid_obs must be provided to validate",
    ):
        replace(
            payload,
            covariance_at_grid=stale_covariance_at_grid,
            optimization_metadata=metadata_without_n_valid,
        )

    metadata_with_fractional_n_valid = dict(payload.optimization_metadata)
    metadata_with_fractional_n_valid["n_valid_obs"] = 5.5
    with pytest.raises(ValueError, match="n_valid_obs must be an integer"):
        replace(payload, optimization_metadata=metadata_with_fractional_n_valid)

    indefinite_covariance_at_grid = np.asarray(
        payload.covariance_at_grid, dtype=float
    ).copy()
    off_diagonal = (
        np.sqrt(
            indefinite_covariance_at_grid[0, 0] * indefinite_covariance_at_grid[1, 1]
        )
        * 1.1
    )
    indefinite_covariance_at_grid[0, 1] = off_diagonal
    indefinite_covariance_at_grid[1, 0] = off_diagonal
    with pytest.raises(
        ValueError,
        match="covariance_at_grid must be positive semidefinite",
    ):
        replace(payload, covariance_at_grid=indefinite_covariance_at_grid)


def test_nonparametric_inference_metadata_reports_covariance_grid_psd_boundary(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    payload, updated_result = inference_module.estimate_nonparametric_inference(
        eq43_nonparametric_reference_slice["score_payload"],
        eq43_nonparametric_reference_slice["estimation_payload"],
        alpha=eq43_nonparametric_reference_slice["alpha"],
        lambda_double_prime=0.01,
        n_boot=32,
        random_state=eq43_nonparametric_reference_slice["random_state"],
    )

    expected_min_eigenvalue = float(
        np.min(
            np.linalg.eigvalsh(
                0.5 * (payload.covariance_at_grid + payload.covariance_at_grid.T)
            )
        )
    )
    assert payload.optimization_metadata["covariance_at_grid_min_eigenvalue"] == (
        pytest.approx(expected_min_eigenvalue)
    )
    assert expected_min_eigenvalue >= -1e-10
    assert updated_result.diagnostics is not None
    assert updated_result.diagnostics.optimization_metadata[
        "covariance_at_grid_min_eigenvalue"
    ] == pytest.approx(expected_min_eigenvalue)
