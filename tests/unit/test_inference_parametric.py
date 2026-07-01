from __future__ import annotations

from dataclasses import replace
import importlib
from statistics import NormalDist

import numpy as np
import pytest


class _StringArrayLike:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=object)

    def __array__(self, dtype: object = None) -> np.ndarray:
        if dtype is None:
            return self._values
        return self._values.astype(dtype)


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


def test_solve_eq42_sparse_direction_matches_small_exact_slice(
    eq41_parametric_reference_slice: dict[str, np.ndarray | float],
) -> None:
    inference_module = _load_inference_module()
    assert hasattr(inference_module, "solve_eq42_sparse_direction")

    w_hat, metadata = inference_module.solve_eq42_sparse_direction(
        sigma_tilde_x_hat=eq41_parametric_reference_slice["sigma_tilde_x_hat"],
        xi=eq41_parametric_reference_slice["xi_matrix"],
        lambda_prime=0.0,
    )

    np.testing.assert_allclose(
        w_hat,
        eq41_parametric_reference_slice["w_matrix"],
        atol=1e-12,
        rtol=0.0,
    )
    assert metadata["lambda_prime"] == pytest.approx(0.0)
    assert metadata["xi_count"] == 3
    assert metadata["sigma_tilde_x_dimension"] == 2
    assert metadata["constraint_violation_max"] == pytest.approx(0.0, abs=1e-12)
    assert metadata["feasible"] is True


def test_solve_eq42_sparse_direction_minimizes_l1_objective() -> None:
    inference_module = _load_inference_module()

    sigma_tilde_x_hat = np.array(
        [
            [4.20108001, 1.17105044],
            [1.17105044, 1.86883287],
        ],
        dtype=float,
    )
    xi = np.array([[1.74106572, 0.38469014]], dtype=float)

    w_hat, metadata = inference_module.solve_eq42_sparse_direction(
        sigma_tilde_x_hat=sigma_tilde_x_hat,
        xi=xi,
        lambda_prime=0.3888502756577331,
    )

    np.testing.assert_allclose(
        w_hat,
        np.array([[-0.321873289993, 0.0]], dtype=float),
        atol=1e-10,
        rtol=0.0,
    )
    assert np.linalg.norm(w_hat[0], ord=1) == pytest.approx(
        0.321873289993,
        abs=1e-10,
    )
    assert metadata["solver"] == "eq42-linear-programming-l1"
    assert metadata["feasible"] is True
    assert metadata["constraint_violation_max"] == pytest.approx(0.0, abs=1e-10)


def test_parametric_payload_accepts_nonunique_eq42_l1_optimum() -> None:
    inference_module = _load_inference_module()
    results_module = _load_results_module()

    sigma_tilde_x_hat = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    xi = np.array([[1.0, 1.0]], dtype=float)
    w_hat = np.array([[-0.5, -0.5]], dtype=float)
    beta_hat = np.zeros(2, dtype=float)
    score_moment = np.zeros(2, dtype=float)
    omega_beta_hat = np.eye(2, dtype=float)
    t_hat = xi @ beta_hat - np.einsum("ij,j->i", w_hat, score_moment)
    asymptotic_variance_hat = np.einsum(
        "ij,jk,ik->i",
        w_hat,
        omega_beta_hat,
        w_hat,
    )
    n_valid = 4
    standard_errors = np.sqrt(asymptotic_variance_hat / n_valid)
    alpha = 0.05
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    payload = inference_module.ParametricInferencePayload(
        w_hat=w_hat,
        sigma_tilde_x_hat=sigma_tilde_x_hat,
        omega_beta_hat=omega_beta_hat,
        score_moment=score_moment,
        xi=xi,
        t_hat=t_hat,
        asymptotic_variance_hat=asymptotic_variance_hat,
        standard_errors=standard_errors,
        confidence_interval=results_module.ConfidenceInterval(
            lower=t_hat - z_critical * standard_errors,
            upper=t_hat + z_critical * standard_errors,
            level=1.0 - alpha,
        ),
        alpha=alpha,
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        optimization_metadata={
            "beta_hat": beta_hat,
            "lambda_prime": 0.0,
            "max_threshold_steps": 25,
            "n_valid_obs": n_valid,
        },
    )

    solver_w_hat, solver_metadata = inference_module.solve_eq42_sparse_direction(
        sigma_tilde_x_hat=sigma_tilde_x_hat,
        xi=xi,
        lambda_prime=0.0,
    )
    assert solver_metadata["feasible"] is True
    assert np.linalg.norm(payload.w_hat[0], ord=1) == pytest.approx(
        np.linalg.norm(solver_w_hat[0], ord=1),
        abs=1e-12,
    )
    np.testing.assert_allclose(
        xi[0] + payload.sigma_tilde_x_hat @ payload.w_hat[0],
        np.zeros(2, dtype=float),
        atol=1e-12,
        rtol=0.0,
    )


def test_parametric_payload_rejects_feasible_but_nonoptimal_eq42_direction() -> None:
    inference_module = _load_inference_module()
    results_module = _load_results_module()

    sigma_tilde_x_hat = np.eye(2, dtype=float)
    xi = np.zeros((1, 2), dtype=float)
    w_hat = np.array([[0.1, 0.0]], dtype=float)
    beta_hat = np.zeros(2, dtype=float)
    score_moment = np.zeros(2, dtype=float)
    omega_beta_hat = np.eye(2, dtype=float)
    t_hat = xi @ beta_hat - np.einsum("ij,j->i", w_hat, score_moment)
    asymptotic_variance_hat = np.einsum(
        "ij,jk,ik->i",
        w_hat,
        omega_beta_hat,
        w_hat,
    )
    n_valid = 4
    standard_errors = np.sqrt(asymptotic_variance_hat / n_valid)
    alpha = 0.05
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    inference_module.ParametricInferencePayload._validate_optimality = True
    try:
        with pytest.raises(
            ValueError,
            match=r"w_hat must attain the Eq\. \(4\.2\) sparse direction L1 optimum",
        ):
            inference_module.ParametricInferencePayload(
            w_hat=w_hat,
            sigma_tilde_x_hat=sigma_tilde_x_hat,
            omega_beta_hat=omega_beta_hat,
            score_moment=score_moment,
            xi=xi,
            t_hat=t_hat,
            asymptotic_variance_hat=asymptotic_variance_hat,
            standard_errors=standard_errors,
            confidence_interval=results_module.ConfidenceInterval(
                lower=t_hat - z_critical * standard_errors,
                upper=t_hat + z_critical * standard_errors,
                level=1.0 - alpha,
            ),
            alpha=alpha,
            basis_family="polynomial",
            basis_degree=1,
            oracle_lane="r-parity-polynomial",
            optimization_metadata={
                "beta_hat": beta_hat,
                "lambda_prime": 0.2,
                "max_threshold_steps": 25,
                "n_valid_obs": n_valid,
            },
            )
    finally:
        inference_module.ParametricInferencePayload._validate_optimality = False


def test_solve_eq42_sparse_direction_rejects_boolean_xi_targets() -> None:
    inference_module = _load_inference_module()

    with pytest.raises(ValueError, match="xi must be numeric, not boolean"):
        inference_module.solve_eq42_sparse_direction(
            sigma_tilde_x_hat=np.eye(2, dtype=float),
            xi=np.array([[True, False]]),
            lambda_prime=0.0,
        )


def test_solve_eq42_sparse_direction_rejects_string_numeric_aliases() -> None:
    inference_module = _load_inference_module()

    with pytest.raises(ValueError, match="xi must be numeric, not string"):
        inference_module.solve_eq42_sparse_direction(
            sigma_tilde_x_hat=np.eye(2, dtype=float),
            xi=np.array([["1.0", "0.0"]]),
            lambda_prime=0.0,
        )

    with pytest.raises(ValueError, match="xi must be numeric, not string"):
        inference_module.solve_eq42_sparse_direction(
            sigma_tilde_x_hat=np.eye(2, dtype=float),
            xi=np.array([[b"1.0", b"0.0"]]),
            lambda_prime=0.0,
        )

    with pytest.raises(ValueError, match="xi must be numeric, not string"):
        inference_module.solve_eq42_sparse_direction(
            sigma_tilde_x_hat=np.eye(2, dtype=float),
            xi=_StringArrayLike(np.array([["1.0", "0.0"]])),
            lambda_prime=0.0,
        )

    with pytest.raises(ValueError, match="lambda_prime must be numeric, not string"):
        inference_module.solve_eq42_sparse_direction(
            sigma_tilde_x_hat=np.eye(2, dtype=float),
            xi=np.array([[1.0, 0.0]]),
            lambda_prime="0.0",
        )

    with pytest.raises(ValueError, match="lambda_prime must be numeric, not string"):
        inference_module.solve_eq42_sparse_direction(
            sigma_tilde_x_hat=np.eye(2, dtype=float),
            xi=np.array([[1.0, 0.0]]),
            lambda_prime=b"0.0",
        )

    with pytest.raises(
        ValueError,
        match="lambda_prime must be numeric, not string",
    ):
        inference_module.solve_eq42_sparse_direction(
            sigma_tilde_x_hat=np.eye(2, dtype=float),
            xi=np.array([[1.0, 0.0]]),
            lambda_prime=np.array("0.0", dtype=object),
        )


def test_estimate_parametric_inference_accepts_explicit_xi_and_preserves_oracle_lane(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    inference_module = _load_inference_module()
    assert hasattr(inference_module, "ParametricInferencePayload")
    assert hasattr(inference_module, "estimate_parametric_inference")

    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    estimation_payload, result = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.2,
    )
    xi = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=float)

    payload, updated_result = inference_module.estimate_parametric_inference(
        score_payload,
        estimation_payload,
        result=result,
        xi=xi,
        alpha=eq31_manual_slice["data"].alpha,
        lambda_prime=0.0,
    )

    np.testing.assert_allclose(payload.xi, xi, atol=0.0, rtol=0.0)
    expected_t_hat = xi @ estimation_payload.beta_hat - np.einsum(
        "ij,j->i", payload.w_hat, payload.score_moment
    )
    np.testing.assert_allclose(payload.t_hat, expected_t_hat, atol=1e-10, rtol=0.0)
    assert np.all(payload.standard_errors > 0.0)
    assert payload.t_hat.shape == (2,)
    assert payload.oracle_lane == "r-parity-polynomial"
    assert payload.optimization_metadata["xi_count"] == 2
    assert updated_result.diagnostics is not None
    assert updated_result.diagnostics.oracle_lane == "r-parity-polynomial"
    assert updated_result.parametric_estimates["t_hat"].shape == (2,)
    markdown = updated_result.to_markdown(style="legacy", missing_value="")
    for index, (beta_hat, t_hat, standard_error, lower, upper) in enumerate(
        zip(
            estimation_payload.beta_hat,
            payload.t_hat,
            payload.standard_errors,
            payload.confidence_interval.lower,
            payload.confidence_interval.upper,
            strict=True,
        )
    ):
        assert (
            f"| Parametric | beta_hat | {index} | {beta_hat:.4f} |  |  |"
            in markdown
        )
        assert (
            f"| Parametric | t_hat | {index} | {t_hat:.4f} | "
            f"{standard_error:.4f} | [{lower:.4f}, {upper:.4f}] "
            f"({(1.0 - payload.alpha) * 100.0:.1f}%) |"
            in markdown
        )


def test_estimate_parametric_inference_rejects_degenerate_zero_variance(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    inference_module = _load_inference_module()

    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    estimation_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
    )
    xi = np.array([1.0, -0.5], dtype=float)

    with pytest.raises(inference_module.NonpositiveVarianceError) as excinfo:
        inference_module.estimate_parametric_inference(
            score_payload,
            estimation_payload,
            xi=xi,
            alpha=eq31_manual_slice["data"].alpha,
            lambda_prime=0.0,
        )

    assert excinfo.value.metadata["target_kind"] == "parametric"
    assert excinfo.value.metadata["failure_kind"] == "nonpositive-variance"
    assert excinfo.value.metadata["matrix_name"] == "asymptotic_variance_hat"
    assert excinfo.value.metadata["minimum_value"] == pytest.approx(0.0)
    assert excinfo.value.metadata["nonpositive_entry_count"] == 1


def test_estimate_parametric_inference_rejects_alpha_string_alias(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    inference_module = _load_inference_module()

    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    estimation_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.2,
    )

    with pytest.raises(ValueError, match="alpha must be numeric, not string"):
        inference_module.estimate_parametric_inference(
            score_payload,
            estimation_payload,
            alpha="0.05",
            lambda_prime=0.0,
        )

    with pytest.raises(ValueError, match="alpha must be numeric, not string"):
        inference_module.estimate_parametric_inference(
            score_payload,
            estimation_payload,
            alpha=np.array("0.05", dtype=object),
            lambda_prime=0.0,
        )

    with pytest.raises(ValueError, match="alpha must be numeric"):
        inference_module.estimate_parametric_inference(
            score_payload,
            estimation_payload,
            alpha=True,
            lambda_prime=0.0,
        )

    for alpha in ([0.05], np.array([0.05])):
        with pytest.raises(ValueError, match="alpha must be a scalar"):
            inference_module.estimate_parametric_inference(
                score_payload,
                estimation_payload,
                alpha=alpha,
                lambda_prime=0.0,
            )


def test_estimate_parametric_inference_rejects_empty_default_target() -> None:
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

    with pytest.raises(inference_module.InvalidInferenceInputError) as excinfo:
        inference_module.estimate_parametric_inference(
            score_payload,
            estimation_payload,
        )

    assert excinfo.value.metadata["target_kind"] == "parametric"
    assert excinfo.value.metadata["failure_kind"] == "missing-parametric-target"
    assert excinfo.value.metadata["beta_dimension"] == 0


def test_estimate_parametric_inference_allows_high_dimensional_singular_sigma_tilde_x() -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    inference_module = _load_inference_module()

    x_valid = np.array(
        [
            [1.5, -0.5, 0.2, 1.0, -1.0, 0.3],
            [-0.5, 1.5, -0.2, -1.0, 1.0, -0.3],
            [0.5, -1.5, 0.4, -0.5, 0.5, 0.7],
            [-1.5, 0.5, -0.4, 0.5, -0.5, -0.7],
        ],
        dtype=float,
    )
    basis_valid_full = np.ones((x_valid.shape[0], 1), dtype=float)
    residual_valid = np.array([0.4, -0.6, 0.8, -1.0], dtype=float)
    score_payload = score_module.ScorePayload(
        delta_y=residual_valid,
        s_hat=residual_valid,
        s_hat_valid=residual_valid,
        valid_mask=np.ones(x_valid.shape[0], dtype=bool),
        fold_ids=np.ones(x_valid.shape[0], dtype=int),
        fold_diagnostics=[],
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        basis_matrix=basis_valid_full,
        x_valid=x_valid,
        basis_valid_full=basis_valid_full,
        basis_design_valid=np.empty((x_valid.shape[0], 0), dtype=float),
        evaluation_basis=np.ones((1, 1), dtype=float),
        pi_hat=np.full(x_valid.shape[0], 0.5, dtype=float),
        phi0_hat=np.zeros(x_valid.shape[0], dtype=float),
        phi1_hat=np.zeros(x_valid.shape[0], dtype=float),
        rho_hat=np.ones(x_valid.shape[0], dtype=float),
        intercept_dropped_for_design=True,
    )
    estimation_payload = estimation_module.EstimationPayload(
        beta_hat=np.zeros(x_valid.shape[1], dtype=float),
        gamma_hat=np.zeros(1, dtype=float),
        fitted_f_valid=np.zeros(x_valid.shape[0], dtype=float),
        second_stage_prediction_valid=np.zeros(x_valid.shape[0], dtype=float),
        residual_valid=residual_valid,
        projection_x_valid=x_valid,
        f_hat_at_z0=np.zeros(1, dtype=float),
        optimization_metadata={
            "solver": "manual-high-dimensional-parametric-slice",
            "n_valid_obs": int(x_valid.shape[0]),
            "beta_dimension": int(x_valid.shape[1]),
            "basis_dimension_full": 1,
        },
    )
    sigma_tilde_x_hat = x_valid.T @ x_valid / x_valid.shape[0]
    xi = -sigma_tilde_x_hat[:, [0]].T

    payload, updated_result = inference_module.estimate_parametric_inference(
        score_payload,
        estimation_payload,
        xi=xi,
        alpha=0.1,
        lambda_prime=0.0,
    )

    assert np.linalg.matrix_rank(payload.sigma_tilde_x_hat) < x_valid.shape[1]
    assert payload.optimization_metadata["sigma_tilde_x_rank"] == int(
        np.linalg.matrix_rank(payload.sigma_tilde_x_hat)
    )
    assert payload.optimization_metadata["constraint_violation_max"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert np.all(payload.standard_errors > 0.0)
    assert updated_result.standard_errors["parametric_se"].shape == (1,)


def test_estimate_parametric_inference_fills_beta_se_ci_and_diagnostics(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    inference_module = _load_inference_module()

    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    estimation_payload, result = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.2,
    )

    payload, updated_result = inference_module.estimate_parametric_inference(
        score_payload,
        estimation_payload,
        result=result,
        alpha=eq31_manual_slice["data"].alpha,
        lambda_prime=0.0,
    )

    beta_se = updated_result.standard_errors["beta_se"]
    parametric_ci = updated_result.intervals["parametric_ci"]

    assert payload.optimization_metadata["lambda_prime"] == pytest.approx(0.0)
    assert (
        payload.optimization_metadata["sigma_tilde_x_min_eigenvalue"]
        == pytest.approx(
            float(np.min(np.linalg.eigvalsh(payload.sigma_tilde_x_hat)))
        )
    )
    assert (
        payload.optimization_metadata["omega_beta_min_eigenvalue"]
        == pytest.approx(float(np.min(np.linalg.eigvalsh(payload.omega_beta_hat))))
    )
    assert payload.optimization_metadata["sigma_tilde_x_min_eigenvalue"] > 0.0
    assert payload.optimization_metadata["omega_beta_min_eigenvalue"] >= -1e-10
    assert (
        payload.optimization_metadata["xi_count"]
        == estimation_payload.beta_hat.shape[0]
    )
    assert (
        payload.optimization_metadata["sigma_tilde_x_dimension"]
        == estimation_payload.beta_hat.shape[0]
    )
    assert beta_se.shape == estimation_payload.beta_hat.shape
    assert parametric_ci.lower.shape == estimation_payload.beta_hat.shape
    assert parametric_ci.upper.shape == estimation_payload.beta_hat.shape
    assert np.all(parametric_ci.lower <= parametric_ci.upper)
    assert updated_result.diagnostics is not None
    assert updated_result.diagnostics.optimization_metadata[
        "lambda_prime"
    ] == pytest.approx(0.0)
    assert updated_result.diagnostics.optimization_metadata[
        "sigma_tilde_x_min_eigenvalue"
    ] == pytest.approx(payload.optimization_metadata["sigma_tilde_x_min_eigenvalue"])
    assert updated_result.diagnostics.optimization_metadata[
        "omega_beta_min_eigenvalue"
    ] == pytest.approx(payload.optimization_metadata["omega_beta_min_eigenvalue"])
    assert (
        updated_result.diagnostics.optimization_metadata["xi_count"]
        == estimation_payload.beta_hat.shape[0]
    )


def test_estimate_parametric_inference_revalidates_mutable_result_input(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    inference_module = _load_inference_module()

    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    estimation_payload, result = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.2,
    )
    result.parametric_estimates[1] = np.array([1.0])

    with pytest.raises(ValueError, match="parametric_estimates keys must be strings"):
        inference_module.estimate_parametric_inference(
            score_payload,
            estimation_payload,
            result=result,
            alpha=eq31_manual_slice["data"].alpha,
            lambda_prime=0.0,
        )


def test_parametric_inference_payload_rejects_nonfinite_direct_outputs(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    inference_module = _load_inference_module()
    results_module = _load_results_module()

    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    estimation_payload, result = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.2,
    )
    payload, _ = inference_module.estimate_parametric_inference(
        score_payload,
        estimation_payload,
        result=result,
        alpha=eq31_manual_slice["data"].alpha,
        lambda_prime=0.0,
    )

    with pytest.raises(ValueError, match="t_hat must contain only finite values"):
        replace(payload, t_hat=np.full_like(payload.t_hat, np.nan))

    z_critical = inference_module.NormalDist().inv_cdf(1.0 - payload.alpha / 2.0)
    forged_t_hat = np.asarray(payload.t_hat, dtype=float) + 10.0
    with pytest.raises(
        ValueError,
        match="t_hat must equal xi @ beta_hat - w_hat @ score_moment",
    ):
        replace(
            payload,
            t_hat=forged_t_hat,
            confidence_interval=results_module.ConfidenceInterval(
                lower=forged_t_hat - z_critical * payload.standard_errors,
                upper=forged_t_hat + z_critical * payload.standard_errors,
                level=payload.confidence_interval.level,
            ),
        )

    missing_beta_metadata = dict(payload.optimization_metadata)
    missing_beta_metadata.pop("beta_hat", None)
    with pytest.raises(
        ValueError,
        match="beta_hat metadata must be provided to validate t_hat",
    ):
        replace(payload, optimization_metadata=missing_beta_metadata)

    stale_beta_metadata = dict(payload.optimization_metadata)
    stale_beta_metadata["beta_hat"] = np.asarray(
        payload.optimization_metadata["beta_hat"],
        dtype=float,
    ) + 1.0
    with pytest.raises(
        ValueError,
        match="t_hat must equal xi @ beta_hat - w_hat @ score_moment",
    ):
        replace(payload, optimization_metadata=stale_beta_metadata)

    with pytest.raises(ValueError, match="alpha must be numeric, not string"):
        replace(payload, alpha="0.05")

    with pytest.raises(ValueError, match="alpha must be numeric, not string"):
        replace(payload, alpha=np.array("0.05", dtype=object))

    with pytest.raises(ValueError, match="alpha must be numeric"):
        replace(payload, alpha=False)

    stale_w_hat = np.asarray(payload.w_hat, dtype=float).copy()
    stale_w_hat[0, 0] += 1e-4
    stale_w_t_hat = payload.xi @ payload.optimization_metadata[
        "beta_hat"
    ] - np.einsum("ij,j->i", stale_w_hat, payload.score_moment)
    with pytest.raises(
        ValueError,
        match=r"w_hat must satisfy the Eq\. \(4\.2\) sparse direction constraint",
    ):
        replace(
            payload,
            w_hat=stale_w_hat,
            t_hat=stale_w_t_hat,
            confidence_interval=results_module.ConfidenceInterval(
                lower=stale_w_t_hat - z_critical * payload.standard_errors,
                upper=stale_w_t_hat + z_critical * payload.standard_errors,
                level=payload.confidence_interval.level,
            ),
        )

    fractional_solver_metadata = dict(payload.optimization_metadata)
    fractional_solver_metadata["max_threshold_steps"] = (
        float(payload.optimization_metadata["max_threshold_steps"]) + 0.5
    )
    with pytest.raises(ValueError, match="max_threshold_steps must be an integer"):
        replace(payload, optimization_metadata=fractional_solver_metadata)

    boolean_lambda_metadata = dict(payload.optimization_metadata)
    boolean_lambda_metadata["lambda_prime"] = False
    with pytest.raises(ValueError, match="lambda_prime must be numeric"):
        replace(payload, optimization_metadata=boolean_lambda_metadata)

    string_lambda_metadata = dict(payload.optimization_metadata)
    string_lambda_metadata["lambda_prime"] = "0.0"
    with pytest.raises(ValueError, match="lambda_prime must be numeric, not string"):
        replace(payload, optimization_metadata=string_lambda_metadata)

    with pytest.raises(ValueError, match="basis_degree must be an integer"):
        replace(payload, basis_degree=float(payload.basis_degree) + 0.5)

    with pytest.raises(
        ValueError,
        match="xi must contain at least one parametric target",
    ):
        inference_module.ParametricInferencePayload(
            xi=np.empty((0, 0), dtype=float),
            t_hat=np.empty(0, dtype=float),
            w_hat=np.empty((0, 0), dtype=float),
            sigma_tilde_x_hat=np.empty((0, 0), dtype=float),
            omega_beta_hat=np.empty((0, 0), dtype=float),
            score_moment=np.empty(0, dtype=float),
            asymptotic_variance_hat=np.empty(0, dtype=float),
            standard_errors=np.empty(0, dtype=float),
            confidence_interval=results_module.ConfidenceInterval(
                lower=np.empty(0, dtype=float),
                upper=np.empty(0, dtype=float),
                level=1.0 - payload.alpha,
            ),
            alpha=payload.alpha,
            basis_family=payload.basis_family,
            basis_degree=payload.basis_degree,
            oracle_lane=payload.oracle_lane,
            optimization_metadata={
                "n_valid_obs": payload.optimization_metadata["n_valid_obs"],
                "lambda_prime": payload.optimization_metadata["lambda_prime"],
                "max_threshold_steps": payload.optimization_metadata[
                    "max_threshold_steps"
                ],
            },
        )

    stale_asymptotic_variance = payload.asymptotic_variance_hat * 1.1
    stale_standard_errors = np.sqrt(
        stale_asymptotic_variance / payload.optimization_metadata["n_valid_obs"]
    )
    with pytest.raises(
        ValueError,
        match="asymptotic_variance_hat must equal w_hat @ omega_beta_hat @ w_hat",
    ):
        replace(
            payload,
            asymptotic_variance_hat=stale_asymptotic_variance,
            standard_errors=stale_standard_errors,
            confidence_interval=results_module.ConfidenceInterval(
                lower=payload.t_hat - z_critical * stale_standard_errors,
                upper=payload.t_hat + z_critical * stale_standard_errors,
                level=payload.confidence_interval.level,
            ),
        )

    with pytest.raises(
        ValueError,
        match="confidence_interval.lower must contain only finite values",
    ):
        replace(
            payload,
            confidence_interval=results_module.ConfidenceInterval(
                lower=np.full_like(payload.t_hat, np.nan),
                upper=np.asarray(payload.confidence_interval.upper, dtype=float),
                level=payload.confidence_interval.level,
            ),
        )
