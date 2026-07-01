from __future__ import annotations

from dataclasses import replace
import importlib
from types import SimpleNamespace

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


def _load_validation_module():
    try:
        return importlib.import_module("hddid.validation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"hddid.validation is missing: {exc}")


def _build_phase7_nonparametric_failure_slice():
    validation_module = _load_validation_module()
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    nuisance_module = importlib.import_module("hddid.nuisance")
    splitting_module = importlib.import_module("hddid.splitting")

    replication_seed = 460490113
    design = validation_module.MonteCarloDesign(
        dgp_name="DGP2",
        n_obs=200,
        p=10,
        basis_family="trigonometric",
        basis_degree=8,
        oracle_lane="paper-trigonometric",
    )
    dataset = validation_module.generate_paper_dgp2(
        design,
        random_state=replication_seed,
    )
    data = dataset.to_validated_data()
    splits = splitting_module.make_crossfit_splits(
        n_obs=data.n_obs,
        n_folds=design.n_folds,
        random_state=replication_seed,
        trim_lower=design.trim_lower,
        trim_upper=design.trim_upper,
    )
    nuisance_payload = nuisance_module.CrossfitNuisanceEstimator(
        oracle_lane=design.oracle_lane,
    ).fit(data, splits)
    score_payload = score_module.build_score_payload(data, nuisance_payload)
    estimation_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
    )
    return score_payload, estimation_payload, design, replication_seed


def _build_phase7_dgp2_seed202_blocker_slice():
    validation_module = _load_validation_module()
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    nuisance_module = importlib.import_module("hddid.nuisance")
    splitting_module = importlib.import_module("hddid.splitting")

    replication_seed = 202
    design = validation_module.MonteCarloDesign(
        dgp_name="DGP2",
        n_obs=500,
        p=50,
        basis_family="trigonometric",
        basis_degree=8,
        oracle_lane="paper-trigonometric",
        evaluation_grid=np.array([0.05, 0.15, 0.25], dtype=float),
    )
    dataset = validation_module.generate_paper_dgp2(
        design,
        random_state=replication_seed,
    )
    data = dataset.to_validated_data()
    splits = splitting_module.make_crossfit_splits(
        n_obs=data.n_obs,
        n_folds=design.n_folds,
        random_state=replication_seed,
        trim_lower=design.trim_lower,
        trim_upper=design.trim_upper,
    )
    nuisance_payload = nuisance_module.CrossfitNuisanceEstimator(
        oracle_lane=design.oracle_lane,
    ).fit(data, splits)
    score_payload = score_module.build_score_payload(data, nuisance_payload)
    estimation_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
    )
    return score_payload, estimation_payload, design, dataset, replication_seed


def _build_parametric_payload(eq31_manual_slice: dict[str, object]):
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()

    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    estimation_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
    )
    return score_payload, estimation_payload


def test_estimate_nonparametric_inference_rejects_mismatched_payload_pair_before_math(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    score_payload = eq43_nonparametric_reference_slice["score_payload"]
    reference_payload = eq43_nonparametric_reference_slice["estimation_payload"]
    estimation_payload = SimpleNamespace(
        beta_hat=np.asarray(reference_payload.beta_hat, dtype=float),
        gamma_hat=np.asarray(reference_payload.gamma_hat, dtype=float),
        fitted_f_valid=np.asarray(reference_payload.fitted_f_valid, dtype=float),
        second_stage_prediction_valid=np.asarray(
            reference_payload.second_stage_prediction_valid,
            dtype=float,
        ),
        residual_valid=np.asarray(reference_payload.residual_valid, dtype=float)[:-1],
        projection_x_valid=np.asarray(
            reference_payload.projection_x_valid, dtype=float
        ),
        f_hat_at_z0=np.asarray(reference_payload.f_hat_at_z0, dtype=float),
        optimization_metadata=dict(reference_payload.optimization_metadata),
    )

    with pytest.raises(inference_module.InvalidInferenceInputError) as excinfo:
        inference_module.estimate_nonparametric_inference(
            score_payload,
            estimation_payload,
            alpha=eq43_nonparametric_reference_slice["alpha"],
            lambda_double_prime=eq43_nonparametric_reference_slice[
                "lambda_double_prime"
            ],
            n_boot=eq43_nonparametric_reference_slice["n_boot"],
            random_state=eq43_nonparametric_reference_slice["random_state"],
        )

    assert excinfo.value.metadata["target_kind"] == "nonparametric"
    assert excinfo.value.metadata["failure_kind"] == "invalid-input"
    assert (
        excinfo.value.metadata["n_valid_obs"]
        == eq43_nonparametric_reference_slice["n_valid"]
    )


def test_estimate_nonparametric_inference_rejects_same_shape_stale_estimation_payload(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()
    estimation_payload = eq43_nonparametric_reference_slice["estimation_payload"]
    stale_payload = replace(
        estimation_payload,
        fitted_f_valid=np.asarray(estimation_payload.fitted_f_valid, dtype=float)
        + 0.125,
    )

    with pytest.raises(inference_module.InvalidInferenceInputError) as excinfo:
        inference_module.estimate_nonparametric_inference(
            eq43_nonparametric_reference_slice["score_payload"],
            stale_payload,
            alpha=eq43_nonparametric_reference_slice["alpha"],
            lambda_double_prime=eq43_nonparametric_reference_slice[
                "lambda_double_prime"
            ],
            n_boot=eq43_nonparametric_reference_slice["n_boot"],
            random_state=eq43_nonparametric_reference_slice["random_state"],
        )

    assert excinfo.value.metadata["target_kind"] == "nonparametric"
    assert excinfo.value.metadata["failure_kind"] == "payload-score-mismatch"
    assert excinfo.value.metadata["field_name"] == "fitted_f_valid"
    assert excinfo.value.metadata["max_abs_diff"] == pytest.approx(0.125)


def test_estimate_parametric_inference_rejects_unidentified_singular_target_at_eq31(
    eq31_manual_slice: dict[str, object],
) -> None:
    estimation_module = _load_estimation_module()
    score_payload, _ = _build_parametric_payload(eq31_manual_slice)
    singular_x_valid = np.column_stack(
        [score_payload.x_valid[:, 0], score_payload.x_valid[:, 0]]
    )
    singular_score_payload = replace(
        score_payload,
        x_valid=singular_x_valid,
    )

    with pytest.raises(
        estimation_module.Eq31ProjectionRankError,
        match=r"Eq\. \(3\.1\) unpenalized beta block requires full column rank",
    ) as excinfo:
        estimation_module.estimate_eq31_mainline(
            singular_score_payload,
            penalty_lambda=0.0,
        )

    assert excinfo.value.metadata["matrix_name"] == "projection_x_valid"
    assert excinfo.value.metadata["matrix_shape"] == (5, 2)
    assert excinfo.value.metadata["matrix_rank"] == 1
    assert excinfo.value.metadata["required_rank"] == 2


def test_estimate_parametric_inference_rejects_same_shape_stale_estimation_payload(
    eq31_manual_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()
    score_payload, estimation_payload = _build_parametric_payload(eq31_manual_slice)
    stale_payload = replace(
        estimation_payload,
        residual_valid=np.asarray(estimation_payload.residual_valid, dtype=float)
        + 0.25,
    )

    with pytest.raises(inference_module.InvalidInferenceInputError) as excinfo:
        inference_module.estimate_parametric_inference(
            score_payload,
            stale_payload,
            alpha=eq31_manual_slice["data"].alpha,
            lambda_prime=0.0,
        )

    assert excinfo.value.metadata["target_kind"] == "parametric"
    assert excinfo.value.metadata["failure_kind"] == "payload-score-mismatch"
    assert excinfo.value.metadata["field_name"] == "residual_valid"
    assert excinfo.value.metadata["max_abs_diff"] == pytest.approx(0.25)


def test_estimate_parametric_inference_rejects_empty_valid_sample_before_matrix_math() -> None:
    inference_module = _load_inference_module()
    score_payload = SimpleNamespace(
        oracle_lane="r-parity-polynomial",
        basis_family="polynomial",
        basis_degree=0,
    )
    estimation_payload = SimpleNamespace(
        projection_x_valid=np.empty((0, 1), dtype=float),
        residual_valid=np.empty(0, dtype=float),
        beta_hat=np.zeros(1, dtype=float),
        fitted_f_valid=np.empty(0, dtype=float),
        second_stage_prediction_valid=np.empty(0, dtype=float),
        optimization_metadata={"n_valid_obs": 0},
    )

    with pytest.raises(inference_module.InvalidInferenceInputError) as excinfo:
        inference_module.estimate_parametric_inference(
            score_payload,
            estimation_payload,
            alpha=0.1,
        )

    assert excinfo.value.metadata["target_kind"] == "parametric"
    assert excinfo.value.metadata["failure_kind"] == "invalid-input"
    assert excinfo.value.metadata["n_valid_obs"] == 0
    assert excinfo.value.metadata["valid_sample_size"] == 0
    assert excinfo.value.metadata["projection_shape"] == (0, 1)


def test_parametric_inference_payload_rejects_asymmetric_covariance_objects() -> None:
    inference_module = _load_inference_module()
    results_module = importlib.import_module("hddid.results")
    z_critical = 1.6448536269514715

    payload = inference_module.ParametricInferencePayload(
        xi=np.eye(2, dtype=float),
        t_hat=np.zeros(2, dtype=float),
        w_hat=np.eye(2, dtype=float),
        sigma_tilde_x_hat=np.eye(2, dtype=float),
        omega_beta_hat=np.eye(2, dtype=float),
        score_moment=np.zeros(2, dtype=float),
        asymptotic_variance_hat=np.ones(2, dtype=float),
        standard_errors=np.full(2, 0.1, dtype=float),
        confidence_interval=results_module.ConfidenceInterval(
            lower=np.full(2, -z_critical * 0.1, dtype=float),
            upper=np.full(2, z_critical * 0.1, dtype=float),
            level=0.9,
        ),
        alpha=0.1,
        basis_family="polynomial",
        basis_degree=2,
        oracle_lane="r-parity-polynomial",
        optimization_metadata={
            "solver": "manual-positive-variance",
            "beta_hat": np.zeros(2, dtype=float),
            "n_valid_obs": 100,
        },
    )

    asymmetric_sigma = np.asarray(payload.sigma_tilde_x_hat, dtype=float).copy()
    asymmetric_sigma[0, 1] += 1e-4
    with pytest.raises(ValueError, match="sigma_tilde_x_hat must be symmetric"):
        replace(payload, sigma_tilde_x_hat=asymmetric_sigma)

    indefinite_sigma = np.asarray(payload.sigma_tilde_x_hat, dtype=float).copy()
    indefinite_sigma[0, 0] = -1e-6
    with pytest.raises(
        ValueError,
        match="sigma_tilde_x_hat must be positive semidefinite",
    ):
        replace(payload, sigma_tilde_x_hat=indefinite_sigma)

    with pytest.raises(
        ValueError,
        match="sigma_tilde_x_hat must align with xi columns",
    ):
        replace(
            payload,
            sigma_tilde_x_hat=np.eye(3, dtype=float),
            omega_beta_hat=np.eye(3, dtype=float),
            score_moment=np.zeros(3, dtype=float),
        )

    asymmetric_omega = np.asarray(payload.omega_beta_hat, dtype=float).copy()
    asymmetric_omega[0, 1] += 1e-4
    with pytest.raises(ValueError, match="omega_beta_hat must be symmetric"):
        replace(payload, omega_beta_hat=asymmetric_omega)

    indefinite_omega = np.asarray(payload.omega_beta_hat, dtype=float).copy()
    indefinite_omega[0, 0] = -1e-6
    with pytest.raises(
        ValueError,
        match="omega_beta_hat must be positive semidefinite",
    ):
        replace(payload, omega_beta_hat=indefinite_omega)

    with pytest.raises(ValueError, match="score_moment must align with xi columns"):
        replace(payload, score_moment=np.zeros(3, dtype=float))

    with pytest.raises(
        ValueError,
        match="asymptotic_variance_hat must contain only finite values",
    ):
        replace(
            payload,
            asymptotic_variance_hat=np.full_like(
                payload.asymptotic_variance_hat,
                np.inf,
            ),
        )

    with pytest.raises(
        ValueError,
        match="asymptotic_variance_hat must be strictly positive",
    ):
        replace(
            payload,
            asymptotic_variance_hat=np.zeros_like(payload.asymptotic_variance_hat),
        )

    with pytest.raises(ValueError, match="standard_errors must be strictly positive"):
        replace(payload, standard_errors=np.zeros_like(payload.standard_errors))

    with pytest.raises(
        ValueError,
        match="standard_errors must equal "
        r"sqrt\(asymptotic_variance_hat / n_valid_obs\)",
    ):
        replace(payload, standard_errors=np.full_like(payload.standard_errors, 0.2))

    metadata_without_n_valid = dict(payload.optimization_metadata)
    metadata_without_n_valid.pop("n_valid_obs")
    with pytest.raises(
        ValueError,
        match="n_valid_obs must be provided to validate standard_errors",
    ):
        replace(payload, optimization_metadata=metadata_without_n_valid)

    metadata_with_fractional_n_valid = dict(payload.optimization_metadata)
    metadata_with_fractional_n_valid["n_valid_obs"] = 100.5
    with pytest.raises(ValueError, match="n_valid_obs must be an integer"):
        replace(payload, optimization_metadata=metadata_with_fractional_n_valid)

    with pytest.raises(
        ValueError,
        match="confidence_interval.level must equal 1 - alpha",
    ):
        replace(
            payload,
            confidence_interval=results_module.ConfidenceInterval(
                lower=np.asarray(payload.confidence_interval.lower, dtype=float),
                upper=np.asarray(payload.confidence_interval.upper, dtype=float),
                level=0.95,
            ),
        )

    stale_lower = np.asarray(payload.confidence_interval.lower, dtype=float).copy()
    stale_lower[0] -= 0.01
    with pytest.raises(
        ValueError,
        match=r"confidence_interval\.lower must equal center - critical_value \* scale",
    ):
        replace(
            payload,
            confidence_interval=results_module.ConfidenceInterval(
                lower=stale_lower,
                upper=np.asarray(payload.confidence_interval.upper, dtype=float),
                level=payload.confidence_interval.level,
            ),
        )


def test_parametric_payload_accepts_alternate_equal_l1_sparse_direction() -> None:
    inference_module = _load_inference_module()
    results_module = importlib.import_module("hddid.results")

    sigma_tilde_x_hat = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
    xi = np.array([[-1.0, -1.0]], dtype=float)
    expected_w_hat, metadata = inference_module.solve_eq42_sparse_direction(
        sigma_tilde_x_hat=sigma_tilde_x_hat,
        xi=xi,
        lambda_prime=0.0,
    )
    forged_w_hat = np.array([[0.0, 1.0]], dtype=float)

    assert bool(metadata["feasible"]) is True
    np.testing.assert_allclose(expected_w_hat, np.array([[1.0, 0.0]], dtype=float))
    assert np.sum(np.abs(forged_w_hat)) == pytest.approx(
        np.sum(np.abs(expected_w_hat))
    )

    omega_beta_hat = np.eye(2, dtype=float)
    score_moment = np.zeros(2, dtype=float)
    beta_hat = np.zeros(2, dtype=float)
    t_hat = xi @ beta_hat - forged_w_hat @ score_moment
    asymptotic_variance_hat = np.einsum(
        "ij,jk,ik->i",
        forged_w_hat,
        omega_beta_hat,
        forged_w_hat,
    )
    standard_errors = np.sqrt(asymptotic_variance_hat / 100)
    z_critical = 1.9599639845400536

    payload = inference_module.ParametricInferencePayload(
        xi=xi,
        t_hat=t_hat,
        w_hat=forged_w_hat,
        sigma_tilde_x_hat=sigma_tilde_x_hat,
        omega_beta_hat=omega_beta_hat,
        score_moment=score_moment,
        asymptotic_variance_hat=asymptotic_variance_hat,
        standard_errors=standard_errors,
        confidence_interval=results_module.ConfidenceInterval(
            lower=t_hat - z_critical * standard_errors,
            upper=t_hat + z_critical * standard_errors,
            level=0.95,
        ),
        alpha=0.05,
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        optimization_metadata={
            "beta_hat": beta_hat,
            "n_valid_obs": 100,
            "lambda_prime": 0.0,
            "max_threshold_steps": 25,
        },
    )

    np.testing.assert_allclose(payload.w_hat, forged_w_hat, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"lambda_prime": np.inf}, "lambda_prime must be finite"),
        ({"lambda_prime": False}, "lambda_prime must be numeric"),
        ({"lambda_prime": [0.0]}, "lambda_prime must be a scalar"),
        ({"lambda_prime": np.array([0.0])}, "lambda_prime must be a scalar"),
        ({"max_threshold_steps": 0}, "max_threshold_steps must be positive"),
        ({"max_threshold_steps": 1.5}, "max_threshold_steps must be an integer"),
    ),
)
def test_solve_eq42_sparse_direction_rejects_invalid_solver_controls(
    kwargs: dict[str, object],
    match: str,
) -> None:
    inference_module = _load_inference_module()
    call_kwargs = {"lambda_prime": 0.0, **kwargs}

    with pytest.raises(ValueError, match=match):
        inference_module.solve_eq42_sparse_direction(
            sigma_tilde_x_hat=np.eye(2, dtype=float),
            xi=np.eye(2, dtype=float),
            **call_kwargs,
        )


def test_solve_eq42_sparse_direction_raises_structured_infeasible_error() -> None:
    inference_module = _load_inference_module()

    with pytest.raises(inference_module.SparseDirectionInfeasibleError) as excinfo:
        inference_module.solve_eq42_sparse_direction(
            sigma_tilde_x_hat=np.array([[0.0]], dtype=float),
            xi=np.array([[1.0]], dtype=float),
            lambda_prime=0.0,
        )

    assert excinfo.value.metadata["failure_kind"] == "eq42-infeasible"
    assert excinfo.value.metadata["target_kind"] == "eq42-sparse-direction"
    assert excinfo.value.metadata["row_index"] == 0
    assert excinfo.value.metadata["xi_count"] == 1
    assert excinfo.value.metadata["sigma_tilde_x_dimension"] == 1
    assert excinfo.value.metadata["constraint_violation"] == np.inf
    assert excinfo.value.metadata["lambda_prime"] == 0.0
    assert excinfo.value.metadata["linprog_status"] == 2


@pytest.mark.parametrize(
    ("sigma_tilde_x_hat", "xi", "match"),
    (
        (
            np.empty((0, 0), dtype=float),
            np.empty((0, 0), dtype=float),
            r"sigma_tilde_x_hat must have positive Eq\. \(4\.2\) dimension",
        ),
        (
            np.eye(2, dtype=float),
            np.empty((0, 2), dtype=float),
            "xi must contain at least one parametric target",
        ),
    ),
)
def test_solve_eq42_sparse_direction_rejects_empty_target_surface(
    sigma_tilde_x_hat: np.ndarray,
    xi: np.ndarray,
    match: str,
) -> None:
    inference_module = _load_inference_module()

    with pytest.raises(ValueError, match=match):
        inference_module.solve_eq42_sparse_direction(
            sigma_tilde_x_hat=sigma_tilde_x_hat,
            xi=xi,
            lambda_prime=0.0,
        )


@pytest.mark.parametrize(
    ("sigma_tilde_x_hat", "match"),
    (
        (
            np.array([[1.0, 0.5], [0.0, 1.0]], dtype=float),
            "sigma_tilde_x_hat must be symmetric",
        ),
        (
            np.array([[1.0, 2.0], [2.0, 1.0]], dtype=float),
            "sigma_tilde_x_hat must be positive semidefinite",
        ),
    ),
)
def test_solve_eq42_sparse_direction_rejects_invalid_covariance_object(
    sigma_tilde_x_hat: np.ndarray,
    match: str,
) -> None:
    inference_module = _load_inference_module()

    with pytest.raises(ValueError, match=match):
        inference_module.solve_eq42_sparse_direction(
            sigma_tilde_x_hat=sigma_tilde_x_hat,
            xi=np.eye(2, dtype=float),
            lambda_prime=0.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"lambda_double_prime": np.nan}, "lambda_double_prime must be finite"),
        ({"lambda_double_prime": True}, "lambda_double_prime must be numeric"),
        (
            {"lambda_double_prime": [0.0]},
            "lambda_double_prime must be a scalar",
        ),
        (
            {"lambda_double_prime": np.array([0.0])},
            "lambda_double_prime must be a scalar",
        ),
        ({"max_threshold_steps": 0}, "max_threshold_steps must be positive"),
        ({"max_threshold_steps": True}, "max_threshold_steps must be an integer"),
    ),
)
def test_solve_eq43_projection_matrix_rejects_invalid_solver_controls(
    kwargs: dict[str, object],
    match: str,
) -> None:
    inference_module = _load_inference_module()
    call_kwargs = {"lambda_double_prime": 0.0, **kwargs}

    with pytest.raises(ValueError, match=match):
        inference_module.solve_eq43_projection_matrix(
            sigma_x_hat=np.eye(2, dtype=float),
            cross_moment_hat=np.eye(2, dtype=float),
            **call_kwargs,
        )


@pytest.mark.parametrize(
    ("sigma_x_hat", "match"),
    (
        (
            np.array([[1.0, 0.5], [0.0, 1.0]], dtype=float),
            "sigma_x_hat must be symmetric",
        ),
        (
            np.array([[1.0, 2.0], [2.0, 1.0]], dtype=float),
            "sigma_x_hat must be positive semidefinite",
        ),
    ),
)
def test_solve_eq43_projection_matrix_rejects_invalid_covariance_object(
    sigma_x_hat: np.ndarray,
    match: str,
) -> None:
    inference_module = _load_inference_module()

    with pytest.raises(ValueError, match=match):
        inference_module.solve_eq43_projection_matrix(
            sigma_x_hat=sigma_x_hat,
            cross_moment_hat=np.eye(2, dtype=float),
            lambda_double_prime=0.0,
        )


def test_solve_eq43_projection_matrix_rejects_empty_basis_target_surface() -> None:
    inference_module = _load_inference_module()

    with pytest.raises(
        ValueError,
        match=r"cross_moment_hat must contain at least one Eq\. \(4\.3\) basis target",
    ):
        inference_module.solve_eq43_projection_matrix(
            sigma_x_hat=np.eye(2, dtype=float),
            cross_moment_hat=np.empty((0, 2), dtype=float),
            lambda_double_prime=0.0,
        )


def test_solve_eq43_projection_matrix_rejects_string_backed_cross_moment() -> None:
    inference_module = _load_inference_module()

    with pytest.raises(ValueError, match="cross_moment_hat must be numeric, not string"):
        inference_module.solve_eq43_projection_matrix(
            sigma_x_hat=np.eye(2, dtype=float),
            cross_moment_hat=_StringArrayLike(np.eye(2).astype(str)),
            lambda_double_prime=0.0,
        )


def test_solve_eq43_projection_matrix_raises_structured_infeasible_error() -> None:
    inference_module = _load_inference_module()

    with pytest.raises(inference_module.SparseDirectionInfeasibleError) as excinfo:
        inference_module.solve_eq43_projection_matrix(
            sigma_x_hat=np.zeros((1, 1), dtype=float),
            cross_moment_hat=np.array([[1.0]], dtype=float),
            lambda_double_prime=0.0,
        )

    assert excinfo.value.metadata["target_kind"] == "eq43-projection"
    assert excinfo.value.metadata["failure_kind"] == "eq43-infeasible"
    assert excinfo.value.metadata["row_index"] == 0
    assert excinfo.value.metadata["x_dimension"] == 1
    assert excinfo.value.metadata["basis_dimension_full"] == 1
    assert excinfo.value.metadata["constraint_violation"] == np.inf
    assert excinfo.value.metadata["lambda_double_prime"] == 0.0


def test_estimate_nonparametric_inference_adds_context_to_eq43_infeasibility(
    eq43_nonparametric_reference_slice: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_module = _load_inference_module()

    def raise_eq43_infeasible(**_kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        raise inference_module.SparseDirectionInfeasibleError(
            "forced Eq. (4.3) infeasibility",
            metadata={
                "target_kind": "eq43-projection",
                "constraint_violation": 0.25,
                "lambda_double_prime": 0.0,
                "row_index": 2,
                "linprog_status": 2,
                "linprog_message": "infeasible",
            },
        )

    monkeypatch.setattr(
        inference_module,
        "solve_eq43_projection_matrix",
        raise_eq43_infeasible,
    )

    with pytest.raises(inference_module.SparseDirectionInfeasibleError) as excinfo:
        inference_module.estimate_nonparametric_inference(
            eq43_nonparametric_reference_slice["score_payload"],
            eq43_nonparametric_reference_slice["estimation_payload"],
            alpha=eq43_nonparametric_reference_slice["alpha"],
            lambda_double_prime=eq43_nonparametric_reference_slice[
                "lambda_double_prime"
            ],
            n_boot=eq43_nonparametric_reference_slice["n_boot"],
            random_state=eq43_nonparametric_reference_slice["random_state"],
        )

    metadata = excinfo.value.metadata
    assert metadata["target_kind"] == "nonparametric"
    assert metadata["failure_kind"] == "eq43-infeasible"
    assert metadata["solver_target_kind"] == "eq43-projection"
    assert metadata["constraint_violation_max"] == pytest.approx(0.25)
    assert metadata["row_index"] == 2
    assert metadata["linprog_status"] == 2
    assert metadata["linprog_message"] == "infeasible"
    assert metadata["basis_dimension_full"] == (
        eq43_nonparametric_reference_slice["score_payload"].basis_valid_full.shape[1]
    )
    assert metadata["grid_size"] == (
        eq43_nonparametric_reference_slice["score_payload"].evaluation_basis.shape[0]
    )
    assert metadata["n_valid_obs"] == eq43_nonparametric_reference_slice["n_valid"]


def test_estimate_nonparametric_inference_rejects_empty_valid_sample_before_matrix_math() -> None:
    inference_module = _load_inference_module()
    score_payload = SimpleNamespace(
        basis_valid_full=np.empty((0, 1), dtype=float),
        evaluation_basis=np.ones((1, 1), dtype=float),
        x_valid=np.empty((0, 0), dtype=float),
        oracle_lane="r-parity-polynomial",
        basis_family="polynomial",
        basis_degree=0,
    )
    estimation_payload = SimpleNamespace(
        residual_valid=np.empty(0, dtype=float),
        gamma_hat=np.zeros(1, dtype=float),
        optimization_metadata={"n_valid_obs": 0},
    )

    with pytest.raises(inference_module.InvalidInferenceInputError) as excinfo:
        inference_module.estimate_nonparametric_inference(
            score_payload,
            estimation_payload,
            alpha=0.1,
            n_boot=2,
        )

    assert excinfo.value.metadata["target_kind"] == "nonparametric"
    assert excinfo.value.metadata["failure_kind"] == "invalid-input"
    assert excinfo.value.metadata["n_valid_obs"] == 0
    assert excinfo.value.metadata["valid_sample_size"] == 0
    assert excinfo.value.metadata["basis_valid_full_shape"] == (0, 1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"n_boot": 1.5}, "n_boot must be an integer"),
        ({"n_boot": True}, "n_boot must be an integer"),
        ({"random_state": 202.5}, "random_state must be an integer"),
        ({"random_state": False}, "random_state must be an integer"),
    ),
)
def test_estimate_nonparametric_inference_rejects_truncated_uniform_band_controls(
    eq43_nonparametric_reference_slice: dict[str, object],
    kwargs: dict[str, object],
    match: str,
) -> None:
    inference_module = _load_inference_module()
    call_kwargs = {
        "alpha": eq43_nonparametric_reference_slice["alpha"],
        "lambda_double_prime": eq43_nonparametric_reference_slice[
            "lambda_double_prime"
        ],
        "n_boot": eq43_nonparametric_reference_slice["n_boot"],
        "random_state": eq43_nonparametric_reference_slice["random_state"],
        **kwargs,
    }

    with pytest.raises(ValueError, match=match):
        inference_module.estimate_nonparametric_inference(
            eq43_nonparametric_reference_slice["score_payload"],
            eq43_nonparametric_reference_slice["estimation_payload"],
            **call_kwargs,
        )


def test_estimate_nonparametric_inference_raises_structured_singularity_for_sigma_f(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    score_payload = eq43_nonparametric_reference_slice["score_payload"]
    singular_basis_valid_full = np.asarray(
        score_payload.basis_valid_full, dtype=float
    ).copy()
    singular_basis_valid_full[:, 2] = singular_basis_valid_full[:, 1]
    singular_score_payload = replace(
        score_payload,
        basis_matrix=singular_basis_valid_full,
        basis_valid_full=singular_basis_valid_full,
        basis_design_valid=singular_basis_valid_full[:, 1:],
    )
    base_estimation_payload = eq43_nonparametric_reference_slice["estimation_payload"]
    beta_hat = np.asarray(base_estimation_payload.beta_hat, dtype=float)
    gamma_hat = np.asarray(base_estimation_payload.gamma_hat, dtype=float)
    fitted_f_valid = singular_basis_valid_full @ gamma_hat
    second_stage_prediction_valid = (
        np.asarray(score_payload.x_valid, dtype=float) @ beta_hat + fitted_f_valid
    )
    residual_valid = np.asarray(score_payload.s_hat_valid, dtype=float) - (
        second_stage_prediction_valid
    )
    singular_estimation_payload = replace(
        base_estimation_payload,
        fitted_f_valid=fitted_f_valid,
        second_stage_prediction_valid=second_stage_prediction_valid,
        residual_valid=residual_valid,
        f_hat_at_z0=np.asarray(score_payload.evaluation_basis, dtype=float)
        @ gamma_hat,
    )

    with pytest.raises(inference_module.SingularCovarianceError) as excinfo:
        inference_module.estimate_nonparametric_inference(
            singular_score_payload,
            singular_estimation_payload,
            alpha=eq43_nonparametric_reference_slice["alpha"],
            lambda_double_prime=eq43_nonparametric_reference_slice[
                "lambda_double_prime"
            ],
            n_boot=eq43_nonparametric_reference_slice["n_boot"],
            random_state=eq43_nonparametric_reference_slice["random_state"],
        )

    assert excinfo.value.metadata["target_kind"] == "nonparametric"
    assert excinfo.value.metadata["matrix_name"] == "sigma_f_hat"
    assert excinfo.value.metadata["matrix_shape"] == (3, 3)


def test_estimate_nonparametric_inference_raises_nonpositive_variance_for_zero_grid_variance(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    score_payload = eq43_nonparametric_reference_slice["score_payload"]
    zero_variance_grid = np.zeros(
        (1, score_payload.evaluation_basis.shape[1]), dtype=float
    )
    zero_variance_score_payload = replace(
        score_payload, evaluation_basis=zero_variance_grid
    )
    estimation_payload = eq43_nonparametric_reference_slice["estimation_payload"]
    zero_variance_estimation_payload = replace(
        estimation_payload,
        f_hat_at_z0=zero_variance_grid
        @ np.asarray(estimation_payload.gamma_hat, dtype=float),
    )

    with pytest.raises(inference_module.NonpositiveVarianceError) as excinfo:
        inference_module.estimate_nonparametric_inference(
            zero_variance_score_payload,
            zero_variance_estimation_payload,
            alpha=eq43_nonparametric_reference_slice["alpha"],
            lambda_double_prime=eq43_nonparametric_reference_slice[
                "lambda_double_prime"
            ],
            n_boot=eq43_nonparametric_reference_slice["n_boot"],
            random_state=eq43_nonparametric_reference_slice["random_state"],
        )

    assert excinfo.value.metadata["target_kind"] == "nonparametric"
    assert excinfo.value.metadata["grid_size"] == 1
    assert excinfo.value.metadata["failure_kind"] == "nonpositive-variance"
    assert excinfo.value.metadata["minimum_value"] == pytest.approx(0.0)
    assert excinfo.value.metadata["maximum_value"] == pytest.approx(0.0)
    assert excinfo.value.metadata["nonpositive_entry_count"] == 1
    assert excinfo.value.metadata["nonpositive_entry_indices"] == (0,)


def test_estimate_nonparametric_inference_requires_explicit_evaluation_grid(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    score_payload = eq43_nonparametric_reference_slice["score_payload"]

    with pytest.raises(
        ValueError,
        match="evaluation_basis must contain at least one grid point",
    ):
        replace(
            score_payload,
            evaluation_basis=np.empty(
                (0, score_payload.evaluation_basis.shape[1]), dtype=float
            ),
        )


def test_estimate_nonparametric_inference_rejects_non_psd_paper_omega_by_default() -> (
    None
):
    inference_module = _load_inference_module()
    score_payload, estimation_payload, design, _replication_seed = (
        _build_phase7_nonparametric_failure_slice()
    )

    with pytest.raises(inference_module.NonpositiveVarianceError) as excinfo:
        inference_module.estimate_nonparametric_inference(
            score_payload,
            estimation_payload,
            alpha=design.alpha,
            lambda_double_prime=0.0,
            n_boot=64,
        )

    metadata = excinfo.value.metadata
    assert metadata["target_kind"] == "nonparametric"
    assert metadata["failure_kind"] == "nonpositive-variance"
    assert metadata["matrix_name"] == "omega_f_hat"
    assert metadata["omega_f_primary_psd"] is False
    assert metadata["omega_f_fallback_attempted"] is False
    assert metadata["omega_f_fallback_used"] is False
    assert metadata["omega_f_selected_strategy"] == "paper-difference"
    assert metadata["basis_dimension_full"] == 17
    assert metadata["x_dimension"] == 10
    assert metadata["n_valid_obs"] > 0
    assert metadata["sigma_x_rank"] <= metadata["x_dimension"]
    assert metadata["sigma_f_min_singular_value"] > 0.0
    assert metadata["sigma_f_condition_number"] >= 1.0
    assert metadata["eq43_solver"] == "eq43-linear-programming-l1"
    assert metadata["eq43_feasible"] is True
    assert metadata["eq43_constraint_violation_max"] == pytest.approx(0.0)
    assert metadata["paper_object"] == "Omega_f"
    assert metadata["paper_formula"] == (
        "E[sigma_i^2 psi psi'] - M E[sigma_i^2 X X'] M'"
    )
    assert metadata["omega_f_basis_component_min_eigenvalue"] >= -1e-10
    assert metadata["omega_f_x_component_min_eigenvalue"] >= -1e-10
    assert metadata["omega_f_basis_component_trace"] > 0.0
    assert metadata["omega_f_x_component_trace"] > 0.0
    assert metadata["omega_f_paper_difference_trace"] == pytest.approx(
        metadata["omega_f_basis_component_trace"]
        - metadata["omega_f_x_component_trace"]
    )
    assert metadata["omega_f_paper_difference_negative_eigenvalue_count"] > 0
    assert metadata["omega_f_paper_difference_negative_eigenvalue_mass"] > 0.0
    assert metadata["omega_f_orthogonal_score_negative_eigenvalue_count"] == 0
    assert metadata["omega_f_orthogonal_score_negative_eigenvalue_mass"] == 0.0
    assert (
        metadata["omega_f_residual_weighted_cross_correction_negative_eigenvalue_count"]
        > 0
    )
    assert (
        metadata["omega_f_residual_weighted_cross_correction_negative_eigenvalue_mass"]
        > 0.0
    )
    assert metadata["omega_f_orthogonal_score_min_eigenvalue"] >= -1e-10
    assert metadata["omega_f_orthogonal_score_trace"] > 0.0
    assert metadata["omega_f_paper_difference_vs_orthogonal_score_max_abs"] > 0.0
    assert metadata["omega_f_unweighted_cross_identity_max_abs"] <= 1e-10
    assert metadata["omega_f_residual_weighted_cross_identity_max_abs"] > 0.0


def test_estimate_nonparametric_inference_rejects_fallback_flag_before_failure_slice() -> (
    None
):
    inference_module = _load_inference_module()
    score_payload, estimation_payload, design, replication_seed = (
        _build_phase7_nonparametric_failure_slice()
    )

    with pytest.raises(
        ValueError,
        match="allow_omega_f_fallback is not supported",
    ):
        inference_module.estimate_nonparametric_inference(
            score_payload,
            estimation_payload,
            alpha=design.alpha,
            lambda_double_prime=0.0,
            n_boot=64,
            random_state=replication_seed,
            allow_omega_f_fallback=True,
        )


def test_estimate_nonparametric_inference_rejects_fallback_control_on_valid_slice(
    eq43_nonparametric_reference_slice: dict[str, object],
) -> None:
    inference_module = _load_inference_module()

    with pytest.raises(
        ValueError,
        match="allow_omega_f_fallback is not supported",
    ):
        inference_module.estimate_nonparametric_inference(
            eq43_nonparametric_reference_slice["score_payload"],
            eq43_nonparametric_reference_slice["estimation_payload"],
            alpha=eq43_nonparametric_reference_slice["alpha"],
            lambda_double_prime=eq43_nonparametric_reference_slice[
                "lambda_double_prime"
            ],
            n_boot=eq43_nonparametric_reference_slice["n_boot"],
            random_state=eq43_nonparametric_reference_slice["random_state"],
            allow_omega_f_fallback=True,
        )


def test_estimate_nonparametric_inference_rejects_fallback_flag_on_dgp2_seed202_slice() -> (
    None
):
    inference_module = _load_inference_module()
    score_payload, estimation_payload, design, _dataset, replication_seed = (
        _build_phase7_dgp2_seed202_blocker_slice()
    )

    with pytest.raises(
        ValueError,
        match="allow_omega_f_fallback is not supported",
    ):
        inference_module.estimate_nonparametric_inference(
            score_payload,
            estimation_payload,
            alpha=design.alpha,
            lambda_double_prime=0.0,
            n_boot=64,
            random_state=replication_seed,
            allow_omega_f_fallback=True,
        )


def test_nonparametric_payload_rejects_forged_omega_f_strategy_metadata(
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
    forged_metadata = dict(payload.optimization_metadata)
    forged_metadata["omega_f_strategy"] = "orthogonal-score-fallback"
    forged_metadata["omega_f_selected_strategy"] = "orthogonal-score-fallback"
    forged_metadata["omega_f_fallback_attempted"] = True
    forged_metadata["omega_f_fallback_used"] = True
    forged_metadata["omega_f_fallback_min_eigenvalue"] = 1.0

    with pytest.raises(
        ValueError,
        match="omega_f_strategy metadata must equal paper-difference",
    ):
        replace(payload, optimization_metadata=forged_metadata)


def test_nonparametric_payload_rejects_forged_omega_f_eigenvalue_metadata(
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
    selected_min_eigenvalue = float(
        np.min(
            np.linalg.eigvalsh(
                0.5 * (payload.omega_f_hat + payload.omega_f_hat.T)
            )
        )
    )
    forged_metadata = dict(payload.optimization_metadata)
    forged_metadata["omega_f_selected_min_eigenvalue"] = selected_min_eigenvalue * 10.0

    with pytest.raises(
        ValueError,
        match="omega_f_selected_min_eigenvalue metadata must match omega_f_hat",
    ):
        replace(payload, optimization_metadata=forged_metadata)


@pytest.mark.parametrize(
    ("metadata_value", "message"),
    (
        ([True, False, True], "gamma_hat must be numeric, not boolean"),
        (["0.4", "-0.2", "0.15"], "gamma_hat must be numeric, not string"),
    ),
)
def test_nonparametric_payload_rejects_gamma_hat_metadata_aliases(
    metadata_value: list[object],
    message: str,
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
    forged_metadata = dict(payload.optimization_metadata)
    forged_metadata["gamma_hat"] = metadata_value

    with pytest.raises(ValueError, match=message):
        replace(payload, optimization_metadata=forged_metadata)


@pytest.mark.parametrize("forged_primary_min", (np.nan, -np.inf))
def test_nonparametric_payload_rejects_nonfinite_primary_omega_eigenvalue_metadata(
    forged_primary_min: float,
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
    forged_metadata = dict(payload.optimization_metadata)
    forged_metadata["omega_f_primary_min_eigenvalue"] = forged_primary_min

    with pytest.raises(
        ValueError,
        match="omega_f_primary_min_eigenvalue metadata must be finite",
    ):
        replace(payload, optimization_metadata=forged_metadata)


def test_estimate_nonparametric_inference_rejects_fallback_flag_without_consulting_score_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_module = _load_inference_module()
    score_payload, estimation_payload, design, _dataset, replication_seed = (
        _build_phase7_dgp2_seed202_blocker_slice()
    )

    def _indefinite_fallback_omega(**kwargs: object) -> np.ndarray:
        orthogonal_basis_valid = np.asarray(kwargs["orthogonal_basis_valid"], dtype=float)
        return -np.eye(orthogonal_basis_valid.shape[1], dtype=float)

    monkeypatch.setattr(
        inference_module,
        "_orthogonal_score_omega_f_hat",
        _indefinite_fallback_omega,
    )

    with pytest.raises(
        ValueError,
        match="allow_omega_f_fallback is not supported",
    ):
        inference_module.estimate_nonparametric_inference(
            score_payload,
            estimation_payload,
            alpha=design.alpha,
            lambda_double_prime=0.0,
            n_boot=64,
            random_state=replication_seed,
            allow_omega_f_fallback=True,
        )
