from __future__ import annotations

import importlib

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


class _StringVectorLike:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=object)

    def __array__(self, dtype: object = None) -> np.ndarray:
        if dtype is None:
            return self._values
        return self._values.astype(dtype)


class _StringMatrixLike:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=object)

    def __array__(self, dtype: object = None) -> np.ndarray:
        if dtype is None:
            return self._values
        return self._values.astype(dtype)


def test_estimate_eq31_mainline_zero_penalty_recovers_beta_gamma(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    assert hasattr(estimation_module, "EstimationPayload")
    assert hasattr(estimation_module, "estimate_eq31_mainline")

    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    estimation_payload, result = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
    )

    expected_projection_x = (
        score_payload.x_valid
        - score_payload.basis_valid_full
        @ np.linalg.lstsq(
            score_payload.basis_valid_full,
            score_payload.x_valid,
            rcond=None,
        )[0]
    )
    expected_fitted_f = score_payload.basis_valid_full @ eq31_manual_slice["gamma_true"]
    expected_prediction = (
        score_payload.x_valid @ eq31_manual_slice["beta_true"] + expected_fitted_f
    )

    np.testing.assert_allclose(
        estimation_payload.beta_hat,
        eq31_manual_slice["beta_true"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        estimation_payload.gamma_hat,
        eq31_manual_slice["gamma_true"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        estimation_payload.projection_x_valid,
        expected_projection_x,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        estimation_payload.fitted_f_valid,
        expected_fitted_f,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        estimation_payload.second_stage_prediction_valid,
        expected_prediction,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        estimation_payload.residual_valid,
        np.zeros_like(score_payload.s_hat_valid),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.parametric_estimates["beta_hat"],
        eq31_manual_slice["beta_true"],
        atol=1e-9,
    )


def test_estimate_eq31_mainline_routes_penalty_only_to_beta(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )

    baseline_payload, _ = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
    )
    penalized_payload, result = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.2,
    )

    assert np.linalg.norm(penalized_payload.beta_hat, ord=1) < np.linalg.norm(
        baseline_payload.beta_hat,
        ord=1,
    )
    assert penalized_payload.optimization_metadata["penalty_target"] == "beta-only"
    assert penalized_payload.optimization_metadata["penalty_lambda"] == pytest.approx(
        0.2
    )
    assert penalized_payload.optimization_metadata["basis_intercept_dropped"] is True
    assert result.standard_errors == {}
    assert result.intervals == {}
    assert result.diagnostics is not None
    assert result.diagnostics.n_holdout_raw == 6
    assert result.diagnostics.n_trimmed_propensity == 1
    assert result.diagnostics.n_valid_holdout == 5
    assert result.diagnostics.optimization_metadata["penalty_target"] == "beta-only"
    assert result.nonparametric_estimates["gamma_hat"].shape == (3,)
    assert result.nonparametric_estimates["f_hat_at_z0"].shape == (2,)
    assert result.parametric_estimates["beta_hat"].shape == (2,)
    assert result.diagnostics.fold_diagnostics[0].n_valid_holdout == 1


def test_estimate_eq31_mainline_uses_eq31_lasso_scale() -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()

    x_valid = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=float)
    s_hat_valid = x_valid[:, 0].copy()
    valid_mask = np.ones(x_valid.shape[0], dtype=bool)
    basis = np.ones((x_valid.shape[0], 1), dtype=float)
    score_payload = score_module.ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=valid_mask,
        fold_ids=np.ones(x_valid.shape[0], dtype=int),
        fold_diagnostics=[],
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        basis_matrix=basis,
        x_valid=x_valid,
        basis_valid_full=basis,
        basis_design_valid=np.empty((x_valid.shape[0], 0), dtype=float),
        evaluation_basis=np.ones((1, 1), dtype=float),
        pi_hat=np.full(x_valid.shape[0], 0.5, dtype=float),
        phi0_hat=np.zeros(x_valid.shape[0], dtype=float),
        phi1_hat=np.zeros(x_valid.shape[0], dtype=float),
        rho_hat=np.ones(x_valid.shape[0], dtype=float),
        intercept_dropped_for_design=True,
    )

    estimation_payload, result = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.4,
        solver="builtin",
    )

    second_moment = float(np.mean(x_valid[:, 0] ** 2))
    correlation = float(np.mean(x_valid[:, 0] * s_hat_valid))
    expected_beta = (correlation - 0.4 / 2.0) / second_moment

    assert estimation_payload.beta_hat[0] == pytest.approx(expected_beta)
    assert estimation_payload.beta_hat[0] == pytest.approx(0.9)
    assert result.diagnostics.optimization_metadata["penalty_lambda"] == pytest.approx(
        0.4
    )
    assert result.diagnostics.optimization_metadata[
        "coordinate_descent_threshold"
    ] == pytest.approx(0.2)


def test_estimate_eq31_mainline_accepts_kkt_certified_single_sweep_lasso() -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()

    x_valid = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=float)
    s_hat_valid = x_valid[:, 0].copy()
    valid_mask = np.ones(x_valid.shape[0], dtype=bool)
    basis = np.ones((x_valid.shape[0], 1), dtype=float)
    score_payload = score_module.ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=valid_mask,
        fold_ids=np.ones(x_valid.shape[0], dtype=int),
        fold_diagnostics=[],
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        basis_matrix=basis,
        x_valid=x_valid,
        basis_valid_full=basis,
        basis_design_valid=np.empty((x_valid.shape[0], 0), dtype=float),
        evaluation_basis=np.ones((1, 1), dtype=float),
        pi_hat=np.full(x_valid.shape[0], 0.5, dtype=float),
        phi0_hat=np.zeros(x_valid.shape[0], dtype=float),
        phi1_hat=np.zeros(x_valid.shape[0], dtype=float),
        rho_hat=np.ones(x_valid.shape[0], dtype=float),
        intercept_dropped_for_design=True,
    )

    estimation_payload, result = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.4,
        max_iter=1,
        tol=1e-20,
        solver="builtin",
    )

    metadata = estimation_payload.optimization_metadata
    assert metadata["solver"] == "coordinate-descent-lasso"
    assert metadata["coordinate_descent_converged"] is True
    assert metadata["coordinate_descent_iterations"] == 1
    assert metadata["coordinate_descent_convergence_criterion"] == "kkt-optimality"
    assert metadata["coordinate_descent_kkt_satisfied"] is True
    assert metadata["coordinate_descent_kkt_violation_max"] == pytest.approx(0.0)
    assert metadata["penalty_lambda"] == pytest.approx(0.4)
    assert metadata["n_valid_obs"] == 5
    assert metadata["beta_dimension"] == 1
    assert result.diagnostics.optimization_metadata[
        "coordinate_descent_convergence_criterion"
    ] == "kkt-optimality"


def test_estimate_eq31_mainline_rejects_nonoptimal_coordinate_descent() -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()

    x_valid = np.array(
        [
            [-2.0, -1.0],
            [-1.0, 0.5],
            [0.0, 0.0],
            [1.0, -0.5],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    s_hat_valid = x_valid @ np.array([1.0, 1.0], dtype=float)
    valid_mask = np.ones(x_valid.shape[0], dtype=bool)
    basis = np.ones((x_valid.shape[0], 1), dtype=float)
    score_payload = score_module.ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=valid_mask,
        fold_ids=np.ones(x_valid.shape[0], dtype=int),
        fold_diagnostics=[],
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        basis_matrix=basis,
        x_valid=x_valid,
        basis_valid_full=basis,
        basis_design_valid=np.empty((x_valid.shape[0], 0), dtype=float),
        evaluation_basis=np.ones((1, 1), dtype=float),
        pi_hat=np.full(x_valid.shape[0], 0.5, dtype=float),
        phi0_hat=np.zeros(x_valid.shape[0], dtype=float),
        phi1_hat=np.zeros(x_valid.shape[0], dtype=float),
        rho_hat=np.ones(x_valid.shape[0], dtype=float),
        intercept_dropped_for_design=True,
    )

    with pytest.raises(
        estimation_module.Eq31SolverConvergenceError,
        match="coordinate-descent Lasso failed to converge",
    ) as exc_info:
        estimation_module.estimate_eq31_mainline(
            score_payload,
            penalty_lambda=0.2,
            max_iter=1,
            tol=1e-20,
            solver="builtin",
        )

    metadata = exc_info.value.metadata
    assert metadata["solver"] == "coordinate-descent-lasso"
    assert metadata["coordinate_descent_converged"] is False
    assert metadata["coordinate_descent_iterations"] == 1
    assert metadata["coordinate_descent_convergence_criterion"] == "not-converged"
    assert metadata["coordinate_descent_kkt_satisfied"] is False
    assert metadata["coordinate_descent_kkt_violation_max"] > 0.1
    assert metadata["penalty_lambda"] == pytest.approx(0.2)
    assert metadata["max_iter"] == 1
    assert metadata["tol"] == pytest.approx(1e-20)
    assert metadata["n_valid_obs"] == 5
    assert metadata["beta_dimension"] == 2


def test_estimate_eq31_mainline_allows_empty_beta_block() -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()

    s_hat_valid = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
    valid_mask = np.ones(s_hat_valid.shape[0], dtype=bool)
    basis = np.column_stack(
        [np.ones(s_hat_valid.shape[0], dtype=float), np.linspace(0.0, 1.0, 4)]
    )
    x_valid = np.empty((s_hat_valid.shape[0], 0), dtype=float)
    score_payload = score_module.ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=valid_mask,
        fold_ids=np.ones(s_hat_valid.shape[0], dtype=int),
        fold_diagnostics=[],
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        basis_matrix=basis,
        x_valid=x_valid,
        basis_valid_full=basis,
        basis_design_valid=basis[:, 1:],
        evaluation_basis=np.array([[1.0, 0.5]], dtype=float),
        pi_hat=np.full(s_hat_valid.shape[0], 0.5, dtype=float),
        phi0_hat=np.zeros(s_hat_valid.shape[0], dtype=float),
        phi1_hat=np.zeros(s_hat_valid.shape[0], dtype=float),
        rho_hat=np.ones(s_hat_valid.shape[0], dtype=float),
        intercept_dropped_for_design=True,
    )

    estimation_payload, result = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.4,
        solver="builtin",
    )

    expected_gamma, *_ = np.linalg.lstsq(basis, s_hat_valid, rcond=None)
    expected_fitted = basis @ expected_gamma

    assert estimation_payload.optimization_metadata["solver"] == "empty-beta-block"
    assert estimation_payload.optimization_metadata["beta_dimension"] == 0
    assert estimation_payload.beta_hat.shape == (0,)
    assert estimation_payload.projection_x_valid.shape == (s_hat_valid.shape[0], 0)
    np.testing.assert_allclose(estimation_payload.gamma_hat, expected_gamma, atol=1e-12)
    np.testing.assert_allclose(
        estimation_payload.second_stage_prediction_valid,
        expected_fitted,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        estimation_payload.residual_valid,
        s_hat_valid - expected_fitted,
        atol=1e-12,
    )
    assert result.parametric_estimates["beta_hat"].shape == (0,)


def test_estimate_eq31_mainline_rejects_rank_deficient_sieve_projection() -> None:
    public_module = importlib.import_module("hddid")
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()

    assert public_module.Eq31ProjectionRankError is (
        estimation_module.Eq31ProjectionRankError
    )

    s_hat_valid = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
    valid_mask = np.ones(s_hat_valid.shape[0], dtype=bool)
    z_values = np.linspace(0.0, 1.0, s_hat_valid.shape[0], dtype=float)
    duplicated_basis = np.column_stack(
        [np.ones(s_hat_valid.shape[0], dtype=float), z_values, z_values]
    )
    score_payload = score_module.ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=valid_mask,
        fold_ids=np.ones(s_hat_valid.shape[0], dtype=int),
        fold_diagnostics=[],
        basis_family="polynomial",
        basis_degree=1,
        oracle_lane="r-parity-polynomial",
        basis_matrix=duplicated_basis,
        x_valid=np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=float),
        basis_valid_full=duplicated_basis,
        basis_design_valid=duplicated_basis[:, 1:],
        evaluation_basis=np.array([[1.0, 0.5, 0.5]], dtype=float),
        pi_hat=np.full(s_hat_valid.shape[0], 0.5, dtype=float),
        phi0_hat=np.zeros(s_hat_valid.shape[0], dtype=float),
        phi1_hat=np.zeros(s_hat_valid.shape[0], dtype=float),
        rho_hat=np.ones(s_hat_valid.shape[0], dtype=float),
        intercept_dropped_for_design=True,
    )

    with pytest.raises(
        estimation_module.Eq31ProjectionRankError,
        match=r"Eq\. \(3\.1\) sieve projection requires full column rank",
    ) as exc_info:
        estimation_module.estimate_eq31_mainline(score_payload)

    assert exc_info.value.metadata["matrix_name"] == "basis_valid_full"
    assert exc_info.value.metadata["matrix_shape"] == (4, 3)
    assert exc_info.value.metadata["matrix_rank"] == 2


def test_estimate_eq31_mainline_rejects_unidentified_unpenalized_beta_block() -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()

    s_hat_valid = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
    valid_mask = np.ones(s_hat_valid.shape[0], dtype=bool)
    basis = np.ones((s_hat_valid.shape[0], 1), dtype=float)
    duplicated_x = np.array(
        [
            [-1.0, -1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    score_payload = score_module.ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=valid_mask,
        fold_ids=np.ones(s_hat_valid.shape[0], dtype=int),
        fold_diagnostics=[],
        basis_family="polynomial",
        basis_degree=0,
        oracle_lane="r-parity-polynomial",
        basis_matrix=basis,
        x_valid=duplicated_x,
        basis_valid_full=basis,
        basis_design_valid=np.empty((s_hat_valid.shape[0], 0), dtype=float),
        evaluation_basis=np.ones((1, 1), dtype=float),
        pi_hat=np.full(s_hat_valid.shape[0], 0.5, dtype=float),
        phi0_hat=np.zeros(s_hat_valid.shape[0], dtype=float),
        phi1_hat=np.zeros(s_hat_valid.shape[0], dtype=float),
        rho_hat=np.ones(s_hat_valid.shape[0], dtype=float),
        intercept_dropped_for_design=True,
    )

    with pytest.raises(
        estimation_module.Eq31ProjectionRankError,
        match=r"Eq\. \(3\.1\) unpenalized beta block requires full column rank",
    ) as exc_info:
        estimation_module.estimate_eq31_mainline(
            score_payload,
            penalty_lambda=0.0,
        )

    assert exc_info.value.metadata["matrix_name"] == "projection_x_valid"
    assert exc_info.value.metadata["matrix_shape"] == (4, 2)
    assert exc_info.value.metadata["matrix_rank"] == 1
    assert exc_info.value.metadata["required_rank"] == 2


def test_estimate_eq31_mainline_rejects_invalid_solver_controls(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    estimation_module = _load_estimation_module()
    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )

    invalid_cases = (
        ({"penalty_lambda": np.nan}, "penalty_lambda must be finite"),
        ({"penalty_lambda": np.inf}, "penalty_lambda must be finite"),
        ({"penalty_lambda": -0.1}, "penalty_lambda must be non-negative"),
        ({"penalty_lambda": False}, "penalty_lambda must be numeric"),
        ({"penalty_lambda": "0.1"}, "penalty_lambda must be numeric"),
        ({"penalty_lambda": [0.1]}, "penalty_lambda must be a scalar"),
        ({"penalty_lambda": np.array([0.1])}, "penalty_lambda must be a scalar"),
        (
            {"penalty_lambda": np.array("0.1", dtype=object)},
            "penalty_lambda must be numeric",
        ),
        ({"max_iter": 0}, "max_iter must be positive"),
        ({"max_iter": 1.5}, "max_iter must be an integer"),
        ({"tol": np.nan}, "tol must be finite"),
        ({"tol": 0.0}, "tol must be positive"),
        ({"tol": True}, "tol must be numeric"),
        ({"tol": "1e-8"}, "tol must be numeric"),
        ({"tol": [1e-8]}, "tol must be a scalar"),
        ({"tol": np.array([1e-8])}, "tol must be a scalar"),
        ({"tol": np.array("1e-8", dtype=object)}, "tol must be numeric"),
    )

    for kwargs, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            estimation_module.estimate_eq31_mainline(score_payload, **kwargs)


def test_estimation_payload_rejects_boolean_numeric_evidence(
    eq31_manual_slice: dict[str, object],
) -> None:
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
    base_kwargs = {
        field_name: getattr(estimation_payload, field_name)
        for field_name in estimation_module.EstimationPayload.__dataclass_fields__
    }

    boolean_cases = (
        ("beta_hat", np.array([True, False], dtype=bool)),
        ("gamma_hat", np.array([True, False, True], dtype=bool)),
        ("fitted_f_valid", [True, False, True, False, True]),
        ("second_stage_prediction_valid", [True, False, True, False, True]),
        ("residual_valid", [True, False, True, False, True]),
        (
            "projection_x_valid",
            np.ones_like(estimation_payload.projection_x_valid, dtype=bool),
        ),
        ("f_hat_at_z0", [True, False]),
    )

    for field_name, replacement in boolean_cases:
        with pytest.raises(
            ValueError,
            match=f"{field_name} must be numeric, not boolean",
        ):
            estimation_module.EstimationPayload(
                **{**base_kwargs, field_name: replacement}
            )


def test_estimation_payload_rejects_string_numeric_evidence(
    eq31_manual_slice: dict[str, object],
) -> None:
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
    base_kwargs = {
        field_name: getattr(estimation_payload, field_name)
        for field_name in estimation_module.EstimationPayload.__dataclass_fields__
    }

    string_cases = (
        ("beta_hat", estimation_payload.beta_hat.astype(str)),
        ("gamma_hat", estimation_payload.gamma_hat.astype(str)),
        ("fitted_f_valid", estimation_payload.fitted_f_valid.astype(str)),
        (
            "second_stage_prediction_valid",
            estimation_payload.second_stage_prediction_valid.astype(str),
        ),
        ("residual_valid", estimation_payload.residual_valid.astype(str)),
        ("projection_x_valid", estimation_payload.projection_x_valid.astype(str)),
        ("f_hat_at_z0", estimation_payload.f_hat_at_z0.astype(str)),
        (
            "fitted_f_valid",
            _StringVectorLike(estimation_payload.fitted_f_valid.astype(str)),
        ),
        (
            "projection_x_valid",
            _StringMatrixLike(estimation_payload.projection_x_valid.astype(str)),
        ),
    )

    for field_name, replacement in string_cases:
        with pytest.raises(
            ValueError,
            match=f"{field_name} must be numeric, not boolean or string",
        ):
            estimation_module.EstimationPayload(
                **{**base_kwargs, field_name: replacement}
            )


def test_estimation_payload_rejects_nonfinite_numeric_values(
    eq31_manual_slice: dict[str, object],
) -> None:
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
    base_kwargs = {
        field_name: getattr(estimation_payload, field_name)
        for field_name in estimation_module.EstimationPayload.__dataclass_fields__
    }

    nonfinite_cases = (
        ("beta_hat", [np.nan, 0.0]),
        ("gamma_hat", [0.0, np.inf, 0.0]),
        ("fitted_f_valid", [0.0, 0.0, np.nan, 0.0, 0.0]),
        ("residual_valid", [0.0, 0.0, 0.0, np.inf, 0.0]),
        (
            "projection_x_valid",
            [
                [0.0, 0.0],
                [0.0, np.nan],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ),
        ("f_hat_at_z0", [np.inf, 0.0]),
    )

    for field_name, replacement in nonfinite_cases:
        with pytest.raises(ValueError, match=f"{field_name} must contain only finite"):
            estimation_module.EstimationPayload(
                **{**base_kwargs, field_name: replacement}
            )


def test_estimation_payload_binds_eq31_metadata_dimensions(
    eq31_manual_slice: dict[str, object],
) -> None:
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
    base_kwargs = {
        field_name: getattr(estimation_payload, field_name)
        for field_name in estimation_module.EstimationPayload.__dataclass_fields__
    }

    metadata_cases = (
        ("n_valid_obs", None, "n_valid_obs must be provided"),
        ("n_valid_obs", score_payload.x_valid.shape[0] + 1, "n_valid_obs must equal"),
        ("n_valid_obs", 1.5, "n_valid_obs must be an integer"),
        ("beta_dimension", None, "beta_dimension must be provided"),
        ("beta_dimension", -1, "beta_dimension must be non-negative"),
        (
            "beta_dimension",
            score_payload.x_valid.shape[1] + 1,
            "beta_hat must align with beta_dimension",
        ),
        ("basis_dimension_full", None, "basis_dimension_full must be provided"),
        (
            "basis_dimension_full",
            score_payload.basis_valid_full.shape[1] + 1,
            "gamma_hat must align with basis_dimension_full",
        ),
    )

    for key, replacement, message in metadata_cases:
        metadata = dict(estimation_payload.optimization_metadata)
        if replacement is None:
            metadata.pop(key, None)
        else:
            metadata[key] = replacement
        with pytest.raises(ValueError, match=message):
            estimation_module.EstimationPayload(
                **{**base_kwargs, "optimization_metadata": metadata}
            )

    stale_projection = np.column_stack(
        [estimation_payload.projection_x_valid, np.zeros(score_payload.x_valid.shape[0])]
    )
    with pytest.raises(ValueError, match="projection_x_valid must align"):
        estimation_module.EstimationPayload(
            **{**base_kwargs, "projection_x_valid": stale_projection}
        )

    with pytest.raises(
        ValueError,
        match="f_hat_at_z0 must contain at least one evaluation point",
    ):
        estimation_module.EstimationPayload(
            **{**base_kwargs, "f_hat_at_z0": np.empty(0, dtype=float)}
        )
