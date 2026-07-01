import numpy as np
import pytest

from hddid.inputs import validate_inputs
from hddid.nuisance import (
    CrossfitNuisanceEstimator,
    NuisanceEstimator,
    NuisancePayload,
    NuisanceTrainingSupportError,
    _fit_logistic_irls,
)
from hddid.results import FoldDiagnostics
from hddid.splitting import CrossfitFold, CrossfitPlan, make_crossfit_splits


class _StringVectorLike:
    def __array__(self, dtype: object = None) -> np.ndarray:
        values = np.array(["0.25", "0.75"], dtype=object)
        if dtype is None:
            return values
        return values.astype(dtype)


def test_nuisance_payload_from_predictions_derives_rho_and_keeps_metadata() -> None:
    payload = NuisancePayload.from_predictions(
        pi_hat=[0.25, 0.75],
        phi0_hat=[1.0, 2.0],
        phi1_hat=[1.5, 2.5],
        treat=[0, 1],
        fold_ids=[1, 2],
        fold_diagnostics=[
            FoldDiagnostics(
                fold_id=1, n_holdout_raw=1, n_trimmed_propensity=0, n_valid_holdout=1
            ),
            FoldDiagnostics(
                fold_id=2, n_holdout_raw=1, n_trimmed_propensity=0, n_valid_holdout=1
            ),
        ],
        basis_family="polynomial",
        basis_degree=3,
        oracle_lane="r_parity",
    )

    expected_rho = np.array([-4.0 / 3.0, 4.0 / 3.0])
    np.testing.assert_allclose(payload.rho_hat, expected_rho)
    np.testing.assert_array_equal(payload.fold_ids, np.array([1, 2]))
    assert payload.basis_family == "polynomial"
    assert payload.basis_degree == 3
    assert payload.oracle_lane == "r-parity-polynomial"
    assert "valid_mask" in NuisancePayload.__dataclass_fields__
    np.testing.assert_array_equal(payload.valid_mask, np.array([True, True]))


def test_nuisance_payload_from_predictions_rejects_degenerate_propensity() -> None:
    with pytest.raises(ValueError, match="pi_hat"):
        NuisancePayload.from_predictions(
            pi_hat=[0.0, 0.75],
            phi0_hat=[1.0, 2.0],
            phi1_hat=[1.5, 2.5],
            treat=[0, 1],
            fold_ids=[1, 2],
            fold_diagnostics=[],
            basis_family="polynomial",
            basis_degree=3,
            oracle_lane="paper",
        )


def test_logistic_nuisance_predictions_do_not_mask_trim_boundary_extremes() -> None:
    train_features = np.array([[-20.0], [-10.0], [10.0], [20.0]], dtype=float)
    treat = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    prediction_features = np.array([[-20.0], [0.0], [20.0]], dtype=float)

    pi_hat = _fit_logistic_irls(train_features, treat, prediction_features)

    assert 0.0 < pi_hat[0] < 0.01
    assert pi_hat[1] == pytest.approx(0.5, abs=1e-6)
    assert 0.99 < pi_hat[2] < 1.0


def test_nuisance_payload_from_predictions_rejects_treat_alignment_drift() -> None:
    with pytest.raises(ValueError, match="treat must align with pi_hat"):
        NuisancePayload.from_predictions(
            pi_hat=[0.25, 0.75],
            phi0_hat=[1.0, 2.0],
            phi1_hat=[1.5, 2.5],
            treat=[1],
            fold_ids=[1, 2],
            fold_diagnostics=[],
            basis_family="polynomial",
            basis_degree=3,
            oracle_lane="r_parity",
        )


def test_nuisance_payload_from_predictions_rejects_treat_without_both_groups() -> None:
    base_kwargs = {
        "pi_hat": [0.25, 0.5, 0.75],
        "phi0_hat": [1.0, 2.0, 3.0],
        "phi1_hat": [1.5, 2.5, 3.5],
        "fold_ids": [1, 2, 3],
        "fold_diagnostics": [],
        "basis_family": "polynomial",
        "basis_degree": 3,
        "oracle_lane": "r_parity",
    }

    for treat in ([1, 1, 1], [0, 0, 0]):
        with pytest.raises(
            ValueError,
            match="at least one treated and one control observation",
        ):
            NuisancePayload.from_predictions(**base_kwargs, treat=treat)

    payload = NuisancePayload.from_predictions(**base_kwargs, treat=[1, 0, 1])
    np.testing.assert_allclose(payload.rho_hat, np.array([4.0, -2.0, 4.0 / 3.0]))


def test_crossfit_nuisance_estimator_rejects_missing_training_group_support() -> None:
    data = validate_inputs(
        y0=[0.0, 0.0, 0.0, 0.0],
        y1=[1.0, 2.0, 3.0, 4.0],
        treat=[1, 1, 0, 0],
        x=[[0.0], [1.0], [2.0], [3.0]],
        z=[0.0, 0.1, 0.2, 0.3],
        z0=0.15,
        basis_family="polynomial",
        basis_degree=1,
        alpha=0.1,
    )
    splits = CrossfitPlan(
        fold_ids=np.array([2, 2, 1, 1], dtype=int),
        folds=[
            CrossfitFold(
                fold_id=1,
                train_indices=np.array([0, 1], dtype=int),
                holdout_indices=np.array([2, 3], dtype=int),
                diagnostics=FoldDiagnostics(
                    fold_id=1,
                    n_holdout_raw=2,
                    n_trimmed_propensity=0,
                    n_valid_holdout=2,
                    trim_lower=0.0,
                    trim_upper=1.0,
                ),
            ),
            CrossfitFold(
                fold_id=2,
                train_indices=np.array([2, 3], dtype=int),
                holdout_indices=np.array([0, 1], dtype=int),
                diagnostics=FoldDiagnostics(
                    fold_id=2,
                    n_holdout_raw=2,
                    n_trimmed_propensity=0,
                    n_valid_holdout=2,
                    trim_lower=0.0,
                    trim_upper=1.0,
                ),
            ),
        ],
    )

    with pytest.raises(NuisanceTrainingSupportError) as exc_info:
        CrossfitNuisanceEstimator().fit(data, splits)

    assert exc_info.value.fold_id == 1
    assert exc_info.value.n_train_treated == 2
    assert exc_info.value.n_train_control == 0
    assert "separate Phi1/Phi0 nuisance fits" in str(exc_info.value)


def test_nuisance_payload_rejects_fractional_basis_degree_metadata() -> None:
    with pytest.raises(ValueError, match="basis_degree must be an integer"):
        NuisancePayload.from_predictions(
            pi_hat=[0.25, 0.75],
            phi0_hat=[1.0, 2.0],
            phi1_hat=[1.5, 2.5],
            treat=[0, 1],
            fold_ids=[1, 2],
            fold_diagnostics=[],
            basis_family="polynomial",
            basis_degree=3.5,
            oracle_lane="r_parity",
        )

    with pytest.raises(
        ValueError,
        match="basis_degree must be positive for trigonometric basis",
    ):
        NuisancePayload.from_predictions(
            pi_hat=[0.25, 0.75],
            phi0_hat=[1.0, 2.0],
            phi1_hat=[1.5, 2.5],
            treat=[0, 1],
            fold_ids=[1, 2],
            fold_diagnostics=[],
            basis_family="trigonometric",
            basis_degree=0,
            oracle_lane="paper",
        )


def test_nuisance_payload_rejects_invalid_fold_id_metadata() -> None:
    base_kwargs = {
        "pi_hat": [0.25, 0.75],
        "phi0_hat": [1.0, 2.0],
        "phi1_hat": [1.5, 2.5],
        "treat": [0, 1],
        "fold_diagnostics": [],
        "basis_family": "polynomial",
        "basis_degree": 3,
        "oracle_lane": "r_parity",
    }

    for invalid_fold_ids in ([1.5, 2.5], [True, False], [1, 0], [1, -2]):
        with pytest.raises(
            ValueError,
            match="fold_ids must contain positive integer values",
        ):
            NuisancePayload.from_predictions(
                **base_kwargs,
                fold_ids=invalid_fold_ids,
            )


class DummyNuisanceEstimator:
    def fit(self, data, splits) -> NuisancePayload:
        return NuisancePayload.from_predictions(
            pi_hat=np.full(data.n_obs, 0.5),
            phi0_hat=data.y0,
            phi1_hat=data.y1,
            treat=data.treat,
            fold_ids=splits.fold_ids,
            fold_diagnostics=[fold.diagnostics for fold in splits.folds],
            basis_family=data.basis_family,
            basis_degree=data.basis_degree,
            oracle_lane="r_parity",
        )


def test_nuisance_estimator_protocol_accepts_dummy_adapter(
    sample_hddid_inputs: dict[str, object],
) -> None:
    data = validate_inputs(**sample_hddid_inputs)
    splits = make_crossfit_splits(n_obs=data.n_obs, n_folds=3, random_state=7)
    adapter = DummyNuisanceEstimator()
    assert isinstance(adapter, NuisanceEstimator)

    payload = adapter.fit(data, splits)

    np.testing.assert_array_equal(payload.fold_ids, splits.fold_ids)
    assert len(payload.fold_diagnostics) == len(splits.folds)
    assert payload.oracle_lane == "r-parity-polynomial"


def test_nuisance_payload_rejects_oracle_lane_basis_mismatch() -> None:
    with pytest.raises(ValueError, match="oracle_lane"):
        NuisancePayload.from_predictions(
            pi_hat=[0.25, 0.75],
            phi0_hat=[1.0, 2.0],
            phi1_hat=[1.5, 2.5],
            treat=[0, 1],
            fold_ids=[1, 2],
            fold_diagnostics=[],
            basis_family="polynomial",
            basis_degree=3,
            oracle_lane="paper",
        )


def test_nuisance_payload_requires_valid_mask_alignment() -> None:
    assert "valid_mask" in NuisancePayload.__dataclass_fields__

    with pytest.raises(ValueError, match="valid_mask must align with pi_hat"):
        NuisancePayload.from_predictions(
            pi_hat=[0.25, 0.75],
            phi0_hat=[1.0, 2.0],
            phi1_hat=[1.5, 2.5],
            treat=[0, 1],
            fold_ids=[1, 2],
            valid_mask=[True],
            fold_diagnostics=[],
            basis_family="polynomial",
            basis_degree=3,
            oracle_lane="r_parity",
        )


def test_nuisance_payload_rejects_non_boolean_valid_mask_values() -> None:
    base_kwargs = {
        "pi_hat": [0.25, 0.75],
        "phi0_hat": [1.0, 2.0],
        "phi1_hat": [1.5, 2.5],
        "treat": [0, 1],
        "fold_ids": [1, 2],
        "fold_diagnostics": [],
        "basis_family": "polynomial",
        "basis_degree": 3,
        "oracle_lane": "r_parity",
    }

    for invalid_mask in ([True, np.nan], [True, 0.5], [True, "yes"]):
        with pytest.raises(ValueError, match="valid_mask must contain only boolean"):
            NuisancePayload.from_predictions(
                **base_kwargs,
                valid_mask=invalid_mask,
            )


def test_nuisance_payload_rejects_nonfinite_predictions() -> None:
    base_kwargs = {
        "pi_hat": [0.25, 0.75],
        "phi0_hat": [1.0, 2.0],
        "phi1_hat": [1.5, 2.5],
        "treat": [0, 1],
        "fold_ids": [1, 2],
        "fold_diagnostics": [],
        "basis_family": "polynomial",
        "basis_degree": 3,
        "oracle_lane": "r_parity",
    }

    for field_name, replacement in (
        ("pi_hat", [0.25, np.nan]),
        ("phi0_hat", [1.0, np.inf]),
        ("phi1_hat", [np.nan, 2.5]),
    ):
        with pytest.raises(ValueError, match=f"{field_name} must contain only finite"):
            NuisancePayload.from_predictions(
                **{**base_kwargs, field_name: replacement}
            )


def test_nuisance_payload_rejects_boolean_numeric_evidence() -> None:
    base_kwargs = {
        "pi_hat": [0.25, 0.75],
        "phi0_hat": [1.0, 2.0],
        "phi1_hat": [1.5, 2.5],
        "rho_hat": [-4.0 / 3.0, 4.0 / 3.0],
        "fold_ids": [1, 2],
        "fold_diagnostics": [],
        "basis_family": "polynomial",
        "basis_degree": 3,
        "oracle_lane": "r_parity",
    }

    for field_name, replacement in (
        ("pi_hat", [True, False]),
        ("phi0_hat", [True, False]),
        ("phi1_hat", [False, True]),
        ("rho_hat", [True, False]),
    ):
        with pytest.raises(
            ValueError,
            match=f"{field_name} must be numeric, not boolean",
        ):
            NuisancePayload(**{**base_kwargs, field_name: replacement})


def test_nuisance_payload_rejects_string_numeric_evidence() -> None:
    base_kwargs = {
        "pi_hat": [0.25, 0.75],
        "phi0_hat": [1.0, 2.0],
        "phi1_hat": [1.5, 2.5],
        "rho_hat": [-4.0 / 3.0, 4.0 / 3.0],
        "fold_ids": [1, 2],
        "fold_diagnostics": [],
        "basis_family": "polynomial",
        "basis_degree": 3,
        "oracle_lane": "r_parity",
    }

    for field_name, replacement in (
        ("pi_hat", ["0.25", "0.75"]),
        ("phi0_hat", ["1.0", "2.0"]),
        ("phi1_hat", ["1.5", "2.5"]),
        ("rho_hat", ["-1.3333333333333333", "1.3333333333333333"]),
        ("pi_hat", _StringVectorLike()),
    ):
        with pytest.raises(
            ValueError,
            match=f"{field_name} must be numeric, not boolean or string",
        ):
            NuisancePayload(**{**base_kwargs, field_name: replacement})


def test_nuisance_payload_from_predictions_rejects_boolean_numeric_evidence() -> None:
    base_kwargs = {
        "pi_hat": [0.25, 0.75],
        "phi0_hat": [1.0, 2.0],
        "phi1_hat": [1.5, 2.5],
        "treat": [0, 1],
        "fold_ids": [1, 2],
        "fold_diagnostics": [],
        "basis_family": "polynomial",
        "basis_degree": 3,
        "oracle_lane": "r_parity",
    }

    for field_name, replacement in (
        ("pi_hat", [True, False]),
        ("phi0_hat", [True, False]),
        ("phi1_hat", [False, True]),
        ("treat", [False, True]),
    ):
        with pytest.raises(
            ValueError,
            match=f"{field_name} must be numeric, not boolean",
        ):
            NuisancePayload.from_predictions(
                **{**base_kwargs, field_name: replacement}
            )


def test_nuisance_payload_from_predictions_rejects_string_numeric_evidence() -> None:
    base_kwargs = {
        "pi_hat": [0.25, 0.75],
        "phi0_hat": [1.0, 2.0],
        "phi1_hat": [1.5, 2.5],
        "treat": [0, 1],
        "fold_ids": [1, 2],
        "fold_diagnostics": [],
        "basis_family": "polynomial",
        "basis_degree": 3,
        "oracle_lane": "r_parity",
    }

    for field_name, replacement in (
        ("pi_hat", ["0.25", "0.75"]),
        ("phi0_hat", ["1.0", "2.0"]),
        ("phi1_hat", ["1.5", "2.5"]),
        ("treat", ["0", "1"]),
        ("pi_hat", _StringVectorLike()),
    ):
        with pytest.raises(
            ValueError,
            match=f"{field_name} must be numeric, not boolean or string",
        ):
            NuisancePayload.from_predictions(
                **{**base_kwargs, field_name: replacement}
            )
