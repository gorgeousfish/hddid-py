from __future__ import annotations

import numpy as np
import pytest

import hddid.nuisance as nuisance_module
from hddid.inputs import validate_inputs
from hddid.results import FoldDiagnostics
from hddid.splitting import CrossfitFold, CrossfitPlan, make_crossfit_splits


def test_crossfit_nuisance_estimator_returns_full_length_payload(
    sample_hddid_inputs: dict[str, object],
) -> None:
    estimator_cls = getattr(nuisance_module, "CrossfitNuisanceEstimator", None)
    assert estimator_cls is not None

    data = validate_inputs(**sample_hddid_inputs)
    splits = make_crossfit_splits(
        n_obs=data.n_obs,
        n_folds=3,
        random_state=7,
        trim_lower=0.0,
        trim_upper=1.0,
    )

    payload = estimator_cls().fit(data, splits)

    assert payload.pi_hat.shape == (data.n_obs,)
    assert payload.phi0_hat.shape == (data.n_obs,)
    assert payload.phi1_hat.shape == (data.n_obs,)
    assert payload.valid_mask.shape == (data.n_obs,)
    assert payload.valid_mask.dtype == np.bool_
    np.testing.assert_array_equal(payload.fold_ids, splits.fold_ids)
    assert len(payload.fold_diagnostics) == len(splits.folds)
    assert all(diag.n_valid_holdout is not None for diag in payload.fold_diagnostics)


def test_crossfit_nuisance_estimator_changes_with_split_seed(
    sample_hddid_inputs: dict[str, object],
) -> None:
    estimator_cls = getattr(nuisance_module, "CrossfitNuisanceEstimator", None)
    assert estimator_cls is not None

    data = validate_inputs(**sample_hddid_inputs)
    splits_a = make_crossfit_splits(
        n_obs=data.n_obs,
        n_folds=3,
        random_state=3,
        trim_lower=0.0,
        trim_upper=1.0,
    )
    splits_b = make_crossfit_splits(
        n_obs=data.n_obs,
        n_folds=3,
        random_state=19,
        trim_lower=0.0,
        trim_upper=1.0,
    )

    payload_a = estimator_cls().fit(data, splits_a)
    payload_b = estimator_cls().fit(data, splits_b)

    assert not np.array_equal(payload_a.fold_ids, payload_b.fold_ids)
    assert payload_a.pi_hat.shape == payload_b.pi_hat.shape
    assert payload_a.valid_mask.shape == payload_b.valid_mask.shape


def test_crossfit_nuisance_estimator_keeps_fold_predictions_holdout_safe(
    sample_hddid_inputs: dict[str, object],
) -> None:
    estimator_cls = getattr(nuisance_module, "CrossfitNuisanceEstimator", None)
    assert estimator_cls is not None

    data = validate_inputs(**sample_hddid_inputs)
    splits = make_crossfit_splits(
        n_obs=data.n_obs,
        n_folds=3,
        random_state=7,
        trim_lower=0.0,
        trim_upper=1.0,
    )
    reference_payload = estimator_cls().fit(data, splits)

    first_fold = splits.folds[0]
    holdout_indices = first_fold.holdout_indices
    mutated_inputs = dict(sample_hddid_inputs)
    mutated_y0 = np.asarray(sample_hddid_inputs["y0"], dtype=float).copy()
    mutated_y1 = np.asarray(sample_hddid_inputs["y1"], dtype=float).copy()
    mutated_y0[holdout_indices] += 100.0
    mutated_y1[holdout_indices] -= 75.0
    mutated_inputs["y0"] = mutated_y0
    mutated_inputs["y1"] = mutated_y1

    mutated_data = validate_inputs(**mutated_inputs)
    mutated_payload = estimator_cls().fit(mutated_data, splits)

    np.testing.assert_allclose(
        reference_payload.pi_hat[holdout_indices],
        mutated_payload.pi_hat[holdout_indices],
    )
    np.testing.assert_allclose(
        reference_payload.phi0_hat[holdout_indices],
        mutated_payload.phi0_hat[holdout_indices],
    )
    np.testing.assert_allclose(
        reference_payload.phi1_hat[holdout_indices],
        mutated_payload.phi1_hat[holdout_indices],
    )
    np.testing.assert_array_equal(
        reference_payload.valid_mask[holdout_indices],
        mutated_payload.valid_mask[holdout_indices],
    )


def test_crossfit_nuisance_estimator_checks_training_support_before_trim() -> None:
    estimator_cls = getattr(nuisance_module, "CrossfitNuisanceEstimator", None)
    error_cls = getattr(nuisance_module, "NuisanceTrainingSupportError", None)
    assert estimator_cls is not None
    assert error_cls is not None

    data = validate_inputs(
        y0=[0.0, 0.0, 0.0, 0.0],
        y1=[1.0, 1.0, 1.0, 1.0],
        treat=[1, 1, 0, 0],
        x=[[10.0], [11.0], [-10.0], [-11.0]],
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
                    trim_lower=0.01,
                    trim_upper=0.99,
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
                    trim_lower=0.01,
                    trim_upper=0.99,
                ),
            ),
        ],
    )

    with pytest.raises(error_cls) as exc_info:
        estimator_cls().fit(data, splits)

    assert exc_info.value.fold_id == 1
    assert exc_info.value.n_train_treated == 2
    assert exc_info.value.n_train_control == 0
