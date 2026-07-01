import numpy as np
import pytest

from hddid.splitting import CrossfitPlan, make_crossfit_splits


def test_make_crossfit_splits_is_deterministic_and_preserves_guard_fields(
    sample_hddid_inputs: dict[str, object],
) -> None:
    n_obs = len(sample_hddid_inputs["y0"])
    first = make_crossfit_splits(n_obs=n_obs, n_folds=3, random_state=123)
    second = make_crossfit_splits(n_obs=n_obs, n_folds=3, random_state=123)

    assert isinstance(first, CrossfitPlan)
    np.testing.assert_array_equal(first.fold_ids, second.fold_ids)
    np.testing.assert_array_equal(np.sort(first.fold_ids), np.array([1, 1, 2, 2, 3, 3]))
    assert len(first.folds) == 3

    reconstructed_holdout = np.concatenate(
        [fold.holdout_indices for fold in first.folds]
    )
    np.testing.assert_array_equal(np.sort(reconstructed_holdout), np.arange(n_obs))

    for fold in first.folds:
        assert fold.train_indices.ndim == 1
        assert fold.holdout_indices.ndim == 1
        assert fold.diagnostics.fold_id == fold.fold_id
        assert fold.diagnostics.n_holdout_raw == fold.holdout_indices.size
        assert fold.diagnostics.n_trimmed_propensity == 0
        assert fold.diagnostics.n_valid_holdout == fold.holdout_indices.size
        assert fold.diagnostics.trim_lower == pytest.approx(0.01)
        assert fold.diagnostics.trim_upper == pytest.approx(0.99)


def test_make_crossfit_splits_rejects_more_folds_than_observations() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        make_crossfit_splits(n_obs=3, n_folds=4, random_state=0)


@pytest.mark.parametrize("random_state", [True, 1.5, -1])
def test_make_crossfit_splits_rejects_invalid_random_state(
    random_state: object,
) -> None:
    with pytest.raises(ValueError, match="random_state"):
        make_crossfit_splits(
            n_obs=6,
            n_folds=3,
            random_state=random_state,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("trim_lower", "trim_upper"),
    [
        (False, 0.99),
        (0.01, True),
        (np.bool_(False), 0.99),
        (0.01, np.bool_(True)),
        ("0.01", 0.99),
        (0.01, "0.99"),
        (b"0.01", 0.99),
        (0.01, np.bytes_(b"0.99")),
        (np.str_("0.01"), 0.99),
        (np.array("0.01"), 0.99),
        (0.01, np.array("0.99")),
        ([0.01], 0.99),
        (0.01, [0.99]),
        (np.array([0.01]), 0.99),
        (0.01, np.array([0.99])),
        (float("nan"), 0.99),
        (0.01, float("inf")),
    ],
)
def test_make_crossfit_splits_rejects_invalid_trim_bounds(
    trim_lower: object, trim_upper: object
) -> None:
    with pytest.raises(ValueError, match="trim"):
        make_crossfit_splits(
            n_obs=6,
            n_folds=3,
            random_state=0,
            trim_lower=trim_lower,  # type: ignore[arg-type]
            trim_upper=trim_upper,  # type: ignore[arg-type]
        )
