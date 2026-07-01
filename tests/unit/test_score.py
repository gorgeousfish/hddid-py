from __future__ import annotations

import importlib

import numpy as np
import pytest


def _load_score_module():
    try:
        return importlib.import_module("hddid.score")
    except ModuleNotFoundError as exc:
        pytest.fail(f"hddid.score is missing: {exc}")


class _StringArrayLike:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=object)

    def __array__(self, dtype: object = None) -> np.ndarray:
        if dtype is None:
            return self._values
        return self._values.astype(dtype)


def test_build_score_payload_matches_eq31_formula(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    assert hasattr(score_module, "ScorePayload")
    assert hasattr(score_module, "build_score_payload")

    data = eq31_manual_slice["data"]
    nuisance_payload = eq31_manual_slice["nuisance_payload"]
    score_payload = score_module.build_score_payload(data, nuisance_payload)

    expected_delta_y = data.y1 - data.y0
    expected_s_hat = nuisance_payload.rho_hat * (
        expected_delta_y
        - (1.0 - nuisance_payload.pi_hat) * nuisance_payload.phi1_hat
        - nuisance_payload.pi_hat * nuisance_payload.phi0_hat
    )

    np.testing.assert_allclose(score_payload.delta_y, expected_delta_y)
    np.testing.assert_allclose(score_payload.s_hat, expected_s_hat)
    np.testing.assert_allclose(score_payload.s_hat, eq31_manual_slice["s_hat_full"])
    assert score_payload.basis_family == "polynomial"
    assert score_payload.basis_degree == 2
    assert score_payload.oracle_lane == "r-parity-polynomial"


def test_build_score_payload_rejects_stale_rho_hat_formula(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    nuisance_payload = eq31_manual_slice["nuisance_payload"]
    nuisance_module = importlib.import_module("hddid.nuisance")

    forged_rho_hat = np.asarray(nuisance_payload.rho_hat, dtype=float).copy()
    forged_rho_hat[0] += 1.0
    forged_payload = nuisance_module.NuisancePayload(
        pi_hat=nuisance_payload.pi_hat,
        phi0_hat=nuisance_payload.phi0_hat,
        phi1_hat=nuisance_payload.phi1_hat,
        rho_hat=forged_rho_hat,
        fold_ids=nuisance_payload.fold_ids,
        fold_diagnostics=nuisance_payload.fold_diagnostics,
        basis_family=nuisance_payload.basis_family,
        basis_degree=nuisance_payload.basis_degree,
        oracle_lane=nuisance_payload.oracle_lane,
        valid_mask=nuisance_payload.valid_mask,
    )

    with pytest.raises(
        ValueError,
        match=r"rho_hat must match the Eq\. \(2\.5\) propensity score weight",
    ):
        score_module.build_score_payload(eq31_manual_slice["data"], forged_payload)


def test_build_score_payload_keeps_valid_sample_alignment(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    data = eq31_manual_slice["data"]
    nuisance_payload = eq31_manual_slice["nuisance_payload"]
    valid_mask = eq31_manual_slice["valid_mask"]

    score_payload = score_module.build_score_payload(data, nuisance_payload)
    expected_basis_valid_full = data.build_basis_matrix()[valid_mask]

    np.testing.assert_array_equal(score_payload.valid_mask, valid_mask)
    np.testing.assert_allclose(
        score_payload.s_hat_valid,
        eq31_manual_slice["s_hat_full"][valid_mask],
    )
    np.testing.assert_allclose(score_payload.x_valid, data.x[valid_mask])
    np.testing.assert_allclose(
        score_payload.basis_valid_full,
        expected_basis_valid_full,
    )
    np.testing.assert_allclose(
        score_payload.basis_design_valid,
        expected_basis_valid_full[:, 1:],
    )
    np.testing.assert_allclose(
        score_payload.evaluation_basis,
        data.build_evaluation_basis(),
    )
    assert score_payload.s_hat_valid.shape[0] == score_payload.x_valid.shape[0]
    assert (
        score_payload.s_hat_valid.shape[0] == score_payload.basis_design_valid.shape[0]
    )


def test_score_payload_rejects_non_boolean_valid_mask_values(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )

    base_kwargs = {
        field_name: getattr(score_payload, field_name)
        for field_name in score_module.ScorePayload.__dataclass_fields__
    }

    for invalid_mask in (
        [True, 0, True, True, True, True],
        [True, "false", True, True, True, True],
    ):
        with pytest.raises(ValueError, match="valid_mask must contain only boolean"):
            score_module.ScorePayload(
                **{**base_kwargs, "valid_mask": invalid_mask}
            )


def test_score_payload_rejects_fractional_basis_degree_metadata(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    base_kwargs = {
        field_name: getattr(score_payload, field_name)
        for field_name in score_module.ScorePayload.__dataclass_fields__
    }

    with pytest.raises(ValueError, match="basis_degree must be an integer"):
        score_module.ScorePayload(
            **{**base_kwargs, "basis_degree": float(score_payload.basis_degree) + 0.5}
        )

    with pytest.raises(
        ValueError,
        match="basis_degree must be positive for trigonometric basis",
    ):
        score_module.ScorePayload(
            **{**base_kwargs, "basis_family": "trigonometric", "basis_degree": 0}
        )


def test_score_payload_rejects_invalid_fold_id_metadata(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    base_kwargs = {
        field_name: getattr(score_payload, field_name)
        for field_name in score_module.ScorePayload.__dataclass_fields__
    }

    invalid_cases = (
        [1.5] * score_payload.delta_y.shape[0],
        [True] * score_payload.delta_y.shape[0],
        [0] + [1] * (score_payload.delta_y.shape[0] - 1),
        [-1] + [1] * (score_payload.delta_y.shape[0] - 1),
    )
    for invalid_fold_ids in invalid_cases:
        with pytest.raises(
            ValueError,
            match="fold_ids must contain positive integer values",
        ):
            score_module.ScorePayload(
                **{**base_kwargs, "fold_ids": invalid_fold_ids}
            )


def test_score_payload_rejects_non_boolean_intercept_metadata(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    base_kwargs = {
        field_name: getattr(score_payload, field_name)
        for field_name in score_module.ScorePayload.__dataclass_fields__
    }

    for invalid_value in ("False", 0, 1, np.array(True)):
        with pytest.raises(
            ValueError,
            match="intercept_dropped_for_design must be a boolean",
        ):
            score_module.ScorePayload(
                **{**base_kwargs, "intercept_dropped_for_design": invalid_value}
            )


def test_score_payload_rejects_nonfinite_numeric_values(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )

    base_kwargs = {
        field_name: getattr(score_payload, field_name)
        for field_name in score_module.ScorePayload.__dataclass_fields__
    }

    nonfinite_cases = (
        ("delta_y", [0.0, np.nan, 0.3, 0.4, 0.5, 0.6]),
        ("s_hat", [0.0, 0.1, np.inf, 0.3, 0.4, 0.5]),
        ("s_hat_valid", [0.0, np.nan, 0.2, 0.3, 0.4, 0.5]),
        (
            "basis_matrix",
            [
                [1.0, 0.0],
                [1.0, 0.2],
                [1.0, np.inf],
                [1.0, 0.6],
                [1.0, 0.8],
                [1.0, 1.0],
            ],
        ),
        ("pi_hat", [0.25, 0.35, np.nan, 0.45, 0.55, 0.65]),
        ("rho_hat", [1.0, 0.5, 0.0, -0.5, np.inf, -1.0]),
    )

    for field_name, replacement in nonfinite_cases:
        with pytest.raises(ValueError, match=f"{field_name} must contain only finite"):
            score_module.ScorePayload(
                **{**base_kwargs, field_name: replacement}
            )


def test_score_payload_rejects_stale_eq31_score_evidence(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    base_kwargs = {
        field_name: getattr(score_payload, field_name)
        for field_name in score_module.ScorePayload.__dataclass_fields__
    }

    forged_s_hat = score_payload.s_hat.copy()
    forged_s_hat[0] += 10.0
    with pytest.raises(
        ValueError,
        match=r"s_hat must match the Eq\. \(3\.1\) score formula",
    ):
        score_module.ScorePayload(**{**base_kwargs, "s_hat": forged_s_hat})

    forged_s_hat_valid = score_payload.s_hat_valid.copy()
    forged_s_hat_valid[-1] += 10.0
    with pytest.raises(
        ValueError,
        match="s_hat_valid must match s_hat on valid_mask",
    ):
        score_module.ScorePayload(
            **{**base_kwargs, "s_hat_valid": forged_s_hat_valid}
        )

    with pytest.raises(ValueError, match="pi_hat must lie strictly between 0 and 1"):
        score_module.ScorePayload(
            **{**base_kwargs, "pi_hat": np.full(score_payload.pi_hat.shape, 1.0)}
        )


def test_score_payload_rejects_stale_eq31_basis_evidence(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )
    base_kwargs = {
        field_name: getattr(score_payload, field_name)
        for field_name in score_module.ScorePayload.__dataclass_fields__
    }

    forged_basis_valid = score_payload.basis_valid_full.copy()
    forged_basis_valid[:, -1] += 100.0
    with pytest.raises(
        ValueError,
        match=r"basis_valid_full must match the Eq\. \(3\.1\) basis construction",
    ):
        score_module.ScorePayload(
            **{**base_kwargs, "basis_valid_full": forged_basis_valid}
        )

    forged_basis_design = score_payload.basis_design_valid.copy()
    forged_basis_design[:, 0] += 100.0
    with pytest.raises(
        ValueError,
        match=r"basis_design_valid must match the Eq\. \(3\.1\) basis construction",
    ):
        score_module.ScorePayload(
            **{**base_kwargs, "basis_design_valid": forged_basis_design}
        )

    with pytest.raises(
        ValueError,
        match=(
            r"intercept_dropped_for_design must match the Eq\. \(3\.1\) "
            r"basis construction"
        ),
    ):
        score_module.ScorePayload(
            **{
                **base_kwargs,
                "intercept_dropped_for_design": (
                    not score_payload.intercept_dropped_for_design
                ),
            }
        )

    with pytest.raises(
        ValueError,
        match="evaluation_basis must contain at least one grid point",
    ):
        score_module.ScorePayload(
            **{
                **base_kwargs,
                "evaluation_basis": np.empty(
                    (0, score_payload.evaluation_basis.shape[1]),
                    dtype=float,
                ),
            }
        )


def test_score_payload_rejects_boolean_numeric_evidence(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )

    base_kwargs = {
        field_name: getattr(score_payload, field_name)
        for field_name in score_module.ScorePayload.__dataclass_fields__
    }

    boolean_cases = (
        ("delta_y", [True, False, True, False, True, False]),
        ("s_hat", [True, False, True, False, True, False]),
        ("s_hat_valid", [True, False, True, False, True]),
        ("basis_matrix", np.ones_like(score_payload.basis_matrix, dtype=bool)),
        ("x_valid", np.ones_like(score_payload.x_valid, dtype=bool)),
        ("basis_valid_full", np.ones_like(score_payload.basis_valid_full, dtype=bool)),
        (
            "basis_design_valid",
            np.ones_like(score_payload.basis_design_valid, dtype=bool),
        ),
        ("evaluation_basis", np.ones_like(score_payload.evaluation_basis, dtype=bool)),
        ("pi_hat", [True, False, True, False, True, False]),
        ("phi0_hat", [True, False, True, False, True, False]),
        ("phi1_hat", [True, False, True, False, True, False]),
        ("rho_hat", [True, False, True, False, True, False]),
    )

    for field_name, replacement in boolean_cases:
        with pytest.raises(
            ValueError,
            match=f"{field_name} must be numeric, not boolean",
        ):
            score_module.ScorePayload(
                **{**base_kwargs, field_name: replacement}
            )


def test_score_payload_rejects_string_numeric_evidence(
    eq31_manual_slice: dict[str, object],
) -> None:
    score_module = _load_score_module()
    score_payload = score_module.build_score_payload(
        eq31_manual_slice["data"],
        eq31_manual_slice["nuisance_payload"],
    )

    base_kwargs = {
        field_name: getattr(score_payload, field_name)
        for field_name in score_module.ScorePayload.__dataclass_fields__
    }

    string_cases = (
        ("delta_y", ["1.0"] * score_payload.delta_y.shape[0]),
        ("s_hat", ["1.0"] * score_payload.s_hat.shape[0]),
        ("s_hat_valid", ["1.0"] * score_payload.s_hat_valid.shape[0]),
        ("basis_matrix", score_payload.basis_matrix.astype(str)),
        ("x_valid", score_payload.x_valid.astype(str)),
        ("basis_valid_full", score_payload.basis_valid_full.astype(str)),
        ("basis_design_valid", score_payload.basis_design_valid.astype(str)),
        ("evaluation_basis", score_payload.evaluation_basis.astype(str)),
        ("s_hat_valid", _StringArrayLike(score_payload.s_hat_valid.astype(str))),
        (
            "basis_valid_full",
            _StringArrayLike(score_payload.basis_valid_full.astype(str)),
        ),
        ("pi_hat", ["0.5"] * score_payload.pi_hat.shape[0]),
        ("phi0_hat", ["1.0"] * score_payload.phi0_hat.shape[0]),
        ("phi1_hat", ["1.0"] * score_payload.phi1_hat.shape[0]),
        ("rho_hat", ["1.0"] * score_payload.rho_hat.shape[0]),
    )

    for field_name, replacement in string_cases:
        with pytest.raises(
            ValueError,
            match=f"{field_name} must be numeric, not boolean or string",
        ):
            score_module.ScorePayload(
                **{**base_kwargs, field_name: replacement}
            )
