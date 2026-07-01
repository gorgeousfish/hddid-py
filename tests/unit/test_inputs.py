import numpy as np
import pytest

from hddid.inputs import ValidatedHDDIDData, validate_inputs


class _StringVectorLike:
    def __array__(self, dtype: object = None) -> np.ndarray:
        values = np.array(["0.0", "1.0", "2.0", "3.0", "4.0", "5.0"], dtype=object)
        if dtype is None:
            return values
        return values.astype(dtype)


class _StringMatrixLike:
    def __array__(self, dtype: object = None) -> np.ndarray:
        values = np.array([["1.0", "0.0"]] * 6, dtype=object)
        if dtype is None:
            return values
        return values.astype(dtype)


def test_validate_inputs_normalizes_arrays_and_basis_configuration(
    sample_hddid_inputs: dict[str, object],
) -> None:
    validated = validate_inputs(
        **{**sample_hddid_inputs, "basis_family": "TrIgOnOmEtRiC"}
    )

    assert isinstance(validated, ValidatedHDDIDData)
    assert validated.basis_family == "trigonometric"
    assert validated.basis_degree == 2
    assert validated.alpha == pytest.approx(0.1)
    assert validated.z.shape == (6,)
    assert validated.z0.shape == (1,)
    np.testing.assert_array_equal(validated.treat, np.array([0, 1, 0, 1, 0, 1]))
    assert validated.build_basis_matrix().shape == (6, 5)
    assert validated.build_evaluation_basis().shape == (1, 5)


def test_validate_inputs_rejects_misaligned_sample_sizes() -> None:
    with pytest.raises(ValueError, match="same number of observations"):
        validate_inputs(
            y0=[0.0, 1.0],
            y1=[0.5],
            treat=[0, 1],
            x=[[1.0], [2.0]],
            z=[0.0, 0.5],
            z0=[0.25],
            basis_family="polynomial",
            basis_degree=2,
            alpha=0.1,
        )


def test_validate_inputs_rejects_empty_raw_sample() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        validate_inputs(
            y0=np.array([]),
            y1=np.array([]),
            treat=np.array([]),
            x=np.empty((0, 1)),
            z=np.array([]),
            z0=[0.25],
            basis_family="polynomial",
            basis_degree=2,
            alpha=0.1,
        )


def test_validate_inputs_rejects_empty_evaluation_grid(
    sample_hddid_inputs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="at least one evaluation point"):
        validate_inputs(**{**sample_hddid_inputs, "z0": []})


def test_validate_inputs_allows_zero_linear_covariates(
    sample_hddid_inputs: dict[str, object],
) -> None:
    validated = validate_inputs(
        **{
            **sample_hddid_inputs,
            "x": np.empty((len(sample_hddid_inputs["y0"]), 0)),
        }
    )

    assert validated.x.shape == (len(sample_hddid_inputs["y0"]), 0)
    assert validated.build_basis_matrix().shape[0] == validated.n_obs
    assert validated.build_evaluation_basis().shape[0] == 1


def test_validate_inputs_rejects_non_binary_treatment(
    sample_hddid_inputs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="binary"):
        validate_inputs(**{**sample_hddid_inputs, "treat": [0, 2, 0, 1, 0, 1]})


def test_validate_inputs_rejects_treatment_without_both_groups(
    sample_hddid_inputs: dict[str, object],
) -> None:
    for treat in (
        np.zeros(len(sample_hddid_inputs["treat"]), dtype=float),
        np.ones(len(sample_hddid_inputs["treat"]), dtype=float),
    ):
        with pytest.raises(
            ValueError,
            match="at least one treated and one control observation",
        ):
            validate_inputs(**{**sample_hddid_inputs, "treat": treat})


def test_validate_inputs_rejects_multivariate_z_in_phase_two() -> None:
    with pytest.raises(ValueError, match="one-dimensional z"):
        validate_inputs(
            y0=[0.0, 1.0],
            y1=[0.5, 1.5],
            treat=[0, 1],
            x=[[1.0], [2.0]],
            z=[[0.0, 0.1], [0.5, 0.6]],
            z0=[[0.25, 0.3]],
            basis_family="polynomial",
            basis_degree=2,
            alpha=0.1,
        )


def test_validate_inputs_rejects_unknown_basis_family(
    sample_hddid_inputs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="basis_family"):
        validate_inputs(**{**sample_hddid_inputs, "basis_family": "wavelet"})


def test_validate_inputs_rejects_alpha_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="alpha"):
        validate_inputs(
            y0=[0.0, 1.0],
            y1=[0.5, 1.5],
            treat=[0, 1],
            x=[[1.0], [2.0]],
            z=[0.0, 0.5],
            z0=[0.25],
            basis_family="polynomial",
            basis_degree=2,
            alpha=1.0,
        )


def test_validate_inputs_rejects_nonfinite_numeric_values(
    sample_hddid_inputs: dict[str, object],
) -> None:
    for field_name, replacement in (
        ("y0", [0.0, np.nan, 2.0, 3.0, 4.0, 5.0]),
        ("y1", [0.2, 1.2, np.inf, 3.6, 4.8, 6.0]),
        ("treat", [0.0, 1.0, np.nan, 1.0, 0.0, 1.0]),
        (
            "x",
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, np.nan],
                [1.0, 4.0],
                [1.0, 5.0],
            ],
        ),
        ("z", [0.0, 0.2, 0.4, np.inf, 0.8, 1.0]),
        ("z0", [np.nan]),
    ):
        with pytest.raises(ValueError, match=f"{field_name} must contain only finite"):
            validate_inputs(**{**sample_hddid_inputs, field_name: replacement})


def test_validate_inputs_rejects_boolean_raw_numeric_evidence(
    sample_hddid_inputs: dict[str, object],
) -> None:
    boolean_cases = (
        ("y0", [False, True, False, True, False, True]),
        ("y1", [True, False, True, False, True, False]),
        ("treat", [False, True, False, True, False, True]),
        ("x", np.ones_like(np.asarray(sample_hddid_inputs["x"], dtype=float), dtype=bool)),
        ("z", [False, True, False, True, False, True]),
        ("z0", [True]),
        ("alpha", True),
    )

    for field_name, replacement in boolean_cases:
        with pytest.raises(
            ValueError,
            match=f"{field_name} must be numeric, not boolean",
        ):
            validate_inputs(**{**sample_hddid_inputs, field_name: replacement})


def test_validate_inputs_rejects_string_raw_numeric_evidence(
    sample_hddid_inputs: dict[str, object],
) -> None:
    string_cases = (
        ("y0", ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0"]),
        ("y1", ["0.2", "1.2", "2.4", "3.6", "4.8", "6.0"]),
        ("treat", ["0", "1", "0", "1", "0", "1"]),
        ("x", [["1.0", "0.0"]] * len(sample_hddid_inputs["y0"])),
        ("z", ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]),
        ("z0", ["0.25"]),
        ("y0", _StringVectorLike()),
        ("x", _StringMatrixLike()),
    )

    for field_name, replacement in string_cases:
        with pytest.raises(
            ValueError,
            match=f"{field_name} must be numeric, not boolean or string",
        ):
            validate_inputs(**{**sample_hddid_inputs, field_name: replacement})

    with pytest.raises(ValueError, match="alpha must be numeric, not string"):
        validate_inputs(**{**sample_hddid_inputs, "alpha": "0.1"})
