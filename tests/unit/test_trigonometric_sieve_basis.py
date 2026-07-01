import numpy as np
import pytest

from hddid.basis import polynomial_sieve_basis, trigonometric_sieve_basis


class _StringArrayLike:
    def __array__(self, dtype: object = None) -> np.ndarray:
        values = np.array(["0.0", "0.5"], dtype=object)
        if dtype is None:
            return values
        return values.astype(dtype)


def test_trigonometric_sieve_basis_matches_r_helper_ordering() -> None:
    z = np.array([0.0, 0.25])

    basis = trigonometric_sieve_basis(z, degree=2)

    expected = np.array(
        [
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, -1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(basis, expected, atol=1e-12)


def test_trigonometric_sieve_basis_accepts_column_vector_input() -> None:
    z = np.array([[0.0], [0.5]])

    basis = trigonometric_sieve_basis(z, degree=1)

    expected = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    np.testing.assert_allclose(basis, expected, atol=1e-12)


def test_trigonometric_sieve_basis_rejects_non_positive_degree() -> None:
    with pytest.raises(ValueError, match="degree must be at least 1"):
        trigonometric_sieve_basis(np.array([0.0, 0.5]), degree=0)


@pytest.mark.parametrize("degree", [True, 1.5])
def test_trigonometric_sieve_basis_rejects_non_integer_degree(degree: object) -> None:
    with pytest.raises(ValueError, match="degree must be an integer"):
        trigonometric_sieve_basis(np.array([0.0, 0.5]), degree=degree)  # type: ignore[arg-type]


def test_trigonometric_sieve_basis_rejects_multivariate_z() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        trigonometric_sieve_basis(np.ones((2, 2)), degree=1)


def test_trigonometric_sieve_basis_rejects_empty_z() -> None:
    with pytest.raises(ValueError, match="z must contain at least one coordinate"):
        trigonometric_sieve_basis(np.array([], dtype=float), degree=1)


@pytest.mark.parametrize(
    "z",
    (
        [True, False],
        ["0.0", "0.5"],
        [b"0.0", b"0.5"],
        np.array(["0.0", "0.5"]),
        np.array([b"0.0", b"0.5"]),
        _StringArrayLike(),
    ),
)
def test_public_sieve_basis_rejects_boolean_or_string_numeric_evidence(
    z: object,
) -> None:
    for basis_function in (polynomial_sieve_basis, trigonometric_sieve_basis):
        with pytest.raises(ValueError, match="z must be numeric, not boolean or string"):
            basis_function(z, degree=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("z", (np.nan, np.inf, [0.0, np.nan], [0.0, np.inf]))
def test_public_sieve_basis_rejects_nonfinite_z(z: object) -> None:
    for basis_function in (polynomial_sieve_basis, trigonometric_sieve_basis):
        with pytest.raises(ValueError, match="z must contain only finite values"):
            basis_function(z, degree=1)  # type: ignore[arg-type]


def test_trigonometric_sieve_basis_rejects_overflowed_angle_columns() -> None:
    with pytest.raises(
        ValueError,
        match="trigonometric basis must contain only finite values",
    ):
        trigonometric_sieve_basis(np.array([np.finfo(float).max], dtype=float), degree=1)
