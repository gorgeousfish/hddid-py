import numpy as np
import pytest

from hddid.basis import polynomial_sieve_basis


def test_polynomial_sieve_basis_includes_intercept_and_powers() -> None:
    z = np.array([0.0, 0.5, 2.0])

    basis = polynomial_sieve_basis(z, degree=3)

    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, 0.25, 0.125],
            [1.0, 2.0, 4.0, 8.0],
        ]
    )
    np.testing.assert_allclose(basis, expected)


def test_polynomial_sieve_basis_accepts_column_vector_input() -> None:
    z = np.array([[1.0], [3.0]])

    basis = polynomial_sieve_basis(z, degree=2)

    expected = np.array([[1.0, 1.0, 1.0], [1.0, 3.0, 9.0]])
    np.testing.assert_allclose(basis, expected)


def test_polynomial_sieve_basis_allows_zero_degree_for_intercept_only_basis() -> None:
    z = np.array([0.0, 1.0, -2.5])

    basis = polynomial_sieve_basis(z, degree=0)

    expected = np.ones((3, 1))
    np.testing.assert_allclose(basis, expected)


def test_polynomial_sieve_basis_rejects_negative_degree() -> None:
    with pytest.raises(ValueError, match="degree must be non-negative"):
        polynomial_sieve_basis(np.array([0.0, 1.0]), degree=-1)


@pytest.mark.parametrize("degree", [True, 2.5])
def test_polynomial_sieve_basis_rejects_non_integer_degree(degree: object) -> None:
    with pytest.raises(ValueError, match="degree must be an integer"):
        polynomial_sieve_basis(np.array([0.0, 1.0]), degree=degree)  # type: ignore[arg-type]


def test_polynomial_sieve_basis_rejects_multivariate_z() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        polynomial_sieve_basis(np.ones((3, 2)), degree=2)


def test_polynomial_sieve_basis_rejects_empty_z() -> None:
    with pytest.raises(ValueError, match="z must contain at least one coordinate"):
        polynomial_sieve_basis(np.array([], dtype=float), degree=2)


def test_polynomial_sieve_basis_rejects_overflowed_basis_columns() -> None:
    with pytest.raises(
        ValueError,
        match="polynomial basis must contain only finite values",
    ):
        polynomial_sieve_basis(np.array([1e308], dtype=float), degree=2)
