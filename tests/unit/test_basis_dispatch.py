import numpy as np
import pytest

from hddid.basis import build_sieve_basis


def test_build_sieve_basis_routes_polynomial_family() -> None:
    z = np.array([0.0, 2.0])

    basis = build_sieve_basis(z, basis_family="polynomial", degree=2)

    expected = np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 4.0]])
    np.testing.assert_allclose(basis, expected)


def test_build_sieve_basis_routes_trigonometric_family() -> None:
    z = np.array([0.0, 0.25])

    basis = build_sieve_basis(z, basis_family="trigonometric", degree=1)

    expected = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    np.testing.assert_allclose(basis, expected, atol=1e-12)


def test_build_sieve_basis_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="basis_family"):
        build_sieve_basis(np.array([0.0, 1.0]), basis_family="tri", degree=1)


@pytest.mark.parametrize(
    ("basis_family", "degree"),
    [
        ("polynomial", True),
        ("polynomial", 2.5),
        ("trigonometric", True),
        ("trigonometric", 1.5),
    ],
)
def test_build_sieve_basis_rejects_non_integer_degree(
    basis_family: str, degree: object
) -> None:
    with pytest.raises(ValueError, match="degree must be an integer"):
        build_sieve_basis(
            np.array([0.0, 1.0]),
            basis_family=basis_family,
            degree=degree,  # type: ignore[arg-type]
        )
