from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from hddid.basis import polynomial_sieve_basis, trigonometric_sieve_basis


REPO_ROOT = Path(__file__).resolve().parents[3]
R_HELPER_FILE = REPO_ROOT / "hddid-r" / "R" / "Examplehighdimdiffindiff.R"
MACHINE_ATOL = 32.0 * np.finfo(float).eps


def _run_r_basis(function_name: str, z: np.ndarray, degree: int) -> np.ndarray:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is required for parity tests")

    z_values = ", ".join(format(float(value), ".17g") for value in z)
    script = "\n".join(
        [
            f'source("{R_HELPER_FILE.as_posix()}")',
            f"x <- c({z_values})",
            f"basis <- {function_name}(x, {degree})",
            'cat(nrow(basis), ncol(basis), "\\n")',
            'cat(paste(formatC(c(t(basis)), digits=17, format="fg"), collapse=" "))',
        ]
    )
    result = subprocess.run(
        [rscript, "--vanilla", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().splitlines()
    rows, cols = map(int, lines[0].split())
    values = np.fromstring(lines[1], sep=" ")
    return values.reshape(rows, cols)


def test_polynomial_sieve_basis_matches_r_snapshot_helper() -> None:
    z = np.array([0.0, 0.125, -0.3, 0.75])

    expected = _run_r_basis("sieve.Pol", z, degree=3)
    actual = polynomial_sieve_basis(z, degree=3)

    np.testing.assert_allclose(actual, expected, atol=MACHINE_ATOL, rtol=0.0)


def test_polynomial_sieve_basis_matches_r_snapshot_helper_for_zero_degree() -> None:
    z = np.array([0.0, 0.125, -0.3, 0.75])

    expected = _run_r_basis("sieve.Pol", z, degree=0)
    actual = polynomial_sieve_basis(z, degree=0)

    np.testing.assert_allclose(actual, expected, atol=MACHINE_ATOL, rtol=0.0)


def test_trigonometric_sieve_basis_matches_r_snapshot_helper() -> None:
    z = np.array([0.11, 0.37, -0.29, 0.93])

    expected = _run_r_basis("sieve.TriPol", z, degree=4)
    actual = trigonometric_sieve_basis(z, degree=4)

    np.testing.assert_allclose(actual, expected, atol=MACHINE_ATOL, rtol=0.0)
