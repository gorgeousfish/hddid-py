from __future__ import annotations

import numpy as np

from ._inference_common import _PSD_EIGENVALUE_TOL


def _matrix_square_root_psd(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue < -_PSD_EIGENVALUE_TOL:
        raise ValueError("matrix square root requires positive semidefinite input")
    clipped = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(clipped))


def _bootstrap_suprema_summary(
    simulated_suprema: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float]:
    values = np.asarray(simulated_suprema, dtype=float).reshape(-1)
    return {
        "quantile_level": float(1.0 - alpha),
        "simulated_suprema_min": float(np.min(values)),
        "simulated_suprema_max": float(np.max(values)),
        "simulated_suprema_mean": float(np.mean(values)),
        "simulated_suprema_std": float(np.std(values)),
    }


def _validate_uniform_band_bootstrap_summary(
    metadata: dict[str, object],
    expected_summary: dict[str, float],
) -> None:
    for key, expected_value in expected_summary.items():
        if key not in metadata:
            raise ValueError(f"uniform_band metadata {key} must be provided")
        observed_value = float(metadata[key])  # type: ignore[arg-type]
        if not np.isclose(
            observed_value,
            expected_value,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(f"uniform_band metadata {key} must match current bootstrap draws")


def compute_uniform_band_critical_value(
    covariance_at_grid: np.ndarray,
    standardization: np.ndarray,
    *,
    alpha: float,
    n_boot: int,
    random_state: int,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """
    Compute the bootstrap critical value for a uniform confidence band.

    Based on Theorem 3 of Ning, Peng, and Tao (2020): simulates the supremum
    of a standardized Gaussian process over the evaluation grid.

    Parameters
    ----------
    covariance_at_grid : ndarray, shape (grid_size, grid_size)
        The covariance matrix of f_hat at the evaluation points.
    standardization : ndarray, shape (grid_size,)
        Pointwise standard deviations (sigma_z_hat) for standardization.
    alpha : float
        Significance level (e.g. 0.05 for 95% band).
    n_boot : int
        Number of bootstrap replications.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    critical_value : float
        The (1-alpha) quantile of simulated suprema.
    simulated_suprema : ndarray, shape (n_boot,)
        All simulated supremum values.
    bootstrap_summary : dict
        Summary statistics of the bootstrap distribution.
    """
    covariance_sqrt = _matrix_square_root_psd(covariance_at_grid)
    rng = np.random.default_rng(random_state)
    gaussian_draws = (
        rng.standard_normal(size=(n_boot, covariance_at_grid.shape[0]))
        @ covariance_sqrt.T
    )
    simulated_suprema = np.max(
        np.abs(gaussian_draws / standardization[None, :]),
        axis=1,
    )
    critical_value = float(np.quantile(simulated_suprema, 1.0 - alpha))
    bootstrap_summary = _bootstrap_suprema_summary(
        simulated_suprema,
        alpha=alpha,
    )
    return critical_value, simulated_suprema, bootstrap_summary
