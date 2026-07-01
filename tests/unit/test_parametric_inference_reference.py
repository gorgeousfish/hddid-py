from __future__ import annotations

from statistics import NormalDist

import numpy as np


def test_eq41_reference_slice_recovers_sigma_and_score_moment(
    eq41_parametric_reference_slice: dict[str, np.ndarray | float],
) -> None:
    projection_x_valid = np.asarray(
        eq41_parametric_reference_slice["projection_x_valid"],
        dtype=float,
    )
    residual_valid = np.asarray(
        eq41_parametric_reference_slice["residual_valid"],
        dtype=float,
    )

    expected_sigma = (
        projection_x_valid.T @ projection_x_valid / projection_x_valid.shape[0]
    )
    expected_score_moment = np.mean(
        residual_valid[:, None] * projection_x_valid,
        axis=0,
    )

    np.testing.assert_allclose(
        eq41_parametric_reference_slice["sigma_tilde_x_hat"],
        expected_sigma,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        eq41_parametric_reference_slice["score_moment"],
        expected_score_moment,
        atol=1e-12,
        rtol=0.0,
    )


def test_eq41_componentwise_r_mapping_matches_paper_special_case(
    eq41_parametric_reference_slice: dict[str, np.ndarray | float],
) -> None:
    sigma_tilde_x_hat = np.asarray(
        eq41_parametric_reference_slice["sigma_tilde_x_hat"],
        dtype=float,
    )
    score_moment = np.asarray(
        eq41_parametric_reference_slice["score_moment"],
        dtype=float,
    )
    beta_hat = np.asarray(eq41_parametric_reference_slice["beta_hat"], dtype=float)
    xi_matrix = np.asarray(eq41_parametric_reference_slice["xi_matrix"], dtype=float)
    w_matrix = np.asarray(eq41_parametric_reference_slice["w_matrix"], dtype=float)

    covinv = np.linalg.inv(sigma_tilde_x_hat)
    expected_w_matrix = -(covinv @ xi_matrix.T).T
    expected_componentwise = beta_hat + score_moment @ covinv
    paper_componentwise = beta_hat - np.sum(
        w_matrix[: beta_hat.shape[0]] * score_moment[None, :],
        axis=1,
    )

    np.testing.assert_allclose(w_matrix, expected_w_matrix, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        eq41_parametric_reference_slice["r_componentwise_xdebias"],
        expected_componentwise,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        paper_componentwise, expected_componentwise, atol=1e-12, rtol=0.0
    )


def test_eq41_reference_slice_confidence_intervals_use_scalar_variance_over_n(
    eq41_parametric_reference_slice: dict[str, np.ndarray | float],
) -> None:
    w_matrix = np.asarray(eq41_parametric_reference_slice["w_matrix"], dtype=float)
    omega_beta_hat = np.asarray(
        eq41_parametric_reference_slice["omega_beta_hat"],
        dtype=float,
    )
    t_hat = np.asarray(eq41_parametric_reference_slice["t_hat"], dtype=float)
    standard_errors = np.asarray(
        eq41_parametric_reference_slice["standard_errors"],
        dtype=float,
    )
    ci_lower = np.asarray(eq41_parametric_reference_slice["ci_lower"], dtype=float)
    ci_upper = np.asarray(eq41_parametric_reference_slice["ci_upper"], dtype=float)
    n_valid = int(eq41_parametric_reference_slice["n_valid"])
    alpha = float(eq41_parametric_reference_slice["alpha"])

    expected_variance = np.einsum("ij,jk,ik->i", w_matrix, omega_beta_hat, w_matrix)
    expected_se = np.sqrt(expected_variance / n_valid)
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    np.testing.assert_allclose(
        eq41_parametric_reference_slice["asymptotic_variance_hat"],
        expected_variance,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(standard_errors, expected_se, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        ci_lower, t_hat - z_critical * expected_se, atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        ci_upper, t_hat + z_critical * expected_se, atol=1e-12, rtol=0.0
    )
    assert np.all(ci_lower < t_hat)
    assert np.all(t_hat < ci_upper)
