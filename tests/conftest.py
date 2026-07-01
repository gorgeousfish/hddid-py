from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


_RESEARCH_PROBE_NAME_TOKENS = (
    "automation",
    "closeout",
    "contract_audit",
    "frontier",
    "live_gap",
    "monte_carlo_widening",
    "next_milestone",
    "outer_inference",
    "phase7",
    "research_backfill",
    "runtime_bridge",
    "same_seed",
    "seed303",
    "trigger",
)

_REVIEWER_TEST_NAMES = {
    "test_basis_dispatch.py",
    "test_estimation.py",
    "test_fit_hddid.py",
    "test_inference_edge_cases.py",
    "test_inference_nonparametric.py",
    "test_inference_parametric.py",
    "test_inputs.py",
    "test_manuscript_citation_evidence_contract.py",
    "test_manuscript_narrative_audit.py",
    "test_manuscript_replication_entrypoint.py",
    "test_manuscript_section6_payload_boundary.py",
    "test_manuscript_source_integrity.py",
    "test_nuisance.py",
    "test_nuisance_mainline.py",
    "test_packaging_metadata.py",
    "test_parametric_inference_reference.py",
    "test_plotting.py",
    "test_polynomial_sieve_basis.py",
    "test_public_root_exports.py",
    "test_replication_plot_artifacts.py",
    "test_results.py",
    "test_score.py",
    "test_section6_loader.py",
    "test_section6_roster.py",
    "test_splitting.py",
    "test_trigonometric_sieve_basis.py",
    "test_validation_lane_routing.py",
    "test_validation_source_checkout_boundary.py",
}


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    for item in items:
        file_name = Path(str(item.fspath)).name
        nodeid = item.nodeid.lower()
        if file_name in _REVIEWER_TEST_NAMES:
            item.add_marker(pytest.mark.reviewer)
        if any(token in nodeid for token in _RESEARCH_PROBE_NAME_TOKENS):
            item.add_marker(pytest.mark.research_probe)


@pytest.fixture
def sample_hddid_inputs() -> dict[str, object]:
    return {
        "y0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y1": [0.2, 1.2, 2.4, 3.6, 4.8, 6.0],
        "treat": [0, 1, 0, 1, 0, 1],
        "x": [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
        ],
        "z": [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]],
        "z0": 0.5,
        "basis_family": "polynomial",
        "basis_degree": 2,
        "alpha": 0.1,
    }


@pytest.fixture
def eq31_manual_slice() -> dict[str, object]:
    from hddid.inputs import validate_inputs
    from hddid.nuisance import NuisancePayload
    from hddid.results import FoldDiagnostics

    x = np.array(
        [
            [-1.0, 0.5],
            [0.0, 1.0],
            [1.0, -1.5],
            [2.0, 0.25],
            [-0.5, 1.5],
            [1.5, -0.75],
        ],
        dtype=float,
    )
    z = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
    basis_full = np.column_stack([np.ones_like(z), z, z**2])
    beta_true = np.array([1.0, -0.5], dtype=float)
    gamma_true = np.array([0.3, 0.2, -0.1], dtype=float)
    s_hat_full = x @ beta_true + basis_full @ gamma_true
    valid_mask = np.array([True, False, True, True, True, True], dtype=bool)
    treat = np.array([1, 0, 1, 1, 1, 1], dtype=int)
    delta_y = 0.5 * s_hat_full * np.where(treat == 1, 1.0, -1.0)

    data = validate_inputs(
        y0=np.zeros_like(s_hat_full),
        y1=delta_y,
        treat=treat,
        x=x,
        z=z,
        z0=np.array([0.25, 0.75], dtype=float),
        basis_family="polynomial",
        basis_degree=2,
        alpha=0.1,
    )
    nuisance_payload = NuisancePayload.from_predictions(
        pi_hat=np.full(data.n_obs, 0.5, dtype=float),
        phi0_hat=np.zeros(data.n_obs, dtype=float),
        phi1_hat=np.zeros(data.n_obs, dtype=float),
        treat=data.treat,
        fold_ids=np.array([1, 1, 2, 2, 3, 3], dtype=int),
        valid_mask=valid_mask,
        fold_diagnostics=[
            FoldDiagnostics(
                fold_id=1,
                basis_family="polynomial",
                basis_degree=2,
                oracle_lane="r_parity",
                n_holdout_raw=2,
                n_trimmed_propensity=1,
                n_valid_holdout=1,
                trim_lower=0.01,
                trim_upper=0.99,
            ),
            FoldDiagnostics(
                fold_id=2,
                basis_family="polynomial",
                basis_degree=2,
                oracle_lane="r_parity",
                n_holdout_raw=2,
                n_trimmed_propensity=0,
                n_valid_holdout=2,
                trim_lower=0.01,
                trim_upper=0.99,
            ),
            FoldDiagnostics(
                fold_id=3,
                basis_family="polynomial",
                basis_degree=2,
                oracle_lane="r_parity",
                n_holdout_raw=2,
                n_trimmed_propensity=0,
                n_valid_holdout=2,
                trim_lower=0.01,
                trim_upper=0.99,
            ),
        ],
        basis_family="polynomial",
        basis_degree=2,
        oracle_lane="r_parity",
    )
    return {
        "data": data,
        "nuisance_payload": nuisance_payload,
        "beta_true": beta_true,
        "gamma_true": gamma_true,
        "s_hat_full": s_hat_full,
        "basis_full": basis_full,
        "valid_mask": valid_mask,
    }


@pytest.fixture
def eq31_parity_slice(eq31_manual_slice: dict[str, object]) -> dict[str, object]:
    from hddid.inputs import validate_inputs
    from hddid.nuisance import NuisancePayload
    from hddid.score import build_score_payload

    base_data = eq31_manual_slice["data"]
    base_nuisance = eq31_manual_slice["nuisance_payload"]
    valid_mask = np.asarray(eq31_manual_slice["valid_mask"], dtype=bool)

    x = np.asarray(base_data.x, dtype=float)
    z = np.asarray(base_data.z, dtype=float)
    z0 = np.asarray(base_data.z0, dtype=float)
    basis_full = np.column_stack([np.ones_like(z), z, z**2])

    beta_true = np.array([0.9, -0.55], dtype=float)
    gamma_true = np.array([0.0, 0.2, -0.1], dtype=float)
    s_hat_full = x @ beta_true + basis_full @ gamma_true
    treat = np.array([1, 0, 1, 1, 1, 1], dtype=int)
    delta_y = 0.5 * s_hat_full * np.where(treat == 1, 1.0, -1.0)

    data = validate_inputs(
        y0=np.zeros_like(s_hat_full),
        y1=delta_y,
        treat=treat,
        x=x,
        z=z,
        z0=z0,
        basis_family="polynomial",
        basis_degree=2,
        alpha=0.1,
    )
    nuisance_payload = NuisancePayload.from_predictions(
        pi_hat=np.asarray(base_nuisance.pi_hat, dtype=float),
        phi0_hat=np.zeros(data.n_obs, dtype=float),
        phi1_hat=np.zeros(data.n_obs, dtype=float),
        treat=data.treat,
        fold_ids=np.asarray(base_nuisance.fold_ids, dtype=int),
        valid_mask=valid_mask,
        fold_diagnostics=list(base_nuisance.fold_diagnostics),
        basis_family="polynomial",
        basis_degree=2,
        oracle_lane="r-parity-polynomial",
    )
    score_payload = build_score_payload(data, nuisance_payload)

    return {
        "data": data,
        "nuisance_payload": nuisance_payload,
        "score_payload": score_payload,
        "beta_true": beta_true,
        "gamma_true": gamma_true,
        "combined_coefficients": np.concatenate([beta_true, gamma_true[1:]]),
        "delta_y_valid": np.asarray(data.y1 - data.y0, dtype=float)[valid_mask],
        "pi_hat_valid": np.asarray(nuisance_payload.pi_hat, dtype=float)[valid_mask],
        "phi1_hat_valid": np.asarray(nuisance_payload.phi1_hat, dtype=float)[
            valid_mask
        ],
        "phi0_hat_valid": np.asarray(nuisance_payload.phi0_hat, dtype=float)[
            valid_mask
        ],
        "rho_hat_valid": np.asarray(nuisance_payload.rho_hat, dtype=float)[valid_mask],
        "x_valid": np.asarray(data.x, dtype=float)[valid_mask],
        "z_valid": np.asarray(data.z, dtype=float)[valid_mask],
        "z0": z0,
        "basis_family": data.basis_family,
        "basis_degree": data.basis_degree,
        "oracle_lane": score_payload.oracle_lane,
        "valid_mask": valid_mask,
        "n_valid": int(valid_mask.sum()),
        "foldid": np.asarray(base_nuisance.fold_ids, dtype=int)[valid_mask],
    }


@pytest.fixture
def eq41_parametric_reference_slice() -> dict[str, object]:
    projection_x_valid = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    residual_valid = np.array([0.2, -0.4, 0.6, 0.8], dtype=float)
    beta_hat = np.array([1.1, -0.2], dtype=float)
    xi_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    alpha = 0.05
    n_valid = int(projection_x_valid.shape[0])

    sigma_tilde_x_hat = projection_x_valid.T @ projection_x_valid / n_valid
    score_moment = np.mean(residual_valid[:, None] * projection_x_valid, axis=0)
    omega_beta_hat = (
        (projection_x_valid * residual_valid[:, None]).T
        @ (projection_x_valid * residual_valid[:, None])
        / n_valid
    )
    w_matrix = -(np.linalg.inv(sigma_tilde_x_hat) @ xi_matrix.T).T
    t_hat = xi_matrix @ beta_hat - np.sum(w_matrix * score_moment[None, :], axis=1)
    asymptotic_variance_hat = np.einsum(
        "ij,jk,ik->i", w_matrix, omega_beta_hat, w_matrix
    )
    standard_errors = np.sqrt(asymptotic_variance_hat / n_valid)
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    ci_lower = t_hat - z_critical * standard_errors
    ci_upper = t_hat + z_critical * standard_errors

    return {
        "projection_x_valid": projection_x_valid,
        "residual_valid": residual_valid,
        "beta_hat": beta_hat,
        "xi_matrix": xi_matrix,
        "n_valid": n_valid,
        "alpha": alpha,
        "sigma_tilde_x_hat": sigma_tilde_x_hat,
        "score_moment": score_moment,
        "omega_beta_hat": omega_beta_hat,
        "w_matrix": w_matrix,
        "t_hat": t_hat,
        "asymptotic_variance_hat": asymptotic_variance_hat,
        "standard_errors": standard_errors,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "r_componentwise_xdebias": beta_hat
        + score_moment @ np.linalg.inv(sigma_tilde_x_hat),
    }


@pytest.fixture
def eq43_nonparametric_reference_slice() -> dict[str, object]:
    from hddid.estimation import EstimationPayload
    from hddid.results import FoldDiagnostics
    from hddid.score import ScorePayload

    x_valid = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [-1.0, 0.5],
            [0.0, -1.5],
            [1.5, 0.75],
        ],
        dtype=float,
    )
    basis_valid_full = np.array(
        [
            [1.0, -1.0, 1.0],
            [1.0, -0.25, 0.0625],
            [1.0, 0.5, 0.25],
            [1.0, 1.0, 1.0],
            [1.0, 1.5, 2.25],
        ],
        dtype=float,
    )
    evaluation_basis = np.array(
        [
            [1.0, -0.5, 0.25],
            [1.0, 0.75, 0.5625],
            [1.0, 1.25, 1.5625],
        ],
        dtype=float,
    )
    beta_hat = np.array([0.6, -0.3], dtype=float)
    gamma_hat = np.array([0.4, -0.2, 0.15], dtype=float)
    residual_valid = np.array([0.4, 0.2, -0.1, 0.3, -0.2], dtype=float)
    fitted_f_valid = basis_valid_full @ gamma_hat
    second_stage_prediction_valid = x_valid @ beta_hat + fitted_f_valid
    s_hat_valid = second_stage_prediction_valid + residual_valid

    valid_mask = np.ones(x_valid.shape[0], dtype=bool)
    fold_diagnostics = [
        FoldDiagnostics(
            fold_id=1,
            basis_family="polynomial",
            basis_degree=2,
            oracle_lane="r_parity",
            n_holdout_raw=2,
            n_trimmed_propensity=0,
            n_valid_holdout=2,
            trim_lower=0.01,
            trim_upper=0.99,
        ),
        FoldDiagnostics(
            fold_id=2,
            basis_family="polynomial",
            basis_degree=2,
            oracle_lane="r_parity",
            n_holdout_raw=2,
            n_trimmed_propensity=0,
            n_valid_holdout=2,
            trim_lower=0.01,
            trim_upper=0.99,
        ),
        FoldDiagnostics(
            fold_id=3,
            basis_family="polynomial",
            basis_degree=2,
            oracle_lane="r_parity",
            n_holdout_raw=1,
            n_trimmed_propensity=0,
            n_valid_holdout=1,
            trim_lower=0.01,
            trim_upper=0.99,
        ),
    ]
    score_payload = ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=valid_mask,
        fold_ids=np.array([1, 1, 2, 2, 3], dtype=int),
        fold_diagnostics=fold_diagnostics,
        basis_family="polynomial",
        basis_degree=2,
        oracle_lane="r_parity",
        basis_matrix=basis_valid_full,
        x_valid=x_valid,
        basis_valid_full=basis_valid_full,
        basis_design_valid=basis_valid_full[:, 1:],
        evaluation_basis=evaluation_basis,
        pi_hat=np.full(x_valid.shape[0], 0.5, dtype=float),
        phi0_hat=np.zeros(x_valid.shape[0], dtype=float),
        phi1_hat=np.zeros(x_valid.shape[0], dtype=float),
        rho_hat=np.ones(x_valid.shape[0], dtype=float),
        intercept_dropped_for_design=True,
    )
    estimation_payload = EstimationPayload(
        beta_hat=beta_hat,
        gamma_hat=gamma_hat,
        fitted_f_valid=fitted_f_valid,
        second_stage_prediction_valid=second_stage_prediction_valid,
        residual_valid=residual_valid,
        projection_x_valid=x_valid,
        f_hat_at_z0=evaluation_basis @ gamma_hat,
        optimization_metadata={
            "solver": "manual-eq43-reference",
            "n_valid_obs": int(x_valid.shape[0]),
            "beta_dimension": int(x_valid.shape[1]),
            "basis_dimension_full": int(basis_valid_full.shape[1]),
        },
    )

    n_valid = int(x_valid.shape[0])
    sigma_x_hat = x_valid.T @ x_valid / n_valid
    cross_moment_hat = basis_valid_full.T @ x_valid / n_valid
    m_hat = cross_moment_hat @ np.linalg.inv(sigma_x_hat)
    orthogonal_basis_valid = basis_valid_full - x_valid @ m_hat.T
    sigma_f_hat = orthogonal_basis_valid.T @ basis_valid_full / n_valid
    omega_f_hat = (basis_valid_full * residual_valid[:, None]).T @ (
        basis_valid_full * residual_valid[:, None]
    ) / n_valid - m_hat @ (
        (x_valid * residual_valid[:, None]).T
        @ (x_valid * residual_valid[:, None])
        / n_valid
    ) @ m_hat.T
    score_moment = np.mean(residual_valid[:, None] * orthogonal_basis_valid, axis=0)
    bar_gamma_hat = gamma_hat - np.linalg.solve(sigma_f_hat, score_moment)
    sigma_f_inverse = np.linalg.inv(sigma_f_hat)
    v_f_hat = sigma_f_inverse @ omega_f_hat @ sigma_f_inverse.T
    sigma_z_squared_hat = np.einsum(
        "ij,jk,ik->i",
        evaluation_basis,
        v_f_hat,
        evaluation_basis,
    )
    pointwise_standard_errors = np.sqrt(sigma_z_squared_hat / n_valid)
    alpha = 0.1
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    bar_f_at_z0 = evaluation_basis @ bar_gamma_hat
    pointwise_ci_lower = bar_f_at_z0 - z_critical * pointwise_standard_errors
    pointwise_ci_upper = bar_f_at_z0 + z_critical * pointwise_standard_errors

    random_state = 123
    n_boot = 256
    covariance_at_grid = evaluation_basis @ v_f_hat @ evaluation_basis.T / n_valid
    standardization = np.sqrt(np.diag(covariance_at_grid))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_at_grid)
    covariance_sqrt = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
    rng = np.random.default_rng(random_state)
    gaussian_draws = (
        rng.standard_normal(size=(n_boot, evaluation_basis.shape[0]))
        @ covariance_sqrt.T
    )
    simulated_suprema = np.max(
        np.abs(gaussian_draws / standardization[None, :]),
        axis=1,
    )
    uniform_critical_value = float(np.quantile(simulated_suprema, 1.0 - alpha))
    uniform_lower = bar_f_at_z0 - uniform_critical_value * pointwise_standard_errors
    uniform_upper = bar_f_at_z0 + uniform_critical_value * pointwise_standard_errors

    return {
        "score_payload": score_payload,
        "estimation_payload": estimation_payload,
        "n_valid": n_valid,
        "alpha": alpha,
        "lambda_double_prime": 0.0,
        "sigma_x_hat": sigma_x_hat,
        "cross_moment_hat": cross_moment_hat,
        "m_hat": m_hat,
        "orthogonal_basis_valid": orthogonal_basis_valid,
        "sigma_f_hat": sigma_f_hat,
        "omega_f_hat": omega_f_hat,
        "score_moment": score_moment,
        "bar_gamma_hat": bar_gamma_hat,
        "v_f_hat": v_f_hat,
        "evaluation_basis": evaluation_basis,
        "sigma_z_squared_hat": sigma_z_squared_hat,
        "pointwise_standard_errors": pointwise_standard_errors,
        "bar_f_at_z0": bar_f_at_z0,
        "pointwise_ci_lower": pointwise_ci_lower,
        "pointwise_ci_upper": pointwise_ci_upper,
        "random_state": random_state,
        "n_boot": n_boot,
        "uniform_critical_value": uniform_critical_value,
        "uniform_lower": uniform_lower,
        "uniform_upper": uniform_upper,
    }
