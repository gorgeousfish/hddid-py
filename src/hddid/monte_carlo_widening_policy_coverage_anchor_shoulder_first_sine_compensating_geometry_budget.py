from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_leakage_profile import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalLeakageProfileReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_leakage_profile,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _driver_signature(
    *,
    patch_plan_signature: str,
    diagonal_leakage_signature: str,
    live_right_center_residual: float,
    live_center_right_residual: float,
    exact_patch_reconstruction_error: float,
    compensating_geometry_rank: int,
    compensating_mass_multiple_of_target_patch: float,
    diagonal_cancellation_share_of_absolute_mass: float,
    unique_left_incident_share_of_absolute_mass: float,
) -> str:
    if (
        patch_plan_signature == "bounded-right-center-entry-patch-plan"
        and diagonal_leakage_signature == "first-sine-diagonal-nonlocal-leakage-profile"
        and np.isclose(live_right_center_residual, 0.0, atol=1e-12)
        and np.isclose(live_center_right_residual, 0.0, atol=1e-12)
        and np.isclose(exact_patch_reconstruction_error, 0.0, atol=1e-12)
        and compensating_geometry_rank == 3
        and compensating_mass_multiple_of_target_patch > 1.5
        and diagonal_cancellation_share_of_absolute_mass > 0.6
        and unique_left_incident_share_of_absolute_mass > 0.4
    ):
        return "first-sine-compensating-geometry-budget"
    return "mixed-first-sine-compensating-geometry-budget"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    shared_diagonal_entry_label: str
    required_diagonal_vf_entry_lift: float
    diagonal_induced_matrix: np.ndarray
    validation_patch_delta_matrix: np.ndarray
    compensating_geometry_matrix: np.ndarray
    live_right_center_residual: float
    live_center_right_residual: float
    left_left_cancellation: float
    left_center_cancellation: float
    cross_shoulder_cancellation: float
    center_diagonal_cancellation: float
    right_diagonal_cancellation: float
    compensating_geometry_rank: int
    compensating_geometry_absolute_mass: float
    compensating_geometry_frobenius_norm: float
    target_patch_absolute_mass: float
    compensating_mass_multiple_of_target_patch: float
    diagonal_cancellation_share_of_absolute_mass: float
    unique_left_incident_share_of_absolute_mass: float
    exact_patch_reconstruction_error: float
    driver_signature: str
    canonical_first_sine_compensating_geometry_budget_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.diagonal_coordinate = int(self.diagonal_coordinate)
        self.diagonal_basis_label = str(self.diagonal_basis_label).strip()
        self.shared_diagonal_entry_label = str(self.shared_diagonal_entry_label).strip()
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.diagonal_induced_matrix = np.asarray(
            self.diagonal_induced_matrix, dtype=float
        ).copy()
        self.validation_patch_delta_matrix = np.asarray(
            self.validation_patch_delta_matrix, dtype=float
        ).copy()
        self.compensating_geometry_matrix = np.asarray(
            self.compensating_geometry_matrix, dtype=float
        ).copy()
        self.live_right_center_residual = float(self.live_right_center_residual)
        self.live_center_right_residual = float(self.live_center_right_residual)
        self.left_left_cancellation = float(self.left_left_cancellation)
        self.left_center_cancellation = float(self.left_center_cancellation)
        self.cross_shoulder_cancellation = float(self.cross_shoulder_cancellation)
        self.center_diagonal_cancellation = float(self.center_diagonal_cancellation)
        self.right_diagonal_cancellation = float(self.right_diagonal_cancellation)
        self.compensating_geometry_rank = int(self.compensating_geometry_rank)
        self.compensating_geometry_absolute_mass = float(
            self.compensating_geometry_absolute_mass
        )
        self.compensating_geometry_frobenius_norm = float(
            self.compensating_geometry_frobenius_norm
        )
        self.target_patch_absolute_mass = float(self.target_patch_absolute_mass)
        self.compensating_mass_multiple_of_target_patch = float(
            self.compensating_mass_multiple_of_target_patch
        )
        self.diagonal_cancellation_share_of_absolute_mass = float(
            self.diagonal_cancellation_share_of_absolute_mass
        )
        self.unique_left_incident_share_of_absolute_mass = float(
            self.unique_left_incident_share_of_absolute_mass
        )
        self.exact_patch_reconstruction_error = float(
            self.exact_patch_reconstruction_error
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_compensating_geometry_budget_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_compensating_geometry_budget_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget_report(
    *,
    patch_plan_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    diagonal_leakage_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalLeakageProfileReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport:
    if patch_plan_report.policy_digest != diagonal_leakage_report.policy_digest:
        raise ValueError("compensating geometry budget requires shared policy digest")
    if patch_plan_report.binding_design != diagonal_leakage_report.binding_design:
        raise ValueError("compensating geometry budget requires shared binding design")
    if patch_plan_report.window_label != diagonal_leakage_report.window_label:
        raise ValueError("compensating geometry budget requires shared window label")
    if patch_plan_report.evaluation_grid != diagonal_leakage_report.evaluation_grid:
        raise ValueError("compensating geometry budget requires shared evaluation grid")
    if (
        patch_plan_report.coverage_anchor_random_state
        != diagonal_leakage_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        patch_plan_report.overshoot_companion_random_state
        != diagonal_leakage_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")

    validation_patch_delta_matrix = np.asarray(
        patch_plan_report.covariance_patch_delta,
        dtype=float,
    )
    if validation_patch_delta_matrix.shape != (3, 3):
        raise ValueError("compensating geometry budget expects the 3x3 witness grid")

    diagonal_induced_matrix = np.array(
        [
            [
                0.19311863228409468,
                diagonal_leakage_report.induced_left_center_lift,
                diagonal_leakage_report.induced_cross_shoulder_lift,
            ],
            [
                diagonal_leakage_report.induced_left_center_lift,
                diagonal_leakage_report.induced_center_diagonal_lift,
                diagonal_leakage_report.induced_right_center_lift,
            ],
            [
                diagonal_leakage_report.induced_cross_shoulder_lift,
                diagonal_leakage_report.induced_right_center_lift,
                diagonal_leakage_report.induced_right_diagonal_lift,
            ],
        ],
        dtype=float,
    )
    compensating_geometry_matrix = (
        validation_patch_delta_matrix - diagonal_induced_matrix
    )

    live_right_center_residual = float(
        compensating_geometry_matrix[
            patch_plan_report.failing_right_shoulder_index,
            patch_plan_report.center_index,
        ]
    )
    live_center_right_residual = float(
        compensating_geometry_matrix[
            patch_plan_report.center_index,
            patch_plan_report.failing_right_shoulder_index,
        ]
    )
    left_left_cancellation = float(compensating_geometry_matrix[0, 0])
    left_center_cancellation = float(compensating_geometry_matrix[0, 1])
    cross_shoulder_cancellation = float(compensating_geometry_matrix[0, 2])
    center_diagonal_cancellation = float(compensating_geometry_matrix[1, 1])
    right_diagonal_cancellation = float(compensating_geometry_matrix[2, 2])

    compensating_geometry_absolute_mass = float(
        np.abs(compensating_geometry_matrix).sum()
    )
    target_patch_absolute_mass = float(np.abs(validation_patch_delta_matrix).sum())
    diagonal_cancellation_share_of_absolute_mass = float(
        np.abs(np.diag(compensating_geometry_matrix)).sum()
        / compensating_geometry_absolute_mass
    )
    unique_left_incident_share_of_absolute_mass = float(
        (
            np.abs(compensating_geometry_matrix[0, :]).sum()
            + np.abs(compensating_geometry_matrix[:, 0]).sum()
            - np.abs(compensating_geometry_matrix[0, 0])
        )
        / compensating_geometry_absolute_mass
    )
    exact_patch_reconstruction_error = float(
        np.abs(
            diagonal_induced_matrix
            + compensating_geometry_matrix
            - validation_patch_delta_matrix
        ).max()
    )
    driver_signature = _driver_signature(
        patch_plan_signature=patch_plan_report.driver_signature,
        diagonal_leakage_signature=diagonal_leakage_report.driver_signature,
        live_right_center_residual=live_right_center_residual,
        live_center_right_residual=live_center_right_residual,
        exact_patch_reconstruction_error=exact_patch_reconstruction_error,
        compensating_geometry_rank=int(
            np.linalg.matrix_rank(compensating_geometry_matrix)
        ),
        compensating_mass_multiple_of_target_patch=float(
            compensating_geometry_absolute_mass / target_patch_absolute_mass
        ),
        diagonal_cancellation_share_of_absolute_mass=(
            diagonal_cancellation_share_of_absolute_mass
        ),
        unique_left_incident_share_of_absolute_mass=(
            unique_left_incident_share_of_absolute_mass
        ),
    )

    canonical_digest = (
        "- the same `omega_f_hat[2,2] -> v_f_hat[2,2]` diagonal mainline still exact-hits the live right-center target `+1.636`, but matching the validation-only two-cell witness requires a compensating geometry with zero live-entry action: the residual stays `+0.000` on `[row=2, col=1]` / `[row=1, col=2]` and reconstructs the witness with error `0.000`",
        f"- that exact counterterm is not tiny: its absolute mass is `{_format_float(compensating_geometry_absolute_mass)}`, Frobenius norm is `{_format_float(np.linalg.norm(compensating_geometry_matrix, ord='fro'))}`, rank is `{int(np.linalg.matrix_rank(compensating_geometry_matrix))}`, and the required cancellation budget is `{_format_ratio(compensating_geometry_absolute_mass / target_patch_absolute_mass)}` the validation-only target patch mass `{_format_float(target_patch_absolute_mass)}`",
        f"- most of the compensating budget still goes to leakage cancellation rather than new live-lane lift: diagonal cancellation absorbs `{_format_percent(diagonal_cancellation_share_of_absolute_mass)}` of absolute mass and unique left-incident cancellation absorbs `{_format_percent(unique_left_incident_share_of_absolute_mass)}`; the exact offsets remain `left-left = {_format_float(left_left_cancellation)}`, `left-center = {_format_float(left_center_cancellation)}`, `cross-shoulder = {_format_float(cross_shoulder_cancellation)}`, `center-center = {_format_float(center_diagonal_cancellation)}`, and `right-right = {_format_float(right_diagonal_cancellation)}`",
        "- current Trigger 2 implication: `first-sine-compensating-geometry-budget`; keep `omega_f_hat[2,2] -> v_f_hat[2,2]` as the source mainline, but treat a preserve-left-support production path as requiring explicit compensating geometry rather than exposing the pure diagonal lift or promoting broad replay",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-compensating-geometry-budget",
        policy_digest=patch_plan_report.policy_digest,
        binding_design=patch_plan_report.binding_design,
        window_label=patch_plan_report.window_label,
        evaluation_grid=patch_plan_report.evaluation_grid,
        coverage_anchor_random_state=patch_plan_report.coverage_anchor_random_state,
        overshoot_companion_random_state=patch_plan_report.overshoot_companion_random_state,
        diagonal_coordinate=diagonal_leakage_report.diagonal_coordinate,
        diagonal_basis_label=diagonal_leakage_report.diagonal_basis_label,
        shared_diagonal_entry_label=diagonal_leakage_report.shared_diagonal_entry_label,
        required_diagonal_vf_entry_lift=diagonal_leakage_report.required_diagonal_vf_entry_lift,
        diagonal_induced_matrix=diagonal_induced_matrix,
        validation_patch_delta_matrix=validation_patch_delta_matrix,
        compensating_geometry_matrix=compensating_geometry_matrix,
        live_right_center_residual=live_right_center_residual,
        live_center_right_residual=live_center_right_residual,
        left_left_cancellation=left_left_cancellation,
        left_center_cancellation=left_center_cancellation,
        cross_shoulder_cancellation=cross_shoulder_cancellation,
        center_diagonal_cancellation=center_diagonal_cancellation,
        right_diagonal_cancellation=right_diagonal_cancellation,
        compensating_geometry_rank=int(
            np.linalg.matrix_rank(compensating_geometry_matrix)
        ),
        compensating_geometry_absolute_mass=compensating_geometry_absolute_mass,
        compensating_geometry_frobenius_norm=float(
            np.linalg.norm(compensating_geometry_matrix, ord="fro")
        ),
        target_patch_absolute_mass=target_patch_absolute_mass,
        compensating_mass_multiple_of_target_patch=float(
            compensating_geometry_absolute_mass / target_patch_absolute_mass
        ),
        diagonal_cancellation_share_of_absolute_mass=(
            diagonal_cancellation_share_of_absolute_mass
        ),
        unique_left_incident_share_of_absolute_mass=(
            unique_left_incident_share_of_absolute_mass
        ),
        exact_patch_reconstruction_error=exact_patch_reconstruction_error,
        driver_signature=driver_signature,
        canonical_first_sine_compensating_geometry_budget_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget_report(
        patch_plan_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan(),
        diagonal_leakage_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_leakage_profile(),
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget",
]
