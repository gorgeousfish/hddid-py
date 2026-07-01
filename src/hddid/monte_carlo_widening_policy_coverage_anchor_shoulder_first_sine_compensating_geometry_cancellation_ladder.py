from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _driver_signature(
    *,
    upstream_driver_signature: str,
    live_right_center_residual: float,
    live_center_right_residual: float,
    diagonal_total_share_of_absolute_mass: float,
    top_two_diagonal_share_of_absolute_mass: float,
    top_two_diagonal_to_left_incident_ratio: float,
    cross_shoulder_pair_share_of_absolute_mass: float,
    left_center_pair_share_of_absolute_mass: float,
) -> str:
    if (
        upstream_driver_signature == "first-sine-compensating-geometry-budget"
        and np.isclose(live_right_center_residual, 0.0, atol=1e-12)
        and np.isclose(live_center_right_residual, 0.0, atol=1e-12)
        and diagonal_total_share_of_absolute_mass > 0.6
        and top_two_diagonal_share_of_absolute_mass > 0.55
        and top_two_diagonal_to_left_incident_ratio > 1.4
        and cross_shoulder_pair_share_of_absolute_mass
        > left_center_pair_share_of_absolute_mass
    ):
        return "first-sine-compensating-geometry-cancellation-ladder"
    return "mixed-first-sine-compensating-geometry-cancellation-ladder"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryCancellationLadderReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    total_compensating_absolute_mass: float
    diagonal_total_share_of_absolute_mass: float
    left_incident_off_diagonal_share_of_absolute_mass: float
    top_two_diagonal_share_of_absolute_mass: float
    top_two_diagonal_to_left_incident_ratio: float
    right_diagonal_share_of_diagonal_mass: float
    center_diagonal_share_of_diagonal_mass: float
    left_left_share_of_diagonal_mass: float
    left_center_pair_share_of_absolute_mass: float
    cross_shoulder_pair_share_of_absolute_mass: float
    driver_signature: str
    canonical_first_sine_compensating_geometry_cancellation_ladder_digest: tuple[
        str, ...
    ]

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
        self.total_compensating_absolute_mass = float(
            self.total_compensating_absolute_mass
        )
        self.diagonal_total_share_of_absolute_mass = float(
            self.diagonal_total_share_of_absolute_mass
        )
        self.left_incident_off_diagonal_share_of_absolute_mass = float(
            self.left_incident_off_diagonal_share_of_absolute_mass
        )
        self.top_two_diagonal_share_of_absolute_mass = float(
            self.top_two_diagonal_share_of_absolute_mass
        )
        self.top_two_diagonal_to_left_incident_ratio = float(
            self.top_two_diagonal_to_left_incident_ratio
        )
        self.right_diagonal_share_of_diagonal_mass = float(
            self.right_diagonal_share_of_diagonal_mass
        )
        self.center_diagonal_share_of_diagonal_mass = float(
            self.center_diagonal_share_of_diagonal_mass
        )
        self.left_left_share_of_diagonal_mass = float(
            self.left_left_share_of_diagonal_mass
        )
        self.left_center_pair_share_of_absolute_mass = float(
            self.left_center_pair_share_of_absolute_mass
        )
        self.cross_shoulder_pair_share_of_absolute_mass = float(
            self.cross_shoulder_pair_share_of_absolute_mass
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_compensating_geometry_cancellation_ladder_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_compensating_geometry_cancellation_ladder_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_cancellation_ladder_report(
    *,
    budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryCancellationLadderReport:
    matrix = np.abs(np.asarray(budget_report.compensating_geometry_matrix, dtype=float))
    if matrix.shape != (3, 3):
        raise ValueError("cancellation ladder expects the 3x3 witness grid")

    total_compensating_absolute_mass = float(matrix.sum())
    diagonal_mass = np.diag(matrix)
    diagonal_total_share_of_absolute_mass = float(
        diagonal_mass.sum() / total_compensating_absolute_mass
    )
    off_diagonal_left_incident_mass = float(
        matrix[0, 1] + matrix[1, 0] + matrix[0, 2] + matrix[2, 0]
    )
    left_incident_off_diagonal_share_of_absolute_mass = float(
        off_diagonal_left_incident_mass / total_compensating_absolute_mass
    )
    top_two_diagonal_share_of_absolute_mass = float(
        np.sort(diagonal_mass)[-2:].sum() / total_compensating_absolute_mass
    )
    top_two_diagonal_to_left_incident_ratio = float(
        np.sort(diagonal_mass)[-2:].sum() / off_diagonal_left_incident_mass
    )
    diagonal_total_mass = float(diagonal_mass.sum())
    right_diagonal_share_of_diagonal_mass = float(matrix[2, 2] / diagonal_total_mass)
    center_diagonal_share_of_diagonal_mass = float(matrix[1, 1] / diagonal_total_mass)
    left_left_share_of_diagonal_mass = float(matrix[0, 0] / diagonal_total_mass)
    left_center_pair_share_of_absolute_mass = float(
        (matrix[0, 1] + matrix[1, 0]) / total_compensating_absolute_mass
    )
    cross_shoulder_pair_share_of_absolute_mass = float(
        (matrix[0, 2] + matrix[2, 0]) / total_compensating_absolute_mass
    )
    driver_signature = _driver_signature(
        upstream_driver_signature=budget_report.driver_signature,
        live_right_center_residual=budget_report.live_right_center_residual,
        live_center_right_residual=budget_report.live_center_right_residual,
        diagonal_total_share_of_absolute_mass=diagonal_total_share_of_absolute_mass,
        top_two_diagonal_share_of_absolute_mass=top_two_diagonal_share_of_absolute_mass,
        top_two_diagonal_to_left_incident_ratio=top_two_diagonal_to_left_incident_ratio,
        cross_shoulder_pair_share_of_absolute_mass=cross_shoulder_pair_share_of_absolute_mass,
        left_center_pair_share_of_absolute_mass=left_center_pair_share_of_absolute_mass,
    )
    canonical_digest = (
        f"- the compensating geometry still stays entirely off the live right-center entry while carrying total absolute mass `{_format_float(total_compensating_absolute_mass)}`; its cancellation ladder remains cell-local rather than broad replay",
        f"- diagonal cancellation still dominates the compensating budget: diagonal cells absorb `{_format_percent(diagonal_total_share_of_absolute_mass)}` of total absolute mass, with the top two diagonal cells alone already consuming `{_format_percent(top_two_diagonal_share_of_absolute_mass)}`; that exceeds the full left-incident off-diagonal budget `{_format_percent(left_incident_off_diagonal_share_of_absolute_mass)}` by `{_format_ratio(top_two_diagonal_to_left_incident_ratio)}`",
        f"- within the diagonal budget, right-right cancellation remains the primary sink at `{_format_percent(right_diagonal_share_of_diagonal_mass)}`, center-center follows at `{_format_percent(center_diagonal_share_of_diagonal_mass)}`, and left-left stays residual at `{_format_percent(left_left_share_of_diagonal_mass)}`; among off-diagonal left-incident pairs, cross-shoulder consumes `{_format_percent(cross_shoulder_pair_share_of_absolute_mass)}` of total absolute mass versus `{_format_percent(left_center_pair_share_of_absolute_mass)}` for left-center",
        "- current Trigger 2 implication: `first-sine-compensating-geometry-cancellation-ladder`; a preserve-left-support implementation path still needs to neutralize diagonal spillover first, so fixing only left-side leakage would leave most of the compensating budget unresolved",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryCancellationLadderReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-compensating-geometry-cancellation-ladder",
        policy_digest=budget_report.policy_digest,
        binding_design=budget_report.binding_design,
        window_label=budget_report.window_label,
        evaluation_grid=budget_report.evaluation_grid,
        coverage_anchor_random_state=budget_report.coverage_anchor_random_state,
        overshoot_companion_random_state=budget_report.overshoot_companion_random_state,
        diagonal_coordinate=budget_report.diagonal_coordinate,
        diagonal_basis_label=budget_report.diagonal_basis_label,
        total_compensating_absolute_mass=total_compensating_absolute_mass,
        diagonal_total_share_of_absolute_mass=diagonal_total_share_of_absolute_mass,
        left_incident_off_diagonal_share_of_absolute_mass=(
            left_incident_off_diagonal_share_of_absolute_mass
        ),
        top_two_diagonal_share_of_absolute_mass=(
            top_two_diagonal_share_of_absolute_mass
        ),
        top_two_diagonal_to_left_incident_ratio=(
            top_two_diagonal_to_left_incident_ratio
        ),
        right_diagonal_share_of_diagonal_mass=right_diagonal_share_of_diagonal_mass,
        center_diagonal_share_of_diagonal_mass=(center_diagonal_share_of_diagonal_mass),
        left_left_share_of_diagonal_mass=left_left_share_of_diagonal_mass,
        left_center_pair_share_of_absolute_mass=(
            left_center_pair_share_of_absolute_mass
        ),
        cross_shoulder_pair_share_of_absolute_mass=(
            cross_shoulder_pair_share_of_absolute_mass
        ),
        driver_signature=driver_signature,
        canonical_first_sine_compensating_geometry_cancellation_ladder_digest=(
            canonical_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_cancellation_ladder() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryCancellationLadderReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_cancellation_ladder_report(
        budget_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget()
    )
