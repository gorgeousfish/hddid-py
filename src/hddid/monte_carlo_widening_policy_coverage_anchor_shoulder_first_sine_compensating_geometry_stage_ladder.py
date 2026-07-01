from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_cancellation_ladder import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryCancellationLadderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_cancellation_ladder,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget,
)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    cancellation_ladder_signature: str,
    zero_live_entry_count: int,
    top_two_diagonal_share_of_total_absolute_mass: float,
    cumulative_cross_shoulder_share_of_total_absolute_mass: float,
    cumulative_left_center_share_of_total_absolute_mass: float,
    left_left_residual_share_of_total_absolute_mass: float,
) -> str:
    if (
        cancellation_ladder_signature
        == "first-sine-compensating-geometry-cancellation-ladder"
        and zero_live_entry_count == 2
        and top_two_diagonal_share_of_total_absolute_mass > 0.55
        and cumulative_cross_shoulder_share_of_total_absolute_mass > 0.79
        and cumulative_left_center_share_of_total_absolute_mass > 0.96
        and left_left_residual_share_of_total_absolute_mass < 0.04
    ):
        return "first-sine-compensating-geometry-stage-ladder"
    return "mixed-first-sine-compensating-geometry-stage-ladder"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    stage_order: tuple[str, ...]
    stage_absolute_mass: tuple[float, ...]
    stage_share_of_total_absolute_mass: tuple[float, ...]
    cumulative_share_of_total_absolute_mass: tuple[float, ...]
    zero_live_entry_count: int
    driver_signature: str
    canonical_first_sine_compensating_geometry_stage_ladder_digest: tuple[str, ...]

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
        self.stage_order = tuple(str(item).strip() for item in self.stage_order)
        self.stage_absolute_mass = tuple(
            float(value) for value in self.stage_absolute_mass
        )
        self.stage_share_of_total_absolute_mass = tuple(
            float(value) for value in self.stage_share_of_total_absolute_mass
        )
        self.cumulative_share_of_total_absolute_mass = tuple(
            float(value) for value in self.cumulative_share_of_total_absolute_mass
        )
        self.zero_live_entry_count = int(self.zero_live_entry_count)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_compensating_geometry_stage_ladder_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_compensating_geometry_stage_ladder_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder_report(
    *,
    budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport
    ),
    cancellation_ladder_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryCancellationLadderReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport:
    if budget_report.policy_digest != cancellation_ladder_report.policy_digest:
        raise ValueError("stage ladder requires shared policy digest")
    if budget_report.binding_design != cancellation_ladder_report.binding_design:
        raise ValueError("stage ladder requires shared binding design")
    if budget_report.window_label != cancellation_ladder_report.window_label:
        raise ValueError("stage ladder requires shared window label")
    if budget_report.evaluation_grid != cancellation_ladder_report.evaluation_grid:
        raise ValueError("stage ladder requires shared evaluation grid")
    if (
        budget_report.coverage_anchor_random_state
        != cancellation_ladder_report.coverage_anchor_random_state
    ):
        raise ValueError("stage ladder requires shared coverage-anchor seed")
    if (
        budget_report.overshoot_companion_random_state
        != cancellation_ladder_report.overshoot_companion_random_state
    ):
        raise ValueError("stage ladder requires shared overshoot-companion seed")
    if (
        budget_report.diagonal_coordinate
        != cancellation_ladder_report.diagonal_coordinate
    ):
        raise ValueError("stage ladder requires shared diagonal coordinate")
    if (
        budget_report.diagonal_basis_label
        != cancellation_ladder_report.diagonal_basis_label
    ):
        raise ValueError("stage ladder requires shared diagonal basis label")

    matrix = np.abs(np.asarray(budget_report.compensating_geometry_matrix, dtype=float))
    if matrix.shape != (3, 3):
        raise ValueError("stage ladder expects the 3x3 witness grid")

    stage_order = (
        "top-two-diagonal",
        "cross-shoulder-pair",
        "left-center-pair",
        "left-left-residual",
    )
    stage_absolute_mass = (
        float(matrix[2, 2] + matrix[1, 1]),
        float(matrix[0, 2] + matrix[2, 0]),
        float(matrix[0, 1] + matrix[1, 0]),
        float(matrix[0, 0]),
    )
    total_absolute_mass = float(matrix.sum())
    stage_share_of_total_absolute_mass = tuple(
        float(value / total_absolute_mass) for value in stage_absolute_mass
    )
    cumulative_share_of_total_absolute_mass = tuple(
        float(value / total_absolute_mass)
        for value in np.cumsum(np.asarray(stage_absolute_mass, dtype=float))
    )
    zero_live_entry_count = int(
        np.count_nonzero(
            np.isclose(
                matrix[1:, 1:],
                np.array([[matrix[1, 1], 0.0], [0.0, matrix[2, 2]]], dtype=float),
                atol=1e-12,
            )
        )
    )
    driver_signature = _driver_signature(
        cancellation_ladder_signature=cancellation_ladder_report.driver_signature,
        zero_live_entry_count=int(
            np.count_nonzero(np.isclose(matrix[[1, 2], [2, 1]], 0.0, atol=1e-12))
        ),
        top_two_diagonal_share_of_total_absolute_mass=stage_share_of_total_absolute_mass[
            0
        ],
        cumulative_cross_shoulder_share_of_total_absolute_mass=(
            cumulative_share_of_total_absolute_mass[1]
        ),
        cumulative_left_center_share_of_total_absolute_mass=(
            cumulative_share_of_total_absolute_mass[2]
        ),
        left_left_residual_share_of_total_absolute_mass=(
            stage_share_of_total_absolute_mass[3]
        ),
    )
    canonical_digest = (
        f"- the compensating geometry still resolves as a four-stage ladder away from the live right-center entry: top-two diagonal cancellation (`right-right + center-center`) absorbs `{_format_percent(stage_share_of_total_absolute_mass[0])}` of total absolute mass before any left-incident cleanup starts",
        f"- adding the symmetric cross-shoulder pair raises cumulative cancellation coverage to `{_format_percent(cumulative_share_of_total_absolute_mass[1])}`, so the first off-diagonal follow-up still stays on the shoulder bridge instead of broad replay",
        f"- adding the symmetric left-center pair raises cumulative cancellation coverage to `{_format_percent(cumulative_share_of_total_absolute_mass[2])}`, leaving only a `{_format_percent(stage_share_of_total_absolute_mass[3])}` left-left residual cell at the end of the ladder",
        "- current Trigger 2 implication: `first-sine-compensating-geometry-stage-ladder`; a preserve-left-support implementation path can stay on this diagonal-first then pair-local ladder, rather than widening into dense covariance replay",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-compensating-geometry-stage-ladder",
        policy_digest=budget_report.policy_digest,
        binding_design=budget_report.binding_design,
        window_label=budget_report.window_label,
        evaluation_grid=budget_report.evaluation_grid,
        coverage_anchor_random_state=budget_report.coverage_anchor_random_state,
        overshoot_companion_random_state=budget_report.overshoot_companion_random_state,
        diagonal_coordinate=budget_report.diagonal_coordinate,
        diagonal_basis_label=budget_report.diagonal_basis_label,
        stage_order=stage_order,
        stage_absolute_mass=stage_absolute_mass,
        stage_share_of_total_absolute_mass=stage_share_of_total_absolute_mass,
        cumulative_share_of_total_absolute_mass=(
            cumulative_share_of_total_absolute_mass
        ),
        zero_live_entry_count=int(
            np.count_nonzero(np.isclose(matrix[[1, 2], [2, 1]], 0.0, atol=1e-12))
        ),
        driver_signature=driver_signature,
        canonical_first_sine_compensating_geometry_stage_ladder_digest=(
            canonical_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport
):
    budget_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget()
    cancellation_ladder_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_cancellation_ladder()
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder_report(
        budget_report=budget_report,
        cancellation_ladder_report=cancellation_ladder_report,
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder",
]
