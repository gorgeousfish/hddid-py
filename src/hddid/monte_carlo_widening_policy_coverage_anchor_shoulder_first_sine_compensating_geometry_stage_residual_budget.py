from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder,
)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    budget_driver_signature: str,
    stage_ladder_driver_signature: str,
    remaining_share_after_top_two_diagonal: float,
    remaining_share_after_cross_shoulder_pair: float,
    remaining_share_after_left_center_pair: float,
    remaining_frobenius_share_after_left_center_pair: float,
) -> str:
    if (
        budget_driver_signature == "first-sine-compensating-geometry-budget"
        and stage_ladder_driver_signature
        == "first-sine-compensating-geometry-stage-ladder"
        and remaining_share_after_top_two_diagonal > 0.4
        and remaining_share_after_cross_shoulder_pair > 0.2
        and remaining_share_after_left_center_pair < 0.04
        and remaining_frobenius_share_after_left_center_pair < 0.08
    ):
        return "first-sine-compensating-geometry-stage-residual-budget"
    return "mixed-first-sine-compensating-geometry-stage-residual-budget"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageResidualBudgetReport:
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
    remaining_absolute_mass_after_stage: tuple[float, ...]
    remaining_share_of_total_absolute_mass_after_stage: tuple[float, ...]
    remaining_frobenius_norm_after_stage: tuple[float, ...]
    remaining_share_of_total_frobenius_norm_after_stage: tuple[float, ...]
    driver_signature: str
    canonical_first_sine_compensating_geometry_stage_residual_budget_digest: tuple[
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
        self.stage_order = tuple(str(item).strip() for item in self.stage_order)
        self.remaining_absolute_mass_after_stage = tuple(
            float(value) for value in self.remaining_absolute_mass_after_stage
        )
        self.remaining_share_of_total_absolute_mass_after_stage = tuple(
            float(value)
            for value in self.remaining_share_of_total_absolute_mass_after_stage
        )
        self.remaining_frobenius_norm_after_stage = tuple(
            float(value) for value in self.remaining_frobenius_norm_after_stage
        )
        self.remaining_share_of_total_frobenius_norm_after_stage = tuple(
            float(value)
            for value in self.remaining_share_of_total_frobenius_norm_after_stage
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_compensating_geometry_stage_residual_budget_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_compensating_geometry_stage_residual_budget_digest
        )


def _stage_cells(stage_name: str) -> tuple[tuple[int, int], ...]:
    stage_to_cells = {
        "top-two-diagonal": ((2, 2), (1, 1)),
        "cross-shoulder-pair": ((0, 2), (2, 0)),
        "left-center-pair": ((0, 1), (1, 0)),
        "left-left-residual": ((0, 0),),
    }
    try:
        return stage_to_cells[stage_name]
    except KeyError as exc:
        raise ValueError(f"unknown stage name: {stage_name}") from exc


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_residual_budget_report(
    *,
    budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport
    ),
    stage_ladder_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageResidualBudgetReport:
    if budget_report.policy_digest != stage_ladder_report.policy_digest:
        raise ValueError("stage residual budget requires shared policy digest")
    if budget_report.binding_design != stage_ladder_report.binding_design:
        raise ValueError("stage residual budget requires shared binding design")
    if budget_report.window_label != stage_ladder_report.window_label:
        raise ValueError("stage residual budget requires shared window label")
    if budget_report.evaluation_grid != stage_ladder_report.evaluation_grid:
        raise ValueError("stage residual budget requires shared evaluation grid")
    if (
        budget_report.coverage_anchor_random_state
        != stage_ladder_report.coverage_anchor_random_state
    ):
        raise ValueError("stage residual budget requires shared coverage-anchor seed")
    if (
        budget_report.overshoot_companion_random_state
        != stage_ladder_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "stage residual budget requires shared overshoot-companion seed"
        )
    if budget_report.diagonal_coordinate != stage_ladder_report.diagonal_coordinate:
        raise ValueError("stage residual budget requires shared diagonal coordinate")
    if budget_report.diagonal_basis_label != stage_ladder_report.diagonal_basis_label:
        raise ValueError("stage residual budget requires shared diagonal basis label")

    matrix = np.abs(np.asarray(budget_report.compensating_geometry_matrix, dtype=float))
    if matrix.shape != (3, 3):
        raise ValueError("stage residual budget expects the 3x3 witness grid")

    total_absolute_mass = float(matrix.sum())
    total_frobenius_norm = float(np.linalg.norm(matrix, ord="fro"))
    remaining_absolute_mass_after_stage: list[float] = []
    remaining_share_of_total_absolute_mass_after_stage: list[float] = []
    remaining_frobenius_norm_after_stage: list[float] = []
    remaining_share_of_total_frobenius_norm_after_stage: list[float] = []
    residual_matrix = matrix.copy()

    for stage_name in stage_ladder_report.stage_order:
        for row_index, column_index in _stage_cells(stage_name):
            residual_matrix[row_index, column_index] = 0.0
        remaining_absolute_mass = float(residual_matrix.sum())
        remaining_frobenius_norm = float(np.linalg.norm(residual_matrix, ord="fro"))
        remaining_absolute_mass_after_stage.append(remaining_absolute_mass)
        remaining_share_of_total_absolute_mass_after_stage.append(
            float(remaining_absolute_mass / total_absolute_mass)
        )
        remaining_frobenius_norm_after_stage.append(remaining_frobenius_norm)
        remaining_share_of_total_frobenius_norm_after_stage.append(
            float(remaining_frobenius_norm / total_frobenius_norm)
        )

    driver_signature = _driver_signature(
        budget_driver_signature=budget_report.driver_signature,
        stage_ladder_driver_signature=stage_ladder_report.driver_signature,
        remaining_share_after_top_two_diagonal=(
            remaining_share_of_total_absolute_mass_after_stage[0]
        ),
        remaining_share_after_cross_shoulder_pair=(
            remaining_share_of_total_absolute_mass_after_stage[1]
        ),
        remaining_share_after_left_center_pair=(
            remaining_share_of_total_absolute_mass_after_stage[2]
        ),
        remaining_frobenius_share_after_left_center_pair=(
            remaining_share_of_total_frobenius_norm_after_stage[2]
        ),
    )
    canonical_digest = (
        f"- after the top-two diagonal stage, the compensating geometry still carries `{_format_percent(remaining_share_of_total_absolute_mass_after_stage[0])}` of absolute mass and `{_format_percent(remaining_share_of_total_frobenius_norm_after_stage[0])}` of Frobenius norm, so diagonal-first alone leaves a material off-live cleanup budget",
        f"- adding the symmetric cross-shoulder pair cuts the residual to `{_format_percent(remaining_share_of_total_absolute_mass_after_stage[1])}` of absolute mass and `{_format_percent(remaining_share_of_total_frobenius_norm_after_stage[1])}` of Frobenius norm, but that still leaves a non-trivial left-center remainder rather than a production-ready witness",
        f"- only after the symmetric left-center pair does the residual collapse to the `{_format_percent(remaining_share_of_total_absolute_mass_after_stage[2])}` / `{_format_percent(remaining_share_of_total_frobenius_norm_after_stage[2])}` left-left tail, so a preserve-left-support implementation path cannot stop after cross-shoulder cleanup if it wants the near-complete validation-only witness",
        "- current Trigger 2 implication: `first-sine-compensating-geometry-stage-residual-budget`; treat left-center cleanup as the last material residual stage after diagonal-first plus cross-shoulder repair, not as optional polish or a reason to widen into dense replay",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageResidualBudgetReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-compensating-geometry-stage-residual-budget",
        policy_digest=budget_report.policy_digest,
        binding_design=budget_report.binding_design,
        window_label=budget_report.window_label,
        evaluation_grid=budget_report.evaluation_grid,
        coverage_anchor_random_state=budget_report.coverage_anchor_random_state,
        overshoot_companion_random_state=budget_report.overshoot_companion_random_state,
        diagonal_coordinate=budget_report.diagonal_coordinate,
        diagonal_basis_label=budget_report.diagonal_basis_label,
        stage_order=stage_ladder_report.stage_order,
        remaining_absolute_mass_after_stage=tuple(remaining_absolute_mass_after_stage),
        remaining_share_of_total_absolute_mass_after_stage=tuple(
            remaining_share_of_total_absolute_mass_after_stage
        ),
        remaining_frobenius_norm_after_stage=tuple(
            remaining_frobenius_norm_after_stage
        ),
        remaining_share_of_total_frobenius_norm_after_stage=tuple(
            remaining_share_of_total_frobenius_norm_after_stage
        ),
        driver_signature=driver_signature,
        canonical_first_sine_compensating_geometry_stage_residual_budget_digest=(
            canonical_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_residual_budget() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageResidualBudgetReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_residual_budget_report(
        budget_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget(),
        stage_ladder_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder(),
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageResidualBudgetReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_residual_budget_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_residual_budget",
]
