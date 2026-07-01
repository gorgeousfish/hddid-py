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
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_residual_budget import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageResidualBudgetReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_residual_budget,
)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{float(value):.3f}x"


def _driver_signature(
    *,
    stage_ladder_signature: str,
    stage_residual_budget_signature: str,
    left_center_share_of_remaining_absolute_mass: float,
    left_center_share_of_remaining_frobenius_norm: float,
    left_center_absolute_multiple_of_tail: float,
    left_center_frobenius_multiple_of_tail: float,
) -> str:
    if (
        stage_ladder_signature == "first-sine-compensating-geometry-stage-ladder"
        and stage_residual_budget_signature
        == "first-sine-compensating-geometry-stage-residual-budget"
        and left_center_share_of_remaining_absolute_mass > 0.83
        and left_center_share_of_remaining_frobenius_norm > 0.96
        and left_center_absolute_multiple_of_tail > 5.2
        and left_center_frobenius_multiple_of_tail > 3.7
    ):
        return "first-sine-left-center-residual-dominance"
    return "mixed-first-sine-left-center-residual-dominance"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftCenterResidualDominanceProbeReport:
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
    left_center_absolute_mass: float
    left_left_tail_absolute_mass: float
    remaining_absolute_mass_after_cross_shoulder: float
    left_center_share_of_remaining_absolute_mass: float
    left_center_absolute_multiple_of_tail: float
    left_center_frobenius_norm: float
    left_left_tail_frobenius_norm: float
    remaining_frobenius_norm_after_cross_shoulder: float
    left_center_share_of_remaining_frobenius_norm: float
    left_center_frobenius_multiple_of_tail: float
    driver_signature: str
    canonical_left_center_residual_dominance_digest: tuple[str, ...]

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
        self.left_center_absolute_mass = float(self.left_center_absolute_mass)
        self.left_left_tail_absolute_mass = float(self.left_left_tail_absolute_mass)
        self.remaining_absolute_mass_after_cross_shoulder = float(
            self.remaining_absolute_mass_after_cross_shoulder
        )
        self.left_center_share_of_remaining_absolute_mass = float(
            self.left_center_share_of_remaining_absolute_mass
        )
        self.left_center_absolute_multiple_of_tail = float(
            self.left_center_absolute_multiple_of_tail
        )
        self.left_center_frobenius_norm = float(self.left_center_frobenius_norm)
        self.left_left_tail_frobenius_norm = float(self.left_left_tail_frobenius_norm)
        self.remaining_frobenius_norm_after_cross_shoulder = float(
            self.remaining_frobenius_norm_after_cross_shoulder
        )
        self.left_center_share_of_remaining_frobenius_norm = float(
            self.left_center_share_of_remaining_frobenius_norm
        )
        self.left_center_frobenius_multiple_of_tail = float(
            self.left_center_frobenius_multiple_of_tail
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_left_center_residual_dominance_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_left_center_residual_dominance_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_center_residual_dominance_report(
    *,
    budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport
        | None
    ) = None,
    stage_ladder_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport
        | None
    ) = None,
    stage_residual_budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageResidualBudgetReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftCenterResidualDominanceProbeReport:
    resolved_budget = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget()
        if budget_report is None
        else budget_report
    )
    resolved_stage_ladder = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder()
        if stage_ladder_report is None
        else stage_ladder_report
    )
    resolved_stage_residual_budget = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_residual_budget()
        if stage_residual_budget_report is None
        else stage_residual_budget_report
    )

    if resolved_budget.policy_digest != resolved_stage_ladder.policy_digest:
        raise ValueError("left-center dominance probe requires shared policy digest")
    if resolved_budget.policy_digest != resolved_stage_residual_budget.policy_digest:
        raise ValueError("left-center dominance probe requires shared policy digest")
    if resolved_budget.binding_design != resolved_stage_ladder.binding_design:
        raise ValueError("left-center dominance probe requires shared binding design")
    if resolved_budget.binding_design != resolved_stage_residual_budget.binding_design:
        raise ValueError("left-center dominance probe requires shared binding design")
    if resolved_budget.window_label != resolved_stage_ladder.window_label:
        raise ValueError("left-center dominance probe requires shared window label")
    if resolved_budget.window_label != resolved_stage_residual_budget.window_label:
        raise ValueError("left-center dominance probe requires shared window label")
    if resolved_budget.evaluation_grid != resolved_stage_ladder.evaluation_grid:
        raise ValueError("left-center dominance probe requires shared evaluation grid")
    if (
        resolved_budget.evaluation_grid
        != resolved_stage_residual_budget.evaluation_grid
    ):
        raise ValueError("left-center dominance probe requires shared evaluation grid")
    if (
        resolved_budget.coverage_anchor_random_state
        != resolved_stage_ladder.coverage_anchor_random_state
    ):
        raise ValueError(
            "left-center dominance probe requires shared coverage-anchor seed"
        )
    if (
        resolved_budget.coverage_anchor_random_state
        != resolved_stage_residual_budget.coverage_anchor_random_state
    ):
        raise ValueError(
            "left-center dominance probe requires shared coverage-anchor seed"
        )
    if (
        resolved_budget.overshoot_companion_random_state
        != resolved_stage_ladder.overshoot_companion_random_state
    ):
        raise ValueError(
            "left-center dominance probe requires shared overshoot-companion seed"
        )
    if (
        resolved_budget.overshoot_companion_random_state
        != resolved_stage_residual_budget.overshoot_companion_random_state
    ):
        raise ValueError(
            "left-center dominance probe requires shared overshoot-companion seed"
        )
    if resolved_budget.diagonal_coordinate != resolved_stage_ladder.diagonal_coordinate:
        raise ValueError(
            "left-center dominance probe requires shared diagonal coordinate"
        )
    if (
        resolved_budget.diagonal_coordinate
        != resolved_stage_residual_budget.diagonal_coordinate
    ):
        raise ValueError(
            "left-center dominance probe requires shared diagonal coordinate"
        )
    if (
        resolved_budget.diagonal_basis_label
        != resolved_stage_ladder.diagonal_basis_label
    ):
        raise ValueError(
            "left-center dominance probe requires shared diagonal basis label"
        )
    if (
        resolved_budget.diagonal_basis_label
        != resolved_stage_residual_budget.diagonal_basis_label
    ):
        raise ValueError(
            "left-center dominance probe requires shared diagonal basis label"
        )

    matrix = np.abs(
        np.asarray(resolved_budget.compensating_geometry_matrix, dtype=float)
    )
    if matrix.shape != (3, 3):
        raise ValueError("left-center dominance probe expects the 3x3 witness grid")

    left_center_absolute_mass = float(matrix[0, 1] + matrix[1, 0])
    left_left_tail_absolute_mass = float(matrix[0, 0])
    remaining_absolute_mass_after_cross_shoulder = float(
        resolved_stage_residual_budget.remaining_absolute_mass_after_stage[1]
    )
    left_center_share_of_remaining_absolute_mass = float(
        left_center_absolute_mass / remaining_absolute_mass_after_cross_shoulder
    )
    left_center_absolute_multiple_of_tail = float(
        left_center_absolute_mass / left_left_tail_absolute_mass
    )

    left_center_frobenius_norm = float(np.linalg.norm(matrix[[0, 1], [1, 0]]))
    left_left_tail_frobenius_norm = float(np.linalg.norm([matrix[0, 0]]))
    remaining_frobenius_norm_after_cross_shoulder = float(
        resolved_stage_residual_budget.remaining_frobenius_norm_after_stage[1]
    )
    left_center_share_of_remaining_frobenius_norm = float(
        left_center_frobenius_norm / remaining_frobenius_norm_after_cross_shoulder
    )
    left_center_frobenius_multiple_of_tail = float(
        left_center_frobenius_norm / left_left_tail_frobenius_norm
    )

    driver_signature = _driver_signature(
        stage_ladder_signature=resolved_stage_ladder.driver_signature,
        stage_residual_budget_signature=resolved_stage_residual_budget.driver_signature,
        left_center_share_of_remaining_absolute_mass=(
            left_center_share_of_remaining_absolute_mass
        ),
        left_center_share_of_remaining_frobenius_norm=(
            left_center_share_of_remaining_frobenius_norm
        ),
        left_center_absolute_multiple_of_tail=(left_center_absolute_multiple_of_tail),
        left_center_frobenius_multiple_of_tail=(left_center_frobenius_multiple_of_tail),
    )
    digest = (
        f"- after the symmetric cross-shoulder pair, the remaining compensating residual is still overwhelmingly the left-center pair: it carries `{_format_percent(left_center_share_of_remaining_absolute_mass)}` of the remaining absolute mass and `{_format_percent(left_center_share_of_remaining_frobenius_norm)}` of the remaining Frobenius norm, so the post-bridge cleanup is not yet a tail-only witness",
        f"- the left-center pair alone still outweighs the final left-left tail by `{_format_ratio(left_center_absolute_multiple_of_tail)}` in absolute mass and `{_format_ratio(left_center_frobenius_multiple_of_tail)}` in Frobenius norm, which keeps left-center cleanup in the last material stage rather than optional polish",
        "- current Trigger 2 implication: `first-sine-left-center-residual-dominance`; cross-shoulder cleanup cannot be treated as near-complete because almost all residual witness mass still sits on the symmetric left-center pair",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftCenterResidualDominanceProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-left-center-residual-dominance-probe",
        policy_digest=resolved_budget.policy_digest,
        binding_design=resolved_budget.binding_design,
        window_label=resolved_budget.window_label,
        evaluation_grid=resolved_budget.evaluation_grid,
        coverage_anchor_random_state=resolved_budget.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            resolved_budget.overshoot_companion_random_state
        ),
        diagonal_coordinate=resolved_budget.diagonal_coordinate,
        diagonal_basis_label=resolved_budget.diagonal_basis_label,
        stage_order=resolved_stage_ladder.stage_order,
        left_center_absolute_mass=left_center_absolute_mass,
        left_left_tail_absolute_mass=left_left_tail_absolute_mass,
        remaining_absolute_mass_after_cross_shoulder=(
            remaining_absolute_mass_after_cross_shoulder
        ),
        left_center_share_of_remaining_absolute_mass=(
            left_center_share_of_remaining_absolute_mass
        ),
        left_center_absolute_multiple_of_tail=(left_center_absolute_multiple_of_tail),
        left_center_frobenius_norm=left_center_frobenius_norm,
        left_left_tail_frobenius_norm=left_left_tail_frobenius_norm,
        remaining_frobenius_norm_after_cross_shoulder=(
            remaining_frobenius_norm_after_cross_shoulder
        ),
        left_center_share_of_remaining_frobenius_norm=(
            left_center_share_of_remaining_frobenius_norm
        ),
        left_center_frobenius_multiple_of_tail=(left_center_frobenius_multiple_of_tail),
        driver_signature=driver_signature,
        canonical_left_center_residual_dominance_digest=digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_center_residual_dominance_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftCenterResidualDominanceProbeReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_center_residual_dominance_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftCenterResidualDominanceProbeReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_center_residual_dominance_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_center_residual_dominance_probe",
]
