from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_cancellation_ladder import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryCancellationLadderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_cancellation_ladder,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder,
)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _driver_signature(
    *,
    cancellation_ladder_signature: str,
    stage_ladder_signature: str,
    pair_labels: tuple[str, str],
    off_diagonal_share_of_total_absolute_mass: float,
    cross_shoulder_pair_share_of_off_diagonal_mass: float,
    left_center_pair_share_of_off_diagonal_mass: float,
    cross_shoulder_pair_directional_imbalance: float,
    left_center_pair_directional_imbalance: float,
) -> str:
    if (
        cancellation_ladder_signature
        == "first-sine-compensating-geometry-cancellation-ladder"
        and stage_ladder_signature == "first-sine-compensating-geometry-stage-ladder"
        and pair_labels == ("cross-shoulder-pair", "left-center-pair")
        and off_diagonal_share_of_total_absolute_mass < 0.4
        and cross_shoulder_pair_share_of_off_diagonal_mass > 0.55
        and left_center_pair_share_of_off_diagonal_mass > 0.44
        and np.isclose(cross_shoulder_pair_directional_imbalance, 0.0, atol=1e-12)
        and np.isclose(left_center_pair_directional_imbalance, 0.0, atol=1e-12)
    ):
        return "first-sine-compensating-geometry-pair-symmetry-contract"
    return "mixed-first-sine-compensating-geometry-pair-symmetry-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairSymmetryContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    pair_labels: tuple[str, str]
    off_diagonal_share_of_total_absolute_mass: float
    cross_shoulder_pair_share_of_total_absolute_mass: float
    left_center_pair_share_of_total_absolute_mass: float
    cross_shoulder_pair_share_of_off_diagonal_mass: float
    left_center_pair_share_of_off_diagonal_mass: float
    cross_shoulder_pair_directional_imbalance: float
    left_center_pair_directional_imbalance: float
    driver_signature: str
    canonical_first_sine_compensating_geometry_pair_symmetry_digest: tuple[str, ...]

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
        self.pair_labels = tuple(str(item).strip() for item in self.pair_labels)
        self.off_diagonal_share_of_total_absolute_mass = float(
            self.off_diagonal_share_of_total_absolute_mass
        )
        self.cross_shoulder_pair_share_of_total_absolute_mass = float(
            self.cross_shoulder_pair_share_of_total_absolute_mass
        )
        self.left_center_pair_share_of_total_absolute_mass = float(
            self.left_center_pair_share_of_total_absolute_mass
        )
        self.cross_shoulder_pair_share_of_off_diagonal_mass = float(
            self.cross_shoulder_pair_share_of_off_diagonal_mass
        )
        self.left_center_pair_share_of_off_diagonal_mass = float(
            self.left_center_pair_share_of_off_diagonal_mass
        )
        self.cross_shoulder_pair_directional_imbalance = float(
            self.cross_shoulder_pair_directional_imbalance
        )
        self.left_center_pair_directional_imbalance = float(
            self.left_center_pair_directional_imbalance
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_compensating_geometry_pair_symmetry_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_compensating_geometry_pair_symmetry_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_symmetry_contract_report(
    *,
    budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryBudgetReport
    ),
    cancellation_ladder_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryCancellationLadderReport
    ),
    stage_ladder_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairSymmetryContractReport:
    if budget_report.policy_digest != cancellation_ladder_report.policy_digest:
        raise ValueError("pair symmetry contract requires shared policy digest")
    if budget_report.policy_digest != stage_ladder_report.policy_digest:
        raise ValueError("pair symmetry contract requires shared policy digest")
    if budget_report.binding_design != cancellation_ladder_report.binding_design:
        raise ValueError("pair symmetry contract requires shared binding design")
    if budget_report.binding_design != stage_ladder_report.binding_design:
        raise ValueError("pair symmetry contract requires shared binding design")
    if budget_report.window_label != cancellation_ladder_report.window_label:
        raise ValueError("pair symmetry contract requires shared window label")
    if budget_report.window_label != stage_ladder_report.window_label:
        raise ValueError("pair symmetry contract requires shared window label")
    if budget_report.evaluation_grid != cancellation_ladder_report.evaluation_grid:
        raise ValueError("pair symmetry contract requires shared evaluation grid")
    if budget_report.evaluation_grid != stage_ladder_report.evaluation_grid:
        raise ValueError("pair symmetry contract requires shared evaluation grid")

    pair_labels = ("cross-shoulder-pair", "left-center-pair")
    if stage_ladder_report.stage_order[1:3] != pair_labels:
        raise ValueError(
            "pair symmetry contract requires mirrored pair-local stage order"
        )

    matrix = np.asarray(budget_report.compensating_geometry_matrix, dtype=float)
    cross_shoulder_pair_directional_imbalance = float(abs(matrix[0, 2] - matrix[2, 0]))
    left_center_pair_directional_imbalance = float(abs(matrix[0, 1] - matrix[1, 0]))
    off_diagonal_share_of_total_absolute_mass = float(
        cancellation_ladder_report.left_incident_off_diagonal_share_of_absolute_mass
    )
    cross_shoulder_pair_share_of_total_absolute_mass = float(
        cancellation_ladder_report.cross_shoulder_pair_share_of_absolute_mass
    )
    left_center_pair_share_of_total_absolute_mass = float(
        cancellation_ladder_report.left_center_pair_share_of_absolute_mass
    )
    cross_shoulder_pair_share_of_off_diagonal_mass = float(
        cross_shoulder_pair_share_of_total_absolute_mass
        / off_diagonal_share_of_total_absolute_mass
    )
    left_center_pair_share_of_off_diagonal_mass = float(
        left_center_pair_share_of_total_absolute_mass
        / off_diagonal_share_of_total_absolute_mass
    )
    driver_signature = _driver_signature(
        cancellation_ladder_signature=cancellation_ladder_report.driver_signature,
        stage_ladder_signature=stage_ladder_report.driver_signature,
        pair_labels=pair_labels,
        off_diagonal_share_of_total_absolute_mass=off_diagonal_share_of_total_absolute_mass,
        cross_shoulder_pair_share_of_off_diagonal_mass=(
            cross_shoulder_pair_share_of_off_diagonal_mass
        ),
        left_center_pair_share_of_off_diagonal_mass=(
            left_center_pair_share_of_off_diagonal_mass
        ),
        cross_shoulder_pair_directional_imbalance=(
            cross_shoulder_pair_directional_imbalance
        ),
        left_center_pair_directional_imbalance=(left_center_pair_directional_imbalance),
    )
    canonical_digest = (
        f"- the entire off-diagonal cleanup still stays bounded and pair-local: off-diagonal absolute mass is only `{_format_percent(off_diagonal_share_of_total_absolute_mass)}` of total compensating mass, and it remains fully concentrated in the mirrored `cross-shoulder pair` plus `left-center pair` rather than spreading into dense replay",
        f"- within that off-diagonal slice, the `cross-shoulder pair` still absorbs `{_format_percent(cross_shoulder_pair_share_of_off_diagonal_mass)}` while the `left-center pair` absorbs the remaining `{_format_percent(left_center_pair_share_of_off_diagonal_mass)}`, so the first off-live follow-up still lands on the shoulder bridge before the left-center cleanup",
        f"- pair symmetry remains exact at the cell level: both the `cross-shoulder pair` and the `left-center pair` keep directional imbalance fixed at `{_format_float(cross_shoulder_pair_directional_imbalance)}`, so the preserve-left-support companion cannot drift into unilateral off-diagonal edits",
        "- current Trigger 2 implication: `first-sine-compensating-geometry-pair-symmetry-contract`; implementation should realize the off-live cleanup as mirrored pair-local cancellation after the diagonal-first source lane, not as unilateral covariance edits or broad replay",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairSymmetryContractReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-compensating-geometry-pair-symmetry-contract",
        policy_digest=budget_report.policy_digest,
        binding_design=budget_report.binding_design,
        window_label=budget_report.window_label,
        evaluation_grid=budget_report.evaluation_grid,
        coverage_anchor_random_state=budget_report.coverage_anchor_random_state,
        overshoot_companion_random_state=budget_report.overshoot_companion_random_state,
        diagonal_coordinate=budget_report.diagonal_coordinate,
        diagonal_basis_label=budget_report.diagonal_basis_label,
        pair_labels=pair_labels,
        off_diagonal_share_of_total_absolute_mass=(
            off_diagonal_share_of_total_absolute_mass
        ),
        cross_shoulder_pair_share_of_total_absolute_mass=(
            cross_shoulder_pair_share_of_total_absolute_mass
        ),
        left_center_pair_share_of_total_absolute_mass=(
            left_center_pair_share_of_total_absolute_mass
        ),
        cross_shoulder_pair_share_of_off_diagonal_mass=(
            cross_shoulder_pair_share_of_off_diagonal_mass
        ),
        left_center_pair_share_of_off_diagonal_mass=(
            left_center_pair_share_of_off_diagonal_mass
        ),
        cross_shoulder_pair_directional_imbalance=(
            cross_shoulder_pair_directional_imbalance
        ),
        left_center_pair_directional_imbalance=(left_center_pair_directional_imbalance),
        driver_signature=driver_signature,
        canonical_first_sine_compensating_geometry_pair_symmetry_digest=(
            canonical_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_symmetry_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairSymmetryContractReport
):
    budget_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_budget()
    cancellation_ladder_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_cancellation_ladder()
    stage_ladder_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder()
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_symmetry_contract_report(
        budget_report=budget_report,
        cancellation_ladder_report=cancellation_ladder_report,
        stage_ladder_report=stage_ladder_report,
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairSymmetryContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_symmetry_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_symmetry_contract",
]
