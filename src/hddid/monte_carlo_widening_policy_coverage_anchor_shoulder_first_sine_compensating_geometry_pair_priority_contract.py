from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_symmetry_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairSymmetryContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_symmetry_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder,
)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{float(value):.3f}x"


def _driver_signature(
    *,
    pair_symmetry_signature: str,
    stage_ladder_signature: str,
    pair_labels: tuple[str, str],
    cross_shoulder_pair_share_of_off_diagonal_mass: float,
    left_center_pair_share_of_off_diagonal_mass: float,
    cross_shoulder_priority_margin_share_of_off_diagonal_mass: float,
    cumulative_share_after_cross_shoulder_stage: float,
    left_left_residual_share_of_total_absolute_mass: float,
) -> str:
    if (
        pair_symmetry_signature
        == "first-sine-compensating-geometry-pair-symmetry-contract"
        and stage_ladder_signature == "first-sine-compensating-geometry-stage-ladder"
        and pair_labels == ("cross-shoulder-pair", "left-center-pair")
        and cross_shoulder_pair_share_of_off_diagonal_mass
        > left_center_pair_share_of_off_diagonal_mass
        and cross_shoulder_priority_margin_share_of_off_diagonal_mass > 0.1
        and cumulative_share_after_cross_shoulder_stage > 0.79
        and left_left_residual_share_of_total_absolute_mass < 0.04
    ):
        return "first-sine-compensating-geometry-pair-priority-contract"
    return "mixed-first-sine-compensating-geometry-pair-priority-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairPriorityContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    pair_labels: tuple[str, str]
    off_diagonal_share_of_total_absolute_mass: float
    cross_shoulder_pair_share_of_total_absolute_mass: float
    left_center_pair_share_of_total_absolute_mass: float
    cross_shoulder_pair_share_of_off_diagonal_mass: float
    left_center_pair_share_of_off_diagonal_mass: float
    cross_shoulder_priority_margin_share_of_total_absolute_mass: float
    cross_shoulder_priority_margin_share_of_off_diagonal_mass: float
    cross_shoulder_to_left_center_priority_ratio: float
    cumulative_share_after_cross_shoulder_stage: float
    remaining_share_after_cross_shoulder_stage: float
    left_left_residual_share_of_total_absolute_mass: float
    driver_signature: str
    canonical_first_sine_compensating_geometry_pair_priority_digest: tuple[str, ...]

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
        self.cross_shoulder_priority_margin_share_of_total_absolute_mass = float(
            self.cross_shoulder_priority_margin_share_of_total_absolute_mass
        )
        self.cross_shoulder_priority_margin_share_of_off_diagonal_mass = float(
            self.cross_shoulder_priority_margin_share_of_off_diagonal_mass
        )
        self.cross_shoulder_to_left_center_priority_ratio = float(
            self.cross_shoulder_to_left_center_priority_ratio
        )
        self.cumulative_share_after_cross_shoulder_stage = float(
            self.cumulative_share_after_cross_shoulder_stage
        )
        self.remaining_share_after_cross_shoulder_stage = float(
            self.remaining_share_after_cross_shoulder_stage
        )
        self.left_left_residual_share_of_total_absolute_mass = float(
            self.left_left_residual_share_of_total_absolute_mass
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_compensating_geometry_pair_priority_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_compensating_geometry_pair_priority_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_priority_contract_report(
    *,
    pair_symmetry_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairSymmetryContractReport
    ),
    stage_ladder_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairPriorityContractReport:
    if pair_symmetry_report.policy_digest != stage_ladder_report.policy_digest:
        raise ValueError("pair priority contract requires shared policy digest")
    if pair_symmetry_report.binding_design != stage_ladder_report.binding_design:
        raise ValueError("pair priority contract requires shared binding design")
    if pair_symmetry_report.window_label != stage_ladder_report.window_label:
        raise ValueError("pair priority contract requires shared window label")
    if pair_symmetry_report.evaluation_grid != stage_ladder_report.evaluation_grid:
        raise ValueError("pair priority contract requires shared evaluation grid")
    if (
        pair_symmetry_report.coverage_anchor_random_state
        != stage_ladder_report.coverage_anchor_random_state
    ):
        raise ValueError("pair priority contract requires shared coverage-anchor seed")
    if (
        pair_symmetry_report.overshoot_companion_random_state
        != stage_ladder_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "pair priority contract requires shared overshoot-companion seed"
        )

    cross_shoulder_priority_margin_share_of_total_absolute_mass = (
        pair_symmetry_report.cross_shoulder_pair_share_of_total_absolute_mass
        - pair_symmetry_report.left_center_pair_share_of_total_absolute_mass
    )
    cross_shoulder_priority_margin_share_of_off_diagonal_mass = (
        pair_symmetry_report.cross_shoulder_pair_share_of_off_diagonal_mass
        - pair_symmetry_report.left_center_pair_share_of_off_diagonal_mass
    )
    cross_shoulder_to_left_center_priority_ratio = (
        pair_symmetry_report.cross_shoulder_pair_share_of_off_diagonal_mass
        / pair_symmetry_report.left_center_pair_share_of_off_diagonal_mass
    )
    cumulative_share_after_cross_shoulder_stage = (
        stage_ladder_report.cumulative_share_of_total_absolute_mass[1]
    )
    remaining_share_after_cross_shoulder_stage = (
        1.0 - cumulative_share_after_cross_shoulder_stage
    )
    left_left_residual_share_of_total_absolute_mass = (
        stage_ladder_report.stage_share_of_total_absolute_mass[3]
    )
    driver_signature = _driver_signature(
        pair_symmetry_signature=pair_symmetry_report.driver_signature,
        stage_ladder_signature=stage_ladder_report.driver_signature,
        pair_labels=pair_symmetry_report.pair_labels,
        cross_shoulder_pair_share_of_off_diagonal_mass=(
            pair_symmetry_report.cross_shoulder_pair_share_of_off_diagonal_mass
        ),
        left_center_pair_share_of_off_diagonal_mass=(
            pair_symmetry_report.left_center_pair_share_of_off_diagonal_mass
        ),
        cross_shoulder_priority_margin_share_of_off_diagonal_mass=(
            cross_shoulder_priority_margin_share_of_off_diagonal_mass
        ),
        cumulative_share_after_cross_shoulder_stage=(
            cumulative_share_after_cross_shoulder_stage
        ),
        left_left_residual_share_of_total_absolute_mass=(
            left_left_residual_share_of_total_absolute_mass
        ),
    )
    canonical_digest = (
        f"- the off-live compensating geometry remains pair-local after the diagonal-first source lane: off-diagonal absolute mass is only `{_format_percent(pair_symmetry_report.off_diagonal_share_of_total_absolute_mass)}` of total compensating mass, and all of that slice still sits inside the mirrored `cross-shoulder pair` plus `left-center pair` rather than dense replay",
        f"- pair ordering stays asymmetric even though each pair is internally mirrored: the `cross-shoulder pair` still absorbs `{_format_percent(pair_symmetry_report.cross_shoulder_pair_share_of_off_diagonal_mass)}` of off-diagonal mass versus `{_format_percent(pair_symmetry_report.left_center_pair_share_of_off_diagonal_mass)}` for the `left-center pair`, a lead of `{_format_percent(cross_shoulder_priority_margin_share_of_off_diagonal_mass)}` of off-diagonal mass (`{_format_percent(cross_shoulder_priority_margin_share_of_total_absolute_mass)}` of total mass, `{_format_ratio(cross_shoulder_to_left_center_priority_ratio)}` larger)",
        f"- stage-ladder carryover therefore still lands on the shoulder bridge first: cumulative cancellation coverage already reaches `{_format_percent(cumulative_share_after_cross_shoulder_stage)}` after the `cross-shoulder pair`, leaving only `{_format_percent(remaining_share_after_cross_shoulder_stage)}` of total compensating mass outside that rung and just `{_format_percent(left_left_residual_share_of_total_absolute_mass)}` in the terminal left-left residual cell",
        "- current Trigger 2 implication: `first-sine-compensating-geometry-pair-priority-contract`; any preserve-left-support follow-up must neutralize the mirrored `cross-shoulder pair` before the mirrored `left-center pair`, and must keep this ordering validation-only rather than promoting it over live `bounded-right-center-execution-contract` routing",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairPriorityContractReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-compensating-geometry-pair-priority-contract",
        policy_digest=pair_symmetry_report.policy_digest,
        binding_design=pair_symmetry_report.binding_design,
        window_label=pair_symmetry_report.window_label,
        evaluation_grid=pair_symmetry_report.evaluation_grid,
        coverage_anchor_random_state=(
            pair_symmetry_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            pair_symmetry_report.overshoot_companion_random_state
        ),
        pair_labels=pair_symmetry_report.pair_labels,
        off_diagonal_share_of_total_absolute_mass=(
            pair_symmetry_report.off_diagonal_share_of_total_absolute_mass
        ),
        cross_shoulder_pair_share_of_total_absolute_mass=(
            pair_symmetry_report.cross_shoulder_pair_share_of_total_absolute_mass
        ),
        left_center_pair_share_of_total_absolute_mass=(
            pair_symmetry_report.left_center_pair_share_of_total_absolute_mass
        ),
        cross_shoulder_pair_share_of_off_diagonal_mass=(
            pair_symmetry_report.cross_shoulder_pair_share_of_off_diagonal_mass
        ),
        left_center_pair_share_of_off_diagonal_mass=(
            pair_symmetry_report.left_center_pair_share_of_off_diagonal_mass
        ),
        cross_shoulder_priority_margin_share_of_total_absolute_mass=(
            cross_shoulder_priority_margin_share_of_total_absolute_mass
        ),
        cross_shoulder_priority_margin_share_of_off_diagonal_mass=(
            cross_shoulder_priority_margin_share_of_off_diagonal_mass
        ),
        cross_shoulder_to_left_center_priority_ratio=(
            cross_shoulder_to_left_center_priority_ratio
        ),
        cumulative_share_after_cross_shoulder_stage=(
            cumulative_share_after_cross_shoulder_stage
        ),
        remaining_share_after_cross_shoulder_stage=(
            remaining_share_after_cross_shoulder_stage
        ),
        left_left_residual_share_of_total_absolute_mass=(
            left_left_residual_share_of_total_absolute_mass
        ),
        driver_signature=driver_signature,
        canonical_first_sine_compensating_geometry_pair_priority_digest=(
            canonical_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_priority_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairPriorityContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_priority_contract_report(
        pair_symmetry_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_symmetry_contract(),
        stage_ladder_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder(),
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairPriorityContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_priority_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_priority_contract",
]
