from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_residual_budget import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageResidualBudgetReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_residual_budget,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_center_residual_dominance_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftCenterResidualDominanceProbeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_center_residual_dominance_probe,
)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{float(value):.3f}x"


def _driver_signature(
    *,
    stage_ladder_signature: str,
    stage_residual_budget_signature: str,
    left_center_dominance_signature: str,
    cumulative_share_after_left_center_pair: float,
    left_left_tail_share_of_total_absolute_mass: float,
    left_left_tail_share_of_total_frobenius_norm: float,
    tail_share_of_post_cross_absolute_residual: float,
    tail_share_of_post_cross_frobenius_residual: float,
) -> str:
    if (
        stage_ladder_signature == "first-sine-compensating-geometry-stage-ladder"
        and stage_residual_budget_signature
        == "first-sine-compensating-geometry-stage-residual-budget"
        and left_center_dominance_signature
        == "first-sine-left-center-residual-dominance"
        and cumulative_share_after_left_center_pair > 0.96
        and left_left_tail_share_of_total_absolute_mass < 0.04
        and left_left_tail_share_of_total_frobenius_norm < 0.08
        and tail_share_of_post_cross_absolute_residual < 0.17
        and tail_share_of_post_cross_frobenius_residual < 0.27
    ):
        return "first-sine-left-left-tail-terminal-contract"
    return "mixed-first-sine-left-left-tail-terminal-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftLeftTailTerminalContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    stage_order: tuple[str, ...]
    cumulative_share_after_left_center_pair: float
    left_left_tail_absolute_mass: float
    left_left_tail_share_of_total_absolute_mass: float
    left_left_tail_frobenius_norm: float
    left_left_tail_share_of_total_frobenius_norm: float
    tail_share_of_post_cross_absolute_residual: float
    tail_share_of_post_cross_frobenius_residual: float
    left_center_absolute_multiple_of_tail: float
    left_center_frobenius_multiple_of_tail: float
    driver_signature: str
    canonical_first_sine_left_left_tail_terminal_contract_digest: tuple[str, ...]

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
        self.stage_order = tuple(str(item).strip() for item in self.stage_order)
        self.cumulative_share_after_left_center_pair = float(
            self.cumulative_share_after_left_center_pair
        )
        self.left_left_tail_absolute_mass = float(self.left_left_tail_absolute_mass)
        self.left_left_tail_share_of_total_absolute_mass = float(
            self.left_left_tail_share_of_total_absolute_mass
        )
        self.left_left_tail_frobenius_norm = float(self.left_left_tail_frobenius_norm)
        self.left_left_tail_share_of_total_frobenius_norm = float(
            self.left_left_tail_share_of_total_frobenius_norm
        )
        self.tail_share_of_post_cross_absolute_residual = float(
            self.tail_share_of_post_cross_absolute_residual
        )
        self.tail_share_of_post_cross_frobenius_residual = float(
            self.tail_share_of_post_cross_frobenius_residual
        )
        self.left_center_absolute_multiple_of_tail = float(
            self.left_center_absolute_multiple_of_tail
        )
        self.left_center_frobenius_multiple_of_tail = float(
            self.left_center_frobenius_multiple_of_tail
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_left_left_tail_terminal_contract_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_left_left_tail_terminal_contract_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_left_tail_terminal_contract_report(
    *,
    stage_ladder_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport
        | None
    ) = None,
    stage_residual_budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageResidualBudgetReport
        | None
    ) = None,
    left_center_dominance_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftCenterResidualDominanceProbeReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftLeftTailTerminalContractReport:
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
    resolved_left_center_dominance = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_center_residual_dominance_probe()
        if left_center_dominance_report is None
        else left_center_dominance_report
    )

    if (
        resolved_stage_ladder.policy_digest
        != resolved_stage_residual_budget.policy_digest
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared policy digest"
        )
    if (
        resolved_stage_ladder.policy_digest
        != resolved_left_center_dominance.policy_digest
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared policy digest"
        )
    if (
        resolved_stage_ladder.binding_design
        != resolved_stage_residual_budget.binding_design
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared binding design"
        )
    if (
        resolved_stage_ladder.binding_design
        != resolved_left_center_dominance.binding_design
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared binding design"
        )
    if (
        resolved_stage_ladder.window_label
        != resolved_stage_residual_budget.window_label
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared window label"
        )
    if (
        resolved_stage_ladder.window_label
        != resolved_left_center_dominance.window_label
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared window label"
        )
    if (
        resolved_stage_ladder.evaluation_grid
        != resolved_stage_residual_budget.evaluation_grid
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared evaluation grid"
        )
    if (
        resolved_stage_ladder.evaluation_grid
        != resolved_left_center_dominance.evaluation_grid
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared evaluation grid"
        )
    if (
        resolved_stage_ladder.coverage_anchor_random_state
        != resolved_stage_residual_budget.coverage_anchor_random_state
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared coverage-anchor seed"
        )
    if (
        resolved_stage_ladder.coverage_anchor_random_state
        != resolved_left_center_dominance.coverage_anchor_random_state
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared coverage-anchor seed"
        )
    if (
        resolved_stage_ladder.overshoot_companion_random_state
        != resolved_stage_residual_budget.overshoot_companion_random_state
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared overshoot-companion seed"
        )
    if (
        resolved_stage_ladder.overshoot_companion_random_state
        != resolved_left_center_dominance.overshoot_companion_random_state
    ):
        raise ValueError(
            "left-left tail terminal contract requires shared overshoot-companion seed"
        )
    if resolved_stage_ladder.stage_order != resolved_left_center_dominance.stage_order:
        raise ValueError("left-left tail terminal contract requires shared stage order")

    cumulative_share_after_left_center_pair = (
        resolved_stage_ladder.cumulative_share_of_total_absolute_mass[2]
    )
    left_left_tail_share_of_total_absolute_mass = (
        resolved_stage_ladder.stage_share_of_total_absolute_mass[3]
    )
    left_left_tail_absolute_mass = (
        resolved_stage_residual_budget.remaining_absolute_mass_after_stage[2]
    )
    left_left_tail_frobenius_norm = (
        resolved_stage_residual_budget.remaining_frobenius_norm_after_stage[2]
    )
    left_left_tail_share_of_total_frobenius_norm = resolved_stage_residual_budget.remaining_share_of_total_frobenius_norm_after_stage[
        2
    ]
    tail_share_of_post_cross_absolute_residual = (
        left_left_tail_share_of_total_absolute_mass
        / resolved_stage_residual_budget.remaining_share_of_total_absolute_mass_after_stage[
            1
        ]
    )
    tail_share_of_post_cross_frobenius_residual = (
        left_left_tail_share_of_total_frobenius_norm
        / resolved_stage_residual_budget.remaining_share_of_total_frobenius_norm_after_stage[
            1
        ]
    )

    driver_signature = _driver_signature(
        stage_ladder_signature=resolved_stage_ladder.driver_signature,
        stage_residual_budget_signature=resolved_stage_residual_budget.driver_signature,
        left_center_dominance_signature=resolved_left_center_dominance.driver_signature,
        cumulative_share_after_left_center_pair=cumulative_share_after_left_center_pair,
        left_left_tail_share_of_total_absolute_mass=(
            left_left_tail_share_of_total_absolute_mass
        ),
        left_left_tail_share_of_total_frobenius_norm=(
            left_left_tail_share_of_total_frobenius_norm
        ),
        tail_share_of_post_cross_absolute_residual=(
            tail_share_of_post_cross_absolute_residual
        ),
        tail_share_of_post_cross_frobenius_residual=(
            tail_share_of_post_cross_frobenius_residual
        ),
    )
    canonical_digest = (
        f"- after the symmetric left-center pair, cumulative cancellation coverage already reaches `{_format_percent(cumulative_share_after_left_center_pair)}` of total absolute mass while only `{_format_percent(left_left_tail_share_of_total_frobenius_norm)}` of Frobenius norm remains unresolved, so the preserve-left-support companion is already a near-complete validation-only witness",
        f"- the unrepaired terminal `left-left` cell now carries only `{_format_percent(left_left_tail_share_of_total_absolute_mass)}` of total absolute mass, which is just `{_format_percent(tail_share_of_post_cross_absolute_residual)}` of the post-cross-shoulder absolute residual and `{_format_percent(tail_share_of_post_cross_frobenius_residual)}` of the post-cross-shoulder Frobenius residual",
        f"- the tail therefore no longer defines another material cleanup stage: it sits `{_format_ratio(resolved_left_center_dominance.left_center_absolute_multiple_of_tail)}` below the prior `left-center pair` in absolute mass and `{_format_ratio(resolved_left_center_dominance.left_center_frobenius_multiple_of_tail)}` below it in Frobenius norm, so future implementation should stop at left-center completion rather than widen into dense replay to chase the terminal cell",
        "- current Trigger 2 implication: `first-sine-left-left-tail-terminal-contract`; preserve-left-support follow-up may treat the terminal `left-left` residual as bounded tail-only remainder, not as justification for a new material replay stage or live-routing promotion",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftLeftTailTerminalContractReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-left-left-tail-terminal-contract",
        policy_digest=resolved_stage_ladder.policy_digest,
        binding_design=resolved_stage_ladder.binding_design,
        window_label=resolved_stage_ladder.window_label,
        evaluation_grid=resolved_stage_ladder.evaluation_grid,
        coverage_anchor_random_state=(
            resolved_stage_ladder.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            resolved_stage_ladder.overshoot_companion_random_state
        ),
        stage_order=resolved_stage_ladder.stage_order,
        cumulative_share_after_left_center_pair=(
            cumulative_share_after_left_center_pair
        ),
        left_left_tail_absolute_mass=left_left_tail_absolute_mass,
        left_left_tail_share_of_total_absolute_mass=(
            left_left_tail_share_of_total_absolute_mass
        ),
        left_left_tail_frobenius_norm=left_left_tail_frobenius_norm,
        left_left_tail_share_of_total_frobenius_norm=(
            left_left_tail_share_of_total_frobenius_norm
        ),
        tail_share_of_post_cross_absolute_residual=(
            tail_share_of_post_cross_absolute_residual
        ),
        tail_share_of_post_cross_frobenius_residual=(
            tail_share_of_post_cross_frobenius_residual
        ),
        left_center_absolute_multiple_of_tail=(
            resolved_left_center_dominance.left_center_absolute_multiple_of_tail
        ),
        left_center_frobenius_multiple_of_tail=(
            resolved_left_center_dominance.left_center_frobenius_multiple_of_tail
        ),
        driver_signature=driver_signature,
        canonical_first_sine_left_left_tail_terminal_contract_digest=(canonical_digest),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_left_tail_terminal_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftLeftTailTerminalContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_left_tail_terminal_contract_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineLeftLeftTailTerminalContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_left_tail_terminal_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_left_left_tail_terminal_contract",
]
