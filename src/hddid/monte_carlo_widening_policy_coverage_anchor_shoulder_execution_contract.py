from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_gap_budget_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryGapBudgetReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_gap_budget_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_priority_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPriorityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_priority_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_repair_acceptance_guard_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairAcceptanceGuardReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_acceptance_guard_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _driver_signature(
    *,
    correlation_repair_lane_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport
    ),
    entry_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPriorityReport
    ),
    gap_budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryGapBudgetReport
    ),
    repair_acceptance_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairAcceptanceGuardReport
    ),
) -> str:
    if (
        correlation_repair_lane_report.driver_signature
        == "directional-covariance-entry-repair-lane"
        and entry_priority_report.driver_signature
        == "directional-right-center-first-repair-priority"
        and gap_budget_report.driver_signature
        == "bounded-right-center-gap-budget-localization"
        and repair_acceptance_guard_report.driver_signature
        == "bounded-right-center-repair-acceptance-guard"
        and correlation_repair_lane_report.coverage_anchor_local_scale_access_shortfall_share
        > 0.0
        and entry_priority_report.anchor_left_shoulder_reserve > 0.0
        and entry_priority_report.anchor_failing_right_shoulder_reserve < 0.0
        and gap_budget_report.required_increment_share_of_entry_gap < 0.1
        and repair_acceptance_guard_report.zero_partial_right_center_correlation < -1.0
    ):
        return "bounded-right-center-execution-contract"
    return "mixed-right-center-execution-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    coverage_anchor_local_scale_access_shortfall_share: float
    current_anchor_correlation: float
    required_repaired_correlation: float
    correlation_repair_increment: float
    required_gap_share: float
    anchor_left_shoulder_reserve: float
    anchor_failing_right_shoulder_reserve: float
    anchor_right_to_left_covariance_share: float
    companion_right_to_left_covariance_share: float
    directional_flip_ratio: float
    current_abs_right_center_covariance: float
    required_abs_right_center_covariance: float
    required_incremental_right_center_covariance_lift: float
    required_increment_share_of_entry_gap: float
    required_increment_share_of_right_row_gap: float
    required_increment_share_of_center_column_gap: float
    residual_entry_gap_share_after_required_increment: float
    zero_partial_right_center_correlation: float
    zero_partial_lower_bound_gap: float
    required_repair_distance_to_zero_partial_target: float
    shoulder_sigma_log_gap_share: float
    center_sigma_log_gap_share: float
    correlation_log_gap_share: float
    correlation_vs_shoulder_sigma_ratio: float
    correlation_vs_center_sigma_ratio: float
    driver_signature: str
    canonical_execution_contract_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.left_shoulder_grid_value = float(self.left_shoulder_grid_value)
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.coverage_anchor_local_scale_access_shortfall_share = float(
            self.coverage_anchor_local_scale_access_shortfall_share
        )
        self.current_anchor_correlation = float(self.current_anchor_correlation)
        self.required_repaired_correlation = float(self.required_repaired_correlation)
        self.correlation_repair_increment = float(self.correlation_repair_increment)
        self.required_gap_share = float(self.required_gap_share)
        self.anchor_left_shoulder_reserve = float(self.anchor_left_shoulder_reserve)
        self.anchor_failing_right_shoulder_reserve = float(
            self.anchor_failing_right_shoulder_reserve
        )
        self.anchor_right_to_left_covariance_share = float(
            self.anchor_right_to_left_covariance_share
        )
        self.companion_right_to_left_covariance_share = float(
            self.companion_right_to_left_covariance_share
        )
        self.directional_flip_ratio = float(self.directional_flip_ratio)
        self.current_abs_right_center_covariance = float(
            self.current_abs_right_center_covariance
        )
        self.required_abs_right_center_covariance = float(
            self.required_abs_right_center_covariance
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.required_increment_share_of_entry_gap = float(
            self.required_increment_share_of_entry_gap
        )
        self.required_increment_share_of_right_row_gap = float(
            self.required_increment_share_of_right_row_gap
        )
        self.required_increment_share_of_center_column_gap = float(
            self.required_increment_share_of_center_column_gap
        )
        self.residual_entry_gap_share_after_required_increment = float(
            self.residual_entry_gap_share_after_required_increment
        )
        self.zero_partial_right_center_correlation = float(
            self.zero_partial_right_center_correlation
        )
        self.zero_partial_lower_bound_gap = float(self.zero_partial_lower_bound_gap)
        self.required_repair_distance_to_zero_partial_target = float(
            self.required_repair_distance_to_zero_partial_target
        )
        self.shoulder_sigma_log_gap_share = float(self.shoulder_sigma_log_gap_share)
        self.center_sigma_log_gap_share = float(self.center_sigma_log_gap_share)
        self.correlation_log_gap_share = float(self.correlation_log_gap_share)
        self.correlation_vs_shoulder_sigma_ratio = float(
            self.correlation_vs_shoulder_sigma_ratio
        )
        self.correlation_vs_center_sigma_ratio = float(
            self.correlation_vs_center_sigma_ratio
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_execution_contract_digest = tuple(
            str(line).rstrip() for line in self.canonical_execution_contract_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "left_shoulder_grid_value": self.left_shoulder_grid_value,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "coverage_anchor_local_scale_access_shortfall_share": (
                self.coverage_anchor_local_scale_access_shortfall_share
            ),
            "current_anchor_correlation": self.current_anchor_correlation,
            "required_repaired_correlation": self.required_repaired_correlation,
            "correlation_repair_increment": self.correlation_repair_increment,
            "required_gap_share": self.required_gap_share,
            "anchor_left_shoulder_reserve": self.anchor_left_shoulder_reserve,
            "anchor_failing_right_shoulder_reserve": (
                self.anchor_failing_right_shoulder_reserve
            ),
            "anchor_right_to_left_covariance_share": (
                self.anchor_right_to_left_covariance_share
            ),
            "companion_right_to_left_covariance_share": (
                self.companion_right_to_left_covariance_share
            ),
            "directional_flip_ratio": self.directional_flip_ratio,
            "current_abs_right_center_covariance": self.current_abs_right_center_covariance,
            "required_abs_right_center_covariance": (
                self.required_abs_right_center_covariance
            ),
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "required_increment_share_of_entry_gap": (
                self.required_increment_share_of_entry_gap
            ),
            "required_increment_share_of_right_row_gap": (
                self.required_increment_share_of_right_row_gap
            ),
            "required_increment_share_of_center_column_gap": (
                self.required_increment_share_of_center_column_gap
            ),
            "residual_entry_gap_share_after_required_increment": (
                self.residual_entry_gap_share_after_required_increment
            ),
            "zero_partial_right_center_correlation": (
                self.zero_partial_right_center_correlation
            ),
            "zero_partial_lower_bound_gap": self.zero_partial_lower_bound_gap,
            "required_repair_distance_to_zero_partial_target": (
                self.required_repair_distance_to_zero_partial_target
            ),
            "shoulder_sigma_log_gap_share": self.shoulder_sigma_log_gap_share,
            "center_sigma_log_gap_share": self.center_sigma_log_gap_share,
            "correlation_log_gap_share": self.correlation_log_gap_share,
            "correlation_vs_shoulder_sigma_ratio": (
                self.correlation_vs_shoulder_sigma_ratio
            ),
            "correlation_vs_center_sigma_ratio": (
                self.correlation_vs_center_sigma_ratio
            ),
            "driver_signature": self.driver_signature,
            "canonical_execution_contract_digest": list(
                self.canonical_execution_contract_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract_report(
    *,
    correlation_repair_lane_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport
    ),
    entry_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPriorityReport
    ),
    gap_budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryGapBudgetReport
    ),
    repair_acceptance_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairAcceptanceGuardReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport:
    if (
        correlation_repair_lane_report.policy_digest
        != entry_priority_report.policy_digest
    ):
        raise ValueError("execution contract requires a shared policy digest")
    if correlation_repair_lane_report.policy_digest != gap_budget_report.policy_digest:
        raise ValueError("execution contract requires a shared policy digest")
    if (
        correlation_repair_lane_report.policy_digest
        != repair_acceptance_guard_report.policy_digest
    ):
        raise ValueError("execution contract requires a shared policy digest")
    if (
        correlation_repair_lane_report.binding_design
        != entry_priority_report.binding_design
    ):
        raise ValueError("execution contract requires a shared binding design")
    if (
        correlation_repair_lane_report.binding_design
        != gap_budget_report.binding_design
    ):
        raise ValueError("execution contract requires a shared binding design")
    if (
        correlation_repair_lane_report.binding_design
        != repair_acceptance_guard_report.binding_design
    ):
        raise ValueError("execution contract requires a shared binding design")
    if (
        correlation_repair_lane_report.window_label
        != entry_priority_report.window_label
    ):
        raise ValueError("execution contract requires a shared window label")
    if correlation_repair_lane_report.window_label != gap_budget_report.window_label:
        raise ValueError("execution contract requires a shared window label")
    if (
        correlation_repair_lane_report.window_label
        != repair_acceptance_guard_report.window_label
    ):
        raise ValueError("execution contract requires a shared window label")
    if (
        correlation_repair_lane_report.coverage_anchor_random_state
        != entry_priority_report.coverage_anchor_random_state
        or correlation_repair_lane_report.coverage_anchor_random_state
        != gap_budget_report.coverage_anchor_random_state
        or correlation_repair_lane_report.coverage_anchor_random_state
        != repair_acceptance_guard_report.coverage_anchor_random_state
    ):
        raise ValueError("execution contract requires the same coverage-anchor seed")
    if (
        correlation_repair_lane_report.overshoot_companion_random_state
        != entry_priority_report.overshoot_companion_random_state
        or correlation_repair_lane_report.overshoot_companion_random_state
        != gap_budget_report.overshoot_companion_random_state
        or correlation_repair_lane_report.overshoot_companion_random_state
        != repair_acceptance_guard_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "execution contract requires the same overshoot-companion seed"
        )
    if not isclose(
        entry_priority_report.left_shoulder_grid_value,
        repair_acceptance_guard_report.left_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError("execution contract requires the same left shoulder grid")
    if (
        not isclose(
            correlation_repair_lane_report.center_grid_value,
            entry_priority_report.center_grid_value,
            abs_tol=1e-12,
        )
        or not isclose(
            correlation_repair_lane_report.center_grid_value,
            gap_budget_report.center_grid_value,
            abs_tol=1e-12,
        )
        or not isclose(
            correlation_repair_lane_report.center_grid_value,
            repair_acceptance_guard_report.center_grid_value,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("execution contract requires the same center grid")
    if (
        not isclose(
            correlation_repair_lane_report.failing_right_shoulder_grid_value,
            entry_priority_report.failing_right_shoulder_grid_value,
            abs_tol=1e-12,
        )
        or not isclose(
            correlation_repair_lane_report.failing_right_shoulder_grid_value,
            gap_budget_report.failing_right_shoulder_grid_value,
            abs_tol=1e-12,
        )
        or not isclose(
            correlation_repair_lane_report.failing_right_shoulder_grid_value,
            repair_acceptance_guard_report.failing_right_shoulder_grid_value,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "execution contract requires the same failing right shoulder grid"
        )
    if not isclose(
        correlation_repair_lane_report.current_abs_shoulder_center_covariance,
        entry_priority_report.current_abs_right_center_covariance,
        abs_tol=1e-12,
    ) or not isclose(
        correlation_repair_lane_report.current_abs_shoulder_center_covariance,
        gap_budget_report.current_abs_right_center_covariance,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "execution contract requires the same current right-center covariance"
        )
    if not isclose(
        correlation_repair_lane_report.required_abs_shoulder_center_covariance_at_fixed_center_sigma,
        entry_priority_report.required_abs_right_center_covariance,
        abs_tol=1e-12,
    ) or not isclose(
        correlation_repair_lane_report.required_abs_shoulder_center_covariance_at_fixed_center_sigma,
        gap_budget_report.required_abs_right_center_covariance,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "execution contract requires the same required right-center covariance"
        )
    if (
        not isclose(
            correlation_repair_lane_report.zero_partial_right_center_correlation,
            repair_acceptance_guard_report.zero_partial_right_center_correlation,
            abs_tol=1e-12,
        )
        or not isclose(
            correlation_repair_lane_report.zero_partial_lower_bound_gap,
            repair_acceptance_guard_report.zero_partial_lower_bound_gap,
            abs_tol=1e-12,
        )
        or not isclose(
            correlation_repair_lane_report.required_repair_distance_to_zero_partial_target,
            repair_acceptance_guard_report.required_repair_distance_to_zero_partial_target,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "execution contract requires feasibility and acceptance-guard identity"
        )

    driver_signature = _driver_signature(
        correlation_repair_lane_report=correlation_repair_lane_report,
        entry_priority_report=entry_priority_report,
        gap_budget_report=gap_budget_report,
        repair_acceptance_guard_report=repair_acceptance_guard_report,
    )

    canonical_digest = (
        f"- binding design `DGP2/500/50` on `{correlation_repair_lane_report.window_label}`: the last local-scale miss still stays `{_format_percent(correlation_repair_lane_report.coverage_anchor_local_scale_access_shortfall_share)}` at failing right shoulder `z = {_format_grid_value(correlation_repair_lane_report.failing_right_shoulder_grid_value)}`, yet the live lane only needs right-center correlation to rise from `{_format_float(correlation_repair_lane_report.current_anchor_correlation)}` to `{_format_float(correlation_repair_lane_report.required_repaired_correlation)}` (increment `{_format_signed_float(correlation_repair_lane_report.correlation_repair_increment)}`, `{_format_percent(correlation_repair_lane_report.required_gap_share)}` of the full gap) and `|covariance({_format_grid_value(correlation_repair_lane_report.failing_right_shoulder_grid_value)}, {_format_grid_value(correlation_repair_lane_report.center_grid_value)})|` to rise from `{_format_float(entry_priority_report.current_abs_right_center_covariance)}` to `{_format_float(entry_priority_report.required_abs_right_center_covariance)}` (increment `{_format_signed_float(entry_priority_report.required_incremental_right_center_covariance_lift)}`) while left shoulder `z = {_format_grid_value(entry_priority_report.left_shoulder_grid_value)}` still keeps `{_format_signed_float(entry_priority_report.anchor_left_shoulder_reserve)}` reserve and the right shoulder stays at `{_format_signed_float(entry_priority_report.anchor_failing_right_shoulder_reserve)}`",
        f"- the miss therefore remains directional and right-specific: anchor seed `{entry_priority_report.coverage_anchor_random_state}` keeps only `{_format_percent(entry_priority_report.anchor_right_to_left_covariance_share)}` of left coupled access on the right side, whereas companion seed `{entry_priority_report.overshoot_companion_random_state}` restores the right side to `{_format_percent(entry_priority_report.companion_right_to_left_covariance_share)}` of left access (`{_format_ratio(entry_priority_report.directional_flip_ratio)}` directional flip), so the live repair priority still points to the bounded `z = {_format_grid_value(entry_priority_report.failing_right_shoulder_grid_value)} -> z = {_format_grid_value(entry_priority_report.center_grid_value)}` entry",
        f"- the required bounded lift stays tiny relative to companion geometry: `{_format_signed_float(entry_priority_report.required_incremental_right_center_covariance_lift)}` consumes only `{_format_percent(gap_budget_report.required_increment_share_of_entry_gap)}` of the single-entry gap, `{_format_percent(gap_budget_report.required_increment_share_of_right_row_gap)}` of the right-row gap, and `{_format_percent(gap_budget_report.required_increment_share_of_center_column_gap)}` of the center-column gap, leaving `{_format_percent(gap_budget_report.residual_entry_gap_share_after_required_increment)}` / `{_format_percent(gap_budget_report.residual_right_row_gap_share_after_required_increment)}` / `{_format_percent(gap_budget_report.residual_center_column_gap_share_after_required_increment)}` of those companion gaps outside the current lane even after repair",
        f"- out-of-lane targets remain explicitly excluded: zeroing the center-conditioned direct residual would require right-center correlation `{_format_float(repair_acceptance_guard_report.zero_partial_right_center_correlation)}`, i.e. `{_format_float(repair_acceptance_guard_report.zero_partial_lower_bound_gap)}` below the feasible lower bound `-1.000` and `{_format_float(repair_acceptance_guard_report.required_repair_distance_to_zero_partial_target)}` away from the bounded repair target, while shoulder / center `sigma_z_hat` still explain only `{_format_percent(repair_acceptance_guard_report.shoulder_sigma_log_gap_share)}` / `{_format_percent(repair_acceptance_guard_report.center_sigma_log_gap_share)}` of the log gap versus `{_format_percent(repair_acceptance_guard_report.correlation_log_gap_share)}` from correlation",
        f"- current Trigger 2 implication: `{driver_signature}`; next implementation should raise only the bounded `z = {_format_grid_value(entry_priority_report.failing_right_shoulder_grid_value)} -> z = {_format_grid_value(entry_priority_report.center_grid_value)}` entry while preserving left-side support and leaving sign-healing, denominator compression, and whole-row / whole-column replay outside the live repair lane",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "execution-contract"
        ),
        policy_digest=correlation_repair_lane_report.policy_digest,
        binding_design=correlation_repair_lane_report.binding_design,
        window_label=correlation_repair_lane_report.window_label,
        coverage_anchor_random_state=(
            correlation_repair_lane_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            correlation_repair_lane_report.overshoot_companion_random_state
        ),
        left_shoulder_grid_value=entry_priority_report.left_shoulder_grid_value,
        center_grid_value=correlation_repair_lane_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            correlation_repair_lane_report.failing_right_shoulder_grid_value
        ),
        coverage_anchor_local_scale_access_shortfall_share=(
            correlation_repair_lane_report.coverage_anchor_local_scale_access_shortfall_share
        ),
        current_anchor_correlation=correlation_repair_lane_report.current_anchor_correlation,
        required_repaired_correlation=(
            correlation_repair_lane_report.required_repaired_correlation
        ),
        correlation_repair_increment=(
            correlation_repair_lane_report.correlation_repair_increment
        ),
        required_gap_share=correlation_repair_lane_report.required_gap_share,
        anchor_left_shoulder_reserve=entry_priority_report.anchor_left_shoulder_reserve,
        anchor_failing_right_shoulder_reserve=(
            entry_priority_report.anchor_failing_right_shoulder_reserve
        ),
        anchor_right_to_left_covariance_share=(
            entry_priority_report.anchor_right_to_left_covariance_share
        ),
        companion_right_to_left_covariance_share=(
            entry_priority_report.companion_right_to_left_covariance_share
        ),
        directional_flip_ratio=entry_priority_report.directional_flip_ratio,
        current_abs_right_center_covariance=(
            entry_priority_report.current_abs_right_center_covariance
        ),
        required_abs_right_center_covariance=(
            entry_priority_report.required_abs_right_center_covariance
        ),
        required_incremental_right_center_covariance_lift=(
            entry_priority_report.required_incremental_right_center_covariance_lift
        ),
        required_increment_share_of_entry_gap=(
            gap_budget_report.required_increment_share_of_entry_gap
        ),
        required_increment_share_of_right_row_gap=(
            gap_budget_report.required_increment_share_of_right_row_gap
        ),
        required_increment_share_of_center_column_gap=(
            gap_budget_report.required_increment_share_of_center_column_gap
        ),
        residual_entry_gap_share_after_required_increment=(
            gap_budget_report.residual_entry_gap_share_after_required_increment
        ),
        zero_partial_right_center_correlation=(
            repair_acceptance_guard_report.zero_partial_right_center_correlation
        ),
        zero_partial_lower_bound_gap=(
            repair_acceptance_guard_report.zero_partial_lower_bound_gap
        ),
        required_repair_distance_to_zero_partial_target=(
            repair_acceptance_guard_report.required_repair_distance_to_zero_partial_target
        ),
        shoulder_sigma_log_gap_share=(
            repair_acceptance_guard_report.shoulder_sigma_log_gap_share
        ),
        center_sigma_log_gap_share=(
            repair_acceptance_guard_report.center_sigma_log_gap_share
        ),
        correlation_log_gap_share=(
            repair_acceptance_guard_report.correlation_log_gap_share
        ),
        correlation_vs_shoulder_sigma_ratio=(
            repair_acceptance_guard_report.correlation_vs_shoulder_sigma_ratio
        ),
        correlation_vs_center_sigma_ratio=(
            repair_acceptance_guard_report.correlation_vs_center_sigma_ratio
        ),
        driver_signature=driver_signature,
        canonical_execution_contract_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract_report(
        correlation_repair_lane_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot()
        ),
        entry_priority_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_priority_probe()
        ),
        gap_budget_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_gap_budget_probe()
        ),
        repair_acceptance_guard_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_acceptance_guard_probe()
        ),
    )
