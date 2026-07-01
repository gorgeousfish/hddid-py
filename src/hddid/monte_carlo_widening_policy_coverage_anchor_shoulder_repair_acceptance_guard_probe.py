from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryFactorReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_priority_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPriorityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_priority_probe,
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
    entry_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPriorityReport
    ),
    direct_residual_feasibility_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport
    ),
    covariance_entry_factor_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryFactorReport
    ),
) -> str:
    if (
        entry_priority_report.driver_signature
        == "directional-right-center-first-repair-priority"
        and direct_residual_feasibility_report.driver_signature
        == "sign-healing-outside-correlation-feasible-repair-lane"
        and covariance_entry_factor_report.driver_signature
        == "correlation-led-covariance-entry-repair-target"
        and entry_priority_report.anchor_left_shoulder_reserve > 0.0
        and entry_priority_report.anchor_failing_right_shoulder_reserve < 0.0
        and direct_residual_feasibility_report.zero_partial_right_center_correlation
        < -1.0
        and covariance_entry_factor_report.correlation_log_gap_share
        > covariance_entry_factor_report.shoulder_sigma_log_gap_share
        and covariance_entry_factor_report.correlation_log_gap_share
        > covariance_entry_factor_report.center_sigma_log_gap_share
    ):
        return "bounded-right-center-repair-acceptance-guard"
    return "mixed-repair-acceptance-guard"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairAcceptanceGuardReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    anchor_left_shoulder_reserve: float
    anchor_failing_right_shoulder_reserve: float
    current_anchor_partial_cross_shoulder_correlation: float
    repaired_anchor_partial_cross_shoulder_correlation: float
    zero_partial_right_center_correlation: float
    zero_partial_lower_bound_gap: float
    required_repair_distance_to_zero_partial_target: float
    current_abs_right_center_covariance: float
    required_abs_right_center_covariance: float
    required_incremental_right_center_covariance_lift: float
    required_incremental_share_of_companion_right_row_mass: float
    required_incremental_share_of_companion_center_column_mass: float
    shoulder_sigma_log_gap_share: float
    center_sigma_log_gap_share: float
    correlation_log_gap_share: float
    correlation_vs_shoulder_sigma_ratio: float
    correlation_vs_center_sigma_ratio: float
    driver_signature: str
    canonical_repair_acceptance_guard_digest: tuple[str, ...]

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
        self.anchor_left_shoulder_reserve = float(self.anchor_left_shoulder_reserve)
        self.anchor_failing_right_shoulder_reserve = float(
            self.anchor_failing_right_shoulder_reserve
        )
        self.current_anchor_partial_cross_shoulder_correlation = float(
            self.current_anchor_partial_cross_shoulder_correlation
        )
        self.repaired_anchor_partial_cross_shoulder_correlation = float(
            self.repaired_anchor_partial_cross_shoulder_correlation
        )
        self.zero_partial_right_center_correlation = float(
            self.zero_partial_right_center_correlation
        )
        self.zero_partial_lower_bound_gap = float(self.zero_partial_lower_bound_gap)
        self.required_repair_distance_to_zero_partial_target = float(
            self.required_repair_distance_to_zero_partial_target
        )
        self.current_abs_right_center_covariance = float(
            self.current_abs_right_center_covariance
        )
        self.required_abs_right_center_covariance = float(
            self.required_abs_right_center_covariance
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.required_incremental_share_of_companion_right_row_mass = float(
            self.required_incremental_share_of_companion_right_row_mass
        )
        self.required_incremental_share_of_companion_center_column_mass = float(
            self.required_incremental_share_of_companion_center_column_mass
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
        self.canonical_repair_acceptance_guard_digest = tuple(
            str(line).rstrip() for line in self.canonical_repair_acceptance_guard_digest
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
            "anchor_left_shoulder_reserve": self.anchor_left_shoulder_reserve,
            "anchor_failing_right_shoulder_reserve": (
                self.anchor_failing_right_shoulder_reserve
            ),
            "current_anchor_partial_cross_shoulder_correlation": (
                self.current_anchor_partial_cross_shoulder_correlation
            ),
            "repaired_anchor_partial_cross_shoulder_correlation": (
                self.repaired_anchor_partial_cross_shoulder_correlation
            ),
            "zero_partial_right_center_correlation": (
                self.zero_partial_right_center_correlation
            ),
            "zero_partial_lower_bound_gap": self.zero_partial_lower_bound_gap,
            "required_repair_distance_to_zero_partial_target": (
                self.required_repair_distance_to_zero_partial_target
            ),
            "current_abs_right_center_covariance": (
                self.current_abs_right_center_covariance
            ),
            "required_abs_right_center_covariance": (
                self.required_abs_right_center_covariance
            ),
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "required_incremental_share_of_companion_right_row_mass": (
                self.required_incremental_share_of_companion_right_row_mass
            ),
            "required_incremental_share_of_companion_center_column_mass": (
                self.required_incremental_share_of_companion_center_column_mass
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
            "canonical_repair_acceptance_guard_digest": list(
                self.canonical_repair_acceptance_guard_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_acceptance_guard_report(
    *,
    entry_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPriorityReport
    ),
    direct_residual_feasibility_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport
    ),
    covariance_entry_factor_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryFactorReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairAcceptanceGuardReport:
    if (
        entry_priority_report.policy_digest
        != direct_residual_feasibility_report.policy_digest
    ):
        raise ValueError("repair acceptance guard requires a shared policy digest")
    if (
        entry_priority_report.policy_digest
        != covariance_entry_factor_report.policy_digest
    ):
        raise ValueError("repair acceptance guard requires a shared policy digest")
    if (
        entry_priority_report.binding_design
        != direct_residual_feasibility_report.binding_design
    ):
        raise ValueError("repair acceptance guard requires a shared binding design")
    if (
        entry_priority_report.binding_design
        != covariance_entry_factor_report.binding_design
    ):
        raise ValueError("repair acceptance guard requires a shared binding design")
    if (
        entry_priority_report.coverage_anchor_random_state
        != direct_residual_feasibility_report.coverage_anchor_random_state
        or entry_priority_report.coverage_anchor_random_state
        != covariance_entry_factor_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "repair acceptance guard requires the same coverage-anchor seed"
        )
    if (
        entry_priority_report.overshoot_companion_random_state
        != direct_residual_feasibility_report.overshoot_companion_random_state
        or entry_priority_report.overshoot_companion_random_state
        != covariance_entry_factor_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "repair acceptance guard requires the same overshoot-companion seed"
        )
    if not isclose(
        entry_priority_report.center_grid_value,
        direct_residual_feasibility_report.center_grid_value,
        abs_tol=1e-12,
    ) or not isclose(
        entry_priority_report.center_grid_value,
        covariance_entry_factor_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError("repair acceptance guard requires the same center grid")
    if not isclose(
        entry_priority_report.left_shoulder_grid_value,
        direct_residual_feasibility_report.left_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError("repair acceptance guard requires the same left shoulder grid")
    if not isclose(
        entry_priority_report.failing_right_shoulder_grid_value,
        direct_residual_feasibility_report.failing_right_shoulder_grid_value,
        abs_tol=1e-12,
    ) or not isclose(
        entry_priority_report.failing_right_shoulder_grid_value,
        covariance_entry_factor_report.failing_right_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "repair acceptance guard requires the same failing shoulder grid"
        )
    if (
        direct_residual_feasibility_report.required_incremental_share_of_companion_right_row_mass
        if hasattr(
            direct_residual_feasibility_report,
            "required_incremental_share_of_companion_right_row_mass",
        )
        else None
    ) is not None:
        raise ValueError(
            "repair acceptance guard should source bounded mass shares from the entry-priority report"
        )

    driver_signature = _driver_signature(
        entry_priority_report=entry_priority_report,
        direct_residual_feasibility_report=direct_residual_feasibility_report,
        covariance_entry_factor_report=covariance_entry_factor_report,
    )
    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: left shoulder "
        f"`z = {_format_grid_value(entry_priority_report.left_shoulder_grid_value)}` still keeps "
        f"`{_format_signed_float(entry_priority_report.anchor_left_shoulder_reserve)}` reserve while failing "
        f"right shoulder `z = {_format_grid_value(entry_priority_report.failing_right_shoulder_grid_value)}` "
        f"stays at `{_format_signed_float(entry_priority_report.anchor_failing_right_shoulder_reserve)}`, "
        "so any future Trigger 2 fix must preserve already-positive left support and keep the live miss scoped to the right-center lane",
        "- sign healing remains explicitly out of lane: zeroing the center-conditioned direct "
        "residual would require right-center correlation "
        f"`{_format_float(direct_residual_feasibility_report.zero_partial_right_center_correlation)}`, "
        f"i.e. `{_format_float(direct_residual_feasibility_report.zero_partial_lower_bound_gap)}` below the "
        "feasible lower bound `-1.000`; even the bounded repair target keeps anchor partial "
        "cross-shoulder correlation negative at "
        f"`{_format_float(direct_residual_feasibility_report.repaired_anchor_partial_cross_shoulder_correlation)}` "
        "versus "
        f"`{_format_float(direct_residual_feasibility_report.current_anchor_partial_cross_shoulder_correlation)}` today",
        "- bounded right-center repair still stays local: "
        f"`|covariance({_format_grid_value(entry_priority_report.failing_right_shoulder_grid_value)}, "
        f"{_format_grid_value(entry_priority_report.center_grid_value)})|` only needs to rise from "
        f"`{_format_float(entry_priority_report.current_abs_right_center_covariance)}` to "
        f"`{_format_float(entry_priority_report.required_abs_right_center_covariance)}` "
        f"(increment `{_format_signed_float(entry_priority_report.required_incremental_right_center_covariance_lift)}`), "
        "and that increment is still just "
        f"`{_format_percent(entry_priority_report.required_incremental_share_of_companion_right_row_mass)}` / "
        f"`{_format_percent(entry_priority_report.required_incremental_share_of_companion_center_column_mass)}` "
        "of companion right-row / center-column covariance mass, so full row or column replay would exceed the minimal repair lane",
        "- the live knob is correlation-led rather than denominator-led: shoulder / center "
        f"`sigma_z_hat` contribute `{_format_percent(covariance_entry_factor_report.shoulder_sigma_log_gap_share)}` / "
        f"`{_format_percent(covariance_entry_factor_report.center_sigma_log_gap_share)}` of the covariance-entry log gap, "
        "while shoulder-center correlation contributes the dominant "
        f"`{_format_percent(covariance_entry_factor_report.correlation_log_gap_share)}`; "
        "correlation amplification still outpaces shoulder / center scale widening by "
        f"`{_format_ratio(covariance_entry_factor_report.correlation_vs_shoulder_sigma_ratio)}` / "
        f"`{_format_ratio(covariance_entry_factor_report.correlation_vs_center_sigma_ratio)}`",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; source-level follow-up should raise only the bounded "
        f"`z = {_format_grid_value(entry_priority_report.failing_right_shoulder_grid_value)} -> "
        f"z = {_format_grid_value(entry_priority_report.center_grid_value)}` entry while preserving "
        "left-center support and treating sign-healing, denominator compression, and full local-window replay as out-of-lane side quests",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairAcceptanceGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "repair-acceptance-guard-probe"
        ),
        policy_digest=entry_priority_report.policy_digest,
        binding_design=entry_priority_report.binding_design,
        window_label=entry_priority_report.window_label,
        coverage_anchor_random_state=entry_priority_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            entry_priority_report.overshoot_companion_random_state
        ),
        left_shoulder_grid_value=entry_priority_report.left_shoulder_grid_value,
        center_grid_value=entry_priority_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            entry_priority_report.failing_right_shoulder_grid_value
        ),
        anchor_left_shoulder_reserve=entry_priority_report.anchor_left_shoulder_reserve,
        anchor_failing_right_shoulder_reserve=(
            entry_priority_report.anchor_failing_right_shoulder_reserve
        ),
        current_anchor_partial_cross_shoulder_correlation=(
            direct_residual_feasibility_report.current_anchor_partial_cross_shoulder_correlation
        ),
        repaired_anchor_partial_cross_shoulder_correlation=(
            direct_residual_feasibility_report.repaired_anchor_partial_cross_shoulder_correlation
        ),
        zero_partial_right_center_correlation=(
            direct_residual_feasibility_report.zero_partial_right_center_correlation
        ),
        zero_partial_lower_bound_gap=(
            direct_residual_feasibility_report.zero_partial_lower_bound_gap
        ),
        required_repair_distance_to_zero_partial_target=(
            direct_residual_feasibility_report.required_repair_distance_to_zero_partial_target
        ),
        current_abs_right_center_covariance=(
            entry_priority_report.current_abs_right_center_covariance
        ),
        required_abs_right_center_covariance=(
            entry_priority_report.required_abs_right_center_covariance
        ),
        required_incremental_right_center_covariance_lift=(
            entry_priority_report.required_incremental_right_center_covariance_lift
        ),
        required_incremental_share_of_companion_right_row_mass=(
            entry_priority_report.required_incremental_share_of_companion_right_row_mass
        ),
        required_incremental_share_of_companion_center_column_mass=(
            entry_priority_report.required_incremental_share_of_companion_center_column_mass
        ),
        shoulder_sigma_log_gap_share=(
            covariance_entry_factor_report.shoulder_sigma_log_gap_share
        ),
        center_sigma_log_gap_share=(
            covariance_entry_factor_report.center_sigma_log_gap_share
        ),
        correlation_log_gap_share=(
            covariance_entry_factor_report.correlation_log_gap_share
        ),
        correlation_vs_shoulder_sigma_ratio=(
            covariance_entry_factor_report.correlation_vs_shoulder_sigma_ratio
        ),
        correlation_vs_center_sigma_ratio=(
            covariance_entry_factor_report.correlation_vs_center_sigma_ratio
        ),
        driver_signature=driver_signature,
        canonical_repair_acceptance_guard_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_acceptance_guard_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairAcceptanceGuardReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_acceptance_guard_report(
        entry_priority_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_priority_probe()
        ),
        direct_residual_feasibility_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_probe()
        ),
        covariance_entry_factor_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe()
        ),
    )
