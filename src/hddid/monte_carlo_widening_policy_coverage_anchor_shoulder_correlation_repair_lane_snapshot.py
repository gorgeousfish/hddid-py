from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    correlation_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport
    ),
    feasibility_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport
    ),
    channel_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport
    ),
) -> str:
    if (
        correlation_gap_report.driver_signature
        == "partial-correlation-gap-bridge-target"
        and feasibility_report.driver_signature
        == "sign-healing-outside-correlation-feasible-repair-lane"
        and channel_report.driver_signature == "covariance-entry-first-repair-target"
    ):
        return "directional-covariance-entry-repair-lane"
    return "mixed-correlation-repair-lane"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    coverage_anchor_local_scale_access_shortfall_share: float
    current_anchor_correlation: float
    required_repaired_correlation: float
    correlation_repair_increment: float
    required_gap_share: float
    zero_partial_right_center_correlation: float
    zero_partial_lower_bound_gap: float
    required_repair_distance_to_zero_partial_target: float
    current_abs_shoulder_center_covariance: float
    required_abs_shoulder_center_covariance_at_fixed_center_sigma: float
    required_covariance_share_of_companion: float
    current_anchor_shoulder_sigma_z_hat: float
    required_shoulder_sigma_z_hat_at_fixed_covariance_and_center_sigma: float
    required_shoulder_sigma_share_of_anchor: float
    required_shoulder_sigma_share_of_companion: float
    current_anchor_center_sigma_z_hat: float
    required_center_sigma_z_hat_at_fixed_covariance: float
    required_center_sigma_share_of_anchor: float
    required_center_sigma_share_of_companion: float
    driver_signature: str
    canonical_correlation_repair_lane_snapshot_digest: tuple[str, ...]

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
        self.zero_partial_right_center_correlation = float(
            self.zero_partial_right_center_correlation
        )
        self.zero_partial_lower_bound_gap = float(self.zero_partial_lower_bound_gap)
        self.required_repair_distance_to_zero_partial_target = float(
            self.required_repair_distance_to_zero_partial_target
        )
        self.current_abs_shoulder_center_covariance = float(
            self.current_abs_shoulder_center_covariance
        )
        self.required_abs_shoulder_center_covariance_at_fixed_center_sigma = float(
            self.required_abs_shoulder_center_covariance_at_fixed_center_sigma
        )
        self.required_covariance_share_of_companion = float(
            self.required_covariance_share_of_companion
        )
        self.current_anchor_shoulder_sigma_z_hat = float(
            self.current_anchor_shoulder_sigma_z_hat
        )
        self.required_shoulder_sigma_z_hat_at_fixed_covariance_and_center_sigma = float(
            self.required_shoulder_sigma_z_hat_at_fixed_covariance_and_center_sigma
        )
        self.required_shoulder_sigma_share_of_anchor = float(
            self.required_shoulder_sigma_share_of_anchor
        )
        self.required_shoulder_sigma_share_of_companion = float(
            self.required_shoulder_sigma_share_of_companion
        )
        self.current_anchor_center_sigma_z_hat = float(
            self.current_anchor_center_sigma_z_hat
        )
        self.required_center_sigma_z_hat_at_fixed_covariance = float(
            self.required_center_sigma_z_hat_at_fixed_covariance
        )
        self.required_center_sigma_share_of_anchor = float(
            self.required_center_sigma_share_of_anchor
        )
        self.required_center_sigma_share_of_companion = float(
            self.required_center_sigma_share_of_companion
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_correlation_repair_lane_snapshot_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_correlation_repair_lane_snapshot_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "coverage_anchor_local_scale_access_shortfall_share": (
                self.coverage_anchor_local_scale_access_shortfall_share
            ),
            "current_anchor_correlation": self.current_anchor_correlation,
            "required_repaired_correlation": self.required_repaired_correlation,
            "correlation_repair_increment": self.correlation_repair_increment,
            "required_gap_share": self.required_gap_share,
            "zero_partial_right_center_correlation": (
                self.zero_partial_right_center_correlation
            ),
            "zero_partial_lower_bound_gap": self.zero_partial_lower_bound_gap,
            "required_repair_distance_to_zero_partial_target": (
                self.required_repair_distance_to_zero_partial_target
            ),
            "current_abs_shoulder_center_covariance": (
                self.current_abs_shoulder_center_covariance
            ),
            "required_abs_shoulder_center_covariance_at_fixed_center_sigma": (
                self.required_abs_shoulder_center_covariance_at_fixed_center_sigma
            ),
            "required_covariance_share_of_companion": (
                self.required_covariance_share_of_companion
            ),
            "current_anchor_shoulder_sigma_z_hat": (
                self.current_anchor_shoulder_sigma_z_hat
            ),
            "required_shoulder_sigma_z_hat_at_fixed_covariance_and_center_sigma": (
                self.required_shoulder_sigma_z_hat_at_fixed_covariance_and_center_sigma
            ),
            "required_shoulder_sigma_share_of_anchor": (
                self.required_shoulder_sigma_share_of_anchor
            ),
            "required_shoulder_sigma_share_of_companion": (
                self.required_shoulder_sigma_share_of_companion
            ),
            "current_anchor_center_sigma_z_hat": (
                self.current_anchor_center_sigma_z_hat
            ),
            "required_center_sigma_z_hat_at_fixed_covariance": (
                self.required_center_sigma_z_hat_at_fixed_covariance
            ),
            "required_center_sigma_share_of_anchor": (
                self.required_center_sigma_share_of_anchor
            ),
            "required_center_sigma_share_of_companion": (
                self.required_center_sigma_share_of_companion
            ),
            "driver_signature": self.driver_signature,
            "canonical_correlation_repair_lane_snapshot_digest": list(
                self.canonical_correlation_repair_lane_snapshot_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot_report(
    *,
    correlation_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport
    ),
    center_coupling_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport
    ),
    channel_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport
    ),
    feasibility_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport:
    if correlation_gap_report.policy_digest != center_coupling_report.policy_digest:
        raise ValueError(
            "correlation repair lane snapshot requires a shared policy digest"
        )
    if correlation_gap_report.policy_digest != channel_report.policy_digest:
        raise ValueError(
            "correlation repair lane snapshot requires a shared policy digest"
        )
    if correlation_gap_report.policy_digest != feasibility_report.policy_digest:
        raise ValueError(
            "correlation repair lane snapshot requires a shared policy digest"
        )
    if correlation_gap_report.binding_design != center_coupling_report.binding_design:
        raise ValueError(
            "correlation repair lane snapshot requires a shared binding design"
        )
    if correlation_gap_report.binding_design != channel_report.binding_design:
        raise ValueError(
            "correlation repair lane snapshot requires a shared binding design"
        )
    if correlation_gap_report.binding_design != feasibility_report.binding_design:
        raise ValueError(
            "correlation repair lane snapshot requires a shared binding design"
        )
    if (
        correlation_gap_report.coverage_anchor_random_state
        != center_coupling_report.coverage_anchor_random_state
        or correlation_gap_report.coverage_anchor_random_state
        != channel_report.coverage_anchor_random_state
        or correlation_gap_report.coverage_anchor_random_state
        != feasibility_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "correlation repair lane snapshot requires the same coverage-anchor seed"
        )
    if (
        correlation_gap_report.overshoot_companion_random_state
        != center_coupling_report.overshoot_companion_random_state
        or correlation_gap_report.overshoot_companion_random_state
        != channel_report.overshoot_companion_random_state
        or correlation_gap_report.overshoot_companion_random_state
        != feasibility_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "correlation repair lane snapshot requires the same overshoot-companion seed"
        )
    if not isclose(
        correlation_gap_report.center_grid_value,
        center_coupling_report.center_grid_value,
        abs_tol=1e-12,
    ) or not isclose(
        correlation_gap_report.center_grid_value,
        channel_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "correlation repair lane snapshot requires the same center grid value"
        )
    if not isclose(
        correlation_gap_report.center_grid_value,
        feasibility_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "correlation repair lane snapshot requires the same center grid value"
        )
    if not isclose(
        correlation_gap_report.failing_right_shoulder_grid_value,
        center_coupling_report.failing_right_shoulder_grid_value,
        abs_tol=1e-12,
    ) or not isclose(
        correlation_gap_report.failing_right_shoulder_grid_value,
        channel_report.failing_right_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "correlation repair lane snapshot requires the same failing shoulder grid value"
        )
    if not isclose(
        correlation_gap_report.failing_right_shoulder_grid_value,
        feasibility_report.failing_right_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "correlation repair lane snapshot requires the same failing shoulder grid value"
        )
    if not isclose(
        correlation_gap_report.current_anchor_correlation,
        center_coupling_report.coverage_anchor_shoulder_center_correlation,
        abs_tol=1e-12,
    ) or not isclose(
        correlation_gap_report.current_anchor_correlation,
        feasibility_report.current_anchor_right_center_correlation,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "correlation repair lane snapshot requires the same current anchor correlation"
        )
    if not isclose(
        correlation_gap_report.required_repaired_correlation,
        feasibility_report.required_repaired_right_center_correlation,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "correlation repair lane snapshot requires the same bounded repair target"
        )

    required_shoulder_sigma = float(
        channel_report.current_abs_shoulder_center_covariance
        / (
            correlation_gap_report.required_repaired_correlation
            * channel_report.coverage_anchor_center_sigma_z_hat
        )
    )
    required_shoulder_sigma_share_of_anchor = float(
        required_shoulder_sigma
        / center_coupling_report.coverage_anchor_shoulder_sigma_z_hat
    )
    required_shoulder_sigma_share_of_companion = float(
        required_shoulder_sigma
        / center_coupling_report.overshoot_companion_shoulder_sigma_z_hat
    )
    if not isclose(
        required_shoulder_sigma_share_of_anchor,
        channel_report.required_center_sigma_share_of_anchor,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "correlation repair lane snapshot requires denominator-share parity"
        )

    driver_signature = _driver_signature(
        correlation_gap_report=correlation_gap_report,
        feasibility_report=feasibility_report,
        channel_report=channel_report,
    )
    digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: coverage anchor seed "
        f"`{correlation_gap_report.coverage_anchor_random_state}` only needs right-center "
        f"correlation to rise from `{_format_float(correlation_gap_report.current_anchor_correlation)}` "
        f"to `{_format_float(correlation_gap_report.required_repaired_correlation)}` "
        f"(increment `+{_format_float(correlation_gap_report.correlation_repair_increment)}`, "
        f"just `{_format_percent(correlation_gap_report.required_gap_share)}` of the full "
        "anchor-to-companion gap) while the last local-scale miss still stays "
        f"`{_format_percent(center_coupling_report.coverage_anchor_local_scale_access_shortfall_share)}` "
        f"at `z = {_format_grid_value(correlation_gap_report.failing_right_shoulder_grid_value)}`",
        "- sign healing is outside the feasible repair lane: zeroing the center-conditioned "
        "direct residual would require right-center correlation "
        f"`{_format_float(feasibility_report.zero_partial_right_center_correlation)}`, "
        f"i.e. `{_format_float(feasibility_report.zero_partial_lower_bound_gap)}` below the "
        "feasible lower bound `-1.000` and "
        f"`{_format_float(feasibility_report.required_repair_distance_to_zero_partial_target)}` "
        "away from the bounded repair target "
        f"`{_format_float(correlation_gap_report.required_repaired_correlation)}`",
        "- numerator repair is the only realistic bridge: with center scale fixed, required "
        "`|covariance(z=0.25, z=0.15)|` is "
        f"`{_format_float(channel_report.required_abs_shoulder_center_covariance_at_fixed_center_sigma)}`, "
        "still just "
        f"`{_format_percent(channel_report.required_covariance_share_of_companion)}` of companion "
        f"covariance `{_format_float(channel_report.companion_abs_shoulder_center_covariance)}`; "
        "with covariance fixed instead, required shoulder `sigma_z_hat(z=0.25)` would collapse to "
        f"`{_format_float(required_shoulder_sigma)}` and required center `sigma_z_hat(z=0.15)` "
        f"to `{_format_float(channel_report.required_center_sigma_z_hat_at_fixed_covariance)}`, "
        "each only "
        f"`{_format_percent(required_shoulder_sigma_share_of_anchor)}` of current anchor scale",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; source-level follow-up should restore a bounded slice of seed "
        f"`{correlation_gap_report.coverage_anchor_random_state}` right-shoulder / center "
        "covariance entry access while treating negative direct residual as background "
        "and excluding denominator-compression or sign-healing side quests",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "correlation-repair-lane-snapshot"
        ),
        policy_digest=tuple(correlation_gap_report.policy_digest),
        binding_design=tuple(correlation_gap_report.binding_design),
        window_label=center_coupling_report.window_label,
        coverage_anchor_random_state=correlation_gap_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            correlation_gap_report.overshoot_companion_random_state
        ),
        center_grid_value=correlation_gap_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            correlation_gap_report.failing_right_shoulder_grid_value
        ),
        coverage_anchor_local_scale_access_shortfall_share=(
            center_coupling_report.coverage_anchor_local_scale_access_shortfall_share
        ),
        current_anchor_correlation=correlation_gap_report.current_anchor_correlation,
        required_repaired_correlation=(
            correlation_gap_report.required_repaired_correlation
        ),
        correlation_repair_increment=(
            correlation_gap_report.correlation_repair_increment
        ),
        required_gap_share=correlation_gap_report.required_gap_share,
        zero_partial_right_center_correlation=(
            feasibility_report.zero_partial_right_center_correlation
        ),
        zero_partial_lower_bound_gap=(feasibility_report.zero_partial_lower_bound_gap),
        required_repair_distance_to_zero_partial_target=(
            feasibility_report.required_repair_distance_to_zero_partial_target
        ),
        current_abs_shoulder_center_covariance=(
            channel_report.current_abs_shoulder_center_covariance
        ),
        required_abs_shoulder_center_covariance_at_fixed_center_sigma=(
            channel_report.required_abs_shoulder_center_covariance_at_fixed_center_sigma
        ),
        required_covariance_share_of_companion=(
            channel_report.required_covariance_share_of_companion
        ),
        current_anchor_shoulder_sigma_z_hat=(
            center_coupling_report.coverage_anchor_shoulder_sigma_z_hat
        ),
        required_shoulder_sigma_z_hat_at_fixed_covariance_and_center_sigma=(
            required_shoulder_sigma
        ),
        required_shoulder_sigma_share_of_anchor=(
            required_shoulder_sigma_share_of_anchor
        ),
        required_shoulder_sigma_share_of_companion=(
            required_shoulder_sigma_share_of_companion
        ),
        current_anchor_center_sigma_z_hat=(
            channel_report.coverage_anchor_center_sigma_z_hat
        ),
        required_center_sigma_z_hat_at_fixed_covariance=(
            channel_report.required_center_sigma_z_hat_at_fixed_covariance
        ),
        required_center_sigma_share_of_anchor=(
            channel_report.required_center_sigma_share_of_anchor
        ),
        required_center_sigma_share_of_companion=(
            channel_report.required_center_sigma_share_of_companion
        ),
        driver_signature=driver_signature,
        canonical_correlation_repair_lane_snapshot_digest=digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot_report(
        correlation_gap_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe()
        ),
        center_coupling_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe()
        ),
        channel_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe()
        ),
        feasibility_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_probe()
        ),
    )
