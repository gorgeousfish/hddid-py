from __future__ import annotations

from dataclasses import dataclass

from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe,
)


def _format_design_key(design: tuple[str, int, int]) -> str:
    return f"{design[0]}/{design[1]}/{design[2]}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    current_anchor_correlation: float
    required_repaired_correlation: float
    companion_correlation: float
    correlation_repair_increment: float
    full_anchor_to_companion_correlation_gap: float
    required_gap_share: float
    remaining_gap_headroom_share: float
    remaining_headroom_multiple_vs_required_increment: float
    driver_signature: str
    canonical_correlation_gap_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        dgp, n_obs, p = self.binding_design
        self.binding_design = (str(dgp).strip(), int(n_obs), int(p))
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.current_anchor_correlation = float(self.current_anchor_correlation)
        self.required_repaired_correlation = float(self.required_repaired_correlation)
        self.companion_correlation = float(self.companion_correlation)
        self.correlation_repair_increment = float(self.correlation_repair_increment)
        self.full_anchor_to_companion_correlation_gap = float(
            self.full_anchor_to_companion_correlation_gap
        )
        self.required_gap_share = float(self.required_gap_share)
        self.remaining_gap_headroom_share = float(self.remaining_gap_headroom_share)
        self.remaining_headroom_multiple_vs_required_increment = float(
            self.remaining_headroom_multiple_vs_required_increment
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_correlation_gap_digest = tuple(
            str(line).rstrip() for line in self.canonical_correlation_gap_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "current_anchor_correlation": self.current_anchor_correlation,
            "required_repaired_correlation": self.required_repaired_correlation,
            "companion_correlation": self.companion_correlation,
            "correlation_repair_increment": self.correlation_repair_increment,
            "full_anchor_to_companion_correlation_gap": self.full_anchor_to_companion_correlation_gap,
            "required_gap_share": self.required_gap_share,
            "remaining_gap_headroom_share": self.remaining_gap_headroom_share,
            "remaining_headroom_multiple_vs_required_increment": self.remaining_headroom_multiple_vs_required_increment,
            "driver_signature": self.driver_signature,
            "canonical_correlation_gap_digest": list(
                self.canonical_correlation_gap_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_report(
    *,
    center_coupling_report,
    repair_report,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport:
    if center_coupling_report.policy_digest != repair_report.policy_digest:
        raise ValueError("correlation gap probe requires a shared policy digest")
    if center_coupling_report.binding_design != repair_report.binding_design:
        raise ValueError("correlation gap probe requires a shared binding design")
    current = float(center_coupling_report.coverage_anchor_shoulder_center_correlation)
    required = float(repair_report.fixed_sigma_required_correlation)
    companion = float(center_coupling_report.overshoot_companion_shoulder_center_correlation)
    increment = required - current
    full_gap = companion - current
    if full_gap <= 0.0 or increment <= 0.0:
        raise ValueError("correlation gap probe requires positive repair headroom")
    required_gap_share = increment / full_gap
    remaining_share = 1.0 - required_gap_share
    remaining_multiple = (companion - required) / increment
    design_label = _format_design_key(center_coupling_report.binding_design)
    digest = (
        f"- binding design `{design_label}`: at the failing shoulder `z = {center_coupling_report.failing_right_shoulder_grid_value:.2f}` versus center `z = {center_coupling_report.center_grid_value:.2f}`, coverage anchor seed `{center_coupling_report.coverage_anchor_random_state}` keeps shoulder-center correlation only at `{current:.3f}`, while companion seed `{center_coupling_report.overshoot_companion_random_state}` sits at `{companion:.3f}`",
        "- closing the last Trigger 2 miss does not require replaying the whole companion state: "
        f"the fixed-scale repaired correlation only needs to rise to `{required:.3f}`, so the incremental repair is just `+{increment:.3f}`",
        "- that repair increment consumes only "
        f"`{100.0 * required_gap_share:.1f}%` of the full anchor-to-companion correlation gap `+{full_gap:.3f}`, leaving `{100.0 * remaining_share:.1f}%` of the gap unused; remaining headroom is still `x{remaining_multiple:.3f}` the required increment",
        "- current Trigger 2 implication: `partial-correlation-gap-bridge-target`; implementation follow-up should recover a bounded slice of right-shoulder / center correlation access for seed `202`, not force companion-level covariance geometry",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-correlation-gap-probe",
        policy_digest=center_coupling_report.policy_digest,
        binding_design=center_coupling_report.binding_design,
        coverage_anchor_random_state=center_coupling_report.coverage_anchor_random_state,
        overshoot_companion_random_state=center_coupling_report.overshoot_companion_random_state,
        center_grid_value=center_coupling_report.center_grid_value,
        failing_right_shoulder_grid_value=center_coupling_report.failing_right_shoulder_grid_value,
        current_anchor_correlation=current,
        required_repaired_correlation=required,
        companion_correlation=companion,
        correlation_repair_increment=increment,
        full_anchor_to_companion_correlation_gap=full_gap,
        required_gap_share=required_gap_share,
        remaining_gap_headroom_share=remaining_share,
        remaining_headroom_multiple_vs_required_increment=remaining_multiple,
        driver_signature="partial-correlation-gap-bridge-target",
        canonical_correlation_gap_digest=digest,
    )


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_report(
        center_coupling_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe(),
        repair_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe(),
    )
