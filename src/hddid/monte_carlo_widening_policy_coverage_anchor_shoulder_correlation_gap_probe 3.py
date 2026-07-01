from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _driver_signature(
    *,
    repair_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport
    ),
    required_gap_share: float,
    remaining_gap_headroom_share: float,
) -> str:
    if (
        repair_report.driver_signature == "coupling-first-repair-target"
        and 0.0 < required_gap_share < 0.5
        and remaining_gap_headroom_share > required_gap_share
    ):
        return "partial-correlation-gap-bridge-target"
    if required_gap_share >= 1.0:
        return "full-companion-correlation-replay-needed"
    return "mixed-correlation-gap-target"


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
    remaining_gap_headroom: float
    remaining_gap_headroom_share: float
    remaining_headroom_multiple_vs_required_increment: float
    driver_signature: str
    canonical_correlation_gap_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
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
        self.remaining_gap_headroom = float(self.remaining_gap_headroom)
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
            "full_anchor_to_companion_correlation_gap": (
                self.full_anchor_to_companion_correlation_gap
            ),
            "required_gap_share": self.required_gap_share,
            "remaining_gap_headroom": self.remaining_gap_headroom,
            "remaining_gap_headroom_share": self.remaining_gap_headroom_share,
            "remaining_headroom_multiple_vs_required_increment": (
                self.remaining_headroom_multiple_vs_required_increment
            ),
            "driver_signature": self.driver_signature,
            "canonical_correlation_gap_digest": list(
                self.canonical_correlation_gap_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_report(
    *,
    center_coupling_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport
    ),
    repair_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport:
    if center_coupling_report.policy_digest != repair_report.policy_digest:
        raise ValueError("correlation gap probe requires a single policy digest")
    if center_coupling_report.binding_design != repair_report.binding_design:
        raise ValueError("correlation gap probe requires a single binding design")
    if (
        center_coupling_report.coverage_anchor_random_state
        != repair_report.coverage_anchor_random_state
        or center_coupling_report.overshoot_companion_random_state
        != repair_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "correlation gap probe requires the same anchor/companion seeds"
        )
    if center_coupling_report.center_grid_value != repair_report.center_grid_value:
        raise ValueError("correlation gap probe requires the same center grid value")
    if (
        center_coupling_report.failing_right_shoulder_grid_value
        != repair_report.failing_right_shoulder_grid_value
    ):
        raise ValueError(
            "correlation gap probe requires the same failing shoulder grid value"
        )

    current_anchor_correlation = float(
        center_coupling_report.coverage_anchor_shoulder_center_correlation
    )
    required_repaired_correlation = float(
        repair_report.fixed_sigma_required_correlation
    )
    companion_correlation = float(
        center_coupling_report.overshoot_companion_shoulder_center_correlation
    )
    if current_anchor_correlation <= 0.0:
        raise ValueError("current anchor correlation must be positive")
    if companion_correlation <= current_anchor_correlation:
        raise ValueError("companion correlation must exceed the current anchor value")
    if required_repaired_correlation < current_anchor_correlation:
        raise ValueError("required repaired correlation cannot fall below current")
    if required_repaired_correlation > companion_correlation:
        raise ValueError("required repaired correlation cannot exceed companion")

    correlation_repair_increment = float(
        required_repaired_correlation - current_anchor_correlation
    )
    full_anchor_to_companion_correlation_gap = float(
        companion_correlation - current_anchor_correlation
    )
    remaining_gap_headroom = float(
        companion_correlation - required_repaired_correlation
    )
    required_gap_share = _positive_ratio(
        correlation_repair_increment,
        full_anchor_to_companion_correlation_gap,
        label="required_gap_share",
    )
    remaining_gap_headroom_share = _positive_ratio(
        remaining_gap_headroom,
        full_anchor_to_companion_correlation_gap,
        label="remaining_gap_headroom_share",
    )
    remaining_headroom_multiple_vs_required_increment = _positive_ratio(
        remaining_gap_headroom,
        correlation_repair_increment,
        label="remaining_headroom_multiple_vs_required_increment",
    )
    driver_signature = _driver_signature(
        repair_report=repair_report,
        required_gap_share=required_gap_share,
        remaining_gap_headroom_share=remaining_gap_headroom_share,
    )

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*center_coupling_report.binding_design)}`: at the failing "
        f"shoulder `z = {_format_grid_value(center_coupling_report.failing_right_shoulder_grid_value)}` "
        f"versus center `z = {_format_grid_value(center_coupling_report.center_grid_value)}`, "
        f"coverage anchor seed `{center_coupling_report.coverage_anchor_random_state}` keeps "
        f"shoulder-center correlation only at `{_format_float(current_anchor_correlation)}`, "
        f"while companion seed `{center_coupling_report.overshoot_companion_random_state}` "
        f"sits at `{_format_float(companion_correlation)}`",
        "- closing the last Trigger 2 miss does not require replaying the whole companion "
        f"state: the fixed-scale repaired correlation only needs to rise to "
        f"`{_format_float(required_repaired_correlation)}`, so the incremental repair is just "
        f"`{_format_signed_float(correlation_repair_increment)}`",
        "- that repair increment consumes only "
        f"`{_format_percent(required_gap_share)}` of the full anchor-to-companion correlation "
        f"gap `{_format_signed_float(full_anchor_to_companion_correlation_gap)}`, leaving "
        f"`{_format_percent(remaining_gap_headroom_share)}` of the gap unused; remaining "
        f"headroom is still `{_format_ratio(remaining_headroom_multiple_vs_required_increment)}` "
        "the required increment",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; implementation follow-up should recover a bounded slice of "
        f"right-shoulder / center correlation access for seed "
        f"`{center_coupling_report.coverage_anchor_random_state}`, not force companion-level "
        "covariance geometry",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-correlation-gap-probe"
        ),
        policy_digest=center_coupling_report.policy_digest,
        binding_design=center_coupling_report.binding_design,
        coverage_anchor_random_state=center_coupling_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            center_coupling_report.overshoot_companion_random_state
        ),
        center_grid_value=center_coupling_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            center_coupling_report.failing_right_shoulder_grid_value
        ),
        current_anchor_correlation=current_anchor_correlation,
        required_repaired_correlation=required_repaired_correlation,
        companion_correlation=companion_correlation,
        correlation_repair_increment=correlation_repair_increment,
        full_anchor_to_companion_correlation_gap=(
            full_anchor_to_companion_correlation_gap
        ),
        required_gap_share=required_gap_share,
        remaining_gap_headroom=remaining_gap_headroom,
        remaining_gap_headroom_share=remaining_gap_headroom_share,
        remaining_headroom_multiple_vs_required_increment=(
            remaining_headroom_multiple_vs_required_increment
        ),
        driver_signature=driver_signature,
        canonical_correlation_gap_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_report(
        center_coupling_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe()
        ),
        repair_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe()
        ),
    )
