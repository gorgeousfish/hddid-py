from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryFactorReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_partial_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderPartialReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_partial_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _driver_signature(
    *,
    partial_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderPartialReport
    ),
    correlation_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport
    ),
    covariance_entry_factor_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryFactorReport
    ),
) -> str:
    if (
        partial_report.driver_signature
        == "center-conditioned-cross-shoulder-sign-fracture"
        and correlation_gap_report.driver_signature
        == "partial-correlation-gap-bridge-target"
        and covariance_entry_factor_report.driver_signature
        == "correlation-led-covariance-entry-repair-target"
    ):
        return "direct-residual-correlation-repair-target"
    return "mixed-direct-residual-repair-target"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    anchor_partial_cross_shoulder_correlation: float
    companion_partial_cross_shoulder_correlation: float
    anchor_center_mediated_share_of_abs_cross_shoulder: float
    companion_center_mediated_share_of_cross_shoulder: float
    required_repaired_correlation: float
    correlation_repair_increment: float
    required_gap_share: float
    remaining_gap_headroom_share: float
    correlation_log_gap_share: float
    shoulder_sigma_log_gap_share: float
    center_sigma_log_gap_share: float
    driver_signature: str
    canonical_direct_residual_digest: tuple[str, ...]

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
        self.left_shoulder_grid_value = float(self.left_shoulder_grid_value)
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.anchor_partial_cross_shoulder_correlation = float(
            self.anchor_partial_cross_shoulder_correlation
        )
        self.companion_partial_cross_shoulder_correlation = float(
            self.companion_partial_cross_shoulder_correlation
        )
        self.anchor_center_mediated_share_of_abs_cross_shoulder = float(
            self.anchor_center_mediated_share_of_abs_cross_shoulder
        )
        self.companion_center_mediated_share_of_cross_shoulder = float(
            self.companion_center_mediated_share_of_cross_shoulder
        )
        self.required_repaired_correlation = float(self.required_repaired_correlation)
        self.correlation_repair_increment = float(self.correlation_repair_increment)
        self.required_gap_share = float(self.required_gap_share)
        self.remaining_gap_headroom_share = float(self.remaining_gap_headroom_share)
        self.correlation_log_gap_share = float(self.correlation_log_gap_share)
        self.shoulder_sigma_log_gap_share = float(self.shoulder_sigma_log_gap_share)
        self.center_sigma_log_gap_share = float(self.center_sigma_log_gap_share)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_direct_residual_digest = tuple(
            str(line).rstrip() for line in self.canonical_direct_residual_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "left_shoulder_grid_value": self.left_shoulder_grid_value,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "anchor_partial_cross_shoulder_correlation": (
                self.anchor_partial_cross_shoulder_correlation
            ),
            "companion_partial_cross_shoulder_correlation": (
                self.companion_partial_cross_shoulder_correlation
            ),
            "anchor_center_mediated_share_of_abs_cross_shoulder": (
                self.anchor_center_mediated_share_of_abs_cross_shoulder
            ),
            "companion_center_mediated_share_of_cross_shoulder": (
                self.companion_center_mediated_share_of_cross_shoulder
            ),
            "required_repaired_correlation": self.required_repaired_correlation,
            "correlation_repair_increment": self.correlation_repair_increment,
            "required_gap_share": self.required_gap_share,
            "remaining_gap_headroom_share": self.remaining_gap_headroom_share,
            "correlation_log_gap_share": self.correlation_log_gap_share,
            "shoulder_sigma_log_gap_share": self.shoulder_sigma_log_gap_share,
            "center_sigma_log_gap_share": self.center_sigma_log_gap_share,
            "driver_signature": self.driver_signature,
            "canonical_direct_residual_digest": list(
                self.canonical_direct_residual_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_report(
    *,
    partial_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderPartialReport
    ),
    correlation_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport
    ),
    covariance_entry_factor_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryFactorReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualReport:
    if partial_report.policy_digest != correlation_gap_report.policy_digest:
        raise ValueError(
            "direct residual probe requires a single canonical policy digest"
        )
    if partial_report.policy_digest != covariance_entry_factor_report.policy_digest:
        raise ValueError(
            "direct residual probe requires a single canonical policy digest"
        )
    if partial_report.binding_design != correlation_gap_report.binding_design:
        raise ValueError("direct residual probe requires a single binding design")
    if partial_report.binding_design != covariance_entry_factor_report.binding_design:
        raise ValueError("direct residual probe requires a single binding design")
    if (
        partial_report.coverage_anchor_random_state
        != correlation_gap_report.coverage_anchor_random_state
        or partial_report.coverage_anchor_random_state
        != covariance_entry_factor_report.coverage_anchor_random_state
    ):
        raise ValueError("direct residual probe requires a single coverage-anchor seed")
    if (
        partial_report.overshoot_companion_random_state
        != correlation_gap_report.overshoot_companion_random_state
        or partial_report.overshoot_companion_random_state
        != covariance_entry_factor_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "direct residual probe requires a single overshoot companion seed"
        )
    if partial_report.center_grid_value != correlation_gap_report.center_grid_value:
        raise ValueError("direct residual probe requires a shared center grid value")
    if (
        partial_report.center_grid_value
        != covariance_entry_factor_report.center_grid_value
    ):
        raise ValueError("direct residual probe requires a shared center grid value")
    if (
        partial_report.failing_right_shoulder_grid_value
        != correlation_gap_report.failing_right_shoulder_grid_value
    ):
        raise ValueError(
            "direct residual probe requires a shared failing right-shoulder grid value"
        )
    if (
        partial_report.failing_right_shoulder_grid_value
        != covariance_entry_factor_report.failing_right_shoulder_grid_value
    ):
        raise ValueError(
            "direct residual probe requires a shared failing right-shoulder grid value"
        )

    driver_signature = _driver_signature(
        partial_report=partial_report,
        correlation_gap_report=correlation_gap_report,
        covariance_entry_factor_report=covariance_entry_factor_report,
    )
    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*partial_report.binding_design)}` on `near_zero_grid`: "
        f"conditioning on center `z = {_format_grid_value(partial_report.center_grid_value)}` "
        "still leaves coverage anchor seed "
        f"`{partial_report.coverage_anchor_random_state}` with partial cross-shoulder "
        f"correlation `{_format_float(partial_report.anchor_center_conditioned_cross_shoulder_partial_correlation)}`, "
        "versus "
        f"`{_format_signed_float(partial_report.companion_center_conditioned_cross_shoulder_partial_correlation)}` "
        f"for overshoot companion seed `{partial_report.overshoot_companion_random_state}`; "
        "center mediation explains only "
        f"`{_format_percent(partial_report.anchor_center_mediated_share_of_abs_cross_shoulder)}` "
        "of the anchor's absolute cross-shoulder magnitude but "
        f"`{_format_percent(partial_report.companion_center_mediated_share_of_cross_shoulder)}` "
        "of the companion's positive level",
        "- closing the last Trigger 2 miss still needs only a bounded shoulder-center "
        "correlation bridge: repaired correlation rises to "
        f"`{_format_float(correlation_gap_report.required_repaired_correlation)}`, "
        "so the required increment stays at "
        f"`{_format_signed_float(correlation_gap_report.correlation_repair_increment)}`, "
        "which consumes just "
        f"`{_format_percent(correlation_gap_report.required_gap_share)}` "
        "of the full anchor-to-companion gap and leaves "
        f"`{_format_percent(correlation_gap_report.remaining_gap_headroom_share)}` "
        "unused headroom",
        "- raw covariance-entry suppression remains correlation-led even after that "
        "bridge view: shoulder-center correlation still contributes "
        f"`{_format_percent(covariance_entry_factor_report.correlation_log_gap_share)}` "
        "of the log-gap, versus "
        f"`{_format_percent(covariance_entry_factor_report.shoulder_sigma_log_gap_share)}` "
        "from shoulder `sigma_z_hat` and "
        f"`{_format_percent(covariance_entry_factor_report.center_sigma_log_gap_share)}` "
        "from center `sigma_z_hat`",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; source-level follow-up should explain why seed "
        f"`{partial_report.coverage_anchor_random_state}` retains a negative direct left-right "
        "residual while recovering only a bounded slice of shoulder-center correlation access, "
        "not generic center smoothing or full companion replay",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-direct-residual-probe",
        policy_digest=partial_report.policy_digest,
        binding_design=partial_report.binding_design,
        coverage_anchor_random_state=partial_report.coverage_anchor_random_state,
        overshoot_companion_random_state=partial_report.overshoot_companion_random_state,
        left_shoulder_grid_value=partial_report.left_shoulder_grid_value,
        center_grid_value=partial_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            partial_report.failing_right_shoulder_grid_value
        ),
        anchor_partial_cross_shoulder_correlation=(
            partial_report.anchor_center_conditioned_cross_shoulder_partial_correlation
        ),
        companion_partial_cross_shoulder_correlation=(
            partial_report.companion_center_conditioned_cross_shoulder_partial_correlation
        ),
        anchor_center_mediated_share_of_abs_cross_shoulder=(
            partial_report.anchor_center_mediated_share_of_abs_cross_shoulder
        ),
        companion_center_mediated_share_of_cross_shoulder=(
            partial_report.companion_center_mediated_share_of_cross_shoulder
        ),
        required_repaired_correlation=(
            correlation_gap_report.required_repaired_correlation
        ),
        correlation_repair_increment=(
            correlation_gap_report.correlation_repair_increment
        ),
        required_gap_share=correlation_gap_report.required_gap_share,
        remaining_gap_headroom_share=(
            correlation_gap_report.remaining_gap_headroom_share
        ),
        correlation_log_gap_share=(
            covariance_entry_factor_report.correlation_log_gap_share
        ),
        shoulder_sigma_log_gap_share=(
            covariance_entry_factor_report.shoulder_sigma_log_gap_share
        ),
        center_sigma_log_gap_share=(
            covariance_entry_factor_report.center_sigma_log_gap_share
        ),
        driver_signature=driver_signature,
        canonical_direct_residual_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_report(
        partial_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_partial_probe()
        ),
        correlation_gap_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe()
        ),
        covariance_entry_factor_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe()
        ),
    )
