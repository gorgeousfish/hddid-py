from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose, log

from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _positive_log_share(
    component_ratio: float,
    total_ratio: float,
    *,
    label: str,
) -> float:
    component_value = float(component_ratio)
    total_value = float(total_ratio)
    if component_value <= 1.0:
        raise ValueError(f"{label} component ratio must exceed 1.0")
    if total_value <= 1.0:
        raise ValueError(f"{label} total ratio must exceed 1.0")
    return float(log(component_value) / log(total_value))


def _driver_signature(
    *,
    center_coupling_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport,
    center_coupling_channel_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport
    ),
    shoulder_sigma_log_gap_share: float,
    center_sigma_log_gap_share: float,
    correlation_log_gap_share: float,
) -> str:
    if (
        center_coupling_report.driver_signature
        == "right-shoulder-center-coupling-access-bottleneck"
        and center_coupling_channel_report.driver_signature
        == "covariance-entry-first-repair-target"
        and correlation_log_gap_share > shoulder_sigma_log_gap_share
        and correlation_log_gap_share > center_sigma_log_gap_share
    ):
        return "correlation-led-covariance-entry-repair-target"
    if shoulder_sigma_log_gap_share >= max(
        center_sigma_log_gap_share, correlation_log_gap_share
    ):
        return "shoulder-scale-led-covariance-entry-repair-target"
    if center_sigma_log_gap_share >= max(
        shoulder_sigma_log_gap_share, correlation_log_gap_share
    ):
        return "center-scale-led-covariance-entry-repair-target"
    return "mixed-covariance-entry-repair-target"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryFactorReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    shoulder_sigma_z_hat_ratio: float
    center_sigma_z_hat_ratio: float
    shoulder_center_correlation_ratio: float
    covariance_entry_ratio: float
    reconstructed_covariance_entry_ratio: float
    shoulder_sigma_log_gap_share: float
    center_sigma_log_gap_share: float
    correlation_log_gap_share: float
    correlation_vs_shoulder_sigma_ratio: float
    correlation_vs_center_sigma_ratio: float
    driver_signature: str
    canonical_covariance_entry_factor_digest: tuple[str, ...]

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
        self.shoulder_sigma_z_hat_ratio = float(self.shoulder_sigma_z_hat_ratio)
        self.center_sigma_z_hat_ratio = float(self.center_sigma_z_hat_ratio)
        self.shoulder_center_correlation_ratio = float(
            self.shoulder_center_correlation_ratio
        )
        self.covariance_entry_ratio = float(self.covariance_entry_ratio)
        self.reconstructed_covariance_entry_ratio = float(
            self.reconstructed_covariance_entry_ratio
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
        self.canonical_covariance_entry_factor_digest = tuple(
            str(line).rstrip() for line in self.canonical_covariance_entry_factor_digest
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
            "shoulder_sigma_z_hat_ratio": self.shoulder_sigma_z_hat_ratio,
            "center_sigma_z_hat_ratio": self.center_sigma_z_hat_ratio,
            "shoulder_center_correlation_ratio": (
                self.shoulder_center_correlation_ratio
            ),
            "covariance_entry_ratio": self.covariance_entry_ratio,
            "reconstructed_covariance_entry_ratio": (
                self.reconstructed_covariance_entry_ratio
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
            "canonical_covariance_entry_factor_digest": list(
                self.canonical_covariance_entry_factor_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_report(
    *,
    center_coupling_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport,
    center_coupling_channel_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryFactorReport:
    if (
        center_coupling_report.policy_digest
        != center_coupling_channel_report.policy_digest
    ):
        raise ValueError(
            "covariance entry factor probe requires a single policy digest"
        )
    if (
        center_coupling_report.binding_design
        != center_coupling_channel_report.binding_design
    ):
        raise ValueError(
            "covariance entry factor probe requires a single binding design"
        )
    if (
        center_coupling_report.coverage_anchor_random_state
        != center_coupling_channel_report.coverage_anchor_random_state
        or center_coupling_report.overshoot_companion_random_state
        != center_coupling_channel_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "covariance entry factor probe requires the same anchor/companion seeds"
        )
    if not isclose(
        center_coupling_report.center_grid_value,
        center_coupling_channel_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError("covariance entry factor probe requires the same center grid")
    if not isclose(
        center_coupling_report.failing_right_shoulder_grid_value,
        center_coupling_channel_report.failing_right_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "covariance entry factor probe requires the same failing shoulder grid"
        )

    reconstructed_covariance_entry_ratio = (
        center_coupling_report.shoulder_sigma_z_hat_ratio
        * center_coupling_channel_report.center_sigma_z_hat_ratio
        * center_coupling_report.shoulder_center_correlation_ratio
    )
    if not isclose(
        reconstructed_covariance_entry_ratio,
        center_coupling_channel_report.covariance_entry_ratio,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "covariance entry factor probe requires exact factorization of the covariance-entry gap"
        )

    shoulder_sigma_log_gap_share = _positive_log_share(
        center_coupling_report.shoulder_sigma_z_hat_ratio,
        center_coupling_channel_report.covariance_entry_ratio,
        label="shoulder_sigma_log_gap_share",
    )
    center_sigma_log_gap_share = _positive_log_share(
        center_coupling_channel_report.center_sigma_z_hat_ratio,
        center_coupling_channel_report.covariance_entry_ratio,
        label="center_sigma_log_gap_share",
    )
    correlation_log_gap_share = _positive_log_share(
        center_coupling_report.shoulder_center_correlation_ratio,
        center_coupling_channel_report.covariance_entry_ratio,
        label="correlation_log_gap_share",
    )
    correlation_vs_shoulder_sigma_ratio = (
        center_coupling_report.shoulder_center_correlation_ratio
        / center_coupling_report.shoulder_sigma_z_hat_ratio
    )
    correlation_vs_center_sigma_ratio = (
        center_coupling_report.shoulder_center_correlation_ratio
        / center_coupling_channel_report.center_sigma_z_hat_ratio
    )
    driver_signature = _driver_signature(
        center_coupling_report=center_coupling_report,
        center_coupling_channel_report=center_coupling_channel_report,
        shoulder_sigma_log_gap_share=shoulder_sigma_log_gap_share,
        center_sigma_log_gap_share=center_sigma_log_gap_share,
        correlation_log_gap_share=correlation_log_gap_share,
    )
    canonical_digest = (
        "- raw shoulder-center covariance entry still factorizes exactly as "
        "`sigma_shoulder * sigma_center * correlation`: the companion-to-anchor gap is "
        f"`{_format_ratio(center_coupling_channel_report.covariance_entry_ratio)}`, with "
        f"`{_format_ratio(center_coupling_report.shoulder_sigma_z_hat_ratio)}` shoulder sigma, "
        f"`{_format_ratio(center_coupling_channel_report.center_sigma_z_hat_ratio)}` center sigma, "
        f"and `{_format_ratio(center_coupling_report.shoulder_center_correlation_ratio)}` correlation "
        f"at `z = {center_coupling_report.failing_right_shoulder_grid_value:.2f} -> "
        f"{center_coupling_report.center_grid_value:.2f}`",
        "- on the log-gap scale, shoulder `sigma_z_hat` contributes "
        f"`{_format_percent(shoulder_sigma_log_gap_share)}`, center `sigma_z_hat` contributes "
        f"`{_format_percent(center_sigma_log_gap_share)}`, and shoulder-center correlation "
        f"contributes the dominant `{_format_percent(correlation_log_gap_share)}` "
        "of the covariance-entry suppression",
        "- this keeps the current Trigger 2 channel numerator-led twice over: even inside "
        "the raw covariance entry, correlation amplification outpaces shoulder-scale widening "
        f"by `{_format_ratio(correlation_vs_shoulder_sigma_ratio)}` and center-scale widening "
        f"by `{_format_ratio(correlation_vs_center_sigma_ratio)}`",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; source-level follow-up should restore seed "
        f"`{center_coupling_report.coverage_anchor_random_state}` shoulder-center correlation "
        "access before broad shoulder or center scale retuning",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryFactorReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-covariance-entry-factor-probe",
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
        shoulder_sigma_z_hat_ratio=center_coupling_report.shoulder_sigma_z_hat_ratio,
        center_sigma_z_hat_ratio=(
            center_coupling_channel_report.center_sigma_z_hat_ratio
        ),
        shoulder_center_correlation_ratio=(
            center_coupling_report.shoulder_center_correlation_ratio
        ),
        covariance_entry_ratio=center_coupling_channel_report.covariance_entry_ratio,
        reconstructed_covariance_entry_ratio=reconstructed_covariance_entry_ratio,
        shoulder_sigma_log_gap_share=shoulder_sigma_log_gap_share,
        center_sigma_log_gap_share=center_sigma_log_gap_share,
        correlation_log_gap_share=correlation_log_gap_share,
        correlation_vs_shoulder_sigma_ratio=correlation_vs_shoulder_sigma_ratio,
        correlation_vs_center_sigma_ratio=correlation_vs_center_sigma_ratio,
        driver_signature=driver_signature,
        canonical_covariance_entry_factor_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryFactorReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_report(
        center_coupling_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe(),
        center_coupling_channel_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe(),
    )
