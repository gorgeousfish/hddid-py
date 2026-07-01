from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _contains_grid_value(evaluation_grid: tuple[float, ...], target: float) -> bool:
    target_value = float(target)
    return any(
        isclose(float(value), target_value, abs_tol=1e-12) for value in evaluation_grid
    )


def _driver_signature(
    *,
    access_share_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    covariance_amplification_dominance_factor: float,
) -> str:
    if (
        access_share_report.driver_signature
        == "single-share-local-scale-access-shortfall"
        and seed_window_covariance_report.window_covariance_driver
        == "covariance-coupling-and-scale-overshoot"
        and covariance_amplification_dominance_factor > 1.0
    ):
        return "covariance-process-sigma-z-hat-access-bottleneck"
    if covariance_amplification_dominance_factor <= 1.0:
        return "uniform-critical-value-access-bottleneck"
    return "mixed-local-scale-access-bottleneck"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessDriverReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    failing_right_shoulder_grid_value: float
    window_label: str
    interval_scale_contract: float
    local_scale_access_shortfall_share: float
    local_scale_reserve_share: float
    covariance_trace_ratio: float
    uniform_critical_value_ratio: float
    covariance_amplification_dominance_factor: float
    off_center_correlation_gap: float
    leading_positive_share_gap: float
    window_covariance_driver: str
    driver_signature: str
    canonical_coverage_anchor_shoulder_access_driver_digest: tuple[str, ...]

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
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.window_label = str(self.window_label).strip()
        self.interval_scale_contract = float(self.interval_scale_contract)
        self.local_scale_access_shortfall_share = float(
            self.local_scale_access_shortfall_share
        )
        self.local_scale_reserve_share = float(self.local_scale_reserve_share)
        self.covariance_trace_ratio = float(self.covariance_trace_ratio)
        self.uniform_critical_value_ratio = float(self.uniform_critical_value_ratio)
        self.covariance_amplification_dominance_factor = float(
            self.covariance_amplification_dominance_factor
        )
        self.off_center_correlation_gap = float(self.off_center_correlation_gap)
        self.leading_positive_share_gap = float(self.leading_positive_share_gap)
        self.window_covariance_driver = str(self.window_covariance_driver).strip()
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_coverage_anchor_shoulder_access_driver_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_access_driver_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "window_label": self.window_label,
            "interval_scale_contract": self.interval_scale_contract,
            "local_scale_access_shortfall_share": self.local_scale_access_shortfall_share,
            "local_scale_reserve_share": self.local_scale_reserve_share,
            "covariance_trace_ratio": self.covariance_trace_ratio,
            "uniform_critical_value_ratio": self.uniform_critical_value_ratio,
            "covariance_amplification_dominance_factor": (
                self.covariance_amplification_dominance_factor
            ),
            "off_center_correlation_gap": self.off_center_correlation_gap,
            "leading_positive_share_gap": self.leading_positive_share_gap,
            "window_covariance_driver": self.window_covariance_driver,
            "driver_signature": self.driver_signature,
            "canonical_coverage_anchor_shoulder_access_driver_digest": list(
                self.canonical_coverage_anchor_shoulder_access_driver_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_driver_report(
    *,
    access_share_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessDriverReport:
    if access_share_report.policy_digest != seed_window_covariance_report.policy_digest:
        raise ValueError(
            "access driver probe requires a single Trigger 2 policy digest"
        )
    if (
        access_share_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError("access driver probe requires a single binding design")
    if (
        access_share_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or access_share_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError("access driver probe requires the same anchor/companion seeds")
    if not _contains_grid_value(
        seed_window_covariance_report.evaluation_grid,
        access_share_report.failing_right_shoulder_grid_value,
    ):
        raise ValueError(
            "access driver probe requires the failing shoulder to stay in view"
        )
    if seed_window_covariance_report.uniform_critical_value_ratio <= 0.0:
        raise ValueError("access driver probe requires a positive critical-value ratio")

    covariance_amplification_dominance_factor = (
        seed_window_covariance_report.covariance_trace_ratio
        / seed_window_covariance_report.uniform_critical_value_ratio
    )
    driver_signature = _driver_signature(
        access_share_report=access_share_report,
        seed_window_covariance_report=seed_window_covariance_report,
        covariance_amplification_dominance_factor=(
            covariance_amplification_dominance_factor
        ),
    )
    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*access_share_report.binding_design)}`: at the failing shoulder "
        f"`z = {_format_grid_value(access_share_report.failing_right_shoulder_grid_value)}`, "
        f"coverage anchor seed `{access_share_report.coverage_anchor_random_state}` still misses only "
        f"`{_format_percent(access_share_report.local_scale_access_shortfall_share)}` local scale access "
        f"while overshoot companion seed `{access_share_report.overshoot_companion_random_state}` retains "
        f"`{_format_percent(access_share_report.local_scale_reserve_share)}` same-point reserve "
        f"under interval-scale contract `{_format_float(access_share_report.interval_scale_contract)}`",
        "- on the same "
        f"`{seed_window_covariance_report.window_label}`, the anchor/companion window split stays "
        f"`covariance trace ratio = {_format_ratio(seed_window_covariance_report.covariance_trace_ratio)}`, "
        f"`critical ratio = {_format_ratio(seed_window_covariance_report.uniform_critical_value_ratio)}`, "
        f"`center-coupling gap = {_format_signed(seed_window_covariance_report.off_center_correlation_gap)}`, "
        f"`leading-share gap = {_format_signed(seed_window_covariance_report.leading_positive_share_gap)}`",
        "- covariance amplification therefore dominates critical drift by "
        f"`{_format_ratio(covariance_amplification_dominance_factor)}`, so the last same-point "
        f"`{_format_percent(access_share_report.local_scale_access_shortfall_share)}` shortfall "
        "cannot be explained by uniform critical tuning alone",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; inspect local covariance / `sigma_z_hat` object flow "
        f"for seed `{access_share_report.coverage_anchor_random_state}` at "
        f"`z = {_format_grid_value(access_share_report.failing_right_shoulder_grid_value)}`, "
        "not the global interval rule",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessDriverReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-access-driver-probe",
        policy_digest=access_share_report.policy_digest,
        binding_design=access_share_report.binding_design,
        coverage_anchor_random_state=access_share_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            access_share_report.overshoot_companion_random_state
        ),
        failing_right_shoulder_grid_value=(
            access_share_report.failing_right_shoulder_grid_value
        ),
        window_label=seed_window_covariance_report.window_label,
        interval_scale_contract=access_share_report.interval_scale_contract,
        local_scale_access_shortfall_share=(
            access_share_report.local_scale_access_shortfall_share
        ),
        local_scale_reserve_share=access_share_report.local_scale_reserve_share,
        covariance_trace_ratio=seed_window_covariance_report.covariance_trace_ratio,
        uniform_critical_value_ratio=(
            seed_window_covariance_report.uniform_critical_value_ratio
        ),
        covariance_amplification_dominance_factor=(
            covariance_amplification_dominance_factor
        ),
        off_center_correlation_gap=seed_window_covariance_report.off_center_correlation_gap,
        leading_positive_share_gap=(
            seed_window_covariance_report.leading_positive_share_gap
        ),
        window_covariance_driver=seed_window_covariance_report.window_covariance_driver,
        driver_signature=driver_signature,
        canonical_coverage_anchor_shoulder_access_driver_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_driver_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessDriverReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_driver_report(
        access_share_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe(),
        seed_window_covariance_report=run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe(),
    )
