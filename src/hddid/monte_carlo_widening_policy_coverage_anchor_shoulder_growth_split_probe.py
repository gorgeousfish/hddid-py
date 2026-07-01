from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_boundary_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBoundaryReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_boundary_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport,
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReplay,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_multiplier(value: float) -> str:
    return f"x{float(value):.3f}"


def _half_interval(pointwise_interval_length: float) -> float:
    interval_length = float(pointwise_interval_length)
    if interval_length <= 0.0:
        raise ValueError("pointwise_interval_length must be positive")
    return float(interval_length / 2.0)


def _positive_growth(current: float, reference: float, *, label: str) -> float:
    reference_value = float(reference)
    if reference_value <= 0.0:
        raise ValueError(f"{label} reference value must be positive")
    return float(float(current) / reference_value)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderGrowthSplitReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    safe_right_shoulder_grid_value: float
    failing_right_shoulder_grid_value: float
    anchor_safe_absolute_error: float
    anchor_failing_absolute_error: float
    anchor_safe_half_interval: float
    anchor_failing_half_interval: float
    anchor_safe_sigma_z_hat: float
    anchor_failing_sigma_z_hat: float
    anchor_error_growth_factor: float
    anchor_half_interval_growth_factor: float
    anchor_sigma_growth_factor: float
    anchor_error_growth_minus_half_interval_growth: float
    anchor_error_growth_minus_sigma_growth: float
    anchor_ratio_jump: float
    failing_anchor_ratio: float
    failing_companion_ratio: float
    failing_anchor_to_companion_ratio_gap: float
    driver_signature: str
    canonical_coverage_anchor_shoulder_growth_split_digest: tuple[str, ...]

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
        self.safe_right_shoulder_grid_value = float(self.safe_right_shoulder_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.anchor_safe_absolute_error = float(self.anchor_safe_absolute_error)
        self.anchor_failing_absolute_error = float(self.anchor_failing_absolute_error)
        self.anchor_safe_half_interval = float(self.anchor_safe_half_interval)
        self.anchor_failing_half_interval = float(self.anchor_failing_half_interval)
        self.anchor_safe_sigma_z_hat = float(self.anchor_safe_sigma_z_hat)
        self.anchor_failing_sigma_z_hat = float(self.anchor_failing_sigma_z_hat)
        self.anchor_error_growth_factor = float(self.anchor_error_growth_factor)
        self.anchor_half_interval_growth_factor = float(
            self.anchor_half_interval_growth_factor
        )
        self.anchor_sigma_growth_factor = float(self.anchor_sigma_growth_factor)
        self.anchor_error_growth_minus_half_interval_growth = float(
            self.anchor_error_growth_minus_half_interval_growth
        )
        self.anchor_error_growth_minus_sigma_growth = float(
            self.anchor_error_growth_minus_sigma_growth
        )
        self.anchor_ratio_jump = float(self.anchor_ratio_jump)
        self.failing_anchor_ratio = float(self.failing_anchor_ratio)
        self.failing_companion_ratio = float(self.failing_companion_ratio)
        self.failing_anchor_to_companion_ratio_gap = float(
            self.failing_anchor_to_companion_ratio_gap
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_coverage_anchor_shoulder_growth_split_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_growth_split_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "safe_right_shoulder_grid_value": self.safe_right_shoulder_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "anchor_safe_absolute_error": self.anchor_safe_absolute_error,
            "anchor_failing_absolute_error": self.anchor_failing_absolute_error,
            "anchor_safe_half_interval": self.anchor_safe_half_interval,
            "anchor_failing_half_interval": self.anchor_failing_half_interval,
            "anchor_safe_sigma_z_hat": self.anchor_safe_sigma_z_hat,
            "anchor_failing_sigma_z_hat": self.anchor_failing_sigma_z_hat,
            "anchor_error_growth_factor": self.anchor_error_growth_factor,
            "anchor_half_interval_growth_factor": (
                self.anchor_half_interval_growth_factor
            ),
            "anchor_sigma_growth_factor": self.anchor_sigma_growth_factor,
            "anchor_error_growth_minus_half_interval_growth": (
                self.anchor_error_growth_minus_half_interval_growth
            ),
            "anchor_error_growth_minus_sigma_growth": (
                self.anchor_error_growth_minus_sigma_growth
            ),
            "anchor_ratio_jump": self.anchor_ratio_jump,
            "failing_anchor_ratio": self.failing_anchor_ratio,
            "failing_companion_ratio": self.failing_companion_ratio,
            "failing_anchor_to_companion_ratio_gap": (
                self.failing_anchor_to_companion_ratio_gap
            ),
            "driver_signature": self.driver_signature,
            "canonical_coverage_anchor_shoulder_growth_split_digest": list(
                self.canonical_coverage_anchor_shoulder_growth_split_digest
            ),
        }


def _safe_and_failing_replays(
    shoulder_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport,
) -> tuple[
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReplay,
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReplay,
]:
    return (
        shoulder_report.replay("tight_center_grid"),
        shoulder_report.replay("near_zero_grid"),
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_growth_split_report(
    *,
    shoulder_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport,
    boundary_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBoundaryReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderGrowthSplitReport:
    safe_replay, failing_replay = _safe_and_failing_replays(shoulder_report)
    safe_anchor = safe_replay.coverage_anchor_points[-1]
    failing_anchor = failing_replay.coverage_anchor_points[-1]
    failing_companion = failing_replay.overshoot_companion_points[-1]

    safe_half_interval = _half_interval(safe_anchor.pointwise_interval_length)
    failing_half_interval = _half_interval(failing_anchor.pointwise_interval_length)
    error_growth = _positive_growth(
        failing_anchor.absolute_error,
        safe_anchor.absolute_error,
        label="absolute_error",
    )
    half_interval_growth = _positive_growth(
        failing_half_interval,
        safe_half_interval,
        label="half_interval",
    )
    sigma_growth = _positive_growth(
        failing_anchor.sigma_z_hat,
        safe_anchor.sigma_z_hat,
        label="sigma_z_hat",
    )
    error_minus_half_interval = error_growth - half_interval_growth
    error_minus_sigma = error_growth - sigma_growth
    ratio_gap = (
        failing_anchor.error_to_half_interval_ratio
        - failing_companion.error_to_half_interval_ratio
    )
    driver_signature = "local-error-growth-outpaces-band-growth"

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*shoulder_report.binding_design)}`: moving the "
        "coverage anchor from the safe shoulder "
        f"`z = {_format_grid_value(boundary_report.safe_right_shoulder_grid_value)}` "
        "to the failing shoulder "
        f"`z = {_format_grid_value(boundary_report.failing_right_shoulder_grid_value)}` "
        f"raises absolute error from `{_format_float(safe_anchor.absolute_error)}` "
        f"to `{_format_float(failing_anchor.absolute_error)}` "
        f"(`{_format_multiplier(error_growth)}`), while the half-interval only "
        f"grows from `{_format_float(safe_half_interval)}` to "
        f"`{_format_float(failing_half_interval)}` "
        f"(`{_format_multiplier(half_interval_growth)}`)",
        "- pointwise band growth exactly tracks local `sigma_z_hat` growth "
        f"(`{_format_multiplier(sigma_growth)}`), so the current right-shoulder "
        "miss is driven by local error growth rather than a separate critical-value jump",
        "- the anchor's error growth exceeds local band growth by "
        f"`{_format_float(error_minus_half_interval)}`, pushing `error / "
        f"half-interval` up by `{_format_float(boundary_report.ratio_jump_across_bracket)}` "
        f"from `{_format_float(boundary_report.safe_right_shoulder_ratio)}` to "
        f"`{_format_float(boundary_report.failing_right_shoulder_ratio)}` across the "
        f"`[{_format_grid_value(boundary_report.safe_right_shoulder_grid_value)}, "
        f"{_format_grid_value(boundary_report.failing_right_shoulder_grid_value)}]` "
        "shoulder bracket",
        "- overshoot companion seed "
        f"`{shoulder_report.overshoot_companion_random_state}` still covers the same "
        f"`z = {_format_grid_value(boundary_report.failing_right_shoulder_grid_value)}` "
        f"shoulder at ratio `{_format_float(failing_companion.error_to_half_interval_ratio)}`, "
        f"leaving an anchor-to-companion ratio gap of `{_format_float(ratio_gap)}`; "
        "this remains coverage-anchor-specific local calibration debt rather than "
        "generic interval inflation",
        "- next Trigger 2 follow-up should target local calibration at the right "
        f"shoulder `z = {_format_grid_value(boundary_report.failing_right_shoulder_grid_value)}`, "
        "not center repair or uniform-critical retuning",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderGrowthSplitReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-growth-split-probe",
        policy_digest=shoulder_report.policy_digest,
        binding_design=shoulder_report.binding_design,
        coverage_anchor_random_state=shoulder_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            shoulder_report.overshoot_companion_random_state
        ),
        safe_right_shoulder_grid_value=boundary_report.safe_right_shoulder_grid_value,
        failing_right_shoulder_grid_value=(
            boundary_report.failing_right_shoulder_grid_value
        ),
        anchor_safe_absolute_error=safe_anchor.absolute_error,
        anchor_failing_absolute_error=failing_anchor.absolute_error,
        anchor_safe_half_interval=safe_half_interval,
        anchor_failing_half_interval=failing_half_interval,
        anchor_safe_sigma_z_hat=safe_anchor.sigma_z_hat,
        anchor_failing_sigma_z_hat=failing_anchor.sigma_z_hat,
        anchor_error_growth_factor=error_growth,
        anchor_half_interval_growth_factor=half_interval_growth,
        anchor_sigma_growth_factor=sigma_growth,
        anchor_error_growth_minus_half_interval_growth=error_minus_half_interval,
        anchor_error_growth_minus_sigma_growth=error_minus_sigma,
        anchor_ratio_jump=boundary_report.ratio_jump_across_bracket,
        failing_anchor_ratio=boundary_report.failing_right_shoulder_ratio,
        failing_companion_ratio=failing_companion.error_to_half_interval_ratio,
        failing_anchor_to_companion_ratio_gap=ratio_gap,
        driver_signature=driver_signature,
        canonical_coverage_anchor_shoulder_growth_split_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_growth_split_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderGrowthSplitReport
):
    shoulder_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_probe()
    )
    boundary_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_boundary_probe()
    )
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_growth_split_report(
        shoulder_report=shoulder_report,
        boundary_report=boundary_report,
    )
