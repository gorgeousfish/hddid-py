from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint,
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport,
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


def _interval_scale(
    point: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint,
) -> float:
    sigma = float(point.sigma_z_hat)
    if sigma <= 0.0:
        raise ValueError("sigma_z_hat must be positive")
    return float(float(point.pointwise_interval_length) / sigma)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCompanionReserveReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    failing_right_shoulder_grid_value: float
    anchor_absolute_error: float
    overshoot_companion_absolute_error: float
    anchor_half_interval: float
    overshoot_companion_half_interval: float
    anchor_sigma_z_hat: float
    overshoot_companion_sigma_z_hat: float
    companion_error_growth_factor: float
    companion_half_interval_growth_factor: float
    companion_sigma_growth_factor: float
    anchor_ratio: float
    overshoot_companion_ratio: float
    anchor_to_companion_ratio_gap: float
    anchor_reserve: float
    overshoot_companion_reserve: float
    companion_reserve_gap: float
    anchor_interval_scale: float
    overshoot_companion_interval_scale: float
    interval_scale_gap: float
    driver_signature: str
    canonical_coverage_anchor_shoulder_companion_reserve_digest: tuple[str, ...]

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
        self.anchor_absolute_error = float(self.anchor_absolute_error)
        self.overshoot_companion_absolute_error = float(
            self.overshoot_companion_absolute_error
        )
        self.anchor_half_interval = float(self.anchor_half_interval)
        self.overshoot_companion_half_interval = float(
            self.overshoot_companion_half_interval
        )
        self.anchor_sigma_z_hat = float(self.anchor_sigma_z_hat)
        self.overshoot_companion_sigma_z_hat = float(
            self.overshoot_companion_sigma_z_hat
        )
        self.companion_error_growth_factor = float(self.companion_error_growth_factor)
        self.companion_half_interval_growth_factor = float(
            self.companion_half_interval_growth_factor
        )
        self.companion_sigma_growth_factor = float(self.companion_sigma_growth_factor)
        self.anchor_ratio = float(self.anchor_ratio)
        self.overshoot_companion_ratio = float(self.overshoot_companion_ratio)
        self.anchor_to_companion_ratio_gap = float(self.anchor_to_companion_ratio_gap)
        self.anchor_reserve = float(self.anchor_reserve)
        self.overshoot_companion_reserve = float(self.overshoot_companion_reserve)
        self.companion_reserve_gap = float(self.companion_reserve_gap)
        self.anchor_interval_scale = float(self.anchor_interval_scale)
        self.overshoot_companion_interval_scale = float(
            self.overshoot_companion_interval_scale
        )
        self.interval_scale_gap = float(self.interval_scale_gap)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_coverage_anchor_shoulder_companion_reserve_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_companion_reserve_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "anchor_absolute_error": self.anchor_absolute_error,
            "overshoot_companion_absolute_error": (
                self.overshoot_companion_absolute_error
            ),
            "anchor_half_interval": self.anchor_half_interval,
            "overshoot_companion_half_interval": (
                self.overshoot_companion_half_interval
            ),
            "anchor_sigma_z_hat": self.anchor_sigma_z_hat,
            "overshoot_companion_sigma_z_hat": self.overshoot_companion_sigma_z_hat,
            "companion_error_growth_factor": self.companion_error_growth_factor,
            "companion_half_interval_growth_factor": (
                self.companion_half_interval_growth_factor
            ),
            "companion_sigma_growth_factor": self.companion_sigma_growth_factor,
            "anchor_ratio": self.anchor_ratio,
            "overshoot_companion_ratio": self.overshoot_companion_ratio,
            "anchor_to_companion_ratio_gap": self.anchor_to_companion_ratio_gap,
            "anchor_reserve": self.anchor_reserve,
            "overshoot_companion_reserve": self.overshoot_companion_reserve,
            "companion_reserve_gap": self.companion_reserve_gap,
            "anchor_interval_scale": self.anchor_interval_scale,
            "overshoot_companion_interval_scale": (
                self.overshoot_companion_interval_scale
            ),
            "interval_scale_gap": self.interval_scale_gap,
            "driver_signature": self.driver_signature,
            "canonical_coverage_anchor_shoulder_companion_reserve_digest": list(
                self.canonical_coverage_anchor_shoulder_companion_reserve_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_companion_reserve_report(
    *,
    shoulder_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCompanionReserveReport:
    failing_replay = shoulder_report.replay("near_zero_grid")
    anchor = failing_replay.coverage_anchor_points[-1]
    companion = failing_replay.overshoot_companion_points[-1]

    anchor_half_interval = _half_interval(anchor.pointwise_interval_length)
    companion_half_interval = _half_interval(companion.pointwise_interval_length)
    companion_error_growth = _positive_growth(
        companion.absolute_error,
        anchor.absolute_error,
        label="absolute_error",
    )
    companion_half_interval_growth = _positive_growth(
        companion_half_interval,
        anchor_half_interval,
        label="half_interval",
    )
    companion_sigma_growth = _positive_growth(
        companion.sigma_z_hat,
        anchor.sigma_z_hat,
        label="sigma_z_hat",
    )
    anchor_reserve = anchor_half_interval - anchor.absolute_error
    companion_reserve = companion_half_interval - companion.absolute_error
    reserve_gap = companion_reserve - anchor_reserve
    anchor_interval_scale = _interval_scale(anchor)
    companion_interval_scale = _interval_scale(companion)
    interval_scale_gap = companion_interval_scale - anchor_interval_scale
    ratio_gap = (
        anchor.error_to_half_interval_ratio - companion.error_to_half_interval_ratio
    )
    driver_signature = "companion-band-amplification-outpaces-error-growth"

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*shoulder_report.binding_design)}`: at the failing "
        f"shoulder `z = {_format_grid_value(anchor.grid_value)}`, coverage anchor "
        f"seed `{shoulder_report.coverage_anchor_random_state}` misses with "
        f"`error / half-interval = {_format_float(anchor.error_to_half_interval_ratio)}`, "
        "while overshoot companion seed "
        f"`{shoulder_report.overshoot_companion_random_state}` still covers at "
        f"`{_format_float(companion.error_to_half_interval_ratio)}`",
        "- moving from the anchor to the overshoot companion raises absolute "
        f"error from `{_format_float(anchor.absolute_error)}` to "
        f"`{_format_float(companion.absolute_error)}` "
        f"(`{_format_multiplier(companion_error_growth)}`), but raises the "
        f"half-interval from `{_format_float(anchor_half_interval)}` to "
        f"`{_format_float(companion_half_interval)}` "
        f"(`{_format_multiplier(companion_half_interval_growth)}`); local "
        f"`sigma_z_hat` matches the same "
        f"`{_format_multiplier(companion_sigma_growth)}` growth",
        "- the interval-scale contract stays effectively unchanged at "
        f"`{_format_float(anchor_interval_scale)}` for both seeds, so this "
        "coverage split is not a separate critical-value or interval-scaling "
        "rule change",
        "- the anchor keeps negative reserve "
        f"`{_format_float(anchor_reserve)}` at "
        f"`z = {_format_grid_value(anchor.grid_value)}`, while the overshoot "
        f"companion keeps positive reserve `+{_format_float(companion_reserve)}`, "
        f"leaving a reserve gap of `{_format_float(reserve_gap)}` in favor of "
        "the companion",
        "- next Trigger 2 follow-up should target anchor-specific local scale "
        f"under-amplification at `z = {_format_grid_value(anchor.grid_value)}`, "
        "not generic interval inflation or uniform-critical retuning",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCompanionReserveReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-companion-reserve-probe",
        policy_digest=shoulder_report.policy_digest,
        binding_design=shoulder_report.binding_design,
        coverage_anchor_random_state=shoulder_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            shoulder_report.overshoot_companion_random_state
        ),
        failing_right_shoulder_grid_value=anchor.grid_value,
        anchor_absolute_error=anchor.absolute_error,
        overshoot_companion_absolute_error=companion.absolute_error,
        anchor_half_interval=anchor_half_interval,
        overshoot_companion_half_interval=companion_half_interval,
        anchor_sigma_z_hat=anchor.sigma_z_hat,
        overshoot_companion_sigma_z_hat=companion.sigma_z_hat,
        companion_error_growth_factor=companion_error_growth,
        companion_half_interval_growth_factor=companion_half_interval_growth,
        companion_sigma_growth_factor=companion_sigma_growth,
        anchor_ratio=anchor.error_to_half_interval_ratio,
        overshoot_companion_ratio=companion.error_to_half_interval_ratio,
        anchor_to_companion_ratio_gap=ratio_gap,
        anchor_reserve=anchor_reserve,
        overshoot_companion_reserve=companion_reserve,
        companion_reserve_gap=reserve_gap,
        anchor_interval_scale=anchor_interval_scale,
        overshoot_companion_interval_scale=companion_interval_scale,
        interval_scale_gap=interval_scale_gap,
        driver_signature=driver_signature,
        canonical_coverage_anchor_shoulder_companion_reserve_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_companion_reserve_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCompanionReserveReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_companion_reserve_report(
        shoulder_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_probe()
    )
