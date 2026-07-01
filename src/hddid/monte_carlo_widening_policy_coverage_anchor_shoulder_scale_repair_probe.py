from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_companion_reserve_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCompanionReserveReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_companion_reserve_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_multiplier(value: float) -> str:
    return f"x{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _required_sigma_z_hat(
    required_half_interval: float,
    interval_scale_contract: float,
) -> float:
    interval_scale = float(interval_scale_contract)
    if interval_scale <= 0.0:
        raise ValueError("interval_scale_contract must be positive")
    return float(2.0 * float(required_half_interval) / interval_scale)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderScaleRepairReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    failing_right_shoulder_grid_value: float
    interval_scale_contract: float
    anchor_absolute_error: float
    anchor_current_half_interval: float
    anchor_current_sigma_z_hat: float
    anchor_required_half_interval: float
    anchor_required_sigma_z_hat: float
    anchor_half_interval_repair: float
    anchor_sigma_z_hat_repair: float
    anchor_scale_repair_factor: float
    anchor_scale_shortfall_share: float
    overshoot_companion_half_interval: float
    overshoot_companion_sigma_z_hat: float
    overshoot_companion_half_interval_surplus_over_requirement: float
    overshoot_companion_sigma_z_hat_surplus_over_requirement: float
    overshoot_companion_surplus_factor_over_requirement: float
    driver_signature: str
    canonical_coverage_anchor_shoulder_scale_repair_digest: tuple[str, ...]

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
        self.interval_scale_contract = float(self.interval_scale_contract)
        self.anchor_absolute_error = float(self.anchor_absolute_error)
        self.anchor_current_half_interval = float(self.anchor_current_half_interval)
        self.anchor_current_sigma_z_hat = float(self.anchor_current_sigma_z_hat)
        self.anchor_required_half_interval = float(self.anchor_required_half_interval)
        self.anchor_required_sigma_z_hat = float(self.anchor_required_sigma_z_hat)
        self.anchor_half_interval_repair = float(self.anchor_half_interval_repair)
        self.anchor_sigma_z_hat_repair = float(self.anchor_sigma_z_hat_repair)
        self.anchor_scale_repair_factor = float(self.anchor_scale_repair_factor)
        self.anchor_scale_shortfall_share = float(self.anchor_scale_shortfall_share)
        self.overshoot_companion_half_interval = float(
            self.overshoot_companion_half_interval
        )
        self.overshoot_companion_sigma_z_hat = float(
            self.overshoot_companion_sigma_z_hat
        )
        self.overshoot_companion_half_interval_surplus_over_requirement = float(
            self.overshoot_companion_half_interval_surplus_over_requirement
        )
        self.overshoot_companion_sigma_z_hat_surplus_over_requirement = float(
            self.overshoot_companion_sigma_z_hat_surplus_over_requirement
        )
        self.overshoot_companion_surplus_factor_over_requirement = float(
            self.overshoot_companion_surplus_factor_over_requirement
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_coverage_anchor_shoulder_scale_repair_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_scale_repair_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "interval_scale_contract": self.interval_scale_contract,
            "anchor_absolute_error": self.anchor_absolute_error,
            "anchor_current_half_interval": self.anchor_current_half_interval,
            "anchor_current_sigma_z_hat": self.anchor_current_sigma_z_hat,
            "anchor_required_half_interval": self.anchor_required_half_interval,
            "anchor_required_sigma_z_hat": self.anchor_required_sigma_z_hat,
            "anchor_half_interval_repair": self.anchor_half_interval_repair,
            "anchor_sigma_z_hat_repair": self.anchor_sigma_z_hat_repair,
            "anchor_scale_repair_factor": self.anchor_scale_repair_factor,
            "anchor_scale_shortfall_share": self.anchor_scale_shortfall_share,
            "overshoot_companion_half_interval": (
                self.overshoot_companion_half_interval
            ),
            "overshoot_companion_sigma_z_hat": self.overshoot_companion_sigma_z_hat,
            "overshoot_companion_half_interval_surplus_over_requirement": (
                self.overshoot_companion_half_interval_surplus_over_requirement
            ),
            "overshoot_companion_sigma_z_hat_surplus_over_requirement": (
                self.overshoot_companion_sigma_z_hat_surplus_over_requirement
            ),
            "overshoot_companion_surplus_factor_over_requirement": (
                self.overshoot_companion_surplus_factor_over_requirement
            ),
            "driver_signature": self.driver_signature,
            "canonical_coverage_anchor_shoulder_scale_repair_digest": list(
                self.canonical_coverage_anchor_shoulder_scale_repair_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_scale_repair_report(
    *,
    companion_reserve_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCompanionReserveReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderScaleRepairReport:
    required_half_interval = companion_reserve_report.anchor_absolute_error
    required_sigma_z_hat = _required_sigma_z_hat(
        required_half_interval,
        companion_reserve_report.anchor_interval_scale,
    )
    anchor_half_interval_repair = (
        required_half_interval - companion_reserve_report.anchor_half_interval
    )
    anchor_sigma_z_hat_repair = (
        required_sigma_z_hat - companion_reserve_report.anchor_sigma_z_hat
    )
    anchor_scale_repair_factor = (
        required_half_interval / companion_reserve_report.anchor_half_interval
    )
    anchor_scale_shortfall_share = anchor_scale_repair_factor - 1.0
    companion_half_interval_surplus = (
        companion_reserve_report.overshoot_companion_half_interval
        - required_half_interval
    )
    companion_sigma_z_hat_surplus = (
        companion_reserve_report.overshoot_companion_sigma_z_hat - required_sigma_z_hat
    )
    companion_surplus_factor = (
        companion_reserve_report.overshoot_companion_half_interval
        / required_half_interval
    )
    driver_signature = "anchor-specific-local-scale-under-amplification"

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*companion_reserve_report.binding_design)}`: at the "
        f"failing shoulder `z = {_format_grid_value(companion_reserve_report.failing_right_shoulder_grid_value)}`, "
        f"coverage anchor seed `{companion_reserve_report.coverage_anchor_random_state}` "
        f"carries half-interval `{_format_float(companion_reserve_report.anchor_half_interval)}` "
        f"against absolute error `{_format_float(companion_reserve_report.anchor_absolute_error)}`, "
        f"leaving a local band miss of `{_format_float(anchor_half_interval_repair)}`",
        "- keeping the pointwise interval-scale contract fixed at "
        f"`{_format_float(companion_reserve_report.anchor_interval_scale)}`, seed "
        f"`{companion_reserve_report.coverage_anchor_random_state}` would need "
        f"half-interval `{_format_float(required_half_interval)}` and `sigma_z_hat` "
        f"`{_format_float(required_sigma_z_hat)}` to just reach zero reserve; that "
        f"is `+{_format_float(anchor_half_interval_repair)}` half-interval, "
        f"`+{_format_float(anchor_sigma_z_hat_repair)}` `sigma_z_hat`, and "
        f"`{_format_multiplier(anchor_scale_repair_factor)}` local scale repair over "
        "the current shoulder",
        "- overshoot companion seed "
        f"`{companion_reserve_report.overshoot_companion_random_state}` already "
        f"carries half-interval `{_format_float(companion_reserve_report.overshoot_companion_half_interval)}` "
        f"and `sigma_z_hat` `{_format_float(companion_reserve_report.overshoot_companion_sigma_z_hat)}` "
        f"at the same `z = {_format_grid_value(companion_reserve_report.failing_right_shoulder_grid_value)}`, "
        f"leaving surplus `+{_format_float(companion_half_interval_surplus)}` half-interval "
        f"and `+{_format_float(companion_sigma_z_hat_surplus)}` `sigma_z_hat` above "
        f"the anchor's boundary requirement (`{_format_multiplier(companion_surplus_factor)}` "
        "of required half-interval)",
        "- because both seeds still obey the same interval-scale contract "
        f"`{_format_float(companion_reserve_report.anchor_interval_scale)}`, the "
        "open debt is anchor-specific local scale under-amplification, not a "
        "different critical-value or interval-scaling rule",
        "- next Trigger 2 follow-up should explain why seed "
        f"`{companion_reserve_report.coverage_anchor_random_state}` stalls "
        f"`{_format_percent(anchor_scale_shortfall_share)}` below the local scale "
        f"needed at `z = {_format_grid_value(companion_reserve_report.failing_right_shoulder_grid_value)}`, "
        "rather than widening all shoulders indiscriminately",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderScaleRepairReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-scale-repair-probe",
        policy_digest=companion_reserve_report.policy_digest,
        binding_design=companion_reserve_report.binding_design,
        coverage_anchor_random_state=(
            companion_reserve_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            companion_reserve_report.overshoot_companion_random_state
        ),
        failing_right_shoulder_grid_value=(
            companion_reserve_report.failing_right_shoulder_grid_value
        ),
        interval_scale_contract=companion_reserve_report.anchor_interval_scale,
        anchor_absolute_error=companion_reserve_report.anchor_absolute_error,
        anchor_current_half_interval=companion_reserve_report.anchor_half_interval,
        anchor_current_sigma_z_hat=companion_reserve_report.anchor_sigma_z_hat,
        anchor_required_half_interval=required_half_interval,
        anchor_required_sigma_z_hat=required_sigma_z_hat,
        anchor_half_interval_repair=anchor_half_interval_repair,
        anchor_sigma_z_hat_repair=anchor_sigma_z_hat_repair,
        anchor_scale_repair_factor=anchor_scale_repair_factor,
        anchor_scale_shortfall_share=anchor_scale_shortfall_share,
        overshoot_companion_half_interval=(
            companion_reserve_report.overshoot_companion_half_interval
        ),
        overshoot_companion_sigma_z_hat=(
            companion_reserve_report.overshoot_companion_sigma_z_hat
        ),
        overshoot_companion_half_interval_surplus_over_requirement=(
            companion_half_interval_surplus
        ),
        overshoot_companion_sigma_z_hat_surplus_over_requirement=(
            companion_sigma_z_hat_surplus
        ),
        overshoot_companion_surplus_factor_over_requirement=(companion_surplus_factor),
        driver_signature=driver_signature,
        canonical_coverage_anchor_shoulder_scale_repair_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_scale_repair_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderScaleRepairReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_scale_repair_report(
        companion_reserve_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_companion_reserve_probe()
    )
