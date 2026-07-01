from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_scale_repair_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderScaleRepairReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_scale_repair_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _positive_share(part: float, whole: float, *, label: str) -> float:
    whole_value = float(whole)
    if whole_value <= 0.0:
        raise ValueError(f"{label} whole must be positive")
    return float(float(part) / whole_value)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairShareReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    failing_right_shoulder_grid_value: float
    interval_scale_contract: float
    anchor_half_interval_repair: float
    anchor_sigma_z_hat_repair: float
    overshoot_companion_half_interval_surplus_over_requirement: float
    overshoot_companion_sigma_z_hat_surplus_over_requirement: float
    anchor_half_interval_repair_share_of_companion_surplus: float
    anchor_sigma_z_hat_repair_share_of_companion_surplus: float
    residual_companion_half_interval_surplus_after_anchor_repair: float
    residual_companion_sigma_z_hat_surplus_after_anchor_repair: float
    residual_companion_half_interval_surplus_share: float
    residual_companion_sigma_z_hat_surplus_share: float
    driver_signature: str
    canonical_coverage_anchor_shoulder_repair_share_digest: tuple[str, ...]

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
        self.anchor_half_interval_repair = float(self.anchor_half_interval_repair)
        self.anchor_sigma_z_hat_repair = float(self.anchor_sigma_z_hat_repair)
        self.overshoot_companion_half_interval_surplus_over_requirement = float(
            self.overshoot_companion_half_interval_surplus_over_requirement
        )
        self.overshoot_companion_sigma_z_hat_surplus_over_requirement = float(
            self.overshoot_companion_sigma_z_hat_surplus_over_requirement
        )
        self.anchor_half_interval_repair_share_of_companion_surplus = float(
            self.anchor_half_interval_repair_share_of_companion_surplus
        )
        self.anchor_sigma_z_hat_repair_share_of_companion_surplus = float(
            self.anchor_sigma_z_hat_repair_share_of_companion_surplus
        )
        self.residual_companion_half_interval_surplus_after_anchor_repair = float(
            self.residual_companion_half_interval_surplus_after_anchor_repair
        )
        self.residual_companion_sigma_z_hat_surplus_after_anchor_repair = float(
            self.residual_companion_sigma_z_hat_surplus_after_anchor_repair
        )
        self.residual_companion_half_interval_surplus_share = float(
            self.residual_companion_half_interval_surplus_share
        )
        self.residual_companion_sigma_z_hat_surplus_share = float(
            self.residual_companion_sigma_z_hat_surplus_share
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_coverage_anchor_shoulder_repair_share_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_repair_share_digest
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
            "anchor_half_interval_repair": self.anchor_half_interval_repair,
            "anchor_sigma_z_hat_repair": self.anchor_sigma_z_hat_repair,
            "overshoot_companion_half_interval_surplus_over_requirement": (
                self.overshoot_companion_half_interval_surplus_over_requirement
            ),
            "overshoot_companion_sigma_z_hat_surplus_over_requirement": (
                self.overshoot_companion_sigma_z_hat_surplus_over_requirement
            ),
            "anchor_half_interval_repair_share_of_companion_surplus": (
                self.anchor_half_interval_repair_share_of_companion_surplus
            ),
            "anchor_sigma_z_hat_repair_share_of_companion_surplus": (
                self.anchor_sigma_z_hat_repair_share_of_companion_surplus
            ),
            "residual_companion_half_interval_surplus_after_anchor_repair": (
                self.residual_companion_half_interval_surplus_after_anchor_repair
            ),
            "residual_companion_sigma_z_hat_surplus_after_anchor_repair": (
                self.residual_companion_sigma_z_hat_surplus_after_anchor_repair
            ),
            "residual_companion_half_interval_surplus_share": (
                self.residual_companion_half_interval_surplus_share
            ),
            "residual_companion_sigma_z_hat_surplus_share": (
                self.residual_companion_sigma_z_hat_surplus_share
            ),
            "driver_signature": self.driver_signature,
            "canonical_coverage_anchor_shoulder_repair_share_digest": list(
                self.canonical_coverage_anchor_shoulder_repair_share_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_share_report(
    *,
    scale_repair_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderScaleRepairReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairShareReport:
    anchor_half_interval_repair_share = _positive_share(
        scale_repair_report.anchor_half_interval_repair,
        scale_repair_report.overshoot_companion_half_interval_surplus_over_requirement,
        label="half_interval_surplus",
    )
    anchor_sigma_repair_share = _positive_share(
        scale_repair_report.anchor_sigma_z_hat_repair,
        scale_repair_report.overshoot_companion_sigma_z_hat_surplus_over_requirement,
        label="sigma_z_hat_surplus",
    )
    residual_half_interval_surplus = (
        scale_repair_report.overshoot_companion_half_interval_surplus_over_requirement
        - scale_repair_report.anchor_half_interval_repair
    )
    residual_sigma_z_hat_surplus = (
        scale_repair_report.overshoot_companion_sigma_z_hat_surplus_over_requirement
        - scale_repair_report.anchor_sigma_z_hat_repair
    )
    residual_half_interval_surplus_share = _positive_share(
        residual_half_interval_surplus,
        scale_repair_report.overshoot_companion_half_interval_surplus_over_requirement,
        label="residual_half_interval_surplus",
    )
    residual_sigma_z_hat_surplus_share = _positive_share(
        residual_sigma_z_hat_surplus,
        scale_repair_report.overshoot_companion_sigma_z_hat_surplus_over_requirement,
        label="residual_sigma_z_hat_surplus",
    )
    driver_signature = "anchor-repair-is-small-share-of-companion-surplus"

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*scale_repair_report.binding_design)}`: at the "
        f"failing shoulder `z = {_format_grid_value(scale_repair_report.failing_right_shoulder_grid_value)}`, "
        f"coverage anchor seed `{scale_repair_report.coverage_anchor_random_state}` "
        "still needs "
        f"`+{_format_float(scale_repair_report.anchor_half_interval_repair)}` "
        "half-interval and "
        f"`+{_format_float(scale_repair_report.anchor_sigma_z_hat_repair)}` "
        f"`sigma_z_hat` to reach zero reserve, while overshoot companion seed "
        f"`{scale_repair_report.overshoot_companion_random_state}` keeps same-point "
        f"surplus `+{_format_float(scale_repair_report.overshoot_companion_half_interval_surplus_over_requirement)}` "
        f"half-interval and `+{_format_float(scale_repair_report.overshoot_companion_sigma_z_hat_surplus_over_requirement)}` "
        f"`sigma_z_hat` above that boundary",
        "- the anchor would need only "
        f"`{_format_percent(anchor_half_interval_repair_share)}` of the companion's "
        "same-point half-interval surplus and "
        f"`{_format_percent(anchor_sigma_repair_share)}` of the companion's same-point "
        f"`sigma_z_hat` surplus to repair the miss at `z = {_format_grid_value(scale_repair_report.failing_right_shoulder_grid_value)}`",
        "- even after subtracting an anchor-sized repair, the companion would still "
        f"keep `+{_format_float(residual_half_interval_surplus)}` half-interval "
        f"and `+{_format_float(residual_sigma_z_hat_surplus)}` `sigma_z_hat` "
        f"surplus at the same shoulder, i.e. `{_format_percent(residual_half_interval_surplus_share)}` "
        "of its current surplus remains untouched",
        "- because both seeds still obey the same pointwise interval-scale contract "
        f"`{_format_float(scale_repair_report.interval_scale_contract)}`, the open "
        "debt is not a global shortage of local band width but anchor-specific "
        "failure to access the same local scale amplification",
        "- next Trigger 2 follow-up should explain why seed "
        f"`{scale_repair_report.coverage_anchor_random_state}` misses this last "
        f"`{_format_percent(anchor_half_interval_repair_share)}` of same-point scale "
        f"amplification at `z = {_format_grid_value(scale_repair_report.failing_right_shoulder_grid_value)}`, "
        "rather than widening every shoulder band",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairShareReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-repair-share-probe",
        policy_digest=scale_repair_report.policy_digest,
        binding_design=scale_repair_report.binding_design,
        coverage_anchor_random_state=scale_repair_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            scale_repair_report.overshoot_companion_random_state
        ),
        failing_right_shoulder_grid_value=(
            scale_repair_report.failing_right_shoulder_grid_value
        ),
        interval_scale_contract=scale_repair_report.interval_scale_contract,
        anchor_half_interval_repair=scale_repair_report.anchor_half_interval_repair,
        anchor_sigma_z_hat_repair=scale_repair_report.anchor_sigma_z_hat_repair,
        overshoot_companion_half_interval_surplus_over_requirement=(
            scale_repair_report.overshoot_companion_half_interval_surplus_over_requirement
        ),
        overshoot_companion_sigma_z_hat_surplus_over_requirement=(
            scale_repair_report.overshoot_companion_sigma_z_hat_surplus_over_requirement
        ),
        anchor_half_interval_repair_share_of_companion_surplus=(
            anchor_half_interval_repair_share
        ),
        anchor_sigma_z_hat_repair_share_of_companion_surplus=(
            anchor_sigma_repair_share
        ),
        residual_companion_half_interval_surplus_after_anchor_repair=(
            residual_half_interval_surplus
        ),
        residual_companion_sigma_z_hat_surplus_after_anchor_repair=(
            residual_sigma_z_hat_surplus
        ),
        residual_companion_half_interval_surplus_share=(
            residual_half_interval_surplus_share
        ),
        residual_companion_sigma_z_hat_surplus_share=(
            residual_sigma_z_hat_surplus_share
        ),
        driver_signature=driver_signature,
        canonical_coverage_anchor_shoulder_repair_share_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_share_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairShareReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_share_report(
        scale_repair_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_scale_repair_probe()
    )
