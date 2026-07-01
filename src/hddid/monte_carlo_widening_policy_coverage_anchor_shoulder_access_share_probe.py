from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_repair_share_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairShareReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_share_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _alignment_gap(first: float, second: float) -> float:
    return float(abs(float(first) - float(second)))


def _mean_share(first: float, second: float) -> float:
    return float((float(first) + float(second)) / 2.0)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    failing_right_shoulder_grid_value: float
    interval_scale_contract: float
    half_interval_access_shortfall_share: float
    sigma_z_hat_access_shortfall_share: float
    access_shortfall_share_alignment_gap: float
    local_scale_access_shortfall_share: float
    half_interval_reserve_share: float
    sigma_z_hat_reserve_share: float
    reserve_share_alignment_gap: float
    local_scale_reserve_share: float
    driver_signature: str
    canonical_coverage_anchor_shoulder_access_share_digest: tuple[str, ...]

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
        self.half_interval_access_shortfall_share = float(
            self.half_interval_access_shortfall_share
        )
        self.sigma_z_hat_access_shortfall_share = float(
            self.sigma_z_hat_access_shortfall_share
        )
        self.access_shortfall_share_alignment_gap = float(
            self.access_shortfall_share_alignment_gap
        )
        self.local_scale_access_shortfall_share = float(
            self.local_scale_access_shortfall_share
        )
        self.half_interval_reserve_share = float(self.half_interval_reserve_share)
        self.sigma_z_hat_reserve_share = float(self.sigma_z_hat_reserve_share)
        self.reserve_share_alignment_gap = float(self.reserve_share_alignment_gap)
        self.local_scale_reserve_share = float(self.local_scale_reserve_share)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_coverage_anchor_shoulder_access_share_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_access_share_digest
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
            "half_interval_access_shortfall_share": (
                self.half_interval_access_shortfall_share
            ),
            "sigma_z_hat_access_shortfall_share": (
                self.sigma_z_hat_access_shortfall_share
            ),
            "access_shortfall_share_alignment_gap": (
                self.access_shortfall_share_alignment_gap
            ),
            "local_scale_access_shortfall_share": (
                self.local_scale_access_shortfall_share
            ),
            "half_interval_reserve_share": self.half_interval_reserve_share,
            "sigma_z_hat_reserve_share": self.sigma_z_hat_reserve_share,
            "reserve_share_alignment_gap": self.reserve_share_alignment_gap,
            "local_scale_reserve_share": self.local_scale_reserve_share,
            "driver_signature": self.driver_signature,
            "canonical_coverage_anchor_shoulder_access_share_digest": list(
                self.canonical_coverage_anchor_shoulder_access_share_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_report(
    *,
    repair_share_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairShareReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport:
    half_interval_access_shortfall_share = (
        repair_share_report.anchor_half_interval_repair_share_of_companion_surplus
    )
    sigma_z_hat_access_shortfall_share = (
        repair_share_report.anchor_sigma_z_hat_repair_share_of_companion_surplus
    )
    access_alignment_gap = _alignment_gap(
        half_interval_access_shortfall_share,
        sigma_z_hat_access_shortfall_share,
    )
    local_scale_access_shortfall_share = _mean_share(
        half_interval_access_shortfall_share,
        sigma_z_hat_access_shortfall_share,
    )

    half_interval_reserve_share = (
        repair_share_report.residual_companion_half_interval_surplus_share
    )
    sigma_z_hat_reserve_share = (
        repair_share_report.residual_companion_sigma_z_hat_surplus_share
    )
    reserve_alignment_gap = _alignment_gap(
        half_interval_reserve_share,
        sigma_z_hat_reserve_share,
    )
    local_scale_reserve_share = _mean_share(
        half_interval_reserve_share,
        sigma_z_hat_reserve_share,
    )

    driver_signature = "single-share-local-scale-access-shortfall"
    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*repair_share_report.binding_design)}`: at the failing shoulder "
        f"`z = {_format_grid_value(repair_share_report.failing_right_shoulder_grid_value)}`, "
        f"coverage anchor seed `{repair_share_report.coverage_anchor_random_state}` "
        "needs only "
        f"`{_format_percent(half_interval_access_shortfall_share)}` of overshoot companion "
        f"seed `{repair_share_report.overshoot_companion_random_state}` same-point "
        "half-interval surplus to repair the miss, and the required `sigma_z_hat` "
        f"share is the same `{_format_percent(sigma_z_hat_access_shortfall_share)}`",
        "- because both seeds still obey the same pointwise interval-scale contract "
        f"`{_format_float(repair_share_report.interval_scale_contract)}`, the "
        "half-interval and `sigma_z_hat` repair shares coincide up to machine "
        "rounding, so the open debt can be summarized as a single "
        f"`local_scale_access_shortfall_share = {_format_percent(local_scale_access_shortfall_share)}`",
        "- the untouched reserve share also collapses to the same single "
        f"companion-retained share `{_format_percent(local_scale_reserve_share)}`, "
        "rather than two competing band-vs-scale stories",
        "- current Trigger 2 debt is therefore not shoulder-wide interval inflation; "
        "it is a one-scalar local-scale access shortfall at "
        f"`z = {_format_grid_value(repair_share_report.failing_right_shoulder_grid_value)}`",
        "- next Trigger 2 follow-up should explain which source-level object "
        f"prevents seed `{repair_share_report.coverage_anchor_random_state}` from "
        f"accessing this last `{_format_percent(local_scale_access_shortfall_share)}` "
        "same-point local scale, not whether the global interval rule should change",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-access-share-probe",
        policy_digest=repair_share_report.policy_digest,
        binding_design=repair_share_report.binding_design,
        coverage_anchor_random_state=repair_share_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            repair_share_report.overshoot_companion_random_state
        ),
        failing_right_shoulder_grid_value=(
            repair_share_report.failing_right_shoulder_grid_value
        ),
        interval_scale_contract=repair_share_report.interval_scale_contract,
        half_interval_access_shortfall_share=half_interval_access_shortfall_share,
        sigma_z_hat_access_shortfall_share=sigma_z_hat_access_shortfall_share,
        access_shortfall_share_alignment_gap=access_alignment_gap,
        local_scale_access_shortfall_share=local_scale_access_shortfall_share,
        half_interval_reserve_share=half_interval_reserve_share,
        sigma_z_hat_reserve_share=sigma_z_hat_reserve_share,
        reserve_share_alignment_gap=reserve_alignment_gap,
        local_scale_reserve_share=local_scale_reserve_share,
        driver_signature=driver_signature,
        canonical_coverage_anchor_shoulder_access_share_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_report(
        repair_share_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_share_probe()
    )
