from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe,
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


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _required_coupled_sigma_access(
    *,
    current_coupled_sigma_access: float,
    companion_coupled_sigma_access: float,
    local_scale_access_shortfall_share: float,
) -> float:
    current_value = float(current_coupled_sigma_access)
    companion_value = float(companion_coupled_sigma_access)
    shortfall_share = float(local_scale_access_shortfall_share)
    if not 0.0 <= shortfall_share <= 1.0:
        raise ValueError("local_scale_access_shortfall_share must lie in [0, 1]")
    return float(current_value + shortfall_share * (companion_value - current_value))


def _driver_signature(
    *,
    access_share_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport
    ),
    center_coupling_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport
    ),
    fixed_sigma_required_correlation: float,
    fixed_correlation_required_sigma_z_hat: float,
) -> str:
    if (
        access_share_report.driver_signature
        == "single-share-local-scale-access-shortfall"
        and fixed_sigma_required_correlation
        <= center_coupling_report.overshoot_companion_shoulder_center_correlation
        and fixed_correlation_required_sigma_z_hat
        > center_coupling_report.overshoot_companion_shoulder_sigma_z_hat
    ):
        return "coupling-first-repair-target"
    if (
        fixed_correlation_required_sigma_z_hat
        <= center_coupling_report.overshoot_companion_shoulder_sigma_z_hat
    ):
        return "scale-first-repair-target"
    return "mixed-repair-target"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    local_scale_access_shortfall_share: float
    current_coupled_sigma_access: float
    companion_coupled_sigma_access: float
    required_coupled_sigma_access: float
    required_coupled_sigma_access_share_of_companion: float
    fixed_sigma_required_correlation: float
    fixed_sigma_correlation_multiple_vs_anchor: float
    fixed_sigma_correlation_share_of_companion: float
    fixed_correlation_required_sigma_z_hat: float
    fixed_correlation_sigma_multiple_vs_anchor: float
    fixed_correlation_sigma_multiple_vs_companion: float
    driver_signature: str
    canonical_coverage_anchor_shoulder_center_coupling_repair_digest: tuple[str, ...]

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
        self.local_scale_access_shortfall_share = float(
            self.local_scale_access_shortfall_share
        )
        self.current_coupled_sigma_access = float(self.current_coupled_sigma_access)
        self.companion_coupled_sigma_access = float(self.companion_coupled_sigma_access)
        self.required_coupled_sigma_access = float(self.required_coupled_sigma_access)
        self.required_coupled_sigma_access_share_of_companion = float(
            self.required_coupled_sigma_access_share_of_companion
        )
        self.fixed_sigma_required_correlation = float(
            self.fixed_sigma_required_correlation
        )
        self.fixed_sigma_correlation_multiple_vs_anchor = float(
            self.fixed_sigma_correlation_multiple_vs_anchor
        )
        self.fixed_sigma_correlation_share_of_companion = float(
            self.fixed_sigma_correlation_share_of_companion
        )
        self.fixed_correlation_required_sigma_z_hat = float(
            self.fixed_correlation_required_sigma_z_hat
        )
        self.fixed_correlation_sigma_multiple_vs_anchor = float(
            self.fixed_correlation_sigma_multiple_vs_anchor
        )
        self.fixed_correlation_sigma_multiple_vs_companion = float(
            self.fixed_correlation_sigma_multiple_vs_companion
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_coverage_anchor_shoulder_center_coupling_repair_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_center_coupling_repair_digest
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
            "local_scale_access_shortfall_share": self.local_scale_access_shortfall_share,
            "current_coupled_sigma_access": self.current_coupled_sigma_access,
            "companion_coupled_sigma_access": self.companion_coupled_sigma_access,
            "required_coupled_sigma_access": self.required_coupled_sigma_access,
            "required_coupled_sigma_access_share_of_companion": (
                self.required_coupled_sigma_access_share_of_companion
            ),
            "fixed_sigma_required_correlation": self.fixed_sigma_required_correlation,
            "fixed_sigma_correlation_multiple_vs_anchor": (
                self.fixed_sigma_correlation_multiple_vs_anchor
            ),
            "fixed_sigma_correlation_share_of_companion": (
                self.fixed_sigma_correlation_share_of_companion
            ),
            "fixed_correlation_required_sigma_z_hat": (
                self.fixed_correlation_required_sigma_z_hat
            ),
            "fixed_correlation_sigma_multiple_vs_anchor": (
                self.fixed_correlation_sigma_multiple_vs_anchor
            ),
            "fixed_correlation_sigma_multiple_vs_companion": (
                self.fixed_correlation_sigma_multiple_vs_companion
            ),
            "driver_signature": self.driver_signature,
            "canonical_coverage_anchor_shoulder_center_coupling_repair_digest": list(
                self.canonical_coverage_anchor_shoulder_center_coupling_repair_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_report(
    *,
    access_share_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport
    ),
    center_coupling_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport:
    if access_share_report.policy_digest != center_coupling_report.policy_digest:
        raise ValueError("center coupling repair probe requires a single policy digest")
    if access_share_report.binding_design != center_coupling_report.binding_design:
        raise ValueError(
            "center coupling repair probe requires a single binding design"
        )
    if (
        access_share_report.coverage_anchor_random_state
        != center_coupling_report.coverage_anchor_random_state
        or access_share_report.overshoot_companion_random_state
        != center_coupling_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "center coupling repair probe requires the same anchor/companion seeds"
        )
    if (
        access_share_report.failing_right_shoulder_grid_value
        != center_coupling_report.failing_right_shoulder_grid_value
    ):
        raise ValueError(
            "center coupling repair probe requires the same failing shoulder grid value"
        )

    current_coupled_sigma_access = (
        center_coupling_report.coverage_anchor_shoulder_center_coupled_sigma_access
    )
    companion_coupled_sigma_access = (
        center_coupling_report.overshoot_companion_shoulder_center_coupled_sigma_access
    )
    local_scale_access_shortfall_share = (
        access_share_report.local_scale_access_shortfall_share
    )
    if companion_coupled_sigma_access >= current_coupled_sigma_access:
        required_coupled_sigma_access = _required_coupled_sigma_access(
            current_coupled_sigma_access=current_coupled_sigma_access,
            companion_coupled_sigma_access=companion_coupled_sigma_access,
            local_scale_access_shortfall_share=local_scale_access_shortfall_share,
        )
        fixed_sigma_required_correlation = _positive_ratio(
            required_coupled_sigma_access,
            center_coupling_report.coverage_anchor_shoulder_sigma_z_hat,
            label="fixed_sigma_required_correlation",
        )
    else:
        current_correlation = (
            center_coupling_report.coverage_anchor_shoulder_center_correlation
        )
        companion_correlation = (
            center_coupling_report.overshoot_companion_shoulder_center_correlation
        )
        if companion_correlation < current_correlation:
            raise ValueError(
                "companion shoulder-center correlation must be at least the anchor correlation"
            )
        fixed_sigma_required_correlation = float(
            current_correlation
            + local_scale_access_shortfall_share
            * (companion_correlation - current_correlation)
        )
        required_coupled_sigma_access = float(
            fixed_sigma_required_correlation
            * center_coupling_report.coverage_anchor_shoulder_sigma_z_hat
        )
    fixed_correlation_required_sigma_z_hat = _positive_ratio(
        required_coupled_sigma_access,
        center_coupling_report.coverage_anchor_shoulder_center_correlation,
        label="fixed_correlation_required_sigma_z_hat",
    )
    required_coupled_sigma_access_share_of_companion = _positive_ratio(
        required_coupled_sigma_access,
        center_coupling_report.overshoot_companion_shoulder_center_coupled_sigma_access,
        label="required_coupled_sigma_access_share_of_companion",
    )
    fixed_sigma_correlation_multiple_vs_anchor = _positive_ratio(
        fixed_sigma_required_correlation,
        center_coupling_report.coverage_anchor_shoulder_center_correlation,
        label="fixed_sigma_correlation_multiple_vs_anchor",
    )
    fixed_sigma_correlation_share_of_companion = _positive_ratio(
        fixed_sigma_required_correlation,
        center_coupling_report.overshoot_companion_shoulder_center_correlation,
        label="fixed_sigma_correlation_share_of_companion",
    )
    fixed_correlation_sigma_multiple_vs_anchor = _positive_ratio(
        fixed_correlation_required_sigma_z_hat,
        center_coupling_report.coverage_anchor_shoulder_sigma_z_hat,
        label="fixed_correlation_sigma_multiple_vs_anchor",
    )
    fixed_correlation_sigma_multiple_vs_companion = _positive_ratio(
        fixed_correlation_required_sigma_z_hat,
        center_coupling_report.overshoot_companion_shoulder_sigma_z_hat,
        label="fixed_correlation_sigma_multiple_vs_companion",
    )
    driver_signature = _driver_signature(
        access_share_report=access_share_report,
        center_coupling_report=center_coupling_report,
        fixed_sigma_required_correlation=fixed_sigma_required_correlation,
        fixed_correlation_required_sigma_z_hat=fixed_correlation_required_sigma_z_hat,
    )

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*access_share_report.binding_design)}`: to close the last "
        f"`{_format_percent(access_share_report.local_scale_access_shortfall_share)}` "
        "local-scale access shortfall at the failing shoulder "
        f"`z = {_format_grid_value(access_share_report.failing_right_shoulder_grid_value)}`, "
        f"coverage anchor seed `{access_share_report.coverage_anchor_random_state}` only needs coupled "
        f"`sigma_z_hat` access to rise from "
        f"`{_format_float(center_coupling_report.coverage_anchor_shoulder_center_coupled_sigma_access)}` "
        f"to `{_format_float(required_coupled_sigma_access)}`, which "
        + (
            f"is still just `{_format_percent(required_coupled_sigma_access_share_of_companion)}` of companion "
            f"seed `{access_share_report.overshoot_companion_random_state}` access "
            f"`{_format_float(center_coupling_report.overshoot_companion_shoulder_center_coupled_sigma_access)}`"
            if required_coupled_sigma_access_share_of_companion <= 1.0
            else f"now exceeds companion seed `{access_share_report.overshoot_companion_random_state}` access at "
            f"`{_format_percent(required_coupled_sigma_access_share_of_companion)}` of "
            f"`{_format_float(center_coupling_report.overshoot_companion_shoulder_center_coupled_sigma_access)}`"
        ),
        "- if shoulder scale stays fixed at "
        f"`sigma_z_hat = {_format_float(center_coupling_report.coverage_anchor_shoulder_sigma_z_hat)}`, "
        "the required shoulder-center correlation only has to rise from "
        f"`{_format_float(center_coupling_report.coverage_anchor_shoulder_center_correlation)}` "
        f"to `{_format_float(fixed_sigma_required_correlation)}`: that is "
        f"`{_format_ratio(fixed_sigma_correlation_multiple_vs_anchor)}` the current anchor "
        "correlation, but still only "
        f"`{_format_percent(fixed_sigma_correlation_share_of_companion)}` of companion "
        f"correlation `{_format_float(center_coupling_report.overshoot_companion_shoulder_center_correlation)}`",
        "- if shoulder-center correlation stayed frozen at "
        f"`{_format_float(center_coupling_report.coverage_anchor_shoulder_center_correlation)}`, "
        f"the same repair would instead require `sigma_z_hat = {_format_float(fixed_correlation_required_sigma_z_hat)}`: "
        f"`{_format_ratio(fixed_correlation_sigma_multiple_vs_anchor)}` the current anchor shoulder "
        "scale and "
        f"`{_format_ratio(fixed_correlation_sigma_multiple_vs_companion)}` companion shoulder "
        f"scale `{_format_float(center_coupling_report.overshoot_companion_shoulder_sigma_z_hat)}`",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; source-level follow-up should target partial "
        f"shoulder-center covariance access restoration for seed "
        f"`{access_share_report.coverage_anchor_random_state}`, not plain shoulder "
        "scale inflation or global interval retuning",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-center-coupling-repair-probe",
        policy_digest=access_share_report.policy_digest,
        binding_design=access_share_report.binding_design,
        coverage_anchor_random_state=access_share_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            access_share_report.overshoot_companion_random_state
        ),
        center_grid_value=center_coupling_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            access_share_report.failing_right_shoulder_grid_value
        ),
        local_scale_access_shortfall_share=(
            access_share_report.local_scale_access_shortfall_share
        ),
        current_coupled_sigma_access=(
            center_coupling_report.coverage_anchor_shoulder_center_coupled_sigma_access
        ),
        companion_coupled_sigma_access=(
            center_coupling_report.overshoot_companion_shoulder_center_coupled_sigma_access
        ),
        required_coupled_sigma_access=required_coupled_sigma_access,
        required_coupled_sigma_access_share_of_companion=(
            required_coupled_sigma_access_share_of_companion
        ),
        fixed_sigma_required_correlation=fixed_sigma_required_correlation,
        fixed_sigma_correlation_multiple_vs_anchor=(
            fixed_sigma_correlation_multiple_vs_anchor
        ),
        fixed_sigma_correlation_share_of_companion=(
            fixed_sigma_correlation_share_of_companion
        ),
        fixed_correlation_required_sigma_z_hat=fixed_correlation_required_sigma_z_hat,
        fixed_correlation_sigma_multiple_vs_anchor=(
            fixed_correlation_sigma_multiple_vs_anchor
        ),
        fixed_correlation_sigma_multiple_vs_companion=(
            fixed_correlation_sigma_multiple_vs_companion
        ),
        driver_signature=driver_signature,
        canonical_coverage_anchor_shoulder_center_coupling_repair_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_report(
        access_share_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe(),
        center_coupling_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe(),
    )
