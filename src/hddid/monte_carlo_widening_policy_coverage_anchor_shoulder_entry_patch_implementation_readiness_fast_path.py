from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessFastPathReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    live_driver_signature: str
    source_driver_signature: str
    intake_driver_signature: str
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    required_patch_increment: float
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    source_target_omega_diagonal_entry: float
    source_required_omega_diagonal_increment: float
    shared_vf_entry_label: str
    required_patch_share_of_psd_boundary: float
    remaining_psd_headroom_multiple_of_required_patch: float
    left_incident_absolute_patch_mass: float
    max_abs_diagonal_delta: float
    prohibited_actions: tuple[str, ...]
    driver_signature: str
    canonical_fast_path_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.live_driver_signature = str(self.live_driver_signature).strip()
        self.source_driver_signature = str(self.source_driver_signature).strip()
        self.intake_driver_signature = str(self.intake_driver_signature).strip()
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.required_patch_increment = float(self.required_patch_increment)
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.source_target_omega_diagonal_entry = float(
            self.source_target_omega_diagonal_entry
        )
        self.source_required_omega_diagonal_increment = float(
            self.source_required_omega_diagonal_increment
        )
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.required_patch_share_of_psd_boundary = float(
            self.required_patch_share_of_psd_boundary
        )
        self.remaining_psd_headroom_multiple_of_required_patch = float(
            self.remaining_psd_headroom_multiple_of_required_patch
        )
        self.left_incident_absolute_patch_mass = float(
            self.left_incident_absolute_patch_mass
        )
        self.max_abs_diagonal_delta = float(self.max_abs_diagonal_delta)
        self.prohibited_actions = tuple(
            str(item).strip() for item in self.prohibited_actions
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_fast_path_digest = tuple(
            str(line).rstrip() for line in self.canonical_fast_path_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path_report(
    *,
    implementation_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport
    ),
    source_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport
    ),
    intake_bundle_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessFastPathReport:
    if implementation_priority_report.policy_digest != source_bridge_report.policy_digest:
        raise ValueError("fast-path readiness requires shared policy digest")
    if implementation_priority_report.policy_digest != intake_bundle_report.policy_digest:
        raise ValueError("fast-path readiness requires shared policy digest")
    if implementation_priority_report.binding_design != source_bridge_report.binding_design:
        raise ValueError("fast-path readiness requires shared binding design")
    if implementation_priority_report.binding_design != intake_bundle_report.binding_design:
        raise ValueError("fast-path readiness requires shared binding design")
    if implementation_priority_report.window_label != source_bridge_report.window_label:
        raise ValueError("fast-path readiness requires shared window label")
    if implementation_priority_report.window_label != intake_bundle_report.window_label:
        raise ValueError("fast-path readiness requires shared window label")
    if (
        implementation_priority_report.coverage_anchor_random_state
        != source_bridge_report.coverage_anchor_random_state
    ):
        raise ValueError("fast-path readiness requires shared coverage-anchor seed")
    if (
        implementation_priority_report.coverage_anchor_random_state
        != intake_bundle_report.coverage_anchor_random_state
    ):
        raise ValueError("fast-path readiness requires shared coverage-anchor seed")
    if (
        implementation_priority_report.overshoot_companion_random_state
        != source_bridge_report.overshoot_companion_random_state
    ):
        raise ValueError("fast-path readiness requires shared overshoot-companion seed")
    if (
        implementation_priority_report.overshoot_companion_random_state
        != intake_bundle_report.overshoot_companion_random_state
    ):
        raise ValueError("fast-path readiness requires shared overshoot-companion seed")
    if (
        implementation_priority_report.source_diagonal_coordinate
        != source_bridge_report.source_diagonal_coordinate
    ):
        raise ValueError("fast-path readiness requires shared source diagonal coordinate")
    if (
        implementation_priority_report.source_diagonal_basis_label
        != source_bridge_report.source_diagonal_basis_label
    ):
        raise ValueError("fast-path readiness requires shared source diagonal basis")
    if (
        implementation_priority_report.shared_vf_entry_label
        != source_bridge_report.source_shared_vf_entry_label
    ):
        raise ValueError("fast-path readiness requires shared vf entry label")
    uses_live_regrounded_priority = (
        implementation_priority_report.source_target_omega_diagonal_entry
        != source_bridge_report.source_target_omega_diagonal_entry
        or implementation_priority_report.source_required_omega_diagonal_increment
        != source_bridge_report.source_required_omega_diagonal_increment
    )
    if (
        source_bridge_report.validation_patch_increment
        != intake_bundle_report.required_patch_increment
    ):
        raise ValueError("fast-path readiness requires source bridge and intake patch agreement")

    canonical_digest = (
        f"- fast-path witness keeps the live Trigger 2 token fixed at `bounded-right-center-execution-contract` while reusing the same bounded patch `{_format_signed(source_bridge_report.validation_patch_increment)}` on `{'/'.join(map(str, implementation_priority_report.binding_design))}` / `{implementation_priority_report.window_label}` for `z = {_format_float(source_bridge_report.failing_right_shoulder_grid_value)} -> {_format_float(source_bridge_report.center_grid_value)}`",
        f"- source priority remains `{implementation_priority_report.source_diagonal_coordinate} = {implementation_priority_report.source_diagonal_basis_label}` routed through `omega_f_hat[{implementation_priority_report.source_diagonal_coordinate},{implementation_priority_report.source_diagonal_coordinate}] -> {implementation_priority_report.shared_vf_entry_label}`, with bounded diagonal omega lift `{_format_signed(implementation_priority_report.source_required_omega_diagonal_increment)}` up to `{_format_float(implementation_priority_report.source_target_omega_diagonal_entry)}`"
        + (
            f" while the frozen canonical `{_format_float(source_bridge_report.source_target_omega_diagonal_entry)}` packet stays validation-only"
            if uses_live_regrounded_priority
            else ""
        ),
        f"- intake guard still keeps the bounded patch small and local: `{_format_percent(intake_bundle_report.required_patch_share_of_psd_boundary)}` of PSD budget, `{_format_ratio(intake_bundle_report.remaining_psd_headroom_multiple_of_required_patch)}` remaining headroom, and `{_format_float(intake_bundle_report.left_incident_absolute_patch_mass)}` left-incident / `{_format_float(intake_bundle_report.max_abs_diagonal_delta)}` diagonal drift",
        "- current implication: `bounded-right-center-entry-patch-implementation-readiness-fast-path`; this is a validation-only fast-path witness and does not replace the heavier execution-contract replay",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessFastPathReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "entry-patch-implementation-readiness-fast-path"
        ),
        policy_digest=implementation_priority_report.policy_digest,
        binding_design=implementation_priority_report.binding_design,
        window_label=implementation_priority_report.window_label,
        coverage_anchor_random_state=implementation_priority_report.coverage_anchor_random_state,
        overshoot_companion_random_state=implementation_priority_report.overshoot_companion_random_state,
        live_driver_signature="bounded-right-center-execution-contract",
        source_driver_signature=implementation_priority_report.driver_signature,
        intake_driver_signature=intake_bundle_report.driver_signature,
        center_grid_value=source_bridge_report.center_grid_value,
        failing_right_shoulder_grid_value=source_bridge_report.failing_right_shoulder_grid_value,
        required_patch_increment=source_bridge_report.validation_patch_increment,
        source_diagonal_coordinate=implementation_priority_report.source_diagonal_coordinate,
        source_diagonal_basis_label=implementation_priority_report.source_diagonal_basis_label,
        source_target_omega_diagonal_entry=implementation_priority_report.source_target_omega_diagonal_entry,
        source_required_omega_diagonal_increment=implementation_priority_report.source_required_omega_diagonal_increment,
        shared_vf_entry_label=implementation_priority_report.shared_vf_entry_label,
        required_patch_share_of_psd_boundary=intake_bundle_report.required_patch_share_of_psd_boundary,
        remaining_psd_headroom_multiple_of_required_patch=intake_bundle_report.remaining_psd_headroom_multiple_of_required_patch,
        left_incident_absolute_patch_mass=intake_bundle_report.left_incident_absolute_patch_mass,
        max_abs_diagonal_delta=intake_bundle_report.max_abs_diagonal_delta,
        prohibited_actions=implementation_priority_report.prohibited_actions,
        driver_signature=(
            "bounded-right-center-entry-patch-implementation-readiness-fast-path"
        ),
        canonical_fast_path_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessFastPathReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path_report(
        implementation_priority_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest(),
        source_bridge_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract(),
        intake_bundle_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe(),
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessFastPathReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path",
]
