from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessFastPathReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _driver_signature(
    *,
    execution_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
    implementation_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport
    ),
    intake_bundle_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport
    ),
    readiness_contract_holds: bool,
) -> str:
    if (
        execution_contract_report.driver_signature
        == "bounded-right-center-execution-contract"
        and implementation_priority_report.driver_signature
        == "trigger2-entry-patch-implementation-priority-digest"
        and intake_bundle_report.driver_signature
        == "bounded-right-center-entry-patch-intake-bundle"
        and readiness_contract_holds
    ):
        return "bounded-right-center-entry-patch-implementation-readiness"
    return "mixed-right-center-entry-patch-implementation-readiness"


def _driver_signature_from_fast_path(
    *,
    fast_path_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessFastPathReport
    ),
    readiness_contract_holds: bool,
) -> str:
    if (
        fast_path_report.live_driver_signature
        == "bounded-right-center-execution-contract"
        and fast_path_report.source_driver_signature
        == "trigger2-entry-patch-implementation-priority-digest"
        and fast_path_report.intake_driver_signature
        == "bounded-right-center-entry-patch-intake-bundle"
        and fast_path_report.driver_signature
        == "bounded-right-center-entry-patch-implementation-readiness-fast-path"
        and readiness_contract_holds
    ):
        return "bounded-right-center-entry-patch-implementation-readiness"
    return "mixed-right-center-entry-patch-implementation-readiness"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    live_driver_signature: str
    source_driver_signature: str
    intake_driver_signature: str
    failing_right_shoulder_grid_value: float
    center_grid_value: float
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
    readiness_contract_holds: bool
    driver_signature: str
    canonical_implementation_readiness_digest: tuple[str, ...]

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
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.center_grid_value = float(self.center_grid_value)
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
        self.readiness_contract_holds = bool(self.readiness_contract_holds)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_implementation_readiness_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_implementation_readiness_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "live_driver_signature": self.live_driver_signature,
            "source_driver_signature": self.source_driver_signature,
            "intake_driver_signature": self.intake_driver_signature,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "center_grid_value": self.center_grid_value,
            "required_patch_increment": self.required_patch_increment,
            "source_diagonal_coordinate": self.source_diagonal_coordinate,
            "source_diagonal_basis_label": self.source_diagonal_basis_label,
            "source_target_omega_diagonal_entry": self.source_target_omega_diagonal_entry,
            "source_required_omega_diagonal_increment": self.source_required_omega_diagonal_increment,
            "shared_vf_entry_label": self.shared_vf_entry_label,
            "required_patch_share_of_psd_boundary": self.required_patch_share_of_psd_boundary,
            "remaining_psd_headroom_multiple_of_required_patch": self.remaining_psd_headroom_multiple_of_required_patch,
            "left_incident_absolute_patch_mass": self.left_incident_absolute_patch_mass,
            "max_abs_diagonal_delta": self.max_abs_diagonal_delta,
            "prohibited_actions": list(self.prohibited_actions),
            "readiness_contract_holds": self.readiness_contract_holds,
            "driver_signature": self.driver_signature,
            "canonical_implementation_readiness_digest": list(
                self.canonical_implementation_readiness_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_report(
    *,
    execution_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
    implementation_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport
    ),
    intake_bundle_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport:
    if (
        execution_contract_report.policy_digest
        != implementation_priority_report.policy_digest
    ):
        raise ValueError("implementation readiness probe requires shared policy digest")
    if execution_contract_report.policy_digest != intake_bundle_report.policy_digest:
        raise ValueError("implementation readiness probe requires shared policy digest")
    if (
        execution_contract_report.binding_design
        != implementation_priority_report.binding_design
    ):
        raise ValueError(
            "implementation readiness probe requires shared binding design"
        )
    if execution_contract_report.binding_design != intake_bundle_report.binding_design:
        raise ValueError(
            "implementation readiness probe requires shared binding design"
        )
    if (
        execution_contract_report.window_label
        != implementation_priority_report.window_label
    ):
        raise ValueError("implementation readiness probe requires shared window label")
    if execution_contract_report.window_label != intake_bundle_report.window_label:
        raise ValueError("implementation readiness probe requires shared window label")
    if (
        execution_contract_report.coverage_anchor_random_state
        != implementation_priority_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        execution_contract_report.coverage_anchor_random_state
        != intake_bundle_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        execution_contract_report.overshoot_companion_random_state
        != implementation_priority_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if (
        execution_contract_report.overshoot_companion_random_state
        != intake_bundle_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")

    readiness_contract_holds = bool(
        execution_contract_report.driver_signature
        == "bounded-right-center-execution-contract"
        and implementation_priority_report.driver_signature
        == "trigger2-entry-patch-implementation-priority-digest"
        and intake_bundle_report.intake_guard_bundle_holds
        and execution_contract_report.required_incremental_right_center_covariance_lift
        > 0.0
        and intake_bundle_report.required_patch_increment
        == execution_contract_report.required_incremental_right_center_covariance_lift
    )
    driver_signature = _driver_signature(
        execution_contract_report=execution_contract_report,
        implementation_priority_report=implementation_priority_report,
        intake_bundle_report=intake_bundle_report,
        readiness_contract_holds=readiness_contract_holds,
    )

    canonical_digest = (
        f"- live Trigger 2 repair remains `bounded-right-center-execution-contract`: binding design `{execution_contract_report.binding_design[0]}/{execution_contract_report.binding_design[1]}/{execution_contract_report.binding_design[2]}` on `{execution_contract_report.window_label}` still only needs the bounded `z = {_format_grid_value(execution_contract_report.failing_right_shoulder_grid_value)} -> {_format_grid_value(execution_contract_report.center_grid_value)}` shared-entry lift `{_format_signed_float(execution_contract_report.required_incremental_right_center_covariance_lift)}` while preserving left-side reserve and keeping sign-healing / denominator compression / whole-row replay out of lane",
        f"- source priority remains validation-only but pinned: coordinate `{implementation_priority_report.source_diagonal_coordinate} = {implementation_priority_report.source_diagonal_basis_label}` still routes the preferred source action through `omega_f_hat[2,2] -> {implementation_priority_report.shared_vf_entry_label}`, with bounded source realization `{_format_signed_float(implementation_priority_report.source_required_omega_diagonal_increment)}` up to `{_format_float(implementation_priority_report.source_target_omega_diagonal_entry)}`, so implementation should enter through the positive first-sine diagonal before any coordinate-`2` off-diagonal fallback",
        f"- intake guards still clear the patch for bounded estimator-path realization: the same patch consumes only `{_format_percent(intake_bundle_report.required_patch_share_of_psd_boundary)}` of PSD budget, leaves `{_format_ratio(intake_bundle_report.remaining_psd_headroom_multiple_of_required_patch)}` required-patch headroom, keeps left incident mass at `{_format_float(intake_bundle_report.left_incident_absolute_patch_mass)}`, and preserves diagonal / std deltas at `{_format_float(intake_bundle_report.max_abs_diagonal_delta)}`",
        "- current Trigger 2 implication: `bounded-right-center-entry-patch-implementation-readiness`; this is a validation-only implementation-readiness witness that compresses live repair scope, source priority, and intake safety into one handoff object, but it must not replace the live `bounded-right-center-execution-contract` routing surface",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-implementation-readiness-probe"
        ),
        policy_digest=execution_contract_report.policy_digest,
        binding_design=execution_contract_report.binding_design,
        window_label=execution_contract_report.window_label,
        coverage_anchor_random_state=execution_contract_report.coverage_anchor_random_state,
        overshoot_companion_random_state=execution_contract_report.overshoot_companion_random_state,
        live_driver_signature=execution_contract_report.driver_signature,
        source_driver_signature=implementation_priority_report.driver_signature,
        intake_driver_signature=intake_bundle_report.driver_signature,
        failing_right_shoulder_grid_value=execution_contract_report.failing_right_shoulder_grid_value,
        center_grid_value=execution_contract_report.center_grid_value,
        required_patch_increment=execution_contract_report.required_incremental_right_center_covariance_lift,
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
        readiness_contract_holds=readiness_contract_holds,
        driver_signature=driver_signature,
        canonical_implementation_readiness_digest=canonical_digest,
    )


def _build_from_fast_path(
    fast_path_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessFastPathReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport:
    readiness_contract_holds = bool(
        fast_path_report.live_driver_signature
        == "bounded-right-center-execution-contract"
        and fast_path_report.source_driver_signature
        == "trigger2-entry-patch-implementation-priority-digest"
        and fast_path_report.intake_driver_signature
        == "bounded-right-center-entry-patch-intake-bundle"
        and fast_path_report.driver_signature
        == "bounded-right-center-entry-patch-implementation-readiness-fast-path"
    )
    driver_signature = _driver_signature_from_fast_path(
        fast_path_report=fast_path_report,
        readiness_contract_holds=readiness_contract_holds,
    )
    canonical_digest = (
        f"- live Trigger 2 repair remains `bounded-right-center-execution-contract`: binding design `{fast_path_report.binding_design[0]}/{fast_path_report.binding_design[1]}/{fast_path_report.binding_design[2]}` on `{fast_path_report.window_label}` still only needs the bounded `z = {_format_grid_value(fast_path_report.failing_right_shoulder_grid_value)} -> {_format_grid_value(fast_path_report.center_grid_value)}` shared-entry lift `{_format_signed_float(fast_path_report.required_patch_increment)}` while preserving left-side reserve and keeping sign-healing / denominator compression / whole-row replay out of lane",
        f"- source priority remains validation-only but pinned: coordinate `{fast_path_report.source_diagonal_coordinate} = {fast_path_report.source_diagonal_basis_label}` still routes the preferred source action through `omega_f_hat[2,2] -> {fast_path_report.shared_vf_entry_label}`, with bounded source realization `{_format_signed_float(fast_path_report.source_required_omega_diagonal_increment)}` up to `{_format_float(fast_path_report.source_target_omega_diagonal_entry)}`, so implementation should enter through the positive first-sine diagonal before any coordinate-`2` off-diagonal fallback",
        f"- intake guards still clear the patch for bounded estimator-path realization: the same patch consumes only `{_format_percent(fast_path_report.required_patch_share_of_psd_boundary)}` of PSD budget, leaves `{_format_ratio(fast_path_report.remaining_psd_headroom_multiple_of_required_patch)}` required-patch headroom, keeps left incident mass at `{_format_float(fast_path_report.left_incident_absolute_patch_mass)}`, and preserves diagonal / std deltas at `{_format_float(fast_path_report.max_abs_diagonal_delta)}`",
        "- current Trigger 2 implication: `bounded-right-center-entry-patch-implementation-readiness`; this is a validation-only implementation-readiness witness that compresses live repair scope, source priority, and intake safety into one handoff object, but it must not replace the live `bounded-right-center-execution-contract` routing surface",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-implementation-readiness-probe"
        ),
        policy_digest=fast_path_report.policy_digest,
        binding_design=fast_path_report.binding_design,
        window_label=fast_path_report.window_label,
        coverage_anchor_random_state=fast_path_report.coverage_anchor_random_state,
        overshoot_companion_random_state=fast_path_report.overshoot_companion_random_state,
        live_driver_signature=fast_path_report.live_driver_signature,
        source_driver_signature=fast_path_report.source_driver_signature,
        intake_driver_signature=fast_path_report.intake_driver_signature,
        failing_right_shoulder_grid_value=fast_path_report.failing_right_shoulder_grid_value,
        center_grid_value=fast_path_report.center_grid_value,
        required_patch_increment=fast_path_report.required_patch_increment,
        source_diagonal_coordinate=fast_path_report.source_diagonal_coordinate,
        source_diagonal_basis_label=fast_path_report.source_diagonal_basis_label,
        source_target_omega_diagonal_entry=fast_path_report.source_target_omega_diagonal_entry,
        source_required_omega_diagonal_increment=fast_path_report.source_required_omega_diagonal_increment,
        shared_vf_entry_label=fast_path_report.shared_vf_entry_label,
        required_patch_share_of_psd_boundary=fast_path_report.required_patch_share_of_psd_boundary,
        remaining_psd_headroom_multiple_of_required_patch=fast_path_report.remaining_psd_headroom_multiple_of_required_patch,
        left_incident_absolute_patch_mass=fast_path_report.left_incident_absolute_patch_mass,
        max_abs_diagonal_delta=fast_path_report.max_abs_diagonal_delta,
        prohibited_actions=fast_path_report.prohibited_actions,
        readiness_contract_holds=readiness_contract_holds,
        driver_signature=driver_signature,
        canonical_implementation_readiness_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport
):
    return _build_from_fast_path(
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path()
    )
