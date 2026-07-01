from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_left_support_null_action_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchLeftSupportNullActionReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_left_support_null_action_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSpectralAdmissibilityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_variance_preservation_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchVariancePreservationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_variance_preservation_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _driver_signature(
    *,
    spectral_admissibility_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSpectralAdmissibilityReport
    ),
    left_support_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchLeftSupportNullActionReport
    ),
    variance_preservation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchVariancePreservationReport
    ),
    intake_guard_bundle_holds: bool,
) -> str:
    if (
        spectral_admissibility_report.driver_signature
        == "bounded-right-center-entry-patch-spectral-admissibility"
        and left_support_report.driver_signature
        == "bounded-right-center-entry-patch-left-support-null-action"
        and variance_preservation_report.driver_signature
        == "bounded-right-center-entry-patch-variance-preservation"
        and intake_guard_bundle_holds
    ):
        return "bounded-right-center-entry-patch-intake-bundle"
    return "mixed-right-center-entry-patch-intake-bundle"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    required_patch_increment: float
    psd_boundary_increment: float
    required_patch_share_of_psd_boundary: float
    remaining_psd_headroom: float
    remaining_psd_headroom_multiple_of_required_patch: float
    left_incident_absolute_patch_mass: float
    left_support_projection_norm: float
    trace_delta: float
    max_abs_diagonal_delta: float
    max_abs_std_delta: float
    spectral_admissible: bool
    preserves_left_support_contract: bool
    diagonal_preserved: bool
    intake_guard_bundle_holds: bool
    companion_driver_signatures: tuple[str, ...]
    driver_signature: str
    canonical_entry_patch_intake_bundle_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.required_patch_increment = float(self.required_patch_increment)
        self.psd_boundary_increment = float(self.psd_boundary_increment)
        self.required_patch_share_of_psd_boundary = float(
            self.required_patch_share_of_psd_boundary
        )
        self.remaining_psd_headroom = float(self.remaining_psd_headroom)
        self.remaining_psd_headroom_multiple_of_required_patch = float(
            self.remaining_psd_headroom_multiple_of_required_patch
        )
        self.left_incident_absolute_patch_mass = float(
            self.left_incident_absolute_patch_mass
        )
        self.left_support_projection_norm = float(self.left_support_projection_norm)
        self.trace_delta = float(self.trace_delta)
        self.max_abs_diagonal_delta = float(self.max_abs_diagonal_delta)
        self.max_abs_std_delta = float(self.max_abs_std_delta)
        self.spectral_admissible = bool(self.spectral_admissible)
        self.preserves_left_support_contract = bool(
            self.preserves_left_support_contract
        )
        self.diagonal_preserved = bool(self.diagonal_preserved)
        self.intake_guard_bundle_holds = bool(self.intake_guard_bundle_holds)
        self.companion_driver_signatures = tuple(
            str(item).strip() for item in self.companion_driver_signatures
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_entry_patch_intake_bundle_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_entry_patch_intake_bundle_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "evaluation_grid": list(self.evaluation_grid),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "required_patch_increment": self.required_patch_increment,
            "psd_boundary_increment": self.psd_boundary_increment,
            "required_patch_share_of_psd_boundary": (
                self.required_patch_share_of_psd_boundary
            ),
            "remaining_psd_headroom": self.remaining_psd_headroom,
            "remaining_psd_headroom_multiple_of_required_patch": (
                self.remaining_psd_headroom_multiple_of_required_patch
            ),
            "left_incident_absolute_patch_mass": self.left_incident_absolute_patch_mass,
            "left_support_projection_norm": self.left_support_projection_norm,
            "trace_delta": self.trace_delta,
            "max_abs_diagonal_delta": self.max_abs_diagonal_delta,
            "max_abs_std_delta": self.max_abs_std_delta,
            "spectral_admissible": self.spectral_admissible,
            "preserves_left_support_contract": self.preserves_left_support_contract,
            "diagonal_preserved": self.diagonal_preserved,
            "intake_guard_bundle_holds": self.intake_guard_bundle_holds,
            "companion_driver_signatures": list(self.companion_driver_signatures),
            "driver_signature": self.driver_signature,
            "canonical_entry_patch_intake_bundle_digest": list(
                self.canonical_entry_patch_intake_bundle_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_report(
    *,
    spectral_admissibility_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSpectralAdmissibilityReport
    ),
    left_support_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchLeftSupportNullActionReport
    ),
    variance_preservation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchVariancePreservationReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport:
    if spectral_admissibility_report.policy_digest != left_support_report.policy_digest:
        raise ValueError("intake bundle requires a shared policy digest")
    if (
        spectral_admissibility_report.policy_digest
        != variance_preservation_report.policy_digest
    ):
        raise ValueError("intake bundle requires a shared policy digest")
    if (
        spectral_admissibility_report.binding_design
        != left_support_report.binding_design
    ):
        raise ValueError("intake bundle requires a shared binding design")
    if (
        spectral_admissibility_report.binding_design
        != variance_preservation_report.binding_design
    ):
        raise ValueError("intake bundle requires a shared binding design")
    if spectral_admissibility_report.window_label != left_support_report.window_label:
        raise ValueError("intake bundle requires a shared window label")
    if (
        spectral_admissibility_report.window_label
        != variance_preservation_report.window_label
    ):
        raise ValueError("intake bundle requires a shared window label")

    companion_driver_signatures = (
        spectral_admissibility_report.driver_signature,
        left_support_report.driver_signature,
        variance_preservation_report.driver_signature,
    )
    spectral_admissible = bool(
        spectral_admissibility_report.diagonal_preserved
        and spectral_admissibility_report.patched_min_eigenvalue > 0.0
        and spectral_admissibility_report.required_patch_share_of_psd_boundary < 0.2
    )
    preserves_left_support_contract = bool(
        left_support_report.preserves_left_support_contract
        and left_support_report.confined_to_center_right_subspace
    )
    diagonal_preserved = bool(variance_preservation_report.diagonal_preserved)
    intake_guard_bundle_holds = bool(
        spectral_admissible and preserves_left_support_contract and diagonal_preserved
    )
    driver_signature = _driver_signature(
        spectral_admissibility_report=spectral_admissibility_report,
        left_support_report=left_support_report,
        variance_preservation_report=variance_preservation_report,
        intake_guard_bundle_holds=intake_guard_bundle_holds,
    )

    canonical_digest = (
        f"- binding design `DGP2/500/50` on `{spectral_admissibility_report.window_label}`: the same bounded two-cell patch still lifts the right-center covariance entry by `{_format_signed_float(spectral_admissibility_report.required_patch_increment)}` while keeping all three implementation-intake guards aligned on the exact `z = 0.05 / 0.15 / 0.25` witness grid",
        f"- spectral guard stays comfortable: the required patch still consumes only `{_format_percent(spectral_admissibility_report.required_patch_share_of_psd_boundary)}` of admissible PSD budget and keeps `{_format_float(spectral_admissibility_report.remaining_psd_headroom)}` headroom (`{_format_ratio(spectral_admissibility_report.remaining_psd_headroom_multiple_of_required_patch)}` the required patch) before the smallest eigenvalue reaches zero",
        f"- left-support guard and diagonal-preservation guard remain exact zero-action contracts: `left incident patch mass = {_format_float(left_support_report.left_incident_absolute_patch_mass)}`, `left-basis projection norm = {_format_float(left_support_report.left_support_projection_norm)}`, `max |diagonal delta| = {_format_float(variance_preservation_report.max_abs_diagonal_delta)}`, and `max |std delta| = {_format_float(variance_preservation_report.max_abs_std_delta)}`",
        "- current Trigger 2 implication: `bounded-right-center-entry-patch-intake-bundle`; this compresses `bounded-right-center-entry-patch-spectral-admissibility`, `bounded-right-center-entry-patch-left-support-null-action`, and `bounded-right-center-entry-patch-variance-preservation` into one validation-only implementation-intake witness, but it must not replace the live `bounded-right-center-execution-contract` routing surface",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-intake-bundle-probe"
        ),
        policy_digest=spectral_admissibility_report.policy_digest,
        binding_design=spectral_admissibility_report.binding_design,
        window_label=spectral_admissibility_report.window_label,
        evaluation_grid=spectral_admissibility_report.evaluation_grid,
        coverage_anchor_random_state=(
            spectral_admissibility_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            spectral_admissibility_report.overshoot_companion_random_state
        ),
        required_patch_increment=spectral_admissibility_report.required_patch_increment,
        psd_boundary_increment=spectral_admissibility_report.psd_boundary_increment,
        required_patch_share_of_psd_boundary=(
            spectral_admissibility_report.required_patch_share_of_psd_boundary
        ),
        remaining_psd_headroom=(spectral_admissibility_report.remaining_psd_headroom),
        remaining_psd_headroom_multiple_of_required_patch=(
            spectral_admissibility_report.remaining_psd_headroom_multiple_of_required_patch
        ),
        left_incident_absolute_patch_mass=(
            left_support_report.left_incident_absolute_patch_mass
        ),
        left_support_projection_norm=(left_support_report.left_support_projection_norm),
        trace_delta=variance_preservation_report.trace_delta,
        max_abs_diagonal_delta=(variance_preservation_report.max_abs_diagonal_delta),
        max_abs_std_delta=variance_preservation_report.max_abs_std_delta,
        spectral_admissible=spectral_admissible,
        preserves_left_support_contract=preserves_left_support_contract,
        diagonal_preserved=diagonal_preserved,
        intake_guard_bundle_holds=intake_guard_bundle_holds,
        companion_driver_signatures=companion_driver_signatures,
        driver_signature=driver_signature,
        canonical_entry_patch_intake_bundle_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_report(
        spectral_admissibility_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe()
        ),
        left_support_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_left_support_null_action_probe()
        ),
        variance_preservation_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_variance_preservation_probe()
        ),
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe",
]
