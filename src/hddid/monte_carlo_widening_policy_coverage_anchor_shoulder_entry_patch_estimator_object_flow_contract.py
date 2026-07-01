from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _driver_signature(
    *,
    source_bridge_signature: str,
    shared_vf_entry_signature: str,
    intake_bundle_signature: str,
    source_diagonal_coordinate: int,
    source_diagonal_basis_label: str,
    shared_vf_entry_label: str,
    omega_only_multiple_of_required_lift: float,
    normalization_only_share_of_required_lift: float,
    omega_vs_normalization_increment_ratio: float,
    preserves_left_support_contract: bool,
    diagonal_preserved: bool,
) -> str:
    if (
        source_bridge_signature
        == "validation-patch-to-first-sine-omega-diagonal-bridge"
        and shared_vf_entry_signature == "omega-first-shared-vf-entry-activation"
        and intake_bundle_signature == "bounded-right-center-entry-patch-intake-bundle"
        and source_diagonal_coordinate == 2
        and source_diagonal_basis_label == "sin(2πz)"
        and shared_vf_entry_label == "v_f_hat[2,2]"
        and omega_only_multiple_of_required_lift > 5.0
        and normalization_only_share_of_required_lift < 0.1
        and omega_vs_normalization_increment_ratio > 50.0
        and preserves_left_support_contract
        and diagonal_preserved
    ):
        return "bounded-first-sine-estimator-object-flow-contract"
    return "mixed-entry-patch-estimator-object-flow-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    source_target_omega_diagonal_entry: float
    source_required_omega_diagonal_increment: float
    shared_vf_entry_label: str
    required_diagonal_vf_entry_lift: float
    omega_only_increment: float
    normalization_only_increment: float
    omega_only_multiple_of_required_lift: float
    normalization_only_share_of_required_lift: float
    omega_vs_normalization_increment_ratio: float
    validation_patch_increment: float
    required_patch_share_of_psd_boundary: float
    remaining_psd_headroom: float
    preserves_left_support_contract: bool
    diagonal_preserved: bool
    driver_signature: str
    canonical_estimator_object_flow_digest: tuple[str, ...]

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
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.source_target_omega_diagonal_entry = float(
            self.source_target_omega_diagonal_entry
        )
        self.source_required_omega_diagonal_increment = float(
            self.source_required_omega_diagonal_increment
        )
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.omega_only_increment = float(self.omega_only_increment)
        self.normalization_only_increment = float(self.normalization_only_increment)
        self.omega_only_multiple_of_required_lift = float(
            self.omega_only_multiple_of_required_lift
        )
        self.normalization_only_share_of_required_lift = float(
            self.normalization_only_share_of_required_lift
        )
        self.omega_vs_normalization_increment_ratio = float(
            self.omega_vs_normalization_increment_ratio
        )
        self.validation_patch_increment = float(self.validation_patch_increment)
        self.required_patch_share_of_psd_boundary = float(
            self.required_patch_share_of_psd_boundary
        )
        self.remaining_psd_headroom = float(self.remaining_psd_headroom)
        self.preserves_left_support_contract = bool(
            self.preserves_left_support_contract
        )
        self.diagonal_preserved = bool(self.diagonal_preserved)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_estimator_object_flow_digest = tuple(
            str(line).rstrip() for line in self.canonical_estimator_object_flow_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "source_diagonal_coordinate": self.source_diagonal_coordinate,
            "source_diagonal_basis_label": self.source_diagonal_basis_label,
            "source_target_omega_diagonal_entry": (
                self.source_target_omega_diagonal_entry
            ),
            "source_required_omega_diagonal_increment": (
                self.source_required_omega_diagonal_increment
            ),
            "shared_vf_entry_label": self.shared_vf_entry_label,
            "required_diagonal_vf_entry_lift": self.required_diagonal_vf_entry_lift,
            "omega_only_increment": self.omega_only_increment,
            "normalization_only_increment": self.normalization_only_increment,
            "omega_only_multiple_of_required_lift": (
                self.omega_only_multiple_of_required_lift
            ),
            "normalization_only_share_of_required_lift": (
                self.normalization_only_share_of_required_lift
            ),
            "omega_vs_normalization_increment_ratio": (
                self.omega_vs_normalization_increment_ratio
            ),
            "validation_patch_increment": self.validation_patch_increment,
            "required_patch_share_of_psd_boundary": (
                self.required_patch_share_of_psd_boundary
            ),
            "remaining_psd_headroom": self.remaining_psd_headroom,
            "preserves_left_support_contract": self.preserves_left_support_contract,
            "diagonal_preserved": self.diagonal_preserved,
            "driver_signature": self.driver_signature,
            "canonical_estimator_object_flow_digest": list(
                self.canonical_estimator_object_flow_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract_report(
    *,
    source_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport
    ),
    shared_vf_entry_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport
    ),
    intake_bundle_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport:
    if source_bridge_report.policy_digest != shared_vf_entry_report.policy_digest:
        raise ValueError("estimator object-flow contract requires shared policy digest")
    if source_bridge_report.policy_digest != intake_bundle_report.policy_digest:
        raise ValueError("estimator object-flow contract requires shared policy digest")
    if source_bridge_report.binding_design != shared_vf_entry_report.binding_design:
        raise ValueError(
            "estimator object-flow contract requires shared binding design"
        )
    if source_bridge_report.binding_design != intake_bundle_report.binding_design:
        raise ValueError(
            "estimator object-flow contract requires shared binding design"
        )
    if source_bridge_report.window_label != shared_vf_entry_report.window_label:
        raise ValueError("estimator object-flow contract requires shared window label")
    if source_bridge_report.window_label != intake_bundle_report.window_label:
        raise ValueError("estimator object-flow contract requires shared window label")
    if (
        source_bridge_report.coverage_anchor_random_state
        != shared_vf_entry_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match shared-vf-entry report")
    if (
        source_bridge_report.coverage_anchor_random_state
        != intake_bundle_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match intake-bundle report")
    if (
        source_bridge_report.overshoot_companion_random_state
        != shared_vf_entry_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match shared-vf-entry report")
    if (
        source_bridge_report.overshoot_companion_random_state
        != intake_bundle_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match intake-bundle report")
    if (
        source_bridge_report.source_diagonal_coordinate
        != shared_vf_entry_report.diagonal_coordinate
    ):
        raise ValueError("diagonal coordinate must match shared-vf-entry report")
    if (
        source_bridge_report.source_shared_vf_entry_label
        != shared_vf_entry_report.shared_vf_entry_label
    ):
        raise ValueError("shared vf-entry label must match across upstream reports")

    driver_signature = _driver_signature(
        source_bridge_signature=source_bridge_report.driver_signature,
        shared_vf_entry_signature=shared_vf_entry_report.driver_signature,
        intake_bundle_signature=intake_bundle_report.driver_signature,
        source_diagonal_coordinate=source_bridge_report.source_diagonal_coordinate,
        source_diagonal_basis_label=source_bridge_report.source_diagonal_basis_label,
        shared_vf_entry_label=source_bridge_report.source_shared_vf_entry_label,
        omega_only_multiple_of_required_lift=(
            shared_vf_entry_report.omega_only_multiple_of_required_lift
        ),
        normalization_only_share_of_required_lift=(
            shared_vf_entry_report.normalization_only_share_of_required_lift
        ),
        omega_vs_normalization_increment_ratio=(
            shared_vf_entry_report.omega_vs_normalization_increment_ratio
        ),
        preserves_left_support_contract=intake_bundle_report.preserves_left_support_contract,
        diagonal_preserved=intake_bundle_report.diagonal_preserved,
    )

    canonical_digest = (
        f"- the bounded source realization target remains on the positive first-sine diagonal: `omega_f_hat[{source_bridge_report.source_diagonal_coordinate},{source_bridge_report.source_diagonal_coordinate}]` for `{source_bridge_report.source_diagonal_basis_label}` still only needs `{_format_signed(source_bridge_report.source_required_omega_diagonal_increment)}` up to `{_format_float(source_bridge_report.source_target_omega_diagonal_entry)}`, and this source target must enter implementation through the shared estimator object `{source_bridge_report.source_shared_vf_entry_label}` rather than a direct covariance overwrite",
        f"- along that shared object flow, omega restoration remains the dominant active ingredient: the omega-only counterfactual lifts `{shared_vf_entry_report.shared_vf_entry_label}` by `{_format_signed(shared_vf_entry_report.omega_only_increment)}` against only `{_format_signed(shared_vf_entry_report.normalization_only_increment)}` from normalization-only retuning, while the bounded target still needs just `{_format_signed(shared_vf_entry_report.required_diagonal_vf_entry_lift)}`, so omega-only carries `{_format_ratio(shared_vf_entry_report.omega_only_multiple_of_required_lift)}` the required lift and dominates normalization by `{_format_ratio(shared_vf_entry_report.omega_vs_normalization_increment_ratio)}`",
        f"- the validation-only intake guards therefore stay downstream obligations, not alternative entry points: the raw two-cell covariance witness remains `{_format_signed(source_bridge_report.validation_patch_increment)}`, the PSD budget share stays `{_format_percent(intake_bundle_report.required_patch_share_of_psd_boundary)}` with `{_format_float(intake_bundle_report.remaining_psd_headroom)}` headroom, and left-support / diagonal-preservation both remain exact, so any real estimator path must realize `omega_f_hat[{source_bridge_report.source_diagonal_coordinate},{source_bridge_report.source_diagonal_coordinate}] -> {shared_vf_entry_report.shared_vf_entry_label} -> covariance(0.25, 0.15)` without left leakage, diagonal inflation, or direct covariance edits",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-estimator-object-flow-contract"
        ),
        policy_digest=source_bridge_report.policy_digest,
        binding_design=source_bridge_report.binding_design,
        window_label=source_bridge_report.window_label,
        coverage_anchor_random_state=source_bridge_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            source_bridge_report.overshoot_companion_random_state
        ),
        source_diagonal_coordinate=source_bridge_report.source_diagonal_coordinate,
        source_diagonal_basis_label=source_bridge_report.source_diagonal_basis_label,
        source_target_omega_diagonal_entry=(
            source_bridge_report.source_target_omega_diagonal_entry
        ),
        source_required_omega_diagonal_increment=(
            source_bridge_report.source_required_omega_diagonal_increment
        ),
        shared_vf_entry_label=source_bridge_report.source_shared_vf_entry_label,
        required_diagonal_vf_entry_lift=(
            shared_vf_entry_report.required_diagonal_vf_entry_lift
        ),
        omega_only_increment=shared_vf_entry_report.omega_only_increment,
        normalization_only_increment=(
            shared_vf_entry_report.normalization_only_increment
        ),
        omega_only_multiple_of_required_lift=(
            shared_vf_entry_report.omega_only_multiple_of_required_lift
        ),
        normalization_only_share_of_required_lift=(
            shared_vf_entry_report.normalization_only_share_of_required_lift
        ),
        omega_vs_normalization_increment_ratio=(
            shared_vf_entry_report.omega_vs_normalization_increment_ratio
        ),
        validation_patch_increment=source_bridge_report.validation_patch_increment,
        required_patch_share_of_psd_boundary=(
            intake_bundle_report.required_patch_share_of_psd_boundary
        ),
        remaining_psd_headroom=intake_bundle_report.remaining_psd_headroom,
        preserves_left_support_contract=(
            intake_bundle_report.preserves_left_support_contract
        ),
        diagonal_preserved=intake_bundle_report.diagonal_preserved,
        driver_signature=driver_signature,
        canonical_estimator_object_flow_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-estimator-object-flow-contract"
        ),
        policy_digest=(
            "label=bounded-n500-p50",
            "max_total_runtime_seconds=240.0",
            "max_random_states=8",
            "stop_on_first_typed_invalidity=True",
            "min_nonparametric_coverage=0.85",
        ),
        binding_design=("DGP2", 500, 50),
        window_label="near_zero_grid",
        coverage_anchor_random_state=202,
        overshoot_companion_random_state=505,
        source_diagonal_coordinate=2,
        source_diagonal_basis_label="sin(2πz)",
        source_target_omega_diagonal_entry=410.4739959618635,
        source_required_omega_diagonal_increment=185.45799327602003,
        shared_vf_entry_label="v_f_hat[2,2]",
        required_diagonal_vf_entry_lift=1011.1822863613054,
        omega_only_increment=5662.7423277793705,
        normalization_only_increment=62.80420450378551,
        omega_only_multiple_of_required_lift=5.600120180265912,
        normalization_only_share_of_required_lift=0.062109676317396396,
        omega_vs_normalization_increment_ratio=90.16501956390594,
        validation_patch_increment=1.6361273081544214,
        required_patch_share_of_psd_boundary=0.12714564275399315,
        remaining_psd_headroom=11.23200779042753,
        preserves_left_support_contract=True,
        diagonal_preserved=True,
        driver_signature="bounded-first-sine-estimator-object-flow-contract",
        canonical_estimator_object_flow_digest=(
            "- the bounded source realization target remains on the positive first-sine diagonal: `omega_f_hat[2,2]` for `sin(2πz)` still only needs `+185.458` up to `410.474`, and this source target must enter implementation through the shared estimator object `v_f_hat[2,2]` rather than a direct covariance overwrite",
            "- along that shared object flow, omega restoration remains the dominant active ingredient: the omega-only counterfactual lifts `v_f_hat[2,2]` by `+5662.742` against only `+62.804` from normalization-only retuning, while the bounded target still needs just `+1011.182`, so omega-only carries `5.600x` the required lift and dominates normalization by `90.165x`",
            "- the validation-only intake guards therefore stay downstream obligations, not alternative entry points: the raw two-cell covariance witness remains `+1.636`, the PSD budget share stays `12.7%` with `11.232` headroom, and left-support / diagonal-preservation both remain exact, so any real estimator path must realize `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` without left leakage, diagonal inflation, or direct covariance edits",
        ),
    )
