from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_stage_ladder,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_shared_entry_patch_budget import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSharedEntryPatchBudgetReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_shared_entry_patch_budget,
)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    estimator_object_flow_signature: str,
    shared_entry_budget_signature: str,
    compensating_stage_ladder_signature: str,
    source_diagonal_coordinate: int,
    source_diagonal_basis_label: str,
    shared_vf_entry_label: str,
    preserves_left_support_contract: bool,
    diagonal_preserved: bool,
    direct_covariance_edits_allowed: bool,
    required_patch_share_of_full_shared_vf_gap: float,
    required_patch_share_of_omega_only_shared_vf_increment: float,
    required_patch_share_of_diagonal_omega_gap: float,
    required_patch_share_of_psd_boundary: float,
    compensating_cumulative_share_after_left_center: float,
    compensating_zero_live_entry_count: int,
) -> str:
    if (
        estimator_object_flow_signature
        == "bounded-first-sine-estimator-object-flow-contract"
        and shared_entry_budget_signature
        == "bounded-first-sine-shared-entry-patch-budget"
        and compensating_stage_ladder_signature
        == "first-sine-compensating-geometry-stage-ladder"
        and source_diagonal_coordinate == 2
        and source_diagonal_basis_label == "sin(2πz)"
        and shared_vf_entry_label == "v_f_hat[2,2]"
        and preserves_left_support_contract
        and diagonal_preserved
        and not direct_covariance_edits_allowed
        and 0.11 < required_patch_share_of_full_shared_vf_gap < 0.13
        and 0.17 < required_patch_share_of_omega_only_shared_vf_increment < 0.19
        and 0.24 < required_patch_share_of_diagonal_omega_gap < 0.25
        and 0.12 < required_patch_share_of_psd_boundary < 0.13
        and compensating_cumulative_share_after_left_center > 0.96
        and compensating_zero_live_entry_count == 2
    ):
        return "first-sine-preserve-left-support-runtime-witness-contract"
    return "mixed-first-sine-runtime-witness-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    shared_vf_entry_label: str
    live_covariance_entry_label: str
    runtime_witness_path: tuple[str, ...]
    preserves_left_support_contract: bool
    diagonal_preserved: bool
    direct_covariance_edits_allowed: bool
    required_patch_share_of_full_shared_vf_gap: float
    required_patch_share_of_omega_only_shared_vf_increment: float
    required_patch_share_of_diagonal_omega_gap: float
    required_patch_share_of_psd_boundary: float
    remaining_psd_headroom: float
    compensating_stage_order: tuple[str, ...]
    compensating_cumulative_share_of_total_absolute_mass: tuple[float, ...]
    compensating_zero_live_entry_count: int
    driver_signature: str
    canonical_runtime_witness_digest: tuple[str, ...]

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
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.live_covariance_entry_label = str(self.live_covariance_entry_label).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.preserves_left_support_contract = bool(
            self.preserves_left_support_contract
        )
        self.diagonal_preserved = bool(self.diagonal_preserved)
        self.direct_covariance_edits_allowed = bool(
            self.direct_covariance_edits_allowed
        )
        self.required_patch_share_of_full_shared_vf_gap = float(
            self.required_patch_share_of_full_shared_vf_gap
        )
        self.required_patch_share_of_omega_only_shared_vf_increment = float(
            self.required_patch_share_of_omega_only_shared_vf_increment
        )
        self.required_patch_share_of_diagonal_omega_gap = float(
            self.required_patch_share_of_diagonal_omega_gap
        )
        self.required_patch_share_of_psd_boundary = float(
            self.required_patch_share_of_psd_boundary
        )
        self.remaining_psd_headroom = float(self.remaining_psd_headroom)
        self.compensating_stage_order = tuple(
            str(item).strip() for item in self.compensating_stage_order
        )
        self.compensating_cumulative_share_of_total_absolute_mass = tuple(
            float(value)
            for value in self.compensating_cumulative_share_of_total_absolute_mass
        )
        self.compensating_zero_live_entry_count = int(
            self.compensating_zero_live_entry_count
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_runtime_witness_digest = tuple(
            str(line).rstrip() for line in self.canonical_runtime_witness_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract_report(
    *,
    estimator_object_flow_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport
    ),
    shared_entry_budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSharedEntryPatchBudgetReport
    ),
    compensating_stage_ladder_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryStageLadderReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport:
    if (
        estimator_object_flow_report.policy_digest
        != shared_entry_budget_report.policy_digest
    ):
        raise ValueError("runtime witness contract requires shared policy digest")
    if (
        estimator_object_flow_report.policy_digest
        != compensating_stage_ladder_report.policy_digest
    ):
        raise ValueError("runtime witness contract requires shared policy digest")
    if (
        estimator_object_flow_report.binding_design
        != shared_entry_budget_report.binding_design
    ):
        raise ValueError("runtime witness contract requires shared binding design")
    if (
        estimator_object_flow_report.binding_design
        != compensating_stage_ladder_report.binding_design
    ):
        raise ValueError("runtime witness contract requires shared binding design")
    if (
        estimator_object_flow_report.window_label
        != shared_entry_budget_report.window_label
    ):
        raise ValueError("runtime witness contract requires shared window label")
    if (
        estimator_object_flow_report.window_label
        != compensating_stage_ladder_report.window_label
    ):
        raise ValueError("runtime witness contract requires shared window label")
    if (
        estimator_object_flow_report.coverage_anchor_random_state
        != shared_entry_budget_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "runtime witness contract requires shared coverage-anchor seed"
        )
    if (
        estimator_object_flow_report.coverage_anchor_random_state
        != compensating_stage_ladder_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "runtime witness contract requires shared coverage-anchor seed"
        )
    if (
        estimator_object_flow_report.overshoot_companion_random_state
        != shared_entry_budget_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "runtime witness contract requires shared overshoot-companion seed"
        )
    if (
        estimator_object_flow_report.overshoot_companion_random_state
        != compensating_stage_ladder_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "runtime witness contract requires shared overshoot-companion seed"
        )
    if (
        estimator_object_flow_report.source_diagonal_coordinate
        != shared_entry_budget_report.diagonal_coordinate
    ):
        raise ValueError("runtime witness contract requires shared diagonal coordinate")
    if (
        estimator_object_flow_report.source_diagonal_coordinate
        != compensating_stage_ladder_report.diagonal_coordinate
    ):
        raise ValueError("runtime witness contract requires shared diagonal coordinate")
    if (
        estimator_object_flow_report.source_diagonal_basis_label
        != shared_entry_budget_report.diagonal_basis_label
    ):
        raise ValueError(
            "runtime witness contract requires shared diagonal basis label"
        )
    if (
        estimator_object_flow_report.source_diagonal_basis_label
        != compensating_stage_ladder_report.diagonal_basis_label
    ):
        raise ValueError(
            "runtime witness contract requires shared diagonal basis label"
        )
    if (
        estimator_object_flow_report.shared_vf_entry_label
        != shared_entry_budget_report.shared_vf_entry_label
    ):
        raise ValueError("runtime witness contract requires shared vf entry label")

    runtime_witness_path = (
        f"omega_f_hat[{estimator_object_flow_report.source_diagonal_coordinate},{estimator_object_flow_report.source_diagonal_coordinate}]",
        estimator_object_flow_report.shared_vf_entry_label,
        "covariance(0.25, 0.15)",
    )
    direct_covariance_edits_allowed = False
    driver_signature = _driver_signature(
        estimator_object_flow_signature=estimator_object_flow_report.driver_signature,
        shared_entry_budget_signature=shared_entry_budget_report.driver_signature,
        compensating_stage_ladder_signature=(
            compensating_stage_ladder_report.driver_signature
        ),
        source_diagonal_coordinate=(
            estimator_object_flow_report.source_diagonal_coordinate
        ),
        source_diagonal_basis_label=(
            estimator_object_flow_report.source_diagonal_basis_label
        ),
        shared_vf_entry_label=estimator_object_flow_report.shared_vf_entry_label,
        preserves_left_support_contract=(
            estimator_object_flow_report.preserves_left_support_contract
        ),
        diagonal_preserved=estimator_object_flow_report.diagonal_preserved,
        direct_covariance_edits_allowed=direct_covariance_edits_allowed,
        required_patch_share_of_full_shared_vf_gap=(
            shared_entry_budget_report.required_share_of_full_shared_vf_gap
        ),
        required_patch_share_of_omega_only_shared_vf_increment=(
            shared_entry_budget_report.required_share_of_omega_only_shared_vf_increment
        ),
        required_patch_share_of_diagonal_omega_gap=(
            shared_entry_budget_report.required_share_of_diagonal_omega_gap
        ),
        required_patch_share_of_psd_boundary=(
            estimator_object_flow_report.required_patch_share_of_psd_boundary
        ),
        compensating_cumulative_share_after_left_center=(
            compensating_stage_ladder_report.cumulative_share_of_total_absolute_mass[2]
        ),
        compensating_zero_live_entry_count=(
            compensating_stage_ladder_report.zero_live_entry_count
        ),
    )

    canonical_digest = (
        f"- the bounded runtime witness must stay on the nested first-sine path `{runtime_witness_path[0]} -> {runtime_witness_path[1]} -> {runtime_witness_path[2]}`, because the live repair still starts from the positive `{estimator_object_flow_report.source_diagonal_basis_label}` diagonal and keeps left-support preservation plus diagonal invariance as hard obligations",
        f"- that path remains a bounded patch rather than broad replay: the required lift still uses only `{_format_percent(shared_entry_budget_report.required_share_of_full_shared_vf_gap)}` of the full shared `{shared_entry_budget_report.shared_vf_entry_label}` gap, `{_format_percent(shared_entry_budget_report.required_share_of_omega_only_shared_vf_increment)}` of the omega-only shared-entry increment, `{_format_percent(shared_entry_budget_report.required_share_of_diagonal_omega_gap)}` of the positive first-sine diagonal omega gap, and `{_format_percent(estimator_object_flow_report.required_patch_share_of_psd_boundary)}` of the PSD boundary",
        f"- the compensating geometry still has to be consumed in the same diagonal-first order: top-two diagonal cancellation reaches `{_format_percent(compensating_stage_ladder_report.cumulative_share_of_total_absolute_mass[0])}`, cross-shoulder raises cumulative coverage to `{_format_percent(compensating_stage_ladder_report.cumulative_share_of_total_absolute_mass[1])}`, left-center raises it to `{_format_percent(compensating_stage_ladder_report.cumulative_share_of_total_absolute_mass[2])}`, and only then does the `{_format_percent(1.0 - compensating_stage_ladder_report.cumulative_share_of_total_absolute_mass[2])}` left-left residual remain",
        "- current Trigger 2 implication: `first-sine-preserve-left-support-runtime-witness-contract`; future estimator evidence must realize this bounded path without direct covariance edits or dense replay, but the contract itself remains validation-only companion evidence rather than a promoted live routing token",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-preserve-left-support-runtime-witness-contract"
        ),
        policy_digest=estimator_object_flow_report.policy_digest,
        binding_design=estimator_object_flow_report.binding_design,
        window_label=estimator_object_flow_report.window_label,
        coverage_anchor_random_state=(
            estimator_object_flow_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            estimator_object_flow_report.overshoot_companion_random_state
        ),
        source_diagonal_coordinate=(
            estimator_object_flow_report.source_diagonal_coordinate
        ),
        source_diagonal_basis_label=(
            estimator_object_flow_report.source_diagonal_basis_label
        ),
        shared_vf_entry_label=estimator_object_flow_report.shared_vf_entry_label,
        live_covariance_entry_label="covariance(0.25, 0.15)",
        runtime_witness_path=runtime_witness_path,
        preserves_left_support_contract=(
            estimator_object_flow_report.preserves_left_support_contract
        ),
        diagonal_preserved=estimator_object_flow_report.diagonal_preserved,
        direct_covariance_edits_allowed=direct_covariance_edits_allowed,
        required_patch_share_of_full_shared_vf_gap=(
            shared_entry_budget_report.required_share_of_full_shared_vf_gap
        ),
        required_patch_share_of_omega_only_shared_vf_increment=(
            shared_entry_budget_report.required_share_of_omega_only_shared_vf_increment
        ),
        required_patch_share_of_diagonal_omega_gap=(
            shared_entry_budget_report.required_share_of_diagonal_omega_gap
        ),
        required_patch_share_of_psd_boundary=(
            estimator_object_flow_report.required_patch_share_of_psd_boundary
        ),
        remaining_psd_headroom=(estimator_object_flow_report.remaining_psd_headroom),
        compensating_stage_order=(compensating_stage_ladder_report.stage_order),
        compensating_cumulative_share_of_total_absolute_mass=(
            compensating_stage_ladder_report.cumulative_share_of_total_absolute_mass
        ),
        compensating_zero_live_entry_count=(
            compensating_stage_ladder_report.zero_live_entry_count
        ),
        driver_signature=driver_signature,
        canonical_runtime_witness_digest=canonical_digest,
    )


def _build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-preserve-left-support-runtime-witness-contract",
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
        shared_vf_entry_label="v_f_hat[2,2]",
        live_covariance_entry_label="covariance(0.25, 0.15)",
        runtime_witness_path=(
            "omega_f_hat[2,2]",
            "v_f_hat[2,2]",
            "covariance(0.25, 0.15)",
        ),
        preserves_left_support_contract=True,
        diagonal_preserved=True,
        direct_covariance_edits_allowed=False,
        required_patch_share_of_full_shared_vf_gap=0.11972498526885547,
        required_patch_share_of_omega_only_shared_vf_increment=0.17856759637478292,
        required_patch_share_of_diagonal_omega_gap=0.24167652126504635,
        required_patch_share_of_psd_boundary=0.12714564275399315,
        remaining_psd_headroom=11.23200779042753,
        compensating_stage_order=(
            "top-two-diagonal",
            "cross-shoulder-pair",
            "left-center-pair",
            "left-left-residual",
        ),
        compensating_cumulative_share_of_total_absolute_mass=(
            0.5768789713362612,
            0.7923694401292339,
            0.9667048915085733,
            1.0,
        ),
        compensating_zero_live_entry_count=2,
        driver_signature="first-sine-preserve-left-support-runtime-witness-contract",
        canonical_runtime_witness_digest=(
            "- the bounded runtime witness must stay on the nested first-sine path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`, because the live repair still starts from the positive `sin(2πz)` diagonal and keeps left-support preservation plus diagonal invariance as hard obligations",
            "- that path remains a bounded patch rather than broad replay: the required lift still uses only `12.0%` of the full shared `v_f_hat[2,2]` gap, `17.9%` of the omega-only shared-entry increment, `24.2%` of the positive first-sine diagonal omega gap, and `12.7%` of the PSD boundary",
            "- the compensating geometry still has to be consumed in the same diagonal-first order: top-two diagonal cancellation reaches `57.7%`, cross-shoulder raises cumulative coverage to `79.2%`, left-center raises it to `96.7%`, and only then does the `3.3%` left-left residual remain",
            "- current Trigger 2 implication: `first-sine-preserve-left-support-runtime-witness-contract`; future estimator evidence must realize this bounded path without direct covariance edits or dense replay, but the contract itself remains validation-only companion evidence rather than a promoted live routing token",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
):
    return _build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract_snapshot_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract",
]
