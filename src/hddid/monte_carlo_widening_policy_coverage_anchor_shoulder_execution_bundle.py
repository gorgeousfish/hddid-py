from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFreshEstimatorRerunIntakeContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairTargetSnapshotReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot,
)
from .monte_carlo_widening_policy_acceptance_preview import (
    build_phase7_monte_carlo_widening_policy_acceptance_preview_repo_side_report,
)
from .monte_carlo_widening_policy_quality_risk_probe import (
    Phase7MonteCarloWideningPolicyQualityRiskDesignSummary,
    Phase7MonteCarloWideningPolicyQualityRiskProbeReport,
    _build_repo_side_quality_risk_probe_report,
    run_phase7_monte_carlo_widening_policy_quality_risk_probe,
)
from .monte_carlo_widening_policy_runtime_evidence_packet import (
    Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport,
    build_phase7_monte_carlo_widening_policy_runtime_evidence_packet_repo_side_report,
    run_phase7_monte_carlo_widening_policy_runtime_evidence_packet,
)
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOMATION_STATE_PATH = _REPO_ROOT / "Docs" / "automation" / "automation-state.yaml"
_LIVE_SOURCE_TARGET_ALIGNMENT_HELPER = (
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe()"
)
_LIVE_SOURCE_TARGET_ALIGNMENT_NOTE = (
    "Docs/research/phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe.md"
)
_LIVE_SOURCE_TARGET_ALIGNMENT_TARGET_GAP_FIELD = "canonical_vs_live_target_gap"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _format_design_key(binding_design: tuple[str, int, int]) -> str:
    return f"{binding_design[0]}/{binding_design[1]}/{binding_design[2]}"


def _parse_design_key(design_label: str) -> tuple[str, int, int]:
    dgp_name, n_obs, p = str(design_label).split("/")
    return (dgp_name, int(n_obs), int(p))


def _source_diagonal_lane(
    *,
    source_diagonal_coordinate: int,
    source_diagonal_basis_label: str,
    shared_vf_entry_label: str,
) -> tuple[int, str, str]:
    return (
        int(source_diagonal_coordinate),
        str(source_diagonal_basis_label).strip(),
        str(shared_vf_entry_label).strip(),
    )


def _load_trigger2_gate_state() -> dict[str, object]:
    state = yaml.safe_load(_AUTOMATION_STATE_PATH.read_text(encoding="utf-8"))
    return state["feature_completion_gate"]["checks"]["monte_carlo_validation_ready"]


def _repo_side_quality_risk_probe_from_gate_state() -> (
    Phase7MonteCarloWideningPolicyQualityRiskProbeReport
):
    return _build_repo_side_quality_risk_probe_report(
        policy=build_phase7_canonical_monte_carlo_widening_policy()
    )


def _repo_side_quality_risk_probe_for_execution_bundle(
    live_report: Phase7MonteCarloWideningPolicyQualityRiskProbeReport,
) -> Phase7MonteCarloWideningPolicyQualityRiskProbeReport:
    return live_report


def _repo_side_runtime_evidence_packet_for_execution_bundle() -> (
    Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport
):
    acceptance_preview = (
        build_phase7_monte_carlo_widening_policy_acceptance_preview_repo_side_report(
            _REPO_ROOT
        )
    )
    return build_phase7_monte_carlo_widening_policy_runtime_evidence_packet_repo_side_report(
        acceptance_preview
    )


def _repo_side_execution_contract_for_execution_bundle() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
):
    gate_state = _load_trigger2_gate_state()
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-execution-contract"
        ),
        policy_digest=tuple(str(item) for item in gate_state["policy_digest"]),
        binding_design=_parse_design_key(str(gate_state["runtime_evidence_binding_design"])),
        window_label="near_zero_grid",
        coverage_anchor_random_state=202,
        overshoot_companion_random_state=505,
        left_shoulder_grid_value=0.05,
        center_grid_value=0.15,
        failing_right_shoulder_grid_value=0.25,
        coverage_anchor_local_scale_access_shortfall_share=0.127,
        current_anchor_correlation=0.00724926035214676,
        required_repaired_correlation=0.13276252551993098,
        correlation_repair_increment=0.12551326516778422,
        required_gap_share=0.09,
        anchor_left_shoulder_reserve=1.0,
        anchor_failing_right_shoulder_reserve=-1.0,
        anchor_right_to_left_covariance_share=0.0,
        companion_right_to_left_covariance_share=0.0,
        directional_flip_ratio=1.0,
        current_abs_right_center_covariance=0.0,
        required_abs_right_center_covariance=1.6361273081544214,
        required_incremental_right_center_covariance_lift=1.6361273081544214,
        required_increment_share_of_entry_gap=0.09,
        required_increment_share_of_right_row_gap=0.09,
        required_increment_share_of_center_column_gap=0.09,
        residual_entry_gap_share_after_required_increment=0.91,
        zero_partial_right_center_correlation=-2.0,
        zero_partial_lower_bound_gap=1.0,
        required_repair_distance_to_zero_partial_target=1.0,
        shoulder_sigma_log_gap_share=0.0,
        center_sigma_log_gap_share=0.0,
        correlation_log_gap_share=1.0,
        correlation_vs_shoulder_sigma_ratio=1.0,
        correlation_vs_center_sigma_ratio=1.0,
        driver_signature="bounded-right-center-execution-contract",
        canonical_execution_contract_digest=(
            "- repo-side bounded right-center execution contract from feature-completion gate",
        ),
    )


def _repo_side_repair_target_for_execution_bundle(
    execution_contract: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairTargetSnapshotReport:
    gate_state = _load_trigger2_gate_state()
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairTargetSnapshotReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-repair-target-snapshot"
        ),
        policy_digest=execution_contract.policy_digest,
        binding_design=execution_contract.binding_design,
        binding_driver=str(gate_state["quality_risk_binding_driver"]),
        seed_dispersion_driver=str(gate_state["seed_dispersion_driver"]),
        seed_role_split_label="coverage-anchor-vs-overshoot-companion",
        coverage_anchor_random_state=execution_contract.coverage_anchor_random_state,
        overshoot_companion_random_state=execution_contract.overshoot_companion_random_state,
        left_shoulder_grid_value=execution_contract.left_shoulder_grid_value,
        center_grid_value=execution_contract.center_grid_value,
        failing_right_shoulder_grid_value=execution_contract.failing_right_shoulder_grid_value,
        local_scale_access_shortfall_share=(
            execution_contract.coverage_anchor_local_scale_access_shortfall_share
        ),
        anchor_right_to_left_center_access_share=(
            execution_contract.anchor_right_to_left_covariance_share
        ),
        companion_right_to_left_center_access_share=(
            execution_contract.companion_right_to_left_covariance_share
        ),
        directional_flip_ratio=execution_contract.directional_flip_ratio,
        anchor_partial_cross_shoulder_correlation=0.0,
        companion_partial_cross_shoulder_correlation=0.0,
        anchor_center_mediated_share_of_abs_cross_shoulder=0.0,
        required_repaired_correlation=execution_contract.required_repaired_correlation,
        correlation_repair_increment=execution_contract.correlation_repair_increment,
        required_gap_share=execution_contract.required_gap_share,
        correlation_log_gap_share=execution_contract.correlation_log_gap_share,
        repair_target_signature=str(gate_state["repair_target_signature"]),
        calibration_debt_snapshot_digest=(),
        directionality_digest=(),
        direct_residual_digest=(),
        canonical_repair_target_snapshot_digest=(
            "- repo-side directional direct residual correlation repair target",
        ),
    )


def _repo_side_implementation_readiness_for_execution_bundle(
    execution_contract: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport:
    report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_probe()
    )
    if report.policy_digest != execution_contract.policy_digest:
        raise ValueError("execution bundle readiness policy drifted")
    if report.binding_design != execution_contract.binding_design:
        raise ValueError("execution bundle readiness binding design drifted")
    if report.window_label != execution_contract.window_label:
        raise ValueError("execution bundle readiness window label drifted")
    if (
        report.coverage_anchor_random_state
        != execution_contract.coverage_anchor_random_state
    ):
        raise ValueError("execution bundle readiness coverage-anchor seed drifted")
    if (
        report.overshoot_companion_random_state
        != execution_contract.overshoot_companion_random_state
    ):
        raise ValueError("execution bundle readiness overshoot seed drifted")
    if report.live_driver_signature != execution_contract.driver_signature:
        raise ValueError("execution bundle readiness live driver drifted")
    return report


def _repo_side_fresh_estimator_rerun_intake_for_execution_bundle(
    *,
    execution_contract: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
    implementation_readiness: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFreshEstimatorRerunIntakeContractReport:
    gate_state = _load_trigger2_gate_state()
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFreshEstimatorRerunIntakeContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-fresh-estimator-rerun-intake-contract"
        ),
        policy_digest=execution_contract.policy_digest,
        binding_design=execution_contract.binding_design,
        window_label=execution_contract.window_label,
        coverage_anchor_random_state=execution_contract.coverage_anchor_random_state,
        overshoot_companion_random_state=execution_contract.overshoot_companion_random_state,
        live_driver_signature=execution_contract.driver_signature,
        spend_order_driver_signature="runtime-witness-rerun-spend-order-disciplined",
        intake_driver_signature=str(gate_state["runtime_evidence_fresh_estimator_evidence_driver"]),
        runtime_witness_path=(
            "omega_f_hat[2,2]",
            implementation_readiness.shared_vf_entry_label,
            "covariance(0.25, 0.15)",
        ),
        source_diagonal_coordinate=implementation_readiness.source_diagonal_coordinate,
        source_diagonal_basis_label=implementation_readiness.source_diagonal_basis_label,
        shared_vf_entry_label=implementation_readiness.shared_vf_entry_label,
        binding_budget_label="random-state-budget",
        binding_budget_share=1.0,
        source_required_omega_diagonal_increment=(
            implementation_readiness.source_required_omega_diagonal_increment
        ),
        required_diagonal_vf_entry_lift=1011.1822863613054,
        validation_patch_increment=implementation_readiness.required_patch_increment,
        patch_frobenius_share_of_baseline=0.0,
        patch_spectral_share_of_baseline=0.0,
        companion_right_row_replay_multiple_vs_required_increment=0.0,
        companion_center_column_replay_multiple_vs_required_increment=0.0,
        effective_fresh_reruns=int(gate_state["runtime_evidence_effective_fresh_reruns"]),
        supported_floor_ceiling=7.0 / 9.0,
        canonical_floor=0.85,
        canonical_floor_shortfall=0.85 - (7.0 / 9.0),
        required_evidence_conditions=(
            "keep `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` as the bounded object flow",
            "reject whole-row / whole-column / dense covariance replay",
            "treat fresh reruns as informative only when they lift the binding floor witness",
        ),
        spend_order_steps=(
            "reground-live-source-target-alignment",
            "spend-fresh-reruns-only-on-floor-lift-evidence",
        ),
        contract_holds=True,
        driver_signature=str(gate_state["runtime_evidence_fresh_estimator_rerun_driver"]),
        current_implication="spend-fresh-reruns-only-after-bounded-estimator-intake-and-floor-lift-evidence",
        canonical_fresh_estimator_rerun_intake_digest=(
            "- repo-side fresh estimator rerun intake contract",
        ),
    )


def _repo_side_live_source_target_alignment_for_execution_bundle(
    *,
    execution_contract: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
    implementation_readiness: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport
    ),
    fresh_estimator_rerun_intake: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFreshEstimatorRerunIntakeContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport:
    gate_state = _load_trigger2_gate_state()
    report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe()
    )
    if report.policy_digest != tuple(str(item) for item in gate_state["policy_digest"]):
        raise ValueError("execution bundle live source-target policy drifted")
    if report.binding_design != execution_contract.binding_design:
        raise ValueError("execution bundle live source-target binding design drifted")
    if report.window_label != execution_contract.window_label:
        raise ValueError("execution bundle live source-target window label drifted")
    if (
        report.coverage_anchor_random_state
        != execution_contract.coverage_anchor_random_state
    ):
        raise ValueError("execution bundle live source-target coverage-anchor seed drifted")
    if (
        report.overshoot_companion_random_state
        != execution_contract.overshoot_companion_random_state
    ):
        raise ValueError("execution bundle live source-target overshoot seed drifted")
    if (
        report.source_diagonal_coordinate
        != implementation_readiness.source_diagonal_coordinate
    ):
        raise ValueError("execution bundle live source-target source coordinate drifted")
    if (
        report.source_diagonal_basis_label
        != implementation_readiness.source_diagonal_basis_label
    ):
        raise ValueError("execution bundle live source-target source basis drifted")
    if report.shared_vf_entry_label != implementation_readiness.shared_vf_entry_label:
        raise ValueError("execution bundle live source-target shared object drifted")
    if (
        report.required_diagonal_vf_entry_lift
        != fresh_estimator_rerun_intake.required_diagonal_vf_entry_lift
    ):
        raise ValueError("execution bundle live source-target diagonal lift drifted")
    return report


def _driver_signature(
    *,
    quality_risk_probe: Phase7MonteCarloWideningPolicyQualityRiskProbeReport,
    repair_target_snapshot: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairTargetSnapshotReport
    ),
    execution_contract: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
    implementation_readiness: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport
    ),
    fresh_estimator_rerun_intake: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFreshEstimatorRerunIntakeContractReport
    ),
    runtime_evidence_packet: Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport,
    live_source_target_alignment: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport
        | None
    ),
) -> str:
    if (
        quality_risk_probe.binding_driver
        in {
            "rmse-outpaces-average-se",
            "interval-per-rmse-headroom-trails-best-design",
        }
        and repair_target_snapshot.repair_target_signature
        == "directional-direct-residual-correlation-repair-target"
        and execution_contract.driver_signature
        == "bounded-right-center-execution-contract"
        and implementation_readiness.driver_signature
        == "bounded-right-center-entry-patch-implementation-readiness"
        and fresh_estimator_rerun_intake.driver_signature
        == "fresh-estimator-rerun-intake-contract-disciplined"
        and runtime_evidence_packet.driver_signature
        == "trigger2-runtime-evidence-packet-open"
        and (
            live_source_target_alignment is None
            or live_source_target_alignment.driver_signature
            == "live-source-target-alignment-drift"
        )
    ):
        return "bounded-right-center-execution-bundle"
    return "mixed-right-center-execution-bundle"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionBundleReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    quality_risk_binding_driver: str
    repair_target_signature: str
    live_driver_signature: str
    readiness_driver_signature: str
    fresh_estimator_driver_signature: str
    runtime_evidence_driver: str
    runtime_evidence_route_label: str
    live_source_target_alignment_driver: str
    live_source_target_alignment_target_gap: float
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    shared_vf_entry_label: str
    required_repaired_correlation: float
    correlation_repair_increment: float
    required_diagonal_vf_entry_lift: float
    required_patch_increment: float
    required_patch_share_of_psd_boundary: float
    remaining_psd_headroom_multiple_of_required_patch: float
    effective_fresh_reruns: int
    supported_floor_ceiling: float
    canonical_floor: float
    canonical_floor_shortfall: float
    runtime_witness_path: tuple[str, ...]
    prohibited_actions: tuple[str, ...]
    required_evidence_conditions: tuple[str, ...]
    recommended_execution_steps: tuple[str, ...]
    driver_signature: str
    canonical_execution_bundle_digest: tuple[str, ...]
    quality_risk_source_mode: str = "live-runtime-probe"
    quality_risk_source_note: str = "live runtime probe"

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
        self.quality_risk_binding_driver = str(self.quality_risk_binding_driver).strip()
        self.repair_target_signature = str(self.repair_target_signature).strip()
        self.live_driver_signature = str(self.live_driver_signature).strip()
        self.readiness_driver_signature = str(self.readiness_driver_signature).strip()
        self.fresh_estimator_driver_signature = str(
            self.fresh_estimator_driver_signature
        ).strip()
        self.runtime_evidence_driver = str(self.runtime_evidence_driver).strip()
        self.runtime_evidence_route_label = str(self.runtime_evidence_route_label).strip()
        self.live_source_target_alignment_driver = str(
            self.live_source_target_alignment_driver
        ).strip()
        self.live_source_target_alignment_target_gap = float(
            self.live_source_target_alignment_target_gap
        )
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.required_repaired_correlation = float(self.required_repaired_correlation)
        self.correlation_repair_increment = float(self.correlation_repair_increment)
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.required_patch_increment = float(self.required_patch_increment)
        self.required_patch_share_of_psd_boundary = float(
            self.required_patch_share_of_psd_boundary
        )
        self.remaining_psd_headroom_multiple_of_required_patch = float(
            self.remaining_psd_headroom_multiple_of_required_patch
        )
        self.effective_fresh_reruns = int(self.effective_fresh_reruns)
        self.supported_floor_ceiling = float(self.supported_floor_ceiling)
        self.canonical_floor = float(self.canonical_floor)
        self.canonical_floor_shortfall = float(self.canonical_floor_shortfall)
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.prohibited_actions = tuple(
            str(item).strip() for item in self.prohibited_actions
        )
        self.required_evidence_conditions = tuple(
            str(item).strip() for item in self.required_evidence_conditions
        )
        self.recommended_execution_steps = tuple(
            str(item).strip() for item in self.recommended_execution_steps
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_execution_bundle_digest = tuple(
            str(line).rstrip() for line in self.canonical_execution_bundle_digest
        )
        self.quality_risk_source_mode = str(self.quality_risk_source_mode).strip()
        self.quality_risk_source_note = str(self.quality_risk_source_note).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "quality_risk_binding_driver": self.quality_risk_binding_driver,
            "repair_target_signature": self.repair_target_signature,
            "live_driver_signature": self.live_driver_signature,
            "readiness_driver_signature": self.readiness_driver_signature,
            "fresh_estimator_driver_signature": (
                self.fresh_estimator_driver_signature
            ),
            "runtime_evidence_driver": self.runtime_evidence_driver,
            "runtime_evidence_route_label": self.runtime_evidence_route_label,
            "live_source_target_alignment_driver": (
                self.live_source_target_alignment_driver
            ),
            "live_source_target_alignment_target_gap": (
                self.live_source_target_alignment_target_gap
            ),
            "source_diagonal_coordinate": self.source_diagonal_coordinate,
            "source_diagonal_basis_label": self.source_diagonal_basis_label,
            "shared_vf_entry_label": self.shared_vf_entry_label,
            "required_repaired_correlation": self.required_repaired_correlation,
            "correlation_repair_increment": self.correlation_repair_increment,
            "required_diagonal_vf_entry_lift": self.required_diagonal_vf_entry_lift,
            "required_patch_increment": self.required_patch_increment,
            "required_patch_share_of_psd_boundary": (
                self.required_patch_share_of_psd_boundary
            ),
            "remaining_psd_headroom_multiple_of_required_patch": (
                self.remaining_psd_headroom_multiple_of_required_patch
            ),
            "effective_fresh_reruns": self.effective_fresh_reruns,
            "supported_floor_ceiling": self.supported_floor_ceiling,
            "canonical_floor": self.canonical_floor,
            "canonical_floor_shortfall": self.canonical_floor_shortfall,
            "runtime_witness_path": list(self.runtime_witness_path),
            "prohibited_actions": list(self.prohibited_actions),
            "required_evidence_conditions": list(self.required_evidence_conditions),
            "recommended_execution_steps": list(self.recommended_execution_steps),
            "driver_signature": self.driver_signature,
            "canonical_execution_bundle_digest": list(
                self.canonical_execution_bundle_digest
            ),
            "quality_risk_source_mode": self.quality_risk_source_mode,
            "quality_risk_source_note": self.quality_risk_source_note,
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_bundle_report(
    *,
    quality_risk_probe: Phase7MonteCarloWideningPolicyQualityRiskProbeReport,
    repair_target_snapshot: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairTargetSnapshotReport
    ),
    execution_contract: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
    implementation_readiness: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport
    ),
    fresh_estimator_rerun_intake: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFreshEstimatorRerunIntakeContractReport
    ),
    runtime_evidence_packet: Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport,
    live_source_target_alignment: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionBundleReport:
    quality_risk_probe = _repo_side_quality_risk_probe_for_execution_bundle(
        quality_risk_probe
    )
    if quality_risk_probe.policy_digest != repair_target_snapshot.policy_digest:
        raise ValueError("execution bundle requires shared policy digest")
    if quality_risk_probe.policy_digest != execution_contract.policy_digest:
        raise ValueError("execution bundle requires shared policy digest")
    if quality_risk_probe.policy_digest != implementation_readiness.policy_digest:
        raise ValueError("execution bundle requires shared policy digest")
    if quality_risk_probe.policy_digest != fresh_estimator_rerun_intake.policy_digest:
        raise ValueError("execution bundle requires shared policy digest")
    if quality_risk_probe.policy_digest != runtime_evidence_packet.policy_digest:
        raise ValueError("execution bundle requires shared policy digest")
    for report_binding_design in (
        repair_target_snapshot.binding_design,
        implementation_readiness.binding_design,
        fresh_estimator_rerun_intake.binding_design,
        runtime_evidence_packet.binding_design,
    ):
        if report_binding_design != execution_contract.binding_design:
            raise ValueError("execution bundle requires shared binding design")
    if execution_contract.window_label != implementation_readiness.window_label:
        raise ValueError("execution bundle requires shared window label")
    if execution_contract.window_label != fresh_estimator_rerun_intake.window_label:
        raise ValueError("execution bundle requires shared window label")
    if (
        execution_contract.coverage_anchor_random_state
        != repair_target_snapshot.coverage_anchor_random_state
        or execution_contract.coverage_anchor_random_state
        != implementation_readiness.coverage_anchor_random_state
        or execution_contract.coverage_anchor_random_state
        != fresh_estimator_rerun_intake.coverage_anchor_random_state
    ):
        raise ValueError("execution bundle requires shared coverage-anchor seed")
    if (
        execution_contract.overshoot_companion_random_state
        != repair_target_snapshot.overshoot_companion_random_state
        or execution_contract.overshoot_companion_random_state
        != implementation_readiness.overshoot_companion_random_state
        or execution_contract.overshoot_companion_random_state
        != fresh_estimator_rerun_intake.overshoot_companion_random_state
    ):
        raise ValueError("execution bundle requires shared overshoot-companion seed")
    if fresh_estimator_rerun_intake.runtime_witness_path != (
        "omega_f_hat[2,2]",
        "v_f_hat[2,2]",
        "covariance(0.25, 0.15)",
    ):
        raise ValueError("execution bundle requires the canonical bounded runtime path")
    if runtime_evidence_packet.runtime_witness_path != (
        "omega_f_hat[2,2]",
        "v_f_hat[2,2]",
        "covariance(0.25, 0.15)",
    ):
        raise ValueError(
            "execution bundle requires the runtime-evidence packet to preserve the canonical bounded runtime path"
        )
    canonical_source_diagonal_lane = _source_diagonal_lane(
        source_diagonal_coordinate=implementation_readiness.source_diagonal_coordinate,
        source_diagonal_basis_label=implementation_readiness.source_diagonal_basis_label,
        shared_vf_entry_label=implementation_readiness.shared_vf_entry_label,
    )
    if canonical_source_diagonal_lane != _source_diagonal_lane(
        source_diagonal_coordinate=fresh_estimator_rerun_intake.source_diagonal_coordinate,
        source_diagonal_basis_label=fresh_estimator_rerun_intake.source_diagonal_basis_label,
        shared_vf_entry_label=fresh_estimator_rerun_intake.shared_vf_entry_label,
    ):
        raise ValueError(
            "execution bundle requires fresh-estimator rerun intake to preserve the canonical source diagonal lane"
        )
    if live_source_target_alignment is not None:
        if quality_risk_probe.policy_digest != live_source_target_alignment.policy_digest:
            raise ValueError("execution bundle requires shared policy digest")
        if (
            execution_contract.binding_design
            != live_source_target_alignment.binding_design
        ):
            raise ValueError(
                "execution bundle requires live source-target alignment on the binding design"
            )
        if execution_contract.coverage_anchor_random_state != (
            live_source_target_alignment.coverage_anchor_random_state
        ):
            raise ValueError(
                "execution bundle requires live source-target alignment on the anchor seed"
            )
        if execution_contract.overshoot_companion_random_state != (
            live_source_target_alignment.overshoot_companion_random_state
        ):
            raise ValueError(
                "execution bundle requires live source-target alignment on the companion seed"
            )
        if canonical_source_diagonal_lane != _source_diagonal_lane(
            source_diagonal_coordinate=live_source_target_alignment.source_diagonal_coordinate,
            source_diagonal_basis_label=live_source_target_alignment.source_diagonal_basis_label,
            shared_vf_entry_label=live_source_target_alignment.shared_vf_entry_label,
        ):
            raise ValueError(
                "execution bundle requires live source-target alignment to preserve the canonical source diagonal lane"
            )

    driver_signature = _driver_signature(
        quality_risk_probe=quality_risk_probe,
        repair_target_snapshot=repair_target_snapshot,
        execution_contract=execution_contract,
        implementation_readiness=implementation_readiness,
        fresh_estimator_rerun_intake=fresh_estimator_rerun_intake,
        runtime_evidence_packet=runtime_evidence_packet,
        live_source_target_alignment=live_source_target_alignment,
    )
    binding_design_key = _format_design_key(execution_contract.binding_design)
    live_source_target_alignment_driver = (
        "not-evaluated"
        if live_source_target_alignment is None
        else live_source_target_alignment.driver_signature
    )
    live_source_target_alignment_target_gap = (
        0.0
        if live_source_target_alignment is None
        else live_source_target_alignment.canonical_vs_live_target_gap
    )
    live_source_target_alignment_step = (
        ("reground-live-source-target-alignment",)
        if live_source_target_alignment is not None
        and live_source_target_alignment.driver_signature
        == "live-source-target-alignment-drift"
        else ()
    )
    live_source_target_alignment_digest = (
        (
            "- live source-target alignment must be regrounded before replay: "
            f"`{live_source_target_alignment.driver_signature}` still keeps the "
            "archived positive first-sine ceiling "
            f"`{_format_float(live_source_target_alignment.canonical_target_omega_diagonal_entry)}` "
            "out of the live path because the current worktree target sits at "
            f"`{_format_float(live_source_target_alignment.live_target_omega_diagonal_entry)}` "
            f"(`{_format_signed_float(live_source_target_alignment.canonical_vs_live_target_gap)}` gap), "
            "so the bounded replay cannot spend fresh reruns until the seed-202/505 "
            "diagonal target is re-grounded on the same source lane"
        ),
    ) if live_source_target_alignment is not None else ()
    canonical_digest = (
        f"- live Trigger 2 blocker remains quality-bound on `{binding_design_key}` / `{execution_contract.window_label}`: quality-risk driver `{quality_risk_probe.binding_driver}` still narrows the source-level fix to `{repair_target_snapshot.repair_target_signature}`, so execution stays on `{execution_contract.driver_signature}` rather than widening beyond the bounded right-center lane",
        *live_source_target_alignment_digest,
        f"- bounded implementation intake is still compressed into one execution bundle after that regrounding step: coordinate `{implementation_readiness.source_diagonal_coordinate} = {implementation_readiness.source_diagonal_basis_label}` routes `omega_f_hat[2,2] -> {implementation_readiness.shared_vf_entry_label} -> covariance(0.25, 0.15)`, with repaired correlation `{_format_float(execution_contract.required_repaired_correlation)}`, correlation increment `{_format_signed_float(execution_contract.correlation_repair_increment)}`, diagonal vf-entry lift `{_format_signed_float(fresh_estimator_rerun_intake.required_diagonal_vf_entry_lift)}`, raw covariance lift `{_format_signed_float(implementation_readiness.required_patch_increment)}`, PSD budget share `{_format_percent(implementation_readiness.required_patch_share_of_psd_boundary)}`, and `{_format_ratio(implementation_readiness.remaining_psd_headroom_multiple_of_required_patch)}` remaining headroom",
        f"- fresh runtime evidence is still scarce and disciplined: open trigger `{runtime_evidence_packet.route_label}` keeps driver `{runtime_evidence_packet.driver_signature}`, only `{fresh_estimator_rerun_intake.effective_fresh_reruns}` fresh reruns remain, and canonical floor `{_format_float(fresh_estimator_rerun_intake.canonical_floor)}` still sits `{_format_float(fresh_estimator_rerun_intake.canonical_floor_shortfall)}` above supported ceiling `{_format_float(fresh_estimator_rerun_intake.supported_floor_ceiling)}`, so reruns are informative only when they preserve the bounded object flow and lift the binding floor witness",
        f"- current Trigger 2 implication: `{driver_signature}`; next honest main-exec spend first regrounds the live seed-202/505 source target and only then replays the bounded execution contract through the regrounded positive first-sine diagonal while rejecting direct covariance overwrite, coordinate-`2` off-diagonal fallback, whole-row / whole-column replay, and dense widening detours",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionBundleReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-execution-bundle"
        ),
        policy_digest=quality_risk_probe.policy_digest,
        binding_design=execution_contract.binding_design,
        window_label=execution_contract.window_label,
        coverage_anchor_random_state=execution_contract.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            execution_contract.overshoot_companion_random_state
        ),
        quality_risk_binding_driver=quality_risk_probe.binding_driver,
        repair_target_signature=repair_target_snapshot.repair_target_signature,
        live_driver_signature=execution_contract.driver_signature,
        readiness_driver_signature=implementation_readiness.driver_signature,
        fresh_estimator_driver_signature=(
            fresh_estimator_rerun_intake.driver_signature
        ),
        runtime_evidence_driver=runtime_evidence_packet.driver_signature,
        runtime_evidence_route_label=runtime_evidence_packet.route_label,
        quality_risk_source_mode=quality_risk_probe.quality_risk_source_mode,
        quality_risk_source_note=quality_risk_probe.quality_risk_source_note,
        live_source_target_alignment_driver=live_source_target_alignment_driver,
        live_source_target_alignment_target_gap=live_source_target_alignment_target_gap,
        source_diagonal_coordinate=implementation_readiness.source_diagonal_coordinate,
        source_diagonal_basis_label=implementation_readiness.source_diagonal_basis_label,
        shared_vf_entry_label=implementation_readiness.shared_vf_entry_label,
        required_repaired_correlation=execution_contract.required_repaired_correlation,
        correlation_repair_increment=execution_contract.correlation_repair_increment,
        required_diagonal_vf_entry_lift=(
            fresh_estimator_rerun_intake.required_diagonal_vf_entry_lift
        ),
        required_patch_increment=implementation_readiness.required_patch_increment,
        required_patch_share_of_psd_boundary=(
            implementation_readiness.required_patch_share_of_psd_boundary
        ),
        remaining_psd_headroom_multiple_of_required_patch=(
            implementation_readiness.remaining_psd_headroom_multiple_of_required_patch
        ),
        effective_fresh_reruns=fresh_estimator_rerun_intake.effective_fresh_reruns,
        supported_floor_ceiling=fresh_estimator_rerun_intake.supported_floor_ceiling,
        canonical_floor=fresh_estimator_rerun_intake.canonical_floor,
        canonical_floor_shortfall=fresh_estimator_rerun_intake.canonical_floor_shortfall,
        runtime_witness_path=fresh_estimator_rerun_intake.runtime_witness_path,
        prohibited_actions=implementation_readiness.prohibited_actions,
        required_evidence_conditions=(
            fresh_estimator_rerun_intake.required_evidence_conditions
        ),
        recommended_execution_steps=(
            *live_source_target_alignment_step,
            "replay-bounded-right-center-execution-contract",
            "enter-through-regrounded-positive-first-sine-diagonal",
            "preserve-left-support-and-diagonal-invariance",
            "spend-fresh-reruns-only-on-floor-lift-evidence",
        ),
        driver_signature=driver_signature,
        canonical_execution_bundle_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_bundle() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionBundleReport
):
    gate_state = _load_trigger2_gate_state()
    if (
        str(gate_state["live_source_target_alignment_helper"])
        != _LIVE_SOURCE_TARGET_ALIGNMENT_HELPER
    ):
        raise ValueError(
            "execution bundle live source-target alignment helper drifted from gate state"
        )
    if (
        str(gate_state["live_source_target_alignment_note"])
        != _LIVE_SOURCE_TARGET_ALIGNMENT_NOTE
    ):
        raise ValueError(
            "execution bundle live source-target alignment note drifted from gate state"
        )
    if (
        str(gate_state["live_source_target_alignment_target_gap_field"])
        != _LIVE_SOURCE_TARGET_ALIGNMENT_TARGET_GAP_FIELD
    ):
        raise ValueError(
            "execution bundle live source-target alignment target gap field drifted from gate state"
        )
    execution_contract = _repo_side_execution_contract_for_execution_bundle()
    implementation_readiness = _repo_side_implementation_readiness_for_execution_bundle(
        execution_contract
    )
    fresh_estimator_rerun_intake = _repo_side_fresh_estimator_rerun_intake_for_execution_bundle(
        execution_contract=execution_contract,
        implementation_readiness=implementation_readiness,
    )
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_bundle_report(
        quality_risk_probe=_repo_side_quality_risk_probe_from_gate_state(),
        repair_target_snapshot=_repo_side_repair_target_for_execution_bundle(
            execution_contract
        ),
        execution_contract=execution_contract,
        implementation_readiness=implementation_readiness,
        fresh_estimator_rerun_intake=fresh_estimator_rerun_intake,
        runtime_evidence_packet=_repo_side_runtime_evidence_packet_for_execution_bundle(),
        live_source_target_alignment=_repo_side_live_source_target_alignment_for_execution_bundle(
            execution_contract=execution_contract,
            implementation_readiness=implementation_readiness,
            fresh_estimator_rerun_intake=fresh_estimator_rerun_intake,
        ),
    )
