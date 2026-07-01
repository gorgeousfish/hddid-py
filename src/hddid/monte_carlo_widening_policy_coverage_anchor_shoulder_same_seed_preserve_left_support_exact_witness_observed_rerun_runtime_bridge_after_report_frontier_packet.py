from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_binding_slot_priority_bridge import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_binding_slot_priority_bridge,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportFrontierPacketReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    admission_order_driver_signature: str
    runtime_bridge_after_report_intake_driver_signature: str
    runtime_bridge_source_guard_driver_signature: str
    live_gap_packet_driver_signature: str
    binding_slot_priority_bridge_driver_signature: str
    after_report_supplied: bool
    intake_disposition: str
    binding_only_payload_rejected: bool
    completion_after_report_accepted: bool
    queued_residual_slot_still_live: bool
    current_runtime_bridge_state: str
    current_rung_status: str
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    acceptance_shortfall: float
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_index: int
    binding_slot_z_value: float
    binding_slot_repair_stage: str
    binding_slot_repair_role: str
    binding_slot_repair_priority: int
    residual_slot_random_state: int
    residual_slot_seed_group: str
    residual_slot_z_index: int
    residual_slot_z_value: float
    residual_slot_repair_stage: str
    residual_slot_repair_role: str
    residual_slot_repair_priority: int
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    exact_trim_floor_value: float
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    binding_replication_seed: int
    driver_signature: str
    canonical_runtime_bridge_after_report_frontier_packet_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.same_seed_random_states = tuple(
            int(value) for value in self.same_seed_random_states
        )
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.runtime_bridge_after_report_intake_driver_signature = str(
            self.runtime_bridge_after_report_intake_driver_signature
        ).strip()
        self.runtime_bridge_source_guard_driver_signature = str(
            self.runtime_bridge_source_guard_driver_signature
        ).strip()
        self.live_gap_packet_driver_signature = str(
            self.live_gap_packet_driver_signature
        ).strip()
        self.binding_slot_priority_bridge_driver_signature = str(
            self.binding_slot_priority_bridge_driver_signature
        ).strip()
        self.after_report_supplied = bool(self.after_report_supplied)
        self.intake_disposition = str(self.intake_disposition).strip()
        self.binding_only_payload_rejected = bool(self.binding_only_payload_rejected)
        self.completion_after_report_accepted = bool(
            self.completion_after_report_accepted
        )
        self.queued_residual_slot_still_live = bool(
            self.queued_residual_slot_still_live
        )
        self.current_runtime_bridge_state = str(
            self.current_runtime_bridge_state
        ).strip()
        self.current_rung_status = str(self.current_rung_status).strip()
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.acceptance_shortfall = float(self.acceptance_shortfall)
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_seed_group = str(self.binding_slot_seed_group).strip().lower()
        self.binding_slot_z_index = int(self.binding_slot_z_index)
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.binding_slot_repair_stage = str(self.binding_slot_repair_stage).strip()
        self.binding_slot_repair_role = str(self.binding_slot_repair_role).strip()
        self.binding_slot_repair_priority = int(self.binding_slot_repair_priority)
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_seed_group = (
            str(self.residual_slot_seed_group).strip().lower()
        )
        self.residual_slot_z_index = int(self.residual_slot_z_index)
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.residual_slot_repair_stage = str(self.residual_slot_repair_stage).strip()
        self.residual_slot_repair_role = str(self.residual_slot_repair_role).strip()
        self.residual_slot_repair_priority = int(self.residual_slot_repair_priority)
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.exact_trim_floor_value = float(self.exact_trim_floor_value)
        self.target_fold_id = int(self.target_fold_id)
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.target_exact_trim_floor_weighted_retention = float(
            self.target_exact_trim_floor_weighted_retention
        )
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_runtime_bridge_after_report_frontier_packet_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_runtime_bridge_after_report_frontier_packet_digest
        )


def _ensure_shared_frontier(
    *,
    admission_order_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport
    ),
    after_report_intake_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport
    ),
    source_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport
    ),
    live_gap_packet_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport
    ),
    binding_slot_priority_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport
    ),
) -> None:
    policy_digest = admission_order_report.policy_digest
    binding_design = admission_order_report.binding_design
    window_label = admission_order_report.window_label
    random_states = admission_order_report.same_seed_random_states

    for report in (
        after_report_intake_report,
        source_guard_report,
        live_gap_packet_report,
        binding_slot_priority_bridge_report,
    ):
        if report.policy_digest != policy_digest:
            raise ValueError(
                "runtime bridge after-report frontier packet requires shared policy digest"
            )
        if report.binding_design != binding_design:
            raise ValueError(
                "runtime bridge after-report frontier packet requires shared binding design"
            )
        if report.window_label != window_label:
            raise ValueError(
                "runtime bridge after-report frontier packet requires shared window label"
            )
        if report.same_seed_random_states != random_states:
            raise ValueError(
                "runtime bridge after-report frontier packet requires shared same-seed ordering"
            )

    if (
        live_gap_packet_report.runtime_witness_path
        != after_report_intake_report.runtime_witness_path
    ):
        raise ValueError(
            "runtime bridge after-report frontier packet requires shared runtime witness path"
        )
    if (
        live_gap_packet_report.acceptance_readout_path
        != after_report_intake_report.acceptance_readout_path
    ):
        raise ValueError(
            "runtime bridge after-report frontier packet requires shared acceptance readout path"
        )
    if (
        live_gap_packet_report.binding_slot_random_state,
        live_gap_packet_report.binding_slot_z_index,
    ) != (
        binding_slot_priority_bridge_report.next_required_slot_random_state,
        binding_slot_priority_bridge_report.next_required_slot_z_index,
    ):
        raise ValueError(
            "runtime bridge after-report frontier packet requires shared seed303 binding slot"
        )
    if (
        live_gap_packet_report.residual_slot_random_state,
        live_gap_packet_report.residual_slot_z_index,
    ) != (
        binding_slot_priority_bridge_report.residual_slot_random_state,
        binding_slot_priority_bridge_report.residual_slot_z_index,
    ):
        raise ValueError(
            "runtime bridge after-report frontier packet requires shared seed707 residual slot"
        )


def _driver_signature(
    *,
    admission_order_driver_signature: str,
    intake_driver_signature: str,
    source_guard_driver_signature: str,
    live_gap_packet_driver_signature: str,
    binding_slot_priority_bridge_driver_signature: str,
    after_report_supplied: bool,
    intake_disposition: str,
    binding_only_payload_rejected: bool,
    completion_after_report_accepted: bool,
    queued_residual_slot_still_live: bool,
) -> str:
    if (
        admission_order_driver_signature == "same-seed-admission-order"
        and intake_driver_signature
        == "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-intake-open"
        and source_guard_driver_signature
        == "same-seed-exact-witness-observed-rerun-runtime-bridge-source-guard-open"
        and live_gap_packet_driver_signature
        == "same-seed-exact-witness-observed-rerun-live-gap-packet-open"
        and binding_slot_priority_bridge_driver_signature
        == "same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-open"
        and not after_report_supplied
        and intake_disposition == "baseline-no-after-report"
        and not binding_only_payload_rejected
        and not completion_after_report_accepted
        and queued_residual_slot_still_live
    ):
        return "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-frontier-packet-open"
    return "mixed-same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-frontier-packet"


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportFrontierPacketReport
    ),
) -> tuple[str, ...]:
    return (
        "- the higher same-seed frontier now keeps the exact replay order explicit before any after_report payload is supplied: "
        f"`same-seed-admission-order`, baseline/no-after-report intake, and witness floor `{_format_ninths(report.current_witness_floor)}` "
        "all still point first to seed `303` / `z = 0.15` while seed `707` / `z = 0.25` remains queued residual-only cleanup",
        "- the same higher packet keeps the current RED frontier machine-readable on one reusable validation-only asset: "
        "`same-seed-exact-witness-observed-rerun-live-gap-packet-open`, `same-seed-exact-witness-target-gap-open`, "
        "`same-seed-exact-witness-binding-slot-progress-profile`, `same-seed-exact-witness-residual-slot-completion-profile`, "
        "`same-seed-exact-witness-binding-slot-live-gap-open`, `same-seed-exact-witness-residual-slot-live-gap-open`, "
        "`candidate-below-binding-slot-progress-profile`, `same-seed-estimator-evidence-rejected`, and "
        "`same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-open` all remain aligned on the same seed303-before-seed707 packet",
        "- the higher packet also keeps the after-report aware runtime bridge and source guards in sync with the trim-floor packet: "
        f"baseline intake stays `baseline-no-after-report`, the current runtime bridge remains open, and the exact trim-floor row `pi_hat = {report.exact_trim_floor_value:.6f}` "
        f"still carries `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the gross inverse-`pi_hat` conduit, "
        f"`{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, "
        f"`{_format_percent(report.target_exact_trim_floor_weighted_retention)}` retention after weighting, replication seed `{report.binding_replication_seed}`, "
        f"and acceptance readout `{report.acceptance_readout_path[0]} -> {report.acceptance_readout_path[1]} -> {report.acceptance_readout_path[2]}`",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-frontier-packet-open`; "
        "keep live entry on `trigger2-policy-spec`, keep this helper validation-only, land seed `303` / witness / `z = 0.15` before queued residual seed `707` / `z = 0.25`, "
        "and only treat completion after_report as admissible once `same-seed-before-after-acceptance-satisfied` is truly observed on the canonical seed order",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_frontier_packet_report(
    *,
    admission_order_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport
        | None
    ) = None,
    after_report_intake_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport
        | None
    ) = None,
    source_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport
        | None
    ) = None,
    live_gap_packet_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport
        | None
    ) = None,
    binding_slot_priority_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportFrontierPacketReport:
    resolved_admission = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order()
        if admission_order_report is None
        else admission_order_report
    )
    resolved_source_guard = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard()
        if source_guard_report is None
        else source_guard_report
    )
    resolved_after_report_intake = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake_report(
            source_guard_report=resolved_source_guard
        )
        if after_report_intake_report is None
        else after_report_intake_report
    )
    resolved_live_gap_packet = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet()
        if live_gap_packet_report is None
        else live_gap_packet_report
    )
    resolved_binding_slot_priority_bridge = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_binding_slot_priority_bridge()
        if binding_slot_priority_bridge_report is None
        else binding_slot_priority_bridge_report
    )

    _ensure_shared_frontier(
        admission_order_report=resolved_admission,
        after_report_intake_report=resolved_after_report_intake,
        source_guard_report=resolved_source_guard,
        live_gap_packet_report=resolved_live_gap_packet,
        binding_slot_priority_bridge_report=resolved_binding_slot_priority_bridge,
    )

    driver_signature = _driver_signature(
        admission_order_driver_signature=resolved_admission.driver_signature,
        intake_driver_signature=resolved_after_report_intake.driver_signature,
        source_guard_driver_signature=resolved_source_guard.driver_signature,
        live_gap_packet_driver_signature=resolved_live_gap_packet.driver_signature,
        binding_slot_priority_bridge_driver_signature=(
            resolved_binding_slot_priority_bridge.driver_signature
        ),
        after_report_supplied=resolved_after_report_intake.after_report_supplied,
        intake_disposition=resolved_after_report_intake.intake_disposition,
        binding_only_payload_rejected=(
            resolved_after_report_intake.binding_only_payload_rejected
        ),
        completion_after_report_accepted=(
            resolved_after_report_intake.completion_after_report_accepted
        ),
        queued_residual_slot_still_live=(
            resolved_after_report_intake.queued_residual_slot_still_live
        ),
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportFrontierPacketReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "runtime-bridge-after-report-frontier-packet"
        ),
        policy_digest=resolved_admission.policy_digest,
        binding_design=resolved_admission.binding_design,
        window_label=resolved_admission.window_label,
        same_seed_random_states=resolved_admission.same_seed_random_states,
        admission_order_driver_signature=resolved_admission.driver_signature,
        runtime_bridge_after_report_intake_driver_signature=(
            resolved_after_report_intake.driver_signature
        ),
        runtime_bridge_source_guard_driver_signature=(
            resolved_source_guard.driver_signature
        ),
        live_gap_packet_driver_signature=resolved_live_gap_packet.driver_signature,
        binding_slot_priority_bridge_driver_signature=(
            resolved_binding_slot_priority_bridge.driver_signature
        ),
        after_report_supplied=resolved_after_report_intake.after_report_supplied,
        intake_disposition=resolved_after_report_intake.intake_disposition,
        binding_only_payload_rejected=(
            resolved_after_report_intake.binding_only_payload_rejected
        ),
        completion_after_report_accepted=(
            resolved_after_report_intake.completion_after_report_accepted
        ),
        queued_residual_slot_still_live=(
            resolved_after_report_intake.queued_residual_slot_still_live
        ),
        current_runtime_bridge_state=(
            resolved_after_report_intake.current_runtime_bridge_state
        ),
        current_rung_status=resolved_live_gap_packet.observed_rerun_rung_status,
        current_point_miss_vector=resolved_live_gap_packet.current_point_miss_vector,
        current_band_miss_vector=resolved_live_gap_packet.current_band_miss_vector,
        current_witness_floor=resolved_live_gap_packet.current_witness_floor,
        required_min_witness_floor=resolved_live_gap_packet.required_min_witness_floor,
        acceptance_shortfall=resolved_after_report_intake.acceptance_shortfall,
        binding_slot_random_state=resolved_live_gap_packet.binding_slot_random_state,
        binding_slot_seed_group=resolved_live_gap_packet.binding_slot_seed_group,
        binding_slot_z_index=resolved_live_gap_packet.binding_slot_z_index,
        binding_slot_z_value=resolved_live_gap_packet.binding_slot_z_value,
        binding_slot_repair_stage=resolved_live_gap_packet.binding_slot_repair_stage,
        binding_slot_repair_role=resolved_live_gap_packet.binding_slot_repair_role,
        binding_slot_repair_priority=resolved_live_gap_packet.binding_slot_repair_priority,
        residual_slot_random_state=resolved_live_gap_packet.residual_slot_random_state,
        residual_slot_seed_group=resolved_live_gap_packet.residual_slot_seed_group,
        residual_slot_z_index=resolved_live_gap_packet.residual_slot_z_index,
        residual_slot_z_value=resolved_live_gap_packet.residual_slot_z_value,
        residual_slot_repair_stage=(
            resolved_live_gap_packet.residual_slot_repair_stage
        ),
        residual_slot_repair_role=resolved_live_gap_packet.residual_slot_repair_role,
        residual_slot_repair_priority=(
            resolved_live_gap_packet.residual_slot_repair_priority
        ),
        runtime_witness_path=resolved_live_gap_packet.runtime_witness_path,
        acceptance_readout_path=resolved_live_gap_packet.acceptance_readout_path,
        exact_trim_floor_value=resolved_after_report_intake.exact_trim_floor_value,
        target_fold_id=resolved_after_report_intake.target_fold_id,
        target_exact_trim_floor_count=(
            resolved_after_report_intake.target_exact_trim_floor_count
        ),
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            resolved_after_report_intake.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            resolved_after_report_intake.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            resolved_after_report_intake.target_exact_trim_floor_weighted_retention
        ),
        binding_replication_seed=resolved_after_report_intake.binding_replication_seed,
        driver_signature=driver_signature,
        canonical_runtime_bridge_after_report_frontier_packet_digest=(),
    )
    report.canonical_runtime_bridge_after_report_frontier_packet_digest = (
        _canonical_digest(report=report)
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_frontier_packet() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportFrontierPacketReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_frontier_packet_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportFrontierPacketReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_frontier_packet_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_frontier_packet",
]
