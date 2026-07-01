from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract_report,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport,
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence_report,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence,
)

_RUNTIME_WITNESS_PATH = (
    "omega_f_hat[2,2]",
    "v_f_hat[2,2]",
    "covariance(0.25, 0.15)",
)
_RUNTIME_OBJECT_FLOW_FOCUS = (
    "v_f_hat[2,1]",
    "bar_f_at_z0[1]",
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ratio(value: float) -> str:
    return f"{float(value):.3f}x"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacket:
    random_state: int
    seed_group: str
    replication_seed: int
    z_index: int
    z_value: float
    repair_stage: str
    repair_role: str
    repair_priority: int
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    baseline_open_witness_floor_gap: float
    share_of_baseline_open_witness_floor_gap: float
    center_truth: float
    center_estimate: float
    center_error: float
    center_pointwise_interval_lower: float
    center_pointwise_interval_upper: float
    center_half_interval: float
    center_sigma: float
    center_lower_minus_truth: float
    center_error_to_half_interval_ratio: float
    vf_cross_entry: float
    runtime_object_flow_focus: tuple[str, ...]

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.replication_seed = int(self.replication_seed)
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)
        self.repair_stage = str(self.repair_stage).strip()
        self.repair_role = str(self.repair_role).strip()
        self.repair_priority = int(self.repair_priority)
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
        self.baseline_open_witness_floor_gap = float(
            self.baseline_open_witness_floor_gap
        )
        self.share_of_baseline_open_witness_floor_gap = float(
            self.share_of_baseline_open_witness_floor_gap
        )
        self.center_truth = float(self.center_truth)
        self.center_estimate = float(self.center_estimate)
        self.center_error = float(self.center_error)
        self.center_pointwise_interval_lower = float(
            self.center_pointwise_interval_lower
        )
        self.center_pointwise_interval_upper = float(
            self.center_pointwise_interval_upper
        )
        self.center_half_interval = float(self.center_half_interval)
        self.center_sigma = float(self.center_sigma)
        self.center_lower_minus_truth = float(self.center_lower_minus_truth)
        self.center_error_to_half_interval_ratio = float(
            self.center_error_to_half_interval_ratio
        )
        self.vf_cross_entry = float(self.vf_cross_entry)
        self.runtime_object_flow_focus = tuple(
            str(item).strip() for item in self.runtime_object_flow_focus
        )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    current_rung_status: str
    first_hop_contract_driver_signature: str
    seed303_object_flow_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    runtime_object_flow_focus: tuple[str, ...]
    actionable_now: bool
    next_required_slot_random_state: int | None
    next_required_slot_seed_group: str | None
    next_required_slot_z_index: int | None
    next_required_slot_z_value: float | None
    next_required_slot_repair_stage: str | None
    next_required_slot_repair_role: str | None
    next_required_slot_repair_priority: int | None
    binding_runtime_packet: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacket
    driver_signature: str
    canonical_observed_rerun_first_hop_runtime_binding_packet_digest: tuple[str, ...]

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
        self.current_rung_status = str(self.current_rung_status).strip()
        self.first_hop_contract_driver_signature = str(
            self.first_hop_contract_driver_signature
        ).strip()
        self.seed303_object_flow_driver_signature = str(
            self.seed303_object_flow_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.runtime_object_flow_focus = tuple(
            str(item).strip() for item in self.runtime_object_flow_focus
        )
        self.actionable_now = bool(self.actionable_now)
        self.next_required_slot_random_state = (
            None
            if self.next_required_slot_random_state is None
            else int(self.next_required_slot_random_state)
        )
        self.next_required_slot_seed_group = (
            None
            if self.next_required_slot_seed_group is None
            else str(self.next_required_slot_seed_group).strip().lower()
        )
        self.next_required_slot_z_index = (
            None
            if self.next_required_slot_z_index is None
            else int(self.next_required_slot_z_index)
        )
        self.next_required_slot_z_value = (
            None
            if self.next_required_slot_z_value is None
            else float(self.next_required_slot_z_value)
        )
        self.next_required_slot_repair_stage = (
            None
            if self.next_required_slot_repair_stage is None
            else str(self.next_required_slot_repair_stage).strip()
        )
        self.next_required_slot_repair_role = (
            None
            if self.next_required_slot_repair_role is None
            else str(self.next_required_slot_repair_role).strip()
        )
        self.next_required_slot_repair_priority = (
            None
            if self.next_required_slot_repair_priority is None
            else int(self.next_required_slot_repair_priority)
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_first_hop_runtime_binding_packet_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_first_hop_runtime_binding_packet_digest
        )


def _slot_matches(
    *,
    random_state: int | None,
    z_index: int | None,
    expected_random_state: int,
    expected_z_index: int,
) -> bool:
    return bool(
        random_state is not None
        and z_index is not None
        and int(random_state) == int(expected_random_state)
        and int(z_index) == int(expected_z_index)
    )


def _build_binding_runtime_packet(
    *,
    first_hop_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport
    ),
    focus_seed_runtime_snapshot: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacket:
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacket(
        random_state=first_hop_contract_report.binding_slot_random_state,
        seed_group=first_hop_contract_report.binding_slot_seed_group,
        replication_seed=focus_seed_runtime_snapshot.replication_seed,
        z_index=first_hop_contract_report.binding_slot_z_index,
        z_value=first_hop_contract_report.binding_slot_z_value,
        repair_stage=first_hop_contract_report.binding_slot_repair_stage,
        repair_role=first_hop_contract_report.binding_slot_repair_role,
        repair_priority=first_hop_contract_report.binding_slot_repair_priority,
        target_fold_id=first_hop_contract_report.target_fold_id,
        target_exact_trim_floor_count=first_hop_contract_report.target_exact_trim_floor_count,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=first_hop_contract_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi,
        target_exact_trim_floor_weighted_share_of_fold3_weighted=first_hop_contract_report.target_exact_trim_floor_weighted_share_of_fold3_weighted,
        target_exact_trim_floor_weighted_retention=first_hop_contract_report.target_exact_trim_floor_weighted_retention,
        baseline_open_witness_floor_gap=first_hop_contract_report.baseline_open_witness_floor_gap,
        share_of_baseline_open_witness_floor_gap=first_hop_contract_report.binding_slot_share_of_baseline_open_witness_floor_gap,
        center_truth=focus_seed_runtime_snapshot.center_truth,
        center_estimate=focus_seed_runtime_snapshot.center_estimate,
        center_error=focus_seed_runtime_snapshot.center_error,
        center_pointwise_interval_lower=focus_seed_runtime_snapshot.center_pointwise_interval_lower,
        center_pointwise_interval_upper=focus_seed_runtime_snapshot.center_pointwise_interval_upper,
        center_half_interval=focus_seed_runtime_snapshot.center_half_interval,
        center_sigma=focus_seed_runtime_snapshot.center_sigma,
        center_lower_minus_truth=focus_seed_runtime_snapshot.center_lower_minus_truth,
        center_error_to_half_interval_ratio=focus_seed_runtime_snapshot.center_error_to_half_interval_ratio,
        vf_cross_entry=focus_seed_runtime_snapshot.vf_cross_entry,
        runtime_object_flow_focus=_RUNTIME_OBJECT_FLOW_FOCUS,
    )


def _driver_signature(
    *,
    first_hop_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport
    ),
    seed303_object_flow_runtime_evidence_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport
    ),
    binding_runtime_packet: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacket
    ),
) -> str:
    if (
        first_hop_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-contract-open"
        and seed303_object_flow_runtime_evidence_report.driver_signature
        == "same-seed-seed303-object-flow-runtime-evidence"
        and seed303_object_flow_runtime_evidence_report.runtime_witness_path
        == _RUNTIME_WITNESS_PATH
        and _slot_matches(
            random_state=first_hop_contract_report.next_required_slot_random_state,
            z_index=first_hop_contract_report.next_required_slot_z_index,
            expected_random_state=binding_runtime_packet.random_state,
            expected_z_index=binding_runtime_packet.z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-open"
    if (
        first_hop_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-contract-residual-only"
        and seed303_object_flow_runtime_evidence_report.driver_signature
        == "same-seed-seed303-object-flow-runtime-evidence"
        and seed303_object_flow_runtime_evidence_report.runtime_witness_path
        == _RUNTIME_WITNESS_PATH
        and _slot_matches(
            random_state=first_hop_contract_report.next_required_slot_random_state,
            z_index=first_hop_contract_report.next_required_slot_z_index,
            expected_random_state=first_hop_contract_report.residual_slot_random_state,
            expected_z_index=first_hop_contract_report.residual_slot_z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-landed"
    if (
        first_hop_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-contract-closed"
        and seed303_object_flow_runtime_evidence_report.driver_signature
        == "same-seed-seed303-object-flow-runtime-evidence"
        and first_hop_contract_report.next_required_slot_random_state is None
        and first_hop_contract_report.next_required_slot_z_index is None
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-closed"
    return (
        "mixed-same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet"
    )


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport
    ),
) -> tuple[str, ...]:
    packet = report.binding_runtime_packet
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-closed"
    ):
        return (
            f"- the observed-rerun first hop runtime binding packet is fully consumed: witness floor already sits at `{_format_ninths(packet.baseline_open_witness_floor_gap + (8.0 / 9.0 - packet.baseline_open_witness_floor_gap))}` only because the current same-seed rerun has already landed both the seed `303` first hop and the queued seed `707` residual slot",
            "- the machine-readable packet therefore becomes historical provenance only: the seed `303` packet still records replication seed `883193502`, the runtime path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`, and the downstream object-flow focus `v_f_hat[2,1] -> bar_f_at_z0[1]` that had to land first",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-closed`; keep the helper validation-only and out of live routing surfaces",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-landed"
    ):
        return (
            f"- the observed-rerun first hop runtime binding packet has already landed the seed `303` admissibility-facing slot: the machine-readable packet still records seed `303` / witness / `z = 0.15` / replication seed `{packet.replication_seed}` while current queue ownership moves to seed `707` / `z = 0.25` residual cleanup",
            f"- the packet keeps the same object-flow reason frozen for auditability: runtime path `{' -> '.join(report.runtime_witness_path)}` with downstream object-flow focus `{' -> '.join(report.runtime_object_flow_focus)}`, truth `{_format_float(packet.center_truth)}` versus `bar_f_at_z0 = {_format_float(packet.center_estimate)}`, lower bound `{_format_float(packet.center_pointwise_interval_lower)}`, error-to-half-width ratio `{_format_ratio(packet.center_error_to_half_interval_ratio)}`, and `v_f_hat[2,1] = {_format_float(packet.vf_cross_entry)}` remain the why-now provenance for the first hop",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-landed`; keep the helper validation-only and spend the next rerun only on the queued seed `707` / `z = 0.25` residual slot",
        )
    return (
        "- the observed-rerun first hop is now pinned as a machine-readable runtime binding packet: seed `303` / witness / `z = 0.15` / replication seed `883193502` stays priority `1` / `binding-witness-floor-lift`, shares the same runtime path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` with downstream object-flow focus `v_f_hat[2,1] -> bar_f_at_z0[1]`, and remains the actionable next slot",
        "- the packet confirms this is not a width-only miss: truth `1.162` versus `bar_f_at_z0 = 9.186`, lower bound `2.463` still above truth by `1.301`, absolute error `8.024` already at `1.194x` the half-width `6.723`, and `v_f_hat[2,1]` remains negative at `-287.557`",
        "- the packet stays one-sided and source-backed: seed `303` keeps the only exact trim-floor row in fold `3`, retains `99.0%` of the fold-`3` weighted conduit, and still owns `1/9 = 0.111` / `100.0%` of the open witness-floor gap before any residual seed `707` spend",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-open`; feed the real same-seed rerun through this validation-only runtime binding packet before spending the queued residual slot on seed `707` / `z = 0.25`",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet_report(
    *,
    first_hop_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport
        | None
    ) = None,
    seed303_object_flow_runtime_evidence_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport:
    resolved_first_hop_contract = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract()
        if first_hop_contract_report is None
        else first_hop_contract_report
    )
    resolved_object_flow = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence()
        if seed303_object_flow_runtime_evidence_report is None
        else seed303_object_flow_runtime_evidence_report
    )

    if resolved_first_hop_contract.policy_digest != resolved_object_flow.policy_digest:
        raise ValueError("runtime binding packet requires shared policy digest")
    if (
        resolved_first_hop_contract.binding_design
        != resolved_object_flow.binding_design
    ):
        raise ValueError("runtime binding packet requires shared binding design")
    if resolved_first_hop_contract.window_label != resolved_object_flow.window_label:
        raise ValueError("runtime binding packet requires shared window label")
    if (
        resolved_first_hop_contract.same_seed_random_states
        != resolved_object_flow.same_seed_random_states
    ):
        raise ValueError(
            "runtime binding packet requires shared exact same-seed ordering"
        )
    if resolved_object_flow.runtime_witness_path != _RUNTIME_WITNESS_PATH:
        raise ValueError(
            "runtime binding packet requires the preserved runtime witness path"
        )

    binding_runtime_packet = _build_binding_runtime_packet(
        first_hop_contract_report=resolved_first_hop_contract,
        focus_seed_runtime_snapshot=resolved_object_flow.focus_seed_runtime_snapshot,
    )
    driver_signature = _driver_signature(
        first_hop_contract_report=resolved_first_hop_contract,
        seed303_object_flow_runtime_evidence_report=resolved_object_flow,
        binding_runtime_packet=binding_runtime_packet,
    )
    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "first-hop-runtime-binding-packet"
        ),
        policy_digest=resolved_first_hop_contract.policy_digest,
        binding_design=resolved_first_hop_contract.binding_design,
        window_label=resolved_first_hop_contract.window_label,
        same_seed_random_states=resolved_first_hop_contract.same_seed_random_states,
        current_rung_status=resolved_first_hop_contract.current_rung_status,
        first_hop_contract_driver_signature=resolved_first_hop_contract.driver_signature,
        seed303_object_flow_driver_signature=resolved_object_flow.driver_signature,
        runtime_witness_path=resolved_object_flow.runtime_witness_path,
        runtime_object_flow_focus=binding_runtime_packet.runtime_object_flow_focus,
        actionable_now=(
            driver_signature
            == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-open"
        ),
        next_required_slot_random_state=resolved_first_hop_contract.next_required_slot_random_state,
        next_required_slot_seed_group=resolved_first_hop_contract.next_required_slot_seed_group,
        next_required_slot_z_index=resolved_first_hop_contract.next_required_slot_z_index,
        next_required_slot_z_value=resolved_first_hop_contract.next_required_slot_z_value,
        next_required_slot_repair_stage=resolved_first_hop_contract.next_required_slot_repair_stage,
        next_required_slot_repair_role=resolved_first_hop_contract.next_required_slot_repair_role,
        next_required_slot_repair_priority=resolved_first_hop_contract.next_required_slot_repair_priority,
        binding_runtime_packet=binding_runtime_packet,
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_runtime_binding_packet_digest=(),
    )
    report.canonical_observed_rerun_first_hop_runtime_binding_packet_digest = (
        _canonical_digest(report=report)
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacket",
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet",
]
