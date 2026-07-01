from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda_report,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _slot_matches(
    *,
    random_state: int | None,
    z_index: int | None,
    expected_random_state: int | None,
    expected_z_index: int | None,
) -> bool:
    if (
        random_state is None
        or z_index is None
        or expected_random_state is None
        or expected_z_index is None
    ):
        return (
            random_state is None
            and z_index is None
            and expected_random_state is None
            and expected_z_index is None
        )
    return bool(
        int(random_state) == int(expected_random_state)
        and int(z_index) == int(expected_z_index)
    )


def _driver_signature(
    *,
    repair_agenda_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
    ),
    runtime_binding_packet_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport
    ),
    first_hop_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport
    ),
) -> str:
    if (
        repair_agenda_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-open"
        and runtime_binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-open"
        and repair_agenda_report.current_rung_status
        == "same-seed-exact-witness-observed-rerun-open"
        and _slot_matches(
            random_state=repair_agenda_report.next_required_slot.random_state
            if repair_agenda_report.next_required_slot is not None
            else None,
            z_index=repair_agenda_report.next_required_slot.z_index
            if repair_agenda_report.next_required_slot is not None
            else None,
            expected_random_state=first_hop_contract_report.binding_slot_random_state,
            expected_z_index=first_hop_contract_report.binding_slot_z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-landing-guard-open"
    if (
        repair_agenda_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-residual-only"
        and runtime_binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-landed"
        and repair_agenda_report.current_rung_status
        == "same-seed-exact-witness-binding-slot-progress-landed"
        and _slot_matches(
            random_state=repair_agenda_report.next_required_slot.random_state
            if repair_agenda_report.next_required_slot is not None
            else None,
            z_index=repair_agenda_report.next_required_slot.z_index
            if repair_agenda_report.next_required_slot is not None
            else None,
            expected_random_state=first_hop_contract_report.residual_slot_random_state,
            expected_z_index=first_hop_contract_report.residual_slot_z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-landing-guard-landed"
    if (
        repair_agenda_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-closed"
        and runtime_binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-closed"
        and repair_agenda_report.current_rung_status
        == "same-seed-exact-witness-residual-slot-completion-landed"
        and repair_agenda_report.next_required_slot is None
        and runtime_binding_packet_report.next_required_slot_random_state is None
        and runtime_binding_packet_report.next_required_slot_z_index is None
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-landing-guard-closed"
    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-landing-guard"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    current_rung_status: str
    repair_agenda_driver_signature: str
    runtime_binding_packet_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    first_hop_landed: bool
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_index: int
    binding_slot_z_value: float
    binding_slot_repair_stage: str
    binding_slot_repair_role: str
    binding_slot_repair_priority: int
    binding_slot_replication_seed: int
    queued_residual_slot_random_state: int
    queued_residual_slot_seed_group: str
    queued_residual_slot_z_index: int
    queued_residual_slot_z_value: float
    next_required_slot_random_state: int | None
    next_required_slot_seed_group: str | None
    next_required_slot_z_index: int | None
    next_required_slot_z_value: float | None
    next_required_slot_repair_stage: str | None
    next_required_slot_repair_role: str | None
    next_required_slot_repair_priority: int | None
    driver_signature: str
    canonical_observed_rerun_first_hop_landing_guard_digest: tuple[str, ...]

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
        self.repair_agenda_driver_signature = str(
            self.repair_agenda_driver_signature
        ).strip()
        self.runtime_binding_packet_driver_signature = str(
            self.runtime_binding_packet_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.first_hop_landed = bool(self.first_hop_landed)
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_seed_group = str(self.binding_slot_seed_group).strip().lower()
        self.binding_slot_z_index = int(self.binding_slot_z_index)
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.binding_slot_repair_stage = str(self.binding_slot_repair_stage).strip()
        self.binding_slot_repair_role = str(self.binding_slot_repair_role).strip()
        self.binding_slot_repair_priority = int(self.binding_slot_repair_priority)
        self.binding_slot_replication_seed = int(self.binding_slot_replication_seed)
        self.queued_residual_slot_random_state = int(
            self.queued_residual_slot_random_state
        )
        self.queued_residual_slot_seed_group = (
            str(self.queued_residual_slot_seed_group).strip().lower()
        )
        self.queued_residual_slot_z_index = int(self.queued_residual_slot_z_index)
        self.queued_residual_slot_z_value = float(self.queued_residual_slot_z_value)
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
        self.canonical_observed_rerun_first_hop_landing_guard_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_observed_rerun_first_hop_landing_guard_digest
        )


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-landing-guard-closed"
    ):
        return (
            "- the observed-rerun repair agenda and the runtime binding packet are now jointly exhausted: seed `303` / witness / `z = 0.15` and seed `707` / fresh / `z = 0.25` have both landed, so the first-hop ordering contract no longer leaves a pending slot",
            f"- the landing guard therefore becomes historical provenance only: current same-seed readout already sits at point miss `{list(report.current_point_miss_vector)}`, band miss `{list(report.current_band_miss_vector)}`, and witness floor `{_format_ninths(report.current_witness_floor)}` while the source-backed packet keeps replication seed `883193502` and the runtime path frozen for auditability",
            "- the helper remains validation-only: it closes the first-hop order without reopening the left guard `z = 0.05` or promoting any new live routing token",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-landing-guard-closed`; keep the helper validation-only and out of live routing surfaces",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-landing-guard-landed"
    ):
        return (
            "- the observed-rerun repair agenda and the runtime binding packet agree that the first hop has landed: seed `303` / witness / `z = 0.15` is no longer pending, and only seed `707` / fresh / `z = 0.25` remains queued as the residual repair",
            f"- current same-seed readout now matches the binding-only landing profile: point miss is `{list(report.current_point_miss_vector)}`, band miss stays `{list(report.current_band_miss_vector)}`, and witness floor holds at `{_format_ninths(report.current_witness_floor)}` while the left guard `z = 0.05` stays untouched",
            "- the landed state keeps the same source-backed packet in force for auditability: replication seed `883193502` still pins `bar_f_at_z0[1]`, `v_f_hat[2,1]`, and the preserved runtime path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-landing-guard-landed`; spend the next rerun only on the queued residual slot at seed `707` / `z = 0.25`",
        )
    return (
        "- the observed-rerun repair agenda and the runtime binding packet still agree on the same open first hop: seed `303` / witness / `z = 0.15` remains the actionable next slot, while seed `707` / fresh / `z = 0.25` stays queued as the residual repair",
        f"- current same-seed readout is still below first-hop landing: point miss stays `{list(report.current_point_miss_vector)}`, band miss stays `{list(report.current_band_miss_vector)}`, and witness floor remains `{_format_ninths(report.current_witness_floor)}`, so the open admissibility gap is still fully owned by the seed `303` binding slot",
        "- the landing guard keeps the same source-backed runtime packet intact: replication seed `883193502` still carries `bar_f_at_z0[1]`, `v_f_hat[2,1]`, and the preserved runtime path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` without opening the left guard `z = 0.05`",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-landing-guard-open`; feed any real same-seed rerun through this validation-only landing guard and land seed `303` before spending the queued residual slot on seed `707` / `z = 0.25`",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard_report(
    *,
    after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    repair_agenda_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
        | None
    ) = None,
    first_hop_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport
        | None
    ) = None,
    runtime_binding_packet_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport:
    resolved_after = (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        if after_report is None
        else after_report
    )
    resolved_repair_agenda = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda_report(
            after_report=resolved_after
        )
        if repair_agenda_report is None
        else repair_agenda_report
    )
    resolved_first_hop_contract = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract_report(
            after_report=resolved_after
        )
        if first_hop_contract_report is None
        else first_hop_contract_report
    )
    resolved_runtime_binding_packet = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet_report(
            first_hop_contract_report=resolved_first_hop_contract
        )
        if runtime_binding_packet_report is None
        else runtime_binding_packet_report
    )

    if (
        resolved_repair_agenda.policy_digest
        != resolved_runtime_binding_packet.policy_digest
    ):
        raise ValueError("first-hop landing guard requires shared policy digest")
    if (
        resolved_repair_agenda.binding_design
        != resolved_runtime_binding_packet.binding_design
    ):
        raise ValueError("first-hop landing guard requires shared binding design")
    if (
        resolved_repair_agenda.window_label
        != resolved_runtime_binding_packet.window_label
    ):
        raise ValueError("first-hop landing guard requires shared window label")
    if (
        resolved_repair_agenda.same_seed_random_states
        != resolved_runtime_binding_packet.same_seed_random_states
    ):
        raise ValueError(
            "first-hop landing guard requires shared exact same-seed ordering"
        )
    if (
        resolved_repair_agenda.current_rung_status
        != resolved_runtime_binding_packet.current_rung_status
    ):
        raise ValueError("first-hop landing guard requires shared current rung status")
    if (
        resolved_repair_agenda.runtime_witness_path
        != resolved_runtime_binding_packet.runtime_witness_path
    ):
        raise ValueError(
            "first-hop landing guard requires the preserved runtime witness path"
        )

    next_required_slot = resolved_repair_agenda.next_required_slot
    if not _slot_matches(
        random_state=next_required_slot.random_state if next_required_slot else None,
        z_index=next_required_slot.z_index if next_required_slot else None,
        expected_random_state=resolved_runtime_binding_packet.next_required_slot_random_state,
        expected_z_index=resolved_runtime_binding_packet.next_required_slot_z_index,
    ):
        raise ValueError(
            "first-hop landing guard requires packet and agenda to agree on the next slot"
        )

    driver_signature = _driver_signature(
        repair_agenda_report=resolved_repair_agenda,
        runtime_binding_packet_report=resolved_runtime_binding_packet,
        first_hop_contract_report=resolved_first_hop_contract,
    )
    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "first-hop-landing-guard"
        ),
        policy_digest=resolved_repair_agenda.policy_digest,
        binding_design=resolved_repair_agenda.binding_design,
        window_label=resolved_repair_agenda.window_label,
        same_seed_random_states=resolved_repair_agenda.same_seed_random_states,
        current_rung_status=resolved_repair_agenda.current_rung_status,
        repair_agenda_driver_signature=resolved_repair_agenda.driver_signature,
        runtime_binding_packet_driver_signature=resolved_runtime_binding_packet.driver_signature,
        runtime_witness_path=resolved_repair_agenda.runtime_witness_path,
        current_point_miss_vector=resolved_repair_agenda.current_point_miss_vector,
        current_band_miss_vector=resolved_repair_agenda.current_band_miss_vector,
        current_witness_floor=resolved_repair_agenda.current_witness_floor,
        required_min_witness_floor=resolved_repair_agenda.required_min_witness_floor,
        first_hop_landed=(
            resolved_repair_agenda.current_rung_status
            != "same-seed-exact-witness-observed-rerun-open"
        ),
        binding_slot_random_state=resolved_first_hop_contract.binding_slot_random_state,
        binding_slot_seed_group=resolved_first_hop_contract.binding_slot_seed_group,
        binding_slot_z_index=resolved_first_hop_contract.binding_slot_z_index,
        binding_slot_z_value=resolved_first_hop_contract.binding_slot_z_value,
        binding_slot_repair_stage=resolved_first_hop_contract.binding_slot_repair_stage,
        binding_slot_repair_role=resolved_first_hop_contract.binding_slot_repair_role,
        binding_slot_repair_priority=resolved_first_hop_contract.binding_slot_repair_priority,
        binding_slot_replication_seed=resolved_runtime_binding_packet.binding_runtime_packet.replication_seed,
        queued_residual_slot_random_state=resolved_first_hop_contract.residual_slot_random_state,
        queued_residual_slot_seed_group=resolved_first_hop_contract.residual_slot_seed_group,
        queued_residual_slot_z_index=resolved_first_hop_contract.residual_slot_z_index,
        queued_residual_slot_z_value=resolved_first_hop_contract.residual_slot_z_value,
        next_required_slot_random_state=resolved_runtime_binding_packet.next_required_slot_random_state,
        next_required_slot_seed_group=resolved_runtime_binding_packet.next_required_slot_seed_group,
        next_required_slot_z_index=resolved_runtime_binding_packet.next_required_slot_z_index,
        next_required_slot_z_value=resolved_runtime_binding_packet.next_required_slot_z_value,
        next_required_slot_repair_stage=resolved_runtime_binding_packet.next_required_slot_repair_stage,
        next_required_slot_repair_role=resolved_runtime_binding_packet.next_required_slot_repair_role,
        next_required_slot_repair_priority=resolved_runtime_binding_packet.next_required_slot_repair_priority,
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_landing_guard_digest=(),
    )
    report.canonical_observed_rerun_first_hop_landing_guard_digest = _canonical_digest(
        report=report
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard",
]
