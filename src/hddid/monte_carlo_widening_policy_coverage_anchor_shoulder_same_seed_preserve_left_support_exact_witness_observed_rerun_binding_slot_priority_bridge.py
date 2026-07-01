from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport,
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _slot_matches(
    slot: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot
    | None,
    *,
    random_state: int,
    z_index: int,
) -> bool:
    return bool(
        slot is not None
        and slot.random_state == int(random_state)
        and slot.z_index == int(z_index)
    )


def _driver_signature(
    *,
    agenda_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
    ),
    priority_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport
    ),
) -> str:
    if (
        priority_trace_report.driver_signature
        != "same-seed-seed303-fold3-trim-floor-binding-slot-priority-confirmed"
    ):
        return (
            "mixed-same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge"
        )
    if (
        agenda_report.current_rung_status
        == "same-seed-exact-witness-observed-rerun-open"
        and agenda_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-open"
        and _slot_matches(
            agenda_report.next_required_slot,
            random_state=priority_trace_report.binding_slot_random_state,
            z_index=priority_trace_report.binding_slot_z_index,
        )
    ):
        return (
            "same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-open"
        )
    if (
        agenda_report.current_rung_status
        == "same-seed-exact-witness-binding-slot-progress-landed"
        and agenda_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-residual-only"
        and _slot_matches(
            agenda_report.next_required_slot,
            random_state=priority_trace_report.residual_slot_random_state,
            z_index=priority_trace_report.residual_slot_z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-residual-only"
    if (
        agenda_report.current_rung_status
        == "same-seed-exact-witness-residual-slot-completion-landed"
        and agenda_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-closed"
        and agenda_report.next_required_slot is None
    ):
        return (
            "same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-closed"
        )
    return "mixed-same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge"


def _next_required_slot_fields(
    *,
    agenda_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
    ),
    priority_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport
    ),
) -> tuple[
    int | None, str | None, int | None, float | None, str | None, str | None, int | None
]:
    next_slot = agenda_report.next_required_slot
    if next_slot is None:
        return None, None, None, None, None, None, None
    if _slot_matches(
        next_slot,
        random_state=priority_trace_report.binding_slot_random_state,
        z_index=priority_trace_report.binding_slot_z_index,
    ):
        return (
            next_slot.random_state,
            next_slot.seed_group,
            next_slot.z_index,
            next_slot.z_value,
            next_slot.repair_stage,
            priority_trace_report.binding_slot_repair_role,
            priority_trace_report.binding_slot_repair_priority,
        )
    if _slot_matches(
        next_slot,
        random_state=priority_trace_report.residual_slot_random_state,
        z_index=priority_trace_report.residual_slot_z_index,
    ):
        return (
            next_slot.random_state,
            next_slot.seed_group,
            next_slot.z_index,
            next_slot.z_value,
            next_slot.repair_stage,
            priority_trace_report.residual_slot_repair_role,
            priority_trace_report.residual_slot_repair_priority,
        )
    raise ValueError(
        "observed-rerun binding-slot priority bridge requires the next slot to match the seed303 binding slot or the seed707 residual slot"
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    current_rung_status: str
    agenda_driver_signature: str
    binding_slot_priority_driver_signature: str
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    current_to_next_rung_pointwise_gap_vector: tuple[int, int, int]
    current_to_next_rung_band_gap_vector: tuple[int, int, int]
    current_to_completion_pointwise_gap_vector: tuple[int, int, int]
    current_to_completion_band_gap_vector: tuple[int, int, int]
    next_required_slot_random_state: int | None
    next_required_slot_seed_group: str | None
    next_required_slot_z_index: int | None
    next_required_slot_z_value: float | None
    next_required_slot_repair_stage: str | None
    next_required_slot_repair_role: str | None
    next_required_slot_repair_priority: int | None
    residual_slot_random_state: int
    residual_slot_seed_group: str
    residual_slot_z_index: int
    residual_slot_z_value: float
    residual_slot_repair_stage: str
    residual_slot_repair_role: str
    residual_slot_repair_priority: int
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    driver_signature: str
    canonical_observed_rerun_binding_slot_priority_bridge_digest: tuple[str, ...]

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
        self.agenda_driver_signature = str(self.agenda_driver_signature).strip()
        self.binding_slot_priority_driver_signature = str(
            self.binding_slot_priority_driver_signature
        ).strip()
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.current_to_next_rung_pointwise_gap_vector = tuple(
            int(value) for value in self.current_to_next_rung_pointwise_gap_vector
        )
        self.current_to_next_rung_band_gap_vector = tuple(
            int(value) for value in self.current_to_next_rung_band_gap_vector
        )
        self.current_to_completion_pointwise_gap_vector = tuple(
            int(value) for value in self.current_to_completion_pointwise_gap_vector
        )
        self.current_to_completion_band_gap_vector = tuple(
            int(value) for value in self.current_to_completion_band_gap_vector
        )
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
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_seed_group = (
            str(self.residual_slot_seed_group).strip().lower()
        )
        self.residual_slot_z_index = int(self.residual_slot_z_index)
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.residual_slot_repair_stage = str(self.residual_slot_repair_stage).strip()
        self.residual_slot_repair_role = str(self.residual_slot_repair_role).strip()
        self.residual_slot_repair_priority = int(self.residual_slot_repair_priority)
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
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_binding_slot_priority_bridge_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_binding_slot_priority_bridge_digest
        )


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-closed"
    ):
        return (
            f"- observed same-seed rerun no longer needs a first-hop bridge: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor already holds at `{_format_ninths(report.current_witness_floor)}`",
            "- the seed `303` fold-`3` trim-floor bridge is therefore consumed: the binding witness-center hop is no longer pending, and the residual shoulder queue is already empty as well",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-closed`; keep this helper validation-only and out of live routing surfaces",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-residual-only"
    ):
        return (
            f"- observed same-seed rerun has already consumed the seed `303` first-hop bridge: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor already sits at `{_format_ninths(report.current_witness_floor)}`",
            "- the fold-`3` trim-floor conduit remains the provenance for why seed `303` had to land first, but the next actionable slot is now the queued residual seed `707` / `z = 0.25`",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-residual-only`; keep this helper validation-only and spend the next rerun on the residual slot only",
        )
    if report.target_exact_trim_floor_count == 0:
        bridge_line = (
            "- the observed rerun and the trim-floor bridge do not share a confirmed exact-row hop on this worktree: seed `303` has no fold-`3` exact trim-floor row, so the exact-row gross inverse-`pi_hat` lift, final weighted burden, and weighted retention are all "
            f"`{_format_percent(0.0)}`; the actionable first hop remains the broader seed `303` / fold `3` / low-`pi_hat` conduit"
        )
        implication_line = (
            f"- current Trigger 2 implication: `{report.driver_signature}`; feed real same-seed reruns through this validation-only bridge, but do not claim the seed `303` exact trim-floor row is landed before the live replay exposes a confirmed exact-row conduit"
        )
    else:
        bridge_line = (
            "- the observed rerun and the trim-floor bridge now share the same first hop: seed `303` keeps exactly one fold-`3` trim-floor row carrying "
            f"`{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of fold-`3` gross inverse-`pi_hat` lift, "
            f"`{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the final weighted fold-`3` burden, and "
            f"`{_format_percent(report.target_exact_trim_floor_weighted_retention)}` weighted retention after `(1-pi_hat)`"
        )
        implication_line = (
            f"- current Trigger 2 implication: `{report.driver_signature}`; feed real same-seed reruns through this validation-only bridge before claiming seed `303` is landed or spending the queued residual slot on seed `707` / `z = 0.25`"
        )
    return (
        f"- observed same-seed rerun still enters the bridge at `{report.current_rung_status}`: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, witness floor stays `{_format_ninths(report.current_witness_floor)}`, and the ladder still points first to seed `303` / witness / `z = 0.15`",
        bridge_line,
        "- that makes the bridge executable rather than rhetorical: the actionable next slot is still priority `1` / `binding-witness-floor-lift`, while seed `707` / `z = 0.25` stays queued as priority `2` / `residual-total-miss-closure`",
        implication_line,
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_binding_slot_priority_bridge_report(
    *,
    agenda_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
        | None
    ) = None,
    priority_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport:
    resolved_agenda = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda()
        if agenda_report is None
        else agenda_report
    )
    resolved_priority_trace = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace()
        if priority_trace_report is None
        else priority_trace_report
    )

    if resolved_agenda.policy_digest != resolved_priority_trace.policy_digest:
        raise ValueError(
            "observed-rerun binding-slot priority bridge requires shared policy digest"
        )
    if resolved_agenda.binding_design != resolved_priority_trace.binding_design:
        raise ValueError(
            "observed-rerun binding-slot priority bridge requires shared binding design"
        )
    if resolved_agenda.window_label != resolved_priority_trace.window_label:
        raise ValueError(
            "observed-rerun binding-slot priority bridge requires shared window label"
        )
    if (
        resolved_agenda.same_seed_random_states
        != resolved_priority_trace.same_seed_random_states
    ):
        raise ValueError(
            "observed-rerun binding-slot priority bridge requires shared exact same-seed ordering"
        )

    next_slot_fields = _next_required_slot_fields(
        agenda_report=resolved_agenda,
        priority_trace_report=resolved_priority_trace,
    )
    driver_signature = _driver_signature(
        agenda_report=resolved_agenda,
        priority_trace_report=resolved_priority_trace,
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "binding-slot-priority-bridge"
        ),
        policy_digest=resolved_agenda.policy_digest,
        binding_design=resolved_agenda.binding_design,
        window_label=resolved_agenda.window_label,
        same_seed_random_states=resolved_agenda.same_seed_random_states,
        current_rung_status=resolved_agenda.current_rung_status,
        agenda_driver_signature=resolved_agenda.driver_signature,
        binding_slot_priority_driver_signature=resolved_priority_trace.driver_signature,
        current_point_miss_vector=resolved_agenda.current_point_miss_vector,
        current_band_miss_vector=resolved_agenda.current_band_miss_vector,
        current_witness_floor=resolved_agenda.current_witness_floor,
        required_min_witness_floor=resolved_agenda.required_min_witness_floor,
        current_to_next_rung_pointwise_gap_vector=(
            resolved_agenda.current_to_next_rung_pointwise_gap_vector
        ),
        current_to_next_rung_band_gap_vector=(
            resolved_agenda.current_to_next_rung_band_gap_vector
        ),
        current_to_completion_pointwise_gap_vector=(
            resolved_agenda.current_to_completion_pointwise_gap_vector
        ),
        current_to_completion_band_gap_vector=(
            resolved_agenda.current_to_completion_band_gap_vector
        ),
        next_required_slot_random_state=next_slot_fields[0],
        next_required_slot_seed_group=next_slot_fields[1],
        next_required_slot_z_index=next_slot_fields[2],
        next_required_slot_z_value=next_slot_fields[3],
        next_required_slot_repair_stage=next_slot_fields[4],
        next_required_slot_repair_role=next_slot_fields[5],
        next_required_slot_repair_priority=next_slot_fields[6],
        residual_slot_random_state=resolved_priority_trace.residual_slot_random_state,
        residual_slot_seed_group=resolved_priority_trace.residual_slot_seed_group,
        residual_slot_z_index=resolved_priority_trace.residual_slot_z_index,
        residual_slot_z_value=resolved_priority_trace.residual_slot_z_value,
        residual_slot_repair_stage=resolved_priority_trace.residual_slot_repair_stage,
        residual_slot_repair_role=resolved_priority_trace.residual_slot_repair_role,
        residual_slot_repair_priority=resolved_priority_trace.residual_slot_repair_priority,
        target_fold_id=resolved_priority_trace.target_fold_id,
        target_exact_trim_floor_count=resolved_priority_trace.target_exact_trim_floor_count,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            resolved_priority_trace.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            resolved_priority_trace.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            resolved_priority_trace.target_exact_trim_floor_weighted_retention
        ),
        driver_signature=driver_signature,
        canonical_observed_rerun_binding_slot_priority_bridge_digest=(),
    )
    report.canonical_observed_rerun_binding_slot_priority_bridge_digest = (
        _canonical_digest(report=report)
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_binding_slot_priority_bridge() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_binding_slot_priority_bridge_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_binding_slot_priority_bridge_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_binding_slot_priority_bridge",
]
