from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport,
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPriorityContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorInversePiConcentrationTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _pending_slot(
    *,
    agenda_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
    ),
    random_state: int,
    z_index: int,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot:
    for slot in agenda_report.pending_repair_slots:
        if slot.random_state == int(random_state) and slot.z_index == int(z_index):
            return slot
    raise ValueError("binding-slot priority trace requires the requested pending slot")


def _driver_signature(
    *,
    inverse_pi_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorInversePiConcentrationTraceReport
    ),
    agenda_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
    ),
    priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPriorityContractReport
    ),
    seed707_exact_trim_floor_count: int,
    seed707_fold3_inverse_pi_phi1_center_projection: float,
) -> str:
    if (
        inverse_pi_report.driver_signature
        == "same-seed-seed303-fold3-trim-floor-inverse-pi-concentration-confirmed"
        and agenda_report.current_rung_status
        == "same-seed-exact-witness-observed-rerun-open"
        and priority_report.driver_signature
        == "same-seed-exact-witness-repair-priority-contract"
        and agenda_report.next_required_slot is not None
        and agenda_report.next_required_slot.random_state
        == priority_report.binding_repair_slot.random_state
        and agenda_report.next_required_slot.z_index
        == priority_report.binding_repair_slot.z_index
        and priority_report.binding_repair_slot.random_state == 303
        and priority_report.binding_repair_slot.repair_priority == 1
        and priority_report.residual_repair_slot.random_state == 707
        and priority_report.residual_repair_slot.repair_priority == 2
        and priority_report.binding_witness_floor_increment > 0.1
        and priority_report.residual_witness_floor_increment == 0.0
        and inverse_pi_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        > 0.4
        and inverse_pi_report.target_exact_trim_floor_weighted_share_of_fold3_weighted
        > 0.4
        and seed707_exact_trim_floor_count == 0
        and seed707_fold3_inverse_pi_phi1_center_projection < 0.0
    ):
        return "same-seed-seed303-fold3-trim-floor-binding-slot-priority-confirmed"
    return "mixed-same-seed-seed303-fold3-trim-floor-binding-slot-priority-trace"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    same_seed_random_states: tuple[int, ...]
    current_rung_status: str
    repair_priority_driver_signature: str
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    current_to_next_rung_pointwise_gap_vector: tuple[int, int, int]
    current_to_next_rung_band_gap_vector: tuple[int, int, int]
    current_to_completion_pointwise_gap_vector: tuple[int, int, int]
    current_to_completion_band_gap_vector: tuple[int, int, int]
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_index: int
    binding_slot_z_value: float
    binding_slot_repair_stage: str
    binding_slot_repair_role: str
    binding_slot_repair_priority: int
    binding_witness_floor_increment: float
    residual_slot_random_state: int
    residual_slot_seed_group: str
    residual_slot_z_index: int
    residual_slot_z_value: float
    residual_slot_repair_stage: str
    residual_slot_repair_role: str
    residual_slot_repair_priority: int
    residual_witness_floor_increment: float
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    comparator_random_states: tuple[int, ...]
    comparator_exact_trim_floor_counts: tuple[int, ...]
    comparator_fold3_inverse_pi_phi1_center_projections: tuple[float, ...]
    driver_signature: str
    canonical_seed303_low_pi_fold3_trim_floor_binding_slot_priority_digest: tuple[
        str, ...
    ]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.focus_random_states = tuple(
            int(value) for value in self.focus_random_states
        )
        self.same_seed_random_states = tuple(
            int(value) for value in self.same_seed_random_states
        )
        self.current_rung_status = str(self.current_rung_status).strip()
        self.repair_priority_driver_signature = str(
            self.repair_priority_driver_signature
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
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_seed_group = str(self.binding_slot_seed_group).strip().lower()
        self.binding_slot_z_index = int(self.binding_slot_z_index)
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.binding_slot_repair_stage = str(self.binding_slot_repair_stage).strip()
        self.binding_slot_repair_role = str(self.binding_slot_repair_role).strip()
        self.binding_slot_repair_priority = int(self.binding_slot_repair_priority)
        self.binding_witness_floor_increment = float(
            self.binding_witness_floor_increment
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
        self.residual_witness_floor_increment = float(
            self.residual_witness_floor_increment
        )
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
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_exact_trim_floor_counts = tuple(
            int(value) for value in self.comparator_exact_trim_floor_counts
        )
        self.comparator_fold3_inverse_pi_phi1_center_projections = tuple(
            float(value)
            for value in self.comparator_fold3_inverse_pi_phi1_center_projections
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_low_pi_fold3_trim_floor_binding_slot_priority_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_low_pi_fold3_trim_floor_binding_slot_priority_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport
):
    inverse_pi_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace()
    agenda_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda()
    priority_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract()

    if inverse_pi_report.policy_digest != agenda_report.policy_digest:
        raise ValueError("binding-slot priority trace requires shared policy digest")
    if inverse_pi_report.policy_digest != priority_report.policy_digest:
        raise ValueError("binding-slot priority trace requires priority policy sync")
    if inverse_pi_report.binding_design != agenda_report.binding_design:
        raise ValueError("binding-slot priority trace requires shared binding design")
    if inverse_pi_report.binding_design != priority_report.binding_design:
        raise ValueError("binding-slot priority trace requires priority design sync")
    if inverse_pi_report.window_label != agenda_report.window_label:
        raise ValueError("binding-slot priority trace requires shared window label")
    if inverse_pi_report.window_label != priority_report.window_label:
        raise ValueError("binding-slot priority trace requires priority window sync")
    if priority_report.same_seed_random_states != agenda_report.same_seed_random_states:
        raise ValueError(
            "binding-slot priority trace requires shared exact same-seed ordering"
        )
    if agenda_report.next_required_slot is None:
        raise ValueError("binding-slot priority trace requires an actionable next slot")

    binding_agenda_slot = _pending_slot(
        agenda_report=agenda_report,
        random_state=priority_report.binding_repair_slot.random_state,
        z_index=priority_report.binding_repair_slot.z_index,
    )
    residual_agenda_slot = _pending_slot(
        agenda_report=agenda_report,
        random_state=priority_report.residual_repair_slot.random_state,
        z_index=priority_report.residual_repair_slot.z_index,
    )
    if (
        agenda_report.next_required_slot.random_state
        != binding_agenda_slot.random_state
    ):
        raise ValueError("binding-slot priority trace requires seed303 to stay next")
    if agenda_report.next_required_slot.z_index != binding_agenda_slot.z_index:
        raise ValueError("binding-slot priority trace requires the same binding z slot")

    seed707_index = inverse_pi_report.comparator_random_states.index(
        priority_report.residual_repair_slot.random_state
    )
    seed707_exact_trim_floor_count = (
        inverse_pi_report.comparator_exact_trim_floor_counts[seed707_index]
    )
    seed707_fold3_inverse_pi_phi1_center_projection = (
        inverse_pi_report.comparator_fold3_inverse_pi_phi1_center_projections[
            seed707_index
        ]
    )

    driver_signature = _driver_signature(
        inverse_pi_report=inverse_pi_report,
        agenda_report=agenda_report,
        priority_report=priority_report,
        seed707_exact_trim_floor_count=seed707_exact_trim_floor_count,
        seed707_fold3_inverse_pi_phi1_center_projection=(
            seed707_fold3_inverse_pi_phi1_center_projection
        ),
    )
    canonical_digest = (
        f"- observed same-seed ladder still stays `same-seed-exact-witness-observed-rerun-open`: point miss is `{list(agenda_report.current_point_miss_vector)}`, band miss is `{list(agenda_report.current_band_miss_vector)}`, witness floor stays `{_format_ninths(agenda_report.current_witness_floor)}`, and the next actionable slot is seed `303` / witness / `z = 0.15` before the queued residual seed `707` / `z = 0.25`",
        "- that queue order is now backed by the fold-`3` trim-floor conduit rather than generic witness prose: seed `303` keeps exactly one trim-floor row, and that single row already carries "
        f"`{_format_percent(inverse_pi_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of fold-`3` gross inverse-`pi_hat` lift plus "
        f"`{_format_percent(inverse_pi_report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the final weighted fold-`3` nuisance burden while retaining "
        f"`{_format_percent(inverse_pi_report.target_exact_trim_floor_weighted_retention)}` of its own gross lift after `(1-pi_hat)`",
        "- the repair-priority contract matches the same ladder: seed `303` / `z = 0.15` is priority `1` and is the only pending repair that lifts witness floor by "
        f"`{_format_ninths(priority_report.binding_witness_floor_increment)}`, whereas seed `707` / `z = 0.25` stays priority `2` with residual witness-floor increment "
        f"`{_format_float(priority_report.residual_witness_floor_increment)}`",
        "- comparator seed `202` and residual seed `707` still have no exact trim-floor row in the same fold-`3` low-`pi_hat` treated slice, and seed `707` keeps the full fold-`3` gross inverse-`pi_hat` conduit negative at "
        f"`{_format_float(seed707_fold3_inverse_pi_phi1_center_projection)}`; current Trigger 2 implication: `{driver_signature}`, so the next bounded repair should feed the real same-seed observed rerun through the ladder guard using this seed `303` trim-floor binding-slot priority bridge before spending the queued residual slot on seed `707` / `z = 0.25`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-low-pi-fold3-trim-floor-"
            "binding-slot-priority-trace"
        ),
        policy_digest=agenda_report.policy_digest,
        binding_design=agenda_report.binding_design,
        window_label=agenda_report.window_label,
        focus_random_states=inverse_pi_report.focus_random_states,
        same_seed_random_states=agenda_report.same_seed_random_states,
        current_rung_status=agenda_report.current_rung_status,
        repair_priority_driver_signature=priority_report.driver_signature,
        current_point_miss_vector=agenda_report.current_point_miss_vector,
        current_band_miss_vector=agenda_report.current_band_miss_vector,
        current_witness_floor=agenda_report.current_witness_floor,
        required_min_witness_floor=agenda_report.required_min_witness_floor,
        current_to_next_rung_pointwise_gap_vector=(
            agenda_report.current_to_next_rung_pointwise_gap_vector
        ),
        current_to_next_rung_band_gap_vector=(
            agenda_report.current_to_next_rung_band_gap_vector
        ),
        current_to_completion_pointwise_gap_vector=(
            agenda_report.current_to_completion_pointwise_gap_vector
        ),
        current_to_completion_band_gap_vector=(
            agenda_report.current_to_completion_band_gap_vector
        ),
        binding_slot_random_state=binding_agenda_slot.random_state,
        binding_slot_seed_group=binding_agenda_slot.seed_group,
        binding_slot_z_index=binding_agenda_slot.z_index,
        binding_slot_z_value=binding_agenda_slot.z_value,
        binding_slot_repair_stage=binding_agenda_slot.repair_stage,
        binding_slot_repair_role=priority_report.binding_repair_slot.repair_role,
        binding_slot_repair_priority=priority_report.binding_repair_slot.repair_priority,
        binding_witness_floor_increment=priority_report.binding_witness_floor_increment,
        residual_slot_random_state=residual_agenda_slot.random_state,
        residual_slot_seed_group=residual_agenda_slot.seed_group,
        residual_slot_z_index=residual_agenda_slot.z_index,
        residual_slot_z_value=residual_agenda_slot.z_value,
        residual_slot_repair_stage=residual_agenda_slot.repair_stage,
        residual_slot_repair_role=priority_report.residual_repair_slot.repair_role,
        residual_slot_repair_priority=(
            priority_report.residual_repair_slot.repair_priority
        ),
        residual_witness_floor_increment=(
            priority_report.residual_witness_floor_increment
        ),
        target_fold_id=inverse_pi_report.target_fold_id,
        target_exact_trim_floor_count=inverse_pi_report.target_exact_trim_floor_count,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            inverse_pi_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            inverse_pi_report.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            inverse_pi_report.target_exact_trim_floor_weighted_retention
        ),
        comparator_random_states=inverse_pi_report.comparator_random_states,
        comparator_exact_trim_floor_counts=(
            inverse_pi_report.comparator_exact_trim_floor_counts
        ),
        comparator_fold3_inverse_pi_phi1_center_projections=(
            inverse_pi_report.comparator_fold3_inverse_pi_phi1_center_projections
        ),
        driver_signature=driver_signature,
        canonical_seed303_low_pi_fold3_trim_floor_binding_slot_priority_digest=(
            canonical_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace_report()
