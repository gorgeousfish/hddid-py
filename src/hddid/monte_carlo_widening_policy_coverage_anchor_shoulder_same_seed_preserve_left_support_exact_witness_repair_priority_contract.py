from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_candidate_profile import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessCandidateProfileReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_candidate_profile,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapReport,
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapSlot,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPrioritySlot:
    random_state: int
    seed_group: str
    z_index: int
    z_value: float
    repair_role: str
    repair_priority: int

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)
        self.repair_role = str(self.repair_role).strip().lower()
        self.repair_priority = int(self.repair_priority)


def _repair_slot(
    slot: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapSlot
    ),
    *,
    repair_role: str,
    repair_priority: int,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPrioritySlot:
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPrioritySlot(
        random_state=slot.random_state,
        seed_group=slot.seed_group,
        z_index=slot.z_index,
        z_value=slot.z_value,
        repair_role=repair_role,
        repair_priority=repair_priority,
    )


def _binding_and_residual_slots(
    gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapReport
    ),
) -> tuple[
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPrioritySlot,
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPrioritySlot,
]:
    witness_slots = tuple(
        slot
        for slot in gap_report.pending_pointwise_repairs
        if slot.seed_group == "witness"
    )
    fresh_slots = tuple(
        slot
        for slot in gap_report.pending_pointwise_repairs
        if slot.seed_group == "fresh"
    )
    if len(witness_slots) != 1 or len(fresh_slots) != 1:
        raise ValueError(
            "exact witness repair priority contract requires exactly one witness repair slot and one fresh repair slot"
        )
    binding_slot = _repair_slot(
        witness_slots[0],
        repair_role="binding-witness-floor-lift",
        repair_priority=1,
    )
    residual_slot = _repair_slot(
        fresh_slots[0],
        repair_role="residual-total-miss-closure",
        repair_priority=2,
    )
    return binding_slot, residual_slot


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPriorityContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    current_point_miss_vector: tuple[int, int, int]
    target_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    target_band_miss_vector: tuple[int, int, int]
    runtime_witness_path: tuple[str, ...]
    left_guard_grid_value: float
    binding_repair_slot: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPrioritySlot
    residual_repair_slot: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPrioritySlot
    repair_priority_order: tuple[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPrioritySlot,
        ...,
    ]
    binding_witness_floor_increment: float
    residual_witness_floor_increment: float
    min_required_total_point_miss_reduction: int
    repair_only_within_residual_lane: bool
    driver_signature: str
    canonical_exact_witness_repair_priority_digest: tuple[str, ...]

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
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.target_point_miss_vector = tuple(
            int(value) for value in self.target_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.target_band_miss_vector = tuple(
            int(value) for value in self.target_band_miss_vector
        )
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.left_guard_grid_value = float(self.left_guard_grid_value)
        self.repair_priority_order = tuple(self.repair_priority_order)
        self.binding_witness_floor_increment = float(
            self.binding_witness_floor_increment
        )
        self.residual_witness_floor_increment = float(
            self.residual_witness_floor_increment
        )
        self.min_required_total_point_miss_reduction = int(
            self.min_required_total_point_miss_reduction
        )
        self.repair_only_within_residual_lane = bool(
            self.repair_only_within_residual_lane
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_exact_witness_repair_priority_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_exact_witness_repair_priority_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract_report(
    *,
    gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapReport
        | None
    ) = None,
    candidate_profile_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessCandidateProfileReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPriorityContractReport:
    resolved_gap = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap()
        if gap_report is None
        else gap_report
    )
    resolved_candidate_profile = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_candidate_profile()
        if candidate_profile_report is None
        else candidate_profile_report
    )

    if resolved_gap.policy_digest != resolved_candidate_profile.policy_digest:
        raise ValueError(
            "exact witness repair priority contract requires shared policy digest"
        )
    if resolved_gap.binding_design != resolved_candidate_profile.binding_design:
        raise ValueError(
            "exact witness repair priority contract requires shared binding design"
        )
    if resolved_gap.window_label != resolved_candidate_profile.window_label:
        raise ValueError(
            "exact witness repair priority contract requires shared window label"
        )
    if (
        resolved_gap.same_seed_random_states
        != resolved_candidate_profile.same_seed_random_states
    ):
        raise ValueError(
            "exact witness repair priority contract requires shared exact same-seed ordering"
        )
    if resolved_gap.driver_signature != "same-seed-exact-witness-target-gap-open":
        raise ValueError(
            "exact witness repair priority contract requires the current exact-witness target gap to remain open"
        )

    binding_slot, residual_slot = _binding_and_residual_slots(resolved_gap)
    binding_witness_floor_increment = (
        resolved_candidate_profile.target_witness_floor
        - resolved_candidate_profile.baseline_witness_floor
    )
    residual_witness_floor_increment = 0.0
    repair_only_within_residual_lane = bool(
        binding_slot.z_value in resolved_candidate_profile.residual_lane_grid_values
        and residual_slot.z_value
        in resolved_candidate_profile.residual_lane_grid_values
        and resolved_gap.left_guard_already_matches_target
    )
    canonical_digest = (
        "- exact-witness repair priority is now fully localized: seed `303` at witness-center `z = 0.15` is the unique binding slot because it is the only pending repair that directly lifts the witness group from `[1, 2, 0]` to `[1, 1, 0]`",
        "- seed `707` at fresh right-shoulder `z = 0.25` remains necessary but secondary: it closes the residual all-seed point-miss gap from `[1, 2, 2]` to `[1, 2, 1]` without changing witness-floor arithmetic",
        "- bounded object flow and left guard remain fixed: future estimator evidence must stay on `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`, keep all new repair spend inside `z in {0.15, 0.25}`, and preserve `z = 0.05`",
        "- current Trigger 2 implication: `same-seed-exact-witness-repair-priority-contract`; next real rerun should prioritize the binding witness-center slot before spending on the residual fresh shoulder",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPriorityContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-repair-priority-contract"
        ),
        policy_digest=resolved_gap.policy_digest,
        binding_design=resolved_gap.binding_design,
        window_label=resolved_gap.window_label,
        same_seed_random_states=resolved_gap.same_seed_random_states,
        current_point_miss_vector=resolved_gap.current_point_miss_vector,
        target_point_miss_vector=resolved_gap.target_point_miss_vector,
        current_band_miss_vector=resolved_gap.current_band_miss_vector,
        target_band_miss_vector=resolved_gap.target_band_miss_vector,
        runtime_witness_path=resolved_gap.runtime_witness_path,
        left_guard_grid_value=resolved_gap.left_guard_grid_value,
        binding_repair_slot=binding_slot,
        residual_repair_slot=residual_slot,
        repair_priority_order=(binding_slot, residual_slot),
        binding_witness_floor_increment=binding_witness_floor_increment,
        residual_witness_floor_increment=residual_witness_floor_increment,
        min_required_total_point_miss_reduction=resolved_gap.remaining_total_point_miss_reduction,
        repair_only_within_residual_lane=repair_only_within_residual_lane,
        driver_signature="same-seed-exact-witness-repair-priority-contract",
        canonical_exact_witness_repair_priority_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPriorityContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPriorityContractReport",
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPrioritySlot",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract",
]
