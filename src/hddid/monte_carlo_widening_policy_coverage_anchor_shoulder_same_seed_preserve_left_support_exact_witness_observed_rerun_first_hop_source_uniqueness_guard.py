from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
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
    expected_random_state: int,
    expected_z_index: int,
) -> bool:
    return bool(
        random_state is not None
        and z_index is not None
        and int(random_state) == int(expected_random_state)
        and int(z_index) == int(expected_z_index)
    )


def _uniqueness_condition(
    *,
    priority_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport
    ),
) -> bool:
    return bool(
        priority_trace_report.driver_signature
        == "same-seed-seed303-fold3-trim-floor-binding-slot-priority-confirmed"
        and priority_trace_report.target_exact_trim_floor_count == 1
        and tuple(priority_trace_report.comparator_exact_trim_floor_counts) == (0, 0)
        and all(
            float(value) < 0.0
            for value in priority_trace_report.comparator_fold3_inverse_pi_phi1_center_projections
        )
    )


def _driver_signature(
    *,
    source_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport
    ),
    priority_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport
    ),
) -> str:
    if not _uniqueness_condition(priority_trace_report=priority_trace_report):
        return (
            "mixed-same-seed-exact-witness-observed-rerun-first-hop-source-"
            "uniqueness-guard"
        )
    if (
        source_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-open"
        and _slot_matches(
            random_state=source_bridge_report.next_required_slot_random_state,
            z_index=source_bridge_report.next_required_slot_z_index,
            expected_random_state=priority_trace_report.binding_slot_random_state,
            expected_z_index=priority_trace_report.binding_slot_z_index,
        )
    ):
        return (
            "same-seed-exact-witness-observed-rerun-first-hop-source-"
            "uniqueness-guard-open"
        )
    if (
        source_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-residual-only"
        and _slot_matches(
            random_state=source_bridge_report.next_required_slot_random_state,
            z_index=source_bridge_report.next_required_slot_z_index,
            expected_random_state=priority_trace_report.residual_slot_random_state,
            expected_z_index=priority_trace_report.residual_slot_z_index,
        )
    ):
        return (
            "same-seed-exact-witness-observed-rerun-first-hop-source-"
            "uniqueness-guard-residual-only"
        )
    if (
        source_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-closed"
        and source_bridge_report.next_required_slot_random_state is None
        and source_bridge_report.next_required_slot_z_index is None
    ):
        return (
            "same-seed-exact-witness-observed-rerun-first-hop-source-"
            "uniqueness-guard-closed"
        )
    return (
        "mixed-same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard"
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    binding_slot_landed: bool
    residual_slot_landed: bool
    current_rung_candidate_status: str
    current_rung_status: str
    first_hop_source_bridge_driver_signature: str
    binding_slot_priority_driver_signature: str
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_index: int
    binding_slot_z_value: float
    binding_slot_repair_stage: str
    binding_slot_repair_role: str
    binding_slot_repair_priority: int
    binding_witness_floor_increment: float
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
    residual_witness_floor_increment: float
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    comparator_random_states: tuple[int, int]
    comparator_exact_trim_floor_counts: tuple[int, int]
    comparator_fold3_inverse_pi_phi1_center_projections: tuple[float, float]
    driver_signature: str
    canonical_observed_rerun_first_hop_source_uniqueness_guard_digest: tuple[str, ...]

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
        self.binding_slot_landed = bool(self.binding_slot_landed)
        self.residual_slot_landed = bool(self.residual_slot_landed)
        self.current_rung_candidate_status = str(
            self.current_rung_candidate_status
        ).strip()
        self.current_rung_status = str(self.current_rung_status).strip()
        self.first_hop_source_bridge_driver_signature = str(
            self.first_hop_source_bridge_driver_signature
        ).strip()
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
        self.canonical_observed_rerun_first_hop_source_uniqueness_guard_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_first_hop_source_uniqueness_guard_digest
        )


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-closed"
    ):
        return (
            f"- observed same-seed rerun has already consumed the unique seed `303` first hop and the queued residual slot: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor already holds at `{_format_ninths(report.current_witness_floor)}`",
            "- the uniqueness guard is therefore fully absorbed: seed `303` no longer governs an open queue, and seed `707` is no longer pending either",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-closed`; keep this helper validation-only and out of live routing surfaces",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-residual-only"
    ):
        return (
            f"- observed same-seed rerun has already consumed the unique seed `303` first hop: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor already sits at `{_format_ninths(report.current_witness_floor)}`",
            "- the uniqueness evidence is now provenance rather than the active queue: seed `303` was still the only exact fold-`3` trim-floor row, while comparator seed `202` and residual seed `707` kept zero exact trim-floor rows and negative full fold-`3` gross inverse-`pi_hat` conduits",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-residual-only`; keep this helper validation-only and spend the next rerun on the queued residual seed `707` / `z = 0.25` only",
        )
    return (
        f"- observed same-seed rerun still sits below the first executable same-seed rung: point miss stays `{list(report.current_point_miss_vector)}`, band miss stays `{list(report.current_band_miss_vector)}`, and witness floor therefore remains `{_format_ninths(report.current_witness_floor)}`",
        "- the first hop is now machine-readably unique rather than merely prioritized: seed `303` / witness / `z = 0.15` keeps the only exact fold-`3` trim-floor row, worth "
        f"`{_format_ninths(report.binding_witness_floor_increment)}` witness-floor lift, while comparator seed `202` and residual seed `707` both keep zero exact trim-floor rows and negative full fold-`3` gross inverse-`pi_hat` conduits at "
        f"`{_format_float(report.comparator_fold3_inverse_pi_phi1_center_projections[0])}` and "
        f"`{_format_float(report.comparator_fold3_inverse_pi_phi1_center_projections[1])}`",
        "- that uniqueness blocks queue drift inside the observed-rerun ladder: seed `303` remains the only pending `binding-witness-floor-lift`, while seed `707` / `z = 0.25` stays queued as residual-only cleanup with witness-floor increment "
        f"`{_format_float(report.residual_witness_floor_increment)}`",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-open`; feed real same-seed reruns through this validation-only uniqueness guard before claiming seed `303` is landed or spending the queued residual slot on seed `707` / `z = 0.25`",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard_report(
    *,
    after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    before_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    source_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport
        | None
    ) = None,
    priority_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorBindingSlotPriorityTraceReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport:
    resolved_source_bridge = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge_report(
            after_report=after_report,
            before_report=before_report,
        )
        if source_bridge_report is None
        else source_bridge_report
    )
    resolved_priority_trace = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace()
        if priority_trace_report is None
        else priority_trace_report
    )

    if resolved_source_bridge.policy_digest != resolved_priority_trace.policy_digest:
        raise ValueError(
            "first-hop source uniqueness guard requires shared policy digest"
        )
    if resolved_source_bridge.binding_design != resolved_priority_trace.binding_design:
        raise ValueError(
            "first-hop source uniqueness guard requires shared binding design"
        )
    if resolved_source_bridge.window_label != resolved_priority_trace.window_label:
        raise ValueError(
            "first-hop source uniqueness guard requires shared window label"
        )
    if (
        resolved_source_bridge.same_seed_random_states
        != resolved_priority_trace.same_seed_random_states
    ):
        raise ValueError(
            "first-hop source uniqueness guard requires shared exact same-seed ordering"
        )
    if (
        resolved_source_bridge.residual_slot_random_state
        != resolved_priority_trace.residual_slot_random_state
    ):
        raise ValueError(
            "first-hop source uniqueness guard requires seed707 residual-slot sync"
        )
    if (
        resolved_source_bridge.residual_slot_z_index
        != resolved_priority_trace.residual_slot_z_index
    ):
        raise ValueError(
            "first-hop source uniqueness guard requires seed707 residual z-index sync"
        )

    driver_signature = _driver_signature(
        source_bridge_report=resolved_source_bridge,
        priority_trace_report=resolved_priority_trace,
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "first-hop-source-uniqueness-guard"
        ),
        policy_digest=resolved_source_bridge.policy_digest,
        binding_design=resolved_source_bridge.binding_design,
        window_label=resolved_source_bridge.window_label,
        same_seed_random_states=resolved_source_bridge.same_seed_random_states,
        binding_slot_landed=resolved_source_bridge.binding_slot_landed,
        residual_slot_landed=resolved_source_bridge.residual_slot_landed,
        current_rung_candidate_status=resolved_source_bridge.current_rung_candidate_status,
        current_rung_status=resolved_source_bridge.current_rung_status,
        first_hop_source_bridge_driver_signature=(
            resolved_source_bridge.driver_signature
        ),
        binding_slot_priority_driver_signature=(
            resolved_priority_trace.driver_signature
        ),
        current_point_miss_vector=resolved_source_bridge.current_point_miss_vector,
        current_band_miss_vector=resolved_source_bridge.current_band_miss_vector,
        current_witness_floor=resolved_source_bridge.current_witness_floor,
        required_min_witness_floor=resolved_source_bridge.required_min_witness_floor,
        binding_slot_random_state=resolved_priority_trace.binding_slot_random_state,
        binding_slot_seed_group=resolved_priority_trace.binding_slot_seed_group,
        binding_slot_z_index=resolved_priority_trace.binding_slot_z_index,
        binding_slot_z_value=resolved_priority_trace.binding_slot_z_value,
        binding_slot_repair_stage=resolved_priority_trace.binding_slot_repair_stage,
        binding_slot_repair_role=resolved_priority_trace.binding_slot_repair_role,
        binding_slot_repair_priority=resolved_priority_trace.binding_slot_repair_priority,
        binding_witness_floor_increment=(
            resolved_priority_trace.binding_witness_floor_increment
        ),
        next_required_slot_random_state=(
            resolved_source_bridge.next_required_slot_random_state
        ),
        next_required_slot_seed_group=(
            resolved_source_bridge.next_required_slot_seed_group
        ),
        next_required_slot_z_index=resolved_source_bridge.next_required_slot_z_index,
        next_required_slot_z_value=resolved_source_bridge.next_required_slot_z_value,
        next_required_slot_repair_stage=(
            resolved_source_bridge.next_required_slot_repair_stage
        ),
        next_required_slot_repair_role=(
            resolved_source_bridge.next_required_slot_repair_role
        ),
        next_required_slot_repair_priority=(
            resolved_source_bridge.next_required_slot_repair_priority
        ),
        residual_slot_random_state=resolved_source_bridge.residual_slot_random_state,
        residual_slot_seed_group=resolved_source_bridge.residual_slot_seed_group,
        residual_slot_z_index=resolved_source_bridge.residual_slot_z_index,
        residual_slot_z_value=resolved_source_bridge.residual_slot_z_value,
        residual_slot_repair_stage=resolved_source_bridge.residual_slot_repair_stage,
        residual_slot_repair_role=resolved_source_bridge.residual_slot_repair_role,
        residual_slot_repair_priority=resolved_source_bridge.residual_slot_repair_priority,
        residual_witness_floor_increment=(
            resolved_priority_trace.residual_witness_floor_increment
        ),
        target_fold_id=resolved_priority_trace.target_fold_id,
        target_exact_trim_floor_count=(
            resolved_priority_trace.target_exact_trim_floor_count
        ),
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            resolved_priority_trace.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            resolved_priority_trace.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            resolved_priority_trace.target_exact_trim_floor_weighted_retention
        ),
        comparator_random_states=resolved_priority_trace.comparator_random_states,
        comparator_exact_trim_floor_counts=(
            resolved_priority_trace.comparator_exact_trim_floor_counts
        ),
        comparator_fold3_inverse_pi_phi1_center_projections=(
            resolved_priority_trace.comparator_fold3_inverse_pi_phi1_center_projections
        ),
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_source_uniqueness_guard_digest=(),
    )
    report.canonical_observed_rerun_first_hop_source_uniqueness_guard_digest = (
        _canonical_digest(report=report)
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard_report()
