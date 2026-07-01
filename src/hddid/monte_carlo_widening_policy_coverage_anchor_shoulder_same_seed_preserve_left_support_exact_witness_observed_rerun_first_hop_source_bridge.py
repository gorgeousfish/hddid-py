from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_binding_slot_priority_bridge import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_binding_slot_priority_bridge_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard_report,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _slot_matches(
    rung_guard_slot: object | None,
    *,
    random_state: int | None,
    z_index: int | None,
) -> bool:
    if rung_guard_slot is None or random_state is None or z_index is None:
        return rung_guard_slot is None and random_state is None and z_index is None
    return bool(
        getattr(rung_guard_slot, "random_state", None) == int(random_state)
        and getattr(rung_guard_slot, "z_index", None) == int(z_index)
    )


def _driver_signature(
    *,
    rung_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport
    ),
    bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport
    ),
) -> str:
    if (
        bridge_report.binding_slot_priority_driver_signature
        != "same-seed-seed303-fold3-trim-floor-binding-slot-priority-confirmed"
    ):
        return "mixed-same-seed-exact-witness-observed-rerun-first-hop-source-bridge"
    if (
        rung_guard_report.resulting_rung_status
        == "same-seed-exact-witness-observed-rerun-open"
        and bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-open"
        and _slot_matches(
            rung_guard_report.next_required_slot,
            random_state=bridge_report.next_required_slot_random_state,
            z_index=bridge_report.next_required_slot_z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-open"
    if (
        rung_guard_report.resulting_rung_status
        == "same-seed-exact-witness-binding-slot-progress-landed"
        and bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-residual-only"
        and _slot_matches(
            rung_guard_report.next_required_slot,
            random_state=bridge_report.next_required_slot_random_state,
            z_index=bridge_report.next_required_slot_z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-residual-only"
    if (
        rung_guard_report.resulting_rung_status
        == "same-seed-exact-witness-residual-slot-completion-landed"
        and bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-binding-slot-priority-bridge-closed"
        and rung_guard_report.next_required_slot is None
        and bridge_report.next_required_slot_random_state is None
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-closed"
    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-source-bridge"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    binding_slot_landed: bool
    residual_slot_landed: bool
    current_rung_candidate_status: str
    current_rung_status: str
    bridge_driver_signature: str
    binding_slot_priority_driver_signature: str
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
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
    canonical_observed_rerun_first_hop_source_bridge_digest: tuple[str, ...]

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
        self.bridge_driver_signature = str(self.bridge_driver_signature).strip()
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
        self.canonical_observed_rerun_first_hop_source_bridge_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_first_hop_source_bridge_digest
        )


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-closed"
    ):
        return (
            f"- observed same-seed rerun has already consumed the full source-backed ladder: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor now holds at `{_format_ninths(report.current_witness_floor)}`",
            "- the single ladder guard and the seed `303` trim-floor source reason are both fully absorbed: the first-hop source bridge is closed because the residual seed `707` repair is already landed as well",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-source-bridge-closed`; keep this helper validation-only and out of live routing surfaces",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-residual-only"
    ):
        return (
            f"- observed same-seed rerun has already landed the source-backed first hop: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor now sits at `{_format_ninths(report.current_witness_floor)}`",
            "- the seed `303` / witness / `z = 0.15` trim-floor source bridge is therefore consumed, and the only remaining next slot is the queued residual seed `707` / `z = 0.25`",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-source-bridge-residual-only`; keep this helper validation-only and spend the next rerun on the residual slot only",
        )
    return (
        f"- observed same-seed rerun still sits at the single ladder guard baseline: point miss stays `{list(report.current_point_miss_vector)}`, band miss stays `{list(report.current_band_miss_vector)}`, and witness floor therefore remains `{_format_ninths(report.current_witness_floor)}`",
        "- the same first hop is now source-backed in the same object: the single ladder guard still requires seed `303` / witness / `z = 0.15`, and that slot is backed by the unique fold-`3` trim-floor row carrying "
        f"`{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of fold-`3` gross inverse-`pi_hat` lift, "
        f"`{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the final weighted fold-`3` burden, and "
        f"`{_format_percent(report.target_exact_trim_floor_weighted_retention)}` weighted retention after `(1-pi_hat)`",
        "- that removes the last manual join between ladder and source evidence: seed `303` stays priority `1` / `binding-witness-floor-lift`, while seed `707` / `z = 0.25` remains queued as priority `2` / `residual-total-miss-closure`",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-source-bridge-open`; feed real same-seed reruns through this validation-only source bridge before claiming the ladder has moved past seed `303` or spending the queued residual slot on seed `707` / `z = 0.25`",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge_report(
    *,
    after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    before_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    rung_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport
        | None
    ) = None,
    agenda_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
        | None
    ) = None,
    bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunBindingSlotPriorityBridgeReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport:
    resolved_before = (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        if before_report is None
        else before_report
    )
    resolved_after = resolved_before if after_report is None else after_report
    resolved_rung_guard = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard_report(
            after_report=resolved_after,
            before_report=resolved_before,
        )
        if rung_guard_report is None
        else rung_guard_report
    )
    resolved_agenda = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda_report(
            after_report=resolved_after,
            before_report=resolved_before,
            observed_rerun_rung_guard_report=resolved_rung_guard,
        )
        if agenda_report is None
        else agenda_report
    )
    resolved_bridge = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_binding_slot_priority_bridge_report(
            agenda_report=resolved_agenda,
        )
        if bridge_report is None
        else bridge_report
    )

    if resolved_rung_guard.policy_digest != resolved_bridge.policy_digest:
        raise ValueError("first-hop source bridge requires shared policy digest")
    if resolved_rung_guard.binding_design != resolved_bridge.binding_design:
        raise ValueError("first-hop source bridge requires shared binding design")
    if resolved_rung_guard.window_label != resolved_bridge.window_label:
        raise ValueError("first-hop source bridge requires shared window label")
    if (
        resolved_rung_guard.same_seed_random_states
        != resolved_bridge.same_seed_random_states
    ):
        raise ValueError(
            "first-hop source bridge requires shared exact same-seed ordering"
        )
    if (
        resolved_rung_guard.candidate_point_miss_vector
        != resolved_bridge.current_point_miss_vector
    ):
        raise ValueError("first-hop source bridge requires point-miss sync")
    if (
        resolved_rung_guard.candidate_band_miss_vector
        != resolved_bridge.current_band_miss_vector
    ):
        raise ValueError("first-hop source bridge requires band-miss sync")
    if (
        resolved_rung_guard.required_min_witness_floor
        != resolved_bridge.required_min_witness_floor
    ):
        raise ValueError("first-hop source bridge requires witness-threshold sync")
    if not _slot_matches(
        resolved_rung_guard.residual_repair_slot,
        random_state=resolved_bridge.residual_slot_random_state,
        z_index=resolved_bridge.residual_slot_z_index,
    ):
        raise ValueError("first-hop source bridge requires residual-slot sync")
    if not _slot_matches(
        resolved_rung_guard.next_required_slot,
        random_state=resolved_bridge.next_required_slot_random_state,
        z_index=resolved_bridge.next_required_slot_z_index,
    ):
        raise ValueError("first-hop source bridge requires next-slot sync")

    driver_signature = _driver_signature(
        rung_guard_report=resolved_rung_guard,
        bridge_report=resolved_bridge,
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "first-hop-source-bridge"
        ),
        policy_digest=resolved_rung_guard.policy_digest,
        binding_design=resolved_rung_guard.binding_design,
        window_label=resolved_rung_guard.window_label,
        same_seed_random_states=resolved_rung_guard.same_seed_random_states,
        binding_slot_landed=resolved_rung_guard.binding_slot_landed,
        residual_slot_landed=resolved_rung_guard.residual_slot_landed,
        current_rung_candidate_status=resolved_rung_guard.current_rung_candidate_status,
        current_rung_status=resolved_rung_guard.resulting_rung_status,
        bridge_driver_signature=resolved_bridge.driver_signature,
        binding_slot_priority_driver_signature=(
            resolved_bridge.binding_slot_priority_driver_signature
        ),
        current_point_miss_vector=resolved_rung_guard.candidate_point_miss_vector,
        current_band_miss_vector=resolved_rung_guard.candidate_band_miss_vector,
        current_witness_floor=resolved_bridge.current_witness_floor,
        required_min_witness_floor=resolved_bridge.required_min_witness_floor,
        next_required_slot_random_state=resolved_bridge.next_required_slot_random_state,
        next_required_slot_seed_group=resolved_bridge.next_required_slot_seed_group,
        next_required_slot_z_index=resolved_bridge.next_required_slot_z_index,
        next_required_slot_z_value=resolved_bridge.next_required_slot_z_value,
        next_required_slot_repair_stage=resolved_bridge.next_required_slot_repair_stage,
        next_required_slot_repair_role=resolved_bridge.next_required_slot_repair_role,
        next_required_slot_repair_priority=(
            resolved_bridge.next_required_slot_repair_priority
        ),
        residual_slot_random_state=resolved_bridge.residual_slot_random_state,
        residual_slot_seed_group=resolved_bridge.residual_slot_seed_group,
        residual_slot_z_index=resolved_bridge.residual_slot_z_index,
        residual_slot_z_value=resolved_bridge.residual_slot_z_value,
        residual_slot_repair_stage=resolved_bridge.residual_slot_repair_stage,
        residual_slot_repair_role=resolved_bridge.residual_slot_repair_role,
        residual_slot_repair_priority=resolved_bridge.residual_slot_repair_priority,
        target_fold_id=resolved_bridge.target_fold_id,
        target_exact_trim_floor_count=resolved_bridge.target_exact_trim_floor_count,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            resolved_bridge.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            resolved_bridge.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            resolved_bridge.target_exact_trim_floor_weighted_retention
        ),
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_source_bridge_digest=(),
    )
    report.canonical_observed_rerun_first_hop_source_bridge_digest = _canonical_digest(
        report=report
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge",
]
