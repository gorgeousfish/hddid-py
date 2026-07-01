from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCompletionProfileReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot:
    random_state: int
    seed_group: str
    z_index: int
    z_value: float
    repair_stage: str
    repair_order: int
    current_covered: bool
    completion_target_covered: bool
    currently_actionable: bool

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)
        self.repair_stage = str(self.repair_stage).strip().lower()
        self.repair_order = int(self.repair_order)
        self.current_covered = bool(self.current_covered)
        self.completion_target_covered = bool(self.completion_target_covered)
        self.currently_actionable = bool(self.currently_actionable)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    current_rung_status: str
    highest_landed_rung: str
    downstream_evidence_status: str
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    current_to_next_rung_pointwise_gap_vector: tuple[int, int, int]
    current_to_next_rung_band_gap_vector: tuple[int, int, int]
    current_to_completion_pointwise_gap_vector: tuple[int, int, int]
    current_to_completion_band_gap_vector: tuple[int, int, int]
    runtime_witness_path: tuple[str, ...]
    next_required_slot: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot
        | None
    )
    pending_repair_slots: tuple[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot,
        ...,
    ]
    driver_signature: str
    canonical_observed_rerun_repair_agenda_digest: tuple[str, ...]

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
        self.highest_landed_rung = str(self.highest_landed_rung).strip()
        self.downstream_evidence_status = str(self.downstream_evidence_status).strip()
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
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.pending_repair_slots = tuple(self.pending_repair_slots)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_repair_agenda_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_repair_agenda_digest
        )


def _gap_vector(
    current_vector: tuple[int, int, int],
    target_vector: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(
        max(int(current_value) - int(target_value), 0)
        for current_value, target_value in zip(current_vector, target_vector)
    )


def _driver_signature(
    *,
    current_rung_status: str,
    pending_repair_slots: tuple[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot,
        ...,
    ],
) -> str:
    if current_rung_status == "same-seed-exact-witness-residual-slot-completion-landed":
        return "same-seed-exact-witness-observed-rerun-repair-agenda-closed"
    if current_rung_status == "same-seed-exact-witness-binding-slot-progress-landed":
        if (
            len(pending_repair_slots) == 1
            and pending_repair_slots[0].currently_actionable
        ):
            return "same-seed-exact-witness-observed-rerun-repair-agenda-residual-only"
    if current_rung_status == "same-seed-exact-witness-observed-rerun-open":
        if (
            len(pending_repair_slots) == 2
            and pending_repair_slots[0].currently_actionable
        ):
            return "same-seed-exact-witness-observed-rerun-repair-agenda-open"
    return "mixed-same-seed-exact-witness-observed-rerun-repair-agenda"


def _build_slot(
    *,
    after_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    completion_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    random_state: int,
    z_index: int,
    repair_stage: str,
    repair_order: int,
    currently_actionable: bool,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot:
    current_observation = after_report.seed_observation(random_state)
    completion_observation = completion_report.seed_observation(random_state)
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot(
        random_state=random_state,
        seed_group=current_observation.seed_group,
        z_index=z_index,
        z_value=after_report.evaluation_grid[z_index],
        repair_stage=repair_stage,
        repair_order=repair_order,
        current_covered=current_observation.pointwise_coverage_by_z[z_index],
        completion_target_covered=completion_observation.pointwise_coverage_by_z[
            z_index
        ],
        currently_actionable=currently_actionable,
    )


def _pending_slots(
    *,
    after_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    completion_profile_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCompletionProfileReport,
    current_rung_status: str,
) -> tuple[
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot,
    ...,
]:
    completion_report = completion_profile_report.completion_after_report
    if current_rung_status == "same-seed-exact-witness-residual-slot-completion-landed":
        return ()
    if current_rung_status == "same-seed-exact-witness-binding-slot-progress-landed":
        return (
            _build_slot(
                after_report=after_report,
                completion_report=completion_report,
                random_state=707,
                z_index=2,
                repair_stage="residual-slot-completion",
                repair_order=2,
                currently_actionable=True,
            ),
        )
    return (
        _build_slot(
            after_report=after_report,
            completion_report=completion_report,
            random_state=303,
            z_index=1,
            repair_stage="binding-slot-progress",
            repair_order=1,
            currently_actionable=True,
        ),
        _build_slot(
            after_report=after_report,
            completion_report=completion_report,
            random_state=707,
            z_index=2,
            repair_stage="residual-slot-completion",
            repair_order=2,
            currently_actionable=False,
        ),
    )


def _canonical_digest(
    *,
    report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport,
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-closed"
    ):
        return (
            f"- observed exact same-seed rerun now has no remaining completion-facing repairs: point miss is already `{list(report.current_point_miss_vector)}`, band miss stays `{list(report.current_band_miss_vector)}`, and witness floor holds at `8/9 = {_format_float(report.current_witness_floor)}`",
            "- the dynamic repair agenda is therefore empty: both the binding witness-center slot at seed `303` / `z = 0.15` and the residual fresh shoulder at seed `707` / `z = 0.25` are already landed",
            "- bounded runtime discipline is unchanged: the admissible rerun still stays on `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` and leaves the left guard `z = 0.05` untouched",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-repair-agenda-closed`; no further repair agenda needs to be queued before treating this observed rerun as admissible",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-residual-only"
    ):
        return (
            "- observed exact same-seed rerun has already absorbed the binding witness-center repair: seed `303` at witness-center `z = 0.15` is no longer pending, so the repair agenda collapses to the residual fresh shoulder only",
            f"- the remaining gap to the next rung is pointwise `{list(report.current_to_next_rung_pointwise_gap_vector)}` with band `{list(report.current_to_next_rung_band_gap_vector)}`, and the full completion gap matches that same residual-only vector `{list(report.current_to_completion_pointwise_gap_vector)}` / `{list(report.current_to_completion_band_gap_vector)}`",
            "- bounded runtime discipline is unchanged: future reruns must keep the preserve-left-support path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` and leave the left guard `z = 0.05` untouched",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-repair-agenda-residual-only`; route the next observed rerun directly to the actionable seed `707` residual repair",
        )
    return (
        "- observed exact same-seed rerun still has two ordered completion-facing repairs: seed `303` at witness-center `z = 0.15` is actionable now, while seed `707` at fresh right-shoulder `z = 0.25` stays queued behind the binding landing",
        f"- the remaining gap to the next rung is pointwise `{list(report.current_to_next_rung_pointwise_gap_vector)}` with band `{list(report.current_to_next_rung_band_gap_vector)}`, while the remaining gap to full admissible completion is pointwise `{list(report.current_to_completion_pointwise_gap_vector)}` with band `{list(report.current_to_completion_band_gap_vector)}`",
        "- bounded runtime discipline is unchanged: future reruns must keep the preserve-left-support path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` and leave the left guard `z = 0.05` untouched",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-repair-agenda-open`; land the actionable seed `303` slot before spending on the queued residual seed `707` repair",
    )


def _current_witness_floor(
    *,
    current_rung_status: str,
    rung_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport
    ),
    binding_slot_progress_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport
    ),
    residual_slot_completion_profile_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCompletionProfileReport
    ),
) -> float:
    if current_rung_status == "same-seed-exact-witness-binding-slot-progress-landed":
        return binding_slot_progress_report.projected_binding_witness_floor
    if current_rung_status == "same-seed-exact-witness-residual-slot-completion-landed":
        return residual_slot_completion_profile_report.completion_witness_floor
    return binding_slot_progress_report.baseline_witness_floor


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda_report(
    *,
    after_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    before_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    observed_rerun_rung_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport
        | None
    ) = None,
    binding_slot_progress_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport
        | None
    ) = None,
    residual_slot_completion_profile_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCompletionProfileReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport:
    resolved_before = (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        if before_report is None
        else before_report
    )
    resolved_rung_guard = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard_report(
            after_report=after_report,
            before_report=resolved_before,
        )
        if observed_rerun_rung_guard_report is None
        else observed_rerun_rung_guard_report
    )
    resolved_binding_progress = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile()
        if binding_slot_progress_report is None
        else binding_slot_progress_report
    )
    resolved_completion_profile = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile()
        if residual_slot_completion_profile_report is None
        else residual_slot_completion_profile_report
    )

    for peer_report in (resolved_binding_progress, resolved_completion_profile):
        if resolved_rung_guard.policy_digest != peer_report.policy_digest:
            raise ValueError(
                "observed rerun repair agenda requires shared policy digest across rung and profile reports"
            )
        if resolved_rung_guard.binding_design != peer_report.binding_design:
            raise ValueError(
                "observed rerun repair agenda requires shared binding design across rung and profile reports"
            )
        if resolved_rung_guard.window_label != peer_report.window_label:
            raise ValueError(
                "observed rerun repair agenda requires shared window label across rung and profile reports"
            )
        if (
            resolved_rung_guard.same_seed_random_states
            != peer_report.same_seed_random_states
        ):
            raise ValueError(
                "observed rerun repair agenda requires shared exact same-seed ordering across rung and profile reports"
            )

    current_all_summary = after_report.group_summary("all")
    next_rung_summary = (
        current_all_summary
        if resolved_rung_guard.resulting_rung_status
        == "same-seed-exact-witness-residual-slot-completion-landed"
        else resolved_completion_profile.completion_after_report.group_summary("all")
        if resolved_rung_guard.resulting_rung_status
        == "same-seed-exact-witness-binding-slot-progress-landed"
        else resolved_binding_progress.binding_only_after_report.group_summary("all")
    )
    completion_summary = (
        resolved_completion_profile.completion_after_report.group_summary("all")
    )

    pending_repair_slots = _pending_slots(
        after_report=after_report,
        completion_profile_report=resolved_completion_profile,
        current_rung_status=resolved_rung_guard.resulting_rung_status,
    )
    next_required_slot = next(
        (slot for slot in pending_repair_slots if slot.currently_actionable),
        None,
    )
    driver_signature = _driver_signature(
        current_rung_status=resolved_rung_guard.resulting_rung_status,
        pending_repair_slots=pending_repair_slots,
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-observed-rerun-repair-agenda"
        ),
        policy_digest=resolved_rung_guard.policy_digest,
        binding_design=resolved_rung_guard.binding_design,
        window_label=resolved_rung_guard.window_label,
        same_seed_random_states=resolved_rung_guard.same_seed_random_states,
        current_rung_status=resolved_rung_guard.resulting_rung_status,
        highest_landed_rung=resolved_rung_guard.highest_landed_rung,
        downstream_evidence_status=resolved_rung_guard.downstream_evidence_status,
        current_point_miss_vector=resolved_rung_guard.candidate_point_miss_vector,
        current_band_miss_vector=resolved_rung_guard.candidate_band_miss_vector,
        current_witness_floor=_current_witness_floor(
            current_rung_status=resolved_rung_guard.resulting_rung_status,
            rung_guard_report=resolved_rung_guard,
            binding_slot_progress_report=resolved_binding_progress,
            residual_slot_completion_profile_report=resolved_completion_profile,
        ),
        required_min_witness_floor=resolved_rung_guard.required_min_witness_floor,
        current_to_next_rung_pointwise_gap_vector=_gap_vector(
            current_all_summary.point_miss_count_by_z,
            next_rung_summary.point_miss_count_by_z,
        ),
        current_to_next_rung_band_gap_vector=_gap_vector(
            current_all_summary.uniform_band_miss_count_by_z,
            next_rung_summary.uniform_band_miss_count_by_z,
        ),
        current_to_completion_pointwise_gap_vector=_gap_vector(
            current_all_summary.point_miss_count_by_z,
            completion_summary.point_miss_count_by_z,
        ),
        current_to_completion_band_gap_vector=_gap_vector(
            current_all_summary.uniform_band_miss_count_by_z,
            completion_summary.uniform_band_miss_count_by_z,
        ),
        runtime_witness_path=resolved_rung_guard.runtime_witness_path,
        next_required_slot=next_required_slot,
        pending_repair_slots=pending_repair_slots,
        driver_signature=driver_signature,
        canonical_observed_rerun_repair_agenda_digest=(),
    )
    report.canonical_observed_rerun_repair_agenda_digest = _canonical_digest(
        report=report
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda_report(
        after_report=run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport",
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaSlot",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda",
]
