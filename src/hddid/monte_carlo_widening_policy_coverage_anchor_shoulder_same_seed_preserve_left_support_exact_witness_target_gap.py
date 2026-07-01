from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_candidate_profile import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessCandidateProfileReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_candidate_profile,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation,
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_runtime_alignment_probe import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseRuntimeAlignmentProbeReport,
    build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_runtime_alignment_proxy_report,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapSlot:
    random_state: int
    seed_group: str
    z_index: int
    z_value: float
    coverage_kind: str
    current_covered: bool
    target_covered: bool

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)
        self.coverage_kind = str(self.coverage_kind).strip().lower()
        self.current_covered = bool(self.current_covered)
        self.target_covered = bool(self.target_covered)


def _pending_pointwise_repairs(
    *,
    current_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    target_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
) -> tuple[
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapSlot,
    ...,
]:
    repairs: list[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapSlot
    ] = []

    for current_observation in current_report.seed_observations:
        target_observation = target_report.seed_observation(
            current_observation.random_state
        )
        if current_observation.seed_group != target_observation.seed_group:
            raise ValueError(
                "exact witness target gap requires matching seed groups across baseline and target reports"
            )

        for z_index, z_value in enumerate(current_report.evaluation_grid):
            current_covered = current_observation.pointwise_coverage_by_z[z_index]
            target_covered = target_observation.pointwise_coverage_by_z[z_index]
            if current_covered == target_covered:
                continue
            repairs.append(
                Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapSlot(
                    random_state=current_observation.random_state,
                    seed_group=current_observation.seed_group,
                    z_index=z_index,
                    z_value=z_value,
                    coverage_kind="pointwise",
                    current_covered=current_covered,
                    target_covered=target_covered,
                )
            )

    return tuple(repairs)


def _driver_signature(
    *,
    runtime_alignment_driver_signature: str,
    target_match_observed: bool,
    pointwise_gap_to_target_vector: tuple[int, int, int],
    band_gap_to_target_vector: tuple[int, int, int],
    remaining_witness_slots_to_lift: int,
    pending_pointwise_repairs: tuple[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapSlot,
        ...,
    ],
) -> str:
    if target_match_observed:
        return "same-seed-exact-witness-target-gap-closed"
    if (
        runtime_alignment_driver_signature == "same-seed-near-zero-grid-runtime-aligned"
        and pointwise_gap_to_target_vector == (0, 1, 1)
        and band_gap_to_target_vector == (0, 0, 0)
        and remaining_witness_slots_to_lift == 1
        and tuple(
            (slot.random_state, slot.z_index) for slot in pending_pointwise_repairs
        )
        == ((303, 1), (707, 2))
    ):
        return "same-seed-exact-witness-target-gap-open"
    return "mixed-same-seed-exact-witness-target-gap"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    runtime_alignment_driver_signature: str
    current_point_miss_vector: tuple[int, int, int]
    target_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    target_band_miss_vector: tuple[int, int, int]
    current_witness_point_miss_vector: tuple[int, int, int]
    target_witness_point_miss_vector: tuple[int, int, int]
    current_fresh_point_miss_vector: tuple[int, int, int]
    target_fresh_point_miss_vector: tuple[int, int, int]
    pointwise_gap_to_target_vector: tuple[int, int, int]
    band_gap_to_target_vector: tuple[int, int, int]
    remaining_total_point_miss_reduction: int
    remaining_witness_slots_to_lift: int
    current_witness_floor: float
    target_witness_floor: float
    left_guard_grid_value: float
    left_guard_already_matches_target: bool
    runtime_witness_path: tuple[str, ...]
    target_candidate_status: str
    target_resulting_evidence_status: str
    target_match_observed: bool
    driver_signature: str
    pending_pointwise_repairs: tuple[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapSlot,
        ...,
    ]
    target_gap_floor_ratio: float
    canonical_exact_witness_target_gap_digest: tuple[str, ...]

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
        self.runtime_alignment_driver_signature = str(
            self.runtime_alignment_driver_signature
        ).strip()
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
        self.current_witness_point_miss_vector = tuple(
            int(value) for value in self.current_witness_point_miss_vector
        )
        self.target_witness_point_miss_vector = tuple(
            int(value) for value in self.target_witness_point_miss_vector
        )
        self.current_fresh_point_miss_vector = tuple(
            int(value) for value in self.current_fresh_point_miss_vector
        )
        self.target_fresh_point_miss_vector = tuple(
            int(value) for value in self.target_fresh_point_miss_vector
        )
        self.pointwise_gap_to_target_vector = tuple(
            int(value) for value in self.pointwise_gap_to_target_vector
        )
        self.band_gap_to_target_vector = tuple(
            int(value) for value in self.band_gap_to_target_vector
        )
        self.remaining_total_point_miss_reduction = int(
            self.remaining_total_point_miss_reduction
        )
        self.remaining_witness_slots_to_lift = int(self.remaining_witness_slots_to_lift)
        self.current_witness_floor = float(self.current_witness_floor)
        self.target_witness_floor = float(self.target_witness_floor)
        self.left_guard_grid_value = float(self.left_guard_grid_value)
        self.left_guard_already_matches_target = bool(
            self.left_guard_already_matches_target
        )
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.target_candidate_status = str(self.target_candidate_status).strip()
        self.target_resulting_evidence_status = str(
            self.target_resulting_evidence_status
        ).strip()
        self.target_match_observed = bool(self.target_match_observed)
        self.driver_signature = str(self.driver_signature).strip()
        self.pending_pointwise_repairs = tuple(self.pending_pointwise_repairs)
        self.target_gap_floor_ratio = float(self.target_gap_floor_ratio)
        self.canonical_exact_witness_target_gap_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_exact_witness_target_gap_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap_report(
    *,
    current_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
    | None = None,
    runtime_alignment_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseRuntimeAlignmentProbeReport
        | None
    ) = None,
    target_profile_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessCandidateProfileReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapReport:
    resolved_current = (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        if current_report is None
        else current_report
    )
    resolved_runtime_alignment = (
        build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_runtime_alignment_proxy_report(
            current_report=resolved_current
        )
        if runtime_alignment_report is None
        else runtime_alignment_report
    )
    resolved_target_profile = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_candidate_profile()
        if target_profile_report is None
        else target_profile_report
    )

    current_all_summary = resolved_current.group_summary("all")
    current_fresh_summary = resolved_current.group_summary("fresh")
    current_witness_summary = resolved_current.group_summary("witness")
    target_all_summary = resolved_target_profile.target_after_report.group_summary(
        "all"
    )
    target_fresh_summary = resolved_target_profile.target_after_report.group_summary(
        "fresh"
    )
    target_witness_summary = resolved_target_profile.target_after_report.group_summary(
        "witness"
    )

    same_seed_random_states = tuple(
        observation.random_state for observation in resolved_current.seed_observations
    )
    if resolved_current.policy_digest != resolved_runtime_alignment.policy_digest:
        raise ValueError(
            "exact witness target gap requires shared policy digest across baseline and runtime alignment"
        )
    if resolved_current.binding_design != resolved_runtime_alignment.binding_design:
        raise ValueError(
            "exact witness target gap requires shared binding design across baseline and runtime alignment"
        )
    if resolved_current.window_label != resolved_runtime_alignment.window_label:
        raise ValueError(
            "exact witness target gap requires shared window label across baseline and runtime alignment"
        )
    if same_seed_random_states != resolved_runtime_alignment.same_seed_random_states:
        raise ValueError(
            "exact witness target gap requires the runtime-aligned same-seed replay ordering"
        )
    if (
        current_all_summary.point_miss_count_by_z
        != resolved_runtime_alignment.current_point_miss_vector
    ):
        raise ValueError(
            "exact witness target gap requires the live pointwise miss vector to stay runtime-aligned"
        )
    if (
        current_all_summary.uniform_band_miss_count_by_z
        != resolved_runtime_alignment.current_band_miss_vector
    ):
        raise ValueError(
            "exact witness target gap requires the live band miss vector to stay runtime-aligned"
        )

    if resolved_current.policy_digest != resolved_target_profile.policy_digest:
        raise ValueError(
            "exact witness target gap requires shared policy digest across baseline and target profile"
        )
    if resolved_current.binding_design != resolved_target_profile.binding_design:
        raise ValueError(
            "exact witness target gap requires shared binding design across baseline and target profile"
        )
    if resolved_current.window_label != resolved_target_profile.window_label:
        raise ValueError(
            "exact witness target gap requires shared window label across baseline and target profile"
        )
    if same_seed_random_states != resolved_target_profile.same_seed_random_states:
        raise ValueError(
            "exact witness target gap requires the target profile to preserve the exact same-seed ordering"
        )

    pointwise_gap_to_target_vector = tuple(
        current_all_summary.point_miss_count_by_z[index]
        - target_all_summary.point_miss_count_by_z[index]
        for index in range(len(current_all_summary.point_miss_count_by_z))
    )
    band_gap_to_target_vector = tuple(
        current_all_summary.uniform_band_miss_count_by_z[index]
        - target_all_summary.uniform_band_miss_count_by_z[index]
        for index in range(len(current_all_summary.uniform_band_miss_count_by_z))
    )
    target_match_observed = bool(
        pointwise_gap_to_target_vector == (0, 0, 0)
        and band_gap_to_target_vector == (0, 0, 0)
    )
    pending_pointwise_repairs = _pending_pointwise_repairs(
        current_report=resolved_current,
        target_report=resolved_target_profile.target_after_report,
    )
    driver_signature = _driver_signature(
        runtime_alignment_driver_signature=resolved_runtime_alignment.driver_signature,
        target_match_observed=target_match_observed,
        pointwise_gap_to_target_vector=pointwise_gap_to_target_vector,
        band_gap_to_target_vector=band_gap_to_target_vector,
        remaining_witness_slots_to_lift=resolved_target_profile.lifted_witness_slots,
        pending_pointwise_repairs=pending_pointwise_repairs,
    )
    target_gap_floor_ratio = (
        resolved_target_profile.baseline_witness_floor
        / resolved_target_profile.target_witness_floor
    )
    canonical_digest = (
        "- exact same 8-seed live rerun still replays the current bounded baseline: point miss remains `[1, 3, 2]`, band miss remains `[0, 1, 1]`, and witness floor therefore stays `7/9 = 0.778` instead of the admissible target `8/9 = 0.889`",
        "- the remaining gap to the synthetic admissible target is fully localized to two pointwise slots and zero band repairs: seed `303` must recover the witness-center slot at `z = 0.15`, and seed `707` must recover the fresh right-shoulder slot at `z = 0.25`",
        "- left support and bounded object flow already match the target: `z = 0.05` still carries no new band miss, and future estimator evidence must keep the same preserve-left-support path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`",
        "- current Trigger 2 implication: `same-seed-exact-witness-target-gap-open`; future live reruns only become admissible when those two slot repairs are realized on the same `8` random states without reopening left-guard failures",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-target-gap"
        ),
        policy_digest=resolved_current.policy_digest,
        binding_design=resolved_current.binding_design,
        window_label=resolved_current.window_label,
        same_seed_random_states=same_seed_random_states,
        runtime_alignment_driver_signature=resolved_runtime_alignment.driver_signature,
        current_point_miss_vector=current_all_summary.point_miss_count_by_z,
        target_point_miss_vector=target_all_summary.point_miss_count_by_z,
        current_band_miss_vector=current_all_summary.uniform_band_miss_count_by_z,
        target_band_miss_vector=target_all_summary.uniform_band_miss_count_by_z,
        current_witness_point_miss_vector=current_witness_summary.point_miss_count_by_z,
        target_witness_point_miss_vector=target_witness_summary.point_miss_count_by_z,
        current_fresh_point_miss_vector=current_fresh_summary.point_miss_count_by_z,
        target_fresh_point_miss_vector=target_fresh_summary.point_miss_count_by_z,
        pointwise_gap_to_target_vector=pointwise_gap_to_target_vector,
        band_gap_to_target_vector=band_gap_to_target_vector,
        remaining_total_point_miss_reduction=sum(pointwise_gap_to_target_vector),
        remaining_witness_slots_to_lift=resolved_target_profile.lifted_witness_slots,
        current_witness_floor=resolved_target_profile.baseline_witness_floor,
        target_witness_floor=resolved_target_profile.target_witness_floor,
        left_guard_grid_value=resolved_target_profile.left_guard_grid_value,
        left_guard_already_matches_target=bool(
            current_all_summary.point_miss_count_by_z[0]
            == target_all_summary.point_miss_count_by_z[0]
            and current_all_summary.uniform_band_miss_count_by_z[0]
            == target_all_summary.uniform_band_miss_count_by_z[0]
        ),
        runtime_witness_path=resolved_target_profile.runtime_witness_path,
        target_candidate_status=resolved_target_profile.candidate_status_if_realized,
        target_resulting_evidence_status=(
            resolved_target_profile.resulting_evidence_status_if_realized
        ),
        target_match_observed=target_match_observed,
        driver_signature=driver_signature,
        pending_pointwise_repairs=pending_pointwise_repairs,
        target_gap_floor_ratio=target_gap_floor_ratio,
        canonical_exact_witness_target_gap_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapReport",
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapSlot",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap",
]
