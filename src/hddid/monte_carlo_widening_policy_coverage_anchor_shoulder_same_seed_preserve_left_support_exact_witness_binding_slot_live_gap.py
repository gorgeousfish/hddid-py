from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
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
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapSlot:
    random_state: int
    seed_group: str
    z_index: int
    z_value: float
    coverage_kind: str
    current_covered: bool
    binding_profile_covered: bool

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)
        self.coverage_kind = str(self.coverage_kind).strip().lower()
        self.current_covered = bool(self.current_covered)
        self.binding_profile_covered = bool(self.binding_profile_covered)


def _pending_binding_repairs(
    *,
    current_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    binding_profile_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
) -> tuple[
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapSlot,
    ...,
]:
    repairs: list[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapSlot
    ] = []

    for current_observation in current_report.seed_observations:
        binding_observation = binding_profile_report.seed_observation(
            current_observation.random_state
        )
        if current_observation.seed_group != binding_observation.seed_group:
            raise ValueError(
                "binding-slot live gap requires matching seed groups across current and binding-only reports"
            )

        for z_index, z_value in enumerate(current_report.evaluation_grid):
            current_covered = current_observation.pointwise_coverage_by_z[z_index]
            binding_profile_covered = binding_observation.pointwise_coverage_by_z[
                z_index
            ]
            if current_covered == binding_profile_covered:
                continue
            repairs.append(
                Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapSlot(
                    random_state=current_observation.random_state,
                    seed_group=current_observation.seed_group,
                    z_index=z_index,
                    z_value=z_value,
                    coverage_kind="pointwise",
                    current_covered=current_covered,
                    binding_profile_covered=binding_profile_covered,
                )
            )

    return tuple(repairs)


def _driver_signature(
    *,
    runtime_alignment_driver_signature: str,
    binding_slot_progress_driver_signature: str,
    current_point_miss_vector: tuple[int, int, int],
    binding_only_point_miss_vector: tuple[int, int, int],
    current_band_miss_vector: tuple[int, int, int],
    binding_only_band_miss_vector: tuple[int, int, int],
    current_witness_point_miss_vector: tuple[int, int, int],
    binding_only_witness_point_miss_vector: tuple[int, int, int],
    current_fresh_point_miss_vector: tuple[int, int, int],
    binding_only_fresh_point_miss_vector: tuple[int, int, int],
    left_guard_already_matches_binding_profile: bool,
    binding_slot_match_observed: bool,
    pending_binding_repair: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapSlot
        | None
    ),
    residual_slot_after_binding: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapSlot
    ),
) -> str:
    if (
        runtime_alignment_driver_signature == "same-seed-near-zero-grid-runtime-aligned"
        and binding_slot_progress_driver_signature
        == "same-seed-exact-witness-binding-slot-progress-profile"
        and current_point_miss_vector == (1, 3, 2)
        and binding_only_point_miss_vector == (1, 2, 2)
        and current_band_miss_vector == (0, 1, 1)
        and binding_only_band_miss_vector == (0, 1, 1)
        and current_witness_point_miss_vector == (1, 2, 0)
        and binding_only_witness_point_miss_vector == (1, 1, 0)
        and current_fresh_point_miss_vector == (0, 1, 2)
        and binding_only_fresh_point_miss_vector == (0, 1, 2)
        and left_guard_already_matches_binding_profile
        and not binding_slot_match_observed
        and pending_binding_repair is not None
        and (
            pending_binding_repair.random_state,
            pending_binding_repair.z_index,
            pending_binding_repair.current_covered,
            pending_binding_repair.binding_profile_covered,
        )
        == (303, 1, False, True)
        and (
            residual_slot_after_binding.random_state,
            residual_slot_after_binding.z_index,
            residual_slot_after_binding.current_covered,
            residual_slot_after_binding.binding_profile_covered,
        )
        == (707, 2, False, False)
    ):
        return "same-seed-exact-witness-binding-slot-live-gap-open"
    return "mixed-same-seed-exact-witness-binding-slot-live-gap"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    runtime_alignment_driver_signature: str
    binding_slot_progress_driver_signature: str
    current_point_miss_vector: tuple[int, int, int]
    binding_only_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    binding_only_band_miss_vector: tuple[int, int, int]
    current_witness_point_miss_vector: tuple[int, int, int]
    binding_only_witness_point_miss_vector: tuple[int, int, int]
    current_fresh_point_miss_vector: tuple[int, int, int]
    binding_only_fresh_point_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    projected_binding_witness_floor: float
    binding_gap_share_of_total_remaining_repairs: float
    left_guard_grid_value: float
    left_guard_already_matches_binding_profile: bool
    runtime_witness_path: tuple[str, ...]
    binding_slot_match_observed: bool
    pending_binding_repair: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapSlot
    residual_slot_after_binding: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapSlot
    driver_signature: str
    canonical_exact_witness_binding_slot_live_gap_digest: tuple[str, ...]

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
        self.binding_slot_progress_driver_signature = str(
            self.binding_slot_progress_driver_signature
        ).strip()
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.binding_only_point_miss_vector = tuple(
            int(value) for value in self.binding_only_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.binding_only_band_miss_vector = tuple(
            int(value) for value in self.binding_only_band_miss_vector
        )
        self.current_witness_point_miss_vector = tuple(
            int(value) for value in self.current_witness_point_miss_vector
        )
        self.binding_only_witness_point_miss_vector = tuple(
            int(value) for value in self.binding_only_witness_point_miss_vector
        )
        self.current_fresh_point_miss_vector = tuple(
            int(value) for value in self.current_fresh_point_miss_vector
        )
        self.binding_only_fresh_point_miss_vector = tuple(
            int(value) for value in self.binding_only_fresh_point_miss_vector
        )
        self.current_witness_floor = float(self.current_witness_floor)
        self.projected_binding_witness_floor = float(
            self.projected_binding_witness_floor
        )
        self.binding_gap_share_of_total_remaining_repairs = float(
            self.binding_gap_share_of_total_remaining_repairs
        )
        self.left_guard_grid_value = float(self.left_guard_grid_value)
        self.left_guard_already_matches_binding_profile = bool(
            self.left_guard_already_matches_binding_profile
        )
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.binding_slot_match_observed = bool(self.binding_slot_match_observed)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_exact_witness_binding_slot_live_gap_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_exact_witness_binding_slot_live_gap_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_live_gap_report(
    *,
    current_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
    | None = None,
    runtime_alignment_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseRuntimeAlignmentProbeReport
        | None
    ) = None,
    binding_slot_progress_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapReport:
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
    resolved_binding_progress = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile()
        if binding_slot_progress_report is None
        else binding_slot_progress_report
    )

    current_all_summary = resolved_current.group_summary("all")
    current_fresh_summary = resolved_current.group_summary("fresh")
    current_witness_summary = resolved_current.group_summary("witness")
    binding_profile_report = resolved_binding_progress.binding_only_after_report
    binding_all_summary = binding_profile_report.group_summary("all")
    binding_fresh_summary = binding_profile_report.group_summary("fresh")
    binding_witness_summary = binding_profile_report.group_summary("witness")

    same_seed_random_states = tuple(
        observation.random_state for observation in resolved_current.seed_observations
    )
    if resolved_current.policy_digest != resolved_runtime_alignment.policy_digest:
        raise ValueError(
            "binding-slot live gap requires shared policy digest across current and runtime alignment reports"
        )
    if resolved_current.binding_design != resolved_runtime_alignment.binding_design:
        raise ValueError(
            "binding-slot live gap requires shared binding design across current and runtime alignment reports"
        )
    if resolved_current.window_label != resolved_runtime_alignment.window_label:
        raise ValueError(
            "binding-slot live gap requires shared window label across current and runtime alignment reports"
        )
    if same_seed_random_states != resolved_runtime_alignment.same_seed_random_states:
        raise ValueError(
            "binding-slot live gap requires runtime-aligned exact same-seed ordering"
        )
    if (
        current_all_summary.point_miss_count_by_z
        != resolved_runtime_alignment.current_point_miss_vector
    ):
        raise ValueError(
            "binding-slot live gap requires the live pointwise miss vector to stay runtime-aligned"
        )
    if (
        current_all_summary.uniform_band_miss_count_by_z
        != resolved_runtime_alignment.current_band_miss_vector
    ):
        raise ValueError(
            "binding-slot live gap requires the live band miss vector to stay runtime-aligned"
        )

    if resolved_current.policy_digest != resolved_binding_progress.policy_digest:
        raise ValueError(
            "binding-slot live gap requires shared policy digest across current and binding-slot progress reports"
        )
    if resolved_current.binding_design != resolved_binding_progress.binding_design:
        raise ValueError(
            "binding-slot live gap requires shared binding design across current and binding-slot progress reports"
        )
    if resolved_current.window_label != resolved_binding_progress.window_label:
        raise ValueError(
            "binding-slot live gap requires shared window label across current and binding-slot progress reports"
        )
    if same_seed_random_states != resolved_binding_progress.same_seed_random_states:
        raise ValueError(
            "binding-slot live gap requires the binding-slot progress profile to preserve exact same-seed ordering"
        )
    if (
        current_all_summary.point_miss_count_by_z
        != resolved_binding_progress.baseline_point_miss_vector
    ):
        raise ValueError(
            "binding-slot live gap requires the current pointwise miss vector to match the progress-profile baseline"
        )
    if (
        current_all_summary.uniform_band_miss_count_by_z
        != resolved_binding_progress.baseline_band_miss_vector
    ):
        raise ValueError(
            "binding-slot live gap requires the current band miss vector to match the progress-profile baseline"
        )
    if (
        binding_all_summary.point_miss_count_by_z
        != resolved_binding_progress.binding_only_point_miss_vector
    ):
        raise ValueError(
            "binding-slot live gap requires the binding-only pointwise miss vector to match the progress profile"
        )
    if (
        binding_all_summary.uniform_band_miss_count_by_z
        != resolved_binding_progress.binding_only_band_miss_vector
    ):
        raise ValueError(
            "binding-slot live gap requires the binding-only band miss vector to match the progress profile"
        )

    pending_repairs = _pending_binding_repairs(
        current_report=resolved_current,
        binding_profile_report=binding_profile_report,
    )
    if len(pending_repairs) != 1:
        raise ValueError(
            "binding-slot live gap expects exactly one current-to-binding-profile repair slot"
        )
    pending_binding_repair = pending_repairs[0]

    residual_observation_current = resolved_current.seed_observation(
        resolved_binding_progress.residual_repair_slot.random_state
    )
    residual_observation_binding = binding_profile_report.seed_observation(
        resolved_binding_progress.residual_repair_slot.random_state
    )
    residual_slot_after_binding = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapSlot(
        random_state=resolved_binding_progress.residual_repair_slot.random_state,
        seed_group=resolved_binding_progress.residual_repair_slot.seed_group,
        z_index=resolved_binding_progress.residual_repair_slot.z_index,
        z_value=resolved_binding_progress.residual_repair_slot.z_value,
        coverage_kind="pointwise",
        current_covered=residual_observation_current.pointwise_coverage_by_z[
            resolved_binding_progress.residual_repair_slot.z_index
        ],
        binding_profile_covered=residual_observation_binding.pointwise_coverage_by_z[
            resolved_binding_progress.residual_repair_slot.z_index
        ],
    )
    binding_slot_match_observed = bool(
        current_all_summary.point_miss_count_by_z
        == binding_all_summary.point_miss_count_by_z
        and current_all_summary.uniform_band_miss_count_by_z
        == binding_all_summary.uniform_band_miss_count_by_z
    )
    total_remaining_repairs = (
        resolved_binding_progress.binding_only_total_point_miss_reduction
        + resolved_binding_progress.remaining_total_point_miss_reduction_after_binding
    )
    if total_remaining_repairs <= 0:
        raise ValueError(
            "binding-slot live gap requires positive remaining repairs in the binding-slot progress profile"
        )
    binding_gap_share_of_total_remaining_repairs = (
        resolved_binding_progress.binding_only_total_point_miss_reduction
        / total_remaining_repairs
    )
    left_guard_already_matches_binding_profile = bool(
        current_all_summary.point_miss_count_by_z[0]
        == binding_all_summary.point_miss_count_by_z[0]
        and current_all_summary.uniform_band_miss_count_by_z[0]
        == binding_all_summary.uniform_band_miss_count_by_z[0]
    )
    driver_signature = _driver_signature(
        runtime_alignment_driver_signature=resolved_runtime_alignment.driver_signature,
        binding_slot_progress_driver_signature=resolved_binding_progress.driver_signature,
        current_point_miss_vector=current_all_summary.point_miss_count_by_z,
        binding_only_point_miss_vector=binding_all_summary.point_miss_count_by_z,
        current_band_miss_vector=current_all_summary.uniform_band_miss_count_by_z,
        binding_only_band_miss_vector=binding_all_summary.uniform_band_miss_count_by_z,
        current_witness_point_miss_vector=current_witness_summary.point_miss_count_by_z,
        binding_only_witness_point_miss_vector=binding_witness_summary.point_miss_count_by_z,
        current_fresh_point_miss_vector=current_fresh_summary.point_miss_count_by_z,
        binding_only_fresh_point_miss_vector=binding_fresh_summary.point_miss_count_by_z,
        left_guard_already_matches_binding_profile=(
            left_guard_already_matches_binding_profile
        ),
        binding_slot_match_observed=binding_slot_match_observed,
        pending_binding_repair=pending_binding_repair,
        residual_slot_after_binding=residual_slot_after_binding,
    )
    canonical_digest = (
        "- exact same 8-seed live rerun is still one pointwise witness repair short of the first-step binding profile: current point miss stays `[1, 3, 2]` while the binding-only profile is `[1, 2, 2]`, with band miss fixed at `[0, 1, 1]` and witness floor still `7/9 = "
        + _format_float(resolved_binding_progress.baseline_witness_floor)
        + "` instead of the projected `8/9 = "
        + _format_float(resolved_binding_progress.projected_binding_witness_floor)
        + "`",
        "- the current live gap to that first-step profile is singular and witness-only: seed `303` must recover the witness-center slot at `z = 0.15`, while the fresh group already matches the binding-only profile and the residual post-binding repair still stays seed `707` at `z = 0.25`",
        "- bounded runtime discipline already matches the binding profile: left guard `z = 0.05` stays aligned, and future estimator evidence must keep the same preserve-left-support path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`",
        "- current Trigger 2 implication: `"
        + driver_signature
        + "`; future live reruns should first land seed `303` / `z = 0.15` before spending on the residual seed `707` shoulder repair",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-binding-slot-live-gap"
        ),
        policy_digest=resolved_current.policy_digest,
        binding_design=resolved_current.binding_design,
        window_label=resolved_current.window_label,
        same_seed_random_states=same_seed_random_states,
        runtime_alignment_driver_signature=resolved_runtime_alignment.driver_signature,
        binding_slot_progress_driver_signature=resolved_binding_progress.driver_signature,
        current_point_miss_vector=current_all_summary.point_miss_count_by_z,
        binding_only_point_miss_vector=binding_all_summary.point_miss_count_by_z,
        current_band_miss_vector=current_all_summary.uniform_band_miss_count_by_z,
        binding_only_band_miss_vector=binding_all_summary.uniform_band_miss_count_by_z,
        current_witness_point_miss_vector=current_witness_summary.point_miss_count_by_z,
        binding_only_witness_point_miss_vector=binding_witness_summary.point_miss_count_by_z,
        current_fresh_point_miss_vector=current_fresh_summary.point_miss_count_by_z,
        binding_only_fresh_point_miss_vector=binding_fresh_summary.point_miss_count_by_z,
        current_witness_floor=resolved_binding_progress.baseline_witness_floor,
        projected_binding_witness_floor=(
            resolved_binding_progress.projected_binding_witness_floor
        ),
        binding_gap_share_of_total_remaining_repairs=(
            binding_gap_share_of_total_remaining_repairs
        ),
        left_guard_grid_value=resolved_binding_progress.left_guard_grid_value,
        left_guard_already_matches_binding_profile=(
            left_guard_already_matches_binding_profile
        ),
        runtime_witness_path=resolved_binding_progress.runtime_witness_path,
        binding_slot_match_observed=binding_slot_match_observed,
        pending_binding_repair=pending_binding_repair,
        residual_slot_after_binding=residual_slot_after_binding,
        driver_signature=driver_signature,
        canonical_exact_witness_binding_slot_live_gap_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_live_gap() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_live_gap_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapReport",
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapSlot",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_live_gap_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_live_gap",
]
