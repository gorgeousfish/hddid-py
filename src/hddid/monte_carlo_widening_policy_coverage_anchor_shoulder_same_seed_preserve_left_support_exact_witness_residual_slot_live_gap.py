from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCompletionProfileReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapSlot:
    random_state: int
    seed_group: str
    z_index: int
    z_value: float
    coverage_kind: str
    binding_profile_covered: bool
    completion_profile_covered: bool

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)
        self.coverage_kind = str(self.coverage_kind).strip().lower()
        self.binding_profile_covered = bool(self.binding_profile_covered)
        self.completion_profile_covered = bool(self.completion_profile_covered)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    binding_slot_progress_driver_signature: str
    residual_slot_completion_driver_signature: str
    binding_only_point_miss_vector: tuple[int, int, int]
    completion_point_miss_vector: tuple[int, int, int]
    binding_only_band_miss_vector: tuple[int, int, int]
    completion_band_miss_vector: tuple[int, int, int]
    binding_only_witness_point_miss_vector: tuple[int, int, int]
    completion_witness_point_miss_vector: tuple[int, int, int]
    binding_only_fresh_point_miss_vector: tuple[int, int, int]
    completion_fresh_point_miss_vector: tuple[int, int, int]
    binding_only_witness_floor: float
    completion_witness_floor: float
    residual_gap_share_of_post_binding_remaining_repairs: float
    left_guard_grid_value: float
    left_guard_already_matches_completion_profile: bool
    runtime_witness_path: tuple[str, ...]
    binding_slot_already_matches_completion_profile: bool
    pending_residual_repair: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapSlot
    driver_signature: str
    canonical_exact_witness_residual_slot_live_gap_digest: tuple[str, ...]

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
        self.binding_slot_progress_driver_signature = str(
            self.binding_slot_progress_driver_signature
        ).strip()
        self.residual_slot_completion_driver_signature = str(
            self.residual_slot_completion_driver_signature
        ).strip()
        self.binding_only_point_miss_vector = tuple(
            int(value) for value in self.binding_only_point_miss_vector
        )
        self.completion_point_miss_vector = tuple(
            int(value) for value in self.completion_point_miss_vector
        )
        self.binding_only_band_miss_vector = tuple(
            int(value) for value in self.binding_only_band_miss_vector
        )
        self.completion_band_miss_vector = tuple(
            int(value) for value in self.completion_band_miss_vector
        )
        self.binding_only_witness_point_miss_vector = tuple(
            int(value) for value in self.binding_only_witness_point_miss_vector
        )
        self.completion_witness_point_miss_vector = tuple(
            int(value) for value in self.completion_witness_point_miss_vector
        )
        self.binding_only_fresh_point_miss_vector = tuple(
            int(value) for value in self.binding_only_fresh_point_miss_vector
        )
        self.completion_fresh_point_miss_vector = tuple(
            int(value) for value in self.completion_fresh_point_miss_vector
        )
        self.binding_only_witness_floor = float(self.binding_only_witness_floor)
        self.completion_witness_floor = float(self.completion_witness_floor)
        self.residual_gap_share_of_post_binding_remaining_repairs = float(
            self.residual_gap_share_of_post_binding_remaining_repairs
        )
        self.left_guard_grid_value = float(self.left_guard_grid_value)
        self.left_guard_already_matches_completion_profile = bool(
            self.left_guard_already_matches_completion_profile
        )
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.binding_slot_already_matches_completion_profile = bool(
            self.binding_slot_already_matches_completion_profile
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_exact_witness_residual_slot_live_gap_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_exact_witness_residual_slot_live_gap_digest
        )


def _pending_residual_repairs(
    *,
    binding_profile_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    completion_profile_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
) -> tuple[
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapSlot,
    ...,
]:
    repairs: list[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapSlot
    ] = []

    for binding_observation in binding_profile_report.seed_observations:
        completion_observation = completion_profile_report.seed_observation(
            binding_observation.random_state
        )
        if binding_observation.seed_group != completion_observation.seed_group:
            raise ValueError(
                "residual-slot live gap requires matching seed groups across binding and completion reports"
            )

        for z_index, z_value in enumerate(binding_profile_report.evaluation_grid):
            binding_profile_covered = binding_observation.pointwise_coverage_by_z[
                z_index
            ]
            completion_profile_covered = completion_observation.pointwise_coverage_by_z[
                z_index
            ]
            if binding_profile_covered == completion_profile_covered:
                continue
            repairs.append(
                Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapSlot(
                    random_state=binding_observation.random_state,
                    seed_group=binding_observation.seed_group,
                    z_index=z_index,
                    z_value=z_value,
                    coverage_kind="pointwise",
                    binding_profile_covered=binding_profile_covered,
                    completion_profile_covered=completion_profile_covered,
                )
            )

    return tuple(repairs)


def _driver_signature(
    *,
    binding_slot_progress_driver_signature: str,
    residual_slot_completion_driver_signature: str,
    binding_only_point_miss_vector: tuple[int, int, int],
    completion_point_miss_vector: tuple[int, int, int],
    binding_only_band_miss_vector: tuple[int, int, int],
    completion_band_miss_vector: tuple[int, int, int],
    binding_only_witness_point_miss_vector: tuple[int, int, int],
    completion_witness_point_miss_vector: tuple[int, int, int],
    binding_only_fresh_point_miss_vector: tuple[int, int, int],
    completion_fresh_point_miss_vector: tuple[int, int, int],
    left_guard_already_matches_completion_profile: bool,
    binding_slot_already_matches_completion_profile: bool,
    pending_residual_repair: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapSlot
        | None
    ),
) -> str:
    if (
        binding_slot_progress_driver_signature
        == "same-seed-exact-witness-binding-slot-progress-profile"
        and residual_slot_completion_driver_signature
        == "same-seed-exact-witness-residual-slot-completion-profile"
        and binding_only_point_miss_vector == (1, 2, 2)
        and completion_point_miss_vector == (1, 2, 1)
        and binding_only_band_miss_vector == (0, 1, 1)
        and completion_band_miss_vector == (0, 1, 1)
        and binding_only_witness_point_miss_vector == (1, 1, 0)
        and completion_witness_point_miss_vector == (1, 1, 0)
        and binding_only_fresh_point_miss_vector == (0, 1, 2)
        and completion_fresh_point_miss_vector == (0, 1, 1)
        and left_guard_already_matches_completion_profile
        and binding_slot_already_matches_completion_profile
        and pending_residual_repair is not None
        and (
            pending_residual_repair.random_state,
            pending_residual_repair.z_index,
            pending_residual_repair.binding_profile_covered,
            pending_residual_repair.completion_profile_covered,
        )
        == (707, 2, False, True)
    ):
        return "same-seed-exact-witness-residual-slot-live-gap-open"
    return "mixed-same-seed-exact-witness-residual-slot-live-gap"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_live_gap_report(
    *,
    binding_slot_progress_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport
        | None
    ) = None,
    residual_slot_completion_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCompletionProfileReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapReport:
    resolved_binding_progress = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile()
        if binding_slot_progress_report is None
        else binding_slot_progress_report
    )
    resolved_residual_completion = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile()
        if residual_slot_completion_report is None
        else residual_slot_completion_report
    )

    binding_profile_report = resolved_binding_progress.binding_only_after_report
    completion_profile_report = resolved_residual_completion.completion_after_report
    binding_all_summary = binding_profile_report.group_summary("all")
    binding_fresh_summary = binding_profile_report.group_summary("fresh")
    binding_witness_summary = binding_profile_report.group_summary("witness")
    completion_all_summary = completion_profile_report.group_summary("all")
    completion_fresh_summary = completion_profile_report.group_summary("fresh")
    completion_witness_summary = completion_profile_report.group_summary("witness")

    same_seed_random_states = tuple(
        observation.random_state
        for observation in binding_profile_report.seed_observations
    )
    if binding_profile_report.policy_digest != resolved_binding_progress.policy_digest:
        raise ValueError(
            "residual-slot live gap requires the binding-only report to preserve the progress-profile policy digest"
        )
    if (
        binding_profile_report.binding_design
        != resolved_binding_progress.binding_design
    ):
        raise ValueError(
            "residual-slot live gap requires the binding-only report to preserve the progress-profile binding design"
        )
    if binding_profile_report.window_label != resolved_binding_progress.window_label:
        raise ValueError(
            "residual-slot live gap requires the binding-only report to preserve the progress-profile window label"
        )
    if (
        binding_all_summary.point_miss_count_by_z
        != resolved_binding_progress.binding_only_point_miss_vector
    ):
        raise ValueError(
            "residual-slot live gap requires the binding-only pointwise miss vector to match the progress profile"
        )
    if (
        binding_all_summary.uniform_band_miss_count_by_z
        != resolved_binding_progress.binding_only_band_miss_vector
    ):
        raise ValueError(
            "residual-slot live gap requires the binding-only band miss vector to match the progress profile"
        )

    if (
        binding_profile_report.policy_digest
        != resolved_residual_completion.policy_digest
    ):
        raise ValueError(
            "residual-slot live gap requires shared policy digest across binding and completion reports"
        )
    if (
        binding_profile_report.binding_design
        != resolved_residual_completion.binding_design
    ):
        raise ValueError(
            "residual-slot live gap requires shared binding design across binding and completion reports"
        )
    if binding_profile_report.window_label != resolved_residual_completion.window_label:
        raise ValueError(
            "residual-slot live gap requires shared window label across binding and completion reports"
        )
    if same_seed_random_states != resolved_residual_completion.same_seed_random_states:
        raise ValueError(
            "residual-slot live gap requires the completion profile to preserve exact same-seed ordering"
        )
    if (
        completion_all_summary.point_miss_count_by_z
        != resolved_residual_completion.completion_point_miss_vector
    ):
        raise ValueError(
            "residual-slot live gap requires the completion pointwise miss vector to match the completion profile"
        )
    if (
        completion_all_summary.uniform_band_miss_count_by_z
        != resolved_residual_completion.completion_band_miss_vector
    ):
        raise ValueError(
            "residual-slot live gap requires the completion band miss vector to match the completion profile"
        )

    pending_repairs = _pending_residual_repairs(
        binding_profile_report=binding_profile_report,
        completion_profile_report=completion_profile_report,
    )
    if len(pending_repairs) != 1:
        raise ValueError(
            "residual-slot live gap expects exactly one binding-to-completion repair slot"
        )
    pending_residual_repair = pending_repairs[0]

    left_guard_already_matches_completion_profile = bool(
        binding_all_summary.point_miss_count_by_z[0]
        == completion_all_summary.point_miss_count_by_z[0]
        and binding_all_summary.uniform_band_miss_count_by_z[0]
        == completion_all_summary.uniform_band_miss_count_by_z[0]
    )
    binding_observation = binding_profile_report.seed_observation(
        resolved_binding_progress.binding_repair_slot.random_state
    )
    completion_observation = completion_profile_report.seed_observation(
        resolved_binding_progress.binding_repair_slot.random_state
    )
    binding_slot_already_matches_completion_profile = bool(
        binding_observation.pointwise_coverage_by_z[
            resolved_binding_progress.binding_repair_slot.z_index
        ]
        == completion_observation.pointwise_coverage_by_z[
            resolved_binding_progress.binding_repair_slot.z_index
        ]
    )
    driver_signature = _driver_signature(
        binding_slot_progress_driver_signature=resolved_binding_progress.driver_signature,
        residual_slot_completion_driver_signature=(
            resolved_residual_completion.driver_signature
        ),
        binding_only_point_miss_vector=binding_all_summary.point_miss_count_by_z,
        completion_point_miss_vector=completion_all_summary.point_miss_count_by_z,
        binding_only_band_miss_vector=binding_all_summary.uniform_band_miss_count_by_z,
        completion_band_miss_vector=completion_all_summary.uniform_band_miss_count_by_z,
        binding_only_witness_point_miss_vector=(
            binding_witness_summary.point_miss_count_by_z
        ),
        completion_witness_point_miss_vector=(
            completion_witness_summary.point_miss_count_by_z
        ),
        binding_only_fresh_point_miss_vector=(
            binding_fresh_summary.point_miss_count_by_z
        ),
        completion_fresh_point_miss_vector=(
            completion_fresh_summary.point_miss_count_by_z
        ),
        left_guard_already_matches_completion_profile=(
            left_guard_already_matches_completion_profile
        ),
        binding_slot_already_matches_completion_profile=(
            binding_slot_already_matches_completion_profile
        ),
        pending_residual_repair=pending_residual_repair,
    )
    canonical_digest = (
        "- once the binding step is treated as landed, the exact same-seed live gap to full admissibility is now a single fresh-side pointwise repair: binding-only point miss is `[1, 2, 2]` while the completion profile is `[1, 2, 1]`, with band miss fixed at `[0, 1, 1]` and witness floor already holding at `8/9 = "
        + _format_float(resolved_binding_progress.projected_binding_witness_floor)
        + "`",
        "- the only remaining post-binding gap is seed `707` at fresh right-shoulder `z = 0.25`: witness coverage already matches the completion profile, and seed `303` / witness / `z = 0.15` no longer contributes additional debt at this rung",
        "- bounded runtime discipline is unchanged: the residual live gap still sits on `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` and keeps the left guard `z = 0.05` untouched",
        "- current Trigger 2 implication: `"
        + driver_signature
        + "`; once an observed rerun has landed the binding slot, it should close this single fresh residual repair before claiming `same-seed-estimator-evidence-admissible`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-residual-slot-live-gap"
        ),
        policy_digest=binding_profile_report.policy_digest,
        binding_design=binding_profile_report.binding_design,
        window_label=binding_profile_report.window_label,
        same_seed_random_states=same_seed_random_states,
        binding_slot_progress_driver_signature=resolved_binding_progress.driver_signature,
        residual_slot_completion_driver_signature=(
            resolved_residual_completion.driver_signature
        ),
        binding_only_point_miss_vector=binding_all_summary.point_miss_count_by_z,
        completion_point_miss_vector=completion_all_summary.point_miss_count_by_z,
        binding_only_band_miss_vector=binding_all_summary.uniform_band_miss_count_by_z,
        completion_band_miss_vector=completion_all_summary.uniform_band_miss_count_by_z,
        binding_only_witness_point_miss_vector=(
            binding_witness_summary.point_miss_count_by_z
        ),
        completion_witness_point_miss_vector=(
            completion_witness_summary.point_miss_count_by_z
        ),
        binding_only_fresh_point_miss_vector=(
            binding_fresh_summary.point_miss_count_by_z
        ),
        completion_fresh_point_miss_vector=(
            completion_fresh_summary.point_miss_count_by_z
        ),
        binding_only_witness_floor=resolved_binding_progress.projected_binding_witness_floor,
        completion_witness_floor=resolved_residual_completion.completion_witness_floor,
        residual_gap_share_of_post_binding_remaining_repairs=1.0,
        left_guard_grid_value=resolved_binding_progress.left_guard_grid_value,
        left_guard_already_matches_completion_profile=(
            left_guard_already_matches_completion_profile
        ),
        runtime_witness_path=resolved_binding_progress.runtime_witness_path,
        binding_slot_already_matches_completion_profile=(
            binding_slot_already_matches_completion_profile
        ),
        pending_residual_repair=pending_residual_repair,
        driver_signature=driver_signature,
        canonical_exact_witness_residual_slot_live_gap_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_live_gap() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_live_gap_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapReport",
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapSlot",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_live_gap_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_live_gap",
]
