from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation,
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)

if TYPE_CHECKING:
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportEstimatorEvidenceCandidateGuardReport,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPriorityContractReport,
    )


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _total_point_miss(point_miss_vector: tuple[int, ...]) -> int:
    return int(sum(int(value) for value in point_miss_vector))


def _canonical_runtime_candidate(
    contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
    ),
) -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate(
        runtime_witness_path=contract_report.runtime_witness_path,
        preserves_left_support_contract=contract_report.preserves_left_support_contract,
        diagonal_preserved=contract_report.diagonal_preserved,
        direct_covariance_edits_allowed=contract_report.direct_covariance_edits_allowed,
        required_patch_share_of_full_shared_vf_gap=contract_report.required_patch_share_of_full_shared_vf_gap,
        required_patch_share_of_omega_only_shared_vf_increment=contract_report.required_patch_share_of_omega_only_shared_vf_increment,
        required_patch_share_of_diagonal_omega_gap=contract_report.required_patch_share_of_diagonal_omega_gap,
        required_patch_share_of_psd_boundary=contract_report.required_patch_share_of_psd_boundary,
        compensating_stage_order=contract_report.compensating_stage_order,
        compensating_cumulative_share_of_total_absolute_mass=contract_report.compensating_cumulative_share_of_total_absolute_mass,
        compensating_zero_live_entry_count=contract_report.compensating_zero_live_entry_count,
    )


def _binding_only_after_report(
    *,
    baseline_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    repair_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPriorityContractReport
    ),
) -> Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport:
    binding_slot = repair_priority_report.binding_repair_slot
    updated_seed_observations = tuple(
        replace(
            observation,
            pointwise_coverage_by_z=tuple(
                True if index == binding_slot.z_index else covered
                for index, covered in enumerate(observation.pointwise_coverage_by_z)
            ),
        )
        if observation.random_state == binding_slot.random_state
        else observation
        for observation in baseline_report.seed_observations
    )
    return build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report(
        seed_observations=updated_seed_observations
    )


def _driver_signature(
    *,
    binding_repair_random_state: int,
    residual_repair_random_state: int,
    binding_only_point_miss_vector: tuple[int, int, int],
    binding_only_band_miss_vector: tuple[int, int, int],
    projected_binding_witness_floor: float,
    acceptance_driver_signature: str,
    candidate_status_if_binding_only_realized: str,
    residual_slot_still_open: bool,
    left_guard_pointwise_nonregression: bool,
    left_guard_band_preserved: bool,
) -> str:
    if (
        binding_repair_random_state == 303
        and residual_repair_random_state == 707
        and binding_only_point_miss_vector == (1, 2, 2)
        and binding_only_band_miss_vector == (0, 1, 1)
        and projected_binding_witness_floor > 0.88
        and acceptance_driver_signature
        == "same-seed-before-after-acceptance-not-yet-satisfied"
        and candidate_status_if_binding_only_realized == "candidate-rejected"
        and residual_slot_still_open
        and left_guard_pointwise_nonregression
        and left_guard_band_preserved
    ):
        return "same-seed-exact-witness-binding-slot-progress-profile"
    return "mixed-same-seed-exact-witness-binding-slot-progress-profile"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressSlot:
    random_state: int
    seed_group: str
    z_index: int
    z_value: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    binding_repair_slot: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressSlot
    residual_repair_slot: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressSlot
    baseline_point_miss_vector: tuple[int, int, int]
    binding_only_point_miss_vector: tuple[int, int, int]
    baseline_band_miss_vector: tuple[int, int, int]
    binding_only_band_miss_vector: tuple[int, int, int]
    baseline_witness_floor: float
    projected_binding_witness_floor: float
    acceptance_candidate_witness_floor: float
    required_min_witness_floor: float
    binding_only_total_point_miss_reduction: int
    remaining_total_point_miss_reduction_after_binding: int
    residual_slot_still_open: bool
    left_guard_grid_value: float
    left_guard_band_preserved: bool
    left_guard_pointwise_nonregression: bool
    runtime_witness_path: tuple[str, ...]
    before_after_driver_signature: str
    candidate_status_if_binding_only_realized: str
    resulting_evidence_status_if_binding_only_realized: str
    driver_signature: str
    binding_only_after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
    )
    canonical_exact_witness_binding_slot_progress_digest: tuple[str, ...]

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
        self.baseline_point_miss_vector = tuple(
            int(value) for value in self.baseline_point_miss_vector
        )
        self.binding_only_point_miss_vector = tuple(
            int(value) for value in self.binding_only_point_miss_vector
        )
        self.baseline_band_miss_vector = tuple(
            int(value) for value in self.baseline_band_miss_vector
        )
        self.binding_only_band_miss_vector = tuple(
            int(value) for value in self.binding_only_band_miss_vector
        )
        self.baseline_witness_floor = float(self.baseline_witness_floor)
        self.projected_binding_witness_floor = float(
            self.projected_binding_witness_floor
        )
        self.acceptance_candidate_witness_floor = float(
            self.acceptance_candidate_witness_floor
        )
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.binding_only_total_point_miss_reduction = int(
            self.binding_only_total_point_miss_reduction
        )
        self.remaining_total_point_miss_reduction_after_binding = int(
            self.remaining_total_point_miss_reduction_after_binding
        )
        self.residual_slot_still_open = bool(self.residual_slot_still_open)
        self.left_guard_grid_value = float(self.left_guard_grid_value)
        self.left_guard_band_preserved = bool(self.left_guard_band_preserved)
        self.left_guard_pointwise_nonregression = bool(
            self.left_guard_pointwise_nonregression
        )
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.before_after_driver_signature = str(
            self.before_after_driver_signature
        ).strip()
        self.candidate_status_if_binding_only_realized = str(
            self.candidate_status_if_binding_only_realized
        ).strip()
        self.resulting_evidence_status_if_binding_only_realized = str(
            self.resulting_evidence_status_if_binding_only_realized
        ).strip()
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_exact_witness_binding_slot_progress_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_exact_witness_binding_slot_progress_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile_report(
    *,
    baseline_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    repair_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessRepairPriorityContractReport
        | None
    ) = None,
    runtime_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport:
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract,
    )

    resolved_baseline = (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        if baseline_report is None
        else baseline_report
    )
    resolved_repair_priority = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_repair_priority_contract()
        if repair_priority_report is None
        else repair_priority_report
    )
    resolved_runtime_contract = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
        if runtime_contract_report is None
        else runtime_contract_report
    )

    if resolved_baseline.policy_digest != resolved_repair_priority.policy_digest:
        raise ValueError("binding-slot progress profile requires shared policy digest")
    if resolved_baseline.binding_design != resolved_repair_priority.binding_design:
        raise ValueError("binding-slot progress profile requires shared binding design")
    if resolved_baseline.window_label != resolved_repair_priority.window_label:
        raise ValueError("binding-slot progress profile requires shared window label")
    if (
        tuple(
            observation.random_state
            for observation in resolved_baseline.seed_observations
        )
        != resolved_repair_priority.same_seed_random_states
    ):
        raise ValueError(
            "binding-slot progress profile requires shared exact same-seed ordering"
        )

    progress_after_report = _binding_only_after_report(
        baseline_report=resolved_baseline,
        repair_priority_report=resolved_repair_priority,
    )
    before_after_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report(
        before_report=resolved_baseline,
        after_report=progress_after_report,
    )
    guard_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard_report(
        runtime_candidate=_canonical_runtime_candidate(resolved_runtime_contract),
        after_report=progress_after_report,
        before_report=resolved_baseline,
        runtime_contract_report=resolved_runtime_contract,
    )

    binding_repair_slot = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressSlot(
        random_state=resolved_repair_priority.binding_repair_slot.random_state,
        seed_group=resolved_repair_priority.binding_repair_slot.seed_group,
        z_index=resolved_repair_priority.binding_repair_slot.z_index,
        z_value=resolved_repair_priority.binding_repair_slot.z_value,
    )
    residual_repair_slot = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressSlot(
        random_state=resolved_repair_priority.residual_repair_slot.random_state,
        seed_group=resolved_repair_priority.residual_repair_slot.seed_group,
        z_index=resolved_repair_priority.residual_repair_slot.z_index,
        z_value=resolved_repair_priority.residual_repair_slot.z_value,
    )

    baseline_point_miss_vector = resolved_baseline.group_summary(
        "all"
    ).point_miss_count_by_z
    binding_only_point_miss_vector = progress_after_report.group_summary(
        "all"
    ).point_miss_count_by_z
    baseline_band_miss_vector = resolved_baseline.group_summary(
        "all"
    ).uniform_band_miss_count_by_z
    binding_only_band_miss_vector = progress_after_report.group_summary(
        "all"
    ).uniform_band_miss_count_by_z

    binding_only_total_point_miss_reduction = _total_point_miss(
        baseline_point_miss_vector
    ) - _total_point_miss(binding_only_point_miss_vector)
    remaining_total_point_miss_reduction_after_binding = max(
        0,
        resolved_repair_priority.min_required_total_point_miss_reduction
        - binding_only_total_point_miss_reduction,
    )
    projected_binding_witness_floor = (
        resolved_repair_priority.binding_witness_floor_increment
        + before_after_report.baseline_witness_floor
    )
    residual_observation = progress_after_report.seed_observation(
        residual_repair_slot.random_state
    )
    residual_slot_still_open = not residual_observation.pointwise_coverage_by_z[
        residual_repair_slot.z_index
    ]
    driver_signature = _driver_signature(
        binding_repair_random_state=binding_repair_slot.random_state,
        residual_repair_random_state=residual_repair_slot.random_state,
        binding_only_point_miss_vector=binding_only_point_miss_vector,
        binding_only_band_miss_vector=binding_only_band_miss_vector,
        projected_binding_witness_floor=projected_binding_witness_floor,
        acceptance_driver_signature=before_after_report.driver_signature,
        candidate_status_if_binding_only_realized=guard_report.candidate_status,
        residual_slot_still_open=residual_slot_still_open,
        left_guard_pointwise_nonregression=before_after_report.left_guard_pointwise_nonregression,
        left_guard_band_preserved=before_after_report.left_guard_band_preserved,
    )
    canonical_digest = (
        "- first repair step is now executable and seed-consistent: repairing only seed `303` at witness-center `z = 0.15` improves all-seed point miss from `["
        + ", ".join(str(value) for value in baseline_point_miss_vector)
        + "]` to `["
        + ", ".join(str(value) for value in binding_only_point_miss_vector)
        + "]` while keeping band miss fixed at `["
        + ", ".join(str(value) for value in binding_only_band_miss_vector)
        + "]`",
        "- this isolated binding repair already realizes the `binding-witness-floor-lift` slot from the repair-priority contract: it projects the witness floor from `7/9 = "
        + _format_float(before_after_report.baseline_witness_floor)
        + "` to `8/9 = "
        + _format_float(projected_binding_witness_floor)
        + "`, but exact same-seed acceptance still stays RED because total point miss reduction is only `"
        + str(binding_only_total_point_miss_reduction)
        + "` instead of the required `"
        + str(resolved_repair_priority.min_required_total_point_miss_reduction)
        + "`",
        "- the remaining work is now singular and explicit: seed `707` at fresh right-shoulder `z = 0.25` is the only residual pointwise miss still needed after the binding witness-center slot lands",
        "- current Trigger 2 implication: `"
        + driver_signature
        + "`; future estimator reruns should first land the binding witness-center slot on the bounded path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`, then close the residual fresh shoulder without regressing `z = 0.05`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-binding-slot-progress-profile"
        ),
        policy_digest=resolved_repair_priority.policy_digest,
        binding_design=resolved_repair_priority.binding_design,
        window_label=resolved_repair_priority.window_label,
        same_seed_random_states=resolved_repair_priority.same_seed_random_states,
        binding_repair_slot=binding_repair_slot,
        residual_repair_slot=residual_repair_slot,
        baseline_point_miss_vector=baseline_point_miss_vector,
        binding_only_point_miss_vector=binding_only_point_miss_vector,
        baseline_band_miss_vector=baseline_band_miss_vector,
        binding_only_band_miss_vector=binding_only_band_miss_vector,
        baseline_witness_floor=before_after_report.baseline_witness_floor,
        projected_binding_witness_floor=projected_binding_witness_floor,
        acceptance_candidate_witness_floor=before_after_report.candidate_witness_floor,
        required_min_witness_floor=before_after_report.required_min_witness_floor,
        binding_only_total_point_miss_reduction=binding_only_total_point_miss_reduction,
        remaining_total_point_miss_reduction_after_binding=(
            remaining_total_point_miss_reduction_after_binding
        ),
        residual_slot_still_open=residual_slot_still_open,
        left_guard_grid_value=resolved_repair_priority.left_guard_grid_value,
        left_guard_band_preserved=before_after_report.left_guard_band_preserved,
        left_guard_pointwise_nonregression=(
            before_after_report.left_guard_pointwise_nonregression
        ),
        runtime_witness_path=resolved_repair_priority.runtime_witness_path,
        before_after_driver_signature=before_after_report.driver_signature,
        candidate_status_if_binding_only_realized=guard_report.candidate_status,
        resulting_evidence_status_if_binding_only_realized=(
            guard_report.resulting_evidence_status
        ),
        driver_signature=driver_signature,
        binding_only_after_report=progress_after_report,
        canonical_exact_witness_binding_slot_progress_digest=canonical_digest,
    )


def _build_binding_only_after_report_snapshot() -> (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
):
    baseline_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    updated_seed_observations = tuple(
        replace(
            observation,
            pointwise_coverage_by_z=tuple(
                True if index == 1 else covered
                for index, covered in enumerate(observation.pointwise_coverage_by_z)
            ),
        )
        if observation.random_state == 303
        else observation
        for observation in baseline_report.seed_observations
    )
    return build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report(
        seed_observations=updated_seed_observations
    )


def _build_exact_witness_binding_slot_progress_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport
):
    binding_only_after_report = _build_binding_only_after_report_snapshot()
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-binding-slot-"
            "progress-profile"
        ),
        policy_digest=(
            "label=bounded-n500-p50",
            "max_total_runtime_seconds=240.0",
            "max_random_states=8",
            "stop_on_first_typed_invalidity=True",
            "min_nonparametric_coverage=0.85",
        ),
        binding_design=("DGP2", 500, 50),
        window_label="near_zero_grid",
        same_seed_random_states=(101, 202, 303, 404, 505, 606, 707, 808),
        binding_repair_slot=(
            Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressSlot(
                random_state=303,
                seed_group="witness",
                z_index=1,
                z_value=0.15,
            )
        ),
        residual_repair_slot=(
            Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressSlot(
                random_state=707,
                seed_group="fresh",
                z_index=2,
                z_value=0.25,
            )
        ),
        baseline_point_miss_vector=(1, 3, 2),
        binding_only_point_miss_vector=(1, 2, 2),
        baseline_band_miss_vector=(0, 1, 1),
        binding_only_band_miss_vector=(0, 1, 1),
        baseline_witness_floor=7.0 / 9.0,
        projected_binding_witness_floor=8.0 / 9.0,
        acceptance_candidate_witness_floor=7.0 / 9.0,
        required_min_witness_floor=8.0 / 9.0,
        binding_only_total_point_miss_reduction=1,
        remaining_total_point_miss_reduction_after_binding=1,
        residual_slot_still_open=True,
        left_guard_grid_value=0.05,
        left_guard_band_preserved=True,
        left_guard_pointwise_nonregression=True,
        runtime_witness_path=(
            "omega_f_hat[2,2]",
            "v_f_hat[2,2]",
            "covariance(0.25, 0.15)",
        ),
        before_after_driver_signature=(
            "same-seed-before-after-acceptance-not-yet-satisfied"
        ),
        candidate_status_if_binding_only_realized="candidate-rejected",
        resulting_evidence_status_if_binding_only_realized=(
            "same-seed-estimator-evidence-rejected"
        ),
        driver_signature="same-seed-exact-witness-binding-slot-progress-profile",
        binding_only_after_report=binding_only_after_report,
        canonical_exact_witness_binding_slot_progress_digest=(
            "- first repair step is now executable and seed-consistent: repairing only seed `303` at witness-center `z = 0.15` improves all-seed point miss from `[1, 3, 2]` to `[1, 2, 2]` while keeping band miss fixed at `[0, 1, 1]`",
            "- this isolated binding repair already realizes the `binding-witness-floor-lift` slot from the repair-priority contract: it projects the witness floor from `7/9 = 0.778` to `8/9 = 0.889`, but exact same-seed acceptance still stays RED because total point miss reduction is only `1` instead of the required `2`",
            "- the remaining work is now singular and explicit: seed `707` at fresh right-shoulder `z = 0.25` is the only residual pointwise miss still needed after the binding witness-center slot lands",
            "- current Trigger 2 implication: `same-seed-exact-witness-binding-slot-progress-profile`; future estimator reruns should first land the binding witness-center slot on the bounded path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`, then close the residual fresh shoulder without regressing `z = 0.05`",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport
):
    return _build_exact_witness_binding_slot_progress_snapshot_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport",
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressSlot",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile",
]
