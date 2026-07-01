from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_floor_lift_acceptance_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedFloorLiftAcceptanceContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_floor_lift_acceptance_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportBeforeAfterIntakeContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportEstimatorEvidenceCandidateGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard_report,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation,
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


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


def _select_seed_observation(
    baseline_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    *,
    seed_group: str,
    pointwise_coverage_by_z: tuple[bool, bool, bool],
    uniform_band_coverage_by_z: tuple[bool, bool, bool],
) -> Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation:
    matches = tuple(
        observation
        for observation in baseline_report.seed_observations
        if observation.seed_group == seed_group
        and observation.pointwise_coverage_by_z == pointwise_coverage_by_z
        and observation.uniform_band_coverage_by_z == uniform_band_coverage_by_z
    )
    if len(matches) != 1:
        raise ValueError(
            "exact witness candidate profile requires a unique seed-level slot repair target"
        )
    return matches[0]


def _flip_pointwise_slot(
    observation: Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation,
    *,
    z_index: int,
) -> Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation:
    coverage_by_z = list(observation.pointwise_coverage_by_z)
    coverage_by_z[int(z_index)] = True
    return replace(
        observation,
        pointwise_coverage_by_z=tuple(coverage_by_z),
    )


def _build_target_after_report(
    baseline_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
) -> Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport:
    center_witness_candidate = _select_seed_observation(
        baseline_report,
        seed_group="witness",
        pointwise_coverage_by_z=(True, False, True),
        uniform_band_coverage_by_z=(True, True, True),
    )
    right_shoulder_fresh_candidate = _select_seed_observation(
        baseline_report,
        seed_group="fresh",
        pointwise_coverage_by_z=(True, True, False),
        uniform_band_coverage_by_z=(True, True, True),
    )

    updated_seed_observations = tuple(
        _flip_pointwise_slot(observation, z_index=1)
        if observation.random_state == center_witness_candidate.random_state
        else _flip_pointwise_slot(observation, z_index=2)
        if observation.random_state == right_shoulder_fresh_candidate.random_state
        else observation
        for observation in baseline_report.seed_observations
    )
    return build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report(
        seed_observations=updated_seed_observations,
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessCandidateProfileReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    synthetic_profile: bool
    baseline_point_miss_vector: tuple[int, int, int]
    target_point_miss_vector: tuple[int, int, int]
    baseline_band_miss_vector: tuple[int, int, int]
    target_band_miss_vector: tuple[int, int, int]
    total_point_miss_reduction: int
    lifted_witness_slots: int
    baseline_witness_floor: float
    target_witness_floor: float
    required_min_witness_floor: float
    left_guard_grid_value: float
    residual_lane_grid_values: tuple[float, float]
    left_guard_band_preserved: bool
    left_guard_pointwise_nonregression: bool
    runtime_witness_path: tuple[str, ...]
    candidate_status_if_realized: str
    resulting_evidence_status_if_realized: str
    target_after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
    )
    canonical_exact_witness_candidate_profile_digest: tuple[str, ...]

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
        self.synthetic_profile = bool(self.synthetic_profile)
        self.baseline_point_miss_vector = tuple(
            int(value) for value in self.baseline_point_miss_vector
        )
        self.target_point_miss_vector = tuple(
            int(value) for value in self.target_point_miss_vector
        )
        self.baseline_band_miss_vector = tuple(
            int(value) for value in self.baseline_band_miss_vector
        )
        self.target_band_miss_vector = tuple(
            int(value) for value in self.target_band_miss_vector
        )
        self.total_point_miss_reduction = int(self.total_point_miss_reduction)
        self.lifted_witness_slots = int(self.lifted_witness_slots)
        self.baseline_witness_floor = float(self.baseline_witness_floor)
        self.target_witness_floor = float(self.target_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.left_guard_grid_value = float(self.left_guard_grid_value)
        self.residual_lane_grid_values = tuple(
            float(value) for value in self.residual_lane_grid_values
        )
        self.left_guard_band_preserved = bool(self.left_guard_band_preserved)
        self.left_guard_pointwise_nonregression = bool(
            self.left_guard_pointwise_nonregression
        )
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.candidate_status_if_realized = str(
            self.candidate_status_if_realized
        ).strip()
        self.resulting_evidence_status_if_realized = str(
            self.resulting_evidence_status_if_realized
        ).strip()
        self.canonical_exact_witness_candidate_profile_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_exact_witness_candidate_profile_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_candidate_profile_report(
    *,
    baseline_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    floor_lift_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedFloorLiftAcceptanceContractReport
        | None
    ) = None,
    intake_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportBeforeAfterIntakeContractReport
        | None
    ) = None,
    runtime_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessCandidateProfileReport:
    resolved_baseline = (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        if baseline_report is None
        else baseline_report
    )
    resolved_floor_lift = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_floor_lift_acceptance_contract()
        if floor_lift_report is None
        else floor_lift_report
    )
    resolved_intake = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract()
        if intake_contract_report is None
        else intake_contract_report
    )
    resolved_runtime_contract = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
        if runtime_contract_report is None
        else runtime_contract_report
    )

    if resolved_baseline.policy_digest != resolved_intake.policy_digest:
        raise ValueError(
            "exact witness candidate profile requires shared policy digest"
        )
    if resolved_baseline.binding_design != resolved_intake.binding_design:
        raise ValueError(
            "exact witness candidate profile requires shared binding design"
        )
    if resolved_baseline.window_label != resolved_intake.window_label:
        raise ValueError("exact witness candidate profile requires shared window label")

    target_after_report = _build_target_after_report(resolved_baseline)
    guard_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard_report(
        runtime_candidate=_canonical_runtime_candidate(resolved_runtime_contract),
        before_report=resolved_baseline,
        after_report=target_after_report,
        intake_contract_report=resolved_intake,
        runtime_contract_report=resolved_runtime_contract,
    )
    if not guard_report.accepted:
        raise ValueError(
            "exact witness candidate profile requires an admissible same-seed target"
        )

    target_all_summary = target_after_report.group_summary("all")
    total_point_miss_reduction = sum(resolved_intake.baseline_point_miss_vector) - sum(
        target_all_summary.point_miss_count_by_z
    )
    left_guard_band_preserved = bool(
        target_all_summary.uniform_band_miss_count_by_z[0]
        == resolved_intake.baseline_band_miss_vector[0]
        == 0
    )
    left_guard_pointwise_nonregression = bool(
        target_all_summary.point_miss_count_by_z[0]
        <= resolved_intake.baseline_point_miss_vector[0]
    )
    canonical_digest = (
        "- the current same-seed baseline remains point miss `[1, 3, 2]`, band miss `[0, 1, 1]`, and witness floor `7/9 = 0.778`, so the minimal admissible exact-witness target must recover exactly one witness slot rather than rely on fresh-only rerun volume",
        "- the canonical exact-witness target profile is synthetic but executable: it keeps band miss fixed at `[0, 1, 1]`, improves point miss to `[1, 2, 1]`, and therefore maps the same-seed witness floor to `8/9 = 0.889` without regressing the left guard `z = 0.05`",
        "- the bounded object path remains unchanged: future estimator evidence must still travel on `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` and keep any residual miss pressure inside `z in {0.15, 0.25}`",
        "- current Trigger 2 implication: `candidate-matches-exact-witness-contract` / `same-seed-estimator-evidence-admissible` describe the first acceptable observed same-seed floor-lift profile, but this helper itself remains validation-only and does not promote live routing",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessCandidateProfileReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-candidate-profile"
        ),
        policy_digest=resolved_intake.policy_digest,
        binding_design=resolved_intake.binding_design,
        window_label=resolved_intake.window_label,
        same_seed_random_states=resolved_intake.same_seed_random_states,
        synthetic_profile=True,
        baseline_point_miss_vector=resolved_intake.baseline_point_miss_vector,
        target_point_miss_vector=target_all_summary.point_miss_count_by_z,
        baseline_band_miss_vector=resolved_intake.baseline_band_miss_vector,
        target_band_miss_vector=target_all_summary.uniform_band_miss_count_by_z,
        total_point_miss_reduction=total_point_miss_reduction,
        lifted_witness_slots=resolved_floor_lift.required_additional_witnesses_for_floor,
        baseline_witness_floor=resolved_intake.baseline_witness_floor,
        target_witness_floor=guard_report.candidate_witness_floor,
        required_min_witness_floor=resolved_intake.required_min_witness_floor,
        left_guard_grid_value=resolved_floor_lift.left_guard_grid_value,
        residual_lane_grid_values=(
            resolved_floor_lift.center_grid_value,
            resolved_floor_lift.right_shoulder_grid_value,
        ),
        left_guard_band_preserved=left_guard_band_preserved,
        left_guard_pointwise_nonregression=left_guard_pointwise_nonregression,
        runtime_witness_path=resolved_intake.runtime_witness_path,
        candidate_status_if_realized=guard_report.candidate_status,
        resulting_evidence_status_if_realized=guard_report.resulting_evidence_status,
        target_after_report=target_after_report,
        canonical_exact_witness_candidate_profile_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_candidate_profile() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessCandidateProfileReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_candidate_profile_report()
