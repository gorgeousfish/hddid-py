from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportEstimatorEvidenceCandidateGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
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


def _candidate_status(
    *,
    accepted: bool,
    runtime_candidate_status: str,
) -> str:
    if not accepted:
        return "candidate-rejected"
    if runtime_candidate_status == "candidate-tightens-within-runtime-contract":
        return "candidate-tightens-within-binding-slot-progress-profile"
    return "candidate-matches-binding-slot-progress-profile"


def _accepted(
    *,
    exact_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportEstimatorEvidenceCandidateGuardReport
    ),
    binding_slot_progress_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport
    ),
) -> bool:
    return bool(
        exact_guard_report.runtime_contract_status
        == "runtime-witness-contract-satisfied"
        and exact_guard_report.candidate_point_miss_vector
        == binding_slot_progress_report.binding_only_point_miss_vector
        and exact_guard_report.candidate_band_miss_vector
        == binding_slot_progress_report.binding_only_band_miss_vector
        and exact_guard_report.before_after_driver_signature
        == binding_slot_progress_report.before_after_driver_signature
        and exact_guard_report.candidate_witness_floor
        == binding_slot_progress_report.acceptance_candidate_witness_floor
        and exact_guard_report.required_min_witness_floor
        == binding_slot_progress_report.required_min_witness_floor
        and exact_guard_report.resulting_evidence_status
        == binding_slot_progress_report.resulting_evidence_status_if_binding_only_realized
    )


def _canonical_digest(
    *,
    accepted: bool,
    exact_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportEstimatorEvidenceCandidateGuardReport
    ),
    binding_slot_progress_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport
    ),
) -> tuple[str, ...]:
    if not accepted:
        return (
            f"- exact same-seed live rerun still sits below the first-step binding profile: candidate point miss stays `{list(exact_guard_report.candidate_point_miss_vector)}` instead of the binding-only `{list(binding_slot_progress_report.binding_only_point_miss_vector)}`, band miss stays `{list(exact_guard_report.candidate_band_miss_vector)}`, and the acceptance-facing witness floor therefore remains `7/9 = {_format_float(exact_guard_report.candidate_witness_floor)}`",
            "- the missing progress is still singular and witness-only: seed `303` must recover the witness-center slot at `z = 0.15` before seed `707` at fresh right-shoulder `z = 0.25` becomes the only residual repair",
            "- bounded runtime discipline is unchanged: future same-seed reruns must keep the preserve-left-support path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` and continue to leave the left guard `z = 0.05` untouched",
            "- current Trigger 2 implication: `same-seed-exact-witness-binding-slot-progress-open`; future reruns should first land the binding witness-center slot, then hand the remaining shoulder repair to the residual-slot completion profile",
        )
    return (
        f"- binding-slot progress is now landed and seed-consistent: candidate point miss matches the first-step profile `{list(exact_guard_report.candidate_point_miss_vector)}` while band miss stays `{list(exact_guard_report.candidate_band_miss_vector)}` on the same `8` random states",
        f"- this first-step landing now realizes the witness-only lift encoded by `same-seed-exact-witness-binding-slot-progress-profile`: the witness-group projection reaches `8/9 = {_format_float(binding_slot_progress_report.projected_binding_witness_floor)}`, but exact same-seed acceptance still remains `{binding_slot_progress_report.before_after_driver_signature}` and downstream evidence therefore stays RED",
        "- the next repair is now singular and explicit: seed `707` at fresh right-shoulder `z = 0.25` is the only residual slot still needed after the binding witness-center slot lands, while the left guard `z = 0.05` remains unchanged",
        "- current Trigger 2 implication: `same-seed-exact-witness-binding-slot-progress-landed`; future reruns should now spend only on the residual shoulder repair or route a full admissible profile through `same-seed-estimator-evidence-admissible` checks",
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardSlot:
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
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardReport:
    stage_label: str
    accepted: bool
    candidate_status: str
    resulting_binding_progress_status: str
    downstream_evidence_status: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    runtime_witness_path: tuple[str, ...]
    progress_profile_driver_signature: str
    runtime_candidate_status: str
    runtime_contract_status: str
    before_after_driver_signature: str
    candidate_point_miss_vector: tuple[int, int, int]
    candidate_band_miss_vector: tuple[int, int, int]
    acceptance_candidate_witness_floor: float
    projected_binding_witness_floor: float
    required_min_witness_floor: float
    residual_repair_slot: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardSlot
    rejection_reasons: tuple[str, ...]
    max_budget_share_slack: float
    min_compensating_stage_coverage_slack: float
    zero_live_entry_slack: int
    canonical_binding_slot_candidate_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.accepted = bool(self.accepted)
        self.candidate_status = str(self.candidate_status).strip()
        self.resulting_binding_progress_status = str(
            self.resulting_binding_progress_status
        ).strip()
        self.downstream_evidence_status = str(self.downstream_evidence_status).strip()
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
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.progress_profile_driver_signature = str(
            self.progress_profile_driver_signature
        ).strip()
        self.runtime_candidate_status = str(self.runtime_candidate_status).strip()
        self.runtime_contract_status = str(self.runtime_contract_status).strip()
        self.before_after_driver_signature = str(
            self.before_after_driver_signature
        ).strip()
        self.candidate_point_miss_vector = tuple(
            int(value) for value in self.candidate_point_miss_vector
        )
        self.candidate_band_miss_vector = tuple(
            int(value) for value in self.candidate_band_miss_vector
        )
        self.acceptance_candidate_witness_floor = float(
            self.acceptance_candidate_witness_floor
        )
        self.projected_binding_witness_floor = float(
            self.projected_binding_witness_floor
        )
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.rejection_reasons = tuple(
            str(item).strip() for item in self.rejection_reasons
        )
        self.max_budget_share_slack = float(self.max_budget_share_slack)
        self.min_compensating_stage_coverage_slack = float(
            self.min_compensating_stage_coverage_slack
        )
        self.zero_live_entry_slack = int(self.zero_live_entry_slack)
        self.canonical_binding_slot_candidate_digest = tuple(
            str(line).rstrip() for line in self.canonical_binding_slot_candidate_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard_report(
    *,
    runtime_candidate: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
    ),
    after_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    before_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    binding_slot_progress_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport
        | None
    ) = None,
    runtime_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardReport:
    resolved_before = (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        if before_report is None
        else before_report
    )
    resolved_binding_progress = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile()
        if binding_slot_progress_report is None
        else binding_slot_progress_report
    )
    resolved_runtime_contract = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
        if runtime_contract_report is None
        else runtime_contract_report
    )
    exact_guard_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard_report(
        runtime_candidate=runtime_candidate,
        after_report=after_report,
        before_report=resolved_before,
        runtime_contract_report=resolved_runtime_contract,
    )

    if resolved_binding_progress.policy_digest != exact_guard_report.policy_digest:
        raise ValueError(
            "same-seed binding-slot candidate guard requires shared policy digest"
        )
    if resolved_binding_progress.binding_design != exact_guard_report.binding_design:
        raise ValueError(
            "same-seed binding-slot candidate guard requires shared binding design"
        )
    if resolved_binding_progress.window_label != exact_guard_report.window_label:
        raise ValueError(
            "same-seed binding-slot candidate guard requires shared window label"
        )
    if (
        resolved_binding_progress.same_seed_random_states
        != exact_guard_report.same_seed_random_states
    ):
        raise ValueError(
            "same-seed binding-slot candidate guard requires the canonical same-seed random-state roster"
        )
    if (
        resolved_binding_progress.runtime_witness_path
        != exact_guard_report.runtime_witness_path
    ):
        raise ValueError(
            "same-seed binding-slot candidate guard requires the canonical preserve-left-support runtime path"
        )

    accepted = _accepted(
        exact_guard_report=exact_guard_report,
        binding_slot_progress_report=resolved_binding_progress,
    )
    candidate_status = _candidate_status(
        accepted=accepted,
        runtime_candidate_status=exact_guard_report.runtime_candidate_status,
    )
    resulting_binding_progress_status = (
        "same-seed-exact-witness-binding-slot-progress-landed"
        if accepted
        else "same-seed-exact-witness-binding-slot-progress-open"
    )

    rejection_reasons: list[str] = []
    if (
        exact_guard_report.runtime_contract_status
        != "runtime-witness-contract-satisfied"
    ):
        rejection_reasons.append(
            "same-seed exact-witness binding-slot guard requires the preserve-left-support runtime contract to stay GREEN before candidate progress can be credited"
        )
    if not accepted:
        rejection_reasons.append(
            "same-seed exact-witness binding-slot progress remains open: candidate point miss must reach `[1, 2, 2]` with band miss `[0, 1, 1]` on the same `8` random states before seed `707` / `z = 0.25` becomes the only residual step"
        )

    residual_slot = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardSlot(
        random_state=resolved_binding_progress.residual_repair_slot.random_state,
        seed_group=resolved_binding_progress.residual_repair_slot.seed_group,
        z_index=resolved_binding_progress.residual_repair_slot.z_index,
        z_value=resolved_binding_progress.residual_repair_slot.z_value,
    )
    canonical_digest = _canonical_digest(
        accepted=accepted,
        exact_guard_report=exact_guard_report,
        binding_slot_progress_report=resolved_binding_progress,
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-binding-slot-candidate-guard"
        ),
        accepted=accepted,
        candidate_status=candidate_status,
        resulting_binding_progress_status=resulting_binding_progress_status,
        downstream_evidence_status=exact_guard_report.resulting_evidence_status,
        policy_digest=resolved_binding_progress.policy_digest,
        binding_design=resolved_binding_progress.binding_design,
        window_label=resolved_binding_progress.window_label,
        same_seed_random_states=resolved_binding_progress.same_seed_random_states,
        runtime_witness_path=exact_guard_report.runtime_witness_path,
        progress_profile_driver_signature=resolved_binding_progress.driver_signature,
        runtime_candidate_status=exact_guard_report.runtime_candidate_status,
        runtime_contract_status=exact_guard_report.runtime_contract_status,
        before_after_driver_signature=exact_guard_report.before_after_driver_signature,
        candidate_point_miss_vector=exact_guard_report.candidate_point_miss_vector,
        candidate_band_miss_vector=exact_guard_report.candidate_band_miss_vector,
        acceptance_candidate_witness_floor=exact_guard_report.candidate_witness_floor,
        projected_binding_witness_floor=(
            resolved_binding_progress.projected_binding_witness_floor
        ),
        required_min_witness_floor=exact_guard_report.required_min_witness_floor,
        residual_repair_slot=residual_slot,
        rejection_reasons=tuple(rejection_reasons),
        max_budget_share_slack=exact_guard_report.max_budget_share_slack,
        min_compensating_stage_coverage_slack=(
            exact_guard_report.min_compensating_stage_coverage_slack
        ),
        zero_live_entry_slack=exact_guard_report.zero_live_entry_slack,
        canonical_binding_slot_candidate_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardReport
):
    runtime_contract_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard_report(
        runtime_candidate=_canonical_runtime_candidate(runtime_contract_report),
        after_report=seedwise_report,
        before_report=seedwise_report,
        runtime_contract_report=runtime_contract_report,
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardReport",
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardSlot",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard",
]
