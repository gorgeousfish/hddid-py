from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate,
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidateGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportBeforeAfterIntakeContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract,
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


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportEstimatorEvidenceCandidateGuardReport:
    stage_label: str
    accepted: bool
    candidate_status: str
    resulting_evidence_status: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    runtime_witness_path: tuple[str, ...]
    intake_contract_driver_signature: str
    runtime_candidate_status: str
    runtime_contract_status: str
    before_after_driver_signature: str
    candidate_point_miss_vector: tuple[int, int, int]
    candidate_band_miss_vector: tuple[int, int, int]
    candidate_witness_floor: float
    required_min_witness_floor: float
    rejection_reasons: tuple[str, ...]
    max_budget_share_slack: float
    min_compensating_stage_coverage_slack: float
    zero_live_entry_slack: int
    canonical_candidate_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.accepted = bool(self.accepted)
        self.candidate_status = str(self.candidate_status).strip()
        self.resulting_evidence_status = str(self.resulting_evidence_status).strip()
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
        self.intake_contract_driver_signature = str(
            self.intake_contract_driver_signature
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
        self.candidate_witness_floor = float(self.candidate_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.rejection_reasons = tuple(
            str(item).strip() for item in self.rejection_reasons
        )
        self.max_budget_share_slack = float(self.max_budget_share_slack)
        self.min_compensating_stage_coverage_slack = float(
            self.min_compensating_stage_coverage_slack
        )
        self.zero_live_entry_slack = int(self.zero_live_entry_slack)
        self.canonical_candidate_digest = tuple(
            str(item).rstrip() for item in self.canonical_candidate_digest
        )


def _candidate_status(
    *,
    accepted: bool,
    runtime_candidate_status: str,
) -> str:
    if not accepted:
        return "candidate-rejected"
    if runtime_candidate_status == "candidate-tightens-within-runtime-contract":
        return "candidate-tightens-within-exact-witness-contract"
    return "candidate-matches-exact-witness-contract"


def _canonical_digest(
    *,
    accepted: bool,
    runtime_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidateGuardReport
    ),
    before_after_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport
    ),
) -> tuple[str, ...]:
    if not accepted:
        return (
            "- exact same-seed estimator evidence remains RED by default: runtime witness candidate still matches the preserve-left-support contract, but before/after acceptance remains `same-seed-before-after-acceptance-not-yet-satisfied`, so current evidence stays inadmissible until observed floor lift reaches `8/9 = 0.889` without new left-band miss at `z = 0.05`",
            "- canonical runtime witness discipline is unchanged: candidate path stays `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` with left-support preservation, diagonal invariance, and direct-covariance-edit prohibition all still hard requirements",
            f"- current exact same-seed readout stays at point miss `{list(before_after_report.candidate_point_miss_vector)}`, band miss `{list(before_after_report.candidate_band_miss_vector)}`, and witness floor `7/9 = {_format_float(before_after_report.candidate_witness_floor)}`, so admission still fails even though the bounded runtime candidate itself is contract-satisfying",
            "- current Trigger 2 implication: `candidate-rejected`; future exact same-seed estimator evidence becomes admissible only when it both satisfies the preserve-left-support runtime contract and flips before/after acceptance to observed floor lift",
        )

    return (
        f"- exact same-seed estimator evidence is now admissible: runtime candidate status is `{runtime_guard_report.candidate_status}` and before/after acceptance has flipped to `{before_after_report.driver_signature}` on the same `8` random states",
        "- preserve-left-support runtime discipline remains bounded: candidate still stays on `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` with left-support preservation, diagonal invariance, and no direct covariance edits",
        f"- current exact same-seed readout now improves to point miss `{list(before_after_report.candidate_point_miss_vector)}`, band miss `{list(before_after_report.candidate_band_miss_vector)}`, and witness floor `{_format_float(before_after_report.candidate_witness_floor)}`, which clears the required `8/9 = {_format_float(before_after_report.required_min_witness_floor)}` threshold",
        "- current Trigger 2 implication: `same-seed-estimator-evidence-admissible`; future exact same-seed evidence can now be consumed without relaxing the preserve-left-support bridge",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard_report(
    *,
    runtime_candidate: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
    ),
    after_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    before_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
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
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportEstimatorEvidenceCandidateGuardReport:
    resolved_before = (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        if before_report is None
        else before_report
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

    runtime_guard_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard_report(
        candidate=runtime_candidate,
        contract_report=resolved_runtime_contract,
    )
    before_after_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report(
        before_report=resolved_before,
        after_report=after_report,
    )

    if resolved_intake.policy_digest != before_after_report.policy_digest:
        raise ValueError(
            "same-seed estimator evidence candidate guard requires shared policy digest"
        )
    if resolved_intake.binding_design != before_after_report.binding_design:
        raise ValueError(
            "same-seed estimator evidence candidate guard requires shared binding design"
        )
    if resolved_intake.window_label != before_after_report.window_label:
        raise ValueError(
            "same-seed estimator evidence candidate guard requires shared window label"
        )
    if (
        resolved_intake.runtime_witness_path
        != runtime_guard_report.canonical_runtime_witness_path
    ):
        raise ValueError(
            "same-seed estimator evidence candidate guard requires the canonical preserve-left-support runtime path"
        )

    rejection_reasons = list(runtime_guard_report.rejection_reasons)
    if not resolved_intake.contract_holds:
        rejection_reasons.append(
            "same-seed preserve-left-support before/after intake contract must remain green before consuming exact estimator evidence"
        )
    if not before_after_report.acceptance_passed:
        rejection_reasons.append(
            "same-seed before/after acceptance remains `same-seed-before-after-acceptance-not-yet-satisfied` until observed floor lift reaches `8/9 = 0.889` without adding any left-band miss at `z = 0.05`"
        )

    accepted = bool(
        resolved_intake.contract_holds
        and runtime_guard_report.accepted
        and before_after_report.acceptance_passed
        and runtime_guard_report.resulting_contract_status
        == "runtime-witness-contract-satisfied"
    )
    candidate_status = _candidate_status(
        accepted=accepted,
        runtime_candidate_status=runtime_guard_report.candidate_status,
    )
    resulting_evidence_status = (
        "same-seed-estimator-evidence-admissible"
        if accepted
        else "same-seed-estimator-evidence-rejected"
    )
    canonical_digest = _canonical_digest(
        accepted=accepted,
        runtime_guard_report=runtime_guard_report,
        before_after_report=before_after_report,
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportEstimatorEvidenceCandidateGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-estimator-evidence-candidate-guard"
        ),
        accepted=accepted,
        candidate_status=candidate_status,
        resulting_evidence_status=resulting_evidence_status,
        policy_digest=resolved_intake.policy_digest,
        binding_design=resolved_intake.binding_design,
        window_label=resolved_intake.window_label,
        same_seed_random_states=resolved_intake.same_seed_random_states,
        runtime_witness_path=runtime_candidate.runtime_witness_path,
        intake_contract_driver_signature=resolved_intake.driver_signature,
        runtime_candidate_status=runtime_guard_report.candidate_status,
        runtime_contract_status=runtime_guard_report.resulting_contract_status,
        before_after_driver_signature=before_after_report.driver_signature,
        candidate_point_miss_vector=before_after_report.candidate_point_miss_vector,
        candidate_band_miss_vector=before_after_report.candidate_band_miss_vector,
        candidate_witness_floor=before_after_report.candidate_witness_floor,
        required_min_witness_floor=before_after_report.required_min_witness_floor,
        rejection_reasons=tuple(rejection_reasons),
        max_budget_share_slack=runtime_guard_report.max_budget_share_slack,
        min_compensating_stage_coverage_slack=(
            runtime_guard_report.min_compensating_stage_coverage_slack
        ),
        zero_live_entry_slack=runtime_guard_report.zero_live_entry_slack,
        canonical_candidate_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportEstimatorEvidenceCandidateGuardReport
):
    runtime_contract_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard_report(
        runtime_candidate=_canonical_runtime_candidate(runtime_contract_report),
        after_report=seedwise_report,
        before_report=seedwise_report,
        runtime_contract_report=runtime_contract_report,
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportEstimatorEvidenceCandidateGuardReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_estimator_evidence_candidate_guard",
]
