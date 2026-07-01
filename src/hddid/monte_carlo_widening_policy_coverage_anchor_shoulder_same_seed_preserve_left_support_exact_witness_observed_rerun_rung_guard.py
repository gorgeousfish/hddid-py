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
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_candidate_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCandidateGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_candidate_guard_report,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_exact_trim_floor(value: float) -> str:
    return f"{float(value):.6f}"


_ACCEPTANCE_READOUT_PATH = ("omega_f_hat[2,2]", "v_f_hat[2,1]", "bar_f_at_z0[1]")
_EXACT_TRIM_FLOOR_VALUE = 0.010001
_TARGET_FOLD3_LOW_PI_TREATED_COUNT = 13
_TARGET_EXACT_TRIM_FLOOR_COUNT = 1
_TARGET_EXACT_TRIM_FLOOR_SHARE_OF_FOLD3_LOW_PI_COUNT = 1.0 / 13.0
_TARGET_EXACT_TRIM_FLOOR_RAW_PHI1_SHARE_OF_FOLD3_RAW = 0.0932355649789316
_TARGET_EXACT_TRIM_FLOOR_INVERSE_PI_SHARE_OF_FOLD3_INVERSE_PI = 0.4314558010132362
_TARGET_EXACT_TRIM_FLOOR_WEIGHTED_SHARE_OF_FOLD3_WEIGHTED = 0.44786838884494945


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


def _binding_slot_candidate_status(
    *,
    binding_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardReport
    ),
    residual_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCandidateGuardReport
    ),
) -> str:
    if residual_guard_report.accepted:
        return "candidate-surpasses-binding-slot-progress-profile"
    if binding_guard_report.accepted:
        return binding_guard_report.candidate_status
    return "candidate-below-binding-slot-progress-profile"


def _residual_slot_candidate_status(
    *,
    residual_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCandidateGuardReport
    ),
) -> str:
    if residual_guard_report.accepted:
        return residual_guard_report.candidate_status
    return "candidate-below-residual-slot-completion-profile"


def _binding_slot_landed(
    *,
    binding_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotCandidateGuardReport
    ),
    residual_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCandidateGuardReport
    ),
) -> bool:
    return bool(binding_guard_report.accepted or residual_guard_report.accepted)


def _resulting_rung_status(
    *,
    binding_slot_landed: bool,
    residual_slot_landed: bool,
) -> str:
    if residual_slot_landed:
        return "same-seed-exact-witness-residual-slot-completion-landed"
    if binding_slot_landed:
        return "same-seed-exact-witness-binding-slot-progress-landed"
    return "same-seed-exact-witness-observed-rerun-open"


def _highest_landed_rung(
    *,
    binding_slot_landed: bool,
    residual_slot_landed: bool,
) -> str:
    if residual_slot_landed:
        return "residual-slot-completion"
    if binding_slot_landed:
        return "binding-slot-progress"
    return "baseline"


def _acceptance_exact_trim_floor_inverse_pi_concentration_status(
    *,
    binding_slot_landed: bool,
    residual_slot_landed: bool,
) -> str:
    if residual_slot_landed:
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-closed"
    if binding_slot_landed:
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-residual-only"
    return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-open"


def _acceptance_exact_trim_floor_landing_bridge_status(
    *,
    binding_slot_landed: bool,
    residual_slot_landed: bool,
) -> str:
    if residual_slot_landed:
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-closed"
    if binding_slot_landed:
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-residual-only"
    return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-open"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardSlot:
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
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport:
    stage_label: str
    binding_slot_landed: bool
    residual_slot_landed: bool
    binding_slot_candidate_status: str
    binding_slot_progress_status: str
    residual_slot_candidate_status: str
    residual_slot_completion_status: str
    current_rung_candidate_status: str
    resulting_rung_status: str
    downstream_evidence_status: str
    highest_landed_rung: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    runtime_witness_path: tuple[str, ...]
    runtime_candidate_status: str
    runtime_contract_status: str
    before_after_driver_signature: str
    acceptance_exact_trim_floor_landing_bridge_status: str
    acceptance_exact_trim_floor_inverse_pi_concentration_status: str
    acceptance_readout_path: tuple[str, ...]
    exact_trim_floor_value: float
    target_fold3_low_pi_treated_count: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_share_of_fold3_low_pi_count: float
    target_exact_trim_floor_raw_phi1_share_of_fold3_raw: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    candidate_point_miss_vector: tuple[int, int, int]
    candidate_band_miss_vector: tuple[int, int, int]
    acceptance_candidate_witness_floor: float
    required_min_witness_floor: float
    binding_repair_slot: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardSlot
    residual_repair_slot: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardSlot
    next_required_slot: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardSlot
        | None
    )
    rejection_reasons: tuple[str, ...]
    max_budget_share_slack: float
    min_compensating_stage_coverage_slack: float
    zero_live_entry_slack: int
    canonical_observed_rerun_rung_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.binding_slot_landed = bool(self.binding_slot_landed)
        self.residual_slot_landed = bool(self.residual_slot_landed)
        self.binding_slot_candidate_status = str(
            self.binding_slot_candidate_status
        ).strip()
        self.binding_slot_progress_status = str(
            self.binding_slot_progress_status
        ).strip()
        self.residual_slot_candidate_status = str(
            self.residual_slot_candidate_status
        ).strip()
        self.residual_slot_completion_status = str(
            self.residual_slot_completion_status
        ).strip()
        self.current_rung_candidate_status = str(
            self.current_rung_candidate_status
        ).strip()
        self.resulting_rung_status = str(self.resulting_rung_status).strip()
        self.downstream_evidence_status = str(self.downstream_evidence_status).strip()
        self.highest_landed_rung = str(self.highest_landed_rung).strip()
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
        self.runtime_candidate_status = str(self.runtime_candidate_status).strip()
        self.runtime_contract_status = str(self.runtime_contract_status).strip()
        self.before_after_driver_signature = str(
            self.before_after_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_landing_bridge_status = str(
            self.acceptance_exact_trim_floor_landing_bridge_status
        ).strip()
        self.acceptance_exact_trim_floor_inverse_pi_concentration_status = str(
            self.acceptance_exact_trim_floor_inverse_pi_concentration_status
        ).strip()
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.exact_trim_floor_value = float(self.exact_trim_floor_value)
        self.target_fold3_low_pi_treated_count = int(
            self.target_fold3_low_pi_treated_count
        )
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_share_of_fold3_low_pi_count = float(
            self.target_exact_trim_floor_share_of_fold3_low_pi_count
        )
        self.target_exact_trim_floor_raw_phi1_share_of_fold3_raw = float(
            self.target_exact_trim_floor_raw_phi1_share_of_fold3_raw
        )
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.candidate_point_miss_vector = tuple(
            int(value) for value in self.candidate_point_miss_vector
        )
        self.candidate_band_miss_vector = tuple(
            int(value) for value in self.candidate_band_miss_vector
        )
        self.acceptance_candidate_witness_floor = float(
            self.acceptance_candidate_witness_floor
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
        self.canonical_observed_rerun_rung_digest = tuple(
            str(line).rstrip() for line in self.canonical_observed_rerun_rung_digest
        )


def _slot(
    *,
    random_state: int,
    seed_group: str,
    z_index: int,
    z_value: float,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardSlot:
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardSlot(
        random_state=random_state,
        seed_group=seed_group,
        z_index=z_index,
        z_value=z_value,
    )


def _canonical_digest(
    *,
    binding_slot_landed: bool,
    residual_slot_landed: bool,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport
        | None
    ) = None,
) -> tuple[str, ...]:
    assert report is not None
    if residual_slot_landed:
        return (
            f"- observed exact same-seed rerun now lands the full admissible ladder: point miss reaches `{list(report.candidate_point_miss_vector)}`, band miss stays `{list(report.candidate_band_miss_vector)}`, and witness floor holds at `8/9 = {_format_float(report.acceptance_candidate_witness_floor)}`",
            "- the single ladder guard now shows both steps as closed in order: the binding witness-center slot at seed `303` / `z = 0.15` is already absorbed, and the residual fresh shoulder at seed `707` / `z = 0.25` has also landed",
            f"- the same single ladder guard now closes the landing packet as `{report.acceptance_exact_trim_floor_landing_bridge_status}` while `{report.acceptance_exact_trim_floor_inverse_pi_concentration_status}` keeps the seed `303` row-vs-rest split as audit-only provenance once both the binding slot and the queued residual slot are fully absorbed",
            "- bounded runtime discipline still holds: the observed rerun remains on `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` and keeps the left guard `z = 0.05` untouched",
            "- current Trigger 2 implication: `same-seed-exact-witness-residual-slot-completion-landed`; this observed rerun can now be treated as admissible without hand-checking the intermediate guards one by one",
        )
    if binding_slot_landed:
        return (
            f"- observed exact same-seed rerun now clears the first executable rung: point miss improves to `{list(report.candidate_point_miss_vector)}` while band miss stays `{list(report.candidate_band_miss_vector)}`, so the binding witness-center slot is no longer open",
            "- the single ladder guard keeps the remaining obligation explicit: seed `303` / `z = 0.15` is already landed, but seed `707` / `z = 0.25` still remains the final residual repair before admissibility can turn GREEN",
            f"- the same single ladder guard now demotes the landing packet to `{report.acceptance_exact_trim_floor_landing_bridge_status}` and keeps `{report.acceptance_exact_trim_floor_inverse_pi_concentration_status}` underneath it: the seed `303` row-vs-rest split at `pi_hat = {_format_exact_trim_floor(report.exact_trim_floor_value)}` stays historical provenance while seed `707` / `z = 0.25` remains the only queued residual spend",
            "- bounded runtime discipline is unchanged: future reruns must keep the preserve-left-support path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` and leave the left guard `z = 0.05` untouched",
            "- current Trigger 2 implication: `same-seed-exact-witness-binding-slot-progress-landed`; route the next observed rerun directly to the residual-slot completion check instead of replaying the binding-slot guard manually",
        )
    return (
        f"- observed exact same-seed rerun still sits below the first executable rung: point miss stays `{list(report.candidate_point_miss_vector)}`, band miss stays `{list(report.candidate_band_miss_vector)}`, and witness floor therefore remains `7/9 = {_format_float(report.acceptance_candidate_witness_floor)}`",
        "- ladder ordering is now machine-readable in one place: seed `303` at witness-center `z = 0.15` remains the next required repair, while seed `707` at fresh right-shoulder `z = 0.25` stays reserved for the later residual step",
        f"- the same single ladder guard now carries `{report.acceptance_exact_trim_floor_landing_bridge_status}` together with `{report.acceptance_exact_trim_floor_inverse_pi_concentration_status}`: seed `303` keeps the only exact trim-floor row `pi_hat = {_format_exact_trim_floor(report.exact_trim_floor_value)}`, so `{report.target_exact_trim_floor_count}/{report.target_fold3_low_pi_treated_count} = {_format_percent(report.target_exact_trim_floor_share_of_fold3_low_pi_count)}` of fold `3` low-`pi_hat` treated rows still account for `{_format_percent(report.target_exact_trim_floor_raw_phi1_share_of_fold3_raw)}` of raw `phi1_hat`, `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the gross inverse-`pi_hat` conduit, and `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden before any seed `707` spend",
        "- bounded runtime discipline is unchanged: future reruns must keep the preserve-left-support path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` and leave the left guard `z = 0.05` untouched",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-open`; feed future observed reruns through this single ladder guard using the same trim-floor row-vs-rest split before claiming either binding-slot landing or full residual completion",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard_report(
    *,
    after_report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    runtime_candidate: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
        | None
    ) = None,
    before_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    runtime_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport:
    resolved_before = (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        if before_report is None
        else before_report
    )
    resolved_runtime_contract = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
        if runtime_contract_report is None
        else runtime_contract_report
    )
    resolved_runtime_candidate = (
        _canonical_runtime_candidate(resolved_runtime_contract)
        if runtime_candidate is None
        else runtime_candidate
    )
    binding_guard_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard_report(
        runtime_candidate=resolved_runtime_candidate,
        after_report=after_report,
        before_report=resolved_before,
        runtime_contract_report=resolved_runtime_contract,
    )
    residual_guard_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_candidate_guard_report(
        runtime_candidate=resolved_runtime_candidate,
        after_report=after_report,
        before_report=resolved_before,
        runtime_contract_report=resolved_runtime_contract,
    )

    if binding_guard_report.policy_digest != residual_guard_report.policy_digest:
        raise ValueError(
            "observed rerun rung guard requires shared policy digest across binding and residual guards"
        )
    if binding_guard_report.binding_design != residual_guard_report.binding_design:
        raise ValueError(
            "observed rerun rung guard requires shared binding design across binding and residual guards"
        )
    if binding_guard_report.window_label != residual_guard_report.window_label:
        raise ValueError(
            "observed rerun rung guard requires shared window label across binding and residual guards"
        )
    if (
        binding_guard_report.same_seed_random_states
        != residual_guard_report.same_seed_random_states
    ):
        raise ValueError(
            "observed rerun rung guard requires shared same-seed random states across binding and residual guards"
        )
    binding_slot_landed = _binding_slot_landed(
        binding_guard_report=binding_guard_report,
        residual_guard_report=residual_guard_report,
    )
    residual_slot_landed = bool(residual_guard_report.accepted)

    binding_slot_candidate_status = _binding_slot_candidate_status(
        binding_guard_report=binding_guard_report,
        residual_guard_report=residual_guard_report,
    )
    residual_slot_candidate_status = _residual_slot_candidate_status(
        residual_guard_report=residual_guard_report
    )

    binding_repair_slot = _slot(
        random_state=303,
        seed_group="witness",
        z_index=1,
        z_value=0.15,
    )
    residual_repair_slot = _slot(
        random_state=707,
        seed_group="fresh",
        z_index=2,
        z_value=0.25,
    )
    next_required_slot = (
        None
        if residual_slot_landed
        else residual_repair_slot
        if binding_slot_landed
        else binding_repair_slot
    )

    current_rung_candidate_status = (
        residual_slot_candidate_status
        if residual_slot_landed
        else binding_slot_candidate_status
    )
    resulting_rung_status = _resulting_rung_status(
        binding_slot_landed=binding_slot_landed,
        residual_slot_landed=residual_slot_landed,
    )
    highest_landed_rung = _highest_landed_rung(
        binding_slot_landed=binding_slot_landed,
        residual_slot_landed=residual_slot_landed,
    )
    rejection_reasons = (
        ()
        if residual_slot_landed
        else residual_guard_report.rejection_reasons
        if binding_slot_landed
        else binding_guard_report.rejection_reasons
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-observed-rerun-rung-guard"
        ),
        binding_slot_landed=binding_slot_landed,
        residual_slot_landed=residual_slot_landed,
        binding_slot_candidate_status=binding_slot_candidate_status,
        binding_slot_progress_status=(
            "same-seed-exact-witness-binding-slot-progress-landed"
            if binding_slot_landed
            else "same-seed-exact-witness-binding-slot-progress-open"
        ),
        residual_slot_candidate_status=residual_slot_candidate_status,
        residual_slot_completion_status=residual_guard_report.resulting_completion_status,
        current_rung_candidate_status=current_rung_candidate_status,
        resulting_rung_status=resulting_rung_status,
        downstream_evidence_status=(
            residual_guard_report.downstream_evidence_status
            if residual_slot_landed
            else binding_guard_report.downstream_evidence_status
        ),
        highest_landed_rung=highest_landed_rung,
        policy_digest=binding_guard_report.policy_digest,
        binding_design=binding_guard_report.binding_design,
        window_label=binding_guard_report.window_label,
        same_seed_random_states=binding_guard_report.same_seed_random_states,
        runtime_witness_path=binding_guard_report.runtime_witness_path,
        runtime_candidate_status=residual_guard_report.runtime_candidate_status,
        runtime_contract_status=residual_guard_report.runtime_contract_status,
        before_after_driver_signature=residual_guard_report.before_after_driver_signature,
        acceptance_exact_trim_floor_landing_bridge_status=(
            _acceptance_exact_trim_floor_landing_bridge_status(
                binding_slot_landed=binding_slot_landed,
                residual_slot_landed=residual_slot_landed,
            )
        ),
        acceptance_exact_trim_floor_inverse_pi_concentration_status=(
            _acceptance_exact_trim_floor_inverse_pi_concentration_status(
                binding_slot_landed=binding_slot_landed,
                residual_slot_landed=residual_slot_landed,
            )
        ),
        acceptance_readout_path=_ACCEPTANCE_READOUT_PATH,
        exact_trim_floor_value=_EXACT_TRIM_FLOOR_VALUE,
        target_fold3_low_pi_treated_count=_TARGET_FOLD3_LOW_PI_TREATED_COUNT,
        target_exact_trim_floor_count=_TARGET_EXACT_TRIM_FLOOR_COUNT,
        target_exact_trim_floor_share_of_fold3_low_pi_count=_TARGET_EXACT_TRIM_FLOOR_SHARE_OF_FOLD3_LOW_PI_COUNT,
        target_exact_trim_floor_raw_phi1_share_of_fold3_raw=_TARGET_EXACT_TRIM_FLOOR_RAW_PHI1_SHARE_OF_FOLD3_RAW,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=_TARGET_EXACT_TRIM_FLOOR_INVERSE_PI_SHARE_OF_FOLD3_INVERSE_PI,
        target_exact_trim_floor_weighted_share_of_fold3_weighted=_TARGET_EXACT_TRIM_FLOOR_WEIGHTED_SHARE_OF_FOLD3_WEIGHTED,
        candidate_point_miss_vector=residual_guard_report.candidate_point_miss_vector,
        candidate_band_miss_vector=residual_guard_report.candidate_band_miss_vector,
        acceptance_candidate_witness_floor=(
            residual_guard_report.acceptance_candidate_witness_floor
        ),
        required_min_witness_floor=residual_guard_report.required_min_witness_floor,
        binding_repair_slot=binding_repair_slot,
        residual_repair_slot=residual_repair_slot,
        next_required_slot=next_required_slot,
        rejection_reasons=rejection_reasons,
        max_budget_share_slack=residual_guard_report.max_budget_share_slack,
        min_compensating_stage_coverage_slack=(
            residual_guard_report.min_compensating_stage_coverage_slack
        ),
        zero_live_entry_slack=residual_guard_report.zero_live_entry_slack,
        canonical_observed_rerun_rung_digest=(),
    )
    report.canonical_observed_rerun_rung_digest = _canonical_digest(
        binding_slot_landed=binding_slot_landed,
        residual_slot_landed=residual_slot_landed,
        report=report,
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard_report(
        after_report=run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport",
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardSlot",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard",
]
