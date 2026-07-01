from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_runtime_bridge import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportRuntimeBridgeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_runtime_bridge,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportBeforeAfterIntakeContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    admission_order_driver_signature: str
    before_after_driver_signature: str
    runtime_bridge_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    baseline_point_miss_vector: tuple[int, int, int]
    candidate_point_miss_vector: tuple[int, int, int]
    baseline_band_miss_vector: tuple[int, int, int]
    candidate_band_miss_vector: tuple[int, int, int]
    baseline_witness_floor: float
    candidate_witness_floor: float
    required_min_witness_floor: float
    preserves_left_support_contract: bool
    diagonal_preserved: bool
    direct_covariance_edits_allowed: bool
    required_patch_share_of_full_shared_vf_gap: float
    required_patch_share_of_omega_only_shared_vf_increment: float
    required_patch_share_of_diagonal_omega_gap: float
    required_patch_share_of_psd_boundary: float
    compensating_stage_order: tuple[str, ...]
    effective_fresh_reruns: int
    required_evidence_conditions: tuple[str, ...]
    admission_order: tuple[str, ...]
    contract_holds: bool
    driver_signature: str
    current_implication: str
    canonical_same_seed_preserve_left_support_before_after_intake_digest: tuple[
        str, ...
    ]

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
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.before_after_driver_signature = str(
            self.before_after_driver_signature
        ).strip()
        self.runtime_bridge_driver_signature = str(
            self.runtime_bridge_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.baseline_point_miss_vector = tuple(
            int(value) for value in self.baseline_point_miss_vector
        )
        self.candidate_point_miss_vector = tuple(
            int(value) for value in self.candidate_point_miss_vector
        )
        self.baseline_band_miss_vector = tuple(
            int(value) for value in self.baseline_band_miss_vector
        )
        self.candidate_band_miss_vector = tuple(
            int(value) for value in self.candidate_band_miss_vector
        )
        self.baseline_witness_floor = float(self.baseline_witness_floor)
        self.candidate_witness_floor = float(self.candidate_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.preserves_left_support_contract = bool(
            self.preserves_left_support_contract
        )
        self.diagonal_preserved = bool(self.diagonal_preserved)
        self.direct_covariance_edits_allowed = bool(
            self.direct_covariance_edits_allowed
        )
        self.required_patch_share_of_full_shared_vf_gap = float(
            self.required_patch_share_of_full_shared_vf_gap
        )
        self.required_patch_share_of_omega_only_shared_vf_increment = float(
            self.required_patch_share_of_omega_only_shared_vf_increment
        )
        self.required_patch_share_of_diagonal_omega_gap = float(
            self.required_patch_share_of_diagonal_omega_gap
        )
        self.required_patch_share_of_psd_boundary = float(
            self.required_patch_share_of_psd_boundary
        )
        self.compensating_stage_order = tuple(
            str(item).strip() for item in self.compensating_stage_order
        )
        self.effective_fresh_reruns = int(self.effective_fresh_reruns)
        self.required_evidence_conditions = tuple(
            str(item).strip() for item in self.required_evidence_conditions
        )
        self.admission_order = tuple(str(item).strip() for item in self.admission_order)
        self.contract_holds = bool(self.contract_holds)
        self.driver_signature = str(self.driver_signature).strip()
        self.current_implication = str(self.current_implication).strip()
        self.canonical_same_seed_preserve_left_support_before_after_intake_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_same_seed_preserve_left_support_before_after_intake_digest
        )


def _driver_signature(*, contract_holds: bool) -> str:
    if contract_holds:
        return "same-seed-preserve-left-support-before-after-intake-contract"
    return "mixed-same-seed-preserve-left-support-before-after-intake-contract"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract_report(
    *,
    admission_order_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport
    | None = None,
    before_after_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport
    | None = None,
    runtime_bridge_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportRuntimeBridgeReport
    | None = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportBeforeAfterIntakeContractReport:
    resolved_admission_order = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order()
        if admission_order_report is None
        else admission_order_report
    )
    resolved_before_after = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe()
        if before_after_report is None
        else before_after_report
    )
    resolved_runtime_bridge = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_runtime_bridge()
        if runtime_bridge_report is None
        else runtime_bridge_report
    )

    if resolved_admission_order.policy_digest != resolved_before_after.policy_digest:
        raise ValueError(
            "same-seed preserve-left-support before/after intake contract requires shared policy digest"
        )
    if resolved_admission_order.policy_digest != resolved_runtime_bridge.policy_digest:
        raise ValueError(
            "same-seed preserve-left-support before/after intake contract requires shared policy digest"
        )
    if resolved_admission_order.binding_design != resolved_before_after.binding_design:
        raise ValueError(
            "same-seed preserve-left-support before/after intake contract requires shared binding design"
        )
    if (
        resolved_admission_order.binding_design
        != resolved_runtime_bridge.binding_design
    ):
        raise ValueError(
            "same-seed preserve-left-support before/after intake contract requires shared binding design"
        )
    if resolved_admission_order.window_label != resolved_before_after.window_label:
        raise ValueError(
            "same-seed preserve-left-support before/after intake contract requires shared window label"
        )
    if resolved_admission_order.window_label != resolved_runtime_bridge.window_label:
        raise ValueError(
            "same-seed preserve-left-support before/after intake contract requires shared window label"
        )
    if (
        resolved_admission_order.same_seed_random_states
        != resolved_before_after.same_seed_random_states
    ):
        raise ValueError(
            "same-seed preserve-left-support before/after intake contract requires shared same-seed states"
        )
    if (
        resolved_admission_order.same_seed_random_states
        != resolved_runtime_bridge.same_seed_random_states
    ):
        raise ValueError(
            "same-seed preserve-left-support before/after intake contract requires shared same-seed states"
        )

    required_runtime_path = (
        "omega_f_hat[2,2]",
        "v_f_hat[2,2]",
        "covariance(0.25, 0.15)",
    )
    if resolved_runtime_bridge.runtime_witness_path != required_runtime_path:
        raise ValueError(
            "same-seed preserve-left-support before/after intake contract requires the bounded preserve-left-support runtime path"
        )

    required_evidence_conditions = (
        "replay the exact same random states `(101, 202, 303, 404, 505, 606, 707, 808)` and show before/after improvement rather than a fresh-only rerun",
        "keep the bounded object flow on `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` while preserving left support and diagonal invariance",
        "treat direct covariance edits as forbidden and consume compensating geometry in the diagonal-first order before claiming floor-lift evidence",
        "do not claim admission until the witness floor lifts from `7/9` to at least `8/9` without adding any left-band miss at `z = 0.05`",
    )
    contract_holds = bool(
        resolved_admission_order.driver_signature == "same-seed-admission-order"
        and resolved_before_after.driver_signature
        == "same-seed-before-after-acceptance-not-yet-satisfied"
        and resolved_runtime_bridge.driver_signature
        == "same-seed-preserve-left-support-runtime-bridge"
        and resolved_before_after.baseline_point_miss_vector
        == resolved_before_after.candidate_point_miss_vector
        == resolved_runtime_bridge.baseline_point_miss_vector
        and resolved_before_after.baseline_band_miss_vector
        == resolved_before_after.candidate_band_miss_vector
        == resolved_runtime_bridge.baseline_band_miss_vector
        and resolved_before_after.baseline_witness_floor
        == resolved_before_after.candidate_witness_floor
        == resolved_runtime_bridge.baseline_witness_floor
        and resolved_before_after.required_min_witness_floor
        == resolved_runtime_bridge.required_min_witness_floor
        and resolved_runtime_bridge.preserves_left_support_contract
        and resolved_runtime_bridge.diagonal_preserved
        and not resolved_runtime_bridge.direct_covariance_edits_allowed
    )
    driver_signature = _driver_signature(contract_holds=contract_holds)
    current_implication = (
        "exact-same-seed-estimator-evidence-must-follow-preserve-left-support-bridge"
    )
    canonical_digest = (
        "- exact same-seed admission is still the active gate: the current bounded lane remains on `same-seed-admission-order`, so the next admissible estimator evidence must replay `(101, 202, 303, 404, 505, 606, 707, 808)` before any fresh-only rerun can matter",
        "- current before/after evidence is intentionally still RED on uplift: baseline and candidate both keep point miss `[1, 3, 2]`, band miss `[0, 1, 1]`, and witness floor `7/9 = 0.778`, so acceptance is not yet satisfied until an observed lift reaches `8/9 = 0.889`",
        "- the only safe exact-witness path is the preserve-left-support bridge `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`, which keeps left-support preservation and diagonal invariance hard while still using only `12.0%` / `17.9%` / `24.2%` / `12.7%` of the bounded full-gap / omega-only / diagonal-gap / PSD budgets",
        "- current Trigger 2 implication: `same-seed-preserve-left-support-before-after-intake-contract`; future estimator evidence should be admitted only when it follows this preserve-left-support same-seed bridge and flips the before/after acceptance probe, not when it merely spends fresh reruns",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportBeforeAfterIntakeContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-before-after-intake-contract"
        ),
        policy_digest=resolved_admission_order.policy_digest,
        binding_design=resolved_admission_order.binding_design,
        window_label=resolved_admission_order.window_label,
        same_seed_random_states=resolved_admission_order.same_seed_random_states,
        admission_order_driver_signature=resolved_admission_order.driver_signature,
        before_after_driver_signature=resolved_before_after.driver_signature,
        runtime_bridge_driver_signature=resolved_runtime_bridge.driver_signature,
        runtime_witness_path=resolved_runtime_bridge.runtime_witness_path,
        source_diagonal_coordinate=resolved_runtime_bridge.source_diagonal_coordinate,
        source_diagonal_basis_label=resolved_runtime_bridge.source_diagonal_basis_label,
        baseline_point_miss_vector=resolved_before_after.baseline_point_miss_vector,
        candidate_point_miss_vector=resolved_before_after.candidate_point_miss_vector,
        baseline_band_miss_vector=resolved_before_after.baseline_band_miss_vector,
        candidate_band_miss_vector=resolved_before_after.candidate_band_miss_vector,
        baseline_witness_floor=resolved_before_after.baseline_witness_floor,
        candidate_witness_floor=resolved_before_after.candidate_witness_floor,
        required_min_witness_floor=resolved_before_after.required_min_witness_floor,
        preserves_left_support_contract=resolved_runtime_bridge.preserves_left_support_contract,
        diagonal_preserved=resolved_runtime_bridge.diagonal_preserved,
        direct_covariance_edits_allowed=resolved_runtime_bridge.direct_covariance_edits_allowed,
        required_patch_share_of_full_shared_vf_gap=(
            resolved_runtime_bridge.required_patch_share_of_full_shared_vf_gap
        ),
        required_patch_share_of_omega_only_shared_vf_increment=(
            resolved_runtime_bridge.required_patch_share_of_omega_only_shared_vf_increment
        ),
        required_patch_share_of_diagonal_omega_gap=(
            resolved_runtime_bridge.required_patch_share_of_diagonal_omega_gap
        ),
        required_patch_share_of_psd_boundary=(
            resolved_runtime_bridge.required_patch_share_of_psd_boundary
        ),
        compensating_stage_order=resolved_runtime_bridge.compensating_stage_order,
        effective_fresh_reruns=resolved_admission_order.effective_fresh_reruns,
        required_evidence_conditions=required_evidence_conditions,
        admission_order=resolved_admission_order.admission_order,
        contract_holds=contract_holds,
        driver_signature=driver_signature,
        current_implication=current_implication,
        canonical_same_seed_preserve_left_support_before_after_intake_digest=(
            canonical_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportBeforeAfterIntakeContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportBeforeAfterIntakeContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract",
]
