from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_rerun_spend_order import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessRerunSpendOrderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_rerun_spend_order,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_design_key(binding_design: tuple[str, int, int]) -> str:
    return f"{binding_design[0]}/{binding_design[1]}/{binding_design[2]}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFreshEstimatorRerunIntakeContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    live_driver_signature: str
    spend_order_driver_signature: str
    intake_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    shared_vf_entry_label: str
    binding_budget_label: str
    binding_budget_share: float
    source_required_omega_diagonal_increment: float
    required_diagonal_vf_entry_lift: float
    validation_patch_increment: float
    patch_frobenius_share_of_baseline: float
    patch_spectral_share_of_baseline: float
    companion_right_row_replay_multiple_vs_required_increment: float
    companion_center_column_replay_multiple_vs_required_increment: float
    effective_fresh_reruns: int
    supported_floor_ceiling: float
    canonical_floor: float
    canonical_floor_shortfall: float
    required_evidence_conditions: tuple[str, ...]
    spend_order_steps: tuple[str, ...]
    contract_holds: bool
    driver_signature: str
    current_implication: str
    canonical_fresh_estimator_rerun_intake_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.live_driver_signature = str(self.live_driver_signature).strip()
        self.spend_order_driver_signature = str(
            self.spend_order_driver_signature
        ).strip()
        self.intake_driver_signature = str(self.intake_driver_signature).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.binding_budget_label = str(self.binding_budget_label).strip()
        self.binding_budget_share = float(self.binding_budget_share)
        self.source_required_omega_diagonal_increment = float(
            self.source_required_omega_diagonal_increment
        )
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.validation_patch_increment = float(self.validation_patch_increment)
        self.patch_frobenius_share_of_baseline = float(
            self.patch_frobenius_share_of_baseline
        )
        self.patch_spectral_share_of_baseline = float(
            self.patch_spectral_share_of_baseline
        )
        self.companion_right_row_replay_multiple_vs_required_increment = float(
            self.companion_right_row_replay_multiple_vs_required_increment
        )
        self.companion_center_column_replay_multiple_vs_required_increment = float(
            self.companion_center_column_replay_multiple_vs_required_increment
        )
        self.effective_fresh_reruns = int(self.effective_fresh_reruns)
        self.supported_floor_ceiling = float(self.supported_floor_ceiling)
        self.canonical_floor = float(self.canonical_floor)
        self.canonical_floor_shortfall = float(self.canonical_floor_shortfall)
        self.required_evidence_conditions = tuple(
            str(item).strip() for item in self.required_evidence_conditions
        )
        self.spend_order_steps = tuple(
            str(item).strip() for item in self.spend_order_steps
        )
        self.contract_holds = bool(self.contract_holds)
        self.driver_signature = str(self.driver_signature).strip()
        self.current_implication = str(self.current_implication).strip()
        self.canonical_fresh_estimator_rerun_intake_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_fresh_estimator_rerun_intake_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "live_driver_signature": self.live_driver_signature,
            "spend_order_driver_signature": self.spend_order_driver_signature,
            "intake_driver_signature": self.intake_driver_signature,
            "runtime_witness_path": list(self.runtime_witness_path),
            "source_diagonal_coordinate": self.source_diagonal_coordinate,
            "source_diagonal_basis_label": self.source_diagonal_basis_label,
            "shared_vf_entry_label": self.shared_vf_entry_label,
            "binding_budget_label": self.binding_budget_label,
            "binding_budget_share": self.binding_budget_share,
            "source_required_omega_diagonal_increment": (
                self.source_required_omega_diagonal_increment
            ),
            "required_diagonal_vf_entry_lift": self.required_diagonal_vf_entry_lift,
            "validation_patch_increment": self.validation_patch_increment,
            "patch_frobenius_share_of_baseline": (
                self.patch_frobenius_share_of_baseline
            ),
            "patch_spectral_share_of_baseline": (self.patch_spectral_share_of_baseline),
            "companion_right_row_replay_multiple_vs_required_increment": (
                self.companion_right_row_replay_multiple_vs_required_increment
            ),
            "companion_center_column_replay_multiple_vs_required_increment": (
                self.companion_center_column_replay_multiple_vs_required_increment
            ),
            "effective_fresh_reruns": self.effective_fresh_reruns,
            "supported_floor_ceiling": self.supported_floor_ceiling,
            "canonical_floor": self.canonical_floor,
            "canonical_floor_shortfall": self.canonical_floor_shortfall,
            "required_evidence_conditions": list(self.required_evidence_conditions),
            "spend_order_steps": list(self.spend_order_steps),
            "contract_holds": self.contract_holds,
            "driver_signature": self.driver_signature,
            "current_implication": self.current_implication,
            "canonical_fresh_estimator_rerun_intake_digest": list(
                self.canonical_fresh_estimator_rerun_intake_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract_report(
    *,
    spend_order_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessRerunSpendOrderReport
        | None
    ) = None,
    intake_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFreshEstimatorRerunIntakeContractReport:
    resolved_spend_order = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_rerun_spend_order()
        if spend_order_report is None
        else spend_order_report
    )
    resolved_intake_contract = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract()
        if intake_contract_report is None
        else intake_contract_report
    )

    if resolved_spend_order.policy_digest != resolved_intake_contract.policy_digest:
        raise ValueError(
            "fresh estimator rerun intake contract requires shared policy digest"
        )
    if resolved_spend_order.binding_design != resolved_intake_contract.binding_design:
        raise ValueError(
            "fresh estimator rerun intake contract requires shared binding design"
        )
    if resolved_spend_order.window_label != resolved_intake_contract.window_label:
        raise ValueError(
            "fresh estimator rerun intake contract requires shared window label"
        )
    if (
        resolved_spend_order.coverage_anchor_random_state
        != resolved_intake_contract.coverage_anchor_random_state
    ):
        raise ValueError(
            "fresh estimator rerun intake contract requires shared coverage-anchor seed"
        )
    if (
        resolved_spend_order.overshoot_companion_random_state
        != resolved_intake_contract.overshoot_companion_random_state
    ):
        raise ValueError(
            "fresh estimator rerun intake contract requires shared overshoot-companion seed"
        )

    expected_runtime_witness_path = (
        f"omega_f_hat[{resolved_intake_contract.source_diagonal_coordinate},{resolved_intake_contract.source_diagonal_coordinate}]",
        resolved_intake_contract.shared_vf_entry_label,
        "covariance(0.25, 0.15)",
    )
    if resolved_spend_order.runtime_witness_path != expected_runtime_witness_path:
        raise ValueError(
            "fresh estimator rerun intake contract requires the bounded runtime witness "
            "path to match the fresh estimator evidence object flow"
        )

    contract_holds = bool(
        resolved_spend_order.spend_order_holds
        and resolved_intake_contract.intake_contract_holds
        and resolved_spend_order.live_driver_signature
        == "bounded-right-center-execution-contract"
        and resolved_spend_order.driver_signature
        == "runtime-witness-rerun-spend-order-disciplined"
        and resolved_intake_contract.driver_signature
        == "bounded-entry-patch-fresh-estimator-evidence-intake-contract"
        and resolved_spend_order.canonical_floor_shortfall > 0.0
        and resolved_intake_contract.canonical_floor_shortfall > 0.0
    )
    driver_signature = (
        "fresh-estimator-rerun-intake-contract-disciplined"
        if contract_holds
        else "mixed-fresh-estimator-rerun-intake-contract"
    )
    current_implication = (
        "spend-fresh-reruns-only-after-bounded-estimator-intake-and-floor-lift-evidence"
    )
    binding_design_key = _format_design_key(resolved_spend_order.binding_design)

    canonical_digest = (
        f"- fresh rerun spend still stays chained to `{resolved_spend_order.live_driver_signature}`: before any rerun budget is worth spending, the first estimator effort remains the positive first-sine diagonal omega lift on `{' -> '.join(resolved_spend_order.runtime_witness_path)}`, with binding budget share `{_format_percent(resolved_spend_order.binding_budget_share)}` on `{resolved_spend_order.window_label}` for `{binding_design_key}`",
        f"- the bounded intake itself still has to obey the same source-backed object flow: coordinate `{resolved_intake_contract.source_diagonal_coordinate} = {resolved_intake_contract.source_diagonal_basis_label}` must lift `omega_f_hat[2,2]` by `{_format_signed_float(resolved_intake_contract.source_required_omega_diagonal_increment)}`, then lift `{resolved_intake_contract.shared_vf_entry_label}` by `{_format_signed_float(resolved_intake_contract.required_diagonal_vf_entry_lift)}` to explain the raw covariance patch `{_format_signed_float(resolved_intake_contract.validation_patch_increment)}`; widening into dense replay remains invalid because the bounded patch only uses `{_format_percent(resolved_intake_contract.patch_frobenius_share_of_baseline)}` / `{_format_percent(resolved_intake_contract.patch_spectral_share_of_baseline)}` of baseline Frobenius / spectral mass while row / column replay would overshoot by `{_format_ratio(resolved_intake_contract.companion_right_row_replay_multiple_vs_required_increment)}` / `{_format_ratio(resolved_intake_contract.companion_center_column_replay_multiple_vs_required_increment)}`",
        f"- only `{resolved_spend_order.effective_fresh_reruns}` fresh reruns remain, and even an admissible rerun keeps Trigger 2 pinned below the canonical floor: `nonparametric_coverage_floor = {_format_float(resolved_spend_order.canonical_floor)}` still sits `{_format_float(resolved_spend_order.canonical_floor_shortfall)}` above ceiling `{_format_float(resolved_spend_order.supported_floor_ceiling)}`, so reruns are informative only when they preserve the bounded intake and lift the binding floor witness",
        f"- current Trigger 2 implication: `{driver_signature}`; use this as validation-only companion evidence for future estimator-path work, not as a replacement for the live `bounded-right-center-execution-contract` token",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFreshEstimatorRerunIntakeContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-fresh-estimator-rerun-intake-contract"
        ),
        policy_digest=resolved_spend_order.policy_digest,
        binding_design=resolved_spend_order.binding_design,
        window_label=resolved_spend_order.window_label,
        coverage_anchor_random_state=resolved_spend_order.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            resolved_spend_order.overshoot_companion_random_state
        ),
        live_driver_signature=resolved_spend_order.live_driver_signature,
        spend_order_driver_signature=resolved_spend_order.driver_signature,
        intake_driver_signature=resolved_intake_contract.driver_signature,
        runtime_witness_path=resolved_spend_order.runtime_witness_path,
        source_diagonal_coordinate=(
            resolved_intake_contract.source_diagonal_coordinate
        ),
        source_diagonal_basis_label=(
            resolved_intake_contract.source_diagonal_basis_label
        ),
        shared_vf_entry_label=resolved_intake_contract.shared_vf_entry_label,
        binding_budget_label=resolved_spend_order.binding_budget_label,
        binding_budget_share=resolved_spend_order.binding_budget_share,
        source_required_omega_diagonal_increment=(
            resolved_intake_contract.source_required_omega_diagonal_increment
        ),
        required_diagonal_vf_entry_lift=(
            resolved_intake_contract.required_diagonal_vf_entry_lift
        ),
        validation_patch_increment=resolved_intake_contract.validation_patch_increment,
        patch_frobenius_share_of_baseline=(
            resolved_intake_contract.patch_frobenius_share_of_baseline
        ),
        patch_spectral_share_of_baseline=(
            resolved_intake_contract.patch_spectral_share_of_baseline
        ),
        companion_right_row_replay_multiple_vs_required_increment=(
            resolved_intake_contract.companion_right_row_replay_multiple_vs_required_increment
        ),
        companion_center_column_replay_multiple_vs_required_increment=(
            resolved_intake_contract.companion_center_column_replay_multiple_vs_required_increment
        ),
        effective_fresh_reruns=resolved_spend_order.effective_fresh_reruns,
        supported_floor_ceiling=resolved_spend_order.supported_floor_ceiling,
        canonical_floor=resolved_spend_order.canonical_floor,
        canonical_floor_shortfall=resolved_spend_order.canonical_floor_shortfall,
        required_evidence_conditions=(
            resolved_intake_contract.required_evidence_conditions
        ),
        spend_order_steps=resolved_spend_order.spend_order_steps,
        contract_holds=contract_holds,
        driver_signature=driver_signature,
        current_implication=current_implication,
        canonical_fresh_estimator_rerun_intake_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFreshEstimatorRerunIntakeContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract_report()
