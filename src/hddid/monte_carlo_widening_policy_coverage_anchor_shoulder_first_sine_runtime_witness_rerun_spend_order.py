from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_acceptance_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunAcceptanceContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_acceptance_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunGateReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_budget_priority_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessBudgetPriorityProbeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_budget_priority_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidateGuardReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard,
)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessRerunSpendOrderReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    live_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    budget_priority_driver_signature: str
    candidate_status: str
    resulting_contract_status: str
    effective_fresh_reruns: int
    rerun_gate_driver_signature: str
    acceptance_driver_signature: str
    binding_budget_label: str
    binding_budget_share: float
    second_budget_label: str
    second_budget_share: float
    third_budget_label: str
    third_budget_share: float
    fourth_budget_label: str
    fourth_budget_share: float
    supported_floor_ceiling: float
    canonical_floor: float
    canonical_floor_shortfall: float
    spend_order_steps: tuple[str, ...]
    spend_order_holds: bool
    driver_signature: str
    current_implication: str
    canonical_spend_order_digest: tuple[str, ...]

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
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.budget_priority_driver_signature = str(
            self.budget_priority_driver_signature
        ).strip()
        self.candidate_status = str(self.candidate_status).strip()
        self.resulting_contract_status = str(self.resulting_contract_status).strip()
        self.effective_fresh_reruns = int(self.effective_fresh_reruns)
        self.rerun_gate_driver_signature = str(self.rerun_gate_driver_signature).strip()
        self.acceptance_driver_signature = str(self.acceptance_driver_signature).strip()
        self.binding_budget_label = str(self.binding_budget_label).strip()
        self.binding_budget_share = float(self.binding_budget_share)
        self.second_budget_label = str(self.second_budget_label).strip()
        self.second_budget_share = float(self.second_budget_share)
        self.third_budget_label = str(self.third_budget_label).strip()
        self.third_budget_share = float(self.third_budget_share)
        self.fourth_budget_label = str(self.fourth_budget_label).strip()
        self.fourth_budget_share = float(self.fourth_budget_share)
        self.supported_floor_ceiling = float(self.supported_floor_ceiling)
        self.canonical_floor = float(self.canonical_floor)
        self.canonical_floor_shortfall = float(self.canonical_floor_shortfall)
        self.spend_order_steps = tuple(
            str(step).strip() for step in self.spend_order_steps
        )
        self.spend_order_holds = bool(self.spend_order_holds)
        self.driver_signature = str(self.driver_signature).strip()
        self.current_implication = str(self.current_implication).strip()
        self.canonical_spend_order_digest = tuple(
            str(line).rstrip() for line in self.canonical_spend_order_digest
        )


def _driver_signature(*, spend_order_holds: bool) -> str:
    if spend_order_holds:
        return "runtime-witness-rerun-spend-order-disciplined"
    return "mixed-runtime-witness-rerun-spend-order"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_rerun_spend_order_report(
    *,
    budget_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessBudgetPriorityProbeReport
        | None
    ) = None,
    candidate_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidateGuardReport
        | None
    ) = None,
    rerun_gate_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunGateReport
        | None
    ) = None,
    acceptance_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunAcceptanceContractReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessRerunSpendOrderReport:
    resolved_budget = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_budget_priority_probe()
        if budget_priority_report is None
        else budget_priority_report
    )
    resolved_candidate = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard()
        if candidate_guard_report is None
        else candidate_guard_report
    )
    resolved_gate = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate()
        if rerun_gate_report is None
        else rerun_gate_report
    )
    resolved_acceptance = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_acceptance_contract()
        if acceptance_contract_report is None
        else acceptance_contract_report
    )

    if resolved_budget.policy_digest != resolved_candidate.policy_digest:
        raise ValueError(
            "runtime-witness rerun spend order requires shared policy digest"
        )
    if resolved_budget.policy_digest != resolved_gate.policy_digest:
        raise ValueError(
            "runtime-witness rerun spend order requires shared policy digest"
        )
    if resolved_budget.policy_digest != resolved_acceptance.policy_digest:
        raise ValueError(
            "runtime-witness rerun spend order requires shared policy digest"
        )
    if resolved_budget.binding_design != resolved_candidate.binding_design:
        raise ValueError(
            "runtime-witness rerun spend order requires shared binding design"
        )
    if resolved_budget.binding_design != resolved_gate.binding_design:
        raise ValueError(
            "runtime-witness rerun spend order requires shared binding design"
        )
    if resolved_budget.binding_design != resolved_acceptance.binding_design:
        raise ValueError(
            "runtime-witness rerun spend order requires shared binding design"
        )
    if resolved_budget.window_label != resolved_candidate.window_label:
        raise ValueError(
            "runtime-witness rerun spend order requires shared window label"
        )
    if resolved_budget.window_label != resolved_gate.window_label:
        raise ValueError(
            "runtime-witness rerun spend order requires shared window label"
        )
    if resolved_budget.window_label != resolved_acceptance.window_label:
        raise ValueError(
            "runtime-witness rerun spend order requires shared window label"
        )
    if (
        resolved_budget.coverage_anchor_random_state
        != resolved_gate.coverage_anchor_random_state
    ):
        raise ValueError(
            "runtime-witness rerun spend order requires shared coverage-anchor seed"
        )
    if (
        resolved_budget.coverage_anchor_random_state
        != resolved_acceptance.coverage_anchor_random_state
    ):
        raise ValueError(
            "runtime-witness rerun spend order requires shared coverage-anchor seed"
        )
    if (
        resolved_budget.overshoot_companion_random_state
        != resolved_gate.overshoot_companion_random_state
    ):
        raise ValueError(
            "runtime-witness rerun spend order requires shared overshoot-companion seed"
        )
    if (
        resolved_budget.overshoot_companion_random_state
        != resolved_acceptance.overshoot_companion_random_state
    ):
        raise ValueError(
            "runtime-witness rerun spend order requires shared overshoot-companion seed"
        )
    if (
        resolved_budget.runtime_witness_path
        != resolved_candidate.canonical_runtime_witness_path
    ):
        raise ValueError(
            "runtime-witness rerun spend order requires the canonical runtime witness path"
        )

    spend_order_steps = (
        resolved_gate.live_driver_signature,
        resolved_budget.current_implication,
        resolved_gate.driver_signature,
        resolved_acceptance.driver_signature,
    )
    spend_order_holds = bool(
        resolved_gate.live_driver_signature == "bounded-right-center-execution-contract"
        and resolved_budget.driver_signature
        == "first-sine-runtime-witness-budget-priority"
        and resolved_budget.current_implication
        == "prioritize-diagonal-omega-lift-before-psd-or-broad-replay-guards"
        and resolved_candidate.accepted
        and resolved_candidate.candidate_status == "candidate-matches-runtime-contract"
        and resolved_candidate.resulting_contract_status
        == "runtime-witness-contract-satisfied"
        and resolved_gate.rerun_gate_holds
        and resolved_gate.driver_signature
        == "fresh-rerun-gated-by-entry-patch-readiness"
        and resolved_acceptance.acceptance_contract_holds
        and resolved_acceptance.driver_signature
        == "fresh-rerun-must-carry-floor-lift-evidence"
        and resolved_gate.effective_fresh_reruns > 0
    )
    driver_signature = _driver_signature(spend_order_holds=spend_order_holds)
    current_implication = (
        "spend-fresh-reruns-only-after-runtime-witness-floor-lift-discipline"
    )

    digest = (
        f"- the live Trigger 2 repair still enters only through `{resolved_gate.live_driver_signature}`, so fresh rerun spend must stay chained to the bounded runtime witness path `{' -> '.join(resolved_budget.runtime_witness_path)}` on `{resolved_budget.window_label}`",
        f"- before any fresh rerun is worth spending, the first estimator effort still belongs to the positive first-sine diagonal omega lift: budget priority remains `{_format_percent(resolved_budget.binding_budget_share)}` versus `{_format_percent(resolved_budget.second_budget_share)}`, `{_format_percent(resolved_budget.third_budget_share)}`, and `{_format_percent(resolved_budget.fourth_budget_share)}`, while the candidate guard still reports `{resolved_candidate.candidate_status}` / `{resolved_candidate.resulting_contract_status}`",
        f"- only after that bounded runtime witness discipline holds may the remaining `{resolved_gate.effective_fresh_reruns}` fresh reruns be spent on binding design `{resolved_budget.binding_design[0]}/{resolved_budget.binding_design[1]}/{resolved_budget.binding_design[2]}`; the rerun gate still forbids `stable-slice-wide first-sine replay`, `coordinate-`2` off-diagonal fallback`, and `full-matrix widening replay`",
        f"- current Trigger 2 implication: `{driver_signature}`; even an admissible rerun is informative only if it preserves the bounded intake and delivers fresh estimator evidence that lifts the binding floor witness, because canonical `nonparametric_coverage_floor = {_format_float(resolved_acceptance.canonical_floor)}` still stays short by `{_format_float(resolved_acceptance.canonical_floor_shortfall)}` above ceiling `{_format_float(resolved_acceptance.supported_floor_ceiling)}`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessRerunSpendOrderReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-runtime-witness-rerun-spend-order"
        ),
        policy_digest=resolved_budget.policy_digest,
        binding_design=resolved_budget.binding_design,
        window_label=resolved_budget.window_label,
        coverage_anchor_random_state=resolved_budget.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            resolved_budget.overshoot_companion_random_state
        ),
        live_driver_signature=resolved_gate.live_driver_signature,
        runtime_witness_path=resolved_budget.runtime_witness_path,
        budget_priority_driver_signature=resolved_budget.driver_signature,
        candidate_status=resolved_candidate.candidate_status,
        resulting_contract_status=resolved_candidate.resulting_contract_status,
        effective_fresh_reruns=resolved_gate.effective_fresh_reruns,
        rerun_gate_driver_signature=resolved_gate.driver_signature,
        acceptance_driver_signature=resolved_acceptance.driver_signature,
        binding_budget_label=resolved_budget.binding_budget_label,
        binding_budget_share=resolved_budget.binding_budget_share,
        second_budget_label=resolved_budget.second_budget_label,
        second_budget_share=resolved_budget.second_budget_share,
        third_budget_label=resolved_budget.third_budget_label,
        third_budget_share=resolved_budget.third_budget_share,
        fourth_budget_label=resolved_budget.fourth_budget_label,
        fourth_budget_share=resolved_budget.fourth_budget_share,
        supported_floor_ceiling=resolved_acceptance.supported_floor_ceiling,
        canonical_floor=resolved_acceptance.canonical_floor,
        canonical_floor_shortfall=resolved_acceptance.canonical_floor_shortfall,
        spend_order_steps=spend_order_steps,
        spend_order_holds=spend_order_holds,
        driver_signature=driver_signature,
        current_implication=current_implication,
        canonical_spend_order_digest=digest,
    )


def _make_canonical_runtime_witness_rerun_spend_order_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessRerunSpendOrderReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessRerunSpendOrderReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-runtime-witness-rerun-spend-order"
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
        coverage_anchor_random_state=202,
        overshoot_companion_random_state=505,
        live_driver_signature="bounded-right-center-execution-contract",
        runtime_witness_path=(
            "omega_f_hat[2,2]",
            "v_f_hat[2,2]",
            "covariance(0.25, 0.15)",
        ),
        budget_priority_driver_signature="first-sine-runtime-witness-budget-priority",
        candidate_status="candidate-matches-runtime-contract",
        resulting_contract_status="runtime-witness-contract-satisfied",
        effective_fresh_reruns=5,
        rerun_gate_driver_signature="fresh-rerun-gated-by-entry-patch-readiness",
        acceptance_driver_signature="fresh-rerun-must-carry-floor-lift-evidence",
        binding_budget_label="positive-first-sine-diagonal-omega-gap",
        binding_budget_share=0.24167652126504635,
        second_budget_label="omega-only-shared-vf-increment",
        second_budget_share=0.17856759637478292,
        third_budget_label="PSD-boundary",
        third_budget_share=0.12714564275399315,
        fourth_budget_label="full-shared-vf-gap",
        fourth_budget_share=0.11972498526885547,
        supported_floor_ceiling=7.0 / 9.0,
        canonical_floor=0.85,
        canonical_floor_shortfall=0.07222222222222219,
        spend_order_steps=(
            "bounded-right-center-execution-contract",
            "prioritize-diagonal-omega-lift-before-psd-or-broad-replay-guards",
            "fresh-rerun-gated-by-entry-patch-readiness",
            "fresh-rerun-must-carry-floor-lift-evidence",
        ),
        spend_order_holds=True,
        driver_signature="runtime-witness-rerun-spend-order-disciplined",
        current_implication=(
            "spend-fresh-reruns-only-after-runtime-witness-floor-lift-discipline"
        ),
        canonical_spend_order_digest=(
            "- the live Trigger 2 repair still enters only through `bounded-right-center-execution-contract`, so fresh rerun spend must stay chained to the bounded runtime witness path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` on `near_zero_grid`",
            "- before any fresh rerun is worth spending, the first estimator effort still belongs to the positive first-sine diagonal omega lift: budget priority remains `24.2%` versus `17.9%`, `12.7%`, and `12.0%`, while the candidate guard still reports `candidate-matches-runtime-contract` / `runtime-witness-contract-satisfied`",
            "- only after that bounded runtime witness discipline holds may the remaining `5` fresh reruns be spent on binding design `DGP2/500/50`; the rerun gate still forbids `stable-slice-wide first-sine replay`, `coordinate-`2` off-diagonal fallback`, and `full-matrix widening replay`",
            "- current Trigger 2 implication: `runtime-witness-rerun-spend-order-disciplined`; even an admissible rerun is informative only if it preserves the bounded intake and delivers fresh estimator evidence that lifts the binding floor witness, because canonical `nonparametric_coverage_floor = 0.850` still stays short by `0.072` above ceiling `0.778`",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_rerun_spend_order() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessRerunSpendOrderReport
):
    return _make_canonical_runtime_witness_rerun_spend_order_report()
