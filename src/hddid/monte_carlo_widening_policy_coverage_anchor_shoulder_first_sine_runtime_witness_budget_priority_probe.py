from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_support_snapshot import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSupportSnapshotReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_support_snapshot,
)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessBudgetPriorityProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    runtime_witness_path: tuple[str, ...]
    binding_budget_label: str
    binding_budget_share: float
    second_budget_label: str
    second_budget_share: float
    third_budget_label: str
    third_budget_share: float
    fourth_budget_label: str
    fourth_budget_share: float
    second_budget_gap_to_binding: float
    third_budget_gap_to_binding: float
    fourth_budget_gap_to_binding: float
    binding_over_second_budget_multiple: float
    binding_over_third_budget_multiple: float
    binding_over_fourth_budget_multiple: float
    diagonal_gap_share_left_unused_after_binding: float
    driver_signature: str
    current_implication: str
    canonical_budget_priority_digest: tuple[str, ...]

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
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.binding_budget_label = str(self.binding_budget_label).strip()
        self.binding_budget_share = float(self.binding_budget_share)
        self.second_budget_label = str(self.second_budget_label).strip()
        self.second_budget_share = float(self.second_budget_share)
        self.third_budget_label = str(self.third_budget_label).strip()
        self.third_budget_share = float(self.third_budget_share)
        self.fourth_budget_label = str(self.fourth_budget_label).strip()
        self.fourth_budget_share = float(self.fourth_budget_share)
        self.second_budget_gap_to_binding = float(self.second_budget_gap_to_binding)
        self.third_budget_gap_to_binding = float(self.third_budget_gap_to_binding)
        self.fourth_budget_gap_to_binding = float(self.fourth_budget_gap_to_binding)
        self.binding_over_second_budget_multiple = float(
            self.binding_over_second_budget_multiple
        )
        self.binding_over_third_budget_multiple = float(
            self.binding_over_third_budget_multiple
        )
        self.binding_over_fourth_budget_multiple = float(
            self.binding_over_fourth_budget_multiple
        )
        self.diagonal_gap_share_left_unused_after_binding = float(
            self.diagonal_gap_share_left_unused_after_binding
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.current_implication = str(self.current_implication).strip()
        self.canonical_budget_priority_digest = tuple(
            str(line).rstrip() for line in self.canonical_budget_priority_digest
        )


def _driver_signature(
    *,
    budget_order: tuple[str, ...],
    runtime_witness_path: tuple[str, ...],
    diagonal_gap_share_left_unused_after_binding: float,
) -> str:
    if (
        budget_order
        == (
            "positive-first-sine-diagonal-omega-gap",
            "omega-only-shared-vf-increment",
            "PSD-boundary",
            "full-shared-vf-gap",
        )
        and runtime_witness_path
        == ("omega_f_hat[2,2]", "v_f_hat[2,2]", "covariance(0.25, 0.15)")
        and 0.75 < diagonal_gap_share_left_unused_after_binding < 0.77
    ):
        return "first-sine-runtime-witness-budget-priority"
    return "mixed-runtime-witness-budget-priority"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_budget_priority_report(
    *,
    runtime_witness_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
        | None
    ) = None,
    support_snapshot_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSupportSnapshotReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessBudgetPriorityProbeReport:
    resolved_contract = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
        if runtime_witness_contract_report is None
        else runtime_witness_contract_report
    )
    resolved_snapshot = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_support_snapshot()
        if support_snapshot_report is None
        else support_snapshot_report
    )

    if resolved_contract.policy_digest != resolved_snapshot.policy_digest:
        raise ValueError("budget priority probe requires shared policy digest")
    if resolved_contract.binding_design != resolved_snapshot.binding_design:
        raise ValueError("budget priority probe requires shared binding design")
    if resolved_contract.window_label != resolved_snapshot.window_label:
        raise ValueError("budget priority probe requires shared window label")
    if (
        resolved_contract.coverage_anchor_random_state
        != resolved_snapshot.coverage_anchor_random_state
    ):
        raise ValueError("budget priority probe requires shared coverage-anchor seed")
    if (
        resolved_contract.overshoot_companion_random_state
        != resolved_snapshot.overshoot_companion_random_state
    ):
        raise ValueError(
            "budget priority probe requires shared overshoot-companion seed"
        )
    if (
        resolved_contract.required_patch_share_of_diagonal_omega_gap
        != resolved_snapshot.required_patch_share_of_omega_diagonal_gap
    ):
        raise ValueError("budget priority probe requires shared diagonal-gap share")

    budgets = (
        (
            "positive-first-sine-diagonal-omega-gap",
            resolved_contract.required_patch_share_of_diagonal_omega_gap,
        ),
        (
            "omega-only-shared-vf-increment",
            resolved_contract.required_patch_share_of_omega_only_shared_vf_increment,
        ),
        ("PSD-boundary", resolved_contract.required_patch_share_of_psd_boundary),
        (
            "full-shared-vf-gap",
            resolved_contract.required_patch_share_of_full_shared_vf_gap,
        ),
    )
    ordered_budgets = tuple(sorted(budgets, key=lambda item: item[1], reverse=True))

    binding_label, binding_share = ordered_budgets[0]
    second_label, second_share = ordered_budgets[1]
    third_label, third_share = ordered_budgets[2]
    fourth_label, fourth_share = ordered_budgets[3]

    second_gap = binding_share - second_share
    third_gap = binding_share - third_share
    fourth_gap = binding_share - fourth_share
    binding_over_second = binding_share / second_share
    binding_over_third = binding_share / third_share
    binding_over_fourth = binding_share / fourth_share

    driver_signature = _driver_signature(
        budget_order=tuple(label for label, _ in ordered_budgets),
        runtime_witness_path=resolved_contract.runtime_witness_path,
        diagonal_gap_share_left_unused_after_binding=(
            resolved_snapshot.residual_companion_gap_share_after_patch
        ),
    )
    current_implication = (
        "prioritize-diagonal-omega-lift-before-psd-or-broad-replay-guards"
    )
    digest = (
        f"- the bounded runtime witness still binds first on the positive first-sine diagonal omega gap: `{_format_percent(binding_share)}` of that budget is required, versus only `{_format_percent(second_share)}` on the omega-only shared `v_f_hat[2,2]` increment, `{_format_percent(third_share)}` on the PSD boundary, and `{_format_percent(fourth_share)}` on the full shared-gap replay budget",
        f"- that ordering is not marginal: the diagonal budget remains `{binding_over_second:.3f}x` the omega-only shared-entry share, `{binding_over_third:.3f}x` the PSD-boundary share, and `{binding_over_fourth:.3f}x` the broad full-gap share, so the first implementation risk is still realizing the diagonal lift itself rather than guarding PSD or whole-row/column replay",
        f"- even after satisfying the binding diagonal budget, `{_format_percent(resolved_snapshot.residual_companion_gap_share_after_patch)}` of the diagonal omega gap still remains unused, which keeps the runtime witness explicitly bounded instead of promoting the side lane into dense covariance replay",
        "- current implication: `prioritize-diagonal-omega-lift-before-psd-or-broad-replay-guards`; main execution should spend its first estimator effort on the positive `sin(2πz)` diagonal path and treat PSD / broad replay checks as follow-on guards, not the primary repair target",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessBudgetPriorityProbeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-runtime-witness-budget-priority-probe"
        ),
        policy_digest=resolved_contract.policy_digest,
        binding_design=resolved_contract.binding_design,
        window_label=resolved_contract.window_label,
        coverage_anchor_random_state=resolved_contract.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            resolved_contract.overshoot_companion_random_state
        ),
        runtime_witness_path=resolved_contract.runtime_witness_path,
        binding_budget_label=binding_label,
        binding_budget_share=binding_share,
        second_budget_label=second_label,
        second_budget_share=second_share,
        third_budget_label=third_label,
        third_budget_share=third_share,
        fourth_budget_label=fourth_label,
        fourth_budget_share=fourth_share,
        second_budget_gap_to_binding=second_gap,
        third_budget_gap_to_binding=third_gap,
        fourth_budget_gap_to_binding=fourth_gap,
        binding_over_second_budget_multiple=binding_over_second,
        binding_over_third_budget_multiple=binding_over_third,
        binding_over_fourth_budget_multiple=binding_over_fourth,
        diagonal_gap_share_left_unused_after_binding=(
            resolved_snapshot.residual_companion_gap_share_after_patch
        ),
        driver_signature=driver_signature,
        current_implication=current_implication,
        canonical_budget_priority_digest=digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_budget_priority_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessBudgetPriorityProbeReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_budget_priority_report()
