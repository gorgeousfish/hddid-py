from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_cancellation_budget_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCancellationBudgetReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_cancellation_budget_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaDiagonalPriorityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_support_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaSupportContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_support_contract,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _driver_signature(
    *,
    support_contract_signature: str,
    diagonal_priority_signature: str,
    cancellation_budget_signature: str,
    diagonal_coordinate: int,
    diagonal_basis_label: str,
    diagonal_only_multiple_of_required_lift: float,
    required_lift_share_of_diagonal_only_increment: float,
    diagonal_only_slack_multiple_of_required_lift: float,
    required_increment_share_of_anchor_abs_basis_pair_mass: float,
) -> str:
    if (
        support_contract_signature == "positive-first-sine-omega-support-contract"
        and diagonal_priority_signature == "first-sine-omega-diagonal-priority"
        and cancellation_budget_signature == "bounded-first-sine-cancellation-budget"
        and diagonal_coordinate == 2
        and diagonal_basis_label == "sin(2πz)"
        and diagonal_only_multiple_of_required_lift > 4.0
        and required_lift_share_of_diagonal_only_increment < 0.25
        and diagonal_only_slack_multiple_of_required_lift > 3.0
        and required_increment_share_of_anchor_abs_basis_pair_mass < 0.05
    ):
        return "bounded-positive-first-sine-omega-diagonal-sufficiency"
    return "mixed-positive-first-sine-omega-sufficiency"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalSufficiencyContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    shared_vf_entry_label: str
    diagonal_only_increment: float
    required_diagonal_vf_entry_lift: float
    diagonal_only_multiple_of_required_lift: float
    required_lift_share_of_diagonal_only_increment: float
    diagonal_only_slack_above_required_lift: float
    diagonal_only_slack_multiple_of_required_lift: float
    required_incremental_right_center_covariance_lift: float
    required_increment_share_of_anchor_abs_basis_pair_mass: float
    driver_signature: str
    canonical_positive_first_sine_omega_diagonal_sufficiency_digest: tuple[str, ...]

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
        self.diagonal_coordinate = int(self.diagonal_coordinate)
        self.diagonal_basis_label = str(self.diagonal_basis_label).strip()
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.diagonal_only_increment = float(self.diagonal_only_increment)
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.diagonal_only_multiple_of_required_lift = float(
            self.diagonal_only_multiple_of_required_lift
        )
        self.required_lift_share_of_diagonal_only_increment = float(
            self.required_lift_share_of_diagonal_only_increment
        )
        self.diagonal_only_slack_above_required_lift = float(
            self.diagonal_only_slack_above_required_lift
        )
        self.diagonal_only_slack_multiple_of_required_lift = float(
            self.diagonal_only_slack_multiple_of_required_lift
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.required_increment_share_of_anchor_abs_basis_pair_mass = float(
            self.required_increment_share_of_anchor_abs_basis_pair_mass
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_positive_first_sine_omega_diagonal_sufficiency_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_positive_first_sine_omega_diagonal_sufficiency_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "diagonal_coordinate": self.diagonal_coordinate,
            "diagonal_basis_label": self.diagonal_basis_label,
            "shared_vf_entry_label": self.shared_vf_entry_label,
            "diagonal_only_increment": self.diagonal_only_increment,
            "required_diagonal_vf_entry_lift": self.required_diagonal_vf_entry_lift,
            "diagonal_only_multiple_of_required_lift": (
                self.diagonal_only_multiple_of_required_lift
            ),
            "required_lift_share_of_diagonal_only_increment": (
                self.required_lift_share_of_diagonal_only_increment
            ),
            "diagonal_only_slack_above_required_lift": (
                self.diagonal_only_slack_above_required_lift
            ),
            "diagonal_only_slack_multiple_of_required_lift": (
                self.diagonal_only_slack_multiple_of_required_lift
            ),
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "required_increment_share_of_anchor_abs_basis_pair_mass": (
                self.required_increment_share_of_anchor_abs_basis_pair_mass
            ),
            "driver_signature": self.driver_signature,
            "canonical_positive_first_sine_omega_diagonal_sufficiency_digest": list(
                self.canonical_positive_first_sine_omega_diagonal_sufficiency_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_sufficiency_contract_report(
    *,
    support_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaSupportContractReport
    ),
    diagonal_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaDiagonalPriorityReport
    ),
    cancellation_budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCancellationBudgetReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalSufficiencyContractReport:
    if support_contract_report.policy_digest != diagonal_priority_report.policy_digest:
        raise ValueError("omega diagonal sufficiency contract requires shared policy")
    if (
        support_contract_report.binding_design
        != diagonal_priority_report.binding_design
    ):
        raise ValueError(
            "omega diagonal sufficiency contract requires shared binding design"
        )
    if support_contract_report.window_label != diagonal_priority_report.window_label:
        raise ValueError(
            "omega diagonal sufficiency contract requires shared window label"
        )
    if (
        support_contract_report.coverage_anchor_random_state
        != diagonal_priority_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        support_contract_report.overshoot_companion_random_state
        != diagonal_priority_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if (
        support_contract_report.policy_digest
        != cancellation_budget_report.policy_digest
    ):
        raise ValueError(
            "omega diagonal sufficiency contract requires cancellation-budget policy sync"
        )
    if (
        support_contract_report.binding_design
        != cancellation_budget_report.binding_design
    ):
        raise ValueError(
            "omega diagonal sufficiency contract requires cancellation-budget design sync"
        )
    if support_contract_report.window_label != cancellation_budget_report.window_label:
        raise ValueError(
            "omega diagonal sufficiency contract requires cancellation-budget window sync"
        )
    if (
        support_contract_report.coverage_anchor_random_state
        != cancellation_budget_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "coverage-anchor seed must match the cancellation-budget report"
        )
    if (
        support_contract_report.overshoot_companion_random_state
        != cancellation_budget_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "overshoot-companion seed must match the cancellation-budget report"
        )
    if (
        support_contract_report.diagonal_coordinate
        != diagonal_priority_report.diagonal_coordinate
    ):
        raise ValueError("diagonal coordinate must match the diagonal-priority report")

    diagonal_only_increment = float(diagonal_priority_report.diagonal_only_increment)
    required_diagonal_vf_entry_lift = float(
        diagonal_priority_report.required_diagonal_vf_entry_lift
    )
    if diagonal_only_increment <= 0.0:
        raise ValueError("diagonal-only increment must stay positive")
    if required_diagonal_vf_entry_lift <= 0.0:
        raise ValueError("required diagonal lift must stay positive")
    if diagonal_only_increment <= required_diagonal_vf_entry_lift:
        raise ValueError("diagonal-only increment must exceed the bounded target")

    required_lift_share_of_diagonal_only_increment = float(
        required_diagonal_vf_entry_lift / diagonal_only_increment
    )
    diagonal_only_slack_above_required_lift = float(
        diagonal_only_increment - required_diagonal_vf_entry_lift
    )
    diagonal_only_slack_multiple_of_required_lift = float(
        diagonal_only_slack_above_required_lift / required_diagonal_vf_entry_lift
    )

    driver_signature = _driver_signature(
        support_contract_signature=support_contract_report.driver_signature,
        diagonal_priority_signature=diagonal_priority_report.driver_signature,
        cancellation_budget_signature=cancellation_budget_report.driver_signature,
        diagonal_coordinate=support_contract_report.diagonal_coordinate,
        diagonal_basis_label=support_contract_report.diagonal_basis_label,
        diagonal_only_multiple_of_required_lift=(
            diagonal_priority_report.diagonal_only_multiple_of_required_lift
        ),
        required_lift_share_of_diagonal_only_increment=(
            required_lift_share_of_diagonal_only_increment
        ),
        diagonal_only_slack_multiple_of_required_lift=(
            diagonal_only_slack_multiple_of_required_lift
        ),
        required_increment_share_of_anchor_abs_basis_pair_mass=(
            cancellation_budget_report.required_increment_share_of_anchor_abs_basis_pair_mass
        ),
    )

    canonical_digest = (
        f"- swapping only positive `omega_f_hat[{support_contract_report.diagonal_coordinate},{support_contract_report.diagonal_coordinate}]` lifts shared `{diagonal_priority_report.shared_vf_entry_label}` by `{_format_signed(diagonal_only_increment)}` against the bounded target `{_format_signed(required_diagonal_vf_entry_lift)}`, so the diagonal alone already carries `{_format_ratio(diagonal_priority_report.diagonal_only_multiple_of_required_lift)}` the required shared-entry repair and leaves `{_format_signed(diagonal_only_slack_above_required_lift)}` of unused diagonal headroom",
        f"- the live Trigger 2 repair would therefore consume only `{_format_percent(required_lift_share_of_diagonal_only_increment)}` of that diagonal-only lift; the remaining diagonal slack still equals `{_format_ratio(diagonal_only_slack_multiple_of_required_lift)}` the full bounded target before any off-diagonal coordinate-`{support_contract_report.diagonal_coordinate}` axis replay",
        f"- this stays consistent with the current cancellation budget: the same `z = 0.25 -> z = 0.15` lane still needs only `{_format_signed(cancellation_budget_report.required_incremental_right_center_covariance_lift)}` covariance lift, or `{_format_percent(cancellation_budget_report.required_increment_share_of_anchor_abs_basis_pair_mass)}` of anchor gross basis-pair mass, so the current implication remains `{driver_signature}` rather than broad omega-axis or whole-matrix replay",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalSufficiencyContractReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-positive-first-sine-omega-diagonal-sufficiency-contract",
        policy_digest=support_contract_report.policy_digest,
        binding_design=support_contract_report.binding_design,
        window_label=support_contract_report.window_label,
        coverage_anchor_random_state=support_contract_report.coverage_anchor_random_state,
        overshoot_companion_random_state=support_contract_report.overshoot_companion_random_state,
        diagonal_coordinate=support_contract_report.diagonal_coordinate,
        diagonal_basis_label=support_contract_report.diagonal_basis_label,
        shared_vf_entry_label=diagonal_priority_report.shared_vf_entry_label,
        diagonal_only_increment=diagonal_only_increment,
        required_diagonal_vf_entry_lift=required_diagonal_vf_entry_lift,
        diagonal_only_multiple_of_required_lift=(
            diagonal_priority_report.diagonal_only_multiple_of_required_lift
        ),
        required_lift_share_of_diagonal_only_increment=(
            required_lift_share_of_diagonal_only_increment
        ),
        diagonal_only_slack_above_required_lift=(
            diagonal_only_slack_above_required_lift
        ),
        diagonal_only_slack_multiple_of_required_lift=(
            diagonal_only_slack_multiple_of_required_lift
        ),
        required_incremental_right_center_covariance_lift=(
            cancellation_budget_report.required_incremental_right_center_covariance_lift
        ),
        required_increment_share_of_anchor_abs_basis_pair_mass=(
            cancellation_budget_report.required_increment_share_of_anchor_abs_basis_pair_mass
        ),
        driver_signature=driver_signature,
        canonical_positive_first_sine_omega_diagonal_sufficiency_digest=(
            canonical_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_sufficiency_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalSufficiencyContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_sufficiency_contract_report(
        support_contract_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_support_contract(),
        diagonal_priority_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_probe(),
        cancellation_budget_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_cancellation_budget_probe(),
    )
