from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if numerator_value <= 0.0:
        raise ValueError(f"{label} numerator must be positive")
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(numerator_value / denominator_value)


def _driver_signature(
    *,
    first_sine_diagonal_signature: str,
    basis_pair_cancellation_signature: str,
    anchor_gross_to_net_ratio: float,
    required_increment_share_of_anchor_abs_basis_pair_mass: float,
    required_increment_share_of_anchor_positive_basis_pair_mass: float,
    required_increment_share_of_anchor_negative_basis_pair_mass: float,
    required_increment_share_of_diagonal_support_gap: float,
) -> str:
    if (
        first_sine_diagonal_signature == "bounded-first-sine-diagonal-contract"
        and basis_pair_cancellation_signature == "basis-pair-cancellation-bottleneck"
        and anchor_gross_to_net_ratio > 100.0
        and required_increment_share_of_anchor_abs_basis_pair_mass < 0.05
        and required_increment_share_of_anchor_positive_basis_pair_mass < 0.1
        and required_increment_share_of_anchor_negative_basis_pair_mass < 0.1
        and required_increment_share_of_diagonal_support_gap < 0.2
    ):
        return "bounded-first-sine-cancellation-budget"
    return "mixed-first-sine-cancellation-budget"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCancellationBudgetReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    current_abs_right_center_covariance: float
    required_abs_right_center_covariance: float
    required_incremental_right_center_covariance_lift: float
    first_sine_diagonal_basis_label: str
    first_sine_diagonal_support_gap: float
    required_increment_share_of_diagonal_support_gap: float
    diagonal_support_gap_to_required_ratio: float
    anchor_abs_basis_pair_mass: float
    anchor_positive_basis_pair_mass: float
    anchor_negative_basis_pair_mass: float
    anchor_gross_to_net_ratio: float
    anchor_net_share_of_gross_mass: float
    required_increment_share_of_anchor_abs_basis_pair_mass: float
    required_increment_share_of_anchor_positive_basis_pair_mass: float
    required_increment_share_of_anchor_negative_basis_pair_mass: float
    first_sine_diagonal_share_of_anchor_positive_mass: float
    driver_signature: str
    canonical_first_sine_cancellation_budget_digest: tuple[str, ...]

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
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.current_abs_right_center_covariance = float(
            self.current_abs_right_center_covariance
        )
        self.required_abs_right_center_covariance = float(
            self.required_abs_right_center_covariance
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.first_sine_diagonal_basis_label = str(
            self.first_sine_diagonal_basis_label
        ).strip()
        self.first_sine_diagonal_support_gap = float(
            self.first_sine_diagonal_support_gap
        )
        self.required_increment_share_of_diagonal_support_gap = float(
            self.required_increment_share_of_diagonal_support_gap
        )
        self.diagonal_support_gap_to_required_ratio = float(
            self.diagonal_support_gap_to_required_ratio
        )
        self.anchor_abs_basis_pair_mass = float(self.anchor_abs_basis_pair_mass)
        self.anchor_positive_basis_pair_mass = float(
            self.anchor_positive_basis_pair_mass
        )
        self.anchor_negative_basis_pair_mass = float(
            self.anchor_negative_basis_pair_mass
        )
        self.anchor_gross_to_net_ratio = float(self.anchor_gross_to_net_ratio)
        self.anchor_net_share_of_gross_mass = float(self.anchor_net_share_of_gross_mass)
        self.required_increment_share_of_anchor_abs_basis_pair_mass = float(
            self.required_increment_share_of_anchor_abs_basis_pair_mass
        )
        self.required_increment_share_of_anchor_positive_basis_pair_mass = float(
            self.required_increment_share_of_anchor_positive_basis_pair_mass
        )
        self.required_increment_share_of_anchor_negative_basis_pair_mass = float(
            self.required_increment_share_of_anchor_negative_basis_pair_mass
        )
        self.first_sine_diagonal_share_of_anchor_positive_mass = float(
            self.first_sine_diagonal_share_of_anchor_positive_mass
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_cancellation_budget_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_cancellation_budget_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "current_abs_right_center_covariance": (
                self.current_abs_right_center_covariance
            ),
            "required_abs_right_center_covariance": (
                self.required_abs_right_center_covariance
            ),
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "first_sine_diagonal_basis_label": self.first_sine_diagonal_basis_label,
            "first_sine_diagonal_support_gap": self.first_sine_diagonal_support_gap,
            "required_increment_share_of_diagonal_support_gap": (
                self.required_increment_share_of_diagonal_support_gap
            ),
            "diagonal_support_gap_to_required_ratio": (
                self.diagonal_support_gap_to_required_ratio
            ),
            "anchor_abs_basis_pair_mass": self.anchor_abs_basis_pair_mass,
            "anchor_positive_basis_pair_mass": self.anchor_positive_basis_pair_mass,
            "anchor_negative_basis_pair_mass": self.anchor_negative_basis_pair_mass,
            "anchor_gross_to_net_ratio": self.anchor_gross_to_net_ratio,
            "anchor_net_share_of_gross_mass": self.anchor_net_share_of_gross_mass,
            "required_increment_share_of_anchor_abs_basis_pair_mass": (
                self.required_increment_share_of_anchor_abs_basis_pair_mass
            ),
            "required_increment_share_of_anchor_positive_basis_pair_mass": (
                self.required_increment_share_of_anchor_positive_basis_pair_mass
            ),
            "required_increment_share_of_anchor_negative_basis_pair_mass": (
                self.required_increment_share_of_anchor_negative_basis_pair_mass
            ),
            "first_sine_diagonal_share_of_anchor_positive_mass": (
                self.first_sine_diagonal_share_of_anchor_positive_mass
            ),
            "driver_signature": self.driver_signature,
            "canonical_first_sine_cancellation_budget_digest": list(
                self.canonical_first_sine_cancellation_budget_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_cancellation_budget_report(
    *,
    first_sine_diagonal_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalContractReport
    ),
    basis_pair_cancellation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCancellationBudgetReport:
    if (
        first_sine_diagonal_contract_report.policy_digest
        != basis_pair_cancellation_report.policy_digest
    ):
        raise ValueError(
            "first-sine cancellation budget probe requires a shared policy digest"
        )
    if (
        first_sine_diagonal_contract_report.binding_design
        != basis_pair_cancellation_report.binding_design
    ):
        raise ValueError(
            "first-sine cancellation budget probe requires a shared binding design"
        )
    if (
        first_sine_diagonal_contract_report.window_label
        != basis_pair_cancellation_report.window_label
    ):
        raise ValueError(
            "first-sine cancellation budget probe requires a shared window label"
        )
    if (
        first_sine_diagonal_contract_report.coverage_anchor_random_state
        != basis_pair_cancellation_report.coverage_anchor_random_state
        or first_sine_diagonal_contract_report.overshoot_companion_random_state
        != basis_pair_cancellation_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "first-sine cancellation budget probe requires the same anchor/companion seeds"
        )
    if not isclose(
        first_sine_diagonal_contract_report.center_grid_value,
        basis_pair_cancellation_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "first-sine cancellation budget probe requires the same center grid"
        )
    if not isclose(
        first_sine_diagonal_contract_report.failing_right_shoulder_grid_value,
        basis_pair_cancellation_report.failing_right_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "first-sine cancellation budget probe requires the same failing shoulder grid"
        )
    if not isclose(
        first_sine_diagonal_contract_report.required_incremental_right_center_covariance_lift,
        basis_pair_cancellation_report.required_incremental_right_center_covariance_lift,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "first-sine cancellation budget probe requires the same bounded lift"
        )

    required_increment = float(
        first_sine_diagonal_contract_report.required_incremental_right_center_covariance_lift
    )
    anchor_negative_basis_pair_mass = abs(
        float(basis_pair_cancellation_report.anchor_negative_basis_pair_mass)
    )
    required_increment_share_of_anchor_positive_basis_pair_mass = _positive_ratio(
        required_increment,
        basis_pair_cancellation_report.anchor_positive_basis_pair_mass,
        label="required_increment_share_of_anchor_positive_basis_pair_mass",
    )
    required_increment_share_of_anchor_negative_basis_pair_mass = _positive_ratio(
        required_increment,
        anchor_negative_basis_pair_mass,
        label="required_increment_share_of_anchor_negative_basis_pair_mass",
    )
    first_sine_diagonal_share_of_anchor_positive_mass = _positive_ratio(
        first_sine_diagonal_contract_report.diagonal_support_gap,
        basis_pair_cancellation_report.anchor_positive_basis_pair_mass,
        label="first_sine_diagonal_share_of_anchor_positive_mass",
    )

    driver_signature = _driver_signature(
        first_sine_diagonal_signature=first_sine_diagonal_contract_report.driver_signature,
        basis_pair_cancellation_signature=basis_pair_cancellation_report.driver_signature,
        anchor_gross_to_net_ratio=basis_pair_cancellation_report.anchor_gross_to_net_ratio,
        required_increment_share_of_anchor_abs_basis_pair_mass=(
            basis_pair_cancellation_report.required_increment_share_of_anchor_abs_basis_pair_mass
        ),
        required_increment_share_of_anchor_positive_basis_pair_mass=(
            required_increment_share_of_anchor_positive_basis_pair_mass
        ),
        required_increment_share_of_anchor_negative_basis_pair_mass=(
            required_increment_share_of_anchor_negative_basis_pair_mass
        ),
        required_increment_share_of_diagonal_support_gap=(
            first_sine_diagonal_contract_report.required_increment_share_of_diagonal_support_gap
        ),
    )

    canonical_digest = (
        "- binding design `"
        f"{first_sine_diagonal_contract_report.binding_design[0]}/"
        f"{first_sine_diagonal_contract_report.binding_design[1]}/"
        f"{first_sine_diagonal_contract_report.binding_design[2]}` on "
        f"`{first_sine_diagonal_contract_report.window_label}`: the live lane still only needs "
        f"`|covariance({_format_grid_value(first_sine_diagonal_contract_report.failing_right_shoulder_grid_value)}, "
        f"{_format_grid_value(first_sine_diagonal_contract_report.center_grid_value)})|` to rise from "
        f"`{_format_float(first_sine_diagonal_contract_report.current_abs_right_center_covariance)}` "
        f"to `{_format_float(first_sine_diagonal_contract_report.required_abs_right_center_covariance)}` "
        f"(increment `{_format_signed(required_increment)}`) while the last local-scale miss remains "
        f"`{_format_percent(first_sine_diagonal_contract_report.execution_contract_local_scale_shortfall_share)}` "
        f"at failing right shoulder `z = {_format_grid_value(first_sine_diagonal_contract_report.failing_right_shoulder_grid_value)}`",
        "- anchor seed `"
        f"{first_sine_diagonal_contract_report.coverage_anchor_random_state}` still exhibits severe "
        "basis-pair cancellation at that entry: gross absolute right-center mass is "
        f"`{_format_float(basis_pair_cancellation_report.anchor_abs_basis_pair_mass)}`, "
        "but net signed mass is only "
        f"`{_format_float(first_sine_diagonal_contract_report.current_abs_right_center_covariance)}`, "
        "so the gross-to-net ratio stays at "
        f"`{_format_ratio(basis_pair_cancellation_report.anchor_gross_to_net_ratio)}` and the net share of gross mass stays at just "
        f"`{_format_percent(basis_pair_cancellation_report.anchor_net_share_of_gross_mass)}`",
        "- the bounded repair budget is nevertheless tiny relative to that cancellation geometry: "
        f"`{_format_signed(required_increment)}` consumes only "
        f"`{_format_percent(basis_pair_cancellation_report.required_increment_share_of_anchor_abs_basis_pair_mass)}` "
        "of anchor gross mass and only "
        f"`{_format_percent(required_increment_share_of_anchor_positive_basis_pair_mass)}` / "
        f"`{_format_percent(required_increment_share_of_anchor_negative_basis_pair_mass)}` "
        "of anchor positive / negative mass, so current Trigger 2 follow-up does not need whole-matrix cancellation cleanup",
        "- the paper-trigonometric first-sine diagonal remains sufficient despite cancellation: shared "
        f"`{first_sine_diagonal_contract_report.diagonal_basis_label}` support gap is "
        f"`{_format_signed(first_sine_diagonal_contract_report.diagonal_support_gap)}`, or "
        f"`{_format_percent(first_sine_diagonal_share_of_anchor_positive_mass)}` of anchor positive mass, "
        f"which is `{_format_float(first_sine_diagonal_contract_report.diagonal_support_gap_to_required_ratio)}x` "
        "the required lift; the live repair still uses only "
        f"`{_format_percent(first_sine_diagonal_contract_report.required_increment_share_of_diagonal_support_gap)}` "
        "of that diagonal support gap",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; implementation should preserve and reactivate bounded first-sine diagonal support "
        "on `z = 0.25 -> 0.15`, rather than trying to neutralize the full basis-pair cancellation field",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCancellationBudgetReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-cancellation-budget-probe"
        ),
        policy_digest=first_sine_diagonal_contract_report.policy_digest,
        binding_design=first_sine_diagonal_contract_report.binding_design,
        window_label=first_sine_diagonal_contract_report.window_label,
        coverage_anchor_random_state=(
            first_sine_diagonal_contract_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            first_sine_diagonal_contract_report.overshoot_companion_random_state
        ),
        center_grid_value=first_sine_diagonal_contract_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            first_sine_diagonal_contract_report.failing_right_shoulder_grid_value
        ),
        current_abs_right_center_covariance=(
            first_sine_diagonal_contract_report.current_abs_right_center_covariance
        ),
        required_abs_right_center_covariance=(
            first_sine_diagonal_contract_report.required_abs_right_center_covariance
        ),
        required_incremental_right_center_covariance_lift=required_increment,
        first_sine_diagonal_basis_label=(
            first_sine_diagonal_contract_report.diagonal_basis_label
        ),
        first_sine_diagonal_support_gap=(
            first_sine_diagonal_contract_report.diagonal_support_gap
        ),
        required_increment_share_of_diagonal_support_gap=(
            first_sine_diagonal_contract_report.required_increment_share_of_diagonal_support_gap
        ),
        diagonal_support_gap_to_required_ratio=(
            first_sine_diagonal_contract_report.diagonal_support_gap_to_required_ratio
        ),
        anchor_abs_basis_pair_mass=basis_pair_cancellation_report.anchor_abs_basis_pair_mass,
        anchor_positive_basis_pair_mass=(
            basis_pair_cancellation_report.anchor_positive_basis_pair_mass
        ),
        anchor_negative_basis_pair_mass=anchor_negative_basis_pair_mass,
        anchor_gross_to_net_ratio=basis_pair_cancellation_report.anchor_gross_to_net_ratio,
        anchor_net_share_of_gross_mass=(
            basis_pair_cancellation_report.anchor_net_share_of_gross_mass
        ),
        required_increment_share_of_anchor_abs_basis_pair_mass=(
            basis_pair_cancellation_report.required_increment_share_of_anchor_abs_basis_pair_mass
        ),
        required_increment_share_of_anchor_positive_basis_pair_mass=(
            required_increment_share_of_anchor_positive_basis_pair_mass
        ),
        required_increment_share_of_anchor_negative_basis_pair_mass=(
            required_increment_share_of_anchor_negative_basis_pair_mass
        ),
        first_sine_diagonal_share_of_anchor_positive_mass=(
            first_sine_diagonal_share_of_anchor_positive_mass
        ),
        driver_signature=driver_signature,
        canonical_first_sine_cancellation_budget_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_cancellation_budget_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCancellationBudgetReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_cancellation_budget_report(
        first_sine_diagonal_contract_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract()
        ),
        basis_pair_cancellation_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe()
        ),
    )
