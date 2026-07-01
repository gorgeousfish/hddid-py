from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_probe,
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
    omega_axis_signature: str,
    diagonal_coordinate: int,
    diagonal_basis_label: str,
    diagonal_share_of_coordinate_axis_increment: float,
    diagonal_vs_offdiagonal_increment_ratio: float,
    diagonal_only_multiple_of_required_lift: float,
    offdiagonal_only_multiple_of_required_lift: float,
) -> str:
    if (
        support_contract_signature == "positive-first-sine-omega-support-contract"
        and omega_axis_signature == "first-sine-omega-axis-activation"
        and diagonal_coordinate == 2
        and diagonal_basis_label == "sin(2πz)"
        and diagonal_share_of_coordinate_axis_increment > 0.7
        and diagonal_vs_offdiagonal_increment_ratio > 2.5
        and diagonal_only_multiple_of_required_lift > 4.0
        and offdiagonal_only_multiple_of_required_lift > 1.0
    ):
        return "first-sine-omega-diagonal-priority"
    return "mixed-first-sine-omega-priority"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaDiagonalPriorityReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    shared_vf_entry_label: str
    same_sign_positive_support_gap: float
    sign_flip_negative_mass_gap: float
    coordinate_axis_only_increment: float
    diagonal_only_increment: float
    offdiagonal_axis_only_increment: float
    required_diagonal_vf_entry_lift: float
    diagonal_share_of_coordinate_axis_increment: float
    offdiagonal_share_of_coordinate_axis_increment: float
    diagonal_vs_offdiagonal_increment_ratio: float
    diagonal_only_multiple_of_required_lift: float
    offdiagonal_only_multiple_of_required_lift: float
    diagonal_priority_margin: float
    diagonal_priority_margin_multiple_of_required_lift: float
    driver_signature: str
    canonical_first_sine_omega_diagonal_priority_digest: tuple[str, ...]

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
        self.same_sign_positive_support_gap = float(self.same_sign_positive_support_gap)
        self.sign_flip_negative_mass_gap = float(self.sign_flip_negative_mass_gap)
        self.coordinate_axis_only_increment = float(self.coordinate_axis_only_increment)
        self.diagonal_only_increment = float(self.diagonal_only_increment)
        self.offdiagonal_axis_only_increment = float(
            self.offdiagonal_axis_only_increment
        )
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.diagonal_share_of_coordinate_axis_increment = float(
            self.diagonal_share_of_coordinate_axis_increment
        )
        self.offdiagonal_share_of_coordinate_axis_increment = float(
            self.offdiagonal_share_of_coordinate_axis_increment
        )
        self.diagonal_vs_offdiagonal_increment_ratio = float(
            self.diagonal_vs_offdiagonal_increment_ratio
        )
        self.diagonal_only_multiple_of_required_lift = float(
            self.diagonal_only_multiple_of_required_lift
        )
        self.offdiagonal_only_multiple_of_required_lift = float(
            self.offdiagonal_only_multiple_of_required_lift
        )
        self.diagonal_priority_margin = float(self.diagonal_priority_margin)
        self.diagonal_priority_margin_multiple_of_required_lift = float(
            self.diagonal_priority_margin_multiple_of_required_lift
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_omega_diagonal_priority_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_omega_diagonal_priority_digest
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
            "same_sign_positive_support_gap": self.same_sign_positive_support_gap,
            "sign_flip_negative_mass_gap": self.sign_flip_negative_mass_gap,
            "coordinate_axis_only_increment": self.coordinate_axis_only_increment,
            "diagonal_only_increment": self.diagonal_only_increment,
            "offdiagonal_axis_only_increment": self.offdiagonal_axis_only_increment,
            "required_diagonal_vf_entry_lift": self.required_diagonal_vf_entry_lift,
            "diagonal_share_of_coordinate_axis_increment": (
                self.diagonal_share_of_coordinate_axis_increment
            ),
            "offdiagonal_share_of_coordinate_axis_increment": (
                self.offdiagonal_share_of_coordinate_axis_increment
            ),
            "diagonal_vs_offdiagonal_increment_ratio": (
                self.diagonal_vs_offdiagonal_increment_ratio
            ),
            "diagonal_only_multiple_of_required_lift": (
                self.diagonal_only_multiple_of_required_lift
            ),
            "offdiagonal_only_multiple_of_required_lift": (
                self.offdiagonal_only_multiple_of_required_lift
            ),
            "diagonal_priority_margin": self.diagonal_priority_margin,
            "diagonal_priority_margin_multiple_of_required_lift": (
                self.diagonal_priority_margin_multiple_of_required_lift
            ),
            "driver_signature": self.driver_signature,
            "canonical_first_sine_omega_diagonal_priority_digest": list(
                self.canonical_first_sine_omega_diagonal_priority_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_report(
    *,
    support_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaSupportContractReport
    ),
    omega_axis_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport
    ),
    sandwich_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaDiagonalPriorityReport:
    if support_contract_report.policy_digest != omega_axis_report.policy_digest:
        raise ValueError("omega diagonal priority probe requires shared policy")
    if support_contract_report.binding_design != omega_axis_report.binding_design:
        raise ValueError("omega diagonal priority probe requires shared binding design")
    if support_contract_report.window_label != omega_axis_report.window_label:
        raise ValueError("omega diagonal priority probe requires shared window label")
    if (
        support_contract_report.coverage_anchor_random_state
        != omega_axis_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        support_contract_report.overshoot_companion_random_state
        != omega_axis_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if support_contract_report.policy_digest != sandwich_report.policy_digest:
        raise ValueError("omega diagonal priority probe requires sandwich policy sync")
    if support_contract_report.binding_design != sandwich_report.binding_design:
        raise ValueError("omega diagonal priority probe requires sandwich design sync")
    if support_contract_report.window_label != sandwich_report.window_label:
        raise ValueError("omega diagonal priority probe requires sandwich window sync")
    if (
        support_contract_report.coverage_anchor_random_state
        != sandwich_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match the sandwich report")
    if (
        support_contract_report.overshoot_companion_random_state
        != sandwich_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match the sandwich report")
    if (
        support_contract_report.diagonal_coordinate
        != omega_axis_report.diagonal_coordinate
    ):
        raise ValueError("diagonal coordinate must match across upstream reports")

    coordinate_axis_only_increment = float(
        omega_axis_report.coordinate_axis_only_increment
    )
    diagonal_only_increment = float(omega_axis_report.diagonal_only_increment)
    offdiagonal_axis_only_increment = float(
        omega_axis_report.offdiagonal_axis_only_increment
    )
    required_diagonal_vf_entry_lift = float(
        sandwich_report.required_diagonal_vf_entry_lift
    )
    if coordinate_axis_only_increment <= 0.0:
        raise ValueError("coordinate-axis increment must stay positive")
    if required_diagonal_vf_entry_lift <= 0.0:
        raise ValueError("required diagonal vf-entry lift must stay positive")
    if diagonal_only_increment <= offdiagonal_axis_only_increment:
        raise ValueError("diagonal increment must dominate off-diagonal axis lift")

    diagonal_share_of_coordinate_axis_increment = float(
        diagonal_only_increment / coordinate_axis_only_increment
    )
    offdiagonal_share_of_coordinate_axis_increment = float(
        offdiagonal_axis_only_increment / coordinate_axis_only_increment
    )
    diagonal_vs_offdiagonal_increment_ratio = float(
        diagonal_only_increment / offdiagonal_axis_only_increment
    )
    diagonal_only_multiple_of_required_lift = float(
        diagonal_only_increment / required_diagonal_vf_entry_lift
    )
    offdiagonal_only_multiple_of_required_lift = float(
        offdiagonal_axis_only_increment / required_diagonal_vf_entry_lift
    )
    diagonal_priority_margin = float(
        diagonal_only_increment - offdiagonal_axis_only_increment
    )
    diagonal_priority_margin_multiple_of_required_lift = float(
        diagonal_priority_margin / required_diagonal_vf_entry_lift
    )

    driver_signature = _driver_signature(
        support_contract_signature=support_contract_report.driver_signature,
        omega_axis_signature=omega_axis_report.driver_signature,
        diagonal_coordinate=support_contract_report.diagonal_coordinate,
        diagonal_basis_label=support_contract_report.diagonal_basis_label,
        diagonal_share_of_coordinate_axis_increment=(
            diagonal_share_of_coordinate_axis_increment
        ),
        diagonal_vs_offdiagonal_increment_ratio=(
            diagonal_vs_offdiagonal_increment_ratio
        ),
        diagonal_only_multiple_of_required_lift=(
            diagonal_only_multiple_of_required_lift
        ),
        offdiagonal_only_multiple_of_required_lift=(
            offdiagonal_only_multiple_of_required_lift
        ),
    )

    canonical_digest = (
        f"- inside the positive coordinate-`{support_contract_report.diagonal_coordinate} = {support_contract_report.diagonal_basis_label}` omega lane, the bounded repair remains diagonal-first: swapping only `omega_f_hat[{support_contract_report.diagonal_coordinate},{support_contract_report.diagonal_coordinate}]` lifts shared `{omega_axis_report.shared_vf_entry_label}` by `{_format_signed(diagonal_only_increment)}`, while the same coordinate-`{support_contract_report.diagonal_coordinate}` off-diagonal axis adds `{_format_signed(offdiagonal_axis_only_increment)}`, so the diagonal alone supplies `{_format_percent(diagonal_share_of_coordinate_axis_increment)}` of axis lift and leads the off-diagonal remainder by `{_format_signed(diagonal_priority_margin)}`",
        f"- both atoms clear the bounded shared-entry target `{_format_signed(required_diagonal_vf_entry_lift)}`, but the diagonal still carries `{_format_ratio(diagonal_only_multiple_of_required_lift)}` that target versus `{_format_ratio(offdiagonal_only_multiple_of_required_lift)}` for the off-diagonal axis, yielding a diagonal priority ratio of `{_format_ratio(diagonal_vs_offdiagonal_increment_ratio)}` before any broader omega replay",
        f"- current Trigger 2 implication: `{driver_signature}`; source-level follow-up should first explain why anchor seed `{support_contract_report.coverage_anchor_random_state}` under-activates positive `omega_f_hat[{support_contract_report.diagonal_coordinate},{support_contract_report.diagonal_coordinate}]` mass before replaying the rest of the coordinate-`{support_contract_report.diagonal_coordinate}` off-diagonal axis or broader matrix geometry",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaDiagonalPriorityReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-omega-diagonal-priority-probe",
        policy_digest=support_contract_report.policy_digest,
        binding_design=support_contract_report.binding_design,
        window_label=support_contract_report.window_label,
        coverage_anchor_random_state=support_contract_report.coverage_anchor_random_state,
        overshoot_companion_random_state=support_contract_report.overshoot_companion_random_state,
        diagonal_coordinate=support_contract_report.diagonal_coordinate,
        diagonal_basis_label=support_contract_report.diagonal_basis_label,
        shared_vf_entry_label=omega_axis_report.shared_vf_entry_label,
        same_sign_positive_support_gap=support_contract_report.same_sign_positive_support_gap,
        sign_flip_negative_mass_gap=support_contract_report.sign_flip_negative_mass_gap,
        coordinate_axis_only_increment=coordinate_axis_only_increment,
        diagonal_only_increment=diagonal_only_increment,
        offdiagonal_axis_only_increment=offdiagonal_axis_only_increment,
        required_diagonal_vf_entry_lift=required_diagonal_vf_entry_lift,
        diagonal_share_of_coordinate_axis_increment=(
            diagonal_share_of_coordinate_axis_increment
        ),
        offdiagonal_share_of_coordinate_axis_increment=(
            offdiagonal_share_of_coordinate_axis_increment
        ),
        diagonal_vs_offdiagonal_increment_ratio=(
            diagonal_vs_offdiagonal_increment_ratio
        ),
        diagonal_only_multiple_of_required_lift=(
            diagonal_only_multiple_of_required_lift
        ),
        offdiagonal_only_multiple_of_required_lift=(
            offdiagonal_only_multiple_of_required_lift
        ),
        diagonal_priority_margin=diagonal_priority_margin,
        diagonal_priority_margin_multiple_of_required_lift=(
            diagonal_priority_margin_multiple_of_required_lift
        ),
        driver_signature=driver_signature,
        canonical_first_sine_omega_diagonal_priority_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaDiagonalPriorityReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_report(
        support_contract_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_support_contract(),
        omega_axis_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_probe(),
        sandwich_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_probe(),
    )
