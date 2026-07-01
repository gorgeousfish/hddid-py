from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_diagonal_support_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignDiagonalSupportReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_diagonal_support_probe,
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
    support_signature: str,
    omega_axis_signature: str,
    diagonal_coordinate: int,
    diagonal_basis_label: str,
    same_sign_positive_gap_share_of_diagonal_net_gap: float,
    sign_flip_negative_gap_share_of_diagonal_net_gap: float,
    coordinate_axis_share_of_omega_only_increment: float,
    coordinate_axis_multiple_of_required_lift: float,
) -> str:
    if (
        support_signature == "same-sign-diagonal-support-deficit"
        and omega_axis_signature == "first-sine-omega-axis-activation"
        and diagonal_coordinate == 2
        and diagonal_basis_label == "sin(2πz)"
        and same_sign_positive_gap_share_of_diagonal_net_gap > 1.0
        and sign_flip_negative_gap_share_of_diagonal_net_gap < 0.0
        and coordinate_axis_share_of_omega_only_increment > 0.95
        and coordinate_axis_multiple_of_required_lift > 5.0
    ):
        return "positive-first-sine-omega-support-contract"
    return "mixed-first-sine-support-order"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaSupportContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    same_sign_positive_support_gap: float
    same_sign_positive_gap_share_of_diagonal_net_gap: float
    sign_flip_negative_mass_gap: float
    sign_flip_negative_gap_share_of_diagonal_net_gap: float
    coordinate_axis_only_increment: float
    diagonal_only_increment: float
    offdiagonal_axis_only_increment: float
    coordinate_axis_share_of_omega_only_increment: float
    diagonal_share_of_omega_only_increment: float
    offdiagonal_axis_share_of_omega_only_increment: float
    coordinate_axis_multiple_of_required_lift: float
    driver_signature: str
    canonical_positive_first_sine_omega_support_digest: tuple[str, ...]

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
        self.same_sign_positive_support_gap = float(self.same_sign_positive_support_gap)
        self.same_sign_positive_gap_share_of_diagonal_net_gap = float(
            self.same_sign_positive_gap_share_of_diagonal_net_gap
        )
        self.sign_flip_negative_mass_gap = float(self.sign_flip_negative_mass_gap)
        self.sign_flip_negative_gap_share_of_diagonal_net_gap = float(
            self.sign_flip_negative_gap_share_of_diagonal_net_gap
        )
        self.coordinate_axis_only_increment = float(self.coordinate_axis_only_increment)
        self.diagonal_only_increment = float(self.diagonal_only_increment)
        self.offdiagonal_axis_only_increment = float(
            self.offdiagonal_axis_only_increment
        )
        self.coordinate_axis_share_of_omega_only_increment = float(
            self.coordinate_axis_share_of_omega_only_increment
        )
        self.diagonal_share_of_omega_only_increment = float(
            self.diagonal_share_of_omega_only_increment
        )
        self.offdiagonal_axis_share_of_omega_only_increment = float(
            self.offdiagonal_axis_share_of_omega_only_increment
        )
        self.coordinate_axis_multiple_of_required_lift = float(
            self.coordinate_axis_multiple_of_required_lift
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_positive_first_sine_omega_support_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_positive_first_sine_omega_support_digest
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
            "same_sign_positive_support_gap": self.same_sign_positive_support_gap,
            "same_sign_positive_gap_share_of_diagonal_net_gap": (
                self.same_sign_positive_gap_share_of_diagonal_net_gap
            ),
            "sign_flip_negative_mass_gap": self.sign_flip_negative_mass_gap,
            "sign_flip_negative_gap_share_of_diagonal_net_gap": (
                self.sign_flip_negative_gap_share_of_diagonal_net_gap
            ),
            "coordinate_axis_only_increment": self.coordinate_axis_only_increment,
            "diagonal_only_increment": self.diagonal_only_increment,
            "offdiagonal_axis_only_increment": self.offdiagonal_axis_only_increment,
            "coordinate_axis_share_of_omega_only_increment": (
                self.coordinate_axis_share_of_omega_only_increment
            ),
            "diagonal_share_of_omega_only_increment": (
                self.diagonal_share_of_omega_only_increment
            ),
            "offdiagonal_axis_share_of_omega_only_increment": (
                self.offdiagonal_axis_share_of_omega_only_increment
            ),
            "coordinate_axis_multiple_of_required_lift": (
                self.coordinate_axis_multiple_of_required_lift
            ),
            "driver_signature": self.driver_signature,
            "canonical_positive_first_sine_omega_support_digest": list(
                self.canonical_positive_first_sine_omega_support_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_support_contract_report(
    *,
    same_sign_support_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignDiagonalSupportReport
    ),
    omega_axis_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaSupportContractReport:
    if same_sign_support_report.policy_digest != omega_axis_report.policy_digest:
        raise ValueError(
            "positive first-sine omega support contract requires shared policy"
        )
    if same_sign_support_report.binding_design != omega_axis_report.binding_design:
        raise ValueError(
            "positive first-sine omega support contract requires shared binding design"
        )
    if same_sign_support_report.window_label != omega_axis_report.window_label:
        raise ValueError(
            "positive first-sine omega support contract requires shared window label"
        )
    if (
        same_sign_support_report.coverage_anchor_random_state
        != omega_axis_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        same_sign_support_report.overshoot_companion_random_state
        != omega_axis_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if (
        same_sign_support_report.top_same_sign_support_gap_coordinate
        != omega_axis_report.diagonal_coordinate
    ):
        raise ValueError(
            "top same-sign coordinate must match omega-axis diagonal coordinate"
        )

    driver_signature = _driver_signature(
        support_signature=same_sign_support_report.driver_signature,
        omega_axis_signature=omega_axis_report.driver_signature,
        diagonal_coordinate=omega_axis_report.diagonal_coordinate,
        diagonal_basis_label=omega_axis_report.diagonal_basis_label,
        same_sign_positive_gap_share_of_diagonal_net_gap=(
            same_sign_support_report.same_sign_positive_gap_share_of_diagonal_net_gap
        ),
        sign_flip_negative_gap_share_of_diagonal_net_gap=(
            same_sign_support_report.sign_flip_negative_gap_share_of_diagonal_net_gap
        ),
        coordinate_axis_share_of_omega_only_increment=(
            omega_axis_report.coordinate_axis_share_of_omega_only_increment
        ),
        coordinate_axis_multiple_of_required_lift=(
            omega_axis_report.coordinate_axis_multiple_of_required_lift
        ),
    )

    canonical_digest = (
        f"- the diagonal lane remains support-led rather than suppression-led: same-sign positive support rises by `{_format_signed(same_sign_support_report.same_sign_positive_support_gap)}`, which explains `{_format_percent(same_sign_support_report.same_sign_positive_gap_share_of_diagonal_net_gap)}` of the full diagonal-net improvement, while sign-flip negative diagonal mass drifts only `{_format_signed(same_sign_support_report.sign_flip_negative_mass_gap)}` (`{_format_percent(same_sign_support_report.sign_flip_negative_gap_share_of_diagonal_net_gap)}` of the same gap)",
        f"- that positive repair lane is already concentrated on coordinate `{omega_axis_report.diagonal_coordinate} = {omega_axis_report.diagonal_basis_label}`: swapping only the companion coordinate-`{omega_axis_report.diagonal_coordinate}` `omega_f_hat` row/column lifts shared `v_f_hat[2,2]` by `{_format_signed(omega_axis_report.coordinate_axis_only_increment)}` (`{_format_percent(omega_axis_report.coordinate_axis_share_of_omega_only_increment)}` of the omega-only lift and `{_format_ratio(omega_axis_report.coordinate_axis_multiple_of_required_lift)}` the bounded repair target), with diagonal-only lift `{_format_signed(omega_axis_report.diagonal_only_increment)}` and off-diagonal axis lift `{_format_signed(omega_axis_report.offdiagonal_axis_only_increment)}`",
        f"- current Trigger 2 implication: `{driver_signature}`; source-level follow-up should first explain why anchor seed `{same_sign_support_report.coverage_anchor_random_state}` fails to activate positive coordinate-`{omega_axis_report.diagonal_coordinate}` omega support before trying to heal the lane by shrinking sign-flip diagonal opposition or replaying broad matrix geometry",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaSupportContractReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-positive-first-sine-omega-support-contract",
        policy_digest=same_sign_support_report.policy_digest,
        binding_design=same_sign_support_report.binding_design,
        window_label=same_sign_support_report.window_label,
        coverage_anchor_random_state=same_sign_support_report.coverage_anchor_random_state,
        overshoot_companion_random_state=same_sign_support_report.overshoot_companion_random_state,
        diagonal_coordinate=omega_axis_report.diagonal_coordinate,
        diagonal_basis_label=omega_axis_report.diagonal_basis_label,
        same_sign_positive_support_gap=same_sign_support_report.same_sign_positive_support_gap,
        same_sign_positive_gap_share_of_diagonal_net_gap=(
            same_sign_support_report.same_sign_positive_gap_share_of_diagonal_net_gap
        ),
        sign_flip_negative_mass_gap=same_sign_support_report.sign_flip_negative_mass_gap,
        sign_flip_negative_gap_share_of_diagonal_net_gap=(
            same_sign_support_report.sign_flip_negative_gap_share_of_diagonal_net_gap
        ),
        coordinate_axis_only_increment=omega_axis_report.coordinate_axis_only_increment,
        diagonal_only_increment=omega_axis_report.diagonal_only_increment,
        offdiagonal_axis_only_increment=omega_axis_report.offdiagonal_axis_only_increment,
        coordinate_axis_share_of_omega_only_increment=(
            omega_axis_report.coordinate_axis_share_of_omega_only_increment
        ),
        diagonal_share_of_omega_only_increment=(
            omega_axis_report.diagonal_share_of_omega_only_increment
        ),
        offdiagonal_axis_share_of_omega_only_increment=(
            omega_axis_report.offdiagonal_axis_share_of_omega_only_increment
        ),
        coordinate_axis_multiple_of_required_lift=(
            omega_axis_report.coordinate_axis_multiple_of_required_lift
        ),
        driver_signature=driver_signature,
        canonical_positive_first_sine_omega_support_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_support_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaSupportContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_support_contract_report(
        same_sign_support_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_diagonal_support_probe(),
        omega_axis_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_probe(),
    )
