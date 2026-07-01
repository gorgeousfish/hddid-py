from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _share(part: float, total: float, *, label: str) -> float:
    total_value = float(total)
    if total_value <= 0.0:
        raise ValueError(f"{label} total must be positive")
    return float(float(part) / total_value)


def _driver_signature(
    *,
    diagonal_contract_signature: str,
    diagonal_basis_label: str,
    required_increment_share_of_diagonal_support_gap: float,
    residual_diagonal_support_gap_share: float,
    residual_diagonal_support_gap_to_required_ratio: float,
) -> str:
    if (
        diagonal_contract_signature == "bounded-first-sine-diagonal-contract"
        and diagonal_basis_label == "sin(2πz)"
        and required_increment_share_of_diagonal_support_gap < 0.2
        and residual_diagonal_support_gap_share > 0.8
        and residual_diagonal_support_gap_to_required_ratio > 5.0
    ):
        return "first-sine-reserve-dominates-bounded-repair"
    return "mixed-diagonal-reserve-lane"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalReserveReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    diagonal_coordinate: int
    diagonal_basis_label: str
    diagonal_support_gap: float
    required_incremental_right_center_covariance_lift: float
    required_increment_share_of_diagonal_support_gap: float
    residual_diagonal_support_gap_after_bounded_repair: float
    residual_diagonal_support_gap_share: float
    residual_diagonal_support_gap_to_required_ratio: float
    required_diagonal_covariance_lift: float
    diagonal_covariance_gap: float
    residual_diagonal_covariance_gap_after_bounded_repair: float
    residual_diagonal_covariance_gap_share: float
    residual_diagonal_covariance_gap_to_required_ratio: float
    driver_signature: str
    canonical_first_sine_diagonal_reserve_digest: tuple[str, ...]

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
        self.diagonal_coordinate = int(self.diagonal_coordinate)
        self.diagonal_basis_label = str(self.diagonal_basis_label).strip()
        self.diagonal_support_gap = float(self.diagonal_support_gap)
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.required_increment_share_of_diagonal_support_gap = float(
            self.required_increment_share_of_diagonal_support_gap
        )
        self.residual_diagonal_support_gap_after_bounded_repair = float(
            self.residual_diagonal_support_gap_after_bounded_repair
        )
        self.residual_diagonal_support_gap_share = float(
            self.residual_diagonal_support_gap_share
        )
        self.residual_diagonal_support_gap_to_required_ratio = float(
            self.residual_diagonal_support_gap_to_required_ratio
        )
        self.required_diagonal_covariance_lift = float(
            self.required_diagonal_covariance_lift
        )
        self.diagonal_covariance_gap = float(self.diagonal_covariance_gap)
        self.residual_diagonal_covariance_gap_after_bounded_repair = float(
            self.residual_diagonal_covariance_gap_after_bounded_repair
        )
        self.residual_diagonal_covariance_gap_share = float(
            self.residual_diagonal_covariance_gap_share
        )
        self.residual_diagonal_covariance_gap_to_required_ratio = float(
            self.residual_diagonal_covariance_gap_to_required_ratio
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_diagonal_reserve_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_diagonal_reserve_digest
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
            "diagonal_coordinate": self.diagonal_coordinate,
            "diagonal_basis_label": self.diagonal_basis_label,
            "diagonal_support_gap": self.diagonal_support_gap,
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "required_increment_share_of_diagonal_support_gap": (
                self.required_increment_share_of_diagonal_support_gap
            ),
            "residual_diagonal_support_gap_after_bounded_repair": (
                self.residual_diagonal_support_gap_after_bounded_repair
            ),
            "residual_diagonal_support_gap_share": (
                self.residual_diagonal_support_gap_share
            ),
            "residual_diagonal_support_gap_to_required_ratio": (
                self.residual_diagonal_support_gap_to_required_ratio
            ),
            "required_diagonal_covariance_lift": self.required_diagonal_covariance_lift,
            "diagonal_covariance_gap": self.diagonal_covariance_gap,
            "residual_diagonal_covariance_gap_after_bounded_repair": (
                self.residual_diagonal_covariance_gap_after_bounded_repair
            ),
            "residual_diagonal_covariance_gap_share": (
                self.residual_diagonal_covariance_gap_share
            ),
            "residual_diagonal_covariance_gap_to_required_ratio": (
                self.residual_diagonal_covariance_gap_to_required_ratio
            ),
            "driver_signature": self.driver_signature,
            "canonical_first_sine_diagonal_reserve_digest": list(
                self.canonical_first_sine_diagonal_reserve_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_reserve_report(
    *,
    diagonal_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalContractReport
    ),
    activation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalReserveReport:
    if diagonal_contract_report.policy_digest != activation_report.policy_digest:
        raise ValueError(
            "first-sine diagonal reserve probe requires a shared policy digest"
        )
    if diagonal_contract_report.binding_design != activation_report.binding_design:
        raise ValueError(
            "first-sine diagonal reserve probe requires a shared binding design"
        )
    if diagonal_contract_report.window_label != activation_report.window_label:
        raise ValueError(
            "first-sine diagonal reserve probe requires a shared window label"
        )
    if (
        diagonal_contract_report.coverage_anchor_random_state
        != activation_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "first-sine diagonal reserve probe requires the same coverage-anchor seed"
        )
    if (
        diagonal_contract_report.overshoot_companion_random_state
        != activation_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "first-sine diagonal reserve probe requires the same overshoot-companion seed"
        )
    if diagonal_contract_report.diagonal_coordinate != activation_report.top_coordinate:
        raise ValueError(
            "first-sine diagonal reserve probe requires the same diagonal coordinate"
        )

    residual_diagonal_support_gap_share = _share(
        activation_report.residual_top_coordinate_support_gap_after_bounded_repair,
        diagonal_contract_report.diagonal_support_gap,
        label="residual diagonal support gap share",
    )
    residual_diagonal_support_gap_to_required_ratio = (
        activation_report.residual_top_coordinate_support_gap_after_bounded_repair
        / diagonal_contract_report.required_incremental_right_center_covariance_lift
    )
    residual_diagonal_covariance_gap_share = _share(
        activation_report.residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair,
        activation_report.top_coordinate_diagonal_covariance_gap,
        label="residual diagonal covariance gap share",
    )
    residual_diagonal_covariance_gap_to_required_ratio = (
        activation_report.residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair
        / activation_report.required_top_coordinate_diagonal_covariance_lift
    )

    driver_signature = _driver_signature(
        diagonal_contract_signature=diagonal_contract_report.driver_signature,
        diagonal_basis_label=diagonal_contract_report.diagonal_basis_label,
        required_increment_share_of_diagonal_support_gap=(
            diagonal_contract_report.required_increment_share_of_diagonal_support_gap
        ),
        residual_diagonal_support_gap_share=residual_diagonal_support_gap_share,
        residual_diagonal_support_gap_to_required_ratio=(
            residual_diagonal_support_gap_to_required_ratio
        ),
    )

    canonical_digest = (
        f"- binding design `DGP2/500/50` on `{diagonal_contract_report.window_label}`: the bounded right-center repair still only needs `{_format_signed(diagonal_contract_report.required_incremental_right_center_covariance_lift)}` on failing shoulder `z = {_format_grid_value(diagonal_contract_report.failing_right_shoulder_grid_value)}` relative to center `z = {_format_grid_value(diagonal_contract_report.center_grid_value)}`, and the live lane stays pinned to diagonal coordinate `{diagonal_contract_report.diagonal_coordinate} = {diagonal_contract_report.diagonal_basis_label}` on shared `{diagonal_contract_report.shared_diagonal_entry_label}`",
        f"- that repair spends only `{_format_percent(diagonal_contract_report.required_increment_share_of_diagonal_support_gap)}` of the coordinate-`{diagonal_contract_report.diagonal_coordinate}` support gap `{_format_signed(diagonal_contract_report.diagonal_support_gap)}`, leaving `{_format_signed(activation_report.residual_top_coordinate_support_gap_after_bounded_repair)}` (`{_format_percent(residual_diagonal_support_gap_share)}`) still unused after the bounded fix, so the remaining diagonal reserve alone still equals `{_format_ratio(residual_diagonal_support_gap_to_required_ratio)}` the original bounded lift",
        f"- the same reserve survives on the shared diagonal covariance entry: repairing the lane needs only `{_format_signed(activation_report.required_top_coordinate_diagonal_covariance_lift)}` out of the full `{diagonal_contract_report.shared_diagonal_entry_label}` gap `{_format_signed(activation_report.top_coordinate_diagonal_covariance_gap)}`, leaving `{_format_signed(activation_report.residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair)}` (`{_format_percent(residual_diagonal_covariance_gap_share)}`) still unused, again `{_format_ratio(residual_diagonal_covariance_gap_to_required_ratio)}` the original bounded diagonal lift",
        f"- current Trigger 2 implication: `{driver_signature}`; next implementation should restore the missing access to coordinate `{diagonal_contract_report.diagonal_coordinate} = {diagonal_contract_report.diagonal_basis_label}` before replaying other same-sign coordinates, off-diagonal geometry, or whole-row / whole-column structure",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalReserveReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "first-sine-diagonal-reserve-probe"
        ),
        policy_digest=diagonal_contract_report.policy_digest,
        binding_design=diagonal_contract_report.binding_design,
        window_label=diagonal_contract_report.window_label,
        coverage_anchor_random_state=(
            diagonal_contract_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            diagonal_contract_report.overshoot_companion_random_state
        ),
        center_grid_value=diagonal_contract_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            diagonal_contract_report.failing_right_shoulder_grid_value
        ),
        diagonal_coordinate=diagonal_contract_report.diagonal_coordinate,
        diagonal_basis_label=diagonal_contract_report.diagonal_basis_label,
        diagonal_support_gap=diagonal_contract_report.diagonal_support_gap,
        required_incremental_right_center_covariance_lift=(
            diagonal_contract_report.required_incremental_right_center_covariance_lift
        ),
        required_increment_share_of_diagonal_support_gap=(
            diagonal_contract_report.required_increment_share_of_diagonal_support_gap
        ),
        residual_diagonal_support_gap_after_bounded_repair=(
            activation_report.residual_top_coordinate_support_gap_after_bounded_repair
        ),
        residual_diagonal_support_gap_share=residual_diagonal_support_gap_share,
        residual_diagonal_support_gap_to_required_ratio=(
            residual_diagonal_support_gap_to_required_ratio
        ),
        required_diagonal_covariance_lift=(
            activation_report.required_top_coordinate_diagonal_covariance_lift
        ),
        diagonal_covariance_gap=(
            activation_report.top_coordinate_diagonal_covariance_gap
        ),
        residual_diagonal_covariance_gap_after_bounded_repair=(
            activation_report.residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair
        ),
        residual_diagonal_covariance_gap_share=(residual_diagonal_covariance_gap_share),
        residual_diagonal_covariance_gap_to_required_ratio=(
            residual_diagonal_covariance_gap_to_required_ratio
        ),
        driver_signature=driver_signature,
        canonical_first_sine_diagonal_reserve_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_reserve_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalReserveReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_reserve_report(
        diagonal_contract_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract()
        ),
        activation_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe()
        ),
    )
