from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_basis_term_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateBasisTermReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_basis_term_probe,
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


def _driver_signature(
    *,
    execution_contract_signature: str,
    basis_term_signature: str,
    diagonal_coordinate: int,
    diagonal_basis_label: str,
    diagonal_basis_family: str,
    diagonal_basis_harmonic: int,
    required_increment_share_of_entry_gap: float,
    required_increment_share_of_diagonal_support_gap: float,
    max_inert_right_shoulder_abs_basis_value: float,
) -> str:
    if (
        execution_contract_signature == "bounded-right-center-execution-contract"
        and basis_term_signature == "first-sine-harmonic-activation-lane"
        and diagonal_coordinate == 2
        and diagonal_basis_label == "sin(2πz)"
        and diagonal_basis_family == "sine"
        and diagonal_basis_harmonic == 1
        and required_increment_share_of_entry_gap < 0.1
        and required_increment_share_of_diagonal_support_gap < 0.2
        and max_inert_right_shoulder_abs_basis_value <= 1e-12
    ):
        return "bounded-first-sine-diagonal-contract"
    return "mixed-first-sine-diagonal-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    execution_contract_local_scale_shortfall_share: float
    diagonal_coordinate: int
    diagonal_basis_label: str
    diagonal_basis_family: str
    diagonal_basis_harmonic: int
    shared_diagonal_entry_label: str
    diagonal_right_shoulder_basis_value: float
    diagonal_center_basis_value: float
    diagonal_basis_sign_product: float
    diagonal_support_gap: float
    diagonal_support_gap_share: float
    current_abs_right_center_covariance: float
    required_abs_right_center_covariance: float
    required_incremental_right_center_covariance_lift: float
    required_increment_share_of_entry_gap: float
    required_increment_share_of_diagonal_support_gap: float
    diagonal_support_gap_to_required_ratio: float
    max_inert_right_shoulder_abs_basis_value: float
    driver_signature: str
    canonical_first_sine_diagonal_contract_digest: tuple[str, ...]

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
        self.execution_contract_local_scale_shortfall_share = float(
            self.execution_contract_local_scale_shortfall_share
        )
        self.diagonal_coordinate = int(self.diagonal_coordinate)
        self.diagonal_basis_label = str(self.diagonal_basis_label).strip()
        self.diagonal_basis_family = str(self.diagonal_basis_family).strip()
        self.diagonal_basis_harmonic = int(self.diagonal_basis_harmonic)
        self.shared_diagonal_entry_label = str(self.shared_diagonal_entry_label).strip()
        self.diagonal_right_shoulder_basis_value = float(
            self.diagonal_right_shoulder_basis_value
        )
        self.diagonal_center_basis_value = float(self.diagonal_center_basis_value)
        self.diagonal_basis_sign_product = float(self.diagonal_basis_sign_product)
        self.diagonal_support_gap = float(self.diagonal_support_gap)
        self.diagonal_support_gap_share = float(self.diagonal_support_gap_share)
        self.current_abs_right_center_covariance = float(
            self.current_abs_right_center_covariance
        )
        self.required_abs_right_center_covariance = float(
            self.required_abs_right_center_covariance
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.required_increment_share_of_entry_gap = float(
            self.required_increment_share_of_entry_gap
        )
        self.required_increment_share_of_diagonal_support_gap = float(
            self.required_increment_share_of_diagonal_support_gap
        )
        self.diagonal_support_gap_to_required_ratio = float(
            self.diagonal_support_gap_to_required_ratio
        )
        self.max_inert_right_shoulder_abs_basis_value = float(
            self.max_inert_right_shoulder_abs_basis_value
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_diagonal_contract_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_diagonal_contract_digest
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
            "execution_contract_local_scale_shortfall_share": (
                self.execution_contract_local_scale_shortfall_share
            ),
            "diagonal_coordinate": self.diagonal_coordinate,
            "diagonal_basis_label": self.diagonal_basis_label,
            "diagonal_basis_family": self.diagonal_basis_family,
            "diagonal_basis_harmonic": self.diagonal_basis_harmonic,
            "shared_diagonal_entry_label": self.shared_diagonal_entry_label,
            "diagonal_right_shoulder_basis_value": (
                self.diagonal_right_shoulder_basis_value
            ),
            "diagonal_center_basis_value": self.diagonal_center_basis_value,
            "diagonal_basis_sign_product": self.diagonal_basis_sign_product,
            "diagonal_support_gap": self.diagonal_support_gap,
            "diagonal_support_gap_share": self.diagonal_support_gap_share,
            "current_abs_right_center_covariance": (
                self.current_abs_right_center_covariance
            ),
            "required_abs_right_center_covariance": (
                self.required_abs_right_center_covariance
            ),
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "required_increment_share_of_entry_gap": (
                self.required_increment_share_of_entry_gap
            ),
            "required_increment_share_of_diagonal_support_gap": (
                self.required_increment_share_of_diagonal_support_gap
            ),
            "diagonal_support_gap_to_required_ratio": (
                self.diagonal_support_gap_to_required_ratio
            ),
            "max_inert_right_shoulder_abs_basis_value": (
                self.max_inert_right_shoulder_abs_basis_value
            ),
            "driver_signature": self.driver_signature,
            "canonical_first_sine_diagonal_contract_digest": list(
                self.canonical_first_sine_diagonal_contract_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract_report(
    *,
    execution_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
    basis_term_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateBasisTermReport
    ),
) -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalContractReport
):
    if execution_contract_report.policy_digest != basis_term_report.policy_digest:
        raise ValueError("first-sine diagonal contract requires a shared policy digest")
    if execution_contract_report.binding_design != basis_term_report.binding_design:
        raise ValueError(
            "first-sine diagonal contract requires a shared binding design"
        )
    if execution_contract_report.window_label != basis_term_report.window_label:
        raise ValueError("first-sine diagonal contract requires a shared window label")
    if (
        execution_contract_report.coverage_anchor_random_state
        != basis_term_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "first-sine diagonal contract requires the same coverage-anchor seed"
        )
    if (
        execution_contract_report.overshoot_companion_random_state
        != basis_term_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "first-sine diagonal contract requires the same overshoot-companion seed"
        )
    if not isclose(
        execution_contract_report.center_grid_value,
        basis_term_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "first-sine diagonal contract requires the same center grid value"
        )
    if not isclose(
        execution_contract_report.failing_right_shoulder_grid_value,
        basis_term_report.failing_right_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "first-sine diagonal contract requires the same right-shoulder grid value"
        )
    if not isclose(
        execution_contract_report.required_incremental_right_center_covariance_lift,
        basis_term_report.required_incremental_right_center_covariance_lift,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "first-sine diagonal contract requires the same bounded covariance lift"
        )

    driver_signature = _driver_signature(
        execution_contract_signature=execution_contract_report.driver_signature,
        basis_term_signature=basis_term_report.driver_signature,
        diagonal_coordinate=basis_term_report.top_coordinate,
        diagonal_basis_label=basis_term_report.top_coordinate_basis_label,
        diagonal_basis_family=basis_term_report.top_coordinate_family,
        diagonal_basis_harmonic=basis_term_report.top_coordinate_harmonic,
        required_increment_share_of_entry_gap=(
            execution_contract_report.required_increment_share_of_entry_gap
        ),
        required_increment_share_of_diagonal_support_gap=(
            basis_term_report.required_increment_share_of_top_coordinate_support_gap
        ),
        max_inert_right_shoulder_abs_basis_value=(
            basis_term_report.max_inert_right_shoulder_abs_basis_value
        ),
    )
    shared_diagonal_entry_label = f"v_f_hat[{basis_term_report.top_coordinate},{basis_term_report.top_coordinate}]"

    canonical_digest = (
        f"- binding design `DGP2/500/50` on `{execution_contract_report.window_label}`: the last local-scale miss still stays `{_format_percent(execution_contract_report.coverage_anchor_local_scale_access_shortfall_share)}` at failing right shoulder `z = {_format_grid_value(execution_contract_report.failing_right_shoulder_grid_value)}`, and the live lane still only needs `|covariance({_format_grid_value(execution_contract_report.failing_right_shoulder_grid_value)}, {_format_grid_value(execution_contract_report.center_grid_value)})|` to rise from `{_format_float(execution_contract_report.current_abs_right_center_covariance)}` to `{_format_float(execution_contract_report.required_abs_right_center_covariance)}` (increment `{_format_signed(execution_contract_report.required_incremental_right_center_covariance_lift)}`) while the left-side reserve remains intact",
        f"- under the paper-trigonometric degree-8 basis, that bounded entry is already pinned to diagonal coordinate `{basis_term_report.top_coordinate} = {basis_term_report.top_coordinate_basis_label}` on shared `{shared_diagonal_entry_label}`: the basis term evaluates to `{_format_signed(basis_term_report.top_coordinate_right_shoulder_basis_value)}` at `z = {_format_grid_value(basis_term_report.failing_right_shoulder_grid_value)}` and `{_format_signed(basis_term_report.top_coordinate_center_basis_value)}` at `z = {_format_grid_value(basis_term_report.center_grid_value)}`, so the shared sign product stays `{_format_signed(basis_term_report.top_coordinate_basis_sign_product)}`, while inert same-sign coordinates remain numerically zero on the right shoulder (`max |psi_j({_format_grid_value(basis_term_report.failing_right_shoulder_grid_value)})| = {_format_float(basis_term_report.max_inert_right_shoulder_abs_basis_value)}`)",
        f"- coordinate `{basis_term_report.top_coordinate}` therefore carries the implementation atom: its support gap is `{_format_signed(basis_term_report.top_coordinate_support_gap)}` (`{_format_percent(basis_term_report.top_coordinate_support_gap_share)}` of the same-sign gap), which is `{_format_ratio(basis_term_report.top_coordinate_gap_to_required_ratio)}` the required bounded lift, and the current repair only needs `{_format_percent(basis_term_report.required_increment_share_of_top_coordinate_support_gap)}` of that diagonal support gap while consuming `{_format_percent(execution_contract_report.required_increment_share_of_entry_gap)}` of the companion right-center entry gap",
        f"- current Trigger 2 implication: `{driver_signature}`; next implementation should inspect why anchor seed `{execution_contract_report.coverage_anchor_random_state}` under-activates diagonal `{basis_term_report.top_coordinate_basis_label}` support on shared `{shared_diagonal_entry_label}`, rather than rebalancing the whole active same-sign block or replaying whole-row / whole-column geometry",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "first-sine-diagonal-contract"
        ),
        policy_digest=execution_contract_report.policy_digest,
        binding_design=execution_contract_report.binding_design,
        window_label=execution_contract_report.window_label,
        coverage_anchor_random_state=(
            execution_contract_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            execution_contract_report.overshoot_companion_random_state
        ),
        center_grid_value=execution_contract_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            execution_contract_report.failing_right_shoulder_grid_value
        ),
        execution_contract_local_scale_shortfall_share=(
            execution_contract_report.coverage_anchor_local_scale_access_shortfall_share
        ),
        diagonal_coordinate=basis_term_report.top_coordinate,
        diagonal_basis_label=basis_term_report.top_coordinate_basis_label,
        diagonal_basis_family=basis_term_report.top_coordinate_family,
        diagonal_basis_harmonic=basis_term_report.top_coordinate_harmonic,
        shared_diagonal_entry_label=shared_diagonal_entry_label,
        diagonal_right_shoulder_basis_value=(
            basis_term_report.top_coordinate_right_shoulder_basis_value
        ),
        diagonal_center_basis_value=(
            basis_term_report.top_coordinate_center_basis_value
        ),
        diagonal_basis_sign_product=(
            basis_term_report.top_coordinate_basis_sign_product
        ),
        diagonal_support_gap=basis_term_report.top_coordinate_support_gap,
        diagonal_support_gap_share=(basis_term_report.top_coordinate_support_gap_share),
        current_abs_right_center_covariance=(
            execution_contract_report.current_abs_right_center_covariance
        ),
        required_abs_right_center_covariance=(
            execution_contract_report.required_abs_right_center_covariance
        ),
        required_incremental_right_center_covariance_lift=(
            execution_contract_report.required_incremental_right_center_covariance_lift
        ),
        required_increment_share_of_entry_gap=(
            execution_contract_report.required_increment_share_of_entry_gap
        ),
        required_increment_share_of_diagonal_support_gap=(
            basis_term_report.required_increment_share_of_top_coordinate_support_gap
        ),
        diagonal_support_gap_to_required_ratio=(
            basis_term_report.top_coordinate_gap_to_required_ratio
        ),
        max_inert_right_shoulder_abs_basis_value=(
            basis_term_report.max_inert_right_shoulder_abs_basis_value
        ),
        driver_signature=driver_signature,
        canonical_first_sine_diagonal_contract_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract_report(
        execution_contract_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract()
        ),
        basis_term_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_basis_term_probe()
        ),
    )
