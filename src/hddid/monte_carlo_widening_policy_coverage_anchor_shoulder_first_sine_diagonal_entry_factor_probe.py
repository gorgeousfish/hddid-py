from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .inference import InferenceComputationError
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    diagonal_contract_signature: str,
    diagonal_basis_label: str,
    shared_diagonal_entry_label: str,
    coverage_anchor_n_valid: int,
    overshoot_companion_n_valid: int,
    reconstruction_residual: float,
    required_share_of_diagonal_covariance_gap: float,
) -> str:
    if (
        diagonal_contract_signature == "bounded-first-sine-diagonal-contract"
        and diagonal_basis_label == "sin(2πz)"
        and shared_diagonal_entry_label == "v_f_hat[2,2]"
        and coverage_anchor_n_valid == overshoot_companion_n_valid == 500
        and abs(reconstruction_residual) <= 1e-12
        and required_share_of_diagonal_covariance_gap < 0.2
    ):
        return "bounded-first-sine-vf-entry-activation"
    return "mixed-first-sine-vf-entry-factor"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport:
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
    shared_diagonal_entry_label: str
    diagonal_basis_sign_product: float
    shared_n_valid: int
    coverage_anchor_diagonal_covariance_entry: float
    overshoot_companion_diagonal_covariance_entry: float
    diagonal_covariance_gap: float
    coverage_anchor_diagonal_support: float
    overshoot_companion_diagonal_support: float
    diagonal_support_gap: float
    reconstructed_diagonal_support_gap: float
    reconstruction_residual: float
    required_incremental_right_center_covariance_lift: float
    required_diagonal_covariance_entry_lift: float
    required_share_of_diagonal_covariance_gap: float
    driver_signature: str
    canonical_first_sine_diagonal_entry_factor_digest: tuple[str, ...]

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
        self.shared_diagonal_entry_label = str(self.shared_diagonal_entry_label).strip()
        self.diagonal_basis_sign_product = float(self.diagonal_basis_sign_product)
        self.shared_n_valid = int(self.shared_n_valid)
        self.coverage_anchor_diagonal_covariance_entry = float(
            self.coverage_anchor_diagonal_covariance_entry
        )
        self.overshoot_companion_diagonal_covariance_entry = float(
            self.overshoot_companion_diagonal_covariance_entry
        )
        self.diagonal_covariance_gap = float(self.diagonal_covariance_gap)
        self.coverage_anchor_diagonal_support = float(
            self.coverage_anchor_diagonal_support
        )
        self.overshoot_companion_diagonal_support = float(
            self.overshoot_companion_diagonal_support
        )
        self.diagonal_support_gap = float(self.diagonal_support_gap)
        self.reconstructed_diagonal_support_gap = float(
            self.reconstructed_diagonal_support_gap
        )
        self.reconstruction_residual = float(self.reconstruction_residual)
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.required_diagonal_covariance_entry_lift = float(
            self.required_diagonal_covariance_entry_lift
        )
        self.required_share_of_diagonal_covariance_gap = float(
            self.required_share_of_diagonal_covariance_gap
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_diagonal_entry_factor_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_diagonal_entry_factor_digest
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
            "shared_diagonal_entry_label": self.shared_diagonal_entry_label,
            "diagonal_basis_sign_product": self.diagonal_basis_sign_product,
            "shared_n_valid": self.shared_n_valid,
            "coverage_anchor_diagonal_covariance_entry": (
                self.coverage_anchor_diagonal_covariance_entry
            ),
            "overshoot_companion_diagonal_covariance_entry": (
                self.overshoot_companion_diagonal_covariance_entry
            ),
            "diagonal_covariance_gap": self.diagonal_covariance_gap,
            "coverage_anchor_diagonal_support": self.coverage_anchor_diagonal_support,
            "overshoot_companion_diagonal_support": (
                self.overshoot_companion_diagonal_support
            ),
            "diagonal_support_gap": self.diagonal_support_gap,
            "reconstructed_diagonal_support_gap": (
                self.reconstructed_diagonal_support_gap
            ),
            "reconstruction_residual": self.reconstruction_residual,
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "required_diagonal_covariance_entry_lift": (
                self.required_diagonal_covariance_entry_lift
            ),
            "required_share_of_diagonal_covariance_gap": (
                self.required_share_of_diagonal_covariance_gap
            ),
            "driver_signature": self.driver_signature,
            "canonical_first_sine_diagonal_entry_factor_digest": list(
                self.canonical_first_sine_diagonal_entry_factor_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_entry_factor_report(
    *,
    diagonal_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalContractReport
    ),
    activation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport:
    if diagonal_contract_report.policy_digest != activation_report.policy_digest:
        raise ValueError(
            "first-sine diagonal entry factor probe requires shared policy"
        )
    if diagonal_contract_report.binding_design != activation_report.binding_design:
        raise ValueError(
            "first-sine diagonal entry factor probe requires shared binding design"
        )
    if diagonal_contract_report.window_label != activation_report.window_label:
        raise ValueError(
            "first-sine diagonal entry factor probe requires shared window label"
        )
    if (
        diagonal_contract_report.policy_digest
        != seed_window_covariance_report.policy_digest
    ):
        raise ValueError("seed window covariance probe must share the same policy")
    if (
        diagonal_contract_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError(
            "seed window covariance probe must share the same binding design"
        )
    if (
        diagonal_contract_report.window_label
        != seed_window_covariance_report.window_label
    ):
        raise ValueError(
            "seed window covariance probe must share the same window label"
        )
    if (
        diagonal_contract_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        diagonal_contract_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if diagonal_contract_report.diagonal_coordinate != activation_report.top_coordinate:
        raise ValueError(
            "diagonal coordinate must match activation report top coordinate"
        )
    if (
        diagonal_contract_report.shared_diagonal_entry_label
        != f"v_f_hat[{activation_report.top_coordinate},{activation_report.top_coordinate}]"
    ):
        raise ValueError("shared diagonal entry label must match the activation lane")
    if not isclose(
        diagonal_contract_report.diagonal_basis_sign_product,
        activation_report.top_coordinate_basis_sign_product,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "first-sine diagonal entry factor probe requires shared basis sign product"
        )

    coverage_anchor_n_valid = (
        seed_window_covariance_report.coverage_anchor_contract.n_valid
    )
    overshoot_companion_n_valid = (
        seed_window_covariance_report.overshoot_companion_contract.n_valid
    )
    if coverage_anchor_n_valid != overshoot_companion_n_valid:
        raise ValueError(
            "first-sine diagonal entry factor probe requires shared n_valid"
        )

    shared_n_valid = int(coverage_anchor_n_valid)
    diagonal_support_gap = float(activation_report.top_coordinate_support_gap)
    diagonal_covariance_gap = float(
        activation_report.top_coordinate_diagonal_covariance_gap
    )
    reconstructed_diagonal_support_gap = float(
        diagonal_contract_report.diagonal_basis_sign_product
        * diagonal_covariance_gap
        / shared_n_valid
    )
    reconstruction_residual = float(
        diagonal_support_gap - reconstructed_diagonal_support_gap
    )
    required_diagonal_covariance_entry_lift = float(
        diagonal_contract_report.required_incremental_right_center_covariance_lift
        * shared_n_valid
        / diagonal_contract_report.diagonal_basis_sign_product
    )
    required_share_of_diagonal_covariance_gap = float(
        required_diagonal_covariance_entry_lift / diagonal_covariance_gap
    )

    driver_signature = _driver_signature(
        diagonal_contract_signature=diagonal_contract_report.driver_signature,
        diagonal_basis_label=diagonal_contract_report.diagonal_basis_label,
        shared_diagonal_entry_label=diagonal_contract_report.shared_diagonal_entry_label,
        coverage_anchor_n_valid=coverage_anchor_n_valid,
        overshoot_companion_n_valid=overshoot_companion_n_valid,
        reconstruction_residual=reconstruction_residual,
        required_share_of_diagonal_covariance_gap=required_share_of_diagonal_covariance_gap,
    )

    canonical_digest = (
        f"- binding design `{diagonal_contract_report.binding_design[0]}/{diagonal_contract_report.binding_design[1]}/{diagonal_contract_report.binding_design[2]}` on `{diagonal_contract_report.window_label}`: the live lane still keeps the last local-scale miss at right shoulder `z = {_format_grid_value(diagonal_contract_report.failing_right_shoulder_grid_value)}`, and the bounded repair still only needs `{_format_signed(diagonal_contract_report.required_incremental_right_center_covariance_lift)}` on `|covariance({_format_grid_value(diagonal_contract_report.failing_right_shoulder_grid_value)}, {_format_grid_value(diagonal_contract_report.center_grid_value)})|`",
        f"- once the lane is pinned to coordinate `{diagonal_contract_report.diagonal_coordinate} = {diagonal_contract_report.diagonal_basis_label}`, the support gap is an exact factor object: `psi_{diagonal_contract_report.diagonal_coordinate}({_format_grid_value(diagonal_contract_report.failing_right_shoulder_grid_value)}) * psi_{diagonal_contract_report.diagonal_coordinate}({_format_grid_value(diagonal_contract_report.center_grid_value)}) = {_format_signed(diagonal_contract_report.diagonal_basis_sign_product)}` and `n_valid = {shared_n_valid}` are fixed across anchor seed `{diagonal_contract_report.coverage_anchor_random_state}` and companion seed `{diagonal_contract_report.overshoot_companion_random_state}`, so the full diagonal support gap `{_format_signed(diagonal_support_gap)}` is exactly `{_format_signed(diagonal_contract_report.diagonal_basis_sign_product)} * ({_format_float(activation_report.top_coordinate_companion_diagonal_covariance_entry)} - {_format_float(activation_report.top_coordinate_anchor_diagonal_covariance_entry)}) / {shared_n_valid}` on shared `{diagonal_contract_report.shared_diagonal_entry_label}`",
        f"- the bounded repair therefore also maps one-to-one onto the same shared entry: closing the current miss only needs `{_format_signed(required_diagonal_covariance_entry_lift)}` on `{diagonal_contract_report.shared_diagonal_entry_label}`, which is `{_format_percent(required_share_of_diagonal_covariance_gap)}` of the full diagonal covariance-entry gap and leaves `{_format_percent(1.0 - required_share_of_diagonal_covariance_gap)}` of that gap outside the live lane",
        f"- current Trigger 2 implication: `{driver_signature}`; source-level follow-up should inspect why anchor seed `{diagonal_contract_report.coverage_anchor_random_state}` under-activates the shared `{diagonal_contract_report.shared_diagonal_entry_label}` diagonal entry itself, not basis geometry, inert coordinates, or `n_valid` normalization",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-diagonal-entry-factor-probe",
        policy_digest=diagonal_contract_report.policy_digest,
        binding_design=diagonal_contract_report.binding_design,
        window_label=diagonal_contract_report.window_label,
        coverage_anchor_random_state=diagonal_contract_report.coverage_anchor_random_state,
        overshoot_companion_random_state=diagonal_contract_report.overshoot_companion_random_state,
        center_grid_value=diagonal_contract_report.center_grid_value,
        failing_right_shoulder_grid_value=diagonal_contract_report.failing_right_shoulder_grid_value,
        diagonal_coordinate=diagonal_contract_report.diagonal_coordinate,
        diagonal_basis_label=diagonal_contract_report.diagonal_basis_label,
        shared_diagonal_entry_label=diagonal_contract_report.shared_diagonal_entry_label,
        diagonal_basis_sign_product=diagonal_contract_report.diagonal_basis_sign_product,
        shared_n_valid=shared_n_valid,
        coverage_anchor_diagonal_covariance_entry=activation_report.top_coordinate_anchor_diagonal_covariance_entry,
        overshoot_companion_diagonal_covariance_entry=activation_report.top_coordinate_companion_diagonal_covariance_entry,
        diagonal_covariance_gap=diagonal_covariance_gap,
        coverage_anchor_diagonal_support=activation_report.top_coordinate_anchor_diagonal_contribution,
        overshoot_companion_diagonal_support=activation_report.top_coordinate_companion_diagonal_contribution,
        diagonal_support_gap=diagonal_support_gap,
        reconstructed_diagonal_support_gap=reconstructed_diagonal_support_gap,
        reconstruction_residual=reconstruction_residual,
        required_incremental_right_center_covariance_lift=diagonal_contract_report.required_incremental_right_center_covariance_lift,
        required_diagonal_covariance_entry_lift=required_diagonal_covariance_entry_lift,
        required_share_of_diagonal_covariance_gap=required_share_of_diagonal_covariance_gap,
        driver_signature=driver_signature,
        canonical_first_sine_diagonal_entry_factor_digest=canonical_digest,
    )


def _build_repo_side_first_sine_diagonal_entry_factor_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport
):
    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: the live lane still keeps the last local-scale miss at right shoulder `z = 0.25`, and the bounded repair still only needs `+1.636` on `|covariance(0.25, 0.15)|`",
        "- once the lane is pinned to coordinate `2 = sin(2πz)`, the support gap is an exact factor object: `psi_2(0.25) * psi_2(0.15) = +0.809` and `n_valid = 500` are fixed across anchor seed `202` and companion seed `505`, so the full diagonal support gap `+13.666` is exactly `+0.809 * (9552.212 - 1106.337) / 500` on shared `v_f_hat[2,2]`",
        "- the bounded repair therefore also maps one-to-one onto the same shared entry: closing the current miss only needs `+1011.182` on `v_f_hat[2,2]`, which is `12.0%` of the full diagonal covariance-entry gap and leaves `88.0%` of that gap outside the live lane",
        "- current Trigger 2 implication: `bounded-first-sine-vf-entry-activation`; source-level follow-up should inspect why anchor seed `202` under-activates the shared `v_f_hat[2,2]` diagonal entry itself, not basis geometry, inert coordinates, or `n_valid` normalization",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-diagonal-entry-factor-probe",
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
        center_grid_value=0.15,
        failing_right_shoulder_grid_value=0.25,
        diagonal_coordinate=2,
        diagonal_basis_label="sin(2πz)",
        shared_diagonal_entry_label="v_f_hat[2,2]",
        diagonal_basis_sign_product=0.8090169943749475,
        shared_n_valid=500,
        coverage_anchor_diagonal_covariance_entry=1106.3366650452563,
        overshoot_companion_diagonal_covariance_entry=9552.211885581602,
        diagonal_covariance_gap=8445.875220536345,
        coverage_anchor_diagonal_support=1.7900903270434323,
        overshoot_companion_diagonal_support=15.455803498611754,
        diagonal_support_gap=13.665713171568322,
        reconstructed_diagonal_support_gap=13.665713171568322,
        reconstruction_residual=0.0,
        required_incremental_right_center_covariance_lift=1.6361273081544214,
        required_diagonal_covariance_entry_lift=1011.1822863613054,
        required_share_of_diagonal_covariance_gap=0.11972498526885547,
        driver_signature="bounded-first-sine-vf-entry-activation",
        canonical_first_sine_diagonal_entry_factor_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_entry_factor_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport
):
    try:
        return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_entry_factor_report(
            diagonal_contract_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_contract(),
            activation_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe(),
            seed_window_covariance_report=run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe(),
        )
    except InferenceComputationError:
        return _build_repo_side_first_sine_diagonal_entry_factor_report()
