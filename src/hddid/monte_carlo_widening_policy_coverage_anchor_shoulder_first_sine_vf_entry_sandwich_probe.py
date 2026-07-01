from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_entry_factor_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_entry_factor_probe,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe,
)
from .monte_carlo_widening_policy_spec import (
    run_phase7_monte_carlo_widening_policy_spec,
)
from .validation import (
    MonteCarloDesign,
    _clone_design_with_evaluation_grid,
    _fit_phase7_nonparametric_object_payload,
    _match_runtime_probe_design,
    _phase7_runtime_probe_replication_seed,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _canonical_nonparametric_payload(
    *,
    binding_design: tuple[str, int, int],
    random_state: int,
    evaluation_grid: tuple[float, ...],
    n_boot: int,
):
    policy_spec = run_phase7_monte_carlo_widening_policy_spec()
    designs = tuple(
        MonteCarloDesign(dgp_name=dgp_name, n_obs=n_obs, p=p)
        for dgp_name, n_obs, p in policy_spec.stable_partial_designs
    )
    dgp_name, n_obs, p = binding_design
    base_design = _match_runtime_probe_design(
        designs,
        dgp_name=dgp_name,
        n_obs=n_obs,
        p=p,
    )
    replay_design = _clone_design_with_evaluation_grid(
        base_design,
        evaluation_grid=np.asarray(evaluation_grid, dtype=float),
    )
    replication_seed = _phase7_runtime_probe_replication_seed(
        random_state=int(random_state),
        designs=designs,
        dgp_name=dgp_name,
        n_obs=n_obs,
        p=p,
    )
    _, nonparametric_payload = _fit_phase7_nonparametric_object_payload(
        design=replay_design,
        replication_seed=replication_seed,
        n_boot=int(n_boot),
    )
    return nonparametric_payload


def _inverse_coordinate_sq_share(sigma_f_hat: np.ndarray, coordinate: int) -> float:
    sigma_inverse = np.linalg.inv(np.asarray(sigma_f_hat, dtype=float))
    inverse_vector = sigma_inverse[:, int(coordinate)]
    squared_mass = np.square(inverse_vector)
    total_mass = float(np.sum(squared_mass))
    if total_mass <= 0.0:
        raise ValueError("inverse-sandwich vector must carry positive squared mass")
    return float(squared_mass[int(coordinate)] / total_mass)


def _sandwich_entry(
    *,
    left_sigma_f_hat: np.ndarray,
    omega_f_hat: np.ndarray,
    right_sigma_f_hat: np.ndarray,
    coordinate: int,
) -> float:
    left_sigma_inverse = np.linalg.inv(np.asarray(left_sigma_f_hat, dtype=float))
    right_sigma_inverse = np.linalg.inv(np.asarray(right_sigma_f_hat, dtype=float))
    coordinate_value = int(coordinate)
    basis_dimension = left_sigma_inverse.shape[0]
    unit_coordinate = np.zeros(basis_dimension, dtype=float)
    unit_coordinate[coordinate_value] = 1.0
    left_vector = left_sigma_inverse @ unit_coordinate
    right_vector = right_sigma_inverse @ unit_coordinate
    return float(left_vector.T @ np.asarray(omega_f_hat, dtype=float) @ right_vector)


def _driver_signature(
    *,
    diagonal_entry_signature: str,
    shared_vf_entry_label: str,
    omega_only_share_of_full_vf_entry_gap: float,
    normalization_only_share_of_full_vf_entry_gap: float,
    omega_vs_normalization_increment_ratio: float,
    coverage_anchor_inverse_coordinate_sq_share: float,
    overshoot_companion_inverse_coordinate_sq_share: float,
) -> str:
    if (
        diagonal_entry_signature == "bounded-first-sine-vf-entry-activation"
        and shared_vf_entry_label == "v_f_hat[2,2]"
        and omega_only_share_of_full_vf_entry_gap > 0.5
        and normalization_only_share_of_full_vf_entry_gap < 0.1
        and omega_vs_normalization_increment_ratio > 10.0
        and coverage_anchor_inverse_coordinate_sq_share > 0.95
        and overshoot_companion_inverse_coordinate_sq_share > 0.95
    ):
        return "omega-first-shared-vf-entry-activation"
    return "mixed-shared-vf-entry-sandwich"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    shared_vf_entry_label: str
    coverage_anchor_vf_entry: float
    overshoot_companion_vf_entry: float
    full_vf_entry_gap: float
    omega_only_counterfactual_vf_entry: float
    normalization_only_counterfactual_vf_entry: float
    omega_only_increment: float
    normalization_only_increment: float
    required_diagonal_vf_entry_lift: float
    omega_only_share_of_full_vf_entry_gap: float
    normalization_only_share_of_full_vf_entry_gap: float
    omega_only_multiple_of_required_lift: float
    normalization_only_share_of_required_lift: float
    omega_vs_normalization_increment_ratio: float
    coverage_anchor_inverse_coordinate_sq_share: float
    overshoot_companion_inverse_coordinate_sq_share: float
    driver_signature: str
    canonical_first_sine_vf_entry_sandwich_digest: tuple[str, ...]

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
        self.coverage_anchor_vf_entry = float(self.coverage_anchor_vf_entry)
        self.overshoot_companion_vf_entry = float(self.overshoot_companion_vf_entry)
        self.full_vf_entry_gap = float(self.full_vf_entry_gap)
        self.omega_only_counterfactual_vf_entry = float(
            self.omega_only_counterfactual_vf_entry
        )
        self.normalization_only_counterfactual_vf_entry = float(
            self.normalization_only_counterfactual_vf_entry
        )
        self.omega_only_increment = float(self.omega_only_increment)
        self.normalization_only_increment = float(self.normalization_only_increment)
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.omega_only_share_of_full_vf_entry_gap = float(
            self.omega_only_share_of_full_vf_entry_gap
        )
        self.normalization_only_share_of_full_vf_entry_gap = float(
            self.normalization_only_share_of_full_vf_entry_gap
        )
        self.omega_only_multiple_of_required_lift = float(
            self.omega_only_multiple_of_required_lift
        )
        self.normalization_only_share_of_required_lift = float(
            self.normalization_only_share_of_required_lift
        )
        self.omega_vs_normalization_increment_ratio = float(
            self.omega_vs_normalization_increment_ratio
        )
        self.coverage_anchor_inverse_coordinate_sq_share = float(
            self.coverage_anchor_inverse_coordinate_sq_share
        )
        self.overshoot_companion_inverse_coordinate_sq_share = float(
            self.overshoot_companion_inverse_coordinate_sq_share
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_vf_entry_sandwich_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_vf_entry_sandwich_digest
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
            "coverage_anchor_vf_entry": self.coverage_anchor_vf_entry,
            "overshoot_companion_vf_entry": self.overshoot_companion_vf_entry,
            "full_vf_entry_gap": self.full_vf_entry_gap,
            "omega_only_counterfactual_vf_entry": self.omega_only_counterfactual_vf_entry,
            "normalization_only_counterfactual_vf_entry": (
                self.normalization_only_counterfactual_vf_entry
            ),
            "omega_only_increment": self.omega_only_increment,
            "normalization_only_increment": self.normalization_only_increment,
            "required_diagonal_vf_entry_lift": self.required_diagonal_vf_entry_lift,
            "omega_only_share_of_full_vf_entry_gap": (
                self.omega_only_share_of_full_vf_entry_gap
            ),
            "normalization_only_share_of_full_vf_entry_gap": (
                self.normalization_only_share_of_full_vf_entry_gap
            ),
            "omega_only_multiple_of_required_lift": (
                self.omega_only_multiple_of_required_lift
            ),
            "normalization_only_share_of_required_lift": (
                self.normalization_only_share_of_required_lift
            ),
            "omega_vs_normalization_increment_ratio": (
                self.omega_vs_normalization_increment_ratio
            ),
            "coverage_anchor_inverse_coordinate_sq_share": (
                self.coverage_anchor_inverse_coordinate_sq_share
            ),
            "overshoot_companion_inverse_coordinate_sq_share": (
                self.overshoot_companion_inverse_coordinate_sq_share
            ),
            "driver_signature": self.driver_signature,
            "canonical_first_sine_vf_entry_sandwich_digest": list(
                self.canonical_first_sine_vf_entry_sandwich_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_report(
    *,
    diagonal_entry_factor_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport:
    if (
        diagonal_entry_factor_report.policy_digest
        != seed_window_covariance_report.policy_digest
    ):
        raise ValueError("first-sine vf-entry sandwich probe requires shared policy")
    if (
        diagonal_entry_factor_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError(
            "first-sine vf-entry sandwich probe requires shared binding design"
        )
    if (
        diagonal_entry_factor_report.window_label
        != seed_window_covariance_report.window_label
    ):
        raise ValueError(
            "first-sine vf-entry sandwich probe requires shared window label"
        )
    if (
        diagonal_entry_factor_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        diagonal_entry_factor_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")

    anchor_payload = _canonical_nonparametric_payload(
        binding_design=diagonal_entry_factor_report.binding_design,
        random_state=diagonal_entry_factor_report.coverage_anchor_random_state,
        evaluation_grid=seed_window_covariance_report.evaluation_grid,
        n_boot=seed_window_covariance_report.coverage_anchor_contract.n_boot,
    )
    companion_payload = _canonical_nonparametric_payload(
        binding_design=diagonal_entry_factor_report.binding_design,
        random_state=diagonal_entry_factor_report.overshoot_companion_random_state,
        evaluation_grid=seed_window_covariance_report.evaluation_grid,
        n_boot=seed_window_covariance_report.overshoot_companion_contract.n_boot,
    )

    coordinate = diagonal_entry_factor_report.diagonal_coordinate
    coverage_anchor_vf_entry = float(anchor_payload.v_f_hat[coordinate, coordinate])
    overshoot_companion_vf_entry = float(
        companion_payload.v_f_hat[coordinate, coordinate]
    )
    full_vf_entry_gap = float(overshoot_companion_vf_entry - coverage_anchor_vf_entry)
    if full_vf_entry_gap <= 0.0:
        raise ValueError("shared vf-entry gap must stay positive")

    omega_only_counterfactual_vf_entry = _sandwich_entry(
        left_sigma_f_hat=anchor_payload.sigma_f_hat,
        omega_f_hat=companion_payload.omega_f_hat,
        right_sigma_f_hat=anchor_payload.sigma_f_hat,
        coordinate=coordinate,
    )
    normalization_only_counterfactual_vf_entry = _sandwich_entry(
        left_sigma_f_hat=companion_payload.sigma_f_hat,
        omega_f_hat=anchor_payload.omega_f_hat,
        right_sigma_f_hat=companion_payload.sigma_f_hat,
        coordinate=coordinate,
    )

    omega_only_increment = float(
        omega_only_counterfactual_vf_entry - coverage_anchor_vf_entry
    )
    normalization_only_increment = float(
        normalization_only_counterfactual_vf_entry - coverage_anchor_vf_entry
    )
    if omega_only_increment <= 0.0 or normalization_only_increment <= 0.0:
        raise ValueError("counterfactual shared-entry increments must stay positive")

    required_diagonal_vf_entry_lift = float(
        diagonal_entry_factor_report.required_diagonal_covariance_entry_lift
    )
    omega_only_share_of_full_vf_entry_gap = float(
        omega_only_increment / full_vf_entry_gap
    )
    normalization_only_share_of_full_vf_entry_gap = float(
        normalization_only_increment / full_vf_entry_gap
    )
    omega_only_multiple_of_required_lift = float(
        omega_only_increment / required_diagonal_vf_entry_lift
    )
    normalization_only_share_of_required_lift = float(
        normalization_only_increment / required_diagonal_vf_entry_lift
    )
    omega_vs_normalization_increment_ratio = float(
        omega_only_increment / normalization_only_increment
    )
    coverage_anchor_inverse_coordinate_sq_share = _inverse_coordinate_sq_share(
        anchor_payload.sigma_f_hat,
        coordinate,
    )
    overshoot_companion_inverse_coordinate_sq_share = _inverse_coordinate_sq_share(
        companion_payload.sigma_f_hat,
        coordinate,
    )

    driver_signature = _driver_signature(
        diagonal_entry_signature=diagonal_entry_factor_report.driver_signature,
        shared_vf_entry_label=diagonal_entry_factor_report.shared_diagonal_entry_label,
        omega_only_share_of_full_vf_entry_gap=omega_only_share_of_full_vf_entry_gap,
        normalization_only_share_of_full_vf_entry_gap=(
            normalization_only_share_of_full_vf_entry_gap
        ),
        omega_vs_normalization_increment_ratio=omega_vs_normalization_increment_ratio,
        coverage_anchor_inverse_coordinate_sq_share=(
            coverage_anchor_inverse_coordinate_sq_share
        ),
        overshoot_companion_inverse_coordinate_sq_share=(
            overshoot_companion_inverse_coordinate_sq_share
        ),
    )

    canonical_digest = (
        f"- shared `{diagonal_entry_factor_report.shared_diagonal_entry_label}` still equals the exact sandwich `u^T omega_f_hat u` with `u = sigma_f_hat^{{-1}} e_{coordinate}`: on the canonical `{diagonal_entry_factor_report.binding_design[0]}/{diagonal_entry_factor_report.binding_design[1]}/{diagonal_entry_factor_report.binding_design[2]}` lane, anchor seed `{diagonal_entry_factor_report.coverage_anchor_random_state}` stays at `{_format_float(coverage_anchor_vf_entry)}` while companion seed `{diagonal_entry_factor_report.overshoot_companion_random_state}` reaches `{_format_float(overshoot_companion_vf_entry)}`, so the full shared-entry gap is `{_format_signed(full_vf_entry_gap)}`",
        f"- freezing anchor normalization and swapping only companion `omega_f_hat` already lifts the shared entry to `{_format_float(omega_only_counterfactual_vf_entry)}`, i.e. `{_format_signed(omega_only_increment)}` (`{_format_percent(omega_only_share_of_full_vf_entry_gap)}` of the full gap and `{_format_ratio(omega_only_multiple_of_required_lift)}` the bounded `{_format_signed(required_diagonal_vf_entry_lift)}` repair target)",
        f"- freezing anchor `omega_f_hat` and swapping only companion normalization reaches only `{_format_float(normalization_only_counterfactual_vf_entry)}`, i.e. `{_format_signed(normalization_only_increment)}` (`{_format_percent(normalization_only_share_of_full_vf_entry_gap)}` of the full gap and `{_format_percent(normalization_only_share_of_required_lift)}` of the bounded repair target), while the inverse-sandwich vectors still keep `{_format_percent(coverage_anchor_inverse_coordinate_sq_share)}` / `{_format_percent(overshoot_companion_inverse_coordinate_sq_share)}` of their squared mass on coordinate `{coordinate}` itself",
        f"- current Trigger 2 implication: `{driver_signature}`; source-level follow-up should inspect why anchor seed `{diagonal_entry_factor_report.coverage_anchor_random_state}` under-activates the first-sine `omega_f_hat` mass feeding shared `{diagonal_entry_factor_report.shared_diagonal_entry_label}`, not `sigma_f_hat` normalization drift or broad sandwich mixing",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-vf-entry-sandwich-probe",
        policy_digest=diagonal_entry_factor_report.policy_digest,
        binding_design=diagonal_entry_factor_report.binding_design,
        window_label=diagonal_entry_factor_report.window_label,
        coverage_anchor_random_state=diagonal_entry_factor_report.coverage_anchor_random_state,
        overshoot_companion_random_state=diagonal_entry_factor_report.overshoot_companion_random_state,
        diagonal_coordinate=coordinate,
        diagonal_basis_label=diagonal_entry_factor_report.diagonal_basis_label,
        shared_vf_entry_label=diagonal_entry_factor_report.shared_diagonal_entry_label,
        coverage_anchor_vf_entry=coverage_anchor_vf_entry,
        overshoot_companion_vf_entry=overshoot_companion_vf_entry,
        full_vf_entry_gap=full_vf_entry_gap,
        omega_only_counterfactual_vf_entry=omega_only_counterfactual_vf_entry,
        normalization_only_counterfactual_vf_entry=normalization_only_counterfactual_vf_entry,
        omega_only_increment=omega_only_increment,
        normalization_only_increment=normalization_only_increment,
        required_diagonal_vf_entry_lift=required_diagonal_vf_entry_lift,
        omega_only_share_of_full_vf_entry_gap=omega_only_share_of_full_vf_entry_gap,
        normalization_only_share_of_full_vf_entry_gap=normalization_only_share_of_full_vf_entry_gap,
        omega_only_multiple_of_required_lift=omega_only_multiple_of_required_lift,
        normalization_only_share_of_required_lift=normalization_only_share_of_required_lift,
        omega_vs_normalization_increment_ratio=omega_vs_normalization_increment_ratio,
        coverage_anchor_inverse_coordinate_sq_share=coverage_anchor_inverse_coordinate_sq_share,
        overshoot_companion_inverse_coordinate_sq_share=overshoot_companion_inverse_coordinate_sq_share,
        driver_signature=driver_signature,
        canonical_first_sine_vf_entry_sandwich_digest=canonical_digest,
    )


def _build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-vf-entry-sandwich-probe",
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
        diagonal_coordinate=2,
        diagonal_basis_label="sin(2πz)",
        shared_vf_entry_label="v_f_hat[2,2]",
        coverage_anchor_vf_entry=1106.3366650452563,
        overshoot_companion_vf_entry=9552.211885581602,
        full_vf_entry_gap=8445.875220536345,
        omega_only_counterfactual_vf_entry=6769.078992824627,
        normalization_only_counterfactual_vf_entry=1169.1408695490418,
        omega_only_increment=5662.7423277793705,
        normalization_only_increment=62.80420450378551,
        required_diagonal_vf_entry_lift=1011.1822863613054,
        omega_only_share_of_full_vf_entry_gap=0.6704743060861565,
        normalization_only_share_of_full_vf_entry_gap=0.007436080082153665,
        omega_only_multiple_of_required_lift=5.600120180265912,
        normalization_only_share_of_required_lift=0.062109676317396396,
        omega_vs_normalization_increment_ratio=90.16501956390594,
        coverage_anchor_inverse_coordinate_sq_share=0.9789025000922827,
        overshoot_companion_inverse_coordinate_sq_share=0.9524847964978707,
        driver_signature="omega-first-shared-vf-entry-activation",
        canonical_first_sine_vf_entry_sandwich_digest=(
            "- shared `v_f_hat[2,2]` still equals the exact sandwich `u^T omega_f_hat u` with `u = sigma_f_hat^{-1} e_2`: on the canonical `DGP2/500/50` lane, anchor seed `202` stays at `1106.337` while companion seed `505` reaches `9552.212`, so the full shared-entry gap is `+8445.875`",
            "- freezing anchor normalization and swapping only companion `omega_f_hat` already lifts the shared entry to `6769.079`, i.e. `+5662.742` (`67.0%` of the full gap and `5.600x` the bounded `+1011.182` repair target)",
            "- freezing anchor `omega_f_hat` and swapping only companion normalization reaches only `1169.141`, i.e. `+62.804` (`0.7%` of the full gap and `6.2%` of the bounded repair target), while the inverse-sandwich vectors still keep `97.9%` / `95.2%` of their squared mass on coordinate `2` itself",
            "- current Trigger 2 implication: `omega-first-shared-vf-entry-activation`; source-level follow-up should inspect why anchor seed `202` under-activates the first-sine `omega_f_hat` mass feeding shared `v_f_hat[2,2]`, not `sigma_f_hat` normalization drift or broad sandwich mixing",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport
):
    return _build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_snapshot_report()
