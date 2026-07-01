from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_entry_factor_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_entry_factor_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport,
    _canonical_nonparametric_payload,
    _sandwich_entry,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    sandwich_signature: str,
    shared_vf_entry_label: str,
    coordinate_axis_share_of_omega_only_increment: float,
    diagonal_share_of_omega_only_increment: float,
    offdiagonal_axis_share_of_omega_only_increment: float,
) -> str:
    if (
        sandwich_signature == "omega-first-shared-vf-entry-activation"
        and shared_vf_entry_label == "v_f_hat[2,2]"
        and coordinate_axis_share_of_omega_only_increment > 0.95
        and diagonal_share_of_omega_only_increment > 0.7
        and offdiagonal_axis_share_of_omega_only_increment < 0.3
    ):
        return "first-sine-omega-axis-activation"
    return "mixed-first-sine-omega-axis"


def _coordinate_axis_counterfactual_omega(
    *,
    anchor_omega_f_hat: np.ndarray,
    companion_omega_f_hat: np.ndarray,
    coordinate: int,
) -> np.ndarray:
    counterfactual = np.array(anchor_omega_f_hat, dtype=float, copy=True)
    counterfactual[int(coordinate), :] = np.asarray(
        companion_omega_f_hat[int(coordinate), :], dtype=float
    )
    counterfactual[:, int(coordinate)] = np.asarray(
        companion_omega_f_hat[:, int(coordinate)], dtype=float
    )
    return counterfactual


def _diagonal_only_counterfactual_omega(
    *,
    anchor_omega_f_hat: np.ndarray,
    companion_omega_f_hat: np.ndarray,
    coordinate: int,
) -> np.ndarray:
    counterfactual = np.array(anchor_omega_f_hat, dtype=float, copy=True)
    counterfactual[int(coordinate), int(coordinate)] = float(
        companion_omega_f_hat[int(coordinate), int(coordinate)]
    )
    return counterfactual


def _offdiagonal_axis_counterfactual_omega(
    *,
    anchor_omega_f_hat: np.ndarray,
    companion_omega_f_hat: np.ndarray,
    coordinate: int,
) -> np.ndarray:
    counterfactual = _coordinate_axis_counterfactual_omega(
        anchor_omega_f_hat=anchor_omega_f_hat,
        companion_omega_f_hat=companion_omega_f_hat,
        coordinate=coordinate,
    )
    counterfactual[int(coordinate), int(coordinate)] = float(
        anchor_omega_f_hat[int(coordinate), int(coordinate)]
    )
    return counterfactual


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    shared_vf_entry_label: str
    coverage_anchor_omega_diagonal_entry: float
    overshoot_companion_omega_diagonal_entry: float
    omega_diagonal_entry_gap: float
    coordinate_axis_only_counterfactual_vf_entry: float
    diagonal_only_counterfactual_vf_entry: float
    offdiagonal_axis_only_counterfactual_vf_entry: float
    coordinate_axis_only_increment: float
    diagonal_only_increment: float
    offdiagonal_axis_only_increment: float
    coordinate_axis_share_of_omega_only_increment: float
    diagonal_share_of_omega_only_increment: float
    offdiagonal_axis_share_of_omega_only_increment: float
    coordinate_axis_multiple_of_required_lift: float
    diagonal_multiple_of_required_lift: float
    offdiagonal_axis_multiple_of_required_lift: float
    driver_signature: str
    canonical_first_sine_omega_axis_digest: tuple[str, ...]

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
        self.coverage_anchor_omega_diagonal_entry = float(
            self.coverage_anchor_omega_diagonal_entry
        )
        self.overshoot_companion_omega_diagonal_entry = float(
            self.overshoot_companion_omega_diagonal_entry
        )
        self.omega_diagonal_entry_gap = float(self.omega_diagonal_entry_gap)
        self.coordinate_axis_only_counterfactual_vf_entry = float(
            self.coordinate_axis_only_counterfactual_vf_entry
        )
        self.diagonal_only_counterfactual_vf_entry = float(
            self.diagonal_only_counterfactual_vf_entry
        )
        self.offdiagonal_axis_only_counterfactual_vf_entry = float(
            self.offdiagonal_axis_only_counterfactual_vf_entry
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
        self.diagonal_multiple_of_required_lift = float(
            self.diagonal_multiple_of_required_lift
        )
        self.offdiagonal_axis_multiple_of_required_lift = float(
            self.offdiagonal_axis_multiple_of_required_lift
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_omega_axis_digest = tuple(
            str(line).rstrip() for line in self.canonical_first_sine_omega_axis_digest
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
            "coverage_anchor_omega_diagonal_entry": self.coverage_anchor_omega_diagonal_entry,
            "overshoot_companion_omega_diagonal_entry": (
                self.overshoot_companion_omega_diagonal_entry
            ),
            "omega_diagonal_entry_gap": self.omega_diagonal_entry_gap,
            "coordinate_axis_only_counterfactual_vf_entry": (
                self.coordinate_axis_only_counterfactual_vf_entry
            ),
            "diagonal_only_counterfactual_vf_entry": (
                self.diagonal_only_counterfactual_vf_entry
            ),
            "offdiagonal_axis_only_counterfactual_vf_entry": (
                self.offdiagonal_axis_only_counterfactual_vf_entry
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
            "diagonal_multiple_of_required_lift": self.diagonal_multiple_of_required_lift,
            "offdiagonal_axis_multiple_of_required_lift": (
                self.offdiagonal_axis_multiple_of_required_lift
            ),
            "driver_signature": self.driver_signature,
            "canonical_first_sine_omega_axis_digest": list(
                self.canonical_first_sine_omega_axis_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_report(
    *,
    diagonal_entry_factor_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport
    ),
    sandwich_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport:
    if diagonal_entry_factor_report.policy_digest != sandwich_report.policy_digest:
        raise ValueError("first-sine omega-axis probe requires shared policy")
    if diagonal_entry_factor_report.binding_design != sandwich_report.binding_design:
        raise ValueError("first-sine omega-axis probe requires shared binding design")
    if diagonal_entry_factor_report.window_label != sandwich_report.window_label:
        raise ValueError("first-sine omega-axis probe requires shared window label")
    if (
        diagonal_entry_factor_report.coverage_anchor_random_state
        != sandwich_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        diagonal_entry_factor_report.overshoot_companion_random_state
        != sandwich_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")

    coordinate = diagonal_entry_factor_report.diagonal_coordinate
    required_diagonal_vf_entry_lift = float(
        diagonal_entry_factor_report.required_diagonal_covariance_entry_lift
    )
    full_omega_only_increment = float(sandwich_report.omega_only_increment)
    if full_omega_only_increment <= 0.0:
        raise ValueError("omega-only increment must stay positive")

    anchor_payload = _canonical_nonparametric_payload(
        binding_design=diagonal_entry_factor_report.binding_design,
        random_state=diagonal_entry_factor_report.coverage_anchor_random_state,
        evaluation_grid=(0.05, 0.15, 0.25),
        n_boot=64,
    )
    companion_payload = _canonical_nonparametric_payload(
        binding_design=diagonal_entry_factor_report.binding_design,
        random_state=diagonal_entry_factor_report.overshoot_companion_random_state,
        evaluation_grid=(0.05, 0.15, 0.25),
        n_boot=64,
    )

    coverage_anchor_omega_diagonal_entry = float(
        anchor_payload.omega_f_hat[coordinate, coordinate]
    )
    overshoot_companion_omega_diagonal_entry = float(
        companion_payload.omega_f_hat[coordinate, coordinate]
    )
    omega_diagonal_entry_gap = float(
        overshoot_companion_omega_diagonal_entry - coverage_anchor_omega_diagonal_entry
    )
    if omega_diagonal_entry_gap <= 0.0:
        raise ValueError("omega diagonal-entry gap must stay positive")

    coordinate_axis_only_counterfactual_vf_entry = _sandwich_entry(
        left_sigma_f_hat=anchor_payload.sigma_f_hat,
        omega_f_hat=_coordinate_axis_counterfactual_omega(
            anchor_omega_f_hat=anchor_payload.omega_f_hat,
            companion_omega_f_hat=companion_payload.omega_f_hat,
            coordinate=coordinate,
        ),
        right_sigma_f_hat=anchor_payload.sigma_f_hat,
        coordinate=coordinate,
    )
    diagonal_only_counterfactual_vf_entry = _sandwich_entry(
        left_sigma_f_hat=anchor_payload.sigma_f_hat,
        omega_f_hat=_diagonal_only_counterfactual_omega(
            anchor_omega_f_hat=anchor_payload.omega_f_hat,
            companion_omega_f_hat=companion_payload.omega_f_hat,
            coordinate=coordinate,
        ),
        right_sigma_f_hat=anchor_payload.sigma_f_hat,
        coordinate=coordinate,
    )
    offdiagonal_axis_only_counterfactual_vf_entry = _sandwich_entry(
        left_sigma_f_hat=anchor_payload.sigma_f_hat,
        omega_f_hat=_offdiagonal_axis_counterfactual_omega(
            anchor_omega_f_hat=anchor_payload.omega_f_hat,
            companion_omega_f_hat=companion_payload.omega_f_hat,
            coordinate=coordinate,
        ),
        right_sigma_f_hat=anchor_payload.sigma_f_hat,
        coordinate=coordinate,
    )

    coverage_anchor_vf_entry = float(anchor_payload.v_f_hat[coordinate, coordinate])
    coordinate_axis_only_increment = float(
        coordinate_axis_only_counterfactual_vf_entry - coverage_anchor_vf_entry
    )
    diagonal_only_increment = float(
        diagonal_only_counterfactual_vf_entry - coverage_anchor_vf_entry
    )
    offdiagonal_axis_only_increment = float(
        offdiagonal_axis_only_counterfactual_vf_entry - coverage_anchor_vf_entry
    )
    if (
        coordinate_axis_only_increment <= 0.0
        or diagonal_only_increment <= 0.0
        or offdiagonal_axis_only_increment <= 0.0
    ):
        raise ValueError("omega-axis counterfactual increments must stay positive")

    coordinate_axis_share_of_omega_only_increment = float(
        coordinate_axis_only_increment / full_omega_only_increment
    )
    diagonal_share_of_omega_only_increment = float(
        diagonal_only_increment / full_omega_only_increment
    )
    offdiagonal_axis_share_of_omega_only_increment = float(
        offdiagonal_axis_only_increment / full_omega_only_increment
    )
    coordinate_axis_multiple_of_required_lift = float(
        coordinate_axis_only_increment / required_diagonal_vf_entry_lift
    )
    diagonal_multiple_of_required_lift = float(
        diagonal_only_increment / required_diagonal_vf_entry_lift
    )
    offdiagonal_axis_multiple_of_required_lift = float(
        offdiagonal_axis_only_increment / required_diagonal_vf_entry_lift
    )

    driver_signature = _driver_signature(
        sandwich_signature=sandwich_report.driver_signature,
        shared_vf_entry_label=diagonal_entry_factor_report.shared_diagonal_entry_label,
        coordinate_axis_share_of_omega_only_increment=(
            coordinate_axis_share_of_omega_only_increment
        ),
        diagonal_share_of_omega_only_increment=diagonal_share_of_omega_only_increment,
        offdiagonal_axis_share_of_omega_only_increment=(
            offdiagonal_axis_share_of_omega_only_increment
        ),
    )

    canonical_digest = (
        f"- with anchor `sigma_f_hat` frozen, swapping only companion coordinate-`{coordinate}` `omega_f_hat` row/column lifts shared `{diagonal_entry_factor_report.shared_diagonal_entry_label}` to `{_format_float(coordinate_axis_only_counterfactual_vf_entry)}`, i.e. `{_format_signed(coordinate_axis_only_increment)}` (`{_format_percent(coordinate_axis_share_of_omega_only_increment)}` of the full omega-only lift and `{_format_ratio(coordinate_axis_multiple_of_required_lift)}` the bounded `{_format_signed(required_diagonal_vf_entry_lift)}` repair target)",
        f"- swapping only the first-sine diagonal `omega_f_hat[{coordinate},{coordinate}]` already reaches `{_format_float(diagonal_only_counterfactual_vf_entry)}`, i.e. `{_format_signed(diagonal_only_increment)}` (`{_format_percent(diagonal_share_of_omega_only_increment)}` of the omega-only lift), while the coordinate-`{coordinate}` off-diagonal axis alone still contributes `{_format_signed(offdiagonal_axis_only_increment)}` (`{_format_percent(offdiagonal_axis_share_of_omega_only_increment)}`) without any broad off-axis replay",
        f"- the canonical first-sine diagonal `omega_f_hat[{coordinate},{coordinate}]` itself rises from `{_format_float(coverage_anchor_omega_diagonal_entry)}` in coverage anchor seed `{diagonal_entry_factor_report.coverage_anchor_random_state}` to `{_format_float(overshoot_companion_omega_diagonal_entry)}` in overshoot companion seed `{diagonal_entry_factor_report.overshoot_companion_random_state}`, but the decisive source-level atom is the whole coordinate-`{coordinate}` omega axis rather than normalization drift or full-matrix mixing",
        f"- current Trigger 2 implication: `{driver_signature}`; source-level follow-up should inspect why anchor seed `{diagonal_entry_factor_report.coverage_anchor_random_state}` under-activates coordinate-`{coordinate} = {diagonal_entry_factor_report.diagonal_basis_label}` `omega_f_hat` row/column mass feeding shared `{diagonal_entry_factor_report.shared_diagonal_entry_label}`, not `sigma_f_hat` normalization or broad off-axis omega replay",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-omega-axis-probe",
        policy_digest=diagonal_entry_factor_report.policy_digest,
        binding_design=diagonal_entry_factor_report.binding_design,
        window_label=diagonal_entry_factor_report.window_label,
        coverage_anchor_random_state=diagonal_entry_factor_report.coverage_anchor_random_state,
        overshoot_companion_random_state=diagonal_entry_factor_report.overshoot_companion_random_state,
        diagonal_coordinate=coordinate,
        diagonal_basis_label=diagonal_entry_factor_report.diagonal_basis_label,
        shared_vf_entry_label=diagonal_entry_factor_report.shared_diagonal_entry_label,
        coverage_anchor_omega_diagonal_entry=coverage_anchor_omega_diagonal_entry,
        overshoot_companion_omega_diagonal_entry=overshoot_companion_omega_diagonal_entry,
        omega_diagonal_entry_gap=omega_diagonal_entry_gap,
        coordinate_axis_only_counterfactual_vf_entry=coordinate_axis_only_counterfactual_vf_entry,
        diagonal_only_counterfactual_vf_entry=diagonal_only_counterfactual_vf_entry,
        offdiagonal_axis_only_counterfactual_vf_entry=offdiagonal_axis_only_counterfactual_vf_entry,
        coordinate_axis_only_increment=coordinate_axis_only_increment,
        diagonal_only_increment=diagonal_only_increment,
        offdiagonal_axis_only_increment=offdiagonal_axis_only_increment,
        coordinate_axis_share_of_omega_only_increment=coordinate_axis_share_of_omega_only_increment,
        diagonal_share_of_omega_only_increment=diagonal_share_of_omega_only_increment,
        offdiagonal_axis_share_of_omega_only_increment=offdiagonal_axis_share_of_omega_only_increment,
        coordinate_axis_multiple_of_required_lift=coordinate_axis_multiple_of_required_lift,
        diagonal_multiple_of_required_lift=diagonal_multiple_of_required_lift,
        offdiagonal_axis_multiple_of_required_lift=offdiagonal_axis_multiple_of_required_lift,
        driver_signature=driver_signature,
        canonical_first_sine_omega_axis_digest=canonical_digest,
    )


def _build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-omega-axis-probe",
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
        coverage_anchor_omega_diagonal_entry=225.0160026858435,
        overshoot_companion_omega_diagonal_entry=992.3970966590908,
        omega_diagonal_entry_gap=767.3810939732473,
        coordinate_axis_only_counterfactual_vf_entry=6693.727183853571,
        diagonal_only_counterfactual_vf_entry=5290.368614314929,
        offdiagonal_axis_only_counterfactual_vf_entry=2509.6952345838954,
        coordinate_axis_only_increment=5587.3905188083145,
        diagonal_only_increment=4184.031949269673,
        offdiagonal_axis_only_increment=1403.3585695386391,
        coordinate_axis_share_of_omega_only_increment=0.9866934067260297,
        diagonal_share_of_omega_only_increment=0.7388702694000965,
        offdiagonal_axis_share_of_omega_only_increment=0.24782313732593278,
        coordinate_axis_multiple_of_required_lift=5.52560165874176,
        diagonal_multiple_of_required_lift=4.137762306265991,
        offdiagonal_axis_multiple_of_required_lift=1.3878393524757664,
        driver_signature="first-sine-omega-axis-activation",
        canonical_first_sine_omega_axis_digest=(
            "- with anchor `sigma_f_hat` frozen, swapping only companion coordinate-`2` `omega_f_hat` row/column lifts shared `v_f_hat[2,2]` to `6693.727`, i.e. `+5587.391` (`98.7%` of the full omega-only lift and `5.526x` the bounded `+1011.182` repair target)",
            "- swapping only the first-sine diagonal `omega_f_hat[2,2]` already reaches `5290.369`, i.e. `+4184.032` (`73.9%` of the omega-only lift), while the coordinate-`2` off-diagonal axis alone still contributes `+1403.359` (`24.8%`) without any broad off-axis replay",
            "- the canonical first-sine diagonal `omega_f_hat[2,2]` itself rises from `225.016` in coverage anchor seed `202` to `992.397` in overshoot companion seed `505`, but the decisive source-level atom is the whole coordinate-`2` omega axis rather than normalization drift or full-matrix mixing",
            "- current Trigger 2 implication: `first-sine-omega-axis-activation`; source-level follow-up should inspect why anchor seed `202` under-activates coordinate-`2 = sin(2πz)` `omega_f_hat` row/column mass feeding shared `v_f_hat[2,2]`, not `sigma_f_hat` normalization or broad off-axis omega replay",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport
):
    return _build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_snapshot_report()
