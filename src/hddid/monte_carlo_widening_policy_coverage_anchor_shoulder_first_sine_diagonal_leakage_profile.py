from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .basis import trigonometric_sieve_basis


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _driver_signature(
    *,
    induced_right_center_lift: float,
    target_right_center_lift: float,
    induced_left_center_lift: float,
    induced_cross_shoulder_lift: float,
    preserved_left_center_covariance: float,
    anchor_cross_shoulder_covariance: float,
) -> str:
    if (
        np.isclose(induced_right_center_lift, target_right_center_lift, atol=1e-12)
        and induced_left_center_lift > 0.0
        and induced_cross_shoulder_lift > 0.0
        and induced_left_center_lift > preserved_left_center_covariance
        and abs(induced_cross_shoulder_lift) < abs(anchor_cross_shoulder_covariance)
    ):
        return "first-sine-diagonal-nonlocal-leakage-profile"
    return "mixed-first-sine-diagonal-leakage-profile"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalLeakageProfileReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    shared_diagonal_entry_label: str
    shared_n_valid: int
    required_diagonal_vf_entry_lift: float
    induced_right_center_lift: float
    induced_left_center_lift: float
    induced_cross_shoulder_lift: float
    induced_center_diagonal_lift: float
    induced_right_diagonal_lift: float
    left_center_share_of_target_lift: float
    cross_shoulder_share_of_target_lift: float
    center_diagonal_share_of_target_lift: float
    right_diagonal_share_of_target_lift: float
    preserved_left_center_covariance: float
    anchor_cross_shoulder_covariance: float
    left_center_multiple_of_preserved_baseline: float
    cross_shoulder_share_of_anchor_absolute_baseline: float
    driver_signature: str
    canonical_first_sine_diagonal_leakage_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.diagonal_coordinate = int(self.diagonal_coordinate)
        self.diagonal_basis_label = str(self.diagonal_basis_label).strip()
        self.shared_diagonal_entry_label = str(self.shared_diagonal_entry_label).strip()
        self.shared_n_valid = int(self.shared_n_valid)
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.induced_right_center_lift = float(self.induced_right_center_lift)
        self.induced_left_center_lift = float(self.induced_left_center_lift)
        self.induced_cross_shoulder_lift = float(self.induced_cross_shoulder_lift)
        self.induced_center_diagonal_lift = float(self.induced_center_diagonal_lift)
        self.induced_right_diagonal_lift = float(self.induced_right_diagonal_lift)
        self.left_center_share_of_target_lift = float(
            self.left_center_share_of_target_lift
        )
        self.cross_shoulder_share_of_target_lift = float(
            self.cross_shoulder_share_of_target_lift
        )
        self.center_diagonal_share_of_target_lift = float(
            self.center_diagonal_share_of_target_lift
        )
        self.right_diagonal_share_of_target_lift = float(
            self.right_diagonal_share_of_target_lift
        )
        self.preserved_left_center_covariance = float(
            self.preserved_left_center_covariance
        )
        self.anchor_cross_shoulder_covariance = float(
            self.anchor_cross_shoulder_covariance
        )
        self.left_center_multiple_of_preserved_baseline = float(
            self.left_center_multiple_of_preserved_baseline
        )
        self.cross_shoulder_share_of_anchor_absolute_baseline = float(
            self.cross_shoulder_share_of_anchor_absolute_baseline
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_diagonal_leakage_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_diagonal_leakage_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_leakage_profile_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalLeakageProfileReport
):
    policy_digest = (
        "label=bounded-n500-p50",
        "max_total_runtime_seconds=240.0",
        "max_random_states=8",
        "stop_on_first_typed_invalidity=True",
        "min_nonparametric_coverage=0.85",
    )
    binding_design = ("DGP2", 500, 50)
    evaluation_grid = (0.05, 0.15, 0.25)
    shared_n_valid = 500
    diagonal_coordinate = 2
    target_right_center_lift = 1.6361273081544214
    preserved_left_center_covariance = 0.26869174013344194
    anchor_cross_shoulder_covariance = -1.132110735050385

    basis = trigonometric_sieve_basis(
        np.asarray(evaluation_grid, dtype=float), degree=8
    )
    diagonal_values = np.asarray(basis[:, diagonal_coordinate], dtype=float)
    sign_product = float(diagonal_values[2] * diagonal_values[1])
    required_diagonal_vf_entry_lift = (
        target_right_center_lift * float(shared_n_valid) / sign_product
    )
    induced_matrix = (
        np.outer(diagonal_values, diagonal_values)
        * required_diagonal_vf_entry_lift
        / float(shared_n_valid)
    )

    induced_left_center_lift = float(induced_matrix[0, 1])
    induced_cross_shoulder_lift = float(induced_matrix[0, 2])
    induced_center_diagonal_lift = float(induced_matrix[1, 1])
    induced_right_diagonal_lift = float(induced_matrix[2, 2])
    left_center_share_of_target_lift = (
        induced_left_center_lift / target_right_center_lift
    )
    cross_shoulder_share_of_target_lift = (
        induced_cross_shoulder_lift / target_right_center_lift
    )
    center_diagonal_share_of_target_lift = (
        induced_center_diagonal_lift / target_right_center_lift
    )
    right_diagonal_share_of_target_lift = (
        induced_right_diagonal_lift / target_right_center_lift
    )
    left_center_multiple_of_preserved_baseline = (
        induced_left_center_lift / preserved_left_center_covariance
    )
    cross_shoulder_share_of_anchor_absolute_baseline = abs(
        induced_cross_shoulder_lift / anchor_cross_shoulder_covariance
    )
    driver_signature = _driver_signature(
        induced_right_center_lift=float(induced_matrix[2, 1]),
        target_right_center_lift=target_right_center_lift,
        induced_left_center_lift=induced_left_center_lift,
        induced_cross_shoulder_lift=induced_cross_shoulder_lift,
        preserved_left_center_covariance=preserved_left_center_covariance,
        anchor_cross_shoulder_covariance=anchor_cross_shoulder_covariance,
    )
    canonical_digest = (
        f"- the bounded source target still reproduces the live right-center requirement exactly through the first-sine diagonal lane: with `coordinate 2 = sin(2πz)`, `psi_2(0.25) * psi_2(0.15) = {_format_float(sign_product)}` and `n_valid = {shared_n_valid}`, the required shared lift `+{_format_float(required_diagonal_vf_entry_lift)}` on `v_f_hat[2,2]` yields the exact right-center increment `+{_format_float(target_right_center_lift)}`",
        f"- that same rank-one diagonal lift is not left-null: it would also induce `+{_format_float(induced_left_center_lift)}` on left-center and `+{_format_float(induced_cross_shoulder_lift)}` on cross-shoulder, equal to `{_format_percent(left_center_share_of_target_lift)}` / `{_format_percent(cross_shoulder_share_of_target_lift)}` of the target right-center lift; the left-center spillover alone is `{_format_ratio(left_center_multiple_of_preserved_baseline)}` the preserved left-center baseline `{_format_float(preserved_left_center_covariance)}`, while the cross-shoulder spillover is `{_format_percent(cross_shoulder_share_of_anchor_absolute_baseline)}` of the anchor cross-shoulder absolute baseline `{_format_float(abs(anchor_cross_shoulder_covariance))}`",
        f"- diagonal spillover is larger still: the same bounded `v_f_hat[2,2]` lift would add `+{_format_float(induced_center_diagonal_lift)}` to center-center and `+{_format_float(induced_right_diagonal_lift)}` to right-right, i.e. `{_format_percent(center_diagonal_share_of_target_lift)}` / `{_format_percent(right_diagonal_share_of_target_lift)}` of the target right-center increment, so the diagonal mainline explains the source target but not the validation-only exact patch geometry by itself",
        "- current Trigger 2 implication: `first-sine-diagonal-nonlocal-leakage-profile`; keep `omega_f_hat[2,2] -> v_f_hat[2,2]` as the source mainline, but do not misread a pure diagonal lift as a left-support-null production patch because a real estimator path still needs compensating geometry before it can match the validation-only two-cell witness",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalLeakageProfileReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-diagonal-leakage-profile",
        policy_digest=policy_digest,
        binding_design=binding_design,
        window_label="near_zero_grid",
        evaluation_grid=evaluation_grid,
        coverage_anchor_random_state=202,
        overshoot_companion_random_state=505,
        diagonal_coordinate=diagonal_coordinate,
        diagonal_basis_label="sin(2πz)",
        shared_diagonal_entry_label="v_f_hat[2,2]",
        shared_n_valid=shared_n_valid,
        required_diagonal_vf_entry_lift=required_diagonal_vf_entry_lift,
        induced_right_center_lift=float(induced_matrix[2, 1]),
        induced_left_center_lift=induced_left_center_lift,
        induced_cross_shoulder_lift=induced_cross_shoulder_lift,
        induced_center_diagonal_lift=induced_center_diagonal_lift,
        induced_right_diagonal_lift=induced_right_diagonal_lift,
        left_center_share_of_target_lift=left_center_share_of_target_lift,
        cross_shoulder_share_of_target_lift=cross_shoulder_share_of_target_lift,
        center_diagonal_share_of_target_lift=center_diagonal_share_of_target_lift,
        right_diagonal_share_of_target_lift=right_diagonal_share_of_target_lift,
        preserved_left_center_covariance=preserved_left_center_covariance,
        anchor_cross_shoulder_covariance=anchor_cross_shoulder_covariance,
        left_center_multiple_of_preserved_baseline=(
            left_center_multiple_of_preserved_baseline
        ),
        cross_shoulder_share_of_anchor_absolute_baseline=(
            cross_shoulder_share_of_anchor_absolute_baseline
        ),
        driver_signature=driver_signature,
        canonical_first_sine_diagonal_leakage_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_leakage_profile() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalLeakageProfileReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_leakage_profile_report()
