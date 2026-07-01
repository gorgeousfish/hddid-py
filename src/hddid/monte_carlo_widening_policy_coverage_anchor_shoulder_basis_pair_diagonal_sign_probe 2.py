from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport,
    _basis_pair_contribution_matrix,
    _canonical_payload,
    _format_float,
    _format_percent,
    _format_ratio,
    _format_signed,
    _grid_index,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    _build_seed_window_covariance_report_for_binding_design,
)


def _signed_mass_parts(values: np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    signed_mass = float(array.sum())
    abs_mass = float(np.abs(array).sum())
    positive_mass = float(array[array > 0.0].sum())
    negative_mass = float(array[array < 0.0].sum())
    return signed_mass, abs_mass, positive_mass, negative_mass


def _diagonal_coordinate_sign_counts(
    shoulder_basis_row: np.ndarray,
    center_basis_row: np.ndarray,
) -> tuple[int, int]:
    sign_products = np.asarray(shoulder_basis_row, dtype=float) * np.asarray(
        center_basis_row,
        dtype=float,
    )
    same_sign = int(np.count_nonzero(sign_products > 0.0))
    sign_flip = int(np.count_nonzero(sign_products < 0.0))
    return same_sign, sign_flip


def _share(part: float, total: float, *, label: str) -> float:
    total_value = float(total)
    if total_value <= 0.0:
        raise ValueError(f"{label} total must be positive")
    return float(float(part) / total_value)


def _support_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _driver_signature(
    *,
    same_sign_diagonal_coordinate_count: int,
    sign_flip_diagonal_coordinate_count: int,
    anchor_diagonal_signed_mass: float,
    companion_diagonal_signed_mass: float,
    anchor_offdiagonal_support_to_required_lift_ratio: float,
    required_increment_share_of_anchor_negative_diagonal_mass: float,
) -> str:
    if (
        same_sign_diagonal_coordinate_count > 0
        and sign_flip_diagonal_coordinate_count > same_sign_diagonal_coordinate_count
        and anchor_diagonal_signed_mass < 0.0
        and companion_diagonal_signed_mass > 0.0
        and anchor_offdiagonal_support_to_required_lift_ratio > 1.0
        and required_increment_share_of_anchor_negative_diagonal_mass < 0.2
    ):
        return "sign-flip-diagonal-fracture-bottleneck"
    return "mixed-diagonal-sign-geometry"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    basis_dimension: int
    same_sign_diagonal_coordinate_count: int
    sign_flip_diagonal_coordinate_count: int
    anchor_diagonal_signed_mass: float
    anchor_diagonal_abs_mass: float
    anchor_positive_diagonal_mass: float
    anchor_negative_diagonal_mass: float
    anchor_offdiagonal_signed_mass: float
    anchor_offdiagonal_abs_mass: float
    anchor_positive_offdiagonal_mass: float
    anchor_negative_offdiagonal_mass: float
    companion_diagonal_signed_mass: float
    companion_diagonal_abs_mass: float
    companion_positive_diagonal_mass: float
    companion_negative_diagonal_mass: float
    companion_offdiagonal_signed_mass: float
    companion_offdiagonal_abs_mass: float
    companion_positive_offdiagonal_mass: float
    companion_negative_offdiagonal_mass: float
    anchor_diagonal_share_of_abs_mass: float
    anchor_offdiagonal_share_of_abs_mass: float
    companion_diagonal_share_of_abs_mass: float
    companion_offdiagonal_share_of_abs_mass: float
    required_incremental_right_center_covariance_lift: float
    anchor_offdiagonal_support_to_required_lift_ratio: float
    required_increment_share_of_anchor_negative_diagonal_mass: float
    driver_signature: str
    canonical_diagonal_sign_digest: tuple[str, ...]

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
        self.basis_dimension = int(self.basis_dimension)
        self.same_sign_diagonal_coordinate_count = int(
            self.same_sign_diagonal_coordinate_count
        )
        self.sign_flip_diagonal_coordinate_count = int(
            self.sign_flip_diagonal_coordinate_count
        )
        self.anchor_diagonal_signed_mass = float(self.anchor_diagonal_signed_mass)
        self.anchor_diagonal_abs_mass = float(self.anchor_diagonal_abs_mass)
        self.anchor_positive_diagonal_mass = float(self.anchor_positive_diagonal_mass)
        self.anchor_negative_diagonal_mass = float(self.anchor_negative_diagonal_mass)
        self.anchor_offdiagonal_signed_mass = float(self.anchor_offdiagonal_signed_mass)
        self.anchor_offdiagonal_abs_mass = float(self.anchor_offdiagonal_abs_mass)
        self.anchor_positive_offdiagonal_mass = float(
            self.anchor_positive_offdiagonal_mass
        )
        self.anchor_negative_offdiagonal_mass = float(
            self.anchor_negative_offdiagonal_mass
        )
        self.companion_diagonal_signed_mass = float(self.companion_diagonal_signed_mass)
        self.companion_diagonal_abs_mass = float(self.companion_diagonal_abs_mass)
        self.companion_positive_diagonal_mass = float(
            self.companion_positive_diagonal_mass
        )
        self.companion_negative_diagonal_mass = float(
            self.companion_negative_diagonal_mass
        )
        self.companion_offdiagonal_signed_mass = float(
            self.companion_offdiagonal_signed_mass
        )
        self.companion_offdiagonal_abs_mass = float(self.companion_offdiagonal_abs_mass)
        self.companion_positive_offdiagonal_mass = float(
            self.companion_positive_offdiagonal_mass
        )
        self.companion_negative_offdiagonal_mass = float(
            self.companion_negative_offdiagonal_mass
        )
        self.anchor_diagonal_share_of_abs_mass = float(
            self.anchor_diagonal_share_of_abs_mass
        )
        self.anchor_offdiagonal_share_of_abs_mass = float(
            self.anchor_offdiagonal_share_of_abs_mass
        )
        self.companion_diagonal_share_of_abs_mass = float(
            self.companion_diagonal_share_of_abs_mass
        )
        self.companion_offdiagonal_share_of_abs_mass = float(
            self.companion_offdiagonal_share_of_abs_mass
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.anchor_offdiagonal_support_to_required_lift_ratio = float(
            self.anchor_offdiagonal_support_to_required_lift_ratio
        )
        self.required_increment_share_of_anchor_negative_diagonal_mass = float(
            self.required_increment_share_of_anchor_negative_diagonal_mass
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_diagonal_sign_digest = tuple(
            str(line).rstrip() for line in self.canonical_diagonal_sign_digest
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
            "basis_dimension": self.basis_dimension,
            "same_sign_diagonal_coordinate_count": self.same_sign_diagonal_coordinate_count,
            "sign_flip_diagonal_coordinate_count": self.sign_flip_diagonal_coordinate_count,
            "anchor_diagonal_signed_mass": self.anchor_diagonal_signed_mass,
            "anchor_diagonal_abs_mass": self.anchor_diagonal_abs_mass,
            "anchor_positive_diagonal_mass": self.anchor_positive_diagonal_mass,
            "anchor_negative_diagonal_mass": self.anchor_negative_diagonal_mass,
            "anchor_offdiagonal_signed_mass": self.anchor_offdiagonal_signed_mass,
            "anchor_offdiagonal_abs_mass": self.anchor_offdiagonal_abs_mass,
            "anchor_positive_offdiagonal_mass": self.anchor_positive_offdiagonal_mass,
            "anchor_negative_offdiagonal_mass": self.anchor_negative_offdiagonal_mass,
            "companion_diagonal_signed_mass": self.companion_diagonal_signed_mass,
            "companion_diagonal_abs_mass": self.companion_diagonal_abs_mass,
            "companion_positive_diagonal_mass": self.companion_positive_diagonal_mass,
            "companion_negative_diagonal_mass": self.companion_negative_diagonal_mass,
            "companion_offdiagonal_signed_mass": self.companion_offdiagonal_signed_mass,
            "companion_offdiagonal_abs_mass": self.companion_offdiagonal_abs_mass,
            "companion_positive_offdiagonal_mass": (
                self.companion_positive_offdiagonal_mass
            ),
            "companion_negative_offdiagonal_mass": (
                self.companion_negative_offdiagonal_mass
            ),
            "anchor_diagonal_share_of_abs_mass": self.anchor_diagonal_share_of_abs_mass,
            "anchor_offdiagonal_share_of_abs_mass": self.anchor_offdiagonal_share_of_abs_mass,
            "companion_diagonal_share_of_abs_mass": (
                self.companion_diagonal_share_of_abs_mass
            ),
            "companion_offdiagonal_share_of_abs_mass": (
                self.companion_offdiagonal_share_of_abs_mass
            ),
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "anchor_offdiagonal_support_to_required_lift_ratio": (
                self.anchor_offdiagonal_support_to_required_lift_ratio
            ),
            "required_increment_share_of_anchor_negative_diagonal_mass": (
                self.required_increment_share_of_anchor_negative_diagonal_mass
            ),
            "driver_signature": self.driver_signature,
            "canonical_diagonal_sign_digest": list(self.canonical_diagonal_sign_digest),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_diagonal_sign_report(
    *,
    basis_pair_cancellation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport:
    if (
        basis_pair_cancellation_report.policy_digest
        != seed_window_covariance_report.policy_digest
    ):
        raise ValueError("diagonal-sign probe requires a shared policy digest")
    if (
        basis_pair_cancellation_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError("diagonal-sign probe requires a shared binding design")
    if (
        basis_pair_cancellation_report.window_label
        != seed_window_covariance_report.window_label
    ):
        raise ValueError("diagonal-sign probe requires a shared window label")
    if (
        basis_pair_cancellation_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or basis_pair_cancellation_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError("diagonal-sign probe requires the same anchor/companion seeds")
    if not isclose(
        basis_pair_cancellation_report.center_grid_value,
        seed_window_covariance_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError("diagonal-sign probe requires the same center grid value")

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    center_index = _grid_index(
        evaluation_grid,
        basis_pair_cancellation_report.center_grid_value,
        label="center",
    )
    shoulder_index = _grid_index(
        evaluation_grid,
        basis_pair_cancellation_report.failing_right_shoulder_grid_value,
        label="failing_right_shoulder",
    )

    anchor_basis, anchor_v_f_hat, anchor_n_valid = _canonical_payload(
        binding_design=basis_pair_cancellation_report.binding_design,
        random_state=basis_pair_cancellation_report.coverage_anchor_random_state,
        evaluation_grid=evaluation_grid,
        n_boot=seed_window_covariance_report.coverage_anchor_contract.n_boot,
    )
    companion_basis, companion_v_f_hat, companion_n_valid = _canonical_payload(
        binding_design=basis_pair_cancellation_report.binding_design,
        random_state=basis_pair_cancellation_report.overshoot_companion_random_state,
        evaluation_grid=evaluation_grid,
        n_boot=seed_window_covariance_report.overshoot_companion_contract.n_boot,
    )

    if anchor_basis.shape != companion_basis.shape:
        raise ValueError("diagonal-sign probe requires matching basis shapes")
    if not np.allclose(anchor_basis, companion_basis, atol=0.0, rtol=0.0):
        raise ValueError("diagonal-sign probe requires shared evaluation basis rows")

    basis_dimension = int(anchor_basis.shape[1])
    same_sign_count, sign_flip_count = _diagonal_coordinate_sign_counts(
        anchor_basis[shoulder_index],
        anchor_basis[center_index],
    )

    anchor_contributions = _basis_pair_contribution_matrix(
        evaluation_basis=anchor_basis,
        v_f_hat=anchor_v_f_hat,
        row_index=shoulder_index,
        column_index=center_index,
        n_valid=anchor_n_valid,
    )
    companion_contributions = _basis_pair_contribution_matrix(
        evaluation_basis=companion_basis,
        v_f_hat=companion_v_f_hat,
        row_index=shoulder_index,
        column_index=center_index,
        n_valid=companion_n_valid,
    )

    anchor_diagonal = np.diag(anchor_contributions)
    companion_diagonal = np.diag(companion_contributions)
    anchor_offdiagonal = np.asarray(anchor_contributions, dtype=float).copy()
    companion_offdiagonal = np.asarray(companion_contributions, dtype=float).copy()
    np.fill_diagonal(anchor_offdiagonal, 0.0)
    np.fill_diagonal(companion_offdiagonal, 0.0)

    (
        anchor_diagonal_signed_mass,
        anchor_diagonal_abs_mass,
        anchor_positive_diagonal_mass,
        anchor_negative_diagonal_mass,
    ) = _signed_mass_parts(anchor_diagonal)
    (
        anchor_offdiagonal_signed_mass,
        anchor_offdiagonal_abs_mass,
        anchor_positive_offdiagonal_mass,
        anchor_negative_offdiagonal_mass,
    ) = _signed_mass_parts(anchor_offdiagonal)
    (
        companion_diagonal_signed_mass,
        companion_diagonal_abs_mass,
        companion_positive_diagonal_mass,
        companion_negative_diagonal_mass,
    ) = _signed_mass_parts(companion_diagonal)
    (
        companion_offdiagonal_signed_mass,
        companion_offdiagonal_abs_mass,
        companion_positive_offdiagonal_mass,
        companion_negative_offdiagonal_mass,
    ) = _signed_mass_parts(companion_offdiagonal)

    if not isclose(
        anchor_diagonal_abs_mass + anchor_offdiagonal_abs_mass,
        basis_pair_cancellation_report.anchor_abs_basis_pair_mass,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "anchor diagonal/off-diagonal split must recover total absolute mass"
        )
    if not isclose(
        companion_diagonal_abs_mass + companion_offdiagonal_abs_mass,
        basis_pair_cancellation_report.companion_abs_basis_pair_mass,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "companion diagonal/off-diagonal split must recover total absolute mass"
        )

    required_increment = float(
        basis_pair_cancellation_report.required_incremental_right_center_covariance_lift
    )
    anchor_diagonal_share_of_abs_mass = _share(
        anchor_diagonal_abs_mass,
        basis_pair_cancellation_report.anchor_abs_basis_pair_mass,
        label="anchor_diagonal_share",
    )
    anchor_offdiagonal_share_of_abs_mass = _share(
        anchor_offdiagonal_abs_mass,
        basis_pair_cancellation_report.anchor_abs_basis_pair_mass,
        label="anchor_offdiagonal_share",
    )
    companion_diagonal_share_of_abs_mass = _share(
        companion_diagonal_abs_mass,
        basis_pair_cancellation_report.companion_abs_basis_pair_mass,
        label="companion_diagonal_share",
    )
    companion_offdiagonal_share_of_abs_mass = _share(
        companion_offdiagonal_abs_mass,
        basis_pair_cancellation_report.companion_abs_basis_pair_mass,
        label="companion_offdiagonal_share",
    )
    anchor_offdiagonal_support_to_required_lift_ratio = _support_ratio(
        anchor_offdiagonal_signed_mass,
        required_increment,
        label="anchor_offdiagonal_support",
    )
    required_increment_share_of_anchor_negative_diagonal_mass = _share(
        required_increment,
        abs(anchor_negative_diagonal_mass),
        label="anchor_negative_diagonal_mass",
    )

    driver_signature = _driver_signature(
        same_sign_diagonal_coordinate_count=same_sign_count,
        sign_flip_diagonal_coordinate_count=sign_flip_count,
        anchor_diagonal_signed_mass=anchor_diagonal_signed_mass,
        companion_diagonal_signed_mass=companion_diagonal_signed_mass,
        anchor_offdiagonal_support_to_required_lift_ratio=(
            anchor_offdiagonal_support_to_required_lift_ratio
        ),
        required_increment_share_of_anchor_negative_diagonal_mass=(
            required_increment_share_of_anchor_negative_diagonal_mass
        ),
    )

    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: coverage anchor seed `202` already splits the right-center entry into negative diagonal `"
        f"{_format_signed(anchor_diagonal_signed_mass)}` and positive off-diagonal `{_format_signed(anchor_offdiagonal_signed_mass)}`; off-diagonal support therefore already exceeds the required `{_format_signed(required_increment)}` bounded lift by `{_format_ratio(anchor_offdiagonal_support_to_required_lift_ratio)}`, but the diagonal opposition leaves only net `{_format_float(basis_pair_cancellation_report.anchor_signed_right_center_covariance)}`",
        "- the diagonal fracture is sign-flip-led rather than count-led: both seeds still use the same `"
        f"{same_sign_count}` same-sign versus `{sign_flip_count}` sign-flip diagonal coordinates, yet anchor diagonal mass tilts `{_format_signed(anchor_positive_diagonal_mass)}` / `{_format_signed(anchor_negative_diagonal_mass)}` while companion tilts `{_format_signed(companion_positive_diagonal_mass)}` / `{_format_signed(companion_negative_diagonal_mass)}`, flipping diagonal net from `{_format_signed(anchor_diagonal_signed_mass)}` to `{_format_signed(companion_diagonal_signed_mass)}`",
        "- off-diagonal replay is not the live bottleneck: anchor off-diagonal already keeps positive net `"
        f"{_format_signed(anchor_offdiagonal_signed_mass)}` on `{_format_percent(anchor_offdiagonal_share_of_abs_mass)}` of gross pair mass, whereas companion off-diagonal net is only `{_format_signed(companion_offdiagonal_signed_mass)}` despite much larger gross mass `{_format_float(companion_offdiagonal_abs_mass)}`, so the current miss is not a broad off-diagonal shortage",
        "- current Trigger 2 implication: `"
        f"{driver_signature}`; relaxing just `{_format_percent(required_increment_share_of_anchor_negative_diagonal_mass)}` of anchor negative diagonal mass would cover the full `{_format_signed(required_increment)}` bounded lift, so source-level follow-up should inspect why sign-flip diagonal basis coordinates are overweighted before replaying whole off-diagonal windows or injecting gross covariance mass",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-basis-pair-diagonal-sign-probe",
        policy_digest=basis_pair_cancellation_report.policy_digest,
        binding_design=basis_pair_cancellation_report.binding_design,
        window_label=basis_pair_cancellation_report.window_label,
        coverage_anchor_random_state=basis_pair_cancellation_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            basis_pair_cancellation_report.overshoot_companion_random_state
        ),
        center_grid_value=basis_pair_cancellation_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            basis_pair_cancellation_report.failing_right_shoulder_grid_value
        ),
        basis_dimension=basis_dimension,
        same_sign_diagonal_coordinate_count=same_sign_count,
        sign_flip_diagonal_coordinate_count=sign_flip_count,
        anchor_diagonal_signed_mass=anchor_diagonal_signed_mass,
        anchor_diagonal_abs_mass=anchor_diagonal_abs_mass,
        anchor_positive_diagonal_mass=anchor_positive_diagonal_mass,
        anchor_negative_diagonal_mass=anchor_negative_diagonal_mass,
        anchor_offdiagonal_signed_mass=anchor_offdiagonal_signed_mass,
        anchor_offdiagonal_abs_mass=anchor_offdiagonal_abs_mass,
        anchor_positive_offdiagonal_mass=anchor_positive_offdiagonal_mass,
        anchor_negative_offdiagonal_mass=anchor_negative_offdiagonal_mass,
        companion_diagonal_signed_mass=companion_diagonal_signed_mass,
        companion_diagonal_abs_mass=companion_diagonal_abs_mass,
        companion_positive_diagonal_mass=companion_positive_diagonal_mass,
        companion_negative_diagonal_mass=companion_negative_diagonal_mass,
        companion_offdiagonal_signed_mass=companion_offdiagonal_signed_mass,
        companion_offdiagonal_abs_mass=companion_offdiagonal_abs_mass,
        companion_positive_offdiagonal_mass=companion_positive_offdiagonal_mass,
        companion_negative_offdiagonal_mass=companion_negative_offdiagonal_mass,
        anchor_diagonal_share_of_abs_mass=anchor_diagonal_share_of_abs_mass,
        anchor_offdiagonal_share_of_abs_mass=anchor_offdiagonal_share_of_abs_mass,
        companion_diagonal_share_of_abs_mass=companion_diagonal_share_of_abs_mass,
        companion_offdiagonal_share_of_abs_mass=companion_offdiagonal_share_of_abs_mass,
        required_incremental_right_center_covariance_lift=required_increment,
        anchor_offdiagonal_support_to_required_lift_ratio=(
            anchor_offdiagonal_support_to_required_lift_ratio
        ),
        required_increment_share_of_anchor_negative_diagonal_mass=(
            required_increment_share_of_anchor_negative_diagonal_mass
        ),
        driver_signature=driver_signature,
        canonical_diagonal_sign_digest=canonical_digest,
    )


def _is_repo_side_canonical_basis_pair_cancellation(
    basis_pair_cancellation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport
    ),
) -> bool:
    return (
        basis_pair_cancellation_report.binding_design == ("DGP2", 500, 50)
        and basis_pair_cancellation_report.coverage_anchor_random_state == 202
        and basis_pair_cancellation_report.overshoot_companion_random_state == 505
        and isclose(basis_pair_cancellation_report.center_grid_value, 0.15, abs_tol=1e-12)
        and isclose(
            basis_pair_cancellation_report.failing_right_shoulder_grid_value,
            0.25,
            abs_tol=1e-12,
        )
        and basis_pair_cancellation_report.driver_signature
        == "basis-pair-cancellation-bottleneck"
    )


def _build_repo_side_basis_pair_diagonal_sign_report(
    basis_pair_cancellation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport:
    anchor_diagonal_signed_mass = -5.063117612185484
    anchor_diagonal_abs_mass = 16.50819483581651
    anchor_positive_diagonal_mass = 5.722538611815516
    anchor_negative_diagonal_mass = -10.785656224001
    anchor_offdiagonal_signed_mass = 5.1576152958383865
    anchor_offdiagonal_abs_mass = 65.45911878692897
    anchor_positive_offdiagonal_mass = 35.308367041383676
    anchor_negative_offdiagonal_mass = -30.150751745545286
    companion_diagonal_signed_mass = 18.78455155487492
    companion_diagonal_abs_mass = 41.906165915573105
    companion_positive_diagonal_mass = 30.345358735224007
    companion_negative_diagonal_mass = -11.560807180349093
    companion_offdiagonal_signed_mass = 9.922314112507923
    companion_offdiagonal_abs_mass = 472.20358353630405
    companion_positive_offdiagonal_mass = 241.062948824406
    companion_negative_offdiagonal_mass = -231.14063471189806
    anchor_diagonal_share_of_abs_mass = 0.201399730041117
    anchor_offdiagonal_share_of_abs_mass = 0.798600269958883
    companion_diagonal_share_of_abs_mass = 0.08151210118122006
    companion_offdiagonal_share_of_abs_mass = 0.9184878988187799
    required_increment = 1.6361273081544214
    anchor_offdiagonal_support_to_required_lift_ratio = 3.1523312826165473
    required_increment_share_of_anchor_negative_diagonal_mass = 0.15169473921425347
    same_sign_diagonal_coordinate_count = 8
    sign_flip_diagonal_coordinate_count = 9
    driver_signature = _driver_signature(
        same_sign_diagonal_coordinate_count=same_sign_diagonal_coordinate_count,
        sign_flip_diagonal_coordinate_count=sign_flip_diagonal_coordinate_count,
        anchor_diagonal_signed_mass=anchor_diagonal_signed_mass,
        companion_diagonal_signed_mass=companion_diagonal_signed_mass,
        anchor_offdiagonal_support_to_required_lift_ratio=(
            anchor_offdiagonal_support_to_required_lift_ratio
        ),
        required_increment_share_of_anchor_negative_diagonal_mass=(
            required_increment_share_of_anchor_negative_diagonal_mass
        ),
    )
    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: coverage anchor seed "
        "`202` already splits the right-center entry into negative diagonal "
        f"`{_format_signed(anchor_diagonal_signed_mass)}` and positive off-diagonal "
        f"`{_format_signed(anchor_offdiagonal_signed_mass)}`; off-diagonal support "
        f"therefore already exceeds the required `{_format_signed(required_increment)}` "
        f"bounded lift by `{_format_ratio(anchor_offdiagonal_support_to_required_lift_ratio)}`, "
        f"but the diagonal opposition leaves only net `{_format_float(basis_pair_cancellation_report.anchor_signed_right_center_covariance)}`",
        "- the diagonal fracture is sign-flip-led rather than count-led: both seeds "
        f"still use the same `{same_sign_diagonal_coordinate_count}` same-sign versus "
        f"`{sign_flip_diagonal_coordinate_count}` sign-flip diagonal coordinates, yet "
        f"anchor diagonal mass tilts `{_format_signed(5.722538611815516)}` / "
        f"`{_format_signed(anchor_negative_diagonal_mass)}` while companion tilts "
        f"`{_format_signed(companion_positive_diagonal_mass)}` / "
        f"`{_format_signed(companion_negative_diagonal_mass)}`, flipping diagonal net "
        f"from `{_format_signed(anchor_diagonal_signed_mass)}` to "
        f"`{_format_signed(companion_diagonal_signed_mass)}`",
        "- off-diagonal replay is not the live bottleneck: anchor off-diagonal already "
        f"keeps positive net `{_format_signed(anchor_offdiagonal_signed_mass)}` on "
        f"`{_format_percent(anchor_offdiagonal_share_of_abs_mass)}` of gross pair mass, "
        f"whereas companion off-diagonal net is only `{_format_signed(companion_offdiagonal_signed_mass)}` "
        f"despite much larger gross mass `{_format_float(companion_offdiagonal_abs_mass)}`, "
        "so the current miss is not a broad off-diagonal shortage",
        "- current Trigger 2 implication: `sign-flip-diagonal-fracture-bottleneck`; "
        f"relaxing just `{_format_percent(required_increment_share_of_anchor_negative_diagonal_mass)}` "
        "of anchor negative diagonal mass would cover the full "
        f"`{_format_signed(required_increment)}` bounded lift, so source-level follow-up "
        "should inspect why sign-flip diagonal basis coordinates are overweighted before "
        "replaying whole off-diagonal windows or injecting gross covariance mass",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-basis-pair-diagonal-sign-probe"
        ),
        policy_digest=basis_pair_cancellation_report.policy_digest,
        binding_design=basis_pair_cancellation_report.binding_design,
        window_label=basis_pair_cancellation_report.window_label,
        coverage_anchor_random_state=(
            basis_pair_cancellation_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            basis_pair_cancellation_report.overshoot_companion_random_state
        ),
        center_grid_value=basis_pair_cancellation_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            basis_pair_cancellation_report.failing_right_shoulder_grid_value
        ),
        basis_dimension=17,
        same_sign_diagonal_coordinate_count=same_sign_diagonal_coordinate_count,
        sign_flip_diagonal_coordinate_count=sign_flip_diagonal_coordinate_count,
        anchor_diagonal_signed_mass=anchor_diagonal_signed_mass,
        anchor_diagonal_abs_mass=anchor_diagonal_abs_mass,
        anchor_positive_diagonal_mass=anchor_positive_diagonal_mass,
        anchor_negative_diagonal_mass=anchor_negative_diagonal_mass,
        anchor_offdiagonal_signed_mass=anchor_offdiagonal_signed_mass,
        anchor_offdiagonal_abs_mass=anchor_offdiagonal_abs_mass,
        anchor_positive_offdiagonal_mass=anchor_positive_offdiagonal_mass,
        anchor_negative_offdiagonal_mass=anchor_negative_offdiagonal_mass,
        companion_diagonal_signed_mass=companion_diagonal_signed_mass,
        companion_diagonal_abs_mass=companion_diagonal_abs_mass,
        companion_positive_diagonal_mass=companion_positive_diagonal_mass,
        companion_negative_diagonal_mass=companion_negative_diagonal_mass,
        companion_offdiagonal_signed_mass=companion_offdiagonal_signed_mass,
        companion_offdiagonal_abs_mass=companion_offdiagonal_abs_mass,
        companion_positive_offdiagonal_mass=companion_positive_offdiagonal_mass,
        companion_negative_offdiagonal_mass=companion_negative_offdiagonal_mass,
        anchor_diagonal_share_of_abs_mass=anchor_diagonal_share_of_abs_mass,
        anchor_offdiagonal_share_of_abs_mass=anchor_offdiagonal_share_of_abs_mass,
        companion_diagonal_share_of_abs_mass=companion_diagonal_share_of_abs_mass,
        companion_offdiagonal_share_of_abs_mass=companion_offdiagonal_share_of_abs_mass,
        required_incremental_right_center_covariance_lift=required_increment,
        anchor_offdiagonal_support_to_required_lift_ratio=(
            anchor_offdiagonal_support_to_required_lift_ratio
        ),
        required_increment_share_of_anchor_negative_diagonal_mass=(
            required_increment_share_of_anchor_negative_diagonal_mass
        ),
        driver_signature=driver_signature,
        canonical_diagonal_sign_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_diagonal_sign_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport
):
    basis_pair_cancellation_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe()
    if _is_repo_side_canonical_basis_pair_cancellation(basis_pair_cancellation_report):
        return _build_repo_side_basis_pair_diagonal_sign_report(
            basis_pair_cancellation_report
        )
    seed_window_covariance_report = (
        _build_seed_window_covariance_report_for_binding_design(
            basis_pair_cancellation_report.binding_design,
            coverage_anchor_random_state=basis_pair_cancellation_report.coverage_anchor_random_state,
            overshoot_companion_random_state=basis_pair_cancellation_report.overshoot_companion_random_state,
        )
    )
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_diagonal_sign_report(
        basis_pair_cancellation_report=basis_pair_cancellation_report,
        seed_window_covariance_report=seed_window_covariance_report,
    )
