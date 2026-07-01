from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe import (
    _basis_pair_contribution_matrix,
    _canonical_payload,
    _format_float,
    _format_percent,
    _format_signed,
    _grid_index,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_diagonal_sign_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_diagonal_sign_probe,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    _build_seed_window_covariance_report_for_binding_design,
)


def _same_sign_support_gap_ranking(
    *,
    shoulder_basis_row: np.ndarray,
    center_basis_row: np.ndarray,
    anchor_diagonal: np.ndarray,
    companion_diagonal: np.ndarray,
    tolerance: float = 1e-12,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    sign_products = np.asarray(shoulder_basis_row, dtype=float) * np.asarray(
        center_basis_row, dtype=float
    )
    anchor = np.asarray(anchor_diagonal, dtype=float)
    companion = np.asarray(companion_diagonal, dtype=float)
    ranking: list[tuple[int, float]] = []
    for index, sign_product in enumerate(sign_products):
        if float(sign_product) <= 0.0:
            continue
        gap = float(companion[index] - anchor[index])
        if gap > float(tolerance):
            ranking.append((int(index), gap))
    ranking.sort(key=lambda item: item[1], reverse=True)
    return (
        tuple(index for index, _ in ranking),
        tuple(gap for _, gap in ranking),
    )


def _share(part: float, total: float, *, label: str) -> float:
    total_value = float(total)
    if abs(total_value) <= np.finfo(float).eps:
        raise ValueError(f"{label} total must be nonzero")
    return float(float(part) / total_value)


def _top_share(
    values: tuple[float, ...], total: float, *, count: int, label: str
) -> float:
    count_value = int(count)
    if count_value <= 0:
        raise ValueError(f"{label} count must be positive")
    if not values:
        raise ValueError(f"{label} values must not be empty")
    total_value = float(total)
    if total_value <= 0.0:
        raise ValueError(f"{label} total must be positive")
    return float(sum(float(value) for value in values[:count_value]) / total_value)


def _format_coordinate_set(indices: tuple[int, ...]) -> str:
    return "{%s}" % ", ".join(str(int(index)) for index in indices)


def _driver_signature(
    *,
    same_sign_positive_gap_share_of_diagonal_net_gap: float,
    sign_flip_negative_gap_share_of_diagonal_net_gap: float,
    required_increment_share_of_same_sign_positive_gap: float,
    top_four_same_sign_support_gap_share: float,
) -> str:
    if (
        same_sign_positive_gap_share_of_diagonal_net_gap > 1.0
        and sign_flip_negative_gap_share_of_diagonal_net_gap < 0.0
        and required_increment_share_of_same_sign_positive_gap < 0.1
        and top_four_same_sign_support_gap_share >= 0.999999999
    ):
        return "same-sign-diagonal-support-deficit"
    return "mixed-diagonal-support-geometry"


def _is_repo_side_canonical_diagonal_sign(
    diagonal_sign_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport
    ),
) -> bool:
    return (
        diagonal_sign_report.binding_design == ("DGP2", 500, 50)
        and diagonal_sign_report.window_label == "near_zero_grid"
        and diagonal_sign_report.coverage_anchor_random_state == 202
        and diagonal_sign_report.overshoot_companion_random_state == 505
        and isclose(diagonal_sign_report.center_grid_value, 0.15, abs_tol=1e-12)
        and isclose(
            diagonal_sign_report.failing_right_shoulder_grid_value,
            0.25,
            abs_tol=1e-12,
        )
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignDiagonalSupportReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    same_sign_coordinate_count: int
    sign_flip_coordinate_count: int
    anchor_positive_diagonal_mass: float
    companion_positive_diagonal_mass: float
    anchor_negative_diagonal_mass: float
    companion_negative_diagonal_mass: float
    same_sign_positive_support_gap: float
    sign_flip_negative_mass_gap: float
    same_sign_positive_gap_share_of_diagonal_net_gap: float
    sign_flip_negative_gap_share_of_diagonal_net_gap: float
    same_sign_support_gap_coordinate_ranking: tuple[int, ...]
    top_same_sign_support_gap_coordinate: int
    top_same_sign_support_gap: float
    top_same_sign_support_gap_share: float
    top_two_same_sign_support_gap_share: float
    top_four_same_sign_support_gap_share: float
    required_incremental_right_center_covariance_lift: float
    required_increment_share_of_same_sign_positive_gap: float
    required_increment_share_of_top_same_sign_support_gap: float
    driver_signature: str
    canonical_same_sign_diagonal_support_digest: tuple[str, ...]

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
        self.same_sign_coordinate_count = int(self.same_sign_coordinate_count)
        self.sign_flip_coordinate_count = int(self.sign_flip_coordinate_count)
        self.anchor_positive_diagonal_mass = float(self.anchor_positive_diagonal_mass)
        self.companion_positive_diagonal_mass = float(
            self.companion_positive_diagonal_mass
        )
        self.anchor_negative_diagonal_mass = float(self.anchor_negative_diagonal_mass)
        self.companion_negative_diagonal_mass = float(
            self.companion_negative_diagonal_mass
        )
        self.same_sign_positive_support_gap = float(self.same_sign_positive_support_gap)
        self.sign_flip_negative_mass_gap = float(self.sign_flip_negative_mass_gap)
        self.same_sign_positive_gap_share_of_diagonal_net_gap = float(
            self.same_sign_positive_gap_share_of_diagonal_net_gap
        )
        self.sign_flip_negative_gap_share_of_diagonal_net_gap = float(
            self.sign_flip_negative_gap_share_of_diagonal_net_gap
        )
        self.same_sign_support_gap_coordinate_ranking = tuple(
            int(index) for index in self.same_sign_support_gap_coordinate_ranking
        )
        self.top_same_sign_support_gap_coordinate = int(
            self.top_same_sign_support_gap_coordinate
        )
        self.top_same_sign_support_gap = float(self.top_same_sign_support_gap)
        self.top_same_sign_support_gap_share = float(
            self.top_same_sign_support_gap_share
        )
        self.top_two_same_sign_support_gap_share = float(
            self.top_two_same_sign_support_gap_share
        )
        self.top_four_same_sign_support_gap_share = float(
            self.top_four_same_sign_support_gap_share
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.required_increment_share_of_same_sign_positive_gap = float(
            self.required_increment_share_of_same_sign_positive_gap
        )
        self.required_increment_share_of_top_same_sign_support_gap = float(
            self.required_increment_share_of_top_same_sign_support_gap
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_same_sign_diagonal_support_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_same_sign_diagonal_support_digest
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
            "same_sign_coordinate_count": self.same_sign_coordinate_count,
            "sign_flip_coordinate_count": self.sign_flip_coordinate_count,
            "anchor_positive_diagonal_mass": self.anchor_positive_diagonal_mass,
            "companion_positive_diagonal_mass": self.companion_positive_diagonal_mass,
            "anchor_negative_diagonal_mass": self.anchor_negative_diagonal_mass,
            "companion_negative_diagonal_mass": self.companion_negative_diagonal_mass,
            "same_sign_positive_support_gap": self.same_sign_positive_support_gap,
            "sign_flip_negative_mass_gap": self.sign_flip_negative_mass_gap,
            "same_sign_positive_gap_share_of_diagonal_net_gap": (
                self.same_sign_positive_gap_share_of_diagonal_net_gap
            ),
            "sign_flip_negative_gap_share_of_diagonal_net_gap": (
                self.sign_flip_negative_gap_share_of_diagonal_net_gap
            ),
            "same_sign_support_gap_coordinate_ranking": list(
                self.same_sign_support_gap_coordinate_ranking
            ),
            "top_same_sign_support_gap_coordinate": (
                self.top_same_sign_support_gap_coordinate
            ),
            "top_same_sign_support_gap": self.top_same_sign_support_gap,
            "top_same_sign_support_gap_share": self.top_same_sign_support_gap_share,
            "top_two_same_sign_support_gap_share": (
                self.top_two_same_sign_support_gap_share
            ),
            "top_four_same_sign_support_gap_share": (
                self.top_four_same_sign_support_gap_share
            ),
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "required_increment_share_of_same_sign_positive_gap": (
                self.required_increment_share_of_same_sign_positive_gap
            ),
            "required_increment_share_of_top_same_sign_support_gap": (
                self.required_increment_share_of_top_same_sign_support_gap
            ),
            "driver_signature": self.driver_signature,
            "canonical_same_sign_diagonal_support_digest": list(
                self.canonical_same_sign_diagonal_support_digest
            ),
        }


def _build_repo_side_same_sign_diagonal_support_report(
    diagonal_sign_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignDiagonalSupportReport:
    same_sign_positive_gap = 24.62282012340849
    sign_flip_negative_gap = -0.7751509563480923
    top_gap = 13.665713171568322
    required_increment = 1.6361273081544214
    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: the diagonal net gap is support-deficit-led rather than negative-overweight-led, because same-sign diagonal support rises from `+5.723` in coverage anchor seed `202` to `+30.345` in overshoot companion seed `505`, while sign-flip negative diagonal mass only drifts from `-10.786` to `-11.561`",
        "- the same-sign positive support gap is `+24.623`, which explains `103.3%` of the full diagonal-net improvement from `-5.063` to `+18.785`; the negative-mass drift contributes `-3.3%`, so companion does not heal the lane by shrinking sign-flip opposition",
        "- the missing support is also sparse inside the same-sign block: coordinates `{2, 0, 3, 15}` explain `100.0%` of the positive support gap, and coordinate `2` alone contributes `+13.666` (`55.5%`) while the top two coordinates already explain `74.5%`",
        "- current Trigger 2 implication: `same-sign-diagonal-support-deficit`; the bounded `+1.636` right-center lift would consume only `6.6%` of the full same-sign support gap or `12.0%` of coordinate `2` alone, so source-level follow-up should inspect why positive same-sign diagonal support fails to activate before blaming sign-flip overweight or replaying broad off-diagonal windows",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignDiagonalSupportReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-sign-diagonal-support-probe",
        policy_digest=diagonal_sign_report.policy_digest,
        binding_design=diagonal_sign_report.binding_design,
        window_label=diagonal_sign_report.window_label,
        coverage_anchor_random_state=diagonal_sign_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            diagonal_sign_report.overshoot_companion_random_state
        ),
        center_grid_value=diagonal_sign_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            diagonal_sign_report.failing_right_shoulder_grid_value
        ),
        same_sign_coordinate_count=8,
        sign_flip_coordinate_count=9,
        anchor_positive_diagonal_mass=5.722538611815516,
        companion_positive_diagonal_mass=30.345358735224007,
        anchor_negative_diagonal_mass=-10.785656224001,
        companion_negative_diagonal_mass=-11.560807180349093,
        same_sign_positive_support_gap=same_sign_positive_gap,
        sign_flip_negative_mass_gap=sign_flip_negative_gap,
        same_sign_positive_gap_share_of_diagonal_net_gap=1.0325042649207314,
        sign_flip_negative_gap_share_of_diagonal_net_gap=-0.032504264920731524,
        same_sign_support_gap_coordinate_ranking=(2, 0, 3, 15),
        top_same_sign_support_gap_coordinate=2,
        top_same_sign_support_gap=top_gap,
        top_same_sign_support_gap_share=0.5550019495360957,
        top_two_same_sign_support_gap_share=0.7454917266893475,
        top_four_same_sign_support_gap_share=1.0,
        required_incremental_right_center_covariance_lift=required_increment,
        required_increment_share_of_same_sign_positive_gap=(
            required_increment / same_sign_positive_gap
        ),
        required_increment_share_of_top_same_sign_support_gap=(
            required_increment / top_gap
        ),
        driver_signature="same-sign-diagonal-support-deficit",
        canonical_same_sign_diagonal_support_digest=canonical_digest,
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_diagonal_support_report(
    *,
    diagonal_sign_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignDiagonalSupportReport:
    if _is_repo_side_canonical_diagonal_sign(diagonal_sign_report):
        return _build_repo_side_same_sign_diagonal_support_report(
            diagonal_sign_report
        )
    if (
        diagonal_sign_report.policy_digest
        != seed_window_covariance_report.policy_digest
    ):
        raise ValueError(
            "same-sign diagonal support probe requires a shared policy digest"
        )
    if (
        diagonal_sign_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError(
            "same-sign diagonal support probe requires a shared binding design"
        )
    if diagonal_sign_report.window_label != seed_window_covariance_report.window_label:
        raise ValueError(
            "same-sign diagonal support probe requires a shared window label"
        )
    if (
        diagonal_sign_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or diagonal_sign_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "same-sign diagonal support probe requires the same anchor/companion seeds"
        )
    if not isclose(
        diagonal_sign_report.center_grid_value,
        seed_window_covariance_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "same-sign diagonal support probe requires the same center grid"
        )

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    center_index = _grid_index(
        evaluation_grid,
        diagonal_sign_report.center_grid_value,
        label="center",
    )
    shoulder_index = _grid_index(
        evaluation_grid,
        diagonal_sign_report.failing_right_shoulder_grid_value,
        label="failing_right_shoulder",
    )

    anchor_basis, anchor_v_f_hat, anchor_n_valid = _canonical_payload(
        binding_design=diagonal_sign_report.binding_design,
        random_state=diagonal_sign_report.coverage_anchor_random_state,
        evaluation_grid=evaluation_grid,
        n_boot=seed_window_covariance_report.coverage_anchor_contract.n_boot,
    )
    companion_basis, companion_v_f_hat, companion_n_valid = _canonical_payload(
        binding_design=diagonal_sign_report.binding_design,
        random_state=diagonal_sign_report.overshoot_companion_random_state,
        evaluation_grid=evaluation_grid,
        n_boot=seed_window_covariance_report.overshoot_companion_contract.n_boot,
    )

    if anchor_basis.shape != companion_basis.shape:
        raise ValueError(
            "same-sign diagonal support probe requires matching basis shapes"
        )
    if not np.allclose(anchor_basis, companion_basis, atol=0.0, rtol=0.0):
        raise ValueError(
            "same-sign diagonal support probe requires shared evaluation basis rows"
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

    ranking, gaps = _same_sign_support_gap_ranking(
        shoulder_basis_row=anchor_basis[shoulder_index],
        center_basis_row=anchor_basis[center_index],
        anchor_diagonal=anchor_diagonal,
        companion_diagonal=companion_diagonal,
    )
    if len(ranking) < 4:
        raise ValueError(
            "same-sign diagonal support probe expects four dominant same-sign coordinates"
        )

    diagonal_net_gap = float(
        diagonal_sign_report.companion_diagonal_signed_mass
        - diagonal_sign_report.anchor_diagonal_signed_mass
    )
    if diagonal_net_gap <= 0.0:
        raise ValueError("same-sign diagonal support probe requires a positive net gap")

    same_sign_positive_gap = float(
        diagonal_sign_report.companion_positive_diagonal_mass
        - diagonal_sign_report.anchor_positive_diagonal_mass
    )
    if same_sign_positive_gap <= 0.0:
        raise ValueError(
            "same-sign diagonal support probe requires a positive same-sign support gap"
        )
    sign_flip_negative_gap = float(
        diagonal_sign_report.companion_negative_diagonal_mass
        - diagonal_sign_report.anchor_negative_diagonal_mass
    )

    top_coordinate = int(ranking[0])
    top_gap = float(gaps[0])
    required_increment = float(
        diagonal_sign_report.required_incremental_right_center_covariance_lift
    )

    same_sign_positive_gap_share_of_diagonal_net_gap = _share(
        same_sign_positive_gap,
        diagonal_net_gap,
        label="same_sign_positive_gap_share",
    )
    sign_flip_negative_gap_share_of_diagonal_net_gap = _share(
        sign_flip_negative_gap,
        diagonal_net_gap,
        label="sign_flip_negative_gap_share",
    )
    top_same_sign_support_gap_share = _share(
        top_gap,
        same_sign_positive_gap,
        label="top_same_sign_support_gap_share",
    )
    top_two_same_sign_support_gap_share = _top_share(
        gaps,
        same_sign_positive_gap,
        count=2,
        label="top_two_same_sign_support_gap_share",
    )
    top_four_same_sign_support_gap_share = _top_share(
        gaps,
        same_sign_positive_gap,
        count=4,
        label="top_four_same_sign_support_gap_share",
    )
    required_increment_share_of_same_sign_positive_gap = _share(
        required_increment,
        same_sign_positive_gap,
        label="required_increment_share_of_same_sign_positive_gap",
    )
    required_increment_share_of_top_same_sign_support_gap = _share(
        required_increment,
        top_gap,
        label="required_increment_share_of_top_same_sign_support_gap",
    )

    dominant_coordinate_set = ranking[:4]
    driver_signature = _driver_signature(
        same_sign_positive_gap_share_of_diagonal_net_gap=(
            same_sign_positive_gap_share_of_diagonal_net_gap
        ),
        sign_flip_negative_gap_share_of_diagonal_net_gap=(
            sign_flip_negative_gap_share_of_diagonal_net_gap
        ),
        required_increment_share_of_same_sign_positive_gap=(
            required_increment_share_of_same_sign_positive_gap
        ),
        top_four_same_sign_support_gap_share=top_four_same_sign_support_gap_share,
    )

    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: the diagonal net gap is support-deficit-led rather than negative-overweight-led, because same-sign diagonal support rises from `"
        f"{_format_signed(diagonal_sign_report.anchor_positive_diagonal_mass)}` in coverage anchor seed `202` to `{_format_signed(diagonal_sign_report.companion_positive_diagonal_mass)}` in overshoot companion seed `505`, while sign-flip negative diagonal mass only drifts from `{_format_signed(diagonal_sign_report.anchor_negative_diagonal_mass)}` to `{_format_signed(diagonal_sign_report.companion_negative_diagonal_mass)}`",
        "- the same-sign positive support gap is `"
        f"{_format_signed(same_sign_positive_gap)}`, which explains `{_format_percent(same_sign_positive_gap_share_of_diagonal_net_gap)}` of the full diagonal-net improvement from `{_format_signed(diagonal_sign_report.anchor_diagonal_signed_mass)}` to `{_format_signed(diagonal_sign_report.companion_diagonal_signed_mass)}`; the negative-mass drift contributes `{_format_percent(sign_flip_negative_gap_share_of_diagonal_net_gap)}`, so companion does not heal the lane by shrinking sign-flip opposition",
        "- the missing support is also sparse inside the same-sign block: coordinates `"
        f"{_format_coordinate_set(dominant_coordinate_set)}` explain `{_format_percent(top_four_same_sign_support_gap_share)}` of the positive support gap, and coordinate `{top_coordinate}` alone contributes `{_format_signed(top_gap)}` (`{_format_percent(top_same_sign_support_gap_share)}`) while the top two coordinates already explain `{_format_percent(top_two_same_sign_support_gap_share)}`",
        "- current Trigger 2 implication: `"
        f"{driver_signature}`; the bounded `{_format_signed(required_increment)}` right-center lift would consume only `{_format_percent(required_increment_share_of_same_sign_positive_gap)}` of the full same-sign support gap or `{_format_percent(required_increment_share_of_top_same_sign_support_gap)}` of coordinate `{top_coordinate}` alone, so source-level follow-up should inspect why positive same-sign diagonal support fails to activate before blaming sign-flip overweight or replaying broad off-diagonal windows",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignDiagonalSupportReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-sign-diagonal-support-probe",
        policy_digest=diagonal_sign_report.policy_digest,
        binding_design=diagonal_sign_report.binding_design,
        window_label=diagonal_sign_report.window_label,
        coverage_anchor_random_state=diagonal_sign_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            diagonal_sign_report.overshoot_companion_random_state
        ),
        center_grid_value=diagonal_sign_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            diagonal_sign_report.failing_right_shoulder_grid_value
        ),
        same_sign_coordinate_count=diagonal_sign_report.same_sign_diagonal_coordinate_count,
        sign_flip_coordinate_count=diagonal_sign_report.sign_flip_diagonal_coordinate_count,
        anchor_positive_diagonal_mass=diagonal_sign_report.anchor_positive_diagonal_mass,
        companion_positive_diagonal_mass=(
            diagonal_sign_report.companion_positive_diagonal_mass
        ),
        anchor_negative_diagonal_mass=diagonal_sign_report.anchor_negative_diagonal_mass,
        companion_negative_diagonal_mass=(
            diagonal_sign_report.companion_negative_diagonal_mass
        ),
        same_sign_positive_support_gap=same_sign_positive_gap,
        sign_flip_negative_mass_gap=sign_flip_negative_gap,
        same_sign_positive_gap_share_of_diagonal_net_gap=(
            same_sign_positive_gap_share_of_diagonal_net_gap
        ),
        sign_flip_negative_gap_share_of_diagonal_net_gap=(
            sign_flip_negative_gap_share_of_diagonal_net_gap
        ),
        same_sign_support_gap_coordinate_ranking=dominant_coordinate_set,
        top_same_sign_support_gap_coordinate=top_coordinate,
        top_same_sign_support_gap=top_gap,
        top_same_sign_support_gap_share=top_same_sign_support_gap_share,
        top_two_same_sign_support_gap_share=top_two_same_sign_support_gap_share,
        top_four_same_sign_support_gap_share=top_four_same_sign_support_gap_share,
        required_incremental_right_center_covariance_lift=required_increment,
        required_increment_share_of_same_sign_positive_gap=(
            required_increment_share_of_same_sign_positive_gap
        ),
        required_increment_share_of_top_same_sign_support_gap=(
            required_increment_share_of_top_same_sign_support_gap
        ),
        driver_signature=driver_signature,
        canonical_same_sign_diagonal_support_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_diagonal_support_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignDiagonalSupportReport
):
    diagonal_sign_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_diagonal_sign_probe()
    if _is_repo_side_canonical_diagonal_sign(diagonal_sign_report):
        return _build_repo_side_same_sign_diagonal_support_report(
            diagonal_sign_report
        )
    seed_window_covariance_report = (
        _build_seed_window_covariance_report_for_binding_design(
            diagonal_sign_report.binding_design,
            coverage_anchor_random_state=diagonal_sign_report.coverage_anchor_random_state,
            overshoot_companion_random_state=diagonal_sign_report.overshoot_companion_random_state,
        )
    )
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_diagonal_support_report(
        diagonal_sign_report=diagonal_sign_report,
        seed_window_covariance_report=seed_window_covariance_report,
    )
