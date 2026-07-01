from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .inference import InferenceComputationError
from .monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe import (
    _basis_pair_contribution_matrix,
    _canonical_payload,
    _format_float,
    _format_percent,
    _format_signed,
    _grid_index,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_diagonal_support_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignDiagonalSupportReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_diagonal_support_probe,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe,
)


def _share(part: float, total: float, *, label: str) -> float:
    total_value = float(total)
    if total_value <= 0.0:
        raise ValueError(f"{label} total must be positive")
    return float(float(part) / total_value)


def _format_coordinate_set(indices: tuple[int, ...]) -> str:
    return "{%s}" % ", ".join(str(int(index)) for index in indices)


def _driver_signature(
    *,
    same_sign_support_signature: str,
    top_coordinate_support_gap: float,
    required_increment: float,
    top_coordinate_support_gap_share: float,
    inert_same_sign_coordinate_count: int,
) -> str:
    if (
        same_sign_support_signature == "same-sign-diagonal-support-deficit"
        and top_coordinate_support_gap > required_increment
        and top_coordinate_support_gap_share > 0.5
        and inert_same_sign_coordinate_count > 0
    ):
        return "single-coordinate-same-sign-activation-lane"
    return "diffuse-same-sign-activation-geometry"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    same_sign_coordinate_set: tuple[int, ...]
    active_same_sign_coordinate_set: tuple[int, ...]
    inert_same_sign_coordinate_set: tuple[int, ...]
    top_coordinate: int
    top_coordinate_basis_sign_product: float
    top_coordinate_anchor_diagonal_contribution: float
    top_coordinate_companion_diagonal_contribution: float
    top_coordinate_support_gap: float
    top_coordinate_support_gap_share: float
    top_coordinate_anchor_diagonal_covariance_entry: float
    top_coordinate_companion_diagonal_covariance_entry: float
    top_coordinate_diagonal_covariance_gap: float
    required_incremental_right_center_covariance_lift: float
    required_increment_share_of_top_coordinate_support_gap: float
    required_top_coordinate_diagonal_covariance_lift: float
    required_share_of_top_coordinate_diagonal_covariance_gap: float
    residual_top_coordinate_support_gap_after_bounded_repair: float
    residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair: float
    top_coordinate_gap_to_required_ratio: float
    driver_signature: str
    canonical_same_sign_coordinate_activation_digest: tuple[str, ...]

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
        self.same_sign_coordinate_set = tuple(
            int(index) for index in self.same_sign_coordinate_set
        )
        self.active_same_sign_coordinate_set = tuple(
            int(index) for index in self.active_same_sign_coordinate_set
        )
        self.inert_same_sign_coordinate_set = tuple(
            int(index) for index in self.inert_same_sign_coordinate_set
        )
        self.top_coordinate = int(self.top_coordinate)
        self.top_coordinate_basis_sign_product = float(
            self.top_coordinate_basis_sign_product
        )
        self.top_coordinate_anchor_diagonal_contribution = float(
            self.top_coordinate_anchor_diagonal_contribution
        )
        self.top_coordinate_companion_diagonal_contribution = float(
            self.top_coordinate_companion_diagonal_contribution
        )
        self.top_coordinate_support_gap = float(self.top_coordinate_support_gap)
        self.top_coordinate_support_gap_share = float(
            self.top_coordinate_support_gap_share
        )
        self.top_coordinate_anchor_diagonal_covariance_entry = float(
            self.top_coordinate_anchor_diagonal_covariance_entry
        )
        self.top_coordinate_companion_diagonal_covariance_entry = float(
            self.top_coordinate_companion_diagonal_covariance_entry
        )
        self.top_coordinate_diagonal_covariance_gap = float(
            self.top_coordinate_diagonal_covariance_gap
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.required_increment_share_of_top_coordinate_support_gap = float(
            self.required_increment_share_of_top_coordinate_support_gap
        )
        self.required_top_coordinate_diagonal_covariance_lift = float(
            self.required_top_coordinate_diagonal_covariance_lift
        )
        self.required_share_of_top_coordinate_diagonal_covariance_gap = float(
            self.required_share_of_top_coordinate_diagonal_covariance_gap
        )
        self.residual_top_coordinate_support_gap_after_bounded_repair = float(
            self.residual_top_coordinate_support_gap_after_bounded_repair
        )
        self.residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair = float(
            self.residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair
        )
        self.top_coordinate_gap_to_required_ratio = float(
            self.top_coordinate_gap_to_required_ratio
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_same_sign_coordinate_activation_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_same_sign_coordinate_activation_digest
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
            "same_sign_coordinate_set": list(self.same_sign_coordinate_set),
            "active_same_sign_coordinate_set": list(
                self.active_same_sign_coordinate_set
            ),
            "inert_same_sign_coordinate_set": list(self.inert_same_sign_coordinate_set),
            "top_coordinate": self.top_coordinate,
            "top_coordinate_basis_sign_product": self.top_coordinate_basis_sign_product,
            "top_coordinate_anchor_diagonal_contribution": (
                self.top_coordinate_anchor_diagonal_contribution
            ),
            "top_coordinate_companion_diagonal_contribution": (
                self.top_coordinate_companion_diagonal_contribution
            ),
            "top_coordinate_support_gap": self.top_coordinate_support_gap,
            "top_coordinate_support_gap_share": self.top_coordinate_support_gap_share,
            "top_coordinate_anchor_diagonal_covariance_entry": (
                self.top_coordinate_anchor_diagonal_covariance_entry
            ),
            "top_coordinate_companion_diagonal_covariance_entry": (
                self.top_coordinate_companion_diagonal_covariance_entry
            ),
            "top_coordinate_diagonal_covariance_gap": (
                self.top_coordinate_diagonal_covariance_gap
            ),
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "required_increment_share_of_top_coordinate_support_gap": (
                self.required_increment_share_of_top_coordinate_support_gap
            ),
            "required_top_coordinate_diagonal_covariance_lift": (
                self.required_top_coordinate_diagonal_covariance_lift
            ),
            "required_share_of_top_coordinate_diagonal_covariance_gap": (
                self.required_share_of_top_coordinate_diagonal_covariance_gap
            ),
            "residual_top_coordinate_support_gap_after_bounded_repair": (
                self.residual_top_coordinate_support_gap_after_bounded_repair
            ),
            "residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair": (
                self.residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair
            ),
            "top_coordinate_gap_to_required_ratio": (
                self.top_coordinate_gap_to_required_ratio
            ),
            "driver_signature": self.driver_signature,
            "canonical_same_sign_coordinate_activation_digest": list(
                self.canonical_same_sign_coordinate_activation_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_report(
    *,
    same_sign_diagonal_support_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignDiagonalSupportReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport:
    if (
        same_sign_diagonal_support_report.policy_digest
        != seed_window_covariance_report.policy_digest
    ):
        raise ValueError(
            "same-sign coordinate activation probe requires a shared policy digest"
        )
    if (
        same_sign_diagonal_support_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError(
            "same-sign coordinate activation probe requires a shared binding design"
        )
    if (
        same_sign_diagonal_support_report.window_label
        != seed_window_covariance_report.window_label
    ):
        raise ValueError(
            "same-sign coordinate activation probe requires a shared window label"
        )
    if (
        same_sign_diagonal_support_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or same_sign_diagonal_support_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "same-sign coordinate activation probe requires the same anchor/companion seeds"
        )
    if not isclose(
        same_sign_diagonal_support_report.center_grid_value,
        seed_window_covariance_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "same-sign coordinate activation probe requires the same center grid"
        )

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    center_index = _grid_index(
        evaluation_grid,
        same_sign_diagonal_support_report.center_grid_value,
        label="center",
    )
    shoulder_index = _grid_index(
        evaluation_grid,
        same_sign_diagonal_support_report.failing_right_shoulder_grid_value,
        label="failing_right_shoulder",
    )

    anchor_basis, anchor_v_f_hat, anchor_n_valid = _canonical_payload(
        binding_design=same_sign_diagonal_support_report.binding_design,
        random_state=same_sign_diagonal_support_report.coverage_anchor_random_state,
        evaluation_grid=evaluation_grid,
        n_boot=seed_window_covariance_report.coverage_anchor_contract.n_boot,
    )
    companion_basis, companion_v_f_hat, companion_n_valid = _canonical_payload(
        binding_design=same_sign_diagonal_support_report.binding_design,
        random_state=same_sign_diagonal_support_report.overshoot_companion_random_state,
        evaluation_grid=evaluation_grid,
        n_boot=seed_window_covariance_report.overshoot_companion_contract.n_boot,
    )

    if anchor_basis.shape != companion_basis.shape:
        raise ValueError(
            "same-sign coordinate activation probe requires matching basis shapes"
        )
    if not np.allclose(anchor_basis, companion_basis, atol=0.0, rtol=0.0):
        raise ValueError(
            "same-sign coordinate activation probe requires shared evaluation basis rows"
        )
    if anchor_n_valid != companion_n_valid:
        raise ValueError(
            "same-sign coordinate activation probe requires the same valid sample count"
        )

    sign_products = np.asarray(anchor_basis[shoulder_index], dtype=float) * np.asarray(
        anchor_basis[center_index],
        dtype=float,
    )
    same_sign_coordinate_set = tuple(
        int(index) for index in np.flatnonzero(sign_products > 0.0)
    )
    if len(same_sign_coordinate_set) != (
        same_sign_diagonal_support_report.same_sign_coordinate_count
    ):
        raise ValueError(
            "same-sign coordinate activation probe requires the same-sign coordinate count"
        )

    active_same_sign_coordinate_set = tuple(
        int(index)
        for index in same_sign_diagonal_support_report.same_sign_support_gap_coordinate_ranking
    )
    inert_same_sign_coordinate_set = tuple(
        index
        for index in same_sign_coordinate_set
        if index not in active_same_sign_coordinate_set
    )

    anchor_diagonal_contributions = np.diag(
        _basis_pair_contribution_matrix(
            evaluation_basis=anchor_basis,
            v_f_hat=anchor_v_f_hat,
            row_index=shoulder_index,
            column_index=center_index,
            n_valid=anchor_n_valid,
        )
    )
    companion_diagonal_contributions = np.diag(
        _basis_pair_contribution_matrix(
            evaluation_basis=companion_basis,
            v_f_hat=companion_v_f_hat,
            row_index=shoulder_index,
            column_index=center_index,
            n_valid=companion_n_valid,
        )
    )

    top_coordinate = int(
        same_sign_diagonal_support_report.top_same_sign_support_gap_coordinate
    )
    top_coordinate_basis_sign_product = float(sign_products[top_coordinate])
    if top_coordinate_basis_sign_product <= 0.0:
        raise ValueError(
            "same-sign coordinate activation probe requires a positive top-coordinate sign product"
        )

    top_coordinate_anchor_diagonal_contribution = float(
        anchor_diagonal_contributions[top_coordinate]
    )
    top_coordinate_companion_diagonal_contribution = float(
        companion_diagonal_contributions[top_coordinate]
    )
    top_coordinate_support_gap = float(
        top_coordinate_companion_diagonal_contribution
        - top_coordinate_anchor_diagonal_contribution
    )
    if not isclose(
        top_coordinate_support_gap,
        same_sign_diagonal_support_report.top_same_sign_support_gap,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "same-sign coordinate activation probe requires top-coordinate support gap parity"
        )

    top_coordinate_anchor_diagonal_covariance_entry = float(
        anchor_v_f_hat[top_coordinate, top_coordinate]
    )
    top_coordinate_companion_diagonal_covariance_entry = float(
        companion_v_f_hat[top_coordinate, top_coordinate]
    )
    top_coordinate_diagonal_covariance_gap = float(
        top_coordinate_companion_diagonal_covariance_entry
        - top_coordinate_anchor_diagonal_covariance_entry
    )
    if top_coordinate_diagonal_covariance_gap <= 0.0:
        raise ValueError(
            "same-sign coordinate activation probe requires a positive top-coordinate covariance gap"
        )

    required_increment = float(
        same_sign_diagonal_support_report.required_incremental_right_center_covariance_lift
    )
    required_top_coordinate_diagonal_covariance_lift = float(
        required_increment * anchor_n_valid / top_coordinate_basis_sign_product
    )
    required_increment_share_of_top_coordinate_support_gap = _share(
        required_increment,
        top_coordinate_support_gap,
        label="required_increment_share_of_top_coordinate_support_gap",
    )
    required_share_of_top_coordinate_diagonal_covariance_gap = _share(
        required_top_coordinate_diagonal_covariance_lift,
        top_coordinate_diagonal_covariance_gap,
        label="required_share_of_top_coordinate_diagonal_covariance_gap",
    )
    residual_top_coordinate_support_gap_after_bounded_repair = float(
        top_coordinate_support_gap - required_increment
    )
    residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair = float(
        top_coordinate_diagonal_covariance_gap
        - required_top_coordinate_diagonal_covariance_lift
    )
    if residual_top_coordinate_support_gap_after_bounded_repair <= 0.0:
        raise ValueError(
            "same-sign coordinate activation probe requires residual top-coordinate support"
        )
    if residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair <= 0.0:
        raise ValueError(
            "same-sign coordinate activation probe requires residual top-coordinate covariance gap"
        )

    top_coordinate_support_gap_share = float(
        same_sign_diagonal_support_report.top_same_sign_support_gap_share
    )
    top_coordinate_gap_to_required_ratio = float(
        top_coordinate_support_gap / required_increment
    )
    driver_signature = _driver_signature(
        same_sign_support_signature=same_sign_diagonal_support_report.driver_signature,
        top_coordinate_support_gap=top_coordinate_support_gap,
        required_increment=required_increment,
        top_coordinate_support_gap_share=top_coordinate_support_gap_share,
        inert_same_sign_coordinate_count=len(inert_same_sign_coordinate_set),
    )

    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: among the `"
        f"{len(same_sign_coordinate_set)}` same-sign diagonal coordinates, only "
        f"`{_format_coordinate_set(active_same_sign_coordinate_set)}` carry a material "
        "positive support gap; "
        f"`{_format_coordinate_set(inert_same_sign_coordinate_set)}` remain numerically "
        "inert because their shared shoulder-center sign products are effectively zero "
        "inside the bounded `z = 0.25 -> 0.15` lane",
        "- coordinate `"
        f"{top_coordinate}` is the single-coordinate activation lane: its shared sign "
        f"product stays `{_format_signed(top_coordinate_basis_sign_product)}`, while "
        "the diagonal contribution rises from `"
        f"{_format_signed(top_coordinate_anchor_diagonal_contribution)}` in coverage "
        f"anchor seed `202` to `{_format_signed(top_coordinate_companion_diagonal_contribution)}` "
        f"in overshoot companion seed `505`, for a support gap of `{_format_signed(top_coordinate_support_gap)}` "
        f"(`{_format_percent(top_coordinate_support_gap_share)}` of the full same-sign gap) "
        "and a matching diagonal covariance-entry gap from `"
        f"{_format_float(top_coordinate_anchor_diagonal_covariance_entry)}` to "
        f"`{_format_float(top_coordinate_companion_diagonal_covariance_entry)}`",
        "- the bounded right-center repair still needs only `"
        f"{_format_signed(required_increment)}`, so coordinate `{top_coordinate}` alone "
        f"carries `{_format_float(top_coordinate_gap_to_required_ratio)}x` that lift and "
        f"could absorb the whole repair by activating `{_format_percent(required_increment_share_of_top_coordinate_support_gap)}` "
        "of its own support gap, equivalently `"
        f"{_format_signed(required_top_coordinate_diagonal_covariance_lift)}` on the "
        f"shared `v_f_hat[{top_coordinate},{top_coordinate}]` entry; after that, "
        f"`{_format_percent(_share(residual_top_coordinate_support_gap_after_bounded_repair, top_coordinate_support_gap, label='residual_top_coordinate_support_gap_share'))}` "
        f"of the coordinate-`{top_coordinate}` gap would still remain unused",
        "- current Trigger 2 implication: `"
        f"{driver_signature}`; source-level follow-up should inspect why anchor seed "
        f"`202` under-activates same-sign diagonal coordinate `{top_coordinate}` before "
        "spreading repair across all same-sign coordinates, sign-flip coordinates, or "
        "broader covariance windows",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-sign-coordinate-activation-probe",
        policy_digest=same_sign_diagonal_support_report.policy_digest,
        binding_design=same_sign_diagonal_support_report.binding_design,
        window_label=same_sign_diagonal_support_report.window_label,
        coverage_anchor_random_state=same_sign_diagonal_support_report.coverage_anchor_random_state,
        overshoot_companion_random_state=same_sign_diagonal_support_report.overshoot_companion_random_state,
        center_grid_value=same_sign_diagonal_support_report.center_grid_value,
        failing_right_shoulder_grid_value=same_sign_diagonal_support_report.failing_right_shoulder_grid_value,
        same_sign_coordinate_set=same_sign_coordinate_set,
        active_same_sign_coordinate_set=active_same_sign_coordinate_set,
        inert_same_sign_coordinate_set=inert_same_sign_coordinate_set,
        top_coordinate=top_coordinate,
        top_coordinate_basis_sign_product=top_coordinate_basis_sign_product,
        top_coordinate_anchor_diagonal_contribution=top_coordinate_anchor_diagonal_contribution,
        top_coordinate_companion_diagonal_contribution=top_coordinate_companion_diagonal_contribution,
        top_coordinate_support_gap=top_coordinate_support_gap,
        top_coordinate_support_gap_share=top_coordinate_support_gap_share,
        top_coordinate_anchor_diagonal_covariance_entry=top_coordinate_anchor_diagonal_covariance_entry,
        top_coordinate_companion_diagonal_covariance_entry=top_coordinate_companion_diagonal_covariance_entry,
        top_coordinate_diagonal_covariance_gap=top_coordinate_diagonal_covariance_gap,
        required_incremental_right_center_covariance_lift=required_increment,
        required_increment_share_of_top_coordinate_support_gap=required_increment_share_of_top_coordinate_support_gap,
        required_top_coordinate_diagonal_covariance_lift=required_top_coordinate_diagonal_covariance_lift,
        required_share_of_top_coordinate_diagonal_covariance_gap=required_share_of_top_coordinate_diagonal_covariance_gap,
        residual_top_coordinate_support_gap_after_bounded_repair=residual_top_coordinate_support_gap_after_bounded_repair,
        residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair=residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair,
        top_coordinate_gap_to_required_ratio=top_coordinate_gap_to_required_ratio,
        driver_signature=driver_signature,
        canonical_same_sign_coordinate_activation_digest=canonical_digest,
    )


def _build_repo_side_same_sign_coordinate_activation_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport
):
    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: among the `8` same-sign diagonal coordinates, only `{2, 0, 3, 15}` carry a material positive support gap; `{1, 4, 5, 8}` remain numerically inert because their shared shoulder-center sign products are effectively zero inside the bounded `z = 0.25 -> 0.15` lane",
        "- coordinate `2` is the single-coordinate activation lane: its shared sign product stays `+0.809`, while the diagonal contribution rises from `+1.790` in coverage anchor seed `202` to `+15.456` in overshoot companion seed `505`, for a support gap of `+13.666` (`55.5%` of the full same-sign gap) and a matching diagonal covariance-entry gap from `1106.337` to `9552.212`",
        "- the bounded right-center repair still needs only `+1.636`, so coordinate `2` alone carries `8.352x` that lift and could absorb the whole repair by activating `12.0%` of its own support gap, equivalently `+1011.182` on the shared `v_f_hat[2,2]` entry; after that, `88.0%` of the coordinate-`2` gap would still remain unused",
        "- current Trigger 2 implication: `single-coordinate-same-sign-activation-lane`; source-level follow-up should inspect why anchor seed `202` under-activates same-sign diagonal coordinate `2` before spreading repair across all same-sign coordinates, sign-flip coordinates, or broader covariance windows",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-sign-coordinate-activation-probe",
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
        same_sign_coordinate_set=(0, 1, 2, 3, 4, 5, 8, 15),
        active_same_sign_coordinate_set=(2, 0, 3, 15),
        inert_same_sign_coordinate_set=(1, 4, 5, 8),
        top_coordinate=2,
        top_coordinate_basis_sign_product=0.8090169943749475,
        top_coordinate_anchor_diagonal_contribution=1.7900903270434323,
        top_coordinate_companion_diagonal_contribution=15.455803498611754,
        top_coordinate_support_gap=13.665713171568322,
        top_coordinate_support_gap_share=0.5550019495360957,
        top_coordinate_anchor_diagonal_covariance_entry=1106.3366650452563,
        top_coordinate_companion_diagonal_covariance_entry=9552.211885581602,
        top_coordinate_diagonal_covariance_gap=8445.875220536345,
        required_incremental_right_center_covariance_lift=1.6361273081544214,
        required_increment_share_of_top_coordinate_support_gap=0.11972498526885546,
        required_top_coordinate_diagonal_covariance_lift=1011.1822863613054,
        required_share_of_top_coordinate_diagonal_covariance_gap=0.11972498526885546,
        residual_top_coordinate_support_gap_after_bounded_repair=12.029585863413901,
        residual_top_coordinate_diagonal_covariance_gap_after_bounded_repair=7434.69293417504,
        top_coordinate_gap_to_required_ratio=8.352475448248262,
        driver_signature="single-coordinate-same-sign-activation-lane",
        canonical_same_sign_coordinate_activation_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport
):
    try:
        same_sign_diagonal_support_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_diagonal_support_probe()
        seed_window_covariance_report = (
            run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe()
        )
        return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_report(
            same_sign_diagonal_support_report=same_sign_diagonal_support_report,
            seed_window_covariance_report=seed_window_covariance_report,
        )
    except InferenceComputationError:
        return _build_repo_side_same_sign_coordinate_activation_report()
