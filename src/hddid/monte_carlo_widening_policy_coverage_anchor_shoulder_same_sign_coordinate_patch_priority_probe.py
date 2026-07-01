from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe import (
    _basis_pair_contribution_matrix,
    _canonical_payload,
    _grid_index,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_entry_factor_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_entry_factor_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_basis_term_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateBasisTermReport,
    _coordinate_basis_metadata,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_basis_term_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_diagonal_support_probe import (
    _same_sign_support_gap_ranking,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe,
)

_CANONICAL_REQUIRED_PATCH_SHARES = (
    0.21290057041569158,
    0.2498513058970336,
    0.4632363706499719,
    1.1985980371302667,
)
_CANONICAL_REQUIRED_VF_ENTRY_LIFTS = (
    1011.1822863613054,
    818.0636540772107,
    2647.3095945157274,
    2647.3095945157243,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _driver_signature(
    *,
    active_coordinate_priority_ranking: tuple[int, ...],
    active_coordinate_labels: tuple[str, ...],
    top_coordinate: int,
    top_coordinate_basis_label: str,
    top_coordinate_required_patch_share: float,
    intercept_required_patch_share: float,
    tail_share_multiple_relative_to_top: float,
) -> str:
    if (
        active_coordinate_priority_ranking == (2, 0, 3, 15)
        and active_coordinate_labels == ("sin(2πz)", "1", "cos(4πz)", "cos(16πz)")
        and top_coordinate == 2
        and top_coordinate_basis_label == "sin(2πz)"
        and top_coordinate_required_patch_share < intercept_required_patch_share
        and tail_share_multiple_relative_to_top > 5.0
    ):
        return "single-coordinate-first-sine-patch-priority"
    return "diffuse-same-sign-patch-priority"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinatePatchPriorityReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    active_coordinate_priority_ranking: tuple[int, ...]
    active_coordinate_labels: tuple[str, ...]
    active_coordinate_support_gaps: tuple[float, ...]
    active_coordinate_required_patch_shares: tuple[float, ...]
    active_coordinate_required_vf_entry_lifts: tuple[float, ...]
    top_coordinate: int
    top_coordinate_basis_label: str
    top_coordinate_required_patch_share: float
    intercept_coordinate: int
    intercept_basis_label: str
    intercept_required_patch_share: float
    intercept_share_multiple_relative_to_top: float
    third_coordinate: int
    third_basis_label: str
    third_share_multiple_relative_to_top: float
    tail_coordinate: int
    tail_basis_label: str
    tail_share_multiple_relative_to_top: float
    driver_signature: str
    canonical_patch_priority_digest: tuple[str, ...]

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
        self.active_coordinate_priority_ranking = tuple(
            int(index) for index in self.active_coordinate_priority_ranking
        )
        self.active_coordinate_labels = tuple(
            str(label).strip() for label in self.active_coordinate_labels
        )
        self.active_coordinate_support_gaps = tuple(
            float(value) for value in self.active_coordinate_support_gaps
        )
        self.active_coordinate_required_patch_shares = tuple(
            float(value) for value in self.active_coordinate_required_patch_shares
        )
        self.active_coordinate_required_vf_entry_lifts = tuple(
            float(value) for value in self.active_coordinate_required_vf_entry_lifts
        )
        self.top_coordinate = int(self.top_coordinate)
        self.top_coordinate_basis_label = str(self.top_coordinate_basis_label).strip()
        self.top_coordinate_required_patch_share = float(
            self.top_coordinate_required_patch_share
        )
        self.intercept_coordinate = int(self.intercept_coordinate)
        self.intercept_basis_label = str(self.intercept_basis_label).strip()
        self.intercept_required_patch_share = float(self.intercept_required_patch_share)
        self.intercept_share_multiple_relative_to_top = float(
            self.intercept_share_multiple_relative_to_top
        )
        self.third_coordinate = int(self.third_coordinate)
        self.third_basis_label = str(self.third_basis_label).strip()
        self.third_share_multiple_relative_to_top = float(
            self.third_share_multiple_relative_to_top
        )
        self.tail_coordinate = int(self.tail_coordinate)
        self.tail_basis_label = str(self.tail_basis_label).strip()
        self.tail_share_multiple_relative_to_top = float(
            self.tail_share_multiple_relative_to_top
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_patch_priority_digest = tuple(
            str(line).rstrip() for line in self.canonical_patch_priority_digest
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
            "active_coordinate_priority_ranking": list(
                self.active_coordinate_priority_ranking
            ),
            "active_coordinate_labels": list(self.active_coordinate_labels),
            "active_coordinate_support_gaps": list(self.active_coordinate_support_gaps),
            "active_coordinate_required_patch_shares": list(
                self.active_coordinate_required_patch_shares
            ),
            "active_coordinate_required_vf_entry_lifts": list(
                self.active_coordinate_required_vf_entry_lifts
            ),
            "top_coordinate": self.top_coordinate,
            "top_coordinate_basis_label": self.top_coordinate_basis_label,
            "top_coordinate_required_patch_share": (
                self.top_coordinate_required_patch_share
            ),
            "intercept_coordinate": self.intercept_coordinate,
            "intercept_basis_label": self.intercept_basis_label,
            "intercept_required_patch_share": self.intercept_required_patch_share,
            "intercept_share_multiple_relative_to_top": (
                self.intercept_share_multiple_relative_to_top
            ),
            "third_coordinate": self.third_coordinate,
            "third_basis_label": self.third_basis_label,
            "third_share_multiple_relative_to_top": (
                self.third_share_multiple_relative_to_top
            ),
            "tail_coordinate": self.tail_coordinate,
            "tail_basis_label": self.tail_basis_label,
            "tail_share_multiple_relative_to_top": (
                self.tail_share_multiple_relative_to_top
            ),
            "driver_signature": self.driver_signature,
            "canonical_patch_priority_digest": list(
                self.canonical_patch_priority_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_patch_priority_report(
    *,
    activation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport
    ),
    basis_term_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateBasisTermReport
    ),
    diagonal_entry_factor_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineDiagonalEntryFactorReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinatePatchPriorityReport:
    if activation_report.policy_digest != basis_term_report.policy_digest:
        raise ValueError("patch priority probe requires shared policy digest")
    if activation_report.binding_design != basis_term_report.binding_design:
        raise ValueError("patch priority probe requires shared binding design")
    if activation_report.window_label != basis_term_report.window_label:
        raise ValueError("patch priority probe requires shared window label")
    if (
        activation_report.coverage_anchor_random_state
        != basis_term_report.coverage_anchor_random_state
        or activation_report.overshoot_companion_random_state
        != basis_term_report.overshoot_companion_random_state
    ):
        raise ValueError("patch priority probe requires shared anchor/companion seeds")
    if activation_report.policy_digest != diagonal_entry_factor_report.policy_digest:
        raise ValueError("patch priority probe requires diagonal factor policy sync")
    if activation_report.binding_design != diagonal_entry_factor_report.binding_design:
        raise ValueError("patch priority probe requires diagonal factor design sync")
    if activation_report.window_label != diagonal_entry_factor_report.window_label:
        raise ValueError("patch priority probe requires diagonal factor window sync")
    if (
        activation_report.coverage_anchor_random_state
        != diagonal_entry_factor_report.coverage_anchor_random_state
        or activation_report.overshoot_companion_random_state
        != diagonal_entry_factor_report.overshoot_companion_random_state
    ):
        raise ValueError("patch priority probe requires diagonal factor seed sync")
    if activation_report.policy_digest != seed_window_covariance_report.policy_digest:
        raise ValueError("patch priority probe requires seed-window policy sync")
    if activation_report.binding_design != seed_window_covariance_report.binding_design:
        raise ValueError("patch priority probe requires seed-window design sync")
    if activation_report.window_label != seed_window_covariance_report.window_label:
        raise ValueError("patch priority probe requires seed-window window sync")

    anchor_basis, anchor_vf_hat, anchor_n_valid = _canonical_payload(
        binding_design=seed_window_covariance_report.binding_design,
        random_state=seed_window_covariance_report.coverage_anchor_random_state,
        evaluation_grid=seed_window_covariance_report.evaluation_grid,
        n_boot=seed_window_covariance_report.coverage_anchor_contract.n_boot,
    )
    companion_basis, companion_vf_hat, companion_n_valid = _canonical_payload(
        binding_design=seed_window_covariance_report.binding_design,
        random_state=seed_window_covariance_report.overshoot_companion_random_state,
        evaluation_grid=seed_window_covariance_report.evaluation_grid,
        n_boot=seed_window_covariance_report.overshoot_companion_contract.n_boot,
    )

    if anchor_n_valid != companion_n_valid:
        raise ValueError("patch priority probe requires matched valid sample counts")

    shoulder_index = _grid_index(
        seed_window_covariance_report.evaluation_grid,
        activation_report.failing_right_shoulder_grid_value,
        label="failing right shoulder",
    )
    center_index = _grid_index(
        seed_window_covariance_report.evaluation_grid,
        activation_report.center_grid_value,
        label="center",
    )
    anchor_contribution_matrix = _basis_pair_contribution_matrix(
        evaluation_basis=anchor_basis,
        v_f_hat=anchor_vf_hat,
        row_index=shoulder_index,
        column_index=center_index,
        n_valid=anchor_n_valid,
    )
    companion_contribution_matrix = _basis_pair_contribution_matrix(
        evaluation_basis=companion_basis,
        v_f_hat=companion_vf_hat,
        row_index=shoulder_index,
        column_index=center_index,
        n_valid=companion_n_valid,
    )
    ranking, support_gaps = _same_sign_support_gap_ranking(
        shoulder_basis_row=anchor_basis[shoulder_index],
        center_basis_row=anchor_basis[center_index],
        anchor_diagonal=np.diag(anchor_contribution_matrix),
        companion_diagonal=np.diag(companion_contribution_matrix),
    )
    active_coordinate_priority_ranking = tuple(ranking[:4])
    active_coordinate_support_gaps = tuple(float(value) for value in support_gaps[:4])
    if (
        active_coordinate_priority_ranking
        != activation_report.active_same_sign_coordinate_set
    ):
        raise ValueError("patch priority probe requires the active same-sign ranking")

    active_coordinate_labels = tuple(
        _coordinate_basis_metadata(index)[0]
        for index in active_coordinate_priority_ranking
    )
    sign_products = tuple(
        float(anchor_basis[shoulder_index, index] * anchor_basis[center_index, index])
        for index in active_coordinate_priority_ranking
    )
    if active_coordinate_labels != basis_term_report.active_coordinate_labels:
        raise ValueError("patch priority probe requires basis-term active label parity")

    derived_required_patch_shares = tuple(
        float(activation_report.required_incremental_right_center_covariance_lift / gap)
        for gap in active_coordinate_support_gaps
    )
    derived_required_vf_entry_lifts = tuple(
        float(
            activation_report.required_incremental_right_center_covariance_lift
            * anchor_n_valid
            / sign_product
        )
        for sign_product in sign_products
    )
    if not np.allclose(
        derived_required_patch_shares,
        _CANONICAL_REQUIRED_PATCH_SHARES,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("patch priority shares drifted away from the canonical lane")
    if not np.allclose(
        derived_required_vf_entry_lifts,
        _CANONICAL_REQUIRED_VF_ENTRY_LIFTS,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("patch priority vf-entry lifts drifted away from canonical")

    top_coordinate_required_patch_share = _CANONICAL_REQUIRED_PATCH_SHARES[0]
    intercept_required_patch_share = _CANONICAL_REQUIRED_PATCH_SHARES[1]
    third_share_multiple_relative_to_top = float(
        _CANONICAL_REQUIRED_PATCH_SHARES[2] / top_coordinate_required_patch_share
    )
    tail_share_multiple_relative_to_top = float(
        _CANONICAL_REQUIRED_PATCH_SHARES[3] / top_coordinate_required_patch_share
    )
    intercept_share_multiple_relative_to_top = float(
        intercept_required_patch_share / top_coordinate_required_patch_share
    )

    driver_signature = _driver_signature(
        active_coordinate_priority_ranking=active_coordinate_priority_ranking,
        active_coordinate_labels=active_coordinate_labels,
        top_coordinate=active_coordinate_priority_ranking[0],
        top_coordinate_basis_label=active_coordinate_labels[0],
        top_coordinate_required_patch_share=top_coordinate_required_patch_share,
        intercept_required_patch_share=intercept_required_patch_share,
        tail_share_multiple_relative_to_top=tail_share_multiple_relative_to_top,
    )

    canonical_digest = (
        "- active same-sign patch ranking on `DGP2/500/50` / `near_zero_grid` stays `{2, 0, 3, 15}` = `{sin(2πz), 1, cos(4πz), cos(16πz)}` once the bounded `+1.636` right-center repair is normalized by each coordinate's own support gap",
        "- coordinate `2` (`sin(2πz)`) remains the minimal-share lane: it needs only `21.3%` of its own `+7.685` support gap, equivalent to `+1011.182` on shared `v_f_hat[2,2]`; the intercept coordinate `0` would already need `25.0%` of its `+6.548` gap even though its geometric sign product is larger (`+1.000` vs `+0.809`)",
        "- the higher-harmonic fallback lanes are materially less efficient: coordinate `3` (`cos(4πz)`) would need `46.3%` of its own gap and coordinate `15` (`cos(16πz)`) would need `119.9%`; relative to coordinate `2`, those shares are `x2.176` and `x5.630`, so the bounded repair should not start by rebalancing higher harmonics",
        "- current Trigger 2 implication: `single-coordinate-first-sine-patch-priority`; source-level follow-up should patch / explain coordinate `2` before reallocating intercept mass, broad same-sign support, or wider covariance windows",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinatePatchPriorityReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-sign-coordinate-patch-priority-probe",
        policy_digest=activation_report.policy_digest,
        binding_design=activation_report.binding_design,
        window_label=activation_report.window_label,
        coverage_anchor_random_state=activation_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            activation_report.overshoot_companion_random_state
        ),
        center_grid_value=activation_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            activation_report.failing_right_shoulder_grid_value
        ),
        active_coordinate_priority_ranking=active_coordinate_priority_ranking,
        active_coordinate_labels=active_coordinate_labels,
        active_coordinate_support_gaps=active_coordinate_support_gaps,
        active_coordinate_required_patch_shares=(_CANONICAL_REQUIRED_PATCH_SHARES),
        active_coordinate_required_vf_entry_lifts=(_CANONICAL_REQUIRED_VF_ENTRY_LIFTS),
        top_coordinate=active_coordinate_priority_ranking[0],
        top_coordinate_basis_label=active_coordinate_labels[0],
        top_coordinate_required_patch_share=top_coordinate_required_patch_share,
        intercept_coordinate=active_coordinate_priority_ranking[1],
        intercept_basis_label=active_coordinate_labels[1],
        intercept_required_patch_share=intercept_required_patch_share,
        intercept_share_multiple_relative_to_top=(
            intercept_share_multiple_relative_to_top
        ),
        third_coordinate=active_coordinate_priority_ranking[2],
        third_basis_label=active_coordinate_labels[2],
        third_share_multiple_relative_to_top=third_share_multiple_relative_to_top,
        tail_coordinate=active_coordinate_priority_ranking[3],
        tail_basis_label=active_coordinate_labels[3],
        tail_share_multiple_relative_to_top=tail_share_multiple_relative_to_top,
        driver_signature=driver_signature,
        canonical_patch_priority_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_patch_priority_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinatePatchPriorityReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_patch_priority_report(
        activation_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe(),
        basis_term_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_basis_term_probe(),
        diagonal_entry_factor_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_diagonal_entry_factor_probe(),
        seed_window_covariance_report=run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe(),
    )
