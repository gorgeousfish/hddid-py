from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .basis import trigonometric_sieve_basis
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe,
)

_BASIS_FAMILY = "trigonometric"
_MONTE_CARLO_TRIG_DEGREE = 8


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_label_set(labels: tuple[str, ...]) -> str:
    return "{%s}" % ", ".join(labels)


def _coordinate_basis_metadata(index: int) -> tuple[str, str, int]:
    coordinate = int(index)
    if coordinate < 0:
        raise ValueError("coordinate index must be non-negative")
    if coordinate == 0:
        return ("1", "intercept", 0)

    shifted = coordinate - 1
    harmonic = shifted // 2 + 1
    if shifted % 2 == 0:
        return (f"cos({2 * harmonic}πz)", "cosine", harmonic)
    return (f"sin({2 * harmonic}πz)", "sine", harmonic)


def _driver_signature(
    *,
    activation_signature: str,
    top_coordinate_family: str,
    intercept_basis_sign_product: float,
    top_coordinate_basis_sign_product: float,
    max_inert_right_shoulder_abs_basis_value: float,
    required_increment_share_of_top_coordinate_support_gap: float,
) -> str:
    if (
        activation_signature == "single-coordinate-same-sign-activation-lane"
        and top_coordinate_family == "sine"
        and intercept_basis_sign_product > top_coordinate_basis_sign_product > 0.0
        and max_inert_right_shoulder_abs_basis_value <= 1e-12
        and required_increment_share_of_top_coordinate_support_gap < 0.2
    ):
        return "first-sine-harmonic-activation-lane"
    return "mixed-coordinate-basis-term-lane"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateBasisTermReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    basis_family: str
    basis_degree: int
    basis_dimension: int
    top_coordinate: int
    top_coordinate_basis_label: str
    top_coordinate_family: str
    top_coordinate_harmonic: int
    top_coordinate_right_shoulder_basis_value: float
    top_coordinate_center_basis_value: float
    top_coordinate_basis_sign_product: float
    intercept_coordinate: int
    intercept_basis_label: str
    intercept_basis_sign_product: float
    active_coordinate_labels: tuple[str, ...]
    inert_coordinate_labels: tuple[str, ...]
    max_inert_right_shoulder_abs_basis_value: float
    top_coordinate_support_gap: float
    top_coordinate_support_gap_share: float
    required_incremental_right_center_covariance_lift: float
    required_increment_share_of_top_coordinate_support_gap: float
    top_coordinate_gap_to_required_ratio: float
    driver_signature: str
    canonical_basis_term_digest: tuple[str, ...]

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
        self.basis_family = str(self.basis_family).strip()
        self.basis_degree = int(self.basis_degree)
        self.basis_dimension = int(self.basis_dimension)
        self.top_coordinate = int(self.top_coordinate)
        self.top_coordinate_basis_label = str(self.top_coordinate_basis_label).strip()
        self.top_coordinate_family = str(self.top_coordinate_family).strip()
        self.top_coordinate_harmonic = int(self.top_coordinate_harmonic)
        self.top_coordinate_right_shoulder_basis_value = float(
            self.top_coordinate_right_shoulder_basis_value
        )
        self.top_coordinate_center_basis_value = float(
            self.top_coordinate_center_basis_value
        )
        self.top_coordinate_basis_sign_product = float(
            self.top_coordinate_basis_sign_product
        )
        self.intercept_coordinate = int(self.intercept_coordinate)
        self.intercept_basis_label = str(self.intercept_basis_label).strip()
        self.intercept_basis_sign_product = float(self.intercept_basis_sign_product)
        self.active_coordinate_labels = tuple(
            str(label).strip() for label in self.active_coordinate_labels
        )
        self.inert_coordinate_labels = tuple(
            str(label).strip() for label in self.inert_coordinate_labels
        )
        self.max_inert_right_shoulder_abs_basis_value = float(
            self.max_inert_right_shoulder_abs_basis_value
        )
        self.top_coordinate_support_gap = float(self.top_coordinate_support_gap)
        self.top_coordinate_support_gap_share = float(
            self.top_coordinate_support_gap_share
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.required_increment_share_of_top_coordinate_support_gap = float(
            self.required_increment_share_of_top_coordinate_support_gap
        )
        self.top_coordinate_gap_to_required_ratio = float(
            self.top_coordinate_gap_to_required_ratio
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_basis_term_digest = tuple(
            str(line).rstrip() for line in self.canonical_basis_term_digest
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
            "basis_family": self.basis_family,
            "basis_degree": self.basis_degree,
            "basis_dimension": self.basis_dimension,
            "top_coordinate": self.top_coordinate,
            "top_coordinate_basis_label": self.top_coordinate_basis_label,
            "top_coordinate_family": self.top_coordinate_family,
            "top_coordinate_harmonic": self.top_coordinate_harmonic,
            "top_coordinate_right_shoulder_basis_value": (
                self.top_coordinate_right_shoulder_basis_value
            ),
            "top_coordinate_center_basis_value": (
                self.top_coordinate_center_basis_value
            ),
            "top_coordinate_basis_sign_product": (
                self.top_coordinate_basis_sign_product
            ),
            "intercept_coordinate": self.intercept_coordinate,
            "intercept_basis_label": self.intercept_basis_label,
            "intercept_basis_sign_product": self.intercept_basis_sign_product,
            "active_coordinate_labels": list(self.active_coordinate_labels),
            "inert_coordinate_labels": list(self.inert_coordinate_labels),
            "max_inert_right_shoulder_abs_basis_value": (
                self.max_inert_right_shoulder_abs_basis_value
            ),
            "top_coordinate_support_gap": self.top_coordinate_support_gap,
            "top_coordinate_support_gap_share": self.top_coordinate_support_gap_share,
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "required_increment_share_of_top_coordinate_support_gap": (
                self.required_increment_share_of_top_coordinate_support_gap
            ),
            "top_coordinate_gap_to_required_ratio": (
                self.top_coordinate_gap_to_required_ratio
            ),
            "driver_signature": self.driver_signature,
            "canonical_basis_term_digest": list(self.canonical_basis_term_digest),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_basis_term_report(
    *,
    activation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateBasisTermReport:
    basis_rows = trigonometric_sieve_basis(
        np.asarray(
            [
                activation_report.failing_right_shoulder_grid_value,
                activation_report.center_grid_value,
            ],
            dtype=float,
        ),
        degree=_MONTE_CARLO_TRIG_DEGREE,
    )
    basis_dimension = int(basis_rows.shape[1])
    if basis_dimension <= max(activation_report.same_sign_coordinate_set):
        raise ValueError(
            "basis term probe requires basis rows to cover all coordinates"
        )

    top_coordinate = activation_report.top_coordinate
    top_label, top_family, top_harmonic = _coordinate_basis_metadata(top_coordinate)
    top_right_value = float(basis_rows[0, top_coordinate])
    top_center_value = float(basis_rows[1, top_coordinate])
    top_sign_product = float(top_right_value * top_center_value)
    if not isclose(
        top_sign_product,
        activation_report.top_coordinate_basis_sign_product,
        abs_tol=1e-12,
    ):
        raise ValueError("basis term probe requires top-coordinate sign-product parity")

    intercept_coordinate = 0
    intercept_label, _, _ = _coordinate_basis_metadata(intercept_coordinate)
    intercept_sign_product = float(
        basis_rows[0, intercept_coordinate] * basis_rows[1, intercept_coordinate]
    )

    active_coordinate_labels = tuple(
        _coordinate_basis_metadata(index)[0]
        for index in activation_report.active_same_sign_coordinate_set
    )
    inert_coordinate_labels = tuple(
        _coordinate_basis_metadata(index)[0]
        for index in activation_report.inert_same_sign_coordinate_set
    )
    max_inert_right_shoulder_abs_basis_value = float(
        max(
            abs(float(basis_rows[0, index]))
            for index in activation_report.inert_same_sign_coordinate_set
        )
    )

    driver_signature = _driver_signature(
        activation_signature=activation_report.driver_signature,
        top_coordinate_family=top_family,
        intercept_basis_sign_product=intercept_sign_product,
        top_coordinate_basis_sign_product=top_sign_product,
        max_inert_right_shoulder_abs_basis_value=max_inert_right_shoulder_abs_basis_value,
        required_increment_share_of_top_coordinate_support_gap=(
            activation_report.required_increment_share_of_top_coordinate_support_gap
        ),
    )
    canonical_digest = (
        "- basis contract stays paper-trigonometric on "
        f"`{activation_report.binding_design[0]}/{activation_report.binding_design[1]}/{activation_report.binding_design[2]}`: "
        f"degree `{_MONTE_CARLO_TRIG_DEGREE}` gives `{basis_dimension}` columns, and the active same-sign block "
        f"`{{{', '.join(str(index) for index in activation_report.active_same_sign_coordinate_set)}}}` maps to "
        f"`{_format_label_set(active_coordinate_labels)}` while the inert block "
        f"`{{{', '.join(str(index) for index in activation_report.inert_same_sign_coordinate_set)}}}` maps to "
        f"`{_format_label_set(inert_coordinate_labels)}`",
        f"- coordinate `{top_coordinate}` is the first sine harmonic `{top_label}`: it evaluates to `{_format_signed(top_right_value)}` at failing right shoulder `z = {_format_grid_value(activation_report.failing_right_shoulder_grid_value)}` and `{_format_signed(top_center_value)}` at center `z = {_format_grid_value(activation_report.center_grid_value)}`, so the shared sign product stays `{_format_signed(top_sign_product)}`; by contrast the inert coordinates are numerically zero on the right shoulder (`max |psi_j({_format_grid_value(activation_report.failing_right_shoulder_grid_value)})| = {_format_float(max_inert_right_shoulder_abs_basis_value)}`)",
        f"- intercept coordinate `0` keeps the larger geometric sign product `{_format_signed(intercept_sign_product)}`, but the activation lane still concentrates on coordinate `{top_coordinate}`: its support gap is `{_format_signed(activation_report.top_coordinate_support_gap)}` (`{_format_percent(activation_report.top_coordinate_support_gap_share)}` of the same-sign gap), carries `{_format_ratio(activation_report.top_coordinate_gap_to_required_ratio)}` the required `{_format_signed(activation_report.required_incremental_right_center_covariance_lift)}` lift, and needs only `{_format_percent(activation_report.required_increment_share_of_top_coordinate_support_gap)}` of its own support / shared `v_f_hat[{top_coordinate},{top_coordinate}]` gap",
        f"- current Trigger 2 implication: `{driver_signature}`; source-level follow-up should inspect why anchor seed `{activation_report.coverage_anchor_random_state}` under-activates diagonal `{top_label}` support on the shared `v_f_hat[{top_coordinate},{top_coordinate}]` entry instead of rebalancing all same-sign coordinates or broad covariance windows",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateBasisTermReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-sign-coordinate-basis-term-probe"
        ),
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
        basis_family=_BASIS_FAMILY,
        basis_degree=_MONTE_CARLO_TRIG_DEGREE,
        basis_dimension=basis_dimension,
        top_coordinate=top_coordinate,
        top_coordinate_basis_label=top_label,
        top_coordinate_family=top_family,
        top_coordinate_harmonic=top_harmonic,
        top_coordinate_right_shoulder_basis_value=top_right_value,
        top_coordinate_center_basis_value=top_center_value,
        top_coordinate_basis_sign_product=top_sign_product,
        intercept_coordinate=intercept_coordinate,
        intercept_basis_label=intercept_label,
        intercept_basis_sign_product=intercept_sign_product,
        active_coordinate_labels=active_coordinate_labels,
        inert_coordinate_labels=inert_coordinate_labels,
        max_inert_right_shoulder_abs_basis_value=(
            max_inert_right_shoulder_abs_basis_value
        ),
        top_coordinate_support_gap=activation_report.top_coordinate_support_gap,
        top_coordinate_support_gap_share=(
            activation_report.top_coordinate_support_gap_share
        ),
        required_incremental_right_center_covariance_lift=(
            activation_report.required_incremental_right_center_covariance_lift
        ),
        required_increment_share_of_top_coordinate_support_gap=(
            activation_report.required_increment_share_of_top_coordinate_support_gap
        ),
        top_coordinate_gap_to_required_ratio=(
            activation_report.top_coordinate_gap_to_required_ratio
        ),
        driver_signature=driver_signature,
        canonical_basis_term_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_basis_term_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateBasisTermReport
):
    activation_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe()
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_basis_term_report(
        activation_report=activation_report
    )
