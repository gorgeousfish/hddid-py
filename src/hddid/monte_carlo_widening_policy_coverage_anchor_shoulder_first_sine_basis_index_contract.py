from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .basis import trigonometric_sieve_basis
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe,
)

_BASIS_FAMILY = "trigonometric"
_MONTE_CARLO_TRIG_DEGREE = 8


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


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _driver_signature(
    *,
    activation_signature: str,
    prefix_coordinate_labels: tuple[str, ...],
    target_coordinate: int,
    target_coordinate_basis_label: str,
    lower_adjacent_sign_product: float,
    target_basis_sign_product: float,
    upper_adjacent_sign_product: float,
) -> str:
    if (
        activation_signature == "single-coordinate-same-sign-activation-lane"
        and prefix_coordinate_labels
        == ("1", "cos(2πz)", "sin(2πz)", "cos(4πz)", "sin(4πz)")
        and target_coordinate == 2
        and target_coordinate_basis_label == "sin(2πz)"
        and abs(lower_adjacent_sign_product) <= 1e-12
        and target_basis_sign_product > upper_adjacent_sign_product > 0.0
    ):
        return "first-sine-basis-index-contract"
    return "mixed-basis-index-lane"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineBasisIndexContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    basis_family: str
    basis_degree: int
    basis_dimension: int
    prefix_coordinate_labels: tuple[str, ...]
    target_coordinate: int
    target_coordinate_basis_label: str
    target_coordinate_family: str
    target_coordinate_harmonic: int
    lower_adjacent_coordinate: int
    lower_adjacent_basis_label: str
    lower_adjacent_family: str
    lower_adjacent_harmonic: int
    upper_adjacent_coordinate: int
    upper_adjacent_basis_label: str
    upper_adjacent_family: str
    upper_adjacent_harmonic: int
    right_shoulder_grid_value: float
    center_grid_value: float
    target_right_shoulder_basis_value: float
    target_center_basis_value: float
    target_basis_sign_product: float
    lower_adjacent_right_shoulder_basis_value: float
    lower_adjacent_center_basis_value: float
    lower_adjacent_sign_product: float
    upper_adjacent_right_shoulder_basis_value: float
    upper_adjacent_center_basis_value: float
    upper_adjacent_sign_product: float
    target_sign_product_lead_over_lower: float
    target_sign_product_lead_over_upper: float
    shared_diagonal_entry_label: str
    omega_diagonal_entry_label: str
    driver_signature: str
    canonical_basis_index_contract_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.basis_family = str(self.basis_family).strip()
        self.basis_degree = int(self.basis_degree)
        self.basis_dimension = int(self.basis_dimension)
        self.prefix_coordinate_labels = tuple(
            str(label).strip() for label in self.prefix_coordinate_labels
        )
        self.target_coordinate = int(self.target_coordinate)
        self.target_coordinate_basis_label = str(
            self.target_coordinate_basis_label
        ).strip()
        self.target_coordinate_family = str(self.target_coordinate_family).strip()
        self.target_coordinate_harmonic = int(self.target_coordinate_harmonic)
        self.lower_adjacent_coordinate = int(self.lower_adjacent_coordinate)
        self.lower_adjacent_basis_label = str(self.lower_adjacent_basis_label).strip()
        self.lower_adjacent_family = str(self.lower_adjacent_family).strip()
        self.lower_adjacent_harmonic = int(self.lower_adjacent_harmonic)
        self.upper_adjacent_coordinate = int(self.upper_adjacent_coordinate)
        self.upper_adjacent_basis_label = str(self.upper_adjacent_basis_label).strip()
        self.upper_adjacent_family = str(self.upper_adjacent_family).strip()
        self.upper_adjacent_harmonic = int(self.upper_adjacent_harmonic)
        self.right_shoulder_grid_value = float(self.right_shoulder_grid_value)
        self.center_grid_value = float(self.center_grid_value)
        self.target_right_shoulder_basis_value = float(
            self.target_right_shoulder_basis_value
        )
        self.target_center_basis_value = float(self.target_center_basis_value)
        self.target_basis_sign_product = float(self.target_basis_sign_product)
        self.lower_adjacent_right_shoulder_basis_value = float(
            self.lower_adjacent_right_shoulder_basis_value
        )
        self.lower_adjacent_center_basis_value = float(
            self.lower_adjacent_center_basis_value
        )
        self.lower_adjacent_sign_product = float(self.lower_adjacent_sign_product)
        self.upper_adjacent_right_shoulder_basis_value = float(
            self.upper_adjacent_right_shoulder_basis_value
        )
        self.upper_adjacent_center_basis_value = float(
            self.upper_adjacent_center_basis_value
        )
        self.upper_adjacent_sign_product = float(self.upper_adjacent_sign_product)
        self.target_sign_product_lead_over_lower = float(
            self.target_sign_product_lead_over_lower
        )
        self.target_sign_product_lead_over_upper = float(
            self.target_sign_product_lead_over_upper
        )
        self.shared_diagonal_entry_label = str(self.shared_diagonal_entry_label).strip()
        self.omega_diagonal_entry_label = str(self.omega_diagonal_entry_label).strip()
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_basis_index_contract_digest = tuple(
            str(line).rstrip() for line in self.canonical_basis_index_contract_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "basis_family": self.basis_family,
            "basis_degree": self.basis_degree,
            "basis_dimension": self.basis_dimension,
            "prefix_coordinate_labels": list(self.prefix_coordinate_labels),
            "target_coordinate": self.target_coordinate,
            "target_coordinate_basis_label": self.target_coordinate_basis_label,
            "target_coordinate_family": self.target_coordinate_family,
            "target_coordinate_harmonic": self.target_coordinate_harmonic,
            "lower_adjacent_coordinate": self.lower_adjacent_coordinate,
            "lower_adjacent_basis_label": self.lower_adjacent_basis_label,
            "lower_adjacent_family": self.lower_adjacent_family,
            "lower_adjacent_harmonic": self.lower_adjacent_harmonic,
            "upper_adjacent_coordinate": self.upper_adjacent_coordinate,
            "upper_adjacent_basis_label": self.upper_adjacent_basis_label,
            "upper_adjacent_family": self.upper_adjacent_family,
            "upper_adjacent_harmonic": self.upper_adjacent_harmonic,
            "right_shoulder_grid_value": self.right_shoulder_grid_value,
            "center_grid_value": self.center_grid_value,
            "target_right_shoulder_basis_value": self.target_right_shoulder_basis_value,
            "target_center_basis_value": self.target_center_basis_value,
            "target_basis_sign_product": self.target_basis_sign_product,
            "lower_adjacent_right_shoulder_basis_value": (
                self.lower_adjacent_right_shoulder_basis_value
            ),
            "lower_adjacent_center_basis_value": self.lower_adjacent_center_basis_value,
            "lower_adjacent_sign_product": self.lower_adjacent_sign_product,
            "upper_adjacent_right_shoulder_basis_value": (
                self.upper_adjacent_right_shoulder_basis_value
            ),
            "upper_adjacent_center_basis_value": self.upper_adjacent_center_basis_value,
            "upper_adjacent_sign_product": self.upper_adjacent_sign_product,
            "target_sign_product_lead_over_lower": self.target_sign_product_lead_over_lower,
            "target_sign_product_lead_over_upper": self.target_sign_product_lead_over_upper,
            "shared_diagonal_entry_label": self.shared_diagonal_entry_label,
            "omega_diagonal_entry_label": self.omega_diagonal_entry_label,
            "driver_signature": self.driver_signature,
            "canonical_basis_index_contract_digest": list(
                self.canonical_basis_index_contract_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_basis_index_contract_report(
    *,
    activation_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSignCoordinateActivationReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineBasisIndexContractReport:
    if activation_report.top_coordinate <= 0:
        raise ValueError("basis index contract requires a positive target coordinate")

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

    prefix_coordinate_labels = tuple(
        _coordinate_basis_metadata(index)[0] for index in range(5)
    )

    target_coordinate = activation_report.top_coordinate
    lower_adjacent_coordinate = target_coordinate - 1
    upper_adjacent_coordinate = target_coordinate + 1
    if upper_adjacent_coordinate >= basis_rows.shape[1]:
        raise ValueError("upper adjacent coordinate must remain inside the basis")

    target_label, target_family, target_harmonic = _coordinate_basis_metadata(
        target_coordinate
    )
    lower_label, lower_family, lower_harmonic = _coordinate_basis_metadata(
        lower_adjacent_coordinate
    )
    upper_label, upper_family, upper_harmonic = _coordinate_basis_metadata(
        upper_adjacent_coordinate
    )

    target_right_value = float(basis_rows[0, target_coordinate])
    target_center_value = float(basis_rows[1, target_coordinate])
    lower_right_value = float(basis_rows[0, lower_adjacent_coordinate])
    lower_center_value = float(basis_rows[1, lower_adjacent_coordinate])
    upper_right_value = float(basis_rows[0, upper_adjacent_coordinate])
    upper_center_value = float(basis_rows[1, upper_adjacent_coordinate])

    target_sign_product = float(target_right_value * target_center_value)
    lower_sign_product = float(lower_right_value * lower_center_value)
    upper_sign_product = float(upper_right_value * upper_center_value)

    target_sign_product_lead_over_lower = float(
        target_sign_product - lower_sign_product
    )
    target_sign_product_lead_over_upper = float(
        target_sign_product - upper_sign_product
    )

    driver_signature = _driver_signature(
        activation_signature=activation_report.driver_signature,
        prefix_coordinate_labels=prefix_coordinate_labels,
        target_coordinate=target_coordinate,
        target_coordinate_basis_label=target_label,
        lower_adjacent_sign_product=lower_sign_product,
        target_basis_sign_product=target_sign_product,
        upper_adjacent_sign_product=upper_sign_product,
    )

    digest = (
        "- under the paper-trigonometric degree-8 helper order, the live prefix stays "
        f"`[{', '.join(prefix_coordinate_labels)}]`, so Trigger 2 coordinate `{target_coordinate}` "
        f"still points to `{target_label}` and therefore to the shared diagonal patch atom "
        f"`v_f_hat[{target_coordinate},{target_coordinate}]` / "
        f"`omega_f_hat[{target_coordinate},{target_coordinate}]`",
        f"- the index contract is directional, not cosmetic: at failing right shoulder "
        f"`z = {_format_grid_value(activation_report.failing_right_shoulder_grid_value)}`, "
        f"coordinate `{lower_adjacent_coordinate} = {lower_label}` is numerically zero while "
        f"coordinate `{target_coordinate} = {target_label}` equals "
        f"`{_format_signed(target_right_value)}`; at center "
        f"`z = {_format_grid_value(activation_report.center_grid_value)}`, their values are "
        f"`{_format_signed(lower_center_value)}` and `{_format_signed(target_center_value)}`, "
        f"so the shared sign product is `{_format_signed(lower_sign_product)}` for coordinate "
        f"`{lower_adjacent_coordinate}` versus `{_format_signed(target_sign_product)}` for "
        f"coordinate `{target_coordinate}`",
        f"- the nearest upward off-by-one is also out-of-lane: coordinate "
        f"`{upper_adjacent_coordinate} = {upper_label}` keeps a smaller shared sign product "
        f"`{_format_signed(upper_sign_product)}`, trailing coordinate `{target_coordinate}` by "
        f"`{_format_signed(target_sign_product_lead_over_upper)}`, so drifting the live patch to "
        "the next column would move it from the first-sine lane into a weaker higher-harmonic fallback",
        f"- current Trigger 2 implication: `{driver_signature}`; implementation should freeze "
        f"coordinate `{target_coordinate} = {target_label}` before patching bounded right-center "
        "support, rather than swapping sin/cos ordering, reinterpreting the lane as one-based "
        "harmonic counting, or shifting the diagonal target to adjacent columns",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineBasisIndexContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-basis-index-contract"
        ),
        policy_digest=activation_report.policy_digest,
        binding_design=activation_report.binding_design,
        window_label=activation_report.window_label,
        basis_family=_BASIS_FAMILY,
        basis_degree=_MONTE_CARLO_TRIG_DEGREE,
        basis_dimension=int(basis_rows.shape[1]),
        prefix_coordinate_labels=prefix_coordinate_labels,
        target_coordinate=target_coordinate,
        target_coordinate_basis_label=target_label,
        target_coordinate_family=target_family,
        target_coordinate_harmonic=target_harmonic,
        lower_adjacent_coordinate=lower_adjacent_coordinate,
        lower_adjacent_basis_label=lower_label,
        lower_adjacent_family=lower_family,
        lower_adjacent_harmonic=lower_harmonic,
        upper_adjacent_coordinate=upper_adjacent_coordinate,
        upper_adjacent_basis_label=upper_label,
        upper_adjacent_family=upper_family,
        upper_adjacent_harmonic=upper_harmonic,
        right_shoulder_grid_value=activation_report.failing_right_shoulder_grid_value,
        center_grid_value=activation_report.center_grid_value,
        target_right_shoulder_basis_value=target_right_value,
        target_center_basis_value=target_center_value,
        target_basis_sign_product=target_sign_product,
        lower_adjacent_right_shoulder_basis_value=lower_right_value,
        lower_adjacent_center_basis_value=lower_center_value,
        lower_adjacent_sign_product=lower_sign_product,
        upper_adjacent_right_shoulder_basis_value=upper_right_value,
        upper_adjacent_center_basis_value=upper_center_value,
        upper_adjacent_sign_product=upper_sign_product,
        target_sign_product_lead_over_lower=target_sign_product_lead_over_lower,
        target_sign_product_lead_over_upper=target_sign_product_lead_over_upper,
        shared_diagonal_entry_label=f"v_f_hat[{target_coordinate},{target_coordinate}]",
        omega_diagonal_entry_label=f"omega_f_hat[{target_coordinate},{target_coordinate}]",
        driver_signature=driver_signature,
        canonical_basis_index_contract_digest=digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_basis_index_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineBasisIndexContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_basis_index_contract_report(
        activation_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_sign_coordinate_activation_probe()
    )
