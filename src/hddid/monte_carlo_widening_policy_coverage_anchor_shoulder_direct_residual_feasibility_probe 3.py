from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt

from .monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _center_conditioned_partial_correlation(
    cross_shoulder_correlation: float,
    *,
    left_center_correlation: float,
    right_center_correlation: float,
) -> float:
    left_center = float(left_center_correlation)
    right_center = float(right_center_correlation)
    denominator = (1.0 - left_center * left_center) * (
        1.0 - right_center * right_center
    )
    if denominator <= 0.0:
        raise ValueError(
            "partial correlation requires strictly positive conditioning variance"
        )
    return float(
        (float(cross_shoulder_correlation) - left_center * right_center)
        / sqrt(denominator)
    )


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _driver_signature(
    *,
    direct_residual_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualReport
    ),
    zero_partial_right_center_correlation: float,
    repaired_anchor_partial_cross_shoulder_correlation: float,
) -> str:
    if (
        direct_residual_report.driver_signature
        == "direct-residual-correlation-repair-target"
        and zero_partial_right_center_correlation < -1.0
        and repaired_anchor_partial_cross_shoulder_correlation < 0.0
    ):
        return "sign-healing-outside-correlation-feasible-repair-lane"
    if repaired_anchor_partial_cross_shoulder_correlation < 0.0:
        return "bounded-repair-preserves-negative-direct-residual"
    return "mixed-direct-residual-feasibility"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    anchor_left_center_correlation: float
    anchor_cross_shoulder_correlation: float
    current_anchor_right_center_correlation: float
    required_repaired_right_center_correlation: float
    current_anchor_partial_cross_shoulder_correlation: float
    repaired_anchor_partial_cross_shoulder_correlation: float
    zero_partial_right_center_correlation: float
    zero_partial_lower_bound_gap: float
    required_repair_distance_to_zero_partial_target: float
    repaired_partial_abs_multiple_vs_current: float
    driver_signature: str
    canonical_direct_residual_feasibility_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.left_shoulder_grid_value = float(self.left_shoulder_grid_value)
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.anchor_left_center_correlation = float(self.anchor_left_center_correlation)
        self.anchor_cross_shoulder_correlation = float(
            self.anchor_cross_shoulder_correlation
        )
        self.current_anchor_right_center_correlation = float(
            self.current_anchor_right_center_correlation
        )
        self.required_repaired_right_center_correlation = float(
            self.required_repaired_right_center_correlation
        )
        self.current_anchor_partial_cross_shoulder_correlation = float(
            self.current_anchor_partial_cross_shoulder_correlation
        )
        self.repaired_anchor_partial_cross_shoulder_correlation = float(
            self.repaired_anchor_partial_cross_shoulder_correlation
        )
        self.zero_partial_right_center_correlation = float(
            self.zero_partial_right_center_correlation
        )
        self.zero_partial_lower_bound_gap = float(self.zero_partial_lower_bound_gap)
        self.required_repair_distance_to_zero_partial_target = float(
            self.required_repair_distance_to_zero_partial_target
        )
        self.repaired_partial_abs_multiple_vs_current = float(
            self.repaired_partial_abs_multiple_vs_current
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_direct_residual_feasibility_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_direct_residual_feasibility_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "left_shoulder_grid_value": self.left_shoulder_grid_value,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "anchor_left_center_correlation": self.anchor_left_center_correlation,
            "anchor_cross_shoulder_correlation": self.anchor_cross_shoulder_correlation,
            "current_anchor_right_center_correlation": (
                self.current_anchor_right_center_correlation
            ),
            "required_repaired_right_center_correlation": (
                self.required_repaired_right_center_correlation
            ),
            "current_anchor_partial_cross_shoulder_correlation": (
                self.current_anchor_partial_cross_shoulder_correlation
            ),
            "repaired_anchor_partial_cross_shoulder_correlation": (
                self.repaired_anchor_partial_cross_shoulder_correlation
            ),
            "zero_partial_right_center_correlation": (
                self.zero_partial_right_center_correlation
            ),
            "zero_partial_lower_bound_gap": self.zero_partial_lower_bound_gap,
            "required_repair_distance_to_zero_partial_target": (
                self.required_repair_distance_to_zero_partial_target
            ),
            "repaired_partial_abs_multiple_vs_current": (
                self.repaired_partial_abs_multiple_vs_current
            ),
            "driver_signature": self.driver_signature,
            "canonical_direct_residual_feasibility_digest": list(
                self.canonical_direct_residual_feasibility_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_report(
    *,
    cross_shoulder_sign_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport
    ),
    correlation_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport
    ),
    direct_residual_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualReport
    ),
) -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport
):
    if cross_shoulder_sign_report.policy_digest != correlation_gap_report.policy_digest:
        raise ValueError(
            "direct residual feasibility probe requires a single policy digest"
        )
    if cross_shoulder_sign_report.policy_digest != direct_residual_report.policy_digest:
        raise ValueError(
            "direct residual feasibility probe requires a single policy digest"
        )
    if (
        cross_shoulder_sign_report.binding_design
        != correlation_gap_report.binding_design
    ):
        raise ValueError(
            "direct residual feasibility probe requires a single binding design"
        )
    if (
        cross_shoulder_sign_report.binding_design
        != direct_residual_report.binding_design
    ):
        raise ValueError(
            "direct residual feasibility probe requires a single binding design"
        )
    if (
        cross_shoulder_sign_report.coverage_anchor_random_state
        != correlation_gap_report.coverage_anchor_random_state
        or cross_shoulder_sign_report.coverage_anchor_random_state
        != direct_residual_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "direct residual feasibility probe requires a single coverage-anchor seed"
        )
    if (
        cross_shoulder_sign_report.overshoot_companion_random_state
        != correlation_gap_report.overshoot_companion_random_state
        or cross_shoulder_sign_report.overshoot_companion_random_state
        != direct_residual_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "direct residual feasibility probe requires a single overshoot companion seed"
        )
    if (
        cross_shoulder_sign_report.left_shoulder_grid_value
        != direct_residual_report.left_shoulder_grid_value
    ):
        raise ValueError(
            "direct residual feasibility probe requires the same left shoulder grid"
        )
    if (
        cross_shoulder_sign_report.center_grid_value
        != correlation_gap_report.center_grid_value
    ):
        raise ValueError(
            "direct residual feasibility probe requires the same center grid"
        )
    if (
        cross_shoulder_sign_report.center_grid_value
        != direct_residual_report.center_grid_value
    ):
        raise ValueError(
            "direct residual feasibility probe requires the same center grid"
        )
    if (
        cross_shoulder_sign_report.failing_right_shoulder_grid_value
        != correlation_gap_report.failing_right_shoulder_grid_value
    ):
        raise ValueError(
            "direct residual feasibility probe requires the same failing shoulder grid"
        )
    if (
        cross_shoulder_sign_report.failing_right_shoulder_grid_value
        != direct_residual_report.failing_right_shoulder_grid_value
    ):
        raise ValueError(
            "direct residual feasibility probe requires the same failing shoulder grid"
        )

    zero_partial_right_center_correlation = float(
        cross_shoulder_sign_report.anchor_cross_shoulder_correlation
        / cross_shoulder_sign_report.anchor_left_center_correlation
    )
    if zero_partial_right_center_correlation >= -1.0:
        raise ValueError(
            "direct residual feasibility probe expects zero-partial target outside feasible correlation support"
        )

    repaired_anchor_partial_cross_shoulder_correlation = (
        _center_conditioned_partial_correlation(
            cross_shoulder_sign_report.anchor_cross_shoulder_correlation,
            left_center_correlation=(
                cross_shoulder_sign_report.anchor_left_center_correlation
            ),
            right_center_correlation=(
                correlation_gap_report.required_repaired_correlation
            ),
        )
    )
    if repaired_anchor_partial_cross_shoulder_correlation >= 0.0:
        raise ValueError(
            "direct residual feasibility probe expects bounded repair to keep the partial residual negative"
        )

    zero_partial_lower_bound_gap = float(
        abs(zero_partial_right_center_correlation - (-1.0))
    )
    required_repair_distance_to_zero_partial_target = float(
        correlation_gap_report.required_repaired_correlation
        - zero_partial_right_center_correlation
    )
    repaired_partial_abs_multiple_vs_current = _positive_ratio(
        abs(repaired_anchor_partial_cross_shoulder_correlation),
        abs(direct_residual_report.anchor_partial_cross_shoulder_correlation),
        label="repaired_partial_abs_multiple_vs_current",
    )
    driver_signature = _driver_signature(
        direct_residual_report=direct_residual_report,
        zero_partial_right_center_correlation=zero_partial_right_center_correlation,
        repaired_anchor_partial_cross_shoulder_correlation=(
            repaired_anchor_partial_cross_shoulder_correlation
        ),
    )

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*cross_shoulder_sign_report.binding_design)}` on "
        f"`{cross_shoulder_sign_report.window_label}`: anchor seed "
        f"`{cross_shoulder_sign_report.coverage_anchor_random_state}` keeps left-center "
        f"correlation `{_format_float(cross_shoulder_sign_report.anchor_left_center_correlation)}` "
        "and raw cross-shoulder correlation "
        f"`{_format_float(cross_shoulder_sign_report.anchor_cross_shoulder_correlation)}`; "
        "the bounded Trigger 2 repair lane only raises right-center correlation from "
        f"`{_format_float(cross_shoulder_sign_report.anchor_right_center_correlation)}` to "
        f"`{_format_float(correlation_gap_report.required_repaired_correlation)}` at failing "
        f"shoulder `z = {_format_grid_value(cross_shoulder_sign_report.failing_right_shoulder_grid_value)}` "
        f"versus center `z = {_format_grid_value(cross_shoulder_sign_report.center_grid_value)}`",
        "- with raw cross-shoulder correlation fixed, zeroing the center-conditioned "
        "direct residual would require right-center correlation "
        f"`{_format_float(zero_partial_right_center_correlation)}`, which sits "
        f"`{_format_float(zero_partial_lower_bound_gap)}` below the feasible correlation "
        "lower bound `-1.000` and "
        f"`{_format_float(required_repair_distance_to_zero_partial_target)}` away from the "
        f"bounded repair target `{_format_float(correlation_gap_report.required_repaired_correlation)}`",
        "- bounded repair therefore cannot heal the direct residual: anchor partial "
        "cross-shoulder correlation stays negative at "
        f"`{_format_float(repaired_anchor_partial_cross_shoulder_correlation)}` after repair "
        "versus "
        f"`{_format_float(direct_residual_report.anchor_partial_cross_shoulder_correlation)}` "
        f"today, i.e. the conditioned negative residual becomes "
        f"`{_format_ratio(repaired_partial_abs_multiple_vs_current)}` as large in absolute "
        "value rather than disappearing",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; source-level follow-up should treat the negative direct "
        "residual as a fixed background constraint inside the bounded right-shoulder / "
        "center repair lane, not as a co-target of the current fix",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-direct-residual-feasibility-probe"
        ),
        policy_digest=cross_shoulder_sign_report.policy_digest,
        binding_design=cross_shoulder_sign_report.binding_design,
        coverage_anchor_random_state=(
            cross_shoulder_sign_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            cross_shoulder_sign_report.overshoot_companion_random_state
        ),
        left_shoulder_grid_value=(cross_shoulder_sign_report.left_shoulder_grid_value),
        center_grid_value=cross_shoulder_sign_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            cross_shoulder_sign_report.failing_right_shoulder_grid_value
        ),
        anchor_left_center_correlation=(
            cross_shoulder_sign_report.anchor_left_center_correlation
        ),
        anchor_cross_shoulder_correlation=(
            cross_shoulder_sign_report.anchor_cross_shoulder_correlation
        ),
        current_anchor_right_center_correlation=(
            cross_shoulder_sign_report.anchor_right_center_correlation
        ),
        required_repaired_right_center_correlation=(
            correlation_gap_report.required_repaired_correlation
        ),
        current_anchor_partial_cross_shoulder_correlation=(
            direct_residual_report.anchor_partial_cross_shoulder_correlation
        ),
        repaired_anchor_partial_cross_shoulder_correlation=(
            repaired_anchor_partial_cross_shoulder_correlation
        ),
        zero_partial_right_center_correlation=zero_partial_right_center_correlation,
        zero_partial_lower_bound_gap=zero_partial_lower_bound_gap,
        required_repair_distance_to_zero_partial_target=(
            required_repair_distance_to_zero_partial_target
        ),
        repaired_partial_abs_multiple_vs_current=(
            repaired_partial_abs_multiple_vs_current
        ),
        driver_signature=driver_signature,
        canonical_direct_residual_feasibility_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_report(
        cross_shoulder_sign_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe()
        ),
        correlation_gap_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe()
        ),
        direct_residual_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_probe()
        ),
    )
