from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _driver_signature(
    *,
    feasibility_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport
    ),
    required_raw_cross_gap_share: float,
    sign_flip_share_of_required_lift: float,
) -> str:
    if (
        feasibility_report.driver_signature
        == "sign-healing-outside-correlation-feasible-repair-lane"
        and 0.0 < required_raw_cross_gap_share < 0.2
        and sign_flip_share_of_required_lift > 0.9
    ):
        return "bounded-raw-cross-sign-healing-side-lane"
    return "mixed-raw-cross-sign-healing-lane"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualRawCrossLiftReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    anchor_left_center_correlation: float
    current_anchor_cross_shoulder_correlation: float
    bounded_repaired_right_center_correlation: float
    zero_partial_cross_shoulder_correlation_at_bounded_repair: float
    required_raw_cross_shoulder_lift: float
    full_anchor_to_companion_cross_shoulder_gap: float
    required_raw_cross_gap_share: float
    remaining_raw_cross_gap_headroom_share: float
    sign_flip_share_of_required_lift: float
    zero_partial_cross_shoulder_share_of_companion: float
    driver_signature: str
    canonical_direct_residual_raw_cross_lift_digest: tuple[str, ...]

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
        self.current_anchor_cross_shoulder_correlation = float(
            self.current_anchor_cross_shoulder_correlation
        )
        self.bounded_repaired_right_center_correlation = float(
            self.bounded_repaired_right_center_correlation
        )
        self.zero_partial_cross_shoulder_correlation_at_bounded_repair = float(
            self.zero_partial_cross_shoulder_correlation_at_bounded_repair
        )
        self.required_raw_cross_shoulder_lift = float(
            self.required_raw_cross_shoulder_lift
        )
        self.full_anchor_to_companion_cross_shoulder_gap = float(
            self.full_anchor_to_companion_cross_shoulder_gap
        )
        self.required_raw_cross_gap_share = float(self.required_raw_cross_gap_share)
        self.remaining_raw_cross_gap_headroom_share = float(
            self.remaining_raw_cross_gap_headroom_share
        )
        self.sign_flip_share_of_required_lift = float(
            self.sign_flip_share_of_required_lift
        )
        self.zero_partial_cross_shoulder_share_of_companion = float(
            self.zero_partial_cross_shoulder_share_of_companion
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_direct_residual_raw_cross_lift_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_direct_residual_raw_cross_lift_digest
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
            "current_anchor_cross_shoulder_correlation": (
                self.current_anchor_cross_shoulder_correlation
            ),
            "bounded_repaired_right_center_correlation": (
                self.bounded_repaired_right_center_correlation
            ),
            "zero_partial_cross_shoulder_correlation_at_bounded_repair": (
                self.zero_partial_cross_shoulder_correlation_at_bounded_repair
            ),
            "required_raw_cross_shoulder_lift": self.required_raw_cross_shoulder_lift,
            "full_anchor_to_companion_cross_shoulder_gap": (
                self.full_anchor_to_companion_cross_shoulder_gap
            ),
            "required_raw_cross_gap_share": self.required_raw_cross_gap_share,
            "remaining_raw_cross_gap_headroom_share": (
                self.remaining_raw_cross_gap_headroom_share
            ),
            "sign_flip_share_of_required_lift": self.sign_flip_share_of_required_lift,
            "zero_partial_cross_shoulder_share_of_companion": (
                self.zero_partial_cross_shoulder_share_of_companion
            ),
            "driver_signature": self.driver_signature,
            "canonical_direct_residual_raw_cross_lift_digest": list(
                self.canonical_direct_residual_raw_cross_lift_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_raw_cross_lift_report(
    *,
    cross_shoulder_sign_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport
    ),
    correlation_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport
    ),
    feasibility_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport
    ),
) -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualRawCrossLiftReport
):
    if cross_shoulder_sign_report.policy_digest != correlation_gap_report.policy_digest:
        raise ValueError(
            "raw cross lift probe requires a single canonical policy digest"
        )
    if cross_shoulder_sign_report.policy_digest != feasibility_report.policy_digest:
        raise ValueError(
            "raw cross lift probe requires a single canonical policy digest"
        )
    if (
        cross_shoulder_sign_report.binding_design
        != correlation_gap_report.binding_design
    ):
        raise ValueError("raw cross lift probe requires a single binding design")
    if cross_shoulder_sign_report.binding_design != feasibility_report.binding_design:
        raise ValueError("raw cross lift probe requires a single binding design")
    if (
        cross_shoulder_sign_report.coverage_anchor_random_state
        != correlation_gap_report.coverage_anchor_random_state
        or cross_shoulder_sign_report.coverage_anchor_random_state
        != feasibility_report.coverage_anchor_random_state
    ):
        raise ValueError("raw cross lift probe requires a single coverage-anchor seed")
    if (
        cross_shoulder_sign_report.overshoot_companion_random_state
        != correlation_gap_report.overshoot_companion_random_state
        or cross_shoulder_sign_report.overshoot_companion_random_state
        != feasibility_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "raw cross lift probe requires a single overshoot companion seed"
        )
    if (
        cross_shoulder_sign_report.left_shoulder_grid_value
        != feasibility_report.left_shoulder_grid_value
    ):
        raise ValueError("raw cross lift probe requires the same left shoulder grid")
    if (
        cross_shoulder_sign_report.center_grid_value
        != correlation_gap_report.center_grid_value
    ):
        raise ValueError("raw cross lift probe requires the same center grid")
    if (
        cross_shoulder_sign_report.center_grid_value
        != feasibility_report.center_grid_value
    ):
        raise ValueError("raw cross lift probe requires the same center grid")
    if (
        cross_shoulder_sign_report.failing_right_shoulder_grid_value
        != correlation_gap_report.failing_right_shoulder_grid_value
    ):
        raise ValueError("raw cross lift probe requires the same failing shoulder grid")
    if (
        cross_shoulder_sign_report.failing_right_shoulder_grid_value
        != feasibility_report.failing_right_shoulder_grid_value
    ):
        raise ValueError("raw cross lift probe requires the same failing shoulder grid")

    zero_partial_cross_shoulder_correlation_at_bounded_repair = float(
        cross_shoulder_sign_report.anchor_left_center_correlation
        * correlation_gap_report.required_repaired_correlation
    )
    if zero_partial_cross_shoulder_correlation_at_bounded_repair <= 0.0:
        raise ValueError("zero-partial raw cross target must be positive")

    required_raw_cross_shoulder_lift = float(
        zero_partial_cross_shoulder_correlation_at_bounded_repair
        - cross_shoulder_sign_report.anchor_cross_shoulder_correlation
    )
    if required_raw_cross_shoulder_lift <= 0.0:
        raise ValueError("raw cross lift probe expects a positive sign-healing lift")

    full_anchor_to_companion_cross_shoulder_gap = float(
        cross_shoulder_sign_report.companion_cross_shoulder_correlation
        - cross_shoulder_sign_report.anchor_cross_shoulder_correlation
    )
    if full_anchor_to_companion_cross_shoulder_gap <= required_raw_cross_shoulder_lift:
        raise ValueError(
            "full companion raw cross gap must exceed the zero-partial lift"
        )

    required_raw_cross_gap_share = _positive_ratio(
        required_raw_cross_shoulder_lift,
        full_anchor_to_companion_cross_shoulder_gap,
        label="required_raw_cross_gap_share",
    )
    remaining_raw_cross_gap_headroom_share = _positive_ratio(
        cross_shoulder_sign_report.companion_cross_shoulder_correlation
        - zero_partial_cross_shoulder_correlation_at_bounded_repair,
        full_anchor_to_companion_cross_shoulder_gap,
        label="remaining_raw_cross_gap_headroom_share",
    )
    sign_flip_share_of_required_lift = _positive_ratio(
        abs(cross_shoulder_sign_report.anchor_cross_shoulder_correlation),
        required_raw_cross_shoulder_lift,
        label="sign_flip_share_of_required_lift",
    )
    zero_partial_cross_shoulder_share_of_companion = _positive_ratio(
        zero_partial_cross_shoulder_correlation_at_bounded_repair,
        cross_shoulder_sign_report.companion_cross_shoulder_correlation,
        label="zero_partial_cross_shoulder_share_of_companion",
    )
    driver_signature = _driver_signature(
        feasibility_report=feasibility_report,
        required_raw_cross_gap_share=required_raw_cross_gap_share,
        sign_flip_share_of_required_lift=sign_flip_share_of_required_lift,
    )

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*cross_shoulder_sign_report.binding_design)}` on "
        f"`{cross_shoulder_sign_report.window_label}`: holding bounded right-center repair at "
        f"`{_format_float(correlation_gap_report.required_repaired_correlation)}` with anchor "
        f"left-center correlation `{_format_float(cross_shoulder_sign_report.anchor_left_center_correlation)}`, "
        "zeroing the center-conditioned direct residual only requires raw cross-shoulder "
        f"correlation `{_format_float(zero_partial_cross_shoulder_correlation_at_bounded_repair)}` "
        "instead of the current "
        f"`{_format_float(cross_shoulder_sign_report.anchor_cross_shoulder_correlation)}`",
        "- that implies a raw cross-shoulder lift of "
        f"`{_format_signed_float(required_raw_cross_shoulder_lift)}`, which consumes only "
        f"`{_format_percent(required_raw_cross_gap_share)}` of the full anchor-to-companion "
        f"cross-shoulder gap `{_format_signed_float(full_anchor_to_companion_cross_shoulder_gap)}` "
        f"and leaves `{_format_percent(remaining_raw_cross_gap_headroom_share)}` headroom unused; "
        "the zero-partial target is only "
        f"`{_format_percent(zero_partial_cross_shoulder_share_of_companion)}` of companion seed "
        f"`{cross_shoulder_sign_report.overshoot_companion_random_state}` raw cross-shoulder correlation",
        "- almost the entire raw lift is just sign clearance: "
        f"`{_format_percent(sign_flip_share_of_required_lift)}` of the required move is spent "
        "crossing from negative to zero, not replaying companion geometry",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; if future source-level work ever targets direct-residual sign "
        "healing, it should seek a separate raw cross-shoulder entry path while keeping the "
        "current bounded right-shoulder / center repair lane focused on covariance access",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualRawCrossLiftReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-direct-residual-raw-cross-lift-probe"
        ),
        policy_digest=cross_shoulder_sign_report.policy_digest,
        binding_design=cross_shoulder_sign_report.binding_design,
        coverage_anchor_random_state=(
            cross_shoulder_sign_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            cross_shoulder_sign_report.overshoot_companion_random_state
        ),
        left_shoulder_grid_value=cross_shoulder_sign_report.left_shoulder_grid_value,
        center_grid_value=cross_shoulder_sign_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            cross_shoulder_sign_report.failing_right_shoulder_grid_value
        ),
        anchor_left_center_correlation=(
            cross_shoulder_sign_report.anchor_left_center_correlation
        ),
        current_anchor_cross_shoulder_correlation=(
            cross_shoulder_sign_report.anchor_cross_shoulder_correlation
        ),
        bounded_repaired_right_center_correlation=(
            correlation_gap_report.required_repaired_correlation
        ),
        zero_partial_cross_shoulder_correlation_at_bounded_repair=(
            zero_partial_cross_shoulder_correlation_at_bounded_repair
        ),
        required_raw_cross_shoulder_lift=required_raw_cross_shoulder_lift,
        full_anchor_to_companion_cross_shoulder_gap=(
            full_anchor_to_companion_cross_shoulder_gap
        ),
        required_raw_cross_gap_share=required_raw_cross_gap_share,
        remaining_raw_cross_gap_headroom_share=(remaining_raw_cross_gap_headroom_share),
        sign_flip_share_of_required_lift=sign_flip_share_of_required_lift,
        zero_partial_cross_shoulder_share_of_companion=(
            zero_partial_cross_shoulder_share_of_companion
        ),
        driver_signature=driver_signature,
        canonical_direct_residual_raw_cross_lift_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_raw_cross_lift_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualRawCrossLiftReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_raw_cross_lift_report(
        cross_shoulder_sign_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe()
        ),
        correlation_gap_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe()
        ),
        feasibility_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_probe()
        ),
    )
