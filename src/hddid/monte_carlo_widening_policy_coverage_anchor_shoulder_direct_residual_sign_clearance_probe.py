from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_raw_cross_lift_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualRawCrossLiftReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_raw_cross_lift_probe,
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
    raw_cross_lift_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualRawCrossLiftReport
    ),
    sign_clearance_share_of_required_lift: float,
    positive_reserve_share_of_required_lift: float,
) -> str:
    if (
        raw_cross_lift_report.driver_signature
        == "bounded-raw-cross-sign-healing-side-lane"
        and sign_clearance_share_of_required_lift > 0.95
        and positive_reserve_share_of_required_lift < 0.05
    ):
        return "raw-cross-sign-clearance-dominates-side-lane"
    return "mixed-raw-cross-sign-clearance-lane"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualSignClearanceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    current_anchor_cross_shoulder_correlation: float
    zero_partial_cross_shoulder_correlation_at_bounded_repair: float
    required_raw_cross_shoulder_lift: float
    raw_cross_sign_clearance_lift: float
    raw_cross_positive_reserve_lift: float
    sign_clearance_share_of_required_lift: float
    positive_reserve_share_of_required_lift: float
    sign_clearance_share_of_full_gap: float
    positive_reserve_share_of_full_gap: float
    driver_signature: str
    canonical_direct_residual_sign_clearance_digest: tuple[str, ...]

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
        self.current_anchor_cross_shoulder_correlation = float(
            self.current_anchor_cross_shoulder_correlation
        )
        self.zero_partial_cross_shoulder_correlation_at_bounded_repair = float(
            self.zero_partial_cross_shoulder_correlation_at_bounded_repair
        )
        self.required_raw_cross_shoulder_lift = float(
            self.required_raw_cross_shoulder_lift
        )
        self.raw_cross_sign_clearance_lift = float(self.raw_cross_sign_clearance_lift)
        self.raw_cross_positive_reserve_lift = float(
            self.raw_cross_positive_reserve_lift
        )
        self.sign_clearance_share_of_required_lift = float(
            self.sign_clearance_share_of_required_lift
        )
        self.positive_reserve_share_of_required_lift = float(
            self.positive_reserve_share_of_required_lift
        )
        self.sign_clearance_share_of_full_gap = float(
            self.sign_clearance_share_of_full_gap
        )
        self.positive_reserve_share_of_full_gap = float(
            self.positive_reserve_share_of_full_gap
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_direct_residual_sign_clearance_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_direct_residual_sign_clearance_digest
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
            "current_anchor_cross_shoulder_correlation": (
                self.current_anchor_cross_shoulder_correlation
            ),
            "zero_partial_cross_shoulder_correlation_at_bounded_repair": (
                self.zero_partial_cross_shoulder_correlation_at_bounded_repair
            ),
            "required_raw_cross_shoulder_lift": self.required_raw_cross_shoulder_lift,
            "raw_cross_sign_clearance_lift": self.raw_cross_sign_clearance_lift,
            "raw_cross_positive_reserve_lift": self.raw_cross_positive_reserve_lift,
            "sign_clearance_share_of_required_lift": (
                self.sign_clearance_share_of_required_lift
            ),
            "positive_reserve_share_of_required_lift": (
                self.positive_reserve_share_of_required_lift
            ),
            "sign_clearance_share_of_full_gap": self.sign_clearance_share_of_full_gap,
            "positive_reserve_share_of_full_gap": (
                self.positive_reserve_share_of_full_gap
            ),
            "driver_signature": self.driver_signature,
            "canonical_direct_residual_sign_clearance_digest": list(
                self.canonical_direct_residual_sign_clearance_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_sign_clearance_report(
    *,
    raw_cross_lift_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualRawCrossLiftReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualSignClearanceReport:
    current_anchor_cross = float(
        raw_cross_lift_report.current_anchor_cross_shoulder_correlation
    )
    zero_partial_target = float(
        raw_cross_lift_report.zero_partial_cross_shoulder_correlation_at_bounded_repair
    )
    required_raw_cross_lift = float(
        raw_cross_lift_report.required_raw_cross_shoulder_lift
    )
    full_gap = float(raw_cross_lift_report.full_anchor_to_companion_cross_shoulder_gap)

    if current_anchor_cross >= 0.0:
        raise ValueError(
            "sign clearance probe expects a negative raw cross correlation"
        )
    if zero_partial_target <= 0.0:
        raise ValueError("sign clearance probe expects a positive zero-partial target")

    raw_cross_sign_clearance_lift = float(-current_anchor_cross)
    raw_cross_positive_reserve_lift = float(zero_partial_target)
    if not isclose(
        raw_cross_sign_clearance_lift + raw_cross_positive_reserve_lift,
        required_raw_cross_lift,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "sign clearance probe expects sign clearance plus positive reserve to match total raw-cross lift"
        )

    sign_clearance_share_of_required_lift = _positive_ratio(
        raw_cross_sign_clearance_lift,
        required_raw_cross_lift,
        label="sign_clearance_share_of_required_lift",
    )
    positive_reserve_share_of_required_lift = _positive_ratio(
        raw_cross_positive_reserve_lift,
        required_raw_cross_lift,
        label="positive_reserve_share_of_required_lift",
    )
    sign_clearance_share_of_full_gap = _positive_ratio(
        raw_cross_sign_clearance_lift,
        full_gap,
        label="sign_clearance_share_of_full_gap",
    )
    positive_reserve_share_of_full_gap = _positive_ratio(
        raw_cross_positive_reserve_lift,
        full_gap,
        label="positive_reserve_share_of_full_gap",
    )
    driver_signature = _driver_signature(
        raw_cross_lift_report=raw_cross_lift_report,
        sign_clearance_share_of_required_lift=sign_clearance_share_of_required_lift,
        positive_reserve_share_of_required_lift=(
            positive_reserve_share_of_required_lift
        ),
    )

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*raw_cross_lift_report.binding_design)}` on "
        f"`near_zero_grid`: with bounded right-center repair fixed at "
        f"`{_format_float(raw_cross_lift_report.bounded_repaired_right_center_correlation)}`, "
        "the raw cross-shoulder side lane only needs "
        f"`{_format_signed_float(raw_cross_sign_clearance_lift)}` to clear the current negative sign "
        f"at `{_format_float(current_anchor_cross)}` and another "
        f"`{_format_signed_float(raw_cross_positive_reserve_lift)}` beyond zero to satisfy the "
        f"zero-partial target `{_format_float(zero_partial_target)}`",
        "- sign clearance dominates the side-lane budget: "
        f"`{_format_percent(sign_clearance_share_of_required_lift)}` of the required raw-cross lift "
        f"`{_format_signed_float(required_raw_cross_lift)}` is just clearing the negative sign, "
        f"while only `{_format_percent(positive_reserve_share_of_required_lift)}` is positive "
        "reserve beyond zero",
        "- companion overhang stays outside the bounded side lane: sign clearance itself is only "
        f"`{_format_percent(sign_clearance_share_of_full_gap)}` of the full anchor-to-companion "
        "raw-cross gap, and the positive reserve above zero is only "
        f"`{_format_percent(positive_reserve_share_of_full_gap)}` of that full gap",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; any future direct-residual healing should treat positive reserve "
        "as a marginal follow-through after sign clearance rather than replaying companion "
        "raw-cross geometry",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualSignClearanceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-direct-residual-sign-clearance-probe"
        ),
        policy_digest=raw_cross_lift_report.policy_digest,
        binding_design=raw_cross_lift_report.binding_design,
        coverage_anchor_random_state=(
            raw_cross_lift_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            raw_cross_lift_report.overshoot_companion_random_state
        ),
        left_shoulder_grid_value=raw_cross_lift_report.left_shoulder_grid_value,
        center_grid_value=raw_cross_lift_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            raw_cross_lift_report.failing_right_shoulder_grid_value
        ),
        current_anchor_cross_shoulder_correlation=current_anchor_cross,
        zero_partial_cross_shoulder_correlation_at_bounded_repair=(zero_partial_target),
        required_raw_cross_shoulder_lift=required_raw_cross_lift,
        raw_cross_sign_clearance_lift=raw_cross_sign_clearance_lift,
        raw_cross_positive_reserve_lift=raw_cross_positive_reserve_lift,
        sign_clearance_share_of_required_lift=(sign_clearance_share_of_required_lift),
        positive_reserve_share_of_required_lift=(
            positive_reserve_share_of_required_lift
        ),
        sign_clearance_share_of_full_gap=sign_clearance_share_of_full_gap,
        positive_reserve_share_of_full_gap=positive_reserve_share_of_full_gap,
        driver_signature=driver_signature,
        canonical_direct_residual_sign_clearance_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_sign_clearance_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualSignClearanceReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_sign_clearance_report(
        raw_cross_lift_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_raw_cross_lift_probe()
        )
    )
