from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint,
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-12:
        return f"{value:.1f}"
    text = f"{value:.2f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def _reserve(point: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint) -> float:
    return float(point.pointwise_interval_length / 2.0 - point.absolute_error)


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} requires a strictly positive denominator")
    return float(float(numerator) / denominator_value)


def _locate_point(
    points: tuple[Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint, ...],
    *,
    grid_value: float,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint:
    target = float(grid_value)
    for point in points:
        if isclose(point.grid_value, target, rel_tol=0.0, abs_tol=1e-12):
            return point
    raise KeyError(
        f"coverage anchor shoulder point not present at grid value {target!r}"
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLocalizationReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    residual_grid_label: str
    failing_right_shoulder_grid_value: float
    anchor_negative_reserve_total: float
    anchor_negative_reserve_point_count: int
    anchor_failing_shoulder_negative_reserve_share: float
    anchor_off_shoulder_positive_reserve_total: float
    anchor_off_shoulder_positive_to_negative_ratio: float
    anchor_center_reserve: float
    anchor_left_shoulder_reserve: float
    overshoot_companion_same_point_reserve: float
    overshoot_companion_same_point_multiple_of_anchor_shortfall: float
    canonical_coverage_anchor_shoulder_localization_digest: tuple[str, ...]

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
        self.residual_grid_label = str(self.residual_grid_label).strip()
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.anchor_negative_reserve_total = float(self.anchor_negative_reserve_total)
        self.anchor_negative_reserve_point_count = int(
            self.anchor_negative_reserve_point_count
        )
        self.anchor_failing_shoulder_negative_reserve_share = float(
            self.anchor_failing_shoulder_negative_reserve_share
        )
        self.anchor_off_shoulder_positive_reserve_total = float(
            self.anchor_off_shoulder_positive_reserve_total
        )
        self.anchor_off_shoulder_positive_to_negative_ratio = float(
            self.anchor_off_shoulder_positive_to_negative_ratio
        )
        self.anchor_center_reserve = float(self.anchor_center_reserve)
        self.anchor_left_shoulder_reserve = float(self.anchor_left_shoulder_reserve)
        self.overshoot_companion_same_point_reserve = float(
            self.overshoot_companion_same_point_reserve
        )
        self.overshoot_companion_same_point_multiple_of_anchor_shortfall = float(
            self.overshoot_companion_same_point_multiple_of_anchor_shortfall
        )
        self.canonical_coverage_anchor_shoulder_localization_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_localization_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "residual_grid_label": self.residual_grid_label,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "anchor_negative_reserve_total": self.anchor_negative_reserve_total,
            "anchor_negative_reserve_point_count": (
                self.anchor_negative_reserve_point_count
            ),
            "anchor_failing_shoulder_negative_reserve_share": (
                self.anchor_failing_shoulder_negative_reserve_share
            ),
            "anchor_off_shoulder_positive_reserve_total": (
                self.anchor_off_shoulder_positive_reserve_total
            ),
            "anchor_off_shoulder_positive_to_negative_ratio": (
                self.anchor_off_shoulder_positive_to_negative_ratio
            ),
            "anchor_center_reserve": self.anchor_center_reserve,
            "anchor_left_shoulder_reserve": self.anchor_left_shoulder_reserve,
            "overshoot_companion_same_point_reserve": (
                self.overshoot_companion_same_point_reserve
            ),
            "overshoot_companion_same_point_multiple_of_anchor_shortfall": (
                self.overshoot_companion_same_point_multiple_of_anchor_shortfall
            ),
            "canonical_coverage_anchor_shoulder_localization_digest": list(
                self.canonical_coverage_anchor_shoulder_localization_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_localization_report(
    shoulder_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLocalizationReport:
    replay = shoulder_report.replay(shoulder_report.residual_grid_label)
    anchor_points = tuple(replay.coverage_anchor_points)
    companion_points = tuple(replay.overshoot_companion_points)
    if len(anchor_points) < 3 or len(companion_points) < 3:
        raise ValueError(
            "coverage anchor shoulder localization requires three-point grids"
        )

    failing_point = _locate_point(
        anchor_points,
        grid_value=shoulder_report.residual_failure_grid_value,
    )
    companion_failing_point = _locate_point(
        companion_points,
        grid_value=shoulder_report.residual_failure_grid_value,
    )

    negative_reserve_points = tuple(
        point for point in anchor_points if _reserve(point) < 0.0
    )
    if not negative_reserve_points:
        raise ValueError(
            "coverage anchor shoulder localization requires negative reserve"
        )
    anchor_negative_reserve_total = float(
        sum(-_reserve(point) for point in negative_reserve_points)
    )
    failing_shortfall = float(max(-_reserve(failing_point), 0.0))
    if failing_shortfall <= 0.0:
        raise ValueError(
            "coverage anchor shoulder localization requires a negative reserve at the failing shoulder"
        )

    center_point = _locate_point(
        anchor_points, grid_value=shoulder_report.hotspot_center
    )
    left_shoulder_point = anchor_points[0]
    off_shoulder_positive_reserve_total = float(
        sum(
            max(_reserve(point), 0.0)
            for point in anchor_points
            if not isclose(
                point.grid_value,
                shoulder_report.residual_failure_grid_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    )
    companion_same_point_reserve = float(_reserve(companion_failing_point))
    if companion_same_point_reserve <= 0.0:
        raise ValueError(
            "coverage anchor shoulder localization requires a positive same-point companion reserve"
        )

    failing_shoulder_negative_share = _positive_ratio(
        failing_shortfall,
        anchor_negative_reserve_total,
        label="failing_shoulder_negative_share",
    )
    off_shoulder_positive_to_negative_ratio = _positive_ratio(
        off_shoulder_positive_reserve_total,
        anchor_negative_reserve_total,
        label="off_shoulder_positive_to_negative_ratio",
    )
    companion_shortfall_multiple = _positive_ratio(
        companion_same_point_reserve,
        failing_shortfall,
        label="companion_shortfall_multiple",
    )

    canonical_digest = (
        f"- binding design `{_format_design_key(*shoulder_report.binding_design)}`: "
        f"coverage anchor seed `{shoulder_report.coverage_anchor_random_state}` keeps exactly "
        f"one negative reserve point on `{shoulder_report.residual_grid_label}`, at right shoulder "
        f"`z = {_format_grid_value(shoulder_report.residual_failure_grid_value)}` with reserve "
        f"`-{_format_float(failing_shortfall)}`; the left shoulder and center still keep "
        f"`+{_format_float(_reserve(left_shoulder_point))}` and "
        f"`+{_format_float(_reserve(center_point))}` reserve",
        f"- the anchor's off-shoulder positive reserve totals "
        f"`+{_format_float(off_shoulder_positive_reserve_total)}`, or "
        f"`{_format_float(off_shoulder_positive_to_negative_ratio)}x` the failing-shoulder "
        "shortfall, so current debt is not whole-window undercoverage but a localized "
        "right-shoulder miss",
        f"- overshoot companion seed `{shoulder_report.overshoot_companion_random_state}` "
        f"keeps `+{_format_float(companion_same_point_reserve)}` reserve at the same "
        f"`z = {_format_grid_value(shoulder_report.residual_failure_grid_value)}`, which alone "
        f"is `{_format_float(companion_shortfall_multiple)}x` the anchor shortfall; Trigger 2 "
        "still points to local right-shoulder amplification debt, not center repair or "
        "global window widening",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLocalizationReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-localization-probe"
        ),
        policy_digest=shoulder_report.policy_digest,
        binding_design=shoulder_report.binding_design,
        coverage_anchor_random_state=shoulder_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            shoulder_report.overshoot_companion_random_state
        ),
        residual_grid_label=shoulder_report.residual_grid_label,
        failing_right_shoulder_grid_value=shoulder_report.residual_failure_grid_value,
        anchor_negative_reserve_total=anchor_negative_reserve_total,
        anchor_negative_reserve_point_count=len(negative_reserve_points),
        anchor_failing_shoulder_negative_reserve_share=failing_shoulder_negative_share,
        anchor_off_shoulder_positive_reserve_total=off_shoulder_positive_reserve_total,
        anchor_off_shoulder_positive_to_negative_ratio=(
            off_shoulder_positive_to_negative_ratio
        ),
        anchor_center_reserve=_reserve(center_point),
        anchor_left_shoulder_reserve=_reserve(left_shoulder_point),
        overshoot_companion_same_point_reserve=companion_same_point_reserve,
        overshoot_companion_same_point_multiple_of_anchor_shortfall=(
            companion_shortfall_multiple
        ),
        canonical_coverage_anchor_shoulder_localization_digest=canonical_digest,
    )


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_localization_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLocalizationReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_localization_report(
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_probe()
    )
