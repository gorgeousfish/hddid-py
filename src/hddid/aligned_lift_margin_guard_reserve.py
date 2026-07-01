from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from hddid.aligned_lift_margin_guard_profile import (
    Phase7AlignedLiftMarginGuardProfileFocus,
    Phase7AlignedLiftMarginGuardProfileReport,
    run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_probe,
)


_METRIC_ORDER = {"gap": 0, "ratio": 1, "center_ratio": 2}


def _collapse_metric_slacks(
    focus: Phase7AlignedLiftMarginGuardProfileFocus,
) -> dict[str, float]:
    return {
        "gap": focus.normalized_collapse_gap_slack,
        "ratio": focus.normalized_collapse_ratio_slack,
        "center_ratio": focus.normalized_collapse_center_ratio_slack,
    }


def _reserve_components(
    *,
    threshold_crossing_slack: float,
    primary_boundary_slack: float,
) -> tuple[float, float, float, float, float]:
    boundary_shortfall_repair = abs(min(primary_boundary_slack, 0.0))
    positive_reserve = max(threshold_crossing_slack, 0.0)
    total_promotion = threshold_crossing_slack - primary_boundary_slack
    repair_share = boundary_shortfall_repair / total_promotion
    reserve_share = positive_reserve / total_promotion
    return (
        boundary_shortfall_repair,
        positive_reserve,
        total_promotion,
        repair_share,
        reserve_share,
    )


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardReserveMetric:
    metric: str
    threshold_crossing_slack: float
    primary_boundary_slack: float
    boundary_shortfall_repair: float
    positive_reserve: float
    total_promotion: float
    repair_share: float
    reserve_share: float

    def __post_init__(self) -> None:
        self.metric = str(self.metric).strip()
        self.threshold_crossing_slack = float(self.threshold_crossing_slack)
        self.primary_boundary_slack = float(self.primary_boundary_slack)
        self.boundary_shortfall_repair = float(self.boundary_shortfall_repair)
        self.positive_reserve = float(self.positive_reserve)
        self.total_promotion = float(self.total_promotion)
        self.repair_share = float(self.repair_share)
        self.reserve_share = float(self.reserve_share)

    def canonical_digest_line(self) -> str:
        return (
            f"- `{self.metric}`: repair `{self.boundary_shortfall_repair:.3f}` "
            f"({self.repair_share:.3%}), reserve `{self.positive_reserve:.3f}` "
            f"({self.reserve_share:.3%}), total promotion `{self.total_promotion:.3f}`"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "threshold_crossing_slack": self.threshold_crossing_slack,
            "primary_boundary_slack": self.primary_boundary_slack,
            "boundary_shortfall_repair": self.boundary_shortfall_repair,
            "positive_reserve": self.positive_reserve,
            "total_promotion": self.total_promotion,
            "repair_share": self.repair_share,
            "reserve_share": self.reserve_share,
            "canonical_digest_line": self.canonical_digest_line(),
        }


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardReserveReport:
    stage_label: str
    from_random_state: int
    to_random_state: int
    from_signature: str
    to_signature: str
    metric_budgets: tuple[Phase7AlignedLiftMarginGuardReserveMetric, ...]
    weakest_reserve_metric: str
    weakest_reserve_share: float
    strongest_reserve_metric: str
    strongest_reserve_share: float
    canonical_reserve_digest: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.from_random_state = int(self.from_random_state)
        self.to_random_state = int(self.to_random_state)
        self.from_signature = str(self.from_signature).strip()
        self.to_signature = str(self.to_signature).strip()
        self.metric_budgets = tuple(self.metric_budgets)
        self.weakest_reserve_metric = str(self.weakest_reserve_metric).strip()
        self.weakest_reserve_share = float(self.weakest_reserve_share)
        self.strongest_reserve_metric = str(self.strongest_reserve_metric).strip()
        self.strongest_reserve_share = float(self.strongest_reserve_share)
        self.canonical_reserve_digest = tuple(
            str(line).rstrip() for line in self.canonical_reserve_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "from_random_state": self.from_random_state,
            "to_random_state": self.to_random_state,
            "from_signature": self.from_signature,
            "to_signature": self.to_signature,
            "metric_budgets": [metric.to_dict() for metric in self.metric_budgets],
            "weakest_reserve_metric": self.weakest_reserve_metric,
            "weakest_reserve_share": self.weakest_reserve_share,
            "strongest_reserve_metric": self.strongest_reserve_metric,
            "strongest_reserve_share": self.strongest_reserve_share,
            "canonical_reserve_digest": list(self.canonical_reserve_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


def _build_metric_budget(
    metric: str,
    *,
    threshold_crossing_slack: float,
    primary_boundary_slack: float,
) -> Phase7AlignedLiftMarginGuardReserveMetric:
    (
        boundary_shortfall_repair,
        positive_reserve,
        total_promotion,
        repair_share,
        reserve_share,
    ) = _reserve_components(
        threshold_crossing_slack=threshold_crossing_slack,
        primary_boundary_slack=primary_boundary_slack,
    )
    return Phase7AlignedLiftMarginGuardReserveMetric(
        metric=metric,
        threshold_crossing_slack=threshold_crossing_slack,
        primary_boundary_slack=primary_boundary_slack,
        boundary_shortfall_repair=boundary_shortfall_repair,
        positive_reserve=positive_reserve,
        total_promotion=total_promotion,
        repair_share=repair_share,
        reserve_share=reserve_share,
    )


def build_phase7_nonparametric_source_level_aligned_lift_margin_guard_reserve_report(
    guard_profile_report: Phase7AlignedLiftMarginGuardProfileReport,
) -> Phase7AlignedLiftMarginGuardReserveReport:
    threshold_crossing_focus = guard_profile_report.threshold_crossing_focus
    primary_boundary_focus = guard_profile_report.primary_boundary_focus
    threshold_crossing_slacks = _collapse_metric_slacks(threshold_crossing_focus)
    primary_boundary_slacks = _collapse_metric_slacks(primary_boundary_focus)
    metric_budgets = tuple(
        _build_metric_budget(
            metric,
            threshold_crossing_slack=threshold_crossing_slacks[metric],
            primary_boundary_slack=primary_boundary_slacks[metric],
        )
        for metric in ("gap", "ratio", "center_ratio")
    )
    weakest_reserve = min(
        metric_budgets,
        key=lambda item: (item.reserve_share, _METRIC_ORDER[item.metric]),
    )
    strongest_reserve = max(
        metric_budgets,
        key=lambda item: (item.reserve_share, -_METRIC_ORDER[item.metric]),
    )
    canonical_reserve_digest = tuple(
        metric.canonical_digest_line() for metric in metric_budgets
    )
    return Phase7AlignedLiftMarginGuardReserveReport(
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-margin-guard-reserve-probe"
        ),
        from_random_state=primary_boundary_focus.random_state,
        to_random_state=threshold_crossing_focus.random_state,
        from_signature=primary_boundary_focus.dominant_mode_margin_signature,
        to_signature=threshold_crossing_focus.dominant_mode_margin_signature,
        metric_budgets=metric_budgets,
        weakest_reserve_metric=weakest_reserve.metric,
        weakest_reserve_share=weakest_reserve.reserve_share,
        strongest_reserve_metric=strongest_reserve.metric,
        strongest_reserve_share=strongest_reserve.reserve_share,
        canonical_reserve_digest=canonical_reserve_digest,
        recommendation_rationale=(
            "On the `308 -> 303` jump, `gap` spends most of its promotion budget "
            "repairing the boundary shortfall, so it remains the binding collapse "
            "guard; `center_ratio` spends almost all of its budget on positive "
            "reserve, so it explains overshoot but not the classification boundary."
        ),
    )


@lru_cache(maxsize=1)
def run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_reserve_probe() -> (
    Phase7AlignedLiftMarginGuardReserveReport
):
    return build_phase7_nonparametric_source_level_aligned_lift_margin_guard_reserve_report(
        run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_probe()
    )


def run_phase7_nonparametric_source_level_aligned_lift_margin_guard_reserve_probe() -> (
    Phase7AlignedLiftMarginGuardReserveReport
):
    return run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_reserve_probe()
