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


def _weakest_metric(
    metric_separations: tuple["Phase7AlignedLiftMarginGuardMetricSeparation", ...],
    attribute: str,
) -> "Phase7AlignedLiftMarginGuardMetricSeparation":
    return min(
        metric_separations,
        key=lambda item: (getattr(item, attribute), _METRIC_ORDER[item.metric]),
    )


def _strongest_metric(
    metric_separations: tuple["Phase7AlignedLiftMarginGuardMetricSeparation", ...],
    attribute: str,
) -> "Phase7AlignedLiftMarginGuardMetricSeparation":
    return max(
        metric_separations,
        key=lambda item: (getattr(item, attribute), -_METRIC_ORDER[item.metric]),
    )


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardMetricSeparation:
    metric: str
    threshold_crossing_slack: float
    primary_boundary_slack: float
    secondary_support_slack: float
    threshold_crossing_minus_primary_boundary: float
    primary_boundary_minus_secondary_support: float

    def __post_init__(self) -> None:
        self.metric = str(self.metric).strip()
        self.threshold_crossing_slack = float(self.threshold_crossing_slack)
        self.primary_boundary_slack = float(self.primary_boundary_slack)
        self.secondary_support_slack = float(self.secondary_support_slack)
        self.threshold_crossing_minus_primary_boundary = float(
            self.threshold_crossing_minus_primary_boundary
        )
        self.primary_boundary_minus_secondary_support = float(
            self.primary_boundary_minus_secondary_support
        )

    def canonical_digest_line(self) -> str:
        return (
            f"- `{self.metric}`: `303-308 {self.threshold_crossing_minus_primary_boundary:+.3f}`, "
            f"`308-296 {self.primary_boundary_minus_secondary_support:+.3f}`"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "threshold_crossing_slack": self.threshold_crossing_slack,
            "primary_boundary_slack": self.primary_boundary_slack,
            "secondary_support_slack": self.secondary_support_slack,
            "threshold_crossing_minus_primary_boundary": (
                self.threshold_crossing_minus_primary_boundary
            ),
            "primary_boundary_minus_secondary_support": (
                self.primary_boundary_minus_secondary_support
            ),
            "canonical_digest_line": self.canonical_digest_line(),
        }


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardMetricSeparationReport:
    stage_label: str
    metric_separations: tuple[Phase7AlignedLiftMarginGuardMetricSeparation, ...]
    weakest_crossing_separation_metric: str
    weakest_crossing_separation: float
    strongest_crossing_separation_metric: str
    strongest_crossing_separation: float
    weakest_boundary_separation_metric: str
    weakest_boundary_separation: float
    strongest_boundary_separation_metric: str
    strongest_boundary_separation: float
    canonical_metric_digest: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.metric_separations = tuple(self.metric_separations)
        self.weakest_crossing_separation_metric = str(
            self.weakest_crossing_separation_metric
        ).strip()
        self.weakest_crossing_separation = float(self.weakest_crossing_separation)
        self.strongest_crossing_separation_metric = str(
            self.strongest_crossing_separation_metric
        ).strip()
        self.strongest_crossing_separation = float(self.strongest_crossing_separation)
        self.weakest_boundary_separation_metric = str(
            self.weakest_boundary_separation_metric
        ).strip()
        self.weakest_boundary_separation = float(self.weakest_boundary_separation)
        self.strongest_boundary_separation_metric = str(
            self.strongest_boundary_separation_metric
        ).strip()
        self.strongest_boundary_separation = float(self.strongest_boundary_separation)
        self.canonical_metric_digest = tuple(
            str(line).rstrip() for line in self.canonical_metric_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "metric_separations": [
                separation.to_dict() for separation in self.metric_separations
            ],
            "weakest_crossing_separation_metric": (
                self.weakest_crossing_separation_metric
            ),
            "weakest_crossing_separation": self.weakest_crossing_separation,
            "strongest_crossing_separation_metric": (
                self.strongest_crossing_separation_metric
            ),
            "strongest_crossing_separation": self.strongest_crossing_separation,
            "weakest_boundary_separation_metric": (
                self.weakest_boundary_separation_metric
            ),
            "weakest_boundary_separation": self.weakest_boundary_separation,
            "strongest_boundary_separation_metric": (
                self.strongest_boundary_separation_metric
            ),
            "strongest_boundary_separation": self.strongest_boundary_separation,
            "canonical_metric_digest": list(self.canonical_metric_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


def _build_metric_separation(
    metric: str,
    *,
    threshold_crossing_slack: float,
    primary_boundary_slack: float,
    secondary_support_slack: float,
) -> Phase7AlignedLiftMarginGuardMetricSeparation:
    return Phase7AlignedLiftMarginGuardMetricSeparation(
        metric=metric,
        threshold_crossing_slack=threshold_crossing_slack,
        primary_boundary_slack=primary_boundary_slack,
        secondary_support_slack=secondary_support_slack,
        threshold_crossing_minus_primary_boundary=(
            threshold_crossing_slack - primary_boundary_slack
        ),
        primary_boundary_minus_secondary_support=(
            primary_boundary_slack - secondary_support_slack
        ),
    )


def build_phase7_nonparametric_source_level_aligned_lift_margin_guard_metric_separation_report(
    guard_profile_report: Phase7AlignedLiftMarginGuardProfileReport,
) -> Phase7AlignedLiftMarginGuardMetricSeparationReport:
    threshold_crossing_slacks = _collapse_metric_slacks(
        guard_profile_report.threshold_crossing_focus
    )
    primary_boundary_slacks = _collapse_metric_slacks(
        guard_profile_report.primary_boundary_focus
    )
    secondary_support_slacks = _collapse_metric_slacks(
        guard_profile_report.secondary_support_focus
    )
    metric_separations = tuple(
        _build_metric_separation(
            metric,
            threshold_crossing_slack=threshold_crossing_slacks[metric],
            primary_boundary_slack=primary_boundary_slacks[metric],
            secondary_support_slack=secondary_support_slacks[metric],
        )
        for metric in ("gap", "ratio", "center_ratio")
    )
    weakest_crossing = _weakest_metric(
        metric_separations, "threshold_crossing_minus_primary_boundary"
    )
    strongest_crossing = _strongest_metric(
        metric_separations, "threshold_crossing_minus_primary_boundary"
    )
    weakest_boundary = _weakest_metric(
        metric_separations, "primary_boundary_minus_secondary_support"
    )
    strongest_boundary = _strongest_metric(
        metric_separations, "primary_boundary_minus_secondary_support"
    )
    canonical_metric_digest = tuple(
        separation.canonical_digest_line() for separation in metric_separations
    )
    return Phase7AlignedLiftMarginGuardMetricSeparationReport(
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-margin-guard-metric-separation-probe"
        ),
        metric_separations=metric_separations,
        weakest_crossing_separation_metric=weakest_crossing.metric,
        weakest_crossing_separation=(
            weakest_crossing.threshold_crossing_minus_primary_boundary
        ),
        strongest_crossing_separation_metric=strongest_crossing.metric,
        strongest_crossing_separation=(
            strongest_crossing.threshold_crossing_minus_primary_boundary
        ),
        weakest_boundary_separation_metric=weakest_boundary.metric,
        weakest_boundary_separation=(
            weakest_boundary.primary_boundary_minus_secondary_support
        ),
        strongest_boundary_separation_metric=strongest_boundary.metric,
        strongest_boundary_separation=(
            strongest_boundary.primary_boundary_minus_secondary_support
        ),
        canonical_metric_digest=canonical_metric_digest,
        recommendation_rationale=(
            "Every collapse metric now separates `303` from `308`, not just the "
            "binding guard: the weakest `303-308` gap is still +0.438 on `gap`, "
            "while `308-296` shrinks to a weakest +0.058 on `center_ratio`. That "
            "per-metric ladder makes the regime change live on the `308 -> 303` jump, "
            "not on a gradual deepening from `296` upward."
        ),
    )


@lru_cache(maxsize=1)
def run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_metric_separation_probe() -> (
    Phase7AlignedLiftMarginGuardMetricSeparationReport
):
    return build_phase7_nonparametric_source_level_aligned_lift_margin_guard_metric_separation_report(
        run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_probe()
    )


def run_phase7_nonparametric_source_level_aligned_lift_margin_guard_metric_separation_probe() -> (
    Phase7AlignedLiftMarginGuardMetricSeparationReport
):
    return run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_metric_separation_probe()
