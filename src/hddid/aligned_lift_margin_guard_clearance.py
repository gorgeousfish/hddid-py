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


def _ordered_collapse_items(
    focus: Phase7AlignedLiftMarginGuardProfileFocus,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            _collapse_metric_slacks(focus).items(),
            key=lambda item: (-item[1], _METRIC_ORDER[item[0]]),
        )
    )


def _binding_metric(metric_slacks: dict[str, float]) -> str:
    return min(
        metric_slacks,
        key=lambda metric: (abs(metric_slacks[metric]), _METRIC_ORDER[metric]),
    )


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardClearanceFocus:
    random_state: int
    dominant_mode_margin_signature: str
    ordered_collapse_metrics: tuple[str, ...]
    ordered_collapse_slacks: tuple[float, ...]
    binding_metric: str
    binding_slack: float
    leading_metric: str
    leading_slack: float
    tail_metric: str
    tail_slack: float
    positive_metric_count: int
    negative_metric_count: int
    collapse_clearance_range: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.dominant_mode_margin_signature = str(
            self.dominant_mode_margin_signature
        ).strip()
        self.ordered_collapse_metrics = tuple(
            str(metric).strip() for metric in self.ordered_collapse_metrics
        )
        self.ordered_collapse_slacks = tuple(
            float(slack) for slack in self.ordered_collapse_slacks
        )
        self.binding_metric = str(self.binding_metric).strip()
        self.binding_slack = float(self.binding_slack)
        self.leading_metric = str(self.leading_metric).strip()
        self.leading_slack = float(self.leading_slack)
        self.tail_metric = str(self.tail_metric).strip()
        self.tail_slack = float(self.tail_slack)
        self.positive_metric_count = int(self.positive_metric_count)
        self.negative_metric_count = int(self.negative_metric_count)
        self.collapse_clearance_range = float(self.collapse_clearance_range)

    def canonical_digest_line(self) -> str:
        ordered_stack = " > ".join(
            f"{metric} {slack:+.3f}"
            for metric, slack in zip(
                self.ordered_collapse_metrics, self.ordered_collapse_slacks
            )
        )
        return (
            f"- `{self.random_state}` `{self.dominant_mode_margin_signature}`: "
            f"collapse stack `{ordered_stack}`, "
            f"binding `{self.binding_metric} {self.binding_slack:+.3f}`, "
            f"clearance range `{self.collapse_clearance_range:.3f}`"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "random_state": self.random_state,
            "dominant_mode_margin_signature": self.dominant_mode_margin_signature,
            "ordered_collapse_metrics": list(self.ordered_collapse_metrics),
            "ordered_collapse_slacks": list(self.ordered_collapse_slacks),
            "binding_metric": self.binding_metric,
            "binding_slack": self.binding_slack,
            "leading_metric": self.leading_metric,
            "leading_slack": self.leading_slack,
            "tail_metric": self.tail_metric,
            "tail_slack": self.tail_slack,
            "positive_metric_count": self.positive_metric_count,
            "negative_metric_count": self.negative_metric_count,
            "collapse_clearance_range": self.collapse_clearance_range,
            "canonical_digest_line": self.canonical_digest_line(),
        }


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardClearanceReport:
    stage_label: str
    threshold_crossing_focus: Phase7AlignedLiftMarginGuardClearanceFocus
    primary_boundary_focus: Phase7AlignedLiftMarginGuardClearanceFocus
    secondary_support_focus: Phase7AlignedLiftMarginGuardClearanceFocus
    binding_clearance_gap_to_primary_boundary: float
    primary_boundary_binding_gap_to_secondary_support: float
    canonical_clearance_digest: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.binding_clearance_gap_to_primary_boundary = float(
            self.binding_clearance_gap_to_primary_boundary
        )
        self.primary_boundary_binding_gap_to_secondary_support = float(
            self.primary_boundary_binding_gap_to_secondary_support
        )
        self.canonical_clearance_digest = tuple(
            str(line).rstrip() for line in self.canonical_clearance_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "threshold_crossing_focus": self.threshold_crossing_focus.to_dict(),
            "primary_boundary_focus": self.primary_boundary_focus.to_dict(),
            "secondary_support_focus": self.secondary_support_focus.to_dict(),
            "binding_clearance_gap_to_primary_boundary": (
                self.binding_clearance_gap_to_primary_boundary
            ),
            "primary_boundary_binding_gap_to_secondary_support": (
                self.primary_boundary_binding_gap_to_secondary_support
            ),
            "canonical_clearance_digest": list(self.canonical_clearance_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


def _build_focus(
    focus: Phase7AlignedLiftMarginGuardProfileFocus,
) -> Phase7AlignedLiftMarginGuardClearanceFocus:
    metric_slacks = _collapse_metric_slacks(focus)
    ordered_items = _ordered_collapse_items(focus)
    binding_metric = _binding_metric(metric_slacks)
    return Phase7AlignedLiftMarginGuardClearanceFocus(
        random_state=focus.random_state,
        dominant_mode_margin_signature=focus.dominant_mode_margin_signature,
        ordered_collapse_metrics=tuple(metric for metric, _ in ordered_items),
        ordered_collapse_slacks=tuple(slack for _, slack in ordered_items),
        binding_metric=binding_metric,
        binding_slack=metric_slacks[binding_metric],
        leading_metric=ordered_items[0][0],
        leading_slack=ordered_items[0][1],
        tail_metric=ordered_items[-1][0],
        tail_slack=ordered_items[-1][1],
        positive_metric_count=sum(slack >= 0.0 for slack in metric_slacks.values()),
        negative_metric_count=sum(slack < 0.0 for slack in metric_slacks.values()),
        collapse_clearance_range=max(metric_slacks.values())
        - min(metric_slacks.values()),
    )


def build_phase7_nonparametric_source_level_aligned_lift_margin_guard_clearance_report(
    guard_profile_report: Phase7AlignedLiftMarginGuardProfileReport,
) -> Phase7AlignedLiftMarginGuardClearanceReport:
    threshold_crossing_focus = _build_focus(
        guard_profile_report.threshold_crossing_focus
    )
    primary_boundary_focus = _build_focus(guard_profile_report.primary_boundary_focus)
    secondary_support_focus = _build_focus(guard_profile_report.secondary_support_focus)
    canonical_clearance_digest = (
        threshold_crossing_focus.canonical_digest_line(),
        primary_boundary_focus.canonical_digest_line(),
        secondary_support_focus.canonical_digest_line(),
    )
    return Phase7AlignedLiftMarginGuardClearanceReport(
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-margin-guard-clearance-probe"
        ),
        threshold_crossing_focus=threshold_crossing_focus,
        primary_boundary_focus=primary_boundary_focus,
        secondary_support_focus=secondary_support_focus,
        binding_clearance_gap_to_primary_boundary=(
            threshold_crossing_focus.binding_slack
            - primary_boundary_focus.binding_slack
        ),
        primary_boundary_binding_gap_to_secondary_support=(
            primary_boundary_focus.binding_slack - secondary_support_focus.binding_slack
        ),
        canonical_clearance_digest=canonical_clearance_digest,
        recommendation_rationale=(
            "`303` clears all three collapse metrics in descending order "
            "`center_ratio > ratio > gap`, and even its weakest collapse guard "
            "still sits 0.438 normalized units above `308`'s strongest collapse "
            "metric. `308` remains the best sub-zero boundary, while `296` sits "
            "another 0.272 normalized units deeper into collapse shortfall."
        ),
    )


@lru_cache(maxsize=1)
def run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_clearance_probe() -> (
    Phase7AlignedLiftMarginGuardClearanceReport
):
    return build_phase7_nonparametric_source_level_aligned_lift_margin_guard_clearance_report(
        run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_probe()
    )


def run_phase7_nonparametric_source_level_aligned_lift_margin_guard_clearance_probe() -> (
    Phase7AlignedLiftMarginGuardClearanceReport
):
    return run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_clearance_probe()
