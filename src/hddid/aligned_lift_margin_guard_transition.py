from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from hddid.aligned_lift_margin_guard_profile import (
    Phase7AlignedLiftMarginGuardProfileFocus,
    Phase7AlignedLiftMarginGuardProfileReport,
    run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_probe,
)


_METRIC_ORDER = {"gap": 0, "ratio": 1, "center_ratio": 2}
_FINITE_METRICS = ("gap", "ratio")
_COLLAPSE_METRICS = ("gap", "ratio", "center_ratio")


def _ordered_metric_flips(
    before: dict[str, float], after: dict[str, float]
) -> tuple[str, ...]:
    flipped = []
    for metric, before_value in before.items():
        after_value = after[metric]
        if before_value < 0.0 <= after_value:
            flipped.append(metric)
    return tuple(sorted(flipped, key=_METRIC_ORDER.__getitem__))


def _finite_metric_slacks(
    focus: Phase7AlignedLiftMarginGuardProfileFocus,
) -> dict[str, float]:
    return {
        "gap": focus.normalized_finite_gap_slack,
        "ratio": focus.normalized_finite_ratio_slack,
    }


def _collapse_metric_slacks(
    focus: Phase7AlignedLiftMarginGuardProfileFocus,
) -> dict[str, float]:
    return {
        "gap": focus.normalized_collapse_gap_slack,
        "ratio": focus.normalized_collapse_ratio_slack,
        "center_ratio": focus.normalized_collapse_center_ratio_slack,
    }


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardTransitionStep:
    from_random_state: int
    to_random_state: int
    from_signature: str
    to_signature: str
    newly_nonnegative_finite_metrics: tuple[str, ...]
    newly_nonnegative_collapse_metrics: tuple[str, ...]
    finite_breadth_delta: int
    collapse_breadth_delta: int
    finite_binding_metric_before: str
    finite_binding_metric_after: str
    collapse_binding_metric_before: str
    collapse_binding_metric_after: str
    finite_binding_slack_delta: float
    collapse_binding_slack_delta: float

    def __post_init__(self) -> None:
        self.from_random_state = int(self.from_random_state)
        self.to_random_state = int(self.to_random_state)
        self.from_signature = str(self.from_signature).strip()
        self.to_signature = str(self.to_signature).strip()
        self.newly_nonnegative_finite_metrics = tuple(
            str(metric).strip() for metric in self.newly_nonnegative_finite_metrics
        )
        self.newly_nonnegative_collapse_metrics = tuple(
            str(metric).strip() for metric in self.newly_nonnegative_collapse_metrics
        )
        self.finite_breadth_delta = int(self.finite_breadth_delta)
        self.collapse_breadth_delta = int(self.collapse_breadth_delta)
        self.finite_binding_metric_before = str(
            self.finite_binding_metric_before
        ).strip()
        self.finite_binding_metric_after = str(self.finite_binding_metric_after).strip()
        self.collapse_binding_metric_before = str(
            self.collapse_binding_metric_before
        ).strip()
        self.collapse_binding_metric_after = str(
            self.collapse_binding_metric_after
        ).strip()
        self.finite_binding_slack_delta = float(self.finite_binding_slack_delta)
        self.collapse_binding_slack_delta = float(self.collapse_binding_slack_delta)

    def canonical_digest_line(self) -> str:
        finite = ", ".join(self.newly_nonnegative_finite_metrics) or "none"
        collapse = ", ".join(self.newly_nonnegative_collapse_metrics) or "none"
        return (
            f"- `{self.from_random_state} -> {self.to_random_state}`: "
            f"finite promotions `{finite}`, collapse promotions `{collapse}`, "
            f"finite breadth delta `{self.finite_breadth_delta:+d}`, "
            f"collapse breadth delta `{self.collapse_breadth_delta:+d}`, "
            f"finite binding delta `{self.finite_binding_slack_delta:+.3f}`, "
            f"collapse binding delta `{self.collapse_binding_slack_delta:+.3f}`"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "from_random_state": self.from_random_state,
            "to_random_state": self.to_random_state,
            "from_signature": self.from_signature,
            "to_signature": self.to_signature,
            "newly_nonnegative_finite_metrics": list(
                self.newly_nonnegative_finite_metrics
            ),
            "newly_nonnegative_collapse_metrics": list(
                self.newly_nonnegative_collapse_metrics
            ),
            "finite_breadth_delta": self.finite_breadth_delta,
            "collapse_breadth_delta": self.collapse_breadth_delta,
            "finite_binding_metric_before": self.finite_binding_metric_before,
            "finite_binding_metric_after": self.finite_binding_metric_after,
            "collapse_binding_metric_before": self.collapse_binding_metric_before,
            "collapse_binding_metric_after": self.collapse_binding_metric_after,
            "finite_binding_slack_delta": self.finite_binding_slack_delta,
            "collapse_binding_slack_delta": self.collapse_binding_slack_delta,
            "canonical_digest_line": self.canonical_digest_line(),
        }


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardTransitionReport:
    stage_label: str
    shoulder_to_boundary_step: Phase7AlignedLiftMarginGuardTransitionStep
    boundary_to_collapse_step: Phase7AlignedLiftMarginGuardTransitionStep
    canonical_transition_digest: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.canonical_transition_digest = tuple(
            str(line).rstrip() for line in self.canonical_transition_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "shoulder_to_boundary_step": self.shoulder_to_boundary_step.to_dict(),
            "boundary_to_collapse_step": self.boundary_to_collapse_step.to_dict(),
            "canonical_transition_digest": list(self.canonical_transition_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


def _build_transition_step(
    before: Phase7AlignedLiftMarginGuardProfileFocus,
    after: Phase7AlignedLiftMarginGuardProfileFocus,
) -> Phase7AlignedLiftMarginGuardTransitionStep:
    before_finite = _finite_metric_slacks(before)
    after_finite = _finite_metric_slacks(after)
    before_collapse = _collapse_metric_slacks(before)
    after_collapse = _collapse_metric_slacks(after)
    return Phase7AlignedLiftMarginGuardTransitionStep(
        from_random_state=before.random_state,
        to_random_state=after.random_state,
        from_signature=before.dominant_mode_margin_signature,
        to_signature=after.dominant_mode_margin_signature,
        newly_nonnegative_finite_metrics=_ordered_metric_flips(
            before_finite, after_finite
        ),
        newly_nonnegative_collapse_metrics=_ordered_metric_flips(
            before_collapse, after_collapse
        ),
        finite_breadth_delta=after.finite_breadth - before.finite_breadth,
        collapse_breadth_delta=after.collapse_breadth - before.collapse_breadth,
        finite_binding_metric_before=before.finite_binding_metric,
        finite_binding_metric_after=after.finite_binding_metric,
        collapse_binding_metric_before=before.collapse_binding_metric,
        collapse_binding_metric_after=after.collapse_binding_metric,
        finite_binding_slack_delta=(
            after.finite_binding_slack - before.finite_binding_slack
        ),
        collapse_binding_slack_delta=(
            after.collapse_binding_slack - before.collapse_binding_slack
        ),
    )


def build_phase7_nonparametric_source_level_aligned_lift_margin_guard_transition_report(
    guard_profile_report: Phase7AlignedLiftMarginGuardProfileReport,
) -> Phase7AlignedLiftMarginGuardTransitionReport:
    shoulder_to_boundary_step = _build_transition_step(
        guard_profile_report.secondary_support_focus,
        guard_profile_report.primary_boundary_focus,
    )
    boundary_to_collapse_step = _build_transition_step(
        guard_profile_report.primary_boundary_focus,
        guard_profile_report.threshold_crossing_focus,
    )
    canonical_transition_digest = (
        shoulder_to_boundary_step.canonical_digest_line(),
        boundary_to_collapse_step.canonical_digest_line(),
    )
    return Phase7AlignedLiftMarginGuardTransitionReport(
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-margin-guard-transition-probe"
        ),
        shoulder_to_boundary_step=shoulder_to_boundary_step,
        boundary_to_collapse_step=boundary_to_collapse_step,
        canonical_transition_digest=canonical_transition_digest,
        recommendation_rationale=(
            "`296 -> 308` promotes the finite pair only, while `308 -> 303` is the "
            "first discrete rung that flips the full collapse trio. That promotion "
            "ladder shows why `303` is not just a deeper boundary member but a "
            "categorically different collapse state."
        ),
    )


@lru_cache(maxsize=1)
def run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_transition_probe() -> (
    Phase7AlignedLiftMarginGuardTransitionReport
):
    return build_phase7_nonparametric_source_level_aligned_lift_margin_guard_transition_report(
        run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_probe()
    )


def run_phase7_nonparametric_source_level_aligned_lift_margin_guard_transition_probe() -> (
    Phase7AlignedLiftMarginGuardTransitionReport
):
    return run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_transition_probe()
