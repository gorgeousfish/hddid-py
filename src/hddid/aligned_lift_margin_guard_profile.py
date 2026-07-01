from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from hddid.aligned_lift_dominant_mode_margin import (
    run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_signature_slack_probe,
)
from hddid.validation import (
    Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackFocus,
    Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackReport,
)


_METRIC_ORDER = {"gap": 0, "ratio": 1, "center_ratio": 2}


def _binding_metric(metric_slacks: dict[str, float]) -> str:
    return min(
        metric_slacks,
        key=lambda metric: (abs(metric_slacks[metric]), _METRIC_ORDER[metric]),
    )


def _breadth(metric_slacks: dict[str, float]) -> int:
    return sum(slack >= 0.0 for slack in metric_slacks.values())


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardProfileFocus:
    random_state: int
    dominant_mode_margin_signature: str
    normalized_collapse_gap_slack: float
    normalized_collapse_ratio_slack: float
    normalized_collapse_center_ratio_slack: float
    normalized_finite_gap_slack: float
    normalized_finite_ratio_slack: float
    collapse_binding_metric: str
    collapse_binding_slack: float
    collapse_breadth: int
    finite_binding_metric: str
    finite_binding_slack: float
    finite_breadth: int

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.dominant_mode_margin_signature = str(
            self.dominant_mode_margin_signature
        ).strip()
        self.normalized_collapse_gap_slack = float(self.normalized_collapse_gap_slack)
        self.normalized_collapse_ratio_slack = float(
            self.normalized_collapse_ratio_slack
        )
        self.normalized_collapse_center_ratio_slack = float(
            self.normalized_collapse_center_ratio_slack
        )
        self.normalized_finite_gap_slack = float(self.normalized_finite_gap_slack)
        self.normalized_finite_ratio_slack = float(self.normalized_finite_ratio_slack)
        self.collapse_binding_metric = str(self.collapse_binding_metric).strip()
        self.collapse_binding_slack = float(self.collapse_binding_slack)
        self.collapse_breadth = int(self.collapse_breadth)
        self.finite_binding_metric = str(self.finite_binding_metric).strip()
        self.finite_binding_slack = float(self.finite_binding_slack)
        self.finite_breadth = int(self.finite_breadth)

    def canonical_digest_line(self) -> str:
        return (
            f"- `{self.random_state}` `{self.dominant_mode_margin_signature}`: "
            f"collapse breadth `{self.collapse_breadth}/3`, "
            f"collapse binding `{self.collapse_binding_metric} "
            f"{self.collapse_binding_slack:+.3f}`, "
            f"finite breadth `{self.finite_breadth}/2`, "
            f"finite binding `{self.finite_binding_metric} "
            f"{self.finite_binding_slack:+.3f}`"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "random_state": self.random_state,
            "dominant_mode_margin_signature": self.dominant_mode_margin_signature,
            "normalized_collapse_gap_slack": self.normalized_collapse_gap_slack,
            "normalized_collapse_ratio_slack": self.normalized_collapse_ratio_slack,
            "normalized_collapse_center_ratio_slack": (
                self.normalized_collapse_center_ratio_slack
            ),
            "normalized_finite_gap_slack": self.normalized_finite_gap_slack,
            "normalized_finite_ratio_slack": self.normalized_finite_ratio_slack,
            "collapse_binding_metric": self.collapse_binding_metric,
            "collapse_binding_slack": self.collapse_binding_slack,
            "collapse_breadth": self.collapse_breadth,
            "finite_binding_metric": self.finite_binding_metric,
            "finite_binding_slack": self.finite_binding_slack,
            "finite_breadth": self.finite_breadth,
            "canonical_digest_line": self.canonical_digest_line(),
        }


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardProfileReport:
    stage_label: str
    threshold_crossing_focus: Phase7AlignedLiftMarginGuardProfileFocus
    primary_boundary_focus: Phase7AlignedLiftMarginGuardProfileFocus
    secondary_support_focus: Phase7AlignedLiftMarginGuardProfileFocus
    canonical_guard_digest: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.canonical_guard_digest = tuple(
            str(line).rstrip() for line in self.canonical_guard_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "threshold_crossing_focus": self.threshold_crossing_focus.to_dict(),
            "primary_boundary_focus": self.primary_boundary_focus.to_dict(),
            "secondary_support_focus": self.secondary_support_focus.to_dict(),
            "canonical_guard_digest": list(self.canonical_guard_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


def _build_focus(
    focus: Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackFocus,
    *,
    collapse_gap_threshold: float,
    collapse_ratio_threshold: float,
    collapse_center_ratio_threshold: float,
    finite_gap_threshold: float,
    finite_ratio_threshold: float,
) -> Phase7AlignedLiftMarginGuardProfileFocus:
    collapse_metric_slacks = {
        "gap": focus.collapse_gap_slack / collapse_gap_threshold,
        "ratio": focus.collapse_ratio_slack / collapse_ratio_threshold,
        "center_ratio": (
            focus.collapse_center_ratio_slack / collapse_center_ratio_threshold
        ),
    }
    finite_metric_slacks = {
        "gap": focus.finite_gap_slack / finite_gap_threshold,
        "ratio": focus.finite_ratio_slack / finite_ratio_threshold,
    }
    collapse_binding_metric = _binding_metric(collapse_metric_slacks)
    finite_binding_metric = _binding_metric(finite_metric_slacks)
    return Phase7AlignedLiftMarginGuardProfileFocus(
        random_state=focus.random_state,
        dominant_mode_margin_signature=focus.dominant_mode_margin_signature,
        normalized_collapse_gap_slack=collapse_metric_slacks["gap"],
        normalized_collapse_ratio_slack=collapse_metric_slacks["ratio"],
        normalized_collapse_center_ratio_slack=collapse_metric_slacks["center_ratio"],
        normalized_finite_gap_slack=finite_metric_slacks["gap"],
        normalized_finite_ratio_slack=finite_metric_slacks["ratio"],
        collapse_binding_metric=collapse_binding_metric,
        collapse_binding_slack=collapse_metric_slacks[collapse_binding_metric],
        collapse_breadth=_breadth(collapse_metric_slacks),
        finite_binding_metric=finite_binding_metric,
        finite_binding_slack=finite_metric_slacks[finite_binding_metric],
        finite_breadth=_breadth(finite_metric_slacks),
    )


def build_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_report(
    slack_report: Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackReport,
) -> Phase7AlignedLiftMarginGuardProfileReport:
    threshold_crossing_focus = _build_focus(
        slack_report.threshold_crossing_focus,
        collapse_gap_threshold=slack_report.collapse_gap_threshold,
        collapse_ratio_threshold=slack_report.collapse_ratio_threshold,
        collapse_center_ratio_threshold=slack_report.collapse_center_ratio_threshold,
        finite_gap_threshold=slack_report.finite_gap_threshold,
        finite_ratio_threshold=slack_report.finite_ratio_threshold,
    )
    primary_boundary_focus = _build_focus(
        slack_report.primary_boundary_focus,
        collapse_gap_threshold=slack_report.collapse_gap_threshold,
        collapse_ratio_threshold=slack_report.collapse_ratio_threshold,
        collapse_center_ratio_threshold=slack_report.collapse_center_ratio_threshold,
        finite_gap_threshold=slack_report.finite_gap_threshold,
        finite_ratio_threshold=slack_report.finite_ratio_threshold,
    )
    secondary_support_focus = _build_focus(
        slack_report.secondary_support_focus,
        collapse_gap_threshold=slack_report.collapse_gap_threshold,
        collapse_ratio_threshold=slack_report.collapse_ratio_threshold,
        collapse_center_ratio_threshold=slack_report.collapse_center_ratio_threshold,
        finite_gap_threshold=slack_report.finite_gap_threshold,
        finite_ratio_threshold=slack_report.finite_ratio_threshold,
    )
    canonical_guard_digest = (
        threshold_crossing_focus.canonical_digest_line(),
        primary_boundary_focus.canonical_digest_line(),
        secondary_support_focus.canonical_digest_line(),
    )
    return Phase7AlignedLiftMarginGuardProfileReport(
        stage_label="phase7-nonparametric-source-level-aligned-lift-margin-guard-profile-probe",
        threshold_crossing_focus=threshold_crossing_focus,
        primary_boundary_focus=primary_boundary_focus,
        secondary_support_focus=secondary_support_focus,
        canonical_guard_digest=canonical_guard_digest,
        recommendation_rationale=(
            "`303` is no longer just 'inside the corridor': it clears all three "
            "collapse metrics, and even its weakest normalized guard remains above "
            "zero on the gap dimension. `308` still clears the finite pair but fails "
            "all collapse metrics, while `296` clears neither finite metric."
        ),
    )


@lru_cache(maxsize=1)
def run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_probe() -> (
    Phase7AlignedLiftMarginGuardProfileReport
):
    return build_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_report(
        run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_signature_slack_probe()
    )


def run_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_probe() -> (
    Phase7AlignedLiftMarginGuardProfileReport
):
    return run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_probe()
