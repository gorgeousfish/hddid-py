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


def _guard_direction(*slacks: float) -> str:
    return "headroom" if all(slack >= 0.0 for slack in slacks) else "shortfall"


def _guard_fraction(direction: str, *pairs: tuple[float, float]) -> float:
    if direction == "headroom":
        candidates = [
            slack / threshold for slack, threshold in pairs if float(slack) >= 0.0
        ]
    else:
        candidates = [
            abs(slack) / threshold for slack, threshold in pairs if float(slack) < 0.0
        ]
    if not candidates:
        raise ValueError(
            "guard fraction requires at least one slack in the guard direction"
        )
    return min(candidates)


def _corridor_position(
    value: float, lower_threshold: float, upper_threshold: float
) -> float:
    return (value - lower_threshold) / (upper_threshold - lower_threshold)


@dataclass(slots=True)
class Phase7AlignedLiftMarginCorridorFocus:
    random_state: int
    dominant_mode_margin_signature: str
    gap_corridor_position: float
    ratio_corridor_position: float
    center_collapse_multiple: float
    collapse_guard_direction: str
    collapse_guard_fraction: float
    finite_guard_direction: str
    finite_guard_fraction: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.dominant_mode_margin_signature = str(
            self.dominant_mode_margin_signature
        ).strip()
        self.gap_corridor_position = float(self.gap_corridor_position)
        self.ratio_corridor_position = float(self.ratio_corridor_position)
        self.center_collapse_multiple = float(self.center_collapse_multiple)
        self.collapse_guard_direction = str(self.collapse_guard_direction).strip()
        self.collapse_guard_fraction = float(self.collapse_guard_fraction)
        self.finite_guard_direction = str(self.finite_guard_direction).strip()
        self.finite_guard_fraction = float(self.finite_guard_fraction)

    def canonical_digest_line(self) -> str:
        return (
            f"- `{self.random_state}` `{self.dominant_mode_margin_signature}`: "
            f"gap corridor `{self.gap_corridor_position:.3f}`, "
            f"ratio corridor `{self.ratio_corridor_position:.3f}`, "
            f"center multiple `{self.center_collapse_multiple:.3f}`, "
            f"collapse guard `{self.collapse_guard_direction} "
            f"{self.collapse_guard_fraction:.3f}`, "
            f"finite guard `{self.finite_guard_direction} "
            f"{self.finite_guard_fraction:.3f}`"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "random_state": self.random_state,
            "dominant_mode_margin_signature": self.dominant_mode_margin_signature,
            "gap_corridor_position": self.gap_corridor_position,
            "ratio_corridor_position": self.ratio_corridor_position,
            "center_collapse_multiple": self.center_collapse_multiple,
            "collapse_guard_direction": self.collapse_guard_direction,
            "collapse_guard_fraction": self.collapse_guard_fraction,
            "finite_guard_direction": self.finite_guard_direction,
            "finite_guard_fraction": self.finite_guard_fraction,
            "canonical_digest_line": self.canonical_digest_line(),
        }


@dataclass(slots=True)
class Phase7AlignedLiftMarginCorridorReport:
    stage_label: str
    threshold_crossing_focus: Phase7AlignedLiftMarginCorridorFocus
    primary_boundary_focus: Phase7AlignedLiftMarginCorridorFocus
    secondary_support_focus: Phase7AlignedLiftMarginCorridorFocus
    canonical_corridor_digest: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.canonical_corridor_digest = tuple(
            str(line).rstrip() for line in self.canonical_corridor_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "threshold_crossing_focus": self.threshold_crossing_focus.to_dict(),
            "primary_boundary_focus": self.primary_boundary_focus.to_dict(),
            "secondary_support_focus": self.secondary_support_focus.to_dict(),
            "canonical_corridor_digest": list(self.canonical_corridor_digest),
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
) -> Phase7AlignedLiftMarginCorridorFocus:
    collapse_guard_direction = _guard_direction(
        focus.collapse_gap_slack,
        focus.collapse_ratio_slack,
        focus.collapse_center_ratio_slack,
    )
    finite_guard_direction = _guard_direction(
        focus.finite_gap_slack,
        focus.finite_ratio_slack,
    )
    return Phase7AlignedLiftMarginCorridorFocus(
        random_state=focus.random_state,
        dominant_mode_margin_signature=focus.dominant_mode_margin_signature,
        gap_corridor_position=_corridor_position(
            focus.mean_dominant_minus_second_gap,
            finite_gap_threshold,
            collapse_gap_threshold,
        ),
        ratio_corridor_position=_corridor_position(
            focus.mean_dominant_to_second_ratio,
            finite_ratio_threshold,
            collapse_ratio_threshold,
        ),
        center_collapse_multiple=(
            focus.center_dominant_to_second_ratio / collapse_center_ratio_threshold
        ),
        collapse_guard_direction=collapse_guard_direction,
        collapse_guard_fraction=_guard_fraction(
            collapse_guard_direction,
            (focus.collapse_gap_slack, collapse_gap_threshold),
            (focus.collapse_ratio_slack, collapse_ratio_threshold),
            (focus.collapse_center_ratio_slack, collapse_center_ratio_threshold),
        ),
        finite_guard_direction=finite_guard_direction,
        finite_guard_fraction=_guard_fraction(
            finite_guard_direction,
            (focus.finite_gap_slack, finite_gap_threshold),
            (focus.finite_ratio_slack, finite_ratio_threshold),
        ),
    )


def build_phase7_nonparametric_source_level_aligned_lift_margin_corridor_report(
    slack_report: Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackReport,
) -> Phase7AlignedLiftMarginCorridorReport:
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
    canonical_corridor_digest = (
        threshold_crossing_focus.canonical_digest_line(),
        primary_boundary_focus.canonical_digest_line(),
        secondary_support_focus.canonical_digest_line(),
    )
    return Phase7AlignedLiftMarginCorridorReport(
        stage_label="phase7-nonparametric-source-level-aligned-lift-margin-corridor-probe",
        threshold_crossing_focus=threshold_crossing_focus,
        primary_boundary_focus=primary_boundary_focus,
        secondary_support_focus=secondary_support_focus,
        canonical_corridor_digest=canonical_corridor_digest,
        recommendation_rationale=(
            "`303` now overshoots both finite and collapse corridors on normalized "
            "gap/ratio scales; `308` sits only a small fraction inside the finite-to-"
            "collapse corridor while remaining materially below collapse on every "
            "required metric; `296` remains outside the finite corridor."
        ),
    )


@lru_cache(maxsize=1)
def run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_corridor_probe() -> (
    Phase7AlignedLiftMarginCorridorReport
):
    return build_phase7_nonparametric_source_level_aligned_lift_margin_corridor_report(
        run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_signature_slack_probe()
    )


def run_phase7_nonparametric_source_level_aligned_lift_margin_corridor_probe() -> (
    Phase7AlignedLiftMarginCorridorReport
):
    return run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_corridor_probe()
