from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_grid_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorGridReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_grid_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_window_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorWindowReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_window_probe,
)


def _format_grid_value(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return f"{value:.1f}"
    text = f"{value:.2f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_half_width(value: float) -> str:
    return f"{float(value):.2f}"


def _require_finite_values(message: str, values: np.ndarray | tuple[float, ...]) -> None:
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ValueError(message)


def _driver_signature(
    *,
    anchor_left_shoulder_covered: bool,
    anchor_right_shoulder_covered: bool,
    overshoot_companion_left_shoulder_covered: bool,
    overshoot_companion_right_shoulder_covered: bool,
) -> str:
    if (
        anchor_left_shoulder_covered
        and not anchor_right_shoulder_covered
        and overshoot_companion_left_shoulder_covered
        and overshoot_companion_right_shoulder_covered
    ):
        return "right-only-shoulder-coverage-break"
    return "mixed-window-shoulder-parity"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorWindowShoulderParityReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    center_grid_value: float
    left_shoulder_grid_value: float
    right_shoulder_grid_value: float
    symmetric_half_width: float
    anchor_left_shoulder_covered: bool
    anchor_right_shoulder_covered: bool
    overshoot_companion_left_shoulder_covered: bool
    overshoot_companion_right_shoulder_covered: bool
    anchor_shoulder_coverage_count: int
    overshoot_companion_shoulder_coverage_count: int
    anchor_shoulder_coverage_share: float
    companion_minus_anchor_shoulder_coverage_share: float
    driver_signature: str
    canonical_window_shoulder_parity_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.center_grid_value = float(self.center_grid_value)
        self.left_shoulder_grid_value = float(self.left_shoulder_grid_value)
        self.right_shoulder_grid_value = float(self.right_shoulder_grid_value)
        self.symmetric_half_width = float(self.symmetric_half_width)
        _require_finite_values(
            "window shoulder parity report requires finite grid geometry",
            (
                self.center_grid_value,
                self.left_shoulder_grid_value,
                self.right_shoulder_grid_value,
                self.symmetric_half_width,
            ),
        )
        self.anchor_left_shoulder_covered = bool(self.anchor_left_shoulder_covered)
        self.anchor_right_shoulder_covered = bool(self.anchor_right_shoulder_covered)
        self.overshoot_companion_left_shoulder_covered = bool(
            self.overshoot_companion_left_shoulder_covered
        )
        self.overshoot_companion_right_shoulder_covered = bool(
            self.overshoot_companion_right_shoulder_covered
        )
        self.anchor_shoulder_coverage_count = int(self.anchor_shoulder_coverage_count)
        self.overshoot_companion_shoulder_coverage_count = int(
            self.overshoot_companion_shoulder_coverage_count
        )
        self.anchor_shoulder_coverage_share = float(self.anchor_shoulder_coverage_share)
        self.companion_minus_anchor_shoulder_coverage_share = float(
            self.companion_minus_anchor_shoulder_coverage_share
        )
        _require_finite_values(
            "window shoulder parity report requires finite coverage shares",
            (
                self.anchor_shoulder_coverage_share,
                self.companion_minus_anchor_shoulder_coverage_share,
            ),
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_window_shoulder_parity_digest = tuple(
            str(line).rstrip() for line in self.canonical_window_shoulder_parity_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "center_grid_value": self.center_grid_value,
            "left_shoulder_grid_value": self.left_shoulder_grid_value,
            "right_shoulder_grid_value": self.right_shoulder_grid_value,
            "symmetric_half_width": self.symmetric_half_width,
            "anchor_left_shoulder_covered": self.anchor_left_shoulder_covered,
            "anchor_right_shoulder_covered": self.anchor_right_shoulder_covered,
            "overshoot_companion_left_shoulder_covered": (
                self.overshoot_companion_left_shoulder_covered
            ),
            "overshoot_companion_right_shoulder_covered": (
                self.overshoot_companion_right_shoulder_covered
            ),
            "anchor_shoulder_coverage_count": self.anchor_shoulder_coverage_count,
            "overshoot_companion_shoulder_coverage_count": (
                self.overshoot_companion_shoulder_coverage_count
            ),
            "anchor_shoulder_coverage_share": self.anchor_shoulder_coverage_share,
            "companion_minus_anchor_shoulder_coverage_share": (
                self.companion_minus_anchor_shoulder_coverage_share
            ),
            "driver_signature": self.driver_signature,
            "canonical_window_shoulder_parity_digest": list(
                self.canonical_window_shoulder_parity_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_window_shoulder_parity_report(
    *,
    grid_report: Phase7MonteCarloWideningPolicyCoverageAnchorGridReport,
    window_report: Phase7MonteCarloWideningPolicyCoverageAnchorWindowReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorWindowShoulderParityReport:
    if grid_report.policy_digest != window_report.policy_digest:
        raise ValueError("window shoulder parity probe requires a shared policy digest")
    if grid_report.binding_design != window_report.binding_design:
        raise ValueError(
            "window shoulder parity probe requires a shared binding design"
        )

    residual_replay = grid_report.replay(window_report.residual_anchor_preserving_label)
    grid = np.asarray(residual_replay.evaluation_grid, dtype=float)
    center = float(window_report.hotspot_center)
    _require_finite_values(
        "window shoulder parity probe requires finite evaluation grid values",
        grid,
    )
    _require_finite_values(
        "window shoulder parity probe requires a finite hotspot center",
        (center,),
    )

    left_values = grid[grid < center]
    right_values = grid[grid > center]
    residual_values = np.asarray(
        window_report.residual_uncovered_grid_values, dtype=float
    )
    _require_finite_values(
        "window shoulder parity probe requires finite residual shoulder values",
        residual_values,
    )
    if len(left_values) != 1 or len(right_values) != 1 or len(residual_values) != 1:
        raise ValueError(
            "window shoulder parity probe requires a symmetric single-point shoulder pair"
        )
    left_value = float(left_values[0])
    right_value = float(right_values[0])
    if not np.isclose(center - left_value, right_value - center):
        raise ValueError(
            "window shoulder parity probe requires a symmetric single-point shoulder pair"
        )
    if not np.isclose(residual_values[0], right_value):
        raise ValueError(
            "window shoulder parity probe requires a symmetric single-point shoulder pair"
        )

    left_index = int(np.where(np.isclose(grid, left_value))[0][0])
    right_index = int(np.where(np.isclose(grid, right_value))[0][0])

    anchor_left_covered = bool(
        residual_replay.coverage_anchor_slice.pointwise_coverage[left_index]
    )
    anchor_right_covered = bool(
        residual_replay.coverage_anchor_slice.pointwise_coverage[right_index]
    )
    overshoot_companion_left_covered = bool(
        residual_replay.overshoot_companion_slice.pointwise_coverage[left_index]
    )
    overshoot_companion_right_covered = bool(
        residual_replay.overshoot_companion_slice.pointwise_coverage[right_index]
    )
    anchor_count = int(anchor_left_covered) + int(anchor_right_covered)
    companion_count = int(overshoot_companion_left_covered) + int(
        overshoot_companion_right_covered
    )
    anchor_share = anchor_count / 2.0
    share_gap = (companion_count / 2.0) - anchor_share
    driver_signature = _driver_signature(
        anchor_left_shoulder_covered=anchor_left_covered,
        anchor_right_shoulder_covered=anchor_right_covered,
        overshoot_companion_left_shoulder_covered=overshoot_companion_left_covered,
        overshoot_companion_right_shoulder_covered=overshoot_companion_right_covered,
    )

    digest = (
        f"- symmetric anchor-preserving shoulder pair around `z = {_format_grid_value(center)}`: "
        f"left `z = {_format_grid_value(left_value)}` and right `z = {_format_grid_value(right_value)}` "
        f"both sit at half-width `{_format_half_width(right_value - center)}` on "
        f"`{window_report.residual_anchor_preserving_label}`",
        f"- coverage anchor seed `202` keeps left shoulder covered but drops the matched right shoulder, "
        f"so shoulder coverage splits to `{anchor_count}/2` despite a symmetric window",
        f"- overshoot companion seed `505` keeps both matched shoulders covered (`{companion_count}/2`), "
        "so the residual asymmetry stays specific to the anchor rather than the window template itself",
        f"- current Trigger 2 implication: `{driver_signature}`; next follow-up should stay on "
        "right-sided shoulder repair, not symmetric window widening",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorWindowShoulderParityReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-window-shoulder-parity-probe"
        ),
        policy_digest=grid_report.policy_digest,
        binding_design=grid_report.binding_design,
        window_label=window_report.residual_anchor_preserving_label,
        center_grid_value=center,
        left_shoulder_grid_value=left_value,
        right_shoulder_grid_value=right_value,
        symmetric_half_width=right_value - center,
        anchor_left_shoulder_covered=anchor_left_covered,
        anchor_right_shoulder_covered=anchor_right_covered,
        overshoot_companion_left_shoulder_covered=overshoot_companion_left_covered,
        overshoot_companion_right_shoulder_covered=overshoot_companion_right_covered,
        anchor_shoulder_coverage_count=anchor_count,
        overshoot_companion_shoulder_coverage_count=companion_count,
        anchor_shoulder_coverage_share=anchor_share,
        companion_minus_anchor_shoulder_coverage_share=share_gap,
        driver_signature=driver_signature,
        canonical_window_shoulder_parity_digest=digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_window_shoulder_parity_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorWindowShoulderParityReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_window_shoulder_parity_report(
        grid_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_grid_probe(),
        window_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_window_probe(),
    )
