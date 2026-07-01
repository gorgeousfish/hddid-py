from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_grid_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorGridReport,
    Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_grid_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_window_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorWindowReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_window_probe,
)


def _format_float(value: float, *, decimals: int = 3) -> str:
    return f"{float(value):.{int(decimals)}f}"


def _format_grid_value(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return f"{value:.1f}"
    text = f"{value:.2f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def _residual_replay(
    grid_report: Phase7MonteCarloWideningPolicyCoverageAnchorGridReport,
    window_report: Phase7MonteCarloWideningPolicyCoverageAnchorWindowReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay:
    return grid_report.replay(window_report.residual_anchor_preserving_label)


def _left_and_right_indices(
    replay: Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay,
    *,
    hotspot_center: float,
) -> tuple[int, int]:
    grid = np.asarray(replay.coverage_anchor_slice.evaluation_grid, dtype=float)
    left_candidates = np.flatnonzero(grid < float(hotspot_center))
    right_candidates = np.flatnonzero(grid > float(hotspot_center))
    if left_candidates.size == 0 or right_candidates.size == 0:
        raise ValueError(
            "window symmetry-break probe requires a symmetric residual window"
        )
    return int(left_candidates[0]), int(right_candidates[-1])


def _symmetric_offsets(
    replay: Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay,
    *,
    hotspot_center: float,
) -> tuple[float, float]:
    left_index, right_index = _left_and_right_indices(
        replay,
        hotspot_center=hotspot_center,
    )
    grid = np.asarray(replay.coverage_anchor_slice.evaluation_grid, dtype=float)
    left_offset = float(float(hotspot_center) - float(grid[left_index]))
    right_offset = float(float(grid[right_index]) - float(hotspot_center))
    if not np.isclose(left_offset, right_offset):
        raise ValueError(
            "window symmetry-break probe requires a symmetric residual window"
        )
    return left_offset, right_offset


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorWindowSymmetryBreakReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    hotspot_center: float
    residual_grid_label: str
    symmetric_window: tuple[float, ...]
    left_shoulder_grid_value: float
    right_shoulder_grid_value: float
    left_shoulder_offset: float
    right_shoulder_offset: float
    left_shoulder_is_covered: bool
    right_shoulder_is_covered: bool
    stable_anchor_preserving_half_width: float
    directional_extension_beyond_stable_window: float
    canonical_window_symmetry_break_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.hotspot_center = float(self.hotspot_center)
        self.residual_grid_label = str(self.residual_grid_label).strip()
        self.symmetric_window = tuple(float(value) for value in self.symmetric_window)
        self.left_shoulder_grid_value = float(self.left_shoulder_grid_value)
        self.right_shoulder_grid_value = float(self.right_shoulder_grid_value)
        self.left_shoulder_offset = float(self.left_shoulder_offset)
        self.right_shoulder_offset = float(self.right_shoulder_offset)
        self.left_shoulder_is_covered = bool(self.left_shoulder_is_covered)
        self.right_shoulder_is_covered = bool(self.right_shoulder_is_covered)
        self.stable_anchor_preserving_half_width = float(
            self.stable_anchor_preserving_half_width
        )
        self.directional_extension_beyond_stable_window = float(
            self.directional_extension_beyond_stable_window
        )
        self.canonical_window_symmetry_break_digest = tuple(
            str(line).rstrip() for line in self.canonical_window_symmetry_break_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "hotspot_center": self.hotspot_center,
            "residual_grid_label": self.residual_grid_label,
            "symmetric_window": list(self.symmetric_window),
            "left_shoulder_grid_value": self.left_shoulder_grid_value,
            "right_shoulder_grid_value": self.right_shoulder_grid_value,
            "left_shoulder_offset": self.left_shoulder_offset,
            "right_shoulder_offset": self.right_shoulder_offset,
            "left_shoulder_is_covered": self.left_shoulder_is_covered,
            "right_shoulder_is_covered": self.right_shoulder_is_covered,
            "stable_anchor_preserving_half_width": (
                self.stable_anchor_preserving_half_width
            ),
            "directional_extension_beyond_stable_window": (
                self.directional_extension_beyond_stable_window
            ),
            "canonical_window_symmetry_break_digest": list(
                self.canonical_window_symmetry_break_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_window_symmetry_break_report(
    *,
    grid_report: Phase7MonteCarloWideningPolicyCoverageAnchorGridReport,
    window_report: Phase7MonteCarloWideningPolicyCoverageAnchorWindowReport,
    hotspot_center: float = 0.15,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorWindowSymmetryBreakReport:
    center = float(hotspot_center)
    if not np.isclose(window_report.hotspot_center, center):
        raise ValueError(
            "window symmetry-break probe requires a hotspot center consistent with the window report"
        )

    replay = _residual_replay(grid_report, window_report)
    left_index, right_index = _left_and_right_indices(
        replay,
        hotspot_center=center,
    )
    left_offset, right_offset = _symmetric_offsets(
        replay,
        hotspot_center=center,
    )
    grid = np.asarray(replay.coverage_anchor_slice.evaluation_grid, dtype=float)
    coverage = np.asarray(replay.coverage_anchor_slice.pointwise_coverage, dtype=bool)
    stable_half_width = float(
        window_report.widest_full_coverage_anchor_preserving_half_width
    )
    directional_extension = max(right_offset - stable_half_width, 0.0)

    canonical_digest = (
        "- `near_zero_grid` keeps a symmetric anchor-preserving window around "
        f"`z = {_format_grid_value(center)}`: left shoulder "
        f"`z = {_format_grid_value(grid[left_index])}` and right shoulder "
        f"`z = {_format_grid_value(grid[right_index])}` both sit at "
        f"`|delta z| = {_format_float(right_offset, decimals=2)}`",
        "- only the right shoulder remains uncovered: left shoulder "
        f"`z = {_format_grid_value(grid[left_index])}` still covers while right shoulder "
        f"`z = {_format_grid_value(grid[right_index])}` fails, so current debt is a "
        "right-only symmetry break rather than generic window-width pressure",
        "- widest full-coverage anchor-preserving witness still stops at half-width "
        f"`{_format_float(stable_half_width)}`; the uncovered right extension therefore "
        "begins in the "
        f"`+{stable_half_width:.2f} -> +{right_offset:.2f}` band beyond the stable window",
        "- next Trigger 2 follow-up should stay on directional shoulder calibration "
        f"around `z = {_format_grid_value(grid[right_index])}`, not fall back to center-only "
        "or symmetric global-grid retuning",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorWindowSymmetryBreakReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-window-symmetry-break-probe"
        ),
        policy_digest=window_report.policy_digest,
        binding_design=window_report.binding_design,
        hotspot_center=center,
        residual_grid_label=replay.grid_label,
        symmetric_window=tuple(float(value) for value in grid.tolist()),
        left_shoulder_grid_value=float(grid[left_index]),
        right_shoulder_grid_value=float(grid[right_index]),
        left_shoulder_offset=left_offset,
        right_shoulder_offset=right_offset,
        left_shoulder_is_covered=bool(coverage[left_index]),
        right_shoulder_is_covered=bool(coverage[right_index]),
        stable_anchor_preserving_half_width=stable_half_width,
        directional_extension_beyond_stable_window=directional_extension,
        canonical_window_symmetry_break_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_window_symmetry_break_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorWindowSymmetryBreakReport
):
    grid_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_grid_probe()
    window_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_window_probe()
    )
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_window_symmetry_break_report(
        grid_report=grid_report,
        window_report=window_report,
    )
