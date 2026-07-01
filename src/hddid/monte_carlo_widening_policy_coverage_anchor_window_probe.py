from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_grid_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorGridReport,
    Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_grid_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return f"{value:.1f}"
    text = f"{value:.2f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def _format_grid(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(_format_grid_value(value) for value in values) + "]"


def _coverage_count(
    replay: Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay,
) -> int:
    return int(np.count_nonzero(replay.coverage_anchor_slice.pointwise_coverage))


def _grid_size(replay: Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay) -> int:
    return int(replay.coverage_anchor_slice.evaluation_grid.shape[0])


def _contains_hotspot(
    replay: Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay,
    hotspot_center: float,
) -> bool:
    return bool(
        np.any(np.isclose(replay.coverage_anchor_slice.evaluation_grid, hotspot_center))
    )


def _is_full_coverage(
    replay: Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay,
) -> bool:
    return _coverage_count(replay) == _grid_size(replay) and int(
        np.count_nonzero(replay.overshoot_companion_slice.pointwise_coverage)
    ) == int(replay.overshoot_companion_slice.evaluation_grid.shape[0])


def _half_width(
    replay: Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay,
    hotspot_center: float,
) -> float:
    return float(
        np.max(
            np.abs(replay.coverage_anchor_slice.evaluation_grid - float(hotspot_center))
        )
    )


def _uncovered_grid_values(
    replay: Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay,
) -> tuple[float, ...]:
    coverage = replay.coverage_anchor_slice.pointwise_coverage
    grid = replay.coverage_anchor_slice.evaluation_grid
    return tuple(float(value) for value in grid[np.logical_not(coverage)])


def _classify_failure_side(
    uncovered_grid_values: tuple[float, ...],
    hotspot_center: float,
) -> str:
    if not uncovered_grid_values:
        return "no-residual-failure"

    residual = np.asarray(uncovered_grid_values, dtype=float)
    if np.all(residual > hotspot_center):
        return "right-shoulder"
    if np.all(residual < hotspot_center):
        return "left-shoulder"
    if np.any(np.isclose(residual, hotspot_center)):
        return "center-anchor"
    return "mixed-window"


def _render_failure_side(label: str) -> str:
    return str(label).replace("-", " ")


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorWindowReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    hotspot_center: float
    anchor_preserving_labels: tuple[str, ...]
    full_coverage_anchor_preserving_labels: tuple[str, ...]
    residual_anchor_preserving_label: str
    residual_uncovered_grid_values: tuple[float, ...]
    residual_failure_side: str
    narrowest_full_coverage_anchor_preserving_label: str
    narrowest_full_coverage_window: tuple[float, ...]
    narrowest_full_coverage_half_width: float
    widest_full_coverage_anchor_preserving_half_width: float
    canonical_coverage_anchor_window_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.hotspot_center = float(self.hotspot_center)
        self.anchor_preserving_labels = tuple(
            str(label).strip() for label in self.anchor_preserving_labels
        )
        self.full_coverage_anchor_preserving_labels = tuple(
            str(label).strip() for label in self.full_coverage_anchor_preserving_labels
        )
        self.residual_anchor_preserving_label = str(
            self.residual_anchor_preserving_label
        ).strip()
        self.residual_uncovered_grid_values = tuple(
            float(value) for value in self.residual_uncovered_grid_values
        )
        self.residual_failure_side = str(self.residual_failure_side).strip()
        self.narrowest_full_coverage_anchor_preserving_label = str(
            self.narrowest_full_coverage_anchor_preserving_label
        ).strip()
        self.narrowest_full_coverage_window = tuple(
            float(value) for value in self.narrowest_full_coverage_window
        )
        self.narrowest_full_coverage_half_width = float(
            self.narrowest_full_coverage_half_width
        )
        self.widest_full_coverage_anchor_preserving_half_width = float(
            self.widest_full_coverage_anchor_preserving_half_width
        )
        self.canonical_coverage_anchor_window_digest = tuple(
            str(line).rstrip() for line in self.canonical_coverage_anchor_window_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "hotspot_center": self.hotspot_center,
            "anchor_preserving_labels": list(self.anchor_preserving_labels),
            "full_coverage_anchor_preserving_labels": list(
                self.full_coverage_anchor_preserving_labels
            ),
            "residual_anchor_preserving_label": self.residual_anchor_preserving_label,
            "residual_uncovered_grid_values": list(self.residual_uncovered_grid_values),
            "residual_failure_side": self.residual_failure_side,
            "narrowest_full_coverage_anchor_preserving_label": (
                self.narrowest_full_coverage_anchor_preserving_label
            ),
            "narrowest_full_coverage_window": list(self.narrowest_full_coverage_window),
            "narrowest_full_coverage_half_width": (
                self.narrowest_full_coverage_half_width
            ),
            "widest_full_coverage_anchor_preserving_half_width": (
                self.widest_full_coverage_anchor_preserving_half_width
            ),
            "canonical_coverage_anchor_window_digest": list(
                self.canonical_coverage_anchor_window_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_window_report(
    *,
    grid_report: Phase7MonteCarloWideningPolicyCoverageAnchorGridReport,
    hotspot_center: float = 0.15,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorWindowReport:
    center = float(hotspot_center)
    anchor_preserving_replays = tuple(
        replay for replay in grid_report.replays if _contains_hotspot(replay, center)
    )
    if not anchor_preserving_replays:
        raise ValueError(
            "coverage anchor window probe requires at least one anchor-preserving replay"
        )

    full_coverage_anchor_preserving_replays = tuple(
        replay for replay in anchor_preserving_replays if _is_full_coverage(replay)
    )
    if not full_coverage_anchor_preserving_replays:
        raise ValueError(
            "coverage anchor window probe requires at least one "
            "anchor-preserving full-coverage witness"
        )

    partial_anchor_preserving_replays = tuple(
        replay for replay in anchor_preserving_replays if not _is_full_coverage(replay)
    )
    if partial_anchor_preserving_replays:
        residual_replay = max(
            partial_anchor_preserving_replays,
            key=lambda replay: (
                _coverage_count(replay),
                -_half_width(replay, center),
                replay.grid_label,
            ),
        )
    else:
        residual_replay = min(
            anchor_preserving_replays,
            key=lambda replay: (
                _half_width(replay, center),
                replay.grid_label,
            ),
        )

    uncovered_grid_values = _uncovered_grid_values(residual_replay)
    residual_failure_side = _classify_failure_side(uncovered_grid_values, center)

    narrowest_full_coverage_replay = min(
        full_coverage_anchor_preserving_replays,
        key=lambda replay: (
            _half_width(replay, center),
            replay.grid_label,
        ),
    )
    widest_full_coverage_replay = max(
        full_coverage_anchor_preserving_replays,
        key=lambda replay: (
            _half_width(replay, center),
            replay.grid_label,
        ),
    )
    widest_success_half_width = _half_width(widest_full_coverage_replay, center)

    anchor_preserving_labels = tuple(
        replay.grid_label for replay in anchor_preserving_replays
    )
    full_coverage_anchor_preserving_labels = tuple(
        replay.grid_label for replay in full_coverage_anchor_preserving_replays
    )
    residual_left_values = tuple(
        float(value)
        for value in residual_replay.coverage_anchor_slice.evaluation_grid[
            residual_replay.coverage_anchor_slice.pointwise_coverage
        ]
        if value < center
    )
    residual_left_value = residual_left_values[-1] if residual_left_values else None
    residual_right_value = uncovered_grid_values[0] if uncovered_grid_values else None

    canonical_digest = (
        "- anchor-preserving replays around "
        f"`z = {_format_grid_value(center)}`: "
        + ", ".join(f"`{label}`" for label in anchor_preserving_labels)
        + "; only "
        + " and ".join(f"`{label}`" for label in full_coverage_anchor_preserving_labels)
        + f" reach `{_grid_size(narrowest_full_coverage_replay)}/"
        f"{_grid_size(narrowest_full_coverage_replay)}` coverage",
        "- residual anchor-preserving failure stays on "
        f"`{residual_replay.grid_label}` {_render_failure_side(residual_failure_side)} "
        f"`z = {_format_grid_value(residual_right_value)}`; left shoulder "
        f"`z = {_format_grid_value(residual_left_value)}` already covers, so current "
        "miss is no longer center-only",
        "- narrowest full-coverage anchor-preserving window is "
        f"`{narrowest_full_coverage_replay.grid_label}` "
        f"(`{_format_grid(narrowest_full_coverage_replay.evaluation_grid)}`, "
        f"half-width `{_format_float(_half_width(narrowest_full_coverage_replay, center))}`); "
        "widest successful anchor-preserving window is "
        f"`{widest_full_coverage_replay.grid_label}` "
        f"(half-width `{_format_float(widest_success_half_width)}`)",
        "- next Trigger 2 follow-up should treat "
        f"`{grid_report.binding_design[0]}/{grid_report.binding_design[1]}/{grid_report.binding_design[2]}` "
        "as local-window calibration debt around "
        f"`z = {_format_grid_value(center)}`, with residual right-shoulder stress "
        "beyond `+0.05`, not as generic quantile or global-grid tuning",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorWindowReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-window-probe",
        policy_digest=grid_report.policy_digest,
        binding_design=grid_report.binding_design,
        hotspot_center=center,
        anchor_preserving_labels=anchor_preserving_labels,
        full_coverage_anchor_preserving_labels=(full_coverage_anchor_preserving_labels),
        residual_anchor_preserving_label=residual_replay.grid_label,
        residual_uncovered_grid_values=uncovered_grid_values,
        residual_failure_side=residual_failure_side,
        narrowest_full_coverage_anchor_preserving_label=(
            narrowest_full_coverage_replay.grid_label
        ),
        narrowest_full_coverage_window=narrowest_full_coverage_replay.evaluation_grid,
        narrowest_full_coverage_half_width=_half_width(
            narrowest_full_coverage_replay, center
        ),
        widest_full_coverage_anchor_preserving_half_width=widest_success_half_width,
        canonical_coverage_anchor_window_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_window_probe(
    *,
    hotspot_center: float = 0.15,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorWindowReport:
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_window_report(
        grid_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_grid_probe(),
        hotspot_center=hotspot_center,
    )
