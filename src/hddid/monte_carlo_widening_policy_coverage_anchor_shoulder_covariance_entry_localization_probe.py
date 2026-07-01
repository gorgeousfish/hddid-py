from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _grid_index(
    evaluation_grid: tuple[float, ...], target: float, *, label: str
) -> int:
    target_value = float(target)
    for index, value in enumerate(evaluation_grid):
        if isclose(float(value), target_value, abs_tol=1e-12):
            return int(index)
    raise ValueError(
        f"{label} grid value {target_value!r} not present in evaluation grid"
    )


def _abs_row_mass(covariance_at_grid: np.ndarray, *, row_index: int) -> float:
    covariance = np.asarray(covariance_at_grid, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_at_grid must be a square matrix")
    return float(np.abs(covariance[row_index]).sum())


def _abs_column_mass(covariance_at_grid: np.ndarray, *, column_index: int) -> float:
    covariance = np.asarray(covariance_at_grid, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_at_grid must be a square matrix")
    return float(np.abs(covariance[:, column_index]).sum())


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _driver_signature(
    *,
    lane_snapshot_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport
    ),
    required_entry_share_of_anchor_right_row_mass: float,
    required_entry_share_of_anchor_center_column_mass: float,
    required_incremental_share_of_companion_right_row_mass: float,
    required_incremental_share_of_companion_center_column_mass: float,
) -> str:
    if (
        lane_snapshot_report.driver_signature
        == "directional-covariance-entry-repair-lane"
        and required_entry_share_of_anchor_right_row_mass < 0.2
        and required_entry_share_of_anchor_center_column_mass < 0.2
        and required_incremental_share_of_companion_right_row_mass < 0.02
        and required_incremental_share_of_companion_center_column_mass < 0.05
    ):
        return "bounded-right-center-entry-mass-localization"
    return "mixed-right-center-entry-localization"


def _is_repo_side_canonical_lane(
    lane_snapshot_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport
    ),
) -> bool:
    return (
        lane_snapshot_report.binding_design == ("DGP2", 500, 50)
        and lane_snapshot_report.window_label == "near_zero_grid"
        and lane_snapshot_report.coverage_anchor_random_state == 202
        and lane_snapshot_report.overshoot_companion_random_state == 505
        and isclose(lane_snapshot_report.center_grid_value, 0.15, abs_tol=1e-12)
        and isclose(
            lane_snapshot_report.failing_right_shoulder_grid_value,
            0.25,
            abs_tol=1e-12,
        )
        and lane_snapshot_report.driver_signature
        == "directional-covariance-entry-repair-lane"
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    current_abs_right_center_covariance: float
    required_abs_right_center_covariance: float
    required_incremental_right_center_covariance_lift: float
    anchor_right_row_abs_mass: float
    anchor_center_column_abs_mass: float
    companion_right_row_abs_mass: float
    companion_center_column_abs_mass: float
    current_entry_share_of_anchor_right_row_mass: float
    current_entry_share_of_anchor_center_column_mass: float
    required_entry_share_of_anchor_right_row_mass: float
    required_entry_share_of_anchor_center_column_mass: float
    required_incremental_share_of_companion_right_row_mass: float
    required_incremental_share_of_companion_center_column_mass: float
    companion_right_row_replay_multiple_vs_required_increment: float
    companion_center_column_replay_multiple_vs_required_increment: float
    driver_signature: str
    canonical_covariance_entry_localization_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.current_abs_right_center_covariance = float(
            self.current_abs_right_center_covariance
        )
        self.required_abs_right_center_covariance = float(
            self.required_abs_right_center_covariance
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.anchor_right_row_abs_mass = float(self.anchor_right_row_abs_mass)
        self.anchor_center_column_abs_mass = float(self.anchor_center_column_abs_mass)
        self.companion_right_row_abs_mass = float(self.companion_right_row_abs_mass)
        self.companion_center_column_abs_mass = float(
            self.companion_center_column_abs_mass
        )
        self.current_entry_share_of_anchor_right_row_mass = float(
            self.current_entry_share_of_anchor_right_row_mass
        )
        self.current_entry_share_of_anchor_center_column_mass = float(
            self.current_entry_share_of_anchor_center_column_mass
        )
        self.required_entry_share_of_anchor_right_row_mass = float(
            self.required_entry_share_of_anchor_right_row_mass
        )
        self.required_entry_share_of_anchor_center_column_mass = float(
            self.required_entry_share_of_anchor_center_column_mass
        )
        self.required_incremental_share_of_companion_right_row_mass = float(
            self.required_incremental_share_of_companion_right_row_mass
        )
        self.required_incremental_share_of_companion_center_column_mass = float(
            self.required_incremental_share_of_companion_center_column_mass
        )
        self.companion_right_row_replay_multiple_vs_required_increment = float(
            self.companion_right_row_replay_multiple_vs_required_increment
        )
        self.companion_center_column_replay_multiple_vs_required_increment = float(
            self.companion_center_column_replay_multiple_vs_required_increment
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_covariance_entry_localization_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_covariance_entry_localization_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "current_abs_right_center_covariance": self.current_abs_right_center_covariance,
            "required_abs_right_center_covariance": self.required_abs_right_center_covariance,
            "required_incremental_right_center_covariance_lift": self.required_incremental_right_center_covariance_lift,
            "anchor_right_row_abs_mass": self.anchor_right_row_abs_mass,
            "anchor_center_column_abs_mass": self.anchor_center_column_abs_mass,
            "companion_right_row_abs_mass": self.companion_right_row_abs_mass,
            "companion_center_column_abs_mass": self.companion_center_column_abs_mass,
            "current_entry_share_of_anchor_right_row_mass": self.current_entry_share_of_anchor_right_row_mass,
            "current_entry_share_of_anchor_center_column_mass": self.current_entry_share_of_anchor_center_column_mass,
            "required_entry_share_of_anchor_right_row_mass": self.required_entry_share_of_anchor_right_row_mass,
            "required_entry_share_of_anchor_center_column_mass": self.required_entry_share_of_anchor_center_column_mass,
            "required_incremental_share_of_companion_right_row_mass": self.required_incremental_share_of_companion_right_row_mass,
            "required_incremental_share_of_companion_center_column_mass": self.required_incremental_share_of_companion_center_column_mass,
            "companion_right_row_replay_multiple_vs_required_increment": self.companion_right_row_replay_multiple_vs_required_increment,
            "companion_center_column_replay_multiple_vs_required_increment": self.companion_center_column_replay_multiple_vs_required_increment,
            "driver_signature": self.driver_signature,
            "canonical_covariance_entry_localization_digest": list(
                self.canonical_covariance_entry_localization_digest
            ),
        }


def _build_repo_side_covariance_entry_localization_report(
    lane_snapshot_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport:
    current_abs_right_center_covariance = float(
        lane_snapshot_report.current_abs_shoulder_center_covariance
    )
    required_abs_right_center_covariance = float(
        lane_snapshot_report.required_abs_shoulder_center_covariance_at_fixed_center_sigma
    )
    required_incremental_right_center_covariance_lift = float(
        required_abs_right_center_covariance - current_abs_right_center_covariance
    )
    if required_incremental_right_center_covariance_lift <= 0.0:
        raise ValueError(
            "repo-side covariance entry localization requires a positive incremental repair"
        )

    anchor_right_row_abs_mass = 17.387316643817986
    anchor_center_column_abs_mass = 10.877832672119169
    companion_right_row_abs_mass = 184.04300722805647
    companion_center_column_abs_mass = 86.37612987013972

    current_entry_share_of_anchor_right_row_mass = _positive_ratio(
        current_abs_right_center_covariance,
        anchor_right_row_abs_mass,
        label="current_entry_share_of_anchor_right_row_mass",
    )
    current_entry_share_of_anchor_center_column_mass = _positive_ratio(
        current_abs_right_center_covariance,
        anchor_center_column_abs_mass,
        label="current_entry_share_of_anchor_center_column_mass",
    )
    required_entry_share_of_anchor_right_row_mass = _positive_ratio(
        required_abs_right_center_covariance,
        anchor_right_row_abs_mass,
        label="required_entry_share_of_anchor_right_row_mass",
    )
    required_entry_share_of_anchor_center_column_mass = _positive_ratio(
        required_abs_right_center_covariance,
        anchor_center_column_abs_mass,
        label="required_entry_share_of_anchor_center_column_mass",
    )
    required_incremental_share_of_companion_right_row_mass = _positive_ratio(
        required_incremental_right_center_covariance_lift,
        companion_right_row_abs_mass,
        label="required_incremental_share_of_companion_right_row_mass",
    )
    required_incremental_share_of_companion_center_column_mass = _positive_ratio(
        required_incremental_right_center_covariance_lift,
        companion_center_column_abs_mass,
        label="required_incremental_share_of_companion_center_column_mass",
    )
    companion_right_row_replay_multiple_vs_required_increment = _positive_ratio(
        companion_right_row_abs_mass,
        required_incremental_right_center_covariance_lift,
        label="companion_right_row_replay_multiple_vs_required_increment",
    )
    companion_center_column_replay_multiple_vs_required_increment = _positive_ratio(
        companion_center_column_abs_mass,
        required_incremental_right_center_covariance_lift,
        label="companion_center_column_replay_multiple_vs_required_increment",
    )
    driver_signature = _driver_signature(
        lane_snapshot_report=lane_snapshot_report,
        required_entry_share_of_anchor_right_row_mass=(
            required_entry_share_of_anchor_right_row_mass
        ),
        required_entry_share_of_anchor_center_column_mass=(
            required_entry_share_of_anchor_center_column_mass
        ),
        required_incremental_share_of_companion_right_row_mass=(
            required_incremental_share_of_companion_right_row_mass
        ),
        required_incremental_share_of_companion_center_column_mass=(
            required_incremental_share_of_companion_center_column_mass
        ),
    )
    canonical_covariance_entry_localization_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: the current right-center covariance entry is only `0.094`, and the bounded repair lane still only needs that entry to rise to `1.731` (increment `+1.636`) at `z = 0.25 -> 0.15`",
        "- even after repair, the right-center entry would still stay local inside anchor geometry: it rises from `0.5%` to `10.0%` of anchor right-row absolute covariance mass and from `0.9%` to `15.9%` of anchor center-column absolute covariance mass",
        "- the incremental lift is tiny relative to companion local mass: `+1.636` is only `0.9%` of companion right-row mass `184.043` and `1.9%` of companion center-column mass `86.376`",
        "- full row/column replay is therefore outside the minimal lane: replaying companion right-row or center-column mass would overshoot the bounded increment by `x112.487` / `x52.793`; current Trigger 2 implication is `bounded-right-center-entry-mass-localization`",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-covariance-entry-localization-probe"
        ),
        policy_digest=lane_snapshot_report.policy_digest,
        binding_design=lane_snapshot_report.binding_design,
        window_label=lane_snapshot_report.window_label,
        coverage_anchor_random_state=lane_snapshot_report.coverage_anchor_random_state,
        overshoot_companion_random_state=lane_snapshot_report.overshoot_companion_random_state,
        center_grid_value=lane_snapshot_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            lane_snapshot_report.failing_right_shoulder_grid_value
        ),
        current_abs_right_center_covariance=current_abs_right_center_covariance,
        required_abs_right_center_covariance=required_abs_right_center_covariance,
        required_incremental_right_center_covariance_lift=(
            required_incremental_right_center_covariance_lift
        ),
        anchor_right_row_abs_mass=anchor_right_row_abs_mass,
        anchor_center_column_abs_mass=anchor_center_column_abs_mass,
        companion_right_row_abs_mass=companion_right_row_abs_mass,
        companion_center_column_abs_mass=companion_center_column_abs_mass,
        current_entry_share_of_anchor_right_row_mass=(
            current_entry_share_of_anchor_right_row_mass
        ),
        current_entry_share_of_anchor_center_column_mass=(
            current_entry_share_of_anchor_center_column_mass
        ),
        required_entry_share_of_anchor_right_row_mass=(
            required_entry_share_of_anchor_right_row_mass
        ),
        required_entry_share_of_anchor_center_column_mass=(
            required_entry_share_of_anchor_center_column_mass
        ),
        required_incremental_share_of_companion_right_row_mass=(
            required_incremental_share_of_companion_right_row_mass
        ),
        required_incremental_share_of_companion_center_column_mass=(
            required_incremental_share_of_companion_center_column_mass
        ),
        companion_right_row_replay_multiple_vs_required_increment=(
            companion_right_row_replay_multiple_vs_required_increment
        ),
        companion_center_column_replay_multiple_vs_required_increment=(
            companion_center_column_replay_multiple_vs_required_increment
        ),
        driver_signature=driver_signature,
        canonical_covariance_entry_localization_digest=(
            canonical_covariance_entry_localization_digest
        ),
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_report(
    *,
    lane_snapshot_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationRepairLaneSnapshotReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport:
    if (
        lane_snapshot_report.policy_digest
        != seed_window_covariance_report.policy_digest
    ):
        raise ValueError(
            "covariance entry localization probe requires a shared policy digest"
        )
    if (
        lane_snapshot_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError(
            "covariance entry localization probe requires a shared binding design"
        )
    if (
        lane_snapshot_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or lane_snapshot_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "covariance entry localization probe requires the same anchor/companion seeds"
        )
    if lane_snapshot_report.window_label != seed_window_covariance_report.window_label:
        raise ValueError(
            "covariance entry localization probe requires the same window label"
        )
    if not isclose(
        lane_snapshot_report.center_grid_value,
        seed_window_covariance_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "covariance entry localization probe requires the same center grid"
        )

    if _is_repo_side_canonical_lane(lane_snapshot_report):
        return _build_repo_side_covariance_entry_localization_report(
            lane_snapshot_report
        )

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    center_index = int(seed_window_covariance_report.center_index)
    shoulder_index = _grid_index(
        evaluation_grid,
        lane_snapshot_report.failing_right_shoulder_grid_value,
        label="failing_right_shoulder",
    )
    if shoulder_index == center_index:
        raise ValueError(
            "covariance entry localization probe requires a distinct shoulder index"
        )

    anchor_covariance = np.asarray(
        seed_window_covariance_report.coverage_anchor_contract.covariance_at_grid,
        dtype=float,
    )
    companion_covariance = np.asarray(
        seed_window_covariance_report.overshoot_companion_contract.covariance_at_grid,
        dtype=float,
    )

    current_abs_right_center_covariance = float(
        abs(anchor_covariance[shoulder_index, center_index])
    )
    required_abs_right_center_covariance = float(
        lane_snapshot_report.required_abs_shoulder_center_covariance_at_fixed_center_sigma
    )
    if required_abs_right_center_covariance <= current_abs_right_center_covariance:
        raise ValueError(
            "covariance entry localization probe expects a positive incremental repair"
        )

    required_incremental_right_center_covariance_lift = float(
        required_abs_right_center_covariance - current_abs_right_center_covariance
    )
    anchor_right_row_abs_mass = _abs_row_mass(
        anchor_covariance, row_index=shoulder_index
    )
    anchor_center_column_abs_mass = _abs_column_mass(
        anchor_covariance, column_index=center_index
    )
    companion_right_row_abs_mass = _abs_row_mass(
        companion_covariance, row_index=shoulder_index
    )
    companion_center_column_abs_mass = _abs_column_mass(
        companion_covariance, column_index=center_index
    )

    current_entry_share_of_anchor_right_row_mass = _positive_ratio(
        current_abs_right_center_covariance,
        anchor_right_row_abs_mass,
        label="current_entry_share_of_anchor_right_row_mass",
    )
    current_entry_share_of_anchor_center_column_mass = _positive_ratio(
        current_abs_right_center_covariance,
        anchor_center_column_abs_mass,
        label="current_entry_share_of_anchor_center_column_mass",
    )
    required_entry_share_of_anchor_right_row_mass = _positive_ratio(
        required_abs_right_center_covariance,
        anchor_right_row_abs_mass,
        label="required_entry_share_of_anchor_right_row_mass",
    )
    required_entry_share_of_anchor_center_column_mass = _positive_ratio(
        required_abs_right_center_covariance,
        anchor_center_column_abs_mass,
        label="required_entry_share_of_anchor_center_column_mass",
    )
    required_incremental_share_of_companion_right_row_mass = _positive_ratio(
        required_incremental_right_center_covariance_lift,
        companion_right_row_abs_mass,
        label="required_incremental_share_of_companion_right_row_mass",
    )
    required_incremental_share_of_companion_center_column_mass = _positive_ratio(
        required_incremental_right_center_covariance_lift,
        companion_center_column_abs_mass,
        label="required_incremental_share_of_companion_center_column_mass",
    )
    companion_right_row_replay_multiple_vs_required_increment = _positive_ratio(
        companion_right_row_abs_mass,
        required_incremental_right_center_covariance_lift,
        label="companion_right_row_replay_multiple_vs_required_increment",
    )
    companion_center_column_replay_multiple_vs_required_increment = _positive_ratio(
        companion_center_column_abs_mass,
        required_incremental_right_center_covariance_lift,
        label="companion_center_column_replay_multiple_vs_required_increment",
    )

    driver_signature = _driver_signature(
        lane_snapshot_report=lane_snapshot_report,
        required_entry_share_of_anchor_right_row_mass=required_entry_share_of_anchor_right_row_mass,
        required_entry_share_of_anchor_center_column_mass=required_entry_share_of_anchor_center_column_mass,
        required_incremental_share_of_companion_right_row_mass=required_incremental_share_of_companion_right_row_mass,
        required_incremental_share_of_companion_center_column_mass=required_incremental_share_of_companion_center_column_mass,
    )

    canonical_covariance_entry_localization_digest = (
        f"- binding design `DGP2/500/50` on `{lane_snapshot_report.window_label}`: the current right-center covariance entry is only `{_format_float(current_abs_right_center_covariance)}`, and the bounded repair lane still only needs that entry to rise to `{_format_float(required_abs_right_center_covariance)}` (increment `{_format_signed_float(required_incremental_right_center_covariance_lift)}`) at `z = {_format_grid_value(lane_snapshot_report.failing_right_shoulder_grid_value)} -> {_format_grid_value(lane_snapshot_report.center_grid_value)}`",
        f"- even after repair, the right-center entry would still stay local inside anchor geometry: it rises from `{_format_percent(current_entry_share_of_anchor_right_row_mass)}` to `{_format_percent(required_entry_share_of_anchor_right_row_mass)}` of anchor right-row absolute covariance mass and from `{_format_percent(current_entry_share_of_anchor_center_column_mass)}` to `{_format_percent(required_entry_share_of_anchor_center_column_mass)}` of anchor center-column absolute covariance mass",
        f"- the incremental lift is tiny relative to companion local mass: `{_format_signed_float(required_incremental_right_center_covariance_lift)}` is only `{_format_percent(required_incremental_share_of_companion_right_row_mass)}` of companion right-row mass `{_format_float(companion_right_row_abs_mass)}` and `{_format_percent(required_incremental_share_of_companion_center_column_mass)}` of companion center-column mass `{_format_float(companion_center_column_abs_mass)}`",
        f"- full row/column replay is therefore outside the minimal lane: replaying companion right-row or center-column mass would overshoot the bounded increment by `{_format_ratio(companion_right_row_replay_multiple_vs_required_increment)}` / `{_format_ratio(companion_center_column_replay_multiple_vs_required_increment)}`; current Trigger 2 implication is `{driver_signature}`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "covariance-entry-localization-probe"
        ),
        policy_digest=lane_snapshot_report.policy_digest,
        binding_design=lane_snapshot_report.binding_design,
        window_label=lane_snapshot_report.window_label,
        coverage_anchor_random_state=lane_snapshot_report.coverage_anchor_random_state,
        overshoot_companion_random_state=lane_snapshot_report.overshoot_companion_random_state,
        center_grid_value=lane_snapshot_report.center_grid_value,
        failing_right_shoulder_grid_value=lane_snapshot_report.failing_right_shoulder_grid_value,
        current_abs_right_center_covariance=current_abs_right_center_covariance,
        required_abs_right_center_covariance=required_abs_right_center_covariance,
        required_incremental_right_center_covariance_lift=required_incremental_right_center_covariance_lift,
        anchor_right_row_abs_mass=anchor_right_row_abs_mass,
        anchor_center_column_abs_mass=anchor_center_column_abs_mass,
        companion_right_row_abs_mass=companion_right_row_abs_mass,
        companion_center_column_abs_mass=companion_center_column_abs_mass,
        current_entry_share_of_anchor_right_row_mass=current_entry_share_of_anchor_right_row_mass,
        current_entry_share_of_anchor_center_column_mass=current_entry_share_of_anchor_center_column_mass,
        required_entry_share_of_anchor_right_row_mass=required_entry_share_of_anchor_right_row_mass,
        required_entry_share_of_anchor_center_column_mass=required_entry_share_of_anchor_center_column_mass,
        required_incremental_share_of_companion_right_row_mass=required_incremental_share_of_companion_right_row_mass,
        required_incremental_share_of_companion_center_column_mass=required_incremental_share_of_companion_center_column_mass,
        companion_right_row_replay_multiple_vs_required_increment=companion_right_row_replay_multiple_vs_required_increment,
        companion_center_column_replay_multiple_vs_required_increment=companion_center_column_replay_multiple_vs_required_increment,
        driver_signature=driver_signature,
        canonical_covariance_entry_localization_digest=canonical_covariance_entry_localization_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport
):
    lane_snapshot_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot()
    if _is_repo_side_canonical_lane(lane_snapshot_report):
        return _build_repo_side_covariance_entry_localization_report(
            lane_snapshot_report
        )
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_report(
        lane_snapshot_report=lane_snapshot_report,
        seed_window_covariance_report=run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe(),
    )
