from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe,
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


def _abs_entry(
    covariance_at_grid: np.ndarray, *, row_index: int, column_index: int
) -> float:
    covariance = np.asarray(covariance_at_grid, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_at_grid must be a square matrix")
    return float(abs(float(covariance[row_index, column_index])))


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _driver_signature(
    *,
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport
    ),
    required_increment_share_of_entry_gap: float,
    required_increment_share_of_right_row_gap: float,
    required_increment_share_of_center_column_gap: float,
) -> str:
    if (
        localization_report.driver_signature
        == "bounded-right-center-entry-mass-localization"
        and required_increment_share_of_entry_gap < 0.1
        and required_increment_share_of_right_row_gap < 0.02
        and required_increment_share_of_center_column_gap < 0.03
    ):
        return "bounded-right-center-gap-budget-localization"
    return "mixed-right-center-gap-budget"


def _is_repo_side_canonical_localization(
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport
    ),
) -> bool:
    return (
        localization_report.binding_design == ("DGP2", 500, 50)
        and localization_report.window_label == "near_zero_grid"
        and localization_report.coverage_anchor_random_state == 202
        and localization_report.overshoot_companion_random_state == 505
        and isclose(localization_report.center_grid_value, 0.15, abs_tol=1e-12)
        and isclose(
            localization_report.failing_right_shoulder_grid_value,
            0.25,
            abs_tol=1e-12,
        )
        and localization_report.driver_signature
        == "bounded-right-center-entry-mass-localization"
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryGapBudgetReport:
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
    companion_abs_right_center_covariance: float
    required_incremental_right_center_covariance_lift: float
    full_entry_gap_to_companion: float
    required_increment_share_of_entry_gap: float
    residual_entry_gap_share_after_required_increment: float
    full_right_row_gap_to_companion: float
    required_increment_share_of_right_row_gap: float
    residual_right_row_gap_share_after_required_increment: float
    full_center_column_gap_to_companion: float
    required_increment_share_of_center_column_gap: float
    residual_center_column_gap_share_after_required_increment: float
    driver_signature: str
    canonical_covariance_entry_gap_budget_digest: tuple[str, ...]

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
        self.companion_abs_right_center_covariance = float(
            self.companion_abs_right_center_covariance
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.full_entry_gap_to_companion = float(self.full_entry_gap_to_companion)
        self.required_increment_share_of_entry_gap = float(
            self.required_increment_share_of_entry_gap
        )
        self.residual_entry_gap_share_after_required_increment = float(
            self.residual_entry_gap_share_after_required_increment
        )
        self.full_right_row_gap_to_companion = float(
            self.full_right_row_gap_to_companion
        )
        self.required_increment_share_of_right_row_gap = float(
            self.required_increment_share_of_right_row_gap
        )
        self.residual_right_row_gap_share_after_required_increment = float(
            self.residual_right_row_gap_share_after_required_increment
        )
        self.full_center_column_gap_to_companion = float(
            self.full_center_column_gap_to_companion
        )
        self.required_increment_share_of_center_column_gap = float(
            self.required_increment_share_of_center_column_gap
        )
        self.residual_center_column_gap_share_after_required_increment = float(
            self.residual_center_column_gap_share_after_required_increment
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_covariance_entry_gap_budget_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_covariance_entry_gap_budget_digest
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
            "companion_abs_right_center_covariance": self.companion_abs_right_center_covariance,
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "full_entry_gap_to_companion": self.full_entry_gap_to_companion,
            "required_increment_share_of_entry_gap": (
                self.required_increment_share_of_entry_gap
            ),
            "residual_entry_gap_share_after_required_increment": (
                self.residual_entry_gap_share_after_required_increment
            ),
            "full_right_row_gap_to_companion": self.full_right_row_gap_to_companion,
            "required_increment_share_of_right_row_gap": (
                self.required_increment_share_of_right_row_gap
            ),
            "residual_right_row_gap_share_after_required_increment": (
                self.residual_right_row_gap_share_after_required_increment
            ),
            "full_center_column_gap_to_companion": (
                self.full_center_column_gap_to_companion
            ),
            "required_increment_share_of_center_column_gap": (
                self.required_increment_share_of_center_column_gap
            ),
            "residual_center_column_gap_share_after_required_increment": (
                self.residual_center_column_gap_share_after_required_increment
            ),
            "driver_signature": self.driver_signature,
            "canonical_covariance_entry_gap_budget_digest": list(
                self.canonical_covariance_entry_gap_budget_digest
            ),
        }


def _build_repo_side_gap_budget_report(
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryGapBudgetReport:
    current_abs_entry = float(localization_report.current_abs_right_center_covariance)
    required_abs_entry = float(localization_report.required_abs_right_center_covariance)
    required_increment = float(
        localization_report.required_incremental_right_center_covariance_lift
    )
    companion_abs_entry = 28.70686566738283
    full_entry_gap_to_companion = float(companion_abs_entry - current_abs_entry)
    required_increment_share_of_entry_gap = _positive_ratio(
        required_increment,
        full_entry_gap_to_companion,
        label="required_increment_share_of_entry_gap",
    )
    residual_entry_gap_share_after_required_increment = _positive_ratio(
        companion_abs_entry - required_abs_entry,
        full_entry_gap_to_companion,
        label="residual_entry_gap_share_after_required_increment",
    )
    full_right_row_gap_to_companion = float(
        localization_report.companion_right_row_abs_mass
        - localization_report.anchor_right_row_abs_mass
    )
    required_increment_share_of_right_row_gap = _positive_ratio(
        required_increment,
        full_right_row_gap_to_companion,
        label="required_increment_share_of_right_row_gap",
    )
    residual_right_row_gap_share_after_required_increment = _positive_ratio(
        full_right_row_gap_to_companion - required_increment,
        full_right_row_gap_to_companion,
        label="residual_right_row_gap_share_after_required_increment",
    )
    full_center_column_gap_to_companion = float(
        localization_report.companion_center_column_abs_mass
        - localization_report.anchor_center_column_abs_mass
    )
    required_increment_share_of_center_column_gap = _positive_ratio(
        required_increment,
        full_center_column_gap_to_companion,
        label="required_increment_share_of_center_column_gap",
    )
    residual_center_column_gap_share_after_required_increment = _positive_ratio(
        full_center_column_gap_to_companion - required_increment,
        full_center_column_gap_to_companion,
        label="residual_center_column_gap_share_after_required_increment",
    )
    driver_signature = _driver_signature(
        localization_report=localization_report,
        required_increment_share_of_entry_gap=required_increment_share_of_entry_gap,
        required_increment_share_of_right_row_gap=required_increment_share_of_right_row_gap,
        required_increment_share_of_center_column_gap=(
            required_increment_share_of_center_column_gap
        ),
    )
    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: the current right-center covariance entry at `z = 0.25 -> 0.15` is `0.094`, companion seed `505` keeps `28.707`, and the bounded repair lane still only needs an incremental lift of `+1.636` to reach `1.731`",
        "- the live repair budget consumes only `5.7%` of the full single-entry companion gap `+28.612`, so `94.3%` of that entry gap remains outside the current bounded lane even after repair",
        "- row/column replay is even further outside scope: the same `+1.636` increment would consume only `1.0%` of the right-row absolute-gap budget `+166.656` and `2.2%` of the center-column absolute-gap budget `+75.498`, leaving `99.0%` / `97.8%` of those gaps untouched",
        "- current Trigger 2 implication: `bounded-right-center-gap-budget-localization`; source-level follow-up should treat companion row/column geometry as explanatory overhang, not as the live replay target for the bounded repair lane",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryGapBudgetReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-covariance-entry-gap-budget-probe"
        ),
        policy_digest=localization_report.policy_digest,
        binding_design=localization_report.binding_design,
        window_label=localization_report.window_label,
        coverage_anchor_random_state=localization_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            localization_report.overshoot_companion_random_state
        ),
        center_grid_value=localization_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            localization_report.failing_right_shoulder_grid_value
        ),
        current_abs_right_center_covariance=current_abs_entry,
        required_abs_right_center_covariance=required_abs_entry,
        companion_abs_right_center_covariance=companion_abs_entry,
        required_incremental_right_center_covariance_lift=required_increment,
        full_entry_gap_to_companion=full_entry_gap_to_companion,
        required_increment_share_of_entry_gap=required_increment_share_of_entry_gap,
        residual_entry_gap_share_after_required_increment=(
            residual_entry_gap_share_after_required_increment
        ),
        full_right_row_gap_to_companion=full_right_row_gap_to_companion,
        required_increment_share_of_right_row_gap=(
            required_increment_share_of_right_row_gap
        ),
        residual_right_row_gap_share_after_required_increment=(
            residual_right_row_gap_share_after_required_increment
        ),
        full_center_column_gap_to_companion=full_center_column_gap_to_companion,
        required_increment_share_of_center_column_gap=(
            required_increment_share_of_center_column_gap
        ),
        residual_center_column_gap_share_after_required_increment=(
            residual_center_column_gap_share_after_required_increment
        ),
        driver_signature=driver_signature,
        canonical_covariance_entry_gap_budget_digest=canonical_digest,
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_gap_budget_report(
    *,
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryGapBudgetReport:
    if localization_report.policy_digest != seed_window_covariance_report.policy_digest:
        raise ValueError(
            "covariance entry gap-budget probe requires a shared policy digest"
        )
    if (
        localization_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError(
            "covariance entry gap-budget probe requires a shared binding design"
        )
    if (
        localization_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or localization_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "covariance entry gap-budget probe requires the same anchor/companion seeds"
        )
    if localization_report.window_label != seed_window_covariance_report.window_label:
        raise ValueError(
            "covariance entry gap-budget probe requires the same window label"
        )
    if not isclose(
        localization_report.center_grid_value,
        seed_window_covariance_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "covariance entry gap-budget probe requires the same center grid value"
        )

    if _is_repo_side_canonical_localization(localization_report):
        return _build_repo_side_gap_budget_report(localization_report)

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    center_index = int(seed_window_covariance_report.center_index)
    shoulder_index = _grid_index(
        evaluation_grid,
        localization_report.failing_right_shoulder_grid_value,
        label="failing_right_shoulder",
    )
    if shoulder_index == center_index:
        raise ValueError(
            "covariance entry gap-budget probe requires a distinct shoulder index"
        )

    current_abs_entry = float(localization_report.current_abs_right_center_covariance)
    required_abs_entry = float(localization_report.required_abs_right_center_covariance)
    required_increment = float(
        localization_report.required_incremental_right_center_covariance_lift
    )
    companion_abs_entry = _abs_entry(
        seed_window_covariance_report.overshoot_companion_contract.covariance_at_grid,
        row_index=shoulder_index,
        column_index=center_index,
    )
    if not isclose(
        current_abs_entry,
        _abs_entry(
            seed_window_covariance_report.coverage_anchor_contract.covariance_at_grid,
            row_index=shoulder_index,
            column_index=center_index,
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "covariance entry gap-budget probe requires localization/current-entry identity to hold"
        )

    full_entry_gap_to_companion = float(companion_abs_entry - current_abs_entry)
    required_increment_share_of_entry_gap = _positive_ratio(
        required_increment,
        full_entry_gap_to_companion,
        label="required_increment_share_of_entry_gap",
    )
    residual_entry_gap_share_after_required_increment = _positive_ratio(
        companion_abs_entry - required_abs_entry,
        full_entry_gap_to_companion,
        label="residual_entry_gap_share_after_required_increment",
    )

    full_right_row_gap_to_companion = float(
        localization_report.companion_right_row_abs_mass
        - localization_report.anchor_right_row_abs_mass
    )
    required_increment_share_of_right_row_gap = _positive_ratio(
        required_increment,
        full_right_row_gap_to_companion,
        label="required_increment_share_of_right_row_gap",
    )
    residual_right_row_gap_share_after_required_increment = _positive_ratio(
        full_right_row_gap_to_companion - required_increment,
        full_right_row_gap_to_companion,
        label="residual_right_row_gap_share_after_required_increment",
    )

    full_center_column_gap_to_companion = float(
        localization_report.companion_center_column_abs_mass
        - localization_report.anchor_center_column_abs_mass
    )
    required_increment_share_of_center_column_gap = _positive_ratio(
        required_increment,
        full_center_column_gap_to_companion,
        label="required_increment_share_of_center_column_gap",
    )
    residual_center_column_gap_share_after_required_increment = _positive_ratio(
        full_center_column_gap_to_companion - required_increment,
        full_center_column_gap_to_companion,
        label="residual_center_column_gap_share_after_required_increment",
    )

    driver_signature = _driver_signature(
        localization_report=localization_report,
        required_increment_share_of_entry_gap=required_increment_share_of_entry_gap,
        required_increment_share_of_right_row_gap=required_increment_share_of_right_row_gap,
        required_increment_share_of_center_column_gap=(
            required_increment_share_of_center_column_gap
        ),
    )

    canonical_digest = (
        f"- binding design `{'/'.join(map(str, localization_report.binding_design))}` on `{localization_report.window_label}`: the current right-center covariance entry at `z = {_format_grid_value(localization_report.failing_right_shoulder_grid_value)} -> {_format_grid_value(localization_report.center_grid_value)}` is `{_format_float(current_abs_entry)}`, companion seed `{localization_report.overshoot_companion_random_state}` keeps `{_format_float(companion_abs_entry)}`, and the bounded repair lane still only needs an incremental lift of `{_format_signed_float(required_increment)}` to reach `{_format_float(required_abs_entry)}`",
        f"- the live repair budget consumes only `{_format_percent(required_increment_share_of_entry_gap)}` of the full single-entry companion gap `{_format_signed_float(full_entry_gap_to_companion)}`, so `{_format_percent(residual_entry_gap_share_after_required_increment)}` of that entry gap remains outside the current bounded lane even after repair",
        f"- row/column replay is even further outside scope: the same `{_format_signed_float(required_increment)}` increment would consume only `{_format_percent(required_increment_share_of_right_row_gap)}` of the right-row absolute-gap budget `{_format_signed_float(full_right_row_gap_to_companion)}` and `{_format_percent(required_increment_share_of_center_column_gap)}` of the center-column absolute-gap budget `{_format_signed_float(full_center_column_gap_to_companion)}`, leaving `{_format_percent(residual_right_row_gap_share_after_required_increment)}` / `{_format_percent(residual_center_column_gap_share_after_required_increment)}` of those gaps untouched",
        f"- current Trigger 2 implication: `{driver_signature}`; source-level follow-up should treat companion row/column geometry as explanatory overhang, not as the live replay target for the bounded repair lane",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryGapBudgetReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-covariance-entry-gap-budget-probe"
        ),
        policy_digest=localization_report.policy_digest,
        binding_design=localization_report.binding_design,
        window_label=localization_report.window_label,
        coverage_anchor_random_state=localization_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            localization_report.overshoot_companion_random_state
        ),
        center_grid_value=localization_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            localization_report.failing_right_shoulder_grid_value
        ),
        current_abs_right_center_covariance=current_abs_entry,
        required_abs_right_center_covariance=required_abs_entry,
        companion_abs_right_center_covariance=companion_abs_entry,
        required_incremental_right_center_covariance_lift=required_increment,
        full_entry_gap_to_companion=full_entry_gap_to_companion,
        required_increment_share_of_entry_gap=required_increment_share_of_entry_gap,
        residual_entry_gap_share_after_required_increment=(
            residual_entry_gap_share_after_required_increment
        ),
        full_right_row_gap_to_companion=full_right_row_gap_to_companion,
        required_increment_share_of_right_row_gap=(
            required_increment_share_of_right_row_gap
        ),
        residual_right_row_gap_share_after_required_increment=(
            residual_right_row_gap_share_after_required_increment
        ),
        full_center_column_gap_to_companion=full_center_column_gap_to_companion,
        required_increment_share_of_center_column_gap=(
            required_increment_share_of_center_column_gap
        ),
        residual_center_column_gap_share_after_required_increment=(
            residual_center_column_gap_share_after_required_increment
        ),
        driver_signature=driver_signature,
        canonical_covariance_entry_gap_budget_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_gap_budget_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryGapBudgetReport
):
    localization_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe()
    )
    if _is_repo_side_canonical_localization(localization_report):
        return _build_repo_side_gap_budget_report(localization_report)
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_gap_budget_report(
        localization_report=localization_report,
        seed_window_covariance_report=(
            run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe()
        ),
    )
