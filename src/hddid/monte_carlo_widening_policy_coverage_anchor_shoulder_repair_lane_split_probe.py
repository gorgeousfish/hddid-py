from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_raw_cross_lift_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualRawCrossLiftReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_raw_cross_lift_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _driver_signature(
    *,
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport
    ),
    raw_cross_lift_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualRawCrossLiftReport
    ),
) -> str:
    if (
        localization_report.driver_signature
        == "bounded-right-center-entry-mass-localization"
        and raw_cross_lift_report.driver_signature
        == "bounded-raw-cross-sign-healing-side-lane"
    ):
        return "localized-entry-live-lane-vs-bounded-raw-cross-side-lane"
    return "mixed-repair-lane-split"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairLaneSplitReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    live_lane_driver_signature: str
    sign_healing_side_lane_driver_signature: str
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    current_abs_right_center_covariance: float
    required_abs_right_center_covariance: float
    required_incremental_right_center_covariance_lift: float
    required_entry_share_of_anchor_right_row_mass: float
    required_entry_share_of_anchor_center_column_mass: float
    required_incremental_share_of_companion_right_row_mass: float
    required_incremental_share_of_companion_center_column_mass: float
    current_anchor_cross_shoulder_correlation: float
    zero_partial_cross_shoulder_correlation_at_bounded_repair: float
    required_raw_cross_shoulder_lift: float
    required_raw_cross_gap_share: float
    remaining_raw_cross_gap_headroom_share: float
    sign_flip_share_of_required_lift: float
    zero_partial_cross_shoulder_share_of_companion: float
    driver_signature: str
    canonical_repair_lane_split_digest: tuple[str, ...]

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
        self.live_lane_driver_signature = str(self.live_lane_driver_signature).strip()
        self.sign_healing_side_lane_driver_signature = str(
            self.sign_healing_side_lane_driver_signature
        ).strip()
        self.left_shoulder_grid_value = float(self.left_shoulder_grid_value)
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
        self.current_anchor_cross_shoulder_correlation = float(
            self.current_anchor_cross_shoulder_correlation
        )
        self.zero_partial_cross_shoulder_correlation_at_bounded_repair = float(
            self.zero_partial_cross_shoulder_correlation_at_bounded_repair
        )
        self.required_raw_cross_shoulder_lift = float(
            self.required_raw_cross_shoulder_lift
        )
        self.required_raw_cross_gap_share = float(self.required_raw_cross_gap_share)
        self.remaining_raw_cross_gap_headroom_share = float(
            self.remaining_raw_cross_gap_headroom_share
        )
        self.sign_flip_share_of_required_lift = float(
            self.sign_flip_share_of_required_lift
        )
        self.zero_partial_cross_shoulder_share_of_companion = float(
            self.zero_partial_cross_shoulder_share_of_companion
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_repair_lane_split_digest = tuple(
            str(line).rstrip() for line in self.canonical_repair_lane_split_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "live_lane_driver_signature": self.live_lane_driver_signature,
            "sign_healing_side_lane_driver_signature": (
                self.sign_healing_side_lane_driver_signature
            ),
            "left_shoulder_grid_value": self.left_shoulder_grid_value,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "current_abs_right_center_covariance": self.current_abs_right_center_covariance,
            "required_abs_right_center_covariance": self.required_abs_right_center_covariance,
            "required_incremental_right_center_covariance_lift": self.required_incremental_right_center_covariance_lift,
            "required_entry_share_of_anchor_right_row_mass": self.required_entry_share_of_anchor_right_row_mass,
            "required_entry_share_of_anchor_center_column_mass": self.required_entry_share_of_anchor_center_column_mass,
            "required_incremental_share_of_companion_right_row_mass": self.required_incremental_share_of_companion_right_row_mass,
            "required_incremental_share_of_companion_center_column_mass": self.required_incremental_share_of_companion_center_column_mass,
            "current_anchor_cross_shoulder_correlation": self.current_anchor_cross_shoulder_correlation,
            "zero_partial_cross_shoulder_correlation_at_bounded_repair": self.zero_partial_cross_shoulder_correlation_at_bounded_repair,
            "required_raw_cross_shoulder_lift": self.required_raw_cross_shoulder_lift,
            "required_raw_cross_gap_share": self.required_raw_cross_gap_share,
            "remaining_raw_cross_gap_headroom_share": self.remaining_raw_cross_gap_headroom_share,
            "sign_flip_share_of_required_lift": self.sign_flip_share_of_required_lift,
            "zero_partial_cross_shoulder_share_of_companion": self.zero_partial_cross_shoulder_share_of_companion,
            "driver_signature": self.driver_signature,
            "canonical_repair_lane_split_digest": list(
                self.canonical_repair_lane_split_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_lane_split_report(
    *,
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport
    ),
    raw_cross_lift_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualRawCrossLiftReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairLaneSplitReport:
    if localization_report.policy_digest != raw_cross_lift_report.policy_digest:
        raise ValueError("repair lane split probe requires a shared policy digest")
    if localization_report.binding_design != raw_cross_lift_report.binding_design:
        raise ValueError("repair lane split probe requires a shared binding design")
    if (
        localization_report.coverage_anchor_random_state
        != raw_cross_lift_report.coverage_anchor_random_state
        or localization_report.overshoot_companion_random_state
        != raw_cross_lift_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "repair lane split probe requires the same anchor/companion seeds"
        )
    if localization_report.window_label != "near_zero_grid":
        raise ValueError("repair lane split probe expects the near_zero_grid window")
    if not isclose(
        localization_report.center_grid_value,
        raw_cross_lift_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError("repair lane split probe requires the same center grid")
    if not isclose(
        localization_report.failing_right_shoulder_grid_value,
        raw_cross_lift_report.failing_right_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "repair lane split probe requires the same right shoulder grid"
        )

    driver_signature = _driver_signature(
        localization_report=localization_report,
        raw_cross_lift_report=raw_cross_lift_report,
    )

    canonical_digest = (
        "- binding design `DGP2/500/50` on "
        f"`{localization_report.window_label}`: the live repair lane still stays a single "
        "right-center covariance entry, raising "
        f"`{_format_float(localization_report.current_abs_right_center_covariance)}` "
        f"to `{_format_float(localization_report.required_abs_right_center_covariance)}` "
        f"(increment `{_format_signed_float(localization_report.required_incremental_right_center_covariance_lift)}`) "
        f"at `z = {_format_grid_value(localization_report.failing_right_shoulder_grid_value)} -> "
        f"{_format_grid_value(localization_report.center_grid_value)}`, while future sign healing "
        "would use a separate raw cross-shoulder target "
        f"`{_format_float(raw_cross_lift_report.current_anchor_cross_shoulder_correlation)} -> "
        f"{_format_float(raw_cross_lift_report.zero_partial_cross_shoulder_correlation_at_bounded_repair)}` "
        f"(increment `{_format_signed_float(raw_cross_lift_report.required_raw_cross_shoulder_lift)}`) "
        f"at `z = {_format_grid_value(raw_cross_lift_report.left_shoulder_grid_value)} <-> "
        f"{_format_grid_value(raw_cross_lift_report.failing_right_shoulder_grid_value)}`",
        "- the live lane remains localized rather than replaying local geometry: "
        "even after repair, the right-center entry would still be only "
        f"`{_format_percent(localization_report.required_entry_share_of_anchor_right_row_mass)}` / "
        f"`{_format_percent(localization_report.required_entry_share_of_anchor_center_column_mass)}` "
        "of anchor right-row / center-column mass, and the required increment is just "
        f"`{_format_percent(localization_report.required_incremental_share_of_companion_right_row_mass)}` / "
        f"`{_format_percent(localization_report.required_incremental_share_of_companion_center_column_mass)}` "
        "of companion right-row / center-column mass",
        "- the sign-healing path remains a bounded side lane rather than companion replay: "
        "the raw cross lift consumes only "
        f"`{_format_percent(raw_cross_lift_report.required_raw_cross_gap_share)}` of the full "
        "anchor-to-companion cross gap, leaves "
        f"`{_format_percent(raw_cross_lift_report.remaining_raw_cross_gap_headroom_share)}` "
        "headroom unused, and "
        f"`{_format_percent(raw_cross_lift_report.sign_flip_share_of_required_lift)}` of the move "
        "is just sign clearance",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; source-level follow-up should keep the live fix focused on "
        "bounded right-shoulder / center covariance entry access, while any future "
        "direct-residual healing should trace a separate raw cross-shoulder side lane "
        "instead of replaying companion geometry",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairLaneSplitReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "repair-lane-split-probe"
        ),
        policy_digest=localization_report.policy_digest,
        binding_design=localization_report.binding_design,
        window_label=localization_report.window_label,
        coverage_anchor_random_state=localization_report.coverage_anchor_random_state,
        overshoot_companion_random_state=localization_report.overshoot_companion_random_state,
        live_lane_driver_signature=localization_report.driver_signature,
        sign_healing_side_lane_driver_signature=raw_cross_lift_report.driver_signature,
        left_shoulder_grid_value=raw_cross_lift_report.left_shoulder_grid_value,
        center_grid_value=localization_report.center_grid_value,
        failing_right_shoulder_grid_value=localization_report.failing_right_shoulder_grid_value,
        current_abs_right_center_covariance=localization_report.current_abs_right_center_covariance,
        required_abs_right_center_covariance=localization_report.required_abs_right_center_covariance,
        required_incremental_right_center_covariance_lift=localization_report.required_incremental_right_center_covariance_lift,
        required_entry_share_of_anchor_right_row_mass=localization_report.required_entry_share_of_anchor_right_row_mass,
        required_entry_share_of_anchor_center_column_mass=localization_report.required_entry_share_of_anchor_center_column_mass,
        required_incremental_share_of_companion_right_row_mass=localization_report.required_incremental_share_of_companion_right_row_mass,
        required_incremental_share_of_companion_center_column_mass=localization_report.required_incremental_share_of_companion_center_column_mass,
        current_anchor_cross_shoulder_correlation=raw_cross_lift_report.current_anchor_cross_shoulder_correlation,
        zero_partial_cross_shoulder_correlation_at_bounded_repair=raw_cross_lift_report.zero_partial_cross_shoulder_correlation_at_bounded_repair,
        required_raw_cross_shoulder_lift=raw_cross_lift_report.required_raw_cross_shoulder_lift,
        required_raw_cross_gap_share=raw_cross_lift_report.required_raw_cross_gap_share,
        remaining_raw_cross_gap_headroom_share=raw_cross_lift_report.remaining_raw_cross_gap_headroom_share,
        sign_flip_share_of_required_lift=raw_cross_lift_report.sign_flip_share_of_required_lift,
        zero_partial_cross_shoulder_share_of_companion=raw_cross_lift_report.zero_partial_cross_shoulder_share_of_companion,
        driver_signature=driver_signature,
        canonical_repair_lane_split_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_lane_split_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairLaneSplitReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_lane_split_report(
        localization_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe(),
        raw_cross_lift_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_raw_cross_lift_probe(),
    )
