from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe,
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


def _driver_signature(
    *,
    directionality_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport,
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport
    ),
) -> str:
    if (
        directionality_report.driver_signature
        == "right-shoulder-directional-covariance-suppression"
        and localization_report.driver_signature
        == "bounded-right-center-entry-mass-localization"
        and directionality_report.anchor_left_shoulder_reserve > 0.0
        and directionality_report.anchor_failing_right_shoulder_reserve < 0.0
        and directionality_report.anchor_right_center_coupled_sigma_access
        < directionality_report.anchor_left_center_coupled_sigma_access
        and directionality_report.anchor_right_to_left_covariance_share < 1.0
        and directionality_report.companion_right_to_left_covariance_share > 1.0
        and localization_report.required_incremental_share_of_companion_right_row_mass
        < 0.02
        and localization_report.required_incremental_share_of_companion_center_column_mass
        < 0.05
    ):
        return "directional-right-center-first-repair-priority"
    return "mixed-entry-priority"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPriorityReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    anchor_left_shoulder_reserve: float
    anchor_failing_right_shoulder_reserve: float
    anchor_left_center_coupled_sigma_access: float
    anchor_right_center_coupled_sigma_access: float
    anchor_right_to_left_covariance_share: float
    companion_right_to_left_covariance_share: float
    directional_flip_ratio: float
    current_abs_left_center_covariance: float
    current_abs_right_center_covariance: float
    required_abs_right_center_covariance: float
    required_incremental_right_center_covariance_lift: float
    required_entry_share_of_anchor_right_row_mass: float
    required_entry_share_of_anchor_center_column_mass: float
    required_incremental_share_of_companion_right_row_mass: float
    required_incremental_share_of_companion_center_column_mass: float
    driver_signature: str
    canonical_entry_priority_digest: tuple[str, ...]

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
        self.left_shoulder_grid_value = float(self.left_shoulder_grid_value)
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.anchor_left_shoulder_reserve = float(self.anchor_left_shoulder_reserve)
        self.anchor_failing_right_shoulder_reserve = float(
            self.anchor_failing_right_shoulder_reserve
        )
        self.anchor_left_center_coupled_sigma_access = float(
            self.anchor_left_center_coupled_sigma_access
        )
        self.anchor_right_center_coupled_sigma_access = float(
            self.anchor_right_center_coupled_sigma_access
        )
        self.anchor_right_to_left_covariance_share = float(
            self.anchor_right_to_left_covariance_share
        )
        self.companion_right_to_left_covariance_share = float(
            self.companion_right_to_left_covariance_share
        )
        self.directional_flip_ratio = float(self.directional_flip_ratio)
        self.current_abs_left_center_covariance = float(
            self.current_abs_left_center_covariance
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
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_entry_priority_digest = tuple(
            str(line).rstrip() for line in self.canonical_entry_priority_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "left_shoulder_grid_value": self.left_shoulder_grid_value,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "anchor_left_shoulder_reserve": self.anchor_left_shoulder_reserve,
            "anchor_failing_right_shoulder_reserve": (
                self.anchor_failing_right_shoulder_reserve
            ),
            "anchor_left_center_coupled_sigma_access": (
                self.anchor_left_center_coupled_sigma_access
            ),
            "anchor_right_center_coupled_sigma_access": (
                self.anchor_right_center_coupled_sigma_access
            ),
            "anchor_right_to_left_covariance_share": (
                self.anchor_right_to_left_covariance_share
            ),
            "companion_right_to_left_covariance_share": (
                self.companion_right_to_left_covariance_share
            ),
            "directional_flip_ratio": self.directional_flip_ratio,
            "current_abs_left_center_covariance": self.current_abs_left_center_covariance,
            "current_abs_right_center_covariance": self.current_abs_right_center_covariance,
            "required_abs_right_center_covariance": (
                self.required_abs_right_center_covariance
            ),
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "required_entry_share_of_anchor_right_row_mass": (
                self.required_entry_share_of_anchor_right_row_mass
            ),
            "required_entry_share_of_anchor_center_column_mass": (
                self.required_entry_share_of_anchor_center_column_mass
            ),
            "required_incremental_share_of_companion_right_row_mass": (
                self.required_incremental_share_of_companion_right_row_mass
            ),
            "required_incremental_share_of_companion_center_column_mass": (
                self.required_incremental_share_of_companion_center_column_mass
            ),
            "driver_signature": self.driver_signature,
            "canonical_entry_priority_digest": list(
                self.canonical_entry_priority_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_priority_report(
    *,
    directionality_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport,
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPriorityReport:
    if directionality_report.policy_digest != localization_report.policy_digest:
        raise ValueError("entry priority probe requires a shared policy digest")
    if directionality_report.binding_design != localization_report.binding_design:
        raise ValueError("entry priority probe requires a shared binding design")
    if (
        directionality_report.coverage_anchor_random_state
        != localization_report.coverage_anchor_random_state
        or directionality_report.overshoot_companion_random_state
        != localization_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "entry priority probe requires the same anchor/companion seeds"
        )
    if directionality_report.window_label != localization_report.window_label:
        raise ValueError("entry priority probe requires the same window label")
    if not isclose(
        directionality_report.center_grid_value,
        localization_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError("entry priority probe requires the same center grid")
    if not isclose(
        directionality_report.failing_right_shoulder_grid_value,
        localization_report.failing_right_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError("entry priority probe requires the same right shoulder grid")
    if not isclose(
        directionality_report.anchor_abs_right_center_covariance,
        localization_report.current_abs_right_center_covariance,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "entry priority probe requires the same current right-center covariance"
        )

    driver_signature = _driver_signature(
        directionality_report=directionality_report,
        localization_report=localization_report,
    )

    canonical_digest = (
        f"- binding design `DGP2/500/50` on `{directionality_report.window_label}`: coverage anchor seed "
        f"`{directionality_report.coverage_anchor_random_state}` already splits reserve by direction, "
        f"keeping `{_format_signed_float(directionality_report.anchor_left_shoulder_reserve)}` at left shoulder "
        f"`z = {_format_grid_value(directionality_report.left_shoulder_grid_value)}` but "
        f"`{_format_signed_float(directionality_report.anchor_failing_right_shoulder_reserve)}` at failing right shoulder "
        f"`z = {_format_grid_value(directionality_report.failing_right_shoulder_grid_value)}`, so the current miss is right-sided before any whole-window retuning enters the picture",
        "- the center-coupled channel stays right-suppressed in the anchor but flips in the companion: "
        f"current coupled access is `{_format_float(directionality_report.anchor_left_center_coupled_sigma_access)} -> {_format_float(directionality_report.anchor_right_center_coupled_sigma_access)}`, "
        f"so the right side keeps only `{_format_percent(directionality_report.anchor_right_to_left_covariance_share)}` of left access, whereas companion seed "
        f"`{directionality_report.overshoot_companion_random_state}` restores the right side to "
        f"`{_format_percent(directionality_report.companion_right_to_left_covariance_share)}` of left access "
        f"(`{_format_ratio(directionality_report.directional_flip_ratio)}` directional flip)",
        "- prioritizing the right-center entry still remains a bounded local action: "
        f"`|covariance({_format_grid_value(localization_report.failing_right_shoulder_grid_value)}, {_format_grid_value(localization_report.center_grid_value)})|` only needs to rise from "
        f"`{_format_float(localization_report.current_abs_right_center_covariance)}` to `{_format_float(localization_report.required_abs_right_center_covariance)}` "
        f"(increment `{_format_signed_float(localization_report.required_incremental_right_center_covariance_lift)}`), and even after repair the entry would still be only "
        f"`{_format_percent(localization_report.required_entry_share_of_anchor_right_row_mass)}` / "
        f"`{_format_percent(localization_report.required_entry_share_of_anchor_center_column_mass)}` of anchor right-row / center-column mass while the increment is just "
        f"`{_format_percent(localization_report.required_incremental_share_of_companion_right_row_mass)}` / "
        f"`{_format_percent(localization_report.required_incremental_share_of_companion_center_column_mass)}` of companion right-row / center-column mass",
        f"- current Trigger 2 implication: `{driver_signature}`; source-level follow-up should prioritize bounded right-shoulder / center entry access before revisiting already-positive left-center support or replaying the full local covariance window",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPriorityReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "entry-priority-probe"
        ),
        policy_digest=directionality_report.policy_digest,
        binding_design=directionality_report.binding_design,
        window_label=directionality_report.window_label,
        coverage_anchor_random_state=directionality_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            directionality_report.overshoot_companion_random_state
        ),
        left_shoulder_grid_value=directionality_report.left_shoulder_grid_value,
        center_grid_value=directionality_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            directionality_report.failing_right_shoulder_grid_value
        ),
        anchor_left_shoulder_reserve=directionality_report.anchor_left_shoulder_reserve,
        anchor_failing_right_shoulder_reserve=(
            directionality_report.anchor_failing_right_shoulder_reserve
        ),
        anchor_left_center_coupled_sigma_access=(
            directionality_report.anchor_left_center_coupled_sigma_access
        ),
        anchor_right_center_coupled_sigma_access=(
            directionality_report.anchor_right_center_coupled_sigma_access
        ),
        anchor_right_to_left_covariance_share=(
            directionality_report.anchor_right_to_left_covariance_share
        ),
        companion_right_to_left_covariance_share=(
            directionality_report.companion_right_to_left_covariance_share
        ),
        directional_flip_ratio=directionality_report.directional_flip_ratio,
        current_abs_left_center_covariance=(
            directionality_report.anchor_abs_left_center_covariance
        ),
        current_abs_right_center_covariance=(
            localization_report.current_abs_right_center_covariance
        ),
        required_abs_right_center_covariance=(
            localization_report.required_abs_right_center_covariance
        ),
        required_incremental_right_center_covariance_lift=(
            localization_report.required_incremental_right_center_covariance_lift
        ),
        required_entry_share_of_anchor_right_row_mass=(
            localization_report.required_entry_share_of_anchor_right_row_mass
        ),
        required_entry_share_of_anchor_center_column_mass=(
            localization_report.required_entry_share_of_anchor_center_column_mass
        ),
        required_incremental_share_of_companion_right_row_mass=(
            localization_report.required_incremental_share_of_companion_right_row_mass
        ),
        required_incremental_share_of_companion_center_column_mass=(
            localization_report.required_incremental_share_of_companion_center_column_mass
        ),
        driver_signature=driver_signature,
        canonical_entry_priority_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_priority_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPriorityReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_priority_report(
        directionality_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe(),
        localization_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe(),
    )
