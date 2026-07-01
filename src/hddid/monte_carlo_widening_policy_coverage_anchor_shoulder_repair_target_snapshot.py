from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_calibration_debt_snapshot import (
    Phase7MonteCarloWideningPolicyCalibrationDebtSnapshotReport,
    run_phase7_monte_carlo_widening_policy_calibration_debt_snapshot,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _driver_signature(
    *,
    calibration_report: Phase7MonteCarloWideningPolicyCalibrationDebtSnapshotReport,
    directionality_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport
    ),
    direct_residual_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualReport
    ),
) -> str:
    if (
        calibration_report.repair_target_signature == "coupling-first-repair-target"
        and directionality_report.driver_signature
        == "right-shoulder-directional-covariance-suppression"
        and direct_residual_report.driver_signature
        == "direct-residual-correlation-repair-target"
    ):
        return "directional-direct-residual-correlation-repair-target"
    if direct_residual_report.driver_signature == (
        "direct-residual-correlation-repair-target"
    ):
        return "direct-residual-correlation-repair-target"
    return "mixed-trigger2-repair-target"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairTargetSnapshotReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    binding_driver: str
    seed_dispersion_driver: str
    seed_role_split_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    local_scale_access_shortfall_share: float
    anchor_right_to_left_center_access_share: float
    companion_right_to_left_center_access_share: float
    directional_flip_ratio: float
    anchor_partial_cross_shoulder_correlation: float
    companion_partial_cross_shoulder_correlation: float
    anchor_center_mediated_share_of_abs_cross_shoulder: float
    required_repaired_correlation: float
    correlation_repair_increment: float
    required_gap_share: float
    correlation_log_gap_share: float
    repair_target_signature: str
    calibration_debt_snapshot_digest: tuple[str, ...]
    directionality_digest: tuple[str, ...]
    direct_residual_digest: tuple[str, ...]
    canonical_repair_target_snapshot_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.binding_driver = str(self.binding_driver).strip()
        self.seed_dispersion_driver = str(self.seed_dispersion_driver).strip()
        self.seed_role_split_label = str(self.seed_role_split_label).strip()
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.left_shoulder_grid_value = float(self.left_shoulder_grid_value)
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.local_scale_access_shortfall_share = float(
            self.local_scale_access_shortfall_share
        )
        self.anchor_right_to_left_center_access_share = float(
            self.anchor_right_to_left_center_access_share
        )
        self.companion_right_to_left_center_access_share = float(
            self.companion_right_to_left_center_access_share
        )
        self.directional_flip_ratio = float(self.directional_flip_ratio)
        self.anchor_partial_cross_shoulder_correlation = float(
            self.anchor_partial_cross_shoulder_correlation
        )
        self.companion_partial_cross_shoulder_correlation = float(
            self.companion_partial_cross_shoulder_correlation
        )
        self.anchor_center_mediated_share_of_abs_cross_shoulder = float(
            self.anchor_center_mediated_share_of_abs_cross_shoulder
        )
        self.required_repaired_correlation = float(self.required_repaired_correlation)
        self.correlation_repair_increment = float(self.correlation_repair_increment)
        self.required_gap_share = float(self.required_gap_share)
        self.correlation_log_gap_share = float(self.correlation_log_gap_share)
        self.repair_target_signature = str(self.repair_target_signature).strip()
        self.calibration_debt_snapshot_digest = tuple(
            str(line).rstrip() for line in self.calibration_debt_snapshot_digest
        )
        self.directionality_digest = tuple(
            str(line).rstrip() for line in self.directionality_digest
        )
        self.direct_residual_digest = tuple(
            str(line).rstrip() for line in self.direct_residual_digest
        )
        self.canonical_repair_target_snapshot_digest = tuple(
            str(line).rstrip() for line in self.canonical_repair_target_snapshot_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "binding_driver": self.binding_driver,
            "seed_dispersion_driver": self.seed_dispersion_driver,
            "seed_role_split_label": self.seed_role_split_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "left_shoulder_grid_value": self.left_shoulder_grid_value,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "local_scale_access_shortfall_share": (
                self.local_scale_access_shortfall_share
            ),
            "anchor_right_to_left_center_access_share": (
                self.anchor_right_to_left_center_access_share
            ),
            "companion_right_to_left_center_access_share": (
                self.companion_right_to_left_center_access_share
            ),
            "directional_flip_ratio": self.directional_flip_ratio,
            "anchor_partial_cross_shoulder_correlation": (
                self.anchor_partial_cross_shoulder_correlation
            ),
            "companion_partial_cross_shoulder_correlation": (
                self.companion_partial_cross_shoulder_correlation
            ),
            "anchor_center_mediated_share_of_abs_cross_shoulder": (
                self.anchor_center_mediated_share_of_abs_cross_shoulder
            ),
            "required_repaired_correlation": self.required_repaired_correlation,
            "correlation_repair_increment": self.correlation_repair_increment,
            "required_gap_share": self.required_gap_share,
            "correlation_log_gap_share": self.correlation_log_gap_share,
            "repair_target_signature": self.repair_target_signature,
            "calibration_debt_snapshot_digest": list(
                self.calibration_debt_snapshot_digest
            ),
            "directionality_digest": list(self.directionality_digest),
            "direct_residual_digest": list(self.direct_residual_digest),
            "canonical_repair_target_snapshot_digest": list(
                self.canonical_repair_target_snapshot_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot_report(
    *,
    calibration_report: Phase7MonteCarloWideningPolicyCalibrationDebtSnapshotReport,
    directionality_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport
    ),
    direct_residual_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairTargetSnapshotReport:
    if calibration_report.policy_digest != directionality_report.policy_digest:
        raise ValueError("repair target snapshot requires a single policy digest")
    if calibration_report.policy_digest != direct_residual_report.policy_digest:
        raise ValueError("repair target snapshot requires a single policy digest")
    if calibration_report.binding_design != directionality_report.binding_design:
        raise ValueError("repair target snapshot requires a single binding design")
    if calibration_report.binding_design != direct_residual_report.binding_design:
        raise ValueError("repair target snapshot requires a single binding design")
    if (
        calibration_report.coverage_anchor_random_state
        != directionality_report.coverage_anchor_random_state
        or calibration_report.coverage_anchor_random_state
        != direct_residual_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "repair target snapshot requires a single coverage-anchor seed"
        )
    if (
        calibration_report.overshoot_companion_random_state
        != directionality_report.overshoot_companion_random_state
        or calibration_report.overshoot_companion_random_state
        != direct_residual_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "repair target snapshot requires a single overshoot companion seed"
        )
    if not isclose(
        directionality_report.left_shoulder_grid_value,
        direct_residual_report.left_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError("repair target snapshot requires a shared left shoulder grid")
    if not isclose(
        directionality_report.center_grid_value,
        direct_residual_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError("repair target snapshot requires a shared center grid")
    if not isclose(
        directionality_report.failing_right_shoulder_grid_value,
        direct_residual_report.failing_right_shoulder_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "repair target snapshot requires a shared failing right-shoulder grid"
        )

    repair_target_signature = _driver_signature(
        calibration_report=calibration_report,
        directionality_report=directionality_report,
        direct_residual_report=direct_residual_report,
    )
    canonical_digest = (
        "- Trigger 2 binding design still stays "
        f"`{_format_design_key(*calibration_report.binding_design)}`: live calibration debt remains "
        f"`{calibration_report.binding_driver}` plus "
        f"`{calibration_report.seed_dispersion_driver}`, already narrowed to "
        f"`{calibration_report.seed_role_split_label}`; the residual miss is still "
        "`local_scale_access_shortfall_share = "
        f"{_format_percent(calibration_report.local_scale_access_shortfall_share)}` at "
        f"`z = {_format_grid_value(directionality_report.failing_right_shoulder_grid_value)}` "
        f"for coverage anchor seed `{calibration_report.coverage_anchor_random_state}` "
        f"against overshoot companion seed "
        f"`{calibration_report.overshoot_companion_random_state}`",
        "- within anchor seed "
        f"`{directionality_report.coverage_anchor_random_state}`, the residual miss is "
        "right-specific rather than whole-window: right-center coupled access keeps only "
        f"`{_format_percent(directionality_report.anchor_right_to_left_covariance_share)}` "
        "of the anchor left-side level, while companion seed "
        f"`{directionality_report.overshoot_companion_random_state}` flips the right share "
        f"to `{_format_percent(directionality_report.companion_right_to_left_covariance_share)}`; "
        f"the directional flip is therefore `{_format_ratio(directionality_report.directional_flip_ratio)}`",
        "- conditioning on center "
        f"`z = {_format_grid_value(directionality_report.center_grid_value)}` still leaves partial "
        "cross-shoulder correlation at "
        f"`{_format_signed_float(direct_residual_report.anchor_partial_cross_shoulder_correlation)}` "
        f"for seed `{direct_residual_report.coverage_anchor_random_state}` versus "
        f"`{_format_signed_float(direct_residual_report.companion_partial_cross_shoulder_correlation)}` "
        f"for seed `{direct_residual_report.overshoot_companion_random_state}`; center "
        "mediation explains only "
        f"`{_format_percent(direct_residual_report.anchor_center_mediated_share_of_abs_cross_shoulder)}` "
        "of the anchor's absolute cross-shoulder magnitude, so center smoothing alone "
        "cannot heal the sign fracture",
        "- closing the last miss still needs only a bounded right-shoulder / center "
        "correlation bridge: repaired correlation rises to "
        f"`{_format_float(direct_residual_report.required_repaired_correlation)}` via "
        f"increment `{_format_signed_float(direct_residual_report.correlation_repair_increment)}`, "
        "which consumes just "
        f"`{_format_percent(direct_residual_report.required_gap_share)}` of the full "
        "anchor-to-companion gap while shoulder-center correlation still contributes "
        f"`{_format_percent(direct_residual_report.correlation_log_gap_share)}` of the log-gap",
        "- current Trigger 2 source-level repair target therefore tightens to "
        f"`{repair_target_signature}`: recover a bounded slice of seed "
        f"`{direct_residual_report.coverage_anchor_random_state}` right-shoulder / center "
        "correlation access without generic center smoothing, symmetric shoulder "
        "cleanup, or full companion replay",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairTargetSnapshotReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "repair-target-snapshot"
        ),
        policy_digest=calibration_report.policy_digest,
        binding_design=calibration_report.binding_design,
        binding_driver=calibration_report.binding_driver,
        seed_dispersion_driver=calibration_report.seed_dispersion_driver,
        seed_role_split_label=calibration_report.seed_role_split_label,
        coverage_anchor_random_state=calibration_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            calibration_report.overshoot_companion_random_state
        ),
        left_shoulder_grid_value=directionality_report.left_shoulder_grid_value,
        center_grid_value=directionality_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            directionality_report.failing_right_shoulder_grid_value
        ),
        local_scale_access_shortfall_share=(
            calibration_report.local_scale_access_shortfall_share
        ),
        anchor_right_to_left_center_access_share=(
            directionality_report.anchor_right_to_left_covariance_share
        ),
        companion_right_to_left_center_access_share=(
            directionality_report.companion_right_to_left_covariance_share
        ),
        directional_flip_ratio=directionality_report.directional_flip_ratio,
        anchor_partial_cross_shoulder_correlation=(
            direct_residual_report.anchor_partial_cross_shoulder_correlation
        ),
        companion_partial_cross_shoulder_correlation=(
            direct_residual_report.companion_partial_cross_shoulder_correlation
        ),
        anchor_center_mediated_share_of_abs_cross_shoulder=(
            direct_residual_report.anchor_center_mediated_share_of_abs_cross_shoulder
        ),
        required_repaired_correlation=(
            direct_residual_report.required_repaired_correlation
        ),
        correlation_repair_increment=(
            direct_residual_report.correlation_repair_increment
        ),
        required_gap_share=direct_residual_report.required_gap_share,
        correlation_log_gap_share=direct_residual_report.correlation_log_gap_share,
        repair_target_signature=repair_target_signature,
        calibration_debt_snapshot_digest=(
            calibration_report.canonical_calibration_debt_snapshot_digest
        ),
        directionality_digest=(
            directionality_report.canonical_coverage_anchor_shoulder_directionality_digest
        ),
        direct_residual_digest=direct_residual_report.canonical_direct_residual_digest,
        canonical_repair_target_snapshot_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairTargetSnapshotReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot_report(
        calibration_report=run_phase7_monte_carlo_widening_policy_calibration_debt_snapshot(),
        directionality_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe(),
        direct_residual_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_probe(),
    )
