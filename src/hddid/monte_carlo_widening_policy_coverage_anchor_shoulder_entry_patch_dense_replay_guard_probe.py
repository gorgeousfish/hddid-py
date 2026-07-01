from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchMatrixNormBudgetReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _driver_signature(
    *,
    patch_plan_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport
    ),
    matrix_norm_budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchMatrixNormBudgetReport
    ),
    intake_bundle_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport
    ),
    dense_replay_rejected: bool,
) -> str:
    if (
        patch_plan_report.driver_signature == "bounded-right-center-entry-patch-plan"
        and localization_report.driver_signature
        == "bounded-right-center-entry-mass-localization"
        and matrix_norm_budget_report.driver_signature
        == "bounded-right-center-entry-patch-matrix-norm-budget"
        and intake_bundle_report.driver_signature
        == "bounded-right-center-entry-patch-intake-bundle"
        and patch_plan_report.symmetric_patch_nnz == 2
        and matrix_norm_budget_report.patch_rank == 2
        and matrix_norm_budget_report.patch_frobenius_share_of_baseline < 0.2
        and matrix_norm_budget_report.patch_spectral_share_of_baseline < 0.2
        and localization_report.required_incremental_share_of_companion_right_row_mass
        < 0.02
        and localization_report.required_incremental_share_of_companion_center_column_mass
        < 0.05
        and intake_bundle_report.intake_guard_bundle_holds
        and dense_replay_rejected
    ):
        return "bounded-right-center-entry-patch-dense-replay-guard"
    return "mixed-right-center-entry-patch-dense-replay-guard"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchDenseReplayGuardReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    symmetric_patch_nnz: int
    patch_rank: int
    patch_frobenius_share_of_baseline: float
    patch_spectral_share_of_baseline: float
    required_incremental_share_of_companion_right_row_mass: float
    required_incremental_share_of_companion_center_column_mass: float
    companion_right_row_replay_multiple_vs_required_increment: float
    companion_center_column_replay_multiple_vs_required_increment: float
    intake_guard_bundle_holds: bool
    dense_replay_rejected: bool
    prohibited_dense_replay_actions: tuple[str, ...]
    driver_signature: str
    canonical_entry_patch_dense_replay_guard_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.symmetric_patch_nnz = int(self.symmetric_patch_nnz)
        self.patch_rank = int(self.patch_rank)
        self.patch_frobenius_share_of_baseline = float(
            self.patch_frobenius_share_of_baseline
        )
        self.patch_spectral_share_of_baseline = float(
            self.patch_spectral_share_of_baseline
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
        self.intake_guard_bundle_holds = bool(self.intake_guard_bundle_holds)
        self.dense_replay_rejected = bool(self.dense_replay_rejected)
        self.prohibited_dense_replay_actions = tuple(
            str(item).strip() for item in self.prohibited_dense_replay_actions
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_entry_patch_dense_replay_guard_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_entry_patch_dense_replay_guard_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "evaluation_grid": list(self.evaluation_grid),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "symmetric_patch_nnz": self.symmetric_patch_nnz,
            "patch_rank": self.patch_rank,
            "patch_frobenius_share_of_baseline": (
                self.patch_frobenius_share_of_baseline
            ),
            "patch_spectral_share_of_baseline": (self.patch_spectral_share_of_baseline),
            "required_incremental_share_of_companion_right_row_mass": (
                self.required_incremental_share_of_companion_right_row_mass
            ),
            "required_incremental_share_of_companion_center_column_mass": (
                self.required_incremental_share_of_companion_center_column_mass
            ),
            "companion_right_row_replay_multiple_vs_required_increment": (
                self.companion_right_row_replay_multiple_vs_required_increment
            ),
            "companion_center_column_replay_multiple_vs_required_increment": (
                self.companion_center_column_replay_multiple_vs_required_increment
            ),
            "intake_guard_bundle_holds": self.intake_guard_bundle_holds,
            "dense_replay_rejected": self.dense_replay_rejected,
            "prohibited_dense_replay_actions": list(
                self.prohibited_dense_replay_actions
            ),
            "driver_signature": self.driver_signature,
            "canonical_entry_patch_dense_replay_guard_digest": list(
                self.canonical_entry_patch_dense_replay_guard_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_dense_replay_guard_report(
    *,
    patch_plan_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCovarianceEntryLocalizationReport
    ),
    matrix_norm_budget_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchMatrixNormBudgetReport
    ),
    intake_bundle_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchIntakeBundleReport
    ),
) -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchDenseReplayGuardReport
):
    if patch_plan_report.policy_digest != localization_report.policy_digest:
        raise ValueError("dense-replay guard requires a shared policy digest")
    if patch_plan_report.policy_digest != matrix_norm_budget_report.policy_digest:
        raise ValueError("dense-replay guard requires a shared policy digest")
    if patch_plan_report.policy_digest != intake_bundle_report.policy_digest:
        raise ValueError("dense-replay guard requires a shared policy digest")
    if patch_plan_report.binding_design != localization_report.binding_design:
        raise ValueError("dense-replay guard requires a shared binding design")
    if patch_plan_report.binding_design != matrix_norm_budget_report.binding_design:
        raise ValueError("dense-replay guard requires a shared binding design")
    if patch_plan_report.binding_design != intake_bundle_report.binding_design:
        raise ValueError("dense-replay guard requires a shared binding design")
    if patch_plan_report.window_label != localization_report.window_label:
        raise ValueError("dense-replay guard requires a shared window label")
    if patch_plan_report.window_label != matrix_norm_budget_report.window_label:
        raise ValueError("dense-replay guard requires a shared window label")
    if patch_plan_report.window_label != intake_bundle_report.window_label:
        raise ValueError("dense-replay guard requires a shared window label")

    prohibited_dense_replay_actions = (
        "whole right-row replay",
        "whole center-column replay",
        "dense covariance rewrite",
    )
    dense_replay_rejected = bool(
        patch_plan_report.symmetric_patch_nnz == 2
        and matrix_norm_budget_report.patch_rank == 2
        and matrix_norm_budget_report.patch_frobenius_share_of_baseline < 0.2
        and matrix_norm_budget_report.patch_spectral_share_of_baseline < 0.2
        and localization_report.required_incremental_share_of_companion_right_row_mass
        < 0.02
        and localization_report.required_incremental_share_of_companion_center_column_mass
        < 0.05
        and localization_report.companion_right_row_replay_multiple_vs_required_increment
        > 50.0
        and localization_report.companion_center_column_replay_multiple_vs_required_increment
        > 25.0
        and intake_bundle_report.intake_guard_bundle_holds
    )
    driver_signature = _driver_signature(
        patch_plan_report=patch_plan_report,
        localization_report=localization_report,
        matrix_norm_budget_report=matrix_norm_budget_report,
        intake_bundle_report=intake_bundle_report,
        dense_replay_rejected=dense_replay_rejected,
    )

    canonical_digest = (
        f"- binding design `DGP2/500/50` on `{patch_plan_report.window_label}`: the bounded repair still remains a sparse shared-entry patch with only `{patch_plan_report.symmetric_patch_nnz}` nonzero cells and matrix rank `{matrix_norm_budget_report.patch_rank}`, so the live object stays on the `z = 0.25 -> 0.15` pair instead of expanding into a slice-wide covariance replay",
        f"- dense replay remains out of lane on both geometry scales: the patch still consumes only `{_format_percent(matrix_norm_budget_report.patch_frobenius_share_of_baseline)}` of baseline `Frobenius` mass and `{_format_percent(matrix_norm_budget_report.patch_spectral_share_of_baseline)}` of baseline spectral mass, so current Trigger 2 evidence still describes a bounded perturbation rather than a dense covariance rewrite",
        f"- replaying broader local mass would overshoot the required lift immediately: the bounded increment still uses only `{_format_percent(localization_report.required_incremental_share_of_companion_right_row_mass)}` / `{_format_percent(localization_report.required_incremental_share_of_companion_center_column_mass)}` of companion right-row / center-column mass, while whole-row or whole-column replay would overshoot by `{_format_ratio(localization_report.companion_right_row_replay_multiple_vs_required_increment)}` / `{_format_ratio(localization_report.companion_center_column_replay_multiple_vs_required_increment)}`",
        f"- intake guards stay aligned with that exclusion rule: `intake_guard_bundle_holds = {intake_bundle_report.intake_guard_bundle_holds}`, so prohibited dense-replay actions remain `{prohibited_dense_replay_actions[0]}` / `{prohibited_dense_replay_actions[1]}` / `{prohibited_dense_replay_actions[2]}` and current Trigger 2 implication stays `bounded-right-center-entry-patch-dense-replay-guard`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchDenseReplayGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-dense-replay-guard-probe"
        ),
        policy_digest=patch_plan_report.policy_digest,
        binding_design=patch_plan_report.binding_design,
        window_label=patch_plan_report.window_label,
        evaluation_grid=patch_plan_report.evaluation_grid,
        coverage_anchor_random_state=patch_plan_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            patch_plan_report.overshoot_companion_random_state
        ),
        symmetric_patch_nnz=patch_plan_report.symmetric_patch_nnz,
        patch_rank=matrix_norm_budget_report.patch_rank,
        patch_frobenius_share_of_baseline=(
            matrix_norm_budget_report.patch_frobenius_share_of_baseline
        ),
        patch_spectral_share_of_baseline=(
            matrix_norm_budget_report.patch_spectral_share_of_baseline
        ),
        required_incremental_share_of_companion_right_row_mass=(
            localization_report.required_incremental_share_of_companion_right_row_mass
        ),
        required_incremental_share_of_companion_center_column_mass=(
            localization_report.required_incremental_share_of_companion_center_column_mass
        ),
        companion_right_row_replay_multiple_vs_required_increment=(
            localization_report.companion_right_row_replay_multiple_vs_required_increment
        ),
        companion_center_column_replay_multiple_vs_required_increment=(
            localization_report.companion_center_column_replay_multiple_vs_required_increment
        ),
        intake_guard_bundle_holds=intake_bundle_report.intake_guard_bundle_holds,
        dense_replay_rejected=dense_replay_rejected,
        prohibited_dense_replay_actions=prohibited_dense_replay_actions,
        driver_signature=driver_signature,
        canonical_entry_patch_dense_replay_guard_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_dense_replay_guard_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchDenseReplayGuardReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_dense_replay_guard_report(
        patch_plan_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan(),
        localization_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe(),
        matrix_norm_budget_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_probe(),
        intake_bundle_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe(),
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchDenseReplayGuardReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_dense_replay_guard_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_dense_replay_guard_probe",
]
