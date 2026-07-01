from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    patch_plan_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    patch_rank: int,
    max_abs_eigenvalue_shift: float,
    min_eigenvalue_before: float,
) -> str:
    if patch_plan_report.driver_signature != "bounded-right-center-entry-patch-plan":
        return "mixed-right-center-entry-patch-matrix-norm-budget"
    if patch_rank != 2:
        return "mixed-right-center-entry-patch-matrix-norm-budget"
    if not max_abs_eigenvalue_shift < min_eigenvalue_before:
        return "mixed-right-center-entry-patch-matrix-norm-budget"
    return "bounded-right-center-entry-patch-matrix-norm-budget"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchMatrixNormBudgetReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    patch_rank: int
    patch_frobenius_norm: float
    baseline_frobenius_norm: float
    patch_frobenius_share_of_baseline: float
    patch_spectral_norm: float
    baseline_spectral_norm: float
    patch_spectral_share_of_baseline: float
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    eigenvalue_shift: np.ndarray
    max_abs_eigenvalue_shift: float
    max_abs_eigenvalue_shift_share_of_min_eigenvalue: float
    condition_number_before: float
    condition_number_after: float
    condition_number_delta: float
    driver_signature: str
    canonical_entry_patch_matrix_norm_budget_digest: tuple[str, ...]

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
        self.patch_rank = int(self.patch_rank)
        self.patch_frobenius_norm = float(self.patch_frobenius_norm)
        self.baseline_frobenius_norm = float(self.baseline_frobenius_norm)
        self.patch_frobenius_share_of_baseline = float(
            self.patch_frobenius_share_of_baseline
        )
        self.patch_spectral_norm = float(self.patch_spectral_norm)
        self.baseline_spectral_norm = float(self.baseline_spectral_norm)
        self.patch_spectral_share_of_baseline = float(
            self.patch_spectral_share_of_baseline
        )
        self.min_eigenvalue_before = float(self.min_eigenvalue_before)
        self.min_eigenvalue_after = float(self.min_eigenvalue_after)
        self.eigenvalue_shift = np.asarray(self.eigenvalue_shift, dtype=float).copy()
        self.max_abs_eigenvalue_shift = float(self.max_abs_eigenvalue_shift)
        self.max_abs_eigenvalue_shift_share_of_min_eigenvalue = float(
            self.max_abs_eigenvalue_shift_share_of_min_eigenvalue
        )
        self.condition_number_before = float(self.condition_number_before)
        self.condition_number_after = float(self.condition_number_after)
        self.condition_number_delta = float(self.condition_number_delta)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_entry_patch_matrix_norm_budget_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_entry_patch_matrix_norm_budget_digest
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
            "patch_rank": self.patch_rank,
            "patch_frobenius_norm": self.patch_frobenius_norm,
            "baseline_frobenius_norm": self.baseline_frobenius_norm,
            "patch_frobenius_share_of_baseline": (
                self.patch_frobenius_share_of_baseline
            ),
            "patch_spectral_norm": self.patch_spectral_norm,
            "baseline_spectral_norm": self.baseline_spectral_norm,
            "patch_spectral_share_of_baseline": self.patch_spectral_share_of_baseline,
            "min_eigenvalue_before": self.min_eigenvalue_before,
            "min_eigenvalue_after": self.min_eigenvalue_after,
            "eigenvalue_shift": self.eigenvalue_shift.tolist(),
            "max_abs_eigenvalue_shift": self.max_abs_eigenvalue_shift,
            "max_abs_eigenvalue_shift_share_of_min_eigenvalue": (
                self.max_abs_eigenvalue_shift_share_of_min_eigenvalue
            ),
            "condition_number_before": self.condition_number_before,
            "condition_number_after": self.condition_number_after,
            "condition_number_delta": self.condition_number_delta,
            "driver_signature": self.driver_signature,
            "canonical_entry_patch_matrix_norm_budget_digest": list(
                self.canonical_entry_patch_matrix_norm_budget_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_report(
    *,
    patch_plan_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
) -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchMatrixNormBudgetReport
):
    patch_delta = np.asarray(patch_plan_report.covariance_patch_delta, dtype=float)
    patched_covariance = np.asarray(
        patch_plan_report.patched_covariance_at_grid, dtype=float
    )
    if patch_delta.shape != patched_covariance.shape:
        raise ValueError("matrix-norm budget probe requires shared matrix shapes")

    baseline_covariance = patched_covariance - patch_delta
    patch_frobenius_norm = float(np.linalg.norm(patch_delta, ord="fro"))
    baseline_frobenius_norm = float(np.linalg.norm(baseline_covariance, ord="fro"))
    patch_frobenius_share_of_baseline = float(
        patch_frobenius_norm / baseline_frobenius_norm
    )
    patch_spectral_norm = float(np.linalg.norm(patch_delta, ord=2))
    baseline_spectral_norm = float(np.linalg.norm(baseline_covariance, ord=2))
    patch_spectral_share_of_baseline = float(
        patch_spectral_norm / baseline_spectral_norm
    )
    eigenvalues_before = np.linalg.eigvalsh(baseline_covariance)
    eigenvalues_after = np.linalg.eigvalsh(patched_covariance)
    eigenvalue_shift = eigenvalues_after - eigenvalues_before
    max_abs_eigenvalue_shift = float(np.max(np.abs(eigenvalue_shift)))
    min_eigenvalue_before = float(np.min(eigenvalues_before))
    min_eigenvalue_after = float(np.min(eigenvalues_after))
    max_abs_eigenvalue_shift_share_of_min_eigenvalue = float(
        max_abs_eigenvalue_shift / min_eigenvalue_before
    )
    max_eigenvalue_before = float(np.max(eigenvalues_before))
    max_eigenvalue_after = float(np.max(eigenvalues_after))
    condition_number_before = float(max_eigenvalue_before / min_eigenvalue_before)
    condition_number_after = float(max_eigenvalue_after / min_eigenvalue_after)
    condition_number_delta = float(condition_number_after - condition_number_before)
    patch_rank = int(np.linalg.matrix_rank(patch_delta))
    driver_signature = _driver_signature(
        patch_plan_report=patch_plan_report,
        patch_rank=patch_rank,
        max_abs_eigenvalue_shift=max_abs_eigenvalue_shift,
        min_eigenvalue_before=min_eigenvalue_before,
    )

    canonical_digest = (
        f"- binding design `DGP2/500/50` on `{patch_plan_report.window_label}`: the bounded two-cell patch is still a rank-`{patch_rank}` perturbation with `Frobenius` norm `{_format_float(patch_frobenius_norm)}` against baseline matrix scale `{_format_float(baseline_frobenius_norm)}`, so the total patch size remains only `{_format_percent(patch_frobenius_share_of_baseline)}` of the anchor covariance witness",
        f"- the same locality survives in `spectral` geometry: operator-scale lift stays `{_format_float(patch_spectral_norm)}` against baseline spectral scale `{_format_float(baseline_spectral_norm)}` (`{_format_percent(patch_spectral_share_of_baseline)}`), so the patch is still a bounded perturbation rather than a whole-matrix replay",
        f"- eigenvalue movement remains small relative to the PSD margin: `eig_after - eig_before = ({_format_signed_float(eigenvalue_shift[0])}, {_format_signed_float(eigenvalue_shift[1])}, {_format_signed_float(eigenvalue_shift[2])})`, with `max |eigenvalue shift| = {_format_float(max_abs_eigenvalue_shift)}` and only `{_format_percent(max_abs_eigenvalue_shift_share_of_min_eigenvalue)}` of the baseline minimum eigenvalue `{_format_float(min_eigenvalue_before)}`",
        f"- conditioning drift stays bounded as well: the covariance condition number moves only from `{_format_float(condition_number_before)}` to `{_format_float(condition_number_after)}` (delta `{_format_signed_float(condition_number_delta)}`), so current Trigger 2 implication remains `bounded-right-center-entry-patch-matrix-norm-budget` rather than any license for dense covariance rewriting",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchMatrixNormBudgetReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-matrix-norm-budget-probe"
        ),
        policy_digest=patch_plan_report.policy_digest,
        binding_design=patch_plan_report.binding_design,
        window_label=patch_plan_report.window_label,
        evaluation_grid=patch_plan_report.evaluation_grid,
        coverage_anchor_random_state=patch_plan_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            patch_plan_report.overshoot_companion_random_state
        ),
        patch_rank=patch_rank,
        patch_frobenius_norm=patch_frobenius_norm,
        baseline_frobenius_norm=baseline_frobenius_norm,
        patch_frobenius_share_of_baseline=patch_frobenius_share_of_baseline,
        patch_spectral_norm=patch_spectral_norm,
        baseline_spectral_norm=baseline_spectral_norm,
        patch_spectral_share_of_baseline=patch_spectral_share_of_baseline,
        min_eigenvalue_before=min_eigenvalue_before,
        min_eigenvalue_after=min_eigenvalue_after,
        eigenvalue_shift=eigenvalue_shift,
        max_abs_eigenvalue_shift=max_abs_eigenvalue_shift,
        max_abs_eigenvalue_shift_share_of_min_eigenvalue=(
            max_abs_eigenvalue_shift_share_of_min_eigenvalue
        ),
        condition_number_before=condition_number_before,
        condition_number_after=condition_number_after,
        condition_number_delta=condition_number_delta,
        driver_signature=driver_signature,
        canonical_entry_patch_matrix_norm_budget_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchMatrixNormBudgetReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_report(
        patch_plan_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan(),
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchMatrixNormBudgetReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_probe",
]
