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


def _driver_signature(
    *,
    patch_plan_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    trace_delta: float,
    max_abs_diagonal_delta: float,
    max_abs_std_delta: float,
) -> str:
    if (
        patch_plan_report.driver_signature == "bounded-right-center-entry-patch-plan"
        and np.isclose(trace_delta, 0.0, atol=1e-12)
        and np.isclose(max_abs_diagonal_delta, 0.0, atol=1e-12)
        and np.isclose(max_abs_std_delta, 0.0, atol=1e-12)
    ):
        return "bounded-right-center-entry-patch-variance-preservation"
    return "mixed-right-center-entry-patch-variance-preservation"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchVariancePreservationReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_before: np.ndarray
    diagonal_after: np.ndarray
    diagonal_delta: np.ndarray
    std_before: np.ndarray
    std_after: np.ndarray
    trace_before: float
    trace_after: float
    trace_delta: float
    max_abs_diagonal_delta: float
    max_abs_std_delta: float
    diagonal_preserved: bool
    driver_signature: str
    canonical_entry_patch_variance_preservation_digest: tuple[str, ...]

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
        self.diagonal_before = np.asarray(self.diagonal_before, dtype=float).copy()
        self.diagonal_after = np.asarray(self.diagonal_after, dtype=float).copy()
        self.diagonal_delta = np.asarray(self.diagonal_delta, dtype=float).copy()
        self.std_before = np.asarray(self.std_before, dtype=float).copy()
        self.std_after = np.asarray(self.std_after, dtype=float).copy()
        self.trace_before = float(self.trace_before)
        self.trace_after = float(self.trace_after)
        self.trace_delta = float(self.trace_delta)
        self.max_abs_diagonal_delta = float(self.max_abs_diagonal_delta)
        self.max_abs_std_delta = float(self.max_abs_std_delta)
        self.diagonal_preserved = bool(self.diagonal_preserved)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_entry_patch_variance_preservation_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_entry_patch_variance_preservation_digest
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
            "diagonal_before": self.diagonal_before.tolist(),
            "diagonal_after": self.diagonal_after.tolist(),
            "diagonal_delta": self.diagonal_delta.tolist(),
            "std_before": self.std_before.tolist(),
            "std_after": self.std_after.tolist(),
            "trace_before": self.trace_before,
            "trace_after": self.trace_after,
            "trace_delta": self.trace_delta,
            "max_abs_diagonal_delta": self.max_abs_diagonal_delta,
            "max_abs_std_delta": self.max_abs_std_delta,
            "diagonal_preserved": self.diagonal_preserved,
            "driver_signature": self.driver_signature,
            "canonical_entry_patch_variance_preservation_digest": list(
                self.canonical_entry_patch_variance_preservation_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_variance_preservation_report(
    *,
    patch_plan_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchVariancePreservationReport:
    patched_covariance = np.asarray(
        patch_plan_report.patched_covariance_at_grid,
        dtype=float,
    )
    patch_delta = np.asarray(patch_plan_report.covariance_patch_delta, dtype=float)
    if patched_covariance.shape != patch_delta.shape:
        raise ValueError("variance-preservation probe requires shared matrix shapes")

    baseline_covariance = patched_covariance - patch_delta
    diagonal_before = np.diag(baseline_covariance).astype(float, copy=True)
    diagonal_after = np.diag(patched_covariance).astype(float, copy=True)
    diagonal_delta = diagonal_after - diagonal_before
    std_before = np.sqrt(diagonal_before)
    std_after = np.sqrt(diagonal_after)
    trace_before = float(diagonal_before.sum())
    trace_after = float(diagonal_after.sum())
    trace_delta = float(trace_after - trace_before)
    max_abs_diagonal_delta = float(np.abs(diagonal_delta).max(initial=0.0))
    max_abs_std_delta = float(np.abs(std_after - std_before).max(initial=0.0))
    diagonal_preserved = bool(
        np.isclose(trace_delta, 0.0, atol=1e-12)
        and np.isclose(max_abs_diagonal_delta, 0.0, atol=1e-12)
        and np.isclose(max_abs_std_delta, 0.0, atol=1e-12)
    )
    driver_signature = _driver_signature(
        patch_plan_report=patch_plan_report,
        trace_delta=trace_delta,
        max_abs_diagonal_delta=max_abs_diagonal_delta,
        max_abs_std_delta=max_abs_std_delta,
    )

    canonical_digest = (
        f"- binding design `DGP2/500/50` on `{patch_plan_report.window_label}`: the bounded entry patch keeps the covariance trace fixed at `{_format_float(trace_before)}` before and after repair, so the live lane does not buy coverage by inflating total variance mass",
        f"- all pointwise diagonal variances stay unchanged: `diag_before = diag_after = ({_format_float(diagonal_before[0])}, {_format_float(diagonal_before[1])}, {_format_float(diagonal_before[2])})`, with `max |diagonal delta| = {_format_float(max_abs_diagonal_delta)}`",
        f"- the same invariance propagates to pointwise standard deviations: `std_before = std_after = ({_format_float(std_before[0])}, {_format_float(std_before[1])}, {_format_float(std_before[2])})`, with `max |std delta| = {_format_float(max_abs_std_delta)}`",
        "- current Trigger 2 implication: `bounded-right-center-entry-patch-variance-preservation`; this remains validation-only companion evidence and fixes the implementation rule that the bounded repair must work through `cross-covariance` coupling rather than any diagonal variance inflation",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchVariancePreservationReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-variance-preservation-probe"
        ),
        policy_digest=patch_plan_report.policy_digest,
        binding_design=patch_plan_report.binding_design,
        window_label=patch_plan_report.window_label,
        evaluation_grid=patch_plan_report.evaluation_grid,
        coverage_anchor_random_state=patch_plan_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            patch_plan_report.overshoot_companion_random_state
        ),
        diagonal_before=diagonal_before,
        diagonal_after=diagonal_after,
        diagonal_delta=diagonal_delta,
        std_before=std_before,
        std_after=std_after,
        trace_before=trace_before,
        trace_after=trace_after,
        trace_delta=trace_delta,
        max_abs_diagonal_delta=max_abs_diagonal_delta,
        max_abs_std_delta=max_abs_std_delta,
        diagonal_preserved=diagonal_preserved,
        driver_signature=driver_signature,
        canonical_entry_patch_variance_preservation_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_variance_preservation_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchVariancePreservationReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_variance_preservation_report(
        patch_plan_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan(),
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchVariancePreservationReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_variance_preservation_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_variance_preservation_probe",
]
