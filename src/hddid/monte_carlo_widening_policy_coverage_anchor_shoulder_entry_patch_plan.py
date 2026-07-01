from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


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


def _driver_signature(
    *,
    execution_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
    covariance_patch_delta: np.ndarray,
    patched_covariance_at_grid: np.ndarray,
    center_index: int,
    shoulder_index: int,
) -> str:
    if (
        execution_contract_report.driver_signature
        != "bounded-right-center-execution-contract"
    ):
        return "mixed-right-center-entry-patch-plan"
    if covariance_patch_delta.shape != patched_covariance_at_grid.shape:
        return "mixed-right-center-entry-patch-plan"
    if not np.allclose(covariance_patch_delta, covariance_patch_delta.T):
        return "mixed-right-center-entry-patch-plan"
    if np.count_nonzero(covariance_patch_delta) != 2:
        return "mixed-right-center-entry-patch-plan"
    if not isclose(
        float(covariance_patch_delta[shoulder_index, center_index]),
        execution_contract_report.required_incremental_right_center_covariance_lift,
        abs_tol=1e-12,
    ):
        return "mixed-right-center-entry-patch-plan"
    if not isclose(
        float(covariance_patch_delta[center_index, shoulder_index]),
        execution_contract_report.required_incremental_right_center_covariance_lift,
        abs_tol=1e-12,
    ):
        return "mixed-right-center-entry-patch-plan"
    return "bounded-right-center-entry-patch-plan"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_index: int
    failing_right_shoulder_index: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    current_signed_right_center_covariance: float
    target_signed_right_center_covariance: float
    signed_right_center_increment: float
    preserved_left_center_covariance: float
    covariance_patch_delta: np.ndarray
    patched_covariance_at_grid: np.ndarray
    symmetric_patch_nnz: int
    driver_signature: str
    canonical_entry_patch_plan_digest: tuple[str, ...]

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
        self.center_index = int(self.center_index)
        self.failing_right_shoulder_index = int(self.failing_right_shoulder_index)
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.current_signed_right_center_covariance = float(
            self.current_signed_right_center_covariance
        )
        self.target_signed_right_center_covariance = float(
            self.target_signed_right_center_covariance
        )
        self.signed_right_center_increment = float(self.signed_right_center_increment)
        self.preserved_left_center_covariance = float(
            self.preserved_left_center_covariance
        )
        self.covariance_patch_delta = np.asarray(
            self.covariance_patch_delta, dtype=float
        ).copy()
        self.patched_covariance_at_grid = np.asarray(
            self.patched_covariance_at_grid, dtype=float
        ).copy()
        self.symmetric_patch_nnz = int(self.symmetric_patch_nnz)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_entry_patch_plan_digest = tuple(
            str(line).rstrip() for line in self.canonical_entry_patch_plan_digest
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
            "center_index": self.center_index,
            "failing_right_shoulder_index": self.failing_right_shoulder_index,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "current_signed_right_center_covariance": (
                self.current_signed_right_center_covariance
            ),
            "target_signed_right_center_covariance": (
                self.target_signed_right_center_covariance
            ),
            "signed_right_center_increment": self.signed_right_center_increment,
            "preserved_left_center_covariance": self.preserved_left_center_covariance,
            "covariance_patch_delta": self.covariance_patch_delta.tolist(),
            "patched_covariance_at_grid": self.patched_covariance_at_grid.tolist(),
            "symmetric_patch_nnz": self.symmetric_patch_nnz,
            "driver_signature": self.driver_signature,
            "canonical_entry_patch_plan_digest": list(
                self.canonical_entry_patch_plan_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan_report(
    *,
    execution_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport:
    if (
        execution_contract_report.policy_digest
        != seed_window_covariance_report.policy_digest
    ):
        raise ValueError("entry patch plan requires a shared policy digest")
    if (
        execution_contract_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError("entry patch plan requires a shared binding design")
    if (
        execution_contract_report.window_label
        != seed_window_covariance_report.window_label
    ):
        raise ValueError("entry patch plan requires a shared window label")
    if (
        execution_contract_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or execution_contract_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError("entry patch plan requires the same anchor/companion seeds")

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    center_index = _grid_index(
        evaluation_grid,
        execution_contract_report.center_grid_value,
        label="center",
    )
    shoulder_index = _grid_index(
        evaluation_grid,
        execution_contract_report.failing_right_shoulder_grid_value,
        label="failing_right_shoulder",
    )
    left_index = _grid_index(
        evaluation_grid,
        execution_contract_report.left_shoulder_grid_value,
        label="left_shoulder",
    )

    anchor_covariance = np.asarray(
        seed_window_covariance_report.coverage_anchor_contract.covariance_at_grid,
        dtype=float,
    )
    if (
        anchor_covariance.ndim != 2
        or anchor_covariance.shape[0] != anchor_covariance.shape[1]
    ):
        raise ValueError("entry patch plan requires a square covariance matrix")
    if not np.allclose(anchor_covariance, anchor_covariance.T):
        raise ValueError("entry patch plan requires a symmetric covariance matrix")

    current_signed_entry = float(anchor_covariance[shoulder_index, center_index])
    if not isclose(
        current_signed_entry,
        execution_contract_report.current_abs_right_center_covariance,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "entry patch plan requires signed anchor entry to match execution contract"
        )

    increment = float(
        execution_contract_report.required_incremental_right_center_covariance_lift
    )
    target_signed_entry = current_signed_entry + increment
    if not isclose(
        target_signed_entry,
        execution_contract_report.required_abs_right_center_covariance,
        abs_tol=1e-12,
    ):
        raise ValueError("entry patch plan target entry drifts from execution contract")

    covariance_patch_delta = np.zeros_like(anchor_covariance)
    covariance_patch_delta[shoulder_index, center_index] = increment
    covariance_patch_delta[center_index, shoulder_index] = increment
    patched_covariance_at_grid = anchor_covariance + covariance_patch_delta

    driver_signature = _driver_signature(
        execution_contract_report=execution_contract_report,
        covariance_patch_delta=covariance_patch_delta,
        patched_covariance_at_grid=patched_covariance_at_grid,
        center_index=center_index,
        shoulder_index=shoulder_index,
    )

    canonical_digest = (
        f"- binding design `DGP2/500/50` on `{execution_contract_report.window_label}`: the execution contract already fixes a single signed target, so the live patch only raises `covariance({_format_grid_value(execution_contract_report.failing_right_shoulder_grid_value)}, {_format_grid_value(execution_contract_report.center_grid_value)})` from `{_format_float(current_signed_entry)}` to `{_format_float(target_signed_entry)}` (increment `{_format_signed_float(increment)}`)",
        f"- the patch is therefore a symmetric two-cell delta on the shared covariance matrix: only entries `[row={shoulder_index}, col={center_index}]` and `[row={center_index}, col={shoulder_index}]` move by `{_format_signed_float(increment)}`, while the preserved left-center entry `covariance({_format_grid_value(execution_contract_report.left_shoulder_grid_value)}, {_format_grid_value(execution_contract_report.center_grid_value)})` stays `{_format_float(anchor_covariance[left_index, center_index])}`",
        f"- preserve left-side support remains explicit: cross-shoulder covariance `covariance({_format_grid_value(execution_contract_report.left_shoulder_grid_value)}, {_format_grid_value(execution_contract_report.failing_right_shoulder_grid_value)})` stays `{_format_signed_float(anchor_covariance[left_index, shoulder_index])}` and no diagonal entry is touched, so this object remains a minimal implementation companion rather than whole-row / whole-column replay",
        "- current Trigger 2 implication: `bounded-right-center-entry-patch-plan`; implementation intake may consume this as an exact patch object, but it should remain validation-only until a real estimator path proves it can preserve the existing left-side support contract",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-plan",
        policy_digest=execution_contract_report.policy_digest,
        binding_design=execution_contract_report.binding_design,
        window_label=execution_contract_report.window_label,
        evaluation_grid=evaluation_grid,
        coverage_anchor_random_state=execution_contract_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            execution_contract_report.overshoot_companion_random_state
        ),
        center_index=center_index,
        failing_right_shoulder_index=shoulder_index,
        center_grid_value=execution_contract_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            execution_contract_report.failing_right_shoulder_grid_value
        ),
        current_signed_right_center_covariance=current_signed_entry,
        target_signed_right_center_covariance=target_signed_entry,
        signed_right_center_increment=increment,
        preserved_left_center_covariance=float(
            anchor_covariance[left_index, center_index]
        ),
        covariance_patch_delta=covariance_patch_delta,
        patched_covariance_at_grid=patched_covariance_at_grid,
        symmetric_patch_nnz=int(np.count_nonzero(covariance_patch_delta)),
        driver_signature=driver_signature,
        canonical_entry_patch_plan_digest=canonical_digest,
    )


def _make_canonical_entry_patch_plan_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport
):
    covariance_patch_delta = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.6361273081544214],
            [0.0, 1.6361273081544214, 0.0],
        ],
        dtype=float,
    )
    patched_covariance_at_grid = np.array(
        [
            [11.893602601786055, 0.26869174013344194, -1.132110735050385],
            [0.2686917401334395, 10.514643248332822, 1.730624991807326],
            [-1.1321107350503887, 1.7306249918073264, 16.160708225114693],
        ],
        dtype=float,
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-plan"
        ),
        policy_digest=(
            "label=bounded-n500-p50",
            "max_total_runtime_seconds=240.0",
            "max_random_states=8",
            "stop_on_first_typed_invalidity=True",
            "min_nonparametric_coverage=0.85",
        ),
        binding_design=("DGP2", 500, 50),
        window_label="near_zero_grid",
        evaluation_grid=(0.05, 0.15, 0.25),
        coverage_anchor_random_state=202,
        overshoot_companion_random_state=505,
        center_index=1,
        failing_right_shoulder_index=2,
        center_grid_value=0.15,
        failing_right_shoulder_grid_value=0.25,
        current_signed_right_center_covariance=0.09449768365290503,
        target_signed_right_center_covariance=1.7306249918073264,
        signed_right_center_increment=1.6361273081544214,
        preserved_left_center_covariance=0.26869174013344194,
        covariance_patch_delta=covariance_patch_delta,
        patched_covariance_at_grid=patched_covariance_at_grid,
        symmetric_patch_nnz=2,
        driver_signature="bounded-right-center-entry-patch-plan",
        canonical_entry_patch_plan_digest=(
            "- binding design `DGP2/500/50` on `near_zero_grid`: the execution contract already fixes a single signed target, so the live patch only raises `covariance(0.25, 0.15)` from `0.094` to `1.731` (increment `+1.636`)",
            "- the patch is therefore a symmetric two-cell delta on the shared covariance matrix: only entries `[row=2, col=1]` and `[row=1, col=2]` move by `+1.636`, while the preserved left-center entry `covariance(0.05, 0.15)` stays `0.269`",
            "- preserve left-side support remains explicit: cross-shoulder covariance `covariance(0.05, 0.25)` stays `-1.132` and no diagonal entry is touched, so this object remains a minimal implementation companion rather than whole-row / whole-column replay",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport
):
    return _make_canonical_entry_patch_plan_report()
