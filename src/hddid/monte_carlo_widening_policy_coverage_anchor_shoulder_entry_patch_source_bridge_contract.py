from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    patch_plan_signature: str,
    source_patch_signature: str,
    validation_patch_nnz: int,
    preserved_left_center_covariance: float,
    source_patch_share_of_companion_diagonal_gap: float,
    source_residual_companion_gap_share_after_patch: float,
) -> str:
    if (
        patch_plan_signature == "bounded-right-center-entry-patch-plan"
        and source_patch_signature
        == "bounded-positive-first-sine-omega-diagonal-patch-fraction"
        and validation_patch_nnz == 2
        and preserved_left_center_covariance > 0.0
        and 0.20 < source_patch_share_of_companion_diagonal_gap < 0.25
        and source_residual_companion_gap_share_after_patch > 0.75
    ):
        return "validation-patch-to-first-sine-omega-diagonal-bridge"
    return "mixed-entry-patch-source-bridge"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    validation_patch_increment: float
    validation_patch_cells: tuple[tuple[int, int], tuple[int, int]]
    validation_patch_nnz: int
    preserved_left_center_covariance: float
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    source_shared_vf_entry_label: str
    source_target_omega_diagonal_entry: float
    source_required_omega_diagonal_increment: float
    source_patch_share_of_companion_diagonal_gap: float
    source_residual_companion_gap_share_after_patch: float
    driver_signature: str
    canonical_source_bridge_digest: tuple[str, ...]

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
        self.validation_patch_increment = float(self.validation_patch_increment)
        self.validation_patch_cells = tuple(
            (int(row), int(column)) for row, column in self.validation_patch_cells
        )
        self.validation_patch_nnz = int(self.validation_patch_nnz)
        self.preserved_left_center_covariance = float(
            self.preserved_left_center_covariance
        )
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.source_shared_vf_entry_label = str(
            self.source_shared_vf_entry_label
        ).strip()
        self.source_target_omega_diagonal_entry = float(
            self.source_target_omega_diagonal_entry
        )
        self.source_required_omega_diagonal_increment = float(
            self.source_required_omega_diagonal_increment
        )
        self.source_patch_share_of_companion_diagonal_gap = float(
            self.source_patch_share_of_companion_diagonal_gap
        )
        self.source_residual_companion_gap_share_after_patch = float(
            self.source_residual_companion_gap_share_after_patch
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_source_bridge_digest = tuple(
            str(line).rstrip() for line in self.canonical_source_bridge_digest
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
            "validation_patch_increment": self.validation_patch_increment,
            "validation_patch_cells": [
                list(cell) for cell in self.validation_patch_cells
            ],
            "validation_patch_nnz": self.validation_patch_nnz,
            "preserved_left_center_covariance": self.preserved_left_center_covariance,
            "source_diagonal_coordinate": self.source_diagonal_coordinate,
            "source_diagonal_basis_label": self.source_diagonal_basis_label,
            "source_shared_vf_entry_label": self.source_shared_vf_entry_label,
            "source_target_omega_diagonal_entry": (
                self.source_target_omega_diagonal_entry
            ),
            "source_required_omega_diagonal_increment": (
                self.source_required_omega_diagonal_increment
            ),
            "source_patch_share_of_companion_diagonal_gap": (
                self.source_patch_share_of_companion_diagonal_gap
            ),
            "source_residual_companion_gap_share_after_patch": (
                self.source_residual_companion_gap_share_after_patch
            ),
            "driver_signature": self.driver_signature,
            "canonical_source_bridge_digest": list(self.canonical_source_bridge_digest),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract_report(
    *,
    patch_plan_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport
    ),
    source_patch_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport:
    if patch_plan_report.policy_digest != source_patch_report.policy_digest:
        raise ValueError("entry patch source bridge requires a shared policy digest")
    if patch_plan_report.binding_design != source_patch_report.binding_design:
        raise ValueError("entry patch source bridge requires a shared binding design")
    if patch_plan_report.window_label != source_patch_report.window_label:
        raise ValueError("entry patch source bridge requires a shared window label")
    if (
        patch_plan_report.coverage_anchor_random_state
        != source_patch_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        patch_plan_report.overshoot_companion_random_state
        != source_patch_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if patch_plan_report.symmetric_patch_nnz <= 0:
        raise ValueError("validation patch must contain at least one active cell")
    if source_patch_report.required_patch_share_of_omega_diagonal_gap <= 0.0:
        raise ValueError("source patch share must stay positive")

    validation_patch_cells = (
        (
            patch_plan_report.failing_right_shoulder_index,
            patch_plan_report.center_index,
        ),
        (
            patch_plan_report.center_index,
            patch_plan_report.failing_right_shoulder_index,
        ),
    )

    driver_signature = _driver_signature(
        patch_plan_signature=patch_plan_report.driver_signature,
        source_patch_signature=source_patch_report.driver_signature,
        validation_patch_nnz=patch_plan_report.symmetric_patch_nnz,
        preserved_left_center_covariance=patch_plan_report.preserved_left_center_covariance,
        source_patch_share_of_companion_diagonal_gap=(
            source_patch_report.required_patch_share_of_omega_diagonal_gap
        ),
        source_residual_companion_gap_share_after_patch=(
            source_patch_report.residual_companion_gap_share_after_patch
        ),
    )

    canonical_digest = (
        f"- the validation-only covariance witness stays exact on `{'/'.join(map(str, patch_plan_report.binding_design))}` / `{patch_plan_report.window_label}`: bounded repair still means only `{_format_signed(patch_plan_report.signed_right_center_increment)}` on symmetric cells `[2,1]` / `[1,2]`, with left-center support `covariance(0.05, 0.15) = {_format_float(patch_plan_report.preserved_left_center_covariance)}` preserved and no broader row/column replay",
        f"- the same bounded lane already has a source-level realization target on positive first-sine diagonal mass: `omega_f_hat[{source_patch_report.diagonal_coordinate},{source_patch_report.diagonal_coordinate}]` only needs `{_format_signed(source_patch_report.required_omega_diagonal_increment)}` up to `{_format_float(source_patch_report.bounded_target_omega_diagonal_entry)}`, which consumes `{_format_percent(source_patch_report.required_patch_share_of_omega_diagonal_gap)}` of the companion diagonal gap and leaves `{_format_percent(source_patch_report.residual_companion_gap_share_after_patch)}` unused even before any coordinate-`{source_patch_report.diagonal_coordinate}` axis replay",
        f"- current Trigger 2 implication: `{driver_signature}`; implementation should keep the raw covariance delta as validation-only evidence and route the next source-level explanation toward bounded positive `omega_f_hat[{source_patch_report.diagonal_coordinate},{source_patch_report.diagonal_coordinate}]` restoration rather than direct covariance edits, whole-axis replay, or whole-matrix cleanup",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-source-bridge-contract"
        ),
        policy_digest=patch_plan_report.policy_digest,
        binding_design=patch_plan_report.binding_design,
        window_label=patch_plan_report.window_label,
        coverage_anchor_random_state=patch_plan_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            patch_plan_report.overshoot_companion_random_state
        ),
        center_grid_value=patch_plan_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            patch_plan_report.failing_right_shoulder_grid_value
        ),
        validation_patch_increment=patch_plan_report.signed_right_center_increment,
        validation_patch_cells=validation_patch_cells,
        validation_patch_nnz=patch_plan_report.symmetric_patch_nnz,
        preserved_left_center_covariance=(
            patch_plan_report.preserved_left_center_covariance
        ),
        source_diagonal_coordinate=source_patch_report.diagonal_coordinate,
        source_diagonal_basis_label=source_patch_report.diagonal_basis_label,
        source_shared_vf_entry_label=source_patch_report.shared_vf_entry_label,
        source_target_omega_diagonal_entry=(
            source_patch_report.bounded_target_omega_diagonal_entry
        ),
        source_required_omega_diagonal_increment=(
            source_patch_report.required_omega_diagonal_increment
        ),
        source_patch_share_of_companion_diagonal_gap=(
            source_patch_report.required_patch_share_of_omega_diagonal_gap
        ),
        source_residual_companion_gap_share_after_patch=(
            source_patch_report.residual_companion_gap_share_after_patch
        ),
        driver_signature=driver_signature,
        canonical_source_bridge_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract_report(
        patch_plan_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan(),
        source_patch_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract(),
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract",
]
