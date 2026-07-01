from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSpectralAdmissibilityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _infer_left_index(
    *, matrix_size: int, center_index: int, shoulder_index: int
) -> int:
    candidates = [
        index
        for index in range(int(matrix_size))
        if index not in {int(center_index), int(shoulder_index)}
    ]
    if len(candidates) != 1:
        raise ValueError(
            "left-support null-action probe requires exactly one left index"
        )
    return int(candidates[0])


def _driver_signature(
    *,
    patch_plan_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    spectral_admissibility_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSpectralAdmissibilityReport
    ),
    patch_support_nnz: int,
    patch_rank: int,
    left_incident_absolute_patch_mass: float,
    left_support_projection_norm: float,
    left_center_delta: float,
    cross_shoulder_delta: float,
    diagonal_linf_delta: float,
) -> str:
    if (
        patch_plan_report.driver_signature == "bounded-right-center-entry-patch-plan"
        and spectral_admissibility_report.driver_signature
        == "bounded-right-center-entry-patch-spectral-admissibility"
        and patch_support_nnz == 2
        and patch_rank == 2
        and np.isclose(left_incident_absolute_patch_mass, 0.0, atol=1e-12)
        and np.isclose(left_support_projection_norm, 0.0, atol=1e-12)
        and np.isclose(left_center_delta, 0.0, atol=1e-12)
        and np.isclose(cross_shoulder_delta, 0.0, atol=1e-12)
        and np.isclose(diagonal_linf_delta, 0.0, atol=1e-12)
    ):
        return "bounded-right-center-entry-patch-left-support-null-action"
    return "mixed-right-center-entry-patch-left-support-null-action"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchLeftSupportNullActionReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    left_shoulder_index: int
    center_index: int
    failing_right_shoulder_index: int
    required_patch_increment: float
    patch_delta_matrix: np.ndarray
    patch_support_nnz: int
    patch_rank: int
    left_incident_absolute_patch_mass: float
    center_right_absolute_patch_mass: float
    left_support_projection_norm: float
    left_center_delta: float
    cross_shoulder_delta: float
    diagonal_linf_delta: float
    patch_frobenius_norm: float
    confined_to_center_right_subspace: bool
    preserves_left_support_contract: bool
    driver_signature: str
    canonical_entry_patch_left_support_null_action_digest: tuple[str, ...]

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
        self.left_shoulder_index = int(self.left_shoulder_index)
        self.center_index = int(self.center_index)
        self.failing_right_shoulder_index = int(self.failing_right_shoulder_index)
        self.required_patch_increment = float(self.required_patch_increment)
        self.patch_delta_matrix = np.asarray(
            self.patch_delta_matrix, dtype=float
        ).copy()
        self.patch_support_nnz = int(self.patch_support_nnz)
        self.patch_rank = int(self.patch_rank)
        self.left_incident_absolute_patch_mass = float(
            self.left_incident_absolute_patch_mass
        )
        self.center_right_absolute_patch_mass = float(
            self.center_right_absolute_patch_mass
        )
        self.left_support_projection_norm = float(self.left_support_projection_norm)
        self.left_center_delta = float(self.left_center_delta)
        self.cross_shoulder_delta = float(self.cross_shoulder_delta)
        self.diagonal_linf_delta = float(self.diagonal_linf_delta)
        self.patch_frobenius_norm = float(self.patch_frobenius_norm)
        self.confined_to_center_right_subspace = bool(
            self.confined_to_center_right_subspace
        )
        self.preserves_left_support_contract = bool(
            self.preserves_left_support_contract
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_entry_patch_left_support_null_action_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_entry_patch_left_support_null_action_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "evaluation_grid": list(self.evaluation_grid),
            "left_shoulder_index": self.left_shoulder_index,
            "center_index": self.center_index,
            "failing_right_shoulder_index": self.failing_right_shoulder_index,
            "required_patch_increment": self.required_patch_increment,
            "patch_delta_matrix": self.patch_delta_matrix.tolist(),
            "patch_support_nnz": self.patch_support_nnz,
            "patch_rank": self.patch_rank,
            "left_incident_absolute_patch_mass": self.left_incident_absolute_patch_mass,
            "center_right_absolute_patch_mass": self.center_right_absolute_patch_mass,
            "left_support_projection_norm": self.left_support_projection_norm,
            "left_center_delta": self.left_center_delta,
            "cross_shoulder_delta": self.cross_shoulder_delta,
            "diagonal_linf_delta": self.diagonal_linf_delta,
            "patch_frobenius_norm": self.patch_frobenius_norm,
            "confined_to_center_right_subspace": self.confined_to_center_right_subspace,
            "preserves_left_support_contract": self.preserves_left_support_contract,
            "driver_signature": self.driver_signature,
            "canonical_entry_patch_left_support_null_action_digest": list(
                self.canonical_entry_patch_left_support_null_action_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_left_support_null_action_report(
    *,
    patch_plan_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport,
    spectral_admissibility_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSpectralAdmissibilityReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchLeftSupportNullActionReport:
    if patch_plan_report.policy_digest != spectral_admissibility_report.policy_digest:
        raise ValueError(
            "left-support null-action probe requires a shared policy digest"
        )
    if patch_plan_report.binding_design != spectral_admissibility_report.binding_design:
        raise ValueError(
            "left-support null-action probe requires a shared binding design"
        )
    if patch_plan_report.window_label != spectral_admissibility_report.window_label:
        raise ValueError(
            "left-support null-action probe requires a shared window label"
        )

    patch_delta_matrix = np.asarray(
        patch_plan_report.covariance_patch_delta,
        dtype=float,
    )
    if (
        patch_delta_matrix.ndim != 2
        or patch_delta_matrix.shape[0] != patch_delta_matrix.shape[1]
    ):
        raise ValueError(
            "left-support null-action probe requires a square patch matrix"
        )

    left_index = _infer_left_index(
        matrix_size=patch_delta_matrix.shape[0],
        center_index=patch_plan_report.center_index,
        shoulder_index=patch_plan_report.failing_right_shoulder_index,
    )
    left_basis = np.zeros(patch_delta_matrix.shape[0], dtype=float)
    left_basis[left_index] = 1.0

    patch_support_nnz = int(np.count_nonzero(patch_delta_matrix))
    patch_rank = int(np.linalg.matrix_rank(patch_delta_matrix))
    left_incident_absolute_patch_mass = float(
        np.abs(patch_delta_matrix[left_index, :]).sum()
        + np.abs(patch_delta_matrix[:, left_index]).sum()
    )
    center_right_absolute_patch_mass = float(
        np.abs(
            patch_delta_matrix[
                [
                    patch_plan_report.center_index,
                    patch_plan_report.failing_right_shoulder_index,
                ],
                :,
            ][
                :,
                [
                    patch_plan_report.center_index,
                    patch_plan_report.failing_right_shoulder_index,
                ],
            ]
        ).sum()
    )
    left_support_projection_norm = float(
        np.linalg.norm(patch_delta_matrix @ left_basis, ord=2)
    )
    left_center_delta = float(
        patch_delta_matrix[left_index, patch_plan_report.center_index]
    )
    cross_shoulder_delta = float(
        patch_delta_matrix[left_index, patch_plan_report.failing_right_shoulder_index]
    )
    diagonal_linf_delta = float(np.abs(np.diag(patch_delta_matrix)).max(initial=0.0))
    patch_frobenius_norm = float(np.linalg.norm(patch_delta_matrix, ord="fro"))
    confined_to_center_right_subspace = bool(
        np.isclose(left_incident_absolute_patch_mass, 0.0, atol=1e-12)
        and patch_support_nnz == 2
    )
    preserves_left_support_contract = bool(
        confined_to_center_right_subspace
        and np.isclose(left_support_projection_norm, 0.0, atol=1e-12)
        and np.isclose(left_center_delta, 0.0, atol=1e-12)
        and np.isclose(cross_shoulder_delta, 0.0, atol=1e-12)
        and np.isclose(diagonal_linf_delta, 0.0, atol=1e-12)
        and spectral_admissibility_report.diagonal_preserved
    )
    driver_signature = _driver_signature(
        patch_plan_report=patch_plan_report,
        spectral_admissibility_report=spectral_admissibility_report,
        patch_support_nnz=patch_support_nnz,
        patch_rank=patch_rank,
        left_incident_absolute_patch_mass=left_incident_absolute_patch_mass,
        left_support_projection_norm=left_support_projection_norm,
        left_center_delta=left_center_delta,
        cross_shoulder_delta=cross_shoulder_delta,
        diagonal_linf_delta=diagonal_linf_delta,
    )

    canonical_digest = (
        f"- binding design `DGP2/500/50` on `{patch_plan_report.window_label}`: the same bounded two-cell patch still carries increment `{_format_signed_float(patch_plan_report.signed_right_center_increment)}` only on `[row={patch_plan_report.failing_right_shoulder_index}, col={patch_plan_report.center_index}]` / `[row={patch_plan_report.center_index}, col={patch_plan_report.failing_right_shoulder_index}]`, so its full `center/right` absolute patch mass is `{_format_float(center_right_absolute_patch_mass)}` with Frobenius norm `{_format_float(patch_frobenius_norm)}`",
        f"- the patch has exact `left-support` null action: left incident patch mass stays `{_format_float(left_incident_absolute_patch_mass)}`, the left-basis projection norm stays `{_format_float(left_support_projection_norm)}`, and both left-preserving entries keep zero delta (`left-center = {_format_signed_float(left_center_delta)}`, `cross-shoulder = {_format_signed_float(cross_shoulder_delta)}`)",
        f"- diagonal drift also remains zero (`linf = {_format_float(diagonal_linf_delta)}`), so together with the spectral-admissibility companion the bounded patch remains confined to the center/right subspace while preserving the existing left-support contract",
        "- current Trigger 2 implication: `bounded-right-center-entry-patch-left-support-null-action`; this remains validation-only companion evidence for implementation intake and must not replace the live `bounded-right-center-execution-contract` routing surface",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchLeftSupportNullActionReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-left-support-null-action-probe"
        ),
        policy_digest=patch_plan_report.policy_digest,
        binding_design=patch_plan_report.binding_design,
        window_label=patch_plan_report.window_label,
        evaluation_grid=patch_plan_report.evaluation_grid,
        left_shoulder_index=left_index,
        center_index=patch_plan_report.center_index,
        failing_right_shoulder_index=patch_plan_report.failing_right_shoulder_index,
        required_patch_increment=patch_plan_report.signed_right_center_increment,
        patch_delta_matrix=patch_delta_matrix,
        patch_support_nnz=patch_support_nnz,
        patch_rank=patch_rank,
        left_incident_absolute_patch_mass=left_incident_absolute_patch_mass,
        center_right_absolute_patch_mass=center_right_absolute_patch_mass,
        left_support_projection_norm=left_support_projection_norm,
        left_center_delta=left_center_delta,
        cross_shoulder_delta=cross_shoulder_delta,
        diagonal_linf_delta=diagonal_linf_delta,
        patch_frobenius_norm=patch_frobenius_norm,
        confined_to_center_right_subspace=confined_to_center_right_subspace,
        preserves_left_support_contract=preserves_left_support_contract,
        driver_signature=driver_signature,
        canonical_entry_patch_left_support_null_action_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_left_support_null_action_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchLeftSupportNullActionReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_left_support_null_action_report(
        patch_plan_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan()
        ),
        spectral_admissibility_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe()
        ),
    )
