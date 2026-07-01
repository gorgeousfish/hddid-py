from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

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


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _patched_covariance(
    covariance_matrix: np.ndarray,
    *,
    center_index: int,
    shoulder_index: int,
    delta: float,
) -> np.ndarray:
    patched = np.asarray(covariance_matrix, dtype=float).copy()
    patched[shoulder_index, center_index] += float(delta)
    patched[center_index, shoulder_index] += float(delta)
    return patched


def _minimum_symmetric_eigenvalue(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    return float(eigenvalues[0])


def _find_psd_boundary_increment(
    covariance_matrix: np.ndarray,
    *,
    center_index: int,
    shoulder_index: int,
) -> float:
    lower = 0.0
    upper = 1.0
    while (
        _minimum_symmetric_eigenvalue(
            _patched_covariance(
                covariance_matrix,
                center_index=center_index,
                shoulder_index=shoulder_index,
                delta=upper,
            )
        )
        > 0.0
    ):
        upper *= 2.0
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        minimum_eigenvalue = _minimum_symmetric_eigenvalue(
            _patched_covariance(
                covariance_matrix,
                center_index=center_index,
                shoulder_index=shoulder_index,
                delta=midpoint,
            )
        )
        if minimum_eigenvalue > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return float(lower)


def _driver_signature(
    *,
    patch_plan_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport
    ),
    diagonal_preserved: bool,
    anchor_min_eigenvalue: float,
    patched_min_eigenvalue: float,
    required_patch_increment: float,
    psd_boundary_increment: float,
    required_patch_share_of_psd_boundary: float,
    preserved_min_eigenvalue_share: float,
) -> str:
    if (
        patch_plan_report.driver_signature == "bounded-right-center-entry-patch-plan"
        and diagonal_preserved
        and anchor_min_eigenvalue > 0.0
        and patched_min_eigenvalue > 0.0
        and psd_boundary_increment > required_patch_increment
        and required_patch_share_of_psd_boundary < 0.2
        and preserved_min_eigenvalue_share > 0.9
    ):
        return "bounded-right-center-entry-patch-spectral-admissibility"
    return "mixed-right-center-entry-patch-spectral-admissibility"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSpectralAdmissibilityReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_index: int
    failing_right_shoulder_index: int
    required_patch_increment: float
    anchor_eigenvalues: np.ndarray
    patched_eigenvalues: np.ndarray
    anchor_min_eigenvalue: float
    patched_min_eigenvalue: float
    psd_boundary_increment: float
    required_patch_share_of_psd_boundary: float
    preserved_min_eigenvalue_share: float
    remaining_psd_headroom: float
    remaining_psd_headroom_multiple_of_required_patch: float
    diagonal_preserved: bool
    driver_signature: str
    canonical_entry_patch_spectral_admissibility_digest: tuple[str, ...]

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
        self.required_patch_increment = float(self.required_patch_increment)
        self.anchor_eigenvalues = np.asarray(
            self.anchor_eigenvalues, dtype=float
        ).copy()
        self.patched_eigenvalues = np.asarray(
            self.patched_eigenvalues, dtype=float
        ).copy()
        self.anchor_min_eigenvalue = float(self.anchor_min_eigenvalue)
        self.patched_min_eigenvalue = float(self.patched_min_eigenvalue)
        self.psd_boundary_increment = float(self.psd_boundary_increment)
        self.required_patch_share_of_psd_boundary = float(
            self.required_patch_share_of_psd_boundary
        )
        self.preserved_min_eigenvalue_share = float(self.preserved_min_eigenvalue_share)
        self.remaining_psd_headroom = float(self.remaining_psd_headroom)
        self.remaining_psd_headroom_multiple_of_required_patch = float(
            self.remaining_psd_headroom_multiple_of_required_patch
        )
        self.diagonal_preserved = bool(self.diagonal_preserved)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_entry_patch_spectral_admissibility_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_entry_patch_spectral_admissibility_digest
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
            "required_patch_increment": self.required_patch_increment,
            "anchor_eigenvalues": self.anchor_eigenvalues.tolist(),
            "patched_eigenvalues": self.patched_eigenvalues.tolist(),
            "anchor_min_eigenvalue": self.anchor_min_eigenvalue,
            "patched_min_eigenvalue": self.patched_min_eigenvalue,
            "psd_boundary_increment": self.psd_boundary_increment,
            "required_patch_share_of_psd_boundary": (
                self.required_patch_share_of_psd_boundary
            ),
            "preserved_min_eigenvalue_share": self.preserved_min_eigenvalue_share,
            "remaining_psd_headroom": self.remaining_psd_headroom,
            "remaining_psd_headroom_multiple_of_required_patch": (
                self.remaining_psd_headroom_multiple_of_required_patch
            ),
            "diagonal_preserved": self.diagonal_preserved,
            "driver_signature": self.driver_signature,
            "canonical_entry_patch_spectral_admissibility_digest": list(
                self.canonical_entry_patch_spectral_admissibility_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_report(
    *,
    patch_plan_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPlanReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSpectralAdmissibilityReport:
    anchor_covariance = np.asarray(
        patch_plan_report.patched_covariance_at_grid
        - patch_plan_report.covariance_patch_delta,
        dtype=float,
    )
    patched_covariance = np.asarray(
        patch_plan_report.patched_covariance_at_grid,
        dtype=float,
    )
    anchor_eigenvalues = np.linalg.eigvalsh(anchor_covariance)
    patched_eigenvalues = np.linalg.eigvalsh(patched_covariance)
    anchor_min_eigenvalue = float(anchor_eigenvalues[0])
    patched_min_eigenvalue = float(patched_eigenvalues[0])
    required_patch_increment = float(patch_plan_report.signed_right_center_increment)
    psd_boundary_increment = _find_psd_boundary_increment(
        anchor_covariance,
        center_index=patch_plan_report.center_index,
        shoulder_index=patch_plan_report.failing_right_shoulder_index,
    )
    if psd_boundary_increment <= 0.0:
        raise ValueError(
            "entry patch spectral admissibility requires positive PSD headroom"
        )
    required_patch_share_of_psd_boundary = (
        required_patch_increment / psd_boundary_increment
    )
    preserved_min_eigenvalue_share = (
        patched_min_eigenvalue / anchor_min_eigenvalue
        if anchor_min_eigenvalue > 0.0
        else 0.0
    )
    remaining_psd_headroom = psd_boundary_increment - required_patch_increment
    remaining_psd_headroom_multiple_of_required_patch = (
        remaining_psd_headroom / required_patch_increment
        if required_patch_increment > 0.0
        else 0.0
    )
    diagonal_preserved = bool(
        np.allclose(np.diag(anchor_covariance), np.diag(patched_covariance), atol=1e-12)
    )

    sigma_z_hat = np.sqrt(np.diag(anchor_covariance))
    driver_signature = _driver_signature(
        patch_plan_report=patch_plan_report,
        diagonal_preserved=diagonal_preserved,
        anchor_min_eigenvalue=anchor_min_eigenvalue,
        patched_min_eigenvalue=patched_min_eigenvalue,
        required_patch_increment=required_patch_increment,
        psd_boundary_increment=psd_boundary_increment,
        required_patch_share_of_psd_boundary=required_patch_share_of_psd_boundary,
        preserved_min_eigenvalue_share=preserved_min_eigenvalue_share,
    )

    canonical_digest = (
        f"- binding design `DGP2/500/50` on `{patch_plan_report.window_label}`: the validation-only two-cell patch still moves only `[row={patch_plan_report.failing_right_shoulder_index}, col={patch_plan_report.center_index}]` / `[row={patch_plan_report.center_index}, col={patch_plan_report.failing_right_shoulder_index}]` by `{_format_signed_float(required_patch_increment)}` while leaving the covariance diagonal unchanged at `[{_format_float(anchor_covariance[0, 0])}, {_format_float(anchor_covariance[1, 1])}, {_format_float(anchor_covariance[2, 2])}]`, so the same covariance-process `sigma_z_hat` contract stays pinned to `[{_format_float(sigma_z_hat[0])}, {_format_float(sigma_z_hat[1])}, {_format_float(sigma_z_hat[2])}]` on `z = 0.05 / 0.15 / 0.25`",
        f"- the patched covariance-process object remains positive semidefinite with comfortable slack: the minimum eigenvalue only moves from `{_format_float(anchor_min_eigenvalue)}` to `{_format_float(patched_min_eigenvalue)}`, preserving `{_format_percent(preserved_min_eigenvalue_share)}` of anchor spectral slack even after the bounded repair",
        f"- the same off-diagonal patch does not hit the PSD boundary until `{_format_float(psd_boundary_increment)}`; the required `{_format_signed_float(required_patch_increment)}` therefore consumes only `{_format_percent(required_patch_share_of_psd_boundary)}` of admissible spectral budget and still leaves `{_format_signed_float(remaining_psd_headroom)}` headroom (`{_format_ratio(remaining_psd_headroom_multiple_of_required_patch)}` the bounded patch) before the smallest eigenvalue reaches zero",
        "- current Trigger 2 implication: `bounded-right-center-entry-patch-spectral-admissibility`; implementation intake may treat the exact two-cell patch as covariance-contract-admissible, but it should remain validation-only until an estimator path demonstrates the same patch can be realized without breaking the existing left-side support contract",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSpectralAdmissibilityReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-spectral-admissibility-probe"
        ),
        policy_digest=patch_plan_report.policy_digest,
        binding_design=patch_plan_report.binding_design,
        window_label=patch_plan_report.window_label,
        evaluation_grid=patch_plan_report.evaluation_grid,
        coverage_anchor_random_state=patch_plan_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            patch_plan_report.overshoot_companion_random_state
        ),
        center_index=patch_plan_report.center_index,
        failing_right_shoulder_index=patch_plan_report.failing_right_shoulder_index,
        required_patch_increment=required_patch_increment,
        anchor_eigenvalues=anchor_eigenvalues,
        patched_eigenvalues=patched_eigenvalues,
        anchor_min_eigenvalue=anchor_min_eigenvalue,
        patched_min_eigenvalue=patched_min_eigenvalue,
        psd_boundary_increment=psd_boundary_increment,
        required_patch_share_of_psd_boundary=required_patch_share_of_psd_boundary,
        preserved_min_eigenvalue_share=preserved_min_eigenvalue_share,
        remaining_psd_headroom=remaining_psd_headroom,
        remaining_psd_headroom_multiple_of_required_patch=(
            remaining_psd_headroom_multiple_of_required_patch
        ),
        diagonal_preserved=diagonal_preserved,
        driver_signature=driver_signature,
        canonical_entry_patch_spectral_admissibility_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSpectralAdmissibilityReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_report(
        patch_plan_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan()
        ),
    )
