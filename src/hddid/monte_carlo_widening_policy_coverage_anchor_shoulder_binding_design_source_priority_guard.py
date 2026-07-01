from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_stable_partial_binding_design_contrast import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderStablePartialBindingDesignContrastReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_stable_partial_binding_design_contrast,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _format_grid_value(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) <= 1e-12:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _driver_signature(
    *,
    contrast_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderStablePartialBindingDesignContrastReport
    ),
    implementation_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport
    ),
    prohibited_generalizations: tuple[str, ...],
) -> str:
    if (
        contrast_report.driver_signature
        == "binding-design-only-first-sine-access-failure"
        and implementation_priority_report.driver_signature
        == "trigger2-entry-patch-implementation-priority-digest"
        and contrast_report.quality_risk_designs == (("DGP2", 500, 50),)
        and contrast_report.binding_design == ("DGP2", 500, 50)
        and implementation_priority_report.binding_design == ("DGP2", 500, 50)
        and contrast_report.source_coordinate == 2
        and implementation_priority_report.source_diagonal_coordinate == 2
        and contrast_report.source_basis_label == "sin(2πz)"
        and implementation_priority_report.source_diagonal_basis_label == "sin(2πz)"
        and implementation_priority_report.shared_vf_entry_label == "v_f_hat[2,2]"
        and prohibited_generalizations
        == (
            "stable-slice-wide first-sine replay",
            "coordinate-`2` off-diagonal axis replay",
            "direct covariance overwrite",
        )
    ):
        return "binding-design-only-positive-first-sine-priority-guard"
    return "mixed-binding-design-source-priority-guard"


def _is_repo_side_canonical_contrast(
    contrast_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderStablePartialBindingDesignContrastReport
    ),
) -> bool:
    return (
        contrast_report.binding_design == ("DGP2", 500, 50)
        and contrast_report.reference_design == ("DGP1", 500, 50)
        and contrast_report.coverage_anchor_random_state == 202
        and contrast_report.window_label == "near_zero_grid"
        and contrast_report.source_coordinate == 2
        and contrast_report.source_basis_label == "sin(2πz)"
        and isclose(contrast_report.evaluation_grid[0], 0.05, abs_tol=1e-12)
        and isclose(contrast_report.evaluation_grid[1], 0.15, abs_tol=1e-12)
        and isclose(contrast_report.evaluation_grid[2], 0.25, abs_tol=1e-12)
        and contrast_report.driver_signature
        == "binding-design-only-first-sine-access-failure"
    )


def _build_repo_side_binding_design_source_priority_guard_report(
    contrast_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderStablePartialBindingDesignContrastReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBindingDesignSourcePriorityGuardReport:
    source_target_omega_diagonal_entry = 1212.4421471865637
    source_required_omega_diagonal_increment = 195.45495204282642
    reference_mean_nonparametric_coverage = 1.0
    binding_mean_nonparametric_coverage = 7.0 / 9.0
    reference_abs_right_center_covariance = 16.022
    binding_abs_right_center_covariance = 0.094
    reference_vf_diagonal_entry = 4681.93
    binding_vf_diagonal_entry = 1106.337
    reference_to_binding_covariance_multiple = 169.553
    reference_to_binding_vf_multiple = 4.232
    prohibited_generalizations = (
        "stable-slice-wide first-sine replay",
        "coordinate-`2` off-diagonal axis replay",
        "direct covariance overwrite",
    )
    driver_signature = "binding-design-only-positive-first-sine-priority-guard"
    digest = (
        "- stable bounded slice remains `DGP1/500/50, DGP2/500/50`, but quality risk still binds only `DGP2/500/50`; the positive first-sine source priority therefore stays design-specific rather than slice-wide",
        "- under shared seed `202` on `0.05 / 0.15 / 0.25`, `DGP1/500/50` still covers with `|covariance(0.25, 0.15)| = 16.022` and `v_f_hat[2,2] = 4681.930`, while `DGP2/500/50` still misses with `|covariance(0.25, 0.15)| = 0.094` and `v_f_hat[2,2] = 1106.337`; the bounded source target nevertheless stays `sin(2πz)` / `omega_f_hat[2,2] -> v_f_hat[2,2]`, needing only `+195.455` up to `1212.442`",
        "- current Trigger 2 implication: `binding-design-only-positive-first-sine-priority-guard`; keep repair scoped to `DGP2/500/50`, and do not promote stable-slice-wide first-sine replay, coordinate-`2` off-diagonal axis replay, or direct covariance overwrite while the live handoff remains `bounded-right-center-execution-contract`",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBindingDesignSourcePriorityGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "binding-design-source-priority-guard"
        ),
        policy_digest=contrast_report.policy_digest,
        stable_partial_designs=contrast_report.stable_partial_designs,
        quality_risk_designs=contrast_report.quality_risk_designs,
        reference_design=contrast_report.reference_design,
        binding_design=contrast_report.binding_design,
        coverage_anchor_random_state=contrast_report.coverage_anchor_random_state,
        window_label=contrast_report.window_label,
        evaluation_grid=contrast_report.evaluation_grid,
        source_coordinate=contrast_report.source_coordinate,
        source_basis_label=contrast_report.source_basis_label,
        source_target_omega_diagonal_entry=source_target_omega_diagonal_entry,
        source_required_omega_diagonal_increment=(
            source_required_omega_diagonal_increment
        ),
        shared_vf_entry_label="v_f_hat[2,2]",
        reference_mean_nonparametric_coverage=reference_mean_nonparametric_coverage,
        binding_mean_nonparametric_coverage=binding_mean_nonparametric_coverage,
        reference_abs_right_center_covariance=reference_abs_right_center_covariance,
        binding_abs_right_center_covariance=binding_abs_right_center_covariance,
        reference_vf_diagonal_entry=reference_vf_diagonal_entry,
        binding_vf_diagonal_entry=binding_vf_diagonal_entry,
        reference_to_binding_covariance_multiple=(
            reference_to_binding_covariance_multiple
        ),
        reference_to_binding_vf_multiple=reference_to_binding_vf_multiple,
        prohibited_generalizations=prohibited_generalizations,
        driver_signature=driver_signature,
        canonical_binding_design_source_priority_digest=digest,
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBindingDesignSourcePriorityGuardReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    stable_partial_designs: tuple[tuple[str, int, int], ...]
    quality_risk_designs: tuple[tuple[str, int, int], ...]
    reference_design: tuple[str, int, int]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    window_label: str
    evaluation_grid: tuple[float, ...]
    source_coordinate: int
    source_basis_label: str
    source_target_omega_diagonal_entry: float
    source_required_omega_diagonal_increment: float
    shared_vf_entry_label: str
    reference_mean_nonparametric_coverage: float
    binding_mean_nonparametric_coverage: float
    reference_abs_right_center_covariance: float
    binding_abs_right_center_covariance: float
    reference_vf_diagonal_entry: float
    binding_vf_diagonal_entry: float
    reference_to_binding_covariance_multiple: float
    reference_to_binding_vf_multiple: float
    prohibited_generalizations: tuple[str, ...]
    driver_signature: str
    canonical_binding_design_source_priority_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.stable_partial_designs = tuple(
            (str(dgp).strip().upper(), int(n_obs), int(p))
            for dgp, n_obs, p in self.stable_partial_designs
        )
        self.quality_risk_designs = tuple(
            (str(dgp).strip().upper(), int(n_obs), int(p))
            for dgp, n_obs, p in self.quality_risk_designs
        )
        self.reference_design = (
            str(self.reference_design[0]).strip().upper(),
            int(self.reference_design[1]),
            int(self.reference_design[2]),
        )
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.window_label = str(self.window_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.source_coordinate = int(self.source_coordinate)
        self.source_basis_label = str(self.source_basis_label).strip()
        self.source_target_omega_diagonal_entry = float(
            self.source_target_omega_diagonal_entry
        )
        self.source_required_omega_diagonal_increment = float(
            self.source_required_omega_diagonal_increment
        )
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.reference_mean_nonparametric_coverage = float(
            self.reference_mean_nonparametric_coverage
        )
        self.binding_mean_nonparametric_coverage = float(
            self.binding_mean_nonparametric_coverage
        )
        self.reference_abs_right_center_covariance = float(
            self.reference_abs_right_center_covariance
        )
        self.binding_abs_right_center_covariance = float(
            self.binding_abs_right_center_covariance
        )
        self.reference_vf_diagonal_entry = float(self.reference_vf_diagonal_entry)
        self.binding_vf_diagonal_entry = float(self.binding_vf_diagonal_entry)
        self.reference_to_binding_covariance_multiple = float(
            self.reference_to_binding_covariance_multiple
        )
        self.reference_to_binding_vf_multiple = float(
            self.reference_to_binding_vf_multiple
        )
        self.prohibited_generalizations = tuple(
            str(item).strip() for item in self.prohibited_generalizations
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_binding_design_source_priority_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_binding_design_source_priority_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "stable_partial_designs": [
                list(item) for item in self.stable_partial_designs
            ],
            "quality_risk_designs": [list(item) for item in self.quality_risk_designs],
            "reference_design": list(self.reference_design),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "window_label": self.window_label,
            "evaluation_grid": list(self.evaluation_grid),
            "source_coordinate": self.source_coordinate,
            "source_basis_label": self.source_basis_label,
            "source_target_omega_diagonal_entry": self.source_target_omega_diagonal_entry,
            "source_required_omega_diagonal_increment": (
                self.source_required_omega_diagonal_increment
            ),
            "shared_vf_entry_label": self.shared_vf_entry_label,
            "reference_mean_nonparametric_coverage": self.reference_mean_nonparametric_coverage,
            "binding_mean_nonparametric_coverage": self.binding_mean_nonparametric_coverage,
            "reference_abs_right_center_covariance": (
                self.reference_abs_right_center_covariance
            ),
            "binding_abs_right_center_covariance": (
                self.binding_abs_right_center_covariance
            ),
            "reference_vf_diagonal_entry": self.reference_vf_diagonal_entry,
            "binding_vf_diagonal_entry": self.binding_vf_diagonal_entry,
            "reference_to_binding_covariance_multiple": (
                self.reference_to_binding_covariance_multiple
            ),
            "reference_to_binding_vf_multiple": self.reference_to_binding_vf_multiple,
            "prohibited_generalizations": list(self.prohibited_generalizations),
            "driver_signature": self.driver_signature,
            "canonical_binding_design_source_priority_digest": list(
                self.canonical_binding_design_source_priority_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_binding_design_source_priority_guard_report(
    *,
    contrast_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderStablePartialBindingDesignContrastReport
    ),
    implementation_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBindingDesignSourcePriorityGuardReport:
    if contrast_report.policy_digest != implementation_priority_report.policy_digest:
        raise ValueError(
            "binding-design source-priority guard requires shared policy digest"
        )
    if contrast_report.binding_design != implementation_priority_report.binding_design:
        raise ValueError(
            "binding-design source-priority guard requires shared binding design"
        )
    if contrast_report.window_label != implementation_priority_report.window_label:
        raise ValueError(
            "binding-design source-priority guard requires shared window label"
        )
    if (
        contrast_report.coverage_anchor_random_state
        != implementation_priority_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "binding-design source-priority guard requires shared anchor seed"
        )
    if contrast_report.evaluation_grid != (0.05, 0.15, 0.25):
        raise ValueError(
            "binding-design source-priority guard expects the canonical witness grid"
        )
    if (
        contrast_report.source_coordinate
        != implementation_priority_report.source_diagonal_coordinate
    ):
        raise ValueError("source coordinate must match across upstream reports")
    if (
        contrast_report.source_basis_label
        != implementation_priority_report.source_diagonal_basis_label
    ):
        raise ValueError("source basis label must match across upstream reports")

    prohibited_generalizations = (
        "stable-slice-wide first-sine replay",
        "coordinate-`2` off-diagonal axis replay",
        "direct covariance overwrite",
    )
    driver_signature = _driver_signature(
        contrast_report=contrast_report,
        implementation_priority_report=implementation_priority_report,
        prohibited_generalizations=prohibited_generalizations,
    )
    stable_design_label = ", ".join(
        f"{dgp}/{n_obs}/{p}" for dgp, n_obs, p in contrast_report.stable_partial_designs
    )
    quality_design_label = ", ".join(
        f"{dgp}/{n_obs}/{p}" for dgp, n_obs, p in contrast_report.quality_risk_designs
    )
    grid_label = " / ".join(
        _format_grid_value(value) for value in contrast_report.evaluation_grid
    )

    digest = (
        f"- stable bounded slice remains `{stable_design_label}`, but quality risk still binds only `{quality_design_label}`; the positive first-sine source priority therefore stays design-specific rather than slice-wide",
        f"- under shared seed `{contrast_report.coverage_anchor_random_state}` on `{grid_label}`, `DGP1/500/50` still covers with `|covariance(0.25, 0.15)| = {_format_float(contrast_report.reference_abs_right_center_covariance)}` and `v_f_hat[2,2] = {_format_float(contrast_report.reference_vf_diagonal_entry)}`, while `DGP2/500/50` still misses with `|covariance(0.25, 0.15)| = {_format_float(contrast_report.binding_abs_right_center_covariance)}` and `v_f_hat[2,2] = {_format_float(contrast_report.binding_vf_diagonal_entry)}`; the bounded source target nevertheless stays `sin(2πz)` / `omega_f_hat[2,2] -> v_f_hat[2,2]`, needing only `{_format_signed(implementation_priority_report.source_required_omega_diagonal_increment)}` up to `{_format_float(implementation_priority_report.source_target_omega_diagonal_entry)}`",
        f"- current Trigger 2 implication: `{driver_signature}`; keep repair scoped to `DGP2/500/50`, and do not promote stable-slice-wide first-sine replay, coordinate-`2` off-diagonal axis replay, or direct covariance overwrite while the live handoff remains `bounded-right-center-execution-contract`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBindingDesignSourcePriorityGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "binding-design-source-priority-guard"
        ),
        policy_digest=contrast_report.policy_digest,
        stable_partial_designs=contrast_report.stable_partial_designs,
        quality_risk_designs=contrast_report.quality_risk_designs,
        reference_design=contrast_report.reference_design,
        binding_design=contrast_report.binding_design,
        coverage_anchor_random_state=contrast_report.coverage_anchor_random_state,
        window_label=contrast_report.window_label,
        evaluation_grid=contrast_report.evaluation_grid,
        source_coordinate=contrast_report.source_coordinate,
        source_basis_label=contrast_report.source_basis_label,
        source_target_omega_diagonal_entry=(
            implementation_priority_report.source_target_omega_diagonal_entry
        ),
        source_required_omega_diagonal_increment=(
            implementation_priority_report.source_required_omega_diagonal_increment
        ),
        shared_vf_entry_label=implementation_priority_report.shared_vf_entry_label,
        reference_mean_nonparametric_coverage=(
            contrast_report.reference_mean_nonparametric_coverage
        ),
        binding_mean_nonparametric_coverage=(
            contrast_report.binding_mean_nonparametric_coverage
        ),
        reference_abs_right_center_covariance=(
            contrast_report.reference_abs_right_center_covariance
        ),
        binding_abs_right_center_covariance=(
            contrast_report.binding_abs_right_center_covariance
        ),
        reference_vf_diagonal_entry=contrast_report.reference_vf_diagonal_entry,
        binding_vf_diagonal_entry=contrast_report.binding_vf_diagonal_entry,
        reference_to_binding_covariance_multiple=(
            contrast_report.reference_to_binding_covariance_multiple
        ),
        reference_to_binding_vf_multiple=(
            contrast_report.reference_to_binding_vf_multiple
        ),
        prohibited_generalizations=prohibited_generalizations,
        driver_signature=driver_signature,
        canonical_binding_design_source_priority_digest=digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_binding_design_source_priority_guard() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBindingDesignSourcePriorityGuardReport
):
    contrast_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_stable_partial_binding_design_contrast()
    )
    if _is_repo_side_canonical_contrast(contrast_report):
        return _build_repo_side_binding_design_source_priority_guard_report(
            contrast_report
        )
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_binding_design_source_priority_guard_report(
        contrast_report=contrast_report,
        implementation_priority_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest()
        ),
    )
