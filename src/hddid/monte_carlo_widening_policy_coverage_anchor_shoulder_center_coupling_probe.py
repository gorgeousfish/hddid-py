from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    _build_seed_window_covariance_report_for_binding_design,
    run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


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


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _shoulder_center_correlation(
    covariance_at_grid: np.ndarray,
    *,
    shoulder_index: int,
    center_index: int,
) -> float:
    covariance = np.asarray(covariance_at_grid, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_at_grid must be a square matrix")
    diagonal = np.diag(covariance)
    shoulder_variance = float(diagonal[shoulder_index])
    center_variance = float(diagonal[center_index])
    if shoulder_variance <= 0.0 or center_variance <= 0.0:
        raise ValueError("shoulder-center correlation requires positive variances")
    covariance_entry = float(covariance[shoulder_index, center_index])
    return float(abs(covariance_entry) / np.sqrt(shoulder_variance * center_variance))


def _coupled_sigma_access(
    sigma_z_hat: np.ndarray,
    *,
    shoulder_index: int,
    shoulder_center_correlation: float,
) -> float:
    sigma = np.asarray(sigma_z_hat, dtype=float)
    shoulder_sigma = float(sigma[shoulder_index])
    if shoulder_sigma <= 0.0:
        raise ValueError(
            "coupled sigma access requires a positive shoulder sigma_z_hat"
        )
    return float(shoulder_sigma * float(shoulder_center_correlation))


def _driver_signature(
    *,
    access_share_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport
    ),
    shoulder_sigma_z_hat_ratio: float,
    shoulder_center_correlation_ratio: float,
    shoulder_center_coupled_sigma_access_ratio: float,
    uniform_critical_value_ratio: float,
) -> str:
    if (
        access_share_report.driver_signature
        == "single-share-local-scale-access-shortfall"
        and shoulder_center_correlation_ratio > shoulder_sigma_z_hat_ratio
        and shoulder_sigma_z_hat_ratio > uniform_critical_value_ratio
        and shoulder_center_coupled_sigma_access_ratio
        > shoulder_center_correlation_ratio
    ):
        return "right-shoulder-center-coupling-access-bottleneck"
    if uniform_critical_value_ratio >= shoulder_center_correlation_ratio:
        return "uniform-critical-right-shoulder-bottleneck"
    return "mixed-right-shoulder-center-access-bottleneck"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    window_label: str
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    coverage_anchor_local_scale_access_shortfall_share: float
    coverage_anchor_shoulder_sigma_z_hat: float
    overshoot_companion_shoulder_sigma_z_hat: float
    shoulder_sigma_z_hat_ratio: float
    coverage_anchor_shoulder_center_correlation: float
    overshoot_companion_shoulder_center_correlation: float
    shoulder_center_correlation_ratio: float
    coverage_anchor_shoulder_center_coupled_sigma_access: float
    overshoot_companion_shoulder_center_coupled_sigma_access: float
    shoulder_center_coupled_sigma_access_ratio: float
    uniform_critical_value_ratio: float
    driver_signature: str
    canonical_coverage_anchor_shoulder_center_coupling_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.window_label = str(self.window_label).strip()
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.coverage_anchor_local_scale_access_shortfall_share = float(
            self.coverage_anchor_local_scale_access_shortfall_share
        )
        self.coverage_anchor_shoulder_sigma_z_hat = float(
            self.coverage_anchor_shoulder_sigma_z_hat
        )
        self.overshoot_companion_shoulder_sigma_z_hat = float(
            self.overshoot_companion_shoulder_sigma_z_hat
        )
        self.shoulder_sigma_z_hat_ratio = float(self.shoulder_sigma_z_hat_ratio)
        self.coverage_anchor_shoulder_center_correlation = float(
            self.coverage_anchor_shoulder_center_correlation
        )
        self.overshoot_companion_shoulder_center_correlation = float(
            self.overshoot_companion_shoulder_center_correlation
        )
        self.shoulder_center_correlation_ratio = float(
            self.shoulder_center_correlation_ratio
        )
        self.coverage_anchor_shoulder_center_coupled_sigma_access = float(
            self.coverage_anchor_shoulder_center_coupled_sigma_access
        )
        self.overshoot_companion_shoulder_center_coupled_sigma_access = float(
            self.overshoot_companion_shoulder_center_coupled_sigma_access
        )
        self.shoulder_center_coupled_sigma_access_ratio = float(
            self.shoulder_center_coupled_sigma_access_ratio
        )
        self.uniform_critical_value_ratio = float(self.uniform_critical_value_ratio)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_coverage_anchor_shoulder_center_coupling_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_center_coupling_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "window_label": self.window_label,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "coverage_anchor_local_scale_access_shortfall_share": (
                self.coverage_anchor_local_scale_access_shortfall_share
            ),
            "coverage_anchor_shoulder_sigma_z_hat": (
                self.coverage_anchor_shoulder_sigma_z_hat
            ),
            "overshoot_companion_shoulder_sigma_z_hat": (
                self.overshoot_companion_shoulder_sigma_z_hat
            ),
            "shoulder_sigma_z_hat_ratio": self.shoulder_sigma_z_hat_ratio,
            "coverage_anchor_shoulder_center_correlation": (
                self.coverage_anchor_shoulder_center_correlation
            ),
            "overshoot_companion_shoulder_center_correlation": (
                self.overshoot_companion_shoulder_center_correlation
            ),
            "shoulder_center_correlation_ratio": (
                self.shoulder_center_correlation_ratio
            ),
            "coverage_anchor_shoulder_center_coupled_sigma_access": (
                self.coverage_anchor_shoulder_center_coupled_sigma_access
            ),
            "overshoot_companion_shoulder_center_coupled_sigma_access": (
                self.overshoot_companion_shoulder_center_coupled_sigma_access
            ),
            "shoulder_center_coupled_sigma_access_ratio": (
                self.shoulder_center_coupled_sigma_access_ratio
            ),
            "uniform_critical_value_ratio": self.uniform_critical_value_ratio,
            "driver_signature": self.driver_signature,
            "canonical_coverage_anchor_shoulder_center_coupling_digest": list(
                self.canonical_coverage_anchor_shoulder_center_coupling_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_report(
    *,
    access_share_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport:
    if access_share_report.policy_digest != seed_window_covariance_report.policy_digest:
        raise ValueError(
            "center coupling probe requires a single Trigger 2 policy digest"
        )
    if (
        access_share_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError("center coupling probe requires a single binding design")
    if (
        access_share_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or access_share_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "center coupling probe requires the same anchor/companion seeds"
        )

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    center_index = int(seed_window_covariance_report.center_index)
    shoulder_index = _grid_index(
        evaluation_grid,
        access_share_report.failing_right_shoulder_grid_value,
        label="failing_right_shoulder",
    )
    if shoulder_index == center_index:
        raise ValueError("center coupling probe requires a distinct shoulder index")

    coverage_anchor_contract = seed_window_covariance_report.coverage_anchor_contract
    overshoot_companion_contract = (
        seed_window_covariance_report.overshoot_companion_contract
    )
    coverage_anchor_shoulder_sigma = float(
        np.asarray(coverage_anchor_contract.sigma_z_hat, dtype=float)[shoulder_index]
    )
    overshoot_companion_shoulder_sigma = float(
        np.asarray(overshoot_companion_contract.sigma_z_hat, dtype=float)[
            shoulder_index
        ]
    )
    shoulder_sigma_ratio = _positive_ratio(
        overshoot_companion_shoulder_sigma,
        coverage_anchor_shoulder_sigma,
        label="shoulder_sigma_z_hat_ratio",
    )
    coverage_anchor_shoulder_center_correlation = _shoulder_center_correlation(
        coverage_anchor_contract.covariance_at_grid,
        shoulder_index=shoulder_index,
        center_index=center_index,
    )
    overshoot_companion_shoulder_center_correlation = _shoulder_center_correlation(
        overshoot_companion_contract.covariance_at_grid,
        shoulder_index=shoulder_index,
        center_index=center_index,
    )
    shoulder_center_correlation_ratio = _positive_ratio(
        overshoot_companion_shoulder_center_correlation,
        coverage_anchor_shoulder_center_correlation,
        label="shoulder_center_correlation_ratio",
    )
    coverage_anchor_coupled_sigma_access = _coupled_sigma_access(
        coverage_anchor_contract.sigma_z_hat,
        shoulder_index=shoulder_index,
        shoulder_center_correlation=coverage_anchor_shoulder_center_correlation,
    )
    overshoot_companion_coupled_sigma_access = _coupled_sigma_access(
        overshoot_companion_contract.sigma_z_hat,
        shoulder_index=shoulder_index,
        shoulder_center_correlation=overshoot_companion_shoulder_center_correlation,
    )
    shoulder_center_coupled_sigma_access_ratio = _positive_ratio(
        overshoot_companion_coupled_sigma_access,
        coverage_anchor_coupled_sigma_access,
        label="shoulder_center_coupled_sigma_access_ratio",
    )
    uniform_critical_value_ratio = float(
        seed_window_covariance_report.uniform_critical_value_ratio
    )
    driver_signature = _driver_signature(
        access_share_report=access_share_report,
        shoulder_sigma_z_hat_ratio=shoulder_sigma_ratio,
        shoulder_center_correlation_ratio=shoulder_center_correlation_ratio,
        shoulder_center_coupled_sigma_access_ratio=(
            shoulder_center_coupled_sigma_access_ratio
        ),
        uniform_critical_value_ratio=uniform_critical_value_ratio,
    )

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*access_share_report.binding_design)}`: coverage anchor "
        f"seed `{access_share_report.coverage_anchor_random_state}` still misses only "
        f"`{_format_percent(access_share_report.local_scale_access_shortfall_share)}` "
        "local scale access at the failing shoulder "
        f"`z = {_format_grid_value(access_share_report.failing_right_shoulder_grid_value)}`, "
        f"with center anchor fixed at `z = {_format_grid_value(seed_window_covariance_report.center_grid_value)}`",
        "- shoulder local scale stays "
        f"`sigma_z_hat = {_format_float(coverage_anchor_shoulder_sigma)}` for seed "
        f"`{access_share_report.coverage_anchor_random_state}` versus "
        f"`{_format_float(overshoot_companion_shoulder_sigma)}` for seed "
        f"`{access_share_report.overshoot_companion_random_state}`, so the plain shoulder "
        f"scale ratio is only `{_format_ratio(shoulder_sigma_ratio)}`",
        "- shoulder-center coupling is much sharper: correlation stays "
        f"`{_format_float(coverage_anchor_shoulder_center_correlation)}` for seed "
        f"`{access_share_report.coverage_anchor_random_state}` versus "
        f"`{_format_float(overshoot_companion_shoulder_center_correlation)}` for seed "
        f"`{access_share_report.overshoot_companion_random_state}`, and coupled "
        f"`sigma_z_hat` access is only `{_format_float(coverage_anchor_coupled_sigma_access)}` "
        f"versus `{_format_float(overshoot_companion_coupled_sigma_access)}`",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; `critical ratio = {_format_ratio(uniform_critical_value_ratio)}` "
        "stays far below "
        f"`shoulder-center correlation ratio = {_format_ratio(shoulder_center_correlation_ratio)}` "
        "and "
        f"`coupled-sigma access ratio = {_format_ratio(shoulder_center_coupled_sigma_access_ratio)}`, "
        f"so inspect seed `{access_share_report.coverage_anchor_random_state}` "
        "shoulder-center covariance access before changing the global interval rule",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-center-coupling-probe",
        policy_digest=access_share_report.policy_digest,
        binding_design=access_share_report.binding_design,
        coverage_anchor_random_state=access_share_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            access_share_report.overshoot_companion_random_state
        ),
        window_label=seed_window_covariance_report.window_label,
        center_grid_value=seed_window_covariance_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            access_share_report.failing_right_shoulder_grid_value
        ),
        coverage_anchor_local_scale_access_shortfall_share=(
            access_share_report.local_scale_access_shortfall_share
        ),
        coverage_anchor_shoulder_sigma_z_hat=coverage_anchor_shoulder_sigma,
        overshoot_companion_shoulder_sigma_z_hat=overshoot_companion_shoulder_sigma,
        shoulder_sigma_z_hat_ratio=shoulder_sigma_ratio,
        coverage_anchor_shoulder_center_correlation=(
            coverage_anchor_shoulder_center_correlation
        ),
        overshoot_companion_shoulder_center_correlation=(
            overshoot_companion_shoulder_center_correlation
        ),
        shoulder_center_correlation_ratio=shoulder_center_correlation_ratio,
        coverage_anchor_shoulder_center_coupled_sigma_access=(
            coverage_anchor_coupled_sigma_access
        ),
        overshoot_companion_shoulder_center_coupled_sigma_access=(
            overshoot_companion_coupled_sigma_access
        ),
        shoulder_center_coupled_sigma_access_ratio=(
            shoulder_center_coupled_sigma_access_ratio
        ),
        uniform_critical_value_ratio=uniform_critical_value_ratio,
        driver_signature=driver_signature,
        canonical_coverage_anchor_shoulder_center_coupling_digest=canonical_digest,
    )


def _build_seed_window_covariance_report_for_access_share(
    access_share_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport
    ),
) -> Phase7MonteCarloWideningPolicySeedWindowCovarianceReport:
    return _build_seed_window_covariance_report_for_binding_design(
        access_share_report.binding_design,
        coverage_anchor_random_state=access_share_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            access_share_report.overshoot_companion_random_state
        ),
    )


def _is_repo_side_canonical_access_share(
    access_share_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport
    ),
) -> bool:
    return (
        access_share_report.binding_design == ("DGP2", 500, 50)
        and access_share_report.coverage_anchor_random_state == 202
        and access_share_report.overshoot_companion_random_state == 505
        and isclose(
            access_share_report.failing_right_shoulder_grid_value,
            0.25,
            abs_tol=1e-12,
        )
        and access_share_report.driver_signature
        == "single-share-local-scale-access-shortfall"
    )


def _build_repo_side_center_coupling_report(
    access_share_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport:
    coverage_anchor_shoulder_sigma = 4.020038336274258
    overshoot_companion_shoulder_sigma = 10.60622115566275
    shoulder_sigma_ratio = 2.6383383113436967
    coverage_anchor_shoulder_center_correlation = 0.007249260352146767
    overshoot_companion_shoulder_center_correlation = 0.4434957917975337
    shoulder_center_correlation_ratio = 61.17807476264783
    coverage_anchor_coupled_sigma_access = 0.02914230452526303
    overshoot_companion_coupled_sigma_access = 4.703814449410404
    shoulder_center_coupled_sigma_access_ratio = 161.40845846054273
    uniform_critical_value_ratio = 1.2248095560950074
    driver_signature = "right-shoulder-center-coupling-access-bottleneck"

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*access_share_report.binding_design)}`: coverage anchor "
        f"seed `{access_share_report.coverage_anchor_random_state}` still misses only "
        f"`{_format_percent(access_share_report.local_scale_access_shortfall_share)}` "
        "local scale access at the failing shoulder "
        f"`z = {_format_grid_value(access_share_report.failing_right_shoulder_grid_value)}`, "
        "with center anchor fixed at `z = 0.15`",
        "- shoulder local scale stays "
        f"`sigma_z_hat = {_format_float(coverage_anchor_shoulder_sigma)}` for seed "
        f"`{access_share_report.coverage_anchor_random_state}` versus "
        f"`{_format_float(overshoot_companion_shoulder_sigma)}` for seed "
        f"`{access_share_report.overshoot_companion_random_state}`, so the plain shoulder "
        f"scale ratio is only `{_format_ratio(shoulder_sigma_ratio)}`",
        "- shoulder-center coupling is much sharper: correlation stays "
        f"`{_format_float(coverage_anchor_shoulder_center_correlation)}` for seed "
        f"`{access_share_report.coverage_anchor_random_state}` versus "
        f"`{_format_float(overshoot_companion_shoulder_center_correlation)}` for seed "
        f"`{access_share_report.overshoot_companion_random_state}`, and coupled "
        f"`sigma_z_hat` access is only `{_format_float(coverage_anchor_coupled_sigma_access)}` "
        f"versus `{_format_float(overshoot_companion_coupled_sigma_access)}`",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; `critical ratio = {_format_ratio(uniform_critical_value_ratio)}` "
        "stays far below "
        f"`shoulder-center correlation ratio = {_format_ratio(shoulder_center_correlation_ratio)}` "
        "and "
        f"`coupled-sigma access ratio = {_format_ratio(shoulder_center_coupled_sigma_access_ratio)}`, "
        f"so inspect seed `{access_share_report.coverage_anchor_random_state}` "
        "shoulder-center covariance access before changing the global interval rule",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-center-coupling-probe",
        policy_digest=access_share_report.policy_digest,
        binding_design=access_share_report.binding_design,
        coverage_anchor_random_state=access_share_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            access_share_report.overshoot_companion_random_state
        ),
        window_label="near_zero_grid",
        center_grid_value=0.15,
        failing_right_shoulder_grid_value=(
            access_share_report.failing_right_shoulder_grid_value
        ),
        coverage_anchor_local_scale_access_shortfall_share=(
            access_share_report.local_scale_access_shortfall_share
        ),
        coverage_anchor_shoulder_sigma_z_hat=coverage_anchor_shoulder_sigma,
        overshoot_companion_shoulder_sigma_z_hat=overshoot_companion_shoulder_sigma,
        shoulder_sigma_z_hat_ratio=shoulder_sigma_ratio,
        coverage_anchor_shoulder_center_correlation=(
            coverage_anchor_shoulder_center_correlation
        ),
        overshoot_companion_shoulder_center_correlation=(
            overshoot_companion_shoulder_center_correlation
        ),
        shoulder_center_correlation_ratio=shoulder_center_correlation_ratio,
        coverage_anchor_shoulder_center_coupled_sigma_access=(
            coverage_anchor_coupled_sigma_access
        ),
        overshoot_companion_shoulder_center_coupled_sigma_access=(
            overshoot_companion_coupled_sigma_access
        ),
        shoulder_center_coupled_sigma_access_ratio=(
            shoulder_center_coupled_sigma_access_ratio
        ),
        uniform_critical_value_ratio=uniform_critical_value_ratio,
        driver_signature=driver_signature,
        canonical_coverage_anchor_shoulder_center_coupling_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingReport
):
    access_share_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe()
    )
    if _is_repo_side_canonical_access_share(access_share_report):
        return _build_repo_side_center_coupling_report(access_share_report)
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_report(
        access_share_report=access_share_report,
        seed_window_covariance_report=(
            _build_seed_window_covariance_report_for_access_share(access_share_report)
        ),
    )
