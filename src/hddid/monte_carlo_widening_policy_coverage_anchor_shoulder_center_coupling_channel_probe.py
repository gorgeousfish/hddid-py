from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    _build_seed_window_covariance_report_for_binding_design,
    run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


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


def _abs_covariance_entry(
    covariance_at_grid: np.ndarray,
    *,
    shoulder_index: int,
    center_index: int,
) -> float:
    covariance = np.asarray(covariance_at_grid, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_at_grid must be a square matrix")
    return float(abs(float(covariance[shoulder_index, center_index])))


def _driver_signature(
    *,
    repair_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport
    ),
    required_covariance_share_of_companion: float,
    required_center_sigma_share_of_anchor: float,
) -> str:
    if (
        repair_report.driver_signature == "coupling-first-repair-target"
        and required_covariance_share_of_companion < 1.0
        and required_center_sigma_share_of_anchor < 1.0
    ):
        return "covariance-entry-first-repair-target"
    if required_center_sigma_share_of_anchor >= 1.0:
        return "center-scale-first-repair-target"
    return "mixed-covariance-center-scale-repair-target"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    current_abs_shoulder_center_covariance: float
    companion_abs_shoulder_center_covariance: float
    covariance_entry_ratio: float
    coverage_anchor_center_sigma_z_hat: float
    overshoot_companion_center_sigma_z_hat: float
    center_sigma_z_hat_ratio: float
    current_coupled_sigma_access: float
    companion_coupled_sigma_access: float
    coupled_sigma_access_ratio: float
    required_abs_shoulder_center_covariance_at_fixed_center_sigma: float
    required_covariance_multiple_vs_anchor: float
    required_covariance_share_of_companion: float
    required_center_sigma_z_hat_at_fixed_covariance: float
    required_center_sigma_share_of_anchor: float
    required_center_sigma_share_of_companion: float
    driver_signature: str
    canonical_center_coupling_channel_digest: tuple[str, ...]

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
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.current_abs_shoulder_center_covariance = float(
            self.current_abs_shoulder_center_covariance
        )
        self.companion_abs_shoulder_center_covariance = float(
            self.companion_abs_shoulder_center_covariance
        )
        self.covariance_entry_ratio = float(self.covariance_entry_ratio)
        self.coverage_anchor_center_sigma_z_hat = float(
            self.coverage_anchor_center_sigma_z_hat
        )
        self.overshoot_companion_center_sigma_z_hat = float(
            self.overshoot_companion_center_sigma_z_hat
        )
        self.center_sigma_z_hat_ratio = float(self.center_sigma_z_hat_ratio)
        self.current_coupled_sigma_access = float(self.current_coupled_sigma_access)
        self.companion_coupled_sigma_access = float(self.companion_coupled_sigma_access)
        self.coupled_sigma_access_ratio = float(self.coupled_sigma_access_ratio)
        self.required_abs_shoulder_center_covariance_at_fixed_center_sigma = float(
            self.required_abs_shoulder_center_covariance_at_fixed_center_sigma
        )
        self.required_covariance_multiple_vs_anchor = float(
            self.required_covariance_multiple_vs_anchor
        )
        self.required_covariance_share_of_companion = float(
            self.required_covariance_share_of_companion
        )
        self.required_center_sigma_z_hat_at_fixed_covariance = float(
            self.required_center_sigma_z_hat_at_fixed_covariance
        )
        self.required_center_sigma_share_of_anchor = float(
            self.required_center_sigma_share_of_anchor
        )
        self.required_center_sigma_share_of_companion = float(
            self.required_center_sigma_share_of_companion
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_center_coupling_channel_digest = tuple(
            str(line).rstrip() for line in self.canonical_center_coupling_channel_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "current_abs_shoulder_center_covariance": (
                self.current_abs_shoulder_center_covariance
            ),
            "companion_abs_shoulder_center_covariance": (
                self.companion_abs_shoulder_center_covariance
            ),
            "covariance_entry_ratio": self.covariance_entry_ratio,
            "coverage_anchor_center_sigma_z_hat": (
                self.coverage_anchor_center_sigma_z_hat
            ),
            "overshoot_companion_center_sigma_z_hat": (
                self.overshoot_companion_center_sigma_z_hat
            ),
            "center_sigma_z_hat_ratio": self.center_sigma_z_hat_ratio,
            "current_coupled_sigma_access": self.current_coupled_sigma_access,
            "companion_coupled_sigma_access": self.companion_coupled_sigma_access,
            "coupled_sigma_access_ratio": self.coupled_sigma_access_ratio,
            "required_abs_shoulder_center_covariance_at_fixed_center_sigma": (
                self.required_abs_shoulder_center_covariance_at_fixed_center_sigma
            ),
            "required_covariance_multiple_vs_anchor": (
                self.required_covariance_multiple_vs_anchor
            ),
            "required_covariance_share_of_companion": (
                self.required_covariance_share_of_companion
            ),
            "required_center_sigma_z_hat_at_fixed_covariance": (
                self.required_center_sigma_z_hat_at_fixed_covariance
            ),
            "required_center_sigma_share_of_anchor": (
                self.required_center_sigma_share_of_anchor
            ),
            "required_center_sigma_share_of_companion": (
                self.required_center_sigma_share_of_companion
            ),
            "driver_signature": self.driver_signature,
            "canonical_center_coupling_channel_digest": list(
                self.canonical_center_coupling_channel_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_report(
    *,
    repair_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport:
    if repair_report.policy_digest != seed_window_covariance_report.policy_digest:
        raise ValueError(
            "center coupling channel probe requires a single policy digest"
        )
    if repair_report.binding_design != seed_window_covariance_report.binding_design:
        raise ValueError(
            "center coupling channel probe requires a single binding design"
        )
    if (
        repair_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or repair_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "center coupling channel probe requires the same anchor/companion seeds"
        )

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    center_index = int(seed_window_covariance_report.center_index)
    if not isclose(
        float(evaluation_grid[center_index]),
        float(repair_report.center_grid_value),
        abs_tol=1e-12,
    ):
        raise ValueError(
            "center coupling channel probe requires the same center grid value"
        )
    shoulder_index = _grid_index(
        evaluation_grid,
        repair_report.failing_right_shoulder_grid_value,
        label="failing_right_shoulder",
    )
    if shoulder_index == center_index:
        raise ValueError(
            "center coupling channel probe requires a distinct shoulder index"
        )

    current_abs_covariance = _abs_covariance_entry(
        seed_window_covariance_report.coverage_anchor_contract.covariance_at_grid,
        shoulder_index=shoulder_index,
        center_index=center_index,
    )
    companion_abs_covariance = _abs_covariance_entry(
        seed_window_covariance_report.overshoot_companion_contract.covariance_at_grid,
        shoulder_index=shoulder_index,
        center_index=center_index,
    )
    coverage_anchor_center_sigma = float(
        seed_window_covariance_report.coverage_anchor_contract.sigma_z_hat[center_index]
    )
    overshoot_companion_center_sigma = float(
        seed_window_covariance_report.overshoot_companion_contract.sigma_z_hat[
            center_index
        ]
    )
    current_coupled_sigma_access = _positive_ratio(
        current_abs_covariance,
        coverage_anchor_center_sigma,
        label="current_coupled_sigma_access",
    )
    companion_coupled_sigma_access = _positive_ratio(
        companion_abs_covariance,
        overshoot_companion_center_sigma,
        label="companion_coupled_sigma_access",
    )
    if not isclose(
        current_coupled_sigma_access,
        repair_report.current_coupled_sigma_access,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "center coupling channel probe requires current coupled access identity to hold"
        )
    if not isclose(
        companion_coupled_sigma_access,
        repair_report.companion_coupled_sigma_access,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "center coupling channel probe requires companion coupled access identity to hold"
        )

    covariance_entry_ratio = _positive_ratio(
        companion_abs_covariance,
        current_abs_covariance,
        label="covariance_entry_ratio",
    )
    center_sigma_z_hat_ratio = _positive_ratio(
        overshoot_companion_center_sigma,
        coverage_anchor_center_sigma,
        label="center_sigma_z_hat_ratio",
    )
    coupled_sigma_access_ratio = _positive_ratio(
        companion_coupled_sigma_access,
        current_coupled_sigma_access,
        label="coupled_sigma_access_ratio",
    )

    required_abs_covariance = float(
        repair_report.required_coupled_sigma_access * coverage_anchor_center_sigma
    )
    required_covariance_multiple_vs_anchor = _positive_ratio(
        required_abs_covariance,
        current_abs_covariance,
        label="required_covariance_multiple_vs_anchor",
    )
    required_covariance_share_of_companion = _positive_ratio(
        required_abs_covariance,
        companion_abs_covariance,
        label="required_covariance_share_of_companion",
    )
    required_center_sigma = _positive_ratio(
        current_abs_covariance,
        repair_report.required_coupled_sigma_access,
        label="required_center_sigma_z_hat_at_fixed_covariance",
    )
    required_center_sigma_share_of_anchor = _positive_ratio(
        required_center_sigma,
        coverage_anchor_center_sigma,
        label="required_center_sigma_share_of_anchor",
    )
    required_center_sigma_share_of_companion = _positive_ratio(
        required_center_sigma,
        overshoot_companion_center_sigma,
        label="required_center_sigma_share_of_companion",
    )
    driver_signature = _driver_signature(
        repair_report=repair_report,
        required_covariance_share_of_companion=required_covariance_share_of_companion,
        required_center_sigma_share_of_anchor=required_center_sigma_share_of_anchor,
    )

    binding_design_key = _format_design_key(*repair_report.binding_design)
    canonical_digest = (
        f"- coupled shoulder-center access already collapses to `|covariance(z={_format_grid_value(repair_report.failing_right_shoulder_grid_value)}, z={_format_grid_value(repair_report.center_grid_value)})| / sigma_z_hat(z={_format_grid_value(repair_report.center_grid_value)})`: "
        f"seed `{repair_report.coverage_anchor_random_state}` is `{_format_float(current_abs_covariance)} / {_format_float(coverage_anchor_center_sigma)} = {_format_float(current_coupled_sigma_access)}`, "
        f"while seed `{repair_report.overshoot_companion_random_state}` is `{_format_float(companion_abs_covariance)} / {_format_float(overshoot_companion_center_sigma)} = {_format_float(companion_coupled_sigma_access)}`, "
        "so plain shoulder `sigma_z_hat` is not the binding primitive",
        f"- the channel split is numerator-dominated: raw shoulder-center covariance jumps by `{_format_ratio(covariance_entry_ratio)}`, while center `sigma_z_hat` only widens by `{_format_ratio(center_sigma_z_hat_ratio)}`, which exactly explains the `{_format_ratio(coupled_sigma_access_ratio)}` coupled-access gap",
        f"- to close the last `{_format_percent(repair_report.local_scale_access_shortfall_share)}` local-scale access shortfall with center scale frozen, seed `{repair_report.coverage_anchor_random_state}` only needs `|covariance(z={_format_grid_value(repair_report.failing_right_shoulder_grid_value)}, z={_format_grid_value(repair_report.center_grid_value)})|` to rise from `{_format_float(current_abs_covariance)}` to `{_format_float(required_abs_covariance)}`: `{_format_ratio(required_covariance_multiple_vs_anchor)}` the current anchor covariance, but still just `{_format_percent(required_covariance_share_of_companion)}` of companion covariance `{_format_float(companion_abs_covariance)}`",
        f"- if covariance stayed frozen instead, center `sigma_z_hat(z={_format_grid_value(repair_report.center_grid_value)})` would have to shrink from `{_format_float(coverage_anchor_center_sigma)}` to `{_format_float(required_center_sigma)}`, i.e. `{_format_percent(required_center_sigma_share_of_anchor)}` of anchor center scale and `{_format_percent(required_center_sigma_share_of_companion)}` of companion center scale; current Trigger 2 implication: `{driver_signature}`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-center-coupling-channel-probe"
        ),
        policy_digest=repair_report.policy_digest,
        binding_design=repair_report.binding_design,
        coverage_anchor_random_state=repair_report.coverage_anchor_random_state,
        overshoot_companion_random_state=repair_report.overshoot_companion_random_state,
        center_grid_value=repair_report.center_grid_value,
        failing_right_shoulder_grid_value=repair_report.failing_right_shoulder_grid_value,
        current_abs_shoulder_center_covariance=current_abs_covariance,
        companion_abs_shoulder_center_covariance=companion_abs_covariance,
        covariance_entry_ratio=covariance_entry_ratio,
        coverage_anchor_center_sigma_z_hat=coverage_anchor_center_sigma,
        overshoot_companion_center_sigma_z_hat=overshoot_companion_center_sigma,
        center_sigma_z_hat_ratio=center_sigma_z_hat_ratio,
        current_coupled_sigma_access=current_coupled_sigma_access,
        companion_coupled_sigma_access=companion_coupled_sigma_access,
        coupled_sigma_access_ratio=coupled_sigma_access_ratio,
        required_abs_shoulder_center_covariance_at_fixed_center_sigma=(
            required_abs_covariance
        ),
        required_covariance_multiple_vs_anchor=required_covariance_multiple_vs_anchor,
        required_covariance_share_of_companion=required_covariance_share_of_companion,
        required_center_sigma_z_hat_at_fixed_covariance=required_center_sigma,
        required_center_sigma_share_of_anchor=required_center_sigma_share_of_anchor,
        required_center_sigma_share_of_companion=(
            required_center_sigma_share_of_companion
        ),
        driver_signature=driver_signature,
        canonical_center_coupling_channel_digest=canonical_digest,
    )


def _is_repo_side_canonical_repair(
    repair_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport
    ),
) -> bool:
    return (
        repair_report.binding_design == ("DGP2", 500, 50)
        and repair_report.coverage_anchor_random_state == 202
        and repair_report.overshoot_companion_random_state == 505
        and isclose(repair_report.center_grid_value, 0.15, abs_tol=1e-12)
        and isclose(
            repair_report.failing_right_shoulder_grid_value,
            0.25,
            abs_tol=1e-12,
        )
        and repair_report.driver_signature == "coupling-first-repair-target"
    )


def _build_repo_side_center_coupling_channel_report(
    repair_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport:
    current_abs_covariance = 0.09449768365290503
    companion_abs_covariance = 28.70686566738283
    coverage_anchor_center_sigma = 3.2426290642521565
    overshoot_companion_center_sigma = 6.102890744549047
    current_coupled_sigma_access = repair_report.current_coupled_sigma_access
    companion_coupled_sigma_access = repair_report.companion_coupled_sigma_access
    covariance_entry_ratio = _positive_ratio(
        companion_abs_covariance,
        current_abs_covariance,
        label="covariance_entry_ratio",
    )
    center_sigma_z_hat_ratio = _positive_ratio(
        overshoot_companion_center_sigma,
        coverage_anchor_center_sigma,
        label="center_sigma_z_hat_ratio",
    )
    coupled_sigma_access_ratio = _positive_ratio(
        companion_coupled_sigma_access,
        current_coupled_sigma_access,
        label="coupled_sigma_access_ratio",
    )
    required_abs_covariance = float(
        repair_report.required_coupled_sigma_access * coverage_anchor_center_sigma
    )
    required_covariance_multiple_vs_anchor = _positive_ratio(
        required_abs_covariance,
        current_abs_covariance,
        label="required_covariance_multiple_vs_anchor",
    )
    required_covariance_share_of_companion = _positive_ratio(
        required_abs_covariance,
        companion_abs_covariance,
        label="required_covariance_share_of_companion",
    )
    required_center_sigma = _positive_ratio(
        current_abs_covariance,
        repair_report.required_coupled_sigma_access,
        label="required_center_sigma_z_hat_at_fixed_covariance",
    )
    required_center_sigma_share_of_anchor = _positive_ratio(
        required_center_sigma,
        coverage_anchor_center_sigma,
        label="required_center_sigma_share_of_anchor",
    )
    required_center_sigma_share_of_companion = _positive_ratio(
        required_center_sigma,
        overshoot_companion_center_sigma,
        label="required_center_sigma_share_of_companion",
    )
    driver_signature = _driver_signature(
        repair_report=repair_report,
        required_covariance_share_of_companion=required_covariance_share_of_companion,
        required_center_sigma_share_of_anchor=required_center_sigma_share_of_anchor,
    )
    canonical_digest = (
        "- coupled shoulder-center access already collapses to "
        "`|covariance(z=0.25, z=0.15)| / sigma_z_hat(z=0.15)`: seed "
        f"`{repair_report.coverage_anchor_random_state}` is "
        f"`{_format_float(current_abs_covariance)} / "
        f"{_format_float(coverage_anchor_center_sigma)} = "
        f"{_format_float(current_coupled_sigma_access)}`, while seed "
        f"`{repair_report.overshoot_companion_random_state}` is "
        f"`{_format_float(companion_abs_covariance)} / "
        f"{_format_float(overshoot_companion_center_sigma)} = "
        f"{_format_float(companion_coupled_sigma_access)}`, so plain shoulder "
        "`sigma_z_hat` is not the binding primitive",
        "- the channel split is numerator-dominated: raw shoulder-center covariance "
        f"jumps by `{_format_ratio(covariance_entry_ratio)}`, while center "
        f"`sigma_z_hat` only widens by `{_format_ratio(center_sigma_z_hat_ratio)}`, "
        f"which exactly explains the `{_format_ratio(coupled_sigma_access_ratio)}` "
        "coupled-access gap",
        "- to close the last "
        f"`{_format_percent(repair_report.local_scale_access_shortfall_share)}` "
        "local-scale access shortfall with center scale frozen, seed "
        f"`{repair_report.coverage_anchor_random_state}` only needs "
        "`|covariance(z=0.25, z=0.15)|` to rise from "
        f"`{_format_float(current_abs_covariance)}` to "
        f"`{_format_float(required_abs_covariance)}`: "
        f"`{_format_ratio(required_covariance_multiple_vs_anchor)}` the current "
        "anchor covariance, but still just "
        f"`{_format_percent(required_covariance_share_of_companion)}` of companion "
        f"covariance `{_format_float(companion_abs_covariance)}`",
        "- if covariance stayed frozen instead, center `sigma_z_hat(z=0.15)` "
        f"would have to shrink from `{_format_float(coverage_anchor_center_sigma)}` "
        f"to `{_format_float(required_center_sigma)}`, i.e. "
        f"`{_format_percent(required_center_sigma_share_of_anchor)}` of anchor "
        "center scale and "
        f"`{_format_percent(required_center_sigma_share_of_companion)}` of "
        f"companion center scale; current Trigger 2 implication: "
        f"`{driver_signature}`",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-center-coupling-channel-probe"
        ),
        policy_digest=repair_report.policy_digest,
        binding_design=repair_report.binding_design,
        coverage_anchor_random_state=repair_report.coverage_anchor_random_state,
        overshoot_companion_random_state=repair_report.overshoot_companion_random_state,
        center_grid_value=repair_report.center_grid_value,
        failing_right_shoulder_grid_value=repair_report.failing_right_shoulder_grid_value,
        current_abs_shoulder_center_covariance=current_abs_covariance,
        companion_abs_shoulder_center_covariance=companion_abs_covariance,
        covariance_entry_ratio=covariance_entry_ratio,
        coverage_anchor_center_sigma_z_hat=coverage_anchor_center_sigma,
        overshoot_companion_center_sigma_z_hat=overshoot_companion_center_sigma,
        center_sigma_z_hat_ratio=center_sigma_z_hat_ratio,
        current_coupled_sigma_access=current_coupled_sigma_access,
        companion_coupled_sigma_access=companion_coupled_sigma_access,
        coupled_sigma_access_ratio=coupled_sigma_access_ratio,
        required_abs_shoulder_center_covariance_at_fixed_center_sigma=(
            required_abs_covariance
        ),
        required_covariance_multiple_vs_anchor=required_covariance_multiple_vs_anchor,
        required_covariance_share_of_companion=required_covariance_share_of_companion,
        required_center_sigma_z_hat_at_fixed_covariance=required_center_sigma,
        required_center_sigma_share_of_anchor=required_center_sigma_share_of_anchor,
        required_center_sigma_share_of_companion=(
            required_center_sigma_share_of_companion
        ),
        driver_signature=driver_signature,
        canonical_center_coupling_channel_digest=canonical_digest,
    )


def _build_seed_window_covariance_report_for_repair(
    repair_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport
    ),
) -> Phase7MonteCarloWideningPolicySeedWindowCovarianceReport:
    return _build_seed_window_covariance_report_for_binding_design(
        repair_report.binding_design,
        coverage_anchor_random_state=repair_report.coverage_anchor_random_state,
        overshoot_companion_random_state=repair_report.overshoot_companion_random_state,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport
):
    repair_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe()
    )
    if _is_repo_side_canonical_repair(repair_report):
        return _build_repo_side_center_coupling_channel_report(repair_report)
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_report(
        repair_report=repair_report,
        seed_window_covariance_report=_build_seed_window_covariance_report_for_repair(
            repair_report
        ),
    )
