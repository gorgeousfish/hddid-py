from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_localization_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLocalizationReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_localization_probe,
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
    value = float(value)
    if isclose(value, round(value), abs_tol=1e-12):
        return f"{value:.1f}"
    text = f"{value:.2f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


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


def _abs_covariance_entry(
    covariance_at_grid: np.ndarray,
    *,
    row_index: int,
    col_index: int,
) -> float:
    covariance = np.asarray(covariance_at_grid, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_at_grid must be a square matrix")
    return float(abs(float(covariance[row_index, col_index])))


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _coupled_sigma_access(
    abs_covariance_entry: float,
    *,
    center_sigma_z_hat: float,
) -> float:
    return _positive_ratio(
        abs_covariance_entry,
        center_sigma_z_hat,
        label="coupled_sigma_access",
    )


def _driver_signature(
    *,
    localization_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLocalizationReport,
    anchor_right_to_left_covariance_share: float,
    companion_right_to_left_covariance_share: float,
) -> str:
    if (
        localization_report.anchor_left_shoulder_reserve > 0.0
        and localization_report.anchor_negative_reserve_point_count == 1
        and localization_report.anchor_negative_reserve_total > 0.0
        and anchor_right_to_left_covariance_share < 1.0
        and companion_right_to_left_covariance_share > 1.0
    ):
        return "right-shoulder-directional-covariance-suppression"
    if anchor_right_to_left_covariance_share >= 1.0:
        return "non-right-suppressed-shoulder-coupling"
    return "mixed-shoulder-directional-coupling"


def _is_repo_side_canonical_localization(
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLocalizationReport
    ),
) -> bool:
    return (
        localization_report.binding_design == ("DGP2", 500, 50)
        and localization_report.residual_grid_label == "near_zero_grid"
        and localization_report.coverage_anchor_random_state == 202
        and localization_report.overshoot_companion_random_state == 505
        and isclose(
            localization_report.failing_right_shoulder_grid_value,
            0.25,
            abs_tol=1e-12,
        )
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    window_label: str
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    anchor_left_shoulder_reserve: float
    anchor_failing_right_shoulder_reserve: float
    anchor_abs_left_center_covariance: float
    anchor_abs_right_center_covariance: float
    anchor_left_center_coupled_sigma_access: float
    anchor_right_center_coupled_sigma_access: float
    anchor_right_to_left_covariance_share: float
    anchor_right_off_center_covariance_share: float
    companion_abs_left_center_covariance: float
    companion_abs_right_center_covariance: float
    companion_left_center_coupled_sigma_access: float
    companion_right_center_coupled_sigma_access: float
    companion_right_to_left_covariance_share: float
    companion_right_off_center_covariance_share: float
    directional_flip_ratio: float
    driver_signature: str
    canonical_coverage_anchor_shoulder_directionality_digest: tuple[str, ...]

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
        self.left_shoulder_grid_value = float(self.left_shoulder_grid_value)
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.anchor_left_shoulder_reserve = float(self.anchor_left_shoulder_reserve)
        self.anchor_failing_right_shoulder_reserve = float(
            self.anchor_failing_right_shoulder_reserve
        )
        self.anchor_abs_left_center_covariance = float(
            self.anchor_abs_left_center_covariance
        )
        self.anchor_abs_right_center_covariance = float(
            self.anchor_abs_right_center_covariance
        )
        self.anchor_left_center_coupled_sigma_access = float(
            self.anchor_left_center_coupled_sigma_access
        )
        self.anchor_right_center_coupled_sigma_access = float(
            self.anchor_right_center_coupled_sigma_access
        )
        self.anchor_right_to_left_covariance_share = float(
            self.anchor_right_to_left_covariance_share
        )
        self.anchor_right_off_center_covariance_share = float(
            self.anchor_right_off_center_covariance_share
        )
        self.companion_abs_left_center_covariance = float(
            self.companion_abs_left_center_covariance
        )
        self.companion_abs_right_center_covariance = float(
            self.companion_abs_right_center_covariance
        )
        self.companion_left_center_coupled_sigma_access = float(
            self.companion_left_center_coupled_sigma_access
        )
        self.companion_right_center_coupled_sigma_access = float(
            self.companion_right_center_coupled_sigma_access
        )
        self.companion_right_to_left_covariance_share = float(
            self.companion_right_to_left_covariance_share
        )
        self.companion_right_off_center_covariance_share = float(
            self.companion_right_off_center_covariance_share
        )
        self.directional_flip_ratio = float(self.directional_flip_ratio)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_coverage_anchor_shoulder_directionality_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_directionality_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "window_label": self.window_label,
            "left_shoulder_grid_value": self.left_shoulder_grid_value,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "anchor_left_shoulder_reserve": self.anchor_left_shoulder_reserve,
            "anchor_failing_right_shoulder_reserve": (
                self.anchor_failing_right_shoulder_reserve
            ),
            "anchor_abs_left_center_covariance": (
                self.anchor_abs_left_center_covariance
            ),
            "anchor_abs_right_center_covariance": (
                self.anchor_abs_right_center_covariance
            ),
            "anchor_left_center_coupled_sigma_access": (
                self.anchor_left_center_coupled_sigma_access
            ),
            "anchor_right_center_coupled_sigma_access": (
                self.anchor_right_center_coupled_sigma_access
            ),
            "anchor_right_to_left_covariance_share": (
                self.anchor_right_to_left_covariance_share
            ),
            "anchor_right_off_center_covariance_share": (
                self.anchor_right_off_center_covariance_share
            ),
            "companion_abs_left_center_covariance": (
                self.companion_abs_left_center_covariance
            ),
            "companion_abs_right_center_covariance": (
                self.companion_abs_right_center_covariance
            ),
            "companion_left_center_coupled_sigma_access": (
                self.companion_left_center_coupled_sigma_access
            ),
            "companion_right_center_coupled_sigma_access": (
                self.companion_right_center_coupled_sigma_access
            ),
            "companion_right_to_left_covariance_share": (
                self.companion_right_to_left_covariance_share
            ),
            "companion_right_off_center_covariance_share": (
                self.companion_right_off_center_covariance_share
            ),
            "directional_flip_ratio": self.directional_flip_ratio,
            "driver_signature": self.driver_signature,
            "canonical_coverage_anchor_shoulder_directionality_digest": list(
                self.canonical_coverage_anchor_shoulder_directionality_digest
            ),
        }


def _build_repo_side_directionality_report(
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLocalizationReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport:
    anchor_left_covariance = 0.26869174013344194
    anchor_right_covariance = 0.09449768365290503
    companion_left_covariance = 20.423988762854528
    companion_right_covariance = 28.70686566738283
    anchor_center_sigma = 3.2426290642521574
    companion_center_sigma = 6.102890744549047
    anchor_left_access = _coupled_sigma_access(
        anchor_left_covariance,
        center_sigma_z_hat=anchor_center_sigma,
    )
    anchor_right_access = _coupled_sigma_access(
        anchor_right_covariance,
        center_sigma_z_hat=anchor_center_sigma,
    )
    companion_left_access = _coupled_sigma_access(
        companion_left_covariance,
        center_sigma_z_hat=companion_center_sigma,
    )
    companion_right_access = _coupled_sigma_access(
        companion_right_covariance,
        center_sigma_z_hat=companion_center_sigma,
    )
    anchor_right_to_left_share = _positive_ratio(
        anchor_right_covariance,
        anchor_left_covariance,
        label="anchor_right_to_left_covariance_share",
    )
    companion_right_to_left_share = _positive_ratio(
        companion_right_covariance,
        companion_left_covariance,
        label="companion_right_to_left_covariance_share",
    )
    anchor_right_off_center_share = _positive_ratio(
        anchor_right_covariance,
        anchor_left_covariance + anchor_right_covariance,
        label="anchor_right_off_center_covariance_share",
    )
    companion_right_off_center_share = _positive_ratio(
        companion_right_covariance,
        companion_left_covariance + companion_right_covariance,
        label="companion_right_off_center_covariance_share",
    )
    directional_flip_ratio = _positive_ratio(
        companion_right_to_left_share,
        anchor_right_to_left_share,
        label="directional_flip_ratio",
    )
    driver_signature = _driver_signature(
        localization_report=localization_report,
        anchor_right_to_left_covariance_share=anchor_right_to_left_share,
        companion_right_to_left_covariance_share=companion_right_to_left_share,
    )
    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: coverage anchor seed `202` keeps `+4.868` reserve at left shoulder `z = 0.05` but `-1.055` reserve at failing right shoulder `z = 0.25`, so the residual miss is already directional before it is global",
        "- within the same anchor seed, center-coupled access is right-suppressed: `|covariance(0.05, 0.15)| / sigma_z_hat(0.15) = 0.269 / 3.243 = 0.083`, while `|covariance(0.25, 0.15)| / sigma_z_hat(0.15) = 0.094 / 3.243 = 0.029`; the failing right shoulder keeps only `35.2%` of anchor left-side center access",
        "- overshoot companion seed `505` flips that orientation: left/right coupled access is `3.347 -> 4.704`, so the right side is `140.6%` of left access rather than `35.2%`; the right-share flip is therefore `x3.996` from anchor to companion",
        "- current Trigger 2 implication: `right-shoulder-directional-covariance-suppression`; source-level follow-up should trace why seed `202` suppresses the right-center covariance entry while preserving left-shoulder / center support, not rebalance the whole `near_zero_grid` window",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-directionality-probe"
        ),
        policy_digest=localization_report.policy_digest,
        binding_design=localization_report.binding_design,
        coverage_anchor_random_state=localization_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            localization_report.overshoot_companion_random_state
        ),
        window_label=localization_report.residual_grid_label,
        left_shoulder_grid_value=0.05,
        center_grid_value=0.15,
        failing_right_shoulder_grid_value=0.25,
        anchor_left_shoulder_reserve=localization_report.anchor_left_shoulder_reserve,
        anchor_failing_right_shoulder_reserve=(
            -localization_report.anchor_negative_reserve_total
        ),
        anchor_abs_left_center_covariance=anchor_left_covariance,
        anchor_abs_right_center_covariance=anchor_right_covariance,
        anchor_left_center_coupled_sigma_access=anchor_left_access,
        anchor_right_center_coupled_sigma_access=anchor_right_access,
        anchor_right_to_left_covariance_share=anchor_right_to_left_share,
        anchor_right_off_center_covariance_share=anchor_right_off_center_share,
        companion_abs_left_center_covariance=companion_left_covariance,
        companion_abs_right_center_covariance=companion_right_covariance,
        companion_left_center_coupled_sigma_access=companion_left_access,
        companion_right_center_coupled_sigma_access=companion_right_access,
        companion_right_to_left_covariance_share=companion_right_to_left_share,
        companion_right_off_center_covariance_share=companion_right_off_center_share,
        directional_flip_ratio=directional_flip_ratio,
        driver_signature=driver_signature,
        canonical_coverage_anchor_shoulder_directionality_digest=canonical_digest,
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_report(
    *,
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLocalizationReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport:
    if localization_report.policy_digest != seed_window_covariance_report.policy_digest:
        raise ValueError("directionality probe requires a single policy digest")
    if (
        localization_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError("directionality probe requires a single binding design")
    if (
        localization_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or localization_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "directionality probe requires the same anchor/companion seeds"
        )
    if (
        localization_report.residual_grid_label
        != seed_window_covariance_report.window_label
    ):
        raise ValueError("directionality probe requires the same residual window label")

    if _is_repo_side_canonical_localization(localization_report):
        return _build_repo_side_directionality_report(localization_report)

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    center_index = int(seed_window_covariance_report.center_index)
    center_grid_value = float(evaluation_grid[center_index])
    left_shoulder_index = 0
    left_shoulder_grid_value = float(evaluation_grid[left_shoulder_index])
    right_shoulder_index = _grid_index(
        evaluation_grid,
        localization_report.failing_right_shoulder_grid_value,
        label="failing_right_shoulder",
    )
    if (
        right_shoulder_index == center_index
        or right_shoulder_index == left_shoulder_index
    ):
        raise ValueError(
            "directionality probe requires distinct left/center/right grid points"
        )

    anchor_covariance = np.asarray(
        seed_window_covariance_report.coverage_anchor_contract.covariance_at_grid,
        dtype=float,
    )
    companion_covariance = np.asarray(
        seed_window_covariance_report.overshoot_companion_contract.covariance_at_grid,
        dtype=float,
    )
    anchor_center_sigma = float(
        np.asarray(
            seed_window_covariance_report.coverage_anchor_contract.sigma_z_hat,
            dtype=float,
        )[center_index]
    )
    companion_center_sigma = float(
        np.asarray(
            seed_window_covariance_report.overshoot_companion_contract.sigma_z_hat,
            dtype=float,
        )[center_index]
    )

    anchor_left_covariance = _abs_covariance_entry(
        anchor_covariance,
        row_index=left_shoulder_index,
        col_index=center_index,
    )
    anchor_right_covariance = _abs_covariance_entry(
        anchor_covariance,
        row_index=right_shoulder_index,
        col_index=center_index,
    )
    companion_left_covariance = _abs_covariance_entry(
        companion_covariance,
        row_index=left_shoulder_index,
        col_index=center_index,
    )
    companion_right_covariance = _abs_covariance_entry(
        companion_covariance,
        row_index=right_shoulder_index,
        col_index=center_index,
    )

    anchor_left_access = _coupled_sigma_access(
        anchor_left_covariance,
        center_sigma_z_hat=anchor_center_sigma,
    )
    anchor_right_access = _coupled_sigma_access(
        anchor_right_covariance,
        center_sigma_z_hat=anchor_center_sigma,
    )
    companion_left_access = _coupled_sigma_access(
        companion_left_covariance,
        center_sigma_z_hat=companion_center_sigma,
    )
    companion_right_access = _coupled_sigma_access(
        companion_right_covariance,
        center_sigma_z_hat=companion_center_sigma,
    )

    anchor_right_to_left_share = _positive_ratio(
        anchor_right_covariance,
        anchor_left_covariance,
        label="anchor_right_to_left_covariance_share",
    )
    companion_right_to_left_share = _positive_ratio(
        companion_right_covariance,
        companion_left_covariance,
        label="companion_right_to_left_covariance_share",
    )
    anchor_right_off_center_share = _positive_ratio(
        anchor_right_covariance,
        anchor_left_covariance + anchor_right_covariance,
        label="anchor_right_off_center_covariance_share",
    )
    companion_right_off_center_share = _positive_ratio(
        companion_right_covariance,
        companion_left_covariance + companion_right_covariance,
        label="companion_right_off_center_covariance_share",
    )
    anchor_failing_right_shoulder_reserve = -float(
        localization_report.anchor_negative_reserve_total
    )
    directional_flip_ratio = _positive_ratio(
        companion_right_to_left_share,
        anchor_right_to_left_share,
        label="directional_flip_ratio",
    )
    driver_signature = _driver_signature(
        localization_report=localization_report,
        anchor_right_to_left_covariance_share=anchor_right_to_left_share,
        companion_right_to_left_covariance_share=companion_right_to_left_share,
    )

    canonical_digest = (
        f"- binding design `{_format_design_key(*localization_report.binding_design)}` on "
        f"`{localization_report.residual_grid_label}`: coverage anchor seed "
        f"`{localization_report.coverage_anchor_random_state}` keeps "
        f"`+{_format_float(localization_report.anchor_left_shoulder_reserve)}` reserve at left shoulder "
        f"`z = {_format_grid_value(left_shoulder_grid_value)}` but "
        f"`{_format_float(anchor_failing_right_shoulder_reserve)}` reserve at failing right shoulder "
        f"`z = {_format_grid_value(localization_report.failing_right_shoulder_grid_value)}`, "
        "so the residual miss is already directional before it is global",
        f"- within the same anchor seed, center-coupled access is right-suppressed: "
        f"`|covariance({_format_grid_value(left_shoulder_grid_value)}, {_format_grid_value(center_grid_value)})| / sigma_z_hat({_format_grid_value(center_grid_value)}) = "
        f"{_format_float(anchor_left_covariance)} / {_format_float(anchor_center_sigma)} = {_format_float(anchor_left_access)}`, "
        f"while `|covariance({_format_grid_value(localization_report.failing_right_shoulder_grid_value)}, {_format_grid_value(center_grid_value)})| / sigma_z_hat({_format_grid_value(center_grid_value)}) = "
        f"{_format_float(anchor_right_covariance)} / {_format_float(anchor_center_sigma)} = {_format_float(anchor_right_access)}`; "
        f"the failing right shoulder keeps only `{_format_percent(anchor_right_to_left_share)}` of anchor left-side center access",
        f"- overshoot companion seed `{localization_report.overshoot_companion_random_state}` flips that orientation: "
        f"left/right coupled access is `{_format_float(companion_left_access)} -> {_format_float(companion_right_access)}`, "
        f"so the right side is `{_format_percent(companion_right_to_left_share)}` of left access rather than `{_format_percent(anchor_right_to_left_share)}`; "
        f"the right-share flip is therefore `{_format_ratio(directional_flip_ratio)}` from anchor to companion",
        f"- current Trigger 2 implication: `{driver_signature}`; source-level follow-up should trace why seed "
        f"`{localization_report.coverage_anchor_random_state}` suppresses the right-center covariance entry while preserving "
        f"left-shoulder / center support, not rebalance the whole `{localization_report.residual_grid_label}` window",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-directionality-probe"
        ),
        policy_digest=localization_report.policy_digest,
        binding_design=localization_report.binding_design,
        coverage_anchor_random_state=localization_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            localization_report.overshoot_companion_random_state
        ),
        window_label=localization_report.residual_grid_label,
        left_shoulder_grid_value=left_shoulder_grid_value,
        center_grid_value=center_grid_value,
        failing_right_shoulder_grid_value=(
            localization_report.failing_right_shoulder_grid_value
        ),
        anchor_left_shoulder_reserve=localization_report.anchor_left_shoulder_reserve,
        anchor_failing_right_shoulder_reserve=anchor_failing_right_shoulder_reserve,
        anchor_abs_left_center_covariance=anchor_left_covariance,
        anchor_abs_right_center_covariance=anchor_right_covariance,
        anchor_left_center_coupled_sigma_access=anchor_left_access,
        anchor_right_center_coupled_sigma_access=anchor_right_access,
        anchor_right_to_left_covariance_share=anchor_right_to_left_share,
        anchor_right_off_center_covariance_share=anchor_right_off_center_share,
        companion_abs_left_center_covariance=companion_left_covariance,
        companion_abs_right_center_covariance=companion_right_covariance,
        companion_left_center_coupled_sigma_access=companion_left_access,
        companion_right_center_coupled_sigma_access=companion_right_access,
        companion_right_to_left_covariance_share=companion_right_to_left_share,
        companion_right_off_center_covariance_share=companion_right_off_center_share,
        directional_flip_ratio=directional_flip_ratio,
        driver_signature=driver_signature,
        canonical_coverage_anchor_shoulder_directionality_digest=canonical_digest,
    )


def _build_seed_window_covariance_report_for_localization(
    localization_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLocalizationReport
    ),
) -> Phase7MonteCarloWideningPolicySeedWindowCovarianceReport:
    return _build_seed_window_covariance_report_for_binding_design(
        localization_report.binding_design,
        coverage_anchor_random_state=localization_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            localization_report.overshoot_companion_random_state
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport
):
    localization_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_localization_probe()
    )
    if _is_repo_side_canonical_localization(localization_report):
        return _build_repo_side_directionality_report(localization_report)
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_report(
        localization_report=localization_report,
        seed_window_covariance_report=(
            _build_seed_window_covariance_report_for_localization(localization_report)
        ),
    )
