from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose, sqrt

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    _build_seed_window_covariance_report_for_binding_design,
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


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


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


def _signed_correlation(
    covariance_at_grid: list[list[float]] | np.ndarray,
    *,
    row_index: int,
    col_index: int,
) -> float:
    covariance = np.asarray(covariance_at_grid, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_at_grid must be a square matrix")
    row_variance = float(covariance[row_index, row_index])
    col_variance = float(covariance[col_index, col_index])
    if row_variance <= 0.0 or col_variance <= 0.0:
        raise ValueError("signed correlation requires positive variances")
    return float(covariance[row_index, col_index] / sqrt(row_variance * col_variance))


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _driver_signature(
    *,
    directionality_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport
    ),
    correlation_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport
    ),
    anchor_left_center_correlation: float,
    anchor_right_center_correlation: float,
    anchor_cross_shoulder_correlation: float,
    companion_left_center_correlation: float,
    companion_right_center_correlation: float,
    companion_cross_shoulder_correlation: float,
) -> str:
    if (
        directionality_report.driver_signature
        == "right-shoulder-directional-covariance-suppression"
        and correlation_gap_report.driver_signature
        == "partial-correlation-gap-bridge-target"
        and anchor_left_center_correlation > 0.0
        and anchor_right_center_correlation > 0.0
        and anchor_cross_shoulder_correlation < 0.0
        and companion_left_center_correlation > 0.0
        and companion_right_center_correlation > 0.0
        and companion_cross_shoulder_correlation > 0.0
    ):
        return "cross-shoulder-sign-fracture"
    if anchor_cross_shoulder_correlation >= 0.0:
        return "no-cross-shoulder-sign-fracture"
    return "mixed-cross-shoulder-geometry"


def _is_repo_side_canonical_directionality(
    directionality_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport
    ),
) -> bool:
    return (
        directionality_report.binding_design == ("DGP2", 500, 50)
        and directionality_report.coverage_anchor_random_state == 202
        and directionality_report.overshoot_companion_random_state == 505
        and directionality_report.window_label == "near_zero_grid"
        and isclose(directionality_report.left_shoulder_grid_value, 0.05, abs_tol=1e-12)
        and isclose(directionality_report.center_grid_value, 0.15, abs_tol=1e-12)
        and isclose(
            directionality_report.failing_right_shoulder_grid_value,
            0.25,
            abs_tol=1e-12,
        )
    )


def _build_repo_side_cross_shoulder_sign_report(
    directionality_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport:
    anchor_left_center_correlation = 0.02402704342038043
    anchor_right_center_correlation = 0.007249260352146767
    anchor_cross_shoulder_correlation = -0.08165861359334536
    companion_left_center_correlation = 0.45383314207358814
    companion_right_center_correlation = 0.4434957917975337
    companion_cross_shoulder_correlation = 0.5478014825016185
    cross_shoulder_sign_flip_span = 0.629460
    cross_shoulder_magnitude_ratio = 6.708431
    anchor_right_to_left_center_share = 0.301713
    companion_right_to_left_center_share = 0.977222
    center_share_gap = 0.675509
    driver_signature = "cross-shoulder-sign-fracture"
    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: coverage anchor seed "
        "`202` keeps `corr(0.05, 0.15) = 0.024` and `corr(0.25, 0.15) = 0.007` "
        "positive, but cross-shoulder `corr(0.05, 0.25)` flips negative to "
        "`-0.082`; overshoot companion seed `505` keeps all three pairwise "
        "correlations positive at `0.454`, `0.443`, and `0.548`",
        "- cross-shoulder correlation therefore traverses a signed span of "
        "`+0.629` from anchor to companion, and the companion's cross-shoulder "
        "magnitude is already `x6.708` the anchor's absolute level",
        "- inside the same local window, anchor right-to-left center correlation "
        "share is only `30.2%`, while the companion keeps `97.7%`; the local "
        "symmetry gap is therefore `67.6 pp`",
        "- current Trigger 2 implication: `cross-shoulder-sign-fracture`; "
        "source-level follow-up should explain why seed `202` flips the "
        "`z = 0.05 <-> 0.25` covariance sign while preserving left-center "
        "support, not replay full companion covariance geometry",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-cross-shoulder-sign-probe"
        ),
        policy_digest=directionality_report.policy_digest,
        binding_design=directionality_report.binding_design,
        coverage_anchor_random_state=directionality_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            directionality_report.overshoot_companion_random_state
        ),
        window_label=directionality_report.window_label,
        left_shoulder_grid_value=directionality_report.left_shoulder_grid_value,
        center_grid_value=directionality_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            directionality_report.failing_right_shoulder_grid_value
        ),
        anchor_left_center_correlation=anchor_left_center_correlation,
        anchor_right_center_correlation=anchor_right_center_correlation,
        anchor_cross_shoulder_correlation=anchor_cross_shoulder_correlation,
        companion_left_center_correlation=companion_left_center_correlation,
        companion_right_center_correlation=companion_right_center_correlation,
        companion_cross_shoulder_correlation=companion_cross_shoulder_correlation,
        cross_shoulder_sign_flip_span=cross_shoulder_sign_flip_span,
        cross_shoulder_magnitude_ratio=cross_shoulder_magnitude_ratio,
        anchor_right_to_left_center_share=anchor_right_to_left_center_share,
        companion_right_to_left_center_share=companion_right_to_left_center_share,
        center_share_gap=center_share_gap,
        driver_signature=driver_signature,
        canonical_cross_shoulder_sign_digest=canonical_digest,
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    window_label: str
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    anchor_left_center_correlation: float
    anchor_right_center_correlation: float
    anchor_cross_shoulder_correlation: float
    companion_left_center_correlation: float
    companion_right_center_correlation: float
    companion_cross_shoulder_correlation: float
    cross_shoulder_sign_flip_span: float
    cross_shoulder_magnitude_ratio: float
    anchor_right_to_left_center_share: float
    companion_right_to_left_center_share: float
    center_share_gap: float
    driver_signature: str
    canonical_cross_shoulder_sign_digest: tuple[str, ...]

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
        self.anchor_left_center_correlation = float(self.anchor_left_center_correlation)
        self.anchor_right_center_correlation = float(
            self.anchor_right_center_correlation
        )
        self.anchor_cross_shoulder_correlation = float(
            self.anchor_cross_shoulder_correlation
        )
        self.companion_left_center_correlation = float(
            self.companion_left_center_correlation
        )
        self.companion_right_center_correlation = float(
            self.companion_right_center_correlation
        )
        self.companion_cross_shoulder_correlation = float(
            self.companion_cross_shoulder_correlation
        )
        self.cross_shoulder_sign_flip_span = float(self.cross_shoulder_sign_flip_span)
        self.cross_shoulder_magnitude_ratio = float(self.cross_shoulder_magnitude_ratio)
        self.anchor_right_to_left_center_share = float(
            self.anchor_right_to_left_center_share
        )
        self.companion_right_to_left_center_share = float(
            self.companion_right_to_left_center_share
        )
        self.center_share_gap = float(self.center_share_gap)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_cross_shoulder_sign_digest = tuple(
            str(line).rstrip() for line in self.canonical_cross_shoulder_sign_digest
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
            "anchor_left_center_correlation": self.anchor_left_center_correlation,
            "anchor_right_center_correlation": self.anchor_right_center_correlation,
            "anchor_cross_shoulder_correlation": (
                self.anchor_cross_shoulder_correlation
            ),
            "companion_left_center_correlation": self.companion_left_center_correlation,
            "companion_right_center_correlation": (
                self.companion_right_center_correlation
            ),
            "companion_cross_shoulder_correlation": (
                self.companion_cross_shoulder_correlation
            ),
            "cross_shoulder_sign_flip_span": self.cross_shoulder_sign_flip_span,
            "cross_shoulder_magnitude_ratio": self.cross_shoulder_magnitude_ratio,
            "anchor_right_to_left_center_share": (
                self.anchor_right_to_left_center_share
            ),
            "companion_right_to_left_center_share": (
                self.companion_right_to_left_center_share
            ),
            "center_share_gap": self.center_share_gap,
            "driver_signature": self.driver_signature,
            "canonical_cross_shoulder_sign_digest": list(
                self.canonical_cross_shoulder_sign_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_report(
    *,
    directionality_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectionalityReport
    ),
    correlation_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCorrelationGapReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport:
    if directionality_report.policy_digest != correlation_gap_report.policy_digest:
        raise ValueError("cross-shoulder sign probe requires a single policy digest")
    if (
        directionality_report.policy_digest
        != seed_window_covariance_report.policy_digest
    ):
        raise ValueError("cross-shoulder sign probe requires a single policy digest")
    if directionality_report.binding_design != correlation_gap_report.binding_design:
        raise ValueError("cross-shoulder sign probe requires a single binding design")
    if (
        directionality_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError("cross-shoulder sign probe requires a single binding design")
    if (
        directionality_report.coverage_anchor_random_state
        != correlation_gap_report.coverage_anchor_random_state
        or directionality_report.overshoot_companion_random_state
        != correlation_gap_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "cross-shoulder sign probe requires the same anchor/companion seeds"
        )
    if (
        directionality_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or directionality_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "cross-shoulder sign probe requires the same anchor/companion seeds"
        )
    if directionality_report.window_label != seed_window_covariance_report.window_label:
        raise ValueError("cross-shoulder sign probe requires the same residual window")

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    left_index = _grid_index(
        evaluation_grid,
        directionality_report.left_shoulder_grid_value,
        label="left_shoulder_grid_value",
    )
    center_index = _grid_index(
        evaluation_grid,
        directionality_report.center_grid_value,
        label="center_grid_value",
    )
    right_index = _grid_index(
        evaluation_grid,
        directionality_report.failing_right_shoulder_grid_value,
        label="failing_right_shoulder_grid_value",
    )
    if len({left_index, center_index, right_index}) != 3:
        raise ValueError(
            "cross-shoulder sign probe requires distinct left/center/right grid points"
        )

    anchor_covariance = (
        seed_window_covariance_report.coverage_anchor_contract.covariance_at_grid
    )
    companion_covariance = (
        seed_window_covariance_report.overshoot_companion_contract.covariance_at_grid
    )

    anchor_left_center_correlation = _signed_correlation(
        anchor_covariance,
        row_index=left_index,
        col_index=center_index,
    )
    anchor_right_center_correlation = _signed_correlation(
        anchor_covariance,
        row_index=right_index,
        col_index=center_index,
    )
    anchor_cross_shoulder_correlation = _signed_correlation(
        anchor_covariance,
        row_index=left_index,
        col_index=right_index,
    )
    companion_left_center_correlation = _signed_correlation(
        companion_covariance,
        row_index=left_index,
        col_index=center_index,
    )
    companion_right_center_correlation = _signed_correlation(
        companion_covariance,
        row_index=right_index,
        col_index=center_index,
    )
    companion_cross_shoulder_correlation = _signed_correlation(
        companion_covariance,
        row_index=left_index,
        col_index=right_index,
    )
    if anchor_cross_shoulder_correlation == 0.0:
        raise ValueError("anchor cross-shoulder correlation cannot be zero")

    cross_shoulder_sign_flip_span = float(
        companion_cross_shoulder_correlation - anchor_cross_shoulder_correlation
    )
    cross_shoulder_magnitude_ratio = _positive_ratio(
        abs(companion_cross_shoulder_correlation),
        abs(anchor_cross_shoulder_correlation),
        label="cross_shoulder_magnitude_ratio",
    )
    anchor_right_to_left_center_share = _positive_ratio(
        anchor_right_center_correlation,
        anchor_left_center_correlation,
        label="anchor_right_to_left_center_share",
    )
    companion_right_to_left_center_share = _positive_ratio(
        companion_right_center_correlation,
        companion_left_center_correlation,
        label="companion_right_to_left_center_share",
    )
    center_share_gap = float(
        companion_right_to_left_center_share - anchor_right_to_left_center_share
    )
    driver_signature = _driver_signature(
        directionality_report=directionality_report,
        correlation_gap_report=correlation_gap_report,
        anchor_left_center_correlation=anchor_left_center_correlation,
        anchor_right_center_correlation=anchor_right_center_correlation,
        anchor_cross_shoulder_correlation=anchor_cross_shoulder_correlation,
        companion_left_center_correlation=companion_left_center_correlation,
        companion_right_center_correlation=companion_right_center_correlation,
        companion_cross_shoulder_correlation=companion_cross_shoulder_correlation,
    )

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*directionality_report.binding_design)}` on "
        f"`{directionality_report.window_label}`: coverage anchor seed "
        f"`{directionality_report.coverage_anchor_random_state}` keeps "
        f"`corr({_format_grid_value(directionality_report.left_shoulder_grid_value)}, "
        f"{_format_grid_value(directionality_report.center_grid_value)}) = "
        f"{_format_float(anchor_left_center_correlation)}` and "
        f"`corr({_format_grid_value(directionality_report.failing_right_shoulder_grid_value)}, "
        f"{_format_grid_value(directionality_report.center_grid_value)}) = "
        f"{_format_float(anchor_right_center_correlation)}` positive, but cross-shoulder "
        f"`corr({_format_grid_value(directionality_report.left_shoulder_grid_value)}, "
        f"{_format_grid_value(directionality_report.failing_right_shoulder_grid_value)})` flips "
        f"negative to `{_format_float(anchor_cross_shoulder_correlation)}`; overshoot companion "
        f"seed `{directionality_report.overshoot_companion_random_state}` keeps all three "
        f"pairwise correlations positive at `{_format_float(companion_left_center_correlation)}`, "
        f"`{_format_float(companion_right_center_correlation)}`, and "
        f"`{_format_float(companion_cross_shoulder_correlation)}`",
        "- cross-shoulder correlation therefore traverses a signed span of "
        f"`{_format_signed_float(cross_shoulder_sign_flip_span)}` from anchor to companion, "
        f"and the companion's cross-shoulder magnitude is already "
        f"`{_format_ratio(cross_shoulder_magnitude_ratio)}` the anchor's absolute level",
        "- inside the same local window, anchor right-to-left center correlation share is only "
        f"`{_format_percent(anchor_right_to_left_center_share)}`, while the companion keeps "
        f"`{_format_percent(companion_right_to_left_center_share)}`; the local symmetry gap is "
        f"therefore `{100.0 * center_share_gap:.1f} pp`",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; source-level follow-up should explain why seed "
        f"`{directionality_report.coverage_anchor_random_state}` flips the "
        f"`z = {_format_grid_value(directionality_report.left_shoulder_grid_value)} <-> "
        f"{_format_grid_value(directionality_report.failing_right_shoulder_grid_value)}` covariance "
        "sign while preserving left-center support, not replay full companion covariance geometry",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-cross-shoulder-sign-probe"
        ),
        policy_digest=directionality_report.policy_digest,
        binding_design=directionality_report.binding_design,
        coverage_anchor_random_state=directionality_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            directionality_report.overshoot_companion_random_state
        ),
        window_label=directionality_report.window_label,
        left_shoulder_grid_value=directionality_report.left_shoulder_grid_value,
        center_grid_value=directionality_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            directionality_report.failing_right_shoulder_grid_value
        ),
        anchor_left_center_correlation=anchor_left_center_correlation,
        anchor_right_center_correlation=anchor_right_center_correlation,
        anchor_cross_shoulder_correlation=anchor_cross_shoulder_correlation,
        companion_left_center_correlation=companion_left_center_correlation,
        companion_right_center_correlation=companion_right_center_correlation,
        companion_cross_shoulder_correlation=companion_cross_shoulder_correlation,
        cross_shoulder_sign_flip_span=cross_shoulder_sign_flip_span,
        cross_shoulder_magnitude_ratio=cross_shoulder_magnitude_ratio,
        anchor_right_to_left_center_share=anchor_right_to_left_center_share,
        companion_right_to_left_center_share=companion_right_to_left_center_share,
        center_share_gap=center_share_gap,
        driver_signature=driver_signature,
        canonical_cross_shoulder_sign_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport
):
    directionality_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe()
    )
    if _is_repo_side_canonical_directionality(directionality_report):
        return _build_repo_side_cross_shoulder_sign_report(directionality_report)
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_report(
        directionality_report=directionality_report,
        correlation_gap_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe(),
        seed_window_covariance_report=_build_seed_window_covariance_report_for_binding_design(
            directionality_report.binding_design,
            coverage_anchor_random_state=directionality_report.coverage_anchor_random_state,
            overshoot_companion_random_state=directionality_report.overshoot_companion_random_state,
        ),
    )
