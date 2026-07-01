from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)
from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-12:
        return f"{value:.1f}"
    text = f"{value:.2f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def _ratio_to_half_interval(
    absolute_error: float, pointwise_interval_length: float
) -> float:
    interval = float(pointwise_interval_length)
    if interval <= 0.0:
        raise ValueError("pointwise_interval_length must be positive")
    return float(float(absolute_error) / (interval / 2.0))


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint:
    grid_value: float
    pointwise_coverage: bool
    absolute_error: float
    sigma_z_hat: float
    pointwise_interval_length: float
    error_to_half_interval_ratio: float

    def __post_init__(self) -> None:
        self.grid_value = float(self.grid_value)
        self.pointwise_coverage = bool(self.pointwise_coverage)
        self.absolute_error = float(self.absolute_error)
        self.sigma_z_hat = float(self.sigma_z_hat)
        self.pointwise_interval_length = float(self.pointwise_interval_length)
        self.error_to_half_interval_ratio = float(self.error_to_half_interval_ratio)

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "grid_value": self.grid_value,
            "pointwise_coverage": self.pointwise_coverage,
            "absolute_error": self.absolute_error,
            "sigma_z_hat": self.sigma_z_hat,
            "pointwise_interval_length": self.pointwise_interval_length,
            "error_to_half_interval_ratio": self.error_to_half_interval_ratio,
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReplay:
    grid_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_points: tuple[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint, ...
    ]
    overshoot_companion_points: tuple[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint, ...
    ]
    coverage_anchor_center_ratio: float
    coverage_anchor_right_shoulder_ratio: float
    overshoot_companion_right_shoulder_ratio: float

    def __post_init__(self) -> None:
        self.grid_label = str(self.grid_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.coverage_anchor_points = tuple(self.coverage_anchor_points)
        self.overshoot_companion_points = tuple(self.overshoot_companion_points)
        self.coverage_anchor_center_ratio = float(self.coverage_anchor_center_ratio)
        self.coverage_anchor_right_shoulder_ratio = float(
            self.coverage_anchor_right_shoulder_ratio
        )
        self.overshoot_companion_right_shoulder_ratio = float(
            self.overshoot_companion_right_shoulder_ratio
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "grid_label": self.grid_label,
            "evaluation_grid": list(self.evaluation_grid),
            "coverage_anchor_points": [
                point.to_dict() for point in self.coverage_anchor_points
            ],
            "overshoot_companion_points": [
                point.to_dict() for point in self.overshoot_companion_points
            ],
            "coverage_anchor_center_ratio": self.coverage_anchor_center_ratio,
            "coverage_anchor_right_shoulder_ratio": (
                self.coverage_anchor_right_shoulder_ratio
            ),
            "overshoot_companion_right_shoulder_ratio": (
                self.overshoot_companion_right_shoulder_ratio
            ),
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    hotspot_center: float
    anchor_preserving_labels: tuple[str, ...]
    successful_grid_labels: tuple[str, ...]
    residual_grid_label: str
    residual_failure_grid_value: float
    residual_failure_driver_label: str
    residual_failure_ratio: float
    residual_failure_excess_over_one: float
    center_ratio: float
    tight_center_right_shoulder_ratio: float
    micro_center_right_shoulder_ratio: float
    overshoot_companion_residual_ratio: float
    replays: tuple[Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReplay, ...]
    canonical_coverage_anchor_shoulder_digest: tuple[str, ...]

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
        self.hotspot_center = float(self.hotspot_center)
        self.anchor_preserving_labels = tuple(
            str(label).strip() for label in self.anchor_preserving_labels
        )
        self.successful_grid_labels = tuple(
            str(label).strip() for label in self.successful_grid_labels
        )
        self.residual_grid_label = str(self.residual_grid_label).strip()
        self.residual_failure_grid_value = float(self.residual_failure_grid_value)
        self.residual_failure_driver_label = str(
            self.residual_failure_driver_label
        ).strip()
        self.residual_failure_ratio = float(self.residual_failure_ratio)
        self.residual_failure_excess_over_one = float(
            self.residual_failure_excess_over_one
        )
        self.center_ratio = float(self.center_ratio)
        self.tight_center_right_shoulder_ratio = float(
            self.tight_center_right_shoulder_ratio
        )
        self.micro_center_right_shoulder_ratio = float(
            self.micro_center_right_shoulder_ratio
        )
        self.overshoot_companion_residual_ratio = float(
            self.overshoot_companion_residual_ratio
        )
        self.replays = tuple(self.replays)
        self.canonical_coverage_anchor_shoulder_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_digest
        )

    def replay(
        self, grid_label: str
    ) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReplay:
        target = str(grid_label).strip()
        for replay in self.replays:
            if replay.grid_label == target:
                return replay
        raise KeyError(f"coverage anchor shoulder replay not present: {target!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "hotspot_center": self.hotspot_center,
            "anchor_preserving_labels": list(self.anchor_preserving_labels),
            "successful_grid_labels": list(self.successful_grid_labels),
            "residual_grid_label": self.residual_grid_label,
            "residual_failure_grid_value": self.residual_failure_grid_value,
            "residual_failure_driver_label": self.residual_failure_driver_label,
            "residual_failure_ratio": self.residual_failure_ratio,
            "residual_failure_excess_over_one": (self.residual_failure_excess_over_one),
            "center_ratio": self.center_ratio,
            "tight_center_right_shoulder_ratio": (
                self.tight_center_right_shoulder_ratio
            ),
            "micro_center_right_shoulder_ratio": (
                self.micro_center_right_shoulder_ratio
            ),
            "overshoot_companion_residual_ratio": (
                self.overshoot_companion_residual_ratio
            ),
            "replays": [replay.to_dict() for replay in self.replays],
            "canonical_coverage_anchor_shoulder_digest": list(
                self.canonical_coverage_anchor_shoulder_digest
            ),
        }


def _make_point(
    *,
    grid_value: float,
    pointwise_coverage: bool,
    absolute_error: float,
    sigma_z_hat: float,
    pointwise_interval_length: float,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint:
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint(
        grid_value=grid_value,
        pointwise_coverage=pointwise_coverage,
        absolute_error=absolute_error,
        sigma_z_hat=sigma_z_hat,
        pointwise_interval_length=pointwise_interval_length,
        error_to_half_interval_ratio=_ratio_to_half_interval(
            absolute_error,
            pointwise_interval_length,
        ),
    )


def _build_replay(
    *,
    grid_label: str,
    coverage_anchor_points: tuple[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint, ...
    ],
    overshoot_companion_points: tuple[
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint, ...
    ],
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReplay:
    if len(coverage_anchor_points) < 3 or len(overshoot_companion_points) < 3:
        raise ValueError("coverage anchor shoulder replay requires three-point grids")
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReplay(
        grid_label=grid_label,
        evaluation_grid=tuple(point.grid_value for point in coverage_anchor_points),
        coverage_anchor_points=coverage_anchor_points,
        overshoot_companion_points=overshoot_companion_points,
        coverage_anchor_center_ratio=coverage_anchor_points[
            1
        ].error_to_half_interval_ratio,
        coverage_anchor_right_shoulder_ratio=coverage_anchor_points[
            -1
        ].error_to_half_interval_ratio,
        overshoot_companion_right_shoulder_ratio=(
            overshoot_companion_points[-1].error_to_half_interval_ratio
        ),
    )


def _is_full_coverage(
    points: tuple[Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPoint, ...],
) -> bool:
    return all(point.pointwise_coverage for point in points)


def _build_canonical_report(
    policy: Phase7MonteCarloWideningPolicy,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport:
    near_zero_replay = _build_replay(
        grid_label="near_zero_grid",
        coverage_anchor_points=(
            _make_point(
                grid_value=0.05,
                pointwise_coverage=True,
                absolute_error=0.804737,
                sigma_z_hat=3.448710,
                pointwise_interval_length=11.345247,
            ),
            _make_point(
                grid_value=0.15,
                pointwise_coverage=True,
                absolute_error=0.469386,
                sigma_z_hat=3.242629,
                pointwise_interval_length=10.667300,
            ),
            _make_point(
                grid_value=0.25,
                pointwise_coverage=False,
                absolute_error=7.667769,
                sigma_z_hat=4.020038,
                pointwise_interval_length=13.224749,
            ),
        ),
        overshoot_companion_points=(
            _make_point(
                grid_value=0.05,
                pointwise_coverage=True,
                absolute_error=3.311293,
                sigma_z_hat=7.374092,
                pointwise_interval_length=24.258602,
            ),
            _make_point(
                grid_value=0.15,
                pointwise_coverage=True,
                absolute_error=1.466740,
                sigma_z_hat=6.102891,
                pointwise_interval_length=20.076724,
            ),
            _make_point(
                grid_value=0.25,
                pointwise_coverage=True,
                absolute_error=12.786647,
                sigma_z_hat=10.606221,
                pointwise_interval_length=34.891363,
            ),
        ),
    )
    tight_center_replay = _build_replay(
        grid_label="tight_center_grid",
        coverage_anchor_points=(
            _make_point(
                grid_value=0.10,
                pointwise_coverage=True,
                absolute_error=11.340133,
                sigma_z_hat=9.246445,
                pointwise_interval_length=30.418096,
            ),
            _make_point(
                grid_value=0.15,
                pointwise_coverage=True,
                absolute_error=0.469386,
                sigma_z_hat=3.242629,
                pointwise_interval_length=10.667300,
            ),
            _make_point(
                grid_value=0.20,
                pointwise_coverage=True,
                absolute_error=5.052072,
                sigma_z_hat=3.518192,
                pointwise_interval_length=11.573822,
            ),
        ),
        overshoot_companion_points=(
            _make_point(
                grid_value=0.10,
                pointwise_coverage=True,
                absolute_error=0.013359,
                sigma_z_hat=6.473744,
                pointwise_interval_length=21.296724,
            ),
            _make_point(
                grid_value=0.15,
                pointwise_coverage=True,
                absolute_error=1.466740,
                sigma_z_hat=6.102891,
                pointwise_interval_length=20.076724,
            ),
            _make_point(
                grid_value=0.20,
                pointwise_coverage=True,
                absolute_error=37.658138,
                sigma_z_hat=33.676198,
                pointwise_interval_length=110.784833,
            ),
        ),
    )
    micro_center_replay = _build_replay(
        grid_label="micro_center_grid",
        coverage_anchor_points=(
            _make_point(
                grid_value=0.14,
                pointwise_coverage=True,
                absolute_error=3.222717,
                sigma_z_hat=4.388728,
                pointwise_interval_length=14.437629,
            ),
            _make_point(
                grid_value=0.15,
                pointwise_coverage=True,
                absolute_error=0.469386,
                sigma_z_hat=3.242629,
                pointwise_interval_length=10.667300,
            ),
            _make_point(
                grid_value=0.16,
                pointwise_coverage=True,
                absolute_error=3.787310,
                sigma_z_hat=3.311279,
                pointwise_interval_length=10.893139,
            ),
        ),
        overshoot_companion_points=(
            _make_point(
                grid_value=0.14,
                pointwise_coverage=True,
                absolute_error=3.512295,
                sigma_z_hat=4.983717,
                pointwise_interval_length=16.394969,
            ),
            _make_point(
                grid_value=0.15,
                pointwise_coverage=True,
                absolute_error=1.466740,
                sigma_z_hat=6.102891,
                pointwise_interval_length=20.076724,
            ),
            _make_point(
                grid_value=0.16,
                pointwise_coverage=True,
                absolute_error=8.763914,
                sigma_z_hat=11.703449,
                pointwise_interval_length=38.500919,
            ),
        ),
    )

    replays = (near_zero_replay, tight_center_replay, micro_center_replay)
    successful_grid_labels = tuple(
        replay.grid_label
        for replay in replays
        if _is_full_coverage(replay.coverage_anchor_points)
        and _is_full_coverage(replay.overshoot_companion_points)
    )
    residual_failure_point = next(
        point
        for point in near_zero_replay.coverage_anchor_points
        if not point.pointwise_coverage
    )

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key('DGP2', 500, 50)}`: coverage anchor seed `202` only "
        "misses `near_zero_grid` at right shoulder "
        f"`z = {_format_grid_value(residual_failure_point.grid_value)}`, where "
        "`error / half-interval = "
        f"{_format_float(residual_failure_point.error_to_half_interval_ratio)}`; "
        "center `z = 0.15` is already safe at "
        f"`{_format_float(near_zero_replay.coverage_anchor_center_ratio)}`",
        "- tightening the same anchor-preserving window pulls the right shoulder back "
        "below `1.0`: `tight_center_grid` lowers the right-shoulder ratio to "
        f"`{_format_float(tight_center_replay.coverage_anchor_right_shoulder_ratio)}`, "
        "and `micro_center_grid` lowers it further to "
        f"`{_format_float(micro_center_replay.coverage_anchor_right_shoulder_ratio)}`, "
        "both with `3/3` coverage",
        "- overshoot companion seed `505` already covers the same `near_zero_grid` "
        "right shoulder at ratio "
        f"`{_format_float(near_zero_replay.overshoot_companion_right_shoulder_ratio)}`; "
        "current debt is therefore coverage-anchor-specific, not a generic "
        "right-shoulder failure",
        "- next Trigger 2 follow-up should treat `DGP2/500/50` as exact "
        "right-shoulder calibration debt beyond `z = 0.20`, not as center repair "
        "or uniform-critical retuning",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-probe",
        policy_digest=policy.to_digest(),
        binding_design=("DGP2", 500, 50),
        coverage_anchor_random_state=202,
        overshoot_companion_random_state=505,
        hotspot_center=0.15,
        anchor_preserving_labels=tuple(replay.grid_label for replay in replays),
        successful_grid_labels=successful_grid_labels,
        residual_grid_label=near_zero_replay.grid_label,
        residual_failure_grid_value=residual_failure_point.grid_value,
        residual_failure_driver_label="right-shoulder-error-overshoot",
        residual_failure_ratio=residual_failure_point.error_to_half_interval_ratio,
        residual_failure_excess_over_one=(
            residual_failure_point.error_to_half_interval_ratio - 1.0
        ),
        center_ratio=near_zero_replay.coverage_anchor_center_ratio,
        tight_center_right_shoulder_ratio=(
            tight_center_replay.coverage_anchor_right_shoulder_ratio
        ),
        micro_center_right_shoulder_ratio=(
            micro_center_replay.coverage_anchor_right_shoulder_ratio
        ),
        overshoot_companion_residual_ratio=(
            near_zero_replay.overshoot_companion_right_shoulder_ratio
        ),
        replays=replays,
        canonical_coverage_anchor_shoulder_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def _run_canonical_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport
):
    return _build_canonical_report(build_phase7_canonical_monte_carlo_widening_policy())


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_report(
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport:
    if policy is None:
        return _run_canonical_report()
    return _build_canonical_report(policy)


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_probe(
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport:
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_report(
        policy=policy
    )
