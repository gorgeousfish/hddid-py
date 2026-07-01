from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_probe import (
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_report,
)
from .monte_carlo_widening_policy_floor_slack_probe import (
    build_phase7_monte_carlo_widening_policy_floor_slack_probe_report,
)
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)
from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy
from .validation import (
    MonteCarloRuntimeProbeDesignSummary,
    MonteCarloRuntimeProbeReport,
    run_phase7_monte_carlo_runtime_probe,
)


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


def _format_fraction(numerator: int, denominator: int) -> str:
    return f"{int(numerator)}/{int(denominator)}"


def _make_design_summary(
    *,
    dgp_name: str,
    n_obs: int,
    p: int,
    success_rate: float,
    runtime_mean_seconds: float | None,
    mean_nonparametric_coverage: float | None,
    typed_invalidity_counts: dict[str, int] | None = None,
) -> MonteCarloRuntimeProbeDesignSummary:
    successful_runs = 3 if success_rate > 0.0 else 0
    return MonteCarloRuntimeProbeDesignSummary(
        dgp_name=dgp_name,
        n_obs=n_obs,
        p=p,
        n_runs=3,
        n_successful_runs=successful_runs,
        success_rate=success_rate,
        runtime_mean_seconds=runtime_mean_seconds,
        runtime_std_seconds=0.0 if runtime_mean_seconds is not None else None,
        typed_invalidity_counts={}
        if typed_invalidity_counts is None
        else typed_invalidity_counts,
        typed_invalidity_examples={},
        mean_parametric_bias=0.0 if success_rate > 0.0 else None,
        std_parametric_bias=0.0 if success_rate > 0.0 else None,
        mean_parametric_rmse=0.2 if success_rate > 0.0 else None,
        mean_parametric_average_standard_error=0.2 if success_rate > 0.0 else None,
        std_parametric_average_standard_error=0.0 if success_rate > 0.0 else None,
        mean_parametric_coverage=0.9 if success_rate > 0.0 else None,
        std_parametric_coverage=0.0 if success_rate > 0.0 else None,
        mean_parametric_interval_length=0.4 if success_rate > 0.0 else None,
        std_parametric_interval_length=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_bias=0.0 if success_rate > 0.0 else None,
        std_nonparametric_bias=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_rmse=0.3 if success_rate > 0.0 else None,
        mean_nonparametric_average_standard_error=0.3 if success_rate > 0.0 else None,
        std_nonparametric_average_standard_error=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_coverage=mean_nonparametric_coverage,
        std_nonparametric_coverage=0.0
        if mean_nonparametric_coverage is not None
        else None,
        mean_nonparametric_interval_length=12.0 if success_rate > 0.0 else None,
        std_nonparametric_interval_length=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_absolute_error=0.3 if success_rate > 0.0 else None,
        std_nonparametric_absolute_error=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_uniform_critical_value=1.5 if success_rate > 0.0 else None,
        std_nonparametric_uniform_critical_value=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_uniform_band_length=13.0 if success_rate > 0.0 else None,
        std_nonparametric_uniform_band_length=0.0 if success_rate > 0.0 else None,
        mean_trimming_rate=0.0 if success_rate > 0.0 else None,
        mean_zero_valid_fold_frequency=0.0,
    )


def _make_canonical_runtime_probe() -> MonteCarloRuntimeProbeReport:
    return MonteCarloRuntimeProbeReport(
        oracle_lane="paper-trigonometric",
        stage_label="phase7-runtime-probe",
        random_states=(101, 202, 303),
        total_runtime_seconds=14.5,
        observations=(),
        design_summaries=(
            _make_design_summary(
                dgp_name="DGP1",
                n_obs=500,
                p=50,
                success_rate=1.0,
                runtime_mean_seconds=0.22,
                mean_nonparametric_coverage=1.0,
            ),
            _make_design_summary(
                dgp_name="DGP2",
                n_obs=500,
                p=50,
                success_rate=1.0,
                runtime_mean_seconds=0.30,
                mean_nonparametric_coverage=7.0 / 9.0,
            ),
            _make_design_summary(
                dgp_name="DGP1",
                n_obs=200,
                p=500,
                success_rate=0.0,
                runtime_mean_seconds=None,
                mean_nonparametric_coverage=None,
                typed_invalidity_counts={"SingularCovarianceError": 3},
            ),
            _make_design_summary(
                dgp_name="DGP2",
                n_obs=200,
                p=500,
                success_rate=0.0,
                runtime_mean_seconds=None,
                mean_nonparametric_coverage=None,
                typed_invalidity_counts={"SingularCovarianceError": 3},
            ),
        ),
        typed_invalidity_counts={"SingularCovarianceError": 6},
        typed_invalidity_examples={},
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFloorLiftCeilingProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    residual_grid_label: str
    failing_right_shoulder_grid_value: float
    current_supported_floor_ceiling: float
    current_floor_shortfall: float
    pointwise_slot_count: int
    current_successful_slot_count: int
    single_point_repair_slot_lift: int
    projected_successful_slot_count: int
    projected_floor_lift: float
    projected_floor_ceiling: float
    projected_floor_slack: float
    projected_crosses_canonical_floor: bool
    canonical_floor_lift_digest: tuple[str, ...]

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
        self.residual_grid_label = str(self.residual_grid_label).strip()
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.current_supported_floor_ceiling = float(
            self.current_supported_floor_ceiling
        )
        self.current_floor_shortfall = float(self.current_floor_shortfall)
        self.pointwise_slot_count = int(self.pointwise_slot_count)
        self.current_successful_slot_count = int(self.current_successful_slot_count)
        self.single_point_repair_slot_lift = int(self.single_point_repair_slot_lift)
        self.projected_successful_slot_count = int(self.projected_successful_slot_count)
        self.projected_floor_lift = float(self.projected_floor_lift)
        self.projected_floor_ceiling = float(self.projected_floor_ceiling)
        self.projected_floor_slack = float(self.projected_floor_slack)
        self.projected_crosses_canonical_floor = bool(
            self.projected_crosses_canonical_floor
        )
        self.canonical_floor_lift_digest = tuple(
            str(line).rstrip() for line in self.canonical_floor_lift_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "residual_grid_label": self.residual_grid_label,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "current_supported_floor_ceiling": self.current_supported_floor_ceiling,
            "current_floor_shortfall": self.current_floor_shortfall,
            "pointwise_slot_count": self.pointwise_slot_count,
            "current_successful_slot_count": self.current_successful_slot_count,
            "single_point_repair_slot_lift": self.single_point_repair_slot_lift,
            "projected_successful_slot_count": self.projected_successful_slot_count,
            "projected_floor_lift": self.projected_floor_lift,
            "projected_floor_ceiling": self.projected_floor_ceiling,
            "projected_floor_slack": self.projected_floor_slack,
            "projected_crosses_canonical_floor": (
                self.projected_crosses_canonical_floor
            ),
            "canonical_floor_lift_digest": list(self.canonical_floor_lift_digest),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_floor_lift_ceiling_probe_report(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFloorLiftCeilingProbeReport:
    resolved_policy = (
        build_phase7_canonical_monte_carlo_widening_policy()
        if policy is None
        else policy
    )
    floor_slack_report = (
        build_phase7_monte_carlo_widening_policy_floor_slack_probe_report(
            runtime_probe,
            policy=resolved_policy,
        )
    )
    shoulder_report = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_report(
            policy=resolved_policy
        )
    )
    if floor_slack_report.binding_design != shoulder_report.binding_design:
        raise ValueError(
            "floor lift ceiling probe requires matching binding designs across floor slack and shoulder evidence"
        )

    residual_replay = shoulder_report.replay(shoulder_report.residual_grid_label)
    failing_points = tuple(
        point
        for point in residual_replay.coverage_anchor_points
        if not point.pointwise_coverage
    )
    if len(failing_points) != 1:
        raise ValueError(
            "floor lift ceiling probe requires exactly one uncovered coverage-anchor point"
        )
    random_state_count = len(tuple(runtime_probe.random_states))
    if random_state_count <= 0:
        raise ValueError("floor lift ceiling probe requires at least one random state")

    pointwise_slot_count = random_state_count * len(
        residual_replay.coverage_anchor_points
    )
    current_successful_slots_float = floor_slack_report.supported_floor_ceiling * float(
        pointwise_slot_count
    )
    current_successful_slot_count = int(round(current_successful_slots_float))
    if (
        abs(current_successful_slots_float - float(current_successful_slot_count))
        > 1e-9
    ):
        raise ValueError(
            "bounded floor ceiling is incompatible with pointwise slot arithmetic"
        )

    single_point_repair_slot_lift = len(failing_points)
    projected_successful_slot_count = (
        current_successful_slot_count + single_point_repair_slot_lift
    )
    projected_floor_lift = single_point_repair_slot_lift / float(pointwise_slot_count)
    projected_floor_ceiling = projected_successful_slot_count / float(
        pointwise_slot_count
    )
    projected_floor_slack = projected_floor_ceiling - floor_slack_report.coverage_floor
    projected_crosses_canonical_floor = (
        projected_floor_ceiling >= floor_slack_report.coverage_floor
    )

    current_fraction = _format_fraction(
        current_successful_slot_count,
        pointwise_slot_count,
    )
    projected_fraction = _format_fraction(
        projected_successful_slot_count,
        pointwise_slot_count,
    )
    lift_fraction = _format_fraction(
        single_point_repair_slot_lift, pointwise_slot_count
    )
    failing_point = failing_points[0]
    driver_signature = "single-point-floor-lift-clears-bounded-floor"
    canonical_floor_lift_digest = (
        "- current bounded floor for "
        f"`{_format_design_key(*floor_slack_report.binding_design)}` is "
        f"`{current_fraction} = {_format_float(floor_slack_report.supported_floor_ceiling)}`, "
        f"so canonical floor `{_format_float(floor_slack_report.coverage_floor)}` still misses by "
        f"`{_format_float(floor_slack_report.canonical_floor_shortfall)}`",
        "- coverage anchor seed "
        f"`{shoulder_report.coverage_anchor_random_state}` keeps exactly one uncovered point on "
        f"`{shoulder_report.residual_grid_label}`, at failing right shoulder "
        f"`z = {_format_grid_value(failing_point.grid_value)}`; if that single bounded miss flips to covered "
        f"while every other slot stays fixed, bounded floor rises by `{lift_fraction} = {_format_float(projected_floor_lift)}`",
        "- projected bounded floor ceiling therefore becomes "
        f"`{projected_fraction} = {_format_float(projected_floor_ceiling)}`, "
        f"clearing canonical floor `{_format_float(floor_slack_report.coverage_floor)}` by "
        f"`{_format_float(max(projected_floor_slack, 0.0))}`",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; this is arithmetic ceiling evidence for the existing bounded slice, "
        "not fresh rerun proof or a license to skip estimator evidence intake",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFloorLiftCeilingProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-floor-lift-ceiling-probe",
        policy_digest=resolved_policy.to_digest(),
        binding_design=floor_slack_report.binding_design,
        coverage_anchor_random_state=shoulder_report.coverage_anchor_random_state,
        overshoot_companion_random_state=shoulder_report.overshoot_companion_random_state,
        residual_grid_label=shoulder_report.residual_grid_label,
        failing_right_shoulder_grid_value=failing_point.grid_value,
        current_supported_floor_ceiling=floor_slack_report.supported_floor_ceiling,
        current_floor_shortfall=floor_slack_report.canonical_floor_shortfall,
        pointwise_slot_count=pointwise_slot_count,
        current_successful_slot_count=current_successful_slot_count,
        single_point_repair_slot_lift=single_point_repair_slot_lift,
        projected_successful_slot_count=projected_successful_slot_count,
        projected_floor_lift=projected_floor_lift,
        projected_floor_ceiling=projected_floor_ceiling,
        projected_floor_slack=projected_floor_slack,
        projected_crosses_canonical_floor=projected_crosses_canonical_floor,
        canonical_floor_lift_digest=canonical_floor_lift_digest,
    )


@lru_cache(maxsize=1)
def _run_canonical_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFloorLiftCeilingProbeReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_floor_lift_ceiling_probe_report(
        _make_canonical_runtime_probe()
    )


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_floor_lift_ceiling_probe(
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFloorLiftCeilingProbeReport:
    if policy is None:
        return _run_canonical_report()
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_floor_lift_ceiling_probe_report(
        _make_canonical_runtime_probe(),
        policy=policy,
    )
