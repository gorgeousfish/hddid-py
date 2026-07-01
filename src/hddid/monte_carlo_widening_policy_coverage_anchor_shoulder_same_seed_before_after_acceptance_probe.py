from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)

_POINTWISE_MISS_REDUCTION_PER_WITNESS_QUOTA = 2
_WITNESS_TOTAL_SLOTS = 9


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _same_seed_tuple(
    report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
) -> tuple[int, ...]:
    return tuple(observation.random_state for observation in report.seed_observations)


def _total_point_miss(point_miss_vector: tuple[int, ...]) -> int:
    return int(sum(int(value) for value in point_miss_vector))


def _implied_witness_floor(
    *,
    baseline_witness_floor: float,
    baseline_point_miss_vector: tuple[int, ...],
    candidate_point_miss_vector: tuple[int, ...],
) -> float:
    observed_point_miss_reduction = _total_point_miss(
        baseline_point_miss_vector
    ) - _total_point_miss(candidate_point_miss_vector)
    recovered_witness_slots = max(
        0,
        observed_point_miss_reduction // _POINTWISE_MISS_REDUCTION_PER_WITNESS_QUOTA,
    )
    return min(
        1.0,
        float(baseline_witness_floor)
        + float(recovered_witness_slots) / float(_WITNESS_TOTAL_SLOTS),
    )


def _driver_signature(*, acceptance_passed: bool) -> str:
    if acceptance_passed:
        return "same-seed-before-after-acceptance-satisfied"
    return "same-seed-before-after-acceptance-not-yet-satisfied"


def _require_seed_consistent_coverage_summaries(
    report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    *,
    report_label: str,
) -> None:
    rebuilt_report = build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report(
        seed_observations=report.seed_observations,
    )
    if any(
        summary.group_label != rebuilt_summary.group_label
        or summary.n_random_states != rebuilt_summary.n_random_states
        or summary.point_miss_count_by_z != rebuilt_summary.point_miss_count_by_z
        or summary.uniform_band_miss_count_by_z
        != rebuilt_summary.uniform_band_miss_count_by_z
        or summary.max_uniform_critical_value_random_state
        != rebuilt_summary.max_uniform_critical_value_random_state
        or not all(
            isclose(value, rebuilt_value, rel_tol=0.0, abs_tol=1e-12)
            for value, rebuilt_value in zip(
                summary.point_cover_rate_by_z,
                rebuilt_summary.point_cover_rate_by_z,
            )
        )
        or not all(
            isclose(value, rebuilt_value, rel_tol=0.0, abs_tol=1e-12)
            for value, rebuilt_value in zip(
                summary.uniform_band_cover_rate_by_z,
                rebuilt_summary.uniform_band_cover_rate_by_z,
            )
        )
        or not all(
            isclose(value, rebuilt_value, rel_tol=0.0, abs_tol=1e-12)
            for value, rebuilt_value in zip(
                summary.mean_sigma_z_hat_by_z,
                rebuilt_summary.mean_sigma_z_hat_by_z,
            )
        )
        or not isclose(
            summary.max_uniform_critical_value,
            rebuilt_summary.max_uniform_critical_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for summary, rebuilt_summary in zip(
            report.group_summaries,
            rebuilt_report.group_summaries,
        )
    ):
        raise ValueError(
            "same-seed before/after acceptance probe requires seed-consistent coverage summaries "
            f"for the {report_label} report"
        )
    if (
        report.canonical_seedwise_decomposition_digest
        != rebuilt_report.canonical_seedwise_decomposition_digest
    ):
        raise ValueError(
            "same-seed before/after acceptance probe requires a seed-consistent canonical digest "
            f"for the {report_label} report"
        )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    baseline_point_miss_vector: tuple[int, int, int]
    candidate_point_miss_vector: tuple[int, int, int]
    baseline_band_miss_vector: tuple[int, int, int]
    candidate_band_miss_vector: tuple[int, int, int]
    baseline_witness_floor: float
    candidate_witness_floor: float
    required_min_witness_floor: float
    same_seed_identity_holds: bool
    witness_floor_gap: float
    floor_lift_observed: bool
    left_guard_band_preserved: bool
    left_guard_pointwise_nonregression: bool
    pointwise_total_miss_reduced: bool
    right_center_lane_residual_only: bool
    acceptance_passed: bool
    driver_signature: str
    canonical_same_seed_before_after_acceptance_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.same_seed_random_states = tuple(
            int(value) for value in self.same_seed_random_states
        )
        self.baseline_point_miss_vector = tuple(
            int(value) for value in self.baseline_point_miss_vector
        )
        self.candidate_point_miss_vector = tuple(
            int(value) for value in self.candidate_point_miss_vector
        )
        self.baseline_band_miss_vector = tuple(
            int(value) for value in self.baseline_band_miss_vector
        )
        self.candidate_band_miss_vector = tuple(
            int(value) for value in self.candidate_band_miss_vector
        )
        self.baseline_witness_floor = float(self.baseline_witness_floor)
        self.candidate_witness_floor = float(self.candidate_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.same_seed_identity_holds = bool(self.same_seed_identity_holds)
        self.witness_floor_gap = float(self.witness_floor_gap)
        self.floor_lift_observed = bool(self.floor_lift_observed)
        self.left_guard_band_preserved = bool(self.left_guard_band_preserved)
        self.left_guard_pointwise_nonregression = bool(
            self.left_guard_pointwise_nonregression
        )
        self.pointwise_total_miss_reduced = bool(self.pointwise_total_miss_reduced)
        self.right_center_lane_residual_only = bool(
            self.right_center_lane_residual_only
        )
        self.acceptance_passed = bool(self.acceptance_passed)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_same_seed_before_after_acceptance_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_same_seed_before_after_acceptance_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report(
    *,
    before_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
    ),
    after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
    ),
    admission_order_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport:
    if admission_order_report is None:
        admission_order_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order()

    _require_seed_consistent_coverage_summaries(before_report, report_label="before")
    _require_seed_consistent_coverage_summaries(after_report, report_label="after")

    if before_report.policy_digest != after_report.policy_digest:
        raise ValueError(
            "same-seed before/after acceptance probe requires shared policy digest"
        )
    if before_report.binding_design != after_report.binding_design:
        raise ValueError(
            "same-seed before/after acceptance probe requires shared binding design"
        )
    if before_report.window_label != after_report.window_label:
        raise ValueError(
            "same-seed before/after acceptance probe requires shared window label"
        )
    if before_report.policy_digest != admission_order_report.policy_digest:
        raise ValueError(
            "same-seed before/after acceptance probe requires admission-order policy digest"
        )
    if before_report.binding_design != admission_order_report.binding_design:
        raise ValueError(
            "same-seed before/after acceptance probe requires admission-order binding design"
        )
    if before_report.window_label != admission_order_report.window_label:
        raise ValueError(
            "same-seed before/after acceptance probe requires admission-order window label"
        )

    before_all = before_report.group_summary("all")
    after_all = after_report.group_summary("all")

    baseline_point_miss_vector = before_all.point_miss_count_by_z
    candidate_point_miss_vector = after_all.point_miss_count_by_z
    baseline_band_miss_vector = before_all.uniform_band_miss_count_by_z
    candidate_band_miss_vector = after_all.uniform_band_miss_count_by_z
    baseline_witness_floor = admission_order_report.baseline_witness_floor_ceiling
    required_min_witness_floor = admission_order_report.projected_floor_ceiling
    candidate_witness_floor = _implied_witness_floor(
        baseline_witness_floor=baseline_witness_floor,
        baseline_point_miss_vector=baseline_point_miss_vector,
        candidate_point_miss_vector=candidate_point_miss_vector,
    )
    same_seed_identity_holds = bool(
        _same_seed_tuple(before_report) == admission_order_report.same_seed_random_states
        and _same_seed_tuple(after_report) == admission_order_report.same_seed_random_states
    )
    witness_floor_gap = max(
        0.0,
        float(required_min_witness_floor) - float(candidate_witness_floor),
    )
    floor_lift_observed = bool(candidate_witness_floor >= required_min_witness_floor)
    left_guard_band_preserved = bool(candidate_band_miss_vector[0] == 0)
    left_guard_pointwise_nonregression = bool(
        candidate_point_miss_vector[0] <= baseline_point_miss_vector[0]
    )
    pointwise_total_miss_reduced = bool(
        _total_point_miss(candidate_point_miss_vector)
        < _total_point_miss(baseline_point_miss_vector)
    )
    right_center_lane_residual_only = bool(
        candidate_band_miss_vector[0] == 0
        and candidate_point_miss_vector[0] <= baseline_point_miss_vector[0]
    )
    acceptance_passed = bool(
        same_seed_identity_holds
        and floor_lift_observed
        and left_guard_band_preserved
        and left_guard_pointwise_nonregression
        and pointwise_total_miss_reduced
        and right_center_lane_residual_only
    )
    driver_signature = _driver_signature(acceptance_passed=acceptance_passed)
    if not floor_lift_observed and candidate_witness_floor == baseline_witness_floor:
        witness_floor_line = (
            "- witness-floor requirement remains unmet: baseline and candidate both stay "
            f"`7/9 = {_format_float(baseline_witness_floor)}`, below the required "
            f"`8/9 = {_format_float(required_min_witness_floor)}` threshold for "
            "promotion-informative evidence"
        )
    else:
        witness_floor_line = (
            f"- witness-floor requirement {'is met' if floor_lift_observed else 'remains unmet'}: "
            f"baseline stays `7/9 = {_format_float(baseline_witness_floor)}`, "
            f"candidate maps to `{_format_float(candidate_witness_floor)}`, and "
            f"required threshold remains `8/9 = {_format_float(required_min_witness_floor)}`"
        )
    canonical_digest = (
        f"- exact same-seed replay identity check {'passes' if acceptance_passed else 'fails'} promotion by construction: candidate remains at baseline point miss `{list(candidate_point_miss_vector) if not pointwise_total_miss_reduced else list(candidate_point_miss_vector)}`, so total point misses stay `{_total_point_miss(candidate_point_miss_vector)}` and {'floor-lift evidence is observed' if floor_lift_observed else 'no floor-lift evidence is observed'}",
        witness_floor_line,
        f"- left guard is {'preserved' if left_guard_band_preserved and left_guard_pointwise_nonregression else 'not preserved'} but {'sufficient' if acceptance_passed else 'insufficient'} alone: left `z = 0.05` band miss remains `{candidate_band_miss_vector[0]}` and pointwise miss {'does not increase' if left_guard_pointwise_nonregression else 'increases'}, yet Trigger 2 acceptance {'passes' if acceptance_passed else 'still fails'} without witness-floor lift plus total miss reduction",
        f"- current Trigger 2 implication: `{driver_signature}`; next execution must show observed same-seed floor lift rather than fresh-only rerun volume",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-before-after-acceptance-probe"
        ),
        policy_digest=before_report.policy_digest,
        binding_design=before_report.binding_design,
        window_label=before_report.window_label,
        same_seed_random_states=admission_order_report.same_seed_random_states,
        baseline_point_miss_vector=baseline_point_miss_vector,
        candidate_point_miss_vector=candidate_point_miss_vector,
        baseline_band_miss_vector=baseline_band_miss_vector,
        candidate_band_miss_vector=candidate_band_miss_vector,
        baseline_witness_floor=baseline_witness_floor,
        candidate_witness_floor=candidate_witness_floor,
        required_min_witness_floor=required_min_witness_floor,
        same_seed_identity_holds=same_seed_identity_holds,
        witness_floor_gap=witness_floor_gap,
        floor_lift_observed=floor_lift_observed,
        left_guard_band_preserved=left_guard_band_preserved,
        left_guard_pointwise_nonregression=left_guard_pointwise_nonregression,
        pointwise_total_miss_reduced=pointwise_total_miss_reduced,
        right_center_lane_residual_only=right_center_lane_residual_only,
        acceptance_passed=acceptance_passed,
        driver_signature=driver_signature,
        canonical_same_seed_before_after_acceptance_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport
):
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report(
        before_report=seedwise_report,
        after_report=seedwise_report,
        admission_order_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order(),
    )
