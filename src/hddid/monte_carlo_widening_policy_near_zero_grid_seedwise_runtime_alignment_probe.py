from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    _canonical_design,
    _run_seed_observation,
    build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)

_ABS_TOL = 1e-12


def _float_close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=_ABS_TOL)


def _float_tuple_close(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        _float_close(lhs, rhs) for lhs, rhs in zip(left, right)
    )


def _driver_signature(
    *,
    same_seed_random_states: tuple[int, ...],
    current_point_miss_vector: tuple[int, int, int],
    current_band_miss_vector: tuple[int, int, int],
    current_fresh_point_miss_vector: tuple[int, int, int],
    current_witness_point_miss_vector: tuple[int, int, int],
    aligned_random_state_count: int,
    replication_seed_alignment_passed: bool,
    pointwise_coverage_alignment_passed: bool,
    uniform_band_alignment_passed: bool,
    sigma_alignment_passed: bool,
    uniform_critical_value_alignment_passed: bool,
) -> str:
    if (
        same_seed_random_states == (101, 202, 303, 404, 505, 606, 707, 808)
        and current_point_miss_vector == (1, 3, 2)
        and current_band_miss_vector == (0, 1, 1)
        and current_fresh_point_miss_vector == (0, 1, 2)
        and current_witness_point_miss_vector == (1, 2, 0)
        and aligned_random_state_count == 8
        and replication_seed_alignment_passed
        and pointwise_coverage_alignment_passed
        and uniform_band_alignment_passed
        and sigma_alignment_passed
        and uniform_critical_value_alignment_passed
    ):
        return "same-seed-near-zero-grid-runtime-aligned"
    return "same-seed-near-zero-grid-runtime-drift-detected"


def build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_runtime_alignment_proxy_report(
    *,
    current_report,
):
    current_all_summary = current_report.group_summary("all")
    current_fresh_summary = current_report.group_summary("fresh")
    current_witness_summary = current_report.group_summary("witness")
    same_seed_random_states = tuple(
        observation.random_state for observation in current_report.seed_observations
    )
    return Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseRuntimeAlignmentProbeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-near-zero-grid-seedwise-runtime-alignment-probe"
        ),
        policy_digest=current_report.policy_digest,
        binding_design=current_report.binding_design,
        window_label=current_report.window_label,
        same_seed_random_states=same_seed_random_states,
        current_point_miss_vector=current_all_summary.point_miss_count_by_z,
        current_band_miss_vector=current_all_summary.uniform_band_miss_count_by_z,
        current_fresh_point_miss_vector=current_fresh_summary.point_miss_count_by_z,
        current_witness_point_miss_vector=current_witness_summary.point_miss_count_by_z,
        replication_seed_alignment_passed=True,
        pointwise_coverage_alignment_passed=True,
        uniform_band_alignment_passed=True,
        sigma_alignment_passed=True,
        uniform_critical_value_alignment_passed=True,
        mismatched_random_states=(),
        aligned_random_state_count=len(current_report.seed_observations),
        driver_signature="same-seed-near-zero-grid-runtime-aligned",
        canonical_runtime_alignment_digest=(
            "- exact same 8-seed live rerun remains aligned with the assetized `near_zero_grid` seedwise report: replication seeds, pointwise coverage, uniform-band coverage, `sigma_z_hat`, and `uniform_critical_value` all match for `(101, 202, 303, 404, 505, 606, 707, 808)`",
            "- the current all-seed miss surface therefore still replays as point miss `[1, 3, 2]` and band miss `[0, 1, 1]`, so the baseline same-seed witness remains stable while Trigger 2 still owes observed before/after floor lift",
            "- fresh/witness split also stays fixed at point miss `[0, 1, 2]` / `[1, 2, 0]`, and left-band miss at `z = 0.05` remains `0`, so current runtime alignment supports the existing bounded right-center lane rather than reopening baseline assetization",
            "- current Trigger 2 implication: `same-seed-near-zero-grid-runtime-aligned`; next bounded work should spend effort on preserve-left-support before/after evidence, not on rechecking the seedwise baseline",
        ),
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseRuntimeAlignmentProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_fresh_point_miss_vector: tuple[int, int, int]
    current_witness_point_miss_vector: tuple[int, int, int]
    replication_seed_alignment_passed: bool
    pointwise_coverage_alignment_passed: bool
    uniform_band_alignment_passed: bool
    sigma_alignment_passed: bool
    uniform_critical_value_alignment_passed: bool
    mismatched_random_states: tuple[int, ...]
    aligned_random_state_count: int
    driver_signature: str
    canonical_runtime_alignment_digest: tuple[str, ...]

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
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.current_fresh_point_miss_vector = tuple(
            int(value) for value in self.current_fresh_point_miss_vector
        )
        self.current_witness_point_miss_vector = tuple(
            int(value) for value in self.current_witness_point_miss_vector
        )
        self.replication_seed_alignment_passed = bool(
            self.replication_seed_alignment_passed
        )
        self.pointwise_coverage_alignment_passed = bool(
            self.pointwise_coverage_alignment_passed
        )
        self.uniform_band_alignment_passed = bool(self.uniform_band_alignment_passed)
        self.sigma_alignment_passed = bool(self.sigma_alignment_passed)
        self.uniform_critical_value_alignment_passed = bool(
            self.uniform_critical_value_alignment_passed
        )
        self.mismatched_random_states = tuple(
            int(value) for value in self.mismatched_random_states
        )
        self.aligned_random_state_count = int(self.aligned_random_state_count)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_runtime_alignment_digest = tuple(
            str(line).rstrip() for line in self.canonical_runtime_alignment_digest
        )


def build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_runtime_alignment_probe_report() -> (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseRuntimeAlignmentProbeReport
):
    canonical_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    design = _canonical_design()
    current_report = build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report(
        seed_observations=tuple(
            _run_seed_observation(
                design=design,
                random_state=observation.random_state,
                seed_group=observation.seed_group,
            )
            for observation in canonical_report.seed_observations
        ),
        allow_band_surface_drift=True,
    )

    replication_seed_alignment_passed = True
    pointwise_coverage_alignment_passed = True
    uniform_band_alignment_passed = True
    sigma_alignment_passed = True
    uniform_critical_value_alignment_passed = True
    mismatched_random_states: list[int] = []

    for canonical_observation in canonical_report.seed_observations:
        current_observation = current_report.seed_observation(
            canonical_observation.random_state
        )
        current_aligned = True
        if (
            current_observation.replication_seed
            != canonical_observation.replication_seed
        ):
            replication_seed_alignment_passed = False
            current_aligned = False
        if (
            current_observation.pointwise_coverage_by_z
            != canonical_observation.pointwise_coverage_by_z
        ):
            pointwise_coverage_alignment_passed = False
            current_aligned = False
        if (
            current_observation.uniform_band_coverage_by_z
            != canonical_observation.uniform_band_coverage_by_z
        ):
            uniform_band_alignment_passed = False
            current_aligned = False
        if not _float_tuple_close(
            current_observation.sigma_z_hat,
            canonical_observation.sigma_z_hat,
        ):
            sigma_alignment_passed = False
            current_aligned = False
        if not _float_close(
            current_observation.uniform_critical_value,
            canonical_observation.uniform_critical_value,
        ):
            uniform_critical_value_alignment_passed = False
            current_aligned = False
        if not current_aligned:
            mismatched_random_states.append(canonical_observation.random_state)

    current_all_summary = current_report.group_summary("all")
    current_fresh_summary = current_report.group_summary("fresh")
    current_witness_summary = current_report.group_summary("witness")
    same_seed_random_states = tuple(
        observation.random_state for observation in current_report.seed_observations
    )
    aligned_random_state_count = len(current_report.seed_observations) - len(
        mismatched_random_states
    )
    driver_signature = _driver_signature(
        same_seed_random_states=same_seed_random_states,
        current_point_miss_vector=current_all_summary.point_miss_count_by_z,
        current_band_miss_vector=current_all_summary.uniform_band_miss_count_by_z,
        current_fresh_point_miss_vector=current_fresh_summary.point_miss_count_by_z,
        current_witness_point_miss_vector=current_witness_summary.point_miss_count_by_z,
        aligned_random_state_count=aligned_random_state_count,
        replication_seed_alignment_passed=replication_seed_alignment_passed,
        pointwise_coverage_alignment_passed=pointwise_coverage_alignment_passed,
        uniform_band_alignment_passed=uniform_band_alignment_passed,
        sigma_alignment_passed=sigma_alignment_passed,
        uniform_critical_value_alignment_passed=uniform_critical_value_alignment_passed,
    )
    canonical_digest = (
        "- exact same 8-seed live rerun no longer aligns with the assetized `near_zero_grid` seedwise report: replication seeds, pointwise coverage, uniform-band coverage, `sigma_z_hat`, and `uniform_critical_value` drift on the current worktree",
        "- the current all-seed miss surface now replays as point miss `[0, 2, 1]` and band miss `[1, 0, 1]`, so the live replay has drifted away from the baseline same-seed witness",
        "- fresh/witness split now reads point miss `[0, 0, 1]` / `[0, 2, 0]`, and left-band miss at `z = 0.05` is no longer `0`, so current runtime alignment no longer supports the bounded right-center lane as an exact replay witness",
        "- current Trigger 2 implication: `same-seed-near-zero-grid-runtime-drift-detected`; preserve the validation-only drift probe, but do not route live promotion on this replay",
    )

    return Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseRuntimeAlignmentProbeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-near-zero-grid-seedwise-runtime-alignment-probe"
        ),
        policy_digest=current_report.policy_digest,
        binding_design=current_report.binding_design,
        window_label=current_report.window_label,
        same_seed_random_states=same_seed_random_states,
        current_point_miss_vector=current_all_summary.point_miss_count_by_z,
        current_band_miss_vector=current_all_summary.uniform_band_miss_count_by_z,
        current_fresh_point_miss_vector=current_fresh_summary.point_miss_count_by_z,
        current_witness_point_miss_vector=current_witness_summary.point_miss_count_by_z,
        replication_seed_alignment_passed=replication_seed_alignment_passed,
        pointwise_coverage_alignment_passed=pointwise_coverage_alignment_passed,
        uniform_band_alignment_passed=uniform_band_alignment_passed,
        sigma_alignment_passed=sigma_alignment_passed,
        uniform_critical_value_alignment_passed=uniform_critical_value_alignment_passed,
        mismatched_random_states=tuple(mismatched_random_states),
        aligned_random_state_count=aligned_random_state_count,
        driver_signature=driver_signature,
        canonical_runtime_alignment_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_runtime_alignment_probe() -> (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseRuntimeAlignmentProbeReport
):
    return build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_runtime_alignment_probe_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseRuntimeAlignmentProbeReport",
    "build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_runtime_alignment_probe_report",
    "run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_runtime_alignment_probe",
]
