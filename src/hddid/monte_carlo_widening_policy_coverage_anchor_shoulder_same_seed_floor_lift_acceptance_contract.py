from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_floor_lift_ceiling_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFloorLiftCeilingProbeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_floor_lift_ceiling_probe,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)


_FRESH_RANDOM_STATES = (404, 505, 606, 707, 808)
_LEFT_GUARD_GRID_VALUE = 0.05
_CENTER_GRID_VALUE = 0.15
_RIGHT_SHOULDER_GRID_VALUE = 0.25
_POINTWISE_MISS_REDUCTION_PER_WITNESS_LIFT = 2


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _driver_signature(
    *,
    floor_lift_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFloorLiftCeilingProbeReport
    ),
    acceptance_contract_holds: bool,
) -> str:
    if (
        floor_lift_report.projected_crosses_canonical_floor
        and floor_lift_report.single_point_repair_slot_lift == 1
        and acceptance_contract_holds
    ):
        return "same-seed-floor-lift-acceptance-contract"
    return "mixed-same-seed-floor-lift-acceptance-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedFloorLiftAcceptanceContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    witness_random_states: tuple[int, ...]
    fresh_random_states: tuple[int, ...]
    all_random_states: tuple[int, ...]
    required_additional_witnesses_for_floor: int
    minimum_total_point_miss_reduction_for_floor: int
    baseline_witness_floor_ceiling: float
    projected_floor_ceiling: float
    projected_floor_slack: float
    canonical_floor: float
    all_seed_point_miss_vector: tuple[int, int, int]
    all_seed_band_miss_vector: tuple[int, int, int]
    left_guard_grid_value: float
    center_grid_value: float
    right_shoulder_grid_value: float
    left_point_miss_count: int
    left_band_miss_count: int
    acceptance_conditions: tuple[str, ...]
    acceptance_contract_holds: bool
    driver_signature: str
    canonical_same_seed_floor_lift_acceptance_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.witness_random_states = tuple(
            int(value) for value in self.witness_random_states
        )
        self.fresh_random_states = tuple(
            int(value) for value in self.fresh_random_states
        )
        self.all_random_states = tuple(int(value) for value in self.all_random_states)
        self.required_additional_witnesses_for_floor = int(
            self.required_additional_witnesses_for_floor
        )
        self.minimum_total_point_miss_reduction_for_floor = int(
            self.minimum_total_point_miss_reduction_for_floor
        )
        self.baseline_witness_floor_ceiling = float(self.baseline_witness_floor_ceiling)
        self.projected_floor_ceiling = float(self.projected_floor_ceiling)
        self.projected_floor_slack = float(self.projected_floor_slack)
        self.canonical_floor = float(self.canonical_floor)
        self.all_seed_point_miss_vector = tuple(
            int(value) for value in self.all_seed_point_miss_vector
        )
        self.all_seed_band_miss_vector = tuple(
            int(value) for value in self.all_seed_band_miss_vector
        )
        self.left_guard_grid_value = float(self.left_guard_grid_value)
        self.center_grid_value = float(self.center_grid_value)
        self.right_shoulder_grid_value = float(self.right_shoulder_grid_value)
        self.left_point_miss_count = int(self.left_point_miss_count)
        self.left_band_miss_count = int(self.left_band_miss_count)
        self.acceptance_conditions = tuple(
            str(item).strip() for item in self.acceptance_conditions
        )
        self.acceptance_contract_holds = bool(self.acceptance_contract_holds)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_same_seed_floor_lift_acceptance_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_same_seed_floor_lift_acceptance_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_floor_lift_acceptance_contract_report(
    *,
    floor_lift_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFloorLiftCeilingProbeReport
    ),
    seedwise_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedFloorLiftAcceptanceContractReport:
    if floor_lift_report.binding_design != seedwise_report.binding_design:
        raise ValueError(
            "same-seed floor-lift acceptance contract requires shared binding design"
        )
    if floor_lift_report.residual_grid_label != seedwise_report.window_label:
        raise ValueError(
            "same-seed floor-lift acceptance contract requires shared window label"
        )

    all_summary = seedwise_report.group_summary("all")
    witness_random_states = tuple(seedwise_report.witness_random_states)
    all_random_states = witness_random_states + _FRESH_RANDOM_STATES
    minimum_total_point_miss_reduction_for_floor = (
        int(floor_lift_report.single_point_repair_slot_lift)
        * _POINTWISE_MISS_REDUCTION_PER_WITNESS_LIFT
    )
    acceptance_conditions = (
        "compare the repaired estimator path against the exact same 8 random states `(101, 202, 303, 404, 505, 606, 707, 808)` on `near_zero_grid`",
        "reduce total pointwise misses by at least `2` so the same-seed replay can map the bounded witness floor from `7/9` to at least `8/9`",
        "do not increase pointwise misses at left guard `z = 0.05`, and keep left-band misses at `0`",
        "keep any remaining misses localized to the bounded right-center lane `z in {0.15, 0.25}` rather than creating new off-lane band regressions",
    )
    acceptance_contract_holds = bool(
        floor_lift_report.single_point_repair_slot_lift == 1
        and minimum_total_point_miss_reduction_for_floor == 2
        and all_summary.point_miss_count_by_z[0] == 1
        and all_summary.uniform_band_miss_count_by_z[0] == 0
    )
    driver_signature = _driver_signature(
        floor_lift_report=floor_lift_report,
        acceptance_contract_holds=acceptance_contract_holds,
    )
    canonical_digest = (
        "- future estimator evidence must replay the exact same `8` random states `(101, 202, 303, 404, 505, 606, 707, 808)` on `DGP2/500/50` with `near_zero_grid`, because the current all-seed baseline miss vectors are pointwise "
        f"`{list(all_summary.point_miss_count_by_z)}` and band `{list(all_summary.uniform_band_miss_count_by_z)}` at `z = (0.05, 0.15, 0.25)`",
        "- witness-floor arithmetic remains the promotion gate: current bounded floor stays "
        f"`7/9 = {_format_float(floor_lift_report.current_supported_floor_ceiling)}`, "
        f"one additional witness would project the ceiling to `8/9 = {_format_float(floor_lift_report.projected_floor_ceiling)}`, and the same-seed replay therefore needs at least "
        f"`{minimum_total_point_miss_reduction_for_floor}` fewer point misses to count as floor-lift evidence above canonical floor "
        f"`{_format_float(floor_lift_report.current_supported_floor_ceiling + floor_lift_report.current_floor_shortfall)}`",
        "- left-side support is currently the hard no-regression guard: `z = 0.05` has only "
        f"`{all_summary.point_miss_count_by_z[0]}` pointwise miss and `{all_summary.uniform_band_miss_count_by_z[0]}` band misses across all `8` seeds, so future evidence cannot spend the floor lift by introducing any new left-band failure or extra left pointwise miss",
        "- current Trigger 2 implication: `same-seed-floor-lift-acceptance-contract`; treat future estimator reruns as promotion-informative only if they satisfy exact same-seed replay, reduce total point misses by at least "
        f"`{minimum_total_point_miss_reduction_for_floor}`, and keep remaining miss pressure inside the bounded right-center lane",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedFloorLiftAcceptanceContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-floor-lift-acceptance-contract"
        ),
        policy_digest=floor_lift_report.policy_digest,
        binding_design=floor_lift_report.binding_design,
        window_label="near_zero_grid",
        witness_random_states=witness_random_states,
        fresh_random_states=_FRESH_RANDOM_STATES,
        all_random_states=all_random_states,
        required_additional_witnesses_for_floor=floor_lift_report.single_point_repair_slot_lift,
        minimum_total_point_miss_reduction_for_floor=(
            minimum_total_point_miss_reduction_for_floor
        ),
        baseline_witness_floor_ceiling=floor_lift_report.current_supported_floor_ceiling,
        projected_floor_ceiling=floor_lift_report.projected_floor_ceiling,
        projected_floor_slack=floor_lift_report.projected_floor_slack,
        canonical_floor=floor_lift_report.current_supported_floor_ceiling
        + floor_lift_report.current_floor_shortfall,
        all_seed_point_miss_vector=all_summary.point_miss_count_by_z,
        all_seed_band_miss_vector=all_summary.uniform_band_miss_count_by_z,
        left_guard_grid_value=_LEFT_GUARD_GRID_VALUE,
        center_grid_value=_CENTER_GRID_VALUE,
        right_shoulder_grid_value=_RIGHT_SHOULDER_GRID_VALUE,
        left_point_miss_count=all_summary.point_miss_count_by_z[0],
        left_band_miss_count=all_summary.uniform_band_miss_count_by_z[0],
        acceptance_conditions=acceptance_conditions,
        acceptance_contract_holds=acceptance_contract_holds,
        driver_signature=driver_signature,
        canonical_same_seed_floor_lift_acceptance_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_floor_lift_acceptance_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedFloorLiftAcceptanceContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_floor_lift_acceptance_contract_report(
        floor_lift_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_floor_lift_ceiling_probe(),
        seedwise_report=run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition(),
    )
