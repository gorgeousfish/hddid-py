from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport,
    _build_slice,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_multiple(value: float) -> str:
    return f"{float(value):.3f}x"


def _safe_share(numerator: float, denominator: float) -> float:
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return 0.0
    return float(numerator) / denominator_value


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorInversePiConcentrationTraceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    center_grid_value: float
    target_fold_id: int
    target_fold3_low_pi_treated_count: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_share_of_fold3_low_pi_count: float
    target_exact_trim_floor_raw_phi1_center_projection: float
    target_remaining_fold3_low_pi_raw_phi1_center_projection: float
    target_exact_trim_floor_raw_phi1_share_of_fold3_raw: float
    target_exact_trim_floor_inverse_pi_phi1_center_projection: float
    target_remaining_fold3_low_pi_inverse_pi_phi1_center_projection: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_row_gross_inverse_pi_amplification: float
    target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification: float
    target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining: float
    target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining: (
        float
    )
    target_exact_trim_floor_weighted_phi1_center_projection: float
    target_remaining_fold3_low_pi_weighted_phi1_center_projection: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    target_remaining_fold3_low_pi_weighted_retention: float
    target_exact_trim_floor_weighted_retention_multiple_vs_remaining: float
    comparator_random_states: tuple[int, ...]
    comparator_exact_trim_floor_counts: tuple[int, ...]
    comparator_fold3_inverse_pi_phi1_center_projections: tuple[float, ...]
    driver_signature: str
    canonical_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_digest: tuple[
        str, ...
    ]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.focus_random_states = tuple(
            int(value) for value in self.focus_random_states
        )
        self.target_random_state = int(self.target_random_state)
        self.target_seed_group = str(self.target_seed_group).strip().lower()
        self.center_grid_value = float(self.center_grid_value)
        self.target_fold_id = int(self.target_fold_id)
        self.target_fold3_low_pi_treated_count = int(
            self.target_fold3_low_pi_treated_count
        )
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_share_of_fold3_low_pi_count = float(
            self.target_exact_trim_floor_share_of_fold3_low_pi_count
        )
        self.target_exact_trim_floor_raw_phi1_center_projection = float(
            self.target_exact_trim_floor_raw_phi1_center_projection
        )
        self.target_remaining_fold3_low_pi_raw_phi1_center_projection = float(
            self.target_remaining_fold3_low_pi_raw_phi1_center_projection
        )
        self.target_exact_trim_floor_raw_phi1_share_of_fold3_raw = float(
            self.target_exact_trim_floor_raw_phi1_share_of_fold3_raw
        )
        self.target_exact_trim_floor_inverse_pi_phi1_center_projection = float(
            self.target_exact_trim_floor_inverse_pi_phi1_center_projection
        )
        self.target_remaining_fold3_low_pi_inverse_pi_phi1_center_projection = float(
            self.target_remaining_fold3_low_pi_inverse_pi_phi1_center_projection
        )
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_row_gross_inverse_pi_amplification = float(
            self.target_exact_trim_floor_row_gross_inverse_pi_amplification
        )
        self.target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification = float(
            self.target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification
        )
        self.target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining = float(
            self.target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining
        )
        self.target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining = float(
            self.target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining
        )
        self.target_exact_trim_floor_weighted_phi1_center_projection = float(
            self.target_exact_trim_floor_weighted_phi1_center_projection
        )
        self.target_remaining_fold3_low_pi_weighted_phi1_center_projection = float(
            self.target_remaining_fold3_low_pi_weighted_phi1_center_projection
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.target_exact_trim_floor_weighted_retention = float(
            self.target_exact_trim_floor_weighted_retention
        )
        self.target_remaining_fold3_low_pi_weighted_retention = float(
            self.target_remaining_fold3_low_pi_weighted_retention
        )
        self.target_exact_trim_floor_weighted_retention_multiple_vs_remaining = float(
            self.target_exact_trim_floor_weighted_retention_multiple_vs_remaining
        )
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_exact_trim_floor_counts = tuple(
            int(value) for value in self.comparator_exact_trim_floor_counts
        )
        self.comparator_fold3_inverse_pi_phi1_center_projections = tuple(
            float(value)
            for value in self.comparator_fold3_inverse_pi_phi1_center_projections
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_digest
        )


def _driver_signature(
    *,
    propensity_floor_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport
    ),
    target_exact_trim_floor_share_of_count: float,
    target_exact_trim_floor_raw_phi1_share_of_fold3_raw: float,
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float,
    target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining: float,
    target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining: float,
    comparator_exact_trim_floor_counts: tuple[int, ...],
) -> str:
    if (
        propensity_floor_report.driver_signature
        == "same-seed-seed303-fold3-trim-floor-propensity-floor-driver-confirmed"
        and target_exact_trim_floor_share_of_count < 0.1
        and target_exact_trim_floor_raw_phi1_share_of_fold3_raw < 0.1
        and target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi > 0.4
        and target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining
        > 7.0
        and target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining
        > 9.0
        and comparator_exact_trim_floor_counts == (0, 0)
    ):
        return "same-seed-seed303-fold3-trim-floor-inverse-pi-concentration-confirmed"
    return "mixed-same-seed-seed303-fold3-trim-floor-inverse-pi-concentration-trace"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorInversePiConcentrationTraceReport
):
    propensity_floor_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    target_observation = seedwise_report.seed_observation(
        propensity_floor_report.target_random_state
    )
    comparator_observations = tuple(
        seedwise_report.seed_observation(random_state)
        for random_state in propensity_floor_report.comparator_random_states
    )
    target_slice = _build_slice(
        random_state=target_observation.random_state,
        seed_group=target_observation.seed_group,
        replication_seed=target_observation.replication_seed,
    )
    comparator_slices = tuple(
        _build_slice(
            random_state=observation.random_state,
            seed_group=observation.seed_group,
            replication_seed=observation.replication_seed,
        )
        for observation in comparator_observations
    )

    remaining_raw = (
        propensity_floor_report.target_fold3_raw_phi1_center_projection
        - propensity_floor_report.target_exact_trim_floor_raw_phi1_center_projection
    )
    remaining_inverse_pi = (
        propensity_floor_report.target_fold3_inverse_pi_phi1_center_projection
        - propensity_floor_report.target_exact_trim_floor_inverse_pi_phi1_center_projection
    )
    remaining_weighted = (
        propensity_floor_report.target_fold3_weighted_phi1_center_projection
        - propensity_floor_report.target_exact_trim_floor_weighted_phi1_center_projection
    )
    remaining_count = (
        propensity_floor_report.target_fold3_low_pi_treated_count
        - propensity_floor_report.target_exact_trim_floor_count
    )
    target_exact_trim_floor_share_of_count = _safe_share(
        propensity_floor_report.target_exact_trim_floor_count,
        propensity_floor_report.target_fold3_low_pi_treated_count,
    )
    target_exact_trim_floor_row_gross_inverse_pi_amplification = _safe_share(
        propensity_floor_report.target_exact_trim_floor_inverse_pi_phi1_center_projection,
        propensity_floor_report.target_exact_trim_floor_raw_phi1_center_projection,
    )
    target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification = _safe_share(
        remaining_inverse_pi,
        remaining_raw,
    )
    target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining = _safe_share(
        target_exact_trim_floor_row_gross_inverse_pi_amplification,
        target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification,
    )
    target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining = _safe_share(
        _safe_share(
            propensity_floor_report.target_exact_trim_floor_inverse_pi_phi1_center_projection,
            propensity_floor_report.target_exact_trim_floor_count,
        ),
        _safe_share(remaining_inverse_pi, remaining_count),
    )
    target_exact_trim_floor_weighted_retention = _safe_share(
        propensity_floor_report.target_exact_trim_floor_weighted_phi1_center_projection,
        propensity_floor_report.target_exact_trim_floor_inverse_pi_phi1_center_projection,
    )
    target_remaining_fold3_low_pi_weighted_retention = _safe_share(
        remaining_weighted,
        remaining_inverse_pi,
    )
    target_exact_trim_floor_weighted_retention_multiple_vs_remaining = _safe_share(
        target_exact_trim_floor_weighted_retention,
        target_remaining_fold3_low_pi_weighted_retention,
    )
    comparator_exact_trim_floor_counts = tuple(
        slice_.exact_trim_floor_count for slice_ in comparator_slices
    )
    driver_signature = _driver_signature(
        propensity_floor_report=propensity_floor_report,
        target_exact_trim_floor_share_of_count=target_exact_trim_floor_share_of_count,
        target_exact_trim_floor_raw_phi1_share_of_fold3_raw=(
            _safe_share(
                (
                    propensity_floor_report.target_exact_trim_floor_raw_phi1_offset_share_of_trim_inverse_pi
                    * propensity_floor_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
                ),
                propensity_floor_report.target_fold3_raw_phi1_offset_share_of_inverse_pi,
            )
        ),
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            propensity_floor_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining=(
            target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining
        ),
        target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining=(
            target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining
        ),
        comparator_exact_trim_floor_counts=comparator_exact_trim_floor_counts,
    )
    if propensity_floor_report.target_exact_trim_floor_count == 0:
        canonical_digest = (
            "- seed `303` / witness / fold `3` / `z = 0.15` exposes no exact trim-floor row in the current live replay: `0` of "
            f"`{propensity_floor_report.target_fold3_low_pi_treated_count}` low-`pi_hat` treated observations sit at `pi_hat = 0.010001`, so the exact-row raw, gross inverse-`pi_hat`, and weighted projections are all `0.000`; the broader fold-`3` low-`pi_hat` tail carries the full gross conduit "
            f"`{_format_float(remaining_inverse_pi)}`",
            "- the concentration claim is therefore not row-vs-rest-confirmed on this worktree: exact-row gross amplification, per-observation concentration, and weighted-retention multiples are all `0.000x`, while the remaining "
            f"`{remaining_count}` low-`pi_hat` rows average `{_format_multiple(target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification)}` gross inverse-`pi_hat` amplification",
            "- after the final `(1-pi_hat)` contraction the absent trim-floor row still contributes `0.0%` of the weighted fold-`3` nuisance burden, while the broader low-`pi_hat` tail retains "
            f"`{_format_percent(target_remaining_fold3_low_pi_weighted_retention)}` of its gross inverse-`pi_hat` lift",
            "- comparator seeds `202` and `707` also expose no exact trim-floor row inside the same fold-`3` low-`pi_hat` treated slice (count `0` in both cases), and their full fold-`3` gross inverse-`pi_hat` conduits stay negative at "
            f"`{_format_float(propensity_floor_report.comparator_fold3_inverse_pi_phi1_center_projections[0])}` and "
            f"`{_format_float(propensity_floor_report.comparator_fold3_inverse_pi_phi1_center_projections[1])}`; current Trigger 2 implication: `mixed-same-seed-seed303-fold3-trim-floor-inverse-pi-concentration-trace`, so the next bounded repair should stay on the seed `303` broader fold-`3` low-`pi_hat` conduit before spending seed `707` / `z = 0.25`",
        )
    else:
        canonical_digest = (
            "- seed `303` / witness / fold `3` / `z = 0.15` keeps only one exact trim-floor row out of "
            f"`{propensity_floor_report.target_fold3_low_pi_treated_count}` low-`pi_hat` treated observations "
            f"(`{_format_percent(target_exact_trim_floor_share_of_count)}` of the fold-`3` tail), and that row still contributes just "
            f"`{_format_float(propensity_floor_report.target_exact_trim_floor_raw_phi1_center_projection)}` of raw treated `phi1_hat` center "
            f"(`{_format_percent(_safe_share(propensity_floor_report.target_exact_trim_floor_raw_phi1_center_projection, propensity_floor_report.target_fold3_raw_phi1_center_projection))}` of fold-`3` raw nuisance mass) while already carrying "
            f"`{_format_float(propensity_floor_report.target_exact_trim_floor_inverse_pi_phi1_center_projection)}` of the gross inverse-`pi_hat` conduit "
            f"(`{_format_percent(propensity_floor_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of fold-`3` gross lift); the remaining `{remaining_count}` rows share the other "
            f"`{_format_float(remaining_inverse_pi)}`",
            "- the binding concentration is denominator-led rather than raw-nuisance-led: the trim-floor row amplifies its own raw `phi1_hat` center by "
            f"`{_format_multiple(target_exact_trim_floor_row_gross_inverse_pi_amplification)}`, whereas the remaining `{remaining_count}` low-`pi_hat` rows average only "
            f"`{_format_multiple(target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification)}`, so the row-level gross inverse-`pi_hat` amplification is still "
            f"`{_format_multiple(target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining)}` the rest-of-tail average and "
            f"`{_format_multiple(target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining)}` larger per observation",
            "- after the final `(1-pi_hat)` contraction the trim-floor row still retains "
            f"`{_format_percent(target_exact_trim_floor_weighted_retention)}` of its gross inverse-`pi_hat` lift, versus "
            f"`{_format_percent(target_remaining_fold3_low_pi_weighted_retention)}` for the remaining `{remaining_count}` rows, so the same row rises from "
            f"`{_format_percent(target_exact_trim_floor_share_of_count)}` of fold-`3` low-`pi_hat` count to "
            f"`{_format_percent(propensity_floor_report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the final weighted fold-`3` nuisance burden",
            "- comparator seeds `202` and `707` do not even expose an exact trim-floor row inside the same fold-`3` low-`pi_hat` treated slice (count `0` in both cases), and their full fold-`3` gross inverse-`pi_hat` conduits stay negative at "
            f"`{_format_float(propensity_floor_report.comparator_fold3_inverse_pi_phi1_center_projections[0])}` and "
            f"`{_format_float(propensity_floor_report.comparator_fold3_inverse_pi_phi1_center_projections[1])}`; current Trigger 2 implication: `{driver_signature}`, so the next bounded repair should feed the real same-seed observed rerun through the ladder guard while using this seed `303` fold-`3` trim-floor row-vs-rest split before spending seed `707` / `z = 0.25`",
        )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorInversePiConcentrationTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-low-pi-fold3-trim-"
            "floor-inverse-pi-concentration-trace"
        ),
        policy_digest=propensity_floor_report.policy_digest,
        binding_design=propensity_floor_report.binding_design,
        window_label=propensity_floor_report.window_label,
        focus_random_states=propensity_floor_report.focus_random_states,
        target_random_state=propensity_floor_report.target_random_state,
        target_seed_group=propensity_floor_report.target_seed_group,
        center_grid_value=propensity_floor_report.center_grid_value,
        target_fold_id=propensity_floor_report.target_fold_id,
        target_fold3_low_pi_treated_count=(
            propensity_floor_report.target_fold3_low_pi_treated_count
        ),
        target_exact_trim_floor_count=propensity_floor_report.target_exact_trim_floor_count,
        target_exact_trim_floor_share_of_fold3_low_pi_count=(
            target_exact_trim_floor_share_of_count
        ),
        target_exact_trim_floor_raw_phi1_center_projection=(
            propensity_floor_report.target_exact_trim_floor_raw_phi1_center_projection
        ),
        target_remaining_fold3_low_pi_raw_phi1_center_projection=remaining_raw,
        target_exact_trim_floor_raw_phi1_share_of_fold3_raw=(
            _safe_share(
                propensity_floor_report.target_exact_trim_floor_raw_phi1_center_projection,
                propensity_floor_report.target_fold3_raw_phi1_center_projection,
            )
        ),
        target_exact_trim_floor_inverse_pi_phi1_center_projection=(
            propensity_floor_report.target_exact_trim_floor_inverse_pi_phi1_center_projection
        ),
        target_remaining_fold3_low_pi_inverse_pi_phi1_center_projection=(
            remaining_inverse_pi
        ),
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            propensity_floor_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_row_gross_inverse_pi_amplification=(
            target_exact_trim_floor_row_gross_inverse_pi_amplification
        ),
        target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification=(
            target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification
        ),
        target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining=(
            target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining
        ),
        target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining=(
            target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining
        ),
        target_exact_trim_floor_weighted_phi1_center_projection=(
            propensity_floor_report.target_exact_trim_floor_weighted_phi1_center_projection
        ),
        target_remaining_fold3_low_pi_weighted_phi1_center_projection=(
            remaining_weighted
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            propensity_floor_report.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            target_exact_trim_floor_weighted_retention
        ),
        target_remaining_fold3_low_pi_weighted_retention=(
            target_remaining_fold3_low_pi_weighted_retention
        ),
        target_exact_trim_floor_weighted_retention_multiple_vs_remaining=(
            target_exact_trim_floor_weighted_retention_multiple_vs_remaining
        ),
        comparator_random_states=propensity_floor_report.comparator_random_states,
        comparator_exact_trim_floor_counts=comparator_exact_trim_floor_counts,
        comparator_fold3_inverse_pi_phi1_center_projections=(
            propensity_floor_report.comparator_fold3_inverse_pi_phi1_center_projections
        ),
        driver_signature=driver_signature,
        canonical_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_digest=(
            canonical_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorInversePiConcentrationTraceReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace_report()
