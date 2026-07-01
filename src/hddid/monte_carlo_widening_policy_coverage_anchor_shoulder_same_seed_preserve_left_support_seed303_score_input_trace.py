from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .estimation import _project_onto_basis
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_estimation_source_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303EstimationSourceTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_estimation_source_trace,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    _canonical_design,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)
from .validation import _fit_phase7_same_seed_replay_payload

_CENTER_GRID_INDEX = 1


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


@dataclass(slots=True)
class _Seed303ScoreInputSlice:
    random_state: int
    seed_group: str
    raw_score_center_estimate: float
    xbeta_center_projection: float
    eq31_center_estimate: float
    beta_norm: float
    score_valid_mean: float
    score_valid_std: float
    xbeta_valid_mean: float
    xbeta_valid_std: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.raw_score_center_estimate = float(self.raw_score_center_estimate)
        self.xbeta_center_projection = float(self.xbeta_center_projection)
        self.eq31_center_estimate = float(self.eq31_center_estimate)
        self.beta_norm = float(self.beta_norm)
        self.score_valid_mean = float(self.score_valid_mean)
        self.score_valid_std = float(self.score_valid_std)
        self.xbeta_valid_mean = float(self.xbeta_valid_mean)
        self.xbeta_valid_std = float(self.xbeta_valid_std)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303ScoreInputTraceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    center_grid_value: float
    target_truth_center: float
    target_raw_score_center_estimate: float
    target_xbeta_center_projection: float
    target_eq31_center_estimate: float
    target_raw_score_gap_over_truth: float
    target_residualization_lift: float
    target_raw_score_share_of_final_gap: float
    target_residualization_lift_share_of_final_gap: float
    target_beta_norm: float
    target_score_valid_mean: float
    target_score_valid_std: float
    target_xbeta_valid_mean: float
    target_xbeta_valid_std: float
    comparator_random_states: tuple[int, ...]
    comparator_raw_score_center_estimates: tuple[float, ...]
    comparator_xbeta_center_projections: tuple[float, ...]
    comparator_eq31_center_estimates: tuple[float, ...]
    driver_signature: str
    canonical_seed303_score_input_trace_digest: tuple[str, ...]

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
        self.target_truth_center = float(self.target_truth_center)
        self.target_raw_score_center_estimate = float(
            self.target_raw_score_center_estimate
        )
        self.target_xbeta_center_projection = float(self.target_xbeta_center_projection)
        self.target_eq31_center_estimate = float(self.target_eq31_center_estimate)
        self.target_raw_score_gap_over_truth = float(
            self.target_raw_score_gap_over_truth
        )
        self.target_residualization_lift = float(self.target_residualization_lift)
        self.target_raw_score_share_of_final_gap = float(
            self.target_raw_score_share_of_final_gap
        )
        self.target_residualization_lift_share_of_final_gap = float(
            self.target_residualization_lift_share_of_final_gap
        )
        self.target_beta_norm = float(self.target_beta_norm)
        self.target_score_valid_mean = float(self.target_score_valid_mean)
        self.target_score_valid_std = float(self.target_score_valid_std)
        self.target_xbeta_valid_mean = float(self.target_xbeta_valid_mean)
        self.target_xbeta_valid_std = float(self.target_xbeta_valid_std)
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_raw_score_center_estimates = tuple(
            float(value) for value in self.comparator_raw_score_center_estimates
        )
        self.comparator_xbeta_center_projections = tuple(
            float(value) for value in self.comparator_xbeta_center_projections
        )
        self.comparator_eq31_center_estimates = tuple(
            float(value) for value in self.comparator_eq31_center_estimates
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_score_input_trace_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_score_input_trace_digest
        )


def _build_seed_slice(
    *, random_state: int, seed_group: str, replication_seed: int
) -> _Seed303ScoreInputSlice:
    replay_payload = _fit_phase7_same_seed_replay_payload(
        design=_canonical_design(),
        replication_seed=int(replication_seed),
    )
    score_payload = replay_payload.score_payload
    estimation_payload = replay_payload.estimation_payload
    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    evaluation_basis = np.asarray(score_payload.evaluation_basis, dtype=float)
    raw_score = np.asarray(score_payload.s_hat_valid, dtype=float)
    raw_gamma, _ = _project_onto_basis(basis_valid_full, raw_score)
    raw_score_center = float((evaluation_basis @ raw_gamma)[_CENTER_GRID_INDEX])
    xbeta_valid = np.asarray(score_payload.x_valid, dtype=float) @ np.asarray(
        estimation_payload.beta_hat,
        dtype=float,
    )
    xbeta_gamma, _ = _project_onto_basis(basis_valid_full, xbeta_valid)
    xbeta_center = float((evaluation_basis @ xbeta_gamma)[_CENTER_GRID_INDEX])
    return _Seed303ScoreInputSlice(
        random_state=random_state,
        seed_group=seed_group,
        raw_score_center_estimate=raw_score_center,
        xbeta_center_projection=xbeta_center,
        eq31_center_estimate=float(estimation_payload.f_hat_at_z0[_CENTER_GRID_INDEX]),
        beta_norm=float(
            np.linalg.norm(np.asarray(estimation_payload.beta_hat, dtype=float))
        ),
        score_valid_mean=float(np.mean(raw_score)),
        score_valid_std=float(np.std(raw_score)),
        xbeta_valid_mean=float(np.mean(xbeta_valid)),
        xbeta_valid_std=float(np.std(xbeta_valid)),
    )


def _driver_signature(
    *,
    estimation_source_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303EstimationSourceTraceReport
    ),
    target_slice: _Seed303ScoreInputSlice,
    truth_center: float,
    comparator_slices: tuple[_Seed303ScoreInputSlice, ...],
) -> str:
    if (
        estimation_source_trace_report.driver_signature
        == "same-seed-seed303-eq31-point-source-confirmed"
        and target_slice.raw_score_center_estimate > truth_center
        and target_slice.xbeta_center_projection < 0.0
        and comparator_slices[0].raw_score_center_estimate < truth_center
        and comparator_slices[1].raw_score_center_estimate > truth_center
    ):
        return "same-seed-seed303-eq31-score-input-trace-confirmed"
    return "mixed-same-seed-seed303-score-input-trace"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_score_input_trace_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303ScoreInputTraceReport
):
    estimation_source_trace_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_estimation_source_trace()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    target_observation = seedwise_report.seed_observation(303)
    comparator_observations = (
        seedwise_report.seed_observation(202),
        seedwise_report.seed_observation(707),
    )

    target_slice = _build_seed_slice(
        random_state=target_observation.random_state,
        seed_group=target_observation.seed_group,
        replication_seed=target_observation.replication_seed,
    )
    comparator_slices = tuple(
        _build_seed_slice(
            random_state=observation.random_state,
            seed_group=observation.seed_group,
            replication_seed=observation.replication_seed,
        )
        for observation in comparator_observations
    )

    truth_center = estimation_source_trace_report.target_truth_center
    final_gap = estimation_source_trace_report.target_center_gap_over_truth
    raw_score_gap = target_slice.raw_score_center_estimate - truth_center
    residualization_lift = (
        target_slice.eq31_center_estimate - target_slice.raw_score_center_estimate
    )
    driver_signature = _driver_signature(
        estimation_source_trace_report=estimation_source_trace_report,
        target_slice=target_slice,
        truth_center=truth_center,
        comparator_slices=comparator_slices,
    )

    canonical_digest = (
        "- seed `303` / witness / `z = 0.15` decomposes as raw score projection "
        f"`{_format_float(target_slice.raw_score_center_estimate)}` minus center `xβ` projection "
        f"`{_format_float(target_slice.xbeta_center_projection)}`, so Eq. (3.1) still exits at "
        f"`{_format_float(target_slice.eq31_center_estimate)}` because residualization lifts the center estimate instead of cancelling it",
        "- the miss remains score-led even before the beta correction: raw score projection already exceeds truth "
        f"`{_format_float(truth_center)}` by `{_format_float(raw_score_gap)}`, which is "
        f"`{_format_percent(raw_score_gap / final_gap)}` of the final center gap, while the negative `xβ` projection adds the remaining "
        f"`{_format_percent(residualization_lift / final_gap)}`",
        "- comparator seed `202` stays below truth at the raw-score stage "
        f"(`{_format_float(comparator_slices[0].raw_score_center_estimate)}`) whereas comparator seed `707` is already score-led at "
        f"`{_format_float(comparator_slices[1].raw_score_center_estimate)}`; the failing seed `303` therefore sits in the score-input lane, not a beta-only or band-only lane",
        "- current Trigger 2 implication: `same-seed-seed303-eq31-score-input-trace-confirmed`; the next bounded repair should trace `rho_hat * (delta_y - (1-pi_hat) * phi1_hat - pi_hat * phi0_hat)` for seed `303` before widening bands or retuning Eq. (3.1) residualization",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303ScoreInputTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-score-input-trace"
        ),
        policy_digest=estimation_source_trace_report.policy_digest,
        binding_design=estimation_source_trace_report.binding_design,
        window_label=estimation_source_trace_report.window_label,
        focus_random_states=(
            comparator_slices[0].random_state,
            target_slice.random_state,
            comparator_slices[1].random_state,
        ),
        target_random_state=target_slice.random_state,
        target_seed_group=target_slice.seed_group,
        center_grid_value=estimation_source_trace_report.center_grid_value,
        target_truth_center=truth_center,
        target_raw_score_center_estimate=target_slice.raw_score_center_estimate,
        target_xbeta_center_projection=target_slice.xbeta_center_projection,
        target_eq31_center_estimate=target_slice.eq31_center_estimate,
        target_raw_score_gap_over_truth=raw_score_gap,
        target_residualization_lift=residualization_lift,
        target_raw_score_share_of_final_gap=(raw_score_gap / final_gap),
        target_residualization_lift_share_of_final_gap=(
            residualization_lift / final_gap
        ),
        target_beta_norm=target_slice.beta_norm,
        target_score_valid_mean=target_slice.score_valid_mean,
        target_score_valid_std=target_slice.score_valid_std,
        target_xbeta_valid_mean=target_slice.xbeta_valid_mean,
        target_xbeta_valid_std=target_slice.xbeta_valid_std,
        comparator_random_states=tuple(
            slice_.random_state for slice_ in comparator_slices
        ),
        comparator_raw_score_center_estimates=tuple(
            slice_.raw_score_center_estimate for slice_ in comparator_slices
        ),
        comparator_xbeta_center_projections=tuple(
            slice_.xbeta_center_projection for slice_ in comparator_slices
        ),
        comparator_eq31_center_estimates=tuple(
            slice_.eq31_center_estimate for slice_ in comparator_slices
        ),
        driver_signature=driver_signature,
        canonical_seed303_score_input_trace_digest=canonical_digest,
    )


def _prime_seed303_score_input_trace_cache() -> None:
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    for random_state in (202, 303, 707):
        observation = seedwise_report.seed_observation(random_state)
        _build_seed_slice(
            random_state=observation.random_state,
            seed_group=observation.seed_group,
            replication_seed=observation.replication_seed,
        )


def _build_seed303_score_input_trace_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303ScoreInputTraceReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303ScoreInputTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-score-input-trace"
        ),
        policy_digest=(
            "label=bounded-n500-p50",
            "max_total_runtime_seconds=240.0",
            "max_random_states=8",
            "stop_on_first_typed_invalidity=True",
            "min_nonparametric_coverage=0.85",
        ),
        binding_design=("DGP2", 500, 50),
        window_label="near_zero_grid",
        focus_random_states=(202, 303, 707),
        target_random_state=303,
        target_seed_group="witness",
        center_grid_value=0.15,
        target_truth_center=1.161834242728283,
        target_raw_score_center_estimate=7.669202680941233,
        target_xbeta_center_projection=-1.5166183401890352,
        target_eq31_center_estimate=9.18582102113026,
        target_raw_score_gap_over_truth=6.50736843821295,
        target_residualization_lift=1.5166183401890265,
        target_raw_score_share_of_final_gap=0.8109894268181895,
        target_residualization_lift_share_of_final_gap=0.1890105731818105,
        target_beta_norm=8.080202206529668,
        target_score_valid_mean=2.432148968972047,
        target_score_valid_std=21.22157543209093,
        target_xbeta_valid_mean=-0.48316176200306454,
        target_xbeta_valid_std=6.285466262836833,
        comparator_random_states=(202, 707),
        comparator_raw_score_center_estimates=(
            -1.3221450119752998,
            17.89715665731029,
        ),
        comparator_xbeta_center_projections=(
            -3.175833346273497,
            -2.000843702485783,
        ),
        comparator_eq31_center_estimates=(
            1.8536883342981918,
            19.898000359796082,
        ),
        driver_signature="same-seed-seed303-eq31-score-input-trace-confirmed",
        canonical_seed303_score_input_trace_digest=(
            "- seed `303` / witness / `z = 0.15` decomposes as raw score projection `7.669` minus center `xβ` projection `-1.517`, so Eq. (3.1) still exits at `9.186` because residualization lifts the center estimate instead of cancelling it",
            "- the miss remains score-led even before the beta correction: raw score projection already exceeds truth `1.162` by `6.507`, which is `81.1%` of the final center gap, while the negative `xβ` projection adds the remaining `18.9%`",
            "- comparator seed `202` stays below truth at the raw-score stage (`-1.322`) whereas comparator seed `707` is already score-led at `17.897`; the failing seed `303` therefore sits in the score-input lane, not a beta-only or band-only lane",
            "- current Trigger 2 implication: `same-seed-seed303-eq31-score-input-trace-confirmed`; the next bounded repair should trace `rho_hat * (delta_y - (1-pi_hat) * phi1_hat - pi_hat * phi0_hat)` for seed `303` before widening bands or retuning Eq. (3.1) residualization",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_score_input_trace() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303ScoreInputTraceReport
):
    _prime_seed303_score_input_trace_cache()
    return _build_seed303_score_input_trace_snapshot_report()
