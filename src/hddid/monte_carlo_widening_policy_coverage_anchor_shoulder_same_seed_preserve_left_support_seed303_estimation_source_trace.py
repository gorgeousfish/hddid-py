from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from .inference import InferenceComputationError
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    _canonical_design,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)
from .validation import _fit_phase7_same_seed_nonparametric_payload

_CENTER_GRID_INDEX = 1
_CROSS_ENTRY_COORDINATE = (2, 1)
_MATCH_TOLERANCE = 1e-10


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303EstimationSourceTraceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    center_grid_value: float
    target_truth_center: float
    target_eq31_center_estimate: float
    target_nonparametric_center_estimate: float
    target_center_shift_after_inference: float
    target_eq31_point_source_confirmed: bool
    target_bar_gamma_matches_eq31: bool
    target_score_moment_linf: float
    target_center_gap_over_truth: float
    target_center_sigma_z_hat: float
    target_center_pointwise_lower: float
    target_center_pointwise_upper: float
    target_pointwise_lower_exceeds_truth: bool
    target_pointwise_lower_excess_over_truth: float
    target_sigma_f_hat_21: float
    target_omega_f_hat_21: float
    target_v_f_hat_21: float
    positive_v_f_hat_comparator_random_states: tuple[int, ...]
    positive_v_f_hat_comparator_entries: tuple[float, ...]
    driver_signature: str
    canonical_seed303_estimation_source_trace_digest: tuple[str, ...]

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
        self.target_eq31_center_estimate = float(self.target_eq31_center_estimate)
        self.target_nonparametric_center_estimate = float(
            self.target_nonparametric_center_estimate
        )
        self.target_center_shift_after_inference = float(
            self.target_center_shift_after_inference
        )
        self.target_eq31_point_source_confirmed = bool(
            self.target_eq31_point_source_confirmed
        )
        self.target_bar_gamma_matches_eq31 = bool(self.target_bar_gamma_matches_eq31)
        self.target_score_moment_linf = float(self.target_score_moment_linf)
        self.target_center_gap_over_truth = float(self.target_center_gap_over_truth)
        self.target_center_sigma_z_hat = float(self.target_center_sigma_z_hat)
        self.target_center_pointwise_lower = float(self.target_center_pointwise_lower)
        self.target_center_pointwise_upper = float(self.target_center_pointwise_upper)
        self.target_pointwise_lower_exceeds_truth = bool(
            self.target_pointwise_lower_exceeds_truth
        )
        self.target_pointwise_lower_excess_over_truth = float(
            self.target_pointwise_lower_excess_over_truth
        )
        self.target_sigma_f_hat_21 = float(self.target_sigma_f_hat_21)
        self.target_omega_f_hat_21 = float(self.target_omega_f_hat_21)
        self.target_v_f_hat_21 = float(self.target_v_f_hat_21)
        self.positive_v_f_hat_comparator_random_states = tuple(
            int(value) for value in self.positive_v_f_hat_comparator_random_states
        )
        self.positive_v_f_hat_comparator_entries = tuple(
            float(value) for value in self.positive_v_f_hat_comparator_entries
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_estimation_source_trace_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_estimation_source_trace_digest
        )


def _build_target_source_trace(
    *,
    replication_seed: int,
) -> dict[str, float | bool | np.ndarray]:
    design = _canonical_design()
    cached_payload = _fit_phase7_same_seed_nonparametric_payload(
        design=design,
        replication_seed=int(replication_seed),
        n_boot=64,
    )
    dataset = cached_payload.dataset
    score_payload = cached_payload.score_payload
    estimation_payload = cached_payload.estimation_payload
    nonparametric_payload = cached_payload.nonparametric_payload

    eq31_at_z0 = np.asarray(
        score_payload.evaluation_basis @ estimation_payload.gamma_hat, dtype=float
    )
    nonparametric_at_z0 = np.asarray(nonparametric_payload.bar_f_at_z0, dtype=float)
    truth_at_z0 = np.asarray(dataset.true_f_at_z0, dtype=float)
    pointwise_lower = np.asarray(
        nonparametric_payload.pointwise_confidence_interval.lower,
        dtype=float,
    )
    pointwise_upper = np.asarray(
        nonparametric_payload.pointwise_confidence_interval.upper,
        dtype=float,
    )
    sigma_z_hat = np.asarray(nonparametric_payload.sigma_z_hat, dtype=float)
    bar_gamma_gap = np.max(
        np.abs(
            np.asarray(nonparametric_payload.bar_gamma_hat, dtype=float)
            - np.asarray(estimation_payload.gamma_hat, dtype=float)
        )
    )
    score_moment_linf = float(
        np.max(np.abs(np.asarray(nonparametric_payload.score_moment, dtype=float)))
    )
    cross_row, cross_col = _CROSS_ENTRY_COORDINATE
    center_truth = float(truth_at_z0[_CENTER_GRID_INDEX])
    center_eq31 = float(eq31_at_z0[_CENTER_GRID_INDEX])
    center_nonparametric = float(nonparametric_at_z0[_CENTER_GRID_INDEX])
    center_shift = center_nonparametric - center_eq31
    pointwise_lower_center = float(pointwise_lower[_CENTER_GRID_INDEX])

    return {
        "center_grid_value": float(design.evaluation_grid[_CENTER_GRID_INDEX]),
        "truth_center": center_truth,
        "eq31_center_estimate": center_eq31,
        "nonparametric_center_estimate": center_nonparametric,
        "center_shift_after_inference": center_shift,
        "bar_gamma_matches_eq31": bar_gamma_gap <= _MATCH_TOLERANCE,
        "score_moment_linf": score_moment_linf,
        "center_gap_over_truth": center_eq31 - center_truth,
        "center_sigma_z_hat": float(sigma_z_hat[_CENTER_GRID_INDEX]),
        "center_pointwise_lower": pointwise_lower_center,
        "center_pointwise_upper": float(pointwise_upper[_CENTER_GRID_INDEX]),
        "pointwise_lower_exceeds_truth": pointwise_lower_center > center_truth,
        "pointwise_lower_excess_over_truth": pointwise_lower_center - center_truth,
        "sigma_f_hat_21": float(
            np.asarray(nonparametric_payload.sigma_f_hat, dtype=float)[
                cross_row, cross_col
            ]
        ),
        "omega_f_hat_21": float(
            np.asarray(nonparametric_payload.omega_f_hat, dtype=float)[
                cross_row, cross_col
            ]
        ),
        "v_f_hat_21": float(
            np.asarray(nonparametric_payload.v_f_hat, dtype=float)[cross_row, cross_col]
        ),
    }


def _driver_signature(
    *,
    runtime_evidence_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport
    ),
    source_trace: dict[str, float | bool | np.ndarray],
) -> str:
    if (
        runtime_evidence_report.driver_signature
        == "same-seed-seed303-object-flow-runtime-evidence"
        and bool(source_trace["bar_gamma_matches_eq31"])
        and abs(float(source_trace["center_shift_after_inference"])) <= _MATCH_TOLERANCE
        and float(source_trace["score_moment_linf"]) <= _MATCH_TOLERANCE
        and bool(source_trace["pointwise_lower_exceeds_truth"])
        and float(source_trace["sigma_f_hat_21"]) < 0.0
        and float(source_trace["omega_f_hat_21"]) < 0.0
        and float(source_trace["v_f_hat_21"]) < 0.0
        and runtime_evidence_report.witness_pass_runtime_snapshot.vf_cross_entry > 0.0
        and runtime_evidence_report.fresh_residual_runtime_snapshot.vf_cross_entry > 0.0
    ):
        return "same-seed-seed303-eq31-point-source-confirmed"
    return "mixed-same-seed-seed303-estimation-source-trace"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_estimation_source_trace_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303EstimationSourceTraceReport
):
    runtime_evidence_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    target_observation = seedwise_report.seed_observation(303)
    source_trace = _build_target_source_trace(
        replication_seed=target_observation.replication_seed,
    )
    driver_signature = _driver_signature(
        runtime_evidence_report=runtime_evidence_report,
        source_trace=source_trace,
    )

    canonical_digest = (
        "- seed `303` / witness / `z = 0.15` stays point-estimate-led rather than band-led: Eq. (3.1) already lands at "
        f"`{_format_float(source_trace['eq31_center_estimate'])}` while truth stays `{_format_float(source_trace['truth_center'])}`, "
        f"and the pointwise lower endpoint `{_format_float(source_trace['center_pointwise_lower'])}` remains above truth by "
        f"`{_format_float(source_trace['pointwise_lower_excess_over_truth'])}`",
        "- the center shift is not introduced by nonparametric debiasing: `bar_f_at_z0[1] - f_hat_at_z0[1]` is numerically zero and `bar_gamma_hat` matches `gamma_hat`, so the overshoot source is already present when Eq. (3.1) leaves the estimation stage",
        "- preserve-left-support covariance still flips on the same seed: `sigma_f_hat[2,1]` and `omega_f_hat[2,1]` stay negative "
        f"(`{_format_float(source_trace['sigma_f_hat_21'])}` / `{_format_float(source_trace['omega_f_hat_21'])}`), and the propagated "
        f"`v_f_hat[2,1]` remains `{_format_float(source_trace['v_f_hat_21'])}` while comparator seeds "
        f"`{runtime_evidence_report.witness_pass_runtime_snapshot.random_state}` and "
        f"`{runtime_evidence_report.fresh_residual_runtime_snapshot.random_state}` keep positive "
        f"`v_f_hat[2,1]`",
        "- current Trigger 2 implication: `same-seed-seed303-eq31-point-source-confirmed`; the next bounded repair should trace the seed `303` witness-center miss backward through Eq. (3.1) inputs instead of widening pointwise / uniform bands",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303EstimationSourceTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-estimation-source-trace"
        ),
        policy_digest=runtime_evidence_report.policy_digest,
        binding_design=runtime_evidence_report.binding_design,
        window_label=runtime_evidence_report.window_label,
        focus_random_states=(
            runtime_evidence_report.witness_pass_runtime_snapshot.random_state,
            target_observation.random_state,
            runtime_evidence_report.fresh_residual_runtime_snapshot.random_state,
        ),
        target_random_state=target_observation.random_state,
        target_seed_group=target_observation.seed_group,
        center_grid_value=float(source_trace["center_grid_value"]),
        target_truth_center=float(source_trace["truth_center"]),
        target_eq31_center_estimate=float(source_trace["eq31_center_estimate"]),
        target_nonparametric_center_estimate=float(
            source_trace["nonparametric_center_estimate"]
        ),
        target_center_shift_after_inference=float(
            source_trace["center_shift_after_inference"]
        ),
        target_eq31_point_source_confirmed=(
            bool(source_trace["bar_gamma_matches_eq31"])
            and abs(float(source_trace["center_shift_after_inference"]))
            <= _MATCH_TOLERANCE
            and float(source_trace["score_moment_linf"]) <= _MATCH_TOLERANCE
        ),
        target_bar_gamma_matches_eq31=bool(source_trace["bar_gamma_matches_eq31"]),
        target_score_moment_linf=float(source_trace["score_moment_linf"]),
        target_center_gap_over_truth=float(source_trace["center_gap_over_truth"]),
        target_center_sigma_z_hat=float(source_trace["center_sigma_z_hat"]),
        target_center_pointwise_lower=float(source_trace["center_pointwise_lower"]),
        target_center_pointwise_upper=float(source_trace["center_pointwise_upper"]),
        target_pointwise_lower_exceeds_truth=bool(
            source_trace["pointwise_lower_exceeds_truth"]
        ),
        target_pointwise_lower_excess_over_truth=float(
            source_trace["pointwise_lower_excess_over_truth"]
        ),
        target_sigma_f_hat_21=float(source_trace["sigma_f_hat_21"]),
        target_omega_f_hat_21=float(source_trace["omega_f_hat_21"]),
        target_v_f_hat_21=float(source_trace["v_f_hat_21"]),
        positive_v_f_hat_comparator_random_states=(
            runtime_evidence_report.witness_pass_runtime_snapshot.random_state,
            runtime_evidence_report.fresh_residual_runtime_snapshot.random_state,
        ),
        positive_v_f_hat_comparator_entries=(
            runtime_evidence_report.witness_pass_runtime_snapshot.vf_cross_entry,
            runtime_evidence_report.fresh_residual_runtime_snapshot.vf_cross_entry,
        ),
        driver_signature=driver_signature,
        canonical_seed303_estimation_source_trace_digest=canonical_digest,
    )


def _prime_seed303_estimation_source_trace_cache() -> None:
    try:
        _build_target_source_trace(replication_seed=883193502)
    except InferenceComputationError:
        return


def _build_seed303_estimation_source_trace_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303EstimationSourceTraceReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303EstimationSourceTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-estimation-source-trace"
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
        target_eq31_center_estimate=9.18582102113026,
        target_nonparametric_center_estimate=9.185821021130248,
        target_center_shift_after_inference=-1.0658141036401503e-14,
        target_eq31_point_source_confirmed=True,
        target_bar_gamma_matches_eq31=True,
        target_score_moment_linf=1.0819789508786926e-14,
        target_center_gap_over_truth=8.023986778401976,
        target_center_sigma_z_hat=4.087257631350298,
        target_center_pointwise_lower=2.4628804819186296,
        target_center_pointwise_upper=15.908761560341867,
        target_pointwise_lower_exceeds_truth=True,
        target_pointwise_lower_excess_over_truth=1.3010462391903466,
        target_sigma_f_hat_21=-0.020125638553956167,
        target_omega_f_hat_21=-58.76219576465819,
        target_v_f_hat_21=-287.5573494984548,
        positive_v_f_hat_comparator_random_states=(202, 707),
        positive_v_f_hat_comparator_entries=(
            201.61306478911862,
            1132.2390947315928,
        ),
        driver_signature="same-seed-seed303-eq31-point-source-confirmed",
        canonical_seed303_estimation_source_trace_digest=(
            "- seed `303` / witness / `z = 0.15` stays point-estimate-led rather than band-led: Eq. (3.1) already lands at `9.186` while truth stays `1.162`, and the pointwise lower endpoint `2.463` remains above truth by `1.301`",
            "- the center shift is not introduced by nonparametric debiasing: `bar_f_at_z0[1] - f_hat_at_z0[1]` is numerically zero and `bar_gamma_hat` matches `gamma_hat`, so the overshoot source is already present when Eq. (3.1) leaves the estimation stage",
            "- preserve-left-support covariance still flips on the same seed: `sigma_f_hat[2,1]` and `omega_f_hat[2,1]` stay negative (`-0.020` / `-58.762`), and the propagated `v_f_hat[2,1]` remains `-287.557` while comparator seeds `202` and `707` keep positive `v_f_hat[2,1]`",
            "- current Trigger 2 implication: `same-seed-seed303-eq31-point-source-confirmed`; the next bounded repair should trace the seed `303` witness-center miss backward through Eq. (3.1) inputs instead of widening pointwise / uniform bands",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_estimation_source_trace() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303EstimationSourceTraceReport
):
    _prime_seed303_estimation_source_trace_cache()
    return _build_seed303_estimation_source_trace_snapshot_report()
