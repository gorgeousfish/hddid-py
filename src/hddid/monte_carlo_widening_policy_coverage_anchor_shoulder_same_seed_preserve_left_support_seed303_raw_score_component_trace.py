from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .estimation import _project_onto_basis
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_score_input_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303ScoreInputTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_score_input_trace,
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
class _Seed303RawScoreComponentSlice:
    random_state: int
    seed_group: str
    raw_score_center_estimate: float
    rho_delta_y_center_projection: float
    rho_one_minus_pi_phi1_center_projection: float
    rho_pi_phi0_center_projection: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.raw_score_center_estimate = float(self.raw_score_center_estimate)
        self.rho_delta_y_center_projection = float(self.rho_delta_y_center_projection)
        self.rho_one_minus_pi_phi1_center_projection = float(
            self.rho_one_minus_pi_phi1_center_projection
        )
        self.rho_pi_phi0_center_projection = float(self.rho_pi_phi0_center_projection)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RawScoreComponentTraceReport:
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
    target_rho_delta_y_center_projection: float
    target_rho_one_minus_pi_phi1_center_projection: float
    target_rho_pi_phi0_center_projection: float
    target_delta_y_gap_over_truth: float
    target_phi1_relief_share_of_delta_y_gap: float
    target_phi0_reinforcement_share_of_delta_y_gap: float
    target_raw_score_share_of_delta_y_gap: float
    comparator_random_states: tuple[int, ...]
    comparator_rho_delta_y_center_projections: tuple[float, ...]
    comparator_rho_one_minus_pi_phi1_center_projections: tuple[float, ...]
    comparator_rho_pi_phi0_center_projections: tuple[float, ...]
    comparator_raw_score_center_estimates: tuple[float, ...]
    driver_signature: str
    canonical_seed303_raw_score_component_trace_digest: tuple[str, ...]

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
        self.target_rho_delta_y_center_projection = float(
            self.target_rho_delta_y_center_projection
        )
        self.target_rho_one_minus_pi_phi1_center_projection = float(
            self.target_rho_one_minus_pi_phi1_center_projection
        )
        self.target_rho_pi_phi0_center_projection = float(
            self.target_rho_pi_phi0_center_projection
        )
        self.target_delta_y_gap_over_truth = float(self.target_delta_y_gap_over_truth)
        self.target_phi1_relief_share_of_delta_y_gap = float(
            self.target_phi1_relief_share_of_delta_y_gap
        )
        self.target_phi0_reinforcement_share_of_delta_y_gap = float(
            self.target_phi0_reinforcement_share_of_delta_y_gap
        )
        self.target_raw_score_share_of_delta_y_gap = float(
            self.target_raw_score_share_of_delta_y_gap
        )
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_rho_delta_y_center_projections = tuple(
            float(value) for value in self.comparator_rho_delta_y_center_projections
        )
        self.comparator_rho_one_minus_pi_phi1_center_projections = tuple(
            float(value)
            for value in self.comparator_rho_one_minus_pi_phi1_center_projections
        )
        self.comparator_rho_pi_phi0_center_projections = tuple(
            float(value) for value in self.comparator_rho_pi_phi0_center_projections
        )
        self.comparator_raw_score_center_estimates = tuple(
            float(value) for value in self.comparator_raw_score_center_estimates
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_raw_score_component_trace_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_raw_score_component_trace_digest
        )


def _project_center(
    basis_valid_full: np.ndarray,
    evaluation_basis: np.ndarray,
    values: np.ndarray,
) -> float:
    gamma, _ = _project_onto_basis(basis_valid_full, np.asarray(values, dtype=float))
    return float((evaluation_basis @ gamma)[_CENTER_GRID_INDEX])


def _build_component_slice(
    *,
    random_state: int,
    seed_group: str,
    replication_seed: int,
) -> _Seed303RawScoreComponentSlice:
    replay_payload = _fit_phase7_same_seed_replay_payload(
        design=_canonical_design(),
        replication_seed=int(replication_seed),
    )
    score_payload = replay_payload.score_payload
    valid_mask = np.asarray(score_payload.valid_mask, dtype=bool)
    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    evaluation_basis = np.asarray(score_payload.evaluation_basis, dtype=float)
    delta_y = np.asarray(score_payload.delta_y, dtype=float)[valid_mask]
    pi_hat = np.asarray(score_payload.pi_hat, dtype=float)[valid_mask]
    phi1_hat = np.asarray(score_payload.phi1_hat, dtype=float)[valid_mask]
    phi0_hat = np.asarray(score_payload.phi0_hat, dtype=float)[valid_mask]
    rho_hat = np.asarray(score_payload.rho_hat, dtype=float)[valid_mask]
    return _Seed303RawScoreComponentSlice(
        random_state=random_state,
        seed_group=seed_group,
        raw_score_center_estimate=_project_center(
            basis_valid_full,
            evaluation_basis,
            np.asarray(score_payload.s_hat_valid, dtype=float),
        ),
        rho_delta_y_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            rho_hat * delta_y,
        ),
        rho_one_minus_pi_phi1_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            rho_hat * (1.0 - pi_hat) * phi1_hat,
        ),
        rho_pi_phi0_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            rho_hat * pi_hat * phi0_hat,
        ),
    )


def _driver_signature(
    *,
    score_input_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303ScoreInputTraceReport
    ),
    target_slice: _Seed303RawScoreComponentSlice,
    truth_center: float,
    comparator_slices: tuple[_Seed303RawScoreComponentSlice, ...],
) -> str:
    if (
        score_input_trace_report.driver_signature
        == "same-seed-seed303-eq31-score-input-trace-confirmed"
        and target_slice.rho_delta_y_center_projection > truth_center
        and target_slice.rho_delta_y_center_projection
        > target_slice.rho_one_minus_pi_phi1_center_projection
        and target_slice.rho_pi_phi0_center_projection < 0.0
        and comparator_slices[0].rho_pi_phi0_center_projection
        > comparator_slices[0].rho_delta_y_center_projection
        and comparator_slices[1].rho_one_minus_pi_phi1_center_projection < 0.0
        and comparator_slices[1].rho_pi_phi0_center_projection < 0.0
    ):
        return "same-seed-seed303-raw-score-delta-y-driver-confirmed"
    return "mixed-same-seed-seed303-raw-score-component-trace"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RawScoreComponentTraceReport
):
    score_input_trace_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_score_input_trace()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    target_observation = seedwise_report.seed_observation(303)
    comparator_observations = (
        seedwise_report.seed_observation(202),
        seedwise_report.seed_observation(707),
    )

    target_slice = _build_component_slice(
        random_state=target_observation.random_state,
        seed_group=target_observation.seed_group,
        replication_seed=target_observation.replication_seed,
    )
    comparator_slices = tuple(
        _build_component_slice(
            random_state=observation.random_state,
            seed_group=observation.seed_group,
            replication_seed=observation.replication_seed,
        )
        for observation in comparator_observations
    )

    truth_center = score_input_trace_report.target_truth_center
    delta_y_gap_over_truth = target_slice.rho_delta_y_center_projection - truth_center
    raw_score_gap_over_truth = target_slice.raw_score_center_estimate - truth_center
    phi1_relief_share_of_delta_y_gap = (
        target_slice.rho_one_minus_pi_phi1_center_projection / delta_y_gap_over_truth
    )
    phi0_reinforcement_share_of_delta_y_gap = (
        -target_slice.rho_pi_phi0_center_projection / delta_y_gap_over_truth
    )
    raw_score_share_of_delta_y_gap = raw_score_gap_over_truth / delta_y_gap_over_truth
    driver_signature = _driver_signature(
        score_input_trace_report=score_input_trace_report,
        target_slice=target_slice,
        truth_center=truth_center,
        comparator_slices=comparator_slices,
    )

    canonical_digest = (
        "- seed `303` / witness / `z = 0.15` decomposes raw score "
        f"`{_format_float(target_slice.raw_score_center_estimate)}` into `rho_hat * delta_y = {_format_float(target_slice.rho_delta_y_center_projection)}`, "
        f"`rho_hat * (1-pi_hat) * phi1_hat = {_format_float(target_slice.rho_one_minus_pi_phi1_center_projection)}`, and "
        f"`rho_hat * pi_hat * phi0_hat = {_format_float(target_slice.rho_pi_phi0_center_projection)}`, so the center miss is still delta-y-led before Eq. (3.1) residualization",
        "- the `rho_hat * delta_y` lane alone already overshoots truth "
        f"`{_format_float(truth_center)}` by `{_format_float(delta_y_gap_over_truth)}`; the `phi1_hat` term claws back "
        f"`{_format_percent(phi1_relief_share_of_delta_y_gap)}` of that overshoot, but the negative `phi0_hat` term adds back another "
        f"`{_format_percent(phi0_reinforcement_share_of_delta_y_gap)}`, leaving raw score still `{_format_float(raw_score_gap_over_truth)}` above truth",
        "- comparator seed `202` stays below truth because its positive "
        f"`rho_hat * pi_hat * phi0_hat = {_format_float(comparator_slices[0].rho_pi_phi0_center_projection)}` overwhelms `rho_hat * delta_y = {_format_float(comparator_slices[0].rho_delta_y_center_projection)}`, "
        "whereas comparator seed `707` is already score-led because both nuisance terms are negative "
        f"(`{_format_float(comparator_slices[1].rho_one_minus_pi_phi1_center_projection)}`, `{_format_float(comparator_slices[1].rho_pi_phi0_center_projection)}`) and therefore amplify the positive score lane",
        "- current Trigger 2 implication: `same-seed-seed303-raw-score-delta-y-driver-confirmed`; the next bounded repair should split seed `303` between `rho_hat` and `delta_y` inputs before spending the queued residual slot on seed `707`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RawScoreComponentTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-raw-score-component-trace"
        ),
        policy_digest=score_input_trace_report.policy_digest,
        binding_design=score_input_trace_report.binding_design,
        window_label=score_input_trace_report.window_label,
        focus_random_states=(
            comparator_slices[0].random_state,
            target_slice.random_state,
            comparator_slices[1].random_state,
        ),
        target_random_state=target_slice.random_state,
        target_seed_group=target_slice.seed_group,
        center_grid_value=score_input_trace_report.center_grid_value,
        target_truth_center=truth_center,
        target_raw_score_center_estimate=target_slice.raw_score_center_estimate,
        target_rho_delta_y_center_projection=target_slice.rho_delta_y_center_projection,
        target_rho_one_minus_pi_phi1_center_projection=(
            target_slice.rho_one_minus_pi_phi1_center_projection
        ),
        target_rho_pi_phi0_center_projection=target_slice.rho_pi_phi0_center_projection,
        target_delta_y_gap_over_truth=delta_y_gap_over_truth,
        target_phi1_relief_share_of_delta_y_gap=phi1_relief_share_of_delta_y_gap,
        target_phi0_reinforcement_share_of_delta_y_gap=(
            phi0_reinforcement_share_of_delta_y_gap
        ),
        target_raw_score_share_of_delta_y_gap=raw_score_share_of_delta_y_gap,
        comparator_random_states=tuple(
            slice_.random_state for slice_ in comparator_slices
        ),
        comparator_rho_delta_y_center_projections=tuple(
            slice_.rho_delta_y_center_projection for slice_ in comparator_slices
        ),
        comparator_rho_one_minus_pi_phi1_center_projections=tuple(
            slice_.rho_one_minus_pi_phi1_center_projection
            for slice_ in comparator_slices
        ),
        comparator_rho_pi_phi0_center_projections=tuple(
            slice_.rho_pi_phi0_center_projection for slice_ in comparator_slices
        ),
        comparator_raw_score_center_estimates=tuple(
            slice_.raw_score_center_estimate for slice_ in comparator_slices
        ),
        driver_signature=driver_signature,
        canonical_seed303_raw_score_component_trace_digest=canonical_digest,
    )


def _prime_seed303_raw_score_component_trace_cache() -> None:
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    for random_state in (202, 303, 707):
        observation = seedwise_report.seed_observation(random_state)
        _build_component_slice(
            random_state=observation.random_state,
            seed_group=observation.seed_group,
            replication_seed=observation.replication_seed,
        )


def _build_seed303_raw_score_component_trace_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RawScoreComponentTraceReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RawScoreComponentTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-raw-score-component-trace"
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
        target_rho_delta_y_center_projection=10.848038686264498,
        target_rho_one_minus_pi_phi1_center_projection=3.9407484748101655,
        target_rho_pi_phi0_center_projection=-0.7619124694869097,
        target_delta_y_gap_over_truth=9.686204443536216,
        target_phi1_relief_share_of_delta_y_gap=0.4068413482063039,
        target_phi0_reinforcement_share_of_delta_y_gap=0.07865954863211132,
        target_raw_score_share_of_delta_y_gap=0.6718182004258064,
        comparator_random_states=(202, 707),
        comparator_rho_delta_y_center_projections=(
            1.6587382183951593,
            2.2939236450687024,
        ),
        comparator_rho_one_minus_pi_phi1_center_projections=(
            -0.3383574154198694,
            -14.967660616986867,
        ),
        comparator_rho_pi_phi0_center_projections=(
            3.3192406457903454,
            -0.6355723952547364,
        ),
        comparator_raw_score_center_estimates=(
            -1.3221450119752998,
            17.89715665731029,
        ),
        driver_signature="same-seed-seed303-raw-score-delta-y-driver-confirmed",
        canonical_seed303_raw_score_component_trace_digest=(
            "- seed `303` / witness / `z = 0.15` decomposes raw score `7.669` into `rho_hat * delta_y = 10.848`, `rho_hat * (1-pi_hat) * phi1_hat = 3.941`, and `rho_hat * pi_hat * phi0_hat = -0.762`, so the center miss is still delta-y-led before Eq. (3.1) residualization",
            "- the `rho_hat * delta_y` lane alone already overshoots truth `1.162` by `9.686`; the `phi1_hat` term claws back `40.7%` of that overshoot, but the negative `phi0_hat` term adds back another `7.9%`, leaving raw score still `6.507` above truth",
            "- comparator seed `202` stays below truth because its positive `rho_hat * pi_hat * phi0_hat = 3.319` overwhelms `rho_hat * delta_y = 1.659`, whereas comparator seed `707` is already score-led because both nuisance terms are negative (`-14.968`, `-0.636`) and therefore amplify the positive score lane",
            "- current Trigger 2 implication: `same-seed-seed303-raw-score-delta-y-driver-confirmed`; the next bounded repair should split seed `303` between `rho_hat` and `delta_y` inputs before spending the queued residual slot on seed `707`",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RawScoreComponentTraceReport
):
    _prime_seed303_raw_score_component_trace_cache()
    return _build_seed303_raw_score_component_trace_snapshot_report()
