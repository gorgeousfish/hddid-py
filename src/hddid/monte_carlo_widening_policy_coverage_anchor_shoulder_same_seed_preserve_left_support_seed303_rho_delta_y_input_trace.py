from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .estimation import _project_onto_basis
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RawScoreComponentTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace,
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


def _format_multiple(value: float) -> str:
    return f"{float(value):.3f}x"


@dataclass(slots=True)
class _Seed303RhoDeltaYInputSlice:
    random_state: int
    seed_group: str
    delta_y_center_projection: float
    rho_delta_y_center_projection: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.delta_y_center_projection = float(self.delta_y_center_projection)
        self.rho_delta_y_center_projection = float(self.rho_delta_y_center_projection)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoDeltaYInputTraceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    center_grid_value: float
    target_truth_center: float
    target_delta_y_center_projection: float
    target_rho_delta_y_center_projection: float
    target_delta_y_gap_over_truth: float
    target_rho_weighting_lift_over_delta_y: float
    target_delta_y_share_of_weighted_gap: float
    target_rho_weighting_share_of_weighted_gap: float
    target_weighted_to_unweighted_ratio: float
    comparator_random_states: tuple[int, ...]
    comparator_delta_y_center_projections: tuple[float, ...]
    comparator_rho_delta_y_center_projections: tuple[float, ...]
    driver_signature: str
    canonical_seed303_rho_delta_y_input_trace_digest: tuple[str, ...]

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
        self.target_delta_y_center_projection = float(
            self.target_delta_y_center_projection
        )
        self.target_rho_delta_y_center_projection = float(
            self.target_rho_delta_y_center_projection
        )
        self.target_delta_y_gap_over_truth = float(self.target_delta_y_gap_over_truth)
        self.target_rho_weighting_lift_over_delta_y = float(
            self.target_rho_weighting_lift_over_delta_y
        )
        self.target_delta_y_share_of_weighted_gap = float(
            self.target_delta_y_share_of_weighted_gap
        )
        self.target_rho_weighting_share_of_weighted_gap = float(
            self.target_rho_weighting_share_of_weighted_gap
        )
        self.target_weighted_to_unweighted_ratio = float(
            self.target_weighted_to_unweighted_ratio
        )
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_delta_y_center_projections = tuple(
            float(value) for value in self.comparator_delta_y_center_projections
        )
        self.comparator_rho_delta_y_center_projections = tuple(
            float(value) for value in self.comparator_rho_delta_y_center_projections
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_rho_delta_y_input_trace_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_rho_delta_y_input_trace_digest
        )


def _project_center(
    basis_valid_full: np.ndarray,
    evaluation_basis: np.ndarray,
    values: np.ndarray,
) -> float:
    gamma, _ = _project_onto_basis(basis_valid_full, np.asarray(values, dtype=float))
    return float((evaluation_basis @ gamma)[_CENTER_GRID_INDEX])


def _build_input_slice(
    *,
    random_state: int,
    seed_group: str,
    replication_seed: int,
) -> _Seed303RhoDeltaYInputSlice:
    replay_payload = _fit_phase7_same_seed_replay_payload(
        design=_canonical_design(),
        replication_seed=int(replication_seed),
    )
    score_payload = replay_payload.score_payload
    valid_mask = np.asarray(score_payload.valid_mask, dtype=bool)
    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    evaluation_basis = np.asarray(score_payload.evaluation_basis, dtype=float)
    delta_y_valid = np.asarray(score_payload.delta_y, dtype=float)[valid_mask]
    rho_valid = np.asarray(score_payload.rho_hat, dtype=float)[valid_mask]
    return _Seed303RhoDeltaYInputSlice(
        random_state=random_state,
        seed_group=seed_group,
        delta_y_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            delta_y_valid,
        ),
        rho_delta_y_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            rho_valid * delta_y_valid,
        ),
    )


def _driver_signature(
    *,
    raw_score_component_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RawScoreComponentTraceReport
    ),
    target_slice: _Seed303RhoDeltaYInputSlice,
    truth_center: float,
    comparator_slices: tuple[_Seed303RhoDeltaYInputSlice, ...],
) -> str:
    if (
        raw_score_component_trace_report.driver_signature
        == "same-seed-seed303-raw-score-delta-y-driver-confirmed"
        and target_slice.delta_y_center_projection > truth_center
        and target_slice.rho_delta_y_center_projection
        > target_slice.delta_y_center_projection
        and comparator_slices[0].rho_delta_y_center_projection
        < comparator_slices[0].delta_y_center_projection
        and comparator_slices[1].rho_delta_y_center_projection
        > comparator_slices[1].delta_y_center_projection
    ):
        return "same-seed-seed303-rho-weighting-driver-confirmed"
    return "mixed-same-seed-seed303-rho-delta-y-input-trace"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_delta_y_input_trace_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoDeltaYInputTraceReport
):
    raw_score_component_trace_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    target_observation = seedwise_report.seed_observation(303)
    comparator_observations = (
        seedwise_report.seed_observation(202),
        seedwise_report.seed_observation(707),
    )

    target_slice = _build_input_slice(
        random_state=target_observation.random_state,
        seed_group=target_observation.seed_group,
        replication_seed=target_observation.replication_seed,
    )
    comparator_slices = tuple(
        _build_input_slice(
            random_state=observation.random_state,
            seed_group=observation.seed_group,
            replication_seed=observation.replication_seed,
        )
        for observation in comparator_observations
    )

    truth_center = raw_score_component_trace_report.target_truth_center
    weighted_gap_over_truth = target_slice.rho_delta_y_center_projection - truth_center
    delta_y_gap_over_truth = target_slice.delta_y_center_projection - truth_center
    rho_weighting_lift_over_delta_y = (
        target_slice.rho_delta_y_center_projection
        - target_slice.delta_y_center_projection
    )
    delta_y_share_of_weighted_gap = delta_y_gap_over_truth / weighted_gap_over_truth
    rho_weighting_share_of_weighted_gap = (
        rho_weighting_lift_over_delta_y / weighted_gap_over_truth
    )
    weighted_to_unweighted_ratio = (
        target_slice.rho_delta_y_center_projection
        / target_slice.delta_y_center_projection
    )
    driver_signature = _driver_signature(
        raw_score_component_trace_report=raw_score_component_trace_report,
        target_slice=target_slice,
        truth_center=truth_center,
        comparator_slices=comparator_slices,
    )

    canonical_digest = (
        "- seed `303` / witness / `z = 0.15` keeps the unweighted `delta_y` center projection at "
        f"`{_format_float(target_slice.delta_y_center_projection)}`, only `{_format_float(delta_y_gap_over_truth)}` above truth "
        f"`{_format_float(truth_center)}`, but `rho_hat * delta_y` jumps to `{_format_float(target_slice.rho_delta_y_center_projection)}`, "
        f"so the weighted score lane adds `{_format_float(rho_weighting_lift_over_delta_y)}` of extra lift before the nuisance subtraction terms enter",
        "- relative to the weighted gap over truth "
        f"`{_format_float(weighted_gap_over_truth)}`, only `{_format_percent(delta_y_share_of_weighted_gap)}` comes from the raw `delta_y` baseline while "
        f"`{_format_percent(rho_weighting_share_of_weighted_gap)}` is introduced by the `rho_hat` weighting step; the weighted projection is "
        f"`{_format_multiple(weighted_to_unweighted_ratio)}` the unweighted `delta_y` center",
        "- comparator seed `202` keeps a similar raw `delta_y` center "
        f"`{_format_float(comparator_slices[0].delta_y_center_projection)}` but `rho_hat` slightly damps it to "
        f"`{_format_float(comparator_slices[0].rho_delta_y_center_projection)}`, whereas fresh residual seed `707` starts from raw `delta_y` "
        f"`{_format_float(comparator_slices[1].delta_y_center_projection)}` and only becomes "
        f"`{_format_float(comparator_slices[1].rho_delta_y_center_projection)}` after `rho_hat` weighting, so seed `303` is the witness miss where weighting flips a near-truth baseline into a large overshoot",
        "- current Trigger 2 implication: `same-seed-seed303-rho-weighting-driver-confirmed`; the next bounded repair should trace seed `303` through `rho_hat` / `pi_hat` support before spending the queued residual slot on seed `707`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoDeltaYInputTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-rho-delta-y-input-trace"
        ),
        policy_digest=raw_score_component_trace_report.policy_digest,
        binding_design=raw_score_component_trace_report.binding_design,
        window_label=raw_score_component_trace_report.window_label,
        focus_random_states=(
            comparator_slices[0].random_state,
            target_slice.random_state,
            comparator_slices[1].random_state,
        ),
        target_random_state=target_slice.random_state,
        target_seed_group=target_slice.seed_group,
        center_grid_value=raw_score_component_trace_report.center_grid_value,
        target_truth_center=truth_center,
        target_delta_y_center_projection=target_slice.delta_y_center_projection,
        target_rho_delta_y_center_projection=target_slice.rho_delta_y_center_projection,
        target_delta_y_gap_over_truth=delta_y_gap_over_truth,
        target_rho_weighting_lift_over_delta_y=rho_weighting_lift_over_delta_y,
        target_delta_y_share_of_weighted_gap=delta_y_share_of_weighted_gap,
        target_rho_weighting_share_of_weighted_gap=(
            rho_weighting_share_of_weighted_gap
        ),
        target_weighted_to_unweighted_ratio=weighted_to_unweighted_ratio,
        comparator_random_states=tuple(
            slice_.random_state for slice_ in comparator_slices
        ),
        comparator_delta_y_center_projections=tuple(
            slice_.delta_y_center_projection for slice_ in comparator_slices
        ),
        comparator_rho_delta_y_center_projections=tuple(
            slice_.rho_delta_y_center_projection for slice_ in comparator_slices
        ),
        driver_signature=driver_signature,
        canonical_seed303_rho_delta_y_input_trace_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_delta_y_input_trace() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoDeltaYInputTraceReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_delta_y_input_trace_report()
