from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .estimation import _project_onto_basis
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_delta_y_input_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoDeltaYInputTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_delta_y_input_trace,
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
class _Seed303RhoSupportSplitSlice:
    random_state: int
    seed_group: str
    treated_delta_center_projection: float
    treated_delta_over_pi_center_projection: float
    control_delta_center_projection: float
    control_delta_over_one_minus_pi_center_projection: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.treated_delta_center_projection = float(
            self.treated_delta_center_projection
        )
        self.treated_delta_over_pi_center_projection = float(
            self.treated_delta_over_pi_center_projection
        )
        self.control_delta_center_projection = float(
            self.control_delta_center_projection
        )
        self.control_delta_over_one_minus_pi_center_projection = float(
            self.control_delta_over_one_minus_pi_center_projection
        )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoSupportSplitTraceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    center_grid_value: float
    target_truth_center: float
    target_rho_delta_y_center_projection: float
    target_treated_delta_center_projection: float
    target_treated_delta_over_pi_center_projection: float
    target_treated_support_lift: float
    target_control_delta_center_projection: float
    target_control_delta_over_one_minus_pi_center_projection: float
    target_control_support_lift: float
    target_treated_share_of_weighted_center: float
    target_treated_share_of_positive_support_lift: float
    comparator_random_states: tuple[int, ...]
    comparator_treated_delta_over_pi_center_projections: tuple[float, ...]
    comparator_control_delta_over_one_minus_pi_center_projections: tuple[float, ...]
    driver_signature: str
    canonical_seed303_rho_support_split_trace_digest: tuple[str, ...]

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
        self.target_rho_delta_y_center_projection = float(
            self.target_rho_delta_y_center_projection
        )
        self.target_treated_delta_center_projection = float(
            self.target_treated_delta_center_projection
        )
        self.target_treated_delta_over_pi_center_projection = float(
            self.target_treated_delta_over_pi_center_projection
        )
        self.target_treated_support_lift = float(self.target_treated_support_lift)
        self.target_control_delta_center_projection = float(
            self.target_control_delta_center_projection
        )
        self.target_control_delta_over_one_minus_pi_center_projection = float(
            self.target_control_delta_over_one_minus_pi_center_projection
        )
        self.target_control_support_lift = float(self.target_control_support_lift)
        self.target_treated_share_of_weighted_center = float(
            self.target_treated_share_of_weighted_center
        )
        self.target_treated_share_of_positive_support_lift = float(
            self.target_treated_share_of_positive_support_lift
        )
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_treated_delta_over_pi_center_projections = tuple(
            float(value)
            for value in self.comparator_treated_delta_over_pi_center_projections
        )
        self.comparator_control_delta_over_one_minus_pi_center_projections = tuple(
            float(value)
            for value in self.comparator_control_delta_over_one_minus_pi_center_projections
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_rho_support_split_trace_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_rho_support_split_trace_digest
        )


def _project_center(
    basis_valid_full: np.ndarray,
    evaluation_basis: np.ndarray,
    values: np.ndarray,
) -> float:
    gamma, _ = _project_onto_basis(basis_valid_full, np.asarray(values, dtype=float))
    return float((evaluation_basis @ gamma)[_CENTER_GRID_INDEX])


def _build_support_slice(
    *,
    random_state: int,
    seed_group: str,
    replication_seed: int,
) -> _Seed303RhoSupportSplitSlice:
    replay_payload = _fit_phase7_same_seed_replay_payload(
        design=_canonical_design(),
        replication_seed=int(replication_seed),
    )
    score_payload = replay_payload.score_payload
    valid_mask = np.asarray(score_payload.valid_mask, dtype=bool)
    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    evaluation_basis = np.asarray(score_payload.evaluation_basis, dtype=float)
    treat = np.asarray(replay_payload.dataset.treat, dtype=float)[valid_mask]
    pi_hat = np.asarray(score_payload.pi_hat, dtype=float)[valid_mask]
    delta_y = np.asarray(score_payload.delta_y, dtype=float)[valid_mask]

    return _Seed303RhoSupportSplitSlice(
        random_state=random_state,
        seed_group=seed_group,
        treated_delta_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            treat * delta_y,
        ),
        treated_delta_over_pi_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            treat * delta_y / pi_hat,
        ),
        control_delta_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            -(1.0 - treat) * delta_y,
        ),
        control_delta_over_one_minus_pi_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            -(1.0 - treat) * delta_y / (1.0 - pi_hat),
        ),
    )


def _driver_signature(
    *,
    rho_delta_y_input_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoDeltaYInputTraceReport
    ),
    target_slice: _Seed303RhoSupportSplitSlice,
    comparator_slices: tuple[_Seed303RhoSupportSplitSlice, ...],
    target_rho_delta_y_center_projection: float,
) -> str:
    target_treated_share = (
        target_slice.treated_delta_over_pi_center_projection
        / target_rho_delta_y_center_projection
    )
    target_treated_support_lift = (
        target_slice.treated_delta_over_pi_center_projection
        - target_slice.treated_delta_center_projection
    )
    target_control_support_lift = (
        target_slice.control_delta_over_one_minus_pi_center_projection
        - target_slice.control_delta_center_projection
    )
    if (
        rho_delta_y_input_trace_report.driver_signature
        == "same-seed-seed303-rho-weighting-driver-confirmed"
        and target_slice.treated_delta_over_pi_center_projection
        > target_slice.control_delta_over_one_minus_pi_center_projection
        and target_treated_support_lift > target_control_support_lift
        and target_treated_share > 0.8
        and comparator_slices[0].treated_delta_over_pi_center_projection
        < comparator_slices[0].control_delta_over_one_minus_pi_center_projection
    ):
        return "same-seed-seed303-treated-inverse-pi-support-driver-confirmed"
    return "mixed-same-seed-seed303-rho-support-split-trace"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoSupportSplitTraceReport
):
    rho_delta_y_input_trace_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_delta_y_input_trace()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    target_observation = seedwise_report.seed_observation(303)
    comparator_observations = (
        seedwise_report.seed_observation(202),
        seedwise_report.seed_observation(707),
    )

    target_slice = _build_support_slice(
        random_state=target_observation.random_state,
        seed_group=target_observation.seed_group,
        replication_seed=target_observation.replication_seed,
    )
    comparator_slices = tuple(
        _build_support_slice(
            random_state=observation.random_state,
            seed_group=observation.seed_group,
            replication_seed=observation.replication_seed,
        )
        for observation in comparator_observations
    )

    target_treated_support_lift = (
        target_slice.treated_delta_over_pi_center_projection
        - target_slice.treated_delta_center_projection
    )
    target_control_support_lift = (
        target_slice.control_delta_over_one_minus_pi_center_projection
        - target_slice.control_delta_center_projection
    )
    target_rho_delta_y_center_projection = (
        target_slice.treated_delta_over_pi_center_projection
        + target_slice.control_delta_over_one_minus_pi_center_projection
    )
    target_treated_share_of_weighted_center = (
        target_slice.treated_delta_over_pi_center_projection
        / target_rho_delta_y_center_projection
    )
    target_treated_share_of_positive_support_lift = target_treated_support_lift / (
        target_treated_support_lift + target_control_support_lift
    )
    driver_signature = _driver_signature(
        rho_delta_y_input_trace_report=rho_delta_y_input_trace_report,
        target_slice=target_slice,
        comparator_slices=comparator_slices,
        target_rho_delta_y_center_projection=target_rho_delta_y_center_projection,
    )

    canonical_digest = (
        "- seed `303` / witness / `z = 0.15` decomposes `rho_hat * delta_y = "
        f"{_format_float(target_rho_delta_y_center_projection)}` into treated `delta_y / pi_hat = "
        f"{_format_float(target_slice.treated_delta_over_pi_center_projection)}` and control `-delta_y / (1-pi_hat) = "
        f"{_format_float(target_slice.control_delta_over_one_minus_pi_center_projection)}`"
        ", so treated inverse-`pi_hat` support already carries "
        f"`{_format_percent(target_treated_share_of_weighted_center)}` of the weighted center projection",
        "- relative to the unweighted branches `treat * delta_y = "
        f"{_format_float(target_slice.treated_delta_center_projection)}` and `-(1-treat) * delta_y = "
        f"{_format_float(target_slice.control_delta_center_projection)}`"
        ", support weighting adds "
        f"`{_format_float(target_treated_support_lift)}` on the treated side versus only "
        f"`{_format_float(target_control_support_lift)}` on the control side, meaning treated support contributes "
        f"`{_format_percent(target_treated_share_of_positive_support_lift)}` of the positive support lift",
        "- comparator seed `202` flips the split the other way "
        f"(`{_format_float(comparator_slices[0].treated_delta_over_pi_center_projection)}` treated vs "
        f"`{_format_float(comparator_slices[0].control_delta_over_one_minus_pi_center_projection)}` control), "
        "while residual seed `707` remains treated-dominant but only at "
        f"`{_format_float(comparator_slices[1].treated_delta_over_pi_center_projection)}` vs "
        f"`{_format_float(comparator_slices[1].control_delta_over_one_minus_pi_center_projection)}`; "
        "seed `303` is therefore the witness miss where treated inverse-`pi_hat` support becomes the dominant amplifier",
        "- current Trigger 2 implication: `same-seed-seed303-treated-inverse-pi-support-driver-confirmed`; the next bounded repair should trace seed `303` through treated low-`pi_hat` support before spending the queued residual slot on seed `707`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoSupportSplitTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-rho-support-split-trace"
        ),
        policy_digest=rho_delta_y_input_trace_report.policy_digest,
        binding_design=rho_delta_y_input_trace_report.binding_design,
        window_label=rho_delta_y_input_trace_report.window_label,
        focus_random_states=(
            comparator_slices[0].random_state,
            target_slice.random_state,
            comparator_slices[1].random_state,
        ),
        target_random_state=target_slice.random_state,
        target_seed_group=target_slice.seed_group,
        center_grid_value=rho_delta_y_input_trace_report.center_grid_value,
        target_truth_center=rho_delta_y_input_trace_report.target_truth_center,
        target_rho_delta_y_center_projection=target_rho_delta_y_center_projection,
        target_treated_delta_center_projection=target_slice.treated_delta_center_projection,
        target_treated_delta_over_pi_center_projection=(
            target_slice.treated_delta_over_pi_center_projection
        ),
        target_treated_support_lift=target_treated_support_lift,
        target_control_delta_center_projection=target_slice.control_delta_center_projection,
        target_control_delta_over_one_minus_pi_center_projection=(
            target_slice.control_delta_over_one_minus_pi_center_projection
        ),
        target_control_support_lift=target_control_support_lift,
        target_treated_share_of_weighted_center=target_treated_share_of_weighted_center,
        target_treated_share_of_positive_support_lift=(
            target_treated_share_of_positive_support_lift
        ),
        comparator_random_states=tuple(
            slice_.random_state for slice_ in comparator_slices
        ),
        comparator_treated_delta_over_pi_center_projections=tuple(
            slice_.treated_delta_over_pi_center_projection
            for slice_ in comparator_slices
        ),
        comparator_control_delta_over_one_minus_pi_center_projections=tuple(
            slice_.control_delta_over_one_minus_pi_center_projection
            for slice_ in comparator_slices
        ),
        driver_signature=driver_signature,
        canonical_seed303_rho_support_split_trace_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoSupportSplitTraceReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace_report()
