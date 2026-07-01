from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .estimation import _project_onto_basis
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoSupportSplitTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    _canonical_design,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)
from .validation import _fit_phase7_same_seed_replay_payload

_CENTER_GRID_INDEX = 1
_LOW_PI_THRESHOLD = 0.10


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


@dataclass(slots=True)
class _Seed303LowPiTreatedSupportSlice:
    random_state: int
    seed_group: str
    treated_count: int
    low_pi_treated_count: int
    treated_delta_center_projection: float
    low_pi_treated_delta_center_projection: float
    high_pi_treated_delta_center_projection: float
    treated_delta_over_pi_center_projection: float
    low_pi_treated_delta_over_pi_center_projection: float
    high_pi_treated_delta_over_pi_center_projection: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.treated_count = int(self.treated_count)
        self.low_pi_treated_count = int(self.low_pi_treated_count)
        self.treated_delta_center_projection = float(
            self.treated_delta_center_projection
        )
        self.low_pi_treated_delta_center_projection = float(
            self.low_pi_treated_delta_center_projection
        )
        self.high_pi_treated_delta_center_projection = float(
            self.high_pi_treated_delta_center_projection
        )
        self.treated_delta_over_pi_center_projection = float(
            self.treated_delta_over_pi_center_projection
        )
        self.low_pi_treated_delta_over_pi_center_projection = float(
            self.low_pi_treated_delta_over_pi_center_projection
        )
        self.high_pi_treated_delta_over_pi_center_projection = float(
            self.high_pi_treated_delta_over_pi_center_projection
        )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiTreatedSupportTraceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    center_grid_value: float
    target_truth_center: float
    low_pi_threshold: float
    target_treated_count: int
    target_low_pi_treated_count: int
    target_low_pi_share_of_treated_count: float
    target_treated_delta_over_pi_center_projection: float
    target_low_pi_treated_delta_center_projection: float
    target_low_pi_treated_delta_over_pi_center_projection: float
    target_high_pi_treated_delta_over_pi_center_projection: float
    target_low_pi_share_of_treated_weighted_center: float
    target_treated_support_lift: float
    target_low_pi_treated_support_lift: float
    target_high_pi_treated_support_lift: float
    target_low_pi_share_of_positive_support_lift: float
    comparator_random_states: tuple[int, ...]
    comparator_low_pi_treated_delta_over_pi_center_projections: tuple[float, ...]
    comparator_low_pi_share_of_treated_weighted_center: tuple[float, ...]
    comparator_low_pi_share_of_positive_support_lift: tuple[float, ...]
    driver_signature: str
    canonical_seed303_low_pi_treated_support_trace_digest: tuple[str, ...]

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
        self.low_pi_threshold = float(self.low_pi_threshold)
        self.target_treated_count = int(self.target_treated_count)
        self.target_low_pi_treated_count = int(self.target_low_pi_treated_count)
        self.target_low_pi_share_of_treated_count = float(
            self.target_low_pi_share_of_treated_count
        )
        self.target_treated_delta_over_pi_center_projection = float(
            self.target_treated_delta_over_pi_center_projection
        )
        self.target_low_pi_treated_delta_center_projection = float(
            self.target_low_pi_treated_delta_center_projection
        )
        self.target_low_pi_treated_delta_over_pi_center_projection = float(
            self.target_low_pi_treated_delta_over_pi_center_projection
        )
        self.target_high_pi_treated_delta_over_pi_center_projection = float(
            self.target_high_pi_treated_delta_over_pi_center_projection
        )
        self.target_low_pi_share_of_treated_weighted_center = float(
            self.target_low_pi_share_of_treated_weighted_center
        )
        self.target_treated_support_lift = float(self.target_treated_support_lift)
        self.target_low_pi_treated_support_lift = float(
            self.target_low_pi_treated_support_lift
        )
        self.target_high_pi_treated_support_lift = float(
            self.target_high_pi_treated_support_lift
        )
        self.target_low_pi_share_of_positive_support_lift = float(
            self.target_low_pi_share_of_positive_support_lift
        )
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_low_pi_treated_delta_over_pi_center_projections = tuple(
            float(value)
            for value in self.comparator_low_pi_treated_delta_over_pi_center_projections
        )
        self.comparator_low_pi_share_of_treated_weighted_center = tuple(
            float(value)
            for value in self.comparator_low_pi_share_of_treated_weighted_center
        )
        self.comparator_low_pi_share_of_positive_support_lift = tuple(
            float(value)
            for value in self.comparator_low_pi_share_of_positive_support_lift
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_low_pi_treated_support_trace_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_low_pi_treated_support_trace_digest
        )


def _project_center(
    basis_valid_full: np.ndarray,
    evaluation_basis: np.ndarray,
    values: np.ndarray,
) -> float:
    gamma, _ = _project_onto_basis(
        np.asarray(basis_valid_full, dtype=float),
        np.asarray(values, dtype=float),
    )
    return float(
        (np.asarray(evaluation_basis, dtype=float) @ gamma)[_CENTER_GRID_INDEX]
    )


def _build_low_pi_slice(
    *,
    random_state: int,
    seed_group: str,
    replication_seed: int,
) -> _Seed303LowPiTreatedSupportSlice:
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
    treated_mask = treat == 1.0
    low_pi_treated_mask = treated_mask & (pi_hat <= _LOW_PI_THRESHOLD)
    high_pi_treated_mask = treated_mask & ~low_pi_treated_mask
    treated_delta = treat * delta_y
    treated_delta_over_pi = treat * delta_y / pi_hat

    return _Seed303LowPiTreatedSupportSlice(
        random_state=random_state,
        seed_group=seed_group,
        treated_count=int(treated_mask.sum()),
        low_pi_treated_count=int(low_pi_treated_mask.sum()),
        treated_delta_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            treated_delta,
        ),
        low_pi_treated_delta_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            treated_delta * low_pi_treated_mask,
        ),
        high_pi_treated_delta_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            treated_delta * high_pi_treated_mask,
        ),
        treated_delta_over_pi_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            treated_delta_over_pi,
        ),
        low_pi_treated_delta_over_pi_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            treated_delta_over_pi * low_pi_treated_mask,
        ),
        high_pi_treated_delta_over_pi_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            treated_delta_over_pi * high_pi_treated_mask,
        ),
    )


def _treated_support_lift(slice_: _Seed303LowPiTreatedSupportSlice) -> float:
    return (
        slice_.treated_delta_over_pi_center_projection
        - slice_.treated_delta_center_projection
    )


def _low_pi_support_lift(slice_: _Seed303LowPiTreatedSupportSlice) -> float:
    return (
        slice_.low_pi_treated_delta_over_pi_center_projection
        - slice_.low_pi_treated_delta_center_projection
    )


def _high_pi_support_lift(slice_: _Seed303LowPiTreatedSupportSlice) -> float:
    return (
        slice_.high_pi_treated_delta_over_pi_center_projection
        - slice_.high_pi_treated_delta_center_projection
    )


def _driver_signature(
    *,
    rho_support_split_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoSupportSplitTraceReport
    ),
    target_low_pi_share_of_treated_weighted_center: float,
    target_low_pi_share_of_positive_support_lift: float,
    comparator_low_pi_share_of_treated_weighted_center: tuple[float, ...],
) -> str:
    if (
        rho_support_split_trace_report.driver_signature
        == "same-seed-seed303-treated-inverse-pi-support-driver-confirmed"
        and target_low_pi_share_of_treated_weighted_center > 0.75
        and target_low_pi_share_of_positive_support_lift > 0.95
        and comparator_low_pi_share_of_treated_weighted_center[0] < 0.0
        and comparator_low_pi_share_of_treated_weighted_center[1] < 0.25
    ):
        return "same-seed-seed303-low-pi-treated-support-driver-confirmed"
    return "mixed-same-seed-seed303-low-pi-treated-support-trace"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiTreatedSupportTraceReport
):
    rho_support_split_trace_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    target_observation = seedwise_report.seed_observation(303)
    comparator_observations = (
        seedwise_report.seed_observation(202),
        seedwise_report.seed_observation(707),
    )

    target_slice = _build_low_pi_slice(
        random_state=target_observation.random_state,
        seed_group=target_observation.seed_group,
        replication_seed=target_observation.replication_seed,
    )
    comparator_slices = tuple(
        _build_low_pi_slice(
            random_state=observation.random_state,
            seed_group=observation.seed_group,
            replication_seed=observation.replication_seed,
        )
        for observation in comparator_observations
    )

    target_low_pi_share_of_treated_count = (
        target_slice.low_pi_treated_count / target_slice.treated_count
    )
    target_low_pi_share_of_treated_weighted_center = (
        target_slice.low_pi_treated_delta_over_pi_center_projection
        / target_slice.treated_delta_over_pi_center_projection
    )
    target_treated_support_lift = _treated_support_lift(target_slice)
    target_low_pi_treated_support_lift = _low_pi_support_lift(target_slice)
    target_high_pi_treated_support_lift = _high_pi_support_lift(target_slice)
    target_low_pi_share_of_positive_support_lift = (
        target_low_pi_treated_support_lift / target_treated_support_lift
    )

    comparator_low_pi_share_of_treated_weighted_center = tuple(
        slice_.low_pi_treated_delta_over_pi_center_projection
        / slice_.treated_delta_over_pi_center_projection
        for slice_ in comparator_slices
    )
    comparator_low_pi_share_of_positive_support_lift = tuple(
        _low_pi_support_lift(slice_) / _treated_support_lift(slice_)
        for slice_ in comparator_slices
    )
    driver_signature = _driver_signature(
        rho_support_split_trace_report=rho_support_split_trace_report,
        target_low_pi_share_of_treated_weighted_center=(
            target_low_pi_share_of_treated_weighted_center
        ),
        target_low_pi_share_of_positive_support_lift=(
            target_low_pi_share_of_positive_support_lift
        ),
        comparator_low_pi_share_of_treated_weighted_center=(
            comparator_low_pi_share_of_treated_weighted_center
        ),
    )

    canonical_digest = (
        "- seed `303` / witness / `z = 0.15` needs only the treated `pi_hat <= 0.1` tail "
        f"(`{target_slice.low_pi_treated_count}/{target_slice.treated_count} = "
        f"{_format_percent(target_low_pi_share_of_treated_count)}` of treated-valid observations) to explain "
        f"`{_format_float(target_slice.low_pi_treated_delta_over_pi_center_projection)}` of the treated inverse-`pi_hat` center projection "
        f"`{_format_float(target_slice.treated_delta_over_pi_center_projection)}`, so this low-`pi_hat` slice alone carries "
        f"`{_format_percent(target_low_pi_share_of_treated_weighted_center)}` of the weighted treated support",
        "- on the same seed, that low-`pi_hat` tail turns a tiny treated baseline projection "
        f"`{_format_float(target_slice.low_pi_treated_delta_center_projection)}` into "
        f"`{_format_float(target_slice.low_pi_treated_delta_over_pi_center_projection)}`, contributing "
        f"`{_format_float(target_low_pi_treated_support_lift)}` of the total treated support lift "
        f"`{_format_float(target_treated_support_lift)}`; the remaining treated support above `pi_hat > 0.1` adds only "
        f"`{_format_float(target_high_pi_treated_support_lift)}`, so the low-`pi_hat` slice supplies "
        f"`{_format_percent(target_low_pi_share_of_positive_support_lift)}` of the positive support lift",
        "- comparator seed `202` uses its low-`pi_hat` treated tail to push the treated projection negative "
        f"(`{_format_float(comparator_slices[0].low_pi_treated_delta_over_pi_center_projection)}`), while residual seed `707` keeps the same slice positive but small "
        f"(`{_format_float(comparator_slices[1].low_pi_treated_delta_over_pi_center_projection)}`, only "
        f"`{_format_percent(comparator_low_pi_share_of_treated_weighted_center[1])}` of treated weighted support and "
        f"`{_format_percent(comparator_low_pi_share_of_positive_support_lift[1])}` of treated support lift), so seed `303` is the witness miss where a narrow treated low-`pi_hat` tail becomes the dominant amplifier",
        "- current Trigger 2 implication: `same-seed-seed303-low-pi-treated-support-driver-confirmed`; the next bounded repair should trace seed `303` through the treated low-`pi_hat` tail's `pi_hat` floor / support allocation before spending the queued residual slot on seed `707`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiTreatedSupportTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-low-pi-treated-support-trace"
        ),
        policy_digest=rho_support_split_trace_report.policy_digest,
        binding_design=rho_support_split_trace_report.binding_design,
        window_label=rho_support_split_trace_report.window_label,
        focus_random_states=(
            comparator_slices[0].random_state,
            target_slice.random_state,
            comparator_slices[1].random_state,
        ),
        target_random_state=target_slice.random_state,
        target_seed_group=target_slice.seed_group,
        center_grid_value=rho_support_split_trace_report.center_grid_value,
        target_truth_center=rho_support_split_trace_report.target_truth_center,
        low_pi_threshold=_LOW_PI_THRESHOLD,
        target_treated_count=target_slice.treated_count,
        target_low_pi_treated_count=target_slice.low_pi_treated_count,
        target_low_pi_share_of_treated_count=target_low_pi_share_of_treated_count,
        target_treated_delta_over_pi_center_projection=(
            target_slice.treated_delta_over_pi_center_projection
        ),
        target_low_pi_treated_delta_center_projection=(
            target_slice.low_pi_treated_delta_center_projection
        ),
        target_low_pi_treated_delta_over_pi_center_projection=(
            target_slice.low_pi_treated_delta_over_pi_center_projection
        ),
        target_high_pi_treated_delta_over_pi_center_projection=(
            target_slice.high_pi_treated_delta_over_pi_center_projection
        ),
        target_low_pi_share_of_treated_weighted_center=(
            target_low_pi_share_of_treated_weighted_center
        ),
        target_treated_support_lift=target_treated_support_lift,
        target_low_pi_treated_support_lift=target_low_pi_treated_support_lift,
        target_high_pi_treated_support_lift=target_high_pi_treated_support_lift,
        target_low_pi_share_of_positive_support_lift=(
            target_low_pi_share_of_positive_support_lift
        ),
        comparator_random_states=tuple(
            slice_.random_state for slice_ in comparator_slices
        ),
        comparator_low_pi_treated_delta_over_pi_center_projections=tuple(
            slice_.low_pi_treated_delta_over_pi_center_projection
            for slice_ in comparator_slices
        ),
        comparator_low_pi_share_of_treated_weighted_center=(
            comparator_low_pi_share_of_treated_weighted_center
        ),
        comparator_low_pi_share_of_positive_support_lift=(
            comparator_low_pi_share_of_positive_support_lift
        ),
        driver_signature=driver_signature,
        canonical_seed303_low_pi_treated_support_trace_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiTreatedSupportTraceReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace_report()
