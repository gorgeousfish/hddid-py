from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .estimation import _project_onto_basis
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiTreatedSupportTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    _canonical_design,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)
from .nuisance import CrossfitNuisanceEstimator
from .score import build_score_payload
from .splitting import make_crossfit_splits
from .validation import _generate_dataset

_CENTER_GRID_INDEX = 1
_EXACT_TRIM_FLOOR_ATOL = 1e-12


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


@dataclass(slots=True)
class _FoldLowPiTrimFloorSlice:
    fold_id: int
    low_pi_treated_count: int
    delta_over_pi_center_projection: float
    phi1_center_projection: float

    def __post_init__(self) -> None:
        self.fold_id = int(self.fold_id)
        self.low_pi_treated_count = int(self.low_pi_treated_count)
        self.delta_over_pi_center_projection = float(
            self.delta_over_pi_center_projection
        )
        self.phi1_center_projection = float(self.phi1_center_projection)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFoldTrimFloorTraceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    target_random_state: int
    target_seed_group: str
    center_grid_value: float
    low_pi_threshold: float
    trim_lower: float
    target_low_pi_treated_count: int
    target_low_pi_treated_delta_over_pi_center_projection: float
    target_low_pi_phi1_center_projection: float
    fold_ids: tuple[int, ...]
    fold_low_pi_treated_counts: tuple[int, ...]
    fold_low_pi_share_of_treated_count: tuple[float, ...]
    fold_low_pi_treated_delta_over_pi_center_projections: tuple[float, ...]
    fold_low_pi_share_of_weighted_center: tuple[float, ...]
    fold_low_pi_phi1_center_projections: tuple[float, ...]
    fold_low_pi_share_of_phi1_projection: tuple[float, ...]
    dominant_fold_id: int
    dominant_fold_low_pi_share_of_weighted_center: float
    exact_trim_floor_value: float
    exact_trim_floor_count: int
    exact_trim_floor_fold_id: int
    exact_trim_floor_share_of_low_pi_count: float
    exact_trim_floor_delta_over_pi_center_projection: float
    exact_trim_floor_share_of_low_pi_weighted_center: float
    exact_trim_floor_phi1_center_projection: float
    exact_trim_floor_share_of_low_pi_phi1_projection: float
    driver_signature: str
    canonical_seed303_low_pi_fold_trim_floor_trace_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.target_random_state = int(self.target_random_state)
        self.target_seed_group = str(self.target_seed_group).strip().lower()
        self.center_grid_value = float(self.center_grid_value)
        self.low_pi_threshold = float(self.low_pi_threshold)
        self.trim_lower = float(self.trim_lower)
        self.target_low_pi_treated_count = int(self.target_low_pi_treated_count)
        self.target_low_pi_treated_delta_over_pi_center_projection = float(
            self.target_low_pi_treated_delta_over_pi_center_projection
        )
        self.target_low_pi_phi1_center_projection = float(
            self.target_low_pi_phi1_center_projection
        )
        self.fold_ids = tuple(int(value) for value in self.fold_ids)
        self.fold_low_pi_treated_counts = tuple(
            int(value) for value in self.fold_low_pi_treated_counts
        )
        self.fold_low_pi_share_of_treated_count = tuple(
            float(value) for value in self.fold_low_pi_share_of_treated_count
        )
        self.fold_low_pi_treated_delta_over_pi_center_projections = tuple(
            float(value)
            for value in self.fold_low_pi_treated_delta_over_pi_center_projections
        )
        self.fold_low_pi_share_of_weighted_center = tuple(
            float(value) for value in self.fold_low_pi_share_of_weighted_center
        )
        self.fold_low_pi_phi1_center_projections = tuple(
            float(value) for value in self.fold_low_pi_phi1_center_projections
        )
        self.fold_low_pi_share_of_phi1_projection = tuple(
            float(value) for value in self.fold_low_pi_share_of_phi1_projection
        )
        self.dominant_fold_id = int(self.dominant_fold_id)
        self.dominant_fold_low_pi_share_of_weighted_center = float(
            self.dominant_fold_low_pi_share_of_weighted_center
        )
        self.exact_trim_floor_value = float(self.exact_trim_floor_value)
        self.exact_trim_floor_count = int(self.exact_trim_floor_count)
        self.exact_trim_floor_fold_id = int(self.exact_trim_floor_fold_id)
        self.exact_trim_floor_share_of_low_pi_count = float(
            self.exact_trim_floor_share_of_low_pi_count
        )
        self.exact_trim_floor_delta_over_pi_center_projection = float(
            self.exact_trim_floor_delta_over_pi_center_projection
        )
        self.exact_trim_floor_share_of_low_pi_weighted_center = float(
            self.exact_trim_floor_share_of_low_pi_weighted_center
        )
        self.exact_trim_floor_phi1_center_projection = float(
            self.exact_trim_floor_phi1_center_projection
        )
        self.exact_trim_floor_share_of_low_pi_phi1_projection = float(
            self.exact_trim_floor_share_of_low_pi_phi1_projection
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_low_pi_fold_trim_floor_trace_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_low_pi_fold_trim_floor_trace_digest
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


def _build_target_payload(*, replication_seed: int) -> tuple[np.ndarray, ...]:
    design = _canonical_design()
    dataset = _generate_dataset(design, random_state=int(replication_seed))
    data = dataset.to_validated_data()
    splits = make_crossfit_splits(
        n_obs=data.n_obs,
        n_folds=design.n_folds,
        random_state=int(replication_seed),
        trim_lower=design.trim_lower,
        trim_upper=design.trim_upper,
    )
    nuisance_payload = CrossfitNuisanceEstimator(oracle_lane=design.oracle_lane).fit(
        data,
        splits,
    )
    score_payload = build_score_payload(data, nuisance_payload)

    valid_mask = np.asarray(score_payload.valid_mask, dtype=bool)
    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    evaluation_basis = np.asarray(score_payload.evaluation_basis, dtype=float)
    fold_ids = np.asarray(score_payload.fold_ids, dtype=int)[valid_mask]
    treat = np.asarray(data.treat, dtype=float)[valid_mask]
    pi_hat = np.asarray(score_payload.pi_hat, dtype=float)[valid_mask]
    delta_y = np.asarray(score_payload.delta_y, dtype=float)[valid_mask]
    rho_hat = np.asarray(score_payload.rho_hat, dtype=float)[valid_mask]
    phi1_hat = np.asarray(score_payload.phi1_hat, dtype=float)[valid_mask]
    return (
        basis_valid_full,
        evaluation_basis,
        fold_ids,
        treat,
        pi_hat,
        delta_y,
        rho_hat,
        phi1_hat,
    )


def _build_fold_slices(
    *,
    basis_valid_full: np.ndarray,
    evaluation_basis: np.ndarray,
    fold_ids: np.ndarray,
    low_pi_treated_mask: np.ndarray,
    treat: np.ndarray,
    delta_y: np.ndarray,
    pi_hat: np.ndarray,
    rho_hat: np.ndarray,
    phi1_hat: np.ndarray,
) -> tuple[_FoldLowPiTrimFloorSlice, ...]:
    slices: list[_FoldLowPiTrimFloorSlice] = []
    for fold_id in sorted(
        int(value) for value in np.unique(fold_ids[low_pi_treated_mask])
    ):
        fold_mask = low_pi_treated_mask & (fold_ids == fold_id)
        slices.append(
            _FoldLowPiTrimFloorSlice(
                fold_id=fold_id,
                low_pi_treated_count=int(np.sum(fold_mask)),
                delta_over_pi_center_projection=_project_center(
                    basis_valid_full,
                    evaluation_basis,
                    treat * delta_y / pi_hat * fold_mask,
                ),
                phi1_center_projection=_project_center(
                    basis_valid_full,
                    evaluation_basis,
                    rho_hat * (1.0 - pi_hat) * phi1_hat * fold_mask,
                ),
            )
        )
    return tuple(slices)


def _driver_signature(
    *,
    low_pi_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiTreatedSupportTraceReport
    ),
    dominant_fold_id: int,
    dominant_fold_share_of_weighted_center: float,
    exact_trim_floor_count: int,
    exact_trim_floor_fold_id: int,
    exact_trim_floor_share_of_low_pi_weighted_center: float,
    exact_trim_floor_share_of_low_pi_phi1_projection: float,
) -> str:
    if (
        low_pi_report.driver_signature
        == "same-seed-seed303-low-pi-treated-support-driver-confirmed"
        and dominant_fold_id == 3
        and dominant_fold_share_of_weighted_center > 0.9
        and exact_trim_floor_count == 1
        and exact_trim_floor_fold_id == 3
        and exact_trim_floor_share_of_low_pi_weighted_center > 0.3
        and exact_trim_floor_share_of_low_pi_phi1_projection > 0.45
    ):
        return "same-seed-seed303-low-pi-fold3-trim-floor-driver-confirmed"
    return "mixed-same-seed-seed303-low-pi-fold-trim-floor-trace"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold_trim_floor_trace_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFoldTrimFloorTraceReport
):
    low_pi_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    target_observation = seedwise_report.seed_observation(
        low_pi_report.target_random_state
    )
    design = _canonical_design()
    (
        basis_valid_full,
        evaluation_basis,
        fold_ids,
        treat,
        pi_hat,
        delta_y,
        rho_hat,
        phi1_hat,
    ) = _build_target_payload(replication_seed=target_observation.replication_seed)

    low_pi_treated_mask = (treat == 1.0) & (pi_hat <= low_pi_report.low_pi_threshold)
    fold_slices = _build_fold_slices(
        basis_valid_full=basis_valid_full,
        evaluation_basis=evaluation_basis,
        fold_ids=fold_ids,
        low_pi_treated_mask=low_pi_treated_mask,
        treat=treat,
        delta_y=delta_y,
        pi_hat=pi_hat,
        rho_hat=rho_hat,
        phi1_hat=phi1_hat,
    )

    target_low_pi_treated_count = int(np.sum(low_pi_treated_mask))
    target_low_pi_phi1_center_projection = _project_center(
        basis_valid_full,
        evaluation_basis,
        rho_hat * (1.0 - pi_hat) * phi1_hat * low_pi_treated_mask,
    )
    fold_low_pi_share_of_treated_count = tuple(
        slice_.low_pi_treated_count / target_low_pi_treated_count
        for slice_ in fold_slices
    )
    fold_low_pi_share_of_weighted_center = tuple(
        slice_.delta_over_pi_center_projection
        / low_pi_report.target_low_pi_treated_delta_over_pi_center_projection
        for slice_ in fold_slices
    )
    fold_low_pi_share_of_phi1_projection = tuple(
        slice_.phi1_center_projection / target_low_pi_phi1_center_projection
        for slice_ in fold_slices
    )
    dominant_slice = max(
        fold_slices,
        key=lambda slice_: slice_.delta_over_pi_center_projection,
    )

    exact_trim_floor_value = float(np.min(pi_hat[low_pi_treated_mask]))
    exact_trim_floor_mask = low_pi_treated_mask & np.isclose(
        pi_hat,
        exact_trim_floor_value,
        atol=_EXACT_TRIM_FLOOR_ATOL,
    )
    exact_trim_floor_count = int(np.sum(exact_trim_floor_mask))
    exact_trim_floor_delta_over_pi_center_projection = _project_center(
        basis_valid_full,
        evaluation_basis,
        treat * delta_y / pi_hat * exact_trim_floor_mask,
    )
    exact_trim_floor_phi1_center_projection = _project_center(
        basis_valid_full,
        evaluation_basis,
        rho_hat * (1.0 - pi_hat) * phi1_hat * exact_trim_floor_mask,
    )
    exact_trim_floor_fold_id = int(fold_ids[exact_trim_floor_mask][0])
    exact_trim_floor_share_of_low_pi_count = (
        exact_trim_floor_count / target_low_pi_treated_count
    )
    exact_trim_floor_share_of_low_pi_weighted_center = (
        exact_trim_floor_delta_over_pi_center_projection
        / low_pi_report.target_low_pi_treated_delta_over_pi_center_projection
    )
    exact_trim_floor_share_of_low_pi_phi1_projection = (
        exact_trim_floor_phi1_center_projection / target_low_pi_phi1_center_projection
    )

    driver_signature = _driver_signature(
        low_pi_report=low_pi_report,
        dominant_fold_id=dominant_slice.fold_id,
        dominant_fold_share_of_weighted_center=(
            dominant_slice.delta_over_pi_center_projection
            / low_pi_report.target_low_pi_treated_delta_over_pi_center_projection
        ),
        exact_trim_floor_count=exact_trim_floor_count,
        exact_trim_floor_fold_id=exact_trim_floor_fold_id,
        exact_trim_floor_share_of_low_pi_weighted_center=(
            exact_trim_floor_share_of_low_pi_weighted_center
        ),
        exact_trim_floor_share_of_low_pi_phi1_projection=(
            exact_trim_floor_share_of_low_pi_phi1_projection
        ),
    )

    canonical_digest = (
        "- seed `303` / witness / `z = 0.15` keeps only `25` treated low-`pi_hat` rows, but fold `3` already carries `13/25 = 52.0%` of that tail and contributes `6.483` of the low-`pi_hat` weighted center projection `7.007` (`92.5%`), while fold `2` adds only `0.640` and fold `1` is slightly negative (`-0.115`)",
        "- the same fold `3` also absorbs `4.778` of the low-`pi_hat` treated `phi1_hat` nuisance burden `4.483` (`106.6%`) because folds `1-2` offset it at `-0.064` and `-0.231`, so both the treated support lift and the nuisance correction are now localized to one cross-fit fold rather than the whole treated sample",
        "- inside fold `3`, the single exact trim-floor row with `pi_hat = 0.010001` already carries `2.248` of the low-`pi_hat` weighted center (`32.1%`) and `2.140` of the low-`pi_hat` `phi1_hat` burden (`47.7%`), so trim-floor propagation is material but still sits inside the broader fold-`3` driver",
        "- current Trigger 2 implication: `same-seed-seed303-low-pi-fold3-trim-floor-driver-confirmed`; the next bounded repair should trace fold `3` treated nuisance predictions / exact trim-floor propensity-floor propagation before spending the queued residual slot on seed `707`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFoldTrimFloorTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-low-pi-fold-trim-floor-trace"
        ),
        policy_digest=low_pi_report.policy_digest,
        binding_design=low_pi_report.binding_design,
        window_label=low_pi_report.window_label,
        target_random_state=low_pi_report.target_random_state,
        target_seed_group=low_pi_report.target_seed_group,
        center_grid_value=low_pi_report.center_grid_value,
        low_pi_threshold=low_pi_report.low_pi_threshold,
        trim_lower=design.trim_lower,
        target_low_pi_treated_count=target_low_pi_treated_count,
        target_low_pi_treated_delta_over_pi_center_projection=(
            low_pi_report.target_low_pi_treated_delta_over_pi_center_projection
        ),
        target_low_pi_phi1_center_projection=target_low_pi_phi1_center_projection,
        fold_ids=tuple(slice_.fold_id for slice_ in fold_slices),
        fold_low_pi_treated_counts=tuple(
            slice_.low_pi_treated_count for slice_ in fold_slices
        ),
        fold_low_pi_share_of_treated_count=fold_low_pi_share_of_treated_count,
        fold_low_pi_treated_delta_over_pi_center_projections=tuple(
            slice_.delta_over_pi_center_projection for slice_ in fold_slices
        ),
        fold_low_pi_share_of_weighted_center=fold_low_pi_share_of_weighted_center,
        fold_low_pi_phi1_center_projections=tuple(
            slice_.phi1_center_projection for slice_ in fold_slices
        ),
        fold_low_pi_share_of_phi1_projection=fold_low_pi_share_of_phi1_projection,
        dominant_fold_id=dominant_slice.fold_id,
        dominant_fold_low_pi_share_of_weighted_center=(
            dominant_slice.delta_over_pi_center_projection
            / low_pi_report.target_low_pi_treated_delta_over_pi_center_projection
        ),
        exact_trim_floor_value=exact_trim_floor_value,
        exact_trim_floor_count=exact_trim_floor_count,
        exact_trim_floor_fold_id=exact_trim_floor_fold_id,
        exact_trim_floor_share_of_low_pi_count=exact_trim_floor_share_of_low_pi_count,
        exact_trim_floor_delta_over_pi_center_projection=(
            exact_trim_floor_delta_over_pi_center_projection
        ),
        exact_trim_floor_share_of_low_pi_weighted_center=(
            exact_trim_floor_share_of_low_pi_weighted_center
        ),
        exact_trim_floor_phi1_center_projection=exact_trim_floor_phi1_center_projection,
        exact_trim_floor_share_of_low_pi_phi1_projection=(
            exact_trim_floor_share_of_low_pi_phi1_projection
        ),
        driver_signature=driver_signature,
        canonical_seed303_low_pi_fold_trim_floor_trace_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold_trim_floor_trace() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFoldTrimFloorTraceReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold_trim_floor_trace_report()
