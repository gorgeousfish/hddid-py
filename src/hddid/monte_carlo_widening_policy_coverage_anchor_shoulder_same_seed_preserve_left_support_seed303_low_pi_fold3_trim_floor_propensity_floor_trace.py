from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .estimation import _project_onto_basis
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_treated_phi1_prediction_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TreatedPhi1PredictionTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_treated_phi1_prediction_trace,
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
_TARGET_FOLD_ID = 3
_LOW_PI_THRESHOLD = 0.10
_TRIM_FLOOR_ATOL = 1e-12


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _safe_share(numerator: float, denominator: float) -> float:
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return 0.0
    return float(numerator) / denominator_value


@dataclass(slots=True)
class _Fold3TrimFloorPropensityFloorSlice:
    random_state: int
    seed_group: str
    fold3_low_pi_treated_count: int
    fold3_raw_phi1_center_projection: float
    fold3_inverse_pi_phi1_center_projection: float
    fold3_weighted_phi1_center_projection: float
    exact_trim_floor_count: int
    exact_trim_floor_raw_phi1_center_projection: float
    exact_trim_floor_inverse_pi_phi1_center_projection: float
    exact_trim_floor_weighted_phi1_center_projection: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.fold3_low_pi_treated_count = int(self.fold3_low_pi_treated_count)
        self.fold3_raw_phi1_center_projection = float(
            self.fold3_raw_phi1_center_projection
        )
        self.fold3_inverse_pi_phi1_center_projection = float(
            self.fold3_inverse_pi_phi1_center_projection
        )
        self.fold3_weighted_phi1_center_projection = float(
            self.fold3_weighted_phi1_center_projection
        )
        self.exact_trim_floor_count = int(self.exact_trim_floor_count)
        self.exact_trim_floor_raw_phi1_center_projection = float(
            self.exact_trim_floor_raw_phi1_center_projection
        )
        self.exact_trim_floor_inverse_pi_phi1_center_projection = float(
            self.exact_trim_floor_inverse_pi_phi1_center_projection
        )
        self.exact_trim_floor_weighted_phi1_center_projection = float(
            self.exact_trim_floor_weighted_phi1_center_projection
        )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    center_grid_value: float
    low_pi_threshold: float
    target_fold_id: int
    target_fold3_low_pi_treated_count: int
    target_fold3_raw_phi1_center_projection: float
    target_fold3_inverse_pi_phi1_center_projection: float
    target_fold3_weighted_phi1_center_projection: float
    target_fold3_raw_phi1_offset_share_of_inverse_pi: float
    target_fold3_retained_inverse_pi_share_after_floor: float
    target_fold3_inverse_pi_lift_over_raw_phi1: float
    target_exact_trim_floor_count: int
    target_exact_trim_floor_value: float
    target_exact_trim_floor_raw_phi1_center_projection: float
    target_exact_trim_floor_inverse_pi_phi1_center_projection: float
    target_exact_trim_floor_weighted_phi1_center_projection: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_raw_phi1_offset_share_of_trim_inverse_pi: float
    target_exact_trim_floor_retained_inverse_pi_share_after_floor: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    comparator_random_states: tuple[int, ...]
    comparator_fold3_raw_phi1_center_projections: tuple[float, ...]
    comparator_fold3_inverse_pi_phi1_center_projections: tuple[float, ...]
    comparator_fold3_weighted_phi1_center_projections: tuple[float, ...]
    driver_signature: str
    canonical_seed303_low_pi_fold3_trim_floor_propensity_floor_trace_digest: tuple[
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
        self.low_pi_threshold = float(self.low_pi_threshold)
        self.target_fold_id = int(self.target_fold_id)
        self.target_fold3_low_pi_treated_count = int(
            self.target_fold3_low_pi_treated_count
        )
        self.target_fold3_raw_phi1_center_projection = float(
            self.target_fold3_raw_phi1_center_projection
        )
        self.target_fold3_inverse_pi_phi1_center_projection = float(
            self.target_fold3_inverse_pi_phi1_center_projection
        )
        self.target_fold3_weighted_phi1_center_projection = float(
            self.target_fold3_weighted_phi1_center_projection
        )
        self.target_fold3_raw_phi1_offset_share_of_inverse_pi = float(
            self.target_fold3_raw_phi1_offset_share_of_inverse_pi
        )
        self.target_fold3_retained_inverse_pi_share_after_floor = float(
            self.target_fold3_retained_inverse_pi_share_after_floor
        )
        self.target_fold3_inverse_pi_lift_over_raw_phi1 = float(
            self.target_fold3_inverse_pi_lift_over_raw_phi1
        )
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_value = float(self.target_exact_trim_floor_value)
        self.target_exact_trim_floor_raw_phi1_center_projection = float(
            self.target_exact_trim_floor_raw_phi1_center_projection
        )
        self.target_exact_trim_floor_inverse_pi_phi1_center_projection = float(
            self.target_exact_trim_floor_inverse_pi_phi1_center_projection
        )
        self.target_exact_trim_floor_weighted_phi1_center_projection = float(
            self.target_exact_trim_floor_weighted_phi1_center_projection
        )
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_raw_phi1_offset_share_of_trim_inverse_pi = float(
            self.target_exact_trim_floor_raw_phi1_offset_share_of_trim_inverse_pi
        )
        self.target_exact_trim_floor_retained_inverse_pi_share_after_floor = float(
            self.target_exact_trim_floor_retained_inverse_pi_share_after_floor
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_fold3_raw_phi1_center_projections = tuple(
            float(value) for value in self.comparator_fold3_raw_phi1_center_projections
        )
        self.comparator_fold3_inverse_pi_phi1_center_projections = tuple(
            float(value)
            for value in self.comparator_fold3_inverse_pi_phi1_center_projections
        )
        self.comparator_fold3_weighted_phi1_center_projections = tuple(
            float(value)
            for value in self.comparator_fold3_weighted_phi1_center_projections
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_low_pi_fold3_trim_floor_propensity_floor_trace_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_low_pi_fold3_trim_floor_propensity_floor_trace_digest
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


def _build_slice(
    *,
    random_state: int,
    seed_group: str,
    replication_seed: int,
) -> _Fold3TrimFloorPropensityFloorSlice:
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
    phi1_hat = np.asarray(score_payload.phi1_hat, dtype=float)[valid_mask]
    rho_hat = np.asarray(score_payload.rho_hat, dtype=float)[valid_mask]

    fold3_low_pi_treated_mask = (
        (treat == 1.0) & (pi_hat <= _LOW_PI_THRESHOLD) & (fold_ids == _TARGET_FOLD_ID)
    )
    exact_trim_floor_mask = fold3_low_pi_treated_mask & np.isclose(
        pi_hat,
        design.trim_lower + 1e-6,
        atol=_TRIM_FLOOR_ATOL,
    )
    inverse_pi_phi1 = rho_hat * phi1_hat
    weighted_phi1 = rho_hat * (1.0 - pi_hat) * phi1_hat

    return _Fold3TrimFloorPropensityFloorSlice(
        random_state=random_state,
        seed_group=seed_group,
        fold3_low_pi_treated_count=int(np.sum(fold3_low_pi_treated_mask)),
        fold3_raw_phi1_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            phi1_hat * fold3_low_pi_treated_mask,
        ),
        fold3_inverse_pi_phi1_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            inverse_pi_phi1 * fold3_low_pi_treated_mask,
        ),
        fold3_weighted_phi1_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            weighted_phi1 * fold3_low_pi_treated_mask,
        ),
        exact_trim_floor_count=int(np.sum(exact_trim_floor_mask)),
        exact_trim_floor_raw_phi1_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            phi1_hat * exact_trim_floor_mask,
        )
        if np.any(exact_trim_floor_mask)
        else 0.0,
        exact_trim_floor_inverse_pi_phi1_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            inverse_pi_phi1 * exact_trim_floor_mask,
        )
        if np.any(exact_trim_floor_mask)
        else 0.0,
        exact_trim_floor_weighted_phi1_center_projection=_project_center(
            basis_valid_full,
            evaluation_basis,
            weighted_phi1 * exact_trim_floor_mask,
        )
        if np.any(exact_trim_floor_mask)
        else 0.0,
    )


def _driver_signature(
    *,
    low_pi_fold3_treated_phi1_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TreatedPhi1PredictionTraceReport
    ),
    target_slice: _Fold3TrimFloorPropensityFloorSlice,
    target_fold3_raw_phi1_offset_share: float,
    target_fold3_retained_inverse_pi_share: float,
    target_exact_trim_floor_inverse_pi_share: float,
    target_exact_trim_floor_retained_inverse_pi_share: float,
    comparator_slices: tuple[_Fold3TrimFloorPropensityFloorSlice, ...],
) -> str:
    if (
        low_pi_fold3_treated_phi1_report.driver_signature
        == "same-seed-seed303-fold3-treated-phi1-prediction-driver-confirmed"
        and target_slice.fold3_inverse_pi_phi1_center_projection
        > target_slice.fold3_weighted_phi1_center_projection
        > target_slice.fold3_raw_phi1_center_projection
        > 0.0
        and target_fold3_raw_phi1_offset_share < 0.05
        and target_fold3_retained_inverse_pi_share > 0.95
        and target_exact_trim_floor_inverse_pi_share > 0.4
        and target_exact_trim_floor_retained_inverse_pi_share > 0.98
        and comparator_slices[0].fold3_inverse_pi_phi1_center_projection < 0.0
        and comparator_slices[1].fold3_inverse_pi_phi1_center_projection < 0.0
    ):
        return "same-seed-seed303-fold3-trim-floor-propensity-floor-driver-confirmed"
    return "mixed-same-seed-seed303-fold3-trim-floor-propensity-floor-trace"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport
):
    low_pi_fold3_treated_phi1_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_treated_phi1_prediction_trace()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    target_observation = seedwise_report.seed_observation(303)
    comparator_observations = (
        seedwise_report.seed_observation(202),
        seedwise_report.seed_observation(707),
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

    target_fold3_raw_phi1_offset_share = _safe_share(
        target_slice.fold3_raw_phi1_center_projection,
        target_slice.fold3_inverse_pi_phi1_center_projection,
    )
    target_fold3_retained_inverse_pi_share = _safe_share(
        target_slice.fold3_weighted_phi1_center_projection,
        target_slice.fold3_inverse_pi_phi1_center_projection,
    )
    target_fold3_inverse_pi_lift_over_raw_phi1 = _safe_share(
        target_slice.fold3_inverse_pi_phi1_center_projection,
        target_slice.fold3_raw_phi1_center_projection,
    )
    target_exact_trim_floor_inverse_pi_share = _safe_share(
        target_slice.exact_trim_floor_inverse_pi_phi1_center_projection,
        target_slice.fold3_inverse_pi_phi1_center_projection,
    )
    target_exact_trim_floor_raw_phi1_offset_share = _safe_share(
        target_slice.exact_trim_floor_raw_phi1_center_projection,
        target_slice.exact_trim_floor_inverse_pi_phi1_center_projection,
    )
    target_exact_trim_floor_retained_inverse_pi_share = _safe_share(
        target_slice.exact_trim_floor_weighted_phi1_center_projection,
        target_slice.exact_trim_floor_inverse_pi_phi1_center_projection,
    )
    target_exact_trim_floor_weighted_share = _safe_share(
        target_slice.exact_trim_floor_weighted_phi1_center_projection,
        target_slice.fold3_weighted_phi1_center_projection,
    )
    driver_signature = _driver_signature(
        low_pi_fold3_treated_phi1_report=low_pi_fold3_treated_phi1_report,
        target_slice=target_slice,
        target_fold3_raw_phi1_offset_share=target_fold3_raw_phi1_offset_share,
        target_fold3_retained_inverse_pi_share=target_fold3_retained_inverse_pi_share,
        target_exact_trim_floor_inverse_pi_share=target_exact_trim_floor_inverse_pi_share,
        target_exact_trim_floor_retained_inverse_pi_share=(
            target_exact_trim_floor_retained_inverse_pi_share
        ),
        comparator_slices=comparator_slices,
    )

    exact_trim_floor_value = _canonical_design().trim_lower + 1e-6
    exact_trim_floor_line = (
        "- inside that fold-`3` slice, the exact trim-floor row "
        f"`pi_hat = {exact_trim_floor_value:.6f}` is absent in the current live replay: it contributes "
        f"`{_format_float(target_slice.exact_trim_floor_inverse_pi_phi1_center_projection)}` "
        "of the gross inverse-`pi_hat` conduit "
        f"(`{_format_percent(target_exact_trim_floor_inverse_pi_share)}` of fold-`3` gross lift) "
        "and `0.000` after weighting, so the live first-hop evidence should remain on the "
        "broader fold-`3` low-`pi_hat` conduit rather than an exact trim-floor row"
        if target_slice.exact_trim_floor_count == 0
        else "- inside that fold-`3` slice, the exact trim-floor row `pi_hat = "
        f"{exact_trim_floor_value:.6f}` already contributes "
        f"`{_format_float(target_slice.exact_trim_floor_inverse_pi_phi1_center_projection)}` of the gross inverse-`pi_hat` conduit "
        f"(`{_format_percent(target_exact_trim_floor_inverse_pi_share)}` of fold-`3` gross lift) and still lands at "
        f"`{_format_float(target_slice.exact_trim_floor_weighted_phi1_center_projection)}` after weighting because "
        f"`{_format_percent(target_exact_trim_floor_retained_inverse_pi_share)}` of its inverse-`pi_hat` lift survives once `(1-pi_hat)` is applied, so the trim floor is now a dominant propagation conduit rather than just a large row"
    )
    canonical_digest = (
        "- seed `303` / witness / fold `3` / `z = 0.15` turns the raw treated `phi1_hat` center projection "
        f"`{_format_float(target_slice.fold3_raw_phi1_center_projection)}` into a gross inverse-`pi_hat` conduit "
        f"`{_format_float(target_slice.fold3_inverse_pi_phi1_center_projection)}` under `phi1_hat / pi_hat`; subtracting the raw `phi1_hat` offset leaves "
        f"`{_format_float(target_slice.fold3_weighted_phi1_center_projection)}`, so "
        f"`{_format_percent(target_fold3_retained_inverse_pi_share)}` of the inverse-`pi_hat` lift survives the final treated propensity-floor propagation",
        exact_trim_floor_line,
        "- comparator seed `202` keeps fold-`3` gross inverse-`pi_hat` propagation negative at "
        f"`{_format_float(comparator_slices[0].fold3_inverse_pi_phi1_center_projection)}` despite a small positive raw `phi1_hat` center "
        f"`{_format_float(comparator_slices[0].fold3_raw_phi1_center_projection)}`, while residual seed `707` is both raw-negative "
        f"`{_format_float(comparator_slices[1].fold3_raw_phi1_center_projection)}` and gross-negative "
        f"`{_format_float(comparator_slices[1].fold3_inverse_pi_phi1_center_projection)}`; only seed `303` keeps a positive inverse-`pi_hat` conduit that survives the final `(1-pi_hat)` contraction",
        f"- current Trigger 2 implication: `{driver_signature}`; the next bounded repair should trace the fold-`3` low-`pi_hat` inverse-`pi_hat` conduit inside seed `303` before spending the queued residual slot on seed `707`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-seed303-low-pi-fold3-trim-"
            "floor-propensity-floor-trace"
        ),
        policy_digest=low_pi_fold3_treated_phi1_report.policy_digest,
        binding_design=low_pi_fold3_treated_phi1_report.binding_design,
        window_label=low_pi_fold3_treated_phi1_report.window_label,
        focus_random_states=(202, 303, 707),
        target_random_state=target_observation.random_state,
        target_seed_group=target_observation.seed_group,
        center_grid_value=low_pi_fold3_treated_phi1_report.center_grid_value,
        low_pi_threshold=low_pi_fold3_treated_phi1_report.low_pi_threshold,
        target_fold_id=_TARGET_FOLD_ID,
        target_fold3_low_pi_treated_count=target_slice.fold3_low_pi_treated_count,
        target_fold3_raw_phi1_center_projection=(
            target_slice.fold3_raw_phi1_center_projection
        ),
        target_fold3_inverse_pi_phi1_center_projection=(
            target_slice.fold3_inverse_pi_phi1_center_projection
        ),
        target_fold3_weighted_phi1_center_projection=(
            target_slice.fold3_weighted_phi1_center_projection
        ),
        target_fold3_raw_phi1_offset_share_of_inverse_pi=(
            target_fold3_raw_phi1_offset_share
        ),
        target_fold3_retained_inverse_pi_share_after_floor=(
            target_fold3_retained_inverse_pi_share
        ),
        target_fold3_inverse_pi_lift_over_raw_phi1=(
            target_fold3_inverse_pi_lift_over_raw_phi1
        ),
        target_exact_trim_floor_count=target_slice.exact_trim_floor_count,
        target_exact_trim_floor_value=exact_trim_floor_value,
        target_exact_trim_floor_raw_phi1_center_projection=(
            target_slice.exact_trim_floor_raw_phi1_center_projection
        ),
        target_exact_trim_floor_inverse_pi_phi1_center_projection=(
            target_slice.exact_trim_floor_inverse_pi_phi1_center_projection
        ),
        target_exact_trim_floor_weighted_phi1_center_projection=(
            target_slice.exact_trim_floor_weighted_phi1_center_projection
        ),
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            target_exact_trim_floor_inverse_pi_share
        ),
        target_exact_trim_floor_raw_phi1_offset_share_of_trim_inverse_pi=(
            target_exact_trim_floor_raw_phi1_offset_share
        ),
        target_exact_trim_floor_retained_inverse_pi_share_after_floor=(
            target_exact_trim_floor_retained_inverse_pi_share
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            target_exact_trim_floor_weighted_share
        ),
        comparator_random_states=tuple(
            observation.random_state for observation in comparator_observations
        ),
        comparator_fold3_raw_phi1_center_projections=tuple(
            slice_.fold3_raw_phi1_center_projection for slice_ in comparator_slices
        ),
        comparator_fold3_inverse_pi_phi1_center_projections=tuple(
            slice_.fold3_inverse_pi_phi1_center_projection
            for slice_ in comparator_slices
        ),
        comparator_fold3_weighted_phi1_center_projections=tuple(
            slice_.fold3_weighted_phi1_center_projection for slice_ in comparator_slices
        ),
        driver_signature=driver_signature,
        canonical_seed303_low_pi_fold3_trim_floor_propensity_floor_trace_digest=(
            canonical_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace_report()
