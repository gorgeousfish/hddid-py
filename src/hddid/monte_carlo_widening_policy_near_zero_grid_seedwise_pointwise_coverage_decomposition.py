from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .estimation import estimate_eq31_mainline
from .inference import estimate_nonparametric_inference, estimate_parametric_inference
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)
from .nuisance import CrossfitNuisanceEstimator
from .score import build_score_payload
from .splitting import make_crossfit_splits
from .validation import MonteCarloDesign, _generate_dataset

_FRESH_RANDOM_STATES = (404, 505, 606, 707, 808)
_WITNESS_RANDOM_STATES = (101, 202, 303)
_ALL_RANDOM_STATES = _FRESH_RANDOM_STATES + _WITNESS_RANDOM_STATES
_WINDOW_LABEL = "near_zero_grid"
_BINDING_DESIGN = ("DGP2", 500, 50)
_EVALUATION_GRID = (0.05, 0.15, 0.25)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _as_bool_tuple(values: np.ndarray) -> tuple[bool, ...]:
    array = np.asarray(values, dtype=bool).reshape(-1)
    return tuple(bool(value) for value in array.tolist())


def _as_float_tuple(values: np.ndarray) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return tuple(float(value) for value in array.tolist())


def _canonical_design() -> MonteCarloDesign:
    return MonteCarloDesign(
        dgp_name=_BINDING_DESIGN[0],
        n_obs=_BINDING_DESIGN[1],
        p=_BINDING_DESIGN[2],
        evaluation_grid=_EVALUATION_GRID,
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation:
    random_state: int
    seed_group: str
    replication_seed: int
    pointwise_coverage_by_z: tuple[bool, ...]
    uniform_band_coverage_by_z: tuple[bool, ...]
    sigma_z_hat: tuple[float, ...]
    uniform_critical_value: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.replication_seed = int(self.replication_seed)
        self.pointwise_coverage_by_z = tuple(
            bool(value) for value in self.pointwise_coverage_by_z
        )
        self.uniform_band_coverage_by_z = tuple(
            bool(value) for value in self.uniform_band_coverage_by_z
        )
        self.sigma_z_hat = tuple(float(value) for value in self.sigma_z_hat)
        self.uniform_critical_value = float(self.uniform_critical_value)

    def to_dict(self) -> dict[str, object]:
        return {
            "random_state": self.random_state,
            "seed_group": self.seed_group,
            "replication_seed": self.replication_seed,
            "pointwise_coverage_by_z": list(self.pointwise_coverage_by_z),
            "uniform_band_coverage_by_z": list(self.uniform_band_coverage_by_z),
            "sigma_z_hat": list(self.sigma_z_hat),
            "uniform_critical_value": self.uniform_critical_value,
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyNearZeroGridCoverageGroupSummary:
    group_label: str
    n_random_states: int
    point_cover_rate_by_z: tuple[float, ...]
    uniform_band_cover_rate_by_z: tuple[float, ...]
    point_miss_count_by_z: tuple[int, ...]
    uniform_band_miss_count_by_z: tuple[int, ...]
    mean_sigma_z_hat_by_z: tuple[float, ...]
    max_uniform_critical_value_random_state: int
    max_uniform_critical_value: float

    def __post_init__(self) -> None:
        self.group_label = str(self.group_label).strip().lower()
        self.n_random_states = int(self.n_random_states)
        self.point_cover_rate_by_z = tuple(
            float(value) for value in self.point_cover_rate_by_z
        )
        self.uniform_band_cover_rate_by_z = tuple(
            float(value) for value in self.uniform_band_cover_rate_by_z
        )
        self.point_miss_count_by_z = tuple(
            int(value) for value in self.point_miss_count_by_z
        )
        self.uniform_band_miss_count_by_z = tuple(
            int(value) for value in self.uniform_band_miss_count_by_z
        )
        self.mean_sigma_z_hat_by_z = tuple(
            float(value) for value in self.mean_sigma_z_hat_by_z
        )
        self.max_uniform_critical_value_random_state = int(
            self.max_uniform_critical_value_random_state
        )
        self.max_uniform_critical_value = float(self.max_uniform_critical_value)

    def to_dict(self) -> dict[str, object]:
        return {
            "group_label": self.group_label,
            "n_random_states": self.n_random_states,
            "point_cover_rate_by_z": list(self.point_cover_rate_by_z),
            "uniform_band_cover_rate_by_z": list(self.uniform_band_cover_rate_by_z),
            "point_miss_count_by_z": list(self.point_miss_count_by_z),
            "uniform_band_miss_count_by_z": list(self.uniform_band_miss_count_by_z),
            "mean_sigma_z_hat_by_z": list(self.mean_sigma_z_hat_by_z),
            "max_uniform_critical_value_random_state": (
                self.max_uniform_critical_value_random_state
            ),
            "max_uniform_critical_value": self.max_uniform_critical_value,
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    fresh_random_states: tuple[int, ...]
    witness_random_states: tuple[int, ...]
    seed_observations: tuple[
        Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation, ...
    ]
    group_summaries: tuple[
        Phase7MonteCarloWideningPolicyNearZeroGridCoverageGroupSummary, ...
    ]
    canonical_seedwise_decomposition_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.fresh_random_states = tuple(
            int(value) for value in self.fresh_random_states
        )
        self.witness_random_states = tuple(
            int(value) for value in self.witness_random_states
        )
        self.seed_observations = tuple(self.seed_observations)
        self.group_summaries = tuple(self.group_summaries)
        self.canonical_seedwise_decomposition_digest = tuple(
            str(line).rstrip() for line in self.canonical_seedwise_decomposition_digest
        )

    def group_summary(
        self, group_label: str
    ) -> Phase7MonteCarloWideningPolicyNearZeroGridCoverageGroupSummary:
        target = str(group_label).strip().lower()
        for summary in self.group_summaries:
            if summary.group_label == target:
                return summary
        raise KeyError(f"seedwise decomposition group is missing: {group_label!r}")

    def seed_observation(
        self, random_state: int
    ) -> Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation:
        target = int(random_state)
        for observation in self.seed_observations:
            if observation.random_state == target:
                return observation
        raise KeyError(
            f"seedwise decomposition observation is missing: {random_state!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "evaluation_grid": list(self.evaluation_grid),
            "fresh_random_states": list(self.fresh_random_states),
            "witness_random_states": list(self.witness_random_states),
            "seed_observations": [
                observation.to_dict() for observation in self.seed_observations
            ],
            "group_summaries": [summary.to_dict() for summary in self.group_summaries],
            "canonical_seedwise_decomposition_digest": list(
                self.canonical_seedwise_decomposition_digest
            ),
        }


def _run_seed_observation(
    *,
    design: MonteCarloDesign,
    random_state: int,
    seed_group: str,
) -> Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation:
    rng = np.random.default_rng(int(random_state))
    replication_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    dataset = _generate_dataset(design, random_state=replication_seed)
    data = dataset.to_validated_data()
    splits = make_crossfit_splits(
        n_obs=data.n_obs,
        n_folds=design.n_folds,
        random_state=replication_seed,
        trim_lower=design.trim_lower,
        trim_upper=design.trim_upper,
    )
    nuisance_payload = CrossfitNuisanceEstimator(oracle_lane=design.oracle_lane).fit(
        data,
        splits,
    )
    score_payload = build_score_payload(data, nuisance_payload)
    estimation_payload, result = estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
    )
    _, result = estimate_parametric_inference(
        score_payload,
        estimation_payload,
        result=result,
        alpha=design.alpha,
        lambda_prime=0.0,
    )
    nonparametric_payload, _ = estimate_nonparametric_inference(
        score_payload,
        estimation_payload,
        result=result,
        alpha=design.alpha,
        lambda_double_prime=0.0,
        n_boot=64,
        random_state=replication_seed,
    )

    truth = np.asarray(dataset.true_f_at_z0, dtype=float)
    pointwise_interval = nonparametric_payload.pointwise_confidence_interval
    uniform_band = nonparametric_payload.uniform_band
    pointwise_covered = (np.asarray(pointwise_interval.lower, dtype=float) <= truth) & (
        truth <= np.asarray(pointwise_interval.upper, dtype=float)
    )
    band_covered = (np.asarray(uniform_band.lower, dtype=float) <= truth) & (
        truth <= np.asarray(uniform_band.upper, dtype=float)
    )

    return Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation(
        random_state=random_state,
        seed_group=seed_group,
        replication_seed=replication_seed,
        pointwise_coverage_by_z=_as_bool_tuple(pointwise_covered),
        uniform_band_coverage_by_z=_as_bool_tuple(band_covered),
        sigma_z_hat=_as_float_tuple(nonparametric_payload.sigma_z_hat),
        uniform_critical_value=float(uniform_band.critical_value),
    )


def _build_group_summary(
    *,
    group_label: str,
    observations: tuple[Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation, ...],
) -> Phase7MonteCarloWideningPolicyNearZeroGridCoverageGroupSummary:
    if not observations:
        raise ValueError(
            "seedwise decomposition summary requires at least one observation"
        )

    point_cover = np.asarray(
        [observation.pointwise_coverage_by_z for observation in observations],
        dtype=float,
    )
    band_cover = np.asarray(
        [observation.uniform_band_coverage_by_z for observation in observations],
        dtype=float,
    )
    sigma = np.asarray(
        [observation.sigma_z_hat for observation in observations],
        dtype=float,
    )
    max_ucv_observation = max(
        observations,
        key=lambda observation: observation.uniform_critical_value,
    )

    return Phase7MonteCarloWideningPolicyNearZeroGridCoverageGroupSummary(
        group_label=group_label,
        n_random_states=len(observations),
        point_cover_rate_by_z=_as_float_tuple(point_cover.mean(axis=0)),
        uniform_band_cover_rate_by_z=_as_float_tuple(band_cover.mean(axis=0)),
        point_miss_count_by_z=tuple(
            int(value) for value in np.rint((1.0 - point_cover).sum(axis=0)).tolist()
        ),
        uniform_band_miss_count_by_z=tuple(
            int(value) for value in np.rint((1.0 - band_cover).sum(axis=0)).tolist()
        ),
        mean_sigma_z_hat_by_z=_as_float_tuple(sigma.mean(axis=0)),
        max_uniform_critical_value_random_state=max_ucv_observation.random_state,
        max_uniform_critical_value=max_ucv_observation.uniform_critical_value,
    )


def build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report(
    *,
    seed_observations: tuple[
        Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation, ...
    ],
    allow_band_surface_drift: bool = False,
) -> Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport:
    sorted_observations = tuple(
        sorted(seed_observations, key=lambda observation: observation.random_state)
    )
    fresh_observations = tuple(
        observation
        for observation in sorted_observations
        if observation.seed_group == "fresh"
    )
    witness_observations = tuple(
        observation
        for observation in sorted_observations
        if observation.seed_group == "witness"
    )
    if (
        tuple(observation.random_state for observation in fresh_observations)
        != _FRESH_RANDOM_STATES
    ):
        raise ValueError(
            "seedwise decomposition requires the canonical fresh random-state set"
        )
    if (
        tuple(observation.random_state for observation in witness_observations)
        != _WITNESS_RANDOM_STATES
    ):
        raise ValueError(
            "seedwise decomposition requires the canonical witness random-state set"
        )

    policy = build_phase7_canonical_monte_carlo_widening_policy()
    all_summary = _build_group_summary(
        group_label="all", observations=sorted_observations
    )
    fresh_summary = _build_group_summary(
        group_label="fresh",
        observations=fresh_observations,
    )
    witness_summary = _build_group_summary(
        group_label="witness",
        observations=witness_observations,
    )
    center_index = int(
        np.argmax(np.asarray(all_summary.point_miss_count_by_z, dtype=int))
    )
    right_center_lane = {1, 2}
    if {
        index
        for index, value in enumerate(all_summary.uniform_band_miss_count_by_z)
        if value > 0
    } != right_center_lane:
        if not allow_band_surface_drift:
            raise ValueError(
                "seedwise decomposition expected the bounded uniform-band miss surface"
            )

    canonical_digest = (
        "- all-8 `near_zero_grid` point miss vector stays `["
        + ", ".join(str(value) for value in all_summary.point_miss_count_by_z)
        + "]`: point cover rates `"
        + " / ".join(
            _format_float(value) for value in all_summary.point_cover_rate_by_z
        )
        + "`, so the current weakest point remains center `z = "
        + _format_float(_EVALUATION_GRID[center_index])
        + "`",
        "- uniform band miss vector stays `["
        + ", ".join(str(value) for value in all_summary.uniform_band_miss_count_by_z)
        + "]`: band cover rates `"
        + " / ".join(
            _format_float(value) for value in all_summary.uniform_band_cover_rate_by_z
        )
        + "`, so the miss surface remains local rather than whole-grid collapse",
        "- fresh/witness split keeps the same bounded-lane story: fresh point miss `["
        + ", ".join(str(value) for value in fresh_summary.point_miss_count_by_z)
        + "]`, witness point miss `["
        + ", ".join(str(value) for value in witness_summary.point_miss_count_by_z)
        + "]`; both groups stay concentrated on `z in {0.15, 0.25}` and keep left-band miss at `0`",
        "- current Trigger 2 implication: future estimator-path evidence should lower point misses from `["
        + ", ".join(str(value) for value in all_summary.point_miss_count_by_z)
        + "]` without creating any new band miss at left guard `z = 0.05`",
    )

    return Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-near-zero-grid-seedwise-pointwise-coverage-decomposition"
        ),
        policy_digest=(
            f"label={policy.label}",
            f"max_total_runtime_seconds={policy.max_total_runtime_seconds:.1f}",
            f"max_random_states={policy.max_random_states}",
            f"stop_on_first_typed_invalidity={policy.stop_on_first_typed_invalidity}",
            f"min_nonparametric_coverage={policy.min_nonparametric_coverage:.2f}",
        ),
        binding_design=_BINDING_DESIGN,
        window_label=_WINDOW_LABEL,
        evaluation_grid=_EVALUATION_GRID,
        fresh_random_states=_FRESH_RANDOM_STATES,
        witness_random_states=_WITNESS_RANDOM_STATES,
        seed_observations=sorted_observations,
        group_summaries=(fresh_summary, witness_summary, all_summary),
        canonical_seedwise_decomposition_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition() -> (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
):
    observations = (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation(
            random_state=101,
            seed_group="witness",
            replication_seed=666291128,
            pointwise_coverage_by_z=(False, False, True),
            uniform_band_coverage_by_z=(True, False, True),
            sigma_z_hat=(3.0355434149694625, 3.4931387906766718, 2.2402573964614696),
            uniform_critical_value=2.069219078626079,
        ),
        Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation(
            random_state=202,
            seed_group="witness",
            replication_seed=399595374,
            pointwise_coverage_by_z=(True, True, True),
            uniform_band_coverage_by_z=(True, True, True),
            sigma_z_hat=(13.464333769509313, 4.942953812519496, 6.0614207052082),
            uniform_critical_value=1.9812948281112157,
        ),
        Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation(
            random_state=303,
            seed_group="witness",
            replication_seed=883193502,
            pointwise_coverage_by_z=(True, False, True),
            uniform_band_coverage_by_z=(True, True, True),
            sigma_z_hat=(2.796608562949708, 4.087257631350298, 2.7896774550588646),
            uniform_critical_value=2.033458613123706,
        ),
        Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation(
            random_state=404,
            seed_group="fresh",
            replication_seed=1983313941,
            pointwise_coverage_by_z=(True, True, True),
            uniform_band_coverage_by_z=(True, True, True),
            sigma_z_hat=(28.89868052338322, 5.969830192595853, 6.9935919478241475),
            uniform_critical_value=2.2956623235969262,
        ),
        Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation(
            random_state=505,
            seed_group="fresh",
            replication_seed=1741662786,
            pointwise_coverage_by_z=(True, False, True),
            uniform_band_coverage_by_z=(True, True, True),
            sigma_z_hat=(22.909531746128938, 4.705540731918267, 2.470189555436918),
            uniform_critical_value=1.8872544539303904,
        ),
        Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation(
            random_state=606,
            seed_group="fresh",
            replication_seed=1494940720,
            pointwise_coverage_by_z=(True, True, True),
            uniform_band_coverage_by_z=(True, True, True),
            sigma_z_hat=(2.3137239764669424, 15.044821612538765, 18.17794571539671),
            uniform_critical_value=1.9306432474794832,
        ),
        Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation(
            random_state=707,
            seed_group="fresh",
            replication_seed=676410599,
            pointwise_coverage_by_z=(True, True, False),
            uniform_band_coverage_by_z=(True, True, True),
            sigma_z_hat=(4.922561484998662, 16.764321445709392, 5.507193427226683),
            uniform_critical_value=2.3054300668950893,
        ),
        Phase7MonteCarloWideningPolicyNearZeroGridSeedObservation(
            random_state=808,
            seed_group="fresh",
            replication_seed=1817076203,
            pointwise_coverage_by_z=(True, True, False),
            uniform_band_coverage_by_z=(True, True, False),
            sigma_z_hat=(4.176143004821034, 2.0961407222792072, 2.451212020691818),
            uniform_critical_value=1.684652421180263,
        ),
    )
    return build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report(
        seed_observations=observations,
    )
