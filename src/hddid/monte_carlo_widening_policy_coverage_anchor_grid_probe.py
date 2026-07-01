from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)
from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy
from .validation import Phase7NonparametricCalibrationObjectSlice


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_grid_value(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return f"{value:.1f}"
    text = f"{value:.2f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def _format_grid(values: np.ndarray) -> str:
    return "[" + ", ".join(_format_grid_value(value) for value in values) + "]"


def _slice_design_key(
    object_slice: Phase7NonparametricCalibrationObjectSlice,
) -> tuple[str, int, int]:
    return (
        str(object_slice.dgp_name).strip().upper(),
        int(object_slice.n_obs),
        int(object_slice.p),
    )


def _coverage_count(object_slice: Phase7NonparametricCalibrationObjectSlice) -> int:
    return int(np.count_nonzero(object_slice.pointwise_coverage))


def _grid_width(object_slice: Phase7NonparametricCalibrationObjectSlice) -> float:
    return float(
        np.max(object_slice.evaluation_grid) - np.min(object_slice.evaluation_grid)
    )


def _uniform_critical_value(
    object_slice: Phase7NonparametricCalibrationObjectSlice,
) -> float:
    if object_slice.uniform_critical_value is None:
        raise ValueError(
            "coverage anchor grid probe requires uniform critical values on every slice"
        )
    return float(object_slice.uniform_critical_value)


def _require_matching_designs(
    left: Phase7NonparametricCalibrationObjectSlice,
    right: Phase7NonparametricCalibrationObjectSlice,
) -> None:
    if _slice_design_key(left) != _slice_design_key(right):
        raise ValueError("coverage anchor grid probe requires matching binding designs")


def _require_matching_grid(
    left: Phase7NonparametricCalibrationObjectSlice,
    right: Phase7NonparametricCalibrationObjectSlice,
) -> None:
    if not np.allclose(left.evaluation_grid, right.evaluation_grid):
        raise ValueError(
            "coverage anchor grid probe requires matching evaluation grids"
        )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorGridReplayInput:
    grid_label: str
    coverage_anchor_slice: Phase7NonparametricCalibrationObjectSlice
    overshoot_companion_slice: Phase7NonparametricCalibrationObjectSlice

    def __post_init__(self) -> None:
        self.grid_label = str(self.grid_label).strip()
        if not self.grid_label:
            raise ValueError("grid_label must not be empty")
        _require_matching_designs(
            self.coverage_anchor_slice, self.overshoot_companion_slice
        )
        _require_matching_grid(
            self.coverage_anchor_slice, self.overshoot_companion_slice
        )

    @property
    def binding_design(self) -> tuple[str, int, int]:
        return _slice_design_key(self.coverage_anchor_slice)

    @property
    def evaluation_grid(self) -> tuple[float, ...]:
        return tuple(
            float(value) for value in self.coverage_anchor_slice.evaluation_grid
        )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay:
    grid_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_slice: Phase7NonparametricCalibrationObjectSlice
    overshoot_companion_slice: Phase7NonparametricCalibrationObjectSlice
    coverage_anchor_coverage_count_delta: int
    coverage_anchor_mean_absolute_error_delta: float
    coverage_anchor_mean_sigma_z_hat_delta: float
    coverage_anchor_mean_pointwise_interval_length_delta: float
    coverage_anchor_uniform_critical_value_delta: float
    overshoot_companion_coverage_count_delta: int
    overshoot_companion_mean_absolute_error_delta: float
    overshoot_companion_mean_sigma_z_hat_delta: float
    overshoot_companion_mean_pointwise_interval_length_delta: float
    overshoot_companion_uniform_critical_value_delta: float

    def __post_init__(self) -> None:
        self.grid_label = str(self.grid_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.coverage_anchor_coverage_count_delta = int(
            self.coverage_anchor_coverage_count_delta
        )
        self.coverage_anchor_mean_absolute_error_delta = float(
            self.coverage_anchor_mean_absolute_error_delta
        )
        self.coverage_anchor_mean_sigma_z_hat_delta = float(
            self.coverage_anchor_mean_sigma_z_hat_delta
        )
        self.coverage_anchor_mean_pointwise_interval_length_delta = float(
            self.coverage_anchor_mean_pointwise_interval_length_delta
        )
        self.coverage_anchor_uniform_critical_value_delta = float(
            self.coverage_anchor_uniform_critical_value_delta
        )
        self.overshoot_companion_coverage_count_delta = int(
            self.overshoot_companion_coverage_count_delta
        )
        self.overshoot_companion_mean_absolute_error_delta = float(
            self.overshoot_companion_mean_absolute_error_delta
        )
        self.overshoot_companion_mean_sigma_z_hat_delta = float(
            self.overshoot_companion_mean_sigma_z_hat_delta
        )
        self.overshoot_companion_mean_pointwise_interval_length_delta = float(
            self.overshoot_companion_mean_pointwise_interval_length_delta
        )
        self.overshoot_companion_uniform_critical_value_delta = float(
            self.overshoot_companion_uniform_critical_value_delta
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "grid_label": self.grid_label,
            "evaluation_grid": list(self.evaluation_grid),
            "coverage_anchor_slice": self.coverage_anchor_slice.to_dict(),
            "overshoot_companion_slice": self.overshoot_companion_slice.to_dict(),
            "coverage_anchor_coverage_count_delta": (
                self.coverage_anchor_coverage_count_delta
            ),
            "coverage_anchor_mean_absolute_error_delta": (
                self.coverage_anchor_mean_absolute_error_delta
            ),
            "coverage_anchor_mean_sigma_z_hat_delta": (
                self.coverage_anchor_mean_sigma_z_hat_delta
            ),
            "coverage_anchor_mean_pointwise_interval_length_delta": (
                self.coverage_anchor_mean_pointwise_interval_length_delta
            ),
            "coverage_anchor_uniform_critical_value_delta": (
                self.coverage_anchor_uniform_critical_value_delta
            ),
            "overshoot_companion_coverage_count_delta": (
                self.overshoot_companion_coverage_count_delta
            ),
            "overshoot_companion_mean_absolute_error_delta": (
                self.overshoot_companion_mean_absolute_error_delta
            ),
            "overshoot_companion_mean_sigma_z_hat_delta": (
                self.overshoot_companion_mean_sigma_z_hat_delta
            ),
            "overshoot_companion_mean_pointwise_interval_length_delta": (
                self.overshoot_companion_mean_pointwise_interval_length_delta
            ),
            "overshoot_companion_uniform_critical_value_delta": (
                self.overshoot_companion_uniform_critical_value_delta
            ),
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorGridReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_current_slice: Phase7NonparametricCalibrationObjectSlice
    overshoot_companion_current_slice: Phase7NonparametricCalibrationObjectSlice
    replays: tuple[Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay, ...]
    full_coverage_witness_labels: tuple[str, ...]
    recommended_grid_label: str
    recommended_grid: tuple[float, ...]
    canonical_coverage_anchor_grid_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.replays = tuple(self.replays)
        self.full_coverage_witness_labels = tuple(
            str(label).strip() for label in self.full_coverage_witness_labels
        )
        self.recommended_grid_label = str(self.recommended_grid_label).strip()
        self.recommended_grid = tuple(float(value) for value in self.recommended_grid)
        self.canonical_coverage_anchor_grid_digest = tuple(
            str(line).rstrip() for line in self.canonical_coverage_anchor_grid_digest
        )

    def replay(
        self, grid_label: str
    ) -> Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay:
        target = str(grid_label).strip()
        for replay in self.replays:
            if replay.grid_label == target:
                return replay
        raise KeyError(f"coverage anchor grid replay not present: {target!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_current_slice": (
                self.coverage_anchor_current_slice.to_dict()
            ),
            "overshoot_companion_current_slice": (
                self.overshoot_companion_current_slice.to_dict()
            ),
            "replays": [replay.to_dict() for replay in self.replays],
            "full_coverage_witness_labels": list(self.full_coverage_witness_labels),
            "recommended_grid_label": self.recommended_grid_label,
            "recommended_grid": list(self.recommended_grid),
            "canonical_coverage_anchor_grid_digest": list(
                self.canonical_coverage_anchor_grid_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_grid_report(
    *,
    coverage_anchor_current_slice: Phase7NonparametricCalibrationObjectSlice,
    overshoot_companion_current_slice: Phase7NonparametricCalibrationObjectSlice,
    replays: tuple[
        Phase7MonteCarloWideningPolicyCoverageAnchorGridReplayInput, ...
    ] = (),
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorGridReport:
    resolved_policy = (
        build_phase7_canonical_monte_carlo_widening_policy()
        if policy is None
        else policy
    )
    _require_matching_designs(
        coverage_anchor_current_slice, overshoot_companion_current_slice
    )
    _require_matching_grid(
        coverage_anchor_current_slice, overshoot_companion_current_slice
    )
    binding_design = _slice_design_key(coverage_anchor_current_slice)

    current_anchor_coverage_count = _coverage_count(coverage_anchor_current_slice)
    current_overshoot_coverage_count = _coverage_count(
        overshoot_companion_current_slice
    )
    current_anchor_uniform_critical_value = _uniform_critical_value(
        coverage_anchor_current_slice
    )
    current_overshoot_uniform_critical_value = _uniform_critical_value(
        overshoot_companion_current_slice
    )

    replay_reports: list[Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay] = []
    full_coverage_witness_labels: list[str] = []
    partial_witness_fragments: list[str] = []
    for replay_input in replays:
        if replay_input.binding_design != binding_design:
            raise ValueError(
                "coverage anchor grid probe requires matching binding designs"
            )
        _require_matching_grid(
            replay_input.coverage_anchor_slice,
            replay_input.overshoot_companion_slice,
        )

        coverage_anchor_coverage_count = _coverage_count(
            replay_input.coverage_anchor_slice
        )
        overshoot_companion_coverage_count = _coverage_count(
            replay_input.overshoot_companion_slice
        )
        coverage_anchor_grid_size = int(
            replay_input.coverage_anchor_slice.evaluation_grid.shape[0]
        )
        overshoot_companion_grid_size = int(
            replay_input.overshoot_companion_slice.evaluation_grid.shape[0]
        )

        replay_report = Phase7MonteCarloWideningPolicyCoverageAnchorGridReplay(
            grid_label=replay_input.grid_label,
            evaluation_grid=replay_input.evaluation_grid,
            coverage_anchor_slice=replay_input.coverage_anchor_slice,
            overshoot_companion_slice=replay_input.overshoot_companion_slice,
            coverage_anchor_coverage_count_delta=(
                coverage_anchor_coverage_count - current_anchor_coverage_count
            ),
            coverage_anchor_mean_absolute_error_delta=(
                replay_input.coverage_anchor_slice.mean_absolute_error
                - coverage_anchor_current_slice.mean_absolute_error
            ),
            coverage_anchor_mean_sigma_z_hat_delta=(
                replay_input.coverage_anchor_slice.mean_sigma_z_hat
                - coverage_anchor_current_slice.mean_sigma_z_hat
            ),
            coverage_anchor_mean_pointwise_interval_length_delta=(
                replay_input.coverage_anchor_slice.mean_pointwise_interval_length
                - coverage_anchor_current_slice.mean_pointwise_interval_length
            ),
            coverage_anchor_uniform_critical_value_delta=(
                _uniform_critical_value(replay_input.coverage_anchor_slice)
                - current_anchor_uniform_critical_value
            ),
            overshoot_companion_coverage_count_delta=(
                overshoot_companion_coverage_count - current_overshoot_coverage_count
            ),
            overshoot_companion_mean_absolute_error_delta=(
                replay_input.overshoot_companion_slice.mean_absolute_error
                - overshoot_companion_current_slice.mean_absolute_error
            ),
            overshoot_companion_mean_sigma_z_hat_delta=(
                replay_input.overshoot_companion_slice.mean_sigma_z_hat
                - overshoot_companion_current_slice.mean_sigma_z_hat
            ),
            overshoot_companion_mean_pointwise_interval_length_delta=(
                replay_input.overshoot_companion_slice.mean_pointwise_interval_length
                - overshoot_companion_current_slice.mean_pointwise_interval_length
            ),
            overshoot_companion_uniform_critical_value_delta=(
                _uniform_critical_value(replay_input.overshoot_companion_slice)
                - current_overshoot_uniform_critical_value
            ),
        )
        replay_reports.append(replay_report)

        if (
            coverage_anchor_coverage_count == coverage_anchor_grid_size
            and overshoot_companion_coverage_count == overshoot_companion_grid_size
        ):
            full_coverage_witness_labels.append(replay_input.grid_label)
        else:
            partial_witness_fragments.append(
                f"`{replay_input.grid_label}` only lifts coverage to "
                f"`{coverage_anchor_coverage_count}/{coverage_anchor_grid_size}`"
            )

    if not full_coverage_witness_labels:
        raise ValueError(
            "coverage anchor grid probe requires at least one full-coverage witness"
        )

    recommended_replay = min(
        (
            replay
            for replay in replay_reports
            if replay.grid_label in full_coverage_witness_labels
        ),
        key=lambda replay: (
            _grid_width(replay.coverage_anchor_slice),
            replay.coverage_anchor_slice.mean_sigma_z_hat,
            replay.coverage_anchor_slice.mean_absolute_error,
            replay.grid_label,
        ),
    )

    witness_line = (
        "- full-coverage witness grids for coverage anchor seed "
        f"`{coverage_anchor_current_slice.random_state}`: "
        + ", ".join(f"`{label}`" for label in full_coverage_witness_labels)
    )
    if partial_witness_fragments:
        witness_line += "; " + "; ".join(partial_witness_fragments)

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*binding_design)}`: coverage anchor seed "
        f"`{coverage_anchor_current_slice.random_state}` stays "
        f"`{current_anchor_coverage_count}/"
        f"{coverage_anchor_current_slice.evaluation_grid.shape[0]}` on current grid "
        f"`{_format_grid(coverage_anchor_current_slice.evaluation_grid)}` with mean "
        "absolute error "
        f"`{_format_float(coverage_anchor_current_slice.mean_absolute_error)}`, "
        "average `sigma_z_hat` "
        f"`{_format_float(coverage_anchor_current_slice.mean_sigma_z_hat)}`, "
        "interval length "
        f"`{_format_float(coverage_anchor_current_slice.mean_pointwise_interval_length)}`; "
        "overshoot companion seed "
        f"`{overshoot_companion_current_slice.random_state}` already stays "
        f"`{current_overshoot_coverage_count}/"
        f"{overshoot_companion_current_slice.evaluation_grid.shape[0]}`",
        witness_line,
        "- narrowest full-coverage witness "
        f"`{recommended_replay.grid_label}` "
        f"(`{_format_grid(recommended_replay.coverage_anchor_slice.evaluation_grid)}`) "
        f"lowers seed `{coverage_anchor_current_slice.random_state}` mean absolute "
        "error by "
        f"`{_format_float(-recommended_replay.coverage_anchor_mean_absolute_error_delta)}`, "
        "average `sigma_z_hat` by "
        f"`{_format_float(-recommended_replay.coverage_anchor_mean_sigma_z_hat_delta)}`, "
        "and interval length by "
        f"`{_format_float(-recommended_replay.coverage_anchor_mean_pointwise_interval_length_delta)}` "
        "while uniform critical value only moves by "
        f"`{_format_signed(recommended_replay.coverage_anchor_uniform_critical_value_delta)}`",
        "- the same "
        f"`{recommended_replay.grid_label}` keeps overshoot companion seed "
        f"`{overshoot_companion_current_slice.random_state}` at "
        f"`{current_overshoot_coverage_count + recommended_replay.overshoot_companion_coverage_count_delta}/"
        f"{recommended_replay.overshoot_companion_slice.evaluation_grid.shape[0]}` "
        "and lowers average `sigma_z_hat` by "
        f"`{_format_float(-recommended_replay.overshoot_companion_mean_sigma_z_hat_delta)}`; "
        "next Trigger 2 follow-up should stay on coverage-anchor "
        "evaluation-grid calibration around `z = "
        f"{_format_grid_value(recommended_replay.evaluation_grid[len(recommended_replay.evaluation_grid) // 2])}`, "
        "not on broader runtime or quantile tuning",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorGridReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-grid-probe",
        policy_digest=resolved_policy.to_digest(),
        binding_design=binding_design,
        coverage_anchor_current_slice=coverage_anchor_current_slice,
        overshoot_companion_current_slice=overshoot_companion_current_slice,
        replays=tuple(replay_reports),
        full_coverage_witness_labels=tuple(full_coverage_witness_labels),
        recommended_grid_label=recommended_replay.grid_label,
        recommended_grid=recommended_replay.evaluation_grid,
        canonical_coverage_anchor_grid_digest=canonical_digest,
    )


def _make_slice(
    *,
    random_state: int,
    replication_seed: int,
    evaluation_grid: tuple[float, float, float],
    pointwise_coverage: tuple[bool, bool, bool],
    mean_absolute_error: float,
    mean_sigma_z_hat: float,
    mean_pointwise_interval_length: float,
    uniform_critical_value: float,
) -> Phase7NonparametricCalibrationObjectSlice:
    grid = np.asarray(evaluation_grid, dtype=float)
    absolute_error_at_z0 = np.full(grid.shape[0], mean_absolute_error, dtype=float)
    sigma_z_hat = np.full(grid.shape[0], mean_sigma_z_hat, dtype=float)
    pointwise_interval_length = np.full(
        grid.shape[0], mean_pointwise_interval_length, dtype=float
    )
    pointwise_coverage_array = np.asarray(pointwise_coverage, dtype=bool)
    return Phase7NonparametricCalibrationObjectSlice(
        dgp_name="DGP2",
        n_obs=500,
        p=50,
        random_state=random_state,
        replication_seed=replication_seed,
        evaluation_grid=grid,
        true_f_at_z0=np.zeros(grid.shape[0], dtype=float),
        bar_f_at_z0=absolute_error_at_z0.copy(),
        absolute_error_at_z0=absolute_error_at_z0,
        sigma_z_hat=sigma_z_hat,
        pointwise_interval_length=pointwise_interval_length,
        pointwise_coverage=pointwise_coverage_array,
        mean_absolute_error=mean_absolute_error,
        max_absolute_error_grid_value=float(grid[0]),
        max_absolute_error=mean_absolute_error,
        mean_sigma_z_hat=mean_sigma_z_hat,
        max_sigma_z_hat_grid_value=float(grid[0]),
        max_sigma_z_hat=mean_sigma_z_hat,
        mean_pointwise_interval_length=mean_pointwise_interval_length,
        max_pointwise_interval_grid_value=float(grid[0]),
        max_pointwise_interval_length=mean_pointwise_interval_length,
        uniform_critical_value=uniform_critical_value,
        mean_uniform_band_length=uniform_critical_value * mean_sigma_z_hat,
    )


def _build_canonical_report(
    policy: Phase7MonteCarloWideningPolicy,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorGridReport:
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_grid_report(
        coverage_anchor_current_slice=_make_slice(
            random_state=202,
            replication_seed=743646743,
            evaluation_grid=(-1.0, 0.0, 1.0),
            pointwise_coverage=(False, False, False),
            mean_absolute_error=13.636,
            mean_sigma_z_hat=6.770,
            mean_pointwise_interval_length=22.270,
            uniform_critical_value=1.370,
        ),
        overshoot_companion_current_slice=_make_slice(
            random_state=505,
            replication_seed=1826269414,
            evaluation_grid=(-1.0, 0.0, 1.0),
            pointwise_coverage=(True, True, True),
            mean_absolute_error=16.590,
            mean_sigma_z_hat=13.197,
            mean_pointwise_interval_length=43.414,
            uniform_critical_value=2.072,
        ),
        replays=(
            Phase7MonteCarloWideningPolicyCoverageAnchorGridReplayInput(
                grid_label="contrast_grid",
                coverage_anchor_slice=_make_slice(
                    random_state=202,
                    replication_seed=743646743,
                    evaluation_grid=(0.1, 0.3, 0.7),
                    pointwise_coverage=(True, True, True),
                    mean_absolute_error=5.714,
                    mean_sigma_z_hat=5.245,
                    mean_pointwise_interval_length=17.256,
                    uniform_critical_value=1.791,
                ),
                overshoot_companion_slice=_make_slice(
                    random_state=505,
                    replication_seed=1826269414,
                    evaluation_grid=(0.1, 0.3, 0.7),
                    pointwise_coverage=(True, True, True),
                    mean_absolute_error=3.297,
                    mean_sigma_z_hat=5.506,
                    mean_pointwise_interval_length=18.114,
                    uniform_critical_value=2.285,
                ),
            ),
            Phase7MonteCarloWideningPolicyCoverageAnchorGridReplayInput(
                grid_label="near_zero_grid",
                coverage_anchor_slice=_make_slice(
                    random_state=202,
                    replication_seed=743646743,
                    evaluation_grid=(0.05, 0.15, 0.25),
                    pointwise_coverage=(True, True, False),
                    mean_absolute_error=2.981,
                    mean_sigma_z_hat=3.570,
                    mean_pointwise_interval_length=11.746,
                    uniform_critical_value=1.849,
                ),
                overshoot_companion_slice=_make_slice(
                    random_state=505,
                    replication_seed=1826269414,
                    evaluation_grid=(0.05, 0.15, 0.25),
                    pointwise_coverage=(True, True, True),
                    mean_absolute_error=5.855,
                    mean_sigma_z_hat=8.028,
                    mean_pointwise_interval_length=26.409,
                    uniform_critical_value=2.264,
                ),
            ),
            Phase7MonteCarloWideningPolicyCoverageAnchorGridReplayInput(
                grid_label="tight_center_grid",
                coverage_anchor_slice=_make_slice(
                    random_state=202,
                    replication_seed=743646743,
                    evaluation_grid=(0.1, 0.15, 0.2),
                    pointwise_coverage=(True, True, True),
                    mean_absolute_error=5.621,
                    mean_sigma_z_hat=5.336,
                    mean_pointwise_interval_length=17.553,
                    uniform_critical_value=1.871,
                ),
                overshoot_companion_slice=_make_slice(
                    random_state=505,
                    replication_seed=1826269414,
                    evaluation_grid=(0.1, 0.15, 0.2),
                    pointwise_coverage=(True, True, True),
                    mean_absolute_error=13.046,
                    mean_sigma_z_hat=15.418,
                    mean_pointwise_interval_length=50.719,
                    uniform_critical_value=2.286,
                ),
            ),
            Phase7MonteCarloWideningPolicyCoverageAnchorGridReplayInput(
                grid_label="micro_center_grid",
                coverage_anchor_slice=_make_slice(
                    random_state=202,
                    replication_seed=743646743,
                    evaluation_grid=(0.14, 0.15, 0.16),
                    pointwise_coverage=(True, True, True),
                    mean_absolute_error=2.493,
                    mean_sigma_z_hat=3.648,
                    mean_pointwise_interval_length=11.999,
                    uniform_critical_value=1.727,
                ),
                overshoot_companion_slice=_make_slice(
                    random_state=505,
                    replication_seed=1826269414,
                    evaluation_grid=(0.14, 0.15, 0.16),
                    pointwise_coverage=(True, True, True),
                    mean_absolute_error=4.581,
                    mean_sigma_z_hat=7.597,
                    mean_pointwise_interval_length=24.991,
                    uniform_critical_value=2.305,
                ),
            ),
        ),
        policy=policy,
    )


@lru_cache(maxsize=1)
def _run_canonical_report() -> Phase7MonteCarloWideningPolicyCoverageAnchorGridReport:
    return _build_canonical_report(build_phase7_canonical_monte_carlo_widening_policy())


def run_phase7_monte_carlo_widening_policy_coverage_anchor_grid_probe(
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorGridReport:
    if policy is None:
        return _run_canonical_report()
    return _build_canonical_report(policy)
