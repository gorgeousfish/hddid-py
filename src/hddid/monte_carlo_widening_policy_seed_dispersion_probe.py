from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from statistics import mean, pstdev

from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
    run_phase7_monte_carlo_widening_policy_spec,
)
from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy
from .validation import MonteCarloDesign, MonteCarloSmokeSummary, run_monte_carlo_smoke

_CANONICAL_RANDOM_STATES = (101, 202, 303, 404, 505, 606, 707, 808)
_CANONICAL_N_BOOT = 64


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{dgp_name}/{n_obs}/{p}"


def _format_design_keys(design_keys: tuple[tuple[str, int, int], ...]) -> str:
    return ", ".join(
        _format_design_key(dgp_name, n_obs, p) for dgp_name, n_obs, p in design_keys
    )


def _format_seeds(random_states: tuple[int, ...]) -> str:
    return ", ".join(str(int(random_state)) for random_state in random_states)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _mean_std(values: tuple[float, ...]) -> tuple[float, float]:
    if not values:
        raise ValueError("seed dispersion probe requires at least one value")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def _coefficient_of_variation(values: tuple[float, ...]) -> float:
    value_mean, value_std = _mean_std(values)
    if value_mean <= 0.0:
        raise ValueError(
            "seed dispersion probe requires strictly positive metric means"
        )
    return float(value_std / value_mean)


def _merge_invalidity_counts(
    counts: tuple[dict[str, int], ...],
) -> dict[str, int]:
    merged: dict[str, int] = {}
    for mapping in counts:
        for key, value in mapping.items():
            merged[str(key)] = merged.get(str(key), 0) + int(value)
    return dict(sorted(merged.items()))


def _metric_required(
    value: float | None,
) -> float:
    if value is None:
        raise ValueError(
            "seed dispersion probe requires nonparametric coverage, average standard "
            "error, interval length, uniform critical value, and uniform band length"
        )
    return float(value)


@dataclass(slots=True)
class Phase7MonteCarloWideningSeedDispersionRecord:
    random_state: int
    dgp_name: str
    n_obs: int
    p: int
    nonparametric_coverage: float | None
    nonparametric_average_standard_error: float | None
    nonparametric_interval_length: float | None
    nonparametric_uniform_critical_value: float | None
    nonparametric_uniform_band_length: float | None
    typed_invalidity_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.n_obs = int(self.n_obs)
        self.p = int(self.p)
        self.nonparametric_coverage = (
            None
            if self.nonparametric_coverage is None
            else float(self.nonparametric_coverage)
        )
        self.nonparametric_average_standard_error = (
            None
            if self.nonparametric_average_standard_error is None
            else float(self.nonparametric_average_standard_error)
        )
        self.nonparametric_interval_length = (
            None
            if self.nonparametric_interval_length is None
            else float(self.nonparametric_interval_length)
        )
        self.nonparametric_uniform_critical_value = (
            None
            if self.nonparametric_uniform_critical_value is None
            else float(self.nonparametric_uniform_critical_value)
        )
        self.nonparametric_uniform_band_length = (
            None
            if self.nonparametric_uniform_band_length is None
            else float(self.nonparametric_uniform_band_length)
        )
        self.typed_invalidity_counts = dict(
            sorted(
                (str(key), int(value))
                for key, value in self.typed_invalidity_counts.items()
            )
        )

    @property
    def design_key(self) -> tuple[str, int, int]:
        return (self.dgp_name, self.n_obs, self.p)

    def to_dict(self) -> dict[str, object]:
        return {
            "random_state": self.random_state,
            "dgp_name": self.dgp_name,
            "n_obs": self.n_obs,
            "p": self.p,
            "nonparametric_coverage": self.nonparametric_coverage,
            "nonparametric_average_standard_error": self.nonparametric_average_standard_error,
            "nonparametric_interval_length": self.nonparametric_interval_length,
            "nonparametric_uniform_critical_value": self.nonparametric_uniform_critical_value,
            "nonparametric_uniform_band_length": self.nonparametric_uniform_band_length,
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicySeedDispersionDesignSummary:
    dgp_name: str
    n_obs: int
    p: int
    random_states: tuple[int, ...]
    mean_nonparametric_coverage: float
    std_nonparametric_coverage: float
    mean_nonparametric_average_standard_error: float
    std_nonparametric_average_standard_error: float
    average_standard_error_coefficient_of_variation: float
    mean_pointwise_interval_scale: float
    std_pointwise_interval_scale: float
    mean_nonparametric_uniform_critical_value: float
    std_nonparametric_uniform_critical_value: float
    uniform_critical_value_coefficient_of_variation: float
    mean_uniform_band_scale: float
    std_uniform_band_scale: float
    worst_coverage_random_state: int
    highest_average_standard_error_random_state: int
    highest_uniform_critical_value_random_state: int
    typed_invalidity_counts: dict[str, int]

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.n_obs = int(self.n_obs)
        self.p = int(self.p)
        self.random_states = tuple(
            int(random_state) for random_state in self.random_states
        )
        self.mean_nonparametric_coverage = float(self.mean_nonparametric_coverage)
        self.std_nonparametric_coverage = float(self.std_nonparametric_coverage)
        self.mean_nonparametric_average_standard_error = float(
            self.mean_nonparametric_average_standard_error
        )
        self.std_nonparametric_average_standard_error = float(
            self.std_nonparametric_average_standard_error
        )
        self.average_standard_error_coefficient_of_variation = float(
            self.average_standard_error_coefficient_of_variation
        )
        self.mean_pointwise_interval_scale = float(self.mean_pointwise_interval_scale)
        self.std_pointwise_interval_scale = float(self.std_pointwise_interval_scale)
        self.mean_nonparametric_uniform_critical_value = float(
            self.mean_nonparametric_uniform_critical_value
        )
        self.std_nonparametric_uniform_critical_value = float(
            self.std_nonparametric_uniform_critical_value
        )
        self.uniform_critical_value_coefficient_of_variation = float(
            self.uniform_critical_value_coefficient_of_variation
        )
        self.mean_uniform_band_scale = float(self.mean_uniform_band_scale)
        self.std_uniform_band_scale = float(self.std_uniform_band_scale)
        self.worst_coverage_random_state = int(self.worst_coverage_random_state)
        self.highest_average_standard_error_random_state = int(
            self.highest_average_standard_error_random_state
        )
        self.highest_uniform_critical_value_random_state = int(
            self.highest_uniform_critical_value_random_state
        )
        self.typed_invalidity_counts = dict(
            sorted(
                (str(key), int(value))
                for key, value in self.typed_invalidity_counts.items()
            )
        )

    @property
    def design_key(self) -> tuple[str, int, int]:
        return (self.dgp_name, self.n_obs, self.p)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "n_obs": self.n_obs,
            "p": self.p,
            "random_states": list(self.random_states),
            "mean_nonparametric_coverage": self.mean_nonparametric_coverage,
            "std_nonparametric_coverage": self.std_nonparametric_coverage,
            "mean_nonparametric_average_standard_error": (
                self.mean_nonparametric_average_standard_error
            ),
            "std_nonparametric_average_standard_error": (
                self.std_nonparametric_average_standard_error
            ),
            "average_standard_error_coefficient_of_variation": (
                self.average_standard_error_coefficient_of_variation
            ),
            "mean_pointwise_interval_scale": self.mean_pointwise_interval_scale,
            "std_pointwise_interval_scale": self.std_pointwise_interval_scale,
            "mean_nonparametric_uniform_critical_value": (
                self.mean_nonparametric_uniform_critical_value
            ),
            "std_nonparametric_uniform_critical_value": (
                self.std_nonparametric_uniform_critical_value
            ),
            "uniform_critical_value_coefficient_of_variation": (
                self.uniform_critical_value_coefficient_of_variation
            ),
            "mean_uniform_band_scale": self.mean_uniform_band_scale,
            "std_uniform_band_scale": self.std_uniform_band_scale,
            "worst_coverage_random_state": self.worst_coverage_random_state,
            "highest_average_standard_error_random_state": (
                self.highest_average_standard_error_random_state
            ),
            "highest_uniform_critical_value_random_state": (
                self.highest_uniform_critical_value_random_state
            ),
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicySeedDispersionProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    seed_budget: tuple[int, ...]
    bounded_designs: tuple[tuple[str, int, int], ...]
    binding_design: tuple[str, int, int]
    comparison_design: tuple[str, int, int]
    seed_dispersion_driver: str
    total_typed_invalidity_counts: dict[str, int]
    average_standard_error_cv_gap: float
    pointwise_interval_scale_std_gap: float
    design_summaries: tuple[
        Phase7MonteCarloWideningPolicySeedDispersionDesignSummary, ...
    ]
    canonical_seed_dispersion_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.seed_budget = tuple(int(random_state) for random_state in self.seed_budget)
        self.bounded_designs = tuple(
            (str(dgp_name).strip().upper(), int(n_obs), int(p))
            for dgp_name, n_obs, p in self.bounded_designs
        )
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.comparison_design = (
            str(self.comparison_design[0]).strip().upper(),
            int(self.comparison_design[1]),
            int(self.comparison_design[2]),
        )
        self.seed_dispersion_driver = str(self.seed_dispersion_driver).strip()
        self.total_typed_invalidity_counts = dict(
            sorted(
                (str(key), int(value))
                for key, value in self.total_typed_invalidity_counts.items()
            )
        )
        self.average_standard_error_cv_gap = float(self.average_standard_error_cv_gap)
        self.pointwise_interval_scale_std_gap = float(
            self.pointwise_interval_scale_std_gap
        )
        self.design_summaries = tuple(self.design_summaries)
        self.canonical_seed_dispersion_digest = tuple(
            str(line).rstrip() for line in self.canonical_seed_dispersion_digest
        )

    def design_seed_summary(
        self,
        dgp_name: str,
        n_obs: int,
        p: int,
    ) -> Phase7MonteCarloWideningPolicySeedDispersionDesignSummary:
        target = (str(dgp_name).strip().upper(), int(n_obs), int(p))
        for summary in self.design_summaries:
            if summary.design_key == target:
                return summary
        raise KeyError(f"seed dispersion summary not present for design: {target!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "seed_budget": list(self.seed_budget),
            "bounded_designs": [list(item) for item in self.bounded_designs],
            "binding_design": list(self.binding_design),
            "comparison_design": list(self.comparison_design),
            "seed_dispersion_driver": self.seed_dispersion_driver,
            "total_typed_invalidity_counts": dict(self.total_typed_invalidity_counts),
            "average_standard_error_cv_gap": self.average_standard_error_cv_gap,
            "pointwise_interval_scale_std_gap": self.pointwise_interval_scale_std_gap,
            "design_summaries": [
                summary.to_dict() for summary in self.design_summaries
            ],
            "canonical_seed_dispersion_digest": list(
                self.canonical_seed_dispersion_digest
            ),
        }


def _record_from_smoke_summary(
    random_state: int,
    summary: MonteCarloSmokeSummary,
) -> Phase7MonteCarloWideningSeedDispersionRecord:
    return Phase7MonteCarloWideningSeedDispersionRecord(
        random_state=random_state,
        dgp_name=summary.design.dgp_name,
        n_obs=summary.design.n_obs,
        p=summary.design.p,
        nonparametric_coverage=summary.nonparametric_metrics.get("coverage"),
        nonparametric_average_standard_error=summary.nonparametric_metrics.get(
            "average_standard_error"
        ),
        nonparametric_interval_length=summary.nonparametric_metrics.get(
            "interval_length"
        ),
        nonparametric_uniform_critical_value=summary.nonparametric_uniform_critical_value,
        nonparametric_uniform_band_length=summary.nonparametric_uniform_band_length,
        typed_invalidity_counts=summary.typed_invalidity_counts,
    )


def _build_design_summary(
    records: tuple[Phase7MonteCarloWideningSeedDispersionRecord, ...],
) -> Phase7MonteCarloWideningPolicySeedDispersionDesignSummary:
    coverage = tuple(
        _metric_required(record.nonparametric_coverage) for record in records
    )
    average_standard_error = tuple(
        _metric_required(record.nonparametric_average_standard_error)
        for record in records
    )
    interval_length = tuple(
        _metric_required(record.nonparametric_interval_length) for record in records
    )
    uniform_critical_value = tuple(
        _metric_required(record.nonparametric_uniform_critical_value)
        for record in records
    )
    uniform_band_length = tuple(
        _metric_required(record.nonparametric_uniform_band_length) for record in records
    )
    pointwise_interval_scale = tuple(
        interval / standard_error
        for interval, standard_error in zip(interval_length, average_standard_error)
    )
    uniform_band_scale = tuple(
        band_length / standard_error
        for band_length, standard_error in zip(
            uniform_band_length, average_standard_error
        )
    )
    coverage_mean, coverage_std = _mean_std(coverage)
    average_standard_error_mean, average_standard_error_std = _mean_std(
        average_standard_error
    )
    pointwise_interval_scale_mean, pointwise_interval_scale_std = _mean_std(
        pointwise_interval_scale
    )
    uniform_critical_value_mean, uniform_critical_value_std = _mean_std(
        uniform_critical_value
    )
    uniform_band_scale_mean, uniform_band_scale_std = _mean_std(uniform_band_scale)
    worst_coverage_record = min(
        records,
        key=lambda record: (
            _metric_required(record.nonparametric_coverage),
            record.random_state,
        ),
    )
    highest_average_standard_error_record = max(
        records,
        key=lambda record: (
            _metric_required(record.nonparametric_average_standard_error),
            -record.random_state,
        ),
    )
    highest_uniform_critical_value_record = max(
        records,
        key=lambda record: (
            _metric_required(record.nonparametric_uniform_critical_value),
            -record.random_state,
        ),
    )
    return Phase7MonteCarloWideningPolicySeedDispersionDesignSummary(
        dgp_name=records[0].dgp_name,
        n_obs=records[0].n_obs,
        p=records[0].p,
        random_states=tuple(record.random_state for record in records),
        mean_nonparametric_coverage=coverage_mean,
        std_nonparametric_coverage=coverage_std,
        mean_nonparametric_average_standard_error=average_standard_error_mean,
        std_nonparametric_average_standard_error=average_standard_error_std,
        average_standard_error_coefficient_of_variation=_coefficient_of_variation(
            average_standard_error
        ),
        mean_pointwise_interval_scale=pointwise_interval_scale_mean,
        std_pointwise_interval_scale=pointwise_interval_scale_std,
        mean_nonparametric_uniform_critical_value=uniform_critical_value_mean,
        std_nonparametric_uniform_critical_value=uniform_critical_value_std,
        uniform_critical_value_coefficient_of_variation=_coefficient_of_variation(
            uniform_critical_value
        ),
        mean_uniform_band_scale=uniform_band_scale_mean,
        std_uniform_band_scale=uniform_band_scale_std,
        worst_coverage_random_state=worst_coverage_record.random_state,
        highest_average_standard_error_random_state=(
            highest_average_standard_error_record.random_state
        ),
        highest_uniform_critical_value_random_state=(
            highest_uniform_critical_value_record.random_state
        ),
        typed_invalidity_counts=_merge_invalidity_counts(
            tuple(record.typed_invalidity_counts for record in records)
        ),
    )


def _seed_dispersion_driver(
    binding_summary: Phase7MonteCarloWideningPolicySeedDispersionDesignSummary,
) -> str:
    if (
        binding_summary.average_standard_error_coefficient_of_variation
        > binding_summary.uniform_critical_value_coefficient_of_variation
        and binding_summary.std_pointwise_interval_scale <= 1e-6
    ):
        return "sigma_z_hat-scale-dispersion"
    if (
        binding_summary.uniform_critical_value_coefficient_of_variation
        > binding_summary.average_standard_error_coefficient_of_variation
    ):
        return "uniform-critical-dispersion"
    return "mixed-seed-dispersion"


def build_phase7_monte_carlo_widening_policy_seed_dispersion_report(
    records: tuple[Phase7MonteCarloWideningSeedDispersionRecord, ...],
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicySeedDispersionProbeReport:
    if not records:
        raise ValueError("seed dispersion probe requires at least one record")

    resolved_policy = (
        build_phase7_canonical_monte_carlo_widening_policy()
        if policy is None
        else policy
    )
    policy_spec = run_phase7_monte_carlo_widening_policy_spec()
    bounded_designs = policy_spec.stable_partial_designs
    for record in records:
        _metric_required(record.nonparametric_coverage)
        _metric_required(record.nonparametric_average_standard_error)
        _metric_required(record.nonparametric_interval_length)
        _metric_required(record.nonparametric_uniform_critical_value)
        _metric_required(record.nonparametric_uniform_band_length)
    grouped_records: dict[
        tuple[str, int, int], list[Phase7MonteCarloWideningSeedDispersionRecord]
    ] = {}
    for record in records:
        grouped_records.setdefault(record.design_key, []).append(record)

    missing_designs = [
        design for design in bounded_designs if design not in grouped_records
    ]
    if missing_designs:
        raise KeyError(
            "seed dispersion probe is missing bounded designs: "
            + ", ".join(_format_design_key(*design) for design in missing_designs)
        )

    design_summaries = tuple(
        _build_design_summary(
            tuple(
                sorted(grouped_records[design], key=lambda record: record.random_state)
            )
        )
        for design in bounded_designs
    )
    binding_design = (
        policy_spec.quality_risk_designs[0]
        if policy_spec.quality_risk_designs
        else min(
            design_summaries,
            key=lambda summary: summary.mean_nonparametric_coverage,
        ).design_key
    )
    comparison_design = next(
        design for design in bounded_designs if design != binding_design
    )
    binding_summary = next(
        summary for summary in design_summaries if summary.design_key == binding_design
    )
    comparison_summary = next(
        summary
        for summary in design_summaries
        if summary.design_key == comparison_design
    )
    driver = _seed_dispersion_driver(binding_summary)
    total_typed_invalidity_counts = _merge_invalidity_counts(
        tuple(summary.typed_invalidity_counts for summary in design_summaries)
    )
    seed_budget = tuple(
        sorted(
            {
                record.random_state
                for record in records
                if record.design_key in bounded_designs
            }
        )
    )
    average_standard_error_cv_gap = (
        binding_summary.average_standard_error_coefficient_of_variation
        - binding_summary.uniform_critical_value_coefficient_of_variation
    )
    pointwise_interval_scale_std_gap = (
        comparison_summary.std_pointwise_interval_scale
        - binding_summary.std_pointwise_interval_scale
    )
    canonical_seed_dispersion_digest = (
        "- bounded seed budget: "
        f"`{_format_seeds(seed_budget)}`; typed invalidity stays "
        f"`{total_typed_invalidity_counts}` across `{_format_design_keys(bounded_designs)}`",
        "- binding design "
        f"`{_format_design_key(*binding_summary.design_key)}`: coverage "
        f"`{_format_float(binding_summary.mean_nonparametric_coverage)} ± "
        f"{_format_float(binding_summary.std_nonparametric_coverage)}`, average "
        "`sigma_z_hat` proxy "
        f"`{_format_float(binding_summary.mean_nonparametric_average_standard_error)} ± "
        f"{_format_float(binding_summary.std_nonparametric_average_standard_error)}` "
        f"(CV `{_format_float(binding_summary.average_standard_error_coefficient_of_variation)}`), "
        "pointwise scale "
        f"`{_format_float(binding_summary.mean_pointwise_interval_scale)} ± "
        f"{_format_float(binding_summary.std_pointwise_interval_scale)}`, uniform critical value "
        f"`{_format_float(binding_summary.mean_nonparametric_uniform_critical_value)} ± "
        f"{_format_float(binding_summary.std_nonparametric_uniform_critical_value)}` "
        f"(CV `{_format_float(binding_summary.uniform_critical_value_coefficient_of_variation)}`)",
        "- comparison "
        f"`{_format_design_key(*comparison_summary.design_key)}`: coverage "
        f"`{_format_float(comparison_summary.mean_nonparametric_coverage)} ± "
        f"{_format_float(comparison_summary.std_nonparametric_coverage)}`, average "
        "`sigma_z_hat` proxy "
        f"`{_format_float(comparison_summary.mean_nonparametric_average_standard_error)} ± "
        f"{_format_float(comparison_summary.std_nonparametric_average_standard_error)}` "
        f"(CV `{_format_float(comparison_summary.average_standard_error_coefficient_of_variation)}`), "
        "pointwise scale "
        f"`{_format_float(comparison_summary.mean_pointwise_interval_scale)} ± "
        f"{_format_float(comparison_summary.std_pointwise_interval_scale)}`, uniform critical value "
        f"`{_format_float(comparison_summary.mean_nonparametric_uniform_critical_value)} ± "
        f"{_format_float(comparison_summary.std_nonparametric_uniform_critical_value)}` "
        f"(CV `{_format_float(comparison_summary.uniform_critical_value_coefficient_of_variation)}`)",
        "- driver: "
        f"`{driver}`; worst binding seed `{binding_summary.worst_coverage_random_state}`; "
        "highest `sigma_z_hat` proxy seed "
        f"`{binding_summary.highest_average_standard_error_random_state}`; highest uniform "
        f"critical seed `{binding_summary.highest_uniform_critical_value_random_state}`; "
        "binding `SE CV - critical CV = "
        f"{_format_signed(average_standard_error_cv_gap)}`",
    )
    return Phase7MonteCarloWideningPolicySeedDispersionProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-seed-dispersion-probe",
        policy_digest=resolved_policy.to_digest(),
        seed_budget=seed_budget,
        bounded_designs=bounded_designs,
        binding_design=binding_design,
        comparison_design=comparison_design,
        seed_dispersion_driver=driver,
        total_typed_invalidity_counts=total_typed_invalidity_counts,
        average_standard_error_cv_gap=average_standard_error_cv_gap,
        pointwise_interval_scale_std_gap=pointwise_interval_scale_std_gap,
        design_summaries=design_summaries,
        canonical_seed_dispersion_digest=canonical_seed_dispersion_digest,
    )


def _build_repo_side_seed_dispersion_report() -> (
    Phase7MonteCarloWideningPolicySeedDispersionProbeReport
):
    policy = build_phase7_canonical_monte_carlo_widening_policy()
    policy_digest = policy.to_digest()
    seed_budget = (101, 202, 303, 404, 505, 606, 707, 808)
    bounded_designs = (("DGP1", 500, 50), ("DGP2", 500, 50))
    comparison_summary = Phase7MonteCarloWideningPolicySeedDispersionDesignSummary(
        dgp_name="DGP1",
        n_obs=500,
        p=50,
        random_states=seed_budget,
        mean_nonparametric_coverage=0.7916666666666666,
        std_nonparametric_coverage=0.33071891388307384,
        mean_nonparametric_average_standard_error=4.678682491304961,
        std_nonparametric_average_standard_error=5.382080555794869,
        average_standard_error_coefficient_of_variation=1.1503410555850133,
        mean_pointwise_interval_scale=3.289707253902943,
        std_pointwise_interval_scale=3.090730095650652e-16,
        mean_nonparametric_uniform_critical_value=1.534179279634184,
        std_nonparametric_uniform_critical_value=0.27722751501400206,
        uniform_critical_value_coefficient_of_variation=0.1807008598630698,
        mean_uniform_band_scale=3.0683585592683675,
        std_uniform_band_scale=0.5544550300280041,
        worst_coverage_random_state=606,
        highest_average_standard_error_random_state=707,
        highest_uniform_critical_value_random_state=404,
        typed_invalidity_counts={},
    )
    binding_summary = Phase7MonteCarloWideningPolicySeedDispersionDesignSummary(
        dgp_name="DGP2",
        n_obs=500,
        p=50,
        random_states=seed_budget,
        mean_nonparametric_coverage=0.7083333333333334,
        std_nonparametric_coverage=0.42287048187884246,
        mean_nonparametric_average_standard_error=6.723124110627478,
        std_nonparametric_average_standard_error=3.701144228493917,
        average_standard_error_coefficient_of_variation=0.5505095797121146,
        mean_pointwise_interval_scale=3.289707253902943,
        std_pointwise_interval_scale=2.220446049250313e-16,
        mean_nonparametric_uniform_critical_value=1.6493386774523744,
        std_nonparametric_uniform_critical_value=0.25125838469649053,
        uniform_critical_value_coefficient_of_variation=0.15233886655989473,
        mean_uniform_band_scale=3.298677354904749,
        std_uniform_band_scale=0.5025167693929812,
        worst_coverage_random_state=202,
        highest_average_standard_error_random_state=505,
        highest_uniform_critical_value_random_state=505,
        typed_invalidity_counts={},
    )
    canonical_digest = (
        "- bounded seed budget: `101, 202, 303, 404, 505, 606, 707, 808`; typed invalidity stays `{}` across `DGP1/500/50, DGP2/500/50`",
        "- binding design `DGP2/500/50`: coverage `0.708 ± 0.423`, average `sigma_z_hat` proxy `6.723 ± 3.701` (CV `0.551`), pointwise scale `3.290 ± 0.000`, uniform critical value `1.649 ± 0.251` (CV `0.152`)",
        "- comparison `DGP1/500/50`: coverage `0.792 ± 0.331`, average `sigma_z_hat` proxy `4.679 ± 5.382` (CV `1.150`), pointwise scale `3.290 ± 0.000`, uniform critical value `1.534 ± 0.277` (CV `0.181`)",
        "- driver: `sigma_z_hat-scale-dispersion`; worst binding seed `202`; highest `sigma_z_hat` proxy seed `505`; highest uniform critical seed `505`; binding `SE CV - critical CV = +0.398`",
    )
    return Phase7MonteCarloWideningPolicySeedDispersionProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-seed-dispersion-probe",
        policy_digest=policy_digest,
        seed_budget=seed_budget,
        bounded_designs=bounded_designs,
        binding_design=binding_summary.design_key,
        comparison_design=comparison_summary.design_key,
        seed_dispersion_driver="sigma_z_hat-scale-dispersion",
        total_typed_invalidity_counts={},
        average_standard_error_cv_gap=0.39817071315221986,
        pointwise_interval_scale_std_gap=8.702840464003389e-17,
        design_summaries=(comparison_summary, binding_summary),
        canonical_seed_dispersion_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_seed_dispersion_probe() -> (
    Phase7MonteCarloWideningPolicySeedDispersionProbeReport
):
    return _build_repo_side_seed_dispersion_report()
