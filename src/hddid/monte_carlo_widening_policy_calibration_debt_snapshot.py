from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe,
)
from .monte_carlo_widening_policy_quality_risk_probe import (
    Phase7MonteCarloWideningPolicyQualityRiskDesignSummary,
    Phase7MonteCarloWideningPolicyQualityRiskProbeReport,
    _build_repo_side_quality_risk_probe_report,
    _binding_driver as _quality_risk_binding_driver,
    run_phase7_monte_carlo_widening_policy_quality_risk_probe,
)
from .monte_carlo_widening_policy_seed_dispersion_probe import (
    Phase7MonteCarloWideningPolicySeedDispersionDesignSummary,
    Phase7MonteCarloWideningPolicySeedDispersionProbeReport,
    Phase7MonteCarloWideningSeedDispersionRecord,
    _build_design_summary as _build_seed_dispersion_design_summary,
    _merge_invalidity_counts as _merge_seed_invalidity_counts,
    _seed_dispersion_driver as _seed_dispersion_driver,
    run_phase7_monte_carlo_widening_policy_seed_dispersion_probe,
)
from .monte_carlo_widening_policy_seed_role_split_probe import (
    Phase7MonteCarloWideningPolicySeedRole,
    Phase7MonteCarloWideningPolicySeedRoleSplitReport,
    run_phase7_monte_carlo_widening_policy_seed_role_split_probe,
)
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
    run_phase7_monte_carlo_widening_policy_spec,
)
from .validation import MonteCarloDesign, run_monte_carlo_smoke


_CANONICAL_RANDOM_STATES = (101, 202, 303, 404, 505, 606, 707, 808)
_CANONICAL_N_BOOT = 64


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _retarget_quality_risk_report(
    report: Phase7MonteCarloWideningPolicyQualityRiskProbeReport,
    *,
    binding_design: tuple[str, int, int],
) -> Phase7MonteCarloWideningPolicyQualityRiskProbeReport:
    design_summary_map = {
        summary.design_key: summary for summary in report.design_quality_summaries
    }
    if binding_design not in design_summary_map:
        raise KeyError(
            "quality risk retarget is missing binding design: "
            + _format_design_key(*binding_design)
        )
    comparison_summary = next(
        summary
        for key, summary in design_summary_map.items()
        if key != binding_design
    )
    binding_summary = design_summary_map[binding_design]
    binding_driver = _quality_risk_binding_driver(binding_summary, comparison_summary)
    rmse_ratio_gap = (
        binding_summary.rmse_to_standard_error_ratio
        - comparison_summary.rmse_to_standard_error_ratio
    )
    standard_error_reserve_gap = (
        comparison_summary.standard_error_reserve
        - binding_summary.standard_error_reserve
    )
    interval_length_to_rmse_gap = (
        comparison_summary.interval_length_to_rmse_ratio
        - binding_summary.interval_length_to_rmse_ratio
    )
    supported_floor_ceiling = binding_summary.mean_nonparametric_coverage
    canonical_floor_shortfall = max(-binding_summary.coverage_floor_slack, 0.0)
    blocker_reason = (
        "" if binding_driver == "quality-risk-cleared"
        else "quality-risk-keeps-trigger2-bounded"
    )
    canonical_quality_risk_digest = (
        "- bounded-slice quality risk: "
        f"`{_format_design_key(*binding_summary.design_key)}` coverage "
        f"`{binding_summary.mean_nonparametric_coverage:.3f} "
        f"({_format_signed(binding_summary.coverage_floor_slack)})`, "
        f"`RMSE/SE = {binding_summary.rmse_to_standard_error_ratio:.3f}`, "
        f"`SE-RMSE = {_format_signed(binding_summary.standard_error_reserve)}`, "
        f"`interval/RMSE = {binding_summary.interval_length_to_rmse_ratio:.3f}`; "
        f"comparison `{_format_design_key(*comparison_summary.design_key)}` coverage "
        f"`{comparison_summary.mean_nonparametric_coverage:.3f} "
        f"({_format_signed(comparison_summary.coverage_floor_slack)})`, "
        f"`RMSE/SE = {comparison_summary.rmse_to_standard_error_ratio:.3f}`, "
        f"`SE-RMSE = {_format_signed(comparison_summary.standard_error_reserve)}`, "
        f"`interval/RMSE = {comparison_summary.interval_length_to_rmse_ratio:.3f}`",
        "- binding driver: "
        f"`{binding_driver}`; supported floor ceiling "
        f"`{supported_floor_ceiling:.3f}`; canonical floor "
        f"`0.850`; shortfall `{canonical_floor_shortfall:.3f}`",
        "- calibration headroom gap versus best bounded design: "
        f"`RMSE/SE gap = {_format_signed(rmse_ratio_gap)}`, "
        f"`SE-RMSE gap = {_format_signed(standard_error_reserve_gap)}`, "
        f"`interval/RMSE gap = {_format_signed(interval_length_to_rmse_gap)}`",
    )
    return Phase7MonteCarloWideningPolicyQualityRiskProbeReport(
        stage_label=report.stage_label,
        policy_digest=report.policy_digest,
        binding_design=binding_summary.design_key,
        best_design=comparison_summary.design_key,
        binding_driver=binding_driver,
        blocker_reason=blocker_reason,
        supported_floor_ceiling=supported_floor_ceiling,
        canonical_floor_shortfall=canonical_floor_shortfall,
        rmse_to_standard_error_ratio_gap=rmse_ratio_gap,
        standard_error_reserve_gap=standard_error_reserve_gap,
        interval_length_to_rmse_gap=interval_length_to_rmse_gap,
        design_quality_summaries=(binding_summary, comparison_summary),
        canonical_quality_risk_digest=canonical_quality_risk_digest,
        quality_risk_source_mode=report.quality_risk_source_mode,
        quality_risk_source_note=report.quality_risk_source_note,
    )


def _retarget_seed_dispersion_report(
    report: Phase7MonteCarloWideningPolicySeedDispersionProbeReport,
    *,
    binding_design: tuple[str, int, int],
) -> Phase7MonteCarloWideningPolicySeedDispersionProbeReport:
    design_summary_map = {
        summary.design_key: summary for summary in report.design_summaries
    }
    if binding_design not in design_summary_map:
        raise KeyError(
            "seed dispersion retarget is missing binding design: "
            + _format_design_key(*binding_design)
        )
    comparison_design = next(
        design for design in report.bounded_designs if design != binding_design
    )
    binding_summary = design_summary_map[binding_design]
    comparison_summary = design_summary_map[comparison_design]
    driver = _seed_dispersion_driver(binding_summary)
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
        f"`{', '.join(str(seed) for seed in report.seed_budget)}`; typed invalidity stays "
        f"`{report.total_typed_invalidity_counts}` across "
        f"`{', '.join(_format_design_key(*design) for design in report.bounded_designs)}`",
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
        stage_label=report.stage_label,
        policy_digest=report.policy_digest,
        seed_budget=report.seed_budget,
        bounded_designs=report.bounded_designs,
        binding_design=binding_design,
        comparison_design=comparison_design,
        seed_dispersion_driver=driver,
        total_typed_invalidity_counts=report.total_typed_invalidity_counts,
        average_standard_error_cv_gap=average_standard_error_cv_gap,
        pointwise_interval_scale_std_gap=pointwise_interval_scale_std_gap,
        design_summaries=(binding_summary, comparison_summary),
        canonical_seed_dispersion_digest=canonical_seed_dispersion_digest,
    )


def _collect_bounded_seed_records(
    bounded_designs: tuple[tuple[str, int, int], ...],
) -> tuple[Phase7MonteCarloWideningSeedDispersionRecord, ...]:
    designs = tuple(
        MonteCarloDesign(dgp_name=dgp_name, n_obs=n_obs, p=p)
        for dgp_name, n_obs, p in bounded_designs
    )
    records: list[Phase7MonteCarloWideningSeedDispersionRecord] = []
    for random_state in _CANONICAL_RANDOM_STATES:
        smoke_report = run_monte_carlo_smoke(
            designs=designs,
            n_replications=1,
            random_state=random_state,
            n_boot=_CANONICAL_N_BOOT,
        )
        records.extend(
            Phase7MonteCarloWideningSeedDispersionRecord(
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
            for summary in smoke_report.summaries
        )
    return tuple(records)


def _build_seed_role(
    record: Phase7MonteCarloWideningSeedDispersionRecord,
    role_label: str,
) -> Phase7MonteCarloWideningPolicySeedRole:
    average_standard_error = float(record.nonparametric_average_standard_error)
    interval_length = float(record.nonparametric_interval_length)
    return Phase7MonteCarloWideningPolicySeedRole(
        role_label=role_label,
        random_state=record.random_state,
        dgp_name=record.dgp_name,
        n_obs=record.n_obs,
        p=record.p,
        nonparametric_coverage=float(record.nonparametric_coverage),
        nonparametric_average_standard_error=average_standard_error,
        nonparametric_uniform_critical_value=float(
            record.nonparametric_uniform_critical_value
        ),
        pointwise_interval_scale=interval_length / average_standard_error,
    )


def _role_split_label(
    coverage_anchor: Phase7MonteCarloWideningPolicySeedRole,
    scale_peak: Phase7MonteCarloWideningPolicySeedRole,
    critical_peak: Phase7MonteCarloWideningPolicySeedRole,
) -> str:
    if (
        coverage_anchor.random_state
        == scale_peak.random_state
        == critical_peak.random_state
    ):
        return "single-seed-anchor"
    if (
        scale_peak.random_state == critical_peak.random_state
        and coverage_anchor.random_state != scale_peak.random_state
    ):
        return "coverage-anchor-vs-overshoot-companion"
    if coverage_anchor.random_state == scale_peak.random_state:
        return "coverage-scale-anchor-vs-critical-companion"
    return "coverage-scale-critical-split"


def _build_seed_role_split_report_for_binding_design(
    records: tuple[Phase7MonteCarloWideningSeedDispersionRecord, ...],
    *,
    binding_design: tuple[str, int, int],
    policy_digest: tuple[str, ...],
) -> Phase7MonteCarloWideningPolicySeedRoleSplitReport:
    binding_records = tuple(
        record for record in records if record.design_key == binding_design
    )
    if not binding_records:
        raise KeyError(
            "seed role split probe is missing binding design: "
            + _format_design_key(*binding_design)
        )
    coverage_anchor_record = min(
        binding_records,
        key=lambda record: (
            float(record.nonparametric_coverage),
            record.random_state,
        ),
    )
    scale_peak_record = max(
        binding_records,
        key=lambda record: (
            float(record.nonparametric_average_standard_error),
            -record.random_state,
        ),
    )
    critical_peak_record = max(
        binding_records,
        key=lambda record: (
            float(record.nonparametric_uniform_critical_value),
            -record.random_state,
        ),
    )
    coverage_anchor = _build_seed_role(coverage_anchor_record, "coverage anchor")
    scale_peak = _build_seed_role(scale_peak_record, "scale overshoot companion")
    critical_peak = _build_seed_role(
        critical_peak_record, "critical-value overshoot companion"
    )
    role_split_label = _role_split_label(coverage_anchor, scale_peak, critical_peak)
    coverage_gap = (
        scale_peak.nonparametric_coverage - coverage_anchor.nonparametric_coverage
    )
    average_standard_error_gap = (
        scale_peak.nonparametric_average_standard_error
        - coverage_anchor.nonparametric_average_standard_error
    )
    uniform_critical_value_gap = (
        critical_peak.nonparametric_uniform_critical_value
        - coverage_anchor.nonparametric_uniform_critical_value
    )
    pointwise_interval_scale_gap = (
        scale_peak.pointwise_interval_scale - coverage_anchor.pointwise_interval_scale
    )
    if role_split_label == "coverage-anchor-vs-overshoot-companion":
        companion_line = (
            "- overshoot companion seed "
            f"`{scale_peak.random_state}`: higher `sigma_z_hat` proxy "
            f"`{_format_float(scale_peak.nonparametric_average_standard_error)}`, "
            "higher uniform critical value "
            f"`{_format_float(scale_peak.nonparametric_uniform_critical_value)}`, "
            "but coverage improves to "
            f"`{_format_float(scale_peak.nonparametric_coverage)}`; pointwise scale "
            f"stays `{_format_float(scale_peak.pointwise_interval_scale)}`"
        )
        implication_line = (
            "- current next action: trace "
            f"`{_format_design_key(*binding_design)}` evaluation-grid calibration from "
            f"coverage anchor seed `{coverage_anchor.random_state}`, not from the "
            f"overshoot companion seed `{scale_peak.random_state}`"
        )
    else:
        companion_line = (
            "- peak seeds: scale peak "
            f"`{scale_peak.random_state}` / critical peak `{critical_peak.random_state}`; "
            "follow-up should preserve the same role ordering before changing policy "
            "tokens"
        )
        implication_line = (
            "- current next action: keep the binding design on the same exact-seed "
            "coverage anchor before widening the policy surface"
        )
    canonical_seed_role_split_digest = (
        "- binding design "
        f"`{_format_design_key(*binding_design)}`: coverage anchor seed "
        f"`{coverage_anchor.random_state}` keeps the worst coverage "
        f"`{_format_float(coverage_anchor.nonparametric_coverage)}` with average "
        "`sigma_z_hat` proxy "
        f"`{_format_float(coverage_anchor.nonparametric_average_standard_error)}`, "
        "uniform critical value "
        f"`{_format_float(coverage_anchor.nonparametric_uniform_critical_value)}`, "
        "pointwise scale "
        f"`{_format_float(coverage_anchor.pointwise_interval_scale)}`",
        companion_line,
        "- role split: "
        f"`{role_split_label}`; coverage gap `{_format_signed(coverage_gap)}`, "
        "`sigma_z_hat` gap "
        f"`{_format_signed(average_standard_error_gap)}`, critical gap "
        f"`{_format_signed(uniform_critical_value_gap)}`, pointwise-scale gap "
        f"`{_format_signed(pointwise_interval_scale_gap)}`",
        implication_line,
    )
    return Phase7MonteCarloWideningPolicySeedRoleSplitReport(
        stage_label="phase7-monte-carlo-widening-policy-seed-role-split-probe",
        policy_digest=policy_digest,
        binding_design=binding_design,
        coverage_anchor=coverage_anchor,
        scale_peak=scale_peak,
        critical_peak=critical_peak,
        role_split_label=role_split_label,
        coverage_gap_anchor_to_scale_peak=coverage_gap,
        average_standard_error_gap_anchor_to_scale_peak=average_standard_error_gap,
        uniform_critical_value_gap_anchor_to_critical_peak=uniform_critical_value_gap,
        pointwise_interval_scale_gap_anchor_to_scale_peak=pointwise_interval_scale_gap,
        canonical_seed_role_split_digest=canonical_seed_role_split_digest,
    )


def _build_seed_dispersion_report_for_binding_design(
    records: tuple[Phase7MonteCarloWideningSeedDispersionRecord, ...],
    *,
    bounded_designs: tuple[tuple[str, int, int], ...],
    binding_design: tuple[str, int, int],
    policy_digest: tuple[str, ...],
) -> Phase7MonteCarloWideningPolicySeedDispersionProbeReport:
    grouped_records: dict[
        tuple[str, int, int], list[Phase7MonteCarloWideningSeedDispersionRecord]
    ] = {}
    for record in records:
        grouped_records.setdefault(record.design_key, []).append(record)
    design_summaries = tuple(
        _build_seed_dispersion_design_summary(
            tuple(sorted(grouped_records[design], key=lambda record: record.random_state))
        )
        for design in bounded_designs
    )
    design_summary_map = {
        summary.design_key: summary for summary in design_summaries
    }
    comparison_design = next(
        design for design in bounded_designs if design != binding_design
    )
    binding_summary = design_summary_map[binding_design]
    comparison_summary = design_summary_map[comparison_design]
    driver = _seed_dispersion_driver(binding_summary)
    total_typed_invalidity_counts = _merge_seed_invalidity_counts(
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
        f"`{', '.join(str(seed) for seed in seed_budget)}`; typed invalidity stays "
        f"`{total_typed_invalidity_counts}` across "
        f"`{', '.join(_format_design_key(*design) for design in bounded_designs)}`",
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
        policy_digest=policy_digest,
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


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCalibrationDebtSnapshotReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    binding_driver: str
    seed_dispersion_driver: str
    seed_role_split_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    supported_floor_ceiling: float
    canonical_floor_shortfall: float
    local_scale_access_shortfall_share: float
    required_coupled_sigma_access_share_of_companion: float
    fixed_sigma_correlation_share_of_companion: float
    fixed_correlation_sigma_multiple_vs_companion: float
    repair_target_signature: str
    quality_risk_digest: tuple[str, ...]
    quality_risk_source_mode: str
    quality_risk_source_note: str
    seed_dispersion_digest: tuple[str, ...]
    seed_role_split_digest: tuple[str, ...]
    access_share_digest: tuple[str, ...]
    center_coupling_repair_digest: tuple[str, ...]
    canonical_calibration_debt_snapshot_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.binding_driver = str(self.binding_driver).strip()
        self.seed_dispersion_driver = str(self.seed_dispersion_driver).strip()
        self.seed_role_split_label = str(self.seed_role_split_label).strip()
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.supported_floor_ceiling = float(self.supported_floor_ceiling)
        self.canonical_floor_shortfall = float(self.canonical_floor_shortfall)
        self.local_scale_access_shortfall_share = float(
            self.local_scale_access_shortfall_share
        )
        self.required_coupled_sigma_access_share_of_companion = float(
            self.required_coupled_sigma_access_share_of_companion
        )
        self.fixed_sigma_correlation_share_of_companion = float(
            self.fixed_sigma_correlation_share_of_companion
        )
        self.fixed_correlation_sigma_multiple_vs_companion = float(
            self.fixed_correlation_sigma_multiple_vs_companion
        )
        self.repair_target_signature = str(self.repair_target_signature).strip()
        self.quality_risk_digest = tuple(
            str(line).rstrip() for line in self.quality_risk_digest
        )
        self.quality_risk_source_mode = str(self.quality_risk_source_mode).strip()
        self.quality_risk_source_note = str(self.quality_risk_source_note).strip()
        self.seed_dispersion_digest = tuple(
            str(line).rstrip() for line in self.seed_dispersion_digest
        )
        self.seed_role_split_digest = tuple(
            str(line).rstrip() for line in self.seed_role_split_digest
        )
        self.access_share_digest = tuple(
            str(line).rstrip() for line in self.access_share_digest
        )
        self.center_coupling_repair_digest = tuple(
            str(line).rstrip() for line in self.center_coupling_repair_digest
        )
        self.canonical_calibration_debt_snapshot_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_calibration_debt_snapshot_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "binding_driver": self.binding_driver,
            "seed_dispersion_driver": self.seed_dispersion_driver,
            "seed_role_split_label": self.seed_role_split_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "supported_floor_ceiling": self.supported_floor_ceiling,
            "canonical_floor_shortfall": self.canonical_floor_shortfall,
            "local_scale_access_shortfall_share": (
                self.local_scale_access_shortfall_share
            ),
            "required_coupled_sigma_access_share_of_companion": (
                self.required_coupled_sigma_access_share_of_companion
            ),
            "fixed_sigma_correlation_share_of_companion": (
                self.fixed_sigma_correlation_share_of_companion
            ),
            "fixed_correlation_sigma_multiple_vs_companion": (
                self.fixed_correlation_sigma_multiple_vs_companion
            ),
            "repair_target_signature": self.repair_target_signature,
            "quality_risk_digest": list(self.quality_risk_digest),
            "quality_risk_source_mode": self.quality_risk_source_mode,
            "quality_risk_source_note": self.quality_risk_source_note,
            "seed_dispersion_digest": list(self.seed_dispersion_digest),
            "seed_role_split_digest": list(self.seed_role_split_digest),
            "access_share_digest": list(self.access_share_digest),
            "center_coupling_repair_digest": list(self.center_coupling_repair_digest),
            "canonical_calibration_debt_snapshot_digest": list(
                self.canonical_calibration_debt_snapshot_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_calibration_debt_snapshot_report(
    *,
    quality_risk_report: Phase7MonteCarloWideningPolicyQualityRiskProbeReport,
    seed_dispersion_report: Phase7MonteCarloWideningPolicySeedDispersionProbeReport,
    seed_role_split_report: Phase7MonteCarloWideningPolicySeedRoleSplitReport,
    access_share_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderAccessShareReport,
    center_coupling_repair_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingRepairReport
    ),
) -> Phase7MonteCarloWideningPolicyCalibrationDebtSnapshotReport:
    policy_digest = quality_risk_report.policy_digest
    binding_design = access_share_report.binding_design
    for report_policy_digest in (
        seed_dispersion_report.policy_digest,
        seed_role_split_report.policy_digest,
        access_share_report.policy_digest,
        center_coupling_repair_report.policy_digest,
    ):
        if report_policy_digest != policy_digest:
            raise ValueError(
                "calibration debt snapshot requires a single canonical policy digest"
            )
    for report_binding_design in (
        seed_dispersion_report.binding_design,
        seed_role_split_report.binding_design,
        access_share_report.binding_design,
        center_coupling_repair_report.binding_design,
    ):
        if report_binding_design != binding_design:
            raise ValueError(
                "calibration debt snapshot requires a single binding design"
            )

    coverage_anchor_random_state = seed_role_split_report.coverage_anchor.random_state
    overshoot_companion_random_state = seed_role_split_report.scale_peak.random_state
    if (
        seed_role_split_report.critical_peak.random_state
        != overshoot_companion_random_state
    ):
        raise ValueError(
            "calibration debt snapshot expects the scale peak and critical peak to "
            "share the same overshoot companion seed"
        )
    if (
        access_share_report.coverage_anchor_random_state != coverage_anchor_random_state
        or center_coupling_repair_report.coverage_anchor_random_state
        != coverage_anchor_random_state
    ):
        raise ValueError(
            "calibration debt snapshot requires a single coverage-anchor seed"
        )
    if (
        access_share_report.overshoot_companion_random_state
        != overshoot_companion_random_state
        or center_coupling_repair_report.overshoot_companion_random_state
        != overshoot_companion_random_state
    ):
        raise ValueError(
            "calibration debt snapshot requires a single overshoot companion seed"
        )

    canonical_digest = (
        "- Trigger 2 binding design stays "
        f"`{_format_design_key(*binding_design)}`: canonical floor `0.850` still exceeds "
        f"the supported ceiling `{_format_float(quality_risk_report.supported_floor_ceiling)}` "
        f"by `{_format_float(quality_risk_report.canonical_floor_shortfall)}`, and the live "
        f"quality-risk driver remains `{quality_risk_report.binding_driver}` rather than "
        "runtime-budget exhaustion",
        "- seed-level volatility is still "
        f"`{seed_dispersion_report.seed_dispersion_driver}`, but its binding seed split is "
        f"already narrowed to `{seed_role_split_report.role_split_label}`: coverage anchor "
        f"seed `{coverage_anchor_random_state}` versus overshoot companion seed "
        f"`{overshoot_companion_random_state}`",
        "- the residual right-shoulder miss is now a one-scalar "
        f"`local_scale_access_shortfall_share = {_format_percent(access_share_report.local_scale_access_shortfall_share)}` "
        "at `z = 0.25`, not a whole-window undercoverage or global interval-rule debt",
        "- closing that last shoulder miss only requires coupled access to reach "
        f"`{_format_percent(center_coupling_repair_report.required_coupled_sigma_access_share_of_companion)}` "
        f"of companion seed `{overshoot_companion_random_state}`, or equivalently "
        f"`{_format_percent(center_coupling_repair_report.fixed_sigma_correlation_share_of_companion)}` "
        "of companion shoulder-center correlation; freezing correlation instead would "
        "force shoulder `sigma_z_hat` up to "
        f"`{_format_ratio(center_coupling_repair_report.fixed_correlation_sigma_multiple_vs_companion)}` "
        "companion scale",
        "- predecessor calibration-stage implication stays "
        f"`{center_coupling_repair_report.driver_signature}`: trace partial "
        "shoulder-center covariance access restoration for coverage anchor seed "
        f"`{coverage_anchor_random_state}`; downstream source-level routing may "
        "tighten this object without reopening floor relaxation, extra seeds, or "
        "uniform-critical retuning",
    )

    return Phase7MonteCarloWideningPolicyCalibrationDebtSnapshotReport(
        stage_label="phase7-monte-carlo-widening-policy-calibration-debt-snapshot",
        policy_digest=policy_digest,
        binding_design=binding_design,
        binding_driver=quality_risk_report.binding_driver,
        seed_dispersion_driver=seed_dispersion_report.seed_dispersion_driver,
        seed_role_split_label=seed_role_split_report.role_split_label,
        coverage_anchor_random_state=coverage_anchor_random_state,
        overshoot_companion_random_state=overshoot_companion_random_state,
        supported_floor_ceiling=quality_risk_report.supported_floor_ceiling,
        canonical_floor_shortfall=quality_risk_report.canonical_floor_shortfall,
        local_scale_access_shortfall_share=(
            access_share_report.local_scale_access_shortfall_share
        ),
        required_coupled_sigma_access_share_of_companion=(
            center_coupling_repair_report.required_coupled_sigma_access_share_of_companion
        ),
        fixed_sigma_correlation_share_of_companion=(
            center_coupling_repair_report.fixed_sigma_correlation_share_of_companion
        ),
        fixed_correlation_sigma_multiple_vs_companion=(
            center_coupling_repair_report.fixed_correlation_sigma_multiple_vs_companion
        ),
        repair_target_signature=center_coupling_repair_report.driver_signature,
        quality_risk_digest=quality_risk_report.canonical_quality_risk_digest,
        quality_risk_source_mode=quality_risk_report.quality_risk_source_mode,
        quality_risk_source_note=quality_risk_report.quality_risk_source_note,
        seed_dispersion_digest=seed_dispersion_report.canonical_seed_dispersion_digest,
        seed_role_split_digest=seed_role_split_report.canonical_seed_role_split_digest,
        access_share_digest=(
            access_share_report.canonical_coverage_anchor_shoulder_access_share_digest
        ),
        center_coupling_repair_digest=(
            center_coupling_repair_report.canonical_coverage_anchor_shoulder_center_coupling_repair_digest
        ),
        canonical_calibration_debt_snapshot_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_calibration_debt_snapshot() -> (
    Phase7MonteCarloWideningPolicyCalibrationDebtSnapshotReport
):
    access_share_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe()
    )
    target_binding_design = access_share_report.binding_design
    quality_risk_report = run_phase7_monte_carlo_widening_policy_quality_risk_probe()
    if quality_risk_report.binding_design != target_binding_design:
        quality_risk_report = _build_repo_side_quality_risk_probe_report(
            policy=build_phase7_canonical_monte_carlo_widening_policy()
        )
        policy_spec = run_phase7_monte_carlo_widening_policy_spec()
        bounded_designs = policy_spec.stable_partial_designs
        records = _collect_bounded_seed_records(bounded_designs)
        seed_dispersion_report = _build_seed_dispersion_report_for_binding_design(
            records,
            bounded_designs=bounded_designs,
            binding_design=target_binding_design,
            policy_digest=quality_risk_report.policy_digest,
        )
        seed_role_split_report = _build_seed_role_split_report_for_binding_design(
            records,
            binding_design=target_binding_design,
            policy_digest=quality_risk_report.policy_digest,
        )
    else:
        seed_dispersion_report = (
            run_phase7_monte_carlo_widening_policy_seed_dispersion_probe()
        )
        seed_role_split_report = (
            run_phase7_monte_carlo_widening_policy_seed_role_split_probe()
        )
    return build_phase7_monte_carlo_widening_policy_calibration_debt_snapshot_report(
        quality_risk_report=quality_risk_report,
        seed_dispersion_report=seed_dispersion_report,
        seed_role_split_report=seed_role_split_report,
        access_share_report=access_share_report,
        center_coupling_repair_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe(),
    )
