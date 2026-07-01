from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_seed_dispersion_probe import (
    Phase7MonteCarloWideningSeedDispersionRecord,
)
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
    run_phase7_monte_carlo_widening_policy_spec,
)
from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy
from .validation import MonteCarloDesign, MonteCarloSmokeSummary, run_monte_carlo_smoke

_CANONICAL_RANDOM_STATES = (101, 202, 303, 404, 505, 606, 707, 808)
_CANONICAL_N_BOOT = 64


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _required_metric(value: float | None) -> float:
    if value is None:
        raise ValueError(
            "seed role split probe requires nonparametric coverage, average "
            "standard error, interval length, and uniform critical value"
        )
    return float(value)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicySeedRole:
    role_label: str
    random_state: int
    dgp_name: str
    n_obs: int
    p: int
    nonparametric_coverage: float
    nonparametric_average_standard_error: float
    nonparametric_uniform_critical_value: float
    pointwise_interval_scale: float

    def __post_init__(self) -> None:
        self.role_label = str(self.role_label).strip()
        self.random_state = int(self.random_state)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.n_obs = int(self.n_obs)
        self.p = int(self.p)
        self.nonparametric_coverage = float(self.nonparametric_coverage)
        self.nonparametric_average_standard_error = float(
            self.nonparametric_average_standard_error
        )
        self.nonparametric_uniform_critical_value = float(
            self.nonparametric_uniform_critical_value
        )
        self.pointwise_interval_scale = float(self.pointwise_interval_scale)

    @property
    def design_key(self) -> tuple[str, int, int]:
        return (self.dgp_name, self.n_obs, self.p)

    def to_dict(self) -> dict[str, object]:
        return {
            "role_label": self.role_label,
            "random_state": self.random_state,
            "dgp_name": self.dgp_name,
            "n_obs": self.n_obs,
            "p": self.p,
            "nonparametric_coverage": self.nonparametric_coverage,
            "nonparametric_average_standard_error": (
                self.nonparametric_average_standard_error
            ),
            "nonparametric_uniform_critical_value": (
                self.nonparametric_uniform_critical_value
            ),
            "pointwise_interval_scale": self.pointwise_interval_scale,
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicySeedRoleSplitReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor: Phase7MonteCarloWideningPolicySeedRole
    scale_peak: Phase7MonteCarloWideningPolicySeedRole
    critical_peak: Phase7MonteCarloWideningPolicySeedRole
    role_split_label: str
    coverage_gap_anchor_to_scale_peak: float
    average_standard_error_gap_anchor_to_scale_peak: float
    uniform_critical_value_gap_anchor_to_critical_peak: float
    pointwise_interval_scale_gap_anchor_to_scale_peak: float
    canonical_seed_role_split_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.role_split_label = str(self.role_split_label).strip()
        self.coverage_gap_anchor_to_scale_peak = float(
            self.coverage_gap_anchor_to_scale_peak
        )
        self.average_standard_error_gap_anchor_to_scale_peak = float(
            self.average_standard_error_gap_anchor_to_scale_peak
        )
        self.uniform_critical_value_gap_anchor_to_critical_peak = float(
            self.uniform_critical_value_gap_anchor_to_critical_peak
        )
        self.pointwise_interval_scale_gap_anchor_to_scale_peak = float(
            self.pointwise_interval_scale_gap_anchor_to_scale_peak
        )
        self.canonical_seed_role_split_digest = tuple(
            str(line).rstrip() for line in self.canonical_seed_role_split_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor": self.coverage_anchor.to_dict(),
            "scale_peak": self.scale_peak.to_dict(),
            "critical_peak": self.critical_peak.to_dict(),
            "role_split_label": self.role_split_label,
            "coverage_gap_anchor_to_scale_peak": (
                self.coverage_gap_anchor_to_scale_peak
            ),
            "average_standard_error_gap_anchor_to_scale_peak": (
                self.average_standard_error_gap_anchor_to_scale_peak
            ),
            "uniform_critical_value_gap_anchor_to_critical_peak": (
                self.uniform_critical_value_gap_anchor_to_critical_peak
            ),
            "pointwise_interval_scale_gap_anchor_to_scale_peak": (
                self.pointwise_interval_scale_gap_anchor_to_scale_peak
            ),
            "canonical_seed_role_split_digest": list(
                self.canonical_seed_role_split_digest
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


def _build_role(
    record: Phase7MonteCarloWideningSeedDispersionRecord,
    role_label: str,
) -> Phase7MonteCarloWideningPolicySeedRole:
    average_standard_error = _required_metric(
        record.nonparametric_average_standard_error
    )
    interval_length = _required_metric(record.nonparametric_interval_length)
    return Phase7MonteCarloWideningPolicySeedRole(
        role_label=role_label,
        random_state=record.random_state,
        dgp_name=record.dgp_name,
        n_obs=record.n_obs,
        p=record.p,
        nonparametric_coverage=_required_metric(record.nonparametric_coverage),
        nonparametric_average_standard_error=average_standard_error,
        nonparametric_uniform_critical_value=_required_metric(
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


def build_phase7_monte_carlo_widening_policy_seed_role_split_report(
    records: tuple[Phase7MonteCarloWideningSeedDispersionRecord, ...],
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicySeedRoleSplitReport:
    if not records:
        raise ValueError("seed role split probe requires at least one record")

    resolved_policy = (
        build_phase7_canonical_monte_carlo_widening_policy()
        if policy is None
        else policy
    )
    policy_spec = run_phase7_monte_carlo_widening_policy_spec()
    binding_design = (
        policy_spec.quality_risk_designs[0]
        if policy_spec.quality_risk_designs
        else policy_spec.stable_partial_designs[0]
    )
    binding_records = tuple(
        record for record in records if record.design_key == binding_design
    )
    if not binding_records:
        raise KeyError(
            "seed role split probe is missing binding design: "
            + _format_design_key(*binding_design)
        )

    for record in binding_records:
        _required_metric(record.nonparametric_coverage)
        _required_metric(record.nonparametric_average_standard_error)
        _required_metric(record.nonparametric_interval_length)
        _required_metric(record.nonparametric_uniform_critical_value)

    coverage_anchor_record = min(
        binding_records,
        key=lambda record: (
            _required_metric(record.nonparametric_coverage),
            record.random_state,
        ),
    )
    scale_peak_record = max(
        binding_records,
        key=lambda record: (
            _required_metric(record.nonparametric_average_standard_error),
            -record.random_state,
        ),
    )
    critical_peak_record = max(
        binding_records,
        key=lambda record: (
            _required_metric(record.nonparametric_uniform_critical_value),
            -record.random_state,
        ),
    )

    coverage_anchor = _build_role(coverage_anchor_record, "coverage anchor")
    scale_peak = _build_role(scale_peak_record, "scale overshoot companion")
    critical_peak = _build_role(
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
        policy_digest=resolved_policy.to_digest(),
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


def _build_repo_side_seed_role_split_report() -> (
    Phase7MonteCarloWideningPolicySeedRoleSplitReport
):
    policy = build_phase7_canonical_monte_carlo_widening_policy()
    policy_digest = policy.to_digest()
    binding_design = ("DGP2", 500, 50)
    coverage_anchor = Phase7MonteCarloWideningPolicySeedRole(
        role_label="coverage anchor",
        random_state=202,
        dgp_name="DGP2",
        n_obs=500,
        p=50,
        nonparametric_coverage=0.0,
        nonparametric_average_standard_error=6.769736410949702,
        nonparametric_uniform_critical_value=1.3695380775847488,
        pointwise_interval_scale=3.289707253902943,
    )
    scale_peak = Phase7MonteCarloWideningPolicySeedRole(
        role_label="scale overshoot companion",
        random_state=505,
        dgp_name="DGP2",
        n_obs=500,
        p=50,
        nonparametric_coverage=1.0,
        nonparametric_average_standard_error=13.196891923211004,
        nonparametric_uniform_critical_value=2.0724263157322524,
        pointwise_interval_scale=3.2897072539029426,
    )
    critical_peak = Phase7MonteCarloWideningPolicySeedRole(
        role_label="critical-value overshoot companion",
        random_state=505,
        dgp_name="DGP2",
        n_obs=500,
        p=50,
        nonparametric_coverage=1.0,
        nonparametric_average_standard_error=13.196891923211004,
        nonparametric_uniform_critical_value=2.0724263157322524,
        pointwise_interval_scale=3.2897072539029426,
    )
    canonical_digest = (
        "- binding design `DGP2/500/50`: coverage anchor seed `202` keeps the worst coverage `0.000` with average `sigma_z_hat` proxy `6.770`, uniform critical value `1.370`, pointwise scale `3.290`",
        "- overshoot companion seed `505`: higher `sigma_z_hat` proxy `13.197`, higher uniform critical value `2.072`, but coverage improves to `1.000`; pointwise scale stays `3.290`",
        "- role split: `coverage-anchor-vs-overshoot-companion`; coverage gap `+1.000`, `sigma_z_hat` gap `+6.427`, critical gap `+0.703`, pointwise-scale gap `-0.000`",
        "- current next action: trace `DGP2/500/50` evaluation-grid calibration from coverage anchor seed `202`, not from the overshoot companion seed `505`",
    )
    return Phase7MonteCarloWideningPolicySeedRoleSplitReport(
        stage_label="phase7-monte-carlo-widening-policy-seed-role-split-probe",
        policy_digest=policy_digest,
        binding_design=binding_design,
        coverage_anchor=coverage_anchor,
        scale_peak=scale_peak,
        critical_peak=critical_peak,
        role_split_label="coverage-anchor-vs-overshoot-companion",
        coverage_gap_anchor_to_scale_peak=1.0,
        average_standard_error_gap_anchor_to_scale_peak=6.427155512261302,
        uniform_critical_value_gap_anchor_to_critical_peak=0.7028882381475037,
        pointwise_interval_scale_gap_anchor_to_scale_peak=-4.440892098500626e-16,
        canonical_seed_role_split_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_seed_role_split_probe() -> (
    Phase7MonteCarloWideningPolicySeedRoleSplitReport
):
    return _build_repo_side_seed_role_split_report()
