from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_guard_probe import (
    build_phase7_monte_carlo_widening_policy_guard_probe_report,
    run_phase7_monte_carlo_widening_policy_guard_probe,
)
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)
from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy
from .validation import (
    MonteCarloRuntimeProbeReport,
    run_phase7_monte_carlo_runtime_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{dgp_name}/{n_obs}/{p}"


def _format_signed(value: float) -> str:
    return f"{value:+.3f}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyFloorSlackDesignSummary:
    dgp_name: str
    n_obs: int
    p: int
    mean_nonparametric_coverage: float
    coverage_floor: float
    coverage_floor_slack: float
    shortfall_to_canonical_floor: float
    floor_ceiling_headroom: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip()
        self.n_obs = int(self.n_obs)
        self.p = int(self.p)
        self.mean_nonparametric_coverage = float(self.mean_nonparametric_coverage)
        self.coverage_floor = float(self.coverage_floor)
        self.coverage_floor_slack = float(self.coverage_floor_slack)
        self.shortfall_to_canonical_floor = float(self.shortfall_to_canonical_floor)
        self.floor_ceiling_headroom = float(self.floor_ceiling_headroom)

    @property
    def design_key(self) -> tuple[str, int, int]:
        return (self.dgp_name, self.n_obs, self.p)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "n_obs": self.n_obs,
            "p": self.p,
            "mean_nonparametric_coverage": self.mean_nonparametric_coverage,
            "coverage_floor": self.coverage_floor,
            "coverage_floor_slack": self.coverage_floor_slack,
            "shortfall_to_canonical_floor": self.shortfall_to_canonical_floor,
            "floor_ceiling_headroom": self.floor_ceiling_headroom,
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyFloorSlackProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    coverage_floor: float
    supported_floor_ceiling: float
    canonical_floor_shortfall: float
    binding_design: tuple[str, int, int]
    best_design: tuple[str, int, int]
    slack_spread: float
    design_floor_summaries: tuple[
        Phase7MonteCarloWideningPolicyFloorSlackDesignSummary, ...
    ]
    canonical_floor_slack_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.coverage_floor = float(self.coverage_floor)
        self.supported_floor_ceiling = float(self.supported_floor_ceiling)
        self.canonical_floor_shortfall = float(self.canonical_floor_shortfall)
        self.binding_design = (
            str(self.binding_design[0]).strip(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.best_design = (
            str(self.best_design[0]).strip(),
            int(self.best_design[1]),
            int(self.best_design[2]),
        )
        self.slack_spread = float(self.slack_spread)
        self.design_floor_summaries = tuple(self.design_floor_summaries)
        self.canonical_floor_slack_digest = tuple(
            str(line).rstrip() for line in self.canonical_floor_slack_digest
        )

    def design_floor_summary(
        self,
        dgp_name: str,
        n_obs: int,
        p: int,
    ) -> Phase7MonteCarloWideningPolicyFloorSlackDesignSummary:
        target = (str(dgp_name).strip(), int(n_obs), int(p))
        for summary in self.design_floor_summaries:
            if summary.design_key == target:
                return summary
        raise KeyError(f"floor slack summary not present for design: {target!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "coverage_floor": self.coverage_floor,
            "supported_floor_ceiling": self.supported_floor_ceiling,
            "canonical_floor_shortfall": self.canonical_floor_shortfall,
            "binding_design": list(self.binding_design),
            "best_design": list(self.best_design),
            "slack_spread": self.slack_spread,
            "design_floor_summaries": [
                summary.to_dict() for summary in self.design_floor_summaries
            ],
            "canonical_floor_slack_digest": list(self.canonical_floor_slack_digest),
        }


def build_phase7_monte_carlo_widening_policy_floor_slack_probe_report(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicyFloorSlackProbeReport:
    resolved_policy = (
        build_phase7_canonical_monte_carlo_widening_policy()
        if policy is None
        else policy
    )
    if resolved_policy.min_nonparametric_coverage is None:
        raise ValueError("policy floor slack probe requires min_nonparametric_coverage")

    guard_probe = build_phase7_monte_carlo_widening_policy_guard_probe_report(
        runtime_probe,
        policy=resolved_policy,
    )
    coverage_floor = float(resolved_policy.min_nonparametric_coverage)
    if not guard_probe.design_guard_summaries:
        raise ValueError("policy floor slack probe requires bounded widening summaries")

    supported_floor_ceiling = min(
        summary.mean_nonparametric_coverage
        for summary in guard_probe.design_guard_summaries
    )
    ordered_guard_summaries = tuple(
        sorted(
            guard_probe.design_guard_summaries,
            key=lambda summary: (
                summary.coverage_floor_slack,
                summary.mean_nonparametric_coverage,
                summary.dgp_name,
                summary.n_obs,
                summary.p,
            ),
        )
    )
    design_floor_summaries = tuple(
        Phase7MonteCarloWideningPolicyFloorSlackDesignSummary(
            dgp_name=summary.dgp_name,
            n_obs=summary.n_obs,
            p=summary.p,
            mean_nonparametric_coverage=summary.mean_nonparametric_coverage,
            coverage_floor=coverage_floor,
            coverage_floor_slack=summary.coverage_floor_slack,
            shortfall_to_canonical_floor=max(
                coverage_floor - summary.mean_nonparametric_coverage,
                0.0,
            ),
            floor_ceiling_headroom=(
                summary.mean_nonparametric_coverage - supported_floor_ceiling
            ),
        )
        for summary in ordered_guard_summaries
    )

    binding_summary = design_floor_summaries[0]
    best_summary = max(
        design_floor_summaries,
        key=lambda summary: (
            summary.coverage_floor_slack,
            summary.mean_nonparametric_coverage,
            summary.dgp_name,
            summary.n_obs,
            summary.p,
        ),
    )
    canonical_floor_shortfall = max(coverage_floor - supported_floor_ceiling, 0.0)
    slack_spread = (
        best_summary.coverage_floor_slack - binding_summary.coverage_floor_slack
    )

    slack_ranking = " < ".join(
        (
            f"`{_format_design_key(*summary.design_key)} -> "
            f"{summary.mean_nonparametric_coverage:.3f} "
            f"({_format_signed(summary.coverage_floor_slack)})`"
        )
        for summary in design_floor_summaries
    )
    canonical_floor_slack_digest = (
        f"- bounded-slice coverage floor slack ranking: {slack_ranking}",
        "- supported floor ceiling from current bounded slice: "
        f"`{supported_floor_ceiling:.3f}`; canonical floor "
        f"`{coverage_floor:.3f}` overshoots by "
        f"`{canonical_floor_shortfall:.3f}`",
        "- binding design: "
        f"`{_format_design_key(*binding_summary.design_key)}`; best design: "
        f"`{_format_design_key(*best_summary.design_key)}`; slack spread "
        f"`{slack_spread:.3f}`",
    )

    return Phase7MonteCarloWideningPolicyFloorSlackProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-floor-slack-probe",
        policy_digest=resolved_policy.to_digest(),
        coverage_floor=coverage_floor,
        supported_floor_ceiling=supported_floor_ceiling,
        canonical_floor_shortfall=canonical_floor_shortfall,
        binding_design=binding_summary.design_key,
        best_design=best_summary.design_key,
        slack_spread=slack_spread,
        design_floor_summaries=design_floor_summaries,
        canonical_floor_slack_digest=canonical_floor_slack_digest,
    )


def _build_floor_slack_report_from_guard_probe(
    guard_probe,
    *,
    policy: Phase7MonteCarloWideningPolicy,
) -> Phase7MonteCarloWideningPolicyFloorSlackProbeReport:
    coverage_floor = float(policy.min_nonparametric_coverage)
    if not guard_probe.design_guard_summaries:
        raise ValueError("policy floor slack probe requires bounded widening summaries")

    supported_floor_ceiling = min(
        summary.mean_nonparametric_coverage
        for summary in guard_probe.design_guard_summaries
    )
    ordered_guard_summaries = tuple(
        sorted(
            guard_probe.design_guard_summaries,
            key=lambda summary: (
                summary.coverage_floor_slack,
                summary.mean_nonparametric_coverage,
                summary.dgp_name,
                summary.n_obs,
                summary.p,
            ),
        )
    )
    design_floor_summaries = tuple(
        Phase7MonteCarloWideningPolicyFloorSlackDesignSummary(
            dgp_name=summary.dgp_name,
            n_obs=summary.n_obs,
            p=summary.p,
            mean_nonparametric_coverage=summary.mean_nonparametric_coverage,
            coverage_floor=coverage_floor,
            coverage_floor_slack=summary.coverage_floor_slack,
            shortfall_to_canonical_floor=max(
                coverage_floor - summary.mean_nonparametric_coverage,
                0.0,
            ),
            floor_ceiling_headroom=(
                summary.mean_nonparametric_coverage - supported_floor_ceiling
            ),
        )
        for summary in ordered_guard_summaries
    )

    binding_summary = design_floor_summaries[0]
    best_summary = max(
        design_floor_summaries,
        key=lambda summary: (
            summary.coverage_floor_slack,
            summary.mean_nonparametric_coverage,
            summary.dgp_name,
            summary.n_obs,
            summary.p,
        ),
    )
    canonical_floor_shortfall = max(coverage_floor - supported_floor_ceiling, 0.0)
    slack_spread = (
        best_summary.coverage_floor_slack - binding_summary.coverage_floor_slack
    )
    slack_ranking = " < ".join(
        (
            f"`{_format_design_key(*summary.design_key)} -> "
            f"{summary.mean_nonparametric_coverage:.3f} "
            f"({_format_signed(summary.coverage_floor_slack)})`"
        )
        for summary in design_floor_summaries
    )
    canonical_floor_slack_digest = (
        f"- bounded-slice coverage floor slack ranking: {slack_ranking}",
        "- supported floor ceiling from current bounded slice: "
        f"`{supported_floor_ceiling:.3f}`; canonical floor "
        f"`{coverage_floor:.3f}` overshoots by "
        f"`{canonical_floor_shortfall:.3f}`",
        "- binding design: "
        f"`{_format_design_key(*binding_summary.design_key)}`; best design: "
        f"`{_format_design_key(*best_summary.design_key)}`; slack spread "
        f"`{slack_spread:.3f}`",
    )
    return Phase7MonteCarloWideningPolicyFloorSlackProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-floor-slack-probe",
        policy_digest=policy.to_digest(),
        coverage_floor=coverage_floor,
        supported_floor_ceiling=supported_floor_ceiling,
        canonical_floor_shortfall=canonical_floor_shortfall,
        binding_design=binding_summary.design_key,
        best_design=best_summary.design_key,
        slack_spread=slack_spread,
        design_floor_summaries=design_floor_summaries,
        canonical_floor_slack_digest=canonical_floor_slack_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_floor_slack_probe() -> (
    Phase7MonteCarloWideningPolicyFloorSlackProbeReport
):
    policy = build_phase7_canonical_monte_carlo_widening_policy()
    try:
        return build_phase7_monte_carlo_widening_policy_floor_slack_probe_report(
            run_phase7_monte_carlo_runtime_probe(),
            policy=policy,
        )
    except ValueError as exc:
        if str(exc) != "runtime probe has no bounded widening slice summaries":
            raise
    return _build_floor_slack_report_from_guard_probe(
        run_phase7_monte_carlo_widening_policy_guard_probe(),
        policy=policy,
    )
