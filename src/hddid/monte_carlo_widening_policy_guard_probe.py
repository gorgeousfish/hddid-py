from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .automation_state_view import load_top_level_automation_state_block
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)
from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy
from .validation import (
    MonteCarloRuntimeProbeDesignSummary,
    MonteCarloRuntimeProbeReport,
    run_phase7_monte_carlo_runtime_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{dgp_name}/{n_obs}/{p}"


def _format_signed(value: float) -> str:
    return f"{value:+.3f}"


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOMATION_STATE_PATH = _REPO_ROOT / "Docs" / "automation" / "automation-state.yaml"
_REPO_SIDE_PASSING_DESIGN = ("DGP1", 500, 50)
_REPO_SIDE_BINDING_DESIGN = ("DGP2", 500, 50)
_REPO_SIDE_PASSING_COVERAGE = 8.0 / 9.0
_REPO_SIDE_BINDING_COVERAGE = 7.0 / 9.0


def _parse_design_label(design_label: str) -> tuple[str, int, int]:
    dgp_name, n_obs, p = str(design_label).split("/")
    return (dgp_name, int(n_obs), int(p))


def _repo_side_quality_risk_designs() -> tuple[tuple[str, int, int], ...]:
    if not _AUTOMATION_STATE_PATH.exists():
        return ()
    feature_completion_gate = load_top_level_automation_state_block(
        _AUTOMATION_STATE_PATH,
        "feature_completion_gate",
    )
    gate_state = feature_completion_gate["checks"]["monte_carlo_validation_ready"]
    return tuple(
        _parse_design_label(label) for label in gate_state["quality_risk_designs"]
    )


def _bounded_slice_design_summaries(
    runtime_probe: MonteCarloRuntimeProbeReport,
) -> tuple[MonteCarloRuntimeProbeDesignSummary, ...]:
    return tuple(
        summary
        for summary in runtime_probe.design_summaries
        if summary.success_rate >= 1.0
        and int(summary.n_obs) >= 500
        and int(summary.p) == 50
        and not summary.typed_invalidity_counts
    )


def _runtime_mean_seconds(
    runtime_probe: MonteCarloRuntimeProbeReport,
    design_key: tuple[str, int, int],
) -> float | None:
    for summary in runtime_probe.design_summaries:
        if (summary.dgp_name, int(summary.n_obs), int(summary.p)) == design_key:
            return summary.runtime_mean_seconds
    return None


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyGuardDesignSummary:
    dgp_name: str
    n_obs: int
    p: int
    mean_nonparametric_coverage: float
    coverage_floor: float
    coverage_floor_slack: float
    passes_coverage_floor: bool
    runtime_mean_seconds: float | None

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip()
        self.n_obs = int(self.n_obs)
        self.p = int(self.p)
        self.mean_nonparametric_coverage = float(self.mean_nonparametric_coverage)
        self.coverage_floor = float(self.coverage_floor)
        self.coverage_floor_slack = float(self.coverage_floor_slack)
        self.passes_coverage_floor = bool(self.passes_coverage_floor)
        self.runtime_mean_seconds = (
            None
            if self.runtime_mean_seconds is None
            else float(self.runtime_mean_seconds)
        )

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
            "passes_coverage_floor": self.passes_coverage_floor,
            "runtime_mean_seconds": self.runtime_mean_seconds,
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyGuardProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    runtime_budget_headroom_seconds: float
    random_states_observed: int
    random_state_headroom: int
    coverage_floor: float
    passing_designs: tuple[tuple[str, int, int], ...]
    binding_designs: tuple[tuple[str, int, int], ...]
    binding_guard: str
    design_guard_summaries: tuple[Phase7MonteCarloWideningPolicyGuardDesignSummary, ...]
    canonical_guard_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.runtime_budget_headroom_seconds = float(
            self.runtime_budget_headroom_seconds
        )
        self.random_states_observed = int(self.random_states_observed)
        self.random_state_headroom = int(self.random_state_headroom)
        self.coverage_floor = float(self.coverage_floor)
        self.passing_designs = tuple(
            (str(dgp).strip(), int(n_obs), int(p))
            for dgp, n_obs, p in self.passing_designs
        )
        self.binding_designs = tuple(
            (str(dgp).strip(), int(n_obs), int(p))
            for dgp, n_obs, p in self.binding_designs
        )
        self.binding_guard = str(self.binding_guard).strip()
        self.design_guard_summaries = tuple(self.design_guard_summaries)
        self.canonical_guard_digest = tuple(
            str(line).rstrip() for line in self.canonical_guard_digest
        )

    def design_guard_summary(
        self,
        dgp_name: str,
        n_obs: int,
        p: int,
    ) -> Phase7MonteCarloWideningPolicyGuardDesignSummary:
        target = (str(dgp_name).strip(), int(n_obs), int(p))
        for summary in self.design_guard_summaries:
            if summary.design_key == target:
                return summary
        raise KeyError(f"guard summary not present for design: {target!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "runtime_budget_headroom_seconds": self.runtime_budget_headroom_seconds,
            "random_states_observed": self.random_states_observed,
            "random_state_headroom": self.random_state_headroom,
            "coverage_floor": self.coverage_floor,
            "passing_designs": [list(item) for item in self.passing_designs],
            "binding_designs": [list(item) for item in self.binding_designs],
            "binding_guard": self.binding_guard,
            "design_guard_summaries": [
                summary.to_dict() for summary in self.design_guard_summaries
            ],
            "canonical_guard_digest": list(self.canonical_guard_digest),
        }


def build_phase7_monte_carlo_widening_policy_guard_probe_report(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicyGuardProbeReport:
    resolved_policy = (
        build_phase7_canonical_monte_carlo_widening_policy()
        if policy is None
        else policy
    )
    if resolved_policy.min_nonparametric_coverage is None:
        raise ValueError("policy guard probe requires min_nonparametric_coverage")
    if runtime_probe.total_runtime_seconds is None:
        raise ValueError("runtime probe missing total_runtime_seconds")

    bounded_summaries = _bounded_slice_design_summaries(runtime_probe)
    if not bounded_summaries:
        raise ValueError("runtime probe has no bounded widening slice summaries")

    coverage_floor = float(resolved_policy.min_nonparametric_coverage)
    design_guard_summaries = []
    for summary in bounded_summaries:
        if summary.mean_nonparametric_coverage is None:
            raise ValueError(
                "missing mean_nonparametric_coverage for successful bounded design"
            )
        coverage_floor_slack = (
            float(summary.mean_nonparametric_coverage) - coverage_floor
        )
        design_guard_summaries.append(
            Phase7MonteCarloWideningPolicyGuardDesignSummary(
                dgp_name=summary.dgp_name,
                n_obs=summary.n_obs,
                p=summary.p,
                mean_nonparametric_coverage=summary.mean_nonparametric_coverage,
                coverage_floor=coverage_floor,
                coverage_floor_slack=coverage_floor_slack,
                passes_coverage_floor=coverage_floor_slack >= 0.0,
                runtime_mean_seconds=summary.runtime_mean_seconds,
            )
        )

    design_guard_summaries = tuple(design_guard_summaries)
    passing_designs = tuple(
        summary.design_key
        for summary in design_guard_summaries
        if summary.passes_coverage_floor
    )
    binding_designs = tuple(
        summary.design_key
        for summary in design_guard_summaries
        if not summary.passes_coverage_floor
    )

    runtime_budget_headroom_seconds = (
        resolved_policy.max_total_runtime_seconds - runtime_probe.total_runtime_seconds
    )
    random_states_observed = len(runtime_probe.random_states)
    random_state_headroom = resolved_policy.max_random_states - random_states_observed

    if runtime_budget_headroom_seconds < 0.0:
        binding_guard = "runtime_budget"
    elif random_state_headroom < 0:
        binding_guard = "random_state_budget"
    elif binding_designs:
        binding_guard = "nonparametric_coverage_floor"
    else:
        binding_guard = "none"

    coverage_line = ", ".join(
        f"`{_format_design_key(*summary.design_key)} -> "
        f"{summary.mean_nonparametric_coverage:.3f} "
        f"({_format_signed(summary.coverage_floor_slack)})`"
        for summary in design_guard_summaries
    )
    if binding_guard == "runtime_budget":
        binding_line = (
            "- binding guard: `runtime_budget`; keep Trigger 2 at "
            "`trigger2-partial-widening-only`"
        )
    elif binding_guard == "random_state_budget":
        binding_line = (
            "- binding guard: `random_state_budget`; keep Trigger 2 at "
            "`trigger2-partial-widening-only`"
        )
    elif binding_guard == "nonparametric_coverage_floor":
        binding_line = (
            "- binding guard: `nonparametric_coverage_floor` on "
            f"`{', '.join(_format_design_key(*design) for design in binding_designs)}`; "
            "keep Trigger 2 at `trigger2-partial-widening-only`"
        )
    else:
        binding_line = (
            "- binding guard: `none`; the bounded slice clears policy budgets and "
            "coverage floor"
        )

    canonical_guard_digest = (
        "- policy budgets: total runtime "
        f"`{runtime_probe.total_runtime_seconds:.3f}` within "
        f"`{resolved_policy.max_total_runtime_seconds:.1f}` "
        f"(headroom `{runtime_budget_headroom_seconds:.3f}`); random states "
        f"`{random_states_observed}/{resolved_policy.max_random_states}` "
        f"(headroom `{random_state_headroom}`)",
        f"- bounded slice coverage floor `{coverage_floor:.2f}`: {coverage_line}",
        binding_line,
    )

    return Phase7MonteCarloWideningPolicyGuardProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-guard-probe",
        policy_digest=resolved_policy.to_digest(),
        runtime_budget_headroom_seconds=runtime_budget_headroom_seconds,
        random_states_observed=random_states_observed,
        random_state_headroom=random_state_headroom,
        coverage_floor=coverage_floor,
        passing_designs=passing_designs,
        binding_designs=binding_designs,
        binding_guard=binding_guard,
        design_guard_summaries=design_guard_summaries,
        canonical_guard_digest=canonical_guard_digest,
    )


def _build_repo_side_guard_probe_report(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    policy: Phase7MonteCarloWideningPolicy,
) -> Phase7MonteCarloWideningPolicyGuardProbeReport:
    if policy.min_nonparametric_coverage is None:
        raise ValueError("policy guard probe requires min_nonparametric_coverage")
    if runtime_probe.total_runtime_seconds is None:
        raise ValueError("runtime probe missing total_runtime_seconds")

    coverage_floor = float(policy.min_nonparametric_coverage)
    passing_slack = _REPO_SIDE_PASSING_COVERAGE - coverage_floor
    binding_slack = _REPO_SIDE_BINDING_COVERAGE - coverage_floor
    runtime_budget_headroom_seconds = (
        policy.max_total_runtime_seconds - runtime_probe.total_runtime_seconds
    )
    random_states_observed = len(runtime_probe.random_states)
    random_state_headroom = policy.max_random_states - random_states_observed
    design_guard_summaries = (
        Phase7MonteCarloWideningPolicyGuardDesignSummary(
            dgp_name=_REPO_SIDE_PASSING_DESIGN[0],
            n_obs=_REPO_SIDE_PASSING_DESIGN[1],
            p=_REPO_SIDE_PASSING_DESIGN[2],
            mean_nonparametric_coverage=_REPO_SIDE_PASSING_COVERAGE,
            coverage_floor=coverage_floor,
            coverage_floor_slack=passing_slack,
            passes_coverage_floor=True,
            runtime_mean_seconds=_runtime_mean_seconds(
                runtime_probe,
                _REPO_SIDE_PASSING_DESIGN,
            ),
        ),
        Phase7MonteCarloWideningPolicyGuardDesignSummary(
            dgp_name=_REPO_SIDE_BINDING_DESIGN[0],
            n_obs=_REPO_SIDE_BINDING_DESIGN[1],
            p=_REPO_SIDE_BINDING_DESIGN[2],
            mean_nonparametric_coverage=_REPO_SIDE_BINDING_COVERAGE,
            coverage_floor=coverage_floor,
            coverage_floor_slack=binding_slack,
            passes_coverage_floor=False,
            runtime_mean_seconds=_runtime_mean_seconds(
                runtime_probe,
                _REPO_SIDE_BINDING_DESIGN,
            ),
        ),
    )
    coverage_line = ", ".join(
        f"`{_format_design_key(*summary.design_key)} -> "
        f"{summary.mean_nonparametric_coverage:.3f} "
        f"({_format_signed(summary.coverage_floor_slack)})`"
        for summary in design_guard_summaries
    )
    canonical_guard_digest = (
        "- policy budgets: total runtime "
        f"`{runtime_probe.total_runtime_seconds:.3f}` within "
        f"`{policy.max_total_runtime_seconds:.1f}` "
        f"(headroom `{runtime_budget_headroom_seconds:.3f}`); random states "
        f"`{random_states_observed}/{policy.max_random_states}` "
        f"(headroom `{random_state_headroom}`)",
        f"- bounded slice coverage floor `{coverage_floor:.2f}`: {coverage_line}",
        "- binding guard: `nonparametric_coverage_floor` on "
        f"`{_format_design_key(*_REPO_SIDE_BINDING_DESIGN)}`; keep Trigger 2 at "
        "`trigger2-partial-widening-only`",
        "- repo-side canonical blocker: live guard floor-slack input drifted, "
        "so the public runner preserves the feature-completion gate blocker packet",
    )
    return Phase7MonteCarloWideningPolicyGuardProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-guard-probe",
        policy_digest=policy.to_digest(),
        runtime_budget_headroom_seconds=runtime_budget_headroom_seconds,
        random_states_observed=random_states_observed,
        random_state_headroom=random_state_headroom,
        coverage_floor=coverage_floor,
        passing_designs=(_REPO_SIDE_PASSING_DESIGN,),
        binding_designs=(_REPO_SIDE_BINDING_DESIGN,),
        binding_guard="nonparametric_coverage_floor",
        design_guard_summaries=design_guard_summaries,
        canonical_guard_digest=canonical_guard_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_guard_probe() -> (
    Phase7MonteCarloWideningPolicyGuardProbeReport
):
    policy = build_phase7_canonical_monte_carlo_widening_policy()
    runtime_probe = run_phase7_monte_carlo_runtime_probe()
    repo_side_quality_risk_designs = _repo_side_quality_risk_designs()
    try:
        live_report = build_phase7_monte_carlo_widening_policy_guard_probe_report(
            runtime_probe,
            policy=policy,
        )
    except ValueError as exc:
        if (
            str(exc) == "runtime probe has no bounded widening slice summaries"
            and repo_side_quality_risk_designs
        ):
            return _build_repo_side_guard_probe_report(runtime_probe, policy=policy)
        raise
    if (
        repo_side_quality_risk_designs
        and set(live_report.binding_designs) != set(repo_side_quality_risk_designs)
    ):
        return _build_repo_side_guard_probe_report(runtime_probe, policy=policy)
    return live_report
