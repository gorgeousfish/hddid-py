from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import floor

from .monte_carlo_widening_policy_guard_probe import (
    Phase7MonteCarloWideningPolicyGuardProbeReport,
    run_phase7_monte_carlo_widening_policy_guard_probe,
)
from .validation import (
    MonteCarloRuntimeProbeReport,
    run_phase7_monte_carlo_runtime_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{dgp_name}/{n_obs}/{p}"


def _policy_label(policy_digest: tuple[str, ...]) -> str:
    for item in policy_digest:
        if item.startswith("label="):
            return item.removeprefix("label=")
    raise ValueError("policy digest is missing label=... token")


def _nonnegative_floor_ratio(numerator: float, denominator: float) -> int:
    if denominator <= 0.0:
        raise ValueError("binding design runtime_mean_seconds must be positive")
    if numerator <= 0.0:
        return 0
    return max(0, int(floor(numerator / denominator)))


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    binding_design_runtime_mean_seconds: float
    runtime_budget_headroom_seconds: float
    random_state_headroom: int
    runtime_limited_additional_reruns: int
    seed_limited_additional_reruns: int
    effective_additional_reruns: int
    limiting_budget: str
    current_implication: str
    canonical_rerun_capacity_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.binding_design_runtime_mean_seconds = float(
            self.binding_design_runtime_mean_seconds
        )
        self.runtime_budget_headroom_seconds = float(
            self.runtime_budget_headroom_seconds
        )
        self.random_state_headroom = int(self.random_state_headroom)
        self.runtime_limited_additional_reruns = int(
            self.runtime_limited_additional_reruns
        )
        self.seed_limited_additional_reruns = int(self.seed_limited_additional_reruns)
        self.effective_additional_reruns = int(self.effective_additional_reruns)
        self.limiting_budget = str(self.limiting_budget).strip()
        self.current_implication = str(self.current_implication).strip()
        self.canonical_rerun_capacity_digest = tuple(
            str(line).rstrip() for line in self.canonical_rerun_capacity_digest
        )


def build_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe_report(
    *,
    guard_probe_report: Phase7MonteCarloWideningPolicyGuardProbeReport | None = None,
    runtime_probe_report: MonteCarloRuntimeProbeReport | None = None,
) -> Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport:
    resolved_guard = (
        run_phase7_monte_carlo_widening_policy_guard_probe()
        if guard_probe_report is None
        else guard_probe_report
    )
    resolved_runtime_probe = (
        run_phase7_monte_carlo_runtime_probe()
        if runtime_probe_report is None
        else runtime_probe_report
    )

    if not resolved_guard.binding_designs:
        raise ValueError("rerun capacity probe requires at least one binding design")
    if len(resolved_guard.binding_designs) != 1:
        raise ValueError("rerun capacity probe requires exactly one binding design")

    binding_design = resolved_guard.binding_designs[0]
    binding_summary = resolved_runtime_probe.design_summary(*binding_design)
    if binding_summary.runtime_mean_seconds is None:
        raise ValueError("binding design is missing runtime_mean_seconds")

    binding_runtime_mean_seconds = float(binding_summary.runtime_mean_seconds)
    runtime_budget_headroom_seconds = float(
        resolved_guard.runtime_budget_headroom_seconds
    )
    random_state_headroom = max(0, int(resolved_guard.random_state_headroom))
    runtime_limited_additional_reruns = _nonnegative_floor_ratio(
        runtime_budget_headroom_seconds,
        binding_runtime_mean_seconds,
    )
    seed_limited_additional_reruns = random_state_headroom
    effective_additional_reruns = min(
        runtime_limited_additional_reruns,
        seed_limited_additional_reruns,
    )

    if runtime_limited_additional_reruns < seed_limited_additional_reruns:
        limiting_budget = "runtime_budget"
    elif seed_limited_additional_reruns < runtime_limited_additional_reruns:
        limiting_budget = "random_state_budget"
    else:
        limiting_budget = "shared_budget"

    current_implication = "spend-remaining-fresh-reruns-on-binding-design"
    policy_label = _policy_label(resolved_guard.policy_digest)
    binding_design_key = _format_design_key(*binding_design)
    canonical_rerun_capacity_digest = (
        "- binding design rerun capacity is still "
        f"{'seed-limited, not runtime-limited' if limiting_budget == 'random_state_budget' else 'runtime-limited, not seed-limited' if limiting_budget == 'runtime_budget' else 'jointly limited'}: "
        f"`{binding_design_key}` averages `{binding_runtime_mean_seconds:.3f}` seconds per successful bounded rerun, "
        f"so runtime headroom `{runtime_budget_headroom_seconds:.3f}` still fits "
        f"`{runtime_limited_additional_reruns}` additional reruns while the canonical seed cap leaves only "
        f"`{seed_limited_additional_reruns}` fresh random states",
        "- effective fresh rerun capacity therefore stays at "
        f"`{effective_additional_reruns}` on the binding design before the current "
        f"`{policy_label}` policy object is exhausted",
        "- current implication: `spend-remaining-fresh-reruns-on-binding-design`; "
        f"if main execution wants new runtime evidence, spend the remaining fresh reruns on `{binding_design_key}` "
        "rather than on already-passing `DGP1/500/50` or blocked full-matrix designs",
    )

    return Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-binding-design-rerun-capacity-probe"
        ),
        policy_digest=resolved_guard.policy_digest,
        binding_design=binding_design,
        binding_design_runtime_mean_seconds=binding_runtime_mean_seconds,
        runtime_budget_headroom_seconds=runtime_budget_headroom_seconds,
        random_state_headroom=random_state_headroom,
        runtime_limited_additional_reruns=runtime_limited_additional_reruns,
        seed_limited_additional_reruns=seed_limited_additional_reruns,
        effective_additional_reruns=effective_additional_reruns,
        limiting_budget=limiting_budget,
        current_implication=current_implication,
        canonical_rerun_capacity_digest=canonical_rerun_capacity_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe() -> (
    Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport
):
    return build_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe_report()
