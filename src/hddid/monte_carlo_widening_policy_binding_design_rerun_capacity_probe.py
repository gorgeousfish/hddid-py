from __future__ import annotations

from dataclasses import dataclass
import math

from .monte_carlo_widening_policy_guard_probe import (
    Phase7MonteCarloWideningPolicyGuardProbeReport,
    run_phase7_monte_carlo_widening_policy_guard_probe,
)
from .validation import (
    MonteCarloRuntimeProbeReport,
    run_phase7_monte_carlo_runtime_probe,
)


def _format_design_key(design: tuple[str, int, int]) -> str:
    return f"{design[0]}/{design[1]}/{design[2]}"


def _binding_design_runtime_mean_seconds(
    runtime_probe_report: MonteCarloRuntimeProbeReport,
    binding_design: tuple[str, int, int],
) -> float:
    for summary in runtime_probe_report.design_summaries:
        design_key = (summary.dgp_name, int(summary.n_obs), int(summary.p))
        if design_key != binding_design:
            continue
        runtime_mean_seconds = summary.runtime_mean_seconds
        if runtime_mean_seconds is None:
            raise ValueError("binding design is missing runtime_mean_seconds")
        number = float(runtime_mean_seconds)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError("binding design requires positive runtime_mean_seconds")
        return number
    raise ValueError("binding design is missing from runtime probe summaries")


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
        dgp, n_obs, p = self.binding_design
        self.binding_design = (str(dgp).strip(), int(n_obs), int(p))
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

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "binding_design_runtime_mean_seconds": (
                self.binding_design_runtime_mean_seconds
            ),
            "runtime_budget_headroom_seconds": self.runtime_budget_headroom_seconds,
            "random_state_headroom": self.random_state_headroom,
            "runtime_limited_additional_reruns": (
                self.runtime_limited_additional_reruns
            ),
            "seed_limited_additional_reruns": self.seed_limited_additional_reruns,
            "effective_additional_reruns": self.effective_additional_reruns,
            "limiting_budget": self.limiting_budget,
            "current_implication": self.current_implication,
            "canonical_rerun_capacity_digest": list(
                self.canonical_rerun_capacity_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe_report(
    *,
    guard_probe_report: Phase7MonteCarloWideningPolicyGuardProbeReport,
    runtime_probe_report: MonteCarloRuntimeProbeReport,
) -> Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport:
    if not guard_probe_report.binding_designs:
        raise ValueError("rerun capacity requires at least one binding design")
    binding_design = guard_probe_report.binding_designs[0]
    runtime_mean_seconds = _binding_design_runtime_mean_seconds(
        runtime_probe_report,
        binding_design,
    )
    runtime_headroom = float(
        guard_probe_report.policy_max_total_runtime_seconds
        - runtime_probe_report.total_runtime_seconds
    )
    random_state_headroom = int(
        guard_probe_report.policy_max_random_states
        - len(tuple(runtime_probe_report.random_states))
    )
    runtime_limited = max(0, int(math.floor(runtime_headroom / runtime_mean_seconds)))
    seed_limited = max(0, random_state_headroom)
    effective = min(runtime_limited, seed_limited)
    limiting_budget = (
        "runtime_budget"
        if runtime_limited < seed_limited
        else "random_state_budget"
    )
    current_implication = "spend-remaining-fresh-reruns-on-binding-design"
    design_label = _format_design_key(binding_design)
    capacity_mode = (
        "runtime-limited"
        if limiting_budget == "runtime_budget"
        else "seed-limited"
    )
    other_mode = (
        "seed-limited"
        if limiting_budget == "runtime_budget"
        else "runtime-limited"
    )
    canonical_digest = (
        (
            "- binding design rerun capacity is still "
            f"{capacity_mode}, not {other_mode}: `{design_label}` averages "
            f"`{runtime_mean_seconds:.3f}` seconds per successful bounded rerun, "
            f"so runtime headroom `{runtime_headroom:.3f}` still fits "
            f"`{runtime_limited}` additional reruns while the canonical seed cap "
            f"leaves only `{seed_limited}` fresh random states"
        ),
        (
            "- effective fresh rerun capacity therefore stays at "
            f"`{effective}` on the binding design before the current "
            "`bounded-n500-p50` policy object is exhausted"
        ),
        (
            f"- current implication: `{current_implication}`; if main execution "
            f"wants new runtime evidence, spend the remaining fresh reruns on "
            f"`{design_label}` rather than on already-passing `DGP1/500/50` or "
            "blocked full-matrix designs"
        ),
    )
    return Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-binding-design-rerun-capacity-probe"
        ),
        policy_digest=guard_probe_report.policy_digest,
        binding_design=binding_design,
        binding_design_runtime_mean_seconds=runtime_mean_seconds,
        runtime_budget_headroom_seconds=runtime_headroom,
        random_state_headroom=random_state_headroom,
        runtime_limited_additional_reruns=runtime_limited,
        seed_limited_additional_reruns=seed_limited,
        effective_additional_reruns=effective,
        limiting_budget=limiting_budget,
        current_implication=current_implication,
        canonical_rerun_capacity_digest=canonical_digest,
    )


def run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe() -> (
    Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport
):
    runtime_probe = run_phase7_monte_carlo_runtime_probe()
    return build_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe_report(
        guard_probe_report=run_phase7_monte_carlo_widening_policy_guard_probe(),
        runtime_probe_report=runtime_probe,
    )
