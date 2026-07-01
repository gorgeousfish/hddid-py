from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_trigger_gate import (
    Phase7MonteCarloWideningPolicy,
    Phase7MonteCarloWideningTriggerGateReport,
    _build_state_backed_trigger_gate_report,
    build_phase7_monte_carlo_widening_trigger_gate_report,
    run_phase7_monte_carlo_widening_trigger_gate,
)


def _format_design_keys(
    design_keys: tuple[tuple[str, int, int], ...],
) -> str:
    if not design_keys:
        return "none"
    return ", ".join(f"{dgp}/{n_obs}/{p}" for dgp, n_obs, p in design_keys)


def _bounded_quality_risk_designs(
    target_gate: Phase7MonteCarloWideningTriggerGateReport,
) -> tuple[tuple[str, int, int], ...]:
    bounded_designs = set(target_gate.stable_partial_designs)
    return tuple(
        design
        for design in target_gate.quality_risk_designs
        if design in bounded_designs
    )


def build_phase7_canonical_monte_carlo_widening_policy() -> (
    Phase7MonteCarloWideningPolicy
):
    return Phase7MonteCarloWideningPolicy(
        label="bounded-n500-p50",
        max_total_runtime_seconds=240.0,
        max_random_states=8,
        stop_on_first_typed_invalidity=True,
        min_nonparametric_coverage=0.85,
    )


def decode_phase7_monte_carlo_widening_policy_digest(
    policy_digest: tuple[str, ...],
) -> Phase7MonteCarloWideningPolicy | None:
    if not policy_digest:
        return None
    parsed: dict[str, str] = {}
    for item in policy_digest:
        if "=" not in item:
            raise ValueError(f"policy digest item must use key=value syntax: {item!r}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    required = {
        "label",
        "max_total_runtime_seconds",
        "max_random_states",
        "stop_on_first_typed_invalidity",
    }
    missing = sorted(required.difference(parsed))
    if missing:
        raise ValueError(f"policy digest missing required keys: {', '.join(missing)}")
    return Phase7MonteCarloWideningPolicy(
        label=parsed["label"],
        max_total_runtime_seconds=float(parsed["max_total_runtime_seconds"]),
        max_random_states=int(parsed["max_random_states"]),
        stop_on_first_typed_invalidity=parsed["stop_on_first_typed_invalidity"],
        min_nonparametric_coverage=(
            None
            if "min_nonparametric_coverage" not in parsed
            else float(parsed["min_nonparametric_coverage"])
        ),
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyTransition:
    trigger_label: str
    current_gate_status: str
    target_gate_status: str
    policy_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.trigger_label = str(self.trigger_label).strip()
        self.current_gate_status = str(self.current_gate_status).strip()
        self.target_gate_status = str(self.target_gate_status).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)

    def to_dict(self) -> dict[str, object]:
        canonical_policy = decode_phase7_monte_carlo_widening_policy_digest(
            self.policy_digest
        )
        return {
            "trigger_label": self.trigger_label,
            "current_gate_status": self.current_gate_status,
            "target_gate_status": self.target_gate_status,
            "policy_digest": list(self.policy_digest),
            "canonical_policy": (
                None if canonical_policy is None else canonical_policy.to_dict()
            ),
        }


def build_phase7_monte_carlo_widening_policy_transition(
    current_gate: Phase7MonteCarloWideningTriggerGateReport,
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicyTransition:
    resolved_policy = (
        build_phase7_canonical_monte_carlo_widening_policy()
        if policy is None
        else policy
    )
    if current_gate.gate_status == "trigger2-policy-missing":
        target_gate_status = "trigger2-partial-widening-only"
        policy_digest = resolved_policy.to_digest()
    else:
        target_gate_status = current_gate.gate_status
        policy_digest = ()
    return Phase7MonteCarloWideningPolicyTransition(
        trigger_label="trigger2-policy-spec",
        current_gate_status=current_gate.gate_status,
        target_gate_status=target_gate_status,
        policy_digest=policy_digest,
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicySpecReport:
    stage_label: str
    trigger_label: str
    current_gate_status: str
    target_gate_status: str
    stable_partial_designs: tuple[tuple[str, int, int], ...]
    blocked_full_matrix_designs: tuple[tuple[str, int, int], ...]
    quality_risk_designs: tuple[tuple[str, int, int], ...]
    policy_digest: tuple[str, ...]
    canonical_policy_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.trigger_label = str(self.trigger_label).strip()
        self.current_gate_status = str(self.current_gate_status).strip()
        self.target_gate_status = str(self.target_gate_status).strip()
        self.stable_partial_designs = tuple(
            (str(dgp).strip(), int(n_obs), int(p))
            for dgp, n_obs, p in self.stable_partial_designs
        )
        self.blocked_full_matrix_designs = tuple(
            (str(dgp).strip(), int(n_obs), int(p))
            for dgp, n_obs, p in self.blocked_full_matrix_designs
        )
        self.quality_risk_designs = tuple(
            (str(dgp).strip(), int(n_obs), int(p))
            for dgp, n_obs, p in self.quality_risk_designs
        )
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.canonical_policy_digest = tuple(
            str(item).rstrip() for item in self.canonical_policy_digest
        )

    def to_dict(self) -> dict[str, object]:
        canonical_policy = decode_phase7_monte_carlo_widening_policy_digest(
            self.policy_digest
        )
        return {
            "stage_label": self.stage_label,
            "trigger_label": self.trigger_label,
            "current_gate_status": self.current_gate_status,
            "target_gate_status": self.target_gate_status,
            "stable_partial_designs": [
                list(item) for item in self.stable_partial_designs
            ],
            "blocked_full_matrix_designs": [
                list(item) for item in self.blocked_full_matrix_designs
            ],
            "quality_risk_designs": [list(item) for item in self.quality_risk_designs],
            "policy_digest": list(self.policy_digest),
            "canonical_policy": (
                None if canonical_policy is None else canonical_policy.to_dict()
            ),
            "canonical_policy_label": (
                None if canonical_policy is None else canonical_policy.label
            ),
            "canonical_policy_max_total_runtime_seconds": (
                None
                if canonical_policy is None
                else canonical_policy.max_total_runtime_seconds
            ),
            "canonical_policy_max_random_states": (
                None if canonical_policy is None else canonical_policy.max_random_states
            ),
            "canonical_policy_stop_on_first_typed_invalidity": (
                None
                if canonical_policy is None
                else canonical_policy.stop_on_first_typed_invalidity
            ),
            "canonical_policy_min_nonparametric_coverage": (
                None
                if canonical_policy is None
                else canonical_policy.min_nonparametric_coverage
            ),
            "canonical_policy_digest": list(self.canonical_policy_digest),
        }


def build_phase7_monte_carlo_widening_policy_spec_report(
    current_gate: Phase7MonteCarloWideningTriggerGateReport,
    target_gate: Phase7MonteCarloWideningTriggerGateReport,
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicySpecReport:
    resolved_policy = (
        build_phase7_canonical_monte_carlo_widening_policy()
        if policy is None
        else policy
    )
    transition = build_phase7_monte_carlo_widening_policy_transition(
        current_gate,
        policy=resolved_policy,
    )
    if target_gate.gate_status != transition.target_gate_status:
        raise ValueError(
            "target gate status does not match canonical trigger2 policy transition"
        )
    policy_digest = transition.policy_digest
    bounded_quality_risk_designs = _bounded_quality_risk_designs(target_gate)
    canonical_policy_digest = (
        f"- current Trigger 2 gate: `{current_gate.gate_status}`",
        "- canonical policy: " + ", ".join(f"`{item}`" for item in policy_digest),
        "- bounded widening slice: "
        f"`{_format_design_keys(target_gate.stable_partial_designs)}`; "
        "blocked full-matrix slice remains "
        f"`{_format_design_keys(target_gate.blocked_full_matrix_designs)}`",
        "- quality risk design: "
        f"`{_format_design_keys(bounded_quality_risk_designs)}`; "
        f"target gate after policy: `{target_gate.gate_status}`",
    )
    return Phase7MonteCarloWideningPolicySpecReport(
        stage_label="phase7-monte-carlo-widening-policy-spec",
        trigger_label=transition.trigger_label,
        current_gate_status=transition.current_gate_status,
        target_gate_status=transition.target_gate_status,
        stable_partial_designs=target_gate.stable_partial_designs,
        blocked_full_matrix_designs=target_gate.blocked_full_matrix_designs,
        quality_risk_designs=bounded_quality_risk_designs,
        policy_digest=policy_digest,
        canonical_policy_digest=canonical_policy_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_spec() -> (
    Phase7MonteCarloWideningPolicySpecReport
):
    policy = build_phase7_canonical_monte_carlo_widening_policy()
    current_gate = run_phase7_monte_carlo_widening_trigger_gate()
    target_gate = _build_state_backed_trigger_gate_report(policy)
    return build_phase7_monte_carlo_widening_policy_spec_report(
        current_gate,
        target_gate,
        policy=policy,
    )
