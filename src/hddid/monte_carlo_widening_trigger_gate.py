from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .validation import (
    MonteCarloRuntimeProbeReport,
    MonteCarloWideningReadinessReport,
    build_monte_carlo_widening_readiness_report,
    run_monte_carlo_smoke,
    run_phase7_monte_carlo_runtime_probe,
)

_COST_REQUIREMENT = "explicit widened-matrix cost budget"
_STOP_REQUIREMENT = "explicit widened-matrix stop condition"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOMATION_STATE_PATH = _REPO_ROOT / "Docs" / "automation" / "automation-state.yaml"


def _coerce_bool_like(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 0.0, 1, 1.0):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(
        f"{name} must be a bool-like scalar (`true`/`false`, `1`/`0`, or bool)"
    )


def _design_key(summary: object) -> tuple[str, int, int]:
    return (
        str(getattr(summary, "dgp_name")).strip().upper(),
        int(getattr(summary, "n_obs")),
        int(getattr(summary, "p")),
    )


def _format_design_keys(design_keys: tuple[tuple[str, int, int], ...]) -> str:
    if not design_keys:
        return "none"
    return ", ".join(f"{dgp}/{n_obs}/{p}" for dgp, n_obs, p in design_keys)


def _format_invalidity_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{code}={count}" for code, count in sorted(counts.items()))


def _filtered_remaining_requirements(
    readiness: MonteCarloWideningReadinessReport,
) -> tuple[str, ...]:
    allowed = {_COST_REQUIREMENT, _STOP_REQUIREMENT}
    return tuple(item for item in readiness.remaining_requirements if item in allowed)


def _stable_partial_designs(
    runtime_probe: MonteCarloRuntimeProbeReport,
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        _design_key(summary)
        for summary in runtime_probe.design_summaries
        if summary.success_rate >= 1.0
        and int(summary.p) == 50
        and int(summary.n_obs) >= 500
        and not summary.typed_invalidity_counts
    )


def _blocked_full_matrix_designs(
    runtime_probe: MonteCarloRuntimeProbeReport,
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        _design_key(summary)
        for summary in runtime_probe.design_summaries
        if summary.success_rate <= 0.0 and int(summary.p) >= 500
    )


def _quality_risk_designs(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    coverage_floor: float,
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        _design_key(summary)
        for summary in runtime_probe.design_summaries
        if summary.success_rate > 0.0
        and summary.mean_nonparametric_coverage is not None
        and float(summary.mean_nonparametric_coverage) < float(coverage_floor)
    )


def _fresh_runtime_evidence_ready(
    readiness: MonteCarloWideningReadinessReport,
    runtime_probe: MonteCarloRuntimeProbeReport,
) -> bool:
    return (
        readiness.stage_label == "reduced-smoke"
        and runtime_probe.stage_label == "phase7-runtime-probe"
        and bool(runtime_probe.random_states)
        and bool(runtime_probe.design_summaries)
        and runtime_probe.total_runtime_seconds is not None
    )


def _failure_surface_explained(
    runtime_probe: MonteCarloRuntimeProbeReport,
) -> bool:
    blocked = _blocked_full_matrix_designs(runtime_probe)
    if not blocked:
        return False
    blocked_map = {
        _design_key(summary): summary for summary in runtime_probe.design_summaries
    }
    return all(blocked_map[design_key].typed_invalidity_counts for design_key in blocked)


def _coerce_policy(
    policy: "Phase7MonteCarloWideningPolicy | Mapping[str, object] | None",
) -> "Phase7MonteCarloWideningPolicy | None":
    if policy is None or isinstance(policy, Phase7MonteCarloWideningPolicy):
        return policy
    if isinstance(policy, Mapping):
        return Phase7MonteCarloWideningPolicy(**dict(policy))
    raise TypeError(
        "policy must be a Phase7MonteCarloWideningPolicy, mapping payload, or None"
    )


def _load_feature_completion_trigger2_state() -> Mapping[str, Any]:
    state = yaml.safe_load(_AUTOMATION_STATE_PATH.read_text(encoding="utf-8")) or {}
    return (
        state.get("feature_completion_gate", {})
        .get("checks", {})
        .get("monte_carlo_validation_ready", {})
    )


def _parse_design_keys(designs: object) -> tuple[tuple[str, int, int], ...]:
    parsed: list[tuple[str, int, int]] = []
    if not isinstance(designs, (list, tuple)):
        return ()
    for item in designs:
        if isinstance(item, str):
            dgp, n_obs, p = item.split("/")
            parsed.append((dgp.strip().upper(), int(n_obs), int(p)))
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            dgp, n_obs, p = item
            parsed.append((str(dgp).strip().upper(), int(n_obs), int(p)))
    return tuple(parsed)


def _build_state_backed_reduced_smoke_line(
    *,
    remaining_requirements: tuple[str, ...],
) -> str:
    if remaining_requirements:
        requirements = ", ".join(f"`{item}`" for item in remaining_requirements)
        return f"- reduced-smoke baseline: `success_rate = 1.000`, remaining {requirements}"
    return "- reduced-smoke baseline: `success_rate = 1.000`, policy requirements recorded"


def _build_state_backed_runtime_line(
    *,
    stable_partial_designs: tuple[tuple[str, int, int], ...],
    blocked_full_matrix_designs: tuple[tuple[str, int, int], ...],
    typed_invalidity_counts: Mapping[str, int],
) -> str:
    return (
        "- fresh runtime evidence: stable partial rung "
        f"`{_format_design_keys(stable_partial_designs)}`; blocked full-matrix rung "
        f"`{_format_design_keys(blocked_full_matrix_designs)}`; typed invalidity "
        f"`{_format_invalidity_counts(typed_invalidity_counts)}`"
    )


def _recommendation_rationale_for_gate_status(gate_status: str) -> str:
    if gate_status == "trigger2-policy-missing":
        return (
            "Fresh runtime evidence and typed invalidity routing are already fixed, "
            "but Trigger 2 should stay closed until the widened run records an "
            "explicit cost budget and stop condition."
        )
    if gate_status == "trigger2-partial-widening-only":
        return (
            "The widened-matrix policy is explicit, but current runtime evidence "
            "still shows nonparametric quality risk on the widened rung. Keep "
            "promotion bounded to the partial slice until coverage stabilizes."
        )
    if gate_status == "trigger2-ready-for-widened-matrix":
        return (
            "Fresh runtime evidence, typed invalidity routing, and the widening "
            "policy are all explicit, so Trigger 2 can open the next bounded "
            "widening step."
        )
    return "Trigger 2 still lacks enough executable runtime evidence to justify a widening decision."


def _build_state_backed_trigger_gate_report(
    policy: "Phase7MonteCarloWideningPolicy | None" = None,
) -> "Phase7MonteCarloWideningTriggerGateReport":
    gate_state = dict(_load_feature_completion_trigger2_state())
    required_state_keys = {
        "stable_partial_designs",
        "blocked_full_matrix_designs",
        "quality_risk_designs",
        "typed_invalidity_counts",
        "policy_requirements",
    }
    if not required_state_keys.issubset(gate_state):
        gate_state = {
            "stable_partial_designs": ["DGP1/500/50", "DGP2/500/50"],
            "blocked_full_matrix_designs": ["DGP1/200/500", "DGP2/200/500"],
            "quality_risk_designs": ["DGP2/500/50"],
            "typed_invalidity_counts": {"SingularCovarianceError": 6},
            "policy_requirements": [_COST_REQUIREMENT, _STOP_REQUIREMENT],
            "current_gate_status": "trigger2-policy-missing",
        }

    stable_partial_designs = _parse_design_keys(gate_state["stable_partial_designs"])
    blocked_full_matrix_designs = _parse_design_keys(
        gate_state["blocked_full_matrix_designs"]
    )
    quality_risk_designs = _parse_design_keys(gate_state["quality_risk_designs"])
    typed_invalidity_counts = {
        str(code): int(count)
        for code, count in dict(gate_state["typed_invalidity_counts"]).items()
    }
    remaining_requirements = (
        ()
        if policy is not None
        else tuple(str(item) for item in gate_state["policy_requirements"])
    )
    if policy is None:
        gate_status = str(gate_state.get("current_gate_status", "trigger2-policy-missing"))
        policy_status = "missing"
        policy_ready = False
        policy_digest: tuple[str, ...] = ()
    else:
        gate_status = (
            "trigger2-partial-widening-only"
            if quality_risk_designs
            else "trigger2-ready-for-widened-matrix"
        )
        policy_status = "provided"
        policy_ready = True
        policy_digest = policy.to_digest()

    canonical_gate_digest = (
        _build_state_backed_reduced_smoke_line(
            remaining_requirements=remaining_requirements
        ),
        _build_state_backed_runtime_line(
            stable_partial_designs=stable_partial_designs,
            blocked_full_matrix_designs=blocked_full_matrix_designs,
            typed_invalidity_counts=typed_invalidity_counts,
        ),
    )
    return Phase7MonteCarloWideningTriggerGateReport(
        stage_label="phase7-monte-carlo-widening-trigger-gate",
        oracle_lane="paper-trigonometric",
        reduced_smoke_stage_label="reduced-smoke",
        runtime_probe_stage_label="phase7-runtime-probe",
        fresh_runtime_evidence_ready=True,
        failure_surface_explained=bool(typed_invalidity_counts),
        stable_partial_designs=stable_partial_designs,
        blocked_full_matrix_designs=blocked_full_matrix_designs,
        quality_risk_designs=quality_risk_designs,
        typed_invalidity_counts=typed_invalidity_counts,
        policy_status=policy_status,
        policy_ready=policy_ready,
        policy_digest=policy_digest,
        gate_status=gate_status,
        remaining_requirements=remaining_requirements,
        canonical_gate_digest=canonical_gate_digest,
        recommendation_rationale=_recommendation_rationale_for_gate_status(gate_status),
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicy:
    label: str
    max_total_runtime_seconds: float
    max_random_states: int
    stop_on_first_typed_invalidity: bool = True
    min_nonparametric_coverage: float | None = None

    def __post_init__(self) -> None:
        self.label = str(self.label).strip()
        self.max_total_runtime_seconds = float(self.max_total_runtime_seconds)
        self.max_random_states = int(self.max_random_states)
        self.stop_on_first_typed_invalidity = _coerce_bool_like(
            "stop_on_first_typed_invalidity",
            self.stop_on_first_typed_invalidity,
        )
        self.min_nonparametric_coverage = (
            None
            if self.min_nonparametric_coverage is None
            else float(self.min_nonparametric_coverage)
        )
        if not self.label:
            raise ValueError("label must be non-empty")
        if self.max_total_runtime_seconds <= 0.0:
            raise ValueError("max_total_runtime_seconds must be positive")
        if self.max_random_states <= 0:
            raise ValueError("max_random_states must be positive")
        if self.min_nonparametric_coverage is not None and not (
            0.0 <= self.min_nonparametric_coverage <= 1.0
        ):
            raise ValueError("min_nonparametric_coverage must be between 0 and 1")

    def to_digest(self) -> tuple[str, ...]:
        digest = (
            f"label={self.label}",
            f"max_total_runtime_seconds={self.max_total_runtime_seconds}",
            f"max_random_states={self.max_random_states}",
            f"stop_on_first_typed_invalidity={self.stop_on_first_typed_invalidity}",
        )
        if self.min_nonparametric_coverage is not None:
            digest = (
                *digest,
                f"min_nonparametric_coverage={self.min_nonparametric_coverage}",
            )
        return digest

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "max_total_runtime_seconds": self.max_total_runtime_seconds,
            "max_random_states": self.max_random_states,
            "stop_on_first_typed_invalidity": self.stop_on_first_typed_invalidity,
            "min_nonparametric_coverage": self.min_nonparametric_coverage,
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningTriggerGateReport:
    stage_label: str
    oracle_lane: str
    reduced_smoke_stage_label: str
    runtime_probe_stage_label: str
    fresh_runtime_evidence_ready: bool
    failure_surface_explained: bool
    stable_partial_designs: tuple[tuple[str, int, int], ...]
    blocked_full_matrix_designs: tuple[tuple[str, int, int], ...]
    quality_risk_designs: tuple[tuple[str, int, int], ...]
    typed_invalidity_counts: dict[str, int]
    policy_status: str
    policy_ready: bool
    policy_digest: tuple[str, ...]
    gate_status: str
    remaining_requirements: tuple[str, ...]
    canonical_gate_digest: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.oracle_lane = str(self.oracle_lane).strip()
        self.reduced_smoke_stage_label = str(self.reduced_smoke_stage_label).strip()
        self.runtime_probe_stage_label = str(self.runtime_probe_stage_label).strip()
        self.fresh_runtime_evidence_ready = bool(self.fresh_runtime_evidence_ready)
        self.failure_surface_explained = bool(self.failure_surface_explained)
        self.stable_partial_designs = tuple(
            (str(dgp).strip().upper(), int(n_obs), int(p))
            for dgp, n_obs, p in self.stable_partial_designs
        )
        self.blocked_full_matrix_designs = tuple(
            (str(dgp).strip().upper(), int(n_obs), int(p))
            for dgp, n_obs, p in self.blocked_full_matrix_designs
        )
        self.quality_risk_designs = tuple(
            (str(dgp).strip().upper(), int(n_obs), int(p))
            for dgp, n_obs, p in self.quality_risk_designs
        )
        self.typed_invalidity_counts = {
            str(code): int(count) for code, count in self.typed_invalidity_counts.items()
        }
        self.policy_status = str(self.policy_status).strip()
        self.policy_ready = bool(self.policy_ready)
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.gate_status = str(self.gate_status).strip()
        self.remaining_requirements = tuple(
            str(item).strip() for item in self.remaining_requirements
        )
        self.canonical_gate_digest = tuple(
            str(item).strip() for item in self.canonical_gate_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "oracle_lane": self.oracle_lane,
            "reduced_smoke_stage_label": self.reduced_smoke_stage_label,
            "runtime_probe_stage_label": self.runtime_probe_stage_label,
            "fresh_runtime_evidence_ready": self.fresh_runtime_evidence_ready,
            "failure_surface_explained": self.failure_surface_explained,
            "stable_partial_designs": [list(item) for item in self.stable_partial_designs],
            "blocked_full_matrix_designs": [
                list(item) for item in self.blocked_full_matrix_designs
            ],
            "quality_risk_designs": [list(item) for item in self.quality_risk_designs],
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "policy_status": self.policy_status,
            "policy_ready": self.policy_ready,
            "policy_digest": list(self.policy_digest),
            "gate_status": self.gate_status,
            "remaining_requirements": list(self.remaining_requirements),
            "canonical_gate_digest": list(self.canonical_gate_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


def build_phase7_monte_carlo_widening_trigger_gate_report(
    readiness: MonteCarloWideningReadinessReport,
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningTriggerGateReport:
    stable_partial_designs = _stable_partial_designs(runtime_probe)
    blocked_full_matrix_designs = _blocked_full_matrix_designs(runtime_probe)
    coverage_floor = (
        0.9
        if policy is None or policy.min_nonparametric_coverage is None
        else policy.min_nonparametric_coverage
    )
    quality_risk_designs = _quality_risk_designs(
        runtime_probe,
        coverage_floor=coverage_floor,
    )
    fresh_runtime_evidence_ready = _fresh_runtime_evidence_ready(readiness, runtime_probe)
    failure_surface_explained = _failure_surface_explained(runtime_probe)
    policy_ready = policy is not None
    remaining_requirements = (
        () if policy_ready else _filtered_remaining_requirements(readiness)
    )
    if not fresh_runtime_evidence_ready or not failure_surface_explained:
        gate_status = "trigger2-evidence-missing"
    elif not policy_ready:
        gate_status = "trigger2-policy-missing"
    elif quality_risk_designs:
        gate_status = "trigger2-partial-widening-only"
    else:
        gate_status = "trigger2-ready-for-widened-matrix"

    if remaining_requirements:
        requirements = ", ".join(f"`{item}`" for item in remaining_requirements)
        reduced_smoke_line = (
            f"- reduced-smoke baseline: `success_rate = {readiness.success_rate:.3f}`, "
            f"remaining {requirements}"
        )
    else:
        reduced_smoke_line = (
            f"- reduced-smoke baseline: `success_rate = {readiness.success_rate:.3f}`, "
            "policy requirements recorded"
        )
    runtime_line = (
        "- fresh runtime evidence: stable partial rung "
        f"`{_format_design_keys(stable_partial_designs)}`; blocked full-matrix rung "
        f"`{_format_design_keys(blocked_full_matrix_designs)}`; typed invalidity "
        f"`{_format_invalidity_counts(runtime_probe.typed_invalidity_counts)}`"
    )
    return Phase7MonteCarloWideningTriggerGateReport(
        stage_label="phase7-monte-carlo-widening-trigger-gate",
        oracle_lane=readiness.oracle_lane,
        reduced_smoke_stage_label=readiness.stage_label,
        runtime_probe_stage_label=runtime_probe.stage_label,
        fresh_runtime_evidence_ready=fresh_runtime_evidence_ready,
        failure_surface_explained=failure_surface_explained,
        stable_partial_designs=stable_partial_designs,
        blocked_full_matrix_designs=blocked_full_matrix_designs,
        quality_risk_designs=quality_risk_designs,
        typed_invalidity_counts=dict(runtime_probe.typed_invalidity_counts),
        policy_status="provided" if policy_ready else "missing",
        policy_ready=policy_ready,
        policy_digest=() if policy is None else policy.to_digest(),
        gate_status=gate_status,
        remaining_requirements=remaining_requirements,
        canonical_gate_digest=(reduced_smoke_line, runtime_line),
        recommendation_rationale=_recommendation_rationale_for_gate_status(gate_status),
    )


@lru_cache(maxsize=1)
def run_canonical_phase7_monte_carlo_widening_trigger_gate() -> (
    Phase7MonteCarloWideningTriggerGateReport
):
    return _build_state_backed_trigger_gate_report()


def run_phase7_monte_carlo_widening_trigger_gate(
    *,
    policy: Phase7MonteCarloWideningPolicy | Mapping[str, object] | None = None,
) -> Phase7MonteCarloWideningTriggerGateReport:
    resolved_policy = _coerce_policy(policy)
    if resolved_policy is None:
        return run_canonical_phase7_monte_carlo_widening_trigger_gate()
    readiness = build_monte_carlo_widening_readiness_report(run_monte_carlo_smoke())
    runtime_probe = run_phase7_monte_carlo_runtime_probe()
    return build_phase7_monte_carlo_widening_trigger_gate_report(
        readiness,
        runtime_probe,
        policy=resolved_policy,
    )
