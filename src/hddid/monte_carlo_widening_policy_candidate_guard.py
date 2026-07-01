from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from .monte_carlo_widening_policy_floor_slack_probe import (
    _build_floor_slack_report_from_guard_probe,
    build_phase7_monte_carlo_widening_policy_floor_slack_probe_report,
)
from .monte_carlo_widening_policy_guard_probe import (
    _build_repo_side_guard_probe_report,
    build_phase7_monte_carlo_widening_policy_guard_probe_report,
)
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)
from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy
from .validation import (
    MonteCarloRuntimeProbeReport,
    run_phase7_monte_carlo_runtime_probe,
)


_REQUIRED_POLICY_KEYS = (
    "label",
    "max_total_runtime_seconds",
    "max_random_states",
    "stop_on_first_typed_invalidity",
    "min_nonparametric_coverage",
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANDIDATE_GUARD_CACHE_EPOCH_PATHS = (
    Path("Docs/automation/automation-state.yaml"),
    Path(".planning/STATE.md"),
    Path(
        ".planning/phases/07-final-hardening-and-verification-debt-closure/07-NEXT-MILESTONE-HANDOFF.md"
    ),
    Path("Docs/research/phase7_next_milestone_trigger_snapshot.md"),
    Path("Docs/research/phase7_monte_carlo_widening_policy_candidate_guard.md"),
    Path("Docs/research/phase7_monte_carlo_widening_policy_spec.md"),
    Path("Docs/research/phase7_monte_carlo_widening_policy_guard_probe.md"),
    Path("Docs/research/phase7_monte_carlo_widening_policy_acceptance_preview.md"),
    Path("hddid-py/src/hddid/next_milestone_trigger_snapshot.py"),
    Path("hddid-py/src/hddid/monte_carlo_widening_policy_candidate_guard.py"),
    Path("hddid-py/src/hddid/monte_carlo_widening_policy_spec.py"),
    Path("hddid-py/src/hddid/monte_carlo_widening_policy_guard_probe.py"),
    Path("hddid-py/src/hddid/monte_carlo_widening_trigger_gate.py"),
    Path("hddid-py/src/hddid/validation.py"),
)
_LAST_CANDIDATE_GUARD_CACHE_EPOCH: int | None = None


def _candidate_input_mode(
    policy: Phase7MonteCarloWideningPolicy | Mapping[str, object] | None,
) -> str:
    if policy is None:
        return "canonical-replay"
    if isinstance(policy, Phase7MonteCarloWideningPolicy):
        return "provided-policy-object"
    if isinstance(policy, Mapping):
        return "provided-policy-mapping"
    raise TypeError(
        "policy candidate must be a Phase7MonteCarloWideningPolicy or mapping"
    )


def _coerce_policy_candidate(
    policy: Phase7MonteCarloWideningPolicy | Mapping[str, object] | None,
) -> Phase7MonteCarloWideningPolicy:
    if policy is None:
        return build_phase7_canonical_monte_carlo_widening_policy()
    if isinstance(policy, Phase7MonteCarloWideningPolicy):
        return policy
    if not isinstance(policy, Mapping):
        raise TypeError(
            "policy candidate must be a Phase7MonteCarloWideningPolicy or mapping"
        )

    missing = tuple(key for key in _REQUIRED_POLICY_KEYS if key not in policy)
    if missing:
        raise ValueError(f"missing candidate policy keys: {', '.join(missing)}")

    unexpected = tuple(
        sorted(str(key) for key in policy if key not in _REQUIRED_POLICY_KEYS)
    )
    if unexpected:
        raise ValueError(f"unexpected candidate policy keys: {', '.join(unexpected)}")

    return Phase7MonteCarloWideningPolicy(
        label=policy["label"],
        max_total_runtime_seconds=policy["max_total_runtime_seconds"],
        max_random_states=policy["max_random_states"],
        stop_on_first_typed_invalidity=policy["stop_on_first_typed_invalidity"],
        min_nonparametric_coverage=policy["min_nonparametric_coverage"],
    )


def _coverage_floor_relation(
    policy: Phase7MonteCarloWideningPolicy,
    canonical_policy: Phase7MonteCarloWideningPolicy,
) -> str:
    candidate_floor = policy.min_nonparametric_coverage
    canonical_floor = canonical_policy.min_nonparametric_coverage
    if candidate_floor is None:
        return "weakens-canonical"
    if canonical_floor is None:
        return "matches-canonical"
    if candidate_floor < canonical_floor:
        return "weakens-canonical"
    if candidate_floor > canonical_floor:
        return "tightens-canonical"
    return "matches-canonical"


def _floor_ceiling_relation(
    candidate_floor: float | None,
    supported_floor_ceiling: float,
) -> str:
    if candidate_floor is None:
        return "missing-floor"
    gap = float(candidate_floor) - float(supported_floor_ceiling)
    if gap > 1e-12:
        return "above-supported-ceiling"
    if gap < -1e-12:
        return "below-supported-ceiling"
    return "matches-supported-ceiling"


def _candidate_rejection_reasons(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    policy: Phase7MonteCarloWideningPolicy,
    canonical_policy: Phase7MonteCarloWideningPolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if policy.label != canonical_policy.label:
        reasons.append("label drift weakens the canonical routing token")
    if policy.max_total_runtime_seconds > canonical_policy.max_total_runtime_seconds:
        reasons.append("runtime budget exceeds the canonical bounded widening cap")
    if runtime_probe.total_runtime_seconds is not None and (
        policy.max_total_runtime_seconds < runtime_probe.total_runtime_seconds
    ):
        reasons.append("runtime budget falls below the current bounded replay")
    if policy.max_random_states > canonical_policy.max_random_states:
        reasons.append("random-state budget exceeds the canonical bounded widening cap")
    if policy.max_random_states < len(runtime_probe.random_states):
        reasons.append("random-state budget falls below the current bounded replay")
    if not policy.stop_on_first_typed_invalidity:
        reasons.append("stop_on_first_typed_invalidity must remain True")
    if (
        policy.min_nonparametric_coverage is None
        or canonical_policy.min_nonparametric_coverage is None
        or policy.min_nonparametric_coverage
        < canonical_policy.min_nonparametric_coverage
    ):
        reasons.append("coverage floor falls below the canonical quality guard")
    return tuple(reasons)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCandidateGuardReport:
    stage_label: str
    accepted: bool
    candidate_input_mode: str
    candidate_status: str
    resulting_gate_status: str
    candidate_guard_replay_path: str
    policy_digest: tuple[str, ...]
    canonical_policy_digest: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    coverage_floor_relation: str
    supported_floor_ceiling: float
    floor_ceiling_relation: str
    floor_ceiling_gap: float
    runtime_budget_headroom_seconds: float
    random_state_headroom: int
    binding_guard: str
    canonical_candidate_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.accepted = bool(self.accepted)
        self.candidate_input_mode = str(self.candidate_input_mode).strip()
        self.candidate_status = str(self.candidate_status).strip()
        self.resulting_gate_status = str(self.resulting_gate_status).strip()
        self.candidate_guard_replay_path = str(
            self.candidate_guard_replay_path
        ).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.canonical_policy_digest = tuple(
            str(item).strip() for item in self.canonical_policy_digest
        )
        self.rejection_reasons = tuple(
            str(item).strip() for item in self.rejection_reasons
        )
        self.coverage_floor_relation = str(self.coverage_floor_relation).strip()
        self.supported_floor_ceiling = float(self.supported_floor_ceiling)
        self.floor_ceiling_relation = str(self.floor_ceiling_relation).strip()
        self.floor_ceiling_gap = float(self.floor_ceiling_gap)
        self.runtime_budget_headroom_seconds = float(
            self.runtime_budget_headroom_seconds
        )
        self.random_state_headroom = int(self.random_state_headroom)
        self.binding_guard = str(self.binding_guard).strip()
        self.canonical_candidate_digest = tuple(
            str(item).rstrip() for item in self.canonical_candidate_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "accepted": self.accepted,
            "candidate_input_mode": self.candidate_input_mode,
            "candidate_status": self.candidate_status,
            "resulting_gate_status": self.resulting_gate_status,
            "candidate_guard_replay_path": self.candidate_guard_replay_path,
            "policy_digest": list(self.policy_digest),
            "canonical_policy_digest": list(self.canonical_policy_digest),
            "rejection_reasons": list(self.rejection_reasons),
            "coverage_floor_relation": self.coverage_floor_relation,
            "supported_floor_ceiling": self.supported_floor_ceiling,
            "floor_ceiling_relation": self.floor_ceiling_relation,
            "floor_ceiling_gap": self.floor_ceiling_gap,
            "runtime_budget_headroom_seconds": self.runtime_budget_headroom_seconds,
            "random_state_headroom": self.random_state_headroom,
            "binding_guard": self.binding_guard,
            "canonical_candidate_digest": list(self.canonical_candidate_digest),
        }


def build_phase7_monte_carlo_widening_policy_candidate_guard_report(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    policy: Phase7MonteCarloWideningPolicy | Mapping[str, object] | None = None,
    candidate_guard_replay_path: str = "runtime-probe-direct-build",
) -> Phase7MonteCarloWideningPolicyCandidateGuardReport:
    canonical_policy = build_phase7_canonical_monte_carlo_widening_policy()
    candidate_input_mode = _candidate_input_mode(policy)
    resolved_policy = _coerce_policy_candidate(policy)
    try:
        guard_probe = build_phase7_monte_carlo_widening_policy_guard_probe_report(
            runtime_probe,
            policy=resolved_policy,
        )
    except ValueError as exc:
        if str(exc) != "runtime probe has no bounded widening slice summaries":
            raise
        guard_probe = _build_repo_side_guard_probe_report(
            runtime_probe,
            policy=resolved_policy,
        )
    try:
        floor_slack_probe = (
            build_phase7_monte_carlo_widening_policy_floor_slack_probe_report(
                runtime_probe,
                policy=resolved_policy,
            )
        )
    except ValueError as exc:
        if str(exc) != "runtime probe has no bounded widening slice summaries":
            raise
        floor_slack_probe = _build_floor_slack_report_from_guard_probe(
            guard_probe,
            policy=resolved_policy,
        )
    coverage_floor_relation = _coverage_floor_relation(
        resolved_policy,
        canonical_policy,
    )
    candidate_floor = resolved_policy.min_nonparametric_coverage
    supported_floor_ceiling = floor_slack_probe.supported_floor_ceiling
    floor_ceiling_relation = _floor_ceiling_relation(
        candidate_floor,
        supported_floor_ceiling,
    )
    floor_ceiling_gap = (
        0.0
        if candidate_floor is None
        else float(candidate_floor) - supported_floor_ceiling
    )
    rejection_reasons = _candidate_rejection_reasons(
        runtime_probe,
        policy=resolved_policy,
        canonical_policy=canonical_policy,
    )
    accepted = not rejection_reasons

    if accepted:
        if resolved_policy.to_digest() == canonical_policy.to_digest():
            candidate_status = "candidate-matches-canonical"
        else:
            candidate_status = "candidate-tightens-within-bounded-slice"
        resulting_gate_status = "trigger2-partial-widening-only"
        binding_guard = guard_probe.binding_guard
    else:
        candidate_status = "candidate-rejected"
        resulting_gate_status = "trigger2-policy-missing"
        binding_guard = "candidate-policy-rejected"

    if rejection_reasons:
        rejection_line = "- rejection reasons: " + ", ".join(
            f"`{reason}`" for reason in rejection_reasons
        )
    else:
        rejection_line = (
            "- rejection reasons: `none`; candidate preserves the bounded widening "
            "intake invariants"
        )

    canonical_candidate_digest = (
        "- candidate input mode: "
        f"`{candidate_input_mode}`; candidate policy status: "
        f"`{candidate_status}`; resulting gate status: `{resulting_gate_status}`",
        "- candidate digest: "
        + ", ".join(f"`{item}`" for item in resolved_policy.to_digest()),
        "- canonical intake invariants: "
        f"`label={canonical_policy.label}`; "
        f"`max_total_runtime_seconds<={canonical_policy.max_total_runtime_seconds}`; "
        f"`max_random_states<={canonical_policy.max_random_states}`; "
        f"`stop_on_first_typed_invalidity={canonical_policy.stop_on_first_typed_invalidity}`; "
        f"`min_nonparametric_coverage>={canonical_policy.min_nonparametric_coverage}`",
        "- observed bounded replay fit: runtime headroom "
        f"`{guard_probe.runtime_budget_headroom_seconds:.3f}`; random-state headroom "
        f"`{guard_probe.random_state_headroom}`; coverage relation "
        f"`{coverage_floor_relation}`; supported floor ceiling "
        f"`{supported_floor_ceiling:.3f}`; candidate floor relation "
        f"`{floor_ceiling_relation}`; candidate floor gap "
        f"`{floor_ceiling_gap:+.3f}`; post-intake binding guard `{binding_guard}`",
        rejection_line,
    )

    return Phase7MonteCarloWideningPolicyCandidateGuardReport(
        stage_label="phase7-monte-carlo-widening-policy-candidate-guard",
        accepted=accepted,
        candidate_input_mode=candidate_input_mode,
        candidate_status=candidate_status,
        resulting_gate_status=resulting_gate_status,
        candidate_guard_replay_path=candidate_guard_replay_path,
        policy_digest=resolved_policy.to_digest(),
        canonical_policy_digest=canonical_policy.to_digest(),
        rejection_reasons=rejection_reasons,
        coverage_floor_relation=coverage_floor_relation,
        supported_floor_ceiling=supported_floor_ceiling,
        floor_ceiling_relation=floor_ceiling_relation,
        floor_ceiling_gap=floor_ceiling_gap,
        runtime_budget_headroom_seconds=guard_probe.runtime_budget_headroom_seconds,
        random_state_headroom=guard_probe.random_state_headroom,
        binding_guard=binding_guard,
        canonical_candidate_digest=canonical_candidate_digest,
    )


def _phase7_monte_carlo_widening_policy_candidate_guard_cache_epoch() -> int:
    return max(
        (_REPO_ROOT / path).stat().st_mtime_ns
        for path in _CANDIDATE_GUARD_CACHE_EPOCH_PATHS
        if (_REPO_ROOT / path).exists()
    )


@lru_cache(maxsize=1)
def _run_phase7_monte_carlo_widening_policy_candidate_guard_cached(
    cache_epoch: int,
) -> (
    Phase7MonteCarloWideningPolicyCandidateGuardReport
):
    return build_phase7_monte_carlo_widening_policy_candidate_guard_report(
        run_phase7_monte_carlo_runtime_probe(),
        candidate_guard_replay_path="canonical-no-policy-lru-cache-miss",
    )


def run_phase7_monte_carlo_widening_policy_candidate_guard(
    policy: Phase7MonteCarloWideningPolicy | Mapping[str, object] | None = None,
) -> Phase7MonteCarloWideningPolicyCandidateGuardReport:
    if policy is None:
        global _LAST_CANDIDATE_GUARD_CACHE_EPOCH
        cache_epoch = _phase7_monte_carlo_widening_policy_candidate_guard_cache_epoch()
        cache_info_before = (
            _run_phase7_monte_carlo_widening_policy_candidate_guard_cached.cache_info()
        )
        report = _run_phase7_monte_carlo_widening_policy_candidate_guard_cached(
            cache_epoch
        )
        cache_info_after = (
            _run_phase7_monte_carlo_widening_policy_candidate_guard_cached.cache_info()
        )
        replay_path = "canonical-no-policy-lru-cache-miss"
        if cache_info_after.hits > cache_info_before.hits:
            replay_path = "canonical-no-policy-lru-cache-hit"
        elif (
            cache_info_before.currsize > 0
            and _LAST_CANDIDATE_GUARD_CACHE_EPOCH is not None
            and _LAST_CANDIDATE_GUARD_CACHE_EPOCH != cache_epoch
        ):
            replay_path = "canonical-no-policy-lru-cache-stale-rebuild"
        _LAST_CANDIDATE_GUARD_CACHE_EPOCH = cache_epoch
        return replace(report, candidate_guard_replay_path=replay_path)
    return build_phase7_monte_carlo_widening_policy_candidate_guard_report(
        run_phase7_monte_carlo_runtime_probe(),
        policy=policy,
        candidate_guard_replay_path="explicit-policy-live-rebuild",
    )
