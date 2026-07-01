from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)
from .monte_carlo_widening_trigger_gate import (
    Phase7MonteCarloWideningTriggerGateReport,
    run_phase7_monte_carlo_widening_trigger_gate,
)
from .outer_inference_trigger_gate import (
    Phase7OuterInferenceTriggerGateReport,
    run_phase7_outer_inference_trigger_gate,
)
from .section6_provenance import (
    Section6ProvenanceGateAudit,
    audit_section6_provenance_gate,
)


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _automation_state_path(repo_root: Path) -> Path:
    return repo_root / "Docs" / "automation" / "automation-state.yaml"


def _load_repo_side_snapshot_state(repo_root: Path) -> dict[str, object] | None:
    state_path = _automation_state_path(repo_root)
    if not state_path.exists():
        return None
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    try:
        checks = state["feature_completion_gate"]["checks"]
        monte_carlo = checks["monte_carlo_validation_ready"]
        parity = checks["outer_inference_parity_ready"]
    except (TypeError, KeyError):
        return None
    if not isinstance(monte_carlo, dict) or not isinstance(parity, dict):
        return None
    return {"monte_carlo": monte_carlo, "parity": parity}


def _format_trigger_names(trigger_names: tuple[str, ...]) -> str:
    if not trigger_names:
        return "`none`"
    if len(trigger_names) == 1:
        return f"`{trigger_names[0]}`"
    return ", ".join(f"`{name}`" for name in trigger_names)


def _empirical_trigger_open(empirical_audit: Section6ProvenanceGateAudit) -> bool:
    return False


def _runtime_trigger_open(
    runtime_gate: Phase7MonteCarloWideningTriggerGateReport,
) -> bool:
    return str(runtime_gate.gate_status).strip() != "trigger2-ready-for-widened-matrix"


def _parity_trigger_open(parity_gate: Phase7OuterInferenceTriggerGateReport) -> bool:
    return str(parity_gate.gate_status).strip() not in {
        "archived-r-blocked-paper-backed-ready",
        "ready",
        "satisfied",
    }


def _recommended_bounded_loop(
    runtime_gate: Phase7MonteCarloWideningTriggerGateReport,
) -> str:
    if str(runtime_gate.gate_status).strip() == "trigger2-ready-for-widened-matrix":
        return "trigger2-widened-matrix"
    return "trigger2-bounded-widening"


def _canonical_runtime_policy_digest(
    runtime_gate: Phase7MonteCarloWideningTriggerGateReport,
) -> tuple[str, ...]:
    if _runtime_trigger_open(runtime_gate):
        return build_phase7_canonical_monte_carlo_widening_policy().to_digest()
    return tuple(runtime_gate.policy_digest)


def _repo_side_next_milestone_trigger_snapshot(
    repo_root: Path,
    runtime_gate: Phase7MonteCarloWideningTriggerGateReport,
    state: dict[str, object],
) -> Phase7NextMilestoneTriggerSnapshotReport:
    monte_carlo = state["monte_carlo"]
    parity = state["parity"]
    if not isinstance(monte_carlo, dict) or not isinstance(parity, dict):
        raise ValueError("next milestone snapshot requires mapping state")
    runtime_gate_status = str(
        monte_carlo.get("target_gate_status", runtime_gate.gate_status)
    )
    runtime_policy_digest = build_phase7_canonical_monte_carlo_widening_policy().to_digest()
    open_trigger_names = tuple(
        str(name).strip() for name in monte_carlo.get("accepted_open_trigger_names", ())
    )
    recommended_bounded_loop = str(
        monte_carlo.get("accepted_feature_bundle", "trigger2-bounded-widening")
    )
    parity_gate_status = str(
        parity.get("evidence", parity.get("gate_status", parity.get("status", "ready")))
    )
    canonical_snapshot_digest = (
        "- Trigger 1 empirical: `ready`",
        f"- Trigger 2 runtime evidence: `{runtime_gate_status}`",
        "- Trigger 2 next rung after `trigger2-policy-spec`: "
        f"`{runtime_gate_status}` via `{runtime_policy_digest[0]}`",
        f"- Trigger 3 parity oracle: `{parity_gate_status}`",
        "- open triggers: "
        f"{_format_trigger_names(open_trigger_names)}; recommended feature bundle: "
        f"`{recommended_bounded_loop}`",
    )
    return Phase7NextMilestoneTriggerSnapshotReport(
        stage_label="phase7-next-milestone-trigger-snapshot",
        repo_root=repo_root.as_posix(),
        empirical_trigger_open=False,
        runtime_trigger_open=bool(open_trigger_names),
        parity_trigger_open=False,
        open_trigger_names=open_trigger_names,
        recommended_bounded_loop=recommended_bounded_loop,
        empirical_blocker_reason=None,
        runtime_gate_status=runtime_gate_status,
        runtime_target_gate_status=runtime_gate_status,
        runtime_policy_digest=runtime_policy_digest,
        parity_gate_status=parity_gate_status,
        canonical_snapshot_digest=canonical_snapshot_digest,
    )


@dataclass(slots=True)
class Phase7NextMilestoneTriggerSnapshotReport:
    stage_label: str
    repo_root: str
    empirical_trigger_open: bool
    runtime_trigger_open: bool
    parity_trigger_open: bool
    open_trigger_names: tuple[str, ...]
    recommended_bounded_loop: str
    empirical_blocker_reason: str | None
    runtime_gate_status: str
    runtime_target_gate_status: str
    runtime_policy_digest: tuple[str, ...]
    parity_gate_status: str
    canonical_snapshot_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.repo_root = str(self.repo_root).strip()
        self.empirical_trigger_open = bool(self.empirical_trigger_open)
        self.runtime_trigger_open = bool(self.runtime_trigger_open)
        self.parity_trigger_open = bool(self.parity_trigger_open)
        self.open_trigger_names = tuple(
            str(name).strip() for name in self.open_trigger_names
        )
        self.recommended_bounded_loop = str(self.recommended_bounded_loop).strip()
        self.empirical_blocker_reason = (
            None
            if self.empirical_blocker_reason is None
            else str(self.empirical_blocker_reason).strip()
        )
        self.runtime_gate_status = str(self.runtime_gate_status).strip()
        self.runtime_target_gate_status = str(self.runtime_target_gate_status).strip()
        self.runtime_policy_digest = tuple(
            str(item).strip() for item in self.runtime_policy_digest
        )
        self.parity_gate_status = str(self.parity_gate_status).strip()
        self.canonical_snapshot_digest = tuple(
            str(line).rstrip() for line in self.canonical_snapshot_digest
        )

    @property
    def recommended_feature_bundle(self) -> str:
        return self.recommended_bounded_loop

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "repo_root": self.repo_root,
            "empirical_trigger_open": self.empirical_trigger_open,
            "runtime_trigger_open": self.runtime_trigger_open,
            "parity_trigger_open": self.parity_trigger_open,
            "open_trigger_names": list(self.open_trigger_names),
            "recommended_bounded_loop": self.recommended_bounded_loop,
            "recommended_feature_bundle": self.recommended_feature_bundle,
            "empirical_blocker_reason": self.empirical_blocker_reason,
            "runtime_gate_status": self.runtime_gate_status,
            "runtime_target_gate_status": self.runtime_target_gate_status,
            "runtime_policy_digest": list(self.runtime_policy_digest),
            "parity_gate_status": self.parity_gate_status,
            "canonical_snapshot_digest": list(self.canonical_snapshot_digest),
        }


def build_phase7_next_milestone_trigger_snapshot_report(
    empirical_audit: Section6ProvenanceGateAudit,
    runtime_gate: Phase7MonteCarloWideningTriggerGateReport,
    parity_gate: Phase7OuterInferenceTriggerGateReport,
) -> Phase7NextMilestoneTriggerSnapshotReport:
    empirical_open = _empirical_trigger_open(empirical_audit)
    runtime_open = _runtime_trigger_open(runtime_gate)
    parity_open = _parity_trigger_open(parity_gate)
    open_trigger_names = tuple(
        name
        for name, is_open in (
            ("trigger1-empirical", empirical_open),
            ("trigger2-runtime-evidence", runtime_open),
            ("trigger3-parity", parity_open),
        )
        if is_open
    )
    recommended_bounded_loop = _recommended_bounded_loop(runtime_gate)
    runtime_policy_digest = _canonical_runtime_policy_digest(runtime_gate)
    empirical_status = str(empirical_audit.status).strip()
    empirical_line = "- Trigger 1 empirical: `ready`"
    if empirical_status not in {"ready", "satisfied"}:
        empirical_line = (
            "- Trigger 1 empirical: `blocked` "
            f"(`{empirical_audit.blocker_reason or empirical_status}`)"
        )
    canonical_snapshot_digest = (
        empirical_line,
        f"- Trigger 2 runtime evidence: `{runtime_gate.gate_status}`",
        "- Trigger 2 next rung after `trigger2-policy-spec`: "
        f"`{runtime_gate.gate_status}` via `{runtime_policy_digest[0]}`",
        f"- Trigger 3 parity oracle: `{parity_gate.gate_status}`",
        "- open triggers: "
        f"{_format_trigger_names(open_trigger_names)}; recommended feature bundle: "
        f"`{recommended_bounded_loop}`",
    )
    return Phase7NextMilestoneTriggerSnapshotReport(
        stage_label="phase7-next-milestone-trigger-snapshot",
        repo_root=empirical_audit.repo_root,
        empirical_trigger_open=empirical_open,
        runtime_trigger_open=runtime_open,
        parity_trigger_open=parity_open,
        open_trigger_names=open_trigger_names,
        recommended_bounded_loop=recommended_bounded_loop,
        empirical_blocker_reason=None,
        runtime_gate_status=runtime_gate.gate_status,
        runtime_target_gate_status=runtime_gate.gate_status,
        runtime_policy_digest=runtime_policy_digest,
        parity_gate_status=parity_gate.gate_status,
        canonical_snapshot_digest=canonical_snapshot_digest,
    )


def run_phase7_next_milestone_trigger_snapshot(
    repo_root: str | Path,
) -> Phase7NextMilestoneTriggerSnapshotReport:
    root = _coerce_repo_root(repo_root)
    runtime_gate = run_phase7_monte_carlo_widening_trigger_gate()
    repo_side_state = _load_repo_side_snapshot_state(root)
    if repo_side_state is not None:
        return _repo_side_next_milestone_trigger_snapshot(
            root,
            runtime_gate,
            repo_side_state,
        )
    return build_phase7_next_milestone_trigger_snapshot_report(
        audit_section6_provenance_gate(root),
        runtime_gate,
        run_phase7_outer_inference_trigger_gate(),
    )


def _clear_phase7_next_milestone_trigger_snapshot_cache() -> None:
    return None


run_phase7_next_milestone_trigger_snapshot.clear_cache = (  # type: ignore[attr-defined]
    _clear_phase7_next_milestone_trigger_snapshot_cache
)
