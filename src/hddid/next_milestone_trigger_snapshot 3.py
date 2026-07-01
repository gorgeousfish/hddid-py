from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
    build_phase7_monte_carlo_widening_policy_transition,
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


def _runtime_trigger_open(
    runtime_gate: Phase7MonteCarloWideningTriggerGateReport,
) -> bool:
    return runtime_gate.gate_status in (
        "trigger2-partial-widening-only",
        "trigger2-ready-for-widened-matrix",
    )


def _parity_trigger_open(
    parity_gate: Phase7OuterInferenceTriggerGateReport,
) -> bool:
    return parity_gate.gate_status in (
        "trigger3-paper-backed-parity-ready",
        "trigger3-ready-for-parity",
    )


def _build_repo_side_target_gate(
    current_gate: Phase7MonteCarloWideningTriggerGateReport,
) -> Phase7MonteCarloWideningTriggerGateReport:
    canonical_policy = build_phase7_canonical_monte_carlo_widening_policy()
    return replace(
        current_gate,
        policy_status="provided",
        policy_ready=True,
        policy_digest=canonical_policy.to_digest(),
        gate_status="trigger2-partial-widening-only",
        remaining_requirements=(),
    )


def _runtime_target_gate_status(
    transition_trigger_status: str,
) -> str:
    return transition_trigger_status


def _runtime_policy_digest(
    runtime_gate: Phase7MonteCarloWideningTriggerGateReport,
    transition_policy_digest: tuple[str, ...],
) -> tuple[str, ...]:
    canonical_policy_digest = (
        build_phase7_canonical_monte_carlo_widening_policy().to_digest()
    )
    if transition_policy_digest:
        return canonical_policy_digest
    if runtime_gate.policy_ready:
        return canonical_policy_digest
    return ()


def _open_trigger_names(
    *,
    empirical_trigger_open: bool,
    runtime_trigger_open: bool,
    parity_trigger_open: bool,
) -> tuple[str, ...]:
    names: list[str] = []
    if empirical_trigger_open:
        names.append("trigger1-empirical-assets")
    if runtime_trigger_open:
        names.append("trigger2-runtime-evidence")
    if parity_trigger_open:
        names.append("trigger3-parity-oracle")
    return tuple(names)


def _recommended_bounded_loop(
    empirical_audit: Section6ProvenanceGateAudit,
    runtime_gate: Phase7MonteCarloWideningTriggerGateReport,
    parity_gate: Phase7OuterInferenceTriggerGateReport,
    *,
    empirical_trigger_open: bool,
    runtime_trigger_open: bool,
    parity_trigger_open: bool,
) -> str:
    if empirical_trigger_open:
        return "trigger1-real-data-rerun"
    if runtime_trigger_open:
        return "trigger2-bounded-widening"
    if parity_trigger_open:
        return "trigger3-paper-backed-parity"
    if runtime_gate.gate_status == "trigger2-policy-missing":
        return "trigger2-policy-spec"
    if runtime_gate.gate_status == "trigger2-evidence-missing":
        return "trigger2-runtime-evidence"
    if empirical_audit.blocker_reason in {
        "missing-section6-manifest",
        "manifest-mismatch",
    }:
        return "trigger1-provenance-repair"
    if empirical_audit.blocker_reason == "missing-local-dataset":
        return "trigger1-local-dataset-intake"
    if parity_gate.gate_status == "archived-r-blocked-paper-backed-ready":
        return "trigger3-object-parity-spec"
    return "trigger2-policy-spec"


@dataclass(slots=True)
class Phase7NextMilestoneTriggerSnapshotReport:
    stage_label: str
    repo_root: str
    empirical_trigger_open: bool
    runtime_trigger_open: bool
    parity_trigger_open: bool
    open_trigger_names: tuple[str, ...]
    empirical_blocker_reason: str | None
    runtime_gate_status: str
    runtime_target_gate_status: str
    runtime_policy_digest: tuple[str, ...]
    parity_gate_status: str
    recommended_bounded_loop: str
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
        self.recommended_bounded_loop = str(self.recommended_bounded_loop).strip()
        self.canonical_snapshot_digest = tuple(
            str(line).rstrip() for line in self.canonical_snapshot_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "repo_root": self.repo_root,
            "empirical_trigger_open": self.empirical_trigger_open,
            "runtime_trigger_open": self.runtime_trigger_open,
            "parity_trigger_open": self.parity_trigger_open,
            "open_trigger_names": list(self.open_trigger_names),
            "empirical_blocker_reason": self.empirical_blocker_reason,
            "runtime_gate_status": self.runtime_gate_status,
            "runtime_target_gate_status": self.runtime_target_gate_status,
            "runtime_policy_digest": list(self.runtime_policy_digest),
            "parity_gate_status": self.parity_gate_status,
            "recommended_bounded_loop": self.recommended_bounded_loop,
            "recommended_feature_bundle": self.recommended_feature_bundle,
            "canonical_snapshot_digest": list(self.canonical_snapshot_digest),
        }

    @property
    def recommended_feature_bundle(self) -> str:
        return self.recommended_bounded_loop


def build_phase7_next_milestone_trigger_snapshot_report(
    empirical_audit: Section6ProvenanceGateAudit,
    runtime_gate: Phase7MonteCarloWideningTriggerGateReport,
    parity_gate: Phase7OuterInferenceTriggerGateReport,
) -> Phase7NextMilestoneTriggerSnapshotReport:
    empirical_trigger_open = False
    runtime_trigger_open = _runtime_trigger_open(runtime_gate)
    parity_trigger_open = _parity_trigger_open(parity_gate)
    open_trigger_names = _open_trigger_names(
        empirical_trigger_open=empirical_trigger_open,
        runtime_trigger_open=runtime_trigger_open,
        parity_trigger_open=parity_trigger_open,
    )
    runtime_transition = build_phase7_monte_carlo_widening_policy_transition(
        runtime_gate
    )
    runtime_target_gate_status = _runtime_target_gate_status(
        runtime_transition.target_gate_status
    )
    runtime_policy_digest = _runtime_policy_digest(
        runtime_gate,
        runtime_transition.policy_digest,
    )
    runtime_next_rung_line = None
    if runtime_policy_digest:
        runtime_next_rung_line = (
            f"- Trigger 2 next rung after `{runtime_transition.trigger_label}`: "
            f"`{runtime_target_gate_status}` via "
            f"`{runtime_policy_digest[0]}`"
        )
    recommended_bounded_loop = _recommended_bounded_loop(
        empirical_audit,
        runtime_gate,
        parity_gate,
        empirical_trigger_open=empirical_trigger_open,
        runtime_trigger_open=runtime_trigger_open,
        parity_trigger_open=parity_trigger_open,
    )
    empirical_status_line = (
        f"- Trigger 1 empirical: `{empirical_audit.status}` "
        f"(`{empirical_audit.blocker_reason}`)"
        if empirical_audit.blocker_reason is not None
        else f"- Trigger 1 empirical: `{empirical_audit.status}`"
    )
    open_trigger_digest = (
        ", ".join(f"`{name}`" for name in open_trigger_names)
        if open_trigger_names
        else "`none`"
    )
    canonical_snapshot_digest = (
        empirical_status_line,
        f"- Trigger 2 runtime evidence: `{runtime_gate.gate_status}`",
        *((runtime_next_rung_line,) if runtime_next_rung_line is not None else ()),
        f"- Trigger 3 parity oracle: `{parity_gate.gate_status}`",
        "- open triggers: "
        f"{open_trigger_digest}; recommended feature bundle: "
        f"`{recommended_bounded_loop}`",
    )
    return Phase7NextMilestoneTriggerSnapshotReport(
        stage_label="phase7-next-milestone-trigger-snapshot",
        repo_root=empirical_audit.repo_root,
        empirical_trigger_open=empirical_trigger_open,
        runtime_trigger_open=runtime_trigger_open,
        parity_trigger_open=parity_trigger_open,
        open_trigger_names=open_trigger_names,
        empirical_blocker_reason=empirical_audit.blocker_reason,
        runtime_gate_status=runtime_gate.gate_status,
        runtime_target_gate_status=runtime_target_gate_status,
        runtime_policy_digest=runtime_policy_digest,
        parity_gate_status=parity_gate.gate_status,
        recommended_bounded_loop=recommended_bounded_loop,
        canonical_snapshot_digest=canonical_snapshot_digest,
    )


def run_phase7_next_milestone_trigger_snapshot(
    repo_root: str | Path,
) -> Phase7NextMilestoneTriggerSnapshotReport:
    root = _coerce_repo_root(repo_root)
    current_gate = run_phase7_monte_carlo_widening_trigger_gate()
    return build_phase7_next_milestone_trigger_snapshot_report(
        audit_section6_provenance_gate(root),
        _build_repo_side_target_gate(current_gate),
        run_phase7_outer_inference_trigger_gate(),
    )
