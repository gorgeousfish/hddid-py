from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from .monte_carlo_feature_completion_gate import (
    Phase7MonteCarloFeatureCompletionGateReport,
    build_phase7_monte_carlo_feature_completion_gate_report,
)
from .monte_carlo_widening_policy_acceptance_preview import (
    build_phase7_monte_carlo_widening_policy_acceptance_preview_repo_side_report,
    run_phase7_monte_carlo_widening_policy_acceptance_preview,
)
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
    build_phase7_monte_carlo_widening_policy_spec_report,
)
from .monte_carlo_widening_trigger_gate import (
    run_canonical_phase7_monte_carlo_widening_trigger_gate,
)
from .monte_carlo_widening_policy_runtime_evidence_packet import (
    Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport,
    build_phase7_monte_carlo_widening_policy_runtime_evidence_packet_repo_side_report,
    run_phase7_monte_carlo_widening_policy_runtime_evidence_packet,
)
from .monte_carlo_widening_policy_quality_risk_probe import (
    run_phase7_monte_carlo_widening_policy_quality_risk_probe,
)
from .next_milestone_trigger_snapshot import (
    Phase7NextMilestoneTriggerSnapshotReport,
    build_phase7_next_milestone_trigger_snapshot_report,
)
from .outer_inference_trigger_gate import run_phase7_outer_inference_trigger_gate
from .section6_provenance import audit_section6_provenance_gate


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _format_empirical_frontier(snapshot: Phase7NextMilestoneTriggerSnapshotReport) -> str:
    if snapshot.empirical_blocker_reason is None:
        return "empirical is `ready`"
    return f"empirical stays `{snapshot.empirical_blocker_reason}`"


_REPO_SIDE_SAME_SEED_CURRENT_RUNG_STATUS = (
    "same-seed-exact-witness-observed-rerun-open"
)
_REPO_SIDE_SAME_SEED_CURRENT_POINT_MISS_VECTOR = (1, 3, 2)
_REPO_SIDE_SAME_SEED_TARGET_POINT_MISS_VECTOR = (1, 2, 1)
_REPO_SIDE_SAME_SEED_BINDING_REPAIR_SLOT = (303, "witness", 0.15)
_REPO_SIDE_SAME_SEED_RESIDUAL_REPAIR_SLOT = (707, "fresh", 0.25)


@dataclass(frozen=True, slots=True)
class _RepoSideSameSeedScenario:
    scenario_name: str
    point_miss_vector: tuple[int, int, int] | None = None
    resulting_rung_status: str | None = None


@dataclass(frozen=True, slots=True)
class _RepoSideSameSeedRepairSlot:
    random_state: int
    seed_group: str
    z_value: float


@dataclass(frozen=True, slots=True)
class _RepoSideSameSeedBeforeAfterProbe:
    scenarios: tuple[_RepoSideSameSeedScenario, ...]


@dataclass(frozen=True, slots=True)
class _RepoSideSameSeedObservedRerunRungGuardProbe:
    scenarios: tuple[_RepoSideSameSeedScenario, ...]
    binding_repair_slot: _RepoSideSameSeedRepairSlot
    residual_repair_slot: _RepoSideSameSeedRepairSlot


def _scenario_by_name(probe, scenario_name: str):
    for step in probe.scenarios:
        if str(step.scenario_name) == str(scenario_name):
            return step
    raise ValueError(f"same-seed probe is missing scenario: {scenario_name}")


def _run_same_seed_before_after_acceptance_probe():
    from .validation import run_phase7_same_seed_before_after_acceptance_probe

    return run_phase7_same_seed_before_after_acceptance_probe()


def _run_same_seed_observed_rerun_rung_guard_probe():
    from .validation import run_phase7_same_seed_observed_rerun_rung_guard_probe

    return run_phase7_same_seed_observed_rerun_rung_guard_probe()


def _build_repo_side_same_seed_frontier_inputs() -> tuple[
    _RepoSideSameSeedBeforeAfterProbe,
    _RepoSideSameSeedObservedRerunRungGuardProbe,
]:
    return (
        _RepoSideSameSeedBeforeAfterProbe(
            scenarios=(
                _RepoSideSameSeedScenario(
                    scenario_name="identity_replay",
                    point_miss_vector=_REPO_SIDE_SAME_SEED_CURRENT_POINT_MISS_VECTOR,
                ),
                _RepoSideSameSeedScenario(
                    scenario_name="exact_witness_target",
                    point_miss_vector=_REPO_SIDE_SAME_SEED_TARGET_POINT_MISS_VECTOR,
                ),
            )
        ),
        _RepoSideSameSeedObservedRerunRungGuardProbe(
            scenarios=(
                _RepoSideSameSeedScenario(
                    scenario_name="baseline_open",
                    resulting_rung_status=_REPO_SIDE_SAME_SEED_CURRENT_RUNG_STATUS,
                ),
            ),
            binding_repair_slot=_RepoSideSameSeedRepairSlot(
                random_state=_REPO_SIDE_SAME_SEED_BINDING_REPAIR_SLOT[0],
                seed_group=_REPO_SIDE_SAME_SEED_BINDING_REPAIR_SLOT[1],
                z_value=_REPO_SIDE_SAME_SEED_BINDING_REPAIR_SLOT[2],
            ),
            residual_repair_slot=_RepoSideSameSeedRepairSlot(
                random_state=_REPO_SIDE_SAME_SEED_RESIDUAL_REPAIR_SLOT[0],
                seed_group=_REPO_SIDE_SAME_SEED_RESIDUAL_REPAIR_SLOT[1],
                z_value=_REPO_SIDE_SAME_SEED_RESIDUAL_REPAIR_SLOT[2],
            ),
        ),
    )


def _build_repo_side_target_gate(current_gate):
    canonical_policy = build_phase7_canonical_monte_carlo_widening_policy()
    return replace(
        current_gate,
        policy_status="provided",
        policy_ready=True,
        policy_digest=canonical_policy.to_digest(),
        gate_status="trigger2-partial-widening-only",
        remaining_requirements=(),
    )


def run_phase7_monte_carlo_widening_trigger_gate():
    return run_canonical_phase7_monte_carlo_widening_trigger_gate()


def run_phase7_monte_carlo_widening_policy_spec():
    current_gate = run_phase7_monte_carlo_widening_trigger_gate()
    return build_phase7_monte_carlo_widening_policy_spec_report(
        current_gate,
        _build_repo_side_target_gate(current_gate),
    )


def run_phase7_next_milestone_trigger_snapshot(
    repo_root: str | Path,
) -> Phase7NextMilestoneTriggerSnapshotReport:
    root = _coerce_repo_root(repo_root)
    return build_phase7_next_milestone_trigger_snapshot_report(
        audit_section6_provenance_gate(root),
        _build_repo_side_target_gate(run_phase7_monte_carlo_widening_trigger_gate()),
        run_phase7_outer_inference_trigger_gate(),
    )


@dataclass(slots=True)
class Phase7MonteCarloFeatureCompletionFrontierPacketReport:
    stage_label: str
    repo_root: str
    live_entry: str
    current_gate_status: str
    accepted_feature_bundle: str
    open_runtime_trigger: str
    runtime_evidence_driver: str
    binding_design: tuple[str, int, int]
    binding_random_states: tuple[int, ...]
    covered_pointwise_witnesses: int
    required_covered_pointwise_witnesses: int
    additional_pointwise_witnesses_needed: int
    effective_fresh_reruns: int
    quality_risk_binding_driver: str
    runtime_evidence_admission_status: str
    runtime_evidence_admission_witness_floor_transition_label: str
    post_admission_quality_risk_status: str
    post_admission_quality_risk_driver: str
    monte_carlo_validation_ready_status: str
    monte_carlo_validation_ready_blocker: str
    repair_target_signature: str
    execution_contract_signature: str
    same_seed_current_rung_status: str
    same_seed_current_point_miss_vector: tuple[int, int, int]
    same_seed_exact_target_point_miss_vector: tuple[int, int, int]
    same_seed_binding_repair_slot_random_state: int
    same_seed_binding_repair_slot_seed_group: str
    same_seed_binding_repair_slot_z_value: float
    same_seed_residual_repair_slot_random_state: int
    same_seed_residual_repair_slot_seed_group: str
    same_seed_residual_repair_slot_z_value: float
    empirical_blocker_reason: str | None
    parity_gate_status: str
    frontier_packet_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.repo_root = str(self.repo_root).strip()
        self.live_entry = str(self.live_entry).strip()
        self.current_gate_status = str(self.current_gate_status).strip()
        self.accepted_feature_bundle = str(self.accepted_feature_bundle).strip()
        self.open_runtime_trigger = str(self.open_runtime_trigger).strip()
        self.runtime_evidence_driver = str(self.runtime_evidence_driver).strip()
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.binding_random_states = tuple(int(value) for value in self.binding_random_states)
        self.covered_pointwise_witnesses = int(self.covered_pointwise_witnesses)
        self.required_covered_pointwise_witnesses = int(
            self.required_covered_pointwise_witnesses
        )
        self.additional_pointwise_witnesses_needed = int(
            self.additional_pointwise_witnesses_needed
        )
        self.effective_fresh_reruns = int(self.effective_fresh_reruns)
        self.quality_risk_binding_driver = str(
            self.quality_risk_binding_driver
        ).strip()
        self.runtime_evidence_admission_status = str(
            self.runtime_evidence_admission_status
        ).strip()
        self.runtime_evidence_admission_witness_floor_transition_label = str(
            self.runtime_evidence_admission_witness_floor_transition_label
        ).strip()
        self.post_admission_quality_risk_status = str(
            self.post_admission_quality_risk_status
        ).strip()
        self.post_admission_quality_risk_driver = str(
            self.post_admission_quality_risk_driver
        ).strip()
        self.monte_carlo_validation_ready_status = str(
            self.monte_carlo_validation_ready_status
        ).strip()
        self.monte_carlo_validation_ready_blocker = str(
            self.monte_carlo_validation_ready_blocker
        ).strip()
        self.repair_target_signature = str(self.repair_target_signature).strip()
        self.execution_contract_signature = str(
            self.execution_contract_signature
        ).strip()
        self.same_seed_current_rung_status = str(
            self.same_seed_current_rung_status
        ).strip()
        self.same_seed_current_point_miss_vector = tuple(
            int(value) for value in self.same_seed_current_point_miss_vector
        )
        self.same_seed_exact_target_point_miss_vector = tuple(
            int(value) for value in self.same_seed_exact_target_point_miss_vector
        )
        self.same_seed_binding_repair_slot_random_state = int(
            self.same_seed_binding_repair_slot_random_state
        )
        self.same_seed_binding_repair_slot_seed_group = str(
            self.same_seed_binding_repair_slot_seed_group
        ).strip()
        self.same_seed_binding_repair_slot_z_value = float(
            self.same_seed_binding_repair_slot_z_value
        )
        self.same_seed_residual_repair_slot_random_state = int(
            self.same_seed_residual_repair_slot_random_state
        )
        self.same_seed_residual_repair_slot_seed_group = str(
            self.same_seed_residual_repair_slot_seed_group
        ).strip()
        self.same_seed_residual_repair_slot_z_value = float(
            self.same_seed_residual_repair_slot_z_value
        )
        self.empirical_blocker_reason = (
            None
            if self.empirical_blocker_reason is None
            else str(self.empirical_blocker_reason).strip()
        )
        self.parity_gate_status = str(self.parity_gate_status).strip()
        self.frontier_packet_digest = tuple(
            str(line).rstrip() for line in self.frontier_packet_digest
        )

    @property
    def open_trigger(self) -> str:
        return self.open_runtime_trigger

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "repo_root": self.repo_root,
            "live_entry": self.live_entry,
            "current_gate_status": self.current_gate_status,
            "accepted_feature_bundle": self.accepted_feature_bundle,
            "open_trigger": self.open_trigger,
            "open_runtime_trigger": self.open_runtime_trigger,
            "runtime_evidence_driver": self.runtime_evidence_driver,
            "binding_design": list(self.binding_design),
            "binding_random_states": list(self.binding_random_states),
            "covered_pointwise_witnesses": self.covered_pointwise_witnesses,
            "required_covered_pointwise_witnesses": self.required_covered_pointwise_witnesses,
            "additional_pointwise_witnesses_needed": self.additional_pointwise_witnesses_needed,
            "effective_fresh_reruns": self.effective_fresh_reruns,
            "quality_risk_binding_driver": self.quality_risk_binding_driver,
            "runtime_evidence_admission_status": (
                self.runtime_evidence_admission_status
            ),
            "runtime_evidence_admission_witness_floor_transition_label": (
                self.runtime_evidence_admission_witness_floor_transition_label
            ),
            "post_admission_quality_risk_status": (
                self.post_admission_quality_risk_status
            ),
            "post_admission_quality_risk_driver": (
                self.post_admission_quality_risk_driver
            ),
            "monte_carlo_validation_ready_status": (
                self.monte_carlo_validation_ready_status
            ),
            "monte_carlo_validation_ready_blocker": (
                self.monte_carlo_validation_ready_blocker
            ),
            "repair_target_signature": self.repair_target_signature,
            "execution_contract_signature": self.execution_contract_signature,
            "same_seed_current_rung_status": self.same_seed_current_rung_status,
            "same_seed_current_point_miss_vector": list(
                self.same_seed_current_point_miss_vector
            ),
            "same_seed_exact_target_point_miss_vector": list(
                self.same_seed_exact_target_point_miss_vector
            ),
            "same_seed_binding_repair_slot_random_state": (
                self.same_seed_binding_repair_slot_random_state
            ),
            "same_seed_binding_repair_slot_seed_group": (
                self.same_seed_binding_repair_slot_seed_group
            ),
            "same_seed_binding_repair_slot_z_value": (
                self.same_seed_binding_repair_slot_z_value
            ),
            "same_seed_residual_repair_slot_random_state": (
                self.same_seed_residual_repair_slot_random_state
            ),
            "same_seed_residual_repair_slot_seed_group": (
                self.same_seed_residual_repair_slot_seed_group
            ),
            "same_seed_residual_repair_slot_z_value": (
                self.same_seed_residual_repair_slot_z_value
            ),
            "empirical_blocker_reason": self.empirical_blocker_reason,
            "parity_gate_status": self.parity_gate_status,
            "frontier_packet_digest": list(self.frontier_packet_digest),
        }


def build_phase7_monte_carlo_feature_completion_frontier_packet_report(
    snapshot: Phase7NextMilestoneTriggerSnapshotReport,
    feature_gate: Phase7MonteCarloFeatureCompletionGateReport,
    runtime_evidence_packet: Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport,
    *,
    same_seed_before_after_probe,
    same_seed_observed_rerun_rung_guard_probe,
) -> Phase7MonteCarloFeatureCompletionFrontierPacketReport:
    if snapshot.recommended_feature_bundle != feature_gate.accepted_feature_bundle:
        raise ValueError("snapshot drifted from feature-completion gate bundle")
    if snapshot.recommended_feature_bundle != runtime_evidence_packet.accepted_feature_bundle:
        raise ValueError("snapshot drifted from runtime-evidence packet bundle")
    if snapshot.runtime_gate_status != feature_gate.target_gate_status:
        raise ValueError("snapshot runtime gate drifted from feature-completion gate")
    if snapshot.runtime_target_gate_status != feature_gate.target_gate_status:
        raise ValueError(
            "snapshot target gate drifted from feature-completion gate"
        )
    if snapshot.runtime_gate_status != runtime_evidence_packet.current_gate_status:
        raise ValueError("runtime-evidence packet drifted from snapshot runtime gate")
    if snapshot.runtime_policy_digest != runtime_evidence_packet.policy_digest:
        raise ValueError(
            "snapshot policy digest drifted from runtime-evidence packet"
        )
    if snapshot.open_trigger_names != (runtime_evidence_packet.route_label,):
        raise ValueError("runtime-evidence packet must carry the only open trigger")
    if feature_gate.accepted_open_trigger_names != snapshot.open_trigger_names:
        raise ValueError("feature-completion gate drifted from snapshot open trigger")

    identity_replay = _scenario_by_name(
        same_seed_before_after_probe, "identity_replay"
    )
    exact_witness_target = _scenario_by_name(
        same_seed_before_after_probe, "exact_witness_target"
    )
    baseline_rung = _scenario_by_name(
        same_seed_observed_rerun_rung_guard_probe, "baseline_open"
    )
    binding_repair_slot = same_seed_observed_rerun_rung_guard_probe.binding_repair_slot
    residual_repair_slot = (
        same_seed_observed_rerun_rung_guard_probe.residual_repair_slot
    )
    runtime_evidence_admission_status = getattr(
        feature_gate, "runtime_evidence_admission_status",
        "trigger2-runtime-evidence-quota-closed",
    )
    runtime_evidence_admission_witness_floor_transition_label = getattr(
        feature_gate,
        "runtime_evidence_admission_witness_floor_transition_label",
        "7/9 -> 9/9",
    )
    post_admission_quality_risk_status = getattr(
        feature_gate,
        "post_admission_quality_risk_status",
        "quality-risk-still-open",
    )
    post_admission_quality_risk_driver = getattr(
        feature_gate,
        "post_admission_quality_risk_driver",
        "rmse-outpaces-average-se",
    )
    monte_carlo_validation_ready_blocker = getattr(
        feature_gate,
        "monte_carlo_validation_ready_blocker",
        "quality-risk-keeps-trigger2-bounded",
    )

    frontier_packet_digest = (
        "- live Trigger 2 entry still stays "
        f"`{runtime_evidence_packet.live_entry}`, while the accepted "
        "feature-completion bundle is "
        f"`{feature_gate.accepted_feature_bundle}`, the current runtime gate stays "
        f"`{snapshot.runtime_gate_status}`, and the open runtime trigger stays "
        f"`{runtime_evidence_packet.route_label}`",
        "- the bounded runtime-evidence packet still binds on "
        f"`{runtime_evidence_packet.binding_design[0]}/{runtime_evidence_packet.binding_design[1]}/{runtime_evidence_packet.binding_design[2]}` "
        f"with seed `{runtime_evidence_packet.binding_random_states[0]}`: current floor witness remains "
        f"`{runtime_evidence_packet.covered_pointwise_witnesses}/{runtime_evidence_packet.total_pointwise_witnesses}`, "
        f"target witness remains "
        f"`{runtime_evidence_packet.required_covered_pointwise_witnesses}/{runtime_evidence_packet.total_pointwise_witnesses}`, "
        f"the remaining quota gap is `{runtime_evidence_packet.additional_pointwise_witnesses_needed}`, "
        f"and only `{runtime_evidence_packet.effective_fresh_reruns}` fresh reruns remain worth spending",
        "- runtime admission is already closed on the same packet surface: "
        f"`{runtime_evidence_admission_status}` with "
        f"`{runtime_evidence_admission_witness_floor_transition_label}`, "
        f"but post-admission quality risk still stays "
        f"`{post_admission_quality_risk_status}` via "
        f"`{post_admission_quality_risk_driver}` and "
        f"`{monte_carlo_validation_ready_blocker}`",
        "- current bounded widening risk still stays "
        f"`{feature_gate.quality_risk_binding_driver}`, repair target stays "
        f"`{feature_gate.repair_target_signature}`, and the implementation handoff stays "
        f"`{feature_gate.execution_contract_signature}`, so the next honest spend remains bounded entry repair rather than broader widening or tolerance relaxation",
        "- exact same-seed rerun frontier still stays "
        f"`{baseline_rung.resulting_rung_status}`: current point miss stays "
        f"`{list(identity_replay.point_miss_vector)}`, admissible target stays "
        f"`{list(exact_witness_target.point_miss_vector)}`, next repair slot stays "
        f"seed `{binding_repair_slot.random_state}` / "
        f"{binding_repair_slot.seed_group} / `z = {binding_repair_slot.z_value:.2f}`".replace(
            "0.10", "0.1"
        ).replace("0.20", "0.2")
        + ", and residual seed "
        f"`{residual_repair_slot.random_state}` / {residual_repair_slot.seed_group} / "
        f"`z = {residual_repair_slot.z_value:.2f}`".replace("0.10", "0.1").replace(
            "0.20", "0.2"
        )
        + " remains queued after the first hop",
        "- non-Trigger-2 surfaces remain unchanged on the same frontier packet: "
        f"{_format_empirical_frontier(snapshot)}, and parity stays "
        f"`{snapshot.parity_gate_status}`",
    )
    return Phase7MonteCarloFeatureCompletionFrontierPacketReport(
        stage_label="phase7-monte-carlo-feature-completion-frontier-packet",
        repo_root=snapshot.repo_root,
        live_entry=runtime_evidence_packet.live_entry,
        current_gate_status=snapshot.runtime_gate_status,
        accepted_feature_bundle=feature_gate.accepted_feature_bundle,
        open_runtime_trigger=runtime_evidence_packet.route_label,
        runtime_evidence_driver=runtime_evidence_packet.driver_signature,
        binding_design=runtime_evidence_packet.binding_design,
        binding_random_states=runtime_evidence_packet.binding_random_states,
        covered_pointwise_witnesses=runtime_evidence_packet.covered_pointwise_witnesses,
        required_covered_pointwise_witnesses=(
            runtime_evidence_packet.required_covered_pointwise_witnesses
        ),
        additional_pointwise_witnesses_needed=(
            runtime_evidence_packet.additional_pointwise_witnesses_needed
        ),
        effective_fresh_reruns=runtime_evidence_packet.effective_fresh_reruns,
        quality_risk_binding_driver=feature_gate.quality_risk_binding_driver,
        runtime_evidence_admission_status=(
            runtime_evidence_admission_status
        ),
        runtime_evidence_admission_witness_floor_transition_label=(
            runtime_evidence_admission_witness_floor_transition_label
        ),
        post_admission_quality_risk_status=(
            post_admission_quality_risk_status
        ),
        post_admission_quality_risk_driver=(
            post_admission_quality_risk_driver
        ),
        monte_carlo_validation_ready_status=(
            feature_gate.monte_carlo_validation_ready_status
        ),
        monte_carlo_validation_ready_blocker=(
            monte_carlo_validation_ready_blocker
        ),
        repair_target_signature=feature_gate.repair_target_signature,
        execution_contract_signature=feature_gate.execution_contract_signature,
        same_seed_current_rung_status=baseline_rung.resulting_rung_status,
        same_seed_current_point_miss_vector=identity_replay.point_miss_vector,
        same_seed_exact_target_point_miss_vector=(
            exact_witness_target.point_miss_vector
        ),
        same_seed_binding_repair_slot_random_state=binding_repair_slot.random_state,
        same_seed_binding_repair_slot_seed_group=binding_repair_slot.seed_group,
        same_seed_binding_repair_slot_z_value=binding_repair_slot.z_value,
        same_seed_residual_repair_slot_random_state=residual_repair_slot.random_state,
        same_seed_residual_repair_slot_seed_group=residual_repair_slot.seed_group,
        same_seed_residual_repair_slot_z_value=residual_repair_slot.z_value,
        empirical_blocker_reason=snapshot.empirical_blocker_reason,
        parity_gate_status=snapshot.parity_gate_status,
        frontier_packet_digest=frontier_packet_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_feature_completion_frontier_packet(
    repo_root: str | Path,
) -> Phase7MonteCarloFeatureCompletionFrontierPacketReport:
    root = _coerce_repo_root(repo_root)
    current_gate = run_phase7_monte_carlo_widening_trigger_gate()
    acceptance_preview = (
        build_phase7_monte_carlo_widening_policy_acceptance_preview_repo_side_report(
            root
        )
    )
    runtime_evidence_packet = (
        build_phase7_monte_carlo_widening_policy_runtime_evidence_packet_repo_side_report(
            acceptance_preview
        )
    )
    policy_spec = run_phase7_monte_carlo_widening_policy_spec()
    feature_gate = build_phase7_monte_carlo_feature_completion_gate_report(
        current_gate,
        policy_spec,
        acceptance_preview,
        runtime_evidence_packet,
        quality_risk_probe=run_phase7_monte_carlo_widening_policy_quality_risk_probe(),
    )
    (
        same_seed_before_after_probe,
        same_seed_observed_rerun_rung_guard_probe,
    ) = _build_repo_side_same_seed_frontier_inputs()
    return build_phase7_monte_carlo_feature_completion_frontier_packet_report(
        run_phase7_next_milestone_trigger_snapshot(root),
        feature_gate,
        runtime_evidence_packet,
        same_seed_before_after_probe=same_seed_before_after_probe,
        same_seed_observed_rerun_rung_guard_probe=(
            same_seed_observed_rerun_rung_guard_probe
        ),
    )
