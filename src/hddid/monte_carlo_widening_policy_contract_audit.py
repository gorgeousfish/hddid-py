from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from .monte_carlo_widening_policy_acceptance_preview import (
    Phase7MonteCarloWideningPolicyAcceptancePreviewReport,
    run_phase7_monte_carlo_widening_policy_acceptance_preview,
)
from .monte_carlo_widening_policy_candidate_guard import (
    Phase7MonteCarloWideningPolicyCandidateGuardReport,
    run_phase7_monte_carlo_widening_policy_candidate_guard,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract,
)
from .monte_carlo_widening_policy_guard_probe import (
    Phase7MonteCarloWideningPolicyGuardProbeReport,
    run_phase7_monte_carlo_widening_policy_guard_probe,
)
from .monte_carlo_widening_policy_spec import (
    Phase7MonteCarloWideningPolicySpecReport,
    run_phase7_monte_carlo_widening_policy_spec,
)
from .monte_carlo_widening_trigger_gate import (
    Phase7MonteCarloWideningPolicy,
    run_phase7_monte_carlo_widening_trigger_gate,
)
from .next_milestone_trigger_snapshot import (
    Phase7NextMilestoneTriggerSnapshotReport,
    build_phase7_next_milestone_trigger_snapshot_report,
)
from .outer_inference_trigger_gate import run_phase7_outer_inference_trigger_gate
from .section6_provenance import audit_section6_provenance_gate


_CONTRACT_AUDIT_CACHE_EPOCH_RELATIVE_PATHS = (
    Path("Docs/automation/automation-state.yaml"),
    Path(".planning/STATE.md"),
    Path(
        ".planning/phases/07-final-hardening-and-verification-debt-closure/07-NEXT-MILESTONE-HANDOFF.md"
    ),
    Path("Docs/research/phase7_next_milestone_trigger_snapshot.md"),
    Path("Docs/research/phase7_monte_carlo_widening_policy_candidate_guard.md"),
    Path("Docs/research/phase7_monte_carlo_widening_policy_contract_audit.md"),
    Path("Docs/research/phase7_monte_carlo_widening_policy_spec.md"),
    Path("Docs/research/phase7_monte_carlo_widening_policy_acceptance_preview.md"),
    Path("Docs/research/phase7_monte_carlo_widening_policy_guard_probe.md"),
    Path(
        "Docs/research/phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract.md"
    ),
    Path("hddid-py/src/hddid/next_milestone_trigger_snapshot.py"),
    Path("hddid-py/src/hddid/monte_carlo_widening_policy_candidate_guard.py"),
    Path("hddid-py/src/hddid/monte_carlo_widening_policy_contract_audit.py"),
    Path("hddid-py/src/hddid/monte_carlo_widening_policy_spec.py"),
    Path("hddid-py/src/hddid/monte_carlo_widening_policy_acceptance_preview.py"),
    Path("hddid-py/src/hddid/monte_carlo_widening_policy_guard_probe.py"),
    Path(
        "hddid-py/src/hddid/monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract.py"
    ),
)
_LAST_CONTRACT_AUDIT_CACHE_EPOCH_BY_ROOT: dict[str, int] = {}


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _build_phase7_monte_carlo_widening_policy_contract_audit_closeout_snapshot(
    repo_root: str | Path,
) -> Phase7NextMilestoneTriggerSnapshotReport:
    root = _coerce_repo_root(repo_root)
    return build_phase7_next_milestone_trigger_snapshot_report(
        audit_section6_provenance_gate(root),
        run_phase7_monte_carlo_widening_trigger_gate(),
        run_phase7_outer_inference_trigger_gate(),
    )


def _phase7_monte_carlo_widening_policy_contract_audit_cache_epoch(
    repo_root: str | Path,
) -> int:
    root = _coerce_repo_root(repo_root)
    return max(
        (root / relative_path).stat().st_mtime_ns
        for relative_path in _CONTRACT_AUDIT_CACHE_EPOCH_RELATIVE_PATHS
        if (root / relative_path).exists()
    )


def _format_design_keys(
    design_keys: tuple[tuple[str, int, int], ...],
) -> str:
    if not design_keys:
        return "none"
    return ", ".join(f"{dgp}/{n_obs}/{p}" for dgp, n_obs, p in design_keys)


def _format_trigger_names(trigger_names: tuple[str, ...]) -> str:
    if not trigger_names:
        return "`none`"
    if len(trigger_names) == 1:
        return f"`{trigger_names[0]}`"
    return ", ".join(f"`{name}`" for name in trigger_names)


def _format_empirical_status(blocker_reason: str | None) -> str:
    if blocker_reason is None:
        return "empirical `ready`"
    return f"empirical `{blocker_reason}`"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyContractAuditReport:
    stage_label: str
    repo_root: str
    current_runtime_gate_status: str
    accepted_runtime_gate_status: str
    current_recommended_bounded_loop: str
    accepted_recommended_bounded_loop: str
    scope_guard_status: str
    implementation_handoff_contract: str
    validation_only_companion_bundle: str
    validation_only_estimator_object_flow_contract: str
    validation_only_dense_replay_guard: str
    validation_only_fresh_rerun_acceptance_contract: str
    validation_only_estimator_evidence_intake_contract: str
    validation_only_estimator_evidence_target_gate_status: str
    validation_only_estimator_evidence_conditions: tuple[str, ...]
    validation_only_estimator_evidence_digest: tuple[str, ...]
    current_open_trigger_names: tuple[str, ...]
    accepted_open_trigger_names: tuple[str, ...]
    empirical_blocker_reason: str | None
    parity_gate_status: str
    policy_digest: tuple[str, ...]
    binding_guard: str
    binding_designs: tuple[tuple[str, int, int], ...]
    contract_audit_replay_path: str
    candidate_input_mode: str
    candidate_status: str
    candidate_resulting_gate_status: str
    candidate_policy_digest: tuple[str, ...]
    candidate_rejection_reasons: tuple[str, ...]
    candidate_binding_guard: str
    stable_partial_designs: tuple[tuple[str, int, int], ...]
    quality_risk_designs: tuple[tuple[str, int, int], ...]
    blocked_full_matrix_designs: tuple[tuple[str, int, int], ...]
    canonical_contract_digest: tuple[str, ...]
    scope_guard_digest: tuple[str, ...]
    candidate_guard_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.repo_root = str(self.repo_root).strip()
        self.current_runtime_gate_status = str(self.current_runtime_gate_status).strip()
        self.accepted_runtime_gate_status = str(
            self.accepted_runtime_gate_status
        ).strip()
        self.current_recommended_bounded_loop = str(
            self.current_recommended_bounded_loop
        ).strip()
        self.accepted_recommended_bounded_loop = str(
            self.accepted_recommended_bounded_loop
        ).strip()
        self.scope_guard_status = str(self.scope_guard_status).strip()
        self.implementation_handoff_contract = str(
            self.implementation_handoff_contract
        ).strip()
        self.validation_only_companion_bundle = str(
            self.validation_only_companion_bundle
        ).strip()
        self.validation_only_estimator_object_flow_contract = str(
            self.validation_only_estimator_object_flow_contract
        ).strip()
        self.validation_only_dense_replay_guard = str(
            self.validation_only_dense_replay_guard
        ).strip()
        self.validation_only_fresh_rerun_acceptance_contract = str(
            self.validation_only_fresh_rerun_acceptance_contract
        ).strip()
        self.validation_only_estimator_evidence_intake_contract = str(
            self.validation_only_estimator_evidence_intake_contract
        ).strip()
        self.validation_only_estimator_evidence_target_gate_status = str(
            self.validation_only_estimator_evidence_target_gate_status
        ).strip()
        self.validation_only_estimator_evidence_conditions = tuple(
            str(item).strip()
            for item in self.validation_only_estimator_evidence_conditions
        )
        self.validation_only_estimator_evidence_digest = tuple(
            str(line).rstrip()
            for line in self.validation_only_estimator_evidence_digest
        )
        self.current_open_trigger_names = tuple(
            str(name).strip() for name in self.current_open_trigger_names
        )
        self.accepted_open_trigger_names = tuple(
            str(name).strip() for name in self.accepted_open_trigger_names
        )
        self.empirical_blocker_reason = (
            None
            if self.empirical_blocker_reason is None
            else str(self.empirical_blocker_reason).strip()
        )
        self.parity_gate_status = str(self.parity_gate_status).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_guard = str(self.binding_guard).strip()
        self.binding_designs = tuple(
            (str(dgp).strip(), int(n_obs), int(p))
            for dgp, n_obs, p in self.binding_designs
        )
        self.contract_audit_replay_path = str(self.contract_audit_replay_path).strip()
        self.candidate_input_mode = str(self.candidate_input_mode).strip()
        self.candidate_status = str(self.candidate_status).strip()
        self.candidate_resulting_gate_status = str(
            self.candidate_resulting_gate_status
        ).strip()
        self.candidate_policy_digest = tuple(
            str(item).strip() for item in self.candidate_policy_digest
        )
        self.candidate_rejection_reasons = tuple(
            str(item).strip() for item in self.candidate_rejection_reasons
        )
        self.candidate_binding_guard = str(self.candidate_binding_guard).strip()
        self.stable_partial_designs = tuple(
            (str(dgp).strip(), int(n_obs), int(p))
            for dgp, n_obs, p in self.stable_partial_designs
        )
        self.quality_risk_designs = tuple(
            (str(dgp).strip(), int(n_obs), int(p))
            for dgp, n_obs, p in self.quality_risk_designs
        )
        self.blocked_full_matrix_designs = tuple(
            (str(dgp).strip(), int(n_obs), int(p))
            for dgp, n_obs, p in self.blocked_full_matrix_designs
        )
        self.canonical_contract_digest = tuple(
            str(line).rstrip() for line in self.canonical_contract_digest
        )
        self.scope_guard_digest = tuple(
            str(line).rstrip() for line in self.scope_guard_digest
        )
        self.candidate_guard_digest = tuple(
            str(line).rstrip() for line in self.candidate_guard_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "repo_root": self.repo_root,
            "current_runtime_gate_status": self.current_runtime_gate_status,
            "accepted_runtime_gate_status": self.accepted_runtime_gate_status,
            "current_recommended_bounded_loop": self.current_recommended_bounded_loop,
            "accepted_recommended_bounded_loop": self.accepted_recommended_bounded_loop,
            "current_recommended_feature_bundle": self.current_recommended_feature_bundle,
            "accepted_recommended_feature_bundle": self.accepted_recommended_feature_bundle,
            "scope_guard_status": self.scope_guard_status,
            "implementation_handoff_contract": self.implementation_handoff_contract,
            "validation_only_companion_bundle": self.validation_only_companion_bundle,
            "validation_only_estimator_object_flow_contract": (
                self.validation_only_estimator_object_flow_contract
            ),
            "validation_only_dense_replay_guard": (
                self.validation_only_dense_replay_guard
            ),
            "validation_only_fresh_rerun_acceptance_contract": (
                self.validation_only_fresh_rerun_acceptance_contract
            ),
            "validation_only_estimator_evidence_intake_contract": (
                self.validation_only_estimator_evidence_intake_contract
            ),
            "validation_only_estimator_evidence_target_gate_status": (
                self.validation_only_estimator_evidence_target_gate_status
            ),
            "validation_only_estimator_evidence_conditions": list(
                self.validation_only_estimator_evidence_conditions
            ),
            "validation_only_estimator_evidence_digest": list(
                self.validation_only_estimator_evidence_digest
            ),
            "current_open_trigger_names": list(self.current_open_trigger_names),
            "accepted_open_trigger_names": list(self.accepted_open_trigger_names),
            "empirical_blocker_reason": self.empirical_blocker_reason,
            "parity_gate_status": self.parity_gate_status,
            "policy_digest": list(self.policy_digest),
            "binding_guard": self.binding_guard,
            "binding_designs": [list(item) for item in self.binding_designs],
            "contract_audit_replay_path": self.contract_audit_replay_path,
            "candidate_input_mode": self.candidate_input_mode,
            "candidate_status": self.candidate_status,
            "candidate_resulting_gate_status": self.candidate_resulting_gate_status,
            "candidate_policy_digest": list(self.candidate_policy_digest),
            "candidate_rejection_reasons": list(self.candidate_rejection_reasons),
            "candidate_binding_guard": self.candidate_binding_guard,
            "stable_partial_designs": [
                list(item) for item in self.stable_partial_designs
            ],
            "quality_risk_designs": [list(item) for item in self.quality_risk_designs],
            "blocked_full_matrix_designs": [
                list(item) for item in self.blocked_full_matrix_designs
            ],
            "canonical_contract_digest": list(self.canonical_contract_digest),
            "scope_guard_digest": list(self.scope_guard_digest),
            "candidate_guard_digest": list(self.candidate_guard_digest),
        }

    @property
    def current_recommended_feature_bundle(self) -> str:
        return self.current_recommended_bounded_loop

    @property
    def accepted_recommended_feature_bundle(self) -> str:
        return self.accepted_recommended_bounded_loop


def build_phase7_monte_carlo_widening_policy_contract_audit_report(
    current_snapshot: Phase7NextMilestoneTriggerSnapshotReport,
    policy_spec: Phase7MonteCarloWideningPolicySpecReport,
    acceptance_preview: Phase7MonteCarloWideningPolicyAcceptancePreviewReport,
    guard_probe: Phase7MonteCarloWideningPolicyGuardProbeReport,
    candidate_guard: Phase7MonteCarloWideningPolicyCandidateGuardReport,
    fresh_estimator_evidence_contract: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport
    ),
    contract_audit_replay_path: str = "explicit-policy-live-rebuild",
) -> Phase7MonteCarloWideningPolicyContractAuditReport:
    if current_snapshot.repo_root != acceptance_preview.repo_root:
        raise ValueError("policy contract audit requires a single repo root")
    if current_snapshot.runtime_gate_status != policy_spec.current_gate_status:
        raise ValueError(
            "current snapshot drifted from canonical Trigger 2 policy spec"
        )
    if current_snapshot.runtime_target_gate_status != policy_spec.target_gate_status:
        raise ValueError(
            "current snapshot target gate drifted from canonical policy spec"
        )
    if (
        acceptance_preview.current_runtime_gate_status
        != current_snapshot.runtime_gate_status
    ):
        raise ValueError(
            "acceptance preview current gate drifted from closeout snapshot"
        )
    if (
        acceptance_preview.accepted_runtime_gate_status
        != policy_spec.target_gate_status
    ):
        raise ValueError(
            "acceptance preview accepted gate drifted from canonical policy spec"
        )
    if (
        current_snapshot.recommended_bounded_loop
        != acceptance_preview.current_recommended_bounded_loop
    ):
        raise ValueError(
            "acceptance preview current loop drifted from closeout snapshot"
        )
    if (
        current_snapshot.empirical_blocker_reason
        != acceptance_preview.empirical_blocker_reason
        or current_snapshot.parity_gate_status != acceptance_preview.parity_gate_status
    ):
        raise ValueError("acceptance preview drifted outside Trigger 2 blockers")

    policy_digest = policy_spec.policy_digest
    if (
        current_snapshot.runtime_policy_digest != policy_digest
        or acceptance_preview.policy_digest != policy_digest
        or guard_probe.policy_digest != policy_digest
    ):
        raise ValueError("policy digest drifted across Trigger 2 contract audit")
    if candidate_guard.canonical_policy_digest != policy_digest:
        raise ValueError(
            "candidate guard drifted from canonical Trigger 2 policy digest"
        )
    if fresh_estimator_evidence_contract.policy_digest != policy_digest:
        raise ValueError(
            "fresh estimator evidence intake drifted from canonical Trigger 2 policy digest"
        )

    if acceptance_preview.stable_partial_designs != policy_spec.stable_partial_designs:
        raise ValueError("stable partial slice drifted across Trigger 2 contract audit")
    if acceptance_preview.quality_risk_designs != policy_spec.quality_risk_designs:
        raise ValueError("quality risk slice drifted across Trigger 2 contract audit")
    if (
        acceptance_preview.blocked_full_matrix_designs
        != policy_spec.blocked_full_matrix_designs
    ):
        raise ValueError(
            "blocked full-matrix slice drifted across Trigger 2 contract audit"
        )
    if guard_probe.binding_guard != "nonparametric_coverage_floor":
        raise ValueError(
            "binding guard drifted from canonical Trigger 2 policy contract"
        )
    if guard_probe.binding_designs != policy_spec.quality_risk_designs:
        raise ValueError("binding design drifted from canonical Trigger 2 quality risk")
    if (
        fresh_estimator_evidence_contract.binding_design
        != policy_spec.quality_risk_designs[0]
    ):
        raise ValueError(
            "fresh estimator evidence intake drifted from canonical Trigger 2 quality risk"
        )
    if (
        fresh_estimator_evidence_contract.target_gate_status
        != acceptance_preview.accepted_runtime_gate_status
    ):
        raise ValueError(
            "fresh estimator evidence intake target gate drifted from accepted preview boundary"
        )
    if candidate_guard.accepted:
        if (
            candidate_guard.resulting_gate_status
            != acceptance_preview.accepted_runtime_gate_status
        ):
            raise ValueError(
                "candidate intake gate drifted from accepted preview boundary"
            )
    elif candidate_guard.resulting_gate_status != current_snapshot.runtime_gate_status:
        raise ValueError(
            "candidate rejection must keep the contract audit on the current closeout gate"
        )

    canonical_contract_digest = (
        "- current closeout snapshot: runtime gate "
        f"`{current_snapshot.runtime_gate_status}`; recommended feature bundle "
        f"`{current_snapshot.recommended_bounded_loop}`; open triggers "
        f"{_format_trigger_names(current_snapshot.open_trigger_names)}",
        "- accepted preview after canonical policy: runtime gate "
        f"`{acceptance_preview.accepted_runtime_gate_status}`; recommended feature bundle "
        f"`{acceptance_preview.accepted_recommended_bounded_loop}`; open trigger "
        f"{_format_trigger_names(acceptance_preview.accepted_open_trigger_names)}",
        "- unchanged blockers outside Trigger 2: "
        f"{_format_empirical_status(current_snapshot.empirical_blocker_reason)}; parity "
        f"`{current_snapshot.parity_gate_status}`",
        "- canonical policy digest: "
        + ", ".join(f"`{item}`" for item in policy_digest),
        "- bounded widening slice stays "
        f"`{_format_design_keys(policy_spec.stable_partial_designs)}`; "
        f"quality risk stays `{_format_design_keys(policy_spec.quality_risk_designs)}`; "
        "blocked full-matrix stays "
        f"`{_format_design_keys(policy_spec.blocked_full_matrix_designs)}`; "
        f"binding guard `{guard_probe.binding_guard}` stays on "
        f"`{_format_design_keys(guard_probe.binding_designs)}`",
        "- policy-candidate intake on the same contract surface: input mode "
        f"`{candidate_guard.candidate_input_mode}`; status "
        f"`{candidate_guard.candidate_status}`; resulting gate "
        f"`{candidate_guard.resulting_gate_status}`; binding guard "
        f"`{candidate_guard.binding_guard}`",
        f"- contract-audit replay path: `{contract_audit_replay_path}`",
    )
    scope_guard_status = "policy-contract-scope-only"
    implementation_handoff_contract = "bounded-right-center-execution-contract"
    validation_only_companion_bundle = "bounded-right-center-entry-patch-intake-bundle"
    validation_only_estimator_object_flow_contract = (
        fresh_estimator_evidence_contract.object_flow_driver_signature
    )
    validation_only_dense_replay_guard = (
        fresh_estimator_evidence_contract.dense_replay_driver_signature
    )
    validation_only_fresh_rerun_acceptance_contract = (
        fresh_estimator_evidence_contract.acceptance_driver_signature
    )
    validation_only_estimator_evidence_intake_contract = (
        fresh_estimator_evidence_contract.driver_signature
    )
    validation_only_estimator_evidence_target_gate_status = (
        fresh_estimator_evidence_contract.target_gate_status
    )
    validation_only_estimator_evidence_conditions = (
        fresh_estimator_evidence_contract.required_evidence_conditions
    )
    validation_only_estimator_evidence_digest = fresh_estimator_evidence_contract.canonical_fresh_estimator_evidence_intake_digest
    scope_guard_digest = (
        "- policy contract audit stays "
        f"`{scope_guard_status}`; live implementation handoff remains "
        f"`{implementation_handoff_contract}`",
        "- validation-only companion "
        f"`{validation_only_companion_bundle}` cannot replace the closeout main entry "
        f"or rewrite `{current_snapshot.recommended_bounded_loop}`",
        "- higher validation-only estimator-evidence companion "
        f"`{validation_only_estimator_evidence_intake_contract}` keeps "
        f"`{validation_only_estimator_object_flow_contract}`, "
        f"`{validation_only_dense_replay_guard}`, and "
        f"`{validation_only_fresh_rerun_acceptance_contract}` on the same "
        "policy-contract surface while Trigger 2 still stays pinned to "
        f"`{validation_only_estimator_evidence_target_gate_status}`",
    )
    return Phase7MonteCarloWideningPolicyContractAuditReport(
        stage_label="phase7-monte-carlo-widening-policy-contract-audit",
        repo_root=current_snapshot.repo_root,
        current_runtime_gate_status=current_snapshot.runtime_gate_status,
        accepted_runtime_gate_status=acceptance_preview.accepted_runtime_gate_status,
        current_recommended_bounded_loop=current_snapshot.recommended_bounded_loop,
        accepted_recommended_bounded_loop=acceptance_preview.accepted_recommended_bounded_loop,
        scope_guard_status=scope_guard_status,
        implementation_handoff_contract=implementation_handoff_contract,
        validation_only_companion_bundle=validation_only_companion_bundle,
        validation_only_estimator_object_flow_contract=(
            validation_only_estimator_object_flow_contract
        ),
        validation_only_dense_replay_guard=validation_only_dense_replay_guard,
        validation_only_fresh_rerun_acceptance_contract=(
            validation_only_fresh_rerun_acceptance_contract
        ),
        validation_only_estimator_evidence_intake_contract=(
            validation_only_estimator_evidence_intake_contract
        ),
        validation_only_estimator_evidence_target_gate_status=(
            validation_only_estimator_evidence_target_gate_status
        ),
        validation_only_estimator_evidence_conditions=(
            validation_only_estimator_evidence_conditions
        ),
        validation_only_estimator_evidence_digest=(
            validation_only_estimator_evidence_digest
        ),
        current_open_trigger_names=current_snapshot.open_trigger_names,
        accepted_open_trigger_names=acceptance_preview.accepted_open_trigger_names,
        empirical_blocker_reason=current_snapshot.empirical_blocker_reason,
        parity_gate_status=current_snapshot.parity_gate_status,
        policy_digest=policy_digest,
        binding_guard=guard_probe.binding_guard,
        binding_designs=guard_probe.binding_designs,
        contract_audit_replay_path=contract_audit_replay_path,
        candidate_input_mode=candidate_guard.candidate_input_mode,
        candidate_status=candidate_guard.candidate_status,
        candidate_resulting_gate_status=candidate_guard.resulting_gate_status,
        candidate_policy_digest=candidate_guard.policy_digest,
        candidate_rejection_reasons=candidate_guard.rejection_reasons,
        candidate_binding_guard=candidate_guard.binding_guard,
        stable_partial_designs=policy_spec.stable_partial_designs,
        quality_risk_designs=policy_spec.quality_risk_designs,
        blocked_full_matrix_designs=policy_spec.blocked_full_matrix_designs,
        canonical_contract_digest=canonical_contract_digest,
        scope_guard_digest=scope_guard_digest,
        candidate_guard_digest=candidate_guard.canonical_candidate_digest,
    )


def run_phase7_monte_carlo_widening_policy_contract_audit(
    repo_root: str | Path,
    *,
    policy: Phase7MonteCarloWideningPolicy | dict[str, object] | None = None,
) -> Phase7MonteCarloWideningPolicyContractAuditReport:
    root = _coerce_repo_root(repo_root)
    if policy is None:
        cache_epoch = _phase7_monte_carlo_widening_policy_contract_audit_cache_epoch(
            root
        )
        prev_epoch = _LAST_CONTRACT_AUDIT_CACHE_EPOCH_BY_ROOT.get(str(root))
        cache_info_before = (
            _run_phase7_monte_carlo_widening_policy_contract_audit_cached.cache_info()
        )
        report = _run_phase7_monte_carlo_widening_policy_contract_audit_cached(
            str(root), cache_epoch
        )
        cache_info_after = (
            _run_phase7_monte_carlo_widening_policy_contract_audit_cached.cache_info()
        )
        replay_path = "canonical-no-policy-lru-cache-miss"
        if cache_info_after.hits > cache_info_before.hits:
            replay_path = "canonical-no-policy-lru-cache-hit"
        elif (
            cache_info_before.currsize > 0
            and prev_epoch is not None
            and prev_epoch != cache_epoch
        ):
            replay_path = "canonical-no-policy-lru-cache-stale-rebuild"
        _LAST_CONTRACT_AUDIT_CACHE_EPOCH_BY_ROOT[str(root)] = cache_epoch
        return replace(report, contract_audit_replay_path=replay_path)
    return build_phase7_monte_carlo_widening_policy_contract_audit_report(
        _build_phase7_monte_carlo_widening_policy_contract_audit_closeout_snapshot(
            root
        ),
        run_phase7_monte_carlo_widening_policy_spec(),
        run_phase7_monte_carlo_widening_policy_acceptance_preview(root),
        run_phase7_monte_carlo_widening_policy_guard_probe(),
        run_phase7_monte_carlo_widening_policy_candidate_guard(policy=policy),
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract(),
        contract_audit_replay_path="explicit-policy-live-rebuild",
    )


@lru_cache(maxsize=4)
def _run_phase7_monte_carlo_widening_policy_contract_audit_cached(
    repo_root: str,
    cache_epoch: int,
) -> Phase7MonteCarloWideningPolicyContractAuditReport:
    root = _coerce_repo_root(repo_root)
    return build_phase7_monte_carlo_widening_policy_contract_audit_report(
        _build_phase7_monte_carlo_widening_policy_contract_audit_closeout_snapshot(
            root
        ),
        run_phase7_monte_carlo_widening_policy_spec(),
        run_phase7_monte_carlo_widening_policy_acceptance_preview(root),
        run_phase7_monte_carlo_widening_policy_guard_probe(),
        run_phase7_monte_carlo_widening_policy_candidate_guard(),
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract(),
    )
