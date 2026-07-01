from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .automation_state_view import load_top_level_automation_state_block
from .monte_carlo_widening_policy_spec import (
    Phase7MonteCarloWideningPolicySpecReport,
    build_phase7_canonical_monte_carlo_widening_policy,
    build_phase7_monte_carlo_widening_policy_spec_report,
    decode_phase7_monte_carlo_widening_policy_digest,
    run_phase7_monte_carlo_widening_policy_spec,
)
from .monte_carlo_widening_trigger_gate import (
    _build_state_backed_trigger_gate_report,
    run_phase7_monte_carlo_widening_trigger_gate,
)
from .next_milestone_trigger_snapshot import (
    Phase7NextMilestoneTriggerSnapshotReport,
    build_phase7_next_milestone_trigger_snapshot_report,
)
from .outer_inference_trigger_gate import run_phase7_outer_inference_trigger_gate
from .section6_provenance import audit_section6_provenance_gate


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOMATION_STATE_PATH = _REPO_ROOT / "Docs" / "automation" / "automation-state.yaml"


def _load_feature_completion_state() -> dict[str, object]:
    return {
        "feature_completion_gate": dict(
            load_top_level_automation_state_block(
                _AUTOMATION_STATE_PATH,
                "feature_completion_gate",
            )
        )
    }


def _parse_design_keys(designs: object) -> tuple[tuple[str, int, int], ...]:
    parsed: list[tuple[str, int, int]] = []
    for item in designs if isinstance(designs, list | tuple) else ():
        dgp, n_obs, p = str(item).split("/")
        parsed.append((dgp, int(n_obs), int(p)))
    return tuple(parsed)


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


def _format_empirical_status(snapshot: Phase7NextMilestoneTriggerSnapshotReport) -> str:
    if snapshot.empirical_blocker_reason is None:
        return "empirical `ready`"
    return f"empirical `{snapshot.empirical_blocker_reason}`"


def _repo_side_live_route(gate_state: dict[str, object]) -> str:
    return str(
        gate_state.get(
            "route",
            gate_state.get(
                "frontier_packet_live_entry",
                "trigger2-policy-spec",
            ),
        )
    )


def _repo_side_empirical_blocker_reason(
    empirical_state: dict[str, object],
    repo_root: Path,
) -> str | None:
    empirical_status = str(empirical_state.get("status", "ready")).strip()
    if empirical_status in {"ready", "satisfied"}:
        return None
    if "blocker" in empirical_state:
        return str(empirical_state["blocker"])
    if "blocker_reason" in empirical_state:
        return str(empirical_state["blocker_reason"])
    if empirical_status == "live-derived":
        empirical_gate = audit_section6_provenance_gate(repo_root)
        if empirical_gate.status == "ready":
            return None
        return empirical_gate.blocker_reason or empirical_gate.status
    return empirical_status or None


def _repo_side_parity_gate_status(parity_state: dict[str, object]) -> str:
    for key in ("evidence", "gate_status", "status"):
        if key in parity_state:
            value = str(parity_state[key]).strip()
            if value not in {"ready", "satisfied", "live-derived"}:
                return value
    return run_phase7_outer_inference_trigger_gate().gate_status


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyAcceptancePreviewReport:
    stage_label: str
    repo_root: str
    live_route_label: str
    current_runtime_gate_status: str
    accepted_runtime_gate_status: str
    current_recommended_bounded_loop: str
    accepted_recommended_bounded_loop: str
    current_open_trigger_names: tuple[str, ...]
    accepted_open_trigger_names: tuple[str, ...]
    empirical_blocker_reason: str | None
    parity_gate_status: str
    policy_digest: tuple[str, ...]
    stable_partial_designs: tuple[tuple[str, int, int], ...]
    quality_risk_designs: tuple[tuple[str, int, int], ...]
    blocked_full_matrix_designs: tuple[tuple[str, int, int], ...]
    canonical_preview_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.repo_root = str(self.repo_root).strip()
        self.live_route_label = str(self.live_route_label).strip()
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
        self.canonical_preview_digest = tuple(
            str(line).rstrip() for line in self.canonical_preview_digest
        )

    def to_dict(self) -> dict[str, object]:
        canonical_policy = decode_phase7_monte_carlo_widening_policy_digest(
            self.policy_digest
        )
        return {
            "stage_label": self.stage_label,
            "repo_root": self.repo_root,
            "live_route_label": self.live_route_label,
            "current_runtime_gate_status": self.current_runtime_gate_status,
            "accepted_runtime_gate_status": self.accepted_runtime_gate_status,
            "current_recommended_bounded_loop": self.current_recommended_bounded_loop,
            "accepted_recommended_bounded_loop": self.accepted_recommended_bounded_loop,
            "current_recommended_feature_bundle": self.current_recommended_feature_bundle,
            "accepted_recommended_feature_bundle": self.accepted_recommended_feature_bundle,
            "current_open_trigger_names": list(self.current_open_trigger_names),
            "accepted_open_trigger_names": list(self.accepted_open_trigger_names),
            "empirical_blocker_reason": self.empirical_blocker_reason,
            "parity_gate_status": self.parity_gate_status,
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
            "stable_partial_designs": [
                list(item) for item in self.stable_partial_designs
            ],
            "quality_risk_designs": [list(item) for item in self.quality_risk_designs],
            "blocked_full_matrix_designs": [
                list(item) for item in self.blocked_full_matrix_designs
            ],
            "canonical_preview_digest": list(self.canonical_preview_digest),
        }

    @property
    def current_recommended_feature_bundle(self) -> str:
        return self.current_recommended_bounded_loop

    @property
    def accepted_recommended_feature_bundle(self) -> str:
        return self.accepted_recommended_bounded_loop


def build_phase7_monte_carlo_widening_policy_acceptance_preview_report(
    current_snapshot: Phase7NextMilestoneTriggerSnapshotReport,
    accepted_policy_snapshot: Phase7NextMilestoneTriggerSnapshotReport,
    policy_spec: Phase7MonteCarloWideningPolicySpecReport,
) -> Phase7MonteCarloWideningPolicyAcceptancePreviewReport:
    if current_snapshot.repo_root != accepted_policy_snapshot.repo_root:
        raise ValueError("policy acceptance preview requires a single repo root")
    if (
        current_snapshot.empirical_blocker_reason
        != accepted_policy_snapshot.empirical_blocker_reason
        or current_snapshot.parity_gate_status
        != accepted_policy_snapshot.parity_gate_status
    ):
        raise ValueError(
            "policy acceptance preview must not change empirical or parity blockers"
        )
    if current_snapshot.runtime_gate_status != policy_spec.current_gate_status:
        raise ValueError(
            "current snapshot drifted from canonical Trigger 2 policy spec"
        )
    if accepted_policy_snapshot.runtime_gate_status != policy_spec.target_gate_status:
        raise ValueError(
            "accepted preview drifted from canonical Trigger 2 target gate"
        )

    canonical_preview_digest = (
        "- current closeout snapshot: runtime gate "
        f"`{current_snapshot.runtime_gate_status}`; recommended feature bundle "
        f"`{current_snapshot.recommended_bounded_loop}`; live route "
        f"`{policy_spec.trigger_label}`; open triggers "
        f"{_format_trigger_names(current_snapshot.open_trigger_names)}",
        "- explicit policy acceptance preview: runtime gate "
        f"`{accepted_policy_snapshot.runtime_gate_status}`; recommended feature bundle "
        f"`{accepted_policy_snapshot.recommended_bounded_loop}`; open trigger "
        f"{_format_trigger_names(accepted_policy_snapshot.open_trigger_names)}",
        "- unchanged blockers outside Trigger 2: "
        f"{_format_empirical_status(current_snapshot)}; parity "
        f"`{current_snapshot.parity_gate_status}`",
        "- unchanged bounded widening slice after acceptance: stable "
        f"`{_format_design_keys(policy_spec.stable_partial_designs)}`; quality risk "
        f"`{_format_design_keys(policy_spec.quality_risk_designs)}`; blocked full-matrix "
        f"`{_format_design_keys(policy_spec.blocked_full_matrix_designs)}`",
        "- canonical policy digest: "
        + ", ".join(f"`{item}`" for item in policy_spec.policy_digest),
    )
    return Phase7MonteCarloWideningPolicyAcceptancePreviewReport(
        stage_label="phase7-monte-carlo-widening-policy-acceptance-preview",
        repo_root=current_snapshot.repo_root,
        live_route_label=policy_spec.trigger_label,
        current_runtime_gate_status=current_snapshot.runtime_gate_status,
        accepted_runtime_gate_status=accepted_policy_snapshot.runtime_gate_status,
        current_recommended_bounded_loop=current_snapshot.recommended_bounded_loop,
        accepted_recommended_bounded_loop=accepted_policy_snapshot.recommended_bounded_loop,
        current_open_trigger_names=current_snapshot.open_trigger_names,
        accepted_open_trigger_names=accepted_policy_snapshot.open_trigger_names,
        empirical_blocker_reason=current_snapshot.empirical_blocker_reason,
        parity_gate_status=current_snapshot.parity_gate_status,
        policy_digest=policy_spec.policy_digest,
        stable_partial_designs=policy_spec.stable_partial_designs,
        quality_risk_designs=policy_spec.quality_risk_designs,
        blocked_full_matrix_designs=policy_spec.blocked_full_matrix_designs,
        canonical_preview_digest=canonical_preview_digest,
    )


def build_phase7_monte_carlo_widening_policy_acceptance_preview_repo_side_report(
    repo_root: str | Path,
) -> Phase7MonteCarloWideningPolicyAcceptancePreviewReport:
    root = _coerce_repo_root(repo_root)
    state = _load_feature_completion_state()
    gate_state = state["feature_completion_gate"]["checks"][
        "monte_carlo_validation_ready"
    ]
    required_gate_keys = {
        "current_gate_status",
        "target_gate_status",
        "accepted_feature_bundle",
        "accepted_open_trigger_names",
        "policy_digest",
        "stable_partial_designs",
        "quality_risk_designs",
        "blocked_full_matrix_designs",
    }
    if not required_gate_keys.issubset(gate_state):
        try:
            return run_phase7_monte_carlo_widening_policy_acceptance_preview(root)
        except AssertionError:
            empirical_audit = audit_section6_provenance_gate(root)
            parity_gate = run_phase7_outer_inference_trigger_gate()
            current_gate = _build_state_backed_trigger_gate_report()
            canonical_policy = build_phase7_canonical_monte_carlo_widening_policy()
            accepted_gate = _build_state_backed_trigger_gate_report(canonical_policy)
            policy_spec = build_phase7_monte_carlo_widening_policy_spec_report(
                current_gate,
                accepted_gate,
                policy=canonical_policy,
            )
            return build_phase7_monte_carlo_widening_policy_acceptance_preview_report(
                build_phase7_next_milestone_trigger_snapshot_report(
                    empirical_audit,
                    current_gate,
                    parity_gate,
                ),
                build_phase7_next_milestone_trigger_snapshot_report(
                    empirical_audit,
                    accepted_gate,
                    parity_gate,
                ),
                policy_spec,
            )

    empirical_state = state["feature_completion_gate"]["checks"]["empirical_lane_ready"]
    parity_state = state["feature_completion_gate"]["checks"][
        "outer_inference_parity_ready"
    ]

    current_gate_status = str(gate_state["current_gate_status"])
    accepted_gate_status = str(gate_state["target_gate_status"])
    current_recommended_bounded_loop = _repo_side_live_route(gate_state)
    accepted_recommended_bounded_loop = str(gate_state["accepted_feature_bundle"])
    accepted_open_trigger_names = tuple(
        str(name) for name in gate_state["accepted_open_trigger_names"]
    )
    empirical_blocker_reason = _repo_side_empirical_blocker_reason(
        empirical_state,
        root,
    )
    parity_gate_status = _repo_side_parity_gate_status(parity_state)
    policy_digest = tuple(str(item) for item in gate_state["policy_digest"])
    stable_partial_designs = _parse_design_keys(gate_state["stable_partial_designs"])
    quality_risk_designs = _parse_design_keys(gate_state["quality_risk_designs"])
    blocked_full_matrix_designs = _parse_design_keys(
        gate_state["blocked_full_matrix_designs"]
    )
    canonical_preview_digest = (
        "- current closeout snapshot: runtime gate "
        f"`{current_gate_status}`; recommended feature bundle "
        f"`{current_recommended_bounded_loop}`; live route "
        f"`{current_recommended_bounded_loop}`; open triggers `none`",
        "- explicit policy acceptance preview: runtime gate "
        f"`{accepted_gate_status}`; recommended feature bundle "
        f"`{accepted_recommended_bounded_loop}`; open trigger "
        f"{_format_trigger_names(accepted_open_trigger_names)}",
        "- unchanged blockers outside Trigger 2: "
        f"{'empirical `ready`' if empirical_blocker_reason is None else f'empirical `{empirical_blocker_reason}`'}; "
        f"parity `{parity_gate_status}`",
        "- unchanged bounded widening slice after acceptance: stable "
        f"`{_format_design_keys(stable_partial_designs)}`; quality risk "
        f"`{_format_design_keys(quality_risk_designs)}`; blocked full-matrix "
        f"`{_format_design_keys(blocked_full_matrix_designs)}`",
        "- canonical policy digest: "
        + ", ".join(f"`{item}`" for item in policy_digest),
    )
    return Phase7MonteCarloWideningPolicyAcceptancePreviewReport(
        stage_label="phase7-monte-carlo-widening-policy-acceptance-preview",
        repo_root=root.as_posix(),
        live_route_label=current_recommended_bounded_loop,
        current_runtime_gate_status=current_gate_status,
        accepted_runtime_gate_status=accepted_gate_status,
        current_recommended_bounded_loop=current_recommended_bounded_loop,
        accepted_recommended_bounded_loop=accepted_recommended_bounded_loop,
        current_open_trigger_names=(),
        accepted_open_trigger_names=accepted_open_trigger_names,
        empirical_blocker_reason=empirical_blocker_reason,
        parity_gate_status=parity_gate_status,
        policy_digest=policy_digest,
        stable_partial_designs=stable_partial_designs,
        quality_risk_designs=quality_risk_designs,
        blocked_full_matrix_designs=blocked_full_matrix_designs,
        canonical_preview_digest=canonical_preview_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_acceptance_preview(
    repo_root: str | Path,
) -> Phase7MonteCarloWideningPolicyAcceptancePreviewReport:
    root = _coerce_repo_root(repo_root)
    empirical_audit = audit_section6_provenance_gate(root)
    parity_gate = run_phase7_outer_inference_trigger_gate()
    current_gate = run_phase7_monte_carlo_widening_trigger_gate()
    current_snapshot = build_phase7_next_milestone_trigger_snapshot_report(
        empirical_audit,
        current_gate,
        parity_gate,
    )
    canonical_policy = build_phase7_canonical_monte_carlo_widening_policy()
    accepted_policy_snapshot = build_phase7_next_milestone_trigger_snapshot_report(
        empirical_audit,
        run_phase7_monte_carlo_widening_trigger_gate(policy=canonical_policy),
        parity_gate,
    )
    return build_phase7_monte_carlo_widening_policy_acceptance_preview_report(
        current_snapshot,
        accepted_policy_snapshot,
        run_phase7_monte_carlo_widening_policy_spec(),
    )
