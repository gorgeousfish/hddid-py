from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOMATION_STATE_PATH = _REPO_ROOT / "Docs" / "automation" / "automation-state.yaml"
_WINDOW_LABEL = "near_zero_grid"
_ARCHIVED_TARGET_OMEGA_DIAGONAL_ENTRY = 410.474
_SOURCE_DIAGONAL_COORDINATE = 2
_SOURCE_DIAGONAL_BASIS_LABEL = "sin(2πz)"
_SHARED_VF_ENTRY_LABEL = "v_f_hat[2,2]"
_RUNTIME_WITNESS_PATH = (
    "omega_f_hat[2,2]",
    "v_f_hat[2,2]",
    "covariance(0.25, 0.15)",
)
_PRERUN_CONTRACT_STEPS = (
    "reject-archived-positive-first-sine-ceiling",
    "bind-regrounded-live-target-before-rerun",
    "preserve-bounded-right-center-runtime-path",
    "spend-fresh-reruns-only-after-reground-contract",
)


def _format_design_key(binding_design: tuple[str, int, int]) -> str:
    return f"{binding_design[0]}/{binding_design[1]}/{binding_design[2]}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _parse_design_key(design_label: str) -> tuple[str, int, int]:
    dgp, n_obs, p = str(design_label).split("/")
    return (dgp, int(n_obs), int(p))


def _load_trigger2_gate_state() -> dict[str, object]:
    state = yaml.safe_load(_AUTOMATION_STATE_PATH.read_text(encoding="utf-8"))
    return state["feature_completion_gate"]["checks"]["monte_carlo_validation_ready"]


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetRegroundContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    coverage_anchor_replication_seed: int
    overshoot_companion_replication_seed: int
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    shared_vf_entry_label: str
    archived_target_omega_diagonal_entry: float
    regrounded_target_omega_diagonal_entry: float
    canonical_vs_live_target_gap: float
    runtime_witness_path: tuple[str, ...]
    prerun_contract_steps: tuple[str, ...]
    contract_holds: bool
    driver_signature: str
    canonical_reground_contract_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.coverage_anchor_replication_seed = int(self.coverage_anchor_replication_seed)
        self.overshoot_companion_replication_seed = int(
            self.overshoot_companion_replication_seed
        )
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.archived_target_omega_diagonal_entry = float(
            self.archived_target_omega_diagonal_entry
        )
        self.regrounded_target_omega_diagonal_entry = float(
            self.regrounded_target_omega_diagonal_entry
        )
        self.canonical_vs_live_target_gap = float(self.canonical_vs_live_target_gap)
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.prerun_contract_steps = tuple(
            str(item).strip() for item in self.prerun_contract_steps
        )
        self.contract_holds = bool(self.contract_holds)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_reground_contract_digest = tuple(
            str(line).rstrip() for line in self.canonical_reground_contract_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "coverage_anchor_replication_seed": self.coverage_anchor_replication_seed,
            "overshoot_companion_replication_seed": (
                self.overshoot_companion_replication_seed
            ),
            "source_diagonal_coordinate": self.source_diagonal_coordinate,
            "source_diagonal_basis_label": self.source_diagonal_basis_label,
            "shared_vf_entry_label": self.shared_vf_entry_label,
            "archived_target_omega_diagonal_entry": (
                self.archived_target_omega_diagonal_entry
            ),
            "regrounded_target_omega_diagonal_entry": (
                self.regrounded_target_omega_diagonal_entry
            ),
            "canonical_vs_live_target_gap": self.canonical_vs_live_target_gap,
            "runtime_witness_path": list(self.runtime_witness_path),
            "prerun_contract_steps": list(self.prerun_contract_steps),
            "contract_holds": self.contract_holds,
            "driver_signature": self.driver_signature,
            "canonical_reground_contract_digest": list(
                self.canonical_reground_contract_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_reground_contract_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetRegroundContractReport
):
    gate_state = _load_trigger2_gate_state()
    live_alignment_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe()
    )
    binding_design = live_alignment_report.binding_design
    binding_random_states = (
        live_alignment_report.coverage_anchor_random_state,
        live_alignment_report.overshoot_companion_random_state,
    )
    binding_replication_seeds = (
        live_alignment_report.coverage_anchor_replication_seed,
        live_alignment_report.overshoot_companion_replication_seed,
    )
    target_gap = float(live_alignment_report.canonical_vs_live_target_gap)
    regrounded_target = _ARCHIVED_TARGET_OMEGA_DIAGONAL_ENTRY + target_gap
    alignment_driver = live_alignment_report.driver_signature
    execution_driver = str(gate_state["execution_contract_signature"])
    runtime_driver = str(gate_state["runtime_evidence_driver"])
    contract_holds = bool(
        alignment_driver == "live-source-target-alignment-drift"
        and execution_driver == "bounded-right-center-execution-contract"
        and runtime_driver == "trigger2-runtime-evidence-packet-open"
        and binding_design == ("DGP2", 500, 50)
        and binding_random_states == (202, 505)
        and target_gap > 0.0
    )
    driver_signature = (
        "live-source-target-reground-contract"
        if contract_holds
        else "mixed-live-source-target-reground-contract"
    )
    binding_design_key = _format_design_key(binding_design)
    canonical_digest = (
        f"- live source-target regrounding is a pre-rerun contract on `{binding_design_key}` / `{_WINDOW_LABEL}`: the inherited alignment driver `{alignment_driver}` keeps the archived positive first-sine ceiling `{_format_float(_ARCHIVED_TARGET_OMEGA_DIAGONAL_ENTRY)}` out of the fresh rerun path",
        f"- the regrounded target for seed `202` with companion seed `505` is `{_format_float(regrounded_target)}`, a `{_format_signed_float(target_gap)}` live-target gap over the archived packet, on `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`",
        "- current Trigger 2 implication: `live-source-target-reground-contract`; no fresh rerun is spent by this helper, and seed-202 evidence remains admissible only after the bounded right-center runtime path is bound to the regrounded live target",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetRegroundContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-live-source-target-reground-contract"
        ),
        policy_digest=live_alignment_report.policy_digest,
        binding_design=binding_design,
        window_label=_WINDOW_LABEL,
        coverage_anchor_random_state=binding_random_states[0],
        overshoot_companion_random_state=binding_random_states[1],
        coverage_anchor_replication_seed=binding_replication_seeds[0],
        overshoot_companion_replication_seed=binding_replication_seeds[1],
        source_diagonal_coordinate=_SOURCE_DIAGONAL_COORDINATE,
        source_diagonal_basis_label=_SOURCE_DIAGONAL_BASIS_LABEL,
        shared_vf_entry_label=_SHARED_VF_ENTRY_LABEL,
        archived_target_omega_diagonal_entry=_ARCHIVED_TARGET_OMEGA_DIAGONAL_ENTRY,
        regrounded_target_omega_diagonal_entry=regrounded_target,
        canonical_vs_live_target_gap=target_gap,
        runtime_witness_path=_RUNTIME_WITNESS_PATH,
        prerun_contract_steps=_PRERUN_CONTRACT_STEPS,
        contract_holds=contract_holds,
        driver_signature=driver_signature,
        canonical_reground_contract_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_reground_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetRegroundContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_reground_contract_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetRegroundContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_reground_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_reground_contract",
]
