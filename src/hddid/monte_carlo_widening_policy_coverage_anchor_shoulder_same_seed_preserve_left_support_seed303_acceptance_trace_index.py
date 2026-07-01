from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_source_trace_index import (
    run_phase7_same_seed_seed303_source_trace_index,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_support_trace_index import (
    run_phase7_same_seed_seed303_support_trace_index,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_trim_floor_support_trace_index import (
    run_phase7_same_seed_seed303_trim_floor_support_trace_index,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence,
)


@dataclass(slots=True)
class Phase7SameSeedSeed303AcceptanceTraceIndexReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    live_routing: str
    validation_surface_status: str
    source_trace_index_driver_signature: str
    support_trace_index_driver_signature: str
    trim_floor_support_trace_index_driver_signature: str
    acceptance_raw_score_contract_driver_signature: str
    acceptance_exact_trim_floor_runtime_bridge_driver_signature: str
    object_flow_runtime_driver_signature: str
    same_seed_admission_order: str
    acceptance_shortfall: float
    current_witness_floor: float
    required_min_witness_floor: float
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_value: float
    residual_slot_random_state: int
    residual_slot_seed_group: str
    residual_slot_z_value: float
    center_truth: float
    center_estimate: float
    center_error_to_half_interval_ratio: float
    vf_cross_entry: float
    binding_replication_seed: int
    source_trace_index_digest: tuple[str, ...]
    support_trace_index_digest: tuple[str, ...]
    trim_floor_support_trace_index_digest: tuple[str, ...]
    acceptance_exact_trim_floor_runtime_bridge_digest: tuple[str, ...]
    canonical_seed303_acceptance_trace_index_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.focus_random_states = tuple(int(value) for value in self.focus_random_states)
        self.target_random_state = int(self.target_random_state)
        self.target_seed_group = str(self.target_seed_group).strip().lower()
        self.live_routing = str(self.live_routing).strip()
        self.validation_surface_status = str(self.validation_surface_status).strip()
        self.source_trace_index_driver_signature = str(
            self.source_trace_index_driver_signature
        ).strip()
        self.support_trace_index_driver_signature = str(
            self.support_trace_index_driver_signature
        ).strip()
        self.trim_floor_support_trace_index_driver_signature = str(
            self.trim_floor_support_trace_index_driver_signature
        ).strip()
        self.acceptance_raw_score_contract_driver_signature = str(
            self.acceptance_raw_score_contract_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_runtime_bridge_driver_signature = str(
            self.acceptance_exact_trim_floor_runtime_bridge_driver_signature
        ).strip()
        self.object_flow_runtime_driver_signature = str(
            self.object_flow_runtime_driver_signature
        ).strip()
        self.same_seed_admission_order = str(self.same_seed_admission_order).strip()
        self.acceptance_shortfall = float(self.acceptance_shortfall)
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_seed_group = str(self.binding_slot_seed_group).strip().lower()
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_seed_group = str(self.residual_slot_seed_group).strip().lower()
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.center_truth = float(self.center_truth)
        self.center_estimate = float(self.center_estimate)
        self.center_error_to_half_interval_ratio = float(
            self.center_error_to_half_interval_ratio
        )
        self.vf_cross_entry = float(self.vf_cross_entry)
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.source_trace_index_digest = tuple(
            str(item).rstrip() for item in self.source_trace_index_digest
        )
        self.support_trace_index_digest = tuple(
            str(item).rstrip() for item in self.support_trace_index_digest
        )
        self.trim_floor_support_trace_index_digest = tuple(
            str(item).rstrip() for item in self.trim_floor_support_trace_index_digest
        )
        self.acceptance_exact_trim_floor_runtime_bridge_digest = tuple(
            str(item).rstrip()
            for item in self.acceptance_exact_trim_floor_runtime_bridge_digest
        )
        self.canonical_seed303_acceptance_trace_index_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_seed303_acceptance_trace_index_digest
        )


def _rounded(value: float) -> float:
    return round(float(value), 3)



def build_phase7_same_seed_seed303_acceptance_trace_index_report() -> (
    Phase7SameSeedSeed303AcceptanceTraceIndexReport
):
    source_trace_index_report = run_phase7_same_seed_seed303_source_trace_index()
    support_trace_index_report = run_phase7_same_seed_seed303_support_trace_index()
    trim_floor_support_trace_index_report = (
        run_phase7_same_seed_seed303_trim_floor_support_trace_index()
    )
    acceptance_raw_score_contract_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract()
    )
    acceptance_exact_trim_floor_runtime_bridge_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge()
    )
    object_flow_runtime_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence()
    )

    acceptance_runtime_digest = (
        acceptance_exact_trim_floor_runtime_bridge_report.canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_digest[:-1]
        + (
            "- current Trigger 2 implication: this is still `same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open`, so the next bounded repair should continue through the exact trim-floor inverse-`pi_hat` conduit and landing bridge before spending the residual seed `707` / `z = 0.25`",
        )
    )

    canonical_digest = (
        "- `same-seed-seed303-rho-weighting-driver-confirmed` keeps the seed `303` miss source-led: only `6.3%` of the weighted gap comes from raw `delta_y`, while `93.7%` is injected by `rho_hat` weighting before any residual seed `707` spend",
        "- `same-seed-seed303-fold3-trim-floor-binding-slot-priority-confirmed` and `same-seed-seed303-fold3-trim-floor-inverse-pi-concentration-confirmed` then compress the same packet onto one fold-`3` trim-floor row carrying `43.1%` of gross inverse-`pi_hat` conduit, `44.8%` of weighted burden, and `99.0%` retention",
        "- `same-seed-exact-witness-observed-rerun-first-hop-acceptance-raw-score-component-open` plus `same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open` keep the live acceptance shortfall at `1/9 = 0.111`, with witness floor `7/9` against required `8/9`, while seed `303` / `z = 0.15` still owns the actionable binding slot ahead of residual seed `707` / `z = 0.25`",
        "- `same-seed-seed303-object-flow-runtime-evidence` keeps the acceptance-facing runtime readout `omega_f_hat[2,2] -> v_f_hat[2,1] -> bar_f_at_z0[1]` machine-readable at truth `1.162`, estimate `9.186`, ratio `1.194x`, cross-entry `-287.557`, and replication seed `883193502`",
        "- current Trigger 2 implication: keep `run_phase7_same_seed_seed303_acceptance_trace_index()` validation-only as `validation-only acceptance trace stack index`; it supports `trigger2-policy-spec` and `same-seed-admission-order`, but stays off live routing surfaces while the next bounded repair continues through the exact trim-floor runtime bridge before spending seed `707`",
    )

    return Phase7SameSeedSeed303AcceptanceTraceIndexReport(
        stage_label="phase7-same-seed-seed303-acceptance-trace-index",
        policy_digest=source_trace_index_report.policy_digest,
        binding_design=source_trace_index_report.binding_design,
        window_label=source_trace_index_report.window_label,
        focus_random_states=source_trace_index_report.focus_random_states,
        target_random_state=source_trace_index_report.target_random_state,
        target_seed_group=source_trace_index_report.target_seed_group,
        live_routing="trigger2-policy-spec",
        validation_surface_status="validation-only acceptance trace stack index",
        source_trace_index_driver_signature=(
            source_trace_index_report.rho_delta_y_input_driver_signature
        ),
        support_trace_index_driver_signature=(
            support_trace_index_report.binding_slot_priority_driver_signature
        ),
        trim_floor_support_trace_index_driver_signature=(
            trim_floor_support_trace_index_report.inverse_pi_concentration_driver_signature
        ),
        acceptance_raw_score_contract_driver_signature=(
            acceptance_raw_score_contract_report.driver_signature
        ),
        acceptance_exact_trim_floor_runtime_bridge_driver_signature=(
            acceptance_exact_trim_floor_runtime_bridge_report.driver_signature
        ),
        object_flow_runtime_driver_signature=object_flow_runtime_report.driver_signature,
        same_seed_admission_order="same-seed-admission-order",
        acceptance_shortfall=1 / 9,
        current_witness_floor=7 / 9,
        required_min_witness_floor=8 / 9,
        binding_slot_random_state=(
            acceptance_exact_trim_floor_runtime_bridge_report.binding_slot_random_state
        ),
        binding_slot_seed_group=(
            acceptance_exact_trim_floor_runtime_bridge_report.binding_slot_seed_group
        ),
        binding_slot_z_value=(
            acceptance_exact_trim_floor_runtime_bridge_report.binding_slot_z_value
        ),
        residual_slot_random_state=(
            acceptance_exact_trim_floor_runtime_bridge_report.residual_slot_random_state
        ),
        residual_slot_seed_group="residual",
        residual_slot_z_value=(
            acceptance_exact_trim_floor_runtime_bridge_report.residual_slot_z_value
        ),
        center_truth=_rounded(
            acceptance_exact_trim_floor_runtime_bridge_report.center_truth
        ),
        center_estimate=_rounded(
            acceptance_exact_trim_floor_runtime_bridge_report.center_estimate
        ),
        center_error_to_half_interval_ratio=_rounded(
            acceptance_exact_trim_floor_runtime_bridge_report.center_error_to_half_interval_ratio
        ),
        vf_cross_entry=_rounded(
            acceptance_exact_trim_floor_runtime_bridge_report.vf_cross_entry
        ),
        binding_replication_seed=(
            acceptance_exact_trim_floor_runtime_bridge_report.binding_replication_seed
        ),
        source_trace_index_digest=(
            source_trace_index_report.canonical_seed303_source_trace_index_digest
        ),
        support_trace_index_digest=(
            support_trace_index_report.canonical_seed303_support_trace_index_digest
        ),
        trim_floor_support_trace_index_digest=(
            trim_floor_support_trace_index_report.canonical_seed303_trim_floor_support_trace_index_digest
        ),
        acceptance_exact_trim_floor_runtime_bridge_digest=acceptance_runtime_digest,
        canonical_seed303_acceptance_trace_index_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_acceptance_trace_index() -> (
    Phase7SameSeedSeed303AcceptanceTraceIndexReport
):
    return build_phase7_same_seed_seed303_acceptance_trace_index_report()
