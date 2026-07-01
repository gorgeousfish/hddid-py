from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_trim_floor_support_trace_index import (
    run_phase7_same_seed_seed303_trim_floor_support_trace_index,
)


@dataclass(slots=True)
class Phase7SameSeedSeed303AcceptanceExactTrimFloorStackIndexReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    live_routing: str
    validation_surface_status: str
    trim_floor_support_trace_index_driver_signature: str
    acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature: str
    acceptance_exact_trim_floor_inverse_pi_conduit_driver_signature: str
    acceptance_exact_trim_floor_landing_bridge_driver_signature: str
    acceptance_exact_trim_floor_runtime_bridge_driver_signature: str
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
    binding_replication_seed: int
    exact_trim_floor_value: float
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_share_of_fold3_low_pi_count: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    center_truth: float
    center_estimate: float
    center_error_to_half_interval_ratio: float
    vf_cross_entry: float
    trim_floor_support_trace_index_digest: tuple[str, ...]
    acceptance_exact_trim_floor_inverse_pi_concentration_digest: tuple[str, ...]
    acceptance_exact_trim_floor_inverse_pi_conduit_digest: tuple[str, ...]
    acceptance_exact_trim_floor_landing_bridge_digest: tuple[str, ...]
    acceptance_exact_trim_floor_runtime_bridge_digest: tuple[str, ...]
    canonical_seed303_acceptance_exact_trim_floor_stack_index_digest: tuple[str, ...]

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
        self.trim_floor_support_trace_index_driver_signature = str(
            self.trim_floor_support_trace_index_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature = str(
            self.acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_inverse_pi_conduit_driver_signature = str(
            self.acceptance_exact_trim_floor_inverse_pi_conduit_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_landing_bridge_driver_signature = str(
            self.acceptance_exact_trim_floor_landing_bridge_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_runtime_bridge_driver_signature = str(
            self.acceptance_exact_trim_floor_runtime_bridge_driver_signature
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
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.exact_trim_floor_value = float(self.exact_trim_floor_value)
        self.target_fold_id = int(self.target_fold_id)
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_share_of_fold3_low_pi_count = float(
            self.target_exact_trim_floor_share_of_fold3_low_pi_count
        )
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.target_exact_trim_floor_weighted_retention = float(
            self.target_exact_trim_floor_weighted_retention
        )
        self.center_truth = float(self.center_truth)
        self.center_estimate = float(self.center_estimate)
        self.center_error_to_half_interval_ratio = float(
            self.center_error_to_half_interval_ratio
        )
        self.vf_cross_entry = float(self.vf_cross_entry)
        self.trim_floor_support_trace_index_digest = tuple(
            str(item).rstrip() for item in self.trim_floor_support_trace_index_digest
        )
        self.acceptance_exact_trim_floor_inverse_pi_concentration_digest = tuple(
            str(item).rstrip()
            for item in self.acceptance_exact_trim_floor_inverse_pi_concentration_digest
        )
        self.acceptance_exact_trim_floor_inverse_pi_conduit_digest = tuple(
            str(item).rstrip()
            for item in self.acceptance_exact_trim_floor_inverse_pi_conduit_digest
        )
        self.acceptance_exact_trim_floor_landing_bridge_digest = tuple(
            str(item).rstrip()
            for item in self.acceptance_exact_trim_floor_landing_bridge_digest
        )
        self.acceptance_exact_trim_floor_runtime_bridge_digest = tuple(
            str(item).rstrip()
            for item in self.acceptance_exact_trim_floor_runtime_bridge_digest
        )
        self.canonical_seed303_acceptance_exact_trim_floor_stack_index_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_seed303_acceptance_exact_trim_floor_stack_index_digest
        )


def _rounded(value: float) -> float:
    return round(float(value), 3)



def build_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index_report() -> (
    Phase7SameSeedSeed303AcceptanceExactTrimFloorStackIndexReport
):
    trim_floor_support_trace_index_report = (
        run_phase7_same_seed_seed303_trim_floor_support_trace_index()
    )
    inverse_pi_concentration_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract()
    )
    inverse_pi_conduit_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract()
    )
    landing_bridge_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge()
    )
    runtime_bridge_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge()
    )

    canonical_digest = (
        "- `same-seed-seed303-fold3-trim-floor-inverse-pi-concentration-confirmed` localizes the actionable seed `303` packet onto one exact trim-floor row: `1/13 = 7.7%` of fold-`3` low-`pi_hat` treated rows already carry `43.1%` of the gross inverse-`pi_hat` conduit, `44.8%` of the weighted burden, and `99.0%` retention after propensity-floor weighting",
        "- `same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-open` plus `same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-open` keep the same denominator-led packet machine-readable through the retained inverse-`pi_hat` conduit before any residual seed `707` spend",
        "- `same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-open` and `same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open` still leave the live acceptance shortfall at `1/9 = 0.111`, with witness floor `7/9` against required `8/9`, while seed `303` / `z = 0.15` keeps the binding landing hop ahead of residual seed `707` / `z = 0.25`",
        "- the same exact-trim-floor stack keeps the acceptance-facing runtime readout `omega_f_hat[2,2] -> v_f_hat[2,1] -> bar_f_at_z0[1]` machine-readable at truth `1.162`, estimate `9.186`, ratio `1.194x`, cross-entry `-287.557`, and replication seed `883193502`",
        "- current Trigger 2 implication: keep `run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index()` validation-only as `validation-only acceptance exact trim-floor stack index`; it supports `trigger2-policy-spec` and `same-seed-admission-order`, but stays off live routing surfaces while the next bounded repair continues through the exact trim-floor runtime bridge before spending seed `707`",
    )

    return Phase7SameSeedSeed303AcceptanceExactTrimFloorStackIndexReport(
        stage_label="phase7-same-seed-seed303-acceptance-exact-trim-floor-stack-index",
        policy_digest=trim_floor_support_trace_index_report.policy_digest,
        binding_design=trim_floor_support_trace_index_report.binding_design,
        window_label=trim_floor_support_trace_index_report.window_label,
        focus_random_states=trim_floor_support_trace_index_report.focus_random_states,
        target_random_state=trim_floor_support_trace_index_report.target_random_state,
        target_seed_group=trim_floor_support_trace_index_report.target_seed_group,
        live_routing="trigger2-policy-spec",
        validation_surface_status="validation-only acceptance exact trim-floor stack index",
        trim_floor_support_trace_index_driver_signature=(
            trim_floor_support_trace_index_report.inverse_pi_concentration_driver_signature
        ),
        acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature=(
            inverse_pi_concentration_report.driver_signature
        ),
        acceptance_exact_trim_floor_inverse_pi_conduit_driver_signature=(
            inverse_pi_conduit_report.driver_signature
        ),
        acceptance_exact_trim_floor_landing_bridge_driver_signature=(
            landing_bridge_report.driver_signature
        ),
        acceptance_exact_trim_floor_runtime_bridge_driver_signature=(
            runtime_bridge_report.driver_signature
        ),
        same_seed_admission_order="same-seed-admission-order",
        acceptance_shortfall=1 / 9,
        current_witness_floor=7 / 9,
        required_min_witness_floor=8 / 9,
        binding_slot_random_state=runtime_bridge_report.binding_slot_random_state,
        binding_slot_seed_group=runtime_bridge_report.binding_slot_seed_group,
        binding_slot_z_value=runtime_bridge_report.binding_slot_z_value,
        residual_slot_random_state=runtime_bridge_report.residual_slot_random_state,
        residual_slot_seed_group="residual",
        residual_slot_z_value=runtime_bridge_report.residual_slot_z_value,
        binding_replication_seed=runtime_bridge_report.binding_replication_seed,
        exact_trim_floor_value=runtime_bridge_report.exact_trim_floor_value,
        target_fold_id=runtime_bridge_report.target_fold_id,
        target_exact_trim_floor_count=runtime_bridge_report.target_exact_trim_floor_count,
        target_exact_trim_floor_share_of_fold3_low_pi_count=(
            runtime_bridge_report.target_exact_trim_floor_share_of_fold3_low_pi_count
        ),
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=_rounded(
            runtime_bridge_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=_rounded(
            runtime_bridge_report.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=_rounded(
            runtime_bridge_report.target_exact_trim_floor_weighted_retention
        ),
        center_truth=_rounded(runtime_bridge_report.center_truth),
        center_estimate=_rounded(runtime_bridge_report.center_estimate),
        center_error_to_half_interval_ratio=_rounded(
            runtime_bridge_report.center_error_to_half_interval_ratio
        ),
        vf_cross_entry=_rounded(runtime_bridge_report.vf_cross_entry),
        trim_floor_support_trace_index_digest=(
            trim_floor_support_trace_index_report.canonical_seed303_trim_floor_support_trace_index_digest
        ),
        acceptance_exact_trim_floor_inverse_pi_concentration_digest=(
            inverse_pi_concentration_report.canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract_digest
        ),
        acceptance_exact_trim_floor_inverse_pi_conduit_digest=(
            inverse_pi_conduit_report.canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract_digest
        ),
        acceptance_exact_trim_floor_landing_bridge_digest=(
            landing_bridge_report.canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge_digest
        ),
        acceptance_exact_trim_floor_runtime_bridge_digest=(
            runtime_bridge_report.canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_digest
        ),
        canonical_seed303_acceptance_exact_trim_floor_stack_index_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index() -> (
    Phase7SameSeedSeed303AcceptanceExactTrimFloorStackIndexReport
):
    return build_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index_report()
