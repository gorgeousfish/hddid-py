from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_delta_y_input_trace import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_delta_y_input_trace,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_score_input_trace import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_score_input_trace,
)


@dataclass(slots=True)
class Phase7SameSeedSeed303SourceTraceIndexReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    live_routing: str
    validation_surface_status: str
    first_hop_acceptance_driver_signature: str
    score_input_driver_signature: str
    raw_score_component_driver_signature: str
    rho_delta_y_input_driver_signature: str
    score_input_trace_digest: tuple[str, ...]
    raw_score_component_trace_digest: tuple[str, ...]
    rho_delta_y_input_trace_digest: tuple[str, ...]
    canonical_seed303_source_trace_index_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.focus_random_states = tuple(
            int(value) for value in self.focus_random_states
        )
        self.target_random_state = int(self.target_random_state)
        self.target_seed_group = str(self.target_seed_group).strip().lower()
        self.live_routing = str(self.live_routing).strip()
        self.validation_surface_status = str(self.validation_surface_status).strip()
        self.first_hop_acceptance_driver_signature = str(
            self.first_hop_acceptance_driver_signature
        ).strip()
        self.score_input_driver_signature = str(
            self.score_input_driver_signature
        ).strip()
        self.raw_score_component_driver_signature = str(
            self.raw_score_component_driver_signature
        ).strip()
        self.rho_delta_y_input_driver_signature = str(
            self.rho_delta_y_input_driver_signature
        ).strip()
        self.score_input_trace_digest = tuple(
            str(item).rstrip() for item in self.score_input_trace_digest
        )
        self.raw_score_component_trace_digest = tuple(
            str(item).rstrip() for item in self.raw_score_component_trace_digest
        )
        self.rho_delta_y_input_trace_digest = tuple(
            str(item).rstrip() for item in self.rho_delta_y_input_trace_digest
        )
        self.canonical_seed303_source_trace_index_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_seed303_source_trace_index_digest
        )


def build_phase7_same_seed_seed303_source_trace_index_report() -> (
    Phase7SameSeedSeed303SourceTraceIndexReport
):
    score_input_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_score_input_trace()
    raw_score_component_trace_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace()
    rho_delta_y_input_trace_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_delta_y_input_trace()
    first_hop_acceptance_contract_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract()

    canonical_digest = (
        "- `same-seed-seed303-eq31-score-input-trace-confirmed` already pins seed `303` / witness / `z = 0.15` at raw score `7.669` and Eq. (3.1) center `9.186`, so the open first-hop acceptance miss is score-led before any deeper nuisance split",
        "- `same-seed-seed303-raw-score-delta-y-driver-confirmed` then keeps `rho_hat * delta_y = 10.848`, `rho_hat * (1-pi_hat) * phi1_hat = 3.941`, and `rho_hat * pi_hat * phi0_hat = -0.762` on the same witness packet, showing the raw-score miss remains delta-y-led even before residual seed `707` is spent",
        "- `same-seed-seed303-rho-weighting-driver-confirmed` narrows the packet further: only `6.3%` of the weighted gap comes from raw `delta_y`, while `93.7%` is injected by `rho_hat` weighting, so the next bounded repair should trace `rho_hat / pi_hat` support rather than reopen band widening prose",
        "- current Trigger 2 implication: keep `run_phase7_same_seed_seed303_source_trace_index()` validation-only as `validation-only score-input helper quarantine`; it supports `trigger2-policy-spec` and seed `303` before seed `707`, but stays off live routing surfaces",
    )

    return Phase7SameSeedSeed303SourceTraceIndexReport(
        stage_label="phase7-same-seed-seed303-source-trace-index",
        policy_digest=score_input_report.policy_digest,
        binding_design=score_input_report.binding_design,
        window_label=score_input_report.window_label,
        focus_random_states=score_input_report.focus_random_states,
        target_random_state=score_input_report.target_random_state,
        target_seed_group=score_input_report.target_seed_group,
        live_routing="trigger2-policy-spec",
        validation_surface_status="validation-only score-input helper quarantine",
        first_hop_acceptance_driver_signature=first_hop_acceptance_contract_report.driver_signature,
        score_input_driver_signature=score_input_report.driver_signature,
        raw_score_component_driver_signature=raw_score_component_trace_report.driver_signature,
        rho_delta_y_input_driver_signature=rho_delta_y_input_trace_report.driver_signature,
        score_input_trace_digest=score_input_report.canonical_seed303_score_input_trace_digest,
        raw_score_component_trace_digest=(
            raw_score_component_trace_report.canonical_seed303_raw_score_component_trace_digest
        ),
        rho_delta_y_input_trace_digest=(
            rho_delta_y_input_trace_report.canonical_seed303_rho_delta_y_input_trace_digest
        ),
        canonical_seed303_source_trace_index_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_source_trace_index() -> (
    Phase7SameSeedSeed303SourceTraceIndexReport
):
    return build_phase7_same_seed_seed303_source_trace_index_report()
