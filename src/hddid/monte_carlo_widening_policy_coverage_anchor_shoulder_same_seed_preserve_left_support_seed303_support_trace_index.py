from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace,
)


@dataclass(slots=True)
class Phase7SameSeedSeed303SupportTraceIndexReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    live_routing: str
    validation_surface_status: str
    rho_support_split_driver_signature: str
    low_pi_treated_support_driver_signature: str
    binding_slot_priority_driver_signature: str
    rho_support_split_trace_digest: tuple[str, ...]
    low_pi_treated_support_trace_digest: tuple[str, ...]
    binding_slot_priority_trace_digest: tuple[str, ...]
    canonical_seed303_support_trace_index_digest: tuple[str, ...]

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
        self.rho_support_split_driver_signature = str(
            self.rho_support_split_driver_signature
        ).strip()
        self.low_pi_treated_support_driver_signature = str(
            self.low_pi_treated_support_driver_signature
        ).strip()
        self.binding_slot_priority_driver_signature = str(
            self.binding_slot_priority_driver_signature
        ).strip()
        self.rho_support_split_trace_digest = tuple(
            str(item).rstrip() for item in self.rho_support_split_trace_digest
        )
        self.low_pi_treated_support_trace_digest = tuple(
            str(item).rstrip() for item in self.low_pi_treated_support_trace_digest
        )
        self.binding_slot_priority_trace_digest = tuple(
            str(item).rstrip() for item in self.binding_slot_priority_trace_digest
        )
        self.canonical_seed303_support_trace_index_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_seed303_support_trace_index_digest
        )


def build_phase7_same_seed_seed303_support_trace_index_report() -> (
    Phase7SameSeedSeed303SupportTraceIndexReport
):
    rho_support_split_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace()
    low_pi_treated_support_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace()
    binding_slot_priority_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace()

    canonical_digest = (
        "- `same-seed-seed303-treated-inverse-pi-support-driver-confirmed` keeps seed `303` / witness / `z = 0.15` at treated inverse-`pi_hat` center `9.005` versus control `1.843`, so treated support already owns `83.0%` of the weighted center projection and `82.3%` of the positive support lift",
        "- `same-seed-seed303-low-pi-treated-support-driver-confirmed` then narrows the same packet to the treated `pi_hat <= 0.1` tail: `25/250 = 10.0%` of treated-valid rows already supply `7.007` of the weighted treated center, `77.8%` of the treated support, and `96.4%` of the positive support lift",
        "- `same-seed-seed303-fold3-trim-floor-binding-slot-priority-confirmed` closes the support ladder onto one fold-`3` trim-floor row carrying `43.1%` of gross inverse-`pi_hat` conduit, `44.8%` of weighted burden, and `99.0%` retention, while keeping seed `303` / `z = 0.15` ahead of residual seed `707` / `z = 0.25` by `1/9 = 0.111` versus `0.000` witness-floor lift",
        "- current Trigger 2 implication: keep `run_phase7_same_seed_seed303_support_trace_index()` validation-only as `validation-only support-trace helper bundle`; it supports `trigger2-policy-spec` and seed `303` before seed `707`, but stays off live routing surfaces",
    )

    return Phase7SameSeedSeed303SupportTraceIndexReport(
        stage_label="phase7-same-seed-seed303-support-trace-index",
        policy_digest=rho_support_split_report.policy_digest,
        binding_design=rho_support_split_report.binding_design,
        window_label=rho_support_split_report.window_label,
        focus_random_states=rho_support_split_report.focus_random_states,
        target_random_state=rho_support_split_report.target_random_state,
        target_seed_group=rho_support_split_report.target_seed_group,
        live_routing="trigger2-policy-spec",
        validation_surface_status="validation-only support-trace helper bundle",
        rho_support_split_driver_signature=rho_support_split_report.driver_signature,
        low_pi_treated_support_driver_signature=(
            low_pi_treated_support_report.driver_signature
        ),
        binding_slot_priority_driver_signature=(
            binding_slot_priority_report.driver_signature
        ),
        rho_support_split_trace_digest=(
            rho_support_split_report.canonical_seed303_rho_support_split_trace_digest
        ),
        low_pi_treated_support_trace_digest=(
            low_pi_treated_support_report.canonical_seed303_low_pi_treated_support_trace_digest
        ),
        binding_slot_priority_trace_digest=(
            binding_slot_priority_report.canonical_seed303_low_pi_fold3_trim_floor_binding_slot_priority_digest
        ),
        canonical_seed303_support_trace_index_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_support_trace_index() -> (
    Phase7SameSeedSeed303SupportTraceIndexReport
):
    return build_phase7_same_seed_seed303_support_trace_index_report()
