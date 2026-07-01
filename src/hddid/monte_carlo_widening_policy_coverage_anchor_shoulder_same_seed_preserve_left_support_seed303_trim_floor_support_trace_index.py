from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_treated_phi1_prediction_trace import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_treated_phi1_prediction_trace,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace,
)


@dataclass(slots=True)
class Phase7SameSeedSeed303TrimFloorSupportTraceIndexReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    live_routing: str
    validation_surface_status: str
    treated_phi1_prediction_driver_signature: str
    propensity_floor_driver_signature: str
    inverse_pi_concentration_driver_signature: str
    treated_phi1_prediction_trace_digest: tuple[str, ...]
    propensity_floor_trace_digest: tuple[str, ...]
    inverse_pi_concentration_trace_digest: tuple[str, ...]
    canonical_seed303_trim_floor_support_trace_index_digest: tuple[str, ...]

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
        self.treated_phi1_prediction_driver_signature = str(
            self.treated_phi1_prediction_driver_signature
        ).strip()
        self.propensity_floor_driver_signature = str(
            self.propensity_floor_driver_signature
        ).strip()
        self.inverse_pi_concentration_driver_signature = str(
            self.inverse_pi_concentration_driver_signature
        ).strip()
        self.treated_phi1_prediction_trace_digest = tuple(
            str(item).rstrip() for item in self.treated_phi1_prediction_trace_digest
        )
        self.propensity_floor_trace_digest = tuple(
            str(item).rstrip() for item in self.propensity_floor_trace_digest
        )
        self.inverse_pi_concentration_trace_digest = tuple(
            str(item).rstrip() for item in self.inverse_pi_concentration_trace_digest
        )
        self.canonical_seed303_trim_floor_support_trace_index_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_seed303_trim_floor_support_trace_index_digest
        )


def build_phase7_same_seed_seed303_trim_floor_support_trace_index_report() -> (
    Phase7SameSeedSeed303TrimFloorSupportTraceIndexReport
):
    treated_phi1_prediction_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_treated_phi1_prediction_trace()
    propensity_floor_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace()
    inverse_pi_concentration_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace()

    canonical_digest = (
        "- `same-seed-seed303-fold3-treated-phi1-prediction-driver-confirmed` keeps seed `303` / witness / fold `3` / `z = 0.15` at raw treated `phi1_hat` center `0.232`, but the same slice jumps to weighted burden `4.778` once the treated low-`pi_hat` propensity factor is applied, so the open miss is not raw-predictor blow-up by itself",
        "- `same-seed-seed303-fold3-trim-floor-propensity-floor-driver-confirmed` then narrows the packet to the exact trim-floor row `pi_hat = 0.010001`: `99.0%` of its gross inverse-`pi_hat` lift survives the final propensity-floor contraction, and that single row already contributes `44.8%` of the weighted fold-`3` nuisance burden",
        "- `same-seed-seed303-fold3-trim-floor-inverse-pi-concentration-confirmed` closes the drilldown onto a row-vs-rest split: one trim-floor row out of `13` low-`pi_hat` treated observations (`7.7%`) already carries `43.1%` of gross inverse-`pi_hat` conduit and does so with `7.380x` the rest-of-tail amplification",
        "- current Trigger 2 implication: keep `run_phase7_same_seed_seed303_trim_floor_support_trace_index()` validation-only as `validation-only trim-floor support-trace helper bundle`; it supports `trigger2-policy-spec` and seed `303` before seed `707`, but stays off live routing surfaces",
    )

    return Phase7SameSeedSeed303TrimFloorSupportTraceIndexReport(
        stage_label="phase7-same-seed-seed303-trim-floor-support-trace-index",
        policy_digest=treated_phi1_prediction_report.policy_digest,
        binding_design=treated_phi1_prediction_report.binding_design,
        window_label=treated_phi1_prediction_report.window_label,
        focus_random_states=treated_phi1_prediction_report.focus_random_states,
        target_random_state=treated_phi1_prediction_report.target_random_state,
        target_seed_group=treated_phi1_prediction_report.target_seed_group,
        live_routing="trigger2-policy-spec",
        validation_surface_status="validation-only trim-floor support-trace helper bundle",
        treated_phi1_prediction_driver_signature=treated_phi1_prediction_report.driver_signature,
        propensity_floor_driver_signature=propensity_floor_report.driver_signature,
        inverse_pi_concentration_driver_signature=inverse_pi_concentration_report.driver_signature,
        treated_phi1_prediction_trace_digest=(
            treated_phi1_prediction_report.canonical_seed303_low_pi_fold3_treated_phi1_prediction_trace_digest
        ),
        propensity_floor_trace_digest=(
            propensity_floor_report.canonical_seed303_low_pi_fold3_trim_floor_propensity_floor_trace_digest
        ),
        inverse_pi_concentration_trace_digest=(
            inverse_pi_concentration_report.canonical_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_digest
        ),
        canonical_seed303_trim_floor_support_trace_index_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_trim_floor_support_trace_index() -> (
    Phase7SameSeedSeed303TrimFloorSupportTraceIndexReport
):
    return build_phase7_same_seed_seed303_trim_floor_support_trace_index_report()
