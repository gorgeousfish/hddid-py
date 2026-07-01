from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorRuntimeBridgeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{float(value):.3f}x"


def _format_exact_trim_floor(value: float) -> str:
    return f"{float(value):.6f}"


def _ensure_shared_frontier(
    *,
    live_gap_packet_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport
    ),
    runtime_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorRuntimeBridgeReport
    ),
    landing_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport
    ),
) -> None:
    for report in (runtime_bridge_report, landing_guard_report):
        if report.policy_digest != live_gap_packet_report.policy_digest:
            raise ValueError("live-gap packet contract requires shared policy digest")
        if report.binding_design != live_gap_packet_report.binding_design:
            raise ValueError("live-gap packet contract requires shared binding design")
        if report.window_label != live_gap_packet_report.window_label:
            raise ValueError("live-gap packet contract requires shared window label")
        if (
            report.same_seed_random_states
            != live_gap_packet_report.same_seed_random_states
        ):
            raise ValueError(
                "live-gap packet contract requires shared same-seed ordering"
            )
        if report.runtime_witness_path != live_gap_packet_report.runtime_witness_path:
            raise ValueError(
                "live-gap packet contract requires shared runtime witness path"
            )

    if (
        runtime_bridge_report.acceptance_readout_path
        != live_gap_packet_report.acceptance_readout_path
    ):
        raise ValueError("live-gap packet contract requires shared acceptance readout")
    if (
        landing_guard_report.binding_slot_random_state,
        landing_guard_report.binding_slot_z_index,
    ) != (
        live_gap_packet_report.binding_slot_random_state,
        live_gap_packet_report.binding_slot_z_index,
    ):
        raise ValueError("live-gap packet contract requires the same binding slot")
    if (
        landing_guard_report.queued_residual_slot_random_state,
        landing_guard_report.queued_residual_slot_z_index,
    ) != (
        live_gap_packet_report.residual_slot_random_state,
        live_gap_packet_report.residual_slot_z_index,
    ):
        raise ValueError(
            "live-gap packet contract requires the same queued residual slot"
        )
    if (
        runtime_bridge_report.binding_slot_random_state,
        runtime_bridge_report.binding_slot_z_index,
    ) != (
        live_gap_packet_report.binding_slot_random_state,
        live_gap_packet_report.binding_slot_z_index,
    ):
        raise ValueError(
            "live-gap packet contract requires the same binding runtime slot"
        )
    if (
        runtime_bridge_report.residual_slot_random_state,
        runtime_bridge_report.residual_slot_z_index,
    ) != (
        live_gap_packet_report.residual_slot_random_state,
        live_gap_packet_report.residual_slot_z_index,
    ):
        raise ValueError(
            "live-gap packet contract requires the same residual runtime slot"
        )
    if (
        landing_guard_report.binding_slot_replication_seed
        != live_gap_packet_report.binding_replication_seed
        or runtime_bridge_report.binding_replication_seed
        != live_gap_packet_report.binding_replication_seed
    ):
        raise ValueError("live-gap packet contract requires the same replication seed")
    if (
        landing_guard_report.current_witness_floor
        != live_gap_packet_report.current_witness_floor
    ):
        raise ValueError("live-gap packet contract requires the same witness floor")
    if (
        landing_guard_report.required_min_witness_floor
        != live_gap_packet_report.required_min_witness_floor
    ):
        raise ValueError(
            "live-gap packet contract requires the same required witness floor"
        )
    if (
        landing_guard_report.next_required_slot_random_state,
        landing_guard_report.next_required_slot_z_index,
    ) != (
        live_gap_packet_report.binding_slot_random_state,
        live_gap_packet_report.binding_slot_z_index,
    ):
        raise ValueError(
            "live-gap packet contract requires the landing guard to keep seed303 next"
        )


def _driver_signature(
    *,
    live_gap_packet_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport
    ),
    runtime_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorRuntimeBridgeReport
    ),
    landing_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport
    ),
) -> str:
    if (
        live_gap_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-live-gap-packet-open"
        and runtime_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open"
        and landing_guard_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-landing-guard-open"
    ):
        return "same-seed-exact-witness-observed-rerun-live-gap-packet-contract-open"
    return "mixed-same-seed-exact-witness-observed-rerun-live-gap-packet-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    observed_rerun_live_gap_packet_driver_signature: str
    acceptance_exact_trim_floor_runtime_bridge_driver_signature: str
    first_hop_landing_guard_driver_signature: str
    observed_rerun_candidate_status: str
    observed_rerun_rung_status: str
    repair_agenda_driver_signature: str
    downstream_evidence_status: str
    current_witness_floor: float
    required_min_witness_floor: float
    acceptance_shortfall: float
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_index: int
    binding_slot_z_value: float
    residual_slot_random_state: int
    residual_slot_seed_group: str
    residual_slot_z_index: int
    residual_slot_z_value: float
    next_required_slot_random_state: int | None
    next_required_slot_seed_group: str | None
    next_required_slot_z_index: int | None
    next_required_slot_z_value: float | None
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    binding_replication_seed: int
    exact_trim_floor_value: float
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_fold3_low_pi_denominator: int
    target_exact_trim_floor_share_of_fold3_low_pi_count: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    target_remaining_fold3_low_pi_weighted_retention: float
    center_truth: float
    center_estimate: float
    center_pointwise_interval_lower: float
    center_lower_minus_truth: float
    center_error_to_half_interval_ratio: float
    vf_cross_entry: float
    driver_signature: str
    canonical_observed_rerun_live_gap_packet_contract_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.same_seed_random_states = tuple(
            int(value) for value in self.same_seed_random_states
        )
        self.observed_rerun_live_gap_packet_driver_signature = str(
            self.observed_rerun_live_gap_packet_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_runtime_bridge_driver_signature = str(
            self.acceptance_exact_trim_floor_runtime_bridge_driver_signature
        ).strip()
        self.first_hop_landing_guard_driver_signature = str(
            self.first_hop_landing_guard_driver_signature
        ).strip()
        self.observed_rerun_candidate_status = str(
            self.observed_rerun_candidate_status
        ).strip()
        self.observed_rerun_rung_status = str(self.observed_rerun_rung_status).strip()
        self.repair_agenda_driver_signature = str(
            self.repair_agenda_driver_signature
        ).strip()
        self.downstream_evidence_status = str(self.downstream_evidence_status).strip()
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.acceptance_shortfall = float(self.acceptance_shortfall)
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_seed_group = str(self.binding_slot_seed_group).strip().lower()
        self.binding_slot_z_index = int(self.binding_slot_z_index)
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_seed_group = (
            str(self.residual_slot_seed_group).strip().lower()
        )
        self.residual_slot_z_index = int(self.residual_slot_z_index)
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.next_required_slot_random_state = (
            None
            if self.next_required_slot_random_state is None
            else int(self.next_required_slot_random_state)
        )
        self.next_required_slot_seed_group = (
            None
            if self.next_required_slot_seed_group is None
            else str(self.next_required_slot_seed_group).strip().lower()
        )
        self.next_required_slot_z_index = (
            None
            if self.next_required_slot_z_index is None
            else int(self.next_required_slot_z_index)
        )
        self.next_required_slot_z_value = (
            None
            if self.next_required_slot_z_value is None
            else float(self.next_required_slot_z_value)
        )
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.exact_trim_floor_value = float(self.exact_trim_floor_value)
        self.target_fold_id = int(self.target_fold_id)
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_fold3_low_pi_denominator = int(self.target_fold3_low_pi_denominator)
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
        self.target_remaining_fold3_low_pi_weighted_retention = float(
            self.target_remaining_fold3_low_pi_weighted_retention
        )
        self.center_truth = float(self.center_truth)
        self.center_estimate = float(self.center_estimate)
        self.center_pointwise_interval_lower = float(
            self.center_pointwise_interval_lower
        )
        self.center_lower_minus_truth = float(self.center_lower_minus_truth)
        self.center_error_to_half_interval_ratio = float(
            self.center_error_to_half_interval_ratio
        )
        self.vf_cross_entry = float(self.vf_cross_entry)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_live_gap_packet_contract_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_live_gap_packet_contract_digest
        )


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketContractReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        != "same-seed-exact-witness-observed-rerun-live-gap-packet-contract-open"
    ):
        return (
            "- the observed-rerun live-gap packet contract is no longer in the canonical open state, so this helper currently only provides a mixed validation-only read",
        )

    return (
        "- the validation-only observed-rerun live-gap packet still keeps the current RED state on one paired read: "
        f"`{report.observed_rerun_live_gap_packet_driver_signature}`, "
        f"`{report.acceptance_exact_trim_floor_runtime_bridge_driver_signature}`, and "
        f"`{report.first_hop_landing_guard_driver_signature}` all continue to point at the same seed "
        f"`{report.binding_slot_random_state}` before seed `{report.residual_slot_random_state}` frontier",
        "- that paired contract remains below admissibility in the same way across the packet and the landing guard: "
        f"current candidate status stays `{report.observed_rerun_candidate_status}`, current rung stays "
        f"`{report.observed_rerun_rung_status}`, downstream evidence stays "
        f"`{report.downstream_evidence_status}`, witness floor stays "
        f"`{_format_ninths(report.current_witness_floor)}`, and required floor stays "
        f"`{_format_ninths(report.required_min_witness_floor)}` while the open acceptance shortfall remains "
        f"`{_format_ninths(report.acceptance_shortfall)}`",
        "- the same contract also keeps the acceptance-facing trim-floor runtime stack machine-readable on replication seed "
        f"`{report.binding_replication_seed}`: exact trim-floor row `pi_hat = {_format_exact_trim_floor(report.exact_trim_floor_value)}` "
        f"is still `1/{report.target_fold3_low_pi_denominator} = {_format_percent(report.target_exact_trim_floor_share_of_fold3_low_pi_count)}` "
        f"of fold-`{report.target_fold_id}` low-`pi_hat` treated rows while carrying "
        f"`{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the gross inverse-`pi_hat` conduit, "
        f"`{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, and "
        f"`{_format_percent(report.target_exact_trim_floor_weighted_retention)}` weighted retention versus "
        f"`{_format_percent(report.target_remaining_fold3_low_pi_weighted_retention)}` for the remaining rows; truth "
        f"`{_format_float(report.center_truth)}`, estimate `{_format_float(report.center_estimate)}`, lower bound "
        f"`{_format_float(report.center_pointwise_interval_lower)}`, lower-minus-truth "
        f"`{_format_float(report.center_lower_minus_truth)}`, error-to-half-width ratio "
        f"`{_format_ratio(report.center_error_to_half_interval_ratio)}`, and `v_f_hat[2,1] = {_format_float(report.vf_cross_entry)}` all stay on the same runtime packet",
        "- current Trigger 2 implication: "
        f"`{report.driver_signature}`; keep live entry at `trigger2-policy-spec`, preserve the validation-only live-gap packet contract, and feed the real same-seed observed rerun through seed "
        f"`{report.binding_slot_random_state}` / `{report.binding_slot_seed_group}` / `z = {_format_float(report.binding_slot_z_value)}` before spending the queued residual seed "
        f"`{report.residual_slot_random_state}` / `{report.residual_slot_seed_group}` / `z = {_format_float(report.residual_slot_z_value)}`",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet_contract_report(
    *,
    live_gap_packet_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport
        | None
    ) = None,
    runtime_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorRuntimeBridgeReport
        | None
    ) = None,
    landing_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketContractReport:
    if live_gap_packet_report is None:
        live_gap_packet_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet()
    if runtime_bridge_report is None:
        runtime_bridge_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge()
    if landing_guard_report is None:
        landing_guard_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard()

    _ensure_shared_frontier(
        live_gap_packet_report=live_gap_packet_report,
        runtime_bridge_report=runtime_bridge_report,
        landing_guard_report=landing_guard_report,
    )
    driver_signature = _driver_signature(
        live_gap_packet_report=live_gap_packet_report,
        runtime_bridge_report=runtime_bridge_report,
        landing_guard_report=landing_guard_report,
    )
    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-"
            "preserve-left-support-exact-witness-observed-rerun-live-gap-packet-contract"
        ),
        policy_digest=live_gap_packet_report.policy_digest,
        binding_design=live_gap_packet_report.binding_design,
        window_label=live_gap_packet_report.window_label,
        same_seed_random_states=live_gap_packet_report.same_seed_random_states,
        observed_rerun_live_gap_packet_driver_signature=live_gap_packet_report.driver_signature,
        acceptance_exact_trim_floor_runtime_bridge_driver_signature=runtime_bridge_report.driver_signature,
        first_hop_landing_guard_driver_signature=landing_guard_report.driver_signature,
        observed_rerun_candidate_status=live_gap_packet_report.observed_rerun_candidate_status,
        observed_rerun_rung_status=live_gap_packet_report.observed_rerun_rung_status,
        repair_agenda_driver_signature=landing_guard_report.repair_agenda_driver_signature,
        downstream_evidence_status=live_gap_packet_report.downstream_evidence_status,
        current_witness_floor=live_gap_packet_report.current_witness_floor,
        required_min_witness_floor=live_gap_packet_report.required_min_witness_floor,
        acceptance_shortfall=runtime_bridge_report.acceptance_shortfall,
        binding_slot_random_state=live_gap_packet_report.binding_slot_random_state,
        binding_slot_seed_group=live_gap_packet_report.binding_slot_seed_group,
        binding_slot_z_index=live_gap_packet_report.binding_slot_z_index,
        binding_slot_z_value=live_gap_packet_report.binding_slot_z_value,
        residual_slot_random_state=live_gap_packet_report.residual_slot_random_state,
        residual_slot_seed_group=live_gap_packet_report.residual_slot_seed_group,
        residual_slot_z_index=live_gap_packet_report.residual_slot_z_index,
        residual_slot_z_value=live_gap_packet_report.residual_slot_z_value,
        next_required_slot_random_state=landing_guard_report.next_required_slot_random_state,
        next_required_slot_seed_group=landing_guard_report.next_required_slot_seed_group,
        next_required_slot_z_index=landing_guard_report.next_required_slot_z_index,
        next_required_slot_z_value=landing_guard_report.next_required_slot_z_value,
        runtime_witness_path=live_gap_packet_report.runtime_witness_path,
        acceptance_readout_path=live_gap_packet_report.acceptance_readout_path,
        binding_replication_seed=runtime_bridge_report.binding_replication_seed,
        exact_trim_floor_value=runtime_bridge_report.exact_trim_floor_value,
        target_fold_id=runtime_bridge_report.target_fold_id,
        target_exact_trim_floor_count=runtime_bridge_report.target_exact_trim_floor_count,
        target_fold3_low_pi_denominator=13,
        target_exact_trim_floor_share_of_fold3_low_pi_count=(
            runtime_bridge_report.target_exact_trim_floor_share_of_fold3_low_pi_count
        ),
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            runtime_bridge_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            runtime_bridge_report.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            runtime_bridge_report.target_exact_trim_floor_weighted_retention
        ),
        target_remaining_fold3_low_pi_weighted_retention=(
            runtime_bridge_report.target_remaining_fold3_low_pi_weighted_retention
        ),
        center_truth=runtime_bridge_report.center_truth,
        center_estimate=runtime_bridge_report.center_estimate,
        center_pointwise_interval_lower=runtime_bridge_report.center_pointwise_interval_lower,
        center_lower_minus_truth=runtime_bridge_report.center_lower_minus_truth,
        center_error_to_half_interval_ratio=(
            runtime_bridge_report.center_error_to_half_interval_ratio
        ),
        vf_cross_entry=runtime_bridge_report.vf_cross_entry,
        driver_signature=driver_signature,
        canonical_observed_rerun_live_gap_packet_contract_digest=(),
    )
    object.__setattr__(
        report,
        "canonical_observed_rerun_live_gap_packet_contract_digest",
        _canonical_digest(report=report),
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet_contract_report()
