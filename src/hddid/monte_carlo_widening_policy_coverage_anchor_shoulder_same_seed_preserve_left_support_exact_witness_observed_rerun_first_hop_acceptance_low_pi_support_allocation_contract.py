from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceDeltaYSupportChainContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold_trim_floor_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFoldTrimFloorTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold_trim_floor_trace,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _format_exact_trim_floor(value: float) -> str:
    return f"{float(value):.6f}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceLowPiSupportAllocationContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    acceptance_delta_y_support_chain_driver_signature: str
    low_pi_fold_trim_floor_driver_signature: str
    low_pi_fold3_trim_floor_propensity_floor_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    acceptance_shortfall: float
    current_witness_floor: float
    required_min_witness_floor: float
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_index: int
    binding_slot_z_value: float
    residual_slot_random_state: int | None
    residual_slot_seed_group: str | None
    residual_slot_z_index: int | None
    residual_slot_z_value: float | None
    binding_replication_seed: int
    low_pi_threshold: float
    trim_lower: float
    target_treated_count: int
    target_low_pi_treated_count: int
    target_low_pi_phi1_center_projection: float
    fold_ids: tuple[int, ...]
    fold_low_pi_treated_counts: tuple[int, ...]
    fold_low_pi_share_of_treated_count: tuple[float, ...]
    fold_low_pi_treated_delta_over_pi_center_projections: tuple[float, ...]
    fold_low_pi_share_of_weighted_center: tuple[float, ...]
    fold_low_pi_phi1_center_projections: tuple[float, ...]
    fold_low_pi_share_of_phi1_projection: tuple[float, ...]
    dominant_fold_id: int
    dominant_fold_low_pi_share_of_weighted_center: float
    exact_trim_floor_value: float
    exact_trim_floor_count: int
    exact_trim_floor_fold_id: int
    exact_trim_floor_share_of_low_pi_count: float
    exact_trim_floor_delta_over_pi_center_projection: float
    exact_trim_floor_share_of_low_pi_weighted_center: float
    exact_trim_floor_phi1_center_projection: float
    exact_trim_floor_share_of_low_pi_phi1_projection: float
    target_fold3_raw_phi1_center_projection: float
    target_fold3_inverse_pi_phi1_center_projection: float
    target_fold3_weighted_phi1_center_projection: float
    target_fold3_raw_phi1_offset_share_of_inverse_pi: float
    target_fold3_retained_inverse_pi_share_after_floor: float
    target_exact_trim_floor_inverse_pi_phi1_center_projection: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_retained_inverse_pi_share_after_floor: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    comparator_random_states: tuple[int, ...]
    comparator_fold3_raw_phi1_center_projections: tuple[float, ...]
    comparator_fold3_inverse_pi_phi1_center_projections: tuple[float, ...]
    comparator_fold3_weighted_phi1_center_projections: tuple[float, ...]
    driver_signature: str
    canonical_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract_digest: tuple[
        str, ...
    ]

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
        self.acceptance_delta_y_support_chain_driver_signature = str(
            self.acceptance_delta_y_support_chain_driver_signature
        ).strip()
        self.low_pi_fold_trim_floor_driver_signature = str(
            self.low_pi_fold_trim_floor_driver_signature
        ).strip()
        self.low_pi_fold3_trim_floor_propensity_floor_driver_signature = str(
            self.low_pi_fold3_trim_floor_propensity_floor_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.acceptance_shortfall = float(self.acceptance_shortfall)
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_seed_group = str(self.binding_slot_seed_group).strip().lower()
        self.binding_slot_z_index = int(self.binding_slot_z_index)
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.residual_slot_random_state = (
            None
            if self.residual_slot_random_state is None
            else int(self.residual_slot_random_state)
        )
        self.residual_slot_seed_group = (
            None
            if self.residual_slot_seed_group is None
            else str(self.residual_slot_seed_group).strip().lower()
        )
        self.residual_slot_z_index = (
            None
            if self.residual_slot_z_index is None
            else int(self.residual_slot_z_index)
        )
        self.residual_slot_z_value = (
            None
            if self.residual_slot_z_value is None
            else float(self.residual_slot_z_value)
        )
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.low_pi_threshold = float(self.low_pi_threshold)
        self.trim_lower = float(self.trim_lower)
        self.target_treated_count = int(self.target_treated_count)
        self.target_low_pi_treated_count = int(self.target_low_pi_treated_count)
        self.target_low_pi_phi1_center_projection = float(
            self.target_low_pi_phi1_center_projection
        )
        self.fold_ids = tuple(int(value) for value in self.fold_ids)
        self.fold_low_pi_treated_counts = tuple(
            int(value) for value in self.fold_low_pi_treated_counts
        )
        self.fold_low_pi_share_of_treated_count = tuple(
            float(value) for value in self.fold_low_pi_share_of_treated_count
        )
        self.fold_low_pi_treated_delta_over_pi_center_projections = tuple(
            float(value)
            for value in self.fold_low_pi_treated_delta_over_pi_center_projections
        )
        self.fold_low_pi_share_of_weighted_center = tuple(
            float(value) for value in self.fold_low_pi_share_of_weighted_center
        )
        self.fold_low_pi_phi1_center_projections = tuple(
            float(value) for value in self.fold_low_pi_phi1_center_projections
        )
        self.fold_low_pi_share_of_phi1_projection = tuple(
            float(value) for value in self.fold_low_pi_share_of_phi1_projection
        )
        self.dominant_fold_id = int(self.dominant_fold_id)
        self.dominant_fold_low_pi_share_of_weighted_center = float(
            self.dominant_fold_low_pi_share_of_weighted_center
        )
        self.exact_trim_floor_value = float(self.exact_trim_floor_value)
        self.exact_trim_floor_count = int(self.exact_trim_floor_count)
        self.exact_trim_floor_fold_id = int(self.exact_trim_floor_fold_id)
        self.exact_trim_floor_share_of_low_pi_count = float(
            self.exact_trim_floor_share_of_low_pi_count
        )
        self.exact_trim_floor_delta_over_pi_center_projection = float(
            self.exact_trim_floor_delta_over_pi_center_projection
        )
        self.exact_trim_floor_share_of_low_pi_weighted_center = float(
            self.exact_trim_floor_share_of_low_pi_weighted_center
        )
        self.exact_trim_floor_phi1_center_projection = float(
            self.exact_trim_floor_phi1_center_projection
        )
        self.exact_trim_floor_share_of_low_pi_phi1_projection = float(
            self.exact_trim_floor_share_of_low_pi_phi1_projection
        )
        self.target_fold3_raw_phi1_center_projection = float(
            self.target_fold3_raw_phi1_center_projection
        )
        self.target_fold3_inverse_pi_phi1_center_projection = float(
            self.target_fold3_inverse_pi_phi1_center_projection
        )
        self.target_fold3_weighted_phi1_center_projection = float(
            self.target_fold3_weighted_phi1_center_projection
        )
        self.target_fold3_raw_phi1_offset_share_of_inverse_pi = float(
            self.target_fold3_raw_phi1_offset_share_of_inverse_pi
        )
        self.target_fold3_retained_inverse_pi_share_after_floor = float(
            self.target_fold3_retained_inverse_pi_share_after_floor
        )
        self.target_exact_trim_floor_inverse_pi_phi1_center_projection = float(
            self.target_exact_trim_floor_inverse_pi_phi1_center_projection
        )
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_retained_inverse_pi_share_after_floor = float(
            self.target_exact_trim_floor_retained_inverse_pi_share_after_floor
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_fold3_raw_phi1_center_projections = tuple(
            float(value) for value in self.comparator_fold3_raw_phi1_center_projections
        )
        self.comparator_fold3_inverse_pi_phi1_center_projections = tuple(
            float(value)
            for value in self.comparator_fold3_inverse_pi_phi1_center_projections
        )
        self.comparator_fold3_weighted_phi1_center_projections = tuple(
            float(value)
            for value in self.comparator_fold3_weighted_phi1_center_projections
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract_digest
        )


def _driver_signature(
    *,
    acceptance_delta_y_support_chain_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceDeltaYSupportChainContractReport
    ),
    low_pi_fold_trim_floor_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFoldTrimFloorTraceReport
    ),
    low_pi_fold3_trim_floor_propensity_floor_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport
    ),
) -> str:
    fold_trim_floor_confirmed = (
        low_pi_fold_trim_floor_trace_report.driver_signature
        == "same-seed-seed303-low-pi-fold3-trim-floor-driver-confirmed"
    )
    fold3_propensity_floor_confirmed = (
        low_pi_fold3_trim_floor_propensity_floor_trace_report.driver_signature
        == "same-seed-seed303-fold3-trim-floor-propensity-floor-driver-confirmed"
    )
    if (
        acceptance_delta_y_support_chain_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-delta-y-support-chain-open"
        and fold_trim_floor_confirmed
        and fold3_propensity_floor_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-low-pi-support-allocation-open"
    if (
        acceptance_delta_y_support_chain_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-delta-y-support-chain-residual-only"
        and fold_trim_floor_confirmed
        and fold3_propensity_floor_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-low-pi-support-allocation-residual-only"
    if (
        acceptance_delta_y_support_chain_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-delta-y-support-chain-closed"
        and fold_trim_floor_confirmed
        and fold3_propensity_floor_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-low-pi-support-allocation-closed"
    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-acceptance-low-pi-support-allocation-contract"


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceLowPiSupportAllocationContractReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-low-pi-support-allocation-closed"
    ):
        return (
            "- both the acceptance bridge and the residual slot are already closed, so the acceptance-facing low-`pi_hat` support allocation now stays as audit-only provenance",
            f"- the landed first hop still keeps the same support-allocation packet machine-readable: fold `3` owns `{_format_percent(report.dominant_fold_low_pi_share_of_weighted_center)}` of the low-`pi_hat` weighted center, the exact trim-floor row `pi_hat = {_format_exact_trim_floor(report.exact_trim_floor_value)}` retains `{_format_percent(report.target_exact_trim_floor_retained_inverse_pi_share_after_floor)}` of its gross inverse-`pi_hat` conduit after propensity-floor weighting, and that same row still lands at `{_format_float(report.exact_trim_floor_phi1_center_projection)}` of the fold-`3` weighted `phi1_hat` center `{_format_float(report.target_fold3_weighted_phi1_center_projection)}`",
            "- current Trigger 2 implication: keep this acceptance low-`pi_hat` support-allocation contract validation-only and out of live routing surfaces once the first-hop lane is fully closed",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-low-pi-support-allocation-residual-only"
    ):
        return (
            "- the first-hop acceptance shortfall is already closed at `0/9 = 0.000`, so the seed `303` low-`pi_hat` support allocation is now historical provenance rather than a live acceptance blocker",
            f"- that historical packet still records that fold `3` owns `{_format_percent(report.dominant_fold_low_pi_share_of_weighted_center)}` of the low-`pi_hat` weighted center, the exact trim-floor row `pi_hat = {_format_exact_trim_floor(report.exact_trim_floor_value)}` keeps `{_format_percent(report.exact_trim_floor_share_of_low_pi_weighted_center)}` of the low-`pi_hat` weighted center and `{_format_percent(report.exact_trim_floor_share_of_low_pi_phi1_projection)}` of the low-`pi_hat` treated `phi1_hat` burden, and the same row still retains `{_format_percent(report.target_exact_trim_floor_retained_inverse_pi_share_after_floor)}` of its gross inverse-`pi_hat` conduit after propensity-floor weighting while the queued residual slot at seed `707` / fresh / `z = 0.25` remains the only actionable next spend",
            f"- current Trigger 2 implication: `{report.driver_signature}`; keep live entry at `trigger2-policy-spec`, preserve this historical low-`pi_hat` support-allocation packet, and spend the next rerun only on seed `707` residual-only cleanup",
        )
    return (
        f"- the open acceptance shortfall still stays `{_format_ninths(report.acceptance_shortfall)}`, so seed `303` / witness / `z = 0.15` remains the actionable first hop and the acceptance-facing readout `{' -> '.join(report.acceptance_readout_path)}` is still live while the treated low-`pi_hat` tail stays machine-readable inside the acceptance delta-y support chain",
        f"- within that tail (`{report.target_low_pi_treated_count}/{report.target_treated_count} = {_format_percent(report.target_low_pi_treated_count / report.target_treated_count)}` of treated-valid rows), fold `3` already carries `{report.fold_low_pi_treated_counts[2]}/{report.target_low_pi_treated_count} = {_format_percent(report.fold_low_pi_share_of_treated_count[2])}` of the treated low-`pi_hat` support allocation, contributes `{_format_float(report.fold_low_pi_treated_delta_over_pi_center_projections[2])}` of the low-`pi_hat` weighted center `{_format_float(sum(report.fold_low_pi_treated_delta_over_pi_center_projections))}` (`{_format_percent(report.fold_low_pi_share_of_weighted_center[2])}`), and absorbs `{_format_float(report.target_fold3_weighted_phi1_center_projection)}` of the low-`pi_hat` treated `phi1_hat` nuisance burden `{_format_float(report.target_low_pi_phi1_center_projection)}` (`{_format_percent(report.target_fold3_weighted_phi1_center_projection / report.target_low_pi_phi1_center_projection)}`) because folds `1-2` offset it at `{_format_float(report.fold_low_pi_phi1_center_projections[0])}` and `{_format_float(report.fold_low_pi_phi1_center_projections[1])}`",
        f"- inside fold `3`, the single exact trim-floor row `pi_hat = {_format_exact_trim_floor(report.exact_trim_floor_value)}` already carries `{_format_float(report.exact_trim_floor_delta_over_pi_center_projection)}` of the low-`pi_hat` weighted center (`{_format_percent(report.exact_trim_floor_share_of_low_pi_weighted_center)}`) and `{_format_float(report.exact_trim_floor_phi1_center_projection)}` of the low-`pi_hat` treated `phi1_hat` burden (`{_format_percent(report.exact_trim_floor_share_of_low_pi_phi1_projection)}`); the same row contributes `{_format_float(report.target_exact_trim_floor_inverse_pi_phi1_center_projection)}` of the fold-`3` gross inverse-`pi_hat` conduit `{_format_float(report.target_fold3_inverse_pi_phi1_center_projection)}` (`{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}`) and still lands at `{_format_float(report.exact_trim_floor_phi1_center_projection)}` after propensity-floor weighting because `{_format_percent(report.target_exact_trim_floor_retained_inverse_pi_share_after_floor)}` of its inverse-`pi_hat` lift survives once `(1-pi_hat)` is applied, i.e. `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the fold-`3` weighted `phi1_hat` center `{_format_float(report.target_fold3_weighted_phi1_center_projection)}`",
        f"- comparator seed `202` keeps fold `3` gross inverse-`pi_hat` propagation negative at `{_format_float(report.comparator_fold3_inverse_pi_phi1_center_projections[0])}` despite a small positive raw `phi1_hat` center `{_format_float(report.comparator_fold3_raw_phi1_center_projections[0])}`, while residual seed `707` is both raw-negative `{_format_float(report.comparator_fold3_raw_phi1_center_projections[1])}` and gross-negative `{_format_float(report.comparator_fold3_inverse_pi_phi1_center_projections[1])}`; current Trigger 2 implication: `{report.driver_signature}`; keep live entry at `trigger2-policy-spec`, trace the exact trim-floor low-`pi_hat` inverse-`pi_hat` conduit inside fold `3` before spending the queued residual slot on seed `707`",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract_report(
    *,
    acceptance_delta_y_support_chain_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceDeltaYSupportChainContractReport
        | None
    ) = None,
    low_pi_fold_trim_floor_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFoldTrimFloorTraceReport
        | None
    ) = None,
    low_pi_fold3_trim_floor_propensity_floor_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceLowPiSupportAllocationContractReport:
    resolved_acceptance_delta_y_support_chain = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract()
        if acceptance_delta_y_support_chain_contract_report is None
        else acceptance_delta_y_support_chain_contract_report
    )
    resolved_low_pi_fold_trim_floor = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold_trim_floor_trace()
        if low_pi_fold_trim_floor_trace_report is None
        else low_pi_fold_trim_floor_trace_report
    )
    resolved_low_pi_fold3_trim_floor_propensity_floor = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace()
        if low_pi_fold3_trim_floor_propensity_floor_trace_report is None
        else low_pi_fold3_trim_floor_propensity_floor_trace_report
    )

    if (
        resolved_acceptance_delta_y_support_chain.policy_digest
        != resolved_low_pi_fold_trim_floor.policy_digest
        or resolved_acceptance_delta_y_support_chain.policy_digest
        != resolved_low_pi_fold3_trim_floor_propensity_floor.policy_digest
    ):
        raise ValueError(
            "acceptance low-pi support allocation contract requires shared policy digest"
        )
    if (
        resolved_acceptance_delta_y_support_chain.binding_design
        != resolved_low_pi_fold_trim_floor.binding_design
        or resolved_acceptance_delta_y_support_chain.binding_design
        != resolved_low_pi_fold3_trim_floor_propensity_floor.binding_design
    ):
        raise ValueError(
            "acceptance low-pi support allocation contract requires shared binding design"
        )
    if (
        resolved_acceptance_delta_y_support_chain.window_label
        != resolved_low_pi_fold_trim_floor.window_label
        or resolved_acceptance_delta_y_support_chain.window_label
        != resolved_low_pi_fold3_trim_floor_propensity_floor.window_label
    ):
        raise ValueError(
            "acceptance low-pi support allocation contract requires shared window label"
        )
    if (
        resolved_acceptance_delta_y_support_chain.binding_slot_random_state
        != resolved_low_pi_fold_trim_floor.target_random_state
        or resolved_acceptance_delta_y_support_chain.binding_slot_random_state
        != resolved_low_pi_fold3_trim_floor_propensity_floor.target_random_state
    ):
        raise ValueError(
            "acceptance low-pi support allocation contract expects the trim-floor traces to target the binding slot"
        )
    if (
        resolved_acceptance_delta_y_support_chain.binding_slot_seed_group
        != resolved_low_pi_fold_trim_floor.target_seed_group
        or resolved_acceptance_delta_y_support_chain.binding_slot_seed_group
        != resolved_low_pi_fold3_trim_floor_propensity_floor.target_seed_group
    ):
        raise ValueError(
            "acceptance low-pi support allocation contract expects the trim-floor traces to match the binding slot seed group"
        )
    if (
        abs(
            resolved_low_pi_fold_trim_floor.exact_trim_floor_phi1_center_projection
            - resolved_low_pi_fold3_trim_floor_propensity_floor.target_exact_trim_floor_weighted_phi1_center_projection
        )
        > 1e-12
    ):
        raise ValueError(
            "acceptance low-pi support allocation contract requires aligned exact trim-floor weighted phi1 projections"
        )
    if (
        abs(
            resolved_low_pi_fold_trim_floor.exact_trim_floor_value
            - resolved_low_pi_fold3_trim_floor_propensity_floor.target_exact_trim_floor_value
        )
        > 1e-12
    ):
        raise ValueError(
            "acceptance low-pi support allocation contract requires aligned exact trim-floor values"
        )

    driver_signature = _driver_signature(
        acceptance_delta_y_support_chain_contract_report=resolved_acceptance_delta_y_support_chain,
        low_pi_fold_trim_floor_trace_report=resolved_low_pi_fold_trim_floor,
        low_pi_fold3_trim_floor_propensity_floor_trace_report=resolved_low_pi_fold3_trim_floor_propensity_floor,
    )
    resolved_target_treated_count = getattr(
        resolved_acceptance_delta_y_support_chain,
        "target_treated_count",
        int(
            round(
                resolved_low_pi_fold_trim_floor.target_low_pi_treated_count
                / resolved_acceptance_delta_y_support_chain.target_low_pi_share_of_treated_count
            )
        ),
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceLowPiSupportAllocationContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "first-hop-acceptance-low-pi-support-allocation-contract"
        ),
        policy_digest=resolved_acceptance_delta_y_support_chain.policy_digest,
        binding_design=resolved_acceptance_delta_y_support_chain.binding_design,
        window_label=resolved_acceptance_delta_y_support_chain.window_label,
        same_seed_random_states=resolved_acceptance_delta_y_support_chain.same_seed_random_states,
        acceptance_delta_y_support_chain_driver_signature=resolved_acceptance_delta_y_support_chain.driver_signature,
        low_pi_fold_trim_floor_driver_signature=resolved_low_pi_fold_trim_floor.driver_signature,
        low_pi_fold3_trim_floor_propensity_floor_driver_signature=resolved_low_pi_fold3_trim_floor_propensity_floor.driver_signature,
        runtime_witness_path=resolved_acceptance_delta_y_support_chain.runtime_witness_path,
        acceptance_readout_path=resolved_acceptance_delta_y_support_chain.acceptance_readout_path,
        acceptance_shortfall=resolved_acceptance_delta_y_support_chain.acceptance_shortfall,
        current_witness_floor=resolved_acceptance_delta_y_support_chain.current_witness_floor,
        required_min_witness_floor=resolved_acceptance_delta_y_support_chain.required_min_witness_floor,
        binding_slot_random_state=resolved_acceptance_delta_y_support_chain.binding_slot_random_state,
        binding_slot_seed_group=resolved_acceptance_delta_y_support_chain.binding_slot_seed_group,
        binding_slot_z_index=resolved_acceptance_delta_y_support_chain.binding_slot_z_index,
        binding_slot_z_value=resolved_acceptance_delta_y_support_chain.binding_slot_z_value,
        residual_slot_random_state=resolved_acceptance_delta_y_support_chain.residual_slot_random_state,
        residual_slot_seed_group=resolved_acceptance_delta_y_support_chain.residual_slot_seed_group,
        residual_slot_z_index=resolved_acceptance_delta_y_support_chain.residual_slot_z_index,
        residual_slot_z_value=resolved_acceptance_delta_y_support_chain.residual_slot_z_value,
        binding_replication_seed=resolved_acceptance_delta_y_support_chain.binding_replication_seed,
        low_pi_threshold=resolved_low_pi_fold_trim_floor.low_pi_threshold,
        trim_lower=resolved_low_pi_fold_trim_floor.trim_lower,
        target_treated_count=resolved_target_treated_count,
        target_low_pi_treated_count=resolved_low_pi_fold_trim_floor.target_low_pi_treated_count,
        target_low_pi_phi1_center_projection=resolved_low_pi_fold_trim_floor.target_low_pi_phi1_center_projection,
        fold_ids=resolved_low_pi_fold_trim_floor.fold_ids,
        fold_low_pi_treated_counts=resolved_low_pi_fold_trim_floor.fold_low_pi_treated_counts,
        fold_low_pi_share_of_treated_count=resolved_low_pi_fold_trim_floor.fold_low_pi_share_of_treated_count,
        fold_low_pi_treated_delta_over_pi_center_projections=resolved_low_pi_fold_trim_floor.fold_low_pi_treated_delta_over_pi_center_projections,
        fold_low_pi_share_of_weighted_center=resolved_low_pi_fold_trim_floor.fold_low_pi_share_of_weighted_center,
        fold_low_pi_phi1_center_projections=resolved_low_pi_fold_trim_floor.fold_low_pi_phi1_center_projections,
        fold_low_pi_share_of_phi1_projection=resolved_low_pi_fold_trim_floor.fold_low_pi_share_of_phi1_projection,
        dominant_fold_id=resolved_low_pi_fold_trim_floor.dominant_fold_id,
        dominant_fold_low_pi_share_of_weighted_center=resolved_low_pi_fold_trim_floor.dominant_fold_low_pi_share_of_weighted_center,
        exact_trim_floor_value=resolved_low_pi_fold_trim_floor.exact_trim_floor_value,
        exact_trim_floor_count=resolved_low_pi_fold_trim_floor.exact_trim_floor_count,
        exact_trim_floor_fold_id=resolved_low_pi_fold_trim_floor.exact_trim_floor_fold_id,
        exact_trim_floor_share_of_low_pi_count=resolved_low_pi_fold_trim_floor.exact_trim_floor_share_of_low_pi_count,
        exact_trim_floor_delta_over_pi_center_projection=resolved_low_pi_fold_trim_floor.exact_trim_floor_delta_over_pi_center_projection,
        exact_trim_floor_share_of_low_pi_weighted_center=resolved_low_pi_fold_trim_floor.exact_trim_floor_share_of_low_pi_weighted_center,
        exact_trim_floor_phi1_center_projection=resolved_low_pi_fold_trim_floor.exact_trim_floor_phi1_center_projection,
        exact_trim_floor_share_of_low_pi_phi1_projection=resolved_low_pi_fold_trim_floor.exact_trim_floor_share_of_low_pi_phi1_projection,
        target_fold3_raw_phi1_center_projection=resolved_low_pi_fold3_trim_floor_propensity_floor.target_fold3_raw_phi1_center_projection,
        target_fold3_inverse_pi_phi1_center_projection=resolved_low_pi_fold3_trim_floor_propensity_floor.target_fold3_inverse_pi_phi1_center_projection,
        target_fold3_weighted_phi1_center_projection=resolved_low_pi_fold3_trim_floor_propensity_floor.target_fold3_weighted_phi1_center_projection,
        target_fold3_raw_phi1_offset_share_of_inverse_pi=resolved_low_pi_fold3_trim_floor_propensity_floor.target_fold3_raw_phi1_offset_share_of_inverse_pi,
        target_fold3_retained_inverse_pi_share_after_floor=resolved_low_pi_fold3_trim_floor_propensity_floor.target_fold3_retained_inverse_pi_share_after_floor,
        target_exact_trim_floor_inverse_pi_phi1_center_projection=resolved_low_pi_fold3_trim_floor_propensity_floor.target_exact_trim_floor_inverse_pi_phi1_center_projection,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=resolved_low_pi_fold3_trim_floor_propensity_floor.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi,
        target_exact_trim_floor_retained_inverse_pi_share_after_floor=resolved_low_pi_fold3_trim_floor_propensity_floor.target_exact_trim_floor_retained_inverse_pi_share_after_floor,
        target_exact_trim_floor_weighted_share_of_fold3_weighted=resolved_low_pi_fold3_trim_floor_propensity_floor.target_exact_trim_floor_weighted_share_of_fold3_weighted,
        comparator_random_states=resolved_low_pi_fold3_trim_floor_propensity_floor.comparator_random_states,
        comparator_fold3_raw_phi1_center_projections=resolved_low_pi_fold3_trim_floor_propensity_floor.comparator_fold3_raw_phi1_center_projections,
        comparator_fold3_inverse_pi_phi1_center_projections=resolved_low_pi_fold3_trim_floor_propensity_floor.comparator_fold3_inverse_pi_phi1_center_projections,
        comparator_fold3_weighted_phi1_center_projections=resolved_low_pi_fold3_trim_floor_propensity_floor.comparator_fold3_weighted_phi1_center_projections,
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract_digest=(),
    )
    report = replace(
        report,
        canonical_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract_digest=_canonical_digest(
            report=report
        ),
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceLowPiSupportAllocationContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceLowPiSupportAllocationContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract",
]
