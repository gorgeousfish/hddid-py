from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceRawScoreComponentContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiTreatedSupportTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoSupportSplitTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_threshold(value: float) -> str:
    return f"{float(value):.1f}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceDeltaYSupportChainContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    acceptance_raw_score_component_driver_signature: str
    rho_support_split_driver_signature: str
    low_pi_treated_support_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    current_witness_floor: float
    required_min_witness_floor: float
    acceptance_shortfall: float
    current_actionable_slot_random_state: int | None
    current_actionable_slot_seed_group: str | None
    current_actionable_slot_z_index: int | None
    current_actionable_slot_z_value: float | None
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_index: int
    binding_slot_z_value: float
    residual_slot_random_state: int | None
    residual_slot_seed_group: str | None
    residual_slot_z_index: int | None
    residual_slot_z_value: float | None
    binding_replication_seed: int
    target_truth_center: float
    target_raw_score_center_estimate: float
    target_rho_delta_y_center_projection: float
    target_treated_delta_over_pi_center_projection: float
    target_control_delta_over_one_minus_pi_center_projection: float
    target_treated_share_of_weighted_center: float
    target_treated_share_of_positive_support_lift: float
    low_pi_threshold: float
    target_treated_count: int
    target_low_pi_treated_count: int
    target_low_pi_share_of_treated_count: float
    target_low_pi_treated_delta_over_pi_center_projection: float
    target_high_pi_treated_delta_over_pi_center_projection: float
    target_low_pi_share_of_treated_weighted_center: float
    target_low_pi_treated_support_lift: float
    target_high_pi_treated_support_lift: float
    target_low_pi_share_of_positive_support_lift: float
    comparator_random_states: tuple[int, ...]
    comparator_low_pi_treated_delta_over_pi_center_projections: tuple[float, ...]
    comparator_low_pi_share_of_treated_weighted_center: tuple[float, ...]
    comparator_low_pi_share_of_positive_support_lift: tuple[float, ...]
    delta_y_support_chain_is_live: bool
    delta_y_support_chain_is_historical: bool
    driver_signature: str
    canonical_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract_digest: tuple[
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
        self.acceptance_raw_score_component_driver_signature = str(
            self.acceptance_raw_score_component_driver_signature
        ).strip()
        self.rho_support_split_driver_signature = str(
            self.rho_support_split_driver_signature
        ).strip()
        self.low_pi_treated_support_driver_signature = str(
            self.low_pi_treated_support_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.acceptance_shortfall = float(self.acceptance_shortfall)
        self.current_actionable_slot_random_state = (
            None
            if self.current_actionable_slot_random_state is None
            else int(self.current_actionable_slot_random_state)
        )
        self.current_actionable_slot_seed_group = (
            None
            if self.current_actionable_slot_seed_group is None
            else str(self.current_actionable_slot_seed_group).strip().lower()
        )
        self.current_actionable_slot_z_index = (
            None
            if self.current_actionable_slot_z_index is None
            else int(self.current_actionable_slot_z_index)
        )
        self.current_actionable_slot_z_value = (
            None
            if self.current_actionable_slot_z_value is None
            else float(self.current_actionable_slot_z_value)
        )
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
        self.target_truth_center = float(self.target_truth_center)
        self.target_raw_score_center_estimate = float(
            self.target_raw_score_center_estimate
        )
        self.target_rho_delta_y_center_projection = float(
            self.target_rho_delta_y_center_projection
        )
        self.target_treated_delta_over_pi_center_projection = float(
            self.target_treated_delta_over_pi_center_projection
        )
        self.target_control_delta_over_one_minus_pi_center_projection = float(
            self.target_control_delta_over_one_minus_pi_center_projection
        )
        self.target_treated_share_of_weighted_center = float(
            self.target_treated_share_of_weighted_center
        )
        self.target_treated_share_of_positive_support_lift = float(
            self.target_treated_share_of_positive_support_lift
        )
        self.low_pi_threshold = float(self.low_pi_threshold)
        self.target_treated_count = int(self.target_treated_count)
        self.target_low_pi_treated_count = int(self.target_low_pi_treated_count)
        self.target_low_pi_share_of_treated_count = float(
            self.target_low_pi_share_of_treated_count
        )
        self.target_low_pi_treated_delta_over_pi_center_projection = float(
            self.target_low_pi_treated_delta_over_pi_center_projection
        )
        self.target_high_pi_treated_delta_over_pi_center_projection = float(
            self.target_high_pi_treated_delta_over_pi_center_projection
        )
        self.target_low_pi_share_of_treated_weighted_center = float(
            self.target_low_pi_share_of_treated_weighted_center
        )
        self.target_low_pi_treated_support_lift = float(
            self.target_low_pi_treated_support_lift
        )
        self.target_high_pi_treated_support_lift = float(
            self.target_high_pi_treated_support_lift
        )
        self.target_low_pi_share_of_positive_support_lift = float(
            self.target_low_pi_share_of_positive_support_lift
        )
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_low_pi_treated_delta_over_pi_center_projections = tuple(
            float(value)
            for value in self.comparator_low_pi_treated_delta_over_pi_center_projections
        )
        self.comparator_low_pi_share_of_treated_weighted_center = tuple(
            float(value)
            for value in self.comparator_low_pi_share_of_treated_weighted_center
        )
        self.comparator_low_pi_share_of_positive_support_lift = tuple(
            float(value)
            for value in self.comparator_low_pi_share_of_positive_support_lift
        )
        self.delta_y_support_chain_is_live = bool(self.delta_y_support_chain_is_live)
        self.delta_y_support_chain_is_historical = bool(
            self.delta_y_support_chain_is_historical
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract_digest
        )


def _driver_signature(
    *,
    acceptance_raw_score_component_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceRawScoreComponentContractReport
    ),
    rho_support_split_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoSupportSplitTraceReport
    ),
    low_pi_treated_support_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiTreatedSupportTraceReport
    ),
) -> str:
    support_split_confirmed = (
        rho_support_split_trace_report.driver_signature
        == "same-seed-seed303-treated-inverse-pi-support-driver-confirmed"
    )
    low_pi_support_confirmed = (
        low_pi_treated_support_trace_report.driver_signature
        == "same-seed-seed303-low-pi-treated-support-driver-confirmed"
    )
    if (
        acceptance_raw_score_component_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-raw-score-component-open"
        and support_split_confirmed
        and low_pi_support_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-delta-y-support-chain-open"
    if (
        acceptance_raw_score_component_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-raw-score-component-residual-only"
        and support_split_confirmed
        and low_pi_support_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-delta-y-support-chain-residual-only"
    if (
        acceptance_raw_score_component_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-raw-score-component-closed"
        and support_split_confirmed
        and low_pi_support_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-delta-y-support-chain-closed"
    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-acceptance-delta-y-support-chain-contract"


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceDeltaYSupportChainContractReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-delta-y-support-chain-closed"
    ):
        return (
            "- both the acceptance bridge and the residual slot are already closed, so the acceptance-facing delta-y support chain now stays as audit-only provenance",
            f"- the landed first hop still keeps the same support packet machine-readable: treated `delta_y / pi_hat = {_format_float(report.target_treated_delta_over_pi_center_projection)}` versus control `-delta_y / (1-pi_hat) = {_format_float(report.target_control_delta_over_one_minus_pi_center_projection)}`, with the low-`pi_hat` tail contributing `{_format_float(report.target_low_pi_treated_delta_over_pi_center_projection)}` of `{_format_float(report.target_treated_delta_over_pi_center_projection)}` and `{_format_float(report.target_low_pi_treated_support_lift)}` of the treated support lift `{_format_float(report.target_low_pi_treated_support_lift + report.target_high_pi_treated_support_lift)}`",
            "- current Trigger 2 implication: keep this acceptance delta-y support chain contract validation-only and out of live routing surfaces once the first-hop lane is fully closed",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-delta-y-support-chain-residual-only"
    ):
        return (
            "- the first-hop acceptance shortfall is already closed at `0/9 = 0.000`, so the seed `303` acceptance-facing delta-y support chain is now historical provenance rather than a live blocker",
            f"- that historical chain still records the treated inverse-`pi_hat` support split (`{_format_float(report.target_treated_delta_over_pi_center_projection)}` treated vs `{_format_float(report.target_control_delta_over_one_minus_pi_center_projection)}` control) and the low-`pi_hat` tail domination (`{_format_float(report.target_low_pi_treated_delta_over_pi_center_projection)}` of `{_format_float(report.target_treated_delta_over_pi_center_projection)}`, `{_format_percent(report.target_low_pi_share_of_treated_weighted_center)}` of treated weighted support, `{_format_percent(report.target_low_pi_share_of_positive_support_lift)}` of treated support lift) while the queued residual slot at seed `707` / fresh / `z = 0.25` remains the only actionable next spend",
            f"- current Trigger 2 implication: `{report.driver_signature}`; keep live entry at `trigger2-policy-spec`, preserve this historical support-chain packet, and spend the next rerun only on seed `707` residual-only cleanup",
        )
    return (
        f"- the open acceptance shortfall still stays `{_format_ninths(report.acceptance_shortfall)}`, so seed `303` / witness / `z = 0.15` remains the actionable first hop and the acceptance-facing readout `{' -> '.join(report.acceptance_readout_path)}` is still live",
        f"- inside that live readout, the delta-y packet `rho_hat * delta_y = {_format_float(report.target_rho_delta_y_center_projection)}` already splits to treated `delta_y / pi_hat = {_format_float(report.target_treated_delta_over_pi_center_projection)}` versus control `-delta_y / (1-pi_hat) = {_format_float(report.target_control_delta_over_one_minus_pi_center_projection)}`, so treated inverse-`pi_hat` support carries `{_format_percent(report.target_treated_share_of_weighted_center)}` of the weighted center and `{_format_percent(report.target_treated_share_of_positive_support_lift)}` of the positive support lift before any residual spend",
        f"- within the treated branch, only the low-`pi_hat` tail `pi_hat <= {_format_threshold(report.low_pi_threshold)}` (`{report.target_low_pi_treated_count}/{report.target_treated_count} = {_format_percent(report.target_low_pi_share_of_treated_count)}` of treated-valid rows) already contributes `{_format_float(report.target_low_pi_treated_delta_over_pi_center_projection)}` of `{_format_float(report.target_treated_delta_over_pi_center_projection)}` (`{_format_percent(report.target_low_pi_share_of_treated_weighted_center)}`) and `{_format_float(report.target_low_pi_treated_support_lift)}` of the total treated support lift `{_format_float(report.target_low_pi_treated_support_lift + report.target_high_pi_treated_support_lift)}` (`{_format_percent(report.target_low_pi_share_of_positive_support_lift)}`), while the remaining high-`pi_hat` treated support adds only `{_format_float(report.target_high_pi_treated_support_lift)}`; comparator seed `202` flips the same tail negative at `{_format_float(report.comparator_low_pi_treated_delta_over_pi_center_projections[0])}`, and residual seed `707` keeps it small at `{_format_float(report.comparator_low_pi_treated_delta_over_pi_center_projections[1])}`",
        f"- current Trigger 2 implication: `{report.driver_signature}`; keep live entry at `trigger2-policy-spec`, trace the treated low-`pi_hat` tail's `pi_hat` floor / support allocation for seed `303`, and spend seed `707` only after the first hop lands",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract_report(
    *,
    acceptance_raw_score_component_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceRawScoreComponentContractReport
        | None
    ) = None,
    rho_support_split_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RhoSupportSplitTraceReport
        | None
    ) = None,
    low_pi_treated_support_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiTreatedSupportTraceReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceDeltaYSupportChainContractReport:
    resolved_acceptance_raw_score_component_contract = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract()
        if acceptance_raw_score_component_contract_report is None
        else acceptance_raw_score_component_contract_report
    )
    resolved_rho_support_split_trace = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace()
        if rho_support_split_trace_report is None
        else rho_support_split_trace_report
    )
    resolved_low_pi_treated_support_trace = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace()
        if low_pi_treated_support_trace_report is None
        else low_pi_treated_support_trace_report
    )

    if (
        resolved_acceptance_raw_score_component_contract.policy_digest
        != resolved_rho_support_split_trace.policy_digest
        or resolved_acceptance_raw_score_component_contract.policy_digest
        != resolved_low_pi_treated_support_trace.policy_digest
    ):
        raise ValueError(
            "acceptance delta-y support chain contract requires shared policy digest"
        )
    if (
        resolved_acceptance_raw_score_component_contract.binding_design
        != resolved_rho_support_split_trace.binding_design
        or resolved_acceptance_raw_score_component_contract.binding_design
        != resolved_low_pi_treated_support_trace.binding_design
    ):
        raise ValueError(
            "acceptance delta-y support chain contract requires shared binding design"
        )
    if (
        resolved_acceptance_raw_score_component_contract.window_label
        != resolved_rho_support_split_trace.window_label
        or resolved_acceptance_raw_score_component_contract.window_label
        != resolved_low_pi_treated_support_trace.window_label
    ):
        raise ValueError(
            "acceptance delta-y support chain contract requires shared window label"
        )
    if (
        resolved_acceptance_raw_score_component_contract.binding_slot_random_state
        != resolved_rho_support_split_trace.target_random_state
        or resolved_acceptance_raw_score_component_contract.binding_slot_random_state
        != resolved_low_pi_treated_support_trace.target_random_state
    ):
        raise ValueError(
            "acceptance delta-y support chain contract expects the support traces to target the binding slot"
        )
    if (
        resolved_acceptance_raw_score_component_contract.binding_slot_seed_group
        != resolved_rho_support_split_trace.target_seed_group
        or resolved_acceptance_raw_score_component_contract.binding_slot_seed_group
        != resolved_low_pi_treated_support_trace.target_seed_group
    ):
        raise ValueError(
            "acceptance delta-y support chain contract expects the support traces to match the binding slot seed group"
        )
    if (
        abs(
            resolved_rho_support_split_trace.target_treated_delta_over_pi_center_projection
            - resolved_low_pi_treated_support_trace.target_treated_delta_over_pi_center_projection
        )
        > 1e-12
    ):
        raise ValueError(
            "acceptance delta-y support chain contract requires aligned treated delta/pi support totals"
        )

    driver_signature = _driver_signature(
        acceptance_raw_score_component_contract_report=resolved_acceptance_raw_score_component_contract,
        rho_support_split_trace_report=resolved_rho_support_split_trace,
        low_pi_treated_support_trace_report=resolved_low_pi_treated_support_trace,
    )
    delta_y_support_chain_is_live = bool(
        resolved_acceptance_raw_score_component_contract.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-raw-score-component-open"
    )
    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceDeltaYSupportChainContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "first-hop-acceptance-delta-y-support-chain-contract"
        ),
        policy_digest=resolved_acceptance_raw_score_component_contract.policy_digest,
        binding_design=resolved_acceptance_raw_score_component_contract.binding_design,
        window_label=resolved_acceptance_raw_score_component_contract.window_label,
        same_seed_random_states=resolved_acceptance_raw_score_component_contract.same_seed_random_states,
        acceptance_raw_score_component_driver_signature=resolved_acceptance_raw_score_component_contract.driver_signature,
        rho_support_split_driver_signature=resolved_rho_support_split_trace.driver_signature,
        low_pi_treated_support_driver_signature=resolved_low_pi_treated_support_trace.driver_signature,
        runtime_witness_path=resolved_acceptance_raw_score_component_contract.runtime_witness_path,
        acceptance_readout_path=resolved_acceptance_raw_score_component_contract.acceptance_readout_path,
        current_witness_floor=resolved_acceptance_raw_score_component_contract.current_witness_floor,
        required_min_witness_floor=resolved_acceptance_raw_score_component_contract.required_min_witness_floor,
        acceptance_shortfall=resolved_acceptance_raw_score_component_contract.acceptance_shortfall,
        current_actionable_slot_random_state=resolved_acceptance_raw_score_component_contract.current_actionable_slot_random_state,
        current_actionable_slot_seed_group=resolved_acceptance_raw_score_component_contract.current_actionable_slot_seed_group,
        current_actionable_slot_z_index=resolved_acceptance_raw_score_component_contract.current_actionable_slot_z_index,
        current_actionable_slot_z_value=resolved_acceptance_raw_score_component_contract.current_actionable_slot_z_value,
        binding_slot_random_state=resolved_acceptance_raw_score_component_contract.binding_slot_random_state,
        binding_slot_seed_group=resolved_acceptance_raw_score_component_contract.binding_slot_seed_group,
        binding_slot_z_index=resolved_acceptance_raw_score_component_contract.binding_slot_z_index,
        binding_slot_z_value=resolved_acceptance_raw_score_component_contract.binding_slot_z_value,
        residual_slot_random_state=resolved_acceptance_raw_score_component_contract.residual_slot_random_state,
        residual_slot_seed_group=resolved_acceptance_raw_score_component_contract.residual_slot_seed_group,
        residual_slot_z_index=resolved_acceptance_raw_score_component_contract.residual_slot_z_index,
        residual_slot_z_value=resolved_acceptance_raw_score_component_contract.residual_slot_z_value,
        binding_replication_seed=resolved_acceptance_raw_score_component_contract.binding_replication_seed,
        target_truth_center=resolved_acceptance_raw_score_component_contract.target_truth_center,
        target_raw_score_center_estimate=resolved_acceptance_raw_score_component_contract.target_raw_score_center_estimate,
        target_rho_delta_y_center_projection=resolved_acceptance_raw_score_component_contract.target_rho_delta_y_center_projection,
        target_treated_delta_over_pi_center_projection=resolved_rho_support_split_trace.target_treated_delta_over_pi_center_projection,
        target_control_delta_over_one_minus_pi_center_projection=resolved_rho_support_split_trace.target_control_delta_over_one_minus_pi_center_projection,
        target_treated_share_of_weighted_center=resolved_rho_support_split_trace.target_treated_share_of_weighted_center,
        target_treated_share_of_positive_support_lift=resolved_rho_support_split_trace.target_treated_share_of_positive_support_lift,
        low_pi_threshold=resolved_low_pi_treated_support_trace.low_pi_threshold,
        target_treated_count=resolved_low_pi_treated_support_trace.target_treated_count,
        target_low_pi_treated_count=resolved_low_pi_treated_support_trace.target_low_pi_treated_count,
        target_low_pi_share_of_treated_count=resolved_low_pi_treated_support_trace.target_low_pi_share_of_treated_count,
        target_low_pi_treated_delta_over_pi_center_projection=resolved_low_pi_treated_support_trace.target_low_pi_treated_delta_over_pi_center_projection,
        target_high_pi_treated_delta_over_pi_center_projection=resolved_low_pi_treated_support_trace.target_high_pi_treated_delta_over_pi_center_projection,
        target_low_pi_share_of_treated_weighted_center=resolved_low_pi_treated_support_trace.target_low_pi_share_of_treated_weighted_center,
        target_low_pi_treated_support_lift=resolved_low_pi_treated_support_trace.target_low_pi_treated_support_lift,
        target_high_pi_treated_support_lift=resolved_low_pi_treated_support_trace.target_high_pi_treated_support_lift,
        target_low_pi_share_of_positive_support_lift=resolved_low_pi_treated_support_trace.target_low_pi_share_of_positive_support_lift,
        comparator_random_states=resolved_low_pi_treated_support_trace.comparator_random_states,
        comparator_low_pi_treated_delta_over_pi_center_projections=resolved_low_pi_treated_support_trace.comparator_low_pi_treated_delta_over_pi_center_projections,
        comparator_low_pi_share_of_treated_weighted_center=resolved_low_pi_treated_support_trace.comparator_low_pi_share_of_treated_weighted_center,
        comparator_low_pi_share_of_positive_support_lift=resolved_low_pi_treated_support_trace.comparator_low_pi_share_of_positive_support_lift,
        delta_y_support_chain_is_live=delta_y_support_chain_is_live,
        delta_y_support_chain_is_historical=not delta_y_support_chain_is_live,
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract_digest=(),
    )
    report = replace(
        report,
        canonical_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract_digest=_canonical_digest(
            report=report
        ),
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceDeltaYSupportChainContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceDeltaYSupportChainContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract",
]
