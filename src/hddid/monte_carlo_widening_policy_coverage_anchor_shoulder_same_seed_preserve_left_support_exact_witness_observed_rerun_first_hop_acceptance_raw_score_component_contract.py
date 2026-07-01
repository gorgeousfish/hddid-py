from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RawScoreComponentTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceRawScoreComponentContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    acceptance_eq31_source_driver_signature: str
    raw_score_component_driver_signature: str
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
    open_blocker_center_estimate: float
    open_blocker_pointwise_lower: float
    open_blocker_vf_cross_entry: float
    eq31_center_estimate: float
    target_truth_center: float
    target_raw_score_center_estimate: float
    target_rho_delta_y_center_projection: float
    target_rho_one_minus_pi_phi1_center_projection: float
    target_rho_pi_phi0_center_projection: float
    target_delta_y_gap_over_truth: float
    target_phi1_relief_share_of_delta_y_gap: float
    target_phi0_reinforcement_share_of_delta_y_gap: float
    target_raw_score_share_of_delta_y_gap: float
    comparator_random_states: tuple[int, ...]
    comparator_rho_delta_y_center_projections: tuple[float, ...]
    comparator_rho_one_minus_pi_phi1_center_projections: tuple[float, ...]
    comparator_rho_pi_phi0_center_projections: tuple[float, ...]
    raw_score_component_is_live: bool
    raw_score_component_is_historical: bool
    driver_signature: str
    canonical_observed_rerun_first_hop_acceptance_raw_score_component_contract_digest: (
        tuple[str, ...]
    )

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
        self.acceptance_eq31_source_driver_signature = str(
            self.acceptance_eq31_source_driver_signature
        ).strip()
        self.raw_score_component_driver_signature = str(
            self.raw_score_component_driver_signature
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
        self.open_blocker_center_estimate = float(self.open_blocker_center_estimate)
        self.open_blocker_pointwise_lower = float(self.open_blocker_pointwise_lower)
        self.open_blocker_vf_cross_entry = float(self.open_blocker_vf_cross_entry)
        self.eq31_center_estimate = float(self.eq31_center_estimate)
        self.target_truth_center = float(self.target_truth_center)
        self.target_raw_score_center_estimate = float(
            self.target_raw_score_center_estimate
        )
        self.target_rho_delta_y_center_projection = float(
            self.target_rho_delta_y_center_projection
        )
        self.target_rho_one_minus_pi_phi1_center_projection = float(
            self.target_rho_one_minus_pi_phi1_center_projection
        )
        self.target_rho_pi_phi0_center_projection = float(
            self.target_rho_pi_phi0_center_projection
        )
        self.target_delta_y_gap_over_truth = float(self.target_delta_y_gap_over_truth)
        self.target_phi1_relief_share_of_delta_y_gap = float(
            self.target_phi1_relief_share_of_delta_y_gap
        )
        self.target_phi0_reinforcement_share_of_delta_y_gap = float(
            self.target_phi0_reinforcement_share_of_delta_y_gap
        )
        self.target_raw_score_share_of_delta_y_gap = float(
            self.target_raw_score_share_of_delta_y_gap
        )
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_rho_delta_y_center_projections = tuple(
            float(value) for value in self.comparator_rho_delta_y_center_projections
        )
        self.comparator_rho_one_minus_pi_phi1_center_projections = tuple(
            float(value)
            for value in self.comparator_rho_one_minus_pi_phi1_center_projections
        )
        self.comparator_rho_pi_phi0_center_projections = tuple(
            float(value) for value in self.comparator_rho_pi_phi0_center_projections
        )
        self.raw_score_component_is_live = bool(self.raw_score_component_is_live)
        self.raw_score_component_is_historical = bool(
            self.raw_score_component_is_historical
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_first_hop_acceptance_raw_score_component_contract_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_observed_rerun_first_hop_acceptance_raw_score_component_contract_digest
        )


def _driver_signature(
    *,
    acceptance_eq31_source_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport
    ),
    raw_score_component_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RawScoreComponentTraceReport
    ),
) -> str:
    component_confirmed = (
        raw_score_component_trace_report.driver_signature
        == "same-seed-seed303-raw-score-delta-y-driver-confirmed"
    )
    if (
        acceptance_eq31_source_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-eq31-source-open"
        and component_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-raw-score-component-open"
    if (
        acceptance_eq31_source_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-eq31-source-residual-only"
        and component_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-raw-score-component-residual-only"
    if (
        acceptance_eq31_source_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-eq31-source-closed"
        and component_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-raw-score-component-closed"
    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-acceptance-raw-score-component-contract"


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceRawScoreComponentContractReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-raw-score-component-closed"
    ):
        return (
            "- both the acceptance bridge and the residual slot are already closed, so the acceptance-facing readout, its Eq. (3.1) source, and its raw-score component packet now stay as audit-only provenance",
            f"- the landed first hop still keeps the same component packet machine-readable: raw score `{_format_float(report.target_raw_score_center_estimate)}`, `rho_hat * delta_y = {_format_float(report.target_rho_delta_y_center_projection)}`, `rho_hat * (1-pi_hat) * phi1_hat = {_format_float(report.target_rho_one_minus_pi_phi1_center_projection)}`, and `rho_hat * pi_hat * phi0_hat = {_format_float(report.target_rho_pi_phi0_center_projection)}`",
            "- current Trigger 2 implication: keep this acceptance raw-score component contract validation-only and out of live routing surfaces once the first-hop lane is fully closed",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-raw-score-component-residual-only"
    ):
        return (
            "- the first-hop acceptance shortfall is already closed at `0/9 = 0.000`, so the seed `303` acceptance-facing readout, its Eq. (3.1) source, and its raw-score component packet are now historical provenance rather than a live blocker",
            f"- the same historical packet still records that raw score `{_format_float(report.target_raw_score_center_estimate)}` was delta-y-led before residualization (`rho_hat * delta_y = {_format_float(report.target_rho_delta_y_center_projection)}`, `rho_hat * (1-pi_hat) * phi1_hat = {_format_float(report.target_rho_one_minus_pi_phi1_center_projection)}`, `rho_hat * pi_hat * phi0_hat = {_format_float(report.target_rho_pi_phi0_center_projection)}`) while the queued residual slot at seed `707` / fresh / `z = 0.25` remains the only actionable next spend",
            f"- current Trigger 2 implication: `{report.driver_signature}`; keep live entry at `trigger2-policy-spec`, preserve the historical raw-score packet, and spend the next rerun only on seed `707` residual-only cleanup",
        )
    return (
        f"- the open acceptance shortfall still stays `{_format_ninths(report.acceptance_shortfall)}`, so seed `303` / witness / `z = 0.15` remains the actionable first hop and the acceptance-facing readout `{' -> '.join(report.acceptance_readout_path)}` is still live",
        f"- that live readout is now tied back one layer deeper than Eq. (3.1): raw score `{_format_float(report.target_raw_score_center_estimate)}` decomposes into `rho_hat * delta_y = {_format_float(report.target_rho_delta_y_center_projection)}`, `rho_hat * (1-pi_hat) * phi1_hat = {_format_float(report.target_rho_one_minus_pi_phi1_center_projection)}`, and `rho_hat * pi_hat * phi0_hat = {_format_float(report.target_rho_pi_phi0_center_projection)}`, so the acceptance-facing miss is already delta-y-led before residualization",
        f"- the `rho_hat * delta_y` lane alone overshoots truth `{_format_float(report.target_truth_center)}` by `{_format_float(report.target_delta_y_gap_over_truth)}`; `phi1_hat` claws back `{100.0 * report.target_phi1_relief_share_of_delta_y_gap:.1f}%` of that overshoot, but negative `phi0_hat` adds back `{100.0 * report.target_phi0_reinforcement_share_of_delta_y_gap:.1f}%`, leaving raw score still `{_format_float(report.target_raw_score_center_estimate - report.target_truth_center)}` above truth while residual seed `707` stays queued at `0/9 = 0.000`",
        f"- current Trigger 2 implication: `{report.driver_signature}`; keep live entry at `trigger2-policy-spec`, split seed `303` between `rho_hat` and `delta_y` inputs first, and spend seed `707` only after the first hop lands",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract_report(
    *,
    acceptance_eq31_source_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport
        | None
    ) = None,
    raw_score_component_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303RawScoreComponentTraceReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceRawScoreComponentContractReport:
    resolved_acceptance_eq31_source_contract = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract()
        if acceptance_eq31_source_contract_report is None
        else acceptance_eq31_source_contract_report
    )
    resolved_raw_score_component_trace = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace()
        if raw_score_component_trace_report is None
        else raw_score_component_trace_report
    )

    if (
        resolved_acceptance_eq31_source_contract.policy_digest
        != resolved_raw_score_component_trace.policy_digest
    ):
        raise ValueError(
            "acceptance raw-score component contract requires shared policy digest"
        )
    if (
        resolved_acceptance_eq31_source_contract.binding_design
        != resolved_raw_score_component_trace.binding_design
    ):
        raise ValueError(
            "acceptance raw-score component contract requires shared binding design"
        )
    if (
        resolved_acceptance_eq31_source_contract.window_label
        != resolved_raw_score_component_trace.window_label
    ):
        raise ValueError(
            "acceptance raw-score component contract requires shared window label"
        )
    if (
        resolved_acceptance_eq31_source_contract.binding_slot_random_state
        != resolved_raw_score_component_trace.target_random_state
    ):
        raise ValueError(
            "acceptance raw-score component contract expects the component trace target to match the binding slot"
        )
    if (
        resolved_acceptance_eq31_source_contract.binding_slot_seed_group
        != resolved_raw_score_component_trace.target_seed_group
    ):
        raise ValueError(
            "acceptance raw-score component contract expects the component trace seed group to match the binding slot"
        )

    driver_signature = _driver_signature(
        acceptance_eq31_source_contract_report=resolved_acceptance_eq31_source_contract,
        raw_score_component_trace_report=resolved_raw_score_component_trace,
    )
    raw_score_component_is_live = bool(
        resolved_acceptance_eq31_source_contract.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-eq31-source-open"
    )
    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceRawScoreComponentContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "first-hop-acceptance-raw-score-component-contract"
        ),
        policy_digest=resolved_acceptance_eq31_source_contract.policy_digest,
        binding_design=resolved_acceptance_eq31_source_contract.binding_design,
        window_label=resolved_acceptance_eq31_source_contract.window_label,
        same_seed_random_states=resolved_acceptance_eq31_source_contract.same_seed_random_states,
        acceptance_eq31_source_driver_signature=resolved_acceptance_eq31_source_contract.driver_signature,
        raw_score_component_driver_signature=resolved_raw_score_component_trace.driver_signature,
        runtime_witness_path=resolved_acceptance_eq31_source_contract.runtime_witness_path,
        acceptance_readout_path=resolved_acceptance_eq31_source_contract.acceptance_readout_path,
        current_witness_floor=resolved_acceptance_eq31_source_contract.current_witness_floor,
        required_min_witness_floor=resolved_acceptance_eq31_source_contract.required_min_witness_floor,
        acceptance_shortfall=resolved_acceptance_eq31_source_contract.acceptance_shortfall,
        current_actionable_slot_random_state=resolved_acceptance_eq31_source_contract.current_actionable_slot_random_state,
        current_actionable_slot_seed_group=resolved_acceptance_eq31_source_contract.current_actionable_slot_seed_group,
        current_actionable_slot_z_index=resolved_acceptance_eq31_source_contract.current_actionable_slot_z_index,
        current_actionable_slot_z_value=resolved_acceptance_eq31_source_contract.current_actionable_slot_z_value,
        binding_slot_random_state=resolved_acceptance_eq31_source_contract.binding_slot_random_state,
        binding_slot_seed_group=resolved_acceptance_eq31_source_contract.binding_slot_seed_group,
        binding_slot_z_index=resolved_acceptance_eq31_source_contract.binding_slot_z_index,
        binding_slot_z_value=resolved_acceptance_eq31_source_contract.binding_slot_z_value,
        residual_slot_random_state=resolved_acceptance_eq31_source_contract.residual_slot_random_state,
        residual_slot_seed_group=resolved_acceptance_eq31_source_contract.residual_slot_seed_group,
        residual_slot_z_index=resolved_acceptance_eq31_source_contract.residual_slot_z_index,
        residual_slot_z_value=resolved_acceptance_eq31_source_contract.residual_slot_z_value,
        binding_replication_seed=resolved_acceptance_eq31_source_contract.binding_replication_seed,
        open_blocker_center_estimate=resolved_acceptance_eq31_source_contract.open_blocker_center_estimate,
        open_blocker_pointwise_lower=resolved_acceptance_eq31_source_contract.open_blocker_pointwise_lower,
        open_blocker_vf_cross_entry=resolved_acceptance_eq31_source_contract.open_blocker_vf_cross_entry,
        eq31_center_estimate=resolved_acceptance_eq31_source_contract.eq31_center_estimate,
        target_truth_center=resolved_raw_score_component_trace.target_truth_center,
        target_raw_score_center_estimate=resolved_raw_score_component_trace.target_raw_score_center_estimate,
        target_rho_delta_y_center_projection=resolved_raw_score_component_trace.target_rho_delta_y_center_projection,
        target_rho_one_minus_pi_phi1_center_projection=resolved_raw_score_component_trace.target_rho_one_minus_pi_phi1_center_projection,
        target_rho_pi_phi0_center_projection=resolved_raw_score_component_trace.target_rho_pi_phi0_center_projection,
        target_delta_y_gap_over_truth=resolved_raw_score_component_trace.target_delta_y_gap_over_truth,
        target_phi1_relief_share_of_delta_y_gap=resolved_raw_score_component_trace.target_phi1_relief_share_of_delta_y_gap,
        target_phi0_reinforcement_share_of_delta_y_gap=resolved_raw_score_component_trace.target_phi0_reinforcement_share_of_delta_y_gap,
        target_raw_score_share_of_delta_y_gap=resolved_raw_score_component_trace.target_raw_score_share_of_delta_y_gap,
        comparator_random_states=resolved_raw_score_component_trace.comparator_random_states,
        comparator_rho_delta_y_center_projections=resolved_raw_score_component_trace.comparator_rho_delta_y_center_projections,
        comparator_rho_one_minus_pi_phi1_center_projections=resolved_raw_score_component_trace.comparator_rho_one_minus_pi_phi1_center_projections,
        comparator_rho_pi_phi0_center_projections=resolved_raw_score_component_trace.comparator_rho_pi_phi0_center_projections,
        raw_score_component_is_live=raw_score_component_is_live,
        raw_score_component_is_historical=not raw_score_component_is_live,
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_acceptance_raw_score_component_contract_digest=(),
    )
    report = replace(
        report,
        canonical_observed_rerun_first_hop_acceptance_raw_score_component_contract_digest=_canonical_digest(
            report=report
        ),
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceRawScoreComponentContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceRawScoreComponentContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract",
]
