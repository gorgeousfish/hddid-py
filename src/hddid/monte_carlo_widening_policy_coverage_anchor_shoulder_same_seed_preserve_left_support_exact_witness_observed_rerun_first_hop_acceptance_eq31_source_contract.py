from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceObjectFlowContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_estimation_source_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303EstimationSourceTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_estimation_source_trace,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _format_scientific(value: float) -> str:
    return f"{float(value):.3e}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    acceptance_object_flow_driver_signature: str
    source_trace_driver_signature: str
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
    center_shift_after_inference: float
    bar_gamma_matches_eq31: bool
    score_moment_linf: float
    source_trace_is_live: bool
    source_trace_is_historical: bool
    driver_signature: str
    canonical_observed_rerun_first_hop_acceptance_eq31_source_contract_digest: tuple[
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
        self.acceptance_object_flow_driver_signature = str(
            self.acceptance_object_flow_driver_signature
        ).strip()
        self.source_trace_driver_signature = str(
            self.source_trace_driver_signature
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
        self.center_shift_after_inference = float(self.center_shift_after_inference)
        self.bar_gamma_matches_eq31 = bool(self.bar_gamma_matches_eq31)
        self.score_moment_linf = float(self.score_moment_linf)
        self.source_trace_is_live = bool(self.source_trace_is_live)
        self.source_trace_is_historical = bool(self.source_trace_is_historical)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_first_hop_acceptance_eq31_source_contract_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_observed_rerun_first_hop_acceptance_eq31_source_contract_digest
        )


def _driver_signature(
    *,
    acceptance_object_flow_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceObjectFlowContractReport
    ),
    source_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303EstimationSourceTraceReport
    ),
) -> str:
    source_confirmed = (
        source_trace_report.driver_signature
        == "same-seed-seed303-eq31-point-source-confirmed"
    )
    if (
        acceptance_object_flow_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-object-flow-open"
        and source_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-eq31-source-open"
    if (
        acceptance_object_flow_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-object-flow-residual-only"
        and source_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-eq31-source-residual-only"
    if (
        acceptance_object_flow_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-object-flow-closed"
        and source_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-eq31-source-closed"
    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-acceptance-eq31-source-contract"


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-eq31-source-closed"
    ):
        return (
            "- both the acceptance bridge and the residual slot are already closed, so the acceptance-facing readout and its Eq. (3.1) source now stay as audit-only provenance",
            f"- the landed first hop still keeps the same source packet machine-readable: Eq. (3.1) point estimate `{_format_float(report.eq31_center_estimate)}`, `bar_f_at_z0[1] = {_format_float(report.open_blocker_center_estimate)}`, `v_f_hat[2,1] = {_format_float(report.open_blocker_vf_cross_entry)}`, and `score_moment = {_format_scientific(report.score_moment_linf)}`",
            "- current Trigger 2 implication: keep this acceptance Eq. (3.1) source contract validation-only and out of live routing surfaces once the first-hop lane is fully closed",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-eq31-source-residual-only"
    ):
        return (
            "- the first-hop acceptance shortfall is already closed at `0/9 = 0.000`, so the seed `303` acceptance-facing object flow and its Eq. (3.1) source are now historical provenance rather than a live blocker",
            "- the same historical packet still records that Eq. (3.1) already landed at the seed `303` overshoot before inference, while the queued residual slot at seed `707` / fresh / `z = 0.25` remains the only actionable next spend",
            f"- current Trigger 2 implication: `{report.driver_signature}`; keep live entry at `trigger2-policy-spec`, preserve the historical Eq. (3.1) source packet, and spend the next rerun only on seed `707` residual-only cleanup",
        )
    return (
        f"- the open acceptance shortfall still stays `{_format_ninths(report.acceptance_shortfall)}`, so seed `303` / witness / `z = 0.15` remains the actionable first hop and the acceptance-facing readout `{' -> '.join(report.acceptance_readout_path)}` is still live",
        f"- that live readout is now tied back to Eq. (3.1): the seed `303` point estimate already lands at `{_format_float(report.eq31_center_estimate)}`, `bar_f_at_z0[1] - f_hat_at_z0[1]` stays numerically zero, `bar_gamma_hat` matches `gamma_hat`, and `score_moment` remains at machine precision `{_format_scientific(report.score_moment_linf)}`",
        f"- the open blocker therefore remains source-led rather than band-led: `bar_f_at_z0[1] = {_format_float(report.open_blocker_center_estimate)}`, pointwise lower bound `{_format_float(report.open_blocker_pointwise_lower)}`, and `v_f_hat[2,1] = {_format_float(report.open_blocker_vf_cross_entry)}` still identify the acceptance-facing miss while residual seed `707` stays queued at `0/9 = 0.000`",
        f"- current Trigger 2 implication: `{report.driver_signature}`; keep live entry at `trigger2-policy-spec`, trace the seed `303` witness-center miss backward through Eq. (3.1) inputs first, and spend seed `707` only after the first hop lands",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract_report(
    *,
    acceptance_object_flow_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceObjectFlowContractReport
        | None
    ) = None,
    source_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303EstimationSourceTraceReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport:
    resolved_acceptance_object_flow_contract = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract()
        if acceptance_object_flow_contract_report is None
        else acceptance_object_flow_contract_report
    )
    resolved_source_trace = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_estimation_source_trace()
        if source_trace_report is None
        else source_trace_report
    )

    if (
        resolved_acceptance_object_flow_contract.policy_digest
        != resolved_source_trace.policy_digest
    ):
        raise ValueError(
            "acceptance Eq. (3.1) source contract requires shared policy digest"
        )
    if (
        resolved_acceptance_object_flow_contract.binding_design
        != resolved_source_trace.binding_design
    ):
        raise ValueError(
            "acceptance Eq. (3.1) source contract requires shared binding design"
        )
    if (
        resolved_acceptance_object_flow_contract.window_label
        != resolved_source_trace.window_label
    ):
        raise ValueError(
            "acceptance Eq. (3.1) source contract requires shared window label"
        )
    if (
        resolved_acceptance_object_flow_contract.binding_slot_random_state
        != resolved_source_trace.target_random_state
    ):
        raise ValueError(
            "acceptance Eq. (3.1) source contract expects the source trace target to match the binding slot"
        )
    if (
        resolved_acceptance_object_flow_contract.binding_slot_seed_group
        != resolved_source_trace.target_seed_group
    ):
        raise ValueError(
            "acceptance Eq. (3.1) source contract expects the source trace seed group to match the binding slot"
        )
    if resolved_acceptance_object_flow_contract.acceptance_readout_path != (
        resolved_acceptance_object_flow_contract.runtime_witness_path[0],
        "v_f_hat[2,1]",
        "bar_f_at_z0[1]",
    ):
        raise ValueError(
            "acceptance Eq. (3.1) source contract expects the canonical acceptance readout path"
        )

    driver_signature = _driver_signature(
        acceptance_object_flow_contract_report=resolved_acceptance_object_flow_contract,
        source_trace_report=resolved_source_trace,
    )
    source_trace_is_live = bool(
        resolved_acceptance_object_flow_contract.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-object-flow-open"
    )
    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-observed-rerun-first-hop-acceptance-eq31-source-contract"
        ),
        policy_digest=resolved_acceptance_object_flow_contract.policy_digest,
        binding_design=resolved_acceptance_object_flow_contract.binding_design,
        window_label=resolved_acceptance_object_flow_contract.window_label,
        same_seed_random_states=resolved_acceptance_object_flow_contract.same_seed_random_states,
        acceptance_object_flow_driver_signature=resolved_acceptance_object_flow_contract.driver_signature,
        source_trace_driver_signature=resolved_source_trace.driver_signature,
        runtime_witness_path=resolved_acceptance_object_flow_contract.runtime_witness_path,
        acceptance_readout_path=resolved_acceptance_object_flow_contract.acceptance_readout_path,
        current_witness_floor=resolved_acceptance_object_flow_contract.current_witness_floor,
        required_min_witness_floor=resolved_acceptance_object_flow_contract.required_min_witness_floor,
        acceptance_shortfall=resolved_acceptance_object_flow_contract.acceptance_shortfall,
        current_actionable_slot_random_state=resolved_acceptance_object_flow_contract.current_actionable_slot_random_state,
        current_actionable_slot_seed_group=resolved_acceptance_object_flow_contract.current_actionable_slot_seed_group,
        current_actionable_slot_z_index=resolved_acceptance_object_flow_contract.current_actionable_slot_z_index,
        current_actionable_slot_z_value=resolved_acceptance_object_flow_contract.current_actionable_slot_z_value,
        binding_slot_random_state=resolved_acceptance_object_flow_contract.binding_slot_random_state,
        binding_slot_seed_group=resolved_acceptance_object_flow_contract.binding_slot_seed_group,
        binding_slot_z_index=resolved_acceptance_object_flow_contract.binding_slot_z_index,
        binding_slot_z_value=resolved_acceptance_object_flow_contract.binding_slot_z_value,
        residual_slot_random_state=resolved_acceptance_object_flow_contract.residual_slot_random_state,
        residual_slot_seed_group=resolved_acceptance_object_flow_contract.residual_slot_seed_group,
        residual_slot_z_index=resolved_acceptance_object_flow_contract.residual_slot_z_index,
        residual_slot_z_value=resolved_acceptance_object_flow_contract.residual_slot_z_value,
        binding_replication_seed=resolved_acceptance_object_flow_contract.binding_replication_seed,
        open_blocker_center_estimate=resolved_acceptance_object_flow_contract.open_blocker_center_estimate,
        open_blocker_pointwise_lower=resolved_acceptance_object_flow_contract.open_blocker_pointwise_lower,
        open_blocker_vf_cross_entry=resolved_acceptance_object_flow_contract.open_blocker_vf_cross_entry,
        eq31_center_estimate=resolved_source_trace.target_eq31_center_estimate,
        center_shift_after_inference=resolved_source_trace.target_center_shift_after_inference,
        bar_gamma_matches_eq31=resolved_source_trace.target_bar_gamma_matches_eq31,
        score_moment_linf=resolved_source_trace.target_score_moment_linf,
        source_trace_is_live=source_trace_is_live,
        source_trace_is_historical=not source_trace_is_live,
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_acceptance_eq31_source_contract_digest=(),
    )
    report = replace(
        report,
        canonical_observed_rerun_first_hop_acceptance_eq31_source_contract_digest=_canonical_digest(
            report=report
        ),
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract_report()


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract_report_legacy_alias(
    **kwargs,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport:
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract_report(
        **kwargs
    )


# Backward-compatible alias for pre-closeout test helpers that preserved the older CamelCase splice.
build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_supportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract_report_legacy_alias


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract_report",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_supportExactWitnessObservedRerunFirstHopAcceptanceEq31SourceContractReport",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract",
]
