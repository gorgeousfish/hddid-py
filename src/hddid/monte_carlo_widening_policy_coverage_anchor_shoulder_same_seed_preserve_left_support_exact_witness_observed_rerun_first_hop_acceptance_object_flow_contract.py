from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceBridgeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_seed303_object_flow_blocker import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_seed303_object_flow_blocker,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _slot_matches(
    *,
    random_state: int | None,
    z_index: int | None,
    expected_random_state: int | None,
    expected_z_index: int | None,
) -> bool:
    if (
        random_state is None
        or z_index is None
        or expected_random_state is None
        or expected_z_index is None
    ):
        return (
            random_state is None
            and z_index is None
            and expected_random_state is None
            and expected_z_index is None
        )
    return bool(
        int(random_state) == int(expected_random_state)
        and int(z_index) == int(expected_z_index)
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceObjectFlowContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    acceptance_bridge_driver_signature: str
    object_flow_blocker_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    current_witness_floor: float
    required_min_witness_floor: float
    acceptance_shortfall: float
    object_flow_blocker_is_live: bool
    object_flow_blocker_is_historical: bool
    current_actionable_slot_random_state: int | None
    current_actionable_slot_seed_group: str | None
    current_actionable_slot_z_index: int | None
    current_actionable_slot_z_value: float | None
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_index: int
    binding_slot_z_value: float
    residual_slot_random_state: int
    residual_slot_seed_group: str
    residual_slot_z_index: int
    residual_slot_z_value: float
    binding_replication_seed: int
    open_blocker_center_estimate: float
    open_blocker_pointwise_lower: float
    open_blocker_vf_cross_entry: float
    driver_signature: str
    canonical_observed_rerun_first_hop_acceptance_object_flow_contract_digest: tuple[
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
        self.acceptance_bridge_driver_signature = str(
            self.acceptance_bridge_driver_signature
        ).strip()
        self.object_flow_blocker_driver_signature = str(
            self.object_flow_blocker_driver_signature
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
        self.object_flow_blocker_is_live = bool(self.object_flow_blocker_is_live)
        self.object_flow_blocker_is_historical = bool(
            self.object_flow_blocker_is_historical
        )
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
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_seed_group = (
            str(self.residual_slot_seed_group).strip().lower()
        )
        self.residual_slot_z_index = int(self.residual_slot_z_index)
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.open_blocker_center_estimate = float(self.open_blocker_center_estimate)
        self.open_blocker_pointwise_lower = float(self.open_blocker_pointwise_lower)
        self.open_blocker_vf_cross_entry = float(self.open_blocker_vf_cross_entry)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_first_hop_acceptance_object_flow_contract_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_observed_rerun_first_hop_acceptance_object_flow_contract_digest
        )


def _driver_signature(
    *,
    acceptance_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceBridgeReport
    ),
    object_flow_blocker_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerReport
    ),
) -> str:
    blocker_signature = (
        "same-seed-exact-witness-seed303-pointwise-overshoot-object-flow-blocker"
    )
    if (
        acceptance_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-bridge-open"
        and object_flow_blocker_report.driver_signature == blocker_signature
        and _slot_matches(
            random_state=acceptance_bridge_report.next_required_slot_random_state,
            z_index=acceptance_bridge_report.next_required_slot_z_index,
            expected_random_state=object_flow_blocker_report.target_random_state,
            expected_z_index=object_flow_blocker_report.target_z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-object-flow-open"
    if (
        acceptance_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-bridge-residual-only"
        and object_flow_blocker_report.driver_signature == blocker_signature
        and _slot_matches(
            random_state=acceptance_bridge_report.next_required_slot_random_state,
            z_index=acceptance_bridge_report.next_required_slot_z_index,
            expected_random_state=acceptance_bridge_report.queued_residual_slot_random_state,
            expected_z_index=acceptance_bridge_report.queued_residual_slot_z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-object-flow-residual-only"
    if (
        acceptance_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-bridge-closed"
        and object_flow_blocker_report.driver_signature == blocker_signature
        and acceptance_bridge_report.next_required_slot_random_state is None
        and acceptance_bridge_report.next_required_slot_z_index is None
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-object-flow-closed"
    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-acceptance-object-flow-contract"


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceObjectFlowContractReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-object-flow-closed"
    ):
        return (
            "- the acceptance bridge is already closed, so the acceptance-facing readout `omega_f_hat[2,2] -> v_f_hat[2,1] -> bar_f_at_z0[1]` now stays as historical provenance only rather than a live queueing aid",
            f"- the seed `303` object-flow blocker is therefore historical too: replication seed `{report.binding_replication_seed}`, `bar_f_at_z0[1] = {_format_float(report.open_blocker_center_estimate)}`, `v_f_hat[2,1] = {_format_float(report.open_blocker_vf_cross_entry)}`, and pointwise lower bound `{_format_float(report.open_blocker_pointwise_lower)}` remain audit evidence for the first hop that already landed",
            "- current Trigger 2 implication: keep this acceptance object-flow contract validation-only and out of live routing surfaces once the residual slot is also closed",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-object-flow-residual-only"
    ):
        return (
            "- the first-hop acceptance shortfall is already closed at `0/9 = 0.000`, so the seed `303` acceptance-facing object flow is no longer live even though the residual slot at seed `707` / fresh / `z = 0.25` remains queued",
            "- the acceptance-facing readout `omega_f_hat[2,2] -> v_f_hat[2,1] -> bar_f_at_z0[1]` therefore stays machine-readable as historical first-hop provenance rather than a fresh blocker, while residual-only cleanup remains the only actionable next slot",
            f"- current Trigger 2 implication: `{report.driver_signature}`; keep live entry at `trigger2-policy-spec`, preserve the historical seed `303` object-flow blocker, and spend the next rerun only on the queued residual slot",
        )
    return (
        "- the open acceptance bridge still carries the exact shortfall `1/9 = 0.111`, so seed `303` / witness / `z = 0.15` remains the actionable first hop while seed `707` / fresh / `z = 0.25` stays queued residual-only cleanup",
        "- the acceptance-facing object-flow readout is now ordered end-to-end as `omega_f_hat[2,2] -> v_f_hat[2,1] -> bar_f_at_z0[1]`, which keeps the preserve-left-support runtime path tied directly to the open seed `303` readout that still sits above truth",
        f"- that open readout is still blocked by the live seed `303` object-flow blocker rather than generic band widening: `bar_f_at_z0[1] = {_format_float(report.open_blocker_center_estimate)}`, `v_f_hat[2,1] = {_format_float(report.open_blocker_vf_cross_entry)}`, and the pointwise lower bound `{_format_float(report.open_blocker_pointwise_lower)}` remains above truth while the uniform band still covers the slot",
        f"- current Trigger 2 implication: `{report.driver_signature}`; keep live entry at `trigger2-policy-spec`, clear the seed `303` acceptance-facing object flow first, and spend seed `707` only after the first hop lands",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract_report(
    *,
    acceptance_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceBridgeReport
        | None
    ) = None,
    object_flow_blocker_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceObjectFlowContractReport:
    resolved_acceptance_bridge = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge()
        if acceptance_bridge_report is None
        else acceptance_bridge_report
    )
    resolved_object_flow_blocker = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_seed303_object_flow_blocker()
        if object_flow_blocker_report is None
        else object_flow_blocker_report
    )

    if (
        resolved_acceptance_bridge.policy_digest
        != resolved_object_flow_blocker.policy_digest
    ):
        raise ValueError(
            "acceptance object-flow contract requires shared policy digest"
        )
    if (
        resolved_acceptance_bridge.binding_design
        != resolved_object_flow_blocker.binding_design
    ):
        raise ValueError(
            "acceptance object-flow contract requires shared binding design"
        )
    if (
        resolved_acceptance_bridge.window_label
        != resolved_object_flow_blocker.window_label
    ):
        raise ValueError("acceptance object-flow contract requires shared window label")
    if (
        resolved_acceptance_bridge.same_seed_random_states
        != resolved_object_flow_blocker.same_seed_random_states
    ):
        raise ValueError(
            "acceptance object-flow contract requires shared same-seed random states"
        )
    if (
        resolved_acceptance_bridge.runtime_witness_path
        != resolved_object_flow_blocker.runtime_witness_path
    ):
        raise ValueError(
            "acceptance object-flow contract requires shared runtime witness path"
        )
    if resolved_object_flow_blocker.vf_cross_entry_label != "v_f_hat[2,1]":
        raise ValueError(
            "acceptance object-flow contract expects the blocker to read v_f_hat[2,1]"
        )
    if (
        resolved_acceptance_bridge.binding_runtime_packet.random_state
        != resolved_object_flow_blocker.target_random_state
    ):
        raise ValueError(
            "acceptance object-flow contract expects the blocker target to match the binding slot"
        )
    if (
        resolved_acceptance_bridge.binding_runtime_packet.seed_group
        != resolved_object_flow_blocker.target_seed_group
    ):
        raise ValueError(
            "acceptance object-flow contract expects the blocker seed group to match the binding slot"
        )
    if (
        resolved_acceptance_bridge.binding_runtime_packet.z_index
        != resolved_object_flow_blocker.target_z_index
    ):
        raise ValueError(
            "acceptance object-flow contract expects the blocker z index to match the binding slot"
        )
    if (
        abs(
            float(resolved_acceptance_bridge.binding_runtime_packet.z_value)
            - float(resolved_object_flow_blocker.target_z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "acceptance object-flow contract expects the blocker z value to match the binding slot"
        )

    acceptance_readout_path = (
        resolved_acceptance_bridge.runtime_witness_path[0],
        *resolved_acceptance_bridge.runtime_object_flow_focus,
    )
    driver_signature = _driver_signature(
        acceptance_bridge_report=resolved_acceptance_bridge,
        object_flow_blocker_report=resolved_object_flow_blocker,
    )
    blocker_is_live = bool(
        resolved_acceptance_bridge.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-bridge-open"
    )
    actionable_slot_random_state = (
        resolved_acceptance_bridge.next_required_slot_random_state
    )
    actionable_slot_seed_group = (
        resolved_acceptance_bridge.next_required_slot_seed_group
    )
    actionable_slot_z_index = resolved_acceptance_bridge.next_required_slot_z_index
    actionable_slot_z_value = resolved_acceptance_bridge.next_required_slot_z_value

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceObjectFlowContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-observed-rerun-first-hop-acceptance-object-flow-contract"
        ),
        policy_digest=resolved_acceptance_bridge.policy_digest,
        binding_design=resolved_acceptance_bridge.binding_design,
        window_label=resolved_acceptance_bridge.window_label,
        same_seed_random_states=resolved_acceptance_bridge.same_seed_random_states,
        acceptance_bridge_driver_signature=resolved_acceptance_bridge.driver_signature,
        object_flow_blocker_driver_signature=resolved_object_flow_blocker.driver_signature,
        runtime_witness_path=resolved_acceptance_bridge.runtime_witness_path,
        acceptance_readout_path=acceptance_readout_path,
        current_witness_floor=resolved_acceptance_bridge.current_witness_floor,
        required_min_witness_floor=resolved_acceptance_bridge.required_min_witness_floor,
        acceptance_shortfall=resolved_acceptance_bridge.acceptance_shortfall,
        object_flow_blocker_is_live=blocker_is_live,
        object_flow_blocker_is_historical=not blocker_is_live,
        current_actionable_slot_random_state=actionable_slot_random_state,
        current_actionable_slot_seed_group=actionable_slot_seed_group,
        current_actionable_slot_z_index=actionable_slot_z_index,
        current_actionable_slot_z_value=actionable_slot_z_value,
        binding_slot_random_state=resolved_acceptance_bridge.binding_runtime_packet.random_state,
        binding_slot_seed_group=resolved_acceptance_bridge.binding_runtime_packet.seed_group,
        binding_slot_z_index=resolved_acceptance_bridge.binding_runtime_packet.z_index,
        binding_slot_z_value=resolved_acceptance_bridge.binding_runtime_packet.z_value,
        residual_slot_random_state=resolved_acceptance_bridge.queued_residual_slot_random_state,
        residual_slot_seed_group=resolved_acceptance_bridge.queued_residual_slot_seed_group,
        residual_slot_z_index=resolved_acceptance_bridge.queued_residual_slot_z_index,
        residual_slot_z_value=resolved_acceptance_bridge.queued_residual_slot_z_value,
        binding_replication_seed=resolved_acceptance_bridge.binding_runtime_packet.replication_seed,
        open_blocker_center_estimate=resolved_object_flow_blocker.target_bar_f_at_z,
        open_blocker_pointwise_lower=resolved_object_flow_blocker.target_pointwise_lower,
        open_blocker_vf_cross_entry=resolved_object_flow_blocker.target_vf_cross_entry,
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_acceptance_object_flow_contract_digest=(),
    )
    report = replace(
        report,
        canonical_observed_rerun_first_hop_acceptance_object_flow_contract_digest=_canonical_digest(
            report=report
        ),
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceObjectFlowContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceObjectFlowContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract",
]
