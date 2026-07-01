from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportBeforeAfterIntakeContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacket,
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


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
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceBridgeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    admission_order_driver_signature: str
    before_after_driver_signature: str
    before_after_intake_driver_signature: str
    first_hop_contract_driver_signature: str
    first_hop_witness_floor_quota_driver_signature: str
    first_hop_runtime_binding_packet_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    runtime_object_flow_focus: tuple[str, ...]
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    acceptance_shortfall: float
    first_hop_open_witness_floor_gap: float
    binding_slot_share_of_acceptance_shortfall: float
    residual_slot_share_of_acceptance_shortfall: float
    shortfall_is_exactly_first_hop_gap: bool
    next_required_slot_random_state: int | None
    next_required_slot_seed_group: str | None
    next_required_slot_z_index: int | None
    next_required_slot_z_value: float | None
    queued_residual_slot_random_state: int | None
    queued_residual_slot_seed_group: str | None
    queued_residual_slot_z_index: int | None
    queued_residual_slot_z_value: float | None
    binding_runtime_packet: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacket
    driver_signature: str
    canonical_observed_rerun_first_hop_acceptance_bridge_digest: tuple[str, ...]

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
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.before_after_driver_signature = str(
            self.before_after_driver_signature
        ).strip()
        self.before_after_intake_driver_signature = str(
            self.before_after_intake_driver_signature
        ).strip()
        self.first_hop_contract_driver_signature = str(
            self.first_hop_contract_driver_signature
        ).strip()
        self.first_hop_witness_floor_quota_driver_signature = str(
            self.first_hop_witness_floor_quota_driver_signature
        ).strip()
        self.first_hop_runtime_binding_packet_driver_signature = str(
            self.first_hop_runtime_binding_packet_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.runtime_object_flow_focus = tuple(
            str(item).strip() for item in self.runtime_object_flow_focus
        )
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.acceptance_shortfall = float(self.acceptance_shortfall)
        self.first_hop_open_witness_floor_gap = float(
            self.first_hop_open_witness_floor_gap
        )
        self.binding_slot_share_of_acceptance_shortfall = float(
            self.binding_slot_share_of_acceptance_shortfall
        )
        self.residual_slot_share_of_acceptance_shortfall = float(
            self.residual_slot_share_of_acceptance_shortfall
        )
        self.shortfall_is_exactly_first_hop_gap = bool(
            self.shortfall_is_exactly_first_hop_gap
        )
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
        self.queued_residual_slot_random_state = (
            None
            if self.queued_residual_slot_random_state is None
            else int(self.queued_residual_slot_random_state)
        )
        self.queued_residual_slot_seed_group = (
            None
            if self.queued_residual_slot_seed_group is None
            else str(self.queued_residual_slot_seed_group).strip().lower()
        )
        self.queued_residual_slot_z_index = (
            None
            if self.queued_residual_slot_z_index is None
            else int(self.queued_residual_slot_z_index)
        )
        self.queued_residual_slot_z_value = (
            None
            if self.queued_residual_slot_z_value is None
            else float(self.queued_residual_slot_z_value)
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_first_hop_acceptance_bridge_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_first_hop_acceptance_bridge_digest
        )


def _driver_signature(
    *,
    before_after_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport
    ),
    before_after_intake_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportBeforeAfterIntakeContractReport
    ),
    first_hop_runtime_binding_packet_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport
    ),
    first_hop_witness_floor_quota_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport
    ),
    acceptance_shortfall: float,
) -> str:
    shortfall_matches_packet_gap = bool(
        abs(
            float(acceptance_shortfall)
            - float(
                first_hop_runtime_binding_packet_report.binding_runtime_packet.baseline_open_witness_floor_gap
            )
        )
        <= 1e-12
    )
    if (
        before_after_report.driver_signature
        == "same-seed-before-after-acceptance-not-yet-satisfied"
        and before_after_intake_report.driver_signature
        == "same-seed-preserve-left-support-before-after-intake-contract"
        and before_after_intake_report.admission_order_driver_signature
        == "same-seed-admission-order"
        and first_hop_runtime_binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-open"
        and first_hop_witness_floor_quota_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-open"
        and shortfall_matches_packet_gap
        and abs(
            float(
                first_hop_witness_floor_quota_report.binding_slot_share_of_baseline_open_witness_floor_gap
            )
            - 1.0
        )
        <= 1e-12
        and abs(
            float(
                first_hop_witness_floor_quota_report.residual_slot_share_of_baseline_open_witness_floor_gap
            )
        )
        <= 1e-12
        and _slot_matches(
            random_state=first_hop_runtime_binding_packet_report.next_required_slot_random_state,
            z_index=first_hop_runtime_binding_packet_report.next_required_slot_z_index,
            expected_random_state=first_hop_witness_floor_quota_report.binding_slot_random_state,
            expected_z_index=first_hop_witness_floor_quota_report.binding_slot_z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-bridge-open"

    if (
        before_after_report.driver_signature
        == "same-seed-before-after-acceptance-satisfied"
        and first_hop_runtime_binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-landed"
        and abs(float(acceptance_shortfall)) <= 1e-12
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-bridge-residual-only"

    if (
        before_after_report.driver_signature
        == "same-seed-before-after-acceptance-satisfied"
        and first_hop_runtime_binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-closed"
        and abs(float(acceptance_shortfall)) <= 1e-12
    ):
        return (
            "same-seed-exact-witness-observed-rerun-first-hop-acceptance-bridge-closed"
        )

    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-acceptance-bridge"


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceBridgeReport
    ),
) -> tuple[str, ...]:
    packet = report.binding_runtime_packet
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-bridge-closed"
    ):
        return (
            "- exact same-seed before/after replay already satisfies promotion: the acceptance shortfall is `0/9 = 0.000`, so the bridge becomes historical provenance only",
            "- both the seed `303` first hop and the queued seed `707` residual slot are already landed, so there is no remaining acceptance-facing same-line work on this validation-only bridge",
            "- current Trigger 2 implication: keep the helper validation-only and out of live routing surfaces",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-bridge-residual-only"
    ):
        return (
            "- exact same-seed before/after replay has already crossed the acceptance floor, so the remaining acceptance shortfall is `0/9 = 0.000` and only the queued residual cleanup remains",
            "- seed `303` no longer owns an open acceptance gap, while seed `707` stays queued for residual-only cleanup on the same validation-only bridge",
            "- current Trigger 2 implication: keep live entry at `trigger2-policy-spec` and spend the next rerun only on the queued residual slot rather than reopening the first hop",
        )
    return (
        f"- exact same-seed before/after replay still misses promotion by one witness only: current witness floor stays `{_format_ninths(report.current_witness_floor)}` against the required `{_format_ninths(report.required_min_witness_floor)}`, so the remaining acceptance shortfall is exactly `{_format_ninths(report.acceptance_shortfall)}`",
        "- that acceptance shortfall is not a separate live mystery: it matches the validation-only first-hop runtime binding packet's open witness-floor gap exactly, so seed `303` / witness / `z = 0.15` remains the only acceptance-relevant next slot",
        f"- the packet keeps the same object-flow reason machine-readable while acceptance is still open: replication seed `{packet.replication_seed}`, runtime path `{' -> '.join(report.runtime_witness_path)}` with downstream object-flow focus `{' -> '.join(report.runtime_object_flow_focus)}`, `bar_f_at_z0[1] = {_format_float(packet.center_estimate)}`, pointwise lower bound `{_format_float(packet.center_pointwise_interval_lower)}`, and `v_f_hat[2,1] = {_format_float(packet.vf_cross_entry)}` still explain why the first hop must land before any residual spend",
        f"- witness-floor ownership remains one-sided at the acceptance bridge: seed `303` owns `{_format_percent(report.binding_slot_share_of_acceptance_shortfall)}` of the remaining acceptance shortfall while seed `707` / `z = 0.25` stays queued at `{_format_percent(report.residual_slot_share_of_acceptance_shortfall)}` residual-only cleanup",
        "- current Trigger 2 implication: keep live entry at `trigger2-policy-spec`, preserve `same-seed-admission-order` and `same-seed-before-after-acceptance-not-yet-satisfied`, and feed real same-seed observed reruns through this validation-only first-hop acceptance bridge before spending seed `707`",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge_report(
    *,
    before_after_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport
        | None
    ) = None,
    before_after_intake_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportBeforeAfterIntakeContractReport
        | None
    ) = None,
    first_hop_runtime_binding_packet_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport
        | None
    ) = None,
    first_hop_witness_floor_quota_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceBridgeReport:
    resolved_before_after = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe()
        if before_after_report is None
        else before_after_report
    )
    resolved_before_after_intake = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract()
        if before_after_intake_report is None
        else before_after_intake_report
    )
    resolved_first_hop_runtime_binding_packet = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet()
        if first_hop_runtime_binding_packet_report is None
        else first_hop_runtime_binding_packet_report
    )
    resolved_first_hop_witness_floor_quota = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota()
        if first_hop_witness_floor_quota_report is None
        else first_hop_witness_floor_quota_report
    )

    for peer in (
        resolved_before_after_intake,
        resolved_first_hop_runtime_binding_packet,
        resolved_first_hop_witness_floor_quota,
    ):
        if resolved_before_after.policy_digest != peer.policy_digest:
            raise ValueError(
                "observed-rerun first-hop acceptance bridge requires shared policy digest"
            )
        if resolved_before_after.binding_design != peer.binding_design:
            raise ValueError(
                "observed-rerun first-hop acceptance bridge requires shared binding design"
            )
        if resolved_before_after.window_label != peer.window_label:
            raise ValueError(
                "observed-rerun first-hop acceptance bridge requires shared window label"
            )
        if (
            resolved_before_after.same_seed_random_states
            != peer.same_seed_random_states
        ):
            raise ValueError(
                "observed-rerun first-hop acceptance bridge requires shared same-seed random states"
            )

    acceptance_shortfall = max(
        0.0,
        float(resolved_before_after.required_min_witness_floor)
        - float(resolved_before_after.candidate_witness_floor),
    )
    packet = resolved_first_hop_runtime_binding_packet.binding_runtime_packet
    packet_gap = float(packet.baseline_open_witness_floor_gap)
    shortfall_is_exactly_first_hop_gap = bool(
        abs(float(acceptance_shortfall) - float(packet_gap)) <= 1e-12
    )
    binding_share = float(
        resolved_first_hop_witness_floor_quota.binding_slot_share_of_baseline_open_witness_floor_gap
    )
    residual_share = float(
        resolved_first_hop_witness_floor_quota.residual_slot_share_of_baseline_open_witness_floor_gap
    )

    driver_signature = _driver_signature(
        before_after_report=resolved_before_after,
        before_after_intake_report=resolved_before_after_intake,
        first_hop_runtime_binding_packet_report=resolved_first_hop_runtime_binding_packet,
        first_hop_witness_floor_quota_report=resolved_first_hop_witness_floor_quota,
        acceptance_shortfall=acceptance_shortfall,
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceBridgeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-observed-rerun-first-hop-acceptance-bridge"
        ),
        policy_digest=resolved_before_after.policy_digest,
        binding_design=resolved_before_after.binding_design,
        window_label=resolved_before_after.window_label,
        same_seed_random_states=resolved_before_after.same_seed_random_states,
        admission_order_driver_signature=(
            resolved_before_after_intake.admission_order_driver_signature
        ),
        before_after_driver_signature=resolved_before_after.driver_signature,
        before_after_intake_driver_signature=resolved_before_after_intake.driver_signature,
        first_hop_contract_driver_signature=(
            resolved_first_hop_runtime_binding_packet.first_hop_contract_driver_signature
        ),
        first_hop_witness_floor_quota_driver_signature=(
            resolved_first_hop_witness_floor_quota.driver_signature
        ),
        first_hop_runtime_binding_packet_driver_signature=(
            resolved_first_hop_runtime_binding_packet.driver_signature
        ),
        runtime_witness_path=resolved_before_after_intake.runtime_witness_path,
        runtime_object_flow_focus=(
            resolved_first_hop_runtime_binding_packet.runtime_object_flow_focus
        ),
        current_point_miss_vector=resolved_before_after.candidate_point_miss_vector,
        current_band_miss_vector=resolved_before_after.candidate_band_miss_vector,
        current_witness_floor=resolved_before_after.candidate_witness_floor,
        required_min_witness_floor=resolved_before_after.required_min_witness_floor,
        acceptance_shortfall=acceptance_shortfall,
        first_hop_open_witness_floor_gap=packet_gap,
        binding_slot_share_of_acceptance_shortfall=binding_share,
        residual_slot_share_of_acceptance_shortfall=residual_share,
        shortfall_is_exactly_first_hop_gap=shortfall_is_exactly_first_hop_gap,
        next_required_slot_random_state=(
            resolved_first_hop_runtime_binding_packet.next_required_slot_random_state
        ),
        next_required_slot_seed_group=(
            resolved_first_hop_runtime_binding_packet.next_required_slot_seed_group
        ),
        next_required_slot_z_index=(
            resolved_first_hop_runtime_binding_packet.next_required_slot_z_index
        ),
        next_required_slot_z_value=(
            resolved_first_hop_runtime_binding_packet.next_required_slot_z_value
        ),
        queued_residual_slot_random_state=(
            resolved_first_hop_witness_floor_quota.residual_slot_random_state
        ),
        queued_residual_slot_seed_group=(
            resolved_first_hop_witness_floor_quota.residual_slot_seed_group
        ),
        queued_residual_slot_z_index=(
            resolved_first_hop_witness_floor_quota.residual_slot_z_index
        ),
        queued_residual_slot_z_value=(
            resolved_first_hop_witness_floor_quota.residual_slot_z_value
        ),
        binding_runtime_packet=packet,
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_acceptance_bridge_digest=(),
    )
    report = replace(
        report,
        canonical_observed_rerun_first_hop_acceptance_bridge_digest=(
            _canonical_digest(report=report)
        ),
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceBridgeReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge_report()


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge_report_legacy_alias(
    **kwargs,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceBridgeReport:
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge_report(
        **kwargs
    )


# Backward-compatible alias for pre-closeout test helpers that preserved the older CamelCase splice.
build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witnessObservedRerunFirstHopAcceptanceBridgeReport = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge_report_legacy_alias


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceBridgeReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge_report",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witnessObservedRerunFirstHopAcceptanceBridgeReport",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge",
]
