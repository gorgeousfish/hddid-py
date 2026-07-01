from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_live_gap import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_live_gap,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_live_gap import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_live_gap,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    target_gap_driver_signature: str
    binding_slot_live_gap_driver_signature: str
    residual_slot_live_gap_driver_signature: str
    observed_rerun_candidate_status: str
    observed_rerun_rung_status: str
    downstream_evidence_status: str
    acceptance_exact_trim_floor_landing_bridge_driver_signature: str
    acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature: str
    first_hop_landing_guard_driver_signature: str
    current_point_miss_vector: tuple[int, int, int]
    target_point_miss_vector: tuple[int, int, int]
    binding_only_point_miss_vector: tuple[int, int, int]
    completion_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    target_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    projected_binding_witness_floor: float
    completion_witness_floor: float
    required_min_witness_floor: float
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_index: int
    binding_slot_z_value: float
    binding_slot_repair_stage: str
    binding_slot_repair_role: str
    binding_slot_repair_priority: int
    residual_slot_random_state: int
    residual_slot_seed_group: str
    residual_slot_z_index: int
    residual_slot_z_value: float
    residual_slot_repair_stage: str
    residual_slot_repair_role: str
    residual_slot_repair_priority: int
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    binding_replication_seed: int
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_share_of_fold3_low_pi_count: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    driver_signature: str
    canonical_observed_rerun_live_gap_packet_digest: tuple[str, ...]

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
        self.target_gap_driver_signature = str(self.target_gap_driver_signature).strip()
        self.binding_slot_live_gap_driver_signature = str(
            self.binding_slot_live_gap_driver_signature
        ).strip()
        self.residual_slot_live_gap_driver_signature = str(
            self.residual_slot_live_gap_driver_signature
        ).strip()
        self.observed_rerun_candidate_status = str(
            self.observed_rerun_candidate_status
        ).strip()
        self.observed_rerun_rung_status = str(self.observed_rerun_rung_status).strip()
        self.downstream_evidence_status = str(self.downstream_evidence_status).strip()
        self.acceptance_exact_trim_floor_landing_bridge_driver_signature = str(
            self.acceptance_exact_trim_floor_landing_bridge_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature = str(
            self.acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature
        ).strip()
        self.first_hop_landing_guard_driver_signature = str(
            self.first_hop_landing_guard_driver_signature
        ).strip()
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.target_point_miss_vector = tuple(
            int(value) for value in self.target_point_miss_vector
        )
        self.binding_only_point_miss_vector = tuple(
            int(value) for value in self.binding_only_point_miss_vector
        )
        self.completion_point_miss_vector = tuple(
            int(value) for value in self.completion_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.target_band_miss_vector = tuple(
            int(value) for value in self.target_band_miss_vector
        )
        self.current_witness_floor = float(self.current_witness_floor)
        self.projected_binding_witness_floor = float(
            self.projected_binding_witness_floor
        )
        self.completion_witness_floor = float(self.completion_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_seed_group = str(self.binding_slot_seed_group).strip().lower()
        self.binding_slot_z_index = int(self.binding_slot_z_index)
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.binding_slot_repair_stage = str(self.binding_slot_repair_stage).strip()
        self.binding_slot_repair_role = str(self.binding_slot_repair_role).strip()
        self.binding_slot_repair_priority = int(self.binding_slot_repair_priority)
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_seed_group = (
            str(self.residual_slot_seed_group).strip().lower()
        )
        self.residual_slot_z_index = int(self.residual_slot_z_index)
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.residual_slot_repair_stage = str(self.residual_slot_repair_stage).strip()
        self.residual_slot_repair_role = str(self.residual_slot_repair_role).strip()
        self.residual_slot_repair_priority = int(self.residual_slot_repair_priority)
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.target_fold_id = int(self.target_fold_id)
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
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
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_live_gap_packet_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_live_gap_packet_digest
        )

    @property
    def target_witness_floor(self) -> float:
        return self.completion_witness_floor

    @property
    def exact_trim_floor_value(self) -> float:
        return 0.010001


def _ensure_shared_frontier(
    *,
    target_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapReport
    ),
    binding_slot_live_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapReport
    ),
    residual_slot_live_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapReport
    ),
    observed_rerun_rung_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport
    ),
    landing_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport
    ),
) -> None:
    policy_digest = target_gap_report.policy_digest
    binding_design = target_gap_report.binding_design
    window_label = target_gap_report.window_label
    random_states = target_gap_report.same_seed_random_states
    runtime_witness_path = target_gap_report.runtime_witness_path

    for report in (
        binding_slot_live_gap_report,
        residual_slot_live_gap_report,
        observed_rerun_rung_guard_report,
        landing_bridge_report,
    ):
        if report.policy_digest != policy_digest:
            raise ValueError(
                "observed-rerun live-gap packet requires shared policy digest"
            )
        if report.binding_design != binding_design:
            raise ValueError(
                "observed-rerun live-gap packet requires shared binding design"
            )
        if report.window_label != window_label:
            raise ValueError(
                "observed-rerun live-gap packet requires shared window label"
            )
        if report.same_seed_random_states != random_states:
            raise ValueError(
                "observed-rerun live-gap packet requires shared same-seed ordering"
            )
        if report.runtime_witness_path != runtime_witness_path:
            raise ValueError(
                "observed-rerun live-gap packet requires shared runtime witness path"
            )

    if (
        observed_rerun_rung_guard_report.binding_repair_slot.random_state,
        observed_rerun_rung_guard_report.binding_repair_slot.z_index,
    ) != (
        binding_slot_live_gap_report.pending_binding_repair.random_state,
        binding_slot_live_gap_report.pending_binding_repair.z_index,
    ):
        raise ValueError(
            "observed-rerun live-gap packet requires shared seed303 binding slot"
        )
    if (
        observed_rerun_rung_guard_report.residual_repair_slot.random_state,
        observed_rerun_rung_guard_report.residual_repair_slot.z_index,
    ) != (
        residual_slot_live_gap_report.pending_residual_repair.random_state,
        residual_slot_live_gap_report.pending_residual_repair.z_index,
    ):
        raise ValueError(
            "observed-rerun live-gap packet requires shared seed707 residual slot"
        )
    if (
        landing_bridge_report.binding_slot_random_state,
        landing_bridge_report.binding_slot_z_index,
    ) != (
        observed_rerun_rung_guard_report.binding_repair_slot.random_state,
        observed_rerun_rung_guard_report.binding_repair_slot.z_index,
    ):
        raise ValueError(
            "observed-rerun live-gap packet requires landing-bridge seed303 alignment"
        )
    if (
        landing_bridge_report.residual_slot_random_state,
        landing_bridge_report.residual_slot_z_index,
    ) != (
        observed_rerun_rung_guard_report.residual_repair_slot.random_state,
        observed_rerun_rung_guard_report.residual_repair_slot.z_index,
    ):
        raise ValueError(
            "observed-rerun live-gap packet requires landing-bridge seed707 alignment"
        )


def _driver_signature(
    *,
    target_gap_driver_signature: str,
    binding_slot_live_gap_driver_signature: str,
    residual_slot_live_gap_driver_signature: str,
    observed_rerun_candidate_status: str,
    observed_rerun_rung_status: str,
    downstream_evidence_status: str,
    landing_bridge_driver_signature: str,
    landing_guard_driver_signature: str,
) -> str:
    if (
        target_gap_driver_signature == "same-seed-exact-witness-target-gap-open"
        and binding_slot_live_gap_driver_signature
        == "same-seed-exact-witness-binding-slot-live-gap-open"
        and residual_slot_live_gap_driver_signature
        == "same-seed-exact-witness-residual-slot-live-gap-open"
        and observed_rerun_candidate_status
        == "candidate-below-binding-slot-progress-profile"
        and observed_rerun_rung_status == "same-seed-exact-witness-observed-rerun-open"
        and downstream_evidence_status == "same-seed-estimator-evidence-rejected"
        and landing_bridge_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-open"
        and landing_guard_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-landing-guard-open"
    ):
        return "same-seed-exact-witness-observed-rerun-live-gap-packet-open"
    return "mixed-same-seed-exact-witness-observed-rerun-live-gap-packet"


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport
    ),
) -> tuple[str, ...]:
    return (
        f"- the current exact same-seed observed rerun still sits below admissibility in one machine-readable packet: target gap stays `{list(report.current_point_miss_vector)} -> {list(report.target_point_miss_vector)}`, the first executable live gap is still the binding witness slot at seed `303` / `z = 0.15`, and the queued residual live gap remains seed `707` / `z = 0.25`",
        f"- the packet therefore keeps the rung ordering explicit instead of splitting it across helpers: `{report.target_gap_driver_signature}`, `{report.binding_slot_live_gap_driver_signature}`, `{report.residual_slot_live_gap_driver_signature}`, and `{report.observed_rerun_candidate_status}` all describe the same current RED state with witness floor `7/9 = {_format_float(report.current_witness_floor)}` below the required `8/9 = {_format_float(report.required_min_witness_floor)}`",
        f"- the same packet also preserves the acceptance-facing trim-floor landing contract on replication seed `{report.binding_replication_seed}`: the single exact trim-floor row is still `1/13 = 7.7%` of fold-`3` low-`pi_hat` treated rows while carrying `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the gross inverse-`pi_hat` conduit, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, and `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` weighted retention after propensity-floor weighting",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-live-gap-packet-open`; feed real same-seed observed reruns through this validation-only packet before claiming seed `303` is landed or spending the queued residual slot on seed `707` / `z = 0.25`",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet_report(
    *,
    target_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessTargetGapReport
        | None
    ) = None,
    binding_slot_live_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotLiveGapReport
        | None
    ) = None,
    residual_slot_live_gap_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotLiveGapReport
        | None
    ) = None,
    observed_rerun_rung_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport
        | None
    ) = None,
    landing_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport:
    resolved_target_gap = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap()
        if target_gap_report is None
        else target_gap_report
    )
    resolved_binding_slot_live_gap = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_live_gap()
        if binding_slot_live_gap_report is None
        else binding_slot_live_gap_report
    )
    resolved_residual_slot_live_gap = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_live_gap()
        if residual_slot_live_gap_report is None
        else residual_slot_live_gap_report
    )
    resolved_observed_rerun_rung_guard = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard()
        if observed_rerun_rung_guard_report is None
        else observed_rerun_rung_guard_report
    )
    resolved_landing_bridge = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge()
        if landing_bridge_report is None
        else landing_bridge_report
    )

    _ensure_shared_frontier(
        target_gap_report=resolved_target_gap,
        binding_slot_live_gap_report=resolved_binding_slot_live_gap,
        residual_slot_live_gap_report=resolved_residual_slot_live_gap,
        observed_rerun_rung_guard_report=resolved_observed_rerun_rung_guard,
        landing_bridge_report=resolved_landing_bridge,
    )

    driver_signature = _driver_signature(
        target_gap_driver_signature=resolved_target_gap.driver_signature,
        binding_slot_live_gap_driver_signature=resolved_binding_slot_live_gap.driver_signature,
        residual_slot_live_gap_driver_signature=resolved_residual_slot_live_gap.driver_signature,
        observed_rerun_candidate_status=resolved_observed_rerun_rung_guard.current_rung_candidate_status,
        observed_rerun_rung_status=resolved_observed_rerun_rung_guard.resulting_rung_status,
        downstream_evidence_status=resolved_observed_rerun_rung_guard.downstream_evidence_status,
        landing_bridge_driver_signature=resolved_landing_bridge.driver_signature,
        landing_guard_driver_signature=resolved_landing_bridge.first_hop_landing_guard_driver_signature,
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-"
            "preserve-left-support-exact-witness-observed-rerun-live-gap-packet"
        ),
        policy_digest=resolved_target_gap.policy_digest,
        binding_design=resolved_target_gap.binding_design,
        window_label=resolved_target_gap.window_label,
        same_seed_random_states=resolved_target_gap.same_seed_random_states,
        target_gap_driver_signature=resolved_target_gap.driver_signature,
        binding_slot_live_gap_driver_signature=resolved_binding_slot_live_gap.driver_signature,
        residual_slot_live_gap_driver_signature=resolved_residual_slot_live_gap.driver_signature,
        observed_rerun_candidate_status=resolved_observed_rerun_rung_guard.current_rung_candidate_status,
        observed_rerun_rung_status=resolved_observed_rerun_rung_guard.resulting_rung_status,
        downstream_evidence_status=resolved_observed_rerun_rung_guard.downstream_evidence_status,
        acceptance_exact_trim_floor_landing_bridge_driver_signature=resolved_landing_bridge.driver_signature,
        acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature=resolved_landing_bridge.acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature,
        first_hop_landing_guard_driver_signature=resolved_landing_bridge.first_hop_landing_guard_driver_signature,
        current_point_miss_vector=resolved_target_gap.current_point_miss_vector,
        target_point_miss_vector=resolved_target_gap.target_point_miss_vector,
        binding_only_point_miss_vector=resolved_binding_slot_live_gap.binding_only_point_miss_vector,
        completion_point_miss_vector=resolved_residual_slot_live_gap.completion_point_miss_vector,
        current_band_miss_vector=resolved_target_gap.current_band_miss_vector,
        target_band_miss_vector=resolved_target_gap.target_band_miss_vector,
        current_witness_floor=resolved_target_gap.current_witness_floor,
        projected_binding_witness_floor=resolved_binding_slot_live_gap.projected_binding_witness_floor,
        completion_witness_floor=resolved_residual_slot_live_gap.completion_witness_floor,
        required_min_witness_floor=resolved_target_gap.target_witness_floor,
        binding_slot_random_state=resolved_observed_rerun_rung_guard.binding_repair_slot.random_state,
        binding_slot_seed_group=resolved_observed_rerun_rung_guard.binding_repair_slot.seed_group,
        binding_slot_z_index=resolved_observed_rerun_rung_guard.binding_repair_slot.z_index,
        binding_slot_z_value=resolved_observed_rerun_rung_guard.binding_repair_slot.z_value,
        binding_slot_repair_stage="binding-slot-progress",
        binding_slot_repair_role="binding-witness-floor-lift",
        binding_slot_repair_priority=1,
        residual_slot_random_state=resolved_observed_rerun_rung_guard.residual_repair_slot.random_state,
        residual_slot_seed_group=resolved_observed_rerun_rung_guard.residual_repair_slot.seed_group,
        residual_slot_z_index=resolved_observed_rerun_rung_guard.residual_repair_slot.z_index,
        residual_slot_z_value=resolved_observed_rerun_rung_guard.residual_repair_slot.z_value,
        residual_slot_repair_stage="residual-slot-completion",
        residual_slot_repair_role="residual-total-miss-closure",
        residual_slot_repair_priority=2,
        runtime_witness_path=resolved_target_gap.runtime_witness_path,
        acceptance_readout_path=resolved_landing_bridge.acceptance_readout_path,
        binding_replication_seed=resolved_landing_bridge.binding_replication_seed,
        target_fold_id=resolved_landing_bridge.target_fold_id,
        target_exact_trim_floor_count=resolved_landing_bridge.target_exact_trim_floor_count,
        target_exact_trim_floor_share_of_fold3_low_pi_count=resolved_landing_bridge.target_exact_trim_floor_count
        / resolved_observed_rerun_rung_guard.target_fold3_low_pi_treated_count,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=resolved_landing_bridge.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi,
        target_exact_trim_floor_weighted_share_of_fold3_weighted=resolved_landing_bridge.target_exact_trim_floor_weighted_share_of_fold3_weighted,
        target_exact_trim_floor_weighted_retention=resolved_landing_bridge.target_exact_trim_floor_weighted_retention,
        driver_signature=driver_signature,
        canonical_observed_rerun_live_gap_packet_digest=(),
    )
    report.canonical_observed_rerun_live_gap_packet_digest = _canonical_digest(
        report=report
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunLiveGapPacketReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet",
]
