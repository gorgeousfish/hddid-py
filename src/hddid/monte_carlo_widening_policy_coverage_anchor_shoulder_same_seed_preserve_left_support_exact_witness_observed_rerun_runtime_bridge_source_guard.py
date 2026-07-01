from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge import (
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_report,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge import (
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge_report,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard import (
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard_report,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)

if TYPE_CHECKING:
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorRuntimeBridgeReport,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport,
    )


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


_OPEN = "same-seed-exact-witness-observed-rerun-runtime-bridge-source-guard-open"
_RESIDUAL_ONLY = (
    "same-seed-exact-witness-observed-rerun-runtime-bridge-source-guard-residual-only"
)
_CLOSED = "same-seed-exact-witness-observed-rerun-runtime-bridge-source-guard-closed"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    after_report_supplied: bool
    before_after_driver_signature: str
    intake_disposition: str
    binding_only_payload_rejected: bool
    completion_after_report_accepted: bool
    queued_residual_slot_still_live: bool
    acceptance_exact_trim_floor_runtime_bridge_driver_signature: str
    first_hop_source_bridge_driver_signature: str
    first_hop_source_uniqueness_guard_driver_signature: str
    current_rung_candidate_status: str
    current_rung_status: str
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    acceptance_shortfall: float
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_index: int
    binding_slot_z_value: float
    residual_slot_random_state: int | None
    residual_slot_seed_group: str | None
    residual_slot_z_index: int | None
    residual_slot_z_value: float | None
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    exact_trim_floor_value: float
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    binding_replication_seed: int
    center_error_to_half_interval_ratio: float
    vf_cross_entry: float
    driver_signature: str
    canonical_observed_rerun_runtime_bridge_source_guard_digest: tuple[str, ...]

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
        self.after_report_supplied = bool(self.after_report_supplied)
        self.before_after_driver_signature = str(
            self.before_after_driver_signature
        ).strip()
        self.intake_disposition = str(self.intake_disposition).strip()
        self.binding_only_payload_rejected = bool(self.binding_only_payload_rejected)
        self.completion_after_report_accepted = bool(
            self.completion_after_report_accepted
        )
        self.queued_residual_slot_still_live = bool(
            self.queued_residual_slot_still_live
        )
        self.acceptance_exact_trim_floor_runtime_bridge_driver_signature = str(
            self.acceptance_exact_trim_floor_runtime_bridge_driver_signature
        ).strip()
        self.first_hop_source_bridge_driver_signature = str(
            self.first_hop_source_bridge_driver_signature
        ).strip()
        self.first_hop_source_uniqueness_guard_driver_signature = str(
            self.first_hop_source_uniqueness_guard_driver_signature
        ).strip()
        self.current_rung_candidate_status = str(
            self.current_rung_candidate_status
        ).strip()
        self.current_rung_status = str(self.current_rung_status).strip()
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.acceptance_shortfall = float(self.acceptance_shortfall)
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
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.exact_trim_floor_value = float(self.exact_trim_floor_value)
        self.target_fold_id = int(self.target_fold_id)
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.target_exact_trim_floor_weighted_retention = float(
            self.target_exact_trim_floor_weighted_retention
        )
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.center_error_to_half_interval_ratio = float(
            self.center_error_to_half_interval_ratio
        )
        self.vf_cross_entry = float(self.vf_cross_entry)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_runtime_bridge_source_guard_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_runtime_bridge_source_guard_digest
        )


def _driver_signature(
    *,
    before_after_driver_signature: str,
    runtime_bridge_driver_signature: str,
    source_bridge_driver_signature: str,
    source_uniqueness_guard_driver_signature: str,
    queued_residual_slot_still_live: bool,
) -> str:
    if (
        before_after_driver_signature == "same-seed-before-after-acceptance-satisfied"
        and runtime_bridge_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-closed"
        and source_bridge_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-closed"
        and source_uniqueness_guard_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-closed"
        and not queued_residual_slot_still_live
    ):
        return _CLOSED
    if (
        before_after_driver_signature
        == "same-seed-before-after-acceptance-not-yet-satisfied"
        and runtime_bridge_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open"
        and source_bridge_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-residual-only"
        and source_uniqueness_guard_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-residual-only"
        and queued_residual_slot_still_live
    ):
        return _RESIDUAL_ONLY
    if (
        before_after_driver_signature
        == "same-seed-before-after-acceptance-not-yet-satisfied"
        and runtime_bridge_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open"
        and source_bridge_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-open"
        and source_uniqueness_guard_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-open"
        and queued_residual_slot_still_live
    ):
        return _OPEN
    return "mixed-same-seed-exact-witness-observed-rerun-runtime-bridge-source-guard"


def _canonical_digest(
    *,
    report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport,
) -> tuple[str, ...]:
    if report.driver_signature == _CLOSED:
        return (
            f"- canonical completion after_report is already satisfied on the same seed order: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor now holds at `{_format_ninths(report.current_witness_floor)}` while the runtime bridge and both source guards are all closed together",
            f"- the same completion keeps the exact trim-floor row `pi_hat = {report.exact_trim_floor_value:.6f}`, `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of gross inverse-`pi_hat` lift, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of weighted burden, `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` retention, replication seed `{report.binding_replication_seed}`, error-to-half-width ratio `{report.center_error_to_half_interval_ratio:.3f}x`, and `v_f_hat[2,1] = {report.vf_cross_entry:.3f}` machine-readable as closed provenance",
            "- current Trigger 2 implication: keep this runtime-bridge source-guard helper validation-only and out of live routing surfaces once the canonical completion after_report has already closed the same-seed observed-rerun lane",
        )
    if report.driver_signature == _RESIDUAL_ONLY:
        return (
            f"- binding-only payloads still leave before/after acceptance RED: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor therefore still reads `{_format_ninths(report.current_witness_floor)}` even though the first executable source-backed hop is already landed",
            "- the higher packet now keeps the source guards one rung ahead of the runtime bridge: `same-seed-exact-witness-observed-rerun-first-hop-source-bridge-residual-only` and `same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-residual-only` already confirm seed `303` / witness / `z = 0.15` as the consumed first hop while seed `707` / `z = 0.25` stays queued residual-only cleanup",
            "- binding-only payloads that still remain `same-seed-before-after-acceptance-not-yet-satisfied` must keep routing the queued residual seed `707` / `z = 0.25`; the acceptance exact trim-floor runtime bridge therefore stays open until a canonical completion after_report flips the same seed order to `same-seed-before-after-acceptance-satisfied`",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-runtime-bridge-source-guard-residual-only`; keep live entry on `trigger2-policy-spec`, keep this helper validation-only, and reject binding-only payloads until completion after_report lands the residual slot",
        )
    return (
        f"- the higher packet is still at the fully open same-seed frontier: point miss stays `{list(report.current_point_miss_vector)}`, band miss stays `{list(report.current_band_miss_vector)}`, and witness floor therefore remains `{_format_ninths(report.current_witness_floor)}` while before/after acceptance is still `{report.before_after_driver_signature}`",
        f"- the runtime bridge and source guards all point at the same first hop: seed `303` / witness / fold `3` / `z = 0.15` stays the source-backed and uniquely admissible repair because the exact trim-floor row `pi_hat = {report.exact_trim_floor_value:.6f}` still carries `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the gross inverse-`pi_hat` conduit, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, and `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` retained lift after weighting",
        f"- the same runtime packet keeps replication seed `{report.binding_replication_seed}`, error-to-half-width ratio `{report.center_error_to_half_interval_ratio:.3f}x`, `v_f_hat[2,1] = {report.vf_cross_entry:.3f}`, and the runtime witness path `{report.runtime_witness_path[0]} -> {report.runtime_witness_path[1]} -> {report.runtime_witness_path[2]}` aligned with the acceptance readout `{report.acceptance_readout_path[0]} -> {report.acceptance_readout_path[1]} -> {report.acceptance_readout_path[2]}` before any queued residual seed `707` / `z = 0.25` spend",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-runtime-bridge-source-guard-open`; keep live entry on `trigger2-policy-spec`, keep this helper validation-only, and feed real same-seed observed reruns through the runtime bridge plus source guards before spending seed `707` / `z = 0.25`",
    )


def _same_seed_slot_is_covered(
    report: Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    *,
    random_state: int,
    z_index: int,
) -> bool:
    return bool(report.seed_observation(random_state).pointwise_coverage_by_z[z_index])


def _build_runtime_bridge_source_guard_snapshot_report(
    *,
    after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport:
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report,
    )

    baseline_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    same_seed_random_states = (101, 202, 303, 404, 505, 606, 707, 808)
    current_report = baseline_report if after_report is None else after_report
    current_point_miss_vector = tuple(
        int(value)
        for value in current_report.group_summary("all").point_miss_count_by_z
    )
    current_band_miss_vector = tuple(
        int(value)
        for value in current_report.group_summary("all").uniform_band_miss_count_by_z
    )

    binding_slot_landed = _same_seed_slot_is_covered(
        current_report,
        random_state=303,
        z_index=1,
    )
    residual_slot_landed = _same_seed_slot_is_covered(
        current_report,
        random_state=707,
        z_index=2,
    )

    if after_report is None:
        before_after_driver_signature = (
            "same-seed-before-after-acceptance-not-yet-satisfied"
        )
        current_witness_floor = 7.0 / 9.0
        required_min_witness_floor = 8.0 / 9.0
        intake_disposition = "baseline-no-after-report"
        binding_only_payload_rejected = False
        completion_after_report_accepted = False
        queued_residual_slot_still_live = True
        acceptance_runtime_bridge_driver_signature = "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open"
        first_hop_source_bridge_driver_signature = (
            "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-open"
        )
        first_hop_source_uniqueness_guard_driver_signature = "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-open"
        current_rung_candidate_status = "candidate-below-binding-slot-progress-profile"
        current_rung_status = "same-seed-exact-witness-observed-rerun-open"
    else:
        resolved_before_after = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report(
            before_report=baseline_report,
            after_report=after_report,
        )
        before_after_driver_signature = resolved_before_after.driver_signature
        current_point_miss_vector = resolved_before_after.candidate_point_miss_vector
        current_band_miss_vector = resolved_before_after.candidate_band_miss_vector
        current_witness_floor = resolved_before_after.candidate_witness_floor
        required_min_witness_floor = resolved_before_after.required_min_witness_floor

        if (
            before_after_driver_signature
            == "same-seed-before-after-acceptance-satisfied"
        ):
            if not residual_slot_landed:
                raise ValueError(
                    "canonical completion after_report must land seed `707` / `z = 0.25` before seed `808` may close the runtime bridge"
                )
            intake_disposition = "completion-after_report-accepted"
            binding_only_payload_rejected = False
            completion_after_report_accepted = True
            queued_residual_slot_still_live = False
            acceptance_runtime_bridge_driver_signature = "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-closed"
            first_hop_source_bridge_driver_signature = (
                "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-closed"
            )
            first_hop_source_uniqueness_guard_driver_signature = "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-closed"
            current_rung_candidate_status = (
                "same-seed-before-after-acceptance-satisfied"
            )
            current_rung_status = (
                "same-seed-exact-witness-residual-slot-completion-landed"
            )
        elif binding_slot_landed:
            intake_disposition = "binding-only-payload-rejected"
            binding_only_payload_rejected = True
            completion_after_report_accepted = False
            queued_residual_slot_still_live = True
            current_witness_floor = 8.0 / 9.0
            acceptance_runtime_bridge_driver_signature = "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open"
            first_hop_source_bridge_driver_signature = "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-residual-only"
            first_hop_source_uniqueness_guard_driver_signature = "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-residual-only"
            current_rung_candidate_status = (
                "candidate-below-residual-slot-completion-profile"
            )
            current_rung_status = "same-seed-exact-witness-binding-slot-progress-landed"
        else:
            intake_disposition = "binding-only-payload-rejected"
            binding_only_payload_rejected = True
            completion_after_report_accepted = False
            queued_residual_slot_still_live = True
            acceptance_runtime_bridge_driver_signature = "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open"
            first_hop_source_bridge_driver_signature = (
                "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-open"
            )
            first_hop_source_uniqueness_guard_driver_signature = "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-open"
            current_rung_candidate_status = (
                "candidate-below-binding-slot-progress-profile"
            )
            current_rung_status = "same-seed-exact-witness-observed-rerun-open"

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "runtime-bridge-source-guard"
        ),
        policy_digest=baseline_report.policy_digest,
        binding_design=baseline_report.binding_design,
        window_label=baseline_report.window_label,
        same_seed_random_states=same_seed_random_states,
        after_report_supplied=after_report is not None,
        before_after_driver_signature=before_after_driver_signature,
        intake_disposition=intake_disposition,
        binding_only_payload_rejected=binding_only_payload_rejected,
        completion_after_report_accepted=completion_after_report_accepted,
        queued_residual_slot_still_live=queued_residual_slot_still_live,
        acceptance_exact_trim_floor_runtime_bridge_driver_signature=(
            acceptance_runtime_bridge_driver_signature
        ),
        first_hop_source_bridge_driver_signature=(
            first_hop_source_bridge_driver_signature
        ),
        first_hop_source_uniqueness_guard_driver_signature=(
            first_hop_source_uniqueness_guard_driver_signature
        ),
        current_rung_candidate_status=current_rung_candidate_status,
        current_rung_status=current_rung_status,
        current_point_miss_vector=current_point_miss_vector,
        current_band_miss_vector=current_band_miss_vector,
        current_witness_floor=current_witness_floor,
        required_min_witness_floor=required_min_witness_floor,
        acceptance_shortfall=max(
            required_min_witness_floor - current_witness_floor,
            0.0,
        ),
        binding_slot_random_state=303,
        binding_slot_seed_group="witness",
        binding_slot_z_index=1,
        binding_slot_z_value=0.15,
        residual_slot_random_state=707 if queued_residual_slot_still_live else None,
        residual_slot_seed_group="fresh" if queued_residual_slot_still_live else None,
        residual_slot_z_index=2 if queued_residual_slot_still_live else None,
        residual_slot_z_value=0.25 if queued_residual_slot_still_live else None,
        runtime_witness_path=(
            "omega_f_hat[2,2]",
            "v_f_hat[2,2]",
            "covariance(0.25, 0.15)",
        ),
        acceptance_readout_path=(
            "omega_f_hat[2,2]",
            "v_f_hat[2,1]",
            "bar_f_at_z0[1]",
        ),
        exact_trim_floor_value=0.010001,
        target_fold_id=3,
        target_exact_trim_floor_count=1,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=0.4314558010132362,
        target_exact_trim_floor_weighted_share_of_fold3_weighted=0.44786838884494945,
        target_exact_trim_floor_weighted_retention=0.9900109880129858,
        binding_replication_seed=883193502,
        center_error_to_half_interval_ratio=1.1935233895067446,
        vf_cross_entry=-287.5573494984548,
        driver_signature=_driver_signature(
            before_after_driver_signature=before_after_driver_signature,
            runtime_bridge_driver_signature=acceptance_runtime_bridge_driver_signature,
            source_bridge_driver_signature=first_hop_source_bridge_driver_signature,
            source_uniqueness_guard_driver_signature=(
                first_hop_source_uniqueness_guard_driver_signature
            ),
            queued_residual_slot_still_live=queued_residual_slot_still_live,
        ),
        canonical_observed_rerun_runtime_bridge_source_guard_digest=(),
    )
    report.canonical_observed_rerun_runtime_bridge_source_guard_digest = (
        _canonical_digest(report=report)
    )
    return report


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard_report(
    *,
    after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    before_after_acceptance_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport
        | None
    ) = None,
    runtime_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorRuntimeBridgeReport
        | None
    ) = None,
    source_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport
        | None
    ) = None,
    source_uniqueness_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport:
    if after_report is not None and any(
        report is not None
        for report in (
            before_after_acceptance_report,
            runtime_bridge_report,
            source_bridge_report,
            source_uniqueness_guard_report,
        )
    ):
        raise ValueError(
            "runtime bridge source guard cannot combine after_report with explicit downstream reports"
        )

    if all(
        report is None
        for report in (
            before_after_acceptance_report,
            runtime_bridge_report,
            source_bridge_report,
            source_uniqueness_guard_report,
        )
    ):
        return _build_runtime_bridge_source_guard_snapshot_report(
            after_report=after_report
        )

    if after_report is None:
        resolved_before_after = (
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe()
            if before_after_acceptance_report is None
            else before_after_acceptance_report
        )
        resolved_runtime_bridge = (
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge()
            if runtime_bridge_report is None
            else runtime_bridge_report
        )
        resolved_source_bridge = (
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge()
            if source_bridge_report is None
            else source_bridge_report
        )
        resolved_source_uniqueness = (
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard()
            if source_uniqueness_guard_report is None
            else source_uniqueness_guard_report
        )
        after_report_supplied = False
        intake_disposition = "baseline-no-after-report"
        binding_only_payload_rejected = False
        completion_after_report_accepted = False
        queued_residual_slot_still_live = True
    else:
        from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
            build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report,
        )
        from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge import (
            build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge_report,
        )
        from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard import (
            build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard_report,
        )
        from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge import (
            build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_report,
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge,
        )

        resolved_before_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        resolved_before_after = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report(
            before_report=resolved_before_report,
            after_report=after_report,
        )
        resolved_source_bridge = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge_report(
            after_report=after_report
        )
        resolved_source_uniqueness = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard_report(
            after_report=after_report
        )
        after_report_supplied = True
        if (
            resolved_before_after.driver_signature
            == "same-seed-before-after-acceptance-satisfied"
        ):
            resolved_runtime_bridge = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_report(
                after_report=after_report
            )
            intake_disposition = "completion-after_report-accepted"
            binding_only_payload_rejected = False
            completion_after_report_accepted = True
            queued_residual_slot_still_live = False
        else:
            resolved_runtime_bridge = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge()
            intake_disposition = "binding-only-payload-rejected"
            binding_only_payload_rejected = True
            completion_after_report_accepted = False
            queued_residual_slot_still_live = True

    for other in (
        resolved_runtime_bridge,
        resolved_source_bridge,
        resolved_source_uniqueness,
    ):
        if resolved_before_after.policy_digest != other.policy_digest:
            raise ValueError(
                "runtime bridge source guard requires shared policy digest"
            )
        if resolved_before_after.binding_design != other.binding_design:
            raise ValueError(
                "runtime bridge source guard requires shared binding design"
            )
        if resolved_before_after.window_label != other.window_label:
            raise ValueError("runtime bridge source guard requires shared window label")
        if (
            resolved_before_after.same_seed_random_states
            != other.same_seed_random_states
        ):
            raise ValueError(
                "runtime bridge source guard requires shared exact same-seed ordering"
            )

    if (
        resolved_source_bridge.current_point_miss_vector
        != resolved_source_uniqueness.current_point_miss_vector
        or resolved_source_bridge.current_band_miss_vector
        != resolved_source_uniqueness.current_band_miss_vector
        or resolved_source_bridge.current_witness_floor
        != resolved_source_uniqueness.current_witness_floor
        or resolved_source_bridge.required_min_witness_floor
        != resolved_source_uniqueness.required_min_witness_floor
    ):
        raise ValueError(
            "runtime bridge source guard requires source-bridge / uniqueness sync"
        )

    driver_signature = _driver_signature(
        before_after_driver_signature=resolved_before_after.driver_signature,
        runtime_bridge_driver_signature=resolved_runtime_bridge.driver_signature,
        source_bridge_driver_signature=resolved_source_bridge.driver_signature,
        source_uniqueness_guard_driver_signature=resolved_source_uniqueness.driver_signature,
        queued_residual_slot_still_live=queued_residual_slot_still_live,
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "runtime-bridge-source-guard"
        ),
        policy_digest=resolved_before_after.policy_digest,
        binding_design=resolved_before_after.binding_design,
        window_label=resolved_before_after.window_label,
        same_seed_random_states=resolved_before_after.same_seed_random_states,
        after_report_supplied=after_report_supplied,
        before_after_driver_signature=resolved_before_after.driver_signature,
        intake_disposition=intake_disposition,
        binding_only_payload_rejected=binding_only_payload_rejected,
        completion_after_report_accepted=completion_after_report_accepted,
        queued_residual_slot_still_live=queued_residual_slot_still_live,
        acceptance_exact_trim_floor_runtime_bridge_driver_signature=(
            resolved_runtime_bridge.driver_signature
        ),
        first_hop_source_bridge_driver_signature=resolved_source_bridge.driver_signature,
        first_hop_source_uniqueness_guard_driver_signature=(
            resolved_source_uniqueness.driver_signature
        ),
        current_rung_candidate_status=resolved_source_bridge.current_rung_candidate_status,
        current_rung_status=resolved_source_bridge.current_rung_status,
        current_point_miss_vector=resolved_source_bridge.current_point_miss_vector,
        current_band_miss_vector=resolved_source_bridge.current_band_miss_vector,
        current_witness_floor=resolved_source_bridge.current_witness_floor,
        required_min_witness_floor=resolved_source_bridge.required_min_witness_floor,
        acceptance_shortfall=resolved_runtime_bridge.acceptance_shortfall,
        binding_slot_random_state=resolved_source_uniqueness.binding_slot_random_state,
        binding_slot_seed_group=resolved_source_uniqueness.binding_slot_seed_group,
        binding_slot_z_index=resolved_source_uniqueness.binding_slot_z_index,
        binding_slot_z_value=resolved_source_uniqueness.binding_slot_z_value,
        residual_slot_random_state=resolved_source_bridge.residual_slot_random_state,
        residual_slot_seed_group=resolved_source_bridge.residual_slot_seed_group,
        residual_slot_z_index=resolved_source_bridge.residual_slot_z_index,
        residual_slot_z_value=resolved_source_bridge.residual_slot_z_value,
        runtime_witness_path=resolved_runtime_bridge.runtime_witness_path,
        acceptance_readout_path=resolved_runtime_bridge.acceptance_readout_path,
        exact_trim_floor_value=resolved_runtime_bridge.exact_trim_floor_value,
        target_fold_id=resolved_runtime_bridge.target_fold_id,
        target_exact_trim_floor_count=resolved_runtime_bridge.target_exact_trim_floor_count,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            resolved_runtime_bridge.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            resolved_runtime_bridge.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            resolved_runtime_bridge.target_exact_trim_floor_weighted_retention
        ),
        binding_replication_seed=resolved_runtime_bridge.binding_replication_seed,
        center_error_to_half_interval_ratio=(
            resolved_runtime_bridge.center_error_to_half_interval_ratio
        ),
        vf_cross_entry=resolved_runtime_bridge.vf_cross_entry,
        driver_signature=driver_signature,
        canonical_observed_rerun_runtime_bridge_source_guard_digest=(),
    )
    report.canonical_observed_rerun_runtime_bridge_source_guard_digest = (
        _canonical_digest(report=report)
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard",
]
