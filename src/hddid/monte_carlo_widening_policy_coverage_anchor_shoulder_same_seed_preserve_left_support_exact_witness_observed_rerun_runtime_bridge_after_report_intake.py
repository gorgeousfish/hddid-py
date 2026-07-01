from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard_report,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
)


_OPEN = "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-intake-open"
_RESIDUAL_ONLY = "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-intake-residual-only"
_CLOSED = (
    "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-intake-closed"
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport:
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
    current_runtime_bridge_state: str
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
    canonical_runtime_bridge_after_report_intake_digest: tuple[str, ...]

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
        self.intake_disposition = (
            str(self.intake_disposition).strip().replace("after_report", "after-report")
        )
        self.binding_only_payload_rejected = bool(self.binding_only_payload_rejected)
        self.completion_after_report_accepted = bool(
            self.completion_after_report_accepted
        )
        self.queued_residual_slot_still_live = bool(
            self.queued_residual_slot_still_live
        )
        self.current_runtime_bridge_state = str(
            self.current_runtime_bridge_state
        ).strip()
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
        self.canonical_runtime_bridge_after_report_intake_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_runtime_bridge_after_report_intake_digest
        )


def _driver_signature(*, intake_disposition: str) -> str:
    normalized = intake_disposition.replace("after_report", "after-report")
    if normalized == "completion-after-report-accepted":
        return _CLOSED
    if normalized == "binding-only-payload-rejected":
        return _RESIDUAL_ONLY
    return _OPEN


def _canonical_digest(
    *,
    report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport,
) -> tuple[str, ...]:
    if report.driver_signature == _CLOSED:
        return (
            f"- canonical completion after_report is already satisfied on the same seed order: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor now holds at `{_format_ninths(report.current_witness_floor)}` while the runtime bridge is closed on the same seed303-before-seed707 packet",
            f"- the same completion keeps the exact trim-floor row `pi_hat = {report.exact_trim_floor_value:.6f}`, `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of gross inverse-`pi_hat` lift, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of weighted burden, `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` retention, replication seed `{report.binding_replication_seed}`, error-to-half-width ratio `{report.center_error_to_half_interval_ratio:.3f}x`, and `v_f_hat[2,1] = {report.vf_cross_entry:.3f}` machine-readable as closed runtime-bridge intake provenance",
            "- current Trigger 2 implication: keep this runtime-bridge after-report intake helper validation-only and out of live routing surfaces once the canonical completion after_report has already closed the same-seed observed-rerun lane; only completion after_report payloads that already flip `same-seed-before-after-acceptance-satisfied` may close the acceptance exact trim-floor runtime bridge",
        )
    if report.driver_signature == _RESIDUAL_ONLY:
        return (
            f"- binding-only payloads still leave before/after acceptance RED: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor therefore still reads `{_format_ninths(report.current_witness_floor)}` even though the binding slot is already consumed",
            "- binding-only payloads that still remain `same-seed-before-after-acceptance-not-yet-satisfied` must keep routing the queued residual seed `707` / `z = 0.25`; the acceptance exact trim-floor runtime bridge therefore stays open until a canonical completion after_report flips the same seed order to `same-seed-before-after-acceptance-satisfied`",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-intake-residual-only`; keep live entry on `trigger2-policy-spec`, keep this helper validation-only, reject binding-only payloads, and only accept completion after_report on the canonical seed303-before-seed707 packet",
        )
    return (
        f"- the runtime bridge after-report intake is still at the fully open same-seed frontier: point miss stays `{list(report.current_point_miss_vector)}`, band miss stays `{list(report.current_band_miss_vector)}`, and witness floor therefore remains `{_format_ninths(report.current_witness_floor)}` while before/after acceptance is still `{report.before_after_driver_signature}`",
        f"- baseline/no-after-report keeps seed `303` / witness / fold `3` / `z = 0.15` as the executable acceptance slot because the exact trim-floor row `pi_hat = {report.exact_trim_floor_value:.6f}` still carries `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the gross inverse-`pi_hat` conduit, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, and `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` retained lift after weighting while queued residual seed `707` / `z = 0.25` stays live",
        f"- the same runtime packet keeps replication seed `{report.binding_replication_seed}`, error-to-half-width ratio `{report.center_error_to_half_interval_ratio:.3f}x`, `v_f_hat[2,1] = {report.vf_cross_entry:.3f}`, runtime witness path `{report.runtime_witness_path[0]} -> {report.runtime_witness_path[1]} -> {report.runtime_witness_path[2]}`, and acceptance readout `{report.acceptance_readout_path[0]} -> {report.acceptance_readout_path[1]} -> {report.acceptance_readout_path[2]}` machine-readable before any after_report is supplied",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-intake-open`; keep live entry on `trigger2-policy-spec`, keep this helper validation-only, reject binding-only payloads, and only accept completion after_report on the canonical seed303-before-seed707 packet",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake_report(
    *,
    after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    source_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeSourceGuardReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport:
    if after_report is not None and source_guard_report is not None:
        raise ValueError(
            "runtime bridge after-report intake helper cannot combine after_report with an explicit source_guard_report"
        )

    resolved_source_guard = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard_report(
            after_report=after_report
        )
        if source_guard_report is None
        else source_guard_report
    )
    driver_signature = _driver_signature(
        intake_disposition=resolved_source_guard.intake_disposition
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "runtime-bridge-after-report-intake"
        ),
        policy_digest=resolved_source_guard.policy_digest,
        binding_design=resolved_source_guard.binding_design,
        window_label=resolved_source_guard.window_label,
        same_seed_random_states=resolved_source_guard.same_seed_random_states,
        after_report_supplied=resolved_source_guard.after_report_supplied,
        before_after_driver_signature=(
            resolved_source_guard.before_after_driver_signature
        ),
        intake_disposition=resolved_source_guard.intake_disposition,
        binding_only_payload_rejected=(
            resolved_source_guard.binding_only_payload_rejected
        ),
        completion_after_report_accepted=(
            resolved_source_guard.completion_after_report_accepted
        ),
        queued_residual_slot_still_live=(
            resolved_source_guard.queued_residual_slot_still_live
        ),
        current_runtime_bridge_state=(
            resolved_source_guard.acceptance_exact_trim_floor_runtime_bridge_driver_signature
        ),
        current_point_miss_vector=resolved_source_guard.current_point_miss_vector,
        current_band_miss_vector=resolved_source_guard.current_band_miss_vector,
        current_witness_floor=resolved_source_guard.current_witness_floor,
        required_min_witness_floor=(resolved_source_guard.required_min_witness_floor),
        acceptance_shortfall=resolved_source_guard.acceptance_shortfall,
        binding_slot_random_state=(resolved_source_guard.binding_slot_random_state),
        binding_slot_seed_group=resolved_source_guard.binding_slot_seed_group,
        binding_slot_z_index=resolved_source_guard.binding_slot_z_index,
        binding_slot_z_value=resolved_source_guard.binding_slot_z_value,
        residual_slot_random_state=(resolved_source_guard.residual_slot_random_state),
        residual_slot_seed_group=resolved_source_guard.residual_slot_seed_group,
        residual_slot_z_index=resolved_source_guard.residual_slot_z_index,
        residual_slot_z_value=resolved_source_guard.residual_slot_z_value,
        runtime_witness_path=resolved_source_guard.runtime_witness_path,
        acceptance_readout_path=resolved_source_guard.acceptance_readout_path,
        exact_trim_floor_value=resolved_source_guard.exact_trim_floor_value,
        target_fold_id=resolved_source_guard.target_fold_id,
        target_exact_trim_floor_count=(
            resolved_source_guard.target_exact_trim_floor_count
        ),
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            resolved_source_guard.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            resolved_source_guard.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            resolved_source_guard.target_exact_trim_floor_weighted_retention
        ),
        binding_replication_seed=resolved_source_guard.binding_replication_seed,
        center_error_to_half_interval_ratio=(
            resolved_source_guard.center_error_to_half_interval_ratio
        ),
        vf_cross_entry=resolved_source_guard.vf_cross_entry,
        driver_signature=driver_signature,
        canonical_runtime_bridge_after_report_intake_digest=(),
    )
    report.canonical_runtime_bridge_after_report_intake_digest = _canonical_digest(
        report=report
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake",
]
