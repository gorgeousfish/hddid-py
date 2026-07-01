from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake_report,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
)


_OPEN = "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-admission-order-open"
_RESIDUAL_ONLY = "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-admission-order-residual-only"
_CLOSED = "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-admission-order-closed"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportAdmissionOrderReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    after_report_supplied: bool
    same_seed_admission_required: bool
    admission_driver_signature: str
    admission_order_preserved: bool
    canonical_seed_order: tuple[int, ...]
    admission_order: tuple[str, ...]
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
    binding_replication_seed: int
    center_error_to_half_interval_ratio: float
    vf_cross_entry: float
    driver_signature: str
    canonical_runtime_bridge_after_report_admission_order_digest: tuple[str, ...]

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
        self.same_seed_admission_required = bool(self.same_seed_admission_required)
        self.admission_driver_signature = str(self.admission_driver_signature).strip()
        self.admission_order_preserved = bool(self.admission_order_preserved)
        self.canonical_seed_order = tuple(
            int(value) for value in self.canonical_seed_order
        )
        self.admission_order = tuple(str(item).strip() for item in self.admission_order)
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
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.center_error_to_half_interval_ratio = float(
            self.center_error_to_half_interval_ratio
        )
        self.vf_cross_entry = float(self.vf_cross_entry)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_runtime_bridge_after_report_admission_order_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_runtime_bridge_after_report_admission_order_digest
        )


def _driver_signature(
    *,
    admission_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport,
    intake_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport,
    admission_order_preserved: bool,
) -> str:
    if (
        not admission_order_preserved
        or not admission_report.same_seed_admission_required
        or admission_report.driver_signature != "same-seed-admission-order"
    ):
        return "mixed-same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-admission-order"
    if intake_report.driver_signature == _CLOSED.replace("admission-order", "intake"):
        return _CLOSED
    if intake_report.driver_signature == _RESIDUAL_ONLY.replace(
        "admission-order", "intake"
    ):
        return _RESIDUAL_ONLY
    if intake_report.driver_signature == _OPEN.replace("admission-order", "intake"):
        return _OPEN
    return "mixed-same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-admission-order"


def _canonical_digest(
    *,
    report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportAdmissionOrderReport,
) -> tuple[str, ...]:
    seed_order = f"{report.canonical_seed_order}"
    if report.driver_signature == _CLOSED:
        return (
            f"- canonical completion after_report now lands on the exact same-seed replay order {seed_order}: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor now holds at `{_format_ninths(report.current_witness_floor)}` while the runtime bridge closes on the same seed303-before-seed707 packet",
            f"- the same completion keeps `same-seed-admission-order`, the exact trim-floor row `pi_hat = {report.exact_trim_floor_value:.6f}`, replication seed `{report.binding_replication_seed}`, error-to-half-width ratio `{report.center_error_to_half_interval_ratio:.3f}x`, `v_f_hat[2,1] = {_format_float(report.vf_cross_entry)}`, runtime witness path `{report.runtime_witness_path[0]} -> {report.runtime_witness_path[1]} -> {report.runtime_witness_path[2]}`, and acceptance readout `{report.acceptance_readout_path[0]} -> {report.acceptance_readout_path[1]} -> {report.acceptance_readout_path[2]}` machine-readable on the closed after-report packet",
            "- current Trigger 2 implication: keep this runtime-bridge after-report admission-order helper validation-only and out of live routing surfaces once the canonical completion after_report has already closed the same-seed observed-rerun lane; only completion after_report payloads that already flip `same-seed-before-after-acceptance-satisfied` may close the acceptance exact trim-floor runtime bridge",
        )
    if report.driver_signature == _RESIDUAL_ONLY:
        return (
            f"- exact same-seed replay order {seed_order} is still preserved, but binding-only payloads still leave before/after acceptance RED: point miss is `{list(report.current_point_miss_vector)}`, band miss is `{list(report.current_band_miss_vector)}`, and witness floor therefore still reads `{_format_ninths(report.current_witness_floor)}` even though the binding slot is already consumed",
            "- `same-seed-admission-order` therefore keeps queue semantics explicit on the higher after-report packet: seed `303` / witness / fold `3` / `z = 0.15` remains the consumed binding witness-floor lift, while queued residual seed `707` / `z = 0.25` must stay residual-only cleanup until a canonical completion after_report flips `same-seed-before-after-acceptance-satisfied` on the same seed order",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-admission-order-residual-only`; keep live entry on `trigger2-policy-spec`, keep this helper validation-only, keep the exact same-seed replay order machine-readable, reject binding-only payloads, and only accept completion after_report on the canonical seed303-before-seed707 packet",
        )
    return (
        f"- exact same-seed replay order {seed_order} still anchors the open after-report packet: point miss stays `{list(report.current_point_miss_vector)}`, band miss stays `{list(report.current_band_miss_vector)}`, and witness floor therefore remains `{_format_ninths(report.current_witness_floor)}` while before/after acceptance is still `{report.before_after_driver_signature}`",
        "- `same-seed-admission-order` keeps the queue explicit on the same higher packet: seed `303` / witness / fold `3` / `z = 0.15` stays the executable binding witness-floor lift, left guard `z = 0.05` remains the no-regression guard, and queued residual seed `707` / `z = 0.25` stays live until completion after_report arrives on the canonical order",
        f"- the same runtime packet keeps the exact trim-floor row `pi_hat = {report.exact_trim_floor_value:.6f}`, replication seed `{report.binding_replication_seed}`, error-to-half-width ratio `{report.center_error_to_half_interval_ratio:.3f}x`, `v_f_hat[2,1] = {_format_float(report.vf_cross_entry)}`, runtime witness path `{report.runtime_witness_path[0]} -> {report.runtime_witness_path[1]} -> {report.runtime_witness_path[2]}`, and acceptance readout `{report.acceptance_readout_path[0]} -> {report.acceptance_readout_path[1]} -> {report.acceptance_readout_path[2]}` machine-readable before any after_report is supplied",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-admission-order-open`; keep live entry on `trigger2-policy-spec`, keep this helper validation-only, keep the exact same-seed replay order machine-readable, reject binding-only payloads, and only accept completion after_report on the canonical seed303-before-seed707 packet",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_admission_order_report(
    *,
    after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    admission_order_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport
        | None
    ) = None,
    after_report_intake_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportIntakeReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportAdmissionOrderReport:
    if after_report is not None and after_report_intake_report is not None:
        raise ValueError(
            "runtime bridge after-report admission-order helper cannot combine after_report with an explicit after_report_intake_report"
        )

    resolved_admission_order = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order()
        if admission_order_report is None
        else admission_order_report
    )
    resolved_after_report_intake = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake_report(
            after_report=after_report
        )
        if after_report_intake_report is None
        else after_report_intake_report
    )

    if (
        resolved_admission_order.policy_digest
        != resolved_after_report_intake.policy_digest
    ):
        raise ValueError(
            "runtime bridge after-report admission-order helper requires shared policy digest"
        )
    if (
        resolved_admission_order.binding_design
        != resolved_after_report_intake.binding_design
    ):
        raise ValueError(
            "runtime bridge after-report admission-order helper requires shared binding design"
        )
    if (
        resolved_admission_order.window_label
        != resolved_after_report_intake.window_label
    ):
        raise ValueError(
            "runtime bridge after-report admission-order helper requires shared window label"
        )
    if (
        resolved_admission_order.same_seed_random_states
        != resolved_after_report_intake.same_seed_random_states
    ):
        raise ValueError(
            "runtime bridge after-report admission-order helper requires shared same-seed random states"
        )

    admission_order_preserved = bool(
        resolved_admission_order.same_seed_admission_required
        and resolved_admission_order.driver_signature == "same-seed-admission-order"
        and resolved_admission_order.same_seed_random_states
        == resolved_after_report_intake.same_seed_random_states
    )

    driver_signature = _driver_signature(
        admission_report=resolved_admission_order,
        intake_report=resolved_after_report_intake,
        admission_order_preserved=admission_order_preserved,
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportAdmissionOrderReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "runtime-bridge-after-report-admission-order"
        ),
        policy_digest=resolved_admission_order.policy_digest,
        binding_design=resolved_admission_order.binding_design,
        window_label=resolved_admission_order.window_label,
        same_seed_random_states=resolved_admission_order.same_seed_random_states,
        after_report_supplied=resolved_after_report_intake.after_report_supplied,
        same_seed_admission_required=(
            resolved_admission_order.same_seed_admission_required
        ),
        admission_driver_signature=resolved_admission_order.driver_signature,
        admission_order_preserved=admission_order_preserved,
        canonical_seed_order=resolved_admission_order.same_seed_random_states,
        admission_order=resolved_admission_order.admission_order,
        before_after_driver_signature=(
            resolved_after_report_intake.before_after_driver_signature
        ),
        intake_disposition=resolved_after_report_intake.intake_disposition,
        binding_only_payload_rejected=(
            resolved_after_report_intake.binding_only_payload_rejected
        ),
        completion_after_report_accepted=(
            resolved_after_report_intake.completion_after_report_accepted
        ),
        queued_residual_slot_still_live=(
            resolved_after_report_intake.queued_residual_slot_still_live
        ),
        current_runtime_bridge_state=(
            resolved_after_report_intake.current_runtime_bridge_state
        ),
        current_point_miss_vector=(
            resolved_after_report_intake.current_point_miss_vector
        ),
        current_band_miss_vector=(
            resolved_after_report_intake.current_band_miss_vector
        ),
        current_witness_floor=resolved_after_report_intake.current_witness_floor,
        required_min_witness_floor=(
            resolved_after_report_intake.required_min_witness_floor
        ),
        acceptance_shortfall=resolved_after_report_intake.acceptance_shortfall,
        binding_slot_random_state=(
            resolved_after_report_intake.binding_slot_random_state
        ),
        binding_slot_seed_group=(resolved_after_report_intake.binding_slot_seed_group),
        binding_slot_z_index=resolved_after_report_intake.binding_slot_z_index,
        binding_slot_z_value=resolved_after_report_intake.binding_slot_z_value,
        residual_slot_random_state=(
            resolved_after_report_intake.residual_slot_random_state
        ),
        residual_slot_seed_group=(
            resolved_after_report_intake.residual_slot_seed_group
        ),
        residual_slot_z_index=(resolved_after_report_intake.residual_slot_z_index),
        residual_slot_z_value=(resolved_after_report_intake.residual_slot_z_value),
        runtime_witness_path=resolved_after_report_intake.runtime_witness_path,
        acceptance_readout_path=(resolved_after_report_intake.acceptance_readout_path),
        exact_trim_floor_value=(resolved_after_report_intake.exact_trim_floor_value),
        binding_replication_seed=(
            resolved_after_report_intake.binding_replication_seed
        ),
        center_error_to_half_interval_ratio=(
            resolved_after_report_intake.center_error_to_half_interval_ratio
        ),
        vf_cross_entry=resolved_after_report_intake.vf_cross_entry,
        driver_signature=driver_signature,
        canonical_runtime_bridge_after_report_admission_order_digest=(),
    )
    report.canonical_runtime_bridge_after_report_admission_order_digest = (
        _canonical_digest(report=report)
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_admission_order() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportAdmissionOrderReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_admission_order_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRuntimeBridgeAfterReportAdmissionOrderReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_admission_order_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_admission_order",
]
