from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConduitContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet_report,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ratio(value: float) -> str:
    return f"{float(value):.3f}x"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _format_exact_trim_floor(value: float) -> str:
    return f"{float(value):.6f}"


def _seedwise_signature(
    report,
) -> tuple[tuple[int, str, tuple[bool, ...], tuple[bool, ...]], ...]:
    return tuple(
        (
            observation.random_state,
            observation.seed_group,
            tuple(observation.pointwise_coverage_by_z),
            tuple(observation.uniform_band_coverage_by_z),
        )
        for observation in report.seed_observations
    )


def _seedwise_signature_mismatch_random_states(
    *, current_report, canonical_report
) -> tuple[int, ...]:
    current = {
        observation.random_state: (
            tuple(observation.pointwise_coverage_by_z),
            tuple(observation.uniform_band_coverage_by_z),
        )
        for observation in current_report.seed_observations
    }
    canonical = {
        observation.random_state: (
            tuple(observation.pointwise_coverage_by_z),
            tuple(observation.uniform_band_coverage_by_z),
        )
        for observation in canonical_report.seed_observations
    }
    return tuple(
        sorted(
            random_state
            for random_state in canonical
            if current.get(random_state) != canonical.get(random_state)
        )
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorRuntimeBridgeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    acceptance_exact_trim_floor_inverse_pi_conduit_driver_signature: str
    acceptance_exact_trim_floor_landing_bridge_driver_signature: str
    first_hop_runtime_binding_packet_driver_signature: str
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
    exact_trim_floor_value: float
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_fold3_low_pi_treated_count: int
    target_exact_trim_floor_share_of_fold3_low_pi_count: float
    target_exact_trim_floor_inverse_pi_phi1_center_projection: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_phi1_center_projection: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    target_remaining_fold3_low_pi_weighted_retention: float
    comparator_random_states: tuple[int, ...]
    comparator_exact_trim_floor_counts: tuple[int, ...]
    comparator_fold3_inverse_pi_phi1_center_projections: tuple[float, ...]
    center_truth: float
    center_estimate: float
    center_error: float
    center_pointwise_interval_lower: float
    center_pointwise_interval_upper: float
    center_half_interval: float
    center_sigma: float
    center_lower_minus_truth: float
    center_error_to_half_interval_ratio: float
    vf_cross_entry: float
    driver_signature: str
    canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_digest: tuple[
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
        self.acceptance_exact_trim_floor_inverse_pi_conduit_driver_signature = str(
            self.acceptance_exact_trim_floor_inverse_pi_conduit_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_landing_bridge_driver_signature = str(
            self.acceptance_exact_trim_floor_landing_bridge_driver_signature
        ).strip()
        self.first_hop_runtime_binding_packet_driver_signature = str(
            self.first_hop_runtime_binding_packet_driver_signature
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
        self.exact_trim_floor_value = float(self.exact_trim_floor_value)
        self.target_fold_id = int(self.target_fold_id)
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_fold3_low_pi_treated_count = int(
            self.target_fold3_low_pi_treated_count
        )
        self.target_exact_trim_floor_share_of_fold3_low_pi_count = float(
            self.target_exact_trim_floor_share_of_fold3_low_pi_count
        )
        self.target_exact_trim_floor_inverse_pi_phi1_center_projection = float(
            self.target_exact_trim_floor_inverse_pi_phi1_center_projection
        )
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_weighted_phi1_center_projection = float(
            self.target_exact_trim_floor_weighted_phi1_center_projection
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.target_exact_trim_floor_weighted_retention = float(
            self.target_exact_trim_floor_weighted_retention
        )
        self.target_remaining_fold3_low_pi_weighted_retention = float(
            self.target_remaining_fold3_low_pi_weighted_retention
        )
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_exact_trim_floor_counts = tuple(
            int(value) for value in self.comparator_exact_trim_floor_counts
        )
        self.comparator_fold3_inverse_pi_phi1_center_projections = tuple(
            float(value)
            for value in self.comparator_fold3_inverse_pi_phi1_center_projections
        )
        self.center_truth = float(self.center_truth)
        self.center_estimate = float(self.center_estimate)
        self.center_error = float(self.center_error)
        self.center_pointwise_interval_lower = float(
            self.center_pointwise_interval_lower
        )
        self.center_pointwise_interval_upper = float(
            self.center_pointwise_interval_upper
        )
        self.center_half_interval = float(self.center_half_interval)
        self.center_sigma = float(self.center_sigma)
        self.center_lower_minus_truth = float(self.center_lower_minus_truth)
        self.center_error_to_half_interval_ratio = float(
            self.center_error_to_half_interval_ratio
        )
        self.vf_cross_entry = float(self.vf_cross_entry)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_digest
        )


def _driver_signature(
    *,
    conduit_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConduitContractReport
    ),
    landing_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport
    ),
    runtime_binding_packet_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport
    ),
) -> str:
    if (
        conduit_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-open"
        and landing_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-open"
        and runtime_binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-open"
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open"
    if (
        conduit_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-residual-only"
        and landing_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-residual-only"
        and runtime_binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-landed"
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-residual-only"
    if (
        conduit_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-closed"
        and landing_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-closed"
        and runtime_binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-closed"
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-closed"
    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge"


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorRuntimeBridgeReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-closed"
    ):
        return (
            "- both the binding seed `303` hop and the queued seed `707` residual slot are already closed, so the exact-trim-floor runtime bridge now remains validation-only audit provenance inside the same-seed acceptance stack",
            f"- that historical bridge still records the exact trim-floor row `pi_hat = {_format_exact_trim_floor(report.exact_trim_floor_value)}`, `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the gross inverse-`pi_hat` conduit, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` retention, and the runtime packet on replication seed `{report.binding_replication_seed}`",
            "- current Trigger 2 implication: keep this acceptance exact trim-floor runtime bridge validation-only and out of live routing surfaces once the same-seed observed-rerun lane is fully closed",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-residual-only"
    ):
        return (
            "- the binding seed `303` hop has already landed, so the exact-trim-floor runtime bridge now stays as residual-only provenance while seed `707` / fresh / `z = 0.25` remains the only queued residual spend",
            f"- that bridge still records the denominator-led reason for the landed first hop: the exact trim-floor row `pi_hat = {_format_exact_trim_floor(report.exact_trim_floor_value)}` kept `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the gross inverse-`pi_hat` conduit, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, and `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` retention while the runtime packet stays machine-readable on replication seed `{report.binding_replication_seed}`",
            "- current Trigger 2 implication: keep this acceptance exact trim-floor runtime bridge validation-only while the live spend remains the queued residual seed `707` / `z = 0.25`",
        )
    return (
        f"- the open acceptance shortfall still stays `{_format_ninths(report.acceptance_shortfall)}`, and the retained exact trim-floor conduit, the landing bridge, and the runtime binding packet all point at the same actionable repair: seed `303` / witness / fold `3` / `z = 0.15` remains the binding hop before any residual seed `707` spend",
        f"- the same trim-floor row `pi_hat = {_format_exact_trim_floor(report.exact_trim_floor_value)}` still carries `{_format_float(report.target_exact_trim_floor_inverse_pi_phi1_center_projection)}` / `5.010` (`{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}`) of the fold-`3` gross inverse-`pi_hat` conduit and `{_format_float(report.target_exact_trim_floor_weighted_phi1_center_projection)}` / `4.778` (`{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}`) of the weighted burden while retaining `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` of its own gross lift versus `{_format_percent(report.target_remaining_fold3_low_pi_weighted_retention)}` for the remaining rows, so the runtime miss stays denominator-led rather than a generic width-only failure",
        f"- that bridge also keeps the runtime path `{' -> '.join(report.runtime_witness_path)}` and the acceptance-facing readout `{' -> '.join(report.acceptance_readout_path)}` on the same replication seed `{report.binding_replication_seed}`: truth `{_format_float(report.center_truth)}` versus `bar_f_at_z0 = {_format_float(report.center_estimate)}`, lower bound `{_format_float(report.center_pointwise_interval_lower)}` still above truth by `{_format_float(report.center_lower_minus_truth)}`, error-to-half-width ratio `{_format_ratio(report.center_error_to_half_interval_ratio)}`, and `v_f_hat[2,1] = {_format_float(report.vf_cross_entry)}` remain machine-readable",
        f"- comparator seeds `202` and `707` still expose zero exact trim-floor rows inside the same fold-`3` low-`pi_hat` slice and keep full fold-`3` gross inverse-`pi_hat` conduits negative at `{_format_float(report.comparator_fold3_inverse_pi_phi1_center_projections[0])}` and `{_format_float(report.comparator_fold3_inverse_pi_phi1_center_projections[1])}`; current Trigger 2 implication: `{report.driver_signature}`, so keep live entry at `trigger2-policy-spec`, preserve the validation-only exact-trim-floor runtime bridge, and feed the real same-seed observed rerun through this bridge before spending seed `707` / `z = 0.25`",
    )


def _build_runtime_bridge_reports_from_after_report(*, after_report):
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota import (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota_report,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile,
    )
    from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
    )

    baseline_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    before_after_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report(
        before_report=baseline_report,
        after_report=after_report,
    )
    if (
        before_after_report.driver_signature
        != "same-seed-before-after-acceptance-satisfied"
    ):
        raise ValueError(
            "exact trim-floor runtime bridge only accepts an after_report that already flips "
            "`same-seed-before-after-acceptance-satisfied`; the supplied report remains "
            f"`{before_after_report.driver_signature}` and must still route the queued residual "
            "seed `707` / `z = 0.25` before this bridge can turn residual-only or closed"
        )
    canonical_completion_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile()
    if _seedwise_signature(after_report) != _seedwise_signature(
        canonical_completion_report.completion_after_report
    ):
        differing_random_states = _seedwise_signature_mismatch_random_states(
            current_report=after_report,
            canonical_report=canonical_completion_report.completion_after_report,
        )
        differing_seed_labels = ", ".join(
            f"seed `{random_state}`" for random_state in differing_random_states
        )
        raise ValueError(
            "exact trim-floor runtime bridge requires the canonical completion after_report "
            "that lands the queued residual seed `707` / `z = 0.25`; the supplied satisfied "
            "report diverges from the canonical exact-witness target on "
            f"{differing_seed_labels}"
        )
    intake_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_before_after_intake_contract_report(
        before_after_report=before_after_report
    )
    first_hop_contract_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract_report(
        before_report=baseline_report,
        after_report=after_report,
    )
    runtime_binding_packet_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet_report(
        first_hop_contract_report=first_hop_contract_report
    )
    first_hop_witness_floor_quota_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota_report(
        before_report=baseline_report,
        after_report=after_report,
    )
    acceptance_bridge_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_bridge_report(
        before_after_report=before_after_report,
        before_after_intake_report=intake_report,
        first_hop_runtime_binding_packet_report=runtime_binding_packet_report,
        first_hop_witness_floor_quota_report=first_hop_witness_floor_quota_report,
    )
    acceptance_object_flow_contract_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract_report(
        acceptance_bridge_report=acceptance_bridge_report
    )
    acceptance_eq31_source_contract_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract_report(
        acceptance_object_flow_contract_report=acceptance_object_flow_contract_report
    )
    acceptance_raw_score_component_contract_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract_report(
        acceptance_eq31_source_contract_report=acceptance_eq31_source_contract_report
    )
    acceptance_delta_y_support_chain_contract_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract_report(
        acceptance_raw_score_component_contract_report=acceptance_raw_score_component_contract_report
    )
    acceptance_low_pi_support_allocation_contract_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract_report(
        acceptance_delta_y_support_chain_contract_report=acceptance_delta_y_support_chain_contract_report
    )
    concentration_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract_report(
        acceptance_low_pi_support_allocation_contract_report=acceptance_low_pi_support_allocation_contract_report
    )
    landing_guard_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard_report(
        after_report=after_report
    )
    conduit_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract_report(
        acceptance_exact_trim_floor_inverse_pi_concentration_contract_report=concentration_report
    )
    landing_bridge_report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge_report(
        acceptance_exact_trim_floor_inverse_pi_concentration_contract_report=concentration_report,
        first_hop_landing_guard_report=landing_guard_report,
    )
    return conduit_report, landing_bridge_report, runtime_binding_packet_report


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_report(
    *,
    conduit_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConduitContractReport
        | None
    ) = None,
    landing_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport
        | None
    ) = None,
    runtime_binding_packet_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopRuntimeBindingPacketReport
        | None
    ) = None,
    after_report=None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorRuntimeBridgeReport:
    if after_report is not None:
        if any(
            report is not None
            for report in (
                conduit_contract_report,
                landing_bridge_report,
                runtime_binding_packet_report,
            )
        ):
            raise ValueError(
                "exact trim-floor runtime bridge cannot combine after_report with explicit downstream reports"
            )
        (
            resolved_conduit,
            resolved_landing_bridge,
            resolved_runtime_packet,
        ) = _build_runtime_bridge_reports_from_after_report(after_report=after_report)
    else:
        resolved_conduit = (
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract()
            if conduit_contract_report is None
            else conduit_contract_report
        )
        resolved_landing_bridge = (
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge()
            if landing_bridge_report is None
            else landing_bridge_report
        )
        resolved_runtime_packet = (
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet()
            if runtime_binding_packet_report is None
            else runtime_binding_packet_report
        )

    if resolved_conduit.policy_digest != resolved_landing_bridge.policy_digest:
        raise ValueError(
            "exact trim-floor runtime bridge requires shared policy digest"
        )
    if resolved_conduit.policy_digest != resolved_runtime_packet.policy_digest:
        raise ValueError(
            "exact trim-floor runtime bridge requires shared policy digest"
        )
    if resolved_conduit.binding_design != resolved_landing_bridge.binding_design:
        raise ValueError(
            "exact trim-floor runtime bridge requires shared binding design"
        )
    if resolved_conduit.binding_design != resolved_runtime_packet.binding_design:
        raise ValueError(
            "exact trim-floor runtime bridge requires shared binding design"
        )
    if resolved_conduit.window_label != resolved_landing_bridge.window_label:
        raise ValueError("exact trim-floor runtime bridge requires shared window label")
    if resolved_conduit.window_label != resolved_runtime_packet.window_label:
        raise ValueError("exact trim-floor runtime bridge requires shared window label")
    if (
        resolved_conduit.same_seed_random_states
        != resolved_landing_bridge.same_seed_random_states
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires shared exact same-seed ordering"
        )
    if (
        resolved_conduit.same_seed_random_states
        != resolved_runtime_packet.same_seed_random_states
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires shared exact same-seed ordering"
        )
    if (
        resolved_conduit.runtime_witness_path
        != resolved_landing_bridge.runtime_witness_path
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires shared runtime witness path"
        )
    if (
        resolved_conduit.runtime_witness_path
        != resolved_runtime_packet.runtime_witness_path
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires shared runtime witness path"
        )
    if (
        resolved_conduit.acceptance_readout_path
        != resolved_landing_bridge.acceptance_readout_path
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires shared acceptance readout path"
        )
    if (
        tuple(resolved_conduit.acceptance_readout_path[1:])
        != resolved_runtime_packet.runtime_object_flow_focus
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires aligned runtime object-flow focus"
        )
    if (
        resolved_conduit.binding_slot_random_state
        != resolved_landing_bridge.binding_slot_random_state
        or resolved_conduit.binding_slot_z_index
        != resolved_landing_bridge.binding_slot_z_index
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires the same binding slot"
        )
    if (
        resolved_conduit.binding_slot_random_state
        != resolved_runtime_packet.binding_runtime_packet.random_state
        or resolved_conduit.binding_slot_z_index
        != resolved_runtime_packet.binding_runtime_packet.z_index
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires runtime packet alignment on the binding slot"
        )
    if (
        resolved_conduit.residual_slot_random_state
        != resolved_landing_bridge.residual_slot_random_state
        or resolved_conduit.residual_slot_z_index
        != resolved_landing_bridge.residual_slot_z_index
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires the same residual slot"
        )
    if (
        abs(
            resolved_conduit.acceptance_shortfall
            - resolved_landing_bridge.acceptance_shortfall
        )
        > 1e-12
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires aligned acceptance shortfall"
        )
    if (
        resolved_runtime_packet.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-open"
        and abs(
            resolved_conduit.acceptance_shortfall
            - resolved_runtime_packet.binding_runtime_packet.baseline_open_witness_floor_gap
        )
        > 1e-12
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires aligned baseline open witness-floor gap"
        )
    if (
        abs(
            resolved_conduit.current_witness_floor
            - resolved_landing_bridge.current_witness_floor
        )
        > 1e-12
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires aligned current witness floor"
        )
    if (
        abs(
            resolved_conduit.current_witness_floor
            - resolved_landing_bridge.current_witness_floor
        )
        > 1e-12
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires aligned current witness floor"
        )
    if (
        abs(
            resolved_conduit.required_min_witness_floor
            - resolved_landing_bridge.required_min_witness_floor
        )
        > 1e-12
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires aligned required witness floor"
        )
    if (
        resolved_conduit.binding_replication_seed
        != resolved_landing_bridge.binding_replication_seed
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires aligned replication seeds"
        )
    if (
        resolved_conduit.binding_replication_seed
        != resolved_runtime_packet.binding_runtime_packet.replication_seed
    ):
        raise ValueError(
            "exact trim-floor runtime bridge requires aligned runtime packet replication seeds"
        )

    driver_signature = _driver_signature(
        conduit_contract_report=resolved_conduit,
        landing_bridge_report=resolved_landing_bridge,
        runtime_binding_packet_report=resolved_runtime_packet,
    )
    packet = resolved_runtime_packet.binding_runtime_packet
    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorRuntimeBridgeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-"
            "preserve-left-support-exact-witness-observed-rerun-first-hop-"
            "acceptance-exact-trim-floor-runtime-bridge"
        ),
        policy_digest=resolved_conduit.policy_digest,
        binding_design=resolved_conduit.binding_design,
        window_label=resolved_conduit.window_label,
        same_seed_random_states=resolved_conduit.same_seed_random_states,
        acceptance_exact_trim_floor_inverse_pi_conduit_driver_signature=resolved_conduit.driver_signature,
        acceptance_exact_trim_floor_landing_bridge_driver_signature=resolved_landing_bridge.driver_signature,
        first_hop_runtime_binding_packet_driver_signature=resolved_runtime_packet.driver_signature,
        runtime_witness_path=resolved_conduit.runtime_witness_path,
        acceptance_readout_path=resolved_conduit.acceptance_readout_path,
        acceptance_shortfall=resolved_conduit.acceptance_shortfall,
        current_witness_floor=resolved_conduit.current_witness_floor,
        required_min_witness_floor=resolved_conduit.required_min_witness_floor,
        binding_slot_random_state=resolved_conduit.binding_slot_random_state,
        binding_slot_seed_group=resolved_conduit.binding_slot_seed_group,
        binding_slot_z_index=resolved_conduit.binding_slot_z_index,
        binding_slot_z_value=resolved_conduit.binding_slot_z_value,
        residual_slot_random_state=resolved_conduit.residual_slot_random_state,
        residual_slot_seed_group=resolved_conduit.residual_slot_seed_group,
        residual_slot_z_index=resolved_conduit.residual_slot_z_index,
        residual_slot_z_value=resolved_conduit.residual_slot_z_value,
        binding_replication_seed=resolved_conduit.binding_replication_seed,
        exact_trim_floor_value=resolved_conduit.exact_trim_floor_value,
        target_fold_id=resolved_conduit.target_fold_id,
        target_exact_trim_floor_count=resolved_conduit.target_exact_trim_floor_count,
        target_fold3_low_pi_treated_count=resolved_conduit.target_fold3_low_pi_treated_count,
        target_exact_trim_floor_share_of_fold3_low_pi_count=resolved_conduit.target_exact_trim_floor_share_of_fold3_low_pi_count,
        target_exact_trim_floor_inverse_pi_phi1_center_projection=resolved_conduit.target_exact_trim_floor_inverse_pi_phi1_center_projection,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=resolved_conduit.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi,
        target_exact_trim_floor_weighted_phi1_center_projection=resolved_conduit.target_exact_trim_floor_weighted_phi1_center_projection,
        target_exact_trim_floor_weighted_share_of_fold3_weighted=resolved_conduit.target_exact_trim_floor_weighted_share_of_fold3_weighted,
        target_exact_trim_floor_weighted_retention=resolved_conduit.target_exact_trim_floor_weighted_retention,
        target_remaining_fold3_low_pi_weighted_retention=resolved_conduit.target_remaining_fold3_low_pi_weighted_retention,
        comparator_random_states=resolved_conduit.comparator_random_states,
        comparator_exact_trim_floor_counts=resolved_conduit.comparator_exact_trim_floor_counts,
        comparator_fold3_inverse_pi_phi1_center_projections=resolved_conduit.comparator_fold3_inverse_pi_phi1_center_projections,
        center_truth=packet.center_truth,
        center_estimate=packet.center_estimate,
        center_error=packet.center_error,
        center_pointwise_interval_lower=packet.center_pointwise_interval_lower,
        center_pointwise_interval_upper=packet.center_pointwise_interval_upper,
        center_half_interval=packet.center_half_interval,
        center_sigma=packet.center_sigma,
        center_lower_minus_truth=packet.center_lower_minus_truth,
        center_error_to_half_interval_ratio=packet.center_error_to_half_interval_ratio,
        vf_cross_entry=packet.vf_cross_entry,
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_digest=(),
    )
    return replace(
        report,
        canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_digest=_canonical_digest(
            report=report
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorRuntimeBridgeReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge_report()
