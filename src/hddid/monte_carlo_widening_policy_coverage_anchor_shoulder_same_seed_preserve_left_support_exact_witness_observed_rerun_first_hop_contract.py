from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard_report,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
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


def _driver_signature(
    *,
    rung_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport
    ),
    source_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport
    ),
    uniqueness_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport
    ),
    witness_floor_quota_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport
    ),
) -> str:
    one_sided_quota = bool(
        abs(
            float(
                witness_floor_quota_report.binding_slot_share_of_baseline_open_witness_floor_gap
            )
            - 1.0
        )
        <= 1e-12
        and abs(
            float(
                witness_floor_quota_report.residual_slot_share_of_baseline_open_witness_floor_gap
            )
        )
        <= 1e-12
    )
    shared_residual_slot = _slot_matches(
        random_state=source_bridge_report.residual_slot_random_state,
        z_index=source_bridge_report.residual_slot_z_index,
        expected_random_state=witness_floor_quota_report.residual_slot_random_state,
        expected_z_index=witness_floor_quota_report.residual_slot_z_index,
    )
    if not (shared_residual_slot and one_sided_quota):
        return "mixed-same-seed-exact-witness-observed-rerun-first-hop-contract"

    if (
        rung_guard_report.resulting_rung_status
        == "same-seed-exact-witness-residual-slot-completion-landed"
        and source_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-closed"
        and uniqueness_guard_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-closed"
        and witness_floor_quota_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-closed"
        and rung_guard_report.next_required_slot is None
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-contract-closed"

    if (
        rung_guard_report.resulting_rung_status
        == "same-seed-exact-witness-binding-slot-progress-landed"
        and source_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-residual-only"
        and uniqueness_guard_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-residual-only"
        and witness_floor_quota_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-residual-only"
        and _slot_matches(
            random_state=source_bridge_report.next_required_slot_random_state,
            z_index=source_bridge_report.next_required_slot_z_index,
            expected_random_state=source_bridge_report.residual_slot_random_state,
            expected_z_index=source_bridge_report.residual_slot_z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-contract-residual-only"

    if (
        rung_guard_report.resulting_rung_status
        == "same-seed-exact-witness-observed-rerun-open"
        and source_bridge_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-bridge-open"
        and uniqueness_guard_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-open"
        and witness_floor_quota_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-open"
        and rung_guard_report.next_required_slot is not None
        and _slot_matches(
            random_state=source_bridge_report.next_required_slot_random_state,
            z_index=source_bridge_report.next_required_slot_z_index,
            expected_random_state=witness_floor_quota_report.binding_slot_random_state,
            expected_z_index=witness_floor_quota_report.binding_slot_z_index,
        )
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-contract-open"

    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    current_rung_candidate_status: str
    current_rung_status: str
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    first_hop_source_bridge_driver_signature: str
    first_hop_source_uniqueness_guard_driver_signature: str
    first_hop_witness_floor_quota_driver_signature: str
    next_required_slot_random_state: int | None
    next_required_slot_seed_group: str | None
    next_required_slot_z_index: int | None
    next_required_slot_z_value: float | None
    next_required_slot_repair_stage: str | None
    next_required_slot_repair_role: str | None
    next_required_slot_repair_priority: int | None
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
    baseline_open_witness_floor_gap: float
    binding_slot_witness_floor_increment: float
    residual_slot_witness_floor_increment: float
    binding_slot_share_of_baseline_open_witness_floor_gap: float
    residual_slot_share_of_baseline_open_witness_floor_gap: float
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    comparator_random_states: tuple[int, int]
    comparator_exact_trim_floor_counts: tuple[int, int]
    comparator_fold3_inverse_pi_phi1_center_projections: tuple[float, float]
    driver_signature: str
    canonical_observed_rerun_first_hop_contract_digest: tuple[str, ...]

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
        self.first_hop_source_bridge_driver_signature = str(
            self.first_hop_source_bridge_driver_signature
        ).strip()
        self.first_hop_source_uniqueness_guard_driver_signature = str(
            self.first_hop_source_uniqueness_guard_driver_signature
        ).strip()
        self.first_hop_witness_floor_quota_driver_signature = str(
            self.first_hop_witness_floor_quota_driver_signature
        ).strip()
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
        self.next_required_slot_repair_stage = (
            None
            if self.next_required_slot_repair_stage is None
            else str(self.next_required_slot_repair_stage).strip()
        )
        self.next_required_slot_repair_role = (
            None
            if self.next_required_slot_repair_role is None
            else str(self.next_required_slot_repair_role).strip()
        )
        self.next_required_slot_repair_priority = (
            None
            if self.next_required_slot_repair_priority is None
            else int(self.next_required_slot_repair_priority)
        )
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
        self.baseline_open_witness_floor_gap = float(
            self.baseline_open_witness_floor_gap
        )
        self.binding_slot_witness_floor_increment = float(
            self.binding_slot_witness_floor_increment
        )
        self.residual_slot_witness_floor_increment = float(
            self.residual_slot_witness_floor_increment
        )
        self.binding_slot_share_of_baseline_open_witness_floor_gap = float(
            self.binding_slot_share_of_baseline_open_witness_floor_gap
        )
        self.residual_slot_share_of_baseline_open_witness_floor_gap = float(
            self.residual_slot_share_of_baseline_open_witness_floor_gap
        )
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
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_first_hop_contract_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_first_hop_contract_digest
        )


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-contract-closed"
    ):
        return (
            f"- observed same-seed rerun has already consumed the full first-hop contract: point miss holds at `{list(report.current_point_miss_vector)}`, band miss holds at `{list(report.current_band_miss_vector)}`, and witness floor holds at `{_format_ninths(report.current_witness_floor)}`",
            "- the ladder guard, source bridge, uniqueness guard, and witness-floor quota are now all historical because both seed `303` / `z = 0.15` and seed `707` / `z = 0.25` are already landed",
            "- no further same-line first-hop alignment work remains on this validation-only object",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-contract-closed`; keep this helper validation-only and out of live routing surfaces",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-contract-residual-only"
    ):
        return (
            f"- observed same-seed rerun has already consumed the seed `303` first-hop contract: point miss now sits at `{list(report.current_point_miss_vector)}`, band miss stays `{list(report.current_band_miss_vector)}`, and witness floor already sits at `{_format_ninths(report.current_witness_floor)}`",
            "- the ladder guard, source bridge, uniqueness guard, and witness-floor quota all agree that the admissibility-facing first hop is closed and only the residual seed `707` / `z = 0.25` cleanup remains",
            "- the quota split stays historical and one-sided: seed `303` already supplied the full `1/9 = 0.111` / `100.0%` witness-floor lift while seed `707` stayed `0/9 = 0.000` witness-floor-neutral",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-contract-residual-only`; keep this helper validation-only and route the next rerun only to the residual slot",
        )
    return (
        f"- observed same-seed rerun keeps the full seed303 first-hop stack aligned at the ladder baseline: point miss stays `{list(report.current_point_miss_vector)}`, band miss stays `{list(report.current_band_miss_vector)}`, and witness floor therefore stays `{_format_ninths(report.current_witness_floor)}`",
        "- the single ladder guard, first-hop source bridge, first-hop source uniqueness guard, and first-hop witness-floor quota all point to the same actionable slot: seed `303` / witness / `z = 0.15` stays priority `1` / `binding-witness-floor-lift`, while seed `707` / fresh / `z = 0.25` remains priority `2` / `residual-total-miss-closure`",
        "- the contract is one-sided and source-backed: seed `303` keeps the only exact trim-floor row in fold `3`, owns "
        f"`{_format_ninths(report.binding_slot_witness_floor_increment)}` / "
        f"`{_format_percent(report.binding_slot_share_of_baseline_open_witness_floor_gap)}` of the open witness-floor gap, and comparator seeds `202` and `707` keep zero exact trim-floor rows with full fold-`3` gross inverse-`pi_hat` conduits "
        f"`{_format_float(report.comparator_fold3_inverse_pi_phi1_center_projections[0])}` / "
        f"`{_format_float(report.comparator_fold3_inverse_pi_phi1_center_projections[1])}`",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-contract-open`; feed real same-seed reruns through this validation-only first-hop contract before spending the queued residual slot on seed `707` / `z = 0.25`",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract_report(
    *,
    after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    before_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    rung_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRungGuardReport
        | None
    ) = None,
    source_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceBridgeReport
        | None
    ) = None,
    uniqueness_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport
        | None
    ) = None,
    witness_floor_quota_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport:
    resolved_after = (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        if after_report is None
        else after_report
    )
    resolved_before = (
        run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
        if before_report is None
        else before_report
    )
    resolved_rung_guard = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard()
        if rung_guard_report is None and after_report is None and before_report is None
        else (
            build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard_report(
                after_report=resolved_after,
                before_report=resolved_before,
            )
            if rung_guard_report is None
            else rung_guard_report
        )
    )
    resolved_source_bridge = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge_report(
            after_report=resolved_after,
            before_report=resolved_before,
        )
        if source_bridge_report is None
        else source_bridge_report
    )
    resolved_uniqueness_guard = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard_report(
            after_report=resolved_after,
            before_report=resolved_before,
            source_bridge_report=resolved_source_bridge,
        )
        if uniqueness_guard_report is None
        else uniqueness_guard_report
    )
    resolved_witness_floor_quota = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota_report(
            after_report=resolved_after,
            before_report=resolved_before,
            uniqueness_guard_report=resolved_uniqueness_guard,
        )
        if witness_floor_quota_report is None
        else witness_floor_quota_report
    )

    for report in (
        resolved_source_bridge,
        resolved_uniqueness_guard,
        resolved_witness_floor_quota,
    ):
        if resolved_rung_guard.policy_digest != report.policy_digest:
            raise ValueError("first-hop contract requires shared policy digest")
        if resolved_rung_guard.binding_design != report.binding_design:
            raise ValueError("first-hop contract requires shared binding design")
        if resolved_rung_guard.window_label != report.window_label:
            raise ValueError("first-hop contract requires shared window label")
        if (
            resolved_rung_guard.same_seed_random_states
            != report.same_seed_random_states
        ):
            raise ValueError(
                "first-hop contract requires shared exact same-seed ordering"
            )

    if (
        resolved_rung_guard.current_rung_candidate_status
        != resolved_source_bridge.current_rung_candidate_status
        or resolved_rung_guard.current_rung_candidate_status
        != resolved_uniqueness_guard.current_rung_candidate_status
    ):
        raise ValueError(
            "first-hop contract requires rung guard, source bridge, and uniqueness guard candidate status sync"
        )
    if (
        resolved_rung_guard.resulting_rung_status
        != resolved_source_bridge.current_rung_status
        or resolved_rung_guard.resulting_rung_status
        != resolved_uniqueness_guard.current_rung_status
        or resolved_rung_guard.resulting_rung_status
        != resolved_witness_floor_quota.current_rung_status
    ):
        raise ValueError(
            "first-hop contract requires shared current rung status across the first-hop stack"
        )
    if (
        resolved_source_bridge.current_point_miss_vector
        != resolved_uniqueness_guard.current_point_miss_vector
    ):
        raise ValueError(
            "first-hop contract requires shared point-miss vector across source bridge and uniqueness guard"
        )
    if (
        resolved_source_bridge.current_band_miss_vector
        != resolved_uniqueness_guard.current_band_miss_vector
    ):
        raise ValueError(
            "first-hop contract requires shared band-miss vector across source bridge and uniqueness guard"
        )
    if (
        abs(
            float(resolved_source_bridge.current_witness_floor)
            - float(resolved_uniqueness_guard.current_witness_floor)
        )
        > 1e-12
        or abs(
            float(resolved_source_bridge.current_witness_floor)
            - float(resolved_witness_floor_quota.current_witness_floor)
        )
        > 1e-12
    ):
        raise ValueError(
            "first-hop contract requires shared witness-floor readout across the first-hop stack"
        )
    if (
        resolved_source_bridge.residual_slot_random_state
        != resolved_uniqueness_guard.residual_slot_random_state
        or resolved_source_bridge.residual_slot_random_state
        != resolved_witness_floor_quota.residual_slot_random_state
        or resolved_source_bridge.residual_slot_z_index
        != resolved_uniqueness_guard.residual_slot_z_index
        or resolved_source_bridge.residual_slot_z_index
        != resolved_witness_floor_quota.residual_slot_z_index
    ):
        raise ValueError(
            "first-hop contract requires shared residual-slot routing across the first-hop stack"
        )
    if (
        resolved_uniqueness_guard.binding_slot_random_state
        != resolved_witness_floor_quota.binding_slot_random_state
        or resolved_uniqueness_guard.binding_slot_z_index
        != resolved_witness_floor_quota.binding_slot_z_index
    ):
        raise ValueError(
            "first-hop contract requires shared binding-slot routing across uniqueness guard and quota"
        )

    rung_next_random_state = (
        None
        if resolved_rung_guard.next_required_slot is None
        else resolved_rung_guard.next_required_slot.random_state
    )
    rung_next_seed_group = (
        None
        if resolved_rung_guard.next_required_slot is None
        else resolved_rung_guard.next_required_slot.seed_group
    )
    rung_next_z_index = (
        None
        if resolved_rung_guard.next_required_slot is None
        else resolved_rung_guard.next_required_slot.z_index
    )
    rung_next_z_value = (
        None
        if resolved_rung_guard.next_required_slot is None
        else resolved_rung_guard.next_required_slot.z_value
    )
    if not _slot_matches(
        random_state=rung_next_random_state,
        z_index=rung_next_z_index,
        expected_random_state=resolved_source_bridge.next_required_slot_random_state,
        expected_z_index=resolved_source_bridge.next_required_slot_z_index,
    ):
        raise ValueError(
            "first-hop contract requires rung guard and source bridge next-slot sync"
        )
    if not _slot_matches(
        random_state=rung_next_random_state,
        z_index=rung_next_z_index,
        expected_random_state=resolved_uniqueness_guard.next_required_slot_random_state,
        expected_z_index=resolved_uniqueness_guard.next_required_slot_z_index,
    ):
        raise ValueError(
            "first-hop contract requires rung guard and uniqueness guard next-slot sync"
        )

    driver_signature = _driver_signature(
        rung_guard_report=resolved_rung_guard,
        source_bridge_report=resolved_source_bridge,
        uniqueness_guard_report=resolved_uniqueness_guard,
        witness_floor_quota_report=resolved_witness_floor_quota,
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "first-hop-contract"
        ),
        policy_digest=resolved_rung_guard.policy_digest,
        binding_design=resolved_rung_guard.binding_design,
        window_label=resolved_rung_guard.window_label,
        same_seed_random_states=resolved_rung_guard.same_seed_random_states,
        current_rung_candidate_status=resolved_rung_guard.current_rung_candidate_status,
        current_rung_status=resolved_rung_guard.resulting_rung_status,
        current_point_miss_vector=resolved_source_bridge.current_point_miss_vector,
        current_band_miss_vector=resolved_source_bridge.current_band_miss_vector,
        current_witness_floor=resolved_source_bridge.current_witness_floor,
        required_min_witness_floor=resolved_source_bridge.required_min_witness_floor,
        first_hop_source_bridge_driver_signature=resolved_source_bridge.driver_signature,
        first_hop_source_uniqueness_guard_driver_signature=(
            resolved_uniqueness_guard.driver_signature
        ),
        first_hop_witness_floor_quota_driver_signature=(
            resolved_witness_floor_quota.driver_signature
        ),
        next_required_slot_random_state=rung_next_random_state,
        next_required_slot_seed_group=rung_next_seed_group,
        next_required_slot_z_index=rung_next_z_index,
        next_required_slot_z_value=rung_next_z_value,
        next_required_slot_repair_stage=(
            None
            if resolved_rung_guard.next_required_slot is None
            else "binding-slot-progress"
            if resolved_rung_guard.next_required_slot.random_state == 303
            else "residual-slot-completion"
        ),
        next_required_slot_repair_role=(
            None
            if resolved_source_bridge.next_required_slot_repair_role is None
            else resolved_source_bridge.next_required_slot_repair_role
        ),
        next_required_slot_repair_priority=(
            resolved_source_bridge.next_required_slot_repair_priority
        ),
        binding_slot_random_state=resolved_witness_floor_quota.binding_slot_random_state,
        binding_slot_seed_group=resolved_witness_floor_quota.binding_slot_seed_group,
        binding_slot_z_index=resolved_witness_floor_quota.binding_slot_z_index,
        binding_slot_z_value=resolved_witness_floor_quota.binding_slot_z_value,
        binding_slot_repair_stage=resolved_witness_floor_quota.binding_slot_repair_stage,
        binding_slot_repair_role=resolved_witness_floor_quota.binding_slot_repair_role,
        binding_slot_repair_priority=resolved_witness_floor_quota.binding_slot_repair_priority,
        residual_slot_random_state=resolved_witness_floor_quota.residual_slot_random_state,
        residual_slot_seed_group=resolved_witness_floor_quota.residual_slot_seed_group,
        residual_slot_z_index=resolved_witness_floor_quota.residual_slot_z_index,
        residual_slot_z_value=resolved_witness_floor_quota.residual_slot_z_value,
        residual_slot_repair_stage=resolved_witness_floor_quota.residual_slot_repair_stage,
        residual_slot_repair_role=resolved_witness_floor_quota.residual_slot_repair_role,
        residual_slot_repair_priority=resolved_witness_floor_quota.residual_slot_repair_priority,
        baseline_open_witness_floor_gap=resolved_witness_floor_quota.baseline_open_witness_floor_gap,
        binding_slot_witness_floor_increment=(
            resolved_witness_floor_quota.binding_slot_witness_floor_increment
        ),
        residual_slot_witness_floor_increment=(
            resolved_witness_floor_quota.residual_slot_witness_floor_increment
        ),
        binding_slot_share_of_baseline_open_witness_floor_gap=(
            resolved_witness_floor_quota.binding_slot_share_of_baseline_open_witness_floor_gap
        ),
        residual_slot_share_of_baseline_open_witness_floor_gap=(
            resolved_witness_floor_quota.residual_slot_share_of_baseline_open_witness_floor_gap
        ),
        target_fold_id=resolved_uniqueness_guard.target_fold_id,
        target_exact_trim_floor_count=resolved_uniqueness_guard.target_exact_trim_floor_count,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            resolved_uniqueness_guard.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            resolved_uniqueness_guard.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            resolved_uniqueness_guard.target_exact_trim_floor_weighted_retention
        ),
        comparator_random_states=resolved_uniqueness_guard.comparator_random_states,
        comparator_exact_trim_floor_counts=(
            resolved_uniqueness_guard.comparator_exact_trim_floor_counts
        ),
        comparator_fold3_inverse_pi_phi1_center_projections=(
            resolved_uniqueness_guard.comparator_fold3_inverse_pi_phi1_center_projections
        ),
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_contract_digest=(),
    )
    return dataclass_replace(
        report,
        canonical_observed_rerun_first_hop_contract_digest=_canonical_digest(
            report=report
        ),
    )


def dataclass_replace(
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport
    ),
    *,
    canonical_observed_rerun_first_hop_contract_digest: tuple[str, ...],
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport:
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport(
        stage_label=report.stage_label,
        policy_digest=report.policy_digest,
        binding_design=report.binding_design,
        window_label=report.window_label,
        same_seed_random_states=report.same_seed_random_states,
        current_rung_candidate_status=report.current_rung_candidate_status,
        current_rung_status=report.current_rung_status,
        current_point_miss_vector=report.current_point_miss_vector,
        current_band_miss_vector=report.current_band_miss_vector,
        current_witness_floor=report.current_witness_floor,
        required_min_witness_floor=report.required_min_witness_floor,
        first_hop_source_bridge_driver_signature=report.first_hop_source_bridge_driver_signature,
        first_hop_source_uniqueness_guard_driver_signature=report.first_hop_source_uniqueness_guard_driver_signature,
        first_hop_witness_floor_quota_driver_signature=report.first_hop_witness_floor_quota_driver_signature,
        next_required_slot_random_state=report.next_required_slot_random_state,
        next_required_slot_seed_group=report.next_required_slot_seed_group,
        next_required_slot_z_index=report.next_required_slot_z_index,
        next_required_slot_z_value=report.next_required_slot_z_value,
        next_required_slot_repair_stage=report.next_required_slot_repair_stage,
        next_required_slot_repair_role=report.next_required_slot_repair_role,
        next_required_slot_repair_priority=report.next_required_slot_repair_priority,
        binding_slot_random_state=report.binding_slot_random_state,
        binding_slot_seed_group=report.binding_slot_seed_group,
        binding_slot_z_index=report.binding_slot_z_index,
        binding_slot_z_value=report.binding_slot_z_value,
        binding_slot_repair_stage=report.binding_slot_repair_stage,
        binding_slot_repair_role=report.binding_slot_repair_role,
        binding_slot_repair_priority=report.binding_slot_repair_priority,
        residual_slot_random_state=report.residual_slot_random_state,
        residual_slot_seed_group=report.residual_slot_seed_group,
        residual_slot_z_index=report.residual_slot_z_index,
        residual_slot_z_value=report.residual_slot_z_value,
        residual_slot_repair_stage=report.residual_slot_repair_stage,
        residual_slot_repair_role=report.residual_slot_repair_role,
        residual_slot_repair_priority=report.residual_slot_repair_priority,
        baseline_open_witness_floor_gap=report.baseline_open_witness_floor_gap,
        binding_slot_witness_floor_increment=report.binding_slot_witness_floor_increment,
        residual_slot_witness_floor_increment=report.residual_slot_witness_floor_increment,
        binding_slot_share_of_baseline_open_witness_floor_gap=report.binding_slot_share_of_baseline_open_witness_floor_gap,
        residual_slot_share_of_baseline_open_witness_floor_gap=report.residual_slot_share_of_baseline_open_witness_floor_gap,
        target_fold_id=report.target_fold_id,
        target_exact_trim_floor_count=report.target_exact_trim_floor_count,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi,
        target_exact_trim_floor_weighted_share_of_fold3_weighted=report.target_exact_trim_floor_weighted_share_of_fold3_weighted,
        target_exact_trim_floor_weighted_retention=report.target_exact_trim_floor_weighted_retention,
        comparator_random_states=report.comparator_random_states,
        comparator_exact_trim_floor_counts=report.comparator_exact_trim_floor_counts,
        comparator_fold3_inverse_pi_phi1_center_projections=report.comparator_fold3_inverse_pi_phi1_center_projections,
        driver_signature=report.driver_signature,
        canonical_observed_rerun_first_hop_contract_digest=canonical_observed_rerun_first_hop_contract_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract_report()
