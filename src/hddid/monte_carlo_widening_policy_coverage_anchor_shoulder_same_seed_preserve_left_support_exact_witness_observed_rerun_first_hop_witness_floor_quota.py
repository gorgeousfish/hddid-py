from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard_report,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCompletionProfileReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _remaining_gap(
    *, current_witness_floor: float, required_min_witness_floor: float
) -> float:
    return max(float(required_min_witness_floor) - float(current_witness_floor), 0.0)


def _share_of_gap(*, increment: float, baseline_gap: float) -> float:
    if baseline_gap <= 0.0:
        return 0.0
    return float(increment) / float(baseline_gap)


def _driver_signature(
    *,
    uniqueness_guard_driver_signature: str,
    current_remaining_witness_floor_gap: float,
    baseline_open_witness_floor_gap: float,
    binding_slot_witness_floor_increment: float,
    residual_slot_witness_floor_increment: float,
    projected_binding_witness_floor: float,
    completion_witness_floor: float,
    required_min_witness_floor: float,
) -> str:
    quota_is_one_sided = (
        abs(
            float(binding_slot_witness_floor_increment)
            - float(baseline_open_witness_floor_gap)
        )
        <= 1e-12
        and abs(float(residual_slot_witness_floor_increment)) <= 1e-12
        and abs(
            float(projected_binding_witness_floor) - float(required_min_witness_floor)
        )
        <= 1e-12
        and abs(float(completion_witness_floor) - float(required_min_witness_floor))
        <= 1e-12
    )
    if not quota_is_one_sided:
        return (
            "mixed-same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota"
        )
    if (
        uniqueness_guard_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-closed"
        and abs(float(current_remaining_witness_floor_gap)) <= 1e-12
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-closed"
    if (
        uniqueness_guard_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-residual-only"
        and abs(float(current_remaining_witness_floor_gap)) <= 1e-12
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-residual-only"
    if (
        uniqueness_guard_driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-source-uniqueness-guard-open"
        and abs(
            float(current_remaining_witness_floor_gap)
            - float(baseline_open_witness_floor_gap)
        )
        <= 1e-12
    ):
        return (
            "same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-open"
        )
    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    binding_slot_landed: bool
    residual_slot_landed: bool
    current_rung_status: str
    first_hop_source_uniqueness_guard_driver_signature: str
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
    baseline_point_miss_vector: tuple[int, int, int]
    binding_only_point_miss_vector: tuple[int, int, int]
    completion_point_miss_vector: tuple[int, int, int]
    baseline_band_miss_vector: tuple[int, int, int]
    binding_only_band_miss_vector: tuple[int, int, int]
    completion_band_miss_vector: tuple[int, int, int]
    baseline_witness_floor: float
    current_witness_floor: float
    projected_binding_witness_floor: float
    completion_witness_floor: float
    required_min_witness_floor: float
    baseline_open_witness_floor_gap: float
    current_remaining_witness_floor_gap: float
    binding_slot_witness_floor_increment: float
    residual_slot_witness_floor_increment: float
    binding_slot_share_of_baseline_open_witness_floor_gap: float
    residual_slot_share_of_baseline_open_witness_floor_gap: float
    driver_signature: str
    canonical_observed_rerun_first_hop_witness_floor_quota_digest: tuple[str, ...]

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
        self.binding_slot_landed = bool(self.binding_slot_landed)
        self.residual_slot_landed = bool(self.residual_slot_landed)
        self.current_rung_status = str(self.current_rung_status).strip()
        self.first_hop_source_uniqueness_guard_driver_signature = str(
            self.first_hop_source_uniqueness_guard_driver_signature
        ).strip()
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
        self.baseline_point_miss_vector = tuple(
            int(value) for value in self.baseline_point_miss_vector
        )
        self.binding_only_point_miss_vector = tuple(
            int(value) for value in self.binding_only_point_miss_vector
        )
        self.completion_point_miss_vector = tuple(
            int(value) for value in self.completion_point_miss_vector
        )
        self.baseline_band_miss_vector = tuple(
            int(value) for value in self.baseline_band_miss_vector
        )
        self.binding_only_band_miss_vector = tuple(
            int(value) for value in self.binding_only_band_miss_vector
        )
        self.completion_band_miss_vector = tuple(
            int(value) for value in self.completion_band_miss_vector
        )
        self.baseline_witness_floor = float(self.baseline_witness_floor)
        self.current_witness_floor = float(self.current_witness_floor)
        self.projected_binding_witness_floor = float(
            self.projected_binding_witness_floor
        )
        self.completion_witness_floor = float(self.completion_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.baseline_open_witness_floor_gap = float(
            self.baseline_open_witness_floor_gap
        )
        self.current_remaining_witness_floor_gap = float(
            self.current_remaining_witness_floor_gap
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
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_observed_rerun_first_hop_witness_floor_quota_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_first_hop_witness_floor_quota_digest
        )


def _canonical_digest(
    *,
    report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport
    ),
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-closed"
    ):
        return (
            f"- observed same-seed rerun has already consumed the full first-hop witness-floor quota: witness floor holds at `{_format_ninths(report.current_witness_floor)}`, so the remaining open gap is `0/9 = {_format_float(report.current_remaining_witness_floor_gap)}`",
            "- the quota split is therefore historical but exact: seed `303` already supplied the full `1/9 = 0.111` / `100.0%` admissibility-facing floor lift, while seed `707` remained `0/9 = 0.000` witness-floor-neutral cleanup",
            "- pointwise completion is already fully consumed as well, so no same-line rerun budget remains on this object",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-closed`; keep this helper validation-only and out of live routing surfaces",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-residual-only"
    ):
        return (
            f"- observed same-seed rerun has already consumed the entire admissibility-facing floor quota: current witness floor sits at `{_format_ninths(report.current_witness_floor)}`, so the remaining open gap is `0/9 = {_format_float(report.current_remaining_witness_floor_gap)}`",
            "- that confirms the quota split was one-sided: seed `303` supplied `1/9 = 0.111` / `100.0%` of the baseline open witness-floor gap, while seed `707` still carries `0/9 = 0.000` witness-floor increment",
            f"- residual work is now purely total-miss cleanup: point miss can still fall from `{list(report.binding_only_point_miss_vector)}` to `{list(report.completion_point_miss_vector)}`, but witness-floor arithmetic is already closed",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-residual-only`; keep the helper validation-only and route the next rerun only to the residual slot",
        )
    return (
        f"- observed same-seed rerun still keeps an open admissibility floor gap: witness floor stays `{_format_ninths(report.current_witness_floor)}`, required minimum stays `{_format_ninths(report.required_min_witness_floor)}`, so the remaining open gap is exactly `{_format_ninths(report.current_remaining_witness_floor_gap)}`",
        "- the unique seed `303` first hop already owns that entire quota: seed `303` / witness / `z = 0.15` supplies "
        f"`{_format_ninths(report.binding_slot_witness_floor_increment)}`, i.e. "
        f"`{_format_percent(report.binding_slot_share_of_baseline_open_witness_floor_gap)}` of the baseline open witness-floor gap, while residual seed `707` contributes "
        f"`{_format_ninths(report.residual_slot_witness_floor_increment)}` to witness-floor arithmetic",
        f"- queue semantics therefore stay asymmetric even after the source-backed uniqueness guard: seed `303` is the only admissibility-facing first hop, while seed `707` / fresh / `z = 0.25` remains witness-floor-neutral residual-total-miss-closure from `{list(report.binding_only_point_miss_vector)}` to `{list(report.completion_point_miss_vector)}`",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-witness-floor-quota-open`; land the seed `303` first hop before treating any seed `707` spend as acceptance-relevant",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota_report(
    *,
    after_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    before_report: (
        Phase7MonteCarloWideningPolicyNearZeroGridSeedwiseCoverageDecompositionReport
        | None
    ) = None,
    uniqueness_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopSourceUniquenessGuardReport
        | None
    ) = None,
    binding_slot_progress_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessBindingSlotProgressProfileReport
        | None
    ) = None,
    residual_slot_completion_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessResidualSlotCompletionProfileReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport:
    resolved_uniqueness_guard = (
        build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard_report(
            after_report=after_report,
            before_report=before_report,
        )
        if uniqueness_guard_report is None
        else uniqueness_guard_report
    )
    resolved_binding_progress = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile()
        if binding_slot_progress_report is None
        else binding_slot_progress_report
    )
    resolved_residual_completion = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile()
        if residual_slot_completion_report is None
        else residual_slot_completion_report
    )

    if (
        resolved_uniqueness_guard.policy_digest
        != resolved_binding_progress.policy_digest
    ):
        raise ValueError("first-hop witness-floor quota requires shared policy digest")
    if (
        resolved_uniqueness_guard.binding_design
        != resolved_binding_progress.binding_design
    ):
        raise ValueError("first-hop witness-floor quota requires shared binding design")
    if resolved_uniqueness_guard.window_label != resolved_binding_progress.window_label:
        raise ValueError("first-hop witness-floor quota requires shared window label")
    if (
        resolved_uniqueness_guard.same_seed_random_states
        != resolved_binding_progress.same_seed_random_states
    ):
        raise ValueError(
            "first-hop witness-floor quota requires shared exact same-seed ordering"
        )
    if (
        resolved_binding_progress.policy_digest
        != resolved_residual_completion.policy_digest
    ):
        raise ValueError(
            "first-hop witness-floor quota requires completion policy sync"
        )
    if (
        resolved_binding_progress.binding_design
        != resolved_residual_completion.binding_design
    ):
        raise ValueError(
            "first-hop witness-floor quota requires completion design sync"
        )
    if (
        resolved_binding_progress.window_label
        != resolved_residual_completion.window_label
    ):
        raise ValueError(
            "first-hop witness-floor quota requires completion window sync"
        )

    baseline_open_gap = _remaining_gap(
        current_witness_floor=resolved_binding_progress.baseline_witness_floor,
        required_min_witness_floor=resolved_binding_progress.required_min_witness_floor,
    )
    current_remaining_gap = _remaining_gap(
        current_witness_floor=resolved_uniqueness_guard.current_witness_floor,
        required_min_witness_floor=resolved_uniqueness_guard.required_min_witness_floor,
    )
    binding_share = _share_of_gap(
        increment=resolved_uniqueness_guard.binding_witness_floor_increment,
        baseline_gap=baseline_open_gap,
    )
    residual_share = _share_of_gap(
        increment=resolved_uniqueness_guard.residual_witness_floor_increment,
        baseline_gap=baseline_open_gap,
    )

    driver_signature = _driver_signature(
        uniqueness_guard_driver_signature=resolved_uniqueness_guard.driver_signature,
        current_remaining_witness_floor_gap=current_remaining_gap,
        baseline_open_witness_floor_gap=baseline_open_gap,
        binding_slot_witness_floor_increment=(
            resolved_uniqueness_guard.binding_witness_floor_increment
        ),
        residual_slot_witness_floor_increment=(
            resolved_uniqueness_guard.residual_witness_floor_increment
        ),
        projected_binding_witness_floor=(
            resolved_binding_progress.projected_binding_witness_floor
        ),
        completion_witness_floor=resolved_residual_completion.completion_witness_floor,
        required_min_witness_floor=resolved_uniqueness_guard.required_min_witness_floor,
    )

    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "first-hop-witness-floor-quota"
        ),
        policy_digest=resolved_uniqueness_guard.policy_digest,
        binding_design=resolved_uniqueness_guard.binding_design,
        window_label=resolved_uniqueness_guard.window_label,
        same_seed_random_states=resolved_uniqueness_guard.same_seed_random_states,
        binding_slot_landed=resolved_uniqueness_guard.binding_slot_landed,
        residual_slot_landed=resolved_uniqueness_guard.residual_slot_landed,
        current_rung_status=resolved_uniqueness_guard.current_rung_status,
        first_hop_source_uniqueness_guard_driver_signature=(
            resolved_uniqueness_guard.driver_signature
        ),
        binding_slot_random_state=resolved_uniqueness_guard.binding_slot_random_state,
        binding_slot_seed_group=resolved_uniqueness_guard.binding_slot_seed_group,
        binding_slot_z_index=resolved_uniqueness_guard.binding_slot_z_index,
        binding_slot_z_value=resolved_uniqueness_guard.binding_slot_z_value,
        binding_slot_repair_stage=resolved_uniqueness_guard.binding_slot_repair_stage,
        binding_slot_repair_role=resolved_uniqueness_guard.binding_slot_repair_role,
        binding_slot_repair_priority=resolved_uniqueness_guard.binding_slot_repair_priority,
        residual_slot_random_state=resolved_uniqueness_guard.residual_slot_random_state,
        residual_slot_seed_group=resolved_uniqueness_guard.residual_slot_seed_group,
        residual_slot_z_index=resolved_uniqueness_guard.residual_slot_z_index,
        residual_slot_z_value=resolved_uniqueness_guard.residual_slot_z_value,
        residual_slot_repair_stage=resolved_uniqueness_guard.residual_slot_repair_stage,
        residual_slot_repair_role=resolved_uniqueness_guard.residual_slot_repair_role,
        residual_slot_repair_priority=resolved_uniqueness_guard.residual_slot_repair_priority,
        baseline_point_miss_vector=resolved_binding_progress.baseline_point_miss_vector,
        binding_only_point_miss_vector=(
            resolved_binding_progress.binding_only_point_miss_vector
        ),
        completion_point_miss_vector=(
            resolved_residual_completion.completion_point_miss_vector
        ),
        baseline_band_miss_vector=resolved_binding_progress.baseline_band_miss_vector,
        binding_only_band_miss_vector=(
            resolved_binding_progress.binding_only_band_miss_vector
        ),
        completion_band_miss_vector=(
            resolved_residual_completion.completion_band_miss_vector
        ),
        baseline_witness_floor=resolved_binding_progress.baseline_witness_floor,
        current_witness_floor=resolved_uniqueness_guard.current_witness_floor,
        projected_binding_witness_floor=(
            resolved_binding_progress.projected_binding_witness_floor
        ),
        completion_witness_floor=resolved_residual_completion.completion_witness_floor,
        required_min_witness_floor=resolved_uniqueness_guard.required_min_witness_floor,
        baseline_open_witness_floor_gap=baseline_open_gap,
        current_remaining_witness_floor_gap=current_remaining_gap,
        binding_slot_witness_floor_increment=(
            resolved_uniqueness_guard.binding_witness_floor_increment
        ),
        residual_slot_witness_floor_increment=(
            resolved_uniqueness_guard.residual_witness_floor_increment
        ),
        binding_slot_share_of_baseline_open_witness_floor_gap=binding_share,
        residual_slot_share_of_baseline_open_witness_floor_gap=residual_share,
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_witness_floor_quota_digest=(),
    )
    return replace(
        report,
        canonical_observed_rerun_first_hop_witness_floor_quota_digest=_canonical_digest(
            report=report
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota_report()


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_witness_floor_quota_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopWitnessFloorQuotaReport
):
    return (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota()
    )
