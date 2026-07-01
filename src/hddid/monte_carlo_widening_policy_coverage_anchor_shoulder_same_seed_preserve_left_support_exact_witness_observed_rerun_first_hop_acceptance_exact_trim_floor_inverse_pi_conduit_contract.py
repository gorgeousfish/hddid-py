from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConcentrationContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


def _format_exact_trim_floor(value: float) -> str:
    return f"{float(value):.6f}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConduitContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature: str
    low_pi_fold3_trim_floor_propensity_floor_driver_signature: str
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
    low_pi_threshold: float
    trim_lower: float
    exact_trim_floor_value: float
    target_fold_id: int
    target_fold3_low_pi_treated_count: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_share_of_fold3_low_pi_count: float
    target_exact_trim_floor_raw_phi1_center_projection: float
    target_exact_trim_floor_raw_phi1_share_of_fold3_raw: float
    target_exact_trim_floor_inverse_pi_phi1_center_projection: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_phi1_center_projection: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    target_remaining_fold3_low_pi_weighted_retention: float
    target_exact_trim_floor_weighted_retention_multiple_vs_remaining: float
    target_exact_trim_floor_row_gross_inverse_pi_amplification: float
    target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification: float
    target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining: float
    target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining: (
        float
    )
    comparator_random_states: tuple[int, ...]
    comparator_exact_trim_floor_counts: tuple[int, ...]
    comparator_fold3_inverse_pi_phi1_center_projections: tuple[float, ...]
    driver_signature: str
    canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract_digest: tuple[
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
        self.acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature = str(
            self.acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature
        ).strip()
        self.low_pi_fold3_trim_floor_propensity_floor_driver_signature = str(
            self.low_pi_fold3_trim_floor_propensity_floor_driver_signature
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
        self.low_pi_threshold = float(self.low_pi_threshold)
        self.trim_lower = float(self.trim_lower)
        self.exact_trim_floor_value = float(self.exact_trim_floor_value)
        self.target_fold_id = int(self.target_fold_id)
        self.target_fold3_low_pi_treated_count = int(
            self.target_fold3_low_pi_treated_count
        )
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_share_of_fold3_low_pi_count = float(
            self.target_exact_trim_floor_share_of_fold3_low_pi_count
        )
        self.target_exact_trim_floor_raw_phi1_center_projection = float(
            self.target_exact_trim_floor_raw_phi1_center_projection
        )
        self.target_exact_trim_floor_raw_phi1_share_of_fold3_raw = float(
            self.target_exact_trim_floor_raw_phi1_share_of_fold3_raw
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
        self.target_exact_trim_floor_weighted_retention_multiple_vs_remaining = float(
            self.target_exact_trim_floor_weighted_retention_multiple_vs_remaining
        )
        self.target_exact_trim_floor_row_gross_inverse_pi_amplification = float(
            self.target_exact_trim_floor_row_gross_inverse_pi_amplification
        )
        self.target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification = float(
            self.target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification
        )
        self.target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining = float(
            self.target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining
        )
        self.target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining = float(
            self.target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining
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
        self.canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract_digest
        )


def _driver_signature(
    *,
    acceptance_exact_trim_floor_inverse_pi_concentration_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConcentrationContractReport
    ),
    low_pi_fold3_trim_floor_propensity_floor_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport
    ),
) -> str:
    propensity_floor_confirmed = (
        low_pi_fold3_trim_floor_propensity_floor_trace_report.driver_signature
        == "same-seed-seed303-fold3-trim-floor-propensity-floor-driver-confirmed"
    )
    if (
        acceptance_exact_trim_floor_inverse_pi_concentration_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-open"
        and propensity_floor_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-open"
    if (
        acceptance_exact_trim_floor_inverse_pi_concentration_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-residual-only"
        and propensity_floor_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-residual-only"
    if (
        acceptance_exact_trim_floor_inverse_pi_concentration_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-closed"
        and propensity_floor_confirmed
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-closed"
    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-contract"


def _canonical_digest(
    *,
    report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConduitContractReport,
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-closed"
    ):
        return (
            "- both the acceptance bridge and the queued residual slot are already closed, so the retained inverse-`pi_hat` conduit now stays as audit-only provenance inside the first-hop acceptance stack",
            f"- that historical conduit still records that the exact trim-floor row `pi_hat = {_format_exact_trim_floor(report.exact_trim_floor_value)}` kept `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the fold-`3` gross inverse-`pi_hat` conduit, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, and `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` of its own gross lift after propensity-floor weighting",
            "- current Trigger 2 implication: keep this acceptance exact trim-floor inverse-`pi_hat` conduit contract validation-only and out of live routing surfaces once the same-seed observed-rerun lane is fully closed",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-residual-only"
    ):
        return (
            "- the open acceptance shortfall has already closed to `0/9 = 0.000`, so the retained inverse-`pi_hat` conduit now stays as residual-only provenance while seed `707` / fresh / `z = 0.25` remains the only queued residual spend",
            f"- that historical conduit still records that the exact trim-floor row `pi_hat = {_format_exact_trim_floor(report.exact_trim_floor_value)}` kept `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the fold-`3` gross inverse-`pi_hat` conduit, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, and `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` of its own gross lift after propensity-floor weighting even though it was only `{report.target_exact_trim_floor_count}/{report.target_fold3_low_pi_treated_count} = {_format_percent(report.target_exact_trim_floor_share_of_fold3_low_pi_count)}` of the low-`pi_hat` treated rows",
            "- current Trigger 2 implication: keep this acceptance exact trim-floor inverse-`pi_hat` conduit contract validation-only while the live spend remains the queued residual seed `707` / `z = 0.25`",
        )
    return (
        f"- the open acceptance shortfall still stays `{_format_ninths(report.acceptance_shortfall)}`, so the validation-only first-hop acceptance exact trim-floor inverse-`pi_hat` concentration contract remains live while seed `303` / witness / fold `3` / `z = 0.15` keeps the acceptance-facing readout `omega_f_hat[2,2] -> v_f_hat[2,1] -> bar_f_at_z0[1]` actionable before any residual seed `707` spend",
        f"- within that same packet, the single exact trim-floor row still carries `{_format_float(report.target_exact_trim_floor_inverse_pi_phi1_center_projection)}` / `5.010` (`{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}`) of the fold-`3` gross inverse-`pi_hat` conduit and `{_format_float(report.target_exact_trim_floor_weighted_phi1_center_projection)}` / `4.778` (`{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}`) of the weighted burden even though it is only `{report.target_exact_trim_floor_count}/{report.target_fold3_low_pi_treated_count} = {_format_percent(report.target_exact_trim_floor_share_of_fold3_low_pi_count)}` of the low-`pi_hat` treated rows and `{_format_float(report.target_exact_trim_floor_raw_phi1_center_projection)}` / `0.232` (`{_format_percent(report.target_exact_trim_floor_raw_phi1_share_of_fold3_raw)}`) of the fold-`3` raw treated `phi1_hat` center",
        f"- the conduit therefore stays retained rather than merely concentrated: the trim-floor row keeps `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` of its own gross lift after propensity-floor weighting versus `{_format_percent(report.target_remaining_fold3_low_pi_weighted_retention)}` for the remaining rows, while its gross amplification still sits at `{_format_float(report.target_exact_trim_floor_row_gross_inverse_pi_amplification)}x` versus `{_format_float(report.target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification)}x` for the remaining tail, i.e. `{_format_float(report.target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining)}x` the rest-of-tail gross multiple and `{_format_float(report.target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining)}x` per observation",
        f"- comparator seeds `202` and `707` still expose zero exact trim-floor rows inside the same fold-`3` low-`pi_hat` slice and keep full fold-`3` gross inverse-`pi_hat` conduits negative at `{_format_float(report.comparator_fold3_inverse_pi_phi1_center_projections[0])}` and `{_format_float(report.comparator_fold3_inverse_pi_phi1_center_projections[1])}`; current Trigger 2 implication: `{report.driver_signature}`, so keep live entry at `trigger2-policy-spec`, preserve the acceptance bridge / object-flow / Eq. (3.1) / raw-score / delta-y / low-`pi_hat` support-allocation / exact-trim-floor inverse-`pi_hat` concentration stack, and feed the real same-seed observed rerun through the landing bridge using this retained conduit before spending seed `707` / `z = 0.25`",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract_report(
    *,
    acceptance_exact_trim_floor_inverse_pi_concentration_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConcentrationContractReport
        | None
    ) = None,
    low_pi_fold3_trim_floor_propensity_floor_trace_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportSeed303LowPiFold3TrimFloorPropensityFloorTraceReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConduitContractReport:
    resolved_concentration_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract()
        if acceptance_exact_trim_floor_inverse_pi_concentration_contract_report is None
        else acceptance_exact_trim_floor_inverse_pi_concentration_contract_report
    )
    resolved_propensity_floor_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace()
        if low_pi_fold3_trim_floor_propensity_floor_trace_report is None
        else low_pi_fold3_trim_floor_propensity_floor_trace_report
    )
    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConduitContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-"
            "preserve-left-support-exact-witness-observed-rerun-first-hop-"
            "acceptance-exact-trim-floor-inverse-pi-conduit-contract"
        ),
        policy_digest=resolved_concentration_report.policy_digest,
        binding_design=resolved_concentration_report.binding_design,
        window_label=resolved_concentration_report.window_label,
        same_seed_random_states=resolved_concentration_report.same_seed_random_states,
        acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature=(
            resolved_concentration_report.driver_signature
        ),
        low_pi_fold3_trim_floor_propensity_floor_driver_signature=(
            resolved_propensity_floor_report.driver_signature
        ),
        runtime_witness_path=resolved_concentration_report.runtime_witness_path,
        acceptance_readout_path=resolved_concentration_report.acceptance_readout_path,
        acceptance_shortfall=resolved_concentration_report.acceptance_shortfall,
        current_witness_floor=resolved_concentration_report.current_witness_floor,
        required_min_witness_floor=resolved_concentration_report.required_min_witness_floor,
        binding_slot_random_state=resolved_concentration_report.binding_slot_random_state,
        binding_slot_seed_group=resolved_concentration_report.binding_slot_seed_group,
        binding_slot_z_index=resolved_concentration_report.binding_slot_z_index,
        binding_slot_z_value=resolved_concentration_report.binding_slot_z_value,
        residual_slot_random_state=resolved_concentration_report.residual_slot_random_state,
        residual_slot_seed_group=resolved_concentration_report.residual_slot_seed_group,
        residual_slot_z_index=resolved_concentration_report.residual_slot_z_index,
        residual_slot_z_value=resolved_concentration_report.residual_slot_z_value,
        binding_replication_seed=resolved_concentration_report.binding_replication_seed,
        low_pi_threshold=resolved_concentration_report.low_pi_threshold,
        trim_lower=resolved_concentration_report.trim_lower,
        exact_trim_floor_value=resolved_concentration_report.exact_trim_floor_value,
        target_fold_id=resolved_concentration_report.target_fold_id,
        target_fold3_low_pi_treated_count=resolved_concentration_report.target_fold3_low_pi_treated_count,
        target_exact_trim_floor_count=resolved_concentration_report.target_exact_trim_floor_count,
        target_exact_trim_floor_share_of_fold3_low_pi_count=resolved_concentration_report.target_exact_trim_floor_share_of_fold3_low_pi_count,
        target_exact_trim_floor_raw_phi1_center_projection=resolved_concentration_report.target_exact_trim_floor_raw_phi1_center_projection,
        target_exact_trim_floor_raw_phi1_share_of_fold3_raw=resolved_concentration_report.target_exact_trim_floor_raw_phi1_share_of_fold3_raw,
        target_exact_trim_floor_inverse_pi_phi1_center_projection=resolved_concentration_report.target_exact_trim_floor_inverse_pi_phi1_center_projection,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=resolved_concentration_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi,
        target_exact_trim_floor_weighted_phi1_center_projection=resolved_concentration_report.target_exact_trim_floor_weighted_phi1_center_projection,
        target_exact_trim_floor_weighted_share_of_fold3_weighted=resolved_concentration_report.target_exact_trim_floor_weighted_share_of_fold3_weighted,
        target_exact_trim_floor_weighted_retention=resolved_concentration_report.target_exact_trim_floor_weighted_retention,
        target_remaining_fold3_low_pi_weighted_retention=resolved_concentration_report.target_remaining_fold3_low_pi_weighted_retention,
        target_exact_trim_floor_weighted_retention_multiple_vs_remaining=resolved_concentration_report.target_exact_trim_floor_weighted_retention_multiple_vs_remaining,
        target_exact_trim_floor_row_gross_inverse_pi_amplification=resolved_concentration_report.target_exact_trim_floor_row_gross_inverse_pi_amplification,
        target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification=resolved_concentration_report.target_remaining_fold3_low_pi_average_gross_inverse_pi_amplification,
        target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining=resolved_concentration_report.target_exact_trim_floor_gross_inverse_pi_amplification_multiple_vs_remaining,
        target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining=resolved_concentration_report.target_exact_trim_floor_gross_inverse_pi_per_observation_multiple_vs_remaining,
        comparator_random_states=resolved_concentration_report.comparator_random_states,
        comparator_exact_trim_floor_counts=resolved_concentration_report.comparator_exact_trim_floor_counts,
        comparator_fold3_inverse_pi_phi1_center_projections=resolved_concentration_report.comparator_fold3_inverse_pi_phi1_center_projections,
        driver_signature=_driver_signature(
            acceptance_exact_trim_floor_inverse_pi_concentration_contract_report=resolved_concentration_report,
            low_pi_fold3_trim_floor_propensity_floor_trace_report=resolved_propensity_floor_report,
        ),
        canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract_digest=(),
    )
    return replace(
        report,
        canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract_digest=_canonical_digest(
            report=report
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConduitContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract_report()
