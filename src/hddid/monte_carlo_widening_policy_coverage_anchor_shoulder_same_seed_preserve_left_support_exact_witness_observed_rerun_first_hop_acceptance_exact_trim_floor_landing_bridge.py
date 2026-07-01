from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConcentrationContractReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract_report,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport,
    build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard_report,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {_format_float(value)}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature: str
    first_hop_landing_guard_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    acceptance_shortfall: float
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_index: int
    binding_slot_z_value: float
    binding_slot_repair_stage: str
    binding_slot_repair_role: str
    binding_slot_repair_priority: int
    binding_replication_seed: int
    residual_slot_random_state: int
    residual_slot_seed_group: str
    residual_slot_z_index: int
    residual_slot_z_value: float
    residual_slot_repair_stage: str
    residual_slot_repair_role: str
    residual_slot_repair_priority: int
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_raw_phi1_share_of_fold3_raw: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    driver_signature: str
    canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge_digest: tuple[
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
        self.first_hop_landing_guard_driver_signature = str(
            self.first_hop_landing_guard_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
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
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_seed_group = str(self.binding_slot_seed_group).strip().lower()
        self.binding_slot_z_index = int(self.binding_slot_z_index)
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.binding_slot_repair_stage = str(self.binding_slot_repair_stage).strip()
        self.binding_slot_repair_role = str(self.binding_slot_repair_role).strip()
        self.binding_slot_repair_priority = int(self.binding_slot_repair_priority)
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_seed_group = (
            str(self.residual_slot_seed_group).strip().lower()
        )
        self.residual_slot_z_index = int(self.residual_slot_z_index)
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.residual_slot_repair_stage = str(self.residual_slot_repair_stage).strip()
        self.residual_slot_repair_role = str(self.residual_slot_repair_role).strip()
        self.residual_slot_repair_priority = int(self.residual_slot_repair_priority)
        self.target_fold_id = int(self.target_fold_id)
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_raw_phi1_share_of_fold3_raw = float(
            self.target_exact_trim_floor_raw_phi1_share_of_fold3_raw
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
        self.canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge_digest
        )


def _driver_signature(
    *,
    acceptance_exact_trim_floor_inverse_pi_concentration_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConcentrationContractReport
    ),
    first_hop_landing_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport
    ),
) -> str:
    if (
        acceptance_exact_trim_floor_inverse_pi_concentration_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-open"
        and first_hop_landing_guard_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-landing-guard-open"
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-open"
    if (
        acceptance_exact_trim_floor_inverse_pi_concentration_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-residual-only"
        and first_hop_landing_guard_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-landing-guard-landed"
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-residual-only"
    if (
        acceptance_exact_trim_floor_inverse_pi_concentration_contract_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-closed"
        and first_hop_landing_guard_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-landing-guard-closed"
    ):
        return "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-closed"
    return "mixed-same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge"


def _canonical_digest(
    *,
    report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport,
) -> tuple[str, ...]:
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-closed"
    ):
        return (
            "- the acceptance exact trim-floor packet and the first-hop landing guard are both fully exhausted, so the seed `303` binding hop and the queued seed `707` residual slot now remain as audit-only provenance inside the same-seed acceptance stack",
            f"- the historical bridge still records that the binding trim-floor row carried `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the gross inverse-`pi_hat` conduit, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, and `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` retention after propensity-floor weighting while the runtime path and acceptance readout stayed machine-readable on replication seed `{report.binding_replication_seed}`",
            "- current Trigger 2 implication: keep this acceptance exact trim-floor landing bridge as validation-only historical provenance once both seed `303` and seed `707` have landed",
        )
    if (
        report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-residual-only"
    ):
        return (
            "- the seed `303` binding landing hop is already absorbed, so the acceptance shortfall is no longer owned by the first hop and only the queued seed `707` / `z = 0.25` residual slot remains actionable",
            f"- even after that landing, the bridge keeps the same denominator-led audit packet machine-readable: the exact trim-floor row still anchors `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the gross inverse-`pi_hat` conduit, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, and `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` retention on replication seed `{report.binding_replication_seed}`",
            "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-residual-only`; spend seed `707` only as queued residual cleanup while keeping the bridge validation-only",
        )
    return (
        f"- the open acceptance shortfall `{_format_ninths(report.acceptance_shortfall)}` and the first-hop landing guard still point at the same actionable repair: seed `303` / witness / `z = 0.15` remains the binding landing hop, while seed `707` / fresh / `z = 0.25` stays queued as residual-only cleanup",
        f"- that shared landing hop is denominator-led rather than mass-led: the single exact trim-floor row is only `1/13 = 7.7%` of fold-`3` low-`pi_hat` treated rows and only `{_format_percent(report.target_exact_trim_floor_raw_phi1_share_of_fold3_raw)}` of the raw treated `phi1_hat` center, yet it already carries `{_format_percent(report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi)}` of the gross inverse-`pi_hat` conduit, `{_format_percent(report.target_exact_trim_floor_weighted_share_of_fold3_weighted)}` of the weighted burden, and `{_format_percent(report.target_exact_trim_floor_weighted_retention)}` weighted retention after propensity-floor weighting",
        f"- the bridge therefore keeps the runtime path `{' -> '.join(report.runtime_witness_path)}` and the acceptance-facing readout `{' -> '.join(report.acceptance_readout_path)}` machine-readable on the same replication seed `{report.binding_replication_seed}` without reopening the left guard `z = 0.05`",
        "- current Trigger 2 implication: `same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-open`; feed real same-seed reruns through this validation-only bridge before claiming seed `303` is landed or spending the queued residual slot on seed `707` / `z = 0.25`",
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge_report(
    *,
    acceptance_exact_trim_floor_inverse_pi_concentration_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorInversePiConcentrationContractReport
        | None
    ) = None,
    first_hop_landing_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopLandingGuardReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport:
    resolved_acceptance_exact_trim_floor = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract()
        if acceptance_exact_trim_floor_inverse_pi_concentration_contract_report is None
        else acceptance_exact_trim_floor_inverse_pi_concentration_contract_report
    )
    resolved_first_hop_landing_guard = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard()
        if first_hop_landing_guard_report is None
        else first_hop_landing_guard_report
    )

    if (
        resolved_acceptance_exact_trim_floor.policy_digest
        != resolved_first_hop_landing_guard.policy_digest
    ):
        raise ValueError(
            "acceptance exact trim-floor landing bridge requires shared policy digest"
        )
    if (
        resolved_acceptance_exact_trim_floor.binding_design
        != resolved_first_hop_landing_guard.binding_design
    ):
        raise ValueError(
            "acceptance exact trim-floor landing bridge requires shared binding design"
        )
    if (
        resolved_acceptance_exact_trim_floor.window_label
        != resolved_first_hop_landing_guard.window_label
    ):
        raise ValueError(
            "acceptance exact trim-floor landing bridge requires shared window label"
        )
    if (
        resolved_acceptance_exact_trim_floor.binding_slot_random_state
        != resolved_first_hop_landing_guard.binding_slot_random_state
        or resolved_acceptance_exact_trim_floor.binding_slot_z_index
        != resolved_first_hop_landing_guard.binding_slot_z_index
    ):
        raise ValueError(
            "acceptance exact trim-floor landing bridge requires the same binding slot"
        )
    if (
        resolved_acceptance_exact_trim_floor.residual_slot_random_state
        != resolved_first_hop_landing_guard.queued_residual_slot_random_state
        or resolved_acceptance_exact_trim_floor.residual_slot_z_index
        != resolved_first_hop_landing_guard.queued_residual_slot_z_index
    ):
        raise ValueError(
            "acceptance exact trim-floor landing bridge requires the same residual slot"
        )
    if (
        abs(
            resolved_acceptance_exact_trim_floor.current_witness_floor
            - resolved_first_hop_landing_guard.current_witness_floor
        )
        > 1e-12
    ):
        raise ValueError(
            "acceptance exact trim-floor landing bridge requires aligned witness floors"
        )
    if (
        abs(
            resolved_acceptance_exact_trim_floor.required_min_witness_floor
            - resolved_first_hop_landing_guard.required_min_witness_floor
        )
        > 1e-12
    ):
        raise ValueError(
            "acceptance exact trim-floor landing bridge requires aligned witness floor requirements"
        )
    if (
        resolved_acceptance_exact_trim_floor.binding_replication_seed
        != resolved_first_hop_landing_guard.binding_slot_replication_seed
    ):
        raise ValueError(
            "acceptance exact trim-floor landing bridge requires aligned replication seeds"
        )
    if (
        resolved_acceptance_exact_trim_floor.runtime_witness_path
        != resolved_first_hop_landing_guard.runtime_witness_path
    ):
        raise ValueError(
            "acceptance exact trim-floor landing bridge requires aligned runtime witness paths"
        )

    driver_signature = _driver_signature(
        acceptance_exact_trim_floor_inverse_pi_concentration_contract_report=resolved_acceptance_exact_trim_floor,
        first_hop_landing_guard_report=resolved_first_hop_landing_guard,
    )
    report = Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-exact-witness-observed-rerun-"
            "first-hop-acceptance-exact-trim-floor-landing-bridge"
        ),
        policy_digest=resolved_acceptance_exact_trim_floor.policy_digest,
        binding_design=resolved_acceptance_exact_trim_floor.binding_design,
        window_label=resolved_acceptance_exact_trim_floor.window_label,
        same_seed_random_states=resolved_acceptance_exact_trim_floor.same_seed_random_states,
        acceptance_exact_trim_floor_inverse_pi_concentration_driver_signature=resolved_acceptance_exact_trim_floor.driver_signature,
        first_hop_landing_guard_driver_signature=resolved_first_hop_landing_guard.driver_signature,
        runtime_witness_path=resolved_acceptance_exact_trim_floor.runtime_witness_path,
        acceptance_readout_path=resolved_acceptance_exact_trim_floor.acceptance_readout_path,
        current_point_miss_vector=resolved_first_hop_landing_guard.current_point_miss_vector,
        current_band_miss_vector=resolved_first_hop_landing_guard.current_band_miss_vector,
        current_witness_floor=resolved_first_hop_landing_guard.current_witness_floor,
        required_min_witness_floor=resolved_first_hop_landing_guard.required_min_witness_floor,
        acceptance_shortfall=resolved_acceptance_exact_trim_floor.acceptance_shortfall,
        binding_slot_random_state=resolved_first_hop_landing_guard.binding_slot_random_state,
        binding_slot_seed_group=resolved_first_hop_landing_guard.binding_slot_seed_group,
        binding_slot_z_index=resolved_first_hop_landing_guard.binding_slot_z_index,
        binding_slot_z_value=resolved_first_hop_landing_guard.binding_slot_z_value,
        binding_slot_repair_stage=resolved_first_hop_landing_guard.binding_slot_repair_stage,
        binding_slot_repair_role=resolved_first_hop_landing_guard.binding_slot_repair_role,
        binding_slot_repair_priority=resolved_first_hop_landing_guard.binding_slot_repair_priority,
        binding_replication_seed=resolved_first_hop_landing_guard.binding_slot_replication_seed,
        residual_slot_random_state=resolved_first_hop_landing_guard.queued_residual_slot_random_state,
        residual_slot_seed_group=resolved_first_hop_landing_guard.queued_residual_slot_seed_group,
        residual_slot_z_index=resolved_first_hop_landing_guard.queued_residual_slot_z_index,
        residual_slot_z_value=resolved_first_hop_landing_guard.queued_residual_slot_z_value,
        residual_slot_repair_stage="residual-slot-completion",
        residual_slot_repair_role="residual-total-miss-closure",
        residual_slot_repair_priority=2,
        target_fold_id=resolved_acceptance_exact_trim_floor.target_fold_id,
        target_exact_trim_floor_count=resolved_acceptance_exact_trim_floor.target_exact_trim_floor_count,
        target_exact_trim_floor_raw_phi1_share_of_fold3_raw=resolved_acceptance_exact_trim_floor.target_exact_trim_floor_raw_phi1_share_of_fold3_raw,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=resolved_acceptance_exact_trim_floor.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi,
        target_exact_trim_floor_weighted_share_of_fold3_weighted=resolved_acceptance_exact_trim_floor.target_exact_trim_floor_weighted_share_of_fold3_weighted,
        target_exact_trim_floor_weighted_retention=resolved_acceptance_exact_trim_floor.target_exact_trim_floor_weighted_retention,
        driver_signature=driver_signature,
        canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge_digest=(),
    )
    report = replace(
        report,
        canonical_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge_digest=_canonical_digest(
            report=report
        ),
    )
    return report


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunFirstHopAcceptanceExactTrimFloorLandingBridgeReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge",
]
