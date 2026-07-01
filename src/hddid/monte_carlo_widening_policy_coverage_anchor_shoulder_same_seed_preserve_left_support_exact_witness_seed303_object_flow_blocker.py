from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerSlice:
    random_state: int
    seed_group: str
    z_index: int
    z_value: float
    truth_at_z: float
    bar_f_at_z: float
    sigma_z_hat: float
    pointwise_lower: float
    pointwise_upper: float
    pointwise_covered: bool
    uniform_band_covered: bool
    vf_cross_entry: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)
        self.truth_at_z = float(self.truth_at_z)
        self.bar_f_at_z = float(self.bar_f_at_z)
        self.sigma_z_hat = float(self.sigma_z_hat)
        self.pointwise_lower = float(self.pointwise_lower)
        self.pointwise_upper = float(self.pointwise_upper)
        self.pointwise_covered = bool(self.pointwise_covered)
        self.uniform_band_covered = bool(self.uniform_band_covered)
        self.vf_cross_entry = float(self.vf_cross_entry)


def _driver_signature(
    *,
    repair_agenda_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
    ),
    target_slice: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerSlice,
    passing_witness_slice: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerSlice,
    residual_fresh_slice: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerSlice,
    target_center_gap_to_truth: float,
    target_lower_gap_to_truth: float,
    sigma_ratio_vs_passing_witness: float,
    sigma_ratio_vs_residual_fresh: float,
) -> str:
    next_required_slot = repair_agenda_report.next_required_slot
    if (
        repair_agenda_report.current_rung_status
        == "same-seed-exact-witness-observed-rerun-open"
        and repair_agenda_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-open"
        and next_required_slot is not None
        and next_required_slot.random_state == 303
        and next_required_slot.seed_group == "witness"
        and next_required_slot.z_index == 1
        and abs(next_required_slot.z_value - 0.15) <= 1e-12
        and target_slice.random_state == 303
        and target_slice.seed_group == "witness"
        and target_slice.z_index == 1
        and abs(target_slice.z_value - 0.15) <= 1e-12
        and not target_slice.pointwise_covered
        and target_slice.uniform_band_covered
        and target_center_gap_to_truth > 0.0
        and target_lower_gap_to_truth > 0.0
        and passing_witness_slice.random_state == 202
        and passing_witness_slice.seed_group == "witness"
        and residual_fresh_slice.random_state == 707
        and residual_fresh_slice.seed_group == "fresh"
        and sigma_ratio_vs_passing_witness < 1.0
        and sigma_ratio_vs_residual_fresh < 1.0
        and target_slice.vf_cross_entry < 0.0
        and passing_witness_slice.vf_cross_entry > 0.0
        and residual_fresh_slice.vf_cross_entry > 0.0
    ):
        return "same-seed-exact-witness-seed303-pointwise-overshoot-object-flow-blocker"
    return "mixed-same-seed-exact-witness-seed303-object-flow-blocker"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    current_rung_status: str
    repair_agenda_signature: str
    target_random_state: int
    target_seed_group: str
    target_z_index: int
    target_z_value: float
    target_truth_at_z: float
    target_bar_f_at_z: float
    target_sigma_z_hat: float
    target_pointwise_lower: float
    target_pointwise_upper: float
    target_pointwise_covered: bool
    target_uniform_band_covered: bool
    target_center_gap_to_truth: float
    target_lower_gap_to_truth: float
    passing_witness_random_state: int
    passing_witness_sigma_z_hat: float
    residual_fresh_random_state: int
    residual_fresh_sigma_z_hat: float
    sigma_ratio_vs_passing_witness: float
    sigma_ratio_vs_residual_fresh: float
    vf_cross_entry_label: str
    target_vf_cross_entry: float
    passing_witness_vf_cross_entry: float
    residual_fresh_vf_cross_entry: float
    runtime_witness_path: tuple[str, ...]
    next_runtime_focus: tuple[str, ...]
    driver_signature: str
    canonical_seed303_object_flow_blocker_digest: tuple[str, ...]

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
        self.current_rung_status = str(self.current_rung_status).strip()
        self.repair_agenda_signature = str(self.repair_agenda_signature).strip()
        self.target_random_state = int(self.target_random_state)
        self.target_seed_group = str(self.target_seed_group).strip().lower()
        self.target_z_index = int(self.target_z_index)
        self.target_z_value = float(self.target_z_value)
        self.target_truth_at_z = float(self.target_truth_at_z)
        self.target_bar_f_at_z = float(self.target_bar_f_at_z)
        self.target_sigma_z_hat = float(self.target_sigma_z_hat)
        self.target_pointwise_lower = float(self.target_pointwise_lower)
        self.target_pointwise_upper = float(self.target_pointwise_upper)
        self.target_pointwise_covered = bool(self.target_pointwise_covered)
        self.target_uniform_band_covered = bool(self.target_uniform_band_covered)
        self.target_center_gap_to_truth = float(self.target_center_gap_to_truth)
        self.target_lower_gap_to_truth = float(self.target_lower_gap_to_truth)
        self.passing_witness_random_state = int(self.passing_witness_random_state)
        self.passing_witness_sigma_z_hat = float(self.passing_witness_sigma_z_hat)
        self.residual_fresh_random_state = int(self.residual_fresh_random_state)
        self.residual_fresh_sigma_z_hat = float(self.residual_fresh_sigma_z_hat)
        self.sigma_ratio_vs_passing_witness = float(self.sigma_ratio_vs_passing_witness)
        self.sigma_ratio_vs_residual_fresh = float(self.sigma_ratio_vs_residual_fresh)
        self.vf_cross_entry_label = str(self.vf_cross_entry_label).strip()
        self.target_vf_cross_entry = float(self.target_vf_cross_entry)
        self.passing_witness_vf_cross_entry = float(self.passing_witness_vf_cross_entry)
        self.residual_fresh_vf_cross_entry = float(self.residual_fresh_vf_cross_entry)
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.next_runtime_focus = tuple(
            str(item).strip() for item in self.next_runtime_focus
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_object_flow_blocker_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_object_flow_blocker_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_seed303_object_flow_blocker_report(
    *,
    repair_agenda_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
    ),
    target_slice: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerSlice,
    passing_witness_slice: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerSlice,
    residual_fresh_slice: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerSlice,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerReport:
    next_required_slot = repair_agenda_report.next_required_slot
    if next_required_slot is None:
        raise ValueError("seed303 object-flow blocker requires an actionable next slot")
    if (
        repair_agenda_report.current_rung_status
        != "same-seed-exact-witness-observed-rerun-open"
    ):
        raise ValueError(
            "seed303 object-flow blocker expects the observed rerun to remain open"
        )
    if target_slice.random_state != next_required_slot.random_state:
        raise ValueError("target slice must match the actionable repair slot")
    if target_slice.seed_group != next_required_slot.seed_group:
        raise ValueError(
            "target slice must match the actionable repair slot seed group"
        )
    if target_slice.z_index != next_required_slot.z_index:
        raise ValueError("target slice must match the actionable repair slot z index")
    if abs(target_slice.z_value - next_required_slot.z_value) > 1e-12:
        raise ValueError("target slice must match the actionable repair slot z value")
    if passing_witness_slice.random_state == target_slice.random_state:
        raise ValueError("passing witness comparator must differ from the target seed")
    if residual_fresh_slice.random_state == target_slice.random_state:
        raise ValueError("residual fresh comparator must differ from the target seed")
    if (
        target_slice.z_index != passing_witness_slice.z_index
        or target_slice.z_index != residual_fresh_slice.z_index
    ):
        raise ValueError("all object-flow blocker slices must use the same z index")
    if (
        abs(target_slice.z_value - passing_witness_slice.z_value) > 1e-12
        or abs(target_slice.z_value - residual_fresh_slice.z_value) > 1e-12
    ):
        raise ValueError("all object-flow blocker slices must use the same z value")

    target_center_gap_to_truth = target_slice.bar_f_at_z - target_slice.truth_at_z
    target_lower_gap_to_truth = target_slice.pointwise_lower - target_slice.truth_at_z
    sigma_ratio_vs_passing_witness = (
        target_slice.sigma_z_hat / passing_witness_slice.sigma_z_hat
    )
    sigma_ratio_vs_residual_fresh = (
        target_slice.sigma_z_hat / residual_fresh_slice.sigma_z_hat
    )
    next_runtime_focus = (
        "bar_f_at_z0[1]",
        "v_f_hat[2,1]",
        "covariance(0.25, 0.15)",
    )
    driver_signature = _driver_signature(
        repair_agenda_report=repair_agenda_report,
        target_slice=target_slice,
        passing_witness_slice=passing_witness_slice,
        residual_fresh_slice=residual_fresh_slice,
        target_center_gap_to_truth=target_center_gap_to_truth,
        target_lower_gap_to_truth=target_lower_gap_to_truth,
        sigma_ratio_vs_passing_witness=sigma_ratio_vs_passing_witness,
        sigma_ratio_vs_residual_fresh=sigma_ratio_vs_residual_fresh,
    )
    canonical_digest = (
        f"- seed `303` witness-center `z = {_format_grid_value(target_slice.z_value)}` is a pointwise-only miss rather than a generic band-width blocker: truth `{_format_float(target_slice.truth_at_z)}`, `bar_f_at_z0[1] = {_format_float(target_slice.bar_f_at_z)}`, pointwise interval `[{_format_float(target_slice.pointwise_lower)}, {_format_float(target_slice.pointwise_upper)}]`, and the lower bound already sits `+{_format_float(target_lower_gap_to_truth)}` above truth while the uniform band still covers this slot",
        f"- same-z scale does not explain the miss: `sigma_z_hat[1] = {_format_float(target_slice.sigma_z_hat)}` for seed `303`, only `{_format_percent(sigma_ratio_vs_passing_witness)}` of passing witness seed `202` at `{_format_float(passing_witness_slice.sigma_z_hat)}` and `{_format_percent(sigma_ratio_vs_residual_fresh)}` of residual comparator seed `707` at `{_format_float(residual_fresh_slice.sigma_z_hat)}`",
        f"- the bounded runtime path remains `{' -> '.join(repair_agenda_report.runtime_witness_path)}`, but the next object-flow focus inside that path is `{next_runtime_focus[0]}` plus `{next_runtime_focus[1]}`: seed `303` flips `{next_runtime_focus[1]}` to `{target_slice.vf_cross_entry:+.3f}` while seeds `202` / `707` keep `{passing_witness_slice.vf_cross_entry:+.3f}` / `{residual_fresh_slice.vf_cross_entry:+.3f}`, so seed `707` should stay queued until the seed `303` overshoot is explained",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; keep live entry at `trigger2-policy-spec`, treat seed `303` as an object-flow blocker rather than a band-widening blocker, and only then spend on seed `707` residual closure",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-exact-witness-seed303-object-flow-blocker"
        ),
        policy_digest=repair_agenda_report.policy_digest,
        binding_design=repair_agenda_report.binding_design,
        window_label=repair_agenda_report.window_label,
        same_seed_random_states=repair_agenda_report.same_seed_random_states,
        current_rung_status=repair_agenda_report.current_rung_status,
        repair_agenda_signature=repair_agenda_report.driver_signature,
        target_random_state=target_slice.random_state,
        target_seed_group=target_slice.seed_group,
        target_z_index=target_slice.z_index,
        target_z_value=target_slice.z_value,
        target_truth_at_z=target_slice.truth_at_z,
        target_bar_f_at_z=target_slice.bar_f_at_z,
        target_sigma_z_hat=target_slice.sigma_z_hat,
        target_pointwise_lower=target_slice.pointwise_lower,
        target_pointwise_upper=target_slice.pointwise_upper,
        target_pointwise_covered=target_slice.pointwise_covered,
        target_uniform_band_covered=target_slice.uniform_band_covered,
        target_center_gap_to_truth=target_center_gap_to_truth,
        target_lower_gap_to_truth=target_lower_gap_to_truth,
        passing_witness_random_state=passing_witness_slice.random_state,
        passing_witness_sigma_z_hat=passing_witness_slice.sigma_z_hat,
        residual_fresh_random_state=residual_fresh_slice.random_state,
        residual_fresh_sigma_z_hat=residual_fresh_slice.sigma_z_hat,
        sigma_ratio_vs_passing_witness=sigma_ratio_vs_passing_witness,
        sigma_ratio_vs_residual_fresh=sigma_ratio_vs_residual_fresh,
        vf_cross_entry_label="v_f_hat[2,1]",
        target_vf_cross_entry=target_slice.vf_cross_entry,
        passing_witness_vf_cross_entry=passing_witness_slice.vf_cross_entry,
        residual_fresh_vf_cross_entry=residual_fresh_slice.vf_cross_entry,
        runtime_witness_path=repair_agenda_report.runtime_witness_path,
        next_runtime_focus=next_runtime_focus,
        driver_signature=driver_signature,
        canonical_seed303_object_flow_blocker_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_seed303_object_flow_blocker() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_seed303_object_flow_blocker_report(
        repair_agenda_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda(),
        target_slice=Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerSlice(
            random_state=303,
            seed_group="witness",
            z_index=1,
            z_value=0.15,
            truth_at_z=1.161834242728283,
            bar_f_at_z=9.185821021130248,
            sigma_z_hat=4.087257631350298,
            pointwise_lower=2.4628804819186296,
            pointwise_upper=15.908761560341867,
            pointwise_covered=False,
            uniform_band_covered=True,
            vf_cross_entry=-287.5573494984548,
        ),
        passing_witness_slice=Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerSlice(
            random_state=202,
            seed_group="witness",
            z_index=1,
            z_value=0.15,
            truth_at_z=1.161834242728283,
            bar_f_at_z=1.8536883342981971,
            sigma_z_hat=4.942953812519496,
            pointwise_lower=-6.2767471720781005,
            pointwise_upper=9.984123840674496,
            pointwise_covered=True,
            uniform_band_covered=True,
            vf_cross_entry=201.61306478911862,
        ),
        residual_fresh_slice=Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerSlice(
            random_state=707,
            seed_group="fresh",
            z_index=1,
            z_value=0.15,
            truth_at_z=1.161834242728283,
            bar_f_at_z=19.89800035979604,
            sigma_z_hat=16.764321445709392,
            pointwise_lower=-7.676854573559389,
            pointwise_upper=47.47285529315147,
            pointwise_covered=True,
            uniform_band_covered=True,
            vf_cross_entry=1132.2390947315928,
        ),
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerReport",
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessSeed303ObjectFlowBlockerSlice",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_seed303_object_flow_blocker_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_seed303_object_flow_blocker",
]
