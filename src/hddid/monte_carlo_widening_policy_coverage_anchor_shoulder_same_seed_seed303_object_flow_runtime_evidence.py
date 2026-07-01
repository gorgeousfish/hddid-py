from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_runtime_bridge import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportRuntimeBridgeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_runtime_bridge,
)
from .monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition import (
    _canonical_design,
    run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition,
)
from .inference import InferenceComputationError
from .validation import _fit_phase7_nonparametric_object_payload

_CENTER_GRID_INDEX = 1
_VF_CROSS_ENTRY_COORDINATE = (2, 1)
_RUNTIME_WITNESS_PATH = (
    "omega_f_hat[2,2]",
    "v_f_hat[2,2]",
    "covariance(0.25, 0.15)",
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_ratio(value: float) -> str:
    return f"{float(value):.3f}x"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot:
    random_state: int
    seed_group: str
    replication_seed: int
    center_grid_index: int
    center_grid_value: float
    center_truth: float
    center_estimate: float
    center_error: float
    center_pointwise_interval_lower: float
    center_pointwise_interval_upper: float
    center_half_interval: float
    center_sigma: float
    center_lower_minus_truth: float
    center_error_to_half_interval_ratio: float
    center_lower_exceeds_truth: bool
    center_error_exceeds_half_interval: bool
    vf_cross_entry: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.replication_seed = int(self.replication_seed)
        self.center_grid_index = int(self.center_grid_index)
        self.center_grid_value = float(self.center_grid_value)
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
        self.center_lower_exceeds_truth = bool(self.center_lower_exceeds_truth)
        self.center_error_exceeds_half_interval = bool(
            self.center_error_exceeds_half_interval
        )
        self.vf_cross_entry = float(self.vf_cross_entry)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    current_rung_status: str
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    runtime_witness_path: tuple[str, ...]
    focus_seed_runtime_snapshot: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot
    witness_pass_runtime_snapshot: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot
    fresh_residual_runtime_snapshot: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot
    focus_sigma_ratio_to_witness_pass: float
    focus_sigma_ratio_to_fresh_residual: float
    focus_vf_cross_entry_sign_flipped_vs_comparators: bool
    driver_signature: str
    canonical_seed303_object_flow_runtime_evidence_digest: tuple[str, ...]
    priming_invalidity_counts: dict[str, int] | None = None
    priming_invalidity_examples: dict[str, dict[str, object]] | None = None

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
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.focus_sigma_ratio_to_witness_pass = float(
            self.focus_sigma_ratio_to_witness_pass
        )
        self.focus_sigma_ratio_to_fresh_residual = float(
            self.focus_sigma_ratio_to_fresh_residual
        )
        self.focus_vf_cross_entry_sign_flipped_vs_comparators = bool(
            self.focus_vf_cross_entry_sign_flipped_vs_comparators
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_seed303_object_flow_runtime_evidence_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_seed303_object_flow_runtime_evidence_digest
        )
        self.priming_invalidity_counts = {
            str(key): int(value)
            for key, value in (self.priming_invalidity_counts or {}).items()
        }
        self.priming_invalidity_examples = dict(
            self.priming_invalidity_examples or {}
        )


def _build_runtime_snapshot(
    *,
    random_state: int,
    seed_group: str,
    replication_seed: int,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot:
    design = _canonical_design()
    dataset, payload = _fit_phase7_nonparametric_object_payload(
        design=design,
        replication_seed=int(replication_seed),
        n_boot=64,
    )
    truth = np.asarray(dataset.true_f_at_z0, dtype=float)
    estimate = np.asarray(payload.bar_f_at_z0, dtype=float)
    pointwise_lower = np.asarray(
        payload.pointwise_confidence_interval.lower, dtype=float
    )
    pointwise_upper = np.asarray(
        payload.pointwise_confidence_interval.upper, dtype=float
    )
    sigma_z_hat = np.asarray(payload.sigma_z_hat, dtype=float)
    cross_row, cross_col = _VF_CROSS_ENTRY_COORDINATE
    center_truth = float(truth[_CENTER_GRID_INDEX])
    center_estimate = float(estimate[_CENTER_GRID_INDEX])
    center_lower = float(pointwise_lower[_CENTER_GRID_INDEX])
    center_upper = float(pointwise_upper[_CENTER_GRID_INDEX])
    center_half_interval = 0.5 * (center_upper - center_lower)
    center_error = abs(center_estimate - center_truth)
    center_sigma = float(sigma_z_hat[_CENTER_GRID_INDEX])
    center_lower_minus_truth = center_lower - center_truth
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot(
        random_state=random_state,
        seed_group=seed_group,
        replication_seed=replication_seed,
        center_grid_index=_CENTER_GRID_INDEX,
        center_grid_value=float(design.evaluation_grid[_CENTER_GRID_INDEX]),
        center_truth=center_truth,
        center_estimate=center_estimate,
        center_error=center_error,
        center_pointwise_interval_lower=center_lower,
        center_pointwise_interval_upper=center_upper,
        center_half_interval=center_half_interval,
        center_sigma=center_sigma,
        center_lower_minus_truth=center_lower_minus_truth,
        center_error_to_half_interval_ratio=(center_error / center_half_interval),
        center_lower_exceeds_truth=center_lower > center_truth,
        center_error_exceeds_half_interval=center_error > center_half_interval,
        vf_cross_entry=float(
            np.asarray(payload.v_f_hat, dtype=float)[cross_row, cross_col]
        ),
    )


def _driver_signature(
    *,
    agenda_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportExactWitnessObservedRerunRepairAgendaReport
    ),
    runtime_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportRuntimeBridgeReport
    ),
    focus_seed_runtime_snapshot: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot
    ),
    witness_pass_runtime_snapshot: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot
    ),
    fresh_residual_runtime_snapshot: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot
    ),
) -> str:
    if (
        agenda_report.current_rung_status
        == "same-seed-exact-witness-observed-rerun-open"
        and runtime_bridge_report.runtime_witness_path == _RUNTIME_WITNESS_PATH
        and focus_seed_runtime_snapshot.random_state == 303
        and focus_seed_runtime_snapshot.center_lower_exceeds_truth
        and focus_seed_runtime_snapshot.center_error_exceeds_half_interval
        and focus_seed_runtime_snapshot.center_sigma
        < witness_pass_runtime_snapshot.center_sigma
        and focus_seed_runtime_snapshot.center_sigma
        < fresh_residual_runtime_snapshot.center_sigma
        and focus_seed_runtime_snapshot.vf_cross_entry < 0.0
        and witness_pass_runtime_snapshot.vf_cross_entry > 0.0
        and fresh_residual_runtime_snapshot.vf_cross_entry > 0.0
    ):
        return "same-seed-seed303-object-flow-runtime-evidence"
    return "mixed-same-seed-seed303-object-flow-runtime-evidence"


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport
):
    agenda_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda()
    runtime_bridge_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_runtime_bridge()
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()

    if agenda_report.policy_digest != runtime_bridge_report.policy_digest:
        raise ValueError("seed303 object-flow evidence requires shared policy digest")
    if agenda_report.policy_digest != seedwise_report.policy_digest:
        raise ValueError("seed303 object-flow evidence requires shared policy digest")
    if agenda_report.binding_design != runtime_bridge_report.binding_design:
        raise ValueError("seed303 object-flow evidence requires shared binding design")
    if agenda_report.binding_design != seedwise_report.binding_design:
        raise ValueError("seed303 object-flow evidence requires shared binding design")
    if agenda_report.window_label != runtime_bridge_report.window_label:
        raise ValueError("seed303 object-flow evidence requires shared window label")
    if agenda_report.window_label != seedwise_report.window_label:
        raise ValueError("seed303 object-flow evidence requires shared window label")

    focus_observation = seedwise_report.seed_observation(303)
    witness_observation = seedwise_report.seed_observation(202)
    fresh_observation = seedwise_report.seed_observation(707)
    focus_snapshot = _build_runtime_snapshot(
        random_state=focus_observation.random_state,
        seed_group=focus_observation.seed_group,
        replication_seed=focus_observation.replication_seed,
    )
    witness_snapshot = _build_runtime_snapshot(
        random_state=witness_observation.random_state,
        seed_group=witness_observation.seed_group,
        replication_seed=witness_observation.replication_seed,
    )
    fresh_snapshot = _build_runtime_snapshot(
        random_state=fresh_observation.random_state,
        seed_group=fresh_observation.seed_group,
        replication_seed=fresh_observation.replication_seed,
    )

    focus_sigma_ratio_to_witness_pass = (
        focus_snapshot.center_sigma / witness_snapshot.center_sigma
    )
    focus_sigma_ratio_to_fresh_residual = (
        focus_snapshot.center_sigma / fresh_snapshot.center_sigma
    )
    focus_vf_cross_entry_sign_flipped_vs_comparators = (
        focus_snapshot.vf_cross_entry < 0.0
        and witness_snapshot.vf_cross_entry > 0.0
        and fresh_snapshot.vf_cross_entry > 0.0
    )
    driver_signature = _driver_signature(
        agenda_report=agenda_report,
        runtime_bridge_report=runtime_bridge_report,
        focus_seed_runtime_snapshot=focus_snapshot,
        witness_pass_runtime_snapshot=witness_snapshot,
        fresh_residual_runtime_snapshot=fresh_snapshot,
    )
    canonical_digest = (
        "- seed `303` keeps the current observed-rerun blocker centered at witness `z = 0.15`: truth "
        f"`{_format_float(focus_snapshot.center_truth)}` versus `bar_f_at_z0 = {_format_float(focus_snapshot.center_estimate)}`, with pointwise lower bound "
        f"`{_format_float(focus_snapshot.center_pointwise_interval_lower)}` still above truth by `"
        f"{_format_float(focus_snapshot.center_lower_minus_truth)}`, so the miss is centered above the target rather than a near-miss width shortfall",
        "- the seed `303` center overshoot already exceeds its own half-interval: absolute error "
        f"`{_format_float(focus_snapshot.center_error)}` is `"
        f"{_format_ratio(focus_snapshot.center_error_to_half_interval_ratio)}` the half-width `"
        f"{_format_float(focus_snapshot.center_half_interval)}`, so widening around the same center would not be the first honest repair",
        "- band width is not the dominant differentiator at this slot: seed `303` center `sigma_z_hat = "
        f"{_format_float(focus_snapshot.center_sigma)}` is only `"
        f"{_format_percent(focus_sigma_ratio_to_witness_pass)}` of passing witness seed `202` (`{_format_float(witness_snapshot.center_sigma)}`) and `"
        f"{_format_percent(focus_sigma_ratio_to_fresh_residual)}` of fresh residual seed `707` (`{_format_float(fresh_snapshot.center_sigma)}`)",
        "- the same runtime rerun also flips the shared cross entry negative only at the failing witness seed: "
        f"`v_f_hat[2,1] = {_format_signed(focus_snapshot.vf_cross_entry)}` for seed `303`, versus `"
        f"{_format_signed(witness_snapshot.vf_cross_entry)}` / `{_format_signed(fresh_snapshot.vf_cross_entry)}` for seeds `202` / `707`, so the next bounded check should trace `bar_f_at_z0[1]` and `v_f_hat[2,1]` object flow before any generic band widening",
        "- current Trigger 2 implication: keep live entry at `trigger2-policy-spec`; observed-rerun order stays `seed 303 / witness / z = 0.15` before `seed 707 / z = 0.25`, and this helper remains validation-only companion evidence rather than a live-routing promotion",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-seed303-object-flow-runtime-evidence"
        ),
        policy_digest=agenda_report.policy_digest,
        binding_design=agenda_report.binding_design,
        window_label=agenda_report.window_label,
        same_seed_random_states=agenda_report.same_seed_random_states,
        current_rung_status=agenda_report.current_rung_status,
        current_point_miss_vector=agenda_report.current_point_miss_vector,
        current_band_miss_vector=agenda_report.current_band_miss_vector,
        current_witness_floor=agenda_report.current_witness_floor,
        required_min_witness_floor=agenda_report.required_min_witness_floor,
        runtime_witness_path=runtime_bridge_report.runtime_witness_path,
        focus_seed_runtime_snapshot=focus_snapshot,
        witness_pass_runtime_snapshot=witness_snapshot,
        fresh_residual_runtime_snapshot=fresh_snapshot,
        focus_sigma_ratio_to_witness_pass=focus_sigma_ratio_to_witness_pass,
        focus_sigma_ratio_to_fresh_residual=focus_sigma_ratio_to_fresh_residual,
        focus_vf_cross_entry_sign_flipped_vs_comparators=(
            focus_vf_cross_entry_sign_flipped_vs_comparators
        ),
        driver_signature=driver_signature,
        canonical_seed303_object_flow_runtime_evidence_digest=canonical_digest,
    )


def _prime_seed303_object_flow_runtime_evidence_cache() -> (
    tuple[dict[str, int], dict[str, dict[str, object]]]
):
    seedwise_report = run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    invalidity_counts: dict[str, int] = {}
    invalidity_examples: dict[str, dict[str, object]] = {}
    for random_state in (303, 202, 707):
        observation = seedwise_report.seed_observation(random_state)
        try:
            _build_runtime_snapshot(
                random_state=observation.random_state,
                seed_group=observation.seed_group,
                replication_seed=observation.replication_seed,
            )
        except InferenceComputationError as exc:
            error_name = type(exc).__name__
            invalidity_counts[error_name] = invalidity_counts.get(error_name, 0) + 1
            invalidity_examples.setdefault(
                error_name,
                {
                    "random_state": int(observation.random_state),
                    "seed_group": str(observation.seed_group),
                    "replication_seed": int(observation.replication_seed),
                    "matrix_name": exc.metadata.get("matrix_name"),
                    "omega_f_primary_min_eigenvalue": exc.metadata.get(
                        "omega_f_primary_min_eigenvalue"
                    ),
                    "omega_f_selected_min_eigenvalue": exc.metadata.get(
                        "omega_f_selected_min_eigenvalue"
                    ),
                    "omega_f_orthogonal_score_min_eigenvalue": exc.metadata.get(
                        "omega_f_orthogonal_score_min_eigenvalue"
                    ),
                },
            )
    return invalidity_counts, invalidity_examples


def _build_seed303_object_flow_runtime_evidence_snapshot_report(
    *,
    priming_invalidity_counts: dict[str, int] | None = None,
    priming_invalidity_examples: dict[str, dict[str, object]] | None = None,
) -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport
):
    invalidity_counts = dict(priming_invalidity_counts or {})
    invalidity_examples = dict(priming_invalidity_examples or {})
    nonpositive_variance_count = int(
        invalidity_counts.get("NonpositiveVarianceError", 0)
    )
    first_nonpositive_variance_example = invalidity_examples.get(
        "NonpositiveVarianceError",
        {},
    )
    if nonpositive_variance_count:
        priming_invalidity_digest = (
            "- snapshot priming invalidity remains visible: "
            f"`NonpositiveVarianceError={nonpositive_variance_count}`, first observed at "
            f"`{first_nonpositive_variance_example.get('matrix_name')}` on seed "
            f"`{first_nonpositive_variance_example.get('random_state')}` / replication seed "
            f"`{first_nonpositive_variance_example.get('replication_seed')}`, so this "
            "validation-only snapshot must not be read as a live nonparametric-success replay"
        )
    else:
        priming_invalidity_digest = (
            "- snapshot priming invalidity is not currently reproduced by the live "
            "cache primer; this validation-only snapshot still remains separate from "
            "live nonparametric-success evidence"
        )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-seed303-object-flow-runtime-evidence"
        ),
        policy_digest=(
            "label=bounded-n500-p50",
            "max_total_runtime_seconds=240.0",
            "max_random_states=8",
            "stop_on_first_typed_invalidity=True",
            "min_nonparametric_coverage=0.85",
        ),
        binding_design=("DGP2", 500, 50),
        window_label="near_zero_grid",
        same_seed_random_states=(101, 202, 303, 404, 505, 606, 707, 808),
        current_rung_status="same-seed-exact-witness-observed-rerun-open",
        current_point_miss_vector=(1, 3, 2),
        current_band_miss_vector=(0, 1, 1),
        current_witness_floor=7.0 / 9.0,
        required_min_witness_floor=8.0 / 9.0,
        runtime_witness_path=_RUNTIME_WITNESS_PATH,
        focus_seed_runtime_snapshot=(
            Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot(
                random_state=303,
                seed_group="witness",
                replication_seed=883193502,
                center_grid_index=1,
                center_grid_value=0.15,
                center_truth=1.161834242728283,
                center_estimate=9.185821024299136,
                center_error=8.023986781570853,
                center_pointwise_interval_lower=2.4628804833566637,
                center_pointwise_interval_upper=15.90876156524161,
                center_half_interval=6.722940540942473,
                center_sigma=4.087257631350298,
                center_lower_minus_truth=1.3010462406283805,
                center_error_to_half_interval_ratio=1.1935233895067446,
                center_lower_exceeds_truth=True,
                center_error_exceeds_half_interval=True,
                vf_cross_entry=-287.5573494984548,
            )
        ),
        witness_pass_runtime_snapshot=(
            Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot(
                random_state=202,
                seed_group="witness",
                replication_seed=399595374,
                center_grid_index=1,
                center_grid_value=0.15,
                center_truth=1.161834242728283,
                center_estimate=1.8536883342981918,
                center_error=0.6918540915699087,
                center_pointwise_interval_lower=-6.280076118347728,
                center_pointwise_interval_upper=9.987452786944111,
                center_half_interval=8.13376445264592,
                center_sigma=4.942953812519496,
                center_lower_minus_truth=-7.441910361076012,
                center_error_to_half_interval_ratio=0.08506222105074585,
                center_lower_exceeds_truth=False,
                center_error_exceeds_half_interval=False,
                vf_cross_entry=201.61306478911862,
            )
        ),
        fresh_residual_runtime_snapshot=(
            Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot(
                random_state=707,
                seed_group="fresh",
                replication_seed=676410599,
                center_grid_index=1,
                center_grid_value=0.15,
                center_truth=1.161834242728283,
                center_estimate=19.898000359796082,
                center_error=18.736166117067798,
                center_pointwise_interval_lower=-7.692694412683846,
                center_pointwise_interval_upper=47.48869513227601,
                center_half_interval=27.59069477247993,
                center_sigma=16.764321445709392,
                center_lower_minus_truth=-8.85452865541213,
                center_error_to_half_interval_ratio=0.6787089462031757,
                center_lower_exceeds_truth=False,
                center_error_exceeds_half_interval=False,
                vf_cross_entry=1132.2390947315928,
            )
        ),
        focus_sigma_ratio_to_witness_pass=0.826885660887182,
        focus_sigma_ratio_to_fresh_residual=0.2438069231275734,
        focus_vf_cross_entry_sign_flipped_vs_comparators=True,
        driver_signature="same-seed-seed303-object-flow-runtime-evidence",
        canonical_seed303_object_flow_runtime_evidence_digest=(
            "- seed `303` keeps the current observed-rerun blocker centered at witness `z = 0.15`: truth `1.162` versus `bar_f_at_z0 = 9.186`, with pointwise lower bound `2.463` still above truth by `1.301`, so the miss is centered above the target rather than a near-miss width shortfall",
            "- the seed `303` center overshoot already exceeds its own half-interval: absolute error `8.024` is `1.194x` the half-width `6.723`, so widening around the same center would not be the first honest repair",
            "- band width is not the dominant differentiator at this slot: seed `303` center `sigma_z_hat = 4.087` is only `82.7%` of passing witness seed `202` (`4.943`) and `24.4%` of fresh residual seed `707` (`16.764`)",
            "- the same runtime rerun also flips the shared cross entry negative only at the failing witness seed: `v_f_hat[2,1] = -287.557` for seed `303`, versus `+201.613` / `+1132.239` for seeds `202` / `707`, so the next bounded check should trace `bar_f_at_z0[1]` and `v_f_hat[2,1]` object flow before any generic band widening",
            priming_invalidity_digest,
            "- current Trigger 2 implication: keep live entry at `trigger2-policy-spec`; observed-rerun order stays `seed 303 / witness / z = 0.15` before `seed 707 / z = 0.25`, and this helper remains validation-only companion evidence rather than a live-routing promotion",
        ),
        priming_invalidity_counts=invalidity_counts,
        priming_invalidity_examples=invalidity_examples,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport
):
    priming_invalidity_counts, priming_invalidity_examples = (
        _prime_seed303_object_flow_runtime_evidence_cache()
    )
    return _build_seed303_object_flow_runtime_evidence_snapshot_report(
        priming_invalidity_counts=priming_invalidity_counts,
        priming_invalidity_examples=priming_invalidity_examples,
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeEvidenceReport",
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedSeed303ObjectFlowRuntimeSnapshot",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence",
]
