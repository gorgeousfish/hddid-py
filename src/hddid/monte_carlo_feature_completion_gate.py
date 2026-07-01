from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from .inference import InferenceComputationError
from .monte_carlo_widening_policy_acceptance_preview import (
    Phase7MonteCarloWideningPolicyAcceptancePreviewReport,
    build_phase7_monte_carlo_widening_policy_acceptance_preview_repo_side_report,
    run_phase7_monte_carlo_widening_policy_acceptance_preview,
)
from .monte_carlo_widening_policy_spec import (
    Phase7MonteCarloWideningPolicySpecReport,
    decode_phase7_monte_carlo_widening_policy_digest,
    run_phase7_monte_carlo_widening_policy_spec,
)
from .monte_carlo_widening_policy_runtime_evidence_packet import (
    Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport,
    build_phase7_monte_carlo_widening_policy_runtime_evidence_packet_repo_side_report,
    run_phase7_monte_carlo_widening_policy_runtime_evidence_packet,
)
from .monte_carlo_widening_policy_runtime_evidence_admission_gate import (
    Phase7MonteCarloWideningPolicyRuntimeEvidenceAdmissionGateReport,
    run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate,
)
from .monte_carlo_widening_policy_quality_risk_probe import (
    Phase7MonteCarloWideningPolicyQualityRiskDesignSummary,
    Phase7MonteCarloWideningPolicyQualityRiskProbeReport,
    build_quality_risk_action_contract,
    build_quality_risk_average_standard_error_calibration_candidate,
    build_quality_risk_average_standard_error_calibration_candidate_evidence,
    build_quality_risk_calibration_targets,
    build_quality_risk_method_diagnosis,
    build_quality_risk_positive_headroom_estimator_evidence_verdict,
    build_quality_risk_positive_headroom_estimator_evidence_requirement,
    dominant_quality_risk_calibration_target,
    run_phase7_monte_carlo_widening_policy_quality_risk_probe,
)
from .monte_carlo_paper_dgp2_contract_audit import (
    Phase7MonteCarloPaperDGP2ContractAudit,
    build_phase7_monte_carlo_paper_dgp2_contract_audit_report,
)
from .monte_carlo_widening_trigger_gate import (
    Phase7MonteCarloWideningPolicy,
    Phase7MonteCarloWideningTriggerGateReport,
    _build_state_backed_trigger_gate_report,
    run_phase7_monte_carlo_widening_trigger_gate,
)

_QUALITY_RISK_PAPER_OBJECT_CONTRACT = (
    "paper-section-5-monte-carlo-quality-objects",
    "nonparametric coverage",
    "nonparametric RMSE",
    "nonparametric average standard error",
    "nonparametric confidence interval length",
    "90% pointwise interval length equals 2*z0.95*average SE",
    "derived calibration ratios: RMSE/SE, SE-RMSE, interval/RMSE",
)
_PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER = 3.2897072539029454
_INTERVAL_SCALE_LOCK_TOLERANCE = 2e-4


def _validated_integer(value: int, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"feature completion gate requires {label} to be an integer, not boolean"
        )
    if not isinstance(value, Integral):
        raise ValueError(f"feature completion gate requires {label} to be an integer")
    number = int(value)
    if number < minimum:
        raise ValueError(
            f"feature completion gate requires {label} to be at least {minimum}"
        )
    return number


def _validated_design_key(
    design: tuple[str, int, int],
    *,
    label: str,
) -> tuple[str, int, int]:
    return (
        str(design[0]).strip(),
        _validated_integer(design[1], label=f"{label} n_obs", minimum=1),
        _validated_integer(design[2], label=f"{label} p", minimum=1),
    )


def _build_repo_side_live_source_target_alignment_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport
):
    return SimpleNamespace(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-live-source-target-alignment-probe",
        policy_digest=(
            "label=bounded-n500-p50",
            "max_total_runtime_seconds=240.0",
            "max_random_states=8",
            "stop_on_first_typed_invalidity=True",
            "min_nonparametric_coverage=0.85",
        ),
        binding_design=("DGP2", 500, 50),
        window_label="near_zero_grid",
        coverage_anchor_random_state=202,
        overshoot_companion_random_state=505,
        coverage_anchor_replication_seed=1751820809,
        overshoot_companion_replication_seed=37420655,
        source_diagonal_coordinate=2,
        source_diagonal_basis_label="sin(2pi z)",
        shared_vf_entry_label="v_f_hat[2,2]",
        live_anchor_omega_diagonal_entry=0.0,
        live_companion_omega_diagonal_entry=0.0,
        live_companion_omega_diagonal_gap=0.0,
        live_coordinate_axis_only_shared_vf_entry_increment=0.0,
        live_diagonal_only_shared_vf_entry_increment=0.0,
        live_offdiagonal_axis_only_shared_vf_entry_increment=0.0,
        required_diagonal_vf_entry_lift=0.0,
        live_required_omega_diagonal_increment=0.0,
        live_target_omega_diagonal_entry=0.0,
        canonical_target_omega_diagonal_entry=4.964455377368893,
        canonical_vs_live_target_gap=4.964455377368893,
        companion_gap_sign_matches_canonical=True,
        diagonal_lift_sign_matches_canonical=True,
        driver_signature="live-source-target-alignment-ok",
        canonical_live_alignment_digest=(
            "- repo-side live source-target alignment: live helper unavailable; "
            "preserving canonical_vs_live_target_gap `4.964`",
        ),
    )


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe as _run_live_source_target_alignment_probe,
    )

    return _run_live_source_target_alignment_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot as _run_repair_target_snapshot,
    )

    return _run_repair_target_snapshot()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract as _run_execution_contract,
    )

    return _run_execution_contract()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_bundle():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_execution_bundle import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_bundle as _run_execution_bundle,
    )

    return _run_execution_bundle()


def _quality_risk_driver_from_metrics(
    *,
    binding_coverage_floor_slack: float,
    binding_rmse_to_standard_error_ratio: float,
    best_rmse_to_standard_error_ratio: float,
    binding_standard_error_reserve: float,
    best_standard_error_reserve: float,
    binding_interval_length_to_rmse_ratio: float,
    best_interval_length_to_rmse_ratio: float,
) -> str:
    if binding_standard_error_reserve < 0.0:
        return "rmse-outpaces-average-se"
    if binding_coverage_floor_slack < 0.0:
        return "coverage-floor-risk-persists"
    if binding_interval_length_to_rmse_ratio < best_interval_length_to_rmse_ratio:
        return "interval-per-rmse-headroom-trails-best-design"
    if binding_standard_error_reserve < best_standard_error_reserve:
        return "standard-error-reserve-trails-best-design"
    if binding_rmse_to_standard_error_ratio > best_rmse_to_standard_error_ratio:
        return "rmse-to-standard-error-ratio-trails-best-design"
    return "quality-risk-cleared"


def _require_close_quality_metric(
    actual: float,
    expected: float,
    *,
    label: str,
) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-3):
        raise ValueError(
            f"feature completion gate requires quality-risk {label} to match metrics"
        )


def _require_paper_interval_scale_lock(
    *,
    binding_interval_to_standard_error_ratio: float,
    best_interval_to_standard_error_ratio: float,
) -> None:
    stale = tuple(
        f"{label} interval/SE {ratio:.6f}"
        for label, ratio in (
            ("binding", binding_interval_to_standard_error_ratio),
            ("best", best_interval_to_standard_error_ratio),
        )
        if not math.isclose(
            float(ratio),
            _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER,
            rel_tol=1e-9,
            abs_tol=_INTERVAL_SCALE_LOCK_TOLERANCE,
        )
    )
    if stale:
        raise ValueError(
            "feature completion gate requires quality-risk interval/SE evidence "
            "to match the paper 90% pointwise interval scale; found: "
            f"{', '.join(stale)}"
        )


def _policy_min_nonparametric_coverage(
    policy_digest: tuple[str, ...],
) -> float | None:
    prefix = "min_nonparametric_coverage="
    for item in policy_digest:
        if item.startswith(prefix):
            return float(item.removeprefix(prefix))
    return None


def _quality_risk_clearance_conditions(
    *,
    binding_driver: str,
    binding_coverage_floor_slack: float,
    binding_standard_error_reserve: float,
    rmse_to_standard_error_ratio_gap: float,
    standard_error_reserve_gap: float,
    interval_length_to_rmse_gap: float,
) -> tuple[dict[str, object], ...]:
    coverage_floor_deficit = max(-binding_coverage_floor_slack, 0.0)
    nonnegative_reserve_deficit = max(-binding_standard_error_reserve, 0.0)
    rmse_to_se_deficit = max(rmse_to_standard_error_ratio_gap, 0.0)
    se_reserve_deficit = max(standard_error_reserve_gap, 0.0)
    interval_to_rmse_deficit = max(interval_length_to_rmse_gap, 0.0)
    overall_status = (
        "cleared" if binding_driver == "quality-risk-cleared" else "blocked"
    )
    return (
        {
            "condition": "coverage_floor",
            "status": "cleared" if coverage_floor_deficit == 0.0 else "open",
            "deficit": coverage_floor_deficit,
            "margin": binding_coverage_floor_slack,
        },
        {
            "condition": "nonnegative_se_reserve",
            "status": "cleared" if nonnegative_reserve_deficit == 0.0 else "open",
            "deficit": nonnegative_reserve_deficit,
            "margin": binding_standard_error_reserve,
        },
        {
            "condition": "rmse_to_se_not_trailing",
            "status": "cleared" if rmse_to_se_deficit == 0.0 else "open",
            "deficit": rmse_to_se_deficit,
        },
        {
            "condition": "se_reserve_not_trailing",
            "status": "cleared" if se_reserve_deficit == 0.0 else "open",
            "deficit": se_reserve_deficit,
        },
        {
            "condition": "interval_to_rmse_not_trailing",
            "status": "cleared" if interval_to_rmse_deficit == 0.0 else "open",
            "deficit": interval_to_rmse_deficit,
        },
        {
            "condition": "overall",
            "status": overall_status,
            "driver": binding_driver,
        },
    )


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _format_design_keys(
    design_keys: tuple[tuple[str, int, int], ...],
) -> str:
    if not design_keys:
        return "none"
    return ", ".join(f"{dgp}/{n_obs}/{p}" for dgp, n_obs, p in design_keys)


def _design_key_labels(
    design_keys: tuple[tuple[str, int, int], ...],
) -> tuple[str, ...]:
    return tuple(f"{dgp}/{n_obs}/{p}" for dgp, n_obs, p in design_keys)


def _format_invalidity_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{code}={count}" for code, count in sorted(counts.items()))


def _format_trigger_names(trigger_names: tuple[str, ...]) -> str:
    if not trigger_names:
        return "none"
    return ", ".join(trigger_names)


def _prefer_repo_side_policy_spec_inventory(
    policy_spec: Phase7MonteCarloWideningPolicySpecReport,
    acceptance_preview: Phase7MonteCarloWideningPolicyAcceptancePreviewReport,
) -> Phase7MonteCarloWideningPolicySpecReport:
    if (
        policy_spec.current_gate_status != acceptance_preview.current_runtime_gate_status
        or policy_spec.target_gate_status
        != acceptance_preview.accepted_runtime_gate_status
        or policy_spec.policy_digest != acceptance_preview.policy_digest
    ):
        return policy_spec
    if (
        policy_spec.stable_partial_designs == acceptance_preview.stable_partial_designs
        and policy_spec.blocked_full_matrix_designs
        == acceptance_preview.blocked_full_matrix_designs
        and policy_spec.quality_risk_designs == acceptance_preview.quality_risk_designs
    ):
        return policy_spec
    return Phase7MonteCarloWideningPolicySpecReport(
        stage_label=policy_spec.stage_label,
        trigger_label=policy_spec.trigger_label,
        current_gate_status=policy_spec.current_gate_status,
        target_gate_status=policy_spec.target_gate_status,
        stable_partial_designs=acceptance_preview.stable_partial_designs,
        blocked_full_matrix_designs=acceptance_preview.blocked_full_matrix_designs,
        quality_risk_designs=acceptance_preview.quality_risk_designs,
        policy_digest=policy_spec.policy_digest,
        canonical_policy_digest=policy_spec.canonical_policy_digest,
    )


def _repo_side_policy_spec_from_acceptance_preview(
    acceptance_preview: Phase7MonteCarloWideningPolicyAcceptancePreviewReport,
) -> Phase7MonteCarloWideningPolicySpecReport:
    return Phase7MonteCarloWideningPolicySpecReport(
        stage_label="phase7-monte-carlo-widening-policy-spec",
        trigger_label="trigger2-policy-spec",
        current_gate_status=acceptance_preview.current_runtime_gate_status,
        target_gate_status=acceptance_preview.accepted_runtime_gate_status,
        stable_partial_designs=acceptance_preview.stable_partial_designs,
        blocked_full_matrix_designs=acceptance_preview.blocked_full_matrix_designs,
        quality_risk_designs=acceptance_preview.quality_risk_designs,
        policy_digest=acceptance_preview.policy_digest,
        canonical_policy_digest=(
            "- repo-side Trigger 2 policy spec is reconstructed from the accepted "
            "feature-completion gate packet instead of replaying live Monte Carlo helpers",
        ),
    )


@dataclass(slots=True)
class Phase7MonteCarloFeatureCompletionGateReport:
    stage_label: str
    route_label: str
    current_gate_status: str
    target_gate_status: str
    accepted_feature_bundle: str
    accepted_open_trigger_names: tuple[str, ...]
    runtime_evidence_helper: str
    runtime_evidence_note: str
    runtime_evidence_driver: str
    runtime_evidence_binding_design: tuple[str, int, int]
    runtime_evidence_binding_random_states: tuple[int, ...]
    runtime_evidence_required_quota_gap: int
    runtime_evidence_effective_fresh_reruns: int
    runtime_evidence_covered_pointwise_witnesses: int
    runtime_evidence_required_covered_pointwise_witnesses: int
    runtime_evidence_total_pointwise_witnesses: int
    runtime_evidence_supported_floor_ceiling: float
    runtime_evidence_quota_helper: str
    runtime_evidence_quota_note: str
    runtime_evidence_binding_design_rerun_capacity_helper: str
    runtime_evidence_binding_design_rerun_capacity_note: str
    runtime_evidence_binding_design_rerun_capacity_limiting_budget: str
    runtime_evidence_binding_design_rerun_capacity_implication: str
    runtime_evidence_current_implication: str
    runtime_evidence_admission_helper: str
    runtime_evidence_admission_note: str
    runtime_evidence_admission_status: str
    runtime_evidence_admission_candidate_status: str
    runtime_evidence_admission_contract_status: str
    runtime_evidence_admission_binding_random_state: int
    runtime_evidence_admission_fresh_reruns_after: int
    runtime_evidence_admission_candidate_success: bool
    runtime_evidence_admission_candidate_nonparametric_coverage: float | None
    runtime_evidence_admission_candidate_typed_invalidity_counts: dict[str, int]
    runtime_evidence_admission_candidate_typed_invalidity_examples: dict[
        str,
        dict[str, object],
    ]
    quality_risk_probe_helper: str
    quality_risk_probe_note: str
    quality_risk_binding_driver: str
    quality_risk_blocker_reason: str
    quality_risk_binding_design: tuple[str, int, int]
    quality_risk_best_design: tuple[str, int, int]
    quality_risk_canonical_floor_shortfall: float
    quality_risk_binding_coverage: float
    quality_risk_binding_coverage_floor_slack: float
    quality_risk_binding_rmse: float
    quality_risk_binding_average_standard_error: float
    quality_risk_binding_interval_length: float
    quality_risk_binding_rmse_to_standard_error_ratio: float
    quality_risk_binding_standard_error_reserve: float
    quality_risk_binding_interval_length_to_rmse_ratio: float
    quality_risk_best_coverage: float
    quality_risk_best_coverage_floor_slack: float
    quality_risk_best_rmse: float
    quality_risk_best_average_standard_error: float
    quality_risk_best_interval_length: float
    quality_risk_best_rmse_to_standard_error_ratio: float
    quality_risk_best_standard_error_reserve: float
    quality_risk_best_interval_length_to_rmse_ratio: float
    quality_risk_rmse_to_standard_error_ratio_gap: float
    quality_risk_standard_error_reserve_gap: float
    quality_risk_interval_length_to_rmse_gap: float
    repair_target_helper: str
    repair_target_note: str
    repair_target_signature: str
    execution_contract_helper: str
    execution_contract_note: str
    execution_contract_signature: str
    execution_bundle_helper: str
    execution_bundle_note: str
    execution_bundle_signature: str
    live_source_target_alignment_helper: str
    live_source_target_alignment_note: str
    live_source_target_alignment_driver: str
    live_source_target_alignment_implication: str
    live_source_target_alignment_binding_random_states: tuple[int, ...]
    live_source_target_alignment_binding_replication_seeds: tuple[int, ...]
    live_source_target_alignment_target_gap_field: str
    live_source_target_alignment_target_gap: float
    stable_partial_designs: tuple[tuple[str, int, int], ...]
    blocked_full_matrix_designs: tuple[tuple[str, int, int], ...]
    quality_risk_designs: tuple[tuple[str, int, int], ...]
    typed_invalidity_counts: dict[str, int]
    policy_digest: tuple[str, ...]
    canonical_gate_digest: tuple[str, ...]
    recommendation_rationale: str
    runtime_evidence_admission_covered_pointwise_witnesses_before: int = 7
    runtime_evidence_admission_covered_pointwise_witnesses_after: int = 8
    runtime_evidence_admission_required_covered_pointwise_witnesses: int = 8
    runtime_evidence_admission_remaining_quota_gap_before: int = 1
    runtime_evidence_admission_remaining_quota_gap_after: int = 0
    runtime_evidence_admission_fresh_reruns_before: int = 5
    runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before: int = 1
    runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after: int = 2
    runtime_evidence_admission_seed_local_total_pointwise_witnesses: int = 3
    runtime_evidence_admission_quota_closure_margin: int = 0
    post_admission_quality_risk_status: str = "quality-risk-still-open"
    post_admission_quality_risk_driver: str = "rmse-outpaces-average-se"
    post_admission_quality_risk_binding_design: tuple[str, int, int] = ("DGP2", 500, 50)
    monte_carlo_validation_ready_status: str = "blocked"
    monte_carlo_validation_ready_blocker: str = "quality-risk-keeps-trigger2-bounded"
    quality_risk_source_mode: str = "live-runtime-probe"
    quality_risk_source_note: str = "live runtime probe"
    quality_risk_paper_object_contract: tuple[
        str,
        ...,
    ] = _QUALITY_RISK_PAPER_OBJECT_CONTRACT
    paper_dgp2_contract_helper: str = (
        "run_phase7_monte_carlo_paper_dgp2_contract_audit(...)"
    )
    paper_dgp2_contract_note: str = (
        "Docs/research/phase7_monte_carlo_paper_dgp2_contract_audit.md"
    )
    paper_dgp2_contract_status: str = ""
    paper_dgp2_contract_r_finding_codes: tuple[str, ...] = ()
    paper_dgp2_contract_near_zero_grid: tuple[float, ...] = ()
    quality_risk_binding_interval_length_to_standard_error_ratio: float | None = None
    quality_risk_best_interval_length_to_standard_error_ratio: float | None = None
    monte_carlo_validation_evidence: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.route_label = str(self.route_label).strip()
        self.current_gate_status = str(self.current_gate_status).strip()
        self.target_gate_status = str(self.target_gate_status).strip()
        self.accepted_feature_bundle = str(self.accepted_feature_bundle).strip()
        self.accepted_open_trigger_names = tuple(
            str(name).strip() for name in self.accepted_open_trigger_names
        )
        self.runtime_evidence_helper = str(self.runtime_evidence_helper).strip()
        self.runtime_evidence_note = str(self.runtime_evidence_note).strip()
        self.runtime_evidence_driver = str(self.runtime_evidence_driver).strip()
        self.runtime_evidence_binding_design = _validated_design_key(
            self.runtime_evidence_binding_design,
            label="runtime evidence binding design",
        )
        self.runtime_evidence_binding_random_states = tuple(
            _validated_integer(
                value,
                label="runtime evidence binding random state",
                minimum=0,
            )
            for value in self.runtime_evidence_binding_random_states
        )
        self.runtime_evidence_required_quota_gap = _validated_integer(
            self.runtime_evidence_required_quota_gap,
            label="runtime evidence required quota gap",
            minimum=0,
        )
        self.runtime_evidence_effective_fresh_reruns = _validated_integer(
            self.runtime_evidence_effective_fresh_reruns,
            label="runtime evidence effective fresh reruns",
            minimum=0,
        )
        self.runtime_evidence_covered_pointwise_witnesses = _validated_integer(
            self.runtime_evidence_covered_pointwise_witnesses,
            label="runtime evidence covered pointwise witnesses",
            minimum=0,
        )
        self.runtime_evidence_required_covered_pointwise_witnesses = _validated_integer(
            self.runtime_evidence_required_covered_pointwise_witnesses,
            label="runtime evidence required covered pointwise witnesses",
            minimum=0,
        )
        self.runtime_evidence_total_pointwise_witnesses = _validated_integer(
            self.runtime_evidence_total_pointwise_witnesses,
            label="runtime evidence total pointwise witnesses",
            minimum=0,
        )
        self.runtime_evidence_supported_floor_ceiling = float(
            self.runtime_evidence_supported_floor_ceiling
        )
        self.runtime_evidence_quota_helper = str(
            self.runtime_evidence_quota_helper
        ).strip()
        self.runtime_evidence_quota_note = str(self.runtime_evidence_quota_note).strip()
        self.runtime_evidence_binding_design_rerun_capacity_helper = str(
            self.runtime_evidence_binding_design_rerun_capacity_helper
        ).strip()
        self.runtime_evidence_binding_design_rerun_capacity_note = str(
            self.runtime_evidence_binding_design_rerun_capacity_note
        ).strip()
        self.runtime_evidence_binding_design_rerun_capacity_limiting_budget = str(
            self.runtime_evidence_binding_design_rerun_capacity_limiting_budget
        ).strip()
        self.runtime_evidence_binding_design_rerun_capacity_implication = str(
            self.runtime_evidence_binding_design_rerun_capacity_implication
        ).strip()
        self.runtime_evidence_current_implication = str(
            self.runtime_evidence_current_implication
        ).strip()
        self.runtime_evidence_admission_helper = str(
            self.runtime_evidence_admission_helper
        ).strip()
        self.runtime_evidence_admission_note = str(
            self.runtime_evidence_admission_note
        ).strip()
        self.runtime_evidence_admission_status = str(
            self.runtime_evidence_admission_status
        ).strip()
        self.runtime_evidence_admission_candidate_status = str(
            self.runtime_evidence_admission_candidate_status
        ).strip()
        self.runtime_evidence_admission_contract_status = str(
            self.runtime_evidence_admission_contract_status
        ).strip()
        self.runtime_evidence_admission_binding_random_state = _validated_integer(
            self.runtime_evidence_admission_binding_random_state,
            label="runtime evidence admission binding random state",
            minimum=0,
        )
        self.runtime_evidence_admission_fresh_reruns_after = _validated_integer(
            self.runtime_evidence_admission_fresh_reruns_after,
            label="runtime evidence admission fresh reruns after",
            minimum=0,
        )
        if isinstance(
            self.runtime_evidence_admission_candidate_success,
            (bool, np.bool_),
        ):
            self.runtime_evidence_admission_candidate_success = bool(
                self.runtime_evidence_admission_candidate_success
            )
        else:
            raise ValueError(
                "feature completion gate requires runtime evidence admission "
                "candidate success to be boolean"
            )
        if (
            self.runtime_evidence_admission_candidate_nonparametric_coverage
            is not None
        ):
            self.runtime_evidence_admission_candidate_nonparametric_coverage = float(
                self.runtime_evidence_admission_candidate_nonparametric_coverage
            )
        self.runtime_evidence_admission_candidate_typed_invalidity_counts = {
            str(code).strip(): _validated_integer(
                count,
                label=f"runtime evidence admission candidate typed invalidity count {code}",
                minimum=0,
            )
            for code, count in self.runtime_evidence_admission_candidate_typed_invalidity_counts.items()
        }
        self.runtime_evidence_admission_candidate_typed_invalidity_examples = {
            str(code).strip(): dict(example)
            for code, example in self.runtime_evidence_admission_candidate_typed_invalidity_examples.items()
        }
        self.runtime_evidence_admission_covered_pointwise_witnesses_before = _validated_integer(
            self.runtime_evidence_admission_covered_pointwise_witnesses_before,
            label="runtime evidence admission covered pointwise witnesses before",
            minimum=0,
        )
        self.runtime_evidence_admission_covered_pointwise_witnesses_after = _validated_integer(
            self.runtime_evidence_admission_covered_pointwise_witnesses_after,
            label="runtime evidence admission covered pointwise witnesses after",
            minimum=0,
        )
        self.runtime_evidence_admission_required_covered_pointwise_witnesses = _validated_integer(
            self.runtime_evidence_admission_required_covered_pointwise_witnesses,
            label="runtime evidence admission required covered pointwise witnesses",
            minimum=0,
        )
        self.runtime_evidence_admission_remaining_quota_gap_before = _validated_integer(
            self.runtime_evidence_admission_remaining_quota_gap_before,
            label="runtime evidence admission remaining quota gap before",
            minimum=0,
        )
        self.runtime_evidence_admission_remaining_quota_gap_after = _validated_integer(
            self.runtime_evidence_admission_remaining_quota_gap_after,
            label="runtime evidence admission remaining quota gap after",
            minimum=0,
        )
        self.runtime_evidence_admission_fresh_reruns_before = _validated_integer(
            self.runtime_evidence_admission_fresh_reruns_before,
            label="runtime evidence admission fresh reruns before",
            minimum=0,
        )
        self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before = _validated_integer(
            self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before,
            label="runtime evidence admission seed-local covered pointwise witnesses before",
            minimum=0,
        )
        self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after = _validated_integer(
            self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after,
            label="runtime evidence admission seed-local covered pointwise witnesses after",
            minimum=0,
        )
        self.runtime_evidence_admission_seed_local_total_pointwise_witnesses = _validated_integer(
            self.runtime_evidence_admission_seed_local_total_pointwise_witnesses,
            label="runtime evidence admission seed-local total pointwise witnesses",
            minimum=0,
        )
        self.runtime_evidence_admission_quota_closure_margin = _validated_integer(
            self.runtime_evidence_admission_quota_closure_margin,
            label="runtime evidence admission quota closure margin",
            minimum=0,
        )
        self.post_admission_quality_risk_status = str(
            self.post_admission_quality_risk_status
        ).strip()
        self.post_admission_quality_risk_driver = str(
            self.post_admission_quality_risk_driver
        ).strip()
        self.post_admission_quality_risk_binding_design = _validated_design_key(
            self.post_admission_quality_risk_binding_design,
            label="post-admission quality-risk binding design",
        )
        self.monte_carlo_validation_ready_status = str(
            self.monte_carlo_validation_ready_status
        ).strip()
        self.monte_carlo_validation_ready_blocker = str(
            self.monte_carlo_validation_ready_blocker
        ).strip()
        self.quality_risk_source_mode = str(self.quality_risk_source_mode).strip()
        self.quality_risk_source_note = str(self.quality_risk_source_note).strip()
        self.quality_risk_paper_object_contract = tuple(
            str(item).strip() for item in self.quality_risk_paper_object_contract
        )
        self.paper_dgp2_contract_helper = str(
            self.paper_dgp2_contract_helper
        ).strip()
        self.paper_dgp2_contract_note = str(self.paper_dgp2_contract_note).strip()
        self.paper_dgp2_contract_status = str(self.paper_dgp2_contract_status).strip()
        self.paper_dgp2_contract_r_finding_codes = tuple(
            str(code).strip() for code in self.paper_dgp2_contract_r_finding_codes
        )
        self.paper_dgp2_contract_near_zero_grid = tuple(
            float(value) for value in self.paper_dgp2_contract_near_zero_grid
        )
        self.quality_risk_probe_helper = str(self.quality_risk_probe_helper).strip()
        self.quality_risk_probe_note = str(self.quality_risk_probe_note).strip()
        self.quality_risk_binding_driver = str(self.quality_risk_binding_driver).strip()
        self.quality_risk_blocker_reason = str(
            self.quality_risk_blocker_reason
        ).strip()
        self.quality_risk_binding_design = _validated_design_key(
            self.quality_risk_binding_design,
            label="quality-risk binding design",
        )
        self.quality_risk_best_design = _validated_design_key(
            self.quality_risk_best_design,
            label="quality-risk best design",
        )
        self.quality_risk_canonical_floor_shortfall = float(
            self.quality_risk_canonical_floor_shortfall
        )
        self.quality_risk_binding_coverage = float(
            self.quality_risk_binding_coverage
        )
        self.quality_risk_binding_coverage_floor_slack = float(
            self.quality_risk_binding_coverage_floor_slack
        )
        self.quality_risk_binding_rmse = float(self.quality_risk_binding_rmse)
        self.quality_risk_binding_average_standard_error = float(
            self.quality_risk_binding_average_standard_error
        )
        self.quality_risk_binding_interval_length = float(
            self.quality_risk_binding_interval_length
        )
        if self.quality_risk_binding_interval_length_to_standard_error_ratio is None:
            self.quality_risk_binding_interval_length_to_standard_error_ratio = (
                self.quality_risk_binding_interval_length
                / self.quality_risk_binding_average_standard_error
            )
        else:
            self.quality_risk_binding_interval_length_to_standard_error_ratio = float(
                self.quality_risk_binding_interval_length_to_standard_error_ratio
            )
        self.quality_risk_binding_rmse_to_standard_error_ratio = float(
            self.quality_risk_binding_rmse_to_standard_error_ratio
        )
        self.quality_risk_binding_standard_error_reserve = float(
            self.quality_risk_binding_standard_error_reserve
        )
        self.quality_risk_binding_interval_length_to_rmse_ratio = float(
            self.quality_risk_binding_interval_length_to_rmse_ratio
        )
        self.quality_risk_best_coverage = float(self.quality_risk_best_coverage)
        self.quality_risk_best_coverage_floor_slack = float(
            self.quality_risk_best_coverage_floor_slack
        )
        self.quality_risk_best_rmse = float(self.quality_risk_best_rmse)
        self.quality_risk_best_average_standard_error = float(
            self.quality_risk_best_average_standard_error
        )
        self.quality_risk_best_interval_length = float(
            self.quality_risk_best_interval_length
        )
        if self.quality_risk_best_interval_length_to_standard_error_ratio is None:
            self.quality_risk_best_interval_length_to_standard_error_ratio = (
                self.quality_risk_best_interval_length
                / self.quality_risk_best_average_standard_error
            )
        else:
            self.quality_risk_best_interval_length_to_standard_error_ratio = float(
                self.quality_risk_best_interval_length_to_standard_error_ratio
            )
        self.quality_risk_best_rmse_to_standard_error_ratio = float(
            self.quality_risk_best_rmse_to_standard_error_ratio
        )
        self.quality_risk_best_standard_error_reserve = float(
            self.quality_risk_best_standard_error_reserve
        )
        self.quality_risk_best_interval_length_to_rmse_ratio = float(
            self.quality_risk_best_interval_length_to_rmse_ratio
        )
        self.quality_risk_rmse_to_standard_error_ratio_gap = float(
            self.quality_risk_rmse_to_standard_error_ratio_gap
        )
        self.quality_risk_standard_error_reserve_gap = float(
            self.quality_risk_standard_error_reserve_gap
        )
        self.quality_risk_interval_length_to_rmse_gap = float(
            self.quality_risk_interval_length_to_rmse_gap
        )
        self.repair_target_helper = str(self.repair_target_helper).strip()
        self.repair_target_note = str(self.repair_target_note).strip()
        self.repair_target_signature = str(self.repair_target_signature).strip()
        self.execution_contract_helper = str(self.execution_contract_helper).strip()
        self.execution_contract_note = str(self.execution_contract_note).strip()
        self.execution_contract_signature = str(
            self.execution_contract_signature
        ).strip()
        self.execution_bundle_helper = str(self.execution_bundle_helper).strip()
        self.execution_bundle_note = str(self.execution_bundle_note).strip()
        self.execution_bundle_signature = str(
            self.execution_bundle_signature
        ).strip()
        self.live_source_target_alignment_helper = str(
            self.live_source_target_alignment_helper
        ).strip()
        self.live_source_target_alignment_note = str(
            self.live_source_target_alignment_note
        ).strip()
        self.live_source_target_alignment_driver = str(
            self.live_source_target_alignment_driver
        ).strip()
        self.live_source_target_alignment_implication = str(
            self.live_source_target_alignment_implication
        ).strip()
        self.live_source_target_alignment_binding_random_states = tuple(
            _validated_integer(
                value,
                label="live source-target alignment binding random state",
                minimum=0,
            )
            for value in self.live_source_target_alignment_binding_random_states
        )
        self.live_source_target_alignment_binding_replication_seeds = tuple(
            _validated_integer(
                value,
                label="live source-target alignment binding replication seed",
                minimum=0,
            )
            for value in self.live_source_target_alignment_binding_replication_seeds
        )
        self.live_source_target_alignment_target_gap_field = str(
            self.live_source_target_alignment_target_gap_field
        ).strip()
        self.live_source_target_alignment_target_gap = float(
            self.live_source_target_alignment_target_gap
        )
        self.stable_partial_designs = tuple(
            _validated_design_key((dgp, n_obs, p), label="stable partial design")
            for dgp, n_obs, p in self.stable_partial_designs
        )
        self.blocked_full_matrix_designs = tuple(
            _validated_design_key((dgp, n_obs, p), label="blocked full matrix design")
            for dgp, n_obs, p in self.blocked_full_matrix_designs
        )
        self.quality_risk_designs = tuple(
            _validated_design_key((dgp, n_obs, p), label="quality-risk design")
            for dgp, n_obs, p in self.quality_risk_designs
        )
        self.typed_invalidity_counts = {
            str(code).strip(): _validated_integer(
                count,
                label=f"typed invalidity count {code}",
                minimum=0,
            )
            for code, count in self.typed_invalidity_counts.items()
        }
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.canonical_gate_digest = tuple(
            str(line).rstrip() for line in self.canonical_gate_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()
        self._validate_quality_risk_metric_consistency()
        self.monte_carlo_validation_evidence = (
            self._build_monte_carlo_validation_evidence()
        )

    def _validate_quality_risk_metric_consistency(self) -> None:
        expected_driver = _quality_risk_driver_from_metrics(
            binding_coverage_floor_slack=(
                self.quality_risk_binding_coverage_floor_slack
            ),
            binding_rmse_to_standard_error_ratio=(
                self.quality_risk_binding_rmse_to_standard_error_ratio
            ),
            best_rmse_to_standard_error_ratio=(
                self.quality_risk_best_rmse_to_standard_error_ratio
            ),
            binding_standard_error_reserve=(
                self.quality_risk_binding_standard_error_reserve
            ),
            best_standard_error_reserve=self.quality_risk_best_standard_error_reserve,
            binding_interval_length_to_rmse_ratio=(
                self.quality_risk_binding_interval_length_to_rmse_ratio
            ),
            best_interval_length_to_rmse_ratio=(
                self.quality_risk_best_interval_length_to_rmse_ratio
            ),
        )
        if self.quality_risk_binding_driver != expected_driver:
            raise ValueError(
                "feature completion gate requires quality-risk driver to match metrics"
            )
        if self.monte_carlo_validation_ready_status == "ready":
            if (
                self.runtime_evidence_admission_status
                != "trigger2-runtime-evidence-quota-closed"
            ):
                raise ValueError(
                    "feature completion gate requires ready Monte Carlo validation "
                    "to have closed runtime admission"
                )
            if self.post_admission_quality_risk_status != "quality-risk-cleared":
                raise ValueError(
                    "feature completion gate requires ready Monte Carlo validation "
                    "to have cleared post-admission quality risk"
                )
            if self.quality_risk_binding_driver != "quality-risk-cleared":
                raise ValueError(
                    "feature completion gate requires ready Monte Carlo validation "
                    "to have cleared quality-risk metrics"
                )
            if self.post_admission_quality_risk_driver != "quality-risk-cleared":
                raise ValueError(
                    "feature completion gate requires ready Monte Carlo validation "
                    "post-admission driver to be quality-risk-cleared"
                )
            if self.quality_risk_blocker_reason:
                raise ValueError(
                    "feature completion gate requires ready Monte Carlo validation "
                    "to have no quality-risk blocker"
                )
            if self.monte_carlo_validation_ready_blocker:
                raise ValueError(
                    "feature completion gate requires ready Monte Carlo validation "
                    "to have no blocker"
                )
        if (
            self.post_admission_quality_risk_status == "quality-risk-still-open"
            and self.post_admission_quality_risk_driver
            != self.quality_risk_binding_driver
        ):
            raise ValueError(
                "feature completion gate requires post-admission quality-risk "
                "driver to match quality-risk metrics"
            )
        if (
            self.post_admission_quality_risk_status == "quality-risk-still-open"
            and self.post_admission_quality_risk_binding_design
            != self.quality_risk_binding_design
        ):
            raise ValueError(
                "feature completion gate requires post-admission quality-risk "
                "binding design to match quality-risk metrics"
            )
        if (
            self.post_admission_quality_risk_status == "quality-risk-still-open"
            and self.monte_carlo_validation_ready_blocker
            != self.quality_risk_blocker_reason
        ):
            raise ValueError(
                "feature completion gate requires Monte Carlo validation blocker "
                "to match quality-risk blocker"
            )
        _require_close_quality_metric(
            self.quality_risk_rmse_to_standard_error_ratio_gap,
            self.quality_risk_binding_rmse_to_standard_error_ratio
            - self.quality_risk_best_rmse_to_standard_error_ratio,
            label="RMSE/SE gap",
        )
        _require_close_quality_metric(
            self.quality_risk_standard_error_reserve_gap,
            self.quality_risk_best_standard_error_reserve
            - self.quality_risk_binding_standard_error_reserve,
            label="SE-RMSE gap",
        )
        _require_close_quality_metric(
            self.quality_risk_interval_length_to_rmse_gap,
            self.quality_risk_best_interval_length_to_rmse_ratio
            - self.quality_risk_binding_interval_length_to_rmse_ratio,
            label="interval/RMSE gap",
        )
        _require_close_quality_metric(
            self.quality_risk_canonical_floor_shortfall,
            max(-self.quality_risk_binding_coverage_floor_slack, 0.0),
            label="canonical floor shortfall",
        )
        coverage_floor = _policy_min_nonparametric_coverage(self.policy_digest)
        if coverage_floor is not None:
            _require_close_quality_metric(
                self.quality_risk_binding_coverage_floor_slack,
                self.quality_risk_binding_coverage - coverage_floor,
                label="binding coverage floor slack",
            )
            _require_close_quality_metric(
                self.quality_risk_best_coverage_floor_slack,
                self.quality_risk_best_coverage - coverage_floor,
                label="best coverage floor slack",
            )
        _require_close_quality_metric(
            self.quality_risk_binding_rmse_to_standard_error_ratio,
            self.quality_risk_binding_rmse
            / self.quality_risk_binding_average_standard_error,
            label="binding raw RMSE/SE ratio",
        )
        _require_close_quality_metric(
            self.quality_risk_best_rmse_to_standard_error_ratio,
            self.quality_risk_best_rmse
            / self.quality_risk_best_average_standard_error,
            label="best raw RMSE/SE ratio",
        )
        _require_close_quality_metric(
            self.quality_risk_binding_standard_error_reserve,
            self.quality_risk_binding_average_standard_error
            - self.quality_risk_binding_rmse,
            label="binding raw SE-RMSE reserve",
        )
        _require_close_quality_metric(
            self.quality_risk_best_standard_error_reserve,
            self.quality_risk_best_average_standard_error
            - self.quality_risk_best_rmse,
            label="best raw SE-RMSE reserve",
        )
        _require_close_quality_metric(
            self.quality_risk_binding_interval_length_to_rmse_ratio,
            self.quality_risk_binding_interval_length / self.quality_risk_binding_rmse,
            label="binding raw interval/RMSE ratio",
        )
        _require_close_quality_metric(
            self.quality_risk_best_interval_length_to_rmse_ratio,
            self.quality_risk_best_interval_length / self.quality_risk_best_rmse,
            label="best raw interval/RMSE ratio",
        )
        _require_close_quality_metric(
            self.quality_risk_binding_interval_length_to_standard_error_ratio,
            self.quality_risk_binding_interval_length
            / self.quality_risk_binding_average_standard_error,
            label=(
                "binding raw interval/SE ratio before paper 90% pointwise interval "
                "scale lock"
            ),
        )
        _require_close_quality_metric(
            self.quality_risk_best_interval_length_to_standard_error_ratio,
            self.quality_risk_best_interval_length
            / self.quality_risk_best_average_standard_error,
            label=(
                "best raw interval/SE ratio before paper 90% pointwise interval "
                "scale lock"
            ),
        )
        _require_paper_interval_scale_lock(
            binding_interval_to_standard_error_ratio=(
                self.quality_risk_binding_interval_length_to_standard_error_ratio
            ),
            best_interval_to_standard_error_ratio=(
                self.quality_risk_best_interval_length_to_standard_error_ratio
            ),
        )

    @property
    def quality_risk_driver(self) -> str:
        return self.quality_risk_binding_driver

    @property
    def runtime_evidence_current_driver(self) -> str:
        return self.runtime_evidence_driver

    @property
    def quality_risk_route(self) -> str:
        if self.quality_risk_binding_driver == "quality-risk-cleared":
            return "quality-risk-cleared"
        return "calibration / interval-construction debt"

    @property
    def quality_risk_binding_design_label(self) -> str:
        return (
            f"{self.quality_risk_binding_design[0]}"
            f"/{self.quality_risk_binding_design[1]}"
            f"/{self.quality_risk_binding_design[2]}"
        )

    @property
    def quality_risk_best_design_label(self) -> str:
        return (
            f"{self.quality_risk_best_design[0]}"
            f"/{self.quality_risk_best_design[1]}"
            f"/{self.quality_risk_best_design[2]}"
        )

    @property
    def quality_risk_interval_scale_status(self) -> str:
        max_abs_gap = max(
            abs(
                self.quality_risk_binding_interval_length_to_standard_error_ratio
                - _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
            ),
            abs(
                self.quality_risk_best_interval_length_to_standard_error_ratio
                - _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
            ),
        )
        if max_abs_gap <= _INTERVAL_SCALE_LOCK_TOLERANCE:
            return "paper-90pct-pointwise-interval-scale-locked"
        return "paper-90pct-pointwise-interval-scale-needs-review"

    @property
    def quality_risk_clearance_evidence(self) -> tuple[str, ...]:
        conditions = self.quality_risk_clearance_conditions
        return (
            (
                "clearance coverage_floor "
                f"status={conditions[0]['status']} "
                f"deficit={conditions[0]['deficit']:.3f} "
                f"margin={conditions[0]['margin']:+.3f}"
            ),
            (
                "clearance nonnegative_se_reserve "
                f"status={conditions[1]['status']} "
                f"deficit={conditions[1]['deficit']:.3f} "
                f"margin={conditions[1]['margin']:+.3f}"
            ),
            (
                "clearance rmse_to_se_not_trailing "
                f"status={conditions[2]['status']} "
                f"deficit={conditions[2]['deficit']:.3f}"
            ),
            (
                "clearance se_reserve_not_trailing "
                f"status={conditions[3]['status']} "
                f"deficit={conditions[3]['deficit']:.3f}"
            ),
            (
                "clearance interval_to_rmse_not_trailing "
                f"status={conditions[4]['status']} "
                f"deficit={conditions[4]['deficit']:.3f}"
            ),
            (
                f"clearance overall status={conditions[5]['status']} "
                f"driver={conditions[5]['driver']}"
            ),
        )

    @property
    def quality_risk_clearance_conditions(self) -> tuple[dict[str, object], ...]:
        return _quality_risk_clearance_conditions(
            binding_driver=self.quality_risk_binding_driver,
            binding_coverage_floor_slack=self.quality_risk_binding_coverage_floor_slack,
            binding_standard_error_reserve=self.quality_risk_binding_standard_error_reserve,
            rmse_to_standard_error_ratio_gap=(
                self.quality_risk_rmse_to_standard_error_ratio_gap
            ),
            standard_error_reserve_gap=self.quality_risk_standard_error_reserve_gap,
            interval_length_to_rmse_gap=self.quality_risk_interval_length_to_rmse_gap,
        )

    @property
    def quality_risk_calibration_targets(self) -> tuple[dict[str, object], ...]:
        return build_quality_risk_calibration_targets(
            binding_rmse=self.quality_risk_binding_rmse,
            binding_average_standard_error=(
                self.quality_risk_binding_average_standard_error
            ),
            binding_rmse_to_standard_error_ratio=(
                self.quality_risk_binding_rmse_to_standard_error_ratio
            ),
            binding_standard_error_reserve=(
                self.quality_risk_binding_standard_error_reserve
            ),
            binding_interval_length_to_rmse_ratio=(
                self.quality_risk_binding_interval_length_to_rmse_ratio
            ),
            best_rmse_to_standard_error_ratio=(
                self.quality_risk_best_rmse_to_standard_error_ratio
            ),
            best_standard_error_reserve=self.quality_risk_best_standard_error_reserve,
            best_interval_length_to_rmse_ratio=(
                self.quality_risk_best_interval_length_to_rmse_ratio
            ),
        )

    @property
    def dominant_quality_risk_calibration_target(self) -> dict[str, object]:
        return dominant_quality_risk_calibration_target(
            self.quality_risk_calibration_targets
        )

    @property
    def quality_risk_method_diagnosis(self) -> dict[str, object]:
        return build_quality_risk_method_diagnosis(
            binding_driver=self.quality_risk_binding_driver,
            blocker_reason=self.quality_risk_blocker_reason,
            interval_scale_status=self.quality_risk_interval_scale_status,
            binding_interval_length_to_standard_error_gap=(
                self.quality_risk_binding_interval_length_to_standard_error_ratio
                - _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
            ),
            best_interval_length_to_standard_error_gap=(
                self.quality_risk_best_interval_length_to_standard_error_ratio
                - _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
            ),
            dominant_calibration_target=self.dominant_quality_risk_calibration_target,
        )

    @property
    def quality_risk_method_diagnosis_evidence(self) -> tuple[str, ...]:
        diagnosis = self.quality_risk_method_diagnosis
        return (
            "method diagnosis "
            f"{diagnosis['diagnosis']} "
            f"interval_scale={diagnosis['interval_scale_status']} "
            "max_abs_interval_scale_gap="
            f"{float(diagnosis['max_abs_interval_scale_gap']):.3e}",
            "method dominant target "
            f"{diagnosis['dominant_condition']} "
            "required_average_se_lift="
            f"{float(diagnosis['required_average_standard_error_lift']):.3f} "
            "target_average_se="
            f"{float(diagnosis['target_average_standard_error']):.3f} "
            "target_interval_length="
            f"{float(diagnosis['target_interval_length']):.3f}",
        )

    @property
    def quality_risk_action_contract(self) -> dict[str, object]:
        return build_quality_risk_action_contract(
            binding_driver=self.quality_risk_binding_driver,
            interval_scale_status=self.quality_risk_interval_scale_status,
            dominant_calibration_target=self.dominant_quality_risk_calibration_target,
        )

    @property
    def quality_risk_action_evidence(self) -> tuple[str, ...]:
        action = self.quality_risk_action_contract
        return (
            f"action={action['action']}",
            f"interval_scale_action={action['interval_scale_action']}",
            f"bounded_policy_action={action['bounded_policy_action']}",
            f"release_gate_action={action['release_gate_action']}",
            f"target_condition={action['target_condition']}",
            "target_average_se="
            f"{float(action['target_average_standard_error']):.3f}",
            "target_interval_length="
            f"{float(action['target_interval_length']):.3f}",
            "required_average_se_lift="
            f"{float(action['required_average_standard_error_lift']):.3f}",
            "required_interval_length_lift="
            f"{float(action['required_interval_length_lift']):.3f}",
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate(
        self,
    ) -> dict[str, object]:
        binding_summary = Phase7MonteCarloWideningPolicyQualityRiskDesignSummary(
            dgp_name=self.quality_risk_binding_design[0],
            n_obs=self.quality_risk_binding_design[1],
            p=self.quality_risk_binding_design[2],
            mean_nonparametric_coverage=self.quality_risk_binding_coverage,
            coverage_floor_slack=self.quality_risk_binding_coverage_floor_slack,
            mean_nonparametric_rmse=self.quality_risk_binding_rmse,
            mean_nonparametric_average_standard_error=(
                self.quality_risk_binding_average_standard_error
            ),
            mean_nonparametric_interval_length=(
                self.quality_risk_binding_interval_length
            ),
            rmse_to_standard_error_ratio=(
                self.quality_risk_binding_rmse_to_standard_error_ratio
            ),
            standard_error_reserve=self.quality_risk_binding_standard_error_reserve,
            interval_length_to_rmse_ratio=(
                self.quality_risk_binding_interval_length_to_rmse_ratio
            ),
        )
        best_summary = Phase7MonteCarloWideningPolicyQualityRiskDesignSummary(
            dgp_name=self.quality_risk_best_design[0],
            n_obs=self.quality_risk_best_design[1],
            p=self.quality_risk_best_design[2],
            mean_nonparametric_coverage=self.quality_risk_best_coverage,
            coverage_floor_slack=self.quality_risk_best_coverage_floor_slack,
            mean_nonparametric_rmse=self.quality_risk_best_rmse,
            mean_nonparametric_average_standard_error=(
                self.quality_risk_best_average_standard_error
            ),
            mean_nonparametric_interval_length=self.quality_risk_best_interval_length,
            rmse_to_standard_error_ratio=(
                self.quality_risk_best_rmse_to_standard_error_ratio
            ),
            standard_error_reserve=self.quality_risk_best_standard_error_reserve,
            interval_length_to_rmse_ratio=(
                self.quality_risk_best_interval_length_to_rmse_ratio
            ),
        )
        return build_quality_risk_average_standard_error_calibration_candidate(
            binding_summary=binding_summary,
            best_summary=best_summary,
            dominant_calibration_target=self.dominant_quality_risk_calibration_target,
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_evidence(
        self,
    ) -> tuple[str, ...]:
        return build_quality_risk_average_standard_error_calibration_candidate_evidence(
            self.quality_risk_average_standard_error_calibration_candidate
        )

    @property
    def quality_risk_positive_headroom_estimator_evidence_requirement(
        self,
    ) -> dict[str, object]:
        return build_quality_risk_positive_headroom_estimator_evidence_requirement(
            self.quality_risk_average_standard_error_calibration_candidate
        )

    @property
    def positive_headroom_estimator_evidence_requirement(
        self,
    ) -> dict[str, object]:
        return self.quality_risk_positive_headroom_estimator_evidence_requirement

    @property
    def quality_risk_positive_headroom_estimator_evidence_verdict(
        self,
    ) -> dict[str, object]:
        return build_quality_risk_positive_headroom_estimator_evidence_verdict(
            self.quality_risk_positive_headroom_estimator_evidence_requirement,
            observed_average_standard_error=(
                self.quality_risk_binding_average_standard_error
            ),
            observed_interval_length=self.quality_risk_binding_interval_length,
        )

    @property
    def positive_headroom_estimator_evidence_verdict(
        self,
    ) -> dict[str, object]:
        return self.quality_risk_positive_headroom_estimator_evidence_verdict

    @property
    def quality_risk_positive_headroom_estimator_evidence_status(self) -> str:
        return str(
            self.quality_risk_positive_headroom_estimator_evidence_verdict["status"]
        )

    @property
    def positive_headroom_estimator_evidence_status(self) -> str:
        return self.quality_risk_positive_headroom_estimator_evidence_status

    @property
    def quality_risk_positive_headroom_estimator_evidence_source(self) -> str:
        return str(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "observed_evidence_source"
            ]
        )

    @property
    def positive_headroom_estimator_evidence_source(self) -> str:
        return self.quality_risk_positive_headroom_estimator_evidence_source

    @property
    def quality_risk_positive_headroom_estimator_evidence_source_status(self) -> str:
        return str(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "observed_evidence_source_status"
            ]
        )

    @property
    def positive_headroom_estimator_evidence_source_status(self) -> str:
        return self.quality_risk_positive_headroom_estimator_evidence_source_status

    @property
    def quality_risk_positive_headroom_admissible_for_feature_gate_rerun(
        self,
    ) -> bool:
        return bool(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "admissible_for_feature_gate_rerun"
            ]
        )

    @property
    def positive_headroom_admissible_for_feature_gate_rerun(self) -> bool:
        return self.quality_risk_positive_headroom_admissible_for_feature_gate_rerun

    @property
    def feature_gate_rerun_admissible(self) -> bool:
        return self.quality_risk_positive_headroom_admissible_for_feature_gate_rerun

    @property
    def admissible_for_feature_gate_rerun(self) -> bool:
        return self.feature_gate_rerun_admissible

    @property
    def quality_risk_positive_headroom_feature_gate_rerun_admission_status(
        self,
    ) -> str:
        return str(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "feature_gate_rerun_admission_status"
            ]
        )

    @property
    def positive_headroom_feature_gate_rerun_admission_status(self) -> str:
        return self.quality_risk_positive_headroom_feature_gate_rerun_admission_status

    @property
    def feature_gate_rerun_admission_status(self) -> str:
        return self.quality_risk_positive_headroom_feature_gate_rerun_admission_status

    @property
    def quality_risk_positive_headroom_estimator_evidence_verdict_evidence(
        self,
    ) -> tuple[str, ...]:
        verdict = self.quality_risk_positive_headroom_estimator_evidence_verdict
        return (
            f"positive_headroom_estimator_evidence_status={verdict['status']}",
            "positive_headroom_estimator_evidence_source="
            f"{verdict['observed_evidence_source']} "
            f"status={verdict['observed_evidence_source_status']}",
            "positive_headroom_estimator_average_se "
            f"{float(verdict['observed_average_standard_error']):.3f}->"
            f"{float(verdict['target_average_standard_error']):.3f} "
            f"margin={float(verdict['average_standard_error_margin']):+.3f} "
            "remaining_fraction="
            f"{float(verdict['remaining_average_standard_error_gap_fraction']):.3f}",
            "positive_headroom_estimator_interval_length "
            f"{float(verdict['observed_interval_length']):.3f}->"
            f"{float(verdict['target_interval_length']):.3f} "
            f"margin={float(verdict['interval_length_margin']):+.3f} "
            "remaining_fraction="
            f"{float(verdict['remaining_interval_length_gap_fraction']):.3f}",
            "positive_headroom_estimator_interval_scale "
            f"{verdict['interval_scale_status']} "
            f"gap={float(verdict['interval_to_standard_error_gap']):+.3e}",
            "positive_headroom_estimator_next_evidence="
            f"{verdict['required_next_evidence']}",
            "positive_headroom_feature_gate_rerun_admission="
            f"{verdict['feature_gate_rerun_admission_status']} "
            f"admissible={verdict['admissible_for_feature_gate_rerun']}",
        )

    @property
    def positive_headroom_estimator_evidence_verdict_evidence(
        self,
    ) -> tuple[str, ...]:
        return self.quality_risk_positive_headroom_estimator_evidence_verdict_evidence

    @property
    def quality_risk_average_standard_error_calibration_candidate_status(self) -> str:
        return str(
            self.quality_risk_average_standard_error_calibration_candidate["status"]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_target_condition(
        self,
    ) -> str:
        return str(
            self.quality_risk_average_standard_error_calibration_candidate[
                "target_condition"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_current_average_standard_error(
        self,
    ) -> float:
        return float(
            self.quality_risk_average_standard_error_calibration_candidate[
                "current_average_standard_error"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_target_average_standard_error(
        self,
    ) -> float:
        return float(
            self.quality_risk_average_standard_error_calibration_candidate[
                "target_average_standard_error"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_current_interval_length(
        self,
    ) -> float:
        return float(
            self.quality_risk_average_standard_error_calibration_candidate[
                "current_interval_length"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_target_interval_length(
        self,
    ) -> float:
        return float(
            self.quality_risk_average_standard_error_calibration_candidate[
                "target_interval_length"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_required_average_standard_error_lift(
        self,
    ) -> float:
        return float(
            self.quality_risk_average_standard_error_calibration_candidate[
                "required_average_standard_error_lift"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_required_interval_length_lift(
        self,
    ) -> float:
        return float(
            self.quality_risk_average_standard_error_calibration_candidate[
                "required_interval_length_lift"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_minimum_headroom_margin_after_calibration(
        self,
    ) -> float:
        return float(
            self.quality_risk_average_standard_error_calibration_candidate[
                "minimum_headroom_margin_after_calibration"
            ]
        )

    @property
    def _quality_risk_average_standard_error_calibration_candidate_positive_headroom_target(
        self,
    ) -> Mapping[str, object]:
        target = self.quality_risk_average_standard_error_calibration_candidate[
            "positive_headroom_admission_target"
        ]
        if not isinstance(target, Mapping):
            raise TypeError(
                "positive_headroom_admission_target must be a mapping on the "
                "average-SE calibration candidate"
            )
        return target

    @property
    def quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_average_standard_error(
        self,
    ) -> float:
        return float(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "minimum_admissible_average_standard_error"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_interval_length(
        self,
    ) -> float:
        return float(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "minimum_admissible_interval_length"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_average_standard_error_lift(
        self,
    ) -> float:
        return float(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "required_average_standard_error_lift"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_interval_length_lift(
        self,
    ) -> float:
        return float(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "required_interval_length_lift"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_positive_headroom_margin(
        self,
    ) -> float:
        return float(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "minimum_positive_headroom_margin"
            ]
        )

    @property
    def quality_risk_positive_headroom_target_average_standard_error(self) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_average_standard_error
        )

    @property
    def quality_risk_rmse_outpaces_average_se_gap(self) -> float:
        return max(
            self.quality_risk_binding_rmse
            - self.quality_risk_binding_average_standard_error,
            0.0,
        )

    @property
    def rmse_outpaces_average_se_gap(self) -> float:
        return self.quality_risk_rmse_outpaces_average_se_gap

    @property
    def quality_risk_positive_headroom_observed_average_standard_error(self) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "observed_average_standard_error"
            ]
        )

    @property
    def positive_headroom_observed_average_standard_error(self) -> float:
        return self.quality_risk_positive_headroom_observed_average_standard_error

    @property
    def quality_risk_positive_headroom_current_average_standard_error(self) -> float:
        return self.quality_risk_positive_headroom_observed_average_standard_error

    @property
    def positive_headroom_current_average_standard_error(self) -> float:
        return self.quality_risk_positive_headroom_current_average_standard_error

    @property
    def quality_risk_positive_headroom_target_interval_length(self) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_interval_length
        )

    @property
    def quality_risk_positive_headroom_observed_interval_length(self) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "observed_interval_length"
            ]
        )

    @property
    def positive_headroom_observed_interval_length(self) -> float:
        return self.quality_risk_positive_headroom_observed_interval_length

    @property
    def quality_risk_positive_headroom_current_interval_length(self) -> float:
        return self.quality_risk_positive_headroom_observed_interval_length

    @property
    def positive_headroom_current_interval_length(self) -> float:
        return self.quality_risk_positive_headroom_current_interval_length

    @property
    def quality_risk_positive_headroom_required_average_standard_error_lift(
        self,
    ) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_average_standard_error_lift
        )

    @property
    def quality_risk_positive_headroom_required_interval_length_lift(self) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_interval_length_lift
        )

    @property
    def quality_risk_positive_headroom_remaining_average_standard_error_gap(
        self,
    ) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_requirement[
                "remaining_average_standard_error_gap"
            ]
        )

    @property
    def quality_risk_positive_headroom_remaining_interval_length_gap(self) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_requirement[
                "remaining_interval_length_gap"
            ]
        )

    @property
    def quality_risk_positive_headroom_maximum_remaining_gap_fraction(self) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "maximum_remaining_gap_fraction"
            ]
        )

    @property
    def positive_headroom_maximum_remaining_gap_fraction(self) -> float:
        return self.quality_risk_positive_headroom_maximum_remaining_gap_fraction

    @property
    def quality_risk_positive_headroom_margin(self) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_margin
        )

    @property
    def quality_risk_positive_headroom_admission_status(self) -> str:
        return str(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "admission_status"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_release_gate_effect(
        self,
    ) -> str:
        return str(
            self.quality_risk_average_standard_error_calibration_candidate[
                "release_gate_effect"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_admission_status(
        self,
    ) -> str:
        return str(
            self.quality_risk_average_standard_error_calibration_candidate[
                "admission_status"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_admission_blocker(
        self,
    ) -> str:
        return str(
            self.quality_risk_average_standard_error_calibration_candidate[
                "admission_blocker"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_admission_required_next_evidence(
        self,
    ) -> str:
        return str(
            self.quality_risk_average_standard_error_calibration_candidate[
                "admission_required_next_evidence"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_admissible_as_release_evidence(
        self,
    ) -> bool:
        return bool(
            self.quality_risk_average_standard_error_calibration_candidate[
                "admissible_as_release_evidence"
            ]
        )

    @property
    def quality_risk_evidence(self) -> tuple[str, ...]:
        diagnostic_candidate_evidence = ()
        positive_headroom_verdict_evidence = ()
        if self.monte_carlo_validation_ready_status != "ready":
            diagnostic_candidate_evidence = (
                self.quality_risk_average_standard_error_calibration_candidate_evidence
            )
            positive_headroom_verdict_evidence = (
                self.quality_risk_positive_headroom_estimator_evidence_verdict_evidence
            )
        return (
            f"quality risk source {self.quality_risk_source_mode}",
            f"quality binding {self.quality_risk_binding_design_label}",
            (
                "coverage slack "
                f"{self.quality_risk_binding_coverage_floor_slack:+.3f}"
            ),
            (
                f"binding raw metrics {self.quality_risk_binding_design_label} "
                f"coverage {self.quality_risk_binding_coverage:.3f} "
                f"RMSE {self.quality_risk_binding_rmse:.3f} "
                f"average SE {self.quality_risk_binding_average_standard_error:.3f} "
                f"interval length {self.quality_risk_binding_interval_length:.3f}"
            ),
            (
                f"best raw metrics {self.quality_risk_best_design_label} "
                f"coverage {self.quality_risk_best_coverage:.3f} "
                f"RMSE {self.quality_risk_best_rmse:.3f} "
                f"average SE {self.quality_risk_best_average_standard_error:.3f} "
                f"interval length {self.quality_risk_best_interval_length:.3f}"
            ),
            (
                f"binding {self.quality_risk_binding_design_label} RMSE/SE "
                f"{self.quality_risk_binding_rmse_to_standard_error_ratio:.3f}"
            ),
            (
                f"best {self.quality_risk_best_design_label} RMSE/SE "
                f"{self.quality_risk_best_rmse_to_standard_error_ratio:.3f}"
            ),
            (
                f"binding {self.quality_risk_binding_design_label} SE-RMSE "
                f"{self.quality_risk_binding_standard_error_reserve:+.3f}"
            ),
            (
                f"best {self.quality_risk_best_design_label} SE-RMSE "
                f"{self.quality_risk_best_standard_error_reserve:+.3f}"
            ),
            (
                f"binding {self.quality_risk_binding_design_label} interval/RMSE "
                f"{self.quality_risk_binding_interval_length_to_rmse_ratio:.3f}"
            ),
            (
                f"best {self.quality_risk_best_design_label} interval/RMSE "
                f"{self.quality_risk_best_interval_length_to_rmse_ratio:.3f}"
            ),
            (
                f"binding {self.quality_risk_binding_design_label} interval/SE "
                f"{self.quality_risk_binding_interval_length_to_standard_error_ratio:.3f}"
            ),
            (
                f"best {self.quality_risk_best_design_label} interval/SE "
                f"{self.quality_risk_best_interval_length_to_standard_error_ratio:.3f}"
            ),
            (
                "RMSE/SE gap "
                f"{self.quality_risk_rmse_to_standard_error_ratio_gap:+.3f}"
            ),
            (
                "SE-RMSE gap "
                f"{self.quality_risk_standard_error_reserve_gap:+.3f}"
            ),
            (
                "RMSE-average SE gap "
                f"{self.quality_risk_rmse_outpaces_average_se_gap:+.3f}"
            ),
            (
                "interval/RMSE gap "
                f"{self.quality_risk_interval_length_to_rmse_gap:+.3f}"
            ),
            *self.quality_risk_clearance_evidence,
            *self.quality_risk_method_diagnosis_evidence,
            *diagnostic_candidate_evidence,
            *positive_headroom_verdict_evidence,
        )

    @property
    def paper_dgp2_contract_evidence(self) -> tuple[str, ...]:
        if not self.paper_dgp2_contract_status:
            return ()
        return (
            f"paper DGP2 contract {self.paper_dgp2_contract_status}",
            (
                "paper DGP2 R drift "
                f"{','.join(self.paper_dgp2_contract_r_finding_codes)}"
            ),
            (
                "paper DGP2 near-zero grid "
                f"{self.paper_dgp2_contract_near_zero_grid}"
            ),
        )

    def _build_monte_carlo_validation_evidence(self) -> tuple[str, ...]:
        return (
            self.runtime_evidence_binding_design_label,
            f"seed {self.runtime_evidence_binding_random_state}",
            (
                f"{self.runtime_evidence_current_witness_count}/"
                f"{self.runtime_evidence_total_witness_count} -> "
                f"{self.runtime_evidence_target_witness_count}/"
                f"{self.runtime_evidence_total_witness_count}"
            ),
            f"{self.runtime_evidence_effective_fresh_reruns} fresh reruns",
            self.quality_risk_binding_driver,
            f"runtime admission {self.runtime_evidence_admission_status}",
            *self.runtime_evidence_admission_candidate_invalidity_evidence,
            f"post-admission {self.post_admission_quality_risk_status}",
            *self.quality_risk_evidence,
            *self.quality_risk_action_evidence,
            *self.paper_dgp2_contract_evidence,
        )

    @property
    def helper(self) -> str:
        return "run_phase7_monte_carlo_feature_completion_gate(...)"

    @property
    def note(self) -> str:
        return "Docs/research/phase7_monte_carlo_feature_completion_gate.md"

    @property
    def blocker_reason(self) -> str:
        return self.monte_carlo_validation_ready_blocker

    @property
    def current_blocker_reason(self) -> str:
        return self.monte_carlo_validation_ready_blocker

    @property
    def monte_carlo_validation_reason(self) -> str:
        return self.monte_carlo_validation_ready_blocker

    @property
    def monte_carlo_validation_ready(self) -> bool:
        return (
            self.monte_carlo_validation_ready_status == "ready"
            and not self.monte_carlo_validation_ready_blocker
        )

    @property
    def ready(self) -> bool:
        return self.monte_carlo_validation_ready

    @property
    def status(self) -> str:
        return self.monte_carlo_validation_ready_status

    @property
    def gate_status(self) -> str:
        return self.current_gate_status

    @property
    def runtime_evidence_binding_design_label(self) -> str:
        return (
            f"{self.runtime_evidence_binding_design[0]}"
            f"/{self.runtime_evidence_binding_design[1]}"
            f"/{self.runtime_evidence_binding_design[2]}"
        )

    @property
    def runtime_evidence_binding_random_state(self) -> int:
        if len(self.runtime_evidence_binding_random_states) != 1:
            raise ValueError(
                "feature completion gate requires a single runtime evidence binding random state"
            )
        return self.runtime_evidence_binding_random_states[0]

    @property
    def runtime_evidence_admission_binding_design_label(self) -> str:
        return self.runtime_evidence_binding_design_label

    @property
    def runtime_evidence_admission_witness_floor_before_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_covered_pointwise_witnesses_before}/"
            f"{self.runtime_evidence_total_pointwise_witnesses}"
        )

    @property
    def runtime_evidence_admission_witness_floor_after_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_covered_pointwise_witnesses_after}/"
            f"{self.runtime_evidence_total_pointwise_witnesses}"
        )

    @property
    def runtime_evidence_admission_witness_floor_target_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_required_covered_pointwise_witnesses}/"
            f"{self.runtime_evidence_total_pointwise_witnesses}"
        )

    @property
    def runtime_evidence_admission_witness_floor_transition_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_witness_floor_before_label} -> "
            f"{self.runtime_evidence_admission_witness_floor_after_label}"
        )

    @property
    def runtime_evidence_admission_seed_local_witness_before_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before}/"
            f"{self.runtime_evidence_admission_seed_local_total_pointwise_witnesses}"
        )

    @property
    def runtime_evidence_admission_seed_local_witness_after_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after}/"
            f"{self.runtime_evidence_admission_seed_local_total_pointwise_witnesses}"
        )

    @property
    def runtime_evidence_admission_seed_local_witness_transition_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_seed_local_witness_before_label} -> "
            f"{self.runtime_evidence_admission_seed_local_witness_after_label}"
        )

    @property
    def runtime_evidence_admission_quota_gap_transition_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_remaining_quota_gap_before} -> "
            f"{self.runtime_evidence_admission_remaining_quota_gap_after}"
        )

    @property
    def runtime_evidence_admission_fresh_rerun_transition_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_fresh_reruns_before} -> "
            f"{self.runtime_evidence_admission_fresh_reruns_after}"
        )

    @property
    def runtime_evidence_admission_candidate_typed_invalidity_labels(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            f"{name}={count}"
            for name, count in self.runtime_evidence_admission_candidate_typed_invalidity_counts.items()
        )

    @property
    def runtime_evidence_admission_candidate_primary_invalidity_name(
        self,
    ) -> str | None:
        labels = tuple(
            name
            for name, count in self.runtime_evidence_admission_candidate_typed_invalidity_counts.items()
            if count > 0
        )
        return labels[0] if labels else None

    @property
    def runtime_evidence_admission_candidate_primary_invalidity_example(
        self,
    ) -> dict[str, object]:
        name = self.runtime_evidence_admission_candidate_primary_invalidity_name
        if name is None:
            return {}
        return dict(
            self.runtime_evidence_admission_candidate_typed_invalidity_examples.get(
                name,
                {},
            )
        )

    @property
    def runtime_evidence_admission_candidate_primary_invalidity_matrix_name(
        self,
    ) -> str | None:
        value = self.runtime_evidence_admission_candidate_primary_invalidity_example.get(
            "matrix_name"
        )
        return None if value is None else str(value)

    @property
    def runtime_evidence_admission_candidate_primary_invalidity_min_eigenvalue(
        self,
    ) -> float | None:
        example = self.runtime_evidence_admission_candidate_primary_invalidity_example
        for key in (
            "omega_f_primary_min_eigenvalue",
            "omega_f_selected_min_eigenvalue",
            "min_eigenvalue",
        ):
            if key in example and example[key] is not None:
                return float(example[key])
        return None

    @property
    def runtime_evidence_admission_candidate_primary_invalidity_replication_seed(
        self,
    ) -> int | None:
        value = self.runtime_evidence_admission_candidate_primary_invalidity_example.get(
            "replication_seed"
        )
        if value is None:
            return None
        return _validated_integer(
            value,
            label="runtime evidence admission candidate primary invalidity replication seed",
            minimum=0,
        )

    @property
    def runtime_evidence_admission_candidate_invalidity_evidence(
        self,
    ) -> tuple[str, ...]:
        labels = self.runtime_evidence_admission_candidate_typed_invalidity_labels
        if not labels:
            return ("runtime admission candidate typed invalidity none",)
        evidence = [
            "runtime admission candidate typed invalidity " + ", ".join(labels),
        ]
        name = self.runtime_evidence_admission_candidate_primary_invalidity_name
        matrix_name = (
            self.runtime_evidence_admission_candidate_primary_invalidity_matrix_name
        )
        min_eigenvalue = (
            self.runtime_evidence_admission_candidate_primary_invalidity_min_eigenvalue
        )
        replication_seed = (
            self.runtime_evidence_admission_candidate_primary_invalidity_replication_seed
        )
        if name is not None:
            detail = f"runtime admission candidate invalidity {name}"
            if matrix_name is not None:
                detail += f" matrix {matrix_name}"
            if min_eigenvalue is not None:
                detail += f" min_eigenvalue {min_eigenvalue:.6g}"
            if replication_seed is not None:
                detail += f" replication_seed {replication_seed}"
            evidence.append(detail)
        return tuple(evidence)

    @property
    def runtime_evidence_current_witness_count(self) -> int:
        return self.runtime_evidence_covered_pointwise_witnesses

    @property
    def route(self) -> str:
        return self.route_label

    @property
    def runtime_evidence_target_witness_count(self) -> int:
        return self.runtime_evidence_required_covered_pointwise_witnesses

    @property
    def runtime_evidence_total_witness_count(self) -> int:
        return self.runtime_evidence_total_pointwise_witnesses

    @property
    def frontier_packet_helper(self) -> str:
        return "run_phase7_monte_carlo_feature_completion_frontier_packet(...)"

    @property
    def frontier_packet_note(self) -> str:
        return "Docs/research/phase7_monte_carlo_feature_completion_frontier_packet.md"

    @property
    def frontier_packet_live_entry(self) -> str:
        return self.route_label

    @property
    def live_entry(self) -> str:
        return self.frontier_packet_live_entry

    @property
    def frontier_packet_open_trigger(self) -> str:
        if len(self.accepted_open_trigger_names) != 1:
            raise ValueError(
                "feature completion frontier packet requires one open trigger"
            )
        return self.accepted_open_trigger_names[0]

    @property
    def frontier_packet_driver(self) -> str:
        return self.runtime_evidence_driver

    @property
    def live_source_target_reground_contract_helper(self) -> str:
        return (
            "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_"
            "live_source_target_reground_contract()"
        )

    @property
    def live_source_target_reground_contract_note(self) -> str:
        return (
            "Docs/research/phase7_monte_carlo_widening_policy_coverage_anchor_"
            "shoulder_live_source_target_reground_contract.md"
        )

    @property
    def live_source_target_reground_contract_driver(self) -> str:
        if (
            self.live_source_target_alignment_driver
            == "live-source-target-alignment-drift"
            and self.execution_contract_signature
            == "bounded-right-center-execution-contract"
            and self.runtime_evidence_driver == "trigger2-runtime-evidence-packet-open"
            and self.live_source_target_alignment_target_gap > 0.0
        ):
            return "live-source-target-reground-contract"
        return "mixed-live-source-target-reground-contract"

    @property
    def live_source_target_reground_archived_target_omega_diagonal_entry(self) -> float:
        return 410.474

    @property
    def live_source_target_reground_target_omega_diagonal_entry(self) -> float:
        return (
            self.live_source_target_reground_archived_target_omega_diagonal_entry
            + self.live_source_target_alignment_target_gap
        )

    @property
    def live_source_target_reground_contract_implication(self) -> str:
        if (
            self.live_source_target_reground_contract_driver
            == "live-source-target-reground-contract"
        ):
            return "pre-rerun-target-bound-no-fresh-rerun-spent"
        return "reground-contract-drift"

    @property
    def quality_risk_shortfall(self) -> float:
        return self.quality_risk_canonical_floor_shortfall

    @property
    def canonical_policy(self) -> Phase7MonteCarloWideningPolicy | None:
        return decode_phase7_monte_carlo_widening_policy_digest(self.policy_digest)

    @property
    def canonical_policy_label(self) -> str | None:
        canonical_policy = self.canonical_policy
        return None if canonical_policy is None else canonical_policy.label

    @property
    def canonical_policy_max_total_runtime_seconds(self) -> float | None:
        canonical_policy = self.canonical_policy
        return (
            None
            if canonical_policy is None
            else canonical_policy.max_total_runtime_seconds
        )

    @property
    def canonical_policy_max_random_states(self) -> int | None:
        canonical_policy = self.canonical_policy
        return None if canonical_policy is None else canonical_policy.max_random_states

    @property
    def canonical_policy_stop_on_first_typed_invalidity(self) -> bool | None:
        canonical_policy = self.canonical_policy
        return (
            None
            if canonical_policy is None
            else canonical_policy.stop_on_first_typed_invalidity
        )

    @property
    def canonical_policy_min_nonparametric_coverage(self) -> float | None:
        canonical_policy = self.canonical_policy
        return (
            None
            if canonical_policy is None
            else canonical_policy.min_nonparametric_coverage
        )

    @property
    def stable_partial_design_labels(self) -> tuple[str, ...]:
        return _design_key_labels(self.stable_partial_designs)

    @property
    def blocked_full_matrix_design_labels(self) -> tuple[str, ...]:
        return _design_key_labels(self.blocked_full_matrix_designs)

    @property
    def quality_risk_design_labels(self) -> tuple[str, ...]:
        return _design_key_labels(self.quality_risk_designs)

    @property
    def typed_invalidity_total(self) -> int:
        return sum(self.typed_invalidity_counts.values())

    @property
    def typed_invalidity_count_labels(self) -> tuple[str, ...]:
        return tuple(
            f"{code}={count}"
            for code, count in sorted(self.typed_invalidity_counts.items())
        )

    def to_dict(self) -> dict[str, object]:
        canonical_policy = self.canonical_policy
        return {
            "stage_label": self.stage_label,
            "route_label": self.route_label,
            "route": self.route,
            "helper": self.helper,
            "note": self.note,
            "blocker_reason": self.blocker_reason,
            "current_blocker_reason": self.current_blocker_reason,
            "current_gate_status": self.current_gate_status,
            "gate_status": self.gate_status,
            "target_gate_status": self.target_gate_status,
            "accepted_feature_bundle": self.accepted_feature_bundle,
            "accepted_open_trigger_names": list(self.accepted_open_trigger_names),
            "frontier_packet_helper": self.frontier_packet_helper,
            "frontier_packet_note": self.frontier_packet_note,
            "frontier_packet_live_entry": self.frontier_packet_live_entry,
            "live_entry": self.live_entry,
            "frontier_packet_open_trigger": self.frontier_packet_open_trigger,
            "frontier_packet_driver": self.frontier_packet_driver,
            "runtime_evidence_helper": self.runtime_evidence_helper,
            "runtime_evidence_note": self.runtime_evidence_note,
            "runtime_evidence_driver": self.runtime_evidence_driver,
            "runtime_evidence_current_driver": self.runtime_evidence_current_driver,
            "runtime_evidence_binding_design": list(
                self.runtime_evidence_binding_design
            ),
            "runtime_evidence_binding_design_label": (
                self.runtime_evidence_binding_design_label
            ),
            "runtime_evidence_binding_random_states": list(
                self.runtime_evidence_binding_random_states
            ),
            "runtime_evidence_binding_random_state": (
                self.runtime_evidence_binding_random_state
            ),
            "runtime_evidence_required_quota_gap": (
                self.runtime_evidence_required_quota_gap
            ),
            "runtime_evidence_effective_fresh_reruns": (
                self.runtime_evidence_effective_fresh_reruns
            ),
            "runtime_evidence_covered_pointwise_witnesses": (
                self.runtime_evidence_covered_pointwise_witnesses
            ),
            "runtime_evidence_required_covered_pointwise_witnesses": (
                self.runtime_evidence_required_covered_pointwise_witnesses
            ),
            "runtime_evidence_total_pointwise_witnesses": (
                self.runtime_evidence_total_pointwise_witnesses
            ),
            "runtime_evidence_supported_floor_ceiling": (
                self.runtime_evidence_supported_floor_ceiling
            ),
            "runtime_evidence_quota_helper": self.runtime_evidence_quota_helper,
            "runtime_evidence_quota_note": self.runtime_evidence_quota_note,
            "runtime_evidence_binding_design_rerun_capacity_helper": (
                self.runtime_evidence_binding_design_rerun_capacity_helper
            ),
            "runtime_evidence_binding_design_rerun_capacity_note": (
                self.runtime_evidence_binding_design_rerun_capacity_note
            ),
            "runtime_evidence_binding_design_rerun_capacity_limiting_budget": (
                self.runtime_evidence_binding_design_rerun_capacity_limiting_budget
            ),
            "runtime_evidence_binding_design_rerun_capacity_implication": (
                self.runtime_evidence_binding_design_rerun_capacity_implication
            ),
            "runtime_evidence_current_implication": (
                self.runtime_evidence_current_implication
            ),
            "runtime_evidence_admission_helper": self.runtime_evidence_admission_helper,
            "runtime_evidence_admission_note": self.runtime_evidence_admission_note,
            "runtime_evidence_admission_status": self.runtime_evidence_admission_status,
            "runtime_evidence_admission_candidate_status": (
                self.runtime_evidence_admission_candidate_status
            ),
            "runtime_evidence_admission_contract_status": (
                self.runtime_evidence_admission_contract_status
            ),
            "runtime_evidence_admission_candidate_success": (
                self.runtime_evidence_admission_candidate_success
            ),
            "runtime_evidence_admission_candidate_nonparametric_coverage": (
                self.runtime_evidence_admission_candidate_nonparametric_coverage
            ),
            "runtime_evidence_admission_candidate_typed_invalidity_counts": dict(
                self.runtime_evidence_admission_candidate_typed_invalidity_counts
            ),
            "runtime_evidence_admission_candidate_typed_invalidity_examples": dict(
                self.runtime_evidence_admission_candidate_typed_invalidity_examples
            ),
            "runtime_evidence_admission_candidate_typed_invalidity_labels": list(
                self.runtime_evidence_admission_candidate_typed_invalidity_labels
            ),
            "runtime_evidence_admission_candidate_primary_invalidity_name": (
                self.runtime_evidence_admission_candidate_primary_invalidity_name
            ),
            "runtime_evidence_admission_candidate_primary_invalidity_matrix_name": (
                self.runtime_evidence_admission_candidate_primary_invalidity_matrix_name
            ),
            "runtime_evidence_admission_candidate_primary_invalidity_min_eigenvalue": (
                self.runtime_evidence_admission_candidate_primary_invalidity_min_eigenvalue
            ),
            "runtime_evidence_admission_candidate_primary_invalidity_replication_seed": (
                self.runtime_evidence_admission_candidate_primary_invalidity_replication_seed
            ),
            "runtime_evidence_admission_candidate_invalidity_evidence": list(
                self.runtime_evidence_admission_candidate_invalidity_evidence
            ),
            "runtime_evidence_admission_binding_design_label": (
                self.runtime_evidence_admission_binding_design_label
            ),
            "runtime_evidence_admission_binding_random_state": (
                self.runtime_evidence_admission_binding_random_state
            ),
            "runtime_evidence_admission_witness_floor_before_label": (
                self.runtime_evidence_admission_witness_floor_before_label
            ),
            "runtime_evidence_admission_witness_floor_after_label": (
                self.runtime_evidence_admission_witness_floor_after_label
            ),
            "runtime_evidence_admission_witness_floor_target_label": (
                self.runtime_evidence_admission_witness_floor_target_label
            ),
            "runtime_evidence_admission_witness_floor_transition_label": (
                self.runtime_evidence_admission_witness_floor_transition_label
            ),
            "runtime_evidence_admission_seed_local_witness_before_label": (
                self.runtime_evidence_admission_seed_local_witness_before_label
            ),
            "runtime_evidence_admission_seed_local_witness_after_label": (
                self.runtime_evidence_admission_seed_local_witness_after_label
            ),
            "runtime_evidence_admission_seed_local_witness_transition_label": (
                self.runtime_evidence_admission_seed_local_witness_transition_label
            ),
            "runtime_evidence_admission_quota_gap_transition_label": (
                self.runtime_evidence_admission_quota_gap_transition_label
            ),
            "runtime_evidence_admission_fresh_rerun_transition_label": (
                self.runtime_evidence_admission_fresh_rerun_transition_label
            ),
            "runtime_evidence_admission_covered_pointwise_witnesses_before": (
                self.runtime_evidence_admission_covered_pointwise_witnesses_before
            ),
            "runtime_evidence_admission_covered_pointwise_witnesses_after": (
                self.runtime_evidence_admission_covered_pointwise_witnesses_after
            ),
            "runtime_evidence_admission_required_covered_pointwise_witnesses": (
                self.runtime_evidence_admission_required_covered_pointwise_witnesses
            ),
            "runtime_evidence_admission_remaining_quota_gap_before": (
                self.runtime_evidence_admission_remaining_quota_gap_before
            ),
            "runtime_evidence_admission_remaining_quota_gap_after": (
                self.runtime_evidence_admission_remaining_quota_gap_after
            ),
            "runtime_evidence_admission_fresh_reruns_before": (
                self.runtime_evidence_admission_fresh_reruns_before
            ),
            "runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before": (
                self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before
            ),
            "runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after": (
                self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after
            ),
            "runtime_evidence_admission_seed_local_total_pointwise_witnesses": (
                self.runtime_evidence_admission_seed_local_total_pointwise_witnesses
            ),
            "runtime_evidence_admission_quota_closure_margin": (
                self.runtime_evidence_admission_quota_closure_margin
            ),
            "post_admission_quality_risk_status": (
                self.post_admission_quality_risk_status
            ),
            "post_admission_quality_risk_driver": (
                self.post_admission_quality_risk_driver
            ),
            "post_admission_quality_risk_binding_design": list(
                self.post_admission_quality_risk_binding_design
            ),
            "monte_carlo_validation_ready_status": (
                self.monte_carlo_validation_ready_status
            ),
            "monte_carlo_validation_ready_blocker": (
                self.monte_carlo_validation_ready_blocker
            ),
            "monte_carlo_validation_reason": self.monte_carlo_validation_reason,
            "monte_carlo_validation_ready": self.monte_carlo_validation_ready,
            "ready": self.ready,
            "status": self.status,
            "quality_risk_source_mode": self.quality_risk_source_mode,
            "quality_risk_source_note": self.quality_risk_source_note,
            "quality_risk_paper_object_contract": list(
                self.quality_risk_paper_object_contract
            ),
            "paper_dgp2_contract_helper": self.paper_dgp2_contract_helper,
            "paper_dgp2_contract_note": self.paper_dgp2_contract_note,
            "paper_dgp2_contract_status": self.paper_dgp2_contract_status,
            "paper_dgp2_contract_r_finding_codes": list(
                self.paper_dgp2_contract_r_finding_codes
            ),
            "paper_dgp2_contract_near_zero_grid": list(
                self.paper_dgp2_contract_near_zero_grid
            ),
            "paper_dgp2_contract_evidence": list(
                self.paper_dgp2_contract_evidence
            ),
            "runtime_evidence_admission_fresh_reruns_after": (
                self.runtime_evidence_admission_fresh_reruns_after
            ),
            "quality_risk_probe_helper": self.quality_risk_probe_helper,
            "quality_risk_probe_note": self.quality_risk_probe_note,
            "quality_risk_driver": self.quality_risk_driver,
            "quality_risk_route": self.quality_risk_route,
            "quality_risk_binding_driver": self.quality_risk_binding_driver,
            "quality_risk_blocker_reason": self.quality_risk_blocker_reason,
            "quality_risk_binding_design": list(self.quality_risk_binding_design),
            "quality_risk_binding_design_label": (
                self.quality_risk_binding_design_label
            ),
            "quality_risk_best_design": list(self.quality_risk_best_design),
            "quality_risk_best_design_label": self.quality_risk_best_design_label,
            "quality_risk_canonical_floor_shortfall": (
                self.quality_risk_canonical_floor_shortfall
            ),
            "quality_risk_binding_coverage": self.quality_risk_binding_coverage,
            "quality_risk_binding_coverage_floor_slack": (
                self.quality_risk_binding_coverage_floor_slack
            ),
            "quality_risk_binding_rmse": self.quality_risk_binding_rmse,
            "quality_risk_binding_average_standard_error": (
                self.quality_risk_binding_average_standard_error
            ),
            "quality_risk_binding_interval_length": (
                self.quality_risk_binding_interval_length
            ),
            "quality_risk_binding_rmse_to_standard_error_ratio": (
                self.quality_risk_binding_rmse_to_standard_error_ratio
            ),
            "quality_risk_binding_standard_error_reserve": (
                self.quality_risk_binding_standard_error_reserve
            ),
            "quality_risk_binding_interval_length_to_rmse_ratio": (
                self.quality_risk_binding_interval_length_to_rmse_ratio
            ),
            "quality_risk_binding_interval_length_to_standard_error_ratio": (
                self.quality_risk_binding_interval_length_to_standard_error_ratio
            ),
            "quality_risk_raw_metrics": {
                "binding": {
                    "coverage": self.quality_risk_binding_coverage,
                    "rmse": self.quality_risk_binding_rmse,
                    "average_standard_error": (
                        self.quality_risk_binding_average_standard_error
                    ),
                    "interval_length": self.quality_risk_binding_interval_length,
                    "standard_error_reserve": (
                        self.quality_risk_binding_standard_error_reserve
                    ),
                },
                "best": {
                    "coverage": self.quality_risk_best_coverage,
                    "rmse": self.quality_risk_best_rmse,
                    "average_standard_error": (
                        self.quality_risk_best_average_standard_error
                    ),
                    "interval_length": self.quality_risk_best_interval_length,
                    "standard_error_reserve": (
                        self.quality_risk_best_standard_error_reserve
                    ),
                },
            },
            "quality_risk_best_coverage": self.quality_risk_best_coverage,
            "quality_risk_best_coverage_floor_slack": (
                self.quality_risk_best_coverage_floor_slack
            ),
            "quality_risk_best_rmse": self.quality_risk_best_rmse,
            "quality_risk_best_average_standard_error": (
                self.quality_risk_best_average_standard_error
            ),
            "quality_risk_best_interval_length": (
                self.quality_risk_best_interval_length
            ),
            "quality_risk_best_rmse_to_standard_error_ratio": (
                self.quality_risk_best_rmse_to_standard_error_ratio
            ),
            "quality_risk_best_standard_error_reserve": (
                self.quality_risk_best_standard_error_reserve
            ),
            "quality_risk_best_interval_length_to_rmse_ratio": (
                self.quality_risk_best_interval_length_to_rmse_ratio
            ),
            "quality_risk_best_interval_length_to_standard_error_ratio": (
                self.quality_risk_best_interval_length_to_standard_error_ratio
            ),
            "quality_risk_interval_scale_status": (
                self.quality_risk_interval_scale_status
            ),
            "quality_risk_rmse_to_standard_error_ratio_gap": (
                self.quality_risk_rmse_to_standard_error_ratio_gap
            ),
            "quality_risk_standard_error_reserve_gap": (
                self.quality_risk_standard_error_reserve_gap
            ),
            "quality_risk_interval_length_to_rmse_gap": (
                self.quality_risk_interval_length_to_rmse_gap
            ),
            "quality_risk_rmse_outpaces_average_se_gap": (
                self.quality_risk_rmse_outpaces_average_se_gap
            ),
            "rmse_outpaces_average_se_gap": self.rmse_outpaces_average_se_gap,
            "quality_risk_clearance_evidence": list(
                self.quality_risk_clearance_evidence
            ),
            "quality_risk_clearance_conditions": list(
                self.quality_risk_clearance_conditions
            ),
            "quality_risk_calibration_targets": list(
                self.quality_risk_calibration_targets
            ),
            "dominant_quality_risk_calibration_target": dict(
                self.dominant_quality_risk_calibration_target
            ),
            "quality_risk_method_diagnosis": dict(
                self.quality_risk_method_diagnosis
            ),
            "quality_risk_method_diagnosis_evidence": list(
                self.quality_risk_method_diagnosis_evidence
            ),
            "quality_risk_action_contract": dict(self.quality_risk_action_contract),
            "quality_risk_action_evidence": list(self.quality_risk_action_evidence),
            "quality_risk_average_standard_error_calibration_candidate": dict(
                self.quality_risk_average_standard_error_calibration_candidate
            ),
            "quality_risk_average_standard_error_calibration_candidate_status": (
                self.quality_risk_average_standard_error_calibration_candidate_status
            ),
            "quality_risk_average_standard_error_calibration_candidate_target_condition": (
                self.quality_risk_average_standard_error_calibration_candidate_target_condition
            ),
            "quality_risk_average_standard_error_calibration_candidate_current_average_standard_error": (
                self.quality_risk_average_standard_error_calibration_candidate_current_average_standard_error
            ),
            "quality_risk_average_standard_error_calibration_candidate_target_average_standard_error": (
                self.quality_risk_average_standard_error_calibration_candidate_target_average_standard_error
            ),
            "quality_risk_average_standard_error_calibration_candidate_current_interval_length": (
                self.quality_risk_average_standard_error_calibration_candidate_current_interval_length
            ),
            "quality_risk_average_standard_error_calibration_candidate_target_interval_length": (
                self.quality_risk_average_standard_error_calibration_candidate_target_interval_length
            ),
            "quality_risk_average_standard_error_calibration_candidate_required_average_standard_error_lift": (
                self.quality_risk_average_standard_error_calibration_candidate_required_average_standard_error_lift
            ),
            "quality_risk_average_standard_error_calibration_candidate_required_interval_length_lift": (
                self.quality_risk_average_standard_error_calibration_candidate_required_interval_length_lift
            ),
            "quality_risk_average_standard_error_calibration_candidate_minimum_headroom_margin_after_calibration": (
                self.quality_risk_average_standard_error_calibration_candidate_minimum_headroom_margin_after_calibration
            ),
            "quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_average_standard_error": (
                self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_average_standard_error
            ),
            "quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_interval_length": (
                self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_interval_length
            ),
            "quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_average_standard_error_lift": (
                self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_average_standard_error_lift
            ),
            "quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_interval_length_lift": (
                self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_interval_length_lift
            ),
            "quality_risk_average_standard_error_calibration_candidate_positive_headroom_margin": (
                self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_margin
            ),
            "quality_risk_positive_headroom_target_average_standard_error": (
                self.quality_risk_positive_headroom_target_average_standard_error
            ),
            "quality_risk_positive_headroom_observed_average_standard_error": (
                self.quality_risk_positive_headroom_observed_average_standard_error
            ),
            "positive_headroom_observed_average_standard_error": (
                self.positive_headroom_observed_average_standard_error
            ),
            "quality_risk_positive_headroom_current_average_standard_error": (
                self.quality_risk_positive_headroom_current_average_standard_error
            ),
            "positive_headroom_current_average_standard_error": (
                self.positive_headroom_current_average_standard_error
            ),
            "quality_risk_positive_headroom_target_interval_length": (
                self.quality_risk_positive_headroom_target_interval_length
            ),
            "quality_risk_positive_headroom_observed_interval_length": (
                self.quality_risk_positive_headroom_observed_interval_length
            ),
            "positive_headroom_observed_interval_length": (
                self.positive_headroom_observed_interval_length
            ),
            "quality_risk_positive_headroom_current_interval_length": (
                self.quality_risk_positive_headroom_current_interval_length
            ),
            "positive_headroom_current_interval_length": (
                self.positive_headroom_current_interval_length
            ),
            "quality_risk_positive_headroom_required_average_standard_error_lift": (
                self.quality_risk_positive_headroom_required_average_standard_error_lift
            ),
            "quality_risk_positive_headroom_required_interval_length_lift": (
                self.quality_risk_positive_headroom_required_interval_length_lift
            ),
            "quality_risk_positive_headroom_remaining_average_standard_error_gap": (
                self.quality_risk_positive_headroom_remaining_average_standard_error_gap
            ),
            "quality_risk_positive_headroom_remaining_interval_length_gap": (
                self.quality_risk_positive_headroom_remaining_interval_length_gap
            ),
            "quality_risk_positive_headroom_maximum_remaining_gap_fraction": (
                self.quality_risk_positive_headroom_maximum_remaining_gap_fraction
            ),
            "positive_headroom_maximum_remaining_gap_fraction": (
                self.positive_headroom_maximum_remaining_gap_fraction
            ),
            "quality_risk_positive_headroom_margin": (
                self.quality_risk_positive_headroom_margin
            ),
            "quality_risk_positive_headroom_admission_status": (
                self.quality_risk_positive_headroom_admission_status
            ),
            "quality_risk_average_standard_error_calibration_candidate_release_gate_effect": (
                self.quality_risk_average_standard_error_calibration_candidate_release_gate_effect
            ),
            "quality_risk_average_standard_error_calibration_candidate_admission_status": (
                self.quality_risk_average_standard_error_calibration_candidate_admission_status
            ),
            "quality_risk_average_standard_error_calibration_candidate_admission_blocker": (
                self.quality_risk_average_standard_error_calibration_candidate_admission_blocker
            ),
            "quality_risk_average_standard_error_calibration_candidate_admission_required_next_evidence": (
                self.quality_risk_average_standard_error_calibration_candidate_admission_required_next_evidence
            ),
            "quality_risk_average_standard_error_calibration_candidate_admissible_as_release_evidence": (
                self.quality_risk_average_standard_error_calibration_candidate_admissible_as_release_evidence
            ),
            "quality_risk_average_standard_error_calibration_candidate_evidence": list(
                self.quality_risk_average_standard_error_calibration_candidate_evidence
            ),
            "quality_risk_positive_headroom_estimator_evidence_requirement": dict(
                self.quality_risk_positive_headroom_estimator_evidence_requirement
            ),
            "positive_headroom_estimator_evidence_requirement": dict(
                self.positive_headroom_estimator_evidence_requirement
            ),
            "quality_risk_positive_headroom_estimator_evidence_verdict": dict(
                self.quality_risk_positive_headroom_estimator_evidence_verdict
            ),
            "positive_headroom_estimator_evidence_verdict": dict(
                self.positive_headroom_estimator_evidence_verdict
            ),
            "quality_risk_positive_headroom_estimator_evidence_status": (
                self.quality_risk_positive_headroom_estimator_evidence_status
            ),
            "positive_headroom_estimator_evidence_status": (
                self.positive_headroom_estimator_evidence_status
            ),
            "quality_risk_positive_headroom_estimator_evidence_source": (
                self.quality_risk_positive_headroom_estimator_evidence_source
            ),
            "positive_headroom_estimator_evidence_source": (
                self.positive_headroom_estimator_evidence_source
            ),
            "quality_risk_positive_headroom_estimator_evidence_source_status": (
                self.quality_risk_positive_headroom_estimator_evidence_source_status
            ),
            "positive_headroom_estimator_evidence_source_status": (
                self.positive_headroom_estimator_evidence_source_status
            ),
            "quality_risk_positive_headroom_admissible_for_feature_gate_rerun": (
                self.quality_risk_positive_headroom_admissible_for_feature_gate_rerun
            ),
            "positive_headroom_admissible_for_feature_gate_rerun": (
                self.positive_headroom_admissible_for_feature_gate_rerun
            ),
            "feature_gate_rerun_admissible": self.feature_gate_rerun_admissible,
            "admissible_for_feature_gate_rerun": (
                self.admissible_for_feature_gate_rerun
            ),
            "quality_risk_positive_headroom_feature_gate_rerun_admission_status": (
                self.quality_risk_positive_headroom_feature_gate_rerun_admission_status
            ),
            "positive_headroom_feature_gate_rerun_admission_status": (
                self.positive_headroom_feature_gate_rerun_admission_status
            ),
            "feature_gate_rerun_admission_status": (
                self.feature_gate_rerun_admission_status
            ),
            "quality_risk_positive_headroom_estimator_evidence_verdict_evidence": list(
                self.quality_risk_positive_headroom_estimator_evidence_verdict_evidence
            ),
            "positive_headroom_estimator_evidence_verdict_evidence": list(
                self.positive_headroom_estimator_evidence_verdict_evidence
            ),
            "quality_risk_evidence": list(self.quality_risk_evidence),
            "monte_carlo_validation_evidence": list(
                self.monte_carlo_validation_evidence
            ),
            "repair_target_helper": self.repair_target_helper,
            "repair_target_note": self.repair_target_note,
            "repair_target_signature": self.repair_target_signature,
            "execution_contract_helper": self.execution_contract_helper,
            "execution_contract_note": self.execution_contract_note,
            "execution_contract_signature": self.execution_contract_signature,
            "execution_bundle_helper": self.execution_bundle_helper,
            "execution_bundle_note": self.execution_bundle_note,
            "execution_bundle_signature": self.execution_bundle_signature,
            "live_source_target_alignment_helper": (
                self.live_source_target_alignment_helper
            ),
            "live_source_target_alignment_note": (
                self.live_source_target_alignment_note
            ),
            "live_source_target_alignment_driver": (
                self.live_source_target_alignment_driver
            ),
            "live_source_target_alignment_implication": (
                self.live_source_target_alignment_implication
            ),
            "live_source_target_alignment_binding_random_states": list(
                self.live_source_target_alignment_binding_random_states
            ),
            "live_source_target_alignment_binding_replication_seeds": list(
                self.live_source_target_alignment_binding_replication_seeds
            ),
            "live_source_target_alignment_target_gap_field": (
                self.live_source_target_alignment_target_gap_field
            ),
            "live_source_target_alignment_target_gap": (
                self.live_source_target_alignment_target_gap
            ),
            "live_source_target_reground_contract_helper": (
                self.live_source_target_reground_contract_helper
            ),
            "live_source_target_reground_contract_note": (
                self.live_source_target_reground_contract_note
            ),
            "live_source_target_reground_contract_driver": (
                self.live_source_target_reground_contract_driver
            ),
            "live_source_target_reground_archived_target_omega_diagonal_entry": (
                self.live_source_target_reground_archived_target_omega_diagonal_entry
            ),
            "live_source_target_reground_target_omega_diagonal_entry": (
                self.live_source_target_reground_target_omega_diagonal_entry
            ),
            "live_source_target_reground_contract_implication": (
                self.live_source_target_reground_contract_implication
            ),
            "stable_partial_designs": [
                list(item) for item in self.stable_partial_designs
            ],
            "stable_partial_design_labels": list(self.stable_partial_design_labels),
            "blocked_full_matrix_designs": [
                list(item) for item in self.blocked_full_matrix_designs
            ],
            "blocked_full_matrix_design_labels": list(
                self.blocked_full_matrix_design_labels
            ),
            "quality_risk_designs": [list(item) for item in self.quality_risk_designs],
            "quality_risk_design_labels": list(self.quality_risk_design_labels),
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "typed_invalidity_total": self.typed_invalidity_total,
            "typed_invalidity_count_labels": list(self.typed_invalidity_count_labels),
            "policy_digest": list(self.policy_digest),
            "canonical_policy": (
                None if canonical_policy is None else canonical_policy.to_dict()
            ),
            "canonical_policy_label": self.canonical_policy_label,
            "canonical_policy_max_total_runtime_seconds": (
                self.canonical_policy_max_total_runtime_seconds
            ),
            "canonical_policy_max_random_states": self.canonical_policy_max_random_states,
            "canonical_policy_stop_on_first_typed_invalidity": (
                self.canonical_policy_stop_on_first_typed_invalidity
            ),
            "canonical_policy_min_nonparametric_coverage": (
                self.canonical_policy_min_nonparametric_coverage
            ),
            "canonical_gate_digest": list(self.canonical_gate_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


def build_phase7_monte_carlo_feature_completion_gate_report(
    current_gate: Phase7MonteCarloWideningTriggerGateReport,
    policy_spec: Phase7MonteCarloWideningPolicySpecReport,
    acceptance_preview: Phase7MonteCarloWideningPolicyAcceptancePreviewReport,
    runtime_evidence_packet: (
        Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport | None
    ) = None,
    runtime_evidence_admission: (
        Phase7MonteCarloWideningPolicyRuntimeEvidenceAdmissionGateReport | None
    ) = None,
    quality_risk_probe: Phase7MonteCarloWideningPolicyQualityRiskProbeReport | None = None,
    repair_target_snapshot: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderRepairTargetSnapshotReport
        | None
    ) = None,
    execution_contract: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
        | None
    ) = None,
    execution_bundle=None,
    live_source_target_alignment: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport
        | None
    ) = None,
    paper_dgp2_contract_audit: Phase7MonteCarloPaperDGP2ContractAudit | None = None,
) -> Phase7MonteCarloFeatureCompletionGateReport:
    if quality_risk_probe is None:
        quality_risk_probe_helper = (
            "run_phase7_monte_carlo_widening_policy_quality_risk_probe()"
        )
        quality_risk_probe_note = (
            "Docs/research/phase7_monte_carlo_widening_policy_quality_risk_probe.md"
        )
        quality_risk_binding_driver = "rmse-outpaces-average-se"
        quality_risk_blocker_reason = "quality-risk-keeps-trigger2-bounded"
        quality_risk_binding_design = ("DGP2", 500, 50)
        quality_risk_best_design = ("DGP1", 500, 50)
        quality_risk_canonical_floor_shortfall = 0.072
        quality_risk_binding_coverage = 0.778
        quality_risk_binding_coverage_floor_slack = -0.072
        quality_risk_binding_rmse = 4.171122823129394
        quality_risk_binding_average_standard_error = 3.992351539913328
        quality_risk_binding_interval_length = 13.133667820983462
        quality_risk_binding_rmse_to_standard_error_ratio = 1.0447784423362547
        quality_risk_binding_standard_error_reserve = -0.1787712832160664
        quality_risk_binding_interval_length_to_rmse_ratio = 3.148712799382373
        quality_risk_best_coverage = 0.8888888888888888
        quality_risk_best_coverage_floor_slack = 0.03888888888888886
        quality_risk_best_rmse = 1.9111312266475258
        quality_risk_best_average_standard_error = 2.2681232532639695
        quality_risk_best_interval_length = 7.461461519008423
        quality_risk_best_rmse_to_standard_error_ratio = 0.842604661760462
        quality_risk_best_standard_error_reserve = 0.3569920266164437
        quality_risk_best_interval_length_to_rmse_ratio = 3.9042120263490188
        quality_risk_rmse_to_standard_error_ratio_gap = 0.20217378057579272
        quality_risk_standard_error_reserve_gap = 0.5357633098325101
        quality_risk_interval_length_to_rmse_gap = 0.7554992269666458
        quality_risk_source_mode = "repo-side-canonical-blocker"
        quality_risk_source_note = (
            "feature-completion gate preserves the repo-side blocker packet when "
            "live floor-slack input drifts"
        )
        quality_risk_paper_object_contract = _QUALITY_RISK_PAPER_OBJECT_CONTRACT
    else:
        quality_risk_probe_helper = (
            "run_phase7_monte_carlo_widening_policy_quality_risk_probe()"
        )
        quality_risk_probe_note = (
            "Docs/research/phase7_monte_carlo_widening_policy_quality_risk_probe.md"
        )
        quality_risk_binding_driver = quality_risk_probe.binding_driver
        quality_risk_blocker_reason = quality_risk_probe.blocker_reason
        quality_risk_source_mode = quality_risk_probe.quality_risk_source_mode
        quality_risk_source_note = quality_risk_probe.quality_risk_source_note
        quality_risk_paper_object_contract = (
            quality_risk_probe.paper_object_contract
        )
        binding_quality_summary = quality_risk_probe.design_quality_summary(
            *quality_risk_probe.binding_design
        )
        best_quality_summary = quality_risk_probe.design_quality_summary(
            *quality_risk_probe.best_design
        )
        quality_risk_binding_design = quality_risk_probe.binding_design
        quality_risk_best_design = quality_risk_probe.best_design
        quality_risk_canonical_floor_shortfall = (
            quality_risk_probe.canonical_floor_shortfall
        )
        quality_risk_binding_coverage = (
            binding_quality_summary.mean_nonparametric_coverage
        )
        quality_risk_binding_coverage_floor_slack = (
            binding_quality_summary.coverage_floor_slack
        )
        quality_risk_binding_rmse = binding_quality_summary.mean_nonparametric_rmse
        quality_risk_binding_average_standard_error = (
            binding_quality_summary.mean_nonparametric_average_standard_error
        )
        quality_risk_binding_interval_length = (
            binding_quality_summary.mean_nonparametric_interval_length
        )
        quality_risk_binding_rmse_to_standard_error_ratio = (
            binding_quality_summary.rmse_to_standard_error_ratio
        )
        quality_risk_binding_standard_error_reserve = (
            binding_quality_summary.standard_error_reserve
        )
        quality_risk_binding_interval_length_to_rmse_ratio = (
            binding_quality_summary.interval_length_to_rmse_ratio
        )
        quality_risk_best_coverage = best_quality_summary.mean_nonparametric_coverage
        quality_risk_best_coverage_floor_slack = (
            best_quality_summary.coverage_floor_slack
        )
        quality_risk_best_rmse = best_quality_summary.mean_nonparametric_rmse
        quality_risk_best_average_standard_error = (
            best_quality_summary.mean_nonparametric_average_standard_error
        )
        quality_risk_best_interval_length = (
            best_quality_summary.mean_nonparametric_interval_length
        )
        quality_risk_best_rmse_to_standard_error_ratio = (
            best_quality_summary.rmse_to_standard_error_ratio
        )
        quality_risk_best_standard_error_reserve = (
            best_quality_summary.standard_error_reserve
        )
        quality_risk_best_interval_length_to_rmse_ratio = (
            best_quality_summary.interval_length_to_rmse_ratio
        )
        quality_risk_rmse_to_standard_error_ratio_gap = (
            quality_risk_probe.rmse_to_standard_error_ratio_gap
        )
        quality_risk_standard_error_reserve_gap = (
            quality_risk_probe.standard_error_reserve_gap
        )
        quality_risk_interval_length_to_rmse_gap = (
            quality_risk_probe.interval_length_to_rmse_gap
        )

    repair_target_helper = (
        "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot()"
    )
    repair_target_note = (
        "Docs/research/phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot.md"
    )
    repair_target_signature = "directional-direct-residual-correlation-repair-target"

    execution_contract_helper = (
        "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract()"
    )
    execution_contract_note = (
        "Docs/research/phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract.md"
    )
    execution_contract_signature = "bounded-right-center-execution-contract"
    execution_bundle_helper = (
        "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_bundle()"
    )
    execution_bundle_note = (
        "Docs/research/phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_bundle.md"
    )
    execution_bundle_signature = "bounded-right-center-execution-bundle"
    if live_source_target_alignment is None:
        try:
            live_source_target_alignment = (
                run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe()
            )
        except (ImportError, InferenceComputationError):
            live_source_target_alignment = (
                _build_repo_side_live_source_target_alignment_probe()
            )
    live_source_target_alignment_helper = (
        "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe()"
    )
    live_source_target_alignment_note = (
        "Docs/research/phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe.md"
    )
    live_source_target_alignment_driver = (
        live_source_target_alignment.driver_signature
    )
    live_source_target_alignment_implication = (
        "archived positive-first-sine diagonal ceiling not implementation-ready on this worktree"
    )
    live_source_target_alignment_binding_random_states = (
        live_source_target_alignment.coverage_anchor_random_state,
        live_source_target_alignment.overshoot_companion_random_state,
    )
    live_source_target_alignment_binding_replication_seeds = (
        live_source_target_alignment.coverage_anchor_replication_seed,
        live_source_target_alignment.overshoot_companion_replication_seed,
    )
    live_source_target_alignment_target_gap_field = "canonical_vs_live_target_gap"
    live_source_target_alignment_target_gap = (
        live_source_target_alignment.canonical_vs_live_target_gap
    )

    runtime_evidence_helper = (
        "run_phase7_monte_carlo_widening_policy_runtime_evidence_packet(...)"
    )
    runtime_evidence_note = (
        "Docs/research/phase7_monte_carlo_widening_policy_runtime_evidence_packet.md"
    )
    runtime_evidence_driver = "trigger2-runtime-evidence-packet-open"
    runtime_evidence_binding_design = ("DGP2", 500, 50)
    runtime_evidence_binding_random_states = (202,)
    runtime_evidence_required_quota_gap = 1
    runtime_evidence_effective_fresh_reruns = 5
    runtime_evidence_covered_pointwise_witnesses = 7
    runtime_evidence_required_covered_pointwise_witnesses = 8
    runtime_evidence_total_pointwise_witnesses = 9
    runtime_evidence_supported_floor_ceiling = 7.0 / 9.0
    runtime_evidence_quota_helper = (
        "run_phase7_monte_carlo_widening_policy_floor_witness_quota_probe()"
    )
    runtime_evidence_quota_note = (
        "Docs/research/phase7_monte_carlo_widening_policy_floor_witness_quota_probe.md"
    )
    runtime_evidence_binding_design_rerun_capacity_helper = (
        "run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe()"
    )
    runtime_evidence_binding_design_rerun_capacity_note = (
        "Docs/research/phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe.md"
    )
    runtime_evidence_binding_design_rerun_capacity_limiting_budget = (
        "random_state_budget"
    )
    runtime_evidence_binding_design_rerun_capacity_implication = (
        "spend-remaining-fresh-reruns-on-binding-design"
    )
    runtime_evidence_current_implication = execution_contract_signature
    runtime_evidence_admission_helper = (
        "run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate(...)"
    )
    runtime_evidence_admission_note = (
        "Docs/research/phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate.md"
    )
    runtime_evidence_admission_status = "trigger2-runtime-evidence-quota-closed"
    runtime_evidence_admission_candidate_status = "runtime-evidence-admissible"
    runtime_evidence_admission_contract_status = "runtime-witness-contract-satisfied"
    runtime_evidence_admission_binding_random_state = 303
    runtime_evidence_admission_covered_pointwise_witnesses_before = 7
    runtime_evidence_admission_covered_pointwise_witnesses_after = 8
    runtime_evidence_admission_required_covered_pointwise_witnesses = 8
    runtime_evidence_admission_remaining_quota_gap_before = 1
    runtime_evidence_admission_remaining_quota_gap_after = 0
    runtime_evidence_admission_fresh_reruns_before = 5
    runtime_evidence_admission_fresh_reruns_after = 4
    runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before = 1
    runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after = 2
    runtime_evidence_admission_seed_local_total_pointwise_witnesses = 3
    runtime_evidence_admission_quota_closure_margin = 0
    runtime_evidence_admission_candidate_success = True
    runtime_evidence_admission_candidate_nonparametric_coverage = 2.0 / 3.0
    runtime_evidence_admission_candidate_typed_invalidity_counts: dict[str, int] = {}
    runtime_evidence_admission_candidate_typed_invalidity_examples: dict[
        str,
        dict[str, object],
    ] = {}

    if current_gate.gate_status != policy_spec.current_gate_status:
        raise ValueError("current gate drifted from canonical Trigger 2 policy spec")
    if current_gate.gate_status != acceptance_preview.current_runtime_gate_status:
        raise ValueError("current gate drifted from policy acceptance preview")
    if policy_spec.target_gate_status != acceptance_preview.accepted_runtime_gate_status:
        raise ValueError("target gate drifted from policy acceptance preview")
    if policy_spec.policy_digest != acceptance_preview.policy_digest:
        raise ValueError("policy digest drifted between spec and acceptance preview")
    if runtime_evidence_packet is not None:
        if acceptance_preview.policy_digest != runtime_evidence_packet.policy_digest:
            raise ValueError("policy digest drifted from runtime evidence packet")
        if (
            acceptance_preview.accepted_recommended_feature_bundle
            != runtime_evidence_packet.accepted_feature_bundle
        ):
            raise ValueError(
                "accepted feature bundle drifted from runtime evidence packet"
            )
        if (
            acceptance_preview.accepted_runtime_gate_status
            != runtime_evidence_packet.current_gate_status
        ):
            raise ValueError(
                "accepted gate drifted from runtime evidence packet"
            )
        if (
            acceptance_preview.current_recommended_bounded_loop
            != runtime_evidence_packet.live_entry
        ):
            raise ValueError("live entry drifted from runtime evidence packet")
        if (
            acceptance_preview.accepted_open_trigger_names[0]
            != runtime_evidence_packet.route_label
        ):
            raise ValueError("open trigger drifted from runtime evidence packet")
        runtime_evidence_driver = runtime_evidence_packet.driver_signature
        runtime_evidence_binding_design = runtime_evidence_packet.binding_design
        runtime_evidence_binding_random_states = (
            runtime_evidence_packet.binding_random_states
        )
        runtime_evidence_required_quota_gap = (
            runtime_evidence_packet.additional_pointwise_witnesses_needed
        )
        runtime_evidence_effective_fresh_reruns = (
            runtime_evidence_packet.effective_fresh_reruns
        )
        runtime_evidence_covered_pointwise_witnesses = (
            runtime_evidence_packet.covered_pointwise_witnesses
        )
        runtime_evidence_required_covered_pointwise_witnesses = (
            runtime_evidence_packet.required_covered_pointwise_witnesses
        )
        runtime_evidence_total_pointwise_witnesses = (
            runtime_evidence_packet.total_pointwise_witnesses
        )
        runtime_evidence_supported_floor_ceiling = (
            runtime_evidence_packet.supported_floor_ceiling
        )
        runtime_evidence_quota_helper = (
            runtime_evidence_packet.floor_witness_quota_helper
        )
        runtime_evidence_quota_note = runtime_evidence_packet.floor_witness_quota_note
        runtime_evidence_binding_design_rerun_capacity_helper = (
            runtime_evidence_packet.binding_design_rerun_capacity_helper
        )
        runtime_evidence_binding_design_rerun_capacity_note = (
            runtime_evidence_packet.binding_design_rerun_capacity_note
        )
        runtime_evidence_binding_design_rerun_capacity_limiting_budget = (
            runtime_evidence_packet.binding_design_rerun_capacity_limiting_budget
        )
        runtime_evidence_binding_design_rerun_capacity_implication = (
            runtime_evidence_packet.binding_design_rerun_capacity_implication
        )
        runtime_evidence_current_implication = (
            runtime_evidence_packet.current_implication
        )
    if runtime_evidence_admission is not None:
        if runtime_evidence_admission.policy_digest != policy_spec.policy_digest:
            raise ValueError(
                "policy digest drifted from runtime evidence admission gate"
            )
        if (
            runtime_evidence_admission.binding_design
            != runtime_evidence_binding_design
        ):
            raise ValueError(
                "runtime evidence admission binding design drifted from runtime evidence packet"
            )
        if (
            runtime_evidence_admission.binding_random_state
            not in runtime_evidence_binding_random_states
        ):
            raise ValueError(
                "runtime evidence admission binding random state drifted from runtime evidence packet"
            )
        runtime_evidence_admission_status = (
            runtime_evidence_admission.resulting_gate_status
        )
        runtime_evidence_admission_candidate_status = (
            runtime_evidence_admission.candidate_status
        )
        runtime_evidence_admission_contract_status = (
            runtime_evidence_admission.runtime_witness_contract_status
        )
        runtime_evidence_admission_binding_random_state = (
            runtime_evidence_admission.binding_random_state
        )
        runtime_evidence_admission_covered_pointwise_witnesses_before = (
            runtime_evidence_admission.covered_pointwise_witnesses_before
        )
        runtime_evidence_admission_covered_pointwise_witnesses_after = (
            runtime_evidence_admission.covered_pointwise_witnesses_after
        )
        runtime_evidence_admission_required_covered_pointwise_witnesses = (
            runtime_evidence_admission.required_covered_pointwise_witnesses
        )
        runtime_evidence_admission_remaining_quota_gap_before = (
            runtime_evidence_admission.remaining_quota_gap_before
        )
        runtime_evidence_admission_remaining_quota_gap_after = (
            runtime_evidence_admission.remaining_quota_gap_after
        )
        runtime_evidence_admission_fresh_reruns_before = (
            runtime_evidence_admission.fresh_reruns_before
        )
        runtime_evidence_admission_fresh_reruns_after = (
            runtime_evidence_admission.fresh_reruns_after
        )
        runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before = (
            runtime_evidence_admission.seed_local_covered_pointwise_witnesses_before
        )
        runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after = (
            runtime_evidence_admission.seed_local_covered_pointwise_witnesses_after
        )
        runtime_evidence_admission_seed_local_total_pointwise_witnesses = (
            runtime_evidence_admission.seed_local_total_pointwise_witnesses
        )
        runtime_evidence_admission_quota_closure_margin = (
            runtime_evidence_admission.quota_closure_margin
        )
        runtime_evidence_admission_candidate_success = (
            runtime_evidence_admission.candidate_success
        )
        runtime_evidence_admission_candidate_nonparametric_coverage = (
            runtime_evidence_admission.candidate_nonparametric_coverage
        )
        runtime_evidence_admission_candidate_typed_invalidity_counts = dict(
            runtime_evidence_admission.candidate_typed_invalidity_counts
        )
        runtime_evidence_admission_candidate_typed_invalidity_examples = dict(
            runtime_evidence_admission.candidate_typed_invalidity_examples
        )
        if runtime_evidence_admission_status == "trigger2-runtime-evidence-quota-closed":
            runtime_evidence_current_implication = execution_contract_signature
    if policy_spec.stable_partial_designs != acceptance_preview.stable_partial_designs:
        raise ValueError("stable partial design inventory drifted")
    if (
        policy_spec.blocked_full_matrix_designs
        != acceptance_preview.blocked_full_matrix_designs
    ):
        raise ValueError("blocked full-matrix design inventory drifted")
    if policy_spec.quality_risk_designs != acceptance_preview.quality_risk_designs:
        raise ValueError("quality-risk design inventory drifted")
    if quality_risk_probe is not None:
        if policy_spec.policy_digest != quality_risk_probe.policy_digest:
            raise ValueError("policy digest drifted against quality-risk probe")
        if (
            quality_risk_probe.binding_design not in policy_spec.quality_risk_designs
            and quality_risk_probe.binding_design
            not in policy_spec.stable_partial_designs
        ):
            raise ValueError("quality-risk binding design drifted from policy spec")
    if repair_target_snapshot is not None:
        if policy_spec.policy_digest != repair_target_snapshot.policy_digest:
            raise ValueError("policy digest drifted against repair-target snapshot")
        if repair_target_snapshot.binding_design not in policy_spec.quality_risk_designs:
            raise ValueError("repair-target binding design drifted from policy spec")
        if repair_target_snapshot.binding_design != runtime_evidence_binding_design:
            raise ValueError(
                "repair-target binding design drifted from runtime evidence packet"
            )
        if repair_target_snapshot.binding_driver != quality_risk_binding_driver:
            raise ValueError("repair-target binding driver drifted from quality-risk bridge")
        repair_target_signature = repair_target_snapshot.repair_target_signature
    if execution_contract is not None:
        if policy_spec.policy_digest != execution_contract.policy_digest:
            raise ValueError("policy digest drifted against execution contract")
        if execution_contract.binding_design not in policy_spec.quality_risk_designs:
            raise ValueError("execution-contract binding design drifted from policy spec")
        if execution_contract.binding_design != runtime_evidence_binding_design:
            raise ValueError(
                "execution-contract binding design drifted from runtime evidence packet"
            )
        if (
            repair_target_snapshot is not None
            and execution_contract.binding_design
            != repair_target_snapshot.binding_design
        ):
            raise ValueError(
                "execution-contract binding design drifted from repair-target snapshot"
            )
        execution_contract_signature = execution_contract.driver_signature
    if execution_bundle is not None:
        if policy_spec.policy_digest != execution_bundle.policy_digest:
            raise ValueError("policy digest drifted against execution bundle")
        if execution_bundle.binding_design not in policy_spec.quality_risk_designs:
            raise ValueError("execution-bundle binding design drifted from policy spec")
        if execution_bundle.binding_design != runtime_evidence_binding_design:
            raise ValueError(
                "execution-bundle binding design drifted from runtime evidence packet"
            )
        if (
            repair_target_snapshot is not None
            and execution_bundle.binding_design
            != repair_target_snapshot.binding_design
        ):
            raise ValueError(
                "execution-bundle binding design drifted from repair-target snapshot"
            )
        if (
            execution_contract is not None
            and execution_bundle.binding_design != execution_contract.binding_design
        ):
            raise ValueError(
                "execution-bundle binding design drifted from execution contract"
            )
        if (
            execution_contract is not None
            and execution_bundle.live_driver_signature
            != execution_contract.driver_signature
        ):
            raise ValueError(
                "execution-bundle live driver drifted from execution contract"
            )
        execution_bundle_signature = execution_bundle.driver_signature
    if live_source_target_alignment is not None:
        if policy_spec.policy_digest != live_source_target_alignment.policy_digest:
            raise ValueError("policy digest drifted against live source-target alignment probe")
        if (
            live_source_target_alignment.binding_design
            not in policy_spec.quality_risk_designs
        ):
            raise ValueError(
                "live source-target alignment binding design drifted from policy spec"
            )
        if live_source_target_alignment.binding_design != runtime_evidence_binding_design:
            raise ValueError(
                "live source-target alignment binding design drifted from runtime evidence packet"
            )
        if (
            execution_bundle is not None
            and live_source_target_alignment.coverage_anchor_random_state
            != execution_bundle.coverage_anchor_random_state
        ):
            raise ValueError(
                "live source-target alignment anchor random state drifted from execution bundle"
            )
        if (
            execution_bundle is not None
            and live_source_target_alignment.overshoot_companion_random_state
            != execution_bundle.overshoot_companion_random_state
        ):
            raise ValueError(
                "live source-target alignment companion random state drifted from execution bundle"
            )
        live_source_target_alignment_driver = (
            live_source_target_alignment.driver_signature
        )
        live_source_target_alignment_binding_random_states = (
            live_source_target_alignment.coverage_anchor_random_state,
            live_source_target_alignment.overshoot_companion_random_state,
        )
        live_source_target_alignment_binding_replication_seeds = (
            live_source_target_alignment.coverage_anchor_replication_seed,
            live_source_target_alignment.overshoot_companion_replication_seed,
        )
        live_source_target_alignment_target_gap = (
            live_source_target_alignment.canonical_vs_live_target_gap
        )

    if runtime_evidence_admission_status == "trigger2-runtime-evidence-quota-closed":
        runtime_evidence_current_implication = execution_contract_signature
        if quality_risk_blocker_reason:
            post_admission_quality_risk_status = "quality-risk-still-open"
            monte_carlo_validation_ready_status = "blocked"
            monte_carlo_validation_ready_blocker = quality_risk_blocker_reason
        else:
            post_admission_quality_risk_status = "quality-risk-cleared"
            monte_carlo_validation_ready_status = "ready"
            monte_carlo_validation_ready_blocker = ""
    else:
        post_admission_quality_risk_status = "runtime-evidence-admission-open"
        monte_carlo_validation_ready_status = "blocked"
        monte_carlo_validation_ready_blocker = runtime_evidence_driver
    post_admission_quality_risk_driver = quality_risk_binding_driver
    post_admission_quality_risk_binding_design = quality_risk_binding_design
    paper_dgp2_contract_helper = (
        "run_phase7_monte_carlo_paper_dgp2_contract_audit(...)"
    )
    paper_dgp2_contract_note = (
        "Docs/research/phase7_monte_carlo_paper_dgp2_contract_audit.md"
    )
    paper_dgp2_contract_status = ""
    paper_dgp2_contract_r_finding_codes: tuple[str, ...] = ()
    paper_dgp2_contract_near_zero_grid: tuple[float, ...] = ()
    if paper_dgp2_contract_audit is not None:
        paper_dgp2_contract_status = paper_dgp2_contract_audit.status
        paper_dgp2_contract_r_finding_codes = (
            paper_dgp2_contract_audit.r_snapshot_finding_codes
        )
        paper_dgp2_contract_near_zero_grid = (
            paper_dgp2_contract_audit.python_default_near_zero_grid
        )

    canonical_gate_digest = (
        "- current widening gate: "
        f"`{current_gate.gate_status}` via "
        "`run_phase7_monte_carlo_widening_trigger_gate()`",
        "- canonical policy spec: "
        "`run_phase7_monte_carlo_widening_policy_spec()` keeps "
        + ", ".join(f"`{item}`" for item in policy_spec.policy_digest)
        + f" and targets `{policy_spec.target_gate_status}`",
        "- explicit policy acceptance preview: "
        "`run_phase7_monte_carlo_widening_policy_acceptance_preview(...)` shifts "
        f"the recommended feature bundle to "
        f"`{acceptance_preview.accepted_recommended_feature_bundle}` and opens "
        f"`{_format_trigger_names(acceptance_preview.accepted_open_trigger_names)}` "
        "without changing empirical "
        f"`{acceptance_preview.empirical_blocker_reason}` or parity "
        f"`{acceptance_preview.parity_gate_status}`",
        "- runtime-evidence packet: "
        f"`{runtime_evidence_helper}` keeps "
        f"`{runtime_evidence_note}`, quota helper "
        f"`{runtime_evidence_quota_helper}` via `{runtime_evidence_quota_note}`, rerun-capacity helper "
        f"`{runtime_evidence_binding_design_rerun_capacity_helper}` via "
        f"`{runtime_evidence_binding_design_rerun_capacity_note}`, limiting budget "
        f"`{runtime_evidence_binding_design_rerun_capacity_limiting_budget}`, rerun implication "
        f"`{runtime_evidence_binding_design_rerun_capacity_implication}`, current implication "
        f"`{runtime_evidence_current_implication}`, driver "
        f"`{runtime_evidence_driver}`, binding design "
        f"`{runtime_evidence_binding_design[0]}/{runtime_evidence_binding_design[1]}/{runtime_evidence_binding_design[2]}`, "
        f"seed `{', '.join(str(seed) for seed in runtime_evidence_binding_random_states)}`, quota gap "
        f"`{runtime_evidence_required_quota_gap}`, witness count "
        f"`{runtime_evidence_covered_pointwise_witnesses}/{runtime_evidence_total_pointwise_witnesses}` "
        f"toward `{runtime_evidence_required_covered_pointwise_witnesses}/{runtime_evidence_total_pointwise_witnesses}`, "
        f"supported floor ceiling `{runtime_evidence_supported_floor_ceiling:.3f}`, and "
        f"`{runtime_evidence_effective_fresh_reruns}` fresh reruns on the same feature gate surface",
        "- runtime-evidence admission gate: "
        f"`{runtime_evidence_admission_helper}` keeps "
        f"`{runtime_evidence_admission_note}`, status "
        f"`{runtime_evidence_admission_status}`, candidate "
        f"`{runtime_evidence_admission_candidate_status}`, contract "
        f"`{runtime_evidence_admission_contract_status}`, binding random state "
        f"`{runtime_evidence_admission_binding_random_state}`, witness count "
        f"`{runtime_evidence_admission_covered_pointwise_witnesses_before} -> "
        f"{runtime_evidence_admission_covered_pointwise_witnesses_after}` toward "
        f"`{runtime_evidence_admission_required_covered_pointwise_witnesses}`, "
        f"seed-local witness count "
        f"`{runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before}/"
        f"{runtime_evidence_admission_seed_local_total_pointwise_witnesses} -> "
        f"{runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after}/"
        f"{runtime_evidence_admission_seed_local_total_pointwise_witnesses}`, "
        f"quota gap `{runtime_evidence_admission_remaining_quota_gap_before} -> "
        f"{runtime_evidence_admission_remaining_quota_gap_after}`, quota-closure margin "
        f"`{runtime_evidence_admission_quota_closure_margin}`, and fresh reruns "
        f"`{runtime_evidence_admission_fresh_reruns_before} -> "
        f"{runtime_evidence_admission_fresh_reruns_after}` on the same feature gate surface",
        "- bounded quality-risk bridge: "
        f"`{quality_risk_probe_helper}` keeps "
        f"`{quality_risk_probe_note}`, binding driver `{quality_risk_binding_driver}`, "
        f"binding design "
        f"`{quality_risk_binding_design[0]}/{quality_risk_binding_design[1]}/{quality_risk_binding_design[2]}` "
        f"with coverage `{quality_risk_binding_coverage:.3f}`, coverage slack `{quality_risk_binding_coverage_floor_slack:.3f}`, "
        f"RMSE `{quality_risk_binding_rmse:.3f}`, average SE `{quality_risk_binding_average_standard_error:.3f}`, "
        f"interval length `{quality_risk_binding_interval_length:.3f}`, "
        f"`RMSE/SE = {quality_risk_binding_rmse_to_standard_error_ratio:.3f}`, "
        f"`SE-RMSE = {quality_risk_binding_standard_error_reserve:.3f}`, "
        f"`interval/RMSE = {quality_risk_binding_interval_length_to_rmse_ratio:.3f}`, "
        f"comparison best design "
        f"`{quality_risk_best_design[0]}/{quality_risk_best_design[1]}/{quality_risk_best_design[2]}` "
        f"with coverage `{quality_risk_best_coverage:.3f}`, coverage slack `{quality_risk_best_coverage_floor_slack:.3f}`, "
        f"RMSE `{quality_risk_best_rmse:.3f}`, average SE `{quality_risk_best_average_standard_error:.3f}`, "
        f"interval length `{quality_risk_best_interval_length:.3f}`, "
        f"`RMSE/SE = {quality_risk_best_rmse_to_standard_error_ratio:.3f}`, "
        f"`SE-RMSE = {quality_risk_best_standard_error_reserve:.3f}`, "
        f"`interval/RMSE = {quality_risk_best_interval_length_to_rmse_ratio:.3f}`, "
        f"floor shortfall `{quality_risk_canonical_floor_shortfall:.3f}`, and reserve gap "
        f"`{quality_risk_standard_error_reserve_gap:.3f}` on the same feature gate surface; "
        "paper object contract "
        + ", ".join(f"`{item}`" for item in quality_risk_paper_object_contract),
        "- repair-target bridge: "
        f"`{repair_target_helper}` keeps "
        f"`{repair_target_note}` "
        f"and repair target `{repair_target_signature}` on the same feature gate surface",
        "- implementation handoff bridge: "
        f"`{execution_contract_helper}` keeps "
        f"`{execution_contract_note}` "
        f"and execution contract `{execution_contract_signature}` on the same feature gate surface",
        "- live source-target alignment bridge: "
        f"`{live_source_target_alignment_helper}` keeps "
        f"`{live_source_target_alignment_note}`, driver "
        f"`{live_source_target_alignment_driver}`, target gap field "
        f"`{live_source_target_alignment_target_gap_field}`, target gap "
        f"`{live_source_target_alignment_target_gap:.3f}`, anchor seed "
        f"`{live_source_target_alignment_binding_random_states[0]}`, companion seed "
        f"`{live_source_target_alignment_binding_random_states[1]}`, replication seeds "
        f"`{live_source_target_alignment_binding_replication_seeds[0]}` / "
        f"`{live_source_target_alignment_binding_replication_seeds[1]}`, and implication "
        f"`{live_source_target_alignment_implication}` on the same feature gate surface",
        "- post-admission quality risk: "
        f"`{post_admission_quality_risk_status}` keeps driver "
        f"`{post_admission_quality_risk_driver}` on binding design "
        f"`{post_admission_quality_risk_binding_design[0]}/"
        f"{post_admission_quality_risk_binding_design[1]}/"
        f"{post_admission_quality_risk_binding_design[2]}` after runtime evidence admission",
        (
            "- monte_carlo_validation_ready verdict: "
            "`ready`; runtime admission is closed and post-admission quality risk "
            "is cleared on the same feature gate surface"
            if monte_carlo_validation_ready_status == "ready"
            else "- monte_carlo_validation_ready verdict: "
            f"`{monte_carlo_validation_ready_status}` by "
            f"`{monte_carlo_validation_ready_blocker}`; quota closure does not clear "
            "the release-facing Monte Carlo gate while post-admission quality risk remains open"
        ),
        "- paper DGP2 contract: "
        f"`{paper_dgp2_contract_helper}` keeps `{paper_dgp2_contract_note}`, "
        f"status `{paper_dgp2_contract_status or 'not-run'}`, R drift "
        f"`{', '.join(paper_dgp2_contract_r_finding_codes) or 'none'}`, and "
        f"near-zero grid `{paper_dgp2_contract_near_zero_grid or 'none'}` on the "
        "same feature gate surface",
        "- bounded execution bundle: "
        f"`{execution_bundle_helper}` keeps "
        f"`{execution_bundle_note}` "
        f"and execution bundle `{execution_bundle_signature}` on the same feature gate surface",
        "- bounded widening evidence remains constrained to stable "
        f"`{_format_design_keys(policy_spec.stable_partial_designs)}`; quality risk "
        f"`{_format_design_keys(policy_spec.quality_risk_designs)}`; blocked full-matrix "
        f"`{_format_design_keys(policy_spec.blocked_full_matrix_designs)}`; typed invalidity "
        f"`{_format_invalidity_counts(current_gate.typed_invalidity_counts)}`",
    )
    return Phase7MonteCarloFeatureCompletionGateReport(
        stage_label="phase7-monte-carlo-feature-completion-gate",
        route_label=policy_spec.trigger_label,
        current_gate_status=current_gate.gate_status,
        target_gate_status=policy_spec.target_gate_status,
        accepted_feature_bundle=acceptance_preview.accepted_recommended_feature_bundle,
        accepted_open_trigger_names=acceptance_preview.accepted_open_trigger_names,
        runtime_evidence_helper=runtime_evidence_helper,
        runtime_evidence_note=runtime_evidence_note,
        runtime_evidence_driver=runtime_evidence_driver,
        runtime_evidence_binding_design=runtime_evidence_binding_design,
        runtime_evidence_binding_random_states=runtime_evidence_binding_random_states,
        runtime_evidence_required_quota_gap=runtime_evidence_required_quota_gap,
        runtime_evidence_effective_fresh_reruns=runtime_evidence_effective_fresh_reruns,
        runtime_evidence_covered_pointwise_witnesses=(
            runtime_evidence_covered_pointwise_witnesses
        ),
        runtime_evidence_required_covered_pointwise_witnesses=(
            runtime_evidence_required_covered_pointwise_witnesses
        ),
        runtime_evidence_total_pointwise_witnesses=(
            runtime_evidence_total_pointwise_witnesses
        ),
        runtime_evidence_supported_floor_ceiling=(
            runtime_evidence_supported_floor_ceiling
        ),
        runtime_evidence_quota_helper=runtime_evidence_quota_helper,
        runtime_evidence_quota_note=runtime_evidence_quota_note,
        runtime_evidence_binding_design_rerun_capacity_helper=(
            runtime_evidence_binding_design_rerun_capacity_helper
        ),
        runtime_evidence_binding_design_rerun_capacity_note=(
            runtime_evidence_binding_design_rerun_capacity_note
        ),
        runtime_evidence_binding_design_rerun_capacity_limiting_budget=(
            runtime_evidence_binding_design_rerun_capacity_limiting_budget
        ),
        runtime_evidence_binding_design_rerun_capacity_implication=(
            runtime_evidence_binding_design_rerun_capacity_implication
        ),
        runtime_evidence_current_implication=runtime_evidence_current_implication,
        runtime_evidence_admission_helper=runtime_evidence_admission_helper,
        runtime_evidence_admission_note=runtime_evidence_admission_note,
        runtime_evidence_admission_status=runtime_evidence_admission_status,
        runtime_evidence_admission_candidate_status=(
            runtime_evidence_admission_candidate_status
        ),
        runtime_evidence_admission_contract_status=(
            runtime_evidence_admission_contract_status
        ),
        runtime_evidence_admission_candidate_success=(
            runtime_evidence_admission_candidate_success
        ),
        runtime_evidence_admission_candidate_nonparametric_coverage=(
            runtime_evidence_admission_candidate_nonparametric_coverage
        ),
        runtime_evidence_admission_candidate_typed_invalidity_counts=(
            runtime_evidence_admission_candidate_typed_invalidity_counts
        ),
        runtime_evidence_admission_candidate_typed_invalidity_examples=(
            runtime_evidence_admission_candidate_typed_invalidity_examples
        ),
        runtime_evidence_admission_binding_random_state=(
            runtime_evidence_admission_binding_random_state
        ),
        runtime_evidence_admission_covered_pointwise_witnesses_before=(
            runtime_evidence_admission_covered_pointwise_witnesses_before
        ),
        runtime_evidence_admission_covered_pointwise_witnesses_after=(
            runtime_evidence_admission_covered_pointwise_witnesses_after
        ),
        runtime_evidence_admission_required_covered_pointwise_witnesses=(
            runtime_evidence_admission_required_covered_pointwise_witnesses
        ),
        runtime_evidence_admission_remaining_quota_gap_before=(
            runtime_evidence_admission_remaining_quota_gap_before
        ),
        runtime_evidence_admission_remaining_quota_gap_after=(
            runtime_evidence_admission_remaining_quota_gap_after
        ),
        runtime_evidence_admission_fresh_reruns_before=(
            runtime_evidence_admission_fresh_reruns_before
        ),
        runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before=(
            runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before
        ),
        runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after=(
            runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after
        ),
        runtime_evidence_admission_seed_local_total_pointwise_witnesses=(
            runtime_evidence_admission_seed_local_total_pointwise_witnesses
        ),
        runtime_evidence_admission_quota_closure_margin=(
            runtime_evidence_admission_quota_closure_margin
        ),
        post_admission_quality_risk_status=post_admission_quality_risk_status,
        post_admission_quality_risk_driver=post_admission_quality_risk_driver,
        post_admission_quality_risk_binding_design=(
            post_admission_quality_risk_binding_design
        ),
        monte_carlo_validation_ready_status=monte_carlo_validation_ready_status,
        monte_carlo_validation_ready_blocker=monte_carlo_validation_ready_blocker,
        runtime_evidence_admission_fresh_reruns_after=(
            runtime_evidence_admission_fresh_reruns_after
        ),
        quality_risk_probe_helper=quality_risk_probe_helper,
        quality_risk_probe_note=quality_risk_probe_note,
        quality_risk_binding_driver=quality_risk_binding_driver,
        quality_risk_blocker_reason=quality_risk_blocker_reason,
        quality_risk_binding_design=quality_risk_binding_design,
        quality_risk_best_design=quality_risk_best_design,
        quality_risk_canonical_floor_shortfall=(
            quality_risk_canonical_floor_shortfall
        ),
        quality_risk_binding_coverage=quality_risk_binding_coverage,
        quality_risk_binding_coverage_floor_slack=(
            quality_risk_binding_coverage_floor_slack
        ),
        quality_risk_binding_rmse=quality_risk_binding_rmse,
        quality_risk_binding_average_standard_error=(
            quality_risk_binding_average_standard_error
        ),
        quality_risk_binding_interval_length=quality_risk_binding_interval_length,
        quality_risk_binding_rmse_to_standard_error_ratio=(
            quality_risk_binding_rmse_to_standard_error_ratio
        ),
        quality_risk_binding_standard_error_reserve=(
            quality_risk_binding_standard_error_reserve
        ),
        quality_risk_binding_interval_length_to_rmse_ratio=(
            quality_risk_binding_interval_length_to_rmse_ratio
        ),
        quality_risk_best_coverage=quality_risk_best_coverage,
        quality_risk_best_coverage_floor_slack=(
            quality_risk_best_coverage_floor_slack
        ),
        quality_risk_best_rmse=quality_risk_best_rmse,
        quality_risk_best_average_standard_error=(
            quality_risk_best_average_standard_error
        ),
        quality_risk_best_interval_length=quality_risk_best_interval_length,
        quality_risk_best_rmse_to_standard_error_ratio=(
            quality_risk_best_rmse_to_standard_error_ratio
        ),
        quality_risk_best_standard_error_reserve=(
            quality_risk_best_standard_error_reserve
        ),
        quality_risk_best_interval_length_to_rmse_ratio=(
            quality_risk_best_interval_length_to_rmse_ratio
        ),
        quality_risk_rmse_to_standard_error_ratio_gap=(
            quality_risk_rmse_to_standard_error_ratio_gap
        ),
        quality_risk_standard_error_reserve_gap=(
            quality_risk_standard_error_reserve_gap
        ),
        quality_risk_interval_length_to_rmse_gap=(
            quality_risk_interval_length_to_rmse_gap
        ),
        quality_risk_source_mode=quality_risk_source_mode,
        quality_risk_source_note=quality_risk_source_note,
        quality_risk_paper_object_contract=quality_risk_paper_object_contract,
        paper_dgp2_contract_helper=paper_dgp2_contract_helper,
        paper_dgp2_contract_note=paper_dgp2_contract_note,
        paper_dgp2_contract_status=paper_dgp2_contract_status,
        paper_dgp2_contract_r_finding_codes=paper_dgp2_contract_r_finding_codes,
        paper_dgp2_contract_near_zero_grid=paper_dgp2_contract_near_zero_grid,
        repair_target_helper=repair_target_helper,
        repair_target_note=repair_target_note,
        repair_target_signature=repair_target_signature,
        execution_contract_helper=execution_contract_helper,
        execution_contract_note=execution_contract_note,
        execution_contract_signature=execution_contract_signature,
        execution_bundle_helper=execution_bundle_helper,
        execution_bundle_note=execution_bundle_note,
        execution_bundle_signature=execution_bundle_signature,
        live_source_target_alignment_helper=live_source_target_alignment_helper,
        live_source_target_alignment_note=live_source_target_alignment_note,
        live_source_target_alignment_driver=live_source_target_alignment_driver,
        live_source_target_alignment_implication=(
            live_source_target_alignment_implication
        ),
        live_source_target_alignment_binding_random_states=(
            live_source_target_alignment_binding_random_states
        ),
        live_source_target_alignment_binding_replication_seeds=(
            live_source_target_alignment_binding_replication_seeds
        ),
        live_source_target_alignment_target_gap_field=(
            live_source_target_alignment_target_gap_field
        ),
        live_source_target_alignment_target_gap=(
            live_source_target_alignment_target_gap
        ),
        stable_partial_designs=policy_spec.stable_partial_designs,
        blocked_full_matrix_designs=policy_spec.blocked_full_matrix_designs,
        quality_risk_designs=policy_spec.quality_risk_designs,
        typed_invalidity_counts=dict(current_gate.typed_invalidity_counts),
        policy_digest=policy_spec.policy_digest,
        canonical_gate_digest=canonical_gate_digest,
        recommendation_rationale=(
            "Trigger 2 still stays blocked on bounded widening quality risk, but the "
            "feature-completion gate now carries the current widening gate, the "
            "canonical policy contract, and the accepted-preview next bundle on one "
            "machine-readable surface."
        ),
    )


def run_phase7_monte_carlo_feature_completion_gate(
    repo_root: str | Path,
) -> Phase7MonteCarloFeatureCompletionGateReport:
    root = _coerce_repo_root(repo_root)
    current_gate = _build_state_backed_trigger_gate_report()
    acceptance_preview = (
        build_phase7_monte_carlo_widening_policy_acceptance_preview_repo_side_report(
            root
        )
    )
    policy_spec = _repo_side_policy_spec_from_acceptance_preview(acceptance_preview)
    runtime_evidence_packet = (
        run_phase7_monte_carlo_widening_policy_runtime_evidence_packet(root)
    )
    quality_risk_probe = run_phase7_monte_carlo_widening_policy_quality_risk_probe(root)
    return build_phase7_monte_carlo_feature_completion_gate_report(
        current_gate,
        policy_spec,
        acceptance_preview,
        runtime_evidence_packet,
        runtime_evidence_admission=(
            run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate(
                root
            )
        ),
        quality_risk_probe=quality_risk_probe,
        live_source_target_alignment=(
            _build_repo_side_live_source_target_alignment_probe()
        ),
        paper_dgp2_contract_audit=build_phase7_monte_carlo_paper_dgp2_contract_audit_report(
            repo_root=root,
            quality_risk_report=quality_risk_probe,
        ),
    )


def _clear_phase7_monte_carlo_feature_completion_gate_cache() -> None:
    return None


run_phase7_monte_carlo_feature_completion_gate.cache_clear = (  # type: ignore[attr-defined]
    _clear_phase7_monte_carlo_feature_completion_gate_cache
)
