from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_dense_replay_guard_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchDenseReplayGuardReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_dense_replay_guard_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_acceptance_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunAcceptanceContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_acceptance_contract,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOMATION_STATE_PATH = _REPO_ROOT / "Docs" / "automation" / "automation-state.yaml"
_CANONICAL_POLICY_DIGEST = (
    "label=bounded-n500-p50",
    "max_total_runtime_seconds=240.0",
    "max_random_states=8",
    "stop_on_first_typed_invalidity=True",
    "min_nonparametric_coverage=0.85",
)
_BINDING_DESIGN = ("DGP2", 500, 50)
_WINDOW_LABEL = "near_zero_grid"
_COVERAGE_ANCHOR_RANDOM_STATE = 202
_OVERSHOOT_COMPANION_RANDOM_STATE = 505
_SOURCE_DIAGONAL_COORDINATE = 2
_SOURCE_DIAGONAL_BASIS_LABEL = "sin(2πz)"
_SHARED_VF_ENTRY_LABEL = "v_f_hat[2,2]"
_SOURCE_TARGET_OMEGA_DIAGONAL_ENTRY = 410.4739959618635
_SOURCE_REQUIRED_OMEGA_DIAGONAL_INCREMENT = 185.45799327602003
_REQUIRED_DIAGONAL_VF_ENTRY_LIFT = 1011.1822863613054
_VALIDATION_PATCH_INCREMENT = 1.6361273081544214
_PATCH_FROBENIUS_SHARE_OF_BASELINE = 0.10186976130234546
_PATCH_SPECTRAL_SHARE_OF_BASELINE = 0.0995054677493893
_COMPANION_RIGHT_ROW_REPLAY_MULTIPLE = 112.48697232225774
_COMPANION_CENTER_COLUMN_REPLAY_MULTIPLE = 52.79303721638472
_SUPPORTED_FLOOR_CEILING = 7.0 / 9.0
_CANONICAL_FLOOR = 0.85
_REQUIRED_EVIDENCE_CONDITIONS = (
    "keep `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` as the bounded object flow",
    "reject whole-row / whole-column / dense covariance replay",
    "treat fresh reruns as informative only when they lift the binding floor witness",
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _format_design_key(binding_design: tuple[str, int, int]) -> str:
    return f"{binding_design[0]}/{binding_design[1]}/{binding_design[2]}"


def _load_feature_completion_trigger2_state() -> dict[str, object]:
    state = yaml.safe_load(_AUTOMATION_STATE_PATH.read_text(encoding="utf-8"))
    return state["feature_completion_gate"]["checks"]["monte_carlo_validation_ready"]


def _parse_design_key(design_label: str) -> tuple[str, int, int]:
    dgp, n_obs, p = str(design_label).split("/")
    return (str(dgp), int(n_obs), int(p))


def _driver_signature(
    *,
    object_flow_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport
    ),
    dense_replay_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchDenseReplayGuardReport
    ),
    acceptance_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunAcceptanceContractReport
    ),
    intake_contract_holds: bool,
) -> str:
    if (
        object_flow_report.driver_signature
        == "bounded-first-sine-estimator-object-flow-contract"
        and dense_replay_guard_report.driver_signature
        == "bounded-right-center-entry-patch-dense-replay-guard"
        and acceptance_contract_report.driver_signature
        == "fresh-rerun-must-carry-floor-lift-evidence"
        and intake_contract_holds
    ):
        return "bounded-entry-patch-fresh-estimator-evidence-intake-contract"
    return "mixed-entry-patch-fresh-estimator-evidence-intake-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    object_flow_driver_signature: str
    dense_replay_driver_signature: str
    acceptance_driver_signature: str
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    shared_vf_entry_label: str
    source_target_omega_diagonal_entry: float
    source_required_omega_diagonal_increment: float
    required_diagonal_vf_entry_lift: float
    validation_patch_increment: float
    patch_frobenius_share_of_baseline: float
    patch_spectral_share_of_baseline: float
    companion_right_row_replay_multiple_vs_required_increment: float
    companion_center_column_replay_multiple_vs_required_increment: float
    effective_fresh_reruns: int
    supported_floor_ceiling: float
    canonical_floor: float
    canonical_floor_shortfall: float
    target_gate_status: str
    required_evidence_conditions: tuple[str, ...]
    intake_contract_holds: bool
    driver_signature: str
    canonical_fresh_estimator_evidence_intake_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.object_flow_driver_signature = str(
            self.object_flow_driver_signature
        ).strip()
        self.dense_replay_driver_signature = str(
            self.dense_replay_driver_signature
        ).strip()
        self.acceptance_driver_signature = str(self.acceptance_driver_signature).strip()
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.source_target_omega_diagonal_entry = float(
            self.source_target_omega_diagonal_entry
        )
        self.source_required_omega_diagonal_increment = float(
            self.source_required_omega_diagonal_increment
        )
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.validation_patch_increment = float(self.validation_patch_increment)
        self.patch_frobenius_share_of_baseline = float(
            self.patch_frobenius_share_of_baseline
        )
        self.patch_spectral_share_of_baseline = float(
            self.patch_spectral_share_of_baseline
        )
        self.companion_right_row_replay_multiple_vs_required_increment = float(
            self.companion_right_row_replay_multiple_vs_required_increment
        )
        self.companion_center_column_replay_multiple_vs_required_increment = float(
            self.companion_center_column_replay_multiple_vs_required_increment
        )
        self.effective_fresh_reruns = int(self.effective_fresh_reruns)
        self.supported_floor_ceiling = float(self.supported_floor_ceiling)
        self.canonical_floor = float(self.canonical_floor)
        self.canonical_floor_shortfall = float(self.canonical_floor_shortfall)
        self.target_gate_status = str(self.target_gate_status).strip()
        self.required_evidence_conditions = tuple(
            str(item).strip() for item in self.required_evidence_conditions
        )
        self.intake_contract_holds = bool(self.intake_contract_holds)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_fresh_estimator_evidence_intake_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_fresh_estimator_evidence_intake_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "object_flow_driver_signature": self.object_flow_driver_signature,
            "dense_replay_driver_signature": self.dense_replay_driver_signature,
            "acceptance_driver_signature": self.acceptance_driver_signature,
            "source_diagonal_coordinate": self.source_diagonal_coordinate,
            "source_diagonal_basis_label": self.source_diagonal_basis_label,
            "shared_vf_entry_label": self.shared_vf_entry_label,
            "source_target_omega_diagonal_entry": (
                self.source_target_omega_diagonal_entry
            ),
            "source_required_omega_diagonal_increment": (
                self.source_required_omega_diagonal_increment
            ),
            "required_diagonal_vf_entry_lift": self.required_diagonal_vf_entry_lift,
            "validation_patch_increment": self.validation_patch_increment,
            "patch_frobenius_share_of_baseline": (
                self.patch_frobenius_share_of_baseline
            ),
            "patch_spectral_share_of_baseline": self.patch_spectral_share_of_baseline,
            "companion_right_row_replay_multiple_vs_required_increment": (
                self.companion_right_row_replay_multiple_vs_required_increment
            ),
            "companion_center_column_replay_multiple_vs_required_increment": (
                self.companion_center_column_replay_multiple_vs_required_increment
            ),
            "effective_fresh_reruns": self.effective_fresh_reruns,
            "supported_floor_ceiling": self.supported_floor_ceiling,
            "canonical_floor": self.canonical_floor,
            "canonical_floor_shortfall": self.canonical_floor_shortfall,
            "target_gate_status": self.target_gate_status,
            "required_evidence_conditions": list(self.required_evidence_conditions),
            "intake_contract_holds": self.intake_contract_holds,
            "driver_signature": self.driver_signature,
            "canonical_fresh_estimator_evidence_intake_digest": list(
                self.canonical_fresh_estimator_evidence_intake_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract_report(
    *,
    object_flow_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport
    ),
    dense_replay_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchDenseReplayGuardReport
    ),
    acceptance_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunAcceptanceContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport:
    if object_flow_report.policy_digest != dense_replay_guard_report.policy_digest:
        raise ValueError(
            "fresh estimator evidence intake contract requires shared policy digest"
        )
    if object_flow_report.policy_digest != acceptance_contract_report.policy_digest:
        raise ValueError(
            "fresh estimator evidence intake contract requires shared policy digest"
        )
    if object_flow_report.binding_design != dense_replay_guard_report.binding_design:
        raise ValueError(
            "fresh estimator evidence intake contract requires shared binding design"
        )
    if object_flow_report.binding_design != acceptance_contract_report.binding_design:
        raise ValueError(
            "fresh estimator evidence intake contract requires shared binding design"
        )
    if object_flow_report.window_label != dense_replay_guard_report.window_label:
        raise ValueError(
            "fresh estimator evidence intake contract requires shared window label"
        )
    if object_flow_report.window_label != acceptance_contract_report.window_label:
        raise ValueError(
            "fresh estimator evidence intake contract requires shared window label"
        )
    if (
        object_flow_report.coverage_anchor_random_state
        != dense_replay_guard_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match dense-replay guard")
    if (
        object_flow_report.coverage_anchor_random_state
        != acceptance_contract_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match acceptance contract")
    if (
        object_flow_report.overshoot_companion_random_state
        != dense_replay_guard_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match dense-replay guard")
    if (
        object_flow_report.overshoot_companion_random_state
        != acceptance_contract_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match acceptance contract")

    required_evidence_conditions = (
        "keep `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` as the bounded object flow",
        "reject whole-row / whole-column / dense covariance replay",
        "treat fresh reruns as informative only when they lift the binding floor witness",
    )
    intake_contract_holds = bool(
        object_flow_report.driver_signature
        == "bounded-first-sine-estimator-object-flow-contract"
        and dense_replay_guard_report.dense_replay_rejected
        and dense_replay_guard_report.driver_signature
        == "bounded-right-center-entry-patch-dense-replay-guard"
        and acceptance_contract_report.acceptance_contract_holds
        and acceptance_contract_report.driver_signature
        == "fresh-rerun-must-carry-floor-lift-evidence"
        and acceptance_contract_report.canonical_floor_shortfall > 0.0
        and object_flow_report.source_diagonal_coordinate == 2
        and object_flow_report.source_diagonal_basis_label == "sin(2πz)"
        and object_flow_report.shared_vf_entry_label == "v_f_hat[2,2]"
    )
    driver_signature = _driver_signature(
        object_flow_report=object_flow_report,
        dense_replay_guard_report=dense_replay_guard_report,
        acceptance_contract_report=acceptance_contract_report,
        intake_contract_holds=intake_contract_holds,
    )
    binding_design_key = _format_design_key(object_flow_report.binding_design)

    canonical_digest = (
        f"- bounded estimator evidence still has to enter through the same source-backed object flow for `{binding_design_key}` on `{object_flow_report.window_label}`: coordinate `{object_flow_report.source_diagonal_coordinate} = {object_flow_report.source_diagonal_basis_label}` must lift `omega_f_hat[2,2]` by `{_format_signed_float(object_flow_report.source_required_omega_diagonal_increment)}` up to `{_format_float(object_flow_report.source_target_omega_diagonal_entry)}`, then route through `{object_flow_report.shared_vf_entry_label}` with required diagonal lift `{_format_signed_float(object_flow_report.required_diagonal_vf_entry_lift)}` before it can explain the bounded raw covariance patch `{_format_signed_float(object_flow_report.validation_patch_increment)}`",
        f"- the same evidence remains invalid if it widens into dense replay: the bounded patch still uses only `{_format_percent(dense_replay_guard_report.patch_frobenius_share_of_baseline)}` / `{_format_percent(dense_replay_guard_report.patch_spectral_share_of_baseline)}` of baseline Frobenius / spectral mass, while whole right-row or center-column replay would overshoot the required lift by `{_format_ratio(dense_replay_guard_report.companion_right_row_replay_multiple_vs_required_increment)}` / `{_format_ratio(dense_replay_guard_report.companion_center_column_replay_multiple_vs_required_increment)}`",
        f"- rerun admissibility still stays stricter than patch existence: only `{acceptance_contract_report.effective_fresh_reruns}` fresh random states remain, the bounded slice still supports floor ceiling `{_format_float(acceptance_contract_report.supported_floor_ceiling)}` against canonical floor `{_format_float(acceptance_contract_report.canonical_floor)}`, and the remaining shortfall `{_format_float(acceptance_contract_report.canonical_floor_shortfall)}` keeps Trigger 2 pinned to `{acceptance_contract_report.target_gate_status}` unless fresh estimator evidence lifts the binding floor witness",
        f"- current Trigger 2 implication: `{driver_signature}`; use this as a validation-only intake contract for future estimator-path work, not as a replacement for the live `bounded-right-center-execution-contract` token",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-fresh-estimator-evidence-intake-contract"
        ),
        policy_digest=object_flow_report.policy_digest,
        binding_design=object_flow_report.binding_design,
        window_label=object_flow_report.window_label,
        coverage_anchor_random_state=object_flow_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            object_flow_report.overshoot_companion_random_state
        ),
        object_flow_driver_signature=object_flow_report.driver_signature,
        dense_replay_driver_signature=dense_replay_guard_report.driver_signature,
        acceptance_driver_signature=acceptance_contract_report.driver_signature,
        source_diagonal_coordinate=object_flow_report.source_diagonal_coordinate,
        source_diagonal_basis_label=object_flow_report.source_diagonal_basis_label,
        shared_vf_entry_label=object_flow_report.shared_vf_entry_label,
        source_target_omega_diagonal_entry=(
            object_flow_report.source_target_omega_diagonal_entry
        ),
        source_required_omega_diagonal_increment=(
            object_flow_report.source_required_omega_diagonal_increment
        ),
        required_diagonal_vf_entry_lift=(
            object_flow_report.required_diagonal_vf_entry_lift
        ),
        validation_patch_increment=object_flow_report.validation_patch_increment,
        patch_frobenius_share_of_baseline=(
            dense_replay_guard_report.patch_frobenius_share_of_baseline
        ),
        patch_spectral_share_of_baseline=(
            dense_replay_guard_report.patch_spectral_share_of_baseline
        ),
        companion_right_row_replay_multiple_vs_required_increment=(
            dense_replay_guard_report.companion_right_row_replay_multiple_vs_required_increment
        ),
        companion_center_column_replay_multiple_vs_required_increment=(
            dense_replay_guard_report.companion_center_column_replay_multiple_vs_required_increment
        ),
        effective_fresh_reruns=acceptance_contract_report.effective_fresh_reruns,
        supported_floor_ceiling=acceptance_contract_report.supported_floor_ceiling,
        canonical_floor=acceptance_contract_report.canonical_floor,
        canonical_floor_shortfall=(
            acceptance_contract_report.canonical_floor_shortfall
        ),
        target_gate_status=acceptance_contract_report.target_gate_status,
        required_evidence_conditions=required_evidence_conditions,
        intake_contract_holds=intake_contract_holds,
        driver_signature=driver_signature,
        canonical_fresh_estimator_evidence_intake_digest=canonical_digest,
    )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract_repo_side_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport
):
    gate_state = _load_feature_completion_trigger2_state()
    policy_digest = tuple(str(item) for item in gate_state["policy_digest"])
    binding_design = _parse_design_key(str(gate_state["runtime_evidence_binding_design"]))
    effective_fresh_reruns = int(gate_state["runtime_evidence_effective_fresh_reruns"])
    target_gate_status = str(gate_state["target_gate_status"])
    driver_signature = str(
        gate_state["runtime_evidence_fresh_estimator_evidence_driver"]
    )

    if policy_digest != _CANONICAL_POLICY_DIGEST:
        raise ValueError(
            "repo-side fresh estimator evidence intake policy digest drifted from canonical bounded policy"
        )
    if binding_design != _BINDING_DESIGN:
        raise ValueError(
            "repo-side fresh estimator evidence intake binding design drifted from canonical bounded slice"
        )
    if target_gate_status != "trigger2-partial-widening-only":
        raise ValueError(
            "repo-side fresh estimator evidence intake target gate drifted from canonical bounded target"
        )
    if driver_signature != "bounded-entry-patch-fresh-estimator-evidence-intake-contract":
        raise ValueError(
            "repo-side fresh estimator evidence intake driver drifted from canonical validation-only contract"
        )

    canonical_floor_shortfall = _CANONICAL_FLOOR - _SUPPORTED_FLOOR_CEILING
    canonical_digest = (
        f"- bounded estimator evidence still has to enter through the same source-backed object flow for `{_format_design_key(binding_design)}` on `{_WINDOW_LABEL}`: coordinate `{_SOURCE_DIAGONAL_COORDINATE} = {_SOURCE_DIAGONAL_BASIS_LABEL}` must lift `omega_f_hat[2,2]` by `{_format_signed_float(_SOURCE_REQUIRED_OMEGA_DIAGONAL_INCREMENT)}` up to `{_format_float(_SOURCE_TARGET_OMEGA_DIAGONAL_ENTRY)}`, then route through `{_SHARED_VF_ENTRY_LABEL}` with required diagonal lift `{_format_signed_float(_REQUIRED_DIAGONAL_VF_ENTRY_LIFT)}` before it can explain the bounded raw covariance patch `{_format_signed_float(_VALIDATION_PATCH_INCREMENT)}`",
        f"- the same evidence remains invalid if it widens into dense replay: the bounded patch still uses only `{_format_percent(_PATCH_FROBENIUS_SHARE_OF_BASELINE)}` / `{_format_percent(_PATCH_SPECTRAL_SHARE_OF_BASELINE)}` of baseline Frobenius / spectral mass, while whole right-row or center-column replay would overshoot the required lift by `{_format_ratio(_COMPANION_RIGHT_ROW_REPLAY_MULTIPLE)}` / `{_format_ratio(_COMPANION_CENTER_COLUMN_REPLAY_MULTIPLE)}`",
        f"- rerun admissibility still stays stricter than patch existence: only `{effective_fresh_reruns}` fresh random states remain, the bounded slice still supports floor ceiling `{_format_float(_SUPPORTED_FLOOR_CEILING)}` against canonical floor `{_format_float(_CANONICAL_FLOOR)}`, and the remaining shortfall `{_format_float(canonical_floor_shortfall)}` keeps Trigger 2 pinned to `{target_gate_status}` unless fresh estimator evidence lifts the binding floor witness",
        "- current Trigger 2 implication: `bounded-entry-patch-fresh-estimator-evidence-intake-contract`; use this as a validation-only intake contract for future estimator-path work, not as a replacement for the live `bounded-right-center-execution-contract` token",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-fresh-estimator-evidence-intake-contract"
        ),
        policy_digest=policy_digest,
        binding_design=binding_design,
        window_label=_WINDOW_LABEL,
        coverage_anchor_random_state=_COVERAGE_ANCHOR_RANDOM_STATE,
        overshoot_companion_random_state=_OVERSHOOT_COMPANION_RANDOM_STATE,
        object_flow_driver_signature="bounded-first-sine-estimator-object-flow-contract",
        dense_replay_driver_signature="bounded-right-center-entry-patch-dense-replay-guard",
        acceptance_driver_signature="fresh-rerun-must-carry-floor-lift-evidence",
        source_diagonal_coordinate=_SOURCE_DIAGONAL_COORDINATE,
        source_diagonal_basis_label=_SOURCE_DIAGONAL_BASIS_LABEL,
        shared_vf_entry_label=_SHARED_VF_ENTRY_LABEL,
        source_target_omega_diagonal_entry=_SOURCE_TARGET_OMEGA_DIAGONAL_ENTRY,
        source_required_omega_diagonal_increment=_SOURCE_REQUIRED_OMEGA_DIAGONAL_INCREMENT,
        required_diagonal_vf_entry_lift=_REQUIRED_DIAGONAL_VF_ENTRY_LIFT,
        validation_patch_increment=_VALIDATION_PATCH_INCREMENT,
        patch_frobenius_share_of_baseline=_PATCH_FROBENIUS_SHARE_OF_BASELINE,
        patch_spectral_share_of_baseline=_PATCH_SPECTRAL_SHARE_OF_BASELINE,
        companion_right_row_replay_multiple_vs_required_increment=_COMPANION_RIGHT_ROW_REPLAY_MULTIPLE,
        companion_center_column_replay_multiple_vs_required_increment=_COMPANION_CENTER_COLUMN_REPLAY_MULTIPLE,
        effective_fresh_reruns=effective_fresh_reruns,
        supported_floor_ceiling=_SUPPORTED_FLOOR_CEILING,
        canonical_floor=_CANONICAL_FLOOR,
        canonical_floor_shortfall=canonical_floor_shortfall,
        target_gate_status=target_gate_status,
        required_evidence_conditions=_REQUIRED_EVIDENCE_CONDITIONS,
        intake_contract_holds=True,
        driver_signature=driver_signature,
        canonical_fresh_estimator_evidence_intake_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract_repo_side_report(
    )
