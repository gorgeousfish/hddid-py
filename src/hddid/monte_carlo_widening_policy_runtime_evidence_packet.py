from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .automation_state_view import load_top_level_automation_state_block
from .monte_carlo_widening_policy_acceptance_preview import (
    Phase7MonteCarloWideningPolicyAcceptancePreviewReport,
    build_phase7_monte_carlo_widening_policy_acceptance_preview_repo_side_report,
    run_phase7_monte_carlo_widening_policy_acceptance_preview,
)
from .monte_carlo_widening_policy_binding_design_rerun_capacity_probe import (
    Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport,
    run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe,
)
from .monte_carlo_widening_policy_floor_witness_quota_probe import (
    Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport,
    build_phase7_monte_carlo_widening_policy_floor_witness_quota_probe_repo_side_report,
    run_phase7_monte_carlo_widening_policy_floor_witness_quota_probe,
)
from .monte_carlo_widening_policy_quality_risk_probe import (
    run_phase7_monte_carlo_widening_policy_quality_risk_probe,
)


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _format_design_keys(
    design_keys: tuple[tuple[str, int, int], ...],
) -> str:
    if not design_keys:
        return "none"
    return ", ".join(f"{dgp}/{n_obs}/{p}" for dgp, n_obs, p in design_keys)


def _floats_match(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-12


def _coerce_nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not boolean")
    if not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _coerce_positive_integer(name: str, value: int) -> int:
    number = _coerce_nonnegative_integer(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _coerce_probability(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _validate_witness_packet_counts(
    *,
    covered: int,
    required: int,
    total: int,
    additional_needed: int,
    supported_floor_ceiling: float,
    canonical_floor: float,
    canonical_floor_shortfall: float,
    label: str,
) -> None:
    if covered > required:
        raise ValueError(f"{label} covered witnesses cannot exceed required witnesses")
    if required > total:
        raise ValueError(f"{label} required witnesses cannot exceed total witnesses")
    expected_additional = required - covered
    if additional_needed != expected_additional:
        raise ValueError(
            f"{label} additional witness gap must equal required minus covered witnesses"
        )
    expected_supported_floor = covered / total
    if not _floats_match(supported_floor_ceiling, expected_supported_floor):
        raise ValueError(
            f"{label} supported floor ceiling must equal covered divided by total witnesses"
        )
    expected_shortfall = max(canonical_floor - supported_floor_ceiling, 0.0)
    if not _floats_match(canonical_floor_shortfall, expected_shortfall):
        raise ValueError(
            f"{label} canonical floor shortfall must equal max(canonical floor minus supported floor ceiling, 0)"
        )


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOMATION_STATE_PATH = _REPO_ROOT / "Docs" / "automation" / "automation-state.yaml"
_STABLE_PARTIAL_DESIGNS = (
    ("DGP1", 500, 50),
    ("DGP2", 500, 50),
)
_BLOCKED_FULL_MATRIX_DESIGNS = (
    ("DGP1", 200, 500),
    ("DGP2", 200, 500),
)
_BINDING_DESIGN = ("DGP2", 500, 50)
_BINDING_RANDOM_STATES = (303,)
_WINDOW_LABEL = "near_zero_grid"
_COVERED_POINTWISE_WITNESSES = 7
_REQUIRED_POINTWISE_WITNESSES = 8
_TOTAL_POINTWISE_WITNESSES = 9
_CANONICAL_FLOOR = 0.85
_SUPPORTED_FLOOR_CEILING = 7.0 / 9.0
_RUNTIME_WITNESS_PATH = (
    "omega_f_hat[2,2]",
    "v_f_hat[2,2]",
    "covariance(0.25, 0.15)",
)
_FRESH_ESTIMATOR_EVIDENCE_HELPER = (
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract()"
)
_FRESH_ESTIMATOR_EVIDENCE_DRIVER = (
    "bounded-entry-patch-fresh-estimator-evidence-intake-contract"
)
_FRESH_ESTIMATOR_RERUN_HELPER = (
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract()"
)
_FRESH_ESTIMATOR_RERUN_DRIVER = "fresh-estimator-rerun-intake-contract-disciplined"
_RERUN_CAPACITY_HELPER = (
    "run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe()"
)
_RERUN_CAPACITY_NOTE = (
    "Docs/research/phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe.md"
)
_RERUN_CAPACITY_LIMITING_BUDGET = "random_state_budget"
_RERUN_CAPACITY_IMPLICATION = "spend-remaining-fresh-reruns-on-binding-design"
_CURRENT_IMPLICATION = "bounded-right-center-execution-contract"
_DRIVER_SIGNATURE = "trigger2-runtime-evidence-packet-open"
_ADMISSION_HELPER = (
    "run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate(...)"
)
_ADMISSION_NOTE = (
    "Docs/research/phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate.md"
)
_ADMISSION_STATUS = "trigger2-runtime-evidence-packet-open"
_ADMISSION_CANDIDATE_STATUS = "runtime-evidence-rejected"
_ADMISSION_CONTRACT_STATUS = "runtime-witness-contract-rejected"
_QUALITY_RISK_STATUS = "quality-risk-still-open"
_QUALITY_RISK_DRIVER = "rmse-outpaces-average-se"
_MONTE_CARLO_VALIDATION_READY_STATUS = "blocked"
_MONTE_CARLO_VALIDATION_READY_BLOCKER = "quality-risk-keeps-trigger2-bounded"
_REQUIRED_EVIDENCE_CONDITIONS = (
    "keep `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` as the bounded object flow",
    "reject whole-row / whole-column / dense covariance replay",
    "treat fresh reruns as informative only when they lift the binding floor witness",
)
_LIVE_PACKET_REPO_SIDE_FALLBACK_MESSAGES = (
    "shared Trigger 2 binding design",
    "binding design drifted from gate state",
    "binding random states drifted from gate state",
    "quota gap drifted from gate state",
    "fresh rerun budget drifted from gate state",
)


def _floor_quota_state_matches_report(
    report: "Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport",
    floor_witness_quota: Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport,
) -> bool:
    return (
        report.binding_design == floor_witness_quota.binding_design
        and report.binding_random_states == floor_witness_quota.binding_random_states
        and report.covered_pointwise_witnesses
        == floor_witness_quota.covered_pointwise_witnesses
        and report.required_covered_pointwise_witnesses
        == floor_witness_quota.required_covered_pointwise_witnesses
        and report.total_pointwise_witnesses
        == floor_witness_quota.total_pointwise_witnesses
        and report.additional_pointwise_witnesses_needed
        == floor_witness_quota.additional_pointwise_witnesses_needed
    )


def _with_live_quality_risk(
    report: "Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport",
    quality_risk_report: object,
) -> "Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport":
    return replace(
        report,
        post_admission_quality_risk_driver=str(
            getattr(quality_risk_report, "binding_driver")
        ),
        post_admission_quality_risk_binding_design=tuple(
            getattr(quality_risk_report, "binding_design")
        ),
    )


def _load_feature_completion_trigger2_state() -> dict[str, object]:
    feature_completion_gate = load_top_level_automation_state_block(
        _AUTOMATION_STATE_PATH,
        "feature_completion_gate",
    )
    return feature_completion_gate["checks"]["monte_carlo_validation_ready"]


def _should_fallback_to_repo_side_packet(exc: ValueError) -> bool:
    message = str(exc)
    return any(
        token in message for token in _LIVE_PACKET_REPO_SIDE_FALLBACK_MESSAGES
    )


def _parse_design_key(design_label: str) -> tuple[str, int, int]:
    dgp, n_obs, p = str(design_label).split("/")
    return (
        dgp,
        _coerce_positive_integer("design n_obs", int(n_obs)),
        _coerce_positive_integer("design p", int(p)),
    )


def _validate_quota_gap_bounds(
    additional_pointwise_witnesses_needed: int,
    required_covered_pointwise_witnesses: int,
) -> None:
    gap = _coerce_nonnegative_integer(
        "additional_pointwise_witnesses_needed",
        additional_pointwise_witnesses_needed,
    )
    required = _coerce_nonnegative_integer(
        "required_covered_pointwise_witnesses",
        required_covered_pointwise_witnesses,
    )
    if gap > required:
        raise ValueError(
            "runtime evidence packet quota gap drifted outside canonical witness bounds"
        )


def _load_runtime_evidence_packet_boundary_state() -> dict[str, object]:
    gate_state = dict(_load_feature_completion_trigger2_state())
    required_tokens = {
        "runtime_evidence_admission_helper": _ADMISSION_HELPER,
        "runtime_evidence_admission_note": _ADMISSION_NOTE,
        "runtime_evidence_admission_status": _ADMISSION_STATUS,
        "runtime_evidence_admission_candidate_status": _ADMISSION_CANDIDATE_STATUS,
        "quality_risk_driver": _QUALITY_RISK_DRIVER,
        "quality_risk_binding_driver": _QUALITY_RISK_DRIVER,
        "blocker": _MONTE_CARLO_VALIDATION_READY_BLOCKER,
        "status": _MONTE_CARLO_VALIDATION_READY_STATUS,
    }
    for key, expected in required_tokens.items():
        if key not in gate_state:
            gate_state[key] = expected
            continue
        if str(gate_state[key]) == "live-derived":
            gate_state[key] = expected
            continue
        if str(gate_state[key]) != expected:
            raise ValueError(
                f"runtime evidence packet boundary token `{key}` drifted from gate state"
            )
    binding_random_state = _coerce_nonnegative_integer(
        "runtime_evidence_admission_binding_random_state",
        gate_state.get(
            "runtime_evidence_admission_binding_random_state",
            _BINDING_RANDOM_STATES[0],
        )
    )
    if binding_random_state != _BINDING_RANDOM_STATES[0]:
        raise ValueError(
            "runtime evidence packet admission binding random state drifted from the canonical product blocker"
        )
    gate_state["runtime_evidence_admission_binding_random_state"] = binding_random_state
    if "runtime_evidence_admission_contract_status" not in gate_state:
        gate_state["runtime_evidence_admission_contract_status"] = (
            _ADMISSION_CONTRACT_STATUS
        )
    gate_state.setdefault("runtime_evidence_admission_fresh_reruns_after", 5)
    return gate_state


def _design_label(design_key: tuple[str, int, int]) -> str:
    return f"{design_key[0]}/{design_key[1]}/{design_key[2]}"


def _reject_explicit_gate_state_drift(gate_state: dict[str, object]) -> None:
    checks = (
        (
            "runtime_evidence_driver",
            _DRIVER_SIGNATURE,
            "runtime evidence packet driver signature drifted from gate state",
        ),
        (
            "runtime_evidence_fresh_estimator_evidence_helper",
            _FRESH_ESTIMATOR_EVIDENCE_HELPER,
            "runtime evidence packet fresh estimator evidence helper drifted from gate state",
        ),
        (
            "runtime_evidence_fresh_estimator_rerun_helper",
            _FRESH_ESTIMATOR_RERUN_HELPER,
            "runtime evidence packet fresh estimator rerun helper drifted from gate state",
        ),
        (
            "runtime_evidence_binding_design_rerun_capacity_helper",
            _RERUN_CAPACITY_HELPER,
            "runtime evidence packet rerun capacity helper drifted from gate state",
        ),
        (
            "runtime_evidence_binding_design_rerun_capacity_note",
            _RERUN_CAPACITY_NOTE,
            "runtime evidence packet rerun capacity note drifted from gate state",
        ),
        (
            "runtime_evidence_binding_design_rerun_capacity_limiting_budget",
            _RERUN_CAPACITY_LIMITING_BUDGET,
            "runtime evidence packet rerun capacity limiting budget drifted from gate state",
        ),
    )
    for key, expected, message in checks:
        if key in gate_state and str(gate_state[key]) != expected:
            raise ValueError(message)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport:
    stage_label: str
    live_entry: str
    route_label: str
    current_gate_status: str
    accepted_feature_bundle: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    binding_random_states: tuple[int, ...]
    window_label: str
    covered_pointwise_witnesses: int
    required_covered_pointwise_witnesses: int
    additional_pointwise_witnesses_needed: int
    total_pointwise_witnesses: int
    effective_fresh_reruns: int
    canonical_floor: float
    supported_floor_ceiling: float
    canonical_floor_shortfall: float
    floor_witness_quota_helper: str
    floor_witness_quota_note: str
    fresh_estimator_evidence_intake_helper: str
    fresh_estimator_evidence_intake_driver: str
    fresh_estimator_rerun_intake_helper: str
    fresh_estimator_rerun_intake_driver: str
    binding_design_rerun_capacity_helper: str
    binding_design_rerun_capacity_note: str
    binding_design_rerun_capacity_limiting_budget: str
    binding_design_rerun_capacity_implication: str
    runtime_witness_path: tuple[str, ...]
    required_evidence_conditions: tuple[str, ...]
    driver_signature: str
    current_implication: str
    canonical_runtime_evidence_digest: tuple[str, ...]
    runtime_evidence_admission_helper: str = _ADMISSION_HELPER
    runtime_evidence_admission_note: str = _ADMISSION_NOTE
    runtime_evidence_admission_status: str = _ADMISSION_STATUS
    runtime_evidence_admission_candidate_status: str = _ADMISSION_CANDIDATE_STATUS
    runtime_evidence_admission_contract_status: str = _ADMISSION_CONTRACT_STATUS
    runtime_evidence_admission_binding_random_state: int = 303
    runtime_evidence_admission_covered_pointwise_witnesses_before: int = 7
    runtime_evidence_admission_covered_pointwise_witnesses_after: int = 8
    runtime_evidence_admission_required_covered_pointwise_witnesses: int = 8
    runtime_evidence_admission_remaining_quota_gap_before: int = 1
    runtime_evidence_admission_remaining_quota_gap_after: int = 0
    runtime_evidence_admission_fresh_reruns_before: int = 5
    runtime_evidence_admission_fresh_reruns_after: int = 4
    runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before: int = 1
    runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after: int = 2
    runtime_evidence_admission_seed_local_total_pointwise_witnesses: int = 3
    runtime_evidence_admission_quota_closure_margin: int = 1
    post_admission_quality_risk_status: str = _QUALITY_RISK_STATUS
    post_admission_quality_risk_driver: str = _QUALITY_RISK_DRIVER
    post_admission_quality_risk_binding_design: tuple[str, int, int] = _BINDING_DESIGN
    monte_carlo_validation_ready_status: str = _MONTE_CARLO_VALIDATION_READY_STATUS
    monte_carlo_validation_ready_blocker: str = _MONTE_CARLO_VALIDATION_READY_BLOCKER

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.live_entry = str(self.live_entry).strip()
        self.route_label = str(self.route_label).strip()
        self.current_gate_status = str(self.current_gate_status).strip()
        self.accepted_feature_bundle = str(self.accepted_feature_bundle).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            _coerce_positive_integer("binding_design n_obs", self.binding_design[1]),
            _coerce_positive_integer("binding_design p", self.binding_design[2]),
        )
        self.binding_random_states = tuple(
            _coerce_nonnegative_integer("binding_random_states", value)
            for value in self.binding_random_states
        )
        if not self.binding_random_states:
            raise ValueError("binding_random_states must not be empty")
        self.window_label = str(self.window_label).strip()
        self.covered_pointwise_witnesses = _coerce_nonnegative_integer(
            "covered_pointwise_witnesses", self.covered_pointwise_witnesses
        )
        self.required_covered_pointwise_witnesses = _coerce_nonnegative_integer(
            "required_covered_pointwise_witnesses",
            self.required_covered_pointwise_witnesses,
        )
        self.additional_pointwise_witnesses_needed = _coerce_nonnegative_integer(
            "additional_pointwise_witnesses_needed",
            self.additional_pointwise_witnesses_needed,
        )
        self.total_pointwise_witnesses = _coerce_positive_integer(
            "total_pointwise_witnesses", self.total_pointwise_witnesses
        )
        self.effective_fresh_reruns = _coerce_nonnegative_integer(
            "effective_fresh_reruns", self.effective_fresh_reruns
        )
        self.canonical_floor = _coerce_probability(
            "canonical_floor", self.canonical_floor
        )
        self.supported_floor_ceiling = _coerce_probability(
            "supported_floor_ceiling", self.supported_floor_ceiling
        )
        self.canonical_floor_shortfall = _coerce_probability(
            "canonical_floor_shortfall", self.canonical_floor_shortfall
        )
        self.floor_witness_quota_helper = str(self.floor_witness_quota_helper).strip()
        self.floor_witness_quota_note = str(self.floor_witness_quota_note).strip()
        self.fresh_estimator_evidence_intake_helper = str(
            self.fresh_estimator_evidence_intake_helper
        ).strip()
        self.fresh_estimator_evidence_intake_driver = str(
            self.fresh_estimator_evidence_intake_driver
        ).strip()
        self.fresh_estimator_rerun_intake_helper = str(
            self.fresh_estimator_rerun_intake_helper
        ).strip()
        self.fresh_estimator_rerun_intake_driver = str(
            self.fresh_estimator_rerun_intake_driver
        ).strip()
        self.binding_design_rerun_capacity_helper = str(
            self.binding_design_rerun_capacity_helper
        ).strip()
        self.binding_design_rerun_capacity_note = str(
            self.binding_design_rerun_capacity_note
        ).strip()
        self.binding_design_rerun_capacity_limiting_budget = str(
            self.binding_design_rerun_capacity_limiting_budget
        ).strip()
        self.binding_design_rerun_capacity_implication = str(
            self.binding_design_rerun_capacity_implication
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.required_evidence_conditions = tuple(
            str(item).strip() for item in self.required_evidence_conditions
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.current_implication = str(self.current_implication).strip()
        self.canonical_runtime_evidence_digest = tuple(
            str(line).rstrip() for line in self.canonical_runtime_evidence_digest
        )
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
        self.runtime_evidence_admission_binding_random_state = _coerce_nonnegative_integer(
            "runtime_evidence_admission_binding_random_state",
            self.runtime_evidence_admission_binding_random_state
        )
        self.runtime_evidence_admission_covered_pointwise_witnesses_before = _coerce_nonnegative_integer(
            "runtime_evidence_admission_covered_pointwise_witnesses_before",
            self.runtime_evidence_admission_covered_pointwise_witnesses_before
        )
        self.runtime_evidence_admission_covered_pointwise_witnesses_after = _coerce_nonnegative_integer(
            "runtime_evidence_admission_covered_pointwise_witnesses_after",
            self.runtime_evidence_admission_covered_pointwise_witnesses_after
        )
        self.runtime_evidence_admission_required_covered_pointwise_witnesses = _coerce_nonnegative_integer(
            "runtime_evidence_admission_required_covered_pointwise_witnesses",
            self.runtime_evidence_admission_required_covered_pointwise_witnesses
        )
        self.runtime_evidence_admission_remaining_quota_gap_before = _coerce_nonnegative_integer(
            "runtime_evidence_admission_remaining_quota_gap_before",
            self.runtime_evidence_admission_remaining_quota_gap_before
        )
        self.runtime_evidence_admission_remaining_quota_gap_after = _coerce_nonnegative_integer(
            "runtime_evidence_admission_remaining_quota_gap_after",
            self.runtime_evidence_admission_remaining_quota_gap_after
        )
        self.runtime_evidence_admission_fresh_reruns_before = _coerce_nonnegative_integer(
            "runtime_evidence_admission_fresh_reruns_before",
            self.runtime_evidence_admission_fresh_reruns_before
        )
        self.runtime_evidence_admission_fresh_reruns_after = _coerce_nonnegative_integer(
            "runtime_evidence_admission_fresh_reruns_after",
            self.runtime_evidence_admission_fresh_reruns_after
        )
        self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before = _coerce_nonnegative_integer(
            "runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before",
            self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before
        )
        self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after = _coerce_nonnegative_integer(
            "runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after",
            self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after
        )
        self.runtime_evidence_admission_seed_local_total_pointwise_witnesses = _coerce_positive_integer(
            "runtime_evidence_admission_seed_local_total_pointwise_witnesses",
            self.runtime_evidence_admission_seed_local_total_pointwise_witnesses
        )
        self.runtime_evidence_admission_quota_closure_margin = _coerce_nonnegative_integer(
            "runtime_evidence_admission_quota_closure_margin",
            self.runtime_evidence_admission_quota_closure_margin
        )
        self.post_admission_quality_risk_status = str(
            self.post_admission_quality_risk_status
        ).strip()
        self.post_admission_quality_risk_driver = str(
            self.post_admission_quality_risk_driver
        ).strip()
        self.post_admission_quality_risk_binding_design = (
            str(self.post_admission_quality_risk_binding_design[0]).strip().upper(),
            _coerce_positive_integer(
                "post_admission_quality_risk_binding_design n_obs",
                self.post_admission_quality_risk_binding_design[1],
            ),
            _coerce_positive_integer(
                "post_admission_quality_risk_binding_design p",
                self.post_admission_quality_risk_binding_design[2],
            ),
        )
        self.monte_carlo_validation_ready_status = str(
            self.monte_carlo_validation_ready_status
        ).strip()
        self.monte_carlo_validation_ready_blocker = str(
            self.monte_carlo_validation_ready_blocker
        ).strip()
        self._validate_runtime_witness_counts()
        self._validate_admission_witness_counts()

    def _validate_runtime_witness_counts(self) -> None:
        _validate_witness_packet_counts(
            covered=self.covered_pointwise_witnesses,
            required=self.required_covered_pointwise_witnesses,
            total=self.total_pointwise_witnesses,
            additional_needed=self.additional_pointwise_witnesses_needed,
            supported_floor_ceiling=self.supported_floor_ceiling,
            canonical_floor=self.canonical_floor,
            canonical_floor_shortfall=self.canonical_floor_shortfall,
            label="runtime evidence packet",
        )

    def _validate_admission_witness_counts(self) -> None:
        required = self.runtime_evidence_admission_required_covered_pointwise_witnesses
        before = self.runtime_evidence_admission_covered_pointwise_witnesses_before
        after = self.runtime_evidence_admission_covered_pointwise_witnesses_after
        if before > after:
            raise ValueError(
                "runtime evidence admission covered witnesses cannot decrease"
            )
        if before > self.total_pointwise_witnesses or after > self.total_pointwise_witnesses:
            raise ValueError(
                "runtime evidence admission covered witnesses cannot exceed total witnesses"
            )
        if self.runtime_evidence_admission_remaining_quota_gap_before != max(
            required - before,
            0,
        ):
            raise ValueError(
                "runtime evidence admission quota gap before must equal max(required minus covered before, 0)"
            )
        if self.runtime_evidence_admission_remaining_quota_gap_after != max(
            required - after,
            0,
        ):
            raise ValueError(
                "runtime evidence admission quota gap after must equal max(required minus covered after, 0)"
            )
        if self.runtime_evidence_admission_quota_closure_margin != (
            self.runtime_evidence_admission_remaining_quota_gap_before
            - self.runtime_evidence_admission_remaining_quota_gap_after
        ):
            raise ValueError(
                "runtime evidence admission quota closure margin must equal the quota-gap reduction"
            )
        if (
            self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before
            > self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after
            or self.runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after
            > self.runtime_evidence_admission_seed_local_total_pointwise_witnesses
        ):
            raise ValueError(
                "runtime evidence admission seed-local witnesses must be monotone and bounded by the seed-local total"
            )
        if (
            self.runtime_evidence_admission_fresh_reruns_after
            > self.runtime_evidence_admission_fresh_reruns_before
        ):
            raise ValueError(
                "runtime evidence admission fresh reruns after cannot exceed fresh reruns before"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "live_entry": self.live_entry,
            "route_label": self.route_label,
            "current_gate_status": self.current_gate_status,
            "accepted_feature_bundle": self.accepted_feature_bundle,
            "accepted_basis_name": self.accepted_basis_name,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "runtime_evidence_binding_design": list(
                self.runtime_evidence_binding_design
            ),
            "binding_design_label": self.binding_design_label,
            "runtime_evidence_binding_design_label": (
                self.runtime_evidence_binding_design_label
            ),
            "binding_random_states": list(self.binding_random_states),
            "runtime_evidence_binding_random_states": list(
                self.runtime_evidence_binding_random_states
            ),
            "binding_random_state": self.binding_random_state,
            "window_label": self.window_label,
            "grid_label": self.grid_label,
            "covered_pointwise_witnesses": self.covered_pointwise_witnesses,
            "runtime_evidence_covered_pointwise_witnesses": (
                self.runtime_evidence_covered_pointwise_witnesses
            ),
            "current_witness_count": self.current_witness_count,
            "required_covered_pointwise_witnesses": self.required_covered_pointwise_witnesses,
            "runtime_evidence_required_covered_pointwise_witnesses": (
                self.runtime_evidence_required_covered_pointwise_witnesses
            ),
            "target_witness_count": self.target_witness_count,
            "additional_pointwise_witnesses_needed": self.additional_pointwise_witnesses_needed,
            "remaining_quota_gap": self.remaining_quota_gap,
            "runtime_evidence_required_quota_gap": self.remaining_quota_gap,
            "total_pointwise_witnesses": self.total_pointwise_witnesses,
            "runtime_evidence_total_pointwise_witnesses": (
                self.runtime_evidence_total_pointwise_witnesses
            ),
            "total_witness_count": self.total_witness_count,
            "witness_floor_current_label": self.witness_floor_current_label,
            "witness_floor_target_label": self.witness_floor_target_label,
            "witness_floor_transition_label": self.witness_floor_transition_label,
            "effective_fresh_reruns": self.effective_fresh_reruns,
            "fresh_reruns_remaining": self.fresh_reruns_remaining,
            "runtime_evidence_effective_fresh_reruns": self.fresh_reruns_remaining,
            "canonical_floor": self.canonical_floor,
            "supported_floor_ceiling": self.supported_floor_ceiling,
            "runtime_evidence_supported_floor_ceiling": (
                self.runtime_evidence_supported_floor_ceiling
            ),
            "canonical_floor_shortfall": self.canonical_floor_shortfall,
            "floor_witness_quota_helper": self.floor_witness_quota_helper,
            "floor_witness_quota_note": self.floor_witness_quota_note,
            "fresh_estimator_evidence_intake_helper": self.fresh_estimator_evidence_intake_helper,
            "fresh_estimator_evidence_intake_driver": self.fresh_estimator_evidence_intake_driver,
            "fresh_estimator_rerun_intake_helper": self.fresh_estimator_rerun_intake_helper,
            "fresh_estimator_rerun_intake_driver": self.fresh_estimator_rerun_intake_driver,
            "binding_design_rerun_capacity_helper": self.binding_design_rerun_capacity_helper,
            "binding_design_rerun_capacity_note": self.binding_design_rerun_capacity_note,
            "binding_design_rerun_capacity_limiting_budget": self.binding_design_rerun_capacity_limiting_budget,
            "binding_design_rerun_capacity_implication": self.binding_design_rerun_capacity_implication,
            "runtime_witness_path": list(self.runtime_witness_path),
            "required_evidence_conditions": list(self.required_evidence_conditions),
            "driver_signature": self.driver_signature,
            "runtime_evidence_driver": self.runtime_evidence_driver,
            "current_driver": self.current_driver,
            "runtime_evidence_current_driver": self.runtime_evidence_current_driver,
            "current_implication": self.current_implication,
            "canonical_runtime_evidence_digest": list(
                self.canonical_runtime_evidence_digest
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
            "runtime_evidence_admission_fresh_reruns_after": (
                self.runtime_evidence_admission_fresh_reruns_after
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
        }

    @property
    def accepted_basis_name(self) -> str:
        return self.accepted_feature_bundle

    @property
    def binding_design_label(self) -> str:
        return (
            f"{self.binding_design[0]}/{self.binding_design[1]}/{self.binding_design[2]}"
        )

    @property
    def runtime_evidence_binding_design(self) -> tuple[str, int, int]:
        return self.binding_design

    @property
    def runtime_evidence_binding_design_label(self) -> str:
        return self.binding_design_label

    @property
    def runtime_evidence_binding_random_states(self) -> tuple[int, ...]:
        return self.binding_random_states

    @property
    def current_driver(self) -> str:
        return self.driver_signature

    @property
    def runtime_evidence_driver(self) -> str:
        return self.driver_signature

    @property
    def runtime_evidence_current_driver(self) -> str:
        return self.current_driver

    @property
    def binding_random_state(self) -> int:
        if len(self.binding_random_states) != 1:
            raise ValueError(
                "runtime evidence packet requires a single binding random state"
            )
        return self.binding_random_states[0]

    @property
    def grid_label(self) -> str:
        return self.window_label

    @property
    def current_witness_count(self) -> int:
        return self.covered_pointwise_witnesses

    @property
    def runtime_evidence_covered_pointwise_witnesses(self) -> int:
        return self.covered_pointwise_witnesses

    @property
    def target_witness_count(self) -> int:
        return self.required_covered_pointwise_witnesses

    @property
    def runtime_evidence_required_covered_pointwise_witnesses(self) -> int:
        return self.required_covered_pointwise_witnesses

    @property
    def total_witness_count(self) -> int:
        return self.total_pointwise_witnesses

    @property
    def runtime_evidence_total_pointwise_witnesses(self) -> int:
        return self.total_pointwise_witnesses

    @property
    def remaining_quota_gap(self) -> int:
        return self.additional_pointwise_witnesses_needed

    @property
    def runtime_evidence_required_quota_gap(self) -> int:
        return self.remaining_quota_gap

    @property
    def fresh_reruns_remaining(self) -> int:
        return self.effective_fresh_reruns

    @property
    def runtime_evidence_effective_fresh_reruns(self) -> int:
        return self.fresh_reruns_remaining

    @property
    def runtime_evidence_supported_floor_ceiling(self) -> float:
        return self.supported_floor_ceiling

    @property
    def witness_floor_current_label(self) -> str:
        return f"{self.current_witness_count}/{self.total_witness_count}"

    @property
    def witness_floor_target_label(self) -> str:
        return f"{self.target_witness_count}/{self.total_witness_count}"

    @property
    def witness_floor_transition_label(self) -> str:
        return (
            f"{self.witness_floor_current_label} -> "
            f"{self.witness_floor_target_label}"
        )

    @property
    def runtime_evidence_admission_binding_design_label(self) -> str:
        return self.binding_design_label

    @property
    def runtime_evidence_admission_witness_floor_before_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_covered_pointwise_witnesses_before}/"
            f"{self.total_pointwise_witnesses}"
        )

    @property
    def runtime_evidence_admission_witness_floor_after_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_covered_pointwise_witnesses_after}/"
            f"{self.total_pointwise_witnesses}"
        )

    @property
    def runtime_evidence_admission_witness_floor_target_label(self) -> str:
        return (
            f"{self.runtime_evidence_admission_required_covered_pointwise_witnesses}/"
            f"{self.total_pointwise_witnesses}"
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


def build_phase7_monte_carlo_widening_policy_runtime_evidence_packet_report(
    acceptance_preview: Phase7MonteCarloWideningPolicyAcceptancePreviewReport,
    floor_witness_quota: Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport,
    fresh_estimator_rerun_contract,
    binding_design_rerun_capacity: (
        Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport
    ),
) -> Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport:
    if acceptance_preview.accepted_open_trigger_names != ("trigger2-runtime-evidence",):
        raise ValueError(
            "runtime evidence packet requires the accepted preview to open trigger2-runtime-evidence"
        )
    if acceptance_preview.policy_digest != floor_witness_quota.policy_digest:
        raise ValueError(
            "runtime evidence packet requires floor witness quota to share the canonical policy digest"
        )
    if acceptance_preview.policy_digest != fresh_estimator_rerun_contract.policy_digest:
        raise ValueError(
            "runtime evidence packet requires fresh estimator rerun contract to share the canonical policy digest"
        )
    if floor_witness_quota.binding_design != fresh_estimator_rerun_contract.binding_design:
        raise ValueError(
            "runtime evidence packet requires a shared Trigger 2 binding design"
        )
    if not _floats_match(
        floor_witness_quota.supported_floor_ceiling,
        fresh_estimator_rerun_contract.supported_floor_ceiling,
    ):
        raise ValueError(
            "runtime evidence packet requires shared floor ceiling evidence"
        )
    if not _floats_match(
        floor_witness_quota.coverage_floor,
        fresh_estimator_rerun_contract.canonical_floor,
    ):
        raise ValueError(
            "runtime evidence packet requires a shared canonical floor"
        )
    if (
        floor_witness_quota.additional_pointwise_witnesses_needed
        > fresh_estimator_rerun_contract.effective_fresh_reruns
    ):
        raise ValueError(
            "runtime evidence packet quota gap exceeds fresh rerun budget"
        )
    if acceptance_preview.policy_digest != binding_design_rerun_capacity.policy_digest:
        raise ValueError(
            "runtime evidence packet requires rerun capacity to share the canonical policy digest"
        )
    _validate_quota_gap_bounds(
        floor_witness_quota.additional_pointwise_witnesses_needed,
        floor_witness_quota.required_covered_pointwise_witnesses,
    )
    if floor_witness_quota.binding_design != binding_design_rerun_capacity.binding_design:
        raise ValueError(
            "runtime evidence packet requires rerun capacity to stay on the shared Trigger 2 binding design"
        )
    if (
        fresh_estimator_rerun_contract.effective_fresh_reruns
        != binding_design_rerun_capacity.effective_additional_reruns
    ):
        raise ValueError(
            "runtime evidence packet requires rerun capacity to agree on the fresh rerun budget"
        )
    if len(floor_witness_quota.binding_random_states) != 1:
        raise ValueError(
            "runtime evidence packet requires a single binding random state"
        )
    binding_random_state = floor_witness_quota.binding_random_states[0]
    binding_seed_summary = floor_witness_quota.seed_summary(binding_random_state)

    stable_partial_designs = (
        ("DGP1", 500, 50),
        ("DGP2", 500, 50),
    )
    blocked_full_matrix_designs = (
        ("DGP1", 200, 500),
        ("DGP2", 200, 500),
    )
    return Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport(
        stage_label="phase7-monte-carlo-widening-policy-runtime-evidence-packet",
        live_entry=acceptance_preview.current_recommended_bounded_loop,
        route_label=acceptance_preview.accepted_open_trigger_names[0],
        current_gate_status=acceptance_preview.accepted_runtime_gate_status,
        accepted_feature_bundle=acceptance_preview.accepted_recommended_feature_bundle,
        policy_digest=acceptance_preview.policy_digest,
        binding_design=floor_witness_quota.binding_design,
        binding_random_states=floor_witness_quota.binding_random_states,
        window_label=fresh_estimator_rerun_contract.window_label,
        covered_pointwise_witnesses=floor_witness_quota.covered_pointwise_witnesses,
        required_covered_pointwise_witnesses=floor_witness_quota.required_covered_pointwise_witnesses,
        additional_pointwise_witnesses_needed=floor_witness_quota.additional_pointwise_witnesses_needed,
        total_pointwise_witnesses=floor_witness_quota.total_pointwise_witnesses,
        effective_fresh_reruns=fresh_estimator_rerun_contract.effective_fresh_reruns,
        canonical_floor=floor_witness_quota.coverage_floor,
        supported_floor_ceiling=floor_witness_quota.supported_floor_ceiling,
        canonical_floor_shortfall=fresh_estimator_rerun_contract.canonical_floor_shortfall,
        floor_witness_quota_helper="run_phase7_monte_carlo_widening_policy_floor_witness_quota_probe()",
        floor_witness_quota_note="Docs/research/phase7_monte_carlo_widening_policy_floor_witness_quota_probe.md",
        fresh_estimator_evidence_intake_helper=_FRESH_ESTIMATOR_EVIDENCE_HELPER,
        fresh_estimator_evidence_intake_driver=fresh_estimator_rerun_contract.intake_driver_signature,
        fresh_estimator_rerun_intake_helper=_FRESH_ESTIMATOR_RERUN_HELPER,
        fresh_estimator_rerun_intake_driver=fresh_estimator_rerun_contract.driver_signature,
        binding_design_rerun_capacity_helper="run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe()",
        binding_design_rerun_capacity_note="Docs/research/phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe.md",
        binding_design_rerun_capacity_limiting_budget=(
            binding_design_rerun_capacity.limiting_budget
        ),
        binding_design_rerun_capacity_implication=(
            binding_design_rerun_capacity.current_implication
        ),
        runtime_witness_path=fresh_estimator_rerun_contract.runtime_witness_path,
        required_evidence_conditions=fresh_estimator_rerun_contract.required_evidence_conditions,
        driver_signature=_DRIVER_SIGNATURE,
        current_implication=_CURRENT_IMPLICATION,
        canonical_runtime_evidence_digest=(
            "- accepted open trigger `trigger2-runtime-evidence` now resolves to a single bounded runtime-evidence packet on top of live `trigger2-policy-spec`: accepted feature bundle stays `trigger2-bounded-widening`, accepted gate stays `trigger2-partial-widening-only`, and the bounded slice still stays "
            f"`{_format_design_keys(stable_partial_designs)}` while full-matrix `{_format_design_keys(blocked_full_matrix_designs)}` remains blocked",
            f"- the binding design is still `{floor_witness_quota.binding_design[0]}/{floor_witness_quota.binding_design[1]}/{floor_witness_quota.binding_design[2]}`: current floor witness remains `{floor_witness_quota.covered_pointwise_witnesses}/{floor_witness_quota.total_pointwise_witnesses}` against canonical `{floor_witness_quota.required_covered_pointwise_witnesses}/{floor_witness_quota.total_pointwise_witnesses}`, so the remaining quota gap is exactly `{floor_witness_quota.additional_pointwise_witnesses_needed}` covered pointwise witness and it is localized to seed `{floor_witness_quota.binding_random_states[0]}`",
            f"- binding-design rerun capacity is already machine-readable via `run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe()` and `Docs/research/phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe.md`: the current bounded packet is `{binding_design_rerun_capacity.limiting_budget.replace('_', '-')}` and keeps `{binding_design_rerun_capacity.current_implication}` on the same seed-`{binding_random_state}` lane",
            f"- fresh rerun budget remains finite and disciplined: only `{fresh_estimator_rerun_contract.effective_fresh_reruns}` fresh reruns remain, and they are informative only if they preserve `{' -> '.join(fresh_estimator_rerun_contract.runtime_witness_path)}`, reject dense replay, and lift the binding floor witness rather than replaying the validation-only patch object",
            f"- current Trigger 2 implication stays `bounded-right-center-execution-contract`: the open runtime trigger still resolves to the bounded seed-`{binding_random_state}` packet, but live implementation handoff remains the bounded right-center execution lane on `near_zero_grid` rather than broader widening, tolerance relaxation, or same-seed companion detours",
        ),
        runtime_evidence_admission_binding_random_state=binding_random_state,
        runtime_evidence_admission_covered_pointwise_witnesses_before=(
            floor_witness_quota.covered_pointwise_witnesses
        ),
        runtime_evidence_admission_covered_pointwise_witnesses_after=(
            floor_witness_quota.required_covered_pointwise_witnesses
        ),
        runtime_evidence_admission_required_covered_pointwise_witnesses=(
            floor_witness_quota.required_covered_pointwise_witnesses
        ),
        runtime_evidence_admission_remaining_quota_gap_before=(
            floor_witness_quota.additional_pointwise_witnesses_needed
        ),
        runtime_evidence_admission_remaining_quota_gap_after=0,
        runtime_evidence_admission_fresh_reruns_before=(
            fresh_estimator_rerun_contract.effective_fresh_reruns
        ),
        runtime_evidence_admission_fresh_reruns_after=(
            max(
                fresh_estimator_rerun_contract.effective_fresh_reruns
                - floor_witness_quota.additional_pointwise_witnesses_needed,
                0,
            )
        ),
        runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before=(
            binding_seed_summary.covered_pointwise_witnesses
        ),
        runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after=(
            min(
                binding_seed_summary.covered_pointwise_witnesses
                + floor_witness_quota.additional_pointwise_witnesses_needed,
                binding_seed_summary.total_pointwise_witnesses,
            )
        ),
        runtime_evidence_admission_seed_local_total_pointwise_witnesses=(
            binding_seed_summary.total_pointwise_witnesses
        ),
        runtime_evidence_admission_quota_closure_margin=(
            floor_witness_quota.additional_pointwise_witnesses_needed
        ),
    )


def build_phase7_monte_carlo_widening_policy_runtime_evidence_packet_repo_side_report(
    acceptance_preview: Phase7MonteCarloWideningPolicyAcceptancePreviewReport,
) -> Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport:
    gate_state = _load_runtime_evidence_packet_boundary_state()
    accepted_feature_bundle = str(
        gate_state.get(
            "accepted_feature_bundle",
            acceptance_preview.accepted_recommended_feature_bundle,
        )
    )
    current_gate_status = str(
        gate_state.get("target_gate_status", acceptance_preview.accepted_runtime_gate_status)
    )
    policy_digest = tuple(
        str(item)
        for item in gate_state.get("policy_digest", acceptance_preview.policy_digest)
    )
    live_entry = str(
        gate_state.get("route", acceptance_preview.current_recommended_bounded_loop)
    )
    accepted_open_trigger_names = gate_state.get(
        "accepted_open_trigger_names",
        acceptance_preview.accepted_open_trigger_names,
    )
    route_label = str(tuple(accepted_open_trigger_names)[0])
    binding_design = _parse_design_key(
        str(gate_state.get("runtime_evidence_binding_design", _design_label(_BINDING_DESIGN)))
    )
    quality_risk_binding_design = _parse_design_key(
        str(gate_state.get("quality_risk_binding_design", _design_label(binding_design)))
    )
    binding_random_states = tuple(
        _coerce_nonnegative_integer("runtime_evidence_binding_random_states", seed)
        for seed in gate_state.get(
            "runtime_evidence_binding_random_states",
            _BINDING_RANDOM_STATES,
        )
    )
    effective_fresh_reruns = _coerce_nonnegative_integer(
        "runtime_evidence_effective_fresh_reruns",
        gate_state.get("runtime_evidence_effective_fresh_reruns", 5),
    )
    additional_pointwise_witnesses_needed = _coerce_nonnegative_integer(
        "runtime_evidence_required_quota_gap",
        gate_state.get("runtime_evidence_required_quota_gap", 1),
    )
    required_covered_pointwise_witnesses = _REQUIRED_POINTWISE_WITNESSES
    total_pointwise_witnesses = _TOTAL_POINTWISE_WITNESSES
    covered_pointwise_witnesses = (
        required_covered_pointwise_witnesses - additional_pointwise_witnesses_needed
    )
    _validate_quota_gap_bounds(
        additional_pointwise_witnesses_needed,
        required_covered_pointwise_witnesses,
    )
    supported_floor_ceiling = (
        covered_pointwise_witnesses / total_pointwise_witnesses
    )
    canonical_floor_shortfall = max(
        round(_CANONICAL_FLOOR - supported_floor_ceiling, 12),
        0.0,
    )

    if acceptance_preview.accepted_open_trigger_names != (route_label,):
        raise ValueError(
            "repo-side runtime evidence packet requires the accepted preview to keep the canonical open trigger"
        )
    if acceptance_preview.accepted_recommended_feature_bundle != accepted_feature_bundle:
        raise ValueError(
            "repo-side runtime evidence packet feature bundle drifted from accepted preview"
        )
    if acceptance_preview.accepted_runtime_gate_status != current_gate_status:
        raise ValueError(
            "repo-side runtime evidence packet gate status drifted from accepted preview"
        )
    if acceptance_preview.current_recommended_bounded_loop != live_entry:
        raise ValueError(
            "repo-side runtime evidence packet live entry drifted from accepted preview"
        )
    if acceptance_preview.policy_digest != policy_digest:
        raise ValueError(
            "repo-side runtime evidence packet policy digest drifted from accepted preview"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_fresh_estimator_evidence_helper",
                _FRESH_ESTIMATOR_EVIDENCE_HELPER,
            )
        )
        != _FRESH_ESTIMATOR_EVIDENCE_HELPER
    ):
        raise ValueError(
            "fresh estimator evidence helper drifted from canonical runtime evidence packet contract"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_fresh_estimator_evidence_driver",
                _FRESH_ESTIMATOR_EVIDENCE_DRIVER,
            )
        )
        != _FRESH_ESTIMATOR_EVIDENCE_DRIVER
    ):
        raise ValueError(
            "fresh estimator evidence driver drifted from canonical runtime evidence packet contract"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_fresh_estimator_rerun_helper",
                _FRESH_ESTIMATOR_RERUN_HELPER,
            )
        )
        != _FRESH_ESTIMATOR_RERUN_HELPER
    ):
        raise ValueError(
            "fresh estimator rerun helper drifted from canonical runtime evidence packet contract"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_fresh_estimator_rerun_driver",
                _FRESH_ESTIMATOR_RERUN_DRIVER,
            )
        )
        != _FRESH_ESTIMATOR_RERUN_DRIVER
    ):
        raise ValueError(
            "fresh estimator rerun driver drifted from canonical runtime evidence packet contract"
        )
    if (
        str(gate_state.get("runtime_evidence_driver", _DRIVER_SIGNATURE))
        != _DRIVER_SIGNATURE
    ):
        raise ValueError(
            "runtime evidence driver drifted from canonical runtime evidence packet contract"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_binding_design_rerun_capacity_helper",
                _RERUN_CAPACITY_HELPER,
            )
        )
        != _RERUN_CAPACITY_HELPER
    ):
        raise ValueError(
            "rerun capacity helper drifted from canonical runtime evidence packet contract"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_binding_design_rerun_capacity_note",
                _RERUN_CAPACITY_NOTE,
            )
        )
        != _RERUN_CAPACITY_NOTE
    ):
        raise ValueError(
            "rerun capacity note drifted from canonical runtime evidence packet contract"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_binding_design_rerun_capacity_limiting_budget",
                _RERUN_CAPACITY_LIMITING_BUDGET,
            )
        )
        != _RERUN_CAPACITY_LIMITING_BUDGET
    ):
        raise ValueError(
            "rerun capacity limiting budget drifted from canonical runtime evidence packet contract"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_binding_design_rerun_capacity_implication",
                _RERUN_CAPACITY_IMPLICATION,
            )
        )
        != _RERUN_CAPACITY_IMPLICATION
    ):
        raise ValueError(
            "rerun capacity implication drifted from canonical runtime evidence packet contract"
        )
    if (
        str(gate_state.get("runtime_evidence_current_implication", _CURRENT_IMPLICATION))
        != _CURRENT_IMPLICATION
    ):
        raise ValueError(
            "current implication drifted from canonical runtime evidence packet contract"
        )
    if len(binding_random_states) != 1:
        raise ValueError(
            "runtime evidence packet requires a single binding random state"
        )
    if binding_design != _BINDING_DESIGN or binding_random_states != _BINDING_RANDOM_STATES:
        raise ValueError(
            "runtime evidence packet binding lane drifted from the canonical product blocker"
        )
    if additional_pointwise_witnesses_needed > effective_fresh_reruns:
        raise ValueError(
            "runtime evidence packet quota gap exceeds fresh rerun budget"
        )

    return Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport(
        stage_label="phase7-monte-carlo-widening-policy-runtime-evidence-packet",
        live_entry=live_entry,
        route_label=route_label,
        current_gate_status=current_gate_status,
        accepted_feature_bundle=accepted_feature_bundle,
        policy_digest=policy_digest,
        binding_design=binding_design,
        binding_random_states=binding_random_states,
        window_label=_WINDOW_LABEL,
        covered_pointwise_witnesses=covered_pointwise_witnesses,
        required_covered_pointwise_witnesses=required_covered_pointwise_witnesses,
        additional_pointwise_witnesses_needed=additional_pointwise_witnesses_needed,
        total_pointwise_witnesses=total_pointwise_witnesses,
        effective_fresh_reruns=effective_fresh_reruns,
        canonical_floor=_CANONICAL_FLOOR,
        supported_floor_ceiling=supported_floor_ceiling,
        canonical_floor_shortfall=canonical_floor_shortfall,
        floor_witness_quota_helper="run_phase7_monte_carlo_widening_policy_floor_witness_quota_probe()",
        floor_witness_quota_note="Docs/research/phase7_monte_carlo_widening_policy_floor_witness_quota_probe.md",
        fresh_estimator_evidence_intake_helper=_FRESH_ESTIMATOR_EVIDENCE_HELPER,
        fresh_estimator_evidence_intake_driver=_FRESH_ESTIMATOR_EVIDENCE_DRIVER,
        fresh_estimator_rerun_intake_helper=_FRESH_ESTIMATOR_RERUN_HELPER,
        fresh_estimator_rerun_intake_driver=_FRESH_ESTIMATOR_RERUN_DRIVER,
        binding_design_rerun_capacity_helper=_RERUN_CAPACITY_HELPER,
        binding_design_rerun_capacity_note=_RERUN_CAPACITY_NOTE,
        binding_design_rerun_capacity_limiting_budget=_RERUN_CAPACITY_LIMITING_BUDGET,
        binding_design_rerun_capacity_implication=_RERUN_CAPACITY_IMPLICATION,
        runtime_witness_path=_RUNTIME_WITNESS_PATH,
        required_evidence_conditions=_REQUIRED_EVIDENCE_CONDITIONS,
        driver_signature=_DRIVER_SIGNATURE,
        current_implication=_CURRENT_IMPLICATION,
        canonical_runtime_evidence_digest=(
            "- accepted open trigger `trigger2-runtime-evidence` now resolves to a single bounded runtime-evidence packet on top of live `trigger2-policy-spec`: accepted feature bundle stays `trigger2-bounded-widening`, accepted gate stays `trigger2-partial-widening-only`, and the bounded slice still stays "
            f"`{_format_design_keys(_STABLE_PARTIAL_DESIGNS)}` while full-matrix `{_format_design_keys(_BLOCKED_FULL_MATRIX_DESIGNS)}` remains blocked",
            f"- the binding design is still `{binding_design[0]}/{binding_design[1]}/{binding_design[2]}`: current floor witness remains `{covered_pointwise_witnesses}/{total_pointwise_witnesses}` against canonical `{required_covered_pointwise_witnesses}/{total_pointwise_witnesses}`, so the remaining quota gap is exactly `{additional_pointwise_witnesses_needed}` covered pointwise witness and it is localized to seed `{binding_random_states[0]}`",
            f"- binding-design rerun capacity is already machine-readable via `run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe()` and `Docs/research/phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe.md`: the current bounded packet is `{_RERUN_CAPACITY_LIMITING_BUDGET.replace('_', '-')}` and keeps `spend-remaining-fresh-reruns-on-binding-design` on the same seed-`{binding_random_states[0]}` lane",
            f"- fresh rerun budget remains finite and disciplined: only `{effective_fresh_reruns}` fresh reruns remain, and they are informative only if they preserve `{' -> '.join(_RUNTIME_WITNESS_PATH)}`, reject dense replay, and lift the binding floor witness rather than replaying the validation-only patch object",
            f"- current Trigger 2 implication stays `bounded-right-center-execution-contract`: the open runtime trigger still resolves to the bounded seed-`{binding_random_states[0]}` packet, but live implementation handoff remains the bounded right-center execution lane on `near_zero_grid` rather than broader widening, tolerance relaxation, or same-seed companion detours",
        ),
        runtime_evidence_admission_helper=str(
            gate_state.get(
                "runtime_evidence_admission_helper",
                _ADMISSION_HELPER,
            )
        ),
        runtime_evidence_admission_note=str(
            gate_state.get("runtime_evidence_admission_note", _ADMISSION_NOTE)
        ),
        runtime_evidence_admission_status=str(
            gate_state.get("runtime_evidence_admission_status", _ADMISSION_STATUS)
        ),
        runtime_evidence_admission_candidate_status=str(
            gate_state.get(
                "runtime_evidence_admission_candidate_status",
                _ADMISSION_CANDIDATE_STATUS,
            )
        ),
        runtime_evidence_admission_contract_status=str(
            gate_state.get(
                "runtime_evidence_admission_contract_status",
                _ADMISSION_CONTRACT_STATUS,
            )
        ),
        runtime_evidence_admission_binding_random_state=_coerce_nonnegative_integer(
            "runtime_evidence_admission_binding_random_state",
            gate_state.get(
                "runtime_evidence_admission_binding_random_state",
                _BINDING_RANDOM_STATES[0],
            )
        ),
        runtime_evidence_admission_covered_pointwise_witnesses_before=covered_pointwise_witnesses,
        runtime_evidence_admission_covered_pointwise_witnesses_after=(
            required_covered_pointwise_witnesses
        ),
        runtime_evidence_admission_required_covered_pointwise_witnesses=(
            required_covered_pointwise_witnesses
        ),
        runtime_evidence_admission_remaining_quota_gap_before=(
            additional_pointwise_witnesses_needed
        ),
        runtime_evidence_admission_remaining_quota_gap_after=0,
        runtime_evidence_admission_fresh_reruns_before=effective_fresh_reruns,
        runtime_evidence_admission_fresh_reruns_after=_coerce_nonnegative_integer(
            "runtime_evidence_admission_fresh_reruns_after",
            gate_state.get(
                "runtime_evidence_admission_fresh_reruns_after",
                effective_fresh_reruns,
            ),
        ),
        runtime_evidence_admission_seed_local_covered_pointwise_witnesses_before=1,
        runtime_evidence_admission_seed_local_covered_pointwise_witnesses_after=(
            1 + additional_pointwise_witnesses_needed
        ),
        runtime_evidence_admission_seed_local_total_pointwise_witnesses=3,
        runtime_evidence_admission_quota_closure_margin=(
            additional_pointwise_witnesses_needed
        ),
        post_admission_quality_risk_status=_QUALITY_RISK_STATUS,
        post_admission_quality_risk_driver=str(
            gate_state.get("quality_risk_driver", _QUALITY_RISK_DRIVER)
        ),
        post_admission_quality_risk_binding_design=quality_risk_binding_design,
        monte_carlo_validation_ready_status=str(
            gate_state.get("status", _MONTE_CARLO_VALIDATION_READY_STATUS)
        ),
        monte_carlo_validation_ready_blocker=str(
            gate_state.get("blocker", _MONTE_CARLO_VALIDATION_READY_BLOCKER)
        ),
    )


def run_phase7_monte_carlo_widening_policy_runtime_evidence_packet(
    repo_root: str | Path,
) -> Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport:
    root = _coerce_repo_root(repo_root)
    gate_state = _load_feature_completion_trigger2_state()
    _reject_explicit_gate_state_drift(gate_state)
    floor_witness_quota = (
        build_phase7_monte_carlo_widening_policy_floor_witness_quota_probe_repo_side_report()
    )
    quality_risk_report = run_phase7_monte_carlo_widening_policy_quality_risk_probe(
        root
    )
    binding_design_rerun_capacity = None
    fresh_estimator_rerun_contract = None
    try:
        acceptance_preview = (
            build_phase7_monte_carlo_widening_policy_acceptance_preview_repo_side_report(
                root
            )
        )
        report = build_phase7_monte_carlo_widening_policy_runtime_evidence_packet_repo_side_report(
            acceptance_preview
        )
        if not _floor_quota_state_matches_report(report, floor_witness_quota):
            raise ValueError(
                "repo-side runtime evidence packet drifted from live floor witness quota"
            )
    except ValueError:
        acceptance_preview = run_phase7_monte_carlo_widening_policy_acceptance_preview(root)
        from .monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract import (
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract,
        )

        fresh_estimator_rerun_contract = (
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract()
        )
        binding_design_rerun_capacity = (
            run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe()
        )
        try:
            report = build_phase7_monte_carlo_widening_policy_runtime_evidence_packet_report(
                acceptance_preview=acceptance_preview,
                floor_witness_quota=floor_witness_quota,
                fresh_estimator_rerun_contract=fresh_estimator_rerun_contract,
                binding_design_rerun_capacity=binding_design_rerun_capacity,
            )
        except ValueError as exc:
            if not _should_fallback_to_repo_side_packet(exc):
                raise
            report = build_phase7_monte_carlo_widening_policy_runtime_evidence_packet_repo_side_report(
                acceptance_preview
            )
            if not _floor_quota_state_matches_report(report, floor_witness_quota):
                raise ValueError(
                    "repo-side runtime evidence packet drifted from live floor witness quota"
                )

    report = _with_live_quality_risk(report, quality_risk_report)

    accepted_feature_bundle = str(
        gate_state.get(
            "accepted_feature_bundle",
            acceptance_preview.accepted_recommended_feature_bundle,
        )
    )
    current_gate_status = str(
        gate_state.get("target_gate_status", acceptance_preview.accepted_runtime_gate_status)
    )
    policy_digest = tuple(
        str(item)
        for item in gate_state.get("policy_digest", acceptance_preview.policy_digest)
    )
    live_entry = str(
        gate_state.get("route", acceptance_preview.current_recommended_bounded_loop)
    )
    accepted_open_trigger_names = gate_state.get(
        "accepted_open_trigger_names",
        acceptance_preview.accepted_open_trigger_names,
    )
    route_label = str(tuple(accepted_open_trigger_names)[0])
    binding_design = floor_witness_quota.binding_design
    binding_random_states = floor_witness_quota.binding_random_states
    effective_fresh_reruns = (
        fresh_estimator_rerun_contract.effective_fresh_reruns
        if fresh_estimator_rerun_contract is not None
        else _coerce_nonnegative_integer(
            "runtime_evidence_effective_fresh_reruns",
            gate_state.get("runtime_evidence_effective_fresh_reruns", 5),
        )
    )
    additional_pointwise_witnesses_needed = (
        floor_witness_quota.additional_pointwise_witnesses_needed
    )
    _validate_quota_gap_bounds(
        additional_pointwise_witnesses_needed,
        _REQUIRED_POINTWISE_WITNESSES,
    )
    if str(gate_state.get("runtime_evidence_driver", _DRIVER_SIGNATURE)) != _DRIVER_SIGNATURE:
        raise ValueError(
            "runtime evidence packet driver signature drifted from gate state"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_fresh_estimator_evidence_helper",
                _FRESH_ESTIMATOR_EVIDENCE_HELPER,
            )
        )
        != _FRESH_ESTIMATOR_EVIDENCE_HELPER
    ):
        raise ValueError(
            "runtime evidence packet fresh estimator evidence helper drifted from gate state"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_fresh_estimator_rerun_helper",
                _FRESH_ESTIMATOR_RERUN_HELPER,
            )
        )
        != _FRESH_ESTIMATOR_RERUN_HELPER
    ):
        raise ValueError(
            "runtime evidence packet fresh estimator rerun helper drifted from gate state"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_binding_design_rerun_capacity_helper",
                _RERUN_CAPACITY_HELPER,
            )
        )
        != _RERUN_CAPACITY_HELPER
    ):
        raise ValueError(
            "runtime evidence packet rerun capacity helper drifted from gate state"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_binding_design_rerun_capacity_note",
                _RERUN_CAPACITY_NOTE,
            )
        )
        != _RERUN_CAPACITY_NOTE
    ):
        raise ValueError(
            "runtime evidence packet rerun capacity note drifted from gate state"
        )
    if (
        str(
            gate_state.get(
                "runtime_evidence_binding_design_rerun_capacity_limiting_budget",
                _RERUN_CAPACITY_LIMITING_BUDGET,
            )
        )
        != _RERUN_CAPACITY_LIMITING_BUDGET
    ):
        raise ValueError(
            "runtime evidence packet rerun capacity limiting budget drifted from gate state"
        )

    if report.live_entry != live_entry:
        raise ValueError("runtime evidence packet live entry drifted from gate state")
    if report.route_label != route_label:
        raise ValueError("runtime evidence packet route label drifted from gate state")
    if report.current_gate_status != current_gate_status:
        raise ValueError(
            "runtime evidence packet current gate status drifted from gate state"
        )
    if report.accepted_feature_bundle != accepted_feature_bundle:
        raise ValueError(
            "runtime evidence packet accepted feature bundle drifted from gate state"
        )
    if report.policy_digest != policy_digest:
        raise ValueError("runtime evidence packet policy digest drifted from gate state")
    if report.binding_design != binding_design:
        raise ValueError("runtime evidence packet binding design drifted from gate state")
    if report.binding_random_states != binding_random_states:
        raise ValueError(
            "runtime evidence packet binding random states drifted from gate state"
        )
    if (
        report.additional_pointwise_witnesses_needed
        != additional_pointwise_witnesses_needed
    ):
        raise ValueError(
            "runtime evidence packet quota gap drifted from gate state"
        )
    if report.effective_fresh_reruns != effective_fresh_reruns:
        raise ValueError(
            "runtime evidence packet fresh rerun budget drifted from live helper state"
        )
    if report.fresh_estimator_evidence_intake_driver != str(
        gate_state.get(
            "runtime_evidence_fresh_estimator_evidence_driver",
            _FRESH_ESTIMATOR_EVIDENCE_DRIVER,
        )
    ):
        raise ValueError(
            "runtime evidence packet fresh estimator evidence driver drifted from gate state"
        )
    if report.fresh_estimator_evidence_intake_helper != str(
        gate_state.get(
            "runtime_evidence_fresh_estimator_evidence_helper",
            _FRESH_ESTIMATOR_EVIDENCE_HELPER,
        )
    ):
        raise ValueError(
            "runtime evidence packet fresh estimator evidence helper drifted from gate state"
        )
    if report.fresh_estimator_rerun_intake_driver != str(
        gate_state.get(
            "runtime_evidence_fresh_estimator_rerun_driver",
            _FRESH_ESTIMATOR_RERUN_DRIVER,
        )
    ):
        raise ValueError(
            "runtime evidence packet fresh estimator rerun driver drifted from gate state"
        )
    if report.fresh_estimator_rerun_intake_helper != str(
        gate_state.get(
            "runtime_evidence_fresh_estimator_rerun_helper",
            _FRESH_ESTIMATOR_RERUN_HELPER,
        )
    ):
        raise ValueError(
            "runtime evidence packet fresh estimator rerun helper drifted from gate state"
        )
    if (
        report.binding_design_rerun_capacity_implication
        != str(
            gate_state.get(
                "runtime_evidence_binding_design_rerun_capacity_implication",
                _RERUN_CAPACITY_IMPLICATION,
            )
        )
    ):
        raise ValueError(
            "runtime evidence packet rerun capacity implication drifted from gate state"
        )
    if report.current_implication != str(
        gate_state.get("runtime_evidence_current_implication", _CURRENT_IMPLICATION)
    ):
        raise ValueError(
            "runtime evidence packet current implication drifted from gate state"
        )
    if binding_design_rerun_capacity is not None:
        if (
            report.binding_design_rerun_capacity_implication
            != binding_design_rerun_capacity.current_implication
        ):
            raise ValueError(
                "runtime evidence packet rerun capacity implication drifted from the live helper"
            )
        if (
            report.binding_design_rerun_capacity_limiting_budget
            != binding_design_rerun_capacity.limiting_budget
        ):
            raise ValueError(
                "runtime evidence packet rerun capacity limiting budget drifted from the live helper"
            )
    return report


def _clear_phase7_monte_carlo_widening_policy_runtime_evidence_packet_cache() -> None:
    return None


run_phase7_monte_carlo_widening_policy_runtime_evidence_packet.cache_clear = (  # type: ignore[attr-defined]
    _clear_phase7_monte_carlo_widening_policy_runtime_evidence_packet_cache
)
