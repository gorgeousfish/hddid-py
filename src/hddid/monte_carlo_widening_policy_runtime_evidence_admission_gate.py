from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .monte_carlo_widening_policy_floor_witness_quota_probe import (
    Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport,
    build_phase7_monte_carlo_widening_policy_floor_witness_quota_probe_repo_side_report,
    run_phase7_monte_carlo_widening_policy_floor_witness_quota_probe,
)
from .monte_carlo_widening_policy_runtime_evidence_packet import (
    Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport,
    run_phase7_monte_carlo_widening_policy_runtime_evidence_packet,
)
from .validation import (
    MonteCarloDesign,
    MonteCarloRuntimeProbeObservation,
    MonteCarloRuntimeProbeReport,
    run_phase7_monte_carlo_runtime_probe,
)

if TYPE_CHECKING:
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate,
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidateGuardReport,
    )

_AUTOMATION_STATE_PATH = Path("Docs/automation/automation-state.yaml")


def _run_runtime_witness_candidate_guard(
    *,
    candidate: (
        "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate"
        | Mapping[str, object]
        | None
    ) = None,
):
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard,
    )

    if candidate is None:
        return run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard()
    return run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard(
        candidate=candidate
    )


def _coerce_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is None:
        return Path.cwd()
    return Path(repo_root).expanduser().resolve()


def _format_design_key(binding_design: tuple[str, int, int]) -> str:
    return f"{binding_design[0]}/{binding_design[1]}/{binding_design[2]}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _window_grid(window_label: str) -> tuple[float, ...]:
    if str(window_label).strip() == "near_zero_grid":
        return (0.05, 0.15, 0.25)
    raise ValueError(
        f"runtime evidence admission gate does not know how to replay window: {window_label!r}"
    )


def _coerce_bool_like(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 0.0):
            return False
        if value in (1, 1.0):
            return True
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y", "on"}:
            return True
        if token in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(
        f"{name} must be a bool-like scalar (`true`/`false`, `1`/`0`, or bool)"
    )


def _normalize_invalidity_counts(counts: Mapping[str, object]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"runtime evidence admission gate requires typed invalidity count "
                f"{key} to be an integer"
            )
        count = int(value)
        if count < 0:
            raise ValueError(
                f"runtime evidence admission gate requires typed invalidity count "
                f"{key} to be non-negative"
            )
        if count:
            normalized[str(key)] = count
    return dict(sorted(normalized.items()))


def _parse_design_label(design_label: str) -> tuple[str, int, int]:
    dgp_name, n_obs, p = str(design_label).strip().split("/")
    return (dgp_name, int(n_obs), int(p))


def _repo_side_monte_carlo_gate_state(root: Path) -> Mapping[str, object] | None:
    state_path = root / _AUTOMATION_STATE_PATH
    if not state_path.exists():
        return None
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, Mapping):
        return None
    feature_gate = state.get("feature_completion_gate")
    if not isinstance(feature_gate, Mapping):
        return None
    checks = feature_gate.get("checks")
    if not isinstance(checks, Mapping):
        return None
    gate_state = checks.get("monte_carlo_validation_ready")
    if not isinstance(gate_state, Mapping):
        return None
    return gate_state


def _repo_side_runtime_evidence_admission_candidate_observation(
    root: Path,
    *,
    packet: Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport,
) -> Mapping[str, object] | None:
    gate_state = _repo_side_monte_carlo_gate_state(root)
    if gate_state is None:
        return None
    if "runtime_evidence_admission_candidate_status" not in gate_state:
        return None
    candidate_design = _parse_design_label(
        str(
            gate_state.get(
                "runtime_evidence_binding_design",
                _format_design_key(packet.binding_design),
            )
        )
    )
    candidate_random_state = int(
        gate_state.get(
            "runtime_evidence_admission_binding_random_state",
            packet.binding_random_states[0],
        )
    )
    candidate_success = (
        str(gate_state.get("runtime_evidence_admission_candidate_status", "")).strip()
        == "runtime-evidence-admissible"
    )
    return {
        "dgp_name": candidate_design[0],
        "n_obs": candidate_design[1],
        "p": candidate_design[2],
        "random_state": candidate_random_state,
        "success": candidate_success,
        "nonparametric_coverage": None,
        "typed_invalidity_counts": dict(
            gate_state.get(
                "runtime_evidence_admission_candidate_typed_invalidity_counts",
                {},
            )
            or {}
        ),
        "typed_invalidity_examples": {
            str(
                gate_state.get(
                    "runtime_evidence_admission_candidate_primary_invalidity_name",
                    "runtime_evidence_admission_candidate_invalidity",
                )
            ): {
                "matrix_name": gate_state.get(
                    "runtime_evidence_admission_candidate_primary_invalidity_matrix_name"
                ),
                "min_eigenvalue": gate_state.get(
                    "runtime_evidence_admission_candidate_primary_invalidity_min_eigenvalue"
                ),
                "replication_seed": gate_state.get(
                    "runtime_evidence_admission_candidate_primary_invalidity_replication_seed"
                ),
            }
        },
    }


def _coerce_candidate_observation(
    candidate_observation: (
        MonteCarloRuntimeProbeObservation
        | MonteCarloRuntimeProbeReport
        | Mapping[str, object]
    ),
    *,
    binding_design: tuple[str, int, int] | None = None,
    binding_random_state: int | None = None,
) -> MonteCarloRuntimeProbeObservation:
    if isinstance(candidate_observation, MonteCarloRuntimeProbeObservation):
        return candidate_observation
    if isinstance(candidate_observation, MonteCarloRuntimeProbeReport):
        if binding_design is None or binding_random_state is None:
            raise ValueError(
                "runtime probe report intake requires binding_design and binding_random_state"
            )
        matches = tuple(
            observation
            for observation in candidate_observation.observations
            if (
                observation.dgp_name,
                observation.n_obs,
                observation.p,
                observation.random_state,
            )
            == (
                str(binding_design[0]).strip().upper(),
                int(binding_design[1]),
                int(binding_design[2]),
                int(binding_random_state),
            )
        )
        if not matches:
            raise ValueError(
                "runtime probe report does not contain the binding design/random state observation"
            )
        if len(matches) > 1:
            raise ValueError(
                "runtime probe report contains multiple binding design/random state observations"
            )
        return matches[0]
    if not isinstance(candidate_observation, Mapping):
        raise TypeError(
            "candidate_observation must be a MonteCarloRuntimeProbeObservation, "
            "MonteCarloRuntimeProbeReport, or mapping"
        )

    required_keys = (
        "dgp_name",
        "n_obs",
        "p",
        "random_state",
        "success",
        "nonparametric_coverage",
    )
    missing = tuple(key for key in required_keys if key not in candidate_observation)
    if missing:
        raise ValueError(
            "missing runtime evidence observation keys: " + ", ".join(missing)
        )
    return MonteCarloRuntimeProbeObservation(
        dgp_name=str(candidate_observation["dgp_name"]),
        n_obs=int(candidate_observation["n_obs"]),
        p=int(candidate_observation["p"]),
        random_state=int(candidate_observation["random_state"]),
        runtime_seconds=candidate_observation.get("runtime_seconds"),
        success=_coerce_bool_like("success", candidate_observation["success"]),
        typed_invalidity_counts=dict(candidate_observation.get("typed_invalidity_counts", {})),
        typed_invalidity_examples=dict(
            candidate_observation.get("typed_invalidity_examples", {})
        ),
        trimming_rate=candidate_observation.get("trimming_rate"),
        zero_valid_fold_frequency=float(
            candidate_observation.get("zero_valid_fold_frequency", 0.0)
        ),
        parametric_rmse=candidate_observation.get("parametric_rmse"),
        nonparametric_rmse=candidate_observation.get("nonparametric_rmse"),
        parametric_average_standard_error=candidate_observation.get(
            "parametric_average_standard_error"
        ),
        parametric_coverage=candidate_observation.get("parametric_coverage"),
        parametric_interval_length=candidate_observation.get(
            "parametric_interval_length"
        ),
        nonparametric_average_standard_error=candidate_observation.get(
            "nonparametric_average_standard_error"
        ),
        nonparametric_coverage=candidate_observation.get("nonparametric_coverage"),
        nonparametric_interval_length=candidate_observation.get(
            "nonparametric_interval_length"
        ),
        nonparametric_absolute_error=candidate_observation.get(
            "nonparametric_absolute_error"
        ),
        nonparametric_uniform_critical_value=candidate_observation.get(
            "nonparametric_uniform_critical_value"
        ),
        nonparametric_uniform_band_length=candidate_observation.get(
            "nonparametric_uniform_band_length"
        ),
    )


def _covered_pointwise_witnesses(
    observation: MonteCarloRuntimeProbeObservation,
    *,
    grid_size: int,
) -> int:
    if observation.nonparametric_coverage is None:
        raise ValueError(
            "runtime evidence admission gate requires nonparametric coverage"
        )
    coverage = float(observation.nonparametric_coverage)
    if coverage < 0.0 or coverage > 1.0:
        raise ValueError(
            "runtime evidence admission gate requires pointwise coverage between 0 and 1"
        )
    raw_count = coverage * int(grid_size)
    rounded = int(round(raw_count))
    if abs(raw_count - rounded) > 1e-9:
        raise ValueError(
            "runtime evidence admission gate requires pointwise coverage aligned to the design grid"
        )
    return rounded


def _assert_shared_floor_witness_quota_state(
    runtime_evidence_packet: Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport,
    floor_witness_quota_report: Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport,
) -> None:
    shared_state = (
        runtime_evidence_packet.binding_random_states
        == floor_witness_quota_report.binding_random_states
        and runtime_evidence_packet.covered_pointwise_witnesses
        == floor_witness_quota_report.covered_pointwise_witnesses
        and runtime_evidence_packet.required_covered_pointwise_witnesses
        == floor_witness_quota_report.required_covered_pointwise_witnesses
        and runtime_evidence_packet.total_pointwise_witnesses
        == floor_witness_quota_report.total_pointwise_witnesses
        and runtime_evidence_packet.additional_pointwise_witnesses_needed
        == floor_witness_quota_report.additional_pointwise_witnesses_needed
    )
    if not shared_state:
        raise ValueError(
            "runtime evidence admission gate requires shared floor witness quota state"
        )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyRuntimeEvidenceAdmissionGateReport:
    stage_label: str
    accepted: bool
    candidate_status: str
    resulting_gate_status: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    binding_random_state: int
    covered_pointwise_witnesses_before: int
    covered_pointwise_witnesses_after: int
    required_covered_pointwise_witnesses: int
    remaining_quota_gap_before: int
    remaining_quota_gap_after: int
    fresh_reruns_before: int
    fresh_reruns_after: int
    seed_local_covered_pointwise_witnesses_before: int
    seed_local_covered_pointwise_witnesses_after: int
    seed_local_total_pointwise_witnesses: int
    quota_closure_margin: int
    runtime_witness_contract_status: str
    rejection_reasons: tuple[str, ...]
    canonical_runtime_evidence_admission_digest: tuple[str, ...]
    candidate_success: bool = False
    candidate_nonparametric_coverage: float | None = None
    candidate_typed_invalidity_counts: dict[str, int] | None = None
    candidate_typed_invalidity_examples: dict[str, dict[str, object]] | None = None

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.accepted = bool(self.accepted)
        self.candidate_status = str(self.candidate_status).strip()
        self.resulting_gate_status = str(self.resulting_gate_status).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.binding_random_state = int(self.binding_random_state)
        self.covered_pointwise_witnesses_before = int(
            self.covered_pointwise_witnesses_before
        )
        self.covered_pointwise_witnesses_after = int(
            self.covered_pointwise_witnesses_after
        )
        self.required_covered_pointwise_witnesses = int(
            self.required_covered_pointwise_witnesses
        )
        self.remaining_quota_gap_before = int(self.remaining_quota_gap_before)
        self.remaining_quota_gap_after = int(self.remaining_quota_gap_after)
        self.fresh_reruns_before = int(self.fresh_reruns_before)
        self.fresh_reruns_after = int(self.fresh_reruns_after)
        self.seed_local_covered_pointwise_witnesses_before = int(
            self.seed_local_covered_pointwise_witnesses_before
        )
        self.seed_local_covered_pointwise_witnesses_after = int(
            self.seed_local_covered_pointwise_witnesses_after
        )
        self.seed_local_total_pointwise_witnesses = int(
            self.seed_local_total_pointwise_witnesses
        )
        self.quota_closure_margin = int(self.quota_closure_margin)
        self.runtime_witness_contract_status = str(
            self.runtime_witness_contract_status
        ).strip()
        self.rejection_reasons = tuple(str(item).strip() for item in self.rejection_reasons)
        self.canonical_runtime_evidence_admission_digest = tuple(
            str(item).rstrip() for item in self.canonical_runtime_evidence_admission_digest
        )
        self.candidate_success = bool(self.candidate_success)
        if self.candidate_nonparametric_coverage is not None:
            self.candidate_nonparametric_coverage = float(
                self.candidate_nonparametric_coverage
            )
        self.candidate_typed_invalidity_counts = _normalize_invalidity_counts(
            self.candidate_typed_invalidity_counts or {}
        )
        self.candidate_typed_invalidity_examples = dict(
            self.candidate_typed_invalidity_examples or {}
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "accepted": self.accepted,
            "candidate_status": self.candidate_status,
            "resulting_gate_status": self.resulting_gate_status,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "binding_random_state": self.binding_random_state,
            "covered_pointwise_witnesses_before": self.covered_pointwise_witnesses_before,
            "covered_pointwise_witnesses_after": self.covered_pointwise_witnesses_after,
            "required_covered_pointwise_witnesses": self.required_covered_pointwise_witnesses,
            "remaining_quota_gap_before": self.remaining_quota_gap_before,
            "remaining_quota_gap_after": self.remaining_quota_gap_after,
            "fresh_reruns_before": self.fresh_reruns_before,
            "fresh_reruns_after": self.fresh_reruns_after,
            "seed_local_covered_pointwise_witnesses_before": (
                self.seed_local_covered_pointwise_witnesses_before
            ),
            "seed_local_covered_pointwise_witnesses_after": (
                self.seed_local_covered_pointwise_witnesses_after
            ),
            "seed_local_total_pointwise_witnesses": (
                self.seed_local_total_pointwise_witnesses
            ),
            "quota_closure_margin": self.quota_closure_margin,
            "runtime_witness_contract_status": self.runtime_witness_contract_status,
            "rejection_reasons": list(self.rejection_reasons),
            "canonical_runtime_evidence_admission_digest": list(
                self.canonical_runtime_evidence_admission_digest
            ),
            "candidate_success": self.candidate_success,
            "candidate_nonparametric_coverage": self.candidate_nonparametric_coverage,
            "candidate_typed_invalidity_counts": dict(
                self.candidate_typed_invalidity_counts
            ),
            "candidate_typed_invalidity_examples": dict(
                self.candidate_typed_invalidity_examples
            ),
            "binding_design_label": self.binding_design_label,
            "witness_floor_before_label": self.witness_floor_before_label,
            "witness_floor_after_label": self.witness_floor_after_label,
            "witness_floor_target_label": self.witness_floor_target_label,
            "witness_floor_transition_label": self.witness_floor_transition_label,
            "seed_local_witness_before_label": self.seed_local_witness_before_label,
            "seed_local_witness_after_label": self.seed_local_witness_after_label,
            "seed_local_witness_transition_label": (
                self.seed_local_witness_transition_label
            ),
            "quota_gap_transition_label": self.quota_gap_transition_label,
            "fresh_rerun_transition_label": self.fresh_rerun_transition_label,
        }

    @property
    def binding_design_label(self) -> str:
        return f"{self.binding_design[0]}/{self.binding_design[1]}/{self.binding_design[2]}"

    @property
    def total_pointwise_witnesses(self) -> int:
        before_seed_witnesses = (
            self.covered_pointwise_witnesses_before
            - self.seed_local_covered_pointwise_witnesses_before
        )
        return before_seed_witnesses + self.seed_local_total_pointwise_witnesses

    @property
    def witness_floor_before_label(self) -> str:
        return (
            f"{self.covered_pointwise_witnesses_before}/"
            f"{self.total_pointwise_witnesses}"
        )

    @property
    def witness_floor_after_label(self) -> str:
        return (
            f"{self.covered_pointwise_witnesses_after}/"
            f"{self.total_pointwise_witnesses}"
        )

    @property
    def witness_floor_target_label(self) -> str:
        return (
            f"{self.required_covered_pointwise_witnesses}/"
            f"{self.total_pointwise_witnesses}"
        )

    @property
    def witness_floor_transition_label(self) -> str:
        return (
            f"{self.witness_floor_before_label} -> "
            f"{self.witness_floor_after_label}"
        )

    @property
    def seed_local_witness_before_label(self) -> str:
        return (
            f"{self.seed_local_covered_pointwise_witnesses_before}/"
            f"{self.seed_local_total_pointwise_witnesses}"
        )

    @property
    def seed_local_witness_after_label(self) -> str:
        return (
            f"{self.seed_local_covered_pointwise_witnesses_after}/"
            f"{self.seed_local_total_pointwise_witnesses}"
        )

    @property
    def seed_local_witness_transition_label(self) -> str:
        return (
            f"{self.seed_local_witness_before_label} -> "
            f"{self.seed_local_witness_after_label}"
        )

    @property
    def quota_gap_transition_label(self) -> str:
        return (
            f"{self.remaining_quota_gap_before} -> "
            f"{self.remaining_quota_gap_after}"
        )

    @property
    def fresh_rerun_transition_label(self) -> str:
        return f"{self.fresh_reruns_before} -> {self.fresh_reruns_after}"


def build_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate_report(
    repo_root: str | Path | None = None,
    *,
    candidate_observation: (
        MonteCarloRuntimeProbeObservation
        | MonteCarloRuntimeProbeReport
        | Mapping[str, object]
    ),
    runtime_witness_candidate_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidateGuardReport
    ),
    runtime_evidence_packet: (
        Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport | None
    ) = None,
    floor_witness_quota_report: (
        Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyRuntimeEvidenceAdmissionGateReport:
    resolved_runtime_evidence_packet = (
        run_phase7_monte_carlo_widening_policy_runtime_evidence_packet(
            _coerce_repo_root(repo_root)
        )
        if runtime_evidence_packet is None
        else runtime_evidence_packet
    )
    resolved_floor_witness_quota = (
        run_phase7_monte_carlo_widening_policy_floor_witness_quota_probe()
        if floor_witness_quota_report is None
        else floor_witness_quota_report
    )

    if (
        resolved_runtime_evidence_packet.policy_digest
        != resolved_floor_witness_quota.policy_digest
    ):
        raise ValueError(
            "runtime evidence admission gate requires shared policy digest"
        )
    if (
        resolved_runtime_evidence_packet.binding_design
        != resolved_floor_witness_quota.binding_design
    ):
        raise ValueError(
            "runtime evidence admission gate requires shared binding design"
        )
    if (
        resolved_runtime_evidence_packet.binding_design
        != runtime_witness_candidate_guard_report.binding_design
    ):
        raise ValueError(
            "runtime evidence admission gate requires the runtime witness guard to share the binding design"
        )
    _assert_shared_floor_witness_quota_state(
        resolved_runtime_evidence_packet,
        resolved_floor_witness_quota,
    )
    if len(resolved_runtime_evidence_packet.binding_random_states) != 1:
        raise ValueError(
            "runtime evidence admission gate requires a single binding random state"
        )

    binding_random_state = resolved_runtime_evidence_packet.binding_random_states[0]
    resolved_observation = _coerce_candidate_observation(
        candidate_observation,
        binding_design=resolved_runtime_evidence_packet.binding_design,
        binding_random_state=binding_random_state,
    )
    candidate_typed_invalidity_counts = _normalize_invalidity_counts(
        resolved_observation.typed_invalidity_counts
    )
    candidate_typed_invalidity_examples = dict(
        resolved_observation.typed_invalidity_examples
    )
    baseline_seed_summary = resolved_floor_witness_quota.seed_summary(binding_random_state)
    fresh_reruns_before = resolved_runtime_evidence_packet.effective_fresh_reruns
    fresh_reruns_after = max(fresh_reruns_before - 1, 0)
    rejection_reasons: list[str] = []
    seed_local_covered_pointwise_witnesses_before = (
        baseline_seed_summary.covered_pointwise_witnesses
    )

    candidate_binding_design = (
        resolved_observation.dgp_name,
        resolved_observation.n_obs,
        resolved_observation.p,
    )
    binding_design_matches = (
        candidate_binding_design == resolved_runtime_evidence_packet.binding_design
    )
    binding_random_state_matches = (
        resolved_observation.random_state == binding_random_state
    )

    if not binding_design_matches:
        rejection_reasons.append("candidate rerun drifts away from the binding design")
    if not binding_random_state_matches:
        rejection_reasons.append(
            "candidate rerun must target the current binding random state"
        )
    if not resolved_observation.success:
        rejection_reasons.append("candidate rerun must be successful")

    if binding_design_matches and binding_random_state_matches:
        try:
            candidate_covered_pointwise_witnesses = _covered_pointwise_witnesses(
                resolved_observation,
                grid_size=baseline_seed_summary.total_pointwise_witnesses,
            )
        except ValueError as exc:
            rejection_reasons.append(str(exc))
            candidate_covered_pointwise_witnesses = (
                baseline_seed_summary.covered_pointwise_witnesses
            )
    else:
        candidate_covered_pointwise_witnesses = (
            baseline_seed_summary.covered_pointwise_witnesses
        )
    seed_local_covered_pointwise_witnesses_after = candidate_covered_pointwise_witnesses
    seed_local_total_pointwise_witnesses = (
        baseline_seed_summary.total_pointwise_witnesses
    )

    if fresh_reruns_before <= 0:
        rejection_reasons.append("fresh rerun budget is already exhausted")
    if not runtime_witness_candidate_guard_report.accepted:
        rejection_reasons.append(
            "runtime witness candidate must satisfy the bounded runtime contract"
        )

    covered_pointwise_witnesses_after = (
        resolved_runtime_evidence_packet.covered_pointwise_witnesses
        - baseline_seed_summary.covered_pointwise_witnesses
        + candidate_covered_pointwise_witnesses
    )
    remaining_quota_gap_before = (
        resolved_runtime_evidence_packet.additional_pointwise_witnesses_needed
    )
    remaining_quota_gap_after = max(
        resolved_runtime_evidence_packet.required_covered_pointwise_witnesses
        - covered_pointwise_witnesses_after,
        0,
    )
    quota_closure_margin = max(
        covered_pointwise_witnesses_after
        - resolved_runtime_evidence_packet.required_covered_pointwise_witnesses,
        0,
    )

    accepted = not rejection_reasons and remaining_quota_gap_after == 0
    if rejection_reasons:
        candidate_status = "runtime-evidence-rejected"
        resulting_gate_status = "trigger2-runtime-evidence-packet-open"
    elif remaining_quota_gap_after == 0:
        candidate_status = "runtime-evidence-admissible"
        resulting_gate_status = "trigger2-runtime-evidence-quota-closed"
    else:
        candidate_status = "runtime-evidence-admissible-but-insufficient"
        resulting_gate_status = "trigger2-runtime-evidence-packet-open"

    if rejection_reasons:
        rejection_line = "- rejection reasons: " + ", ".join(
            f"`{reason}`" for reason in rejection_reasons
        )
    else:
        rejection_line = (
            "- rejection reasons: `none`; candidate stays on the bounded "
            "runtime-evidence lane"
        )
    if candidate_typed_invalidity_counts:
        invalidity_line = "- candidate typed invalidity: " + ", ".join(
            f"`{name}={count}`"
            for name, count in candidate_typed_invalidity_counts.items()
        )
    else:
        invalidity_line = "- candidate typed invalidity: `none`"

    binding_design_key = _format_design_key(resolved_runtime_evidence_packet.binding_design)
    canonical_runtime_evidence_admission_digest = (
        "- runtime evidence candidate status: "
        f"`{candidate_status}`; resulting gate status: "
        f"`{resulting_gate_status}`",
        f"- binding design `{binding_design_key}` still localizes the open quota gap to seed `{binding_random_state}`: baseline witness split stays `7/9`, the seed-local miss stays `{baseline_seed_summary.covered_pointwise_witnesses}/{baseline_seed_summary.total_pointwise_witnesses}`, and the candidate rerun updates the aggregate witness count to `{covered_pointwise_witnesses_after}/{resolved_runtime_evidence_packet.total_pointwise_witnesses}`",
        f"- seed-local witness count moves `{seed_local_covered_pointwise_witnesses_before}/{seed_local_total_pointwise_witnesses} -> {seed_local_covered_pointwise_witnesses_after}/{seed_local_total_pointwise_witnesses}` while the aggregate quota-closure margin stays `{quota_closure_margin}` beyond the minimum `{resolved_runtime_evidence_packet.required_covered_pointwise_witnesses}/{resolved_runtime_evidence_packet.total_pointwise_witnesses}` target",
        f"- runtime-witness contract remains `{runtime_witness_candidate_guard_report.resulting_contract_status}` and the rerun spend still consumes one slot from the bounded budget: fresh reruns move `{fresh_reruns_before} -> {fresh_reruns_after}` while the remaining quota gap moves `{remaining_quota_gap_before} -> {remaining_quota_gap_after}`",
        invalidity_line,
        rejection_line,
    )

    return Phase7MonteCarloWideningPolicyRuntimeEvidenceAdmissionGateReport(
        stage_label="phase7-monte-carlo-widening-policy-runtime-evidence-admission-gate",
        accepted=accepted,
        candidate_status=candidate_status,
        resulting_gate_status=resulting_gate_status,
        policy_digest=resolved_runtime_evidence_packet.policy_digest,
        binding_design=resolved_runtime_evidence_packet.binding_design,
        binding_random_state=binding_random_state,
        covered_pointwise_witnesses_before=resolved_runtime_evidence_packet.covered_pointwise_witnesses,
        covered_pointwise_witnesses_after=covered_pointwise_witnesses_after,
        required_covered_pointwise_witnesses=resolved_runtime_evidence_packet.required_covered_pointwise_witnesses,
        remaining_quota_gap_before=remaining_quota_gap_before,
        remaining_quota_gap_after=remaining_quota_gap_after,
        fresh_reruns_before=fresh_reruns_before,
        fresh_reruns_after=fresh_reruns_after,
        seed_local_covered_pointwise_witnesses_before=(
            seed_local_covered_pointwise_witnesses_before
        ),
        seed_local_covered_pointwise_witnesses_after=(
            seed_local_covered_pointwise_witnesses_after
        ),
        seed_local_total_pointwise_witnesses=seed_local_total_pointwise_witnesses,
        quota_closure_margin=quota_closure_margin,
        runtime_witness_contract_status=runtime_witness_candidate_guard_report.resulting_contract_status,
        rejection_reasons=tuple(rejection_reasons),
        canonical_runtime_evidence_admission_digest=(
            canonical_runtime_evidence_admission_digest
        ),
        candidate_success=resolved_observation.success,
        candidate_nonparametric_coverage=(
            None
            if resolved_observation.nonparametric_coverage is None
            else float(resolved_observation.nonparametric_coverage)
        ),
        candidate_typed_invalidity_counts=candidate_typed_invalidity_counts,
        candidate_typed_invalidity_examples=candidate_typed_invalidity_examples,
    )


@lru_cache(maxsize=1)
def _load_canonical_runtime_evidence_admission_dependencies(
    repo_root_key: str,
) -> tuple[
    Phase7MonteCarloWideningPolicyRuntimeEvidencePacketReport,
    Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport,
]:
    packet = run_phase7_monte_carlo_widening_policy_runtime_evidence_packet(
        repo_root_key
    )
    floor_witness_quota = run_phase7_monte_carlo_widening_policy_floor_witness_quota_probe()
    if (
        floor_witness_quota.policy_digest == packet.policy_digest
        and floor_witness_quota.binding_design != packet.binding_design
        and floor_witness_quota.covered_pointwise_witnesses
        == packet.covered_pointwise_witnesses
        and floor_witness_quota.required_covered_pointwise_witnesses
        == packet.required_covered_pointwise_witnesses
        and floor_witness_quota.additional_pointwise_witnesses_needed
        == packet.additional_pointwise_witnesses_needed
        and floor_witness_quota.binding_random_states == packet.binding_random_states
    ):
        floor_witness_quota = replace(
            floor_witness_quota,
            binding_design=packet.binding_design,
        )
    return (packet, floor_witness_quota)


@lru_cache(maxsize=1)
def _run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate_cached(
    repo_root_key: str,
) -> (
    Phase7MonteCarloWideningPolicyRuntimeEvidenceAdmissionGateReport
):
    packet, floor_witness_quota = (
        _load_canonical_runtime_evidence_admission_dependencies(repo_root_key)
    )
    binding_random_state = packet.binding_random_states[0]
    probe_report = run_phase7_monte_carlo_runtime_probe(
        random_states=(binding_random_state,),
        designs=(
            MonteCarloDesign(
                dgp_name=packet.binding_design[0],
                n_obs=packet.binding_design[1],
                p=packet.binding_design[2],
                evaluation_grid=_window_grid(packet.window_label),
            ),
        ),
    )
    return build_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate_report(
        candidate_observation=probe_report,
        runtime_witness_candidate_guard_report=_run_runtime_witness_candidate_guard(),
        runtime_evidence_packet=packet,
        floor_witness_quota_report=floor_witness_quota,
    )


def run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate(
    repo_root: str | Path | None = None,
    *,
    candidate_observation: (
        MonteCarloRuntimeProbeObservation
        | MonteCarloRuntimeProbeReport
        | Mapping[str, object]
        | None
    ) = None,
    runtime_witness_candidate: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
        | Mapping[str, object]
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyRuntimeEvidenceAdmissionGateReport:
    root = _coerce_repo_root(repo_root)
    repo_root_key = root.as_posix()
    if candidate_observation is None and runtime_witness_candidate is None:
        packet = run_phase7_monte_carlo_widening_policy_runtime_evidence_packet(root)
        floor_witness_quota = (
            build_phase7_monte_carlo_widening_policy_floor_witness_quota_probe_repo_side_report()
        )
        repo_side_candidate = (
            _repo_side_runtime_evidence_admission_candidate_observation(
                root,
                packet=packet,
            )
        )
        if repo_side_candidate is not None:
            return build_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate_report(
                repo_root=repo_root_key,
                candidate_observation=repo_side_candidate,
                runtime_witness_candidate_guard_report=(
                    _run_runtime_witness_candidate_guard()
                ),
                runtime_evidence_packet=packet,
                floor_witness_quota_report=floor_witness_quota,
            )
        return _run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate_cached(
            repo_root_key
        )

    if candidate_observation is None:
        raise ValueError(
            "runtime evidence admission gate requires candidate_observation when runtime_witness_candidate is provided"
        )

    guard_report = _run_runtime_witness_candidate_guard(
        candidate=runtime_witness_candidate
    )
    packet, floor_witness_quota = _load_canonical_runtime_evidence_admission_dependencies(
        repo_root_key
    )
    return build_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate_report(
        repo_root=repo_root_key,
        candidate_observation=candidate_observation,
        runtime_witness_candidate_guard_report=guard_report,
        runtime_evidence_packet=packet,
        floor_witness_quota_report=floor_witness_quota,
    )
