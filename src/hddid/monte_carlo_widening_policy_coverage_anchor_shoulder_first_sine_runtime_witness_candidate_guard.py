from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract,
)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _isclose(left: float, right: float) -> bool:
    return bool(np.isclose(float(left), float(right), atol=1e-12, rtol=0.0))


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


_REQUIRED_CANDIDATE_KEYS = (
    "runtime_witness_path",
    "preserves_left_support_contract",
    "diagonal_preserved",
    "direct_covariance_edits_allowed",
    "required_patch_share_of_full_shared_vf_gap",
    "required_patch_share_of_omega_only_shared_vf_increment",
    "required_patch_share_of_diagonal_omega_gap",
    "required_patch_share_of_psd_boundary",
    "compensating_stage_order",
    "compensating_cumulative_share_of_total_absolute_mass",
    "compensating_zero_live_entry_count",
)


def _candidate_input_mode(
    candidate: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
        | Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
        | Mapping[str, object]
        | None
    ),
) -> str:
    if candidate is None:
        return "canonical-replay"
    if isinstance(
        candidate,
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport,
    ):
        return "provided-runtime-witness-contract"
    if isinstance(
        candidate,
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate,
    ):
        return "provided-runtime-witness-object"
    if isinstance(candidate, Mapping):
        return "provided-runtime-witness-mapping"
    raise TypeError(
        "runtime witness candidate must be a "
        "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate, "
        "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport, "
        "or mapping"
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate:
    runtime_witness_path: tuple[str, ...]
    preserves_left_support_contract: bool
    diagonal_preserved: bool
    direct_covariance_edits_allowed: bool
    required_patch_share_of_full_shared_vf_gap: float
    required_patch_share_of_omega_only_shared_vf_increment: float
    required_patch_share_of_diagonal_omega_gap: float
    required_patch_share_of_psd_boundary: float
    compensating_stage_order: tuple[str, ...]
    compensating_cumulative_share_of_total_absolute_mass: tuple[float, ...]
    compensating_zero_live_entry_count: int

    def __post_init__(self) -> None:
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.preserves_left_support_contract = _coerce_bool_like(
            "preserves_left_support_contract",
            self.preserves_left_support_contract,
        )
        self.diagonal_preserved = _coerce_bool_like(
            "diagonal_preserved",
            self.diagonal_preserved,
        )
        self.direct_covariance_edits_allowed = _coerce_bool_like(
            "direct_covariance_edits_allowed",
            self.direct_covariance_edits_allowed,
        )
        self.required_patch_share_of_full_shared_vf_gap = float(
            self.required_patch_share_of_full_shared_vf_gap
        )
        self.required_patch_share_of_omega_only_shared_vf_increment = float(
            self.required_patch_share_of_omega_only_shared_vf_increment
        )
        self.required_patch_share_of_diagonal_omega_gap = float(
            self.required_patch_share_of_diagonal_omega_gap
        )
        self.required_patch_share_of_psd_boundary = float(
            self.required_patch_share_of_psd_boundary
        )
        self.compensating_stage_order = tuple(
            str(item).strip() for item in self.compensating_stage_order
        )
        self.compensating_cumulative_share_of_total_absolute_mass = tuple(
            float(value)
            for value in self.compensating_cumulative_share_of_total_absolute_mass
        )
        self.compensating_zero_live_entry_count = int(
            self.compensating_zero_live_entry_count
        )


def _coerce_candidate(
    candidate: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
        | Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
        | Mapping[str, object]
    ),
) -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
):
    if isinstance(
        candidate,
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate,
    ):
        return candidate
    if isinstance(
        candidate,
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport,
    ):
        return _canonical_candidate(candidate)
    if not isinstance(candidate, Mapping):
        raise TypeError(
            "runtime witness candidate must be a "
            "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate "
            "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport "
            "or mapping"
        )

    missing = tuple(key for key in _REQUIRED_CANDIDATE_KEYS if key not in candidate)
    if missing:
        raise ValueError(
            "missing runtime witness candidate keys: " + ", ".join(missing)
        )

    unexpected = tuple(
        sorted(str(key) for key in candidate if key not in _REQUIRED_CANDIDATE_KEYS)
    )
    if unexpected:
        raise ValueError(
            "unexpected runtime witness candidate keys: " + ", ".join(unexpected)
        )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate(
        runtime_witness_path=tuple(candidate["runtime_witness_path"]),
        preserves_left_support_contract=candidate["preserves_left_support_contract"],
        diagonal_preserved=candidate["diagonal_preserved"],
        direct_covariance_edits_allowed=candidate["direct_covariance_edits_allowed"],
        required_patch_share_of_full_shared_vf_gap=candidate[
            "required_patch_share_of_full_shared_vf_gap"
        ],
        required_patch_share_of_omega_only_shared_vf_increment=candidate[
            "required_patch_share_of_omega_only_shared_vf_increment"
        ],
        required_patch_share_of_diagonal_omega_gap=candidate[
            "required_patch_share_of_diagonal_omega_gap"
        ],
        required_patch_share_of_psd_boundary=candidate[
            "required_patch_share_of_psd_boundary"
        ],
        compensating_stage_order=tuple(candidate["compensating_stage_order"]),
        compensating_cumulative_share_of_total_absolute_mass=tuple(
            candidate["compensating_cumulative_share_of_total_absolute_mass"]
        ),
        compensating_zero_live_entry_count=candidate[
            "compensating_zero_live_entry_count"
        ],
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidateGuardReport:
    stage_label: str
    accepted: bool
    candidate_input_mode: str
    candidate_status: str
    resulting_contract_status: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    canonical_runtime_witness_path: tuple[str, ...]
    candidate_runtime_witness_path: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    max_budget_share_slack: float
    min_compensating_stage_coverage_slack: float
    zero_live_entry_slack: int
    canonical_candidate_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.accepted = bool(self.accepted)
        self.candidate_input_mode = str(self.candidate_input_mode).strip()
        self.candidate_status = str(self.candidate_status).strip()
        self.resulting_contract_status = str(self.resulting_contract_status).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.canonical_runtime_witness_path = tuple(
            str(item).strip() for item in self.canonical_runtime_witness_path
        )
        self.candidate_runtime_witness_path = tuple(
            str(item).strip() for item in self.candidate_runtime_witness_path
        )
        self.rejection_reasons = tuple(
            str(item).strip() for item in self.rejection_reasons
        )
        self.max_budget_share_slack = float(self.max_budget_share_slack)
        self.min_compensating_stage_coverage_slack = float(
            self.min_compensating_stage_coverage_slack
        )
        self.zero_live_entry_slack = int(self.zero_live_entry_slack)
        self.canonical_candidate_digest = tuple(
            str(item).rstrip() for item in self.canonical_candidate_digest
        )


def _canonical_candidate(
    contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
    ),
) -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate(
        runtime_witness_path=contract_report.runtime_witness_path,
        preserves_left_support_contract=contract_report.preserves_left_support_contract,
        diagonal_preserved=contract_report.diagonal_preserved,
        direct_covariance_edits_allowed=contract_report.direct_covariance_edits_allowed,
        required_patch_share_of_full_shared_vf_gap=contract_report.required_patch_share_of_full_shared_vf_gap,
        required_patch_share_of_omega_only_shared_vf_increment=contract_report.required_patch_share_of_omega_only_shared_vf_increment,
        required_patch_share_of_diagonal_omega_gap=contract_report.required_patch_share_of_diagonal_omega_gap,
        required_patch_share_of_psd_boundary=contract_report.required_patch_share_of_psd_boundary,
        compensating_stage_order=contract_report.compensating_stage_order,
        compensating_cumulative_share_of_total_absolute_mass=contract_report.compensating_cumulative_share_of_total_absolute_mass,
        compensating_zero_live_entry_count=contract_report.compensating_zero_live_entry_count,
    )


def _rejection_reasons(
    *,
    candidate: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate,
    contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
    ),
) -> tuple[str, ...]:
    reasons: list[str] = []
    if candidate.runtime_witness_path != contract_report.runtime_witness_path:
        reasons.append(
            "runtime witness path drifts away from the canonical `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` chain"
        )
    if not candidate.preserves_left_support_contract:
        reasons.append("left-support preservation must remain True")
    if not candidate.diagonal_preserved:
        reasons.append("diagonal invariance must remain True")
    if candidate.direct_covariance_edits_allowed:
        reasons.append("direct covariance edits must remain disallowed")
    if (
        candidate.required_patch_share_of_full_shared_vf_gap
        > contract_report.required_patch_share_of_full_shared_vf_gap + 1e-12
    ):
        reasons.append("shared-vf gap budget exceeds the canonical bounded witness cap")
    if (
        candidate.required_patch_share_of_omega_only_shared_vf_increment
        > contract_report.required_patch_share_of_omega_only_shared_vf_increment + 1e-12
    ):
        reasons.append(
            "omega-only shared-entry budget exceeds the canonical bounded witness cap"
        )
    if (
        candidate.required_patch_share_of_diagonal_omega_gap
        > contract_report.required_patch_share_of_diagonal_omega_gap + 1e-12
    ):
        reasons.append(
            "positive first-sine diagonal omega budget exceeds the canonical bounded witness cap"
        )
    if (
        candidate.required_patch_share_of_psd_boundary
        > contract_report.required_patch_share_of_psd_boundary + 1e-12
    ):
        reasons.append("PSD-boundary budget exceeds the canonical bounded witness cap")
    if candidate.compensating_stage_order != contract_report.compensating_stage_order:
        reasons.append(
            "compensating stage order drifts away from the diagonal-first ladder"
        )

    candidate_stage_cumulative = (
        candidate.compensating_cumulative_share_of_total_absolute_mass
    )
    canonical_stage_cumulative = (
        contract_report.compensating_cumulative_share_of_total_absolute_mass
    )
    if len(candidate_stage_cumulative) != len(canonical_stage_cumulative):
        reasons.append(
            "compensating stage ladder must preserve the canonical stage count"
        )
    else:
        stage_messages = (
            "top-two diagonal cumulative coverage falls below the canonical witness floor",
            "cross-shoulder cumulative coverage falls below the canonical witness floor",
            "left-center cumulative coverage falls below the canonical witness floor",
            "full compensating coverage must still close at 100.0%",
        )
        for candidate_value, canonical_value, message in zip(
            candidate_stage_cumulative,
            canonical_stage_cumulative,
            stage_messages,
            strict=True,
        ):
            if candidate_value + 1e-12 < canonical_value:
                reasons.append(message)

    if (
        candidate.compensating_zero_live_entry_count
        < contract_report.compensating_zero_live_entry_count
    ):
        reasons.append(
            "zero-live-entry contract falls below the canonical witness requirement"
        )
    return tuple(reasons)


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard_report(
    *,
    candidate: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
        | Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
        | Mapping[str, object]
    ),
    contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
        | None
    ) = None,
    candidate_input_mode_override: str | None = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidateGuardReport:
    resolved_contract = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
        if contract_report is None
        else contract_report
    )
    candidate_input_mode = (
        str(candidate_input_mode_override).strip()
        if candidate_input_mode_override is not None
        else _candidate_input_mode(candidate)
    )
    resolved_candidate = _coerce_candidate(candidate)
    rejection_reasons = _rejection_reasons(
        candidate=resolved_candidate,
        contract_report=resolved_contract,
    )
    accepted = not rejection_reasons

    budget_slacks = (
        resolved_contract.required_patch_share_of_full_shared_vf_gap
        - resolved_candidate.required_patch_share_of_full_shared_vf_gap,
        resolved_contract.required_patch_share_of_omega_only_shared_vf_increment
        - resolved_candidate.required_patch_share_of_omega_only_shared_vf_increment,
        resolved_contract.required_patch_share_of_diagonal_omega_gap
        - resolved_candidate.required_patch_share_of_diagonal_omega_gap,
        resolved_contract.required_patch_share_of_psd_boundary
        - resolved_candidate.required_patch_share_of_psd_boundary,
    )
    max_budget_share_slack = max(0.0, *(float(value) for value in budget_slacks))

    coverage_slacks: tuple[float, ...]
    if len(
        resolved_candidate.compensating_cumulative_share_of_total_absolute_mass
    ) == len(resolved_contract.compensating_cumulative_share_of_total_absolute_mass):
        coverage_slacks = tuple(
            float(candidate_value) - float(canonical_value)
            for candidate_value, canonical_value in zip(
                resolved_candidate.compensating_cumulative_share_of_total_absolute_mass,
                resolved_contract.compensating_cumulative_share_of_total_absolute_mass,
                strict=True,
            )
        )
        min_compensating_stage_coverage_slack = min(coverage_slacks)
    else:
        coverage_slacks = ()
        min_compensating_stage_coverage_slack = float("-inf")

    zero_live_entry_slack = int(
        resolved_candidate.compensating_zero_live_entry_count
        - resolved_contract.compensating_zero_live_entry_count
    )

    canonical_candidate = _canonical_candidate(resolved_contract)
    matches_canonical = (
        resolved_candidate.runtime_witness_path
        == canonical_candidate.runtime_witness_path
        and resolved_candidate.preserves_left_support_contract
        == canonical_candidate.preserves_left_support_contract
        and resolved_candidate.diagonal_preserved
        == canonical_candidate.diagonal_preserved
        and resolved_candidate.direct_covariance_edits_allowed
        == canonical_candidate.direct_covariance_edits_allowed
        and _isclose(
            resolved_candidate.required_patch_share_of_full_shared_vf_gap,
            canonical_candidate.required_patch_share_of_full_shared_vf_gap,
        )
        and _isclose(
            resolved_candidate.required_patch_share_of_omega_only_shared_vf_increment,
            canonical_candidate.required_patch_share_of_omega_only_shared_vf_increment,
        )
        and _isclose(
            resolved_candidate.required_patch_share_of_diagonal_omega_gap,
            canonical_candidate.required_patch_share_of_diagonal_omega_gap,
        )
        and _isclose(
            resolved_candidate.required_patch_share_of_psd_boundary,
            canonical_candidate.required_patch_share_of_psd_boundary,
        )
        and resolved_candidate.compensating_stage_order
        == canonical_candidate.compensating_stage_order
        and len(resolved_candidate.compensating_cumulative_share_of_total_absolute_mass)
        == len(canonical_candidate.compensating_cumulative_share_of_total_absolute_mass)
        and all(
            _isclose(candidate_value, canonical_value)
            for candidate_value, canonical_value in zip(
                resolved_candidate.compensating_cumulative_share_of_total_absolute_mass,
                canonical_candidate.compensating_cumulative_share_of_total_absolute_mass,
                strict=True,
            )
        )
        and resolved_candidate.compensating_zero_live_entry_count
        == canonical_candidate.compensating_zero_live_entry_count
    )

    if accepted:
        candidate_status = (
            "candidate-matches-runtime-contract"
            if matches_canonical
            else "candidate-tightens-within-runtime-contract"
        )
        resulting_contract_status = "runtime-witness-contract-satisfied"
    else:
        candidate_status = "candidate-rejected"
        resulting_contract_status = "runtime-witness-contract-rejected"

    if rejection_reasons:
        rejection_line = "- rejection reasons: " + ", ".join(
            f"`{reason}`" for reason in rejection_reasons
        )
    else:
        rejection_line = (
            "- rejection reasons: `none`; candidate stays within the bounded "
            "runtime-witness contract"
        )

    canonical_candidate_digest = (
        "- candidate input mode: "
        f"`{candidate_input_mode}`; runtime witness candidate status: "
        f"`{candidate_status}`; resulting contract status: "
        f"`{resulting_contract_status}`",
        "- canonical runtime witness path: `"
        + " -> ".join(resolved_contract.runtime_witness_path)
        + "`",
        "- canonical bounded witness caps: "
        f"shared-vf gap `{_format_percent(resolved_contract.required_patch_share_of_full_shared_vf_gap)}`; "
        f"omega-only shared-entry `{_format_percent(resolved_contract.required_patch_share_of_omega_only_shared_vf_increment)}`; "
        f"positive first-sine diagonal omega `{_format_percent(resolved_contract.required_patch_share_of_diagonal_omega_gap)}`; "
        f"PSD boundary `{_format_percent(resolved_contract.required_patch_share_of_psd_boundary)}`",
        "- canonical compensating ladder floors: "
        f"`{resolved_contract.compensating_stage_order[0]} >= {_format_percent(resolved_contract.compensating_cumulative_share_of_total_absolute_mass[0])}`; "
        f"`{resolved_contract.compensating_stage_order[1]} >= {_format_percent(resolved_contract.compensating_cumulative_share_of_total_absolute_mass[1])}`; "
        f"`{resolved_contract.compensating_stage_order[2]} >= {_format_percent(resolved_contract.compensating_cumulative_share_of_total_absolute_mass[2])}`; "
        f"`{resolved_contract.compensating_stage_order[3]} = {_format_percent(resolved_contract.compensating_cumulative_share_of_total_absolute_mass[3])}`; "
        f"`zero_live_entry_count >= {resolved_contract.compensating_zero_live_entry_count}`",
        rejection_line,
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidateGuardReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-runtime-witness-candidate-guard",
        accepted=accepted,
        candidate_input_mode=candidate_input_mode,
        candidate_status=candidate_status,
        resulting_contract_status=resulting_contract_status,
        policy_digest=resolved_contract.policy_digest,
        binding_design=resolved_contract.binding_design,
        window_label=resolved_contract.window_label,
        canonical_runtime_witness_path=resolved_contract.runtime_witness_path,
        candidate_runtime_witness_path=resolved_candidate.runtime_witness_path,
        rejection_reasons=rejection_reasons,
        max_budget_share_slack=max_budget_share_slack,
        min_compensating_stage_coverage_slack=min_compensating_stage_coverage_slack,
        zero_live_entry_slack=zero_live_entry_slack,
        canonical_candidate_digest=canonical_candidate_digest,
    )


@lru_cache(maxsize=1)
def _run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard_cached() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidateGuardReport
):
    contract_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard_report(
        candidate=_canonical_candidate(contract_report),
        contract_report=contract_report,
        candidate_input_mode_override="canonical-replay",
    )


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard(
    candidate: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
        | Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
        | Mapping[str, object]
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidateGuardReport:
    if candidate is None:
        return _run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard_cached()
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard_report(
        candidate=candidate,
        contract_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
        ),
    )
