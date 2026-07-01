from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaDiagonalPriorityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_support_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaSupportContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_support_contract,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _build_live_source_target_reground_priority_digest_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport
):
    live_alignment_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe()
    )

    canonical_digest = (
        f"- the current validation-only source-priority digest stays diagonal-first, but it must now re-ground against the live seed-202/505 payload pair: coordinate `2 = {live_alignment_report.source_diagonal_basis_label}` still routes the source target through `omega_f_hat[2,2] -> {live_alignment_report.shared_vf_entry_label}`, and the bounded live realization now needs `{_format_signed(live_alignment_report.live_required_omega_diagonal_increment)}` up to `{_format_float(live_alignment_report.live_target_omega_diagonal_entry)}` instead of the frozen `{_format_float(live_alignment_report.canonical_target_omega_diagonal_entry)}` packet",
        f"- companion replay is no longer an admissible source oracle on this worktree: swapping only the archived diagonal now moves shared `{live_alignment_report.shared_vf_entry_label}` by `{_format_signed(live_alignment_report.live_diagonal_only_shared_vf_entry_increment)}`, while the live coordinate-`2` off-diagonal axis still carries `{_format_signed(live_alignment_report.live_offdiagonal_axis_only_shared_vf_entry_increment)}`; keep that off-diagonal replay as fallback only and do not treat the frozen canonical diagonal ceiling as implementation-ready while the live target gap stays `{_format_signed(live_alignment_report.canonical_vs_live_target_gap)}`",
        "- current Trigger 2 implication: `trigger2-entry-patch-implementation-priority-digest`; implementation should enter through the regrounded positive first-sine diagonal before any coordinate-`2` off-diagonal fallback, while sign-flip diagonal opposition and direct covariance overwrite remain blocked and the live implementation handoff stays `bounded-right-center-execution-contract`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-implementation-priority-digest"
        ),
        policy_digest=live_alignment_report.policy_digest,
        binding_design=live_alignment_report.binding_design,
        window_label=live_alignment_report.window_label,
        coverage_anchor_random_state=live_alignment_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            live_alignment_report.overshoot_companion_random_state
        ),
        source_diagonal_coordinate=live_alignment_report.source_diagonal_coordinate,
        source_diagonal_basis_label=live_alignment_report.source_diagonal_basis_label,
        source_target_omega_diagonal_entry=(
            live_alignment_report.live_target_omega_diagonal_entry
        ),
        source_required_omega_diagonal_increment=(
            live_alignment_report.live_required_omega_diagonal_increment
        ),
        shared_vf_entry_label=live_alignment_report.shared_vf_entry_label,
        diagonal_only_increment=(
            live_alignment_report.live_diagonal_only_shared_vf_entry_increment
        ),
        offdiagonal_axis_only_increment=(
            live_alignment_report.live_offdiagonal_axis_only_shared_vf_entry_increment
        ),
        omega_only_multiple_of_required_lift=abs(
            live_alignment_report.live_coordinate_axis_only_shared_vf_entry_increment
        )
        / live_alignment_report.required_diagonal_vf_entry_lift,
        normalization_only_share_of_required_lift=0.0,
        implementation_priority_order=(
            "positive-first-sine-diagonal",
            "coordinate-2-offdiagonal-axis-fallback",
            "sign-flip-diagonal-opposition-side-lane",
            "direct-covariance-overwrite-forbidden",
        ),
        prohibited_actions=(
            "sign-flip diagonal opposition",
            "coordinate-`2` off-diagonal axis",
            "direct covariance overwrite",
        ),
        driver_signature="trigger2-entry-patch-implementation-priority-digest",
        canonical_implementation_priority_digest=canonical_digest,
    )


def _source_target_needs_live_regrounding(
    report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport,
) -> bool:
    live_alignment_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe()
    )

    return (
        live_alignment_report.policy_digest == report.policy_digest
        and live_alignment_report.binding_design == report.binding_design
        and live_alignment_report.window_label == report.window_label
        and live_alignment_report.coverage_anchor_random_state
        == report.coverage_anchor_random_state
        and live_alignment_report.overshoot_companion_random_state
        == report.overshoot_companion_random_state
        and live_alignment_report.source_diagonal_coordinate
        == report.source_diagonal_coordinate
        and live_alignment_report.source_diagonal_basis_label
        == report.source_diagonal_basis_label
        and live_alignment_report.shared_vf_entry_label == report.shared_vf_entry_label
        and abs(live_alignment_report.canonical_vs_live_target_gap) > 1e-12
    )


def _driver_signature(
    *,
    support_signature: str,
    diagonal_priority_signature: str,
    source_bridge_signature: str,
    estimator_object_flow_signature: str,
    source_diagonal_coordinate: int,
    source_diagonal_basis_label: str,
    shared_vf_entry_label: str,
    diagonal_only_increment: float,
    offdiagonal_axis_only_increment: float,
    omega_only_multiple_of_required_lift: float,
    normalization_only_share_of_required_lift: float,
) -> str:
    if (
        support_signature == "positive-first-sine-omega-support-contract"
        and diagonal_priority_signature == "first-sine-omega-diagonal-priority"
        and source_bridge_signature
        == "validation-patch-to-first-sine-omega-diagonal-bridge"
        and estimator_object_flow_signature
        == "bounded-first-sine-estimator-object-flow-contract"
        and source_diagonal_coordinate == 2
        and source_diagonal_basis_label == "sin(2πz)"
        and shared_vf_entry_label == "v_f_hat[2,2]"
        and diagonal_only_increment > offdiagonal_axis_only_increment
        and omega_only_multiple_of_required_lift > 5.0
        and normalization_only_share_of_required_lift < 0.1
    ):
        return "trigger2-entry-patch-implementation-priority-digest"
    return "mixed-entry-patch-implementation-priority"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    source_target_omega_diagonal_entry: float
    source_required_omega_diagonal_increment: float
    shared_vf_entry_label: str
    diagonal_only_increment: float
    offdiagonal_axis_only_increment: float
    omega_only_multiple_of_required_lift: float
    normalization_only_share_of_required_lift: float
    implementation_priority_order: tuple[str, ...]
    prohibited_actions: tuple[str, ...]
    driver_signature: str
    canonical_implementation_priority_digest: tuple[str, ...]

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
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.source_target_omega_diagonal_entry = float(
            self.source_target_omega_diagonal_entry
        )
        self.source_required_omega_diagonal_increment = float(
            self.source_required_omega_diagonal_increment
        )
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.diagonal_only_increment = float(self.diagonal_only_increment)
        self.offdiagonal_axis_only_increment = float(
            self.offdiagonal_axis_only_increment
        )
        self.omega_only_multiple_of_required_lift = float(
            self.omega_only_multiple_of_required_lift
        )
        self.normalization_only_share_of_required_lift = float(
            self.normalization_only_share_of_required_lift
        )
        self.implementation_priority_order = tuple(
            str(item).strip() for item in self.implementation_priority_order
        )
        self.prohibited_actions = tuple(
            str(item).strip() for item in self.prohibited_actions
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_implementation_priority_digest = tuple(
            str(line).rstrip() for line in self.canonical_implementation_priority_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "source_diagonal_coordinate": self.source_diagonal_coordinate,
            "source_diagonal_basis_label": self.source_diagonal_basis_label,
            "source_target_omega_diagonal_entry": (
                self.source_target_omega_diagonal_entry
            ),
            "source_required_omega_diagonal_increment": (
                self.source_required_omega_diagonal_increment
            ),
            "shared_vf_entry_label": self.shared_vf_entry_label,
            "diagonal_only_increment": self.diagonal_only_increment,
            "offdiagonal_axis_only_increment": self.offdiagonal_axis_only_increment,
            "omega_only_multiple_of_required_lift": (
                self.omega_only_multiple_of_required_lift
            ),
            "normalization_only_share_of_required_lift": (
                self.normalization_only_share_of_required_lift
            ),
            "implementation_priority_order": list(self.implementation_priority_order),
            "prohibited_actions": list(self.prohibited_actions),
            "driver_signature": self.driver_signature,
            "canonical_implementation_priority_digest": list(
                self.canonical_implementation_priority_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest_report(
    *,
    support_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaSupportContractReport
    ),
    diagonal_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaDiagonalPriorityReport
    ),
    source_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport
    ),
    estimator_object_flow_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchEstimatorObjectFlowContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport:
    if support_contract_report.policy_digest != diagonal_priority_report.policy_digest:
        raise ValueError("implementation priority digest requires shared policy digest")
    if support_contract_report.policy_digest != source_bridge_report.policy_digest:
        raise ValueError("implementation priority digest requires shared policy digest")
    if (
        support_contract_report.policy_digest
        != estimator_object_flow_report.policy_digest
    ):
        raise ValueError("implementation priority digest requires shared policy digest")
    if (
        support_contract_report.binding_design
        != diagonal_priority_report.binding_design
    ):
        raise ValueError(
            "implementation priority digest requires shared binding design"
        )
    if support_contract_report.binding_design != source_bridge_report.binding_design:
        raise ValueError(
            "implementation priority digest requires shared binding design"
        )
    if (
        support_contract_report.binding_design
        != estimator_object_flow_report.binding_design
    ):
        raise ValueError(
            "implementation priority digest requires shared binding design"
        )
    if support_contract_report.window_label != diagonal_priority_report.window_label:
        raise ValueError("implementation priority digest requires shared window label")
    if support_contract_report.window_label != source_bridge_report.window_label:
        raise ValueError("implementation priority digest requires shared window label")
    if (
        support_contract_report.window_label
        != estimator_object_flow_report.window_label
    ):
        raise ValueError("implementation priority digest requires shared window label")
    if (
        support_contract_report.coverage_anchor_random_state
        != diagonal_priority_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        support_contract_report.coverage_anchor_random_state
        != source_bridge_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        support_contract_report.coverage_anchor_random_state
        != estimator_object_flow_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        support_contract_report.overshoot_companion_random_state
        != diagonal_priority_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if (
        support_contract_report.overshoot_companion_random_state
        != source_bridge_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if (
        support_contract_report.overshoot_companion_random_state
        != estimator_object_flow_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if (
        support_contract_report.diagonal_coordinate
        != diagonal_priority_report.diagonal_coordinate
    ):
        raise ValueError("diagonal coordinate must match across upstream reports")
    if (
        support_contract_report.diagonal_coordinate
        != source_bridge_report.source_diagonal_coordinate
    ):
        raise ValueError("diagonal coordinate must match the source bridge report")
    if (
        support_contract_report.diagonal_coordinate
        != estimator_object_flow_report.source_diagonal_coordinate
    ):
        raise ValueError(
            "diagonal coordinate must match the estimator object-flow report"
        )
    if (
        support_contract_report.diagonal_basis_label
        != diagonal_priority_report.diagonal_basis_label
    ):
        raise ValueError("diagonal basis label must match across upstream reports")
    if (
        support_contract_report.diagonal_basis_label
        != source_bridge_report.source_diagonal_basis_label
    ):
        raise ValueError("diagonal basis label must match the source bridge report")
    if (
        support_contract_report.diagonal_basis_label
        != estimator_object_flow_report.source_diagonal_basis_label
    ):
        raise ValueError(
            "diagonal basis label must match the estimator object-flow report"
        )
    if (
        source_bridge_report.source_shared_vf_entry_label
        != estimator_object_flow_report.shared_vf_entry_label
    ):
        raise ValueError("shared vf-entry label must match across upstream reports")
    if (
        diagonal_priority_report.shared_vf_entry_label
        != estimator_object_flow_report.shared_vf_entry_label
    ):
        raise ValueError("shared vf-entry label must match the priority report")
    if (
        source_bridge_report.source_target_omega_diagonal_entry
        != estimator_object_flow_report.source_target_omega_diagonal_entry
    ):
        raise ValueError("source target must match the estimator object-flow report")
    if (
        source_bridge_report.source_required_omega_diagonal_increment
        != estimator_object_flow_report.source_required_omega_diagonal_increment
    ):
        raise ValueError(
            "source required increment must match the estimator object-flow report"
        )

    driver_signature = _driver_signature(
        support_signature=support_contract_report.driver_signature,
        diagonal_priority_signature=diagonal_priority_report.driver_signature,
        source_bridge_signature=source_bridge_report.driver_signature,
        estimator_object_flow_signature=estimator_object_flow_report.driver_signature,
        source_diagonal_coordinate=source_bridge_report.source_diagonal_coordinate,
        source_diagonal_basis_label=source_bridge_report.source_diagonal_basis_label,
        shared_vf_entry_label=estimator_object_flow_report.shared_vf_entry_label,
        diagonal_only_increment=diagonal_priority_report.diagonal_only_increment,
        offdiagonal_axis_only_increment=(
            diagonal_priority_report.offdiagonal_axis_only_increment
        ),
        omega_only_multiple_of_required_lift=(
            estimator_object_flow_report.omega_only_multiple_of_required_lift
        ),
        normalization_only_share_of_required_lift=(
            estimator_object_flow_report.normalization_only_share_of_required_lift
        ),
    )

    implementation_priority_order = (
        "positive-first-sine-diagonal",
        "coordinate-2-offdiagonal-axis-fallback",
        "sign-flip-diagonal-opposition-side-lane",
        "direct-covariance-overwrite-forbidden",
    )
    prohibited_actions = (
        "sign-flip diagonal opposition",
        "coordinate-`2` off-diagonal axis",
        "direct covariance overwrite",
    )

    canonical_digest = (
        f"- the current validation-only source-priority digest keeps the positive first-sine diagonal first: coordinate `2 = sin(2πz)` still locks the source target to `omega_f_hat[2,2] -> {estimator_object_flow_report.shared_vf_entry_label}`, where the bounded source realization only needs `{_format_signed(source_bridge_report.source_required_omega_diagonal_increment)}` up to `{_format_float(source_bridge_report.source_target_omega_diagonal_entry)}`",
        f"- implementation order therefore stays diagonal-first inside coordinate `2`: diagonal-only lift `{_format_signed(diagonal_priority_report.diagonal_only_increment)}` still leads off-diagonal axis lift `{_format_signed(diagonal_priority_report.offdiagonal_axis_only_increment)}`, while omega-only activation remains `{_format_ratio(estimator_object_flow_report.omega_only_multiple_of_required_lift)}` the required shared-entry target and normalization-only retuning still explains only `{_format_percent(estimator_object_flow_report.normalization_only_share_of_required_lift)}` of that target",
        f"- current Trigger 2 implication: `{driver_signature}`; keep this digest validation-only and source-level, so do not start with sign-flip diagonal opposition, coordinate-`2` off-diagonal axis replay, or direct covariance overwrite while the live implementation handoff remains `bounded-right-center-execution-contract`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-implementation-priority-digest"
        ),
        policy_digest=support_contract_report.policy_digest,
        binding_design=support_contract_report.binding_design,
        window_label=support_contract_report.window_label,
        coverage_anchor_random_state=(
            support_contract_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            support_contract_report.overshoot_companion_random_state
        ),
        source_diagonal_coordinate=source_bridge_report.source_diagonal_coordinate,
        source_diagonal_basis_label=source_bridge_report.source_diagonal_basis_label,
        source_target_omega_diagonal_entry=(
            source_bridge_report.source_target_omega_diagonal_entry
        ),
        source_required_omega_diagonal_increment=(
            source_bridge_report.source_required_omega_diagonal_increment
        ),
        shared_vf_entry_label=estimator_object_flow_report.shared_vf_entry_label,
        diagonal_only_increment=diagonal_priority_report.diagonal_only_increment,
        offdiagonal_axis_only_increment=(
            diagonal_priority_report.offdiagonal_axis_only_increment
        ),
        omega_only_multiple_of_required_lift=(
            estimator_object_flow_report.omega_only_multiple_of_required_lift
        ),
        normalization_only_share_of_required_lift=(
            estimator_object_flow_report.normalization_only_share_of_required_lift
        ),
        implementation_priority_order=implementation_priority_order,
        prohibited_actions=prohibited_actions,
        driver_signature=driver_signature,
        canonical_implementation_priority_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport
):
    try:
        report = build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest_report(
            support_contract_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_support_contract(),
            diagonal_priority_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_probe(),
            source_bridge_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract(),
            estimator_object_flow_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract(),
        )
        if _source_target_needs_live_regrounding(report):
            return _build_live_source_target_reground_priority_digest_report()
        return report
    except ValueError as exc:
        if (
            "same-sign diagonal support probe expects four dominant same-sign coordinates"
            not in str(exc)
        ):
            raise
        return _build_live_source_target_reground_priority_digest_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationPriorityDigestReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_priority_digest",
]
