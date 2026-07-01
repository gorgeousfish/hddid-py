from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_replay_overshoot_guard import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaReplayOvershootGuardReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_replay_overshoot_guard,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_ratio(value: float) -> str:
    return f"{float(value):.3f}x"


def _driver_signature(
    *,
    patch_fraction_signature: str,
    replay_guard_signature: str,
    source_bridge_signature: str,
    diagonal_coordinate: int,
    diagonal_basis_label: str,
    required_patch_share_of_omega_diagonal_gap: float,
    required_patch_share_of_coordinate_axis_only_increment: float,
    residual_companion_gap_share_after_ceiling: float,
    diagonal_replay_overshoot_multiple_of_required_lift: float,
    axis_replay_overshoot_multiple_of_required_lift: float,
) -> str:
    if (
        patch_fraction_signature
        == "bounded-positive-first-sine-omega-diagonal-patch-fraction"
        and replay_guard_signature
        == "bounded-positive-first-sine-omega-replay-overshoot-guard"
        and source_bridge_signature
        == "validation-patch-to-first-sine-omega-diagonal-bridge"
        and diagonal_coordinate == 2
        and diagonal_basis_label == "sin(2πz)"
        and 0.20 < required_patch_share_of_omega_diagonal_gap < 0.25
        and required_patch_share_of_coordinate_axis_only_increment < 0.20
        and residual_companion_gap_share_after_ceiling > 0.75
        and diagonal_replay_overshoot_multiple_of_required_lift > 3.0
        and axis_replay_overshoot_multiple_of_required_lift > 4.0
    ):
        return "bounded-positive-first-sine-omega-patch-ceiling"
    return "mixed-positive-first-sine-omega-patch-ceiling"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaPatchCeilingContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    shared_vf_entry_label: str
    coverage_anchor_omega_diagonal_entry: float
    bounded_target_omega_diagonal_entry: float
    overshoot_companion_omega_diagonal_entry: float
    required_omega_diagonal_increment: float
    required_patch_share_of_omega_diagonal_gap: float
    required_patch_share_of_coordinate_axis_only_increment: float
    residual_companion_omega_diagonal_gap_after_ceiling: float
    residual_companion_gap_share_after_ceiling: float
    diagonal_replay_overshoot_multiple_of_required_lift: float
    axis_replay_overshoot_multiple_of_required_lift: float
    driver_signature: str
    canonical_positive_first_sine_omega_patch_ceiling_digest: tuple[str, ...]

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
        self.diagonal_coordinate = int(self.diagonal_coordinate)
        self.diagonal_basis_label = str(self.diagonal_basis_label).strip()
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.coverage_anchor_omega_diagonal_entry = float(
            self.coverage_anchor_omega_diagonal_entry
        )
        self.bounded_target_omega_diagonal_entry = float(
            self.bounded_target_omega_diagonal_entry
        )
        self.overshoot_companion_omega_diagonal_entry = float(
            self.overshoot_companion_omega_diagonal_entry
        )
        self.required_omega_diagonal_increment = float(
            self.required_omega_diagonal_increment
        )
        self.required_patch_share_of_omega_diagonal_gap = float(
            self.required_patch_share_of_omega_diagonal_gap
        )
        self.required_patch_share_of_coordinate_axis_only_increment = float(
            self.required_patch_share_of_coordinate_axis_only_increment
        )
        self.residual_companion_omega_diagonal_gap_after_ceiling = float(
            self.residual_companion_omega_diagonal_gap_after_ceiling
        )
        self.residual_companion_gap_share_after_ceiling = float(
            self.residual_companion_gap_share_after_ceiling
        )
        self.diagonal_replay_overshoot_multiple_of_required_lift = float(
            self.diagonal_replay_overshoot_multiple_of_required_lift
        )
        self.axis_replay_overshoot_multiple_of_required_lift = float(
            self.axis_replay_overshoot_multiple_of_required_lift
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_positive_first_sine_omega_patch_ceiling_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_positive_first_sine_omega_patch_ceiling_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "diagonal_coordinate": self.diagonal_coordinate,
            "diagonal_basis_label": self.diagonal_basis_label,
            "shared_vf_entry_label": self.shared_vf_entry_label,
            "coverage_anchor_omega_diagonal_entry": self.coverage_anchor_omega_diagonal_entry,
            "bounded_target_omega_diagonal_entry": self.bounded_target_omega_diagonal_entry,
            "overshoot_companion_omega_diagonal_entry": self.overshoot_companion_omega_diagonal_entry,
            "required_omega_diagonal_increment": self.required_omega_diagonal_increment,
            "required_patch_share_of_omega_diagonal_gap": self.required_patch_share_of_omega_diagonal_gap,
            "required_patch_share_of_coordinate_axis_only_increment": (
                self.required_patch_share_of_coordinate_axis_only_increment
            ),
            "residual_companion_omega_diagonal_gap_after_ceiling": (
                self.residual_companion_omega_diagonal_gap_after_ceiling
            ),
            "residual_companion_gap_share_after_ceiling": (
                self.residual_companion_gap_share_after_ceiling
            ),
            "diagonal_replay_overshoot_multiple_of_required_lift": (
                self.diagonal_replay_overshoot_multiple_of_required_lift
            ),
            "axis_replay_overshoot_multiple_of_required_lift": (
                self.axis_replay_overshoot_multiple_of_required_lift
            ),
            "driver_signature": self.driver_signature,
            "canonical_positive_first_sine_omega_patch_ceiling_digest": list(
                self.canonical_positive_first_sine_omega_patch_ceiling_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_patch_ceiling_contract_report(
    *,
    patch_fraction_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport
    ),
    replay_overshoot_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaReplayOvershootGuardReport
    ),
    source_bridge_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchSourceBridgeContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaPatchCeilingContractReport:
    if patch_fraction_report.policy_digest != replay_overshoot_report.policy_digest:
        raise ValueError("patch ceiling contract requires shared policy digest")
    if patch_fraction_report.binding_design != replay_overshoot_report.binding_design:
        raise ValueError("patch ceiling contract requires shared binding design")
    if patch_fraction_report.window_label != replay_overshoot_report.window_label:
        raise ValueError("patch ceiling contract requires shared window label")
    if (
        patch_fraction_report.coverage_anchor_random_state
        != replay_overshoot_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        patch_fraction_report.overshoot_companion_random_state
        != replay_overshoot_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if (
        patch_fraction_report.diagonal_coordinate
        != replay_overshoot_report.diagonal_coordinate
    ):
        raise ValueError("diagonal coordinate must match across upstream reports")
    if patch_fraction_report.policy_digest != source_bridge_report.policy_digest:
        raise ValueError("patch ceiling contract requires source-bridge policy sync")
    if patch_fraction_report.binding_design != source_bridge_report.binding_design:
        raise ValueError("patch ceiling contract requires source-bridge design sync")
    if patch_fraction_report.window_label != source_bridge_report.window_label:
        raise ValueError("patch ceiling contract requires source-bridge window sync")
    if (
        patch_fraction_report.coverage_anchor_random_state
        != source_bridge_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match the source-bridge report")
    if (
        patch_fraction_report.overshoot_companion_random_state
        != source_bridge_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match the source-bridge report")
    if (
        patch_fraction_report.diagonal_coordinate
        != source_bridge_report.source_diagonal_coordinate
    ):
        raise ValueError(
            "source-bridge diagonal coordinate must match the bounded patch"
        )
    if (
        patch_fraction_report.shared_vf_entry_label
        != replay_overshoot_report.shared_vf_entry_label
    ):
        raise ValueError("shared vf-entry label must match the replay-overshoot report")
    if (
        patch_fraction_report.shared_vf_entry_label
        != source_bridge_report.source_shared_vf_entry_label
    ):
        raise ValueError("shared vf-entry label must match the source-bridge report")
    if not isclose(
        patch_fraction_report.required_omega_diagonal_increment,
        source_bridge_report.source_required_omega_diagonal_increment,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "source-bridge required omega increment drifted from bounded patch"
        )
    if not isclose(
        patch_fraction_report.bounded_target_omega_diagonal_entry,
        source_bridge_report.source_target_omega_diagonal_entry,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("source-bridge bounded target drifted from bounded patch")

    residual_companion_omega_diagonal_gap_after_ceiling = float(
        patch_fraction_report.overshoot_companion_omega_diagonal_entry
        - patch_fraction_report.bounded_target_omega_diagonal_entry
    )
    if residual_companion_omega_diagonal_gap_after_ceiling <= 0.0:
        raise ValueError(
            "bounded ceiling must remain strictly below the companion replay"
        )

    residual_companion_gap_share_after_ceiling = float(
        residual_companion_omega_diagonal_gap_after_ceiling
        / patch_fraction_report.omega_diagonal_entry_gap
    )

    driver_signature = _driver_signature(
        patch_fraction_signature=patch_fraction_report.driver_signature,
        replay_guard_signature=replay_overshoot_report.driver_signature,
        source_bridge_signature=source_bridge_report.driver_signature,
        diagonal_coordinate=patch_fraction_report.diagonal_coordinate,
        diagonal_basis_label=patch_fraction_report.diagonal_basis_label,
        required_patch_share_of_omega_diagonal_gap=(
            patch_fraction_report.required_patch_share_of_omega_diagonal_gap
        ),
        required_patch_share_of_coordinate_axis_only_increment=(
            replay_overshoot_report.required_patch_share_of_coordinate_axis_only_increment
        ),
        residual_companion_gap_share_after_ceiling=(
            residual_companion_gap_share_after_ceiling
        ),
        diagonal_replay_overshoot_multiple_of_required_lift=(
            replay_overshoot_report.diagonal_replay_overshoot_multiple_of_required_lift
        ),
        axis_replay_overshoot_multiple_of_required_lift=(
            replay_overshoot_report.axis_replay_overshoot_multiple_of_required_lift
        ),
    )

    canonical_digest = (
        f"- the bounded source ceiling stays explicit on `omega_f_hat[{patch_fraction_report.diagonal_coordinate},{patch_fraction_report.diagonal_coordinate}]`: anchor seed `{patch_fraction_report.coverage_anchor_random_state}` only needs `{_format_signed(patch_fraction_report.required_omega_diagonal_increment)}` on top of `{_format_float(patch_fraction_report.coverage_anchor_omega_diagonal_entry)}` up to `{_format_float(patch_fraction_report.bounded_target_omega_diagonal_entry)}`, which still consumes just `{_format_percent(patch_fraction_report.required_patch_share_of_omega_diagonal_gap)}` of the companion diagonal gap and `{_format_percent(replay_overshoot_report.required_patch_share_of_coordinate_axis_only_increment)}` of the full coordinate-`{patch_fraction_report.diagonal_coordinate}` axis replay",
        f"- this ceiling remains far below replay scale: companion `omega_f_hat[{patch_fraction_report.diagonal_coordinate},{patch_fraction_report.diagonal_coordinate}] = {_format_float(patch_fraction_report.overshoot_companion_omega_diagonal_entry)}` still leaves `{_format_signed(residual_companion_omega_diagonal_gap_after_ceiling)}` (`{_format_percent(residual_companion_gap_share_after_ceiling)}`) above the bounded ceiling, while full diagonal replay would overshoot the bounded `{patch_fraction_report.shared_vf_entry_label}` target by `{_format_ratio(replay_overshoot_report.diagonal_replay_overshoot_multiple_of_required_lift)}` and whole coordinate-axis replay would overshoot it by `{_format_ratio(replay_overshoot_report.axis_replay_overshoot_multiple_of_required_lift)}`",
        f"- current Trigger 2 implication: `{driver_signature}`; implementation should treat `omega_f_hat[{patch_fraction_report.diagonal_coordinate},{patch_fraction_report.diagonal_coordinate}] = {_format_float(patch_fraction_report.bounded_target_omega_diagonal_entry)}` as the bounded source-level ceiling, keep raw covariance edits validation-only, and leave full diagonal / off-diagonal axis / whole-axis replay outside the live repair lane",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaPatchCeilingContractReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-positive-first-sine-omega-patch-ceiling-contract",
        policy_digest=patch_fraction_report.policy_digest,
        binding_design=patch_fraction_report.binding_design,
        window_label=patch_fraction_report.window_label,
        coverage_anchor_random_state=patch_fraction_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            patch_fraction_report.overshoot_companion_random_state
        ),
        diagonal_coordinate=patch_fraction_report.diagonal_coordinate,
        diagonal_basis_label=patch_fraction_report.diagonal_basis_label,
        shared_vf_entry_label=patch_fraction_report.shared_vf_entry_label,
        coverage_anchor_omega_diagonal_entry=(
            patch_fraction_report.coverage_anchor_omega_diagonal_entry
        ),
        bounded_target_omega_diagonal_entry=(
            patch_fraction_report.bounded_target_omega_diagonal_entry
        ),
        overshoot_companion_omega_diagonal_entry=(
            patch_fraction_report.overshoot_companion_omega_diagonal_entry
        ),
        required_omega_diagonal_increment=(
            patch_fraction_report.required_omega_diagonal_increment
        ),
        required_patch_share_of_omega_diagonal_gap=(
            patch_fraction_report.required_patch_share_of_omega_diagonal_gap
        ),
        required_patch_share_of_coordinate_axis_only_increment=(
            replay_overshoot_report.required_patch_share_of_coordinate_axis_only_increment
        ),
        residual_companion_omega_diagonal_gap_after_ceiling=(
            residual_companion_omega_diagonal_gap_after_ceiling
        ),
        residual_companion_gap_share_after_ceiling=(
            residual_companion_gap_share_after_ceiling
        ),
        diagonal_replay_overshoot_multiple_of_required_lift=(
            replay_overshoot_report.diagonal_replay_overshoot_multiple_of_required_lift
        ),
        axis_replay_overshoot_multiple_of_required_lift=(
            replay_overshoot_report.axis_replay_overshoot_multiple_of_required_lift
        ),
        driver_signature=driver_signature,
        canonical_positive_first_sine_omega_patch_ceiling_digest=(canonical_digest),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_patch_ceiling_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaPatchCeilingContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_patch_ceiling_contract_report(
        patch_fraction_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract(),
        replay_overshoot_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_replay_overshoot_guard(),
        source_bridge_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract(),
    )


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaPatchCeilingContractReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_patch_ceiling_contract_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_patch_ceiling_contract",
]
