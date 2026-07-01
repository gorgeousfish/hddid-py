from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaDiagonalPriorityReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_ratio(value: float) -> str:
    return f"{_format_float(value)}x"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    diagonal_priority_signature: str,
    patch_fraction_signature: str,
    diagonal_replay_multiple_of_required_lift: float,
    offdiagonal_replay_multiple_of_required_lift: float,
    axis_replay_multiple_of_required_lift: float,
    required_patch_share_of_omega_diagonal_gap: float,
    required_patch_share_of_coordinate_axis_only_increment: float,
) -> str:
    if (
        diagonal_priority_signature == "first-sine-omega-diagonal-priority"
        and patch_fraction_signature
        == "bounded-positive-first-sine-omega-diagonal-patch-fraction"
        and diagonal_replay_multiple_of_required_lift > 4.0
        and offdiagonal_replay_multiple_of_required_lift > 1.0
        and axis_replay_multiple_of_required_lift > 5.0
        and required_patch_share_of_omega_diagonal_gap < 0.25
        and required_patch_share_of_coordinate_axis_only_increment < 0.2
    ):
        return "bounded-positive-first-sine-omega-replay-overshoot-guard"
    return "mixed-positive-first-sine-omega-replay"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaReplayOvershootGuardReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    shared_vf_entry_label: str
    required_diagonal_vf_entry_lift: float
    diagonal_only_increment: float
    offdiagonal_axis_only_increment: float
    coordinate_axis_only_increment: float
    diagonal_replay_multiple_of_required_lift: float
    offdiagonal_replay_multiple_of_required_lift: float
    axis_replay_multiple_of_required_lift: float
    diagonal_replay_overshoot_multiple_of_required_lift: float
    offdiagonal_replay_overshoot_multiple_of_required_lift: float
    axis_replay_overshoot_multiple_of_required_lift: float
    required_patch_share_of_omega_diagonal_gap: float
    required_patch_share_of_coordinate_axis_only_increment: float
    driver_signature: str
    canonical_positive_first_sine_omega_replay_overshoot_digest: tuple[str, ...]

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
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.diagonal_only_increment = float(self.diagonal_only_increment)
        self.offdiagonal_axis_only_increment = float(
            self.offdiagonal_axis_only_increment
        )
        self.coordinate_axis_only_increment = float(self.coordinate_axis_only_increment)
        self.diagonal_replay_multiple_of_required_lift = float(
            self.diagonal_replay_multiple_of_required_lift
        )
        self.offdiagonal_replay_multiple_of_required_lift = float(
            self.offdiagonal_replay_multiple_of_required_lift
        )
        self.axis_replay_multiple_of_required_lift = float(
            self.axis_replay_multiple_of_required_lift
        )
        self.diagonal_replay_overshoot_multiple_of_required_lift = float(
            self.diagonal_replay_overshoot_multiple_of_required_lift
        )
        self.offdiagonal_replay_overshoot_multiple_of_required_lift = float(
            self.offdiagonal_replay_overshoot_multiple_of_required_lift
        )
        self.axis_replay_overshoot_multiple_of_required_lift = float(
            self.axis_replay_overshoot_multiple_of_required_lift
        )
        self.required_patch_share_of_omega_diagonal_gap = float(
            self.required_patch_share_of_omega_diagonal_gap
        )
        self.required_patch_share_of_coordinate_axis_only_increment = float(
            self.required_patch_share_of_coordinate_axis_only_increment
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_positive_first_sine_omega_replay_overshoot_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_positive_first_sine_omega_replay_overshoot_digest
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
            "required_diagonal_vf_entry_lift": self.required_diagonal_vf_entry_lift,
            "diagonal_only_increment": self.diagonal_only_increment,
            "offdiagonal_axis_only_increment": self.offdiagonal_axis_only_increment,
            "coordinate_axis_only_increment": self.coordinate_axis_only_increment,
            "diagonal_replay_multiple_of_required_lift": self.diagonal_replay_multiple_of_required_lift,
            "offdiagonal_replay_multiple_of_required_lift": self.offdiagonal_replay_multiple_of_required_lift,
            "axis_replay_multiple_of_required_lift": self.axis_replay_multiple_of_required_lift,
            "diagonal_replay_overshoot_multiple_of_required_lift": self.diagonal_replay_overshoot_multiple_of_required_lift,
            "offdiagonal_replay_overshoot_multiple_of_required_lift": self.offdiagonal_replay_overshoot_multiple_of_required_lift,
            "axis_replay_overshoot_multiple_of_required_lift": self.axis_replay_overshoot_multiple_of_required_lift,
            "required_patch_share_of_omega_diagonal_gap": self.required_patch_share_of_omega_diagonal_gap,
            "required_patch_share_of_coordinate_axis_only_increment": self.required_patch_share_of_coordinate_axis_only_increment,
            "driver_signature": self.driver_signature,
            "canonical_positive_first_sine_omega_replay_overshoot_digest": list(
                self.canonical_positive_first_sine_omega_replay_overshoot_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_replay_overshoot_guard_report(
    *,
    diagonal_priority_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaDiagonalPriorityReport
    ),
    patch_fraction_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaReplayOvershootGuardReport:
    if diagonal_priority_report.policy_digest != patch_fraction_report.policy_digest:
        raise ValueError("omega replay overshoot guard requires shared policy digest")
    if diagonal_priority_report.binding_design != patch_fraction_report.binding_design:
        raise ValueError("omega replay overshoot guard requires shared binding design")
    if diagonal_priority_report.window_label != patch_fraction_report.window_label:
        raise ValueError("omega replay overshoot guard requires shared window label")
    if (
        diagonal_priority_report.coverage_anchor_random_state
        != patch_fraction_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        diagonal_priority_report.overshoot_companion_random_state
        != patch_fraction_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if (
        diagonal_priority_report.diagonal_coordinate
        != patch_fraction_report.diagonal_coordinate
    ):
        raise ValueError("diagonal coordinate must match across upstream reports")

    required_diagonal_vf_entry_lift = float(
        patch_fraction_report.required_diagonal_vf_entry_lift
    )
    if required_diagonal_vf_entry_lift <= 0.0:
        raise ValueError("required diagonal vf-entry lift must stay positive")

    diagonal_replay_multiple_of_required_lift = float(
        diagonal_priority_report.diagonal_only_multiple_of_required_lift
    )
    offdiagonal_replay_multiple_of_required_lift = float(
        diagonal_priority_report.offdiagonal_only_multiple_of_required_lift
    )
    axis_replay_multiple_of_required_lift = float(
        diagonal_priority_report.coordinate_axis_only_increment
        / required_diagonal_vf_entry_lift
    )
    diagonal_replay_overshoot_multiple_of_required_lift = float(
        diagonal_replay_multiple_of_required_lift - 1.0
    )
    offdiagonal_replay_overshoot_multiple_of_required_lift = float(
        offdiagonal_replay_multiple_of_required_lift - 1.0
    )
    axis_replay_overshoot_multiple_of_required_lift = float(
        axis_replay_multiple_of_required_lift - 1.0
    )
    required_patch_share_of_coordinate_axis_only_increment = float(
        required_diagonal_vf_entry_lift
        / diagonal_priority_report.coordinate_axis_only_increment
    )

    driver_signature = _driver_signature(
        diagonal_priority_signature=diagonal_priority_report.driver_signature,
        patch_fraction_signature=patch_fraction_report.driver_signature,
        diagonal_replay_multiple_of_required_lift=diagonal_replay_multiple_of_required_lift,
        offdiagonal_replay_multiple_of_required_lift=offdiagonal_replay_multiple_of_required_lift,
        axis_replay_multiple_of_required_lift=axis_replay_multiple_of_required_lift,
        required_patch_share_of_omega_diagonal_gap=(
            patch_fraction_report.required_patch_share_of_omega_diagonal_gap
        ),
        required_patch_share_of_coordinate_axis_only_increment=(
            required_patch_share_of_coordinate_axis_only_increment
        ),
    )

    canonical_digest = (
        f"- full positive `omega_f_hat[{patch_fraction_report.diagonal_coordinate},{patch_fraction_report.diagonal_coordinate}]` replay would overshoot the bounded shared-entry target `{patch_fraction_report.required_diagonal_vf_entry_lift:+.3f}` by `{_format_ratio(diagonal_replay_overshoot_multiple_of_required_lift)}`, because diagonal-only lift is already `{_format_ratio(diagonal_replay_multiple_of_required_lift)}` the required `{patch_fraction_report.shared_vf_entry_label}` repair",
        f"- replaying only the same coordinate-`{patch_fraction_report.diagonal_coordinate}` off-diagonal omega axis would still overshoot the same bounded target by `{_format_ratio(offdiagonal_replay_overshoot_multiple_of_required_lift)}`, while replaying the whole coordinate-`{patch_fraction_report.diagonal_coordinate}` axis would overshoot by `{_format_ratio(axis_replay_overshoot_multiple_of_required_lift)}`",
        f"- current Trigger 2 implication: `{driver_signature}`; the live repair must stay a bounded patch (`{_format_percent(patch_fraction_report.required_patch_share_of_omega_diagonal_gap)}` of the diagonal gap and `{_format_percent(required_patch_share_of_coordinate_axis_only_increment)}` of the full coordinate-axis replay), not any full diagonal or axis replay",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaReplayOvershootGuardReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-positive-first-sine-omega-replay-overshoot-guard",
        policy_digest=patch_fraction_report.policy_digest,
        binding_design=patch_fraction_report.binding_design,
        window_label=patch_fraction_report.window_label,
        coverage_anchor_random_state=patch_fraction_report.coverage_anchor_random_state,
        overshoot_companion_random_state=patch_fraction_report.overshoot_companion_random_state,
        diagonal_coordinate=patch_fraction_report.diagonal_coordinate,
        diagonal_basis_label=patch_fraction_report.diagonal_basis_label,
        shared_vf_entry_label=patch_fraction_report.shared_vf_entry_label,
        required_diagonal_vf_entry_lift=required_diagonal_vf_entry_lift,
        diagonal_only_increment=diagonal_priority_report.diagonal_only_increment,
        offdiagonal_axis_only_increment=diagonal_priority_report.offdiagonal_axis_only_increment,
        coordinate_axis_only_increment=diagonal_priority_report.coordinate_axis_only_increment,
        diagonal_replay_multiple_of_required_lift=diagonal_replay_multiple_of_required_lift,
        offdiagonal_replay_multiple_of_required_lift=offdiagonal_replay_multiple_of_required_lift,
        axis_replay_multiple_of_required_lift=axis_replay_multiple_of_required_lift,
        diagonal_replay_overshoot_multiple_of_required_lift=diagonal_replay_overshoot_multiple_of_required_lift,
        offdiagonal_replay_overshoot_multiple_of_required_lift=offdiagonal_replay_overshoot_multiple_of_required_lift,
        axis_replay_overshoot_multiple_of_required_lift=axis_replay_overshoot_multiple_of_required_lift,
        required_patch_share_of_omega_diagonal_gap=patch_fraction_report.required_patch_share_of_omega_diagonal_gap,
        required_patch_share_of_coordinate_axis_only_increment=required_patch_share_of_coordinate_axis_only_increment,
        driver_signature=driver_signature,
        canonical_positive_first_sine_omega_replay_overshoot_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_replay_overshoot_guard() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaReplayOvershootGuardReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_replay_overshoot_guard_report(
        diagonal_priority_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_diagonal_priority_probe(),
        patch_fraction_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract(),
    )
