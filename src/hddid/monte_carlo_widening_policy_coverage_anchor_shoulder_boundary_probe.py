from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_boundary_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBoundaryReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    hotspot_center: float
    safe_right_shoulder_grid_label: str
    safe_right_shoulder_grid_value: float
    failing_right_shoulder_grid_label: str
    failing_right_shoulder_grid_value: float
    bracket_width: float
    safe_right_shoulder_ratio: float
    failing_right_shoulder_ratio: float
    ratio_jump_across_bracket: float
    safe_right_shoulder_margin_to_one: float
    failing_right_shoulder_excess_over_one: float
    overshoot_companion_failing_ratio: float
    overshoot_companion_margin_to_one: float
    boundary_signature: str
    canonical_coverage_anchor_shoulder_boundary_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.hotspot_center = float(self.hotspot_center)
        self.safe_right_shoulder_grid_label = str(
            self.safe_right_shoulder_grid_label
        ).strip()
        self.safe_right_shoulder_grid_value = float(self.safe_right_shoulder_grid_value)
        self.failing_right_shoulder_grid_label = str(
            self.failing_right_shoulder_grid_label
        ).strip()
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.bracket_width = float(self.bracket_width)
        self.safe_right_shoulder_ratio = float(self.safe_right_shoulder_ratio)
        self.failing_right_shoulder_ratio = float(self.failing_right_shoulder_ratio)
        self.ratio_jump_across_bracket = float(self.ratio_jump_across_bracket)
        self.safe_right_shoulder_margin_to_one = float(
            self.safe_right_shoulder_margin_to_one
        )
        self.failing_right_shoulder_excess_over_one = float(
            self.failing_right_shoulder_excess_over_one
        )
        self.overshoot_companion_failing_ratio = float(
            self.overshoot_companion_failing_ratio
        )
        self.overshoot_companion_margin_to_one = float(
            self.overshoot_companion_margin_to_one
        )
        self.boundary_signature = str(self.boundary_signature).strip()
        self.canonical_coverage_anchor_shoulder_boundary_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_coverage_anchor_shoulder_boundary_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "hotspot_center": self.hotspot_center,
            "safe_right_shoulder_grid_label": self.safe_right_shoulder_grid_label,
            "safe_right_shoulder_grid_value": self.safe_right_shoulder_grid_value,
            "failing_right_shoulder_grid_label": self.failing_right_shoulder_grid_label,
            "failing_right_shoulder_grid_value": (
                self.failing_right_shoulder_grid_value
            ),
            "bracket_width": self.bracket_width,
            "safe_right_shoulder_ratio": self.safe_right_shoulder_ratio,
            "failing_right_shoulder_ratio": self.failing_right_shoulder_ratio,
            "ratio_jump_across_bracket": self.ratio_jump_across_bracket,
            "safe_right_shoulder_margin_to_one": (
                self.safe_right_shoulder_margin_to_one
            ),
            "failing_right_shoulder_excess_over_one": (
                self.failing_right_shoulder_excess_over_one
            ),
            "overshoot_companion_failing_ratio": (
                self.overshoot_companion_failing_ratio
            ),
            "overshoot_companion_margin_to_one": (
                self.overshoot_companion_margin_to_one
            ),
            "boundary_signature": self.boundary_signature,
            "canonical_coverage_anchor_shoulder_boundary_digest": list(
                self.canonical_coverage_anchor_shoulder_boundary_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_boundary_report(
    *,
    shoulder_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBoundaryReport:
    safe_replay = shoulder_report.replay("tight_center_grid")
    failing_replay = shoulder_report.replay("near_zero_grid")

    safe_right_value = float(safe_replay.evaluation_grid[-1])
    failing_right_value = float(failing_replay.evaluation_grid[-1])
    safe_ratio = float(safe_replay.coverage_anchor_right_shoulder_ratio)
    failing_ratio = float(failing_replay.coverage_anchor_right_shoulder_ratio)
    overshoot_companion_ratio = float(
        failing_replay.overshoot_companion_right_shoulder_ratio
    )
    bracket_width = failing_right_value - safe_right_value
    safe_margin = safe_ratio - 1.0
    failing_excess = failing_ratio - 1.0
    ratio_jump = failing_ratio - safe_ratio
    companion_margin = overshoot_companion_ratio - 1.0
    boundary_signature = "coverage-anchor-specific-bracketed-right-shoulder"

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*shoulder_report.binding_design)}`: coverage anchor seed "
        f"`{shoulder_report.coverage_anchor_random_state}` still clears the right "
        f"shoulder at `z = {_format_boundary_grid_value(safe_right_value)}` with "
        f"`error / half-interval = {_format_float(safe_ratio)}`, but misses at "
        f"`z = {_format_boundary_grid_value(failing_right_value)}` with "
        f"`{_format_float(failing_ratio)}`; the current Trigger 2 shoulder "
        f"boundary is therefore bracketed inside "
        f"`[{_format_boundary_grid_value(safe_right_value)}, "
        f"{_format_boundary_grid_value(failing_right_value)}]`",
        "- the same coverage anchor moves from safety margin "
        f"`{_format_float(safe_margin)}` at "
        f"`z = {_format_boundary_grid_value(safe_right_value)}` to overshoot "
        f"`+{_format_float(failing_excess)}` at "
        f"`z = {_format_boundary_grid_value(failing_right_value)}`, a ratio jump "
        f"of `{_format_float(ratio_jump)}` across a shoulder width of "
        f"`{_format_float(bracket_width)}`",
        "- overshoot companion seed "
        f"`{shoulder_report.overshoot_companion_random_state}` still covers the same "
        f"`z = {_format_boundary_grid_value(failing_right_value)}` shoulder at ratio "
        f"`{_format_float(overshoot_companion_ratio)}`, so this bracket remains "
        "coverage-anchor-specific rather than a generic right-shoulder failure",
        "- next Trigger 2 follow-up should treat "
        f"`{_format_design_key(*shoulder_report.binding_design)}` as a bracketed "
        "right-shoulder calibration boundary on "
        f"`[{_format_boundary_grid_value(safe_right_value)}, "
        f"{_format_boundary_grid_value(failing_right_value)}]`, not as center "
        "repair or uniform-critical retuning",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBoundaryReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-boundary-probe",
        policy_digest=shoulder_report.policy_digest,
        binding_design=shoulder_report.binding_design,
        coverage_anchor_random_state=shoulder_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            shoulder_report.overshoot_companion_random_state
        ),
        hotspot_center=shoulder_report.hotspot_center,
        safe_right_shoulder_grid_label=safe_replay.grid_label,
        safe_right_shoulder_grid_value=safe_right_value,
        failing_right_shoulder_grid_label=failing_replay.grid_label,
        failing_right_shoulder_grid_value=failing_right_value,
        bracket_width=bracket_width,
        safe_right_shoulder_ratio=safe_ratio,
        failing_right_shoulder_ratio=failing_ratio,
        ratio_jump_across_bracket=ratio_jump,
        safe_right_shoulder_margin_to_one=safe_margin,
        failing_right_shoulder_excess_over_one=failing_excess,
        overshoot_companion_failing_ratio=overshoot_companion_ratio,
        overshoot_companion_margin_to_one=companion_margin,
        boundary_signature=boundary_signature,
        canonical_coverage_anchor_shoulder_boundary_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_boundary_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBoundaryReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_boundary_report(
        shoulder_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_probe()
    )
