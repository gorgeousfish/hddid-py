from __future__ import annotations

from dataclasses import dataclass


_POLICY_DIGEST = (
    "label=bounded-n500-p50",
    "max_total_runtime_seconds=240.0",
    "max_random_states=8",
    "stop_on_first_typed_invalidity=True",
    "min_nonparametric_coverage=0.85",
)


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    anchor_left_center_correlation: float
    anchor_cross_shoulder_correlation: float
    current_anchor_right_center_correlation: float
    required_repaired_right_center_correlation: float
    current_anchor_partial_cross_shoulder_correlation: float
    repaired_anchor_partial_cross_shoulder_correlation: float
    zero_partial_right_center_correlation: float
    zero_partial_lower_bound_gap: float
    required_repair_distance_to_zero_partial_target: float
    repaired_partial_abs_multiple_vs_current: float
    driver_signature: str
    canonical_direct_residual_feasibility_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        dgp, n_obs, p = self.binding_design
        self.binding_design = (str(dgp).strip(), int(n_obs), int(p))
        self.window_label = str(self.window_label).strip()
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.left_shoulder_grid_value = float(self.left_shoulder_grid_value)
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.anchor_left_center_correlation = float(
            self.anchor_left_center_correlation
        )
        self.anchor_cross_shoulder_correlation = float(
            self.anchor_cross_shoulder_correlation
        )
        self.current_anchor_right_center_correlation = float(
            self.current_anchor_right_center_correlation
        )
        self.required_repaired_right_center_correlation = float(
            self.required_repaired_right_center_correlation
        )
        self.current_anchor_partial_cross_shoulder_correlation = float(
            self.current_anchor_partial_cross_shoulder_correlation
        )
        self.repaired_anchor_partial_cross_shoulder_correlation = float(
            self.repaired_anchor_partial_cross_shoulder_correlation
        )
        self.zero_partial_right_center_correlation = float(
            self.zero_partial_right_center_correlation
        )
        self.zero_partial_lower_bound_gap = float(self.zero_partial_lower_bound_gap)
        self.required_repair_distance_to_zero_partial_target = float(
            self.required_repair_distance_to_zero_partial_target
        )
        self.repaired_partial_abs_multiple_vs_current = float(
            self.repaired_partial_abs_multiple_vs_current
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_direct_residual_feasibility_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_direct_residual_feasibility_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "left_shoulder_grid_value": self.left_shoulder_grid_value,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "anchor_left_center_correlation": self.anchor_left_center_correlation,
            "anchor_cross_shoulder_correlation": self.anchor_cross_shoulder_correlation,
            "current_anchor_right_center_correlation": self.current_anchor_right_center_correlation,
            "required_repaired_right_center_correlation": self.required_repaired_right_center_correlation,
            "current_anchor_partial_cross_shoulder_correlation": self.current_anchor_partial_cross_shoulder_correlation,
            "repaired_anchor_partial_cross_shoulder_correlation": self.repaired_anchor_partial_cross_shoulder_correlation,
            "zero_partial_right_center_correlation": self.zero_partial_right_center_correlation,
            "zero_partial_lower_bound_gap": self.zero_partial_lower_bound_gap,
            "required_repair_distance_to_zero_partial_target": self.required_repair_distance_to_zero_partial_target,
            "repaired_partial_abs_multiple_vs_current": self.repaired_partial_abs_multiple_vs_current,
            "driver_signature": self.driver_signature,
            "canonical_direct_residual_feasibility_digest": list(
                self.canonical_direct_residual_feasibility_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_report() -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport:
    digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: anchor seed `202` keeps left-center correlation `0.024` and raw cross-shoulder correlation `-0.082`; the bounded Trigger 2 repair lane only raises right-center correlation from `0.007` to `0.133` at failing shoulder `z = 0.25` versus center `z = 0.15`",
        "- with raw cross-shoulder correlation fixed, zeroing the center-conditioned direct residual would require right-center correlation `-3.399`, which sits `2.399` below the feasible correlation lower bound `-1.000` and `3.531` away from the bounded repair target `0.133`",
        "- bounded repair therefore cannot heal the direct residual: anchor partial cross-shoulder correlation stays negative at `-0.086` after repair versus `-0.082` today, i.e. the conditioned negative residual becomes `x1.046` as large in absolute value rather than disappearing",
        "- current Trigger 2 implication: `sign-healing-outside-correlation-feasible-repair-lane`; source-level follow-up should treat the negative direct residual as a fixed background constraint inside the bounded right-shoulder / center repair lane, not as a co-target of the current fix",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-direct-residual-feasibility-probe",
        policy_digest=_POLICY_DIGEST,
        binding_design=("DGP2", 500, 50),
        window_label="near_zero_grid",
        coverage_anchor_random_state=202,
        overshoot_companion_random_state=505,
        left_shoulder_grid_value=0.05,
        center_grid_value=0.15,
        failing_right_shoulder_grid_value=0.25,
        anchor_left_center_correlation=0.02402704342038043,
        anchor_cross_shoulder_correlation=-0.08165861359334536,
        current_anchor_right_center_correlation=0.007249260352146767,
        required_repaired_right_center_correlation=0.13276252551993098,
        current_anchor_partial_cross_shoulder_correlation=-0.0818585740406725,
        repaired_anchor_partial_cross_shoulder_correlation=-0.08563102308524738,
        zero_partial_right_center_correlation=-3.3986126451197145,
        zero_partial_lower_bound_gap=2.3986126451197145,
        required_repair_distance_to_zero_partial_target=3.5313751706396457,
        repaired_partial_abs_multiple_vs_current=1.0460849591968275,
        driver_signature="sign-healing-outside-correlation-feasible-repair-lane",
        canonical_direct_residual_feasibility_digest=digest,
    )


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_probe() -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderDirectResidualFeasibilityReport:
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_direct_residual_feasibility_report()
