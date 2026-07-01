from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    same_sign_diagonal_coordinate_count: int
    sign_flip_diagonal_coordinate_count: int
    anchor_diagonal_signed_mass: float
    companion_diagonal_signed_mass: float
    anchor_positive_diagonal_mass: float
    companion_positive_diagonal_mass: float
    anchor_negative_diagonal_mass: float
    companion_negative_diagonal_mass: float
    required_incremental_right_center_covariance_lift: float
    driver_signature: str
    canonical_basis_pair_diagonal_sign_digest: tuple[str, ...]

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
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.same_sign_diagonal_coordinate_count = int(
            self.same_sign_diagonal_coordinate_count
        )
        self.sign_flip_diagonal_coordinate_count = int(
            self.sign_flip_diagonal_coordinate_count
        )
        self.anchor_diagonal_signed_mass = float(self.anchor_diagonal_signed_mass)
        self.companion_diagonal_signed_mass = float(
            self.companion_diagonal_signed_mass
        )
        self.anchor_positive_diagonal_mass = float(self.anchor_positive_diagonal_mass)
        self.companion_positive_diagonal_mass = float(
            self.companion_positive_diagonal_mass
        )
        self.anchor_negative_diagonal_mass = float(self.anchor_negative_diagonal_mass)
        self.companion_negative_diagonal_mass = float(
            self.companion_negative_diagonal_mass
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_basis_pair_diagonal_sign_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_basis_pair_diagonal_sign_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": (
                self.failing_right_shoulder_grid_value
            ),
            "same_sign_diagonal_coordinate_count": (
                self.same_sign_diagonal_coordinate_count
            ),
            "sign_flip_diagonal_coordinate_count": (
                self.sign_flip_diagonal_coordinate_count
            ),
            "anchor_diagonal_signed_mass": self.anchor_diagonal_signed_mass,
            "companion_diagonal_signed_mass": self.companion_diagonal_signed_mass,
            "anchor_positive_diagonal_mass": self.anchor_positive_diagonal_mass,
            "companion_positive_diagonal_mass": (
                self.companion_positive_diagonal_mass
            ),
            "anchor_negative_diagonal_mass": self.anchor_negative_diagonal_mass,
            "companion_negative_diagonal_mass": (
                self.companion_negative_diagonal_mass
            ),
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "driver_signature": self.driver_signature,
            "canonical_basis_pair_diagonal_sign_digest": list(
                self.canonical_basis_pair_diagonal_sign_digest
            ),
        }


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_diagonal_sign_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairDiagonalSignReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-basis-pair-diagonal-sign-probe",
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
        center_grid_value=0.15,
        failing_right_shoulder_grid_value=0.25,
        same_sign_diagonal_coordinate_count=8,
        sign_flip_diagonal_coordinate_count=9,
        anchor_diagonal_signed_mass=-5.063117612185484,
        companion_diagonal_signed_mass=18.784551554874914,
        anchor_positive_diagonal_mass=5.722538611815516,
        companion_positive_diagonal_mass=30.345358735224007,
        anchor_negative_diagonal_mass=-10.785656224001,
        companion_negative_diagonal_mass=-11.560807180349093,
        required_incremental_right_center_covariance_lift=1.6361273081544214,
        driver_signature="same-sign-diagonal-support-deficit",
        canonical_basis_pair_diagonal_sign_digest=(
            "- basis-pair diagonal sign: bounded DGP2/500/50 near-zero grid keeps "
            "the right-center covariance shortfall on same-sign diagonal support",
            "- diagonal signed mass moves -5.063 -> +18.785 while positive "
            "diagonal support moves +5.723 -> +30.345",
            "- required right-center lift is +1.636, preserving the bounded "
            "runtime witness path before any fresh rerun admission",
        ),
    )
