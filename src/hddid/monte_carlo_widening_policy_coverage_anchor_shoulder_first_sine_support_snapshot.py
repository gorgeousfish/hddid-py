from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSupportSnapshotReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    axis_driver_signature: str
    support_driver_signature: str
    sufficiency_driver_signature: str
    patch_fraction_driver_signature: str
    live_routing_status: str
    current_live_entry: str
    axis_share_of_omega_only_increment: float
    coordinate_axis_multiple_of_required_lift: float
    same_sign_positive_gap_share_of_diagonal_net_gap: float
    sign_flip_negative_gap_share_of_diagonal_net_gap: float
    required_lift_share_of_diagonal_only_increment: float
    diagonal_only_slack_multiple_of_required_lift: float
    required_patch_share_of_omega_diagonal_gap: float
    residual_companion_gap_share_after_patch: float
    canonical_first_sine_support_snapshot_digest: tuple[str, ...]

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
        self.axis_driver_signature = str(self.axis_driver_signature).strip()
        self.support_driver_signature = str(self.support_driver_signature).strip()
        self.sufficiency_driver_signature = str(
            self.sufficiency_driver_signature
        ).strip()
        self.patch_fraction_driver_signature = str(
            self.patch_fraction_driver_signature
        ).strip()
        self.live_routing_status = str(self.live_routing_status).strip()
        self.current_live_entry = str(self.current_live_entry).strip()
        self.axis_share_of_omega_only_increment = float(
            self.axis_share_of_omega_only_increment
        )
        self.coordinate_axis_multiple_of_required_lift = float(
            self.coordinate_axis_multiple_of_required_lift
        )
        self.same_sign_positive_gap_share_of_diagonal_net_gap = float(
            self.same_sign_positive_gap_share_of_diagonal_net_gap
        )
        self.sign_flip_negative_gap_share_of_diagonal_net_gap = float(
            self.sign_flip_negative_gap_share_of_diagonal_net_gap
        )
        self.required_lift_share_of_diagonal_only_increment = float(
            self.required_lift_share_of_diagonal_only_increment
        )
        self.diagonal_only_slack_multiple_of_required_lift = float(
            self.diagonal_only_slack_multiple_of_required_lift
        )
        self.required_patch_share_of_omega_diagonal_gap = float(
            self.required_patch_share_of_omega_diagonal_gap
        )
        self.residual_companion_gap_share_after_patch = float(
            self.residual_companion_gap_share_after_patch
        )
        self.canonical_first_sine_support_snapshot_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_support_snapshot_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_support_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSupportSnapshotReport
):
    policy_digest = (
        "label=bounded-n500-p50",
        "max_total_runtime_seconds=240.0",
        "max_random_states=8",
        "stop_on_first_typed_invalidity=True",
        "min_nonparametric_coverage=0.85",
    )
    binding_design = ("DGP2", 500, 50)
    digest = (
        "- coordinate-`2 = sin(2πz)` remains the source-level activation axis: swapping only that `omega_f_hat` row/column still explains "
        "`98.7%` of the omega-only lift on `v_f_hat[2,2]` and delivers `5.526x` the bounded repair target",
        "- the same diagonal lane remains support-led: same-sign positive support contributes "
        "`103.3%` of the diagonal-net improvement, while sign-flip negative mass still contributes only `-3.3%` of the same gap",
        "- bounded repair remains a small patch on top of that support lane: it needs only "
        "`24.2%` of the diagonal-only lift, leaves `3.138x` diagonal slack above the required target, and still leaves `75.8%` of the companion diagonal omega gap unused",
        "- current implication: keep this bundle as `validation-only-first-sine-support-snapshot` while the live Trigger 2 entry remains `trigger2-policy-spec`; this snapshot explains why seed `202` under-activates positive first-sine support without promoting the side lane into handoff or final-verification routing",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSupportSnapshotReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-support-snapshot"
        ),
        policy_digest=policy_digest,
        binding_design=binding_design,
        window_label="near_zero_grid",
        coverage_anchor_random_state=202,
        overshoot_companion_random_state=505,
        diagonal_coordinate=2,
        diagonal_basis_label="sin(2πz)",
        axis_driver_signature="first-sine-omega-axis-activation",
        support_driver_signature="positive-first-sine-omega-support-contract",
        sufficiency_driver_signature="bounded-positive-first-sine-omega-diagonal-sufficiency",
        patch_fraction_driver_signature="bounded-positive-first-sine-omega-diagonal-patch-fraction",
        live_routing_status="validation-only-first-sine-support-snapshot",
        current_live_entry="trigger2-policy-spec",
        axis_share_of_omega_only_increment=0.9866934067260297,
        coordinate_axis_multiple_of_required_lift=5.52560165874176,
        same_sign_positive_gap_share_of_diagonal_net_gap=1.0325042649207314,
        sign_flip_negative_gap_share_of_diagonal_net_gap=-0.032504264920731524,
        required_lift_share_of_diagonal_only_increment=0.24167652126504635,
        diagonal_only_slack_multiple_of_required_lift=3.1377623062659908,
        required_patch_share_of_omega_diagonal_gap=0.24167652126504635,
        residual_companion_gap_share_after_patch=0.7583234787349536,
        canonical_first_sine_support_snapshot_digest=digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_support_snapshot() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSupportSnapshotReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_support_snapshot_report()
