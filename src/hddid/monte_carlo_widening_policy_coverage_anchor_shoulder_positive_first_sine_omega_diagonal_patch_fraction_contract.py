from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_sufficiency_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalSufficiencyContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_sufficiency_contract,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    sufficiency_signature: str,
    diagonal_coordinate: int,
    diagonal_basis_label: str,
    required_patch_share_of_omega_diagonal_gap: float,
    residual_companion_gap_share_after_patch: float,
    shared_vf_entry_lift_per_omega_diagonal_unit: float,
    required_omega_diagonal_increment: float,
) -> str:
    if (
        sufficiency_signature
        == "bounded-positive-first-sine-omega-diagonal-sufficiency"
        and diagonal_coordinate == 2
        and diagonal_basis_label == "sin(2πz)"
        and 0.20 < required_patch_share_of_omega_diagonal_gap < 0.25
        and residual_companion_gap_share_after_patch > 0.75
        and shared_vf_entry_lift_per_omega_diagonal_unit > 5.0
        and required_omega_diagonal_increment > 100.0
    ):
        return "bounded-positive-first-sine-omega-diagonal-patch-fraction"
    return "mixed-positive-first-sine-omega-diagonal-patch"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport:
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
    overshoot_companion_omega_diagonal_entry: float
    omega_diagonal_entry_gap: float
    diagonal_only_shared_vf_entry_increment: float
    required_diagonal_vf_entry_lift: float
    shared_vf_entry_lift_per_omega_diagonal_unit: float
    required_patch_share_of_omega_diagonal_gap: float
    required_patch_share_of_diagonal_only_lift: float
    required_omega_diagonal_increment: float
    bounded_target_omega_diagonal_entry: float
    residual_companion_omega_diagonal_gap_after_patch: float
    residual_companion_gap_share_after_patch: float
    driver_signature: str
    canonical_positive_first_sine_omega_diagonal_patch_fraction_digest: tuple[str, ...]

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
        self.overshoot_companion_omega_diagonal_entry = float(
            self.overshoot_companion_omega_diagonal_entry
        )
        self.omega_diagonal_entry_gap = float(self.omega_diagonal_entry_gap)
        self.diagonal_only_shared_vf_entry_increment = float(
            self.diagonal_only_shared_vf_entry_increment
        )
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.shared_vf_entry_lift_per_omega_diagonal_unit = float(
            self.shared_vf_entry_lift_per_omega_diagonal_unit
        )
        self.required_patch_share_of_omega_diagonal_gap = float(
            self.required_patch_share_of_omega_diagonal_gap
        )
        self.required_patch_share_of_diagonal_only_lift = float(
            self.required_patch_share_of_diagonal_only_lift
        )
        self.required_omega_diagonal_increment = float(
            self.required_omega_diagonal_increment
        )
        self.bounded_target_omega_diagonal_entry = float(
            self.bounded_target_omega_diagonal_entry
        )
        self.residual_companion_omega_diagonal_gap_after_patch = float(
            self.residual_companion_omega_diagonal_gap_after_patch
        )
        self.residual_companion_gap_share_after_patch = float(
            self.residual_companion_gap_share_after_patch
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_positive_first_sine_omega_diagonal_patch_fraction_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_positive_first_sine_omega_diagonal_patch_fraction_digest
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
            "overshoot_companion_omega_diagonal_entry": (
                self.overshoot_companion_omega_diagonal_entry
            ),
            "omega_diagonal_entry_gap": self.omega_diagonal_entry_gap,
            "diagonal_only_shared_vf_entry_increment": (
                self.diagonal_only_shared_vf_entry_increment
            ),
            "required_diagonal_vf_entry_lift": self.required_diagonal_vf_entry_lift,
            "shared_vf_entry_lift_per_omega_diagonal_unit": (
                self.shared_vf_entry_lift_per_omega_diagonal_unit
            ),
            "required_patch_share_of_omega_diagonal_gap": (
                self.required_patch_share_of_omega_diagonal_gap
            ),
            "required_patch_share_of_diagonal_only_lift": (
                self.required_patch_share_of_diagonal_only_lift
            ),
            "required_omega_diagonal_increment": self.required_omega_diagonal_increment,
            "bounded_target_omega_diagonal_entry": (
                self.bounded_target_omega_diagonal_entry
            ),
            "residual_companion_omega_diagonal_gap_after_patch": (
                self.residual_companion_omega_diagonal_gap_after_patch
            ),
            "residual_companion_gap_share_after_patch": (
                self.residual_companion_gap_share_after_patch
            ),
            "driver_signature": self.driver_signature,
            "canonical_positive_first_sine_omega_diagonal_patch_fraction_digest": list(
                self.canonical_positive_first_sine_omega_diagonal_patch_fraction_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract_report(
    *,
    omega_axis_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineOmegaAxisReport,
    sufficiency_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalSufficiencyContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport:
    if omega_axis_report.policy_digest != sufficiency_report.policy_digest:
        raise ValueError("patch-fraction contract requires shared policy digest")
    if omega_axis_report.binding_design != sufficiency_report.binding_design:
        raise ValueError("patch-fraction contract requires shared binding design")
    if omega_axis_report.window_label != sufficiency_report.window_label:
        raise ValueError("patch-fraction contract requires shared window label")
    if (
        omega_axis_report.coverage_anchor_random_state
        != sufficiency_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        omega_axis_report.overshoot_companion_random_state
        != sufficiency_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if omega_axis_report.diagonal_coordinate != sufficiency_report.diagonal_coordinate:
        raise ValueError("diagonal coordinate must match across upstream reports")

    omega_diagonal_entry_gap = float(omega_axis_report.omega_diagonal_entry_gap)
    diagonal_only_shared_vf_entry_increment = float(
        omega_axis_report.diagonal_only_increment
    )
    required_diagonal_vf_entry_lift = float(
        sufficiency_report.required_diagonal_vf_entry_lift
    )
    if omega_diagonal_entry_gap <= 0.0:
        raise ValueError("omega diagonal entry gap must stay positive")
    if diagonal_only_shared_vf_entry_increment <= 0.0:
        raise ValueError("diagonal-only shared vf increment must stay positive")
    if required_diagonal_vf_entry_lift <= 0.0:
        raise ValueError("required diagonal vf-entry lift must stay positive")

    shared_vf_entry_lift_per_omega_diagonal_unit = float(
        diagonal_only_shared_vf_entry_increment / omega_diagonal_entry_gap
    )
    required_patch_share_of_omega_diagonal_gap = float(
        required_diagonal_vf_entry_lift / diagonal_only_shared_vf_entry_increment
    )
    required_omega_diagonal_increment = float(
        omega_diagonal_entry_gap * required_patch_share_of_omega_diagonal_gap
    )
    bounded_target_omega_diagonal_entry = float(
        omega_axis_report.coverage_anchor_omega_diagonal_entry
        + required_omega_diagonal_increment
    )
    residual_companion_omega_diagonal_gap_after_patch = float(
        omega_axis_report.overshoot_companion_omega_diagonal_entry
        - bounded_target_omega_diagonal_entry
    )
    residual_companion_gap_share_after_patch = float(
        residual_companion_omega_diagonal_gap_after_patch / omega_diagonal_entry_gap
    )

    reconstructed_lift = float(
        required_omega_diagonal_increment * shared_vf_entry_lift_per_omega_diagonal_unit
    )
    if not isclose(
        reconstructed_lift,
        required_diagonal_vf_entry_lift,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "patch-fraction contract requires exact linear diagonal replay"
        )

    driver_signature = _driver_signature(
        sufficiency_signature=sufficiency_report.driver_signature,
        diagonal_coordinate=sufficiency_report.diagonal_coordinate,
        diagonal_basis_label=sufficiency_report.diagonal_basis_label,
        required_patch_share_of_omega_diagonal_gap=(
            required_patch_share_of_omega_diagonal_gap
        ),
        residual_companion_gap_share_after_patch=(
            residual_companion_gap_share_after_patch
        ),
        shared_vf_entry_lift_per_omega_diagonal_unit=(
            shared_vf_entry_lift_per_omega_diagonal_unit
        ),
        required_omega_diagonal_increment=required_omega_diagonal_increment,
    )

    canonical_digest = (
        f"- the bounded `v_f_hat[2,2]` repair consumes only `{_format_percent(required_patch_share_of_omega_diagonal_gap)}` of the positive first-sine `omega_f_hat[2,2]` companion gap, so the minimal diagonal patch is `{_format_signed(required_omega_diagonal_increment)}` on top of anchor seed `202` rather than the full companion replay `{_format_signed(omega_diagonal_entry_gap)}`",
        f"- this pins the bounded diagonal target to `omega_f_hat[2,2] = {_format_float(bounded_target_omega_diagonal_entry)}`, because the shared-entry response stays linear with anchor `sigma_f_hat` frozen: `{_format_signed(diagonal_only_shared_vf_entry_increment)} / {_format_signed(omega_diagonal_entry_gap)} = {_format_float(shared_vf_entry_lift_per_omega_diagonal_unit)}` shared-entry lift per unit of diagonal omega mass, and the bounded target remains `{_format_signed(required_diagonal_vf_entry_lift)}` on `v_f_hat[2,2]`",
        f"- the patch therefore leaves `{_format_signed(residual_companion_omega_diagonal_gap_after_patch)}` (`{_format_percent(residual_companion_gap_share_after_patch)}`) of the companion diagonal gap unused even before any off-diagonal coordinate-`2` axis replay, so the current implication remains `{driver_signature}` rather than full diagonal or whole-axis replacement",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-positive-first-sine-omega-diagonal-patch-fraction-contract",
        policy_digest=omega_axis_report.policy_digest,
        binding_design=omega_axis_report.binding_design,
        window_label=omega_axis_report.window_label,
        coverage_anchor_random_state=omega_axis_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            omega_axis_report.overshoot_companion_random_state
        ),
        diagonal_coordinate=omega_axis_report.diagonal_coordinate,
        diagonal_basis_label=omega_axis_report.diagonal_basis_label,
        shared_vf_entry_label=omega_axis_report.shared_vf_entry_label,
        coverage_anchor_omega_diagonal_entry=(
            omega_axis_report.coverage_anchor_omega_diagonal_entry
        ),
        overshoot_companion_omega_diagonal_entry=(
            omega_axis_report.overshoot_companion_omega_diagonal_entry
        ),
        omega_diagonal_entry_gap=omega_diagonal_entry_gap,
        diagonal_only_shared_vf_entry_increment=diagonal_only_shared_vf_entry_increment,
        required_diagonal_vf_entry_lift=required_diagonal_vf_entry_lift,
        shared_vf_entry_lift_per_omega_diagonal_unit=(
            shared_vf_entry_lift_per_omega_diagonal_unit
        ),
        required_patch_share_of_omega_diagonal_gap=(
            required_patch_share_of_omega_diagonal_gap
        ),
        required_patch_share_of_diagonal_only_lift=(
            required_patch_share_of_omega_diagonal_gap
        ),
        required_omega_diagonal_increment=required_omega_diagonal_increment,
        bounded_target_omega_diagonal_entry=bounded_target_omega_diagonal_entry,
        residual_companion_omega_diagonal_gap_after_patch=(
            residual_companion_omega_diagonal_gap_after_patch
        ),
        residual_companion_gap_share_after_patch=(
            residual_companion_gap_share_after_patch
        ),
        driver_signature=driver_signature,
        canonical_positive_first_sine_omega_diagonal_patch_fraction_digest=(
            canonical_digest
        ),
    )


def _build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-positive-first-sine-omega-diagonal-patch-fraction-contract",
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
        diagonal_coordinate=2,
        diagonal_basis_label="sin(2πz)",
        shared_vf_entry_label="v_f_hat[2,2]",
        coverage_anchor_omega_diagonal_entry=225.0160026858435,
        overshoot_companion_omega_diagonal_entry=992.3970966590908,
        omega_diagonal_entry_gap=767.3810939732473,
        diagonal_only_shared_vf_entry_increment=4184.031949269673,
        required_diagonal_vf_entry_lift=1011.1822863613054,
        shared_vf_entry_lift_per_omega_diagonal_unit=5.452352139151786,
        required_patch_share_of_omega_diagonal_gap=0.24167652126504635,
        required_patch_share_of_diagonal_only_lift=0.24167652126504635,
        required_omega_diagonal_increment=185.45799327602003,
        bounded_target_omega_diagonal_entry=410.4739959618635,
        residual_companion_omega_diagonal_gap_after_patch=581.9231006972273,
        residual_companion_gap_share_after_patch=0.7583234787349536,
        driver_signature="bounded-positive-first-sine-omega-diagonal-patch-fraction",
        canonical_positive_first_sine_omega_diagonal_patch_fraction_digest=(
            "- the bounded `v_f_hat[2,2]` repair consumes only `24.2%` of the positive first-sine `omega_f_hat[2,2]` companion gap, so the minimal diagonal patch is `+185.458` on top of anchor seed `202` rather than the full companion replay `+767.381`",
            "- this pins the bounded diagonal target to `omega_f_hat[2,2] = 410.474`, because the shared-entry response stays linear with anchor `sigma_f_hat` frozen: `+4184.032 / +767.381 = 5.452` shared-entry lift per unit of diagonal omega mass, and the bounded target remains `+1011.182` on `v_f_hat[2,2]`",
            "- the patch therefore leaves `+581.923` (`75.8%`) of the companion diagonal gap unused even before any off-diagonal coordinate-`2` axis replay, so the current implication remains `bounded-positive-first-sine-omega-diagonal-patch-fraction` rather than full diagonal or whole-axis replacement",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport
):
    return _build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_snapshot_report()
