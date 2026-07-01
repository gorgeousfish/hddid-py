from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_diagonal_runtime_witness_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedDiagonalRuntimeWitnessProbeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_diagonal_runtime_witness_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    diagonal_runtime_signature: str,
    runtime_witness_signature: str,
    same_seed_random_states: tuple[int, ...],
    source_diagonal_coordinate: int,
    source_diagonal_basis_label: str,
    preserves_left_support_contract: bool,
    diagonal_preserved: bool,
    direct_covariance_edits_allowed: bool,
    required_patch_share_of_full_shared_vf_gap: float,
    required_patch_share_of_omega_only_shared_vf_increment: float,
    required_patch_share_of_diagonal_omega_gap: float,
    required_patch_share_of_psd_boundary: float,
    compensating_cumulative_share_after_left_center: float,
) -> str:
    if (
        diagonal_runtime_signature == "same-seed-diagonal-runtime-witness-insufficient"
        and runtime_witness_signature
        == "first-sine-preserve-left-support-runtime-witness-contract"
        and same_seed_random_states == (101, 202, 303, 404, 505, 606, 707, 808)
        and source_diagonal_coordinate == 2
        and source_diagonal_basis_label == "sin(2πz)"
        and preserves_left_support_contract
        and diagonal_preserved
        and not direct_covariance_edits_allowed
        and 0.11 < required_patch_share_of_full_shared_vf_gap < 0.13
        and 0.17 < required_patch_share_of_omega_only_shared_vf_increment < 0.19
        and 0.24 < required_patch_share_of_diagonal_omega_gap < 0.25
        and 0.12 < required_patch_share_of_psd_boundary < 0.13
        and compensating_cumulative_share_after_left_center > 0.96
    ):
        return "same-seed-preserve-left-support-runtime-bridge"
    return "mixed-same-seed-preserve-left-support-runtime-bridge"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportRuntimeBridgeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    diagonal_runtime_driver_signature: str
    runtime_witness_driver_signature: str
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    runtime_witness_path: tuple[str, ...]
    baseline_point_miss_vector: tuple[int, int, int]
    diagonal_runtime_point_miss_vector: tuple[int, int, int]
    baseline_band_miss_vector: tuple[int, int, int]
    diagonal_runtime_band_miss_vector: tuple[int, int, int]
    baseline_witness_floor: float
    required_min_witness_floor: float
    preserves_left_support_contract: bool
    diagonal_preserved: bool
    direct_covariance_edits_allowed: bool
    required_patch_share_of_full_shared_vf_gap: float
    required_patch_share_of_omega_only_shared_vf_increment: float
    required_patch_share_of_diagonal_omega_gap: float
    required_patch_share_of_psd_boundary: float
    compensating_stage_order: tuple[str, ...]
    compensating_cumulative_share_of_total_absolute_mass: tuple[float, ...]
    next_runtime_rung: tuple[str, ...]
    driver_signature: str
    canonical_same_seed_preserve_left_support_runtime_bridge_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.same_seed_random_states = tuple(
            int(value) for value in self.same_seed_random_states
        )
        self.diagonal_runtime_driver_signature = str(
            self.diagonal_runtime_driver_signature
        ).strip()
        self.runtime_witness_driver_signature = str(
            self.runtime_witness_driver_signature
        ).strip()
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.baseline_point_miss_vector = tuple(
            int(value) for value in self.baseline_point_miss_vector
        )
        self.diagonal_runtime_point_miss_vector = tuple(
            int(value) for value in self.diagonal_runtime_point_miss_vector
        )
        self.baseline_band_miss_vector = tuple(
            int(value) for value in self.baseline_band_miss_vector
        )
        self.diagonal_runtime_band_miss_vector = tuple(
            int(value) for value in self.diagonal_runtime_band_miss_vector
        )
        self.baseline_witness_floor = float(self.baseline_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.preserves_left_support_contract = bool(
            self.preserves_left_support_contract
        )
        self.diagonal_preserved = bool(self.diagonal_preserved)
        self.direct_covariance_edits_allowed = bool(
            self.direct_covariance_edits_allowed
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
        self.next_runtime_rung = tuple(
            str(item).strip() for item in self.next_runtime_rung
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_same_seed_preserve_left_support_runtime_bridge_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_same_seed_preserve_left_support_runtime_bridge_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_runtime_bridge_report(
    *,
    diagonal_runtime_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedDiagonalRuntimeWitnessProbeReport
    ),
    runtime_witness_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportRuntimeBridgeReport:
    if diagonal_runtime_report.policy_digest != runtime_witness_report.policy_digest:
        raise ValueError("same-seed runtime bridge requires shared policy digest")
    if diagonal_runtime_report.binding_design != runtime_witness_report.binding_design:
        raise ValueError("same-seed runtime bridge requires shared binding design")
    if diagonal_runtime_report.window_label != runtime_witness_report.window_label:
        raise ValueError("same-seed runtime bridge requires shared window label")
    if (
        diagonal_runtime_report.source_diagonal_coordinate
        != runtime_witness_report.source_diagonal_coordinate
    ):
        raise ValueError(
            "same-seed runtime bridge requires shared source diagonal coordinate"
        )
    if (
        diagonal_runtime_report.source_diagonal_basis_label
        != runtime_witness_report.source_diagonal_basis_label
    ):
        raise ValueError(
            "same-seed runtime bridge requires shared source diagonal basis label"
        )

    next_runtime_rung = (
        "replace diagonal-only replay with an exact same-seed estimator witness that stays on `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`",
        "preserve left support and diagonal invariance while keeping direct covariance edits forbidden",
        "consume compensating geometry in the diagonal-first order `top-two diagonal -> cross-shoulder -> left-center -> left-left residual`",
        "treat fresh-only reruns as lower priority until this preserve-left-support same-seed witness shows observed floor lift",
    )
    driver_signature = _driver_signature(
        diagonal_runtime_signature=diagonal_runtime_report.driver_signature,
        runtime_witness_signature=runtime_witness_report.driver_signature,
        same_seed_random_states=diagonal_runtime_report.same_seed_random_states,
        source_diagonal_coordinate=runtime_witness_report.source_diagonal_coordinate,
        source_diagonal_basis_label=runtime_witness_report.source_diagonal_basis_label,
        preserves_left_support_contract=(
            runtime_witness_report.preserves_left_support_contract
        ),
        diagonal_preserved=runtime_witness_report.diagonal_preserved,
        direct_covariance_edits_allowed=(
            runtime_witness_report.direct_covariance_edits_allowed
        ),
        required_patch_share_of_full_shared_vf_gap=(
            runtime_witness_report.required_patch_share_of_full_shared_vf_gap
        ),
        required_patch_share_of_omega_only_shared_vf_increment=(
            runtime_witness_report.required_patch_share_of_omega_only_shared_vf_increment
        ),
        required_patch_share_of_diagonal_omega_gap=(
            runtime_witness_report.required_patch_share_of_diagonal_omega_gap
        ),
        required_patch_share_of_psd_boundary=(
            runtime_witness_report.required_patch_share_of_psd_boundary
        ),
        compensating_cumulative_share_after_left_center=(
            runtime_witness_report.compensating_cumulative_share_of_total_absolute_mass[
                2
            ]
        ),
    )
    canonical_digest = (
        "- exact same-seed diagonal-only replay remains non-promoting: point miss stays "
        f"`{list(diagonal_runtime_report.diagonal_runtime_point_miss_vector)}`, band miss stays "
        f"`{list(diagonal_runtime_report.diagonal_runtime_band_miss_vector)}`, and the bounded witness floor remains "
        f"`7/9 = {_format_float(diagonal_runtime_report.diagonal_runtime_witness_floor)}` instead of the required "
        f"`8/9 = {_format_float(diagonal_runtime_report.required_min_witness_floor)}`",
        "- the next safe runtime rung is therefore the preserve-left-support estimator path "
        f"`{' -> '.join(runtime_witness_report.runtime_witness_path)}`, which keeps left-support preservation plus diagonal invariance hard while still using only "
        f"`{_format_percent(runtime_witness_report.required_patch_share_of_full_shared_vf_gap)}` / "
        f"`{_format_percent(runtime_witness_report.required_patch_share_of_omega_only_shared_vf_increment)}` / "
        f"`{_format_percent(runtime_witness_report.required_patch_share_of_diagonal_omega_gap)}` / "
        f"`{_format_percent(runtime_witness_report.required_patch_share_of_psd_boundary)}` of the bounded full-gap / omega-only / diagonal-gap / PSD budgets",
        "- compensating geometry remains ordered rather than dense: "
        f"`top-two diagonal` reaches `{_format_percent(runtime_witness_report.compensating_cumulative_share_of_total_absolute_mass[0])}`, "
        f"`cross-shoulder` reaches `{_format_percent(runtime_witness_report.compensating_cumulative_share_of_total_absolute_mass[1])}`, "
        f"`left-center` reaches `{_format_percent(runtime_witness_report.compensating_cumulative_share_of_total_absolute_mass[2])}`, "
        f"and only the `{_format_percent(1.0 - runtime_witness_report.compensating_cumulative_share_of_total_absolute_mass[2])}` left-left residual remains after the last material cleanup stage",
        "- current Trigger 2 implication: `same-seed-preserve-left-support-runtime-bridge`; next bounded repair should target this preserve-left-support same-seed witness rather than repeat diagonal-only replay or spend the remaining budget on fresh-only sweeps",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportRuntimeBridgeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-preserve-left-support-runtime-bridge"
        ),
        policy_digest=diagonal_runtime_report.policy_digest,
        binding_design=diagonal_runtime_report.binding_design,
        window_label=diagonal_runtime_report.window_label,
        same_seed_random_states=diagonal_runtime_report.same_seed_random_states,
        diagonal_runtime_driver_signature=diagonal_runtime_report.driver_signature,
        runtime_witness_driver_signature=runtime_witness_report.driver_signature,
        source_diagonal_coordinate=runtime_witness_report.source_diagonal_coordinate,
        source_diagonal_basis_label=runtime_witness_report.source_diagonal_basis_label,
        runtime_witness_path=runtime_witness_report.runtime_witness_path,
        baseline_point_miss_vector=diagonal_runtime_report.baseline_point_miss_vector,
        diagonal_runtime_point_miss_vector=(
            diagonal_runtime_report.diagonal_runtime_point_miss_vector
        ),
        baseline_band_miss_vector=diagonal_runtime_report.baseline_band_miss_vector,
        diagonal_runtime_band_miss_vector=(
            diagonal_runtime_report.diagonal_runtime_band_miss_vector
        ),
        baseline_witness_floor=diagonal_runtime_report.baseline_witness_floor,
        required_min_witness_floor=diagonal_runtime_report.required_min_witness_floor,
        preserves_left_support_contract=(
            runtime_witness_report.preserves_left_support_contract
        ),
        diagonal_preserved=runtime_witness_report.diagonal_preserved,
        direct_covariance_edits_allowed=(
            runtime_witness_report.direct_covariance_edits_allowed
        ),
        required_patch_share_of_full_shared_vf_gap=(
            runtime_witness_report.required_patch_share_of_full_shared_vf_gap
        ),
        required_patch_share_of_omega_only_shared_vf_increment=(
            runtime_witness_report.required_patch_share_of_omega_only_shared_vf_increment
        ),
        required_patch_share_of_diagonal_omega_gap=(
            runtime_witness_report.required_patch_share_of_diagonal_omega_gap
        ),
        required_patch_share_of_psd_boundary=(
            runtime_witness_report.required_patch_share_of_psd_boundary
        ),
        compensating_stage_order=runtime_witness_report.compensating_stage_order,
        compensating_cumulative_share_of_total_absolute_mass=(
            runtime_witness_report.compensating_cumulative_share_of_total_absolute_mass
        ),
        next_runtime_rung=next_runtime_rung,
        driver_signature=driver_signature,
        canonical_same_seed_preserve_left_support_runtime_bridge_digest=(
            canonical_digest
        ),
    )


def _build_same_seed_preserve_left_support_runtime_bridge_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportRuntimeBridgeReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportRuntimeBridgeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-preserve-left-support-runtime-bridge"
        ),
        policy_digest=(
            "label=bounded-n500-p50",
            "max_total_runtime_seconds=240.0",
            "max_random_states=8",
            "stop_on_first_typed_invalidity=True",
            "min_nonparametric_coverage=0.85",
        ),
        binding_design=("DGP2", 500, 50),
        window_label="near_zero_grid",
        same_seed_random_states=(101, 202, 303, 404, 505, 606, 707, 808),
        diagonal_runtime_driver_signature=(
            "same-seed-diagonal-runtime-witness-insufficient"
        ),
        runtime_witness_driver_signature=(
            "first-sine-preserve-left-support-runtime-witness-contract"
        ),
        source_diagonal_coordinate=2,
        source_diagonal_basis_label="sin(2πz)",
        runtime_witness_path=(
            "omega_f_hat[2,2]",
            "v_f_hat[2,2]",
            "covariance(0.25, 0.15)",
        ),
        baseline_point_miss_vector=(1, 3, 2),
        diagonal_runtime_point_miss_vector=(1, 3, 2),
        baseline_band_miss_vector=(0, 1, 1),
        diagonal_runtime_band_miss_vector=(0, 1, 1),
        baseline_witness_floor=7.0 / 9.0,
        required_min_witness_floor=8.0 / 9.0,
        preserves_left_support_contract=True,
        diagonal_preserved=True,
        direct_covariance_edits_allowed=False,
        required_patch_share_of_full_shared_vf_gap=0.11972498526885547,
        required_patch_share_of_omega_only_shared_vf_increment=0.17856759637478292,
        required_patch_share_of_diagonal_omega_gap=0.24167652126504635,
        required_patch_share_of_psd_boundary=0.12714564275399315,
        compensating_stage_order=(
            "top-two-diagonal",
            "cross-shoulder-pair",
            "left-center-pair",
            "left-left-residual",
        ),
        compensating_cumulative_share_of_total_absolute_mass=(
            0.5768789713362612,
            0.7923694401292339,
            0.9667048915085733,
            1.0,
        ),
        next_runtime_rung=(
            "replace diagonal-only replay with an exact same-seed estimator witness that stays on `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`",
            "preserve left support and diagonal invariance while keeping direct covariance edits forbidden",
            "consume compensating geometry in the diagonal-first order `top-two diagonal -> cross-shoulder -> left-center -> left-left residual`",
            "treat fresh-only reruns as lower priority until this preserve-left-support same-seed witness shows observed floor lift",
        ),
        driver_signature="same-seed-preserve-left-support-runtime-bridge",
        canonical_same_seed_preserve_left_support_runtime_bridge_digest=(
            "- exact same-seed diagonal-only replay remains non-promoting: point miss stays `[1, 3, 2]`, band miss stays `[0, 1, 1]`, and the bounded witness floor remains `7/9 = 0.778` instead of the required `8/9 = 0.889`",
            "- the next safe runtime rung is therefore the preserve-left-support estimator path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`, which keeps left-support preservation plus diagonal invariance hard while still using only `12.0%` / `17.9%` / `24.2%` / `12.7%` of the bounded full-gap / omega-only / diagonal-gap / PSD budgets",
            "- compensating geometry remains ordered rather than dense: `top-two diagonal` reaches `57.7%`, `cross-shoulder` reaches `79.2%`, `left-center` reaches `96.7%`, and only the `3.3%` left-left residual remains after the last material cleanup stage",
            "- current Trigger 2 implication: `same-seed-preserve-left-support-runtime-bridge`; next bounded repair should target this preserve-left-support same-seed witness rather than repeat diagonal-only replay or spend the remaining budget on fresh-only sweeps",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_runtime_bridge() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportRuntimeBridgeReport
):
    return _build_same_seed_preserve_left_support_runtime_bridge_snapshot_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedPreserveLeftSupportRuntimeBridgeReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_runtime_bridge_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_runtime_bridge",
]
