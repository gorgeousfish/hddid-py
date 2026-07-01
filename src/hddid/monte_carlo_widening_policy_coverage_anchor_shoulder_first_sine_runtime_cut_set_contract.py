from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_symmetry_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairSymmetryContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_symmetry_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_support_snapshot import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSupportSnapshotReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_support_snapshot,
)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _driver_signature(
    *,
    support_driver_signature: str,
    runtime_driver_signature: str,
    pair_cleanup_driver_signature: str,
    diagonal_coordinate: int,
    diagonal_basis_label: str,
    preserves_left_support_contract: bool,
    diagonal_preserved: bool,
    direct_covariance_edits_allowed: bool,
    required_patch_share_of_full_shared_vf_gap: float,
    required_patch_share_of_diagonal_omega_gap: float,
    off_diagonal_share_of_total_absolute_mass: float,
    cross_shoulder_pair_directional_imbalance: float,
    left_center_pair_directional_imbalance: float,
) -> str:
    if (
        support_driver_signature == "positive-first-sine-omega-support-contract"
        and runtime_driver_signature
        == "first-sine-preserve-left-support-runtime-witness-contract"
        and pair_cleanup_driver_signature
        == "first-sine-compensating-geometry-pair-symmetry-contract"
        and diagonal_coordinate == 2
        and diagonal_basis_label == "sin(2πz)"
        and preserves_left_support_contract
        and diagonal_preserved
        and not direct_covariance_edits_allowed
        and 0.11 < required_patch_share_of_full_shared_vf_gap < 0.13
        and 0.24 < required_patch_share_of_diagonal_omega_gap < 0.25
        and off_diagonal_share_of_total_absolute_mass < 0.4
        and abs(cross_shoulder_pair_directional_imbalance) < 1e-12
        and abs(left_center_pair_directional_imbalance) < 1e-12
    ):
        return "first-sine-runtime-cut-set-contract"
    return "mixed-first-sine-runtime-cut-set-contract"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeCutSetContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    runtime_witness_path: tuple[str, ...]
    cut_set_labels: tuple[str, ...]
    support_driver_signature: str
    runtime_driver_signature: str
    pair_cleanup_driver_signature: str
    same_sign_positive_gap_share_of_diagonal_net_gap: float
    required_patch_share_of_full_shared_vf_gap: float
    required_patch_share_of_diagonal_omega_gap: float
    required_patch_share_of_psd_boundary: float
    off_diagonal_share_of_total_absolute_mass: float
    cross_shoulder_pair_share_of_off_diagonal_mass: float
    left_center_pair_share_of_off_diagonal_mass: float
    cross_shoulder_pair_directional_imbalance: float
    left_center_pair_directional_imbalance: float
    preserves_left_support_contract: bool
    diagonal_preserved: bool
    direct_covariance_edits_allowed: bool
    live_routing_status: str
    current_live_entry: str
    driver_signature: str
    canonical_first_sine_runtime_cut_set_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.diagonal_coordinate = int(self.diagonal_coordinate)
        self.diagonal_basis_label = str(self.diagonal_basis_label).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.cut_set_labels = tuple(str(item).strip() for item in self.cut_set_labels)
        self.support_driver_signature = str(self.support_driver_signature).strip()
        self.runtime_driver_signature = str(self.runtime_driver_signature).strip()
        self.pair_cleanup_driver_signature = str(
            self.pair_cleanup_driver_signature
        ).strip()
        self.same_sign_positive_gap_share_of_diagonal_net_gap = float(
            self.same_sign_positive_gap_share_of_diagonal_net_gap
        )
        self.required_patch_share_of_full_shared_vf_gap = float(
            self.required_patch_share_of_full_shared_vf_gap
        )
        self.required_patch_share_of_diagonal_omega_gap = float(
            self.required_patch_share_of_diagonal_omega_gap
        )
        self.required_patch_share_of_psd_boundary = float(
            self.required_patch_share_of_psd_boundary
        )
        self.off_diagonal_share_of_total_absolute_mass = float(
            self.off_diagonal_share_of_total_absolute_mass
        )
        self.cross_shoulder_pair_share_of_off_diagonal_mass = float(
            self.cross_shoulder_pair_share_of_off_diagonal_mass
        )
        self.left_center_pair_share_of_off_diagonal_mass = float(
            self.left_center_pair_share_of_off_diagonal_mass
        )
        self.cross_shoulder_pair_directional_imbalance = float(
            self.cross_shoulder_pair_directional_imbalance
        )
        self.left_center_pair_directional_imbalance = float(
            self.left_center_pair_directional_imbalance
        )
        self.preserves_left_support_contract = bool(
            self.preserves_left_support_contract
        )
        self.diagonal_preserved = bool(self.diagonal_preserved)
        self.direct_covariance_edits_allowed = bool(
            self.direct_covariance_edits_allowed
        )
        self.live_routing_status = str(self.live_routing_status).strip()
        self.current_live_entry = str(self.current_live_entry).strip()
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_runtime_cut_set_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_runtime_cut_set_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_cut_set_contract_report(
    *,
    support_snapshot_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSupportSnapshotReport
    ),
    runtime_witness_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSinePreserveLeftSupportRuntimeWitnessContractReport
    ),
    pair_symmetry_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineCompensatingGeometryPairSymmetryContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeCutSetContractReport:
    if support_snapshot_report.policy_digest != runtime_witness_report.policy_digest:
        raise ValueError("runtime cut-set contract requires shared policy digest")
    if support_snapshot_report.policy_digest != pair_symmetry_report.policy_digest:
        raise ValueError("runtime cut-set contract requires shared policy digest")
    if support_snapshot_report.binding_design != runtime_witness_report.binding_design:
        raise ValueError("runtime cut-set contract requires shared binding design")
    if support_snapshot_report.binding_design != pair_symmetry_report.binding_design:
        raise ValueError("runtime cut-set contract requires shared binding design")
    if support_snapshot_report.window_label != runtime_witness_report.window_label:
        raise ValueError("runtime cut-set contract requires shared window label")
    if support_snapshot_report.window_label != pair_symmetry_report.window_label:
        raise ValueError("runtime cut-set contract requires shared window label")
    if (
        support_snapshot_report.coverage_anchor_random_state
        != runtime_witness_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "runtime cut-set contract requires shared coverage-anchor seed"
        )
    if (
        support_snapshot_report.coverage_anchor_random_state
        != pair_symmetry_report.coverage_anchor_random_state
    ):
        raise ValueError(
            "runtime cut-set contract requires shared coverage-anchor seed"
        )
    if (
        support_snapshot_report.overshoot_companion_random_state
        != runtime_witness_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "runtime cut-set contract requires shared overshoot-companion seed"
        )
    if (
        support_snapshot_report.overshoot_companion_random_state
        != pair_symmetry_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "runtime cut-set contract requires shared overshoot-companion seed"
        )
    if (
        support_snapshot_report.diagonal_coordinate
        != runtime_witness_report.source_diagonal_coordinate
    ):
        raise ValueError("runtime cut-set contract requires shared diagonal coordinate")
    if (
        support_snapshot_report.diagonal_coordinate
        != pair_symmetry_report.diagonal_coordinate
    ):
        raise ValueError("runtime cut-set contract requires shared diagonal coordinate")
    if (
        support_snapshot_report.diagonal_basis_label
        != runtime_witness_report.source_diagonal_basis_label
    ):
        raise ValueError(
            "runtime cut-set contract requires shared diagonal basis label"
        )
    if (
        support_snapshot_report.diagonal_basis_label
        != pair_symmetry_report.diagonal_basis_label
    ):
        raise ValueError(
            "runtime cut-set contract requires shared diagonal basis label"
        )

    cut_set_labels = (
        "positive-first-sine-support",
        "bounded-shared-vf-patch",
        "pair-symmetric-off-diagonal-cleanup",
        "no-direct-covariance-edits",
    )
    live_routing_status = "validation-only-first-sine-runtime-cut-set-contract"
    current_live_entry = "bounded-right-center-execution-contract"
    driver_signature = _driver_signature(
        support_driver_signature=support_snapshot_report.support_driver_signature,
        runtime_driver_signature=runtime_witness_report.driver_signature,
        pair_cleanup_driver_signature=pair_symmetry_report.driver_signature,
        diagonal_coordinate=support_snapshot_report.diagonal_coordinate,
        diagonal_basis_label=support_snapshot_report.diagonal_basis_label,
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
        required_patch_share_of_diagonal_omega_gap=(
            runtime_witness_report.required_patch_share_of_diagonal_omega_gap
        ),
        off_diagonal_share_of_total_absolute_mass=(
            pair_symmetry_report.off_diagonal_share_of_total_absolute_mass
        ),
        cross_shoulder_pair_directional_imbalance=(
            pair_symmetry_report.cross_shoulder_pair_directional_imbalance
        ),
        left_center_pair_directional_imbalance=(
            pair_symmetry_report.left_center_pair_directional_imbalance
        ),
    )
    canonical_digest = (
        f"- positive first-sine support still supplies the source activator: `coordinate {support_snapshot_report.diagonal_coordinate} = {support_snapshot_report.diagonal_basis_label}` keeps same-sign support at `{_format_percent(support_snapshot_report.same_sign_positive_gap_share_of_diagonal_net_gap)}` of the diagonal-net gap, while the bounded repair still needs only `{_format_percent(runtime_witness_report.required_patch_share_of_full_shared_vf_gap)}` of the full shared `v_f_hat[2,2]` gap and `{_format_percent(runtime_witness_report.required_patch_share_of_diagonal_omega_gap)}` of the positive diagonal omega gap",
        f"- preserve-left-support implementation still has to stay on the nested runtime path `{' -> '.join(runtime_witness_report.runtime_witness_path)}` with left-support preservation and diagonal invariance as hard obligations; direct covariance edits remain forbidden and the bounded patch still uses only `{_format_percent(runtime_witness_report.required_patch_share_of_psd_boundary)}` of PSD headroom",
        f"- off-live cleanup still has to remain mirrored and pair-local rather than unilateral: off-diagonal mass is only `{_format_percent(pair_symmetry_report.off_diagonal_share_of_total_absolute_mass)}` of total compensating mass, split `{_format_percent(pair_symmetry_report.cross_shoulder_pair_share_of_off_diagonal_mass)}` / `{_format_percent(pair_symmetry_report.left_center_pair_share_of_off_diagonal_mass)}` across the `cross-shoulder pair` and `left-center pair`, and both directional imbalances stay fixed at `{_format_float(pair_symmetry_report.cross_shoulder_pair_directional_imbalance)}`",
        "- current Trigger 2 implication: `first-sine-runtime-cut-set-contract`; any future preserve-left-support estimator witness must satisfy positive first-sine activation, bounded shared-entry lift, pair-symmetric off-diagonal cleanup, and the no-direct-covariance-edit rule together, while this contract remains validation-only companion evidence rather than a promoted live routing token",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeCutSetContractReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-runtime-cut-set-contract",
        policy_digest=support_snapshot_report.policy_digest,
        binding_design=support_snapshot_report.binding_design,
        window_label=support_snapshot_report.window_label,
        evaluation_grid=pair_symmetry_report.evaluation_grid,
        coverage_anchor_random_state=support_snapshot_report.coverage_anchor_random_state,
        overshoot_companion_random_state=support_snapshot_report.overshoot_companion_random_state,
        diagonal_coordinate=support_snapshot_report.diagonal_coordinate,
        diagonal_basis_label=support_snapshot_report.diagonal_basis_label,
        runtime_witness_path=runtime_witness_report.runtime_witness_path,
        cut_set_labels=cut_set_labels,
        support_driver_signature=support_snapshot_report.support_driver_signature,
        runtime_driver_signature=runtime_witness_report.driver_signature,
        pair_cleanup_driver_signature=pair_symmetry_report.driver_signature,
        same_sign_positive_gap_share_of_diagonal_net_gap=(
            support_snapshot_report.same_sign_positive_gap_share_of_diagonal_net_gap
        ),
        required_patch_share_of_full_shared_vf_gap=(
            runtime_witness_report.required_patch_share_of_full_shared_vf_gap
        ),
        required_patch_share_of_diagonal_omega_gap=(
            runtime_witness_report.required_patch_share_of_diagonal_omega_gap
        ),
        required_patch_share_of_psd_boundary=(
            runtime_witness_report.required_patch_share_of_psd_boundary
        ),
        off_diagonal_share_of_total_absolute_mass=(
            pair_symmetry_report.off_diagonal_share_of_total_absolute_mass
        ),
        cross_shoulder_pair_share_of_off_diagonal_mass=(
            pair_symmetry_report.cross_shoulder_pair_share_of_off_diagonal_mass
        ),
        left_center_pair_share_of_off_diagonal_mass=(
            pair_symmetry_report.left_center_pair_share_of_off_diagonal_mass
        ),
        cross_shoulder_pair_directional_imbalance=(
            pair_symmetry_report.cross_shoulder_pair_directional_imbalance
        ),
        left_center_pair_directional_imbalance=(
            pair_symmetry_report.left_center_pair_directional_imbalance
        ),
        preserves_left_support_contract=(
            runtime_witness_report.preserves_left_support_contract
        ),
        diagonal_preserved=runtime_witness_report.diagonal_preserved,
        direct_covariance_edits_allowed=(
            runtime_witness_report.direct_covariance_edits_allowed
        ),
        live_routing_status=live_routing_status,
        current_live_entry=current_live_entry,
        driver_signature=driver_signature,
        canonical_first_sine_runtime_cut_set_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_cut_set_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeCutSetContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_cut_set_contract_report(
        support_snapshot_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_support_snapshot()
        ),
        runtime_witness_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
        ),
        pair_symmetry_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_compensating_geometry_pair_symmetry_contract()
        ),
    )
