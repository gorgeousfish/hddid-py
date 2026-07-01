from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .inference import InferenceComputationError
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_omega_axis_probe import (
    _coordinate_axis_counterfactual_omega,
    _offdiagonal_axis_counterfactual_omega,
)
from .validation import (
    _clone_design_with_evaluation_grid,
    _fit_phase7_nonparametric_object_payload,
    _match_runtime_probe_design,
    _phase7_runtime_probe_replication_seed,
    default_phase7_runtime_probe_designs,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _build_live_payload(seed: int):
    designs = default_phase7_runtime_probe_designs()
    base_design = _match_runtime_probe_design(
        designs,
        dgp_name="DGP2",
        n_obs=500,
        p=50,
    )
    design = _clone_design_with_evaluation_grid(
        base_design,
        evaluation_grid=(0.05, 0.15, 0.25),
    )
    replication_seed = _phase7_runtime_probe_replication_seed(
        random_state=int(seed),
        designs=designs,
        dgp_name="DGP2",
        n_obs=500,
        p=50,
    )
    _, payload = _fit_phase7_nonparametric_object_payload(
        design=design,
        replication_seed=replication_seed,
        n_boot=64,
    )
    return replication_seed, payload


def _driver_signature(
    *,
    live_companion_omega_diagonal_gap: float,
    live_diagonal_only_shared_vf_entry_increment: float,
    canonical_vs_live_target_gap: float,
) -> str:
    if (
        live_companion_omega_diagonal_gap < 0.0
        and live_diagonal_only_shared_vf_entry_increment < 0.0
        and canonical_vs_live_target_gap > 100.0
    ):
        return "live-source-target-alignment-drift"
    return "live-source-target-alignment-ok"


def _build_repo_side_invalidity_alignment_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport
):
    canonical_digest = (
        "- the current live seed-202/505 replay reaches typed Section 4 invalidity before a fresh payload can be built; preserving the last repo-side diagnostic source-target packet instead of treating the live run as successful",
        "- the preserved packet still marks the source lane as implementation-unsafe: `omega_f_hat[2,2]` target remains `1145.636`, the frozen canonical target remains `410.474`, and the repo-side target gap remains `+735.162`",
        "- current Trigger 2 implication: `live-source-target-alignment-drift`; core nonparametric inference must continue to raise on non-PSD `Omega_f`, while this diagnostic runner exposes a stable invalidity boundary for downstream release surfaces",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-live-source-target-alignment-probe",
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
        coverage_anchor_replication_seed=1751820809,
        overshoot_companion_replication_seed=37420655,
        source_diagonal_coordinate=2,
        source_diagonal_basis_label="sin(2πz)",
        shared_vf_entry_label="v_f_hat[2,2]",
        live_anchor_omega_diagonal_entry=950.1808400378586,
        live_companion_omega_diagonal_entry=201.25875574240965,
        live_companion_omega_diagonal_gap=-748.9220842954489,
        live_coordinate_axis_only_shared_vf_entry_increment=-2626.437177435239,
        live_diagonal_only_shared_vf_entry_increment=-3874.533428748402,
        live_offdiagonal_axis_only_shared_vf_entry_increment=1248.0962513131626,
        required_diagonal_vf_entry_lift=1011.1822863613054,
        live_required_omega_diagonal_increment=195.45495204282642,
        live_target_omega_diagonal_entry=1145.635792080685,
        canonical_target_omega_diagonal_entry=410.4739959618635,
        canonical_vs_live_target_gap=735.1617961188215,
        companion_gap_sign_matches_canonical=False,
        diagonal_lift_sign_matches_canonical=False,
        driver_signature="live-source-target-alignment-drift",
        canonical_live_alignment_digest=canonical_digest,
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    coverage_anchor_replication_seed: int
    overshoot_companion_replication_seed: int
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    shared_vf_entry_label: str
    live_anchor_omega_diagonal_entry: float
    live_companion_omega_diagonal_entry: float
    live_companion_omega_diagonal_gap: float
    live_coordinate_axis_only_shared_vf_entry_increment: float
    live_diagonal_only_shared_vf_entry_increment: float
    live_offdiagonal_axis_only_shared_vf_entry_increment: float
    required_diagonal_vf_entry_lift: float
    live_required_omega_diagonal_increment: float
    live_target_omega_diagonal_entry: float
    canonical_target_omega_diagonal_entry: float
    canonical_vs_live_target_gap: float
    companion_gap_sign_matches_canonical: bool
    diagonal_lift_sign_matches_canonical: bool
    driver_signature: str
    canonical_live_alignment_digest: tuple[str, ...]

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
        self.coverage_anchor_replication_seed = int(self.coverage_anchor_replication_seed)
        self.overshoot_companion_replication_seed = int(
            self.overshoot_companion_replication_seed
        )
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.live_anchor_omega_diagonal_entry = float(
            self.live_anchor_omega_diagonal_entry
        )
        self.live_companion_omega_diagonal_entry = float(
            self.live_companion_omega_diagonal_entry
        )
        self.live_companion_omega_diagonal_gap = float(
            self.live_companion_omega_diagonal_gap
        )
        self.live_coordinate_axis_only_shared_vf_entry_increment = float(
            self.live_coordinate_axis_only_shared_vf_entry_increment
        )
        self.live_diagonal_only_shared_vf_entry_increment = float(
            self.live_diagonal_only_shared_vf_entry_increment
        )
        self.live_offdiagonal_axis_only_shared_vf_entry_increment = float(
            self.live_offdiagonal_axis_only_shared_vf_entry_increment
        )
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.live_required_omega_diagonal_increment = float(
            self.live_required_omega_diagonal_increment
        )
        self.live_target_omega_diagonal_entry = float(
            self.live_target_omega_diagonal_entry
        )
        self.canonical_target_omega_diagonal_entry = float(
            self.canonical_target_omega_diagonal_entry
        )
        self.canonical_vs_live_target_gap = float(self.canonical_vs_live_target_gap)
        self.companion_gap_sign_matches_canonical = bool(
            self.companion_gap_sign_matches_canonical
        )
        self.diagonal_lift_sign_matches_canonical = bool(
            self.diagonal_lift_sign_matches_canonical
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_live_alignment_digest = tuple(
            str(line).rstrip() for line in self.canonical_live_alignment_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport
):
    source_bridge_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract()
    )
    patch_plan_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan()
    )

    anchor_replication_seed, anchor_payload = _build_live_payload(202)
    companion_replication_seed, companion_payload = _build_live_payload(505)

    anchor_omega = np.asarray(anchor_payload.omega_f_hat, dtype=float)
    companion_omega = np.asarray(companion_payload.omega_f_hat, dtype=float)
    anchor_vf = np.asarray(anchor_payload.v_f_hat, dtype=float)
    anchor_sigma = np.asarray(anchor_payload.sigma_f_hat, dtype=float)
    anchor_sigma_inverse = np.linalg.inv(anchor_sigma)

    diagonal_coordinate = int(source_bridge_report.source_diagonal_coordinate)
    live_anchor_omega_diagonal_entry = float(
        anchor_omega[diagonal_coordinate, diagonal_coordinate]
    )
    live_companion_omega_diagonal_entry = float(
        companion_omega[diagonal_coordinate, diagonal_coordinate]
    )
    live_companion_omega_diagonal_gap = float(
        live_companion_omega_diagonal_entry - live_anchor_omega_diagonal_entry
    )

    coordinate_axis_counterfactual_vf = (
        anchor_sigma_inverse
        @ _coordinate_axis_counterfactual_omega(
            anchor_omega_f_hat=anchor_omega,
            companion_omega_f_hat=companion_omega,
            coordinate=diagonal_coordinate,
        )
        @ anchor_sigma_inverse
    )
    live_coordinate_axis_only_shared_vf_entry_increment = float(
        coordinate_axis_counterfactual_vf[diagonal_coordinate, diagonal_coordinate]
        - anchor_vf[diagonal_coordinate, diagonal_coordinate]
    )

    diagonal_only_counterfactual_omega = np.array(anchor_omega, dtype=float, copy=True)
    diagonal_only_counterfactual_omega[diagonal_coordinate, diagonal_coordinate] = (
        live_companion_omega_diagonal_entry
    )
    diagonal_only_counterfactual_vf = (
        anchor_sigma_inverse
        @ diagonal_only_counterfactual_omega
        @ anchor_sigma_inverse
    )
    live_diagonal_only_shared_vf_entry_increment = float(
        diagonal_only_counterfactual_vf[diagonal_coordinate, diagonal_coordinate]
        - anchor_vf[diagonal_coordinate, diagonal_coordinate]
    )

    offdiagonal_axis_counterfactual_vf = (
        anchor_sigma_inverse
        @ _offdiagonal_axis_counterfactual_omega(
            anchor_omega_f_hat=anchor_omega,
            companion_omega_f_hat=companion_omega,
            coordinate=diagonal_coordinate,
        )
        @ anchor_sigma_inverse
    )
    live_offdiagonal_axis_only_shared_vf_entry_increment = float(
        offdiagonal_axis_counterfactual_vf[diagonal_coordinate, diagonal_coordinate]
        - anchor_vf[diagonal_coordinate, diagonal_coordinate]
    )

    required_diagonal_vf_entry_lift = float(
        patch_plan_report.signed_right_center_increment
        * int(anchor_payload.optimization_metadata["n_valid_obs"])
        / float(
            np.asarray(anchor_payload.evaluation_basis, dtype=float)[2, diagonal_coordinate]
            * np.asarray(anchor_payload.evaluation_basis, dtype=float)[1, diagonal_coordinate]
        )
    )
    live_required_omega_diagonal_increment = float(
        live_companion_omega_diagonal_gap
        * required_diagonal_vf_entry_lift
        / live_diagonal_only_shared_vf_entry_increment
    )
    live_target_omega_diagonal_entry = float(
        live_anchor_omega_diagonal_entry + live_required_omega_diagonal_increment
    )
    canonical_target_omega_diagonal_entry = float(
        source_bridge_report.source_target_omega_diagonal_entry
    )
    canonical_vs_live_target_gap = float(
        live_target_omega_diagonal_entry - canonical_target_omega_diagonal_entry
    )

    companion_gap_sign_matches_canonical = bool(
        live_companion_omega_diagonal_gap > 0.0
    )
    diagonal_lift_sign_matches_canonical = bool(
        live_diagonal_only_shared_vf_entry_increment > 0.0
    )
    driver_signature = _driver_signature(
        live_companion_omega_diagonal_gap=live_companion_omega_diagonal_gap,
        live_diagonal_only_shared_vf_entry_increment=(
            live_diagonal_only_shared_vf_entry_increment
        ),
        canonical_vs_live_target_gap=abs(canonical_vs_live_target_gap),
    )

    canonical_digest = (
        f"- the current live seed-202 payload no longer matches the frozen positive first-sine diagonal packet: `omega_f_hat[2,2]` is `{_format_float(live_anchor_omega_diagonal_entry)}` on replication seed `{anchor_replication_seed}`, while the archived seed-505 comparator is only `{_format_float(live_companion_omega_diagonal_entry)}` on replication seed `{companion_replication_seed}`, so the live companion gap is `{_format_signed(live_companion_omega_diagonal_gap)}` instead of a positive bounded replay budget",
        f"- with anchor `sigma_f_hat` frozen, swapping only the live seed-505 diagonal now moves shared `v_f_hat[2,2]` by `{_format_signed(live_diagonal_only_shared_vf_entry_increment)}`, so the diagonal-only lift has flipped sign even though the bounded right-center requirement still needs `+{_format_float(required_diagonal_vf_entry_lift)}`; the live bounded diagonal target therefore shifts up to `{_format_float(live_target_omega_diagonal_entry)}`, not the frozen `{_format_float(canonical_target_omega_diagonal_entry)}` packet",
        "- current Trigger 2 implication: `live-source-target-alignment-drift`; do not treat the archived diagonal ceiling as implementation-ready on this worktree until the live seed-202/505 payload pair is re-grounded and the positive first-sine source lane regains the expected sign",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-live-source-target-alignment-probe",
        policy_digest=source_bridge_report.policy_digest,
        binding_design=source_bridge_report.binding_design,
        window_label=source_bridge_report.window_label,
        coverage_anchor_random_state=source_bridge_report.coverage_anchor_random_state,
        overshoot_companion_random_state=source_bridge_report.overshoot_companion_random_state,
        coverage_anchor_replication_seed=anchor_replication_seed,
        overshoot_companion_replication_seed=companion_replication_seed,
        source_diagonal_coordinate=diagonal_coordinate,
        source_diagonal_basis_label=source_bridge_report.source_diagonal_basis_label,
        shared_vf_entry_label=source_bridge_report.source_shared_vf_entry_label,
        live_anchor_omega_diagonal_entry=live_anchor_omega_diagonal_entry,
        live_companion_omega_diagonal_entry=live_companion_omega_diagonal_entry,
        live_companion_omega_diagonal_gap=live_companion_omega_diagonal_gap,
        live_coordinate_axis_only_shared_vf_entry_increment=(
            live_coordinate_axis_only_shared_vf_entry_increment
        ),
        live_diagonal_only_shared_vf_entry_increment=(
            live_diagonal_only_shared_vf_entry_increment
        ),
        live_offdiagonal_axis_only_shared_vf_entry_increment=(
            live_offdiagonal_axis_only_shared_vf_entry_increment
        ),
        required_diagonal_vf_entry_lift=required_diagonal_vf_entry_lift,
        live_required_omega_diagonal_increment=live_required_omega_diagonal_increment,
        live_target_omega_diagonal_entry=live_target_omega_diagonal_entry,
        canonical_target_omega_diagonal_entry=canonical_target_omega_diagonal_entry,
        canonical_vs_live_target_gap=canonical_vs_live_target_gap,
        companion_gap_sign_matches_canonical=companion_gap_sign_matches_canonical,
        diagonal_lift_sign_matches_canonical=diagonal_lift_sign_matches_canonical,
        driver_signature=driver_signature,
        canonical_live_alignment_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport
):
    try:
        return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe_report()
    except InferenceComputationError:
        return _build_repo_side_invalidity_alignment_report()


__all__ = [
    "Phase7MonteCarloWideningPolicyCoverageAnchorShoulderLiveSourceTargetAlignmentProbeReport",
    "build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe_report",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe",
]
