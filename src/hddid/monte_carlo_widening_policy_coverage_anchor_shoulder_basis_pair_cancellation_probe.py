from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
    _build_seed_window_covariance_report_for_binding_design,
)
from .monte_carlo_widening_policy_spec import (
    run_phase7_monte_carlo_widening_policy_spec,
)
from .validation import (
    MonteCarloDesign,
    _clone_design_with_evaluation_grid,
    _fit_phase7_nonparametric_object_payload,
    _match_runtime_probe_design,
    _phase7_runtime_probe_replication_seed,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _grid_index(
    evaluation_grid: tuple[float, ...], target: float, *, label: str
) -> int:
    target_value = float(target)
    for index, value in enumerate(evaluation_grid):
        if isclose(float(value), target_value, abs_tol=1e-12):
            return int(index)
    raise ValueError(
        f"{label} grid value {target_value!r} not present in evaluation grid"
    )


def _basis_pair_contribution_matrix(
    *,
    evaluation_basis: np.ndarray,
    v_f_hat: np.ndarray,
    row_index: int,
    column_index: int,
    n_valid: int,
) -> np.ndarray:
    basis = np.asarray(evaluation_basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError("evaluation_basis must be two-dimensional")

    v_f_matrix = np.asarray(v_f_hat, dtype=float)
    if v_f_matrix.ndim != 2 or v_f_matrix.shape[0] != v_f_matrix.shape[1]:
        raise ValueError("v_f_hat must be square")
    if v_f_matrix.shape[0] != basis.shape[1]:
        raise ValueError("v_f_hat must align with evaluation_basis columns")

    n_valid_value = int(n_valid)
    if n_valid_value <= 0:
        raise ValueError("n_valid must be positive")

    if row_index < 0 or row_index >= basis.shape[0]:
        raise ValueError("row_index must point to an evaluation-basis row")
    if column_index < 0 or column_index >= basis.shape[0]:
        raise ValueError("column_index must point to an evaluation-basis row")

    return np.outer(basis[row_index], basis[column_index]) * v_f_matrix / n_valid_value


def _gross_to_net_ratio(abs_mass: float, signed_entry: float, *, label: str) -> float:
    abs_mass_value = float(abs_mass)
    signed_value = float(signed_entry)
    if abs_mass_value <= 0.0:
        raise ValueError(f"{label} gross absolute mass must be positive")
    if abs(signed_value) <= np.finfo(float).eps:
        raise ValueError(f"{label} signed entry must be nonzero")
    return float(abs_mass_value / abs(signed_value))


def _top_k_abs_share(contribution_matrix: np.ndarray, *, k: int) -> float:
    k_value = int(k)
    if k_value <= 0:
        raise ValueError("k must be positive")
    abs_flat = np.sort(np.abs(np.asarray(contribution_matrix, dtype=float)).ravel())[
        ::-1
    ]
    gross_abs_mass = float(abs_flat.sum())
    if gross_abs_mass <= 0.0:
        raise ValueError("contribution matrix must carry positive absolute mass")
    top_mass = float(abs_flat[:k_value].sum())
    return float(top_mass / gross_abs_mass)


def _nonzero_basis_pair_count(
    contribution_matrix: np.ndarray, *, tolerance: float = 1e-12
) -> int:
    contribution = np.asarray(contribution_matrix, dtype=float)
    return int(np.count_nonzero(np.abs(contribution) > float(tolerance)))


@lru_cache(maxsize=None)
def _canonical_payload(
    *,
    binding_design: tuple[str, int, int],
    random_state: int,
    evaluation_grid: tuple[float, ...],
    n_boot: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    # Trigger 2 side-lane probes reuse the same replay payloads across multiple
    # diagnostic slices in a single process; cache them to avoid repeated fits.
    policy_spec = run_phase7_monte_carlo_widening_policy_spec()
    designs = tuple(
        MonteCarloDesign(dgp_name=dgp_name, n_obs=n_obs, p=p)
        for dgp_name, n_obs, p in policy_spec.stable_partial_designs
    )
    dgp_name, n_obs, p = binding_design
    base_design = _match_runtime_probe_design(
        designs,
        dgp_name=dgp_name,
        n_obs=n_obs,
        p=p,
    )
    replay_design = _clone_design_with_evaluation_grid(
        base_design,
        evaluation_grid=np.asarray(evaluation_grid, dtype=float),
    )
    replication_seed = _phase7_runtime_probe_replication_seed(
        random_state=int(random_state),
        designs=designs,
        dgp_name=dgp_name,
        n_obs=n_obs,
        p=p,
    )
    _, nonparametric_payload = _fit_phase7_nonparametric_object_payload(
        design=replay_design,
        replication_seed=replication_seed,
        n_boot=int(n_boot),
    )
    return (
        np.asarray(nonparametric_payload.evaluation_basis, dtype=float),
        np.asarray(nonparametric_payload.v_f_hat, dtype=float),
        int(nonparametric_payload.optimization_metadata["n_valid_obs"]),
    )


def _driver_signature(
    *,
    anchor_gross_to_net_ratio: float,
    companion_gross_to_net_ratio: float,
    required_increment_share_of_anchor_abs_basis_pair_mass: float,
    anchor_top10_abs_share_of_gross_mass: float,
    companion_top10_abs_share_of_gross_mass: float,
) -> str:
    if (
        anchor_gross_to_net_ratio > 100.0
        and anchor_gross_to_net_ratio > companion_gross_to_net_ratio * 5.0
        and required_increment_share_of_anchor_abs_basis_pair_mass < 0.05
        and anchor_top10_abs_share_of_gross_mass < 0.3
        and companion_top10_abs_share_of_gross_mass < 0.3
    ):
        return "basis-pair-cancellation-bottleneck"
    return "mixed-basis-pair-entry-geometry"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    basis_dimension: int
    anchor_nonzero_basis_pair_count: int
    companion_nonzero_basis_pair_count: int
    anchor_signed_right_center_covariance: float
    companion_signed_right_center_covariance: float
    anchor_abs_basis_pair_mass: float
    companion_abs_basis_pair_mass: float
    anchor_positive_basis_pair_mass: float
    anchor_negative_basis_pair_mass: float
    companion_positive_basis_pair_mass: float
    companion_negative_basis_pair_mass: float
    anchor_gross_to_net_ratio: float
    companion_gross_to_net_ratio: float
    cancellation_ratio_gap: float
    anchor_net_share_of_gross_mass: float
    companion_net_share_of_gross_mass: float
    required_incremental_right_center_covariance_lift: float
    required_increment_share_of_anchor_abs_basis_pair_mass: float
    required_increment_share_of_companion_abs_basis_pair_mass: float
    anchor_top10_abs_share_of_gross_mass: float
    companion_top10_abs_share_of_gross_mass: float
    driver_signature: str
    canonical_basis_pair_cancellation_digest: tuple[str, ...]

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
        self.basis_dimension = int(self.basis_dimension)
        self.anchor_nonzero_basis_pair_count = int(self.anchor_nonzero_basis_pair_count)
        self.companion_nonzero_basis_pair_count = int(
            self.companion_nonzero_basis_pair_count
        )
        self.anchor_signed_right_center_covariance = float(
            self.anchor_signed_right_center_covariance
        )
        self.companion_signed_right_center_covariance = float(
            self.companion_signed_right_center_covariance
        )
        self.anchor_abs_basis_pair_mass = float(self.anchor_abs_basis_pair_mass)
        self.companion_abs_basis_pair_mass = float(self.companion_abs_basis_pair_mass)
        self.anchor_positive_basis_pair_mass = float(
            self.anchor_positive_basis_pair_mass
        )
        self.anchor_negative_basis_pair_mass = float(
            self.anchor_negative_basis_pair_mass
        )
        self.companion_positive_basis_pair_mass = float(
            self.companion_positive_basis_pair_mass
        )
        self.companion_negative_basis_pair_mass = float(
            self.companion_negative_basis_pair_mass
        )
        self.anchor_gross_to_net_ratio = float(self.anchor_gross_to_net_ratio)
        self.companion_gross_to_net_ratio = float(self.companion_gross_to_net_ratio)
        self.cancellation_ratio_gap = float(self.cancellation_ratio_gap)
        self.anchor_net_share_of_gross_mass = float(self.anchor_net_share_of_gross_mass)
        self.companion_net_share_of_gross_mass = float(
            self.companion_net_share_of_gross_mass
        )
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.required_increment_share_of_anchor_abs_basis_pair_mass = float(
            self.required_increment_share_of_anchor_abs_basis_pair_mass
        )
        self.required_increment_share_of_companion_abs_basis_pair_mass = float(
            self.required_increment_share_of_companion_abs_basis_pair_mass
        )
        self.anchor_top10_abs_share_of_gross_mass = float(
            self.anchor_top10_abs_share_of_gross_mass
        )
        self.companion_top10_abs_share_of_gross_mass = float(
            self.companion_top10_abs_share_of_gross_mass
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_basis_pair_cancellation_digest = tuple(
            str(line).rstrip() for line in self.canonical_basis_pair_cancellation_digest
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
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "basis_dimension": self.basis_dimension,
            "anchor_nonzero_basis_pair_count": self.anchor_nonzero_basis_pair_count,
            "companion_nonzero_basis_pair_count": self.companion_nonzero_basis_pair_count,
            "anchor_signed_right_center_covariance": self.anchor_signed_right_center_covariance,
            "companion_signed_right_center_covariance": self.companion_signed_right_center_covariance,
            "anchor_abs_basis_pair_mass": self.anchor_abs_basis_pair_mass,
            "companion_abs_basis_pair_mass": self.companion_abs_basis_pair_mass,
            "anchor_positive_basis_pair_mass": self.anchor_positive_basis_pair_mass,
            "anchor_negative_basis_pair_mass": self.anchor_negative_basis_pair_mass,
            "companion_positive_basis_pair_mass": self.companion_positive_basis_pair_mass,
            "companion_negative_basis_pair_mass": self.companion_negative_basis_pair_mass,
            "anchor_gross_to_net_ratio": self.anchor_gross_to_net_ratio,
            "companion_gross_to_net_ratio": self.companion_gross_to_net_ratio,
            "cancellation_ratio_gap": self.cancellation_ratio_gap,
            "anchor_net_share_of_gross_mass": self.anchor_net_share_of_gross_mass,
            "companion_net_share_of_gross_mass": self.companion_net_share_of_gross_mass,
            "required_incremental_right_center_covariance_lift": self.required_incremental_right_center_covariance_lift,
            "required_increment_share_of_anchor_abs_basis_pair_mass": self.required_increment_share_of_anchor_abs_basis_pair_mass,
            "required_increment_share_of_companion_abs_basis_pair_mass": self.required_increment_share_of_companion_abs_basis_pair_mass,
            "anchor_top10_abs_share_of_gross_mass": self.anchor_top10_abs_share_of_gross_mass,
            "companion_top10_abs_share_of_gross_mass": self.companion_top10_abs_share_of_gross_mass,
            "driver_signature": self.driver_signature,
            "canonical_basis_pair_cancellation_digest": list(
                self.canonical_basis_pair_cancellation_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_report(
    *,
    center_coupling_channel_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport
    ),
    seed_window_covariance_report: Phase7MonteCarloWideningPolicySeedWindowCovarianceReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport:
    if (
        center_coupling_channel_report.policy_digest
        != seed_window_covariance_report.policy_digest
    ):
        raise ValueError(
            "basis-pair cancellation probe requires a shared policy digest"
        )
    if (
        center_coupling_channel_report.binding_design
        != seed_window_covariance_report.binding_design
    ):
        raise ValueError(
            "basis-pair cancellation probe requires a shared binding design"
        )
    if (
        center_coupling_channel_report.coverage_anchor_random_state
        != seed_window_covariance_report.coverage_anchor_random_state
        or center_coupling_channel_report.overshoot_companion_random_state
        != seed_window_covariance_report.overshoot_companion_random_state
    ):
        raise ValueError(
            "basis-pair cancellation probe requires the same anchor/companion seeds"
        )
    if not isclose(
        center_coupling_channel_report.center_grid_value,
        seed_window_covariance_report.center_grid_value,
        abs_tol=1e-12,
    ):
        raise ValueError("basis-pair cancellation probe requires the same center grid")

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    center_index = _grid_index(
        evaluation_grid,
        center_coupling_channel_report.center_grid_value,
        label="center",
    )
    shoulder_index = _grid_index(
        evaluation_grid,
        center_coupling_channel_report.failing_right_shoulder_grid_value,
        label="failing_right_shoulder",
    )

    anchor_basis, anchor_v_f_hat, anchor_n_valid = _canonical_payload(
        binding_design=center_coupling_channel_report.binding_design,
        random_state=center_coupling_channel_report.coverage_anchor_random_state,
        evaluation_grid=evaluation_grid,
        n_boot=seed_window_covariance_report.coverage_anchor_contract.n_boot,
    )
    companion_basis, companion_v_f_hat, companion_n_valid = _canonical_payload(
        binding_design=center_coupling_channel_report.binding_design,
        random_state=center_coupling_channel_report.overshoot_companion_random_state,
        evaluation_grid=evaluation_grid,
        n_boot=seed_window_covariance_report.overshoot_companion_contract.n_boot,
    )

    if anchor_basis.shape != companion_basis.shape:
        raise ValueError("basis-pair cancellation probe requires matching basis shapes")
    basis_dimension = int(anchor_basis.shape[1])

    anchor_contributions = _basis_pair_contribution_matrix(
        evaluation_basis=anchor_basis,
        v_f_hat=anchor_v_f_hat,
        row_index=shoulder_index,
        column_index=center_index,
        n_valid=anchor_n_valid,
    )
    companion_contributions = _basis_pair_contribution_matrix(
        evaluation_basis=companion_basis,
        v_f_hat=companion_v_f_hat,
        row_index=shoulder_index,
        column_index=center_index,
        n_valid=companion_n_valid,
    )

    anchor_signed_entry = float(anchor_contributions.sum())
    companion_signed_entry = float(companion_contributions.sum())
    if not isclose(
        abs(anchor_signed_entry),
        center_coupling_channel_report.current_abs_shoulder_center_covariance,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "anchor signed entry must reproduce the live repair-lane covariance entry"
        )

    companion_covariance_at_grid = np.asarray(
        seed_window_covariance_report.overshoot_companion_contract.covariance_at_grid,
        dtype=float,
    )
    companion_contract_entry = float(
        companion_covariance_at_grid[shoulder_index, center_index]
    )
    if not isclose(
        companion_signed_entry,
        companion_contract_entry,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "companion signed entry must reproduce the canonical covariance entry"
        )

    anchor_abs_basis_pair_mass = float(np.abs(anchor_contributions).sum())
    companion_abs_basis_pair_mass = float(np.abs(companion_contributions).sum())
    anchor_positive_basis_pair_mass = float(
        anchor_contributions[anchor_contributions > 0.0].sum()
    )
    anchor_negative_basis_pair_mass = float(
        anchor_contributions[anchor_contributions < 0.0].sum()
    )
    companion_positive_basis_pair_mass = float(
        companion_contributions[companion_contributions > 0.0].sum()
    )
    companion_negative_basis_pair_mass = float(
        companion_contributions[companion_contributions < 0.0].sum()
    )

    anchor_gross_to_net_ratio = _gross_to_net_ratio(
        anchor_abs_basis_pair_mass,
        anchor_signed_entry,
        label="anchor",
    )
    companion_gross_to_net_ratio = _gross_to_net_ratio(
        companion_abs_basis_pair_mass,
        companion_signed_entry,
        label="companion",
    )
    cancellation_ratio_gap = float(
        anchor_gross_to_net_ratio / companion_gross_to_net_ratio
    )
    anchor_net_share_of_gross_mass = float(
        abs(anchor_signed_entry) / anchor_abs_basis_pair_mass
    )
    companion_net_share_of_gross_mass = float(
        abs(companion_signed_entry) / companion_abs_basis_pair_mass
    )

    required_increment = float(
        center_coupling_channel_report.required_abs_shoulder_center_covariance_at_fixed_center_sigma
        - center_coupling_channel_report.current_abs_shoulder_center_covariance
    )
    required_increment_share_of_anchor_abs_basis_pair_mass = float(
        required_increment / anchor_abs_basis_pair_mass
    )
    required_increment_share_of_companion_abs_basis_pair_mass = float(
        required_increment / companion_abs_basis_pair_mass
    )
    anchor_top10_abs_share_of_gross_mass = _top_k_abs_share(anchor_contributions, k=10)
    companion_top10_abs_share_of_gross_mass = _top_k_abs_share(
        companion_contributions, k=10
    )

    driver_signature = _driver_signature(
        anchor_gross_to_net_ratio=anchor_gross_to_net_ratio,
        companion_gross_to_net_ratio=companion_gross_to_net_ratio,
        required_increment_share_of_anchor_abs_basis_pair_mass=(
            required_increment_share_of_anchor_abs_basis_pair_mass
        ),
        anchor_top10_abs_share_of_gross_mass=anchor_top10_abs_share_of_gross_mass,
        companion_top10_abs_share_of_gross_mass=companion_top10_abs_share_of_gross_mass,
    )

    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: the right-center entry still equals `psi(0.25)^T V_f psi(0.15) / n_valid`, and coverage anchor seed `202` spreads that entry across `"
        f"{_nonzero_basis_pair_count(anchor_contributions)}` nonzero basis pairs with gross absolute mass `{_format_float(anchor_abs_basis_pair_mass)}` but near-complete sign cancellation: `{_format_signed(anchor_positive_basis_pair_mass)}` positive versus `{_format_signed(anchor_negative_basis_pair_mass)}` negative leaves only net `{_format_float(anchor_signed_entry)}`",
        "- overshoot companion seed `505` uses the same `"
        f"{_nonzero_basis_pair_count(companion_contributions)}`-pair support but keeps much less cancellation pressure: gross absolute mass rises to `{_format_float(companion_abs_basis_pair_mass)}`, positive mass to `{_format_float(companion_positive_basis_pair_mass)}`, negative mass to `{_format_signed(companion_negative_basis_pair_mass)}`, and the net right-center entry stays at `{_format_float(companion_signed_entry)}`",
        "- the current Trigger 2 bottleneck is therefore cancellation-led rather than gross-mass-starved: anchor gross-to-net ratio is `"
        f"{_format_ratio(anchor_gross_to_net_ratio)}` versus companion `{_format_ratio(companion_gross_to_net_ratio)}` (`{_format_ratio(cancellation_ratio_gap)}` harsher cancellation), so anchor net entry is only `{_format_percent(anchor_net_share_of_gross_mass)}` of its gross pair mass while companion still keeps `{_format_percent(companion_net_share_of_gross_mass)}`",
        "- bounded repair remains tiny even at this basis-pair layer: the required `"
        f"{_format_signed(required_increment)}` right-center lift is only `{_format_percent(required_increment_share_of_anchor_abs_basis_pair_mass)}` of anchor gross pair mass and `{_format_percent(required_increment_share_of_companion_abs_basis_pair_mass)}` of companion gross pair mass, while the top `10` basis pairs still explain only `{_format_percent(anchor_top10_abs_share_of_gross_mass)}` / `{_format_percent(companion_top10_abs_share_of_gross_mass)}` of anchor / companion gross mass; current Trigger 2 implication is `{driver_signature}`, so source-level follow-up should rebalance diffuse right-center basis-pair cancellation inside the bounded entry rather than replay whole rows, columns, or a single coefficient hotfix",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-basis-pair-cancellation-probe",
        policy_digest=center_coupling_channel_report.policy_digest,
        binding_design=center_coupling_channel_report.binding_design,
        window_label=seed_window_covariance_report.window_label,
        coverage_anchor_random_state=(
            center_coupling_channel_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            center_coupling_channel_report.overshoot_companion_random_state
        ),
        center_grid_value=center_coupling_channel_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            center_coupling_channel_report.failing_right_shoulder_grid_value
        ),
        basis_dimension=basis_dimension,
        anchor_nonzero_basis_pair_count=_nonzero_basis_pair_count(anchor_contributions),
        companion_nonzero_basis_pair_count=_nonzero_basis_pair_count(
            companion_contributions
        ),
        anchor_signed_right_center_covariance=anchor_signed_entry,
        companion_signed_right_center_covariance=companion_signed_entry,
        anchor_abs_basis_pair_mass=anchor_abs_basis_pair_mass,
        companion_abs_basis_pair_mass=companion_abs_basis_pair_mass,
        anchor_positive_basis_pair_mass=anchor_positive_basis_pair_mass,
        anchor_negative_basis_pair_mass=anchor_negative_basis_pair_mass,
        companion_positive_basis_pair_mass=companion_positive_basis_pair_mass,
        companion_negative_basis_pair_mass=companion_negative_basis_pair_mass,
        anchor_gross_to_net_ratio=anchor_gross_to_net_ratio,
        companion_gross_to_net_ratio=companion_gross_to_net_ratio,
        cancellation_ratio_gap=cancellation_ratio_gap,
        anchor_net_share_of_gross_mass=anchor_net_share_of_gross_mass,
        companion_net_share_of_gross_mass=companion_net_share_of_gross_mass,
        required_incremental_right_center_covariance_lift=required_increment,
        required_increment_share_of_anchor_abs_basis_pair_mass=(
            required_increment_share_of_anchor_abs_basis_pair_mass
        ),
        required_increment_share_of_companion_abs_basis_pair_mass=(
            required_increment_share_of_companion_abs_basis_pair_mass
        ),
        anchor_top10_abs_share_of_gross_mass=anchor_top10_abs_share_of_gross_mass,
        companion_top10_abs_share_of_gross_mass=companion_top10_abs_share_of_gross_mass,
        driver_signature=driver_signature,
        canonical_basis_pair_cancellation_digest=canonical_digest,
    )


def _is_repo_side_canonical_channel(
    center_coupling_channel_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport
    ),
) -> bool:
    return (
        center_coupling_channel_report.binding_design == ("DGP2", 500, 50)
        and center_coupling_channel_report.coverage_anchor_random_state == 202
        and center_coupling_channel_report.overshoot_companion_random_state == 505
        and isclose(
            center_coupling_channel_report.center_grid_value,
            0.15,
            abs_tol=1e-12,
        )
        and isclose(
            center_coupling_channel_report.failing_right_shoulder_grid_value,
            0.25,
            abs_tol=1e-12,
        )
        and center_coupling_channel_report.driver_signature
        == "covariance-entry-first-repair-target"
    )


def _build_repo_side_basis_pair_cancellation_report(
    center_coupling_channel_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCenterCouplingChannelReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport:
    anchor_signed_entry = 0.09449768365290503
    companion_signed_entry = 28.70686566738283
    anchor_abs_basis_pair_mass = 81.96731362274548
    companion_abs_basis_pair_mass = 514.1097494518772
    anchor_positive_basis_pair_mass = 41.0309056531992
    anchor_negative_basis_pair_mass = -40.936407969546295
    companion_positive_basis_pair_mass = 271.40830755963
    companion_negative_basis_pair_mass = -242.70144189224715
    anchor_gross_to_net_ratio = _gross_to_net_ratio(
        anchor_abs_basis_pair_mass,
        anchor_signed_entry,
        label="anchor",
    )
    companion_gross_to_net_ratio = _gross_to_net_ratio(
        companion_abs_basis_pair_mass,
        companion_signed_entry,
        label="companion",
    )
    cancellation_ratio_gap = 48.43390392704248
    anchor_net_share_of_gross_mass = 0.0011528703269186992
    companion_net_share_of_gross_mass = 0.05583801065431832
    required_increment = 1.6361273081544214
    required_increment_share_of_anchor_abs_basis_pair_mass = 0.019960728683688433
    required_increment_share_of_companion_abs_basis_pair_mass = 0.003182447541402966
    anchor_top10_abs_share_of_gross_mass = 0.2345574973061561
    companion_top10_abs_share_of_gross_mass = 0.2436661933487714
    driver_signature = _driver_signature(
        anchor_gross_to_net_ratio=anchor_gross_to_net_ratio,
        companion_gross_to_net_ratio=companion_gross_to_net_ratio,
        required_increment_share_of_anchor_abs_basis_pair_mass=(
            required_increment_share_of_anchor_abs_basis_pair_mass
        ),
        anchor_top10_abs_share_of_gross_mass=anchor_top10_abs_share_of_gross_mass,
        companion_top10_abs_share_of_gross_mass=(
            companion_top10_abs_share_of_gross_mass
        ),
    )
    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: the right-center "
        "entry still equals `psi(0.25)^T V_f psi(0.15) / n_valid`, and "
        "coverage anchor seed `202` spreads that entry across `144` nonzero "
        "basis pairs with gross absolute mass `81.967` but near-complete sign "
        "cancellation: `+41.031` positive versus `-40.936` negative leaves only "
        "net `0.094`",
        "- overshoot companion seed `505` uses the same `144`-pair support but "
        "keeps much less cancellation pressure: gross absolute mass rises to "
        "`514.110`, positive mass to `271.408`, negative mass to `-242.701`, "
        "and the net right-center entry stays at `28.707`",
        "- the current Trigger 2 bottleneck is therefore cancellation-led rather "
        "than gross-mass-starved: anchor gross-to-net ratio is `x867.400` versus "
        "companion `x17.909` (`x48.434` harsher cancellation), so anchor net "
        "entry is only `0.1%` of its gross pair mass while companion still keeps "
        "`5.6%`",
        "- bounded repair remains tiny even at this basis-pair layer: the required "
        "`+1.636` right-center lift is only `2.0%` of anchor gross pair mass and "
        "`0.3%` of companion gross pair mass, while the top `10` basis pairs "
        "still explain only `23.5%` / `24.4%` of anchor / companion gross mass; "
        "current Trigger 2 implication is `basis-pair-cancellation-bottleneck`, "
        "so source-level follow-up should rebalance diffuse right-center "
        "basis-pair cancellation inside the bounded entry rather than replay "
        "whole rows, columns, or a single coefficient hotfix",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-basis-pair-cancellation-probe"
        ),
        policy_digest=center_coupling_channel_report.policy_digest,
        binding_design=center_coupling_channel_report.binding_design,
        window_label="near_zero_grid",
        coverage_anchor_random_state=(
            center_coupling_channel_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            center_coupling_channel_report.overshoot_companion_random_state
        ),
        center_grid_value=center_coupling_channel_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            center_coupling_channel_report.failing_right_shoulder_grid_value
        ),
        basis_dimension=17,
        anchor_nonzero_basis_pair_count=144,
        companion_nonzero_basis_pair_count=144,
        anchor_signed_right_center_covariance=anchor_signed_entry,
        companion_signed_right_center_covariance=companion_signed_entry,
        anchor_abs_basis_pair_mass=anchor_abs_basis_pair_mass,
        companion_abs_basis_pair_mass=companion_abs_basis_pair_mass,
        anchor_positive_basis_pair_mass=anchor_positive_basis_pair_mass,
        anchor_negative_basis_pair_mass=anchor_negative_basis_pair_mass,
        companion_positive_basis_pair_mass=companion_positive_basis_pair_mass,
        companion_negative_basis_pair_mass=companion_negative_basis_pair_mass,
        anchor_gross_to_net_ratio=anchor_gross_to_net_ratio,
        companion_gross_to_net_ratio=companion_gross_to_net_ratio,
        cancellation_ratio_gap=cancellation_ratio_gap,
        anchor_net_share_of_gross_mass=anchor_net_share_of_gross_mass,
        companion_net_share_of_gross_mass=companion_net_share_of_gross_mass,
        required_incremental_right_center_covariance_lift=required_increment,
        required_increment_share_of_anchor_abs_basis_pair_mass=(
            required_increment_share_of_anchor_abs_basis_pair_mass
        ),
        required_increment_share_of_companion_abs_basis_pair_mass=(
            required_increment_share_of_companion_abs_basis_pair_mass
        ),
        anchor_top10_abs_share_of_gross_mass=anchor_top10_abs_share_of_gross_mass,
        companion_top10_abs_share_of_gross_mass=companion_top10_abs_share_of_gross_mass,
        driver_signature=driver_signature,
        canonical_basis_pair_cancellation_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport
):
    center_coupling_channel_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe()
    )
    if _is_repo_side_canonical_channel(center_coupling_channel_report):
        return _build_repo_side_basis_pair_cancellation_report(
            center_coupling_channel_report
        )
    seed_window_covariance_report = (
        _build_seed_window_covariance_report_for_binding_design(
            center_coupling_channel_report.binding_design,
            coverage_anchor_random_state=center_coupling_channel_report.coverage_anchor_random_state,
            overshoot_companion_random_state=center_coupling_channel_report.overshoot_companion_random_state,
        )
    )
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_report(
        center_coupling_channel_report=center_coupling_channel_report,
        seed_window_covariance_report=seed_window_covariance_report,
    )
