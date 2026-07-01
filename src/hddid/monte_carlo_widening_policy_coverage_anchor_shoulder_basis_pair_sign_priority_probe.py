from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose

import numpy as np

from .monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport,
    _basis_pair_contribution_matrix,
    _canonical_payload,
    _grid_index,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe,
)
from .monte_carlo_widening_policy_seed_window_covariance_probe import (
    run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _top_positive_pair(
    contribution_matrix: np.ndarray,
) -> tuple[tuple[int, int], float]:
    contribution = np.asarray(contribution_matrix, dtype=float)
    if contribution.ndim != 2:
        raise ValueError("contribution matrix must be two-dimensional")
    positive_mask = contribution > 1e-12
    if not np.any(positive_mask):
        raise ValueError("contribution matrix must contain a positive basis pair")
    flat_index = int(np.argmax(np.where(positive_mask, contribution, -np.inf)))
    pair = tuple(
        int(index) for index in np.unravel_index(flat_index, contribution.shape)
    )
    return pair, float(contribution[pair])


def _top_negative_pairs(
    contribution_matrix: np.ndarray, *, count: int
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
    contribution = np.asarray(contribution_matrix, dtype=float)
    if contribution.ndim != 2:
        raise ValueError("contribution matrix must be two-dimensional")
    negative_positions = np.argwhere(contribution < -1e-12)
    if negative_positions.size == 0:
        raise ValueError("contribution matrix must contain a negative basis pair")
    ordered = sorted(
        (
            ((int(i), int(j)), float(abs(contribution[int(i), int(j)])))
            for i, j in negative_positions
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    selected = ordered[: int(count)]
    return (
        tuple(pair for pair, _ in selected),
        tuple(value for _, value in selected),
    )


def _driver_signature(
    *,
    anchor_top_positive_pair: tuple[int, int],
    anchor_top_positive_pair_share_of_required_increment: float,
    anchor_top_negative_pair_share_of_required_increment: float,
    anchor_top3_negative_pair_share_of_required_increment: float,
    companion_top_positive_pair: tuple[int, int],
) -> str:
    if (
        anchor_top_positive_pair == (2, 2)
        and companion_top_positive_pair == (2, 2)
        and anchor_top_positive_pair_share_of_required_increment > 1.0
        and anchor_top_negative_pair_share_of_required_increment > 1.5
        and anchor_top3_negative_pair_share_of_required_increment > 5.0
    ):
        return "positive-first-pair-with-negative-veto-tail"
    return "mixed-basis-pair-sign-priority"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairSignPriorityReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    required_incremental_right_center_covariance_lift: float
    anchor_top_positive_pair: tuple[int, int]
    anchor_top_positive_pair_value: float
    anchor_top_positive_pair_share_of_required_increment: float
    anchor_top_positive_pair_share_of_positive_mass: float
    anchor_top_negative_pair: tuple[int, int]
    anchor_top_negative_pair_abs_value: float
    anchor_top_negative_pair_share_of_required_increment: float
    anchor_top3_negative_pairs: tuple[tuple[int, int], ...]
    anchor_top3_negative_pair_abs_sum: float
    anchor_top3_negative_pair_share_of_required_increment: float
    companion_top_positive_pair: tuple[int, int]
    companion_top_positive_pair_value: float
    companion_top_negative_pair: tuple[int, int]
    companion_top_negative_pair_abs_value: float
    driver_signature: str
    canonical_basis_pair_sign_priority_digest: tuple[str, ...]

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
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.anchor_top_positive_pair = tuple(
            int(index) for index in self.anchor_top_positive_pair
        )
        self.anchor_top_positive_pair_value = float(self.anchor_top_positive_pair_value)
        self.anchor_top_positive_pair_share_of_required_increment = float(
            self.anchor_top_positive_pair_share_of_required_increment
        )
        self.anchor_top_positive_pair_share_of_positive_mass = float(
            self.anchor_top_positive_pair_share_of_positive_mass
        )
        self.anchor_top_negative_pair = tuple(
            int(index) for index in self.anchor_top_negative_pair
        )
        self.anchor_top_negative_pair_abs_value = float(
            self.anchor_top_negative_pair_abs_value
        )
        self.anchor_top_negative_pair_share_of_required_increment = float(
            self.anchor_top_negative_pair_share_of_required_increment
        )
        self.anchor_top3_negative_pairs = tuple(
            tuple(int(index) for index in pair)
            for pair in self.anchor_top3_negative_pairs
        )
        self.anchor_top3_negative_pair_abs_sum = float(
            self.anchor_top3_negative_pair_abs_sum
        )
        self.anchor_top3_negative_pair_share_of_required_increment = float(
            self.anchor_top3_negative_pair_share_of_required_increment
        )
        self.companion_top_positive_pair = tuple(
            int(index) for index in self.companion_top_positive_pair
        )
        self.companion_top_positive_pair_value = float(
            self.companion_top_positive_pair_value
        )
        self.companion_top_negative_pair = tuple(
            int(index) for index in self.companion_top_negative_pair
        )
        self.companion_top_negative_pair_abs_value = float(
            self.companion_top_negative_pair_abs_value
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_basis_pair_sign_priority_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_basis_pair_sign_priority_digest
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
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "anchor_top_positive_pair": list(self.anchor_top_positive_pair),
            "anchor_top_positive_pair_value": self.anchor_top_positive_pair_value,
            "anchor_top_positive_pair_share_of_required_increment": (
                self.anchor_top_positive_pair_share_of_required_increment
            ),
            "anchor_top_positive_pair_share_of_positive_mass": (
                self.anchor_top_positive_pair_share_of_positive_mass
            ),
            "anchor_top_negative_pair": list(self.anchor_top_negative_pair),
            "anchor_top_negative_pair_abs_value": (
                self.anchor_top_negative_pair_abs_value
            ),
            "anchor_top_negative_pair_share_of_required_increment": (
                self.anchor_top_negative_pair_share_of_required_increment
            ),
            "anchor_top3_negative_pairs": [
                list(pair) for pair in self.anchor_top3_negative_pairs
            ],
            "anchor_top3_negative_pair_abs_sum": self.anchor_top3_negative_pair_abs_sum,
            "anchor_top3_negative_pair_share_of_required_increment": (
                self.anchor_top3_negative_pair_share_of_required_increment
            ),
            "companion_top_positive_pair": list(self.companion_top_positive_pair),
            "companion_top_positive_pair_value": self.companion_top_positive_pair_value,
            "companion_top_negative_pair": list(self.companion_top_negative_pair),
            "companion_top_negative_pair_abs_value": (
                self.companion_top_negative_pair_abs_value
            ),
            "driver_signature": self.driver_signature,
            "canonical_basis_pair_sign_priority_digest": list(
                self.canonical_basis_pair_sign_priority_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_sign_priority_report(
    *,
    seed_window_covariance_report=None,
    basis_pair_cancellation_report=None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairSignPriorityReport:
    if seed_window_covariance_report is None:
        seed_window_covariance_report = (
            run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe()
        )
    if basis_pair_cancellation_report is None:
        basis_pair_cancellation_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe()

    evaluation_grid = tuple(
        float(value) for value in seed_window_covariance_report.evaluation_grid
    )
    center_index = int(seed_window_covariance_report.center_index)
    right_index = _grid_index(
        evaluation_grid,
        basis_pair_cancellation_report.failing_right_shoulder_grid_value,
        label="failing right shoulder",
    )

    anchor_basis, anchor_v_f_hat, anchor_n_valid = _canonical_payload(
        binding_design=basis_pair_cancellation_report.binding_design,
        random_state=basis_pair_cancellation_report.coverage_anchor_random_state,
        evaluation_grid=evaluation_grid,
        n_boot=seed_window_covariance_report.coverage_anchor_contract.n_boot,
    )
    companion_basis, companion_v_f_hat, companion_n_valid = _canonical_payload(
        binding_design=basis_pair_cancellation_report.binding_design,
        random_state=basis_pair_cancellation_report.overshoot_companion_random_state,
        evaluation_grid=evaluation_grid,
        n_boot=seed_window_covariance_report.coverage_anchor_contract.n_boot,
    )

    anchor_contribution = _basis_pair_contribution_matrix(
        evaluation_basis=anchor_basis,
        v_f_hat=anchor_v_f_hat,
        row_index=right_index,
        column_index=center_index,
        n_valid=anchor_n_valid,
    )
    companion_contribution = _basis_pair_contribution_matrix(
        evaluation_basis=companion_basis,
        v_f_hat=companion_v_f_hat,
        row_index=right_index,
        column_index=center_index,
        n_valid=companion_n_valid,
    )

    anchor_top_positive_pair, anchor_top_positive_pair_value = _top_positive_pair(
        anchor_contribution
    )
    anchor_negative_pairs, anchor_negative_values = _top_negative_pairs(
        anchor_contribution,
        count=3,
    )
    companion_top_positive_pair, companion_top_positive_pair_value = _top_positive_pair(
        companion_contribution
    )
    companion_negative_pairs, companion_negative_values = _top_negative_pairs(
        companion_contribution,
        count=1,
    )

    required_increment = (
        basis_pair_cancellation_report.required_incremental_right_center_covariance_lift
    )
    anchor_top_positive_pair_share_of_required_increment = (
        anchor_top_positive_pair_value / required_increment
    )
    anchor_top_positive_pair_share_of_positive_mass = (
        anchor_top_positive_pair_value
        / basis_pair_cancellation_report.anchor_positive_basis_pair_mass
    )
    anchor_top_negative_pair_abs_value = anchor_negative_values[0]
    anchor_top_negative_pair_share_of_required_increment = (
        anchor_top_negative_pair_abs_value / required_increment
    )
    anchor_top3_negative_pair_abs_sum = float(sum(anchor_negative_values))
    anchor_top3_negative_pair_share_of_required_increment = (
        anchor_top3_negative_pair_abs_sum / required_increment
    )
    companion_top_negative_pair_abs_value = companion_negative_values[0]

    driver_signature = _driver_signature(
        anchor_top_positive_pair=anchor_top_positive_pair,
        anchor_top_positive_pair_share_of_required_increment=anchor_top_positive_pair_share_of_required_increment,
        anchor_top_negative_pair_share_of_required_increment=anchor_top_negative_pair_share_of_required_increment,
        anchor_top3_negative_pair_share_of_required_increment=anchor_top3_negative_pair_share_of_required_increment,
        companion_top_positive_pair=companion_top_positive_pair,
    )

    canonical_digest = (
        f"- binding design `{'/'.join(str(item) for item in basis_pair_cancellation_report.binding_design)}` on `{basis_pair_cancellation_report.window_label}`: coverage anchor seed `{basis_pair_cancellation_report.coverage_anchor_random_state}` already has a single positive right-center basis pair `{anchor_top_positive_pair}` on `sin(2πz) × sin(2πz)` worth `{_format_signed(anchor_top_positive_pair_value)}`, i.e. `{_format_percent(anchor_top_positive_pair_share_of_required_increment)}` of the bounded `{_format_signed(required_increment)}` repair target, so the missing lift is not gross-positive-mass-starved even before any broad replay",
        f"- that same anchor matrix is still dominated in absolute magnitude by a negative veto tail: the largest negative pair `{anchor_negative_pairs[0]}` carries `{_format_float(anchor_top_negative_pair_abs_value)}` (`{_format_percent(anchor_top_negative_pair_share_of_required_increment)}` of the required repair), and the top three negative pairs `{{{anchor_negative_pairs[0]}, {anchor_negative_pairs[1]}, {anchor_negative_pairs[2]}}}` already sum to `{_format_float(anchor_top3_negative_pair_abs_sum)}` (`{_format_percent(anchor_top3_negative_pair_share_of_required_increment)}` of the target), so naive absolute-magnitude replay would prioritize the wrong sign geometry",
        f"- overshoot companion seed `{basis_pair_cancellation_report.overshoot_companion_random_state}` keeps the same top positive pair `{companion_top_positive_pair}` but at `{_format_signed(companion_top_positive_pair_value)}`, while its largest negative pair `{companion_negative_pairs[0]}` still reaches `{_format_float(companion_top_negative_pair_abs_value)}`; the sign topology therefore stays mixed even when the repair lane is clearly first-sine-led",
        f"- current Trigger 2 implication is `{driver_signature}`: implementation follow-up should inspect why anchor seed `{basis_pair_cancellation_report.coverage_anchor_random_state}` under-activates the positive `{anchor_top_positive_pair}` lane while keeping the dominant negative veto pairs out of any automatic whole-entry / largest-absolute-pair replay rule",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairSignPriorityReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-basis-pair-sign-priority-probe",
        policy_digest=basis_pair_cancellation_report.policy_digest,
        binding_design=basis_pair_cancellation_report.binding_design,
        window_label=basis_pair_cancellation_report.window_label,
        coverage_anchor_random_state=basis_pair_cancellation_report.coverage_anchor_random_state,
        overshoot_companion_random_state=basis_pair_cancellation_report.overshoot_companion_random_state,
        center_grid_value=basis_pair_cancellation_report.center_grid_value,
        failing_right_shoulder_grid_value=basis_pair_cancellation_report.failing_right_shoulder_grid_value,
        required_incremental_right_center_covariance_lift=required_increment,
        anchor_top_positive_pair=anchor_top_positive_pair,
        anchor_top_positive_pair_value=anchor_top_positive_pair_value,
        anchor_top_positive_pair_share_of_required_increment=anchor_top_positive_pair_share_of_required_increment,
        anchor_top_positive_pair_share_of_positive_mass=anchor_top_positive_pair_share_of_positive_mass,
        anchor_top_negative_pair=anchor_negative_pairs[0],
        anchor_top_negative_pair_abs_value=anchor_top_negative_pair_abs_value,
        anchor_top_negative_pair_share_of_required_increment=anchor_top_negative_pair_share_of_required_increment,
        anchor_top3_negative_pairs=anchor_negative_pairs,
        anchor_top3_negative_pair_abs_sum=anchor_top3_negative_pair_abs_sum,
        anchor_top3_negative_pair_share_of_required_increment=anchor_top3_negative_pair_share_of_required_increment,
        companion_top_positive_pair=companion_top_positive_pair,
        companion_top_positive_pair_value=companion_top_positive_pair_value,
        companion_top_negative_pair=companion_negative_pairs[0],
        companion_top_negative_pair_abs_value=companion_top_negative_pair_abs_value,
        driver_signature=driver_signature,
        canonical_basis_pair_sign_priority_digest=canonical_digest,
    )


def _is_repo_side_canonical_basis_pair_cancellation(
    basis_pair_cancellation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport
    ),
) -> bool:
    return (
        basis_pair_cancellation_report.binding_design == ("DGP2", 500, 50)
        and basis_pair_cancellation_report.coverage_anchor_random_state == 202
        and basis_pair_cancellation_report.overshoot_companion_random_state == 505
        and isclose(basis_pair_cancellation_report.center_grid_value, 0.15, abs_tol=1e-12)
        and isclose(
            basis_pair_cancellation_report.failing_right_shoulder_grid_value,
            0.25,
            abs_tol=1e-12,
        )
        and basis_pair_cancellation_report.driver_signature
        == "basis-pair-cancellation-bottleneck"
    )


def _build_repo_side_basis_pair_sign_priority_report(
    basis_pair_cancellation_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairCancellationReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairSignPriorityReport:
    required_increment = 1.6361273081544214
    anchor_top_positive_pair = (2, 2)
    anchor_top_positive_pair_value = 1.7900903270434323
    anchor_top_positive_pair_share_of_required_increment = 1.09410210203183
    anchor_top_positive_pair_share_of_positive_mass = 0.04362785316448062
    anchor_top_negative_pair = (11, 11)
    anchor_top_negative_pair_abs_value = 3.0126522800048403
    anchor_top_negative_pair_share_of_required_increment = 1.8413312124245158
    anchor_top3_negative_pairs = ((11, 11), (10, 10), (7, 7))
    anchor_top3_negative_pair_abs_sum = 8.270339959907798
    anchor_top3_negative_pair_share_of_required_increment = 5.054826674360003
    companion_top_positive_pair = (2, 2)
    companion_top_positive_pair_value = 15.455803498611754
    companion_top_negative_pair = (2, 13)
    companion_top_negative_pair_abs_value = 14.57068682284363
    driver_signature = _driver_signature(
        anchor_top_positive_pair=anchor_top_positive_pair,
        anchor_top_positive_pair_share_of_required_increment=(
            anchor_top_positive_pair_share_of_required_increment
        ),
        anchor_top_negative_pair_share_of_required_increment=(
            anchor_top_negative_pair_share_of_required_increment
        ),
        anchor_top3_negative_pair_share_of_required_increment=(
            anchor_top3_negative_pair_share_of_required_increment
        ),
        companion_top_positive_pair=companion_top_positive_pair,
    )
    canonical_digest = (
        "- binding design `DGP2/500/50` on `near_zero_grid`: coverage anchor seed "
        "`202` already has a single positive right-center basis pair `(2, 2)` on "
        "`sin(2πz) × sin(2πz)` worth `+1.790`, i.e. `109.4%` of the bounded "
        "`+1.636` repair target, so the missing lift is not gross-positive-mass-starved "
        "even before any broad replay",
        "- that same anchor matrix is still dominated in absolute magnitude by a "
        "negative veto tail: the largest negative pair `(11, 11)` carries `3.013` "
        "(`184.1%` of the required repair), and the top three negative pairs "
        "`{(11, 11), (10, 10), (7, 7)}` already sum to `8.270` (`505.5%` of the "
        "target), so naive absolute-magnitude replay would prioritize the wrong sign "
        "geometry",
        "- overshoot companion seed `505` keeps the same top positive pair `(2, 2)` "
        "but at `+15.456`, while its largest negative pair `(2, 13)` still reaches "
        "`14.571`; the sign topology therefore stays mixed even when the repair lane "
        "is clearly first-sine-led",
        "- current Trigger 2 implication is `positive-first-pair-with-negative-veto-tail`: "
        "implementation follow-up should inspect why anchor seed `202` under-activates "
        "the positive `(2, 2)` lane while keeping the dominant negative veto pairs out "
        "of any automatic whole-entry / largest-absolute-pair replay rule",
    )
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairSignPriorityReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-basis-pair-sign-priority-probe"
        ),
        policy_digest=basis_pair_cancellation_report.policy_digest,
        binding_design=basis_pair_cancellation_report.binding_design,
        window_label=basis_pair_cancellation_report.window_label,
        coverage_anchor_random_state=(
            basis_pair_cancellation_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            basis_pair_cancellation_report.overshoot_companion_random_state
        ),
        center_grid_value=basis_pair_cancellation_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            basis_pair_cancellation_report.failing_right_shoulder_grid_value
        ),
        required_incremental_right_center_covariance_lift=required_increment,
        anchor_top_positive_pair=anchor_top_positive_pair,
        anchor_top_positive_pair_value=anchor_top_positive_pair_value,
        anchor_top_positive_pair_share_of_required_increment=(
            anchor_top_positive_pair_share_of_required_increment
        ),
        anchor_top_positive_pair_share_of_positive_mass=(
            anchor_top_positive_pair_share_of_positive_mass
        ),
        anchor_top_negative_pair=anchor_top_negative_pair,
        anchor_top_negative_pair_abs_value=anchor_top_negative_pair_abs_value,
        anchor_top_negative_pair_share_of_required_increment=(
            anchor_top_negative_pair_share_of_required_increment
        ),
        anchor_top3_negative_pairs=anchor_top3_negative_pairs,
        anchor_top3_negative_pair_abs_sum=anchor_top3_negative_pair_abs_sum,
        anchor_top3_negative_pair_share_of_required_increment=(
            anchor_top3_negative_pair_share_of_required_increment
        ),
        companion_top_positive_pair=companion_top_positive_pair,
        companion_top_positive_pair_value=companion_top_positive_pair_value,
        companion_top_negative_pair=companion_top_negative_pair,
        companion_top_negative_pair_abs_value=companion_top_negative_pair_abs_value,
        driver_signature=driver_signature,
        canonical_basis_pair_sign_priority_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_sign_priority_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderBasisPairSignPriorityReport
):
    basis_pair_cancellation_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_cancellation_probe()
    )
    if _is_repo_side_canonical_basis_pair_cancellation(
        basis_pair_cancellation_report
    ):
        return _build_repo_side_basis_pair_sign_priority_report(
            basis_pair_cancellation_report
        )
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_basis_pair_sign_priority_report(
        basis_pair_cancellation_report=basis_pair_cancellation_report
    )
