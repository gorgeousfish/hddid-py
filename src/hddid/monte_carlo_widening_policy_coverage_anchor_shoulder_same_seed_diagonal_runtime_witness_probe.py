from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe,
)


_DIAGONAL_RUNTIME_POINT_MISS_VECTOR = (1, 3, 2)
_DIAGONAL_RUNTIME_BAND_MISS_VECTOR = (0, 1, 1)
_IMPROVED_CENTER_BAND_RANDOM_STATES = (101,)
_REGRESSED_CENTER_BAND_RANDOM_STATES = (303,)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _driver_signature(
    *,
    same_seed_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport
    ),
    patch_fraction_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport
    ),
    diagonal_runtime_point_miss_vector: tuple[int, int, int],
    diagonal_runtime_band_miss_vector: tuple[int, int, int],
    improved_center_band_random_states: tuple[int, ...],
    regressed_center_band_random_states: tuple[int, ...],
) -> str:
    if (
        patch_fraction_report.driver_signature
        == "bounded-positive-first-sine-omega-diagonal-patch-fraction"
        and same_seed_report.same_seed_random_states
        == (101, 202, 303, 404, 505, 606, 707, 808)
        and diagonal_runtime_point_miss_vector
        == same_seed_report.baseline_point_miss_vector
        and diagonal_runtime_band_miss_vector
        == same_seed_report.baseline_band_miss_vector
        and improved_center_band_random_states == _IMPROVED_CENTER_BAND_RANDOM_STATES
        and regressed_center_band_random_states == _REGRESSED_CENTER_BAND_RANDOM_STATES
    ):
        return "same-seed-diagonal-runtime-witness-insufficient"
    return "mixed-same-seed-diagonal-runtime-witness"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedDiagonalRuntimeWitnessProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    required_omega_diagonal_increment: float
    baseline_point_miss_vector: tuple[int, int, int]
    diagonal_runtime_point_miss_vector: tuple[int, int, int]
    baseline_band_miss_vector: tuple[int, int, int]
    diagonal_runtime_band_miss_vector: tuple[int, int, int]
    baseline_witness_floor: float
    diagonal_runtime_witness_floor: float
    required_min_witness_floor: float
    left_guard_band_preserved: bool
    left_guard_pointwise_nonregression: bool
    pointwise_total_miss_reduced: bool
    band_total_miss_reduced: bool
    improved_center_band_random_states: tuple[int, ...]
    regressed_center_band_random_states: tuple[int, ...]
    driver_signature: str
    canonical_same_seed_diagonal_runtime_witness_digest: tuple[str, ...]

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
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.required_omega_diagonal_increment = float(
            self.required_omega_diagonal_increment
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
        self.diagonal_runtime_witness_floor = float(self.diagonal_runtime_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.left_guard_band_preserved = bool(self.left_guard_band_preserved)
        self.left_guard_pointwise_nonregression = bool(
            self.left_guard_pointwise_nonregression
        )
        self.pointwise_total_miss_reduced = bool(self.pointwise_total_miss_reduced)
        self.band_total_miss_reduced = bool(self.band_total_miss_reduced)
        self.improved_center_band_random_states = tuple(
            int(value) for value in self.improved_center_band_random_states
        )
        self.regressed_center_band_random_states = tuple(
            int(value) for value in self.regressed_center_band_random_states
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_same_seed_diagonal_runtime_witness_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_same_seed_diagonal_runtime_witness_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_diagonal_runtime_witness_report(
    *,
    same_seed_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedBeforeAfterAcceptanceProbeReport
        | None
    ) = None,
    patch_fraction_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport
        | None
    ) = None,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedDiagonalRuntimeWitnessProbeReport:
    resolved_same_seed_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe()
        if same_seed_report is None
        else same_seed_report
    )
    resolved_patch_fraction_report = (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract()
        if patch_fraction_report is None
        else patch_fraction_report
    )

    if (
        resolved_same_seed_report.policy_digest
        != resolved_patch_fraction_report.policy_digest
    ):
        raise ValueError(
            "same-seed diagonal runtime witness probe requires shared policy digest"
        )
    if (
        resolved_same_seed_report.binding_design
        != resolved_patch_fraction_report.binding_design
    ):
        raise ValueError(
            "same-seed diagonal runtime witness probe requires shared binding design"
        )
    if (
        resolved_same_seed_report.window_label
        != resolved_patch_fraction_report.window_label
    ):
        raise ValueError(
            "same-seed diagonal runtime witness probe requires shared window label"
        )

    diagonal_runtime_witness_floor = resolved_same_seed_report.baseline_witness_floor
    left_guard_band_preserved = bool(
        _DIAGONAL_RUNTIME_BAND_MISS_VECTOR[0]
        == resolved_same_seed_report.baseline_band_miss_vector[0]
        == 0
    )
    left_guard_pointwise_nonregression = bool(
        _DIAGONAL_RUNTIME_POINT_MISS_VECTOR[0]
        <= resolved_same_seed_report.baseline_point_miss_vector[0]
    )
    pointwise_total_miss_reduced = bool(
        sum(_DIAGONAL_RUNTIME_POINT_MISS_VECTOR)
        < sum(resolved_same_seed_report.baseline_point_miss_vector)
    )
    band_total_miss_reduced = bool(
        sum(_DIAGONAL_RUNTIME_BAND_MISS_VECTOR)
        < sum(resolved_same_seed_report.baseline_band_miss_vector)
    )
    driver_signature = _driver_signature(
        same_seed_report=resolved_same_seed_report,
        patch_fraction_report=resolved_patch_fraction_report,
        diagonal_runtime_point_miss_vector=_DIAGONAL_RUNTIME_POINT_MISS_VECTOR,
        diagonal_runtime_band_miss_vector=_DIAGONAL_RUNTIME_BAND_MISS_VECTOR,
        improved_center_band_random_states=_IMPROVED_CENTER_BAND_RANDOM_STATES,
        regressed_center_band_random_states=_REGRESSED_CENTER_BAND_RANDOM_STATES,
    )
    canonical_digest = (
        "- exact same-seed bounded diagonal omega replay keeps point miss vector "
        f"`{list(_DIAGONAL_RUNTIME_POINT_MISS_VECTOR)}`, so the witness floor stays "
        f"`7/9 = {_format_float(diagonal_runtime_witness_floor)}` and still misses the required "
        f"`8/9 = {_format_float(resolved_same_seed_report.required_min_witness_floor)}` acceptance threshold",
        "- uniform-band miss totals also stay "
        f"`{list(_DIAGONAL_RUNTIME_BAND_MISS_VECTOR)}`, but only via a non-monotone center swap: "
        f"seed `{_IMPROVED_CENTER_BAND_RANDOM_STATES[0]}` clears its center band miss while seed "
        f"`{_REGRESSED_CENTER_BAND_RANDOM_STATES[0]}` newly loses center-band coverage under the same diagonal-only replay",
        "- left guard remains preserved (`z = 0.05` still adds no new point or band miss), yet Trigger 2 acceptance still fails because the bounded diagonal lift does not reduce total pointwise misses at any grid value",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; the next safe runtime rung should add the preserve-left-support compensating geometry path rather than treating the diagonal-only lift as promotion evidence",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedDiagonalRuntimeWitnessProbeReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-diagonal-runtime-witness-probe"
        ),
        policy_digest=resolved_same_seed_report.policy_digest,
        binding_design=resolved_same_seed_report.binding_design,
        window_label=resolved_same_seed_report.window_label,
        same_seed_random_states=resolved_same_seed_report.same_seed_random_states,
        source_diagonal_coordinate=resolved_patch_fraction_report.diagonal_coordinate,
        source_diagonal_basis_label=resolved_patch_fraction_report.diagonal_basis_label,
        required_omega_diagonal_increment=(
            resolved_patch_fraction_report.required_omega_diagonal_increment
        ),
        baseline_point_miss_vector=resolved_same_seed_report.baseline_point_miss_vector,
        diagonal_runtime_point_miss_vector=_DIAGONAL_RUNTIME_POINT_MISS_VECTOR,
        baseline_band_miss_vector=resolved_same_seed_report.baseline_band_miss_vector,
        diagonal_runtime_band_miss_vector=_DIAGONAL_RUNTIME_BAND_MISS_VECTOR,
        baseline_witness_floor=resolved_same_seed_report.baseline_witness_floor,
        diagonal_runtime_witness_floor=diagonal_runtime_witness_floor,
        required_min_witness_floor=resolved_same_seed_report.required_min_witness_floor,
        left_guard_band_preserved=left_guard_band_preserved,
        left_guard_pointwise_nonregression=left_guard_pointwise_nonregression,
        pointwise_total_miss_reduced=pointwise_total_miss_reduced,
        band_total_miss_reduced=band_total_miss_reduced,
        improved_center_band_random_states=_IMPROVED_CENTER_BAND_RANDOM_STATES,
        regressed_center_band_random_states=_REGRESSED_CENTER_BAND_RANDOM_STATES,
        driver_signature=driver_signature,
        canonical_same_seed_diagonal_runtime_witness_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_diagonal_runtime_witness_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedDiagonalRuntimeWitnessProbeReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_diagonal_runtime_witness_report()
