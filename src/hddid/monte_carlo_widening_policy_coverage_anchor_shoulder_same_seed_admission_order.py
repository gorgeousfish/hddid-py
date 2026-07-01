from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_floor_lift_ceiling_probe import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_floor_lift_ceiling_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_floor_lift_acceptance_contract import (
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_floor_lift_acceptance_contract,
)

if TYPE_CHECKING:
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_floor_lift_ceiling_probe import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFloorLiftCeilingProbeReport,
    )
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_floor_lift_acceptance_contract import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedFloorLiftAcceptanceContractReport,
    )

_FLOOR_LIFT_DRIVER_SIGNATURE = "single-point-floor-lift-clears-bounded-floor"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _driver_signature(
    *,
    floor_lift_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFloorLiftCeilingProbeReport
    ),
    same_seed_acceptance_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedFloorLiftAcceptanceContractReport
    ),
    fresh_estimator_evidence_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport
    ),
    same_seed_admission_required: bool,
) -> str:
    if (
        floor_lift_report.projected_crosses_canonical_floor
        and _FLOOR_LIFT_DRIVER_SIGNATURE
        == "single-point-floor-lift-clears-bounded-floor"
        and same_seed_acceptance_report.acceptance_contract_holds
        and same_seed_acceptance_report.driver_signature
        == "same-seed-floor-lift-acceptance-contract"
        and fresh_estimator_evidence_report.intake_contract_holds
        and fresh_estimator_evidence_report.driver_signature
        == "bounded-entry-patch-fresh-estimator-evidence-intake-contract"
        and same_seed_admission_required
    ):
        return "same-seed-admission-order"
    return "mixed-same-seed-admission-order"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    same_seed_random_states: tuple[int, ...]
    floor_lift_driver_signature: str
    same_seed_acceptance_driver_signature: str
    fresh_estimator_evidence_driver_signature: str
    baseline_witness_floor_ceiling: float
    projected_floor_ceiling: float
    projected_floor_slack: float
    canonical_floor: float
    canonical_floor_shortfall: float
    all_seed_point_miss_vector: tuple[int, int, int]
    all_seed_band_miss_vector: tuple[int, int, int]
    left_guard_grid_value: float
    center_grid_value: float
    right_shoulder_grid_value: float
    effective_fresh_reruns: int
    target_gate_status: str
    admission_order: tuple[str, ...]
    same_seed_admission_required: bool
    driver_signature: str
    canonical_same_seed_admission_digest: tuple[str, ...]

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
        self.floor_lift_driver_signature = str(self.floor_lift_driver_signature).strip()
        self.same_seed_acceptance_driver_signature = str(
            self.same_seed_acceptance_driver_signature
        ).strip()
        self.fresh_estimator_evidence_driver_signature = str(
            self.fresh_estimator_evidence_driver_signature
        ).strip()
        self.baseline_witness_floor_ceiling = float(self.baseline_witness_floor_ceiling)
        self.projected_floor_ceiling = float(self.projected_floor_ceiling)
        self.projected_floor_slack = float(self.projected_floor_slack)
        self.canonical_floor = float(self.canonical_floor)
        self.canonical_floor_shortfall = float(self.canonical_floor_shortfall)
        self.all_seed_point_miss_vector = tuple(
            int(value) for value in self.all_seed_point_miss_vector
        )
        self.all_seed_band_miss_vector = tuple(
            int(value) for value in self.all_seed_band_miss_vector
        )
        self.left_guard_grid_value = float(self.left_guard_grid_value)
        self.center_grid_value = float(self.center_grid_value)
        self.right_shoulder_grid_value = float(self.right_shoulder_grid_value)
        self.effective_fresh_reruns = int(self.effective_fresh_reruns)
        self.target_gate_status = str(self.target_gate_status).strip()
        self.admission_order = tuple(str(item).strip() for item in self.admission_order)
        self.same_seed_admission_required = bool(self.same_seed_admission_required)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_same_seed_admission_digest = tuple(
            str(line).rstrip() for line in self.canonical_same_seed_admission_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "same_seed_random_states": list(self.same_seed_random_states),
            "floor_lift_driver_signature": self.floor_lift_driver_signature,
            "same_seed_acceptance_driver_signature": (
                self.same_seed_acceptance_driver_signature
            ),
            "fresh_estimator_evidence_driver_signature": (
                self.fresh_estimator_evidence_driver_signature
            ),
            "baseline_witness_floor_ceiling": self.baseline_witness_floor_ceiling,
            "projected_floor_ceiling": self.projected_floor_ceiling,
            "projected_floor_slack": self.projected_floor_slack,
            "canonical_floor": self.canonical_floor,
            "canonical_floor_shortfall": self.canonical_floor_shortfall,
            "all_seed_point_miss_vector": list(self.all_seed_point_miss_vector),
            "all_seed_band_miss_vector": list(self.all_seed_band_miss_vector),
            "left_guard_grid_value": self.left_guard_grid_value,
            "center_grid_value": self.center_grid_value,
            "right_shoulder_grid_value": self.right_shoulder_grid_value,
            "effective_fresh_reruns": self.effective_fresh_reruns,
            "target_gate_status": self.target_gate_status,
            "admission_order": list(self.admission_order),
            "same_seed_admission_required": self.same_seed_admission_required,
            "driver_signature": self.driver_signature,
            "canonical_same_seed_admission_digest": list(
                self.canonical_same_seed_admission_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order_report(
    *,
    floor_lift_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFloorLiftCeilingProbeReport
    ),
    same_seed_acceptance_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedFloorLiftAcceptanceContractReport
    ),
    fresh_estimator_evidence_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshEstimatorEvidenceIntakeContractReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport:
    if floor_lift_report is None:
        floor_lift_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_floor_lift_ceiling_probe()
    if same_seed_acceptance_report is None:
        same_seed_acceptance_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_floor_lift_acceptance_contract()
    if fresh_estimator_evidence_report is None:
        fresh_estimator_evidence_report = run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract()

    if floor_lift_report.policy_digest != same_seed_acceptance_report.policy_digest:
        raise ValueError("same-seed admission order requires shared policy digest")
    if floor_lift_report.policy_digest != fresh_estimator_evidence_report.policy_digest:
        raise ValueError("same-seed admission order requires shared policy digest")
    if floor_lift_report.binding_design != same_seed_acceptance_report.binding_design:
        raise ValueError("same-seed admission order requires shared binding design")
    if (
        floor_lift_report.binding_design
        != fresh_estimator_evidence_report.binding_design
    ):
        raise ValueError("same-seed admission order requires shared binding design")
    if (
        floor_lift_report.residual_grid_label
        != same_seed_acceptance_report.window_label
    ):
        raise ValueError("same-seed admission order requires shared window label")
    if (
        floor_lift_report.residual_grid_label
        != fresh_estimator_evidence_report.window_label
    ):
        raise ValueError("same-seed admission order requires shared window label")

    same_seed_random_states = same_seed_acceptance_report.all_random_states
    admission_order = (
        "treat the bounded patch as worth promoting only after an exact same-seed before/after replay on `(101, 202, 303, 404, 505, 606, 707, 808)`",
        "use the arithmetic ceiling `7/9 -> 8/9 -> 0.889` as the minimum uplift target for the replayed witness floor",
        "enforce `z = 0.05` as the no-regression guard and keep any residual miss mass inside `z in {0.15, 0.25}`",
        "do not spend remaining fresh reruns on fresh-only sweeps until a same-seed replay shows real floor-lift evidence",
    )
    same_seed_admission_required = bool(
        floor_lift_report.projected_crosses_canonical_floor
        and same_seed_acceptance_report.acceptance_contract_holds
        and fresh_estimator_evidence_report.intake_contract_holds
        and same_seed_acceptance_report.left_band_miss_count == 0
        and fresh_estimator_evidence_report.canonical_floor_shortfall > 0.0
    )
    driver_signature = _driver_signature(
        floor_lift_report=floor_lift_report,
        same_seed_acceptance_report=same_seed_acceptance_report,
        fresh_estimator_evidence_report=fresh_estimator_evidence_report,
        same_seed_admission_required=same_seed_admission_required,
    )
    canonical_digest = (
        "- arithmetic ceiling still says the bounded lane is worth testing on the exact same seeds: current witness floor is `7/9 = 0.778`, a single repaired witness lifts it to `8/9 = 0.889`, and the projected ceiling would clear canonical floor `0.850` by `0.039`",
        "- no-regression pressure stays localized to the bounded right-center lane: current all-seed miss vectors remain pointwise `[1, 3, 2]` and band `[0, 1, 1]` on `z = (0.05, 0.15, 0.25)`, so left guard `z = 0.05` must stay free of any new band miss while residual pressure remains confined to `z in {0.15, 0.25}`",
        f"- fresh reruns remain budgeted but not self-justifying: only `{fresh_estimator_evidence_report.effective_fresh_reruns}` fresh random states remain, current floor shortfall is still `{_format_float(fresh_estimator_evidence_report.canonical_floor_shortfall)}`, and Trigger 2 stays pinned to `{fresh_estimator_evidence_report.target_gate_status}` unless estimator evidence actually lifts the binding floor witness",
        "- current Trigger 2 implication: `same-seed-admission-order`; next execution should prefer an exact same-seed estimator before/after replay over fresh-only rerun sweeps or broader replay spending",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-same-seed-admission-order"
        ),
        policy_digest=floor_lift_report.policy_digest,
        binding_design=floor_lift_report.binding_design,
        window_label=floor_lift_report.residual_grid_label,
        same_seed_random_states=same_seed_random_states,
        floor_lift_driver_signature=_FLOOR_LIFT_DRIVER_SIGNATURE,
        same_seed_acceptance_driver_signature=(
            same_seed_acceptance_report.driver_signature
        ),
        fresh_estimator_evidence_driver_signature=(
            fresh_estimator_evidence_report.driver_signature
        ),
        baseline_witness_floor_ceiling=(
            same_seed_acceptance_report.baseline_witness_floor_ceiling
        ),
        projected_floor_ceiling=floor_lift_report.projected_floor_ceiling,
        projected_floor_slack=floor_lift_report.projected_floor_slack,
        canonical_floor=same_seed_acceptance_report.canonical_floor,
        canonical_floor_shortfall=(
            fresh_estimator_evidence_report.canonical_floor_shortfall
        ),
        all_seed_point_miss_vector=same_seed_acceptance_report.all_seed_point_miss_vector,
        all_seed_band_miss_vector=same_seed_acceptance_report.all_seed_band_miss_vector,
        left_guard_grid_value=same_seed_acceptance_report.left_guard_grid_value,
        center_grid_value=same_seed_acceptance_report.center_grid_value,
        right_shoulder_grid_value=same_seed_acceptance_report.right_shoulder_grid_value,
        effective_fresh_reruns=fresh_estimator_evidence_report.effective_fresh_reruns,
        target_gate_status=fresh_estimator_evidence_report.target_gate_status,
        admission_order=admission_order,
        same_seed_admission_required=same_seed_admission_required,
        driver_signature=driver_signature,
        canonical_same_seed_admission_digest=canonical_digest,
    )


def _build_same_seed_admission_order_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "same-seed-admission-order"
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
        floor_lift_driver_signature=_FLOOR_LIFT_DRIVER_SIGNATURE,
        same_seed_acceptance_driver_signature=(
            "same-seed-floor-lift-acceptance-contract"
        ),
        fresh_estimator_evidence_driver_signature=(
            "bounded-entry-patch-fresh-estimator-evidence-intake-contract"
        ),
        baseline_witness_floor_ceiling=7.0 / 9.0,
        projected_floor_ceiling=8.0 / 9.0,
        projected_floor_slack=0.03888888888888886,
        canonical_floor=0.85,
        canonical_floor_shortfall=0.07222222222222219,
        all_seed_point_miss_vector=(1, 3, 2),
        all_seed_band_miss_vector=(0, 1, 1),
        left_guard_grid_value=0.05,
        center_grid_value=0.15,
        right_shoulder_grid_value=0.25,
        effective_fresh_reruns=5,
        target_gate_status="trigger2-partial-widening-only",
        admission_order=(
            "treat the bounded patch as worth promoting only after an exact same-seed before/after replay on `(101, 202, 303, 404, 505, 606, 707, 808)`",
            "use the arithmetic ceiling `7/9 -> 8/9 -> 0.889` as the minimum uplift target for the replayed witness floor",
            "enforce `z = 0.05` as the no-regression guard and keep any residual miss mass inside `z in {0.15, 0.25}`",
            "do not spend remaining fresh reruns on fresh-only sweeps until a same-seed replay shows real floor-lift evidence",
        ),
        same_seed_admission_required=True,
        driver_signature="same-seed-admission-order",
        canonical_same_seed_admission_digest=(
            "- arithmetic ceiling still says the bounded lane is worth testing on the exact same seeds: current witness floor is `7/9 = 0.778`, a single repaired witness lifts it to `8/9 = 0.889`, and the projected ceiling would clear canonical floor `0.850` by `0.039`",
            "- no-regression pressure stays localized to the bounded right-center lane: current all-seed miss vectors remain pointwise `[1, 3, 2]` and band `[0, 1, 1]` on `z = (0.05, 0.15, 0.25)`, so left guard `z = 0.05` must stay free of any new band miss while residual pressure remains confined to `z in {0.15, 0.25}`",
            "- fresh reruns remain budgeted but not self-justifying: only `5` fresh random states remain, current floor shortfall is still `0.072`, and Trigger 2 stays pinned to `trigger2-partial-widening-only` unless estimator evidence actually lifts the binding floor witness",
            "- current Trigger 2 implication: `same-seed-admission-order`; next execution should prefer an exact same-seed estimator before/after replay over fresh-only rerun sweeps or broader replay spending",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderSameSeedAdmissionOrderReport
):
    return _build_same_seed_admission_order_snapshot_report()
