from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunGateReport,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_promotion_guard_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPromotionGuardReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_promotion_guard_probe,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_design_key(binding_design: tuple[str, int, int]) -> str:
    return f"{binding_design[0]}/{binding_design[1]}/{binding_design[2]}"


def _driver_signature(
    *,
    rerun_gate_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunGateReport
    ),
    promotion_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPromotionGuardReport
    ),
    acceptance_contract_holds: bool,
) -> str:
    if (
        rerun_gate_report.driver_signature
        == "fresh-rerun-gated-by-entry-patch-readiness"
        and promotion_guard_report.driver_signature
        == "bounded-entry-patch-insufficient-for-floor-promotion"
        and acceptance_contract_holds
    ):
        return "fresh-rerun-must-carry-floor-lift-evidence"
    return "mixed-fresh-rerun-acceptance-contract"


def _make_canonical_rerun_gate_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunGateReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunGateReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-fresh-rerun-gate"
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
        coverage_anchor_random_state=202,
        overshoot_companion_random_state=505,
        live_driver_signature="bounded-right-center-execution-contract",
        readiness_driver_signature=(
            "bounded-right-center-entry-patch-implementation-readiness"
        ),
        limiting_budget="random_state_budget",
        effective_fresh_reruns=5,
        source_diagonal_coordinate=2,
        source_diagonal_basis_label="sin(2πz)",
        shared_vf_entry_label="v_f_hat[2,2]",
        required_source_diagonal_increment=185.4584356925577,
        required_patch_increment=1.6361273081544214,
        required_patch_share_of_psd_boundary=0.12714564275399315,
        remaining_psd_headroom_multiple_of_required_patch=6.8649962227556856,
        prohibited_actions=(
            "stable-slice-wide first-sine replay",
            "coordinate-`2` off-diagonal fallback",
            "full-matrix widening replay",
        ),
        rerun_gate_holds=True,
        driver_signature="fresh-rerun-gated-by-entry-patch-readiness",
        canonical_fresh_rerun_gate_digest=(
            "- fresh rerun spend still stays scarce and seed-limited: only `5` fresh random states remain for binding design `DGP2/500/50`, so any new Trigger 2 runtime evidence must spend that budget on the same bounded lane rather than on already-covered or blocked replays",
            "- that scarce rerun budget is only actionable after the implementation intake stays `bounded-right-center-entry-patch-implementation-readiness`: on `near_zero_grid`, the bounded estimator path must still enter through coordinate `2 = sin(2πz)` on `omega_f_hat[2,2] -> v_f_hat[2,2]`, with source lift `+185.458`, shared-entry lift `+1.636`, PSD usage `12.7%`, and `6.865x` remaining headroom",
            "- current Trigger 2 implication: `fresh-rerun-gated-by-entry-patch-readiness`; keep this as validation-only spend control, and do not burn the remaining reruns on stable-slice-wide first-sine replay, coordinate-`2` off-diagonal fallback, or full-matrix widening replay while the live route remains `bounded-right-center-execution-contract`",
        ),
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunAcceptanceContractReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    limiting_budget: str
    effective_fresh_reruns: int
    readiness_driver_signature: str
    rerun_gate_driver_signature: str
    promotion_driver_signature: str
    required_source_diagonal_increment: float
    required_patch_increment: float
    required_patch_share_of_psd_boundary: float
    remaining_psd_headroom_multiple_of_required_patch: float
    supported_floor_ceiling: float
    canonical_floor: float
    canonical_floor_shortfall: float
    target_gate_status: str
    binding_guard: str
    acceptance_contract_holds: bool
    driver_signature: str
    canonical_fresh_rerun_acceptance_digest: tuple[str, ...]

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
        self.limiting_budget = str(self.limiting_budget).strip()
        self.effective_fresh_reruns = int(self.effective_fresh_reruns)
        self.readiness_driver_signature = str(self.readiness_driver_signature).strip()
        self.rerun_gate_driver_signature = str(self.rerun_gate_driver_signature).strip()
        self.promotion_driver_signature = str(self.promotion_driver_signature).strip()
        self.required_source_diagonal_increment = float(
            self.required_source_diagonal_increment
        )
        self.required_patch_increment = float(self.required_patch_increment)
        self.required_patch_share_of_psd_boundary = float(
            self.required_patch_share_of_psd_boundary
        )
        self.remaining_psd_headroom_multiple_of_required_patch = float(
            self.remaining_psd_headroom_multiple_of_required_patch
        )
        self.supported_floor_ceiling = float(self.supported_floor_ceiling)
        self.canonical_floor = float(self.canonical_floor)
        self.canonical_floor_shortfall = float(self.canonical_floor_shortfall)
        self.target_gate_status = str(self.target_gate_status).strip()
        self.binding_guard = str(self.binding_guard).strip()
        self.acceptance_contract_holds = bool(self.acceptance_contract_holds)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_fresh_rerun_acceptance_digest = tuple(
            str(line).rstrip() for line in self.canonical_fresh_rerun_acceptance_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_acceptance_contract_report(
    *,
    rerun_gate_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunGateReport
    ),
    promotion_guard_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPromotionGuardReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunAcceptanceContractReport:
    if rerun_gate_report.policy_digest != promotion_guard_report.policy_digest:
        raise ValueError(
            "fresh rerun acceptance contract requires shared policy digest"
        )
    if rerun_gate_report.binding_design != promotion_guard_report.binding_design:
        raise ValueError(
            "fresh rerun acceptance contract requires shared binding design"
        )
    if rerun_gate_report.window_label != promotion_guard_report.window_label:
        raise ValueError("fresh rerun acceptance contract requires shared window label")
    if (
        rerun_gate_report.coverage_anchor_random_state
        != promotion_guard_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        rerun_gate_report.overshoot_companion_random_state
        != promotion_guard_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")

    acceptance_contract_holds = bool(
        rerun_gate_report.rerun_gate_holds
        and rerun_gate_report.driver_signature
        == "fresh-rerun-gated-by-entry-patch-readiness"
        and promotion_guard_report.driver_signature
        == "bounded-entry-patch-insufficient-for-floor-promotion"
        and promotion_guard_report.target_gate_status
        == "trigger2-partial-widening-only"
        and promotion_guard_report.canonical_floor_shortfall > 0.0
    )
    driver_signature = _driver_signature(
        rerun_gate_report=rerun_gate_report,
        promotion_guard_report=promotion_guard_report,
        acceptance_contract_holds=acceptance_contract_holds,
    )
    binding_design_key = _format_design_key(rerun_gate_report.binding_design)

    canonical_digest = (
        f"- binding design `{binding_design_key}` still has only `{rerun_gate_report.effective_fresh_reruns}` fresh random states left under `bounded-n500-p50`, and the rerun gate stays admissible only while the bounded intake remains `bounded-right-center-entry-patch-implementation-readiness` on `{rerun_gate_report.window_label}`",
        f"- that admissible intake is still narrowly scoped: coordinate `{rerun_gate_report.source_diagonal_coordinate} = {rerun_gate_report.source_diagonal_basis_label}` must carry source lift `{_format_signed_float(rerun_gate_report.required_source_diagonal_increment)}`, the shared `z = 0.25 -> 0.15` entry still needs only `{_format_signed_float(rerun_gate_report.required_patch_increment)}`, and the same patch still consumes just `{_format_percent(rerun_gate_report.required_patch_share_of_psd_boundary)}` of PSD boundary with `{_format_float(rerun_gate_report.remaining_psd_headroom_multiple_of_required_patch)}x` remaining headroom",
        f"- promotion evidence remains stricter than rerun admissibility: the bounded slice still supports floor ceiling `{_format_float(promotion_guard_report.supported_floor_ceiling)}`, so canonical `{promotion_guard_report.binding_guard} = {_format_float(promotion_guard_report.canonical_floor)}` remains short by `{_format_float(promotion_guard_report.canonical_floor_shortfall)}` and Trigger 2 still cannot promote past `{promotion_guard_report.target_gate_status}` on patch existence alone",
        f"- current Trigger 2 implication: `{driver_signature}`; spend fresh reruns only on the binding design, and treat a rerun as informative only if it keeps the bounded patch intake and delivers fresh estimator evidence that lifts the binding floor witness rather than merely replaying the validation-only patch object",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunAcceptanceContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-fresh-rerun-acceptance-contract"
        ),
        policy_digest=rerun_gate_report.policy_digest,
        binding_design=rerun_gate_report.binding_design,
        window_label=rerun_gate_report.window_label,
        coverage_anchor_random_state=rerun_gate_report.coverage_anchor_random_state,
        overshoot_companion_random_state=(
            rerun_gate_report.overshoot_companion_random_state
        ),
        limiting_budget=rerun_gate_report.limiting_budget,
        effective_fresh_reruns=rerun_gate_report.effective_fresh_reruns,
        readiness_driver_signature=rerun_gate_report.readiness_driver_signature,
        rerun_gate_driver_signature=rerun_gate_report.driver_signature,
        promotion_driver_signature=promotion_guard_report.driver_signature,
        required_source_diagonal_increment=(
            rerun_gate_report.required_source_diagonal_increment
        ),
        required_patch_increment=rerun_gate_report.required_patch_increment,
        required_patch_share_of_psd_boundary=(
            rerun_gate_report.required_patch_share_of_psd_boundary
        ),
        remaining_psd_headroom_multiple_of_required_patch=(
            rerun_gate_report.remaining_psd_headroom_multiple_of_required_patch
        ),
        supported_floor_ceiling=promotion_guard_report.supported_floor_ceiling,
        canonical_floor=promotion_guard_report.canonical_floor,
        canonical_floor_shortfall=promotion_guard_report.canonical_floor_shortfall,
        target_gate_status=promotion_guard_report.target_gate_status,
        binding_guard=promotion_guard_report.binding_guard,
        acceptance_contract_holds=acceptance_contract_holds,
        driver_signature=driver_signature,
        canonical_fresh_rerun_acceptance_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_acceptance_contract() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunAcceptanceContractReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_acceptance_contract_report(
        rerun_gate_report=_make_canonical_rerun_gate_report(),
        promotion_guard_report=run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_promotion_guard_probe(),
    )
