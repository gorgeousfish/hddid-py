from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_binding_design_rerun_capacity_probe import (
    Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport,
    run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_probe,
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
    rerun_capacity_report: (
        Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport
    ),
    readiness_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport
    ),
    rerun_gate_holds: bool,
) -> str:
    if (
        rerun_capacity_report.current_implication
        == "spend-remaining-fresh-reruns-on-binding-design"
        and readiness_report.driver_signature
        == "bounded-right-center-entry-patch-implementation-readiness"
        and rerun_gate_holds
    ):
        return "fresh-rerun-gated-by-entry-patch-readiness"
    return "mixed-fresh-rerun-readiness-gate"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunGateReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    live_driver_signature: str
    readiness_driver_signature: str
    limiting_budget: str
    effective_fresh_reruns: int
    source_diagonal_coordinate: int
    source_diagonal_basis_label: str
    shared_vf_entry_label: str
    required_source_diagonal_increment: float
    required_patch_increment: float
    required_patch_share_of_psd_boundary: float
    remaining_psd_headroom_multiple_of_required_patch: float
    prohibited_actions: tuple[str, ...]
    rerun_gate_holds: bool
    driver_signature: str
    canonical_fresh_rerun_gate_digest: tuple[str, ...]

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
        self.live_driver_signature = str(self.live_driver_signature).strip()
        self.readiness_driver_signature = str(self.readiness_driver_signature).strip()
        self.limiting_budget = str(self.limiting_budget).strip()
        self.effective_fresh_reruns = int(self.effective_fresh_reruns)
        self.source_diagonal_coordinate = int(self.source_diagonal_coordinate)
        self.source_diagonal_basis_label = str(self.source_diagonal_basis_label).strip()
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
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
        self.prohibited_actions = tuple(
            str(item).strip() for item in self.prohibited_actions
        )
        self.rerun_gate_holds = bool(self.rerun_gate_holds)
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_fresh_rerun_gate_digest = tuple(
            str(line).rstrip() for line in self.canonical_fresh_rerun_gate_digest
        )


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate_report(
    *,
    rerun_capacity_report: (
        Phase7MonteCarloWideningPolicyBindingDesignRerunCapacityProbeReport
    ),
    readiness_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchImplementationReadinessReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunGateReport:
    if rerun_capacity_report.policy_digest != readiness_report.policy_digest:
        raise ValueError("fresh rerun gate requires shared policy digest")
    if rerun_capacity_report.binding_design != readiness_report.binding_design:
        raise ValueError("fresh rerun gate requires shared binding design")

    rerun_gate_holds = bool(
        rerun_capacity_report.current_implication
        == "spend-remaining-fresh-reruns-on-binding-design"
        and rerun_capacity_report.effective_additional_reruns > 0
        and readiness_report.readiness_contract_holds
        and readiness_report.driver_signature
        == "bounded-right-center-entry-patch-implementation-readiness"
    )
    driver_signature = _driver_signature(
        rerun_capacity_report=rerun_capacity_report,
        readiness_report=readiness_report,
        rerun_gate_holds=rerun_gate_holds,
    )
    prohibited_actions = (
        "stable-slice-wide first-sine replay",
        "coordinate-`2` off-diagonal fallback",
        "full-matrix widening replay",
    )
    binding_design_key = _format_design_key(readiness_report.binding_design)

    canonical_digest = (
        f"- fresh rerun spend still stays scarce and seed-limited: only `{rerun_capacity_report.effective_additional_reruns}` fresh random states remain for binding design `{binding_design_key}`, so any new Trigger 2 runtime evidence must spend that budget on the same bounded lane rather than on already-covered or blocked replays",
        f"- that scarce rerun budget is only actionable after the implementation intake stays `bounded-right-center-entry-patch-implementation-readiness`: on `{readiness_report.window_label}`, the bounded estimator path must still enter through coordinate `{readiness_report.source_diagonal_coordinate} = {readiness_report.source_diagonal_basis_label}` on `omega_f_hat[2,2] -> {readiness_report.shared_vf_entry_label}`, with source lift `{_format_signed_float(readiness_report.source_required_omega_diagonal_increment)}`, shared-entry lift `{_format_signed_float(readiness_report.required_patch_increment)}`, PSD usage `{_format_percent(readiness_report.required_patch_share_of_psd_boundary)}`, and `{_format_float(readiness_report.remaining_psd_headroom_multiple_of_required_patch)}x` remaining headroom",
        "- current Trigger 2 implication: `fresh-rerun-gated-by-entry-patch-readiness`; keep this as validation-only spend control, and do not burn the remaining reruns on stable-slice-wide first-sine replay, coordinate-`2` off-diagonal fallback, or full-matrix widening replay while the live route remains `bounded-right-center-execution-contract`",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunGateReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-fresh-rerun-gate"
        ),
        policy_digest=readiness_report.policy_digest,
        binding_design=readiness_report.binding_design,
        window_label=readiness_report.window_label,
        coverage_anchor_random_state=readiness_report.coverage_anchor_random_state,
        overshoot_companion_random_state=readiness_report.overshoot_companion_random_state,
        live_driver_signature=readiness_report.live_driver_signature,
        readiness_driver_signature=readiness_report.driver_signature,
        limiting_budget=rerun_capacity_report.limiting_budget,
        effective_fresh_reruns=rerun_capacity_report.effective_additional_reruns,
        source_diagonal_coordinate=readiness_report.source_diagonal_coordinate,
        source_diagonal_basis_label=readiness_report.source_diagonal_basis_label,
        shared_vf_entry_label=readiness_report.shared_vf_entry_label,
        required_source_diagonal_increment=readiness_report.source_required_omega_diagonal_increment,
        required_patch_increment=readiness_report.required_patch_increment,
        required_patch_share_of_psd_boundary=readiness_report.required_patch_share_of_psd_boundary,
        remaining_psd_headroom_multiple_of_required_patch=readiness_report.remaining_psd_headroom_multiple_of_required_patch,
        prohibited_actions=prohibited_actions,
        rerun_gate_holds=rerun_gate_holds,
        driver_signature=driver_signature,
        canonical_fresh_rerun_gate_digest=canonical_digest,
    )


def _make_canonical_fresh_rerun_gate_report() -> (
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
        required_source_diagonal_increment=185.45799327602003,
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


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchFreshRerunGateReport
):
    return _make_canonical_fresh_rerun_gate_report()
