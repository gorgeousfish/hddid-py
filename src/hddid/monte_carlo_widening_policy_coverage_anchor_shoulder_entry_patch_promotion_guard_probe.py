from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport,
)
from .monte_carlo_widening_policy_floor_slack_probe import (
    Phase7MonteCarloWideningPolicyFloorSlackProbeReport,
    build_phase7_monte_carlo_widening_policy_floor_slack_probe_report,
)
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)
from .validation import (
    MonteCarloRuntimeProbeDesignSummary,
    MonteCarloRuntimeProbeReport,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _driver_signature(
    *,
    execution_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
    floor_slack_report: Phase7MonteCarloWideningPolicyFloorSlackProbeReport,
) -> str:
    if (
        execution_contract_report.driver_signature
        == "bounded-right-center-execution-contract"
        and floor_slack_report.supported_floor_ceiling
        < floor_slack_report.coverage_floor
        and floor_slack_report.binding_design
        == execution_contract_report.binding_design
    ):
        return "bounded-entry-patch-insufficient-for-floor-promotion"
    return "mixed-entry-patch-promotion-guard"


def _make_design_summary(
    *,
    dgp_name: str,
    n_obs: int,
    p: int,
    success_rate: float,
    runtime_mean_seconds: float | None,
    mean_nonparametric_coverage: float | None,
    typed_invalidity_counts: dict[str, int] | None = None,
) -> MonteCarloRuntimeProbeDesignSummary:
    successful_runs = 3 if success_rate > 0.0 else 0
    return MonteCarloRuntimeProbeDesignSummary(
        dgp_name=dgp_name,
        n_obs=n_obs,
        p=p,
        n_runs=3,
        n_successful_runs=successful_runs,
        success_rate=success_rate,
        runtime_mean_seconds=runtime_mean_seconds,
        runtime_std_seconds=0.0 if runtime_mean_seconds is not None else None,
        typed_invalidity_counts={}
        if typed_invalidity_counts is None
        else typed_invalidity_counts,
        typed_invalidity_examples={},
        mean_parametric_bias=0.0 if success_rate > 0.0 else None,
        std_parametric_bias=0.0 if success_rate > 0.0 else None,
        mean_parametric_rmse=0.2 if success_rate > 0.0 else None,
        mean_parametric_average_standard_error=0.2 if success_rate > 0.0 else None,
        std_parametric_average_standard_error=0.0 if success_rate > 0.0 else None,
        mean_parametric_coverage=0.9 if success_rate > 0.0 else None,
        std_parametric_coverage=0.0 if success_rate > 0.0 else None,
        mean_parametric_interval_length=0.4 if success_rate > 0.0 else None,
        std_parametric_interval_length=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_bias=0.0 if success_rate > 0.0 else None,
        std_nonparametric_bias=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_rmse=0.3 if success_rate > 0.0 else None,
        mean_nonparametric_average_standard_error=0.3 if success_rate > 0.0 else None,
        std_nonparametric_average_standard_error=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_coverage=mean_nonparametric_coverage,
        std_nonparametric_coverage=0.0
        if mean_nonparametric_coverage is not None
        else None,
        mean_nonparametric_interval_length=12.0 if success_rate > 0.0 else None,
        std_nonparametric_interval_length=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_absolute_error=0.3 if success_rate > 0.0 else None,
        std_nonparametric_absolute_error=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_uniform_critical_value=1.5 if success_rate > 0.0 else None,
        std_nonparametric_uniform_critical_value=0.0 if success_rate > 0.0 else None,
        mean_nonparametric_uniform_band_length=13.0 if success_rate > 0.0 else None,
        std_nonparametric_uniform_band_length=0.0 if success_rate > 0.0 else None,
        mean_trimming_rate=0.0 if success_rate > 0.0 else None,
        mean_zero_valid_fold_frequency=0.0,
    )


def _make_canonical_runtime_probe() -> MonteCarloRuntimeProbeReport:
    return MonteCarloRuntimeProbeReport(
        oracle_lane="paper-trigonometric",
        stage_label="phase7-runtime-probe",
        random_states=(101, 202, 303),
        total_runtime_seconds=14.5,
        observations=(),
        design_summaries=(
            _make_design_summary(
                dgp_name="DGP1",
                n_obs=500,
                p=50,
                success_rate=1.0,
                runtime_mean_seconds=0.22,
                mean_nonparametric_coverage=1.0,
            ),
            _make_design_summary(
                dgp_name="DGP2",
                n_obs=500,
                p=50,
                success_rate=1.0,
                runtime_mean_seconds=0.30,
                mean_nonparametric_coverage=7.0 / 9.0,
            ),
            _make_design_summary(
                dgp_name="DGP1",
                n_obs=200,
                p=500,
                success_rate=0.0,
                runtime_mean_seconds=None,
                mean_nonparametric_coverage=None,
                typed_invalidity_counts={"SingularCovarianceError": 3},
            ),
            _make_design_summary(
                dgp_name="DGP2",
                n_obs=200,
                p=500,
                success_rate=0.0,
                runtime_mean_seconds=None,
                mean_nonparametric_coverage=None,
                typed_invalidity_counts={"SingularCovarianceError": 3},
            ),
        ),
        typed_invalidity_counts={"SingularCovarianceError": 6},
        typed_invalidity_examples={},
    )


def _make_canonical_execution_contract_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-execution-contract"
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
        left_shoulder_grid_value=0.05,
        center_grid_value=0.15,
        failing_right_shoulder_grid_value=0.25,
        coverage_anchor_local_scale_access_shortfall_share=0.10793658294037357,
        current_anchor_correlation=0.007249260352146767,
        required_repaired_correlation=0.13276252551993098,
        correlation_repair_increment=0.12551326516778422,
        required_gap_share=0.28771177790670194,
        anchor_left_shoulder_reserve=4.8678865,
        anchor_failing_right_shoulder_reserve=-1.0553945000000002,
        anchor_right_to_left_covariance_share=0.35169552888367206,
        companion_right_to_left_covariance_share=1.405546487549607,
        directional_flip_ratio=3.9964866542688124,
        current_abs_right_center_covariance=0.09449768365290503,
        required_abs_right_center_covariance=1.7306249918073264,
        required_incremental_right_center_covariance_lift=1.6361273081544214,
        required_increment_share_of_entry_gap=0.057182520128525725,
        required_increment_share_of_right_row_gap=0.009817410389160506,
        required_increment_share_of_center_column_gap=0.021671049134566682,
        residual_entry_gap_share_after_required_increment=0.9428174798714742,
        zero_partial_right_center_correlation=-3.3986126451197145,
        zero_partial_lower_bound_gap=2.3986126451197145,
        required_repair_distance_to_zero_partial_target=3.5313751706396457,
        shoulder_sigma_log_gap_share=0.16971581788552764,
        center_sigma_log_gap_share=0.11062685845075898,
        correlation_log_gap_share=0.7196573236637133,
        correlation_vs_shoulder_sigma_ratio=23.188108401264902,
        correlation_vs_center_sigma_ratio=32.50554722735936,
        driver_signature="bounded-right-center-execution-contract",
        canonical_execution_contract_digest=(
            "- binding design `DGP2/500/50` on `near_zero_grid`: the last local-scale miss still stays `10.8%` at failing right shoulder `z = 0.25`, yet the live lane only needs right-center correlation to rise from `0.007` to `0.133` (increment `+0.126`, `28.8%` of the full gap) and `|covariance(0.25, 0.15)|` to rise from `0.094` to `1.731` (increment `+1.636`) while left shoulder `z = 0.05` still keeps `+4.868` reserve and the right shoulder stays at `-1.055`",
            "- the miss therefore remains directional and right-specific: anchor seed `202` keeps only `35.2%` of left coupled access on the right side, whereas companion seed `505` restores the right side to `140.6%` of left access (`x3.996` directional flip), so the live repair priority still points to the bounded `z = 0.25 -> z = 0.15` entry",
            "- the required bounded lift stays tiny relative to companion geometry: `+1.636` consumes only `5.7%` of the single-entry gap, `1.0%` of the right-row gap, and `2.2%` of the center-column gap, leaving `94.3%` / `99.0%` / `97.8%` of those companion gaps outside the current lane even after repair",
            "- out-of-lane targets remain explicitly excluded: zeroing the center-conditioned direct residual would require right-center correlation `-3.399`, i.e. `2.399` below the feasible lower bound `-1.000` and `3.531` away from the bounded repair target, while shoulder / center `sigma_z_hat` still explain only `17.0%` / `11.1%` of the log gap versus `72.0%` from correlation",
            "- current Trigger 2 implication: `bounded-right-center-execution-contract`; next implementation should raise only the bounded `z = 0.25 -> z = 0.15` entry while preserving left-side support and leaving sign-healing, denominator compression, and whole-row / whole-column replay outside the live repair lane",
        ),
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPromotionGuardReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    coverage_anchor_local_scale_access_shortfall_share: float
    correlation_repair_increment: float
    required_gap_share: float
    required_incremental_right_center_covariance_lift: float
    supported_floor_ceiling: float
    canonical_floor: float
    canonical_floor_shortfall: float
    target_gate_status: str
    binding_guard: str
    driver_signature: str
    canonical_entry_patch_promotion_guard_digest: tuple[str, ...]

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
        self.coverage_anchor_local_scale_access_shortfall_share = float(
            self.coverage_anchor_local_scale_access_shortfall_share
        )
        self.correlation_repair_increment = float(self.correlation_repair_increment)
        self.required_gap_share = float(self.required_gap_share)
        self.required_incremental_right_center_covariance_lift = float(
            self.required_incremental_right_center_covariance_lift
        )
        self.supported_floor_ceiling = float(self.supported_floor_ceiling)
        self.canonical_floor = float(self.canonical_floor)
        self.canonical_floor_shortfall = float(self.canonical_floor_shortfall)
        self.target_gate_status = str(self.target_gate_status).strip()
        self.binding_guard = str(self.binding_guard).strip()
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_entry_patch_promotion_guard_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_entry_patch_promotion_guard_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "coverage_anchor_local_scale_access_shortfall_share": (
                self.coverage_anchor_local_scale_access_shortfall_share
            ),
            "correlation_repair_increment": self.correlation_repair_increment,
            "required_gap_share": self.required_gap_share,
            "required_incremental_right_center_covariance_lift": (
                self.required_incremental_right_center_covariance_lift
            ),
            "supported_floor_ceiling": self.supported_floor_ceiling,
            "canonical_floor": self.canonical_floor,
            "canonical_floor_shortfall": self.canonical_floor_shortfall,
            "target_gate_status": self.target_gate_status,
            "binding_guard": self.binding_guard,
            "driver_signature": self.driver_signature,
            "canonical_entry_patch_promotion_guard_digest": list(
                self.canonical_entry_patch_promotion_guard_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_promotion_guard_report(
    *,
    execution_contract_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderExecutionContractReport
    ),
    floor_slack_report: Phase7MonteCarloWideningPolicyFloorSlackProbeReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPromotionGuardReport:
    if execution_contract_report.policy_digest != floor_slack_report.policy_digest:
        raise ValueError("entry patch promotion guard requires a shared policy digest")
    if execution_contract_report.binding_design != floor_slack_report.binding_design:
        raise ValueError("entry patch promotion guard requires a shared binding design")

    target_gate_status = "trigger2-partial-widening-only"
    binding_guard = "nonparametric_coverage_floor"
    driver_signature = _driver_signature(
        execution_contract_report=execution_contract_report,
        floor_slack_report=floor_slack_report,
    )
    canonical_digest = (
        f"- binding design `{'/'.join(map(str, execution_contract_report.binding_design))}` on `{execution_contract_report.window_label}`: the bounded entry-patch lane still only targets the right shoulder `z = 0.25 -> 0.15` channel, with local-scale access shortfall `{_format_percent(execution_contract_report.coverage_anchor_local_scale_access_shortfall_share)}`, correlation increment `{_format_signed_float(execution_contract_report.correlation_repair_increment)}`, and covariance-entry increment `{_format_signed_float(execution_contract_report.required_incremental_right_center_covariance_lift)}` all scoped to the same validation-only repair object",
        f"- policy-level promotion guard remains stricter than the patch witness: current bounded slice still supports floor ceiling `{_format_float(floor_slack_report.supported_floor_ceiling)}`, so the canonical `{binding_guard} = {_format_float(floor_slack_report.coverage_floor)}` remains short by `{_format_float(floor_slack_report.canonical_floor_shortfall)}` even before asking whether a real estimator path can realize the patch",
        f"- the runtime gate therefore cannot promote past `{target_gate_status}`: a clean bounded patch may reduce the implementation risk on seed `{execution_contract_report.coverage_anchor_random_state}`, but it is still not evidence that the widened matrix has cleared the policy floor or earned fresh runtime promotion",
        f"- current Trigger 2 implication: `{driver_signature}`; keep the entry patch as validation-only implementation intake, and require a fresh estimator rerun that actually lifts the binding floor witness before treating Trigger 2 as promotion-ready",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPromotionGuardReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-entry-patch-promotion-guard-probe"
        ),
        policy_digest=execution_contract_report.policy_digest,
        binding_design=execution_contract_report.binding_design,
        window_label=execution_contract_report.window_label,
        coverage_anchor_random_state=(
            execution_contract_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            execution_contract_report.overshoot_companion_random_state
        ),
        coverage_anchor_local_scale_access_shortfall_share=(
            execution_contract_report.coverage_anchor_local_scale_access_shortfall_share
        ),
        correlation_repair_increment=execution_contract_report.correlation_repair_increment,
        required_gap_share=execution_contract_report.required_gap_share,
        required_incremental_right_center_covariance_lift=(
            execution_contract_report.required_incremental_right_center_covariance_lift
        ),
        supported_floor_ceiling=floor_slack_report.supported_floor_ceiling,
        canonical_floor=floor_slack_report.coverage_floor,
        canonical_floor_shortfall=floor_slack_report.canonical_floor_shortfall,
        target_gate_status=target_gate_status,
        binding_guard=binding_guard,
        driver_signature=driver_signature,
        canonical_entry_patch_promotion_guard_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_promotion_guard_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderEntryPatchPromotionGuardReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_promotion_guard_report(
        execution_contract_report=_make_canonical_execution_contract_report(),
        floor_slack_report=(
            build_phase7_monte_carlo_widening_policy_floor_slack_probe_report(
                _make_canonical_runtime_probe(),
                policy=build_phase7_canonical_monte_carlo_widening_policy(),
            )
        ),
    )
