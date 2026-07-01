from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_vf_entry_sandwich_probe,
)
from .monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_positive_first_sine_omega_diagonal_patch_fraction_contract,
)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _driver_signature(
    *,
    vf_entry_signature: str,
    diagonal_patch_signature: str,
    diagonal_coordinate: int,
    diagonal_basis_label: str,
    required_share_of_full_shared_vf_gap: float,
    required_share_of_omega_only_shared_vf_increment: float,
    required_share_of_diagonal_omega_gap: float,
    normalization_only_share_of_required_lift: float,
) -> str:
    if (
        vf_entry_signature == "omega-first-shared-vf-entry-activation"
        and diagonal_patch_signature
        == "bounded-positive-first-sine-omega-diagonal-patch-fraction"
        and diagonal_coordinate == 2
        and diagonal_basis_label == "sin(2πz)"
        and 0.10 < required_share_of_full_shared_vf_gap < 0.13
        and 0.17 < required_share_of_omega_only_shared_vf_increment < 0.19
        and 0.20 < required_share_of_diagonal_omega_gap < 0.25
        and normalization_only_share_of_required_lift < 0.1
    ):
        return "bounded-first-sine-shared-entry-patch-budget"
    return "mixed-first-sine-shared-entry-budget"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSharedEntryPatchBudgetReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    diagonal_coordinate: int
    diagonal_basis_label: str
    shared_vf_entry_label: str
    required_diagonal_vf_entry_lift: float
    full_shared_vf_entry_gap: float
    omega_only_shared_vf_entry_increment: float
    normalization_only_shared_vf_entry_increment: float
    required_share_of_full_shared_vf_gap: float
    residual_share_of_full_shared_vf_gap_after_patch: float
    required_share_of_omega_only_shared_vf_increment: float
    residual_share_of_omega_only_shared_vf_increment_after_patch: float
    required_share_of_diagonal_omega_gap: float
    residual_share_of_diagonal_omega_gap_after_patch: float
    required_omega_diagonal_increment: float
    bounded_target_omega_diagonal_entry: float
    driver_signature: str
    canonical_first_sine_shared_entry_patch_budget_digest: tuple[str, ...]

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
        self.diagonal_coordinate = int(self.diagonal_coordinate)
        self.diagonal_basis_label = str(self.diagonal_basis_label).strip()
        self.shared_vf_entry_label = str(self.shared_vf_entry_label).strip()
        self.required_diagonal_vf_entry_lift = float(
            self.required_diagonal_vf_entry_lift
        )
        self.full_shared_vf_entry_gap = float(self.full_shared_vf_entry_gap)
        self.omega_only_shared_vf_entry_increment = float(
            self.omega_only_shared_vf_entry_increment
        )
        self.normalization_only_shared_vf_entry_increment = float(
            self.normalization_only_shared_vf_entry_increment
        )
        self.required_share_of_full_shared_vf_gap = float(
            self.required_share_of_full_shared_vf_gap
        )
        self.residual_share_of_full_shared_vf_gap_after_patch = float(
            self.residual_share_of_full_shared_vf_gap_after_patch
        )
        self.required_share_of_omega_only_shared_vf_increment = float(
            self.required_share_of_omega_only_shared_vf_increment
        )
        self.residual_share_of_omega_only_shared_vf_increment_after_patch = float(
            self.residual_share_of_omega_only_shared_vf_increment_after_patch
        )
        self.required_share_of_diagonal_omega_gap = float(
            self.required_share_of_diagonal_omega_gap
        )
        self.residual_share_of_diagonal_omega_gap_after_patch = float(
            self.residual_share_of_diagonal_omega_gap_after_patch
        )
        self.required_omega_diagonal_increment = float(
            self.required_omega_diagonal_increment
        )
        self.bounded_target_omega_diagonal_entry = float(
            self.bounded_target_omega_diagonal_entry
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_first_sine_shared_entry_patch_budget_digest = tuple(
            str(line).rstrip()
            for line in self.canonical_first_sine_shared_entry_patch_budget_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "diagonal_coordinate": self.diagonal_coordinate,
            "diagonal_basis_label": self.diagonal_basis_label,
            "shared_vf_entry_label": self.shared_vf_entry_label,
            "required_diagonal_vf_entry_lift": self.required_diagonal_vf_entry_lift,
            "full_shared_vf_entry_gap": self.full_shared_vf_entry_gap,
            "omega_only_shared_vf_entry_increment": (
                self.omega_only_shared_vf_entry_increment
            ),
            "normalization_only_shared_vf_entry_increment": (
                self.normalization_only_shared_vf_entry_increment
            ),
            "required_share_of_full_shared_vf_gap": (
                self.required_share_of_full_shared_vf_gap
            ),
            "residual_share_of_full_shared_vf_gap_after_patch": (
                self.residual_share_of_full_shared_vf_gap_after_patch
            ),
            "required_share_of_omega_only_shared_vf_increment": (
                self.required_share_of_omega_only_shared_vf_increment
            ),
            "residual_share_of_omega_only_shared_vf_increment_after_patch": (
                self.residual_share_of_omega_only_shared_vf_increment_after_patch
            ),
            "required_share_of_diagonal_omega_gap": (
                self.required_share_of_diagonal_omega_gap
            ),
            "residual_share_of_diagonal_omega_gap_after_patch": (
                self.residual_share_of_diagonal_omega_gap_after_patch
            ),
            "required_omega_diagonal_increment": self.required_omega_diagonal_increment,
            "bounded_target_omega_diagonal_entry": (
                self.bounded_target_omega_diagonal_entry
            ),
            "driver_signature": self.driver_signature,
            "canonical_first_sine_shared_entry_patch_budget_digest": list(
                self.canonical_first_sine_shared_entry_patch_budget_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_shared_entry_patch_budget_report(
    *,
    vf_entry_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineVFEntrySandwichReport,
    diagonal_patch_report: Phase7MonteCarloWideningPolicyCoverageAnchorShoulderPositiveFirstSineOmegaDiagonalPatchFractionContractReport,
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSharedEntryPatchBudgetReport:
    if vf_entry_report.policy_digest != diagonal_patch_report.policy_digest:
        raise ValueError("shared-entry patch budget requires shared policy digest")
    if vf_entry_report.binding_design != diagonal_patch_report.binding_design:
        raise ValueError("shared-entry patch budget requires shared binding design")
    if vf_entry_report.window_label != diagonal_patch_report.window_label:
        raise ValueError("shared-entry patch budget requires shared window label")
    if (
        vf_entry_report.coverage_anchor_random_state
        != diagonal_patch_report.coverage_anchor_random_state
    ):
        raise ValueError("coverage-anchor seed must match across upstream reports")
    if (
        vf_entry_report.overshoot_companion_random_state
        != diagonal_patch_report.overshoot_companion_random_state
    ):
        raise ValueError("overshoot-companion seed must match across upstream reports")
    if vf_entry_report.diagonal_coordinate != diagonal_patch_report.diagonal_coordinate:
        raise ValueError("diagonal coordinate must match across upstream reports")
    if (
        vf_entry_report.diagonal_basis_label
        != diagonal_patch_report.diagonal_basis_label
    ):
        raise ValueError("diagonal basis label must match across upstream reports")
    if (
        vf_entry_report.shared_vf_entry_label
        != diagonal_patch_report.shared_vf_entry_label
    ):
        raise ValueError("shared entry label must match across upstream reports")

    required_diagonal_vf_entry_lift = float(
        vf_entry_report.required_diagonal_vf_entry_lift
    )
    full_shared_vf_entry_gap = float(vf_entry_report.full_vf_entry_gap)
    omega_only_shared_vf_entry_increment = float(vf_entry_report.omega_only_increment)
    normalization_only_shared_vf_entry_increment = float(
        vf_entry_report.normalization_only_increment
    )
    if required_diagonal_vf_entry_lift <= 0.0:
        raise ValueError("required shared-entry lift must stay positive")
    if full_shared_vf_entry_gap <= 0.0:
        raise ValueError("full shared-entry gap must stay positive")
    if omega_only_shared_vf_entry_increment <= 0.0:
        raise ValueError("omega-only shared-entry increment must stay positive")

    required_share_of_full_shared_vf_gap = float(
        required_diagonal_vf_entry_lift / full_shared_vf_entry_gap
    )
    residual_share_of_full_shared_vf_gap_after_patch = float(
        (full_shared_vf_entry_gap - required_diagonal_vf_entry_lift)
        / full_shared_vf_entry_gap
    )
    required_share_of_omega_only_shared_vf_increment = float(
        required_diagonal_vf_entry_lift / omega_only_shared_vf_entry_increment
    )
    residual_share_of_omega_only_shared_vf_increment_after_patch = float(
        (omega_only_shared_vf_entry_increment - required_diagonal_vf_entry_lift)
        / omega_only_shared_vf_entry_increment
    )
    required_share_of_diagonal_omega_gap = float(
        diagonal_patch_report.required_patch_share_of_omega_diagonal_gap
    )
    residual_share_of_diagonal_omega_gap_after_patch = float(
        diagonal_patch_report.residual_companion_gap_share_after_patch
    )

    driver_signature = _driver_signature(
        vf_entry_signature=vf_entry_report.driver_signature,
        diagonal_patch_signature=diagonal_patch_report.driver_signature,
        diagonal_coordinate=vf_entry_report.diagonal_coordinate,
        diagonal_basis_label=vf_entry_report.diagonal_basis_label,
        required_share_of_full_shared_vf_gap=required_share_of_full_shared_vf_gap,
        required_share_of_omega_only_shared_vf_increment=(
            required_share_of_omega_only_shared_vf_increment
        ),
        required_share_of_diagonal_omega_gap=required_share_of_diagonal_omega_gap,
        normalization_only_share_of_required_lift=(
            vf_entry_report.normalization_only_share_of_required_lift
        ),
    )

    canonical_digest = (
        f"- the bounded `{_format_signed(required_diagonal_vf_entry_lift)}` repair target still consumes only `{_format_percent(required_share_of_full_shared_vf_gap)}` of the full shared `{vf_entry_report.shared_vf_entry_label}` gap `{_format_signed(full_shared_vf_entry_gap)}`, so `{_format_percent(residual_share_of_full_shared_vf_gap_after_patch)}` of the anchor-to-companion shared-entry gap remains outside the live patch lane",
        f"- even inside the omega-led shared-entry channel, the bounded target uses only `{_format_percent(required_share_of_omega_only_shared_vf_increment)}` of the omega-only shared-entry increment `{_format_signed(omega_only_shared_vf_entry_increment)}`, leaving `{_format_percent(residual_share_of_omega_only_shared_vf_increment_after_patch)}` of that omega-driven lift unused while normalization-only drift still explains only `{_format_signed(normalization_only_shared_vf_entry_increment)}` (`{_format_percent(vf_entry_report.normalization_only_share_of_required_lift)}` of the bounded target)",
        f"- the source budget remains narrower still on the positive first-sine diagonal: only `{_format_percent(required_share_of_diagonal_omega_gap)}` of the companion `omega_f_hat[2,2]` gap is needed, so the bounded target stays at `omega_f_hat[2,2] = {_format_float(diagonal_patch_report.bounded_target_omega_diagonal_entry)}` after a `{_format_signed(diagonal_patch_report.required_omega_diagonal_increment)}` lift and keeps `{_format_percent(residual_share_of_diagonal_omega_gap_after_patch)}` of the diagonal gap unused",
        f"- current Trigger 2 implication: `{driver_signature}`; next implementation should treat `omega_f_hat[2,2] -> {vf_entry_report.shared_vf_entry_label}` as a nested bounded budget, not as permission for full companion replay, normalization retuning, or broad covariance overwrite",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSharedEntryPatchBudgetReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-shared-entry-patch-budget",
        policy_digest=vf_entry_report.policy_digest,
        binding_design=vf_entry_report.binding_design,
        window_label=vf_entry_report.window_label,
        coverage_anchor_random_state=vf_entry_report.coverage_anchor_random_state,
        overshoot_companion_random_state=vf_entry_report.overshoot_companion_random_state,
        diagonal_coordinate=vf_entry_report.diagonal_coordinate,
        diagonal_basis_label=vf_entry_report.diagonal_basis_label,
        shared_vf_entry_label=vf_entry_report.shared_vf_entry_label,
        required_diagonal_vf_entry_lift=required_diagonal_vf_entry_lift,
        full_shared_vf_entry_gap=full_shared_vf_entry_gap,
        omega_only_shared_vf_entry_increment=omega_only_shared_vf_entry_increment,
        normalization_only_shared_vf_entry_increment=(
            normalization_only_shared_vf_entry_increment
        ),
        required_share_of_full_shared_vf_gap=required_share_of_full_shared_vf_gap,
        residual_share_of_full_shared_vf_gap_after_patch=(
            residual_share_of_full_shared_vf_gap_after_patch
        ),
        required_share_of_omega_only_shared_vf_increment=(
            required_share_of_omega_only_shared_vf_increment
        ),
        residual_share_of_omega_only_shared_vf_increment_after_patch=(
            residual_share_of_omega_only_shared_vf_increment_after_patch
        ),
        required_share_of_diagonal_omega_gap=required_share_of_diagonal_omega_gap,
        residual_share_of_diagonal_omega_gap_after_patch=(
            residual_share_of_diagonal_omega_gap_after_patch
        ),
        required_omega_diagonal_increment=(
            diagonal_patch_report.required_omega_diagonal_increment
        ),
        bounded_target_omega_diagonal_entry=(
            diagonal_patch_report.bounded_target_omega_diagonal_entry
        ),
        driver_signature=driver_signature,
        canonical_first_sine_shared_entry_patch_budget_digest=canonical_digest,
    )


def _build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_shared_entry_patch_budget_snapshot_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSharedEntryPatchBudgetReport
):
    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSharedEntryPatchBudgetReport(
        stage_label="phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-first-sine-shared-entry-patch-budget",
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
        diagonal_coordinate=2,
        diagonal_basis_label="sin(2πz)",
        shared_vf_entry_label="v_f_hat[2,2]",
        required_diagonal_vf_entry_lift=1011.1822863613054,
        full_shared_vf_entry_gap=8445.875220536345,
        omega_only_shared_vf_entry_increment=5662.7423277793705,
        normalization_only_shared_vf_entry_increment=62.80420450378551,
        required_share_of_full_shared_vf_gap=0.11972498526885547,
        residual_share_of_full_shared_vf_gap_after_patch=0.8802750147311446,
        required_share_of_omega_only_shared_vf_increment=0.17856759637478292,
        residual_share_of_omega_only_shared_vf_increment_after_patch=0.8214324036252171,
        required_share_of_diagonal_omega_gap=0.24167652126504635,
        residual_share_of_diagonal_omega_gap_after_patch=0.7583234787349536,
        required_omega_diagonal_increment=185.45799327602003,
        bounded_target_omega_diagonal_entry=410.4739959618635,
        driver_signature="bounded-first-sine-shared-entry-patch-budget",
        canonical_first_sine_shared_entry_patch_budget_digest=(
            "- the bounded `+1011.182` repair target still consumes only `12.0%` of the full shared `v_f_hat[2,2]` gap `+8445.875`, so `88.0%` of the anchor-to-companion shared-entry gap remains outside the live patch lane",
            "- even inside the omega-led shared-entry channel, the bounded target uses only `17.9%` of the omega-only shared-entry increment `+5662.742`, leaving `82.1%` of that omega-driven lift unused while normalization-only drift still explains only `+62.804` (`6.2%` of the bounded target)",
            "- the source budget remains narrower still on the positive first-sine diagonal: only `24.2%` of the companion `omega_f_hat[2,2]` gap is needed, so the bounded target stays at `omega_f_hat[2,2] = 410.474` after a `+185.458` lift and keeps `75.8%` of the diagonal gap unused",
            "- current Trigger 2 implication: `bounded-first-sine-shared-entry-patch-budget`; next implementation should treat `omega_f_hat[2,2] -> v_f_hat[2,2]` as a nested bounded budget, not as permission for full companion replay, normalization retuning, or broad covariance overwrite",
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_shared_entry_patch_budget() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineSharedEntryPatchBudgetReport
):
    return _build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_shared_entry_patch_budget_snapshot_report()
