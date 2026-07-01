from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose, sqrt

from .monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe import (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport,
    run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe,
)


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    return f"{float(value):.2f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _format_signed_float(value: float) -> str:
    return f"{float(value):+.3f}"


def _center_conditioned_partial_correlation(
    cross_shoulder_correlation: float,
    *,
    left_center_correlation: float,
    right_center_correlation: float,
) -> float:
    left_center = float(left_center_correlation)
    right_center = float(right_center_correlation)
    denominator = (1.0 - left_center * left_center) * (
        1.0 - right_center * right_center
    )
    if denominator <= 0.0:
        raise ValueError(
            "partial correlation requires strictly positive conditioning variance"
        )
    return float(
        (float(cross_shoulder_correlation) - left_center * right_center)
        / sqrt(denominator)
    )


def _positive_ratio(numerator: float, denominator: float, *, label: str) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    return float(float(numerator) / denominator_value)


def _driver_signature(
    *,
    cross_shoulder_sign_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport
    ),
    anchor_partial_correlation: float,
    companion_partial_correlation: float,
) -> str:
    if (
        cross_shoulder_sign_report.driver_signature == "cross-shoulder-sign-fracture"
        and anchor_partial_correlation < 0.0
        and companion_partial_correlation > 0.0
    ):
        return "center-conditioned-cross-shoulder-sign-fracture"
    if anchor_partial_correlation >= 0.0:
        return "center-conditioned-sign-fracture-cleared"
    return "mixed-center-conditioned-cross-shoulder-geometry"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderPartialReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    left_shoulder_grid_value: float
    center_grid_value: float
    failing_right_shoulder_grid_value: float
    anchor_cross_shoulder_correlation: float
    companion_cross_shoulder_correlation: float
    anchor_center_mediated_cross_shoulder_correlation: float
    companion_center_mediated_cross_shoulder_correlation: float
    anchor_center_conditioned_cross_shoulder_partial_correlation: float
    companion_center_conditioned_cross_shoulder_partial_correlation: float
    anchor_center_mediated_share_of_abs_cross_shoulder: float
    companion_center_mediated_share_of_cross_shoulder: float
    center_conditioned_partial_span: float
    driver_signature: str
    canonical_cross_shoulder_partial_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.left_shoulder_grid_value = float(self.left_shoulder_grid_value)
        self.center_grid_value = float(self.center_grid_value)
        self.failing_right_shoulder_grid_value = float(
            self.failing_right_shoulder_grid_value
        )
        self.anchor_cross_shoulder_correlation = float(
            self.anchor_cross_shoulder_correlation
        )
        self.companion_cross_shoulder_correlation = float(
            self.companion_cross_shoulder_correlation
        )
        self.anchor_center_mediated_cross_shoulder_correlation = float(
            self.anchor_center_mediated_cross_shoulder_correlation
        )
        self.companion_center_mediated_cross_shoulder_correlation = float(
            self.companion_center_mediated_cross_shoulder_correlation
        )
        self.anchor_center_conditioned_cross_shoulder_partial_correlation = float(
            self.anchor_center_conditioned_cross_shoulder_partial_correlation
        )
        self.companion_center_conditioned_cross_shoulder_partial_correlation = float(
            self.companion_center_conditioned_cross_shoulder_partial_correlation
        )
        self.anchor_center_mediated_share_of_abs_cross_shoulder = float(
            self.anchor_center_mediated_share_of_abs_cross_shoulder
        )
        self.companion_center_mediated_share_of_cross_shoulder = float(
            self.companion_center_mediated_share_of_cross_shoulder
        )
        self.center_conditioned_partial_span = float(
            self.center_conditioned_partial_span
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_cross_shoulder_partial_digest = tuple(
            str(line).rstrip() for line in self.canonical_cross_shoulder_partial_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "left_shoulder_grid_value": self.left_shoulder_grid_value,
            "center_grid_value": self.center_grid_value,
            "failing_right_shoulder_grid_value": self.failing_right_shoulder_grid_value,
            "anchor_cross_shoulder_correlation": self.anchor_cross_shoulder_correlation,
            "companion_cross_shoulder_correlation": self.companion_cross_shoulder_correlation,
            "anchor_center_mediated_cross_shoulder_correlation": (
                self.anchor_center_mediated_cross_shoulder_correlation
            ),
            "companion_center_mediated_cross_shoulder_correlation": (
                self.companion_center_mediated_cross_shoulder_correlation
            ),
            "anchor_center_conditioned_cross_shoulder_partial_correlation": (
                self.anchor_center_conditioned_cross_shoulder_partial_correlation
            ),
            "companion_center_conditioned_cross_shoulder_partial_correlation": (
                self.companion_center_conditioned_cross_shoulder_partial_correlation
            ),
            "anchor_center_mediated_share_of_abs_cross_shoulder": (
                self.anchor_center_mediated_share_of_abs_cross_shoulder
            ),
            "companion_center_mediated_share_of_cross_shoulder": (
                self.companion_center_mediated_share_of_cross_shoulder
            ),
            "center_conditioned_partial_span": self.center_conditioned_partial_span,
            "driver_signature": self.driver_signature,
            "canonical_cross_shoulder_partial_digest": list(
                self.canonical_cross_shoulder_partial_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_partial_report(
    *,
    cross_shoulder_sign_report: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderSignReport
    ),
) -> Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderPartialReport:
    anchor_center_mediated_cross_shoulder_correlation = float(
        cross_shoulder_sign_report.anchor_left_center_correlation
        * cross_shoulder_sign_report.anchor_right_center_correlation
    )
    companion_center_mediated_cross_shoulder_correlation = float(
        cross_shoulder_sign_report.companion_left_center_correlation
        * cross_shoulder_sign_report.companion_right_center_correlation
    )
    anchor_center_conditioned_cross_shoulder_partial_correlation = (
        _center_conditioned_partial_correlation(
            cross_shoulder_sign_report.anchor_cross_shoulder_correlation,
            left_center_correlation=(
                cross_shoulder_sign_report.anchor_left_center_correlation
            ),
            right_center_correlation=(
                cross_shoulder_sign_report.anchor_right_center_correlation
            ),
        )
    )
    companion_center_conditioned_cross_shoulder_partial_correlation = (
        _center_conditioned_partial_correlation(
            cross_shoulder_sign_report.companion_cross_shoulder_correlation,
            left_center_correlation=(
                cross_shoulder_sign_report.companion_left_center_correlation
            ),
            right_center_correlation=(
                cross_shoulder_sign_report.companion_right_center_correlation
            ),
        )
    )
    anchor_center_mediated_share_of_abs_cross_shoulder = _positive_ratio(
        abs(anchor_center_mediated_cross_shoulder_correlation),
        abs(cross_shoulder_sign_report.anchor_cross_shoulder_correlation),
        label="anchor_center_mediated_share_of_abs_cross_shoulder",
    )
    companion_center_mediated_share_of_cross_shoulder = _positive_ratio(
        companion_center_mediated_cross_shoulder_correlation,
        cross_shoulder_sign_report.companion_cross_shoulder_correlation,
        label="companion_center_mediated_share_of_cross_shoulder",
    )
    center_conditioned_partial_span = float(
        companion_center_conditioned_cross_shoulder_partial_correlation
        - anchor_center_conditioned_cross_shoulder_partial_correlation
    )
    if center_conditioned_partial_span <= 0.0:
        raise ValueError("center-conditioned partial span must stay positive")
    driver_signature = _driver_signature(
        cross_shoulder_sign_report=cross_shoulder_sign_report,
        anchor_partial_correlation=(
            anchor_center_conditioned_cross_shoulder_partial_correlation
        ),
        companion_partial_correlation=(
            companion_center_conditioned_cross_shoulder_partial_correlation
        ),
    )

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*cross_shoulder_sign_report.binding_design)}` on "
        f"`{cross_shoulder_sign_report.window_label}`: cross-shoulder "
        f"`corr({_format_grid_value(cross_shoulder_sign_report.left_shoulder_grid_value)}, "
        f"{_format_grid_value(cross_shoulder_sign_report.failing_right_shoulder_grid_value)})` "
        f"stays at `{_format_float(cross_shoulder_sign_report.anchor_cross_shoulder_correlation)}` "
        f"for coverage anchor seed `{cross_shoulder_sign_report.coverage_anchor_random_state}`, "
        f"versus `{_format_signed_float(cross_shoulder_sign_report.companion_cross_shoulder_correlation)}` "
        f"for overshoot companion seed `{cross_shoulder_sign_report.overshoot_companion_random_state}`, "
        f"while center `z = {_format_grid_value(cross_shoulder_sign_report.center_grid_value)}` remains fixed",
        "- the center-mediated bridge "
        f"`corr({_format_grid_value(cross_shoulder_sign_report.left_shoulder_grid_value)}, "
        f"{_format_grid_value(cross_shoulder_sign_report.center_grid_value)}) * "
        f"corr({_format_grid_value(cross_shoulder_sign_report.failing_right_shoulder_grid_value)}, "
        f"{_format_grid_value(cross_shoulder_sign_report.center_grid_value)})` "
        f"is only `{_format_signed_float(anchor_center_mediated_cross_shoulder_correlation)}` "
        f"for anchor seed `{cross_shoulder_sign_report.coverage_anchor_random_state}` and "
        f"`{_format_signed_float(companion_center_mediated_cross_shoulder_correlation)}` "
        f"for companion seed `{cross_shoulder_sign_report.overshoot_companion_random_state}`; "
        "conditioning on center still leaves partial cross-shoulder correlation at "
        f"`{_format_float(anchor_center_conditioned_cross_shoulder_partial_correlation)}` "
        "for anchor and "
        f"`{_format_signed_float(companion_center_conditioned_cross_shoulder_partial_correlation)}` "
        "for companion",
        "- center mediation therefore explains only "
        f"`{_format_percent(anchor_center_mediated_share_of_abs_cross_shoulder)}` "
        "of the anchor's absolute cross-shoulder magnitude and "
        f"`{_format_percent(companion_center_mediated_share_of_cross_shoulder)}` "
        "of the companion's positive cross-shoulder level; the conditioned sign span still stays at "
        f"`{_format_signed_float(center_conditioned_partial_span)}`",
        "- current Trigger 2 implication: "
        f"`{driver_signature}`; source-level follow-up should explain why seed "
        f"`{cross_shoulder_sign_report.coverage_anchor_random_state}` keeps a negative direct left-right "
        "shoulder residual even after center alignment, not assume center mediation alone will heal the sign flip",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderPartialReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-cross-shoulder-partial-probe"
        ),
        policy_digest=cross_shoulder_sign_report.policy_digest,
        binding_design=cross_shoulder_sign_report.binding_design,
        coverage_anchor_random_state=(
            cross_shoulder_sign_report.coverage_anchor_random_state
        ),
        overshoot_companion_random_state=(
            cross_shoulder_sign_report.overshoot_companion_random_state
        ),
        left_shoulder_grid_value=cross_shoulder_sign_report.left_shoulder_grid_value,
        center_grid_value=cross_shoulder_sign_report.center_grid_value,
        failing_right_shoulder_grid_value=(
            cross_shoulder_sign_report.failing_right_shoulder_grid_value
        ),
        anchor_cross_shoulder_correlation=(
            cross_shoulder_sign_report.anchor_cross_shoulder_correlation
        ),
        companion_cross_shoulder_correlation=(
            cross_shoulder_sign_report.companion_cross_shoulder_correlation
        ),
        anchor_center_mediated_cross_shoulder_correlation=(
            anchor_center_mediated_cross_shoulder_correlation
        ),
        companion_center_mediated_cross_shoulder_correlation=(
            companion_center_mediated_cross_shoulder_correlation
        ),
        anchor_center_conditioned_cross_shoulder_partial_correlation=(
            anchor_center_conditioned_cross_shoulder_partial_correlation
        ),
        companion_center_conditioned_cross_shoulder_partial_correlation=(
            companion_center_conditioned_cross_shoulder_partial_correlation
        ),
        anchor_center_mediated_share_of_abs_cross_shoulder=(
            anchor_center_mediated_share_of_abs_cross_shoulder
        ),
        companion_center_mediated_share_of_cross_shoulder=(
            companion_center_mediated_share_of_cross_shoulder
        ),
        center_conditioned_partial_span=center_conditioned_partial_span,
        driver_signature=driver_signature,
        canonical_cross_shoulder_partial_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_partial_probe() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderCrossShoulderPartialReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_partial_report(
        cross_shoulder_sign_report=(
            run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe()
        ),
    )
