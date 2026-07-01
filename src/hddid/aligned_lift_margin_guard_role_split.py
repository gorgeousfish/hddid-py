from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from hddid.aligned_lift_margin_guard_metric_separation import (
    Phase7AlignedLiftMarginGuardMetricSeparationReport,
    run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_metric_separation_probe,
)
from hddid.aligned_lift_margin_guard_profile import (
    Phase7AlignedLiftMarginGuardProfileReport,
    run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_probe,
)
from hddid.aligned_lift_margin_guard_reserve import (
    Phase7AlignedLiftMarginGuardReserveReport,
    run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_reserve_probe,
)


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardMetricRole:
    metric: str
    collapse_binding_consensus: bool
    finite_binding_consensus: bool
    crossing_separation: float
    boundary_separation: float
    reserve_share: float
    is_weakest_crossing_separation: bool
    is_weakest_boundary_separation: bool
    is_weakest_reserve_share: bool
    is_strongest_reserve_share: bool
    role_label: str

    def __post_init__(self) -> None:
        self.metric = str(self.metric).strip()
        self.collapse_binding_consensus = bool(self.collapse_binding_consensus)
        self.finite_binding_consensus = bool(self.finite_binding_consensus)
        self.crossing_separation = float(self.crossing_separation)
        self.boundary_separation = float(self.boundary_separation)
        self.reserve_share = float(self.reserve_share)
        self.is_weakest_crossing_separation = bool(self.is_weakest_crossing_separation)
        self.is_weakest_boundary_separation = bool(self.is_weakest_boundary_separation)
        self.is_weakest_reserve_share = bool(self.is_weakest_reserve_share)
        self.is_strongest_reserve_share = bool(self.is_strongest_reserve_share)
        self.role_label = str(self.role_label).strip()

    def canonical_digest_line(self) -> str:
        return (
            f"- `{self.metric}`: role `{self.role_label}`, "
            f"collapse binding consensus `{self.collapse_binding_consensus}`, "
            f"finite binding consensus `{self.finite_binding_consensus}`, "
            f"`303-308` separation `{self.crossing_separation:+.3f}`, "
            f"`308-296` separation `{self.boundary_separation:+.3f}`, "
            f"reserve share `{self.reserve_share:.3%}`"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "collapse_binding_consensus": self.collapse_binding_consensus,
            "finite_binding_consensus": self.finite_binding_consensus,
            "crossing_separation": self.crossing_separation,
            "boundary_separation": self.boundary_separation,
            "reserve_share": self.reserve_share,
            "is_weakest_crossing_separation": self.is_weakest_crossing_separation,
            "is_weakest_boundary_separation": self.is_weakest_boundary_separation,
            "is_weakest_reserve_share": self.is_weakest_reserve_share,
            "is_strongest_reserve_share": self.is_strongest_reserve_share,
            "role_label": self.role_label,
            "canonical_digest_line": self.canonical_digest_line(),
        }


@dataclass(slots=True)
class Phase7AlignedLiftMarginGuardRoleSplitReport:
    stage_label: str
    metric_roles: tuple[Phase7AlignedLiftMarginGuardMetricRole, ...]
    crossing_anchor_metric: str
    bridge_metric: str
    overshoot_companion_metric: str
    canonical_role_digest: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.metric_roles = tuple(self.metric_roles)
        self.crossing_anchor_metric = str(self.crossing_anchor_metric).strip()
        self.bridge_metric = str(self.bridge_metric).strip()
        self.overshoot_companion_metric = str(self.overshoot_companion_metric).strip()
        self.canonical_role_digest = tuple(
            str(line).rstrip() for line in self.canonical_role_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "metric_roles": [metric.to_dict() for metric in self.metric_roles],
            "crossing_anchor_metric": self.crossing_anchor_metric,
            "bridge_metric": self.bridge_metric,
            "overshoot_companion_metric": self.overshoot_companion_metric,
            "canonical_role_digest": list(self.canonical_role_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


def _profile_focuses(
    profile_report: Phase7AlignedLiftMarginGuardProfileReport,
):
    return (
        profile_report.threshold_crossing_focus,
        profile_report.primary_boundary_focus,
        profile_report.secondary_support_focus,
    )


def _role_label(
    *,
    collapse_binding_consensus: bool,
    finite_binding_consensus: bool,
    is_weakest_crossing_separation: bool,
    is_weakest_boundary_separation: bool,
    is_weakest_reserve_share: bool,
    is_strongest_reserve_share: bool,
) -> str:
    if (
        collapse_binding_consensus
        and finite_binding_consensus
        and is_weakest_crossing_separation
        and is_weakest_reserve_share
    ):
        return "crossing_anchor"
    if is_weakest_boundary_separation and is_strongest_reserve_share:
        return "overshoot_companion"
    return "bridge_metric"


def build_phase7_nonparametric_source_level_aligned_lift_margin_guard_role_split_report(
    profile_report: Phase7AlignedLiftMarginGuardProfileReport,
    metric_separation_report: Phase7AlignedLiftMarginGuardMetricSeparationReport,
    reserve_report: Phase7AlignedLiftMarginGuardReserveReport,
) -> Phase7AlignedLiftMarginGuardRoleSplitReport:
    focuses = _profile_focuses(profile_report)
    weakest_crossing_metric = (
        metric_separation_report.weakest_crossing_separation_metric
    )
    weakest_boundary_metric = (
        metric_separation_report.weakest_boundary_separation_metric
    )
    weakest_reserve_metric = reserve_report.weakest_reserve_metric
    strongest_reserve_metric = reserve_report.strongest_reserve_metric
    metric_roles = []
    for metric_separation, reserve_metric in zip(
        metric_separation_report.metric_separations,
        reserve_report.metric_budgets,
    ):
        metric = metric_separation.metric
        collapse_binding_consensus = all(
            focus.collapse_binding_metric == metric for focus in focuses
        )
        finite_binding_consensus = all(
            focus.finite_binding_metric == metric for focus in focuses
        )
        metric_roles.append(
            Phase7AlignedLiftMarginGuardMetricRole(
                metric=metric,
                collapse_binding_consensus=collapse_binding_consensus,
                finite_binding_consensus=finite_binding_consensus,
                crossing_separation=(
                    metric_separation.threshold_crossing_minus_primary_boundary
                ),
                boundary_separation=(
                    metric_separation.primary_boundary_minus_secondary_support
                ),
                reserve_share=reserve_metric.reserve_share,
                is_weakest_crossing_separation=(metric == weakest_crossing_metric),
                is_weakest_boundary_separation=(metric == weakest_boundary_metric),
                is_weakest_reserve_share=(metric == weakest_reserve_metric),
                is_strongest_reserve_share=(metric == strongest_reserve_metric),
                role_label=_role_label(
                    collapse_binding_consensus=collapse_binding_consensus,
                    finite_binding_consensus=finite_binding_consensus,
                    is_weakest_crossing_separation=(metric == weakest_crossing_metric),
                    is_weakest_boundary_separation=(metric == weakest_boundary_metric),
                    is_weakest_reserve_share=(metric == weakest_reserve_metric),
                    is_strongest_reserve_share=(metric == strongest_reserve_metric),
                ),
            )
        )
    metric_roles = tuple(metric_roles)
    role_lookup = {metric.role_label: metric.metric for metric in metric_roles}
    canonical_role_digest = tuple(
        metric.canonical_digest_line() for metric in metric_roles
    )
    return Phase7AlignedLiftMarginGuardRoleSplitReport(
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-margin-guard-role-split-probe"
        ),
        metric_roles=metric_roles,
        crossing_anchor_metric=role_lookup["crossing_anchor"],
        bridge_metric=role_lookup["bridge_metric"],
        overshoot_companion_metric=role_lookup["overshoot_companion"],
        canonical_role_digest=canonical_role_digest,
        recommendation_rationale=(
            "`gap` remains the crossing anchor because it is the binding guard "
            "for `296/308/303`, the weakest `303-308` separation, and the "
            "smallest reserve share on the `308 -> 303` jump. `center_ratio` "
            "plays the overshoot companion because it is simultaneously the "
            "weakest `308-296` separation and the strongest reserve share, "
            "while `ratio` remains the bridge metric between those two roles."
        ),
    )


@lru_cache(maxsize=1)
def run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_role_split_probe() -> (
    Phase7AlignedLiftMarginGuardRoleSplitReport
):
    return build_phase7_nonparametric_source_level_aligned_lift_margin_guard_role_split_report(
        run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_profile_probe(),
        run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_metric_separation_probe(),
        run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_reserve_probe(),
    )


def run_phase7_nonparametric_source_level_aligned_lift_margin_guard_role_split_probe() -> (
    Phase7AlignedLiftMarginGuardRoleSplitReport
):
    return run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_guard_role_split_probe()
