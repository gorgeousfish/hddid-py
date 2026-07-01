from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hddid.outer_inference_compare_readiness import (
    Phase7OuterInferenceCompareReadinessReport,
    run_phase7_outer_inference_compare_readiness,
)
from hddid.r_snapshot_callable_oracle_boundary import (
    Phase7RSnapshotCallableOracleBoundaryReport,
    run_phase7_r_snapshot_callable_oracle_boundary,
)


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


@dataclass(frozen=True)
class Phase7RSnapshotPaperBackedCompareBoundaryReport:
    stage_label: str
    callable_oracle_status: str
    callable_parity_oracle_exists: bool
    compare_readiness_status: str
    paper_backed_compare_route: str
    object_level_compare_fields: tuple[str, ...]
    blocked_band_field: str
    open_rbug_codes: tuple[str, ...]
    repair_gate_status: str
    boundary_digest: tuple[str, ...]


def build_phase7_r_snapshot_paper_backed_compare_boundary_report(
    report: Phase7RSnapshotPaperBackedCompareBoundaryReport,
) -> str:
    lines = [
        "# Phase 7 R snapshot paper-backed compare boundary",
        "",
        f"- Stage label: `{report.stage_label}`",
        f"- Callable oracle status: `{report.callable_oracle_status}`",
        f"- Compare readiness: `{report.compare_readiness_status}`",
        f"- Paper-backed compare route: `{report.paper_backed_compare_route}`",
        f"- Repair gate: `{report.repair_gate_status}`",
        (f"- Open R-bug codes: `{', '.join(report.open_rbug_codes)}`"),
        (
            "- Object-level compare fields: "
            f"`{', '.join(report.object_level_compare_fields)}`"
        ),
        f"- Blocked band field: `{report.blocked_band_field}`",
        "",
        "## Boundary digest",
        "",
    ]
    lines.extend(report.boundary_digest)
    return "\n".join(lines)


def _collect_open_rbug_codes(
    callable_boundary: Phase7RSnapshotCallableOracleBoundaryReport,
    compare_readiness: Phase7OuterInferenceCompareReadinessReport,
) -> tuple[str, ...]:
    ordered_codes = (
        *callable_boundary.source_chain_finding_codes,
        *callable_boundary.example_interface_finding_codes,
        *compare_readiness.blocking_finding_codes,
    )
    unique_codes: list[str] = []
    for code in ordered_codes:
        if code not in unique_codes:
            unique_codes.append(code)
    return tuple(unique_codes)


def run_phase7_r_snapshot_paper_backed_compare_boundary(
    repo_root: str | Path,
) -> Phase7RSnapshotPaperBackedCompareBoundaryReport:
    root = _coerce_repo_root(repo_root)
    callable_boundary = run_phase7_r_snapshot_callable_oracle_boundary(root)
    compare_readiness = run_phase7_outer_inference_compare_readiness()
    open_rbug_codes = _collect_open_rbug_codes(callable_boundary, compare_readiness)

    return Phase7RSnapshotPaperBackedCompareBoundaryReport(
        stage_label="phase7-r-snapshot-paper-backed-compare-boundary",
        callable_oracle_status=callable_boundary.status,
        callable_parity_oracle_exists=callable_boundary.callable_parity_oracle_exists,
        compare_readiness_status=compare_readiness.compare_readiness_status,
        paper_backed_compare_route=compare_readiness.replacement_target,
        object_level_compare_fields=compare_readiness.object_level_compare_fields,
        blocked_band_field=compare_readiness.blocked_band_field,
        open_rbug_codes=open_rbug_codes,
        repair_gate_status=compare_readiness.repair_gate_status,
        boundary_digest=(
            "- Archived R remains `not-callable-for-parity`; keep legacy alias/source-chain repairs out of the current Trigger 3 compare path while `RBUG-003, RBUG-004, RBUG-014, RBUG-005, RBUG-013` stay open.",
            "- Current Trigger 3 progress is only the paper-backed object compare surface: `evaluation_grid`, `bar_f_at_z0`, `sigma_z_hat`, `covariance_at_grid`, `uniform_critical_value`, `n_boot`, `random_state`.",
            "- `uniform_band_bounds` stays out of the compare set until repaired paths satisfy `repair-obligations-open` and preserve the covariance-process contract.",
        ),
    )
