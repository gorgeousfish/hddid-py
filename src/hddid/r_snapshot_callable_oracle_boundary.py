from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hddid.r_snapshot_audit import (
    audit_r_snapshot_example_interface_source,
    audit_r_snapshot_outer_inference_source,
)
from hddid.r_snapshot_source_chain_audit import audit_r_snapshot_source_chain


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


@dataclass(frozen=True)
class Phase7RSnapshotCallableOracleBoundaryReport:
    stage_label: str
    status: str
    source_chain_status: str
    source_chain_finding_codes: tuple[str, ...]
    example_interface_finding_codes: tuple[str, ...]
    outer_inference_status: str
    outer_inference_blocking_finding_codes: tuple[str, ...]
    callable_parity_oracle_exists: bool
    recommended_routes: tuple[str, str]
    boundary_digest: tuple[str, ...]


def build_phase7_r_snapshot_callable_oracle_boundary_report(
    report: Phase7RSnapshotCallableOracleBoundaryReport,
) -> str:
    lines = [
        "# Phase 7 R snapshot callable oracle boundary",
        "",
        f"- Stage label: `{report.stage_label}`",
        f"- Status: `{report.status}`",
        f"- Source-chain status: `{report.source_chain_status}`",
        (f"- Source-chain findings: `{', '.join(report.source_chain_finding_codes)}`"),
        (
            "- Example-interface findings: "
            f"`{', '.join(report.example_interface_finding_codes)}`"
        ),
        (
            "- Outer-inference blockers: "
            f"`{', '.join(report.outer_inference_blocking_finding_codes)}`"
        ),
        (
            "- Recommended routes: "
            f"`{report.recommended_routes[0]}` and `{report.recommended_routes[1]}`"
        ),
        "",
        "## Boundary digest",
        "",
    ]
    lines.extend(report.boundary_digest)
    return "\n".join(lines)


def run_phase7_r_snapshot_callable_oracle_boundary(
    repo_root: str | Path,
) -> Phase7RSnapshotCallableOracleBoundaryReport:
    root = _coerce_repo_root(repo_root)
    source_chain_audit = audit_r_snapshot_source_chain(root)
    example_interface_audit = audit_r_snapshot_example_interface_source(root)
    outer_inference_audit = audit_r_snapshot_outer_inference_source(root)

    return Phase7RSnapshotCallableOracleBoundaryReport(
        stage_label="phase7-r-snapshot-callable-oracle-boundary",
        status="not-callable-for-parity",
        source_chain_status=source_chain_audit.status,
        source_chain_finding_codes=source_chain_audit.finding_codes,
        example_interface_finding_codes=example_interface_audit.finding_codes,
        outer_inference_status=outer_inference_audit.status,
        outer_inference_blocking_finding_codes=(
            outer_inference_audit.blocking_finding_codes
        ),
        callable_parity_oracle_exists=False,
        recommended_routes=(
            "source-backed wrappers",
            "paper-backed replacement objects",
        ),
        boundary_digest=(
            "- Archived outer wrapper is still blocked by the broken source chain `RBUG-003, RBUG-004`.",
            "- Archived Example is still not callable after legacy-name aliasing because `RBUG-014` leaves `method` drift and missing `z0/alp`.",
            "- Archived outer inference stays `reference-only` until `RBUG-005, RBUG-013` are removed or replaced with paper-backed objects.",
        ),
    )
