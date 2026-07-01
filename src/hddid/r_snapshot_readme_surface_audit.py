from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


_R_README_PATH = Path("hddid-r/README.md")
_LEGACY_STALE_PUBLIC_TARGET = "hddid-stata/"
_LEGACY_STALE_PUBLIC_ROUTE_TERMS = (_LEGACY_STALE_PUBLIC_TARGET, "Stata")
_REPO_LOCAL_PUBLIC_TARGETS = (
    "hddid.validation",
    "hddid-py/README.md",
    "hddid-py/docs/api-reference.md",
)


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


@dataclass(frozen=True)
class Phase7RSnapshotReadmeSurfaceAudit:
    stage_label: str
    status: str
    finding_codes: Tuple[str, ...]
    readme_path: str
    legacy_stale_public_target: str
    legacy_stale_target_lines: Tuple[int, ...]
    legacy_stale_public_target_exists: bool
    repo_local_public_targets: Tuple[str, ...]
    recommended_route: str
    canonical_surface_digest: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "status": self.status,
            "finding_codes": list(self.finding_codes),
            "readme_path": self.readme_path,
            "legacy_stale_public_target": self.legacy_stale_public_target,
            "legacy_stale_target_lines": list(self.legacy_stale_target_lines),
            "legacy_stale_public_target_exists": self.legacy_stale_public_target_exists,
            "repo_local_public_targets": list(self.repo_local_public_targets),
            "recommended_route": self.recommended_route,
            "canonical_surface_digest": list(self.canonical_surface_digest),
        }


def build_phase7_r_snapshot_readme_surface_audit_report(
    audit: Phase7RSnapshotReadmeSurfaceAudit,
) -> str:
    lines = [
        "# Phase 7 R snapshot README surface audit",
        "",
        f"- Stage label: `{audit.stage_label}`",
        f"- Status: `{audit.status}`",
        f"- Finding codes: `{', '.join(audit.finding_codes)}`",
        f"- README path: `{audit.readme_path}`",
        f"- Legacy stale public target: `{audit.legacy_stale_public_target}`",
        f"- Legacy stale target lines: `{audit.legacy_stale_target_lines}`",
        f"- Legacy stale target exists in workspace: `{audit.legacy_stale_public_target_exists}`",
        f"- Repo-local public targets: `{', '.join(audit.repo_local_public_targets)}`",
        "",
        "## Canonical digest",
        "",
    ]
    lines.extend(audit.canonical_surface_digest)
    return "\n".join(lines)


def audit_r_snapshot_readme_surface(
    repo_root: str | Path,
) -> Phase7RSnapshotReadmeSurfaceAudit:
    root = _coerce_repo_root(repo_root)
    readme_lines = (root / _R_README_PATH).read_text(encoding="utf-8").splitlines()
    legacy_stale_target_lines = tuple(
        line_no
        for line_no, line in enumerate(readme_lines, start=1)
        if any(term in line for term in _LEGACY_STALE_PUBLIC_ROUTE_TERMS)
    )
    status = (
        "readme-route-repaired"
        if not legacy_stale_target_lines
        else "archived-readme-routing-drift"
    )
    finding_codes = (
        ("RBUG-007-resolved",)
        if not legacy_stale_target_lines
        else ("RBUG-007",)
    )
    recommended_route = (
        "python-public-surface"
        if not legacy_stale_target_lines
        else "source-level bug evidence only"
    )
    canonical_surface_digest = (
        (
            "- the R README now routes current workspace users to `hddid-py/`, `hddid.validation`, `hddid-py/README.md`, and `hddid-py/docs/api-reference.md` instead of the missing legacy `hddid-stata/` path.",
            "- the repaired README also avoids routing examples or citations through legacy Stata help/workflow prose, so the current public surface remains Python-only.",
            "- the R snapshot remains reference-only: it can inform paper-backed algorithm reading, but it does not define the current repo-facing public contract.",
            "- Trigger 2 / Trigger 3 should treat RBUG-007 as repaired at the README surface while keeping other source-level R snapshot drifts on their own audit helpers.",
        )
        if not legacy_stale_target_lines
        else (
            "- the archived R README still routes current workspace users to `hddid-stata/` as the maintained public implementation even though that path does not exist in this repository.",
            "- the archived R README still contains legacy Stata help/workflow prose, so it can misroute current users even without spelling the stale path literally.",
            "- that README surface therefore cannot define the current repo-facing public contract; future runs must use `hddid.validation`, `hddid-py/README.md`, and `hddid-py/docs/api-reference.md` plus paper-backed routing instead of inheriting the stale Stata handoff.",
            "- Trigger 2 / Trigger 3 should keep the archived README route as source-level bug evidence only, not as a release-facing or parity-facing oracle boundary.",
        )
    )

    return Phase7RSnapshotReadmeSurfaceAudit(
        stage_label="phase7-r-snapshot-readme-surface-audit",
        status=status,
        finding_codes=finding_codes,
        readme_path=str(_R_README_PATH),
        legacy_stale_public_target=_LEGACY_STALE_PUBLIC_TARGET,
        legacy_stale_target_lines=legacy_stale_target_lines,
        legacy_stale_public_target_exists=(
            root / _LEGACY_STALE_PUBLIC_TARGET.rstrip("/")
        ).exists(),
        repo_local_public_targets=_REPO_LOCAL_PUBLIC_TARGETS,
        recommended_route=recommended_route,
        canonical_surface_digest=canonical_surface_digest,
    )
