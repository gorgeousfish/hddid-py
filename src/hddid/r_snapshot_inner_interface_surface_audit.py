from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import re


_R_INNER_SOURCE_PATH = Path("hddid-r/R/highdimdiffindiff_crossfit_inside.R")
_R_INNER_RD_PATH = Path("hddid-r/man/highdimdiffindiff_crossfit_inside.Rd")
_INNER_USAGE_NAME = "highdimdiffindiff_crossfit_inside"
_INNER_SOURCE_NAME = "highdimdiffindiff_crossfit_inside3"


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _parse_argument_names(payload: str) -> tuple[str, ...]:
    names: list[str] = []
    for raw_field in payload.split(","):
        field = raw_field.strip()
        if not field:
            continue
        names.append(field.split("=", 1)[0].strip())
    return tuple(names)


@dataclass(frozen=True)
class Phase7RSnapshotInnerInterfaceSurfaceAudit:
    stage_label: str
    status: str
    finding_codes: Tuple[str, ...]
    rd_usage_name: str
    source_callable_name: str
    rd_usage_line: int
    source_signature_line: int
    rd_usage_arguments: Tuple[str, ...]
    source_formals: Tuple[str, ...]
    doc_only_arguments: Tuple[str, ...]
    source_only_arguments: Tuple[str, ...]
    recommended_route: str
    canonical_surface_digest: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "status": self.status,
            "finding_codes": list(self.finding_codes),
            "rd_usage_name": self.rd_usage_name,
            "source_callable_name": self.source_callable_name,
            "rd_usage_line": self.rd_usage_line,
            "source_signature_line": self.source_signature_line,
            "rd_usage_arguments": list(self.rd_usage_arguments),
            "source_formals": list(self.source_formals),
            "doc_only_arguments": list(self.doc_only_arguments),
            "source_only_arguments": list(self.source_only_arguments),
            "recommended_route": self.recommended_route,
            "canonical_surface_digest": list(self.canonical_surface_digest),
        }


def build_phase7_r_snapshot_inner_interface_surface_audit_report(
    audit: Phase7RSnapshotInnerInterfaceSurfaceAudit,
) -> str:
    lines = [
        "# Phase 7 R snapshot inner interface surface audit",
        "",
        f"- Stage label: `{audit.stage_label}`",
        f"- Status: `{audit.status}`",
        f"- Finding codes: `{', '.join(audit.finding_codes)}`",
        f"- `.Rd` usage name: `{audit.rd_usage_name}`",
        f"- Source callable name: `{audit.source_callable_name}`",
        f"- `.Rd` usage line: `{audit.rd_usage_line}`",
        f"- Source signature line: `{audit.source_signature_line}`",
        "",
        "## Canonical digest",
        "",
    ]
    lines.extend(audit.canonical_surface_digest)
    return "\n".join(lines)


def audit_r_snapshot_inner_interface_surface(
    repo_root: str | Path,
) -> Phase7RSnapshotInnerInterfaceSurfaceAudit:
    root = _coerce_repo_root(repo_root)
    source_lines = (
        (root / _R_INNER_SOURCE_PATH).read_text(encoding="utf-8").splitlines()
    )
    rd_lines = (root / _R_INNER_RD_PATH).read_text(encoding="utf-8").splitlines()
    rd_text = "\n".join(rd_lines)

    source_signature_line = next(
        line_no
        for line_no, line in enumerate(source_lines, start=1)
        if _INNER_SOURCE_NAME in line and "function(" in line
    )
    source_signature = source_lines[source_signature_line - 1].strip()
    source_formals = _parse_argument_names(
        source_signature.split("function(", 1)[1].rsplit(")", 1)[0]
    )

    rd_usage_line = next(
        line_no
        for line_no, line in enumerate(rd_lines, start=1)
        if line.strip().startswith(f"{_INNER_USAGE_NAME}(")
    )
    usage_match = re.search(
        r"\\usage\{\s*(highdimdiffindiff_crossfit_inside)\((.*?)\)\s*\}",
        rd_text,
        flags=re.DOTALL,
    )
    if usage_match is None:
        raise ValueError("expected archived inner Rd to expose a callable usage block")
    rd_usage_name = usage_match.group(1)
    rd_usage_arguments = _parse_argument_names(usage_match.group(2))

    doc_only_arguments = tuple(
        arg for arg in rd_usage_arguments if arg not in set(source_formals)
    )
    source_only_arguments = tuple(
        arg for arg in source_formals if arg not in set(rd_usage_arguments)
    )

    return Phase7RSnapshotInnerInterfaceSurfaceAudit(
        stage_label="phase7-r-snapshot-inner-interface-surface-audit",
        status="archived-inner-interface-surface-drift",
        finding_codes=("RBUG-016",),
        rd_usage_name=rd_usage_name,
        source_callable_name=_INNER_SOURCE_NAME,
        rd_usage_line=rd_usage_line,
        source_signature_line=source_signature_line,
        rd_usage_arguments=rd_usage_arguments,
        source_formals=source_formals,
        doc_only_arguments=doc_only_arguments,
        source_only_arguments=source_only_arguments,
        recommended_route="source-level bug evidence only",
        canonical_surface_digest=(
            "- the archived inner `.Rd` usage still exposes the legacy callable `highdimdiffindiff_crossfit_inside(...)` with `method, q`, while the archived source snapshot only defines `highdimdiffindiff_crossfit_inside3(..., q, z0, alp)`.",
            "- the documented inner surface still advertises `method` but omits required runtime inputs `z0` and `alp`, so the archived inner callable contract is not executable even before the broken source chain is considered.",
            "- Trigger 3 should keep routing archived inner surface drift as source-level bug evidence only; parity work must continue to use paper-backed objects or explicit source-backed wrappers instead of inheriting the archived inner `.Rd` surface.",
        ),
    )
