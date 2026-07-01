from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .r_snapshot_audit import audit_r_snapshot_example_interface_source


_R_NAMESPACE_PATH = Path("hddid-r/NAMESPACE")
_R_RD_PATH = Path("hddid-r/man/highdimdiffindiff_crossfit.Rd")


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _parse_usage_name_and_arguments(rd_text: str) -> tuple[str, tuple[str, ...]]:
    match = re.search(
        r"\\usage\{\s*([A-Za-z0-9_]+)\((.*?)\)\s*\}",
        rd_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("expected to find a callable usage block in the Rd snapshot")
    return match.group(1), _parse_argument_names(match.group(2))


def _parse_argument_names(payload: str) -> tuple[str, ...]:
    names: list[str] = []
    for raw_field in payload.split(","):
        field = raw_field.strip()
        if not field:
            continue
        names.append(field.split("=", 1)[0].strip())
    return tuple(names)


def _parse_example_call_arguments(example_source: str) -> tuple[str, ...]:
    match = re.search(
        r"highdimdiffindiff_crossfit\((.*?)\)",
        example_source,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("expected archived Example to call highdimdiffindiff_crossfit")
    return _parse_argument_names(match.group(1))


@dataclass(frozen=True)
class Phase7RSnapshotPublicInterfaceSurfaceAudit:
    stage_label: str
    status: str
    finding_codes: tuple[str, ...]
    namespace_export_name: str
    rd_usage_name: str
    source_callable_name: str
    namespace_exports_legacy_name: bool
    rd_usage_uses_legacy_name: bool
    example_call_uses_legacy_name: bool
    rd_usage_arguments: tuple[str, ...]
    source_formals: tuple[str, ...]
    example_call_arguments: tuple[str, ...]
    doc_only_arguments: tuple[str, ...]
    source_only_arguments: tuple[str, ...]
    recommended_routes: tuple[str, str]
    canonical_surface_digest: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "status": self.status,
            "finding_codes": list(self.finding_codes),
            "namespace_export_name": self.namespace_export_name,
            "rd_usage_name": self.rd_usage_name,
            "source_callable_name": self.source_callable_name,
            "namespace_exports_legacy_name": self.namespace_exports_legacy_name,
            "rd_usage_uses_legacy_name": self.rd_usage_uses_legacy_name,
            "example_call_uses_legacy_name": self.example_call_uses_legacy_name,
            "rd_usage_arguments": list(self.rd_usage_arguments),
            "source_formals": list(self.source_formals),
            "example_call_arguments": list(self.example_call_arguments),
            "doc_only_arguments": list(self.doc_only_arguments),
            "source_only_arguments": list(self.source_only_arguments),
            "recommended_routes": list(self.recommended_routes),
            "canonical_surface_digest": list(self.canonical_surface_digest),
        }


def build_phase7_r_snapshot_public_interface_surface_audit_report(
    audit: Phase7RSnapshotPublicInterfaceSurfaceAudit,
) -> str:
    lines = [
        "# Phase 7 R snapshot public interface surface audit",
        "",
        f"- Stage label: `{audit.stage_label}`",
        f"- Status: `{audit.status}`",
        f"- Finding codes: `{', '.join(audit.finding_codes)}`",
        f"- NAMESPACE export: `{audit.namespace_export_name}`",
        f"- Rd usage name: `{audit.rd_usage_name}`",
        f"- Source callable name: `{audit.source_callable_name}`",
        "",
        "## Canonical digest",
        "",
    ]
    lines.extend(audit.canonical_surface_digest)
    return "\n".join(lines)


def audit_r_snapshot_public_interface_surface(
    repo_root: str | Path,
) -> Phase7RSnapshotPublicInterfaceSurfaceAudit:
    root = _coerce_repo_root(repo_root)
    namespace_text = (root / _R_NAMESPACE_PATH).read_text(encoding="utf-8")
    rd_text = (root / _R_RD_PATH).read_text(encoding="utf-8")
    example_audit = audit_r_snapshot_example_interface_source(root)
    example_text = Path(example_audit.source_file).read_text(encoding="utf-8")

    namespace_export_name_match = re.search(
        r"export\((highdimdiffindiff_crossfit)\)", namespace_text
    )
    if namespace_export_name_match is None:
        raise ValueError("expected NAMESPACE to export highdimdiffindiff_crossfit")
    namespace_export_name = namespace_export_name_match.group(1)

    rd_usage_name, rd_usage_arguments = _parse_usage_name_and_arguments(rd_text)
    example_call_arguments = _parse_example_call_arguments(example_text)
    source_formals = tuple(example_audit.mainline_formals)
    doc_argument_pool = set(rd_usage_arguments) | set(example_call_arguments)
    source_argument_pool = set(source_formals)

    doc_only_arguments = tuple(
        name
        for name in rd_usage_arguments
        if name in (doc_argument_pool - source_argument_pool)
    )
    source_only_arguments = tuple(
        name
        for name in source_formals
        if name in (source_argument_pool - doc_argument_pool)
    )

    return Phase7RSnapshotPublicInterfaceSurfaceAudit(
        stage_label="phase7-r-snapshot-public-interface-surface-audit",
        status="archived-doc-surface-drift",
        finding_codes=("RBUG-001", "RBUG-002", "RBUG-014"),
        namespace_export_name=namespace_export_name,
        rd_usage_name=rd_usage_name,
        source_callable_name=example_audit.mainline_function_name,
        namespace_exports_legacy_name=namespace_export_name
        != example_audit.mainline_function_name,
        rd_usage_uses_legacy_name=rd_usage_name != example_audit.mainline_function_name,
        example_call_uses_legacy_name=(
            example_audit.example_call_target != example_audit.mainline_function_name
        ),
        rd_usage_arguments=rd_usage_arguments,
        source_formals=source_formals,
        example_call_arguments=example_call_arguments,
        doc_only_arguments=doc_only_arguments,
        source_only_arguments=source_only_arguments,
        recommended_routes=(
            "paper-backed contract",
            "source-backed wrappers only",
        ),
        canonical_surface_digest=(
            "- NAMESPACE and `.Rd` still expose the legacy callable name `highdimdiffindiff_crossfit`, while the archived source snapshot only defines `highdimdiffindiff_crossfit3`.",
            "- the public doc surface still advertises `method` but omits required runtime inputs `z0` and `alp`, so the documented usage cannot serve as a callable parity contract.",
            "- the archived Example repeats that same legacy/doc-only surface, so Trigger 3 must keep treating it as bug evidence and route parity back to paper-backed contracts plus source-backed wrappers only.",
        ),
    )
