from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


_R_INNER_PATH = Path("hddid-r/R/highdimdiffindiff_crossfit_inside.R")
_DOCUMENTED_REFERENCE_URL = (
    "https://mirror.las.iastate.edu/CRAN/web/packages/glmnet/refman/glmnet.html"
)


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


@dataclass(frozen=True)
class Phase7RSnapshotInnerCVArgumentAudit:
    stage_label: str
    status: str
    finding_codes: Tuple[str, ...]
    documented_argument_name: str
    undocumented_argument_name: str
    documented_reference_url: str
    canonical_call_line: int
    canonical_call_snippet: str
    drifted_call_line_numbers: Tuple[int, ...]
    drifted_call_snippets: Tuple[str, ...]
    recommended_route: str
    canonical_surface_digest: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "status": self.status,
            "finding_codes": list(self.finding_codes),
            "documented_argument_name": self.documented_argument_name,
            "undocumented_argument_name": self.undocumented_argument_name,
            "documented_reference_url": self.documented_reference_url,
            "canonical_call_line": self.canonical_call_line,
            "canonical_call_snippet": self.canonical_call_snippet,
            "drifted_call_line_numbers": list(self.drifted_call_line_numbers),
            "drifted_call_snippets": list(self.drifted_call_snippets),
            "recommended_route": self.recommended_route,
            "canonical_surface_digest": list(self.canonical_surface_digest),
        }


def build_phase7_r_snapshot_inner_cv_argument_audit_report(
    audit: Phase7RSnapshotInnerCVArgumentAudit,
) -> str:
    lines = [
        "# Phase 7 R snapshot inner CV argument audit",
        "",
        f"- Stage label: `{audit.stage_label}`",
        f"- Status: `{audit.status}`",
        f"- Finding codes: `{', '.join(audit.finding_codes)}`",
        f"- Documented argument name: `{audit.documented_argument_name}`",
        f"- Drifted archived argument name: `{audit.undocumented_argument_name}`",
        f"- Canonical in-file control: line `{audit.canonical_call_line}`",
        (
            "- Drifted line numbers: `"
            + ", ".join(str(line) for line in audit.drifted_call_line_numbers)
            + "`"
        ),
        f"- Official reference: `{audit.documented_reference_url}`",
        "",
        "## Canonical digest",
        "",
    ]
    lines.extend(audit.canonical_surface_digest)
    return "\n".join(lines)


def audit_r_snapshot_inner_cv_argument_surface(
    repo_root: str | Path,
) -> Phase7RSnapshotInnerCVArgumentAudit:
    root = _coerce_repo_root(repo_root)
    lines = (root / _R_INNER_PATH).read_text(encoding="utf-8").splitlines()

    canonical_line_number = 0
    canonical_line = ""
    drifted_line_numbers = []
    drifted_lines = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if "cv.glmnet" not in line:
            continue
        if "nfolds" in line:
            canonical_line_number = line_number
            canonical_line = line
            continue
        if "nfold=" in line or "nfold =" in line:
            drifted_line_numbers.append(line_number)
            drifted_lines.append(line)

    if canonical_line_number == 0:
        raise ValueError("expected one archived inner cv.glmnet call with `nfolds`")
    if not drifted_line_numbers:
        raise ValueError("expected archived inner snapshot to contain `nfold` drift")

    return Phase7RSnapshotInnerCVArgumentAudit(
        stage_label="phase7-r-snapshot-inner-cv-argument-audit",
        status="archived-inner-cv-argument-drift",
        finding_codes=("RBUG-015",),
        documented_argument_name="nfolds",
        undocumented_argument_name="nfold",
        documented_reference_url=_DOCUMENTED_REFERENCE_URL,
        canonical_call_line=canonical_line_number,
        canonical_call_snippet=canonical_line,
        drifted_call_line_numbers=tuple(drifted_line_numbers),
        drifted_call_snippets=tuple(drifted_lines),
        recommended_route="source-level bug evidence only",
        canonical_surface_digest=(
            "- the archived inner snapshot calls `cv.glmnet(...)` with `nfold` on the outcome and second-stage tuning paths, while the documented control in the CRAN interface is `nfolds`.",
            "- because this drift sits inside the already broken archived source chain, Trigger 3 should keep treating the affected tuning path as source-level bug evidence rather than a callable parity helper.",
            "- Python parity and implementation work should continue to compare paper-backed objects or explicitly fixed tuning inputs, not inherit the archived `nfold` surface.",
        ),
    )
