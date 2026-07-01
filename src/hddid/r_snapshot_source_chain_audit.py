from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_R_OUTER_PATH = Path("hddid-r/R/highdimdiffindiff_crossfit.R")
_R_INNER_PATH = Path("hddid-r/R/highdimdiffindiff_crossfit_inside.R")
_OUTER_SOURCE_LITERAL = 'source("highdimdiffindiff_crossfit_inside3.r")'
_INNER_SOURCE_LITERAL = 'source("Sieve_Functional_Space_Basis.r")'


@dataclass(frozen=True)
class RSnapshotSourceChainAudit:
    repo_root: str
    outer_source_file: str
    inner_source_file: str
    finding_codes: tuple[str, ...]
    status: str
    outer_source_line: int
    outer_source_literal: str
    outer_source_target: str
    outer_source_target_exists: bool
    outer_expected_existing_target: str
    inner_source_line: int
    inner_source_literal: str
    inner_source_target: str
    inner_source_target_exists: bool
    missing_source_targets: tuple[str, ...]
    blocks_runnable_parity_oracle: bool
    recommended_parity_strategy: str


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _read_lines(source_file: Path) -> list[str]:
    if not source_file.is_file():
        raise FileNotFoundError(f"Expected source file is missing: {source_file}")
    return source_file.read_text(encoding="utf-8").splitlines()


def _find_first_line(lines: list[str], needle: str) -> int:
    for line_no, line in enumerate(lines, start=1):
        if needle in line:
            return line_no
    raise ValueError(f"Expected to find {needle!r}")


def _extract_source_target(source_literal: str) -> str:
    prefix = 'source("'
    return source_literal.split(prefix, maxsplit=1)[1].split('"', maxsplit=1)[0]


def audit_r_snapshot_source_chain(
    repo_root: str | Path,
) -> RSnapshotSourceChainAudit:
    root = _coerce_repo_root(repo_root)
    outer_source_file = root / _R_OUTER_PATH
    inner_source_file = root / _R_INNER_PATH

    outer_lines = _read_lines(outer_source_file)
    inner_lines = _read_lines(inner_source_file)

    outer_source_line = _find_first_line(outer_lines, _OUTER_SOURCE_LITERAL)
    inner_source_line = _find_first_line(inner_lines, _INNER_SOURCE_LITERAL)

    outer_source_target = _extract_source_target(_OUTER_SOURCE_LITERAL)
    inner_source_target = _extract_source_target(_INNER_SOURCE_LITERAL)

    outer_target_path = outer_source_file.parent / outer_source_target
    inner_target_path = inner_source_file.parent / inner_source_target
    expected_existing_target = inner_source_file.name

    missing_targets = tuple(
        target
        for target, exists in (
            (outer_source_target, outer_target_path.is_file()),
            (inner_source_target, inner_target_path.is_file()),
        )
        if not exists
    )

    return RSnapshotSourceChainAudit(
        repo_root=root.as_posix(),
        outer_source_file=outer_source_file.as_posix(),
        inner_source_file=inner_source_file.as_posix(),
        finding_codes=("RBUG-003", "RBUG-004"),
        status="broken-source-chain",
        outer_source_line=outer_source_line,
        outer_source_literal=_OUTER_SOURCE_LITERAL,
        outer_source_target=outer_source_target,
        outer_source_target_exists=outer_target_path.is_file(),
        outer_expected_existing_target=expected_existing_target,
        inner_source_line=inner_source_line,
        inner_source_literal=_INNER_SOURCE_LITERAL,
        inner_source_target=inner_source_target,
        inner_source_target_exists=inner_target_path.is_file(),
        missing_source_targets=missing_targets,
        blocks_runnable_parity_oracle=bool(missing_targets),
        recommended_parity_strategy="source-backed wrappers only",
    )


def build_r_snapshot_source_chain_report(audit: RSnapshotSourceChainAudit) -> str:
    lines = [
        "# R snapshot source-chain audit",
        "",
        f"- Outer source file: `{audit.outer_source_file}`",
        f"- Inner source file: `{audit.inner_source_file}`",
        f"- Finding codes: `{', '.join(audit.finding_codes)}`",
        f"- Status: `{audit.status}`",
        "",
        "## Findings",
        "",
        (
            f"- `RBUG-003`: the outer wrapper still sources "
            f"`{audit.outer_source_target}` on line {audit.outer_source_line}, but the "
            f"only matching inner file present in the snapshot is "
            f"`{audit.outer_expected_existing_target}`."
        ),
        (
            f"- `RBUG-004`: the inner routine still sources "
            f"`{audit.inner_source_target}` on line {audit.inner_source_line}, but that "
            "helper file is absent from the repository."
        ),
        "",
        "## Python implication",
        "",
        (
            "- The archived outer R path is still not a runnable parity oracle; use "
            f"`{audit.recommended_parity_strategy}` until the source chain is repaired or "
            "replaced with paper-backed objects."
        ),
    ]
    return "\n".join(lines)
