from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_R_EXAMPLE_PATH = Path("hddid-r/R/Examplehighdimdiffindiff.R")
_R_OUTER_PATH = Path("hddid-r/R/highdimdiffindiff_crossfit.R")
_R_INNER_PATH = Path("hddid-r/R/highdimdiffindiff_crossfit_inside.R")
_PAPER_PATH = Path("paper/hddid_paper.md")
_EXAMPLE_METHOD_PARAM_LITERAL = "@param method basis function"
_EXAMPLE_SIGNATURE_LITERAL = "Examplehighdimdiffindiff <- function"
_OUTER_SIGNATURE_LITERAL = "highdimdiffindiff_crossfit3 <- function"
_INNER_POL_LITERAL = "sieve.Pol("
_INNER_TRIPOL_LITERAL = "sieve.TriPol("
_PAPER_TRIG_LITERAL = "trigonometric polynomial basis"
_PAPER_MONTE_CARLO_LITERAL = "8th degree trigonometric polynomial basis"
_PAPER_EMPIRICAL_LITERAL = "4th degree trigonometric polynomial basis"


@dataclass(frozen=True)
class RSnapshotBasisMethodSourceAudit:
    repo_root: str
    example_source_file: str
    outer_source_file: str
    inner_source_file: str
    paper_source_file: str
    finding_codes: tuple[str, ...]
    status: str
    example_method_documentation_present: bool
    example_signature_has_method: bool
    example_method_default: str | None
    outer_signature_has_method: bool
    inner_polynomial_call_lines: tuple[int, ...]
    inner_tripol_active_call_lines: tuple[int, ...]
    inner_tripol_comment_lines: tuple[int, ...]
    paper_monte_carlo_basis_degree: int
    paper_empirical_basis_degree: int
    paper_trigonometric_anchor_lines: tuple[int, ...]
    parity_lane_split: tuple[str, str]


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


def _find_all_lines(lines: list[str], needle: str) -> tuple[int, ...]:
    return tuple(
        line_no for line_no, line in enumerate(lines, start=1) if needle in line
    )


def _extract_example_method_default(signature_line: str) -> str | None:
    marker = 'method = "'
    if marker not in signature_line:
        return None
    return signature_line.split(marker, maxsplit=1)[1].split('"', maxsplit=1)[0]


def audit_r_snapshot_basis_method_source(
    repo_root: str | Path,
) -> RSnapshotBasisMethodSourceAudit:
    root = _coerce_repo_root(repo_root)
    example_source_file = root / _R_EXAMPLE_PATH
    outer_source_file = root / _R_OUTER_PATH
    inner_source_file = root / _R_INNER_PATH
    paper_source_file = root / _PAPER_PATH

    example_lines = _read_lines(example_source_file)
    outer_lines = _read_lines(outer_source_file)
    inner_lines = _read_lines(inner_source_file)
    paper_lines = _read_lines(paper_source_file)

    example_method_doc_line = _find_first_line(
        example_lines, _EXAMPLE_METHOD_PARAM_LITERAL
    )
    example_signature_line = _find_first_line(example_lines, _EXAMPLE_SIGNATURE_LITERAL)
    outer_signature_line = _find_first_line(outer_lines, _OUTER_SIGNATURE_LITERAL)
    inner_polynomial_call_lines = _find_all_lines(inner_lines, _INNER_POL_LITERAL)
    inner_tripol_comment_lines = tuple(
        line_no
        for line_no in inner_polynomial_call_lines
        if _INNER_TRIPOL_LITERAL in inner_lines[line_no - 1]
    )
    inner_tripol_active_call_lines = tuple(
        line_no
        for line_no, line in enumerate(inner_lines, start=1)
        if _INNER_TRIPOL_LITERAL in line.split("#", maxsplit=1)[0]
    )
    paper_trigonometric_anchor_lines = _find_all_lines(paper_lines, _PAPER_TRIG_LITERAL)
    paper_monte_carlo_line = _find_first_line(paper_lines, _PAPER_MONTE_CARLO_LITERAL)
    paper_empirical_line = _find_first_line(paper_lines, _PAPER_EMPIRICAL_LITERAL)

    example_signature_text = example_lines[example_signature_line - 1]
    outer_signature_text = outer_lines[outer_signature_line - 1]

    finding_codes = ("RBUG-006", "RBUG-008")

    return RSnapshotBasisMethodSourceAudit(
        repo_root=root.as_posix(),
        example_source_file=example_source_file.as_posix(),
        outer_source_file=outer_source_file.as_posix(),
        inner_source_file=inner_source_file.as_posix(),
        paper_source_file=paper_source_file.as_posix(),
        finding_codes=finding_codes,
        status="paper-first split",
        example_method_documentation_present=example_method_doc_line > 0,
        example_signature_has_method="method" in example_signature_text,
        example_method_default=_extract_example_method_default(example_signature_text),
        outer_signature_has_method="method" in outer_signature_text,
        inner_polynomial_call_lines=inner_polynomial_call_lines,
        inner_tripol_active_call_lines=inner_tripol_active_call_lines,
        inner_tripol_comment_lines=inner_tripol_comment_lines,
        paper_monte_carlo_basis_degree=8 if paper_monte_carlo_line > 0 else 0,
        paper_empirical_basis_degree=4 if paper_empirical_line > 0 else 0,
        paper_trigonometric_anchor_lines=paper_trigonometric_anchor_lines,
        parity_lane_split=("paper-trigonometric", "r-parity-polynomial"),
    )


def build_r_snapshot_basis_method_source_report(
    audit: RSnapshotBasisMethodSourceAudit,
) -> str:
    lines = [
        "# R snapshot basis/method source audit",
        "",
        f"- Example source file: `{audit.example_source_file}`",
        f"- Outer source file: `{audit.outer_source_file}`",
        f"- Inner source file: `{audit.inner_source_file}`",
        f"- Paper source file: `{audit.paper_source_file}`",
        f"- Finding codes: `{', '.join(audit.finding_codes)}`",
        f"- Status: `{audit.status}`",
        "",
        "## Findings",
        "",
        (
            f'- `RBUG-006`: the archived example documents `method = "Pol" / "Tri"` '
            'and keeps `method = "Pol"` in the signature, but the estimator mainline '
            "never accepts a `method` argument."
        ),
        (
            f"- `RBUG-008`: the inner estimator still hard-codes `sieve.Pol(...)` on lines "
            f"{audit.inner_polynomial_call_lines} and leaves `sieve.TriPol(...)` only as "
            "commented alternatives, while the paper fixes the Monte Carlo lane at "
            "`8th degree trigonometric polynomial basis` and the empirical lane at "
            "`4th degree trigonometric polynomial basis`."
        ),
        "",
        "## Python implication",
        "",
        (
            f"- Keep the parity split explicit: `{audit.parity_lane_split[0]}` for "
            "paper-backed reproduction and `r-parity-polynomial` for source-level R "
            "comparison. Do not treat the archived R mainline as a trigonometric oracle."
        ),
        (
            "- Any future estimator parity asset must verify basis-family alignment before "
            "comparing coefficients, intervals, or Monte Carlo coverage."
        ),
    ]
    return "\n".join(lines)
