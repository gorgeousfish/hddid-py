from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
_PAPER_RELATIVE_PATH = (
    Path("arXiv-2009.03151v1") / "draft_master_test_August.tex"
)


def _coerce_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is None:
        return _DEFAULT_REPO_ROOT
    return Path(repo_root).expanduser().resolve()


def _read_paper_lines(repo_root: Path) -> list[str]:
    source_path = repo_root / _PAPER_RELATIVE_PATH
    if not source_path.is_file():
        raise FileNotFoundError(
            "required Section 6 method-paper source not found: "
            f"{_PAPER_RELATIVE_PATH.as_posix()}"
        )
    return source_path.read_text(encoding="utf-8").splitlines()


def _find_unique_line(lines: list[str], needle: str) -> int:
    matches = [
        line_number for line_number, line in enumerate(lines, start=1) if needle in line
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one line matching {needle!r}, found {len(matches)}"
        )
    return matches[0]


@dataclass(slots=True)
class Section6PaperSchemaAudit:
    paper_path: str
    empirical_time_window: str
    treated_state_count: int
    excluded_states: tuple[str, ...]
    z_lanes: tuple[str, ...]
    linear_covariate_count: int
    county_characteristics_count: int
    empirical_basis_family: str
    empirical_basis_degree: int
    empirical_confidence_level: float
    simulation_basis_family: str
    simulation_basis_degree: int
    simulation_nominal_coverage: float
    anchor_lines: dict[str, int]
    canonical_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.paper_path = str(self.paper_path).strip()
        self.empirical_time_window = str(self.empirical_time_window).strip()
        self.treated_state_count = int(self.treated_state_count)
        self.excluded_states = tuple(
            str(state).strip() for state in self.excluded_states
        )
        self.z_lanes = tuple(str(lane).strip() for lane in self.z_lanes)
        self.linear_covariate_count = int(self.linear_covariate_count)
        self.county_characteristics_count = int(self.county_characteristics_count)
        self.empirical_basis_family = str(self.empirical_basis_family).strip()
        self.empirical_basis_degree = int(self.empirical_basis_degree)
        self.empirical_confidence_level = float(self.empirical_confidence_level)
        self.simulation_basis_family = str(self.simulation_basis_family).strip()
        self.simulation_basis_degree = int(self.simulation_basis_degree)
        self.simulation_nominal_coverage = float(self.simulation_nominal_coverage)
        self.anchor_lines = {
            str(key).strip(): int(value) for key, value in self.anchor_lines.items()
        }
        self.canonical_digest = tuple(
            str(line).rstrip() for line in self.canonical_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "paper_path": self.paper_path,
            "empirical_time_window": self.empirical_time_window,
            "treated_state_count": self.treated_state_count,
            "excluded_states": list(self.excluded_states),
            "z_lanes": list(self.z_lanes),
            "linear_covariate_count": self.linear_covariate_count,
            "county_characteristics_count": self.county_characteristics_count,
            "empirical_basis_family": self.empirical_basis_family,
            "empirical_basis_degree": self.empirical_basis_degree,
            "empirical_confidence_level": self.empirical_confidence_level,
            "simulation_basis_family": self.simulation_basis_family,
            "simulation_basis_degree": self.simulation_basis_degree,
            "simulation_nominal_coverage": self.simulation_nominal_coverage,
            "anchor_lines": dict(self.anchor_lines),
            "canonical_digest": list(self.canonical_digest),
        }


def build_phase7_section6_paper_schema_audit(
    repo_root: str | Path | None = None,
) -> Section6PaperSchemaAudit:
    root = _coerce_repo_root(repo_root)
    lines = _read_paper_lines(root)
    anchor_lines = {
        "simulation_basis": _find_unique_line(
            lines,
            "In both setting, we use 8th degree trigonometric polynomial basis",
        ),
        "simulation_coverage": _find_unique_line(
            lines,
            r"average coverages for a $90\%$ confidence intervals",
        ),
        "empirical_application": _find_unique_line(
            lines,
            "contains the county level unemployment rates from 2005 to 2007",
        ),
        "empirical_excluded_states": _find_unique_line(
            lines,
            "New Hampshire and Pennsylvania are dropped",
        ),
        "empirical_figure2_contract": _find_unique_line(
            lines,
            "Both methods use 4th degree trigonometric polynomial basis to approximate",
        ),
    }
    canonical_digest = (
        "- empirical Section 6 stays on `2005-2007`, `11` treated states, and explicit `New Hampshire` / `Pennsylvania` exclusion",
        "- empirical Figure 2 keeps `median income` / `population` as the two nonparametric lanes and reports a `703`-covariate linear block tied to `38` county characteristics; `703 = choose(38, 2)` matches pairwise interactions, not 38 main effects plus pairwise interactions",
        "- empirical Section 6 uses `4th degree trigonometric polynomial basis` with `95% confidence intervals`",
        "- Monte Carlo uses `8th degree trigonometric polynomial basis` with `90% confidence intervals`, so future empirical loader / manifest work must not inherit simulation settings by accident",
    )
    return Section6PaperSchemaAudit(
        paper_path=(root / _PAPER_RELATIVE_PATH).as_posix(),
        empirical_time_window="2005-2007",
        treated_state_count=11,
        excluded_states=("New Hampshire", "Pennsylvania"),
        z_lanes=("median income", "population"),
        linear_covariate_count=703,
        county_characteristics_count=38,
        empirical_basis_family="trigonometric",
        empirical_basis_degree=4,
        empirical_confidence_level=0.95,
        simulation_basis_family="trigonometric",
        simulation_basis_degree=8,
        simulation_nominal_coverage=0.90,
        anchor_lines=anchor_lines,
        canonical_digest=canonical_digest,
    )


def run_phase7_section6_paper_schema_audit(
    repo_root: str | Path | None = None,
) -> Section6PaperSchemaAudit:
    return build_phase7_section6_paper_schema_audit(repo_root)


def build_phase7_section6_paper_schema_report(
    audit: Section6PaperSchemaAudit,
) -> str:
    return "\n".join(
        (
            "# Phase 7 Section 6 paper schema source audit",
            "",
            "This helper pins the paper-backed Section 6 contract that future empirical",
            "loader / manifest work must obey.",
            "",
            f"- Paper path: `{audit.paper_path}`",
            f"- Empirical application line {audit.anchor_lines['empirical_application']} fixes the `2005-2007` window and `11 treated states`.",
            f"- Exclusion footnote line {audit.anchor_lines['empirical_excluded_states']} pins `New Hampshire` and `Pennsylvania`.",
            f"- Figure 2 line {audit.anchor_lines['empirical_figure2_contract']} pins `median income` / `population`, a reported `703`-covariate linear block tied to `38 county characteristics`, `4th degree trigonometric polynomial basis`, and `95% confidence intervals`. The count arithmetic `703 = choose(38, 2)` matches pairwise interactions, not 38 main effects plus pairwise interactions.",
            f"- Monte Carlo line {audit.anchor_lines['simulation_basis']} and line {audit.anchor_lines['simulation_coverage']} keep `8th degree trigonometric polynomial basis` with `90% confidence intervals` on the simulation side only.",
            "",
            "Implication: future empirical loader / manifest work must not inherit simulation settings by accident.",
        )
    )
