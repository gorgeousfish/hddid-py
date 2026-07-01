from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_R_EXAMPLE_PATH = Path("hddid-r/R/Examplehighdimdiffindiff.R")
_R_OUTER_INFERENCE_PATH = Path("hddid-r/R/highdimdiffindiff_crossfit.R")
_R_OUTER_INFERENCE_INNER_PATH = Path("hddid-r/R/highdimdiffindiff_crossfit_inside.R")
_RHO_X_PARAM_LINE = "@param rho.X"
_METHOD_PARAM_LINE = "@param method"
_R_EXAMPLE_SIGNATURE = "Examplehighdimdiffindiff <- function"
_R_MAINLINE_SIGNATURE = "highdimdiffindiff_crossfit3 <- function"
_TOEPLITZ_LITERAL = "0.5^(p-toeplitz(p:1))"
_Z_DRAW_LITERAL = "z = rnorm(n0,0, 1)"
_Y0_LITERAL = "y0 = rnorm(n0,0,1)*(1/sqrt(2)*z +1/sqrt(2)*x[,1])"
_PHI1_LITERAL = "Phi1 = (x %*% omega1) +  exp(z)"
_ESTIMATOR_CALL_LITERAL = "highdimdiffindiff_crossfit(y0, y1, treat, x, z"
_CIUNIFORM_LITERAL = "CIuniform = rbind(debias+ff$tc[1]*stdg, gdebias+ff$tc[2]*stdg)"
_OUTER_AGGREGATION_LITERALS = {
    "xdebias": "xdebias = xdebias + ff$n_valid * ff$xdebias",
    "gdebias": "gdebias = gdebias + ff$n_valid * ff$gdebias",
    "stdx": "stdx = stdx + (ff$n_valid * ff$stdx)^2",
    "stdg": "stdg = stdg + (ff$n_valid * ff$stdg)^2",
}
_TC_RETURN_LITERAL = '"tc" = tc'
_BARE_BROWSER_LITERAL = "browser"


@dataclass(frozen=True)
class RSnapshotMonteCarloOracleAudit:
    repo_root: str
    source_file: str
    finding_codes: tuple[str, ...]
    rho_x_parameter_name: str
    rho_x_documentation_present: bool
    rho_x_signature_present: bool
    rho_x_used_in_covariance: bool
    rho_x_dead_parameter: bool
    toeplitz_covariance_literal: str
    initial_z_draw_line: int
    y0_line: int
    redraw_z_line: int
    phi1_line: int
    estimator_call_line: int
    uses_single_observed_z_contract: bool


@dataclass(frozen=True)
class RSnapshotExampleInterfaceSourceAudit:
    repo_root: str
    source_file: str
    mainline_source_file: str
    finding_codes: tuple[str, ...]
    example_function_name: str
    example_call_target: str
    documented_method_parameter_present: bool
    example_signature_includes_method: bool
    example_call_passes_method_argument: bool
    example_call_passes_q_argument: bool
    example_call_passes_k_argument: bool
    example_call_passes_z0_argument: bool
    example_call_passes_alp_argument: bool
    mainline_function_name: str
    mainline_formals: tuple[str, ...]
    mainline_accepts_method_argument: bool
    aliasing_legacy_name_still_leaves_interface_drift: bool


@dataclass(frozen=True)
class RSnapshotOuterInferenceSourceAudit:
    repo_root: str
    outer_source_file: str
    inner_source_file: str
    finding_codes: tuple[str, ...]
    blocking_finding_codes: tuple[str, ...]
    status: str
    outer_aggregated_fold_fields: tuple[str, ...]
    ciuniform_line: int
    tc_return_line: int
    browser_line: int
    ciuniform_uses_undefined_debias: bool
    ciuniform_reuses_last_fold_tc: bool
    browser_token_present: bool
    browser_token_is_call: bool
    browser_token_blocks_runtime: bool
    replacement_target: str


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _read_source_lines(source_file: Path) -> list[str]:
    if not source_file.is_file():
        raise FileNotFoundError(f"R snapshot example is missing: {source_file}")
    return source_file.read_text(encoding="utf-8").splitlines()


def _find_first_line(lines: list[str], needle: str) -> int:
    for line_no, line in enumerate(lines, start=1):
        if needle in line:
            return line_no
    raise ValueError(f"Expected to find {needle!r} in the R snapshot example")


def _find_all_lines(lines: list[str], needle: str) -> tuple[int, ...]:
    matches = tuple(
        line_no for line_no, line in enumerate(lines, start=1) if needle in line
    )
    if not matches:
        raise ValueError(f"Expected to find at least one {needle!r} occurrence")
    return matches


def _find_line_by_predicate(lines: list[str], description: str, predicate) -> int:
    for line_no, line in enumerate(lines, start=1):
        if predicate(line):
            return line_no
    raise ValueError(f"Expected to find {description} in the R snapshot source")


def _parse_formals_from_definition_line(line: str) -> tuple[str, ...]:
    start = line.index("function(") + len("function(")
    end = line.rindex(")")
    payload = line[start:end]
    names: list[str] = []
    for field in payload.split(","):
        token = field.strip()
        if not token:
            continue
        names.append(token.split("=", 1)[0].strip())
    return tuple(names)


def audit_r_snapshot_monte_carlo_oracle(
    repo_root: str | Path,
) -> RSnapshotMonteCarloOracleAudit:
    root = _coerce_repo_root(repo_root)
    source_file = root / _R_EXAMPLE_PATH
    lines = _read_source_lines(source_file)

    documentation_line = _find_first_line(lines, _RHO_X_PARAM_LINE)
    signature_line = _find_first_line(lines, _R_EXAMPLE_SIGNATURE)
    covariance_line = _find_first_line(lines, _TOEPLITZ_LITERAL)
    z_draw_lines = _find_all_lines(lines, _Z_DRAW_LITERAL)
    if len(z_draw_lines) < 2:
        raise ValueError(
            "Expected the archived R helper to contain both the initial and redraw z lines"
        )
    y0_line = _find_first_line(lines, _Y0_LITERAL)
    phi1_line = _find_first_line(lines, _PHI1_LITERAL)
    estimator_call_line = _find_first_line(lines, _ESTIMATOR_CALL_LITERAL)

    rho_x_used_in_covariance = "rho.X" in lines[covariance_line - 1]
    rho_x_dead_parameter = (
        documentation_line < signature_line
        and not rho_x_used_in_covariance
        and _TOEPLITZ_LITERAL in lines[covariance_line - 1]
    )
    uses_single_observed_z_contract = len(z_draw_lines) == 1

    finding_codes: list[str] = []
    if rho_x_dead_parameter:
        finding_codes.append("RBUG-009")
    if z_draw_lines[0] < y0_line < z_draw_lines[1] < phi1_line < estimator_call_line:
        finding_codes.append("RBUG-012")

    return RSnapshotMonteCarloOracleAudit(
        repo_root=root.as_posix(),
        source_file=source_file.as_posix(),
        finding_codes=tuple(finding_codes),
        rho_x_parameter_name="rho.X",
        rho_x_documentation_present=documentation_line > 0,
        rho_x_signature_present="rho.X" in lines[signature_line - 1],
        rho_x_used_in_covariance=rho_x_used_in_covariance,
        rho_x_dead_parameter=rho_x_dead_parameter,
        toeplitz_covariance_literal=_TOEPLITZ_LITERAL,
        initial_z_draw_line=z_draw_lines[0],
        y0_line=y0_line,
        redraw_z_line=z_draw_lines[1],
        phi1_line=phi1_line,
        estimator_call_line=estimator_call_line,
        uses_single_observed_z_contract=uses_single_observed_z_contract,
    )


def audit_r_snapshot_example_interface_source(
    repo_root: str | Path,
) -> RSnapshotExampleInterfaceSourceAudit:
    root = _coerce_repo_root(repo_root)
    source_file = root / _R_EXAMPLE_PATH
    mainline_source_file = root / _R_OUTER_INFERENCE_PATH
    example_lines = _read_source_lines(source_file)
    mainline_lines = _read_source_lines(mainline_source_file)

    method_param_line = _find_first_line(example_lines, _METHOD_PARAM_LINE)
    example_signature_line = _find_first_line(example_lines, _R_EXAMPLE_SIGNATURE)
    example_call_line = _find_first_line(example_lines, _ESTIMATOR_CALL_LITERAL)
    mainline_signature_line = _find_first_line(mainline_lines, _R_MAINLINE_SIGNATURE)

    example_signature = example_lines[example_signature_line - 1]
    example_call = example_lines[example_call_line - 1]
    mainline_signature = mainline_lines[mainline_signature_line - 1]
    mainline_formals = _parse_formals_from_definition_line(mainline_signature)

    example_call_passes_method_argument = (
        "method =method" in example_call or "method = method" in example_call
    )
    example_call_passes_q_argument = "q=q" in example_call or "q = q" in example_call
    example_call_passes_k_argument = "k=3" in example_call or "k = 3" in example_call
    example_call_passes_z0_argument = "z0=" in example_call or "z0 =" in example_call
    example_call_passes_alp_argument = "alp=" in example_call or "alp =" in example_call
    mainline_accepts_method_argument = "method" in mainline_formals

    aliasing_legacy_name_still_leaves_interface_drift = bool(
        example_call_passes_method_argument
        and not mainline_accepts_method_argument
        and (not example_call_passes_z0_argument)
        and (not example_call_passes_alp_argument)
        and "z0" in mainline_formals
        and "alp" in mainline_formals
    )

    finding_codes = (
        ("RBUG-014",) if aliasing_legacy_name_still_leaves_interface_drift else ()
    )

    return RSnapshotExampleInterfaceSourceAudit(
        repo_root=root.as_posix(),
        source_file=source_file.as_posix(),
        mainline_source_file=mainline_source_file.as_posix(),
        finding_codes=finding_codes,
        example_function_name="Examplehighdimdiffindiff",
        example_call_target="highdimdiffindiff_crossfit",
        documented_method_parameter_present=method_param_line > 0,
        example_signature_includes_method="method" in example_signature,
        example_call_passes_method_argument=example_call_passes_method_argument,
        example_call_passes_q_argument=example_call_passes_q_argument,
        example_call_passes_k_argument=example_call_passes_k_argument,
        example_call_passes_z0_argument=example_call_passes_z0_argument,
        example_call_passes_alp_argument=example_call_passes_alp_argument,
        mainline_function_name="highdimdiffindiff_crossfit3",
        mainline_formals=mainline_formals,
        mainline_accepts_method_argument=mainline_accepts_method_argument,
        aliasing_legacy_name_still_leaves_interface_drift=(
            aliasing_legacy_name_still_leaves_interface_drift
        ),
    )


def audit_r_snapshot_outer_inference_source(
    repo_root: str | Path,
) -> RSnapshotOuterInferenceSourceAudit:
    root = _coerce_repo_root(repo_root)
    outer_source_file = root / _R_OUTER_INFERENCE_PATH
    inner_source_file = root / _R_OUTER_INFERENCE_INNER_PATH
    outer_lines = _read_source_lines(outer_source_file)
    inner_lines = _read_source_lines(inner_source_file)

    ciuniform_line = _find_first_line(outer_lines, _CIUNIFORM_LITERAL)
    tc_return_line = _find_first_line(inner_lines, _TC_RETURN_LITERAL)
    browser_line = _find_line_by_predicate(
        inner_lines,
        "a bare browser token",
        lambda line: line.strip() == _BARE_BROWSER_LITERAL,
    )
    browser_token = inner_lines[browser_line - 1].strip()
    browser_token_is_call = browser_token == "browser()"
    browser_token_blocks_runtime = browser_token_is_call

    outer_aggregated_fold_fields = tuple(
        field
        for field, literal in _OUTER_AGGREGATION_LITERALS.items()
        if any(literal in line for line in outer_lines)
    )
    outer_has_tc_aggregation = any(
        "ff$tc" in line and "CIuniform" not in line for line in outer_lines
    ) or any("tc =" in line and "ff$tc" not in line for line in outer_lines)

    ciuniform_line_text = outer_lines[ciuniform_line - 1]
    ciuniform_uses_undefined_debias = "debias+ff$tc[1]*stdg" in ciuniform_line_text
    ciuniform_reuses_last_fold_tc = (
        "ff$tc" in ciuniform_line_text and not outer_has_tc_aggregation
    )

    finding_codes: list[str] = []
    blocking_finding_codes: list[str] = []
    if ciuniform_uses_undefined_debias:
        finding_codes.append("RBUG-005")
        blocking_finding_codes.append("RBUG-005")
    if browser_line > 0:
        finding_codes.append("RBUG-011")
    if ciuniform_reuses_last_fold_tc:
        finding_codes.append("RBUG-013")
        blocking_finding_codes.append("RBUG-013")

    return RSnapshotOuterInferenceSourceAudit(
        repo_root=root.as_posix(),
        outer_source_file=outer_source_file.as_posix(),
        inner_source_file=inner_source_file.as_posix(),
        finding_codes=tuple(finding_codes),
        blocking_finding_codes=tuple(blocking_finding_codes),
        status="reference-only",
        outer_aggregated_fold_fields=outer_aggregated_fold_fields,
        ciuniform_line=ciuniform_line,
        tc_return_line=tc_return_line,
        browser_line=browser_line,
        ciuniform_uses_undefined_debias=ciuniform_uses_undefined_debias,
        ciuniform_reuses_last_fold_tc=ciuniform_reuses_last_fold_tc,
        browser_token_present=browser_line > 0,
        browser_token_is_call=browser_token_is_call,
        browser_token_blocks_runtime=browser_token_blocks_runtime,
        replacement_target="aggregated covariance-process objects",
    )


def build_r_snapshot_example_interface_source_report(
    audit: RSnapshotExampleInterfaceSourceAudit,
) -> str:
    lines = [
        "# R snapshot example interface source audit",
        "",
        f"- Example source file: `{audit.source_file}`",
        f"- Mainline source file: `{audit.mainline_source_file}`",
        f"- Finding codes: `{', '.join(audit.finding_codes)}`",
        "",
        "## Findings",
        "",
        (
            f"- `RBUG-014`: archived `{audit.example_function_name}` still calls "
            f"`{audit.example_call_target}(...)` with `method = method`, `q = q`, and `k = 3`, "
            f"but the current mainline function is `{audit.mainline_function_name}` with formals "
            f"`{', '.join(audit.mainline_formals)}`."
        ),
        (
            "- Fixing only the legacy function name is insufficient: after aliasing "
            "`highdimdiffindiff_crossfit <- highdimdiffindiff_crossfit3`, the Example path still "
            "fails with `unused argument (method = method)` and still omits the required runtime "
            "arguments `z0` and `alp`."
        ),
        "",
        "## Python implication",
        "",
        (
            "- Do not treat `Examplehighdimdiffindiff.R` as a callable oracle for Monte Carlo or "
            "parity handoff. Python should continue to treat the archived helper as bug evidence "
            "and derive callable contracts from the paper plus source-backed mainline objects instead."
        ),
    ]
    return "\n".join(lines)


def build_r_snapshot_monte_carlo_oracle_report(
    audit: RSnapshotMonteCarloOracleAudit,
) -> str:
    lines = [
        "# R snapshot Monte Carlo oracle audit",
        "",
        f"- Source file: `{audit.source_file}`",
        f"- Finding codes: `{', '.join(audit.finding_codes)}`",
        "",
        "## Findings",
        "",
        (
            f"- `RBUG-009`: `rho.X` is documented and exposed in the function signature, "
            f"but the archived helper still hard-codes `{audit.toeplitz_covariance_literal}` "
            "for the Toeplitz covariance. Treat `rho.X` as a dead Monte Carlo parameter."
        ),
        (
            f"- `RBUG-012`: the archived helper does not preserve a single observed z. "
            f"It draws `z` before `y0` on line {audit.initial_z_draw_line}, redraws `z` "
            f"on line {audit.redraw_z_line}, and then uses the second draw for `Phi1` and "
            "the estimator input. This violates the paper DGP2 single observed z contract."
        ),
        "",
        "## Python implication",
        "",
        (
            "- Do not treat `Examplehighdimdiffindiff.R` as a Monte Carlo oracle for DGP2. "
            "Paper-first Python generators must keep `rho_x` live and must reuse one observed z "
            "for the heteroskedastic baseline, `exp(z)`, and the estimator input."
        ),
    ]
    return "\n".join(lines)


def build_r_snapshot_outer_inference_source_report(
    audit: RSnapshotOuterInferenceSourceAudit,
) -> str:
    lines = [
        "# R snapshot outer inference source audit",
        "",
        f"- Outer source file: `{audit.outer_source_file}`",
        f"- Inner source file: `{audit.inner_source_file}`",
        f"- Finding codes: `{', '.join(audit.finding_codes)}`",
        f"- Blocking finding codes: `{', '.join(audit.blocking_finding_codes)}`",
        f"- Status: `{audit.status}`",
        "",
        "## Findings",
        "",
        (
            f"- `RBUG-005`: outer `CIuniform` on line {audit.ciuniform_line} still uses "
            "undefined `debias`, so the archived surface has no valid binding."
        ),
        (
            f"- `RBUG-013`: outer `CIuniform` reads `ff$tc` on line {audit.ciuniform_line} "
            "without any cross-fold aggregation, so the archived surface still exhibits "
            "last-fold `ff$tc` reuse."
        ),
        (
            f"- `RBUG-011`: inner source still contains a bare `browser` token on line "
            f"{audit.browser_line}, but it is not an active `browser()` call and therefore "
            "does not block runtime by itself."
        ),
        "",
        "## Python implication",
        "",
        (
            "- Treat archived R outer inference as `reference-only`. Future parity should "
            f"target {audit.replacement_target}, not the broken `CIuniform` surface."
        ),
    ]
    return "\n".join(lines)
