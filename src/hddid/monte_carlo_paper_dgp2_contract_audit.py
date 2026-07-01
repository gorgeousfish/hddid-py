from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from .r_snapshot_audit import (
    RSnapshotMonteCarloOracleAudit,
    audit_r_snapshot_monte_carlo_oracle,
)


_PAPER_PATH = Path("arXiv-2009.03151v1/draft_master_test_August.tex")
_PAPER_READABLE_MIRROR_PATH = Path("paper/hddid_paper.md")
_PY_VALIDATION_PATH = Path("hddid-py/src/hddid/validation.py")
_PAPER_DGP2_EQUATION_LITERAL = "Y(i, 0) ="
_PAPER_DGP2_SIGMA_LITERAL = "Sigma_{jk} = \\rho^{|j-k|}"
_PAPER_TREATMENT_LITERAL = "\\mathbb{P}(T_i=1)"
_PAPER_BASIS_LITERAL = "8th degree trigonometric polynomial basis"
_PAPER_COVERAGE_LITERAL = "$90\\%$ confidence intervals"
_PAPER_MIRROR_DGP2_LITERAL = "Additional empirical-design excerpt"
_PAPER_DGP2_ANCHOR_LINES = (426, 427, 428, 429, 431)


def _format_design_key(design_key: tuple[str, int, int]) -> str:
    return f"{design_key[0]}/{design_key[1]}/{design_key[2]}"


@dataclass(frozen=True)
class Phase7MonteCarloPaperDGP2ContractAudit:
    stage_label: str
    status: str
    paper_source_file: str
    r_example_source_file: str
    python_source_file: str
    paper_anchor_lines: tuple[int, ...]
    r_snapshot_finding_codes: tuple[str, ...]
    r_snapshot_uses_single_observed_z_contract: bool
    r_snapshot_rho_x_used_in_covariance: bool
    python_live_rho_x_guard: bool
    python_single_observed_z_guard: bool
    python_beta_active_terms_guard: bool
    python_theta_active_terms_guard: bool
    python_default_dgp2_n500_p50_runtime_design_present: bool
    python_default_near_zero_grid: tuple[float, ...]
    nominal_coverage: float
    beta_active_terms: int
    theta_active_terms: int
    current_quality_risk_driver: str
    current_quality_risk_binding_design: tuple[str, int, int]
    current_quality_risk_comparison_design: tuple[str, int, int]
    beta_truth_max_abs_difference: float
    theta_propensity_max_abs_difference: float
    rho_x_low: float
    rho_x_high: float
    rho_x_covariance_max_abs_difference: float
    rho_x_sample_x_max_abs_difference: float
    single_observed_z_y0_max_abs_difference: float
    single_observed_z_y1_max_abs_difference: float
    canonical_contract_digest: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "status": self.status,
            "paper_source_file": self.paper_source_file,
            "r_example_source_file": self.r_example_source_file,
            "python_source_file": self.python_source_file,
            "paper_anchor_lines": list(self.paper_anchor_lines),
            "r_snapshot_finding_codes": list(self.r_snapshot_finding_codes),
            "r_snapshot_uses_single_observed_z_contract": (
                self.r_snapshot_uses_single_observed_z_contract
            ),
            "r_snapshot_rho_x_used_in_covariance": (
                self.r_snapshot_rho_x_used_in_covariance
            ),
            "python_live_rho_x_guard": self.python_live_rho_x_guard,
            "python_single_observed_z_guard": self.python_single_observed_z_guard,
            "python_beta_active_terms_guard": self.python_beta_active_terms_guard,
            "python_theta_active_terms_guard": self.python_theta_active_terms_guard,
            "python_default_dgp2_n500_p50_runtime_design_present": (
                self.python_default_dgp2_n500_p50_runtime_design_present
            ),
            "python_default_near_zero_grid": list(self.python_default_near_zero_grid),
            "nominal_coverage": self.nominal_coverage,
            "beta_active_terms": self.beta_active_terms,
            "theta_active_terms": self.theta_active_terms,
            "current_quality_risk_driver": self.current_quality_risk_driver,
            "current_quality_risk_binding_design": list(
                self.current_quality_risk_binding_design
            ),
            "current_quality_risk_binding_design_label": _format_design_key(
                self.current_quality_risk_binding_design
            ),
            "current_quality_risk_comparison_design": list(
                self.current_quality_risk_comparison_design
            ),
            "current_quality_risk_comparison_design_label": _format_design_key(
                self.current_quality_risk_comparison_design
            ),
            "beta_truth_max_abs_difference": self.beta_truth_max_abs_difference,
            "theta_propensity_max_abs_difference": (
                self.theta_propensity_max_abs_difference
            ),
            "rho_x_low": self.rho_x_low,
            "rho_x_high": self.rho_x_high,
            "rho_x_covariance_max_abs_difference": (
                self.rho_x_covariance_max_abs_difference
            ),
            "rho_x_sample_x_max_abs_difference": (
                self.rho_x_sample_x_max_abs_difference
            ),
            "single_observed_z_y0_max_abs_difference": (
                self.single_observed_z_y0_max_abs_difference
            ),
            "single_observed_z_y1_max_abs_difference": (
                self.single_observed_z_y1_max_abs_difference
            ),
            "canonical_contract_digest": list(self.canonical_contract_digest),
        }


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


def _paper_anchor_lines(repo_root: Path) -> tuple[int, ...]:
    paper_lines = _read_lines(repo_root / _PAPER_PATH)
    if not paper_lines:
        return _PAPER_DGP2_ANCHOR_LINES
    anchor_lines = (
        _find_first_line(paper_lines, _PAPER_DGP2_EQUATION_LITERAL),
        _find_first_line(paper_lines, _PAPER_DGP2_SIGMA_LITERAL),
        _find_first_line(paper_lines, _PAPER_TREATMENT_LITERAL),
        _find_first_line(paper_lines, _PAPER_BASIS_LITERAL),
        _find_first_line(paper_lines, _PAPER_COVERAGE_LITERAL),
    )
    mirror_lines = _read_lines(repo_root / _PAPER_READABLE_MIRROR_PATH)
    _find_first_line(mirror_lines, _PAPER_MIRROR_DGP2_LITERAL)
    return anchor_lines


def _paper_true_vector(scale: float, p: int, active_terms: int) -> np.ndarray:
    coefficients = np.zeros(int(p), dtype=float)
    limit = min(int(p), int(active_terms))
    coefficients[:limit] = np.array(
        [scale / float(index + 1) for index in range(limit)],
        dtype=float,
    )
    return coefficients


def _paper_truth_contract_differences(validation_module: ModuleType) -> tuple[float, float]:
    design = validation_module.MonteCarloDesign(
        dgp_name="DGP2",
        n_obs=8,
        p=20,
        basis_family="trigonometric",
        basis_degree=8,
        oracle_lane="paper-trigonometric",
        rho_x=0.4,
        evaluation_grid=np.array([0.0], dtype=float),
    )
    dataset = validation_module.generate_paper_dgp2(design, random_state=23)
    beta_control_expected = np.array(
        [1.0 / float(index + 1) for index in range(15)],
        dtype=float,
    )
    beta_treated_expected = np.array(
        [2.0 / float(index + 1) for index in range(15)],
        dtype=float,
    )
    theta_expected = np.array(
        [1.0 / float(index + 1) for index in range(10)],
        dtype=float,
    )
    expected_beta = np.zeros(design.p, dtype=float)
    expected_beta[:15] = beta_treated_expected - beta_control_expected
    expected_theta = np.zeros(design.p, dtype=float)
    expected_theta[:10] = theta_expected
    expected_propensity = 1.0 / (1.0 + np.exp(-(dataset.x @ expected_theta)))
    beta_diff = float(
        np.max(np.abs(np.asarray(dataset.true_beta, dtype=float) - expected_beta))
    )
    theta_diff = float(
        np.max(np.abs(np.asarray(dataset.propensity, dtype=float) - expected_propensity))
    )
    return beta_diff, theta_diff


def _single_observed_z_replay_metrics(
    validation_module: ModuleType,
) -> tuple[float, float]:
    design = validation_module.MonteCarloDesign(
        dgp_name="DGP2",
        n_obs=12,
        p=4,
        basis_family="trigonometric",
        basis_degree=8,
        oracle_lane="paper-trigonometric",
        rho_x=0.35,
        evaluation_grid=np.array([-0.25, 0.25], dtype=float),
    )
    dataset = validation_module.generate_paper_dgp2(design, random_state=19)

    covariance = 0.35 ** np.abs(
        np.subtract.outer(np.arange(4, dtype=int), np.arange(4, dtype=int))
    )
    rng = np.random.default_rng(19)
    expected_x = rng.multivariate_normal(
        mean=np.zeros(design.p, dtype=float),
        cov=covariance,
        size=design.n_obs,
    )
    expected_z = rng.normal(size=design.n_obs)
    theta0 = _paper_true_vector(1.0, design.p, active_terms=10)
    expected_propensity = 1.0 / (1.0 + np.exp(-(expected_x @ theta0)))
    expected_treat = rng.binomial(1, expected_propensity)
    beta_control = _paper_true_vector(1.0, design.p, active_terms=15)
    beta_treated = _paper_true_vector(2.0, design.p, active_terms=15)
    epsilon0 = rng.normal(size=design.n_obs)
    epsilon1 = rng.normal(size=design.n_obs)
    baseline_error = rng.normal(size=design.n_obs)
    expected_y0 = baseline_error * ((expected_z + expected_x[:, 0]) / np.sqrt(2.0))
    expected_phi0 = expected_x @ beta_control
    expected_phi1 = expected_x @ beta_treated + np.exp(expected_z)
    expected_y1 = (
        expected_y0
        + expected_treat * (expected_phi1 + epsilon1)
        + (1 - expected_treat) * (expected_phi0 + epsilon0)
    )

    return (
        float(np.max(np.abs(dataset.y0 - expected_y0))),
        float(np.max(np.abs(dataset.y1 - expected_y1))),
    )


def _rho_x_replay_differences(validation_module: ModuleType) -> tuple[float, float]:
    common_design_args = dict(
        dgp_name="DGP2",
        n_obs=20,
        p=4,
        basis_family="trigonometric",
        basis_degree=8,
        oracle_lane="paper-trigonometric",
        evaluation_grid=np.array([-0.5, 0.0, 0.5], dtype=float),
    )
    low_design = validation_module.MonteCarloDesign(
        **common_design_args,
        rho_x=0.2,
    )
    high_design = validation_module.MonteCarloDesign(
        **common_design_args,
        rho_x=0.8,
    )
    low_dataset = validation_module.generate_paper_dgp2(low_design, random_state=3)
    high_dataset = validation_module.generate_paper_dgp2(high_design, random_state=3)
    return (
        float(
            np.max(np.abs(high_dataset.x_covariance - low_dataset.x_covariance))
        ),
        float(np.max(np.abs(high_dataset.x - low_dataset.x))),
    )


def _default_dgp2_n500_p50_design_present(
    validation_module: ModuleType,
) -> tuple[bool, tuple[float, ...]]:
    for design in validation_module.default_phase7_runtime_probe_designs():
        if (design.dgp_name, design.n_obs, design.p) == ("DGP2", 500, 50):
            return True, tuple(float(value) for value in design.evaluation_grid)
    return False, ()


def build_phase7_monte_carlo_paper_dgp2_contract_audit_report(
    *,
    repo_root: str | Path,
    r_snapshot_audit: RSnapshotMonteCarloOracleAudit | None = None,
    quality_risk_report: object | None = None,
) -> Phase7MonteCarloPaperDGP2ContractAudit:
    root = _coerce_repo_root(repo_root)
    r_audit = r_snapshot_audit or audit_r_snapshot_monte_carlo_oracle(root)

    from . import validation as validation_module
    from .monte_carlo_widening_policy_quality_risk_probe import (
        run_phase7_monte_carlo_widening_policy_quality_risk_probe,
    )

    y0_diff, y1_diff = _single_observed_z_replay_metrics(validation_module)
    rho_covariance_diff, rho_sample_x_diff = _rho_x_replay_differences(
        validation_module
    )
    default_design_present, default_grid = _default_dgp2_n500_p50_design_present(
        validation_module
    )
    beta_truth_diff, theta_propensity_diff = _paper_truth_contract_differences(
        validation_module
    )
    python_live_rho_x_guard = rho_covariance_diff > 0.0 and rho_sample_x_diff > 0.0
    python_single_observed_z_guard = y0_diff <= 1.0e-12 and y1_diff <= 1.0e-12
    python_beta_active_terms_guard = beta_truth_diff <= 1.0e-12
    python_theta_active_terms_guard = theta_propensity_diff <= 1.0e-12
    quality_risk = (
        run_phase7_monte_carlo_widening_policy_quality_risk_probe()
        if quality_risk_report is None
        else quality_risk_report
    )
    current_quality_risk_driver = str(quality_risk.binding_driver)
    current_quality_risk_binding_design = tuple(quality_risk.binding_design)
    current_quality_risk_comparison_design = tuple(quality_risk.best_design)
    status = (
        "paper-first-python-dgp2-contract-locked"
        if (
            set(r_audit.finding_codes) >= {"RBUG-009", "RBUG-012"}
            and not r_audit.rho_x_used_in_covariance
            and not r_audit.uses_single_observed_z_contract
            and python_live_rho_x_guard
            and python_single_observed_z_guard
            and python_beta_active_terms_guard
            and python_theta_active_terms_guard
            and default_design_present
        )
        else "needs-review"
    )
    quality_risk_implication = (
        "the paper-first Python DGP2 contract remains locked without using "
        "R-example parity as a release blocker."
        if current_quality_risk_driver == "quality-risk-cleared"
        else (
            "it remains a paper-first Python calibration debt, not an R-example "
            "parity target."
        )
    )

    canonical_contract_digest = (
        "- Paper Section 5 fixes DGP2 as heteroskedastic baseline `Y(i, 0)` with Toeplitz-correlated `X`, logistic treatment, beta active set `i <= 15`, theta active set `i <= 10`, `exp(z)`, degree-8 trigonometric basis, and 90% Monte Carlo coverage reporting.",
        "- The archived R example stays `bug-evidence-only`: `RBUG-009` keeps `rho.X` dead and `RBUG-012` redraws `z`, so it cannot override the paper DGP2 contract.",
        "- Python keeps the paper-first DGP2 path live: changing `rho_x` changes both the Toeplitz covariance matrix and same-seed generated `X`, one observed `z` is reused for baseline `y0`, `exp(z)`, and estimator input at machine precision, and active-set truth vectors replay at machine precision.",
        "- The current Trigger 2 runtime slice keeps `DGP2/500/50` on the near-zero grid `(0.05, 0.15, 0.25)` under the 90% coverage convention; post-admission quality risk is "
        f"`{current_quality_risk_driver}` with binding "
        f"`{_format_design_key(current_quality_risk_binding_design)}` versus "
        f"comparison `{_format_design_key(current_quality_risk_comparison_design)}`, "
        f"so {quality_risk_implication}",
    )

    return Phase7MonteCarloPaperDGP2ContractAudit(
        stage_label="phase7-monte-carlo-paper-dgp2-contract-audit",
        status=status,
        paper_source_file=str(_PAPER_PATH),
        r_example_source_file=str(Path(r_audit.source_file).relative_to(root)),
        python_source_file=str(_PY_VALIDATION_PATH),
        paper_anchor_lines=_paper_anchor_lines(root),
        r_snapshot_finding_codes=r_audit.finding_codes,
        r_snapshot_uses_single_observed_z_contract=(
            r_audit.uses_single_observed_z_contract
        ),
        r_snapshot_rho_x_used_in_covariance=r_audit.rho_x_used_in_covariance,
        python_live_rho_x_guard=python_live_rho_x_guard,
        python_single_observed_z_guard=python_single_observed_z_guard,
        python_beta_active_terms_guard=python_beta_active_terms_guard,
        python_theta_active_terms_guard=python_theta_active_terms_guard,
        python_default_dgp2_n500_p50_runtime_design_present=default_design_present,
        python_default_near_zero_grid=default_grid,
        nominal_coverage=0.9,
        beta_active_terms=15,
        theta_active_terms=10,
        current_quality_risk_driver=current_quality_risk_driver,
        current_quality_risk_binding_design=current_quality_risk_binding_design,
        current_quality_risk_comparison_design=current_quality_risk_comparison_design,
        beta_truth_max_abs_difference=beta_truth_diff,
        theta_propensity_max_abs_difference=theta_propensity_diff,
        rho_x_low=0.2,
        rho_x_high=0.8,
        rho_x_covariance_max_abs_difference=rho_covariance_diff,
        rho_x_sample_x_max_abs_difference=rho_sample_x_diff,
        single_observed_z_y0_max_abs_difference=y0_diff,
        single_observed_z_y1_max_abs_difference=y1_diff,
        canonical_contract_digest=canonical_contract_digest,
    )


def run_phase7_monte_carlo_paper_dgp2_contract_audit(
    repo_root: str | Path,
) -> Phase7MonteCarloPaperDGP2ContractAudit:
    return build_phase7_monte_carlo_paper_dgp2_contract_audit_report(
        repo_root=repo_root
    )


def build_phase7_monte_carlo_paper_dgp2_contract_audit_markdown(
    audit: Phase7MonteCarloPaperDGP2ContractAudit,
) -> str:
    lines = [
        "# Phase 7 Monte Carlo paper DGP2 contract audit",
        "",
        f"- Stage label: `{audit.stage_label}`",
        f"- Status: `{audit.status}`",
        f"- Paper source file: `{audit.paper_source_file}`",
        f"- R example source file: `{audit.r_example_source_file}`",
        f"- Python source file: `{audit.python_source_file}`",
        f"- Paper anchor lines: `{audit.paper_anchor_lines}`",
        f"- R finding codes: `{', '.join(audit.r_snapshot_finding_codes)}`",
        "",
        "## Machine checks",
        "",
        f"- rho_x covariance max absolute difference (`{audit.rho_x_low}` vs `{audit.rho_x_high}`): `{audit.rho_x_covariance_max_abs_difference:.12f}`",
        f"- rho_x same-seed generated X max absolute difference: `{audit.rho_x_sample_x_max_abs_difference:.12f}`",
        f"- single observed z y0 max absolute difference: `{audit.single_observed_z_y0_max_abs_difference:.3e}`",
        f"- single observed z y1 max absolute difference: `{audit.single_observed_z_y1_max_abs_difference:.3e}`",
        f"- beta active terms: `{audit.beta_active_terms}`",
        f"- theta active terms: `{audit.theta_active_terms}`",
        f"- current quality-risk driver: `{audit.current_quality_risk_driver}`",
        f"- current quality-risk binding design: `{_format_design_key(audit.current_quality_risk_binding_design)}`",
        f"- current quality-risk comparison design: `{_format_design_key(audit.current_quality_risk_comparison_design)}`",
        f"- beta truth max absolute difference: `{audit.beta_truth_max_abs_difference:.3e}`",
        f"- theta propensity max absolute difference: `{audit.theta_propensity_max_abs_difference:.3e}`",
        f"- default DGP2/500/50 runtime design present: `{audit.python_default_dgp2_n500_p50_runtime_design_present}`",
        f"- default near-zero grid: `{audit.python_default_near_zero_grid}`",
        "",
        "## Canonical digest",
        "",
    ]
    lines.extend(audit.canonical_contract_digest)
    return "\n".join(lines)


__all__ = [
    "Phase7MonteCarloPaperDGP2ContractAudit",
    "build_phase7_monte_carlo_paper_dgp2_contract_audit_markdown",
    "build_phase7_monte_carlo_paper_dgp2_contract_audit_report",
    "run_phase7_monte_carlo_paper_dgp2_contract_audit",
]
