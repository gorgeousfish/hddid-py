from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "hddid-py"
PYPROJECT_PATH = PACKAGE_ROOT / "pyproject.toml"
README_PATH = PACKAGE_ROOT / "README.md"
RELEASE_CHECKLIST_PATH = PACKAGE_ROOT / "docs" / "release-checklist.md"
RELEASE_DIAGNOSIS_PATH = PACKAGE_ROOT / "docs" / "release-diagnosis.md"
RELEASE_CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release-checks.yml"
LICENSE_PATH = PACKAGE_ROOT / "LICENSE"
BASIC_USAGE_SUMMARY_PATH = (
    REPO_ROOT / "paper" / "replication" / "outputs" / "basic_usage_summary.json"
)
MANUSCRIPT_PATH = REPO_ROOT / "paper" / "论文" / "main.tex"


def _load_pyproject() -> dict[str, object]:
    if not PYPROJECT_PATH.exists():
        pytest.fail(f"pyproject.toml is missing: {PYPROJECT_PATH}")
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def test_pyproject_keeps_research_alpha_metadata_honest() -> None:
    pyproject = _load_pyproject()
    project = pyproject["project"]

    assert project["version"] == "0.1.0"
    assert project["readme"] == "README.md"
    assert README_PATH.exists()
    assert project["license"] == "AGPL-3.0-only"
    assert project["license-files"] == ["LICENSE"]
    assert LICENSE_PATH.exists()
    assert project["authors"] == [
        {"name": "Xuanyu Cai", "email": "xuanyuCAI@outlook.com"},
        {"name": "Wenli Xu", "email": "wlxu@cityu.edu.mo"},
    ]
    assert project["urls"]["Repository"] == "https://github.com/gorgeousfish/hddid-py"

    license_text = LICENSE_PATH.read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3" in license_text

    classifiers = tuple(project.get("classifiers", []))
    assert any(c.startswith("Programming Language :: Python :: 3") for c in classifiers)
    assert any(c.startswith("Topic :: Scientific/Engineering") for c in classifiers)

    description = project["description"].lower()
    assert "paper-first" in description
    assert "research-alpha" in description

    dependencies = tuple(project.get("dependencies", []))
    assert dependencies == ("numpy>=1.24", "scipy>=1.10", "PyYAML>=6.0")

    optional = project.get("optional-dependencies", {})
    dev_dependencies = tuple(optional.get("dev", []))
    assert dev_dependencies
    assert all(dependency.startswith("pytest") for dependency in dev_dependencies)
    release_dependencies = tuple(optional.get("release", []))
    assert release_dependencies == ("build>=1.2", "twine>=5.0")

    pytest_options = pyproject.get("tool", {}).get("pytest", {}).get("ini_options", {})
    assert "asyncio_default_fixture_loop_scope" not in pytest_options


def test_release_checklist_keeps_blockers_and_local_verification_visible() -> None:
    if not RELEASE_CHECKLIST_PATH.exists():
        pytest.fail(f"release checklist is missing: {RELEASE_CHECKLIST_PATH}")

    checklist_text = RELEASE_CHECKLIST_PATH.read_text(encoding="utf-8")

    assert "research-alpha" in checklist_text
    assert "section6_county_panel.tsv" in checklist_text
    assert "broken R outer inference" in checklist_text
    assert "python -m pip install -e './hddid-py[dev]'" in checklist_text
    assert "python -m build './hddid-py' --outdir dist" in checklist_text
    assert (
        '/tmp/hddid-wheel-smoke/bin/python -m pip install "$(ls dist/hddid-*.whl)"'
        in checklist_text
    )
    assert "from hddid import (" in checklist_text
    assert "importlib.import_module(" in checklist_text
    assert "SOURCES.txt" in checklist_text
    assert "RECORD" in checklist_text
    assert "run_phase7_release_matrix_runner" in checklist_text
    assert (
        "pytest hddid-py/tests/validation/test_packaging_metadata.py -q"
        in checklist_text
    )
    assert "hddid-py/docs/release-diagnosis.md" in checklist_text


def test_release_diagnosis_records_current_release_decision() -> None:
    if not RELEASE_DIAGNOSIS_PATH.exists():
        pytest.fail(f"release diagnosis is missing: {RELEASE_DIAGNOSIS_PATH}")

    diagnosis_text = RELEASE_DIAGNOSIS_PATH.read_text(encoding="utf-8")

    for snippet in (
        "Do not make a formal public release from the current source tree.",
        "validation stage",
        "research-alpha / pre-release",
        "monte_carlo_validation_ready",
        "trigger2-runtime-evidence-packet-open",
        "quality-risk-keeps-trigger2-bounded",
        "hddid-0.1.0-py3-none-any.whl",
        "space_suffix_modules = 0",
        "internal_release_modules = 0",
        "probe/trigger/phase7/automation-state modules",
        "separates installed-wheel import smoke from the\nsource-checkout artifact audit",
        "fresh wheel environment is not asked to run\nrepo-local release gates",
        "SourceCheckoutRequiredError",
        "ModuleNotFoundError",
        "local PDF-gated manuscript replication run",
        "manuscript front-matter metadata",
        "full 38-characteristic\nSection 6 empirical estimation design",
        "Full Section 6 case-study upgrade remains incomplete",
        "Section 6 asset/provenance/loader audit is ready",
        "official-QE, LAUS--QE, and\nLAUS--CCDB",
        "section6_reduced_parametric_plot_summary.json",
        "workflow_count = 3",
        "exact 38-characteristic map and 703-covariate estimator-ready join",
        "not substantive\nminimum-wage effects",
        "completed\nlocal `pdf_build_gate`",
        "/private/tmp/pyhddid-jss-build/main.pdf",
        "author/address front-matter placeholders",
        "P0 blockers",
        "P1 gaps",
        "P2 gaps",
        "P3 polish",
    ):
        assert snippet in diagnosis_text

    assert "Section 6\nempirical provenance are not complete" not in diagnosis_text
    assert "`pdf_build_gate = null`" not in diagnosis_text
    assert "missing row-level provenance" not in diagnosis_text
    assert "Section 6 remains a pre-estimation data-readiness audit" not in diagnosis_text


def test_replication_and_manuscript_track_license_metadata_state() -> None:
    if not BASIC_USAGE_SUMMARY_PATH.exists():
        pytest.fail(f"basic usage summary is missing: {BASIC_USAGE_SUMMARY_PATH}")
    if not MANUSCRIPT_PATH.exists():
        pytest.fail(f"manuscript is missing: {MANUSCRIPT_PATH}")

    summary = json.loads(BASIC_USAGE_SUMMARY_PATH.read_text(encoding="utf-8"))
    metadata = summary["project_metadata"]
    assert metadata["license"] == "AGPL-3.0-only"
    assert metadata["license_files"] == ["LICENSE"]
    assert metadata["authors"] == [
        {"name": "Xuanyu Cai"},
        {"name": "Wenli Xu"},
    ]
    assert metadata["urls"]["Repository"] == "https://github.com/gorgeousfish/hddid-py"
    assert metadata["release_dependencies"] == ["build>=1.2", "twine>=5.0"]
    assert "python -m pip install -e './hddid-py[release]'" in summary[
        "install_commands"
    ]

    manuscript_text = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    assert "bundled AGPL license file" in manuscript_text
    for stale_blocker in (
        "license metadata is also a real blocker",
        "does not yet show a \\texttt{LICENSE} file",
        "package metadata license field",
        "license-complete submission package",
    ):
        assert stale_blocker not in manuscript_text


def test_release_ci_workflow_covers_distribution_smoke_boundary() -> None:
    if not RELEASE_CI_WORKFLOW_PATH.exists():
        pytest.fail(f"release CI workflow is missing: {RELEASE_CI_WORKFLOW_PATH}")

    workflow_text = RELEASE_CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    for snippet in (
        "python -m pip install -e './hddid-py[dev,release]'",
        "python -m build hddid-py --sdist --wheel --outdir dist",
        "python -m twine check dist/*",
        "python -m venv /tmp/hddid-wheel-smoke",
        'Path("dist").glob("hddid-*.whl")',
        '/tmp/hddid-wheel-smoke/bin/python -m pip install "$(ls dist/hddid-*.whl)"',
        "importlib.import_module(",
        "SOURCES.txt",
        "RECORD",
        "artifact metadata leaked internal modules",
        "from hddid import (",
        "HDDIDResult",
        "estimate_eq31_mainline",
        "fit_hddid",
        "SourceCheckoutRequiredError",
        "run_phase7_monte_carlo_feature_completion_gate",
        "run_phase7_monte_carlo_widening_trigger_gate",
        "run_phase7_release_maturity_gate",
        "unexpectedly ran from the wheel",
        "audit_phase7_release_artifact_content.__name__",
        "audit_phase7_release_artifact_content(",
        "package_artifact=artifact",
    ):
        assert snippet in workflow_text


def test_basic_usage_summary_keeps_result_markdown_readable() -> None:
    if not BASIC_USAGE_SUMMARY_PATH.exists():
        pytest.fail(f"basic usage summary is missing: {BASIC_USAGE_SUMMARY_PATH}")

    summary = json.loads(BASIC_USAGE_SUMMARY_PATH.read_text(encoding="utf-8"))
    outputs = summary["example_outputs"]
    markdown = outputs["result_markdown"]
    result_summary = outputs["result_summary"]

    assert "| Section | Name | Index | Label | Estimate | Std. Error | Interval |" in markdown
    assert "| Parametric | beta_hat | 0 | x0 | 1.0000 | - | - |" in markdown
    assert "| Parametric | beta_hat | 1 | x1 | -0.5000 | - | - |" in markdown
    assert (
        "| Nonparametric | gamma_hat | 0 | basis: intercept | 0.3000 | - | - |"
        in markdown
    )
    assert "| Nonparametric | f_hat_at_z0 | 1 | z0=0.75 | 0.3938 | - | - |" in markdown
    assert "Diagnostics: basis=polynomial(2)" in markdown
    assert "oracle_lane=r-parity-polynomial" in markdown
    assert "holdout=6, trimmed=1, valid=5" in markdown
    assert "|  |" not in markdown
    assert result_summary["result_contract"] == "hddid-result-summary"
    assert result_summary["row_count"] == 7
    assert result_summary["missing_standard_error_cells"] == 7
    assert result_summary["missing_interval_cells"] == 7
    assert result_summary["estimate_order"][0]["name"] == "beta_hat"

    readme_text = README_PATH.read_text(encoding="utf-8")
    replication_readme = (
        REPO_ROOT / "paper" / "replication" / "README.md"
    ).read_text(encoding="utf-8")
    assert '`to_markdown(...)`' in readme_text
    assert '`to_summary(...)`' in readme_text
    assert "β̂" in readme_text
    assert "explicit `-` cells for unavailable standard errors or intervals" in (
        replication_readme
    )
    assert "structured `HDDIDResult.to_summary(...)` payload" in replication_readme
