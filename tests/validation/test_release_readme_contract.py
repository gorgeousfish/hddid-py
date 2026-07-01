from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "hddid-py" / "README.md"
API_REFERENCE_PATH = REPO_ROOT / "hddid-py" / "docs" / "api-reference.md"
REPLICATION_OUTPUT_DIR = REPO_ROOT / "paper" / "replication" / "outputs"
REDUCED_WORKFLOW_SUMMARY_PATHS = {
    "official_qe": REPLICATION_OUTPUT_DIR
    / "section6_qe_reduced_hddid_run_summary.json",
    "laus_qe": REPLICATION_OUTPUT_DIR / "section6_laus_qe_hddid_run_summary.json",
    "laus_ccdb": REPLICATION_OUTPUT_DIR
    / "section6_laus_ccdb_reduced_qe_hddid_run_summary.json",
}
REDUCED_PARAMETRIC_PLOT_SUMMARY_PATH = (
    REPLICATION_OUTPUT_DIR / "section6_reduced_parametric_plot_summary.json"
)
FORBIDDEN_TOKENS = (
    "highdimdiffindiff_crossfit",
    "validate_inputs",
    "hddid.inputs",
    "research-alpha",
    "release-facing-state",
    "phase7",
)
FORBIDDEN_PHRASES = (
    "runnable empirical reproduction",
    "one-call estimator",
    "python -m hddid",
    "stata bridge",
    "supports a completed minimum-wage empirical case study",
)


def _load_readme_text() -> str:
    if not README_PATH.exists():
        pytest.fail(f"release README is missing: {README_PATH}")
    return README_PATH.read_text(encoding="utf-8")


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.fail(f"replication output is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _searchable(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"{module_name} is missing: {exc}")


def _python_blocks(readme_text: str) -> tuple[str, ...]:
    blocks = re.findall(r"```python\n(.*?)```", readme_text, flags=re.DOTALL)
    return tuple(block.strip() for block in blocks if block.strip())


def _import_statements(readme_text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    statements: list[tuple[str, tuple[str, ...]]] = []
    for block in _python_blocks(readme_text):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            # Some README blocks show function signatures (e.g. with ``*``
            # keyword-only markers) that are not valid standalone statements.
            # Skip them — only parseable blocks contribute import statements.
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in {"hddid", "hddid.validation"}:
                continue
            imported_names = tuple(alias.name for alias in node.names)
            statements.append((node.module, imported_names))
    return tuple(statements)


def test_release_readme_keeps_current_empirical_truth_visible() -> None:
    readme_text = _load_readme_text()

    # The README is a clean user-facing document; section-6 empirical evidence
    # and replication vocabulary live only in the API reference.  The JSON
    # summaries and forbidden-token checks still apply.

    for workflow_id, summary_path in REDUCED_WORKFLOW_SUMMARY_PATHS.items():
        summary = _load_json(summary_path)
        lane = summary["lanes"][0]
        assert summary["fit_status"] == "completed", workflow_id
        assert lane["parametric_inference"]["status"] == "completed", workflow_id
        assert lane["nonparametric_inference"]["status"] == "stopped", workflow_id
        assert "does not reproduce Figure 2" in summary["claim_boundary"]

    plot_summary = _load_json(REDUCED_PARAMETRIC_PLOT_SUMMARY_PATH)
    assert plot_summary["workflow_count"] == 3
    assert tuple(plot_summary["workflows"]) == (
        "laus_ccdb",
        "laus_qe",
        "official_qe",
    )
    assert "software-output visualization" in plot_summary["claim_boundary"]

    for token in FORBIDDEN_TOKENS:
        assert token not in readme_text
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in readme_text


def test_release_readme_keeps_public_workflow_and_remaining_blocker_visible() -> None:
    readme_text = _load_readme_text()
    searchable_text = _searchable(readme_text)

    for snippet in (
        "fit_hddid(...)",
        "cross-fitted nuisance estimation",
        "doubly robust score",
        "Parametric and nonparametric inference are explicit follow-up calls",
        "typed errors instead of silently producing generic standard-error rows",
        "to_summary(...)",
        "dependency-free SVG output",
    ):
        assert snippet in searchable_text


def test_release_readme_distinguishes_wheel_import_surface_from_source_checkout_gates() -> (
    None
):
    """API reference explains the source-checkout boundary.

    The README is a clean user-facing document; the wheel-vs-source-checkout
    distinction is documented only in the API reference.
    """
    api_text = _searchable(API_REFERENCE_PATH.read_text(encoding="utf-8"))

    for snippet in (
        "source-checkout",
        "SourceCheckoutRequiredError",
        "manuscript and development helpers",
    ):
        assert snippet in api_text


def test_release_readme_quickstart_no_longer_replays_old_trigger1_blocker_output() -> None:
    readme_text = _load_readme_text()

    for snippet in (
        "print(audit.status)",
        "print(provenance_gate.status)",
        "ready",
        "None",
        "trigger2-bounded-widening",
    ):
        assert snippet in readme_text

    for stale_snippet in (
        "blocked\nmissing-local-dataset",
    ):
        assert stale_snippet not in readme_text


def test_release_readme_quickstart_keeps_trigger1_loader_provenance_smoke_visible() -> None:
    readme_text = _load_readme_text()

    for snippet in (
        "hddid-py/src/hddid/section6_loader.py",
        "PYTHONPATH=hddid-py/src",
        "not\ninstalled-wheel callable APIs",
        "load_section6_data",
        "audit_phase7_section6_loader_payload",
        "payload = load_section6_data(Path(\".\").resolve())",
        "payload_audit = audit_phase7_section6_loader_payload(payload)",
        "print(payload_audit.status)",
        "print(payload_audit.county_characteristic_coverage_status)",
        "print(payload_audit.observed_county_characteristic_count)",
        "partial-local-payload",
        "print(payload[\"lane_metadata\"][\"observation_shape\"])",
        "print(payload[\"lane_metadata\"][\"basis_degree\"])",
        "print(payload[\"lane_metadata\"][\"confidence_level\"])",
    ):
        assert snippet in readme_text


def test_release_readme_import_snippets_reference_real_python_surface() -> None:
    readme_text = _load_readme_text()
    import_statements = _import_statements(readme_text)

    assert import_statements, "README must include python import examples"

    package_module = _load_module("hddid")

    seen_modules = {module_name for module_name, _ in import_statements}
    assert "hddid" in seen_modules

    for module_name, imported_names in import_statements:
        module = _load_module(module_name)
        for name in imported_names:
            assert hasattr(module, name), (
                f"README imports {name!r} from {module_name}, but that name is not exposed"
            )
