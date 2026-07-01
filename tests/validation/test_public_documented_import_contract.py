from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "hddid-py" / "README.md"
API_REFERENCE_PATH = REPO_ROOT / "hddid-py" / "docs" / "api-reference.md"
PUBLIC_DOC_PATHS = (README_PATH, API_REFERENCE_PATH)
RELEASE_CHECKLIST_PATH = REPO_ROOT / "hddid-py" / "docs" / "release-checklist.md"
# Validation helper imports are documented only in the API reference and
# release checklist.  The README is a clean user-facing document.
VALIDATION_IMPORT_DOC_PATHS = (API_REFERENCE_PATH, RELEASE_CHECKLIST_PATH)
PACKAGE_ROOT_IMPORT_DOC_PATHS = (
    REPO_ROOT / "hddid-py" / "README.md",
    RELEASE_CHECKLIST_PATH,
)
VALIDATION_HELPER_PATTERNS = (
    r"run_phase7_[A-Za-z0-9_]+",
    r"run_monte_carlo_smoke",
    r"audit_section6_[A-Za-z0-9_]+",
    r"generate_paper_dgp[12]",
)


def _documented_validation_imports(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    names: list[str] = []
    for block in re.findall(
        r"from hddid\.validation import \(\n(.*?)\n\)",
        text,
        flags=re.DOTALL,
    ):
        for raw_line in block.splitlines():
            line = raw_line.split("#", 1)[0].strip().rstrip(",")
            if line and re.fullmatch(r"[A-Za-z_]\w*", line):
                names.append(line)
    for raw_names in re.findall(
        r"from hddid\.validation import ([A-Za-z0-9_, ]+)",
        text,
    ):
        for raw_name in raw_names.split(","):
            name = raw_name.strip()
            if name and re.fullmatch(r"[A-Za-z_]\w*", name):
                names.append(name)
    return list(dict.fromkeys(names))


def _documented_package_root_imports(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    names: list[str] = []
    for block in re.findall(
        r"from hddid import \(\n(.*?)\n\)",
        text,
        flags=re.DOTALL,
    ):
        for raw_line in block.splitlines():
            line = raw_line.split("#", 1)[0].strip().rstrip(",")
            if line and re.fullmatch(r"[A-Za-z_]\w*", line):
                names.append(line)
    for raw_names in re.findall(r"from hddid import ([A-Za-z0-9_, ]+)", text):
        for raw_name in raw_names.split(","):
            name = raw_name.strip()
            if name and re.fullmatch(r"[A-Za-z_]\w*", name):
                names.append(name)
    return list(dict.fromkeys(names))


def _documented_validation_helper_references(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    names: list[str] = []
    for pattern in VALIDATION_HELPER_PATTERNS:
        names.extend(
            re.findall(
                rf"`({pattern})(?:\(\.\.\.\)|\(\))?`",
                text,
            )
        )
    return list(dict.fromkeys(names))


@pytest.mark.parametrize("doc_path", VALIDATION_IMPORT_DOC_PATHS)
def test_public_docs_validation_import_blocks_resolve(doc_path: Path) -> None:
    validation_module = importlib.import_module("hddid.validation")
    documented_names = _documented_validation_imports(doc_path)

    assert documented_names, f"{doc_path} should document validation imports"
    missing = [
        name for name in documented_names if not hasattr(validation_module, name)
    ]
    assert missing == []


@pytest.mark.parametrize("doc_path", PACKAGE_ROOT_IMPORT_DOC_PATHS)
def test_public_docs_package_root_import_blocks_resolve(doc_path: Path) -> None:
    package = importlib.import_module("hddid")
    documented_names = _documented_package_root_imports(doc_path)

    assert documented_names, f"{doc_path} should document package-root imports"
    missing = [name for name in documented_names if not hasattr(package, name)]
    assert missing == []


def test_package_root_exports_readme_public_surface() -> None:
    package = importlib.import_module("hddid")
    readme_text = README_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"from hddid import \(\n(.*?)\n\)",
        readme_text,
        flags=re.DOTALL,
    )

    assert match is not None, "README should document the package-root surface"
    documented_names = [
        line.split("#", 1)[0].strip().rstrip(",")
        for line in match.group(1).splitlines()
    ]
    documented_names = [
        name for name in documented_names if re.fullmatch(r"[A-Za-z_]\w*", name)
    ]
    assert documented_names

    missing = [name for name in documented_names if not hasattr(package, name)]
    assert missing == []


@pytest.mark.parametrize("doc_path", (API_REFERENCE_PATH,))
def test_public_docs_inline_validation_helper_references_resolve(
    doc_path: Path,
) -> None:
    validation_module = importlib.import_module("hddid.validation")
    documented_names = _documented_validation_helper_references(doc_path)

    assert documented_names, f"{doc_path} should reference validation helpers"
    missing = [
        name for name in documented_names if not hasattr(validation_module, name)
    ]
    assert missing == []
