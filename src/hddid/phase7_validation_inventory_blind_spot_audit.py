from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _normalize_status_lines(
    status_lines: tuple[str, ...] | list[str] | str,
) -> tuple[str, ...]:
    if isinstance(status_lines, str):
        lines = status_lines.splitlines()
    else:
        lines = status_lines
    return tuple(str(line).rstrip() for line in lines if str(line).strip())


def _collect_untracked_assets(
    status_lines: tuple[str, ...],
    *,
    prefix: str,
    suffix: str,
) -> tuple[str, ...]:
    assets: list[str] = []
    for line in status_lines:
        if not line.startswith("?? "):
            continue
        relative_path = line[3:].strip()
        if relative_path.startswith(prefix) and relative_path.endswith(suffix):
            assets.append(relative_path)
    return tuple(assets)


def _is_listed_in_validation(validation_text: str, asset: str) -> bool:
    asset_name = Path(asset).name
    return asset in validation_text or asset_name in validation_text


@dataclass(slots=True)
class Phase7ValidationInventoryBlindSpotAuditReport:
    stage_label: str
    repo_root: str
    validation_inventory_path: str
    untracked_test_assets: tuple[str, ...]
    untracked_doc_assets: tuple[str, ...]
    missing_test_assets: tuple[str, ...]
    missing_doc_assets: tuple[str, ...]
    current_implication: str
    canonical_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.repo_root = str(self.repo_root).strip()
        self.validation_inventory_path = str(self.validation_inventory_path).strip()
        self.untracked_test_assets = tuple(
            str(item).strip() for item in self.untracked_test_assets
        )
        self.untracked_doc_assets = tuple(
            str(item).strip() for item in self.untracked_doc_assets
        )
        self.missing_test_assets = tuple(
            str(item).strip() for item in self.missing_test_assets
        )
        self.missing_doc_assets = tuple(
            str(item).strip() for item in self.missing_doc_assets
        )
        self.current_implication = str(self.current_implication).strip()
        self.canonical_digest = tuple(
            str(line).rstrip() for line in self.canonical_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "repo_root": self.repo_root,
            "validation_inventory_path": self.validation_inventory_path,
            "untracked_test_assets": list(self.untracked_test_assets),
            "untracked_doc_assets": list(self.untracked_doc_assets),
            "missing_test_assets": list(self.missing_test_assets),
            "missing_doc_assets": list(self.missing_doc_assets),
            "current_implication": self.current_implication,
            "canonical_digest": list(self.canonical_digest),
        }


def build_phase7_validation_inventory_blind_spot_audit_report(
    *,
    repo_root: str | Path,
    validation_text: str,
    status_lines: tuple[str, ...] | list[str] | str,
) -> Phase7ValidationInventoryBlindSpotAuditReport:
    resolved_repo_root = _coerce_repo_root(repo_root)
    normalized_status_lines = _normalize_status_lines(status_lines)
    validation_inventory_path = (
        resolved_repo_root
        / ".planning"
        / "phases"
        / "07-final-hardening-and-verification-debt-closure"
        / "07-VALIDATION.md"
    )
    untracked_test_assets = _collect_untracked_assets(
        normalized_status_lines,
        prefix="hddid-py/tests/",
        suffix=".py",
    )
    untracked_doc_assets = _collect_untracked_assets(
        normalized_status_lines,
        prefix="Docs/research/",
        suffix=".md",
    )
    missing_test_assets = tuple(
        asset
        for asset in untracked_test_assets
        if not _is_listed_in_validation(validation_text, asset)
    )
    missing_doc_assets = tuple(
        asset
        for asset in untracked_doc_assets
        if not _is_listed_in_validation(validation_text, asset)
    )
    if missing_test_assets or missing_doc_assets:
        current_implication = "validation-inventory-blind-spot-open"
    else:
        current_implication = "validation-inventory-aligned"
    canonical_digest = (
        f"- untracked post-closeout tests observed: `{len(untracked_test_assets)}`; missing untracked tests from `07-VALIDATION.md`: `{len(missing_test_assets)}`",
        f"- untracked post-closeout research notes observed: `{len(untracked_doc_assets)}`; missing research notes from `07-VALIDATION.md`: `{len(missing_doc_assets)}`",
        f"- current implication: `{current_implication}`",
    )
    return Phase7ValidationInventoryBlindSpotAuditReport(
        stage_label="phase7-validation-inventory-blind-spot-audit",
        repo_root=str(resolved_repo_root),
        validation_inventory_path=str(validation_inventory_path),
        untracked_test_assets=untracked_test_assets,
        untracked_doc_assets=untracked_doc_assets,
        missing_test_assets=missing_test_assets,
        missing_doc_assets=missing_doc_assets,
        current_implication=current_implication,
        canonical_digest=canonical_digest,
    )


def run_phase7_validation_inventory_blind_spot_audit(
    repo_root: str | Path,
) -> Phase7ValidationInventoryBlindSpotAuditReport:
    resolved_repo_root = _coerce_repo_root(repo_root)
    validation_inventory_path = (
        resolved_repo_root
        / ".planning"
        / "phases"
        / "07-final-hardening-and-verification-debt-closure"
        / "07-VALIDATION.md"
    )
    validation_text = validation_inventory_path.read_text(encoding="utf-8")
    status_output = subprocess.check_output(
        ["git", "status", "--short"],
        cwd=resolved_repo_root,
        text=True,
    )
    return build_phase7_validation_inventory_blind_spot_audit_report(
        repo_root=resolved_repo_root,
        validation_text=validation_text,
        status_lines=status_output,
    )


__all__ = [
    "Phase7ValidationInventoryBlindSpotAuditReport",
    "build_phase7_validation_inventory_blind_spot_audit_report",
    "run_phase7_validation_inventory_blind_spot_audit",
]
