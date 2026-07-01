from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import tarfile
import tomllib
import zipfile


_INTERNAL_RELEASE_MODULE_TOKENS = (
    "probe",
    "trigger",
    "phase7",
    "automation_state",
    "release_artifact",
)
_PUBLIC_RELEASE_MODULES = frozenset(
    {
        "__init__.py",
        "basis.py",
        "estimation.py",
        "estimator.py",
        "inference.py",
        "inputs.py",
        "nuisance.py",
        "plotting.py",
        "results.py",
        "score.py",
        "splitting.py",
        "validation.py",
    }
)


@dataclass(frozen=True, slots=True)
class Phase7ReleaseArtifactAuditReport:
    stage_label: str
    artifact_kind: str
    artifact_path: str
    package_version: str
    release_label: str
    formal_release_candidate: bool
    space_suffix_modules: tuple[str, ...]
    internal_release_modules: tuple[str, ...]
    blocker_reasons: tuple[str, ...]

    @property
    def artifact_pollution_present(self) -> bool:
        return bool(self.space_suffix_modules or self.internal_release_modules)

    @property
    def release_blocking(self) -> bool:
        return self.formal_release_candidate and bool(self.blocker_reasons)

    @property
    def status(self) -> str:
        return "blocked" if self.release_blocking else "ready"

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "artifact_kind": self.artifact_kind,
            "artifact_path": self.artifact_path,
            "package_version": self.package_version,
            "release_label": self.release_label,
            "formal_release_candidate": self.formal_release_candidate,
            "space_suffix_modules": list(self.space_suffix_modules),
            "internal_release_modules": list(self.internal_release_modules),
            "artifact_pollution_present": self.artifact_pollution_present,
            "blocker_reasons": list(self.blocker_reasons),
            "release_blocking": self.release_blocking,
            "status": self.status,
        }


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _load_project_metadata(pyproject_path: Path) -> Mapping[str, object]:
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]


def _release_label(project_metadata: Mapping[str, object]) -> str:
    description = str(project_metadata.get("description", "")).lower()
    if "research-alpha" in description:
        return "research-alpha"
    if "alpha" in description:
        return "alpha"
    return "formal-release-candidate"


def _is_formal_release_candidate(
    project_metadata: Mapping[str, object],
    *,
    release_label: str,
) -> bool:
    version = str(project_metadata.get("version", "")).strip()
    _is_pre_release = version.startswith("0.")
    return not _is_pre_release or release_label != "research-alpha"


def _space_suffix_module_names(module_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in module_names
            if name.endswith(" 2.py") or name.endswith(" 3.py")
        )
    )


def _internal_release_module_names(module_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in module_names
            if (
                name not in _PUBLIC_RELEASE_MODULES
                or any(
                    token in Path(name).stem
                    for token in _INTERNAL_RELEASE_MODULE_TOKENS
                )
            )
        )
    )


def _source_module_names(package_src: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.name
            for path in package_src.glob("*.py")
        )
    )


def _module_name_from_archive_path(path_name: str) -> str | None:
    parts = tuple(part for part in path_name.split("/") if part)
    if len(parts) < 2:
        return None
    filename = parts[-1]
    if not filename.endswith(".py"):
        return None
    if parts[-2] != "hddid":
        return None
    return filename


def _zip_module_names(archive_path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(archive_path) as archive:
        module_names = [
            module_name
            for path_name in archive.namelist()
            if (module_name := _module_name_from_archive_path(path_name)) is not None
        ]
        module_names.extend(_zip_metadata_module_references(archive))
    return tuple(sorted(module_names))


def _tar_module_names(archive_path: Path) -> tuple[str, ...]:
    with tarfile.open(archive_path) as archive:
        module_names = [
            module_name
            for member in archive.getmembers()
            if (module_name := _module_name_from_archive_path(member.name)) is not None
        ]
        module_names.extend(_tar_metadata_module_references(archive))
    return tuple(sorted(module_names))


def _module_references_from_metadata_text(text: str) -> tuple[str, ...]:
    module_names: list[str] = []
    for raw_line in text.splitlines():
        path_text = raw_line.split(",", maxsplit=1)[0].strip()
        if (module_name := _module_name_from_archive_path(path_text)) is not None:
            module_names.append(module_name)
    return tuple(module_names)


def _zip_metadata_module_references(archive: zipfile.ZipFile) -> tuple[str, ...]:
    module_names: list[str] = []
    for path_name in archive.namelist():
        if not path_name.endswith(("SOURCES.txt", "RECORD")):
            continue
        text = archive.read(path_name).decode("utf-8", errors="replace")
        module_names.extend(_module_references_from_metadata_text(text))
    return tuple(module_names)


def _tar_metadata_module_references(archive: tarfile.TarFile) -> tuple[str, ...]:
    module_names: list[str] = []
    for member in archive.getmembers():
        if not member.name.endswith("SOURCES.txt"):
            continue
        extracted = archive.extractfile(member)
        if extracted is None:
            continue
        text = extracted.read().decode("utf-8", errors="replace")
        module_names.extend(_module_references_from_metadata_text(text))
    return tuple(module_names)


def _artifact_module_names(artifact_path: Path) -> tuple[str, ...]:
    suffixes = artifact_path.suffixes
    if artifact_path.suffix == ".whl" or artifact_path.suffix == ".zip":
        return _zip_module_names(artifact_path)
    if artifact_path.suffix == ".tar" or suffixes[-2:] in ([".tar", ".gz"], [".tar", ".bz2"], [".tar", ".xz"]):
        return _tar_module_names(artifact_path)
    raise ValueError(
        "release artifact audit only supports source directories, wheels, zip "
        "archives, or tar sdists"
    )


def audit_phase7_release_artifact_content(
    repo_root: str | Path,
    *,
    project_metadata: Mapping[str, object] | None = None,
    package_src: str | Path | None = None,
    package_artifact: str | Path | None = None,
) -> Phase7ReleaseArtifactAuditReport:
    root = _coerce_repo_root(repo_root)
    if package_src is not None and package_artifact is not None:
        raise ValueError("pass either package_src or package_artifact, not both")
    package_source = Path(package_src).expanduser().resolve() if package_src else None
    artifact_path = (
        Path(package_artifact).expanduser().resolve() if package_artifact else None
    )
    metadata = (
        project_metadata
        if project_metadata is not None
        else _load_project_metadata(root / "hddid-py" / "pyproject.toml")
    )

    label = _release_label(metadata)
    formal_release_candidate = _is_formal_release_candidate(
        metadata,
        release_label=label,
    )
    if artifact_path is not None:
        module_names = _artifact_module_names(artifact_path)
        artifact_kind = "archive"
        artifact_display_path = str(artifact_path)
    else:
        if package_source is None:
            package_source = root / "hddid-py" / "src" / "hddid"
        module_names = _source_module_names(package_source)
        artifact_kind = "source-directory"
        artifact_display_path = str(package_source)

    space_modules = _space_suffix_module_names(module_names)
    internal_modules = _internal_release_module_names(module_names)

    blocker_reasons: list[str] = []
    if formal_release_candidate and space_modules:
        blocker_reasons.append("space-suffixed-source-modules")
    if formal_release_candidate and internal_modules:
        blocker_reasons.append("internal-probe-trigger-modules")

    return Phase7ReleaseArtifactAuditReport(
        stage_label="phase7-release-artifact-content-audit",
        artifact_kind=artifact_kind,
        artifact_path=artifact_display_path,
        package_version=str(metadata.get("version", "")).strip(),
        release_label=label,
        formal_release_candidate=formal_release_candidate,
        space_suffix_modules=space_modules,
        internal_release_modules=internal_modules,
        blocker_reasons=tuple(blocker_reasons),
    )
