from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tomllib

from .monte_carlo_feature_completion_gate import (
    run_phase7_monte_carlo_feature_completion_gate,
)
from .release_artifact_audit import audit_phase7_release_artifact_content


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _clean_blocker(blocker: str | None) -> str:
    if blocker is None:
        return ""
    return str(blocker).strip()


def _feature_gate_monte_carlo_ready(feature_gate) -> bool:
    status = str(
        getattr(feature_gate, "monte_carlo_validation_ready_status", "")
    ).strip()
    blocker = _clean_blocker(
        getattr(feature_gate, "monte_carlo_validation_ready_blocker", None)
    )
    if status == "ready":
        if blocker:
            raise ValueError(
                "release maturity gate requires ready Monte Carlo feature gate "
                "to have no blocker"
            )
        return True
    if not blocker:
        raise ValueError(
            "release maturity gate requires blocked Monte Carlo feature gate "
            "to expose a blocker"
        )
    return False


def _artifact_content_ready(artifact_audit) -> bool:
    release_blocking = getattr(artifact_audit, "release_blocking", None)
    if not isinstance(release_blocking, bool):
        raise ValueError(
            "release maturity gate requires artifact audit to expose "
            "boolean release_blocking"
        )
    blocker_reasons = tuple(
        str(reason).strip()
        for reason in getattr(artifact_audit, "blocker_reasons", ())
        if str(reason).strip()
    )
    if release_blocking and not blocker_reasons:
        raise ValueError(
            "release maturity gate requires blocking artifact audit to expose "
            "blocker reasons"
        )
    if not release_blocking and blocker_reasons:
        raise ValueError(
            "release maturity gate requires nonblocking artifact audit to have "
            "no blocker reasons"
        )
    return not release_blocking


@dataclass(frozen=True, slots=True)
class Phase7ReleaseMaturityGateReport:
    stage_label: str
    route_label: str
    gate_status: str
    blocker: str
    current_release_label: str
    current_version: str
    blocking_gate_names: tuple[str, ...]
    release_surfaces: tuple[str, ...]
    canonical_gate_digest: tuple[str, ...]
    roadmap_phase_count: int
    roadmap_completed_phase_count: int
    roadmap_plan_count: int
    roadmap_completed_plan_count: int
    requirements_total: int
    requirements_complete: int
    milestone_audit_status: str
    git_worktree_clean: bool
    git_dirty_path_count: int
    milestone_archive_ready: bool
    milestone_archive_blockers: tuple[str, ...]
    release_artifact_status: str
    release_artifact_pollution_present: bool
    release_artifact_blockers: tuple[str, ...]

    @property
    def helper(self) -> str:
        return "run_phase7_release_maturity_gate(...)"

    @property
    def note(self) -> str:
        return "Docs/research/phase7_release_maturity_gate.md"

    @property
    def blocker_reason(self) -> str:
        return self.blocker

    @property
    def release_status(self) -> str:
        return self.gate_status

    @property
    def release_label(self) -> str:
        return self.current_release_label

    @property
    def version(self) -> str:
        return self.current_version

    @property
    def package_version(self) -> str:
        return self.current_version

    @property
    def blocked_gates(self) -> tuple[str, ...]:
        return self.blocking_gate_names

    @property
    def release_ready(self) -> bool:
        return self.gate_status == "ready" and not self.blocking_gate_names

    @property
    def release_maturity_ready(self) -> bool:
        return self.release_ready

    @property
    def maturity_note(self) -> str:
        return "\n".join(self.canonical_gate_digest)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "route_label": self.route_label,
            "helper": self.helper,
            "note": self.note,
            "gate_status": self.gate_status,
            "release_status": self.release_status,
            "blocker": self.blocker,
            "blocker_reason": self.blocker_reason,
            "release_ready": self.release_ready,
            "release_maturity_ready": self.release_maturity_ready,
            "current_release_label": self.current_release_label,
            "current_version": self.current_version,
            "version": self.version,
            "blocking_gate_names": list(self.blocking_gate_names),
            "release_surfaces": list(self.release_surfaces),
            "canonical_gate_digest": list(self.canonical_gate_digest),
            "roadmap_phase_count": self.roadmap_phase_count,
            "roadmap_completed_phase_count": self.roadmap_completed_phase_count,
            "roadmap_plan_count": self.roadmap_plan_count,
            "roadmap_completed_plan_count": self.roadmap_completed_plan_count,
            "requirements_total": self.requirements_total,
            "requirements_complete": self.requirements_complete,
            "milestone_audit_status": self.milestone_audit_status,
            "git_worktree_clean": self.git_worktree_clean,
            "git_dirty_path_count": self.git_dirty_path_count,
            "milestone_archive_ready": self.milestone_archive_ready,
            "milestone_archive_blockers": list(self.milestone_archive_blockers),
            "release_artifact_status": self.release_artifact_status,
            "release_artifact_pollution_present": (
                self.release_artifact_pollution_present
            ),
            "release_artifact_blockers": list(self.release_artifact_blockers),
        }


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _milestone_version_sort_key(path: Path) -> tuple[int, tuple[int, ...], str]:
    match = re.match(r"v(\d+(?:\.\d+)*)(?:-|$)", path.name)
    if match is None:
        return (0, (), path.name)
    version_parts = tuple(int(part) for part in match.group(1).split("."))
    return (1, version_parts, path.name)


def _latest_milestone_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=_milestone_version_sort_key, reverse=True)


def _read_latest_archive_with_content(root: Path, suffix: str, pattern: str) -> str:
    archive_dir = root / ".planning" / "milestones"
    if not archive_dir.exists():
        return ""
    for path in _latest_milestone_paths(list(archive_dir.glob(f"*{suffix}"))):
        text = path.read_text(encoding="utf-8")
        if re.search(pattern, text, flags=re.MULTILINE):
            return text
    return ""


def _count_roadmap_progress_from_text(text: str) -> tuple[int, int, int, int]:
    phase_marks = re.findall(r"^- \[([ xX])\] \*\*Phase\s+", text, flags=re.MULTILINE)
    plan_marks = re.findall(r"^- \[([ xX])\]\s+\d", text, flags=re.MULTILINE)
    phase_count = len(phase_marks)
    completed_phase_count = sum(mark.lower() == "x" for mark in phase_marks)
    plan_count = len(plan_marks)
    completed_plan_count = sum(mark.lower() == "x" for mark in plan_marks)
    return phase_count, completed_phase_count, plan_count, completed_plan_count


def _count_roadmap_progress(root: Path) -> tuple[int, int, int, int]:
    text = _read_text_if_exists(root / ".planning" / "ROADMAP.md")
    counts = _count_roadmap_progress_from_text(text)
    if counts[0] > 0:
        return counts
    archive_text = _read_latest_archive_with_content(
        root,
        "-ROADMAP.md",
        r"^- \[[ xX]\] \*\*Phase\s+",
    )
    return _count_roadmap_progress_from_text(archive_text)


def _count_v1_requirements_from_text(text: str) -> tuple[int, int]:
    v1_text = text.split("## v2 Requirements", 1)[0]
    requirement_marks = re.findall(
        r"^- \[([ xX])\] \*\*[A-Z]+-\d+\*\*:",
        v1_text,
        flags=re.MULTILINE,
    )
    total = len(requirement_marks)
    complete = sum(mark.lower() == "x" for mark in requirement_marks)
    return total, complete


def _count_v1_requirements(root: Path) -> tuple[int, int]:
    archive_text = _read_latest_archive_with_content(
        root,
        "-REQUIREMENTS.md",
        r"^- \[[ xX]\] \*\*[A-Z]+-\d+\*\*:",
    )
    archive_counts = _count_v1_requirements_from_text(archive_text)
    if archive_counts[0] > 0:
        return archive_counts

    text = _read_text_if_exists(root / ".planning" / "REQUIREMENTS.md")
    return _count_v1_requirements_from_text(text)


def _milestone_audit_status(root: Path) -> str:
    audit_paths = sorted((root / ".planning").glob("v*-MILESTONE-AUDIT.md"))
    milestone_dir = root / ".planning" / "milestones"
    if milestone_dir.exists():
        audit_paths.extend(sorted(milestone_dir.glob("v*-MILESTONE-AUDIT.md")))
    if not audit_paths:
        return "missing"
    audit_text = _latest_milestone_paths(audit_paths)[0].read_text(encoding="utf-8")
    frontmatter = audit_text.split("---", 2)[1] if audit_text.startswith("---") else ""
    for field in ("audit_status", "status"):
        match = re.search(
            rf"^{re.escape(field)}:\s*([^\n]+)$",
            frontmatter,
            flags=re.MULTILINE,
        )
        if match is not None:
            return match.group(1).strip().strip('"').lower()
    audit_text = audit_text.lower()
    if "gaps_found" in audit_text or "gaps found" in audit_text:
        return "gaps_found"
    if "passed" in audit_text:
        return "passed"
    return "present_unknown"


def _latest_milestone_archive_present(root: Path) -> bool:
    planning_dir = root / ".planning"
    milestone_dir = planning_dir / "milestones"
    latest_roadmaps = _latest_milestone_paths(list(milestone_dir.glob("v*-ROADMAP.md")))
    if not latest_roadmaps:
        return False
    latest_roadmap = latest_roadmaps[0]
    version_prefix = latest_roadmap.name.removesuffix("-ROADMAP.md")
    archive_paths = (
        latest_roadmap,
        milestone_dir / f"{version_prefix}-REQUIREMENTS.md",
        milestone_dir / f"{version_prefix}-MILESTONE-AUDIT.md",
        planning_dir / "MILESTONES.md",
        planning_dir / "ROADMAP.md",
    )
    if not all(path.exists() for path in archive_paths):
        return False
    roadmap = _read_text_if_exists(planning_dir / "ROADMAP.md")
    milestones = _read_text_if_exists(planning_dir / "MILESTONES.md")
    return (
        f"{version_prefix}-ROADMAP.md" in roadmap
        and f"{version_prefix}-REQUIREMENTS.md" in roadmap
        and version_prefix in milestones
        and "Known non-release gap" in milestones
    )


def _v1_milestone_archive_present(root: Path) -> bool:
    return _latest_milestone_archive_present(root)


def _git_worktree_status(root: Path) -> tuple[bool, int]:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False, -1
    dirty_paths = [
        line for line in completed.stdout.splitlines() if line.strip()
    ]
    return len(dirty_paths) == 0, len(dirty_paths)


def run_phase7_release_maturity_gate(
    repo_root: str | Path,
    *,
    feature_gate=None,
    artifact_audit=None,
) -> Phase7ReleaseMaturityGateReport:
    root = _coerce_repo_root(repo_root)
    if feature_gate is None:
        feature_gate = run_phase7_monte_carlo_feature_completion_gate(root)
    if artifact_audit is None:
        artifact_audit = audit_phase7_release_artifact_content(root)
    pyproject = tomllib.loads((root / "hddid-py" / "pyproject.toml").read_text())
    version = str(pyproject["project"]["version"])
    release_label = "research-alpha"
    feature_gate_ready = _feature_gate_monte_carlo_ready(feature_gate)
    artifact_ready = _artifact_content_ready(artifact_audit)
    blocking_gate_names = tuple(
        gate_name
        for gate_name, ready in (
            ("monte_carlo_validation_ready", feature_gate_ready),
            ("release_artifact_content_ready", artifact_ready),
        )
        if not ready
    )
    if not blocking_gate_names:
        gate_status = "ready"
        blocker = ""
    else:
        gate_status = "blocked"
        blocker = (
            "release-facing-state-still-research-alpha-honesty"
            if not feature_gate_ready
            else "release-artifact-content-blocked"
        )
    release_surfaces = (
        "hddid-py/README.md",
        "hddid-py/docs/api-reference.md",
        "hddid-py/docs/validation-lanes.md",
        "hddid-py/docs/release-checklist.md",
        "Docs/parity/phase7_final_verification_matrix.md",
    )
    if not blocking_gate_names:
        digest = (
            "- release route: `release-maturity-gate` via `run_phase7_release_maturity_gate(...)`",
            f"- current release label stays `research-alpha` and current version stays `{version}`",
            "- release maturity row mirrors `run_phase7_release_maturity_gate(...)`: "
            "status `ready`; blocker `none`",
            "- release maturity row clears on the same helper read because "
            "`monte_carlo_validation_ready` is ready and "
            "`release_artifact_content_ready` is ready",
            "- release-facing surfaces remain `hddid-py/README.md`, "
            "`hddid-py/docs/api-reference.md`, `hddid-py/docs/validation-lanes.md`, "
            "`hddid-py/docs/release-checklist.md`, and "
            "`Docs/parity/phase7_final_verification_matrix.md`",
        )
    else:
        digest = (
            "- release route: `release-maturity-gate` via `run_phase7_release_maturity_gate(...)`",
            f"- current release label stays `research-alpha` and current version stays `{version}`",
            "- release maturity row mirrors `run_phase7_release_maturity_gate(...)`: "
            f"status `blocked`; blocker `{blocker}`",
            "- blocker stays active until `monte_carlo_validation_ready` clears and "
            "`release_artifact_content_ready` stays ready while `empirical_lane_ready` "
            "and `section6_provenance_gate_ready` stay satisfied",
            "- release-facing surfaces remain `hddid-py/README.md`, "
            "`hddid-py/docs/api-reference.md`, `hddid-py/docs/validation-lanes.md`, "
            "`hddid-py/docs/release-checklist.md`, and "
            "`Docs/parity/phase7_final_verification_matrix.md`",
        )
    (
        roadmap_phase_count,
        roadmap_completed_phase_count,
        roadmap_plan_count,
        roadmap_completed_plan_count,
    ) = _count_roadmap_progress(root)
    requirements_total, requirements_complete = _count_v1_requirements(root)
    milestone_audit_status = _milestone_audit_status(root)
    git_worktree_clean, git_dirty_path_count = _git_worktree_status(root)
    milestone_archive_blockers = []
    accepted_audit_statuses = {"passed", "accepted_known_gap", "passed_with_known_gap"}
    milestone_archive_present = _latest_milestone_archive_present(root)
    milestone_archive_already_closed = (
        milestone_archive_present and milestone_audit_status in accepted_audit_statuses
    )
    if not milestone_archive_already_closed and milestone_audit_status not in accepted_audit_statuses:
        milestone_archive_blockers.append(f"milestone-audit-{milestone_audit_status}")
    requirements_ready = requirements_total > 0 and requirements_complete >= requirements_total
    if not milestone_archive_already_closed and not requirements_ready:
        milestone_archive_blockers.append("requirements-incomplete")
    if not milestone_archive_already_closed and not git_worktree_clean:
        milestone_archive_blockers.append("git-worktree-dirty")
    if not milestone_archive_already_closed and blocking_gate_names:
        milestone_archive_blockers.append("release-maturity-blocked")

    return Phase7ReleaseMaturityGateReport(
        stage_label="phase7-release-maturity-gate",
        route_label="release-maturity-gate",
        gate_status=gate_status,
        blocker=blocker,
        current_release_label=release_label,
        current_version=version,
        blocking_gate_names=blocking_gate_names,
        release_surfaces=release_surfaces,
        canonical_gate_digest=digest,
        roadmap_phase_count=roadmap_phase_count,
        roadmap_completed_phase_count=roadmap_completed_phase_count,
        roadmap_plan_count=roadmap_plan_count,
        roadmap_completed_plan_count=roadmap_completed_plan_count,
        requirements_total=requirements_total,
        requirements_complete=requirements_complete,
        milestone_audit_status=milestone_audit_status,
        git_worktree_clean=git_worktree_clean,
        git_dirty_path_count=git_dirty_path_count,
        milestone_archive_ready=not milestone_archive_blockers,
        milestone_archive_blockers=tuple(milestone_archive_blockers),
        release_artifact_status=str(getattr(artifact_audit, "status", "")).strip(),
        release_artifact_pollution_present=bool(
            getattr(artifact_audit, "artifact_pollution_present", False)
        ),
        release_artifact_blockers=tuple(
            str(reason).strip()
            for reason in getattr(artifact_audit, "blocker_reasons", ())
            if str(reason).strip()
        ),
    )
