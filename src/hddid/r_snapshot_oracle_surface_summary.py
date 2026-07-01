from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hddid.outer_inference_trigger_gate import (
    Phase7OuterInferenceTriggerGateReport,
    build_phase7_outer_inference_trigger_gate_report,
)
from hddid.r_snapshot_audit import (
    RSnapshotMonteCarloOracleAudit,
    RSnapshotOuterInferenceSourceAudit,
    audit_r_snapshot_monte_carlo_oracle,
    audit_r_snapshot_outer_inference_source,
)
from hddid.r_snapshot_basis_method_audit import (
    RSnapshotBasisMethodSourceAudit,
    audit_r_snapshot_basis_method_source,
)
from hddid.validation import build_phase7_outer_inference_reference_contract


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


@dataclass(slots=True)
class Phase7RSnapshotOracleSurfaceSummaryReport:
    stage_label: str
    monte_carlo_status: str
    monte_carlo_finding_codes: tuple[str, ...]
    monte_carlo_oracle_lane: str
    basis_method_status: str
    basis_method_finding_codes: tuple[str, ...]
    parity_lane_split: tuple[str, str]
    outer_inference_status: str
    outer_inference_blocking_finding_codes: tuple[str, ...]
    outer_inference_nonblocking_finding_codes: tuple[str, ...]
    outer_inference_gate_status: str
    replacement_target: str
    canonical_route_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.monte_carlo_status = str(self.monte_carlo_status).strip()
        self.monte_carlo_finding_codes = tuple(
            str(code).strip() for code in self.monte_carlo_finding_codes
        )
        self.monte_carlo_oracle_lane = str(self.monte_carlo_oracle_lane).strip()
        self.basis_method_status = str(self.basis_method_status).strip()
        self.basis_method_finding_codes = tuple(
            str(code).strip() for code in self.basis_method_finding_codes
        )
        self.parity_lane_split = tuple(
            str(lane).strip() for lane in self.parity_lane_split
        )
        self.outer_inference_status = str(self.outer_inference_status).strip()
        self.outer_inference_blocking_finding_codes = tuple(
            str(code).strip() for code in self.outer_inference_blocking_finding_codes
        )
        self.outer_inference_nonblocking_finding_codes = tuple(
            str(code).strip() for code in self.outer_inference_nonblocking_finding_codes
        )
        self.outer_inference_gate_status = str(self.outer_inference_gate_status).strip()
        self.replacement_target = str(self.replacement_target).strip()
        self.canonical_route_digest = tuple(
            str(line).rstrip() for line in self.canonical_route_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "monte_carlo_status": self.monte_carlo_status,
            "monte_carlo_finding_codes": list(self.monte_carlo_finding_codes),
            "monte_carlo_oracle_lane": self.monte_carlo_oracle_lane,
            "basis_method_status": self.basis_method_status,
            "basis_method_finding_codes": list(self.basis_method_finding_codes),
            "parity_lane_split": list(self.parity_lane_split),
            "outer_inference_status": self.outer_inference_status,
            "outer_inference_blocking_finding_codes": list(
                self.outer_inference_blocking_finding_codes
            ),
            "outer_inference_nonblocking_finding_codes": list(
                self.outer_inference_nonblocking_finding_codes
            ),
            "outer_inference_gate_status": self.outer_inference_gate_status,
            "replacement_target": self.replacement_target,
            "canonical_route_digest": list(self.canonical_route_digest),
        }


def build_phase7_r_snapshot_oracle_surface_summary_report(
    monte_carlo_audit: RSnapshotMonteCarloOracleAudit,
    basis_method_audit: RSnapshotBasisMethodSourceAudit,
    outer_inference_audit: RSnapshotOuterInferenceSourceAudit,
    outer_inference_gate: Phase7OuterInferenceTriggerGateReport,
) -> Phase7RSnapshotOracleSurfaceSummaryReport:
    monte_carlo_status = (
        "bug-evidence-only"
        if set(monte_carlo_audit.finding_codes) >= {"RBUG-009", "RBUG-012"}
        else "needs-review"
    )
    canonical_route_digest = (
        "- Monte Carlo oracle route: `paper-trigonometric`; archived helper stays `bug-evidence-only` because `RBUG-009, RBUG-012`.",
        "- Basis-family route: keep `paper-trigonometric` separate from `r-parity-polynomial` because `RBUG-006, RBUG-008` fix the split at source level.",
        "- Outer-inference route: archived R stays `reference-only` with blockers `RBUG-005, RBUG-013`; consume paper-backed replacement objects instead.",
    )
    return Phase7RSnapshotOracleSurfaceSummaryReport(
        stage_label="phase7-r-snapshot-oracle-surface-summary",
        monte_carlo_status=monte_carlo_status,
        monte_carlo_finding_codes=monte_carlo_audit.finding_codes,
        monte_carlo_oracle_lane="paper-trigonometric",
        basis_method_status=basis_method_audit.status,
        basis_method_finding_codes=basis_method_audit.finding_codes,
        parity_lane_split=basis_method_audit.parity_lane_split,
        outer_inference_status=outer_inference_audit.status,
        outer_inference_blocking_finding_codes=outer_inference_audit.blocking_finding_codes,
        outer_inference_nonblocking_finding_codes=outer_inference_gate.nonblocking_finding_codes,
        outer_inference_gate_status=outer_inference_gate.gate_status,
        replacement_target=outer_inference_gate.replacement_target,
        canonical_route_digest=canonical_route_digest,
    )


def run_phase7_r_snapshot_oracle_surface_summary(
    repo_root: str | Path,
) -> Phase7RSnapshotOracleSurfaceSummaryReport:
    root = _coerce_repo_root(repo_root)
    monte_carlo_audit = audit_r_snapshot_monte_carlo_oracle(root)
    basis_method_audit = audit_r_snapshot_basis_method_source(root)
    outer_inference_audit = audit_r_snapshot_outer_inference_source(root)
    outer_inference_gate = build_phase7_outer_inference_trigger_gate_report(
        outer_inference_audit,
        build_phase7_outer_inference_reference_contract(),
    )
    return build_phase7_r_snapshot_oracle_surface_summary_report(
        monte_carlo_audit,
        basis_method_audit,
        outer_inference_audit,
        outer_inference_gate,
    )
