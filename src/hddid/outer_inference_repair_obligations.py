from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hddid.r_snapshot_audit import (
    RSnapshotOuterInferenceSourceAudit,
    audit_r_snapshot_outer_inference_source,
)
from hddid.validation import (
    Phase7OuterInferenceObjectContract,
    build_phase7_outer_inference_reference_contract,
)


_REPAIR_OBLIGATION_CODES = (
    "repair-undefined-uniform-center",
    "replace-last-fold-tc-with-covariance-process",
    "pin-monte-carlo-functional-fields",
)
_MINIMUM_COMPARE_SEQUENCE = (
    "evaluation_grid",
    "bar_f_at_z0",
    "sigma_z_hat",
    "covariance_at_grid",
    "uniform_critical_value",
    "uniform_band_bounds",
)
_REQUIRED_MONTE_CARLO_FIELDS = ("n_boot", "random_state")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _nonblocking_finding_codes(
    source_audit: RSnapshotOuterInferenceSourceAudit,
) -> tuple[str, ...]:
    blocking = set(source_audit.blocking_finding_codes)
    return tuple(code for code in source_audit.finding_codes if code not in blocking)


@dataclass(slots=True)
class Phase7OuterInferenceRepairObligationsReport:
    stage_label: str
    archived_oracle_status: str
    blocking_finding_codes: tuple[str, ...]
    nonblocking_finding_codes: tuple[str, ...]
    outer_aggregated_fold_fields: tuple[str, ...]
    repair_obligation_codes: tuple[str, ...]
    typo_only_fix_sufficient: bool
    minimum_compare_sequence: tuple[str, ...]
    required_monte_carlo_fields: tuple[str, ...]
    replacement_target: str
    gate_status: str
    canonical_repair_digest: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.archived_oracle_status = str(self.archived_oracle_status).strip()
        self.blocking_finding_codes = tuple(
            str(code).strip() for code in self.blocking_finding_codes
        )
        self.nonblocking_finding_codes = tuple(
            str(code).strip() for code in self.nonblocking_finding_codes
        )
        self.outer_aggregated_fold_fields = tuple(
            str(field).strip() for field in self.outer_aggregated_fold_fields
        )
        self.repair_obligation_codes = tuple(
            str(code).strip() for code in self.repair_obligation_codes
        )
        self.typo_only_fix_sufficient = bool(self.typo_only_fix_sufficient)
        self.minimum_compare_sequence = tuple(
            str(field).strip() for field in self.minimum_compare_sequence
        )
        self.required_monte_carlo_fields = tuple(
            str(field).strip() for field in self.required_monte_carlo_fields
        )
        self.replacement_target = str(self.replacement_target).strip()
        self.gate_status = str(self.gate_status).strip()
        self.canonical_repair_digest = tuple(
            str(line).rstrip() for line in self.canonical_repair_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "archived_oracle_status": self.archived_oracle_status,
            "blocking_finding_codes": list(self.blocking_finding_codes),
            "nonblocking_finding_codes": list(self.nonblocking_finding_codes),
            "outer_aggregated_fold_fields": list(self.outer_aggregated_fold_fields),
            "repair_obligation_codes": list(self.repair_obligation_codes),
            "typo_only_fix_sufficient": self.typo_only_fix_sufficient,
            "minimum_compare_sequence": list(self.minimum_compare_sequence),
            "required_monte_carlo_fields": list(self.required_monte_carlo_fields),
            "replacement_target": self.replacement_target,
            "gate_status": self.gate_status,
            "canonical_repair_digest": list(self.canonical_repair_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


def build_phase7_outer_inference_repair_obligations_report(
    source_audit: RSnapshotOuterInferenceSourceAudit,
    object_contract: Phase7OuterInferenceObjectContract,
) -> Phase7OuterInferenceRepairObligationsReport:
    nonblocking_finding_codes = _nonblocking_finding_codes(source_audit)
    object_contract_ready = (
        object_contract.status == "reference-only"
        and object_contract.evaluation_grid.shape[0] > 0
        and object_contract.covariance_at_grid.shape[0]
        == object_contract.covariance_at_grid.shape[1]
        == object_contract.evaluation_grid.shape[0]
        and object_contract.n_boot > 0
    )
    gate_status = (
        "repair-obligations-open"
        if source_audit.blocking_finding_codes and object_contract_ready
        else "repair-obligations-needs-review"
    )
    canonical_repair_digest = (
        "- source repair 1: replace undefined outer `debias` binding with a defined aggregated nonparametric center before any uniform band is emitted",
        "- source repair 2: do not repair Trigger 3 by aggregating last-fold `ff$tc`; emit a cross-fold `covariance_at_grid` contract first, then pin `n_boot` and `random_state` for `uniform_critical_value`",
    )
    return Phase7OuterInferenceRepairObligationsReport(
        stage_label="phase7-outer-inference-repair-obligations",
        archived_oracle_status=source_audit.status,
        blocking_finding_codes=source_audit.blocking_finding_codes,
        nonblocking_finding_codes=nonblocking_finding_codes,
        outer_aggregated_fold_fields=source_audit.outer_aggregated_fold_fields,
        repair_obligation_codes=_REPAIR_OBLIGATION_CODES,
        typo_only_fix_sufficient=False,
        minimum_compare_sequence=_MINIMUM_COMPARE_SEQUENCE,
        required_monte_carlo_fields=_REQUIRED_MONTE_CARLO_FIELDS,
        replacement_target=source_audit.replacement_target,
        gate_status=gate_status,
        canonical_repair_digest=canonical_repair_digest,
        recommendation_rationale=(
            "Trigger 3 remains closed because removing the undefined `debias` token "
            "alone would still leave outer inference without a cross-fold covariance "
            "contract. A repaired R or Stata path must emit a defined aggregated "
            "nonparametric center, reconstruct `covariance_at_grid`, and pin "
            "`n_boot` plus `random_state` before `uniform_critical_value` is worth "
            "comparing."
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_outer_inference_repair_obligations_probe() -> (
    Phase7OuterInferenceRepairObligationsReport
):
    return build_phase7_outer_inference_repair_obligations_report(
        audit_r_snapshot_outer_inference_source(_repo_root()),
        build_phase7_outer_inference_reference_contract(),
    )
