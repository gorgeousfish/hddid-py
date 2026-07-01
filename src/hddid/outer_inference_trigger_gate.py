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


_DETERMINISTIC_OBJECT_FIELDS = (
    "evaluation_grid",
    "bar_f_at_z0",
    "sigma_z_hat",
    "covariance_at_grid",
    "uniform_critical_value",
    "n_boot",
    "random_state",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _nonblocking_finding_codes(
    source_audit: RSnapshotOuterInferenceSourceAudit,
) -> tuple[str, ...]:
    blocking = set(source_audit.blocking_finding_codes)
    return tuple(code for code in source_audit.finding_codes if code not in blocking)


def _replacement_contract_ready(
    object_contract: Phase7OuterInferenceObjectContract,
) -> bool:
    grid_size = int(object_contract.evaluation_grid.shape[0])
    return (
        grid_size > 0
        and object_contract.covariance_at_grid.shape == (grid_size, grid_size)
        and object_contract.status == "reference-only"
    )


@dataclass(slots=True)
class Phase7OuterInferenceTriggerGateReport:
    stage_label: str
    archived_oracle_status: str
    blocking_finding_codes: tuple[str, ...]
    nonblocking_finding_codes: tuple[str, ...]
    replacement_target: str
    replacement_contract_status: str
    replacement_contract_ready: bool
    deterministic_object_fields: tuple[str, ...]
    covariance_shape: tuple[int, int]
    n_boot: int
    random_state: int
    gate_status: str
    canonical_gate_digest: tuple[str, ...]
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
        self.replacement_target = str(self.replacement_target).strip()
        self.replacement_contract_status = str(self.replacement_contract_status).strip()
        self.replacement_contract_ready = bool(self.replacement_contract_ready)
        self.deterministic_object_fields = tuple(
            str(field).strip() for field in self.deterministic_object_fields
        )
        self.covariance_shape = tuple(int(value) for value in self.covariance_shape)
        self.n_boot = int(self.n_boot)
        self.random_state = int(self.random_state)
        self.gate_status = str(self.gate_status).strip()
        self.canonical_gate_digest = tuple(
            str(line).rstrip() for line in self.canonical_gate_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "archived_oracle_status": self.archived_oracle_status,
            "blocking_finding_codes": list(self.blocking_finding_codes),
            "nonblocking_finding_codes": list(self.nonblocking_finding_codes),
            "replacement_target": self.replacement_target,
            "replacement_contract_status": self.replacement_contract_status,
            "replacement_contract_ready": self.replacement_contract_ready,
            "deterministic_object_fields": list(self.deterministic_object_fields),
            "covariance_shape": list(self.covariance_shape),
            "n_boot": self.n_boot,
            "random_state": self.random_state,
            "gate_status": self.gate_status,
            "canonical_gate_digest": list(self.canonical_gate_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


def build_phase7_outer_inference_trigger_gate_report(
    source_audit: RSnapshotOuterInferenceSourceAudit,
    object_contract: Phase7OuterInferenceObjectContract,
) -> Phase7OuterInferenceTriggerGateReport:
    nonblocking_finding_codes = _nonblocking_finding_codes(source_audit)
    replacement_contract_ready = _replacement_contract_ready(object_contract)
    gate_status = (
        "archived-r-blocked-paper-backed-ready"
        if source_audit.blocking_finding_codes and replacement_contract_ready
        else "trigger3-gate-needs-review"
    )
    canonical_gate_digest = (
        "- archived R oracle: "
        f"`{source_audit.status}`, blockers "
        f"`{', '.join(source_audit.blocking_finding_codes)}`, non-blocking "
        f"`{', '.join(nonblocking_finding_codes)}`",
        "- paper-backed replacement: compare "
        + ", ".join(f"`{field}`" for field in _DETERMINISTIC_OBJECT_FIELDS),
    )
    return Phase7OuterInferenceTriggerGateReport(
        stage_label="phase7-outer-inference-trigger-gate",
        archived_oracle_status=source_audit.status,
        blocking_finding_codes=source_audit.blocking_finding_codes,
        nonblocking_finding_codes=nonblocking_finding_codes,
        replacement_target=source_audit.replacement_target,
        replacement_contract_status=object_contract.status,
        replacement_contract_ready=replacement_contract_ready,
        deterministic_object_fields=_DETERMINISTIC_OBJECT_FIELDS,
        covariance_shape=tuple(
            int(value) for value in object_contract.covariance_at_grid.shape
        ),
        n_boot=object_contract.n_boot,
        random_state=object_contract.random_state,
        gate_status=gate_status,
        canonical_gate_digest=canonical_gate_digest,
        recommendation_rationale=(
            "Trigger 3 should stay closed while archived R outer inference still "
            "carries `RBUG-005` and `RBUG-013`. The next trustworthy parity object "
            "set is the paper-backed covariance-process contract, beginning with "
            "`covariance_at_grid` and `uniform_critical_value`."
        ),
    )


@lru_cache(maxsize=1)
def run_canonical_phase7_outer_inference_trigger_gate() -> (
    Phase7OuterInferenceTriggerGateReport
):
    return build_phase7_outer_inference_trigger_gate_report(
        audit_r_snapshot_outer_inference_source(_repo_root()),
        build_phase7_outer_inference_reference_contract(),
    )


def run_phase7_outer_inference_trigger_gate() -> Phase7OuterInferenceTriggerGateReport:
    return run_canonical_phase7_outer_inference_trigger_gate()
