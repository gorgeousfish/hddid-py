from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from hddid.outer_inference_determinism_probe import (
    Phase7OuterInferenceDeterminismProbeReport,
    run_phase7_outer_inference_determinism_probe,
)
from hddid.outer_inference_repair_obligations import (
    Phase7OuterInferenceRepairObligationsReport,
    run_phase7_outer_inference_repair_obligations_probe,
)
from hddid.outer_inference_trigger_gate import (
    Phase7OuterInferenceTriggerGateReport,
    run_phase7_outer_inference_trigger_gate,
)


_OBJECT_LEVEL_COMPARE_FIELDS = (
    "evaluation_grid",
    "bar_f_at_z0",
    "sigma_z_hat",
    "covariance_at_grid",
    "uniform_critical_value",
    "n_boot",
    "random_state",
)
_BLOCKED_BAND_FIELD = "uniform_band_bounds"
_BAND_PROMOTION_REQUIREMENTS = (
    "close-repair-obligations-on-a-repaired-path",
    "reproduce-object-level-compare-fields-on-the-same-covariance-process-contract",
    "emit-uniform-band-bounds-from-that-repaired-covariance-process-contract",
)


@dataclass(slots=True)
class Phase7OuterInferenceCompareReadinessReport:
    stage_label: str
    archived_oracle_status: str
    trigger_gate_status: str
    repair_gate_status: str
    compare_readiness_status: str
    replacement_target: str
    object_level_compare_fields: tuple[str, ...]
    blocked_band_field: str
    band_promotion_status: str
    band_promotion_requirements: tuple[str, ...]
    replay_matches_contract: bool
    replay_absolute_difference: float
    blocking_finding_codes: tuple[str, ...]
    nonblocking_finding_codes: tuple[str, ...]
    repair_obligation_codes: tuple[str, ...]
    canonical_compare_digest: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.archived_oracle_status = str(self.archived_oracle_status).strip()
        self.trigger_gate_status = str(self.trigger_gate_status).strip()
        self.repair_gate_status = str(self.repair_gate_status).strip()
        self.compare_readiness_status = str(self.compare_readiness_status).strip()
        self.replacement_target = str(self.replacement_target).strip()
        self.object_level_compare_fields = tuple(
            str(field).strip() for field in self.object_level_compare_fields
        )
        self.blocked_band_field = str(self.blocked_band_field).strip()
        self.band_promotion_status = str(self.band_promotion_status).strip()
        self.band_promotion_requirements = tuple(
            str(field).strip() for field in self.band_promotion_requirements
        )
        self.replay_matches_contract = bool(self.replay_matches_contract)
        self.replay_absolute_difference = float(self.replay_absolute_difference)
        self.blocking_finding_codes = tuple(
            str(code).strip() for code in self.blocking_finding_codes
        )
        self.nonblocking_finding_codes = tuple(
            str(code).strip() for code in self.nonblocking_finding_codes
        )
        self.repair_obligation_codes = tuple(
            str(code).strip() for code in self.repair_obligation_codes
        )
        self.canonical_compare_digest = tuple(
            str(line).rstrip() for line in self.canonical_compare_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "archived_oracle_status": self.archived_oracle_status,
            "trigger_gate_status": self.trigger_gate_status,
            "repair_gate_status": self.repair_gate_status,
            "compare_readiness_status": self.compare_readiness_status,
            "replacement_target": self.replacement_target,
            "object_level_compare_fields": list(self.object_level_compare_fields),
            "blocked_band_field": self.blocked_band_field,
            "band_promotion_status": self.band_promotion_status,
            "band_promotion_requirements": list(self.band_promotion_requirements),
            "replay_matches_contract": self.replay_matches_contract,
            "replay_absolute_difference": self.replay_absolute_difference,
            "blocking_finding_codes": list(self.blocking_finding_codes),
            "nonblocking_finding_codes": list(self.nonblocking_finding_codes),
            "repair_obligation_codes": list(self.repair_obligation_codes),
            "canonical_compare_digest": list(self.canonical_compare_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


def build_phase7_outer_inference_compare_readiness_report(
    trigger_gate_report: Phase7OuterInferenceTriggerGateReport,
    repair_report: Phase7OuterInferenceRepairObligationsReport,
    determinism_report: Phase7OuterInferenceDeterminismProbeReport,
) -> Phase7OuterInferenceCompareReadinessReport:
    compare_readiness_status = (
        "object-level-ready-band-level-blocked"
        if trigger_gate_report.gate_status == "archived-r-blocked-paper-backed-ready"
        and repair_report.gate_status == "repair-obligations-open"
        and determinism_report.replay_matches_contract
        else "compare-readiness-needs-review"
    )
    canonical_compare_digest = (
        "- Trigger 3 gate: archived R stays `reference-only` with blockers `RBUG-005, RBUG-013`; paper-backed replacement remains ready at `archived-r-blocked-paper-backed-ready`",
        "- compare-ready now: `evaluation_grid`, `bar_f_at_z0`, `sigma_z_hat`, `covariance_at_grid`, `uniform_critical_value`, `n_boot`, `random_state`",
        "- still blocked: do not compare `uniform_band_bounds` until repaired paths satisfy `repair-obligations-open` and reproduce the covariance-process contract",
    )
    return Phase7OuterInferenceCompareReadinessReport(
        stage_label="phase7-outer-inference-compare-readiness",
        archived_oracle_status=trigger_gate_report.archived_oracle_status,
        trigger_gate_status=trigger_gate_report.gate_status,
        repair_gate_status=repair_report.gate_status,
        compare_readiness_status=compare_readiness_status,
        replacement_target=trigger_gate_report.replacement_target,
        object_level_compare_fields=_OBJECT_LEVEL_COMPARE_FIELDS,
        blocked_band_field=_BLOCKED_BAND_FIELD,
        band_promotion_status="band-level-promotion-blocked",
        band_promotion_requirements=_BAND_PROMOTION_REQUIREMENTS,
        replay_matches_contract=determinism_report.replay_matches_contract,
        replay_absolute_difference=determinism_report.replay_absolute_difference,
        blocking_finding_codes=trigger_gate_report.blocking_finding_codes,
        nonblocking_finding_codes=trigger_gate_report.nonblocking_finding_codes,
        repair_obligation_codes=repair_report.repair_obligation_codes,
        canonical_compare_digest=canonical_compare_digest,
        recommendation_rationale=(
            "Trigger 3 can already compare the paper-backed covariance-process "
            "objects deterministically, but it must not compare final "
            "`uniform_band_bounds` yet. Keep archived R at `reference-only`, "
            "require `repair-obligations-open` to be satisfied by any repaired path, "
            "and only then move from object-level parity to band-level parity."
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_outer_inference_compare_readiness() -> (
    Phase7OuterInferenceCompareReadinessReport
):
    return build_phase7_outer_inference_compare_readiness_report(
        run_phase7_outer_inference_trigger_gate(),
        run_phase7_outer_inference_repair_obligations_probe(),
        run_phase7_outer_inference_determinism_probe(),
    )
