from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path

from .automation_state_view import load_top_level_automation_state_block
from .monte_carlo_feature_completion_gate import (
    run_phase7_monte_carlo_feature_completion_gate,
)
from .outer_inference_trigger_gate import run_phase7_outer_inference_trigger_gate
from .release_maturity_gate import run_phase7_release_maturity_gate
from .monte_carlo_widening_policy_quality_risk_probe import (
    Phase7MonteCarloWideningPolicyQualityRiskDesignSummary,
    build_quality_risk_average_standard_error_calibration_candidate,
    build_quality_risk_average_standard_error_calibration_candidate_evidence,
    build_quality_risk_positive_headroom_estimator_evidence_requirement,
    build_quality_risk_positive_headroom_estimator_evidence_verdict,
)
from .section6_provenance import audit_section6_provenance_gate
from .validation import audit_section6_assets

_PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER = 3.2897072539029454
_QUALITY_RISK_CANONICAL_FLOOR = 0.85
_INTERVAL_SCALE_LOCK_TOLERANCE = 1e-6


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _outer_inference_parity_status(root: Path) -> str:
    feature_gate_state = load_top_level_automation_state_block(
        root / "Docs" / "automation" / "automation-state.yaml",
        "feature_completion_gate",
    )
    checks = feature_gate_state.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("feature_completion_gate.checks must be a mapping")
    parity_state = checks.get("outer_inference_parity_ready")
    if not isinstance(parity_state, Mapping):
        raise ValueError(
            "feature_completion_gate.checks.outer_inference_parity_ready "
            "must be a mapping"
        )
    for key in ("evidence", "gate_status", "status"):
        if key in parity_state:
            value = str(parity_state[key]).strip()
            if value not in {"ready", "satisfied", "live-derived"}:
                return value
    return run_phase7_outer_inference_trigger_gate().gate_status


def _clean_blocker(blocker: str | None) -> str | None:
    if blocker is None:
        return None
    blocker_text = str(blocker).strip()
    return blocker_text or None


def _markdown_cell(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|") or "-"


def _markdown_yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _summarize_markdown_evidence(
    evidence: tuple[str, ...],
    *,
    evidence_items: int,
) -> str:
    if evidence_items < 1:
        raise ValueError("evidence_items must be at least 1")
    selected = tuple(evidence[:evidence_items])
    parts = [_markdown_cell(item) for item in selected]
    remaining = len(evidence) - len(selected)
    if remaining > 0:
        parts.append(f"... (+{remaining} more)")
    return "; ".join(parts) if parts else "-"


def _require_bool(value: bool, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _missing_quality_metric_attrs(feature_gate) -> tuple[str, ...]:
    required_attrs = (
        "quality_risk_binding_design_label",
        "quality_risk_best_design_label",
        "quality_risk_binding_coverage_floor_slack",
        "quality_risk_binding_coverage",
        "quality_risk_binding_rmse",
        "quality_risk_binding_average_standard_error",
        "quality_risk_binding_interval_length",
        "quality_risk_best_coverage",
        "quality_risk_best_rmse",
        "quality_risk_best_average_standard_error",
        "quality_risk_best_interval_length",
        "quality_risk_binding_rmse_to_standard_error_ratio",
        "quality_risk_binding_standard_error_reserve",
        "quality_risk_best_rmse_to_standard_error_ratio",
        "quality_risk_best_standard_error_reserve",
        "quality_risk_binding_interval_length_to_rmse_ratio",
        "quality_risk_best_interval_length_to_rmse_ratio",
        "quality_risk_binding_interval_length_to_standard_error_ratio",
        "quality_risk_best_interval_length_to_standard_error_ratio",
        "quality_risk_rmse_to_standard_error_ratio_gap",
        "quality_risk_standard_error_reserve_gap",
        "quality_risk_interval_length_to_rmse_gap",
        "quality_risk_rmse_outpaces_average_se_gap",
        "quality_risk_clearance_evidence",
        "quality_risk_calibration_targets",
        "dominant_quality_risk_calibration_target",
        "quality_risk_method_diagnosis",
        "quality_risk_method_diagnosis_evidence",
        "quality_risk_action_contract",
        "quality_risk_action_evidence",
    )
    return tuple(attr for attr in required_attrs if not hasattr(feature_gate, attr))


def _require_positive_headroom_verdict_attrs(feature_gate) -> None:
    required_attrs = (
        "quality_risk_positive_headroom_estimator_evidence_verdict",
        "positive_headroom_estimator_evidence_verdict",
        "quality_risk_positive_headroom_estimator_evidence_verdict_evidence",
        "positive_headroom_estimator_evidence_verdict_evidence",
    )
    missing_attrs = tuple(
        attr for attr in required_attrs if not hasattr(feature_gate, attr)
    )
    if missing_attrs:
        raise ValueError(
            "release matrix runner requires Monte Carlo feature gate to expose "
            "positive-headroom verdict attributes before building release row "
            f"metadata; missing: {', '.join(missing_attrs)}"
        )


def _condition_status(condition: dict[str, object]) -> str:
    return str(condition.get("status", "")).strip()


def _expected_quality_risk_clearance_conditions(
    feature_gate,
) -> tuple[dict[str, object], ...]:
    coverage_floor_slack = float(feature_gate.quality_risk_binding_coverage_floor_slack)
    standard_error_reserve = float(
        feature_gate.quality_risk_binding_standard_error_reserve
    )
    rmse_to_se_gap = float(feature_gate.quality_risk_rmse_to_standard_error_ratio_gap)
    se_reserve_gap = float(feature_gate.quality_risk_standard_error_reserve_gap)
    interval_to_rmse_gap = float(feature_gate.quality_risk_interval_length_to_rmse_gap)
    binding_driver = str(feature_gate.quality_risk_binding_driver).strip()
    return (
        {
            "condition": "coverage_floor",
            "status": "cleared" if coverage_floor_slack >= 0.0 else "open",
            "deficit": max(-coverage_floor_slack, 0.0),
            "margin": coverage_floor_slack,
        },
        {
            "condition": "nonnegative_se_reserve",
            "status": "cleared" if standard_error_reserve >= 0.0 else "open",
            "deficit": max(-standard_error_reserve, 0.0),
            "margin": standard_error_reserve,
        },
        {
            "condition": "rmse_to_se_not_trailing",
            "status": "cleared" if rmse_to_se_gap <= 0.0 else "open",
            "deficit": max(rmse_to_se_gap, 0.0),
        },
        {
            "condition": "se_reserve_not_trailing",
            "status": "cleared" if se_reserve_gap <= 0.0 else "open",
            "deficit": max(se_reserve_gap, 0.0),
        },
        {
            "condition": "interval_to_rmse_not_trailing",
            "status": "cleared" if interval_to_rmse_gap <= 0.0 else "open",
            "deficit": max(interval_to_rmse_gap, 0.0),
        },
        {
            "condition": "overall",
            "status": (
                "cleared" if binding_driver == "quality-risk-cleared" else "blocked"
            ),
            "driver": binding_driver,
        },
    )


def _conditions_match(
    actual: tuple[dict[str, object], ...],
    expected: tuple[dict[str, object], ...],
) -> bool:
    for actual_item, expected_item in zip(actual, expected, strict=True):
        for key, expected_value in expected_item.items():
            actual_value = actual_item.get(key)
            if isinstance(expected_value, float):
                if not math.isclose(
                    float(actual_value),
                    expected_value,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    return False
            elif actual_value != expected_value:
                return False
    return True


def _quality_risk_clearance_conditions(feature_gate) -> tuple[dict[str, object], ...]:
    expected_conditions = _expected_quality_risk_clearance_conditions(feature_gate)
    conditions = tuple(
        dict(condition)
        for condition in getattr(
            feature_gate,
            "quality_risk_clearance_conditions",
            expected_conditions,
        )
    )
    if not conditions:
        raise ValueError(
            "release matrix runner requires Monte Carlo feature gate quality "
            "clearance conditions from the live feature-gate read"
        )
    if len(conditions) != 6:
        raise ValueError(
            "release matrix runner requires Monte Carlo feature gate quality "
            "clearance conditions to expose five metric conditions plus the "
            "overall verdict"
        )
    expected_names = (
        "coverage_floor",
        "nonnegative_se_reserve",
        "rmse_to_se_not_trailing",
        "se_reserve_not_trailing",
        "interval_to_rmse_not_trailing",
        "overall",
    )
    actual_names = tuple(str(item.get("condition", "")).strip() for item in conditions)
    if actual_names != expected_names:
        raise ValueError(
            "release matrix runner requires Monte Carlo feature gate quality "
            "clearance conditions to match the canonical condition order"
        )
    if not _conditions_match(conditions, expected_conditions):
        raise ValueError(
            "release matrix runner requires Monte Carlo feature gate quality "
            "clearance conditions to match the live quality metrics"
        )
    return conditions


def _quality_risk_clearance_evidence(feature_gate) -> tuple[str, ...]:
    clearance_evidence = tuple(
        str(item).strip()
        for item in getattr(feature_gate, "quality_risk_clearance_evidence", ())
    )
    if not clearance_evidence:
        raise ValueError(
            "release matrix runner requires Monte Carlo feature gate quality "
            "clearance evidence from the live feature-gate read"
        )
    conditions = _quality_risk_clearance_conditions(feature_gate)
    expected_evidence = (
        (
            "clearance coverage_floor "
            f"status={conditions[0]['status']} "
            f"deficit={float(conditions[0]['deficit']):.3f} "
            f"margin={float(conditions[0]['margin']):+.3f}"
        ),
        (
            "clearance nonnegative_se_reserve "
            f"status={conditions[1]['status']} "
            f"deficit={float(conditions[1]['deficit']):.3f} "
            f"margin={float(conditions[1]['margin']):+.3f}"
        ),
        (
            "clearance rmse_to_se_not_trailing "
            f"status={conditions[2]['status']} "
            f"deficit={float(conditions[2]['deficit']):.3f}"
        ),
        (
            "clearance se_reserve_not_trailing "
            f"status={conditions[3]['status']} "
            f"deficit={float(conditions[3]['deficit']):.3f}"
        ),
        (
            "clearance interval_to_rmse_not_trailing "
            f"status={conditions[4]['status']} "
            f"deficit={float(conditions[4]['deficit']):.3f}"
        ),
        (
            f"clearance overall status={conditions[5]['status']} "
            f"driver={conditions[5]['driver']}"
        ),
    )
    if clearance_evidence != expected_evidence:
        raise ValueError(
            "release matrix runner requires Monte Carlo feature gate quality "
            "clearance evidence to match the structured clearance conditions"
        )
    return clearance_evidence


def _require_blocked_quality_metric_evidence(
    feature_gate, evidence: tuple[str, ...]
) -> None:
    missing_attrs = _missing_quality_metric_attrs(feature_gate)
    if missing_attrs:
        raise ValueError(
            "release matrix runner requires blocked Monte Carlo feature gate "
            "to expose live quality metric attributes before accepting quality "
            "metric evidence; missing: "
            f"{', '.join(missing_attrs)}"
        )

    binding_label = str(feature_gate.quality_risk_binding_design_label)
    best_label = str(feature_gate.quality_risk_best_design_label)
    clearance_markers = _quality_risk_clearance_evidence(feature_gate)
    expected_markers = (
        f"quality binding {binding_label}",
        f"coverage slack {feature_gate.quality_risk_binding_coverage_floor_slack:+.3f}",
        (
            f"binding raw metrics {binding_label} "
            f"coverage {feature_gate.quality_risk_binding_coverage:.3f} "
            f"RMSE {feature_gate.quality_risk_binding_rmse:.3f} "
            f"average SE {feature_gate.quality_risk_binding_average_standard_error:.3f} "
            f"interval length {feature_gate.quality_risk_binding_interval_length:.3f}"
        ),
        (
            f"best raw metrics {best_label} "
            f"coverage {feature_gate.quality_risk_best_coverage:.3f} "
            f"RMSE {feature_gate.quality_risk_best_rmse:.3f} "
            f"average SE {feature_gate.quality_risk_best_average_standard_error:.3f} "
            f"interval length {feature_gate.quality_risk_best_interval_length:.3f}"
        ),
        (
            f"binding {binding_label} RMSE/SE "
            f"{feature_gate.quality_risk_binding_rmse_to_standard_error_ratio:.3f}"
        ),
        (
            f"best {best_label} RMSE/SE "
            f"{feature_gate.quality_risk_best_rmse_to_standard_error_ratio:.3f}"
        ),
        (
            f"binding {binding_label} SE-RMSE "
            f"{feature_gate.quality_risk_binding_standard_error_reserve:+.3f}"
        ),
        (
            f"best {best_label} SE-RMSE "
            f"{feature_gate.quality_risk_best_standard_error_reserve:+.3f}"
        ),
        (
            f"binding {binding_label} interval/RMSE "
            f"{feature_gate.quality_risk_binding_interval_length_to_rmse_ratio:.3f}"
        ),
        (
            f"best {best_label} interval/RMSE "
            f"{feature_gate.quality_risk_best_interval_length_to_rmse_ratio:.3f}"
        ),
        (
            f"binding {binding_label} interval/SE "
            f"{feature_gate.quality_risk_binding_interval_length_to_standard_error_ratio:.3f}"
        ),
        (
            f"best {best_label} interval/SE "
            f"{feature_gate.quality_risk_best_interval_length_to_standard_error_ratio:.3f}"
        ),
        f"RMSE/SE gap {feature_gate.quality_risk_rmse_to_standard_error_ratio_gap:+.3f}",
        f"SE-RMSE gap {feature_gate.quality_risk_standard_error_reserve_gap:+.3f}",
        (
            "RMSE-average SE gap "
            f"{feature_gate.quality_risk_rmse_outpaces_average_se_gap:+.3f}"
        ),
        f"interval/RMSE gap {feature_gate.quality_risk_interval_length_to_rmse_gap:+.3f}",
        *clearance_markers,
        *tuple(feature_gate.quality_risk_method_diagnosis_evidence),
        *tuple(
            getattr(
                feature_gate,
                "quality_risk_average_standard_error_calibration_candidate_evidence",
                (),
            )
        ),
        *tuple(
            getattr(
                feature_gate,
                "quality_risk_positive_headroom_estimator_evidence_verdict_evidence",
                (),
            )
        ),
        *tuple(getattr(feature_gate, "quality_risk_action_evidence", ())),
    )
    missing = tuple(marker for marker in expected_markers if marker not in evidence)
    if missing:
        raise ValueError(
            "release matrix runner requires blocked Monte Carlo feature gate "
            "quality metric evidence to match the live feature-gate metrics; "
            f"missing: {', '.join(missing)}"
        )
    _require_paper_interval_scale_lock(feature_gate)


def _require_ready_quality_metric_evidence(
    feature_gate, evidence: tuple[str, ...]
) -> None:
    missing_attrs = _missing_quality_metric_attrs(feature_gate)
    if missing_attrs:
        raise ValueError(
            "release matrix runner requires ready Monte Carlo feature gate "
            "to expose live quality metric attributes before accepting quality "
            "metric evidence; missing: "
            f"{', '.join(missing_attrs)}"
        )

    binding_label = str(feature_gate.quality_risk_binding_design_label)
    best_label = str(feature_gate.quality_risk_best_design_label)
    clearance_markers = _quality_risk_clearance_evidence(feature_gate)
    clearance_conditions = _quality_risk_clearance_conditions(feature_gate)
    if any(
        _condition_status(condition) == "open" for condition in clearance_conditions
    ):
        raise ValueError(
            "release matrix runner requires ready Monte Carlo feature gate "
            "clearance-condition evidence to be fully cleared"
        )

    expected_markers = (
        "post-admission quality-risk-cleared",
        f"quality binding {binding_label}",
        f"coverage slack {feature_gate.quality_risk_binding_coverage_floor_slack:+.3f}",
        (
            f"binding raw metrics {binding_label} "
            f"coverage {feature_gate.quality_risk_binding_coverage:.3f} "
            f"RMSE {feature_gate.quality_risk_binding_rmse:.3f} "
            f"average SE {feature_gate.quality_risk_binding_average_standard_error:.3f} "
            f"interval length {feature_gate.quality_risk_binding_interval_length:.3f}"
        ),
        (
            f"best raw metrics {best_label} "
            f"coverage {feature_gate.quality_risk_best_coverage:.3f} "
            f"RMSE {feature_gate.quality_risk_best_rmse:.3f} "
            f"average SE {feature_gate.quality_risk_best_average_standard_error:.3f} "
            f"interval length {feature_gate.quality_risk_best_interval_length:.3f}"
        ),
        (
            f"binding {binding_label} RMSE/SE "
            f"{feature_gate.quality_risk_binding_rmse_to_standard_error_ratio:.3f}"
        ),
        (
            f"best {best_label} RMSE/SE "
            f"{feature_gate.quality_risk_best_rmse_to_standard_error_ratio:.3f}"
        ),
        (
            f"binding {binding_label} SE-RMSE "
            f"{feature_gate.quality_risk_binding_standard_error_reserve:+.3f}"
        ),
        (
            f"best {best_label} SE-RMSE "
            f"{feature_gate.quality_risk_best_standard_error_reserve:+.3f}"
        ),
        (
            f"binding {binding_label} interval/RMSE "
            f"{feature_gate.quality_risk_binding_interval_length_to_rmse_ratio:.3f}"
        ),
        (
            f"best {best_label} interval/RMSE "
            f"{feature_gate.quality_risk_best_interval_length_to_rmse_ratio:.3f}"
        ),
        (
            f"binding {binding_label} interval/SE "
            f"{feature_gate.quality_risk_binding_interval_length_to_standard_error_ratio:.3f}"
        ),
        (
            f"best {best_label} interval/SE "
            f"{feature_gate.quality_risk_best_interval_length_to_standard_error_ratio:.3f}"
        ),
        f"RMSE/SE gap {feature_gate.quality_risk_rmse_to_standard_error_ratio_gap:+.3f}",
        f"SE-RMSE gap {feature_gate.quality_risk_standard_error_reserve_gap:+.3f}",
        (
            "RMSE-average SE gap "
            f"{feature_gate.quality_risk_rmse_outpaces_average_se_gap:+.3f}"
        ),
        f"interval/RMSE gap {feature_gate.quality_risk_interval_length_to_rmse_gap:+.3f}",
        *clearance_markers,
        *tuple(feature_gate.quality_risk_method_diagnosis_evidence),
        *tuple(getattr(feature_gate, "quality_risk_action_evidence", ())),
    )
    missing = tuple(marker for marker in expected_markers if marker not in evidence)
    if missing:
        raise ValueError(
            "release matrix runner requires ready Monte Carlo feature gate "
            "quality metric evidence to match the live feature-gate metrics; "
            f"missing: {', '.join(missing)}"
        )
    _require_paper_interval_scale_lock(feature_gate)


def _require_paper_interval_scale_lock(feature_gate) -> None:
    ratios = (
        (
            "binding",
            float(feature_gate.quality_risk_binding_interval_length_to_standard_error_ratio),
        ),
        (
            "best",
            float(feature_gate.quality_risk_best_interval_length_to_standard_error_ratio),
        ),
    )
    stale = tuple(
        f"{label} interval/SE {ratio:.6f}"
        for label, ratio in ratios
        if not math.isclose(
            ratio,
            _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER,
            rel_tol=1e-9,
            abs_tol=_INTERVAL_SCALE_LOCK_TOLERANCE,
        )
    )
    if stale:
        raise ValueError(
            "release matrix runner requires Monte Carlo feature gate interval/SE "
            "evidence to match the paper 90% pointwise interval scale; found: "
            f"{', '.join(stale)}"
        )


def _require_blocked_status_evidence(feature_gate, evidence: tuple[str, ...]) -> None:
    expected_markers: list[str] = []
    if hasattr(feature_gate, "runtime_evidence_admission_status"):
        expected_markers.append(
            "runtime admission "
            f"{str(feature_gate.runtime_evidence_admission_status).strip()}"
        )
    if hasattr(feature_gate, "post_admission_quality_risk_status"):
        expected_markers.append(
            "post-admission "
            f"{str(feature_gate.post_admission_quality_risk_status).strip()}"
        )
    if hasattr(feature_gate, "quality_risk_source_mode"):
        expected_markers.append(
            "quality risk source "
            f"{str(feature_gate.quality_risk_source_mode).strip()}"
        )

    missing = tuple(marker for marker in expected_markers if marker not in evidence)
    if missing:
        raise ValueError(
            "release matrix runner requires blocked Monte Carlo feature gate "
            "status evidence to match the live feature-gate status; missing: "
            f"{', '.join(missing)}"
        )


def _require_blocked_quality_risk_consistency(
    feature_gate,
    *,
    blocker: str,
    evidence: tuple[str, ...],
) -> None:
    if blocker != "quality-risk-keeps-trigger2-bounded":
        return

    if "post-admission quality-risk-cleared" in evidence:
        raise ValueError(
            "release matrix runner requires blocked Monte Carlo feature gate "
            "quality-risk blocker to keep post-admission quality risk open"
        )

    quality_status = getattr(
        feature_gate,
        "post_admission_quality_risk_status",
        None,
    )
    if quality_status is not None and str(quality_status).strip() == (
        "quality-risk-cleared"
    ):
        raise ValueError(
            "release matrix runner requires blocked Monte Carlo feature gate "
            "quality-risk blocker to keep post-admission quality risk open"
        )

    for attr in (
        "quality_risk_binding_driver",
        "post_admission_quality_risk_driver",
    ):
        driver = getattr(feature_gate, attr, None)
        if driver is not None and str(driver).strip() == "quality-risk-cleared":
            raise ValueError(
                "release matrix runner requires blocked Monte Carlo feature gate "
                "quality-risk driver to remain open"
            )

    binding_standard_error_reserve = getattr(
        feature_gate,
        "quality_risk_binding_standard_error_reserve",
        None,
    )
    if (
        binding_standard_error_reserve is not None
        and float(binding_standard_error_reserve) < 0.0
    ):
        for attr in (
            "quality_risk_binding_driver",
            "post_admission_quality_risk_driver",
        ):
            driver = getattr(feature_gate, attr, None)
            if driver is not None and str(driver).strip() != (
                "rmse-outpaces-average-se"
            ):
                raise ValueError(
                    "release matrix runner requires blocked Monte Carlo feature "
                    "gate quality-risk driver to prioritize binding RMSE "
                    "outpacing average SE when binding SE-RMSE is negative"
                )

    quality_blocker = getattr(feature_gate, "quality_risk_blocker_reason", None)
    if quality_blocker is not None and _clean_blocker(quality_blocker) != blocker:
        raise ValueError(
            "release matrix runner requires blocked Monte Carlo feature gate "
            "quality-risk blocker reason to match the Monte Carlo blocker"
        )


def _feature_gate_evidence_tuple(feature_gate) -> tuple[str, ...]:
    evidence = tuple(
        str(item).strip()
        for item in getattr(feature_gate, "monte_carlo_validation_evidence", ())
    )
    status = str(
        getattr(feature_gate, "monte_carlo_validation_ready_status", "")
    ).strip()
    blocker = _clean_blocker(
        getattr(feature_gate, "monte_carlo_validation_ready_blocker", None)
    )
    if status == "ready":
        if blocker is not None:
            raise ValueError(
                "release matrix runner requires ready Monte Carlo feature gate "
                "to have no blocker"
            )
        if not evidence:
            raise ValueError(
                "release matrix runner requires ready Monte Carlo feature gate "
                "evidence to expose promotion markers"
            )
        quality_source = getattr(feature_gate, "quality_risk_source_mode", None)
        if quality_source is not None and str(quality_source).strip() != (
            "live-runtime-probe"
        ):
            raise ValueError(
                "release matrix runner requires ready Monte Carlo feature gate "
                "quality-risk source to be live-runtime-probe"
            )
        stale_blocked_markers = (
            "quality-risk-still-open",
            "quality-risk-keeps-trigger2-bounded",
            "trigger2-runtime-evidence-packet-open",
        )
        stale_evidence = tuple(
            item
            for item in evidence
            if any(marker in item for marker in stale_blocked_markers)
        )
        if stale_evidence:
            raise ValueError(
                "release matrix runner requires ready Monte Carlo feature gate "
                "evidence to exclude stale blocked markers; found: "
                f"{', '.join(stale_evidence)}"
            )
        quality_status = getattr(
            feature_gate,
            "post_admission_quality_risk_status",
            None,
        )
        if (
            quality_status is not None
            and str(quality_status).strip() != "quality-risk-cleared"
        ):
            raise ValueError(
                "release matrix runner requires ready Monte Carlo feature gate "
                "post-admission quality risk to be cleared"
            )
        quality_driver = getattr(feature_gate, "quality_risk_binding_driver", None)
        if quality_driver is not None and str(quality_driver).strip() != (
            "quality-risk-cleared"
        ):
            raise ValueError(
                "release matrix runner requires ready Monte Carlo feature gate "
                "quality-risk metrics to be cleared"
            )
        post_quality_driver = getattr(
            feature_gate,
            "post_admission_quality_risk_driver",
            None,
        )
        if post_quality_driver is not None and str(post_quality_driver).strip() != (
            "quality-risk-cleared"
        ):
            raise ValueError(
                "release matrix runner requires ready Monte Carlo feature gate "
                "post-admission quality-risk driver to be cleared"
            )
        quality_blocker = getattr(feature_gate, "quality_risk_blocker_reason", "")
        if _clean_blocker(quality_blocker) is not None:
            raise ValueError(
                "release matrix runner requires ready Monte Carlo feature gate "
                "to have no quality-risk blocker"
            )
        runtime_admission = getattr(
            feature_gate,
            "runtime_evidence_admission_status",
            None,
        )
        if runtime_admission is not None and str(runtime_admission).strip() != (
            "trigger2-runtime-evidence-quota-closed"
        ):
            raise ValueError(
                "release matrix runner requires ready Monte Carlo feature gate "
                "runtime admission to be closed"
            )
        if evidence and not any(
            item == "post-admission quality-risk-cleared" for item in evidence
        ):
            raise ValueError(
                "release matrix runner requires ready Monte Carlo feature gate "
                "evidence to expose post-admission quality-risk-cleared"
            )
        _require_ready_quality_metric_evidence(feature_gate, evidence)
        return evidence
    if blocker is None:
        raise ValueError(
            "release matrix runner requires blocked Monte Carlo feature gate "
            "to expose a blocker"
        )

    required_markers = (
        "runtime admission ",
        "post-admission ",
        "quality risk source ",
        "quality binding ",
        "coverage slack ",
        "binding raw metrics ",
        "best raw metrics ",
        "binding ",
        "best ",
        "RMSE/SE gap ",
        "SE-RMSE gap ",
        "RMSE-average SE gap ",
        "interval/RMSE gap ",
    )
    missing = tuple(
        marker
        for marker in required_markers
        if not any(item.startswith(marker) for item in evidence)
    )
    if missing:
        raise ValueError(
            "release matrix runner requires blocked Monte Carlo feature gate "
            "evidence to expose runtime admission, post-admission, and quality "
            f"metric driver markers; missing: {', '.join(missing)}"
        )

    quality_driver = getattr(feature_gate, "quality_risk_binding_driver", None)
    if quality_driver is not None and str(quality_driver).strip() not in evidence:
        raise ValueError(
            "release matrix runner requires Monte Carlo evidence to include the "
            "feature gate quality-risk binding driver"
        )
    _require_blocked_quality_metric_evidence(feature_gate, evidence)
    _require_blocked_status_evidence(feature_gate, evidence)
    _require_blocked_quality_risk_consistency(
        feature_gate,
        blocker=blocker,
        evidence=evidence,
    )
    return evidence


def _release_maturity_evidence_tuple(
    release_gate,
    feature_gate,
    monte_carlo_evidence: tuple[str, ...],
) -> tuple[str, ...]:
    monte_carlo_status = str(
        getattr(feature_gate, "monte_carlo_validation_ready_status", "")
    ).strip()
    monte_carlo_blocker = _clean_blocker(
        getattr(feature_gate, "monte_carlo_validation_ready_blocker", None)
    )
    return (
        str(release_gate.current_release_label).strip(),
        str(release_gate.current_version).strip(),
        "monte_carlo_validation_ready",
        f"monte_carlo_validation_ready status {monte_carlo_status}",
        f"monte_carlo_validation_ready blocker {monte_carlo_blocker or 'none'}",
        "release_artifact_content_ready",
        "release_artifact_content_ready status "
        f"{getattr(release_gate, 'release_artifact_status', '') or 'unknown'}",
        "release_artifact_content_ready pollution_present "
        + _markdown_yes_no(
            bool(getattr(release_gate, "release_artifact_pollution_present", False))
        ),
        "release_artifact_content_ready blockers "
        + (
            ",".join(getattr(release_gate, "release_artifact_blockers", ()))
            if getattr(release_gate, "release_artifact_blockers", ())
            else "none"
        ),
        *(
            f"monte_carlo_validation_ready evidence {item}"
            for item in monte_carlo_evidence
        ),
    )


def _quality_risk_raw_metric_metadata(feature_gate) -> dict[str, dict[str, float]]:
    return {
        "binding": {
            "coverage": float(feature_gate.quality_risk_binding_coverage),
            "rmse": float(feature_gate.quality_risk_binding_rmse),
            "average_standard_error": float(
                feature_gate.quality_risk_binding_average_standard_error
            ),
            "interval_length": float(feature_gate.quality_risk_binding_interval_length),
            "standard_error_reserve": float(
                feature_gate.quality_risk_binding_standard_error_reserve
            ),
        },
        "best": {
            "coverage": float(feature_gate.quality_risk_best_coverage),
            "rmse": float(feature_gate.quality_risk_best_rmse),
            "average_standard_error": float(
                feature_gate.quality_risk_best_average_standard_error
            ),
            "interval_length": float(feature_gate.quality_risk_best_interval_length),
            "standard_error_reserve": float(
                feature_gate.quality_risk_best_standard_error_reserve
            ),
        },
    }


def _parse_design_label(label: str) -> tuple[str, int, int]:
    dgp_name, n_obs, p = str(label).strip().split("/")
    return dgp_name, int(n_obs), int(p)


def _quality_risk_average_standard_error_calibration_candidate(
    feature_gate,
) -> dict[str, object]:
    candidate = dict(
        getattr(
            feature_gate,
            "quality_risk_average_standard_error_calibration_candidate",
            {},
        )
    )
    if not candidate:
        raise ValueError(
            "release matrix runner requires Monte Carlo feature gate average-SE "
            "calibration candidate from the live feature-gate read"
    )

    binding_summary = Phase7MonteCarloWideningPolicyQualityRiskDesignSummary(
        *_parse_design_label(feature_gate.quality_risk_binding_design_label),
        mean_nonparametric_coverage=feature_gate.quality_risk_binding_coverage,
        coverage_floor_slack=feature_gate.quality_risk_binding_coverage_floor_slack,
        mean_nonparametric_rmse=feature_gate.quality_risk_binding_rmse,
        mean_nonparametric_average_standard_error=(
            feature_gate.quality_risk_binding_average_standard_error
        ),
        mean_nonparametric_interval_length=(
            feature_gate.quality_risk_binding_interval_length
        ),
        rmse_to_standard_error_ratio=(
            feature_gate.quality_risk_binding_rmse_to_standard_error_ratio
        ),
        standard_error_reserve=feature_gate.quality_risk_binding_standard_error_reserve,
        interval_length_to_rmse_ratio=(
            feature_gate.quality_risk_binding_interval_length_to_rmse_ratio
        ),
    )
    best_summary = Phase7MonteCarloWideningPolicyQualityRiskDesignSummary(
        *_parse_design_label(feature_gate.quality_risk_best_design_label),
        mean_nonparametric_coverage=feature_gate.quality_risk_best_coverage,
        coverage_floor_slack=getattr(
            feature_gate,
            "quality_risk_best_coverage_floor_slack",
            float(feature_gate.quality_risk_best_coverage)
            - _QUALITY_RISK_CANONICAL_FLOOR,
        ),
        mean_nonparametric_rmse=feature_gate.quality_risk_best_rmse,
        mean_nonparametric_average_standard_error=(
            feature_gate.quality_risk_best_average_standard_error
        ),
        mean_nonparametric_interval_length=(
            feature_gate.quality_risk_best_interval_length
        ),
        rmse_to_standard_error_ratio=(
            feature_gate.quality_risk_best_rmse_to_standard_error_ratio
        ),
        standard_error_reserve=feature_gate.quality_risk_best_standard_error_reserve,
        interval_length_to_rmse_ratio=(
            feature_gate.quality_risk_best_interval_length_to_rmse_ratio
        ),
    )
    expected_candidate = build_quality_risk_average_standard_error_calibration_candidate(
        binding_summary=binding_summary,
        best_summary=best_summary,
        dominant_calibration_target=(
            feature_gate.dominant_quality_risk_calibration_target
        ),
    )
    if candidate != expected_candidate:
        raise ValueError(
            "release matrix runner requires average-SE calibration candidate "
            "to match the live quality metrics"
        )
    return candidate


def _quality_risk_average_standard_error_calibration_candidate_evidence(
    feature_gate,
    candidate: dict[str, object],
) -> tuple[str, ...]:
    evidence = tuple(
        str(item).strip()
        for item in getattr(
            feature_gate,
            "quality_risk_average_standard_error_calibration_candidate_evidence",
            (),
        )
    )
    expected_evidence = (
        build_quality_risk_average_standard_error_calibration_candidate_evidence(
            candidate
        )
    )
    if evidence != expected_evidence:
        raise ValueError(
            "release matrix runner requires average-SE calibration candidate "
            "evidence to match the live candidate metrics"
        )
    return evidence


def _quality_risk_positive_headroom_metadata(
    candidate: dict[str, object],
) -> dict[str, object]:
    target = candidate.get("positive_headroom_admission_target")
    if not isinstance(target, dict):
        raise ValueError(
            "release matrix runner requires average-SE calibration candidate "
            "to expose positive-headroom admission target metadata"
        )
    estimator_requirement = (
        build_quality_risk_positive_headroom_estimator_evidence_requirement(
            candidate
        )
    )
    return {
        "quality_risk_positive_headroom_admission_target": dict(target),
        "quality_risk_positive_headroom_target_average_standard_error": float(
            target["minimum_admissible_average_standard_error"]
        ),
        "quality_risk_positive_headroom_target_interval_length": float(
            target["minimum_admissible_interval_length"]
        ),
        "quality_risk_positive_headroom_required_average_standard_error_lift": float(
            target["required_average_standard_error_lift"]
        ),
        "quality_risk_positive_headroom_required_interval_length_lift": float(
            target["required_interval_length_lift"]
        ),
        "quality_risk_positive_headroom_remaining_average_standard_error_gap": float(
            estimator_requirement["remaining_average_standard_error_gap"]
        ),
        "quality_risk_positive_headroom_remaining_interval_length_gap": float(
            estimator_requirement["remaining_interval_length_gap"]
        ),
        "quality_risk_positive_headroom_maximum_remaining_gap_fraction": float(
            max(
                estimator_requirement["remaining_average_standard_error_gap"]
                / estimator_requirement["target_average_standard_error"],
                estimator_requirement["remaining_interval_length_gap"]
                / estimator_requirement["target_interval_length"],
            )
        ),
        "positive_headroom_maximum_remaining_gap_fraction": float(
            max(
                estimator_requirement["remaining_average_standard_error_gap"]
                / estimator_requirement["target_average_standard_error"],
                estimator_requirement["remaining_interval_length_gap"]
                / estimator_requirement["target_interval_length"],
            )
        ),
        "quality_risk_positive_headroom_margin": float(
            target["minimum_positive_headroom_margin"]
        ),
        "quality_risk_positive_headroom_admission_status": str(
            target["admission_status"]
        ),
        "quality_risk_positive_headroom_estimator_evidence_requirement": dict(
            estimator_requirement
        ),
    }


def _quality_risk_positive_headroom_verdict_metadata(
    feature_gate,
    candidate: dict[str, object],
) -> dict[str, object]:
    estimator_requirement = (
        build_quality_risk_positive_headroom_estimator_evidence_requirement(
            candidate
        )
    )
    expected_verdict = build_quality_risk_positive_headroom_estimator_evidence_verdict(
        estimator_requirement,
        observed_average_standard_error=(
            feature_gate.quality_risk_binding_average_standard_error
        ),
        observed_interval_length=feature_gate.quality_risk_binding_interval_length,
    )
    actual_verdict = dict(
        feature_gate.quality_risk_positive_headroom_estimator_evidence_verdict
    )
    actual_alias = dict(feature_gate.positive_headroom_estimator_evidence_verdict)
    if actual_verdict != expected_verdict or actual_alias != expected_verdict:
        raise ValueError(
            "release matrix runner requires positive-headroom estimator verdict "
            "to match the live observed average-SE and interval-length evidence"
        )

    expected_evidence = tuple(
        (
            f"positive_headroom_estimator_evidence_status={expected_verdict['status']}",
            "positive_headroom_estimator_evidence_source="
            f"{expected_verdict['observed_evidence_source']} "
            f"status={expected_verdict['observed_evidence_source_status']}",
            "positive_headroom_estimator_average_se "
            f"{float(expected_verdict['observed_average_standard_error']):.3f}->"
            f"{float(expected_verdict['target_average_standard_error']):.3f} "
            f"margin={float(expected_verdict['average_standard_error_margin']):+.3f} "
            "remaining_fraction="
            f"{float(expected_verdict['remaining_average_standard_error_gap_fraction']):.3f}",
            "positive_headroom_estimator_interval_length "
            f"{float(expected_verdict['observed_interval_length']):.3f}->"
            f"{float(expected_verdict['target_interval_length']):.3f} "
            f"margin={float(expected_verdict['interval_length_margin']):+.3f} "
            "remaining_fraction="
            f"{float(expected_verdict['remaining_interval_length_gap_fraction']):.3f}",
            "positive_headroom_estimator_interval_scale "
            f"{expected_verdict['interval_scale_status']} "
            f"gap={float(expected_verdict['interval_to_standard_error_gap']):+.3e}",
            "positive_headroom_estimator_next_evidence="
            f"{expected_verdict['required_next_evidence']}",
            "positive_headroom_feature_gate_rerun_admission="
            f"{expected_verdict['feature_gate_rerun_admission_status']} "
            f"admissible={expected_verdict['admissible_for_feature_gate_rerun']}",
        )
    )
    actual_evidence = tuple(
        str(item).strip()
        for item in (
            feature_gate.quality_risk_positive_headroom_estimator_evidence_verdict_evidence
        )
    )
    actual_evidence_alias = tuple(
        str(item).strip()
        for item in feature_gate.positive_headroom_estimator_evidence_verdict_evidence
    )
    if actual_evidence != expected_evidence or actual_evidence_alias != expected_evidence:
        raise ValueError(
            "release matrix runner requires positive-headroom estimator verdict "
            "evidence to match the live observed average-SE and interval-length evidence"
        )

    return {
        "quality_risk_positive_headroom_estimator_evidence_verdict": dict(
            expected_verdict
        ),
        "positive_headroom_estimator_evidence_verdict": dict(expected_verdict),
        "quality_risk_positive_headroom_observed_average_standard_error": float(
            expected_verdict["observed_average_standard_error"]
        ),
        "positive_headroom_observed_average_standard_error": float(
            expected_verdict["observed_average_standard_error"]
        ),
        "quality_risk_positive_headroom_current_average_standard_error": float(
            expected_verdict["observed_average_standard_error"]
        ),
        "positive_headroom_current_average_standard_error": float(
            expected_verdict["observed_average_standard_error"]
        ),
        "quality_risk_positive_headroom_observed_interval_length": float(
            expected_verdict["observed_interval_length"]
        ),
        "positive_headroom_observed_interval_length": float(
            expected_verdict["observed_interval_length"]
        ),
        "quality_risk_positive_headroom_current_interval_length": float(
            expected_verdict["observed_interval_length"]
        ),
        "positive_headroom_current_interval_length": float(
            expected_verdict["observed_interval_length"]
        ),
        "quality_risk_positive_headroom_admissible_for_feature_gate_rerun": bool(
            expected_verdict["admissible_for_feature_gate_rerun"]
        ),
        "positive_headroom_admissible_for_feature_gate_rerun": bool(
            expected_verdict["admissible_for_feature_gate_rerun"]
        ),
        "quality_risk_positive_headroom_feature_gate_rerun_admission_status": str(
            expected_verdict["feature_gate_rerun_admission_status"]
        ),
        "positive_headroom_feature_gate_rerun_admission_status": str(
            expected_verdict["feature_gate_rerun_admission_status"]
        ),
        "quality_risk_positive_headroom_estimator_evidence_verdict_evidence": list(
            expected_evidence
        ),
        "positive_headroom_estimator_evidence_verdict_evidence": list(
            expected_evidence
        ),
    }


@dataclass(frozen=True, slots=True)
class Phase7ReleaseMatrixGateRow:
    gate_name: str
    helper: str
    note: str
    status: str
    blocker: str | None
    release_blocking: bool
    evidence: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_name", str(self.gate_name).strip())
        object.__setattr__(self, "helper", str(self.helper).strip())
        object.__setattr__(self, "note", str(self.note).strip())
        object.__setattr__(self, "status", str(self.status).strip())
        object.__setattr__(self, "blocker", _clean_blocker(self.blocker))
        object.__setattr__(
            self,
            "release_blocking",
            _require_bool(self.release_blocking, name="release_blocking"),
        )
        object.__setattr__(
            self,
            "evidence",
            tuple(str(item).strip() for item in self.evidence),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def blocker_reason(self) -> str | None:
        return self.blocker

    @property
    def reason(self) -> str | None:
        return self.blocker_reason

    @property
    def row_id(self) -> str:
        return self.gate_name

    @property
    def name(self) -> str:
        return self.gate_name

    @property
    def ready(self) -> bool:
        return self.status == "ready" and not self.release_blocking

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_name": self.gate_name,
            "row_id": self.row_id,
            "name": self.name,
            "helper": self.helper,
            "note": self.note,
            "status": self.status,
            "ready": self.ready,
            "blocker": self.blocker,
            "blocker_reason": self.blocker_reason,
            "reason": self.reason,
            "release_blocking": self.release_blocking,
            "evidence": list(self.evidence),
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class Phase7ReleaseMatrixRunnerReport:
    stage_label: str
    route_label: str
    repo_root: str
    current_release_label: str
    current_version: str
    matrix_rows: tuple[Phase7ReleaseMatrixGateRow, ...]
    blocking_gate_names: tuple[str, ...]
    release_ready: bool
    canonical_gate_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_label", str(self.stage_label).strip())
        object.__setattr__(self, "route_label", str(self.route_label).strip())
        object.__setattr__(self, "repo_root", str(self.repo_root).strip())
        object.__setattr__(
            self,
            "current_release_label",
            str(self.current_release_label).strip(),
        )
        object.__setattr__(self, "current_version", str(self.current_version).strip())
        object.__setattr__(self, "matrix_rows", tuple(self.matrix_rows))
        object.__setattr__(
            self,
            "blocking_gate_names",
            tuple(str(name).strip() for name in self.blocking_gate_names),
        )
        object.__setattr__(
            self,
            "release_ready",
            _require_bool(self.release_ready, name="release_ready"),
        )
        object.__setattr__(
            self,
            "canonical_gate_digest",
            tuple(str(line).rstrip() for line in self.canonical_gate_digest),
        )
        expected_blocking_gate_names = tuple(
            row.gate_name for row in self.matrix_rows if row.release_blocking
        )
        if self.blocking_gate_names != expected_blocking_gate_names:
            raise ValueError(
                "release matrix runner requires blocking gate names to match "
                "release-blocking rows"
            )
        if self.release_ready != (not self.blocking_gate_names):
            raise ValueError(
                "release matrix runner requires release_ready to match blocking "
                "gate names"
            )

    @property
    def helper(self) -> str:
        return "run_phase7_release_matrix_runner(...)"

    @property
    def note(self) -> str:
        return "Docs/research/phase7_release_matrix_runner.md"

    @property
    def status(self) -> str:
        return "ready" if self.release_ready else "blocked"

    @property
    def matrix_status(self) -> str:
        return self.status

    @property
    def blocked_gates(self) -> tuple[str, ...]:
        return self.blocking_gate_names

    @property
    def rows(self) -> tuple[Phase7ReleaseMatrixGateRow, ...]:
        return self.matrix_rows

    @property
    def blocking(self) -> tuple[str, ...]:
        return self.blocking_gate_names

    @property
    def blocking_rows(self) -> tuple[Phase7ReleaseMatrixGateRow, ...]:
        return tuple(row for row in self.matrix_rows if row.release_blocking)

    @property
    def blocking_row_ids(self) -> tuple[str, ...]:
        return tuple(row.gate_name for row in self.blocking_rows)

    @property
    def rows_by_gate_name(self) -> dict[str, Phase7ReleaseMatrixGateRow]:
        return {row.gate_name: row for row in self.matrix_rows}

    def gate_row(self, gate_name: str) -> Phase7ReleaseMatrixGateRow:
        target = str(gate_name).strip()
        rows_by_gate_name = self.rows_by_gate_name
        if target in rows_by_gate_name:
            return rows_by_gate_name[target]
        raise KeyError(f"release matrix gate row not present: {target!r}")

    def row_by_name(self, gate_name: str) -> Phase7ReleaseMatrixGateRow:
        return self.gate_row(gate_name)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "route_label": self.route_label,
            "helper": self.helper,
            "note": self.note,
            "status": self.status,
            "matrix_status": self.matrix_status,
            "repo_root": self.repo_root,
            "current_release_label": self.current_release_label,
            "current_version": self.current_version,
            "rows": [row.to_dict() for row in self.rows],
            "matrix_rows": [row.to_dict() for row in self.matrix_rows],
            "rows_by_gate_name": {
                gate_name: row.to_dict()
                for gate_name, row in self.rows_by_gate_name.items()
            },
            "blocking_gate_names": list(self.blocking_gate_names),
            "blocking": list(self.blocking),
            "blocked_gates": list(self.blocked_gates),
            "blocking_row_ids": list(self.blocking_row_ids),
            "blocking_rows": [row.to_dict() for row in self.blocking_rows],
            "release_ready": self.release_ready,
            "canonical_gate_digest": list(self.canonical_gate_digest),
        }

    def to_markdown(self, *, evidence_items: int = 5) -> str:
        header = "| Gate | Status | Ready | Blocking | Blocker | Evidence |"
        separator = "| --- | --- | --- | --- | --- | --- |"
        lines = [header, separator]
        for row in self.matrix_rows:
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_cell(row.gate_name),
                        _markdown_cell(row.status),
                        _markdown_yes_no(row.ready),
                        _markdown_yes_no(row.release_blocking),
                        _markdown_cell(row.blocker or "-"),
                        _summarize_markdown_evidence(
                            row.evidence,
                            evidence_items=evidence_items,
                        ),
                    )
                )
                + " |"
            )
        return "\n".join(lines)


def run_phase7_release_matrix_runner(
    repo_root: str | Path,
    *,
    feature_gate=None,
) -> Phase7ReleaseMatrixRunnerReport:
    root = _coerce_repo_root(repo_root)
    empirical_audit = audit_section6_assets(root)
    section6_gate = audit_section6_provenance_gate(root)
    outer_gate_status = _outer_inference_parity_status(root)
    if feature_gate is None:
        feature_gate = run_phase7_monte_carlo_feature_completion_gate(root)
    monte_carlo_evidence = _feature_gate_evidence_tuple(feature_gate)
    quality_clearance_conditions = _quality_risk_clearance_conditions(feature_gate)
    quality_calibration_targets = tuple(
        dict(target)
        for target in getattr(feature_gate, "quality_risk_calibration_targets", ())
    )
    dominant_quality_calibration_target = dict(
        getattr(feature_gate, "dominant_quality_risk_calibration_target", {})
    )
    quality_method_diagnosis = dict(
        getattr(feature_gate, "quality_risk_method_diagnosis", {})
    )
    quality_action_contract = dict(
        getattr(feature_gate, "quality_risk_action_contract", {})
    )
    quality_average_se_calibration_candidate = (
        _quality_risk_average_standard_error_calibration_candidate(feature_gate)
    )
    quality_average_se_calibration_candidate_evidence = (
        _quality_risk_average_standard_error_calibration_candidate_evidence(
            feature_gate,
            quality_average_se_calibration_candidate,
        )
    )
    quality_positive_headroom_metadata = _quality_risk_positive_headroom_metadata(
        quality_average_se_calibration_candidate
    )
    _require_positive_headroom_verdict_attrs(feature_gate)
    quality_positive_headroom_verdict_metadata = (
        _quality_risk_positive_headroom_verdict_metadata(
            feature_gate,
            quality_average_se_calibration_candidate,
        )
    )
    quality_raw_metric_metadata = _quality_risk_raw_metric_metadata(feature_gate)
    release_gate = run_phase7_release_maturity_gate(
        root,
        feature_gate=feature_gate,
    )

    rows = (
        Phase7ReleaseMatrixGateRow(
            gate_name="empirical_lane_ready",
            helper="audit_section6_assets(...)",
            note="hddid-py/docs/validation-lanes.md",
            status=empirical_audit.status,
            blocker=empirical_audit.blocker_reason,
            release_blocking=empirical_audit.status != "ready",
            evidence=(
                "hddid-py/data/section6_county_panel.tsv",
                "hddid-py/data/section6_manifest.json",
                "hddid-py/src/hddid/section6_loader.py",
            ),
        ),
        Phase7ReleaseMatrixGateRow(
            gate_name="section6_provenance_gate_ready",
            helper="audit_section6_provenance_gate(...)",
            note="Docs/research/phase7_section6_provenance_gate.md",
            status=section6_gate.status,
            blocker=section6_gate.blocker_reason,
            release_blocking=section6_gate.status != "ready",
            evidence=(
                "repository-local Section 6 manifest",
                "loader object contract",
                "paper-backed empirical schema",
            ),
        ),
        Phase7ReleaseMatrixGateRow(
            gate_name="outer_inference_parity_ready",
            helper="run_phase7_outer_inference_paper_backed_replacement_path(...)",
            note="Docs/research/phase7_outer_inference_paper_backed_replacement_path.md",
            status=outer_gate_status,
            blocker=None,
            release_blocking=False,
            evidence=(
                "aggregated covariance-process objects",
                "covariance_at_grid",
                "uniform_critical_value",
            ),
        ),
        Phase7ReleaseMatrixGateRow(
            gate_name="monte_carlo_validation_ready",
            helper="run_phase7_monte_carlo_feature_completion_gate(...)",
            note="Docs/research/phase7_monte_carlo_feature_completion_gate.md",
            status=feature_gate.monte_carlo_validation_ready_status,
            blocker=feature_gate.monte_carlo_validation_ready_blocker,
            release_blocking=(
                feature_gate.monte_carlo_validation_ready_status != "ready"
            ),
            evidence=monte_carlo_evidence,
            metadata={
                "quality_risk_binding_design": list(
                    feature_gate.quality_risk_binding_design
                ),
                "quality_risk_best_design": list(feature_gate.quality_risk_best_design),
                "quality_risk_binding_design_label": (
                    feature_gate.quality_risk_binding_design_label
                ),
                "quality_risk_best_design_label": (
                    feature_gate.quality_risk_best_design_label
                ),
                "quality_risk_binding_driver": (
                    feature_gate.quality_risk_binding_driver
                ),
                "quality_risk_blocker_reason": (
                    feature_gate.quality_risk_blocker_reason
                ),
                "runtime_evidence_admission_status": (
                    feature_gate.runtime_evidence_admission_status
                ),
                "runtime_evidence_admission_candidate_status": (
                    feature_gate.runtime_evidence_admission_candidate_status
                ),
                "runtime_evidence_admission_candidate_success": (
                    feature_gate.runtime_evidence_admission_candidate_success
                ),
                "runtime_evidence_admission_candidate_nonparametric_coverage": (
                    feature_gate.runtime_evidence_admission_candidate_nonparametric_coverage
                ),
                "runtime_evidence_admission_candidate_typed_invalidity_counts": dict(
                    feature_gate.runtime_evidence_admission_candidate_typed_invalidity_counts
                ),
                "runtime_evidence_admission_candidate_typed_invalidity_examples": dict(
                    feature_gate.runtime_evidence_admission_candidate_typed_invalidity_examples
                ),
                "runtime_evidence_admission_candidate_typed_invalidity_labels": list(
                    feature_gate.runtime_evidence_admission_candidate_typed_invalidity_labels
                ),
                "runtime_evidence_admission_candidate_primary_invalidity_name": (
                    feature_gate.runtime_evidence_admission_candidate_primary_invalidity_name
                ),
                "runtime_evidence_admission_candidate_primary_invalidity_matrix_name": (
                    feature_gate.runtime_evidence_admission_candidate_primary_invalidity_matrix_name
                ),
                "runtime_evidence_admission_candidate_primary_invalidity_min_eigenvalue": (
                    feature_gate.runtime_evidence_admission_candidate_primary_invalidity_min_eigenvalue
                ),
                "runtime_evidence_admission_candidate_primary_invalidity_replication_seed": (
                    feature_gate.runtime_evidence_admission_candidate_primary_invalidity_replication_seed
                ),
                "quality_risk_clearance_conditions": list(quality_clearance_conditions),
                "quality_risk_calibration_targets": list(
                    quality_calibration_targets
                ),
                "dominant_quality_risk_calibration_target": (
                    dominant_quality_calibration_target
                ),
                "quality_risk_method_diagnosis": quality_method_diagnosis,
                "quality_risk_method_diagnosis_evidence": list(
                    feature_gate.quality_risk_method_diagnosis_evidence
                ),
                "quality_risk_action_contract": quality_action_contract,
                "quality_risk_action_evidence": list(
                    feature_gate.quality_risk_action_evidence
                ),
                "quality_risk_average_standard_error_calibration_candidate": (
                    quality_average_se_calibration_candidate
                ),
                "quality_risk_average_standard_error_calibration_candidate_evidence": (
                    list(quality_average_se_calibration_candidate_evidence)
                ),
                **quality_positive_headroom_metadata,
                **quality_positive_headroom_verdict_metadata,
                "quality_risk_raw_metrics": quality_raw_metric_metadata,
                "quality_risk_binding_coverage": quality_raw_metric_metadata[
                    "binding"
                ]["coverage"],
                "quality_risk_binding_rmse": quality_raw_metric_metadata["binding"][
                    "rmse"
                ],
                "quality_risk_binding_average_standard_error": (
                    quality_raw_metric_metadata["binding"]["average_standard_error"]
                ),
                "quality_risk_binding_interval_length": quality_raw_metric_metadata[
                    "binding"
                ]["interval_length"],
                "quality_risk_binding_standard_error_reserve": (
                    quality_raw_metric_metadata["binding"]["standard_error_reserve"]
                ),
                "quality_risk_rmse_outpaces_average_se_gap": (
                    feature_gate.quality_risk_rmse_outpaces_average_se_gap
                ),
                "rmse_outpaces_average_se_gap": (
                    feature_gate.rmse_outpaces_average_se_gap
                ),
                "quality_risk_best_coverage": quality_raw_metric_metadata["best"][
                    "coverage"
                ],
                "quality_risk_best_rmse": quality_raw_metric_metadata["best"][
                    "rmse"
                ],
                "quality_risk_best_average_standard_error": (
                    quality_raw_metric_metadata["best"]["average_standard_error"]
                ),
                "quality_risk_best_interval_length": quality_raw_metric_metadata[
                    "best"
                ]["interval_length"],
                "quality_risk_best_standard_error_reserve": (
                    quality_raw_metric_metadata["best"]["standard_error_reserve"]
                ),
                "quality_risk_standard_error_reserve_gap": (
                    feature_gate.quality_risk_standard_error_reserve_gap
                ),
            },
        ),
        Phase7ReleaseMatrixGateRow(
            gate_name="release_maturity_ready",
            helper="run_phase7_release_maturity_gate(...)",
            note="Docs/research/phase7_release_maturity_gate.md",
            status=release_gate.gate_status,
            blocker=release_gate.blocker,
            release_blocking=not release_gate.release_ready,
            evidence=_release_maturity_evidence_tuple(
                release_gate,
                feature_gate,
                monte_carlo_evidence,
            ),
        ),
    )
    blocking_gate_names = tuple(row.gate_name for row in rows if row.release_blocking)
    digest = (
        "- release matrix runner: `run_phase7_release_matrix_runner(...)`",
        "- deterministic gate rows: `empirical_lane_ready`, "
        "`section6_provenance_gate_ready`, `outer_inference_parity_ready`, "
        "`monte_carlo_validation_ready`, and `release_maturity_ready`",
        "- release maturity row mirrors `run_phase7_release_maturity_gate(...)`: "
        f"status `{release_gate.gate_status}`; blocker "
        f"`{release_gate.blocker or 'none'}`",
        "- release promotion is allowed only when no release-matrix row remains "
        "`release_blocking` on the same repo-side read",
        "- report aliases `blocking` and `blocked_gates` to the same blocking tuple",
    )
    return Phase7ReleaseMatrixRunnerReport(
        stage_label="phase7-release-matrix-runner",
        route_label="release-matrix-runner",
        repo_root=str(root),
        current_release_label=release_gate.current_release_label,
        current_version=release_gate.current_version,
        matrix_rows=rows,
        blocking_gate_names=blocking_gate_names,
        release_ready=not blocking_gate_names,
        canonical_gate_digest=digest,
    )
