from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import yaml

from .monte_carlo_widening_policy_floor_slack_probe import (
    build_phase7_monte_carlo_widening_policy_floor_slack_probe_report,
)
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)
from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy
from .validation import (
    MonteCarloRuntimeProbeDesignSummary,
    MonteCarloRuntimeProbeReport,
    default_phase7_runtime_probe_designs,
)
from .validation import run_phase7_monte_carlo_runtime_probe

_DEFAULT_RUNTIME_PROBE_RUNNER = run_phase7_monte_carlo_runtime_probe


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{dgp_name}/{n_obs}/{p}"


def _format_signed(value: float) -> str:
    return f"{value:+.3f}"


def _format_unsigned(value: float) -> str:
    return f"{value:.3f}"


def _interval_scale_status_for_summaries(
    binding_summary: "Phase7MonteCarloWideningPolicyQualityRiskDesignSummary",
    best_summary: "Phase7MonteCarloWideningPolicyQualityRiskDesignSummary",
) -> str:
    max_abs_gap = max(
        abs(binding_summary.interval_length_to_standard_error_gap),
        abs(best_summary.interval_length_to_standard_error_gap),
    )
    if max_abs_gap <= _INTERVAL_SCALE_LOCK_TOLERANCE:
        return "paper-90pct-pointwise-interval-scale-locked"
    return "paper-90pct-pointwise-interval-scale-needs-review"


def _interval_scale_digest_line(
    binding_summary: "Phase7MonteCarloWideningPolicyQualityRiskDesignSummary",
    best_summary: "Phase7MonteCarloWideningPolicyQualityRiskDesignSummary",
) -> str:
    return (
        "- interval scale contract: "
        f"`paper_90pct_pointwise_interval_to_se="
        f"{_format_unsigned(_PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER)}`; "
        f"`binding_interval_to_se={_format_design_key(*binding_summary.design_key)} "
        f"{_format_unsigned(binding_summary.interval_length_to_standard_error_ratio)} "
        f"gap={binding_summary.interval_length_to_standard_error_gap:+.3e}`; "
        f"`best_interval_to_se={_format_design_key(*best_summary.design_key)} "
        f"{_format_unsigned(best_summary.interval_length_to_standard_error_ratio)} "
        f"gap={best_summary.interval_length_to_standard_error_gap:+.3e}`; "
        f"`interval_scale_status="
        f"{_interval_scale_status_for_summaries(binding_summary, best_summary)}`"
    )


def _calibration_target_digest_line(
    binding_summary: "Phase7MonteCarloWideningPolicyQualityRiskDesignSummary",
    best_summary: "Phase7MonteCarloWideningPolicyQualityRiskDesignSummary",
) -> str:
    dominant = dominant_quality_risk_calibration_target(
        build_quality_risk_calibration_targets(
            binding_rmse=binding_summary.mean_nonparametric_rmse,
            binding_average_standard_error=(
                binding_summary.mean_nonparametric_average_standard_error
            ),
            binding_rmse_to_standard_error_ratio=(
                binding_summary.rmse_to_standard_error_ratio
            ),
            binding_standard_error_reserve=binding_summary.standard_error_reserve,
            binding_interval_length_to_rmse_ratio=(
                binding_summary.interval_length_to_rmse_ratio
            ),
            best_rmse_to_standard_error_ratio=best_summary.rmse_to_standard_error_ratio,
            best_standard_error_reserve=best_summary.standard_error_reserve,
            best_interval_length_to_rmse_ratio=best_summary.interval_length_to_rmse_ratio,
        )
    )
    return (
        "- calibration target: "
        f"`dominant_condition={dominant['condition']}`; "
        "`required_average_se_lift="
        f"{float(dominant['required_average_standard_error_lift']):.3f}`; "
        "`current_interval_length="
        f"{float(dominant['current_interval_length']):.3f}`; "
        "`target_average_se="
        f"{float(dominant['target_average_standard_error']):.3f}`; "
        "`target_interval_length="
        f"{float(dominant['target_interval_length']):.3f}`; "
        "`required_interval_length_lift="
        f"{float(dominant['required_interval_length_lift']):.3f}`"
    )


def _positive_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None:
        raise ValueError(
            "quality risk probe requires nonparametric rmse, average standard error, and interval length"
        )
    if isinstance(numerator, (bool, np.bool_)) or isinstance(
        denominator, (bool, np.bool_)
    ):
        raise ValueError(
            "quality risk probe requires numeric nonparametric rmse, average standard error, and interval length, not boolean"
        )
    if isinstance(numerator, (str, np.str_)) or isinstance(denominator, (str, np.str_)):
        raise ValueError(
            "quality risk probe requires numeric nonparametric rmse, average standard error, and interval length, not string"
        )
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if (
        not math.isfinite(numerator_value)
        or not math.isfinite(denominator_value)
        or numerator_value <= 0.0
        or denominator_value <= 0.0
    ):
        raise ValueError(
            "quality risk probe requires positive finite nonparametric rmse, average standard error, and interval length"
        )
    return float(numerator_value / denominator_value)


def _validated_coverage(value: float, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"quality risk probe requires numeric {label}, not boolean")
    if isinstance(value, (str, np.str_)):
        raise ValueError(f"quality risk probe requires numeric {label}, not string")
    coverage = float(value)
    if not math.isfinite(coverage) or coverage < 0.0 or coverage > 1.0:
        raise ValueError(f"quality risk probe requires {label} in [0, 1]")
    return coverage


def _validated_finite(value: float, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"quality risk probe requires numeric {label}, not boolean")
    if isinstance(value, (str, np.str_)):
        raise ValueError(f"quality risk probe requires numeric {label}, not string")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"quality risk probe requires finite {label}")
    return number


def _validated_positive_integer(value: int, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"quality risk probe requires {label} to be a positive integer, not boolean"
        )
    if not isinstance(value, Integral):
        raise ValueError(f"quality risk probe requires {label} to be an integer")
    number = int(value)
    if number <= 0:
        raise ValueError(f"quality risk probe requires {label} to be positive")
    return number


def _is_quality_metric_validation_error(exc: ValueError) -> bool:
    return str(exc).startswith("quality risk probe requires ")


def _is_repo_side_quality_risk_fallback_error(exc: ValueError) -> bool:
    message = str(exc)
    return message in {
        "live quality-risk probe unavailable",
        "runtime probe has no bounded widening slice summaries",
    }


def _require_close_metric(
    actual: float,
    expected: float,
    *,
    label: str,
    abs_tol: float = 1e-3,
) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=abs_tol):
        raise ValueError(
            f"quality risk probe requires {label} to match design summaries"
        )


def _require_close_raw_metric(
    actual: float,
    expected: float,
    *,
    label: str,
    abs_tol: float = 1e-3,
) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=abs_tol):
        raise ValueError(f"quality risk probe requires {label} to match raw metrics")


def _policy_min_nonparametric_coverage(
    policy_digest: tuple[str, ...],
) -> float | None:
    prefix = "min_nonparametric_coverage="
    for item in policy_digest:
        if item.startswith(prefix):
            return float(item.removeprefix(prefix))
    return None


def _quality_risk_clearance_conditions(
    *,
    binding_driver: str,
    binding_coverage_floor_slack: float,
    binding_standard_error_reserve: float,
    rmse_to_standard_error_ratio_gap: float,
    standard_error_reserve_gap: float,
    interval_length_to_rmse_gap: float,
) -> tuple[dict[str, object], ...]:
    coverage_floor_deficit = max(-binding_coverage_floor_slack, 0.0)
    nonnegative_reserve_deficit = max(-binding_standard_error_reserve, 0.0)
    rmse_to_se_deficit = max(rmse_to_standard_error_ratio_gap, 0.0)
    se_reserve_deficit = max(standard_error_reserve_gap, 0.0)
    interval_to_rmse_deficit = max(interval_length_to_rmse_gap, 0.0)
    overall_status = (
        "cleared" if binding_driver == "quality-risk-cleared" else "blocked"
    )
    return (
        {
            "condition": "coverage_floor",
            "status": "cleared" if coverage_floor_deficit == 0.0 else "open",
            "deficit": coverage_floor_deficit,
            "margin": binding_coverage_floor_slack,
        },
        {
            "condition": "nonnegative_se_reserve",
            "status": "cleared" if nonnegative_reserve_deficit == 0.0 else "open",
            "deficit": nonnegative_reserve_deficit,
            "margin": binding_standard_error_reserve,
        },
        {
            "condition": "rmse_to_se_not_trailing",
            "status": "cleared" if rmse_to_se_deficit == 0.0 else "open",
            "deficit": rmse_to_se_deficit,
        },
        {
            "condition": "se_reserve_not_trailing",
            "status": "cleared" if se_reserve_deficit == 0.0 else "open",
            "deficit": se_reserve_deficit,
        },
        {
            "condition": "interval_to_rmse_not_trailing",
            "status": "cleared" if interval_to_rmse_deficit == 0.0 else "open",
            "deficit": interval_to_rmse_deficit,
        },
        {
            "condition": "overall",
            "status": overall_status,
            "driver": binding_driver,
        },
    )


def build_quality_risk_calibration_targets(
    *,
    binding_rmse: float,
    binding_average_standard_error: float,
    binding_rmse_to_standard_error_ratio: float,
    binding_standard_error_reserve: float,
    binding_interval_length_to_rmse_ratio: float,
    best_rmse_to_standard_error_ratio: float,
    best_standard_error_reserve: float,
    best_interval_length_to_rmse_ratio: float,
) -> tuple[dict[str, object], ...]:
    binding_rmse = _validated_finite(binding_rmse, label="binding RMSE")
    current_se = _validated_finite(
        binding_average_standard_error,
        label="binding average standard error",
    )
    binding_rmse_to_standard_error_ratio = _positive_ratio(
        binding_rmse_to_standard_error_ratio,
        1.0,
    )
    best_rmse_to_standard_error_ratio = _positive_ratio(
        best_rmse_to_standard_error_ratio,
        1.0,
    )
    binding_standard_error_reserve = _validated_finite(
        binding_standard_error_reserve,
        label="binding SE-RMSE reserve",
    )
    best_standard_error_reserve = _validated_finite(
        best_standard_error_reserve,
        label="best SE-RMSE reserve",
    )
    binding_interval_length_to_rmse_ratio = _positive_ratio(
        binding_interval_length_to_rmse_ratio,
        1.0,
    )
    best_interval_length_to_rmse_ratio = _positive_ratio(
        best_interval_length_to_rmse_ratio,
        1.0,
    )

    targets = (
        (
            "rmse_to_se_not_trailing",
            binding_rmse_to_standard_error_ratio
            > best_rmse_to_standard_error_ratio,
            binding_rmse / best_rmse_to_standard_error_ratio,
        ),
        (
            "se_reserve_not_trailing",
            binding_standard_error_reserve < best_standard_error_reserve,
            binding_rmse + best_standard_error_reserve,
        ),
        (
            "interval_to_rmse_not_trailing",
            binding_interval_length_to_rmse_ratio
            < best_interval_length_to_rmse_ratio,
            (
                best_interval_length_to_rmse_ratio
                * binding_rmse
                / _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
            ),
        ),
    )
    return tuple(
        {
            "condition": condition,
            "status": "open" if is_open else "cleared",
            "current_average_standard_error": current_se,
            "target_average_standard_error": (
                max(target_average_standard_error, current_se)
                if is_open
                else current_se
            ),
            "required_average_standard_error_lift": (
                max(target_average_standard_error - current_se, 0.0)
                if is_open
                else 0.0
            ),
            "current_interval_length": (
                current_se * _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
            ),
            "target_interval_length": (
                max(target_average_standard_error, current_se)
                if is_open
                else current_se
            )
            * _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER,
            "required_interval_length_lift": (
                (
                    max(target_average_standard_error, current_se) - current_se
                )
                * _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
                if is_open
                else 0.0
            ),
        }
        for condition, is_open, target_average_standard_error in targets
    )


def dominant_quality_risk_calibration_target(
    targets: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if not targets:
        return {
            "condition": "quality-risk-cleared",
            "status": "cleared",
            "current_average_standard_error": 0.0,
            "target_average_standard_error": 0.0,
            "required_average_standard_error_lift": 0.0,
            "current_interval_length": 0.0,
            "target_interval_length": 0.0,
            "required_interval_length_lift": 0.0,
        }
    dominant = max(
        targets,
        key=lambda target: _calibration_target_value(
            target,
            "required_average_standard_error_lift",
            label="dominant required average standard error lift",
        ),
    )
    return _validated_dominant_calibration_target(dominant)


def _calibration_target_value(
    target: dict[str, object],
    key: str,
    *,
    label: str,
    default: float = 0.0,
) -> float:
    value = _validated_finite(target.get(key, default), label=label)
    if value < 0.0:
        raise ValueError(f"quality risk probe requires non-negative {label}")
    return value


def _validated_dominant_calibration_target(
    target: dict[str, object],
) -> dict[str, object]:
    normalized = dict(target)
    for key, label in (
        (
            "current_average_standard_error",
            "dominant current average standard error",
        ),
        (
            "target_average_standard_error",
            "dominant target average standard error",
        ),
        (
            "required_average_standard_error_lift",
            "dominant required average standard error lift",
        ),
        ("current_interval_length", "dominant current interval length"),
        ("target_interval_length", "dominant target interval length"),
        (
            "required_interval_length_lift",
            "dominant required interval length lift",
        ),
    ):
        normalized[key] = _calibration_target_value(
            normalized,
            key,
            label=label,
        )
    return normalized


def build_quality_risk_method_diagnosis(
    *,
    binding_driver: str,
    blocker_reason: str,
    interval_scale_status: str,
    binding_interval_length_to_standard_error_gap: float,
    best_interval_length_to_standard_error_gap: float,
    dominant_calibration_target: dict[str, object],
) -> dict[str, object]:
    dominant_calibration_target = _validated_dominant_calibration_target(
        dominant_calibration_target
    )
    max_abs_interval_scale_gap = max(
        abs(float(binding_interval_length_to_standard_error_gap)),
        abs(float(best_interval_length_to_standard_error_gap)),
    )
    if str(binding_driver).strip() == "quality-risk-cleared":
        diagnosis = "quality-risk-cleared"
    elif (
        str(interval_scale_status).strip()
        == "paper-90pct-pointwise-interval-scale-locked"
    ):
        diagnosis = "average-se-calibration-debt-not-interval-scale-drift"
    else:
        diagnosis = "interval-scale-review-required"
    current_average_standard_error = float(
        dominant_calibration_target.get("current_average_standard_error", 0.0)
    )
    required_average_standard_error_lift = float(
        dominant_calibration_target.get(
            "required_average_standard_error_lift",
            0.0,
        )
    )
    current_interval_length = float(
        dominant_calibration_target.get("current_interval_length", 0.0)
    )
    required_interval_length_lift = float(
        dominant_calibration_target.get("required_interval_length_lift", 0.0)
    )
    if current_average_standard_error <= 0.0:
        raise ValueError(
            "quality risk probe requires positive current average standard error for method diagnosis"
        )
    if current_interval_length <= 0.0:
        raise ValueError(
            "quality risk probe requires positive current interval length for method diagnosis"
        )
    return {
        "diagnosis": diagnosis,
        "binding_driver": str(binding_driver).strip(),
        "blocker_reason": str(blocker_reason).strip(),
        "interval_scale_status": str(interval_scale_status).strip(),
        "max_abs_interval_scale_gap": max_abs_interval_scale_gap,
        "dominant_condition": str(
            dominant_calibration_target.get("condition", "")
        ).strip(),
        "current_average_standard_error": current_average_standard_error,
        "required_average_standard_error_lift": required_average_standard_error_lift,
        "required_average_standard_error_lift_fraction_of_current": (
            required_average_standard_error_lift / current_average_standard_error
        ),
        "current_interval_length": current_interval_length,
        "required_interval_length_lift": required_interval_length_lift,
        "required_interval_length_lift_fraction_of_current": (
            required_interval_length_lift / current_interval_length
        ),
        "target_average_standard_error": float(
            dominant_calibration_target.get("target_average_standard_error", 0.0)
        ),
        "target_interval_length": float(
            dominant_calibration_target.get("target_interval_length", 0.0)
        ),
    }


def build_quality_risk_action_contract(
    *,
    binding_driver: str,
    interval_scale_status: str,
    dominant_calibration_target: dict[str, object],
) -> dict[str, object]:
    dominant_calibration_target = _validated_dominant_calibration_target(
        dominant_calibration_target
    )
    interval_scale_locked = (
        str(interval_scale_status).strip()
        == "paper-90pct-pointwise-interval-scale-locked"
    )
    if str(binding_driver).strip() == "quality-risk-cleared":
        action = "quality-risk-cleared"
    elif interval_scale_locked:
        action = "calibrate-average-standard-error-not-interval-multiplier"
    else:
        action = "review-interval-scale-before-average-standard-error-calibration"
    return {
        "action": action,
        "interval_scale_action": (
            "preserve-paper-90pct-pointwise-interval-multiplier"
            if interval_scale_locked
            else "review-paper-90pct-pointwise-interval-multiplier"
        ),
        "bounded_policy_action": "preserve-bounded-widening-policy",
        "release_gate_action": "keep-release-blocked-until-live-helper-clears",
        "target_condition": str(
            dominant_calibration_target.get("condition", "")
        ).strip(),
        "target_average_standard_error": float(
            dominant_calibration_target.get("target_average_standard_error", 0.0)
        ),
        "target_interval_length": float(
            dominant_calibration_target.get("target_interval_length", 0.0)
        ),
        "required_average_standard_error_lift": float(
            dominant_calibration_target.get(
                "required_average_standard_error_lift",
                0.0,
            )
        ),
        "required_interval_length_lift": float(
            dominant_calibration_target.get("required_interval_length_lift", 0.0)
        ),
    }


def build_quality_risk_average_standard_error_calibration_candidate(
    *,
    binding_summary: "Phase7MonteCarloWideningPolicyQualityRiskDesignSummary",
    best_summary: "Phase7MonteCarloWideningPolicyQualityRiskDesignSummary",
    dominant_calibration_target: dict[str, object],
) -> dict[str, object]:
    dominant_calibration_target = _validated_dominant_calibration_target(
        dominant_calibration_target
    )
    current_average_se = float(binding_summary.mean_nonparametric_average_standard_error)
    observed_interval_length = float(binding_summary.mean_nonparametric_interval_length)
    observed_interval_to_se_ratio = (
        binding_summary.interval_length_to_standard_error_ratio
    )
    observed_interval_to_se_gap = binding_summary.interval_length_to_standard_error_gap
    target_average_se = float(
        dominant_calibration_target.get(
            "target_average_standard_error",
            current_average_se,
        )
    )
    calibrated_average_se = max(target_average_se, current_average_se)
    calibrated_interval_length = (
        calibrated_average_se * _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
    )
    calibrated_rmse_to_se = _positive_ratio(
        binding_summary.mean_nonparametric_rmse,
        calibrated_average_se,
    )
    calibrated_se_reserve = (
        calibrated_average_se - binding_summary.mean_nonparametric_rmse
    )
    calibrated_interval_to_rmse = _positive_ratio(
        calibrated_interval_length,
        binding_summary.mean_nonparametric_rmse,
    )
    clearance_margin = min(
        best_summary.rmse_to_standard_error_ratio - calibrated_rmse_to_se,
        calibrated_se_reserve - best_summary.standard_error_reserve,
        calibrated_interval_to_rmse - best_summary.interval_length_to_rmse_ratio,
    )
    headroom_margins = {
        "rmse_to_se_not_trailing": (
            best_summary.rmse_to_standard_error_ratio - calibrated_rmse_to_se
        ),
        "se_reserve_not_trailing": (
            calibrated_se_reserve - best_summary.standard_error_reserve
        ),
        "interval_to_rmse_not_trailing": (
            calibrated_interval_to_rmse - best_summary.interval_length_to_rmse_ratio
        ),
    }
    clearance_conditions = [
        {
            "condition": condition,
            "status": "cleared" if margin >= -1e-9 else "open",
            "margin": margin,
        }
        for condition, margin in headroom_margins.items()
    ]
    clears_headroom = clearance_margin >= -1e-9
    if clearance_margin < -1e-9:
        boundary_status = "trailing-headroom"
    elif clearance_margin <= 1e-9:
        boundary_status = "binding-boundary"
    else:
        boundary_status = "positive-headroom"
    boundary_conditions = tuple(
        condition
        for condition, margin in headroom_margins.items()
        if math.isclose(margin, clearance_margin, rel_tol=1e-9, abs_tol=1e-9)
    )
    status = (
        "candidate-clears-quality-risk-headroom"
        if clears_headroom
        else "candidate-still-trails-quality-risk-headroom"
    )
    release_gate_effect = _NO_PROMOTION_RELEASE_GATE_EFFECT
    admission = build_quality_risk_average_standard_error_calibration_candidate_admission(
        {
            "status": status,
            "headroom_boundary_status": boundary_status,
            "minimum_headroom_margin_after_calibration": clearance_margin,
            "release_gate_effect": release_gate_effect,
        }
    )
    projection = build_quality_risk_average_standard_error_calibration_candidate_projection(
        {
            "status": status,
            "release_gate_effect": release_gate_effect,
            **admission,
        }
    )
    positive_headroom_target = (
        build_quality_risk_positive_headroom_admission_target(
            binding_summary=binding_summary,
            best_summary=best_summary,
            current_average_standard_error=calibrated_average_se,
            target_condition=str(
                dominant_calibration_target.get("condition", "")
            ).strip(),
            current_average_standard_error_before_calibration=current_average_se,
        )
    )
    return {
        "candidate": "average-standard-error-scale-calibration",
        "source": "dominant-quality-risk-calibration-target",
        "status": status,
        "candidate_status": status,
        "binding_design": _format_design_key(*binding_summary.design_key),
        "comparison_design": _format_design_key(*best_summary.design_key),
        "interval_scale_status": _interval_scale_status_for_summaries(
            binding_summary,
            best_summary,
        ),
        "observed_interval_length": observed_interval_length,
        "observed_interval_to_standard_error_ratio": observed_interval_to_se_ratio,
        "observed_interval_to_standard_error_gap": observed_interval_to_se_gap,
        "target_condition": str(
            dominant_calibration_target.get("condition", "")
        ).strip(),
        "current_average_standard_error": current_average_se,
        "target_average_standard_error": calibrated_average_se,
        "calibrated_average_standard_error": calibrated_average_se,
        "required_average_standard_error_lift": max(
            calibrated_average_se - current_average_se,
            0.0,
        ),
        "scale_factor": _positive_ratio(calibrated_average_se, current_average_se),
        "current_interval_length": (
            current_average_se * _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
        ),
        "target_interval_length": calibrated_interval_length,
        "calibrated_interval_length": calibrated_interval_length,
        "required_interval_length_lift": max(
            calibrated_interval_length
            - (
                current_average_se
                * _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
            ),
            0.0,
        ),
        "preserves_interval_multiplier": True,
        "paper_interval_multiplier": _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER,
        "calibrated_interval_to_standard_error_ratio": (
            _positive_ratio(calibrated_interval_length, calibrated_average_se)
        ),
        "calibrated_rmse_to_standard_error_ratio": calibrated_rmse_to_se,
        "comparison_rmse_to_standard_error_ratio": (
            best_summary.rmse_to_standard_error_ratio
        ),
        "calibrated_standard_error_reserve": calibrated_se_reserve,
        "comparison_standard_error_reserve": best_summary.standard_error_reserve,
        "calibrated_interval_length_to_rmse_ratio": calibrated_interval_to_rmse,
        "comparison_interval_length_to_rmse_ratio": (
            best_summary.interval_length_to_rmse_ratio
        ),
        "calibrated_quality_risk_headroom_margins": headroom_margins,
        "calibrated_quality_risk_clearance_conditions": clearance_conditions,
        "minimum_headroom_margin_after_calibration": clearance_margin,
        "headroom_boundary_status": boundary_status,
        "headroom_boundary_conditions": list(boundary_conditions),
        "positive_headroom_admission_target": positive_headroom_target,
        "release_gate_effect": release_gate_effect,
        **projection,
        **admission,
    }


def build_quality_risk_positive_headroom_admission_target(
    *,
    binding_summary: "Phase7MonteCarloWideningPolicyQualityRiskDesignSummary",
    best_summary: "Phase7MonteCarloWideningPolicyQualityRiskDesignSummary",
    current_average_standard_error: float,
    target_condition: str,
    current_average_standard_error_before_calibration: float,
) -> dict[str, object]:
    current_average_standard_error = _validated_finite(
        current_average_standard_error,
        label="current calibrated average standard error",
    )
    current_average_standard_error_before_calibration = _validated_finite(
        current_average_standard_error_before_calibration,
        label="pre-calibration average standard error",
    )
    if current_average_standard_error <= 0.0:
        raise ValueError(
            "quality risk positive-headroom target requires positive current average standard error"
        )
    if current_average_standard_error_before_calibration <= 0.0:
        raise ValueError(
            "quality risk positive-headroom target requires positive pre-calibration average standard error"
        )
    margin = _POSITIVE_HEADROOM_ADMISSION_MARGIN
    best_rmse_to_standard_error_ratio = best_summary.rmse_to_standard_error_ratio
    if best_rmse_to_standard_error_ratio <= margin:
        raise ValueError(
            "quality risk positive-headroom target requires comparison RMSE/SE above the admission margin"
        )
    minimum_average_standard_error = max(
        current_average_standard_error,
        binding_summary.mean_nonparametric_rmse
        / (best_rmse_to_standard_error_ratio - margin),
        binding_summary.mean_nonparametric_rmse
        + best_summary.standard_error_reserve
        + margin,
        (
            (best_summary.interval_length_to_rmse_ratio + margin)
            * binding_summary.mean_nonparametric_rmse
            / _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
        ),
    )
    minimum_interval_length = (
        minimum_average_standard_error
        * _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
    )
    return {
        "target_condition": str(target_condition).strip(),
        "minimum_positive_headroom_margin": margin,
        "minimum_admissible_average_standard_error": minimum_average_standard_error,
        "minimum_admissible_interval_length": minimum_interval_length,
        "required_average_standard_error_lift": max(
            minimum_average_standard_error
            - current_average_standard_error_before_calibration,
            0.0,
        ),
        "required_interval_length_lift": max(
            minimum_interval_length
            - (
                current_average_standard_error_before_calibration
                * _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
            ),
            0.0,
        ),
        "admission_status": "positive-headroom-target-required",
    }


def _positive_headroom_target_scalar(
    target: dict[str, object],
    key: str,
    *,
    label: str,
) -> float:
    value = _validated_finite(target[key], label=f"positive-headroom target {label}")
    if value <= 0.0:
        raise ValueError(
            f"quality risk probe requires positive-headroom target {label} to be positive"
        )
    return value


def _require_positive_headroom_target_interval_scale(
    *,
    target_average_standard_error: float,
    target_interval_length: float,
) -> None:
    expected_interval_length = (
        target_average_standard_error
        * _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
    )
    if not math.isclose(
        target_interval_length,
        expected_interval_length,
        rel_tol=1e-9,
        abs_tol=_INTERVAL_SCALE_LOCK_TOLERANCE,
    ):
        raise ValueError(
            "quality risk probe requires positive-headroom target interval length "
            "to preserve the paper interval scale"
        )


def _display_interval_scale_gap(interval_to_se_gap: float) -> float:
    gap = float(interval_to_se_gap)
    if abs(gap) <= _INTERVAL_SCALE_LOCK_TOLERANCE:
        return 0.0
    return gap


def _calibration_candidate_positive_value(
    candidate: dict[str, object],
    key: str,
    *,
    label: str,
) -> float:
    value = _validated_finite(candidate[key], label=label)
    if value <= 0.0:
        raise ValueError(f"quality risk probe requires positive {label}")
    return value


def _calibration_candidate_finite_value(
    candidate: dict[str, object],
    key: str,
    *,
    label: str,
) -> float:
    return _validated_finite(candidate[key], label=label)


def _calibration_candidate_headroom_margin(
    candidate: dict[str, object],
    key: str,
) -> float:
    margins = dict(candidate["calibrated_quality_risk_headroom_margins"])
    return _validated_finite(
        margins[key],
        label=f"calibration candidate headroom margin {key}",
    )


def build_quality_risk_average_standard_error_calibration_candidate_evidence(
    candidate: dict[str, object],
) -> tuple[str, ...]:
    release_gate_effect = _validated_no_promotion_release_gate_effect(
        candidate["release_gate_effect"]
    )
    positive_headroom_target = dict(candidate["positive_headroom_admission_target"])
    positive_headroom_average_se = _positive_headroom_target_scalar(
        positive_headroom_target,
        "minimum_admissible_average_standard_error",
        label="average standard error",
    )
    positive_headroom_interval_length = _positive_headroom_target_scalar(
        positive_headroom_target,
        "minimum_admissible_interval_length",
        label="interval length",
    )
    _require_positive_headroom_target_interval_scale(
        target_average_standard_error=positive_headroom_average_se,
        target_interval_length=positive_headroom_interval_length,
    )
    positive_headroom_margin = _positive_headroom_target_scalar(
        positive_headroom_target,
        "minimum_positive_headroom_margin",
        label="margin",
    )
    current_average_se = _calibration_candidate_positive_value(
        candidate,
        "current_average_standard_error",
        label="calibration candidate current average standard error",
    )
    calibrated_average_se = _calibration_candidate_positive_value(
        candidate,
        "calibrated_average_standard_error",
        label="calibration candidate calibrated average standard error",
    )
    current_interval_length = _calibration_candidate_positive_value(
        candidate,
        "current_interval_length",
        label="calibration candidate current interval length",
    )
    calibrated_interval_length = _calibration_candidate_positive_value(
        candidate,
        "calibrated_interval_length",
        label="calibration candidate calibrated interval length",
    )
    scale_factor = _calibration_candidate_positive_value(
        candidate,
        "scale_factor",
        label="calibration candidate scale factor",
    )
    minimum_headroom_margin = _calibration_candidate_finite_value(
        candidate,
        "minimum_headroom_margin_after_calibration",
        label="calibration candidate minimum headroom margin",
    )
    rmse_to_se_margin = _calibration_candidate_headroom_margin(
        candidate,
        "rmse_to_se_not_trailing",
    )
    se_reserve_margin = _calibration_candidate_headroom_margin(
        candidate,
        "se_reserve_not_trailing",
    )
    interval_to_rmse_margin = _calibration_candidate_headroom_margin(
        candidate,
        "interval_to_rmse_not_trailing",
    )
    return (
        f"calibration_candidate={candidate['candidate']}",
        f"calibration_candidate_status={candidate['status']}",
        f"calibration_candidate_target={candidate['target_condition']}",
        "calibration_candidate_average_se "
        f"{current_average_se:.3f}->"
        f"{calibrated_average_se:.3f}",
        "calibration_candidate_interval_length "
        f"{current_interval_length:.3f}->"
        f"{calibrated_interval_length:.3f}",
        "calibration_candidate_scale_factor=" f"{scale_factor:.3f}",
        "calibration_candidate_minimum_headroom_margin="
        f"{minimum_headroom_margin:+.3f}",
        "calibration_candidate_headroom_margins "
        "rmse_to_se="
        f"{rmse_to_se_margin:+.3f} "
        "se_reserve="
        f"{se_reserve_margin:+.3f} "
        "interval_to_rmse="
        f"{interval_to_rmse_margin:+.3f}",
        "calibration_candidate_projected_quality_risk_status="
        f"{candidate['projected_quality_risk_status']}",
        "calibration_candidate_headroom_boundary_status="
        f"{candidate['headroom_boundary_status']}",
        "calibration_candidate_headroom_boundary_conditions="
        f"{','.join(str(item) for item in candidate['headroom_boundary_conditions'])}",
        "calibration_candidate_positive_headroom_target_average_se="
        f"{positive_headroom_average_se:.3f}",
        "calibration_candidate_positive_headroom_target_interval_length="
        f"{positive_headroom_interval_length:.3f}",
        "calibration_candidate_positive_headroom_margin="
        f"{positive_headroom_margin:+.3e}",
        f"calibration_candidate_admission_status={candidate['admission_status']}",
        f"calibration_candidate_admission_blocker={candidate['admission_blocker']}",
        "calibration_candidate_admission_next_evidence="
        f"{candidate['admission_required_next_evidence']}",
        f"calibration_candidate_release_gate_effect={release_gate_effect}",
    )


def build_quality_risk_positive_headroom_estimator_evidence_requirement(
    candidate: dict[str, object],
) -> dict[str, object]:
    positive_headroom_target = dict(candidate["positive_headroom_admission_target"])
    current_average_se = _validated_finite(
        candidate["current_average_standard_error"],
        label="positive-headroom current average standard error",
    )
    current_interval_length = _validated_finite(
        candidate["observed_interval_length"],
        label="positive-headroom current interval length",
    )
    target_average_se = _positive_headroom_target_scalar(
        positive_headroom_target,
        "minimum_admissible_average_standard_error",
        label="average standard error",
    )
    target_interval_length = _positive_headroom_target_scalar(
        positive_headroom_target,
        "minimum_admissible_interval_length",
        label="interval length",
    )
    _require_positive_headroom_target_interval_scale(
        target_average_standard_error=target_average_se,
        target_interval_length=target_interval_length,
    )
    remaining_average_se_gap = max(target_average_se - current_average_se, 0.0)
    remaining_interval_length_gap = max(
        target_interval_length - current_interval_length,
        0.0,
    )
    if remaining_average_se_gap > 0.0 or remaining_interval_length_gap > 0.0:
        status = "positive-headroom-estimator-evidence-required"
    else:
        status = "positive-headroom-estimator-evidence-at-target"
    return {
        "status": status,
        "source": "positive-headroom-admission-target",
        "current_average_standard_error": current_average_se,
        "target_average_standard_error": target_average_se,
        "remaining_average_standard_error_gap": remaining_average_se_gap,
        "current_interval_length": current_interval_length,
        "target_interval_length": target_interval_length,
        "remaining_interval_length_gap": remaining_interval_length_gap,
        "required_next_evidence": (
            "fresh-helper-average-se-and-interval-length-evidence"
        ),
        "release_gate_effect": _validated_no_promotion_release_gate_effect(
            candidate["release_gate_effect"]
        ),
    }


def build_quality_risk_positive_headroom_estimator_evidence_verdict(
    requirement: dict[str, object],
    *,
    observed_average_standard_error: float,
    observed_interval_length: float,
    observed_evidence_source: str = "current-binding-runtime-probe",
) -> dict[str, object]:
    requirement_source = str(requirement.get("source", "")).strip()
    if requirement_source != "positive-headroom-admission-target":
        raise ValueError(
            "quality risk probe requires positive-headroom requirement source "
            "to be positive-headroom-admission-target"
        )
    requirement_status = str(requirement.get("status", "")).strip()
    if requirement_status not in {
        "positive-headroom-estimator-evidence-required",
        "positive-headroom-estimator-evidence-at-target",
    }:
        raise ValueError(
            "quality risk probe requires positive-headroom requirement status "
            "to come from the estimator evidence requirement"
        )
    release_gate_effect = str(requirement.get("release_gate_effect", "")).strip()
    if release_gate_effect != _NO_PROMOTION_RELEASE_GATE_EFFECT:
        raise ValueError(
            "quality risk probe requires positive-headroom requirement release "
            "gate effect to preserve the no-promotion contract"
        )
    required_next_evidence = str(
        requirement.get("required_next_evidence", "")
    ).strip()
    if required_next_evidence != _POSITIVE_HEADROOM_FRESH_HELPER_EVIDENCE_SOURCE:
        raise ValueError(
            "quality risk probe requires positive-headroom requirement next evidence "
            "to be fresh-helper-average-se-and-interval-length-evidence"
        )
    observed_average_se = _validated_finite(
        observed_average_standard_error,
        label="positive-headroom observed average standard error",
    )
    observed_interval_length = _validated_finite(
        observed_interval_length,
        label="positive-headroom observed interval length",
    )
    if observed_average_se <= 0.0 or observed_interval_length <= 0.0:
        raise ValueError(
            "quality risk positive-headroom verifier requires positive observed average standard error and interval length"
        )
    target_average_se = _positive_headroom_target_scalar(
        requirement,
        "target_average_standard_error",
        label="average standard error",
    )
    target_interval_length = _positive_headroom_target_scalar(
        requirement,
        "target_interval_length",
        label="interval length",
    )
    _require_positive_headroom_target_interval_scale(
        target_average_standard_error=target_average_se,
        target_interval_length=target_interval_length,
    )
    average_se_margin = observed_average_se - target_average_se
    interval_length_margin = observed_interval_length - target_interval_length
    remaining_average_se_gap = max(target_average_se - observed_average_se, 0.0)
    remaining_interval_length_gap = max(
        target_interval_length - observed_interval_length,
        0.0,
    )
    remaining_average_se_gap_fraction = (
        remaining_average_se_gap / target_average_se
    )
    remaining_interval_length_gap_fraction = (
        remaining_interval_length_gap / target_interval_length
    )
    interval_to_se_ratio = _positive_ratio(
        observed_interval_length,
        observed_average_se,
    )
    interval_to_se_gap = (
        interval_to_se_ratio - _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
    )
    interval_scale_status = (
        "paper-90pct-pointwise-interval-scale-locked"
        if abs(interval_to_se_gap) <= _INTERVAL_SCALE_LOCK_TOLERANCE
        else "paper-90pct-pointwise-interval-scale-needs-review"
    )
    observed_evidence_source = str(observed_evidence_source).strip()
    fresh_helper_evidence = (
        observed_evidence_source == _POSITIVE_HEADROOM_FRESH_HELPER_EVIDENCE_SOURCE
    )
    target_met = (
        average_se_margin >= -1e-9
        and interval_length_margin >= -1e-9
        and interval_scale_status == "paper-90pct-pointwise-interval-scale-locked"
        and fresh_helper_evidence
    )
    if target_met:
        status = "positive-headroom-estimator-evidence-at-target"
        next_action = "rerun-live-feature-gate-before-release-promotion"
        gate_rerun_status = "fresh-target-evidence-ready-for-gate-rerun"
    elif (
        average_se_margin >= -1e-9
        and interval_length_margin >= -1e-9
        and interval_scale_status == "paper-90pct-pointwise-interval-scale-locked"
    ):
        status = "positive-headroom-estimator-evidence-required"
        next_action = "fresh-helper-evidence-source-required"
        gate_rerun_status = "target-shaped-evidence-not-fresh"
    else:
        status = "positive-headroom-estimator-evidence-required"
        next_action = "fresh-helper-average-se-and-interval-length-evidence"
        gate_rerun_status = "target-evidence-not-ready"
    return {
        "status": status,
        "source": "observed-positive-headroom-estimator-evidence",
        "observed_evidence_source": observed_evidence_source,
        "observed_evidence_source_status": (
            "fresh-helper-evidence"
            if fresh_helper_evidence
            else "not-fresh-helper-evidence"
        ),
        "observed_average_standard_error": observed_average_se,
        "target_average_standard_error": target_average_se,
        "average_standard_error_margin": average_se_margin,
        "remaining_average_standard_error_gap": remaining_average_se_gap,
        "remaining_average_standard_error_gap_fraction": (
            remaining_average_se_gap_fraction
        ),
        "observed_interval_length": observed_interval_length,
        "target_interval_length": target_interval_length,
        "interval_length_margin": interval_length_margin,
        "remaining_interval_length_gap": remaining_interval_length_gap,
        "remaining_interval_length_gap_fraction": (
            remaining_interval_length_gap_fraction
        ),
        "maximum_remaining_gap_fraction": max(
            remaining_average_se_gap_fraction,
            remaining_interval_length_gap_fraction,
        ),
        "observed_interval_to_standard_error_ratio": interval_to_se_ratio,
        "paper_interval_to_standard_error_multiplier": (
            _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
        ),
        "interval_to_standard_error_gap": _display_interval_scale_gap(
            interval_to_se_gap
        ),
        "interval_scale_status": interval_scale_status,
        "required_next_evidence": next_action,
        "admissible_for_feature_gate_rerun": target_met,
        "feature_gate_rerun_admission_status": gate_rerun_status,
        "release_gate_effect": release_gate_effect,
    }


def _calibration_candidate_value(
    candidate: dict[str, object],
    key: str,
) -> object:
    return candidate[key]


def _validated_no_promotion_release_gate_effect(value: object) -> str:
    release_gate_effect = str(value).strip()
    if release_gate_effect != _NO_PROMOTION_RELEASE_GATE_EFFECT:
        raise ValueError(
            "quality risk probe requires release gate effect to preserve the "
            "no-promotion-without-fresh-helper-evidence contract"
        )
    return release_gate_effect


def build_quality_risk_average_standard_error_calibration_candidate_projection(
    candidate: dict[str, object],
) -> dict[str, object]:
    release_gate_effect = _validated_no_promotion_release_gate_effect(
        candidate["release_gate_effect"]
    )
    candidate_status = str(candidate.get("status", "")).strip()
    admission_status = str(candidate.get("admission_status", "")).strip()
    admission_blocker = str(candidate.get("admission_blocker", "")).strip()
    if (
        candidate_status == "candidate-clears-quality-risk-headroom"
        and admission_status
        == "candidate-positive-headroom-needs-fresh-helper-evidence"
    ):
        projected_quality_risk_driver = "fresh-helper-evidence-required"
    else:
        projected_quality_risk_driver = (
            admission_blocker or "positive-headroom-estimator-evidence-required"
        )
    return {
        "projected_quality_risk_status": "quality-risk-still-open",
        "projected_quality_risk_driver": projected_quality_risk_driver,
        "projected_release_gate_effect": release_gate_effect,
    }


def build_quality_risk_average_standard_error_calibration_candidate_admission(
    candidate: dict[str, object],
) -> dict[str, object]:
    margin = _calibration_candidate_finite_value(
        candidate,
        "minimum_headroom_margin_after_calibration",
        label="headroom margin for calibration candidate admission",
    )
    candidate_status = str(candidate["status"]).strip()
    boundary_status = str(candidate["headroom_boundary_status"]).strip()
    release_gate_effect = _validated_no_promotion_release_gate_effect(
        candidate["release_gate_effect"]
    )
    if candidate_status != "candidate-clears-quality-risk-headroom":
        admission_status = "candidate-not-admissible"
        admission_blocker = "headroom-still-trailing"
        next_evidence = "revise-average-se-target-before-rerun"
    elif boundary_status != "positive-headroom" or margin <= 1e-9:
        admission_status = "candidate-boundary-only-no-admission"
        admission_blocker = (
            boundary_status
            if boundary_status != "positive-headroom"
            else "nonpositive-headroom-margin"
        )
        next_evidence = "positive-headroom-estimator-evidence-required"
    else:
        admission_status = "candidate-positive-headroom-needs-fresh-helper-evidence"
        admission_blocker = "fresh-helper-evidence-required"
        next_evidence = "bounded-helper-rerun-required-before-release-promotion"
    return {
        "admission_status": admission_status,
        "admission_blocker": admission_blocker,
        "admission_required_next_evidence": next_evidence,
        "admission_release_gate_effect": release_gate_effect,
        "admission_minimum_headroom_margin": margin,
        "admissible_as_release_evidence": False,
    }


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOMATION_STATE_PATH = _REPO_ROOT / "Docs" / "automation" / "automation-state.yaml"
_REPO_SIDE_BINDING_DESIGN = ("DGP2", 500, 50)
_REPO_SIDE_BEST_DESIGN = ("DGP1", 500, 50)
_REPO_SIDE_BINDING_DRIVER = "rmse-outpaces-average-se"
_NO_PROMOTION_RELEASE_GATE_EFFECT = (
    "diagnostic-only-no-promotion-without-fresh-helper-evidence"
)
_PAPER_OBJECT_CONTRACT = (
    "paper-section-5-monte-carlo-quality-objects",
    "nonparametric coverage",
    "nonparametric RMSE",
    "nonparametric average standard error",
    "nonparametric confidence interval length",
    "90% pointwise interval length equals 2*z0.95*average SE",
    "derived calibration ratios: RMSE/SE, SE-RMSE, interval/RMSE",
)
_PAPER_EVIDENCE_REFS = (
    "paper/hddid_paper.md:301 nonparametric pointwise CI formula",
    "paper/hddid_paper.md:322 Monte Carlo reports average SE/RMSE/coverage/CI length",
    "paper/hddid_paper.md:324 finite-sample comparison uses SE/RMSE/CI length",
)
_R_EVIDENCE_REFS = (
    "hddid-r/R/highdimdiffindiff_crossfit.R:34-40 foldwise stdg aggregation",
    "hddid-r/R/highdimdiffindiff_crossfit_inside.R:124-129 fold CIpoint uses qnorm and stdg",
    "hddid-r/R/highdimdiffindiff_crossfit.R:42 CIuniform uses the same fold-level stdg path without a calibration scalar",
)
_PYTHON_EVIDENCE_REFS = (
    "hddid-py/src/hddid/monte_carlo_widening_policy_quality_risk_probe.py:build_quality_risk_positive_headroom_estimator_evidence_verdict",
    "hddid-py/src/hddid/monte_carlo_widening_policy_quality_risk_probe.py:Phase7MonteCarloWideningPolicyQualityRiskProbeReport.quality_risk_positive_headroom_estimator_evidence_verdict",
)
_PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER = 3.2897072539029454
_INTERVAL_SCALE_LOCK_TOLERANCE = 1e-6
_POSITIVE_HEADROOM_ADMISSION_MARGIN = 1e-3
_POSITIVE_HEADROOM_FRESH_HELPER_EVIDENCE_SOURCE = (
    "fresh-helper-average-se-and-interval-length-evidence"
)
_QUALITY_RISK_RUNTIME_PROBE_DESIGNS = default_phase7_runtime_probe_designs()[:6]
_REPO_SIDE_QUALITY_SUMMARIES = {
    ("DGP1", 500, 50): {
        "mean_nonparametric_coverage": 8.0 / 9.0,
        "coverage_floor_slack": (8.0 / 9.0) - 0.85,
        "mean_nonparametric_rmse": 1.9111312266475258,
        "mean_nonparametric_average_standard_error": 2.2681232532639695,
        "mean_nonparametric_interval_length": 7.461461519008423,
    },
    ("DGP2", 500, 50): {
        "mean_nonparametric_coverage": 7.0 / 9.0,
        "coverage_floor_slack": (7.0 / 9.0) - 0.85,
        "mean_nonparametric_rmse": 4.171122823129394,
        "mean_nonparametric_average_standard_error": 3.992351539913328,
        "mean_nonparametric_interval_length": 13.133667820983462,
    },
}


def _parse_design_label(design_label: str) -> tuple[str, int, int]:
    dgp_name, n_obs, p = str(design_label).split("/")
    return (dgp_name, int(n_obs), int(p))


def _repo_side_quality_risk_designs() -> tuple[tuple[str, int, int], ...]:
    if not _AUTOMATION_STATE_PATH.exists():
        return ()
    state = yaml.safe_load(_AUTOMATION_STATE_PATH.read_text(encoding="utf-8"))
    gate_state = state["feature_completion_gate"]["checks"][
        "monte_carlo_validation_ready"
    ]
    return tuple(
        _parse_design_label(label)
        for label in gate_state.get("quality_risk_designs", ["DGP2/500/50"])
    )


def _repo_side_quality_risk_boundary_state() -> tuple[
    tuple[str, int, int],
    tuple[str, int, int],
]:
    if not _AUTOMATION_STATE_PATH.exists():
        return (
            _REPO_SIDE_BINDING_DESIGN,
            _REPO_SIDE_BEST_DESIGN,
        )
    state = yaml.safe_load(_AUTOMATION_STATE_PATH.read_text(encoding="utf-8"))
    gate_state = state["feature_completion_gate"]["checks"][
        "monte_carlo_validation_ready"
    ]
    binding_design = _parse_design_label(
        gate_state.get(
            "quality_risk_binding_design",
            _format_design_key(*_REPO_SIDE_BINDING_DESIGN),
        )
    )
    best_design = _parse_design_label(
        gate_state.get(
            "quality_risk_best_design",
            _format_design_key(*_REPO_SIDE_BEST_DESIGN),
        )
    )
    return binding_design, best_design


def _run_quality_risk_runtime_probe():
    designs = tuple(_QUALITY_RISK_RUNTIME_PROBE_DESIGNS)
    probe_signature = signature(run_phase7_monte_carlo_runtime_probe)
    accepts_designs = "designs" in probe_signature.parameters or any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in probe_signature.parameters.values()
    )
    if accepts_designs:
        return run_phase7_monte_carlo_runtime_probe(designs=designs)
    return run_phase7_monte_carlo_runtime_probe()


def _prefer_repo_side_quality_risk_first(repo_root: str | Path | None) -> bool:
    if repo_root is None:
        return False
    if run_phase7_monte_carlo_runtime_probe is not _DEFAULT_RUNTIME_PROBE_RUNNER:
        return False
    return bool(_repo_side_quality_risk_designs())


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyQualityRiskDesignSummary:
    dgp_name: str
    n_obs: int
    p: int
    mean_nonparametric_coverage: float
    coverage_floor_slack: float
    mean_nonparametric_rmse: float
    mean_nonparametric_average_standard_error: float
    mean_nonparametric_interval_length: float
    rmse_to_standard_error_ratio: float
    standard_error_reserve: float
    interval_length_to_rmse_ratio: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip()
        self.n_obs = _validated_positive_integer(self.n_obs, label="n_obs")
        self.p = _validated_positive_integer(self.p, label="p")
        self.mean_nonparametric_coverage = _validated_coverage(
            self.mean_nonparametric_coverage,
            label="nonparametric coverage",
        )
        self.coverage_floor_slack = _validated_finite(
            self.coverage_floor_slack,
            label="coverage floor slack",
        )
        self.mean_nonparametric_rmse = _validated_finite(
            self.mean_nonparametric_rmse,
            label="nonparametric rmse",
        )
        self.mean_nonparametric_average_standard_error = _validated_finite(
            self.mean_nonparametric_average_standard_error,
            label="nonparametric average standard error",
        )
        self.mean_nonparametric_interval_length = _validated_finite(
            self.mean_nonparametric_interval_length,
            label="nonparametric interval length",
        )
        supplied_rmse_to_se = _positive_ratio(
            self.rmse_to_standard_error_ratio,
            1.0,
        )
        self.rmse_to_standard_error_ratio = _positive_ratio(
            self.mean_nonparametric_rmse,
            self.mean_nonparametric_average_standard_error,
        )
        _require_close_raw_metric(
            supplied_rmse_to_se,
            self.rmse_to_standard_error_ratio,
            label="design summary RMSE/SE",
        )
        self.standard_error_reserve = _validated_finite(
            self.standard_error_reserve,
            label="standard error reserve",
        )
        _require_close_raw_metric(
            self.standard_error_reserve,
            self.mean_nonparametric_average_standard_error
            - self.mean_nonparametric_rmse,
            label="design summary standard error reserve",
        )
        supplied_interval_to_rmse = _positive_ratio(
            self.interval_length_to_rmse_ratio,
            1.0,
        )
        self.interval_length_to_rmse_ratio = _positive_ratio(
            self.mean_nonparametric_interval_length,
            self.mean_nonparametric_rmse,
        )
        _require_close_raw_metric(
            supplied_interval_to_rmse,
            self.interval_length_to_rmse_ratio,
            label="design summary interval/RMSE",
        )

    @property
    def design_key(self) -> tuple[str, int, int]:
        return (self.dgp_name, self.n_obs, self.p)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "n_obs": self.n_obs,
            "p": self.p,
            "mean_nonparametric_coverage": self.mean_nonparametric_coverage,
            "coverage_floor_slack": self.coverage_floor_slack,
            "mean_nonparametric_rmse": self.mean_nonparametric_rmse,
            "mean_nonparametric_average_standard_error": self.mean_nonparametric_average_standard_error,
            "mean_nonparametric_interval_length": self.mean_nonparametric_interval_length,
            "rmse_to_standard_error_ratio": self.rmse_to_standard_error_ratio,
            "standard_error_reserve": self.standard_error_reserve,
            "interval_length_to_rmse_ratio": self.interval_length_to_rmse_ratio,
            "interval_length_to_standard_error_ratio": (
                self.interval_length_to_standard_error_ratio
            ),
            "interval_length_to_standard_error_gap": (
                self.interval_length_to_standard_error_gap
            ),
        }

    @property
    def interval_length_to_standard_error_ratio(self) -> float:
        return _positive_ratio(
            self.mean_nonparametric_interval_length,
            self.mean_nonparametric_average_standard_error,
        )

    @property
    def interval_length_to_standard_error_gap(self) -> float:
        return (
            self.interval_length_to_standard_error_ratio
            - _PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER
        )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyQualityRiskProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    best_design: tuple[str, int, int]
    binding_driver: str
    blocker_reason: str
    supported_floor_ceiling: float
    canonical_floor_shortfall: float
    rmse_to_standard_error_ratio_gap: float
    standard_error_reserve_gap: float
    interval_length_to_rmse_gap: float
    design_quality_summaries: tuple[
        Phase7MonteCarloWideningPolicyQualityRiskDesignSummary, ...
    ]
    canonical_quality_risk_digest: tuple[str, ...]
    quality_risk_source_mode: str = "live-runtime-probe"
    quality_risk_source_note: str = "live runtime probe"

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip(),
            _validated_positive_integer(self.binding_design[1], label="binding n_obs"),
            _validated_positive_integer(self.binding_design[2], label="binding p"),
        )
        self.best_design = (
            str(self.best_design[0]).strip(),
            _validated_positive_integer(self.best_design[1], label="best n_obs"),
            _validated_positive_integer(self.best_design[2], label="best p"),
        )
        self.binding_driver = str(self.binding_driver).strip()
        self.blocker_reason = str(self.blocker_reason).strip()
        self.supported_floor_ceiling = _validated_finite(
            self.supported_floor_ceiling,
            label="supported floor ceiling",
        )
        self.canonical_floor_shortfall = _validated_finite(
            self.canonical_floor_shortfall,
            label="canonical floor shortfall",
        )
        self.rmse_to_standard_error_ratio_gap = _validated_finite(
            self.rmse_to_standard_error_ratio_gap,
            label="RMSE/SE gap",
        )
        self.standard_error_reserve_gap = _validated_finite(
            self.standard_error_reserve_gap,
            label="SE-RMSE gap",
        )
        self.interval_length_to_rmse_gap = _validated_finite(
            self.interval_length_to_rmse_gap,
            label="interval/RMSE gap",
        )
        self.design_quality_summaries = tuple(self.design_quality_summaries)
        self.canonical_quality_risk_digest = tuple(
            str(line).rstrip() for line in self.canonical_quality_risk_digest
        )
        self.quality_risk_source_mode = str(self.quality_risk_source_mode).strip()
        self.quality_risk_source_note = str(self.quality_risk_source_note).strip()
        self._validate_design_summary_raw_metrics()
        self._validate_coverage_slack_metrics()
        self._validate_driver_metric_consistency()

    def _validate_design_summary_raw_metrics(self) -> None:
        for summary in self.design_quality_summaries:
            _require_close_raw_metric(
                summary.rmse_to_standard_error_ratio,
                _positive_ratio(
                    summary.mean_nonparametric_rmse,
                    summary.mean_nonparametric_average_standard_error,
                ),
                label="RMSE/SE",
            )
            _require_close_raw_metric(
                summary.standard_error_reserve,
                summary.mean_nonparametric_average_standard_error
                - summary.mean_nonparametric_rmse,
                label="standard error reserve",
            )
            _require_close_raw_metric(
                summary.interval_length_to_rmse_ratio,
                _positive_ratio(
                    summary.mean_nonparametric_interval_length,
                    summary.mean_nonparametric_rmse,
                ),
                label="interval/RMSE",
            )

    def _validate_coverage_slack_metrics(self) -> None:
        coverage_floor = _policy_min_nonparametric_coverage(self.policy_digest)
        if coverage_floor is None:
            return
        for summary in self.design_quality_summaries:
            _require_close_raw_metric(
                summary.coverage_floor_slack,
                summary.mean_nonparametric_coverage - coverage_floor,
                label=f"{_format_design_key(*summary.design_key)} coverage floor slack",
            )

    def _validate_driver_metric_consistency(self) -> None:
        binding = self.binding_quality_summary
        best = self.best_quality_summary
        expected_driver = _binding_driver(binding, best)
        if self.binding_driver != expected_driver:
            raise ValueError(
                "quality risk probe requires binding driver to match design summaries"
            )
        expected_blocker = _blocker_reason_for_driver(expected_driver)
        if self.blocker_reason != expected_blocker:
            raise ValueError(
                "quality risk probe requires blocker reason to match binding driver"
            )
        _require_close_metric(
            self.rmse_to_standard_error_ratio_gap,
            binding.rmse_to_standard_error_ratio - best.rmse_to_standard_error_ratio,
            label="RMSE/SE gap",
        )
        _require_close_metric(
            self.standard_error_reserve_gap,
            best.standard_error_reserve - binding.standard_error_reserve,
            label="SE-RMSE gap",
        )
        _require_close_metric(
            self.interval_length_to_rmse_gap,
            best.interval_length_to_rmse_ratio
            - binding.interval_length_to_rmse_ratio,
            label="interval/RMSE gap",
        )
        _require_close_metric(
            self.supported_floor_ceiling,
            binding.mean_nonparametric_coverage,
            label="supported floor ceiling to match binding design coverage",
        )
        _require_close_metric(
            self.canonical_floor_shortfall,
            max(-binding.coverage_floor_slack, 0.0),
            label="canonical floor shortfall to match binding design coverage slack",
        )

    def design_quality_summary(
        self,
        dgp_name: str,
        n_obs: int,
        p: int,
    ) -> Phase7MonteCarloWideningPolicyQualityRiskDesignSummary:
        target = (
            str(dgp_name).strip(),
            _validated_positive_integer(n_obs, label="n_obs"),
            _validated_positive_integer(p, label="p"),
        )
        for summary in self.design_quality_summaries:
            if summary.design_key == target:
                return summary
        raise KeyError(f"quality risk summary not present for design: {target!r}")

    @property
    def binding_quality_summary(
        self,
    ) -> Phase7MonteCarloWideningPolicyQualityRiskDesignSummary:
        return self.design_quality_summary(*self.binding_design)

    @property
    def binding_design_summary(
        self,
    ) -> Phase7MonteCarloWideningPolicyQualityRiskDesignSummary:
        return self.binding_quality_summary

    @property
    def best_quality_summary(
        self,
    ) -> Phase7MonteCarloWideningPolicyQualityRiskDesignSummary:
        return self.design_quality_summary(*self.best_design)

    @property
    def best_design_summary(
        self,
    ) -> Phase7MonteCarloWideningPolicyQualityRiskDesignSummary:
        return self.best_quality_summary

    @property
    def binding_coverage(self) -> float:
        return self.binding_quality_summary.mean_nonparametric_coverage

    @property
    def quality_risk_binding_coverage(self) -> float:
        return self.binding_coverage

    @property
    def binding_coverage_floor_slack(self) -> float:
        return self.binding_quality_summary.coverage_floor_slack

    @property
    def coverage_floor_slack(self) -> float:
        return self.binding_coverage_floor_slack

    @property
    def quality_risk_binding_coverage_floor_slack(self) -> float:
        return self.binding_coverage_floor_slack

    @property
    def quality_risk_coverage_floor_slack(self) -> float:
        return self.binding_coverage_floor_slack

    @property
    def quality_risk_canonical_floor_shortfall(self) -> float:
        return self.canonical_floor_shortfall

    @property
    def binding_rmse(self) -> float:
        return self.binding_quality_summary.mean_nonparametric_rmse

    @property
    def quality_risk_binding_rmse(self) -> float:
        return self.binding_rmse

    @property
    def binding_average_standard_error(self) -> float:
        return self.binding_quality_summary.mean_nonparametric_average_standard_error

    @property
    def quality_risk_binding_average_standard_error(self) -> float:
        return self.binding_average_standard_error

    @property
    def binding_standard_error_reserve(self) -> float:
        return self.binding_quality_summary.standard_error_reserve

    @property
    def quality_risk_binding_standard_error_reserve(self) -> float:
        return self.binding_standard_error_reserve

    @property
    def binding_interval_length(self) -> float:
        return self.binding_quality_summary.mean_nonparametric_interval_length

    @property
    def quality_risk_binding_interval_length(self) -> float:
        return self.binding_interval_length

    @property
    def best_coverage(self) -> float:
        return self.best_quality_summary.mean_nonparametric_coverage

    @property
    def quality_risk_best_coverage(self) -> float:
        return self.best_coverage

    @property
    def best_coverage_floor_slack(self) -> float:
        return self.best_quality_summary.coverage_floor_slack

    @property
    def quality_risk_best_coverage_floor_slack(self) -> float:
        return self.best_coverage_floor_slack

    @property
    def best_rmse(self) -> float:
        return self.best_quality_summary.mean_nonparametric_rmse

    @property
    def quality_risk_best_rmse(self) -> float:
        return self.best_rmse

    @property
    def best_average_standard_error(self) -> float:
        return self.best_quality_summary.mean_nonparametric_average_standard_error

    @property
    def quality_risk_best_average_standard_error(self) -> float:
        return self.best_average_standard_error

    @property
    def best_standard_error_reserve(self) -> float:
        return self.best_quality_summary.standard_error_reserve

    @property
    def quality_risk_best_standard_error_reserve(self) -> float:
        return self.best_standard_error_reserve

    @property
    def best_interval_length(self) -> float:
        return self.best_quality_summary.mean_nonparametric_interval_length

    @property
    def quality_risk_best_interval_length(self) -> float:
        return self.best_interval_length

    @property
    def quality_risk_binding_rmse_to_standard_error_ratio(self) -> float:
        return self.binding_quality_summary.rmse_to_standard_error_ratio

    @property
    def quality_risk_best_rmse_to_standard_error_ratio(self) -> float:
        return self.best_quality_summary.rmse_to_standard_error_ratio

    @property
    def quality_risk_binding_interval_length_to_rmse_ratio(self) -> float:
        return self.binding_quality_summary.interval_length_to_rmse_ratio

    @property
    def quality_risk_best_interval_length_to_rmse_ratio(self) -> float:
        return self.best_quality_summary.interval_length_to_rmse_ratio

    @property
    def quality_risk_binding_interval_length_to_standard_error_ratio(self) -> float:
        return self.binding_quality_summary.interval_length_to_standard_error_ratio

    @property
    def quality_risk_best_interval_length_to_standard_error_ratio(self) -> float:
        return self.best_quality_summary.interval_length_to_standard_error_ratio

    @property
    def quality_risk_rmse_to_standard_error_ratio_gap(self) -> float:
        return self.rmse_to_standard_error_ratio_gap

    @property
    def quality_risk_standard_error_reserve_gap(self) -> float:
        return self.standard_error_reserve_gap

    @property
    def quality_risk_interval_length_to_rmse_gap(self) -> float:
        return self.interval_length_to_rmse_gap

    @property
    def rmse_outpaces_average_se_gap(self) -> float:
        return max(self.binding_rmse - self.binding_average_standard_error, 0.0)

    @property
    def quality_risk_rmse_outpaces_average_se_gap(self) -> float:
        return self.rmse_outpaces_average_se_gap

    @property
    def driver_metric_evidence(self) -> tuple[str, ...]:
        binding = self.binding_quality_summary
        best = self.best_quality_summary
        return (
            f"binding_driver={self.binding_driver}",
            (
                f"binding_design={_format_design_key(*binding.design_key)} "
                f"coverage_slack={_format_signed(binding.coverage_floor_slack)} "
                f"rmse_to_se={_format_unsigned(binding.rmse_to_standard_error_ratio)} "
                f"se_minus_rmse={_format_signed(binding.standard_error_reserve)} "
                f"interval_to_rmse={_format_unsigned(binding.interval_length_to_rmse_ratio)}"
            ),
            (
                f"best_design={_format_design_key(*best.design_key)} "
                f"coverage_slack={_format_signed(best.coverage_floor_slack)} "
                f"rmse_to_se={_format_unsigned(best.rmse_to_standard_error_ratio)} "
                f"se_minus_rmse={_format_signed(best.standard_error_reserve)} "
                f"interval_to_rmse={_format_unsigned(best.interval_length_to_rmse_ratio)}"
            ),
            (
                f"rmse_to_se_gap={_format_signed(self.rmse_to_standard_error_ratio_gap)} "
                f"se_minus_rmse_gap={_format_signed(self.standard_error_reserve_gap)} "
                f"interval_to_rmse_gap={_format_signed(self.interval_length_to_rmse_gap)}"
            ),
        )

    @property
    def raw_metric_evidence(self) -> tuple[str, ...]:
        binding = self.binding_quality_summary
        best = self.best_quality_summary
        return (
            (
                f"binding_raw={_format_design_key(*binding.design_key)} "
                f"coverage={_format_unsigned(binding.mean_nonparametric_coverage)} "
                f"rmse={_format_unsigned(binding.mean_nonparametric_rmse)} "
                f"average_se={_format_unsigned(binding.mean_nonparametric_average_standard_error)} "
                f"interval_length={_format_unsigned(binding.mean_nonparametric_interval_length)}"
            ),
            (
                f"best_raw={_format_design_key(*best.design_key)} "
                f"coverage={_format_unsigned(best.mean_nonparametric_coverage)} "
                f"rmse={_format_unsigned(best.mean_nonparametric_rmse)} "
                f"average_se={_format_unsigned(best.mean_nonparametric_average_standard_error)} "
                f"interval_length={_format_unsigned(best.mean_nonparametric_interval_length)}"
            ),
        )

    @property
    def interval_scale_status(self) -> str:
        return _interval_scale_status_for_summaries(
            self.binding_quality_summary,
            self.best_quality_summary,
        )

    @property
    def interval_scale_evidence(self) -> tuple[str, ...]:
        binding = self.binding_quality_summary
        best = self.best_quality_summary
        return (
            (
                "paper_90pct_pointwise_interval_to_se="
                f"{_format_unsigned(_PAPER_90PCT_POINTWISE_INTERVAL_TO_SE_MULTIPLIER)}"
            ),
            (
                f"binding_interval_to_se={_format_design_key(*binding.design_key)} "
                f"{_format_unsigned(binding.interval_length_to_standard_error_ratio)} "
                f"gap={binding.interval_length_to_standard_error_gap:+.3e}"
            ),
            (
                f"best_interval_to_se={_format_design_key(*best.design_key)} "
                f"{_format_unsigned(best.interval_length_to_standard_error_ratio)} "
                f"gap={best.interval_length_to_standard_error_gap:+.3e}"
            ),
            f"interval_scale_status={self.interval_scale_status}",
        )

    @property
    def quality_risk_clearance_evidence(self) -> tuple[str, ...]:
        conditions = self.quality_risk_clearance_conditions
        return (
            (
                "clearance coverage_floor "
                f"status={conditions[0]['status']} "
                f"deficit={conditions[0]['deficit']:.3f} "
                f"margin={conditions[0]['margin']:+.3f}"
            ),
            (
                "clearance nonnegative_se_reserve "
                f"status={conditions[1]['status']} "
                f"deficit={conditions[1]['deficit']:.3f} "
                f"margin={conditions[1]['margin']:+.3f}"
            ),
            (
                "clearance rmse_to_se_not_trailing "
                f"status={conditions[2]['status']} "
                f"deficit={conditions[2]['deficit']:.3f}"
            ),
            (
                "clearance se_reserve_not_trailing "
                f"status={conditions[3]['status']} "
                f"deficit={conditions[3]['deficit']:.3f}"
            ),
            (
                "clearance interval_to_rmse_not_trailing "
                f"status={conditions[4]['status']} "
                f"deficit={conditions[4]['deficit']:.3f}"
            ),
            (
                f"clearance overall status={conditions[5]['status']} "
                f"driver={conditions[5]['driver']}"
            ),
        )

    @property
    def quality_risk_clearance_conditions(self) -> tuple[dict[str, object], ...]:
        binding = self.binding_quality_summary
        return _quality_risk_clearance_conditions(
            binding_driver=self.binding_driver,
            binding_coverage_floor_slack=binding.coverage_floor_slack,
            binding_standard_error_reserve=binding.standard_error_reserve,
            rmse_to_standard_error_ratio_gap=self.rmse_to_standard_error_ratio_gap,
            standard_error_reserve_gap=self.standard_error_reserve_gap,
            interval_length_to_rmse_gap=self.interval_length_to_rmse_gap,
        )

    @property
    def quality_risk_calibration_targets(self) -> tuple[dict[str, object], ...]:
        binding = self.binding_quality_summary
        best = self.best_quality_summary
        return build_quality_risk_calibration_targets(
            binding_rmse=binding.mean_nonparametric_rmse,
            binding_average_standard_error=(
                binding.mean_nonparametric_average_standard_error
            ),
            binding_rmse_to_standard_error_ratio=(
                binding.rmse_to_standard_error_ratio
            ),
            binding_standard_error_reserve=binding.standard_error_reserve,
            binding_interval_length_to_rmse_ratio=(
                binding.interval_length_to_rmse_ratio
            ),
            best_rmse_to_standard_error_ratio=best.rmse_to_standard_error_ratio,
            best_standard_error_reserve=best.standard_error_reserve,
            best_interval_length_to_rmse_ratio=best.interval_length_to_rmse_ratio,
        )

    @property
    def dominant_quality_risk_calibration_target(self) -> dict[str, object]:
        return dominant_quality_risk_calibration_target(
            self.quality_risk_calibration_targets
        )

    @property
    def quality_risk_calibration_target_evidence(self) -> tuple[str, ...]:
        dominant = self.dominant_quality_risk_calibration_target
        return (
            *(
                "calibration target "
                f"{target['condition']} status={target['status']} "
                "required_average_se_lift="
                f"{float(target['required_average_standard_error_lift']):.3f} "
                "current_interval_length="
                f"{float(target['current_interval_length']):.3f} "
                "target_average_se="
                f"{float(target['target_average_standard_error']):.3f} "
                "target_interval_length="
                f"{float(target['target_interval_length']):.3f} "
                "required_interval_length_lift="
                f"{float(target['required_interval_length_lift']):.3f}"
                for target in self.quality_risk_calibration_targets
            ),
            "dominant calibration target "
            f"{dominant['condition']} "
            "required_average_se_lift="
            f"{float(dominant['required_average_standard_error_lift']):.3f} "
            "current_interval_length="
            f"{float(dominant['current_interval_length']):.3f} "
            "target_average_se="
            f"{float(dominant['target_average_standard_error']):.3f} "
            "target_interval_length="
            f"{float(dominant['target_interval_length']):.3f} "
            "required_interval_length_lift="
            f"{float(dominant['required_interval_length_lift']):.3f}",
        )

    @property
    def quality_risk_method_diagnosis(self) -> dict[str, object]:
        binding = self.binding_quality_summary
        best = self.best_quality_summary
        return build_quality_risk_method_diagnosis(
            binding_driver=self.binding_driver,
            blocker_reason=self.blocker_reason,
            interval_scale_status=self.interval_scale_status,
            binding_interval_length_to_standard_error_gap=(
                binding.interval_length_to_standard_error_gap
            ),
            best_interval_length_to_standard_error_gap=(
                best.interval_length_to_standard_error_gap
            ),
            dominant_calibration_target=self.dominant_quality_risk_calibration_target,
        )

    @property
    def quality_risk_method_diagnosis_evidence(self) -> tuple[str, ...]:
        diagnosis = self.quality_risk_method_diagnosis
        return (
            "method diagnosis "
            f"{diagnosis['diagnosis']} "
            f"interval_scale={diagnosis['interval_scale_status']} "
            "max_abs_interval_scale_gap="
            f"{float(diagnosis['max_abs_interval_scale_gap']):.3e}",
            "method dominant target "
            f"{diagnosis['dominant_condition']} "
            "current_average_se="
            f"{float(diagnosis['current_average_standard_error']):.3f} "
            "required_average_se_lift="
            f"{float(diagnosis['required_average_standard_error_lift']):.3f} "
            "lift_fraction_of_current="
            f"{float(diagnosis['required_average_standard_error_lift_fraction_of_current']):.3f} "
            "current_interval_length="
            f"{float(diagnosis['current_interval_length']):.3f} "
            "required_interval_length_lift="
            f"{float(diagnosis['required_interval_length_lift']):.3f} "
            "interval_lift_fraction_of_current="
            f"{float(diagnosis['required_interval_length_lift_fraction_of_current']):.3f} "
            "target_average_se="
            f"{float(diagnosis['target_average_standard_error']):.3f} "
            "target_interval_length="
            f"{float(diagnosis['target_interval_length']):.3f}",
        )

    @property
    def quality_risk_action_contract(self) -> dict[str, object]:
        return build_quality_risk_action_contract(
            binding_driver=self.binding_driver,
            interval_scale_status=self.interval_scale_status,
            dominant_calibration_target=self.dominant_quality_risk_calibration_target,
        )

    @property
    def quality_risk_action_evidence(self) -> tuple[str, ...]:
        action = self.quality_risk_action_contract
        return (
            f"action={action['action']}",
            f"interval_scale_action={action['interval_scale_action']}",
            f"bounded_policy_action={action['bounded_policy_action']}",
            f"release_gate_action={action['release_gate_action']}",
            f"target_condition={action['target_condition']}",
            "target_average_se="
            f"{float(action['target_average_standard_error']):.3f}",
            "target_interval_length="
            f"{float(action['target_interval_length']):.3f}",
            "required_average_se_lift="
            f"{float(action['required_average_standard_error_lift']):.3f}",
            "required_interval_length_lift="
            f"{float(action['required_interval_length_lift']):.3f}",
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate(
        self,
    ) -> dict[str, object]:
        return build_quality_risk_average_standard_error_calibration_candidate(
            binding_summary=self.binding_quality_summary,
            best_summary=self.best_quality_summary,
            dominant_calibration_target=self.dominant_quality_risk_calibration_target,
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_evidence(
        self,
    ) -> tuple[str, ...]:
        return build_quality_risk_average_standard_error_calibration_candidate_evidence(
            self.quality_risk_average_standard_error_calibration_candidate
        )

    @property
    def quality_risk_positive_headroom_estimator_evidence_requirement(
        self,
    ) -> dict[str, object]:
        return build_quality_risk_positive_headroom_estimator_evidence_requirement(
            self.quality_risk_average_standard_error_calibration_candidate
        )

    @property
    def positive_headroom_estimator_evidence_requirement(
        self,
    ) -> dict[str, object]:
        return self.quality_risk_positive_headroom_estimator_evidence_requirement

    @property
    def quality_risk_positive_headroom_estimator_evidence_verdict(
        self,
    ) -> dict[str, object]:
        return build_quality_risk_positive_headroom_estimator_evidence_verdict(
            self.quality_risk_positive_headroom_estimator_evidence_requirement,
            observed_average_standard_error=self.binding_average_standard_error,
            observed_interval_length=self.binding_interval_length,
        )

    @property
    def positive_headroom_estimator_evidence_verdict(
        self,
    ) -> dict[str, object]:
        return self.quality_risk_positive_headroom_estimator_evidence_verdict

    @property
    def quality_risk_positive_headroom_estimator_evidence_verdict_evidence(
        self,
    ) -> tuple[str, ...]:
        verdict = self.quality_risk_positive_headroom_estimator_evidence_verdict
        return (
            f"positive_headroom_estimator_evidence_status={verdict['status']}",
            "positive_headroom_estimator_evidence_source="
            f"{verdict['observed_evidence_source']} "
            f"status={verdict['observed_evidence_source_status']}",
            "positive_headroom_estimator_average_se "
            f"{float(verdict['observed_average_standard_error']):.3f}->"
            f"{float(verdict['target_average_standard_error']):.3f} "
            f"margin={float(verdict['average_standard_error_margin']):+.3f} "
            "remaining_fraction="
            f"{float(verdict['remaining_average_standard_error_gap_fraction']):.3f}",
            "positive_headroom_estimator_interval_length "
            f"{float(verdict['observed_interval_length']):.3f}->"
            f"{float(verdict['target_interval_length']):.3f} "
            f"margin={float(verdict['interval_length_margin']):+.3f} "
            "remaining_fraction="
            f"{float(verdict['remaining_interval_length_gap_fraction']):.3f}",
            "positive_headroom_estimator_interval_scale "
            f"{verdict['interval_scale_status']} "
            f"gap={float(verdict['interval_to_standard_error_gap']):+.3e}",
            "positive_headroom_estimator_next_evidence="
            f"{verdict['required_next_evidence']}",
        )

    @property
    def positive_headroom_estimator_evidence_verdict_evidence(
        self,
    ) -> tuple[str, ...]:
        return self.quality_risk_positive_headroom_estimator_evidence_verdict_evidence

    @property
    def positive_headroom_estimator_evidence_source_chain(self) -> tuple[str, ...]:
        verdict = self.quality_risk_positive_headroom_estimator_evidence_verdict
        return (
            f"paper_refs={'; '.join(self.paper_evidence_refs)}",
            f"r_refs={'; '.join(self.r_evidence_refs)}",
            f"python_refs={'; '.join(self.python_evidence_refs)}",
            (
                f"live_metric_boundary=binding_design={self.binding_design_label} "
                f"evidence_source={verdict['observed_evidence_source']} "
                f"source_status={verdict['observed_evidence_source_status']} "
                "average_se="
                f"{float(verdict['observed_average_standard_error']):.3f}->"
                f"{float(verdict['target_average_standard_error']):.3f} "
                "remaining_gap="
                f"{float(verdict['remaining_average_standard_error_gap']):.3f} "
                "interval_length="
                f"{float(verdict['observed_interval_length']):.3f}->"
                f"{float(verdict['target_interval_length']):.3f} "
                "remaining_gap="
                f"{float(verdict['remaining_interval_length_gap']):.3f} "
                f"interval_scale_status={verdict['interval_scale_status']}"
            ),
            (
                f"verdict={verdict['status']} "
                f"next_evidence={verdict['required_next_evidence']} "
                f"release_gate_effect={verdict['release_gate_effect']}"
            ),
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_status(self) -> str:
        return str(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "status",
            )
        )

    @property
    def candidate_status(self) -> str:
        return self.quality_risk_average_standard_error_calibration_candidate_status

    @property
    def average_standard_error_calibration_candidate_status(self) -> str:
        return self.quality_risk_average_standard_error_calibration_candidate_status

    @property
    def projected_quality_risk_status(self) -> str:
        return str(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "projected_quality_risk_status",
            )
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_target_condition(
        self,
    ) -> str:
        return str(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "target_condition",
            )
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_current_average_standard_error(
        self,
    ) -> float:
        return float(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "current_average_standard_error",
            )
        )

    @property
    def average_standard_error_calibration_candidate_current_average_standard_error(
        self,
    ) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_current_average_standard_error
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_target_average_standard_error(
        self,
    ) -> float:
        return float(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "target_average_standard_error",
            )
        )

    @property
    def average_standard_error_calibration_candidate_target_average_standard_error(
        self,
    ) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_target_average_standard_error
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_current_interval_length(
        self,
    ) -> float:
        return float(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "current_interval_length",
            )
        )

    @property
    def average_standard_error_calibration_candidate_current_interval_length(
        self,
    ) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_current_interval_length
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_target_interval_length(
        self,
    ) -> float:
        return float(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "target_interval_length",
            )
        )

    @property
    def average_standard_error_calibration_candidate_target_interval_length(
        self,
    ) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_target_interval_length
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_required_average_standard_error_lift(
        self,
    ) -> float:
        return float(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "required_average_standard_error_lift",
            )
        )

    @property
    def average_standard_error_calibration_candidate_required_average_standard_error_lift(
        self,
    ) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_required_average_standard_error_lift
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_required_interval_length_lift(
        self,
    ) -> float:
        return float(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "required_interval_length_lift",
            )
        )

    @property
    def average_standard_error_calibration_candidate_required_interval_length_lift(
        self,
    ) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_required_interval_length_lift
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_minimum_headroom_margin_after_calibration(
        self,
    ) -> float:
        return float(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "minimum_headroom_margin_after_calibration",
            )
        )

    @property
    def _quality_risk_average_standard_error_calibration_candidate_positive_headroom_target(
        self,
    ) -> dict[str, object]:
        return dict(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "positive_headroom_admission_target",
            )
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_average_standard_error(
        self,
    ) -> float:
        return float(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "minimum_admissible_average_standard_error"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_interval_length(
        self,
    ) -> float:
        return float(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "minimum_admissible_interval_length"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_average_standard_error_lift(
        self,
    ) -> float:
        return float(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "required_average_standard_error_lift"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_interval_length_lift(
        self,
    ) -> float:
        return float(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "required_interval_length_lift"
            ]
        )

    @property
    def quality_risk_average_standard_error_calibration_candidate_positive_headroom_margin(
        self,
    ) -> float:
        return float(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "minimum_positive_headroom_margin"
            ]
        )

    @property
    def quality_risk_positive_headroom_target_average_standard_error(self) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_requirement[
                "target_average_standard_error"
            ]
        )

    @property
    def positive_headroom_target_average_standard_error(self) -> float:
        return self.quality_risk_positive_headroom_target_average_standard_error

    @property
    def quality_risk_positive_headroom_target_interval_length(self) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_requirement[
                "target_interval_length"
            ]
        )

    @property
    def positive_headroom_target_interval_length(self) -> float:
        return self.quality_risk_positive_headroom_target_interval_length

    @property
    def quality_risk_positive_headroom_required_average_standard_error_lift(
        self,
    ) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_requirement[
                "remaining_average_standard_error_gap"
            ]
        )

    @property
    def positive_headroom_required_average_standard_error_lift(self) -> float:
        return self.quality_risk_positive_headroom_required_average_standard_error_lift

    @property
    def quality_risk_positive_headroom_required_interval_length_lift(self) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_requirement[
                "remaining_interval_length_gap"
            ]
        )

    @property
    def positive_headroom_required_interval_length_lift(self) -> float:
        return self.quality_risk_positive_headroom_required_interval_length_lift

    @property
    def quality_risk_positive_headroom_remaining_average_standard_error_gap(
        self,
    ) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "remaining_average_standard_error_gap"
            ]
        )

    @property
    def positive_headroom_remaining_average_standard_error_gap(self) -> float:
        return self.quality_risk_positive_headroom_remaining_average_standard_error_gap

    @property
    def quality_risk_positive_headroom_remaining_interval_length_gap(self) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "remaining_interval_length_gap"
            ]
        )

    @property
    def positive_headroom_remaining_interval_length_gap(self) -> float:
        return self.quality_risk_positive_headroom_remaining_interval_length_gap

    @property
    def quality_risk_positive_headroom_maximum_remaining_gap_fraction(self) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "maximum_remaining_gap_fraction"
            ]
        )

    @property
    def positive_headroom_maximum_remaining_gap_fraction(self) -> float:
        return self.quality_risk_positive_headroom_maximum_remaining_gap_fraction

    @property
    def quality_risk_positive_headroom_observed_average_standard_error(self) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "observed_average_standard_error"
            ]
        )

    @property
    def positive_headroom_observed_average_standard_error(self) -> float:
        return self.quality_risk_positive_headroom_observed_average_standard_error

    @property
    def quality_risk_positive_headroom_current_average_standard_error(self) -> float:
        return self.quality_risk_positive_headroom_observed_average_standard_error

    @property
    def positive_headroom_current_average_standard_error(self) -> float:
        return self.quality_risk_positive_headroom_current_average_standard_error

    @property
    def quality_risk_positive_headroom_observed_interval_length(self) -> float:
        return float(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "observed_interval_length"
            ]
        )

    @property
    def positive_headroom_observed_interval_length(self) -> float:
        return self.quality_risk_positive_headroom_observed_interval_length

    @property
    def quality_risk_positive_headroom_current_interval_length(self) -> float:
        return self.quality_risk_positive_headroom_observed_interval_length

    @property
    def positive_headroom_current_interval_length(self) -> float:
        return self.quality_risk_positive_headroom_current_interval_length

    @property
    def quality_risk_positive_headroom_margin(self) -> float:
        return (
            self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_margin
        )

    @property
    def quality_risk_positive_headroom_admission_status(self) -> str:
        return str(
            self._quality_risk_average_standard_error_calibration_candidate_positive_headroom_target[
                "admission_status"
            ]
        )

    @property
    def quality_risk_positive_headroom_feature_gate_rerun_admission_status(
        self,
    ) -> str:
        return str(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "feature_gate_rerun_admission_status"
            ]
        )

    @property
    def positive_headroom_feature_gate_rerun_admission_status(self) -> str:
        return self.quality_risk_positive_headroom_feature_gate_rerun_admission_status

    @property
    def quality_risk_positive_headroom_admissible_for_feature_gate_rerun(
        self,
    ) -> bool:
        return bool(
            self.quality_risk_positive_headroom_estimator_evidence_verdict[
                "admissible_for_feature_gate_rerun"
            ]
        )

    @property
    def quality_risk_positive_headroom_feature_gate_rerun_admissible(self) -> bool:
        return self.quality_risk_positive_headroom_admissible_for_feature_gate_rerun

    @property
    def positive_headroom_feature_gate_rerun_admissible(self) -> bool:
        return self.quality_risk_positive_headroom_feature_gate_rerun_admissible

    @property
    def positive_headroom_admissible_for_feature_gate_rerun(self) -> bool:
        return self.quality_risk_positive_headroom_admissible_for_feature_gate_rerun

    @property
    def feature_gate_rerun_admission_status(self) -> str:
        return self.quality_risk_positive_headroom_feature_gate_rerun_admission_status

    @property
    def feature_gate_rerun_admissible(self) -> bool:
        return self.quality_risk_positive_headroom_feature_gate_rerun_admissible

    @property
    def admissible_for_feature_gate_rerun(self) -> bool:
        return self.quality_risk_positive_headroom_admissible_for_feature_gate_rerun

    @property
    def quality_risk_average_standard_error_calibration_candidate_release_gate_effect(
        self,
    ) -> str:
        return str(
            _calibration_candidate_value(
                self.quality_risk_average_standard_error_calibration_candidate,
                "release_gate_effect",
            )
        )

    @property
    def driver(self) -> str:
        return self.binding_driver

    @property
    def quality_risk_driver(self) -> str:
        return self.binding_driver

    @property
    def quality_risk_binding_driver(self) -> str:
        return self.binding_driver

    @property
    def quality_risk_blocker_reason(self) -> str:
        return self.blocker_reason

    @property
    def quality_risk_status(self) -> str:
        return "blocked" if self.blocker_reason else "cleared"

    @property
    def status(self) -> str:
        return self.quality_risk_status

    @property
    def quality_risk_binding_design(self) -> tuple[str, int, int]:
        return self.binding_design

    @property
    def quality_risk_best_design(self) -> tuple[str, int, int]:
        return self.best_design

    @property
    def binding_design_key(self) -> tuple[str, int, int]:
        return self.binding_design_summary.design_key

    @property
    def best_design_key(self) -> tuple[str, int, int]:
        return self.best_design_summary.design_key

    @property
    def quality_risk_binding_design_key(self) -> tuple[str, int, int]:
        return self.binding_design_key

    @property
    def quality_risk_best_design_key(self) -> tuple[str, int, int]:
        return self.best_design_key

    @property
    def quality_risk_route(self) -> str:
        if self.binding_driver == "quality-risk-cleared":
            return "quality-risk-cleared"
        return "calibration / interval-construction debt"

    @property
    def quality_risk_resolution(self) -> str:
        return _quality_risk_resolution_for_driver(self.binding_driver)

    @property
    def quality_risk_resolution_evidence(self) -> tuple[str, ...]:
        return (
            f"resolution={self.quality_risk_resolution}",
            (
                f"driver={self.binding_driver} "
                f"blocker_reason={self.blocker_reason or 'none'}"
            ),
            (
                f"paper_objects={'; '.join(self.paper_object_contract)}"
            ),
            (
                f"paper_refs={'; '.join(self.paper_evidence_refs)}"
            ),
            (
                f"metric_gaps=rmse_to_se:{_format_signed(self.rmse_to_standard_error_ratio_gap)} "
                f"se_minus_rmse:{_format_signed(self.standard_error_reserve_gap)} "
                f"interval_to_rmse:{_format_signed(self.interval_length_to_rmse_gap)}"
            ),
            *self.quality_risk_clearance_evidence,
            *self.quality_risk_calibration_target_evidence,
        )

    @property
    def binding_design_label(self) -> str:
        return _format_design_key(*self.binding_design)

    @property
    def best_design_label(self) -> str:
        return _format_design_key(*self.best_design)

    @property
    def quality_risk_binding_design_label(self) -> str:
        return self.binding_design_label

    @property
    def quality_risk_best_design_label(self) -> str:
        return self.best_design_label

    @property
    def paper_object_contract(self) -> tuple[str, ...]:
        return _PAPER_OBJECT_CONTRACT

    @property
    def paper_evidence_refs(self) -> tuple[str, ...]:
        return _PAPER_EVIDENCE_REFS

    @property
    def r_evidence_refs(self) -> tuple[str, ...]:
        return _R_EVIDENCE_REFS

    @property
    def python_evidence_refs(self) -> tuple[str, ...]:
        return _PYTHON_EVIDENCE_REFS

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "binding_design_key": list(self.binding_design_key),
            "binding_design_label": self.binding_design_label,
            "quality_risk_binding_design": list(self.quality_risk_binding_design),
            "quality_risk_binding_design_key": list(
                self.quality_risk_binding_design_key
            ),
            "quality_risk_binding_design_label": (
                self.quality_risk_binding_design_label
            ),
            "binding_design_summary": self.binding_design_summary.to_dict(),
            "binding_raw_metrics": {
                "coverage": self.binding_coverage,
                "rmse": self.binding_rmse,
                "average_standard_error": self.binding_average_standard_error,
                "standard_error_reserve": self.binding_standard_error_reserve,
                "interval_length": self.binding_interval_length,
            },
            "quality_risk_binding_coverage": self.quality_risk_binding_coverage,
            "quality_risk_binding_rmse": self.quality_risk_binding_rmse,
            "quality_risk_binding_average_standard_error": (
                self.quality_risk_binding_average_standard_error
            ),
            "quality_risk_binding_interval_length": (
                self.quality_risk_binding_interval_length
            ),
            "quality_risk_binding_rmse_to_standard_error_ratio": (
                self.quality_risk_binding_rmse_to_standard_error_ratio
            ),
            "quality_risk_binding_interval_length_to_rmse_ratio": (
                self.quality_risk_binding_interval_length_to_rmse_ratio
            ),
            "quality_risk_binding_interval_length_to_standard_error_ratio": (
                self.quality_risk_binding_interval_length_to_standard_error_ratio
            ),
            "coverage_floor_slack": self.coverage_floor_slack,
            "binding_coverage_floor_slack": self.binding_coverage_floor_slack,
            "quality_risk_coverage_floor_slack": (
                self.quality_risk_coverage_floor_slack
            ),
            "quality_risk_binding_coverage_floor_slack": (
                self.quality_risk_binding_coverage_floor_slack
            ),
            "quality_risk_binding_standard_error_reserve": (
                self.quality_risk_binding_standard_error_reserve
            ),
            "best_design": list(self.best_design),
            "best_design_key": list(self.best_design_key),
            "best_design_label": self.best_design_label,
            "quality_risk_best_design": list(self.quality_risk_best_design),
            "quality_risk_best_design_key": list(
                self.quality_risk_best_design_key
            ),
            "quality_risk_best_design_label": self.quality_risk_best_design_label,
            "best_design_summary": self.best_design_summary.to_dict(),
            "best_raw_metrics": {
                "coverage": self.best_coverage,
                "rmse": self.best_rmse,
                "average_standard_error": self.best_average_standard_error,
                "standard_error_reserve": self.best_standard_error_reserve,
                "interval_length": self.best_interval_length,
            },
            "quality_risk_best_coverage": self.quality_risk_best_coverage,
            "quality_risk_best_rmse": self.quality_risk_best_rmse,
            "quality_risk_best_average_standard_error": (
                self.quality_risk_best_average_standard_error
            ),
            "quality_risk_best_interval_length": (
                self.quality_risk_best_interval_length
            ),
            "quality_risk_best_rmse_to_standard_error_ratio": (
                self.quality_risk_best_rmse_to_standard_error_ratio
            ),
            "quality_risk_best_interval_length_to_rmse_ratio": (
                self.quality_risk_best_interval_length_to_rmse_ratio
            ),
            "quality_risk_best_interval_length_to_standard_error_ratio": (
                self.quality_risk_best_interval_length_to_standard_error_ratio
            ),
            "best_coverage_floor_slack": self.best_coverage_floor_slack,
            "quality_risk_best_coverage_floor_slack": (
                self.quality_risk_best_coverage_floor_slack
            ),
            "quality_risk_best_standard_error_reserve": (
                self.quality_risk_best_standard_error_reserve
            ),
            "binding_driver": self.binding_driver,
            "driver": self.driver,
            "quality_risk_driver": self.quality_risk_driver,
            "quality_risk_binding_driver": self.quality_risk_binding_driver,
            "quality_risk_route": self.quality_risk_route,
            "quality_risk_resolution": self.quality_risk_resolution,
            "quality_risk_resolution_evidence": list(
                self.quality_risk_resolution_evidence
            ),
            "status": self.status,
            "quality_risk_status": self.quality_risk_status,
            "blocker_reason": self.blocker_reason,
            "quality_risk_blocker_reason": self.quality_risk_blocker_reason,
            "supported_floor_ceiling": self.supported_floor_ceiling,
            "canonical_floor_shortfall": self.canonical_floor_shortfall,
            "quality_risk_canonical_floor_shortfall": (
                self.quality_risk_canonical_floor_shortfall
            ),
            "rmse_to_standard_error_ratio_gap": self.rmse_to_standard_error_ratio_gap,
            "standard_error_reserve_gap": self.standard_error_reserve_gap,
            "interval_length_to_rmse_gap": self.interval_length_to_rmse_gap,
            "quality_risk_rmse_to_standard_error_ratio_gap": (
                self.quality_risk_rmse_to_standard_error_ratio_gap
            ),
            "quality_risk_standard_error_reserve_gap": (
                self.quality_risk_standard_error_reserve_gap
            ),
            "quality_risk_interval_length_to_rmse_gap": (
                self.quality_risk_interval_length_to_rmse_gap
            ),
            "rmse_outpaces_average_se_gap": self.rmse_outpaces_average_se_gap,
            "quality_risk_rmse_outpaces_average_se_gap": (
                self.quality_risk_rmse_outpaces_average_se_gap
            ),
            "design_quality_summaries": [
                summary.to_dict() for summary in self.design_quality_summaries
            ],
            "driver_metric_evidence": list(self.driver_metric_evidence),
            "raw_metric_evidence": list(self.raw_metric_evidence),
            "interval_scale_status": self.interval_scale_status,
            "interval_scale_evidence": list(self.interval_scale_evidence),
            "quality_risk_clearance_evidence": list(
                self.quality_risk_clearance_evidence
            ),
            "quality_risk_clearance_conditions": list(
                self.quality_risk_clearance_conditions
            ),
            "quality_risk_calibration_targets": list(
                self.quality_risk_calibration_targets
            ),
            "dominant_quality_risk_calibration_target": dict(
                self.dominant_quality_risk_calibration_target
            ),
            "quality_risk_calibration_target_evidence": list(
                self.quality_risk_calibration_target_evidence
            ),
            "quality_risk_method_diagnosis": dict(
                self.quality_risk_method_diagnosis
            ),
            "quality_risk_method_diagnosis_evidence": list(
                self.quality_risk_method_diagnosis_evidence
            ),
            "quality_risk_action_contract": dict(self.quality_risk_action_contract),
            "quality_risk_action_evidence": list(self.quality_risk_action_evidence),
            "quality_risk_average_standard_error_calibration_candidate": dict(
                self.quality_risk_average_standard_error_calibration_candidate
            ),
            "quality_risk_average_standard_error_calibration_candidate_status": (
                self.quality_risk_average_standard_error_calibration_candidate_status
            ),
            "average_standard_error_calibration_candidate_status": (
                self.average_standard_error_calibration_candidate_status
            ),
            "quality_risk_average_standard_error_calibration_candidate_target_condition": (
                self.quality_risk_average_standard_error_calibration_candidate_target_condition
            ),
            "quality_risk_average_standard_error_calibration_candidate_current_average_standard_error": (
                self.quality_risk_average_standard_error_calibration_candidate_current_average_standard_error
            ),
            "average_standard_error_calibration_candidate_current_average_standard_error": (
                self.average_standard_error_calibration_candidate_current_average_standard_error
            ),
            "quality_risk_average_standard_error_calibration_candidate_target_average_standard_error": (
                self.quality_risk_average_standard_error_calibration_candidate_target_average_standard_error
            ),
            "average_standard_error_calibration_candidate_target_average_standard_error": (
                self.average_standard_error_calibration_candidate_target_average_standard_error
            ),
            "quality_risk_average_standard_error_calibration_candidate_current_interval_length": (
                self.quality_risk_average_standard_error_calibration_candidate_current_interval_length
            ),
            "average_standard_error_calibration_candidate_current_interval_length": (
                self.average_standard_error_calibration_candidate_current_interval_length
            ),
            "quality_risk_average_standard_error_calibration_candidate_target_interval_length": (
                self.quality_risk_average_standard_error_calibration_candidate_target_interval_length
            ),
            "average_standard_error_calibration_candidate_target_interval_length": (
                self.average_standard_error_calibration_candidate_target_interval_length
            ),
            "quality_risk_average_standard_error_calibration_candidate_required_average_standard_error_lift": (
                self.quality_risk_average_standard_error_calibration_candidate_required_average_standard_error_lift
            ),
            "average_standard_error_calibration_candidate_required_average_standard_error_lift": (
                self.average_standard_error_calibration_candidate_required_average_standard_error_lift
            ),
            "quality_risk_average_standard_error_calibration_candidate_required_interval_length_lift": (
                self.quality_risk_average_standard_error_calibration_candidate_required_interval_length_lift
            ),
            "average_standard_error_calibration_candidate_required_interval_length_lift": (
                self.average_standard_error_calibration_candidate_required_interval_length_lift
            ),
            "quality_risk_average_standard_error_calibration_candidate_minimum_headroom_margin_after_calibration": (
                self.quality_risk_average_standard_error_calibration_candidate_minimum_headroom_margin_after_calibration
            ),
            "quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_average_standard_error": (
                self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_average_standard_error
            ),
            "quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_interval_length": (
                self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_target_interval_length
            ),
            "quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_average_standard_error_lift": (
                self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_average_standard_error_lift
            ),
            "quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_interval_length_lift": (
                self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_required_interval_length_lift
            ),
            "quality_risk_average_standard_error_calibration_candidate_positive_headroom_margin": (
                self.quality_risk_average_standard_error_calibration_candidate_positive_headroom_margin
            ),
            "quality_risk_positive_headroom_target_average_standard_error": (
                self.quality_risk_positive_headroom_target_average_standard_error
            ),
            "positive_headroom_target_average_standard_error": (
                self.positive_headroom_target_average_standard_error
            ),
            "quality_risk_positive_headroom_target_interval_length": (
                self.quality_risk_positive_headroom_target_interval_length
            ),
            "positive_headroom_target_interval_length": (
                self.positive_headroom_target_interval_length
            ),
            "quality_risk_positive_headroom_required_average_standard_error_lift": (
                self.quality_risk_positive_headroom_required_average_standard_error_lift
            ),
            "positive_headroom_required_average_standard_error_lift": (
                self.positive_headroom_required_average_standard_error_lift
            ),
            "quality_risk_positive_headroom_required_interval_length_lift": (
                self.quality_risk_positive_headroom_required_interval_length_lift
            ),
            "positive_headroom_required_interval_length_lift": (
                self.positive_headroom_required_interval_length_lift
            ),
            "quality_risk_positive_headroom_remaining_average_standard_error_gap": (
                self.quality_risk_positive_headroom_remaining_average_standard_error_gap
            ),
            "positive_headroom_remaining_average_standard_error_gap": (
                self.positive_headroom_remaining_average_standard_error_gap
            ),
            "quality_risk_positive_headroom_remaining_interval_length_gap": (
                self.quality_risk_positive_headroom_remaining_interval_length_gap
            ),
            "positive_headroom_remaining_interval_length_gap": (
                self.positive_headroom_remaining_interval_length_gap
            ),
            "quality_risk_positive_headroom_maximum_remaining_gap_fraction": (
                self.quality_risk_positive_headroom_maximum_remaining_gap_fraction
            ),
            "positive_headroom_maximum_remaining_gap_fraction": (
                self.positive_headroom_maximum_remaining_gap_fraction
            ),
            "quality_risk_positive_headroom_observed_average_standard_error": (
                self.quality_risk_positive_headroom_observed_average_standard_error
            ),
            "positive_headroom_observed_average_standard_error": (
                self.positive_headroom_observed_average_standard_error
            ),
            "quality_risk_positive_headroom_current_average_standard_error": (
                self.quality_risk_positive_headroom_current_average_standard_error
            ),
            "positive_headroom_current_average_standard_error": (
                self.positive_headroom_current_average_standard_error
            ),
            "quality_risk_positive_headroom_observed_interval_length": (
                self.quality_risk_positive_headroom_observed_interval_length
            ),
            "positive_headroom_observed_interval_length": (
                self.positive_headroom_observed_interval_length
            ),
            "quality_risk_positive_headroom_current_interval_length": (
                self.quality_risk_positive_headroom_current_interval_length
            ),
            "positive_headroom_current_interval_length": (
                self.positive_headroom_current_interval_length
            ),
            "quality_risk_positive_headroom_margin": (
                self.quality_risk_positive_headroom_margin
            ),
            "quality_risk_positive_headroom_admission_status": (
                self.quality_risk_positive_headroom_admission_status
            ),
            "quality_risk_average_standard_error_calibration_candidate_release_gate_effect": (
                self.quality_risk_average_standard_error_calibration_candidate_release_gate_effect
            ),
            "quality_risk_average_standard_error_calibration_candidate_evidence": list(
                self.quality_risk_average_standard_error_calibration_candidate_evidence
            ),
            "positive_headroom_estimator_evidence_requirement": dict(
                self.quality_risk_positive_headroom_estimator_evidence_requirement
            ),
            "quality_risk_positive_headroom_estimator_evidence_requirement": dict(
                self.quality_risk_positive_headroom_estimator_evidence_requirement
            ),
            "positive_headroom_estimator_evidence_verdict": dict(
                self.quality_risk_positive_headroom_estimator_evidence_verdict
            ),
            "quality_risk_positive_headroom_estimator_evidence_verdict": dict(
                self.quality_risk_positive_headroom_estimator_evidence_verdict
            ),
            "quality_risk_positive_headroom_estimator_evidence_verdict_evidence": list(
                self.quality_risk_positive_headroom_estimator_evidence_verdict_evidence
            ),
            "positive_headroom_estimator_evidence_verdict_evidence": list(
                self.positive_headroom_estimator_evidence_verdict_evidence
            ),
            "positive_headroom_estimator_evidence_source_chain": list(
                self.positive_headroom_estimator_evidence_source_chain
            ),
            "quality_risk_positive_headroom_feature_gate_rerun_admission_status": (
                self.quality_risk_positive_headroom_feature_gate_rerun_admission_status
            ),
            "positive_headroom_feature_gate_rerun_admission_status": (
                self.positive_headroom_feature_gate_rerun_admission_status
            ),
            "quality_risk_positive_headroom_admissible_for_feature_gate_rerun": (
                self.quality_risk_positive_headroom_admissible_for_feature_gate_rerun
            ),
            "quality_risk_positive_headroom_feature_gate_rerun_admissible": (
                self.quality_risk_positive_headroom_feature_gate_rerun_admissible
            ),
            "positive_headroom_feature_gate_rerun_admissible": (
                self.positive_headroom_feature_gate_rerun_admissible
            ),
            "positive_headroom_admissible_for_feature_gate_rerun": (
                self.positive_headroom_admissible_for_feature_gate_rerun
            ),
            "feature_gate_rerun_admission_status": (
                self.feature_gate_rerun_admission_status
            ),
            "feature_gate_rerun_admissible": self.feature_gate_rerun_admissible,
            "admissible_for_feature_gate_rerun": (
                self.admissible_for_feature_gate_rerun
            ),
            "paper_object_contract": list(self.paper_object_contract),
            "paper_evidence_refs": list(self.paper_evidence_refs),
            "r_evidence_refs": list(self.r_evidence_refs),
            "python_evidence_refs": list(self.python_evidence_refs),
            "canonical_quality_risk_digest": list(self.canonical_quality_risk_digest),
            "quality_risk_source_mode": self.quality_risk_source_mode,
            "quality_risk_source_note": self.quality_risk_source_note,
        }


def _summary_map(
    runtime_probe: MonteCarloRuntimeProbeReport,
) -> dict[tuple[str, int, int], MonteCarloRuntimeProbeDesignSummary]:
    return {
        (
            summary.dgp_name,
            _validated_positive_integer(summary.n_obs, label="n_obs"),
            _validated_positive_integer(summary.p, label="p"),
        ): summary
        for summary in runtime_probe.design_summaries
    }


def _build_quality_summary(
    summary: MonteCarloRuntimeProbeDesignSummary,
    *,
    coverage_floor: float,
) -> Phase7MonteCarloWideningPolicyQualityRiskDesignSummary:
    coverage = summary.mean_nonparametric_coverage
    rmse = summary.mean_nonparametric_rmse
    standard_error = summary.mean_nonparametric_average_standard_error
    interval_length = summary.mean_nonparametric_interval_length
    if (
        coverage is None
        or rmse is None
        or standard_error is None
        or interval_length is None
    ):
        raise ValueError(
            "quality risk probe requires nonparametric rmse, average standard error, and interval length"
        )
    rmse_value = float(rmse)
    standard_error_value = float(standard_error)
    interval_length_value = float(interval_length)
    return Phase7MonteCarloWideningPolicyQualityRiskDesignSummary(
        dgp_name=summary.dgp_name,
        n_obs=summary.n_obs,
        p=summary.p,
        mean_nonparametric_coverage=coverage,
        coverage_floor_slack=float(coverage) - float(coverage_floor),
        mean_nonparametric_rmse=rmse_value,
        mean_nonparametric_average_standard_error=standard_error_value,
        mean_nonparametric_interval_length=interval_length_value,
        rmse_to_standard_error_ratio=_positive_ratio(rmse_value, standard_error_value),
        standard_error_reserve=standard_error_value - rmse_value,
        interval_length_to_rmse_ratio=_positive_ratio(
            interval_length_value,
            rmse_value,
        ),
    )


def _build_repo_side_quality_summary(
    design_key: tuple[str, int, int],
    values: dict[str, float],
) -> Phase7MonteCarloWideningPolicyQualityRiskDesignSummary:
    rmse = float(values["mean_nonparametric_rmse"])
    standard_error = float(values["mean_nonparametric_average_standard_error"])
    interval_length = float(values["mean_nonparametric_interval_length"])
    return Phase7MonteCarloWideningPolicyQualityRiskDesignSummary(
        dgp_name=design_key[0],
        n_obs=design_key[1],
        p=design_key[2],
        mean_nonparametric_coverage=float(values["mean_nonparametric_coverage"]),
        coverage_floor_slack=float(values["coverage_floor_slack"]),
        mean_nonparametric_rmse=rmse,
        mean_nonparametric_average_standard_error=standard_error,
        mean_nonparametric_interval_length=interval_length,
        rmse_to_standard_error_ratio=_positive_ratio(rmse, standard_error),
        standard_error_reserve=standard_error - rmse,
        interval_length_to_rmse_ratio=_positive_ratio(interval_length, rmse),
    )


def _build_repo_side_quality_risk_probe_report(
    *,
    policy: Phase7MonteCarloWideningPolicy,
) -> Phase7MonteCarloWideningPolicyQualityRiskProbeReport:
    coverage_floor = float(policy.min_nonparametric_coverage)
    binding_design, best_design = _repo_side_quality_risk_boundary_state()
    if (
        binding_design not in _REPO_SIDE_QUALITY_SUMMARIES
        or best_design not in _REPO_SIDE_QUALITY_SUMMARIES
    ):
        raise ValueError(
            "repo-side quality risk fallback lacks current bounded design metrics"
        )
    binding_summary = _build_repo_side_quality_summary(
        binding_design,
        _REPO_SIDE_QUALITY_SUMMARIES[binding_design],
    )
    best_summary = _build_repo_side_quality_summary(
        best_design,
        _REPO_SIDE_QUALITY_SUMMARIES[best_design],
    )
    binding_driver = _binding_driver(binding_summary, best_summary)
    rmse_ratio_gap = (
        binding_summary.rmse_to_standard_error_ratio
        - best_summary.rmse_to_standard_error_ratio
    )
    standard_error_reserve_gap = (
        best_summary.standard_error_reserve - binding_summary.standard_error_reserve
    )
    interval_length_to_rmse_gap = (
        best_summary.interval_length_to_rmse_ratio
        - binding_summary.interval_length_to_rmse_ratio
    )
    supported_floor_ceiling = binding_summary.mean_nonparametric_coverage
    canonical_floor_shortfall = max(coverage_floor - supported_floor_ceiling, 0.0)
    blocker_reason = _blocker_reason_for_driver(binding_driver)
    blocker_digest = (
        "- quality risk status: `quality-risk-cleared`; this report would allow "
        "the Monte Carlo validation gate to promote when runtime admission is closed"
        if not blocker_reason
        else "- blocker reason: "
        "`quality-risk-keeps-trigger2-bounded`; this report preserves the current "
        "Monte Carlo validation blocker packet when live runtime input is unavailable"
    )
    canonical_quality_risk_digest = (
        "- bounded-slice quality risk: "
        f"`{_format_design_key(*binding_summary.design_key)}` coverage "
        f"`{binding_summary.mean_nonparametric_coverage:.3f} "
        f"({_format_signed(binding_summary.coverage_floor_slack)})`, "
        f"`RMSE/SE = {binding_summary.rmse_to_standard_error_ratio:.3f}`, "
        f"`SE-RMSE = {_format_signed(binding_summary.standard_error_reserve)}`, "
        f"`interval/RMSE = {binding_summary.interval_length_to_rmse_ratio:.3f}`; "
        f"comparison `{_format_design_key(*best_summary.design_key)}` coverage "
        f"`{best_summary.mean_nonparametric_coverage:.3f} "
        f"({_format_signed(best_summary.coverage_floor_slack)})`, "
        f"`RMSE/SE = {best_summary.rmse_to_standard_error_ratio:.3f}`, "
        f"`SE-RMSE = {_format_signed(best_summary.standard_error_reserve)}`, "
        f"`interval/RMSE = {best_summary.interval_length_to_rmse_ratio:.3f}`",
        "- binding driver: "
            f"`{binding_driver}`; supported floor ceiling "
            f"`{supported_floor_ceiling:.3f}`; canonical floor "
            f"`{coverage_floor:.3f}`; shortfall "
            f"`{canonical_floor_shortfall:.3f}`",
        blocker_digest,
        "- calibration headroom gap versus best bounded design: "
        f"`RMSE/SE gap = {_format_signed(rmse_ratio_gap)}`, "
        f"`SE-RMSE gap = {_format_signed(standard_error_reserve_gap)}`, "
        f"`interval/RMSE gap = {_format_signed(interval_length_to_rmse_gap)}`",
        _calibration_target_digest_line(binding_summary, best_summary),
        "- quality-risk resolution: "
        f"`{_quality_risk_resolution_for_driver(binding_driver)}`; "
        "the resolution is derived from the binding driver and paper Monte Carlo "
        "quality objects, not from closeout text",
        "- paper object contract: "
        + "; ".join(f"`{item}`" for item in _PAPER_OBJECT_CONTRACT),
        "- paper evidence refs: "
        + "; ".join(f"`{item}`" for item in _PAPER_EVIDENCE_REFS),
        _interval_scale_digest_line(binding_summary, best_summary),
        "- repo-side canonical blocker: live quality-risk floor-slack input drifted, "
        "so the public runner preserves the feature-completion gate blocker packet",
    )
    return Phase7MonteCarloWideningPolicyQualityRiskProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-quality-risk-probe",
        policy_digest=policy.to_digest(),
        binding_design=binding_summary.design_key,
        best_design=best_summary.design_key,
        binding_driver=binding_driver,
        blocker_reason=blocker_reason,
        supported_floor_ceiling=supported_floor_ceiling,
        canonical_floor_shortfall=canonical_floor_shortfall,
        rmse_to_standard_error_ratio_gap=rmse_ratio_gap,
        standard_error_reserve_gap=standard_error_reserve_gap,
        interval_length_to_rmse_gap=interval_length_to_rmse_gap,
        design_quality_summaries=(binding_summary, best_summary),
        canonical_quality_risk_digest=canonical_quality_risk_digest,
        quality_risk_source_mode="repo-side-canonical-blocker",
        quality_risk_source_note=(
            "repo-side canonical blocker preserves the public packet when live "
            "floor-slack input drifts"
        ),
    )


def _binding_driver(
    binding_summary: Phase7MonteCarloWideningPolicyQualityRiskDesignSummary,
    best_summary: Phase7MonteCarloWideningPolicyQualityRiskDesignSummary,
) -> str:
    if binding_summary.standard_error_reserve < 0.0:
        return "rmse-outpaces-average-se"
    if binding_summary.coverage_floor_slack < 0.0:
        return "coverage-floor-risk-persists"
    if (
        binding_summary.interval_length_to_rmse_ratio
        < best_summary.interval_length_to_rmse_ratio
    ):
        return "interval-per-rmse-headroom-trails-best-design"
    if binding_summary.standard_error_reserve < best_summary.standard_error_reserve:
        return "standard-error-reserve-trails-best-design"
    if (
        binding_summary.rmse_to_standard_error_ratio
        > best_summary.rmse_to_standard_error_ratio
    ):
        return "rmse-to-standard-error-ratio-trails-best-design"
    return "quality-risk-cleared"


def _blocker_reason_for_driver(binding_driver: str) -> str:
    if binding_driver == "quality-risk-cleared":
        return ""
    return "quality-risk-keeps-trigger2-bounded"


def _quality_risk_resolution_for_driver(binding_driver: str) -> str:
    if binding_driver == "quality-risk-cleared":
        return "quality-risk-cleared"
    if binding_driver == "coverage-floor-risk-persists":
        return "repair-required-coverage-floor-risk"
    if binding_driver == "rmse-outpaces-average-se":
        return "repair-required-standard-error-scale"
    return "paper-backed-calibration-debt-explanation"


def build_phase7_monte_carlo_widening_policy_quality_risk_probe_report(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicyQualityRiskProbeReport:
    resolved_policy = (
        build_phase7_canonical_monte_carlo_widening_policy()
        if policy is None
        else policy
    )
    if resolved_policy.min_nonparametric_coverage is None:
        raise ValueError("quality risk probe requires min_nonparametric_coverage")

    floor_slack_probe = (
        build_phase7_monte_carlo_widening_policy_floor_slack_probe_report(
            runtime_probe,
            policy=resolved_policy,
        )
    )
    summary_map = _summary_map(runtime_probe)
    binding_design = floor_slack_probe.binding_design
    best_design = floor_slack_probe.best_design
    if binding_design not in summary_map or best_design not in summary_map:
        raise KeyError("quality risk probe requires binding and best bounded designs")

    coverage_floor = float(resolved_policy.min_nonparametric_coverage)
    binding_summary = _build_quality_summary(
        summary_map[binding_design],
        coverage_floor=coverage_floor,
    )
    best_summary = _build_quality_summary(
        summary_map[best_design],
        coverage_floor=coverage_floor,
    )
    binding_driver = _binding_driver(binding_summary, best_summary)
    blocker_reason = _blocker_reason_for_driver(binding_driver)
    rmse_ratio_gap = (
        binding_summary.rmse_to_standard_error_ratio
        - best_summary.rmse_to_standard_error_ratio
    )
    standard_error_reserve_gap = (
        best_summary.standard_error_reserve - binding_summary.standard_error_reserve
    )
    interval_length_to_rmse_gap = (
        best_summary.interval_length_to_rmse_ratio
        - binding_summary.interval_length_to_rmse_ratio
    )

    blocker_digest = (
        "- quality risk status: `quality-risk-cleared`; the public probe has no "
        "post-admission quality blocker when runtime admission is closed"
        if not blocker_reason
        else "- blocker reason: "
        "`quality-risk-keeps-trigger2-bounded`; the public probe stays on the "
        "same blocked Monte Carlo validation surface"
    )

    canonical_quality_risk_digest = (
        "- bounded-slice quality risk: "
        f"`{_format_design_key(*binding_summary.design_key)}` coverage "
        f"`{binding_summary.mean_nonparametric_coverage:.3f} "
        f"({_format_signed(binding_summary.coverage_floor_slack)})`, "
        f"`RMSE/SE = {binding_summary.rmse_to_standard_error_ratio:.3f}`, "
        f"`SE-RMSE = {_format_signed(binding_summary.standard_error_reserve)}`, "
        f"`interval/RMSE = {binding_summary.interval_length_to_rmse_ratio:.3f}`; "
        f"comparison `{_format_design_key(*best_summary.design_key)}` coverage "
        f"`{best_summary.mean_nonparametric_coverage:.3f} "
        f"({_format_signed(best_summary.coverage_floor_slack)})`, "
        f"`RMSE/SE = {best_summary.rmse_to_standard_error_ratio:.3f}`, "
        f"`SE-RMSE = {_format_signed(best_summary.standard_error_reserve)}`, "
        f"`interval/RMSE = {best_summary.interval_length_to_rmse_ratio:.3f}`",
        "- binding driver: "
        f"`{binding_driver}`; supported floor ceiling "
        f"`{floor_slack_probe.supported_floor_ceiling:.3f}`; canonical floor "
        f"`{coverage_floor:.3f}`; shortfall "
        f"`{floor_slack_probe.canonical_floor_shortfall:.3f}`",
        blocker_digest,
        "- calibration headroom gap versus best bounded design: "
        f"`RMSE/SE gap = {_format_signed(rmse_ratio_gap)}`, "
        f"`SE-RMSE gap = {_format_signed(standard_error_reserve_gap)}`, "
        f"`interval/RMSE gap = {_format_signed(interval_length_to_rmse_gap)}`",
        _calibration_target_digest_line(binding_summary, best_summary),
        "- quality-risk resolution: "
        f"`{_quality_risk_resolution_for_driver(binding_driver)}`; "
        "the resolution is derived from the binding driver and paper Monte Carlo "
        "quality objects, not from closeout text",
        "- paper object contract: "
        + "; ".join(f"`{item}`" for item in _PAPER_OBJECT_CONTRACT),
        "- paper evidence refs: "
        + "; ".join(f"`{item}`" for item in _PAPER_EVIDENCE_REFS),
        _interval_scale_digest_line(binding_summary, best_summary),
    )

    return Phase7MonteCarloWideningPolicyQualityRiskProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-quality-risk-probe",
        policy_digest=resolved_policy.to_digest(),
        binding_design=binding_summary.design_key,
        best_design=best_summary.design_key,
        binding_driver=binding_driver,
        blocker_reason=blocker_reason,
        supported_floor_ceiling=floor_slack_probe.supported_floor_ceiling,
        canonical_floor_shortfall=floor_slack_probe.canonical_floor_shortfall,
        rmse_to_standard_error_ratio_gap=rmse_ratio_gap,
        standard_error_reserve_gap=standard_error_reserve_gap,
        interval_length_to_rmse_gap=interval_length_to_rmse_gap,
        design_quality_summaries=(binding_summary, best_summary),
        canonical_quality_risk_digest=canonical_quality_risk_digest,
        quality_risk_source_mode="live-runtime-probe",
        quality_risk_source_note=(
            "current runtime probe summaries drive the bounded quality-risk readout"
        ),
    )


def run_phase7_monte_carlo_widening_policy_quality_risk_probe(
    repo_root: str | Path | None = None,
) -> (
    Phase7MonteCarloWideningPolicyQualityRiskProbeReport
):
    policy = build_phase7_canonical_monte_carlo_widening_policy()
    if _prefer_repo_side_quality_risk_first(repo_root):
        return _build_repo_side_quality_risk_probe_report(policy=policy)
    try:
        return build_phase7_monte_carlo_widening_policy_quality_risk_probe_report(
            _run_quality_risk_runtime_probe(),
            policy=policy,
        )
    except KeyError:
        if _repo_side_quality_risk_designs():
            return _build_repo_side_quality_risk_probe_report(policy=policy)
        raise
    except ValueError as exc:
        if _is_quality_metric_validation_error(exc):
            raise
        if (
            _is_repo_side_quality_risk_fallback_error(exc)
            and _repo_side_quality_risk_designs()
        ):
            return _build_repo_side_quality_risk_probe_report(policy=policy)
        raise
