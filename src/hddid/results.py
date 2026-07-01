from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from numbers import Integral
from typing import Any

import numpy as np

ScalarOrArray = float | np.ndarray

_CANONICAL_ORACLE_LANES = {
    "trigonometric": "paper-trigonometric",
    "polynomial": "r-parity-polynomial",
}

_ORACLE_LANE_ALIASES = {
    "paper": "paper-trigonometric",
    "paper-trigonometric": "paper-trigonometric",
    "paper_trigonometric": "paper-trigonometric",
    "r-parity": "r-parity-polynomial",
    "r_parity": "r-parity-polynomial",
    "r-parity-polynomial": "r-parity-polynomial",
    "r_parity_polynomial": "r-parity-polynomial",
}

_ESTIMATE_RENDER_PRIORITY = {
    "Parametric": {
        "beta_hat": 0,
        "t_hat": 1,
    },
    "Nonparametric": {
        "gamma_hat": 0,
        "bar_gamma_hat": 1,
        "f_hat_at_z0": 2,
        "bar_f_at_z0": 3,
    },
}

_ESTIMATE_DISPLAY_NAMES: dict[str, str] = {
    "beta_hat": "β̂",
    "t_hat": "τ̂",
    "gamma_hat": "γ̂",
    "bar_gamma_hat": "γ̄",
    "f_hat_at_z0": "f̂(z₀)",
    "bar_f_at_z0": "f̄(z₀)",
}


def _coerce_positive_integer(name: str, value: int) -> int:
    """Validate and return a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _coerce_optional_positive_integer(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _coerce_positive_integer(name, value)


def _coerce_basis_degree(basis_family: str, basis_degree: int) -> int:
    """Validate basis degree given the basis family."""
    if isinstance(basis_degree, bool) or not isinstance(basis_degree, Integral):
        raise ValueError("basis_degree must be an integer")
    degree = int(basis_degree)
    if degree < 0:
        raise ValueError("basis_degree must be non-negative")
    if basis_family == "trigonometric" and degree < 1:
        raise ValueError("basis_degree must be positive for trigonometric basis")
    return degree


def _coerce_optional_basis_degree(
    basis_family: str | None,
    basis_degree: int | None,
) -> int | None:
    if basis_degree is None:
        return None
    if basis_family is None:
        return _coerce_optional_positive_integer("basis_degree", basis_degree)
    return _coerce_basis_degree(basis_family, basis_degree)


def _coerce_optional_nonnegative_integer(
    name: str, value: int | None
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _coerce_diagnostic_label(
    name: str,
    value: str | None,
    *,
    allow_none: bool,
) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be a string")
    if isinstance(value, (bool, np.bool_, bytes, np.bytes_)) or not isinstance(
        value,
        str,
    ):
        raise ValueError(f"{name} must be a string")
    label = value.strip()
    if not label:
        raise ValueError(f"{name} must be non-empty")
    return label


def _coerce_optional_finite_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if _contains_boolean_value(value):
        raise ValueError(f"{name} must be numeric")
    if _contains_string_value(value):
        raise ValueError(f"{name} must be numeric")
    raw_value = np.asarray(value)
    if _contains_boolean_value(raw_value):
        raise ValueError(f"{name} must be numeric")
    if _contains_string_value(raw_value):
        raise ValueError(f"{name} must be numeric")
    if raw_value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    numeric = float(raw_value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _contains_boolean_value(value: Any) -> bool:
    """Check whether ``value`` contains booleans (recursively)."""
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return True
        if value.dtype == object:
            return any(_contains_boolean_value(item) for item in value.flat)
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean_value(item) for item in value)
    return False


def _contains_string_value(value: Any) -> bool:
    """Check whether ``value`` contains strings (recursively)."""
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.str_) or np.issubdtype(
            value.dtype,
            np.bytes_,
        ):
            return True
        if value.dtype == object:
            return any(_contains_string_value(item) for item in value.flat)
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_string_value(item) for item in value)
    return False


def _require_numeric_array(
    name: str,
    value: Any,
    *,
    nonnegative: bool = False,
) -> float | np.ndarray:
    if _contains_boolean_value(value):
        raise ValueError(f"{name} must be numeric, not boolean")
    if _contains_string_value(value):
        raise ValueError(f"{name} must be numeric, not string")
    raw_array = np.asarray(value)
    if _contains_boolean_value(raw_array):
        raise ValueError(f"{name} must be numeric, not boolean")
    if _contains_string_value(raw_array):
        raise ValueError(f"{name} must be numeric, not string")
    array = raw_array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return float(array[()]) if array.ndim == 0 else array


def _coerce_string_keyed_mapping(name: str, value: Any) -> dict[str, Any]:
    try:
        mapping = dict(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a mapping") from exc
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{name} keys must be strings")
    if any(not key.strip() for key in mapping):
        raise ValueError(f"{name} keys must be non-empty strings")
    if any(key != key.strip() for key in mapping):
        raise ValueError(f"{name} keys must not have leading or trailing whitespace")
    return mapping


def _validate_count_contract(
    *,
    n_holdout_raw: int | None,
    n_trimmed_propensity: int | None,
    n_valid_holdout: int | None,
) -> None:
    if n_holdout_raw is None:
        return
    if n_trimmed_propensity is not None and n_trimmed_propensity > n_holdout_raw:
        raise ValueError("n_trimmed_propensity cannot exceed n_holdout_raw")
    if n_valid_holdout is not None and n_valid_holdout > n_holdout_raw:
        raise ValueError("n_valid_holdout cannot exceed n_holdout_raw")
    if (
        n_trimmed_propensity is not None
        and n_valid_holdout is not None
        and n_trimmed_propensity + n_valid_holdout != n_holdout_raw
    ):
        raise ValueError(
            "n_trimmed_propensity plus n_valid_holdout must equal n_holdout_raw"
        )


def _validate_trim_bounds(trim_lower: float | None, trim_upper: float | None) -> None:
    """Validate that trim bounds are in [0, 1] and lower < upper."""
    if trim_lower is None and trim_upper is None:
        return
    if trim_lower is None or trim_upper is None:
        raise ValueError("trim_lower and trim_upper must be supplied together")
    if not 0.0 <= trim_lower < trim_upper <= 1.0:
        raise ValueError("trim bounds must satisfy 0 <= trim_lower < trim_upper <= 1")


def _validate_fold_count_total(
    name: str,
    aggregate_value: int | None,
    fold_diagnostics: list[FoldDiagnostics],
) -> None:
    if aggregate_value is None or not fold_diagnostics:
        return
    fold_values = [getattr(fold, name) for fold in fold_diagnostics]
    if any(value is None for value in fold_values):
        return
    if aggregate_value != sum(int(value) for value in fold_values):
        raise ValueError(f"{name} must equal the fold total")


def _validate_fold_design_contract(
    *,
    basis_family: str,
    basis_degree: int,
    oracle_lane: str,
    fold_diagnostics: list[FoldDiagnostics],
) -> None:
    for fold in fold_diagnostics:
        if fold.basis_family is not None and fold.basis_family != basis_family:
            raise ValueError("fold_diagnostics basis_family must match run contract")
        if fold.basis_degree is not None and fold.basis_degree != basis_degree:
            raise ValueError("fold_diagnostics basis_degree must match run contract")
        if fold.oracle_lane is not None and fold.oracle_lane != oracle_lane:
            raise ValueError("fold_diagnostics oracle_lane must match run contract")


def _validate_finite_metadata(path: str, value: Any) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, Integral):
        return
    if isinstance(value, dict):
        for key, nested_value in value.items():
            _validate_finite_metadata(f"{path}.{key}", nested_value)
        return
    if isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            _validate_finite_metadata(f"{path}[{index}]", nested_value)
        return
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path} must be finite")


def _flatten_numeric_values(value: float | np.ndarray) -> list[float]:
    """Flatten a scalar or array into a list of floats."""
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def _format_number(value: float, *, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def _format_compact_number(value: float, *, digits: int) -> str:
    numeric = float(value)
    if numeric == 0.0:
        return "0"
    magnitude = abs(numeric)
    if magnitude >= 10 ** max(digits, 1) or magnitude < 10 ** (-(digits + 1)):
        return f"{numeric:.{digits}e}"
    return f"{numeric:.{digits}f}".rstrip("0").rstrip(".")


def _format_result_number(value: float, *, digits: int, number_format: str) -> str:
    if number_format == "fixed":
        return _format_number(value, digits=digits)
    if number_format == "compact":
        return _format_compact_number(value, digits=digits)
    raise ValueError("number_format must be one of {'fixed', 'compact'}")


def _format_confidence_level_percent(level: float) -> str:
    percent = (Decimal(str(float(level))) * Decimal("100")).quantize(
        Decimal("0.1"),
        rounding=ROUND_DOWN,
    )
    return f"{percent:.1f}%"


def _format_markdown_cell(value: str) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    line_break_token = "\u0000HDDID_BR\u0000"
    text = text.replace("\n", line_break_token).replace("\t", " ")
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(line_break_token, "<br>")
    )
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _format_interval(
    interval: ConfidenceInterval | None,
    index: int,
    *,
    digits: int,
    number_format: str,
) -> str:
    if interval is None:
        return ""
    lower_values = _flatten_numeric_values(interval.lower)
    upper_values = _flatten_numeric_values(interval.upper)
    if index >= len(lower_values) or index >= len(upper_values):
        return ""
    text = (
        "["
        + _format_result_number(
            lower_values[index],
            digits=digits,
            number_format=number_format,
        )
        + ", "
        + _format_result_number(
            upper_values[index],
            digits=digits,
            number_format=number_format,
        )
        + "]"
    )
    if interval.level is not None:
        text = f"{text} ({_format_confidence_level_percent(interval.level)})"
    return text


def _require_render_length(
    *,
    field_path: str,
    rendered_estimate_name: str,
    values: float | np.ndarray,
    expected_length: int,
) -> None:
    actual_length = len(_flatten_numeric_values(values))
    if actual_length != expected_length:
        raise ValueError(
            f"{field_path} must contain {expected_length} value(s) "
            f"to render {rendered_estimate_name}"
        )


def _first_key_containing(mapping: dict[str, Any], token: str) -> str | None:
    token_key = token.lower()
    if len(token_key) < 2:
        return None
    for key in mapping:
        if token_key in str(key).lower():
            return key
    return None


def _key_has_token(key: str, token: str) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return token.lower() in normalized.split("_")


def _matching_standard_error_key(
    estimate_name: str,
    standard_errors: dict[str, Any],
    *,
    section_estimate_names: set[str] | None = None,
) -> str | None:
    section_names = section_estimate_names or set()
    if estimate_name in standard_errors:
        return estimate_name
    if estimate_name.endswith("_hat"):
        candidate = f"{estimate_name[:-4]}_se"
        if candidate in standard_errors:
            if estimate_name == "beta_hat" and "t_hat" in section_names:
                return None
            return candidate
    if estimate_name.endswith("_at_z0"):
        candidate = f"{estimate_name[:-6]}_se"
        if candidate in standard_errors:
            return candidate
    if estimate_name == "bar_f_at_z0":
        for candidate in ("bar_f_se", "nonparametric_se", "f_se"):
            if candidate in standard_errors:
                return candidate
    if estimate_name == "t_hat":
        for candidate in ("t_se", "parametric_se", "beta_se"):
            if candidate in standard_errors:
                return candidate
    base = estimate_name.removesuffix("_hat").removesuffix("_at_z0")
    if estimate_name == "beta_hat" and "t_hat" in section_names:
        return None
    return _first_key_containing(standard_errors, base)


def _matching_interval_key(
    section: str,
    estimate_name: str,
    intervals: dict[str, ConfidenceInterval],
    *,
    section_estimate_count: int = 1,
) -> str | None:
    if estimate_name in intervals:
        return estimate_name
    section_key = section.lower()
    base = estimate_name.removesuffix("_hat").removesuffix("_at_z0")
    for candidate in (
        f"{estimate_name}_ci",
        f"{base}_ci" if base else "",
    ):
        if candidate and candidate in intervals:
            return candidate
    if section_key == "parametric" and estimate_name == "t_hat":
        if "parametric_ci" in intervals:
            return "parametric_ci"
    if section_key == "nonparametric" and estimate_name == "bar_f_at_z0":
        if "nonparametric_ci" in intervals:
            return "nonparametric_ci"
    for key in intervals:
        lowered = str(key).lower()
        if _key_has_token(lowered, section_key) and (
            not base or (len(base) >= 2 and base in lowered)
        ):
            return key
    section_matches = [
        key for key in intervals if _key_has_token(str(key), section_key)
    ]
    if section_estimate_count == 1 and len(section_matches) == 1:
        return section_matches[0]
    return _first_key_containing(intervals, base)


def _ordered_estimate_items(
    section: str,
    estimates: dict[str, Any],
) -> list[tuple[str, Any]]:
    priority = _ESTIMATE_RENDER_PRIORITY.get(section, {})
    return sorted(
        estimates.items(),
        key=lambda item: (priority.get(item[0], len(priority)), str(item[0])),
    )


def _format_optional_count(name: str, value: int | None) -> str | None:
    if value is None:
        return None
    return f"{name}={int(value)}"


def _fold_diagnostic_summary(
    fold_diagnostics: list[FoldDiagnostics],
    *,
    digits: int,
    number_format: str,
) -> str:
    entries: list[str] = []
    for fold in fold_diagnostics:
        parts = [f"fold {fold.fold_id}"]
        counts = (
            _format_optional_count("holdout", fold.n_holdout_raw),
            _format_optional_count("trimmed", fold.n_trimmed_propensity),
            _format_optional_count("valid", fold.n_valid_holdout),
        )
        if any(count is not None for count in counts):
            parts.append(", ".join(count for count in counts if count is not None))
        if fold.trim_lower is not None and fold.trim_upper is not None:
            parts.append(
                "trim="
                + "["
                + _format_result_number(
                    fold.trim_lower,
                    digits=digits,
                    number_format=number_format,
                )
                + ", "
                + _format_result_number(
                    fold.trim_upper,
                    digits=digits,
                    number_format=number_format,
                )
                + "]"
            )
        entries.append("; ".join(parts))
    return "\n".join(_format_markdown_cell(entry) for entry in entries)


def _validate_row_labels(
    row_labels: Mapping[str, Any] | None,
    estimate_lengths: dict[str, int],
) -> dict[str, list[str]]:
    if row_labels is None:
        return {}
    if not isinstance(row_labels, Mapping):
        raise ValueError("row_labels must be a mapping from estimate names to labels")

    non_string_names = [name for name in row_labels if not isinstance(name, str)]
    if non_string_names:
        raise ValueError("row_labels estimate names must be strings")

    unknown_names = sorted(name for name in row_labels if name not in estimate_lengths)
    if unknown_names:
        unknown = ", ".join(str(name) for name in unknown_names)
        raise ValueError(f"row_labels contains unknown estimate name(s): {unknown}")

    validated: dict[str, list[str]] = {}
    for name, labels in row_labels.items():
        if isinstance(labels, (str, bytes)):
            raise ValueError(f"row_labels.{name} must be a sequence of strings")
        try:
            label_list = list(labels)
        except TypeError as exc:
            raise ValueError(
                f"row_labels.{name} must be a sequence of strings"
            ) from exc

        expected_length = estimate_lengths[name]
        if len(label_list) != expected_length:
            raise ValueError(
                f"row_labels.{name} must contain {expected_length} labels"
            )
        if not all(isinstance(label, str) for label in label_list):
            raise ValueError(f"row_labels.{name} must contain only strings")
        validated[name] = label_list
    return validated


def normalize_oracle_lane(
    *,
    basis_family: str | None,
    oracle_lane: str | None,
) -> str | None:
    """Canonicalize the oracle lane string based on basis family.

    Maps shorthand aliases to their canonical lane identifiers used
    for R-parity verification.
    """
    lane_value = _coerce_diagnostic_label(
        "oracle_lane",
        oracle_lane,
        allow_none=True,
    )
    if lane_value is None:
        return None

    lane_key = lane_value.lower().replace(" ", "-")
    lane = _ORACLE_LANE_ALIASES.get(lane_key, lane_key)

    if basis_family is None:
        return lane

    family = _coerce_diagnostic_label(
        "basis_family",
        basis_family,
        allow_none=False,
    )
    family = str(family).lower()
    expected_lane = _CANONICAL_ORACLE_LANES.get(family)
    if expected_lane is None:
        return lane
    if lane != expected_lane:
        raise ValueError(
            f"oracle_lane must be '{expected_lane}' when basis_family is '{family}'"
        )
    return lane


@dataclass(slots=True)
class ConfidenceInterval:
    """Pointwise confidence interval for a scalar or vector estimate.

    Attributes
    ----------
    lower : float or ndarray
        Lower confidence bound.
    upper : float or ndarray
        Upper confidence bound.
    level : float or None
        Nominal coverage level (e.g. 0.95).
    """

    lower: ScalarOrArray
    upper: ScalarOrArray
    level: float | None = None

    def __post_init__(self) -> None:
        field_prefix = (
            "uniform_band"
            if type(self).__name__ == "UniformBand"
            else "confidence_interval"
        )
        if _contains_boolean_value(self.lower):
            raise ValueError(f"{field_prefix}.lower must be numeric, not boolean")
        if _contains_boolean_value(self.upper):
            raise ValueError(f"{field_prefix}.upper must be numeric, not boolean")
        if _contains_string_value(self.lower):
            raise ValueError(f"{field_prefix}.lower must be numeric, not string")
        if _contains_string_value(self.upper):
            raise ValueError(f"{field_prefix}.upper must be numeric, not string")
        raw_lower = np.asarray(self.lower)
        raw_upper = np.asarray(self.upper)
        if _contains_boolean_value(raw_lower):
            raise ValueError(f"{field_prefix}.lower must be numeric, not boolean")
        if _contains_boolean_value(raw_upper):
            raise ValueError(f"{field_prefix}.upper must be numeric, not boolean")
        if _contains_string_value(raw_lower):
            raise ValueError(f"{field_prefix}.lower must be numeric, not string")
        if _contains_string_value(raw_upper):
            raise ValueError(f"{field_prefix}.upper must be numeric, not string")
        lower = raw_lower.astype(float)
        upper = raw_upper.astype(float)
        if lower.shape != upper.shape:
            raise ValueError("confidence interval bounds must share the same shape")
        if not np.all(np.isfinite(lower)):
            raise ValueError(
                f"{field_prefix}.lower must contain only finite values; "
                "confidence interval bounds must be finite"
            )
        if not np.all(np.isfinite(upper)):
            raise ValueError(
                f"{field_prefix}.upper must contain only finite values; "
                "confidence interval bounds must be finite"
            )
        if np.any(lower > upper):
            raise ValueError("confidence interval lower bound must not exceed upper")
        self.lower = float(lower[()]) if lower.ndim == 0 else lower
        self.upper = float(upper[()]) if upper.ndim == 0 else upper
        if self.level is not None:
            if _contains_boolean_value(self.level):
                raise ValueError("level must be numeric, not boolean")
            if _contains_string_value(self.level):
                raise ValueError("level must be numeric, not string")
            raw_level = np.asarray(self.level)
            if _contains_boolean_value(raw_level):
                raise ValueError("level must be numeric, not boolean")
            if _contains_string_value(raw_level):
                raise ValueError("level must be numeric, not string")
            if raw_level.ndim != 0:
                raise ValueError("level must be a scalar")
            level = float(raw_level)
            if not np.isfinite(level):
                raise ValueError("level must be finite")
            if not 0.0 < level < 1.0:
                raise ValueError("level must be in (0, 1)")
            self.level = level


@dataclass(slots=True)
class UniformBand(ConfidenceInterval):
    """Uniform confidence band for the nonparametric function.

    Attributes
    ----------
    lower : float or ndarray
        Lower band boundary.
    upper : float or ndarray
        Upper band boundary.
    level : float or None
        Nominal coverage level.
    critical_value : float or None
        Bootstrap critical value for uniform coverage.
    n_boot : int or None
        Number of bootstrap replications used.
    random_state : int or None
        Random seed for bootstrap reproducibility.
    """

    critical_value: float | None = None
    n_boot: int | None = None
    random_state: int | None = None

    def __post_init__(self) -> None:
        ConfidenceInterval.__post_init__(self)
        if self.critical_value is not None:
            if _contains_boolean_value(self.critical_value):
                raise ValueError("critical_value must be numeric, not boolean")
            if _contains_string_value(self.critical_value):
                raise ValueError("critical_value must be numeric, not string")
            raw_critical_value = np.asarray(self.critical_value)
            if _contains_boolean_value(raw_critical_value):
                raise ValueError("critical_value must be numeric, not boolean")
            if _contains_string_value(raw_critical_value):
                raise ValueError("critical_value must be numeric, not string")
            if raw_critical_value.ndim != 0:
                raise ValueError("critical_value must be a scalar")
            critical_value = float(raw_critical_value)
            if not np.isfinite(critical_value):
                raise ValueError("critical_value must be finite")
            if critical_value <= 0.0:
                raise ValueError("critical_value must be positive")
            self.critical_value = critical_value
        if self.n_boot is not None:
            self.n_boot = _coerce_positive_integer("n_boot", self.n_boot)
        if self.random_state is not None:
            self.random_state = _coerce_optional_nonnegative_integer(
                "random_state",
                self.random_state,
            )


@dataclass(slots=True)
class FoldDiagnostics:
    """Per-fold diagnostic summary from cross-fitting.

    Attributes
    ----------
    fold_id : int
        One-based fold index.
    basis_family : str or None
        Sieve basis family used in this fold.
    basis_degree : int or None
        Sieve basis degree.
    oracle_lane : str or None
        Oracle nuisance lane identifier (simulation only).
    n_holdout_raw : int or None
        Total holdout observations before trimming.
    n_trimmed_propensity : int or None
        Observations removed by propensity score trimming.
    n_valid_holdout : int or None
        Observations remaining after trimming.
    trim_lower : float or None
        Lower propensity trimming threshold applied.
    trim_upper : float or None
        Upper propensity trimming threshold applied.
    """

    fold_id: int
    basis_family: str | None = None
    basis_degree: int | None = None
    oracle_lane: str | None = None
    n_holdout_raw: int | None = None
    n_trimmed_propensity: int | None = None
    n_valid_holdout: int | None = None
    trim_lower: float | None = None
    trim_upper: float | None = None

    def __post_init__(self) -> None:
        self.fold_id = _coerce_positive_integer("fold_id", self.fold_id)
        basis_family = _coerce_diagnostic_label(
            "basis_family",
            self.basis_family,
            allow_none=True,
        )
        self.basis_family = None if basis_family is None else basis_family.lower()
        self.basis_degree = _coerce_optional_basis_degree(
            self.basis_family,
            self.basis_degree,
        )
        self.n_holdout_raw = _coerce_optional_nonnegative_integer(
            "n_holdout_raw", self.n_holdout_raw
        )
        self.n_trimmed_propensity = _coerce_optional_nonnegative_integer(
            "n_trimmed_propensity", self.n_trimmed_propensity
        )
        self.n_valid_holdout = _coerce_optional_nonnegative_integer(
            "n_valid_holdout", self.n_valid_holdout
        )
        self.trim_lower = _coerce_optional_finite_float(
            "trim_lower", self.trim_lower
        )
        self.trim_upper = _coerce_optional_finite_float(
            "trim_upper", self.trim_upper
        )
        _validate_count_contract(
            n_holdout_raw=self.n_holdout_raw,
            n_trimmed_propensity=self.n_trimmed_propensity,
            n_valid_holdout=self.n_valid_holdout,
        )
        _validate_trim_bounds(self.trim_lower, self.trim_upper)
        self.oracle_lane = normalize_oracle_lane(
            basis_family=self.basis_family,
            oracle_lane=self.oracle_lane,
        )


@dataclass(slots=True)
class ResultDiagnostics:
    """Aggregated diagnostics across all cross-fitting folds.

    Attributes
    ----------
    basis_family : str
        Sieve basis family.
    basis_degree : int
        Sieve basis degree.
    oracle_lane : str
        Oracle lane identifier.
    fold_diagnostics : list of FoldDiagnostics
        Per-fold diagnostic records.
    n_holdout_raw : int or None
        Total holdout observations (all folds combined).
    n_trimmed_propensity : int or None
        Total observations trimmed across folds.
    n_valid_holdout : int or None
        Total valid observations after trimming.
    trim_lower : float or None
        Lower propensity trimming threshold.
    trim_upper : float or None
        Upper propensity trimming threshold.
    optimization_metadata : dict
        Solver convergence and iteration metadata.
    """

    basis_family: str
    basis_degree: int
    oracle_lane: str
    fold_diagnostics: list[FoldDiagnostics] = field(default_factory=list)
    n_holdout_raw: int | None = None
    n_trimmed_propensity: int | None = None
    n_valid_holdout: int | None = None
    trim_lower: float | None = None
    trim_upper: float | None = None
    optimization_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.basis_family = _coerce_diagnostic_label(
            "basis_family",
            self.basis_family,
            allow_none=False,
        ).lower()
        self.basis_degree = _coerce_basis_degree(
            self.basis_family,
            self.basis_degree,
        )
        self.fold_diagnostics = list(self.fold_diagnostics)
        if not all(
            isinstance(fold, FoldDiagnostics) for fold in self.fold_diagnostics
        ):
            raise ValueError("fold_diagnostics must contain FoldDiagnostics objects")
        self.n_holdout_raw = _coerce_optional_nonnegative_integer(
            "n_holdout_raw", self.n_holdout_raw
        )
        self.n_trimmed_propensity = _coerce_optional_nonnegative_integer(
            "n_trimmed_propensity", self.n_trimmed_propensity
        )
        self.n_valid_holdout = _coerce_optional_nonnegative_integer(
            "n_valid_holdout", self.n_valid_holdout
        )
        self.trim_lower = _coerce_optional_finite_float(
            "trim_lower", self.trim_lower
        )
        self.trim_upper = _coerce_optional_finite_float(
            "trim_upper", self.trim_upper
        )
        _validate_count_contract(
            n_holdout_raw=self.n_holdout_raw,
            n_trimmed_propensity=self.n_trimmed_propensity,
            n_valid_holdout=self.n_valid_holdout,
        )
        for count_name, aggregate_value in (
            ("n_holdout_raw", self.n_holdout_raw),
            ("n_trimmed_propensity", self.n_trimmed_propensity),
            ("n_valid_holdout", self.n_valid_holdout),
        ):
            _validate_fold_count_total(
                count_name,
                aggregate_value,
                self.fold_diagnostics,
            )
        _validate_trim_bounds(self.trim_lower, self.trim_upper)
        self.optimization_metadata = dict(self.optimization_metadata)
        for key, value in self.optimization_metadata.items():
            _validate_finite_metadata(f"optimization_metadata.{key}", value)
        self.oracle_lane = _coerce_diagnostic_label(
            "oracle_lane",
            self.oracle_lane,
            allow_none=False,
        )
        self.oracle_lane = normalize_oracle_lane(
            basis_family=self.basis_family,
            oracle_lane=self.oracle_lane,
        )
        _validate_fold_design_contract(
            basis_family=self.basis_family,
            basis_degree=self.basis_degree,
            oracle_lane=self.oracle_lane,
            fold_diagnostics=self.fold_diagnostics,
        )


def _format_diagnostics_pretty(
    diagnostics: ResultDiagnostics,
    *,
    digits: int,
    number_format: str,
) -> list[str]:
    """Format diagnostics as a multi-line indented block."""
    lines: list[str] = ["", "Diagnostics"]
    lines.append(f"  Basis: {diagnostics.basis_family}({diagnostics.basis_degree})")
    lines.append(f"  Oracle lane: {diagnostics.oracle_lane}")
    counts = (
        _format_optional_count("holdout", diagnostics.n_holdout_raw),
        _format_optional_count("trimmed", diagnostics.n_trimmed_propensity),
        _format_optional_count("valid", diagnostics.n_valid_holdout),
    )
    if all(count is not None for count in counts):
        lines.append(
            "  Sample: " + ", ".join(c for c in counts if c is not None)
        )
    if diagnostics.trim_lower is not None and diagnostics.trim_upper is not None:
        trim_str = (
            "["
            + _format_result_number(
                diagnostics.trim_lower, digits=digits, number_format=number_format
            )
            + ", "
            + _format_result_number(
                diagnostics.trim_upper, digits=digits, number_format=number_format
            )
            + "]"
        )
        lines.append(f"  Trim: {trim_str}")
    if diagnostics.fold_diagnostics:
        lines.append(f"  Folds: {len(diagnostics.fold_diagnostics)}")
    return lines


@dataclass(slots=True)
class HDDIDResult:
    """User-facing container for HDDID estimation results.

    Attributes
    ----------
    parametric_estimates : dict
        Named parametric point estimates (beta_hat, t_hat).
    nonparametric_estimates : dict
        Named nonparametric estimates (gamma_hat, f_hat_at_z0).
    standard_errors : dict
        Standard errors keyed by estimate name.
    intervals : dict of ConfidenceInterval
        Confidence intervals keyed by estimate name.
    diagnostics : ResultDiagnostics or None
        Cross-fitting and solver diagnostics.
    """

    parametric_estimates: dict[str, Any] = field(default_factory=dict)
    nonparametric_estimates: dict[str, Any] = field(default_factory=dict)
    standard_errors: dict[str, Any] = field(default_factory=dict)
    intervals: dict[str, ConfidenceInterval] = field(default_factory=dict)
    diagnostics: ResultDiagnostics | None = None

    def __post_init__(self) -> None:
        self.parametric_estimates = {
            key: _require_numeric_array(f"parametric_estimates.{key}", value)
            for key, value in _coerce_string_keyed_mapping(
                "parametric_estimates",
                self.parametric_estimates,
            ).items()
        }
        self.nonparametric_estimates = {
            key: _require_numeric_array(f"nonparametric_estimates.{key}", value)
            for key, value in _coerce_string_keyed_mapping(
                "nonparametric_estimates",
                self.nonparametric_estimates,
            ).items()
        }
        self.standard_errors = {
            key: _require_numeric_array(
                f"standard_errors.{key}",
                value,
                nonnegative=True,
            )
            for key, value in _coerce_string_keyed_mapping(
                "standard_errors",
                self.standard_errors,
            ).items()
        }
        self.intervals = _coerce_string_keyed_mapping("intervals", self.intervals)
        for key, value in self.intervals.items():
            if not isinstance(value, ConfidenceInterval):
                raise ValueError(f"intervals.{key} must be a ConfidenceInterval")
        if self.diagnostics is not None and not isinstance(
            self.diagnostics, ResultDiagnostics
        ):
            raise ValueError("diagnostics must be a ResultDiagnostics object")

    def to_markdown(
        self,
        *,
        digits: int = 4,
        number_format: str = "fixed",
        row_labels: Mapping[str, Any] | None = None,
        missing_value: str = "—",
        style: str = "pretty",
    ) -> str:
        if isinstance(digits, bool) or not isinstance(digits, Integral):
            raise ValueError("digits must be an integer")
        digits_value = int(digits)
        if digits_value < 0:
            raise ValueError("digits must be non-negative")
        number_format_value = str(number_format)
        if number_format_value not in {"fixed", "compact"}:
            raise ValueError("number_format must be one of {'fixed', 'compact'}")
        if not isinstance(missing_value, str):
            raise ValueError("missing_value must be a string")
        style_value = str(style)
        if style_value not in {"pretty", "legacy"}:
            raise ValueError("style must be one of {'pretty', 'legacy'}")
        missing_cell = _format_markdown_cell(missing_value)

        estimate_groups = (
            ("Parametric", self.parametric_estimates),
            ("Nonparametric", self.nonparametric_estimates),
        )
        estimate_lengths = {
            name: len(_flatten_numeric_values(values))
            for section, estimates in estimate_groups
            for name, values in _ordered_estimate_items(section, estimates)
        }
        validated_row_labels = _validate_row_labels(row_labels, estimate_lengths)
        include_labels = row_labels is not None

        if include_labels:
            lines = [
                "| Section | Name | Index | Label | Estimate | Std. Error | Interval |",
                "|---|---|---:|---|---:|---:|---|",
            ]
        else:
            lines = [
                "| Section | Name | Index | Estimate | Std. Error | Interval |",
                "|---|---|---:|---:|---:|---|",
            ]

        prev_section = ""
        for section, estimates in estimate_groups:
            section_estimate_names = set(estimates)
            for name, values in _ordered_estimate_items(section, estimates):
                estimate_values = _flatten_numeric_values(values)
                se_key = _matching_standard_error_key(
                    name,
                    self.standard_errors,
                    section_estimate_names=section_estimate_names,
                )
                se_values = (
                    _flatten_numeric_values(self.standard_errors[se_key])
                    if se_key is not None
                    else []
                )
                if se_key is not None:
                    _require_render_length(
                        field_path=f"standard_errors.{se_key}",
                        rendered_estimate_name=name,
                        values=self.standard_errors[se_key],
                        expected_length=len(estimate_values),
                    )
                interval_key = _matching_interval_key(
                    section,
                    name,
                    self.intervals,
                    section_estimate_count=len(estimates),
                )
                interval = self.intervals.get(interval_key) if interval_key else None
                if interval_key is not None and interval is not None:
                    _require_render_length(
                        field_path=f"intervals.{interval_key}",
                        rendered_estimate_name=name,
                        values=interval.lower,
                        expected_length=len(estimate_values),
                    )
                    _require_render_length(
                        field_path=f"intervals.{interval_key}",
                        rendered_estimate_name=name,
                        values=interval.upper,
                        expected_length=len(estimate_values),
                    )
                for index, estimate in enumerate(estimate_values):
                    standard_error = (
                        _format_result_number(
                            se_values[index],
                            digits=digits_value,
                            number_format=number_format_value,
                        )
                        if index < len(se_values)
                        else missing_cell
                    )
                    interval_text = _format_interval(
                        interval,
                        index,
                        digits=digits_value,
                        number_format=number_format_value,
                    )
                    if not interval_text:
                        interval_text = missing_cell
                    if style_value == "pretty":
                        display_name = _ESTIMATE_DISPLAY_NAMES.get(name, name)
                        if section != prev_section:
                            section_cell = _format_markdown_cell(f"**{section}**")
                            prev_section = section
                        else:
                            section_cell = ""
                        common_cells = (
                            section_cell,
                            _format_markdown_cell(display_name),
                            str(index),
                        )
                    else:
                        common_cells = (
                            _format_markdown_cell(section),
                            _format_markdown_cell(name),
                            str(index),
                        )
                    if include_labels:
                        labels = validated_row_labels.get(name, [])
                        label = labels[index] if labels else ""
                        cells = (
                            *common_cells,
                            _format_markdown_cell(label),
                            _format_result_number(
                                estimate,
                                digits=digits_value,
                                number_format=number_format_value,
                            ),
                            standard_error,
                            interval_text,
                        )
                    else:
                        cells = (
                            *common_cells,
                            _format_result_number(
                                estimate,
                                digits=digits_value,
                                number_format=number_format_value,
                            ),
                            standard_error,
                            interval_text,
                        )
                    lines.append("| " + " | ".join(cells) + " |")

        if self.diagnostics is not None:
            if style_value == "pretty":
                lines.extend(
                    _format_diagnostics_pretty(
                        self.diagnostics,
                        digits=digits_value,
                        number_format=number_format_value,
                    )
                )
            else:
                diagnostics = self.diagnostics
                summary_parts = [
                    f"basis={diagnostics.basis_family}({diagnostics.basis_degree})",
                    f"oracle_lane={diagnostics.oracle_lane}",
                ]
                counts = (
                    _format_optional_count("holdout", diagnostics.n_holdout_raw),
                    _format_optional_count("trimmed", diagnostics.n_trimmed_propensity),
                    _format_optional_count("valid", diagnostics.n_valid_holdout),
                )
                if all(count is not None for count in counts):
                    summary_parts.append(
                        ", ".join(count for count in counts if count is not None)
                    )
                if diagnostics.trim_lower is not None and diagnostics.trim_upper is not None:
                    summary_parts.append(
                        "trim="
                        + "["
                        + _format_result_number(
                            diagnostics.trim_lower,
                            digits=digits_value,
                            number_format=number_format_value,
                        )
                        + ", "
                        + _format_result_number(
                            diagnostics.trim_upper,
                            digits=digits_value,
                            number_format=number_format_value,
                        )
                        + "]"
                    )
                if diagnostics.fold_diagnostics:
                    summary_parts.append(f"folds={len(diagnostics.fold_diagnostics)}")
                lines.extend(("", f"Diagnostics: {'; '.join(summary_parts)}"))
                if diagnostics.fold_diagnostics:
                    lines.append(
                        "Fold diagnostics: "
                        + _fold_diagnostic_summary(
                            diagnostics.fold_diagnostics,
                            digits=digits_value,
                            number_format=number_format_value,
                        )
                    )

        return "\n".join(lines)

    def to_summary(
        self,
        *,
        digits: int = 4,
        number_format: str = "fixed",
        row_labels: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(digits, bool) or not isinstance(digits, Integral):
            raise ValueError("digits must be an integer")
        digits_value = int(digits)
        if digits_value < 0:
            raise ValueError("digits must be non-negative")
        number_format_value = str(number_format)
        if number_format_value not in {"fixed", "compact"}:
            raise ValueError("number_format must be one of {'fixed', 'compact'}")

        estimate_groups = (
            ("Parametric", self.parametric_estimates),
            ("Nonparametric", self.nonparametric_estimates),
        )
        estimate_lengths = {
            name: len(_flatten_numeric_values(values))
            for section, estimates in estimate_groups
            for name, values in _ordered_estimate_items(section, estimates)
        }
        validated_row_labels = _validate_row_labels(row_labels, estimate_lengths)

        rows: list[dict[str, Any]] = []
        estimate_order: list[dict[str, Any]] = []
        missing_standard_error_cells = 0
        missing_interval_cells = 0
        sections: dict[str, dict[str, Any]] = {}

        for section, estimates in estimate_groups:
            section_estimate_names = set(estimates)
            ordered_items = _ordered_estimate_items(section, estimates)
            section_row_count = 0
            sections[section] = {
                "estimate_names": [name for name, _ in ordered_items],
                "row_count": 0,
            }
            for name, values in ordered_items:
                estimate_values = _flatten_numeric_values(values)
                se_key = _matching_standard_error_key(
                    name,
                    self.standard_errors,
                    section_estimate_names=section_estimate_names,
                )
                se_values = (
                    _flatten_numeric_values(self.standard_errors[se_key])
                    if se_key is not None
                    else []
                )
                if se_key is not None:
                    _require_render_length(
                        field_path=f"standard_errors.{se_key}",
                        rendered_estimate_name=name,
                        values=self.standard_errors[se_key],
                        expected_length=len(estimate_values),
                    )
                interval_key = _matching_interval_key(
                    section,
                    name,
                    self.intervals,
                    section_estimate_count=len(estimates),
                )
                interval = self.intervals.get(interval_key) if interval_key else None
                if interval_key is not None and interval is not None:
                    _require_render_length(
                        field_path=f"intervals.{interval_key}",
                        rendered_estimate_name=name,
                        values=interval.lower,
                        expected_length=len(estimate_values),
                    )
                    _require_render_length(
                        field_path=f"intervals.{interval_key}",
                        rendered_estimate_name=name,
                        values=interval.upper,
                        expected_length=len(estimate_values),
                    )

                estimate_order.append(
                    {
                        "section": section,
                        "name": name,
                        "length": len(estimate_values),
                        "standard_error_key": se_key,
                        "interval_key": interval_key,
                    }
                )
                section_row_count += len(estimate_values)
                labels = validated_row_labels.get(name, [])
                for index, estimate in enumerate(estimate_values):
                    standard_error = (
                        _format_result_number(
                            se_values[index],
                            digits=digits_value,
                            number_format=number_format_value,
                        )
                        if index < len(se_values)
                        else None
                    )
                    if standard_error is None:
                        missing_standard_error_cells += 1
                    interval_text = _format_interval(
                        interval,
                        index,
                        digits=digits_value,
                        number_format=number_format_value,
                    )
                    if not interval_text:
                        interval_text = None
                        missing_interval_cells += 1
                    row: dict[str, Any] = {
                        "section": section,
                        "name": name,
                        "index": index,
                        "estimate": _format_result_number(
                            estimate,
                            digits=digits_value,
                            number_format=number_format_value,
                        ),
                        "standard_error": standard_error,
                        "interval": interval_text,
                    }
                    if row_labels is not None:
                        row["label"] = labels[index] if labels else None
                    rows.append(row)
            sections[section]["row_count"] = section_row_count

        diagnostics_summary: dict[str, Any] | None = None
        if self.diagnostics is not None:
            diagnostics = self.diagnostics
            diagnostics_summary = {
                "basis_family": diagnostics.basis_family,
                "basis_degree": diagnostics.basis_degree,
                "oracle_lane": diagnostics.oracle_lane,
                "n_holdout_raw": diagnostics.n_holdout_raw,
                "n_trimmed_propensity": diagnostics.n_trimmed_propensity,
                "n_valid_holdout": diagnostics.n_valid_holdout,
                "trim_lower": (
                    _format_result_number(
                        diagnostics.trim_lower,
                        digits=digits_value,
                        number_format=number_format_value,
                    )
                    if diagnostics.trim_lower is not None
                    else None
                ),
                "trim_upper": (
                    _format_result_number(
                        diagnostics.trim_upper,
                        digits=digits_value,
                        number_format=number_format_value,
                    )
                    if diagnostics.trim_upper is not None
                    else None
                ),
                "fold_count": len(diagnostics.fold_diagnostics),
            }

        return {
            "result_contract": "hddid-result-summary",
            "digits": digits_value,
            "number_format": number_format_value,
            "columns": (
                ["Section", "Name", "Index", "Label", "Estimate", "Std. Error", "Interval"]
                if row_labels is not None
                else ["Section", "Name", "Index", "Estimate", "Std. Error", "Interval"]
            ),
            "has_row_labels": row_labels is not None,
            "row_count": len(rows),
            "rows": rows,
            "sections": sections,
            "estimate_order": estimate_order,
            "missing_standard_error_cells": missing_standard_error_cells,
            "missing_interval_cells": missing_interval_cells,
            "diagnostics": diagnostics_summary,
        }
