from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from html import escape
import json
import math
from numbers import Integral
import re
from typing import Any

import numpy as np

from .results import ConfidenceInterval, HDDIDResult, UniformBand

_SVG_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _coerce_finite_vector(name: str, values: Any) -> np.ndarray:
    """Coerce input to a 1-D finite float array."""
    if _contains_boolean_or_string(values):
        raise ValueError(f"{name} must be a numeric vector")
    try:
        raw_array = np.asarray(values)
        if _contains_boolean_or_string(raw_array):
            raise ValueError(f"{name} must be a numeric vector")
        array = raw_array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _contains_boolean_or_string(value: Any) -> bool:
    """Check whether ``value`` contains booleans or strings (recursively)."""
    if isinstance(value, (bool, np.bool_, str, bytes, np.str_, np.bytes_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_ or value.dtype.kind in {"S", "U"}:
            return True
        if value.dtype == object:
            return any(_contains_boolean_or_string(item) for item in value.flat)
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean_or_string(item) for item in value)
    return False


def _coerce_positive_integer(name: str, value: int) -> int:
    """Validate and return a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _coerce_nonnegative_integer(name: str, value: int) -> int:
    """Validate and return a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _coerce_confidence_level(name: str, value: float) -> float:
    """Validate and return a confidence level in (0, 1)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        raise ValueError(f"{name} must be numeric, not string")
    raw_value = np.asarray(value)
    if _contains_boolean_or_string(raw_value):
        if raw_value.dtype == np.bool_ or (
            raw_value.dtype == object
            and any(isinstance(item, (bool, np.bool_)) for item in raw_value.flat)
        ):
            raise ValueError(f"{name} must be numeric, not boolean")
        raise ValueError(f"{name} must be numeric, not string")
    if raw_value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(numeric) or not 0.0 < numeric < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return numeric


def _format_float(value: float, *, digits: int) -> str:
    """Format a float with fixed decimal places."""
    return f"{float(value):.{digits}f}"


def _nice_step(value: float) -> float:
    exponent = math.floor(math.log10(value))
    fraction = value / 10**exponent
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * 10**exponent


def _nice_axis_ticks(
    data_min: float,
    data_max: float,
    *,
    target_count: int = 5,
) -> tuple[float, float, np.ndarray]:
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        raise ValueError("axis bounds must be finite")
    if data_min > data_max:
        raise ValueError("axis lower bound cannot exceed upper bound")
    if data_min == data_max:
        padding = max(abs(data_min) * 0.08, 1.0)
        data_min -= padding
        data_max += padding
    interval_count = max(
        _coerce_positive_integer("target_count", target_count) - 1,
        1,
    )
    raw_step = (data_max - data_min) / interval_count
    step = _nice_step(raw_step)
    axis_min = math.floor(data_min / step) * step
    axis_max = math.ceil(data_max / step) * step
    tick_count = int(round((axis_max - axis_min) / step)) + 1
    ticks = axis_min + step * np.arange(tick_count, dtype=float)
    return float(axis_min), float(axis_max), ticks


def _format_tick(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    magnitude = abs(value)
    if magnitude >= 1000.0 or (0.0 < magnitude < 0.001):
        return f"{value:.2g}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _format_svg_description_number(value: float) -> str:
    return _format_tick(float(value))


def _format_confidence_level_percent(level: float) -> str:
    percent = (Decimal(str(float(level))) * Decimal("100")).quantize(
        Decimal("0.001"),
        rounding=ROUND_DOWN,
    )
    return f"{format(percent.normalize(), 'f')}%"


def _range_includes_zero(lower: float, upper: float) -> bool:
    return float(lower) <= 0.0 <= float(upper)


def _svg_fit_text_attributes(text: str, max_width: float, *, font_size: float) -> str:
    if max_width <= 0.0:
        raise ValueError("text fit width must be positive")
    text_width = max(len(str(text)), 1) * font_size * 0.62
    if text_width <= max_width:
        return ""
    return f' textLength="{max_width:.1f}" lengthAdjust="spacingAndGlyphs"'


def _coerce_svg_id_prefix(name: str, value: str) -> str:
    if isinstance(value, (bool, np.bool_, bytes, np.bytes_)) or not isinstance(
        value,
        str,
    ):
        raise ValueError(f"{name} must be a string SVG id prefix")
    if not _SVG_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{name} must start with a letter or underscore and contain only "
            "letters, digits, underscores, periods, or hyphens"
        )
    return value


def _coerce_svg_label(name: str, value: str) -> str:
    if isinstance(value, (bool, np.bool_, bytes, np.bytes_)) or not isinstance(
        value,
        str,
    ):
        raise ValueError(f"{name} must be a string SVG label")
    label = value.strip()
    if not label:
        raise ValueError(f"{name} must be a non-empty SVG label")
    if any(token in label for token in ("\r", "\n", "\t")):
        raise ValueError(f"{name} must not contain line breaks or tabs")
    return label


def _coerce_plot_name(name: str, value: str) -> str:
    if isinstance(value, (bool, np.bool_, bytes, np.bytes_)) or not isinstance(
        value,
        str,
    ):
        raise ValueError(f"{name} must be a string plot field name")
    field_name = value.strip()
    if not field_name:
        raise ValueError(f"{name} must be a non-empty plot field name")
    if any(token in field_name for token in ("\r", "\n", "\t")):
        raise ValueError(f"{name} must not contain line breaks or tabs")
    return field_name


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _polygon(
    x_values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    x_scale: Any,
    y_scale: Any,
) -> str:
    upper_points = [
        (x_scale(float(x)), y_scale(float(y)))
        for x, y in zip(x_values, upper, strict=True)
    ]
    lower_points = [
        (x_scale(float(x)), y_scale(float(y)))
        for x, y in zip(x_values[::-1], lower[::-1], strict=True)
    ]
    return _polyline(upper_points + lower_points)


def _interval_intersects(
    lower: float,
    upper: float,
    box_lower: float,
    box_upper: float,
) -> bool:
    return max(lower, box_lower) <= min(upper, box_upper)


def _point_in_box(
    x: float,
    y: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> bool:
    return x0 <= x <= x1 and y0 <= y <= y1


def _orientation(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> float:
    return (by - ay) * (cx - bx) - (bx - ax) * (cy - by)


def _segment_intersects_segment(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
    fourth: tuple[float, float],
) -> bool:
    ax, ay = first
    bx, by = second
    cx, cy = third
    dx, dy = fourth
    orientations = (
        _orientation(ax, ay, bx, by, cx, cy),
        _orientation(ax, ay, bx, by, dx, dy),
        _orientation(cx, cy, dx, dy, ax, ay),
        _orientation(cx, cy, dx, dy, bx, by),
    )
    if all(abs(value) <= 1e-12 for value in orientations):
        return (
            max(min(ax, bx), min(cx, dx)) <= min(max(ax, bx), max(cx, dx))
            and max(min(ay, by), min(cy, dy)) <= min(max(ay, by), max(cy, dy))
        )
    return (
        orientations[0] * orientations[1] <= 0.0
        and orientations[2] * orientations[3] <= 0.0
    )


def _segment_intersects_box(
    first: tuple[float, float],
    second: tuple[float, float],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> bool:
    if _point_in_box(first[0], first[1], x0, y0, x1, y1) or _point_in_box(
        second[0],
        second[1],
        x0,
        y0,
        x1,
        y1,
    ):
        return True
    edges = (
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    )
    return any(
        _segment_intersects_segment(first, second, edge[0], edge[1])
        for edge in edges
    )


def _line_intersects_box(
    points: list[tuple[float, float]],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> bool:
    return any(
        _segment_intersects_box(first, second, x0, y0, x1, y1)
        for first, second in zip(points[:-1], points[1:], strict=True)
    )


def _band_segment_intersects_box(
    left_x: float,
    right_x: float,
    left_lower: float,
    right_lower: float,
    left_upper: float,
    right_upper: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> bool:
    overlap_left = max(min(left_x, right_x), x0)
    overlap_right = min(max(left_x, right_x), x1)
    if overlap_left > overlap_right:
        return False
    if abs(right_x - left_x) <= 1e-12:
        sample_x_values = (left_x,)
    else:
        sample_x_values = (overlap_left, overlap_right)
    for sample_x in sample_x_values:
        weight = (
            0.0
            if abs(right_x - left_x) <= 1e-12
            else (sample_x - left_x) / (right_x - left_x)
        )
        lower = left_lower + weight * (right_lower - left_lower)
        upper = left_upper + weight * (right_upper - left_upper)
        if _interval_intersects(min(lower, upper), max(lower, upper), y0, y1):
            return True
    return _segment_intersects_box(
        (left_x, left_lower),
        (right_x, right_lower),
        x0,
        y0,
        x1,
        y1,
    ) or _segment_intersects_box(
        (left_x, left_upper),
        (right_x, right_upper),
        x0,
        y0,
        x1,
        y1,
    )


def _band_intersects_box(
    x_points: list[float],
    lower_points: list[float],
    upper_points: list[float],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> bool:
    return any(
        _band_segment_intersects_box(
            left_x,
            right_x,
            left_lower,
            right_lower,
            left_upper,
            right_upper,
            x0,
            y0,
            x1,
            y1,
        )
        for left_x, right_x, left_lower, right_lower, left_upper, right_upper in zip(
            x_points[:-1],
            x_points[1:],
            lower_points[:-1],
            lower_points[1:],
            upper_points[:-1],
            upper_points[1:],
            strict=True,
        )
    )


def _choose_legend_origin(
    *,
    plot_left: float,
    plot_right: float,
    plot_top: float,
    plot_bottom: float,
    legend_box_width: float,
    legend_box_height: float,
    x_values: np.ndarray,
    estimate_values: np.ndarray,
    pointwise_lower_values: np.ndarray,
    pointwise_upper_values: np.ndarray,
    uniform_lower_values: np.ndarray,
    uniform_upper_values: np.ndarray,
    x_scale: Any,
    y_scale: Any,
) -> tuple[float, float]:
    candidates = (
        (plot_right - legend_box_width - 8.0, plot_top - 1.0),
        (plot_left + 8.0, plot_top - 1.0),
        (plot_right - legend_box_width - 8.0, plot_bottom - legend_box_height - 8.0),
        (plot_left + 8.0, plot_bottom - legend_box_height - 8.0),
    )

    def score(indexed_candidate: tuple[int, tuple[float, float]]) -> tuple[int, int]:
        candidate_index, candidate = indexed_candidate
        x0, y0 = candidate
        x1 = x0 + legend_box_width
        y1 = y0 + legend_box_height
        overlap_score = 0
        scaled_x_values = [x_scale(float(value)) for value in x_values]
        scaled_estimate_values = [y_scale(float(value)) for value in estimate_values]
        for values, weight in (
            (estimate_values, 6),
            (pointwise_lower_values, 3),
            (pointwise_upper_values, 3),
            (uniform_lower_values, 2),
            (uniform_upper_values, 2),
        ):
            for x_value, y_value in zip(x_values, values, strict=True):
                x = x_scale(float(x_value))
                y = y_scale(float(y_value))
                if x0 <= x <= x1 and y0 <= y <= y1:
                    overlap_score += weight
        if _line_intersects_box(
            list(zip(scaled_x_values, scaled_estimate_values, strict=True)),
            x0,
            y0,
            x1,
            y1,
        ):
            overlap_score += 12
        for lower_values, upper_values, weight in (
            (pointwise_lower_values, pointwise_upper_values, 5),
            (uniform_lower_values, uniform_upper_values, 8),
        ):
            for x_value, lower_value, upper_value in zip(
                x_values,
                lower_values,
                upper_values,
                strict=True,
            ):
                x = x_scale(float(x_value))
                if not x0 <= x <= x1:
                    continue
                y_lower = y_scale(float(lower_value))
                y_upper = y_scale(float(upper_value))
                band_top = min(y_lower, y_upper)
                band_bottom = max(y_lower, y_upper)
                if _interval_intersects(band_top, band_bottom, y0, y1):
                    overlap_score += weight
            if _band_intersects_box(
                scaled_x_values,
                [y_scale(float(value)) for value in lower_values],
                [y_scale(float(value)) for value in upper_values],
                x0,
                y0,
                x1,
                y1,
            ):
                overlap_score += weight * 2
        return overlap_score, candidate_index

    return min(enumerate(candidates), key=score)[1]


@dataclass(slots=True)
class NonparametricEffectPlotData:
    """Plot-ready data for the nonparametric effect f(z) visualization.

    Contains the evaluation grid, point estimates, pointwise confidence
    interval bounds, and uniform band bounds extracted from an
    :class:`HDDIDResult`.
    """
    z0: np.ndarray
    estimate: np.ndarray
    pointwise_lower: np.ndarray
    pointwise_upper: np.ndarray
    uniform_lower: np.ndarray
    uniform_upper: np.ndarray
    estimate_name: str = "bar_f_at_z0"
    pointwise_interval_name: str = "nonparametric_ci"
    uniform_band_name: str = "uniform_band"
    pointwise_level: float | None = None
    uniform_level: float | None = None

    def __post_init__(self) -> None:
        self.z0 = _coerce_finite_vector("z0", self.z0)
        self.estimate = _coerce_finite_vector("estimate", self.estimate)
        self.pointwise_lower = _coerce_finite_vector(
            "pointwise_lower",
            self.pointwise_lower,
        )
        self.pointwise_upper = _coerce_finite_vector(
            "pointwise_upper",
            self.pointwise_upper,
        )
        self.uniform_lower = _coerce_finite_vector(
            "uniform_lower",
            self.uniform_lower,
        )
        self.uniform_upper = _coerce_finite_vector(
            "uniform_upper",
            self.uniform_upper,
        )
        lengths = {
            self.z0.shape[0],
            self.estimate.shape[0],
            self.pointwise_lower.shape[0],
            self.pointwise_upper.shape[0],
            self.uniform_lower.shape[0],
            self.uniform_upper.shape[0],
        }
        if len(lengths) != 1:
            raise ValueError("plot vectors must have matching lengths")
        if np.any(np.diff(self.z0) <= 0.0):
            raise ValueError("z0 must be strictly increasing for line plotting")
        if np.any(self.pointwise_lower > self.pointwise_upper):
            raise ValueError("pointwise interval lower bounds cannot exceed upper bounds")
        if np.any(self.uniform_lower > self.uniform_upper):
            raise ValueError("uniform band lower bounds cannot exceed upper bounds")
        if np.any(self.estimate < self.pointwise_lower) or np.any(
            self.estimate > self.pointwise_upper
        ):
            raise ValueError("estimate must lie inside pointwise interval")
        if np.any(self.pointwise_lower < self.uniform_lower) or np.any(
            self.pointwise_upper > self.uniform_upper
        ):
            raise ValueError("pointwise interval must lie inside uniform band")
        self.estimate_name = _coerce_plot_name("estimate_name", self.estimate_name)
        self.pointwise_interval_name = _coerce_plot_name(
            "pointwise_interval_name",
            self.pointwise_interval_name,
        )
        self.uniform_band_name = _coerce_plot_name(
            "uniform_band_name",
            self.uniform_band_name,
        )
        for level_name in ("pointwise_level", "uniform_level"):
            level = getattr(self, level_name)
            if level is None:
                continue
            setattr(self, level_name, _coerce_confidence_level(level_name, level))

    def to_csv(self, *, digits: int = 12) -> str:
        digits_value = _coerce_nonnegative_integer("digits", digits)
        lines = [
            "z0,estimate,pointwise_lower,pointwise_upper,uniform_lower,uniform_upper"
        ]
        for row in zip(
            self.z0,
            self.estimate,
            self.pointwise_lower,
            self.pointwise_upper,
            self.uniform_lower,
            self.uniform_upper,
            strict=True,
        ):
            lines.append(
                ",".join(_format_float(value, digits=digits_value) for value in row)
            )
        return "\n".join(lines) + "\n"

    def to_metadata(self, *, digits: int = 12) -> dict[str, Any]:
        digits_value = _coerce_nonnegative_integer("digits", digits)

        def rounded(value: float) -> float:
            return float(_format_float(float(value), digits=digits_value))

        pointwise_width = self.pointwise_upper - self.pointwise_lower
        uniform_width = self.uniform_upper - self.uniform_lower
        uniform_width_excess = uniform_width - pointwise_width
        estimate_includes_zero = _range_includes_zero(
            float(np.min(self.estimate)),
            float(np.max(self.estimate)),
        )
        pointwise_interval_includes_zero = bool(
            np.any((self.pointwise_lower <= 0.0) & (self.pointwise_upper >= 0.0))
        )
        uniform_band_includes_zero = bool(
            np.any((self.uniform_lower <= 0.0) & (self.uniform_upper >= 0.0))
        )
        return {
            "plot_contract": "nonparametric-effect",
            "estimate_name": self.estimate_name,
            "pointwise_interval_name": self.pointwise_interval_name,
            "uniform_band_name": self.uniform_band_name,
            "pointwise_level": (
                rounded(self.pointwise_level)
                if self.pointwise_level is not None
                else None
            ),
            "uniform_level": (
                rounded(self.uniform_level) if self.uniform_level is not None else None
            ),
            "grid_size": int(self.z0.shape[0]),
            "z0_min": rounded(np.min(self.z0)),
            "z0_max": rounded(np.max(self.z0)),
            "estimate_min": rounded(np.min(self.estimate)),
            "estimate_max": rounded(np.max(self.estimate)),
            "pointwise_width_min": rounded(np.min(pointwise_width)),
            "pointwise_width_mean": rounded(np.mean(pointwise_width)),
            "pointwise_width_max": rounded(np.max(pointwise_width)),
            "uniform_width_min": rounded(np.min(uniform_width)),
            "uniform_width_mean": rounded(np.mean(uniform_width)),
            "uniform_width_max": rounded(np.max(uniform_width)),
            "uniform_width_excess_min": rounded(np.min(uniform_width_excess)),
            "uniform_width_excess_mean": rounded(np.mean(uniform_width_excess)),
            "uniform_width_excess_max": rounded(np.max(uniform_width_excess)),
            "estimate_includes_zero": estimate_includes_zero,
            "pointwise_interval_includes_zero": pointwise_interval_includes_zero,
            "uniform_band_includes_zero": uniform_band_includes_zero,
            "zero_reference_line": (
                estimate_includes_zero
                or pointwise_interval_includes_zero
                or uniform_band_includes_zero
            ),
        }

    def to_svg(
        self,
        *,
        title: str = "Nonparametric HDDID effect",
        x_label: str = "Evaluation point z0",
        y_label: str = "Effect estimate",
        width: int = 720,
        height: int = 420,
        id_prefix: str = "hddid-nonparametric-effect",
    ) -> str:
        width_value = _coerce_positive_integer("width", width)
        height_value = _coerce_positive_integer("height", height)
        if width_value < 360 or height_value < 260:
            raise ValueError("SVG dimensions are too small for labeled axes")
        id_prefix_value = _coerce_svg_id_prefix("id_prefix", id_prefix)
        title_value = _coerce_svg_label("title", title)
        x_label_value = _coerce_svg_label("x_label", x_label)
        y_label_value = _coerce_svg_label("y_label", y_label)
        title_id = f"{id_prefix_value}-title"
        desc_id = f"{id_prefix_value}-desc"

        margin_left = 76.0
        margin_right = 28.0
        margin_top = 46.0
        margin_bottom = 64.0
        plot_left = margin_left
        plot_right = float(width_value) - margin_right
        plot_top = margin_top
        plot_bottom = float(height_value) - margin_bottom
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top
        x_data_min = float(np.min(self.z0))
        x_data_max = float(np.max(self.z0))
        x_padding = max((x_data_max - x_data_min) * 0.04, 1e-9)
        x_min = x_data_min - x_padding
        x_max = x_data_max + x_padding
        y_data_min = float(np.min(self.uniform_lower))
        y_data_max = float(np.max(self.uniform_upper))
        y_padding = max((y_data_max - y_data_min) * 0.08, 1e-9)
        y_min, y_max, y_ticks = _nice_axis_ticks(
            y_data_min - y_padding,
            y_data_max + y_padding,
        )
        estimate_includes_zero = _range_includes_zero(
            float(np.min(self.estimate)),
            float(np.max(self.estimate)),
        )
        pointwise_interval_includes_zero = bool(
            np.any((self.pointwise_lower <= 0.0) & (self.pointwise_upper >= 0.0))
        )
        uniform_band_includes_zero = bool(
            np.any((self.uniform_lower <= 0.0) & (self.uniform_upper >= 0.0))
        )
        zero_reference_line = (
            estimate_includes_zero
            or pointwise_interval_includes_zero
            or uniform_band_includes_zero
        )

        def x_scale(value: float) -> float:
            return plot_left + (value - x_min) / (x_max - x_min) * (
                plot_right - plot_left
            )

        def y_scale(value: float) -> float:
            return plot_bottom - (value - y_min) / (y_max - y_min) * (
                plot_bottom - plot_top
            )

        grid_count = 5
        x_ticks = (
            self.z0
            if self.z0.shape[0] <= 6
            else np.linspace(x_data_min, x_data_max, grid_count)
        )
        estimate_points = [
            (x_scale(float(x)), y_scale(float(y)))
            for x, y in zip(self.z0, self.estimate, strict=True)
        ]
        pointwise_polygon = _polygon(
            self.z0,
            self.pointwise_lower,
            self.pointwise_upper,
            x_scale=x_scale,
            y_scale=y_scale,
        )
        uniform_polygon = _polygon(
            self.z0,
            self.uniform_lower,
            self.uniform_upper,
            x_scale=x_scale,
            y_scale=y_scale,
        )

        title_text = escape(title_value)
        x_label_text = escape(x_label_value)
        y_label_text = escape(y_label_value)
        pointwise_label = (
            f"{_format_confidence_level_percent(self.pointwise_level)} pointwise interval"
            if self.pointwise_level is not None
            else "Pointwise interval"
        )
        uniform_label = (
            f"{_format_confidence_level_percent(self.uniform_level)} uniform band"
            if self.uniform_level is not None
            else "Uniform band"
        )
        title_fit = _svg_fit_text_attributes(
            title_value,
            width_value - 32.0,
            font_size=17.0,
        )
        x_label_fit = _svg_fit_text_attributes(
            x_label_value,
            plot_width,
            font_size=14.0,
        )
        y_label_fit = _svg_fit_text_attributes(
            y_label_value,
            plot_height,
            font_size=14.0,
        )
        legend_box_width = 164.0
        legend_box_height = 61.0
        legend_label_width = legend_box_width - 44.0
        legend_x, legend_y = _choose_legend_origin(
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            legend_box_width=legend_box_width,
            legend_box_height=legend_box_height,
            x_values=self.z0,
            estimate_values=self.estimate,
            pointwise_lower_values=self.pointwise_lower,
            pointwise_upper_values=self.pointwise_upper,
            uniform_lower_values=self.uniform_lower,
            uniform_upper_values=self.uniform_upper,
            x_scale=x_scale,
            y_scale=y_scale,
        )
        uniform_label_fit = _svg_fit_text_attributes(
            uniform_label,
            legend_label_width,
            font_size=12.0,
        )
        pointwise_label_fit = _svg_fit_text_attributes(
            pointwise_label,
            legend_label_width,
            font_size=12.0,
        )
        desc_parts = [
            "Nonparametric effect curve",
            f"{self.z0.shape[0]} plotted evaluation points",
            (
                "z0 range "
                f"{_format_svg_description_number(np.min(self.z0))} to "
                f"{_format_svg_description_number(np.max(self.z0))}"
            ),
            (
                "estimate range "
                f"{_format_svg_description_number(np.min(self.estimate))} to "
                f"{_format_svg_description_number(np.max(self.estimate))}"
            ),
            pointwise_label,
            uniform_label,
        ]
        if estimate_includes_zero:
            desc_parts.append("estimate range includes zero")
        if pointwise_interval_includes_zero:
            desc_parts.append("pointwise interval includes zero")
        if uniform_band_includes_zero:
            desc_parts.append("uniform band includes zero")
        if zero_reference_line:
            desc_parts.append("dashed zero reference line")
        desc_text = escape("; ".join(desc_parts) + ".")
        svg_metadata = {
            "plot_contract": "nonparametric-effect-svg",
            "id_prefix": id_prefix_value,
            "title": title_value,
            "x_label": x_label_value,
            "y_label": y_label_value,
            "width": width_value,
            "height": height_value,
            "data": self.to_metadata(),
            "legend_origin": {
                "x": round(legend_x, 1),
                "y": round(legend_y, 1),
                "width": round(legend_box_width, 1),
                "height": round(legend_box_height, 1),
            },
        }
        svg_metadata_text = escape(
            json.dumps(svg_metadata, sort_keys=True, separators=(",", ":")),
            quote=False,
        )

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_value}" '
            f'height="{height_value}" viewBox="0 0 {width_value} {height_value}" '
            f'role="img" aria-labelledby="{title_id} {desc_id}">',
            f'<title id="{title_id}">{title_text}</title>',
            f'<desc id="{desc_id}">{desc_text}</desc>',
            f"<metadata>{svg_metadata_text}</metadata>",
            "<style>"
            ".axis{stroke:#30343b;stroke-width:1.2}"
            ".grid{stroke:#d8dde6;stroke-width:0.8}"
            ".zero{stroke:#6b7280;stroke-width:1.1;stroke-dasharray:4 3}"
            ".tick{fill:#4f5663;font:12px sans-serif}"
            ".label{fill:#20242b;font:14px sans-serif}"
            ".title{fill:#111827;font:600 17px sans-serif}"
            ".legend-box{fill:white;fill-opacity:0.88;stroke:#d8dde6;stroke-width:0.8}"
            ".legend{fill:#2d3440;font:12px sans-serif}"
            ".estimate{fill:none;stroke:#1f5aa6;stroke-width:2.4}"
            ".pointwise{fill:#7fb7e8;opacity:0.42}"
            ".uniform{fill:#f2b35b;opacity:0.28}"
            ".dot{fill:#1f5aa6;stroke:white;stroke-width:1.2}"
            "</style>",
            f'<rect x="0" y="0" width="{width_value}" height="{height_value}" fill="white"/>',
            f'<text class="title" x="{width_value / 2:.1f}" y="24" '
            f'text-anchor="middle"{title_fit}>{title_text}</text>',
        ]
        for tick in y_ticks:
            y = y_scale(float(tick))
            parts.append(
                f'<line class="grid" x1="{plot_left:.1f}" y1="{y:.1f}" '
                f'x2="{plot_right:.1f}" y2="{y:.1f}"/>'
            )
            parts.append(
                f'<text class="tick" x="{plot_left - 9:.1f}" y="{y + 4:.1f}" '
                f'text-anchor="end">{_format_tick(float(tick))}</text>'
            )
        for tick in x_ticks:
            x = x_scale(float(tick))
            parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{plot_top:.1f}" '
                f'x2="{x:.1f}" y2="{plot_bottom:.1f}"/>'
            )
            parts.append(
                f'<text class="tick" x="{x:.1f}" y="{plot_bottom + 22:.1f}" '
                f'text-anchor="middle">{_format_tick(float(tick))}</text>'
            )
        if zero_reference_line:
            y_zero = y_scale(0.0)
            parts.append(
                f'<line class="zero" x1="{plot_left:.1f}" y1="{y_zero:.1f}" '
                f'x2="{plot_right:.1f}" y2="{y_zero:.1f}"/>'
            )
        parts.extend(
            [
                f'<polygon class="uniform" points="{uniform_polygon}"/>',
                f'<polygon class="pointwise" points="{pointwise_polygon}"/>',
                f'<polyline class="estimate" points="{_polyline(estimate_points)}"/>',
            ]
        )
        for index, (x, y) in enumerate(estimate_points):
            point_title = escape(
                "evaluation point "
                f"{index + 1}: z0={_format_tick(float(self.z0[index]))}, "
                f"estimate={_format_tick(float(self.estimate[index]))}, "
                "pointwise interval "
                f"[{_format_tick(float(self.pointwise_lower[index]))}, "
                f"{_format_tick(float(self.pointwise_upper[index]))}], "
                "uniform band "
                f"[{_format_tick(float(self.uniform_lower[index]))}, "
                f"{_format_tick(float(self.uniform_upper[index]))}]"
            )
            parts.extend(
                [
                    '<g class="data-point">',
                    f"<title>{point_title}</title>",
                    f'<circle class="dot" cx="{x:.2f}" cy="{y:.2f}" r="3.4"/>',
                    "</g>",
                ]
            )
        parts.extend(
            [
                f'<line class="axis" x1="{plot_left:.1f}" y1="{plot_bottom:.1f}" '
                f'x2="{plot_right:.1f}" y2="{plot_bottom:.1f}"/>',
                f'<line class="axis" x1="{plot_left:.1f}" y1="{plot_top:.1f}" '
                f'x2="{plot_left:.1f}" y2="{plot_bottom:.1f}"/>',
                f'<text class="label" x="{(plot_left + plot_right) / 2:.1f}" '
                f'y="{height_value - 18:.1f}" text-anchor="middle"{x_label_fit}>{x_label_text}</text>',
                f'<text class="label" transform="translate(20 {(plot_top + plot_bottom) / 2:.1f}) rotate(-90)" '
                f'text-anchor="middle"{y_label_fit}>{y_label_text}</text>',
                f'<rect class="legend-box" x="{legend_x:.1f}" y="{legend_y:.1f}" width="{legend_box_width:.0f}" height="{legend_box_height:.0f}" rx="2"/>',
                f'<rect class="uniform" x="{legend_x + 12.0:.1f}" y="{legend_y + 7.0:.1f}" width="24" height="10"/>',
                f'<text class="legend" x="{legend_x + 44.0:.1f}" y="{legend_y + 17.0:.1f}"{uniform_label_fit}>{escape(uniform_label)}</text>',
                f'<rect class="pointwise" x="{legend_x + 12.0:.1f}" y="{legend_y + 25.0:.1f}" width="24" height="10"/>',
                f'<text class="legend" x="{legend_x + 44.0:.1f}" y="{legend_y + 35.0:.1f}"{pointwise_label_fit}>{escape(pointwise_label)}</text>',
                f'<line class="estimate" x1="{legend_x + 12.0:.1f}" y1="{legend_y + 49.0:.1f}" '
                f'x2="{legend_x + 36.0:.1f}" y2="{legend_y + 49.0:.1f}"/>',
                f'<text class="legend" x="{legend_x + 44.0:.1f}" y="{legend_y + 53.0:.1f}">Estimate</text>',
                "</svg>",
            ]
        )
        return "\n".join(parts) + "\n"


def build_nonparametric_effect_plot_data(
    result: HDDIDResult,
    z0: Any,
    *,
    estimate_name: str = "bar_f_at_z0",
    pointwise_interval_name: str = "nonparametric_ci",
    uniform_band_name: str = "uniform_band",
) -> NonparametricEffectPlotData:
    """Extract nonparametric effect plot data from an HDDIDResult.

    Parameters
    ----------
    result : HDDIDResult
        Result object containing nonparametric estimates and intervals.
    z0 : array-like
        Evaluation grid points for f(z).
    estimate_name : str, default "bar_f_at_z0"
        Key in ``result.nonparametric_estimates`` for the point estimate.
    pointwise_interval_name : str, default "nonparametric_ci"
        Key in ``result.intervals`` for the pointwise CI.
    uniform_band_name : str, default "uniform_band"
        Key in ``result.intervals`` for the uniform band.

    Returns
    -------
    NonparametricEffectPlotData
        Plot-ready data containing z0, point estimates, pointwise CI,
        and uniform band.
    """
    if not isinstance(result, HDDIDResult):
        raise ValueError("result must be an HDDIDResult")
    estimate_name_value = _coerce_plot_name("estimate_name", estimate_name)
    pointwise_interval_name_value = _coerce_plot_name(
        "pointwise_interval_name",
        pointwise_interval_name,
    )
    uniform_band_name_value = _coerce_plot_name("uniform_band_name", uniform_band_name)
    if estimate_name_value not in result.nonparametric_estimates:
        raise ValueError(
            f"result is missing nonparametric estimate '{estimate_name_value}'"
        )
    if pointwise_interval_name_value not in result.intervals:
        raise ValueError(
            f"result is missing interval '{pointwise_interval_name_value}'"
        )
    if uniform_band_name_value not in result.intervals:
        raise ValueError(f"result is missing interval '{uniform_band_name_value}'")

    pointwise_interval = result.intervals[pointwise_interval_name_value]
    uniform_band = result.intervals[uniform_band_name_value]
    if not isinstance(pointwise_interval, ConfidenceInterval):
        raise ValueError(
            f"interval '{pointwise_interval_name_value}' must be a ConfidenceInterval"
        )
    if not isinstance(uniform_band, UniformBand):
        raise ValueError(f"interval '{uniform_band_name_value}' must be a UniformBand")

    return NonparametricEffectPlotData(
        z0=_coerce_finite_vector("z0", z0),
        estimate=_coerce_finite_vector(
            estimate_name_value,
            result.nonparametric_estimates[estimate_name_value],
        ),
        pointwise_lower=_coerce_finite_vector(
            f"{pointwise_interval_name_value}.lower",
            pointwise_interval.lower,
        ),
        pointwise_upper=_coerce_finite_vector(
            f"{pointwise_interval_name_value}.upper",
            pointwise_interval.upper,
        ),
        uniform_lower=_coerce_finite_vector(
            f"{uniform_band_name_value}.lower",
            uniform_band.lower,
        ),
        uniform_upper=_coerce_finite_vector(
            f"{uniform_band_name_value}.upper",
            uniform_band.upper,
        ),
        estimate_name=estimate_name_value,
        pointwise_interval_name=pointwise_interval_name_value,
        uniform_band_name=uniform_band_name_value,
        pointwise_level=pointwise_interval.level,
        uniform_level=uniform_band.level,
    )
