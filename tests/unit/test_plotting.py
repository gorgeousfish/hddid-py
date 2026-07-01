import html
import json
import re

import numpy as np
import pytest

from hddid.plotting import (
    NonparametricEffectPlotData,
    build_nonparametric_effect_plot_data,
)
from hddid.results import ConfidenceInterval, HDDIDResult, UniformBand


class _StringArrayLike:
    def __array__(self, dtype: object = None) -> np.ndarray:
        values = np.array(["0.2", "0.5", "0.4"], dtype=object)
        if dtype is None:
            return values
        return values.astype(dtype)


def _nonparametric_result() -> HDDIDResult:
    return HDDIDResult(
        nonparametric_estimates={"bar_f_at_z0": np.array([0.2, 0.5, 0.4])},
        intervals={
            "nonparametric_ci": ConfidenceInterval(
                lower=np.array([0.1, 0.35, 0.2]),
                upper=np.array([0.3, 0.65, 0.6]),
                level=0.9,
            ),
            "uniform_band": UniformBand(
                lower=np.array([0.0, 0.2, 0.1]),
                upper=np.array([0.4, 0.8, 0.7]),
                level=0.9,
                critical_value=2.1,
                n_boot=256,
                random_state=123,
            ),
        },
    )


def test_build_nonparametric_effect_plot_data_exports_csv_and_svg() -> None:
    plot_data = build_nonparametric_effect_plot_data(
        _nonparametric_result(),
        z0=np.array([0.25, 0.75, 1.25]),
    )

    csv_text = plot_data.to_csv(digits=4)
    svg_text = plot_data.to_svg(title="Validation slice effect")

    assert csv_text.splitlines()[0] == (
        "z0,estimate,pointwise_lower,pointwise_upper,uniform_lower,uniform_upper"
    )
    assert "0.7500,0.5000,0.3500,0.6500,0.2000,0.8000" in csv_text
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg_text
    assert (
        'role="img" aria-labelledby="hddid-nonparametric-effect-title '
        'hddid-nonparametric-effect-desc"'
    ) in svg_text
    assert (
        '<title id="hddid-nonparametric-effect-title">Validation slice effect</title>'
    ) in svg_text
    assert (
        '<desc id="hddid-nonparametric-effect-desc">Nonparametric effect curve; '
        "3 plotted evaluation points; z0 range 0.25 to 1.25; "
        "estimate range 0.2 to 0.5; "
        "90% pointwise interval; 90% uniform band; "
        "uniform band includes zero; dashed zero reference line"
    ) in svg_text
    metadata_match = re.search(r"<metadata>(.*?)</metadata>", svg_text)
    assert metadata_match is not None
    svg_metadata = json.loads(html.unescape(metadata_match.group(1)))
    assert svg_metadata["plot_contract"] == "nonparametric-effect-svg"
    assert svg_metadata["id_prefix"] == "hddid-nonparametric-effect"
    assert svg_metadata["title"] == "Validation slice effect"
    assert svg_metadata["x_label"] == "Evaluation point z0"
    assert svg_metadata["y_label"] == "Effect estimate"
    assert svg_metadata["width"] == 720
    assert svg_metadata["height"] == 420
    assert svg_metadata["legend_origin"] == {
        "x": 84.0,
        "y": 45.0,
        "width": 164.0,
        "height": 61.0,
    }
    assert svg_metadata["data"]["plot_contract"] == "nonparametric-effect"
    assert svg_metadata["data"]["uniform_band_includes_zero"] is True
    assert "Validation slice effect" in svg_text
    assert "90% pointwise interval" in svg_text
    assert "90% uniform band" in svg_text
    assert '<polyline class="estimate"' in svg_text
    assert svg_text.count("<polygon") == 2
    assert svg_text.count('<g class="data-point">') == 3
    assert svg_text.count("<title>evaluation point") == 3
    assert (
        "<title>evaluation point 2: z0=0.75, estimate=0.5, "
        "pointwise interval [0.35, 0.65], uniform band [0.2, 0.8]</title>"
    ) in svg_text
    assert '<rect class="legend-box"' in svg_text
    assert svg_text.count('textLength="120.0"') == 1
    assert (
        '<text class="legend" x="128.0" y="62.0">90% uniform band</text>'
    ) in svg_text
    assert (
        '<text class="legend" x="128.0" y="80.0" textLength="120.0" '
        'lengthAdjust="spacingAndGlyphs">90% pointwise interval</text>'
    ) in svg_text
    assert ">0.25</text>" in svg_text
    assert ">0.75</text>" in svg_text
    assert ">1.25</text>" in svg_text
    assert ">-0.0625</text>" not in svg_text
    assert ">-0.5</text>" in svg_text
    assert ">0.5</text>" in svg_text
    assert ">1</text>" in svg_text
    assert ">0.864</text>" not in svg_text


def test_nonparametric_effect_plot_data_exports_metadata_summary() -> None:
    plot_data = build_nonparametric_effect_plot_data(
        _nonparametric_result(),
        z0=np.array([0.25, 0.75, 1.25]),
    )

    metadata = plot_data.to_metadata(digits=4)

    assert metadata == {
        "plot_contract": "nonparametric-effect",
        "estimate_name": "bar_f_at_z0",
        "pointwise_interval_name": "nonparametric_ci",
        "uniform_band_name": "uniform_band",
        "pointwise_level": 0.9,
        "uniform_level": 0.9,
        "grid_size": 3,
        "z0_min": 0.25,
        "z0_max": 1.25,
        "estimate_min": 0.2,
        "estimate_max": 0.5,
        "pointwise_width_min": 0.2,
        "pointwise_width_mean": 0.3,
        "pointwise_width_max": 0.4,
        "uniform_width_min": 0.4,
        "uniform_width_mean": 0.5333,
        "uniform_width_max": 0.6,
        "uniform_width_excess_min": 0.2,
        "uniform_width_excess_mean": 0.2333,
        "uniform_width_excess_max": 0.3,
        "estimate_includes_zero": False,
        "pointwise_interval_includes_zero": False,
        "uniform_band_includes_zero": True,
        "zero_reference_line": True,
    }


def test_nonparametric_effect_plot_data_allows_integer_precision_exports() -> None:
    plot_data = build_nonparametric_effect_plot_data(
        _nonparametric_result(),
        z0=np.array([0.25, 0.75, 1.25]),
    )

    csv_text = plot_data.to_csv(digits=0)
    metadata = plot_data.to_metadata(digits=0)

    assert "0,0,0,0,0,0" in csv_text
    assert "1,0,0,1,0,1" in csv_text
    assert ".000" not in csv_text
    assert metadata["z0_min"] == 0.0
    assert metadata["z0_max"] == 1.0
    assert isinstance(metadata["estimate_min"], float)
    assert isinstance(metadata["estimate_max"], float)

    for invalid_digits in (-1, 1.5, True):
        with pytest.raises(ValueError, match="digits"):
            plot_data.to_csv(digits=invalid_digits)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="digits"):
            plot_data.to_metadata(digits=invalid_digits)  # type: ignore[arg-type]


def test_nonparametric_effect_svg_fits_long_labels_inside_canvas() -> None:
    plot_data = build_nonparametric_effect_plot_data(
        _nonparametric_result(),
        z0=np.array([0.25, 0.75, 1.25]),
    )

    svg_text = plot_data.to_svg(
        title="Core validation nonparametric effect with deliberately long appendix label",
        x_label="Evaluation point z0 with long manuscript appendix descriptor",
        y_label="Debiased nonparametric function estimate with uniform confidence band",
        width=360,
        height=260,
    )

    assert 'textLength="328.0"' in svg_text
    assert 'textLength="256.0"' in svg_text
    assert 'textLength="150.0"' in svg_text
    assert "Core validation nonparametric effect" in svg_text
    assert "Debiased nonparametric function estimate" in svg_text


def test_nonparametric_effect_svg_moves_legend_away_from_interval_band() -> None:
    plot_data = NonparametricEffectPlotData(
        z0=np.array([-0.5, 0.75, 1.25]),
        estimate=np.array([0.231, 0.233, 0.542]),
        pointwise_lower=np.array([-0.128, -0.066, 0.054]),
        pointwise_upper=np.array([0.589, 0.531, 1.029]),
        uniform_lower=np.array([-0.206, -0.13, -0.052]),
        uniform_upper=np.array([0.667, 0.596, 1.135]),
        pointwise_level=0.9,
        uniform_level=0.9,
    )

    svg_text = plot_data.to_svg(title="Core validation nonparametric effect")

    assert '<rect class="legend-box" x="84.0" y="45.0"' in svg_text
    assert '<rect class="legend-box" x="520.0" y="45.0"' not in svg_text


def test_nonparametric_effect_svg_avoids_sparse_segment_legend_collision() -> None:
    plot_data = NonparametricEffectPlotData(
        z0=np.array([0.0, 7.2, 10.0]),
        estimate=np.array([0.0, 5.0, 0.0]),
        pointwise_lower=np.array([-0.01, 4.99, -0.01]),
        pointwise_upper=np.array([0.01, 5.01, 0.01]),
        uniform_lower=np.array([-0.02, 4.98, -0.02]),
        uniform_upper=np.array([0.02, 5.02, 0.02]),
        pointwise_level=0.9,
        uniform_level=0.9,
    )

    svg_text = plot_data.to_svg(title="Sparse grid effect")

    assert '<rect class="legend-box" x="84.0" y="45.0"' in svg_text
    assert '<rect class="legend-box" x="520.0" y="45.0"' not in svg_text


def test_nonparametric_effect_svg_avoids_sloped_band_boundary_legend_collision() -> None:
    plot_data = NonparametricEffectPlotData(
        z0=np.array([0.0, 10.0]),
        estimate=np.array([0.0, 6.5]),
        pointwise_lower=np.array([-0.02, 6.48]),
        pointwise_upper=np.array([0.02, 6.52]),
        uniform_lower=np.array([-0.04, 6.46]),
        uniform_upper=np.array([0.04, 6.54]),
        pointwise_level=0.9,
        uniform_level=0.9,
    )

    svg_text = plot_data.to_svg(title="Sloped sparse-grid effect")

    assert '<rect class="legend-box" x="84.0" y="45.0"' in svg_text
    assert '<rect class="legend-box" x="520.0" y="45.0"' not in svg_text


def test_nonparametric_effect_svg_allows_distinct_accessible_id_prefixes() -> None:
    plot_data = build_nonparametric_effect_plot_data(
        _nonparametric_result(),
        z0=np.array([0.25, 0.75, 1.25]),
    )

    first_svg = plot_data.to_svg(title="First", id_prefix="effect-a")
    second_svg = plot_data.to_svg(title="Second", id_prefix="effect-b")

    assert 'aria-labelledby="effect-a-title effect-a-desc"' in first_svg
    assert '<title id="effect-a-title">First</title>' in first_svg
    assert '<desc id="effect-a-desc">Nonparametric effect curve;' in first_svg
    assert 'aria-labelledby="effect-b-title effect-b-desc"' in second_svg
    assert '<title id="effect-b-title">Second</title>' in second_svg
    assert '<desc id="effect-b-desc">Nonparametric effect curve;' in second_svg
    assert 'id="title"' not in first_svg
    assert 'id="desc"' not in first_svg


def test_nonparametric_effect_svg_marks_zero_reference_when_effect_range_crosses_zero() -> None:
    plot_data = NonparametricEffectPlotData(
        z0=np.array([0.25, 0.75, 1.25]),
        estimate=np.array([-0.2, 0.0, 0.3]),
        pointwise_lower=np.array([-0.35, -0.15, 0.1]),
        pointwise_upper=np.array([-0.05, 0.15, 0.5]),
        uniform_lower=np.array([-0.5, -0.25, 0.0]),
        uniform_upper=np.array([0.1, 0.25, 0.65]),
        pointwise_level=0.9,
        uniform_level=0.9,
    )

    svg_text = plot_data.to_svg()

    assert ".zero{stroke:#6b7280" in svg_text
    assert '<line class="zero"' in svg_text
    assert "estimate range includes zero" in svg_text
    assert "pointwise interval includes zero" in svg_text
    assert "uniform band includes zero" in svg_text
    assert "dashed zero reference line" in svg_text


def test_nonparametric_effect_svg_does_not_round_high_confidence_to_100_percent() -> None:
    plot_data = NonparametricEffectPlotData(
        z0=np.array([0.25, 0.75, 1.25]),
        estimate=np.array([0.2, 0.5, 0.4]),
        pointwise_lower=np.array([0.1, 0.35, 0.2]),
        pointwise_upper=np.array([0.3, 0.65, 0.6]),
        uniform_lower=np.array([0.0, 0.2, 0.1]),
        uniform_upper=np.array([0.4, 0.8, 0.7]),
        pointwise_level=0.999,
        uniform_level=0.9999,
    )

    svg_text = plot_data.to_svg()

    assert "99.9% pointwise interval" in svg_text
    assert "99.99% uniform band" in svg_text
    assert "100% pointwise interval" not in svg_text
    assert "100% uniform band" not in svg_text


def test_nonparametric_effect_svg_does_not_draw_padding_only_zero_reference() -> None:
    plot_data = NonparametricEffectPlotData(
        z0=np.array([0.25, 0.75, 1.25]),
        estimate=np.array([0.05, 0.07, 0.09]),
        pointwise_lower=np.array([0.03, 0.05, 0.07]),
        pointwise_upper=np.array([0.07, 0.09, 0.11]),
        uniform_lower=np.array([0.01, 0.03, 0.05]),
        uniform_upper=np.array([0.09, 0.11, 0.13]),
        pointwise_level=0.9,
        uniform_level=0.9,
    )

    svg_text = plot_data.to_svg()
    metadata = plot_data.to_metadata()

    assert '<line class="zero"' not in svg_text
    assert "dashed zero reference line" not in svg_text
    assert metadata["zero_reference_line"] is False
    assert metadata["estimate_includes_zero"] is False
    assert metadata["pointwise_interval_includes_zero"] is False
    assert metadata["uniform_band_includes_zero"] is False


def test_nonparametric_effect_plot_data_rejects_incoherent_plot_contracts() -> None:
    invalid_cases = (
        (
            {"z0": np.array([False, True, True])},
            "z0 must be a numeric vector",
        ),
        (
            {"estimate": ["0.2", "0.5", "0.4"]},
            "estimate must be a numeric vector",
        ),
        (
            {"estimate": _StringArrayLike()},
            "estimate must be a numeric vector",
        ),
        (
            {"z0": np.array([0.25, 0.25, 0.75])},
            "z0 must be strictly increasing",
        ),
        (
            {"pointwise_lower": np.array([0.1, 0.7, 0.2])},
            "pointwise interval lower bounds",
        ),
        (
            {"estimate": np.array([0.2, 0.9, 0.4])},
            "estimate must lie inside pointwise interval",
        ),
        (
            {"uniform_lower": np.array([0.0, 0.4, 0.1])},
            "pointwise interval must lie inside uniform band",
        ),
        (
            {"pointwise_level": "0.9"},
            "pointwise_level must be numeric, not string",
        ),
        (
            {"pointwise_level": np.array("0.9")},
            "pointwise_level must be numeric, not string",
        ),
        (
            {"pointwise_level": np.array([0.9])},
            "pointwise_level must be a scalar",
        ),
        (
            {"uniform_level": np.array("0.9", dtype=object)},
            "uniform_level must be numeric, not string",
        ),
        (
            {"uniform_level": False},
            "uniform_level must be numeric, not boolean",
        ),
        (
            {"uniform_level": np.array(False, dtype=object)},
            "uniform_level must be numeric, not boolean",
        ),
        (
            {"estimate_name": True},
            "estimate_name must be a string plot field name",
        ),
        (
            {"pointwise_interval_name": b"ci"},
            "pointwise_interval_name must be a string plot field name",
        ),
        (
            {"uniform_band_name": "   "},
            "uniform_band_name must be a non-empty plot field name",
        ),
        (
            {"estimate_name": "effect\nname"},
            "estimate_name must not contain line breaks",
        ),
    )

    base_kwargs = {
        "z0": np.array([0.25, 0.75, 1.25]),
        "estimate": np.array([0.2, 0.5, 0.4]),
        "pointwise_lower": np.array([0.1, 0.35, 0.2]),
        "pointwise_upper": np.array([0.3, 0.65, 0.6]),
        "uniform_lower": np.array([0.0, 0.2, 0.1]),
        "uniform_upper": np.array([0.4, 0.8, 0.7]),
    }
    for overrides, message in invalid_cases:
        kwargs = dict(base_kwargs)
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=message):
            NonparametricEffectPlotData(**kwargs)


def test_nonparametric_effect_svg_rejects_invalid_accessible_id_prefix() -> None:
    plot_data = build_nonparametric_effect_plot_data(
        _nonparametric_result(),
        z0=np.array([0.25, 0.75, 1.25]),
    )

    for invalid_prefix in ("", "1plot", "plot id", "plot:id", b"plot", True):
        with pytest.raises(ValueError, match="id_prefix"):
            plot_data.to_svg(id_prefix=invalid_prefix)  # type: ignore[arg-type]


def test_nonparametric_effect_svg_rejects_invalid_visible_labels() -> None:
    plot_data = build_nonparametric_effect_plot_data(
        _nonparametric_result(),
        z0=np.array([0.25, 0.75, 1.25]),
    )

    invalid_cases = (
        ({"title": ""}, "title must be a non-empty SVG label"),
        ({"title": "   "}, "title must be a non-empty SVG label"),
        ({"title": True}, "title must be a string SVG label"),
        ({"x_label": b"z0"}, "x_label must be a string SVG label"),
        ({"y_label": "Effect\nestimate"}, "y_label must not contain line breaks"),
        ({"x_label": "Evaluation\tpoint"}, "x_label must not contain line breaks"),
    )
    for kwargs, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            plot_data.to_svg(**kwargs)  # type: ignore[arg-type]


def test_build_nonparametric_effect_plot_data_rejects_missing_result_fields() -> None:
    result = _nonparametric_result()

    with pytest.raises(ValueError, match="missing nonparametric estimate"):
        build_nonparametric_effect_plot_data(
            result,
            z0=np.array([0.25, 0.75, 1.25]),
            estimate_name="missing",
        )

    for kwargs, message in (
        (
            {"estimate_name": True},
            "estimate_name must be a string plot field name",
        ),
        (
            {"pointwise_interval_name": b"nonparametric_ci"},
            "pointwise_interval_name must be a string plot field name",
        ),
        (
            {"uniform_band_name": ""},
            "uniform_band_name must be a non-empty plot field name",
        ),
        (
            {"estimate_name": "bar_f_at_z0\talias"},
            "estimate_name must not contain line breaks",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            build_nonparametric_effect_plot_data(
                result,
                z0=np.array([0.25, 0.75, 1.25]),
                **kwargs,  # type: ignore[arg-type]
            )

    result_without_uniform = HDDIDResult(
        nonparametric_estimates=result.nonparametric_estimates,
        intervals={"nonparametric_ci": result.intervals["nonparametric_ci"]},
    )
    with pytest.raises(ValueError, match="missing interval 'uniform_band'"):
        build_nonparametric_effect_plot_data(
            result_without_uniform,
            z0=np.array([0.25, 0.75, 1.25]),
        )
