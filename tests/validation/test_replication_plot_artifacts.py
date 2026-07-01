from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_VALIDATION_SVG_PATH = (
    REPO_ROOT
    / "paper"
    / "replication"
    / "outputs"
    / "core_validation_nonparametric_effect.svg"
)


def _load_text(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"replication plot artifact is missing: {path}")
    return path.read_text(encoding="utf-8")


def test_core_validation_nonparametric_svg_has_valid_accessible_name_links() -> None:
    svg_text = _load_text(CORE_VALIDATION_SVG_PATH)

    assert (
        'role="img" aria-labelledby="core-validation-nonparametric-effect-title '
        'core-validation-nonparametric-effect-desc"'
    ) in svg_text
    assert (
        '<title id="core-validation-nonparametric-effect-title">'
        "Core validation nonparametric effect</title>"
    ) in svg_text
    assert (
        '<desc id="core-validation-nonparametric-effect-desc">'
        "Nonparametric effect curve"
    ) in svg_text
    assert "<title>Core validation nonparametric effect</title>" not in svg_text
    assert "<desc>Nonparametric effect curve" not in svg_text
    assert '<line class="zero"' in svg_text
    assert '<rect class="legend-box"' in svg_text
    assert svg_text.count('<g class="data-point">') == 3
    assert svg_text.count("<title>evaluation point") == 3
    assert "3 plotted evaluation points" in svg_text
    assert "evaluation point 1: z0=-0.5" in svg_text
    assert "pointwise interval" in svg_text
    assert "uniform band" in svg_text
    assert "pointwise interval includes zero" in svg_text
    assert "uniform band includes zero" in svg_text
    assert "dashed zero reference line" in svg_text
