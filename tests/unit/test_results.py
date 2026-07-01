import numpy as np
import pytest

from hddid import ConfidenceInterval, FoldDiagnostics, HDDIDResult, ResultDiagnostics


class _StringArrayLike:
    def __array__(self, dtype: object = None) -> np.ndarray:
        values = np.array(["0.1", "0.2"], dtype=object)
        if dtype is None:
            return values
        return values.astype(dtype)


def test_public_namespace_exports_typed_result_objects() -> None:
    assert HDDIDResult.__name__ == "HDDIDResult"
    assert ResultDiagnostics.__name__ == "ResultDiagnostics"
    assert FoldDiagnostics.__name__ == "FoldDiagnostics"
    assert ConfidenceInterval.__name__ == "ConfidenceInterval"


def test_hddid_result_keeps_schema_groups_and_fold_validity_diagnostics() -> None:
    fold = FoldDiagnostics(
        fold_id=1,
        n_holdout_raw=12,
        n_trimmed_propensity=2,
        n_valid_holdout=10,
        trim_lower=0.01,
        trim_upper=0.99,
    )
    diagnostics = ResultDiagnostics(
        basis_family="polynomial",
        basis_degree=3,
        oracle_lane="r-parity",
        fold_diagnostics=[fold],
    )
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0, -0.5])},
        nonparametric_estimates={"f_hat_at_z0": np.array([0.25])},
        standard_errors={"beta_se": np.array([0.1, 0.2])},
        intervals={"parametric_ci": ConfidenceInterval(lower=-0.1, upper=0.1)},
        diagnostics=diagnostics,
    )

    assert set(result.parametric_estimates) == {"beta_hat"}
    assert set(result.nonparametric_estimates) == {"f_hat_at_z0"}
    assert set(result.standard_errors) == {"beta_se"}
    assert set(result.intervals) == {"parametric_ci"}
    assert result.diagnostics.basis_family == "polynomial"
    assert result.diagnostics.basis_degree == 3
    assert result.diagnostics.oracle_lane == "r-parity-polynomial"
    assert result.diagnostics.fold_diagnostics[0].n_valid_holdout == 10


def test_hddid_result_to_markdown_renders_publishable_summary_table() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.23456, -0.5])},
        nonparametric_estimates={"f_hat_at_z0": np.array([0.25])},
        standard_errors={"beta_se": np.array([0.12345, 0.2])},
        intervals={
            "parametric_ci": ConfidenceInterval(
                lower=np.array([0.99, -0.9]),
                upper=np.array([1.48, -0.1]),
                level=0.95,
            )
        },
        diagnostics=ResultDiagnostics(
            basis_family="polynomial",
            basis_degree=3,
            oracle_lane="r-parity",
            n_holdout_raw=12,
            n_trimmed_propensity=2,
            n_valid_holdout=10,
            trim_lower=0.01,
            trim_upper=0.99,
        ),
    )

    markdown = result.to_markdown(style="legacy", missing_value="")

    assert "| Section | Name | Index | Estimate | Std. Error | Interval |" in markdown
    assert "| Parametric | beta_hat | 0 | 1.2346 | 0.1235 | [0.9900, 1.4800] (95.0%) |" in markdown
    assert "| Parametric | beta_hat | 1 | -0.5000 | 0.2000 | [-0.9000, -0.1000] (95.0%) |" in markdown
    assert "| Nonparametric | f_hat_at_z0 | 0 | 0.2500 |  |  |" in markdown
    assert "basis=polynomial(3)" in markdown
    assert "oracle_lane=r-parity-polynomial" in markdown
    assert "holdout=12, trimmed=2, valid=10" in markdown
    assert "trim=[0.0100, 0.9900]" in markdown


def test_hddid_result_to_markdown_can_render_compact_numbers() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([123456.0, 1.2e-8, 1.25])},
        standard_errors={"beta_se": np.array([123.0, 2.5e-9, 0.25])},
        intervals={
            "parametric_ci": ConfidenceInterval(
                lower=np.array([120000.0, -3.1e-8, 0.75]),
                upper=np.array([130000.0, 4.2e-8, 1.75]),
                level=0.95,
            )
        },
        diagnostics=ResultDiagnostics(
            basis_family="polynomial",
            basis_degree=2,
            oracle_lane="r-parity",
            n_holdout_raw=6,
            n_trimmed_propensity=0,
            n_valid_holdout=6,
            trim_lower=0.01,
            trim_upper=0.99,
        ),
    )

    markdown = result.to_markdown(number_format="compact", style="legacy", missing_value="")

    assert "| Parametric | beta_hat | 0 | 1.2346e+05 | 123 | [1.2000e+05, 1.3000e+05] (95.0%) |" in markdown
    assert "| Parametric | beta_hat | 1 | 1.2000e-08 | 2.5000e-09 | [-3.1000e-08, 4.2000e-08] (95.0%) |" in markdown
    assert "| Parametric | beta_hat | 2 | 1.25 | 0.25 | [0.75, 1.75] (95.0%) |" in markdown
    assert "trim=[0.01, 0.99]" in markdown
    assert "123456.0000" not in markdown
    assert "0.0000" not in markdown


def test_hddid_result_to_markdown_renders_fold_level_diagnostics() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0])},
        diagnostics=ResultDiagnostics(
            basis_family="polynomial",
            basis_degree=2,
            oracle_lane="r-parity",
            fold_diagnostics=[
                FoldDiagnostics(
                    fold_id=1,
                    n_holdout_raw=5,
                    n_trimmed_propensity=1,
                    n_valid_holdout=4,
                    trim_lower=0.01,
                    trim_upper=0.99,
                ),
                FoldDiagnostics(
                    fold_id=2,
                    n_holdout_raw=7,
                    n_trimmed_propensity=2,
                    n_valid_holdout=5,
                    trim_lower=0.01,
                    trim_upper=0.99,
                ),
            ],
            n_holdout_raw=12,
            n_trimmed_propensity=3,
            n_valid_holdout=9,
            trim_lower=0.01,
            trim_upper=0.99,
        ),
    )

    markdown = result.to_markdown(digits=3, style="legacy", missing_value="")

    assert "Diagnostics: basis=polynomial(2); oracle_lane=r-parity-polynomial" in markdown
    assert "folds=2" in markdown
    assert (
        "Fold diagnostics: fold 1; holdout=5, trimmed=1, valid=4; "
        "trim=[0.010, 0.990]\nfold 2; holdout=7, trimmed=2, valid=5; "
        "trim=[0.010, 0.990]"
    ) in markdown


def test_hddid_result_to_markdown_can_render_domain_row_labels() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.23456, -0.5])},
        nonparametric_estimates={"f_hat_at_z0": np.array([0.25, 0.5])},
        standard_errors={"beta_se": np.array([0.12345, 0.2])},
        intervals={
            "parametric_ci": ConfidenceInterval(
                lower=np.array([0.99, -0.9]),
                upper=np.array([1.48, -0.1]),
                level=0.95,
            )
        },
    )

    markdown = result.to_markdown(
        row_labels={
            "beta_hat": ["income | baseline", "population delta"],
            "f_hat_at_z0": ["z0=0.25", "z0=0.75"],
        },
        style="legacy",
        missing_value="",
    )

    assert "| Section | Name | Index | Label | Estimate | Std. Error | Interval |" in markdown
    assert (
        "| Parametric | beta_hat | 0 | income \\| baseline | "
        "1.2346 | 0.1235 | [0.9900, 1.4800] (95.0%) |"
    ) in markdown
    assert "| Nonparametric | f_hat_at_z0 | 1 | z0=0.75 | 0.5000 |  |  |" in markdown


def test_hddid_result_to_markdown_can_render_explicit_missing_cells() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0])},
        nonparametric_estimates={"f_hat_at_z0": np.array([0.25])},
        standard_errors={"beta_se": np.array([0.1])},
    )

    markdown = result.to_markdown(
        row_labels={
            "beta_hat": ["x0"],
            "f_hat_at_z0": ["z0=0.25"],
        },
        missing_value="-",
        style="legacy",
    )

    assert "| Parametric | beta_hat | 0 | x0 | 1.0000 | 0.1000 | - |" in markdown
    assert "| Nonparametric | f_hat_at_z0 | 0 | z0=0.25 | 0.2500 | - | - |" in markdown


def test_hddid_result_to_summary_exports_machine_auditable_table_contract() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.23456, -0.5])},
        nonparametric_estimates={"f_hat_at_z0": np.array([0.25])},
        standard_errors={"beta_se": np.array([0.12345, 0.2])},
        intervals={
            "parametric_ci": ConfidenceInterval(
                lower=np.array([0.99, -0.9]),
                upper=np.array([1.48, -0.1]),
                level=0.95,
            )
        },
        diagnostics=ResultDiagnostics(
            basis_family="polynomial",
            basis_degree=3,
            oracle_lane="r-parity",
            n_holdout_raw=12,
            n_trimmed_propensity=2,
            n_valid_holdout=10,
            trim_lower=0.01,
            trim_upper=0.99,
        ),
    )

    summary = result.to_summary(
        row_labels={
            "beta_hat": ["income", "population"],
            "f_hat_at_z0": ["z0=0.25"],
        },
    )

    assert summary["result_contract"] == "hddid-result-summary"
    assert summary["columns"] == [
        "Section",
        "Name",
        "Index",
        "Label",
        "Estimate",
        "Std. Error",
        "Interval",
    ]
    assert summary["row_count"] == 3
    assert summary["has_row_labels"] is True
    assert summary["missing_standard_error_cells"] == 1
    assert summary["missing_interval_cells"] == 1
    assert summary["sections"] == {
        "Parametric": {"estimate_names": ["beta_hat"], "row_count": 2},
        "Nonparametric": {"estimate_names": ["f_hat_at_z0"], "row_count": 1},
    }
    assert summary["estimate_order"] == [
        {
            "section": "Parametric",
            "name": "beta_hat",
            "length": 2,
            "standard_error_key": "beta_se",
            "interval_key": "parametric_ci",
        },
        {
            "section": "Nonparametric",
            "name": "f_hat_at_z0",
            "length": 1,
            "standard_error_key": None,
            "interval_key": None,
        },
    ]
    assert summary["rows"] == [
        {
            "section": "Parametric",
            "name": "beta_hat",
            "index": 0,
            "estimate": "1.2346",
            "standard_error": "0.1235",
            "interval": "[0.9900, 1.4800] (95.0%)",
            "label": "income",
        },
        {
            "section": "Parametric",
            "name": "beta_hat",
            "index": 1,
            "estimate": "-0.5000",
            "standard_error": "0.2000",
            "interval": "[-0.9000, -0.1000] (95.0%)",
            "label": "population",
        },
        {
            "section": "Nonparametric",
            "name": "f_hat_at_z0",
            "index": 0,
            "estimate": "0.2500",
            "standard_error": None,
            "interval": None,
            "label": "z0=0.25",
        },
    ]
    assert summary["diagnostics"] == {
        "basis_family": "polynomial",
        "basis_degree": 3,
        "oracle_lane": "r-parity-polynomial",
        "n_holdout_raw": 12,
        "n_trimmed_propensity": 2,
        "n_valid_holdout": 10,
        "trim_lower": "0.0100",
        "trim_upper": "0.9900",
        "fold_count": 0,
    }


def test_hddid_result_to_summary_uses_markdown_render_validation() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0, -0.5])},
        standard_errors={"beta_se": np.array([0.1])},
    )

    with pytest.raises(
        ValueError,
        match="standard_errors.beta_se must contain 2 value\\(s\\) to render beta_hat",
    ):
        result.to_summary()

    with pytest.raises(ValueError, match="row_labels.beta_hat must contain 2 labels"):
        HDDIDResult(parametric_estimates={"beta_hat": np.array([1.0, -0.5])}).to_summary(
            row_labels={"beta_hat": ["x0"]}
        )


def test_hddid_result_to_markdown_escapes_explicit_missing_cells() -> None:
    result = HDDIDResult(parametric_estimates={"beta_hat": np.array([1.0])})

    markdown = result.to_markdown(missing_value="<missing | unavailable>")

    assert "&lt;missing \\| unavailable&gt;" in markdown
    assert "<missing | unavailable>" not in markdown

    with pytest.raises(ValueError, match="missing_value must be a string"):
        result.to_markdown(missing_value=0)  # type: ignore[arg-type]


def test_hddid_result_to_markdown_keeps_multiline_labels_inside_rows() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta\nhat": np.array([1.0])},
        standard_errors={"beta\nhat": np.array([0.1])},
        intervals={
            "beta\nhat": ConfidenceInterval(
                lower=np.array([0.8]),
                upper=np.array([1.2]),
                level=0.95,
            )
        },
    )

    markdown = result.to_markdown(
        row_labels={"beta\nhat": ["income\r\nbaseline\tcounty | state"]},
        style="legacy",
        missing_value="",
    )

    assert "beta<br>hat" in markdown
    assert "income<br>baseline county \\| state" in markdown
    assert "\r" not in markdown
    assert "\t" not in markdown
    data_rows = [
        line
        for line in markdown.splitlines()
        if line.startswith("| Parametric |")
    ]
    assert len(data_rows) == 1
    assert data_rows[0].endswith(" |")


def test_hddid_result_to_markdown_escapes_html_without_breaking_line_breaks() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0])},
    )

    markdown = result.to_markdown(
        row_labels={
            "beta_hat": [
                "income <script>alert(1)</script>\ncounty & state <br> literal"
            ]
        },
        style="legacy",
        missing_value="",
    )

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in markdown
    assert "county &amp; state" in markdown
    assert "&lt;br&gt; literal" in markdown
    assert "income <script>" not in markdown
    assert "state <br> literal" not in markdown
    assert "<br>county" in markdown


def test_hddid_result_to_markdown_rejects_invalid_row_label_contracts() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0, -0.5])},
        nonparametric_estimates={"f_hat_at_z0": np.array([0.25])},
    )

    invalid_cases = (
        (["not", "a", "mapping"], "row_labels must be a mapping"),
        ({1: ["not a string key"]}, "row_labels estimate names must be strings"),
        ({"unknown_hat": ["unknown"]}, "unknown estimate name"),
        ({"beta_hat": ["one"]}, "row_labels.beta_hat must contain 2 labels"),
        ({"beta_hat": "not-a-sequence"}, "row_labels.beta_hat must be a sequence"),
        ({"beta_hat": ["valid", 2]}, "row_labels.beta_hat must contain only strings"),
    )

    for row_labels, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            result.to_markdown(row_labels=row_labels)


def test_hddid_result_rejects_non_string_result_mapping_keys() -> None:
    invalid_cases = (
        (
            {"parametric_estimates": {1: np.array([1.0])}},
            "parametric_estimates keys must be strings",
        ),
        (
            {"nonparametric_estimates": {("f",): np.array([0.25])}},
            "nonparametric_estimates keys must be strings",
        ),
        (
            {"standard_errors": {0: np.array([0.1])}},
            "standard_errors keys must be strings",
        ),
        (
            {
                "intervals": {
                    1: ConfidenceInterval(lower=np.array([0.8]), upper=np.array([1.2]))
                }
            },
            "intervals keys must be strings",
        ),
    )

    for kwargs, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            HDDIDResult(**kwargs)


def test_hddid_result_rejects_empty_result_mapping_keys() -> None:
    invalid_cases = (
        (
            {"parametric_estimates": {"": np.array([1.0])}},
            "parametric_estimates keys must be non-empty strings",
        ),
        (
            {"nonparametric_estimates": {"   ": np.array([0.25])}},
            "nonparametric_estimates keys must be non-empty strings",
        ),
        (
            {"standard_errors": {"": np.array([0.1])}},
            "standard_errors keys must be non-empty strings",
        ),
        (
            {
                "intervals": {
                    "  ": ConfidenceInterval(
                        lower=np.array([0.8]),
                        upper=np.array([1.2]),
                    )
                }
            },
            "intervals keys must be non-empty strings",
        ),
    )

    for kwargs, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            HDDIDResult(**kwargs)


def test_hddid_result_rejects_result_mapping_keys_with_outer_whitespace() -> None:
    invalid_cases = (
        (
            {"parametric_estimates": {" beta_hat": np.array([1.0])}},
            "parametric_estimates keys must not have leading or trailing whitespace",
        ),
        (
            {"nonparametric_estimates": {"f_hat_at_z0 ": np.array([0.25])}},
            "nonparametric_estimates keys must not have leading or trailing whitespace",
        ),
        (
            {"standard_errors": {" beta_se ": np.array([0.1])}},
            "standard_errors keys must not have leading or trailing whitespace",
        ),
        (
            {
                "intervals": {
                    "\tparametric_ci": ConfidenceInterval(
                        lower=np.array([0.8]),
                        upper=np.array([1.2]),
                    )
                }
            },
            "intervals keys must not have leading or trailing whitespace",
        ),
    )

    for kwargs, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            HDDIDResult(**kwargs)


def test_hddid_result_to_markdown_rejects_truncated_matched_inference_arrays() -> None:
    with pytest.raises(
        ValueError,
        match="standard_errors.beta_se must contain 2 value\\(s\\) to render beta_hat",
    ):
        HDDIDResult(
            parametric_estimates={"beta_hat": np.array([1.0, -0.5])},
            standard_errors={"beta_se": np.array([0.1])},
        ).to_markdown()

    with pytest.raises(
        ValueError,
        match="intervals.parametric_ci must contain 2 value\\(s\\) to render beta_hat",
    ):
        HDDIDResult(
            parametric_estimates={"beta_hat": np.array([1.0, -0.5])},
            intervals={
                "parametric_ci": ConfidenceInterval(
                    lower=np.array([0.8]),
                    upper=np.array([1.2]),
                    level=0.95,
                )
            },
        ).to_markdown()


def test_hddid_result_to_markdown_does_not_reuse_section_interval_for_unmatched_estimates() -> None:
    result = HDDIDResult(
        parametric_estimates={
            "beta_hat": np.array([1.0, -0.5]),
            "t_hat": np.array([0.25, 0.5]),
        },
        standard_errors={"beta_se": np.array([0.1, 0.2])},
        intervals={
            "parametric_ci": ConfidenceInterval(
                lower=np.array([0.8, -0.9]),
                upper=np.array([1.2, -0.1]),
                level=0.95,
            )
        },
    )

    markdown = result.to_markdown(style="legacy", missing_value="")

    assert "| Parametric | beta_hat | 0 | 1.0000 |  |  |" in markdown
    assert "| Parametric | beta_hat | 1 | -0.5000 |  |  |" in markdown
    assert "| Parametric | t_hat | 0 | 0.2500 | 0.1000 | [0.8000, 1.2000] (95.0%) |" in markdown
    assert "| Parametric | t_hat | 1 | 0.5000 | 0.2000 | [-0.9000, -0.1000] (95.0%) |" in markdown


def test_hddid_result_to_markdown_matches_exact_target_interval_names_before_aliases() -> None:
    result = HDDIDResult(
        parametric_estimates={
            "beta_hat": np.array([1.0, -0.5]),
            "t_hat": np.array([0.25]),
        },
        intervals={
            "beta_ci": ConfidenceInterval(
                lower=np.array([0.8, -0.9]),
                upper=np.array([1.2, -0.1]),
                level=0.95,
            ),
            "t_ci": ConfidenceInterval(
                lower=np.array([0.1]),
                upper=np.array([0.4]),
                level=0.9,
            ),
        },
    )

    markdown = result.to_markdown(style="legacy", missing_value="")

    assert "| Parametric | beta_hat | 0 | 1.0000 |  | [0.8000, 1.2000] (95.0%) |" in markdown
    assert "| Parametric | beta_hat | 1 | -0.5000 |  | [-0.9000, -0.1000] (95.0%) |" in markdown
    assert "| Parametric | t_hat | 0 | 0.2500 |  | [0.1000, 0.4000] (90.0%) |" in markdown


def test_hddid_result_to_markdown_binds_nonparametric_inference_to_debiased_effect() -> None:
    result = HDDIDResult(
        nonparametric_estimates={
            "f_hat_at_z0": np.array([0.2, 0.4]),
            "bar_f_at_z0": np.array([0.25, 0.5]),
        },
        standard_errors={"f_se": np.array([0.05, 0.1])},
        intervals={
            "nonparametric_ci": ConfidenceInterval(
                lower=np.array([0.15, 0.3]),
                upper=np.array([0.35, 0.7]),
                level=0.9,
            )
        },
    )

    markdown = result.to_markdown(style="legacy", missing_value="")

    assert "| Nonparametric | f_hat_at_z0 | 0 | 0.2000 |  |  |" in markdown
    assert "| Nonparametric | f_hat_at_z0 | 1 | 0.4000 |  |  |" in markdown
    assert "| Nonparametric | bar_f_at_z0 | 0 | 0.2500 | 0.0500 | [0.1500, 0.3500] (90.0%) |" in markdown
    assert "| Nonparametric | bar_f_at_z0 | 1 | 0.5000 | 0.1000 | [0.3000, 0.7000] (90.0%) |" in markdown


def test_hddid_result_to_markdown_does_not_round_high_confidence_to_100_percent() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0])},
        intervals={
            "beta_ci": ConfidenceInterval(
                lower=np.array([0.8]),
                upper=np.array([1.2]),
                level=0.9999,
            )
        },
    )

    markdown = result.to_markdown(style="legacy", missing_value="")

    assert "[0.8000, 1.2000] (99.9%)" in markdown
    assert "(100.0%)" not in markdown


def test_hddid_result_to_markdown_uses_stable_semantic_estimate_order() -> None:
    result = HDDIDResult(
        parametric_estimates={
            "zeta_hat": np.array([9.0]),
            "t_hat": np.array([0.25]),
            "alpha_hat": np.array([1.5]),
            "beta_hat": np.array([1.0]),
        },
        nonparametric_estimates={
            "z_grid_aux": np.array([3.0]),
            "bar_f_at_z0": np.array([0.5]),
            "f_hat_at_z0": np.array([0.4]),
            "bar_gamma_hat": np.array([0.3]),
            "gamma_hat": np.array([0.2]),
            "a_grid_aux": np.array([2.0]),
        },
    )

    data_rows = [
        line
        for line in result.to_markdown(style="legacy", missing_value="").splitlines()
        if line.startswith("| Parametric |") or line.startswith("| Nonparametric |")
    ]

    rendered_names = [row.split(" | ")[1] for row in data_rows]
    assert rendered_names == [
        "beta_hat",
        "t_hat",
        "alpha_hat",
        "zeta_hat",
        "gamma_hat",
        "bar_gamma_hat",
        "f_hat_at_z0",
        "bar_f_at_z0",
        "a_grid_aux",
        "z_grid_aux",
    ]


def test_hddid_result_to_markdown_does_not_cross_match_parametric_and_nonparametric_tokens() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0])},
        nonparametric_estimates={"bar_f_at_z0": np.array([0.25])},
        standard_errors={"f_se": np.array([0.05])},
        intervals={
            "nonparametric_ci": ConfidenceInterval(
                lower=np.array([0.15]),
                upper=np.array([0.35]),
                level=0.9,
            )
        },
    )

    markdown = result.to_markdown(style="legacy", missing_value="")

    assert "| Parametric | beta_hat | 0 | 1.0000 |  |  |" in markdown
    assert "| Nonparametric | bar_f_at_z0 | 0 | 0.2500 | 0.0500 | [0.1500, 0.3500] (90.0%) |" in markdown


def test_hddid_result_to_markdown_keeps_section_interval_shortcut_for_single_estimate() -> None:
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0, -0.5])},
        intervals={
            "parametric_ci": ConfidenceInterval(
                lower=np.array([0.8, -0.9]),
                upper=np.array([1.2, -0.1]),
                level=0.95,
            )
        },
    )

    markdown = result.to_markdown(style="legacy", missing_value="")

    assert "| Parametric | beta_hat | 0 | 1.0000 |  | [0.8000, 1.2000] (95.0%) |" in markdown
    assert "| Parametric | beta_hat | 1 | -0.5000 |  | [-0.9000, -0.1000] (95.0%) |" in markdown


def test_result_diagnostics_normalizes_paper_lane_to_canonical_label() -> None:
    diagnostics = ResultDiagnostics(
        basis_family="trigonometric",
        basis_degree=4,
        oracle_lane="paper",
        n_holdout_raw=18,
        n_trimmed_propensity=3,
        n_valid_holdout=15,
        trim_lower=0.01,
        trim_upper=0.99,
    )

    assert diagnostics.basis_family == "trigonometric"
    assert diagnostics.basis_degree == 4
    assert diagnostics.oracle_lane == "paper-trigonometric"
    assert diagnostics.n_holdout_raw == 18
    assert diagnostics.n_trimmed_propensity == 3
    assert diagnostics.n_valid_holdout == 15
    assert diagnostics.trim_lower == 0.01
    assert diagnostics.trim_upper == 0.99


def test_polynomial_degree_zero_is_valid_in_result_diagnostics() -> None:
    fold = FoldDiagnostics(
        fold_id=1,
        basis_family="polynomial",
        basis_degree=0,
        oracle_lane="r-parity",
        n_holdout_raw=4,
        n_trimmed_propensity=1,
        n_valid_holdout=3,
    )
    diagnostics = ResultDiagnostics(
        basis_family="polynomial",
        basis_degree=0,
        oracle_lane="r-parity",
        fold_diagnostics=[fold],
        n_holdout_raw=4,
        n_trimmed_propensity=1,
        n_valid_holdout=3,
    )

    assert fold.basis_degree == 0
    assert diagnostics.basis_degree == 0
    assert diagnostics.oracle_lane == "r-parity-polynomial"


def test_trigonometric_degree_zero_remains_invalid_in_result_diagnostics() -> None:
    with pytest.raises(ValueError, match="positive for trigonometric"):
        ResultDiagnostics(
            basis_family="trigonometric",
            basis_degree=0,
            oracle_lane="paper",
        )

    with pytest.raises(ValueError, match="positive for trigonometric"):
        FoldDiagnostics(
            fold_id=1,
            basis_family="trigonometric",
            basis_degree=0,
            oracle_lane="paper",
        )


def test_result_diagnostics_rejects_lane_basis_mismatch() -> None:
    with pytest.raises(ValueError, match="oracle_lane"):
        ResultDiagnostics(
            basis_family="trigonometric",
            basis_degree=4,
            oracle_lane="r-parity",
        )


def test_result_diagnostics_rejects_invalid_design_provenance_labels() -> None:
    result_cases = (
        ({"basis_family": None}, "basis_family must be a string"),
        ({"basis_family": True}, "basis_family must be a string"),
        ({"basis_family": b"polynomial"}, "basis_family must be a string"),
        ({"basis_family": "   "}, "basis_family must be non-empty"),
        ({"oracle_lane": None}, "oracle_lane must be a string"),
        ({"oracle_lane": False}, "oracle_lane must be a string"),
        ({"oracle_lane": b"r-parity"}, "oracle_lane must be a string"),
        ({"oracle_lane": ""}, "oracle_lane must be non-empty"),
    )
    for overrides, message in result_cases:
        kwargs = {
            "basis_family": "polynomial",
            "basis_degree": 2,
            "oracle_lane": "r-parity",
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=message):
            ResultDiagnostics(**kwargs)

    fold_cases = (
        ({"basis_family": True}, "basis_family must be a string"),
        ({"basis_family": b"polynomial"}, "basis_family must be a string"),
        ({"basis_family": "   "}, "basis_family must be non-empty"),
        ({"oracle_lane": False}, "oracle_lane must be a string"),
        ({"oracle_lane": b"r-parity"}, "oracle_lane must be a string"),
        ({"oracle_lane": ""}, "oracle_lane must be non-empty"),
    )
    for overrides, message in fold_cases:
        kwargs = {"fold_id": 1}
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=message):
            FoldDiagnostics(**kwargs)


def test_confidence_interval_rejects_invalid_numeric_contracts() -> None:
    invalid_cases = (
        ({"lower": False, "upper": 0.1}, "lower must be numeric, not boolean"),
        ({"lower": 0.0, "upper": True}, "upper must be numeric, not boolean"),
        ({"lower": "0.0", "upper": 0.1}, "lower must be numeric, not string"),
        (
            {
                "lower": np.array([0.0], dtype=object),
                "upper": np.array(["0.1"], dtype=object),
            },
            "upper must be numeric, not string",
        ),
        (
            {
                "lower": np.array([0.0], dtype=object),
                "upper": np.array([b"0.1"], dtype=object),
            },
            "upper must be numeric, not string",
        ),
        (
            {"lower": _StringArrayLike(), "upper": np.array([0.3, 0.4])},
            "lower must be numeric, not string",
        ),
        (
            {"lower": np.array([0.0, 0.1]), "upper": _StringArrayLike()},
            "upper must be numeric, not string",
        ),
        ({"lower": np.nan, "upper": 0.1}, "bounds must be finite"),
        ({"lower": 0.2, "upper": np.inf}, "bounds must be finite"),
        ({"lower": np.array([0.0, 0.5]), "upper": np.array([0.1, 0.4])}, "lower"),
        ({"lower": 0.0, "upper": 0.1, "level": "0.95"}, "level must be numeric, not string"),
        ({"lower": 0.0, "upper": 0.1, "level": b"0.95"}, "level must be numeric, not string"),
        (
            {"lower": 0.0, "upper": 0.1, "level": np.array("0.95")},
            "level must be numeric, not string",
        ),
        (
            {"lower": 0.0, "upper": 0.1, "level": np.array([0.95])},
            "level must be a scalar",
        ),
        ({"lower": 0.0, "upper": 0.1, "level": np.nan}, "level must be finite"),
        ({"lower": 0.0, "upper": 0.1, "level": 1.0}, "level must be in"),
    )

    for kwargs, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            ConfidenceInterval(**kwargs)


def test_uniform_band_rejects_invalid_simulation_metadata() -> None:
    invalid_cases = (
        ({"critical_value": True}, "critical_value must be numeric, not boolean"),
        ({"critical_value": "1.96"}, "critical_value must be numeric, not string"),
        (
            {"critical_value": np.array("1.96")},
            "critical_value must be numeric, not string",
        ),
        (
            {"critical_value": np.array([1.96])},
            "critical_value must be a scalar",
        ),
        ({"critical_value": np.nan}, "critical_value must be finite"),
        ({"critical_value": 0.0}, "critical_value must be positive"),
        ({"n_boot": 0}, "n_boot must be positive"),
        ({"n_boot": 1.5}, "n_boot must be an integer"),
        ({"random_state": True}, "random_state must be an integer"),
        ({"random_state": -1}, "random_state must be non-negative"),
    )

    for kwargs, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            from hddid import UniformBand

            UniformBand(lower=np.array([-0.2]), upper=np.array([0.2]), **kwargs)


def test_fold_and_result_diagnostics_reject_invalid_counts_and_trim_bounds() -> None:
    fold_cases = (
        ({"fold_id": 0}, "fold_id must be positive"),
        ({"fold_id": 1, "basis_degree": -1}, "basis_degree must be positive"),
        ({"fold_id": 1, "n_holdout_raw": -1}, "n_holdout_raw must be non-negative"),
        ({"fold_id": 1, "trim_lower": False, "trim_upper": True}, "trim_lower must be numeric"),
        (
            {"fold_id": 1, "trim_lower": "0.01", "trim_upper": 0.99},
            "trim_lower must be numeric",
        ),
        (
            {"fold_id": 1, "trim_lower": np.array("0.01"), "trim_upper": 0.99},
            "trim_lower must be numeric",
        ),
        (
            {"fold_id": 1, "trim_lower": np.array([0.01]), "trim_upper": 0.99},
            "trim_lower must be a scalar",
        ),
        (
            {
                "fold_id": 1,
                "n_holdout_raw": 4,
                "n_trimmed_propensity": 2,
                "n_valid_holdout": 3,
            },
            "must equal n_holdout_raw",
        ),
        (
            {
                "fold_id": 1,
                "n_holdout_raw": 10,
                "n_trimmed_propensity": 2,
                "n_valid_holdout": 7,
            },
            "must equal n_holdout_raw",
        ),
        (
            {"fold_id": 1, "trim_lower": 0.99, "trim_upper": 0.01},
            "trim bounds must satisfy",
        ),
    )

    for kwargs, message in fold_cases:
        with pytest.raises(ValueError, match=message):
            FoldDiagnostics(**kwargs)

    with pytest.raises(ValueError, match="optimization_metadata.bad must be finite"):
        ResultDiagnostics(
            basis_family="polynomial",
            basis_degree=3,
            oracle_lane="r-parity",
            n_holdout_raw=4,
            n_trimmed_propensity=1,
            n_valid_holdout=3,
            optimization_metadata={"bad": np.nan},
        )

    with pytest.raises(ValueError, match="must equal n_holdout_raw"):
        ResultDiagnostics(
            basis_family="polynomial",
            basis_degree=3,
            oracle_lane="r-parity",
            n_holdout_raw=10,
            n_trimmed_propensity=2,
            n_valid_holdout=7,
        )

    with pytest.raises(ValueError, match="trim_lower must be numeric"):
        ResultDiagnostics(
            basis_family="polynomial",
            basis_degree=3,
            oracle_lane="r-parity",
            trim_lower=False,
            trim_upper=True,
        )

    with pytest.raises(ValueError, match="trim_upper must be numeric"):
        ResultDiagnostics(
            basis_family="polynomial",
            basis_degree=3,
            oracle_lane="r-parity",
            trim_lower=0.01,
            trim_upper="0.99",
        )


def test_result_diagnostics_rejects_aggregate_count_drift_from_fold_details() -> None:
    folds = [
        FoldDiagnostics(
            fold_id=1,
            n_holdout_raw=5,
            n_trimmed_propensity=1,
            n_valid_holdout=4,
            trim_lower=0.01,
            trim_upper=0.99,
        ),
        FoldDiagnostics(
            fold_id=2,
            n_holdout_raw=7,
            n_trimmed_propensity=2,
            n_valid_holdout=5,
            trim_lower=0.01,
            trim_upper=0.99,
        ),
    ]

    valid = ResultDiagnostics(
        basis_family="polynomial",
        basis_degree=3,
        oracle_lane="r-parity",
        fold_diagnostics=folds,
        n_holdout_raw=12,
        n_trimmed_propensity=3,
        n_valid_holdout=9,
        trim_lower=0.01,
        trim_upper=0.99,
    )

    assert valid.n_holdout_raw == 12
    assert valid.n_trimmed_propensity == 3
    assert valid.n_valid_holdout == 9

    invalid_cases = (
        (
            {"n_holdout_raw": 13, "n_valid_holdout": 10},
            "n_holdout_raw must equal the fold total",
        ),
        (
            {"n_trimmed_propensity": 2, "n_valid_holdout": 10},
            "n_trimmed_propensity must equal the fold total",
        ),
        (
            {"n_holdout_raw": None, "n_valid_holdout": 8},
            "n_valid_holdout must equal the fold total",
        ),
    )

    for overrides, message in invalid_cases:
        kwargs = {
            "basis_family": "polynomial",
            "basis_degree": 3,
            "oracle_lane": "r-parity",
            "fold_diagnostics": folds,
            "n_holdout_raw": 12,
            "n_trimmed_propensity": 3,
            "n_valid_holdout": 9,
            "trim_lower": 0.01,
            "trim_upper": 0.99,
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=message):
            ResultDiagnostics(**kwargs)


def test_result_diagnostics_rejects_fold_design_drift_from_run_contract() -> None:
    base_kwargs = {
        "basis_family": "polynomial",
        "basis_degree": 3,
        "oracle_lane": "r-parity",
        "n_holdout_raw": 12,
        "n_trimmed_propensity": 3,
        "n_valid_holdout": 9,
        "trim_lower": 0.01,
        "trim_upper": 0.99,
    }

    valid = ResultDiagnostics(
        **base_kwargs,
        fold_diagnostics=[
            FoldDiagnostics(
                fold_id=1,
                basis_family="polynomial",
                basis_degree=3,
                oracle_lane="r-parity",
                n_holdout_raw=5,
                n_trimmed_propensity=1,
                n_valid_holdout=4,
                trim_lower=0.01,
                trim_upper=0.99,
            ),
            FoldDiagnostics(
                fold_id=2,
                basis_family="polynomial",
                basis_degree=3,
                oracle_lane="r-parity-polynomial",
                n_holdout_raw=7,
                n_trimmed_propensity=2,
                n_valid_holdout=5,
                trim_lower=0.01,
                trim_upper=0.99,
            ),
        ],
    )

    assert valid.oracle_lane == "r-parity-polynomial"

    drift_cases = (
        (
            {
                "fold_diagnostics": [
                    FoldDiagnostics(
                        fold_id=1,
                        basis_family="trigonometric",
                        n_holdout_raw=12,
                        n_trimmed_propensity=3,
                        n_valid_holdout=9,
                        trim_lower=0.01,
                        trim_upper=0.99,
                    )
                ],
            },
            "fold_diagnostics basis_family",
        ),
        (
            {
                "fold_diagnostics": [
                    FoldDiagnostics(
                        fold_id=1,
                        basis_degree=4,
                        n_holdout_raw=12,
                        n_trimmed_propensity=3,
                        n_valid_holdout=9,
                        trim_lower=0.01,
                        trim_upper=0.99,
                    )
                ],
            },
            "fold_diagnostics basis_degree",
        ),
        (
            {
                "fold_diagnostics": [
                    FoldDiagnostics(
                        fold_id=1,
                        oracle_lane="paper",
                        n_holdout_raw=12,
                        n_trimmed_propensity=3,
                        n_valid_holdout=9,
                        trim_lower=0.01,
                        trim_upper=0.99,
                    )
                ],
            },
            "fold_diagnostics oracle_lane",
        ),
    )

    for overrides, message in drift_cases:
        kwargs = dict(base_kwargs)
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=message):
            ResultDiagnostics(**kwargs)


def test_hddid_result_rejects_nonfinite_public_numeric_payloads() -> None:
    invalid_cases = (
        (
            {"parametric_estimates": {"beta_hat": [True, False]}},
            "parametric_estimates.beta_hat must be numeric, not boolean",
        ),
        (
            {"parametric_estimates": {"beta_hat": ["1.0"]}},
            "parametric_estimates.beta_hat must be numeric, not string",
        ),
        (
            {"parametric_estimates": {"beta_hat": [b"1.0"]}},
            "parametric_estimates.beta_hat must be numeric, not string",
        ),
        (
            {"parametric_estimates": {"beta_hat": _StringArrayLike()}},
            "parametric_estimates.beta_hat must be numeric, not string",
        ),
        (
            {"nonparametric_estimates": {"f_hat_at_z0": [0.0, True]}},
            "nonparametric_estimates.f_hat_at_z0 must be numeric, not boolean",
        ),
        (
            {"nonparametric_estimates": {"f_hat_at_z0": np.array(["0.0"])}},
            "nonparametric_estimates.f_hat_at_z0 must be numeric, not string",
        ),
        (
            {"standard_errors": {"beta_se": [True]}},
            "standard_errors.beta_se must be numeric, not boolean",
        ),
        (
            {"standard_errors": {"beta_se": np.array(["1.0"], dtype=object)}},
            "standard_errors.beta_se must be numeric, not string",
        ),
        (
            {"standard_errors": {"beta_se": _StringArrayLike()}},
            "standard_errors.beta_se must be numeric, not string",
        ),
        (
            {"parametric_estimates": {"beta_hat": np.array([1.0, np.nan])}},
            "parametric_estimates.beta_hat must contain only finite values",
        ),
        (
            {"nonparametric_estimates": {"f_hat_at_z0": [0.0, np.inf]}},
            "nonparametric_estimates.f_hat_at_z0 must contain only finite values",
        ),
        (
            {"standard_errors": {"beta_se": [-0.1, 0.2]}},
            "standard_errors.beta_se must be non-negative",
        ),
        (
            {"intervals": {"parametric_ci": object()}},
            "intervals.parametric_ci must be a ConfidenceInterval",
        ),
    )

    for kwargs, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            HDDIDResult(**kwargs)


def test_to_markdown_pretty_uses_unicode_names() -> None:
    """Pretty mode renders Unicode math symbols for estimate names."""
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0])},
        nonparametric_estimates={"f_hat_at_z0": np.array([0.5])},
        standard_errors={},
        intervals={},
    )
    markdown = result.to_markdown()  # default style="pretty"
    assert "β̂" in markdown
    assert "f̂(z₀)" in markdown
    assert "beta_hat" not in markdown
    assert "f_hat_at_z0" not in markdown


def test_to_markdown_pretty_section_folding() -> None:
    """Pretty mode shows section name only on first row, bold-formatted."""
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0, 2.0])},
        nonparametric_estimates={},
        standard_errors={},
        intervals={},
    )
    markdown = result.to_markdown()
    lines = [l for l in markdown.split("\n") if l.startswith("|") and "β̂" in l]
    # First line has bold section name
    assert "**Parametric**" in lines[0]
    # Second line has empty section cell
    assert lines[1].startswith("|  |") or lines[1].startswith("| |")


def test_to_markdown_pretty_diagnostics_multiline() -> None:
    """Pretty mode renders diagnostics as multi-line indented block."""
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.0])},
        standard_errors={},
        intervals={},
        diagnostics=ResultDiagnostics(
            basis_family="polynomial",
            basis_degree=2,
            oracle_lane="r-parity",
            n_holdout_raw=10,
            n_trimmed_propensity=1,
            n_valid_holdout=9,
            trim_lower=0.01,
            trim_upper=0.99,
        ),
    )
    markdown = result.to_markdown()
    assert "Diagnostics" in markdown
    assert "  Basis: polynomial(2)" in markdown
    assert "  Oracle lane: r-parity" in markdown
    assert "  Sample: holdout=10, trimmed=1, valid=9" in markdown
    assert "  Trim:" in markdown
    # Should NOT have old single-line format
    assert "Diagnostics: basis=" not in markdown


def test_to_markdown_legacy_backward_compat() -> None:
    """Legacy mode produces identical output to pre-change behavior."""
    result = HDDIDResult(
        parametric_estimates={"beta_hat": np.array([1.23456])},
        nonparametric_estimates={"f_hat_at_z0": np.array([0.25])},
        standard_errors={"beta_se": np.array([0.12345])},
        intervals={
            "parametric_ci": ConfidenceInterval(
                lower=np.array([0.99]),
                upper=np.array([1.48]),
                level=0.95,
            )
        },
        diagnostics=ResultDiagnostics(
            basis_family="polynomial",
            basis_degree=3,
            oracle_lane="r-parity",
            n_holdout_raw=12,
            n_trimmed_propensity=2,
            n_valid_holdout=10,
            trim_lower=0.01,
            trim_upper=0.99,
        ),
    )
    markdown = result.to_markdown(style="legacy", missing_value="")
    # Original field names preserved
    assert "beta_hat" in markdown
    assert "f_hat_at_z0" in markdown
    # Original diagnostics format
    assert "Diagnostics: basis=polynomial(3)" in markdown
    # Section repeated on every row
    assert "| Parametric |" in markdown
    assert "| Nonparametric |" in markdown
