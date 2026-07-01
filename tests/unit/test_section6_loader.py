from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


TREATED_STATES = (
    "Arizona",
    "Arkansas",
    "Colorado",
    "Maryland",
    "Michigan",
    "Missouri",
    "Montana",
    "Nevada",
    "North Carolina",
    "Ohio",
    "West Virginia",
)
CONTROL_STATES = (
    "Alabama",
    "Georgia",
    "Idaho",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Mississippi",
    "Nebraska",
    "New Mexico",
    "North Dakota",
    "Oklahoma",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Virginia",
    "Wyoming",
)


def _load_loader_module():
    try:
        return importlib.import_module("hddid.section6_loader")
    except ModuleNotFoundError as exc:
        pytest.fail(f"hddid.section6_loader is missing: {exc}")


def _load_contract_module():
    try:
        return importlib.import_module("hddid.section6_loader_object_contract")
    except ModuleNotFoundError as exc:
        pytest.fail(f"hddid.section6_loader_object_contract is missing: {exc}")


def _write_manifest(manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "time_window": "2005-2007",
                "treated_state_count": 11,
                "treated_states": list(TREATED_STATES),
                "control_states": list(CONTROL_STATES),
                "sample_rule": {
                    "baseline_universe": "2006 federal-binding states",
                    "excluded_states": ["New Hampshire", "Pennsylvania"],
                    "treatment_definition": "state minimum wage increased by Q1 2007",
                    "control_definition": (
                        "state minimum wage did not increase until the federal increase in July 2007"
                    ),
                },
                "excluded_states": ["New Hampshire", "Pennsylvania"],
                "z_lanes": ["median income", "population"],
                "linear_covariate_count": 703,
                "county_characteristics_count": 38,
                "linear_covariate_construction": (
                    "pairwise-baseline-characteristic-interactions"
                ),
                "linear_covariate_count_formula": "choose(38, 2)",
                "linear_covariate_source": (
                    "reported 703-covariate block; arithmetic matches pairwise interactions among 38 county-level characteristics"
                ),
                "sample_roster_source": "Callaway and Li (2019) application sample",
                "minimum_wage_timing_cross_check_source": (
                    "U.S. Department of Labor state minimum wage history"
                ),
                "outcome_source": (
                    "BLS Local Area Unemployment Statistics (February unemployment rate)"
                ),
                "covariate_source_archive": "County and City Data Book: 2000",
                "lane_role_map": {
                    "median income": "population",
                    "population": "median income",
                },
                "basis_family": "trigonometric",
                "basis_degree": 4,
                "confidence_level": 0.95,
                "disallowed_shortcuts": ["synthetic fallback", "figure-only"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_panel_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,year,unemployment_rate,median income,population,"
            "poverty_rate,urban_share\n"
            "04013,Arizona,2005,4.3,55231,4082047,0.127,0.889\n"
            "04013,Arizona,2007,3.8,57102,4218287,0.121,0.894\n"
        ),
        encoding="utf-8",
    )


def _write_treated_and_control_panel_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,year,unemployment_rate,median income,population,"
            "poverty_rate,urban_share\n"
            "04013,Arizona,2005,4.3,55231,4082047,0.127,0.889\n"
            "04013,Arizona,2007,3.8,57102,4218287,0.121,0.894\n"
            "01001,Alabama,2005,5.2,42100,48600,0.151,0.423\n"
            "01001,Alabama,2007,4.9,43800,49300,0.146,0.431\n"
        ),
        encoding="utf-8",
    )


def _write_panel_csv_for_state(path: Path, state: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,year,unemployment_rate,median income,population,"
            "poverty_rate,urban_share\n"
            f"99999,{state},2005,4.3,55231,4082047,0.127,0.889\n"
            f"99999,{state},2007,3.8,57102,4218287,0.121,0.894\n"
        ),
        encoding="utf-8",
    )


def _write_panel_csv_rows(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        f"{county_id},{state},{year},4.3,55231,4082047,0.127,0.889"
        for county_id, state, year in rows
    )
    path.write_text(
        (
            "county_id,state,year,unemployment_rate,median income,population,"
            "poverty_rate,urban_share\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )


def _write_panel_csv_missing_core_field(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,year,unemployment_rate,population,"
            "poverty_rate,urban_share\n"
            "04013,Arizona,2005,4.3,4082047,0.127,0.889\n"
            "04013,Arizona,2007,3.8,4218287,0.121,0.894\n"
        ),
        encoding="utf-8",
    )


def _write_difference_csv_missing_core_field(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,difference_window,median income,population,"
            "poverty_rate,urban_share\n"
            "04013,Arizona,2005-2007,56166.5,4150167,0.124,0.892\n"
        ),
        encoding="utf-8",
    )


def _write_panel_csv_duplicate_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,year,unemployment_rate,median income,population,"
            "urban_share,urban_share\n"
            "04013,Arizona,2005,4.3,55231,4082047,0.127,0.889\n"
            "04013,Arizona,2007,3.8,57102,4218287,0.121,0.894\n"
        ),
        encoding="utf-8",
    )


def _write_panel_csv_extra_row_field(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,year,unemployment_rate,median income,population,"
            "poverty_rate,urban_share\n"
            "04013,Arizona,2005,4.3,55231,4082047,0.127,0.889,unexpected\n"
        ),
        encoding="utf-8",
    )


def _write_panel_csv_missing_row_field(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,year,unemployment_rate,median income,population,"
            "poverty_rate,urban_share\n"
            "04013,Arizona,2005,4.3,55231,4082047,0.127\n"
        ),
        encoding="utf-8",
    )


def _write_panel_csv_missing_characteristic_value(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,year,unemployment_rate,median income,population,"
            "poverty_rate,urban_share\n"
            "04013,Arizona,2005,4.3,55231,4082047,0.127,\n"
            "04013,Arizona,2007,3.8,57102,4218287,0.121,0.894\n"
        ),
        encoding="utf-8",
    )


def _write_panel_csv_with_missing_followup_characteristic(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,year,unemployment_rate,median income,population,"
            "poverty_rate,urban_share\n"
            "04013,Arizona,2005,4.3,55231,4082047,0.127,0.889\n"
            "04013,Arizona,2007,3.8,57102,4218287,0.121\n"
        ),
        encoding="utf-8",
    )


def _write_panel_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id\tstate\tyear\tunemployment_rate\tmedian income\tpopulation\t"
            "poverty_rate\turban_share\n"
            "04013\tArizona\t2005\t4.3\t55231\t4082047\t0.127\t0.889\n"
            "04013\tArizona\t2007\t3.8\t57102\t4218287\t0.121\t0.894\n"
        ),
        encoding="utf-8",
    )


def _write_difference_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,difference_window,unemployment_rate,median income,"
            "population,poverty_rate,urban_share\n"
            "04013,Arizona,2005-2007,-0.5,56166.5,4150167,0.124,0.892\n"
        ),
        encoding="utf-8",
    )


def _write_difference_csv_rows(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        f"{county_id},{state},{difference_window},-0.5,56166.5,4150167,0.124,0.892"
        for county_id, state, difference_window in rows
    )
    path.write_text(
        (
            "county_id,state,difference_window,unemployment_rate,median income,"
            "population,poverty_rate,urban_share\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )


def _write_ambiguous_shape_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,year,difference_window,unemployment_rate,"
            "median income,population,poverty_rate,urban_share\n"
            "04013,Arizona,2005,2005-2007,4.3,55231,4082047,0.127,0.889\n"
        ),
        encoding="utf-8",
    )


def _write_mixed_shape_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "county_id,state,year,difference_window,unemployment_rate,"
            "median income,population,poverty_rate,urban_share\n"
            "04013,Arizona,2005,,4.3,55231,4082047,0.127,0.889\n"
            "01001,Alabama,,2005-2007,-0.3,42100,48600,0.151,0.423\n"
        ),
        encoding="utf-8",
    )


def _write_supported_difference_csv(
    path: Path,
    *,
    n_counties: int,
    state_mode: str = "mixed",
    constant_lanes: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    characteristic_names = tuple(f"county_characteristic_{index:02d}" for index in range(38))
    header = (
        "county_id,state,difference_window,unemployment_rate,median income,population,"
        + ",".join(characteristic_names)
    )
    rows: list[str] = []
    for index in range(n_counties):
        county_id = f"{index + 1:05d}"
        if state_mode == "mixed":
            state = "Arizona" if index == 0 else "Alabama"
        elif state_mode == "treated-only":
            state = "Arizona"
        elif state_mode == "control-only":
            state = "Alabama"
        else:
            raise ValueError(f"unknown fixture state_mode: {state_mode}")
        characteristic_values = ",".join(
            f"{index + characteristic_index / 1000:.3f}"
            for characteristic_index in range(38)
        )
        median_income = 56166.031 + 0.071 * index
        population = 4150167.043 + 0.053 * index
        if constant_lanes:
            median_income = 56166.5
            population = 4150167.0
        rows.append(
            f"{county_id},{state},2005-2007,-0.5,{median_income:.3f},{population:.3f},"
            f"{characteristic_values}"
        )
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_section6_loader_builds_contract_ready_county_year_payload(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()
    contract_module = _load_contract_module()

    _write_panel_csv(tmp_path / "data" / "section6_county_panel.csv")
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    payload = loader_module.load_section6_data(tmp_path)
    audit = contract_module.audit_phase7_section6_loader_payload(payload)

    assert audit.status == "ready"
    assert audit.observed_county_characteristic_count == 2
    assert audit.observed_county_characteristic_names == (
        "poverty_rate",
        "urban_share",
    )
    assert audit.county_characteristic_coverage_status == "partial-local-payload"
    assert payload["lane_metadata"]["observation_shape"] == "county-year panel"
    assert len(payload["analysis_rows"]) == 2
    assert payload["analysis_rows"][0]["county_id"] == "04013"
    assert payload["analysis_rows"][0]["year"] == 2005
    assert payload["analysis_rows"][0]["county_characteristics"] == {
        "poverty_rate": 0.127,
        "urban_share": 0.889,
    }
    assert payload["linear_covariate_manifest"][
        "observed_county_characteristic_count"
    ] == 2
    assert payload["linear_covariate_manifest"][
        "observed_county_characteristic_names"
    ] == ["poverty_rate", "urban_share"]
    assert payload["linear_covariate_manifest"][
        "linear_covariate_construction"
    ] == "pairwise-baseline-characteristic-interactions"
    assert payload["linear_covariate_manifest"][
        "linear_covariate_count_formula"
    ] == "choose(38, 2)"


def test_section6_loader_accepts_county_difference_payload_shape(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()
    contract_module = _load_contract_module()

    _write_difference_csv(tmp_path / "data" / "section6_county_differences.csv")
    _write_manifest(tmp_path / "hddid-py" / "data" / "section6_manifest.json")

    payload = loader_module.load_section6_data(repo_root=tmp_path)
    audit = contract_module.audit_phase7_section6_loader_payload(payload)

    assert audit.status == "ready"
    assert (
        payload["lane_metadata"]["observation_shape"]
        == "county-level 2005-to-2007 difference"
    )
    assert payload["analysis_rows"][0]["county_id"] == "04013"
    assert payload["analysis_rows"][0]["difference_window"] == "2005-2007"
    assert "year" not in payload["analysis_rows"][0]


def test_section6_loader_accepts_tsv_panel_payload(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()
    contract_module = _load_contract_module()

    _write_panel_tsv(tmp_path / "hddid-py" / "data" / "section6_county_panel.tsv")
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    payload = loader_module.load_section6_data(repo_root=tmp_path)
    audit = contract_module.audit_phase7_section6_loader_payload(payload)

    assert audit.status == "ready"
    assert payload["lane_metadata"]["observation_shape"] == "county-year panel"
    assert len(payload["analysis_rows"]) == 2
    assert payload["analysis_rows"][0]["county_id"] == "04013"
    assert payload["analysis_rows"][0]["year"] == 2005
    assert payload["analysis_rows"][0]["county_characteristics"] == {
        "poverty_rate": 0.127,
        "urban_share": 0.889,
    }


def test_section6_loader_excludes_row_metadata_from_county_characteristics(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    data_path = tmp_path / "hddid-py" / "data" / "section6_county_panel.tsv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        (
            "county_id\tstate\tyear\tunemployment_rate\tmedian income\tpopulation\t"
            "poverty_rate\turban_share\trow_provenance\tsource_status\n"
            "04013\tArizona\t2005\t4.3\t55231\t4082047\t0.127\t0.889\t"
            "repository-local-partial-section6-payload\tlocal\n"
            "04013\tArizona\t2007\t3.8\t57102\t4218287\t0.121\t0.894\t"
            "repository-local-partial-section6-payload\tlocal\n"
        ),
        encoding="utf-8",
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    payload = loader_module.load_section6_data(repo_root=tmp_path)
    design = loader_module.build_section6_county_difference_design(tmp_path)

    assert payload["analysis_rows"][0]["county_characteristics"] == {
        "poverty_rate": 0.127,
        "urban_share": 0.889,
    }
    assert payload["linear_covariate_manifest"][
        "observed_county_characteristic_names"
    ] == ["poverty_rate", "urban_share"]
    assert design["design_metadata"]["observed_county_characteristic_count"] == 2
    assert design["difference_rows"][0]["county_characteristic_deltas"] == {
        "poverty_rate": pytest.approx(-0.006),
        "urban_share": pytest.approx(0.005),
    }


def test_section6_county_difference_design_transforms_panel_pairs(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_treated_and_control_panel_csv(tmp_path / "data" / "section6_county_panel.csv")
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    design = loader_module.build_section6_county_difference_design(tmp_path)
    metadata = design["design_metadata"]
    rows = design["difference_rows"]

    assert metadata["source_observation_shape"] == "county-year panel"
    assert metadata["source_transform"] == "paired-2005-2007-panel-difference"
    assert metadata["n_counties"] == 2
    assert metadata["treated_counties"] == 1
    assert metadata["control_counties"] == 1
    assert metadata["observed_sample_support"] == {
        "observed_treated_states": ["Arizona"],
        "observed_control_states": ["Alabama"],
        "observed_treated_county_ids": ["04013"],
        "observed_control_county_ids": ["01001"],
        "missing_sample_roles": [],
        "target_treated_state_count": 11,
        "target_control_state_count": 20,
        "observed_treated_state_count": 1,
        "observed_control_state_count": 1,
        "missing_treated_state_count": 10,
        "missing_control_state_count": 19,
        "missing_treated_states": [
            "Arkansas",
            "Colorado",
            "Maryland",
            "Michigan",
            "Missouri",
            "Montana",
            "Nevada",
            "North Carolina",
            "Ohio",
            "West Virginia",
        ],
        "missing_control_states": [
            "Georgia",
            "Idaho",
            "Indiana",
            "Iowa",
            "Kansas",
            "Kentucky",
            "Louisiana",
            "Mississippi",
            "Nebraska",
            "New Mexico",
            "North Dakota",
            "Oklahoma",
            "South Carolina",
            "South Dakota",
            "Tennessee",
            "Texas",
            "Utah",
            "Virginia",
            "Wyoming",
        ],
    }
    assert metadata["estimation_status"] == "blocked"
    assert metadata["estimation_blockers"] == [
        "incomplete-county-characteristics",
        "n-counties-too-small-for-empirical-trigonometric-basis",
        "rank-deficient-z-lane-basis",
    ]
    assert metadata["paper_county_characteristics_count"] == 38
    assert metadata["linear_covariate_construction"] == (
        "pairwise-baseline-characteristic-interactions"
    )
    assert metadata["linear_covariate_count_formula"] == "choose(38, 2)"
    assert metadata["observed_county_characteristic_count"] == 2
    assert metadata["missing_county_characteristic_count"] == 36
    assert metadata["observed_county_characteristic_names"] == [
        "poverty_rate",
        "urban_share",
    ]
    requirements = metadata["minimum_data_requirements"]
    assert requirements == {
        "minimum_treated_counties": 1,
        "minimum_control_counties": 1,
        "linear_covariate_count": 703,
        "linear_covariate_sample_threshold": (
            "not-a-low-dimensional-readiness-blocker"
        ),
        "minimum_counties_for_empirical_trigonometric_basis": 10,
        "minimum_total_counties": 10,
        "basis_dimension_full": 9,
        "minimum_county_characteristics": 38,
        "observed_county_characteristics": 2,
        "additional_treated_counties_needed": 0,
        "additional_control_counties_needed": 0,
        "additional_counties_needed_for_empirical_trigonometric_basis": 8,
        "additional_counties_needed_for_minimum_total": 8,
        "missing_county_characteristics": 36,
    }
    assert rows[0]["county_id"] == "01001"
    assert rows[0]["treated"] == 0
    assert rows[0]["delta_unemployment_rate"] == pytest.approx(-0.3)
    assert rows[0]["lane_deltas"] == {
        "median income": pytest.approx(1700.0),
        "population": pytest.approx(700.0),
    }
    assert rows[1]["county_id"] == "04013"
    assert rows[1]["treated"] == 1
    assert rows[1]["delta_unemployment_rate"] == pytest.approx(-0.5)
    assert rows[1]["baseline_lanes"] == {
        "median income": pytest.approx(55231.0),
        "population": pytest.approx(4082047.0),
    }


def test_section6_county_difference_design_reports_ready_at_minimum_support(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_supported_difference_csv(
        tmp_path / "data" / "section6_county_differences.csv",
        n_counties=10,
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    design = loader_module.build_section6_county_difference_design(tmp_path)
    metadata = design["design_metadata"]

    assert metadata["source_observation_shape"] == (
        "county-level 2005-to-2007 difference"
    )
    assert metadata["source_transform"] == "precomputed-2005-2007-county-difference"
    assert metadata["n_counties"] == 10
    assert metadata["treated_counties"] == 1
    assert metadata["control_counties"] == 9
    assert metadata["z_lane_basis_rank"] == {
        "median income": {
            "rank": 9,
            "required_rank": 9,
            "full_column_rank": True,
        },
        "population": {
            "rank": 9,
            "required_rank": 9,
            "full_column_rank": True,
        },
    }
    assert metadata["observed_county_characteristic_count"] == 38
    assert metadata["missing_county_characteristic_count"] == 0
    assert metadata["estimation_status"] == "ready"
    assert metadata["estimation_blockers"] == []
    requirements = metadata["minimum_data_requirements"]
    assert requirements["minimum_total_counties"] == 10
    assert requirements["linear_covariate_count"] == 703
    assert (
        requirements["linear_covariate_sample_threshold"]
        == "not-a-low-dimensional-readiness-blocker"
    )
    assert requirements["additional_treated_counties_needed"] == 0
    assert requirements["additional_control_counties_needed"] == 0
    assert requirements[
        "additional_counties_needed_for_empirical_trigonometric_basis"
    ] == 0
    assert requirements["additional_counties_needed_for_minimum_total"] == 0
    assert requirements["missing_county_characteristics"] == 0


def test_section6_hddid_input_builds_ready_pairwise_interaction_design(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_supported_difference_csv(
        tmp_path / "data" / "section6_county_differences.csv",
        n_counties=10,
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    section6_input = loader_module.build_section6_hddid_input(
        tmp_path,
        z_lane="median income",
    )
    data = section6_input.data

    assert section6_input.z_lane == "median income"
    assert section6_input.outcome_encoding == "zero-baseline-delta-outcome"
    assert len(section6_input.x_column_names) == 703
    assert section6_input.x_column_names[0] == (
        "county_characteristic_00*county_characteristic_01"
    )
    assert section6_input.x_column_names[-1] == (
        "county_characteristic_36*county_characteristic_37"
    )
    assert data.n_obs == 10
    assert data.x.shape == (10, 703)
    assert data.basis_family == "trigonometric"
    assert data.basis_degree == 4
    assert data.alpha == pytest.approx(0.05)
    np.testing.assert_allclose(data.y0, np.zeros(10))
    np.testing.assert_allclose(data.y1, np.full(10, -0.5))
    np.testing.assert_array_equal(data.treat, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    expected_median_income = 56166.031 + 0.071 * np.arange(10)
    np.testing.assert_allclose(data.z, expected_median_income)
    np.testing.assert_allclose(data.z0, expected_median_income)
    assert data.x[1, 0] == pytest.approx(1.0 * 1.001)
    assert section6_input.design_metadata["estimation_status"] == "ready"


def test_section6_hddid_input_allows_population_lane_and_explicit_grid(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_supported_difference_csv(
        tmp_path / "data" / "section6_county_differences.csv",
        n_counties=10,
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    section6_input = loader_module.build_section6_hddid_input(
        tmp_path,
        z_lane="population",
        z0=np.array([4000000.0, 5000000.0]),
    )

    expected_population = 4150167.043 + 0.053 * np.arange(10)
    np.testing.assert_allclose(section6_input.data.z, expected_population)
    np.testing.assert_allclose(section6_input.data.z0, np.array([4000000.0, 5000000.0]))


def test_section6_hddid_input_rejects_rank_deficient_z_lane(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_supported_difference_csv(
        tmp_path / "data" / "section6_county_differences.csv",
        n_counties=10,
        constant_lanes=True,
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    design = loader_module.build_section6_county_difference_design(tmp_path)
    assert design["design_metadata"]["z_lane_basis_rank"]["median income"] == {
        "rank": 1,
        "required_rank": 9,
        "full_column_rank": False,
    }
    assert "rank-deficient-z-lane-basis" in design["design_metadata"][
        "estimation_blockers"
    ]
    with pytest.raises(ValueError, match="rank-deficient-z-lane-basis"):
        loader_module.build_section6_hddid_input(tmp_path, z_lane="median income")


def test_section6_hddid_input_reaches_high_dimensional_eq31_lasso(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()
    nuisance_module = importlib.import_module("hddid.nuisance")
    score_module = importlib.import_module("hddid.score")
    estimation_module = importlib.import_module("hddid.estimation")

    _write_supported_difference_csv(
        tmp_path / "data" / "section6_county_differences.csv",
        n_counties=10,
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    section6_input = loader_module.build_section6_hddid_input(tmp_path)
    data = section6_input.data
    nuisance_payload = nuisance_module.NuisancePayload.from_predictions(
        pi_hat=np.full(data.n_obs, 0.5),
        phi0_hat=np.zeros(data.n_obs),
        phi1_hat=np.zeros(data.n_obs),
        treat=data.treat,
        fold_ids=np.ones(data.n_obs, dtype=int),
        fold_diagnostics=[],
        basis_family=data.basis_family,
        basis_degree=data.basis_degree,
        oracle_lane="paper-trigonometric",
    )
    score_payload = score_module.build_score_payload(data, nuisance_payload)

    estimation_payload, result = estimation_module.estimate_eq31_mainline(
        score_payload,
        penalty_lambda=1e12,
        max_iter=5,
    )

    assert score_payload.x_valid.shape == (10, 703)
    assert score_payload.basis_valid_full.shape == (10, 9)
    assert estimation_payload.beta_hat.shape == (703,)
    assert estimation_payload.optimization_metadata["solver"] == (
        "coordinate-descent-lasso"
    )
    assert estimation_payload.optimization_metadata["beta_dimension"] == 703
    assert estimation_payload.optimization_metadata["n_valid_obs"] == 10
    assert result.diagnostics.optimization_metadata["basis_dimension_full"] == 9


def test_section6_hddid_input_rejects_blocked_or_unknown_lane(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_panel_csv(tmp_path / "data" / "section6_county_panel.csv")
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    with pytest.raises(ValueError, match="missing-control-counties"):
        loader_module.build_section6_hddid_input(tmp_path)

    (tmp_path / "data" / "section6_county_panel.csv").unlink()
    _write_supported_difference_csv(
        tmp_path / "data" / "section6_county_differences.csv",
        n_counties=10,
    )
    with pytest.raises(ValueError, match="z_lane must be one of"):
        loader_module.build_section6_hddid_input(tmp_path, z_lane="poverty_rate")


def test_section6_county_difference_design_allows_high_dimensional_linear_block(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_supported_difference_csv(
        tmp_path / "data" / "section6_county_differences.csv",
        n_counties=703,
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    design = loader_module.build_section6_county_difference_design(tmp_path)
    metadata = design["design_metadata"]

    assert metadata["n_counties"] == 703
    assert metadata["treated_counties"] == 1
    assert metadata["control_counties"] == 702
    assert metadata["observed_county_characteristic_count"] == 38
    assert metadata["missing_county_characteristic_count"] == 0
    assert metadata["estimation_status"] == "ready"
    assert metadata["estimation_blockers"] == []
    requirements = metadata["minimum_data_requirements"]
    assert requirements["minimum_total_counties"] == 10
    assert requirements["linear_covariate_count"] == 703
    assert (
        requirements["linear_covariate_sample_threshold"]
        == "not-a-low-dimensional-readiness-blocker"
    )
    assert requirements[
        "additional_counties_needed_for_empirical_trigonometric_basis"
    ] == 0
    assert requirements["additional_counties_needed_for_minimum_total"] == 0
    assert requirements["missing_county_characteristics"] == 0


def test_section6_county_difference_design_blocks_missing_treated_support_only(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_supported_difference_csv(
        tmp_path / "data" / "section6_county_differences.csv",
        n_counties=704,
        state_mode="control-only",
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    design = loader_module.build_section6_county_difference_design(tmp_path)
    metadata = design["design_metadata"]

    assert metadata["n_counties"] == 704
    assert metadata["treated_counties"] == 0
    assert metadata["control_counties"] == 704
    assert metadata["observed_county_characteristic_count"] == 38
    assert metadata["missing_county_characteristic_count"] == 0
    assert metadata["estimation_status"] == "blocked"
    assert metadata["estimation_blockers"] == ["missing-treated-counties"]
    requirements = metadata["minimum_data_requirements"]
    assert requirements["additional_treated_counties_needed"] == 1
    assert requirements["additional_control_counties_needed"] == 0
    assert requirements["additional_counties_needed_for_minimum_total"] == 0
    assert requirements["missing_county_characteristics"] == 0


def test_section6_county_difference_design_blocks_missing_control_support_only(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_supported_difference_csv(
        tmp_path / "data" / "section6_county_differences.csv",
        n_counties=704,
        state_mode="treated-only",
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    design = loader_module.build_section6_county_difference_design(tmp_path)
    metadata = design["design_metadata"]

    assert metadata["n_counties"] == 704
    assert metadata["treated_counties"] == 704
    assert metadata["control_counties"] == 0
    assert metadata["observed_county_characteristic_count"] == 38
    assert metadata["missing_county_characteristic_count"] == 0
    assert metadata["estimation_status"] == "blocked"
    assert metadata["estimation_blockers"] == ["missing-control-counties"]
    requirements = metadata["minimum_data_requirements"]
    assert requirements["additional_treated_counties_needed"] == 0
    assert requirements["additional_control_counties_needed"] == 1
    assert requirements["additional_counties_needed_for_minimum_total"] == 0
    assert requirements["missing_county_characteristics"] == 0


def test_section6_county_difference_design_blocks_small_basis_support(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_supported_difference_csv(
        tmp_path / "data" / "section6_county_differences.csv",
        n_counties=9,
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    design = loader_module.build_section6_county_difference_design(tmp_path)
    metadata = design["design_metadata"]

    assert metadata["n_counties"] == 9
    assert metadata["treated_counties"] == 1
    assert metadata["control_counties"] == 8
    assert metadata["observed_county_characteristic_count"] == 38
    assert metadata["missing_county_characteristic_count"] == 0
    assert metadata["estimation_status"] == "blocked"
    assert metadata["estimation_blockers"] == [
        "n-counties-too-small-for-empirical-trigonometric-basis",
    ]
    requirements = metadata["minimum_data_requirements"]
    assert requirements["minimum_counties_for_empirical_trigonometric_basis"] == 10
    assert requirements[
        "additional_counties_needed_for_empirical_trigonometric_basis"
    ] == 1
    assert requirements["additional_counties_needed_for_minimum_total"] == 1
    assert requirements["missing_county_characteristics"] == 0


def test_section6_county_difference_design_reports_missing_control_blocker(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_panel_csv(tmp_path / "data" / "section6_county_panel.csv")
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    design = loader_module.build_section6_county_difference_design(tmp_path)
    metadata = design["design_metadata"]

    assert metadata["n_counties"] == 1
    assert metadata["treated_counties"] == 1
    assert metadata["control_counties"] == 0
    assert metadata["observed_sample_support"] == {
        "observed_treated_states": ["Arizona"],
        "observed_control_states": [],
        "observed_treated_county_ids": ["04013"],
        "observed_control_county_ids": [],
        "missing_sample_roles": ["control"],
        "target_treated_state_count": 11,
        "target_control_state_count": 20,
        "observed_treated_state_count": 1,
        "observed_control_state_count": 0,
        "missing_treated_state_count": 10,
        "missing_control_state_count": 20,
        "missing_treated_states": [
            "Arkansas",
            "Colorado",
            "Maryland",
            "Michigan",
            "Missouri",
            "Montana",
            "Nevada",
            "North Carolina",
            "Ohio",
            "West Virginia",
        ],
        "missing_control_states": list(CONTROL_STATES),
    }
    assert metadata["estimation_status"] == "blocked"
    assert "missing-control-counties" in metadata["estimation_blockers"]
    assert "incomplete-county-characteristics" in metadata["estimation_blockers"]
    assert metadata["missing_county_characteristic_count"] == 36
    requirements = metadata["minimum_data_requirements"]
    assert requirements["additional_control_counties_needed"] == 1
    assert requirements["additional_counties_needed_for_minimum_total"] == 9
    assert requirements["missing_county_characteristics"] == 36


def test_section6_county_difference_design_rejects_dropped_characteristic_delta(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_panel_csv_with_missing_followup_characteristic(
        tmp_path / "data" / "section6_county_panel.csv"
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    with pytest.raises(
        ValueError,
        match=(
            "Section 6 raw data row has fewer fields than the header: "
            "line 3; urban_share"
        ),
    ):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_conflicting_duplicate_canonical_manifests(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_panel_csv(tmp_path / "data" / "section6_county_panel.csv")
    _write_manifest(tmp_path / "data" / "section6_manifest.json")
    _write_manifest(tmp_path / "hddid-py" / "data" / "section6_manifest.json")

    duplicate_manifest_path = (
        tmp_path / "hddid-py" / "data" / "section6_manifest.json"
    )
    duplicate_manifest = json.loads(duplicate_manifest_path.read_text(encoding="utf-8"))
    duplicate_manifest["treated_state_count"] = 12
    duplicate_manifest_path.write_text(
        json.dumps(duplicate_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting canonical Section 6 manifests"):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_manifest_anchor_drift(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_panel_csv(tmp_path / "data" / "section6_county_panel.csv")
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    manifest_path = tmp_path / "data" / "section6_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["basis_degree"] = 8
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="canonical paper-backed anchors.*basis_degree",
    ):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_malformed_manifest_with_machine_readable_path(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_panel_csv(tmp_path / "data" / "section6_county_panel.csv")
    manifest_path = tmp_path / "data" / "section6_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="malformed-manifest:data/section6_manifest.json",
    ):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_conflicting_duplicate_canonical_panel_assets(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_panel_csv(tmp_path / "data" / "section6_county_panel.csv")
    _write_panel_tsv(tmp_path / "hddid-py" / "data" / "section6_county_panel.tsv")
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    duplicate_panel_path = tmp_path / "hddid-py" / "data" / "section6_county_panel.tsv"
    duplicate_panel_path.write_text(
        (
            "county_id\tstate\tyear\tunemployment_rate\tmedian income\tpopulation\t"
            "poverty_rate\turban_share\n"
            "04013\tArizona\t2005\t4.3\t55231\t4082047\t0.127\t0.889\n"
            "04013\tArizona\t2007\t4.8\t57102\t4218287\t0.121\t0.894\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="conflicting canonical Section 6 raw data assets",
    ):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_ambiguous_multiple_canonical_raw_shapes(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_panel_csv(tmp_path / "data" / "section6_county_panel.csv")
    _write_difference_csv(tmp_path / "hddid-py" / "data" / "section6_county_differences.csv")
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    with pytest.raises(
        ValueError,
        match="multiple canonical Section 6 raw data shapes",
    ):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_excluded_or_out_of_universe_raw_states(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_panel_csv_for_state(
        tmp_path / "data" / "section6_county_panel.csv",
        "New Hampshire",
    )
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    with pytest.raises(
        ValueError,
        match="Section 6 raw data contains excluded states: New Hampshire",
    ):
        loader_module.load_section6_data(tmp_path)

    _write_panel_csv_for_state(
        tmp_path / "data" / "section6_county_panel.csv",
        "California",
    )

    with pytest.raises(
        ValueError,
        match="Section 6 raw data contains out-of-universe states: California",
    ):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_unpaired_or_duplicate_county_year_rows(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()
    raw_path = tmp_path / "data" / "section6_county_panel.csv"

    _write_manifest(tmp_path / "data" / "section6_manifest.json")
    _write_panel_csv_rows(raw_path, [("04013", "Arizona", 2005)])

    with pytest.raises(
        ValueError,
        match="requires paired 2005/2007 rows for each county: 04013",
    ):
        loader_module.load_section6_data(tmp_path)

    _write_panel_csv_rows(
        raw_path,
        [
            ("04013", "Arizona", 2005),
            ("04013", "Arizona", 2005),
            ("04013", "Arizona", 2007),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Section 6 county-year panel has duplicate rows: 04013:2005",
    ):
        loader_module.load_section6_data(tmp_path)

    _write_panel_csv_rows(
        raw_path,
        [
            ("04013", "Arizona", 2005),
            ("04013", "Arizona", 2006),
            ("04013", "Arizona", 2007),
        ],
    )

    with pytest.raises(
        ValueError,
        match="contains years outside 2005/2007: 2006",
    ):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_missing_required_core_fields(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_manifest(tmp_path / "data" / "section6_manifest.json")
    _write_panel_csv_missing_core_field(
        tmp_path / "data" / "section6_county_panel.csv"
    )

    with pytest.raises(
        ValueError,
        match=(
            "Section 6 raw data rows missing required fields: "
            "04013: median income; 04013: median income"
        ),
    ):
        loader_module.load_section6_data(tmp_path)

    (tmp_path / "data" / "section6_county_panel.csv").unlink()
    _write_difference_csv_missing_core_field(
        tmp_path / "data" / "section6_county_differences.csv"
    )

    with pytest.raises(
        ValueError,
        match=(
            "Section 6 raw data rows missing required fields: "
            "04013: unemployment_rate"
        ),
    ):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_duplicate_raw_columns(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()

    _write_manifest(tmp_path / "data" / "section6_manifest.json")
    _write_panel_csv_duplicate_header(
        tmp_path / "data" / "section6_county_panel.csv"
    )

    with pytest.raises(
        ValueError,
        match="Section 6 raw data has duplicate columns: urban_share",
    ):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_ragged_rows_and_empty_characteristics(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()
    raw_path = tmp_path / "data" / "section6_county_panel.csv"

    _write_manifest(tmp_path / "data" / "section6_manifest.json")
    _write_panel_csv_extra_row_field(raw_path)
    with pytest.raises(
        ValueError,
        match="Section 6 raw data row has more fields than the header: line 2",
    ):
        loader_module.load_section6_data(tmp_path)

    _write_panel_csv_missing_row_field(raw_path)
    with pytest.raises(
        ValueError,
        match="Section 6 raw data row has fewer fields than the header: line 2; urban_share",
    ):
        loader_module.load_section6_data(tmp_path)

    _write_panel_csv_missing_characteristic_value(raw_path)
    with pytest.raises(
        ValueError,
        match="missing county-characteristic values: 04013: urban_share",
    ):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_invalid_county_difference_rows(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()
    raw_path = tmp_path / "data" / "section6_county_differences.csv"

    _write_manifest(tmp_path / "data" / "section6_manifest.json")
    _write_difference_csv_rows(raw_path, [("04013", "Arizona", "2004-2007")])

    with pytest.raises(
        ValueError,
        match="county-difference rows must use 2005-2007 window: 2004-2007",
    ):
        loader_module.load_section6_data(tmp_path)

    _write_difference_csv_rows(
        raw_path,
        [
            ("04013", "Arizona", "2005-2007"),
            ("04013", "Arizona", "2005-2007"),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Section 6 county-difference rows duplicate counties: 04013",
    ):
        loader_module.load_section6_data(tmp_path)

    _write_difference_csv_rows(raw_path, [("", "Arizona", "2005-2007")])

    with pytest.raises(
        ValueError,
        match="Section 6 county-difference rows must include county_id",
    ):
        loader_module.load_section6_data(tmp_path)


def test_section6_loader_rejects_ambiguous_or_mixed_raw_shapes(
    tmp_path: Path,
) -> None:
    loader_module = _load_loader_module()
    raw_path = tmp_path / "data" / "section6_county_panel.csv"
    _write_manifest(tmp_path / "data" / "section6_manifest.json")

    _write_ambiguous_shape_csv(raw_path)
    with pytest.raises(
        ValueError,
        match="cannot contain both year and difference_window: 04013",
    ):
        loader_module.load_section6_data(tmp_path)

    _write_mixed_shape_csv(raw_path)
    with pytest.raises(
        ValueError,
        match="Section 6 raw data rows mix panel and difference shapes",
    ):
        loader_module.load_section6_data(tmp_path)
