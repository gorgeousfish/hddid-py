from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .inputs import ValidatedHDDIDData


_RAW_DATA_CANDIDATES = (
    "data/section6_county_panel.csv",
    "data/section6_county_panel.tsv",
    "data/section6_county_differences.csv",
    "data/section6_county_differences.tsv",
    "hddid-py/data/section6_county_panel.csv",
    "hddid-py/data/section6_county_panel.tsv",
    "hddid-py/data/section6_county_differences.csv",
    "hddid-py/data/section6_county_differences.tsv",
)
_MANIFEST_CANDIDATES = (
    "data/section6_manifest.json",
    "hddid-py/data/section6_manifest.json",
)
_CORE_FIELDS = {
    "county_id",
    "state",
    "year",
    "difference_window",
    "unemployment_rate",
    "median income",
    "population",
}
_ROW_METADATA_FIELDS = {
    "row_provenance",
    "provenance",
    "source_status",
}
_PANEL_REQUIRED_CORE_FIELDS = (
    "county_id",
    "state",
    "year",
    "unemployment_rate",
    "median income",
    "population",
)
_DIFFERENCE_REQUIRED_CORE_FIELDS = (
    "county_id",
    "state",
    "difference_window",
    "unemployment_rate",
    "median income",
    "population",
)
_TREATED_STATES = (
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
_CONTROL_STATES = (
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
_EXCLUDED_STATES = ("New Hampshire", "Pennsylvania")
_PANEL_REQUIRED_YEARS = (2005, 2007)
_DIFFERENCE_WINDOW = "2005-2007"
_SAMPLE_RULE = {
    "baseline_universe": "2006 federal-binding states",
    "excluded_states": list(_EXCLUDED_STATES),
    "treatment_definition": "state minimum wage increased by Q1 2007",
    "control_definition": (
        "state minimum wage did not increase until the federal increase in July 2007"
    ),
}
_EXPECTED_MANIFEST = {
    "time_window": "2005-2007",
    "treated_state_count": len(_TREATED_STATES),
    "treated_states": list(_TREATED_STATES),
    "control_states": list(_CONTROL_STATES),
    "sample_rule": _SAMPLE_RULE,
    "excluded_states": list(_EXCLUDED_STATES),
    "z_lanes": ["median income", "population"],
    "linear_covariate_count": 703,
    "county_characteristics_count": 38,
    "linear_covariate_source": "reported 703-covariate block; arithmetic matches pairwise interactions among 38 county-level characteristics",
    "linear_covariate_construction": "pairwise-baseline-characteristic-interactions",
    "linear_covariate_count_formula": "choose(38, 2)",
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
}


@dataclass(slots=True)
class Section6HDDIDInput:
    """Validated package input object built from a ready Section 6 preflight."""

    data: "ValidatedHDDIDData"
    z_lane: str
    x_column_names: tuple[str, ...]
    outcome_encoding: str
    design_metadata: dict[str, object]


def _coerce_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is None:
        return Path(__file__).resolve().parents[3]
    return Path(repo_root).expanduser().resolve()


def _resolve_required_path(repo_root: Path, candidates: tuple[str, ...]) -> Path:
    for relative_path in candidates:
        candidate = repo_root / relative_path
        if candidate.is_file():
            return candidate
    joined = ", ".join(candidates)
    raise FileNotFoundError(f"required Section 6 asset not found under: {joined}")


def _resolve_existing_paths(repo_root: Path, candidates: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(
        repo_root / relative_path
        for relative_path in candidates
        if (repo_root / relative_path).is_file()
    )


def _raw_asset_key(raw_data_path: Path) -> str:
    stem = raw_data_path.stem
    if stem in {"section6_county_panel", "section6_county_differences"}:
        return stem
    return raw_data_path.name


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped == "":
        return ""
    try:
        if any(char in stripped for char in (".", "e", "E")):
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def _parse_field(key: str, value: str) -> Any:
    if key in {"county_id", "state", "difference_window"}:
        return value.strip()
    return _parse_scalar(value)


def _relative_manifest_path(repo_root: Path, manifest_path: Path) -> str:
    try:
        return manifest_path.relative_to(repo_root).as_posix()
    except ValueError:
        return manifest_path.as_posix()


def _read_manifest(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    relative_path = _relative_manifest_path(repo_root, manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed-manifest:{relative_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"malformed-manifest:{relative_path}")
    return payload


def _manifest_values_match(expected: Any, observed: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(observed) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return list(observed) == expected if isinstance(observed, list | tuple) else False
    if isinstance(expected, dict):
        return dict(observed) == expected if isinstance(observed, dict) else False
    return observed == expected


def _validate_manifest_against_paper_contract(manifest: dict[str, Any]) -> None:
    mismatched_fields = [
        key
        for key, expected in _EXPECTED_MANIFEST.items()
        if not _manifest_values_match(expected, manifest.get(key))
    ]
    if mismatched_fields:
        joined = ", ".join(sorted(mismatched_fields))
        raise ValueError(f"canonical paper-backed anchors drifted: {joined}")


def _read_canonical_manifest(repo_root: Path) -> dict[str, Any]:
    manifest_paths = _resolve_existing_paths(repo_root, _MANIFEST_CANDIDATES)
    if not manifest_paths:
        _resolve_required_path(repo_root, _MANIFEST_CANDIDATES)

    manifest_payloads = tuple(_read_manifest(repo_root, path) for path in manifest_paths)
    first_payload = manifest_payloads[0]
    if any(payload != first_payload for payload in manifest_payloads[1:]):
        raise ValueError("conflicting canonical Section 6 manifests detected")
    _validate_manifest_against_paper_contract(first_payload)
    return first_payload


def _build_sieve_basis_for_section6(
    z_values: np.ndarray,
    manifest: dict[str, Any],
) -> np.ndarray:
    try:
        from .basis import build_sieve_basis
    except ImportError:
        from hddid.basis import build_sieve_basis

    return build_sieve_basis(
        z_values,
        str(manifest["basis_family"]),
        int(manifest["basis_degree"]),
    )


def _section6_z_lane_basis_rank(
    difference_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, dict[str, int | bool]]:
    rank_by_lane: dict[str, dict[str, int | bool]] = {}
    for lane in manifest["z_lanes"]:
        lane_name = str(lane)
        z_values = np.asarray(
            [
                _numeric_value(
                    dict(row["baseline_lanes"])[lane_name],
                    f"baseline_lanes.{lane_name}",
                )
                for row in difference_rows
            ],
            dtype=float,
        )
        basis = _build_sieve_basis_for_section6(z_values, manifest)
        rank = int(np.linalg.matrix_rank(basis))
        required_rank = int(basis.shape[1])
        rank_by_lane[lane_name] = {
            "rank": rank,
            "required_rank": required_rank,
            "full_column_rank": rank == required_rank,
        }
    return rank_by_lane


def _validate_analysis_rows_against_sample_roster(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    treated_states = set(manifest["treated_states"])
    control_states = set(manifest["control_states"])
    excluded_states = set(manifest["excluded_states"])
    allowed_states = treated_states | control_states

    observed_states: set[str] = set()
    for row in rows:
        if "state" not in row:
            raise ValueError("Section 6 raw data rows must include state")
        state = str(row["state"]).strip()
        if state == "":
            raise ValueError("Section 6 raw data rows must include state")
        observed_states.add(state)

    observed_excluded = tuple(sorted(observed_states & excluded_states))
    if observed_excluded:
        joined = ", ".join(observed_excluded)
        raise ValueError(f"Section 6 raw data contains excluded states: {joined}")

    observed_out_of_universe = tuple(sorted(observed_states - allowed_states))
    if observed_out_of_universe:
        joined = ", ".join(observed_out_of_universe)
        raise ValueError(
            f"Section 6 raw data contains out-of-universe states: {joined}"
        )


def _validate_required_core_fields(rows: list[dict[str, Any]]) -> None:
    observation_shape = _observation_shape(rows)
    required_fields = (
        _DIFFERENCE_REQUIRED_CORE_FIELDS
        if observation_shape == "county-level 2005-to-2007 difference"
        else _PANEL_REQUIRED_CORE_FIELDS
    )
    rows_with_missing_fields: list[str] = []
    for row in rows:
        missing_fields = tuple(
            field
            for field in required_fields
            if field not in row or str(row[field]).strip() == ""
        )
        if not missing_fields:
            continue
        county_id = str(row.get("county_id", "")).strip() or "<unknown>"
        rows_with_missing_fields.append(
            f"{county_id}: " + ", ".join(missing_fields)
        )

    if rows_with_missing_fields:
        joined = "; ".join(rows_with_missing_fields)
        raise ValueError(f"Section 6 raw data rows missing required fields: {joined}")


def _validate_county_year_panel_pairs(rows: list[dict[str, Any]]) -> None:
    if _observation_shape(rows) != "county-year panel":
        return

    years_by_county: dict[str, set[int]] = {}
    states_by_county: dict[str, set[str]] = {}
    seen_county_years: set[tuple[str, int]] = set()
    duplicate_county_years: set[tuple[str, int]] = set()
    invalid_years: set[int] = set()

    for row in rows:
        county_id = str(row.get("county_id", "")).strip()
        if county_id == "":
            raise ValueError("Section 6 county-year panel rows must include county_id")
        state = str(row.get("state", "")).strip()
        if state == "":
            raise ValueError("Section 6 county-year panel rows must include state")
        try:
            year = int(row["year"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Section 6 county-year panel rows must include year"
            ) from exc

        if year not in _PANEL_REQUIRED_YEARS:
            invalid_years.add(year)
        county_year = (county_id, year)
        if county_year in seen_county_years:
            duplicate_county_years.add(county_year)
        seen_county_years.add(county_year)
        years_by_county.setdefault(county_id, set()).add(year)
        states_by_county.setdefault(county_id, set()).add(state)

    if invalid_years:
        joined = ", ".join(str(year) for year in sorted(invalid_years))
        raise ValueError(
            f"Section 6 county-year panel contains years outside 2005/2007: {joined}"
        )

    if duplicate_county_years:
        joined = ", ".join(
            f"{county_id}:{year}"
            for county_id, year in sorted(duplicate_county_years)
        )
        raise ValueError(f"Section 6 county-year panel has duplicate rows: {joined}")

    inconsistent_counties = tuple(
        sorted(
            county_id
            for county_id, states in states_by_county.items()
            if len(states) > 1
        )
    )
    if inconsistent_counties:
        joined = ", ".join(inconsistent_counties)
        raise ValueError(
            f"Section 6 county-year panel has inconsistent county states: {joined}"
        )

    required_years = set(_PANEL_REQUIRED_YEARS)
    incomplete_counties = tuple(
        sorted(
            county_id
            for county_id, years in years_by_county.items()
            if years != required_years
        )
    )
    if incomplete_counties:
        joined = ", ".join(incomplete_counties)
        raise ValueError(
            "Section 6 county-year panel requires paired 2005/2007 rows "
            f"for each county: {joined}"
        )


def _validate_county_difference_rows(rows: list[dict[str, Any]]) -> None:
    if _observation_shape(rows) != "county-level 2005-to-2007 difference":
        return

    seen_counties: set[str] = set()
    duplicate_counties: set[str] = set()
    invalid_windows: set[str] = set()

    for row in rows:
        county_id = str(row.get("county_id", "")).strip()
        if county_id == "":
            raise ValueError("Section 6 county-difference rows must include county_id")
        state = str(row.get("state", "")).strip()
        if state == "":
            raise ValueError("Section 6 county-difference rows must include state")
        difference_window = str(row.get("difference_window", "")).strip()
        if difference_window == "":
            raise ValueError(
                "Section 6 county-difference rows must include difference_window"
            )
        if difference_window != _DIFFERENCE_WINDOW:
            invalid_windows.add(difference_window)
        if county_id in seen_counties:
            duplicate_counties.add(county_id)
        seen_counties.add(county_id)

    if invalid_windows:
        joined = ", ".join(sorted(invalid_windows))
        raise ValueError(
            "Section 6 county-difference rows must use 2005-2007 window: "
            f"{joined}"
        )

    if duplicate_counties:
        joined = ", ".join(sorted(duplicate_counties))
        raise ValueError(
            f"Section 6 county-difference rows duplicate counties: {joined}"
        )


def _county_characteristic_names(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    if not rows:
        return ()
    names_by_row: list[tuple[str, ...]] = []
    for row in rows:
        county_id = str(row.get("county_id", "")).strip() or "<unknown>"
        characteristics = row.get("county_characteristics")
        if not isinstance(characteristics, dict):
            raise ValueError(
                "Section 6 raw data county_characteristics must be a mapping "
                f"for county {county_id}"
            )
        names_by_row.append(tuple(sorted(str(key) for key in characteristics)))

    first_names = names_by_row[0]
    first_name_set = set(first_names)
    inconsistent_details: list[str] = []
    for row, names in zip(rows, names_by_row, strict=True):
        if names == first_names:
            continue
        county_id = str(row.get("county_id", "")).strip() or "<unknown>"
        name_set = set(names)
        detail_parts: list[str] = []
        missing_from_row = sorted(first_name_set - name_set)
        extra_in_row = sorted(name_set - first_name_set)
        if missing_from_row:
            detail_parts.append("missing " + ", ".join(missing_from_row))
        if extra_in_row:
            detail_parts.append("extra " + ", ".join(extra_in_row))
        inconsistent_details.append(f"{county_id} ({'; '.join(detail_parts)})")
    if inconsistent_details:
        joined = ", ".join(inconsistent_details)
        raise ValueError(
            "Section 6 raw data rows must expose the same county-characteristic "
            f"fields: {joined}"
        )
    return first_names


def _is_treated_state(state: str, manifest: dict[str, Any]) -> bool:
    treated_states = set(manifest["treated_states"])
    control_states = set(manifest["control_states"])
    if state in treated_states:
        return True
    if state in control_states:
        return False
    raise ValueError(f"Section 6 state outside manifest roster: {state}")


def _numeric_difference(later: Any, earlier: Any, field: str) -> float:
    try:
        return float(later) - float(earlier)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Section 6 numeric difference field is non-numeric: {field}") from exc


def _numeric_value(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Section 6 numeric field is non-numeric: {field}") from exc


def _validate_county_characteristic_keys(
    county_id: str,
    baseline_characteristics: dict[str, Any],
    followup_characteristics: dict[str, Any],
) -> None:
    baseline_keys = set(baseline_characteristics)
    followup_keys = set(followup_characteristics)
    if baseline_keys == followup_keys:
        return

    missing_followup = sorted(baseline_keys - followup_keys)
    missing_baseline = sorted(followup_keys - baseline_keys)
    details: list[str] = []
    if missing_followup:
        details.append("missing 2007: " + ", ".join(missing_followup))
    if missing_baseline:
        details.append("missing 2005: " + ", ".join(missing_baseline))
    raise ValueError(
        "Section 6 county-year panel characteristic fields must match within "
        f"county {county_id}: {'; '.join(details)}"
    )


def _panel_rows_to_county_differences(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    by_county: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        county_id = str(row["county_id"])
        year = int(row["year"])
        by_county.setdefault(county_id, {})[year] = row

    difference_rows: list[dict[str, Any]] = []
    lane_names = tuple(str(name) for name in manifest["z_lanes"])
    for county_id in sorted(by_county):
        county_rows = by_county[county_id]
        baseline = county_rows[_PANEL_REQUIRED_YEARS[0]]
        followup = county_rows[_PANEL_REQUIRED_YEARS[1]]
        state = str(baseline["state"])
        baseline_characteristics = dict(baseline["county_characteristics"])
        followup_characteristics = dict(followup["county_characteristics"])
        _validate_county_characteristic_keys(
            county_id,
            baseline_characteristics,
            followup_characteristics,
        )
        characteristic_deltas = {
            key: _numeric_difference(followup_characteristics[key], baseline_value, key)
            for key, baseline_value in baseline_characteristics.items()
        }
        difference_rows.append(
            {
                "county_id": county_id,
                "state": state,
                "difference_window": _DIFFERENCE_WINDOW,
                "treated": int(_is_treated_state(state, manifest)),
                "delta_unemployment_rate": _numeric_difference(
                    followup["unemployment_rate"],
                    baseline["unemployment_rate"],
                    "unemployment_rate",
                ),
                "baseline_lanes": {
                    lane: _numeric_value(baseline[lane], lane) for lane in lane_names
                },
                "followup_lanes": {
                    lane: _numeric_value(followup[lane], lane) for lane in lane_names
                },
                "lane_deltas": {
                    lane: _numeric_difference(followup[lane], baseline[lane], lane)
                    for lane in lane_names
                },
                "baseline_county_characteristics": baseline_characteristics,
                "county_characteristic_deltas": characteristic_deltas,
            }
        )
    return difference_rows


def _difference_rows_to_county_differences(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    lane_names = tuple(str(name) for name in manifest["z_lanes"])
    difference_rows: list[dict[str, Any]] = []
    for row in rows:
        state = str(row["state"])
        difference_rows.append(
            {
                "county_id": str(row["county_id"]),
                "state": state,
                "difference_window": str(row["difference_window"]),
                "treated": int(_is_treated_state(state, manifest)),
                "delta_unemployment_rate": _numeric_value(
                    row["unemployment_rate"], "unemployment_rate"
                ),
                "baseline_lanes": {
                    lane: _numeric_value(row[lane], lane) for lane in lane_names
                },
                "followup_lanes": {},
                "lane_deltas": {},
                "baseline_county_characteristics": dict(row["county_characteristics"]),
                "county_characteristic_deltas": {},
            }
        )
    return difference_rows


def _read_analysis_rows(raw_data_path: Path) -> list[dict[str, Any]]:
    with raw_data_path.open(encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            reader = csv.DictReader(handle, dialect=dialect)
        except csv.Error:
            reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Section 6 raw data must have a header row")
        fieldnames = [field for field in reader.fieldnames if field is not None]
        duplicate_fields = sorted(
            field
            for field in set(fieldnames)
            if fieldnames.count(field) > 1
        )
        if duplicate_fields:
            joined = ", ".join(duplicate_fields)
            raise ValueError(f"Section 6 raw data has duplicate columns: {joined}")

        rows: list[dict[str, Any]] = []
        for row_number, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise ValueError(
                    "Section 6 raw data row has more fields than the header: "
                    f"line {row_number}"
                )
            missing_columns = [
                field
                for field, value in raw_row.items()
                if field is not None and value is None
            ]
            if missing_columns:
                joined = ", ".join(missing_columns)
                raise ValueError(
                    "Section 6 raw data row has fewer fields than the header: "
                    f"line {row_number}; {joined}"
                )
            parsed_row = {
                key: _parse_field(key, value)
                for key, value in raw_row.items()
                if key is not None and value is not None
            }
            characteristic_fields = {
                key: value
                for key, value in parsed_row.items()
                if key not in _CORE_FIELDS and key.lower() not in _ROW_METADATA_FIELDS
            }
            missing_characteristics = [
                key
                for key, value in characteristic_fields.items()
                if str(value).strip() == ""
            ]
            if missing_characteristics:
                county_id = str(parsed_row.get("county_id", "")).strip() or "<unknown>"
                joined = ", ".join(sorted(missing_characteristics))
                raise ValueError(
                    "Section 6 raw data row has missing county-characteristic "
                    f"values: {county_id}: {joined}"
                )
            analysis_row: dict[str, Any] = {
                key: value
                for key, value in parsed_row.items()
                if key in _CORE_FIELDS
            }
            analysis_row["county_characteristics"] = characteristic_fields
            rows.append(analysis_row)
    if not rows:
        raise ValueError("Section 6 raw data must contain at least one observation")
    return rows


def _section6_minimum_data_requirements(
    *,
    n_counties: int,
    treated_count: int,
    control_count: int,
    observed_characteristic_count: int,
    manifest: dict[str, Any],
) -> dict[str, int]:
    basis_degree = int(manifest["basis_degree"])
    basis_dimension_full = 2 * basis_degree + 1
    linear_covariate_count = int(manifest["linear_covariate_count"])
    paper_characteristic_count = int(manifest["county_characteristics_count"])
    minimum_counties_for_basis = basis_dimension_full + 1
    minimum_total_counties = max(2, minimum_counties_for_basis)

    return {
        "minimum_treated_counties": 1,
        "minimum_control_counties": 1,
        "linear_covariate_count": linear_covariate_count,
        "linear_covariate_sample_threshold": (
            "not-a-low-dimensional-readiness-blocker"
        ),
        "minimum_counties_for_empirical_trigonometric_basis": (
            minimum_counties_for_basis
        ),
        "minimum_total_counties": minimum_total_counties,
        "basis_dimension_full": basis_dimension_full,
        "minimum_county_characteristics": paper_characteristic_count,
        "observed_county_characteristics": observed_characteristic_count,
        "additional_treated_counties_needed": max(1 - treated_count, 0),
        "additional_control_counties_needed": max(1 - control_count, 0),
        "additional_counties_needed_for_empirical_trigonometric_basis": max(
            minimum_counties_for_basis - n_counties,
            0,
        ),
        "additional_counties_needed_for_minimum_total": max(
            minimum_total_counties - n_counties,
            0,
        ),
        "missing_county_characteristics": max(
            paper_characteristic_count - observed_characteristic_count,
            0,
        ),
    }


def _sample_support_summary(
    difference_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, object]:
    treated_states = sorted(
        {str(row["state"]) for row in difference_rows if int(row["treated"]) == 1}
    )
    control_states = sorted(
        {str(row["state"]) for row in difference_rows if int(row["treated"]) == 0}
    )
    treated_county_ids = sorted(
        str(row["county_id"]) for row in difference_rows if int(row["treated"]) == 1
    )
    control_county_ids = sorted(
        str(row["county_id"]) for row in difference_rows if int(row["treated"]) == 0
    )
    missing_roles: list[str] = []
    if not treated_county_ids:
        missing_roles.append("treated")
    if not control_county_ids:
        missing_roles.append("control")
    target_treated_states = tuple(str(state) for state in manifest["treated_states"])
    target_control_states = tuple(str(state) for state in manifest["control_states"])
    missing_treated_states = sorted(set(target_treated_states) - set(treated_states))
    missing_control_states = sorted(set(target_control_states) - set(control_states))
    return {
        "observed_treated_states": treated_states,
        "observed_control_states": control_states,
        "observed_treated_county_ids": treated_county_ids,
        "observed_control_county_ids": control_county_ids,
        "missing_sample_roles": missing_roles,
        "target_treated_state_count": len(target_treated_states),
        "target_control_state_count": len(target_control_states),
        "observed_treated_state_count": len(treated_states),
        "observed_control_state_count": len(control_states),
        "missing_treated_state_count": len(missing_treated_states),
        "missing_control_state_count": len(missing_control_states),
        "missing_treated_states": missing_treated_states,
        "missing_control_states": missing_control_states,
    }


def _interaction_column_names(
    characteristic_names: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        f"{left}*{right}"
        for left, right in combinations(characteristic_names, 2)
    )


def _interaction_matrix(
    difference_rows: list[dict[str, Any]],
    characteristic_names: tuple[str, ...],
) -> np.ndarray:
    columns: list[list[float]] = []
    for left, right in combinations(characteristic_names, 2):
        columns.append(
            [
                _numeric_value(
                    dict(row["baseline_county_characteristics"])[left],
                    f"baseline_county_characteristics.{left}",
                )
                * _numeric_value(
                    dict(row["baseline_county_characteristics"])[right],
                    f"baseline_county_characteristics.{right}",
                )
                for row in difference_rows
            ]
        )
    if not columns:
        return np.empty((len(difference_rows), 0), dtype=float)
    return np.asarray(columns, dtype=float).T


def _observation_shape(rows: list[dict[str, Any]]) -> str:
    shapes = {_row_observation_shape(row) for row in rows}
    if len(shapes) != 1:
        raise ValueError("Section 6 raw data rows mix panel and difference shapes")
    return next(iter(shapes))


def _row_observation_shape(row: dict[str, Any]) -> str:
    has_year = "year" in row and str(row["year"]).strip() != ""
    has_difference_window = (
        "difference_window" in row and str(row["difference_window"]).strip() != ""
    )
    if has_year and has_difference_window:
        county_id = str(row.get("county_id", "")).strip() or "<unknown>"
        raise ValueError(
            "Section 6 raw data row cannot contain both year and difference_window: "
            f"{county_id}"
        )
    if has_difference_window:
        return "county-level 2005-to-2007 difference"
    if has_year:
        return "county-year panel"
    county_id = str(row.get("county_id", "")).strip() or "<unknown>"
    raise ValueError(
        "Section 6 raw data row must contain either year or difference_window: "
        f"{county_id}"
    )


def _load_canonical_raw_data(
    repo_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    raw_data_paths = _resolve_existing_paths(repo_root, _RAW_DATA_CANDIDATES)
    if not raw_data_paths:
        _resolve_required_path(repo_root, _RAW_DATA_CANDIDATES)

    logical_asset_keys = {_raw_asset_key(path) for path in raw_data_paths}
    if len(logical_asset_keys) > 1:
        raise ValueError("multiple canonical Section 6 raw data shapes detected")

    parsed_by_path = {
        path: _read_analysis_rows(path)
        for path in raw_data_paths
    }
    grouped_paths: dict[str, list[Path]] = {}
    for path in raw_data_paths:
        grouped_paths.setdefault(_raw_asset_key(path), []).append(path)

    for paths in grouped_paths.values():
        canonical_rows = parsed_by_path[paths[0]]
        for candidate in paths[1:]:
            if parsed_by_path[candidate] != canonical_rows:
                raise ValueError("conflicting canonical Section 6 raw data assets detected")

    observed_shapes = {
        _observation_shape(rows)
        for rows in parsed_by_path.values()
    }
    if len(observed_shapes) > 1:
        raise ValueError("multiple canonical Section 6 raw data shapes detected")

    selected_path = raw_data_paths[0]
    return selected_path, parsed_by_path[selected_path]


def load_section6_data(
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    root = _coerce_repo_root(repo_root)
    manifest = _read_canonical_manifest(root)
    raw_data_path, analysis_rows = _load_canonical_raw_data(root)
    _validate_analysis_rows_against_sample_roster(analysis_rows, manifest)
    _validate_county_year_panel_pairs(analysis_rows)
    _validate_county_difference_rows(analysis_rows)
    _validate_required_core_fields(analysis_rows)
    observed_characteristic_names = _county_characteristic_names(analysis_rows)

    return {
        "analysis_rows": analysis_rows,
        "sample_metadata": {
            "treated_states": manifest["treated_states"],
            "control_states": manifest["control_states"],
            "excluded_states": manifest["excluded_states"],
            "sample_rule": manifest["sample_rule"],
        },
        "lane_metadata": {
            "observation_shape": _observation_shape(analysis_rows),
            "lane_role_map": manifest["lane_role_map"],
            "basis_family": manifest["basis_family"],
            "basis_degree": manifest["basis_degree"],
            "confidence_level": manifest["confidence_level"],
            "disallowed_shortcuts": manifest["disallowed_shortcuts"],
        },
        "linear_covariate_manifest": {
            "linear_covariate_count": manifest["linear_covariate_count"],
            "county_characteristics_count": manifest["county_characteristics_count"],
            "linear_covariate_source": manifest["linear_covariate_source"],
            "linear_covariate_construction": manifest[
                "linear_covariate_construction"
            ],
            "linear_covariate_count_formula": manifest[
                "linear_covariate_count_formula"
            ],
            "observed_county_characteristic_count": len(observed_characteristic_names),
            "observed_county_characteristic_names": list(
                observed_characteristic_names
            ),
        },
    }


def build_section6_county_difference_design(
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Build the checked county-level 2005--2007 design object for Section 6.

    This function only transforms repository-local Section 6 assets that have
    already passed the manifest, roster, and county-pairing checks. It does not
    estimate the empirical application.
    """

    root = _coerce_repo_root(repo_root)
    manifest = _read_canonical_manifest(root)
    raw_data_path, analysis_rows = _load_canonical_raw_data(root)
    _validate_analysis_rows_against_sample_roster(analysis_rows, manifest)
    _validate_county_year_panel_pairs(analysis_rows)
    _validate_county_difference_rows(analysis_rows)
    _validate_required_core_fields(analysis_rows)
    observed_characteristic_names = _county_characteristic_names(analysis_rows)

    observation_shape = _observation_shape(analysis_rows)
    if observation_shape == "county-year panel":
        difference_rows = _panel_rows_to_county_differences(analysis_rows, manifest)
        source_transform = "paired-2005-2007-panel-difference"
    else:
        difference_rows = _difference_rows_to_county_differences(analysis_rows, manifest)
        source_transform = "precomputed-2005-2007-county-difference"

    treated_count = sum(int(row["treated"]) for row in difference_rows)
    control_count = len(difference_rows) - treated_count
    sample_support = _sample_support_summary(difference_rows, manifest)
    observed_characteristic_count = len(observed_characteristic_names)
    paper_characteristic_count = int(manifest["county_characteristics_count"])
    missing_characteristic_count = max(
        paper_characteristic_count - observed_characteristic_count,
        0,
    )
    z_lane_basis_rank = _section6_z_lane_basis_rank(difference_rows, manifest)
    minimum_data_requirements = _section6_minimum_data_requirements(
        n_counties=len(difference_rows),
        treated_count=treated_count,
        control_count=control_count,
        observed_characteristic_count=observed_characteristic_count,
        manifest=manifest,
    )
    blockers: list[str] = []
    if treated_count == 0:
        blockers.append("missing-treated-counties")
    if control_count == 0:
        blockers.append("missing-control-counties")
    if minimum_data_requirements["missing_county_characteristics"] > 0:
        blockers.append("incomplete-county-characteristics")
    if (
        len(difference_rows)
        < minimum_data_requirements[
            "minimum_counties_for_empirical_trigonometric_basis"
        ]
    ):
        blockers.append("n-counties-too-small-for-empirical-trigonometric-basis")
    if any(
        not bool(lane_rank["full_column_rank"])
        for lane_rank in z_lane_basis_rank.values()
    ):
        blockers.append("rank-deficient-z-lane-basis")

    return {
        "difference_rows": difference_rows,
        "design_metadata": {
            "raw_data_asset": raw_data_path.relative_to(root).as_posix(),
            "source_observation_shape": observation_shape,
            "source_transform": source_transform,
            "difference_window": _DIFFERENCE_WINDOW,
            "n_counties": len(difference_rows),
            "treated_counties": treated_count,
            "control_counties": control_count,
            "observed_sample_support": sample_support,
            "z_lanes": list(manifest["z_lanes"]),
            "basis_family": manifest["basis_family"],
            "basis_degree": manifest["basis_degree"],
            "z_lane_basis_rank": z_lane_basis_rank,
            "linear_covariate_count": manifest["linear_covariate_count"],
            "linear_covariate_construction": manifest[
                "linear_covariate_construction"
            ],
            "linear_covariate_count_formula": manifest[
                "linear_covariate_count_formula"
            ],
            "paper_county_characteristics_count": manifest[
                "county_characteristics_count"
            ],
            "observed_county_characteristic_count": observed_characteristic_count,
            "missing_county_characteristic_count": missing_characteristic_count,
            "observed_county_characteristic_names": list(
                observed_characteristic_names
            ),
            "minimum_data_requirements": minimum_data_requirements,
            "estimation_status": "blocked" if blockers else "ready",
            "estimation_blockers": blockers,
        },
    }


def build_section6_hddid_input(
    repo_root: str | Path | None = None,
    *,
    z_lane: str = "median income",
    z0: Any | None = None,
) -> Section6HDDIDInput:
    """Build validated HDDID inputs from a ready Section 6 design preflight.

    The method paper reports a 703-dimensional linear block constructed from
    interactions among 38 county characteristics. Since ``38 choose 2`` equals
    703, this helper builds the linear block from pairwise baseline
    characteristic interactions and refuses to run unless the preflight has the
    full characteristic set and treated/control support.
    """
    try:
        from .inputs import validate_inputs
    except ImportError:
        from hddid.inputs import validate_inputs

    root = _coerce_repo_root(repo_root)
    manifest = _read_canonical_manifest(root)
    design = build_section6_county_difference_design(root)
    metadata = dict(design["design_metadata"])
    blockers = list(metadata["estimation_blockers"])
    if blockers:
        raise ValueError(
            "Section 6 design is not ready for HDDID input construction: "
            + ", ".join(str(blocker) for blocker in blockers)
        )

    lane = str(z_lane)
    lane_names = tuple(str(name) for name in manifest["z_lanes"])
    if lane not in lane_names:
        raise ValueError(
            "z_lane must be one of the Section 6 nonparametric lanes: "
            + ", ".join(lane_names)
        )
    lane_rank = dict(metadata["z_lane_basis_rank"])[lane]
    if not bool(dict(lane_rank)["full_column_rank"]):
        raise ValueError(
            "Section 6 z_lane basis is rank deficient for HDDID input construction: "
            f"{lane} rank {lane_rank['rank']} of {lane_rank['required_rank']}"
        )

    difference_rows = list(design["difference_rows"])
    characteristic_names = tuple(metadata["observed_county_characteristic_names"])
    expected_characteristic_count = int(manifest["county_characteristics_count"])
    if len(characteristic_names) != expected_characteristic_count:
        raise ValueError(
            "Section 6 design must expose the full paper county-characteristic set"
        )

    x_column_names = _interaction_column_names(characteristic_names)
    expected_linear_count = int(manifest["linear_covariate_count"])
    if len(x_column_names) != expected_linear_count:
        raise ValueError(
            "Section 6 interaction design does not match the paper linear-covariate count"
        )

    y_delta = np.asarray(
        [
            _numeric_value(row["delta_unemployment_rate"], "delta_unemployment_rate")
            for row in difference_rows
        ],
        dtype=float,
    )
    z_values = np.asarray(
        [
            _numeric_value(dict(row["baseline_lanes"])[lane], f"baseline_lanes.{lane}")
            for row in difference_rows
        ],
        dtype=float,
    )
    z0_values = np.unique(z_values) if z0 is None else z0
    data = validate_inputs(
        y0=np.zeros_like(y_delta),
        y1=y_delta,
        treat=np.asarray([int(row["treated"]) for row in difference_rows], dtype=int),
        x=_interaction_matrix(difference_rows, characteristic_names),
        z=z_values,
        z0=z0_values,
        basis_family=str(manifest["basis_family"]),
        basis_degree=int(manifest["basis_degree"]),
        alpha=1.0 - float(manifest["confidence_level"]),
    )
    return Section6HDDIDInput(
        data=data,
        z_lane=lane,
        x_column_names=x_column_names,
        outcome_encoding="zero-baseline-delta-outcome",
        design_metadata=metadata,
    )
