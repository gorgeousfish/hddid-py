from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .section6_roster import get_section6_sample_roster


_TIME_WINDOW = "2005-2007"
_REQUIRED_OUTCOME_YEARS = (2005, 2007)
_ACCEPTED_OBSERVATION_SHAPES = (
    "county-year panel",
    "county-level 2005-to-2007 difference",
)
_REQUIRED_SEMANTIC_FIELDS = (
    "county_id",
    "state",
    "year",
    "unemployment_rate",
    "median income",
    "population",
    "county_characteristics",
)
_REQUIRED_LOADER_OUTPUTS = (
    "analysis_rows",
    "sample_metadata",
    "lane_metadata",
    "linear_covariate_manifest",
)
_SAMPLE_METADATA_FIELDS = (
    "treated_states",
    "control_states",
    "excluded_states",
    "sample_rule",
)
_LANE_ROLE_MAP = {
    "median income": "population",
    "population": "median income",
}
_LINEAR_COVARIATE_COUNT = 703
_COUNTY_CHARACTERISTICS_COUNT = 38
_LINEAR_COVARIATE_SOURCE = "reported 703-covariate block; arithmetic matches pairwise interactions among 38 county-level characteristics"
_BASIS_FAMILY = "trigonometric"
_BASIS_DEGREE = 4
_CONFIDENCE_LEVEL = 0.95
_DISALLOWED_SHORTCUTS = ("synthetic fallback", "figure-only")


@dataclass(slots=True)
class Phase7Section6LoaderObjectContract:
    stage_label: str
    time_window: str
    required_outcome_years: tuple[int, ...]
    sample_construction_baseline: str
    accepted_observation_shapes: tuple[str, ...]
    required_semantic_fields: tuple[str, ...]
    required_loader_outputs: tuple[str, ...]
    sample_metadata_fields: tuple[str, ...]
    lane_role_map: dict[str, str]
    treated_state_count: int
    control_state_count: int
    excluded_states: tuple[str, ...]
    linear_covariate_count: int
    county_characteristics_count: int
    linear_covariate_source: str
    basis_family: str
    basis_degree: int
    confidence_level: float
    disallowed_shortcuts: tuple[str, ...]
    canonical_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.time_window = str(self.time_window).strip()
        self.required_outcome_years = tuple(
            int(year) for year in self.required_outcome_years
        )
        self.sample_construction_baseline = str(
            self.sample_construction_baseline
        ).strip()
        self.accepted_observation_shapes = tuple(
            str(shape).strip() for shape in self.accepted_observation_shapes
        )
        self.required_semantic_fields = tuple(
            str(field).strip() for field in self.required_semantic_fields
        )
        self.required_loader_outputs = tuple(
            str(field).strip() for field in self.required_loader_outputs
        )
        self.sample_metadata_fields = tuple(
            str(field).strip() for field in self.sample_metadata_fields
        )
        self.lane_role_map = {
            str(key).strip(): str(value).strip()
            for key, value in self.lane_role_map.items()
        }
        self.treated_state_count = int(self.treated_state_count)
        self.control_state_count = int(self.control_state_count)
        self.excluded_states = tuple(
            str(state).strip() for state in self.excluded_states
        )
        self.linear_covariate_count = int(self.linear_covariate_count)
        self.county_characteristics_count = int(self.county_characteristics_count)
        self.linear_covariate_source = str(self.linear_covariate_source).strip()
        self.basis_family = str(self.basis_family).strip()
        self.basis_degree = int(self.basis_degree)
        self.confidence_level = float(self.confidence_level)
        self.disallowed_shortcuts = tuple(
            str(shortcut).strip() for shortcut in self.disallowed_shortcuts
        )
        self.canonical_digest = tuple(
            str(line).rstrip() for line in self.canonical_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "time_window": self.time_window,
            "required_outcome_years": list(self.required_outcome_years),
            "sample_construction_baseline": self.sample_construction_baseline,
            "accepted_observation_shapes": list(self.accepted_observation_shapes),
            "required_semantic_fields": list(self.required_semantic_fields),
            "required_loader_outputs": list(self.required_loader_outputs),
            "sample_metadata_fields": list(self.sample_metadata_fields),
            "lane_role_map": dict(self.lane_role_map),
            "treated_state_count": self.treated_state_count,
            "control_state_count": self.control_state_count,
            "excluded_states": list(self.excluded_states),
            "linear_covariate_count": self.linear_covariate_count,
            "county_characteristics_count": self.county_characteristics_count,
            "linear_covariate_source": self.linear_covariate_source,
            "basis_family": self.basis_family,
            "basis_degree": self.basis_degree,
            "confidence_level": self.confidence_level,
            "disallowed_shortcuts": list(self.disallowed_shortcuts),
            "canonical_digest": list(self.canonical_digest),
        }


@dataclass(slots=True)
class Phase7Section6LoaderPayloadAudit:
    status: str
    missing_outputs: tuple[str, ...]
    analysis_row_shape: str | None
    analysis_row_field_gaps: dict[str, tuple[str, ...]]
    metadata_mismatches: tuple[str, ...]
    observed_county_characteristic_count: int | None
    observed_county_characteristic_names: tuple[str, ...]
    county_characteristic_coverage_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "missing_outputs": list(self.missing_outputs),
            "analysis_row_shape": self.analysis_row_shape,
            "analysis_row_field_gaps": {
                shape: list(fields)
                for shape, fields in self.analysis_row_field_gaps.items()
            },
            "metadata_mismatches": list(self.metadata_mismatches),
            "observed_county_characteristic_count": (
                self.observed_county_characteristic_count
            ),
            "observed_county_characteristic_names": list(
                self.observed_county_characteristic_names
            ),
            "county_characteristic_coverage_status": (
                self.county_characteristic_coverage_status
            ),
        }


def _coerce_mapping(payload: object) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    if hasattr(payload, "__dict__"):
        return {
            str(key): value
            for key, value in vars(payload).items()
            if not str(key).startswith("_")
        }
    return {}


def _coerce_sequence(payload: object) -> tuple[Any, ...]:
    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        return tuple(payload)
    return ()


def _values_match(expected: object, observed: object) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(observed) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    if isinstance(expected, tuple):
        return tuple(observed) == expected if isinstance(observed, Sequence) else False
    if isinstance(expected, dict):
        return dict(observed) == expected if isinstance(observed, Mapping) else False
    return observed == expected


def _analysis_row_field_gaps(
    analysis_row_shape: str | None,
    analysis_rows: tuple[Any, ...],
    contract: Phase7Section6LoaderObjectContract,
) -> dict[str, tuple[str, ...]]:
    if not analysis_row_shape or analysis_row_shape not in contract.accepted_observation_shapes:
        return {}
    if not analysis_rows:
        return {analysis_row_shape: tuple(contract.required_semantic_fields)}

    first_row = _coerce_mapping(analysis_rows[0])
    required_fields = list(contract.required_semantic_fields)
    if (
        analysis_row_shape == "county-level 2005-to-2007 difference"
        and "year" not in first_row
        and "difference_window" in first_row
    ):
        required_fields.remove("year")

    missing_fields = tuple(field for field in required_fields if field not in first_row)
    if not missing_fields:
        return {}
    return {analysis_row_shape: missing_fields}


def _analysis_row_metadata_mismatches(
    analysis_row_shape: str | None,
    analysis_rows: tuple[Any, ...],
    contract: Phase7Section6LoaderObjectContract,
) -> tuple[str, ...]:
    if not analysis_row_shape or analysis_row_shape not in contract.accepted_observation_shapes:
        return ()
    if not analysis_rows:
        return ()

    rows = tuple(_coerce_mapping(row) for row in analysis_rows)
    mismatches: list[str] = []

    if analysis_row_shape == "county-year panel":
        observed_years = {row.get("year") for row in rows}
        if not observed_years or any(
            year not in contract.required_outcome_years for year in observed_years
        ):
            mismatches.append("analysis_rows.year_window")
        elif not set(contract.required_outcome_years).issubset(observed_years):
            mismatches.append("analysis_rows.required_outcome_years")
    elif any(
        row.get("difference_window") != contract.time_window
        for row in rows
    ):
        mismatches.append("analysis_rows.difference_window")

    return tuple(mismatches)


def _coerce_optional_integer(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _observed_county_characteristic_names(
    linear_covariate_manifest: dict[str, Any],
) -> tuple[str, ...]:
    observed_names = linear_covariate_manifest.get(
        "observed_county_characteristic_names",
        (),
    )
    if not isinstance(observed_names, Sequence) or isinstance(
        observed_names,
        str | bytes,
    ):
        return ()
    return tuple(str(name) for name in observed_names)


def _coverage_status(
    observed_count: int | None,
    target_count: int,
) -> str:
    if observed_count is None:
        return "not-reported"
    if observed_count < target_count:
        return "partial-local-payload"
    if observed_count == target_count:
        return "paper-target-covered"
    return "exceeds-paper-target"


def audit_phase7_section6_loader_payload(
    payload: object,
) -> Phase7Section6LoaderPayloadAudit:
    contract = run_phase7_section6_loader_object_contract()
    roster = get_section6_sample_roster()
    payload_map = _coerce_mapping(payload)
    missing_outputs = tuple(
        output
        for output in contract.required_loader_outputs
        if output not in payload_map
    )

    sample_metadata = _coerce_mapping(payload_map.get("sample_metadata", {}))
    lane_metadata = _coerce_mapping(payload_map.get("lane_metadata", {}))
    linear_covariate_manifest = _coerce_mapping(
        payload_map.get("linear_covariate_manifest", {})
    )
    analysis_rows = _coerce_sequence(payload_map.get("analysis_rows", ()))
    analysis_row_shape = lane_metadata.get("observation_shape")
    if analysis_row_shape is not None:
        analysis_row_shape = str(analysis_row_shape).strip()
    observed_count = _coerce_optional_integer(
        linear_covariate_manifest.get("observed_county_characteristic_count")
    )
    observed_names = _observed_county_characteristic_names(
        linear_covariate_manifest,
    )

    metadata_mismatches: list[str] = []
    if analysis_row_shape not in contract.accepted_observation_shapes:
        metadata_mismatches.append("lane_metadata.observation_shape")
    if not _values_match(contract.lane_role_map, lane_metadata.get("lane_role_map")):
        metadata_mismatches.append("lane_metadata.lane_role_map")
    if not _values_match(contract.basis_family, lane_metadata.get("basis_family")):
        metadata_mismatches.append("lane_metadata.basis_family")
    if not _values_match(contract.basis_degree, lane_metadata.get("basis_degree")):
        metadata_mismatches.append("lane_metadata.basis_degree")
    if not _values_match(
        contract.confidence_level, lane_metadata.get("confidence_level")
    ):
        metadata_mismatches.append("lane_metadata.confidence_level")
    if not _values_match(
        contract.disallowed_shortcuts, lane_metadata.get("disallowed_shortcuts")
    ):
        metadata_mismatches.append("lane_metadata.disallowed_shortcuts")
    if not _values_match(
        roster.treated_states,
        sample_metadata.get("treated_states"),
    ):
        metadata_mismatches.append("sample_metadata.treated_states")
    if not _values_match(
        roster.control_states,
        sample_metadata.get("control_states"),
    ):
        metadata_mismatches.append("sample_metadata.control_states")
    if not _values_match(contract.excluded_states, sample_metadata.get("excluded_states")):
        metadata_mismatches.append("sample_metadata.excluded_states")
    if not _values_match(
        roster.sample_rule,
        sample_metadata.get("sample_rule"),
    ):
        metadata_mismatches.append("sample_metadata.sample_rule")
    if "linear_covariate_manifest" in payload_map:
        if not _values_match(
            contract.linear_covariate_count,
            linear_covariate_manifest.get("linear_covariate_count"),
        ):
            metadata_mismatches.append(
                "linear_covariate_manifest.linear_covariate_count"
            )
        if not _values_match(
            contract.county_characteristics_count,
            linear_covariate_manifest.get("county_characteristics_count"),
        ):
            metadata_mismatches.append(
                "linear_covariate_manifest.county_characteristics_count"
            )
        if not _values_match(
            contract.linear_covariate_source,
            linear_covariate_manifest.get("linear_covariate_source"),
        ):
            metadata_mismatches.append(
                "linear_covariate_manifest.linear_covariate_source"
            )
        if observed_count is not None and observed_count > contract.county_characteristics_count:
            metadata_mismatches.append(
                "linear_covariate_manifest.observed_county_characteristic_count"
            )
        if observed_count is not None and len(observed_names) != observed_count:
            metadata_mismatches.append(
                "linear_covariate_manifest.observed_county_characteristic_names"
            )

    analysis_row_field_gaps = _analysis_row_field_gaps(
        analysis_row_shape,
        analysis_rows,
        contract,
    )
    metadata_mismatches.extend(
        _analysis_row_metadata_mismatches(
            analysis_row_shape,
            analysis_rows,
            contract,
        )
    )
    status = (
        "ready"
        if not missing_outputs
        and not analysis_row_field_gaps
        and not metadata_mismatches
        else "blocked"
    )
    return Phase7Section6LoaderPayloadAudit(
        status=status,
        missing_outputs=missing_outputs,
        analysis_row_shape=analysis_row_shape,
        analysis_row_field_gaps=analysis_row_field_gaps,
        metadata_mismatches=tuple(sorted(metadata_mismatches)),
        observed_county_characteristic_count=observed_count,
        observed_county_characteristic_names=observed_names,
        county_characteristic_coverage_status=_coverage_status(
            observed_count,
            contract.county_characteristics_count,
        ),
    )


def build_phase7_section6_loader_object_contract() -> (
    Phase7Section6LoaderObjectContract
):
    roster = get_section6_sample_roster()
    canonical_digest = (
        "- accepted analysis rows: `county-year panel` or `county-level 2005-to-2007 difference`, but they must preserve `county_id`, `state`, `year`/difference provenance, and `unemployment_rate` semantics",
        "- sample metadata must expose the source-backed `treated_states`, `control_states`, `excluded_states`, and `sample_rule` for the `2006 federal-binding states` universe",
        "- lane metadata must keep `median income -> population` and `population -> median income`, while the feature manifest keeps `703` linear covariates from `38` county characteristics plus interactions",
        "- method anchors remain `trigonometric` basis degree `4`, `95%` confidence intervals, and disallow `synthetic fallback` / `figure-only` shortcuts",
    )
    return Phase7Section6LoaderObjectContract(
        stage_label="phase7-section6-loader-object-contract",
        time_window=_TIME_WINDOW,
        required_outcome_years=_REQUIRED_OUTCOME_YEARS,
        sample_construction_baseline=roster.baseline_universe,
        accepted_observation_shapes=_ACCEPTED_OBSERVATION_SHAPES,
        required_semantic_fields=_REQUIRED_SEMANTIC_FIELDS,
        required_loader_outputs=_REQUIRED_LOADER_OUTPUTS,
        sample_metadata_fields=_SAMPLE_METADATA_FIELDS,
        lane_role_map=_LANE_ROLE_MAP,
        treated_state_count=roster.treated_state_count,
        control_state_count=len(roster.control_states),
        excluded_states=roster.excluded_states,
        linear_covariate_count=_LINEAR_COVARIATE_COUNT,
        county_characteristics_count=_COUNTY_CHARACTERISTICS_COUNT,
        linear_covariate_source=_LINEAR_COVARIATE_SOURCE,
        basis_family=_BASIS_FAMILY,
        basis_degree=_BASIS_DEGREE,
        confidence_level=_CONFIDENCE_LEVEL,
        disallowed_shortcuts=_DISALLOWED_SHORTCUTS,
        canonical_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_section6_loader_object_contract() -> Phase7Section6LoaderObjectContract:
    return build_phase7_section6_loader_object_contract()
