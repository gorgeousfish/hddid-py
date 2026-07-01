from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .section6_loader_object_contract import audit_phase7_section6_loader_payload
from .section6_roster import get_section6_sample_roster
from .validation import audit_section6_assets

_SECTION6_MANIFEST_PATHS = (
    "data/section6_manifest.json",
    "hddid-py/data/section6_manifest.json",
)
_PYTHON_SECTION6_LOADER_PATH = "hddid-py/src/hddid/section6_loader.py"
_SECTION6_SAMPLE_ROSTER = get_section6_sample_roster()
_EXPECTED_MANIFEST = {
    "time_window": "2005-2007",
    "treated_state_count": _SECTION6_SAMPLE_ROSTER.treated_state_count,
    "treated_states": list(_SECTION6_SAMPLE_ROSTER.treated_states),
    "control_states": list(_SECTION6_SAMPLE_ROSTER.control_states),
    "sample_rule": _SECTION6_SAMPLE_ROSTER.sample_rule,
    "excluded_states": list(_SECTION6_SAMPLE_ROSTER.excluded_states),
    "z_lanes": ["median income", "population"],
    "linear_covariate_count": 703,
    "county_characteristics_count": 38,
    "linear_covariate_source": "reported 703-covariate block; arithmetic matches pairwise interactions among 38 county-level characteristics",
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
_REQUIRED_MANIFEST_KEYS = tuple(sorted(_EXPECTED_MANIFEST))


def _coerce_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _collect_manifest_assets(repo_root: Path) -> tuple[str, ...]:
    return tuple(
        relative_path
        for relative_path in _SECTION6_MANIFEST_PATHS
        if (repo_root / relative_path).is_file()
    )


def _load_manifest_payload(repo_root: Path, relative_path: str) -> dict[str, Any]:
    payload = json.loads((repo_root / relative_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Section 6 manifest must decode to a JSON object")
    return payload


def _load_manifest_payloads(
    repo_root: Path, relative_paths: tuple[str, ...]
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    payloads: dict[str, dict[str, Any]] = {}
    malformed_assets: list[str] = []
    for relative_path in relative_paths:
        try:
            payloads[relative_path] = _load_manifest_payload(repo_root, relative_path)
        except (json.JSONDecodeError, ValueError):
            malformed_assets.append(relative_path)
    return payloads, tuple(malformed_assets)


def _values_match(expected: Any, observed: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(observed) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return (
            list(observed) == expected if isinstance(observed, list | tuple) else False
        )
    if isinstance(expected, dict):
        return dict(observed) == expected if isinstance(observed, dict) else False
    return observed == expected


def _load_section6_loader_callable(
    repo_root: Path,
) -> tuple[object | None, str | None]:
    loader_path = repo_root / _PYTHON_SECTION6_LOADER_PATH
    if not loader_path.is_file():
        return None, "missing-python-loader-entrypoint"

    spec = importlib.util.spec_from_file_location(
        f"_pyhddid_section6_loader_{abs(hash(loader_path.as_posix()))}",
        loader_path,
    )
    if spec is None or spec.loader is None:
        return None, "loader-import-spec-unavailable"

    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        return None, "loader-import-error"

    loader = getattr(module, "load_section6_data", None)
    if not callable(loader):
        return None, "missing-load-section6-data-callable"
    return loader, None


def _invoke_section6_loader(loader: object, repo_root: Path) -> tuple[object | None, str | None]:
    if not callable(loader):
        return None, "missing-load-section6-data-callable"

    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        signature = None

    try:
        if signature is None:
            return loader(), None
        parameters = tuple(signature.parameters.values())
        if not parameters:
            return loader(), None
        if len(parameters) == 1:
            parameter = parameters[0]
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                return loader(repo_root), None
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                return loader(repo_root=repo_root), None
        if "repo_root" in signature.parameters:
            return loader(repo_root=repo_root), None
    except ValueError as exc:
        return None, f"loader-validation-error:{exc}"
    except Exception:
        return None, "loader-execution-error"

    return None, "unsupported-loader-signature"


@dataclass(slots=True)
class Section6ProvenanceGateAudit:
    repo_root: str
    status: str
    blocker_reason: str | None
    asset_audit_status: str
    asset_audit_blocker_reason: str | None
    missing_requirements: tuple[str, ...]
    manifest_assets: tuple[str, ...]
    required_manifest_keys: tuple[str, ...]
    mismatched_manifest_fields: tuple[str, ...]
    loader_contract_status: str
    loader_contract_blocker_reason: str | None
    loader_contract_missing_outputs: tuple[str, ...]
    loader_contract_analysis_row_shape: str | None
    loader_contract_analysis_row_field_gaps: dict[str, tuple[str, ...]]
    loader_contract_metadata_mismatches: tuple[str, ...]
    loader_contract_observed_county_characteristic_count: int | None
    loader_contract_observed_county_characteristic_names: tuple[str, ...]
    loader_contract_county_characteristic_coverage_status: str
    time_window: str | None
    treated_state_count: int | None
    treated_states: tuple[str, ...]
    control_states: tuple[str, ...]
    sample_rule: dict[str, object]
    excluded_states: tuple[str, ...]
    z_lanes: tuple[str, ...]
    linear_covariate_count: int | None
    county_characteristics_count: int | None
    linear_covariate_source: str | None
    sample_roster_source: str | None
    minimum_wage_timing_cross_check_source: str | None
    outcome_source: str | None
    covariate_source_archive: str | None
    lane_role_map: dict[str, str]
    basis_family: str | None
    basis_degree: int | None
    confidence_level: float | None
    disallowed_shortcuts: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "status": self.status,
            "blocker_reason": self.blocker_reason,
            "asset_audit_status": self.asset_audit_status,
            "asset_audit_blocker_reason": self.asset_audit_blocker_reason,
            "missing_requirements": list(self.missing_requirements),
            "manifest_assets": list(self.manifest_assets),
            "required_manifest_keys": list(self.required_manifest_keys),
            "mismatched_manifest_fields": list(self.mismatched_manifest_fields),
            "loader_contract_status": self.loader_contract_status,
            "loader_contract_blocker_reason": self.loader_contract_blocker_reason,
            "loader_contract_missing_outputs": list(
                self.loader_contract_missing_outputs
            ),
            "loader_contract_analysis_row_shape": self.loader_contract_analysis_row_shape,
            "loader_contract_analysis_row_field_gaps": {
                shape: list(fields)
                for shape, fields in self.loader_contract_analysis_row_field_gaps.items()
            },
            "loader_contract_metadata_mismatches": list(
                self.loader_contract_metadata_mismatches
            ),
            "loader_contract_observed_county_characteristic_count": (
                self.loader_contract_observed_county_characteristic_count
            ),
            "loader_contract_observed_county_characteristic_names": list(
                self.loader_contract_observed_county_characteristic_names
            ),
            "loader_contract_county_characteristic_coverage_status": (
                self.loader_contract_county_characteristic_coverage_status
            ),
            "time_window": self.time_window,
            "treated_state_count": self.treated_state_count,
            "treated_states": list(self.treated_states),
            "control_states": list(self.control_states),
            "sample_rule": dict(self.sample_rule),
            "excluded_states": list(self.excluded_states),
            "z_lanes": list(self.z_lanes),
            "linear_covariate_count": self.linear_covariate_count,
            "county_characteristics_count": self.county_characteristics_count,
            "linear_covariate_source": self.linear_covariate_source,
            "sample_roster_source": self.sample_roster_source,
            "minimum_wage_timing_cross_check_source": (
                self.minimum_wage_timing_cross_check_source
            ),
            "outcome_source": self.outcome_source,
            "covariate_source_archive": self.covariate_source_archive,
            "lane_role_map": dict(self.lane_role_map),
            "basis_family": self.basis_family,
            "basis_degree": self.basis_degree,
            "confidence_level": self.confidence_level,
            "disallowed_shortcuts": list(self.disallowed_shortcuts),
        }


def audit_section6_provenance_gate(
    repo_root: str | Path,
) -> Section6ProvenanceGateAudit:
    root = _coerce_repo_root(repo_root)
    asset_audit = audit_section6_assets(root)
    manifest_assets = _collect_manifest_assets(root)

    manifest_payload: dict[str, Any] = {}
    manifest_payloads: dict[str, dict[str, Any]] = {}
    malformed_manifest_assets: tuple[str, ...] = ()
    if manifest_assets:
        manifest_payloads, malformed_manifest_assets = _load_manifest_payloads(
            root, manifest_assets
        )
        for relative_path in manifest_assets:
            if relative_path in manifest_payloads:
                manifest_payload = manifest_payloads[relative_path]
                break

    missing_requirements = list(asset_audit.missing_assets)
    if asset_audit.status == "ready" and not manifest_assets:
        missing_requirements.append("section6_manifest")

    mismatched_manifest_fields = [
        f"malformed-manifest:{relative_path}"
        for relative_path in malformed_manifest_assets
    ]
    mismatched_manifest_fields.extend(
        key
        for key, expected in _EXPECTED_MANIFEST.items()
        if manifest_payloads
        and any(
            not _values_match(expected, payload.get(key))
            for payload in manifest_payloads.values()
        )
    )

    loader_contract_status = "not-run"
    loader_contract_blocker_reason = None
    loader_contract_missing_outputs: tuple[str, ...] = ()
    loader_contract_analysis_row_shape: str | None = None
    loader_contract_analysis_row_field_gaps: dict[str, tuple[str, ...]] = {}
    loader_contract_metadata_mismatches: tuple[str, ...] = ()
    loader_contract_observed_county_characteristic_count: int | None = None
    loader_contract_observed_county_characteristic_names: tuple[str, ...] = ()
    loader_contract_county_characteristic_coverage_status = "not-run"
    if asset_audit.status == "ready" and manifest_assets and not mismatched_manifest_fields:
        loader_callable, loader_import_error = _load_section6_loader_callable(root)
        if loader_import_error is not None:
            loader_contract_status = "blocked"
            loader_contract_blocker_reason = loader_import_error
        else:
            loader_payload, loader_execution_error = _invoke_section6_loader(
                loader_callable,
                root,
            )
            if loader_execution_error is not None:
                loader_contract_status = "blocked"
                loader_contract_blocker_reason = loader_execution_error
            else:
                loader_contract_audit = audit_phase7_section6_loader_payload(
                    loader_payload
                )
                loader_contract_status = loader_contract_audit.status
                loader_contract_missing_outputs = (
                    loader_contract_audit.missing_outputs
                )
                loader_contract_analysis_row_shape = (
                    loader_contract_audit.analysis_row_shape
                )
                loader_contract_analysis_row_field_gaps = (
                    loader_contract_audit.analysis_row_field_gaps
                )
                loader_contract_metadata_mismatches = (
                    loader_contract_audit.metadata_mismatches
                )
                loader_contract_observed_county_characteristic_count = (
                    loader_contract_audit.observed_county_characteristic_count
                )
                loader_contract_observed_county_characteristic_names = (
                    loader_contract_audit.observed_county_characteristic_names
                )
                loader_contract_county_characteristic_coverage_status = (
                    loader_contract_audit.county_characteristic_coverage_status
                )
                if loader_contract_audit.status != "ready":
                    loader_contract_blocker_reason = (
                        "payload-does-not-satisfy-loader-object-contract"
                    )
        if (
            loader_contract_status == "blocked"
            and "section6_loader_contract" not in missing_requirements
        ):
            missing_requirements.append("section6_loader_contract")

    if asset_audit.status != "ready":
        status = "blocked"
        blocker_reason = asset_audit.blocker_reason
    elif not manifest_assets:
        status = "blocked"
        blocker_reason = "missing-section6-manifest"
    elif mismatched_manifest_fields:
        status = "blocked"
        blocker_reason = "manifest-mismatch"
    elif loader_contract_status != "ready":
        status = "blocked"
        blocker_reason = "loader-contract-mismatch"
    else:
        status = "ready"
        blocker_reason = None

    return Section6ProvenanceGateAudit(
        repo_root=root.as_posix(),
        status=status,
        blocker_reason=blocker_reason,
        asset_audit_status=asset_audit.status,
        asset_audit_blocker_reason=asset_audit.blocker_reason,
        missing_requirements=tuple(missing_requirements),
        manifest_assets=manifest_assets,
        required_manifest_keys=_REQUIRED_MANIFEST_KEYS,
        mismatched_manifest_fields=tuple(sorted(mismatched_manifest_fields)),
        loader_contract_status=loader_contract_status,
        loader_contract_blocker_reason=loader_contract_blocker_reason,
        loader_contract_missing_outputs=loader_contract_missing_outputs,
        loader_contract_analysis_row_shape=loader_contract_analysis_row_shape,
        loader_contract_analysis_row_field_gaps=loader_contract_analysis_row_field_gaps,
        loader_contract_metadata_mismatches=loader_contract_metadata_mismatches,
        loader_contract_observed_county_characteristic_count=(
            loader_contract_observed_county_characteristic_count
        ),
        loader_contract_observed_county_characteristic_names=(
            loader_contract_observed_county_characteristic_names
        ),
        loader_contract_county_characteristic_coverage_status=(
            loader_contract_county_characteristic_coverage_status
        ),
        time_window=manifest_payload.get("time_window"),
        treated_state_count=manifest_payload.get("treated_state_count"),
        treated_states=tuple(manifest_payload.get("treated_states", ())),
        control_states=tuple(manifest_payload.get("control_states", ())),
        sample_rule=dict(manifest_payload.get("sample_rule", {})),
        excluded_states=tuple(manifest_payload.get("excluded_states", ())),
        z_lanes=tuple(manifest_payload.get("z_lanes", ())),
        linear_covariate_count=manifest_payload.get("linear_covariate_count"),
        county_characteristics_count=manifest_payload.get(
            "county_characteristics_count"
        ),
        linear_covariate_source=manifest_payload.get("linear_covariate_source"),
        sample_roster_source=manifest_payload.get("sample_roster_source"),
        minimum_wage_timing_cross_check_source=manifest_payload.get(
            "minimum_wage_timing_cross_check_source"
        ),
        outcome_source=manifest_payload.get("outcome_source"),
        covariate_source_archive=manifest_payload.get("covariate_source_archive"),
        lane_role_map=dict(manifest_payload.get("lane_role_map", {})),
        basis_family=manifest_payload.get("basis_family"),
        basis_degree=manifest_payload.get("basis_degree"),
        confidence_level=manifest_payload.get("confidence_level"),
        disallowed_shortcuts=tuple(manifest_payload.get("disallowed_shortcuts", ())),
    )
