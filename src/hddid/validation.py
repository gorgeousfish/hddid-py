from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache, wraps
from importlib import import_module
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .basis import build_sieve_basis
from .estimation import (
    Eq31ProjectionRankError,
    EstimationPayload,
    estimate_eq31_mainline,
)
from .inference import (
    InferenceComputationError,
    MissingEvaluationGridError,
    estimate_nonparametric_inference,
    estimate_parametric_inference,
)
from .inputs import ValidatedHDDIDData, validate_inputs
from .nuisance import (
    CrossfitNuisanceEstimator,
    NuisanceTrainingSupportError,
    ZeroValidHoldoutError,
)
from .results import FoldDiagnostics, normalize_oracle_lane
from .score import ScorePayload, build_score_payload
from .splitting import make_crossfit_splits

if TYPE_CHECKING:
    from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard import (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate,
    )

_PAPER_TARGET_SAMPLE_SIZES = (200, 500, 1000)
_PAPER_TARGET_LINEAR_DIMENSIONS = (10, 50, 500, 1000)
_PAPER_NOMINAL_COVERAGE = 0.9
_PAPER_MONTE_CARLO_STAGE = "reduced-smoke"
_DEFAULT_EVALUATION_GRID = np.array([-1.0, 0.0, 1.0], dtype=float)
_PHASE7_RUNTIME_PROBE_EVALUATION_GRID = np.array([0.05, 0.15, 0.25], dtype=float)
_PHASE7_OUTER_INFERENCE_REFERENCE_GRID = np.array([-0.5, 0.75, 1.25], dtype=float)
_SECTION6_ORACLE_LANE = "real-data-asset-audit"
_SECTION6_Z_LANES = ("median income", "population")
_SECTION6_KNOWN_FIGURES = (
    "paper/images/marginaleffectmedinc703.jpg",
    "paper/images/marginaleffectmedincols.jpg",
    "paper/images/marginaleffectpopols.jpg",
    "paper/images/marginaleffectpopu703.jpg",
)
_SECTION6_RAW_DATA_GLOBS = (
    "data/section6*.csv",
    "data/section6*.tsv",
    "data/section6*.xlsx",
    "data/section6*.xls",
    "data/section6*.dta",
    "data/section6*.rds",
    "data/section6*.rda",
    "data/section6*.feather",
    "data/section6*.parquet",
    "data/*county*panel*.*",
    "data/*unemployment*.*",
    "hddid-py/data/section6*.*",
    "hddid-py/data/*county*panel*.*",
    "hddid-py/data/*unemployment*.*",
)
_SECTION6_MANIFEST_ASSET_PATHS = frozenset(
    {
        "data/section6_manifest.json",
        "data/section6_manifest.template.json",
        "hddid-py/data/section6_manifest.json",
        "hddid-py/data/section6_manifest.template.json",
    }
)
_SECTION6_CANONICAL_RAW_ASSET_PATHS = frozenset(
    {
        "data/section6_county_panel.csv",
        "data/section6_county_panel.tsv",
        "data/section6_county_differences.csv",
        "data/section6_county_differences.tsv",
        "hddid-py/data/section6_county_panel.csv",
        "hddid-py/data/section6_county_panel.tsv",
        "hddid-py/data/section6_county_differences.csv",
        "hddid-py/data/section6_county_differences.tsv",
    }
)
_SECTION6_LOADER_GLOBS = (
    "hddid-py/src/hddid/section6_loader.py",
    "hddid-py/src/hddid/empirical_loader.py",
    "hddid-r/R/section6_loader.R",
    "hddid-r/R/load_section6_data.R",
)
_SECTION6_PYTHON_LOADER_ASSET_PATHS = frozenset(
    {
        "hddid-py/src/hddid/section6_loader.py",
        "hddid-py/src/hddid/empirical_loader.py",
    }
)
_SECTION6_DISALLOWED_SHORTCUTS = ("synthetic fallback", "figure-only")
_VALIDATION_LANE_ORDER = (
    "paper-trigonometric",
    "r-parity-polynomial",
    "real-data-asset-audit",
)
_VALIDATION_LANE_ALIASES = {
    "paper": "paper-trigonometric",
    "paper-trigonometric": "paper-trigonometric",
    "paper-trig": "paper-trigonometric",
    "r-parity": "r-parity-polynomial",
    "r-parity-pol": "r-parity-polynomial",
    "r-parity-polynomial": "r-parity-polynomial",
    "real-data-asset-audit": "real-data-asset-audit",
}
_VALID_LANE_STATUSES = frozenset(
    {"blocked", "ready", "sanity-check", "staged-smoke", "validated"}
)
_PHASE7_REFERENCE_ONLY = "reference-only"
_SOURCE_CHECKOUT_ONLY_MODULE_TOKENS = (
    "probe",
    "trigger",
    "phase7",
    "automation_state",
    "release_artifact",
)


class SourceCheckoutRequiredError(RuntimeError):
    """Raised when a repository-local validation helper is called from a wheel."""


def _is_missing_source_checkout_module(module_name: str) -> bool:
    return module_name.startswith(f"{__package__}.") and any(
        token in module_name
        for token in _SOURCE_CHECKOUT_ONLY_MODULE_TOKENS
    )


def _source_checkout_required_error(helper_name: str, exc: ModuleNotFoundError):
    return SourceCheckoutRequiredError(
        f"{helper_name} is a source-checkout validation helper. "
        "It depends on repository-local release-control modules that "
        "are intentionally excluded from clean wheel and sdist "
        "artifacts. Run it from the pyhddid source checkout with "
        "PYTHONPATH=hddid-py/src, or use the installed package for "
        "the core score, estimation, inference, plotting, and "
        "fit_hddid(...) APIs."
    )


def _source_checkout_only_symbol(
    module_name: str,
    symbol_name: str,
    *,
    helper_name: str,
):
    try:
        module = import_module(f"{__package__}.{module_name}")
    except ModuleNotFoundError as exc:
        missing_name = str(exc.name or "")
        if _is_missing_source_checkout_module(missing_name):
            raise _source_checkout_required_error(helper_name, exc) from exc
        raise
    return getattr(module, symbol_name)


def _with_source_checkout_boundary(helper):
    @wraps(helper)
    def wrapped(*args, **kwargs):
        try:
            return helper(*args, **kwargs)
        except ModuleNotFoundError as exc:
            missing_name = str(exc.name or "")
            if _is_missing_source_checkout_module(missing_name):
                raise _source_checkout_required_error(
                    f"{helper.__name__}(...)",
                    exc,
                ) from exc
            raise

    wrapped.__dict__.update(getattr(helper, "__dict__", {}))
    for cache_attr in ("cache_clear", "cache_info", "cache_parameters"):
        if hasattr(helper, cache_attr):
            setattr(wrapped, cache_attr, getattr(helper, cache_attr))
    return wrapped


def run_phase7_monte_carlo_widening_policy_spec():
    from .monte_carlo_widening_policy_spec import (
        run_phase7_monte_carlo_widening_policy_spec as _run_policy_spec,
    )

    return _run_policy_spec()


def _repo_root_or_cwd(repo_root: str | Path | None) -> str | Path:
    if repo_root is None:
        cwd = Path.cwd()
        for candidate in (cwd, *cwd.parents):
            if (
                (candidate / ".planning").is_dir()
                and (candidate / "hddid-py").is_dir()
                and (candidate / "hddid-r").is_dir()
            ):
                return candidate
        return cwd
    return repo_root


def audit_section6_provenance_gate(repo_root: str | Path | None = None):
    from .section6_provenance import (
        audit_section6_provenance_gate as _audit_section6_provenance_gate,
    )

    return _audit_section6_provenance_gate(_repo_root_or_cwd(repo_root))


def run_phase7_monte_carlo_feature_completion_gate(
    repo_root: str | Path | None = None,
):
    _run_feature_completion_gate = _source_checkout_only_symbol(
        "monte_carlo_feature_completion_gate",
        "run_phase7_monte_carlo_feature_completion_gate",
        helper_name="run_phase7_monte_carlo_feature_completion_gate(...)",
    )

    return _run_feature_completion_gate(_repo_root_or_cwd(repo_root))


def _clear_phase7_monte_carlo_feature_completion_gate_cache() -> None:
    return None


run_phase7_monte_carlo_feature_completion_gate.cache_clear = (  # type: ignore[attr-defined]
    _clear_phase7_monte_carlo_feature_completion_gate_cache
)


def run_phase7_monte_carlo_feature_completion_frontier_packet(
    repo_root: str | Path | None = None,
):
    _run_feature_completion_frontier_packet = _source_checkout_only_symbol(
        "monte_carlo_feature_completion_frontier_packet",
        "run_phase7_monte_carlo_feature_completion_frontier_packet",
        helper_name="run_phase7_monte_carlo_feature_completion_frontier_packet(...)",
    )

    return _run_feature_completion_frontier_packet(_repo_root_or_cwd(repo_root))


def run_phase7_release_maturity_gate(
    repo_root: str | Path | None = None,
    *,
    feature_gate=None,
):
    _run_release_maturity_gate = _source_checkout_only_symbol(
        "release_maturity_gate",
        "run_phase7_release_maturity_gate",
        helper_name="run_phase7_release_maturity_gate(...)",
    )

    return _run_release_maturity_gate(
        _repo_root_or_cwd(repo_root),
        feature_gate=feature_gate,
    )


def run_phase7_release_matrix_runner(
    repo_root: str | Path | None = None,
    *,
    feature_gate=None,
):
    _run_release_matrix_runner = _source_checkout_only_symbol(
        "release_matrix_runner",
        "run_phase7_release_matrix_runner",
        helper_name="run_phase7_release_matrix_runner(...)",
    )

    return _run_release_matrix_runner(
        _repo_root_or_cwd(repo_root),
        feature_gate=feature_gate,
    )


def audit_phase7_release_artifact_content(
    repo_root: str | Path | None = None,
    *,
    project_metadata=None,
    package_src: str | Path | None = None,
    package_artifact: str | Path | None = None,
):
    _audit_release_artifact_content = _source_checkout_only_symbol(
        "release_artifact_audit",
        "audit_phase7_release_artifact_content",
        helper_name="audit_phase7_release_artifact_content(...)",
    )

    return _audit_release_artifact_content(
        _repo_root_or_cwd(repo_root),
        project_metadata=project_metadata,
        package_src=package_src,
        package_artifact=package_artifact,
    )


def run_phase7_outer_inference_trigger_gate():
    from .outer_inference_trigger_gate import (
        run_phase7_outer_inference_trigger_gate as _run_outer_inference_trigger_gate,
    )

    return _run_outer_inference_trigger_gate()


def run_phase7_monte_carlo_widening_policy_calibration_debt_snapshot():
    from .monte_carlo_widening_policy_calibration_debt_snapshot import (
        run_phase7_monte_carlo_widening_policy_calibration_debt_snapshot as _run_calibration_debt_snapshot,
    )

    return _run_calibration_debt_snapshot()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot as _run_repair_target_snapshot,
    )

    return _run_repair_target_snapshot()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot as _run_correlation_repair_lane_snapshot,
    )

    return _run_correlation_repair_lane_snapshot()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe as _run_covariance_entry_localization_probe,
    )

    return _run_covariance_entry_localization_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract as _run_execution_contract,
    )

    return _run_execution_contract()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_bundle(
    repo_root: str | Path | None = None,
):
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_execution_bundle import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_bundle as _run_execution_bundle,
    )

    _repo_root_or_cwd(repo_root)
    return _run_execution_bundle()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe as _run_same_seed_before_after_probe,
    )

    return _run_same_seed_before_after_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard(
    *,
    candidate: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
        | Mapping[str, object]
        | None
    ) = None,
):
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard as _run_runtime_witness_candidate_guard,
    )

    return _run_runtime_witness_candidate_guard(candidate=candidate)


def run_phase7_next_milestone_trigger_snapshot(
    repo_root: str | Path | None = None,
):
    _run_trigger_snapshot = _source_checkout_only_symbol(
        "next_milestone_trigger_snapshot",
        "run_phase7_next_milestone_trigger_snapshot",
        helper_name="run_phase7_next_milestone_trigger_snapshot(...)",
    )

    return _run_trigger_snapshot(_repo_root_or_cwd(repo_root))


def run_phase7_monte_carlo_widening_trigger_gate(
    *,
    policy: Phase7MonteCarloWideningPolicy | Mapping[str, object] | None = None,
):
    _run_trigger_gate = _source_checkout_only_symbol(
        "monte_carlo_widening_trigger_gate",
        "run_phase7_monte_carlo_widening_trigger_gate",
        helper_name="run_phase7_monte_carlo_widening_trigger_gate(...)",
    )

    return _run_trigger_gate(policy=policy)


def run_phase7_monte_carlo_widening_policy_candidate_guard(
    *,
    policy: Phase7MonteCarloWideningPolicy | Mapping[str, object] | None = None,
):
    _run_candidate_guard = _source_checkout_only_symbol(
        "monte_carlo_widening_policy_candidate_guard",
        "run_phase7_monte_carlo_widening_policy_candidate_guard",
        helper_name="run_phase7_monte_carlo_widening_policy_candidate_guard(...)",
    )

    return _run_candidate_guard(policy=policy)


def run_phase7_monte_carlo_widening_policy_contract_audit(
    repo_root: str | Path | None = None,
    *,
    policy: Phase7MonteCarloWideningPolicy | Mapping[str, object] | None = None,
):
    _run_contract_audit = _source_checkout_only_symbol(
        "monte_carlo_widening_policy_contract_audit",
        "run_phase7_monte_carlo_widening_policy_contract_audit",
        helper_name="run_phase7_monte_carlo_widening_policy_contract_audit(...)",
    )
    return _run_contract_audit(_repo_root_or_cwd(repo_root), policy=policy)


def run_phase7_monte_carlo_widening_policy_acceptance_preview(
    repo_root: str | Path | None = None,
):
    _run_acceptance_preview = _source_checkout_only_symbol(
        "monte_carlo_widening_policy_acceptance_preview",
        "run_phase7_monte_carlo_widening_policy_acceptance_preview",
        helper_name="run_phase7_monte_carlo_widening_policy_acceptance_preview(...)",
    )
    return _run_acceptance_preview(_repo_root_or_cwd(repo_root))


def run_phase7_monte_carlo_widening_policy_runtime_evidence_packet(
    repo_root: str | Path | None = None,
):
    _run_runtime_evidence_packet = _source_checkout_only_symbol(
        "monte_carlo_widening_policy_runtime_evidence_packet",
        "run_phase7_monte_carlo_widening_policy_runtime_evidence_packet",
        helper_name=(
            "run_phase7_monte_carlo_widening_policy_runtime_evidence_packet(...)"
        ),
    )
    return _run_runtime_evidence_packet(_repo_root_or_cwd(repo_root))


def run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate(
    repo_root: str | Path | None = None,
    *,
    candidate_observation: (
        MonteCarloRuntimeProbeObservation
        | MonteCarloRuntimeProbeReport
        | Mapping[str, object]
        | None
    ) = None,
    runtime_witness_candidate: (
        Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate
        | Mapping[str, object]
        | None
    ) = None,
):
    _run_runtime_evidence_admission_gate = _source_checkout_only_symbol(
        "monte_carlo_widening_policy_runtime_evidence_admission_gate",
        "run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate",
        helper_name=(
            "run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate(...)"
        ),
    )

    if repo_root is None:
        return _run_runtime_evidence_admission_gate(
            candidate_observation=candidate_observation,
            runtime_witness_candidate=runtime_witness_candidate,
        )
    return _run_runtime_evidence_admission_gate(
        _repo_root_or_cwd(repo_root),
        candidate_observation=candidate_observation,
        runtime_witness_candidate=runtime_witness_candidate,
    )


def run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe():
    _run_binding_design_rerun_capacity_probe = _source_checkout_only_symbol(
        "monte_carlo_widening_policy_binding_design_rerun_capacity_probe",
        "run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe",
        helper_name=(
            "run_phase7_monte_carlo_widening_policy_binding_design_rerun_capacity_probe(...)"
        ),
    )

    return _run_binding_design_rerun_capacity_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_binding_design_source_priority_guard():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_binding_design_source_priority_guard import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_binding_design_source_priority_guard as _run_binding_design_source_priority_guard,
    )

    return _run_binding_design_source_priority_guard()


def run_phase7_monte_carlo_widening_policy_floor_witness_quota_probe():
    from .monte_carlo_widening_policy_floor_witness_quota_probe import (
        run_phase7_monte_carlo_widening_policy_floor_witness_quota_probe as _run_floor_witness_quota_probe,
    )

    return _run_floor_witness_quota_probe()


def run_phase7_monte_carlo_widening_policy_guard_probe():
    _run_guard_probe = _source_checkout_only_symbol(
        "monte_carlo_widening_policy_guard_probe",
        "run_phase7_monte_carlo_widening_policy_guard_probe",
        helper_name="run_phase7_monte_carlo_widening_policy_guard_probe(...)",
    )

    return _run_guard_probe()


def run_phase7_monte_carlo_widening_policy_floor_slack_probe():
    _run_floor_slack_probe = _source_checkout_only_symbol(
        "monte_carlo_widening_policy_floor_slack_probe",
        "run_phase7_monte_carlo_widening_policy_floor_slack_probe",
        helper_name="run_phase7_monte_carlo_widening_policy_floor_slack_probe(...)",
    )

    return _run_floor_slack_probe()


def run_phase7_monte_carlo_widening_policy_quality_risk_probe(
    repo_root: str | Path | None = None,
):
    _run_quality_risk_probe = _source_checkout_only_symbol(
        "monte_carlo_widening_policy_quality_risk_probe",
        "run_phase7_monte_carlo_widening_policy_quality_risk_probe",
        helper_name="run_phase7_monte_carlo_widening_policy_quality_risk_probe(...)",
    )

    if repo_root is None:
        return _run_quality_risk_probe()
    return _run_quality_risk_probe(_repo_root_or_cwd(repo_root))


def run_phase7_monte_carlo_widening_policy_seed_dispersion_probe():
    from .monte_carlo_widening_policy_seed_dispersion_probe import (
        run_phase7_monte_carlo_widening_policy_seed_dispersion_probe as _run_seed_dispersion_probe,
    )

    return _run_seed_dispersion_probe()


def run_phase7_monte_carlo_widening_policy_seed_role_split_probe():
    from .monte_carlo_widening_policy_seed_role_split_probe import (
        run_phase7_monte_carlo_widening_policy_seed_role_split_probe as _run_seed_role_split_probe,
    )

    return _run_seed_role_split_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_grid_probe(
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
):
    from .monte_carlo_widening_policy_coverage_anchor_grid_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_grid_probe as _run_coverage_anchor_grid_probe,
    )

    return _run_coverage_anchor_grid_probe(policy=policy)


def run_phase7_monte_carlo_widening_policy_coverage_anchor_window_probe(
    *,
    hotspot_center: float = 0.15,
):
    from .monte_carlo_widening_policy_coverage_anchor_window_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_window_probe as _run_coverage_anchor_window_probe,
    )

    return _run_coverage_anchor_window_probe(hotspot_center=hotspot_center)


def run_phase7_monte_carlo_widening_policy_coverage_anchor_window_symmetry_break_probe():
    from .monte_carlo_widening_policy_coverage_anchor_window_symmetry_break_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_window_symmetry_break_probe as _run_coverage_anchor_window_symmetry_break_probe,
    )

    return _run_coverage_anchor_window_symmetry_break_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_window_shoulder_parity_probe():
    from .monte_carlo_widening_policy_coverage_anchor_window_shoulder_parity_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_window_shoulder_parity_probe as _run_coverage_anchor_window_shoulder_parity_probe,
    )

    return _run_coverage_anchor_window_shoulder_parity_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_probe(
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
):
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_probe as _run_coverage_anchor_shoulder_probe,
    )

    return _run_coverage_anchor_shoulder_probe(policy=policy)


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_boundary_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_boundary_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_boundary_probe as _run_coverage_anchor_shoulder_boundary_probe,
    )

    return _run_coverage_anchor_shoulder_boundary_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_localization_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_localization_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_localization_probe as _run_coverage_anchor_shoulder_localization_probe,
    )

    return _run_coverage_anchor_shoulder_localization_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_companion_reserve_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_companion_reserve_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_companion_reserve_probe as _run_coverage_anchor_shoulder_companion_reserve_probe,
    )

    return _run_coverage_anchor_shoulder_companion_reserve_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_growth_split_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_growth_split_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_growth_split_probe as _run_coverage_anchor_shoulder_growth_split_probe,
    )

    return _run_coverage_anchor_shoulder_growth_split_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_scale_repair_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_scale_repair_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_scale_repair_probe as _run_coverage_anchor_shoulder_scale_repair_probe,
    )

    return _run_coverage_anchor_shoulder_scale_repair_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_acceptance_guard_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_repair_acceptance_guard_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_acceptance_guard_probe as _run_coverage_anchor_shoulder_repair_acceptance_guard_probe,
    )

    return _run_coverage_anchor_shoulder_repair_acceptance_guard_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_share_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_repair_share_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_share_probe as _run_coverage_anchor_shoulder_repair_share_probe,
    )

    return _run_coverage_anchor_shoulder_repair_share_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe as _run_coverage_anchor_shoulder_access_share_probe,
    )

    return _run_coverage_anchor_shoulder_access_share_probe()


def run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe():
    from .monte_carlo_widening_policy_seed_window_covariance_probe import (
        run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe as _run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe,
    )

    return _run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_driver_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_access_driver_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_access_driver_probe as _run_coverage_anchor_shoulder_access_driver_probe,
    )

    return _run_coverage_anchor_shoulder_access_driver_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe as _run_coverage_anchor_shoulder_center_coupling_probe,
    )

    return _run_coverage_anchor_shoulder_center_coupling_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe as _run_coverage_anchor_shoulder_center_coupling_channel_probe,
    )

    return _run_coverage_anchor_shoulder_center_coupling_channel_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe as _run_coverage_anchor_shoulder_center_coupling_repair_probe,
    )

    return _run_coverage_anchor_shoulder_center_coupling_repair_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe as _run_coverage_anchor_shoulder_directionality_probe,
    )

    return _run_coverage_anchor_shoulder_directionality_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe as _run_coverage_anchor_shoulder_correlation_gap_probe,
    )

    return _run_coverage_anchor_shoulder_correlation_gap_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe as _run_coverage_anchor_shoulder_cross_shoulder_sign_probe,
    )

    return _run_coverage_anchor_shoulder_cross_shoulder_sign_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe as _run_coverage_anchor_shoulder_covariance_entry_factor_probe,
    )

    return _run_coverage_anchor_shoulder_covariance_entry_factor_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_probe as _run_coverage_anchor_shoulder_entry_patch_implementation_readiness_probe,
    )

    return _run_coverage_anchor_shoulder_entry_patch_implementation_readiness_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_intake_bundle_probe as _run_coverage_anchor_shoulder_entry_patch_intake_bundle_probe,
    )

    return _run_coverage_anchor_shoulder_entry_patch_intake_bundle_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_plan as _run_coverage_anchor_shoulder_entry_patch_plan,
    )

    return _run_coverage_anchor_shoulder_entry_patch_plan()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_source_bridge_contract as _run_coverage_anchor_shoulder_entry_patch_source_bridge_contract,
    )

    return _run_coverage_anchor_shoulder_entry_patch_source_bridge_contract()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path as _run_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path,
    )

    return (
        _run_coverage_anchor_shoulder_entry_patch_implementation_readiness_fast_path()
    )


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_alignment_probe as _run_live_source_target_alignment_probe,
    )

    return _run_live_source_target_alignment_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_reground_contract(
    repo_root: str | Path | None = None,
):
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_reground_contract import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_live_source_target_reground_contract as _run_live_source_target_reground_contract,
    )

    _repo_root_or_cwd(repo_root)
    return _run_live_source_target_reground_contract()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_left_support_null_action_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_left_support_null_action_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_left_support_null_action_probe as _run_coverage_anchor_shoulder_entry_patch_left_support_null_action_probe,
    )

    return _run_coverage_anchor_shoulder_entry_patch_left_support_null_action_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_probe as _run_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_probe,
    )

    return _run_coverage_anchor_shoulder_entry_patch_matrix_norm_budget_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe as _run_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe,
    )

    return _run_coverage_anchor_shoulder_entry_patch_spectral_admissibility_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_promotion_guard_probe():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_promotion_guard_probe import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_promotion_guard_probe as _run_coverage_anchor_shoulder_entry_patch_promotion_guard_probe,
    )

    return _run_coverage_anchor_shoulder_entry_patch_promotion_guard_probe()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract as _run_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract,
    )

    return _run_coverage_anchor_shoulder_entry_patch_estimator_object_flow_contract()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract as _run_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract,
    )

    return (
        _run_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
    )


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate as _run_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate,
    )

    return _run_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract as _run_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract,
    )

    return (
        _run_coverage_anchor_shoulder_entry_patch_fresh_estimator_evidence_intake_contract()
    )


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_rerun_spend_order():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_rerun_spend_order import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_rerun_spend_order as _run_coverage_anchor_shoulder_first_sine_runtime_witness_rerun_spend_order,
    )

    return _run_coverage_anchor_shoulder_first_sine_runtime_witness_rerun_spend_order()


def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract as _run_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract,
    )

    return _run_coverage_anchor_shoulder_fresh_estimator_rerun_intake_contract()


def _coerce_evaluation_grid(values: np.ndarray | Sequence[float]) -> np.ndarray:
    grid = np.asarray(values, dtype=float)
    if grid.ndim == 0:
        return grid.reshape(1)
    if grid.ndim == 1:
        return grid
    if grid.ndim == 2 and grid.shape[1] == 1:
        return grid[:, 0]
    raise ValueError("evaluation_grid must be one-dimensional")


def _paper_truth_vector(scale: float, p: int, *, active_terms: int) -> np.ndarray:
    coefficients = np.zeros(int(p), dtype=float)
    limit = min(int(p), int(active_terms))
    coefficients[:limit] = np.array(
        [scale / float(index + 1) for index in range(limit)],
        dtype=float,
    )
    return coefficients


def _paper_true_beta(p: int) -> np.ndarray:
    treated = _paper_truth_vector(2.0, p, active_terms=15)
    control = _paper_truth_vector(1.0, p, active_terms=15)
    return treated - control


def _paper_beta_treated(p: int) -> np.ndarray:
    return _paper_truth_vector(2.0, p, active_terms=15)


def _paper_beta_control(p: int) -> np.ndarray:
    return _paper_truth_vector(1.0, p, active_terms=15)


def _paper_theta0(p: int) -> np.ndarray:
    return _paper_truth_vector(1.0, p, active_terms=10)


def _paper_target_matrix() -> dict[str, tuple[int, ...] | float]:
    return {
        "n": _PAPER_TARGET_SAMPLE_SIZES,
        "p": _PAPER_TARGET_LINEAR_DIMENSIONS,
        "nominal_coverage": _PAPER_NOMINAL_COVERAGE,
    }


def _toeplitz_covariance(p: int, rho_x: float) -> np.ndarray:
    index = np.arange(int(p), dtype=int)
    return float(rho_x) ** np.abs(np.subtract.outer(index, index))


def _expit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _metric_summary(
    errors: list[np.ndarray],
    standard_errors: list[np.ndarray],
    coverage: list[np.ndarray],
    interval_lengths: list[np.ndarray],
) -> dict[str, float | None]:
    if not errors:
        return {
            "bias": None,
            "rmse": None,
            "average_standard_error": None,
            "coverage": None,
            "interval_length": None,
        }

    error_values = np.concatenate(
        [np.ravel(np.asarray(value, dtype=float)) for value in errors]
    )
    standard_error_values = np.concatenate(
        [np.ravel(np.asarray(value, dtype=float)) for value in standard_errors]
    )
    coverage_values = np.concatenate(
        [np.ravel(np.asarray(value, dtype=bool)) for value in coverage]
    )
    interval_length_values = np.concatenate(
        [np.ravel(np.asarray(value, dtype=float)) for value in interval_lengths]
    )
    return {
        "bias": float(np.mean(error_values)),
        "rmse": float(np.sqrt(np.mean(error_values**2))),
        "average_standard_error": float(np.mean(standard_error_values)),
        "coverage": float(np.mean(coverage_values)),
        "interval_length": float(np.mean(interval_length_values)),
    }


def _coerce_design_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, not boolean")
    if not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _coerce_design_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        raise ValueError(f"{name} must be numeric, not string")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(slots=True)
class MonteCarloDesign:
    dgp_name: str
    n_obs: int
    p: int
    basis_family: str = "trigonometric"
    basis_degree: int = 8
    oracle_lane: str = "paper-trigonometric"
    alpha: float = 0.1
    rho_x: float = 0.5
    n_folds: int = 3
    trim_lower: float = 0.01
    trim_upper: float = 0.99
    evaluation_grid: np.ndarray = field(
        default_factory=lambda: _DEFAULT_EVALUATION_GRID.copy()
    )

    def __post_init__(self) -> None:
        dgp = str(self.dgp_name).strip().upper()
        if dgp not in {"DGP1", "DGP2"}:
            raise ValueError("dgp_name must be one of {'DGP1', 'DGP2'}")
        self.dgp_name = dgp
        self.n_obs = _coerce_design_integer("n_obs", self.n_obs)
        self.p = _coerce_design_integer("p", self.p)
        self.basis_family = str(self.basis_family).strip().lower()
        self.basis_degree = _coerce_design_integer(
            "basis_degree",
            self.basis_degree,
        )
        self.alpha = _coerce_design_float("alpha", self.alpha)
        self.rho_x = _coerce_design_float("rho_x", self.rho_x)
        self.n_folds = _coerce_design_integer("n_folds", self.n_folds)
        self.trim_lower = _coerce_design_float("trim_lower", self.trim_lower)
        self.trim_upper = _coerce_design_float("trim_upper", self.trim_upper)
        self.evaluation_grid = _coerce_evaluation_grid(self.evaluation_grid)
        self.oracle_lane = normalize_oracle_lane(
            basis_family=self.basis_family,
            oracle_lane=self.oracle_lane,
        )
        if self.n_obs < 2:
            raise ValueError("n_obs must be at least 2")
        if self.p < 1:
            raise ValueError("p must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if self.n_folds < 2 or self.n_folds > self.n_obs:
            raise ValueError("n_folds must be between 2 and n_obs")
        if not np.isfinite(self.rho_x) or not -1.0 < self.rho_x < 1.0:
            raise ValueError("rho_x must lie strictly between -1 and 1")
        if not 0.0 <= self.trim_lower < self.trim_upper <= 1.0:
            raise ValueError(
                "trim bounds must satisfy 0 <= trim_lower < trim_upper <= 1"
            )

    def staged_subset_entry(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "n_obs": self.n_obs,
            "p": self.p,
            "basis_family": self.basis_family,
            "basis_degree": self.basis_degree,
            "oracle_lane": self.oracle_lane,
        }


@dataclass(slots=True)
class SimulationDataset:
    y0: np.ndarray
    y1: np.ndarray
    treat: np.ndarray
    x: np.ndarray
    z: np.ndarray
    z0: np.ndarray
    basis_family: str
    basis_degree: int
    alpha: float
    oracle_lane: str
    dgp_name: str
    true_beta: np.ndarray
    true_f_at_z0: np.ndarray
    propensity: np.ndarray
    x_covariance: np.ndarray

    def __post_init__(self) -> None:
        self.y0 = np.asarray(self.y0, dtype=float)
        self.y1 = np.asarray(self.y1, dtype=float)
        self.treat = np.asarray(self.treat, dtype=int)
        self.x = np.asarray(self.x, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        self.z0 = _coerce_evaluation_grid(self.z0)
        self.basis_family = str(self.basis_family).strip().lower()
        self.basis_degree = int(self.basis_degree)
        self.alpha = float(self.alpha)
        self.oracle_lane = normalize_oracle_lane(
            basis_family=self.basis_family,
            oracle_lane=self.oracle_lane,
        )
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.true_beta = np.asarray(self.true_beta, dtype=float)
        self.true_f_at_z0 = np.asarray(self.true_f_at_z0, dtype=float)
        self.propensity = np.asarray(self.propensity, dtype=float)
        self.x_covariance = np.asarray(self.x_covariance, dtype=float)

    def to_validated_data(self) -> ValidatedHDDIDData:
        return validate_inputs(
            y0=self.y0,
            y1=self.y1,
            treat=self.treat,
            x=self.x,
            z=self.z,
            z0=self.z0,
            basis_family=self.basis_family,
            basis_degree=self.basis_degree,
            alpha=self.alpha,
        )


@dataclass(slots=True)
class MonteCarloSmokeSummary:
    design: MonteCarloDesign
    staging: str
    full_target_matrix: dict[str, tuple[int, ...] | float]
    n_replications: int
    n_successful_replications: int
    typed_invalidity_counts: dict[str, int]
    trimming_rate: float | None
    zero_valid_fold_frequency: float
    parametric_metrics: dict[str, float | None]
    nonparametric_metrics: dict[str, float | None]
    nonparametric_mean_absolute_error: float | None = None
    nonparametric_uniform_critical_value: float | None = None
    nonparametric_uniform_band_length: float | None = None
    runtime_seconds: float | None = None
    typed_invalidity_examples: dict[str, dict[str, object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )
        self.typed_invalidity_examples = _normalize_invalidity_examples(
            self.typed_invalidity_examples
        )
        self.nonparametric_mean_absolute_error = (
            None
            if self.nonparametric_mean_absolute_error is None
            else float(self.nonparametric_mean_absolute_error)
        )
        self.nonparametric_uniform_critical_value = (
            None
            if self.nonparametric_uniform_critical_value is None
            else float(self.nonparametric_uniform_critical_value)
        )
        self.nonparametric_uniform_band_length = (
            None
            if self.nonparametric_uniform_band_length is None
            else float(self.nonparametric_uniform_band_length)
        )
        self.runtime_seconds = (
            None if self.runtime_seconds is None else float(self.runtime_seconds)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.design.dgp_name,
            "n_obs": self.design.n_obs,
            "p": self.design.p,
            "basis_family": self.design.basis_family,
            "basis_degree": self.design.basis_degree,
            "oracle_lane": self.design.oracle_lane,
            "staging": self.staging,
            "full_target_matrix": dict(self.full_target_matrix),
            "staged_subset_entry": self.design.staged_subset_entry(),
            "n_replications": self.n_replications,
            "n_successful_replications": self.n_successful_replications,
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "typed_invalidity_examples": _serialize_invalidity_examples(
                self.typed_invalidity_examples
            ),
            "trimming_rate": self.trimming_rate,
            "zero_valid_fold_frequency": self.zero_valid_fold_frequency,
            "parametric_metrics": dict(self.parametric_metrics),
            "nonparametric_metrics": dict(self.nonparametric_metrics),
            "nonparametric_mean_absolute_error": self.nonparametric_mean_absolute_error,
            "nonparametric_uniform_critical_value": (
                self.nonparametric_uniform_critical_value
            ),
            "nonparametric_uniform_band_length": self.nonparametric_uniform_band_length,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass(slots=True)
class MonteCarloSmokeReport:
    stage_label: str
    staging: bool
    full_target_matrix: dict[str, tuple[int, ...] | float]
    nominal_coverage: float
    staged_subset: tuple[dict[str, object], ...]
    summaries: tuple[MonteCarloSmokeSummary, ...]
    total_runtime_seconds: float | None = None


@dataclass(slots=True)
class MonteCarloWideningDesignEvidence:
    dgp_name: str
    n_obs: int
    p: int
    n_replications: int
    n_successful_replications: int
    success_rate: float
    runtime_seconds: float | None
    parametric_metrics: dict[str, float | None] = field(default_factory=dict)
    nonparametric_metrics: dict[str, float | None] = field(default_factory=dict)
    typed_invalidity_counts: dict[str, int] = field(default_factory=dict)
    trimming_rate: float | None = None
    zero_valid_fold_frequency: float = 0.0
    typed_invalidity_examples: dict[str, dict[str, object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.n_obs = _coerce_runtime_positive_integer("n_obs", self.n_obs)
        self.p = _coerce_runtime_positive_integer("p", self.p)
        self.n_replications = _coerce_runtime_positive_integer(
            "n_replications",
            self.n_replications,
        )
        self.n_successful_replications = _coerce_runtime_nonnegative_integer(
            "n_successful_replications",
            self.n_successful_replications,
        )
        if self.n_successful_replications > self.n_replications:
            raise ValueError(
                "n_successful_replications must be no greater than n_replications"
            )
        self.success_rate = _coerce_runtime_required_probability(
            "success_rate",
            self.success_rate,
        )
        expected_success_rate = self.n_successful_replications / self.n_replications
        if not np.isclose(
            self.success_rate,
            expected_success_rate,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "success_rate must equal n_successful_replications / n_replications"
            )
        self.runtime_seconds = _coerce_runtime_nonnegative_float(
            "runtime_seconds",
            self.runtime_seconds,
        )
        self.parametric_metrics = _coerce_runtime_metric_payload(
            "parametric_metrics",
            self.parametric_metrics,
        )
        self.nonparametric_metrics = _coerce_runtime_metric_payload(
            "nonparametric_metrics",
            self.nonparametric_metrics,
        )
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )
        self.trimming_rate = _coerce_runtime_probability(
            "trimming_rate",
            self.trimming_rate,
        )
        self.zero_valid_fold_frequency = _coerce_runtime_required_probability(
            "zero_valid_fold_frequency",
            self.zero_valid_fold_frequency,
        )
        self.typed_invalidity_examples = _normalize_invalidity_examples(
            self.typed_invalidity_examples
        )
        if self.n_successful_replications > 0:
            self._require_successful_metric_payload(
                "parametric_metrics",
                self.parametric_metrics,
            )
            self._require_successful_metric_payload(
                "nonparametric_metrics",
                self.nonparametric_metrics,
            )

    @staticmethod
    def _require_successful_metric_payload(
        name: str,
        metrics: Mapping[str, float | None],
    ) -> None:
        required_metrics = (
            "bias",
            "rmse",
            "average_standard_error",
            "coverage",
            "interval_length",
        )
        missing_metrics = [
            metric_name
            for metric_name in required_metrics
            if metrics.get(metric_name) is None
        ]
        if missing_metrics:
            raise ValueError(
                "successful Monte Carlo widening design evidence must carry "
                f"{name}.{', '.join(missing_metrics)}"
            )
        if metrics["average_standard_error"] <= 0.0:
            raise ValueError(
                "successful Monte Carlo widening design evidence must carry "
                f"positive {name}.average_standard_error"
            )
        if metrics["interval_length"] <= 0.0:
            raise ValueError(
                "successful Monte Carlo widening design evidence must carry "
                f"positive {name}.interval_length"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "n_obs": self.n_obs,
            "p": self.p,
            "n_replications": self.n_replications,
            "n_successful_replications": self.n_successful_replications,
            "success_rate": self.success_rate,
            "runtime_seconds": self.runtime_seconds,
            "parametric_metrics": dict(self.parametric_metrics),
            "nonparametric_metrics": dict(self.nonparametric_metrics),
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "trimming_rate": self.trimming_rate,
            "zero_valid_fold_frequency": self.zero_valid_fold_frequency,
            "typed_invalidity_examples": _serialize_invalidity_examples(
                self.typed_invalidity_examples
            ),
        }


@dataclass(slots=True)
class MonteCarloWideningReadinessReport:
    oracle_lane: str
    stage_label: str
    staged_matrix: bool
    full_target_matrix: dict[str, tuple[int, ...] | float]
    total_runtime_seconds: float | None
    total_replications: int
    total_successful_replications: int
    success_rate: float
    typed_invalidity_counts: dict[str, int]
    promotion_gate: str
    remaining_requirements: tuple[str, ...]
    recommended_next_step: str
    design_evidence: tuple[MonteCarloWideningDesignEvidence, ...]
    typed_invalidity_examples: dict[str, dict[str, object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.staged_matrix = _coerce_runtime_boolean(
            "staged_matrix",
            self.staged_matrix,
        )
        self.full_target_matrix = dict(self.full_target_matrix)
        self.total_runtime_seconds = _coerce_runtime_nonnegative_float(
            "total_runtime_seconds",
            self.total_runtime_seconds,
        )
        self.total_replications = _coerce_runtime_positive_integer(
            "total_replications",
            self.total_replications,
        )
        self.total_successful_replications = _coerce_runtime_nonnegative_integer(
            "total_successful_replications",
            self.total_successful_replications,
        )
        if self.total_successful_replications > self.total_replications:
            raise ValueError(
                "total_successful_replications must be no greater than total_replications"
            )
        self.success_rate = _coerce_runtime_required_probability(
            "success_rate",
            self.success_rate,
        )
        expected_success_rate = (
            self.total_successful_replications / self.total_replications
        )
        if not np.isclose(
            self.success_rate,
            expected_success_rate,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "success_rate must equal total_successful_replications / total_replications"
            )
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )
        self.promotion_gate = str(self.promotion_gate).strip()
        self.remaining_requirements = _normalize_string_tuple(
            self.remaining_requirements
        )
        self.recommended_next_step = str(self.recommended_next_step).strip()
        self.design_evidence = tuple(self.design_evidence)
        self.typed_invalidity_examples = _normalize_invalidity_examples(
            self.typed_invalidity_examples
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "staged_matrix": self.staged_matrix,
            "full_target_matrix": _serialize_target_matrix(self.full_target_matrix),
            "total_runtime_seconds": self.total_runtime_seconds,
            "total_replications": self.total_replications,
            "total_successful_replications": self.total_successful_replications,
            "success_rate": self.success_rate,
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "promotion_gate": self.promotion_gate,
            "remaining_requirements": list(self.remaining_requirements),
            "recommended_next_step": self.recommended_next_step,
            "typed_invalidity_examples": _serialize_invalidity_examples(
                self.typed_invalidity_examples
            ),
            "design_evidence": [
                design_evidence.to_dict() for design_evidence in self.design_evidence
            ],
        }


@dataclass(slots=True)
class MonteCarloRuntimeProbeObservation:
    dgp_name: str
    n_obs: int
    p: int
    random_state: int
    runtime_seconds: float | None
    success: bool
    typed_invalidity_counts: dict[str, int] = field(default_factory=dict)
    typed_invalidity_examples: dict[str, dict[str, object]] = field(
        default_factory=dict
    )
    trimming_rate: float | None = None
    zero_valid_fold_frequency: float = 0.0
    parametric_bias: float | None = None
    nonparametric_bias: float | None = None
    parametric_rmse: float | None = None
    nonparametric_rmse: float | None = None
    parametric_average_standard_error: float | None = None
    parametric_coverage: float | None = None
    parametric_interval_length: float | None = None
    nonparametric_average_standard_error: float | None = None
    nonparametric_coverage: float | None = None
    nonparametric_interval_length: float | None = None
    nonparametric_absolute_error: float | None = None
    nonparametric_uniform_critical_value: float | None = None
    nonparametric_uniform_band_length: float | None = None

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.n_obs = _coerce_runtime_positive_integer("n_obs", self.n_obs)
        self.p = _coerce_runtime_positive_integer("p", self.p)
        self.random_state = _coerce_runtime_nonnegative_integer(
            "random_state",
            self.random_state,
        )
        self.runtime_seconds = _coerce_runtime_nonnegative_float(
            "runtime_seconds",
            self.runtime_seconds,
        )
        self.success = _coerce_runtime_boolean("success", self.success)
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )
        self.typed_invalidity_examples = _normalize_invalidity_examples(
            self.typed_invalidity_examples
        )
        self.trimming_rate = _coerce_runtime_probability(
            "trimming_rate",
            self.trimming_rate,
        )
        self.zero_valid_fold_frequency = _coerce_runtime_probability(
            "zero_valid_fold_frequency",
            self.zero_valid_fold_frequency,
        )
        if self.zero_valid_fold_frequency is None:
            raise ValueError("zero_valid_fold_frequency must not be None")
        self.parametric_bias = _coerce_runtime_finite_float(
            "parametric_bias",
            self.parametric_bias,
        )
        self.nonparametric_bias = _coerce_runtime_finite_float(
            "nonparametric_bias",
            self.nonparametric_bias,
        )
        self.parametric_rmse = _coerce_runtime_nonnegative_float(
            "parametric_rmse",
            self.parametric_rmse,
        )
        self.nonparametric_rmse = _coerce_runtime_nonnegative_float(
            "nonparametric_rmse",
            self.nonparametric_rmse,
        )
        self.parametric_average_standard_error = _coerce_runtime_nonnegative_float(
            "parametric_average_standard_error",
            self.parametric_average_standard_error,
        )
        self.parametric_coverage = _coerce_runtime_probability(
            "parametric_coverage",
            self.parametric_coverage,
        )
        self.parametric_interval_length = _coerce_runtime_nonnegative_float(
            "parametric_interval_length",
            self.parametric_interval_length,
        )
        self.nonparametric_average_standard_error = (
            _coerce_runtime_nonnegative_float(
                "nonparametric_average_standard_error",
                self.nonparametric_average_standard_error,
            )
        )
        self.nonparametric_coverage = _coerce_runtime_probability(
            "nonparametric_coverage",
            self.nonparametric_coverage,
        )
        self.nonparametric_interval_length = _coerce_runtime_nonnegative_float(
            "nonparametric_interval_length",
            self.nonparametric_interval_length,
        )
        self.nonparametric_absolute_error = _coerce_runtime_nonnegative_float(
            "nonparametric_absolute_error",
            self.nonparametric_absolute_error,
        )
        self.nonparametric_uniform_critical_value = (
            _coerce_runtime_nonnegative_float(
                "nonparametric_uniform_critical_value",
                self.nonparametric_uniform_critical_value,
            )
        )
        self.nonparametric_uniform_band_length = (
            _coerce_runtime_nonnegative_float(
                "nonparametric_uniform_band_length",
                self.nonparametric_uniform_band_length,
            )
        )
        if self.success:
            required_success_fields = (
                "parametric_bias",
                "parametric_rmse",
                "parametric_average_standard_error",
                "parametric_coverage",
                "parametric_interval_length",
                "nonparametric_bias",
                "nonparametric_rmse",
                "nonparametric_average_standard_error",
                "nonparametric_coverage",
                "nonparametric_interval_length",
            )
            missing_fields = [
                field_name
                for field_name in required_success_fields
                if getattr(self, field_name) is None
            ]
            if missing_fields:
                raise ValueError(
                    "successful runtime observations must carry "
                    f"{', '.join(missing_fields)}"
                )
            if self.parametric_average_standard_error <= 0.0:
                raise ValueError(
                    "successful runtime observations must carry positive "
                    "parametric_average_standard_error"
                )
            if self.parametric_interval_length <= 0.0:
                raise ValueError(
                    "successful runtime observations must carry positive "
                    "parametric_interval_length"
                )
            if self.nonparametric_average_standard_error <= 0.0:
                raise ValueError(
                    "successful runtime observations must carry positive "
                    "nonparametric_average_standard_error"
                )
            if self.nonparametric_interval_length <= 0.0:
                raise ValueError(
                    "successful runtime observations must carry positive "
                    "nonparametric_interval_length"
                )
        else:
            success_metric_fields = (
                "parametric_bias",
                "parametric_rmse",
                "nonparametric_bias",
                "nonparametric_rmse",
                "parametric_average_standard_error",
                "parametric_coverage",
                "parametric_interval_length",
                "nonparametric_average_standard_error",
                "nonparametric_coverage",
                "nonparametric_interval_length",
                "nonparametric_absolute_error",
                "nonparametric_uniform_critical_value",
                "nonparametric_uniform_band_length",
            )
            present_fields = [
                field_name
                for field_name in success_metric_fields
                if getattr(self, field_name) is not None
            ]
            if present_fields:
                raise ValueError(
                    "unsuccessful runtime observations must not carry successful "
                    f"metric evidence: {', '.join(present_fields)}"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "n_obs": self.n_obs,
            "p": self.p,
            "random_state": self.random_state,
            "runtime_seconds": self.runtime_seconds,
            "success": self.success,
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "typed_invalidity_examples": _serialize_invalidity_examples(
                self.typed_invalidity_examples
            ),
            "trimming_rate": self.trimming_rate,
            "zero_valid_fold_frequency": self.zero_valid_fold_frequency,
            "parametric_bias": self.parametric_bias,
            "nonparametric_bias": self.nonparametric_bias,
            "parametric_rmse": self.parametric_rmse,
            "nonparametric_rmse": self.nonparametric_rmse,
            "parametric_average_standard_error": self.parametric_average_standard_error,
            "parametric_coverage": self.parametric_coverage,
            "parametric_interval_length": self.parametric_interval_length,
            "nonparametric_average_standard_error": (
                self.nonparametric_average_standard_error
            ),
            "nonparametric_coverage": self.nonparametric_coverage,
            "nonparametric_interval_length": self.nonparametric_interval_length,
            "nonparametric_absolute_error": self.nonparametric_absolute_error,
            "nonparametric_uniform_critical_value": (
                self.nonparametric_uniform_critical_value
            ),
            "nonparametric_uniform_band_length": self.nonparametric_uniform_band_length,
        }


def _coerce_runtime_positive_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, not boolean")
    if not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _coerce_runtime_nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer, not boolean")
    if not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _coerce_runtime_boolean(name: str, value: bool) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _coerce_runtime_finite_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        raise ValueError(f"{name} must be numeric, not string")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _coerce_runtime_nonnegative_float(name: str, value: float | None) -> float | None:
    number = _coerce_runtime_finite_float(name, value)
    if number is not None and number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _coerce_runtime_probability(name: str, value: float | None) -> float | None:
    number = _coerce_runtime_finite_float(name, value)
    if number is not None and not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _coerce_runtime_required_probability(name: str, value: float | None) -> float:
    number = _coerce_runtime_probability(name, value)
    if number is None:
        raise ValueError(f"{name} must not be None")
    return number


def _coerce_runtime_metric_payload(
    name: str,
    metrics: Mapping[str, float | None],
) -> dict[str, float | None]:
    normalized = dict(metrics)
    for key, value in tuple(normalized.items()):
        field_name = f"{name}.{key}"
        if key in {"coverage"}:
            normalized[key] = _coerce_runtime_probability(field_name, value)
        elif key in {
            "rmse",
            "average_standard_error",
            "interval_length",
        }:
            normalized[key] = _coerce_runtime_nonnegative_float(field_name, value)
        elif key in {"bias"}:
            normalized[key] = _coerce_runtime_finite_float(field_name, value)
        else:
            normalized[key] = _coerce_runtime_finite_float(field_name, value)
    return normalized


@dataclass(slots=True)
class MonteCarloRuntimeProbeDesignSummary:
    dgp_name: str
    n_obs: int
    p: int
    n_runs: int
    n_successful_runs: int
    success_rate: float
    runtime_mean_seconds: float | None
    runtime_std_seconds: float | None
    typed_invalidity_counts: dict[str, int] = field(default_factory=dict)
    typed_invalidity_examples: dict[str, dict[str, object]] = field(
        default_factory=dict
    )
    mean_parametric_rmse: float | None = None
    mean_nonparametric_rmse: float | None = None
    mean_parametric_average_standard_error: float | None = None
    std_parametric_average_standard_error: float | None = None
    mean_parametric_coverage: float | None = None
    std_parametric_coverage: float | None = None
    mean_parametric_interval_length: float | None = None
    std_parametric_interval_length: float | None = None
    mean_nonparametric_average_standard_error: float | None = None
    std_nonparametric_average_standard_error: float | None = None
    mean_nonparametric_coverage: float | None = None
    std_nonparametric_coverage: float | None = None
    mean_nonparametric_interval_length: float | None = None
    std_nonparametric_interval_length: float | None = None
    mean_nonparametric_absolute_error: float | None = None
    std_nonparametric_absolute_error: float | None = None
    mean_nonparametric_uniform_critical_value: float | None = None
    std_nonparametric_uniform_critical_value: float | None = None
    mean_nonparametric_uniform_band_length: float | None = None
    std_nonparametric_uniform_band_length: float | None = None
    mean_trimming_rate: float | None = None
    mean_zero_valid_fold_frequency: float = 0.0
    mean_parametric_bias: float | None = None
    std_parametric_bias: float | None = None
    mean_nonparametric_bias: float | None = None
    std_nonparametric_bias: float | None = None

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.n_obs = _coerce_runtime_positive_integer("n_obs", self.n_obs)
        self.p = _coerce_runtime_positive_integer("p", self.p)
        self.n_runs = _coerce_runtime_positive_integer("n_runs", self.n_runs)
        self.n_successful_runs = _coerce_runtime_nonnegative_integer(
            "n_successful_runs",
            self.n_successful_runs,
        )
        if self.n_successful_runs > self.n_runs:
            raise ValueError("n_successful_runs must be no greater than n_runs")
        self.success_rate = _coerce_runtime_probability(
            "success_rate",
            self.success_rate,
        )
        expected_success_rate = self.n_successful_runs / self.n_runs
        if not np.isclose(
            self.success_rate,
            expected_success_rate,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("success_rate must equal n_successful_runs / n_runs")
        self.runtime_mean_seconds = _coerce_runtime_nonnegative_float(
            "runtime_mean_seconds",
            self.runtime_mean_seconds,
        )
        self.runtime_std_seconds = _coerce_runtime_nonnegative_float(
            "runtime_std_seconds",
            self.runtime_std_seconds,
        )
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )
        self.typed_invalidity_examples = _normalize_invalidity_examples(
            self.typed_invalidity_examples
        )
        self.mean_parametric_rmse = _coerce_runtime_nonnegative_float(
            "mean_parametric_rmse",
            self.mean_parametric_rmse,
        )
        self.mean_nonparametric_rmse = _coerce_runtime_nonnegative_float(
            "mean_nonparametric_rmse",
            self.mean_nonparametric_rmse,
        )
        self.mean_parametric_average_standard_error = _coerce_runtime_nonnegative_float(
            "mean_parametric_average_standard_error",
            self.mean_parametric_average_standard_error,
        )
        self.std_parametric_average_standard_error = _coerce_runtime_nonnegative_float(
            "std_parametric_average_standard_error",
            self.std_parametric_average_standard_error,
        )
        self.mean_parametric_coverage = _coerce_runtime_probability(
            "mean_parametric_coverage",
            self.mean_parametric_coverage,
        )
        self.std_parametric_coverage = _coerce_runtime_nonnegative_float(
            "std_parametric_coverage",
            self.std_parametric_coverage,
        )
        self.mean_parametric_interval_length = _coerce_runtime_nonnegative_float(
            "mean_parametric_interval_length",
            self.mean_parametric_interval_length,
        )
        self.std_parametric_interval_length = _coerce_runtime_nonnegative_float(
            "std_parametric_interval_length",
            self.std_parametric_interval_length,
        )
        self.mean_nonparametric_average_standard_error = (
            _coerce_runtime_nonnegative_float(
                "mean_nonparametric_average_standard_error",
                self.mean_nonparametric_average_standard_error,
            )
        )
        self.std_nonparametric_average_standard_error = (
            _coerce_runtime_nonnegative_float(
                "std_nonparametric_average_standard_error",
                self.std_nonparametric_average_standard_error,
            )
        )
        self.mean_nonparametric_coverage = _coerce_runtime_probability(
            "mean_nonparametric_coverage",
            self.mean_nonparametric_coverage,
        )
        self.std_nonparametric_coverage = _coerce_runtime_nonnegative_float(
            "std_nonparametric_coverage",
            self.std_nonparametric_coverage,
        )
        self.mean_nonparametric_interval_length = _coerce_runtime_nonnegative_float(
            "mean_nonparametric_interval_length",
            self.mean_nonparametric_interval_length,
        )
        self.std_nonparametric_interval_length = _coerce_runtime_nonnegative_float(
            "std_nonparametric_interval_length",
            self.std_nonparametric_interval_length,
        )
        self.mean_nonparametric_absolute_error = _coerce_runtime_nonnegative_float(
            "mean_nonparametric_absolute_error",
            self.mean_nonparametric_absolute_error,
        )
        self.std_nonparametric_absolute_error = _coerce_runtime_nonnegative_float(
            "std_nonparametric_absolute_error",
            self.std_nonparametric_absolute_error,
        )
        self.mean_nonparametric_uniform_critical_value = (
            _coerce_runtime_nonnegative_float(
                "mean_nonparametric_uniform_critical_value",
                self.mean_nonparametric_uniform_critical_value,
            )
        )
        self.std_nonparametric_uniform_critical_value = (
            _coerce_runtime_nonnegative_float(
                "std_nonparametric_uniform_critical_value",
                self.std_nonparametric_uniform_critical_value,
            )
        )
        self.mean_nonparametric_uniform_band_length = _coerce_runtime_nonnegative_float(
            "mean_nonparametric_uniform_band_length",
            self.mean_nonparametric_uniform_band_length,
        )
        self.std_nonparametric_uniform_band_length = _coerce_runtime_nonnegative_float(
            "std_nonparametric_uniform_band_length",
            self.std_nonparametric_uniform_band_length,
        )
        self.mean_trimming_rate = _coerce_runtime_probability(
            "mean_trimming_rate",
            self.mean_trimming_rate,
        )
        self.mean_zero_valid_fold_frequency = _coerce_runtime_probability(
            "mean_zero_valid_fold_frequency",
            self.mean_zero_valid_fold_frequency,
        )
        self.mean_parametric_bias = _coerce_runtime_finite_float(
            "mean_parametric_bias",
            self.mean_parametric_bias,
        )
        self.std_parametric_bias = _coerce_runtime_nonnegative_float(
            "std_parametric_bias",
            self.std_parametric_bias,
        )
        self.mean_nonparametric_bias = _coerce_runtime_finite_float(
            "mean_nonparametric_bias",
            self.mean_nonparametric_bias,
        )
        self.std_nonparametric_bias = _coerce_runtime_nonnegative_float(
            "std_nonparametric_bias",
            self.std_nonparametric_bias,
        )
        if self.n_successful_runs > 0:
            required_success_means = (
                "mean_parametric_bias",
                "mean_parametric_rmse",
                "mean_parametric_average_standard_error",
                "mean_parametric_coverage",
                "mean_parametric_interval_length",
                "mean_nonparametric_bias",
                "mean_nonparametric_rmse",
                "mean_nonparametric_average_standard_error",
                "mean_nonparametric_coverage",
                "mean_nonparametric_interval_length",
            )
            missing_means = [
                field_name
                for field_name in required_success_means
                if getattr(self, field_name) is None
            ]
            if missing_means:
                raise ValueError(
                    "successful runtime design summaries must carry "
                    f"{', '.join(missing_means)}"
                )
            if self.mean_parametric_average_standard_error <= 0.0:
                raise ValueError(
                    "successful runtime design summaries must carry positive "
                    "mean_parametric_average_standard_error"
                )
            if self.mean_parametric_interval_length <= 0.0:
                raise ValueError(
                    "successful runtime design summaries must carry positive "
                    "mean_parametric_interval_length"
                )
            if self.mean_nonparametric_average_standard_error <= 0.0:
                raise ValueError(
                    "successful runtime design summaries must carry positive "
                    "mean_nonparametric_average_standard_error"
                )
            if self.mean_nonparametric_interval_length <= 0.0:
                raise ValueError(
                    "successful runtime design summaries must carry positive "
                    "mean_nonparametric_interval_length"
                )
        else:
            successful_mean_fields = (
                "mean_parametric_bias",
                "std_parametric_bias",
                "mean_parametric_rmse",
                "mean_nonparametric_bias",
                "std_nonparametric_bias",
                "mean_nonparametric_rmse",
                "mean_parametric_average_standard_error",
                "std_parametric_average_standard_error",
                "mean_parametric_coverage",
                "std_parametric_coverage",
                "mean_parametric_interval_length",
                "std_parametric_interval_length",
                "mean_nonparametric_average_standard_error",
                "std_nonparametric_average_standard_error",
                "mean_nonparametric_coverage",
                "std_nonparametric_coverage",
                "mean_nonparametric_interval_length",
                "std_nonparametric_interval_length",
                "mean_nonparametric_absolute_error",
                "std_nonparametric_absolute_error",
                "mean_nonparametric_uniform_critical_value",
                "std_nonparametric_uniform_critical_value",
                "mean_nonparametric_uniform_band_length",
                "std_nonparametric_uniform_band_length",
                "mean_trimming_rate",
            )
            present_fields = [
                field_name
                for field_name in successful_mean_fields
                if getattr(self, field_name) is not None
            ]
            if present_fields:
                raise ValueError(
                    "runtime design summaries with zero successful runs must not "
                    f"carry successful metric evidence: {', '.join(present_fields)}"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "n_obs": self.n_obs,
            "p": self.p,
            "n_runs": self.n_runs,
            "n_successful_runs": self.n_successful_runs,
            "success_rate": self.success_rate,
            "runtime_mean_seconds": self.runtime_mean_seconds,
            "runtime_std_seconds": self.runtime_std_seconds,
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "typed_invalidity_examples": _serialize_invalidity_examples(
                self.typed_invalidity_examples
            ),
            "mean_parametric_rmse": self.mean_parametric_rmse,
            "mean_nonparametric_rmse": self.mean_nonparametric_rmse,
            "mean_parametric_average_standard_error": (
                self.mean_parametric_average_standard_error
            ),
            "std_parametric_average_standard_error": (
                self.std_parametric_average_standard_error
            ),
            "mean_parametric_coverage": self.mean_parametric_coverage,
            "std_parametric_coverage": self.std_parametric_coverage,
            "mean_parametric_interval_length": self.mean_parametric_interval_length,
            "std_parametric_interval_length": self.std_parametric_interval_length,
            "mean_nonparametric_average_standard_error": (
                self.mean_nonparametric_average_standard_error
            ),
            "std_nonparametric_average_standard_error": (
                self.std_nonparametric_average_standard_error
            ),
            "mean_nonparametric_coverage": self.mean_nonparametric_coverage,
            "std_nonparametric_coverage": self.std_nonparametric_coverage,
            "mean_nonparametric_interval_length": (
                self.mean_nonparametric_interval_length
            ),
            "std_nonparametric_interval_length": self.std_nonparametric_interval_length,
            "mean_nonparametric_absolute_error": (
                self.mean_nonparametric_absolute_error
            ),
            "std_nonparametric_absolute_error": self.std_nonparametric_absolute_error,
            "mean_nonparametric_uniform_critical_value": (
                self.mean_nonparametric_uniform_critical_value
            ),
            "std_nonparametric_uniform_critical_value": (
                self.std_nonparametric_uniform_critical_value
            ),
            "mean_nonparametric_uniform_band_length": (
                self.mean_nonparametric_uniform_band_length
            ),
            "std_nonparametric_uniform_band_length": (
                self.std_nonparametric_uniform_band_length
            ),
            "mean_trimming_rate": self.mean_trimming_rate,
            "mean_zero_valid_fold_frequency": self.mean_zero_valid_fold_frequency,
            "mean_parametric_bias": self.mean_parametric_bias,
            "std_parametric_bias": self.std_parametric_bias,
            "mean_nonparametric_bias": self.mean_nonparametric_bias,
            "std_nonparametric_bias": self.std_nonparametric_bias,
        }


@dataclass(slots=True)
class MonteCarloRuntimeProbeReport:
    oracle_lane: str
    stage_label: str
    random_states: tuple[int, ...]
    total_runtime_seconds: float | None
    observations: tuple[MonteCarloRuntimeProbeObservation, ...]
    design_summaries: tuple[MonteCarloRuntimeProbeDesignSummary, ...]
    typed_invalidity_counts: dict[str, int] = field(default_factory=dict)
    typed_invalidity_examples: dict[str, dict[str, object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.random_states = tuple(int(value) for value in self.random_states)
        self.total_runtime_seconds = (
            None
            if self.total_runtime_seconds is None
            else float(self.total_runtime_seconds)
        )
        self.observations = tuple(self.observations)
        self.design_summaries = tuple(self.design_summaries)
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )
        self.typed_invalidity_examples = _normalize_invalidity_examples(
            self.typed_invalidity_examples
        )

    def design_summary(
        self, dgp_name: str, n_obs: int, p: int
    ) -> MonteCarloRuntimeProbeDesignSummary:
        target = (str(dgp_name).strip().upper(), int(n_obs), int(p))
        for summary in self.design_summaries:
            if (summary.dgp_name, summary.n_obs, summary.p) == target:
                return summary
        raise KeyError(f"runtime probe design not present: {target!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "random_states": list(self.random_states),
            "total_runtime_seconds": self.total_runtime_seconds,
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "typed_invalidity_examples": _serialize_invalidity_examples(
                self.typed_invalidity_examples
            ),
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "design_summaries": [
                design_summary.to_dict() for design_summary in self.design_summaries
            ],
        }


@dataclass(slots=True)
class Phase7NonparametricCalibrationDecomposition:
    dgp_name: str
    p: int
    target_n_obs: int
    reference_n_obs: int
    nominal_coverage: float
    target_mean_nonparametric_rmse: float | None = None
    reference_mean_nonparametric_rmse: float | None = None
    rmse_inflation_factor: float | None = None
    target_mean_nonparametric_absolute_error: float | None = None
    reference_mean_nonparametric_absolute_error: float | None = None
    absolute_error_inflation_factor: float | None = None
    target_mean_nonparametric_average_standard_error: float | None = None
    reference_mean_nonparametric_average_standard_error: float | None = None
    standard_error_inflation_factor: float | None = None
    target_mean_nonparametric_interval_length: float | None = None
    reference_mean_nonparametric_interval_length: float | None = None
    interval_length_inflation_factor: float | None = None
    target_mean_nonparametric_uniform_critical_value: float | None = None
    reference_mean_nonparametric_uniform_critical_value: float | None = None
    uniform_critical_value_inflation_factor: float | None = None
    target_mean_nonparametric_uniform_band_length: float | None = None
    reference_mean_nonparametric_uniform_band_length: float | None = None
    uniform_band_length_inflation_factor: float | None = None
    target_mean_nonparametric_coverage: float | None = None
    reference_mean_nonparametric_coverage: float | None = None
    target_coverage_gap: float | None = None
    reference_coverage_gap: float | None = None
    target_worst_coverage_random_state: int | None = None
    target_worst_coverage: float | None = None
    reference_worst_coverage_random_state: int | None = None
    reference_worst_coverage: float | None = None
    target_longest_interval_random_state: int | None = None
    target_longest_interval_length: float | None = None
    reference_longest_interval_random_state: int | None = None
    reference_longest_interval_length: float | None = None
    target_largest_uniform_critical_random_state: int | None = None
    target_largest_uniform_critical_value: float | None = None
    reference_largest_uniform_critical_random_state: int | None = None
    reference_largest_uniform_critical_value: float | None = None
    target_longest_uniform_band_random_state: int | None = None
    target_longest_uniform_band_length: float | None = None
    reference_longest_uniform_band_random_state: int | None = None
    reference_longest_uniform_band_length: float | None = None

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.p = int(self.p)
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.nominal_coverage = float(self.nominal_coverage)
        self.target_mean_nonparametric_rmse = (
            None
            if self.target_mean_nonparametric_rmse is None
            else float(self.target_mean_nonparametric_rmse)
        )
        self.reference_mean_nonparametric_rmse = (
            None
            if self.reference_mean_nonparametric_rmse is None
            else float(self.reference_mean_nonparametric_rmse)
        )
        self.rmse_inflation_factor = (
            None
            if self.rmse_inflation_factor is None
            else float(self.rmse_inflation_factor)
        )
        self.target_mean_nonparametric_absolute_error = (
            None
            if self.target_mean_nonparametric_absolute_error is None
            else float(self.target_mean_nonparametric_absolute_error)
        )
        self.reference_mean_nonparametric_absolute_error = (
            None
            if self.reference_mean_nonparametric_absolute_error is None
            else float(self.reference_mean_nonparametric_absolute_error)
        )
        self.absolute_error_inflation_factor = (
            None
            if self.absolute_error_inflation_factor is None
            else float(self.absolute_error_inflation_factor)
        )
        self.target_mean_nonparametric_average_standard_error = (
            None
            if self.target_mean_nonparametric_average_standard_error is None
            else float(self.target_mean_nonparametric_average_standard_error)
        )
        self.reference_mean_nonparametric_average_standard_error = (
            None
            if self.reference_mean_nonparametric_average_standard_error is None
            else float(self.reference_mean_nonparametric_average_standard_error)
        )
        self.standard_error_inflation_factor = (
            None
            if self.standard_error_inflation_factor is None
            else float(self.standard_error_inflation_factor)
        )
        self.target_mean_nonparametric_interval_length = (
            None
            if self.target_mean_nonparametric_interval_length is None
            else float(self.target_mean_nonparametric_interval_length)
        )
        self.reference_mean_nonparametric_interval_length = (
            None
            if self.reference_mean_nonparametric_interval_length is None
            else float(self.reference_mean_nonparametric_interval_length)
        )
        self.interval_length_inflation_factor = (
            None
            if self.interval_length_inflation_factor is None
            else float(self.interval_length_inflation_factor)
        )
        self.target_mean_nonparametric_uniform_critical_value = (
            None
            if self.target_mean_nonparametric_uniform_critical_value is None
            else float(self.target_mean_nonparametric_uniform_critical_value)
        )
        self.reference_mean_nonparametric_uniform_critical_value = (
            None
            if self.reference_mean_nonparametric_uniform_critical_value is None
            else float(self.reference_mean_nonparametric_uniform_critical_value)
        )
        self.uniform_critical_value_inflation_factor = (
            None
            if self.uniform_critical_value_inflation_factor is None
            else float(self.uniform_critical_value_inflation_factor)
        )
        self.target_mean_nonparametric_uniform_band_length = (
            None
            if self.target_mean_nonparametric_uniform_band_length is None
            else float(self.target_mean_nonparametric_uniform_band_length)
        )
        self.reference_mean_nonparametric_uniform_band_length = (
            None
            if self.reference_mean_nonparametric_uniform_band_length is None
            else float(self.reference_mean_nonparametric_uniform_band_length)
        )
        self.uniform_band_length_inflation_factor = (
            None
            if self.uniform_band_length_inflation_factor is None
            else float(self.uniform_band_length_inflation_factor)
        )
        self.target_mean_nonparametric_coverage = (
            None
            if self.target_mean_nonparametric_coverage is None
            else float(self.target_mean_nonparametric_coverage)
        )
        self.reference_mean_nonparametric_coverage = (
            None
            if self.reference_mean_nonparametric_coverage is None
            else float(self.reference_mean_nonparametric_coverage)
        )
        self.target_coverage_gap = (
            None
            if self.target_coverage_gap is None
            else float(self.target_coverage_gap)
        )
        self.reference_coverage_gap = (
            None
            if self.reference_coverage_gap is None
            else float(self.reference_coverage_gap)
        )
        self.target_worst_coverage_random_state = (
            None
            if self.target_worst_coverage_random_state is None
            else int(self.target_worst_coverage_random_state)
        )
        self.target_worst_coverage = (
            None
            if self.target_worst_coverage is None
            else float(self.target_worst_coverage)
        )
        self.reference_worst_coverage_random_state = (
            None
            if self.reference_worst_coverage_random_state is None
            else int(self.reference_worst_coverage_random_state)
        )
        self.reference_worst_coverage = (
            None
            if self.reference_worst_coverage is None
            else float(self.reference_worst_coverage)
        )
        self.target_longest_interval_random_state = (
            None
            if self.target_longest_interval_random_state is None
            else int(self.target_longest_interval_random_state)
        )
        self.target_longest_interval_length = (
            None
            if self.target_longest_interval_length is None
            else float(self.target_longest_interval_length)
        )
        self.reference_longest_interval_random_state = (
            None
            if self.reference_longest_interval_random_state is None
            else int(self.reference_longest_interval_random_state)
        )
        self.reference_longest_interval_length = (
            None
            if self.reference_longest_interval_length is None
            else float(self.reference_longest_interval_length)
        )
        self.target_largest_uniform_critical_random_state = (
            None
            if self.target_largest_uniform_critical_random_state is None
            else int(self.target_largest_uniform_critical_random_state)
        )
        self.target_largest_uniform_critical_value = (
            None
            if self.target_largest_uniform_critical_value is None
            else float(self.target_largest_uniform_critical_value)
        )
        self.reference_largest_uniform_critical_random_state = (
            None
            if self.reference_largest_uniform_critical_random_state is None
            else int(self.reference_largest_uniform_critical_random_state)
        )
        self.reference_largest_uniform_critical_value = (
            None
            if self.reference_largest_uniform_critical_value is None
            else float(self.reference_largest_uniform_critical_value)
        )
        self.target_longest_uniform_band_random_state = (
            None
            if self.target_longest_uniform_band_random_state is None
            else int(self.target_longest_uniform_band_random_state)
        )
        self.target_longest_uniform_band_length = (
            None
            if self.target_longest_uniform_band_length is None
            else float(self.target_longest_uniform_band_length)
        )
        self.reference_longest_uniform_band_random_state = (
            None
            if self.reference_longest_uniform_band_random_state is None
            else int(self.reference_longest_uniform_band_random_state)
        )
        self.reference_longest_uniform_band_length = (
            None
            if self.reference_longest_uniform_band_length is None
            else float(self.reference_longest_uniform_band_length)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "p": self.p,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "nominal_coverage": self.nominal_coverage,
            "target_mean_nonparametric_rmse": self.target_mean_nonparametric_rmse,
            "reference_mean_nonparametric_rmse": self.reference_mean_nonparametric_rmse,
            "rmse_inflation_factor": self.rmse_inflation_factor,
            "target_mean_nonparametric_absolute_error": (
                self.target_mean_nonparametric_absolute_error
            ),
            "reference_mean_nonparametric_absolute_error": (
                self.reference_mean_nonparametric_absolute_error
            ),
            "absolute_error_inflation_factor": self.absolute_error_inflation_factor,
            "target_mean_nonparametric_average_standard_error": (
                self.target_mean_nonparametric_average_standard_error
            ),
            "reference_mean_nonparametric_average_standard_error": (
                self.reference_mean_nonparametric_average_standard_error
            ),
            "standard_error_inflation_factor": self.standard_error_inflation_factor,
            "target_mean_nonparametric_interval_length": (
                self.target_mean_nonparametric_interval_length
            ),
            "reference_mean_nonparametric_interval_length": (
                self.reference_mean_nonparametric_interval_length
            ),
            "interval_length_inflation_factor": self.interval_length_inflation_factor,
            "target_mean_nonparametric_uniform_critical_value": (
                self.target_mean_nonparametric_uniform_critical_value
            ),
            "reference_mean_nonparametric_uniform_critical_value": (
                self.reference_mean_nonparametric_uniform_critical_value
            ),
            "uniform_critical_value_inflation_factor": (
                self.uniform_critical_value_inflation_factor
            ),
            "target_mean_nonparametric_uniform_band_length": (
                self.target_mean_nonparametric_uniform_band_length
            ),
            "reference_mean_nonparametric_uniform_band_length": (
                self.reference_mean_nonparametric_uniform_band_length
            ),
            "uniform_band_length_inflation_factor": (
                self.uniform_band_length_inflation_factor
            ),
            "target_mean_nonparametric_coverage": self.target_mean_nonparametric_coverage,
            "reference_mean_nonparametric_coverage": (
                self.reference_mean_nonparametric_coverage
            ),
            "target_coverage_gap": self.target_coverage_gap,
            "reference_coverage_gap": self.reference_coverage_gap,
            "target_worst_coverage_random_state": self.target_worst_coverage_random_state,
            "target_worst_coverage": self.target_worst_coverage,
            "reference_worst_coverage_random_state": (
                self.reference_worst_coverage_random_state
            ),
            "reference_worst_coverage": self.reference_worst_coverage,
            "target_longest_interval_random_state": (
                self.target_longest_interval_random_state
            ),
            "target_longest_interval_length": self.target_longest_interval_length,
            "reference_longest_interval_random_state": (
                self.reference_longest_interval_random_state
            ),
            "reference_longest_interval_length": self.reference_longest_interval_length,
            "target_largest_uniform_critical_random_state": (
                self.target_largest_uniform_critical_random_state
            ),
            "target_largest_uniform_critical_value": (
                self.target_largest_uniform_critical_value
            ),
            "reference_largest_uniform_critical_random_state": (
                self.reference_largest_uniform_critical_random_state
            ),
            "reference_largest_uniform_critical_value": (
                self.reference_largest_uniform_critical_value
            ),
            "target_longest_uniform_band_random_state": (
                self.target_longest_uniform_band_random_state
            ),
            "target_longest_uniform_band_length": (
                self.target_longest_uniform_band_length
            ),
            "reference_longest_uniform_band_random_state": (
                self.reference_longest_uniform_band_random_state
            ),
            "reference_longest_uniform_band_length": (
                self.reference_longest_uniform_band_length
            ),
        }


@dataclass(slots=True)
class Phase7NonparametricCalibrationReport:
    oracle_lane: str
    stage_label: str
    random_states: tuple[int, ...]
    nominal_coverage: float
    target_n_obs: int
    reference_n_obs: int
    p: int
    decompositions: tuple[Phase7NonparametricCalibrationDecomposition, ...]

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.random_states = tuple(int(value) for value in self.random_states)
        self.nominal_coverage = float(self.nominal_coverage)
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.decompositions = tuple(self.decompositions)

    def decomposition(
        self, dgp_name: str
    ) -> Phase7NonparametricCalibrationDecomposition:
        target_name = str(dgp_name).strip().upper()
        for decomposition in self.decompositions:
            if decomposition.dgp_name == target_name:
                return decomposition
        raise KeyError(
            f"nonparametric calibration decomposition not present: {target_name!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "random_states": list(self.random_states),
            "nominal_coverage": self.nominal_coverage,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "decompositions": [
                decomposition.to_dict() for decomposition in self.decompositions
            ],
        }


@dataclass(slots=True)
class Phase7NonparametricCalibrationObjectSlice:
    dgp_name: str
    n_obs: int
    p: int
    random_state: int
    replication_seed: int
    evaluation_grid: np.ndarray
    true_f_at_z0: np.ndarray
    bar_f_at_z0: np.ndarray
    absolute_error_at_z0: np.ndarray
    sigma_z_hat: np.ndarray
    pointwise_interval_length: np.ndarray
    pointwise_coverage: np.ndarray
    mean_absolute_error: float
    max_absolute_error_grid_value: float
    max_absolute_error: float
    mean_sigma_z_hat: float
    max_sigma_z_hat_grid_value: float
    max_sigma_z_hat: float
    mean_pointwise_interval_length: float
    max_pointwise_interval_grid_value: float
    max_pointwise_interval_length: float
    uniform_critical_value: float | None = None
    mean_uniform_band_length: float | None = None

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.n_obs = int(self.n_obs)
        self.p = int(self.p)
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.evaluation_grid = _coerce_evaluation_grid(self.evaluation_grid)
        self.true_f_at_z0 = _coerce_evaluation_grid(self.true_f_at_z0)
        self.bar_f_at_z0 = _coerce_evaluation_grid(self.bar_f_at_z0)
        self.absolute_error_at_z0 = _coerce_evaluation_grid(self.absolute_error_at_z0)
        self.sigma_z_hat = _coerce_evaluation_grid(self.sigma_z_hat)
        self.pointwise_interval_length = _coerce_evaluation_grid(
            self.pointwise_interval_length
        )
        self.pointwise_coverage = np.asarray(self.pointwise_coverage, dtype=bool)
        if self.pointwise_coverage.ndim != 1:
            raise ValueError("pointwise_coverage must be one-dimensional")
        self.mean_absolute_error = float(self.mean_absolute_error)
        self.max_absolute_error_grid_value = float(self.max_absolute_error_grid_value)
        self.max_absolute_error = float(self.max_absolute_error)
        self.mean_sigma_z_hat = float(self.mean_sigma_z_hat)
        self.max_sigma_z_hat_grid_value = float(self.max_sigma_z_hat_grid_value)
        self.max_sigma_z_hat = float(self.max_sigma_z_hat)
        self.mean_pointwise_interval_length = float(self.mean_pointwise_interval_length)
        self.max_pointwise_interval_grid_value = float(
            self.max_pointwise_interval_grid_value
        )
        self.max_pointwise_interval_length = float(self.max_pointwise_interval_length)
        self.uniform_critical_value = (
            None
            if self.uniform_critical_value is None
            else float(self.uniform_critical_value)
        )
        self.mean_uniform_band_length = (
            None
            if self.mean_uniform_band_length is None
            else float(self.mean_uniform_band_length)
        )
        vector_size = self.evaluation_grid.shape[0]
        for name, value in (
            ("true_f_at_z0", self.true_f_at_z0),
            ("bar_f_at_z0", self.bar_f_at_z0),
            ("absolute_error_at_z0", self.absolute_error_at_z0),
            ("sigma_z_hat", self.sigma_z_hat),
            ("pointwise_interval_length", self.pointwise_interval_length),
            ("pointwise_coverage", self.pointwise_coverage),
        ):
            if value.shape[0] != vector_size:
                raise ValueError(f"{name} must align with evaluation_grid")

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "n_obs": self.n_obs,
            "p": self.p,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "evaluation_grid": self.evaluation_grid.astype(float).tolist(),
            "true_f_at_z0": self.true_f_at_z0.astype(float).tolist(),
            "bar_f_at_z0": self.bar_f_at_z0.astype(float).tolist(),
            "absolute_error_at_z0": self.absolute_error_at_z0.astype(float).tolist(),
            "sigma_z_hat": self.sigma_z_hat.astype(float).tolist(),
            "pointwise_interval_length": self.pointwise_interval_length.astype(
                float
            ).tolist(),
            "pointwise_coverage": self.pointwise_coverage.astype(bool).tolist(),
            "mean_absolute_error": self.mean_absolute_error,
            "max_absolute_error_grid_value": self.max_absolute_error_grid_value,
            "max_absolute_error": self.max_absolute_error,
            "mean_sigma_z_hat": self.mean_sigma_z_hat,
            "max_sigma_z_hat_grid_value": self.max_sigma_z_hat_grid_value,
            "max_sigma_z_hat": self.max_sigma_z_hat,
            "mean_pointwise_interval_length": self.mean_pointwise_interval_length,
            "max_pointwise_interval_grid_value": self.max_pointwise_interval_grid_value,
            "max_pointwise_interval_length": self.max_pointwise_interval_length,
            "uniform_critical_value": self.uniform_critical_value,
            "mean_uniform_band_length": self.mean_uniform_band_length,
        }


@dataclass(slots=True)
class Phase7NonparametricCalibrationObjectDecomposition:
    dgp_name: str
    target_worst_coverage_slice: Phase7NonparametricCalibrationObjectSlice | None = None
    target_longest_interval_slice: Phase7NonparametricCalibrationObjectSlice | None = (
        None
    )
    target_worst_coverage_source: str | None = None
    target_longest_interval_source: str | None = None

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.target_worst_coverage_source = (
            None
            if self.target_worst_coverage_source is None
            else str(self.target_worst_coverage_source).strip()
        )
        self.target_longest_interval_source = (
            None
            if self.target_longest_interval_source is None
            else str(self.target_longest_interval_source).strip()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "target_worst_coverage_slice": (
                None
                if self.target_worst_coverage_slice is None
                else self.target_worst_coverage_slice.to_dict()
            ),
            "target_longest_interval_slice": (
                None
                if self.target_longest_interval_slice is None
                else self.target_longest_interval_slice.to_dict()
            ),
            "target_worst_coverage_source": self.target_worst_coverage_source,
            "target_longest_interval_source": self.target_longest_interval_source,
        }


@dataclass(slots=True)
class Phase7NonparametricCalibrationObjectReport:
    oracle_lane: str
    stage_label: str
    random_states: tuple[int, ...]
    target_n_obs: int
    reference_n_obs: int
    p: int
    decompositions: tuple[Phase7NonparametricCalibrationObjectDecomposition, ...]
    typed_invalidity_counts: dict[str, int] = field(default_factory=dict)
    typed_invalidity_examples: dict[str, dict[str, object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.random_states = tuple(int(value) for value in self.random_states)
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.decompositions = tuple(self.decompositions)
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )
        self.typed_invalidity_examples = _normalize_invalidity_examples(
            self.typed_invalidity_examples
        )

    def decomposition(
        self, dgp_name: str
    ) -> Phase7NonparametricCalibrationObjectDecomposition:
        target_name = str(dgp_name).strip().upper()
        for decomposition in self.decompositions:
            if decomposition.dgp_name == target_name:
                return decomposition
        raise KeyError(
            f"nonparametric calibration object decomposition not present: {target_name!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "random_states": list(self.random_states),
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "decompositions": [
                decomposition.to_dict() for decomposition in self.decompositions
            ],
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "typed_invalidity_examples": _serialize_invalidity_examples(
                self.typed_invalidity_examples
            ),
        }


@dataclass(slots=True)
class Phase7OuterInferenceObjectContract:
    evaluation_grid: np.ndarray
    alpha: float
    n_valid: int
    basis_family: str
    basis_degree: int
    oracle_lane: str
    bar_f_at_z0: np.ndarray
    sigma_z_hat: np.ndarray
    covariance_at_grid: np.ndarray
    uniform_critical_value: float
    n_boot: int
    random_state: int
    status: str = _PHASE7_REFERENCE_ONLY

    def __post_init__(self) -> None:
        self.evaluation_grid = _coerce_evaluation_grid(self.evaluation_grid)
        self.alpha = float(self.alpha)
        self.n_valid = int(self.n_valid)
        self.basis_family = str(self.basis_family).strip().lower()
        self.basis_degree = int(self.basis_degree)
        self.oracle_lane = normalize_oracle_lane(
            basis_family=self.basis_family,
            oracle_lane=self.oracle_lane,
        )
        self.bar_f_at_z0 = _coerce_evaluation_grid(self.bar_f_at_z0)
        self.sigma_z_hat = _coerce_evaluation_grid(self.sigma_z_hat)
        self.covariance_at_grid = np.asarray(self.covariance_at_grid, dtype=float)
        self.uniform_critical_value = float(self.uniform_critical_value)
        self.n_boot = int(self.n_boot)
        self.random_state = int(self.random_state)
        self.status = str(self.status).strip()
        if self.n_valid <= 0:
            raise ValueError("n_valid must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        grid_size = int(self.evaluation_grid.shape[0])
        if self.bar_f_at_z0.shape[0] != grid_size:
            raise ValueError("bar_f_at_z0 must align with evaluation_grid")
        if self.sigma_z_hat.shape[0] != grid_size:
            raise ValueError("sigma_z_hat must align with evaluation_grid")
        if self.covariance_at_grid.shape != (grid_size, grid_size):
            raise ValueError(
                "covariance_at_grid must be square and align with evaluation_grid"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_grid": self.evaluation_grid.astype(float).tolist(),
            "alpha": self.alpha,
            "n_valid": self.n_valid,
            "basis_family": self.basis_family,
            "basis_degree": self.basis_degree,
            "oracle_lane": self.oracle_lane,
            "bar_f_at_z0": self.bar_f_at_z0.astype(float).tolist(),
            "sigma_z_hat": self.sigma_z_hat.astype(float).tolist(),
            "covariance_at_grid": self.covariance_at_grid.astype(float).tolist(),
            "uniform_critical_value": self.uniform_critical_value,
            "n_boot": self.n_boot,
            "random_state": self.random_state,
            "status": self.status,
        }


def build_phase7_outer_inference_object_contract(
    nonparametric_payload: Any,
    *,
    evaluation_grid: np.ndarray | Sequence[float],
    n_valid: int,
) -> Phase7OuterInferenceObjectContract:
    n_valid_value = int(n_valid)
    if n_valid_value <= 0:
        raise ValueError("n_valid must be positive")

    evaluation_grid_value = _coerce_evaluation_grid(evaluation_grid)
    evaluation_basis = np.asarray(nonparametric_payload.evaluation_basis, dtype=float)
    expected_evaluation_basis = build_sieve_basis(
        evaluation_grid_value,
        basis_family=str(nonparametric_payload.basis_family),
        degree=int(nonparametric_payload.basis_degree),
    )
    if expected_evaluation_basis.shape != evaluation_basis.shape or not np.allclose(
        expected_evaluation_basis,
        evaluation_basis,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("evaluation_grid must reproduce evaluation_basis")
    v_f_hat = np.asarray(nonparametric_payload.v_f_hat, dtype=float)
    if evaluation_basis.shape[0] != evaluation_grid_value.shape[0]:
        raise ValueError("evaluation_basis must align with evaluation_grid")

    covariance_at_grid = evaluation_basis @ v_f_hat @ evaluation_basis.T / n_valid_value
    payload_covariance_at_grid = np.asarray(
        getattr(nonparametric_payload, "covariance_at_grid", covariance_at_grid),
        dtype=float,
    )
    if payload_covariance_at_grid.shape != covariance_at_grid.shape or not np.allclose(
        payload_covariance_at_grid,
        covariance_at_grid,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "payload covariance_at_grid must match evaluation_basis @ v_f_hat @ evaluation_basis.T / n_valid"
        )
    sigma_z_hat = _coerce_evaluation_grid(
        np.asarray(nonparametric_payload.sigma_z_hat, dtype=float)
    )
    diagonal = np.asarray(np.diag(covariance_at_grid), dtype=float)
    if not np.allclose(diagonal, sigma_z_hat**2, atol=1e-12, rtol=0.0):
        raise ValueError("covariance_at_grid diagonal must match sigma_z_hat squared")
    payload_standardization = np.asarray(
        getattr(nonparametric_payload, "uniform_standardization", np.sqrt(diagonal)),
        dtype=float,
    )
    if payload_standardization.ndim != 1:
        raise ValueError("uniform_standardization must be one-dimensional")
    if payload_standardization.shape[0] != evaluation_grid_value.shape[0]:
        raise ValueError("uniform_standardization must align with evaluation_grid")
    if not np.allclose(
        payload_standardization,
        np.sqrt(diagonal),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "uniform_standardization must equal sqrt(diag(covariance_at_grid))"
        )

    bar_f_at_z0 = _coerce_evaluation_grid(
        np.asarray(nonparametric_payload.bar_f_at_z0, dtype=float)
    )
    pointwise_center = (
        np.asarray(
            nonparametric_payload.pointwise_confidence_interval.lower, dtype=float
        )
        + np.asarray(
            nonparametric_payload.pointwise_confidence_interval.upper, dtype=float
        )
    ) / 2.0
    if not np.allclose(pointwise_center, bar_f_at_z0, atol=1e-12, rtol=0.0):
        raise ValueError("pointwise interval center must match bar_f_at_z0")

    uniform_center = (
        np.asarray(nonparametric_payload.uniform_band.lower, dtype=float)
        + np.asarray(nonparametric_payload.uniform_band.upper, dtype=float)
    ) / 2.0
    if not np.allclose(uniform_center, bar_f_at_z0, atol=1e-12, rtol=0.0):
        raise ValueError("uniform band center must match bar_f_at_z0")

    return Phase7OuterInferenceObjectContract(
        evaluation_grid=evaluation_grid_value,
        alpha=float(nonparametric_payload.alpha),
        n_valid=n_valid_value,
        basis_family=str(nonparametric_payload.basis_family),
        basis_degree=int(nonparametric_payload.basis_degree),
        oracle_lane=str(nonparametric_payload.oracle_lane),
        bar_f_at_z0=bar_f_at_z0,
        sigma_z_hat=sigma_z_hat,
        covariance_at_grid=covariance_at_grid,
        uniform_critical_value=float(nonparametric_payload.uniform_band.critical_value),
        n_boot=int(nonparametric_payload.uniform_band.n_boot),
        random_state=int(nonparametric_payload.uniform_band.random_state),
    )


def _build_phase7_outer_inference_reference_payloads() -> tuple[
    ScorePayload, EstimationPayload, np.ndarray, int
]:
    x_valid = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [-1.0, 0.5],
            [0.0, -1.5],
            [1.5, 0.75],
        ],
        dtype=float,
    )
    basis_valid_full = np.array(
        [
            [1.0, -1.0, 1.0],
            [1.0, -0.25, 0.0625],
            [1.0, 0.5, 0.25],
            [1.0, 1.0, 1.0],
            [1.0, 1.5, 2.25],
        ],
        dtype=float,
    )
    evaluation_grid = _PHASE7_OUTER_INFERENCE_REFERENCE_GRID.copy()
    evaluation_basis = build_sieve_basis(
        evaluation_grid,
        basis_family="polynomial",
        degree=2,
    )
    beta_hat = np.array([0.6, -0.3], dtype=float)
    gamma_hat = np.array([0.4, -0.2, 0.15], dtype=float)
    residual_valid = np.array([0.4, 0.2, -0.1, 0.3, -0.2], dtype=float)
    fitted_f_valid = basis_valid_full @ gamma_hat
    second_stage_prediction_valid = x_valid @ beta_hat + fitted_f_valid
    s_hat_valid = second_stage_prediction_valid + residual_valid

    score_payload = ScorePayload(
        delta_y=s_hat_valid,
        s_hat=s_hat_valid,
        s_hat_valid=s_hat_valid,
        valid_mask=np.ones(x_valid.shape[0], dtype=bool),
        fold_ids=np.array([1, 1, 2, 2, 3], dtype=int),
        fold_diagnostics=[
            FoldDiagnostics(
                fold_id=1,
                basis_family="polynomial",
                basis_degree=2,
                oracle_lane="r-parity-polynomial",
                n_holdout_raw=2,
                n_trimmed_propensity=0,
                n_valid_holdout=2,
                trim_lower=0.01,
                trim_upper=0.99,
            ),
            FoldDiagnostics(
                fold_id=2,
                basis_family="polynomial",
                basis_degree=2,
                oracle_lane="r-parity-polynomial",
                n_holdout_raw=2,
                n_trimmed_propensity=0,
                n_valid_holdout=2,
                trim_lower=0.01,
                trim_upper=0.99,
            ),
            FoldDiagnostics(
                fold_id=3,
                basis_family="polynomial",
                basis_degree=2,
                oracle_lane="r-parity-polynomial",
                n_holdout_raw=1,
                n_trimmed_propensity=0,
                n_valid_holdout=1,
                trim_lower=0.01,
                trim_upper=0.99,
            ),
        ],
        basis_family="polynomial",
        basis_degree=2,
        oracle_lane="r-parity-polynomial",
        basis_matrix=basis_valid_full,
        x_valid=x_valid,
        basis_valid_full=basis_valid_full,
        basis_design_valid=basis_valid_full[:, 1:],
        evaluation_basis=evaluation_basis,
        pi_hat=np.full(x_valid.shape[0], 0.5, dtype=float),
        phi0_hat=np.zeros(x_valid.shape[0], dtype=float),
        phi1_hat=np.zeros(x_valid.shape[0], dtype=float),
        rho_hat=np.ones(x_valid.shape[0], dtype=float),
        intercept_dropped_for_design=True,
    )
    estimation_payload = EstimationPayload(
        beta_hat=beta_hat,
        gamma_hat=gamma_hat,
        fitted_f_valid=fitted_f_valid,
        second_stage_prediction_valid=second_stage_prediction_valid,
        residual_valid=residual_valid,
        projection_x_valid=x_valid,
        f_hat_at_z0=evaluation_basis @ gamma_hat,
        optimization_metadata={
            "solver": "phase7-outer-inference-reference",
            "n_valid_obs": int(x_valid.shape[0]),
            "beta_dimension": int(x_valid.shape[1]),
            "basis_dimension_full": int(basis_valid_full.shape[1]),
        },
    )
    return score_payload, estimation_payload, evaluation_grid, int(x_valid.shape[0])


def build_phase7_outer_inference_reference_contract() -> (
    Phase7OuterInferenceObjectContract
):
    """Return the canonical Trigger 3 object contract used by final-verification smoke."""

    score_payload, estimation_payload, evaluation_grid, n_valid = (
        _build_phase7_outer_inference_reference_payloads()
    )
    nonparametric_payload, _ = estimate_nonparametric_inference(
        score_payload,
        estimation_payload,
        alpha=0.1,
        lambda_double_prime=0.0,
        n_boot=256,
        random_state=123,
    )
    return build_phase7_outer_inference_object_contract(
        nonparametric_payload,
        evaluation_grid=evaluation_grid,
        n_valid=n_valid,
    )


@dataclass(slots=True)
class Phase7NonparametricContrastGridReplay:
    contrast_grid: tuple[float, ...]
    current_grid_slice: Phase7NonparametricCalibrationObjectSlice
    contrast_grid_slice: Phase7NonparametricCalibrationObjectSlice
    mean_absolute_error_delta: float
    mean_sigma_z_hat_delta: float
    mean_pointwise_interval_length_delta: float
    pointwise_coverage_count_delta: int

    def __post_init__(self) -> None:
        self.contrast_grid = tuple(float(value) for value in self.contrast_grid)
        self.mean_absolute_error_delta = float(self.mean_absolute_error_delta)
        self.mean_sigma_z_hat_delta = float(self.mean_sigma_z_hat_delta)
        self.mean_pointwise_interval_length_delta = float(
            self.mean_pointwise_interval_length_delta
        )
        self.pointwise_coverage_count_delta = int(self.pointwise_coverage_count_delta)

    def to_dict(self) -> dict[str, object]:
        return {
            "contrast_grid": list(self.contrast_grid),
            "current_grid_slice": self.current_grid_slice.to_dict(),
            "contrast_grid_slice": self.contrast_grid_slice.to_dict(),
            "mean_absolute_error_delta": self.mean_absolute_error_delta,
            "mean_sigma_z_hat_delta": self.mean_sigma_z_hat_delta,
            "mean_pointwise_interval_length_delta": (
                self.mean_pointwise_interval_length_delta
            ),
            "pointwise_coverage_count_delta": self.pointwise_coverage_count_delta,
        }


@dataclass(slots=True)
class Phase7NonparametricContrastGridReplayDecomposition:
    dgp_name: str
    target_worst_coverage_replay: Phase7NonparametricContrastGridReplay | None = None
    target_longest_interval_replay: Phase7NonparametricContrastGridReplay | None = None

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "target_worst_coverage_replay": (
                None
                if self.target_worst_coverage_replay is None
                else self.target_worst_coverage_replay.to_dict()
            ),
            "target_longest_interval_replay": (
                None
                if self.target_longest_interval_replay is None
                else self.target_longest_interval_replay.to_dict()
            ),
        }


@dataclass(slots=True)
class Phase7NonparametricContrastGridReplayReport:
    oracle_lane: str
    stage_label: str
    random_states: tuple[int, ...]
    target_n_obs: int
    reference_n_obs: int
    p: int
    contrast_grid: tuple[float, ...]
    decompositions: tuple[Phase7NonparametricContrastGridReplayDecomposition, ...]

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.random_states = tuple(int(value) for value in self.random_states)
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.contrast_grid = tuple(float(value) for value in self.contrast_grid)
        self.decompositions = tuple(self.decompositions)

    def decomposition(
        self, dgp_name: str
    ) -> Phase7NonparametricContrastGridReplayDecomposition:
        target_name = str(dgp_name).strip().upper()
        for decomposition in self.decompositions:
            if decomposition.dgp_name == target_name:
                return decomposition
        raise KeyError(
            f"nonparametric contrast-grid replay not present: {target_name!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "random_states": list(self.random_states),
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "contrast_grid": list(self.contrast_grid),
            "decompositions": [
                decomposition.to_dict() for decomposition in self.decompositions
            ],
        }


@dataclass(slots=True)
class Phase7NonparametricOffIntegerReplay:
    grid_label: str
    evaluation_grid: tuple[float, ...]
    replay_slice: Phase7NonparametricCalibrationObjectSlice
    mean_absolute_error_delta: float
    mean_sigma_z_hat_delta: float
    mean_pointwise_interval_length_delta: float
    pointwise_coverage_count_delta: int

    def __post_init__(self) -> None:
        self.grid_label = str(self.grid_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.mean_absolute_error_delta = float(self.mean_absolute_error_delta)
        self.mean_sigma_z_hat_delta = float(self.mean_sigma_z_hat_delta)
        self.mean_pointwise_interval_length_delta = float(
            self.mean_pointwise_interval_length_delta
        )
        self.pointwise_coverage_count_delta = int(self.pointwise_coverage_count_delta)

    def to_dict(self) -> dict[str, object]:
        return {
            "grid_label": self.grid_label,
            "evaluation_grid": list(self.evaluation_grid),
            "replay_slice": self.replay_slice.to_dict(),
            "mean_absolute_error_delta": self.mean_absolute_error_delta,
            "mean_sigma_z_hat_delta": self.mean_sigma_z_hat_delta,
            "mean_pointwise_interval_length_delta": (
                self.mean_pointwise_interval_length_delta
            ),
            "pointwise_coverage_count_delta": self.pointwise_coverage_count_delta,
        }


@dataclass(slots=True)
class Phase7NonparametricOffIntegerFocus:
    dgp_name: str
    focus_target: str
    current_slice: Phase7NonparametricCalibrationObjectSlice
    off_integer_replays: tuple[Phase7NonparametricOffIntegerReplay, ...]

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_target = str(self.focus_target).strip()
        self.off_integer_replays = tuple(self.off_integer_replays)

    @property
    def focus_label(self) -> str:
        return self.focus_target

    @property
    def current_grid_slice(self) -> Phase7NonparametricCalibrationObjectSlice:
        return self.current_slice

    def to_dict(self) -> dict[str, object]:
        payload = {
            "dgp_name": self.dgp_name,
            "focus_target": self.focus_target,
            "current_slice": self.current_slice.to_dict(),
            "off_integer_replays": [
                replay.to_dict() for replay in self.off_integer_replays
            ],
        }
        payload["focus_label"] = self.focus_label
        payload["current_grid_slice"] = payload["current_slice"]
        return payload


@dataclass(slots=True)
class Phase7NonparametricOffIntegerReplayReport:
    oracle_lane: str
    stage_label: str
    random_states: tuple[int, ...]
    target_n_obs: int
    reference_n_obs: int
    p: int
    grid_labels: tuple[str, ...]
    focuses: tuple[Phase7NonparametricOffIntegerFocus, ...]

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.random_states = tuple(int(value) for value in self.random_states)
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.grid_labels = tuple(str(label).strip() for label in self.grid_labels)
        self.focuses = tuple(self.focuses)

    @property
    def focus_replays(self) -> tuple[Phase7NonparametricOffIntegerFocus, ...]:
        return self.focuses

    def focus(self, dgp_name: str) -> Phase7NonparametricOffIntegerFocus:
        target_name = str(dgp_name).strip().upper()
        for focus in self.focuses:
            if focus.dgp_name == target_name:
                return focus
        raise KeyError(f"nonparametric off-integer focus not present: {target_name!r}")

    def focus_replay(self, dgp_name: str) -> Phase7NonparametricOffIntegerFocus:
        return self.focus(dgp_name)

    def to_dict(self) -> dict[str, object]:
        focus_payloads = [focus.to_dict() for focus in self.focuses]
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "random_states": list(self.random_states),
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "grid_labels": list(self.grid_labels),
            "focuses": focus_payloads,
            "focus_replays": focus_payloads,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelHotspotGridPoint:
    grid_value: float
    pointwise_variance: float
    row_to_center_cosine: float
    weighted_covariance_to_center: float
    weighted_correlation_to_center: float
    leading_eigen_share: float
    dominant_mode_share: float
    second_mode_share: float
    top_three_eigen_share: float
    effective_positive_mode_count: float
    dominant_mode_index: int

    def __post_init__(self) -> None:
        self.grid_value = float(self.grid_value)
        self.pointwise_variance = float(self.pointwise_variance)
        self.row_to_center_cosine = float(self.row_to_center_cosine)
        self.weighted_covariance_to_center = float(self.weighted_covariance_to_center)
        self.weighted_correlation_to_center = float(self.weighted_correlation_to_center)
        self.leading_eigen_share = float(self.leading_eigen_share)
        self.dominant_mode_share = float(self.dominant_mode_share)
        self.second_mode_share = float(self.second_mode_share)
        self.top_three_eigen_share = float(self.top_three_eigen_share)
        self.effective_positive_mode_count = float(self.effective_positive_mode_count)
        self.dominant_mode_index = int(self.dominant_mode_index)

    def to_dict(self) -> dict[str, float]:
        return {
            "grid_value": self.grid_value,
            "pointwise_variance": self.pointwise_variance,
            "row_to_center_cosine": self.row_to_center_cosine,
            "weighted_covariance_to_center": self.weighted_covariance_to_center,
            "weighted_correlation_to_center": self.weighted_correlation_to_center,
            "leading_eigen_share": self.leading_eigen_share,
            "dominant_mode_share": self.dominant_mode_share,
            "second_mode_share": self.second_mode_share,
            "top_three_eigen_share": self.top_three_eigen_share,
            "effective_positive_mode_count": self.effective_positive_mode_count,
            "dominant_mode_index": self.dominant_mode_index,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelHotspotReplay:
    grid_label: str
    replay_slice: Phase7NonparametricCalibrationObjectSlice
    center_grid_value: float
    center_index: int
    leading_eigenvalue: float
    leading_eigenvalue_positive_share: float
    grid_points: tuple[Phase7NonparametricSourceLevelHotspotGridPoint, ...]

    def __post_init__(self) -> None:
        self.grid_label = str(self.grid_label).strip()
        self.center_grid_value = float(self.center_grid_value)
        self.center_index = int(self.center_index)
        self.leading_eigenvalue = float(self.leading_eigenvalue)
        self.leading_eigenvalue_positive_share = float(
            self.leading_eigenvalue_positive_share
        )
        self.grid_points = tuple(self.grid_points)

    @property
    def evaluation_grid(self) -> tuple[float, ...]:
        return tuple(
            float(value) for value in self.replay_slice.evaluation_grid.tolist()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "grid_label": self.grid_label,
            "evaluation_grid": list(self.evaluation_grid),
            "replay_slice": self.replay_slice.to_dict(),
            "center_grid_value": self.center_grid_value,
            "center_index": self.center_index,
            "leading_eigenvalue": self.leading_eigenvalue,
            "leading_eigenvalue_positive_share": (
                self.leading_eigenvalue_positive_share
            ),
            "grid_points": [point.to_dict() for point in self.grid_points],
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelHotspotFocus:
    dgp_name: str
    focus_target: str
    current_slice: Phase7NonparametricCalibrationObjectSlice
    source_level_replays: tuple[Phase7NonparametricSourceLevelHotspotReplay, ...]

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_target = str(self.focus_target).strip()
        self.source_level_replays = tuple(self.source_level_replays)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_target": self.focus_target,
            "current_slice": self.current_slice.to_dict(),
            "source_level_replays": [
                replay.to_dict() for replay in self.source_level_replays
            ],
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelHotspotReport:
    oracle_lane: str
    stage_label: str
    random_states: tuple[int, ...]
    target_n_obs: int
    reference_n_obs: int
    p: int
    grid_labels: tuple[str, ...]
    hotspot_center: float
    focuses: tuple[Phase7NonparametricSourceLevelHotspotFocus, ...]
    typed_invalidity_counts: dict[str, int] = field(default_factory=dict)
    typed_invalidity_examples: dict[str, dict[str, object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.random_states = tuple(int(value) for value in self.random_states)
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.grid_labels = tuple(str(label).strip() for label in self.grid_labels)
        self.hotspot_center = float(self.hotspot_center)
        self.focuses = tuple(self.focuses)
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )
        self.typed_invalidity_examples = _normalize_invalidity_examples(
            self.typed_invalidity_examples
        )

    def focus(self, dgp_name: str) -> Phase7NonparametricSourceLevelHotspotFocus:
        target_name = str(dgp_name).strip().upper()
        for focus in self.focuses:
            if focus.dgp_name == target_name:
                return focus
        raise KeyError(
            f"nonparametric source-level hotspot focus not present: {target_name!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "random_states": list(self.random_states),
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "grid_labels": list(self.grid_labels),
            "hotspot_center": self.hotspot_center,
            "focuses": [focus.to_dict() for focus in self.focuses],
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "typed_invalidity_examples": _serialize_invalidity_examples(
                self.typed_invalidity_examples
            ),
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelModeComparisonFocus:
    dgp_name: str
    focus_target: str
    random_state: int
    replication_seed: int
    current_slice: Phase7NonparametricCalibrationObjectSlice
    source_level_replays: tuple[Phase7NonparametricSourceLevelHotspotReplay, ...]

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_target = str(self.focus_target).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.source_level_replays = tuple(self.source_level_replays)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_target": self.focus_target,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "current_slice": self.current_slice.to_dict(),
            "source_level_replays": [
                replay.to_dict() for replay in self.source_level_replays
            ],
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelModeComparisonReport:
    oracle_lane: str
    stage_label: str
    random_states: tuple[int, ...]
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    grid_labels: tuple[str, ...]
    hotspot_center: float
    focuses: tuple[Phase7NonparametricSourceLevelModeComparisonFocus, ...]

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.random_states = tuple(int(value) for value in self.random_states)
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.grid_labels = tuple(str(label).strip() for label in self.grid_labels)
        self.hotspot_center = float(self.hotspot_center)
        self.focuses = tuple(self.focuses)

    def focus(
        self, random_state: int
    ) -> Phase7NonparametricSourceLevelModeComparisonFocus:
        target_random_state = int(random_state)
        for focus in self.focuses:
            if focus.random_state == target_random_state:
                return focus
        raise KeyError(
            "nonparametric source-level mode-comparison focus not present: "
            f"{target_random_state!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "random_states": list(self.random_states),
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "grid_labels": list(self.grid_labels),
            "hotspot_center": self.hotspot_center,
            "focuses": [focus.to_dict() for focus in self.focuses],
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelSeedNeighborhoodEntry:
    dgp_name: str
    random_state: int
    replication_seed: int
    distance_from_center: int
    current_slice: Phase7NonparametricCalibrationObjectSlice
    source_level_replays: tuple[Phase7NonparametricSourceLevelHotspotReplay, ...]

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.distance_from_center = int(self.distance_from_center)
        self.source_level_replays = tuple(self.source_level_replays)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "distance_from_center": self.distance_from_center,
            "current_slice": self.current_slice.to_dict(),
            "source_level_replays": [
                replay.to_dict() for replay in self.source_level_replays
            ],
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelSeedNeighborhoodReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    center_random_state: int
    neighborhood_random_states: tuple[int, ...]
    grid_labels: tuple[str, ...]
    hotspot_center: float
    entries: tuple[Phase7NonparametricSourceLevelSeedNeighborhoodEntry, ...]
    typed_invalidity_counts: dict[str, int] = field(default_factory=dict)
    typed_invalidity_examples: dict[str, dict[str, object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.center_random_state = int(self.center_random_state)
        self.neighborhood_random_states = tuple(
            int(value) for value in self.neighborhood_random_states
        )
        self.grid_labels = tuple(str(label).strip() for label in self.grid_labels)
        self.hotspot_center = float(self.hotspot_center)
        self.entries = tuple(self.entries)
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )
        self.typed_invalidity_examples = _normalize_invalidity_examples(
            self.typed_invalidity_examples
        )

    def entry(
        self, random_state: int
    ) -> Phase7NonparametricSourceLevelSeedNeighborhoodEntry:
        target_random_state = int(random_state)
        for entry in self.entries:
            if entry.random_state == target_random_state:
                return entry
        available_random_states = tuple(entry.random_state for entry in self.entries)
        raise ValueError(
            "nonparametric source-level seed-neighborhood entry unavailable after "
            "statistical invalidity filtering; "
            f"requested_random_state={target_random_state!r}; "
            f"available_random_states={available_random_states!r}; "
            f"typed_invalidity_counts={self.typed_invalidity_counts!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "center_random_state": self.center_random_state,
            "neighborhood_random_states": list(self.neighborhood_random_states),
            "grid_labels": list(self.grid_labels),
            "hotspot_center": self.hotspot_center,
            "entries": [entry.to_dict() for entry in self.entries],
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "typed_invalidity_examples": _serialize_invalidity_examples(
                self.typed_invalidity_examples
            ),
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelCoverageDecompositionReplay:
    grid_label: str
    evaluation_grid: tuple[float, ...]
    replay_slice: Phase7NonparametricCalibrationObjectSlice
    pointwise_coverage_count: int
    pointwise_coverage_count_delta: int
    mean_absolute_error_delta: float
    mean_absolute_error_ratio_to_current: float
    mean_sigma_z_hat_delta: float
    mean_sigma_z_hat_ratio_to_current: float
    mean_pointwise_interval_length_delta: float
    mean_pointwise_interval_length_ratio_to_current: float
    mean_error_to_half_interval_ratio: float
    mean_error_to_half_interval_ratio_delta: float
    center_leading_eigen_share: float
    leading_eigenvalue_positive_share: float

    def __post_init__(self) -> None:
        self.grid_label = str(self.grid_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.pointwise_coverage_count = int(self.pointwise_coverage_count)
        self.pointwise_coverage_count_delta = int(self.pointwise_coverage_count_delta)
        self.mean_absolute_error_delta = float(self.mean_absolute_error_delta)
        self.mean_absolute_error_ratio_to_current = float(
            self.mean_absolute_error_ratio_to_current
        )
        self.mean_sigma_z_hat_delta = float(self.mean_sigma_z_hat_delta)
        self.mean_sigma_z_hat_ratio_to_current = float(
            self.mean_sigma_z_hat_ratio_to_current
        )
        self.mean_pointwise_interval_length_delta = float(
            self.mean_pointwise_interval_length_delta
        )
        self.mean_pointwise_interval_length_ratio_to_current = float(
            self.mean_pointwise_interval_length_ratio_to_current
        )
        self.mean_error_to_half_interval_ratio = float(
            self.mean_error_to_half_interval_ratio
        )
        self.mean_error_to_half_interval_ratio_delta = float(
            self.mean_error_to_half_interval_ratio_delta
        )
        self.center_leading_eigen_share = float(self.center_leading_eigen_share)
        self.leading_eigenvalue_positive_share = float(
            self.leading_eigenvalue_positive_share
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "grid_label": self.grid_label,
            "evaluation_grid": list(self.evaluation_grid),
            "replay_slice": self.replay_slice.to_dict(),
            "pointwise_coverage_count": self.pointwise_coverage_count,
            "pointwise_coverage_count_delta": self.pointwise_coverage_count_delta,
            "mean_absolute_error_delta": self.mean_absolute_error_delta,
            "mean_absolute_error_ratio_to_current": (
                self.mean_absolute_error_ratio_to_current
            ),
            "mean_sigma_z_hat_delta": self.mean_sigma_z_hat_delta,
            "mean_sigma_z_hat_ratio_to_current": (
                self.mean_sigma_z_hat_ratio_to_current
            ),
            "mean_pointwise_interval_length_delta": (
                self.mean_pointwise_interval_length_delta
            ),
            "mean_pointwise_interval_length_ratio_to_current": (
                self.mean_pointwise_interval_length_ratio_to_current
            ),
            "mean_error_to_half_interval_ratio": (
                self.mean_error_to_half_interval_ratio
            ),
            "mean_error_to_half_interval_ratio_delta": (
                self.mean_error_to_half_interval_ratio_delta
            ),
            "center_leading_eigen_share": self.center_leading_eigen_share,
            "leading_eigenvalue_positive_share": (
                self.leading_eigenvalue_positive_share
            ),
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelCoverageDecompositionFocus:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    current_slice: Phase7NonparametricCalibrationObjectSlice
    current_pointwise_coverage_count: int
    current_mean_error_to_half_interval_ratio: float
    coverage_replays: tuple[
        Phase7NonparametricSourceLevelCoverageDecompositionReplay, ...
    ]

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.current_pointwise_coverage_count = int(
            self.current_pointwise_coverage_count
        )
        self.current_mean_error_to_half_interval_ratio = float(
            self.current_mean_error_to_half_interval_ratio
        )
        self.coverage_replays = tuple(self.coverage_replays)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "current_slice": self.current_slice.to_dict(),
            "current_pointwise_coverage_count": self.current_pointwise_coverage_count,
            "current_mean_error_to_half_interval_ratio": (
                self.current_mean_error_to_half_interval_ratio
            ),
            "coverage_replays": [replay.to_dict() for replay in self.coverage_replays],
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelCoverageDecompositionReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    center_random_state: int
    focus_random_states: tuple[int, ...]
    grid_labels: tuple[str, ...]
    focuses: tuple[Phase7NonparametricSourceLevelCoverageDecompositionFocus, ...]

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.center_random_state = int(self.center_random_state)
        self.focus_random_states = tuple(
            int(value) for value in self.focus_random_states
        )
        self.grid_labels = tuple(str(label).strip() for label in self.grid_labels)
        self.focuses = tuple(self.focuses)

    def focus(
        self, random_state: int
    ) -> Phase7NonparametricSourceLevelCoverageDecompositionFocus:
        target_random_state = int(random_state)
        for focus in self.focuses:
            if focus.random_state == target_random_state:
                return focus
        raise KeyError(
            "nonparametric source-level coverage-decomposition focus not present: "
            f"{target_random_state!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "center_random_state": self.center_random_state,
            "focus_random_states": list(self.focus_random_states),
            "grid_labels": list(self.grid_labels),
            "focuses": [focus.to_dict() for focus in self.focuses],
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelRatioShapePointProfile:
    grid_value: float
    pointwise_coverage: bool
    absolute_error: float
    sigma_z_hat: float
    pointwise_interval_length: float
    error_to_half_interval_ratio: float
    pointwise_ratio_above_one: bool
    row_to_center_cosine: float
    weighted_correlation_to_center: float
    leading_eigen_share: float

    def __post_init__(self) -> None:
        self.grid_value = float(self.grid_value)
        self.pointwise_coverage = bool(self.pointwise_coverage)
        self.absolute_error = float(self.absolute_error)
        self.sigma_z_hat = float(self.sigma_z_hat)
        self.pointwise_interval_length = float(self.pointwise_interval_length)
        self.error_to_half_interval_ratio = float(self.error_to_half_interval_ratio)
        self.pointwise_ratio_above_one = bool(self.pointwise_ratio_above_one)
        self.row_to_center_cosine = float(self.row_to_center_cosine)
        self.weighted_correlation_to_center = float(self.weighted_correlation_to_center)
        self.leading_eigen_share = float(self.leading_eigen_share)

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "grid_value": self.grid_value,
            "pointwise_coverage": self.pointwise_coverage,
            "absolute_error": self.absolute_error,
            "sigma_z_hat": self.sigma_z_hat,
            "pointwise_interval_length": self.pointwise_interval_length,
            "error_to_half_interval_ratio": self.error_to_half_interval_ratio,
            "pointwise_ratio_above_one": self.pointwise_ratio_above_one,
            "row_to_center_cosine": self.row_to_center_cosine,
            "weighted_correlation_to_center": self.weighted_correlation_to_center,
            "leading_eigen_share": self.leading_eigen_share,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelRatioShapeFocus:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    target_grid_label: str
    current_slice: Phase7NonparametricCalibrationObjectSlice
    target_slice: Phase7NonparametricCalibrationObjectSlice
    failure_signature: str
    point_profiles: tuple[Phase7NonparametricSourceLevelRatioShapePointProfile, ...]
    center_index: int
    center_grid_value: float
    center_pointwise_ratio: float
    min_pointwise_ratio: float
    max_pointwise_ratio: float
    pointwise_ratio_range: float
    max_pointwise_ratio_grid_value: float
    min_leading_eigen_share: float
    max_leading_eigen_share: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.target_grid_label = str(self.target_grid_label).strip()
        self.failure_signature = str(self.failure_signature).strip()
        self.point_profiles = tuple(self.point_profiles)
        self.center_index = int(self.center_index)
        self.center_grid_value = float(self.center_grid_value)
        self.center_pointwise_ratio = float(self.center_pointwise_ratio)
        self.min_pointwise_ratio = float(self.min_pointwise_ratio)
        self.max_pointwise_ratio = float(self.max_pointwise_ratio)
        self.pointwise_ratio_range = float(self.pointwise_ratio_range)
        self.max_pointwise_ratio_grid_value = float(self.max_pointwise_ratio_grid_value)
        self.min_leading_eigen_share = float(self.min_leading_eigen_share)
        self.max_leading_eigen_share = float(self.max_leading_eigen_share)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "target_grid_label": self.target_grid_label,
            "current_slice": self.current_slice.to_dict(),
            "target_slice": self.target_slice.to_dict(),
            "failure_signature": self.failure_signature,
            "point_profiles": [profile.to_dict() for profile in self.point_profiles],
            "center_index": self.center_index,
            "center_grid_value": self.center_grid_value,
            "center_pointwise_ratio": self.center_pointwise_ratio,
            "min_pointwise_ratio": self.min_pointwise_ratio,
            "max_pointwise_ratio": self.max_pointwise_ratio,
            "pointwise_ratio_range": self.pointwise_ratio_range,
            "max_pointwise_ratio_grid_value": self.max_pointwise_ratio_grid_value,
            "min_leading_eigen_share": self.min_leading_eigen_share,
            "max_leading_eigen_share": self.max_leading_eigen_share,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelRatioShapeReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    center_random_state: int
    focus_random_states: tuple[int, ...]
    target_grid_label: str
    focuses: tuple[Phase7NonparametricSourceLevelRatioShapeFocus, ...]

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.center_random_state = int(self.center_random_state)
        self.focus_random_states = tuple(
            int(value) for value in self.focus_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.focuses = tuple(self.focuses)

    def focus(self, random_state: int) -> Phase7NonparametricSourceLevelRatioShapeFocus:
        target_random_state = int(random_state)
        for focus in self.focuses:
            if focus.random_state == target_random_state:
                return focus
        raise KeyError(
            "nonparametric source-level ratio-shape focus not present: "
            f"{target_random_state!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "center_random_state": self.center_random_state,
            "focus_random_states": list(self.focus_random_states),
            "target_grid_label": self.target_grid_label,
            "focuses": [focus.to_dict() for focus in self.focuses],
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelComponentShapeFocus:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    target_grid_label: str
    ratio_shape_signature: str
    component_signature: str
    point_profiles: tuple[Phase7NonparametricSourceLevelRatioShapePointProfile, ...]
    center_index: int
    center_grid_value: float
    center_is_max_absolute_error: bool
    center_is_max_sigma_z_hat: bool
    center_is_min_sigma_z_hat: bool
    center_is_max_pointwise_interval_length: bool
    center_is_min_pointwise_interval_length: bool
    min_shoulder_to_center_absolute_error_ratio: float
    max_shoulder_to_center_absolute_error_ratio: float
    min_shoulder_to_center_sigma_z_hat_ratio: float
    max_shoulder_to_center_sigma_z_hat_ratio: float
    min_shoulder_to_center_interval_length_ratio: float
    max_shoulder_to_center_interval_length_ratio: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.target_grid_label = str(self.target_grid_label).strip()
        self.ratio_shape_signature = str(self.ratio_shape_signature).strip()
        self.component_signature = str(self.component_signature).strip()
        self.point_profiles = tuple(self.point_profiles)
        self.center_index = int(self.center_index)
        self.center_grid_value = float(self.center_grid_value)
        self.center_is_max_absolute_error = bool(self.center_is_max_absolute_error)
        self.center_is_max_sigma_z_hat = bool(self.center_is_max_sigma_z_hat)
        self.center_is_min_sigma_z_hat = bool(self.center_is_min_sigma_z_hat)
        self.center_is_max_pointwise_interval_length = bool(
            self.center_is_max_pointwise_interval_length
        )
        self.center_is_min_pointwise_interval_length = bool(
            self.center_is_min_pointwise_interval_length
        )
        self.min_shoulder_to_center_absolute_error_ratio = float(
            self.min_shoulder_to_center_absolute_error_ratio
        )
        self.max_shoulder_to_center_absolute_error_ratio = float(
            self.max_shoulder_to_center_absolute_error_ratio
        )
        self.min_shoulder_to_center_sigma_z_hat_ratio = float(
            self.min_shoulder_to_center_sigma_z_hat_ratio
        )
        self.max_shoulder_to_center_sigma_z_hat_ratio = float(
            self.max_shoulder_to_center_sigma_z_hat_ratio
        )
        self.min_shoulder_to_center_interval_length_ratio = float(
            self.min_shoulder_to_center_interval_length_ratio
        )
        self.max_shoulder_to_center_interval_length_ratio = float(
            self.max_shoulder_to_center_interval_length_ratio
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "target_grid_label": self.target_grid_label,
            "ratio_shape_signature": self.ratio_shape_signature,
            "component_signature": self.component_signature,
            "point_profiles": [profile.to_dict() for profile in self.point_profiles],
            "center_index": self.center_index,
            "center_grid_value": self.center_grid_value,
            "center_is_max_absolute_error": self.center_is_max_absolute_error,
            "center_is_max_sigma_z_hat": self.center_is_max_sigma_z_hat,
            "center_is_min_sigma_z_hat": self.center_is_min_sigma_z_hat,
            "center_is_max_pointwise_interval_length": (
                self.center_is_max_pointwise_interval_length
            ),
            "center_is_min_pointwise_interval_length": (
                self.center_is_min_pointwise_interval_length
            ),
            "min_shoulder_to_center_absolute_error_ratio": (
                self.min_shoulder_to_center_absolute_error_ratio
            ),
            "max_shoulder_to_center_absolute_error_ratio": (
                self.max_shoulder_to_center_absolute_error_ratio
            ),
            "min_shoulder_to_center_sigma_z_hat_ratio": (
                self.min_shoulder_to_center_sigma_z_hat_ratio
            ),
            "max_shoulder_to_center_sigma_z_hat_ratio": (
                self.max_shoulder_to_center_sigma_z_hat_ratio
            ),
            "min_shoulder_to_center_interval_length_ratio": (
                self.min_shoulder_to_center_interval_length_ratio
            ),
            "max_shoulder_to_center_interval_length_ratio": (
                self.max_shoulder_to_center_interval_length_ratio
            ),
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelComponentShapeReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    center_random_state: int
    focus_random_states: tuple[int, ...]
    target_grid_label: str
    focuses: tuple[Phase7NonparametricSourceLevelComponentShapeFocus, ...]

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.center_random_state = int(self.center_random_state)
        self.focus_random_states = tuple(
            int(value) for value in self.focus_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.focuses = tuple(self.focuses)

    def focus(
        self, random_state: int
    ) -> Phase7NonparametricSourceLevelComponentShapeFocus:
        target_random_state = int(random_state)
        for focus in self.focuses:
            if focus.random_state == target_random_state:
                return focus
        raise KeyError(
            "nonparametric source-level component-shape focus not present: "
            f"{target_random_state!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "center_random_state": self.center_random_state,
            "focus_random_states": list(self.focus_random_states),
            "target_grid_label": self.target_grid_label,
            "focuses": [focus.to_dict() for focus in self.focuses],
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelComponentDriverFocus:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    target_grid_label: str
    ratio_shape_signature: str
    component_signature: str
    driver_signature: str
    point_profiles: tuple[Phase7NonparametricSourceLevelRatioShapePointProfile, ...]
    center_index: int
    center_grid_value: float
    min_pointwise_ratio: float
    max_pointwise_ratio: float
    center_to_shoulder_mean_absolute_error_ratio: float
    center_to_shoulder_mean_sigma_z_hat_ratio: float
    center_to_shoulder_mean_interval_length_ratio: float
    error_to_sigma_center_shoulder_gap: float
    error_to_interval_center_shoulder_gap: float
    component_alignment_spread: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.target_grid_label = str(self.target_grid_label).strip()
        self.ratio_shape_signature = str(self.ratio_shape_signature).strip()
        self.component_signature = str(self.component_signature).strip()
        self.driver_signature = str(self.driver_signature).strip()
        self.point_profiles = tuple(self.point_profiles)
        self.center_index = int(self.center_index)
        self.center_grid_value = float(self.center_grid_value)
        self.min_pointwise_ratio = float(self.min_pointwise_ratio)
        self.max_pointwise_ratio = float(self.max_pointwise_ratio)
        self.center_to_shoulder_mean_absolute_error_ratio = float(
            self.center_to_shoulder_mean_absolute_error_ratio
        )
        self.center_to_shoulder_mean_sigma_z_hat_ratio = float(
            self.center_to_shoulder_mean_sigma_z_hat_ratio
        )
        self.center_to_shoulder_mean_interval_length_ratio = float(
            self.center_to_shoulder_mean_interval_length_ratio
        )
        self.error_to_sigma_center_shoulder_gap = float(
            self.error_to_sigma_center_shoulder_gap
        )
        self.error_to_interval_center_shoulder_gap = float(
            self.error_to_interval_center_shoulder_gap
        )
        self.component_alignment_spread = float(self.component_alignment_spread)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "target_grid_label": self.target_grid_label,
            "ratio_shape_signature": self.ratio_shape_signature,
            "component_signature": self.component_signature,
            "driver_signature": self.driver_signature,
            "point_profiles": [profile.to_dict() for profile in self.point_profiles],
            "center_index": self.center_index,
            "center_grid_value": self.center_grid_value,
            "min_pointwise_ratio": self.min_pointwise_ratio,
            "max_pointwise_ratio": self.max_pointwise_ratio,
            "center_to_shoulder_mean_absolute_error_ratio": (
                self.center_to_shoulder_mean_absolute_error_ratio
            ),
            "center_to_shoulder_mean_sigma_z_hat_ratio": (
                self.center_to_shoulder_mean_sigma_z_hat_ratio
            ),
            "center_to_shoulder_mean_interval_length_ratio": (
                self.center_to_shoulder_mean_interval_length_ratio
            ),
            "error_to_sigma_center_shoulder_gap": (
                self.error_to_sigma_center_shoulder_gap
            ),
            "error_to_interval_center_shoulder_gap": (
                self.error_to_interval_center_shoulder_gap
            ),
            "component_alignment_spread": self.component_alignment_spread,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelComponentDriverReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    center_random_state: int
    focus_random_states: tuple[int, ...]
    target_grid_label: str
    focuses: tuple[Phase7NonparametricSourceLevelComponentDriverFocus, ...]

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.center_random_state = int(self.center_random_state)
        self.focus_random_states = tuple(
            int(value) for value in self.focus_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.focuses = tuple(self.focuses)

    def focus(
        self, random_state: int
    ) -> Phase7NonparametricSourceLevelComponentDriverFocus:
        target_random_state = int(random_state)
        for focus in self.focuses:
            if focus.random_state == target_random_state:
                return focus
        raise KeyError(
            "nonparametric source-level component-driver focus not present: "
            f"{target_random_state!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "center_random_state": self.center_random_state,
            "focus_random_states": list(self.focus_random_states),
            "target_grid_label": self.target_grid_label,
            "focuses": [focus.to_dict() for focus in self.focuses],
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelDriverFamilyCandidate:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    driver_signature: str
    driver_family_signature: str
    family_random_states: tuple[int, ...]
    family_support_count: int
    is_unique_threshold_crossing_member: bool
    min_pointwise_ratio: float
    max_pointwise_ratio: float
    error_to_interval_center_shoulder_gap: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.driver_signature = str(self.driver_signature).strip()
        self.driver_family_signature = str(self.driver_family_signature).strip()
        self.family_random_states = tuple(
            int(value) for value in self.family_random_states
        )
        self.family_support_count = int(self.family_support_count)
        self.is_unique_threshold_crossing_member = bool(
            self.is_unique_threshold_crossing_member
        )
        self.min_pointwise_ratio = float(self.min_pointwise_ratio)
        self.max_pointwise_ratio = float(self.max_pointwise_ratio)
        self.error_to_interval_center_shoulder_gap = float(
            self.error_to_interval_center_shoulder_gap
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "driver_signature": self.driver_signature,
            "driver_family_signature": self.driver_family_signature,
            "family_random_states": list(self.family_random_states),
            "family_support_count": self.family_support_count,
            "is_unique_threshold_crossing_member": (
                self.is_unique_threshold_crossing_member
            ),
            "min_pointwise_ratio": self.min_pointwise_ratio,
            "max_pointwise_ratio": self.max_pointwise_ratio,
            "error_to_interval_center_shoulder_gap": (
                self.error_to_interval_center_shoulder_gap
            ),
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelDriverFamilyReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    neighborhood_random_states: tuple[int, ...]
    candidate_random_states: tuple[int, ...]
    target_grid_label: str
    candidates: tuple[Phase7NonparametricSourceLevelDriverFamilyCandidate, ...]
    recommended_next_focus_random_state: int
    recommended_next_focus_signature: str
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.neighborhood_random_states = tuple(
            int(value) for value in self.neighborhood_random_states
        )
        self.candidate_random_states = tuple(
            int(value) for value in self.candidate_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.candidates = tuple(self.candidates)
        self.recommended_next_focus_random_state = int(
            self.recommended_next_focus_random_state
        )
        self.recommended_next_focus_signature = str(
            self.recommended_next_focus_signature
        ).strip()
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def candidate(
        self, random_state: int
    ) -> Phase7NonparametricSourceLevelDriverFamilyCandidate:
        target_random_state = int(random_state)
        for candidate in self.candidates:
            if candidate.random_state == target_random_state:
                return candidate
        raise KeyError(
            "nonparametric source-level driver-family candidate not present: "
            f"{target_random_state!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "neighborhood_random_states": list(self.neighborhood_random_states),
            "candidate_random_states": list(self.candidate_random_states),
            "target_grid_label": self.target_grid_label,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "recommended_next_focus_random_state": (
                self.recommended_next_focus_random_state
            ),
            "recommended_next_focus_signature": (self.recommended_next_focus_signature),
            "recommendation_rationale": self.recommendation_rationale,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftThresholdMember:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    driver_signature: str
    min_pointwise_ratio: float
    max_pointwise_ratio: float
    threshold_margin: float
    stability_range: float
    component_alignment_spread: float
    error_to_interval_center_shoulder_gap: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.driver_signature = str(self.driver_signature).strip()
        self.min_pointwise_ratio = float(self.min_pointwise_ratio)
        self.max_pointwise_ratio = float(self.max_pointwise_ratio)
        self.threshold_margin = float(self.threshold_margin)
        self.stability_range = float(self.stability_range)
        self.component_alignment_spread = float(self.component_alignment_spread)
        self.error_to_interval_center_shoulder_gap = float(
            self.error_to_interval_center_shoulder_gap
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "driver_signature": self.driver_signature,
            "min_pointwise_ratio": self.min_pointwise_ratio,
            "max_pointwise_ratio": self.max_pointwise_ratio,
            "threshold_margin": self.threshold_margin,
            "stability_range": self.stability_range,
            "component_alignment_spread": self.component_alignment_spread,
            "error_to_interval_center_shoulder_gap": (
                self.error_to_interval_center_shoulder_gap
            ),
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftThresholdReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    neighborhood_random_states: tuple[int, ...]
    aligned_lift_family_random_states: tuple[int, ...]
    target_grid_label: str
    members: tuple[Phase7NonparametricSourceLevelAlignedLiftThresholdMember, ...]
    threshold_crossing_random_state: int | None
    recommended_boundary_random_states: tuple[int, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.neighborhood_random_states = tuple(
            int(value) for value in self.neighborhood_random_states
        )
        self.aligned_lift_family_random_states = tuple(
            int(value) for value in self.aligned_lift_family_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.members = tuple(self.members)
        self.threshold_crossing_random_state = (
            None
            if self.threshold_crossing_random_state is None
            else int(self.threshold_crossing_random_state)
        )
        self.recommended_boundary_random_states = tuple(
            int(value) for value in self.recommended_boundary_random_states
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def member(
        self, random_state: int
    ) -> Phase7NonparametricSourceLevelAlignedLiftThresholdMember:
        target_random_state = int(random_state)
        for member in self.members:
            if member.random_state == target_random_state:
                return member
        raise KeyError(
            "nonparametric source-level aligned-lift threshold member not present: "
            f"{target_random_state!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "neighborhood_random_states": list(self.neighborhood_random_states),
            "aligned_lift_family_random_states": list(
                self.aligned_lift_family_random_states
            ),
            "target_grid_label": self.target_grid_label,
            "members": [member.to_dict() for member in self.members],
            "threshold_crossing_random_state": self.threshold_crossing_random_state,
            "recommended_boundary_random_states": list(
                self.recommended_boundary_random_states
            ),
            "recommendation_rationale": self.recommendation_rationale,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftBoundaryComparisonReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    aligned_lift_family_random_states: tuple[int, ...]
    target_grid_label: str
    threshold_crossing_random_state: int
    primary_boundary_random_state: int
    secondary_support_random_state: int
    boundary_ordering_random_states: tuple[int, ...]
    threshold_crossing_member: Phase7NonparametricSourceLevelAlignedLiftThresholdMember
    primary_boundary_member: Phase7NonparametricSourceLevelAlignedLiftThresholdMember
    secondary_support_member: Phase7NonparametricSourceLevelAlignedLiftThresholdMember
    threshold_margin_gap_to_primary_boundary: float
    stability_range_gap_to_primary_boundary: float
    secondary_support_stability_excess_over_primary_boundary: float
    alignment_spread_gap_to_primary_boundary: float
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.aligned_lift_family_random_states = tuple(
            int(value) for value in self.aligned_lift_family_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.threshold_crossing_random_state = int(self.threshold_crossing_random_state)
        self.primary_boundary_random_state = int(self.primary_boundary_random_state)
        self.secondary_support_random_state = int(self.secondary_support_random_state)
        self.boundary_ordering_random_states = tuple(
            int(value) for value in self.boundary_ordering_random_states
        )
        self.threshold_margin_gap_to_primary_boundary = float(
            self.threshold_margin_gap_to_primary_boundary
        )
        self.stability_range_gap_to_primary_boundary = float(
            self.stability_range_gap_to_primary_boundary
        )
        self.secondary_support_stability_excess_over_primary_boundary = float(
            self.secondary_support_stability_excess_over_primary_boundary
        )
        self.alignment_spread_gap_to_primary_boundary = float(
            self.alignment_spread_gap_to_primary_boundary
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "aligned_lift_family_random_states": list(
                self.aligned_lift_family_random_states
            ),
            "target_grid_label": self.target_grid_label,
            "threshold_crossing_random_state": self.threshold_crossing_random_state,
            "primary_boundary_random_state": self.primary_boundary_random_state,
            "secondary_support_random_state": self.secondary_support_random_state,
            "boundary_ordering_random_states": list(
                self.boundary_ordering_random_states
            ),
            "threshold_crossing_member": self.threshold_crossing_member.to_dict(),
            "primary_boundary_member": self.primary_boundary_member.to_dict(),
            "secondary_support_member": self.secondary_support_member.to_dict(),
            "threshold_margin_gap_to_primary_boundary": (
                self.threshold_margin_gap_to_primary_boundary
            ),
            "stability_range_gap_to_primary_boundary": (
                self.stability_range_gap_to_primary_boundary
            ),
            "secondary_support_stability_excess_over_primary_boundary": (
                self.secondary_support_stability_excess_over_primary_boundary
            ),
            "alignment_spread_gap_to_primary_boundary": (
                self.alignment_spread_gap_to_primary_boundary
            ),
            "recommendation_rationale": self.recommendation_rationale,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftSpectralConditionFocus:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    driver_signature: str
    mean_pointwise_ratio: float
    center_pointwise_ratio: float
    threshold_margin: float
    mean_row_to_center_cosine: float
    mean_weighted_correlation_to_center: float
    mean_leading_eigen_share: float
    center_leading_eigen_share: float
    component_alignment_spread: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.driver_signature = str(self.driver_signature).strip()
        self.mean_pointwise_ratio = float(self.mean_pointwise_ratio)
        self.center_pointwise_ratio = float(self.center_pointwise_ratio)
        self.threshold_margin = float(self.threshold_margin)
        self.mean_row_to_center_cosine = float(self.mean_row_to_center_cosine)
        self.mean_weighted_correlation_to_center = float(
            self.mean_weighted_correlation_to_center
        )
        self.mean_leading_eigen_share = float(self.mean_leading_eigen_share)
        self.center_leading_eigen_share = float(self.center_leading_eigen_share)
        self.component_alignment_spread = float(self.component_alignment_spread)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "driver_signature": self.driver_signature,
            "mean_pointwise_ratio": self.mean_pointwise_ratio,
            "center_pointwise_ratio": self.center_pointwise_ratio,
            "threshold_margin": self.threshold_margin,
            "mean_row_to_center_cosine": self.mean_row_to_center_cosine,
            "mean_weighted_correlation_to_center": (
                self.mean_weighted_correlation_to_center
            ),
            "mean_leading_eigen_share": self.mean_leading_eigen_share,
            "center_leading_eigen_share": self.center_leading_eigen_share,
            "component_alignment_spread": self.component_alignment_spread,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftSpectralConditionReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    aligned_lift_family_random_states: tuple[int, ...]
    target_grid_label: str
    threshold_crossing_random_state: int
    primary_boundary_random_state: int
    secondary_support_random_state: int
    threshold_crossing_focus: (
        Phase7NonparametricSourceLevelAlignedLiftSpectralConditionFocus
    )
    primary_boundary_focus: (
        Phase7NonparametricSourceLevelAlignedLiftSpectralConditionFocus
    )
    secondary_support_focus: (
        Phase7NonparametricSourceLevelAlignedLiftSpectralConditionFocus
    )
    mean_pointwise_ratio_gap_to_primary_boundary: float
    mean_row_to_center_cosine_gap_to_primary_boundary: float
    mean_weighted_correlation_gap_to_primary_boundary: float
    mean_leading_eigen_share_gap_to_primary_boundary: float
    center_leading_eigen_share_gap_to_primary_boundary: float
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.aligned_lift_family_random_states = tuple(
            int(value) for value in self.aligned_lift_family_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.threshold_crossing_random_state = int(self.threshold_crossing_random_state)
        self.primary_boundary_random_state = int(self.primary_boundary_random_state)
        self.secondary_support_random_state = int(self.secondary_support_random_state)
        self.mean_pointwise_ratio_gap_to_primary_boundary = float(
            self.mean_pointwise_ratio_gap_to_primary_boundary
        )
        self.mean_row_to_center_cosine_gap_to_primary_boundary = float(
            self.mean_row_to_center_cosine_gap_to_primary_boundary
        )
        self.mean_weighted_correlation_gap_to_primary_boundary = float(
            self.mean_weighted_correlation_gap_to_primary_boundary
        )
        self.mean_leading_eigen_share_gap_to_primary_boundary = float(
            self.mean_leading_eigen_share_gap_to_primary_boundary
        )
        self.center_leading_eigen_share_gap_to_primary_boundary = float(
            self.center_leading_eigen_share_gap_to_primary_boundary
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "aligned_lift_family_random_states": list(
                self.aligned_lift_family_random_states
            ),
            "target_grid_label": self.target_grid_label,
            "threshold_crossing_random_state": self.threshold_crossing_random_state,
            "primary_boundary_random_state": self.primary_boundary_random_state,
            "secondary_support_random_state": self.secondary_support_random_state,
            "threshold_crossing_focus": self.threshold_crossing_focus.to_dict(),
            "primary_boundary_focus": self.primary_boundary_focus.to_dict(),
            "secondary_support_focus": self.secondary_support_focus.to_dict(),
            "mean_pointwise_ratio_gap_to_primary_boundary": (
                self.mean_pointwise_ratio_gap_to_primary_boundary
            ),
            "mean_row_to_center_cosine_gap_to_primary_boundary": (
                self.mean_row_to_center_cosine_gap_to_primary_boundary
            ),
            "mean_weighted_correlation_gap_to_primary_boundary": (
                self.mean_weighted_correlation_gap_to_primary_boundary
            ),
            "mean_leading_eigen_share_gap_to_primary_boundary": (
                self.mean_leading_eigen_share_gap_to_primary_boundary
            ),
            "center_leading_eigen_share_gap_to_primary_boundary": (
                self.center_leading_eigen_share_gap_to_primary_boundary
            ),
            "recommendation_rationale": self.recommendation_rationale,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationFocus:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    driver_signature: str
    spectral_concentration_signature: str
    leading_eigenvalue_positive_share: float
    mean_leading_eigen_share: float
    mean_top_three_eigen_share: float
    mean_nonleading_top_three_share: float
    center_leading_eigen_share: float
    center_top_three_eigen_share: float
    center_nonleading_top_three_share: float
    top_three_eigen_share_range: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.driver_signature = str(self.driver_signature).strip()
        self.spectral_concentration_signature = str(
            self.spectral_concentration_signature
        ).strip()
        self.leading_eigenvalue_positive_share = float(
            self.leading_eigenvalue_positive_share
        )
        self.mean_leading_eigen_share = float(self.mean_leading_eigen_share)
        self.mean_top_three_eigen_share = float(self.mean_top_three_eigen_share)
        self.mean_nonleading_top_three_share = float(
            self.mean_nonleading_top_three_share
        )
        self.center_leading_eigen_share = float(self.center_leading_eigen_share)
        self.center_top_three_eigen_share = float(self.center_top_three_eigen_share)
        self.center_nonleading_top_three_share = float(
            self.center_nonleading_top_three_share
        )
        self.top_three_eigen_share_range = float(self.top_three_eigen_share_range)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "driver_signature": self.driver_signature,
            "spectral_concentration_signature": (self.spectral_concentration_signature),
            "leading_eigenvalue_positive_share": (
                self.leading_eigenvalue_positive_share
            ),
            "mean_leading_eigen_share": self.mean_leading_eigen_share,
            "mean_top_three_eigen_share": self.mean_top_three_eigen_share,
            "mean_nonleading_top_three_share": self.mean_nonleading_top_three_share,
            "center_leading_eigen_share": self.center_leading_eigen_share,
            "center_top_three_eigen_share": self.center_top_three_eigen_share,
            "center_nonleading_top_three_share": (
                self.center_nonleading_top_three_share
            ),
            "top_three_eigen_share_range": self.top_three_eigen_share_range,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    aligned_lift_family_random_states: tuple[int, ...]
    target_grid_label: str
    threshold_crossing_random_state: int
    primary_boundary_random_state: int
    secondary_support_random_state: int
    threshold_crossing_focus: (
        Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationFocus
    )
    primary_boundary_focus: (
        Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationFocus
    )
    secondary_support_focus: (
        Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationFocus
    )
    leading_eigenvalue_positive_share_gap_to_primary_boundary: float
    mean_top_three_eigen_share_gap_to_primary_boundary: float
    mean_nonleading_top_three_share_gap_to_primary_boundary: float
    center_nonleading_top_three_share_gap_to_primary_boundary: float
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.aligned_lift_family_random_states = tuple(
            int(value) for value in self.aligned_lift_family_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.threshold_crossing_random_state = int(self.threshold_crossing_random_state)
        self.primary_boundary_random_state = int(self.primary_boundary_random_state)
        self.secondary_support_random_state = int(self.secondary_support_random_state)
        self.leading_eigenvalue_positive_share_gap_to_primary_boundary = float(
            self.leading_eigenvalue_positive_share_gap_to_primary_boundary
        )
        self.mean_top_three_eigen_share_gap_to_primary_boundary = float(
            self.mean_top_three_eigen_share_gap_to_primary_boundary
        )
        self.mean_nonleading_top_three_share_gap_to_primary_boundary = float(
            self.mean_nonleading_top_three_share_gap_to_primary_boundary
        )
        self.center_nonleading_top_three_share_gap_to_primary_boundary = float(
            self.center_nonleading_top_three_share_gap_to_primary_boundary
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "aligned_lift_family_random_states": list(
                self.aligned_lift_family_random_states
            ),
            "target_grid_label": self.target_grid_label,
            "threshold_crossing_random_state": self.threshold_crossing_random_state,
            "primary_boundary_random_state": self.primary_boundary_random_state,
            "secondary_support_random_state": self.secondary_support_random_state,
            "threshold_crossing_focus": self.threshold_crossing_focus.to_dict(),
            "primary_boundary_focus": self.primary_boundary_focus.to_dict(),
            "secondary_support_focus": self.secondary_support_focus.to_dict(),
            "leading_eigenvalue_positive_share_gap_to_primary_boundary": (
                self.leading_eigenvalue_positive_share_gap_to_primary_boundary
            ),
            "mean_top_three_eigen_share_gap_to_primary_boundary": (
                self.mean_top_three_eigen_share_gap_to_primary_boundary
            ),
            "mean_nonleading_top_three_share_gap_to_primary_boundary": (
                self.mean_nonleading_top_three_share_gap_to_primary_boundary
            ),
            "center_nonleading_top_three_share_gap_to_primary_boundary": (
                self.center_nonleading_top_three_share_gap_to_primary_boundary
            ),
            "recommendation_rationale": self.recommendation_rationale,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankFocus:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    driver_signature: str
    spectral_effective_rank_signature: str
    mean_effective_positive_mode_count: float
    center_effective_positive_mode_count: float
    effective_positive_mode_count_range: float
    mean_tail_share_outside_top_three: float
    center_tail_share_outside_top_three: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.driver_signature = str(self.driver_signature).strip()
        self.spectral_effective_rank_signature = str(
            self.spectral_effective_rank_signature
        ).strip()
        self.mean_effective_positive_mode_count = float(
            self.mean_effective_positive_mode_count
        )
        self.center_effective_positive_mode_count = float(
            self.center_effective_positive_mode_count
        )
        self.effective_positive_mode_count_range = float(
            self.effective_positive_mode_count_range
        )
        self.mean_tail_share_outside_top_three = float(
            self.mean_tail_share_outside_top_three
        )
        self.center_tail_share_outside_top_three = float(
            self.center_tail_share_outside_top_three
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "driver_signature": self.driver_signature,
            "spectral_effective_rank_signature": (
                self.spectral_effective_rank_signature
            ),
            "mean_effective_positive_mode_count": (
                self.mean_effective_positive_mode_count
            ),
            "center_effective_positive_mode_count": (
                self.center_effective_positive_mode_count
            ),
            "effective_positive_mode_count_range": (
                self.effective_positive_mode_count_range
            ),
            "mean_tail_share_outside_top_three": (
                self.mean_tail_share_outside_top_three
            ),
            "center_tail_share_outside_top_three": (
                self.center_tail_share_outside_top_three
            ),
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    aligned_lift_family_random_states: tuple[int, ...]
    target_grid_label: str
    threshold_crossing_random_state: int
    primary_boundary_random_state: int
    secondary_support_random_state: int
    threshold_crossing_focus: (
        Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankFocus
    )
    primary_boundary_focus: (
        Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankFocus
    )
    secondary_support_focus: (
        Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankFocus
    )
    mean_effective_positive_mode_count_gap_to_primary_boundary: float
    mean_tail_share_outside_top_three_gap_to_primary_boundary: float
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.aligned_lift_family_random_states = tuple(
            int(value) for value in self.aligned_lift_family_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.threshold_crossing_random_state = int(self.threshold_crossing_random_state)
        self.primary_boundary_random_state = int(self.primary_boundary_random_state)
        self.secondary_support_random_state = int(self.secondary_support_random_state)
        self.mean_effective_positive_mode_count_gap_to_primary_boundary = float(
            self.mean_effective_positive_mode_count_gap_to_primary_boundary
        )
        self.mean_tail_share_outside_top_three_gap_to_primary_boundary = float(
            self.mean_tail_share_outside_top_three_gap_to_primary_boundary
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "aligned_lift_family_random_states": list(
                self.aligned_lift_family_random_states
            ),
            "target_grid_label": self.target_grid_label,
            "threshold_crossing_random_state": self.threshold_crossing_random_state,
            "primary_boundary_random_state": self.primary_boundary_random_state,
            "secondary_support_random_state": self.secondary_support_random_state,
            "threshold_crossing_focus": self.threshold_crossing_focus.to_dict(),
            "primary_boundary_focus": self.primary_boundary_focus.to_dict(),
            "secondary_support_focus": self.secondary_support_focus.to_dict(),
            "mean_effective_positive_mode_count_gap_to_primary_boundary": (
                self.mean_effective_positive_mode_count_gap_to_primary_boundary
            ),
            "mean_tail_share_outside_top_three_gap_to_primary_boundary": (
                self.mean_tail_share_outside_top_three_gap_to_primary_boundary
            ),
            "recommendation_rationale": self.recommendation_rationale,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceFocus:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    driver_signature: str
    spectral_effective_rank_signature: str
    dominant_mode_persistence_signature: str
    persistent_dominant_mode_index: int
    dominant_mode_index_sequence: tuple[int, ...]
    dominant_mode_switch_count: int
    mean_second_mode_share: float
    center_second_mode_share: float
    second_mode_share_range: float
    mean_nonleading_mass: float
    center_nonleading_mass: float
    nonleading_mass_range: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.driver_signature = str(self.driver_signature).strip()
        self.spectral_effective_rank_signature = str(
            self.spectral_effective_rank_signature
        ).strip()
        self.dominant_mode_persistence_signature = str(
            self.dominant_mode_persistence_signature
        ).strip()
        self.persistent_dominant_mode_index = int(self.persistent_dominant_mode_index)
        self.dominant_mode_index_sequence = tuple(
            int(value) for value in self.dominant_mode_index_sequence
        )
        self.dominant_mode_switch_count = int(self.dominant_mode_switch_count)
        self.mean_second_mode_share = float(self.mean_second_mode_share)
        self.center_second_mode_share = float(self.center_second_mode_share)
        self.second_mode_share_range = float(self.second_mode_share_range)
        self.mean_nonleading_mass = float(self.mean_nonleading_mass)
        self.center_nonleading_mass = float(self.center_nonleading_mass)
        self.nonleading_mass_range = float(self.nonleading_mass_range)

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "driver_signature": self.driver_signature,
            "spectral_effective_rank_signature": (
                self.spectral_effective_rank_signature
            ),
            "dominant_mode_persistence_signature": (
                self.dominant_mode_persistence_signature
            ),
            "persistent_dominant_mode_index": self.persistent_dominant_mode_index,
            "dominant_mode_index_sequence": list(self.dominant_mode_index_sequence),
            "dominant_mode_switch_count": self.dominant_mode_switch_count,
            "mean_second_mode_share": self.mean_second_mode_share,
            "center_second_mode_share": self.center_second_mode_share,
            "second_mode_share_range": self.second_mode_share_range,
            "mean_nonleading_mass": self.mean_nonleading_mass,
            "center_nonleading_mass": self.center_nonleading_mass,
            "nonleading_mass_range": self.nonleading_mass_range,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    aligned_lift_family_random_states: tuple[int, ...]
    target_grid_label: str
    threshold_crossing_random_state: int
    primary_boundary_random_state: int
    secondary_support_random_state: int
    threshold_crossing_focus: (
        Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceFocus
    )
    primary_boundary_focus: (
        Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceFocus
    )
    secondary_support_focus: (
        Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceFocus
    )
    mean_second_mode_share_gap_to_primary_boundary: float
    mean_nonleading_mass_gap_to_primary_boundary: float
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.aligned_lift_family_random_states = tuple(
            int(value) for value in self.aligned_lift_family_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.threshold_crossing_random_state = int(self.threshold_crossing_random_state)
        self.primary_boundary_random_state = int(self.primary_boundary_random_state)
        self.secondary_support_random_state = int(self.secondary_support_random_state)
        self.mean_second_mode_share_gap_to_primary_boundary = float(
            self.mean_second_mode_share_gap_to_primary_boundary
        )
        self.mean_nonleading_mass_gap_to_primary_boundary = float(
            self.mean_nonleading_mass_gap_to_primary_boundary
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "aligned_lift_family_random_states": list(
                self.aligned_lift_family_random_states
            ),
            "target_grid_label": self.target_grid_label,
            "threshold_crossing_random_state": self.threshold_crossing_random_state,
            "primary_boundary_random_state": self.primary_boundary_random_state,
            "secondary_support_random_state": self.secondary_support_random_state,
            "threshold_crossing_focus": self.threshold_crossing_focus.to_dict(),
            "primary_boundary_focus": self.primary_boundary_focus.to_dict(),
            "secondary_support_focus": self.secondary_support_focus.to_dict(),
            "mean_second_mode_share_gap_to_primary_boundary": (
                self.mean_second_mode_share_gap_to_primary_boundary
            ),
            "mean_nonleading_mass_gap_to_primary_boundary": (
                self.mean_nonleading_mass_gap_to_primary_boundary
            ),
            "recommendation_rationale": self.recommendation_rationale,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginFocus:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    driver_signature: str
    spectral_effective_rank_signature: str
    dominant_mode_persistence_signature: str
    dominant_mode_margin_signature: str
    persistent_dominant_mode_index: int
    mean_dominant_share: float
    center_dominant_share: float
    mean_second_mode_share: float
    center_second_mode_share: float
    mean_nonleading_mass: float
    center_nonleading_mass: float
    mean_dominant_minus_second_gap: float
    center_dominant_minus_second_gap: float
    mean_dominant_to_second_ratio: float
    center_dominant_to_second_ratio: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.driver_signature = str(self.driver_signature).strip()
        self.spectral_effective_rank_signature = str(
            self.spectral_effective_rank_signature
        ).strip()
        self.dominant_mode_persistence_signature = str(
            self.dominant_mode_persistence_signature
        ).strip()
        self.dominant_mode_margin_signature = str(
            self.dominant_mode_margin_signature
        ).strip()
        self.persistent_dominant_mode_index = int(self.persistent_dominant_mode_index)
        self.mean_dominant_share = float(self.mean_dominant_share)
        self.center_dominant_share = float(self.center_dominant_share)
        self.mean_second_mode_share = float(self.mean_second_mode_share)
        self.center_second_mode_share = float(self.center_second_mode_share)
        self.mean_nonleading_mass = float(self.mean_nonleading_mass)
        self.center_nonleading_mass = float(self.center_nonleading_mass)
        self.mean_dominant_minus_second_gap = float(self.mean_dominant_minus_second_gap)
        self.center_dominant_minus_second_gap = float(
            self.center_dominant_minus_second_gap
        )
        self.mean_dominant_to_second_ratio = float(self.mean_dominant_to_second_ratio)
        self.center_dominant_to_second_ratio = float(
            self.center_dominant_to_second_ratio
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "driver_signature": self.driver_signature,
            "spectral_effective_rank_signature": (
                self.spectral_effective_rank_signature
            ),
            "dominant_mode_persistence_signature": (
                self.dominant_mode_persistence_signature
            ),
            "dominant_mode_margin_signature": self.dominant_mode_margin_signature,
            "persistent_dominant_mode_index": self.persistent_dominant_mode_index,
            "mean_dominant_share": self.mean_dominant_share,
            "center_dominant_share": self.center_dominant_share,
            "mean_second_mode_share": self.mean_second_mode_share,
            "center_second_mode_share": self.center_second_mode_share,
            "mean_nonleading_mass": self.mean_nonleading_mass,
            "center_nonleading_mass": self.center_nonleading_mass,
            "mean_dominant_minus_second_gap": (self.mean_dominant_minus_second_gap),
            "center_dominant_minus_second_gap": (self.center_dominant_minus_second_gap),
            "mean_dominant_to_second_ratio": (self.mean_dominant_to_second_ratio),
            "center_dominant_to_second_ratio": (self.center_dominant_to_second_ratio),
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    aligned_lift_family_random_states: tuple[int, ...]
    target_grid_label: str
    threshold_crossing_random_state: int
    primary_boundary_random_state: int
    secondary_support_random_state: int
    threshold_crossing_focus: (
        Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginFocus
    )
    primary_boundary_focus: (
        Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginFocus
    )
    secondary_support_focus: (
        Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginFocus
    )
    mean_dominant_minus_second_gap_gap_to_primary_boundary: float
    mean_dominant_to_second_ratio_gap_to_primary_boundary: float
    center_dominant_to_second_ratio_gap_to_primary_boundary: float
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.aligned_lift_family_random_states = tuple(
            int(value) for value in self.aligned_lift_family_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.threshold_crossing_random_state = int(self.threshold_crossing_random_state)
        self.primary_boundary_random_state = int(self.primary_boundary_random_state)
        self.secondary_support_random_state = int(self.secondary_support_random_state)
        self.mean_dominant_minus_second_gap_gap_to_primary_boundary = float(
            self.mean_dominant_minus_second_gap_gap_to_primary_boundary
        )
        self.mean_dominant_to_second_ratio_gap_to_primary_boundary = float(
            self.mean_dominant_to_second_ratio_gap_to_primary_boundary
        )
        self.center_dominant_to_second_ratio_gap_to_primary_boundary = float(
            self.center_dominant_to_second_ratio_gap_to_primary_boundary
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "aligned_lift_family_random_states": list(
                self.aligned_lift_family_random_states
            ),
            "target_grid_label": self.target_grid_label,
            "threshold_crossing_random_state": self.threshold_crossing_random_state,
            "primary_boundary_random_state": self.primary_boundary_random_state,
            "secondary_support_random_state": self.secondary_support_random_state,
            "threshold_crossing_focus": self.threshold_crossing_focus.to_dict(),
            "primary_boundary_focus": self.primary_boundary_focus.to_dict(),
            "secondary_support_focus": self.secondary_support_focus.to_dict(),
            "mean_dominant_minus_second_gap_gap_to_primary_boundary": (
                self.mean_dominant_minus_second_gap_gap_to_primary_boundary
            ),
            "mean_dominant_to_second_ratio_gap_to_primary_boundary": (
                self.mean_dominant_to_second_ratio_gap_to_primary_boundary
            ),
            "center_dominant_to_second_ratio_gap_to_primary_boundary": (
                self.center_dominant_to_second_ratio_gap_to_primary_boundary
            ),
            "recommendation_rationale": self.recommendation_rationale,
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackFocus:
    dgp_name: str
    focus_label: str
    random_state: int
    replication_seed: int
    dominant_mode_margin_signature: str
    mean_dominant_minus_second_gap: float
    mean_dominant_to_second_ratio: float
    center_dominant_to_second_ratio: float
    collapse_gap_slack: float
    collapse_ratio_slack: float
    collapse_center_ratio_slack: float
    finite_gap_slack: float
    finite_ratio_slack: float

    def __post_init__(self) -> None:
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.focus_label = str(self.focus_label).strip()
        self.random_state = int(self.random_state)
        self.replication_seed = int(self.replication_seed)
        self.dominant_mode_margin_signature = str(
            self.dominant_mode_margin_signature
        ).strip()
        self.mean_dominant_minus_second_gap = float(self.mean_dominant_minus_second_gap)
        self.mean_dominant_to_second_ratio = float(self.mean_dominant_to_second_ratio)
        self.center_dominant_to_second_ratio = float(
            self.center_dominant_to_second_ratio
        )
        self.collapse_gap_slack = float(self.collapse_gap_slack)
        self.collapse_ratio_slack = float(self.collapse_ratio_slack)
        self.collapse_center_ratio_slack = float(self.collapse_center_ratio_slack)
        self.finite_gap_slack = float(self.finite_gap_slack)
        self.finite_ratio_slack = float(self.finite_ratio_slack)

    def canonical_digest_line(self) -> str:
        return (
            f"- `{self.random_state}` `{self.dominant_mode_margin_signature}`: "
            f"`{self.mean_dominant_minus_second_gap:+.3f}` gap vs level, "
            f"`{self.mean_dominant_to_second_ratio:+.3f}` mean-ratio level, "
            f"`{self.center_dominant_to_second_ratio:+.3f}` center-ratio level, "
            f"collapse slack `({self.collapse_gap_slack:+.3f}, "
            f"{self.collapse_ratio_slack:+.3f}, {self.collapse_center_ratio_slack:+.3f})`, "
            f"finite slack `({self.finite_gap_slack:+.3f}, {self.finite_ratio_slack:+.3f})`"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dgp_name": self.dgp_name,
            "focus_label": self.focus_label,
            "random_state": self.random_state,
            "replication_seed": self.replication_seed,
            "dominant_mode_margin_signature": self.dominant_mode_margin_signature,
            "mean_dominant_minus_second_gap": self.mean_dominant_minus_second_gap,
            "mean_dominant_to_second_ratio": self.mean_dominant_to_second_ratio,
            "center_dominant_to_second_ratio": self.center_dominant_to_second_ratio,
            "collapse_gap_slack": self.collapse_gap_slack,
            "collapse_ratio_slack": self.collapse_ratio_slack,
            "collapse_center_ratio_slack": self.collapse_center_ratio_slack,
            "finite_gap_slack": self.finite_gap_slack,
            "finite_ratio_slack": self.finite_ratio_slack,
            "canonical_digest_line": self.canonical_digest_line(),
        }


@dataclass(slots=True)
class Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackReport:
    oracle_lane: str
    stage_label: str
    target_n_obs: int
    reference_n_obs: int
    p: int
    dgp_name: str
    aligned_lift_family_random_states: tuple[int, ...]
    target_grid_label: str
    collapse_gap_threshold: float
    collapse_ratio_threshold: float
    collapse_center_ratio_threshold: float
    finite_gap_threshold: float
    finite_ratio_threshold: float
    threshold_crossing_focus: (
        Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackFocus
    )
    primary_boundary_focus: (
        Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackFocus
    )
    secondary_support_focus: (
        Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackFocus
    )
    canonical_slack_digest: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.stage_label = str(self.stage_label).strip()
        self.target_n_obs = int(self.target_n_obs)
        self.reference_n_obs = int(self.reference_n_obs)
        self.p = int(self.p)
        self.dgp_name = str(self.dgp_name).strip().upper()
        self.aligned_lift_family_random_states = tuple(
            int(value) for value in self.aligned_lift_family_random_states
        )
        self.target_grid_label = str(self.target_grid_label).strip()
        self.collapse_gap_threshold = float(self.collapse_gap_threshold)
        self.collapse_ratio_threshold = float(self.collapse_ratio_threshold)
        self.collapse_center_ratio_threshold = float(
            self.collapse_center_ratio_threshold
        )
        self.finite_gap_threshold = float(self.finite_gap_threshold)
        self.finite_ratio_threshold = float(self.finite_ratio_threshold)
        self.canonical_slack_digest = tuple(
            str(line).rstrip() for line in self.canonical_slack_digest
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "stage_label": self.stage_label,
            "target_n_obs": self.target_n_obs,
            "reference_n_obs": self.reference_n_obs,
            "p": self.p,
            "dgp_name": self.dgp_name,
            "aligned_lift_family_random_states": list(
                self.aligned_lift_family_random_states
            ),
            "target_grid_label": self.target_grid_label,
            "collapse_gap_threshold": self.collapse_gap_threshold,
            "collapse_ratio_threshold": self.collapse_ratio_threshold,
            "collapse_center_ratio_threshold": self.collapse_center_ratio_threshold,
            "finite_gap_threshold": self.finite_gap_threshold,
            "finite_ratio_threshold": self.finite_ratio_threshold,
            "threshold_crossing_focus": self.threshold_crossing_focus.to_dict(),
            "primary_boundary_focus": self.primary_boundary_focus.to_dict(),
            "secondary_support_focus": self.secondary_support_focus.to_dict(),
            "canonical_slack_digest": list(self.canonical_slack_digest),
            "recommendation_rationale": self.recommendation_rationale,
        }


@dataclass(slots=True)
class EvaluationGridAliasProbe:
    evaluation_grid: np.ndarray
    basis_row_rank: int
    unique_basis_row_count: int
    aliased_index_groups: tuple[tuple[int, ...], ...]
    sigma_z_squared: np.ndarray

    def __post_init__(self) -> None:
        self.evaluation_grid = _coerce_evaluation_grid(self.evaluation_grid)
        self.basis_row_rank = int(self.basis_row_rank)
        self.unique_basis_row_count = int(self.unique_basis_row_count)
        self.aliased_index_groups = tuple(
            tuple(int(index) for index in group) for group in self.aliased_index_groups
        )
        self.sigma_z_squared = _coerce_evaluation_grid(self.sigma_z_squared)

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_grid": self.evaluation_grid.astype(float).tolist(),
            "basis_row_rank": self.basis_row_rank,
            "unique_basis_row_count": self.unique_basis_row_count,
            "aliased_index_groups": [
                list(group) for group in self.aliased_index_groups
            ],
            "sigma_z_squared": self.sigma_z_squared.astype(float).tolist(),
        }


@dataclass(slots=True)
class Phase7NonparametricAliasProbe:
    monte_carlo_random_state: int
    replication_seed: int
    design: MonteCarloDesign
    current_grid: EvaluationGridAliasProbe
    contrast_grid: EvaluationGridAliasProbe
    integer_grid: EvaluationGridAliasProbe
    omega_f_min_eigenvalue: float
    v_f_min_eigenvalue: float

    def __post_init__(self) -> None:
        self.monte_carlo_random_state = int(self.monte_carlo_random_state)
        self.replication_seed = int(self.replication_seed)
        self.omega_f_min_eigenvalue = float(self.omega_f_min_eigenvalue)
        self.v_f_min_eigenvalue = float(self.v_f_min_eigenvalue)

    def to_dict(self) -> dict[str, object]:
        return {
            "monte_carlo_random_state": self.monte_carlo_random_state,
            "replication_seed": self.replication_seed,
            "design": self.design.staged_subset_entry(),
            "current_grid": self.current_grid.to_dict(),
            "contrast_grid": self.contrast_grid.to_dict(),
            "integer_grid": self.integer_grid.to_dict(),
            "omega_f_min_eigenvalue": self.omega_f_min_eigenvalue,
            "v_f_min_eigenvalue": self.v_f_min_eigenvalue,
        }


@dataclass(slots=True)
class EmpiricalAssetAudit:
    repo_root: str
    oracle_lane: str
    status: str
    blocker_reason: str | None
    missing_assets: tuple[str, ...]
    raw_data_assets: tuple[str, ...]
    loader_assets: tuple[str, ...]
    known_figures: tuple[str, ...]
    time_window: str
    treated_state_count: int
    excluded_states: tuple[str, ...]
    linear_covariate_count: int
    z_lanes: tuple[str, ...]
    basis_family: str
    basis_degree: int
    confidence_level: float
    synthetic_fallback_enabled: bool = False
    figure_only_mode_enabled: bool = False
    disallowed_shortcuts: tuple[str, ...] = _SECTION6_DISALLOWED_SHORTCUTS

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "oracle_lane": self.oracle_lane,
            "status": self.status,
            "blocker_reason": self.blocker_reason,
            "missing_assets": list(self.missing_assets),
            "raw_data_assets": list(self.raw_data_assets),
            "loader_assets": list(self.loader_assets),
            "known_figures": list(self.known_figures),
            "time_window": self.time_window,
            "treated_state_count": self.treated_state_count,
            "excluded_states": list(self.excluded_states),
            "linear_covariate_count": self.linear_covariate_count,
            "z_lanes": list(self.z_lanes),
            "basis_family": self.basis_family,
            "basis_degree": self.basis_degree,
            "confidence_level": self.confidence_level,
            "synthetic_fallback_enabled": self.synthetic_fallback_enabled,
            "figure_only_mode_enabled": self.figure_only_mode_enabled,
            "disallowed_shortcuts": list(self.disallowed_shortcuts),
        }


def _normalize_validation_lane(oracle_lane: str) -> str:
    candidate = str(oracle_lane).strip().lower().replace("_", "-")
    canonical = _VALIDATION_LANE_ALIASES.get(candidate)
    if canonical is None:
        raise ValueError(f"unsupported validation lane: {oracle_lane!r}")
    if canonical == "paper-trigonometric":
        return normalize_oracle_lane(
            basis_family="trigonometric",
            oracle_lane=canonical,
        )
    if canonical == "r-parity-polynomial":
        return normalize_oracle_lane(
            basis_family="polynomial",
            oracle_lane=canonical,
        )
    return canonical


def _normalize_string_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


def _normalize_string_mapping(values: dict[str, str] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in (values or {}).items():
        normalized_key = str(key).strip()
        normalized_value = str(value).strip()
        if normalized_key:
            normalized[normalized_key] = normalized_value
    return dict(sorted(normalized.items()))


def _normalize_status(status: str) -> str:
    normalized = str(status).strip().lower().replace("_", "-")
    if normalized not in _VALID_LANE_STATUSES:
        raise ValueError(f"unsupported validation lane status: {status!r}")
    return normalized


def _normalize_invalidity_counts(
    counts: dict[str, int] | None,
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in (counts or {}).items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("typed invalidity count names must be non-empty")
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"typed invalidity count {normalized_key} must be an integer"
            )
        normalized_count = int(value)
        if normalized_count < 0:
            raise ValueError(
                f"typed invalidity count {normalized_key} must be non-negative"
            )
        if normalized_count:
            normalized[normalized_key] = normalized_count
    return dict(sorted(normalized.items()))


def _merge_invalidity_counts(
    count_maps: Sequence[dict[str, int]],
) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for mapping in count_maps:
        merged.update(_normalize_invalidity_counts(mapping))
    return dict(sorted(merged.items()))


def _normalize_metadata_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _normalize_metadata_value(value.tolist())
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, nested_value in value.items():
            normalized_key = str(key).strip()
            if normalized_key:
                normalized[normalized_key] = _normalize_metadata_value(nested_value)
        return dict(sorted(normalized.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_metadata_value(item) for item in value)
    return value


def _normalize_invalidity_examples(
    examples: dict[str, dict[str, object]] | None,
) -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    for error_name, metadata in (examples or {}).items():
        normalized_error_name = str(error_name).strip()
        if not normalized_error_name:
            continue
        normalized[normalized_error_name] = dict(
            _normalize_metadata_value(dict(metadata or {}))
        )
    return dict(sorted(normalized.items()))


def _merge_invalidity_examples(
    example_maps: Sequence[dict[str, dict[str, object]]],
) -> dict[str, dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for mapping in example_maps:
        for error_name, metadata in _normalize_invalidity_examples(mapping).items():
            merged.setdefault(error_name, metadata)
    return dict(sorted(merged.items()))


def _serialize_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _serialize_metadata_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, tuple):
        return [_serialize_metadata_value(item) for item in value]
    return value


def _serialize_invalidity_examples(
    examples: dict[str, dict[str, object]] | None,
) -> dict[str, dict[str, object]]:
    return {
        error_name: dict(_serialize_metadata_value(metadata))
        for error_name, metadata in _normalize_invalidity_examples(examples).items()
    }


def _capture_invalidity_example(
    error: (
        ZeroValidHoldoutError
        | NuisanceTrainingSupportError
        | Eq31ProjectionRankError
        | InferenceComputationError
    ),
) -> dict[str, object]:
    if isinstance(error, ZeroValidHoldoutError):
        return {
            "failure_kind": "zero-valid-holdout",
            "fold_id": error.fold_id,
            "trim_lower": error.trim_lower,
            "trim_upper": error.trim_upper,
        }
    if isinstance(error, NuisanceTrainingSupportError):
        return {
            "failure_kind": "nuisance-training-support",
            "fold_id": error.fold_id,
            "n_train_treated": error.n_train_treated,
            "n_train_control": error.n_train_control,
        }
    return _normalize_invalidity_examples(
        {type(error).__name__: getattr(error, "metadata", {})}
    ).get(type(error).__name__, {})


def _monte_carlo_invalidity_example(
    error: (
        ZeroValidHoldoutError
        | NuisanceTrainingSupportError
        | Eq31ProjectionRankError
        | InferenceComputationError
    ),
    *,
    design: MonteCarloDesign,
    monte_carlo_random_state: int | None,
    replication_seed: int,
) -> dict[str, object]:
    return _normalize_invalidity_examples(
        {
            type(error).__name__: {
                **_capture_invalidity_example(error),
                "monte_carlo_random_state": (
                    None
                    if monte_carlo_random_state is None
                    else int(monte_carlo_random_state)
                ),
                "replication_seed": int(replication_seed),
                "design": {
                    "dgp_name": design.dgp_name,
                    "n_obs": int(design.n_obs),
                    "p": int(design.p),
                },
            }
        }
    ).get(type(error).__name__, {})


def _merge_string_sequences(sequences: Sequence[Sequence[str]]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for sequence in sequences:
        for value in _normalize_string_tuple(sequence):
            if value not in seen:
                merged.append(value)
                seen.add(value)
    return tuple(merged)


def _canonicalize_basis_row(
    row: np.ndarray,
    *,
    decimals: int = 12,
) -> tuple[float, ...]:
    rounded = np.round(np.asarray(row, dtype=float), decimals=decimals)
    rounded[np.abs(rounded) <= 10.0 ** (-decimals)] = 0.0
    return tuple(float(value) for value in rounded.tolist())


def _group_aliased_basis_rows(
    basis_matrix: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    groups: dict[tuple[float, ...], list[int]] = {}
    for index, row in enumerate(np.asarray(basis_matrix, dtype=float)):
        groups.setdefault(_canonicalize_basis_row(row), []).append(int(index))
    return tuple(
        tuple(group) for _, group in sorted(groups.items(), key=lambda item: item[1][0])
    )


def _minimum_symmetric_eigenvalue(matrix: np.ndarray) -> float:
    symmetric = 0.5 * (
        np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T
    )
    return float(np.min(np.linalg.eigvalsh(symmetric)))


def _build_evaluation_grid_alias_probe(
    *,
    evaluation_grid: np.ndarray,
    evaluation_basis: np.ndarray,
    v_f_hat: np.ndarray,
) -> EvaluationGridAliasProbe:
    sigma_z_squared = np.einsum(
        "ij,jk,ik->i",
        np.asarray(evaluation_basis, dtype=float),
        np.asarray(v_f_hat, dtype=float),
        np.asarray(evaluation_basis, dtype=float),
    )
    alias_groups = _group_aliased_basis_rows(evaluation_basis)
    return EvaluationGridAliasProbe(
        evaluation_grid=np.asarray(evaluation_grid, dtype=float),
        basis_row_rank=int(
            np.linalg.matrix_rank(np.asarray(evaluation_basis, dtype=float))
        ),
        unique_basis_row_count=len(alias_groups),
        aliased_index_groups=alias_groups,
        sigma_z_squared=np.asarray(sigma_z_squared, dtype=float),
    )


def _serialize_target_matrix(
    full_target_matrix: dict[str, tuple[int, ...] | float] | None,
) -> dict[str, object] | None:
    if full_target_matrix is None:
        return None
    serialized: dict[str, object] = {}
    for key, value in full_target_matrix.items():
        if isinstance(value, tuple):
            serialized[key] = list(value)
        else:
            serialized[key] = value
    return serialized


@dataclass(slots=True)
class ParityValidationRoute:
    oracle_lane: str
    evidence_scope: str
    status: str
    primary_oracle: str
    source_classification: str
    allowed_comparisons: tuple[str, ...] = ()
    blocked_comparisons: tuple[str, ...] = ()
    blocker_reason: str | None = None
    typed_invalidity_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        if self.oracle_lane != "r-parity-polynomial":
            raise ValueError(
                "ParityValidationRoute oracle_lane must be r-parity-polynomial"
            )
        self.evidence_scope = str(self.evidence_scope).strip()
        self.status = _normalize_status(self.status)
        self.primary_oracle = str(self.primary_oracle).strip()
        self.source_classification = str(self.source_classification).strip()
        self.allowed_comparisons = _normalize_string_tuple(self.allowed_comparisons)
        self.blocked_comparisons = _normalize_string_tuple(self.blocked_comparisons)
        self.blocker_reason = (
            None if self.blocker_reason is None else str(self.blocker_reason).strip()
        )
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "evidence_scope": self.evidence_scope,
            "status": self.status,
            "primary_oracle": self.primary_oracle,
            "source_classification": self.source_classification,
            "allowed_comparisons": list(self.allowed_comparisons),
            "blocked_comparisons": list(self.blocked_comparisons),
            "blocker_reason": self.blocker_reason,
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
        }


@dataclass(slots=True)
class ValidationLaneSummary:
    oracle_lane: str
    source_kind: str
    status: str
    primary_oracle: str
    allowed_comparisons: tuple[str, ...] = ()
    blocked_comparisons: tuple[str, ...] = ()
    typed_invalidity_counts: dict[str, int] = field(default_factory=dict)
    blocker_reason: str | None = None
    staged_matrix: bool | None = None
    full_target_matrix: dict[str, tuple[int, ...] | float] | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.source_kind = str(self.source_kind).strip()
        self.status = _normalize_status(self.status)
        self.primary_oracle = str(self.primary_oracle).strip()
        self.allowed_comparisons = _normalize_string_tuple(self.allowed_comparisons)
        self.blocked_comparisons = _normalize_string_tuple(self.blocked_comparisons)
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )
        self.blocker_reason = (
            None if self.blocker_reason is None else str(self.blocker_reason).strip()
        )
        self.notes = _normalize_string_tuple(self.notes)
        if self.full_target_matrix is not None:
            self.full_target_matrix = dict(self.full_target_matrix)

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "source_kind": self.source_kind,
            "status": self.status,
            "primary_oracle": self.primary_oracle,
            "allowed_comparisons": list(self.allowed_comparisons),
            "blocked_comparisons": list(self.blocked_comparisons),
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "blocker_reason": self.blocker_reason,
            "staged_matrix": self.staged_matrix,
            "full_target_matrix": _serialize_target_matrix(self.full_target_matrix),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class ValidationMatrixReport:
    stage_label: str
    staged_matrix: bool
    full_target_matrix: dict[str, tuple[int, ...] | float] | None
    lane_summaries: tuple[ValidationLaneSummary, ...]
    typed_invalidity_counts: dict[str, int]
    empirical_blocker_reason: str | None = None
    release_contract_surfaces: tuple[str, ...] = ()

    @property
    def lane_keys(self) -> tuple[str, ...]:
        return tuple(lane.oracle_lane for lane in self.lane_summaries)

    def lane(self, oracle_lane: str) -> ValidationLaneSummary:
        canonical = _normalize_validation_lane(oracle_lane)
        for lane_summary in self.lane_summaries:
            if lane_summary.oracle_lane == canonical:
                return lane_summary
        raise KeyError(f"validation lane not present: {oracle_lane!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "staged_matrix": self.staged_matrix,
            "full_target_matrix": _serialize_target_matrix(self.full_target_matrix),
            "lane_keys": list(self.lane_keys),
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "empirical_blocker_reason": self.empirical_blocker_reason,
            "release_contract_surfaces": list(self.release_contract_surfaces),
            "lanes": {
                lane_summary.oracle_lane: lane_summary.to_dict()
                for lane_summary in self.lane_summaries
            },
        }


@dataclass(slots=True)
class Phase7ValidationDebtLane:
    oracle_lane: str
    status: str
    evidence_floor: tuple[str, ...]
    promotion_gate: str
    blocker_reason: str | None = None
    typed_invalidity_counts: dict[str, int] = field(default_factory=dict)
    limited_surfaces: dict[str, str] = field(default_factory=dict)
    governing_artifacts: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.oracle_lane = _normalize_validation_lane(self.oracle_lane)
        self.status = _normalize_status(self.status)
        self.evidence_floor = _normalize_string_tuple(self.evidence_floor)
        self.promotion_gate = str(self.promotion_gate).strip()
        self.blocker_reason = (
            None if self.blocker_reason is None else str(self.blocker_reason).strip()
        )
        self.typed_invalidity_counts = _normalize_invalidity_counts(
            self.typed_invalidity_counts
        )
        self.limited_surfaces = _normalize_string_mapping(self.limited_surfaces)
        self.governing_artifacts = _normalize_string_tuple(self.governing_artifacts)
        self.missing_evidence = _normalize_string_tuple(self.missing_evidence)
        self.notes = _normalize_string_tuple(self.notes)

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_lane": self.oracle_lane,
            "status": self.status,
            "evidence_floor": list(self.evidence_floor),
            "promotion_gate": self.promotion_gate,
            "blocker_reason": self.blocker_reason,
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "limited_surfaces": dict(self.limited_surfaces),
            "governing_artifacts": list(self.governing_artifacts),
            "missing_evidence": list(self.missing_evidence),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class Phase7ValidationDebtReport:
    stage_label: str
    staged_matrix: bool
    full_target_matrix: dict[str, tuple[int, ...] | float] | None
    lane_summaries: tuple[Phase7ValidationDebtLane, ...]
    typed_invalidity_counts: dict[str, int]
    empirical_blocker_reason: str | None = None

    @property
    def lane_keys(self) -> tuple[str, ...]:
        return tuple(lane.oracle_lane for lane in self.lane_summaries)

    def lane(self, oracle_lane: str) -> Phase7ValidationDebtLane:
        canonical = _normalize_validation_lane(oracle_lane)
        for lane_summary in self.lane_summaries:
            if lane_summary.oracle_lane == canonical:
                return lane_summary
        raise KeyError(f"validation lane not present: {oracle_lane!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "staged_matrix": self.staged_matrix,
            "full_target_matrix": _serialize_target_matrix(self.full_target_matrix),
            "lane_keys": list(self.lane_keys),
            "typed_invalidity_counts": dict(self.typed_invalidity_counts),
            "empirical_blocker_reason": self.empirical_blocker_reason,
            "lanes": {
                lane_summary.oracle_lane: lane_summary.to_dict()
                for lane_summary in self.lane_summaries
            },
        }


def _resolve_parity_lane_status(statuses: Sequence[str]) -> str:
    normalized = {_normalize_status(status) for status in statuses}
    if "blocked" in normalized:
        return "blocked"
    if "validated" in normalized:
        return "validated"
    if "sanity-check" in normalized:
        return "sanity-check"
    if not normalized:
        return "blocked"
    return sorted(normalized)[0]


def _validate_paper_monte_carlo_lane(lanes: set[str]) -> None:
    if lanes != {"paper-trigonometric"}:
        raise ValueError(
            "Monte Carlo validation must stay on the paper-trigonometric lane"
        )


def _build_monte_carlo_lane_summary(
    monte_carlo_report: MonteCarloSmokeReport,
) -> ValidationLaneSummary:
    if not monte_carlo_report.summaries:
        raise ValueError("Monte Carlo validation requires at least one summary")
    lanes = {summary.design.oracle_lane for summary in monte_carlo_report.summaries}
    _validate_paper_monte_carlo_lane(lanes)
    typed_invalidity_counts = _merge_invalidity_counts(
        [summary.typed_invalidity_counts for summary in monte_carlo_report.summaries]
    )
    return ValidationLaneSummary(
        oracle_lane="paper-trigonometric",
        source_kind="monte-carlo-smoke",
        status="staged-smoke" if monte_carlo_report.staging else "validated",
        primary_oracle="paper Section 5 Monte Carlo contract",
        allowed_comparisons=(
            "paper Section 5 Monte Carlo metrics",
            "typed invalidity accounting",
        ),
        blocked_comparisons=(
            "R snapshot helper parity",
            "real-data empirical reproduction",
        ),
        typed_invalidity_counts=typed_invalidity_counts,
        staged_matrix=bool(monte_carlo_report.staging),
        full_target_matrix=dict(monte_carlo_report.full_target_matrix),
        notes=(
            f"stage_label={monte_carlo_report.stage_label}",
            f"design_count={len(monte_carlo_report.summaries)}",
        ),
    )


def _build_parity_lane_summary(
    parity_routes: Sequence[ParityValidationRoute],
) -> ValidationLaneSummary | None:
    routes = tuple(parity_routes)
    if not routes:
        return None
    return ValidationLaneSummary(
        oracle_lane="r-parity-polynomial",
        source_kind="r-parity-routing",
        status=_resolve_parity_lane_status([route.status for route in routes]),
        primary_oracle="; ".join(
            _merge_string_sequences([[route.primary_oracle] for route in routes])
        ),
        allowed_comparisons=_merge_string_sequences(
            [route.allowed_comparisons for route in routes]
        ),
        blocked_comparisons=_merge_string_sequences(
            [route.blocked_comparisons for route in routes]
        ),
        typed_invalidity_counts=_merge_invalidity_counts(
            [route.typed_invalidity_counts for route in routes]
        ),
        blocker_reason=next(
            (route.blocker_reason for route in routes if route.blocker_reason),
            None,
        ),
        staged_matrix=False,
        notes=_merge_string_sequences(
            [
                (
                    f"evidence_scope={route.evidence_scope}",
                    f"source_classification={route.source_classification}",
                )
                for route in routes
            ]
        ),
    )


def _build_empirical_lane_summary(
    empirical_asset_audit: EmpiricalAssetAudit,
) -> ValidationLaneSummary:
    if empirical_asset_audit.oracle_lane != _SECTION6_ORACLE_LANE:
        raise ValueError(
            "Empirical asset audit must stay on the real-data-asset-audit lane"
        )
    if (
        empirical_asset_audit.blocker_reason is not None
        and empirical_asset_audit.status != "blocked"
    ):
        raise ValueError(
            "Empirical asset audit with blocker_reason must remain blocked"
        )
    blocked_comparisons: tuple[str, ...] = ()
    if empirical_asset_audit.status == "blocked":
        blocked_comparisons = (
            "parity success",
            "paper Monte Carlo validation",
        )
    return ValidationLaneSummary(
        oracle_lane=empirical_asset_audit.oracle_lane,
        source_kind="empirical-asset-audit",
        status=empirical_asset_audit.status,
        primary_oracle="paper Section 6 schema + local asset audit",
        allowed_comparisons=(
            "Section 6 schema readiness",
            "known figure inventory",
        ),
        blocked_comparisons=blocked_comparisons,
        blocker_reason=empirical_asset_audit.blocker_reason,
        staged_matrix=False,
        notes=(
            f"missing_assets={','.join(empirical_asset_audit.missing_assets) or 'none'}",
            f"z_lanes={','.join(empirical_asset_audit.z_lanes)}",
        ),
    )


def build_validation_matrix_report(
    *,
    monte_carlo_report: MonteCarloSmokeReport | None,
    parity_routes: Sequence[ParityValidationRoute] = (),
    empirical_asset_audit: EmpiricalAssetAudit | None,
    release_contract_surfaces: Sequence[str] = (
        "run_phase7_release_matrix_runner(...)",
        "run_phase7_monte_carlo_paper_dgp2_contract_audit(...)",
    ),
) -> ValidationMatrixReport:
    lane_summaries: list[ValidationLaneSummary] = []
    if monte_carlo_report is not None:
        lane_summaries.append(_build_monte_carlo_lane_summary(monte_carlo_report))
    parity_lane = _build_parity_lane_summary(parity_routes)
    if parity_lane is not None:
        lane_summaries.append(parity_lane)
    if empirical_asset_audit is not None:
        lane_summaries.append(_build_empirical_lane_summary(empirical_asset_audit))
    if not lane_summaries:
        raise ValueError("build_validation_matrix_report requires at least one lane")

    lane_map = {
        lane_summary.oracle_lane: lane_summary for lane_summary in lane_summaries
    }
    ordered_lane_summaries = tuple(
        lane_map[lane] for lane in _VALIDATION_LANE_ORDER if lane in lane_map
    )
    typed_invalidity_counts = _merge_invalidity_counts(
        [
            lane_summary.typed_invalidity_counts
            for lane_summary in ordered_lane_summaries
        ]
    )
    empirical_blocker_reason = None
    if empirical_asset_audit is not None:
        empirical_blocker_reason = empirical_asset_audit.blocker_reason
    return ValidationMatrixReport(
        stage_label=(
            monte_carlo_report.stage_label
            if monte_carlo_report is not None
            else "validation-matrix"
        ),
        staged_matrix=bool(monte_carlo_report.staging) if monte_carlo_report else False,
        full_target_matrix=(
            dict(monte_carlo_report.full_target_matrix)
            if monte_carlo_report is not None
            else None
        ),
        lane_summaries=ordered_lane_summaries,
        typed_invalidity_counts=typed_invalidity_counts,
        empirical_blocker_reason=empirical_blocker_reason,
        release_contract_surfaces=_normalize_string_tuple(release_contract_surfaces),
    )


def _build_phase7_paper_lane(
    lane_summary: ValidationLaneSummary,
    *,
    validation_matrix_report: ValidationMatrixReport,
) -> Phase7ValidationDebtLane:
    return Phase7ValidationDebtLane(
        oracle_lane=lane_summary.oracle_lane,
        status=lane_summary.status,
        evidence_floor=(
            "build_validation_matrix_report(...)",
            "Docs/parity/phase5_validation_matrix.md",
            "typed_invalidity_counts",
        ),
        promotion_gate=(
            "Promote beyond staged-smoke only after fresh runtime evidence and "
            "typed_invalidity_counts justify widening the paper matrix beyond "
            "reduced-smoke."
        ),
        blocker_reason=lane_summary.blocker_reason,
        typed_invalidity_counts=lane_summary.typed_invalidity_counts,
        governing_artifacts=(
            "Docs/parity/phase5_validation_matrix.md",
            "Docs/research/phase5_failure_debt_register.md",
            "Docs/research/phase7_verification_frontiers.md",
        ),
        missing_evidence=(
            "fresh runtime evidence beyond reduced-smoke",
            "promotion decision for widening the full target matrix",
        ),
        notes=lane_summary.notes
        + (f"staged_matrix={validation_matrix_report.staged_matrix}",),
    )


def _build_phase7_parity_lane(
    lane_summary: ValidationLaneSummary,
) -> Phase7ValidationDebtLane:
    return Phase7ValidationDebtLane(
        oracle_lane=lane_summary.oracle_lane,
        status=lane_summary.status,
        evidence_floor=(
            "build_validation_matrix_report(...)",
            "raw Eq. (3.1) deterministic sanity checks",
            "Docs/parity/phase5_validation_matrix.md",
        ),
        promotion_gate=(
            "Promotion beyond raw Eq. (3.1) requires paper-backed replacement "
            "evidence or source repairs that remove RBUG-005 and RBUG-013; "
            "archived R outer inference stays reference-only until then."
        ),
        blocker_reason=lane_summary.blocker_reason,
        typed_invalidity_counts=lane_summary.typed_invalidity_counts,
        limited_surfaces={
            "archived-r-outer-inference": _PHASE7_REFERENCE_ONLY,
        },
        governing_artifacts=(
            "Docs/parity/phase5_validation_matrix.md",
            "Docs/research/phase7_verification_frontiers.md",
            "RBUG-005",
            "RBUG-013",
        ),
        missing_evidence=(
            "paper-backed outer inference replacement evidence",
            "source-level proof that archived outer inference is runnable without undefined debias objects or last-fold tc reuse",
        ),
        notes=lane_summary.notes + ("archived-r-outer-inference=reference-only",),
    )


def _build_phase7_empirical_lane(
    lane_summary: ValidationLaneSummary,
) -> Phase7ValidationDebtLane:
    return Phase7ValidationDebtLane(
        oracle_lane=lane_summary.oracle_lane,
        status=lane_summary.status,
        evidence_floor=(
            "audit_section6_assets(...)",
            "Docs/parity/phase5_validation_matrix.md",
            "hddid-py/tests/validation/test_real_data_asset_gate.py",
        ),
        promotion_gate=(
            "Promote only after a local raw dataset and a runnable loader both "
            "exist, then rerun the empirical asset audit and confirm the lane "
            "returns ready."
        ),
        blocker_reason=lane_summary.blocker_reason,
        typed_invalidity_counts=lane_summary.typed_invalidity_counts,
        governing_artifacts=(
            "Docs/research/phase5_empirical_asset_audit.md",
            "Docs/research/phase7_verification_frontiers.md",
            "missing-local-dataset",
        ),
        missing_evidence=(
            "local raw dataset",
            "runnable loader",
        ),
        notes=lane_summary.notes,
    )


def _build_phase7_validation_debt_lane(
    lane_summary: ValidationLaneSummary,
    *,
    validation_matrix_report: ValidationMatrixReport,
) -> Phase7ValidationDebtLane:
    if lane_summary.oracle_lane == "paper-trigonometric":
        return _build_phase7_paper_lane(
            lane_summary,
            validation_matrix_report=validation_matrix_report,
        )
    if lane_summary.oracle_lane == "r-parity-polynomial":
        return _build_phase7_parity_lane(lane_summary)
    if lane_summary.oracle_lane == "real-data-asset-audit":
        return _build_phase7_empirical_lane(lane_summary)
    return Phase7ValidationDebtLane(
        oracle_lane=lane_summary.oracle_lane,
        status=lane_summary.status,
        evidence_floor=("build_validation_matrix_report(...)",),
        promotion_gate="No Phase 7 promotion gate has been registered for this lane.",
        blocker_reason=lane_summary.blocker_reason,
        typed_invalidity_counts=lane_summary.typed_invalidity_counts,
        notes=lane_summary.notes,
    )


def build_phase7_validation_debt_report(
    *,
    validation_matrix_report: ValidationMatrixReport | None = None,
    monte_carlo_report: MonteCarloSmokeReport | None = None,
    parity_routes: Sequence[ParityValidationRoute] = (),
    empirical_asset_audit: EmpiricalAssetAudit | None = None,
) -> Phase7ValidationDebtReport:
    if validation_matrix_report is not None and (
        monte_carlo_report is not None
        or parity_routes
        or empirical_asset_audit is not None
    ):
        raise ValueError(
            "build_phase7_validation_debt_report accepts either validation_matrix_report "
            "or raw lane inputs, not both"
        )
    if validation_matrix_report is None:
        validation_matrix_report = build_validation_matrix_report(
            monte_carlo_report=monte_carlo_report,
            parity_routes=parity_routes,
            empirical_asset_audit=empirical_asset_audit,
        )

    lane_summaries = tuple(
        _build_phase7_validation_debt_lane(
            lane_summary,
            validation_matrix_report=validation_matrix_report,
        )
        for lane_summary in validation_matrix_report.lane_summaries
    )
    return Phase7ValidationDebtReport(
        stage_label=validation_matrix_report.stage_label,
        staged_matrix=validation_matrix_report.staged_matrix,
        full_target_matrix=(
            dict(validation_matrix_report.full_target_matrix)
            if validation_matrix_report.full_target_matrix is not None
            else None
        ),
        lane_summaries=lane_summaries,
        typed_invalidity_counts=dict(validation_matrix_report.typed_invalidity_counts),
        empirical_blocker_reason=validation_matrix_report.empirical_blocker_reason,
    )


def build_monte_carlo_widening_readiness_report(
    monte_carlo_report: MonteCarloSmokeReport,
) -> MonteCarloWideningReadinessReport:
    summaries = tuple(monte_carlo_report.summaries)
    if not summaries:
        raise ValueError(
            "build_monte_carlo_widening_readiness_report requires at least one summary"
        )
    _validate_paper_monte_carlo_lane(
        {summary.design.oracle_lane for summary in summaries}
    )
    total_replications = sum(summary.n_replications for summary in summaries)
    total_successful_replications = sum(
        summary.n_successful_replications for summary in summaries
    )
    total_runtime_seconds = monte_carlo_report.total_runtime_seconds
    if total_runtime_seconds is None:
        runtime_values = [summary.runtime_seconds for summary in summaries]
        if all(value is not None for value in runtime_values):
            total_runtime_seconds = float(sum(runtime_values))
    design_evidence = tuple(
        MonteCarloWideningDesignEvidence(
            dgp_name=summary.design.dgp_name,
            n_obs=summary.design.n_obs,
            p=summary.design.p,
            n_replications=summary.n_replications,
            n_successful_replications=summary.n_successful_replications,
            success_rate=(
                float(summary.n_successful_replications / summary.n_replications)
                if summary.n_replications > 0
                else 0.0
            ),
            runtime_seconds=summary.runtime_seconds,
            parametric_metrics=summary.parametric_metrics,
            nonparametric_metrics=summary.nonparametric_metrics,
            typed_invalidity_counts=summary.typed_invalidity_counts,
            trimming_rate=summary.trimming_rate,
            zero_valid_fold_frequency=summary.zero_valid_fold_frequency,
            typed_invalidity_examples=summary.typed_invalidity_examples,
        )
        for summary in summaries
    )
    return MonteCarloWideningReadinessReport(
        oracle_lane="paper-trigonometric",
        stage_label=monte_carlo_report.stage_label,
        staged_matrix=monte_carlo_report.staging,
        full_target_matrix=dict(monte_carlo_report.full_target_matrix),
        total_runtime_seconds=total_runtime_seconds,
        total_replications=total_replications,
        total_successful_replications=total_successful_replications,
        success_rate=(
            float(total_successful_replications / total_replications)
            if total_replications > 0
            else 0.0
        ),
        typed_invalidity_counts=_merge_invalidity_counts(
            [summary.typed_invalidity_counts for summary in summaries]
        ),
        promotion_gate=(
            "Promote beyond reduced-smoke only after runtime evidence and "
            "typed_invalidity_counts support the wider matrix, and the widened "
            "run has an explicit cost budget plus stop condition."
        ),
        remaining_requirements=(
            "fresh runtime evidence beyond reduced-smoke",
            "explicit widened-matrix cost budget",
            "explicit widened-matrix stop condition",
        ),
        recommended_next_step=(
            "Review runtime and invalidity evidence, then record the widened-"
            "matrix cost budget and stop condition before promotion."
        ),
        design_evidence=design_evidence,
        typed_invalidity_examples=_merge_invalidity_examples(
            [summary.typed_invalidity_examples for summary in summaries]
        ),
    )


def _coerce_repo_root(repo_root: str | Path) -> Path:
    root = Path(repo_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"repo_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"repo_root must be a directory: {root}")
    return root


def _collect_relative_matches(
    repo_root: Path, patterns: Sequence[str]
) -> tuple[str, ...]:
    matches: set[str] = set()
    for pattern in patterns:
        for path in repo_root.glob(pattern):
            if path.is_file():
                matches.add(path.relative_to(repo_root).as_posix())
    return tuple(sorted(matches))


def _filter_section6_raw_data_assets(matches: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        match
        for match in matches
        if match not in _SECTION6_MANIFEST_ASSET_PATHS
        and match in _SECTION6_CANONICAL_RAW_ASSET_PATHS
    )


def _filter_section6_python_loader_assets(matches: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        match for match in matches if match in _SECTION6_PYTHON_LOADER_ASSET_PATHS
    )


def _section6_raw_shape_keys(raw_data_assets: Sequence[str]) -> tuple[str, ...]:
    keys = {
        Path(relative_path).stem
        for relative_path in raw_data_assets
    }
    return tuple(sorted(keys))


def audit_section6_assets(repo_root: str | Path) -> EmpiricalAssetAudit:
    root = _coerce_repo_root(repo_root)
    raw_data_assets = _filter_section6_raw_data_assets(
        _collect_relative_matches(root, _SECTION6_RAW_DATA_GLOBS)
    )
    loader_assets = _collect_relative_matches(root, _SECTION6_LOADER_GLOBS)
    python_loader_assets = _filter_section6_python_loader_assets(loader_assets)
    raw_shape_keys = _section6_raw_shape_keys(raw_data_assets)
    known_figures = tuple(
        sorted(
            relative_path
            for relative_path in _SECTION6_KNOWN_FIGURES
            if (root / relative_path).is_file()
        )
    )

    missing_assets: list[str] = []
    if not raw_data_assets or len(raw_shape_keys) > 1:
        missing_assets.append("section6_raw_dataset")
    if not python_loader_assets:
        missing_assets.append("section6_loader")

    status = "ready" if not missing_assets else "blocked"
    blocker_reason = None
    if status != "ready":
        blocker_reason = (
            "multiple-canonical-raw-shapes"
            if len(raw_shape_keys) > 1
            else "missing-local-dataset"
        )
    return EmpiricalAssetAudit(
        repo_root=root.as_posix(),
        oracle_lane=_SECTION6_ORACLE_LANE,
        status=status,
        blocker_reason=blocker_reason,
        missing_assets=tuple(missing_assets),
        raw_data_assets=raw_data_assets,
        loader_assets=loader_assets,
        known_figures=known_figures,
        time_window="2005-2007",
        treated_state_count=11,
        excluded_states=("New Hampshire", "Pennsylvania"),
        linear_covariate_count=703,
        z_lanes=_SECTION6_Z_LANES,
        basis_family="trigonometric",
        basis_degree=4,
        confidence_level=0.95,
        synthetic_fallback_enabled=False,
        figure_only_mode_enabled=False,
    )


def generate_paper_dgp1(
    design: MonteCarloDesign,
    *,
    random_state: int | None = None,
) -> SimulationDataset:
    if design.dgp_name != "DGP1":
        raise ValueError("generate_paper_dgp1 requires DGP1 design")
    rng = np.random.default_rng(random_state)
    x = rng.normal(size=(design.n_obs, design.p))
    z = rng.normal(size=design.n_obs)
    theta0 = _paper_theta0(design.p)
    propensity = _expit(x @ theta0)
    treat = rng.binomial(1, propensity)
    beta_control = _paper_beta_control(design.p)
    beta_treated = _paper_beta_treated(design.p)
    epsilon0 = rng.normal(size=design.n_obs)
    epsilon1 = rng.normal(size=design.n_obs)
    y0 = rng.normal(size=design.n_obs)
    phi0 = x @ beta_control
    phi1 = x @ beta_treated + np.exp(z)
    y1 = y0 + treat * (phi1 + epsilon1) + (1 - treat) * (phi0 + epsilon0)
    return SimulationDataset(
        y0=y0,
        y1=y1,
        treat=treat,
        x=x,
        z=z,
        z0=design.evaluation_grid,
        basis_family=design.basis_family,
        basis_degree=design.basis_degree,
        alpha=design.alpha,
        oracle_lane=design.oracle_lane,
        dgp_name=design.dgp_name,
        true_beta=_paper_true_beta(design.p),
        true_f_at_z0=np.exp(design.evaluation_grid),
        propensity=propensity,
        x_covariance=np.eye(design.p, dtype=float),
    )


def generate_paper_dgp2(
    design: MonteCarloDesign,
    *,
    random_state: int | None = None,
) -> SimulationDataset:
    if design.dgp_name != "DGP2":
        raise ValueError("generate_paper_dgp2 requires DGP2 design")
    rng = np.random.default_rng(random_state)
    covariance = _toeplitz_covariance(design.p, design.rho_x)
    x = rng.multivariate_normal(
        mean=np.zeros(design.p, dtype=float),
        cov=covariance,
        size=design.n_obs,
    )
    z = rng.normal(size=design.n_obs)
    theta0 = _paper_theta0(design.p)
    propensity = _expit(x @ theta0)
    treat = rng.binomial(1, propensity)
    beta_control = _paper_beta_control(design.p)
    beta_treated = _paper_beta_treated(design.p)
    epsilon0 = rng.normal(size=design.n_obs)
    epsilon1 = rng.normal(size=design.n_obs)
    baseline_error = rng.normal(size=design.n_obs)
    heteroskedastic_scale = (z + x[:, 0]) / np.sqrt(2.0)
    y0 = baseline_error * heteroskedastic_scale
    phi0 = x @ beta_control
    phi1 = x @ beta_treated + np.exp(z)
    y1 = y0 + treat * (phi1 + epsilon1) + (1 - treat) * (phi0 + epsilon0)
    return SimulationDataset(
        y0=y0,
        y1=y1,
        treat=treat,
        x=x,
        z=z,
        z0=design.evaluation_grid,
        basis_family=design.basis_family,
        basis_degree=design.basis_degree,
        alpha=design.alpha,
        oracle_lane=design.oracle_lane,
        dgp_name=design.dgp_name,
        true_beta=_paper_true_beta(design.p),
        true_f_at_z0=np.exp(design.evaluation_grid),
        propensity=propensity,
        x_covariance=covariance,
    )


def default_reduced_smoke_designs() -> tuple[MonteCarloDesign, ...]:
    return (
        MonteCarloDesign(dgp_name="DGP1", n_obs=200, p=10),
        MonteCarloDesign(dgp_name="DGP2", n_obs=200, p=10),
    )


def default_phase7_runtime_probe_designs() -> tuple[MonteCarloDesign, ...]:
    return (
        MonteCarloDesign(
            dgp_name="DGP1",
            n_obs=200,
            p=10,
            evaluation_grid=_PHASE7_RUNTIME_PROBE_EVALUATION_GRID.copy(),
        ),
        MonteCarloDesign(
            dgp_name="DGP2",
            n_obs=200,
            p=10,
            evaluation_grid=_PHASE7_RUNTIME_PROBE_EVALUATION_GRID.copy(),
        ),
        MonteCarloDesign(
            dgp_name="DGP1",
            n_obs=200,
            p=50,
            evaluation_grid=_PHASE7_RUNTIME_PROBE_EVALUATION_GRID.copy(),
        ),
        MonteCarloDesign(
            dgp_name="DGP2",
            n_obs=200,
            p=50,
            evaluation_grid=_PHASE7_RUNTIME_PROBE_EVALUATION_GRID.copy(),
        ),
        MonteCarloDesign(
            dgp_name="DGP1",
            n_obs=500,
            p=50,
            evaluation_grid=_PHASE7_RUNTIME_PROBE_EVALUATION_GRID.copy(),
        ),
        MonteCarloDesign(
            dgp_name="DGP2",
            n_obs=500,
            p=50,
            evaluation_grid=_PHASE7_RUNTIME_PROBE_EVALUATION_GRID.copy(),
        ),
        MonteCarloDesign(
            dgp_name="DGP1",
            n_obs=200,
            p=500,
            evaluation_grid=_PHASE7_RUNTIME_PROBE_EVALUATION_GRID.copy(),
        ),
        MonteCarloDesign(
            dgp_name="DGP2",
            n_obs=200,
            p=500,
            evaluation_grid=_PHASE7_RUNTIME_PROBE_EVALUATION_GRID.copy(),
        ),
    )


def default_phase7_nonparametric_calibration_probe_designs() -> tuple[
    MonteCarloDesign, ...
]:
    return (
        MonteCarloDesign(dgp_name="DGP1", n_obs=200, p=50),
        MonteCarloDesign(dgp_name="DGP1", n_obs=500, p=50),
        MonteCarloDesign(dgp_name="DGP2", n_obs=200, p=50),
        MonteCarloDesign(dgp_name="DGP2", n_obs=500, p=50),
    )


def _runtime_probe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _runtime_probe_root_mean_square(values: Sequence[float]) -> float | None:
    if not values:
        return None
    values_array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values_array**2)))


def _runtime_probe_std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.std(np.asarray(values, dtype=float)))


def _runtime_probe_observation_values(
    observations: Sequence[MonteCarloRuntimeProbeObservation],
    attribute: str,
) -> list[float]:
    values: list[float] = []
    for observation in observations:
        value = getattr(observation, attribute)
        if value is not None:
            values.append(float(value))
    return values


def build_monte_carlo_runtime_probe_report(
    *,
    smoke_reports: Sequence[MonteCarloSmokeReport],
    random_states: Sequence[int],
) -> MonteCarloRuntimeProbeReport:
    reports = tuple(smoke_reports)
    seeds = tuple(
        _coerce_runtime_nonnegative_integer("random_states", seed)
        for seed in random_states
    )
    if not reports:
        raise ValueError(
            "build_monte_carlo_runtime_probe_report requires smoke_reports"
        )
    if len(reports) != len(seeds):
        raise ValueError("smoke_reports and random_states must have the same length")

    first_report = reports[0]
    first_stage_label = str(first_report.stage_label).strip()
    first_lane_order = tuple(
        (summary.design.dgp_name, summary.design.n_obs, summary.design.p)
        for summary in first_report.summaries
    )
    observations: list[MonteCarloRuntimeProbeObservation] = []
    total_runtime_seconds = 0.0
    has_total_runtime = True

    for seed, report in zip(seeds, reports):
        if str(report.stage_label).strip() != first_stage_label:
            raise ValueError(
                "runtime probe smoke reports must share the same stage_label"
            )
        lane_order = tuple(
            (summary.design.dgp_name, summary.design.n_obs, summary.design.p)
            for summary in report.summaries
        )
        if lane_order != first_lane_order:
            raise ValueError(
                "runtime probe smoke reports must share the same design grid"
            )
        if report.total_runtime_seconds is None:
            has_total_runtime = False
        else:
            total_runtime_seconds += float(report.total_runtime_seconds)

        for summary in report.summaries:
            if summary.n_replications != 1:
                raise ValueError(
                    "runtime probe smoke summaries must carry exactly one replication"
                )
            observations.append(
                MonteCarloRuntimeProbeObservation(
                    dgp_name=summary.design.dgp_name,
                    n_obs=summary.design.n_obs,
                    p=summary.design.p,
                    random_state=seed,
                    runtime_seconds=summary.runtime_seconds,
                    success=summary.n_successful_replications == 1,
                    typed_invalidity_counts=summary.typed_invalidity_counts,
                    typed_invalidity_examples=summary.typed_invalidity_examples,
                    trimming_rate=summary.trimming_rate,
                    zero_valid_fold_frequency=summary.zero_valid_fold_frequency,
                    parametric_bias=summary.parametric_metrics.get("bias"),
                    nonparametric_bias=summary.nonparametric_metrics.get("bias"),
                    parametric_rmse=summary.parametric_metrics.get("rmse"),
                    nonparametric_rmse=summary.nonparametric_metrics.get("rmse"),
                    parametric_average_standard_error=summary.parametric_metrics.get(
                        "average_standard_error"
                    ),
                    parametric_coverage=summary.parametric_metrics.get("coverage"),
                    parametric_interval_length=summary.parametric_metrics.get(
                        "interval_length"
                    ),
                    nonparametric_average_standard_error=summary.nonparametric_metrics.get(
                        "average_standard_error"
                    ),
                    nonparametric_coverage=summary.nonparametric_metrics.get(
                        "coverage"
                    ),
                    nonparametric_interval_length=summary.nonparametric_metrics.get(
                        "interval_length"
                    ),
                    nonparametric_absolute_error=(
                        summary.nonparametric_mean_absolute_error
                    ),
                    nonparametric_uniform_critical_value=(
                        summary.nonparametric_uniform_critical_value
                    ),
                    nonparametric_uniform_band_length=(
                        summary.nonparametric_uniform_band_length
                    ),
                )
            )

    grouped: dict[tuple[str, int, int], list[MonteCarloRuntimeProbeObservation]] = {}
    for observation in observations:
        key = (observation.dgp_name, observation.n_obs, observation.p)
        grouped.setdefault(key, []).append(observation)

    design_summaries: list[MonteCarloRuntimeProbeDesignSummary] = []
    for key in first_lane_order:
        design_observations = grouped[key]
        runtimes = [
            observation.runtime_seconds
            for observation in design_observations
            if observation.runtime_seconds is not None
        ]
        successful_observations = [
            observation for observation in design_observations if observation.success
        ]
        design_summaries.append(
            MonteCarloRuntimeProbeDesignSummary(
                dgp_name=key[0],
                n_obs=key[1],
                p=key[2],
                n_runs=len(design_observations),
                n_successful_runs=len(successful_observations),
                success_rate=(
                    float(len(successful_observations) / len(design_observations))
                    if design_observations
                    else 0.0
                ),
                runtime_mean_seconds=_runtime_probe_mean(runtimes),
                runtime_std_seconds=_runtime_probe_std(runtimes),
                typed_invalidity_counts=_merge_invalidity_counts(
                    [
                        observation.typed_invalidity_counts
                        for observation in design_observations
                    ]
                ),
                typed_invalidity_examples=_merge_invalidity_examples(
                    [
                        observation.typed_invalidity_examples
                        for observation in design_observations
                    ]
                ),
                mean_parametric_rmse=_runtime_probe_root_mean_square(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "parametric_rmse",
                    )
                ),
                mean_nonparametric_rmse=_runtime_probe_root_mean_square(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_rmse",
                    )
                ),
                mean_parametric_average_standard_error=_runtime_probe_mean(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "parametric_average_standard_error",
                    )
                ),
                std_parametric_average_standard_error=_runtime_probe_std(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "parametric_average_standard_error",
                    )
                ),
                mean_parametric_coverage=_runtime_probe_mean(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "parametric_coverage",
                    )
                ),
                std_parametric_coverage=_runtime_probe_std(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "parametric_coverage",
                    )
                ),
                mean_parametric_interval_length=_runtime_probe_mean(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "parametric_interval_length",
                    )
                ),
                std_parametric_interval_length=_runtime_probe_std(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "parametric_interval_length",
                    )
                ),
                mean_nonparametric_average_standard_error=_runtime_probe_mean(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_average_standard_error",
                    )
                ),
                std_nonparametric_average_standard_error=_runtime_probe_std(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_average_standard_error",
                    )
                ),
                mean_nonparametric_coverage=_runtime_probe_mean(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_coverage",
                    )
                ),
                std_nonparametric_coverage=_runtime_probe_std(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_coverage",
                    )
                ),
                mean_nonparametric_interval_length=_runtime_probe_mean(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_interval_length",
                    )
                ),
                std_nonparametric_interval_length=_runtime_probe_std(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_interval_length",
                    )
                ),
                mean_nonparametric_absolute_error=_runtime_probe_mean(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_absolute_error",
                    )
                ),
                std_nonparametric_absolute_error=_runtime_probe_std(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_absolute_error",
                    )
                ),
                mean_nonparametric_uniform_critical_value=_runtime_probe_mean(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_uniform_critical_value",
                    )
                ),
                std_nonparametric_uniform_critical_value=_runtime_probe_std(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_uniform_critical_value",
                    )
                ),
                mean_nonparametric_uniform_band_length=_runtime_probe_mean(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_uniform_band_length",
                    )
                ),
                std_nonparametric_uniform_band_length=_runtime_probe_std(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_uniform_band_length",
                    )
                ),
                mean_trimming_rate=_runtime_probe_mean(
                    [
                        observation.trimming_rate
                        for observation in successful_observations
                        if observation.trimming_rate is not None
                    ]
                ),
                mean_zero_valid_fold_frequency=_runtime_probe_mean(
                    [
                        observation.zero_valid_fold_frequency
                        for observation in design_observations
                    ]
                )
                or 0.0,
                mean_parametric_bias=_runtime_probe_mean(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "parametric_bias",
                    )
                ),
                std_parametric_bias=_runtime_probe_std(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "parametric_bias",
                    )
                ),
                mean_nonparametric_bias=_runtime_probe_mean(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_bias",
                    )
                ),
                std_nonparametric_bias=_runtime_probe_std(
                    _runtime_probe_observation_values(
                        successful_observations,
                        "nonparametric_bias",
                    )
                ),
            )
        )

    return MonteCarloRuntimeProbeReport(
        oracle_lane="paper-trigonometric",
        stage_label="phase7-runtime-probe",
        random_states=seeds,
        total_runtime_seconds=total_runtime_seconds if has_total_runtime else None,
        observations=tuple(observations),
        design_summaries=tuple(design_summaries),
        typed_invalidity_counts=_merge_invalidity_counts(
            [observation.typed_invalidity_counts for observation in observations]
        ),
        typed_invalidity_examples=_merge_invalidity_examples(
            [observation.typed_invalidity_examples for observation in observations]
        ),
    )


def _matching_successful_runtime_observations(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    dgp_name: str,
    n_obs: int,
    p: int,
) -> tuple[MonteCarloRuntimeProbeObservation, ...]:
    target = (str(dgp_name).strip().upper(), int(n_obs), int(p))
    return tuple(
        observation
        for observation in runtime_probe.observations
        if observation.success
        and (observation.dgp_name, observation.n_obs, observation.p) == target
    )


def _ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(denominator) <= np.finfo(float).eps:
        return None
    return float(numerator / denominator)


def _coverage_gap_or_none(
    coverage: float | None,
    *,
    nominal_coverage: float,
) -> float | None:
    if coverage is None:
        return None
    return float(coverage - nominal_coverage)


def _worst_coverage_observation(
    observations: Sequence[MonteCarloRuntimeProbeObservation],
) -> MonteCarloRuntimeProbeObservation | None:
    candidates = [
        observation
        for observation in observations
        if observation.nonparametric_coverage is not None
    ]
    if not candidates:
        return None
    return min(
        candidates, key=lambda observation: float(observation.nonparametric_coverage)
    )


def _longest_interval_observation(
    observations: Sequence[MonteCarloRuntimeProbeObservation],
) -> MonteCarloRuntimeProbeObservation | None:
    candidates = [
        observation
        for observation in observations
        if observation.nonparametric_interval_length is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda observation: float(observation.nonparametric_interval_length),
    )


def _largest_uniform_critical_observation(
    observations: Sequence[MonteCarloRuntimeProbeObservation],
) -> MonteCarloRuntimeProbeObservation | None:
    candidates = [
        observation
        for observation in observations
        if observation.nonparametric_uniform_critical_value is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda observation: float(observation.nonparametric_uniform_critical_value),
    )


def _longest_uniform_band_observation(
    observations: Sequence[MonteCarloRuntimeProbeObservation],
) -> MonteCarloRuntimeProbeObservation | None:
    candidates = [
        observation
        for observation in observations
        if observation.nonparametric_uniform_band_length is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda observation: float(observation.nonparametric_uniform_band_length),
    )


def build_phase7_nonparametric_calibration_report(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
) -> Phase7NonparametricCalibrationReport:
    _validate_paper_monte_carlo_lane({runtime_probe.oracle_lane})
    target_n_obs_value = int(target_n_obs)
    reference_n_obs_value = int(reference_n_obs)
    p_value = int(p)
    if target_n_obs_value == reference_n_obs_value:
        raise ValueError("target_n_obs and reference_n_obs must differ")

    design_summary_map = {
        (summary.dgp_name, summary.n_obs, summary.p): summary
        for summary in runtime_probe.design_summaries
    }
    dgp_names = tuple(
        summary.dgp_name
        for summary in runtime_probe.design_summaries
        if summary.n_obs == target_n_obs_value and summary.p == p_value
    )
    if not dgp_names:
        raise ValueError(
            "build_phase7_nonparametric_calibration_report requires at least one "
            f"target design for n={target_n_obs_value}, p={p_value}"
        )

    decompositions: list[Phase7NonparametricCalibrationDecomposition] = []
    for dgp_name in dgp_names:
        target_key = (dgp_name, target_n_obs_value, p_value)
        reference_key = (dgp_name, reference_n_obs_value, p_value)
        if (
            target_key not in design_summary_map
            or reference_key not in design_summary_map
        ):
            raise ValueError(
                "build_phase7_nonparametric_calibration_report requires both target "
                f"and reference summaries for {dgp_name} at p={p_value}"
            )

        target_summary = design_summary_map[target_key]
        reference_summary = design_summary_map[reference_key]
        target_observations = _matching_successful_runtime_observations(
            runtime_probe,
            dgp_name=dgp_name,
            n_obs=target_n_obs_value,
            p=p_value,
        )
        reference_observations = _matching_successful_runtime_observations(
            runtime_probe,
            dgp_name=dgp_name,
            n_obs=reference_n_obs_value,
            p=p_value,
        )
        target_worst_coverage = _worst_coverage_observation(target_observations)
        reference_worst_coverage = _worst_coverage_observation(reference_observations)
        target_longest_interval = _longest_interval_observation(target_observations)
        reference_longest_interval = _longest_interval_observation(
            reference_observations
        )
        target_largest_uniform_critical = _largest_uniform_critical_observation(
            target_observations
        )
        reference_largest_uniform_critical = _largest_uniform_critical_observation(
            reference_observations
        )
        target_longest_uniform_band = _longest_uniform_band_observation(
            target_observations
        )
        reference_longest_uniform_band = _longest_uniform_band_observation(
            reference_observations
        )

        decompositions.append(
            Phase7NonparametricCalibrationDecomposition(
                dgp_name=dgp_name,
                p=p_value,
                target_n_obs=target_n_obs_value,
                reference_n_obs=reference_n_obs_value,
                nominal_coverage=_PAPER_NOMINAL_COVERAGE,
                target_mean_nonparametric_rmse=target_summary.mean_nonparametric_rmse,
                reference_mean_nonparametric_rmse=reference_summary.mean_nonparametric_rmse,
                rmse_inflation_factor=_ratio_or_none(
                    target_summary.mean_nonparametric_rmse,
                    reference_summary.mean_nonparametric_rmse,
                ),
                target_mean_nonparametric_absolute_error=(
                    target_summary.mean_nonparametric_absolute_error
                ),
                reference_mean_nonparametric_absolute_error=(
                    reference_summary.mean_nonparametric_absolute_error
                ),
                absolute_error_inflation_factor=_ratio_or_none(
                    target_summary.mean_nonparametric_absolute_error,
                    reference_summary.mean_nonparametric_absolute_error,
                ),
                target_mean_nonparametric_average_standard_error=(
                    target_summary.mean_nonparametric_average_standard_error
                ),
                reference_mean_nonparametric_average_standard_error=(
                    reference_summary.mean_nonparametric_average_standard_error
                ),
                standard_error_inflation_factor=_ratio_or_none(
                    target_summary.mean_nonparametric_average_standard_error,
                    reference_summary.mean_nonparametric_average_standard_error,
                ),
                target_mean_nonparametric_interval_length=(
                    target_summary.mean_nonparametric_interval_length
                ),
                reference_mean_nonparametric_interval_length=(
                    reference_summary.mean_nonparametric_interval_length
                ),
                interval_length_inflation_factor=_ratio_or_none(
                    target_summary.mean_nonparametric_interval_length,
                    reference_summary.mean_nonparametric_interval_length,
                ),
                target_mean_nonparametric_uniform_critical_value=(
                    target_summary.mean_nonparametric_uniform_critical_value
                ),
                reference_mean_nonparametric_uniform_critical_value=(
                    reference_summary.mean_nonparametric_uniform_critical_value
                ),
                uniform_critical_value_inflation_factor=_ratio_or_none(
                    target_summary.mean_nonparametric_uniform_critical_value,
                    reference_summary.mean_nonparametric_uniform_critical_value,
                ),
                target_mean_nonparametric_uniform_band_length=(
                    target_summary.mean_nonparametric_uniform_band_length
                ),
                reference_mean_nonparametric_uniform_band_length=(
                    reference_summary.mean_nonparametric_uniform_band_length
                ),
                uniform_band_length_inflation_factor=_ratio_or_none(
                    target_summary.mean_nonparametric_uniform_band_length,
                    reference_summary.mean_nonparametric_uniform_band_length,
                ),
                target_mean_nonparametric_coverage=target_summary.mean_nonparametric_coverage,
                reference_mean_nonparametric_coverage=(
                    reference_summary.mean_nonparametric_coverage
                ),
                target_coverage_gap=_coverage_gap_or_none(
                    target_summary.mean_nonparametric_coverage,
                    nominal_coverage=_PAPER_NOMINAL_COVERAGE,
                ),
                reference_coverage_gap=_coverage_gap_or_none(
                    reference_summary.mean_nonparametric_coverage,
                    nominal_coverage=_PAPER_NOMINAL_COVERAGE,
                ),
                target_worst_coverage_random_state=(
                    None
                    if target_worst_coverage is None
                    else target_worst_coverage.random_state
                ),
                target_worst_coverage=(
                    None
                    if target_worst_coverage is None
                    else target_worst_coverage.nonparametric_coverage
                ),
                reference_worst_coverage_random_state=(
                    None
                    if reference_worst_coverage is None
                    else reference_worst_coverage.random_state
                ),
                reference_worst_coverage=(
                    None
                    if reference_worst_coverage is None
                    else reference_worst_coverage.nonparametric_coverage
                ),
                target_longest_interval_random_state=(
                    None
                    if target_longest_interval is None
                    else target_longest_interval.random_state
                ),
                target_longest_interval_length=(
                    None
                    if target_longest_interval is None
                    else target_longest_interval.nonparametric_interval_length
                ),
                reference_longest_interval_random_state=(
                    None
                    if reference_longest_interval is None
                    else reference_longest_interval.random_state
                ),
                reference_longest_interval_length=(
                    None
                    if reference_longest_interval is None
                    else reference_longest_interval.nonparametric_interval_length
                ),
                target_largest_uniform_critical_random_state=(
                    None
                    if target_largest_uniform_critical is None
                    else target_largest_uniform_critical.random_state
                ),
                target_largest_uniform_critical_value=(
                    None
                    if target_largest_uniform_critical is None
                    else target_largest_uniform_critical.nonparametric_uniform_critical_value
                ),
                reference_largest_uniform_critical_random_state=(
                    None
                    if reference_largest_uniform_critical is None
                    else reference_largest_uniform_critical.random_state
                ),
                reference_largest_uniform_critical_value=(
                    None
                    if reference_largest_uniform_critical is None
                    else reference_largest_uniform_critical.nonparametric_uniform_critical_value
                ),
                target_longest_uniform_band_random_state=(
                    None
                    if target_longest_uniform_band is None
                    else target_longest_uniform_band.random_state
                ),
                target_longest_uniform_band_length=(
                    None
                    if target_longest_uniform_band is None
                    else target_longest_uniform_band.nonparametric_uniform_band_length
                ),
                reference_longest_uniform_band_random_state=(
                    None
                    if reference_longest_uniform_band is None
                    else reference_longest_uniform_band.random_state
                ),
                reference_longest_uniform_band_length=(
                    None
                    if reference_longest_uniform_band is None
                    else reference_longest_uniform_band.nonparametric_uniform_band_length
                ),
            )
        )

    return Phase7NonparametricCalibrationReport(
        oracle_lane="paper-trigonometric",
        stage_label="phase7-nonparametric-calibration-probe",
        random_states=tuple(runtime_probe.random_states),
        nominal_coverage=_PAPER_NOMINAL_COVERAGE,
        target_n_obs=target_n_obs_value,
        reference_n_obs=reference_n_obs_value,
        p=p_value,
        decompositions=tuple(decompositions),
    )


def run_phase7_nonparametric_calibration_probe(
    *,
    random_states: Sequence[int] = (101, 202, 303),
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricCalibrationReport:
    target_n_obs_value = int(target_n_obs)
    reference_n_obs_value = int(reference_n_obs)
    p_value = int(p)
    default_designs = (
        default_phase7_runtime_probe_designs()
        if (target_n_obs_value, reference_n_obs_value, p_value) == (200, 500, 50)
        else (
            MonteCarloDesign(dgp_name="DGP1", n_obs=target_n_obs_value, p=p_value),
            MonteCarloDesign(dgp_name="DGP1", n_obs=reference_n_obs_value, p=p_value),
            MonteCarloDesign(dgp_name="DGP2", n_obs=target_n_obs_value, p=p_value),
            MonteCarloDesign(dgp_name="DGP2", n_obs=reference_n_obs_value, p=p_value),
        )
    )
    design_sequence = tuple(designs or default_designs)
    runtime_probe = run_phase7_monte_carlo_runtime_probe(
        random_states=random_states,
        n_boot=n_boot,
        designs=design_sequence,
    )
    return build_phase7_nonparametric_calibration_report(
        runtime_probe,
        target_n_obs=target_n_obs_value,
        reference_n_obs=reference_n_obs_value,
        p=p_value,
    )


def _phase7_runtime_probe_replication_seed(
    *,
    random_state: int,
    designs: Sequence[MonteCarloDesign],
    dgp_name: str,
    n_obs: int,
    p: int,
) -> int:
    rng = np.random.default_rng(int(random_state))
    target = (str(dgp_name).strip().upper(), int(n_obs), int(p))
    for design in designs:
        replication_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        if (design.dgp_name, design.n_obs, design.p) == target:
            return replication_seed
    raise KeyError(
        f"runtime probe design not present in canonical design order: {target!r}"
    )


def _match_runtime_probe_design(
    designs: Sequence[MonteCarloDesign],
    *,
    dgp_name: str,
    n_obs: int,
    p: int,
) -> MonteCarloDesign:
    target = (str(dgp_name).strip().upper(), int(n_obs), int(p))
    for design in designs:
        if (design.dgp_name, design.n_obs, design.p) == target:
            return design
    raise KeyError(f"runtime probe design not present: {target!r}")


def _grid_maximum(
    evaluation_grid: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float]:
    grid = np.asarray(evaluation_grid, dtype=float)
    vector = np.asarray(values, dtype=float)
    index = int(np.argmax(vector))
    return float(grid[index]), float(vector[index])


def _clone_design_with_evaluation_grid(
    design: MonteCarloDesign,
    *,
    evaluation_grid: np.ndarray | Sequence[float],
) -> MonteCarloDesign:
    return MonteCarloDesign(
        dgp_name=design.dgp_name,
        n_obs=design.n_obs,
        p=design.p,
        basis_family=design.basis_family,
        basis_degree=design.basis_degree,
        oracle_lane=design.oracle_lane,
        alpha=design.alpha,
        rho_x=design.rho_x,
        n_folds=design.n_folds,
        trim_lower=design.trim_lower,
        trim_upper=design.trim_upper,
        evaluation_grid=np.asarray(evaluation_grid, dtype=float),
    )


@dataclass(slots=True)
class _Phase7SameSeedReplayPayload:
    dataset: SimulationDataset
    score_payload: ScorePayload
    estimation_payload: EstimationPayload
    estimation_result: Any


@dataclass(slots=True)
class _Phase7SameSeedNonparametricPayload:
    dataset: SimulationDataset
    score_payload: ScorePayload
    estimation_payload: EstimationPayload
    estimation_result: Any
    nonparametric_payload: Any


@dataclass(slots=True)
class Phase7SameSeedTracePayloadCacheStep:
    helper_name: str
    hits: int
    misses: int
    delta_hits: int
    delta_misses: int

    def __post_init__(self) -> None:
        self.helper_name = str(self.helper_name).strip()
        self.hits = int(self.hits)
        self.misses = int(self.misses)
        self.delta_hits = int(self.delta_hits)
        self.delta_misses = int(self.delta_misses)

    def to_dict(self) -> dict[str, str | int]:
        return {
            "helper_name": self.helper_name,
            "hits": self.hits,
            "misses": self.misses,
            "delta_hits": self.delta_hits,
            "delta_misses": self.delta_misses,
        }


@dataclass(slots=True)
class Phase7SameSeedTracePayloadCacheBridgeProbe:
    status: str
    replay_steps: tuple[Phase7SameSeedTracePayloadCacheStep, ...]
    nonparametric_steps: tuple[Phase7SameSeedTracePayloadCacheStep, ...]

    def __post_init__(self) -> None:
        self.status = str(self.status).strip()
        self.replay_steps = tuple(self.replay_steps)
        self.nonparametric_steps = tuple(self.nonparametric_steps)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "replay_steps": [step.to_dict() for step in self.replay_steps],
            "nonparametric_steps": [
                step.to_dict() for step in self.nonparametric_steps
            ],
        }


@dataclass(slots=True)
class Phase7SameSeedBeforeAfterAcceptanceStep:
    scenario_name: str
    driver_signature: str
    point_miss_vector: tuple[int, int, int]
    band_miss_vector: tuple[int, int, int]
    witness_floor: float
    floor_lift_observed: bool
    left_guard_band_preserved: bool
    left_guard_pointwise_nonregression: bool
    pointwise_total_miss_reduced: bool
    right_center_lane_residual_only: bool
    acceptance_passed: bool
    seed_consistent_digest_verified: bool
    candidate_status_if_realized: str
    resulting_evidence_status_if_realized: str

    def __post_init__(self) -> None:
        self.scenario_name = str(self.scenario_name).strip()
        self.driver_signature = str(self.driver_signature).strip()
        self.point_miss_vector = tuple(int(value) for value in self.point_miss_vector)
        self.band_miss_vector = tuple(int(value) for value in self.band_miss_vector)
        self.witness_floor = float(self.witness_floor)
        self.floor_lift_observed = bool(self.floor_lift_observed)
        self.left_guard_band_preserved = bool(self.left_guard_band_preserved)
        self.left_guard_pointwise_nonregression = bool(
            self.left_guard_pointwise_nonregression
        )
        self.pointwise_total_miss_reduced = bool(self.pointwise_total_miss_reduced)
        self.right_center_lane_residual_only = bool(
            self.right_center_lane_residual_only
        )
        self.acceptance_passed = bool(self.acceptance_passed)
        self.seed_consistent_digest_verified = bool(
            self.seed_consistent_digest_verified
        )
        self.candidate_status_if_realized = str(
            self.candidate_status_if_realized
        ).strip()
        self.resulting_evidence_status_if_realized = str(
            self.resulting_evidence_status_if_realized
        ).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "driver_signature": self.driver_signature,
            "point_miss_vector": list(self.point_miss_vector),
            "band_miss_vector": list(self.band_miss_vector),
            "witness_floor": self.witness_floor,
            "floor_lift_observed": self.floor_lift_observed,
            "left_guard_band_preserved": self.left_guard_band_preserved,
            "left_guard_pointwise_nonregression": self.left_guard_pointwise_nonregression,
            "pointwise_total_miss_reduced": self.pointwise_total_miss_reduced,
            "right_center_lane_residual_only": self.right_center_lane_residual_only,
            "acceptance_passed": self.acceptance_passed,
            "seed_consistent_digest_verified": self.seed_consistent_digest_verified,
            "candidate_status_if_realized": self.candidate_status_if_realized,
            "resulting_evidence_status_if_realized": self.resulting_evidence_status_if_realized,
        }


@dataclass(slots=True)
class Phase7SameSeedBeforeAfterAcceptanceProbe:
    status: str
    canonical_seed_order: tuple[int, ...]
    admission_order_driver_signature: str
    required_min_witness_floor: float
    runtime_witness_path: tuple[str, ...]
    left_guard_grid_value: float
    residual_lane_grid_values: tuple[float, float]
    scenarios: tuple[Phase7SameSeedBeforeAfterAcceptanceStep, ...]

    def __post_init__(self) -> None:
        self.status = str(self.status).strip()
        self.canonical_seed_order = tuple(
            int(value) for value in self.canonical_seed_order
        )
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.left_guard_grid_value = float(self.left_guard_grid_value)
        self.residual_lane_grid_values = tuple(
            float(value) for value in self.residual_lane_grid_values
        )
        self.scenarios = tuple(self.scenarios)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical_seed_order": list(self.canonical_seed_order),
            "admission_order_driver_signature": self.admission_order_driver_signature,
            "required_min_witness_floor": self.required_min_witness_floor,
            "runtime_witness_path": list(self.runtime_witness_path),
            "left_guard_grid_value": self.left_guard_grid_value,
            "residual_lane_grid_values": list(self.residual_lane_grid_values),
            "scenarios": [step.to_dict() for step in self.scenarios],
        }

    @property
    def current_driver_signature(self) -> str:
        return self.scenarios[0].driver_signature

    @property
    def target_driver_signature(self) -> str:
        return self.scenarios[-1].driver_signature

    @property
    def current_point_miss_vector(self) -> tuple[int, int, int]:
        return self.scenarios[0].point_miss_vector

    @property
    def target_point_miss_vector(self) -> tuple[int, int, int]:
        return self.scenarios[-1].point_miss_vector

    @property
    def current_band_miss_vector(self) -> tuple[int, int, int]:
        return self.scenarios[0].band_miss_vector

    @property
    def target_band_miss_vector(self) -> tuple[int, int, int]:
        return self.scenarios[-1].band_miss_vector

    @property
    def current_witness_floor(self) -> float:
        return self.scenarios[0].witness_floor

    @property
    def target_witness_floor(self) -> float:
        return self.scenarios[-1].witness_floor


@dataclass(slots=True)
class Phase7SameSeedRuntimeBridgeAfterReportScenarioStep:
    scenario_name: str
    after_report_supplied: bool
    before_after_driver_signature: str
    runtime_bridge_driver_signature: str
    runtime_bridge_state: str
    intake_disposition: str
    point_miss_vector: tuple[int, int, int]
    band_miss_vector: tuple[int, int, int]
    witness_floor: float
    queued_residual_slot_still_live: bool

    def __post_init__(self) -> None:
        self.scenario_name = str(self.scenario_name).strip()
        self.after_report_supplied = bool(self.after_report_supplied)
        self.before_after_driver_signature = str(
            self.before_after_driver_signature
        ).strip()
        self.runtime_bridge_driver_signature = str(
            self.runtime_bridge_driver_signature
        ).strip()
        self.runtime_bridge_state = str(self.runtime_bridge_state).strip()
        self.intake_disposition = str(self.intake_disposition).strip()
        self.point_miss_vector = tuple(int(value) for value in self.point_miss_vector)
        self.band_miss_vector = tuple(int(value) for value in self.band_miss_vector)
        self.witness_floor = float(self.witness_floor)
        self.queued_residual_slot_still_live = bool(
            self.queued_residual_slot_still_live
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "after_report_supplied": self.after_report_supplied,
            "before_after_driver_signature": self.before_after_driver_signature,
            "runtime_bridge_driver_signature": self.runtime_bridge_driver_signature,
            "runtime_bridge_state": self.runtime_bridge_state,
            "intake_disposition": self.intake_disposition,
            "point_miss_vector": list(self.point_miss_vector),
            "band_miss_vector": list(self.band_miss_vector),
            "witness_floor": self.witness_floor,
            "queued_residual_slot_still_live": self.queued_residual_slot_still_live,
        }


@dataclass(slots=True)
class Phase7SameSeedRuntimeBridgeAfterReportScenarioProbe:
    status: str
    canonical_seed_order: tuple[int, ...]
    current_frontier_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    scenarios: tuple[Phase7SameSeedRuntimeBridgeAfterReportScenarioStep, ...]

    def __post_init__(self) -> None:
        self.status = str(self.status).strip()
        self.canonical_seed_order = tuple(
            int(value) for value in self.canonical_seed_order
        )
        self.current_frontier_driver_signature = str(
            self.current_frontier_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.scenarios = tuple(self.scenarios)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical_seed_order": list(self.canonical_seed_order),
            "current_frontier_driver_signature": self.current_frontier_driver_signature,
            "runtime_witness_path": list(self.runtime_witness_path),
            "acceptance_readout_path": list(self.acceptance_readout_path),
            "scenarios": [step.to_dict() for step in self.scenarios],
        }


@dataclass(slots=True)
class Phase7SameSeedCompletionLadderRepairSlot:
    random_state: int
    seed_group: str
    z_index: int
    z_value: float

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)

    def to_dict(self) -> dict[str, object]:
        return {
            "random_state": self.random_state,
            "seed_group": self.seed_group,
            "z_index": self.z_index,
            "z_value": self.z_value,
        }


@dataclass(slots=True)
class Phase7SameSeedCompletionLadderStep:
    scenario_name: str
    driver_signature: str
    candidate_status: str
    resulting_evidence_status: str
    before_after_driver_signature: str
    point_miss_vector: tuple[int, int, int]
    band_miss_vector: tuple[int, int, int]
    witness_floor: float
    next_repair_slot: Phase7SameSeedCompletionLadderRepairSlot | None
    queued_residual_slot_still_live: bool

    def __post_init__(self) -> None:
        self.scenario_name = str(self.scenario_name).strip()
        self.driver_signature = str(self.driver_signature).strip()
        self.candidate_status = str(self.candidate_status).strip()
        self.resulting_evidence_status = str(self.resulting_evidence_status).strip()
        self.before_after_driver_signature = str(
            self.before_after_driver_signature
        ).strip()
        self.point_miss_vector = tuple(int(value) for value in self.point_miss_vector)
        self.band_miss_vector = tuple(int(value) for value in self.band_miss_vector)
        self.witness_floor = float(self.witness_floor)
        self.queued_residual_slot_still_live = bool(
            self.queued_residual_slot_still_live
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "driver_signature": self.driver_signature,
            "candidate_status": self.candidate_status,
            "resulting_evidence_status": self.resulting_evidence_status,
            "before_after_driver_signature": self.before_after_driver_signature,
            "point_miss_vector": list(self.point_miss_vector),
            "band_miss_vector": list(self.band_miss_vector),
            "witness_floor": self.witness_floor,
            "next_repair_slot": (
                None
                if self.next_repair_slot is None
                else self.next_repair_slot.to_dict()
            ),
            "queued_residual_slot_still_live": self.queued_residual_slot_still_live,
        }


@dataclass(slots=True)
class Phase7SameSeedCompletionLadderProbe:
    status: str
    canonical_seed_order: tuple[int, ...]
    admission_order_driver_signature: str
    current_frontier_driver_signature: str
    binding_progress_driver_signature: str
    residual_completion_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    required_min_witness_floor: float
    binding_repair_slot: Phase7SameSeedCompletionLadderRepairSlot
    residual_repair_slot: Phase7SameSeedCompletionLadderRepairSlot
    scenarios: tuple[Phase7SameSeedCompletionLadderStep, ...]

    def __post_init__(self) -> None:
        self.status = str(self.status).strip()
        self.canonical_seed_order = tuple(
            int(value) for value in self.canonical_seed_order
        )
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.current_frontier_driver_signature = str(
            self.current_frontier_driver_signature
        ).strip()
        self.binding_progress_driver_signature = str(
            self.binding_progress_driver_signature
        ).strip()
        self.residual_completion_driver_signature = str(
            self.residual_completion_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.scenarios = tuple(self.scenarios)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical_seed_order": list(self.canonical_seed_order),
            "admission_order_driver_signature": self.admission_order_driver_signature,
            "current_frontier_driver_signature": self.current_frontier_driver_signature,
            "binding_progress_driver_signature": self.binding_progress_driver_signature,
            "residual_completion_driver_signature": self.residual_completion_driver_signature,
            "runtime_witness_path": list(self.runtime_witness_path),
            "required_min_witness_floor": self.required_min_witness_floor,
            "binding_repair_slot": self.binding_repair_slot.to_dict(),
            "residual_repair_slot": self.residual_repair_slot.to_dict(),
            "scenarios": [step.to_dict() for step in self.scenarios],
        }


@dataclass(slots=True)
class Phase7SameSeedRepairAgendaProbeSlot:
    random_state: int
    seed_group: str
    z_index: int
    z_value: float
    repair_stage: str
    repair_order: int
    current_covered: bool
    completion_target_covered: bool
    currently_actionable: bool

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)
        self.repair_stage = str(self.repair_stage).strip().lower()
        self.repair_order = int(self.repair_order)
        self.current_covered = bool(self.current_covered)
        self.completion_target_covered = bool(self.completion_target_covered)
        self.currently_actionable = bool(self.currently_actionable)

    def to_dict(self) -> dict[str, object]:
        return {
            "random_state": self.random_state,
            "seed_group": self.seed_group,
            "z_index": self.z_index,
            "z_value": self.z_value,
            "repair_stage": self.repair_stage,
            "repair_order": self.repair_order,
            "current_covered": self.current_covered,
            "completion_target_covered": self.completion_target_covered,
            "currently_actionable": self.currently_actionable,
        }


@dataclass(slots=True)
class Phase7SameSeedRepairAgendaProbe:
    status: str
    canonical_seed_order: tuple[int, ...]
    admission_order_driver_signature: str
    current_frontier_driver_signature: str
    repair_agenda_driver_signature: str
    current_before_after_driver_signature: str
    current_rung_status: str
    highest_landed_rung: str
    downstream_evidence_status: str
    runtime_witness_path: tuple[str, ...]
    required_min_witness_floor: float
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    current_to_next_rung_pointwise_gap_vector: tuple[int, int, int]
    current_to_next_rung_band_gap_vector: tuple[int, int, int]
    current_to_completion_pointwise_gap_vector: tuple[int, int, int]
    current_to_completion_band_gap_vector: tuple[int, int, int]
    next_required_slot: Phase7SameSeedRepairAgendaProbeSlot | None
    pending_repair_slots: tuple[Phase7SameSeedRepairAgendaProbeSlot, ...]
    canonical_repair_agenda_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.status = str(self.status).strip()
        self.canonical_seed_order = tuple(
            int(value) for value in self.canonical_seed_order
        )
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.current_frontier_driver_signature = str(
            self.current_frontier_driver_signature
        ).strip()
        self.repair_agenda_driver_signature = str(
            self.repair_agenda_driver_signature
        ).strip()
        self.current_before_after_driver_signature = str(
            self.current_before_after_driver_signature
        ).strip()
        self.current_rung_status = str(self.current_rung_status).strip()
        self.highest_landed_rung = str(self.highest_landed_rung).strip()
        self.downstream_evidence_status = str(self.downstream_evidence_status).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.current_witness_floor = float(self.current_witness_floor)
        self.current_to_next_rung_pointwise_gap_vector = tuple(
            int(value) for value in self.current_to_next_rung_pointwise_gap_vector
        )
        self.current_to_next_rung_band_gap_vector = tuple(
            int(value) for value in self.current_to_next_rung_band_gap_vector
        )
        self.current_to_completion_pointwise_gap_vector = tuple(
            int(value) for value in self.current_to_completion_pointwise_gap_vector
        )
        self.current_to_completion_band_gap_vector = tuple(
            int(value) for value in self.current_to_completion_band_gap_vector
        )
        self.pending_repair_slots = tuple(self.pending_repair_slots)
        self.canonical_repair_agenda_digest = tuple(
            str(line).rstrip() for line in self.canonical_repair_agenda_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical_seed_order": list(self.canonical_seed_order),
            "admission_order_driver_signature": self.admission_order_driver_signature,
            "current_frontier_driver_signature": self.current_frontier_driver_signature,
            "repair_agenda_driver_signature": self.repair_agenda_driver_signature,
            "current_before_after_driver_signature": (
                self.current_before_after_driver_signature
            ),
            "current_rung_status": self.current_rung_status,
            "highest_landed_rung": self.highest_landed_rung,
            "downstream_evidence_status": self.downstream_evidence_status,
            "runtime_witness_path": list(self.runtime_witness_path),
            "required_min_witness_floor": self.required_min_witness_floor,
            "current_point_miss_vector": list(self.current_point_miss_vector),
            "current_band_miss_vector": list(self.current_band_miss_vector),
            "current_witness_floor": self.current_witness_floor,
            "current_to_next_rung_pointwise_gap_vector": list(
                self.current_to_next_rung_pointwise_gap_vector
            ),
            "current_to_next_rung_band_gap_vector": list(
                self.current_to_next_rung_band_gap_vector
            ),
            "current_to_completion_pointwise_gap_vector": list(
                self.current_to_completion_pointwise_gap_vector
            ),
            "current_to_completion_band_gap_vector": list(
                self.current_to_completion_band_gap_vector
            ),
            "next_required_slot": (
                None
                if self.next_required_slot is None
                else self.next_required_slot.to_dict()
            ),
            "pending_repair_slots": [
                slot.to_dict() for slot in self.pending_repair_slots
            ],
            "canonical_repair_agenda_digest": list(self.canonical_repair_agenda_digest),
        }


@dataclass(slots=True)
class Phase7SameSeedFirstHopRuntimeBindingPacketProbeSlot:
    random_state: int
    seed_group: str
    z_index: int
    z_value: float
    repair_stage: str
    repair_role: str
    repair_priority: int

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)
        self.repair_stage = str(self.repair_stage).strip().lower()
        self.repair_role = str(self.repair_role).strip().lower()
        self.repair_priority = int(self.repair_priority)

    def to_dict(self) -> dict[str, object]:
        return {
            "random_state": self.random_state,
            "seed_group": self.seed_group,
            "z_index": self.z_index,
            "z_value": self.z_value,
            "repair_stage": self.repair_stage,
            "repair_role": self.repair_role,
            "repair_priority": self.repair_priority,
        }


@dataclass(slots=True)
class Phase7SameSeedFirstHopRuntimeBindingPacketProbePacket:
    random_state: int
    seed_group: str
    replication_seed: int
    z_index: int
    z_value: float
    repair_stage: str
    repair_role: str
    repair_priority: int
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    baseline_open_witness_floor_gap: float
    share_of_baseline_open_witness_floor_gap: float
    center_truth: float
    center_estimate: float
    center_error: float
    center_pointwise_interval_lower: float
    center_pointwise_interval_upper: float
    center_half_interval: float
    center_sigma: float
    center_lower_minus_truth: float
    center_error_to_half_interval_ratio: float
    vf_cross_entry: float
    runtime_object_flow_focus: tuple[str, ...]

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.seed_group = str(self.seed_group).strip().lower()
        self.replication_seed = int(self.replication_seed)
        self.z_index = int(self.z_index)
        self.z_value = float(self.z_value)
        self.repair_stage = str(self.repair_stage).strip().lower()
        self.repair_role = str(self.repair_role).strip().lower()
        self.repair_priority = int(self.repair_priority)
        self.target_fold_id = int(self.target_fold_id)
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.target_exact_trim_floor_weighted_retention = float(
            self.target_exact_trim_floor_weighted_retention
        )
        self.baseline_open_witness_floor_gap = float(
            self.baseline_open_witness_floor_gap
        )
        self.share_of_baseline_open_witness_floor_gap = float(
            self.share_of_baseline_open_witness_floor_gap
        )
        self.center_truth = float(self.center_truth)
        self.center_estimate = float(self.center_estimate)
        self.center_error = float(self.center_error)
        self.center_pointwise_interval_lower = float(
            self.center_pointwise_interval_lower
        )
        self.center_pointwise_interval_upper = float(
            self.center_pointwise_interval_upper
        )
        self.center_half_interval = float(self.center_half_interval)
        self.center_sigma = float(self.center_sigma)
        self.center_lower_minus_truth = float(self.center_lower_minus_truth)
        self.center_error_to_half_interval_ratio = float(
            self.center_error_to_half_interval_ratio
        )
        self.vf_cross_entry = float(self.vf_cross_entry)
        self.runtime_object_flow_focus = tuple(
            str(item).strip() for item in self.runtime_object_flow_focus
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "random_state": self.random_state,
            "seed_group": self.seed_group,
            "replication_seed": self.replication_seed,
            "z_index": self.z_index,
            "z_value": self.z_value,
            "repair_stage": self.repair_stage,
            "repair_role": self.repair_role,
            "repair_priority": self.repair_priority,
            "target_fold_id": self.target_fold_id,
            "target_exact_trim_floor_count": self.target_exact_trim_floor_count,
            "target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi": (
                self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
            ),
            "target_exact_trim_floor_weighted_share_of_fold3_weighted": (
                self.target_exact_trim_floor_weighted_share_of_fold3_weighted
            ),
            "target_exact_trim_floor_weighted_retention": (
                self.target_exact_trim_floor_weighted_retention
            ),
            "baseline_open_witness_floor_gap": self.baseline_open_witness_floor_gap,
            "share_of_baseline_open_witness_floor_gap": (
                self.share_of_baseline_open_witness_floor_gap
            ),
            "center_truth": self.center_truth,
            "center_estimate": self.center_estimate,
            "center_error": self.center_error,
            "center_pointwise_interval_lower": self.center_pointwise_interval_lower,
            "center_pointwise_interval_upper": self.center_pointwise_interval_upper,
            "center_half_interval": self.center_half_interval,
            "center_sigma": self.center_sigma,
            "center_lower_minus_truth": self.center_lower_minus_truth,
            "center_error_to_half_interval_ratio": (
                self.center_error_to_half_interval_ratio
            ),
            "vf_cross_entry": self.vf_cross_entry,
            "runtime_object_flow_focus": list(self.runtime_object_flow_focus),
        }


@dataclass(slots=True)
class Phase7SameSeedFirstHopRuntimeBindingPacketProbe:
    status: str
    canonical_seed_order: tuple[int, ...]
    admission_order_driver_signature: str
    current_frontier_driver_signature: str
    repair_agenda_driver_signature: str
    current_before_after_driver_signature: str
    first_hop_contract_driver_signature: str
    binding_packet_driver_signature: str
    seed303_object_flow_driver_signature: str
    current_rung_status: str
    runtime_witness_path: tuple[str, ...]
    runtime_object_flow_focus: tuple[str, ...]
    actionable_now: bool
    next_required_slot: Phase7SameSeedFirstHopRuntimeBindingPacketProbeSlot | None
    binding_runtime_packet: Phase7SameSeedFirstHopRuntimeBindingPacketProbePacket
    canonical_runtime_binding_packet_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.status = str(self.status).strip()
        self.canonical_seed_order = tuple(
            int(value) for value in self.canonical_seed_order
        )
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.current_frontier_driver_signature = str(
            self.current_frontier_driver_signature
        ).strip()
        self.repair_agenda_driver_signature = str(
            self.repair_agenda_driver_signature
        ).strip()
        self.current_before_after_driver_signature = str(
            self.current_before_after_driver_signature
        ).strip()
        self.first_hop_contract_driver_signature = str(
            self.first_hop_contract_driver_signature
        ).strip()
        self.binding_packet_driver_signature = str(
            self.binding_packet_driver_signature
        ).strip()
        self.seed303_object_flow_driver_signature = str(
            self.seed303_object_flow_driver_signature
        ).strip()
        self.current_rung_status = str(self.current_rung_status).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.runtime_object_flow_focus = tuple(
            str(item).strip() for item in self.runtime_object_flow_focus
        )
        self.actionable_now = bool(self.actionable_now)
        self.canonical_runtime_binding_packet_digest = tuple(
            str(line).rstrip() for line in self.canonical_runtime_binding_packet_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical_seed_order": list(self.canonical_seed_order),
            "admission_order_driver_signature": self.admission_order_driver_signature,
            "current_frontier_driver_signature": self.current_frontier_driver_signature,
            "repair_agenda_driver_signature": self.repair_agenda_driver_signature,
            "current_before_after_driver_signature": (
                self.current_before_after_driver_signature
            ),
            "first_hop_contract_driver_signature": (
                self.first_hop_contract_driver_signature
            ),
            "binding_packet_driver_signature": self.binding_packet_driver_signature,
            "seed303_object_flow_driver_signature": (
                self.seed303_object_flow_driver_signature
            ),
            "current_rung_status": self.current_rung_status,
            "runtime_witness_path": list(self.runtime_witness_path),
            "runtime_object_flow_focus": list(self.runtime_object_flow_focus),
            "actionable_now": self.actionable_now,
            "next_required_slot": (
                None
                if self.next_required_slot is None
                else self.next_required_slot.to_dict()
            ),
            "binding_runtime_packet": self.binding_runtime_packet.to_dict(),
            "canonical_runtime_binding_packet_digest": list(
                self.canonical_runtime_binding_packet_digest
            ),
        }


@dataclass(slots=True)
class Phase7SameSeedFirstHopLandingGuardProbeStep:
    scenario_name: str
    landing_guard_driver_signature: str
    binding_packet_driver_signature: str
    repair_agenda_driver_signature: str
    current_rung_status: str
    point_miss_vector: tuple[int, int, int]
    band_miss_vector: tuple[int, int, int]
    witness_floor: float
    first_hop_landed: bool
    queued_residual_slot_still_live: bool
    next_required_slot: Phase7SameSeedFirstHopRuntimeBindingPacketProbeSlot | None
    canonical_landing_guard_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.scenario_name = str(self.scenario_name).strip()
        self.landing_guard_driver_signature = str(
            self.landing_guard_driver_signature
        ).strip()
        self.binding_packet_driver_signature = str(
            self.binding_packet_driver_signature
        ).strip()
        self.repair_agenda_driver_signature = str(
            self.repair_agenda_driver_signature
        ).strip()
        self.current_rung_status = str(self.current_rung_status).strip()
        self.point_miss_vector = tuple(int(value) for value in self.point_miss_vector)
        self.band_miss_vector = tuple(int(value) for value in self.band_miss_vector)
        self.witness_floor = float(self.witness_floor)
        self.first_hop_landed = bool(self.first_hop_landed)
        self.queued_residual_slot_still_live = bool(
            self.queued_residual_slot_still_live
        )
        self.canonical_landing_guard_digest = tuple(
            str(line).rstrip() for line in self.canonical_landing_guard_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "landing_guard_driver_signature": self.landing_guard_driver_signature,
            "binding_packet_driver_signature": self.binding_packet_driver_signature,
            "repair_agenda_driver_signature": self.repair_agenda_driver_signature,
            "current_rung_status": self.current_rung_status,
            "point_miss_vector": list(self.point_miss_vector),
            "band_miss_vector": list(self.band_miss_vector),
            "witness_floor": self.witness_floor,
            "first_hop_landed": self.first_hop_landed,
            "queued_residual_slot_still_live": self.queued_residual_slot_still_live,
            "next_required_slot": (
                None
                if self.next_required_slot is None
                else self.next_required_slot.to_dict()
            ),
            "canonical_landing_guard_digest": list(self.canonical_landing_guard_digest),
        }


@dataclass(slots=True)
class Phase7SameSeedFirstHopLandingGuardProbe:
    status: str
    canonical_seed_order: tuple[int, ...]
    admission_order_driver_signature: str
    current_frontier_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    runtime_object_flow_focus: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    binding_runtime_packet: Phase7SameSeedFirstHopRuntimeBindingPacketProbePacket
    scenarios: tuple[Phase7SameSeedFirstHopLandingGuardProbeStep, ...]

    def __post_init__(self) -> None:
        self.status = str(self.status).strip()
        self.canonical_seed_order = tuple(
            int(value) for value in self.canonical_seed_order
        )
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.current_frontier_driver_signature = str(
            self.current_frontier_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.runtime_object_flow_focus = tuple(
            str(item).strip() for item in self.runtime_object_flow_focus
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.scenarios = tuple(self.scenarios)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical_seed_order": list(self.canonical_seed_order),
            "admission_order_driver_signature": self.admission_order_driver_signature,
            "current_frontier_driver_signature": self.current_frontier_driver_signature,
            "runtime_witness_path": list(self.runtime_witness_path),
            "runtime_object_flow_focus": list(self.runtime_object_flow_focus),
            "acceptance_readout_path": list(self.acceptance_readout_path),
            "binding_runtime_packet": self.binding_runtime_packet.to_dict(),
            "scenarios": [step.to_dict() for step in self.scenarios],
        }


@dataclass(slots=True)
class Phase7SameSeedObservedRerunRungGuardProbeStep:
    scenario_name: str
    binding_slot_candidate_status: str
    residual_slot_candidate_status: str
    current_rung_candidate_status: str
    resulting_rung_status: str
    downstream_evidence_status: str
    highest_landed_rung: str
    landing_guard_driver_signature: str
    completion_ladder_driver_signature: str
    runtime_bridge_driver_signature: str
    before_after_driver_signature: str
    acceptance_exact_trim_floor_landing_bridge_status: str
    acceptance_exact_trim_floor_inverse_pi_concentration_status: str
    point_miss_vector: tuple[int, int, int]
    band_miss_vector: tuple[int, int, int]
    admissibility_witness_floor: float
    witness_floor: float
    next_required_slot: Phase7SameSeedCompletionLadderRepairSlot | None
    queued_residual_slot_still_live: bool
    canonical_observed_rerun_rung_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.scenario_name = str(self.scenario_name).strip()
        self.binding_slot_candidate_status = str(
            self.binding_slot_candidate_status
        ).strip()
        self.residual_slot_candidate_status = str(
            self.residual_slot_candidate_status
        ).strip()
        self.current_rung_candidate_status = str(
            self.current_rung_candidate_status
        ).strip()
        self.resulting_rung_status = str(self.resulting_rung_status).strip()
        self.downstream_evidence_status = str(self.downstream_evidence_status).strip()
        self.highest_landed_rung = str(self.highest_landed_rung).strip()
        self.landing_guard_driver_signature = str(
            self.landing_guard_driver_signature
        ).strip()
        self.completion_ladder_driver_signature = str(
            self.completion_ladder_driver_signature
        ).strip()
        self.runtime_bridge_driver_signature = str(
            self.runtime_bridge_driver_signature
        ).strip()
        self.before_after_driver_signature = str(
            self.before_after_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_landing_bridge_status = str(
            self.acceptance_exact_trim_floor_landing_bridge_status
        ).strip()
        self.acceptance_exact_trim_floor_inverse_pi_concentration_status = str(
            self.acceptance_exact_trim_floor_inverse_pi_concentration_status
        ).strip()
        self.point_miss_vector = tuple(int(value) for value in self.point_miss_vector)
        self.band_miss_vector = tuple(int(value) for value in self.band_miss_vector)
        self.admissibility_witness_floor = float(self.admissibility_witness_floor)
        self.witness_floor = float(self.witness_floor)
        self.queued_residual_slot_still_live = bool(
            self.queued_residual_slot_still_live
        )
        self.canonical_observed_rerun_rung_digest = tuple(
            str(line).rstrip() for line in self.canonical_observed_rerun_rung_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "binding_slot_candidate_status": self.binding_slot_candidate_status,
            "residual_slot_candidate_status": self.residual_slot_candidate_status,
            "current_rung_candidate_status": self.current_rung_candidate_status,
            "resulting_rung_status": self.resulting_rung_status,
            "downstream_evidence_status": self.downstream_evidence_status,
            "highest_landed_rung": self.highest_landed_rung,
            "landing_guard_driver_signature": self.landing_guard_driver_signature,
            "completion_ladder_driver_signature": (
                self.completion_ladder_driver_signature
            ),
            "runtime_bridge_driver_signature": self.runtime_bridge_driver_signature,
            "before_after_driver_signature": self.before_after_driver_signature,
            "acceptance_exact_trim_floor_landing_bridge_status": (
                self.acceptance_exact_trim_floor_landing_bridge_status
            ),
            "acceptance_exact_trim_floor_inverse_pi_concentration_status": (
                self.acceptance_exact_trim_floor_inverse_pi_concentration_status
            ),
            "point_miss_vector": list(self.point_miss_vector),
            "band_miss_vector": list(self.band_miss_vector),
            "admissibility_witness_floor": self.admissibility_witness_floor,
            "witness_floor": self.witness_floor,
            "next_required_slot": (
                None
                if self.next_required_slot is None
                else self.next_required_slot.to_dict()
            ),
            "queued_residual_slot_still_live": self.queued_residual_slot_still_live,
            "canonical_observed_rerun_rung_digest": list(
                self.canonical_observed_rerun_rung_digest
            ),
        }


@dataclass(slots=True)
class Phase7SameSeedObservedRerunRungGuardProbe:
    status: str
    canonical_seed_order: tuple[int, ...]
    admission_order_driver_signature: str
    current_frontier_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    exact_trim_floor_value: float
    target_fold3_low_pi_treated_count: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_share_of_fold3_low_pi_count: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    binding_repair_slot: Phase7SameSeedCompletionLadderRepairSlot
    residual_repair_slot: Phase7SameSeedCompletionLadderRepairSlot
    scenarios: tuple[Phase7SameSeedObservedRerunRungGuardProbeStep, ...]

    def __post_init__(self) -> None:
        self.status = str(self.status).strip()
        self.canonical_seed_order = tuple(
            int(value) for value in self.canonical_seed_order
        )
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.current_frontier_driver_signature = str(
            self.current_frontier_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.exact_trim_floor_value = float(self.exact_trim_floor_value)
        self.target_fold3_low_pi_treated_count = int(
            self.target_fold3_low_pi_treated_count
        )
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_share_of_fold3_low_pi_count = float(
            self.target_exact_trim_floor_share_of_fold3_low_pi_count
        )
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.scenarios = tuple(self.scenarios)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical_seed_order": list(self.canonical_seed_order),
            "admission_order_driver_signature": self.admission_order_driver_signature,
            "current_frontier_driver_signature": self.current_frontier_driver_signature,
            "runtime_witness_path": list(self.runtime_witness_path),
            "acceptance_readout_path": list(self.acceptance_readout_path),
            "exact_trim_floor_value": self.exact_trim_floor_value,
            "target_fold3_low_pi_treated_count": (
                self.target_fold3_low_pi_treated_count
            ),
            "target_exact_trim_floor_count": self.target_exact_trim_floor_count,
            "target_exact_trim_floor_share_of_fold3_low_pi_count": (
                self.target_exact_trim_floor_share_of_fold3_low_pi_count
            ),
            "target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi": (
                self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
            ),
            "target_exact_trim_floor_weighted_share_of_fold3_weighted": (
                self.target_exact_trim_floor_weighted_share_of_fold3_weighted
            ),
            "binding_repair_slot": self.binding_repair_slot.to_dict(),
            "residual_repair_slot": self.residual_repair_slot.to_dict(),
            "scenarios": [step.to_dict() for step in self.scenarios],
        }


@dataclass(slots=True)
class Phase7SameSeedFirstHopExecutionQueueProbeStep:
    scenario_name: str
    landing_guard_driver_signature: str
    completion_ladder_driver_signature: str
    candidate_status: str
    resulting_evidence_status: str
    current_rung_status: str
    point_miss_vector: tuple[int, int, int]
    band_miss_vector: tuple[int, int, int]
    witness_floor: float
    selected_slot: Phase7SameSeedFirstHopRuntimeBindingPacketProbeSlot | None
    queued_residual_slot_still_live: bool

    def __post_init__(self) -> None:
        self.scenario_name = str(self.scenario_name).strip()
        self.landing_guard_driver_signature = str(
            self.landing_guard_driver_signature
        ).strip()
        self.completion_ladder_driver_signature = str(
            self.completion_ladder_driver_signature
        ).strip()
        self.candidate_status = str(self.candidate_status).strip()
        self.resulting_evidence_status = str(self.resulting_evidence_status).strip()
        self.current_rung_status = str(self.current_rung_status).strip()
        self.point_miss_vector = tuple(int(value) for value in self.point_miss_vector)
        self.band_miss_vector = tuple(int(value) for value in self.band_miss_vector)
        self.witness_floor = float(self.witness_floor)
        self.queued_residual_slot_still_live = bool(
            self.queued_residual_slot_still_live
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "landing_guard_driver_signature": self.landing_guard_driver_signature,
            "completion_ladder_driver_signature": (
                self.completion_ladder_driver_signature
            ),
            "candidate_status": self.candidate_status,
            "resulting_evidence_status": self.resulting_evidence_status,
            "current_rung_status": self.current_rung_status,
            "point_miss_vector": list(self.point_miss_vector),
            "band_miss_vector": list(self.band_miss_vector),
            "witness_floor": self.witness_floor,
            "selected_slot": (
                None if self.selected_slot is None else self.selected_slot.to_dict()
            ),
            "queued_residual_slot_still_live": self.queued_residual_slot_still_live,
        }


@dataclass(slots=True)
class Phase7SameSeedFirstHopExecutionQueueProbe:
    status: str
    canonical_seed_order: tuple[int, ...]
    admission_order_driver_signature: str
    current_frontier_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    runtime_object_flow_focus: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    current_rung_status: str
    highest_landed_rung: str
    downstream_evidence_status: str
    current_to_next_rung_pointwise_gap_vector: tuple[int, int, int]
    current_to_next_rung_band_gap_vector: tuple[int, int, int]
    current_to_completion_pointwise_gap_vector: tuple[int, int, int]
    current_to_completion_band_gap_vector: tuple[int, int, int]
    current_actionable_packet: Phase7SameSeedFirstHopRuntimeBindingPacketProbePacket
    queued_residual_slot: Phase7SameSeedFirstHopRuntimeBindingPacketProbeSlot
    scenarios: tuple[Phase7SameSeedFirstHopExecutionQueueProbeStep, ...]

    def __post_init__(self) -> None:
        self.status = str(self.status).strip()
        self.canonical_seed_order = tuple(
            int(value) for value in self.canonical_seed_order
        )
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.current_frontier_driver_signature = str(
            self.current_frontier_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.runtime_object_flow_focus = tuple(
            str(item).strip() for item in self.runtime_object_flow_focus
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.current_rung_status = str(self.current_rung_status).strip()
        self.highest_landed_rung = str(self.highest_landed_rung).strip()
        self.downstream_evidence_status = str(self.downstream_evidence_status).strip()
        self.current_to_next_rung_pointwise_gap_vector = tuple(
            int(value) for value in self.current_to_next_rung_pointwise_gap_vector
        )
        self.current_to_next_rung_band_gap_vector = tuple(
            int(value) for value in self.current_to_next_rung_band_gap_vector
        )
        self.current_to_completion_pointwise_gap_vector = tuple(
            int(value) for value in self.current_to_completion_pointwise_gap_vector
        )
        self.current_to_completion_band_gap_vector = tuple(
            int(value) for value in self.current_to_completion_band_gap_vector
        )
        self.scenarios = tuple(self.scenarios)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical_seed_order": list(self.canonical_seed_order),
            "admission_order_driver_signature": self.admission_order_driver_signature,
            "current_frontier_driver_signature": self.current_frontier_driver_signature,
            "runtime_witness_path": list(self.runtime_witness_path),
            "runtime_object_flow_focus": list(self.runtime_object_flow_focus),
            "acceptance_readout_path": list(self.acceptance_readout_path),
            "current_rung_status": self.current_rung_status,
            "highest_landed_rung": self.highest_landed_rung,
            "downstream_evidence_status": self.downstream_evidence_status,
            "current_to_next_rung_pointwise_gap_vector": list(
                self.current_to_next_rung_pointwise_gap_vector
            ),
            "current_to_next_rung_band_gap_vector": list(
                self.current_to_next_rung_band_gap_vector
            ),
            "current_to_completion_pointwise_gap_vector": list(
                self.current_to_completion_pointwise_gap_vector
            ),
            "current_to_completion_band_gap_vector": list(
                self.current_to_completion_band_gap_vector
            ),
            "current_actionable_packet": self.current_actionable_packet.to_dict(),
            "queued_residual_slot": self.queued_residual_slot.to_dict(),
            "scenarios": [step.to_dict() for step in self.scenarios],
        }


def _phase7_same_seed_completion_slot_matches_runtime_binding_slot(
    completion_slot: Phase7SameSeedCompletionLadderRepairSlot,
    runtime_slot: Phase7SameSeedFirstHopRuntimeBindingPacketProbeSlot,
) -> bool:
    return (
        completion_slot.random_state == runtime_slot.random_state
        and completion_slot.seed_group == runtime_slot.seed_group
        and completion_slot.z_index == runtime_slot.z_index
        and abs(completion_slot.z_value - runtime_slot.z_value) <= 1e-12
    )


def _phase7_same_seed_first_hop_execution_queue_step(
    *,
    scenario_name: str,
    landing_guard_driver_signature: str,
    completion_ladder_driver_signature: str,
    candidate_status: str,
    resulting_evidence_status: str,
    current_rung_status: str,
    point_miss_vector: Sequence[int],
    band_miss_vector: Sequence[int],
    witness_floor: float,
    selected_slot: Phase7SameSeedFirstHopRuntimeBindingPacketProbeSlot | None,
    queued_residual_slot_still_live: bool,
) -> Phase7SameSeedFirstHopExecutionQueueProbeStep:
    return Phase7SameSeedFirstHopExecutionQueueProbeStep(
        scenario_name=scenario_name,
        landing_guard_driver_signature=landing_guard_driver_signature,
        completion_ladder_driver_signature=completion_ladder_driver_signature,
        candidate_status=candidate_status,
        resulting_evidence_status=resulting_evidence_status,
        current_rung_status=current_rung_status,
        point_miss_vector=tuple(int(value) for value in point_miss_vector),
        band_miss_vector=tuple(int(value) for value in band_miss_vector),
        witness_floor=float(witness_floor),
        selected_slot=selected_slot,
        queued_residual_slot_still_live=queued_residual_slot_still_live,
    )


def _phase7_same_seed_trace_payload_cache_step(
    *,
    helper_name: str,
    before: Any,
    after: Any,
) -> Phase7SameSeedTracePayloadCacheStep:
    return Phase7SameSeedTracePayloadCacheStep(
        helper_name=helper_name,
        hits=int(after.hits),
        misses=int(after.misses),
        delta_hits=int(after.hits) - int(before.hits),
        delta_misses=int(after.misses) - int(before.misses),
    )


def _run_phase7_same_seed_trace_payload_cache_step_sequence(
    *,
    cache_info_fn: Any,
    helper_sequence: Sequence[tuple[str, Any]],
) -> tuple[Phase7SameSeedTracePayloadCacheStep, ...]:
    steps: list[Phase7SameSeedTracePayloadCacheStep] = []
    before = cache_info_fn()
    for helper_name, helper in helper_sequence:
        helper()
        after = cache_info_fn()
        steps.append(
            _phase7_same_seed_trace_payload_cache_step(
                helper_name=helper_name,
                before=before,
                after=after,
            )
        )
        before = after
    return tuple(steps)


def _phase7_design_cache_key(
    design: MonteCarloDesign,
) -> tuple[
    str,
    int,
    int,
    str,
    int,
    str,
    float,
    float,
    int,
    float,
    float,
    tuple[float, ...],
]:
    return (
        design.dgp_name,
        design.n_obs,
        design.p,
        design.basis_family,
        design.basis_degree,
        design.oracle_lane,
        design.alpha,
        design.rho_x,
        design.n_folds,
        design.trim_lower,
        design.trim_upper,
        tuple(
            float(value) for value in np.asarray(design.evaluation_grid, dtype=float)
        ),
    )


def _phase7_design_from_cache_key(
    cache_key: tuple[
        str,
        int,
        int,
        str,
        int,
        str,
        float,
        float,
        int,
        float,
        float,
        tuple[float, ...],
    ],
) -> MonteCarloDesign:
    (
        dgp_name,
        n_obs,
        p,
        basis_family,
        basis_degree,
        oracle_lane,
        alpha,
        rho_x,
        n_folds,
        trim_lower,
        trim_upper,
        evaluation_grid,
    ) = cache_key
    return MonteCarloDesign(
        dgp_name=dgp_name,
        n_obs=n_obs,
        p=p,
        basis_family=basis_family,
        basis_degree=basis_degree,
        oracle_lane=oracle_lane,
        alpha=alpha,
        rho_x=rho_x,
        n_folds=n_folds,
        trim_lower=trim_lower,
        trim_upper=trim_upper,
        evaluation_grid=np.asarray(evaluation_grid, dtype=float),
    )


@lru_cache(maxsize=64)
def _fit_phase7_same_seed_replay_payload_cached(
    design_cache_key: tuple[
        str,
        int,
        int,
        str,
        int,
        str,
        float,
        float,
        int,
        float,
        float,
        tuple[float, ...],
    ],
    replication_seed: int,
) -> _Phase7SameSeedReplayPayload:
    design = _phase7_design_from_cache_key(design_cache_key)
    dataset = _generate_dataset(design, random_state=int(replication_seed))
    data = dataset.to_validated_data()
    splits = make_crossfit_splits(
        n_obs=data.n_obs,
        n_folds=design.n_folds,
        random_state=int(replication_seed),
        trim_lower=design.trim_lower,
        trim_upper=design.trim_upper,
    )
    nuisance_payload = CrossfitNuisanceEstimator(oracle_lane=design.oracle_lane).fit(
        data,
        splits,
    )
    score_payload = build_score_payload(data, nuisance_payload)
    estimation_payload, estimation_result = estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
    )
    return _Phase7SameSeedReplayPayload(
        dataset=dataset,
        score_payload=score_payload,
        estimation_payload=estimation_payload,
        estimation_result=estimation_result,
    )


def _fit_phase7_same_seed_replay_payload(
    *,
    design: MonteCarloDesign,
    replication_seed: int,
) -> _Phase7SameSeedReplayPayload:
    return _fit_phase7_same_seed_replay_payload_cached(
        _phase7_design_cache_key(design),
        int(replication_seed),
    )


_fit_phase7_same_seed_replay_payload.cache_clear = (  # type: ignore[attr-defined]
    _fit_phase7_same_seed_replay_payload_cached.cache_clear
)
_fit_phase7_same_seed_replay_payload.cache_info = (  # type: ignore[attr-defined]
    _fit_phase7_same_seed_replay_payload_cached.cache_info
)


@lru_cache(maxsize=64)
def _fit_phase7_same_seed_nonparametric_payload_cached(
    design_cache_key: tuple[
        str,
        int,
        int,
        str,
        int,
        str,
        float,
        float,
        int,
        float,
        float,
        tuple[float, ...],
    ],
    replication_seed: int,
    n_boot: int,
) -> _Phase7SameSeedNonparametricPayload:
    design = _phase7_design_from_cache_key(design_cache_key)
    replay_payload = _fit_phase7_same_seed_replay_payload_cached(
        design_cache_key,
        int(replication_seed),
    )
    nonparametric_payload, _ = estimate_nonparametric_inference(
        replay_payload.score_payload,
        replay_payload.estimation_payload,
        result=replay_payload.estimation_result,
        alpha=design.alpha,
        lambda_double_prime=0.0,
        n_boot=int(n_boot),
        random_state=int(replication_seed),
    )
    return _Phase7SameSeedNonparametricPayload(
        dataset=replay_payload.dataset,
        score_payload=replay_payload.score_payload,
        estimation_payload=replay_payload.estimation_payload,
        estimation_result=replay_payload.estimation_result,
        nonparametric_payload=nonparametric_payload,
    )


def _fit_phase7_same_seed_nonparametric_payload(
    *,
    design: MonteCarloDesign,
    replication_seed: int,
    n_boot: int,
) -> _Phase7SameSeedNonparametricPayload:
    return _fit_phase7_same_seed_nonparametric_payload_cached(
        _phase7_design_cache_key(design),
        int(replication_seed),
        int(n_boot),
    )


_fit_phase7_same_seed_nonparametric_payload.cache_clear = (  # type: ignore[attr-defined]
    _fit_phase7_same_seed_nonparametric_payload_cached.cache_clear
)
_fit_phase7_same_seed_nonparametric_payload.cache_info = (  # type: ignore[attr-defined]
    _fit_phase7_same_seed_nonparametric_payload_cached.cache_info
)


def _fit_phase7_nonparametric_object_payload(
    *,
    design: MonteCarloDesign,
    replication_seed: int,
    n_boot: int,
) -> tuple[SimulationDataset, Any]:
    cached_payload = _fit_phase7_same_seed_nonparametric_payload(
        design=design,
        replication_seed=int(replication_seed),
        n_boot=int(n_boot),
    )
    return cached_payload.dataset, cached_payload.nonparametric_payload


def run_phase7_same_seed_trace_payload_cache_bridge_probe() -> (
    Phase7SameSeedTracePayloadCacheBridgeProbe
):
    import importlib

    seedwise_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition"
    )
    score_input_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_score_input_trace"
    )
    raw_score_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace"
    )
    rho_support_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace"
    )
    low_pi_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace"
    )
    runtime_evidence_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence"
    )
    estimation_source_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_estimation_source_trace"
    )

    seedwise_report = seedwise_module.run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    seed_observations = (
        seedwise_report.seed_observation(303),
        seedwise_report.seed_observation(202),
        seedwise_report.seed_observation(707),
    )

    def _run_score_input_stage() -> None:
        for observation in seed_observations:
            score_input_module._build_seed_slice(
                random_state=observation.random_state,
                seed_group=observation.seed_group,
                replication_seed=observation.replication_seed,
            )

    def _run_raw_score_stage() -> None:
        for observation in seed_observations:
            raw_score_module._build_component_slice(
                random_state=observation.random_state,
                seed_group=observation.seed_group,
                replication_seed=observation.replication_seed,
            )

    def _run_rho_support_stage() -> None:
        for observation in seed_observations:
            rho_support_module._build_support_slice(
                random_state=observation.random_state,
                seed_group=observation.seed_group,
                replication_seed=observation.replication_seed,
            )

    def _run_low_pi_stage() -> None:
        for observation in seed_observations:
            low_pi_module._build_low_pi_slice(
                random_state=observation.random_state,
                seed_group=observation.seed_group,
                replication_seed=observation.replication_seed,
            )

    def _run_runtime_evidence_stage() -> None:
        for observation in seed_observations:
            runtime_evidence_module._build_runtime_snapshot(
                random_state=observation.random_state,
                seed_group=observation.seed_group,
                replication_seed=observation.replication_seed,
            )

    def _run_estimation_source_stage() -> None:
        estimation_source_module._build_target_source_trace(
            replication_seed=seed_observations[0].replication_seed,
        )

    replay_helper_sequence = (
        ("score_input", _run_score_input_stage),
        ("raw_score", _run_raw_score_stage),
        ("rho_support", _run_rho_support_stage),
        ("low_pi", _run_low_pi_stage),
    )
    nonparametric_helper_sequence = (
        ("runtime_evidence", _run_runtime_evidence_stage),
        ("estimation_source", _run_estimation_source_stage),
    )

    _fit_phase7_same_seed_replay_payload.cache_clear()
    _fit_phase7_same_seed_nonparametric_payload.cache_clear()

    try:
        replay_steps = _run_phase7_same_seed_trace_payload_cache_step_sequence(
            cache_info_fn=_fit_phase7_same_seed_replay_payload.cache_info,
            helper_sequence=replay_helper_sequence,
        )
        _fit_phase7_same_seed_replay_payload.cache_clear()
        _fit_phase7_same_seed_nonparametric_payload.cache_clear()
        nonparametric_steps = _run_phase7_same_seed_trace_payload_cache_step_sequence(
            cache_info_fn=_fit_phase7_same_seed_nonparametric_payload.cache_info,
            helper_sequence=nonparametric_helper_sequence,
        )
    finally:
        _fit_phase7_same_seed_replay_payload.cache_clear()
        _fit_phase7_same_seed_nonparametric_payload.cache_clear()

    replay_cache_satisfied = bool(replay_steps) and replay_steps[0].delta_misses >= 1
    replay_cache_satisfied = replay_cache_satisfied and all(
        step.delta_misses == 0 and step.delta_hits >= 1 for step in replay_steps[1:]
    )
    nonparametric_cache_satisfied = (
        bool(nonparametric_steps) and nonparametric_steps[0].delta_misses >= 1
    )
    nonparametric_cache_satisfied = nonparametric_cache_satisfied and all(
        step.delta_misses == 0 and step.delta_hits >= 1
        for step in nonparametric_steps[1:]
    )

    return Phase7SameSeedTracePayloadCacheBridgeProbe(
        status=(
            "same-seed-trace-payload-cache-bridge-satisfied"
            if replay_cache_satisfied and nonparametric_cache_satisfied
            else "same-seed-trace-payload-cache-bridge-open"
        ),
        replay_steps=replay_steps,
        nonparametric_steps=nonparametric_steps,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_runtime_bridge_source_guard() -> Any:
    import importlib

    source_guard_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard"
    )

    return source_guard_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_source_guard()


@lru_cache(maxsize=1)
def run_phase7_same_seed_runtime_bridge_after_report_admission_order() -> Any:
    import importlib

    admission_order_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_admission_order"
    )

    return admission_order_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_admission_order()


@lru_cache(maxsize=1)
def run_phase7_same_seed_runtime_bridge_after_report_intake() -> Any:
    import importlib

    intake_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake"
    )

    return intake_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_intake()


@lru_cache(maxsize=1)
def run_phase7_same_seed_runtime_bridge_after_report_frontier_packet() -> Any:
    import importlib

    frontier_packet_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_frontier_packet"
    )

    return frontier_packet_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_runtime_bridge_after_report_frontier_packet()


@lru_cache(maxsize=1)
def run_phase7_same_seed_exact_witness_target_gap() -> Any:
    import importlib

    target_gap_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap"
    )

    return target_gap_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_target_gap()


def _phase7_same_seed_before_after_acceptance_step(
    *,
    scenario_name: str,
    report: Any,
    seed_consistent_digest_verified: bool,
    candidate_status_if_realized: str = "",
    resulting_evidence_status_if_realized: str = "",
) -> Phase7SameSeedBeforeAfterAcceptanceStep:
    return Phase7SameSeedBeforeAfterAcceptanceStep(
        scenario_name=scenario_name,
        driver_signature=report.driver_signature,
        point_miss_vector=report.candidate_point_miss_vector,
        band_miss_vector=report.candidate_band_miss_vector,
        witness_floor=report.candidate_witness_floor,
        floor_lift_observed=report.floor_lift_observed,
        left_guard_band_preserved=report.left_guard_band_preserved,
        left_guard_pointwise_nonregression=report.left_guard_pointwise_nonregression,
        pointwise_total_miss_reduced=report.pointwise_total_miss_reduced,
        right_center_lane_residual_only=report.right_center_lane_residual_only,
        acceptance_passed=report.acceptance_passed,
        seed_consistent_digest_verified=seed_consistent_digest_verified,
        candidate_status_if_realized=candidate_status_if_realized,
        resulting_evidence_status_if_realized=resulting_evidence_status_if_realized,
    )


_PHASE7_SAME_SEED_BEFORE_AFTER_ACCEPTANCE_CANONICAL_SEED_ORDER = (
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
)


@lru_cache(maxsize=1)
def run_phase7_same_seed_before_after_acceptance_probe() -> (
    Phase7SameSeedBeforeAfterAcceptanceProbe
):
    import importlib

    before_after_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe"
    )
    candidate_profile_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_candidate_profile"
    )
    admission_order_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order"
    )
    seedwise_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition"
    )

    baseline_report = seedwise_module.run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    admission_order_report = admission_order_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order()
    identity_report = before_after_module.build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report(
        before_report=baseline_report,
        after_report=baseline_report,
        admission_order_report=admission_order_report,
    )
    candidate_profile = candidate_profile_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_candidate_profile()
    exact_witness_target_report = before_after_module.build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_report(
        before_report=baseline_report,
        after_report=candidate_profile.target_after_report,
        admission_order_report=admission_order_report,
    )

    canonical_seed_order = tuple(
        int(value) for value in identity_report.same_seed_random_states
    )
    if (
        canonical_seed_order
        != _PHASE7_SAME_SEED_BEFORE_AFTER_ACCEPTANCE_CANONICAL_SEED_ORDER
    ):
        raise ValueError(
            "same-seed before/after acceptance probe requires the canonical seed order "
            f"{_PHASE7_SAME_SEED_BEFORE_AFTER_ACCEPTANCE_CANONICAL_SEED_ORDER!r}"
        )
    if (
        tuple(
            int(value) for value in exact_witness_target_report.same_seed_random_states
        )
        != canonical_seed_order
    ):
        raise ValueError(
            "same-seed before/after acceptance probe requires exact-witness targets to keep the canonical seed order"
        )
    if (
        tuple(int(value) for value in candidate_profile.same_seed_random_states)
        != canonical_seed_order
    ):
        raise ValueError(
            "same-seed before/after acceptance probe requires the exact-witness candidate profile to keep the canonical seed order"
        )
    if (
        exact_witness_target_report.candidate_point_miss_vector
        != candidate_profile.target_point_miss_vector
    ):
        raise ValueError(
            "same-seed before/after acceptance probe requires the exact-witness target point miss vector"
        )
    if (
        exact_witness_target_report.candidate_band_miss_vector
        != candidate_profile.target_band_miss_vector
    ):
        raise ValueError(
            "same-seed before/after acceptance probe requires the exact-witness target band miss vector"
        )

    scenarios = (
        _phase7_same_seed_before_after_acceptance_step(
            scenario_name="identity_replay",
            report=identity_report,
            seed_consistent_digest_verified=True,
        ),
        _phase7_same_seed_before_after_acceptance_step(
            scenario_name="exact_witness_target",
            report=exact_witness_target_report,
            seed_consistent_digest_verified=True,
            candidate_status_if_realized=(
                candidate_profile.candidate_status_if_realized
            ),
            resulting_evidence_status_if_realized=(
                candidate_profile.resulting_evidence_status_if_realized
            ),
        ),
    )
    probe_satisfied = (
        not scenarios[0].acceptance_passed
        and scenarios[1].acceptance_passed
        and scenarios[1].witness_floor
        >= exact_witness_target_report.required_min_witness_floor
    )

    return Phase7SameSeedBeforeAfterAcceptanceProbe(
        status=(
            "same-seed-before-after-acceptance-probe-satisfied"
            if probe_satisfied
            else "same-seed-before-after-acceptance-probe-open"
        ),
        canonical_seed_order=canonical_seed_order,
        admission_order_driver_signature="same-seed-admission-order",
        required_min_witness_floor=exact_witness_target_report.required_min_witness_floor,
        runtime_witness_path=candidate_profile.runtime_witness_path,
        left_guard_grid_value=candidate_profile.left_guard_grid_value,
        residual_lane_grid_values=candidate_profile.residual_lane_grid_values,
        scenarios=scenarios,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_admission_order() -> Any:
    import importlib

    admission_order_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order"
    )

    return admission_order_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order()


def _phase7_same_seed_completion_ladder_step(
    *,
    scenario_name: str,
    driver_signature: str,
    candidate_status: str,
    resulting_evidence_status: str,
    before_after_driver_signature: str,
    point_miss_vector: tuple[int, int, int],
    band_miss_vector: tuple[int, int, int],
    witness_floor: float,
    next_repair_slot: Phase7SameSeedCompletionLadderRepairSlot | None,
    queued_residual_slot_still_live: bool,
) -> Phase7SameSeedCompletionLadderStep:
    return Phase7SameSeedCompletionLadderStep(
        scenario_name=scenario_name,
        driver_signature=driver_signature,
        candidate_status=candidate_status,
        resulting_evidence_status=resulting_evidence_status,
        before_after_driver_signature=before_after_driver_signature,
        point_miss_vector=point_miss_vector,
        band_miss_vector=band_miss_vector,
        witness_floor=witness_floor,
        next_repair_slot=next_repair_slot,
        queued_residual_slot_still_live=queued_residual_slot_still_live,
    )


def _phase7_same_seed_repair_agenda_slot(
    slot: Any,
) -> Phase7SameSeedRepairAgendaProbeSlot:
    return Phase7SameSeedRepairAgendaProbeSlot(
        random_state=slot.random_state,
        seed_group=slot.seed_group,
        z_index=slot.z_index,
        z_value=slot.z_value,
        repair_stage=slot.repair_stage,
        repair_order=slot.repair_order,
        current_covered=slot.current_covered,
        completion_target_covered=slot.completion_target_covered,
        currently_actionable=slot.currently_actionable,
    )


def _phase7_same_seed_gap_vector(
    current_vector: Sequence[int],
    target_vector: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        max(int(current_value) - int(target_value), 0)
        for current_value, target_value in zip(current_vector, target_vector)
    )


def _phase7_same_seed_slot_matches_completion_slot(
    slot: Phase7SameSeedRepairAgendaProbeSlot,
    completion_slot: Phase7SameSeedCompletionLadderRepairSlot,
) -> bool:
    return (
        slot.random_state == completion_slot.random_state
        and slot.seed_group == completion_slot.seed_group
        and slot.z_index == completion_slot.z_index
        and abs(slot.z_value - completion_slot.z_value) <= 1e-12
    )


def _phase7_same_seed_slot_matches_runtime_binding_packet_slot(
    slot: Phase7SameSeedRepairAgendaProbeSlot,
    binding_slot: Phase7SameSeedFirstHopRuntimeBindingPacketProbeSlot,
) -> bool:
    return (
        slot.random_state == binding_slot.random_state
        and slot.seed_group == binding_slot.seed_group
        and slot.z_index == binding_slot.z_index
        and abs(slot.z_value - binding_slot.z_value) <= 1e-12
        and slot.repair_stage == binding_slot.repair_stage
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_completion_ladder_probe() -> (
    Phase7SameSeedCompletionLadderProbe
):
    import importlib

    admission_order_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order"
    )
    runtime_contract_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract"
    )
    runtime_candidate_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_runtime_witness_candidate_guard"
    )
    binding_guard_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard"
    )
    binding_progress_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile"
    )
    residual_completion_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile"
    )

    admission_order_report = admission_order_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_admission_order()
    current_open_guard = binding_guard_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard()
    binding_progress_report = binding_progress_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile()
    residual_completion_report = residual_completion_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile()
    before_after_probe = run_phase7_same_seed_before_after_acceptance_probe()

    canonical_seed_order = tuple(
        int(value) for value in admission_order_report.same_seed_random_states
    )
    if (
        canonical_seed_order
        != _PHASE7_SAME_SEED_BEFORE_AFTER_ACCEPTANCE_CANONICAL_SEED_ORDER
    ):
        raise ValueError(
            "same-seed completion ladder probe requires the canonical seed order "
            f"{_PHASE7_SAME_SEED_BEFORE_AFTER_ACCEPTANCE_CANONICAL_SEED_ORDER!r}"
        )
    if (
        tuple(int(value) for value in current_open_guard.same_seed_random_states)
        != canonical_seed_order
    ):
        raise ValueError(
            "same-seed completion ladder probe requires the binding-slot candidate guard to keep the canonical seed order"
        )
    if (
        tuple(int(value) for value in binding_progress_report.same_seed_random_states)
        != canonical_seed_order
    ):
        raise ValueError(
            "same-seed completion ladder probe requires the binding-slot progress profile to keep the canonical seed order"
        )
    if (
        tuple(
            int(value) for value in residual_completion_report.same_seed_random_states
        )
        != canonical_seed_order
    ):
        raise ValueError(
            "same-seed completion ladder probe requires the residual-slot completion profile to keep the canonical seed order"
        )
    if before_after_probe.canonical_seed_order != canonical_seed_order:
        raise ValueError(
            "same-seed completion ladder probe requires the before/after acceptance probe to keep the canonical seed order"
        )

    runtime_contract_report = runtime_contract_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_first_sine_preserve_left_support_runtime_witness_contract()
    runtime_candidate = runtime_candidate_module.Phase7MonteCarloWideningPolicyCoverageAnchorShoulderFirstSineRuntimeWitnessCandidate(
        runtime_witness_path=runtime_contract_report.runtime_witness_path,
        preserves_left_support_contract=runtime_contract_report.preserves_left_support_contract,
        diagonal_preserved=runtime_contract_report.diagonal_preserved,
        direct_covariance_edits_allowed=runtime_contract_report.direct_covariance_edits_allowed,
        required_patch_share_of_full_shared_vf_gap=runtime_contract_report.required_patch_share_of_full_shared_vf_gap,
        required_patch_share_of_omega_only_shared_vf_increment=runtime_contract_report.required_patch_share_of_omega_only_shared_vf_increment,
        required_patch_share_of_diagonal_omega_gap=runtime_contract_report.required_patch_share_of_diagonal_omega_gap,
        required_patch_share_of_psd_boundary=runtime_contract_report.required_patch_share_of_psd_boundary,
        compensating_stage_order=runtime_contract_report.compensating_stage_order,
        compensating_cumulative_share_of_total_absolute_mass=runtime_contract_report.compensating_cumulative_share_of_total_absolute_mass,
        compensating_zero_live_entry_count=runtime_contract_report.compensating_zero_live_entry_count,
    )
    binding_slot_landed_guard = binding_guard_module.build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard_report(
        runtime_candidate=runtime_candidate,
        after_report=binding_progress_report.binding_only_after_report,
        runtime_contract_report=runtime_contract_report,
    )

    if (
        current_open_guard.runtime_witness_path
        != binding_progress_report.runtime_witness_path
    ):
        raise ValueError(
            "same-seed completion ladder probe requires the binding-slot guard and progress profile to share one runtime witness path"
        )
    if (
        current_open_guard.runtime_witness_path
        != residual_completion_report.runtime_witness_path
    ):
        raise ValueError(
            "same-seed completion ladder probe requires the residual-slot completion profile to share the runtime witness path"
        )
    if (
        before_after_probe.runtime_witness_path
        != current_open_guard.runtime_witness_path
    ):
        raise ValueError(
            "same-seed completion ladder probe requires the before/after acceptance probe to share the runtime witness path"
        )

    if (
        tuple(before_after_probe.scenarios[0].point_miss_vector)
        != current_open_guard.candidate_point_miss_vector
    ):
        raise ValueError(
            "same-seed completion ladder probe requires the identity replay to match the current open point miss vector"
        )
    if (
        tuple(before_after_probe.scenarios[1].point_miss_vector)
        != residual_completion_report.completion_point_miss_vector
    ):
        raise ValueError(
            "same-seed completion ladder probe requires the exact-witness target to match the completion point miss vector"
        )
    if (
        tuple(before_after_probe.scenarios[1].band_miss_vector)
        != residual_completion_report.completion_band_miss_vector
    ):
        raise ValueError(
            "same-seed completion ladder probe requires the exact-witness target to match the completion band miss vector"
        )

    binding_repair_slot = Phase7SameSeedCompletionLadderRepairSlot(
        random_state=binding_progress_report.binding_repair_slot.random_state,
        seed_group=binding_progress_report.binding_repair_slot.seed_group,
        z_index=binding_progress_report.binding_repair_slot.z_index,
        z_value=binding_progress_report.binding_repair_slot.z_value,
    )
    residual_repair_slot = Phase7SameSeedCompletionLadderRepairSlot(
        random_state=binding_progress_report.residual_repair_slot.random_state,
        seed_group=binding_progress_report.residual_repair_slot.seed_group,
        z_index=binding_progress_report.residual_repair_slot.z_index,
        z_value=binding_progress_report.residual_repair_slot.z_value,
    )

    scenarios = (
        _phase7_same_seed_completion_ladder_step(
            scenario_name="current_open",
            driver_signature=current_open_guard.resulting_binding_progress_status,
            candidate_status=current_open_guard.candidate_status,
            resulting_evidence_status=current_open_guard.downstream_evidence_status,
            before_after_driver_signature=current_open_guard.before_after_driver_signature,
            point_miss_vector=current_open_guard.candidate_point_miss_vector,
            band_miss_vector=current_open_guard.candidate_band_miss_vector,
            witness_floor=binding_progress_report.baseline_witness_floor,
            next_repair_slot=binding_repair_slot,
            queued_residual_slot_still_live=True,
        ),
        _phase7_same_seed_completion_ladder_step(
            scenario_name="binding_slot_landed",
            driver_signature=binding_slot_landed_guard.resulting_binding_progress_status,
            candidate_status=binding_slot_landed_guard.candidate_status,
            resulting_evidence_status=binding_slot_landed_guard.downstream_evidence_status,
            before_after_driver_signature=binding_slot_landed_guard.before_after_driver_signature,
            point_miss_vector=binding_slot_landed_guard.candidate_point_miss_vector,
            band_miss_vector=binding_slot_landed_guard.candidate_band_miss_vector,
            witness_floor=binding_progress_report.projected_binding_witness_floor,
            next_repair_slot=residual_repair_slot,
            queued_residual_slot_still_live=True,
        ),
        _phase7_same_seed_completion_ladder_step(
            scenario_name="completion_admissible",
            driver_signature=residual_completion_report.driver_signature,
            candidate_status=residual_completion_report.completion_candidate_status,
            resulting_evidence_status=(
                residual_completion_report.completion_resulting_evidence_status
            ),
            before_after_driver_signature=(
                residual_completion_report.completion_before_after_driver_signature
            ),
            point_miss_vector=residual_completion_report.completion_point_miss_vector,
            band_miss_vector=residual_completion_report.completion_band_miss_vector,
            witness_floor=residual_completion_report.completion_witness_floor,
            next_repair_slot=None,
            queued_residual_slot_still_live=False,
        ),
    )

    return Phase7SameSeedCompletionLadderProbe(
        status="same-seed-completion-ladder-probe-satisfied",
        canonical_seed_order=canonical_seed_order,
        admission_order_driver_signature=admission_order_report.driver_signature,
        current_frontier_driver_signature=(
            "same-seed-exact-witness-observed-rerun-live-gap-packet-open"
        ),
        binding_progress_driver_signature=binding_progress_report.driver_signature,
        residual_completion_driver_signature=residual_completion_report.driver_signature,
        runtime_witness_path=current_open_guard.runtime_witness_path,
        required_min_witness_floor=residual_completion_report.required_min_witness_floor,
        binding_repair_slot=binding_repair_slot,
        residual_repair_slot=residual_repair_slot,
        scenarios=scenarios,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_repair_agenda_probe() -> Phase7SameSeedRepairAgendaProbe:
    import importlib

    repair_agenda_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda"
    )

    repair_agenda_report = repair_agenda_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_repair_agenda()
    before_after_probe = run_phase7_same_seed_before_after_acceptance_probe()
    completion_ladder_probe = run_phase7_same_seed_completion_ladder_probe()

    canonical_seed_order = tuple(
        int(value) for value in repair_agenda_report.same_seed_random_states
    )
    if (
        canonical_seed_order
        != _PHASE7_SAME_SEED_BEFORE_AFTER_ACCEPTANCE_CANONICAL_SEED_ORDER
    ):
        raise ValueError(
            "same-seed repair agenda probe requires the canonical seed order "
            f"{_PHASE7_SAME_SEED_BEFORE_AFTER_ACCEPTANCE_CANONICAL_SEED_ORDER!r}"
        )
    if before_after_probe.canonical_seed_order != canonical_seed_order:
        raise ValueError(
            "same-seed repair agenda probe requires the before/after acceptance probe to keep the canonical seed order"
        )
    if completion_ladder_probe.canonical_seed_order != canonical_seed_order:
        raise ValueError(
            "same-seed repair agenda probe requires the completion ladder probe to keep the canonical seed order"
        )
    if tuple(before_after_probe.runtime_witness_path) != tuple(
        repair_agenda_report.runtime_witness_path
    ):
        raise ValueError(
            "same-seed repair agenda probe requires the before/after acceptance probe to share the runtime witness path"
        )
    if tuple(completion_ladder_probe.runtime_witness_path) != tuple(
        repair_agenda_report.runtime_witness_path
    ):
        raise ValueError(
            "same-seed repair agenda probe requires the completion ladder probe to share the runtime witness path"
        )

    current_step = before_after_probe.scenarios[0]
    binding_step = completion_ladder_probe.scenarios[1]
    completion_step = completion_ladder_probe.scenarios[-1]
    if tuple(current_step.point_miss_vector) != tuple(
        repair_agenda_report.current_point_miss_vector
    ):
        raise ValueError(
            "same-seed repair agenda probe requires the current repair agenda point miss vector to match the current before/after identity replay"
        )
    if tuple(current_step.band_miss_vector) != tuple(
        repair_agenda_report.current_band_miss_vector
    ):
        raise ValueError(
            "same-seed repair agenda probe requires the current repair agenda band miss vector to match the current before/after identity replay"
        )
    if (
        abs(
            float(current_step.witness_floor)
            - repair_agenda_report.current_witness_floor
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed repair agenda probe requires the current repair agenda witness floor to match the current before/after identity replay"
        )

    expected_next_point_gap = _phase7_same_seed_gap_vector(
        current_step.point_miss_vector, binding_step.point_miss_vector
    )
    expected_next_band_gap = _phase7_same_seed_gap_vector(
        current_step.band_miss_vector, binding_step.band_miss_vector
    )
    expected_completion_point_gap = _phase7_same_seed_gap_vector(
        current_step.point_miss_vector, completion_step.point_miss_vector
    )
    expected_completion_band_gap = _phase7_same_seed_gap_vector(
        current_step.band_miss_vector, completion_step.band_miss_vector
    )
    if tuple(repair_agenda_report.current_to_next_rung_pointwise_gap_vector) != tuple(
        expected_next_point_gap
    ):
        raise ValueError(
            "same-seed repair agenda probe requires the next-rung pointwise gap to match the completion ladder delta"
        )
    if tuple(repair_agenda_report.current_to_next_rung_band_gap_vector) != tuple(
        expected_next_band_gap
    ):
        raise ValueError(
            "same-seed repair agenda probe requires the next-rung band gap to match the completion ladder delta"
        )
    if tuple(repair_agenda_report.current_to_completion_pointwise_gap_vector) != tuple(
        expected_completion_point_gap
    ):
        raise ValueError(
            "same-seed repair agenda probe requires the completion pointwise gap to match the completion ladder delta"
        )
    if tuple(repair_agenda_report.current_to_completion_band_gap_vector) != tuple(
        expected_completion_band_gap
    ):
        raise ValueError(
            "same-seed repair agenda probe requires the completion band gap to match the completion ladder delta"
        )

    next_required_slot = (
        None
        if repair_agenda_report.next_required_slot is None
        else _phase7_same_seed_repair_agenda_slot(
            repair_agenda_report.next_required_slot
        )
    )
    pending_repair_slots = tuple(
        _phase7_same_seed_repair_agenda_slot(slot)
        for slot in repair_agenda_report.pending_repair_slots
    )
    if (
        repair_agenda_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-open"
    ):
        if len(pending_repair_slots) != 2 or next_required_slot is None:
            raise ValueError(
                "same-seed repair agenda probe requires two ordered repair slots on the open agenda"
            )
        if not _phase7_same_seed_slot_matches_completion_slot(
            next_required_slot,
            completion_ladder_probe.binding_repair_slot,
        ):
            raise ValueError(
                "same-seed repair agenda probe requires the actionable repair slot to match the binding repair slot"
            )
        if not _phase7_same_seed_slot_matches_completion_slot(
            pending_repair_slots[1],
            completion_ladder_probe.residual_repair_slot,
        ):
            raise ValueError(
                "same-seed repair agenda probe requires the queued repair slot to match the residual repair slot"
            )
    elif (
        repair_agenda_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-residual-only"
    ):
        if len(pending_repair_slots) != 1 or next_required_slot is None:
            raise ValueError(
                "same-seed repair agenda probe requires one actionable residual slot on the residual-only agenda"
            )
        if not _phase7_same_seed_slot_matches_completion_slot(
            next_required_slot,
            completion_ladder_probe.residual_repair_slot,
        ):
            raise ValueError(
                "same-seed repair agenda probe requires the residual-only slot to match the residual repair slot"
            )
    elif (
        repair_agenda_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-repair-agenda-closed"
    ):
        if next_required_slot is not None or pending_repair_slots:
            raise ValueError(
                "same-seed repair agenda probe requires the closed agenda to clear all pending repair slots"
            )

    return Phase7SameSeedRepairAgendaProbe(
        status="same-seed-repair-agenda-probe-satisfied",
        canonical_seed_order=canonical_seed_order,
        admission_order_driver_signature=(
            completion_ladder_probe.admission_order_driver_signature
        ),
        current_frontier_driver_signature=(
            completion_ladder_probe.current_frontier_driver_signature
        ),
        repair_agenda_driver_signature=repair_agenda_report.driver_signature,
        current_before_after_driver_signature=current_step.driver_signature,
        current_rung_status=repair_agenda_report.current_rung_status,
        highest_landed_rung=repair_agenda_report.highest_landed_rung,
        downstream_evidence_status=repair_agenda_report.downstream_evidence_status,
        runtime_witness_path=repair_agenda_report.runtime_witness_path,
        required_min_witness_floor=repair_agenda_report.required_min_witness_floor,
        current_point_miss_vector=repair_agenda_report.current_point_miss_vector,
        current_band_miss_vector=repair_agenda_report.current_band_miss_vector,
        current_witness_floor=repair_agenda_report.current_witness_floor,
        current_to_next_rung_pointwise_gap_vector=(
            repair_agenda_report.current_to_next_rung_pointwise_gap_vector
        ),
        current_to_next_rung_band_gap_vector=(
            repair_agenda_report.current_to_next_rung_band_gap_vector
        ),
        current_to_completion_pointwise_gap_vector=(
            repair_agenda_report.current_to_completion_pointwise_gap_vector
        ),
        current_to_completion_band_gap_vector=(
            repair_agenda_report.current_to_completion_band_gap_vector
        ),
        next_required_slot=next_required_slot,
        pending_repair_slots=pending_repair_slots,
        canonical_repair_agenda_digest=(
            repair_agenda_report.canonical_observed_rerun_repair_agenda_digest
        ),
    )


def _phase7_same_seed_first_hop_runtime_binding_packet_probe_slot(
    *,
    random_state: int,
    seed_group: str,
    z_index: int,
    z_value: float,
    repair_stage: str,
    repair_role: str,
    repair_priority: int,
) -> Phase7SameSeedFirstHopRuntimeBindingPacketProbeSlot:
    return Phase7SameSeedFirstHopRuntimeBindingPacketProbeSlot(
        random_state=random_state,
        seed_group=seed_group,
        z_index=z_index,
        z_value=z_value,
        repair_stage=repair_stage,
        repair_role=repair_role,
        repair_priority=repair_priority,
    )


def _phase7_same_seed_first_hop_runtime_binding_packet_probe_packet(
    packet: Any,
) -> Phase7SameSeedFirstHopRuntimeBindingPacketProbePacket:
    return Phase7SameSeedFirstHopRuntimeBindingPacketProbePacket(
        random_state=packet.random_state,
        seed_group=packet.seed_group,
        replication_seed=packet.replication_seed,
        z_index=packet.z_index,
        z_value=packet.z_value,
        repair_stage=packet.repair_stage,
        repair_role=packet.repair_role,
        repair_priority=packet.repair_priority,
        target_fold_id=packet.target_fold_id,
        target_exact_trim_floor_count=packet.target_exact_trim_floor_count,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            packet.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            packet.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            packet.target_exact_trim_floor_weighted_retention
        ),
        baseline_open_witness_floor_gap=packet.baseline_open_witness_floor_gap,
        share_of_baseline_open_witness_floor_gap=(
            packet.share_of_baseline_open_witness_floor_gap
        ),
        center_truth=packet.center_truth,
        center_estimate=packet.center_estimate,
        center_error=packet.center_error,
        center_pointwise_interval_lower=packet.center_pointwise_interval_lower,
        center_pointwise_interval_upper=packet.center_pointwise_interval_upper,
        center_half_interval=packet.center_half_interval,
        center_sigma=packet.center_sigma,
        center_lower_minus_truth=packet.center_lower_minus_truth,
        center_error_to_half_interval_ratio=packet.center_error_to_half_interval_ratio,
        vf_cross_entry=packet.vf_cross_entry,
        runtime_object_flow_focus=packet.runtime_object_flow_focus,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_runtime_binding_packet_probe() -> (
    Phase7SameSeedFirstHopRuntimeBindingPacketProbe
):
    import importlib

    binding_packet_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet"
    )

    binding_packet_report = binding_packet_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_runtime_binding_packet()
    repair_agenda_probe = run_phase7_same_seed_repair_agenda_probe()

    canonical_seed_order = tuple(
        int(value) for value in binding_packet_report.same_seed_random_states
    )
    if (
        canonical_seed_order
        != _PHASE7_SAME_SEED_BEFORE_AFTER_ACCEPTANCE_CANONICAL_SEED_ORDER
    ):
        raise ValueError(
            "same-seed first-hop runtime binding packet probe requires the canonical seed order "
            f"{_PHASE7_SAME_SEED_BEFORE_AFTER_ACCEPTANCE_CANONICAL_SEED_ORDER!r}"
        )
    if repair_agenda_probe.canonical_seed_order != canonical_seed_order:
        raise ValueError(
            "same-seed first-hop runtime binding packet probe requires the repair agenda probe to keep the canonical seed order"
        )
    if tuple(repair_agenda_probe.runtime_witness_path) != tuple(
        binding_packet_report.runtime_witness_path
    ):
        raise ValueError(
            "same-seed first-hop runtime binding packet probe requires the repair agenda probe to share the runtime witness path"
        )
    if (
        repair_agenda_probe.current_rung_status
        != binding_packet_report.current_rung_status
    ):
        raise ValueError(
            "same-seed first-hop runtime binding packet probe requires the repair agenda probe to share the current rung status"
        )

    next_required_slot = (
        None
        if binding_packet_report.next_required_slot_random_state is None
        else _phase7_same_seed_first_hop_runtime_binding_packet_probe_slot(
            random_state=binding_packet_report.next_required_slot_random_state,
            seed_group=binding_packet_report.next_required_slot_seed_group,
            z_index=binding_packet_report.next_required_slot_z_index,
            z_value=binding_packet_report.next_required_slot_z_value,
            repair_stage=binding_packet_report.next_required_slot_repair_stage,
            repair_role=binding_packet_report.next_required_slot_repair_role,
            repair_priority=binding_packet_report.next_required_slot_repair_priority,
        )
    )
    binding_runtime_packet = (
        _phase7_same_seed_first_hop_runtime_binding_packet_probe_packet(
            binding_packet_report.binding_runtime_packet
        )
    )
    if (
        binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-open"
    ):
        if repair_agenda_probe.next_required_slot is None or next_required_slot is None:
            raise ValueError(
                "same-seed first-hop runtime binding packet probe requires one actionable binding slot on the open agenda"
            )
        if not _phase7_same_seed_slot_matches_runtime_binding_packet_slot(
            repair_agenda_probe.next_required_slot,
            binding_slot=next_required_slot,
        ):
            raise ValueError(
                "same-seed first-hop runtime binding packet probe requires the actionable repair slot to stay aligned with the runtime binding packet"
            )
    elif (
        binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-landed"
    ):
        if repair_agenda_probe.next_required_slot is None or next_required_slot is None:
            raise ValueError(
                "same-seed first-hop runtime binding packet probe requires one residual slot after the binding packet lands"
            )
        if not _phase7_same_seed_slot_matches_runtime_binding_packet_slot(
            repair_agenda_probe.next_required_slot,
            binding_slot=next_required_slot,
        ):
            raise ValueError(
                "same-seed first-hop runtime binding packet probe requires the residual repair slot to stay aligned with the queue after the binding packet lands"
            )
    elif (
        binding_packet_report.driver_signature
        == "same-seed-exact-witness-observed-rerun-first-hop-runtime-binding-packet-closed"
    ):
        if repair_agenda_probe.next_required_slot is not None:
            raise ValueError(
                "same-seed first-hop runtime binding packet probe requires the closed agenda to clear all remaining repair slots"
            )

    return Phase7SameSeedFirstHopRuntimeBindingPacketProbe(
        status="same-seed-first-hop-runtime-binding-packet-probe-satisfied",
        canonical_seed_order=canonical_seed_order,
        admission_order_driver_signature=(
            repair_agenda_probe.admission_order_driver_signature
        ),
        current_frontier_driver_signature=(
            repair_agenda_probe.current_frontier_driver_signature
        ),
        repair_agenda_driver_signature=(
            repair_agenda_probe.repair_agenda_driver_signature
        ),
        current_before_after_driver_signature=(
            repair_agenda_probe.current_before_after_driver_signature
        ),
        first_hop_contract_driver_signature=(
            binding_packet_report.first_hop_contract_driver_signature
        ),
        binding_packet_driver_signature=binding_packet_report.driver_signature,
        seed303_object_flow_driver_signature=(
            binding_packet_report.seed303_object_flow_driver_signature
        ),
        current_rung_status=binding_packet_report.current_rung_status,
        runtime_witness_path=binding_packet_report.runtime_witness_path,
        runtime_object_flow_focus=binding_packet_report.runtime_object_flow_focus,
        actionable_now=binding_packet_report.actionable_now,
        next_required_slot=next_required_slot,
        binding_runtime_packet=binding_runtime_packet,
        canonical_runtime_binding_packet_digest=(
            binding_packet_report.canonical_observed_rerun_first_hop_runtime_binding_packet_digest
        ),
    )


def _phase7_same_seed_runtime_bridge_after_report_scenario_step(
    *,
    scenario_name: str,
    after_report_supplied: bool,
    before_after_driver_signature: str,
    runtime_bridge_driver_signature: str,
    runtime_bridge_state: str,
    intake_disposition: str,
    point_miss_vector: tuple[int, int, int],
    band_miss_vector: tuple[int, int, int],
    witness_floor: float,
    queued_residual_slot_still_live: bool,
) -> Phase7SameSeedRuntimeBridgeAfterReportScenarioStep:
    return Phase7SameSeedRuntimeBridgeAfterReportScenarioStep(
        scenario_name=scenario_name,
        after_report_supplied=bool(after_report_supplied),
        before_after_driver_signature=str(before_after_driver_signature),
        runtime_bridge_driver_signature=str(runtime_bridge_driver_signature),
        runtime_bridge_state=str(runtime_bridge_state),
        intake_disposition=str(intake_disposition),
        point_miss_vector=tuple(int(value) for value in point_miss_vector),
        band_miss_vector=tuple(int(value) for value in band_miss_vector),
        witness_floor=float(witness_floor),
        queued_residual_slot_still_live=bool(queued_residual_slot_still_live),
    )


_PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_CANONICAL_SEED_ORDER = (
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
)
_PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_RUNTIME_WITNESS_PATH = (
    "omega_f_hat[2,2]",
    "v_f_hat[2,2]",
    "covariance(0.25, 0.15)",
)
_PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_ACCEPTANCE_READOUT_PATH = (
    "omega_f_hat[2,2]",
    "v_f_hat[2,1]",
    "bar_f_at_z0[1]",
)
_PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_BINDING_SLOT = (303, 1)
_PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_RESIDUAL_SLOT = (707, 2)
_PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_SCENARIO_EXPECTATIONS = {
    "baseline_open": {
        "after_report_supplied": False,
        "point_miss_vector": (1, 3, 2),
        "band_miss_vector": (0, 1, 1),
        "before_after_driver_signature": (
            "same-seed-before-after-acceptance-not-yet-satisfied"
        ),
        "runtime_bridge_driver_signature": (
            "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-admission-order-open"
        ),
        "runtime_bridge_state": (
            "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open"
        ),
        "intake_disposition": "baseline-no-after-report",
        "witness_floor": 7.0 / 9.0,
        "queued_residual_slot_still_live": True,
    },
    "binding_only": {
        "after_report_supplied": True,
        "point_miss_vector": (1, 2, 2),
        "band_miss_vector": (0, 1, 1),
        "before_after_driver_signature": (
            "same-seed-before-after-acceptance-not-yet-satisfied"
        ),
        "runtime_bridge_driver_signature": (
            "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-admission-order-residual-only"
        ),
        "runtime_bridge_state": (
            "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-open"
        ),
        "intake_disposition": "binding-only-payload-rejected",
        "witness_floor": 8.0 / 9.0,
        "queued_residual_slot_still_live": True,
    },
    "completion": {
        "after_report_supplied": True,
        "point_miss_vector": (1, 2, 1),
        "band_miss_vector": (0, 1, 1),
        "before_after_driver_signature": "same-seed-before-after-acceptance-satisfied",
        "runtime_bridge_driver_signature": (
            "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-admission-order-closed"
        ),
        "runtime_bridge_state": (
            "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-runtime-bridge-closed"
        ),
        "intake_disposition": "completion-after-report-accepted",
        "witness_floor": 8.0 / 9.0,
        "queued_residual_slot_still_live": False,
    },
}


def _phase7_same_seed_runtime_bridge_after_report_scenario_report(
    *,
    report: Any,
    random_state: int,
    z_index: int,
) -> Any:
    import importlib

    seedwise_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition"
    )
    updated_seed_observations = tuple(
        replace(
            observation,
            pointwise_coverage_by_z=tuple(
                True if index == int(z_index) else covered
                for index, covered in enumerate(observation.pointwise_coverage_by_z)
            ),
        )
        if observation.random_state == int(random_state)
        else observation
        for observation in report.seed_observations
    )
    return seedwise_module.build_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition_report(
        seed_observations=updated_seed_observations
    )


def _phase7_same_seed_runtime_bridge_after_report_scenario_vectors(
    report: Any,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    group_summary = report.group_summary("all")
    return (
        tuple(int(value) for value in group_summary.point_miss_count_by_z),
        tuple(int(value) for value in group_summary.uniform_band_miss_count_by_z),
    )


def _require_phase7_same_seed_runtime_bridge_after_report_scenario(
    *,
    scenario_name: str,
    report: Any,
) -> None:
    expected = _PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_SCENARIO_EXPECTATIONS[
        scenario_name
    ]
    point_miss_vector, band_miss_vector = (
        _phase7_same_seed_runtime_bridge_after_report_scenario_vectors(report)
    )
    if point_miss_vector != expected["point_miss_vector"]:
        raise ValueError(
            "same-seed runtime bridge after-report scenario probe expected "
            f"{scenario_name} point miss {expected['point_miss_vector']!r} but saw "
            f"{point_miss_vector!r}"
        )
    if band_miss_vector != expected["band_miss_vector"]:
        raise ValueError(
            "same-seed runtime bridge after-report scenario probe expected "
            f"{scenario_name} band miss {expected['band_miss_vector']!r} but saw "
            f"{band_miss_vector!r}"
        )


@lru_cache(maxsize=1)
def run_phase7_same_seed_runtime_bridge_after_report_scenario_probe() -> (
    Phase7SameSeedRuntimeBridgeAfterReportScenarioProbe
):
    import importlib

    seedwise_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition"
    )

    baseline_report = seedwise_module.run_phase7_monte_carlo_widening_policy_near_zero_grid_seedwise_pointwise_coverage_decomposition()
    canonical_seed_order = tuple(
        int(observation.random_state)
        for observation in baseline_report.seed_observations
    )
    if (
        canonical_seed_order
        != _PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_CANONICAL_SEED_ORDER
    ):
        raise ValueError(
            "same-seed runtime bridge after-report scenario probe requires the canonical seed order "
            f"{_PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_CANONICAL_SEED_ORDER!r}"
        )

    binding_only_report = _phase7_same_seed_runtime_bridge_after_report_scenario_report(
        report=baseline_report,
        random_state=_PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_BINDING_SLOT[0],
        z_index=_PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_BINDING_SLOT[1],
    )
    completion_report = _phase7_same_seed_runtime_bridge_after_report_scenario_report(
        report=binding_only_report,
        random_state=_PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_RESIDUAL_SLOT[0],
        z_index=_PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_RESIDUAL_SLOT[1],
    )

    _require_phase7_same_seed_runtime_bridge_after_report_scenario(
        scenario_name="baseline_open",
        report=baseline_report,
    )
    _require_phase7_same_seed_runtime_bridge_after_report_scenario(
        scenario_name="binding_only",
        report=binding_only_report,
    )
    _require_phase7_same_seed_runtime_bridge_after_report_scenario(
        scenario_name="completion",
        report=completion_report,
    )

    baseline_expectation = (
        _PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_SCENARIO_EXPECTATIONS[
            "baseline_open"
        ]
    )
    binding_expectation = (
        _PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_SCENARIO_EXPECTATIONS[
            "binding_only"
        ]
    )
    completion_expectation = (
        _PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_SCENARIO_EXPECTATIONS[
            "completion"
        ]
    )

    baseline_point_miss_vector, baseline_band_miss_vector = (
        _phase7_same_seed_runtime_bridge_after_report_scenario_vectors(baseline_report)
    )
    binding_point_miss_vector, binding_band_miss_vector = (
        _phase7_same_seed_runtime_bridge_after_report_scenario_vectors(
            binding_only_report
        )
    )
    completion_point_miss_vector, completion_band_miss_vector = (
        _phase7_same_seed_runtime_bridge_after_report_scenario_vectors(
            completion_report
        )
    )

    if baseline_band_miss_vector != binding_band_miss_vector:
        raise ValueError(
            "same-seed runtime bridge after-report scenario probe requires binding-only payloads to preserve the baseline band miss surface"
        )
    if binding_band_miss_vector != completion_band_miss_vector:
        raise ValueError(
            "same-seed runtime bridge after-report scenario probe requires completion payloads to preserve the binding-only band miss surface"
        )
    if baseline_point_miss_vector[0] != completion_point_miss_vector[0]:
        raise ValueError(
            "same-seed runtime bridge after-report scenario probe requires the left guard point miss to remain unchanged across the scenario ladder"
        )

    return Phase7SameSeedRuntimeBridgeAfterReportScenarioProbe(
        status="same-seed-runtime-bridge-after-report-scenario-probe-satisfied",
        canonical_seed_order=canonical_seed_order,
        current_frontier_driver_signature=(
            "same-seed-exact-witness-observed-rerun-runtime-bridge-after-report-frontier-packet-open"
        ),
        runtime_witness_path=(
            _PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_RUNTIME_WITNESS_PATH
        ),
        acceptance_readout_path=(
            _PHASE7_SAME_SEED_RUNTIME_BRIDGE_AFTER_REPORT_ACCEPTANCE_READOUT_PATH
        ),
        scenarios=(
            _phase7_same_seed_runtime_bridge_after_report_scenario_step(
                scenario_name="baseline_open",
                after_report_supplied=baseline_expectation["after_report_supplied"],
                before_after_driver_signature=baseline_expectation[
                    "before_after_driver_signature"
                ],
                runtime_bridge_driver_signature=baseline_expectation[
                    "runtime_bridge_driver_signature"
                ],
                runtime_bridge_state=baseline_expectation["runtime_bridge_state"],
                intake_disposition=baseline_expectation["intake_disposition"],
                point_miss_vector=baseline_point_miss_vector,
                band_miss_vector=baseline_band_miss_vector,
                witness_floor=float(baseline_expectation["witness_floor"]),
                queued_residual_slot_still_live=baseline_expectation[
                    "queued_residual_slot_still_live"
                ],
            ),
            _phase7_same_seed_runtime_bridge_after_report_scenario_step(
                scenario_name="binding_only",
                after_report_supplied=binding_expectation["after_report_supplied"],
                before_after_driver_signature=binding_expectation[
                    "before_after_driver_signature"
                ],
                runtime_bridge_driver_signature=binding_expectation[
                    "runtime_bridge_driver_signature"
                ],
                runtime_bridge_state=binding_expectation["runtime_bridge_state"],
                intake_disposition=binding_expectation["intake_disposition"],
                point_miss_vector=binding_point_miss_vector,
                band_miss_vector=binding_band_miss_vector,
                witness_floor=float(binding_expectation["witness_floor"]),
                queued_residual_slot_still_live=binding_expectation[
                    "queued_residual_slot_still_live"
                ],
            ),
            _phase7_same_seed_runtime_bridge_after_report_scenario_step(
                scenario_name="completion",
                after_report_supplied=completion_expectation["after_report_supplied"],
                before_after_driver_signature=completion_expectation[
                    "before_after_driver_signature"
                ],
                runtime_bridge_driver_signature=completion_expectation[
                    "runtime_bridge_driver_signature"
                ],
                runtime_bridge_state=completion_expectation["runtime_bridge_state"],
                intake_disposition=completion_expectation["intake_disposition"],
                point_miss_vector=completion_point_miss_vector,
                band_miss_vector=completion_band_miss_vector,
                witness_floor=float(completion_expectation["witness_floor"]),
                queued_residual_slot_still_live=completion_expectation[
                    "queued_residual_slot_still_live"
                ],
            ),
        ),
    )


def _phase7_same_seed_first_hop_landing_guard_probe_slot_from_report(
    report: Any,
) -> Phase7SameSeedFirstHopRuntimeBindingPacketProbeSlot | None:
    if report.next_required_slot_random_state is None:
        return None
    return _phase7_same_seed_first_hop_runtime_binding_packet_probe_slot(
        random_state=report.next_required_slot_random_state,
        seed_group=report.next_required_slot_seed_group,
        z_index=report.next_required_slot_z_index,
        z_value=report.next_required_slot_z_value,
        repair_stage=report.next_required_slot_repair_stage,
        repair_role=report.next_required_slot_repair_role,
        repair_priority=report.next_required_slot_repair_priority,
    )


def _phase7_same_seed_first_hop_landing_guard_probe_step(
    *,
    scenario_name: str,
    report: Any,
    scenario_step: Any,
) -> Phase7SameSeedFirstHopLandingGuardProbeStep:
    point_miss_vector = tuple(int(value) for value in report.current_point_miss_vector)
    expected_point_miss_vector = tuple(
        int(value) for value in scenario_step.point_miss_vector
    )
    if point_miss_vector != expected_point_miss_vector:
        raise ValueError(
            "same-seed first-hop landing guard probe requires the "
            f"{scenario_name} landing guard point miss to match the after-report scenario"
        )

    band_miss_vector = tuple(int(value) for value in report.current_band_miss_vector)
    expected_band_miss_vector = tuple(
        int(value) for value in scenario_step.band_miss_vector
    )
    if band_miss_vector != expected_band_miss_vector:
        raise ValueError(
            "same-seed first-hop landing guard probe requires the "
            f"{scenario_name} landing guard band miss to match the after-report scenario"
        )

    witness_floor = float(report.current_witness_floor)
    if abs(witness_floor - float(scenario_step.witness_floor)) > 1e-12:
        raise ValueError(
            "same-seed first-hop landing guard probe requires the "
            f"{scenario_name} landing guard witness floor to match the after-report scenario"
        )

    queued_residual_slot_still_live = bool(
        report.driver_signature
        != "same-seed-exact-witness-observed-rerun-first-hop-landing-guard-closed"
    )
    if queued_residual_slot_still_live != bool(
        scenario_step.queued_residual_slot_still_live
    ):
        raise ValueError(
            "same-seed first-hop landing guard probe requires the "
            f"{scenario_name} queued residual state to match the after-report scenario"
        )

    return Phase7SameSeedFirstHopLandingGuardProbeStep(
        scenario_name=scenario_name,
        landing_guard_driver_signature=report.driver_signature,
        binding_packet_driver_signature=report.runtime_binding_packet_driver_signature,
        repair_agenda_driver_signature=report.repair_agenda_driver_signature,
        current_rung_status=report.current_rung_status,
        point_miss_vector=point_miss_vector,
        band_miss_vector=band_miss_vector,
        witness_floor=witness_floor,
        first_hop_landed=report.first_hop_landed,
        queued_residual_slot_still_live=queued_residual_slot_still_live,
        next_required_slot=_phase7_same_seed_first_hop_landing_guard_probe_slot_from_report(
            report
        ),
        canonical_landing_guard_digest=(
            report.canonical_observed_rerun_first_hop_landing_guard_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_landing_guard_probe() -> (
    Phase7SameSeedFirstHopLandingGuardProbe
):
    import importlib

    landing_guard_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard"
    )
    binding_slot_progress_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile"
    )
    residual_slot_completion_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile"
    )

    runtime_binding_packet_probe = (
        run_phase7_same_seed_first_hop_runtime_binding_packet_probe()
    )
    scenario_probe = run_phase7_same_seed_runtime_bridge_after_report_scenario_probe()

    baseline_report = landing_guard_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard()
    binding_only_after_report = binding_slot_progress_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile().binding_only_after_report
    binding_only_report = landing_guard_module.build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard_report(
        after_report=binding_only_after_report
    )
    completion_after_report = residual_slot_completion_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile().completion_after_report
    completion_report = landing_guard_module.build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_landing_guard_report(
        after_report=completion_after_report
    )

    canonical_seed_order = tuple(
        int(value) for value in baseline_report.same_seed_random_states
    )
    if canonical_seed_order != runtime_binding_packet_probe.canonical_seed_order:
        raise ValueError(
            "same-seed first-hop landing guard probe requires the runtime binding packet probe to keep the canonical seed order"
        )
    if canonical_seed_order != scenario_probe.canonical_seed_order:
        raise ValueError(
            "same-seed first-hop landing guard probe requires the after-report scenario probe to keep the canonical seed order"
        )

    runtime_witness_path = tuple(
        str(item) for item in baseline_report.runtime_witness_path
    )
    if runtime_witness_path != tuple(runtime_binding_packet_probe.runtime_witness_path):
        raise ValueError(
            "same-seed first-hop landing guard probe requires the runtime binding packet probe to share the runtime witness path"
        )
    if runtime_witness_path != tuple(scenario_probe.runtime_witness_path):
        raise ValueError(
            "same-seed first-hop landing guard probe requires the after-report scenario probe to share the runtime witness path"
        )

    baseline_next_required_slot = (
        _phase7_same_seed_first_hop_landing_guard_probe_slot_from_report(
            baseline_report
        )
    )
    if (
        runtime_binding_packet_probe.next_required_slot is None
        or baseline_next_required_slot is None
        or baseline_next_required_slot.to_dict()
        != runtime_binding_packet_probe.next_required_slot.to_dict()
    ):
        raise ValueError(
            "same-seed first-hop landing guard probe requires the open landing guard slot to stay aligned with the runtime binding packet probe"
        )

    if (
        baseline_report.runtime_binding_packet_driver_signature
        != runtime_binding_packet_probe.binding_packet_driver_signature
    ):
        raise ValueError(
            "same-seed first-hop landing guard probe requires the open landing guard state to share the runtime binding packet driver"
        )
    if (
        baseline_report.repair_agenda_driver_signature
        != runtime_binding_packet_probe.repair_agenda_driver_signature
    ):
        raise ValueError(
            "same-seed first-hop landing guard probe requires the open landing guard state to share the repair agenda driver"
        )
    if int(baseline_report.binding_slot_replication_seed) != int(
        runtime_binding_packet_probe.binding_runtime_packet.replication_seed
    ):
        raise ValueError(
            "same-seed first-hop landing guard probe requires the binding runtime packet replication seed to stay aligned"
        )

    scenario_step_by_name = {
        str(step.scenario_name): step for step in scenario_probe.scenarios
    }
    for scenario_name in ("baseline_open", "binding_only", "completion"):
        if scenario_name not in scenario_step_by_name:
            raise ValueError(
                "same-seed first-hop landing guard probe requires the after-report scenario probe to expose "
                f"{scenario_name}"
            )

    scenarios = (
        _phase7_same_seed_first_hop_landing_guard_probe_step(
            scenario_name="baseline_open",
            report=baseline_report,
            scenario_step=scenario_step_by_name["baseline_open"],
        ),
        _phase7_same_seed_first_hop_landing_guard_probe_step(
            scenario_name="binding_only",
            report=binding_only_report,
            scenario_step=scenario_step_by_name["binding_only"],
        ),
        _phase7_same_seed_first_hop_landing_guard_probe_step(
            scenario_name="completion",
            report=completion_report,
            scenario_step=scenario_step_by_name["completion"],
        ),
    )

    return Phase7SameSeedFirstHopLandingGuardProbe(
        status="same-seed-first-hop-landing-guard-probe-satisfied",
        canonical_seed_order=canonical_seed_order,
        admission_order_driver_signature=(
            runtime_binding_packet_probe.admission_order_driver_signature
        ),
        current_frontier_driver_signature=(
            scenario_probe.current_frontier_driver_signature
        ),
        runtime_witness_path=runtime_witness_path,
        runtime_object_flow_focus=(
            runtime_binding_packet_probe.runtime_object_flow_focus
        ),
        acceptance_readout_path=tuple(scenario_probe.acceptance_readout_path),
        binding_runtime_packet=runtime_binding_packet_probe.binding_runtime_packet,
        scenarios=scenarios,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_witness_floor_quota_probe() -> Any:
    import importlib

    quota_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota"
    )

    return quota_module.run_phase7_same_seed_first_hop_witness_floor_quota_probe()


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_execution_queue_probe() -> (
    Phase7SameSeedFirstHopExecutionQueueProbe
):
    landing_guard_probe = run_phase7_same_seed_first_hop_landing_guard_probe()
    completion_ladder_probe = run_phase7_same_seed_completion_ladder_probe()
    repair_agenda_probe = run_phase7_same_seed_repair_agenda_probe()
    runtime_binding_packet_probe = (
        run_phase7_same_seed_first_hop_runtime_binding_packet_probe()
    )

    canonical_seed_order = landing_guard_probe.canonical_seed_order
    if completion_ladder_probe.canonical_seed_order != canonical_seed_order:
        raise ValueError(
            "same-seed first-hop execution queue probe requires the completion ladder probe to keep the canonical seed order"
        )
    if repair_agenda_probe.canonical_seed_order != canonical_seed_order:
        raise ValueError(
            "same-seed first-hop execution queue probe requires the repair agenda probe to keep the canonical seed order"
        )
    if runtime_binding_packet_probe.canonical_seed_order != canonical_seed_order:
        raise ValueError(
            "same-seed first-hop execution queue probe requires the runtime binding packet probe to keep the canonical seed order"
        )
    if (
        completion_ladder_probe.admission_order_driver_signature
        != landing_guard_probe.admission_order_driver_signature
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the completion ladder probe to share the admission-order signature"
        )
    if (
        repair_agenda_probe.admission_order_driver_signature
        != landing_guard_probe.admission_order_driver_signature
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the repair agenda probe to share the admission-order signature"
        )
    if (
        runtime_binding_packet_probe.admission_order_driver_signature
        != landing_guard_probe.admission_order_driver_signature
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the runtime binding packet probe to share the admission-order signature"
        )
    if tuple(completion_ladder_probe.runtime_witness_path) != tuple(
        landing_guard_probe.runtime_witness_path
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the completion ladder probe to share the runtime witness path"
        )
    if tuple(repair_agenda_probe.runtime_witness_path) != tuple(
        landing_guard_probe.runtime_witness_path
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the repair agenda probe to share the runtime witness path"
        )
    if tuple(runtime_binding_packet_probe.runtime_witness_path) != tuple(
        landing_guard_probe.runtime_witness_path
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the runtime binding packet probe to share the runtime witness path"
        )

    landing_guard_scenarios = {
        step.scenario_name: step for step in landing_guard_probe.scenarios
    }
    completion_ladder_scenarios = {
        step.scenario_name: step for step in completion_ladder_probe.scenarios
    }
    for scenario_name in ("baseline_open", "binding_only", "completion"):
        if scenario_name not in landing_guard_scenarios:
            raise ValueError(
                "same-seed first-hop execution queue probe requires the landing guard probe to expose the full ladder"
            )
    for scenario_name in (
        "current_open",
        "binding_slot_landed",
        "completion_admissible",
    ):
        if scenario_name not in completion_ladder_scenarios:
            raise ValueError(
                "same-seed first-hop execution queue probe requires the completion ladder probe to expose the full ladder"
            )

    baseline_open = landing_guard_scenarios["baseline_open"]
    binding_only = landing_guard_scenarios["binding_only"]
    completion = landing_guard_scenarios["completion"]
    current_open = completion_ladder_scenarios["current_open"]
    binding_slot_landed = completion_ladder_scenarios["binding_slot_landed"]
    completion_admissible = completion_ladder_scenarios["completion_admissible"]

    current_actionable_slot = runtime_binding_packet_probe.next_required_slot
    if current_actionable_slot is None:
        raise ValueError(
            "same-seed first-hop execution queue probe requires one actionable current slot"
        )
    if (
        repair_agenda_probe.next_required_slot is None
        or not _phase7_same_seed_slot_matches_runtime_binding_packet_slot(
            repair_agenda_probe.next_required_slot,
            binding_slot=current_actionable_slot,
        )
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the current actionable packet to match the repair agenda"
        )
    if (
        baseline_open.next_required_slot is None
        or baseline_open.next_required_slot.to_dict()
        != current_actionable_slot.to_dict()
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the landing guard baseline to keep the current actionable packet"
        )
    if (
        current_open.next_repair_slot is None
        or not _phase7_same_seed_completion_slot_matches_runtime_binding_slot(
            current_open.next_repair_slot,
            current_actionable_slot,
        )
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the current execution queue slot to match the actionable packet"
        )
    if (
        runtime_binding_packet_probe.binding_runtime_packet.random_state
        != current_actionable_slot.random_state
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the actionable packet to stay on the current runtime-binding random state"
        )
    if (
        runtime_binding_packet_probe.binding_runtime_packet.z_index
        != current_actionable_slot.z_index
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the actionable packet to stay on the current runtime-binding z index"
        )

    queued_residual_slot = binding_only.next_required_slot
    if queued_residual_slot is None:
        raise ValueError(
            "same-seed first-hop execution queue probe requires one queued residual cleanup slot after the binding landing"
        )
    if (
        binding_slot_landed.next_repair_slot is None
        or not _phase7_same_seed_completion_slot_matches_runtime_binding_slot(
            binding_slot_landed.next_repair_slot,
            queued_residual_slot,
        )
    ):
        raise ValueError(
            "same-seed first-hop execution queue probe requires the landed execution queue slot to match the queued residual slot"
        )
    if completion_admissible.next_repair_slot is not None:
        raise ValueError(
            "same-seed first-hop execution queue probe requires the admissible completion rung to clear the queue"
        )
    if completion.next_required_slot is not None:
        raise ValueError(
            "same-seed first-hop execution queue probe requires the closed landing guard rung to clear the queue"
        )

    scenario_pairs = (
        (
            "current_open",
            baseline_open,
            current_open,
            current_actionable_slot,
            repair_agenda_probe.current_rung_status,
        ),
        (
            "binding_slot_landed",
            binding_only,
            binding_slot_landed,
            queued_residual_slot,
            binding_only.current_rung_status,
        ),
        (
            "completion_admissible",
            completion,
            completion_admissible,
            None,
            completion.current_rung_status,
        ),
    )
    scenarios: list[Phase7SameSeedFirstHopExecutionQueueProbeStep] = []
    for (
        scenario_name,
        landing_guard_step,
        completion_step,
        selected_slot,
        current_rung_status,
    ) in scenario_pairs:
        if tuple(landing_guard_step.point_miss_vector) != tuple(
            completion_step.point_miss_vector
        ):
            raise ValueError(
                "same-seed first-hop execution queue probe requires the landing guard and completion ladder point miss vectors to stay aligned"
            )
        if tuple(landing_guard_step.band_miss_vector) != tuple(
            completion_step.band_miss_vector
        ):
            raise ValueError(
                "same-seed first-hop execution queue probe requires the landing guard and completion ladder band miss vectors to stay aligned"
            )
        if (
            abs(
                float(landing_guard_step.witness_floor)
                - float(completion_step.witness_floor)
            )
            > 1e-12
        ):
            raise ValueError(
                "same-seed first-hop execution queue probe requires the landing guard and completion ladder witness floors to stay aligned"
            )
        scenarios.append(
            _phase7_same_seed_first_hop_execution_queue_step(
                scenario_name=scenario_name,
                landing_guard_driver_signature=(
                    landing_guard_step.landing_guard_driver_signature
                ),
                completion_ladder_driver_signature=completion_step.driver_signature,
                candidate_status=completion_step.candidate_status,
                resulting_evidence_status=completion_step.resulting_evidence_status,
                current_rung_status=current_rung_status,
                point_miss_vector=landing_guard_step.point_miss_vector,
                band_miss_vector=landing_guard_step.band_miss_vector,
                witness_floor=landing_guard_step.witness_floor,
                selected_slot=selected_slot,
                queued_residual_slot_still_live=(
                    landing_guard_step.queued_residual_slot_still_live
                ),
            )
        )

    return Phase7SameSeedFirstHopExecutionQueueProbe(
        status="same-seed-first-hop-execution-queue-probe-satisfied",
        canonical_seed_order=canonical_seed_order,
        admission_order_driver_signature=(
            landing_guard_probe.admission_order_driver_signature
        ),
        current_frontier_driver_signature=(
            landing_guard_probe.current_frontier_driver_signature
        ),
        runtime_witness_path=landing_guard_probe.runtime_witness_path,
        runtime_object_flow_focus=landing_guard_probe.runtime_object_flow_focus,
        acceptance_readout_path=landing_guard_probe.acceptance_readout_path,
        current_rung_status=repair_agenda_probe.current_rung_status,
        highest_landed_rung=repair_agenda_probe.highest_landed_rung,
        downstream_evidence_status=repair_agenda_probe.downstream_evidence_status,
        current_to_next_rung_pointwise_gap_vector=(
            repair_agenda_probe.current_to_next_rung_pointwise_gap_vector
        ),
        current_to_next_rung_band_gap_vector=(
            repair_agenda_probe.current_to_next_rung_band_gap_vector
        ),
        current_to_completion_pointwise_gap_vector=(
            repair_agenda_probe.current_to_completion_pointwise_gap_vector
        ),
        current_to_completion_band_gap_vector=(
            repair_agenda_probe.current_to_completion_band_gap_vector
        ),
        current_actionable_packet=runtime_binding_packet_probe.binding_runtime_packet,
        queued_residual_slot=queued_residual_slot,
        scenarios=tuple(scenarios),
    )


def _phase7_same_seed_observed_rerun_rung_guard_probe_slot(
    slot: Any,
) -> Phase7SameSeedCompletionLadderRepairSlot | None:
    if slot is None:
        return None
    if hasattr(slot, "to_dict") and not hasattr(slot, "random_state"):
        slot_dict = slot.to_dict()
        return Phase7SameSeedCompletionLadderRepairSlot(
            random_state=slot_dict["random_state"],
            seed_group=slot_dict["seed_group"],
            z_index=slot_dict["z_index"],
            z_value=slot_dict["z_value"],
        )
    return Phase7SameSeedCompletionLadderRepairSlot(
        random_state=slot.random_state,
        seed_group=slot.seed_group,
        z_index=slot.z_index,
        z_value=slot.z_value,
    )


def _phase7_same_seed_observed_rerun_rung_guard_probe_slot_dict(
    slot: Any,
) -> dict[str, object] | None:
    probe_slot = _phase7_same_seed_observed_rerun_rung_guard_probe_slot(slot)
    return None if probe_slot is None else probe_slot.to_dict()


def _phase7_same_seed_observed_rerun_rung_guard_probe_step(
    *,
    scenario_name: str,
    report: Any,
    landing_guard_step: Phase7SameSeedFirstHopLandingGuardProbeStep,
    completion_ladder_step: Phase7SameSeedCompletionLadderStep,
    scenario_step: Phase7SameSeedRuntimeBridgeAfterReportScenarioStep,
    report_completion_ladder_driver_signature: str | None,
) -> Phase7SameSeedObservedRerunRungGuardProbeStep:
    point_miss_vector = tuple(
        int(value) for value in report.candidate_point_miss_vector
    )
    if point_miss_vector != tuple(
        int(value) for value in landing_guard_step.point_miss_vector
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} point miss to match the landing guard probe"
        )
    if point_miss_vector != tuple(
        int(value) for value in completion_ladder_step.point_miss_vector
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} point miss to match the completion ladder probe"
        )
    if point_miss_vector != tuple(
        int(value) for value in scenario_step.point_miss_vector
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} point miss to match the after-report scenario probe"
        )

    band_miss_vector = tuple(int(value) for value in report.candidate_band_miss_vector)
    if band_miss_vector != tuple(
        int(value) for value in landing_guard_step.band_miss_vector
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} band miss to match the landing guard probe"
        )
    if band_miss_vector != tuple(
        int(value) for value in completion_ladder_step.band_miss_vector
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} band miss to match the completion ladder probe"
        )
    if band_miss_vector != tuple(
        int(value) for value in scenario_step.band_miss_vector
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} band miss to match the after-report scenario probe"
        )

    witness_floor = float(landing_guard_step.witness_floor)
    if abs(witness_floor - float(completion_ladder_step.witness_floor)) > 1e-12:
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} witness floor to match the completion ladder probe"
        )
    if abs(witness_floor - float(scenario_step.witness_floor)) > 1e-12:
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} witness floor to match the after-report scenario probe"
        )

    before_after_driver_signature = str(report.before_after_driver_signature)
    if before_after_driver_signature != str(
        completion_ladder_step.before_after_driver_signature
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} before/after driver to match the completion ladder probe"
        )
    if before_after_driver_signature != str(
        scenario_step.before_after_driver_signature
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} before/after driver to match the after-report scenario probe"
        )

    next_required_slot = _phase7_same_seed_observed_rerun_rung_guard_probe_slot(
        report.next_required_slot
    )
    if _phase7_same_seed_observed_rerun_rung_guard_probe_slot_dict(
        report.next_required_slot
    ) != _phase7_same_seed_observed_rerun_rung_guard_probe_slot_dict(
        landing_guard_step.next_required_slot
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} next required slot to match the landing guard probe"
        )
    if _phase7_same_seed_observed_rerun_rung_guard_probe_slot_dict(
        report.next_required_slot
    ) != _phase7_same_seed_observed_rerun_rung_guard_probe_slot_dict(
        completion_ladder_step.next_repair_slot
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} next required slot to match the completion ladder probe"
        )

    queued_residual_slot_still_live = next_required_slot is not None
    if queued_residual_slot_still_live != bool(
        landing_guard_step.queued_residual_slot_still_live
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} queued residual state to match the landing guard probe"
        )
    if queued_residual_slot_still_live != bool(
        completion_ladder_step.queued_residual_slot_still_live
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} queued residual state to match the completion ladder probe"
        )
    if queued_residual_slot_still_live != bool(
        scenario_step.queued_residual_slot_still_live
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} queued residual state to match the after-report scenario probe"
        )

    if str(report.resulting_rung_status) != str(landing_guard_step.current_rung_status):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} rung status to match the landing guard probe"
        )
    if report_completion_ladder_driver_signature is not None and str(
        report_completion_ladder_driver_signature
    ) != str(completion_ladder_step.driver_signature):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} completion ladder driver to match the source rung guard"
        )
    if str(report.downstream_evidence_status) != str(
        completion_ladder_step.resulting_evidence_status
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the "
            f"{scenario_name} completion ladder evidence status to match the source rung guard"
        )

    return Phase7SameSeedObservedRerunRungGuardProbeStep(
        scenario_name=scenario_name,
        binding_slot_candidate_status=report.binding_slot_candidate_status,
        residual_slot_candidate_status=report.residual_slot_candidate_status,
        current_rung_candidate_status=report.current_rung_candidate_status,
        resulting_rung_status=report.resulting_rung_status,
        downstream_evidence_status=report.downstream_evidence_status,
        highest_landed_rung=report.highest_landed_rung,
        landing_guard_driver_signature=(
            landing_guard_step.landing_guard_driver_signature
        ),
        completion_ladder_driver_signature=str(
            report_completion_ladder_driver_signature or report.resulting_rung_status
        ),
        runtime_bridge_driver_signature=scenario_step.runtime_bridge_driver_signature,
        before_after_driver_signature=before_after_driver_signature,
        acceptance_exact_trim_floor_landing_bridge_status=(
            report.acceptance_exact_trim_floor_landing_bridge_status
        ),
        acceptance_exact_trim_floor_inverse_pi_concentration_status=(
            report.acceptance_exact_trim_floor_inverse_pi_concentration_status
        ),
        point_miss_vector=point_miss_vector,
        band_miss_vector=band_miss_vector,
        admissibility_witness_floor=float(report.acceptance_candidate_witness_floor),
        witness_floor=witness_floor,
        next_required_slot=next_required_slot,
        queued_residual_slot_still_live=queued_residual_slot_still_live,
        canonical_observed_rerun_rung_digest=(
            report.canonical_observed_rerun_rung_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_observed_rerun_rung_guard_probe() -> (
    Phase7SameSeedObservedRerunRungGuardProbe
):
    import importlib

    rung_guard_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard"
    )
    binding_slot_progress_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile"
    )
    residual_slot_completion_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile"
    )

    landing_guard_probe = run_phase7_same_seed_first_hop_landing_guard_probe()
    completion_ladder_probe = run_phase7_same_seed_completion_ladder_probe()
    scenario_probe = run_phase7_same_seed_runtime_bridge_after_report_scenario_probe()

    baseline_report = rung_guard_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard()
    binding_only_after_report = binding_slot_progress_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile().binding_only_after_report
    binding_only_report = rung_guard_module.build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard_report(
        after_report=binding_only_after_report
    )
    completion_after_report = residual_slot_completion_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile().completion_after_report
    completion_report = rung_guard_module.build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_rung_guard_report(
        after_report=completion_after_report
    )

    canonical_seed_order = tuple(
        int(value) for value in baseline_report.same_seed_random_states
    )
    if canonical_seed_order != landing_guard_probe.canonical_seed_order:
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the landing guard probe to keep the canonical seed order"
        )
    if canonical_seed_order != completion_ladder_probe.canonical_seed_order:
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the completion ladder probe to keep the canonical seed order"
        )
    if canonical_seed_order != scenario_probe.canonical_seed_order:
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the after-report scenario probe to keep the canonical seed order"
        )

    runtime_witness_path = tuple(
        str(item) for item in baseline_report.runtime_witness_path
    )
    if runtime_witness_path != tuple(landing_guard_probe.runtime_witness_path):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the landing guard probe to share the runtime witness path"
        )
    if runtime_witness_path != tuple(completion_ladder_probe.runtime_witness_path):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the completion ladder probe to share the runtime witness path"
        )
    if runtime_witness_path != tuple(scenario_probe.runtime_witness_path):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the after-report scenario probe to share the runtime witness path"
        )

    acceptance_readout_path = tuple(
        str(item) for item in baseline_report.acceptance_readout_path
    )
    if acceptance_readout_path != tuple(landing_guard_probe.acceptance_readout_path):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the landing guard probe to share the acceptance readout path"
        )
    if acceptance_readout_path != tuple(scenario_probe.acceptance_readout_path):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the after-report scenario probe to share the acceptance readout path"
        )

    binding_repair_slot = _phase7_same_seed_observed_rerun_rung_guard_probe_slot(
        baseline_report.binding_repair_slot
    )
    residual_repair_slot = _phase7_same_seed_observed_rerun_rung_guard_probe_slot(
        baseline_report.residual_repair_slot
    )
    if (
        binding_repair_slot is None
        or binding_repair_slot.to_dict()
        != completion_ladder_probe.binding_repair_slot.to_dict()
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the binding repair slot to match the completion ladder probe"
        )
    if (
        residual_repair_slot is None
        or residual_repair_slot.to_dict()
        != completion_ladder_probe.residual_repair_slot.to_dict()
    ):
        raise ValueError(
            "same-seed observed rerun rung guard probe requires the residual repair slot to match the completion ladder probe"
        )

    landing_guard_step_by_name = {
        str(step.scenario_name): step for step in landing_guard_probe.scenarios
    }
    scenario_step_by_name = {
        str(step.scenario_name): step for step in scenario_probe.scenarios
    }
    completion_ladder_step_by_name = {
        "baseline_open": completion_ladder_probe.scenarios[0],
        "binding_only": completion_ladder_probe.scenarios[1],
        "completion": completion_ladder_probe.scenarios[2],
    }

    for scenario_name in ("baseline_open", "binding_only", "completion"):
        if scenario_name not in landing_guard_step_by_name:
            raise ValueError(
                "same-seed observed rerun rung guard probe requires the landing guard probe to expose "
                f"{scenario_name}"
            )
        if scenario_name not in scenario_step_by_name:
            raise ValueError(
                "same-seed observed rerun rung guard probe requires the after-report scenario probe to expose "
                f"{scenario_name}"
            )

    scenarios = (
        _phase7_same_seed_observed_rerun_rung_guard_probe_step(
            scenario_name="baseline_open",
            report=baseline_report,
            landing_guard_step=landing_guard_step_by_name["baseline_open"],
            completion_ladder_step=completion_ladder_step_by_name["baseline_open"],
            scenario_step=scenario_step_by_name["baseline_open"],
            report_completion_ladder_driver_signature=(
                baseline_report.binding_slot_progress_status
            ),
        ),
        _phase7_same_seed_observed_rerun_rung_guard_probe_step(
            scenario_name="binding_only",
            report=binding_only_report,
            landing_guard_step=landing_guard_step_by_name["binding_only"],
            completion_ladder_step=completion_ladder_step_by_name["binding_only"],
            scenario_step=scenario_step_by_name["binding_only"],
            report_completion_ladder_driver_signature=(
                binding_only_report.binding_slot_progress_status
            ),
        ),
        _phase7_same_seed_observed_rerun_rung_guard_probe_step(
            scenario_name="completion",
            report=completion_report,
            landing_guard_step=landing_guard_step_by_name["completion"],
            completion_ladder_step=completion_ladder_step_by_name["completion"],
            scenario_step=scenario_step_by_name["completion"],
            report_completion_ladder_driver_signature=None,
        ),
    )

    return Phase7SameSeedObservedRerunRungGuardProbe(
        status="same-seed-observed-rerun-rung-guard-probe-satisfied",
        canonical_seed_order=canonical_seed_order,
        admission_order_driver_signature=(
            landing_guard_probe.admission_order_driver_signature
        ),
        current_frontier_driver_signature=(
            scenario_probe.current_frontier_driver_signature
        ),
        runtime_witness_path=runtime_witness_path,
        acceptance_readout_path=acceptance_readout_path,
        exact_trim_floor_value=baseline_report.exact_trim_floor_value,
        target_fold3_low_pi_treated_count=(
            baseline_report.target_fold3_low_pi_treated_count
        ),
        target_exact_trim_floor_count=baseline_report.target_exact_trim_floor_count,
        target_exact_trim_floor_share_of_fold3_low_pi_count=(
            baseline_report.target_exact_trim_floor_share_of_fold3_low_pi_count
        ),
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            baseline_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            baseline_report.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        binding_repair_slot=binding_repair_slot,
        residual_repair_slot=residual_repair_slot,
        scenarios=scenarios,
    )


@dataclass(slots=True)
class Phase7SameSeedFirstHopSourceQuotaProbeStep:
    scenario_name: str
    current_rung_status: str
    source_uniqueness_guard_driver_signature: str
    witness_floor_quota_driver_signature: str
    point_miss_vector: tuple[int, int, int]
    band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    remaining_witness_floor_gap: float
    binding_slot_witness_floor_increment: float
    residual_slot_witness_floor_increment: float
    binding_slot_share_of_baseline_open_witness_floor_gap: float
    residual_slot_share_of_baseline_open_witness_floor_gap: float
    next_required_slot: Phase7SameSeedCompletionLadderRepairSlot | None
    queued_residual_slot_still_live: bool
    canonical_source_uniqueness_guard_digest: tuple[str, ...]
    canonical_witness_floor_quota_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.scenario_name = str(self.scenario_name).strip()
        self.current_rung_status = str(self.current_rung_status).strip()
        self.source_uniqueness_guard_driver_signature = str(
            self.source_uniqueness_guard_driver_signature
        ).strip()
        self.witness_floor_quota_driver_signature = str(
            self.witness_floor_quota_driver_signature
        ).strip()
        self.point_miss_vector = tuple(int(value) for value in self.point_miss_vector)
        self.band_miss_vector = tuple(int(value) for value in self.band_miss_vector)
        self.current_witness_floor = float(self.current_witness_floor)
        self.remaining_witness_floor_gap = float(self.remaining_witness_floor_gap)
        self.binding_slot_witness_floor_increment = float(
            self.binding_slot_witness_floor_increment
        )
        self.residual_slot_witness_floor_increment = float(
            self.residual_slot_witness_floor_increment
        )
        self.binding_slot_share_of_baseline_open_witness_floor_gap = float(
            self.binding_slot_share_of_baseline_open_witness_floor_gap
        )
        self.residual_slot_share_of_baseline_open_witness_floor_gap = float(
            self.residual_slot_share_of_baseline_open_witness_floor_gap
        )
        self.queued_residual_slot_still_live = bool(
            self.queued_residual_slot_still_live
        )
        self.canonical_source_uniqueness_guard_digest = tuple(
            str(line).rstrip() for line in self.canonical_source_uniqueness_guard_digest
        )
        self.canonical_witness_floor_quota_digest = tuple(
            str(line).rstrip() for line in self.canonical_witness_floor_quota_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "current_rung_status": self.current_rung_status,
            "source_uniqueness_guard_driver_signature": (
                self.source_uniqueness_guard_driver_signature
            ),
            "witness_floor_quota_driver_signature": (
                self.witness_floor_quota_driver_signature
            ),
            "point_miss_vector": list(self.point_miss_vector),
            "band_miss_vector": list(self.band_miss_vector),
            "current_witness_floor": self.current_witness_floor,
            "remaining_witness_floor_gap": self.remaining_witness_floor_gap,
            "binding_slot_witness_floor_increment": (
                self.binding_slot_witness_floor_increment
            ),
            "residual_slot_witness_floor_increment": (
                self.residual_slot_witness_floor_increment
            ),
            "binding_slot_share_of_baseline_open_witness_floor_gap": (
                self.binding_slot_share_of_baseline_open_witness_floor_gap
            ),
            "residual_slot_share_of_baseline_open_witness_floor_gap": (
                self.residual_slot_share_of_baseline_open_witness_floor_gap
            ),
            "next_required_slot": (
                None
                if self.next_required_slot is None
                else self.next_required_slot.to_dict()
            ),
            "queued_residual_slot_still_live": self.queued_residual_slot_still_live,
            "canonical_source_uniqueness_guard_digest": list(
                self.canonical_source_uniqueness_guard_digest
            ),
            "canonical_witness_floor_quota_digest": list(
                self.canonical_witness_floor_quota_digest
            ),
        }


@dataclass(slots=True)
class Phase7SameSeedFirstHopSourceQuotaProbe:
    status: str
    canonical_seed_order: tuple[int, ...]
    admission_order_driver_signature: str
    current_frontier_driver_signature: str
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    exact_trim_floor_value: float
    target_fold3_low_pi_treated_count: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_share_of_fold3_low_pi_count: float
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    comparator_random_states: tuple[int, int]
    comparator_fold3_inverse_pi_phi1_center_projections: tuple[float, float]
    binding_repair_slot: Phase7SameSeedCompletionLadderRepairSlot
    residual_repair_slot: Phase7SameSeedCompletionLadderRepairSlot
    scenarios: tuple[Phase7SameSeedFirstHopSourceQuotaProbeStep, ...]

    def __post_init__(self) -> None:
        self.status = str(self.status).strip()
        self.canonical_seed_order = tuple(
            int(value) for value in self.canonical_seed_order
        )
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.current_frontier_driver_signature = str(
            self.current_frontier_driver_signature
        ).strip()
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.exact_trim_floor_value = float(self.exact_trim_floor_value)
        self.target_fold3_low_pi_treated_count = int(
            self.target_fold3_low_pi_treated_count
        )
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_share_of_fold3_low_pi_count = float(
            self.target_exact_trim_floor_share_of_fold3_low_pi_count
        )
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.target_exact_trim_floor_weighted_retention = float(
            self.target_exact_trim_floor_weighted_retention
        )
        self.comparator_random_states = tuple(
            int(value) for value in self.comparator_random_states
        )
        self.comparator_fold3_inverse_pi_phi1_center_projections = tuple(
            float(value)
            for value in self.comparator_fold3_inverse_pi_phi1_center_projections
        )
        self.scenarios = tuple(self.scenarios)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical_seed_order": list(self.canonical_seed_order),
            "admission_order_driver_signature": self.admission_order_driver_signature,
            "current_frontier_driver_signature": self.current_frontier_driver_signature,
            "runtime_witness_path": list(self.runtime_witness_path),
            "acceptance_readout_path": list(self.acceptance_readout_path),
            "exact_trim_floor_value": self.exact_trim_floor_value,
            "target_fold3_low_pi_treated_count": (
                self.target_fold3_low_pi_treated_count
            ),
            "target_exact_trim_floor_count": self.target_exact_trim_floor_count,
            "target_exact_trim_floor_share_of_fold3_low_pi_count": (
                self.target_exact_trim_floor_share_of_fold3_low_pi_count
            ),
            "target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi": (
                self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
            ),
            "target_exact_trim_floor_weighted_share_of_fold3_weighted": (
                self.target_exact_trim_floor_weighted_share_of_fold3_weighted
            ),
            "target_exact_trim_floor_weighted_retention": (
                self.target_exact_trim_floor_weighted_retention
            ),
            "comparator_random_states": list(self.comparator_random_states),
            "comparator_fold3_inverse_pi_phi1_center_projections": list(
                self.comparator_fold3_inverse_pi_phi1_center_projections
            ),
            "binding_repair_slot": self.binding_repair_slot.to_dict(),
            "residual_repair_slot": self.residual_repair_slot.to_dict(),
            "scenarios": [step.to_dict() for step in self.scenarios],
        }


def _phase7_same_seed_first_hop_source_quota_probe_slot_from_fields(
    *,
    random_state: int | None,
    seed_group: str | None,
    z_index: int | None,
    z_value: float | None,
) -> Phase7SameSeedCompletionLadderRepairSlot | None:
    if random_state is None or seed_group is None or z_index is None or z_value is None:
        return None
    return Phase7SameSeedCompletionLadderRepairSlot(
        random_state=random_state,
        seed_group=seed_group,
        z_index=z_index,
        z_value=z_value,
    )


def _phase7_same_seed_first_hop_source_quota_probe_step_vectors(
    *,
    scenario_name: str,
    quota_report: Any,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if scenario_name == "baseline_open":
        return (
            tuple(int(value) for value in quota_report.baseline_point_miss_vector),
            tuple(int(value) for value in quota_report.baseline_band_miss_vector),
        )
    if scenario_name == "binding_only":
        return (
            tuple(int(value) for value in quota_report.binding_only_point_miss_vector),
            tuple(int(value) for value in quota_report.binding_only_band_miss_vector),
        )
    if scenario_name == "completion":
        return (
            tuple(int(value) for value in quota_report.completion_point_miss_vector),
            tuple(int(value) for value in quota_report.completion_band_miss_vector),
        )
    raise ValueError(
        "same-seed first-hop source quota probe only accepts baseline_open, binding_only, or completion"
    )


def _phase7_same_seed_first_hop_source_quota_probe_step(
    *,
    scenario_name: str,
    source_uniqueness_guard_report: Any,
    witness_floor_quota_report: Any,
) -> Phase7SameSeedFirstHopSourceQuotaProbeStep:
    point_miss_vector, band_miss_vector = (
        _phase7_same_seed_first_hop_source_quota_probe_step_vectors(
            scenario_name=scenario_name,
            quota_report=witness_floor_quota_report,
        )
    )
    if point_miss_vector != tuple(
        int(value) for value in source_uniqueness_guard_report.current_point_miss_vector
    ):
        raise ValueError(
            "same-seed first-hop source quota probe requires source uniqueness and witness-floor quota point-miss sync"
        )
    if band_miss_vector != tuple(
        int(value) for value in source_uniqueness_guard_report.current_band_miss_vector
    ):
        raise ValueError(
            "same-seed first-hop source quota probe requires source uniqueness and witness-floor quota band-miss sync"
        )
    if (
        str(source_uniqueness_guard_report.current_rung_status).strip()
        != str(witness_floor_quota_report.current_rung_status).strip()
    ):
        raise ValueError(
            "same-seed first-hop source quota probe requires source uniqueness and witness-floor quota rung-status sync"
        )
    if (
        abs(
            float(source_uniqueness_guard_report.current_witness_floor)
            - float(witness_floor_quota_report.current_witness_floor)
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed first-hop source quota probe requires source uniqueness and witness-floor quota witness-floor sync"
        )

    next_required_slot = (
        _phase7_same_seed_first_hop_source_quota_probe_slot_from_fields(
            random_state=source_uniqueness_guard_report.next_required_slot_random_state,
            seed_group=source_uniqueness_guard_report.next_required_slot_seed_group,
            z_index=source_uniqueness_guard_report.next_required_slot_z_index,
            z_value=source_uniqueness_guard_report.next_required_slot_z_value,
        )
    )

    return Phase7SameSeedFirstHopSourceQuotaProbeStep(
        scenario_name=scenario_name,
        current_rung_status=source_uniqueness_guard_report.current_rung_status,
        source_uniqueness_guard_driver_signature=(
            source_uniqueness_guard_report.driver_signature
        ),
        witness_floor_quota_driver_signature=(
            witness_floor_quota_report.driver_signature
        ),
        point_miss_vector=point_miss_vector,
        band_miss_vector=band_miss_vector,
        current_witness_floor=source_uniqueness_guard_report.current_witness_floor,
        remaining_witness_floor_gap=(
            witness_floor_quota_report.current_remaining_witness_floor_gap
        ),
        binding_slot_witness_floor_increment=(
            witness_floor_quota_report.binding_slot_witness_floor_increment
        ),
        residual_slot_witness_floor_increment=(
            witness_floor_quota_report.residual_slot_witness_floor_increment
        ),
        binding_slot_share_of_baseline_open_witness_floor_gap=(
            witness_floor_quota_report.binding_slot_share_of_baseline_open_witness_floor_gap
        ),
        residual_slot_share_of_baseline_open_witness_floor_gap=(
            witness_floor_quota_report.residual_slot_share_of_baseline_open_witness_floor_gap
        ),
        next_required_slot=next_required_slot,
        queued_residual_slot_still_live=next_required_slot is not None,
        canonical_source_uniqueness_guard_digest=(
            source_uniqueness_guard_report.canonical_observed_rerun_first_hop_source_uniqueness_guard_digest
        ),
        canonical_witness_floor_quota_digest=(
            witness_floor_quota_report.canonical_observed_rerun_first_hop_witness_floor_quota_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_source_quota_probe() -> (
    Phase7SameSeedFirstHopSourceQuotaProbe
):
    import importlib

    source_uniqueness_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard"
    )
    witness_floor_quota_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota"
    )
    binding_slot_progress_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile"
    )
    residual_slot_completion_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile"
    )

    rung_guard_probe = run_phase7_same_seed_observed_rerun_rung_guard_probe()

    baseline_source_uniqueness_guard = source_uniqueness_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard()
    baseline_witness_floor_quota = witness_floor_quota_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota()

    binding_only_after_report = binding_slot_progress_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile().binding_only_after_report
    binding_only_source_uniqueness_guard = source_uniqueness_module.build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard_report(
        after_report=binding_only_after_report
    )
    binding_only_witness_floor_quota = witness_floor_quota_module.build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota_report(
        after_report=binding_only_after_report
    )

    completion_after_report = residual_slot_completion_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile().completion_after_report
    completion_source_uniqueness_guard = source_uniqueness_module.build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_uniqueness_guard_report(
        after_report=completion_after_report
    )
    completion_witness_floor_quota = witness_floor_quota_module.build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_witness_floor_quota_report(
        after_report=completion_after_report
    )

    canonical_seed_order = tuple(
        int(value) for value in baseline_source_uniqueness_guard.same_seed_random_states
    )
    if canonical_seed_order != rung_guard_probe.canonical_seed_order:
        raise ValueError(
            "same-seed first-hop source quota probe requires the observed rerun rung guard to keep the canonical seed order"
        )

    for report in (
        baseline_witness_floor_quota,
        binding_only_source_uniqueness_guard,
        binding_only_witness_floor_quota,
        completion_source_uniqueness_guard,
        completion_witness_floor_quota,
    ):
        if canonical_seed_order != tuple(
            int(value) for value in report.same_seed_random_states
        ):
            raise ValueError(
                "same-seed first-hop source quota probe requires source uniqueness and witness-floor quota reports to keep the canonical seed order"
            )

    runtime_witness_path = tuple(rung_guard_probe.runtime_witness_path)
    acceptance_readout_path = tuple(rung_guard_probe.acceptance_readout_path)

    binding_repair_slot = (
        _phase7_same_seed_first_hop_source_quota_probe_slot_from_fields(
            random_state=baseline_witness_floor_quota.binding_slot_random_state,
            seed_group=baseline_witness_floor_quota.binding_slot_seed_group,
            z_index=baseline_witness_floor_quota.binding_slot_z_index,
            z_value=baseline_witness_floor_quota.binding_slot_z_value,
        )
    )
    residual_repair_slot = (
        _phase7_same_seed_first_hop_source_quota_probe_slot_from_fields(
            random_state=baseline_witness_floor_quota.residual_slot_random_state,
            seed_group=baseline_witness_floor_quota.residual_slot_seed_group,
            z_index=baseline_witness_floor_quota.residual_slot_z_index,
            z_value=baseline_witness_floor_quota.residual_slot_z_value,
        )
    )

    if (
        binding_repair_slot is None
        or binding_repair_slot.to_dict()
        != rung_guard_probe.binding_repair_slot.to_dict()
    ):
        raise ValueError(
            "same-seed first-hop source quota probe requires the binding repair slot to match the observed rerun rung guard"
        )
    if (
        residual_repair_slot is None
        or residual_repair_slot.to_dict()
        != rung_guard_probe.residual_repair_slot.to_dict()
    ):
        raise ValueError(
            "same-seed first-hop source quota probe requires the residual repair slot to match the observed rerun rung guard"
        )

    rung_guard_step_by_name = {
        str(step.scenario_name): step for step in rung_guard_probe.scenarios
    }
    scenario_reports = (
        (
            "baseline_open",
            baseline_source_uniqueness_guard,
            baseline_witness_floor_quota,
        ),
        (
            "binding_only",
            binding_only_source_uniqueness_guard,
            binding_only_witness_floor_quota,
        ),
        (
            "completion",
            completion_source_uniqueness_guard,
            completion_witness_floor_quota,
        ),
    )

    scenarios: list[Phase7SameSeedFirstHopSourceQuotaProbeStep] = []
    for scenario_name, source_guard_report, quota_report in scenario_reports:
        if scenario_name not in rung_guard_step_by_name:
            raise ValueError(
                "same-seed first-hop source quota probe requires the observed rerun rung guard to expose "
                f"{scenario_name}"
            )
        rung_guard_step = rung_guard_step_by_name[scenario_name]
        if (
            str(source_guard_report.current_rung_status).strip()
            != str(rung_guard_step.resulting_rung_status).strip()
        ):
            raise ValueError(
                "same-seed first-hop source quota probe requires source uniqueness status to match the observed rerun rung guard"
            )
        if (
            abs(
                float(source_guard_report.current_witness_floor)
                - float(rung_guard_step.witness_floor)
            )
            > 1e-12
        ):
            raise ValueError(
                "same-seed first-hop source quota probe requires source uniqueness witness-floor state to match the observed rerun rung guard"
            )
        scenarios.append(
            _phase7_same_seed_first_hop_source_quota_probe_step(
                scenario_name=scenario_name,
                source_uniqueness_guard_report=source_guard_report,
                witness_floor_quota_report=quota_report,
            )
        )

    return Phase7SameSeedFirstHopSourceQuotaProbe(
        status="same-seed-first-hop-source-quota-probe-satisfied",
        canonical_seed_order=canonical_seed_order,
        admission_order_driver_signature=(
            rung_guard_probe.admission_order_driver_signature
        ),
        current_frontier_driver_signature=(
            rung_guard_probe.current_frontier_driver_signature
        ),
        runtime_witness_path=runtime_witness_path,
        acceptance_readout_path=acceptance_readout_path,
        exact_trim_floor_value=rung_guard_probe.exact_trim_floor_value,
        target_fold3_low_pi_treated_count=(
            rung_guard_probe.target_fold3_low_pi_treated_count
        ),
        target_exact_trim_floor_count=rung_guard_probe.target_exact_trim_floor_count,
        target_exact_trim_floor_share_of_fold3_low_pi_count=(
            rung_guard_probe.target_exact_trim_floor_share_of_fold3_low_pi_count
        ),
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            baseline_source_uniqueness_guard.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            baseline_source_uniqueness_guard.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            baseline_source_uniqueness_guard.target_exact_trim_floor_weighted_retention
        ),
        comparator_random_states=baseline_source_uniqueness_guard.comparator_random_states,
        comparator_fold3_inverse_pi_phi1_center_projections=(
            baseline_source_uniqueness_guard.comparator_fold3_inverse_pi_phi1_center_projections
        ),
        binding_repair_slot=binding_repair_slot,
        residual_repair_slot=residual_repair_slot,
        scenarios=tuple(scenarios),
    )


@dataclass(slots=True)
class Phase7SameSeedFirstHopSourceBridgeProbe:
    stage_label: str
    driver_signature: str
    bridge_driver_signature: str
    binding_slot_priority_driver_signature: str
    same_seed_random_states: tuple[int, ...]
    binding_slot_random_state: int
    binding_slot_z_value: float
    binding_slot_repair_role: str
    residual_slot_random_state: int
    residual_slot_z_value: float
    residual_slot_repair_role: str
    current_point_miss_vector: tuple[int, int, int]
    current_band_miss_vector: tuple[int, int, int]
    current_witness_floor: float
    required_min_witness_floor: float
    target_fold_id: int
    target_exact_trim_floor_count: int
    target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi: float
    target_exact_trim_floor_weighted_share_of_fold3_weighted: float
    target_exact_trim_floor_weighted_retention: float
    canonical_source_bridge_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.driver_signature = str(self.driver_signature).strip()
        self.bridge_driver_signature = str(self.bridge_driver_signature).strip()
        self.binding_slot_priority_driver_signature = str(
            self.binding_slot_priority_driver_signature
        ).strip()
        self.same_seed_random_states = tuple(
            int(value) for value in self.same_seed_random_states
        )
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.binding_slot_repair_role = str(self.binding_slot_repair_role).strip()
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.residual_slot_repair_role = str(self.residual_slot_repair_role).strip()
        self.current_point_miss_vector = tuple(
            int(value) for value in self.current_point_miss_vector
        )
        self.current_band_miss_vector = tuple(
            int(value) for value in self.current_band_miss_vector
        )
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.target_fold_id = int(self.target_fold_id)
        self.target_exact_trim_floor_count = int(self.target_exact_trim_floor_count)
        self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi = float(
            self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        )
        self.target_exact_trim_floor_weighted_share_of_fold3_weighted = float(
            self.target_exact_trim_floor_weighted_share_of_fold3_weighted
        )
        self.target_exact_trim_floor_weighted_retention = float(
            self.target_exact_trim_floor_weighted_retention
        )
        self.canonical_source_bridge_digest = tuple(
            str(line).rstrip() for line in self.canonical_source_bridge_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "driver_signature": self.driver_signature,
            "bridge_driver_signature": self.bridge_driver_signature,
            "binding_slot_priority_driver_signature": (
                self.binding_slot_priority_driver_signature
            ),
            "same_seed_random_states": list(self.same_seed_random_states),
            "binding_slot_random_state": self.binding_slot_random_state,
            "binding_slot_z_value": self.binding_slot_z_value,
            "binding_slot_repair_role": self.binding_slot_repair_role,
            "residual_slot_random_state": self.residual_slot_random_state,
            "residual_slot_z_value": self.residual_slot_z_value,
            "residual_slot_repair_role": self.residual_slot_repair_role,
            "current_point_miss_vector": list(self.current_point_miss_vector),
            "current_band_miss_vector": list(self.current_band_miss_vector),
            "current_witness_floor": self.current_witness_floor,
            "required_min_witness_floor": self.required_min_witness_floor,
            "target_fold_id": self.target_fold_id,
            "target_exact_trim_floor_count": self.target_exact_trim_floor_count,
            "target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi": (
                self.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
            ),
            "target_exact_trim_floor_weighted_share_of_fold3_weighted": (
                self.target_exact_trim_floor_weighted_share_of_fold3_weighted
            ),
            "target_exact_trim_floor_weighted_retention": (
                self.target_exact_trim_floor_weighted_retention
            ),
            "canonical_source_bridge_digest": list(self.canonical_source_bridge_digest),
        }


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_source_bridge_probe() -> (
    Phase7SameSeedFirstHopSourceBridgeProbe
):
    import importlib

    source_bridge_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge"
    )

    source_bridge_report = source_bridge_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_source_bridge()
    rung_guard_probe = run_phase7_same_seed_observed_rerun_rung_guard_probe()

    canonical_seed_order = tuple(
        int(value) for value in source_bridge_report.same_seed_random_states
    )
    if canonical_seed_order != rung_guard_probe.canonical_seed_order:
        raise ValueError(
            "same-seed first-hop source bridge probe requires the observed rerun rung guard to keep the canonical seed order"
        )

    binding_repair_slot = rung_guard_probe.binding_repair_slot
    residual_repair_slot = rung_guard_probe.residual_repair_slot
    if (
        source_bridge_report.residual_slot_random_state
        != residual_repair_slot.random_state
    ):
        raise ValueError(
            "same-seed first-hop source bridge probe requires the residual repair slot to stay aligned with the observed rerun rung guard"
        )
    if (
        abs(
            float(source_bridge_report.residual_slot_z_value)
            - float(residual_repair_slot.z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed first-hop source bridge probe requires the residual repair z value to stay aligned with the observed rerun rung guard"
        )
    if source_bridge_report.driver_signature.endswith("-open"):
        if (
            source_bridge_report.next_required_slot_random_state
            != binding_repair_slot.random_state
        ):
            raise ValueError(
                "same-seed first-hop source bridge probe requires the current first hop to stay on seed 303"
            )
        if source_bridge_report.next_required_slot_z_value is None or (
            abs(
                float(source_bridge_report.next_required_slot_z_value)
                - float(binding_repair_slot.z_value)
            )
            > 1e-12
        ):
            raise ValueError(
                "same-seed first-hop source bridge probe requires the current first-hop z value to stay aligned with the observed rerun rung guard"
            )

    return Phase7SameSeedFirstHopSourceBridgeProbe(
        stage_label=source_bridge_report.stage_label,
        driver_signature=source_bridge_report.driver_signature,
        bridge_driver_signature=source_bridge_report.bridge_driver_signature,
        binding_slot_priority_driver_signature=(
            source_bridge_report.binding_slot_priority_driver_signature
        ),
        same_seed_random_states=canonical_seed_order,
        binding_slot_random_state=binding_repair_slot.random_state,
        binding_slot_z_value=binding_repair_slot.z_value,
        binding_slot_repair_role="binding-witness-floor-lift",
        residual_slot_random_state=source_bridge_report.residual_slot_random_state,
        residual_slot_z_value=source_bridge_report.residual_slot_z_value,
        residual_slot_repair_role=source_bridge_report.residual_slot_repair_role,
        current_point_miss_vector=source_bridge_report.current_point_miss_vector,
        current_band_miss_vector=source_bridge_report.current_band_miss_vector,
        current_witness_floor=source_bridge_report.current_witness_floor,
        required_min_witness_floor=source_bridge_report.required_min_witness_floor,
        target_fold_id=source_bridge_report.target_fold_id,
        target_exact_trim_floor_count=source_bridge_report.target_exact_trim_floor_count,
        target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi=(
            source_bridge_report.target_exact_trim_floor_inverse_pi_share_of_fold3_inverse_pi
        ),
        target_exact_trim_floor_weighted_share_of_fold3_weighted=(
            source_bridge_report.target_exact_trim_floor_weighted_share_of_fold3_weighted
        ),
        target_exact_trim_floor_weighted_retention=(
            source_bridge_report.target_exact_trim_floor_weighted_retention
        ),
        canonical_source_bridge_digest=(
            source_bridge_report.canonical_observed_rerun_first_hop_source_bridge_digest
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_contract_stack_probe() -> Any:
    import importlib

    contract_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract"
    )

    return contract_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_contract()


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_acceptance_object_flow_contract() -> Any:
    import importlib

    object_flow_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract"
    )

    return object_flow_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_object_flow_contract()


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_acceptance_eq31_source_contract() -> Any:
    import importlib

    eq31_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract"
    )

    return eq31_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_eq31_source_contract()


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_acceptance_raw_score_component_contract() -> Any:
    import importlib

    raw_score_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract"
    )

    return raw_score_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_raw_score_component_contract()


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_acceptance_delta_y_support_chain_contract() -> Any:
    import importlib

    delta_y_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract"
    )

    return delta_y_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_delta_y_support_chain_contract()


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_acceptance_low_pi_support_allocation_contract() -> (
    Any
):
    import importlib

    low_pi_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract"
    )

    return low_pi_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_low_pi_support_allocation_contract()


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract() -> (
    Any
):
    import importlib

    inverse_pi_concentration_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract"
    )

    return inverse_pi_concentration_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract()


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract() -> (
    Any
):
    import importlib

    inverse_pi_conduit_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract"
    )

    return inverse_pi_conduit_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract()


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_landing_bridge() -> Any:
    import importlib

    landing_bridge_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge"
    )

    return landing_bridge_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_landing_bridge()


@lru_cache(maxsize=1)
def run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_runtime_bridge() -> Any:
    import importlib

    runtime_bridge_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge"
    )

    return runtime_bridge_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_first_hop_acceptance_exact_trim_floor_runtime_bridge()


_PHASE7_SAME_SEED_HELPER_TOKENS = (
    "run_phase7_same_seed_before_after_acceptance_probe",
    "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_before_after_acceptance_probe",
    "run_phase7_same_seed_completion_ladder_probe",
    "run_phase7_same_seed_repair_agenda_probe",
    "run_phase7_same_seed_first_hop_runtime_binding_packet_probe",
    "run_phase7_same_seed_first_hop_landing_guard_probe",
    "run_phase7_same_seed_first_hop_execution_queue_probe",
    "run_phase7_same_seed_first_hop_contract_stack_probe",
    "run_phase7_same_seed_first_hop_source_quota_probe",
    "run_phase7_same_seed_observed_rerun_rung_guard_probe",
    "run_phase7_same_seed_first_hop_source_bridge_probe",
)

_PHASE7_SAME_SEED_ACCEPTANCE_HELPER_TOKENS = (
    "run_phase7_same_seed_first_hop_acceptance_object_flow_contract",
    "run_phase7_same_seed_first_hop_acceptance_eq31_source_contract",
    "run_phase7_same_seed_first_hop_acceptance_raw_score_component_contract",
    "run_phase7_same_seed_first_hop_acceptance_delta_y_support_chain_contract",
    "run_phase7_same_seed_first_hop_acceptance_low_pi_support_allocation_contract",
    "run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_inverse_pi_concentration_contract",
    "run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_inverse_pi_conduit_contract",
    "run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_landing_bridge",
    "run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_runtime_bridge",
)

_PHASE7_SAME_SEED_FRONTIER_HELPER_TOKENS = (
    "run_phase7_same_seed_runtime_bridge_source_guard",
    "run_phase7_same_seed_runtime_bridge_after_report_admission_order",
    "run_phase7_same_seed_runtime_bridge_after_report_intake",
    "run_phase7_same_seed_runtime_bridge_after_report_frontier_packet",
    "run_phase7_same_seed_exact_witness_target_gap",
)

_PHASE7_SAME_SEED_LIGHTWEIGHT_FRONTIER_HELPER_TOKENS = (
    "run_phase7_same_seed_runtime_bridge_after_report_scenario_probe",
    "run_phase7_same_seed_trace_payload_cache_bridge_probe",
)

_PHASE7_SAME_SEED_SEED303_HELPER_TOKENS = (
    "run_phase7_same_seed_seed303_support_trace_index",
    "run_phase7_same_seed_seed303_trim_floor_support_trace_index",
    "run_phase7_same_seed_seed303_object_flow_blocker",
    "run_phase7_same_seed_seed303_frontier_packet",
    "run_phase7_same_seed_seed303_frontier_packet_acceptance_stack",
    "run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index",
)

_PHASE7_SAME_SEED_ACTIONABLE_ORDER = (
    "same-seed-admission-order",
    "acceptance raw-score component contract",
    "acceptance exact trim-floor runtime bridge",
)


@dataclass(slots=True)
class Phase7SameSeedHelperIndexReport:
    stage_label: str
    live_entry: str
    before_after_gate_status: str
    validation_only: bool
    helper_tokens: tuple[str, ...]
    acceptance_helper_tokens: tuple[str, ...]
    frontier_helper_tokens: tuple[str, ...]
    lightweight_frontier_helper_tokens: tuple[str, ...]
    seed303_helper_tokens: tuple[str, ...]
    actionable_order: tuple[str, ...]
    binding_slot_random_state: int
    binding_slot_z_value: float
    residual_slot_random_state: int
    residual_slot_z_value: float
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    runtime_bridge_source_guard_driver_signature: str
    runtime_bridge_after_report_frontier_packet_driver_signature: str
    runtime_bridge_after_report_scenario_status: str
    trace_payload_cache_bridge_status: str
    exact_witness_target_gap_driver_signature: str
    seed303_support_trace_driver_signature: str
    seed303_trim_floor_support_trace_driver_signature: str
    seed303_object_flow_blocker_driver_signature: str
    seed303_acceptance_exact_trim_floor_stack_driver_signature: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.live_entry = str(self.live_entry).strip()
        self.before_after_gate_status = str(self.before_after_gate_status).strip()
        self.validation_only = bool(self.validation_only)
        self.helper_tokens = tuple(str(item).strip() for item in self.helper_tokens)
        self.acceptance_helper_tokens = tuple(
            str(item).strip() for item in self.acceptance_helper_tokens
        )
        self.frontier_helper_tokens = tuple(
            str(item).strip() for item in self.frontier_helper_tokens
        )
        self.lightweight_frontier_helper_tokens = tuple(
            str(item).strip() for item in self.lightweight_frontier_helper_tokens
        )
        self.seed303_helper_tokens = tuple(
            str(item).strip() for item in self.seed303_helper_tokens
        )
        self.actionable_order = tuple(
            str(item).strip() for item in self.actionable_order
        )
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.runtime_bridge_source_guard_driver_signature = str(
            self.runtime_bridge_source_guard_driver_signature
        ).strip()
        self.runtime_bridge_after_report_frontier_packet_driver_signature = str(
            self.runtime_bridge_after_report_frontier_packet_driver_signature
        ).strip()
        self.runtime_bridge_after_report_scenario_status = str(
            self.runtime_bridge_after_report_scenario_status
        ).strip()
        self.trace_payload_cache_bridge_status = str(
            self.trace_payload_cache_bridge_status
        ).strip()
        self.exact_witness_target_gap_driver_signature = str(
            self.exact_witness_target_gap_driver_signature
        ).strip()
        self.seed303_support_trace_driver_signature = str(
            self.seed303_support_trace_driver_signature
        ).strip()
        self.seed303_trim_floor_support_trace_driver_signature = str(
            self.seed303_trim_floor_support_trace_driver_signature
        ).strip()
        self.seed303_object_flow_blocker_driver_signature = str(
            self.seed303_object_flow_blocker_driver_signature
        ).strip()
        self.seed303_acceptance_exact_trim_floor_stack_driver_signature = str(
            self.seed303_acceptance_exact_trim_floor_stack_driver_signature
        ).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "live_entry": self.live_entry,
            "before_after_gate_status": self.before_after_gate_status,
            "validation_only": self.validation_only,
            "helper_tokens": list(self.helper_tokens),
            "acceptance_helper_tokens": list(self.acceptance_helper_tokens),
            "frontier_helper_tokens": list(self.frontier_helper_tokens),
            "lightweight_frontier_helper_tokens": list(
                self.lightweight_frontier_helper_tokens
            ),
            "seed303_helper_tokens": list(self.seed303_helper_tokens),
            "actionable_order": list(self.actionable_order),
            "binding_slot_random_state": self.binding_slot_random_state,
            "binding_slot_z_value": self.binding_slot_z_value,
            "residual_slot_random_state": self.residual_slot_random_state,
            "residual_slot_z_value": self.residual_slot_z_value,
            "runtime_witness_path": list(self.runtime_witness_path),
            "acceptance_readout_path": list(self.acceptance_readout_path),
            "runtime_bridge_source_guard_driver_signature": (
                self.runtime_bridge_source_guard_driver_signature
            ),
            "runtime_bridge_after_report_frontier_packet_driver_signature": (
                self.runtime_bridge_after_report_frontier_packet_driver_signature
            ),
            "runtime_bridge_after_report_scenario_status": (
                self.runtime_bridge_after_report_scenario_status
            ),
            "trace_payload_cache_bridge_status": (
                self.trace_payload_cache_bridge_status
            ),
            "exact_witness_target_gap_driver_signature": (
                self.exact_witness_target_gap_driver_signature
            ),
            "seed303_support_trace_driver_signature": (
                self.seed303_support_trace_driver_signature
            ),
            "seed303_trim_floor_support_trace_driver_signature": (
                self.seed303_trim_floor_support_trace_driver_signature
            ),
            "seed303_object_flow_blocker_driver_signature": (
                self.seed303_object_flow_blocker_driver_signature
            ),
            "seed303_acceptance_exact_trim_floor_stack_driver_signature": (
                self.seed303_acceptance_exact_trim_floor_stack_driver_signature
            ),
        }


@lru_cache(maxsize=1)
def run_phase7_same_seed_helper_index() -> Phase7SameSeedHelperIndexReport:
    before_after_probe = run_phase7_same_seed_before_after_acceptance_probe()
    runtime_bridge_source_guard = run_phase7_same_seed_runtime_bridge_source_guard()
    runtime_bridge_admission_order = (
        run_phase7_same_seed_runtime_bridge_after_report_admission_order()
    )
    runtime_bridge_intake = run_phase7_same_seed_runtime_bridge_after_report_intake()
    runtime_bridge_frontier_packet = (
        run_phase7_same_seed_runtime_bridge_after_report_frontier_packet()
    )
    exact_witness_target_gap = run_phase7_same_seed_exact_witness_target_gap()
    runtime_bridge_after_report_scenario_probe = (
        run_phase7_same_seed_runtime_bridge_after_report_scenario_probe()
    )
    trace_payload_cache_bridge_probe = (
        run_phase7_same_seed_trace_payload_cache_bridge_probe()
    )
    execution_queue_probe = run_phase7_same_seed_first_hop_execution_queue_probe()
    contract_stack_probe = run_phase7_same_seed_first_hop_contract_stack_probe()
    source_bridge_probe = run_phase7_same_seed_first_hop_source_bridge_probe()
    acceptance_runtime_bridge = (
        run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_runtime_bridge()
    )
    seed303_support_trace_index = run_phase7_same_seed_seed303_support_trace_index()
    seed303_trim_floor_support_trace_index = (
        run_phase7_same_seed_seed303_trim_floor_support_trace_index()
    )
    seed303_object_flow_blocker = run_phase7_same_seed_seed303_object_flow_blocker()
    seed303_frontier_packet = run_phase7_same_seed_seed303_frontier_packet()
    seed303_frontier_packet_acceptance_stack = (
        run_phase7_same_seed_seed303_frontier_packet_acceptance_stack()
    )
    seed303_acceptance_exact_trim_floor_stack = (
        run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index()
    )

    baseline_step = next(
        step
        for step in before_after_probe.scenarios
        if step.scenario_name == "identity_replay"
    )
    if (
        baseline_step.driver_signature
        != "same-seed-before-after-acceptance-not-yet-satisfied"
    ):
        raise ValueError(
            "same-seed helper index requires the baseline before/after gate to stay open"
        )
    if (
        before_after_probe.admission_order_driver_signature
        != execution_queue_probe.admission_order_driver_signature
    ):
        raise ValueError(
            "same-seed helper index requires the before/after gate to share the admission-order signature"
        )
    if (
        runtime_bridge_admission_order.admission_driver_signature
        != before_after_probe.admission_order_driver_signature
    ):
        raise ValueError(
            "same-seed helper index requires the runtime-bridge admission-order helper to share the admission-order signature"
        )
    if (
        runtime_bridge_admission_order.before_after_driver_signature
        != baseline_step.driver_signature
        or runtime_bridge_intake.before_after_driver_signature
        != baseline_step.driver_signature
    ):
        raise ValueError(
            "same-seed helper index requires the runtime-bridge frontier helpers to keep the baseline before/after gate"
        )
    if (
        execution_queue_probe.current_actionable_packet.random_state
        != contract_stack_probe.binding_slot_random_state
        or abs(
            float(execution_queue_probe.current_actionable_packet.z_value)
            - float(contract_stack_probe.binding_slot_z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed helper index requires the first-hop contract stack to keep the current actionable packet"
        )
    if (
        execution_queue_probe.queued_residual_slot.random_state
        != contract_stack_probe.residual_slot_random_state
        or abs(
            float(execution_queue_probe.queued_residual_slot.z_value)
            - float(contract_stack_probe.residual_slot_z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed helper index requires the first-hop contract stack to keep the queued residual slot"
        )
    if (
        runtime_bridge_source_guard.binding_slot_random_state
        != contract_stack_probe.binding_slot_random_state
        or abs(
            float(runtime_bridge_source_guard.binding_slot_z_value)
            - float(contract_stack_probe.binding_slot_z_value)
        )
        > 1e-12
        or runtime_bridge_source_guard.residual_slot_random_state
        != contract_stack_probe.residual_slot_random_state
        or abs(
            float(runtime_bridge_source_guard.residual_slot_z_value)
            - float(contract_stack_probe.residual_slot_z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed helper index requires the runtime-bridge source guard to keep the binding/residual slots aligned"
        )
    if (
        source_bridge_probe.binding_slot_random_state
        != contract_stack_probe.binding_slot_random_state
        or abs(
            float(source_bridge_probe.binding_slot_z_value)
            - float(contract_stack_probe.binding_slot_z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed helper index requires the first-hop source bridge to keep the binding slot aligned"
        )
    if (
        source_bridge_probe.residual_slot_random_state
        != contract_stack_probe.residual_slot_random_state
        or abs(
            float(source_bridge_probe.residual_slot_z_value)
            - float(contract_stack_probe.residual_slot_z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed helper index requires the first-hop source bridge to keep the residual slot aligned"
        )
    if (
        runtime_bridge_frontier_packet.binding_slot_random_state
        != contract_stack_probe.binding_slot_random_state
        or abs(
            float(runtime_bridge_frontier_packet.binding_slot_z_value)
            - float(contract_stack_probe.binding_slot_z_value)
        )
        > 1e-12
        or runtime_bridge_frontier_packet.residual_slot_random_state
        != contract_stack_probe.residual_slot_random_state
        or abs(
            float(runtime_bridge_frontier_packet.residual_slot_z_value)
            - float(contract_stack_probe.residual_slot_z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed helper index requires the runtime-bridge frontier packet to keep the binding/residual slots aligned"
        )
    if tuple(runtime_bridge_frontier_packet.acceptance_readout_path) != tuple(
        execution_queue_probe.acceptance_readout_path
    ) or tuple(runtime_bridge_source_guard.acceptance_readout_path) != tuple(
        execution_queue_probe.acceptance_readout_path
    ):
        raise ValueError(
            "same-seed helper index requires the runtime-bridge frontier helpers to keep the acceptance readout aligned"
        )
    if (
        runtime_bridge_after_report_scenario_probe.current_frontier_driver_signature
        != runtime_bridge_frontier_packet.driver_signature
    ):
        raise ValueError(
            "same-seed helper index requires the lightweight after-report scenario probe to keep the runtime-bridge frontier signature aligned"
        )
    if tuple(runtime_bridge_after_report_scenario_probe.runtime_witness_path) != tuple(
        execution_queue_probe.runtime_witness_path
    ) or tuple(
        runtime_bridge_after_report_scenario_probe.acceptance_readout_path
    ) != tuple(execution_queue_probe.acceptance_readout_path):
        raise ValueError(
            "same-seed helper index requires the lightweight after-report scenario probe to keep the runtime witness and acceptance readout aligned"
        )
    if (
        runtime_bridge_after_report_scenario_probe.status
        != "same-seed-runtime-bridge-after-report-scenario-probe-satisfied"
    ):
        raise ValueError(
            "same-seed helper index requires the lightweight after-report scenario probe to stay satisfied"
        )
    if (
        trace_payload_cache_bridge_probe.status
        != "same-seed-trace-payload-cache-bridge-satisfied"
    ):
        raise ValueError(
            "same-seed helper index requires the trace payload cache bridge to stay satisfied"
        )
    if (
        exact_witness_target_gap.driver_signature
        != "same-seed-exact-witness-target-gap-open"
    ):
        raise ValueError(
            "same-seed helper index requires the exact-witness target gap to stay open"
        )
    if tuple(exact_witness_target_gap.runtime_witness_path) != tuple(
        execution_queue_probe.runtime_witness_path
    ):
        raise ValueError(
            "same-seed helper index requires the exact-witness target gap to keep the runtime witness path aligned"
        )
    if tuple(int(value) for value in exact_witness_target_gap.same_seed_random_states) != tuple(
        int(value) for value in runtime_bridge_frontier_packet.same_seed_random_states
    ):
        raise ValueError(
            "same-seed helper index requires the exact-witness target gap to keep the same-seed replay ordering aligned"
        )
    pending_exact_witness_repairs = tuple(
        (int(slot.random_state), float(slot.z_value))
        for slot in exact_witness_target_gap.pending_pointwise_repairs
    )
    expected_exact_witness_repairs = (
        (
            int(contract_stack_probe.binding_slot_random_state),
            float(contract_stack_probe.binding_slot_z_value),
        ),
        (
            int(contract_stack_probe.residual_slot_random_state),
            float(contract_stack_probe.residual_slot_z_value),
        ),
    )
    if (
        len(pending_exact_witness_repairs) != len(expected_exact_witness_repairs)
        or any(
            observed_random_state != expected_random_state
            or abs(observed_z_value - expected_z_value) > 1e-12
            for (observed_random_state, observed_z_value), (
                expected_random_state,
                expected_z_value,
            ) in zip(
                pending_exact_witness_repairs,
                expected_exact_witness_repairs,
                strict=True,
            )
        )
    ):
        raise ValueError(
            "same-seed helper index requires the exact-witness target gap to keep the binding/residual repair slots aligned"
        )
    if (
        seed303_support_trace_index.policy_digest
        != seed303_frontier_packet.policy_digest
        or seed303_trim_floor_support_trace_index.policy_digest
        != seed303_frontier_packet.policy_digest
        or seed303_support_trace_index.binding_design
        != seed303_frontier_packet.binding_design
        or seed303_trim_floor_support_trace_index.binding_design
        != seed303_frontier_packet.binding_design
        or seed303_support_trace_index.window_label
        != seed303_frontier_packet.window_label
        or seed303_trim_floor_support_trace_index.window_label
        != seed303_frontier_packet.window_label
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 support-trace helpers to share the frontier packet scope"
        )
    if (
        seed303_support_trace_index.target_random_state
        != seed303_frontier_packet.target_random_state
        or seed303_trim_floor_support_trace_index.target_random_state
        != seed303_frontier_packet.target_random_state
        or seed303_support_trace_index.target_seed_group
        != seed303_frontier_packet.target_seed_group
        or seed303_trim_floor_support_trace_index.target_seed_group
        != seed303_frontier_packet.target_seed_group
        or seed303_support_trace_index.live_routing
        != seed303_frontier_packet.live_routing
        or seed303_trim_floor_support_trace_index.live_routing
        != seed303_frontier_packet.live_routing
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 support-trace helpers to keep the target seed and routing aligned"
        )
    if (
        seed303_support_trace_index.binding_slot_priority_driver_signature
        != seed303_frontier_packet.support_trace_index_driver_signature
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 support-trace helper to keep the frontier packet support signature aligned"
        )
    if (
        seed303_object_flow_blocker.target_random_state
        != contract_stack_probe.binding_slot_random_state
        or abs(
            float(seed303_object_flow_blocker.target_z_value)
            - float(contract_stack_probe.binding_slot_z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 object-flow blocker to keep the binding slot aligned"
        )
    if not str(acceptance_runtime_bridge.driver_signature).strip().endswith("-open"):
        raise ValueError(
            "same-seed helper index requires the acceptance exact trim-floor runtime bridge to remain open"
        )
    if (
        seed303_acceptance_exact_trim_floor_stack.binding_slot_random_state
        != contract_stack_probe.binding_slot_random_state
        or abs(
            float(seed303_acceptance_exact_trim_floor_stack.binding_slot_z_value)
            - float(contract_stack_probe.binding_slot_z_value)
        )
        > 1e-12
        or seed303_acceptance_exact_trim_floor_stack.residual_slot_random_state
        != contract_stack_probe.residual_slot_random_state
        or abs(
            float(seed303_acceptance_exact_trim_floor_stack.residual_slot_z_value)
            - float(contract_stack_probe.residual_slot_z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 acceptance exact trim-floor stack to keep the binding/residual slots aligned"
        )
    if (
        seed303_acceptance_exact_trim_floor_stack.same_seed_admission_order
        != before_after_probe.admission_order_driver_signature
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 acceptance exact trim-floor stack to keep the admission-order signature aligned"
        )
    if (
        seed303_acceptance_exact_trim_floor_stack.acceptance_exact_trim_floor_runtime_bridge_driver_signature
        != acceptance_runtime_bridge.driver_signature
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 acceptance exact trim-floor stack to keep the acceptance runtime-bridge signature aligned"
        )
    if (
        seed303_frontier_packet.binding_slot_random_state
        != contract_stack_probe.binding_slot_random_state
        or abs(
            float(seed303_frontier_packet.binding_slot_z_value)
            - float(contract_stack_probe.binding_slot_z_value)
        )
        > 1e-12
        or seed303_frontier_packet.residual_slot_random_state
        != contract_stack_probe.residual_slot_random_state
        or abs(
            float(seed303_frontier_packet.residual_slot_z_value)
            - float(contract_stack_probe.residual_slot_z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 frontier packet to keep the binding/residual slots aligned"
        )
    if (
        seed303_frontier_packet.same_seed_admission_order
        != before_after_probe.admission_order_driver_signature
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 frontier packet to keep the admission-order signature aligned"
        )
    if (
        tuple(seed303_frontier_packet.actionable_order)
        != _PHASE7_SAME_SEED_ACTIONABLE_ORDER
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 frontier packet to keep the actionable-order ladder aligned"
        )
    if (
        seed303_frontier_packet.acceptance_exact_trim_floor_stack_driver_signature
        != seed303_acceptance_exact_trim_floor_stack.acceptance_exact_trim_floor_runtime_bridge_driver_signature
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 frontier packet to keep the acceptance exact trim-floor stack signature aligned"
        )
    if (
        seed303_frontier_packet_acceptance_stack.same_seed_admission_order
        != before_after_probe.admission_order_driver_signature
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 frontier packet acceptance stack to keep the admission-order signature aligned"
        )
    if (
        tuple(seed303_frontier_packet_acceptance_stack.actionable_order)
        != _PHASE7_SAME_SEED_ACTIONABLE_ORDER
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 frontier packet acceptance stack to keep the actionable-order ladder aligned"
        )
    if (
        seed303_frontier_packet_acceptance_stack.frontier_packet_driver_signature
        != seed303_frontier_packet.acceptance_exact_trim_floor_stack_driver_signature
    ):
        raise ValueError(
            "same-seed helper index requires the seed303 frontier packet acceptance stack to keep the frontier-packet signature aligned"
        )

    return Phase7SameSeedHelperIndexReport(
        stage_label="phase7-same-seed-helper-index",
        live_entry="trigger2-policy-spec",
        before_after_gate_status=baseline_step.driver_signature,
        validation_only=True,
        helper_tokens=_PHASE7_SAME_SEED_HELPER_TOKENS,
        acceptance_helper_tokens=_PHASE7_SAME_SEED_ACCEPTANCE_HELPER_TOKENS,
        frontier_helper_tokens=_PHASE7_SAME_SEED_FRONTIER_HELPER_TOKENS,
        lightweight_frontier_helper_tokens=(
            _PHASE7_SAME_SEED_LIGHTWEIGHT_FRONTIER_HELPER_TOKENS
        ),
        seed303_helper_tokens=_PHASE7_SAME_SEED_SEED303_HELPER_TOKENS,
        actionable_order=_PHASE7_SAME_SEED_ACTIONABLE_ORDER,
        binding_slot_random_state=contract_stack_probe.binding_slot_random_state,
        binding_slot_z_value=contract_stack_probe.binding_slot_z_value,
        residual_slot_random_state=contract_stack_probe.residual_slot_random_state,
        residual_slot_z_value=contract_stack_probe.residual_slot_z_value,
        runtime_witness_path=tuple(execution_queue_probe.runtime_witness_path),
        acceptance_readout_path=tuple(execution_queue_probe.acceptance_readout_path),
        runtime_bridge_source_guard_driver_signature=(
            runtime_bridge_source_guard.driver_signature
        ),
        runtime_bridge_after_report_frontier_packet_driver_signature=(
            runtime_bridge_frontier_packet.driver_signature
        ),
        runtime_bridge_after_report_scenario_status=(
            runtime_bridge_after_report_scenario_probe.status
        ),
        trace_payload_cache_bridge_status=trace_payload_cache_bridge_probe.status,
        exact_witness_target_gap_driver_signature=(
            exact_witness_target_gap.driver_signature
        ),
        seed303_support_trace_driver_signature=(
            seed303_support_trace_index.binding_slot_priority_driver_signature
        ),
        seed303_trim_floor_support_trace_driver_signature=(
            seed303_trim_floor_support_trace_index.inverse_pi_concentration_driver_signature
        ),
        seed303_object_flow_blocker_driver_signature=(
            seed303_object_flow_blocker.driver_signature
        ),
        seed303_acceptance_exact_trim_floor_stack_driver_signature=(
            seed303_acceptance_exact_trim_floor_stack.acceptance_exact_trim_floor_runtime_bridge_driver_signature
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_r_snapshot_readme_surface_audit(repo_root: str | Path) -> Any:
    import importlib

    readme_audit_module = importlib.import_module(
        "hddid.r_snapshot_readme_surface_audit"
    )

    return readme_audit_module.audit_r_snapshot_readme_surface(repo_root)


@lru_cache(maxsize=1)
def run_phase7_r_snapshot_inner_cv_argument_audit(repo_root: str | Path) -> Any:
    import importlib

    inner_cv_argument_audit_module = importlib.import_module(
        "hddid.r_snapshot_inner_cv_argument_audit"
    )

    return inner_cv_argument_audit_module.audit_r_snapshot_inner_cv_argument_surface(
        repo_root
    )


@lru_cache(maxsize=1)
def run_phase7_r_snapshot_inner_interface_surface_audit(repo_root: str | Path) -> Any:
    import importlib

    inner_interface_audit_module = importlib.import_module(
        "hddid.r_snapshot_inner_interface_surface_audit"
    )

    return inner_interface_audit_module.audit_r_snapshot_inner_interface_surface(
        repo_root
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_paper_dgp2_contract_audit(
    repo_root: str | Path | None = None,
) -> Any:
    import importlib

    audit_module = importlib.import_module(
        "hddid.monte_carlo_paper_dgp2_contract_audit"
    )

    return audit_module.run_phase7_monte_carlo_paper_dgp2_contract_audit(
        _repo_root_or_cwd(repo_root)
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_score_input_trace() -> Any:
    import importlib

    score_input_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_score_input_trace"
    )

    return score_input_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_score_input_trace()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_estimation_source_trace() -> Any:
    import importlib

    estimation_source_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_estimation_source_trace"
    )

    return estimation_source_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_estimation_source_trace()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_source_trace_index() -> Any:
    import importlib

    source_trace_index_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_source_trace_index"
    )

    return source_trace_index_module.run_phase7_same_seed_seed303_source_trace_index()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_rho_delta_y_input_trace() -> Any:
    import importlib

    rho_delta_y_input_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_delta_y_input_trace"
    )

    return rho_delta_y_input_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_delta_y_input_trace()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_rho_support_split_trace() -> Any:
    import importlib

    rho_support_split_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace"
    )

    return rho_support_split_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_rho_support_split_trace()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_low_pi_treated_support_trace() -> Any:
    import importlib

    low_pi_treated_support_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace"
    )

    return low_pi_treated_support_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_treated_support_trace()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_support_trace_index() -> Any:
    import importlib

    support_trace_index_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_support_trace_index"
    )

    return support_trace_index_module.run_phase7_same_seed_seed303_support_trace_index()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_trim_floor_support_trace_index() -> Any:
    import importlib

    trim_floor_support_trace_index_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_trim_floor_support_trace_index"
    )

    return trim_floor_support_trace_index_module.run_phase7_same_seed_seed303_trim_floor_support_trace_index()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_low_pi_fold3_treated_phi1_prediction_trace() -> Any:
    import importlib

    treated_phi1_prediction_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_treated_phi1_prediction_trace"
    )

    return treated_phi1_prediction_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_treated_phi1_prediction_trace()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_low_pi_fold3_trim_floor_propensity_floor_trace() -> (
    Any
):
    import importlib

    propensity_floor_trace_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace"
    )

    return propensity_floor_trace_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_propensity_floor_trace()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace() -> (
    Any
):
    import importlib

    inverse_pi_concentration_trace_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace"
    )

    return inverse_pi_concentration_trace_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_inverse_pi_concentration_trace()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_acceptance_trace_index() -> Any:
    import importlib

    acceptance_trace_index_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_acceptance_trace_index"
    )

    return acceptance_trace_index_module.run_phase7_same_seed_seed303_acceptance_trace_index()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index() -> Any:
    import importlib

    acceptance_exact_trim_floor_stack_index_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_acceptance_exact_trim_floor_stack_index"
    )

    return acceptance_exact_trim_floor_stack_index_module.run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace() -> (
    Any
):
    import importlib

    binding_slot_priority_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace"
    )

    return binding_slot_priority_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_low_pi_fold3_trim_floor_binding_slot_priority_trace()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_raw_score_component_trace() -> Any:
    import importlib

    raw_score_component_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace"
    )

    return raw_score_component_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_seed303_raw_score_component_trace()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_object_flow_blocker() -> Any:
    import importlib

    object_flow_blocker_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_seed303_object_flow_blocker"
    )

    return object_flow_blocker_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_seed303_object_flow_blocker()


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_object_flow_runtime_evidence() -> Any:
    import importlib

    runtime_evidence_module = importlib.import_module(
        "hddid.monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence"
    )

    return runtime_evidence_module.run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_seed303_object_flow_runtime_evidence()


_PHASE7_SAME_SEED_SEED303_FRONTIER_PACKET_HELPER_TOKENS = (
    "run_phase7_same_seed_seed303_source_trace_index",
    "run_phase7_same_seed_seed303_object_flow_runtime_evidence",
    "run_phase7_same_seed_seed303_raw_score_component_trace",
    "run_phase7_same_seed_seed303_support_trace_index",
    "run_phase7_same_seed_seed303_acceptance_trace_index",
    "run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index",
)


def _format_phase7_same_seed_ninths(value: float) -> str:
    numerator = int(round(float(value) * 9.0))
    return f"{numerator}/9 = {float(value):.3f}"


@dataclass(slots=True)
class Phase7SameSeedSeed303FrontierPacketReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    target_random_state: int
    target_seed_group: str
    live_routing: str
    validation_surface_status: str
    helper_tokens: tuple[str, ...]
    actionable_order: tuple[str, ...]
    same_seed_admission_order: str
    source_trace_index_driver_signature: str
    object_flow_runtime_driver_signature: str
    raw_score_component_driver_signature: str
    support_trace_index_driver_signature: str
    acceptance_trace_index_driver_signature: str
    acceptance_exact_trim_floor_stack_driver_signature: str
    acceptance_shortfall: float
    current_witness_floor: float
    required_min_witness_floor: float
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_value: float
    residual_slot_random_state: int
    residual_slot_seed_group: str
    residual_slot_z_value: float
    binding_replication_seed: int
    center_truth: float
    center_estimate: float
    center_error_to_half_interval_ratio: float
    vf_cross_entry: float
    canonical_seed303_frontier_packet_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.focus_random_states = tuple(
            int(value) for value in self.focus_random_states
        )
        self.target_random_state = int(self.target_random_state)
        self.target_seed_group = str(self.target_seed_group).strip().lower()
        self.live_routing = str(self.live_routing).strip()
        self.validation_surface_status = str(self.validation_surface_status).strip()
        self.helper_tokens = tuple(str(item).strip() for item in self.helper_tokens)
        self.actionable_order = tuple(
            str(item).strip() for item in self.actionable_order
        )
        self.same_seed_admission_order = str(self.same_seed_admission_order).strip()
        self.source_trace_index_driver_signature = str(
            self.source_trace_index_driver_signature
        ).strip()
        self.object_flow_runtime_driver_signature = str(
            self.object_flow_runtime_driver_signature
        ).strip()
        self.raw_score_component_driver_signature = str(
            self.raw_score_component_driver_signature
        ).strip()
        self.support_trace_index_driver_signature = str(
            self.support_trace_index_driver_signature
        ).strip()
        self.acceptance_trace_index_driver_signature = str(
            self.acceptance_trace_index_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_stack_driver_signature = str(
            self.acceptance_exact_trim_floor_stack_driver_signature
        ).strip()
        self.acceptance_shortfall = float(self.acceptance_shortfall)
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_seed_group = str(self.binding_slot_seed_group).strip().lower()
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_seed_group = str(self.residual_slot_seed_group).strip().lower()
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.center_truth = float(self.center_truth)
        self.center_estimate = float(self.center_estimate)
        self.center_error_to_half_interval_ratio = float(
            self.center_error_to_half_interval_ratio
        )
        self.vf_cross_entry = float(self.vf_cross_entry)
        self.canonical_seed303_frontier_packet_digest = tuple(
            str(item).rstrip() for item in self.canonical_seed303_frontier_packet_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "focus_random_states": list(self.focus_random_states),
            "target_random_state": self.target_random_state,
            "target_seed_group": self.target_seed_group,
            "live_routing": self.live_routing,
            "validation_surface_status": self.validation_surface_status,
            "helper_tokens": list(self.helper_tokens),
            "actionable_order": list(self.actionable_order),
            "same_seed_admission_order": self.same_seed_admission_order,
            "source_trace_index_driver_signature": self.source_trace_index_driver_signature,
            "object_flow_runtime_driver_signature": self.object_flow_runtime_driver_signature,
            "raw_score_component_driver_signature": self.raw_score_component_driver_signature,
            "support_trace_index_driver_signature": self.support_trace_index_driver_signature,
            "acceptance_trace_index_driver_signature": self.acceptance_trace_index_driver_signature,
            "acceptance_exact_trim_floor_stack_driver_signature": self.acceptance_exact_trim_floor_stack_driver_signature,
            "acceptance_shortfall": self.acceptance_shortfall,
            "current_witness_floor": self.current_witness_floor,
            "required_min_witness_floor": self.required_min_witness_floor,
            "binding_slot_random_state": self.binding_slot_random_state,
            "binding_slot_seed_group": self.binding_slot_seed_group,
            "binding_slot_z_value": self.binding_slot_z_value,
            "residual_slot_random_state": self.residual_slot_random_state,
            "residual_slot_seed_group": self.residual_slot_seed_group,
            "residual_slot_z_value": self.residual_slot_z_value,
            "binding_replication_seed": self.binding_replication_seed,
            "center_truth": self.center_truth,
            "center_estimate": self.center_estimate,
            "center_error_to_half_interval_ratio": self.center_error_to_half_interval_ratio,
            "vf_cross_entry": self.vf_cross_entry,
            "canonical_seed303_frontier_packet_digest": list(
                self.canonical_seed303_frontier_packet_digest
            ),
        }


_PHASE7_SAME_SEED_SEED303_FRONTIER_PACKET_ACCEPTANCE_STACK_HELPER_TOKENS = (
    "run_phase7_same_seed_seed303_frontier_packet",
    "run_phase7_same_seed_admission_order",
    "run_phase7_same_seed_first_hop_acceptance_raw_score_component_contract",
    "run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_runtime_bridge",
)

_PHASE7_SAME_SEED_OBSERVED_RERUN_EVIDENCE_PACKET_HELPER_TOKENS = (
    "run_phase7_same_seed_before_after_acceptance_probe",
    "run_phase7_same_seed_admission_order",
    "run_phase7_same_seed_runtime_bridge_after_report_frontier_packet",
    "run_phase7_same_seed_seed303_frontier_packet_acceptance_stack",
)


@dataclass(slots=True)
class Phase7SameSeedSeed303FrontierPacketAcceptanceStackReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    same_seed_random_states: tuple[int, ...]
    live_routing: str
    validation_surface_status: str
    helper_tokens: tuple[str, ...]
    actionable_order: tuple[str, ...]
    same_seed_admission_order: str
    admission_order_driver_signature: str
    frontier_packet_driver_signature: str
    acceptance_raw_score_component_driver_signature: str
    acceptance_exact_trim_floor_runtime_bridge_driver_signature: str
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_value: float
    residual_slot_random_state: int
    residual_slot_seed_group: str
    residual_slot_z_value: float
    acceptance_shortfall: float
    current_witness_floor: float
    required_min_witness_floor: float
    binding_replication_seed: int
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    canonical_seed303_frontier_packet_acceptance_stack_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.focus_random_states = tuple(
            int(value) for value in self.focus_random_states
        )
        self.same_seed_random_states = tuple(
            int(value) for value in self.same_seed_random_states
        )
        self.live_routing = str(self.live_routing).strip()
        self.validation_surface_status = str(self.validation_surface_status).strip()
        self.helper_tokens = tuple(str(item).strip() for item in self.helper_tokens)
        self.actionable_order = tuple(
            str(item).strip() for item in self.actionable_order
        )
        self.same_seed_admission_order = str(self.same_seed_admission_order).strip()
        self.admission_order_driver_signature = str(
            self.admission_order_driver_signature
        ).strip()
        self.frontier_packet_driver_signature = str(
            self.frontier_packet_driver_signature
        ).strip()
        self.acceptance_raw_score_component_driver_signature = str(
            self.acceptance_raw_score_component_driver_signature
        ).strip()
        self.acceptance_exact_trim_floor_runtime_bridge_driver_signature = str(
            self.acceptance_exact_trim_floor_runtime_bridge_driver_signature
        ).strip()
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_seed_group = str(self.binding_slot_seed_group).strip().lower()
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_seed_group = str(self.residual_slot_seed_group).strip().lower()
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.acceptance_shortfall = float(self.acceptance_shortfall)
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.canonical_seed303_frontier_packet_acceptance_stack_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_seed303_frontier_packet_acceptance_stack_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "focus_random_states": list(self.focus_random_states),
            "same_seed_random_states": list(self.same_seed_random_states),
            "live_routing": self.live_routing,
            "validation_surface_status": self.validation_surface_status,
            "helper_tokens": list(self.helper_tokens),
            "actionable_order": list(self.actionable_order),
            "same_seed_admission_order": self.same_seed_admission_order,
            "admission_order_driver_signature": self.admission_order_driver_signature,
            "frontier_packet_driver_signature": self.frontier_packet_driver_signature,
            "acceptance_raw_score_component_driver_signature": (
                self.acceptance_raw_score_component_driver_signature
            ),
            "acceptance_exact_trim_floor_runtime_bridge_driver_signature": (
                self.acceptance_exact_trim_floor_runtime_bridge_driver_signature
            ),
            "binding_slot_random_state": self.binding_slot_random_state,
            "binding_slot_seed_group": self.binding_slot_seed_group,
            "binding_slot_z_value": self.binding_slot_z_value,
            "residual_slot_random_state": self.residual_slot_random_state,
            "residual_slot_seed_group": self.residual_slot_seed_group,
            "residual_slot_z_value": self.residual_slot_z_value,
            "acceptance_shortfall": self.acceptance_shortfall,
            "current_witness_floor": self.current_witness_floor,
            "required_min_witness_floor": self.required_min_witness_floor,
            "binding_replication_seed": self.binding_replication_seed,
            "runtime_witness_path": list(self.runtime_witness_path),
            "acceptance_readout_path": list(self.acceptance_readout_path),
            "canonical_seed303_frontier_packet_acceptance_stack_digest": list(
                self.canonical_seed303_frontier_packet_acceptance_stack_digest
            ),
        }


@dataclass(slots=True)
class Phase7SameSeedObservedRerunEvidencePacketReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    focus_random_states: tuple[int, ...]
    canonical_seed_order: tuple[int, ...]
    live_routing: str
    validation_surface_status: str
    helper_tokens: tuple[str, ...]
    actionable_order: tuple[str, ...]
    current_before_after_driver_signature: str
    target_before_after_driver_signature: str
    runtime_bridge_frontier_driver_signature: str
    seed303_frontier_packet_acceptance_stack_driver_signature: str
    acceptance_shortfall: float
    current_witness_floor: float
    required_min_witness_floor: float
    binding_slot_random_state: int
    binding_slot_seed_group: str
    binding_slot_z_value: float
    residual_slot_random_state: int
    residual_slot_seed_group: str
    residual_slot_z_value: float
    binding_replication_seed: int
    runtime_witness_path: tuple[str, ...]
    acceptance_readout_path: tuple[str, ...]
    canonical_observed_rerun_evidence_packet_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.focus_random_states = tuple(
            int(value) for value in self.focus_random_states
        )
        self.canonical_seed_order = tuple(
            int(value) for value in self.canonical_seed_order
        )
        self.live_routing = str(self.live_routing).strip()
        self.validation_surface_status = str(self.validation_surface_status).strip()
        self.helper_tokens = tuple(str(item).strip() for item in self.helper_tokens)
        self.actionable_order = tuple(
            str(item).strip() for item in self.actionable_order
        )
        self.current_before_after_driver_signature = str(
            self.current_before_after_driver_signature
        ).strip()
        self.target_before_after_driver_signature = str(
            self.target_before_after_driver_signature
        ).strip()
        self.runtime_bridge_frontier_driver_signature = str(
            self.runtime_bridge_frontier_driver_signature
        ).strip()
        self.seed303_frontier_packet_acceptance_stack_driver_signature = str(
            self.seed303_frontier_packet_acceptance_stack_driver_signature
        ).strip()
        self.acceptance_shortfall = float(self.acceptance_shortfall)
        self.current_witness_floor = float(self.current_witness_floor)
        self.required_min_witness_floor = float(self.required_min_witness_floor)
        self.binding_slot_random_state = int(self.binding_slot_random_state)
        self.binding_slot_seed_group = str(self.binding_slot_seed_group).strip().lower()
        self.binding_slot_z_value = float(self.binding_slot_z_value)
        self.residual_slot_random_state = int(self.residual_slot_random_state)
        self.residual_slot_seed_group = str(self.residual_slot_seed_group).strip().lower()
        self.residual_slot_z_value = float(self.residual_slot_z_value)
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.runtime_witness_path = tuple(
            str(item).strip() for item in self.runtime_witness_path
        )
        self.acceptance_readout_path = tuple(
            str(item).strip() for item in self.acceptance_readout_path
        )
        self.canonical_observed_rerun_evidence_packet_digest = tuple(
            str(item).rstrip()
            for item in self.canonical_observed_rerun_evidence_packet_digest
        )

    @property
    def status(self) -> str:
        return self.validation_surface_status

    @property
    def frontier_driver_signature(self) -> str:
        return self.runtime_bridge_frontier_driver_signature

    @property
    def seed303_acceptance_stack_driver_signature(self) -> str:
        return self.seed303_frontier_packet_acceptance_stack_driver_signature

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "focus_random_states": list(self.focus_random_states),
            "canonical_seed_order": list(self.canonical_seed_order),
            "live_routing": self.live_routing,
            "validation_surface_status": self.validation_surface_status,
            "helper_tokens": list(self.helper_tokens),
            "actionable_order": list(self.actionable_order),
            "current_before_after_driver_signature": (
                self.current_before_after_driver_signature
            ),
            "target_before_after_driver_signature": (
                self.target_before_after_driver_signature
            ),
            "runtime_bridge_frontier_driver_signature": (
                self.runtime_bridge_frontier_driver_signature
            ),
            "seed303_frontier_packet_acceptance_stack_driver_signature": (
                self.seed303_frontier_packet_acceptance_stack_driver_signature
            ),
            "acceptance_shortfall": self.acceptance_shortfall,
            "current_witness_floor": self.current_witness_floor,
            "required_min_witness_floor": self.required_min_witness_floor,
            "binding_slot_random_state": self.binding_slot_random_state,
            "binding_slot_seed_group": self.binding_slot_seed_group,
            "binding_slot_z_value": self.binding_slot_z_value,
            "residual_slot_random_state": self.residual_slot_random_state,
            "residual_slot_seed_group": self.residual_slot_seed_group,
            "residual_slot_z_value": self.residual_slot_z_value,
            "binding_replication_seed": self.binding_replication_seed,
            "runtime_witness_path": list(self.runtime_witness_path),
            "acceptance_readout_path": list(self.acceptance_readout_path),
            "canonical_observed_rerun_evidence_packet_digest": list(
                self.canonical_observed_rerun_evidence_packet_digest
            ),
        }


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_frontier_packet() -> (
    Phase7SameSeedSeed303FrontierPacketReport
):
    source_trace_index = run_phase7_same_seed_seed303_source_trace_index()
    object_flow_runtime_evidence = run_phase7_same_seed_seed303_object_flow_runtime_evidence()
    raw_score_component_trace = run_phase7_same_seed_seed303_raw_score_component_trace()
    support_trace_index = run_phase7_same_seed_seed303_support_trace_index()
    acceptance_trace_index = run_phase7_same_seed_seed303_acceptance_trace_index()
    acceptance_exact_trim_floor_stack_index = (
        run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index()
    )

    if source_trace_index.policy_digest != support_trace_index.policy_digest:
        raise ValueError(
            "seed303 frontier packet requires source and support traces to share the policy digest"
        )
    if source_trace_index.policy_digest != acceptance_trace_index.policy_digest:
        raise ValueError(
            "seed303 frontier packet requires acceptance trace to share the policy digest"
        )
    if source_trace_index.binding_design != support_trace_index.binding_design:
        raise ValueError(
            "seed303 frontier packet requires source and support traces to share the binding design"
        )
    if source_trace_index.binding_design != acceptance_trace_index.binding_design:
        raise ValueError(
            "seed303 frontier packet requires acceptance trace to share the binding design"
        )
    if source_trace_index.window_label != support_trace_index.window_label:
        raise ValueError(
            "seed303 frontier packet requires source and support traces to share the window label"
        )
    if source_trace_index.window_label != acceptance_trace_index.window_label:
        raise ValueError(
            "seed303 frontier packet requires acceptance trace to share the window label"
        )
    if source_trace_index.target_random_state != acceptance_trace_index.target_random_state:
        raise ValueError(
            "seed303 frontier packet requires the source and acceptance traces to target the same seed"
        )
    if source_trace_index.target_seed_group != acceptance_trace_index.target_seed_group:
        raise ValueError(
            "seed303 frontier packet requires the source and acceptance traces to target the same seed group"
        )
    if acceptance_trace_index.same_seed_admission_order != (
        acceptance_exact_trim_floor_stack_index.same_seed_admission_order
    ):
        raise ValueError(
            "seed303 frontier packet requires acceptance traces to share the same-seed admission order"
        )
    if (
        acceptance_trace_index.binding_slot_random_state
        != acceptance_exact_trim_floor_stack_index.binding_slot_random_state
        or abs(
            float(acceptance_trace_index.binding_slot_z_value)
            - float(acceptance_exact_trim_floor_stack_index.binding_slot_z_value)
        )
        > 1e-12
        or acceptance_trace_index.residual_slot_random_state
        != acceptance_exact_trim_floor_stack_index.residual_slot_random_state
        or abs(
            float(acceptance_trace_index.residual_slot_z_value)
            - float(acceptance_exact_trim_floor_stack_index.residual_slot_z_value)
        )
        > 1e-12
    ):
        raise ValueError(
            "seed303 frontier packet requires acceptance traces to keep the binding/residual slots aligned"
        )
    if (
        acceptance_trace_index.acceptance_exact_trim_floor_runtime_bridge_driver_signature
        != acceptance_exact_trim_floor_stack_index.acceptance_exact_trim_floor_runtime_bridge_driver_signature
    ):
        raise ValueError(
            "seed303 frontier packet requires acceptance traces to share the exact trim-floor runtime bridge signature"
        )
    if (
        acceptance_trace_index.binding_replication_seed
        != object_flow_runtime_evidence.focus_seed_runtime_snapshot.replication_seed
    ):
        raise ValueError(
            "seed303 frontier packet requires the acceptance trace and object-flow runtime evidence to share the replication seed"
        )

    canonical_digest = (
        "- "
        f"`{source_trace_index.rho_delta_y_input_driver_signature}` keeps the current seed `303` witness miss source-led, "
        f"while `{raw_score_component_trace.driver_signature}` keeps the raw-score overshoot machine-readable at "
        f"`rho_hat * delta_y = {raw_score_component_trace.target_rho_delta_y_center_projection:.3f}`, "
        f"`{raw_score_component_trace.target_rho_one_minus_pi_phi1_center_projection:.3f}`, and "
        f"`{raw_score_component_trace.target_rho_pi_phi0_center_projection:.3f}` before any residual seed `707` spend",
        "- "
        f"`{object_flow_runtime_evidence.driver_signature}` keeps the same packet anchored on the acceptance-facing runtime readout "
        "`omega_f_hat[2,2] -> v_f_hat[2,1] -> bar_f_at_z0[1]` "
        f"with truth `{acceptance_trace_index.center_truth:.3f}`, estimate `{acceptance_trace_index.center_estimate:.3f}`, "
        f"ratio `{acceptance_trace_index.center_error_to_half_interval_ratio:.3f}x`, cross-entry `{acceptance_trace_index.vf_cross_entry:.3f}`, "
        f"and replication seed `{acceptance_trace_index.binding_replication_seed}`",
        "- "
        f"`{support_trace_index.binding_slot_priority_driver_signature}` keeps the same packet on the fold-`3` trim-floor support ladder, "
        f"with seed `303` / `z = {acceptance_trace_index.binding_slot_z_value:.2f}` still ahead of residual seed `707` / "
        f"`z = {acceptance_trace_index.residual_slot_z_value:.2f}` at live acceptance shortfall "
        f"`{_format_phase7_same_seed_ninths(acceptance_trace_index.acceptance_shortfall)}` versus "
        "`0/9 = 0.000`",
        "- "
        f"`{acceptance_trace_index.acceptance_exact_trim_floor_runtime_bridge_driver_signature}` stays aligned across "
        "`run_phase7_same_seed_seed303_acceptance_trace_index()` and "
        "`run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index()`, "
        "so the current bounded repair still runs through the exact trim-floor runtime bridge before spending seed `707`",
        "- current Trigger 2 implication: keep `run_phase7_same_seed_seed303_frontier_packet()` validation-only as "
        "`validation-only seed303 frontier packet`; it supports `trigger2-policy-spec` and `same-seed-admission-order`, "
        "but stays off live routing surfaces while the next bounded repair continues through the acceptance raw-score component contract "
        "and exact trim-floor runtime bridge",
    )

    return Phase7SameSeedSeed303FrontierPacketReport(
        stage_label="phase7-same-seed-seed303-frontier-packet",
        policy_digest=source_trace_index.policy_digest,
        binding_design=source_trace_index.binding_design,
        window_label=source_trace_index.window_label,
        focus_random_states=source_trace_index.focus_random_states,
        target_random_state=source_trace_index.target_random_state,
        target_seed_group=source_trace_index.target_seed_group,
        live_routing=acceptance_trace_index.live_routing,
        validation_surface_status="validation-only seed303 frontier packet",
        helper_tokens=_PHASE7_SAME_SEED_SEED303_FRONTIER_PACKET_HELPER_TOKENS,
        actionable_order=_PHASE7_SAME_SEED_ACTIONABLE_ORDER,
        same_seed_admission_order=acceptance_trace_index.same_seed_admission_order,
        source_trace_index_driver_signature=source_trace_index.rho_delta_y_input_driver_signature,
        object_flow_runtime_driver_signature=object_flow_runtime_evidence.driver_signature,
        raw_score_component_driver_signature=raw_score_component_trace.driver_signature,
        support_trace_index_driver_signature=support_trace_index.binding_slot_priority_driver_signature,
        acceptance_trace_index_driver_signature=(
            acceptance_trace_index.acceptance_exact_trim_floor_runtime_bridge_driver_signature
        ),
        acceptance_exact_trim_floor_stack_driver_signature=(
            acceptance_exact_trim_floor_stack_index.acceptance_exact_trim_floor_runtime_bridge_driver_signature
        ),
        acceptance_shortfall=acceptance_trace_index.acceptance_shortfall,
        current_witness_floor=acceptance_trace_index.current_witness_floor,
        required_min_witness_floor=acceptance_trace_index.required_min_witness_floor,
        binding_slot_random_state=acceptance_trace_index.binding_slot_random_state,
        binding_slot_seed_group=acceptance_trace_index.binding_slot_seed_group,
        binding_slot_z_value=acceptance_trace_index.binding_slot_z_value,
        residual_slot_random_state=acceptance_trace_index.residual_slot_random_state,
        residual_slot_seed_group=acceptance_trace_index.residual_slot_seed_group,
        residual_slot_z_value=acceptance_trace_index.residual_slot_z_value,
        binding_replication_seed=acceptance_trace_index.binding_replication_seed,
        center_truth=acceptance_trace_index.center_truth,
        center_estimate=acceptance_trace_index.center_estimate,
        center_error_to_half_interval_ratio=(
            acceptance_trace_index.center_error_to_half_interval_ratio
        ),
        vf_cross_entry=acceptance_trace_index.vf_cross_entry,
        canonical_seed303_frontier_packet_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_seed303_frontier_packet_acceptance_stack() -> (
    Phase7SameSeedSeed303FrontierPacketAcceptanceStackReport
):
    frontier_packet = run_phase7_same_seed_seed303_frontier_packet()
    admission_order = run_phase7_same_seed_admission_order()
    acceptance_raw_score_component = (
        run_phase7_same_seed_first_hop_acceptance_raw_score_component_contract()
    )
    acceptance_exact_trim_floor_runtime_bridge = (
        run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_runtime_bridge()
    )

    if frontier_packet.policy_digest != admission_order.policy_digest:
        raise ValueError(
            "seed303 frontier packet acceptance stack requires the frontier packet and admission order to share the policy digest"
        )
    if frontier_packet.policy_digest != acceptance_raw_score_component.policy_digest:
        raise ValueError(
            "seed303 frontier packet acceptance stack requires the frontier packet and raw-score contract to share the policy digest"
        )
    if (
        frontier_packet.policy_digest
        != acceptance_exact_trim_floor_runtime_bridge.policy_digest
    ):
        raise ValueError(
            "seed303 frontier packet acceptance stack requires the frontier packet and exact trim-floor runtime bridge to share the policy digest"
        )
    if frontier_packet.binding_design != admission_order.binding_design:
        raise ValueError(
            "seed303 frontier packet acceptance stack requires the frontier packet and admission order to share the binding design"
        )
    if (
        frontier_packet.binding_design
        != acceptance_raw_score_component.binding_design
        or frontier_packet.binding_design
        != acceptance_exact_trim_floor_runtime_bridge.binding_design
    ):
        raise ValueError(
            "seed303 frontier packet acceptance stack requires all acceptance-stack helpers to share the binding design"
        )
    if frontier_packet.window_label != admission_order.window_label:
        raise ValueError(
            "seed303 frontier packet acceptance stack requires the frontier packet and admission order to share the window label"
        )
    if (
        frontier_packet.window_label != acceptance_raw_score_component.window_label
        or frontier_packet.window_label
        != acceptance_exact_trim_floor_runtime_bridge.window_label
    ):
        raise ValueError(
            "seed303 frontier packet acceptance stack requires all acceptance-stack helpers to share the window label"
        )
    if frontier_packet.same_seed_admission_order != admission_order.driver_signature:
        raise ValueError(
            "seed303 frontier packet acceptance stack requires the frontier packet and admission order to share the same-seed admission driver"
        )
    if (
        frontier_packet.raw_score_component_driver_signature
        != acceptance_raw_score_component.raw_score_component_driver_signature
    ):
        raise ValueError(
            "seed303 frontier packet acceptance stack requires the frontier packet and raw-score contract to share the raw-score driver"
        )
    if (
        frontier_packet.acceptance_exact_trim_floor_stack_driver_signature
        != acceptance_exact_trim_floor_runtime_bridge.driver_signature
    ):
        raise ValueError(
            "seed303 frontier packet acceptance stack requires the frontier packet and exact trim-floor runtime bridge to share the runtime-bridge driver"
        )
    if (
        frontier_packet.binding_slot_random_state
        != acceptance_raw_score_component.binding_slot_random_state
        or abs(
            frontier_packet.binding_slot_z_value
            - acceptance_raw_score_component.binding_slot_z_value
        )
        > 1e-12
        or frontier_packet.binding_slot_random_state
        != acceptance_exact_trim_floor_runtime_bridge.binding_slot_random_state
        or abs(
            frontier_packet.binding_slot_z_value
            - acceptance_exact_trim_floor_runtime_bridge.binding_slot_z_value
        )
        > 1e-12
    ):
        raise ValueError(
            "seed303 frontier packet acceptance stack requires the same binding slot across the frontier packet and acceptance stack"
        )
    if (
        frontier_packet.residual_slot_random_state
        != acceptance_raw_score_component.residual_slot_random_state
        or abs(
            frontier_packet.residual_slot_z_value
            - acceptance_raw_score_component.residual_slot_z_value
        )
        > 1e-12
        or frontier_packet.residual_slot_random_state
        != acceptance_exact_trim_floor_runtime_bridge.residual_slot_random_state
        or abs(
            frontier_packet.residual_slot_z_value
            - acceptance_exact_trim_floor_runtime_bridge.residual_slot_z_value
        )
        > 1e-12
    ):
        raise ValueError(
            "seed303 frontier packet acceptance stack requires the same residual slot across the frontier packet and acceptance stack"
        )
    if (
        frontier_packet.binding_replication_seed
        != acceptance_raw_score_component.binding_replication_seed
        or frontier_packet.binding_replication_seed
        != acceptance_exact_trim_floor_runtime_bridge.binding_replication_seed
    ):
        raise ValueError(
            "seed303 frontier packet acceptance stack requires the same replication seed across the frontier packet and acceptance stack"
        )

    canonical_digest = (
        "- `run_phase7_same_seed_seed303_frontier_packet()` keeps the seed `303` witness packet machine-readable on the narrower frontier read, while `run_phase7_same_seed_admission_order()` keeps seed `303` / `z = 0.15` ahead of residual seed `707` / `z = 0.25` on the same canonical replay order",
        f"- `run_phase7_same_seed_first_hop_acceptance_raw_score_component_contract()` keeps the open acceptance shortfall source-led at `rho_hat * delta_y = {acceptance_raw_score_component.target_rho_delta_y_center_projection:.3f}`, with the same actionable packet still owned by seed `303` before any residual spend",
        "- `run_phase7_same_seed_first_hop_acceptance_exact_trim_floor_runtime_bridge()` keeps the runtime path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)` and acceptance-facing readout `omega_f_hat[2,2] -> v_f_hat[2,1] -> bar_f_at_z0[1]` aligned on replication seed "
        f"`{acceptance_exact_trim_floor_runtime_bridge.binding_replication_seed}`, so the current bounded repair still runs through the exact trim-floor runtime bridge",
        "- current Trigger 2 implication: keep `run_phase7_same_seed_seed303_frontier_packet_acceptance_stack()` validation-only as `validation-only seed303 frontier packet acceptance stack`; it supports `trigger2-policy-spec`, `same-seed-admission-order`, and the acceptance stack rerun carry, but it does not promote seed `303` onto live routing above the closeout-state policy surface",
    )

    return Phase7SameSeedSeed303FrontierPacketAcceptanceStackReport(
        stage_label="phase7-same-seed-seed303-frontier-packet-acceptance-stack",
        policy_digest=frontier_packet.policy_digest,
        binding_design=frontier_packet.binding_design,
        window_label=frontier_packet.window_label,
        focus_random_states=frontier_packet.focus_random_states,
        same_seed_random_states=admission_order.same_seed_random_states,
        live_routing=frontier_packet.live_routing,
        validation_surface_status=(
            "validation-only seed303 frontier packet acceptance stack"
        ),
        helper_tokens=_PHASE7_SAME_SEED_SEED303_FRONTIER_PACKET_ACCEPTANCE_STACK_HELPER_TOKENS,
        actionable_order=frontier_packet.actionable_order,
        same_seed_admission_order=frontier_packet.same_seed_admission_order,
        admission_order_driver_signature=admission_order.driver_signature,
        frontier_packet_driver_signature=(
            frontier_packet.acceptance_exact_trim_floor_stack_driver_signature
        ),
        acceptance_raw_score_component_driver_signature=(
            acceptance_raw_score_component.raw_score_component_driver_signature
        ),
        acceptance_exact_trim_floor_runtime_bridge_driver_signature=(
            acceptance_exact_trim_floor_runtime_bridge.driver_signature
        ),
        binding_slot_random_state=frontier_packet.binding_slot_random_state,
        binding_slot_seed_group=frontier_packet.binding_slot_seed_group,
        binding_slot_z_value=frontier_packet.binding_slot_z_value,
        residual_slot_random_state=frontier_packet.residual_slot_random_state,
        residual_slot_seed_group=frontier_packet.residual_slot_seed_group,
        residual_slot_z_value=frontier_packet.residual_slot_z_value,
        acceptance_shortfall=frontier_packet.acceptance_shortfall,
        current_witness_floor=frontier_packet.current_witness_floor,
        required_min_witness_floor=frontier_packet.required_min_witness_floor,
        binding_replication_seed=frontier_packet.binding_replication_seed,
        runtime_witness_path=(
            acceptance_exact_trim_floor_runtime_bridge.runtime_witness_path
        ),
        acceptance_readout_path=(
            acceptance_exact_trim_floor_runtime_bridge.acceptance_readout_path
        ),
        canonical_seed303_frontier_packet_acceptance_stack_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_same_seed_binding_slot_candidate_guard():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_candidate_guard as _run_binding_slot_candidate_guard,
    )

    return _run_binding_slot_candidate_guard()


@lru_cache(maxsize=1)
def run_phase7_same_seed_residual_slot_candidate_guard():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_candidate_guard import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_candidate_guard as _run_residual_slot_candidate_guard,
    )

    return _run_residual_slot_candidate_guard()


@lru_cache(maxsize=1)
def run_phase7_same_seed_observed_rerun_live_gap_packet():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_observed_rerun_live_gap_packet as _run_live_gap_packet,
    )

    return _run_live_gap_packet()


@lru_cache(maxsize=1)
def run_phase7_same_seed_binding_slot_progress_profile():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_binding_slot_progress_profile as _run_binding_slot_progress_profile,
    )

    return _run_binding_slot_progress_profile()


@lru_cache(maxsize=1)
def run_phase7_same_seed_residual_slot_completion_profile():
    from .monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile import (
        run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_preserve_left_support_exact_witness_residual_slot_completion_profile as _run_residual_slot_completion_profile,
    )

    return _run_residual_slot_completion_profile()


@lru_cache(maxsize=1)
def run_phase7_same_seed_observed_rerun_evidence_packet() -> (
    Phase7SameSeedObservedRerunEvidencePacketReport
):
    before_after_probe = run_phase7_same_seed_before_after_acceptance_probe()
    admission_order = run_phase7_same_seed_admission_order()
    runtime_bridge_frontier_packet = (
        run_phase7_same_seed_runtime_bridge_after_report_frontier_packet()
    )
    seed303_frontier_packet_acceptance_stack = (
        run_phase7_same_seed_seed303_frontier_packet_acceptance_stack()
    )

    if before_after_probe.canonical_seed_order != admission_order.same_seed_random_states:
        raise ValueError(
            "same-seed observed rerun evidence packet requires the before/after probe and admission order to share the canonical seed order"
        )
    if (
        before_after_probe.canonical_seed_order
        != seed303_frontier_packet_acceptance_stack.same_seed_random_states
    ):
        raise ValueError(
            "same-seed observed rerun evidence packet requires the seed303 acceptance stack to share the canonical seed order"
        )
    if (
        seed303_frontier_packet_acceptance_stack.same_seed_admission_order
        != admission_order.driver_signature
        or runtime_bridge_frontier_packet.admission_order_driver_signature
        != admission_order.driver_signature
    ):
        raise ValueError(
            "same-seed observed rerun evidence packet requires one shared same-seed admission-order driver"
        )
    if (
        seed303_frontier_packet_acceptance_stack.binding_slot_random_state
        != runtime_bridge_frontier_packet.binding_slot_random_state
        or abs(
            seed303_frontier_packet_acceptance_stack.binding_slot_z_value
            - runtime_bridge_frontier_packet.binding_slot_z_value
        )
        > 1e-12
        or seed303_frontier_packet_acceptance_stack.residual_slot_random_state
        != runtime_bridge_frontier_packet.residual_slot_random_state
        or abs(
            seed303_frontier_packet_acceptance_stack.residual_slot_z_value
            - runtime_bridge_frontier_packet.residual_slot_z_value
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed observed rerun evidence packet requires the runtime-bridge frontier and seed303 acceptance stack to keep one binding/residual slot ladder"
        )
    if (
        abs(
            seed303_frontier_packet_acceptance_stack.acceptance_shortfall
            - runtime_bridge_frontier_packet.acceptance_shortfall
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed observed rerun evidence packet requires the runtime-bridge frontier and seed303 acceptance stack to share the acceptance shortfall"
        )
    if (
        abs(
            before_after_probe.scenarios[0].witness_floor
            - seed303_frontier_packet_acceptance_stack.current_witness_floor
        )
        > 1e-12
        or abs(
            before_after_probe.required_min_witness_floor
            - seed303_frontier_packet_acceptance_stack.required_min_witness_floor
        )
        > 1e-12
    ):
        raise ValueError(
            "same-seed observed rerun evidence packet requires the before/after probe and seed303 acceptance stack to share the witness-floor gate"
        )

    canonical_digest = (
        "- `run_phase7_same_seed_before_after_acceptance_probe()` keeps the exact same-seed replay gate honest at `7/9 -> 8/9`, so fresh evidence must still clear `same-seed-before-after-acceptance-not-yet-satisfied` before the packet can close",
        "- `run_phase7_same_seed_admission_order()` keeps seed `303` / `z = 0.15` ahead of residual seed `707` / `z = 0.25` on the canonical replay order `(101, 202, 303, 404, 505, 606, 707, 808)`",
        "- `run_phase7_same_seed_runtime_bridge_after_report_frontier_packet()` keeps the current frontier executable at `baseline-no-after-report`, `binding-only-payload-rejected`, and `completion-after-report-accepted` while the live gap still stays on seed `303` before seed `707`",
        "- `run_phase7_same_seed_seed303_frontier_packet_acceptance_stack()` keeps the actionable packet source-led through `acceptance raw-score component contract` and `acceptance exact trim-floor runtime bridge`, with replication seed `883193502` and runtime path `omega_f_hat[2,2] -> v_f_hat[2,2] -> covariance(0.25, 0.15)`",
        "- current Trigger 2 implication: keep `run_phase7_same_seed_observed_rerun_evidence_packet()` validation-only as `validation-only same-seed observed rerun evidence packet`; it shortens the current fresh-rerun evidence read for `trigger2-policy-spec`, but it does not promote seed `303` above the closeout-state live routing surface",
    )

    return Phase7SameSeedObservedRerunEvidencePacketReport(
        stage_label="phase7-same-seed-observed-rerun-evidence-packet",
        policy_digest=seed303_frontier_packet_acceptance_stack.policy_digest,
        binding_design=seed303_frontier_packet_acceptance_stack.binding_design,
        window_label=seed303_frontier_packet_acceptance_stack.window_label,
        focus_random_states=seed303_frontier_packet_acceptance_stack.focus_random_states,
        canonical_seed_order=before_after_probe.canonical_seed_order,
        live_routing=seed303_frontier_packet_acceptance_stack.live_routing,
        validation_surface_status="validation-only same-seed observed rerun evidence packet",
        helper_tokens=_PHASE7_SAME_SEED_OBSERVED_RERUN_EVIDENCE_PACKET_HELPER_TOKENS,
        actionable_order=seed303_frontier_packet_acceptance_stack.actionable_order,
        current_before_after_driver_signature=before_after_probe.scenarios[0].driver_signature,
        target_before_after_driver_signature=before_after_probe.scenarios[1].driver_signature,
        runtime_bridge_frontier_driver_signature=runtime_bridge_frontier_packet.driver_signature,
        seed303_frontier_packet_acceptance_stack_driver_signature=(
            seed303_frontier_packet_acceptance_stack.frontier_packet_driver_signature
        ),
        acceptance_shortfall=seed303_frontier_packet_acceptance_stack.acceptance_shortfall,
        current_witness_floor=seed303_frontier_packet_acceptance_stack.current_witness_floor,
        required_min_witness_floor=(
            seed303_frontier_packet_acceptance_stack.required_min_witness_floor
        ),
        binding_slot_random_state=(
            seed303_frontier_packet_acceptance_stack.binding_slot_random_state
        ),
        binding_slot_seed_group=(
            seed303_frontier_packet_acceptance_stack.binding_slot_seed_group
        ),
        binding_slot_z_value=seed303_frontier_packet_acceptance_stack.binding_slot_z_value,
        residual_slot_random_state=(
            seed303_frontier_packet_acceptance_stack.residual_slot_random_state
        ),
        residual_slot_seed_group="residual",
        residual_slot_z_value=(
            seed303_frontier_packet_acceptance_stack.residual_slot_z_value
        ),
        binding_replication_seed=(
            seed303_frontier_packet_acceptance_stack.binding_replication_seed
        ),
        runtime_witness_path=seed303_frontier_packet_acceptance_stack.runtime_witness_path,
        acceptance_readout_path=(
            seed303_frontier_packet_acceptance_stack.acceptance_readout_path
        ),
        canonical_observed_rerun_evidence_packet_digest=canonical_digest,
    )


def _summarize_phase7_nonparametric_object_slice(
    *,
    design: MonteCarloDesign,
    dataset: SimulationDataset,
    nonparametric_payload: Any,
    random_state: int,
    replication_seed: int,
) -> Phase7NonparametricCalibrationObjectSlice:
    evaluation_grid = np.asarray(dataset.z0, dtype=float)
    truth_at_z0 = np.asarray(dataset.true_f_at_z0, dtype=float)
    bar_f_at_z0 = np.asarray(nonparametric_payload.bar_f_at_z0, dtype=float)
    absolute_error_at_z0 = np.abs(bar_f_at_z0 - truth_at_z0)
    sigma_z_hat = np.asarray(nonparametric_payload.sigma_z_hat, dtype=float)
    pointwise_interval_length = np.asarray(
        nonparametric_payload.pointwise_confidence_interval.upper, dtype=float
    ) - np.asarray(
        nonparametric_payload.pointwise_confidence_interval.lower, dtype=float
    )
    pointwise_coverage = (
        np.asarray(
            nonparametric_payload.pointwise_confidence_interval.lower, dtype=float
        )
        <= truth_at_z0
    ) & (
        truth_at_z0
        <= np.asarray(
            nonparametric_payload.pointwise_confidence_interval.upper, dtype=float
        )
    )
    uniform_band_length = np.asarray(
        nonparametric_payload.uniform_band.upper, dtype=float
    ) - np.asarray(nonparametric_payload.uniform_band.lower, dtype=float)
    max_absolute_error_grid_value, max_absolute_error = _grid_maximum(
        evaluation_grid,
        absolute_error_at_z0,
    )
    max_sigma_z_hat_grid_value, max_sigma_z_hat = _grid_maximum(
        evaluation_grid,
        sigma_z_hat,
    )
    max_pointwise_interval_grid_value, max_pointwise_interval_length = _grid_maximum(
        evaluation_grid,
        pointwise_interval_length,
    )
    return Phase7NonparametricCalibrationObjectSlice(
        dgp_name=design.dgp_name,
        n_obs=design.n_obs,
        p=design.p,
        random_state=int(random_state),
        replication_seed=int(replication_seed),
        evaluation_grid=evaluation_grid,
        true_f_at_z0=truth_at_z0,
        bar_f_at_z0=bar_f_at_z0,
        absolute_error_at_z0=absolute_error_at_z0,
        sigma_z_hat=sigma_z_hat,
        pointwise_interval_length=pointwise_interval_length,
        pointwise_coverage=pointwise_coverage,
        mean_absolute_error=float(np.mean(absolute_error_at_z0)),
        max_absolute_error_grid_value=max_absolute_error_grid_value,
        max_absolute_error=max_absolute_error,
        mean_sigma_z_hat=float(np.mean(sigma_z_hat)),
        max_sigma_z_hat_grid_value=max_sigma_z_hat_grid_value,
        max_sigma_z_hat=max_sigma_z_hat,
        mean_pointwise_interval_length=float(np.mean(pointwise_interval_length)),
        max_pointwise_interval_grid_value=max_pointwise_interval_grid_value,
        max_pointwise_interval_length=max_pointwise_interval_length,
        uniform_critical_value=float(nonparametric_payload.uniform_band.critical_value),
        mean_uniform_band_length=float(np.mean(uniform_band_length)),
    )


def _build_phase7_nonparametric_calibration_object_slice(
    *,
    design: MonteCarloDesign,
    random_state: int,
    replication_seed: int,
    n_boot: int,
) -> Phase7NonparametricCalibrationObjectSlice:
    dataset, nonparametric_payload = _fit_phase7_nonparametric_object_payload(
        design=design,
        replication_seed=int(replication_seed),
        n_boot=int(n_boot),
    )
    return _summarize_phase7_nonparametric_object_slice(
        design=design,
        dataset=dataset,
        nonparametric_payload=nonparametric_payload,
        random_state=int(random_state),
        replication_seed=int(replication_seed),
    )


def _phase7_object_slice_invalidity_example(
    error: (
        ZeroValidHoldoutError
        | NuisanceTrainingSupportError
        | Eq31ProjectionRankError
        | InferenceComputationError
    ),
    *,
    design: MonteCarloDesign,
    random_state: int,
    replication_seed: int,
    slice_source: str,
) -> dict[str, object]:
    return {
        **_monte_carlo_invalidity_example(
            error,
            design=design,
            monte_carlo_random_state=int(random_state),
            replication_seed=int(replication_seed),
        ),
        "slice_source": str(slice_source).strip(),
    }


def build_phase7_nonparametric_calibration_object_report(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    designs: Sequence[MonteCarloDesign],
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
) -> Phase7NonparametricCalibrationObjectReport:
    design_sequence = tuple(designs)
    if not design_sequence:
        raise ValueError(
            "build_phase7_nonparametric_calibration_object_report requires designs"
        )
    calibration_report = build_phase7_nonparametric_calibration_report(
        runtime_probe,
        target_n_obs=target_n_obs,
        reference_n_obs=reference_n_obs,
        p=p,
    )
    n_boot_value = _coerce_runtime_positive_integer("n_boot", n_boot)

    object_slice_cache: dict[
        tuple[str, int, int], Phase7NonparametricCalibrationObjectSlice
    ] = {}
    typed_invalidity_counts: Counter[str] = Counter()
    typed_invalidity_examples: dict[str, dict[str, object]] = {}
    for observation in runtime_probe.observations:
        if int(observation.n_obs) != int(target_n_obs):
            continue
        if int(observation.p) != int(p):
            continue
        typed_invalidity_counts.update(
            _normalize_invalidity_counts(observation.typed_invalidity_counts)
        )
        for error_name, metadata in _normalize_invalidity_examples(
            observation.typed_invalidity_examples
        ).items():
            typed_invalidity_examples.setdefault(error_name, metadata)

    decompositions: list[Phase7NonparametricCalibrationObjectDecomposition] = []
    for decomposition in calibration_report.decompositions:
        def _slice_for_random_state(
            target_random_state: int | None,
            *,
            n_obs: int,
            slice_source: str,
        ) -> Phase7NonparametricCalibrationObjectSlice | None:
            if target_random_state is None:
                return None
            design = _match_runtime_probe_design(
                design_sequence,
                dgp_name=decomposition.dgp_name,
                n_obs=int(n_obs),
                p=calibration_report.p,
            )
            cache_key = (
                decomposition.dgp_name,
                int(n_obs),
                int(target_random_state),
            )
            cached = object_slice_cache.get(cache_key)
            if cached is not None:
                return cached
            replication_seed = _phase7_runtime_probe_replication_seed(
                random_state=int(target_random_state),
                designs=design_sequence,
                dgp_name=decomposition.dgp_name,
                n_obs=int(n_obs),
                p=calibration_report.p,
            )
            try:
                cached = _build_phase7_nonparametric_calibration_object_slice(
                    design=design,
                    random_state=int(target_random_state),
                    replication_seed=replication_seed,
                    n_boot=n_boot_value,
                )
            except (
                ZeroValidHoldoutError,
                NuisanceTrainingSupportError,
                Eq31ProjectionRankError,
                InferenceComputationError,
            ) as exc:
                error_name = type(exc).__name__
                typed_invalidity_counts[error_name] += 1
                typed_invalidity_examples.setdefault(
                    error_name,
                    _phase7_object_slice_invalidity_example(
                        exc,
                        design=design,
                        random_state=int(target_random_state),
                        replication_seed=replication_seed,
                        slice_source=slice_source,
                    ),
                )
                return None
            object_slice_cache[cache_key] = cached
            return cached

        worst_random_state = decomposition.target_worst_coverage_random_state
        worst_n_obs = calibration_report.target_n_obs
        worst_source = "target_worst_coverage_slice"
        if worst_random_state is None:
            worst_random_state = decomposition.reference_worst_coverage_random_state
            worst_n_obs = calibration_report.reference_n_obs
            worst_source = "reference_worst_coverage_slice"

        longest_random_state = decomposition.target_longest_interval_random_state
        longest_n_obs = calibration_report.target_n_obs
        longest_source = "target_longest_interval_slice"
        if longest_random_state is None:
            longest_random_state = decomposition.reference_longest_interval_random_state
            longest_n_obs = calibration_report.reference_n_obs
            longest_source = "reference_longest_interval_slice"

        decompositions.append(
            Phase7NonparametricCalibrationObjectDecomposition(
                dgp_name=decomposition.dgp_name,
                target_worst_coverage_slice=_slice_for_random_state(
                    worst_random_state,
                    n_obs=worst_n_obs,
                    slice_source=worst_source,
                ),
                target_longest_interval_slice=_slice_for_random_state(
                    longest_random_state,
                    n_obs=longest_n_obs,
                    slice_source=longest_source,
                ),
                target_worst_coverage_source=worst_source,
                target_longest_interval_source=longest_source,
            )
        )

    return Phase7NonparametricCalibrationObjectReport(
        oracle_lane="paper-trigonometric",
        stage_label="phase7-nonparametric-object-probe",
        random_states=tuple(runtime_probe.random_states),
        target_n_obs=int(target_n_obs),
        reference_n_obs=int(reference_n_obs),
        p=int(p),
        decompositions=tuple(decompositions),
        typed_invalidity_counts=dict(sorted(typed_invalidity_counts.items())),
        typed_invalidity_examples=typed_invalidity_examples,
    )


def run_phase7_nonparametric_object_probe(
    *,
    random_states: Sequence[int] = (101, 202, 303),
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricCalibrationObjectReport:
    target_n_obs_value = int(target_n_obs)
    reference_n_obs_value = int(reference_n_obs)
    p_value = int(p)
    default_designs = (
        default_phase7_runtime_probe_designs()
        if (target_n_obs_value, reference_n_obs_value, p_value) == (200, 500, 50)
        else default_phase7_nonparametric_calibration_probe_designs()
    )
    design_sequence = tuple(designs or default_designs)
    runtime_probe = run_phase7_monte_carlo_runtime_probe(
        random_states=random_states,
        n_boot=n_boot,
        designs=design_sequence,
    )
    return build_phase7_nonparametric_calibration_object_report(
        runtime_probe,
        designs=design_sequence,
        n_boot=n_boot,
        target_n_obs=target_n_obs_value,
        reference_n_obs=reference_n_obs_value,
        p=p_value,
    )


def build_phase7_nonparametric_contrast_grid_replay_report(
    object_report: Phase7NonparametricCalibrationObjectReport,
    *,
    designs: Sequence[MonteCarloDesign],
    contrast_grid: Sequence[float] = (0.1, 0.3, 0.7),
    n_boot: int = 64,
) -> Phase7NonparametricContrastGridReplayReport:
    design_sequence = tuple(designs)
    if not design_sequence:
        raise ValueError(
            "build_phase7_nonparametric_contrast_grid_replay_report requires designs"
        )
    n_boot_value = _coerce_runtime_positive_integer("n_boot", n_boot)
    contrast_grid_array = _coerce_evaluation_grid(contrast_grid)
    if contrast_grid_array.size == 0:
        raise ValueError("contrast_grid must not be empty")
    contrast_grid_key = tuple(float(value) for value in contrast_grid_array.tolist())

    contrast_slice_cache: dict[
        tuple[str, int, int, tuple[float, ...]],
        Phase7NonparametricCalibrationObjectSlice,
    ] = {}
    decompositions: list[Phase7NonparametricContrastGridReplayDecomposition] = []

    for decomposition in object_report.decompositions:
        design = _match_runtime_probe_design(
            design_sequence,
            dgp_name=decomposition.dgp_name,
            n_obs=object_report.target_n_obs,
            p=object_report.p,
        )

        def _replay(
            current_slice: Phase7NonparametricCalibrationObjectSlice | None,
        ) -> Phase7NonparametricContrastGridReplay | None:
            if current_slice is None:
                return None
            cache_key = (
                decomposition.dgp_name,
                current_slice.random_state,
                current_slice.replication_seed,
                contrast_grid_key,
            )
            contrast_slice = contrast_slice_cache.get(cache_key)
            if contrast_slice is None:
                contrast_slice = _build_phase7_nonparametric_calibration_object_slice(
                    design=_clone_design_with_evaluation_grid(
                        design,
                        evaluation_grid=contrast_grid_array,
                    ),
                    random_state=current_slice.random_state,
                    replication_seed=current_slice.replication_seed,
                    n_boot=n_boot_value,
                )
                contrast_slice_cache[cache_key] = contrast_slice
            return Phase7NonparametricContrastGridReplay(
                contrast_grid=contrast_grid_key,
                current_grid_slice=current_slice,
                contrast_grid_slice=contrast_slice,
                mean_absolute_error_delta=(
                    contrast_slice.mean_absolute_error
                    - current_slice.mean_absolute_error
                ),
                mean_sigma_z_hat_delta=(
                    contrast_slice.mean_sigma_z_hat - current_slice.mean_sigma_z_hat
                ),
                mean_pointwise_interval_length_delta=(
                    contrast_slice.mean_pointwise_interval_length
                    - current_slice.mean_pointwise_interval_length
                ),
                pointwise_coverage_count_delta=(
                    int(np.sum(contrast_slice.pointwise_coverage, dtype=int))
                    - int(np.sum(current_slice.pointwise_coverage, dtype=int))
                ),
            )

        decompositions.append(
            Phase7NonparametricContrastGridReplayDecomposition(
                dgp_name=decomposition.dgp_name,
                target_worst_coverage_replay=_replay(
                    decomposition.target_worst_coverage_slice
                ),
                target_longest_interval_replay=_replay(
                    decomposition.target_longest_interval_slice
                ),
            )
        )

    return Phase7NonparametricContrastGridReplayReport(
        oracle_lane=object_report.oracle_lane,
        stage_label="phase7-nonparametric-contrast-grid-probe",
        random_states=object_report.random_states,
        target_n_obs=object_report.target_n_obs,
        reference_n_obs=object_report.reference_n_obs,
        p=object_report.p,
        contrast_grid=contrast_grid_key,
        decompositions=tuple(decompositions),
    )


def run_phase7_nonparametric_contrast_grid_probe(
    *,
    random_states: Sequence[int] = (101, 202, 303),
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    contrast_grid: Sequence[float] = (0.1, 0.3, 0.7),
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricContrastGridReplayReport:
    target_n_obs_value = int(target_n_obs)
    reference_n_obs_value = int(reference_n_obs)
    p_value = int(p)
    default_designs = (
        default_phase7_runtime_probe_designs()
        if (target_n_obs_value, reference_n_obs_value, p_value) == (200, 500, 50)
        else default_phase7_nonparametric_calibration_probe_designs()
    )
    design_sequence = tuple(designs or default_designs)
    object_report = run_phase7_nonparametric_object_probe(
        random_states=random_states,
        n_boot=n_boot,
        target_n_obs=target_n_obs_value,
        reference_n_obs=reference_n_obs_value,
        p=p_value,
        designs=design_sequence,
    )
    return build_phase7_nonparametric_contrast_grid_replay_report(
        object_report,
        designs=design_sequence,
        contrast_grid=contrast_grid,
        n_boot=n_boot,
    )


def _normalize_phase7_off_integer_grids(
    off_integer_grids: (
        Mapping[str, Sequence[float]] | Sequence[tuple[str, Sequence[float]]]
    ),
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if isinstance(off_integer_grids, Mapping):
        raw_items = tuple(off_integer_grids.items())
    else:
        raw_items = tuple(off_integer_grids)
    if not raw_items:
        raise ValueError("off_integer_grids must not be empty")

    normalized_items: list[tuple[str, tuple[float, ...]]] = []
    seen_labels: set[str] = set()
    for label, grid in raw_items:
        normalized_label = str(label).strip()
        if not normalized_label:
            raise ValueError("off_integer_grids labels must not be empty")
        if normalized_label in seen_labels:
            raise ValueError(
                f"off_integer_grids contains duplicate label: {normalized_label!r}"
            )
        seen_labels.add(normalized_label)
        normalized_grid = tuple(
            float(value) for value in _coerce_evaluation_grid(grid).tolist()
        )
        if not normalized_grid:
            raise ValueError("off_integer_grids entries must not be empty")
        normalized_items.append((normalized_label, normalized_grid))
    return tuple(normalized_items)


def _select_phase7_off_integer_focus(
    decomposition: Phase7NonparametricCalibrationObjectDecomposition,
) -> tuple[str, Phase7NonparametricCalibrationObjectSlice] | None:
    worst_slice = decomposition.target_worst_coverage_slice
    longest_slice = decomposition.target_longest_interval_slice
    if worst_slice is None and longest_slice is None:
        return None
    if worst_slice is None:
        return (
            decomposition.target_longest_interval_source
            or "target_longest_interval_slice",
            longest_slice,
        )
    if longest_slice is None:
        return (
            decomposition.target_worst_coverage_source or "target_worst_coverage_slice",
            worst_slice,
        )
    if (
        worst_slice.random_state,
        worst_slice.replication_seed,
    ) == (
        longest_slice.random_state,
        longest_slice.replication_seed,
    ):
        return (
            decomposition.target_worst_coverage_source or "target_worst_coverage_slice",
            worst_slice,
        )
    return (
        decomposition.target_longest_interval_source
        or "target_longest_interval_slice",
        longest_slice,
    )


def build_phase7_nonparametric_off_integer_replay_report(
    object_report: Phase7NonparametricCalibrationObjectReport,
    *,
    designs: Sequence[MonteCarloDesign],
    off_integer_grids: (
        Mapping[str, Sequence[float]] | Sequence[tuple[str, Sequence[float]]]
    ),
    n_boot: int = 64,
    stage_label: str = "phase7-nonparametric-off-integer-probe",
) -> Phase7NonparametricOffIntegerReplayReport:
    design_sequence = tuple(designs)
    if not design_sequence:
        raise ValueError(
            "build_phase7_nonparametric_off_integer_replay_report requires designs"
        )
    n_boot_value = _coerce_runtime_positive_integer("n_boot", n_boot)

    normalized_grids = _normalize_phase7_off_integer_grids(off_integer_grids)
    stage_label_value = str(stage_label).strip()
    if not stage_label_value:
        raise ValueError("stage_label must not be empty")

    replay_slice_cache: dict[
        tuple[str, int, int, tuple[float, ...]],
        Phase7NonparametricCalibrationObjectSlice,
    ] = {}
    focuses: list[Phase7NonparametricOffIntegerFocus] = []

    for decomposition in object_report.decompositions:
        selected = _select_phase7_off_integer_focus(decomposition)
        if selected is None:
            continue
        focus_target, current_slice = selected
        design = _match_runtime_probe_design(
            design_sequence,
            dgp_name=decomposition.dgp_name,
            n_obs=current_slice.n_obs,
            p=object_report.p,
        )

        replays: list[Phase7NonparametricOffIntegerReplay] = []
        for grid_label, evaluation_grid in normalized_grids:
            cache_key = (
                decomposition.dgp_name,
                current_slice.random_state,
                current_slice.replication_seed,
                evaluation_grid,
            )
            replay_slice = replay_slice_cache.get(cache_key)
            if replay_slice is None:
                replay_slice = _build_phase7_nonparametric_calibration_object_slice(
                    design=_clone_design_with_evaluation_grid(
                        design,
                        evaluation_grid=np.asarray(evaluation_grid, dtype=float),
                    ),
                    random_state=current_slice.random_state,
                    replication_seed=current_slice.replication_seed,
                    n_boot=n_boot_value,
                )
                replay_slice_cache[cache_key] = replay_slice
            replays.append(
                Phase7NonparametricOffIntegerReplay(
                    grid_label=grid_label,
                    evaluation_grid=evaluation_grid,
                    replay_slice=replay_slice,
                    mean_absolute_error_delta=(
                        replay_slice.mean_absolute_error
                        - current_slice.mean_absolute_error
                    ),
                    mean_sigma_z_hat_delta=(
                        replay_slice.mean_sigma_z_hat - current_slice.mean_sigma_z_hat
                    ),
                    mean_pointwise_interval_length_delta=(
                        replay_slice.mean_pointwise_interval_length
                        - current_slice.mean_pointwise_interval_length
                    ),
                    pointwise_coverage_count_delta=(
                        int(np.sum(replay_slice.pointwise_coverage, dtype=int))
                        - int(np.sum(current_slice.pointwise_coverage, dtype=int))
                    ),
                )
            )

        focuses.append(
            Phase7NonparametricOffIntegerFocus(
                dgp_name=decomposition.dgp_name,
                focus_target=focus_target,
                current_slice=current_slice,
                off_integer_replays=tuple(replays),
            )
        )

    return Phase7NonparametricOffIntegerReplayReport(
        oracle_lane=object_report.oracle_lane,
        stage_label=stage_label_value,
        random_states=object_report.random_states,
        target_n_obs=object_report.target_n_obs,
        reference_n_obs=object_report.reference_n_obs,
        p=object_report.p,
        grid_labels=tuple(label for label, _ in normalized_grids),
        focuses=tuple(focuses),
    )


def run_phase7_nonparametric_off_integer_replay_probe(
    *,
    random_states: Sequence[int] = (101, 202, 303),
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    off_integer_grids: Mapping[str, Sequence[float]] | None = None,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricOffIntegerReplayReport:
    return run_phase7_nonparametric_off_integer_probe(
        random_states=random_states,
        n_boot=n_boot,
        target_n_obs=target_n_obs,
        reference_n_obs=reference_n_obs,
        p=p,
        off_integer_grids=off_integer_grids,
        designs=designs,
    )


def build_phase7_nonparametric_local_hotspot_report(
    object_report: Phase7NonparametricCalibrationObjectReport,
    *,
    designs: Sequence[MonteCarloDesign],
    local_grids: Mapping[str, Sequence[float]] | None = None,
    n_boot: int = 64,
) -> Phase7NonparametricOffIntegerReplayReport:
    return build_phase7_nonparametric_off_integer_replay_report(
        object_report,
        designs=designs,
        off_integer_grids=(
            local_grids
            if local_grids is not None
            else {
                "near_zero_grid": (0.05, 0.15, 0.25),
                "tight_center_grid": (0.10, 0.15, 0.20),
                "micro_center_grid": (0.14, 0.15, 0.16),
            }
        ),
        n_boot=n_boot,
        stage_label="phase7-nonparametric-local-hotspot-probe",
    )


def run_phase7_nonparametric_local_hotspot_probe(
    *,
    random_states: Sequence[int] = (101, 202, 303),
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricOffIntegerReplayReport:
    target_n_obs_value = int(target_n_obs)
    reference_n_obs_value = int(reference_n_obs)
    p_value = int(p)
    default_designs = (
        default_phase7_runtime_probe_designs()
        if (target_n_obs_value, reference_n_obs_value, p_value) == (200, 500, 50)
        else default_phase7_nonparametric_calibration_probe_designs()
    )
    local_grids_value = (
        local_grids
        if local_grids is not None
        else {
            "near_zero_grid": (0.05, 0.15, 0.25),
            "tight_center_grid": (0.10, 0.15, 0.20),
            "micro_center_grid": (0.14, 0.15, 0.16),
        }
    )
    design_sequence = tuple(designs or default_designs)
    object_report = run_phase7_nonparametric_object_probe(
        random_states=random_states,
        n_boot=n_boot,
        target_n_obs=target_n_obs_value,
        reference_n_obs=reference_n_obs_value,
        p=p_value,
        designs=design_sequence,
    )
    return build_phase7_nonparametric_local_hotspot_report(
        object_report,
        designs=design_sequence,
        local_grids=local_grids_value,
        n_boot=n_boot,
    )


def _build_phase7_nonparametric_source_level_hotspot_replay(
    *,
    grid_label: str,
    replay_slice: Phase7NonparametricCalibrationObjectSlice,
    evaluation_basis: np.ndarray,
    v_f_hat: np.ndarray,
    n_valid_obs: int,
    hotspot_center: float,
) -> Phase7NonparametricSourceLevelHotspotReplay:
    basis = np.asarray(evaluation_basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError("evaluation_basis must be two-dimensional")
    if basis.shape[0] != replay_slice.evaluation_grid.shape[0]:
        raise ValueError(
            "evaluation_basis must align with replay_slice.evaluation_grid"
        )

    v_f_matrix = np.asarray(v_f_hat, dtype=float)
    if v_f_matrix.ndim != 2 or v_f_matrix.shape[0] != v_f_matrix.shape[1]:
        raise ValueError("v_f_hat must be square")
    if v_f_matrix.shape[0] != basis.shape[1]:
        raise ValueError("v_f_hat must align with evaluation_basis columns")

    n_valid_value = int(n_valid_obs)
    if n_valid_value <= 0:
        raise ValueError("n_valid_obs must be positive")

    hotspot_center_value = float(hotspot_center)
    evaluation_grid = np.asarray(replay_slice.evaluation_grid, dtype=float)
    center_index = int(np.argmin(np.abs(evaluation_grid - hotspot_center_value)))
    center_grid_value = float(evaluation_grid[center_index])
    center_row = basis[center_index]
    center_row_norm = float(np.linalg.norm(center_row))

    weighted_covariance = basis @ v_f_matrix @ basis.T
    pointwise_variance = np.diag(weighted_covariance).astype(float)
    center_variance = float(pointwise_variance[center_index])

    symmetric_v_f = 0.5 * (v_f_matrix + v_f_matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_v_f)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive_mask = eigenvalues > 0.0
    positive_eigenvalues = eigenvalues[positive_mask]
    positive_sum = float(np.sum(positive_eigenvalues))
    leading_eigenvalue = float(eigenvalues[0])
    leading_positive_share = (
        leading_eigenvalue / positive_sum
        if positive_sum > 0.0 and leading_eigenvalue > 0.0
        else 0.0
    )

    spectral_coordinates = eigenvectors.T @ basis.T
    grid_points: list[Phase7NonparametricSourceLevelHotspotGridPoint] = []
    for index, grid_value in enumerate(evaluation_grid):
        row = basis[index]
        row_norm = float(np.linalg.norm(row))
        cosine_denominator = row_norm * center_row_norm
        row_to_center_cosine = (
            float(np.dot(row, center_row) / cosine_denominator)
            if cosine_denominator > 0.0
            else 0.0
        )

        covariance_to_center = float(weighted_covariance[index, center_index])
        correlation_denominator = float(
            np.sqrt(max(pointwise_variance[index], 0.0) * max(center_variance, 0.0))
        )
        weighted_correlation_to_center = (
            covariance_to_center / correlation_denominator
            if correlation_denominator > 0.0
            else 0.0
        )

        row_contributions = eigenvalues * (spectral_coordinates[:, index] ** 2)
        positive_contributions = row_contributions[positive_mask]
        positive_contribution_sum = float(np.sum(positive_contributions))
        leading_eigen_share = (
            float(positive_contributions[0] / positive_contribution_sum)
            if positive_contribution_sum > 0.0 and positive_contributions.size > 0
            else 0.0
        )
        dominant_mode_index = -1
        dominant_mode_share = 0.0
        second_mode_share = 0.0
        if positive_contribution_sum > 0.0 and positive_contributions.size > 0:
            dominant_mode_order = np.argsort(positive_contributions)[::-1]
            dominant_mode_index = int(dominant_mode_order[0])
            dominant_mode_share = float(
                positive_contributions[dominant_mode_order[0]]
                / positive_contribution_sum
            )
            if dominant_mode_order.size > 1:
                second_mode_share = float(
                    positive_contributions[dominant_mode_order[1]]
                    / positive_contribution_sum
                )
        top_three_eigen_share = (
            float(
                np.sum(positive_contributions[: min(3, positive_contributions.size)])
                / positive_contribution_sum
            )
            if positive_contribution_sum > 0.0 and positive_contributions.size > 0
            else 0.0
        )
        effective_positive_mode_count = 0.0
        if positive_contribution_sum > 0.0 and positive_contributions.size > 0:
            normalized_positive_contributions = (
                positive_contributions / positive_contribution_sum
            )
            concentration_mass = float(np.sum(normalized_positive_contributions**2))
            effective_positive_mode_count = (
                float(1.0 / concentration_mass) if concentration_mass > 0.0 else 0.0
            )

        grid_points.append(
            Phase7NonparametricSourceLevelHotspotGridPoint(
                grid_value=float(grid_value),
                pointwise_variance=float(pointwise_variance[index]),
                row_to_center_cosine=row_to_center_cosine,
                weighted_covariance_to_center=covariance_to_center,
                weighted_correlation_to_center=weighted_correlation_to_center,
                leading_eigen_share=leading_eigen_share,
                dominant_mode_share=dominant_mode_share,
                second_mode_share=second_mode_share,
                top_three_eigen_share=top_three_eigen_share,
                effective_positive_mode_count=effective_positive_mode_count,
                dominant_mode_index=dominant_mode_index,
            )
        )

    return Phase7NonparametricSourceLevelHotspotReplay(
        grid_label=grid_label,
        replay_slice=replay_slice,
        center_grid_value=center_grid_value,
        center_index=center_index,
        leading_eigenvalue=leading_eigenvalue,
        leading_eigenvalue_positive_share=leading_positive_share,
        grid_points=tuple(grid_points),
    )


def _build_phase7_nonparametric_source_level_replays(
    *,
    design: MonteCarloDesign,
    source_slice: Phase7NonparametricCalibrationObjectSlice,
    normalized_grids: tuple[tuple[str, tuple[float, ...]], ...],
    n_boot: int,
    hotspot_center: float,
) -> tuple[Phase7NonparametricSourceLevelHotspotReplay, ...]:
    source_level_replays: list[Phase7NonparametricSourceLevelHotspotReplay] = []
    for grid_label, evaluation_grid in normalized_grids:
        replay_design = _clone_design_with_evaluation_grid(
            design,
            evaluation_grid=np.asarray(evaluation_grid, dtype=float),
        )
        dataset, nonparametric_payload = _fit_phase7_nonparametric_object_payload(
            design=replay_design,
            replication_seed=source_slice.replication_seed,
            n_boot=n_boot,
        )
        replay_slice = _summarize_phase7_nonparametric_object_slice(
            design=replay_design,
            dataset=dataset,
            nonparametric_payload=nonparametric_payload,
            random_state=source_slice.random_state,
            replication_seed=source_slice.replication_seed,
        )
        source_level_replays.append(
            _build_phase7_nonparametric_source_level_hotspot_replay(
                grid_label=grid_label,
                replay_slice=replay_slice,
                evaluation_basis=np.asarray(
                    nonparametric_payload.evaluation_basis,
                    dtype=float,
                ),
                v_f_hat=np.asarray(nonparametric_payload.v_f_hat, dtype=float),
                n_valid_obs=int(
                    nonparametric_payload.optimization_metadata["n_valid_obs"]
                ),
                hotspot_center=float(hotspot_center),
            )
        )
    return tuple(source_level_replays)


def build_phase7_nonparametric_source_level_hotspot_report(
    object_report: Phase7NonparametricCalibrationObjectReport,
    *,
    designs: Sequence[MonteCarloDesign],
    local_grids: Mapping[str, Sequence[float]] | None = None,
    n_boot: int = 64,
    hotspot_center: float = 0.15,
) -> Phase7NonparametricSourceLevelHotspotReport:
    design_sequence = tuple(designs)
    if not design_sequence:
        raise ValueError(
            "build_phase7_nonparametric_source_level_hotspot_report requires designs"
        )
    n_boot_value = _coerce_runtime_positive_integer("n_boot", n_boot)

    normalized_grids = _normalize_phase7_off_integer_grids(
        local_grids
        if local_grids is not None
        else {
            "near_zero_grid": (0.05, 0.15, 0.25),
            "tight_center_grid": (0.10, 0.15, 0.20),
            "micro_center_grid": (0.14, 0.15, 0.16),
        }
    )

    focuses: list[Phase7NonparametricSourceLevelHotspotFocus] = []
    for decomposition in object_report.decompositions:
        selected = _select_phase7_off_integer_focus(decomposition)
        if selected is None:
            continue
        focus_target, current_slice = selected
        design = _match_runtime_probe_design(
            design_sequence,
            dgp_name=decomposition.dgp_name,
            n_obs=current_slice.n_obs,
            p=object_report.p,
        )

        focuses.append(
            Phase7NonparametricSourceLevelHotspotFocus(
                dgp_name=decomposition.dgp_name,
                focus_target=focus_target,
                current_slice=current_slice,
                source_level_replays=_build_phase7_nonparametric_source_level_replays(
                    design=design,
                    source_slice=current_slice,
                    normalized_grids=normalized_grids,
                    n_boot=n_boot_value,
                    hotspot_center=float(hotspot_center),
                ),
            )
        )

    return Phase7NonparametricSourceLevelHotspotReport(
        oracle_lane=object_report.oracle_lane,
        stage_label="phase7-nonparametric-source-level-hotspot-probe",
        random_states=object_report.random_states,
        target_n_obs=object_report.target_n_obs,
        reference_n_obs=object_report.reference_n_obs,
        p=object_report.p,
        grid_labels=tuple(label for label, _ in normalized_grids),
        hotspot_center=float(hotspot_center),
        focuses=tuple(focuses),
        typed_invalidity_counts=object_report.typed_invalidity_counts,
        typed_invalidity_examples=object_report.typed_invalidity_examples,
    )


def run_phase7_nonparametric_source_level_hotspot_probe(
    *,
    random_states: Sequence[int] = (101, 202, 303),
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    designs: Sequence[MonteCarloDesign] | None = None,
    hotspot_center: float = 0.15,
) -> Phase7NonparametricSourceLevelHotspotReport:
    target_n_obs_value = int(target_n_obs)
    reference_n_obs_value = int(reference_n_obs)
    p_value = int(p)
    default_designs = (
        default_phase7_runtime_probe_designs()
        if (target_n_obs_value, reference_n_obs_value, p_value) == (200, 500, 50)
        else default_phase7_nonparametric_calibration_probe_designs()
    )
    local_grids_value = (
        local_grids
        if local_grids is not None
        else {
            "near_zero_grid": (0.05, 0.15, 0.25),
            "tight_center_grid": (0.10, 0.15, 0.20),
            "micro_center_grid": (0.14, 0.15, 0.16),
        }
    )
    design_sequence = tuple(designs or default_designs)
    object_report = run_phase7_nonparametric_object_probe(
        random_states=random_states,
        n_boot=n_boot,
        target_n_obs=target_n_obs_value,
        reference_n_obs=reference_n_obs_value,
        p=p_value,
        designs=design_sequence,
    )
    return build_phase7_nonparametric_source_level_hotspot_report(
        object_report,
        designs=design_sequence,
        local_grids=local_grids_value,
        n_boot=n_boot,
        hotspot_center=float(hotspot_center),
    )


def build_phase7_nonparametric_source_level_mode_comparison_report(
    object_report: Phase7NonparametricCalibrationObjectReport,
    *,
    designs: Sequence[MonteCarloDesign],
    local_grids: Mapping[str, Sequence[float]] | None = None,
    n_boot: int = 64,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
) -> Phase7NonparametricSourceLevelModeComparisonReport:
    design_sequence = tuple(designs)
    if not design_sequence:
        raise ValueError(
            "build_phase7_nonparametric_source_level_mode_comparison_report requires designs"
        )
    n_boot_value = _coerce_runtime_positive_integer("n_boot", n_boot)

    normalized_grids = _normalize_phase7_off_integer_grids(
        local_grids
        if local_grids is not None
        else {
            "near_zero_grid": (0.05, 0.15, 0.25),
            "tight_center_grid": (0.10, 0.15, 0.20),
            "micro_center_grid": (0.14, 0.15, 0.16),
        }
    )

    decomposition = object_report.decomposition(dgp_name)
    focus_candidates: list[tuple[str, Phase7NonparametricCalibrationObjectSlice]] = []
    if decomposition.target_worst_coverage_slice is not None:
        focus_candidates.append(
            (
                decomposition.target_worst_coverage_source
                or "target_worst_coverage_slice",
                decomposition.target_worst_coverage_slice,
            )
        )
    if decomposition.target_longest_interval_slice is not None:
        focus_candidates.append(
            (
                decomposition.target_longest_interval_source
                or "target_longest_interval_slice",
                decomposition.target_longest_interval_slice,
            )
        )

    unique_candidates: list[tuple[str, Phase7NonparametricCalibrationObjectSlice]] = []
    seen_focuses: set[tuple[int, int]] = set()
    for focus_target, current_slice in focus_candidates:
        focus_key = (current_slice.random_state, current_slice.replication_seed)
        if focus_key in seen_focuses:
            continue
        seen_focuses.add(focus_key)
        unique_candidates.append((focus_target, current_slice))

    if len(unique_candidates) < 2:
        support_design = _match_runtime_probe_design(
            design_sequence,
            dgp_name=decomposition.dgp_name,
            n_obs=object_report.reference_n_obs,
            p=object_report.p,
        )
        for random_state in object_report.random_states:
            replication_seed = _phase7_runtime_probe_replication_seed(
                random_state=random_state,
                designs=design_sequence,
                dgp_name=decomposition.dgp_name,
                n_obs=object_report.reference_n_obs,
                p=object_report.p,
            )
            focus_key = (int(random_state), int(replication_seed))
            if focus_key in seen_focuses:
                continue
            try:
                support_slice = _build_phase7_nonparametric_calibration_object_slice(
                    design=support_design,
                    random_state=int(random_state),
                    replication_seed=replication_seed,
                    n_boot=n_boot_value,
                )
            except (
                ZeroValidHoldoutError,
                NuisanceTrainingSupportError,
                Eq31ProjectionRankError,
                InferenceComputationError,
            ):
                continue
            unique_candidates.append(("reference_support_slice", support_slice))
            seen_focuses.add(focus_key)
            if len(unique_candidates) >= 2:
                break

    if len(unique_candidates) < 2:
        raise ValueError(
            "source-level mode comparison requires at least two distinct valid focus slices"
        )

    focuses = tuple(
        Phase7NonparametricSourceLevelModeComparisonFocus(
            dgp_name=decomposition.dgp_name,
            focus_target=focus_target,
            random_state=current_slice.random_state,
            replication_seed=current_slice.replication_seed,
            current_slice=current_slice,
            source_level_replays=_build_phase7_nonparametric_source_level_replays(
                design=_match_runtime_probe_design(
                    design_sequence,
                    dgp_name=decomposition.dgp_name,
                    n_obs=current_slice.n_obs,
                    p=object_report.p,
                ),
                source_slice=current_slice,
                normalized_grids=normalized_grids,
                n_boot=n_boot_value,
                hotspot_center=float(hotspot_center),
            ),
        )
        for focus_target, current_slice in unique_candidates
    )

    return Phase7NonparametricSourceLevelModeComparisonReport(
        oracle_lane=object_report.oracle_lane,
        stage_label="phase7-nonparametric-source-level-mode-comparison-probe",
        random_states=object_report.random_states,
        target_n_obs=object_report.target_n_obs,
        reference_n_obs=object_report.reference_n_obs,
        p=object_report.p,
        dgp_name=decomposition.dgp_name,
        grid_labels=tuple(label for label, _ in normalized_grids),
        hotspot_center=float(hotspot_center),
        focuses=focuses,
    )


def run_phase7_nonparametric_source_level_mode_comparison_probe(
    *,
    random_states: Sequence[int] = (101, 202, 303),
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    designs: Sequence[MonteCarloDesign] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
) -> Phase7NonparametricSourceLevelModeComparisonReport:
    target_n_obs_value = int(target_n_obs)
    reference_n_obs_value = int(reference_n_obs)
    p_value = int(p)
    default_designs = (
        default_phase7_runtime_probe_designs()
        if (target_n_obs_value, reference_n_obs_value, p_value) == (200, 500, 50)
        else default_phase7_nonparametric_calibration_probe_designs()
    )
    local_grids_value = (
        local_grids
        if local_grids is not None
        else {
            "near_zero_grid": (0.05, 0.15, 0.25),
            "tight_center_grid": (0.10, 0.15, 0.20),
            "micro_center_grid": (0.14, 0.15, 0.16),
        }
    )
    design_sequence = tuple(designs or default_designs)
    object_report = run_phase7_nonparametric_object_probe(
        random_states=random_states,
        n_boot=n_boot,
        target_n_obs=target_n_obs_value,
        reference_n_obs=reference_n_obs_value,
        p=p_value,
        designs=design_sequence,
    )
    return build_phase7_nonparametric_source_level_mode_comparison_report(
        object_report,
        designs=design_sequence,
        local_grids=local_grids_value,
        n_boot=n_boot,
        hotspot_center=float(hotspot_center),
        dgp_name=dgp_name,
    )


def build_phase7_nonparametric_source_level_seed_neighborhood_report(
    *,
    designs: Sequence[MonteCarloDesign],
    neighborhood_random_states: Sequence[int] = (
        299,
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
    ),
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
) -> Phase7NonparametricSourceLevelSeedNeighborhoodReport:
    design_sequence = tuple(designs)
    if not design_sequence:
        raise ValueError(
            "build_phase7_nonparametric_source_level_seed_neighborhood_report "
            "requires designs"
        )
    n_boot_value = _coerce_runtime_positive_integer("n_boot", n_boot)

    target_n_obs_value = int(target_n_obs)
    reference_n_obs_value = int(reference_n_obs)
    p_value = int(p)
    dgp_name_value = str(dgp_name).strip().upper()
    center_random_state_value = int(center_random_state)
    neighborhood_random_states_value = tuple(
        dict.fromkeys(int(value) for value in neighborhood_random_states)
    )
    if not neighborhood_random_states_value:
        raise ValueError("neighborhood_random_states must not be empty")
    if center_random_state_value not in neighborhood_random_states_value:
        raise ValueError(
            "center_random_state must be included in neighborhood_random_states"
        )

    normalized_grids = _normalize_phase7_off_integer_grids(
        local_grids
        if local_grids is not None
        else {
            "near_zero_grid": (0.05, 0.15, 0.25),
            "tight_center_grid": (0.10, 0.15, 0.20),
            "micro_center_grid": (0.14, 0.15, 0.16),
        }
    )
    entries: list[Phase7NonparametricSourceLevelSeedNeighborhoodEntry] = []
    typed_invalidity_counts: Counter[str] = Counter()
    typed_invalidity_examples: dict[str, dict[str, object]] = {}
    used_n_obs_value = target_n_obs_value

    def _append_entries_for_n_obs(*, n_obs: int, slice_source: str) -> None:
        design = _match_runtime_probe_design(
            design_sequence,
            dgp_name=dgp_name_value,
            n_obs=int(n_obs),
            p=p_value,
        )
        for random_state in neighborhood_random_states_value:
            replication_seed = _phase7_runtime_probe_replication_seed(
                random_state=random_state,
                designs=design_sequence,
                dgp_name=dgp_name_value,
                n_obs=int(n_obs),
                p=p_value,
            )
            try:
                current_slice = _build_phase7_nonparametric_calibration_object_slice(
                    design=design,
                    random_state=random_state,
                    replication_seed=replication_seed,
                    n_boot=n_boot_value,
                )
                source_level_replays = (
                    _build_phase7_nonparametric_source_level_replays(
                        design=design,
                        source_slice=current_slice,
                        normalized_grids=normalized_grids,
                        n_boot=n_boot_value,
                        hotspot_center=float(hotspot_center),
                    )
                )
            except (
                ZeroValidHoldoutError,
                NuisanceTrainingSupportError,
                Eq31ProjectionRankError,
                InferenceComputationError,
            ) as exc:
                error_name = type(exc).__name__
                typed_invalidity_counts[error_name] += 1
                typed_invalidity_examples.setdefault(
                    error_name,
                    _phase7_object_slice_invalidity_example(
                        exc,
                        design=design,
                        random_state=random_state,
                        replication_seed=replication_seed,
                        slice_source=slice_source,
                    ),
                )
                continue
            entries.append(
                Phase7NonparametricSourceLevelSeedNeighborhoodEntry(
                    dgp_name=dgp_name_value,
                    random_state=random_state,
                    replication_seed=replication_seed,
                    distance_from_center=random_state - center_random_state_value,
                    current_slice=current_slice,
                    source_level_replays=source_level_replays,
                )
            )

    _append_entries_for_n_obs(
        n_obs=target_n_obs_value,
        slice_source="seed_neighborhood_target_slice",
    )
    if not entries and target_n_obs_value != reference_n_obs_value:
        used_n_obs_value = reference_n_obs_value
        _append_entries_for_n_obs(
            n_obs=reference_n_obs_value,
            slice_source="seed_neighborhood_reference_slice",
        )

    return Phase7NonparametricSourceLevelSeedNeighborhoodReport(
        oracle_lane="paper-trigonometric",
        stage_label="phase7-nonparametric-source-level-seed-neighborhood-probe",
        target_n_obs=used_n_obs_value,
        reference_n_obs=reference_n_obs_value,
        p=p_value,
        dgp_name=dgp_name_value,
        center_random_state=center_random_state_value,
        neighborhood_random_states=neighborhood_random_states_value,
        grid_labels=tuple(label for label, _ in normalized_grids),
        hotspot_center=float(hotspot_center),
        entries=tuple(entries),
        typed_invalidity_counts=dict(sorted(typed_invalidity_counts.items())),
        typed_invalidity_examples=typed_invalidity_examples,
    )


def run_phase7_nonparametric_source_level_seed_neighborhood_probe(
    *,
    neighborhood_random_states: Sequence[int] = (
        299,
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
    ),
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelSeedNeighborhoodReport:
    target_n_obs_value = int(target_n_obs)
    reference_n_obs_value = int(reference_n_obs)
    p_value = int(p)
    default_designs = (
        default_phase7_runtime_probe_designs()
        if (target_n_obs_value, reference_n_obs_value, p_value) == (200, 500, 50)
        else default_phase7_nonparametric_calibration_probe_designs()
    )
    design_sequence = tuple(designs or default_designs)
    return build_phase7_nonparametric_source_level_seed_neighborhood_report(
        designs=design_sequence,
        neighborhood_random_states=neighborhood_random_states,
        n_boot=n_boot,
        target_n_obs=target_n_obs_value,
        reference_n_obs=reference_n_obs_value,
        p=p_value,
        local_grids=local_grids,
        hotspot_center=float(hotspot_center),
        dgp_name=dgp_name,
        center_random_state=center_random_state,
    )


def _phase7_pointwise_coverage_count(
    object_slice: Phase7NonparametricCalibrationObjectSlice,
) -> int:
    return int(np.count_nonzero(object_slice.pointwise_coverage))


def _phase7_ratio_to_current(current_value: float, replay_value: float) -> float:
    baseline = float(current_value)
    current = float(replay_value)
    if np.isclose(baseline, 0.0):
        return 1.0 if np.isclose(current, 0.0) else float("inf")
    return current / baseline


def _phase7_mean_error_to_half_interval_ratio(
    object_slice: Phase7NonparametricCalibrationObjectSlice,
) -> float:
    half_interval = 0.5 * float(object_slice.mean_pointwise_interval_length)
    mean_error = float(object_slice.mean_absolute_error)
    if np.isclose(half_interval, 0.0):
        return 0.0 if np.isclose(mean_error, 0.0) else float("inf")
    return mean_error / half_interval


def _phase7_pointwise_error_to_half_interval_ratio(
    absolute_error: float, interval_length: float
) -> float:
    half_interval = 0.5 * float(interval_length)
    mean_error = float(absolute_error)
    if np.isclose(half_interval, 0.0):
        return 0.0 if np.isclose(mean_error, 0.0) else float("inf")
    return mean_error / half_interval


def _phase7_ratio_shape_signature(
    pointwise_ratios: Sequence[float], leading_eigen_shares: Sequence[float]
) -> str:
    ratio_values = np.asarray(pointwise_ratios, dtype=float)
    share_values = np.asarray(leading_eigen_shares, dtype=float)
    max_ratio = float(np.max(ratio_values))
    min_ratio = float(np.min(ratio_values))
    ratio_range = max_ratio - min_ratio
    min_share = float(np.min(share_values))
    max_share = float(np.max(share_values))

    if max_ratio < 1.0:
        return "subcritical_control"
    if min_ratio > 1.35 and ratio_range < 0.05 and min_share > 0.95:
        return "dominant_mode_plateau"
    if ratio_range > 0.15 and max_share < 0.2:
        return "center_led_ratio_spike"
    return "mixed_ratio_shape"


def _phase7_shoulder_to_center_ratios(
    values: Sequence[float], center_index: int
) -> tuple[float, ...]:
    numeric_values = tuple(float(value) for value in values)
    center_value = numeric_values[int(center_index)]
    return tuple(
        _phase7_ratio_to_current(center_value, value)
        for index, value in enumerate(numeric_values)
        if index != int(center_index)
    )


def _phase7_center_to_mean_shoulder_ratio(
    values: Sequence[float], center_index: int
) -> float:
    numeric_values = tuple(float(value) for value in values)
    center_value = numeric_values[int(center_index)]
    shoulder_mean = float(
        np.mean(
            [
                value
                for index, value in enumerate(numeric_values)
                if index != int(center_index)
            ]
        )
    )
    return _phase7_ratio_to_current(shoulder_mean, center_value)


def _phase7_component_shape_signature(
    ratio_shape_signature: str,
    *,
    center_is_max_absolute_error: bool,
    center_is_min_sigma_z_hat: bool,
    center_is_max_sigma_z_hat: bool,
    center_is_min_pointwise_interval_length: bool,
    center_is_max_pointwise_interval_length: bool,
    min_shoulder_to_center_absolute_error_ratio: float,
    min_shoulder_to_center_sigma_z_hat_ratio: float,
    min_shoulder_to_center_interval_length_ratio: float,
    max_shoulder_to_center_interval_length_ratio: float,
) -> str:
    if ratio_shape_signature == "subcritical_control":
        return "guarded_control"
    if (
        ratio_shape_signature == "dominant_mode_plateau"
        and center_is_max_absolute_error
        and center_is_max_sigma_z_hat
        and center_is_max_pointwise_interval_length
        and min_shoulder_to_center_absolute_error_ratio > 0.95
        and min_shoulder_to_center_sigma_z_hat_ratio > 0.95
        and min_shoulder_to_center_interval_length_ratio > 0.95
        and max_shoulder_to_center_interval_length_ratio < 1.0
    ):
        return "co_moving_plateau"
    if (
        ratio_shape_signature == "center_led_ratio_spike"
        and center_is_max_absolute_error
        and center_is_min_sigma_z_hat
        and center_is_min_pointwise_interval_length
        and min_shoulder_to_center_absolute_error_ratio < 1.0
        and min_shoulder_to_center_sigma_z_hat_ratio > 1.03
        and min_shoulder_to_center_interval_length_ratio > 1.03
    ):
        return "interval_valley_spike"
    return "mixed_component_shape"


def _phase7_component_driver_signature(
    component_signature: str,
    *,
    min_pointwise_ratio: float,
    max_pointwise_ratio: float,
    center_to_shoulder_mean_absolute_error_ratio: float,
    center_to_shoulder_mean_sigma_z_hat_ratio: float,
    center_to_shoulder_mean_interval_length_ratio: float,
    component_alignment_spread: float,
    error_to_interval_center_shoulder_gap: float,
) -> str:
    aligned_lift = (
        component_signature
        in {"guarded_control", "mixed_component_shape", "co_moving_plateau"}
        and center_to_shoulder_mean_absolute_error_ratio > 1.0
        and center_to_shoulder_mean_sigma_z_hat_ratio > 1.0
        and center_to_shoulder_mean_interval_length_ratio > 1.0
        and component_alignment_spread < 0.04
    )
    if aligned_lift and max_pointwise_ratio <= 1.0:
        return "subcritical_aligned_lift"
    if aligned_lift and min_pointwise_ratio > 1.0:
        return "supercritical_aligned_lift"
    if aligned_lift and min_pointwise_ratio <= 1.0 < max_pointwise_ratio:
        return "threshold_boundary_aligned_lift"
    if (
        component_signature == "interval_valley_spike"
        and center_to_shoulder_mean_absolute_error_ratio > 1.0
        and center_to_shoulder_mean_sigma_z_hat_ratio < 1.0
        and center_to_shoulder_mean_interval_length_ratio < 1.0
        and error_to_interval_center_shoulder_gap > 0.1
    ):
        return "center_interval_deficit"
    return "mixed_component_driver"


def _phase7_component_driver_family_signature(driver_signature: str) -> str:
    normalized_signature = str(driver_signature).strip()
    if normalized_signature in {
        "subcritical_aligned_lift",
        "threshold_boundary_aligned_lift",
        "supercritical_aligned_lift",
    }:
        return "aligned_lift_family"
    if normalized_signature == "center_interval_deficit":
        return "interval_deficit_family"
    return "mixed_driver_family"


def build_phase7_nonparametric_source_level_coverage_decomposition_report(
    seed_neighborhood_report: Phase7NonparametricSourceLevelSeedNeighborhoodReport,
    *,
    focus_random_states: Sequence[int] = (300, 303, 307),
    focus_labels: Mapping[int, str] | None = None,
    grid_labels: Sequence[str] | None = None,
) -> Phase7NonparametricSourceLevelCoverageDecompositionReport:
    focus_random_states_value = tuple(
        dict.fromkeys(int(value) for value in focus_random_states)
    )
    if not focus_random_states_value:
        raise ValueError("focus_random_states must not be empty")

    grid_labels_value = tuple(
        str(label).strip()
        for label in (
            grid_labels
            if grid_labels is not None
            else seed_neighborhood_report.grid_labels
        )
    )
    if not grid_labels_value:
        raise ValueError("grid_labels must not be empty")

    focus_label_map = {
        300: "high_variance_control",
        303: "dominant_mode_hotspot",
        307: "low_share_counterexample",
    }
    if focus_labels is not None:
        focus_label_map.update(
            {int(key): str(value).strip() for key, value in focus_labels.items()}
        )

    focuses: list[Phase7NonparametricSourceLevelCoverageDecompositionFocus] = []
    for random_state in focus_random_states_value:
        entry = seed_neighborhood_report.entry(random_state)
        current_slice = entry.current_slice
        current_coverage_count = _phase7_pointwise_coverage_count(current_slice)
        current_error_to_half_interval_ratio = (
            _phase7_mean_error_to_half_interval_ratio(current_slice)
        )
        source_level_replays = {
            replay.grid_label: replay for replay in entry.source_level_replays
        }
        coverage_replays: list[
            Phase7NonparametricSourceLevelCoverageDecompositionReplay
        ] = []
        for grid_label in grid_labels_value:
            replay = source_level_replays[grid_label]
            replay_slice = replay.replay_slice
            replay_coverage_count = _phase7_pointwise_coverage_count(replay_slice)
            replay_error_to_half_interval_ratio = (
                _phase7_mean_error_to_half_interval_ratio(replay_slice)
            )
            coverage_replays.append(
                Phase7NonparametricSourceLevelCoverageDecompositionReplay(
                    grid_label=replay.grid_label,
                    evaluation_grid=replay.evaluation_grid,
                    replay_slice=replay_slice,
                    pointwise_coverage_count=replay_coverage_count,
                    pointwise_coverage_count_delta=(
                        replay_coverage_count - current_coverage_count
                    ),
                    mean_absolute_error_delta=(
                        replay_slice.mean_absolute_error
                        - current_slice.mean_absolute_error
                    ),
                    mean_absolute_error_ratio_to_current=_phase7_ratio_to_current(
                        current_slice.mean_absolute_error,
                        replay_slice.mean_absolute_error,
                    ),
                    mean_sigma_z_hat_delta=(
                        replay_slice.mean_sigma_z_hat - current_slice.mean_sigma_z_hat
                    ),
                    mean_sigma_z_hat_ratio_to_current=_phase7_ratio_to_current(
                        current_slice.mean_sigma_z_hat,
                        replay_slice.mean_sigma_z_hat,
                    ),
                    mean_pointwise_interval_length_delta=(
                        replay_slice.mean_pointwise_interval_length
                        - current_slice.mean_pointwise_interval_length
                    ),
                    mean_pointwise_interval_length_ratio_to_current=(
                        _phase7_ratio_to_current(
                            current_slice.mean_pointwise_interval_length,
                            replay_slice.mean_pointwise_interval_length,
                        )
                    ),
                    mean_error_to_half_interval_ratio=replay_error_to_half_interval_ratio,
                    mean_error_to_half_interval_ratio_delta=(
                        replay_error_to_half_interval_ratio
                        - current_error_to_half_interval_ratio
                    ),
                    center_leading_eigen_share=(
                        replay.grid_points[replay.center_index].leading_eigen_share
                    ),
                    leading_eigenvalue_positive_share=(
                        replay.leading_eigenvalue_positive_share
                    ),
                )
            )

        focuses.append(
            Phase7NonparametricSourceLevelCoverageDecompositionFocus(
                dgp_name=entry.dgp_name,
                focus_label=focus_label_map.get(
                    random_state, f"random_state_{random_state}"
                ),
                random_state=entry.random_state,
                replication_seed=entry.replication_seed,
                current_slice=current_slice,
                current_pointwise_coverage_count=current_coverage_count,
                current_mean_error_to_half_interval_ratio=(
                    current_error_to_half_interval_ratio
                ),
                coverage_replays=tuple(coverage_replays),
            )
        )

    return Phase7NonparametricSourceLevelCoverageDecompositionReport(
        oracle_lane=seed_neighborhood_report.oracle_lane,
        stage_label="phase7-nonparametric-source-level-coverage-decomposition-probe",
        target_n_obs=seed_neighborhood_report.target_n_obs,
        reference_n_obs=seed_neighborhood_report.reference_n_obs,
        p=seed_neighborhood_report.p,
        dgp_name=seed_neighborhood_report.dgp_name,
        center_random_state=seed_neighborhood_report.center_random_state,
        focus_random_states=focus_random_states_value,
        grid_labels=grid_labels_value,
        focuses=tuple(focuses),
    )


def run_phase7_nonparametric_source_level_coverage_decomposition_probe(
    *,
    focus_random_states: Sequence[int] = (300, 303, 307),
    neighborhood_random_states: Sequence[int] = (
        299,
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
    ),
    focus_labels: Mapping[int, str] | None = None,
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelCoverageDecompositionReport:
    seed_neighborhood_report = (
        run_phase7_nonparametric_source_level_seed_neighborhood_probe(
            neighborhood_random_states=neighborhood_random_states,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    return build_phase7_nonparametric_source_level_coverage_decomposition_report(
        seed_neighborhood_report,
        focus_random_states=focus_random_states,
        focus_labels=focus_labels,
    )


def build_phase7_nonparametric_source_level_ratio_shape_report(
    seed_neighborhood_report: Phase7NonparametricSourceLevelSeedNeighborhoodReport,
    *,
    focus_random_states: Sequence[int] = (300, 303, 307),
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
) -> Phase7NonparametricSourceLevelRatioShapeReport:
    focus_random_states_value = tuple(
        dict.fromkeys(int(value) for value in focus_random_states)
    )
    if not focus_random_states_value:
        raise ValueError("focus_random_states must not be empty")

    target_grid_label_value = str(target_grid_label).strip()
    if not target_grid_label_value:
        raise ValueError("target_grid_label must not be empty")

    focus_label_map = {
        300: "high_variance_control",
        303: "dominant_mode_hotspot",
        307: "low_share_counterexample",
    }
    if focus_labels is not None:
        focus_label_map.update(
            {int(key): str(value).strip() for key, value in focus_labels.items()}
        )

    focuses: list[Phase7NonparametricSourceLevelRatioShapeFocus] = []
    for random_state in focus_random_states_value:
        entry = seed_neighborhood_report.entry(random_state)
        replay_map = {
            replay.grid_label: replay for replay in entry.source_level_replays
        }
        target_replay = replay_map[target_grid_label_value]

        point_profiles: list[Phase7NonparametricSourceLevelRatioShapePointProfile] = []
        pointwise_ratios: list[float] = []
        leading_eigen_shares: list[float] = []
        for index, point in enumerate(target_replay.grid_points):
            pointwise_ratio = _phase7_pointwise_error_to_half_interval_ratio(
                target_replay.replay_slice.absolute_error_at_z0[index],
                target_replay.replay_slice.pointwise_interval_length[index],
            )
            point_profiles.append(
                Phase7NonparametricSourceLevelRatioShapePointProfile(
                    grid_value=point.grid_value,
                    pointwise_coverage=target_replay.replay_slice.pointwise_coverage[
                        index
                    ],
                    absolute_error=target_replay.replay_slice.absolute_error_at_z0[
                        index
                    ],
                    sigma_z_hat=target_replay.replay_slice.sigma_z_hat[index],
                    pointwise_interval_length=(
                        target_replay.replay_slice.pointwise_interval_length[index]
                    ),
                    error_to_half_interval_ratio=pointwise_ratio,
                    pointwise_ratio_above_one=pointwise_ratio > 1.0,
                    row_to_center_cosine=point.row_to_center_cosine,
                    weighted_correlation_to_center=point.weighted_correlation_to_center,
                    leading_eigen_share=point.leading_eigen_share,
                )
            )
            pointwise_ratios.append(pointwise_ratio)
            leading_eigen_shares.append(point.leading_eigen_share)

        ratio_array = np.asarray(pointwise_ratios, dtype=float)
        max_ratio_index = int(np.argmax(ratio_array))
        focuses.append(
            Phase7NonparametricSourceLevelRatioShapeFocus(
                dgp_name=entry.dgp_name,
                focus_label=focus_label_map.get(
                    random_state, f"random_state_{random_state}"
                ),
                random_state=entry.random_state,
                replication_seed=entry.replication_seed,
                target_grid_label=target_grid_label_value,
                current_slice=entry.current_slice,
                target_slice=target_replay.replay_slice,
                failure_signature=_phase7_ratio_shape_signature(
                    pointwise_ratios, leading_eigen_shares
                ),
                point_profiles=tuple(point_profiles),
                center_index=target_replay.center_index,
                center_grid_value=target_replay.center_grid_value,
                center_pointwise_ratio=pointwise_ratios[target_replay.center_index],
                min_pointwise_ratio=float(np.min(ratio_array)),
                max_pointwise_ratio=float(np.max(ratio_array)),
                pointwise_ratio_range=float(np.max(ratio_array) - np.min(ratio_array)),
                max_pointwise_ratio_grid_value=(
                    point_profiles[max_ratio_index].grid_value
                ),
                min_leading_eigen_share=float(np.min(leading_eigen_shares)),
                max_leading_eigen_share=float(np.max(leading_eigen_shares)),
            )
        )

    return Phase7NonparametricSourceLevelRatioShapeReport(
        oracle_lane=seed_neighborhood_report.oracle_lane,
        stage_label="phase7-nonparametric-source-level-ratio-shape-probe",
        target_n_obs=seed_neighborhood_report.target_n_obs,
        reference_n_obs=seed_neighborhood_report.reference_n_obs,
        p=seed_neighborhood_report.p,
        dgp_name=seed_neighborhood_report.dgp_name,
        center_random_state=seed_neighborhood_report.center_random_state,
        focus_random_states=focus_random_states_value,
        target_grid_label=target_grid_label_value,
        focuses=tuple(focuses),
    )


def run_phase7_nonparametric_source_level_ratio_shape_probe(
    *,
    focus_random_states: Sequence[int] = (300, 303, 307),
    neighborhood_random_states: Sequence[int] = (
        299,
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
    ),
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelRatioShapeReport:
    seed_neighborhood_report = (
        run_phase7_nonparametric_source_level_seed_neighborhood_probe(
            neighborhood_random_states=neighborhood_random_states,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    return build_phase7_nonparametric_source_level_ratio_shape_report(
        seed_neighborhood_report,
        focus_random_states=focus_random_states,
        focus_labels=focus_labels,
        target_grid_label=target_grid_label,
    )


def build_phase7_nonparametric_source_level_component_shape_report(
    ratio_shape_report: Phase7NonparametricSourceLevelRatioShapeReport,
) -> Phase7NonparametricSourceLevelComponentShapeReport:
    focuses: list[Phase7NonparametricSourceLevelComponentShapeFocus] = []
    for focus in ratio_shape_report.focuses:
        center_index = focus.center_index
        absolute_errors = tuple(
            profile.absolute_error for profile in focus.point_profiles
        )
        sigma_values = tuple(profile.sigma_z_hat for profile in focus.point_profiles)
        interval_lengths = tuple(
            profile.pointwise_interval_length for profile in focus.point_profiles
        )
        shoulder_absolute_error_ratios = _phase7_shoulder_to_center_ratios(
            absolute_errors, center_index
        )
        shoulder_sigma_ratios = _phase7_shoulder_to_center_ratios(
            sigma_values, center_index
        )
        shoulder_interval_ratios = _phase7_shoulder_to_center_ratios(
            interval_lengths, center_index
        )
        center_is_max_absolute_error = bool(
            np.isclose(absolute_errors[center_index], max(absolute_errors))
        )
        center_is_max_sigma_z_hat = bool(
            np.isclose(sigma_values[center_index], max(sigma_values))
        )
        center_is_min_sigma_z_hat = bool(
            np.isclose(sigma_values[center_index], min(sigma_values))
        )
        center_is_max_interval_length = bool(
            np.isclose(interval_lengths[center_index], max(interval_lengths))
        )
        center_is_min_interval_length = bool(
            np.isclose(interval_lengths[center_index], min(interval_lengths))
        )
        component_signature = _phase7_component_shape_signature(
            focus.failure_signature,
            center_is_max_absolute_error=center_is_max_absolute_error,
            center_is_min_sigma_z_hat=center_is_min_sigma_z_hat,
            center_is_max_sigma_z_hat=center_is_max_sigma_z_hat,
            center_is_min_pointwise_interval_length=center_is_min_interval_length,
            center_is_max_pointwise_interval_length=center_is_max_interval_length,
            min_shoulder_to_center_absolute_error_ratio=min(
                shoulder_absolute_error_ratios
            ),
            min_shoulder_to_center_sigma_z_hat_ratio=min(shoulder_sigma_ratios),
            min_shoulder_to_center_interval_length_ratio=min(shoulder_interval_ratios),
            max_shoulder_to_center_interval_length_ratio=max(shoulder_interval_ratios),
        )
        focuses.append(
            Phase7NonparametricSourceLevelComponentShapeFocus(
                dgp_name=focus.dgp_name,
                focus_label=focus.focus_label,
                random_state=focus.random_state,
                replication_seed=focus.replication_seed,
                target_grid_label=focus.target_grid_label,
                ratio_shape_signature=focus.failure_signature,
                component_signature=component_signature,
                point_profiles=focus.point_profiles,
                center_index=center_index,
                center_grid_value=focus.center_grid_value,
                center_is_max_absolute_error=center_is_max_absolute_error,
                center_is_max_sigma_z_hat=center_is_max_sigma_z_hat,
                center_is_min_sigma_z_hat=center_is_min_sigma_z_hat,
                center_is_max_pointwise_interval_length=center_is_max_interval_length,
                center_is_min_pointwise_interval_length=center_is_min_interval_length,
                min_shoulder_to_center_absolute_error_ratio=min(
                    shoulder_absolute_error_ratios
                ),
                max_shoulder_to_center_absolute_error_ratio=max(
                    shoulder_absolute_error_ratios
                ),
                min_shoulder_to_center_sigma_z_hat_ratio=min(shoulder_sigma_ratios),
                max_shoulder_to_center_sigma_z_hat_ratio=max(shoulder_sigma_ratios),
                min_shoulder_to_center_interval_length_ratio=min(
                    shoulder_interval_ratios
                ),
                max_shoulder_to_center_interval_length_ratio=max(
                    shoulder_interval_ratios
                ),
            )
        )

    return Phase7NonparametricSourceLevelComponentShapeReport(
        oracle_lane=ratio_shape_report.oracle_lane,
        stage_label="phase7-nonparametric-source-level-component-shape-probe",
        target_n_obs=ratio_shape_report.target_n_obs,
        reference_n_obs=ratio_shape_report.reference_n_obs,
        p=ratio_shape_report.p,
        dgp_name=ratio_shape_report.dgp_name,
        center_random_state=ratio_shape_report.center_random_state,
        focus_random_states=ratio_shape_report.focus_random_states,
        target_grid_label=ratio_shape_report.target_grid_label,
        focuses=tuple(focuses),
    )


def run_phase7_nonparametric_source_level_component_shape_probe(
    *,
    focus_random_states: Sequence[int] = (300, 303, 307),
    neighborhood_random_states: Sequence[int] = (
        299,
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
    ),
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelComponentShapeReport:
    neighborhood_random_states_value = tuple(
        dict.fromkeys(int(value) for value in neighborhood_random_states)
    )
    if not neighborhood_random_states_value:
        raise ValueError("neighborhood_random_states must not be empty")
    focus_random_states_value = tuple(
        dict.fromkeys(int(value) for value in focus_random_states)
    )
    if not focus_random_states_value:
        raise ValueError("focus_random_states must not be empty")

    seed_neighborhood_report = (
        run_phase7_nonparametric_source_level_seed_neighborhood_probe(
            neighborhood_random_states=neighborhood_random_states_value,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    if focus_random_states_value == neighborhood_random_states_value:
        focus_random_states_value = tuple(
            entry.random_state for entry in seed_neighborhood_report.entries
        )
        if not focus_random_states_value:
            raise ValueError(
                "component-shape probe has no statistically valid neighborhood "
                "entries after invalidity filtering; "
                f"typed_invalidity_counts={seed_neighborhood_report.typed_invalidity_counts!r}"
            )

    ratio_shape_report = build_phase7_nonparametric_source_level_ratio_shape_report(
        seed_neighborhood_report,
        focus_random_states=focus_random_states_value,
        focus_labels=focus_labels,
        target_grid_label=target_grid_label,
    )
    return build_phase7_nonparametric_source_level_component_shape_report(
        ratio_shape_report
    )


def build_phase7_nonparametric_source_level_component_driver_report(
    component_shape_report: Phase7NonparametricSourceLevelComponentShapeReport,
) -> Phase7NonparametricSourceLevelComponentDriverReport:
    focuses: list[Phase7NonparametricSourceLevelComponentDriverFocus] = []
    for focus in component_shape_report.focuses:
        center_index = focus.center_index
        absolute_errors = tuple(
            profile.absolute_error for profile in focus.point_profiles
        )
        sigma_values = tuple(profile.sigma_z_hat for profile in focus.point_profiles)
        interval_lengths = tuple(
            profile.pointwise_interval_length for profile in focus.point_profiles
        )
        pointwise_ratios = tuple(
            profile.error_to_half_interval_ratio for profile in focus.point_profiles
        )
        center_to_shoulder_mean_absolute_error_ratio = (
            _phase7_center_to_mean_shoulder_ratio(absolute_errors, center_index)
        )
        center_to_shoulder_mean_sigma_ratio = _phase7_center_to_mean_shoulder_ratio(
            sigma_values, center_index
        )
        center_to_shoulder_mean_interval_ratio = _phase7_center_to_mean_shoulder_ratio(
            interval_lengths, center_index
        )
        error_to_sigma_gap = (
            center_to_shoulder_mean_absolute_error_ratio
            - center_to_shoulder_mean_sigma_ratio
        )
        error_to_interval_gap = (
            center_to_shoulder_mean_absolute_error_ratio
            - center_to_shoulder_mean_interval_ratio
        )
        component_alignment_spread = max(
            center_to_shoulder_mean_absolute_error_ratio,
            center_to_shoulder_mean_sigma_ratio,
            center_to_shoulder_mean_interval_ratio,
        ) - min(
            center_to_shoulder_mean_absolute_error_ratio,
            center_to_shoulder_mean_sigma_ratio,
            center_to_shoulder_mean_interval_ratio,
        )
        min_pointwise_ratio = float(np.min(pointwise_ratios))
        max_pointwise_ratio = float(np.max(pointwise_ratios))
        driver_signature = _phase7_component_driver_signature(
            focus.component_signature,
            min_pointwise_ratio=min_pointwise_ratio,
            max_pointwise_ratio=max_pointwise_ratio,
            center_to_shoulder_mean_absolute_error_ratio=(
                center_to_shoulder_mean_absolute_error_ratio
            ),
            center_to_shoulder_mean_sigma_z_hat_ratio=(
                center_to_shoulder_mean_sigma_ratio
            ),
            center_to_shoulder_mean_interval_length_ratio=(
                center_to_shoulder_mean_interval_ratio
            ),
            component_alignment_spread=component_alignment_spread,
            error_to_interval_center_shoulder_gap=error_to_interval_gap,
        )
        focuses.append(
            Phase7NonparametricSourceLevelComponentDriverFocus(
                dgp_name=focus.dgp_name,
                focus_label=focus.focus_label,
                random_state=focus.random_state,
                replication_seed=focus.replication_seed,
                target_grid_label=focus.target_grid_label,
                ratio_shape_signature=focus.ratio_shape_signature,
                component_signature=focus.component_signature,
                driver_signature=driver_signature,
                point_profiles=focus.point_profiles,
                center_index=center_index,
                center_grid_value=focus.center_grid_value,
                min_pointwise_ratio=min_pointwise_ratio,
                max_pointwise_ratio=max_pointwise_ratio,
                center_to_shoulder_mean_absolute_error_ratio=(
                    center_to_shoulder_mean_absolute_error_ratio
                ),
                center_to_shoulder_mean_sigma_z_hat_ratio=(
                    center_to_shoulder_mean_sigma_ratio
                ),
                center_to_shoulder_mean_interval_length_ratio=(
                    center_to_shoulder_mean_interval_ratio
                ),
                error_to_sigma_center_shoulder_gap=error_to_sigma_gap,
                error_to_interval_center_shoulder_gap=error_to_interval_gap,
                component_alignment_spread=component_alignment_spread,
            )
        )

    return Phase7NonparametricSourceLevelComponentDriverReport(
        oracle_lane=component_shape_report.oracle_lane,
        stage_label="phase7-nonparametric-source-level-component-driver-probe",
        target_n_obs=component_shape_report.target_n_obs,
        reference_n_obs=component_shape_report.reference_n_obs,
        p=component_shape_report.p,
        dgp_name=component_shape_report.dgp_name,
        center_random_state=component_shape_report.center_random_state,
        focus_random_states=component_shape_report.focus_random_states,
        target_grid_label=component_shape_report.target_grid_label,
        focuses=tuple(focuses),
    )


def run_phase7_nonparametric_source_level_component_driver_probe(
    *,
    focus_random_states: Sequence[int] = (300, 303, 307),
    neighborhood_random_states: Sequence[int] = (
        299,
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
    ),
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelComponentDriverReport:
    component_shape_report = (
        run_phase7_nonparametric_source_level_component_shape_probe(
            focus_random_states=focus_random_states,
            neighborhood_random_states=neighborhood_random_states,
            focus_labels=focus_labels,
            target_grid_label=target_grid_label,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    return build_phase7_nonparametric_source_level_component_driver_report(
        component_shape_report
    )


def build_phase7_nonparametric_source_level_driver_family_report(
    component_driver_report: Phase7NonparametricSourceLevelComponentDriverReport,
    *,
    candidate_random_states: Sequence[int] = (303, 307),
) -> Phase7NonparametricSourceLevelDriverFamilyReport:
    candidate_random_states_value = tuple(
        dict.fromkeys(int(value) for value in candidate_random_states)
    )
    if not candidate_random_states_value:
        raise ValueError("candidate_random_states must not be empty")

    focus_by_random_state = {
        focus.random_state: focus for focus in component_driver_report.focuses
    }
    missing_candidates = [
        random_state
        for random_state in candidate_random_states_value
        if random_state not in focus_by_random_state
    ]
    if missing_candidates:
        available_random_states = tuple(sorted(focus_by_random_state))
        raise ValueError(
            "component-driver report missing driver-family candidate(s) after "
            "statistical invalidity filtering; "
            f"missing_candidates={missing_candidates!r}; "
            f"available_random_states={available_random_states!r}"
        )

    family_members: dict[str, list[int]] = {}
    unique_threshold_crossing_members: dict[str, tuple[int, ...]] = {}
    for focus in component_driver_report.focuses:
        family_signature = _phase7_component_driver_family_signature(
            focus.driver_signature
        )
        if family_signature == "mixed_driver_family":
            continue
        family_members.setdefault(family_signature, []).append(focus.random_state)

    aligned_threshold_crossers = tuple(
        focus.random_state
        for focus in component_driver_report.focuses
        if focus.driver_signature == "supercritical_aligned_lift"
    )
    if aligned_threshold_crossers:
        unique_threshold_crossing_members["aligned_lift_family"] = (
            aligned_threshold_crossers
        )

    interval_threshold_crossers = tuple(
        focus.random_state
        for focus in component_driver_report.focuses
        if focus.driver_signature == "center_interval_deficit"
    )
    if interval_threshold_crossers:
        unique_threshold_crossing_members["interval_deficit_family"] = (
            interval_threshold_crossers
        )

    candidates: list[Phase7NonparametricSourceLevelDriverFamilyCandidate] = []
    for random_state in candidate_random_states_value:
        focus = focus_by_random_state[random_state]
        family_signature = _phase7_component_driver_family_signature(
            focus.driver_signature
        )
        family_random_states = tuple(
            family_members.get(family_signature, [random_state])
        )
        threshold_crossers = unique_threshold_crossing_members.get(family_signature, ())
        candidates.append(
            Phase7NonparametricSourceLevelDriverFamilyCandidate(
                dgp_name=focus.dgp_name,
                focus_label=focus.focus_label,
                random_state=focus.random_state,
                replication_seed=focus.replication_seed,
                driver_signature=focus.driver_signature,
                driver_family_signature=family_signature,
                family_random_states=family_random_states,
                family_support_count=len(family_random_states),
                is_unique_threshold_crossing_member=(
                    threshold_crossers == (focus.random_state,)
                ),
                min_pointwise_ratio=focus.min_pointwise_ratio,
                max_pointwise_ratio=focus.max_pointwise_ratio,
                error_to_interval_center_shoulder_gap=(
                    focus.error_to_interval_center_shoulder_gap
                ),
            )
        )

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.family_support_count,
            candidate.min_pointwise_ratio,
            -candidate.random_state,
        ),
        reverse=True,
    )
    recommended_candidate = ranked_candidates[0]
    runner_up = ranked_candidates[1] if len(ranked_candidates) > 1 else None
    if runner_up is None:
        recommendation_rationale = (
            f"{recommended_candidate.driver_family_signature} is the only "
            "candidate family in the bounded neighborhood"
        )
    else:
        recommendation_rationale = (
            f"{recommended_candidate.driver_family_signature} recurs across "
            f"{recommended_candidate.family_support_count} seeds "
            f"{recommended_candidate.family_random_states}, while "
            f"{runner_up.driver_family_signature} only appears at "
            f"{runner_up.family_random_states}"
        )

    return Phase7NonparametricSourceLevelDriverFamilyReport(
        oracle_lane=component_driver_report.oracle_lane,
        stage_label="phase7-nonparametric-source-level-driver-family-probe",
        target_n_obs=component_driver_report.target_n_obs,
        reference_n_obs=component_driver_report.reference_n_obs,
        p=component_driver_report.p,
        dgp_name=component_driver_report.dgp_name,
        neighborhood_random_states=component_driver_report.focus_random_states,
        candidate_random_states=candidate_random_states_value,
        target_grid_label=component_driver_report.target_grid_label,
        candidates=tuple(candidates),
        recommended_next_focus_random_state=recommended_candidate.random_state,
        recommended_next_focus_signature=recommended_candidate.driver_signature,
        recommendation_rationale=recommendation_rationale,
    )


def run_phase7_nonparametric_source_level_driver_family_probe(
    *,
    neighborhood_random_states: Sequence[int] = tuple(range(296, 321)),
    candidate_random_states: Sequence[int] = (303, 307),
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelDriverFamilyReport:
    neighborhood_random_states_value = tuple(
        dict.fromkeys(int(value) for value in neighborhood_random_states)
    )
    if not neighborhood_random_states_value:
        raise ValueError("neighborhood_random_states must not be empty")

    normalized_focus_labels = {
        random_state: f"seed_{random_state}"
        for random_state in neighborhood_random_states_value
    }
    if focus_labels is not None:
        normalized_focus_labels.update(
            {int(key): str(value) for key, value in focus_labels.items()}
        )

    component_driver_report = (
        run_phase7_nonparametric_source_level_component_driver_probe(
            focus_random_states=neighborhood_random_states_value,
            neighborhood_random_states=neighborhood_random_states_value,
            focus_labels=normalized_focus_labels,
            target_grid_label=target_grid_label,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    return build_phase7_nonparametric_source_level_driver_family_report(
        component_driver_report,
        candidate_random_states=candidate_random_states,
    )


def build_phase7_nonparametric_source_level_aligned_lift_threshold_report(
    component_driver_report: Phase7NonparametricSourceLevelComponentDriverReport,
    *,
    candidate_random_state: int = 303,
) -> Phase7NonparametricSourceLevelAlignedLiftThresholdReport:
    candidate_random_state_value = int(candidate_random_state)
    focus_by_random_state = {
        focus.random_state: focus for focus in component_driver_report.focuses
    }
    if candidate_random_state_value not in focus_by_random_state:
        available_random_states = tuple(sorted(focus_by_random_state))
        raise ValueError(
            "component-driver report missing aligned-lift threshold candidate after "
            "statistical invalidity filtering; "
            f"candidate_random_state={candidate_random_state_value!r}; "
            f"available_random_states={available_random_states!r}"
        )

    candidate_focus = focus_by_random_state[candidate_random_state_value]
    candidate_family_signature = _phase7_component_driver_family_signature(
        candidate_focus.driver_signature
    )
    if candidate_family_signature != "aligned_lift_family":
        raise ValueError(
            "aligned-lift threshold probe requires an aligned-lift-family candidate"
        )

    aligned_lift_focuses = tuple(
        focus
        for focus in component_driver_report.focuses
        if _phase7_component_driver_family_signature(focus.driver_signature)
        == "aligned_lift_family"
    )
    if not aligned_lift_focuses:
        raise ValueError(
            "aligned-lift threshold probe requires aligned-lift-family focuses"
        )

    members: list[Phase7NonparametricSourceLevelAlignedLiftThresholdMember] = []
    threshold_crossing_members: list[
        Phase7NonparametricSourceLevelAlignedLiftThresholdMember
    ] = []
    subcritical_members: list[
        Phase7NonparametricSourceLevelAlignedLiftThresholdMember
    ] = []

    for focus in aligned_lift_focuses:
        threshold_margin = focus.min_pointwise_ratio - 1.0
        stability_range = focus.max_pointwise_ratio - focus.min_pointwise_ratio
        member = Phase7NonparametricSourceLevelAlignedLiftThresholdMember(
            dgp_name=focus.dgp_name,
            focus_label=focus.focus_label,
            random_state=focus.random_state,
            replication_seed=focus.replication_seed,
            driver_signature=focus.driver_signature,
            min_pointwise_ratio=focus.min_pointwise_ratio,
            max_pointwise_ratio=focus.max_pointwise_ratio,
            threshold_margin=threshold_margin,
            stability_range=stability_range,
            component_alignment_spread=focus.component_alignment_spread,
            error_to_interval_center_shoulder_gap=(
                focus.error_to_interval_center_shoulder_gap
            ),
        )
        members.append(member)
        if member.threshold_margin > 0.0:
            threshold_crossing_members.append(member)
        else:
            subcritical_members.append(member)

    ranked_subcritical_members = sorted(
        subcritical_members,
        key=lambda member: (
            member.threshold_margin,
            -member.stability_range,
            -member.random_state,
        ),
        reverse=True,
    )
    boundary_random_states = tuple(
        member.random_state for member in ranked_subcritical_members[:2]
    )
    if not boundary_random_states:
        raise ValueError(
            "aligned-lift threshold probe requires at least one subcritical family member"
        )

    leading_boundary_member = ranked_subcritical_members[0]
    trailing_boundary_member = (
        ranked_subcritical_members[1]
        if len(ranked_subcritical_members) > 1
        else ranked_subcritical_members[0]
    )
    if len(threshold_crossing_members) == 1:
        threshold_crossing_member = threshold_crossing_members[0]
        threshold_crossing_random_state = threshold_crossing_member.random_state
        recommendation_rationale = (
            f"{threshold_crossing_member.random_state} is the only aligned_lift_family "
            f"member with min pointwise ratio above 1.0; "
            f"{leading_boundary_member.random_state} is the closest stable "
            "subcritical comparison and "
            f"{trailing_boundary_member.random_state} is the next shoulder support"
        )
    elif not threshold_crossing_members:
        threshold_crossing_random_state = None
        recommendation_rationale = (
            "No aligned_lift_family member has min pointwise ratio above 1.0; "
            f"{leading_boundary_member.random_state} is the nearest boundary member "
            f"and {trailing_boundary_member.random_state} is the next support member"
        )
    else:
        raise ValueError(
            "aligned-lift threshold probe found multiple threshold-crossing members"
        )

    return Phase7NonparametricSourceLevelAlignedLiftThresholdReport(
        oracle_lane=component_driver_report.oracle_lane,
        stage_label="phase7-nonparametric-source-level-aligned-lift-threshold-probe",
        target_n_obs=component_driver_report.target_n_obs,
        reference_n_obs=component_driver_report.reference_n_obs,
        p=component_driver_report.p,
        dgp_name=component_driver_report.dgp_name,
        neighborhood_random_states=component_driver_report.focus_random_states,
        aligned_lift_family_random_states=tuple(
            focus.random_state for focus in aligned_lift_focuses
        ),
        target_grid_label=component_driver_report.target_grid_label,
        members=tuple(members),
        threshold_crossing_random_state=threshold_crossing_random_state,
        recommended_boundary_random_states=boundary_random_states,
        recommendation_rationale=recommendation_rationale,
    )


def run_phase7_nonparametric_source_level_aligned_lift_threshold_probe(
    *,
    neighborhood_random_states: Sequence[int] = tuple(range(296, 321)),
    candidate_random_state: int = 303,
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelAlignedLiftThresholdReport:
    neighborhood_random_states_value = tuple(
        dict.fromkeys(int(value) for value in neighborhood_random_states)
    )
    if not neighborhood_random_states_value:
        raise ValueError("neighborhood_random_states must not be empty")

    normalized_focus_labels = {
        random_state: f"seed_{random_state}"
        for random_state in neighborhood_random_states_value
    }
    if focus_labels is not None:
        normalized_focus_labels.update(
            {int(key): str(value) for key, value in focus_labels.items()}
        )

    component_driver_report = (
        run_phase7_nonparametric_source_level_component_driver_probe(
            focus_random_states=neighborhood_random_states_value,
            neighborhood_random_states=neighborhood_random_states_value,
            focus_labels=normalized_focus_labels,
            target_grid_label=target_grid_label,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    return build_phase7_nonparametric_source_level_aligned_lift_threshold_report(
        component_driver_report,
        candidate_random_state=candidate_random_state,
    )


def build_phase7_nonparametric_source_level_aligned_lift_boundary_comparison_report(
    threshold_report: Phase7NonparametricSourceLevelAlignedLiftThresholdReport,
) -> Phase7NonparametricSourceLevelAlignedLiftBoundaryComparisonReport:
    boundary_ordering_random_states = (
        threshold_report.recommended_boundary_random_states
    )
    if len(boundary_ordering_random_states) < 2:
        raise ValueError(
            "aligned-lift boundary comparison requires primary and secondary subcritical "
            "comparators"
        )
    if threshold_report.threshold_crossing_random_state is None:
        raise ValueError(
            "aligned-lift boundary comparison requires a threshold-crossing member"
        )

    threshold_crossing_member = threshold_report.member(
        threshold_report.threshold_crossing_random_state
    )
    primary_boundary_member = threshold_report.member(
        boundary_ordering_random_states[0]
    )
    secondary_support_member = threshold_report.member(
        boundary_ordering_random_states[1]
    )

    if primary_boundary_member.threshold_margin >= 0.0:
        raise ValueError("primary boundary member must remain subcritical")
    if secondary_support_member.threshold_margin >= 0.0:
        raise ValueError("secondary support member must remain subcritical")

    recommendation_rationale = (
        f"{threshold_crossing_member.random_state} remains the unique threshold crossing "
        "member; "
        f"{primary_boundary_member.random_state} is the closest stable boundary because "
        f"its threshold margin ({primary_boundary_member.threshold_margin:+.3f}) is the "
        "nearest subcritical comparator to zero while its stability range "
        f"({primary_boundary_member.stability_range:.3f}) stays tighter than the crossing "
        f"slice ({threshold_crossing_member.stability_range:.3f}); "
        f"{secondary_support_member.random_state} remains the next shoulder support because "
        f"its wider stability range ({secondary_support_member.stability_range:.3f}) keeps "
        "it inside the same aligned-lift family without making it the primary boundary."
    )

    return Phase7NonparametricSourceLevelAlignedLiftBoundaryComparisonReport(
        oracle_lane=threshold_report.oracle_lane,
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-boundary-comparison-probe"
        ),
        target_n_obs=threshold_report.target_n_obs,
        reference_n_obs=threshold_report.reference_n_obs,
        p=threshold_report.p,
        dgp_name=threshold_report.dgp_name,
        aligned_lift_family_random_states=threshold_report.aligned_lift_family_random_states,
        target_grid_label=threshold_report.target_grid_label,
        threshold_crossing_random_state=threshold_crossing_member.random_state,
        primary_boundary_random_state=primary_boundary_member.random_state,
        secondary_support_random_state=secondary_support_member.random_state,
        boundary_ordering_random_states=boundary_ordering_random_states,
        threshold_crossing_member=threshold_crossing_member,
        primary_boundary_member=primary_boundary_member,
        secondary_support_member=secondary_support_member,
        threshold_margin_gap_to_primary_boundary=(
            threshold_crossing_member.threshold_margin
            - primary_boundary_member.threshold_margin
        ),
        stability_range_gap_to_primary_boundary=(
            threshold_crossing_member.stability_range
            - primary_boundary_member.stability_range
        ),
        secondary_support_stability_excess_over_primary_boundary=(
            secondary_support_member.stability_range
            - primary_boundary_member.stability_range
        ),
        alignment_spread_gap_to_primary_boundary=(
            threshold_crossing_member.component_alignment_spread
            - primary_boundary_member.component_alignment_spread
        ),
        recommendation_rationale=recommendation_rationale,
    )


def run_phase7_nonparametric_source_level_aligned_lift_boundary_comparison_probe(
    *,
    neighborhood_random_states: Sequence[int] = tuple(range(296, 321)),
    candidate_random_state: int = 303,
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelAlignedLiftBoundaryComparisonReport:
    threshold_report = (
        run_phase7_nonparametric_source_level_aligned_lift_threshold_probe(
            neighborhood_random_states=neighborhood_random_states,
            candidate_random_state=candidate_random_state,
            focus_labels=focus_labels,
            target_grid_label=target_grid_label,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    return (
        build_phase7_nonparametric_source_level_aligned_lift_boundary_comparison_report(
            threshold_report
        )
    )


def _build_phase7_aligned_lift_spectral_condition_focus(
    focus: Phase7NonparametricSourceLevelComponentDriverFocus,
    *,
    threshold_member: Phase7NonparametricSourceLevelAlignedLiftThresholdMember,
) -> Phase7NonparametricSourceLevelAlignedLiftSpectralConditionFocus:
    point_profiles = focus.point_profiles
    center_profile = point_profiles[focus.center_index]
    mean_pointwise_ratio = float(
        np.mean([profile.error_to_half_interval_ratio for profile in point_profiles])
    )
    mean_row_to_center_cosine = float(
        np.mean([profile.row_to_center_cosine for profile in point_profiles])
    )
    mean_weighted_correlation_to_center = float(
        np.mean([profile.weighted_correlation_to_center for profile in point_profiles])
    )
    mean_leading_eigen_share = float(
        np.mean([profile.leading_eigen_share for profile in point_profiles])
    )
    return Phase7NonparametricSourceLevelAlignedLiftSpectralConditionFocus(
        dgp_name=focus.dgp_name,
        focus_label=focus.focus_label,
        random_state=focus.random_state,
        replication_seed=focus.replication_seed,
        driver_signature=focus.driver_signature,
        mean_pointwise_ratio=mean_pointwise_ratio,
        center_pointwise_ratio=center_profile.error_to_half_interval_ratio,
        threshold_margin=threshold_member.threshold_margin,
        mean_row_to_center_cosine=mean_row_to_center_cosine,
        mean_weighted_correlation_to_center=mean_weighted_correlation_to_center,
        mean_leading_eigen_share=mean_leading_eigen_share,
        center_leading_eigen_share=center_profile.leading_eigen_share,
        component_alignment_spread=focus.component_alignment_spread,
    )


def build_phase7_nonparametric_source_level_aligned_lift_spectral_condition_report(
    component_driver_report: Phase7NonparametricSourceLevelComponentDriverReport,
    boundary_comparison_report: Phase7NonparametricSourceLevelAlignedLiftBoundaryComparisonReport,
) -> Phase7NonparametricSourceLevelAlignedLiftSpectralConditionReport:
    focus_by_random_state = {
        focus.random_state: focus for focus in component_driver_report.focuses
    }
    missing_focus_random_states = [
        random_state
        for random_state in (
            boundary_comparison_report.threshold_crossing_random_state,
            boundary_comparison_report.primary_boundary_random_state,
            boundary_comparison_report.secondary_support_random_state,
        )
        if random_state not in focus_by_random_state
    ]
    if missing_focus_random_states:
        raise KeyError(
            "component-driver report missing aligned-lift spectral-condition focus(es): "
            f"{missing_focus_random_states!r}"
        )

    threshold_crossing_focus = _build_phase7_aligned_lift_spectral_condition_focus(
        focus_by_random_state[
            boundary_comparison_report.threshold_crossing_random_state
        ],
        threshold_member=boundary_comparison_report.threshold_crossing_member,
    )
    primary_boundary_focus = _build_phase7_aligned_lift_spectral_condition_focus(
        focus_by_random_state[boundary_comparison_report.primary_boundary_random_state],
        threshold_member=boundary_comparison_report.primary_boundary_member,
    )
    secondary_support_focus = _build_phase7_aligned_lift_spectral_condition_focus(
        focus_by_random_state[
            boundary_comparison_report.secondary_support_random_state
        ],
        threshold_member=boundary_comparison_report.secondary_support_member,
    )

    recommendation_rationale = (
        f"{threshold_crossing_focus.random_state} and "
        f"{primary_boundary_focus.random_state} preserve nearly identical geometry "
        f"(mean row-to-center cosine "
        f"{threshold_crossing_focus.mean_row_to_center_cosine:.3f} vs "
        f"{primary_boundary_focus.mean_row_to_center_cosine:.3f}) and remain close in "
        "weighted-correlation alignment "
        f"({threshold_crossing_focus.mean_weighted_correlation_to_center:.3f} vs "
        f"{primary_boundary_focus.mean_weighted_correlation_to_center:.3f}), but only "
        f"{threshold_crossing_focus.random_state} adds dominant-mode concentration "
        f"({threshold_crossing_focus.mean_leading_eigen_share:.3f} vs "
        f"{primary_boundary_focus.mean_leading_eigen_share:.3f}); "
        f"{primary_boundary_focus.random_state} therefore remains the closest stable "
        "boundary, while "
        f"{secondary_support_focus.random_state} stays a low-share shoulder support."
    )

    return Phase7NonparametricSourceLevelAlignedLiftSpectralConditionReport(
        oracle_lane=component_driver_report.oracle_lane,
        stage_label="phase7-nonparametric-source-level-aligned-lift-spectral-condition-probe",
        target_n_obs=component_driver_report.target_n_obs,
        reference_n_obs=component_driver_report.reference_n_obs,
        p=component_driver_report.p,
        dgp_name=component_driver_report.dgp_name,
        aligned_lift_family_random_states=(
            boundary_comparison_report.aligned_lift_family_random_states
        ),
        target_grid_label=component_driver_report.target_grid_label,
        threshold_crossing_random_state=threshold_crossing_focus.random_state,
        primary_boundary_random_state=primary_boundary_focus.random_state,
        secondary_support_random_state=secondary_support_focus.random_state,
        threshold_crossing_focus=threshold_crossing_focus,
        primary_boundary_focus=primary_boundary_focus,
        secondary_support_focus=secondary_support_focus,
        mean_pointwise_ratio_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_pointwise_ratio
            - primary_boundary_focus.mean_pointwise_ratio
        ),
        mean_row_to_center_cosine_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_row_to_center_cosine
            - primary_boundary_focus.mean_row_to_center_cosine
        ),
        mean_weighted_correlation_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_weighted_correlation_to_center
            - primary_boundary_focus.mean_weighted_correlation_to_center
        ),
        mean_leading_eigen_share_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_leading_eigen_share
            - primary_boundary_focus.mean_leading_eigen_share
        ),
        center_leading_eigen_share_gap_to_primary_boundary=(
            threshold_crossing_focus.center_leading_eigen_share
            - primary_boundary_focus.center_leading_eigen_share
        ),
        recommendation_rationale=recommendation_rationale,
    )


def run_phase7_nonparametric_source_level_aligned_lift_spectral_condition_probe(
    *,
    neighborhood_random_states: Sequence[int] = tuple(range(296, 321)),
    candidate_random_state: int = 303,
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelAlignedLiftSpectralConditionReport:
    neighborhood_random_states_value = tuple(
        dict.fromkeys(int(value) for value in neighborhood_random_states)
    )
    if not neighborhood_random_states_value:
        raise ValueError("neighborhood_random_states must not be empty")

    normalized_focus_labels = {
        random_state: f"random_state_{random_state}"
        for random_state in neighborhood_random_states_value
    }
    normalized_focus_labels.update(
        {
            303: "dominant_mode_hotspot",
            308: "closest_stable_boundary",
            296: "shoulder_support",
        }
    )
    if focus_labels is not None:
        normalized_focus_labels.update(
            {int(key): str(value) for key, value in focus_labels.items()}
        )

    component_driver_report = (
        run_phase7_nonparametric_source_level_component_driver_probe(
            focus_random_states=neighborhood_random_states_value,
            neighborhood_random_states=neighborhood_random_states_value,
            focus_labels=normalized_focus_labels,
            target_grid_label=target_grid_label,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    threshold_report = (
        build_phase7_nonparametric_source_level_aligned_lift_threshold_report(
            component_driver_report,
            candidate_random_state=candidate_random_state,
        )
    )
    boundary_comparison_report = (
        build_phase7_nonparametric_source_level_aligned_lift_boundary_comparison_report(
            threshold_report
        )
    )
    return (
        build_phase7_nonparametric_source_level_aligned_lift_spectral_condition_report(
            component_driver_report,
            boundary_comparison_report,
        )
    )


def _phase7_aligned_lift_spectral_concentration_signature(
    *,
    mean_top_three_eigen_share: float,
    mean_nonleading_top_three_share: float,
    top_three_eigen_share_range: float,
    leading_eigenvalue_positive_share: float,
) -> str:
    if (
        mean_top_three_eigen_share > 0.95
        and mean_nonleading_top_three_share < 0.05
        and top_three_eigen_share_range < 0.01
    ):
        return "rank_one_lock_in"
    if (
        mean_top_three_eigen_share < 0.20
        and mean_nonleading_top_three_share < 0.15
        and leading_eigenvalue_positive_share > 0.40
    ):
        return "diffuse_multimode_boundary"
    return "multimode_shoulder"


def _build_phase7_aligned_lift_spectral_concentration_focus(
    spectral_focus: Phase7NonparametricSourceLevelAlignedLiftSpectralConditionFocus,
    seed_entry: Phase7NonparametricSourceLevelSeedNeighborhoodEntry,
    *,
    target_grid_label: str,
) -> Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationFocus:
    replay_by_grid_label = {
        replay.grid_label: replay for replay in seed_entry.source_level_replays
    }
    if target_grid_label not in replay_by_grid_label:
        raise KeyError(
            "seed-neighborhood entry missing aligned-lift spectral-concentration "
            f"grid replay: {target_grid_label!r}"
        )

    target_replay = replay_by_grid_label[target_grid_label]
    top_three_eigen_shares = [
        point.top_three_eigen_share for point in target_replay.grid_points
    ]
    nonleading_top_three_shares = [
        point.top_three_eigen_share - point.leading_eigen_share
        for point in target_replay.grid_points
    ]
    center_index = target_replay.center_index
    center_grid_point = target_replay.grid_points[center_index]
    mean_top_three_eigen_share = float(np.mean(top_three_eigen_shares))
    mean_nonleading_top_three_share = float(np.mean(nonleading_top_three_shares))
    top_three_eigen_share_range = float(
        np.max(top_three_eigen_shares) - np.min(top_three_eigen_shares)
    )

    return Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationFocus(
        dgp_name=spectral_focus.dgp_name,
        focus_label=spectral_focus.focus_label,
        random_state=spectral_focus.random_state,
        replication_seed=spectral_focus.replication_seed,
        driver_signature=spectral_focus.driver_signature,
        spectral_concentration_signature=(
            _phase7_aligned_lift_spectral_concentration_signature(
                mean_top_three_eigen_share=mean_top_three_eigen_share,
                mean_nonleading_top_three_share=mean_nonleading_top_three_share,
                top_three_eigen_share_range=top_three_eigen_share_range,
                leading_eigenvalue_positive_share=(
                    target_replay.leading_eigenvalue_positive_share
                ),
            )
        ),
        leading_eigenvalue_positive_share=target_replay.leading_eigenvalue_positive_share,
        mean_leading_eigen_share=spectral_focus.mean_leading_eigen_share,
        mean_top_three_eigen_share=mean_top_three_eigen_share,
        mean_nonleading_top_three_share=mean_nonleading_top_three_share,
        center_leading_eigen_share=spectral_focus.center_leading_eigen_share,
        center_top_three_eigen_share=center_grid_point.top_three_eigen_share,
        center_nonleading_top_three_share=nonleading_top_three_shares[center_index],
        top_three_eigen_share_range=top_three_eigen_share_range,
    )


def build_phase7_nonparametric_source_level_aligned_lift_spectral_concentration_report(
    spectral_condition_report: Phase7NonparametricSourceLevelAlignedLiftSpectralConditionReport,
    seed_neighborhood_report: Phase7NonparametricSourceLevelSeedNeighborhoodReport,
) -> Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationReport:
    threshold_crossing_entry = seed_neighborhood_report.entry(
        spectral_condition_report.threshold_crossing_random_state
    )
    primary_boundary_entry = seed_neighborhood_report.entry(
        spectral_condition_report.primary_boundary_random_state
    )
    secondary_support_entry = seed_neighborhood_report.entry(
        spectral_condition_report.secondary_support_random_state
    )

    threshold_crossing_focus = _build_phase7_aligned_lift_spectral_concentration_focus(
        spectral_condition_report.threshold_crossing_focus,
        threshold_crossing_entry,
        target_grid_label=spectral_condition_report.target_grid_label,
    )
    primary_boundary_focus = _build_phase7_aligned_lift_spectral_concentration_focus(
        spectral_condition_report.primary_boundary_focus,
        primary_boundary_entry,
        target_grid_label=spectral_condition_report.target_grid_label,
    )
    secondary_support_focus = _build_phase7_aligned_lift_spectral_concentration_focus(
        spectral_condition_report.secondary_support_focus,
        secondary_support_entry,
        target_grid_label=spectral_condition_report.target_grid_label,
    )

    recommendation_rationale = (
        f"{threshold_crossing_focus.random_state} and "
        f"{primary_boundary_focus.random_state} remain within the same aligned-lift "
        "geometry, and even their leading-eigenvalue-positive shares stay relatively "
        f"close ({threshold_crossing_focus.leading_eigenvalue_positive_share:.3f} vs "
        f"{primary_boundary_focus.leading_eigenvalue_positive_share:.3f}); but only "
        f"{threshold_crossing_focus.random_state} collapses the micro-center slice into "
        "rank-one lock-in, with mean top-three eigen share "
        f"{threshold_crossing_focus.mean_top_three_eigen_share:.3f} and residual "
        "non-leading top-three share "
        f"{threshold_crossing_focus.mean_nonleading_top_three_share:.3f}. "
        f"{primary_boundary_focus.random_state} therefore remains a diffuse multi-mode "
        "boundary rather than a near-crossing lock-in, while "
        f"{secondary_support_focus.random_state} stays a multi-mode shoulder."
    )

    return Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationReport(
        oracle_lane=spectral_condition_report.oracle_lane,
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-spectral-concentration-probe"
        ),
        target_n_obs=spectral_condition_report.target_n_obs,
        reference_n_obs=spectral_condition_report.reference_n_obs,
        p=spectral_condition_report.p,
        dgp_name=spectral_condition_report.dgp_name,
        aligned_lift_family_random_states=(
            spectral_condition_report.aligned_lift_family_random_states
        ),
        target_grid_label=spectral_condition_report.target_grid_label,
        threshold_crossing_random_state=(
            spectral_condition_report.threshold_crossing_random_state
        ),
        primary_boundary_random_state=(
            spectral_condition_report.primary_boundary_random_state
        ),
        secondary_support_random_state=(
            spectral_condition_report.secondary_support_random_state
        ),
        threshold_crossing_focus=threshold_crossing_focus,
        primary_boundary_focus=primary_boundary_focus,
        secondary_support_focus=secondary_support_focus,
        leading_eigenvalue_positive_share_gap_to_primary_boundary=(
            threshold_crossing_focus.leading_eigenvalue_positive_share
            - primary_boundary_focus.leading_eigenvalue_positive_share
        ),
        mean_top_three_eigen_share_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_top_three_eigen_share
            - primary_boundary_focus.mean_top_three_eigen_share
        ),
        mean_nonleading_top_three_share_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_nonleading_top_three_share
            - primary_boundary_focus.mean_nonleading_top_three_share
        ),
        center_nonleading_top_three_share_gap_to_primary_boundary=(
            threshold_crossing_focus.center_nonleading_top_three_share
            - primary_boundary_focus.center_nonleading_top_three_share
        ),
        recommendation_rationale=recommendation_rationale,
    )


def run_phase7_nonparametric_source_level_aligned_lift_spectral_concentration_probe(
    *,
    neighborhood_random_states: Sequence[int] = tuple(range(296, 321)),
    candidate_random_state: int = 303,
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationReport:
    spectral_condition_report = (
        run_phase7_nonparametric_source_level_aligned_lift_spectral_condition_probe(
            neighborhood_random_states=neighborhood_random_states,
            candidate_random_state=candidate_random_state,
            focus_labels=focus_labels,
            target_grid_label=target_grid_label,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    seed_neighborhood_report = (
        run_phase7_nonparametric_source_level_seed_neighborhood_probe(
            neighborhood_random_states=neighborhood_random_states,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    return build_phase7_nonparametric_source_level_aligned_lift_spectral_concentration_report(
        spectral_condition_report,
        seed_neighborhood_report,
    )


def _phase7_aligned_lift_spectral_effective_rank_signature(
    *,
    mean_effective_positive_mode_count: float,
    effective_positive_mode_count_range: float,
    mean_tail_share_outside_top_three: float,
) -> str:
    if (
        mean_effective_positive_mode_count < 1.1
        and effective_positive_mode_count_range < 0.1
    ):
        return "single_effective_mode"
    if (
        mean_effective_positive_mode_count < 3.0
        and mean_tail_share_outside_top_three > 0.8
    ):
        return "tail_skew_boundary"
    return "broad_support_shoulder"


def _build_phase7_aligned_lift_spectral_effective_rank_focus(
    concentration_focus: Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationFocus,
    seed_entry: Phase7NonparametricSourceLevelSeedNeighborhoodEntry,
    *,
    target_grid_label: str,
) -> Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankFocus:
    replay_by_grid_label = {
        replay.grid_label: replay for replay in seed_entry.source_level_replays
    }
    if target_grid_label not in replay_by_grid_label:
        raise KeyError(
            "seed-neighborhood entry missing aligned-lift spectral-effective-rank "
            f"grid replay: {target_grid_label!r}"
        )

    target_replay = replay_by_grid_label[target_grid_label]
    effective_positive_mode_counts = [
        point.effective_positive_mode_count for point in target_replay.grid_points
    ]
    tail_shares_outside_top_three = [
        1.0 - point.top_three_eigen_share for point in target_replay.grid_points
    ]
    center_index = target_replay.center_index

    mean_effective_positive_mode_count = float(np.mean(effective_positive_mode_counts))
    effective_positive_mode_count_range = float(
        np.max(effective_positive_mode_counts) - np.min(effective_positive_mode_counts)
    )
    mean_tail_share_outside_top_three = float(np.mean(tail_shares_outside_top_three))

    return Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankFocus(
        dgp_name=concentration_focus.dgp_name,
        focus_label=concentration_focus.focus_label,
        random_state=concentration_focus.random_state,
        replication_seed=concentration_focus.replication_seed,
        driver_signature=concentration_focus.driver_signature,
        spectral_effective_rank_signature=(
            _phase7_aligned_lift_spectral_effective_rank_signature(
                mean_effective_positive_mode_count=(mean_effective_positive_mode_count),
                effective_positive_mode_count_range=(
                    effective_positive_mode_count_range
                ),
                mean_tail_share_outside_top_three=(mean_tail_share_outside_top_three),
            )
        ),
        mean_effective_positive_mode_count=mean_effective_positive_mode_count,
        center_effective_positive_mode_count=(
            effective_positive_mode_counts[center_index]
        ),
        effective_positive_mode_count_range=(effective_positive_mode_count_range),
        mean_tail_share_outside_top_three=mean_tail_share_outside_top_three,
        center_tail_share_outside_top_three=(
            tail_shares_outside_top_three[center_index]
        ),
    )


def build_phase7_nonparametric_source_level_aligned_lift_spectral_effective_rank_report(
    spectral_concentration_report: Phase7NonparametricSourceLevelAlignedLiftSpectralConcentrationReport,
    seed_neighborhood_report: Phase7NonparametricSourceLevelSeedNeighborhoodReport,
) -> Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankReport:
    threshold_crossing_entry = seed_neighborhood_report.entry(
        spectral_concentration_report.threshold_crossing_random_state
    )
    primary_boundary_entry = seed_neighborhood_report.entry(
        spectral_concentration_report.primary_boundary_random_state
    )
    secondary_support_entry = seed_neighborhood_report.entry(
        spectral_concentration_report.secondary_support_random_state
    )

    threshold_crossing_focus = _build_phase7_aligned_lift_spectral_effective_rank_focus(
        spectral_concentration_report.threshold_crossing_focus,
        threshold_crossing_entry,
        target_grid_label=spectral_concentration_report.target_grid_label,
    )
    primary_boundary_focus = _build_phase7_aligned_lift_spectral_effective_rank_focus(
        spectral_concentration_report.primary_boundary_focus,
        primary_boundary_entry,
        target_grid_label=spectral_concentration_report.target_grid_label,
    )
    secondary_support_focus = _build_phase7_aligned_lift_spectral_effective_rank_focus(
        spectral_concentration_report.secondary_support_focus,
        secondary_support_entry,
        target_grid_label=spectral_concentration_report.target_grid_label,
    )

    recommendation_rationale = (
        f"{threshold_crossing_focus.random_state} does not just keep the highest "
        "top-three concentration; it collapses the micro-center slice to a single "
        "effective mode, with mean effective_positive_mode_count "
        f"{threshold_crossing_focus.mean_effective_positive_mode_count:.3f} and range "
        f"{threshold_crossing_focus.effective_positive_mode_count_range:.3f}. "
        f"In other words, {threshold_crossing_focus.random_state} has already entered "
        "single effective mode lock-in. "
        f"{primary_boundary_focus.random_state} remains a tail-skew boundary instead, "
        "because its mean effective-positive-mode count stays "
        f"{primary_boundary_focus.mean_effective_positive_mode_count:.3f} while the "
        "mean tail share outside the top three modes stays "
        f"{primary_boundary_focus.mean_tail_share_outside_top_three:.3f}. "
        f"{secondary_support_focus.random_state} therefore stays a broad support "
        "shoulder rather than a near-crossing lock-in."
    )

    return Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankReport(
        oracle_lane=spectral_concentration_report.oracle_lane,
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-spectral-effective-rank-probe"
        ),
        target_n_obs=spectral_concentration_report.target_n_obs,
        reference_n_obs=spectral_concentration_report.reference_n_obs,
        p=spectral_concentration_report.p,
        dgp_name=spectral_concentration_report.dgp_name,
        aligned_lift_family_random_states=(
            spectral_concentration_report.aligned_lift_family_random_states
        ),
        target_grid_label=spectral_concentration_report.target_grid_label,
        threshold_crossing_random_state=(
            spectral_concentration_report.threshold_crossing_random_state
        ),
        primary_boundary_random_state=(
            spectral_concentration_report.primary_boundary_random_state
        ),
        secondary_support_random_state=(
            spectral_concentration_report.secondary_support_random_state
        ),
        threshold_crossing_focus=threshold_crossing_focus,
        primary_boundary_focus=primary_boundary_focus,
        secondary_support_focus=secondary_support_focus,
        mean_effective_positive_mode_count_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_effective_positive_mode_count
            - primary_boundary_focus.mean_effective_positive_mode_count
        ),
        mean_tail_share_outside_top_three_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_tail_share_outside_top_three
            - primary_boundary_focus.mean_tail_share_outside_top_three
        ),
        recommendation_rationale=recommendation_rationale,
    )


def run_phase7_nonparametric_source_level_aligned_lift_spectral_effective_rank_probe(
    *,
    neighborhood_random_states: Sequence[int] = tuple(range(296, 321)),
    candidate_random_state: int = 303,
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankReport:
    spectral_concentration_report = (
        run_phase7_nonparametric_source_level_aligned_lift_spectral_concentration_probe(
            neighborhood_random_states=neighborhood_random_states,
            candidate_random_state=candidate_random_state,
            focus_labels=focus_labels,
            target_grid_label=target_grid_label,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    seed_neighborhood_report = (
        run_phase7_nonparametric_source_level_seed_neighborhood_probe(
            neighborhood_random_states=neighborhood_random_states,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    return build_phase7_nonparametric_source_level_aligned_lift_spectral_effective_rank_report(
        spectral_concentration_report,
        seed_neighborhood_report,
    )


def _phase7_aligned_lift_dominant_mode_persistence_signature(
    *,
    dominant_mode_switch_count: int,
    persistent_dominant_mode_index: int,
    mean_second_mode_share: float,
    mean_nonleading_mass: float,
) -> str:
    if dominant_mode_switch_count != 0:
        return "mode_switching_boundary"
    if (
        persistent_dominant_mode_index == 0
        and mean_second_mode_share < 0.03
        and mean_nonleading_mass < 0.03
    ):
        return "persistent_single_mode_collapse"
    if mean_nonleading_mass < 0.40:
        return "persistent_tail_boundary"
    return "persistent_broad_shoulder"


def _build_phase7_aligned_lift_dominant_mode_persistence_focus(
    effective_rank_focus: Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankFocus,
    seed_entry: Phase7NonparametricSourceLevelSeedNeighborhoodEntry,
    *,
    target_grid_label: str,
) -> Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceFocus:
    replay_by_grid_label = {
        replay.grid_label: replay for replay in seed_entry.source_level_replays
    }
    if target_grid_label not in replay_by_grid_label:
        raise KeyError(
            "seed-neighborhood entry missing aligned-lift dominant-mode-persistence "
            f"grid replay: {target_grid_label!r}"
        )

    target_replay = replay_by_grid_label[target_grid_label]
    dominant_mode_index_sequence = tuple(
        int(point.dominant_mode_index) for point in target_replay.grid_points
    )
    unique_dominant_modes = sorted(set(dominant_mode_index_sequence))
    dominant_mode_switch_count = max(0, len(unique_dominant_modes) - 1)
    persistent_dominant_mode_index = (
        int(unique_dominant_modes[0]) if unique_dominant_modes else -1
    )

    second_mode_shares = [
        point.second_mode_share for point in target_replay.grid_points
    ]
    nonleading_mass = [
        1.0 - point.dominant_mode_share for point in target_replay.grid_points
    ]
    center_index = target_replay.center_index

    mean_second_mode_share = float(np.mean(second_mode_shares))
    second_mode_share_range = float(
        np.max(second_mode_shares) - np.min(second_mode_shares)
    )
    mean_nonleading_mass = float(np.mean(nonleading_mass))
    nonleading_mass_range = float(np.max(nonleading_mass) - np.min(nonleading_mass))

    return Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceFocus(
        dgp_name=effective_rank_focus.dgp_name,
        focus_label=effective_rank_focus.focus_label,
        random_state=effective_rank_focus.random_state,
        replication_seed=effective_rank_focus.replication_seed,
        driver_signature=effective_rank_focus.driver_signature,
        spectral_effective_rank_signature=(
            effective_rank_focus.spectral_effective_rank_signature
        ),
        dominant_mode_persistence_signature=(
            _phase7_aligned_lift_dominant_mode_persistence_signature(
                dominant_mode_switch_count=dominant_mode_switch_count,
                persistent_dominant_mode_index=persistent_dominant_mode_index,
                mean_second_mode_share=mean_second_mode_share,
                mean_nonleading_mass=mean_nonleading_mass,
            )
        ),
        persistent_dominant_mode_index=persistent_dominant_mode_index,
        dominant_mode_index_sequence=dominant_mode_index_sequence,
        dominant_mode_switch_count=dominant_mode_switch_count,
        mean_second_mode_share=mean_second_mode_share,
        center_second_mode_share=second_mode_shares[center_index],
        second_mode_share_range=second_mode_share_range,
        mean_nonleading_mass=mean_nonleading_mass,
        center_nonleading_mass=nonleading_mass[center_index],
        nonleading_mass_range=nonleading_mass_range,
    )


def build_phase7_nonparametric_source_level_aligned_lift_dominant_mode_persistence_report(
    effective_rank_report: Phase7NonparametricSourceLevelAlignedLiftSpectralEffectiveRankReport,
    seed_neighborhood_report: Phase7NonparametricSourceLevelSeedNeighborhoodReport,
) -> Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceReport:
    threshold_crossing_entry = seed_neighborhood_report.entry(
        effective_rank_report.threshold_crossing_random_state
    )
    primary_boundary_entry = seed_neighborhood_report.entry(
        effective_rank_report.primary_boundary_random_state
    )
    secondary_support_entry = seed_neighborhood_report.entry(
        effective_rank_report.secondary_support_random_state
    )

    threshold_crossing_focus = (
        _build_phase7_aligned_lift_dominant_mode_persistence_focus(
            effective_rank_report.threshold_crossing_focus,
            threshold_crossing_entry,
            target_grid_label=effective_rank_report.target_grid_label,
        )
    )
    primary_boundary_focus = _build_phase7_aligned_lift_dominant_mode_persistence_focus(
        effective_rank_report.primary_boundary_focus,
        primary_boundary_entry,
        target_grid_label=effective_rank_report.target_grid_label,
    )
    secondary_support_focus = (
        _build_phase7_aligned_lift_dominant_mode_persistence_focus(
            effective_rank_report.secondary_support_focus,
            secondary_support_entry,
            target_grid_label=effective_rank_report.target_grid_label,
        )
    )

    recommendation_rationale = (
        "This aligned-lift split is no longer about mode switching across the "
        "micro-center grid. All three exact members keep dominant mode persistence "
        f"(switch counts {threshold_crossing_focus.dominant_mode_switch_count}/"
        f"{primary_boundary_focus.dominant_mode_switch_count}/"
        f"{secondary_support_focus.dominant_mode_switch_count}), but only "
        f"{threshold_crossing_focus.random_state} collapses onto persistent dominant "
        f"mode index {threshold_crossing_focus.persistent_dominant_mode_index} while "
        "driving mean second-mode share and mean nonleading mass down to "
        f"{threshold_crossing_focus.mean_second_mode_share:.3f} and "
        f"{threshold_crossing_focus.mean_nonleading_mass:.3f}. That is the "
        "persistent single-mode collapse. "
        f"{primary_boundary_focus.random_state} stays a persistent tail boundary "
        f"instead, because its persistent dominant mode index remains "
        f"{primary_boundary_focus.persistent_dominant_mode_index} with mean second-mode "
        f"share {primary_boundary_focus.mean_second_mode_share:.3f} and mean "
        f"nonleading mass {primary_boundary_focus.mean_nonleading_mass:.3f}. "
        f"{secondary_support_focus.random_state} therefore stays a persistent broad "
        "shoulder rather than a near-collapse boundary."
    )

    return Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceReport(
        oracle_lane=effective_rank_report.oracle_lane,
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-dominant-mode-persistence-probe"
        ),
        target_n_obs=effective_rank_report.target_n_obs,
        reference_n_obs=effective_rank_report.reference_n_obs,
        p=effective_rank_report.p,
        dgp_name=effective_rank_report.dgp_name,
        aligned_lift_family_random_states=(
            effective_rank_report.aligned_lift_family_random_states
        ),
        target_grid_label=effective_rank_report.target_grid_label,
        threshold_crossing_random_state=(
            effective_rank_report.threshold_crossing_random_state
        ),
        primary_boundary_random_state=(
            effective_rank_report.primary_boundary_random_state
        ),
        secondary_support_random_state=(
            effective_rank_report.secondary_support_random_state
        ),
        threshold_crossing_focus=threshold_crossing_focus,
        primary_boundary_focus=primary_boundary_focus,
        secondary_support_focus=secondary_support_focus,
        mean_second_mode_share_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_second_mode_share
            - primary_boundary_focus.mean_second_mode_share
        ),
        mean_nonleading_mass_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_nonleading_mass
            - primary_boundary_focus.mean_nonleading_mass
        ),
        recommendation_rationale=recommendation_rationale,
    )


def run_phase7_nonparametric_source_level_aligned_lift_dominant_mode_persistence_probe(
    *,
    neighborhood_random_states: Sequence[int] = tuple(range(296, 321)),
    candidate_random_state: int = 303,
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceReport:
    effective_rank_report = run_phase7_nonparametric_source_level_aligned_lift_spectral_effective_rank_probe(
        neighborhood_random_states=neighborhood_random_states,
        candidate_random_state=candidate_random_state,
        focus_labels=focus_labels,
        target_grid_label=target_grid_label,
        n_boot=n_boot,
        target_n_obs=target_n_obs,
        reference_n_obs=reference_n_obs,
        p=p,
        local_grids=local_grids,
        hotspot_center=hotspot_center,
        dgp_name=dgp_name,
        center_random_state=center_random_state,
        designs=designs,
    )
    seed_neighborhood_report = (
        run_phase7_nonparametric_source_level_seed_neighborhood_probe(
            neighborhood_random_states=neighborhood_random_states,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    return build_phase7_nonparametric_source_level_aligned_lift_dominant_mode_persistence_report(
        effective_rank_report,
        seed_neighborhood_report,
    )


def _phase7_safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("inf")
    return float(numerator / denominator)


_PHASE7_SHARE_SEPARATION_COLLAPSE_GAP_THRESHOLD = 0.90
_PHASE7_SHARE_SEPARATION_COLLAPSE_RATIO_THRESHOLD = 50.0
_PHASE7_SHARE_SEPARATION_COLLAPSE_CENTER_RATIO_THRESHOLD = 100.0
_PHASE7_FINITE_MARGIN_BOUNDARY_GAP_THRESHOLD = 0.50
_PHASE7_FINITE_MARGIN_BOUNDARY_RATIO_THRESHOLD = 5.0


def _phase7_aligned_lift_dominant_mode_margin_signature(
    *,
    mean_dominant_minus_second_gap: float,
    mean_dominant_to_second_ratio: float,
    center_dominant_to_second_ratio: float,
) -> str:
    if (
        mean_dominant_minus_second_gap > _PHASE7_SHARE_SEPARATION_COLLAPSE_GAP_THRESHOLD
        and mean_dominant_to_second_ratio
        > _PHASE7_SHARE_SEPARATION_COLLAPSE_RATIO_THRESHOLD
        and center_dominant_to_second_ratio
        > _PHASE7_SHARE_SEPARATION_COLLAPSE_CENTER_RATIO_THRESHOLD
    ):
        return "share_separation_collapse"
    if (
        mean_dominant_minus_second_gap > _PHASE7_FINITE_MARGIN_BOUNDARY_GAP_THRESHOLD
        and mean_dominant_to_second_ratio
        > _PHASE7_FINITE_MARGIN_BOUNDARY_RATIO_THRESHOLD
    ):
        return "finite_margin_boundary"
    return "broad_margin_shoulder"


def _build_phase7_aligned_lift_dominant_mode_margin_focus(
    persistence_focus: Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceFocus,
) -> Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginFocus:
    mean_dominant_share = 1.0 - persistence_focus.mean_nonleading_mass
    center_dominant_share = 1.0 - persistence_focus.center_nonleading_mass
    mean_dominant_minus_second_gap = (
        mean_dominant_share - persistence_focus.mean_second_mode_share
    )
    center_dominant_minus_second_gap = (
        center_dominant_share - persistence_focus.center_second_mode_share
    )
    mean_dominant_to_second_ratio = _phase7_safe_ratio(
        mean_dominant_share,
        persistence_focus.mean_second_mode_share,
    )
    center_dominant_to_second_ratio = _phase7_safe_ratio(
        center_dominant_share,
        persistence_focus.center_second_mode_share,
    )

    return Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginFocus(
        dgp_name=persistence_focus.dgp_name,
        focus_label=persistence_focus.focus_label,
        random_state=persistence_focus.random_state,
        replication_seed=persistence_focus.replication_seed,
        driver_signature=persistence_focus.driver_signature,
        spectral_effective_rank_signature=(
            persistence_focus.spectral_effective_rank_signature
        ),
        dominant_mode_persistence_signature=(
            persistence_focus.dominant_mode_persistence_signature
        ),
        dominant_mode_margin_signature=(
            _phase7_aligned_lift_dominant_mode_margin_signature(
                mean_dominant_minus_second_gap=mean_dominant_minus_second_gap,
                mean_dominant_to_second_ratio=mean_dominant_to_second_ratio,
                center_dominant_to_second_ratio=center_dominant_to_second_ratio,
            )
        ),
        persistent_dominant_mode_index=(
            persistence_focus.persistent_dominant_mode_index
        ),
        mean_dominant_share=mean_dominant_share,
        center_dominant_share=center_dominant_share,
        mean_second_mode_share=persistence_focus.mean_second_mode_share,
        center_second_mode_share=persistence_focus.center_second_mode_share,
        mean_nonleading_mass=persistence_focus.mean_nonleading_mass,
        center_nonleading_mass=persistence_focus.center_nonleading_mass,
        mean_dominant_minus_second_gap=mean_dominant_minus_second_gap,
        center_dominant_minus_second_gap=center_dominant_minus_second_gap,
        mean_dominant_to_second_ratio=mean_dominant_to_second_ratio,
        center_dominant_to_second_ratio=center_dominant_to_second_ratio,
    )


def build_phase7_nonparametric_source_level_aligned_lift_dominant_mode_margin_report(
    dominant_mode_persistence_report: Phase7NonparametricSourceLevelAlignedLiftDominantModePersistenceReport,
) -> Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginReport:
    threshold_crossing_focus = _build_phase7_aligned_lift_dominant_mode_margin_focus(
        dominant_mode_persistence_report.threshold_crossing_focus
    )
    primary_boundary_focus = _build_phase7_aligned_lift_dominant_mode_margin_focus(
        dominant_mode_persistence_report.primary_boundary_focus
    )
    secondary_support_focus = _build_phase7_aligned_lift_dominant_mode_margin_focus(
        dominant_mode_persistence_report.secondary_support_focus
    )

    recommendation_rationale = (
        "This dominant-mode margin replay is a more stable source-level oracle than "
        "the raw mode index labels. "
        f"{threshold_crossing_focus.random_state} now separates as share-separation "
        "collapse because its mean dominant-minus-second gap reaches "
        f"{threshold_crossing_focus.mean_dominant_minus_second_gap:.3f} and its mean "
        "dominant-to-second ratio reaches "
        f"{threshold_crossing_focus.mean_dominant_to_second_ratio:.3f}. "
        f"{primary_boundary_focus.random_state} remains a finite-margin boundary "
        f"instead at gap {primary_boundary_focus.mean_dominant_minus_second_gap:.3f} "
        "and ratio "
        f"{primary_boundary_focus.mean_dominant_to_second_ratio:.3f}, while "
        f"{secondary_support_focus.random_state} stays a broad-margin shoulder "
        f"at ratio {secondary_support_focus.mean_dominant_to_second_ratio:.3f}."
    )

    return Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginReport(
        oracle_lane=dominant_mode_persistence_report.oracle_lane,
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-dominant-mode-margin-probe"
        ),
        target_n_obs=dominant_mode_persistence_report.target_n_obs,
        reference_n_obs=dominant_mode_persistence_report.reference_n_obs,
        p=dominant_mode_persistence_report.p,
        dgp_name=dominant_mode_persistence_report.dgp_name,
        aligned_lift_family_random_states=(
            dominant_mode_persistence_report.aligned_lift_family_random_states
        ),
        target_grid_label=dominant_mode_persistence_report.target_grid_label,
        threshold_crossing_random_state=(
            dominant_mode_persistence_report.threshold_crossing_random_state
        ),
        primary_boundary_random_state=(
            dominant_mode_persistence_report.primary_boundary_random_state
        ),
        secondary_support_random_state=(
            dominant_mode_persistence_report.secondary_support_random_state
        ),
        threshold_crossing_focus=threshold_crossing_focus,
        primary_boundary_focus=primary_boundary_focus,
        secondary_support_focus=secondary_support_focus,
        mean_dominant_minus_second_gap_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_dominant_minus_second_gap
            - primary_boundary_focus.mean_dominant_minus_second_gap
        ),
        mean_dominant_to_second_ratio_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_dominant_to_second_ratio
            - primary_boundary_focus.mean_dominant_to_second_ratio
        ),
        center_dominant_to_second_ratio_gap_to_primary_boundary=(
            threshold_crossing_focus.center_dominant_to_second_ratio
            - primary_boundary_focus.center_dominant_to_second_ratio
        ),
        recommendation_rationale=recommendation_rationale,
    )


def _build_phase7_aligned_lift_margin_signature_slack_focus(
    margin_focus: Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginFocus,
) -> Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackFocus:
    return Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackFocus(
        dgp_name=margin_focus.dgp_name,
        focus_label=margin_focus.focus_label,
        random_state=margin_focus.random_state,
        replication_seed=margin_focus.replication_seed,
        dominant_mode_margin_signature=margin_focus.dominant_mode_margin_signature,
        mean_dominant_minus_second_gap=margin_focus.mean_dominant_minus_second_gap,
        mean_dominant_to_second_ratio=margin_focus.mean_dominant_to_second_ratio,
        center_dominant_to_second_ratio=(margin_focus.center_dominant_to_second_ratio),
        collapse_gap_slack=(
            margin_focus.mean_dominant_minus_second_gap
            - _PHASE7_SHARE_SEPARATION_COLLAPSE_GAP_THRESHOLD
        ),
        collapse_ratio_slack=(
            margin_focus.mean_dominant_to_second_ratio
            - _PHASE7_SHARE_SEPARATION_COLLAPSE_RATIO_THRESHOLD
        ),
        collapse_center_ratio_slack=(
            margin_focus.center_dominant_to_second_ratio
            - _PHASE7_SHARE_SEPARATION_COLLAPSE_CENTER_RATIO_THRESHOLD
        ),
        finite_gap_slack=(
            margin_focus.mean_dominant_minus_second_gap
            - _PHASE7_FINITE_MARGIN_BOUNDARY_GAP_THRESHOLD
        ),
        finite_ratio_slack=(
            margin_focus.mean_dominant_to_second_ratio
            - _PHASE7_FINITE_MARGIN_BOUNDARY_RATIO_THRESHOLD
        ),
    )


def build_phase7_nonparametric_source_level_aligned_lift_margin_signature_slack_report(
    dominant_mode_margin_report: Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginReport,
) -> Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackReport:
    threshold_crossing_focus = _build_phase7_aligned_lift_margin_signature_slack_focus(
        dominant_mode_margin_report.threshold_crossing_focus
    )
    primary_boundary_focus = _build_phase7_aligned_lift_margin_signature_slack_focus(
        dominant_mode_margin_report.primary_boundary_focus
    )
    secondary_support_focus = _build_phase7_aligned_lift_margin_signature_slack_focus(
        dominant_mode_margin_report.secondary_support_focus
    )
    canonical_slack_digest = (
        threshold_crossing_focus.canonical_digest_line(),
        primary_boundary_focus.canonical_digest_line(),
        secondary_support_focus.canonical_digest_line(),
    )
    recommendation_rationale = (
        "This helper turns the dominant-mode margin replay into a threshold-slack "
        "surface that is easier to hand off. "
        f"{threshold_crossing_focus.random_state} keeps positive collapse slack "
        f"({threshold_crossing_focus.collapse_gap_slack:+.3f}, "
        f"{threshold_crossing_focus.collapse_ratio_slack:+.3f}, "
        f"{threshold_crossing_focus.collapse_center_ratio_slack:+.3f}), while "
        f"{primary_boundary_focus.random_state} retains only finite-threshold slack "
        f"({primary_boundary_focus.finite_gap_slack:+.3f}, "
        f"{primary_boundary_focus.finite_ratio_slack:+.3f}) and "
        f"{secondary_support_focus.random_state} stays sub-finite on both metrics "
        f"({secondary_support_focus.finite_gap_slack:+.3f}, "
        f"{secondary_support_focus.finite_ratio_slack:+.3f})."
    )
    return Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackReport(
        oracle_lane=dominant_mode_margin_report.oracle_lane,
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-margin-signature-slack-probe"
        ),
        target_n_obs=dominant_mode_margin_report.target_n_obs,
        reference_n_obs=dominant_mode_margin_report.reference_n_obs,
        p=dominant_mode_margin_report.p,
        dgp_name=dominant_mode_margin_report.dgp_name,
        aligned_lift_family_random_states=(
            dominant_mode_margin_report.aligned_lift_family_random_states
        ),
        target_grid_label=dominant_mode_margin_report.target_grid_label,
        collapse_gap_threshold=_PHASE7_SHARE_SEPARATION_COLLAPSE_GAP_THRESHOLD,
        collapse_ratio_threshold=_PHASE7_SHARE_SEPARATION_COLLAPSE_RATIO_THRESHOLD,
        collapse_center_ratio_threshold=(
            _PHASE7_SHARE_SEPARATION_COLLAPSE_CENTER_RATIO_THRESHOLD
        ),
        finite_gap_threshold=_PHASE7_FINITE_MARGIN_BOUNDARY_GAP_THRESHOLD,
        finite_ratio_threshold=_PHASE7_FINITE_MARGIN_BOUNDARY_RATIO_THRESHOLD,
        threshold_crossing_focus=threshold_crossing_focus,
        primary_boundary_focus=primary_boundary_focus,
        secondary_support_focus=secondary_support_focus,
        canonical_slack_digest=canonical_slack_digest,
        recommendation_rationale=recommendation_rationale,
    )


def run_phase7_nonparametric_source_level_aligned_lift_dominant_mode_margin_probe(
    *,
    neighborhood_random_states: Sequence[int] = tuple(range(296, 321)),
    candidate_random_state: int = 303,
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginReport:
    from hddid.aligned_lift_dominant_mode_margin import (
        is_default_aligned_lift_dominant_mode_margin_request,
        run_canonical_phase7_nonparametric_source_level_aligned_lift_dominant_mode_margin_probe,
    )

    if is_default_aligned_lift_dominant_mode_margin_request(
        neighborhood_random_states=neighborhood_random_states,
        candidate_random_state=candidate_random_state,
        focus_labels=focus_labels,
        target_grid_label=target_grid_label,
        n_boot=n_boot,
        target_n_obs=target_n_obs,
        reference_n_obs=reference_n_obs,
        p=p,
        local_grids=local_grids,
        hotspot_center=hotspot_center,
        dgp_name=dgp_name,
        center_random_state=center_random_state,
        designs=designs,
    ):
        return run_canonical_phase7_nonparametric_source_level_aligned_lift_dominant_mode_margin_probe()

    dominant_mode_persistence_report = run_phase7_nonparametric_source_level_aligned_lift_dominant_mode_persistence_probe(
        neighborhood_random_states=neighborhood_random_states,
        candidate_random_state=candidate_random_state,
        focus_labels=focus_labels,
        target_grid_label=target_grid_label,
        n_boot=n_boot,
        target_n_obs=target_n_obs,
        reference_n_obs=reference_n_obs,
        p=p,
        local_grids=local_grids,
        hotspot_center=hotspot_center,
        dgp_name=dgp_name,
        center_random_state=center_random_state,
        designs=designs,
    )
    return build_phase7_nonparametric_source_level_aligned_lift_dominant_mode_margin_report(
        dominant_mode_persistence_report
    )


def run_phase7_nonparametric_source_level_aligned_lift_margin_signature_slack_probe(
    *,
    neighborhood_random_states: Sequence[int] = tuple(range(296, 321)),
    candidate_random_state: int = 303,
    focus_labels: Mapping[int, str] | None = None,
    target_grid_label: str = "micro_center_grid",
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    local_grids: Mapping[str, Sequence[float]] | None = None,
    hotspot_center: float = 0.15,
    dgp_name: str = "DGP2",
    center_random_state: int = 303,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackReport:
    from hddid.aligned_lift_dominant_mode_margin import (
        is_default_aligned_lift_dominant_mode_margin_request,
        run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_signature_slack_probe,
    )

    if is_default_aligned_lift_dominant_mode_margin_request(
        neighborhood_random_states=neighborhood_random_states,
        candidate_random_state=candidate_random_state,
        focus_labels=focus_labels,
        target_grid_label=target_grid_label,
        n_boot=n_boot,
        target_n_obs=target_n_obs,
        reference_n_obs=reference_n_obs,
        p=p,
        local_grids=local_grids,
        hotspot_center=hotspot_center,
        dgp_name=dgp_name,
        center_random_state=center_random_state,
        designs=designs,
    ):
        return run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_signature_slack_probe()

    dominant_mode_margin_report = (
        run_phase7_nonparametric_source_level_aligned_lift_dominant_mode_margin_probe(
            neighborhood_random_states=neighborhood_random_states,
            candidate_random_state=candidate_random_state,
            focus_labels=focus_labels,
            target_grid_label=target_grid_label,
            n_boot=n_boot,
            target_n_obs=target_n_obs,
            reference_n_obs=reference_n_obs,
            p=p,
            local_grids=local_grids,
            hotspot_center=hotspot_center,
            dgp_name=dgp_name,
            center_random_state=center_random_state,
            designs=designs,
        )
    )
    return build_phase7_nonparametric_source_level_aligned_lift_margin_signature_slack_report(
        dominant_mode_margin_report
    )


def run_phase7_nonparametric_off_integer_probe(
    *,
    random_states: Sequence[int] = (101, 202, 303),
    n_boot: int = 64,
    target_n_obs: int = 200,
    reference_n_obs: int = 500,
    p: int = 50,
    off_integer_grids: Mapping[str, Sequence[float]] | None = None,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> Phase7NonparametricOffIntegerReplayReport:
    target_n_obs_value = int(target_n_obs)
    reference_n_obs_value = int(reference_n_obs)
    p_value = int(p)
    default_designs = (
        default_phase7_runtime_probe_designs()
        if (target_n_obs_value, reference_n_obs_value, p_value) == (200, 500, 50)
        else default_phase7_nonparametric_calibration_probe_designs()
    )
    design_sequence = tuple(designs or default_designs)
    object_report = run_phase7_nonparametric_object_probe(
        random_states=random_states,
        n_boot=n_boot,
        target_n_obs=target_n_obs_value,
        reference_n_obs=reference_n_obs_value,
        p=p_value,
        designs=design_sequence,
    )
    return build_phase7_nonparametric_off_integer_replay_report(
        object_report,
        designs=design_sequence,
        off_integer_grids=(
            off_integer_grids
            if off_integer_grids is not None
            else {
                "contrast_grid": (0.1, 0.3, 0.7),
                "near_zero_grid": (0.05, 0.15, 0.25),
            }
        ),
        n_boot=n_boot,
    )


def run_phase7_monte_carlo_runtime_probe(
    *,
    random_states: Sequence[int] = (101, 202, 303),
    n_boot: int = 64,
    designs: Sequence[MonteCarloDesign] | None = None,
) -> MonteCarloRuntimeProbeReport:
    design_sequence = tuple(designs or default_phase7_runtime_probe_designs())
    seeds = tuple(
        _coerce_runtime_nonnegative_integer("random_states", seed)
        for seed in random_states
    )
    if not seeds:
        raise ValueError("run_phase7_monte_carlo_runtime_probe requires random_states")
    n_boot_value = _coerce_runtime_positive_integer("n_boot", n_boot)
    design_cache_keys = tuple(
        _phase7_design_cache_key(design) for design in design_sequence
    )
    return _run_phase7_monte_carlo_runtime_probe_cached(
        seeds,
        n_boot_value,
        design_cache_keys,
    )


@lru_cache(maxsize=16)
def _run_phase7_monte_carlo_runtime_probe_cached(
    random_states: tuple[int, ...],
    n_boot: int,
    design_cache_keys: tuple[
        tuple[
            str,
            int,
            int,
            str,
            int,
            str,
            float,
            float,
            int,
            float,
            float,
            tuple[float, ...],
        ],
        ...,
    ],
) -> MonteCarloRuntimeProbeReport:
    design_sequence = tuple(
        _phase7_design_from_cache_key(cache_key) for cache_key in design_cache_keys
    )
    smoke_reports = tuple(
        run_monte_carlo_smoke(
            designs=design_sequence,
            n_replications=1,
            random_state=seed,
            n_boot=n_boot,
        )
        for seed in random_states
    )
    return build_monte_carlo_runtime_probe_report(
        smoke_reports=smoke_reports,
        random_states=random_states,
    )


run_phase7_monte_carlo_runtime_probe.cache_clear = (  # type: ignore[attr-defined]
    _run_phase7_monte_carlo_runtime_probe_cached.cache_clear
)
run_phase7_monte_carlo_runtime_probe.cache_info = (  # type: ignore[attr-defined]
    _run_phase7_monte_carlo_runtime_probe_cached.cache_info
)


def run_phase7_nonparametric_alias_probe(
    *,
    monte_carlo_random_state: int = 303,
    replication_seed: int = 460490113,
    design: MonteCarloDesign | None = None,
    contrast_grid: Sequence[float] = (0.1, 0.3, 0.7),
    integer_grid: Sequence[float] = (-2.0, -1.0, 0.0, 1.0, 2.0),
) -> Phase7NonparametricAliasProbe:
    design_value = design or MonteCarloDesign(
        dgp_name="DGP2",
        n_obs=200,
        p=10,
        evaluation_grid=_DEFAULT_EVALUATION_GRID.copy(),
    )
    contrast_grid_array = _coerce_evaluation_grid(contrast_grid)
    integer_grid_array = _coerce_evaluation_grid(integer_grid)

    dataset = _generate_dataset(design_value, random_state=int(replication_seed))
    data = dataset.to_validated_data()
    splits = make_crossfit_splits(
        n_obs=data.n_obs,
        n_folds=design_value.n_folds,
        random_state=int(replication_seed),
        trim_lower=design_value.trim_lower,
        trim_upper=design_value.trim_upper,
    )
    nuisance_payload = CrossfitNuisanceEstimator(
        oracle_lane=design_value.oracle_lane
    ).fit(data, splits)
    score_payload = build_score_payload(data, nuisance_payload)
    estimation_payload, _ = estimate_eq31_mainline(
        score_payload,
        penalty_lambda=0.0,
    )

    n_valid = int(score_payload.x_valid.shape[0])
    x_valid = np.asarray(score_payload.x_valid, dtype=float)
    basis_valid_full = np.asarray(score_payload.basis_valid_full, dtype=float)
    residual_valid = np.asarray(estimation_payload.residual_valid, dtype=float)
    sigma_x_hat = x_valid.T @ x_valid / n_valid
    m_hat = (basis_valid_full.T @ x_valid / n_valid) @ np.linalg.inv(sigma_x_hat)
    orthogonal_basis_valid = basis_valid_full - x_valid @ m_hat.T
    sigma_f_hat = orthogonal_basis_valid.T @ basis_valid_full / n_valid
    sigma_f_inverse = np.linalg.inv(sigma_f_hat)
    weighted_basis = basis_valid_full * residual_valid[:, None]
    weighted_x = x_valid * residual_valid[:, None]
    omega_f_hat = (
        weighted_basis.T @ weighted_basis / n_valid
        - m_hat @ (weighted_x.T @ weighted_x / n_valid) @ m_hat.T
    )
    v_f_hat = sigma_f_inverse @ omega_f_hat @ sigma_f_inverse

    current_grid_probe = _build_evaluation_grid_alias_probe(
        evaluation_grid=np.asarray(design_value.evaluation_grid, dtype=float),
        evaluation_basis=np.asarray(score_payload.evaluation_basis, dtype=float),
        v_f_hat=v_f_hat,
    )
    contrast_grid_probe = _build_evaluation_grid_alias_probe(
        evaluation_grid=contrast_grid_array,
        evaluation_basis=build_sieve_basis(
            contrast_grid_array,
            basis_family=design_value.basis_family,
            degree=design_value.basis_degree,
        ),
        v_f_hat=v_f_hat,
    )
    integer_grid_probe = _build_evaluation_grid_alias_probe(
        evaluation_grid=integer_grid_array,
        evaluation_basis=build_sieve_basis(
            integer_grid_array,
            basis_family=design_value.basis_family,
            degree=design_value.basis_degree,
        ),
        v_f_hat=v_f_hat,
    )
    return Phase7NonparametricAliasProbe(
        monte_carlo_random_state=int(monte_carlo_random_state),
        replication_seed=int(replication_seed),
        design=design_value,
        current_grid=current_grid_probe,
        contrast_grid=contrast_grid_probe,
        integer_grid=integer_grid_probe,
        omega_f_min_eigenvalue=_minimum_symmetric_eigenvalue(omega_f_hat),
        v_f_min_eigenvalue=_minimum_symmetric_eigenvalue(v_f_hat),
    )


def _generate_dataset(
    design: MonteCarloDesign,
    *,
    random_state: int | None,
) -> SimulationDataset:
    if design.dgp_name == "DGP1":
        return generate_paper_dgp1(design, random_state=random_state)
    return generate_paper_dgp2(design, random_state=random_state)


def run_monte_carlo_smoke(
    *,
    designs: Sequence[MonteCarloDesign] | None = None,
    n_replications: int = 1,
    random_state: int | None = None,
    n_boot: int = 64,
) -> MonteCarloSmokeReport:
    design_sequence = tuple(designs or default_reduced_smoke_designs())
    if not design_sequence:
        raise ValueError("run_monte_carlo_smoke requires at least one design")
    _validate_paper_monte_carlo_lane({design.oracle_lane for design in design_sequence})
    n_replications_value = _coerce_runtime_positive_integer(
        "n_replications",
        n_replications,
    )
    n_boot_value = _coerce_runtime_positive_integer("n_boot", n_boot)
    random_state_value = (
        None
        if random_state is None
        else _coerce_runtime_nonnegative_integer("random_state", random_state)
    )

    full_target_matrix = _paper_target_matrix()
    rng = np.random.default_rng(random_state_value)
    summaries: list[MonteCarloSmokeSummary] = []
    report_start = perf_counter()

    for design in design_sequence:
        design_start = perf_counter()
        typed_invalidity_counts: Counter[str] = Counter()
        typed_invalidity_examples: dict[str, dict[str, object]] = {}
        zero_valid_fold_count = 0
        successful_replications = 0
        total_trimmed = 0
        total_holdout = 0
        parametric_errors: list[np.ndarray] = []
        parametric_standard_errors: list[np.ndarray] = []
        parametric_coverage: list[np.ndarray] = []
        parametric_interval_lengths: list[np.ndarray] = []
        nonparametric_errors: list[np.ndarray] = []
        nonparametric_standard_errors: list[np.ndarray] = []
        nonparametric_coverage: list[np.ndarray] = []
        nonparametric_interval_lengths: list[np.ndarray] = []
        nonparametric_absolute_errors: list[np.ndarray] = []
        nonparametric_uniform_critical_values: list[float] = []
        nonparametric_uniform_band_lengths: list[np.ndarray] = []

        for _ in range(n_replications_value):
            replication_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            dataset = _generate_dataset(design, random_state=replication_seed)
            try:
                if np.asarray(design.evaluation_grid).shape[0] == 0:
                    raise MissingEvaluationGridError(
                        "uniform band requires an explicit non-empty evaluation grid",
                        metadata={
                            "failure_kind": "missing-evaluation-grid",
                            "grid_size": 0,
                            "target_kind": "nonparametric",
                        },
                    )
                data = dataset.to_validated_data()
                splits = make_crossfit_splits(
                    n_obs=data.n_obs,
                    n_folds=design.n_folds,
                    random_state=replication_seed,
                    trim_lower=design.trim_lower,
                    trim_upper=design.trim_upper,
                )
                nuisance_payload = CrossfitNuisanceEstimator(
                    oracle_lane=design.oracle_lane
                ).fit(data, splits)
                score_payload = build_score_payload(data, nuisance_payload)
                estimation_payload, result = estimate_eq31_mainline(
                    score_payload,
                    penalty_lambda=0.0,
                )
                parametric_payload, result = estimate_parametric_inference(
                    score_payload,
                    estimation_payload,
                    result=result,
                    alpha=design.alpha,
                    lambda_prime=0.0,
                )
                nonparametric_payload, result = estimate_nonparametric_inference(
                    score_payload,
                    estimation_payload,
                    result=result,
                    alpha=design.alpha,
                    lambda_double_prime=0.0,
                    n_boot=n_boot_value,
                    random_state=replication_seed,
                )
            except (
                ZeroValidHoldoutError,
                NuisanceTrainingSupportError,
                Eq31ProjectionRankError,
            ) as exc:
                error_name = type(exc).__name__
                typed_invalidity_counts[error_name] += 1
                typed_invalidity_examples.setdefault(
                    error_name,
                    _monte_carlo_invalidity_example(
                        exc,
                        design=design,
                        monte_carlo_random_state=(
                            None if random_state is None else int(random_state)
                        ),
                        replication_seed=replication_seed,
                    ),
                )
                if isinstance(exc, ZeroValidHoldoutError):
                    zero_valid_fold_count += 1
                continue
            except InferenceComputationError as exc:
                error_name = type(exc).__name__
                typed_invalidity_counts[error_name] += 1
                typed_invalidity_examples.setdefault(
                    error_name,
                    _monte_carlo_invalidity_example(
                        exc,
                        design=design,
                        monte_carlo_random_state=(
                            None if random_state is None else int(random_state)
                        ),
                        replication_seed=replication_seed,
                    ),
                )
                continue

            successful_replications += 1
            if result.diagnostics is not None:
                total_trimmed += int(result.diagnostics.n_trimmed_propensity or 0)
                total_holdout += int(result.diagnostics.n_holdout_raw or data.n_obs)

            parametric_truth = np.asarray(dataset.true_beta, dtype=float)
            parametric_estimate = np.asarray(parametric_payload.t_hat, dtype=float)
            parametric_ci = parametric_payload.confidence_interval
            parametric_errors.append(parametric_estimate - parametric_truth)
            parametric_standard_errors.append(
                np.asarray(parametric_payload.standard_errors, dtype=float)
            )
            parametric_coverage.append(
                (np.asarray(parametric_ci.lower, dtype=float) <= parametric_truth)
                & (parametric_truth <= np.asarray(parametric_ci.upper, dtype=float))
            )
            parametric_interval_lengths.append(
                np.asarray(parametric_ci.upper, dtype=float)
                - np.asarray(parametric_ci.lower, dtype=float)
            )

            nonparametric_truth = np.asarray(dataset.true_f_at_z0, dtype=float)
            nonparametric_estimate = np.asarray(
                nonparametric_payload.bar_f_at_z0, dtype=float
            )
            nonparametric_ci = nonparametric_payload.pointwise_confidence_interval
            nonparametric_errors.append(nonparametric_estimate - nonparametric_truth)
            nonparametric_absolute_errors.append(
                np.abs(nonparametric_estimate - nonparametric_truth)
            )
            nonparametric_standard_errors.append(
                np.asarray(nonparametric_payload.sigma_z_hat, dtype=float)
            )
            nonparametric_coverage.append(
                (np.asarray(nonparametric_ci.lower, dtype=float) <= nonparametric_truth)
                & (
                    nonparametric_truth
                    <= np.asarray(nonparametric_ci.upper, dtype=float)
                )
            )
            nonparametric_interval_lengths.append(
                np.asarray(nonparametric_ci.upper, dtype=float)
                - np.asarray(nonparametric_ci.lower, dtype=float)
            )
            nonparametric_uniform_critical_values.append(
                float(nonparametric_payload.uniform_band.critical_value)
            )
            nonparametric_uniform_band_lengths.append(
                np.asarray(nonparametric_payload.uniform_band.upper, dtype=float)
                - np.asarray(nonparametric_payload.uniform_band.lower, dtype=float)
            )

        summaries.append(
            MonteCarloSmokeSummary(
                design=design,
                staging=_PAPER_MONTE_CARLO_STAGE,
                full_target_matrix=full_target_matrix,
                n_replications=n_replications_value,
                n_successful_replications=successful_replications,
                typed_invalidity_counts=dict(sorted(typed_invalidity_counts.items())),
                trimming_rate=(
                    float(total_trimmed / total_holdout) if total_holdout > 0 else None
                ),
                zero_valid_fold_frequency=float(
                    zero_valid_fold_count / n_replications_value
                ),
                parametric_metrics=_metric_summary(
                    parametric_errors,
                    parametric_standard_errors,
                    parametric_coverage,
                    parametric_interval_lengths,
                ),
                nonparametric_metrics=_metric_summary(
                    nonparametric_errors,
                    nonparametric_standard_errors,
                    nonparametric_coverage,
                    nonparametric_interval_lengths,
                ),
                nonparametric_mean_absolute_error=(
                    None
                    if not nonparametric_absolute_errors
                    else float(
                        np.mean(
                            np.concatenate(
                                [
                                    np.ravel(np.asarray(value, dtype=float))
                                    for value in nonparametric_absolute_errors
                                ]
                            )
                        )
                    )
                ),
                nonparametric_uniform_critical_value=(
                    None
                    if not nonparametric_uniform_critical_values
                    else float(
                        np.mean(
                            np.asarray(
                                nonparametric_uniform_critical_values,
                                dtype=float,
                            )
                        )
                    )
                ),
                nonparametric_uniform_band_length=(
                    None
                    if not nonparametric_uniform_band_lengths
                    else float(
                        np.mean(
                            np.concatenate(
                                [
                                    np.ravel(np.asarray(value, dtype=float))
                                    for value in nonparametric_uniform_band_lengths
                                ]
                            )
                        )
                    )
                ),
                runtime_seconds=perf_counter() - design_start,
                typed_invalidity_examples=typed_invalidity_examples,
            )
        )

    return MonteCarloSmokeReport(
        stage_label=_PAPER_MONTE_CARLO_STAGE,
        staging=True,
        full_target_matrix=full_target_matrix,
        nominal_coverage=_PAPER_NOMINAL_COVERAGE,
        staged_subset=tuple(design.staged_subset_entry() for design in design_sequence),
        summaries=tuple(summaries),
        total_runtime_seconds=perf_counter() - report_start,
    )


for _helper_name, _helper in tuple(globals().items()):
    if _helper_name.startswith("run_phase7_") and callable(_helper):
        globals()[_helper_name] = _with_source_checkout_boundary(_helper)

del _helper_name, _helper
