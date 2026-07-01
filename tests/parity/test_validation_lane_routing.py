from __future__ import annotations

import importlib

import pytest


def _load_validation_module():
    try:
        return importlib.import_module("hddid.validation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"hddid.validation is missing: {exc}")


def _make_metric_payload() -> dict[str, float | None]:
    return {
        "bias": 0.0,
        "rmse": 0.1,
        "average_standard_error": 0.2,
        "coverage": 0.9,
        "interval_length": 0.4,
    }


def _make_monte_carlo_report(
    validation_module,
    *,
    basis_family: str,
    basis_degree: int,
    oracle_lane: str,
):
    full_target_matrix = {
        "n": (200, 500, 1000),
        "p": (10, 50, 500, 1000),
        "nominal_coverage": 0.9,
    }
    design = validation_module.MonteCarloDesign(
        dgp_name="DGP1",
        n_obs=200,
        p=10,
        basis_family=basis_family,
        basis_degree=basis_degree,
        oracle_lane=oracle_lane,
    )
    summary = validation_module.MonteCarloSmokeSummary(
        design=design,
        staging="reduced-smoke",
        full_target_matrix=full_target_matrix,
        n_replications=1,
        n_successful_replications=1,
        typed_invalidity_counts={},
        trimming_rate=0.0,
        zero_valid_fold_frequency=0.0,
        parametric_metrics=_make_metric_payload(),
        nonparametric_metrics=_make_metric_payload(),
    )
    return validation_module.MonteCarloSmokeReport(
        stage_label="reduced-smoke",
        staging=True,
        full_target_matrix=full_target_matrix,
        nominal_coverage=0.9,
        staged_subset=(design.staged_subset_entry(),),
        summaries=(summary,),
    )


def _make_blocked_empirical_audit(validation_module):
    return validation_module.EmpiricalAssetAudit(
        repo_root="/tmp/pyhddid",
        oracle_lane="real-data-asset-audit",
        status="blocked",
        blocker_reason="missing-local-dataset",
        missing_assets=("section6_raw_dataset", "section6_loader"),
        raw_data_assets=(),
        loader_assets=(),
        known_figures=("paper/images/marginaleffectmedinc703.jpg",),
        time_window="2005-2007",
        treated_state_count=11,
        excluded_states=("New Hampshire", "Pennsylvania"),
        linear_covariate_count=703,
        z_lanes=("median income", "population"),
        basis_family="trigonometric",
        basis_degree=4,
        confidence_level=0.95,
        synthetic_fallback_enabled=False,
        figure_only_mode_enabled=False,
    )


def test_polynomial_r_parity_lane_stays_separate_from_paper_trig_lane() -> None:
    validation_module = _load_validation_module()

    with pytest.raises(ValueError, match="paper-trigonometric"):
        validation_module.build_validation_matrix_report(
            monte_carlo_report=_make_monte_carlo_report(
                validation_module,
                basis_family="polynomial",
                basis_degree=2,
                oracle_lane="r-parity-polynomial",
            ),
            parity_routes=(),
            empirical_asset_audit=_make_blocked_empirical_audit(validation_module),
        )


def test_polynomial_r_sanity_checks_cannot_satisfy_paper_lane() -> None:
    validation_module = _load_validation_module()

    with pytest.raises(ValueError, match="r-parity-polynomial"):
        validation_module.build_validation_matrix_report(
            monte_carlo_report=_make_monte_carlo_report(
                validation_module,
                basis_family="trigonometric",
                basis_degree=8,
                oracle_lane="paper-trigonometric",
            ),
            parity_routes=(
                validation_module.ParityValidationRoute(
                    oracle_lane="paper-trigonometric",
                    evidence_scope="raw-eq31-objects",
                    status="validated",
                    primary_oracle="R snapshot raw Eq. (3.1) wrapper",
                    source_classification="R-usable reference",
                    allowed_comparisons=("raw second-stage objects",),
                    blocked_comparisons=("paper Section 5 Monte Carlo truth",),
                ),
            ),
            empirical_asset_audit=_make_blocked_empirical_audit(validation_module),
        )


def test_empirical_blocker_cannot_be_reported_as_parity_success() -> None:
    validation_module = _load_validation_module()

    report = validation_module.build_validation_matrix_report(
        monte_carlo_report=_make_monte_carlo_report(
            validation_module,
            basis_family="trigonometric",
            basis_degree=8,
            oracle_lane="paper-trigonometric",
        ),
        parity_routes=(
            validation_module.ParityValidationRoute(
                oracle_lane="r-parity-polynomial",
                evidence_scope="raw-eq31-objects",
                status="validated",
                primary_oracle="R snapshot raw Eq. (3.1) wrapper",
                source_classification="R-usable reference",
                allowed_comparisons=("raw second-stage objects",),
                blocked_comparisons=("paper Section 5 Monte Carlo truth",),
            ),
        ),
        empirical_asset_audit=_make_blocked_empirical_audit(validation_module),
    )

    assert report.lane("r-parity-polynomial").status == "validated"
    assert report.lane("real-data-asset-audit").status == "blocked"
    assert report.empirical_blocker_reason == "missing-local-dataset"
