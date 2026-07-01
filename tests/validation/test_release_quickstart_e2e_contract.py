from __future__ import annotations

from pathlib import Path

from hddid.section6_loader import load_section6_data
from hddid.section6_loader_object_contract import (
    audit_phase7_section6_loader_payload,
)
from hddid.validation import (
    MonteCarloDesign,
    audit_section6_assets,
    audit_section6_provenance_gate,
    generate_paper_dgp1,
    run_monte_carlo_smoke,
    run_phase7_monte_carlo_widening_policy_spec,
    run_phase7_monte_carlo_widening_trigger_gate,
    run_phase7_next_milestone_trigger_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "hddid-py" / "README.md"


def test_release_readme_quickstart_output_matches_live_e2e_path() -> None:
    readme_text = README_PATH.read_text(encoding="utf-8")
    for snippet in (
        "## Quick Start",
        "fit_hddid",
    ):
        assert snippet in readme_text

    design = MonteCarloDesign(
        dgp_name="DGP1",
        n_obs=80,
        p=5,
        basis_family="trigonometric",
        basis_degree=8,
        oracle_lane="paper-trigonometric",
    )
    dataset = generate_paper_dgp1(design, random_state=11)
    report = run_monte_carlo_smoke(
        designs=(design,),
        n_replications=1,
        random_state=9,
        n_boot=32,
    )
    audit = audit_section6_assets(REPO_ROOT)
    provenance_gate = audit_section6_provenance_gate(REPO_ROOT)
    payload = load_section6_data(REPO_ROOT)
    payload_audit = audit_phase7_section6_loader_payload(payload)
    snapshot = run_phase7_next_milestone_trigger_snapshot(REPO_ROOT)
    trigger_gate = run_phase7_monte_carlo_widening_trigger_gate()
    policy_spec = run_phase7_monte_carlo_widening_policy_spec()
    summary = report.summaries[0]

    assert dataset.oracle_lane == "paper-trigonometric"
    assert dataset.true_beta.shape == (5,)
    assert report.stage_label == "reduced-smoke"
    assert summary.design.oracle_lane == "paper-trigonometric"
    assert audit.status == "ready"
    assert audit.blocker_reason is None
    assert provenance_gate.status == "ready"
    assert provenance_gate.blocker_reason is None
    assert payload_audit.status == "ready"
    assert payload_audit.county_characteristic_coverage_status == (
        "partial-local-payload"
    )
    assert payload_audit.observed_county_characteristic_count == 2
    assert payload["lane_metadata"]["observation_shape"] == "county-year panel"
    assert payload["lane_metadata"]["basis_degree"] == 4
    assert payload["lane_metadata"]["confidence_level"] == 0.95
    assert snapshot.recommended_feature_bundle == "trigger2-bounded-widening"
    assert trigger_gate.gate_status == "trigger2-policy-missing"
    assert policy_spec.trigger_label == "trigger2-policy-spec"
