# Release checklist

This checklist is for the current `research-alpha` release boundary of
`hddid`. Run every command from the repository root before treating the package
metadata or release docs as up to date.

Read `hddid-py/docs/release-diagnosis.md` first when deciding whether the
current package can move beyond research-alpha. It records the current stage,
release conclusion, P0/P1/P2/P3 gaps, and the verification commands that support
those claims.

## Install smoke

```bash
python -m pip install -e './hddid-py'
python -m pip install -e './hddid-py[dev]'
python -m pip install -e './hddid-py[release]'
```

## Distribution smoke

Build both source and wheel distributions from the package directory, then
install the wheel into a fresh virtual environment before replaying the public
import surface. This catches packaging drift that an editable install can hide.

```bash
python -m pip install -e './hddid-py[release]'
python -m build './hddid-py' --outdir dist
python -m venv /tmp/hddid-wheel-smoke
/tmp/hddid-wheel-smoke/bin/python -m pip install "$(ls dist/hddid-*.whl)"
/tmp/hddid-wheel-smoke/bin/python - <<'PY'
import importlib
from pathlib import Path
import zipfile

from hddid import (
    HDDIDResult,
    estimate_eq31_mainline,
    fit_hddid,
    polynomial_sieve_basis,
    trigonometric_sieve_basis,
)
from hddid.validation import (
    SourceCheckoutRequiredError,
    audit_phase7_release_artifact_content,
    run_phase7_monte_carlo_feature_completion_gate,
    run_phase7_monte_carlo_widening_trigger_gate,
    run_phase7_release_maturity_gate,
)

print(HDDIDResult.__name__)
print(estimate_eq31_mainline.__name__)
print(fit_hddid.__name__)
print(polynomial_sieve_basis.__name__)
print(trigonometric_sieve_basis.__name__)
print(audit_phase7_release_artifact_content.__name__)

wheel = next(Path("dist").glob("hddid-*.whl"))
retained_modules = []
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) == 2 and parts[0] == "hddid" and parts[1].endswith(".py"):
            retained_modules.append(parts[1][:-3])
for module_name in sorted(retained_modules):
    importlib.import_module(
        "hddid" if module_name == "__init__" else f"hddid.{module_name}"
    )

for helper, args in (
    (audit_phase7_release_artifact_content, (".",)),
    (run_phase7_monte_carlo_feature_completion_gate, (".",)),
    (run_phase7_monte_carlo_widening_trigger_gate, ()),
    (run_phase7_release_maturity_gate, (".",)),
):
    try:
        helper(*args)
    except SourceCheckoutRequiredError as exc:
        print(type(exc).__name__)
    else:
        raise SystemExit(f"{helper.__name__} unexpectedly ran from the wheel")
PY
```

Before any non-alpha version bump, run
`audit_phase7_release_artifact_content(repo_root)` on the source tree and again
on the built wheel or sdist with `package_artifact=...`. This is a
source-checkout audit, not an installed-wheel smoke: it reads package metadata
from the repository root and compares that release label with either the source
directory or a built archive. The current `0.1.0` source tree is an honest
`research-alpha` artifact, but the same layout is blocked as a formal release
candidate because it still contains research-only probe/trigger/automation-state
modules and content-different `* 2.py` / `* 3.py` files under
`hddid-py/src/hddid`. The build step keeps only the core installed modules in
wheel and sdist artifacts: `__init__`, basis, estimation, estimator, inference,
inputs, nuisance, plotting, results, score, splitting, and validation. Archive
audits should therefore report `0` space-suffixed modules and `0` internal
release-control modules, and archive metadata such as `SOURCES.txt` and `RECORD`
must not retain filtered module references. The archive-level audit prevents a
formal release from passing merely because the source scan and built artifact
disagree.

```bash
python - <<'PY'
from pathlib import Path
from hddid.validation import audit_phase7_release_artifact_content

repo_root = Path(".").resolve()
wheel = next(Path("dist").glob("hddid-*.whl"))
source_report = audit_phase7_release_artifact_content(repo_root)
wheel_report = audit_phase7_release_artifact_content(
    repo_root,
    package_artifact=wheel,
)
print(source_report.to_dict())
print(wheel_report.to_dict())
assert source_report.artifact_pollution_present
assert not wheel_report.space_suffix_modules
assert not wheel_report.internal_release_modules
PY
```

`run_phase7_release_maturity_gate(...)` and
`run_phase7_release_matrix_runner(...)` are source-checkout gates. Run them from
the repository tree, not as installed-wheel smoke checks, because they read the
repo-side validation state and deliberately keep Monte Carlo and manuscript
release blockers visible.

## Import smoke

```bash
python - <<'PY'
from hddid import estimate_eq31_mainline, fit_hddid
from hddid.validation import audit_section6_assets

print(estimate_eq31_mainline.__name__)
print(fit_hddid.__name__)
print(audit_section6_assets.__name__)
PY
```

## Trigger 1 empirical smoke

```bash
python - <<'PY'
from pathlib import Path

from hddid.section6_loader import load_section6_data
from hddid.section6_loader_object_contract import (
    audit_phase7_section6_loader_payload,
)
from hddid.validation import audit_section6_provenance_gate

repo_root = Path(".").resolve()
provenance_gate = audit_section6_provenance_gate(repo_root)
payload = load_section6_data(repo_root)
payload_audit = audit_phase7_section6_loader_payload(payload)

print(provenance_gate.status)
print(provenance_gate.blocker_reason)
print(payload_audit.status)
print(payload["sample_metadata"]["sample_rule"]["baseline_universe"])
print(payload["sample_metadata"]["treated_states"][0])
print(payload["sample_metadata"]["control_states"][0])
print(payload["sample_metadata"]["excluded_states"])
print(payload["lane_metadata"]["observation_shape"])
print(payload["lane_metadata"]["basis_degree"])
print(payload["lane_metadata"]["confidence_level"])
PY
```

The current release-facing Trigger 1 truth is still the repository-local panel
plus manifest on `hddid-py/data/section6_county_panel.tsv` and
`hddid-py/data/section6_manifest.json`, while the canonical loader asset stays
`hddid-py/src/hddid/section6_loader.py`. Keep this smoke executable so the
release checklist proves all three sides of that claim: the provenance gate
stays `ready` with no blocker, the loader payload stays `ready` on the current
county-year panel contract instead of collapsing back into prose-only
readiness, and the release-facing asset gate does not drift back to a
data-plus-manifest-only story. The same smoke should also keep the source-backed
sample roster machine-readable on the release-facing path:
`baseline_universe = 2006 federal-binding states`, the first treated/control
anchors, and the excluded states `["New Hampshire", "Pennsylvania"]` should
replay beside the lane metadata so future real-data intake cannot silently fall
back to a count-only Trigger 1 story.

## Targeted release-contract tests

```bash
pytest hddid-py/tests/validation/test_release_readme_contract.py -q
pytest hddid-py/tests/validation/test_release_docs_contract.py -q
pytest hddid-py/tests/validation/test_packaging_metadata.py -q
pytest hddid-py/tests/unit/test_validation_source_checkout_boundary.py -q
pytest hddid-py/tests/validation/test_release_maturity_gate_surface_sync.py -q
pytest hddid-py/tests/unit/test_release_matrix_runner.py hddid-py/tests/validation/test_release_matrix_runner_surface_sync.py -q
pytest hddid-py/tests/validation/test_public_documented_import_contract.py -q
pytest hddid-py/tests/validation/test_release_quickstart_e2e_contract.py -q
pytest hddid-py/tests/validation/test_trigger2_feature_completion_frontier_packet_surface_sync.py -q
pytest hddid-py/tests/unit/test_monte_carlo_paper_dgp2_contract_audit.py hddid-py/tests/validation/test_monte_carlo_paper_dgp2_contract_e2e.py -q
```

`test_trigger2_feature_completion_frontier_packet_surface_sync.py` keeps
`run_phase7_monte_carlo_feature_completion_frontier_packet(...)`,
`Docs/research/phase7_monte_carlo_feature_completion_frontier_packet.md`,
`trigger2-runtime-evidence-packet-open`, `DGP2/500/50`, `seed 303`, `7/9 -> 8/9`,
and `5 fresh reruns` visible on the release-facing path so the shortest
Trigger 2 completion read does not fall back to handoff prose only.

`test_monte_carlo_paper_dgp2_contract_audit.py` and
`test_monte_carlo_paper_dgp2_contract_e2e.py` keep the DGP2 paper/R/Python
contract audit visible on the release-facing path, and
`run_phase7_monte_carlo_paper_dgp2_contract_audit(...)` /
`build_phase7_monte_carlo_paper_dgp2_contract_audit_report(...)` keep the same
audit machine-readable, with `RBUG-009` / `RBUG-012` still called out, so the
paper-first `rho_x` covariance contract, the observed-`z` replay contract, and
the current `DGP2/500/50` near-zero grid `0.05, 0.15, 0.25` (shown as
`(0.05, 0.15, 0.25)`) remain visible beside the existing Trigger 2 blocker
surface.

`test_release_matrix_runner.py` and
`test_release_matrix_runner_surface_sync.py` keep
`run_phase7_release_matrix_runner(...)` visible on the release-facing path so
the package can report the deterministic `empirical_lane_ready`,
`section6_provenance_gate_ready`, `outer_inference_parity_ready`,
`monte_carlo_validation_ready`, and `release_maturity_ready` rows without
collapsing the release contract back into prose. The companion note
`Docs/research/phase7_release_matrix_runner.md` keeps the same row set and the
current blocker explicit. The report exposes `rows`, `blocking_row_ids`, and
`blocking_rows` for release scripts that need the current row payloads and
blocking row payloads without rebuilding them from prose. Release scripts that
already computed the
feature-completion gate can pass that report as `feature_gate=` to the matrix
runner so the release matrix and maturity row reuse one live evidence read.
The Monte Carlo row must keep the same DGP2 paper/R/Python contract evidence
from the feature gate: `paper DGP2 contract paper-first-python-dgp2-contract-locked`,
`paper DGP2 R drift RBUG-009,RBUG-012`, and
`paper DGP2 near-zero grid (0.05, 0.15, 0.25)`.

`test_public_documented_import_contract.py` keeps the public documented import contract for
the package-root and validation import surface aligned with importable names, while
`test_release_quickstart_e2e_contract.py` keeps the README quickstart E2E and
release-facing path executable beside the release contract instead of
leaving them as prose-only guidance.

## Phase 7 final verification

```bash
pytest hddid-py/tests/validation/test_final_verification_contract.py -q
pytest hddid-py/tests/validation/test_phase7_validation_debt.py hddid-py/tests/validation/test_automation_role_alignment.py hddid-py/tests/validation/test_final_verification_contract.py -q
pytest hddid-py/tests/validation/test_phase7_validation_bundle_inventory.py -q
pytest hddid-py/tests/validation/test_phase7_final_verification_bundle_inventory.py -q
pytest hddid-py/tests/validation/test_automation_two_stage_mode_contract.py -q
pytest hddid-py/tests/validation/test_automation_memory_path_expansion_contract.py -q
pytest hddid-py/tests/validation/test_automation_memory_path_manual_escalation_surface_sync.py -q
pytest -q hddid-py/tests/validation/test_trigger1_section6_manifest_template_asset.py hddid-py/tests/validation/test_trigger1_section6_provenance_surface_sync.py hddid-py/tests/validation/test_trigger1_section6_loader_object_contract_surface_sync.py hddid-py/tests/validation/test_trigger1_section6_paper_schema_surface_sync.py hddid-py/tests/validation/test_trigger2_margin_guard_role_split_inventory.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_surface_sync.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_surface_sync.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_boundary_surface_sync.py hddid-py/tests/validation/test_trigger2_readme_primary_loop_routing.py hddid-py/tests/validation/test_trigger2_handoff_primary_loop_routing.py hddid-py/tests/validation/test_trigger2_state_primary_loop_routing.py hddid-py/tests/validation/test_trigger2_historical_entry_alignment.py hddid-py/tests/validation/test_next_milestone_trigger_snapshot_surface_sync.py hddid-py/tests/validation/test_trigger3_outer_inference_contract_surface_sync.py hddid-py/tests/validation/test_trigger3_outer_inference_repair_obligations_surface_sync.py
pytest -q hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_grid_probe.py hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_window_probe.py hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_probe.py hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_boundary_probe.py
pytest -q hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_window_symmetry_break_probe.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_window_symmetry_break_surface_sync.py
pytest -q hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_companion_reserve_probe.py hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_growth_split_probe.py hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_scale_repair_probe.py hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_share_probe.py hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe.py hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_localization_probe.py hddid-py/tests/unit/test_monte_carlo_widening_policy_seed_window_covariance_probe.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_companion_surface_sync.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_scale_repair_note.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_access_surface_sync.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_localization_surface_sync.py hddid-py/tests/validation/test_trigger2_policy_seed_window_covariance_surface_sync.py
pytest -q hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_access_driver_probe.py hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_driver_surface_sync.py
pytest -q hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_center_coupling_channel_surface_sync.py
pytest -q hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_center_coupling_repair_surface_sync.py
pytest -q hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_directionality_surface_sync.py
pytest -q hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe.py hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_correlation_gap_surface_sync.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_cross_shoulder_sign_surface_sync.py
pytest -q hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_covariance_entry_factor_surface_sync.py
pytest -q hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_repair_target_snapshot_note.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_repair_target_snapshot_surface_sync.py hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot.py hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot_surface_sync.py
pytest -q hddid-py/tests/unit/test_monte_carlo_widening_policy_calibration_debt_snapshot.py hddid-py/tests/validation/test_trigger2_policy_calibration_debt_snapshot_note.py hddid-py/tests/validation/test_trigger2_policy_calibration_debt_snapshot_surface_sync.py
python -m pip install -e './hddid-py[dev]'
python - <<'PY'
from pathlib import Path

from hddid import estimate_eq31_mainline
from hddid.monte_carlo_widening_policy_spec import (
    run_phase7_monte_carlo_widening_policy_spec,
)
from hddid.next_milestone_trigger_snapshot import (
    run_phase7_next_milestone_trigger_snapshot,
)
from hddid.outer_inference_determinism_probe import (
    replay_phase7_outer_inference_uniform_critical_value,
    run_phase7_outer_inference_determinism_probe,
)
from hddid.outer_inference_trigger_gate import run_phase7_outer_inference_trigger_gate
from hddid.validation import (
    audit_section6_assets,
    build_phase7_outer_inference_reference_contract,
    build_phase7_validation_debt_report,
    build_validation_matrix_report,
    run_monte_carlo_smoke,
)

repo_root = Path(".").resolve()
monte_carlo_report = run_monte_carlo_smoke(n_replications=1, random_state=123)
empirical_asset_audit = audit_section6_assets(repo_root)
validation_matrix_report = build_validation_matrix_report(
    monte_carlo_report=monte_carlo_report,
    parity_routes=(),
    empirical_asset_audit=empirical_asset_audit,
)
phase7_validation_debt_report = build_phase7_validation_debt_report(
    validation_matrix_report=validation_matrix_report
)
widening_policy_report = run_phase7_monte_carlo_widening_policy_spec()
outer_inference_contract = build_phase7_outer_inference_reference_contract()
trigger_gate = run_phase7_outer_inference_trigger_gate()
determinism_probe = run_phase7_outer_inference_determinism_probe()
replayed_uniform_critical_value = replay_phase7_outer_inference_uniform_critical_value(
    outer_inference_contract
)
next_milestone_snapshot = run_phase7_next_milestone_trigger_snapshot(repo_root)

print(estimate_eq31_mainline.__name__)
print(validation_matrix_report.stage_label)
print(validation_matrix_report.empirical_blocker_reason)
print(phase7_validation_debt_report.lane_keys)
print(widening_policy_report.target_gate_status)
print(outer_inference_contract.status)
print(outer_inference_contract.uniform_critical_value)
print(trigger_gate.gate_status)
print(determinism_probe.replay_random_state)
print(determinism_probe.replay_n_boot)
print(replayed_uniform_critical_value)
print(next_milestone_snapshot.recommended_feature_bundle)
print(next_milestone_snapshot.recommended_bounded_loop)
PY
```

Keep the release-facing closeout surface explicit about the narrow trigger
companions above:
`hddid-py/tests/validation/test_trigger1_section6_manifest_template_asset.py`
and
`hddid-py/tests/validation/test_trigger1_section6_provenance_surface_sync.py`
anchor Trigger 1 manifest/provenance honesty,
`hddid-py/tests/validation/test_trigger1_section6_loader_object_contract_surface_sync.py`
keeps the Trigger 1 loader-object-contract note visible in live routing and
closeout inventory instead of leaving `asset + loader + manifest` as prose
only, while `hddid-py/tests/unit/test_section6_loader_object_contract.py`
keeps the required `analysis_rows`, `sample_metadata`, `lane_metadata`, and
`linear_covariate_manifest` object surface executable,
`hddid-py/tests/validation/test_trigger1_section6_paper_schema_surface_sync.py`
keeps `Docs/research/phase7_section6_paper_schema_source_audit.md` visible in
live routing and closeout inventory, while
`hddid-py/tests/unit/test_section6_paper_schema_audit.py`
keeps the empirical `2005-2007` / `4th degree trigonometric polynomial basis` /
`95% confidence intervals` split executable against the Monte Carlo
`8th degree trigonometric polynomial basis` / `90% confidence intervals`
baseline so future empirical loader / manifest work must not inherit
simulation settings by accident,
`hddid-py/tests/validation/test_trigger2_margin_guard_role_split_inventory.py`
keeps the Trigger 2 companion chain from truncating at `guard_reserve`, and
`hddid-py/tests/validation/test_phase7_validation_bundle_inventory.py`
keeps the Phase 7 closeout inventory honest by requiring `.planning/.../07-VALIDATION.md`
to keep the `07-CC-17` companion bundle aligned on
`coupling-first-repair-target`, `correlation-led-covariance-entry-repair-target`,
and the calibration-debt snapshot follow-up,
instead of letting future QA / correct-course runs silently drop that bundle
retention regression from the closeout surface,
while
`hddid-py/tests/validation/test_phase7_final_verification_bundle_inventory.py`
keeps this final-verification surface honest by requiring the final matrix and
release checklist to continue naming the widening-trigger, Section 6 roster,
R snapshot basis/method audit, runtime evidence, candidate guard, contract
audit bundle guards, and
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate_validation_only.py`
as the repo-side carry-forward guard for
`phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate.md`,
`run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_entry_patch_fresh_rerun_gate()`,
`fresh-rerun-gated-by-entry-patch-readiness`,
`bounded-right-center-execution-contract`, and
`trigger2-policy-spec`,
while
`hddid-py/tests/validation/test_automation_memory_path_expansion_contract.py`
as the automation-memory hygiene regression, instead of letting future
closeout edits retain the underlying spot checks but silently drop the
regression that keeps that bundle machine-checkable,
while
`hddid-py/tests/validation/test_automation_memory_path_manual_escalation_surface_sync.py`
keeps this same final-verification surface honest by requiring the final
matrix and release checklist to continue naming the repo-root literal
`$CODEX_HOME/automations/**/memory.md` bad artifact and the paired
`git status --short --branch` / `find '$CODEX_HOME' -maxdepth 3 -type f`
manual cleanup repro, instead of letting future closeout edits keep the
expansion contract but silently drop the escalation guard that tells later
runs the artifact still needs a repo-root-authorized cleanup loop,
while
`run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate(...)`
keeps the `validation-only` Trigger 2 runtime-evidence admission gate visible on
this release-facing checklist: the bounded `DGP2/500/50` lane must keep
`trigger2-runtime-evidence-packet-open`, `runtime-evidence-rejected`,
`NonpositiveVarianceError`, and `omega_f_hat` machine-readable beside seed
`303`, the witness threshold context `7/9 -> 8/9`, and
`trigger2-policy-spec`, so release honesty does not hide the current bounded
runtime-evidence candidate rejection behind generic Monte Carlo prose,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_grid_probe.py`
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_window_probe.py`
keep the `micro_center_grid` witness and the `local-window calibration debt`
around `z = 0.15` executable, while
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_window_symmetry_break_probe.py`
and
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_window_symmetry_break_surface_sync.py`
keep the `right-only symmetry break` / `directional shoulder calibration`
readout executable on `near_zero_grid`, including the `z = 0.05` vs `z = 0.25`
split and the `+0.05 -> +0.10` band beyond the stable window, while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_surface_sync.py`
keeps those coverage-anchor grid/window notes visible across the handoff,
automation README, final verification matrix, and this release checklist,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_probe.py`
keeps the `right-shoulder-error-overshoot` boundary executable at `z = 0.25`,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_surface_sync.py`
keeps that exact `right-shoulder calibration debt` visible across the handoff,
automation README, final verification matrix, and this release checklist,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_boundary_probe.py`
keeps the bracketed right-shoulder calibration boundary executable on
`[0.20, 0.25]`, while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_boundary_surface_sync.py`
keeps the `coverage-anchor-specific-bracketed-right-shoulder` surface visible
across the handoff, automation README, final verification matrix, and this
release checklist,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_companion_reserve_probe.py`
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_growth_split_probe.py`
keep the next narrower Trigger 2 decomposition executable after the shoulder
boundary by preserving `companion-band-amplification-outpaces-error-growth`
and `local-error-growth-outpaces-band-growth` as closeout-visible companion
objects rather than letting the surface collapse back into generic shoulder
prose,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_scale_repair_probe.py`
keeps the exact same-point repair object executable, while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_companion_surface_sync.py`
and
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_scale_repair_note.py`
keep the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on
`anchor-specific local scale under-amplification`, the `+0.642` `sigma_z_hat`
repair, the `x1.160` local scale repair factor, and the `16.0%` shortfall,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_share_probe.py`
keeps the next same-point carry-over object executable by freezing
`anchor-repair-is-small-share-of-companion-surplus`, the `10.8%` repair share,
and the `89.2%` untouched companion surplus,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_access_share_probe.py`
keeps the one-scalar next rung executable by freezing
`single-share-local-scale-access-shortfall` and
`local_scale_access_shortfall_share = 10.8%`,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_access_surface_sync.py`
keeps the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on those
repair-share / access-share signatures instead of letting the surface collapse
back to the broader scale-repair prose,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_localization_probe.py`
keeps the residual-window localization object executable by freezing
`localized right-shoulder miss` over `whole-window undercoverage` on
`near_zero_grid`, with the unique failing shoulder at `z = 0.25`, the
anchor's off-shoulder reserve still at `9.221x` the failing shortfall, and the
overshoot companion's same-point reserve still at `4.414x` the anchor
shortfall, while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_localization_surface_sync.py`
keeps the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on that
localized right-shoulder story instead of letting the surface collapse back to
whole-window undercoverage prose,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe.py`
keeps
`Docs/research/phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe.md`
and
`run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_localization_probe()`
executable by freezing
`bounded-right-center-entry-mass-localization` on the failing shoulder
`z = 0.25` relative to the center anchor `z = 0.15`, with current
right-center covariance entry `0.094`, required bounded repair `1.731`,
increment `+1.636`, repaired entry shares `10.0%` / `15.9%` of anchor
right-row / center-column absolute covariance mass, required increment shares
`0.9%` / `1.9%` of companion right-row / center-column mass, and whole-row /
whole-column replay overshoot multiples `x112.487` / `x52.793`,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_covariance_entry_localization_surface_sync.py`
keeps the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on that
localized right-center entry-mass story instead of letting the surface drift
back to whole-row / whole-column replay prose,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract.py`
keeps the next implementation object executable by freezing
`phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract.md`
/
`run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_execution_contract()`
as `bounded-right-center-execution-contract`, with working grid
`z = 0.05`, `z = 0.15`, `z = 0.25`, the last-mile miss still fixed at
`10.8%`, the bounded right-center correlation increment still fixed at
`+0.126` (`28.8%` of the full gap), the raw covariance-entry increment
still fixed at `+1.636`, the anchor right side still retaining only
`35.2%` of left access while the companion restores it to `140.6%`
(`x3.996` directional flip), and the required lift still consuming only
`5.7%` / `1.0%` / `2.2%` of the single-entry / right-row / center-column
gaps,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_execution_contract_routing_guard.py`
keeps the handoff, automation README, frontier note, validation inventory,
STATE, final verification matrix, and this release checklist aligned on that
minimal execution contract instead of letting the live surface stop one rung
earlier at localization prose,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_seed_window_covariance_probe.py`
keeps the next window-object split executable by freezing
`covariance-coupling-and-scale-overshoot` on
`near_zero_grid`, with coverage anchor seed `202`, overshoot companion seed
`505`, covariance trace ratio `x5.292`, and uniform critical ratio `x1.225`,
while
`hddid-py/tests/validation/test_trigger2_policy_seed_window_covariance_surface_sync.py`
keeps the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on that local
covariance / `sigma_z_hat` access story instead of letting the surface drift
back to generic uniform-critical retuning,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_access_driver_probe.py`
keeps the next object-level driver split executable by freezing
`covariance-process-sigma-z-hat-access-bottleneck` and the
`x4.321` dominance of covariance amplification over critical drift,
while
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_probe.py`
keeps the narrower shoulder-center channel split executable by freezing
`right-shoulder-center-coupling-access-bottleneck`, the center anchor
`z = 0.15`, the `x61.178` shoulder-center correlation ratio, and the
`x161.408` coupled-sigma access ratio,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_driver_surface_sync.py`
keeps the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on those two
driver-bottleneck companions instead of letting the surface stop at the wider
`seed_window_covariance` prose,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_channel_probe.py`
keeps the narrower channel-level repair split executable by freezing
`covariance-entry-first-repair-target`, with the failing shoulder `z = 0.25`
relative to the center anchor `z = 0.15`, the raw shoulder-center covariance
entry still widening by `x303.784` while center `sigma_z_hat` widens only
`x1.882`, so the last-mile access repair still needs just `6.0%` of
companion covariance while the frozen-covariance fallback would squeeze center
`sigma_z_hat` to `5.5%` of anchor center scale,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_center_coupling_channel_surface_sync.py`
keeps the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on that
covariance-entry repair-target companion instead of letting the surface jump
straight from the broader driver bottleneck prose to the later
`coupling-first-repair-target` summary,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_center_coupling_repair_probe.py`
keeps the next repair-target split executable by freezing
`coupling-first-repair-target` on the failing shoulder `z = 0.25` relative to
the center anchor `z = 0.15`, with the minimal repair object compressed to
just `11.3%` of companion coupled access, or `29.9%` of companion
shoulder-center correlation, while freezing current-correlation-only shoulder
inflation at `x6.941` of companion shoulder scale,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_center_coupling_repair_surface_sync.py`
keeps the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on that
coupling-first repair-target companion instead of letting the surface stop at
the broader `right-shoulder-center-coupling-access-bottleneck` channel split,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_directionality_probe.py`
keeps the next directional split executable by freezing
`right-shoulder-directional-covariance-suppression`, with left shoulder
`z = 0.05` and failing right shoulder `z = 0.25` measured against center
`z = 0.15`, so the coverage anchor right-side coupled-access share stays at
`35.2%`, the overshoot companion flips that share to `140.6%`, and the
resulting directional flip stays pinned at `x3.996`,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_directionality_surface_sync.py`
keeps the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on that
directional follow-up instead of letting the surface fall back to a symmetric
`near_zero_grid` rebalance story,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_gap_probe.py`
keeps the bounded shoulder-center bridge executable by freezing
`partial-correlation-gap-bridge-target` on failing shoulder `z = 0.25`
relative to center `z = 0.15`, with the repair increment pinned at `+0.126`,
the consumed gap share pinned at `28.8%`, the unused headroom pinned at
`71.2%`, and the remaining headroom multiple pinned at `x2.476`,
while
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_cross_shoulder_sign_probe.py`
keeps the next geometry-level split executable by freezing
`cross-shoulder-sign-fracture` on `z = 0.05`, `z = 0.15`, and `z = 0.25`,
with anchor cross-shoulder correlation pinned at `-0.082`, companion
magnitude pinned at `x6.708`, and anchor / companion right-to-left center
shares pinned at `30.2%` / `97.7%`,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_correlation_gap_surface_sync.py`
and
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_cross_shoulder_sign_surface_sync.py`
keep the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on that bounded
bridge plus local sign-fracture follow-up instead of letting the surface
collapse back to generic covariance amplification prose,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_covariance_entry_factor_probe.py`
keeps the next factor-level split executable by freezing
`correlation-led-covariance-entry-repair-target`, with the failing shoulder
`z = 0.25` relative to the center anchor `z = 0.15`, so the raw covariance
entry gap stays pinned at total `x303.784`, with factor contributions
`x2.638`, `x1.882`, and `x61.178`, while the log-gap contribution shares stay
fixed at `17.0%`, `11.1%`, and `72.0%` for shoulder `sigma_z_hat`, center
`sigma_z_hat`, and shoulder-center correlation,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_covariance_entry_factor_surface_sync.py`
keeps the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on that
correlation-led covariance-entry factor follow-up instead of letting the
surface fall back to broad shoulder inflation or center-scale retuning prose,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot.py`
keeps the repair-target snapshot executable by freezing
`Docs/research/phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot.md`
and
`run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_repair_target_snapshot()`
as the current `directional-direct-residual-correlation-repair-target` object
for `rmse-outpaces-average-se`,
`sigma_z_hat-scale-dispersion`, and
`coverage-anchor-vs-overshoot-companion`, including the
`z = 0.05` / `z = 0.15` / `z = 0.25` directional slice, the
`35.2%` / `140.6%` right-to-left center shares, the `x3.996`
directional flip, the `-0.082` / `+0.434` partial cross-shoulder split, the
`0.2%` center mediation, and the bounded bridge
`0.133` / `+0.126` / `28.8%` / `72.0%`,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_repair_target_snapshot_note.py`
and
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_repair_target_snapshot_surface_sync.py`
keep the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on that
repair-target snapshot rather than letting the surface collapse back to
broader calibration-debt prose,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot.py`
keeps the next narrower lane snapshot executable by freezing
`Docs/research/phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot.md`
and
`run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot()`
as the current `directional-covariance-entry-repair-lane` object, including
the `10.8%` last-mile miss at `z = 0.25`, the bounded bridge `28.8%`,
the infeasible sign-healing target `-3.399`, the required absolute
shoulder-center covariance target `1.731`
(`|covariance(z = 0.25, z = 0.15)| = 1.731`), the companion covariance share
`6.0%`, and the covariance-fixed fallback `sigma_z_hat` levels
`0.220` / `0.177` that each leave only `5.5%` of the current anchor scale,
while
`hddid-py/tests/validation/test_trigger2_policy_coverage_anchor_shoulder_correlation_repair_lane_snapshot_surface_sync.py`
keeps the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on that
correlation-repair lane snapshot rather than letting the surface collapse
back to covariance-entry factorization or denominator compression prose,
and
`hddid-py/tests/unit/test_monte_carlo_widening_policy_calibration_debt_snapshot.py`
keeps the aggregate Trigger 2 calibration-debt snapshot executable by freezing
`Docs/research/phase7_monte_carlo_widening_policy_calibration_debt_snapshot.md`
and
`run_phase7_monte_carlo_widening_policy_calibration_debt_snapshot()`
as the canonical digest of `rmse-outpaces-average-se`,
`sigma_z_hat-scale-dispersion`,
`coverage-anchor-vs-overshoot-companion`, and
`coupling-first-repair-target`, while freezing the last-mile same-point
shortfall / coupled-access share / correlation share / fixed-correlation
shoulder inflation at `10.8%`, `11.3%`, `29.9%`, and `x6.941`,
while
`hddid-py/tests/validation/test_trigger2_policy_calibration_debt_snapshot_note.py`
and
`hddid-py/tests/validation/test_trigger2_policy_calibration_debt_snapshot_surface_sync.py`
keep the handoff, automation README, frontier note, validation inventory,
final verification matrix, and this release checklist aligned on that
aggregate calibration-debt snapshot instead of forcing future reruns to
reconstruct the calibration chain from five narrower companion notes,
and
`hddid-py/tests/validation/test_trigger2_readme_primary_loop_routing.py`
and
`hddid-py/tests/validation/test_trigger2_handoff_primary_loop_routing.py`
keep the release-facing README and closeout handoff explicitly frozen on
`trigger2-policy-spec` as the default feature bundle, while
`hddid-py/tests/validation/test_trigger2_state_primary_loop_routing.py`
keeps `.planning/STATE.md` frozen on the aggregate closeout snapshot, the
`trigger2-policy-spec` primary loop, and the
`historical / explanatory provenance` split for source-level Trigger 2 notes,
while
`hddid-py/tests/validation/test_trigger2_historical_entry_alignment.py`
keeps historical Trigger 2 source-level rungs from drifting back into
“current entry” wording, and
`hddid-py/tests/validation/test_next_milestone_trigger_snapshot_surface_sync.py`
keeps the aggregate closeout snapshot visible across live routing surfaces, and
`hddid-py/tests/validation/test_trigger3_outer_inference_contract_surface_sync.py`
keeps the Trigger 3 object-contract note visible in live routing instead of
leaving it implicit behind `test_trigger*.py`.
`hddid-py/tests/validation/test_trigger3_outer_inference_repair_obligations_surface_sync.py`
keeps the Trigger 3 repair-obligations companion visible alongside the object
contract, so future parity planning does not silently drop the canonical repair
trio from the closeout surface.

Keep
`Docs/research/phase7_next_milestone_trigger_snapshot.md`
aligned with
`run_phase7_next_milestone_trigger_snapshot(...)`:
the aggregate closeout snapshot should continue to show no open triggers and
the recommended feature bundle `trigger2-policy-spec`.
Keep
`Docs/research/phase7_monte_carlo_widening_policy_spec.md`
aligned with
`run_phase7_monte_carlo_widening_policy_spec()`:
the final-verification smoke should continue to replay the bounded widening
policy and keep its target gate pinned at `trigger2-partial-widening-only`.

## Long-batch Trigger 2 replays

```bash
pytest hddid-py/tests/validation/test_monte_carlo_runtime_probe.py -q
pytest hddid-py/tests/validation -q
```

Treat both commands above as long-batch replays, not quick spot check
commands. The runtime-probe replay remains the dominant Trigger 2 cost surface,
so release-facing closeout reruns should keep it separate from the targeted
Phase 7 contract checks above.

## Blocker review

Do not describe the package as empirically release-ready unless this review is
still honest:

- Keep the empirical Section 6 lane tied to the repository-local payload at
  `hddid-py/data/section6_county_panel.tsv` plus
  `hddid-py/data/section6_manifest.json`.
  Review `Docs/research/phase7_section6_loader_object_contract.md` before
  widening this language: future loader intake must continue to expose
  `analysis_rows`, `sample_metadata`, `lane_metadata`, and
  `linear_covariate_manifest`, rather than collapsing Trigger 1 into a black-box
  “loader runs” claim. When the payload changes, prefer
  `audit_phase7_section6_loader_payload(...)` to check shape provenance,
  lane swap, and method-anchor drift before softening this blocker review.
  Review `Docs/research/phase7_section6_paper_schema_source_audit.md` too:
  `run_phase7_section6_paper_schema_audit()` and
  `build_phase7_section6_paper_schema_report(...)` should continue to show that
  future empirical loader / manifest work must not inherit simulation settings
  by accident, so empirical Section 6 keeps `2005-2007`,
  `4th degree trigonometric polynomial basis`, and
  `95% confidence intervals` instead of drifting to Monte Carlo's
  `8th degree trigonometric polynomial basis` and
  `90% confidence intervals`.
- Keep the broken R outer inference path in `reference-only` status; it is not a
  release-ready parity oracle for `CIpoint` or `CIuniform`. Review
  `Docs/research/phase7_r_snapshot_oracle_surface_summary.md` and
  `run_phase7_r_snapshot_oracle_surface_summary(...)` first: they should
  continue to keep the Monte Carlo archived helper on `paper-trigonometric`,
  the source-level estimator parity lane on `r-parity-polynomial`, and the
  outer-inference comparison target on `aggregated covariance-process objects`,
  rather than letting archived helper prose or broken `CIuniform` drift back
  into the release-facing oracle story. Review
  `Docs/research/phase7_outer_inference_source_audit.md`,
  `Docs/research/phase7_outer_inference_paper_backed_replacement_path.md`
  `Docs/research/phase7_outer_inference_object_serialization_contract.md` and
  `Docs/research/phase7_outer_inference_trigger_gate.md`
  `Docs/research/phase7_outer_inference_determinism_probe.md`
  before relaxing this language: until `RBUG-005` and `RBUG-013` are removed,
  future parity must stay anchored on aggregated covariance-process objects,
  `covariance_at_grid`, and `uniform_critical_value`, not archived `CIuniform`.
  The executable source-audit helper is
  `audit_r_snapshot_outer_inference_source(...)`; it should continue to confirm
  that `RBUG-011` is only a non-blocking debug token while `RBUG-005` and
  `RBUG-013` remain the actual blockers.
  The canonical helper for this object surface remains
  `build_phase7_outer_inference_object_contract(...)`. The final-verification
  smoke helper for replaying the current canonical object contract is
  `build_phase7_outer_inference_reference_contract(...)`.
  The executable gate helper is `run_phase7_outer_inference_trigger_gate()`;
  it should continue to return `archived-r-blocked-paper-backed-ready`.
  The fixed-seed replay helpers are
  `run_phase7_outer_inference_determinism_probe()` and
  `replay_phase7_outer_inference_uniform_critical_value(...)`; they should
  continue to show that changing `random_state` or `n_boot` only moves the
  downstream `uniform_critical_value`, not the upstream `covariance_at_grid`
  contract.
  The repair-obligations companion is
  `Docs/research/phase7_outer_inference_repair_obligations.md`; the executable
  helper is `run_phase7_outer_inference_repair_obligations_probe()`, and it
  should continue to keep `repair-undefined-uniform-center`,
  `replace-last-fold-tc-with-covariance-process`, and
  `pin-monte-carlo-functional-fields` visible as the canonical open trio.
- Keep the aggregate next-milestone intake snapshot in
  `Docs/research/phase7_next_milestone_trigger_snapshot.md`;
  `run_phase7_next_milestone_trigger_snapshot(...)` should continue to report
  `trigger2-policy-spec` until at least one trigger is truly open.
- Keep the current remaining product-complete blocker fixed at
  `monte_carlo_validation_ready`. Trigger 1 is already satisfied on
  `hddid-py/data/section6_county_panel.tsv` plus
  `hddid-py/data/section6_manifest.json`, so release-facing honesty must not
  backslide to the older `missing-local-dataset` story while Trigger 2 still
  owns the live completion gate. The current open trigger must stay
  `trigger2-runtime-evidence`, and the bounded blocker packet must remain
  machine-readable as `trigger2-runtime-evidence-packet-open` with a rejected
  candidate (`runtime-evidence-rejected`, `NonpositiveVarianceError=1` at
  `omega_f_hat`) on `DGP2/500/50` / seed `303` / `7/9 -> 8/9` /
  `5 fresh reruns`. The downstream quality-risk evidence still remains visible
  as `quality-risk-keeps-trigger2-bounded` / `rmse-outpaces-average-se` against
  the `DGP1/500/50` comparison slice.
- Keep
  `run_phase7_monte_carlo_widening_policy_runtime_evidence_admission_gate(...)`
  visible on the same release-facing checklist as the runtime-admission
  candidate verifier: it must preserve `trigger2-runtime-evidence-packet-open`,
  `runtime-evidence-rejected`, `DGP2/500/50`, seed `303`, `7/9 -> 8/9`, and
  `trigger2-policy-spec` without overstating the current live gate. Its
  candidate diagnostic surface must also remain visible through
  `candidate_success`, `candidate_nonparametric_coverage`,
  `candidate_typed_invalidity_counts`, `candidate_typed_invalidity_examples`,
  `runtime_evidence_admission_candidate_primary_invalidity_name`,
  `runtime_evidence_admission_candidate_primary_invalidity_matrix_name`,
  `runtime_evidence_admission_candidate_primary_invalidity_min_eigenvalue`,
  `runtime_evidence_admission_candidate_primary_invalidity_replication_seed`,
  and the `candidate typed invalidity` digest line so rejected reruns expose
  `NonpositiveVarianceError` at `omega_f_hat` instead of hiding behind a
  generic missing-coverage status.
- Keep `run_phase7_release_matrix_runner(...)` evidence explicit about the same
  boundary: the Monte Carlo row should carry
  `runtime admission trigger2-runtime-evidence-packet-open`,
  `runtime admission candidate typed invalidity NonpositiveVarianceError=1`,
  and the candidate invalidity example beside downstream quality-risk evidence,
  so release promotion remains tied to the live runtime-admission blocker rather
  than a validation-only quota closure.
- Keep the repo-root literal `$CODEX_HOME/automations/**/memory.md` bad artifact
  escalated as shared-worktree hygiene debt. Re-run
  `git status --short --branch` and `find '$CODEX_HOME' -maxdepth 3 -type f`;
  if those paths still appear, treat them as foreign dirty artifacts rather
  than a valid automation memory sink until a repo-root-authorized cleanup loop
  removes them.
- Keep release maturity at `research-alpha` while the hardening and validation
  backlog remains open, specifically while `monte_carlo_validation_ready`
  remains blocked.
