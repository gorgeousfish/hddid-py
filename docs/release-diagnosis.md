# Release diagnosis

This document records the current release-stage diagnosis for `hddid` 0.1.0.
It is a release decision aid, not a substitute for the source-checkout gates in
`hddid.validation`.

## Current stage

`hddid` is a Python package for the paper-first HDDID estimator workflow. The
current package exposes a minimal root workflow through Eq. (3.1) via
`fit_hddid(...)`, lower-level score/estimation/inference helpers, repository
local Section 6 data-readiness helpers, and source-checkout validation gates.

The package is in the validation stage. Core implementation, typed results,
focused numerical fixtures, a local PDF-gated manuscript replication run, and
reduced Section 6 real-data software workflows exist, but formal Monte Carlo
validation, manuscript front-matter metadata, and the full 38-characteristic
Section 6 empirical estimation design are not complete.

## Release decision

Do not make a formal public release from the current source tree.

The current state is suitable only as a `0.1.0` research-alpha / pre-release
artifact. Built wheel and sdist artifacts now have a clean package boundary:
the archive audit reports zero space-suffixed modules and zero internal
probe/trigger/phase7/automation-state modules. The source checkout is still
intentionally broader because it contains repo-side validation and research
materials. The formal release gate remains blocked by Monte Carlo validation,
not by built artifact pollution.

Resolved in the current release-diagnosis pass: the stale secondary
`README 2.md` no longer contradicts the current `fit_hddid(...)` root workflow,
README runtime dependencies now include PyYAML, and release-facing pytest
configuration no longer depends on an unloaded `pytest-asyncio` plugin option.
The release checklist now also separates installed-wheel import smoke from the
source-checkout artifact audit, so a fresh wheel environment is not asked to run
repo-local release gates that require the repository root. Installed-wheel
callable smoke now also checks that source-checkout-only release helpers raise
`SourceCheckoutRequiredError` instead of leaking an internal
`ModuleNotFoundError`.

Current source-checkout gate snapshot:

- `run_phase7_monte_carlo_feature_completion_gate(".")`:
  `monte_carlo_validation_ready_status = "blocked"`.
- Monte Carlo blocker: `trigger2-runtime-evidence-packet-open`.
- Runtime-evidence candidate: rejected on `DGP2/500/50`, seed `303`;
  `candidate_success = False`, `candidate_nonparametric_coverage = None`,
  typed invalidity `NonpositiveVarianceError=1`.
- Primary invalidity: `matrix_name = "omega_f_hat"`, minimum eigenvalue
  `-38.0305649466643`, replication seed `883193502`.
- Diagnostic boundary: the orthogonal-score covariance is positive on the same
  replay (`omega_f_orthogonal_score_min_eigenvalue = 21.499226581401206`), but
  it is retained only as diagnostic evidence, not as a fallback covariance.
- Quality-risk driver: `rmse-outpaces-average-se`.
- Quality-risk blocker: `quality-risk-keeps-trigger2-bounded`.
- Binding design: `DGP2/500/50`, coverage `0.778`, RMSE `4.171`, average SE
  `3.992`, SE reserve `-0.179`.
- Best design: `DGP1/500/50`, coverage `0.889`, RMSE `1.911`, average SE
  `2.268`, SE reserve `0.357`.
- `run_phase7_release_maturity_gate(".")`: blocked on
  `monte_carlo_validation_ready`.
- `run_phase7_release_matrix_runner(".")`: blocked on
  `monte_carlo_validation_ready` and `release_maturity_ready`.

Current built-archive snapshot:

- `hddid-0.1.0-py3-none-any.whl`: formal archive audit `ready`,
  `space_suffix_modules = 0`, `internal_release_modules = 0`.
- `hddid-0.1.0.tar.gz`: formal archive audit `ready`,
  `space_suffix_modules = 0`, `internal_release_modules = 0`.

## P0 blockers

### Monte Carlo validation is not release-ready

Evidence: `run_phase7_monte_carlo_feature_completion_gate(".")` reports a
blocked Monte Carlo validation state. The runtime-evidence admission candidate
for `DGP2/500/50`, seed `303`, fails with `NonpositiveVarianceError=1` at the
paper-difference `omega_f_hat` object, so it supplies no nonparametric coverage
witness. The binding DGP2 design also has negative SE reserve: average SE is
below RMSE.

Why it matters: the package cannot claim formal release validation or paper
Monte Carlo adequacy while the bounded quality-risk gate remains open.

Minimum fix: produce a predeclared Monte Carlo validation asset that clears the
current runtime-evidence packet and quality-risk conditions, or change the
estimator/inference construction so the binding design no longer has negative
SE reserve and the resulting metrics satisfy the gate.

Minimum verification:

```bash
PYTHONPATH=hddid-py/src python - <<'PY'
from hddid.validation import (
    run_phase7_monte_carlo_feature_completion_gate,
    run_phase7_release_maturity_gate,
    run_phase7_release_matrix_runner,
)

feature = run_phase7_monte_carlo_feature_completion_gate(".")
release = run_phase7_release_maturity_gate(".", feature_gate=feature)
matrix = run_phase7_release_matrix_runner(".", feature_gate=feature)
print(feature.monte_carlo_validation_ready_status)
print(feature.monte_carlo_validation_ready_blocker)
print(release.gate_status, release.blocking_gate_names)
print(matrix.status, matrix.blocking_gate_names)
PY
```

The gate is releasable only when the Monte Carlo status is `ready` and release
maturity no longer blocks on `monte_carlo_validation_ready`.

### Manuscript front matter remains incomplete

Evidence: `paper/论文/main.tex` still contains author TODO fields.  The current
`paper/replication/outputs/all_replication_summary.json` records a completed
local `pdf_build_gate` with output `/private/tmp/pyhddid-jss-build/main.pdf`,
but that local build is not a substitute for author-confirmed front matter or
final submission metadata review.

Why it matters: a JSS-style submission archive cannot contain author
placeholders, even when the reviewer-facing manuscript/package checks and local
PDF build pass.

Minimum fix: fill author metadata and rerun the full replication entry point
with the PDF gate from a clean source checkout.

Minimum verification:

```bash
rg -n "TODO|TBD|placeholder" paper/论文/main.tex paper/submission-readiness.md
python3 paper/replication/replicate_all.py --with-pdf
python3 paper/submission/run_reviewer_checks.py
```

The resulting `all_replication_summary.json` must contain a completed
`pdf_build_gate`, the LaTeX build must have no fatal errors, unresolved
citations, or unresolved references, and the publication audit must no longer
report author/address front-matter placeholders.

## P1 gaps

### Full Section 6 case-study upgrade remains incomplete

Evidence: the current Section 6 asset/provenance/loader audit is ready, and the
reduced real-data workflows now run the public estimator and Eq. (4.2)
parametric inference on three source-checkout routes: official-QE, LAUS--QE, and
LAUS--CCDB.  The same runs stop the requested Eq. (4.3) nonparametric paths at
named covariance boundaries, and `section6_reduced_parametric_plot_summary.json`
records `workflow_count = 3` for the reduced parametric software-output plot.
The full case-study upgrade remains incomplete because the source route still
lacks the exact 38-characteristic map and 703-covariate estimator-ready join.

Why it matters: the paper and package can claim real-data software workflow
execution for the reduced official/source-aligned inputs, but not substantive
minimum-wage effects, nonparametric treatment-effect curves, the published
38-characteristic design, or a replicated Figure 2.

Minimum fix: map the exact method-paper 38 characteristics to an admitted
County Data Book or equivalent source extract, build the deterministic
703-covariate estimator-ready join, then rerun the Section 6 design preflight
and reduced/full estimator scripts without promoting candidate-source triage
into a treatment-effect result.

Minimum verification:

```bash
PYTHONPATH=hddid-py/src python3 paper/replication/replicate_section6_asset_audit.py
PYTHONPATH=hddid-py/src python3 paper/replication/replicate_section6_design_preflight.py
PYTHONPATH=hddid-py/src python3 paper/replication/replicate_section6_qe_reduced_hddid_run.py
PYTHONPATH=hddid-py/src python3 paper/replication/replicate_section6_laus_qe_hddid_run.py
PYTHONPATH=hddid-py/src python3 paper/replication/replicate_section6_laus_ccdb_reduced_qe_hddid_run.py
PYTHONPATH=hddid-py/src python3 paper/replication/replicate_section6_reduced_parametric_plot.py
PYTHONPATH=hddid-py/src python -m pytest \
  hddid-py/tests/validation/test_manuscript_replication_entrypoint.py \
  hddid-py/tests/validation/test_manuscript_citation_evidence_contract.py \
  -q
```

Any future full empirical estimates also need generated tables or figures and a
claim-level evidence-map entry that distinguishes them from the existing
reduced software-output workflows.

### Reference parity is still narrow

Evidence: the current parity coverage strongly supports Eq. (3.1) snapshots,
but not full Eq. (4.2), Eq. (4.3), uniform-band, and Monte Carlo metrics against
an independent reference implementation.

Why it matters: formal release and software-paper claims should not overstate
equivalence to the R/reference implementation.

Minimum fix: add fixed reference fixtures for Eq. (4.2) sparse directions,
Eq. (4.3) projection/covariance objects, and nonparametric interval outputs.

Minimum verification:

```bash
PYTHONPATH=hddid-py/src python -m pytest \
  hddid-py/tests/parity \
  hddid-py/tests/unit/test_inference_parametric.py \
  hddid-py/tests/unit/test_inference_nonparametric.py \
  -q
```

### README and API docs still expose broad validation-only surfaces

Evidence: README and API reference document many `run_phase7_*` validation
helpers. Clean wheel smoke confirms the documented validation import names
resolve, and callable smoke now confirms source-checkout-only helpers fail with
`SourceCheckoutRequiredError` rather than `ModuleNotFoundError`. The
release-control workflows still depend on repository-local notes, generated
paper outputs, and current source-tree state. Installed-wheel smoke therefore
must not be treated as proof that every repo-side release gate can run as a
standalone user API.

Why it matters: users should see a package API, not a release-control surface.
The distinction is also necessary for a clean formal archive.

Minimum fix: keep user-facing README focused on install, `fit_hddid`, core
inference helpers, Section 6 limitations, and replication commands. Move or
collapse long validation-only helper lists into source-checkout release docs.

Minimum verification:

```bash
rg -n "Trigger|trigger|phase7|handoff|closeout|seed303" hddid-py/README.md
PYTHONPATH=hddid-py/src python -m pytest \
  hddid-py/tests/validation/test_release_readme_contract.py \
  hddid-py/tests/validation/test_public_documented_import_contract.py \
  -q
```

## P2 gaps

- `fit_hddid(...)` stops at Eq. (3.1). Parametric and nonparametric inference
  remain explicit lower-level calls. This is acceptable for research-alpha, but
  a later user-facing release should consider an integrated inference workflow
  or a clearer post-estimation API.
- Package metadata is honest for research-alpha, but formal release metadata
  still needs a real version, docs/issues/changelog URLs, and a documented
  dependency support matrix.
- README now labels the package-root import block as common entry points rather
  than the complete `hddid.__all__` surface. A formal release should still give
  the full root API contract in API reference material.
- Current tests include many surface-contract checks. The formal validation
  command should continue separating numerical evidence from release-control
  synchronization tests.

## P3 polish

- Trim project-management language from the README once the formal public API
  is settled.
- Keep `docs/release-checklist.md` short enough for an external maintainer to
  execute without reading the full Phase 7 history.
- Re-run final bibliography and source-evidence checks before any manuscript
  submission archive.

## Current verification commands

The latest release-facing verification set is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=hddid-py/src python -m pytest \
  hddid-py/tests/unit/test_fit_hddid.py \
  hddid-py/tests/unit/test_public_root_exports.py \
  hddid-py/tests/unit/test_release_maturity_gate.py \
  hddid-py/tests/unit/test_release_matrix_runner.py \
  hddid-py/tests/unit/test_validation_source_checkout_boundary.py \
  hddid-py/tests/validation/test_release_artifact_content.py \
  hddid-py/tests/validation/test_release_docs_contract.py \
  hddid-py/tests/validation/test_release_readme_contract.py \
  hddid-py/tests/validation/test_public_documented_import_contract.py \
  hddid-py/tests/validation/test_packaging_metadata.py \
  hddid-py/tests/validation/test_manuscript_replication_entrypoint.py \
  hddid-py/tests/validation/test_manuscript_citation_evidence_contract.py \
  hddid-py/tests/validation/test_manuscript_section6_payload_boundary.py \
  -q
python -m build hddid-py --sdist --wheel --outdir /tmp/pyhddid-formal-boundary-dist
python -m twine check /tmp/pyhddid-formal-boundary-dist/*
```

Those commands verify release-facing contracts and clean built artifacts. They
do not prove formal release readiness while the Monte Carlo and manuscript
submission gates above remain open.
