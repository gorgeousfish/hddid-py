# Validation lanes and limitations

`hddid` is a `research-alpha` package. Its release-facing validation story is
explicitly lane-based so that paper evidence, archived R sanity checks, and
empirical blockers do not get merged into one undifferentiated status line.

## Truth order

Interpret validation claims in this order:

1. The HDDID paper and paper-aligned mathematical derivations.
2. The current Python implementation and typed diagnostics.
3. Archived R source paths that remain explainable.
4. Archived R documentation text.

The archived R snapshot is therefore a constrained reference. It is not an
installation dependency and it is not the final oracle for empirical claims.

## Active validation lanes

| Lane | What it currently validates | Current status | Guardrail |
| --- | --- | --- | --- |
| `paper-trigonometric` | Paper-first synthetic Monte Carlo objects built around trigonometric bases. | `staged-smoke` | Reduced smoke evidence does not replace the full paper matrix. |
| `r-parity-polynomial` | Explainable raw-object sanity checks against the archived R snapshot. | `validated` for the narrow raw Eq. (3.1) scope | This lane does not replace the paper Monte Carlo lane. |
| `real-data-asset-audit` | Repository-local Section 6 readiness for raw data plus a loader. | ready on `hddid-py/data/section6_county_panel.tsv` + `hddid-py/data/section6_manifest.json` + `hddid-py/src/hddid/section6_loader.py` | A figure inventory or other reference material still does not upgrade the lane into an empirical success. |
| `reduced-real-data-workflows` | Three source-checkout reduced Section 6 workflows: official-QE, LAUS--QE, and LAUS--CCDB. | public estimator and Eq. (4.2) parametric inference complete; requested Eq. (4.3) nonparametric paths stop at named covariance boundaries | Software-output evidence only; not a substantive minimum-wage effect estimate, not the 38-characteristic design, and not Figure 2. |

## What remains blocked

### Empirical Section 6 path

The Section 6 evidence now has two different layers, and they should not be
collapsed into one status line.  The first layer is the canonical loader/source
payload.  It carries all three Trigger 1 prerequisites together:

- a repository-local raw Section 6 dataset
- a canonical manifest aligned to the paper schema
- a runnable repository-local Python loader payload

The current canonical payload is `hddid-py/data/section6_county_panel.tsv`
paired with `hddid-py/data/section6_manifest.json`, and the executable loader
asset remains `hddid-py/src/hddid/section6_loader.py`.

For this lane, an archived R snapshot loader on its own is still not enough.
`audit_section6_assets(...)` now reports the repository-local payload as ready,
and `audit_section6_provenance_gate(...)` keeps the lane honest by checking the
canonical manifest alignment plus the loader object contract on every replay.
That release-facing provenance read is now intentionally object-level too:
`audit_phase7_section6_loader_payload(...)` remains the executable guard for
the current Trigger 1 loader surface, so the ready path must keep
`analysis_rows`, `sample_metadata`, `lane_metadata`, and
`linear_covariate_manifest` machine-readable instead of collapsing back into a
generic "loader runs" claim. On the current canonical payload, that also means
raw rows must declare exactly one time shape: `year` panel rows and
`difference_window` county-difference rows cannot be mixed silently. The
loader also rejects ragged CSV/TSV rows and blank county-characteristic cells
before they can distort `partial-local-payload` coverage or
interaction-design construction. The
accepted difference-shaped replay still has to pin
`difference_window = 2005-2007`, and the empirical Section 6 contract must
continue to preserve the paper-backed `95% confidence intervals` rather than
inherit Monte Carlo settings by accident.

The loader-object audit also reports local coverage separately from the paper
target. The current repository payload is `partial-local-payload`: it observes
two county-characteristic columns, `poverty_rate` and `urban_share`, while the
paper schema still targets `38` county characteristics and a reported
`703`-covariate linear block. The manifest records that block as
`pairwise-baseline-characteristic-interactions` with count formula
`choose(38, 2)`, so the count arithmetic is pairwise interactions among the 38
characteristics rather than 38 main effects plus those interactions. That
coverage status is not a loader-object failure, but it is a
hard boundary against presenting the local payload as a completed empirical
Section 6 replication.
The county-difference preflight also promotes that coverage gap into estimation
readiness: until the local payload reaches the paper's 38 county-characteristic
fields, the design metadata must keep `incomplete-county-characteristics` and
the current `36` missing fields machine-readable alongside the zero control counties
and sample-size blockers. The same preflight now reports
`minimum_data_requirements` explicitly: the current checkout needs at least
`10` county differences for the fourth-degree empirical trigonometric basis,
one additional control county, and all `38` paper
county-characteristic fields before Section 6 estimation can be attempted. The paper-aligned `703`-covariate
linear block remains visible as design metadata, but it is not a
low-dimensional sample-size blocker because the empirical path is high
dimensional.

The same release-facing read also has to keep the paper-backed sample and
schema anchors explicit. `sample_metadata` must continue to replay
`baseline_universe = 2006 federal-binding states` together with the explicit
excluded states `New Hampshire` and `Pennsylvania`, rather than collapsing the
sample contract into a treated-state count only. The empirical Figure 2 schema
must likewise stay pinned to `median income` / `population` as the two
nonparametric lanes, the reported `703`-covariate linear block tied to `38`
county characteristics, and the empirical `4th degree trigonometric polynomial
basis` with `95% confidence intervals`. The current manifest-backed Python
construction is the pairwise baseline-characteristic interaction block whose
count is `choose(38, 2) = 703`; counting the 38 main effects separately would
give 741 columns and is not admitted by the current source map. That paper contract is intentionally distinct
from the Monte Carlo lane's `8th degree trigonometric polynomial basis` with
`90% confidence intervals`, so future empirical loader / manifest work must not inherit simulation settings by accident.

That raw asset check is now intentionally canonical-loader scoped: placeholder
files such as `section6_notes.csv` do not count as a Section 6 dataset, and the
asset gate only recognizes the current loader-supported raw paths
`section6_county_panel` / `section6_county_differences` under `data/` or
`hddid-py/data/` as `.csv` / `.tsv`.
If both canonical raw shapes appear in the same repository read, the asset gate
keeps the lane blocked with `multiple-canonical-raw-shapes` instead of
pretending the repository already has one canonical empirical payload.

The second layer is the reduced real-data software workflow.  Three generated
summaries now show that real county-level records can enter the public
estimator and Eq. (4.2) parametric inference path:

- `section6_qe_reduced_hddid_run_summary.json` uses the official-QE
  county-difference outcome path and six exported application covariates.
- `section6_laus_qe_hddid_run_summary.json` joins February 2005 and 2007 LAUS
  county outcome differences to the official QE application covariates.
- `section6_laus_ccdb_reduced_qe_hddid_run_summary.json` joins the same LAUS
  outcome layer to the reduced QE covariates rebuilt from official CCDB table
  text.

All three workflows complete the public fit and Eq. (4.2) parametric inference
for the two real-data lanes, while the requested Eq. (4.3) nonparametric paths
stop with named covariance diagnostics rather than reporting unsupported
curves.  `section6_reduced_parametric_plot_summary.json` records
`workflow_count = 3` and summarizes the same parametric target objects as a
software-output visualization.  This is stronger than a raw asset audit, but it
is still not the full empirical Section 6 design: it does not reconstruct the
exact 38 baseline characteristics, does not build the 703-covariate input, does
not report nonparametric treatment-effect curves, and does not reproduce
Figure 2.

### Archived R outer inference path

The package also keeps the broken R outer inference path in a reference-only
state. The archived snapshot still contains a broken R outer inference path, so
release docs do not present it as a parity oracle for `CIpoint` or
`CIuniform`.

### Release-facing Trigger 2 honesty

The release-facing gate for the current package boundary is still
`run_phase7_release_maturity_gate(...)`. Its current verdict remains
`release-facing-state-still-research-alpha-honesty`: live routing still stays `trigger2-policy-spec`,
the open Trigger 2 release-facing item still stays
`trigger2-runtime-evidence`, and the remaining product-complete blocker still
stays `monte_carlo_validation_ready` under
`quality-risk-keeps-trigger2-bounded` with the live quality-risk driver on
`DGP2/500/50` against the `DGP1/500/50` comparison slice, seed `303`, with the
bounded witness gap still at `7/9 -> 8/9` and only `5 fresh reruns` left worth
spending.

## Known limitations

- Release maturity remains `research-alpha`.
- The current implementation supports one-dimensional `z`.
- The Monte Carlo lane is still staged smoke rather than the full paper matrix.
- The empirical source lane is ready on the repository-local Section 6 payload
  at the asset/provenance/loader-object level, and the reduced real-data
  workflow lane now runs public estimation plus Eq. (4.2) parametric inference
  on official/source-aligned county data.  The full Section 6 treatment-effect
  case study remains blocked because the exact 38-characteristic map and
  703-covariate estimator-ready join are still missing.
- The archived R outer inference path stays reference-only while the snapshot
  retains missing files, naming drift, and broken outer output code.

## How to read the current status

- Use `paper-trigonometric` for paper-first synthetic evidence.
- Use `r-parity-polynomial` for narrow raw-object regression checks.
- Use `real-data-asset-audit` to inspect the current repository-local Section 6 payload.
- Use `reduced-real-data-workflows` to inspect the official-QE, LAUS--QE, and
  LAUS--CCDB software-output runs.

That split is deliberate. It prevents the package from overstating parity or
empirical readiness while the hardening and validation backlog is still active.
