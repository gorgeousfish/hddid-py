# API Reference

This page summarizes the public API for `hddid` version `0.1.0`.  It is written
for users and reviewers who need to understand the package boundary used by the
manuscript.

## Package Root

Common package-root imports:

```python
from hddid import (
    HDDIDResult,
    NuisancePayload,
    bspline_sieve_basis,
    build_score_payload,
    diagnose_nonparametric_omega_f,
    estimate_eq31_mainline,
    estimate_nonparametric_inference,
    estimate_parametric_inference,
    fit_hddid,
    polynomial_sieve_basis,
    suggest_basis_degree,
    trigonometric_sieve_basis,
)
```

The following names are available from `import hddid`:

| Group | Names |
| --- | --- |
| Metadata | `__version__` — installed package version string (e.g. `"0.1.0"`) |
| Basis construction | `polynomial_sieve_basis`, `trigonometric_sieve_basis`, `bspline_sieve_basis`, `suggest_basis_degree` |
| Cross-fitting and nuisance prediction | `CrossfitFold`, `CrossfitPlan`, `make_crossfit_splits`, `CrossfitNuisanceEstimator`, `NuisancePayload` |
| Score construction | `ScorePayload`, `build_score_payload` |
| Eq. (3.1) estimation | `EstimationPayload`, `estimate_eq31_mainline`, `Eq31ProjectionRankError`, `Eq31SolverConvergenceError` |
| End-to-end point-estimation workflow | `HDDIDFit`, `fit_hddid` |
| Eq. (4.1)--Eq. (4.3) inference | `ParametricInferencePayload`, `NonparametricInferencePayload`, `estimate_parametric_inference`, `estimate_nonparametric_inference`, `diagnose_nonparametric_omega_f`, `solve_eq42_sparse_direction`, `solve_eq43_projection_matrix` |
| Result containers | `HDDIDResult`, `ConfidenceInterval`, `UniformBand`, `FoldDiagnostics`, `ResultDiagnostics` |
| Inference errors | `InferenceComputationError`, `InvalidInferenceInputError`, `MissingEvaluationGridError`, `SingularCovarianceError`, `SparseDirectionInfeasibleError`, `NonpositiveVarianceError` |

Plotting helpers are module-local and should be imported from `hddid.plotting`.
Section 6 loader utilities are source-checkout helpers and should be imported
from `hddid.section6_loader` only when reproducing the manuscript's local
minimum-wage data-preparation evidence.

The current Section 6 evidence is deliberately narrower than an empirical
replication claim.  The source-checkout replication outputs include three
reduced real-data workflows: an official-QE outcome path, a joined LAUS--QE
path, and a joined LAUS--CCDB path.  All three run the public estimator through
Eq. (4.2) parametric inference and stop the requested Eq. (4.3) nonparametric
path at named covariance boundaries.  The same evidence set includes a
three-workflow reduced parametric software-output plot and a 2,424-county
LAUS--ICPSR candidate input that satisfies the public data contract for the
candidate workflow.  These outputs do not provide the exact 38-characteristic
County Data Book reconstruction, the 703-covariate Section 6 join, a
substantive minimum-wage effect estimate, or a completed reproduction of the
published empirical figure.

## `fit_hddid(...)`

```python
fit_hddid(
    *,
    y0,
    y1,
    treat,
    x,
    z,
    z0,
    basis_family="polynomial",
    basis_degree=1,
    alpha=0.1,
    n_folds=2,
    random_state=0,
    trim_lower=0.01,
    trim_upper=0.99,
    nuisance_payload=None,
    oracle_lane=None,
    nuisance_estimator=None,
    penalty_lambda=0.0,
    max_iter=10000,
    tol=1e-10,
    n_jobs=1,
    solver="sklearn",
)
```

`fit_hddid(...)` validates raw inputs, constructs or accepts nuisance
predictions, builds the doubly robust score, and runs the Eq. (3.1)
second-stage fit.  It returns an `HDDIDFit` with `data`, `crossfit_plan`,
`nuisance_payload`, `score_payload`, `estimation_payload`, and `result`.

Supplying `nuisance_payload` is the preferred route when nuisance predictions
come from a documented external cross-fitting procedure.  Omitting it uses the
package's built-in cross-fit nuisance estimator.

`nuisance_estimator` accepts a user-supplied sklearn-compatible estimator (or a
dict with ``"propensity"`` and ``"outcome"`` keys) for the first-stage
propensity and outcome models.  `n_jobs` controls parallel fold processing.  When
`n_jobs > 1`, each fold receives a deterministic worker seed (`0x5EED + fold_id`)
to ensure reproducibility regardless of execution order.  If you pass a custom
`nuisance_estimator` that relies on global `np.random` state, set its
`random_state` explicitly for consistent results across serial and parallel modes.
`solver` selects the Lasso backend (``"sklearn"``, ``"builtin"``, or
``"native"``).  If scikit-learn is not installed, `solver="sklearn"` silently
falls back to the built-in coordinate-descent solver and emits a `UserWarning`.
To guarantee use of the sklearn Lasso backend, ensure scikit-learn is installed
in your environment.

## Inference Helpers

`estimate_parametric_inference(score_payload, estimation_payload, ...)` computes
Eq. (4.1)--Eq. (4.2) parametric targets for explicit `xi` directions.  It returns
a `ParametricInferencePayload` and, when supplied, an updated `HDDIDResult`.

`estimate_nonparametric_inference(score_payload, estimation_payload, ...)`
computes the Eq. (4.3) nonparametric projection, one-step `f(z0)` update,
pointwise intervals, and finite-grid uniform band.  It returns a
`NonparametricInferencePayload` and, when supplied, an updated `HDDIDResult`.

Both helpers check that the estimation payload is reconstructable from the
current score payload before inference begins.  They also reject non-finite
inputs, incompatible covariance objects, infeasible sparse directions, missing
evaluation grids, and nonpositive variance objects with typed errors.

## Plotting

```python
from hddid.plotting import build_nonparametric_effect_plot_data

plot_data = build_nonparametric_effect_plot_data(
    result,
    z0=fit.data.z0,
    title="Nonparametric effect",
)

plot_data.to_csv("effect.csv")
plot_data.to_svg("effect.svg")
```

The plotting path is dependency-free.  It validates grid order, interval
nesting, labels, accessible SVG metadata, and the zero-reference status used in
the generated figure.

## Input Contract

Raw arrays must be finite numeric arrays.  Treatment indicators must contain
both treated and control observations.  The current package supports a
one-dimensional `z` lane with at least one evaluation point `z0`.  Propensity
scores must lie strictly inside `(0, 1)`, trim bounds and solver controls must be
finite numeric values, and confidence levels must be strictly inside `(0, 1)`.

The payload classes repeat these checks so stale same-shape arrays cannot be
attached to fresh score, estimation, or inference objects.  Invalid states fail
early with typed exceptions rather than flowing into reported estimates.

## Source-Checkout Validation Helpers

`hddid.validation` exposes manuscript and development helpers used by the
repository's replication scripts.  They are not part of the installed package's
ordinary user workflow.  Some validation helpers depend on source-checkout-only
modules, repository-local notes, generated paper outputs, or current source-tree
state that are deliberately excluded from clean wheel and sdist artifacts; if
called from an installed wheel, they raise `SourceCheckoutRequiredError` with
instructions to rerun them from the repository checkout.

```python
from hddid.validation import (
    audit_section6_assets,
    audit_section6_provenance_gate,
    generate_paper_dgp1,
    generate_paper_dgp2,
    run_monte_carlo_smoke,
)
```

The manuscript replication scripts use `audit_section6_assets(...)`,
`audit_section6_provenance_gate(...)`, `generate_paper_dgp1(...)`,
`generate_paper_dgp2(...)`, and `run_monte_carlo_smoke(...)` for bounded
source-checkout validation artifacts.
