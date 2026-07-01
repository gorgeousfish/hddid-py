# hddid

**High-Dimensional Doubly Robust Difference-in-Differences**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-green.svg)](https://github.com/gorgeousfish/hddid-py)

## Overview

`hddid` implements the high-dimensional doubly robust difference-in-differences
estimator of Ning, Peng, and Tao (2020). One call to `fit_hddid(...)` runs
cross-fitted nuisance estimation, constructs the doubly robust score, solves the
partially-linear sieve regression, and returns an `HDDIDFit` object containing
all intermediate payloads and the user-facing result. Parametric debiased
inference (Eq. 4.1–4.2) and nonparametric uniform bands (Eq. 4.3) attach to the
same object as explicit follow-up calls.

## Requirements

| Requirement | Description |
| :--- | :--- |
| **Python 3.10+** | NumPy, SciPy, and PyYAML are required at runtime. |
| **NumPy ≥ 1.24** | Core array computation. |
| **SciPy ≥ 1.10** | Optimization and linear algebra backends. |
| **PyYAML ≥ 6.0** | Configuration parsing. |
| **scikit-learn** | Optional; enables the `solver="sklearn"` Lasso backend. |
| **pandas** | Optional; enables `HDDIDFit.from_dataframe(...)`. |
| **Data format** | NumPy arrays: outcomes `y0`/`y1` (shape *n*), treatment `treat` (0/1), covariates `x` (shape *n* × *p*), scalar variable `z` (shape *n*). |

## Installation

### From PyPI

```bash
pip install hddid
```

### Development install

```bash
git clone https://github.com/gorgeousfish/hddid-py.git
cd hddid-py
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from hddid import fit_hddid

rng = np.random.default_rng(42)
n_obs = 30
x = rng.normal(size=(n_obs, 2))
z = rng.normal(size=n_obs)
treat = np.array([0, 1] * 15, dtype=int)
rng.shuffle(treat)
y0 = rng.normal(scale=0.1, size=n_obs)
y1 = (
    y0
    + 0.2
    + 0.4 * x[:, 0]
    - 0.2 * x[:, 1]
    + 0.3 * z
    + rng.normal(scale=0.05, size=n_obs)
)

fit = fit_hddid(
    y0=y0,
    y1=y1,
    treat=treat,
    x=x,
    z=z,
    z0=np.array([-0.5, 0.0, 0.5]),
    basis_family='polynomial',
    basis_degree=1,
    n_folds=3,
    random_state=1,
    trim_lower=0.0,
    trim_upper=1.0,
    penalty_lambda=0.01,
)

print(fit.to_markdown())
```

**Output:**

```
| Section | Name | Index | Estimate | Std. Error | Interval |
|---|---|---:|---:|---:|---|
| **Parametric** | β̂ | 0 | 0.0347 | — | — |
|  | β̂ | 1 | 0.0341 | — | — |
| **Nonparametric** | γ̂ | 0 | 0.0018 | — | — |
|  | γ̂ | 1 | 0.0064 | — | — |
|  | f̂(z₀) | 0 | -0.0014 | — | — |
|  | f̂(z₀) | 1 | 0.0018 | — | — |
|  | f̂(z₀) | 2 | 0.0050 | — | — |

Diagnostics
  Basis: polynomial(1)
  Oracle lane: r-parity-polynomial
  Sample: holdout=30, trimmed=0, valid=30
  Trim: [0.0000, 1.0000]
  Folds: 3
```

## API Reference

### fit_hddid

```python
from hddid import fit_hddid

fit = fit_hddid(
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
    penalty_lambda=0.0,
    max_iter=10_000,
    tol=1e-10,
    n_jobs=1,
    solver="sklearn",
)
```

#### Required Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `y0` | array-like (*n*,) | Pre-treatment outcome vector. |
| `y1` | array-like (*n*,) | Post-treatment outcome vector. |
| `treat` | array-like (*n*,) | Binary treatment indicator (1 = treated, 0 = control). |
| `x` | array-like (*n*, *p*) | High-dimensional covariate matrix. |
| `z` | array-like (*n*,) | Scalar nonparametric variable entering the sieve component. |
| `z0` | float or array-like | Evaluation grid point(s) at which *f*(*z*₀) is estimated. |

#### Optional Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `basis_family` | `"polynomial"` | Sieve basis family. One of `"polynomial"`, `"trigonometric"`, or `"bspline"`. |
| `basis_degree` | `1` | Degree (order) of the sieve basis expansion. |
| `alpha` | `0.1` | Significance level for confidence intervals. |
| `n_folds` | `2` | Number of cross-fitting folds. |
| `random_state` | `0` | Seed for fold assignment. Use `None` for non-deterministic splits. |
| `trim_lower` | `0.01` | Lower propensity score trimming threshold. |
| `trim_upper` | `0.99` | Upper propensity score trimming threshold. |
| `nuisance_payload` | `None` | Pre-computed nuisance estimates; skips cross-fitting when provided. |
| `oracle_lane` | `None` | Computational lane identifier for R-parity verification. |
| `nuisance_estimator` | `None` | Custom nuisance estimator implementing `.fit()` / `.predict()`. |
| `penalty_lambda` | `0.0` | L1 penalty on parametric coefficients. Set to `"auto"` for rate-optimal choice. |
| `max_iter` | `10_000` | Maximum coordinate-descent iterations for the Lasso solver. |
| `tol` | `1e-10` | Convergence tolerance for coordinate descent. |
| `n_jobs` | `1` | Number of parallel jobs for cross-fit fold processing. |
| `solver` | `"sklearn"` | Solver backend: `"sklearn"` (scikit-learn Lasso) or `"builtin"` (pure-NumPy). |

### Return Object: HDDIDFit

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `data` | `ValidatedHDDIDData` | Validated inputs and basis configuration. |
| `crossfit_plan` | `CrossfitPlan | None` | Fold assignments (`None` if nuisance provided externally). |
| `nuisance_payload` | `NuisancePayload` | Cross-fitted propensity scores and outcome predictions. |
| `score_payload` | `ScorePayload` | Doubly robust score (Eq. 2.5) and valid-sample masks. |
| `estimation_payload` | `EstimationPayload` | Eq. (3.1) beta, sieve coefficients, residuals, and projection objects. |
| `result` | `HDDIDResult` | User-facing estimates, standard errors, intervals, and diagnostics. |

`HDDIDResult` exposes `to_markdown(...)` for readable table output and
`to_summary(...)` for structured data extraction.

## Inference

Parametric and nonparametric inference are explicit follow-up calls that attach
Eq. (4.1)–(4.2) and Eq. (4.3) objects to an existing fit:

```python
from hddid import (
    estimate_parametric_inference,
    estimate_nonparametric_inference,
)

parametric, result = estimate_parametric_inference(
    fit.score_payload,
    fit.estimation_payload,
    result=fit.result,
    xi=np.eye(fit.estimation_payload.beta_hat.size),
    alpha=0.1,
    lambda_prime=0.0,
)

nonparametric, result = estimate_nonparametric_inference(
    fit.score_payload,
    fit.estimation_payload,
    result=result,
    alpha=0.1,
    lambda_double_prime=0.0,
    n_boot=256,
    random_state=123,
)

print(result.to_markdown())
```

**Output:**

```
| Section | Name | Index | Estimate | Std. Error | Interval |
|---|---|---:|---:|---:|---|
| **Parametric** | β̂ | 0 | 0.0347 | — | — |
|  | β̂ | 1 | 0.0341 | — | — |
|  | τ̂ | 0 | 0.0412 | 0.0336 | [-0.0141, 0.0965] (90.0%) |
|  | τ̂ | 1 | 0.0422 | 0.0335 | [-0.0130, 0.0973] (90.0%) |
| **Nonparametric** | γ̂ | 0 | 0.0018 | — | — |
|  | γ̂ | 1 | 0.0064 | — | — |
|  | γ̄ | 0 | 0.0029 | — | — |
|  | γ̄ | 1 | 0.0088 | — | — |
|  | f̂(z₀) | 0 | -0.0014 | — | — |
|  | f̂(z₀) | 1 | 0.0018 | — | — |
|  | f̂(z₀) | 2 | 0.0050 | — | — |
|  | f̄(z₀) | 0 | -0.0015 | 0.0341 | [-0.0577, 0.0546] (90.0%) |
|  | f̄(z₀) | 1 | 0.0029 | 0.0298 | [-0.0461, 0.0519] (90.0%) |
|  | f̄(z₀) | 2 | 0.0072 | 0.0426 | [-0.0628, 0.0773] (90.0%) |

Diagnostics
  Basis: polynomial(1)
  Oracle lane: r-parity-polynomial
  Sample: holdout=30, trimmed=0, valid=30
  Trim: [0.0000, 1.0000]
  Folds: 3
```

Infeasible directions, missing evaluation grids, and nonpositive variance
objects raise typed errors instead of silently producing generic standard-error
rows.

## Key Formulas

### Doubly Robust Score (Eq. 2.5)

The cross-fitted doubly robust score assembles the treatment-weighted residual
from cross-fitted nuisance estimates:

> *Ŝ* = *ρ̂* × (Δ*Y* − (1 − *π̂*) *φ̂*₁ − *π̂* *φ̂*₀)

where *π̂* = *P̂*(*D* = 1 | *X*) is the propensity score, *φ̂*₀ and *φ̂*₁ are
conditional mean estimates, and the propensity weight is:

> *ρ̂* = (*D* − *π̂*) / [*π̂*(1 − *π̂*)]

### Second-Stage Estimator (Eq. 3.1)

The score is decomposed into a high-dimensional linear part and a nonparametric
sieve component:

> *Ŝ* = *X*'*β̂* + *f̂*(*Z*) + *ε̂*

where *f̂*(*Z*) = *ψ*(*Z*)'*γ̂* is a sieve approximation using a polynomial,
trigonometric, or B-spline basis. The parametric coefficients *β̂* are obtained
via penalized (Lasso) regression on the Frisch–Waugh residualized design.

### Parametric Inference (Eq. 4.2)

Debiased inference on linear functionals *ξ*' *β* uses a sparse direction *ŵ*
solved from:

> *ŵ* = arg min *w*' Σ̃<sub>X</sub> *w* − 2 *w*' *σ̂*<sub>X,ψ</sub> + *λ*' ‖*w*‖₁

The debiased estimator and its confidence interval are:

> *t̂* = *ξ*' *β̂* − *ŵ*' *Ŝ*<sub>moment</sub>

> *CI* = *t̂* ± *z*<sub>1−α/2</sub> × *√*(*ŵ*' Ω̂<sub>β</sub> *ŵ*)

### Nonparametric Inference (Eq. 4.3)

Debiased inference on *f*(*z*) orthogonalizes the sieve basis against the
high-dimensional covariates via a projection matrix *M̂*:

> *ψ̃* = *ψ* − *M̂* *X*

The debiased sieve estimator and its uniform band are:

> *f̄*(*z*₀) = *f̂*(*z*₀) − (*ψ̃*₀)' Σ̂<sub>f</sub><sup>−1</sup> *M̂* *Ŝ*<sub>moment</sub>

> *CI*<sub>uniform</sub> = *f̄*(*z*₀) ± *ĉ* × *√*(diag(*V̂*<sub>f</sub>))

where *ĉ* is a bootstrap critical value and *V̂*<sub>f</sub> is the sandwich
variance Σ̂<sub>f</sub><sup>−1</sup> Ω̂<sub>f</sub> Σ̂<sub>f</sub><sup>−1</sup>.

## Plot Output

`hddid.plotting.build_nonparametric_effect_plot_data(...)` extracts a
`NonparametricEffectPlotData` object from an inference result, containing the
evaluation grid, point estimates, pointwise confidence intervals, and the
uniform band. The object provides `.to_svg(...)` for dependency-free SVG output
and `.to_csv(...)` for tabular export.

```python
from hddid.plotting import build_nonparametric_effect_plot_data

plot_data = build_nonparametric_effect_plot_data(
    result=result,
    z0=fit.data.z0,
)

svg_string = plot_data.to_svg()
```

## References

Ning, Y., Peng, S., & Tao, J. (2020). Doubly robust semiparametric
difference-in-differences estimators with high-dimensional data. *arXiv
preprint* arXiv:2009.03151. https://arxiv.org/abs/2009.03151

## Authors

**Python Implementation:**

- **Xuanyu Cai**, City University of Macau
- **Wenli Xu**, City University of Macau

**Methodology:**

- **Yang Ning**, Cornell University
- **Shuang Peng**, Cornell University
- **Jin Tao**, Cornell University

## License

AGPL-3.0. See [LICENSE](LICENSE) for details.

## Citation

If you use this package in your research, please cite both the software and the
methodology paper:

**APA Format:**

> Cai, X., & Xu, W. (2026). *hddid: High-dimensional doubly robust difference-in-differences* (Version 0.1.0) [Computer software]. GitHub. https://github.com/gorgeousfish/hddid-py
>
> Ning, Y., Peng, S., & Tao, J. (2020). Doubly robust semiparametric difference-in-differences estimators with high-dimensional data. *arXiv preprint* arXiv:2009.03151. https://arxiv.org/abs/2009.03151

**BibTeX:**

```bibtex
@software{hddid2026python,
      title={hddid: High-dimensional doubly robust difference-in-differences},
      author={Xuanyu Cai and Wenli Xu},
      year={2026},
      version={0.1.0},
      url={https://github.com/gorgeousfish/hddid-py}
}

@misc{ning2020doublyrobust,
      title={Doubly Robust Semiparametric Difference-in-Differences Estimators
             with High Dimensional Data},
      author={Yang Ning and Shuang Peng and Jin Tao},
      year={2020},
      eprint={2009.03151},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2009.03151}
}
```
