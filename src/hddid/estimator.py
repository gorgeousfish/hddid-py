"""Top-level estimator orchestrating the HD-DID pipeline.

This module exposes :func:`fit_hddid`, the primary entry point that
sequences input validation, cross-fit splitting, nuisance estimation,
doubly-robust score construction, and the Eq. (3.1) second-stage
sieve regression into a single callable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from .estimation import EstimationPayload, estimate_eq31_mainline
from .inputs import ValidatedHDDIDData, validate_inputs
from .nuisance import CrossfitNuisanceEstimator, NuisancePayload
from .results import HDDIDResult
from .score import ScorePayload, build_score_payload
from .splitting import CrossfitPlan, make_crossfit_splits


@dataclass(frozen=True, slots=True)
class HDDIDFit:
    """Container for a complete HDDID estimation run.

    Attributes
    ----------
    data : ValidatedHDDIDData
        Validated inputs and basis configuration.
    crossfit_plan : CrossfitPlan or None
        Cross-fitting fold assignments (None if nuisance provided externally).
    nuisance_payload : NuisancePayload
        Cross-fitted propensity scores and outcome predictions.
    score_payload : ScorePayload
        Doubly-robust score vector and basis matrices.
    estimation_payload : EstimationPayload
        Eq. (3.1) regression outputs (beta_hat, gamma_hat, residuals).
    result : HDDIDResult
        User-facing estimates, standard errors, intervals, and diagnostics.
    """

    data: ValidatedHDDIDData
    crossfit_plan: CrossfitPlan | None
    nuisance_payload: NuisancePayload
    score_payload: ScorePayload
    estimation_payload: EstimationPayload
    result: HDDIDResult

    def to_markdown(self, **kwargs: Any) -> str:
        return self.result.to_markdown(**kwargs)

    def to_summary(self, **kwargs: Any) -> dict[str, object]:
        return self.result.to_summary(**kwargs)

    @classmethod
    def from_dataframe(
        cls,
        df,
        *,
        y0_col: str,
        y1_col: str,
        treat_col: str,
        x_cols: list[str],
        z_col: str,
        z0,
        **kwargs,
    ) -> "HDDIDFit":
        """Construct an HDDIDFit from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing all required columns.
        y0_col : str
            Column name for pre-treatment outcome.
        y1_col : str
            Column name for post-treatment outcome.
        treat_col : str
            Column name for binary treatment indicator.
        x_cols : list[str]
            Column names for high-dimensional covariates.
        z_col : str
            Column name for the scalar nonparametric variable.
        z0 : float or array-like
            Evaluation grid points for the nonparametric function.
        **kwargs
            Additional arguments passed to fit_hddid (basis_family, basis_degree,
            alpha, n_folds, random_state, trim_lower, trim_upper, etc.)

        Returns
        -------
        HDDIDFit
            Fitted HDDID model.

        Raises
        ------
        ImportError
            If pandas is not installed.
        KeyError
            If a specified column name does not exist in the DataFrame.
        ValueError
            If data validation fails (e.g., non-numeric columns, NaN values).
        """
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            raise ImportError(
                "from_dataframe requires pandas. Install with: pip install pandas"
            ) from None

        # Validate all required columns exist
        all_cols = [y0_col, y1_col, treat_col, z_col] + list(x_cols)
        for col in all_cols:
            if col not in df.columns:
                raise KeyError(
                    f"Column '{col}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )

        # Extract arrays
        y0 = df[y0_col].to_numpy(dtype=float)
        y1 = df[y1_col].to_numpy(dtype=float)
        treat = df[treat_col].to_numpy(dtype=float)
        x = df[x_cols].to_numpy(dtype=float)
        z = df[z_col].to_numpy(dtype=float)

        return fit_hddid(y0=y0, y1=y1, treat=treat, x=x, z=z, z0=z0, **kwargs)

    def summary(self, format: str = "text") -> "str | dict":
        """Generate a formatted summary of the estimation results.

        Parameters
        ----------
        format : str, default "text"
            Output format. One of:
            - "text": Human-readable plain text summary
            - "markdown": Markdown-formatted table (delegates to to_markdown)
            - "dict": Structured dictionary (delegates to to_summary)

        Returns
        -------
        str or dict
            Formatted estimation summary including parameter estimates,
            confidence intervals, standard errors, and diagnostics.
        """
        if format == "markdown":
            return self.to_markdown()
        if format == "dict":
            return self.to_summary()
        if format != "text":
            raise ValueError(
                f"Invalid format '{format}'. Valid options: 'text', 'markdown', 'dict'"
            )

        summary_data = self.to_summary(digits=4)
        diag = summary_data.get("diagnostics")
        rows = summary_data.get("rows", [])

        lines: list[str] = []
        lines.append("--- HDDID Estimation Results ---")

        # Diagnostics header
        if diag:
            basis_family = diag.get("basis_family", "unknown")
            basis_degree = diag.get("basis_degree", "?")
            lines.append(f"Basis: {basis_family} (degree={basis_degree})")
            fold_count = diag.get("fold_count", "?")
            n_valid = diag.get("n_valid_holdout", "?")
            alpha_val = self.data.alpha if hasattr(self.data, "alpha") else "?"
            lines.append(
                f"Folds: {fold_count} | Valid obs: {n_valid} | Alpha: {alpha_val}"
            )
        lines.append("")

        # Resolve z0 grid for f_hat_at_z0 labels
        z0_grid = None
        if hasattr(self.data, "z0"):
            z0_arr = np.asarray(self.data.z0).ravel()
            z0_grid = z0_arr

        # Group rows by section
        current_section = None
        for row in rows:
            section = row["section"]
            if section != current_section:
                current_section = section
                lines.append(f"{section} Estimates:")
                lines.append(
                    f"  {'Name':<16}{'Estimate':<12}{'SE':<12}{'95% CI'}"
                )

            name = row.get("name", "")
            idx = row.get("index", 0)
            estimate = row.get("estimate", "")
            se = row.get("standard_error", "")
            interval = row.get("interval", "")

            if name == "f_hat_at_z0" and z0_grid is not None and idx < len(z0_grid):
                label = f"f(z0={z0_grid[idx]:.2f})"
            else:
                label = f"{name}[{idx + 1}]"

            se_str = se if se else "-"
            ci_str = interval if interval else "-"
            lines.append(
                f"  {label:<16}{estimate:<12}{se_str:<12}{ci_str}"
            )

        lines.append("")
        return "\n".join(lines)


def fit_hddid(
    *,
    y0: Any,
    y1: Any,
    treat: Any,
    x: Any,
    z: Any,
    z0: Any,
    basis_family: str = "polynomial",
    basis_degree: int = 1,
    alpha: float = 0.1,
    n_folds: int = 2,
    random_state: int | None = 0,
    trim_lower: float = 0.01,
    trim_upper: float = 0.99,
    nuisance_payload: NuisancePayload | None = None,
    oracle_lane: str | None = None,
    nuisance_estimator: Any | None = None,
    penalty_lambda: Union[float, str] = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-10,
    n_jobs: int = 1,
    solver: str = "sklearn",
) -> HDDIDFit:
    """Fit the High-Dimensional Difference-in-Differences model.

    Executes the full HD-DID pipeline: input validation, cross-fit sample
    splitting, nuisance parameter estimation (propensity score, conditional
    means), doubly-robust score construction, and the partially-linear
    sieve regression of Eq. (3.1).

    Parameters
    ----------
    y0 : array-like of shape (n,)
        Pre-treatment outcome vector.
    y1 : array-like of shape (n,)
        Post-treatment outcome vector.
    treat : array-like of shape (n,)
        Binary treatment indicator (1 = treated, 0 = control).
    x : array-like of shape (n, p)
        High-dimensional covariate matrix.
    z : array-like of shape (n,)
        Scalar nonparametric variable entering the sieve component.
    z0 : float or array-like
        Evaluation grid point(s) at which the nonparametric function
        f(z0) is estimated.
    basis_family : str, default "polynomial"
        Sieve basis family. One of "polynomial", "trigonometric", or
        "bspline".
    basis_degree : int, default 1
        Degree (order) of the sieve basis expansion.
    alpha : float, default 0.1
        Significance level for confidence intervals.
    n_folds : int, default 2
        Number of cross-fitting folds.
    random_state : int or None, default 0
        Seed for the random number generator controlling fold assignment.
        Use None for non-deterministic splits.
    trim_lower : float, default 0.01
        Lower propensity score trimming threshold.
    trim_upper : float, default 0.99
        Upper propensity score trimming threshold.
    nuisance_payload : NuisancePayload or None, default None
        Pre-computed nuisance estimates. When provided, cross-fitting is
        skipped and this payload is used directly.
    oracle_lane : str or None, default None
        Oracle nuisance lane identifier for simulation studies.
        Only valid when nuisance_payload is None.
    nuisance_estimator : estimator, dict, or None, default None
        User-supplied sklearn-compatible first-stage estimator(s). If a
        single estimator is passed, it is used for both the propensity score
        and the conditional outcome means. If a dict is passed, it must
        contain the keys ``"propensity"`` and ``"outcome"``. The propensity
        estimator must implement ``.fit(X, y)`` and ``.predict_proba(X)``;
        the outcome estimator must implement ``.fit(X, y)`` and
        ``.predict(X)``. When None, the built-in IRLS logistic regression
        and OLS linear regression are used, exactly as before.
    penalty_lambda : float or "auto", default 0.0
        L1 penalty on the parametric coefficients in Eq. (3.1).
        Set to "auto" for the rate-optimal choice
        2.2 * sqrt(log(p) / n_valid).
    max_iter : int, default 10000
        Maximum coordinate-descent iterations for the Lasso solver.
    tol : float, default 1e-10
        Convergence tolerance for coordinate descent.
    n_jobs : int, default 1
        Number of parallel jobs for cross-fit fold processing.
    solver : str, default "sklearn"
        Solver backend for the beta block. One of "sklearn"
        (scikit-learn Lasso), "builtin" (pure-NumPy coordinate descent),
        or "native" (alias for "builtin").

    Returns
    -------
    HDDIDFit
        Frozen dataclass containing all intermediate payloads and the
        final HDDIDResult with point estimates for beta_hat, gamma_hat,
        and f_hat_at_z0.

    Raises
    ------
    ValueError
        If inputs fail validation (non-numeric, mismatched shapes,
        invalid basis_family, trim bounds, etc.).
    Eq31SolverConvergenceError
        If the coordinate-descent Lasso fails to converge within
        max_iter iterations.
    Eq31ProjectionRankError
        If the sieve basis matrix is rank-deficient.

    Notes
    -----
    Implements the two-stage estimation procedure from Section 3 of
    Ning, Peng, and Tao (2020). The first stage constructs the
    doubly-robust score S_hat (Eq. 2.5/2.7); the second stage
    solves the partially-linear model S_hat = X'beta + f(Z) + error
    via sieve projection and penalized regression (Eq. 3.1).

    Reference: Ning, Peng, and Tao (2020), arXiv preprint arXiv:2009.03151.

    Examples
    --------
    >>> import numpy as np
    >>> from hddid import fit_hddid
    >>> rng = np.random.default_rng(42)
    >>> n, p = 200, 5
    >>> x = rng.standard_normal((n, p))
    >>> z = rng.uniform(0, 1, n)
    >>> treat = (rng.uniform(size=n) > 0.5).astype(float)
    >>> y0 = x @ rng.standard_normal(p) + rng.standard_normal(n)
    >>> y1 = y0 + 1.0 + rng.standard_normal(n)
    >>> result = fit_hddid(
    ...     y0=y0, y1=y1, treat=treat, x=x, z=z,
    ...     z0=np.array([0.25, 0.5, 0.75]),
    ... )
    """
    data = validate_inputs(
        y0=y0,
        y1=y1,
        treat=treat,
        x=x,
        z=z,
        z0=z0,
        basis_family=basis_family,
        basis_degree=basis_degree,
        alpha=alpha,
    )

    if nuisance_payload is not None and nuisance_estimator is not None:
        raise ValueError(
            "nuisance_estimator cannot be used together with nuisance_payload"
        )

    crossfit_plan: CrossfitPlan | None = None
    if nuisance_payload is None:
        crossfit_plan = make_crossfit_splits(
            data.n_obs,
            n_folds,
            random_state,
            trim_lower=trim_lower,
            trim_upper=trim_upper,
        )
        nuisance_payload = CrossfitNuisanceEstimator(
            oracle_lane=oracle_lane,
            nuisance_estimator=nuisance_estimator,
        ).fit(
            data,
            crossfit_plan,
            n_jobs=n_jobs,
        )
    elif oracle_lane is not None:
        raise ValueError("oracle_lane is only used when nuisance_payload is omitted")

    score_payload = build_score_payload(data, nuisance_payload)

    # Resolve penalty_lambda
    if isinstance(penalty_lambda, str):
        if penalty_lambda == "auto":
            n_valid = int(score_payload.s_hat_valid.shape[0])
            p_dim = int(score_payload.x_valid.shape[1])
            computed_lambda = 2.2 * np.sqrt(
                np.log(max(p_dim, 2)) / max(n_valid, 1)
            )
        else:
            raise ValueError(
                f"penalty_lambda must be a float or 'auto', got '{penalty_lambda}'"
            )
    else:
        computed_lambda = float(penalty_lambda)

    estimation_payload, result = estimate_eq31_mainline(
        score_payload,
        penalty_lambda=computed_lambda,
        max_iter=max_iter,
        tol=tol,
        solver=solver,
    )
    return HDDIDFit(
        data=data,
        crossfit_plan=crossfit_plan,
        nuisance_payload=nuisance_payload,
        score_payload=score_payload,
        estimation_payload=estimation_payload,
        result=result,
    )
