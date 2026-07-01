from __future__ import annotations

import numpy as np

from hddid import HDDIDFit, fit_hddid


def test_fit_hddid_runs_minimal_component_workflow_from_raw_arrays() -> None:
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
        + 0.1 * treat
        + rng.normal(scale=0.05, size=n_obs)
    )

    fit = fit_hddid(
        y0=y0,
        y1=y1,
        treat=treat,
        x=x,
        z=z,
        z0=np.array([-0.5, 0.0, 0.5]),
        basis_family="polynomial",
        basis_degree=1,
        n_folds=3,
        random_state=1,
        trim_lower=0.0,
        trim_upper=1.0,
        penalty_lambda=0.01,
    )

    assert isinstance(fit, HDDIDFit)
    assert fit.crossfit_plan is not None
    assert fit.data.n_obs == n_obs
    assert int(fit.score_payload.s_hat_valid.shape[0]) == n_obs
    assert fit.estimation_payload.beta_hat.shape == (2,)
    assert fit.estimation_payload.f_hat_at_z0.shape == (3,)
    summary = fit.to_summary()
    assert summary["result_contract"] == "hddid-result-summary"
    assert summary["row_count"] == 7
    markdown = fit.to_markdown(missing_value="-", style="legacy")
    assert "| Parametric | beta_hat |" in markdown
    assert "| Nonparametric | f_hat_at_z0 |" in markdown
