from __future__ import annotations

import numpy as np
import pytest

from hddid import fit_hddid
from hddid.inputs import validate_inputs
from hddid.nuisance import CrossfitNuisanceEstimator
from hddid.splitting import make_crossfit_splits


def _make_regression_inputs(
    *,
    n: int = 200,
    p: int = 5,
    rng_seed: int = 42,
) -> dict[str, object]:
    """Generate a deterministic toy data set for nuisance estimator tests."""
    rng = np.random.default_rng(rng_seed)
    x = rng.standard_normal((n, p))
    z = rng.uniform(0.0, 1.0, n)
    beta = rng.standard_normal(p)

    # Moderately separated propensity score
    logit = 0.5 * x[:, 0] + 0.3 * x[:, 1] + rng.normal(0.0, 0.5, size=n)
    treat_prob = 1.0 / (1.0 + np.exp(-logit))
    treat = (rng.uniform(size=n) < treat_prob).astype(float)

    y0 = x @ beta + rng.normal(0.0, 1.0, size=n)
    delta = 1.0 + 0.5 * x[:, 0] + rng.normal(0.0, 1.0, size=n)
    y1 = y0 + delta

    return {
        "y0": y0,
        "y1": y1,
        "treat": treat,
        "x": x,
        "z": z,
        "z0": np.array([0.25, 0.5, 0.75]),
        "basis_family": "polynomial",
        "basis_degree": 2,
        "alpha": 0.1,
        "n_folds": 2,
        "random_state": 7,
    }


def _fit_default_and_payload(
    inputs: dict[str, object],
) -> tuple[dict[str, object], object]:
    """Fit with default nuisance estimators and return inputs + HDDIDFit."""
    fit = fit_hddid(**inputs)
    return inputs, fit


def test_default_nuisance_estimator_matches_explicit_none() -> None:
    """Passing nuisance_estimator=None must be numerically identical to omitting it."""
    base_inputs = _make_regression_inputs()
    default_inputs = dict(base_inputs)
    explicit_inputs = {**base_inputs, "nuisance_estimator": None}

    default_fit = fit_hddid(**default_inputs)
    explicit_fit = fit_hddid(**explicit_inputs)

    np.testing.assert_allclose(
        default_fit.nuisance_payload.pi_hat,
        explicit_fit.nuisance_payload.pi_hat,
    )
    np.testing.assert_allclose(
        default_fit.nuisance_payload.phi0_hat,
        explicit_fit.nuisance_payload.phi0_hat,
    )
    np.testing.assert_allclose(
        default_fit.nuisance_payload.phi1_hat,
        explicit_fit.nuisance_payload.phi1_hat,
    )
    np.testing.assert_array_equal(
        default_fit.nuisance_payload.valid_mask,
        explicit_fit.nuisance_payload.valid_mask,
    )
    np.testing.assert_allclose(
        default_fit.result.parametric_estimates["beta_hat"],
        explicit_fit.result.parametric_estimates["beta_hat"],
    )
    np.testing.assert_allclose(
        default_fit.result.nonparametric_estimates["f_hat_at_z0"],
        explicit_fit.result.nonparametric_estimates["f_hat_at_z0"],
        rtol=1e-12,
    )


def test_sklearn_estimator_matches_builtin() -> None:
    """Unpenalized sklearn logistic + linear regression should approximate builtins."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression, LogisticRegression

    base_inputs = _make_regression_inputs()
    builtin_fit = fit_hddid(**base_inputs)

    custom_inputs = {
        **base_inputs,
        "nuisance_estimator": LogisticRegression(
            C=np.inf,
            max_iter=2000,
        ),
    }
    # Re-use the same outcome estimator type so that both propensity and outcome
    # use sklearn.  A single estimator must expose both predict_proba and predict,
    # so we wrap the outcome model in a tiny adapter.
    class _PropensityAndOutcomeAdapter:
        def __init__(self, propensity, outcome):
            self.propensity = propensity
            self.outcome = outcome

        def fit(self, X, y):
            raise RuntimeError("should be split by CrossfitNuisanceEstimator")

        def predict(self, X):
            return self.outcome.predict(X)

        def predict_proba(self, X):
            return self.propensity.predict_proba(X)

    custom_inputs["nuisance_estimator"] = {
        "propensity": LogisticRegression(
            C=np.inf,
            max_iter=2000,
        ),
        "outcome": LinearRegression(),
    }
    custom_fit = fit_hddid(**custom_inputs)

    # The unpenalized sklearn models should be close to the built-in IRLS/OLS.
    np.testing.assert_allclose(
        builtin_fit.nuisance_payload.pi_hat,
        custom_fit.nuisance_payload.pi_hat,
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        builtin_fit.nuisance_payload.phi0_hat,
        custom_fit.nuisance_payload.phi0_hat,
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        builtin_fit.nuisance_payload.phi1_hat,
        custom_fit.nuisance_payload.phi1_hat,
        rtol=1e-3,
        atol=1e-3,
    )


def test_dict_nuisance_estimator_accepted() -> None:
    """Separate propensity/outcome estimators via a dict must be accepted."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression

    inputs = _make_regression_inputs()
    inputs["nuisance_estimator"] = {
        "propensity": LogisticRegression(C=np.inf, max_iter=2000),
        "outcome": GradientBoostingRegressor(n_estimators=10, random_state=0),
    }
    fit = fit_hddid(**inputs)

    assert fit.nuisance_payload.pi_hat.shape == (len(inputs["y0"]),)
    assert fit.nuisance_payload.phi0_hat.shape == (len(inputs["y0"]),)
    assert fit.nuisance_payload.phi1_hat.shape == (len(inputs["y0"]),)
    assert fit.result.parametric_estimates["beta_hat"].shape == (inputs["x"].shape[1],)


def test_invalid_single_estimator_missing_methods() -> None:
    """A single nuisance_estimator without required methods must raise a clear error."""
    inputs = _make_regression_inputs()

    class MissingPredictProba:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    inputs["nuisance_estimator"] = MissingPredictProba()
    with pytest.raises(ValueError, match="predict_proba"):
        fit_hddid(**inputs)

    class MissingPredict:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])

    inputs["nuisance_estimator"] = MissingPredict()
    with pytest.raises(ValueError, match=r"predict\(X\)"):
        fit_hddid(**inputs)

    class MissingFit:
        def predict(self, X):
            return np.zeros(len(X))

        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])

    inputs["nuisance_estimator"] = MissingFit()
    with pytest.raises(ValueError, match=r"fit\(X, y\)"):
        fit_hddid(**inputs)


def test_invalid_dict_estimator_missing_keys() -> None:
    """A dict nuisance_estimator without propensity/outcome keys must raise."""
    inputs = _make_regression_inputs()
    inputs["nuisance_estimator"] = {"propensity": object()}
    with pytest.raises(ValueError, match="'propensity' and 'outcome'"):
        fit_hddid(**inputs)


def test_dict_outcome_estimator_missing_predict() -> None:
    """A dict outcome estimator without predict must raise a clear error."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    inputs = _make_regression_inputs()

    class PropensityOnly:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])

    class OutcomeNoPredict:
        def fit(self, X, y):
            return self

    inputs["nuisance_estimator"] = {
        "propensity": PropensityOnly(),
        "outcome": OutcomeNoPredict(),
    }
    with pytest.raises(ValueError, match=r"nuisance_estimator\['outcome'\].*predict\(X\)"):
        fit_hddid(**inputs)


def test_nuisance_estimator_not_mutated_by_cross_fitting() -> None:
    """Each fold must receive a fresh clone; the original estimator stays unfitted."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression, LogisticRegression

    propensity_est = LogisticRegression(C=np.inf, max_iter=2000)
    outcome_est = LinearRegression()

    inputs = _make_regression_inputs()
    inputs["nuisance_estimator"] = {
        "propensity": propensity_est,
        "outcome": outcome_est,
    }
    fit_hddid(**inputs)

    assert not hasattr(propensity_est, "coef_")
    assert not hasattr(outcome_est, "coef_")


def test_crossfit_nuisance_estimator_custom_directly() -> None:
    """CrossfitNuisanceEstimator accepts custom estimators and produces a payload."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression, LogisticRegression

    inputs = _make_regression_inputs()
    data = validate_inputs(**{k: inputs[k] for k in (
        "y0", "y1", "treat", "x", "z", "z0", "basis_family", "basis_degree", "alpha"
    )})
    splits = make_crossfit_splits(
        data.n_obs,
        n_folds=inputs["n_folds"],
        random_state=inputs["random_state"],
    )

    estimator = CrossfitNuisanceEstimator(
        nuisance_estimator={
            "propensity": LogisticRegression(
                C=np.inf,
                max_iter=2000,
            ),
            "outcome": LinearRegression(),
        },
    )
    payload = estimator.fit(data, splits)

    assert payload.pi_hat.shape == (data.n_obs,)
    assert payload.phi0_hat.shape == (data.n_obs,)
    assert payload.phi1_hat.shape == (data.n_obs,)
    assert payload.valid_mask.shape == (data.n_obs,)
    assert len(payload.fold_diagnostics) == inputs["n_folds"]


def test_nuisance_estimator_incompatible_with_payload() -> None:
    """Passing both nuisance_payload and nuisance_estimator must raise."""
    sklearn = pytest.importorskip("sklearn")
    from hddid.nuisance import NuisancePayload
    from hddid.results import FoldDiagnostics

    inputs = _make_regression_inputs()
    data = validate_inputs(**{k: inputs[k] for k in (
        "y0", "y1", "treat", "x", "z", "z0", "basis_family", "basis_degree", "alpha"
    )})
    payload = NuisancePayload.from_predictions(
        pi_hat=np.full(data.n_obs, 0.5),
        phi0_hat=np.zeros(data.n_obs),
        phi1_hat=np.zeros(data.n_obs),
        treat=data.treat,
        fold_ids=np.ones(data.n_obs, dtype=int),
        fold_diagnostics=[
            FoldDiagnostics(
                fold_id=1,
                n_holdout_raw=data.n_obs,
                n_trimmed_propensity=0,
                n_valid_holdout=data.n_obs,
                trim_lower=0.01,
                trim_upper=0.99,
            ),
        ],
        basis_family=data.basis_family,
        basis_degree=data.basis_degree,
        oracle_lane="r-parity-polynomial",
    )

    from sklearn.linear_model import LogisticRegression
    with pytest.raises(ValueError, match="nuisance_estimator cannot be used"):
        fit_hddid(
            **inputs,
            nuisance_payload=payload,
            nuisance_estimator=LogisticRegression(C=np.inf, max_iter=2000),
        )
