from __future__ import annotations

import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .inputs import ValidatedHDDIDData
from .results import FoldDiagnostics, normalize_oracle_lane
from .splitting import CrossfitFold, CrossfitPlan

try:
    import joblib

    _HAS_JOBLIB = True
except Exception:  # pragma: no cover
    _HAS_JOBLIB = False


def _contains_boolean_or_string_alias(value: Any) -> bool:
    """Check whether ``value`` contains booleans or strings (recursively)."""
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_ or value.dtype.kind in {"S", "U"}:
            return True
        if value.dtype == object:
            return any(
                _contains_boolean_or_string_alias(item) for item in value.flat
            )
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean_or_string_alias(item) for item in value)
    return False


def _coerce_numeric_vector(name: str, values: Any) -> np.ndarray:
    """Coerce input to a 1-D float array, rejecting booleans and strings."""
    if _contains_boolean_or_string_alias(values):
        raise ValueError(f"{name} must be numeric, not boolean or string")
    raw_array = np.asarray(values)
    if _contains_boolean_or_string_alias(raw_array):
        raise ValueError(f"{name} must be numeric, not boolean or string")
    array = raw_array.astype(float)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    raise ValueError(f"{name} must be one-dimensional")


def _coerce_fold_ids(values: Any) -> np.ndarray:
    raw_array = np.asarray(values)
    if raw_array.dtype == np.bool_:
        raise ValueError("fold_ids must contain positive integer values")
    if np.issubdtype(raw_array.dtype, np.integer):
        array = raw_array.astype(int, copy=False)
    elif raw_array.dtype == object:
        flat_values = raw_array.reshape(-1)
        if any(isinstance(value, bool) for value in flat_values):
            raise ValueError("fold_ids must contain positive integer values")
        if not all(isinstance(value, Integral) for value in flat_values):
            raise ValueError("fold_ids must contain positive integer values")
        array = raw_array.astype(int, copy=False)
    else:
        raise ValueError("fold_ids must contain positive integer values")
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 1:
        result = array
    elif array.ndim == 2 and array.shape[1] == 1:
        result = array[:, 0]
    else:
        raise ValueError("fold_ids must be one-dimensional")
    if np.any(result <= 0):
        raise ValueError("fold_ids must contain positive integer values")
    return result


def _coerce_boolean_mask(name: str, values: Any) -> np.ndarray:
    """Coerce input to a 1-D boolean array."""
    raw_array = np.asarray(values)
    if raw_array.dtype != np.bool_:
        raise ValueError(f"{name} must contain only boolean values")
    array = raw_array.astype(bool, copy=False)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    raise ValueError(f"{name} must be one-dimensional")


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")


def _coerce_basis_degree(basis_family: str, basis_degree: int) -> int:
    """Validate basis degree given the basis family."""
    if isinstance(basis_degree, bool) or not isinstance(basis_degree, Integral):
        raise ValueError("basis_degree must be an integer")
    degree = int(basis_degree)
    if degree < 0:
        raise ValueError("basis_degree must be non-negative")
    if basis_family == "trigonometric" and degree < 1:
        raise ValueError("basis_degree must be positive for trigonometric basis")
    return degree


def _design_with_intercept(features: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to the feature matrix."""
    if features.ndim != 2:
        raise ValueError("features must be a matrix")
    intercept = np.ones((features.shape[0], 1), dtype=float)
    return np.hstack([intercept, features])


def _safe_sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid with numerical clipping."""
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _has_method(obj: Any, name: str) -> bool:
    """Check whether ``obj`` has a callable attribute named ``name``."""
    attr = getattr(obj, name, None)
    return callable(attr)


def _validate_nuisance_estimator(
    name: str,
    estimator: Any,
    *,
    needs_predict: bool = True,
    needs_proba: bool = False,
) -> None:
    """Validate a user-supplied sklearn-compatible estimator."""
    if estimator is None:
        raise ValueError(f"{name} estimator must not be None")
    if not _has_method(estimator, "fit"):
        raise ValueError(f"{name} estimator must provide a .fit(X, y) method")
    if needs_predict and not _has_method(estimator, "predict"):
        raise ValueError(f"{name} estimator must provide a .predict(X) method")
    if needs_proba and not _has_method(estimator, "predict_proba"):
        raise ValueError(
            f"{name} estimator must provide a .predict_proba(X) method "
            "because it is used to estimate propensity scores"
        )


def _resolve_nuisance_estimators(
    nuisance_estimator: Any,
) -> tuple[Any, Any] | tuple[None, None]:
    """Resolve a single estimator or dict into (propensity, outcome) pair.

    Returns (None, None) when ``nuisance_estimator`` is None, preserving the
    original builtin logistic/OLS code path unchanged.
    """
    if nuisance_estimator is None:
        return None, None

    if isinstance(nuisance_estimator, dict):
        if "propensity" not in nuisance_estimator or "outcome" not in nuisance_estimator:
            raise ValueError(
                "nuisance_estimator dict must contain the keys "
                "'propensity' and 'outcome'"
            )
        propensity_estimator = nuisance_estimator["propensity"]
        outcome_estimator = nuisance_estimator["outcome"]
        _validate_nuisance_estimator(
            "nuisance_estimator['propensity']",
            propensity_estimator,
            needs_predict=False,
            needs_proba=True,
        )
        _validate_nuisance_estimator(
            "nuisance_estimator['outcome']",
            outcome_estimator,
            needs_predict=True,
            needs_proba=False,
        )
        return propensity_estimator, outcome_estimator

    _validate_nuisance_estimator(
        "nuisance_estimator",
        nuisance_estimator,
        needs_predict=True,
        needs_proba=True,
    )
    return nuisance_estimator, nuisance_estimator


def _fit_logistic_irls(
    train_features: np.ndarray,
    target: np.ndarray,
    prediction_features: np.ndarray,
    *,
    ridge: float = 1e-8,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Fit a logistic regression via IRLS and predict propensity scores.

    Implements iteratively reweighted least squares (IRLS) for the
    propensity score model P(D=1|X), used in the first-stage
    nuisance estimation of the doubly robust DiD estimator.

    The propensity score pi(W_i) is a key component of the Eq. (5)
    doubly robust estimand, entering through the weight
    rho = (D - pi) / [pi(1 - pi)].

    Parameters
    ----------
    train_features : ndarray of shape (n_train, k)
        Training covariates (may include sieve basis columns).
    target : ndarray of shape (n_train,)
        Binary treatment indicator D_i (0 or 1).
    prediction_features : ndarray of shape (n_pred, k)
        Features at which to predict propensity scores.
    ridge : float, default 1e-8
        Ridge regularization for the Hessian.
    max_iter : int, default 100
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance on the coefficient update.

    Returns
    -------
    ndarray of shape (n_pred,)
        Predicted propensity scores, clipped to (tiny, 1 - epsilon).

    Notes
    -----
    References: Eq. (5) and Section 3 of Ning, Peng & Tao (2020),
    arXiv preprint arXiv:2009.03151.
    """
    if train_features.shape[0] != target.shape[0]:
        raise ValueError("train_features and target must align")

    unique_target = np.unique(target)
    if unique_target.size == 1:
        constant_probability = 1e-6 if unique_target[0] <= 0.0 else 1.0 - 1e-6
        return np.full(prediction_features.shape[0], constant_probability, dtype=float)

    design = _design_with_intercept(train_features)
    prediction_design = _design_with_intercept(prediction_features)
    coefficients = np.zeros(design.shape[1], dtype=float)

    for _ in range(max_iter):
        linear_index = design @ coefficients
        probability = np.clip(_safe_sigmoid(linear_index), 1e-6, 1.0 - 1e-6)
        weights = probability * (1.0 - probability)
        adjusted_response = linear_index + (target - probability) / weights

        weighted_design = design * weights[:, None]
        hessian = design.T @ weighted_design + ridge * np.eye(design.shape[1])
        score = design.T @ (weights * adjusted_response)
        updated = np.linalg.solve(hessian, score)
        if np.max(np.abs(updated - coefficients)) <= tol:
            coefficients = updated
            break
        coefficients = updated

    return np.clip(
        _safe_sigmoid(prediction_design @ coefficients),
        np.finfo(float).tiny,
        np.nextafter(1.0, 0.0),
    )


def _predict_linear_regression(
    train_features: np.ndarray,
    train_target: np.ndarray,
    holdout_features: np.ndarray,
) -> np.ndarray:
    """Fit OLS and predict conditional outcome means Phi_d(W_i).

    Estimates the conditional mean E[DeltaY | X, D=d] via ordinary
    least squares, used for the nuisance parameters Phi_0 and Phi_1
    in the Eq. (5) doubly robust score.

    Parameters
    ----------
    train_features : ndarray of shape (n_train, k)
        Training covariates for the treated (d=1) or control (d=0) group.
    train_target : ndarray of shape (n_train,)
        First-differenced outcomes DeltaY for the training group.
    holdout_features : ndarray of shape (n_holdout, k)
        Features at which to predict conditional means.

    Returns
    -------
    ndarray of shape (n_holdout,)
        Predicted conditional mean E[DeltaY | X, D=d].

    Notes
    -----
    References: Eq. (5) and Section 3 of Ning, Peng & Tao (2020),
    arXiv preprint arXiv:2009.03151.
    """
    if train_features.shape[0] == 0:
        raise ValueError("linear regression training sample must be nonempty")

    train_design = _design_with_intercept(train_features)
    holdout_design = _design_with_intercept(holdout_features)
    coefficients, *_ = np.linalg.lstsq(train_design, train_target, rcond=None)
    return holdout_design @ coefficients


def _feature_block(
    data: ValidatedHDDIDData,
    *,
    basis_matrix: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Build the combined feature block [X, sieve_basis] for a fold.

    Concatenates the high-dimensional covariates X with the sieve basis
    columns (excluding the intercept if present) for use in nuisance
    estimation.

    Parameters
    ----------
    data : ValidatedHDDIDData
        Validated input data containing covariates X.
    basis_matrix : ndarray of shape (n, L)
        Full-sample sieve basis matrix.
    indices : ndarray of int
        Observation indices to select.

    Returns
    -------
    ndarray of shape (len(indices), p + L')
        Combined feature block where L' = L - 1 if the basis has an
        intercept column (which is dropped), else L' = L.
    """
    basis_features = basis_matrix[indices]
    if basis_features.shape[1] > 0 and np.allclose(basis_features[:, 0], 1.0):
        basis_features = basis_features[:, 1:]
    if basis_features.shape[1] == 0:
        return np.asarray(data.x[indices], dtype=float)
    return np.hstack([np.asarray(data.x[indices], dtype=float), basis_features])


def _default_oracle_lane(basis_family: str) -> str:
    """Return the default oracle lane for the given basis family."""
    if basis_family == "trigonometric":
        return "paper-trigonometric"
    return "r-parity-polynomial"


class ZeroValidHoldoutError(RuntimeError):
    def __init__(self, *, fold_id: int, trim_lower: float, trim_upper: float) -> None:
        self.fold_id = int(fold_id)
        self.trim_lower = float(trim_lower)
        self.trim_upper = float(trim_upper)
        super().__init__(
            "Fold "
            f"{self.fold_id} has zero valid holdout observations after propensity "
            f"trimming at [{self.trim_lower}, {self.trim_upper}]. "
            "Consider increasing sample size, reducing n_folds, or relaxing trim bounds."
        )

    def __str__(self) -> str:
        return self.args[0]


class NuisanceTrainingSupportError(RuntimeError):
    def __init__(
        self,
        *,
        fold_id: int,
        n_train_treated: int,
        n_train_control: int,
    ) -> None:
        self.fold_id = int(fold_id)
        self.n_train_treated = int(n_train_treated)
        self.n_train_control = int(n_train_control)
        super().__init__(
            "Fold "
            f"{self.fold_id} training sample must contain treated and control "
            "observations for separate Phi1/Phi0 nuisance fits; found "
            f"{self.n_train_treated} treated and {self.n_train_control} control"
        )

    def __str__(self) -> str:
        return self.args[0]


@dataclass(slots=True)
class NuisancePayload:
    """Cross-fitted first-stage nuisance estimates for Eq. (2.5).

    Stores propensity scores, conditional outcome means, and the
    propensity weight rho required to construct the doubly-robust
    score in the second stage.

    Attributes
    ----------
    pi_hat : ndarray of float, shape (n,)
        Estimated propensity scores P(D=1|X), strictly in (0, 1).
    phi0_hat : ndarray of float, shape (n,)
        Estimated conditional mean E[DeltaY | X, D=0].
    phi1_hat : ndarray of float, shape (n,)
        Estimated conditional mean E[DeltaY | X, D=1].
    rho_hat : ndarray of float, shape (n,)
        Propensity weight (D - pi) / [pi(1 - pi)].
    fold_ids : ndarray of int, shape (n,)
        Cross-fitting fold assignment per observation.
    fold_diagnostics : list of FoldDiagnostics
        Per-fold trimming and sample size diagnostics.
    basis_family : str
        Sieve basis family used ("polynomial", "trigonometric", or "bspline").
    basis_degree : int
        Truncation parameter for the sieve basis.
    oracle_lane : str
        Computational lane identifier for R-parity verification.
    valid_mask : ndarray of bool or None
        Mask indicating observations retained after propensity trimming.
    """

    pi_hat: np.ndarray
    phi0_hat: np.ndarray
    phi1_hat: np.ndarray
    rho_hat: np.ndarray
    fold_ids: np.ndarray
    fold_diagnostics: list[FoldDiagnostics]
    basis_family: str
    basis_degree: int
    oracle_lane: str
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.pi_hat = _coerce_numeric_vector("pi_hat", self.pi_hat)
        self.phi0_hat = _coerce_numeric_vector("phi0_hat", self.phi0_hat)
        self.phi1_hat = _coerce_numeric_vector("phi1_hat", self.phi1_hat)
        self.rho_hat = _coerce_numeric_vector("rho_hat", self.rho_hat)
        self.fold_ids = _coerce_fold_ids(self.fold_ids)
        self.basis_family = str(self.basis_family).strip().lower()
        self.basis_degree = _coerce_basis_degree(
            self.basis_family,
            self.basis_degree,
        )
        self.oracle_lane = normalize_oracle_lane(
            basis_family=self.basis_family,
            oracle_lane=str(self.oracle_lane),
        )

        n_obs = self.pi_hat.shape[0]
        for name, candidate in (
            ("phi0_hat", self.phi0_hat),
            ("phi1_hat", self.phi1_hat),
            ("rho_hat", self.rho_hat),
            ("fold_ids", self.fold_ids),
        ):
            if candidate.shape[0] != n_obs:
                raise ValueError(f"{name} must align with pi_hat")
        if self.valid_mask is None:
            self.valid_mask = np.ones(n_obs, dtype=bool)
        else:
            self.valid_mask = _coerce_boolean_mask("valid_mask", self.valid_mask)
            if self.valid_mask.shape[0] != n_obs:
                raise ValueError("valid_mask must align with pi_hat")

        for name, values in (
            ("pi_hat", self.pi_hat),
            ("phi0_hat", self.phi0_hat),
            ("phi1_hat", self.phi1_hat),
            ("rho_hat", self.rho_hat),
        ):
            _require_finite(name, values)
        if np.any((self.pi_hat <= 0.0) | (self.pi_hat >= 1.0)):
            raise ValueError("pi_hat must lie strictly between 0 and 1")

    @classmethod
    def from_predictions(
        cls,
        *,
        pi_hat: Any,
        phi0_hat: Any,
        phi1_hat: Any,
        treat: Any,
        fold_ids: Any,
        fold_diagnostics: list[FoldDiagnostics],
        basis_family: str,
        basis_degree: int,
        oracle_lane: str,
        valid_mask: Any | None = None,
    ) -> NuisancePayload:
        pi_array = _coerce_numeric_vector("pi_hat", pi_hat)
        treat_array = _coerce_numeric_vector("treat", treat)
        _require_finite("pi_hat", pi_array)
        if treat_array.shape[0] != pi_array.shape[0]:
            raise ValueError("treat must align with pi_hat")
        if not np.isin(treat_array, (0.0, 1.0)).all():
            raise ValueError("treat must be binary")
        if not (np.any(treat_array == 1.0) and np.any(treat_array == 0.0)):
            raise ValueError(
                "treat must contain at least one treated and one control observation"
            )
        if np.any((pi_array <= 0.0) | (pi_array >= 1.0)):
            raise ValueError("pi_hat must lie strictly between 0 and 1")

        rho_hat = (treat_array - pi_array) / (pi_array * (1.0 - pi_array))
        return cls(
            pi_hat=pi_array,
            phi0_hat=phi0_hat,
            phi1_hat=phi1_hat,
            rho_hat=rho_hat,
            fold_ids=fold_ids,
            valid_mask=valid_mask,
            fold_diagnostics=fold_diagnostics,
            basis_family=basis_family,
            basis_degree=basis_degree,
            oracle_lane=oracle_lane,
        )


@runtime_checkable
class NuisanceEstimator(Protocol):
    def fit(
        self,
        data: ValidatedHDDIDData,
        splits: CrossfitPlan,
    ) -> NuisancePayload: ...


_MASTER_SEED = 0x5EED


def _resolve_n_jobs(n_jobs: int) -> int:
    """Normalize n_jobs to a positive worker count."""
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs < 0:
        resolved = os.cpu_count() or 1
    else:
        resolved = max(1, int(n_jobs))
    if sys.version_info >= (3, 14):
        warnings.warn(
            "n_jobs>1 is not supported on Python 3.14+ due to joblib/loky "
            "compatibility issues; falling back to n_jobs=1.",
            UserWarning,
            stacklevel=2,
        )
        return 1
    return resolved


def _process_single_fold(
    fold: CrossfitFold,
    data: ValidatedHDDIDData,
    basis_matrix: np.ndarray,
    delta_y: np.ndarray,
    lane: str,
    *,
    propensity_estimator: Any | None = None,
    outcome_estimator: Any | None = None,
    worker_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FoldDiagnostics]:
    """Process a single cross-fitting fold for nuisance estimation.

    Fits propensity scores (pi_hat) and conditional outcome means
    (Phi_0, Phi_1) on the training set, then predicts on the holdout
    set.  Applies propensity trimming to determine valid holdout
    observations.

    This is the per-fold worker function called by
    :meth:`CrossfitNuisanceEstimator.fit`.  It is thread-safe (no shared
    mutable state) and can be parallelized via joblib or
    ProcessPoolExecutor.

    Parameters
    ----------
    fold : CrossfitFold
        Fold with train/holdout index partitions.
    data : ValidatedHDDIDData
        Full validated dataset.
    basis_matrix : ndarray of shape (n, L)
        Full-sample sieve basis matrix.
    delta_y : ndarray of shape (n,)
        First-differenced outcome Y1 - Y0.
    lane : str
        Oracle lane identifier for diagnostics.
    propensity_estimator : sklearn estimator or None
        Optional custom propensity score estimator.  When None, uses
        the built-in IRLS logistic regression.
    outcome_estimator : sklearn estimator or None
        Optional custom outcome model estimator.  When None, uses
        the built-in OLS.
    worker_seed : int or None
        Random seed for this worker (for reproducibility in parallel).

    Returns
    -------
    pi_holdout : ndarray of shape (n_holdout,)
        Predicted propensity scores on the holdout set.
    phi0_holdout : ndarray of shape (n_holdout,)
        Predicted conditional mean for controls on holdout.
    phi1_holdout : ndarray of shape (n_holdout,)
        Predicted conditional mean for treated on holdout.
    holdout_valid_mask : ndarray of bool, shape (n_holdout,)
        Mask of valid (non-trimmed) holdout observations.
    fold_diag : FoldDiagnostics
        Per-fold trimming and sample size diagnostics.

    Raises
    ------
    NuisanceTrainingSupportError
        If the training set lacks treated or control observations.
    ZeroValidHoldoutError
        If all holdout observations are trimmed by propensity bounds.

    Notes
    -----
    Implements the per-fold step of the K-fold cross-fitting procedure
    described in Section 3 of Ning, Peng & Tao (2020),
    arXiv preprint arXiv:2009.03151.  Cross-fitting avoids Donsker-class restrictions
    on the nuisance estimators.
    """
    if worker_seed is not None:
        np.random.seed(worker_seed)

    train_indices = np.asarray(fold.train_indices, dtype=int)
    holdout_indices = np.asarray(fold.holdout_indices, dtype=int)
    train_features = _feature_block(
        data,
        basis_matrix=basis_matrix,
        indices=train_indices,
    )
    holdout_features = _feature_block(
        data,
        basis_matrix=basis_matrix,
        indices=holdout_indices,
    )
    train_treat = np.asarray(data.treat[train_indices], dtype=int)
    train_target = delta_y[train_indices]
    treated_mask = train_treat == 1
    control_mask = train_treat == 0
    n_train_treated = int(np.sum(treated_mask))
    n_train_control = int(np.sum(control_mask))
    if n_train_treated == 0 or n_train_control == 0:
        raise NuisanceTrainingSupportError(
            fold_id=fold.fold_id,
            n_train_treated=n_train_treated,
            n_train_control=n_train_control,
        )

    if propensity_estimator is not None:
        from sklearn.base import clone

        pi_est = clone(propensity_estimator)
        pi_est.fit(train_features, train_treat)
        probas = np.asarray(pi_est.predict_proba(holdout_features), dtype=float)
        if probas.ndim != 2 or probas.shape[1] < 2:
            raise ValueError(
                "propensity estimator's predict_proba must return a 2-D array "
                "with at least two columns for binary treatment"
            )
        classes_arr = np.asarray(
            getattr(pi_est, "classes_", np.arange(probas.shape[1]))
        )
        if classes_arr.ndim != 1:
            raise ValueError("propensity estimator classes_ must be 1-D")
        pos_indices = np.flatnonzero(classes_arr == 1)
        if pos_indices.size == 1:
            pos_col = int(pos_indices[0])
        else:
            pos_col = probas.shape[1] - 1
        pi_holdout = probas[:, pos_col]
        pi_holdout = np.clip(
            pi_holdout,
            np.finfo(float).tiny,
            np.nextafter(1.0, 0.0),
        )
    else:
        pi_holdout = _fit_logistic_irls(
            train_features,
            train_treat.astype(float),
            holdout_features,
        )

    trim_lower = (
        fold.diagnostics.trim_lower
        if fold.diagnostics.trim_lower is not None
        else 0.01
    )
    trim_upper = (
        fold.diagnostics.trim_upper
        if fold.diagnostics.trim_upper is not None
        else 0.99
    )
    holdout_valid_mask = (pi_holdout >= trim_lower) & (pi_holdout <= trim_upper)
    n_valid_holdout = int(np.sum(holdout_valid_mask))
    if n_valid_holdout == 0:
        raise ZeroValidHoldoutError(
            fold_id=fold.fold_id,
            trim_lower=float(trim_lower),
            trim_upper=float(trim_upper),
        )

    if outcome_estimator is not None:
        from sklearn.base import clone

        phi1_est = clone(outcome_estimator)
        phi1_est.fit(train_features[treated_mask], train_target[treated_mask])
        phi1_holdout = np.asarray(
            phi1_est.predict(holdout_features), dtype=float
        ).ravel()

        phi0_est = clone(outcome_estimator)
        phi0_est.fit(train_features[control_mask], train_target[control_mask])
        phi0_holdout = np.asarray(
            phi0_est.predict(holdout_features), dtype=float
        ).ravel()
    else:
        phi1_holdout = _predict_linear_regression(
            train_features[treated_mask],
            train_target[treated_mask],
            holdout_features,
        )
        phi0_holdout = _predict_linear_regression(
            train_features[control_mask],
            train_target[control_mask],
            holdout_features,
        )

    fold_diag = FoldDiagnostics(
        fold_id=fold.fold_id,
        basis_family=data.basis_family,
        basis_degree=data.basis_degree,
        oracle_lane=lane,
        n_holdout_raw=int(holdout_indices.shape[0]),
        n_trimmed_propensity=int(holdout_indices.shape[0] - n_valid_holdout),
        n_valid_holdout=n_valid_holdout,
        trim_lower=float(trim_lower),
        trim_upper=float(trim_upper),
    )

    return pi_holdout, phi0_holdout, phi1_holdout, holdout_valid_mask, fold_diag


@dataclass(slots=True)
class CrossfitNuisanceEstimator:
    """Cross-fitting estimator for nuisance parameters.

    Orchestrates K-fold cross-fitting of propensity scores (pi_hat),
    conditional outcome means (Phi_0, Phi_1), and the propensity weight
    rho required by the doubly robust score in Eq. (5).  Cross-fitting
    avoids empirical-process (Donsker) restrictions by estimating
    nuisance functions on training folds and predicting on held-out folds.

    Attributes
    ----------
    oracle_lane : str or None
        If set, forces a specific nuisance model lane (e.g. ``'oracle'``).
    nuisance_estimator : Any or None
        Optional custom sklearn-compatible estimator for nuisance models.
    """
    oracle_lane: str | None = None
    nuisance_estimator: Any | None = None

    def fit(
        self,
        data: ValidatedHDDIDData,
        splits: CrossfitPlan,
        *,
        n_jobs: int = 1,
    ) -> NuisancePayload:
        """Estimate nuisance parameters via K-fold cross-fitting.

        Fits propensity scores pi_hat, conditional outcome means Phi_0
        and Phi_1, and the propensity weight rho using K-fold
        cross-fitting to avoid Donsker-class restrictions.  The resulting
        NuisancePayload feeds into the Eq. (5) doubly robust score
        construction.

        Parameters
        ----------
        data : ValidatedHDDIDData
            Validated input data with outcomes, treatment, covariates,
            and basis configuration.
        splits : CrossfitPlan
            K-fold cross-fitting partition from
            :func:`make_crossfit_splits`.
        n_jobs : int, default 1
            Number of parallel workers for fold processing.  Use -1
            for all available cores.  When 1, folds are processed
            sequentially.

        Returns
        -------
        NuisancePayload
            Contains pi_hat, phi0_hat, phi1_hat, rho_hat, valid_mask,
            fold_diagnostics, and basis configuration.

        Raises
        ------
        NuisanceTrainingSupportError
            If any fold's training set lacks treated or control obs.
        ZeroValidHoldoutError
            If any fold has zero valid holdout after trimming.

        Notes
        -----
        Implements the first-stage cross-fitting of Section 3 in
        Ning, Peng & Tao (2020), arXiv preprint arXiv:2009.03151.  The weight
        rho = (D - pi) / [pi(1-pi)] enters the Eq. (5) score.
        """
        basis_matrix = data.build_basis_matrix()
        delta_y = np.asarray(data.y1 - data.y0, dtype=float)
        n_obs = data.n_obs

        pi_hat = np.empty(n_obs, dtype=float)
        phi0_hat = np.empty(n_obs, dtype=float)
        phi1_hat = np.empty(n_obs, dtype=float)
        valid_mask = np.zeros(n_obs, dtype=bool)
        fold_diagnostics: list[FoldDiagnostics] = []

        lane = normalize_oracle_lane(
            basis_family=data.basis_family,
            oracle_lane=self.oracle_lane or _default_oracle_lane(data.basis_family),
        )

        propensity_estimator, outcome_estimator = _resolve_nuisance_estimators(
            self.nuisance_estimator
        )

        n_jobs = _resolve_n_jobs(n_jobs)

        if n_jobs == 1:
            for fold in splits.folds:
                result = _process_single_fold(
                    fold,
                    data,
                    basis_matrix,
                    delta_y,
                    lane,
                    propensity_estimator=propensity_estimator,
                    outcome_estimator=outcome_estimator,
                )
                pi_holdout, phi0_holdout, phi1_holdout, holdout_valid_mask, fold_diag = result
                holdout_indices = np.asarray(fold.holdout_indices, dtype=int)
                pi_hat[holdout_indices] = pi_holdout
                phi0_hat[holdout_indices] = phi0_holdout
                phi1_hat[holdout_indices] = phi1_holdout
                valid_mask[holdout_indices] = holdout_valid_mask
                fold_diagnostics.append(fold_diag)
        else:
            max_workers = min(n_jobs, len(splits.folds))

            if _HAS_JOBLIB:
                results = joblib.Parallel(n_jobs=max_workers, backend="loky")(
                    joblib.delayed(_process_single_fold)(
                        fold,
                        data,
                        basis_matrix,
                        delta_y,
                        lane,
                        propensity_estimator=propensity_estimator,
                        outcome_estimator=outcome_estimator,
                        worker_seed=_MASTER_SEED + fold.fold_id,
                    )
                    for fold in splits.folds
                )
            else:
                mp_context = mp.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=max_workers, mp_context=mp_context
                ) as executor:
                    futures = [
                        executor.submit(
                            _process_single_fold,
                            fold,
                            data,
                            basis_matrix,
                            delta_y,
                            lane,
                            propensity_estimator=propensity_estimator,
                            outcome_estimator=outcome_estimator,
                            worker_seed=_MASTER_SEED + fold.fold_id,
                        )
                        for fold in splits.folds
                    ]
                    results = [future.result() for future in futures]

            for fold, result in zip(splits.folds, results):
                pi_holdout, phi0_holdout, phi1_holdout, holdout_valid_mask, fold_diag = result
                holdout_indices = np.asarray(fold.holdout_indices, dtype=int)
                pi_hat[holdout_indices] = pi_holdout
                phi0_hat[holdout_indices] = phi0_holdout
                phi1_hat[holdout_indices] = phi1_holdout
                valid_mask[holdout_indices] = holdout_valid_mask
                fold_diagnostics.append(fold_diag)

        return NuisancePayload.from_predictions(
            pi_hat=pi_hat,
            phi0_hat=phi0_hat,
            phi1_hat=phi1_hat,
            treat=data.treat,
            fold_ids=splits.fold_ids,
            valid_mask=valid_mask,
            fold_diagnostics=fold_diagnostics,
            basis_family=data.basis_family,
            basis_degree=data.basis_degree,
            oracle_lane=lane,
        )
