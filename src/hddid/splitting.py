from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from .results import FoldDiagnostics


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
            return any(_contains_boolean_or_string_alias(item) for item in value.flat)
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean_or_string_alias(item) for item in value)
    return False


@dataclass(slots=True)
class CrossfitFold:
    """A single cross-fitting fold with train/holdout index partitions.

    Each fold serves as one split in the K-fold cross-fitting procedure
    used for nuisance estimation (Section 3 of the paper).

    Attributes
    ----------
    fold_id : int
        One-based fold identifier.
    train_indices : ndarray of int
        Indices of observations used for first-stage nuisance estimation.
    holdout_indices : ndarray of int
        Indices of observations used for second-stage score evaluation.
    diagnostics : FoldDiagnostics
        Per-fold propensity trimming and sample size diagnostics.
    """

    fold_id: int
    train_indices: np.ndarray
    holdout_indices: np.ndarray
    diagnostics: FoldDiagnostics


@dataclass(slots=True)
class CrossfitPlan:
    """Complete K-fold cross-fitting partition of the sample.

    Produced by :func:`make_crossfit_splits` and consumed by the
    nuisance estimation stage to avoid Donsker-class restrictions.

    Attributes
    ----------
    fold_ids : ndarray of int, shape (n_obs,)
        Fold assignment for each observation (1-based).
    folds : list of CrossfitFold
        Ordered list of K fold objects with train/holdout splits.
    """

    fold_ids: np.ndarray
    folds: list[CrossfitFold]


def _require_nonnegative_seed(name: str, value: int | None) -> int | None:
    """Validate that ``value`` is a non-negative integer or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer or None")
    seed = int(value)
    if seed < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return seed


def _require_trim_bound(name: str, value: Any) -> float:
    """Validate that ``value`` is a finite scalar trim bound."""
    if _contains_boolean_or_string_alias(value):
        raise ValueError(f"{name} must be a finite numeric trim bound")
    raw_value = np.asarray(value)
    if _contains_boolean_or_string_alias(raw_value):
        raise ValueError(f"{name} must be a finite numeric trim bound")
    if raw_value.ndim != 0:
        raise ValueError(f"{name} must be a scalar trim bound")
    numeric = float(raw_value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite numeric trim bound")
    return numeric


def make_crossfit_splits(
    n_obs: int,
    n_folds: int,
    random_state: int | None,
    *,
    trim_lower: float = 0.01,
    trim_upper: float = 0.99,
) -> CrossfitPlan:
    """Generate cross-fitting sample splits for nuisance estimation.

    Randomly partitions the sample into K non-overlapping folds of
    approximately equal size. Each fold serves as the holdout set
    once, with the remaining observations used for training.

    Parameters
    ----------
    n_obs : int
        Total number of observations. Must be >= 2.
    n_folds : int
        Number of cross-fitting folds K. Must satisfy
        2 <= n_folds <= n_obs.
    random_state : int or None
        Seed for the random number generator controlling the
        permutation. Use None for non-deterministic splits.
    trim_lower : float, default 0.01
        Lower propensity score trimming threshold recorded in
        fold diagnostics.
    trim_upper : float, default 0.99
        Upper propensity score trimming threshold recorded in
        fold diagnostics.

    Returns
    -------
    CrossfitPlan
        Contains fold_ids (array of fold assignments per observation)
        and a list of CrossfitFold objects with train/holdout indices
        and per-fold diagnostics.

    Raises
    ------
    ValueError
        If n_obs < 2, n_folds is out of range, random_state is
        invalid, or trim bounds are inconsistent.

    Notes
    -----
    Cross-fitting is used to avoid Donsker-class restrictions on the
    nuisance estimators, as described in Section 3 of
    Ning, Peng, and Tao (2020). Each fold's nuisance estimates are
    computed on the training set and evaluated on the holdout set.

    Reference: Ning, Peng, and Tao (2020), arXiv preprint arXiv:2009.03151.
    """
    if isinstance(n_obs, bool) or not isinstance(n_obs, Integral) or int(n_obs) < 2:
        raise ValueError("n_obs must be an integer greater than or equal to 2")
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral):
        raise ValueError("n_folds must be an integer")

    n_obs_int = int(n_obs)
    n_folds_int = int(n_folds)
    if n_folds_int < 2 or n_folds_int > n_obs_int:
        raise ValueError("n_folds must be between 2 and n_obs")

    seed_value = _require_nonnegative_seed("random_state", random_state)
    trim_lower_value = _require_trim_bound("trim_lower", trim_lower)
    trim_upper_value = _require_trim_bound("trim_upper", trim_upper)
    if not 0.0 <= trim_lower_value < trim_upper_value <= 1.0:
        raise ValueError("trim bounds must satisfy 0 <= trim_lower < trim_upper <= 1")

    rng = np.random.default_rng(seed_value)
    permutation = rng.permutation(n_obs_int)
    holdout_groups = np.array_split(permutation, n_folds_int)
    all_indices = np.arange(n_obs_int, dtype=int)
    fold_ids = np.zeros(n_obs_int, dtype=int)
    folds: list[CrossfitFold] = []

    for fold_number, holdout_indices in enumerate(holdout_groups, start=1):
        holdout = np.sort(holdout_indices.astype(int, copy=False))
        train = np.setdiff1d(all_indices, holdout, assume_unique=True)
        fold_ids[holdout] = fold_number
        diagnostics = FoldDiagnostics(
            fold_id=fold_number,
            n_holdout_raw=int(holdout.size),
            n_trimmed_propensity=0,
            n_valid_holdout=int(holdout.size),
            trim_lower=trim_lower_value,
            trim_upper=trim_upper_value,
        )
        folds.append(
            CrossfitFold(
                fold_id=fold_number,
                train_indices=train,
                holdout_indices=holdout,
                diagnostics=diagnostics,
            )
        )

    return CrossfitPlan(fold_ids=fold_ids, folds=folds)
