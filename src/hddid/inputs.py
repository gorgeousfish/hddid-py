from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from .basis import build_sieve_basis


def _contains_boolean_or_string_alias(value: Any) -> bool:
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


def _coerce_vector(name: str, values: Any, *, allow_scalar: bool = False) -> np.ndarray:
    if _contains_boolean_or_string_alias(values):
        raise ValueError(f"{name} must be numeric, not boolean or string")
    raw_array = np.asarray(values)
    if _contains_boolean_or_string_alias(raw_array):
        raise ValueError(f"{name} must be numeric, not boolean or string")
    array = raw_array.astype(float)
    if array.ndim == 0:
        if allow_scalar:
            return array.reshape(1)
        raise ValueError(f"{name} must have the same number of observations")
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    raise ValueError(f"{name} must be one-dimensional z in Phase 2")


def _coerce_matrix(name: str, values: Any) -> np.ndarray:
    if _contains_boolean_or_string_alias(values):
        raise ValueError(f"{name} must be numeric, not boolean or string")
    raw_array = np.asarray(values)
    if _contains_boolean_or_string_alias(raw_array):
        raise ValueError(f"{name} must be numeric, not boolean or string")
    array = raw_array.astype(float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return array


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(slots=True)
class ValidatedHDDIDData:
    y0: np.ndarray
    y1: np.ndarray
    treat: np.ndarray
    x: np.ndarray
    z: np.ndarray
    z0: np.ndarray
    basis_family: str
    basis_degree: int
    alpha: float

    @property
    def n_obs(self) -> int:
        return int(self.y0.shape[0])

    def build_basis_matrix(self) -> np.ndarray:
        return build_sieve_basis(self.z, self.basis_family, self.basis_degree)

    def build_evaluation_basis(self) -> np.ndarray:
        return build_sieve_basis(self.z0, self.basis_family, self.basis_degree)


def validate_inputs(
    *,
    y0: Any,
    y1: Any,
    treat: Any,
    x: Any,
    z: Any,
    z0: Any,
    basis_family: str,
    basis_degree: int,
    alpha: float,
) -> ValidatedHDDIDData:
    for _name, _value in (
        ("y0", y0),
        ("y1", y1),
        ("treat", treat),
        ("x", x),
        ("z", z),
        ("z0", z0),
    ):
        if _value is None:
            raise ValueError(f"{_name} must not be None")

    for _name, _value in (
        ("y0", y0),
        ("y1", y1),
        ("treat", treat),
        ("x", x),
        ("z", z),
        ("z0", z0),
    ):
        if _value is None:
            raise ValueError(f"{_name} must not be None")

    y0_array = _coerce_vector("y0", y0)
    y1_array = _coerce_vector("y1", y1)
    treat_array = _coerce_vector("treat", treat)
    x_array = _coerce_matrix("x", x)
    z_array = _coerce_vector("z", z)
    z0_array = _coerce_vector("z0", z0, allow_scalar=True)

    for name, values in (
        ("y0", y0_array),
        ("y1", y1_array),
        ("treat", treat_array),
        ("x", x_array),
        ("z", z_array),
        ("z0", z0_array),
    ):
        _require_finite(name, values)

    n_obs = y0_array.shape[0]
    if n_obs == 0:
        raise ValueError("raw inputs must contain at least one observation")
    if any(
        candidate.shape[0] != n_obs
        for candidate in (y1_array, treat_array, x_array, z_array)
    ):
        raise ValueError("All raw inputs must have the same number of observations")
    if z0_array.shape[0] == 0:
        raise ValueError("z0 must contain at least one evaluation point")

    if not np.isin(treat_array, (0.0, 1.0)).all():
        raise ValueError("treat must be binary")
    treat_indicator = treat_array.astype(int, copy=False)
    if not (np.any(treat_indicator == 1) and np.any(treat_indicator == 0)):
        raise ValueError(
            "treat must contain at least one treated and one control observation"
        )

    if z_array.ndim != 1:
        raise ValueError("Phase 2 currently supports one-dimensional z only")

    family = str(basis_family).strip().lower()
    if isinstance(basis_degree, bool) or not isinstance(basis_degree, Integral):
        raise ValueError("basis_degree must be an integer")
    degree = int(basis_degree)

    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be numeric, not boolean")
    if isinstance(alpha, (str, bytes, np.str_, np.bytes_)):
        raise ValueError("alpha must be numeric, not string")
    if not isinstance(alpha, Real):
        raise ValueError("alpha must be numeric")
    alpha_value = float(alpha)
    if not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")

    # Validate the basis contract now so downstream estimators do not guess it later.
    build_sieve_basis(z_array, family, degree)
    build_sieve_basis(z0_array, family, degree)

    return ValidatedHDDIDData(
        y0=y0_array,
        y1=y1_array,
        treat=treat_indicator,
        x=x_array,
        z=z_array,
        z0=z0_array,
        basis_family=family,
        basis_degree=degree,
        alpha=alpha_value,
    )
