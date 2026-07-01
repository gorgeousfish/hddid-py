from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from hddid.validation import (
    Phase7OuterInferenceObjectContract,
    build_phase7_outer_inference_reference_contract,
)


_DETERMINISTIC_OBJECT_FIELDS = (
    "evaluation_grid",
    "bar_f_at_z0",
    "sigma_z_hat",
    "covariance_at_grid",
)
_MONTE_CARLO_FUNCTIONAL_FIELDS = (
    "uniform_critical_value",
    "n_boot",
    "random_state",
)


def _matrix_square_root_psd(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    clipped = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(clipped))


def replay_phase7_outer_inference_uniform_critical_value(
    contract: Phase7OuterInferenceObjectContract,
    *,
    random_state: int | None = None,
    n_boot: int | None = None,
) -> float:
    random_state_value = (
        contract.random_state if random_state is None else int(random_state)
    )
    n_boot_value = contract.n_boot if n_boot is None else int(n_boot)
    if n_boot_value <= 0:
        raise ValueError("n_boot must be positive")

    covariance_at_grid = np.asarray(contract.covariance_at_grid, dtype=float)
    standardization = np.sqrt(np.maximum(np.diag(covariance_at_grid), 0.0))
    if np.any(standardization <= 0.0):
        raise ValueError("covariance_at_grid diagonal must be strictly positive")

    covariance_sqrt = _matrix_square_root_psd(covariance_at_grid)
    rng = np.random.default_rng(random_state_value)
    gaussian_draws = (
        rng.standard_normal(size=(n_boot_value, covariance_at_grid.shape[0]))
        @ covariance_sqrt.T
    )
    simulated_suprema = np.max(
        np.abs(gaussian_draws / standardization[None, :]),
        axis=1,
    )
    return float(np.quantile(simulated_suprema, 1.0 - float(contract.alpha)))


@dataclass(slots=True)
class Phase7OuterInferenceDeterminismProbeReport:
    stage_label: str
    contract_status: str
    deterministic_object_fields: tuple[str, ...]
    monte_carlo_functional_fields: tuple[str, ...]
    replay_n_boot: int
    replay_random_state: int
    contract_uniform_critical_value: float
    replay_uniform_critical_value: float
    replay_absolute_difference: float
    replay_matches_contract: bool
    alternate_random_state: int
    alternate_random_state_uniform_critical_value: float
    alternate_random_state_delta: float
    alternate_n_boot: int
    alternate_n_boot_uniform_critical_value: float
    alternate_n_boot_delta: float
    invariant_object_fields: tuple[str, ...]
    recommendation_rationale: str

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.contract_status = str(self.contract_status).strip()
        self.deterministic_object_fields = tuple(
            str(field).strip() for field in self.deterministic_object_fields
        )
        self.monte_carlo_functional_fields = tuple(
            str(field).strip() for field in self.monte_carlo_functional_fields
        )
        self.replay_n_boot = int(self.replay_n_boot)
        self.replay_random_state = int(self.replay_random_state)
        self.contract_uniform_critical_value = float(
            self.contract_uniform_critical_value
        )
        self.replay_uniform_critical_value = float(self.replay_uniform_critical_value)
        self.replay_absolute_difference = float(self.replay_absolute_difference)
        self.replay_matches_contract = bool(self.replay_matches_contract)
        self.alternate_random_state = int(self.alternate_random_state)
        self.alternate_random_state_uniform_critical_value = float(
            self.alternate_random_state_uniform_critical_value
        )
        self.alternate_random_state_delta = float(self.alternate_random_state_delta)
        self.alternate_n_boot = int(self.alternate_n_boot)
        self.alternate_n_boot_uniform_critical_value = float(
            self.alternate_n_boot_uniform_critical_value
        )
        self.alternate_n_boot_delta = float(self.alternate_n_boot_delta)
        self.invariant_object_fields = tuple(
            str(field).strip() for field in self.invariant_object_fields
        )
        self.recommendation_rationale = str(self.recommendation_rationale).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "contract_status": self.contract_status,
            "deterministic_object_fields": list(self.deterministic_object_fields),
            "monte_carlo_functional_fields": list(self.monte_carlo_functional_fields),
            "replay_n_boot": self.replay_n_boot,
            "replay_random_state": self.replay_random_state,
            "contract_uniform_critical_value": self.contract_uniform_critical_value,
            "replay_uniform_critical_value": self.replay_uniform_critical_value,
            "replay_absolute_difference": self.replay_absolute_difference,
            "replay_matches_contract": self.replay_matches_contract,
            "alternate_random_state": self.alternate_random_state,
            "alternate_random_state_uniform_critical_value": (
                self.alternate_random_state_uniform_critical_value
            ),
            "alternate_random_state_delta": self.alternate_random_state_delta,
            "alternate_n_boot": self.alternate_n_boot,
            "alternate_n_boot_uniform_critical_value": self.alternate_n_boot_uniform_critical_value,
            "alternate_n_boot_delta": self.alternate_n_boot_delta,
            "invariant_object_fields": list(self.invariant_object_fields),
            "recommendation_rationale": self.recommendation_rationale,
        }


def build_phase7_outer_inference_determinism_probe_report(
    contract: Phase7OuterInferenceObjectContract,
    *,
    alternate_random_state: int = 321,
    alternate_n_boot: int = 1024,
) -> Phase7OuterInferenceDeterminismProbeReport:
    replay_uniform_critical_value = (
        replay_phase7_outer_inference_uniform_critical_value(contract)
    )
    alternate_random_state_uniform_critical_value = (
        replay_phase7_outer_inference_uniform_critical_value(
            contract,
            random_state=alternate_random_state,
            n_boot=contract.n_boot,
        )
    )
    alternate_n_boot_uniform_critical_value = (
        replay_phase7_outer_inference_uniform_critical_value(
            contract,
            random_state=contract.random_state,
            n_boot=alternate_n_boot,
        )
    )
    replay_absolute_difference = abs(
        replay_uniform_critical_value - float(contract.uniform_critical_value)
    )
    return Phase7OuterInferenceDeterminismProbeReport(
        stage_label="phase7-outer-inference-determinism-probe",
        contract_status=contract.status,
        deterministic_object_fields=_DETERMINISTIC_OBJECT_FIELDS,
        monte_carlo_functional_fields=_MONTE_CARLO_FUNCTIONAL_FIELDS,
        replay_n_boot=contract.n_boot,
        replay_random_state=contract.random_state,
        contract_uniform_critical_value=contract.uniform_critical_value,
        replay_uniform_critical_value=replay_uniform_critical_value,
        replay_absolute_difference=replay_absolute_difference,
        replay_matches_contract=replay_absolute_difference <= 1e-12,
        alternate_random_state=alternate_random_state,
        alternate_random_state_uniform_critical_value=(
            alternate_random_state_uniform_critical_value
        ),
        alternate_random_state_delta=abs(
            alternate_random_state_uniform_critical_value
            - replay_uniform_critical_value
        ),
        alternate_n_boot=alternate_n_boot,
        alternate_n_boot_uniform_critical_value=alternate_n_boot_uniform_critical_value,
        alternate_n_boot_delta=abs(
            alternate_n_boot_uniform_critical_value - replay_uniform_critical_value
        ),
        invariant_object_fields=_DETERMINISTIC_OBJECT_FIELDS,
        recommendation_rationale=(
            "Future Trigger 3 parity should pin the deterministic upstream objects "
            "first and only then compare the Monte Carlo functional. Under a fixed "
            "`covariance_at_grid`, changing `random_state` or `n_boot` moves only "
            "`uniform_critical_value`; it does not justify returning to archived "
            "`CIuniform` surface matching."
        ),
    )


@lru_cache(maxsize=1)
def run_phase7_outer_inference_determinism_probe() -> (
    Phase7OuterInferenceDeterminismProbeReport
):
    return build_phase7_outer_inference_determinism_probe_report(
        build_phase7_outer_inference_reference_contract()
    )
