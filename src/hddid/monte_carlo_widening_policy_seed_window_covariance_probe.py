from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .monte_carlo_widening_policy_seed_role_split_probe import (
    run_phase7_monte_carlo_widening_policy_seed_role_split_probe,
)
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
    run_phase7_monte_carlo_widening_policy_spec,
)
from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy
from .validation import (
    MonteCarloDesign,
    Phase7OuterInferenceObjectContract,
    _clone_design_with_evaluation_grid,
    _fit_phase7_nonparametric_object_payload,
    _match_runtime_probe_design,
    _phase7_runtime_probe_replication_seed,
    build_phase7_outer_inference_object_contract,
)

_CANONICAL_WINDOW_LABEL = "near_zero_grid"
_CANONICAL_WINDOW_GRID = (0.05, 0.15, 0.25)
_CANONICAL_N_BOOT = 64


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{str(dgp_name).strip().upper()}/{int(n_obs)}/{int(p)}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_grid_value(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return f"{value:.1f}"
    text = f"{value:.2f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def _format_grid(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(_format_grid_value(value) for value in values) + "]"


def _format_ratio(value: float) -> str:
    return f"x{_format_float(value)}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _center_index(evaluation_grid: np.ndarray) -> int:
    grid = np.asarray(evaluation_grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("evaluation_grid must be one-dimensional")
    if grid.size == 0:
        raise ValueError("evaluation_grid must not be empty")
    midpoint = float((np.min(grid) + np.max(grid)) / 2.0)
    return int(np.argmin(np.abs(grid - midpoint)))


def _mean_off_center_correlation_to_center(
    covariance_at_grid: np.ndarray,
    *,
    center_index: int,
) -> float:
    covariance = np.asarray(covariance_at_grid, dtype=float)
    grid_size = covariance.shape[0]
    if covariance.shape != (grid_size, grid_size):
        raise ValueError("covariance_at_grid must be square")

    diagonal = np.diag(covariance).astype(float)
    if np.any(diagonal <= 0.0):
        raise ValueError("covariance_at_grid diagonal must be strictly positive")

    correlations: list[float] = []
    center_variance = float(diagonal[center_index])
    for index in range(grid_size):
        if index == center_index:
            continue
        denominator = float(np.sqrt(diagonal[index] * center_variance))
        correlations.append(abs(float(covariance[index, center_index] / denominator)))
    if not correlations:
        raise ValueError(
            "seed window covariance probe requires at least one off-center point"
        )
    return float(np.mean(correlations))


def _leading_positive_share(covariance_at_grid: np.ndarray) -> float:
    covariance = np.asarray(covariance_at_grid, dtype=float)
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    positive = eigenvalues[eigenvalues > 0.0]
    if positive.size == 0:
        raise ValueError(
            "seed window covariance probe requires positive covariance mass"
        )
    return float(positive[-1] / np.sum(positive))


def _require_matching_grids(
    coverage_anchor_contract: Phase7OuterInferenceObjectContract,
    overshoot_companion_contract: Phase7OuterInferenceObjectContract,
) -> tuple[float, ...]:
    anchor_grid = tuple(
        float(value) for value in coverage_anchor_contract.evaluation_grid
    )
    companion_grid = tuple(
        float(value) for value in overshoot_companion_contract.evaluation_grid
    )
    if not np.allclose(anchor_grid, companion_grid):
        raise ValueError(
            "seed window covariance probe requires matching evaluation grids"
        )
    return anchor_grid


def _window_covariance_driver(
    *,
    covariance_trace_ratio: float,
    uniform_critical_value_ratio: float,
    off_center_correlation_gap: float,
) -> str:
    if (
        covariance_trace_ratio > uniform_critical_value_ratio
        and off_center_correlation_gap > 0.0
    ):
        return "covariance-coupling-and-scale-overshoot"
    if uniform_critical_value_ratio >= covariance_trace_ratio:
        return "uniform-critical-drift"
    return "mixed-window-covariance-drift"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicySeedWindowCovarianceReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    window_label: str
    evaluation_grid: tuple[float, ...]
    coverage_anchor_contract: Phase7OuterInferenceObjectContract
    overshoot_companion_contract: Phase7OuterInferenceObjectContract
    coverage_anchor_random_state: int
    overshoot_companion_random_state: int
    center_index: int
    center_grid_value: float
    coverage_anchor_covariance_trace: float
    overshoot_companion_covariance_trace: float
    covariance_trace_ratio: float
    coverage_anchor_mean_off_center_correlation_to_center: float
    overshoot_companion_mean_off_center_correlation_to_center: float
    off_center_correlation_gap: float
    coverage_anchor_leading_positive_share: float
    overshoot_companion_leading_positive_share: float
    leading_positive_share_gap: float
    uniform_critical_value_ratio: float
    window_covariance_driver: str
    canonical_seed_window_covariance_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.window_label = str(self.window_label).strip()
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.overshoot_companion_random_state = int(
            self.overshoot_companion_random_state
        )
        self.center_index = int(self.center_index)
        self.center_grid_value = float(self.center_grid_value)
        self.coverage_anchor_covariance_trace = float(
            self.coverage_anchor_covariance_trace
        )
        self.overshoot_companion_covariance_trace = float(
            self.overshoot_companion_covariance_trace
        )
        self.covariance_trace_ratio = float(self.covariance_trace_ratio)
        self.coverage_anchor_mean_off_center_correlation_to_center = float(
            self.coverage_anchor_mean_off_center_correlation_to_center
        )
        self.overshoot_companion_mean_off_center_correlation_to_center = float(
            self.overshoot_companion_mean_off_center_correlation_to_center
        )
        self.off_center_correlation_gap = float(self.off_center_correlation_gap)
        self.coverage_anchor_leading_positive_share = float(
            self.coverage_anchor_leading_positive_share
        )
        self.overshoot_companion_leading_positive_share = float(
            self.overshoot_companion_leading_positive_share
        )
        self.leading_positive_share_gap = float(self.leading_positive_share_gap)
        self.uniform_critical_value_ratio = float(self.uniform_critical_value_ratio)
        self.window_covariance_driver = str(self.window_covariance_driver).strip()
        self.canonical_seed_window_covariance_digest = tuple(
            str(line).rstrip() for line in self.canonical_seed_window_covariance_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "window_label": self.window_label,
            "evaluation_grid": list(self.evaluation_grid),
            "coverage_anchor_contract": self.coverage_anchor_contract.to_dict(),
            "overshoot_companion_contract": self.overshoot_companion_contract.to_dict(),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "overshoot_companion_random_state": self.overshoot_companion_random_state,
            "center_index": self.center_index,
            "center_grid_value": self.center_grid_value,
            "coverage_anchor_covariance_trace": self.coverage_anchor_covariance_trace,
            "overshoot_companion_covariance_trace": (
                self.overshoot_companion_covariance_trace
            ),
            "covariance_trace_ratio": self.covariance_trace_ratio,
            "coverage_anchor_mean_off_center_correlation_to_center": (
                self.coverage_anchor_mean_off_center_correlation_to_center
            ),
            "overshoot_companion_mean_off_center_correlation_to_center": (
                self.overshoot_companion_mean_off_center_correlation_to_center
            ),
            "off_center_correlation_gap": self.off_center_correlation_gap,
            "coverage_anchor_leading_positive_share": (
                self.coverage_anchor_leading_positive_share
            ),
            "overshoot_companion_leading_positive_share": (
                self.overshoot_companion_leading_positive_share
            ),
            "leading_positive_share_gap": self.leading_positive_share_gap,
            "uniform_critical_value_ratio": self.uniform_critical_value_ratio,
            "window_covariance_driver": self.window_covariance_driver,
            "canonical_seed_window_covariance_digest": list(
                self.canonical_seed_window_covariance_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_seed_window_covariance_report(
    *,
    coverage_anchor_contract: Phase7OuterInferenceObjectContract,
    overshoot_companion_contract: Phase7OuterInferenceObjectContract,
    binding_design: tuple[str, int, int],
    window_label: str,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicySeedWindowCovarianceReport:
    resolved_policy = (
        build_phase7_canonical_monte_carlo_widening_policy()
        if policy is None
        else policy
    )
    evaluation_grid = _require_matching_grids(
        coverage_anchor_contract, overshoot_companion_contract
    )
    center_index = _center_index(np.asarray(evaluation_grid, dtype=float))
    center_grid_value = float(evaluation_grid[center_index])

    coverage_anchor_trace = float(
        np.trace(np.asarray(coverage_anchor_contract.covariance_at_grid, dtype=float))
    )
    overshoot_companion_trace = float(
        np.trace(
            np.asarray(overshoot_companion_contract.covariance_at_grid, dtype=float)
        )
    )
    if coverage_anchor_trace <= 0.0:
        raise ValueError("coverage anchor covariance trace must be positive")

    coverage_anchor_mean_correlation = _mean_off_center_correlation_to_center(
        coverage_anchor_contract.covariance_at_grid,
        center_index=center_index,
    )
    overshoot_companion_mean_correlation = _mean_off_center_correlation_to_center(
        overshoot_companion_contract.covariance_at_grid,
        center_index=center_index,
    )
    coverage_anchor_leading_share = _leading_positive_share(
        coverage_anchor_contract.covariance_at_grid
    )
    overshoot_companion_leading_share = _leading_positive_share(
        overshoot_companion_contract.covariance_at_grid
    )
    covariance_trace_ratio = float(overshoot_companion_trace / coverage_anchor_trace)
    off_center_correlation_gap = float(
        overshoot_companion_mean_correlation - coverage_anchor_mean_correlation
    )
    leading_positive_share_gap = float(
        overshoot_companion_leading_share - coverage_anchor_leading_share
    )
    uniform_critical_value_ratio = float(
        overshoot_companion_contract.uniform_critical_value
        / coverage_anchor_contract.uniform_critical_value
    )
    driver = _window_covariance_driver(
        covariance_trace_ratio=covariance_trace_ratio,
        uniform_critical_value_ratio=uniform_critical_value_ratio,
        off_center_correlation_gap=off_center_correlation_gap,
    )

    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*binding_design)}` on `{str(window_label).strip()}` "
        f"(`{_format_grid(evaluation_grid)}`): coverage anchor seed "
        f"`{coverage_anchor_contract.random_state}` keeps covariance trace "
        f"`{_format_float(coverage_anchor_trace)}`, mean off-center correlation to "
        f"center `{_format_float(coverage_anchor_mean_correlation)}`, leading "
        f"positive share `{_format_float(coverage_anchor_leading_share)}`, uniform "
        f"critical value `{_format_float(coverage_anchor_contract.uniform_critical_value)}`",
        "- overshoot companion seed "
        f"`{overshoot_companion_contract.random_state}`: covariance trace "
        f"`{_format_float(overshoot_companion_trace)}`, mean off-center correlation "
        f"to center `{_format_float(overshoot_companion_mean_correlation)}`, leading "
        f"positive share `{_format_float(overshoot_companion_leading_share)}`, "
        f"uniform critical value `{_format_float(overshoot_companion_contract.uniform_critical_value)}`",
        "- window covariance split: trace ratio "
        f"`{_format_ratio(covariance_trace_ratio)}`, center-coupling gap "
        f"`{_format_signed(off_center_correlation_gap)}`, leading-share gap "
        f"`{_format_signed(leading_positive_share_gap)}`, critical ratio "
        f"`{_format_ratio(uniform_critical_value_ratio)}`",
        "- current implication: "
        f"`{driver}`; inspect local covariance / `sigma_z_hat` access before "
        "changing uniform critical tuning",
    )
    return Phase7MonteCarloWideningPolicySeedWindowCovarianceReport(
        stage_label="phase7-monte-carlo-widening-policy-seed-window-covariance-probe",
        policy_digest=resolved_policy.to_digest(),
        binding_design=binding_design,
        window_label=window_label,
        evaluation_grid=evaluation_grid,
        coverage_anchor_contract=coverage_anchor_contract,
        overshoot_companion_contract=overshoot_companion_contract,
        coverage_anchor_random_state=coverage_anchor_contract.random_state,
        overshoot_companion_random_state=overshoot_companion_contract.random_state,
        center_index=center_index,
        center_grid_value=center_grid_value,
        coverage_anchor_covariance_trace=coverage_anchor_trace,
        overshoot_companion_covariance_trace=overshoot_companion_trace,
        covariance_trace_ratio=covariance_trace_ratio,
        coverage_anchor_mean_off_center_correlation_to_center=(
            coverage_anchor_mean_correlation
        ),
        overshoot_companion_mean_off_center_correlation_to_center=(
            overshoot_companion_mean_correlation
        ),
        off_center_correlation_gap=off_center_correlation_gap,
        coverage_anchor_leading_positive_share=coverage_anchor_leading_share,
        overshoot_companion_leading_positive_share=overshoot_companion_leading_share,
        leading_positive_share_gap=leading_positive_share_gap,
        uniform_critical_value_ratio=uniform_critical_value_ratio,
        window_covariance_driver=driver,
        canonical_seed_window_covariance_digest=canonical_digest,
    )


def _canonical_window_contract(
    *,
    random_state: int,
    binding_design: tuple[str, int, int],
    designs: tuple[MonteCarloDesign, ...],
    window_grid: tuple[float, ...],
    n_boot: int,
) -> Phase7OuterInferenceObjectContract:
    dgp_name, n_obs, p = binding_design
    base_design = _match_runtime_probe_design(
        designs,
        dgp_name=dgp_name,
        n_obs=n_obs,
        p=p,
    )
    replay_design = _clone_design_with_evaluation_grid(
        base_design,
        evaluation_grid=np.asarray(window_grid, dtype=float),
    )
    replication_seed = _phase7_runtime_probe_replication_seed(
        random_state=int(random_state),
        designs=designs,
        dgp_name=dgp_name,
        n_obs=n_obs,
        p=p,
    )
    dataset, nonparametric_payload = _fit_phase7_nonparametric_object_payload(
        design=replay_design,
        replication_seed=replication_seed,
        n_boot=int(n_boot),
    )
    contract = build_phase7_outer_inference_object_contract(
        nonparametric_payload,
        evaluation_grid=dataset.z0,
        n_valid=int(nonparametric_payload.optimization_metadata["n_valid_obs"]),
    )
    contract.random_state = int(random_state)
    return contract


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_seed_window_covariance_probe() -> (
    Phase7MonteCarloWideningPolicySeedWindowCovarianceReport
):
    policy_spec = run_phase7_monte_carlo_widening_policy_spec()
    binding_design = (
        policy_spec.quality_risk_designs[0]
        if policy_spec.quality_risk_designs
        else policy_spec.stable_partial_designs[0]
    )
    designs = tuple(
        MonteCarloDesign(dgp_name=dgp_name, n_obs=n_obs, p=p)
        for dgp_name, n_obs, p in policy_spec.stable_partial_designs
    )
    seed_role_split = run_phase7_monte_carlo_widening_policy_seed_role_split_probe()
    coverage_anchor_contract = _canonical_window_contract(
        random_state=seed_role_split.coverage_anchor.random_state,
        binding_design=binding_design,
        designs=designs,
        window_grid=_CANONICAL_WINDOW_GRID,
        n_boot=_CANONICAL_N_BOOT,
    )
    overshoot_companion_contract = _canonical_window_contract(
        random_state=seed_role_split.scale_peak.random_state,
        binding_design=binding_design,
        designs=designs,
        window_grid=_CANONICAL_WINDOW_GRID,
        n_boot=_CANONICAL_N_BOOT,
    )
    return build_phase7_monte_carlo_widening_policy_seed_window_covariance_report(
        coverage_anchor_contract=coverage_anchor_contract,
        overshoot_companion_contract=overshoot_companion_contract,
        binding_design=binding_design,
        window_label=_CANONICAL_WINDOW_LABEL,
    )


def _build_seed_window_covariance_report_for_binding_design(
    binding_design: tuple[str, int, int],
    *,
    coverage_anchor_random_state: int,
    overshoot_companion_random_state: int,
    policy: Phase7MonteCarloWideningPolicy | None = None,
) -> Phase7MonteCarloWideningPolicySeedWindowCovarianceReport:
    policy_spec = run_phase7_monte_carlo_widening_policy_spec()
    designs = tuple(
        MonteCarloDesign(dgp_name=dgp_name, n_obs=n_obs, p=p)
        for dgp_name, n_obs, p in policy_spec.stable_partial_designs
    )
    coverage_anchor_contract = _canonical_window_contract(
        random_state=int(coverage_anchor_random_state),
        binding_design=binding_design,
        designs=designs,
        window_grid=_CANONICAL_WINDOW_GRID,
        n_boot=_CANONICAL_N_BOOT,
    )
    overshoot_companion_contract = _canonical_window_contract(
        random_state=int(overshoot_companion_random_state),
        binding_design=binding_design,
        designs=designs,
        window_grid=_CANONICAL_WINDOW_GRID,
        n_boot=_CANONICAL_N_BOOT,
    )
    return build_phase7_monte_carlo_widening_policy_seed_window_covariance_report(
        coverage_anchor_contract=coverage_anchor_contract,
        overshoot_companion_contract=overshoot_companion_contract,
        binding_design=binding_design,
        window_label=_CANONICAL_WINDOW_LABEL,
        policy=policy,
    )
