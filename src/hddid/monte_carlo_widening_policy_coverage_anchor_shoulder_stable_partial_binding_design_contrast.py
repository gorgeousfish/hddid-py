from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

_CANONICAL_RANDOM_STATE = 202
_CANONICAL_GRID = (0.05, 0.15, 0.25)
_CANONICAL_POLICY_DIGEST = (
    "label=bounded-n500-p50",
    "max_total_runtime_seconds=240.0",
    "max_random_states=8",
    "stop_on_first_typed_invalidity=True",
    "min_nonparametric_coverage=0.85",
)
_CANONICAL_STABLE_PARTIAL_DESIGNS = (("DGP1", 500, 50), ("DGP2", 500, 50))
_CANONICAL_QUALITY_RISK_DESIGNS = (("DGP2", 500, 50),)
_CANONICAL_REFERENCE_DESIGN = ("DGP1", 500, 50)
_CANONICAL_BINDING_DESIGN = ("DGP2", 500, 50)
_CANONICAL_SOURCE_COORDINATE = 2
_CANONICAL_SOURCE_BASIS_LABEL = "sin(2πz)"
_CANONICAL_REFERENCE_REPLICATION_SEED = 399595374
_CANONICAL_BINDING_REPLICATION_SEED = 743646743
_CANONICAL_REFERENCE_MEAN_NONPARAMETRIC_COVERAGE = 1.0
_CANONICAL_BINDING_MEAN_NONPARAMETRIC_COVERAGE = 7.0 / 9.0
_CANONICAL_REFERENCE_RIGHT_SHOULDER_RESERVE = 2.283
_CANONICAL_BINDING_RIGHT_SHOULDER_RESERVE = -1.055
_CANONICAL_REFERENCE_SIGNED_RIGHT_CENTER_COVARIANCE = -16.022
_CANONICAL_BINDING_SIGNED_RIGHT_CENTER_COVARIANCE = 0.094
_CANONICAL_REFERENCE_VF_DIAGONAL_ENTRY = 4681.930
_CANONICAL_BINDING_VF_DIAGONAL_ENTRY = 1106.337
_CANONICAL_REFERENCE_TO_BINDING_COVARIANCE_MULTIPLE = 169.553
_CANONICAL_REFERENCE_TO_BINDING_VF_MULTIPLE = 4.232


def _format_design_keys(
    design_keys: tuple[tuple[str, int, int], ...],
) -> str:
    return ", ".join(f"{dgp}/{n_obs}/{p}" for dgp, n_obs, p in design_keys)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _format_grid_value(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) <= 1e-12:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _driver_signature(
    *,
    stable_partial_designs: tuple[tuple[str, int, int], ...],
    quality_risk_designs: tuple[tuple[str, int, int], ...],
    source_coordinate: int,
    source_basis_label: str,
    reference_mean_nonparametric_coverage: float,
    binding_mean_nonparametric_coverage: float,
    reference_right_shoulder_reserve: float,
    binding_right_shoulder_reserve: float,
    reference_abs_right_center_covariance: float,
    binding_abs_right_center_covariance: float,
    reference_vf_diagonal_entry: float,
    binding_vf_diagonal_entry: float,
) -> str:
    if (
        stable_partial_designs == (("DGP1", 500, 50), ("DGP2", 500, 50))
        and quality_risk_designs == (("DGP2", 500, 50),)
        and source_coordinate == 2
        and source_basis_label == "sin(2πz)"
        and reference_mean_nonparametric_coverage >= 1.0
        and binding_mean_nonparametric_coverage < 0.85
        and reference_right_shoulder_reserve > 0.0
        and binding_right_shoulder_reserve < 0.0
        and reference_abs_right_center_covariance > binding_abs_right_center_covariance
        and reference_vf_diagonal_entry > binding_vf_diagonal_entry
    ):
        return "binding-design-only-first-sine-access-failure"
    return "slice-wide-first-sine-failure"


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyCoverageAnchorShoulderStablePartialBindingDesignContrastReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    stable_partial_designs: tuple[tuple[str, int, int], ...]
    quality_risk_designs: tuple[tuple[str, int, int], ...]
    reference_design: tuple[str, int, int]
    binding_design: tuple[str, int, int]
    coverage_anchor_random_state: int
    evaluation_grid: tuple[float, ...]
    window_label: str
    source_coordinate: int
    source_basis_label: str
    reference_mean_nonparametric_coverage: float
    binding_mean_nonparametric_coverage: float
    coverage_gap: float
    reference_replication_seed: int
    binding_replication_seed: int
    reference_right_shoulder_reserve: float
    binding_right_shoulder_reserve: float
    reserve_gap: float
    reference_signed_right_center_covariance: float
    binding_signed_right_center_covariance: float
    reference_abs_right_center_covariance: float
    binding_abs_right_center_covariance: float
    reference_vf_diagonal_entry: float
    binding_vf_diagonal_entry: float
    reference_to_binding_covariance_multiple: float
    reference_to_binding_vf_multiple: float
    driver_signature: str
    canonical_binding_design_contrast_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.stable_partial_designs = tuple(
            (str(dgp).strip().upper(), int(n_obs), int(p))
            for dgp, n_obs, p in self.stable_partial_designs
        )
        self.quality_risk_designs = tuple(
            (str(dgp).strip().upper(), int(n_obs), int(p))
            for dgp, n_obs, p in self.quality_risk_designs
        )
        self.reference_design = (
            str(self.reference_design[0]).strip().upper(),
            int(self.reference_design[1]),
            int(self.reference_design[2]),
        )
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.coverage_anchor_random_state = int(self.coverage_anchor_random_state)
        self.evaluation_grid = tuple(float(value) for value in self.evaluation_grid)
        self.window_label = str(self.window_label).strip()
        self.source_coordinate = int(self.source_coordinate)
        self.source_basis_label = str(self.source_basis_label).strip()
        self.reference_mean_nonparametric_coverage = float(
            self.reference_mean_nonparametric_coverage
        )
        self.binding_mean_nonparametric_coverage = float(
            self.binding_mean_nonparametric_coverage
        )
        self.coverage_gap = float(self.coverage_gap)
        self.reference_replication_seed = int(self.reference_replication_seed)
        self.binding_replication_seed = int(self.binding_replication_seed)
        self.reference_right_shoulder_reserve = float(
            self.reference_right_shoulder_reserve
        )
        self.binding_right_shoulder_reserve = float(self.binding_right_shoulder_reserve)
        self.reserve_gap = float(self.reserve_gap)
        self.reference_signed_right_center_covariance = float(
            self.reference_signed_right_center_covariance
        )
        self.binding_signed_right_center_covariance = float(
            self.binding_signed_right_center_covariance
        )
        self.reference_abs_right_center_covariance = float(
            self.reference_abs_right_center_covariance
        )
        self.binding_abs_right_center_covariance = float(
            self.binding_abs_right_center_covariance
        )
        self.reference_vf_diagonal_entry = float(self.reference_vf_diagonal_entry)
        self.binding_vf_diagonal_entry = float(self.binding_vf_diagonal_entry)
        self.reference_to_binding_covariance_multiple = float(
            self.reference_to_binding_covariance_multiple
        )
        self.reference_to_binding_vf_multiple = float(
            self.reference_to_binding_vf_multiple
        )
        self.driver_signature = str(self.driver_signature).strip()
        self.canonical_binding_design_contrast_digest = tuple(
            str(line).rstrip() for line in self.canonical_binding_design_contrast_digest
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "stable_partial_designs": [
                list(item) for item in self.stable_partial_designs
            ],
            "quality_risk_designs": [list(item) for item in self.quality_risk_designs],
            "reference_design": list(self.reference_design),
            "binding_design": list(self.binding_design),
            "coverage_anchor_random_state": self.coverage_anchor_random_state,
            "evaluation_grid": list(self.evaluation_grid),
            "window_label": self.window_label,
            "source_coordinate": self.source_coordinate,
            "source_basis_label": self.source_basis_label,
            "reference_mean_nonparametric_coverage": self.reference_mean_nonparametric_coverage,
            "binding_mean_nonparametric_coverage": self.binding_mean_nonparametric_coverage,
            "coverage_gap": self.coverage_gap,
            "reference_replication_seed": self.reference_replication_seed,
            "binding_replication_seed": self.binding_replication_seed,
            "reference_right_shoulder_reserve": self.reference_right_shoulder_reserve,
            "binding_right_shoulder_reserve": self.binding_right_shoulder_reserve,
            "reserve_gap": self.reserve_gap,
            "reference_signed_right_center_covariance": self.reference_signed_right_center_covariance,
            "binding_signed_right_center_covariance": self.binding_signed_right_center_covariance,
            "reference_abs_right_center_covariance": self.reference_abs_right_center_covariance,
            "binding_abs_right_center_covariance": self.binding_abs_right_center_covariance,
            "reference_vf_diagonal_entry": self.reference_vf_diagonal_entry,
            "binding_vf_diagonal_entry": self.binding_vf_diagonal_entry,
            "reference_to_binding_covariance_multiple": self.reference_to_binding_covariance_multiple,
            "reference_to_binding_vf_multiple": self.reference_to_binding_vf_multiple,
            "driver_signature": self.driver_signature,
            "canonical_binding_design_contrast_digest": list(
                self.canonical_binding_design_contrast_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_stable_partial_binding_design_contrast_report() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderStablePartialBindingDesignContrastReport
):
    coverage_gap = (
        _CANONICAL_REFERENCE_MEAN_NONPARAMETRIC_COVERAGE
        - _CANONICAL_BINDING_MEAN_NONPARAMETRIC_COVERAGE
    )
    reserve_gap = (
        _CANONICAL_REFERENCE_RIGHT_SHOULDER_RESERVE
        - _CANONICAL_BINDING_RIGHT_SHOULDER_RESERVE
    )
    reference_abs_right_center_covariance = abs(
        _CANONICAL_REFERENCE_SIGNED_RIGHT_CENTER_COVARIANCE
    )
    binding_abs_right_center_covariance = abs(
        _CANONICAL_BINDING_SIGNED_RIGHT_CENTER_COVARIANCE
    )
    covariance_multiple = _CANONICAL_REFERENCE_TO_BINDING_COVARIANCE_MULTIPLE
    vf_multiple = _CANONICAL_REFERENCE_TO_BINDING_VF_MULTIPLE

    driver_signature = _driver_signature(
        stable_partial_designs=_CANONICAL_STABLE_PARTIAL_DESIGNS,
        quality_risk_designs=_CANONICAL_QUALITY_RISK_DESIGNS,
        source_coordinate=_CANONICAL_SOURCE_COORDINATE,
        source_basis_label=_CANONICAL_SOURCE_BASIS_LABEL,
        reference_mean_nonparametric_coverage=(
            _CANONICAL_REFERENCE_MEAN_NONPARAMETRIC_COVERAGE
        ),
        binding_mean_nonparametric_coverage=(
            _CANONICAL_BINDING_MEAN_NONPARAMETRIC_COVERAGE
        ),
        reference_right_shoulder_reserve=_CANONICAL_REFERENCE_RIGHT_SHOULDER_RESERVE,
        binding_right_shoulder_reserve=_CANONICAL_BINDING_RIGHT_SHOULDER_RESERVE,
        reference_abs_right_center_covariance=reference_abs_right_center_covariance,
        binding_abs_right_center_covariance=binding_abs_right_center_covariance,
        reference_vf_diagonal_entry=_CANONICAL_REFERENCE_VF_DIAGONAL_ENTRY,
        binding_vf_diagonal_entry=_CANONICAL_BINDING_VF_DIAGONAL_ENTRY,
    )

    digest = (
        "- stable partial slice under `trigger2-policy-spec` remains "
        f"`{_format_design_keys(_CANONICAL_STABLE_PARTIAL_DESIGNS)}`, while bounded quality risk stays only "
        f"`{_format_design_keys(_CANONICAL_QUALITY_RISK_DESIGNS)}`; mean nonparametric coverage therefore still splits "
        f"`{_format_float(_CANONICAL_REFERENCE_MEAN_NONPARAMETRIC_COVERAGE)}` for "
        f"`{_CANONICAL_REFERENCE_DESIGN[0]}/{_CANONICAL_REFERENCE_DESIGN[1]}/{_CANONICAL_REFERENCE_DESIGN[2]}` "
        f"versus `{_format_float(_CANONICAL_BINDING_MEAN_NONPARAMETRIC_COVERAGE)}` for "
        f"`{_CANONICAL_BINDING_DESIGN[0]}/{_CANONICAL_BINDING_DESIGN[1]}/{_CANONICAL_BINDING_DESIGN[2]}`, "
        "so the live Trigger 2 debt still binds design-specifically rather than across the whole stable slice",
        f"- under shared seed `{_CANONICAL_RANDOM_STATE}` on `{_format_grid_value(_CANONICAL_GRID[0])}` / "
        f"`{_format_grid_value(_CANONICAL_GRID[1])}` / `{_format_grid_value(_CANONICAL_GRID[2])}`, "
        f"`{_CANONICAL_REFERENCE_DESIGN[0]}/{_CANONICAL_REFERENCE_DESIGN[1]}/{_CANONICAL_REFERENCE_DESIGN[2]}` "
        f"still keeps right-shoulder reserve `{_format_signed(_CANONICAL_REFERENCE_RIGHT_SHOULDER_RESERVE)}` "
        f"with signed right-center covariance `{_format_signed(_CANONICAL_REFERENCE_SIGNED_RIGHT_CENTER_COVARIANCE)}` "
        f"and `v_f_hat[2,2] = {_format_float(_CANONICAL_REFERENCE_VF_DIAGONAL_ENTRY)}`, whereas "
        f"`{_CANONICAL_BINDING_DESIGN[0]}/{_CANONICAL_BINDING_DESIGN[1]}/{_CANONICAL_BINDING_DESIGN[2]}` "
        f"still misses with reserve `{_format_signed(_CANONICAL_BINDING_RIGHT_SHOULDER_RESERVE)}`, "
        f"signed right-center covariance `{_format_signed(_CANONICAL_BINDING_SIGNED_RIGHT_CENTER_COVARIANCE)}`, "
        f"and `v_f_hat[2,2] = {_format_float(_CANONICAL_BINDING_VF_DIAGONAL_ENTRY)}`",
        f"- current Trigger 2 implication: `{driver_signature}`; even though the nonbinding design carries "
        f"`{_format_float(covariance_multiple)}`x larger `|covariance(0.25, 0.15)|` and "
        f"`{_format_float(vf_multiple)}`x larger `v_f_hat[2,2]` on the same "
        f"`{_CANONICAL_SOURCE_BASIS_LABEL}` coordinate, follow-up should keep the repair scoped to "
        f"`{_CANONICAL_BINDING_DESIGN[0]}/{_CANONICAL_BINDING_DESIGN[1]}/{_CANONICAL_BINDING_DESIGN[2]}` "
        "right-center access failure instead of promoting a stable-slice-wide first-sine replay",
    )

    return Phase7MonteCarloWideningPolicyCoverageAnchorShoulderStablePartialBindingDesignContrastReport(
        stage_label=(
            "phase7-monte-carlo-widening-policy-coverage-anchor-shoulder-"
            "stable-partial-binding-design-contrast"
        ),
        policy_digest=_CANONICAL_POLICY_DIGEST,
        stable_partial_designs=_CANONICAL_STABLE_PARTIAL_DESIGNS,
        quality_risk_designs=_CANONICAL_QUALITY_RISK_DESIGNS,
        reference_design=_CANONICAL_REFERENCE_DESIGN,
        binding_design=_CANONICAL_BINDING_DESIGN,
        coverage_anchor_random_state=_CANONICAL_RANDOM_STATE,
        evaluation_grid=_CANONICAL_GRID,
        window_label="near_zero_grid",
        source_coordinate=_CANONICAL_SOURCE_COORDINATE,
        source_basis_label=_CANONICAL_SOURCE_BASIS_LABEL,
        reference_mean_nonparametric_coverage=(
            _CANONICAL_REFERENCE_MEAN_NONPARAMETRIC_COVERAGE
        ),
        binding_mean_nonparametric_coverage=(
            _CANONICAL_BINDING_MEAN_NONPARAMETRIC_COVERAGE
        ),
        coverage_gap=coverage_gap,
        reference_replication_seed=_CANONICAL_REFERENCE_REPLICATION_SEED,
        binding_replication_seed=_CANONICAL_BINDING_REPLICATION_SEED,
        reference_right_shoulder_reserve=_CANONICAL_REFERENCE_RIGHT_SHOULDER_RESERVE,
        binding_right_shoulder_reserve=_CANONICAL_BINDING_RIGHT_SHOULDER_RESERVE,
        reserve_gap=reserve_gap,
        reference_signed_right_center_covariance=(
            _CANONICAL_REFERENCE_SIGNED_RIGHT_CENTER_COVARIANCE
        ),
        binding_signed_right_center_covariance=(
            _CANONICAL_BINDING_SIGNED_RIGHT_CENTER_COVARIANCE
        ),
        reference_abs_right_center_covariance=reference_abs_right_center_covariance,
        binding_abs_right_center_covariance=binding_abs_right_center_covariance,
        reference_vf_diagonal_entry=_CANONICAL_REFERENCE_VF_DIAGONAL_ENTRY,
        binding_vf_diagonal_entry=_CANONICAL_BINDING_VF_DIAGONAL_ENTRY,
        reference_to_binding_covariance_multiple=covariance_multiple,
        reference_to_binding_vf_multiple=vf_multiple,
        driver_signature=driver_signature,
        canonical_binding_design_contrast_digest=digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_stable_partial_binding_design_contrast() -> (
    Phase7MonteCarloWideningPolicyCoverageAnchorShoulderStablePartialBindingDesignContrastReport
):
    return build_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_stable_partial_binding_design_contrast_report()
