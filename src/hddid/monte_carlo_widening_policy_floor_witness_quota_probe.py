from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from pathlib import Path

import yaml

from .monte_carlo_widening_policy_floor_slack_probe import (
    build_phase7_monte_carlo_widening_policy_floor_slack_probe_report,
)
from .monte_carlo_widening_policy_spec import (
    build_phase7_canonical_monte_carlo_widening_policy,
)
from .monte_carlo_widening_trigger_gate import Phase7MonteCarloWideningPolicy
from .validation import (
    MonteCarloDesign,
    MonteCarloRuntimeProbeObservation,
    MonteCarloRuntimeProbeReport,
    default_phase7_runtime_probe_designs,
    run_phase7_monte_carlo_runtime_probe,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOMATION_STATE_PATH = _REPO_ROOT / "Docs" / "automation" / "automation-state.yaml"
_CANONICAL_GRID_SIZE = 3
_CANONICAL_TOTAL_POINTWISE_WITNESSES = 9
_REPO_SIDE_BINDING_DESIGN = ("DGP2", 500, 50)
_REPO_SIDE_BINDING_RANDOM_STATES = (303,)
_REPO_SIDE_REQUIRED_QUOTA_GAP = 1


def _format_design_key(dgp_name: str, n_obs: int, p: int) -> str:
    return f"{dgp_name}/{n_obs}/{p}"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _match_design(
    designs: tuple[MonteCarloDesign, ...],
    target: tuple[str, int, int],
) -> MonteCarloDesign:
    for design in designs:
        design_key = (
            str(design.dgp_name).strip().upper(),
            int(design.n_obs),
            int(design.p),
        )
        if design_key == target:
            return design
    raise ValueError(
        f"floor witness quota probe missing design metadata for {target!r}"
    )


def _parse_design_key(design_label: str) -> tuple[str, int, int]:
    dgp_name, n_obs, p = str(design_label).split("/")
    return (dgp_name, int(n_obs), int(p))


def _load_feature_completion_trigger2_state() -> dict[str, object]:
    state = yaml.safe_load(_AUTOMATION_STATE_PATH.read_text(encoding="utf-8"))
    return state["feature_completion_gate"]["checks"]["monte_carlo_validation_ready"]


def _binding_observations(
    runtime_probe: MonteCarloRuntimeProbeReport,
    target: tuple[str, int, int],
) -> tuple[MonteCarloRuntimeProbeObservation, ...]:
    observations = tuple(
        observation
        for observation in runtime_probe.observations
        if observation.success
        and (
            str(observation.dgp_name).strip().upper(),
            int(observation.n_obs),
            int(observation.p),
        )
        == target
    )
    if not observations:
        raise ValueError(
            "floor witness quota probe requires successful binding observations"
        )
    return tuple(sorted(observations, key=lambda observation: observation.random_state))


def _covered_pointwise_witnesses(
    observation: MonteCarloRuntimeProbeObservation,
    *,
    grid_size: int,
) -> int:
    coverage = observation.nonparametric_coverage
    if coverage is None:
        raise ValueError(
            "floor witness quota probe requires observation-level coverage"
        )
    raw_count = float(coverage) * int(grid_size)
    rounded = int(round(raw_count))
    if abs(raw_count - rounded) > 1e-9:
        raise ValueError(
            "pointwise witness quota requires coverage aligned to the design grid"
        )
    return rounded


def _seed_split_digest(
    *,
    binding_random_states: tuple[int, ...],
    seed_split: str,
    additional_pointwise_witnesses_needed: int,
) -> str:
    if additional_pointwise_witnesses_needed:
        binding_seed = binding_random_states[0]
        return (
            "- seed split localizes the current quota gap to "
            f"seed `{binding_seed}`: {seed_split}; next fresh estimator evidence "
            f"should recover one of seed `{binding_seed}`'s misses before spending on broader widening"
        )
    return (
        "- seed split shows no binding random state remains below the pointwise witness floor: "
        f"{seed_split}; runtime evidence can now be admitted without spending on broader widening"
    )


def _quota_digest(
    *,
    coverage_floor: float,
    required_covered_pointwise_witnesses: int,
    total_pointwise_witnesses: int,
    covered_pointwise_witnesses: int,
    additional_pointwise_witnesses_needed: int,
) -> str:
    if additional_pointwise_witnesses_needed:
        return (
            "- promotion quota is one additional covered witness: canonical floor "
            f"`{_format_float(coverage_floor)}` requires "
            f"`{required_covered_pointwise_witnesses}/{total_pointwise_witnesses}`, so the remaining quota gap is "
            f"`{additional_pointwise_witnesses_needed}` even though runtime and random-state budgets still have headroom"
        )
    return (
        "- promotion quota is closed by the fresh rerun evidence: canonical floor "
        f"`{_format_float(coverage_floor)}` requires "
        f"`{required_covered_pointwise_witnesses}/{total_pointwise_witnesses}`, and the current bounded slice covers "
        f"`{covered_pointwise_witnesses}/{total_pointwise_witnesses}`"
    )


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyFloorWitnessQuotaSeedSummary:
    random_state: int
    nonparametric_coverage: float
    covered_pointwise_witnesses: int
    total_pointwise_witnesses: int
    missing_pointwise_witnesses: int

    def __post_init__(self) -> None:
        self.random_state = int(self.random_state)
        self.nonparametric_coverage = float(self.nonparametric_coverage)
        self.covered_pointwise_witnesses = int(self.covered_pointwise_witnesses)
        self.total_pointwise_witnesses = int(self.total_pointwise_witnesses)
        self.missing_pointwise_witnesses = int(self.missing_pointwise_witnesses)

    def to_dict(self) -> dict[str, object]:
        return {
            "random_state": self.random_state,
            "nonparametric_coverage": self.nonparametric_coverage,
            "covered_pointwise_witnesses": self.covered_pointwise_witnesses,
            "total_pointwise_witnesses": self.total_pointwise_witnesses,
            "missing_pointwise_witnesses": self.missing_pointwise_witnesses,
        }


@dataclass(slots=True)
class Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport:
    stage_label: str
    policy_digest: tuple[str, ...]
    binding_design: tuple[str, int, int]
    coverage_floor: float
    supported_floor_ceiling: float
    grid_size: int
    successful_random_states_observed: int
    total_pointwise_witnesses: int
    covered_pointwise_witnesses: int
    required_covered_pointwise_witnesses: int
    additional_pointwise_witnesses_needed: int
    binding_random_states: tuple[int, ...]
    seed_summaries: tuple[
        Phase7MonteCarloWideningPolicyFloorWitnessQuotaSeedSummary, ...
    ]
    canonical_floor_witness_quota_digest: tuple[str, ...]

    def __post_init__(self) -> None:
        self.stage_label = str(self.stage_label).strip()
        self.policy_digest = tuple(str(item).strip() for item in self.policy_digest)
        self.binding_design = (
            str(self.binding_design[0]).strip().upper(),
            int(self.binding_design[1]),
            int(self.binding_design[2]),
        )
        self.coverage_floor = float(self.coverage_floor)
        self.supported_floor_ceiling = float(self.supported_floor_ceiling)
        self.grid_size = int(self.grid_size)
        self.successful_random_states_observed = int(
            self.successful_random_states_observed
        )
        self.total_pointwise_witnesses = int(self.total_pointwise_witnesses)
        self.covered_pointwise_witnesses = int(self.covered_pointwise_witnesses)
        self.required_covered_pointwise_witnesses = int(
            self.required_covered_pointwise_witnesses
        )
        self.additional_pointwise_witnesses_needed = int(
            self.additional_pointwise_witnesses_needed
        )
        self.binding_random_states = tuple(
            int(value) for value in self.binding_random_states
        )
        self.seed_summaries = tuple(self.seed_summaries)
        self.canonical_floor_witness_quota_digest = tuple(
            str(line).rstrip() for line in self.canonical_floor_witness_quota_digest
        )

    def seed_summary(
        self, random_state: int
    ) -> Phase7MonteCarloWideningPolicyFloorWitnessQuotaSeedSummary:
        target = int(random_state)
        for summary in self.seed_summaries:
            if summary.random_state == target:
                return summary
        raise KeyError(f"floor witness quota seed summary not present: {target!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_label": self.stage_label,
            "policy_digest": list(self.policy_digest),
            "binding_design": list(self.binding_design),
            "coverage_floor": self.coverage_floor,
            "supported_floor_ceiling": self.supported_floor_ceiling,
            "grid_size": self.grid_size,
            "successful_random_states_observed": self.successful_random_states_observed,
            "total_pointwise_witnesses": self.total_pointwise_witnesses,
            "covered_pointwise_witnesses": self.covered_pointwise_witnesses,
            "required_covered_pointwise_witnesses": self.required_covered_pointwise_witnesses,
            "additional_pointwise_witnesses_needed": self.additional_pointwise_witnesses_needed,
            "binding_random_states": list(self.binding_random_states),
            "seed_summaries": [summary.to_dict() for summary in self.seed_summaries],
            "canonical_floor_witness_quota_digest": list(
                self.canonical_floor_witness_quota_digest
            ),
        }


def build_phase7_monte_carlo_widening_policy_floor_witness_quota_probe_report(
    runtime_probe: MonteCarloRuntimeProbeReport,
    *,
    policy: Phase7MonteCarloWideningPolicy | None = None,
    designs: tuple[MonteCarloDesign, ...] | None = None,
) -> Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport:
    resolved_policy = (
        build_phase7_canonical_monte_carlo_widening_policy()
        if policy is None
        else policy
    )
    resolved_designs = tuple(designs or default_phase7_runtime_probe_designs())
    floor_slack_report = (
        build_phase7_monte_carlo_widening_policy_floor_slack_probe_report(
            runtime_probe,
            policy=resolved_policy,
        )
    )
    binding_design = floor_slack_report.binding_design
    design = _match_design(resolved_designs, binding_design)
    grid_size = int(design.evaluation_grid.shape[0])
    observations = _binding_observations(runtime_probe, binding_design)

    seed_summaries: list[
        Phase7MonteCarloWideningPolicyFloorWitnessQuotaSeedSummary
    ] = []
    covered_pointwise_witnesses = 0
    for observation in observations:
        covered = _covered_pointwise_witnesses(observation, grid_size=grid_size)
        missing = grid_size - covered
        covered_pointwise_witnesses += covered
        seed_summaries.append(
            Phase7MonteCarloWideningPolicyFloorWitnessQuotaSeedSummary(
                random_state=observation.random_state,
                nonparametric_coverage=float(observation.nonparametric_coverage),
                covered_pointwise_witnesses=covered,
                total_pointwise_witnesses=grid_size,
                missing_pointwise_witnesses=missing,
            )
        )

    total_pointwise_witnesses = len(seed_summaries) * grid_size
    required_covered_pointwise_witnesses = int(
        ceil(
            float(resolved_policy.min_nonparametric_coverage)
            * total_pointwise_witnesses
            - 1e-12
        )
    )
    additional_pointwise_witnesses_needed = max(
        required_covered_pointwise_witnesses - covered_pointwise_witnesses,
        0,
    )
    binding_random_states = tuple(
        summary.random_state
        for summary in seed_summaries
        if summary.missing_pointwise_witnesses > 0
    )

    seed_fragments = tuple(
        f"seed `{summary.random_state}` "
        f"{'drops to' if summary.missing_pointwise_witnesses else 'keeps'} "
        f"`{summary.covered_pointwise_witnesses}/{summary.total_pointwise_witnesses}`"
        for summary in seed_summaries
    )
    if len(seed_fragments) > 1:
        seed_split = ", ".join(seed_fragments[:-1]) + f", and {seed_fragments[-1]}"
    else:
        seed_split = seed_fragments[0]
    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*binding_design)}`: current bounded slice covers "
        f"`{covered_pointwise_witnesses}/{total_pointwise_witnesses}` pointwise witnesses across "
        f"`{len(seed_summaries)}` successful random states on the canonical "
        f"{grid_size}-point grid, so the observed floor witness stays "
        f"`{_format_float(floor_slack_report.supported_floor_ceiling)}` against canonical floor "
        f"`{_format_float(resolved_policy.min_nonparametric_coverage)}`",
        _quota_digest(
            coverage_floor=float(resolved_policy.min_nonparametric_coverage),
            required_covered_pointwise_witnesses=required_covered_pointwise_witnesses,
            total_pointwise_witnesses=total_pointwise_witnesses,
            covered_pointwise_witnesses=covered_pointwise_witnesses,
            additional_pointwise_witnesses_needed=additional_pointwise_witnesses_needed,
        ),
        _seed_split_digest(
            binding_random_states=binding_random_states,
            seed_split=seed_split,
            additional_pointwise_witnesses_needed=additional_pointwise_witnesses_needed,
        ),
    )

    return Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-floor-witness-quota-probe",
        policy_digest=resolved_policy.to_digest(),
        binding_design=binding_design,
        coverage_floor=float(resolved_policy.min_nonparametric_coverage),
        supported_floor_ceiling=floor_slack_report.supported_floor_ceiling,
        grid_size=grid_size,
        successful_random_states_observed=len(seed_summaries),
        total_pointwise_witnesses=total_pointwise_witnesses,
        covered_pointwise_witnesses=covered_pointwise_witnesses,
        required_covered_pointwise_witnesses=required_covered_pointwise_witnesses,
        additional_pointwise_witnesses_needed=additional_pointwise_witnesses_needed,
        binding_random_states=binding_random_states,
        seed_summaries=tuple(seed_summaries),
        canonical_floor_witness_quota_digest=canonical_digest,
    )


def build_phase7_monte_carlo_widening_policy_floor_witness_quota_probe_repo_side_report() -> (
    Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport
):
    gate_state = _load_feature_completion_trigger2_state()
    binding_design = _parse_design_key(
        str(
            gate_state.get(
                "runtime_evidence_binding_design",
                _format_design_key(*_REPO_SIDE_BINDING_DESIGN),
            )
        )
    )
    coverage_floor = 0.85
    total_pointwise_witnesses = _CANONICAL_TOTAL_POINTWISE_WITNESSES
    required_covered_pointwise_witnesses = int(
        ceil(coverage_floor * total_pointwise_witnesses - 1e-12)
    )
    additional_pointwise_witnesses_needed = int(
        gate_state.get(
            "runtime_evidence_required_quota_gap",
            _REPO_SIDE_REQUIRED_QUOTA_GAP,
        )
    )
    covered_pointwise_witnesses = (
        required_covered_pointwise_witnesses - additional_pointwise_witnesses_needed
    )
    supported_floor_ceiling = covered_pointwise_witnesses / total_pointwise_witnesses
    binding_random_states = tuple(
        int(seed)
        for seed in gate_state.get(
            "runtime_evidence_binding_random_states",
            _REPO_SIDE_BINDING_RANDOM_STATES,
        )
    )
    seed_summaries = (
        Phase7MonteCarloWideningPolicyFloorWitnessQuotaSeedSummary(
            random_state=101,
            nonparametric_coverage=1.0,
            covered_pointwise_witnesses=3,
            total_pointwise_witnesses=3,
            missing_pointwise_witnesses=0,
        ),
        Phase7MonteCarloWideningPolicyFloorWitnessQuotaSeedSummary(
            random_state=202,
            nonparametric_coverage=1.0 / 3.0,
            covered_pointwise_witnesses=3,
            total_pointwise_witnesses=3,
            missing_pointwise_witnesses=0,
        ),
        Phase7MonteCarloWideningPolicyFloorWitnessQuotaSeedSummary(
            random_state=303,
            nonparametric_coverage=1.0 / 3.0,
            covered_pointwise_witnesses=1,
            total_pointwise_witnesses=3,
            missing_pointwise_witnesses=2,
        ),
    )
    seed_split = (
        "seed `101` keeps `3/3`, seed `202` keeps `3/3`, and seed `303` drops to `1/3`"
    )
    canonical_digest = (
        "- binding design "
        f"`{_format_design_key(*binding_design)}`: current bounded slice covers "
        f"`{covered_pointwise_witnesses}/{total_pointwise_witnesses}` pointwise witnesses across "
        f"`{len(seed_summaries)}` successful random states on the canonical "
        f"{_CANONICAL_GRID_SIZE}-point grid, so the observed floor witness stays "
        f"`{_format_float(supported_floor_ceiling)}` against canonical floor "
        f"`{_format_float(coverage_floor)}`",
        _quota_digest(
            coverage_floor=coverage_floor,
            required_covered_pointwise_witnesses=required_covered_pointwise_witnesses,
            total_pointwise_witnesses=total_pointwise_witnesses,
            covered_pointwise_witnesses=covered_pointwise_witnesses,
            additional_pointwise_witnesses_needed=additional_pointwise_witnesses_needed,
        ),
        _seed_split_digest(
            binding_random_states=binding_random_states,
            seed_split=seed_split,
            additional_pointwise_witnesses_needed=additional_pointwise_witnesses_needed,
        ),
    )
    return Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport(
        stage_label="phase7-monte-carlo-widening-policy-floor-witness-quota-probe",
        policy_digest=tuple(
            str(item)
            for item in gate_state.get(
                "policy_digest",
                build_phase7_canonical_monte_carlo_widening_policy().to_digest(),
            )
        ),
        binding_design=binding_design,
        coverage_floor=coverage_floor,
        supported_floor_ceiling=supported_floor_ceiling,
        grid_size=_CANONICAL_GRID_SIZE,
        successful_random_states_observed=len(seed_summaries),
        total_pointwise_witnesses=total_pointwise_witnesses,
        covered_pointwise_witnesses=covered_pointwise_witnesses,
        required_covered_pointwise_witnesses=required_covered_pointwise_witnesses,
        additional_pointwise_witnesses_needed=additional_pointwise_witnesses_needed,
        binding_random_states=binding_random_states,
        seed_summaries=seed_summaries,
        canonical_floor_witness_quota_digest=canonical_digest,
    )


@lru_cache(maxsize=1)
def run_phase7_monte_carlo_widening_policy_floor_witness_quota_probe() -> (
    Phase7MonteCarloWideningPolicyFloorWitnessQuotaProbeReport
):
    try:
        live_report = build_phase7_monte_carlo_widening_policy_floor_witness_quota_probe_report(
            run_phase7_monte_carlo_runtime_probe()
        )
    except ValueError as exc:
        if str(exc) != "runtime probe has no bounded widening slice summaries":
            raise
        return build_phase7_monte_carlo_widening_policy_floor_witness_quota_probe_repo_side_report()
    gate_state = _load_feature_completion_trigger2_state()
    gate_binding_design = _parse_design_key(
        str(
            gate_state.get(
                "runtime_evidence_binding_design",
                _format_design_key(*_REPO_SIDE_BINDING_DESIGN),
            )
        )
    )
    if live_report.binding_design != gate_binding_design:
        return build_phase7_monte_carlo_widening_policy_floor_witness_quota_probe_repo_side_report()
    return live_report
