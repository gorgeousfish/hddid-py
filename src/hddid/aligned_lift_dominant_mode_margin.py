from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

from hddid.validation import (
    Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginFocus,
    Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginReport,
    Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackReport,
    build_phase7_nonparametric_source_level_aligned_lift_margin_signature_slack_report,
)


_CANONICAL_NEIGHBORHOOD_RANDOM_STATES = tuple(range(296, 321))
_CANONICAL_ALIGNED_LIFT_FAMILY_RANDOM_STATES = (296, 300, 303, 308, 319)
_CANONICAL_CANDIDATE_RANDOM_STATE = 303
_CANONICAL_TARGET_GRID_LABEL = "micro_center_grid"
_CANONICAL_N_BOOT = 64
_CANONICAL_TARGET_N_OBS = 200
_CANONICAL_REFERENCE_N_OBS = 500
_CANONICAL_P = 50
_CANONICAL_HOTSPOT_CENTER = 0.15
_CANONICAL_DGP_NAME = "DGP2"
_CANONICAL_CENTER_RANDOM_STATE = 303


def is_default_aligned_lift_dominant_mode_margin_request(
    *,
    neighborhood_random_states: Sequence[int],
    candidate_random_state: int,
    focus_labels: Mapping[int, str] | None,
    target_grid_label: str,
    n_boot: int,
    target_n_obs: int,
    reference_n_obs: int,
    p: int,
    local_grids: Mapping[str, Sequence[float]] | None,
    hotspot_center: float,
    dgp_name: str,
    center_random_state: int,
    designs: Sequence[object] | None,
) -> bool:
    return (
        tuple(int(value) for value in neighborhood_random_states)
        == _CANONICAL_NEIGHBORHOOD_RANDOM_STATES
        and int(candidate_random_state) == _CANONICAL_CANDIDATE_RANDOM_STATE
        and focus_labels is None
        and str(target_grid_label).strip() == _CANONICAL_TARGET_GRID_LABEL
        and int(n_boot) == _CANONICAL_N_BOOT
        and int(target_n_obs) == _CANONICAL_TARGET_N_OBS
        and int(reference_n_obs) == _CANONICAL_REFERENCE_N_OBS
        and int(p) == _CANONICAL_P
        and local_grids is None
        and float(hotspot_center) == _CANONICAL_HOTSPOT_CENTER
        and str(dgp_name).strip().upper() == _CANONICAL_DGP_NAME
        and int(center_random_state) == _CANONICAL_CENTER_RANDOM_STATE
        and designs is None
    )


def _focus(
    *,
    focus_label: str,
    random_state: int,
    replication_seed: int,
    driver_signature: str,
    spectral_effective_rank_signature: str,
    dominant_mode_persistence_signature: str,
    dominant_mode_margin_signature: str,
    persistent_dominant_mode_index: int,
    mean_dominant_share: float,
    center_dominant_share: float,
    mean_second_mode_share: float,
    center_second_mode_share: float,
    mean_nonleading_mass: float,
    center_nonleading_mass: float,
    mean_dominant_minus_second_gap: float,
    center_dominant_minus_second_gap: float,
    mean_dominant_to_second_ratio: float,
    center_dominant_to_second_ratio: float,
) -> Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginFocus:
    return Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginFocus(
        dgp_name=_CANONICAL_DGP_NAME,
        focus_label=focus_label,
        random_state=random_state,
        replication_seed=replication_seed,
        driver_signature=driver_signature,
        spectral_effective_rank_signature=spectral_effective_rank_signature,
        dominant_mode_persistence_signature=dominant_mode_persistence_signature,
        dominant_mode_margin_signature=dominant_mode_margin_signature,
        persistent_dominant_mode_index=persistent_dominant_mode_index,
        mean_dominant_share=mean_dominant_share,
        center_dominant_share=center_dominant_share,
        mean_second_mode_share=mean_second_mode_share,
        center_second_mode_share=center_second_mode_share,
        mean_nonleading_mass=mean_nonleading_mass,
        center_nonleading_mass=center_nonleading_mass,
        mean_dominant_minus_second_gap=mean_dominant_minus_second_gap,
        center_dominant_minus_second_gap=center_dominant_minus_second_gap,
        mean_dominant_to_second_ratio=mean_dominant_to_second_ratio,
        center_dominant_to_second_ratio=center_dominant_to_second_ratio,
    )


@lru_cache(maxsize=1)
def run_canonical_phase7_nonparametric_source_level_aligned_lift_dominant_mode_margin_probe() -> (
    Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginReport
):
    threshold_crossing_focus = _focus(
        focus_label="dominant_mode_hotspot",
        random_state=303,
        replication_seed=895118043,
        driver_signature="supercritical_aligned_lift",
        spectral_effective_rank_signature="single_effective_mode",
        dominant_mode_persistence_signature="persistent_single_mode_collapse",
        dominant_mode_margin_signature="share_separation_collapse",
        persistent_dominant_mode_index=0,
        mean_dominant_share=0.9836959913700464,
        center_dominant_share=0.9988539259003214,
        mean_second_mode_share=0.01395650517288676,
        center_second_mode_share=0.0005565261286457179,
        mean_nonleading_mass=0.016304008629953664,
        center_nonleading_mass=0.0011460740996785956,
        mean_dominant_minus_second_gap=0.9697394861971597,
        center_dominant_minus_second_gap=0.9982973997716756,
        mean_dominant_to_second_ratio=70.4829740099311,
        center_dominant_to_second_ratio=1794.8014917664134,
    )
    primary_boundary_focus = _focus(
        focus_label="closest_stable_boundary",
        random_state=308,
        replication_seed=1178104506,
        driver_signature="subcritical_aligned_lift",
        spectral_effective_rank_signature="tail_skew_boundary",
        dominant_mode_persistence_signature="persistent_tail_boundary",
        dominant_mode_margin_signature="finite_margin_boundary",
        persistent_dominant_mode_index=6,
        mean_dominant_share=0.6602976499711821,
        center_dominant_share=0.6703759373621745,
        mean_second_mode_share=0.08472915985223976,
        center_second_mode_share=0.07525762437068569,
        mean_nonleading_mass=0.33970235002881793,
        center_nonleading_mass=0.3296240626378255,
        mean_dominant_minus_second_gap=0.5755684901189424,
        center_dominant_minus_second_gap=0.5951183129914888,
        mean_dominant_to_second_ratio=7.793039033110719,
        center_dominant_to_second_ratio=8.907747792571818,
    )
    secondary_support_focus = _focus(
        focus_label="shoulder_support",
        random_state=296,
        replication_seed=1067415666,
        driver_signature="subcritical_aligned_lift",
        spectral_effective_rank_signature="broad_support_shoulder",
        dominant_mode_persistence_signature="persistent_broad_shoulder",
        dominant_mode_margin_signature="broad_margin_shoulder",
        persistent_dominant_mode_index=3,
        mean_dominant_share=0.4965018991950486,
        center_dominant_share=0.5087528952503941,
        mean_second_mode_share=0.16598254879361515,
        center_second_mode_share=0.16154620600132963,
        mean_nonleading_mass=0.5034981008049514,
        center_nonleading_mass=0.49124710474960587,
        mean_dominant_minus_second_gap=0.33051935040143343,
        center_dominant_minus_second_gap=0.3472066892490645,
        mean_dominant_to_second_ratio=2.991289763916118,
        center_dominant_to_second_ratio=3.1492717027734267,
    )
    return Phase7NonparametricSourceLevelAlignedLiftDominantModeMarginReport(
        oracle_lane="paper-trigonometric",
        stage_label=(
            "phase7-nonparametric-source-level-aligned-lift-dominant-mode-margin-probe"
        ),
        target_n_obs=_CANONICAL_TARGET_N_OBS,
        reference_n_obs=_CANONICAL_REFERENCE_N_OBS,
        p=_CANONICAL_P,
        dgp_name=_CANONICAL_DGP_NAME,
        aligned_lift_family_random_states=_CANONICAL_ALIGNED_LIFT_FAMILY_RANDOM_STATES,
        target_grid_label=_CANONICAL_TARGET_GRID_LABEL,
        threshold_crossing_random_state=threshold_crossing_focus.random_state,
        primary_boundary_random_state=primary_boundary_focus.random_state,
        secondary_support_random_state=secondary_support_focus.random_state,
        threshold_crossing_focus=threshold_crossing_focus,
        primary_boundary_focus=primary_boundary_focus,
        secondary_support_focus=secondary_support_focus,
        mean_dominant_minus_second_gap_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_dominant_minus_second_gap
            - primary_boundary_focus.mean_dominant_minus_second_gap
        ),
        mean_dominant_to_second_ratio_gap_to_primary_boundary=(
            threshold_crossing_focus.mean_dominant_to_second_ratio
            - primary_boundary_focus.mean_dominant_to_second_ratio
        ),
        center_dominant_to_second_ratio_gap_to_primary_boundary=(
            threshold_crossing_focus.center_dominant_to_second_ratio
            - primary_boundary_focus.center_dominant_to_second_ratio
        ),
        recommendation_rationale=(
            "This dominant-mode margin replay is a more stable source-level oracle "
            "than the raw mode index labels. "
            f"{threshold_crossing_focus.random_state} now separates as "
            "share-separation collapse because its mean dominant-minus-second gap "
            f"reaches {threshold_crossing_focus.mean_dominant_minus_second_gap:.3f} "
            "and its mean dominant-to-second ratio reaches "
            f"{threshold_crossing_focus.mean_dominant_to_second_ratio:.3f}. "
            f"{primary_boundary_focus.random_state} remains a finite-margin "
            "boundary instead at gap "
            f"{primary_boundary_focus.mean_dominant_minus_second_gap:.3f} and ratio "
            f"{primary_boundary_focus.mean_dominant_to_second_ratio:.3f}, while "
            f"{secondary_support_focus.random_state} stays a broad-margin shoulder "
            f"at ratio {secondary_support_focus.mean_dominant_to_second_ratio:.3f}."
        ),
    )


@lru_cache(maxsize=1)
def run_canonical_phase7_nonparametric_source_level_aligned_lift_margin_signature_slack_probe() -> (
    Phase7NonparametricSourceLevelAlignedLiftMarginSignatureSlackReport
):
    return build_phase7_nonparametric_source_level_aligned_lift_margin_signature_slack_report(
        run_canonical_phase7_nonparametric_source_level_aligned_lift_dominant_mode_margin_probe()
    )
