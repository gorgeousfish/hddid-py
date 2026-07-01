from __future__ import annotations

import importlib

import pytest


TREATED_STATES = (
    "Arizona",
    "Arkansas",
    "Colorado",
    "Maryland",
    "Michigan",
    "Missouri",
    "Montana",
    "Nevada",
    "North Carolina",
    "Ohio",
    "West Virginia",
)
CONTROL_STATES = (
    "Alabama",
    "Georgia",
    "Idaho",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Mississippi",
    "Nebraska",
    "New Mexico",
    "North Dakota",
    "Oklahoma",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Virginia",
    "Wyoming",
)
SAMPLE_RULE = {
    "baseline_universe": "2006 federal-binding states",
    "excluded_states": ["New Hampshire", "Pennsylvania"],
    "treatment_definition": "state minimum wage increased by Q1 2007",
    "control_definition": (
        "state minimum wage did not increase until the federal increase in July 2007"
    ),
}


def _load_module():
    try:
        return importlib.import_module("hddid.section6_roster")
    except ModuleNotFoundError as exc:
        pytest.fail(f"hddid.section6_roster is missing: {exc}")


def test_section6_roster_helper_returns_paper_backed_sample_contract() -> None:
    roster_module = _load_module()

    roster = roster_module.get_section6_sample_roster()

    assert roster.treated_states == TREATED_STATES
    assert roster.control_states == CONTROL_STATES
    assert roster.excluded_states == ("New Hampshire", "Pennsylvania")
    assert roster.treated_state_count == 11
    assert roster.sample_rule == SAMPLE_RULE


def test_classify_section6_sample_role_separates_roster_and_out_of_universe() -> None:
    roster_module = _load_module()

    assert roster_module.classify_section6_sample_role("Arizona") == "treated"
    assert roster_module.classify_section6_sample_role("Texas") == "control"
    assert roster_module.classify_section6_sample_role("Pennsylvania") == "excluded"
    assert (
        roster_module.classify_section6_sample_role("California") == "out-of-universe"
    )
