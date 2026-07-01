from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Section6SampleRoster:
    treated_states: tuple[str, ...]
    control_states: tuple[str, ...]
    excluded_states: tuple[str, ...]
    baseline_universe: str
    treatment_definition: str
    control_definition: str

    @property
    def treated_state_count(self) -> int:
        return len(self.treated_states)

    @property
    def sample_rule(self) -> dict[str, object]:
        return {
            "baseline_universe": self.baseline_universe,
            "excluded_states": list(self.excluded_states),
            "treatment_definition": self.treatment_definition,
            "control_definition": self.control_definition,
        }


_SECTION6_SAMPLE_ROSTER = Section6SampleRoster(
    treated_states=(
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
    ),
    control_states=(
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
    ),
    excluded_states=("New Hampshire", "Pennsylvania"),
    baseline_universe="2006 federal-binding states",
    treatment_definition="state minimum wage increased by Q1 2007",
    control_definition=(
        "state minimum wage did not increase until the federal increase in July 2007"
    ),
)


def get_section6_sample_roster() -> Section6SampleRoster:
    return _SECTION6_SAMPLE_ROSTER


def classify_section6_sample_role(state: str) -> str:
    roster = _SECTION6_SAMPLE_ROSTER
    if state in roster.treated_states:
        return "treated"
    if state in roster.control_states:
        return "control"
    if state in roster.excluded_states:
        return "excluded"
    return "out-of-universe"
