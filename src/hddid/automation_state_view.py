from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


def _extract_top_level_block(raw_text: str, key: str) -> str:
    target = f"{key}:"
    lines = raw_text.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if line == target:
            start = index
            break
    if start is None:
        raise KeyError(f"top-level automation-state key not found: {key!r}")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            end = index
            break
    return "\n".join(lines[start:end]) + "\n"


def load_top_level_automation_state_block(
    path: str | Path,
    key: str,
) -> Mapping[str, object]:
    raw_text = Path(path).read_text(encoding="utf-8")
    block_text = _extract_top_level_block(raw_text, key)
    payload = yaml.safe_load(block_text)
    if not isinstance(payload, dict):
        raise ValueError(
            f"automation-state block {key!r} must decode to a mapping"
        )
    block = payload.get(key)
    if not isinstance(block, Mapping):
        raise ValueError(
            f"automation-state block {key!r} must contain a mapping payload"
        )
    return block
