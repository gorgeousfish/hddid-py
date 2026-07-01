from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.<>/-]+$")
_MALFORMED_SUMMARY_MARKERS = ("# ", "## ", "feature_bundle:", "failing_or_fresh_verification:")
_ANCHOR_ONLY_REQUIRED_TOKENS = {"hddid-py/docs/api-reference.md"}
_LATEST_ROLE_SUMMARY_CARRY_TOKENS: dict[str, tuple[str, ...]] = {
    "main-exec": (
        "main-exec closeout marker",
        "main-exec-01",
        "01/16/31/46",
        "historical blocker",
        "object-flow runtime evidence",
        "seed303",
        "seed707",
    ),
}
_ROLE_NEXT_RUNG_REPAIR_RULES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "research-backfill": (
        (
            "keep trigger2 same-seed helper index aligned",
            (
                "research-backfill same-seed helper index frontier public surface sync",
                "run_phase7_same_seed_helper_index()",
            ),
        ),
        ("hddid.validation", ("hddid.validation",)),
        ("hddid-py/README.md", ("hddid-py/README.md",)),
        (
            "hddid-py/docs/api-reference.md",
            ("research-backfill same-seed helper index frontier public surface sync",),
        ),
        ("run_phase7_same_seed_helper_index()", ("run_phase7_same_seed_helper_index()",)),
        (
            "run_phase7_same_seed_before_after_acceptance_probe()",
            ("run_phase7_same_seed_before_after_acceptance_probe()",),
        ),
        (
            "run_phase7_same_seed_first_hop_acceptance_delta_y_support_chain_contract()",
            ("run_phase7_same_seed_first_hop_acceptance_delta_y_support_chain_contract()",),
        ),
        (
            "run_phase7_same_seed_first_hop_acceptance_low_pi_support_allocation_contract()",
            ("run_phase7_same_seed_first_hop_acceptance_low_pi_support_allocation_contract()",),
        ),
        (
            "keep validation-only runtime bridge source-guard helper aligned",
            (
                "validation-only runtime bridge source-guard helper",
                "run_phase7_same_seed_runtime_bridge_source_guard()",
            ),
        ),
        (
            "keep validation-only after-report admission-order helper aligned",
            (
                "validation-only after-report admission-order helper",
                "run_phase7_same_seed_runtime_bridge_after_report_admission_order()",
            ),
        ),
        (
            "keep validation-only runtime bridge after-report intake helper aligned",
            (
                "validation-only runtime bridge after-report intake helper",
                "run_phase7_same_seed_runtime_bridge_after_report_intake()",
            ),
        ),
        (
            "keep validation-only runtime bridge after-report frontier packet aligned",
            (
                "validation-only runtime bridge after-report frontier packet",
                "run_phase7_same_seed_runtime_bridge_after_report_frontier_packet()",
            ),
        ),
        (
            "keep validation-only seed303 object-flow blocker public surface aligned",
            (
                "validation-only seed303 object-flow blocker",
                "run_phase7_same_seed_seed303_object_flow_blocker()",
            ),
        ),
        (
            "keep validation-only seed303 acceptance exact-trim-floor stack public surface aligned",
            (
                "validation-only acceptance exact trim-floor stack index",
                "run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index()",
            ),
        ),
        (
            "run_phase7_same_seed_runtime_bridge_source_guard()",
            ("run_phase7_same_seed_runtime_bridge_source_guard()",),
        ),
        (
            "run_phase7_same_seed_runtime_bridge_after_report_admission_order()",
            ("run_phase7_same_seed_runtime_bridge_after_report_admission_order()",),
        ),
        (
            "run_phase7_same_seed_runtime_bridge_after_report_intake()",
            ("run_phase7_same_seed_runtime_bridge_after_report_intake()",),
        ),
        (
            "run_phase7_same_seed_runtime_bridge_after_report_frontier_packet()",
            ("run_phase7_same_seed_runtime_bridge_after_report_frontier_packet()",),
        ),
        (
            "run_phase7_same_seed_seed303_support_trace_index()",
            ("run_phase7_same_seed_seed303_support_trace_index()",),
        ),
        (
            "run_phase7_same_seed_seed303_trim_floor_support_trace_index()",
            ("run_phase7_same_seed_seed303_trim_floor_support_trace_index()",),
        ),
        (
            "run_phase7_same_seed_seed303_object_flow_blocker()",
            ("run_phase7_same_seed_seed303_object_flow_blocker()",),
        ),
        (
            "run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index()",
            ("run_phase7_same_seed_seed303_acceptance_exact_trim_floor_stack_index()",),
        ),
        (
            "validation-only support-trace helper bundle",
            ("validation-only support-trace helper bundle",),
        ),
        (
            "validation-only trim-floor support-trace helper bundle",
            ("validation-only trim-floor support-trace helper bundle",),
        ),
        (
            "validation-only seed303 object-flow blocker",
            ("validation-only seed303 object-flow blocker",),
        ),
        (
            "validation-only acceptance exact trim-floor stack index",
            ("validation-only acceptance exact trim-floor stack index",),
        ),
        (
            "same-seed-seed303-fold3-trim-floor-inverse-pi-concentration-confirmed",
            ("same-seed-seed303-fold3-trim-floor-inverse-pi-concentration-confirmed",),
        ),
        (
            "same-seed-exact-witness-seed303-pointwise-overshoot-object-flow-blocker",
            ("same-seed-exact-witness-seed303-pointwise-overshoot-object-flow-blocker",),
        ),
        (
            "same-seed-exact-witness-observed-rerun-open",
            ("same-seed-exact-witness-observed-rerun-open",),
        ),
        (
            "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-open",
            (
                "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-concentration-open",
            ),
        ),
        (
            "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-open",
            (
                "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-inverse-pi-conduit-open",
            ),
        ),
        (
            "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-open",
            (
                "same-seed-exact-witness-observed-rerun-first-hop-acceptance-exact-trim-floor-landing-bridge-open",
            ),
        ),
        ("bar_f_at_z0[1]", ("bar_f_at_z0[1]",)),
        ("v_f_hat[2,1]", ("v_f_hat[2,1]",)),
        (
            "seed 303 / z = 0.15",
            (
                "same-seed-seed303-fold3-trim-floor-inverse-pi-concentration-confirmed",
                "same-seed-exact-witness-seed303-pointwise-overshoot-object-flow-blocker",
            ),
        ),
        (
            "seed 707 / z = 0.25",
            (
                "seed303-before-seed707",
                "same-seed-exact-witness-seed303-pointwise-overshoot-object-flow-blocker",
            ),
        ),
        (
            "acceptance delta-y support chain contract",
            ("acceptance delta-y support chain contract",),
        ),
        (
            "acceptance low-pi support allocation contract",
            ("acceptance low-pi support allocation contract",),
        ),
        (
            "acceptance exact trim-floor runtime bridge",
            ("acceptance exact trim-floor runtime bridge",),
        ),
    ),
}
_ROLE_NEXT_RUNG_DYNAMIC_PREFIXES: dict[str, tuple[str, ...]] = {
    "research-backfill": (
        "run_phase7_same_seed_",
        "run_phase7_monte_carlo_widening_policy_coverage_anchor_shoulder_same_seed_",
    ),
}


def canonicalize_automation_state_text(raw_text: str) -> str:
    """Normalize automation-state YAML back to the repo's canonical surface."""
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        raise ValueError("automation-state content must decode to a mapping")
    _repair_latest_role_closeout_summaries(data)
    _repair_role_last_closeout_next_rungs(data)
    return "\n".join(_emit_node(data, indent=0)) + "\n"


def canonicalize_automation_state_file(path: str | Path) -> bool:
    target = Path(path)
    raw_text = target.read_text(encoding="utf-8")
    canonical = canonicalize_automation_state_text(raw_text)
    if canonical == raw_text:
        return False
    target.write_text(canonical, encoding="utf-8")
    return True


def _repair_latest_role_closeout_summaries(data: dict[str, Any]) -> None:
    role_last_closeouts = data.get("role_last_closeouts")
    latest_role_closeouts = data.get("latest_role_closeouts")
    if not isinstance(role_last_closeouts, dict) or not isinstance(latest_role_closeouts, dict):
        return

    for role, latest_payload in latest_role_closeouts.items():
        if not isinstance(latest_payload, dict):
            continue
        role_payload = role_last_closeouts.get(role)
        if not isinstance(role_payload, dict):
            continue

        role_summary = role_payload.get("summary")
        latest_summary = latest_payload.get("summary")
        if not isinstance(role_summary, str) or not isinstance(latest_summary, str):
            continue
        if not _latest_summary_needs_repair(role_summary, latest_summary):
            continue

        repaired_summary = role_summary
        for token in _LATEST_ROLE_SUMMARY_CARRY_TOKENS.get(str(role), ()):
            if token in latest_summary and token not in repaired_summary:
                repaired_summary = f"{repaired_summary} / {token}"
        latest_payload["summary"] = repaired_summary


def _repair_role_last_closeout_next_rungs(data: dict[str, Any]) -> None:
    role_last_closeouts = data.get("role_last_closeouts")
    latest_role_closeouts = data.get("latest_role_closeouts")
    if not isinstance(role_last_closeouts, dict):
        return

    for role, repair_rules in _ROLE_NEXT_RUNG_REPAIR_RULES.items():
        role_payload = role_last_closeouts.get(role)
        if not isinstance(role_payload, dict):
            continue

        next_rung = role_payload.get("next_rung")
        role_summary = role_payload.get("summary")
        latest_payload = (
            latest_role_closeouts.get(role)
            if isinstance(latest_role_closeouts, dict)
            else None
        )
        latest_summary = (
            latest_payload.get("summary")
            if isinstance(latest_payload, dict)
            else None
        )
        if not isinstance(next_rung, str):
            continue

        repaired_next_rung = next_rung
        for required_token, anchors in repair_rules:
            if required_token in repaired_next_rung:
                continue
            if not _token_visible_in_any_text(required_token, anchors, role_summary, latest_summary):
                continue
            repaired_next_rung = f"{repaired_next_rung} / {required_token}"

        for token in _dynamic_next_rung_tokens(str(role), role_summary, latest_summary):
            if token in repaired_next_rung:
                continue
            repaired_next_rung = f"{repaired_next_rung} / {token}"
        role_payload["next_rung"] = repaired_next_rung


def _latest_summary_needs_repair(role_summary: str, latest_summary: str) -> bool:
    if _looks_like_malformed_latest_summary(latest_summary):
        return True
    if "fresh verification / direct repair / rerun green" in latest_summary:
        return False
    if (
        "role_last_closeouts.correct-course / latest_role_closeouts.correct-course closeout marker refreshed together"
        in latest_summary
    ):
        return False
    if (
        "latest_role_closeouts.research-backfill now mirrors the current Trigger 1"
        in latest_summary
    ):
        return False
    return any(token not in latest_summary for token in _summary_tokens(role_summary))


def _looks_like_malformed_latest_summary(summary: str) -> bool:
    return any(marker in summary for marker in _MALFORMED_SUMMARY_MARKERS)


def _token_visible_in_any_text(
    required_token: str,
    anchors: tuple[str, ...],
    *texts: Any,
) -> bool:
    for text in texts:
        if not isinstance(text, str):
            continue
        if required_token in text and required_token not in _ANCHOR_ONLY_REQUIRED_TOKENS:
            return True
        if any(anchor in text for anchor in anchors):
            return True
    return False


def _summary_tokens(summary: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in summary.split(" / ") if token.strip())


def _dynamic_next_rung_tokens(role: str, *texts: Any) -> tuple[str, ...]:
    prefixes = _ROLE_NEXT_RUNG_DYNAMIC_PREFIXES.get(role, ())
    if not prefixes:
        return ()

    seen: set[str] = set()
    ordered_tokens: list[str] = []
    for text in texts:
        if not isinstance(text, str):
            continue
        for token in _summary_tokens(text):
            if token in seen:
                continue
            if not any(token.startswith(prefix) for prefix in prefixes):
                continue
            seen.add(token)
            ordered_tokens.append(token)
    return tuple(ordered_tokens)


def _emit_node(node: Any, indent: int) -> list[str]:
    if isinstance(node, dict):
        return _emit_mapping(node, indent)
    if isinstance(node, list):
        return _emit_sequence(node, indent)
    return [f"{'  ' * indent}{_format_scalar(node)}"]


def _emit_mapping(node: dict[Any, Any], indent: int) -> list[str]:
    lines: list[str] = []
    for key, value in node.items():
        lines.extend(_emit_mapping_item(str(key), value, indent))
    return lines


def _emit_sequence(node: list[Any], indent: int) -> list[str]:
    lines: list[str] = []
    prefix = "  " * indent
    for item in node:
        if isinstance(item, dict):
            items = list(item.items())
            if not items:
                lines.append(f"{prefix}- {{}}")
                continue
            first_key, first_value = items[0]
            lines.extend(
                _emit_mapping_item(
                    str(first_key),
                    first_value,
                    indent,
                    list_prefix=f"{prefix}- ",
                )
            )
            for key, value in items[1:]:
                lines.extend(_emit_mapping_item(str(key), value, indent + 1))
            continue
        if isinstance(item, list):
            lines.append(f"{prefix}-")
            lines.extend(_emit_sequence(item, indent + 1))
            continue
        lines.append(f"{prefix}- {_format_scalar(item)}")
    return lines


def _emit_mapping_item(
    key: str,
    value: Any,
    indent: int,
    list_prefix: str | None = None,
) -> list[str]:
    prefix = list_prefix if list_prefix is not None else "  " * indent
    head = f"{prefix}{_format_key(key)}:"
    if _is_scalar(value):
        return [f"{head} {_format_scalar(value)}"]

    if list_prefix is not None:
        child_indent = indent + 1 if isinstance(value, list) else indent + 2
    else:
        child_indent = indent if isinstance(value, list) else indent + 1
    return [head, *_emit_node(value, child_indent)]


def _format_key(key: str) -> str:
    if _SAFE_KEY.fullmatch(key):
        return key
    return json.dumps(key, ensure_ascii=False)


def _format_scalar(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Canonicalize Docs/automation/automation-state.yaml formatting."
    )
    parser.add_argument("path", nargs="?", default="Docs/automation/automation-state.yaml")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return non-zero if the file is not already canonical.",
    )
    args = parser.parse_args(argv)

    target = Path(args.path)
    raw_text = target.read_text(encoding="utf-8")
    canonical = canonicalize_automation_state_text(raw_text)

    if args.check:
        return 0 if canonical == raw_text else 1

    if canonical != raw_text:
        target.write_text(canonical, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
