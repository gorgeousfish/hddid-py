from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from hddid.estimation import estimate_eq31_mainline
from hddid.score import build_score_payload


REPO_ROOT = Path(__file__).resolve().parents[3]
R_WRAPPER_FILE = Path(__file__).with_name("r_eq31_raw_wrapper.R")
MACHINE_ATOL = 32.0 * np.finfo(float).eps


def _format_vector(values: np.ndarray) -> str:
    return ",".join(
        format(float(value), ".17g") for value in np.asarray(values, dtype=float)
    )


def _format_matrix(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=float)
    return ";".join(_format_vector(row) for row in array)


def _run_r_wrapper(
    parity_slice: dict[str, object],
    *,
    lambda_override: float | None = 0.0,
    foldid: np.ndarray | None = None,
    seed: int | None = None,
) -> subprocess.CompletedProcess[str]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is required for parity tests")
    if not R_WRAPPER_FILE.exists():
        pytest.fail(f"missing parity wrapper: {R_WRAPPER_FILE}")

    command = [
        rscript,
        "--vanilla",
        str(R_WRAPPER_FILE),
        f"--delta-y-valid={_format_vector(parity_slice['delta_y_valid'])}",
        f"--pi-hat-valid={_format_vector(parity_slice['pi_hat_valid'])}",
        f"--phi1-hat-valid={_format_vector(parity_slice['phi1_hat_valid'])}",
        f"--phi0-hat-valid={_format_vector(parity_slice['phi0_hat_valid'])}",
        f"--rho-hat-valid={_format_vector(parity_slice['rho_hat_valid'])}",
        f"--x-valid={_format_matrix(parity_slice['x_valid'])}",
        f"--z-valid={_format_vector(parity_slice['z_valid'])}",
        f"--z0={_format_vector(parity_slice['z0'])}",
        f"--basis-family={parity_slice['basis_family']}",
        f"--basis-degree={parity_slice['basis_degree']}",
        f"--oracle-lane={parity_slice['oracle_lane']}",
    ]
    if lambda_override is not None:
        command.append(f"--lambda-override={format(float(lambda_override), '.17g')}")
    if foldid is not None:
        command.append(f"--foldid={_format_vector(np.asarray(foldid, dtype=float))}")
    if seed is not None:
        command.append(f"--seed={int(seed)}")
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _parse_vector(payload: str) -> np.ndarray:
    return np.fromstring(payload, sep=" ")


def _parse_matrix(tokens: list[str]) -> np.ndarray:
    rows = int(tokens[0])
    cols = int(tokens[1])
    values = np.fromstring(" ".join(tokens[2:]), sep=" ")
    return values.reshape(rows, cols)


def _parse_wrapper_output(stdout: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for raw_line in stdout.strip().splitlines():
        tokens = raw_line.strip().split()
        if not tokens:
            continue
        tag = tokens[0]
        if tag == "META":
            meta: dict[str, object] = {}
            for token in tokens[1:]:
                key, value = token.split("=", maxsplit=1)
                if key in {
                    "n_valid",
                    "rows_w",
                    "cols_w",
                    "rows_basis_full",
                    "cols_basis_full",
                    "rows_basis_design",
                    "cols_basis_design",
                    "basis_degree",
                    "q",
                }:
                    meta[key] = int(float(value))
                elif key == "lambda_used":
                    meta[key] = float(value)
                else:
                    meta[key] = value
            parsed["meta"] = meta
        elif tag in {"NEWY", "BETAHAT"}:
            parsed[tag.lower()] = _parse_vector(" ".join(tokens[1:]))
        elif tag in {"WVALID", "BASISFULL", "BASISDESIGN", "EVALBASIS"}:
            parsed[tag.lower()] = _parse_matrix(tokens[1:])
        else:
            raise ValueError(f"unexpected tag from wrapper: {tag}")
    return parsed


def test_eq31_raw_wrapper_matches_python_score_objects(
    eq31_parity_slice: dict[str, object],
) -> None:
    parity_slice = eq31_parity_slice
    score_payload = build_score_payload(
        eq31_parity_slice["data"],
        eq31_parity_slice["nuisance_payload"],
    )

    result = _run_r_wrapper(parity_slice, lambda_override=0.0)
    parsed = _parse_wrapper_output(result.stdout)

    expected_newy = parity_slice["rho_hat_valid"] * (
        parity_slice["delta_y_valid"]
        - (1.0 - parity_slice["pi_hat_valid"]) * parity_slice["phi1_hat_valid"]
        - parity_slice["pi_hat_valid"] * parity_slice["phi0_hat_valid"]
    )
    expected_w_valid = np.hstack(
        [score_payload.x_valid, score_payload.basis_design_valid]
    )

    np.testing.assert_allclose(
        parsed["newy"], expected_newy, atol=MACHINE_ATOL, rtol=0.0
    )
    np.testing.assert_allclose(
        parsed["newy"],
        score_payload.s_hat_valid,
        atol=MACHINE_ATOL,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        parsed["basisfull"],
        score_payload.basis_valid_full,
        atol=MACHINE_ATOL,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        parsed["basisdesign"],
        score_payload.basis_design_valid,
        atol=MACHINE_ATOL,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        parsed["wvalid"],
        expected_w_valid,
        atol=MACHINE_ATOL,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        parsed["evalbasis"],
        score_payload.evaluation_basis,
        atol=MACHINE_ATOL,
        rtol=0.0,
    )
    assert parsed["meta"]["n_valid"] == parity_slice["n_valid"]
    assert parsed["meta"]["basis_family"] == "polynomial"
    assert parsed["meta"]["basis_degree"] == 2
    assert parsed["meta"]["q"] == 2
    assert parsed["meta"]["oracle_lane"] == "r-parity-polynomial"


def test_eq31_zero_penalty_coefficients_match_r_snapshot(
    eq31_parity_slice: dict[str, object],
) -> None:
    parity_slice = eq31_parity_slice
    score_payload = build_score_payload(
        eq31_parity_slice["data"],
        eq31_parity_slice["nuisance_payload"],
    )

    estimation_payload, _ = estimate_eq31_mainline(score_payload, penalty_lambda=0.0)
    result = _run_r_wrapper(parity_slice, lambda_override=0.0)
    parsed = _parse_wrapper_output(result.stdout)

    expected_coefficients = np.concatenate(
        [estimation_payload.beta_hat, estimation_payload.gamma_hat[1:]],
    )

    np.testing.assert_allclose(
        parsed["betahat"],
        expected_coefficients,
        atol=MACHINE_ATOL,
        rtol=0.0,
    )


def test_eq31_fixed_foldid_keeps_lambda_and_betahat_deterministic(
    eq31_parity_slice: dict[str, object],
) -> None:
    parity_slice = eq31_parity_slice

    first = _parse_wrapper_output(
        _run_r_wrapper(
            parity_slice,
            lambda_override=None,
            foldid=parity_slice["foldid"],
            seed=1,
        ).stdout,
    )
    second = _parse_wrapper_output(
        _run_r_wrapper(
            parity_slice,
            lambda_override=None,
            foldid=parity_slice["foldid"],
            seed=7,
        ).stdout,
    )

    assert first["meta"]["lambda_used"] == pytest.approx(second["meta"]["lambda_used"])
    np.testing.assert_allclose(
        first["betahat"],
        second["betahat"],
        atol=MACHINE_ATOL,
        rtol=0.0,
    )
