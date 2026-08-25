#!/usr/bin/env python3
"""Run the exhaustive temperature-dependent spanning-tree study."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from tree_geometry_core import EXPECTED_TOTAL_TREES, RCOND, S_GRID
from tree_geometry_compute import run_instance

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="repository root",
    )
    parser.add_argument(
        "--clean-results",
        action="store_true",
        help="remove existing generated results before running",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results" / "temperature_tree_geometry"
    atlas = results / "atlas"
    atlas.mkdir(parents=True, exist_ok=True)
    if args.clean_results:
        for path in results.iterdir():
            if path.name == "atlas":
                for child in path.iterdir():
                    if child.is_file():
                        child.unlink()
            elif path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    instance_paths = sorted(
        (Path(__file__).resolve().parents[1] / "instances").glob("n8i*.txt"),
        key=lambda path: int(path.stem.removeprefix("n8i")),
    )
    if len(instance_paths) != 10:
        raise RuntimeError(f"Expected ten instances, found {len(instance_paths)}")

    full_csv_path = results / "temperature_tree_geometry.csv.gz"
    if full_csv_path.exists():
        full_csv_path.unlink()

    all_optima: list[pd.DataFrame] = []
    all_catalogs: list[pd.DataFrame] = []
    validations: list[dict[str, object]] = []
    total_trees = 0
    global_started = time.perf_counter()

    for position, instance_path in enumerate(instance_paths):
        optima, catalog, validation, _ = run_instance(
            instance_path,
            root,
            full_csv_path,
            write_header=position == 0,
        )
        all_optima.append(optima)
        all_catalogs.append(catalog)
        validations.append(validation)
        total_trees += int(validation["tree_count"])

    if total_trees != EXPECTED_TOTAL_TREES:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL_TREES} spanning trees, obtained {total_trees}"
        )

    optima = pd.concat(all_optima, ignore_index=True)
    catalog = pd.concat(all_catalogs, ignore_index=True)
    optima.to_csv(results / "tree_optima_by_temperature.csv", index=False, float_format="%.12g")
    catalog.to_csv(
        results / "tree_catalog.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 6},
        float_format="%.12g",
    )

    instance_summary = pd.DataFrame(validations)
    instance_summary.to_csv(
        results / "instance_gate_summary.csv", index=False, float_format="%.12g"
    )

    # Small focused tables promised by the protocol.
    optima[
        [
            "instance",
            "beta_index",
            "s",
            "q_advantage_fraction_full",
            "hot_defect_recovery",
            "gap_improvement_over_hot",
            "hot_in_q_near_set",
            "min_edge_swaps_hot_to_qopt",
            "edge_frequency_tv_hot_vs_qnear",
        ]
    ].to_csv(results / "maxj_oracle_gap.csv", index=False, float_format="%.12g")
    optima[
        [
            "instance",
            "beta_index",
            "s",
            "qopt_tree_hash",
            "klopt_tree_hash",
            "qopt_klopt_intersect",
            "q_loss_klopt_fraction_full",
            "kl_loss_qopt",
            "gap_qopt_best",
            "gap_klopt_best",
        ]
    ].to_csv(results / "kl_vs_cooling_optima.csv", index=False, float_format="%.12g")
    optima[
        [
            "instance",
            "beta_index",
            "s",
            "xi_maxj",
            "xi_qopt_min",
            "xi_qopt_median",
            "xi_hot_min",
            "xi_hot_median",
            "xi_klopt_min",
            "xi_klopt_median",
            "lambda_sigma_qopt_min",
            "lambda_sigma_hot_min",
        ]
    ].to_csv(results / "tracking_mismatch.csv", index=False, float_format="%.12g")

    gate_a_count = int(instance_summary["gate_a_instance_pass"].sum())
    gate_b_count = int(instance_summary["gate_b_instance_pass"].sum())
    gate_c_count = int(instance_summary["gate_c_instance_pass"].sum())
    gate_d_count = int(instance_summary["gate_d_instance_pass"].sum())
    median_max_defect_recovery = float(instance_summary["max_defect_recovery"].median())
    median_mean_gap_improvement = float(instance_summary["mean_gap_improvement"].median())

    gate_a_project = gate_a_count >= 5
    gate_b_project = gate_b_count >= 5 and median_max_defect_recovery >= 0.10
    gate_c_project = gate_c_count >= 5 and median_mean_gap_improvement > 0.0
    gate_d_project = gate_d_count >= 5

    if gate_a_project and gate_b_project and gate_c_project and gate_d_project:
        decision = "GO_TO_PT1B"
    elif gate_a_project and gate_b_project:
        decision = "GEOMETRY_ONLY_DO_NOT_USE_COOLING_POWER_AS_OPERATIONAL_SELECTOR"
    else:
        decision = "STOP_ADAPTIVE_TREE_BRANCH"

    validation_pass = all(
        int(row["ground_count"]) == 1
        and bool(row["planted_is_ground"])
        and abs(float(row["spectral_gap"]) - 1.0) <= 1.0e-12
        and abs(float(row["beta_cert_residual"])) <= 1.0e-10
        and float(row["hamiltonian_max_residual"]) <= 1.0e-12
        and float(row["beta0_hot_identity_residual"]) <= 1.0e-10
        and float(row["max_projection_normalization_residual"]) <= 1.0e-10
        and float(row["max_projection_moment_residual"]) <= 1.0e-10
        and float(row["minimum_forward_kl"]) >= -1.0e-10
        and float(row["maximum_q_minus_full_variance"]) <= 1.0e-9
        and float(row["maximum_retained_fraction_minus_one"]) <= 1.0e-9
        and bool(row["maxj_is_hot_optimal"])
        for row in validations
    )

    verdict = {
        "protocol": "temperature-dependent tree geometry",
        "date": "2026-08-25",
        "instance_count": len(instance_paths),
        "total_tree_count": total_trees,
        "temperature_points_per_instance": len(S_GRID),
        "tree_temperature_cells": total_trees * len(S_GRID),
        "validation_pass": validation_pass,
        "gate_a": {
            "pass": gate_a_project,
            "passing_instances": gate_a_count,
            "required_instances": 5,
        },
        "gate_b": {
            "pass": gate_b_project,
            "passing_instances": gate_b_count,
            "required_instances": 5,
            "median_max_defect_recovery": median_max_defect_recovery,
            "required_median": 0.10,
        },
        "gate_c": {
            "pass": gate_c_project,
            "passing_instances": gate_c_count,
            "required_instances": 5,
            "median_mean_gap_improvement": median_mean_gap_improvement,
        },
        "gate_d": {
            "pass": gate_d_project,
            "passing_instances": gate_d_count,
            "required_instances": 5,
        },
        "decision": decision,
        "runtime_seconds": time.perf_counter() - global_started,
        "pseudoinverse_rcond": RCOND,
        "normalized_temperature_grid": S_GRID.tolist(),
    }
    (results / "pt1a_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results / "validation_report.json").write_text(
        json.dumps(
            {
                "overall_pass": validation_pass,
                "expected_total_trees": EXPECTED_TOTAL_TREES,
                "observed_total_trees": total_trees,
                "instances": validations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(verdict, indent=2, sort_keys=True), flush=True)
    return 0 if validation_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
