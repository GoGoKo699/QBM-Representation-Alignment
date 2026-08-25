#!/usr/bin/env python3
"""Validate the independent weighted sparse-Ising confirmation."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
OUTPUT = REPOSITORY / "results" / "confirmatory"
FROZEN = EXPERIMENT / "protocol" / "frozen_source"
sys.path.insert(0, str(REPOSITORY / "src"))

from qbm_alignment.sparse_ising import (
    PARAMETER_SEEDS,
    batch_state,
    build_problem,
    canonical_noise,
    exact_state,
    load_manifest,
)


def canonical_payload_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def assert_close(left: float, right: float, tolerance: float, label: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: {left} != {right}")


def main() -> int:
    checks: list[dict[str, object]] = []

    lock = json.loads((EXPERIMENT / "protocol" / "protocol_lock.json").read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        actual = hashlib.sha256((FROZEN / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError((relative, actual, expected))
    compact_lock = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if canonical_payload_hash(compact_lock) != lock["lock_sha256"]:
        raise AssertionError("protocol-lock hash mismatch")

    seed_payload = json.loads((EXPERIMENT / "protocol" / "seed_commitment.json").read_text(encoding="utf-8"))
    compact_seed = {
        "master_seed": seed_payload["master_seed"],
        "engineering_instance_seeds": seed_payload["engineering_instance_seeds"],
        "confirmatory_instance_seeds": seed_payload["confirmatory_instance_seeds"],
    }
    if canonical_payload_hash(compact_seed) != seed_payload["canonical_json_sha256"]:
        raise AssertionError("seed-commitment hash mismatch")
    checks.append({"name": "protocol and seed commitments", "passed": True})

    manifest_path = EXPERIMENT / "instances" / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_without_hash = {
        key: value for key, value in manifest_payload.items() if key != "manifest_sha256"
    }
    if canonical_payload_hash(manifest_without_hash) != manifest_payload["manifest_sha256"]:
        raise AssertionError("manifest canonical hash mismatch")
    instances = load_manifest(manifest_path)
    if len(instances) != 24:
        raise AssertionError("expected 24 confirmatory instances")

    spectral_rows: list[dict[str, object]] = []
    for instance in instances:
        problem = build_problem(instance)
        order = np.argsort(problem.cost, kind="stable")
        ground_index = int(order[0])
        ground_energy = float(problem.cost[ground_index])
        degeneracy = int(
            np.sum(np.isclose(problem.cost, ground_energy, atol=1e-12, rtol=0.0))
        )
        gap = float(problem.cost[int(order[1])] - ground_energy)
        if ground_index != instance.ground_index or degeneracy != 1:
            raise AssertionError(f"ground-state mismatch for {instance.instance_id}")
        assert_close(ground_energy, instance.ground_energy, 2e-10, "ground energy")
        assert_close(gap, instance.spectral_gap, 2e-10, "spectral gap")
        if gap < 0.05:
            raise AssertionError("gap acceptance violation")
        coefficient_rms = float(
            np.sqrt(np.mean(np.concatenate([instance.fields, instance.target_couplings]) ** 2))
        )
        assert_close(coefficient_rms, 1.0, 2e-12, "coefficient RMS")
        for graph_name in ("random_tree", "problem_tree"):
            edges = problem.representations[graph_name].edges
            if len(edges) != 15 or not set(edges).issubset(set(instance.target_edges)):
                raise AssertionError(f"invalid {graph_name} for {instance.instance_id}")
        spectral_rows.append(
            {
                "instance_id": instance.instance_id,
                "ground_index": ground_index,
                "ground_energy": ground_energy,
                "spectral_gap": gap,
                "coefficient_rms": coefficient_rms,
            }
        )
    spectra = pd.DataFrame(spectral_rows)
    spectra.to_csv(OUTPUT / "validation_spectra.csv", index=False)
    checks.append({"name": "24 exact spectra and graph constructions", "passed": True})

    summary_paths = sorted((EXPERIMENT / "results" / "raw").glob("*_summary.csv"))
    log_paths = sorted((EXPERIMENT / "results" / "raw").glob("*_logs.csv.gz"))
    state_paths = sorted((EXPERIMENT / "results" / "states").glob("*_states.npz"))
    if not (len(summary_paths) == len(log_paths) == len(state_paths) == 24):
        raise AssertionError("raw-file count mismatch")
    summaries = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True)
    if len(summaries) != 1440:
        raise AssertionError(f"expected 1440 trajectories, got {len(summaries)}")
    expected_counts = summaries.groupby(["method", "initialization", "graph"]).size().to_dict()
    for method in ("adam", "exact_natural"):
        initializations = ("random", "target_biased") if method == "adam" else ("target_biased",)
        for graph in ("chain", "random_tree", "problem_tree", "full"):
            for initialization in initializations:
                if expected_counts[(method, initialization, graph)] != 120:
                    raise AssertionError("trajectory-cell count mismatch")

    maximum_initial_residual = 0.0
    maximum_final_gap_residual = 0.0
    maximum_final_pstar_residual = 0.0
    seed_array = np.asarray(PARAMETER_SEEDS, dtype=np.int64)
    for instance in instances:
        problem = build_problem(instance)
        state_file = np.load(
            EXPERIMENT / "results" / "states" / f"{instance.instance_id}_states.npz"
        )
        instance_summary = summaries[summaries.instance_id == instance.instance_id]
        for graph_name, representation in problem.representations.items():
            for initialization in ("random", "target_biased"):
                stored_initial = state_file[f"{graph_name}__adam__{initialization}__initial"]
                stored_final = state_file[f"{graph_name}__adam__{initialization}__final"]
                expected_initial = np.vstack(
                    [canonical_noise(seed, representation.edges) for seed in PARAMETER_SEEDS]
                )
                if initialization == "target_biased":
                    expected_initial = expected_initial + representation.coefficients[None, :]
                maximum_initial_residual = max(
                    maximum_initial_residual,
                    float(np.max(np.abs(stored_initial - expected_initial))),
                )
                energy, _gradient, probability, _fishers, _moments = batch_state(
                    stored_final, problem, representation
                )
                stored_rows = (
                    instance_summary[
                        (instance_summary.graph == graph_name)
                        & (instance_summary.method == "adam")
                        & (instance_summary.initialization == initialization)
                    ]
                    .set_index("seed")
                    .loc[list(PARAMETER_SEEDS)]
                )
                gap = energy - instance.ground_energy
                maximum_final_gap_residual = max(
                    maximum_final_gap_residual,
                    float(np.max(np.abs(gap - stored_rows.final_gap.to_numpy(float)))),
                )
                maximum_final_pstar_residual = max(
                    maximum_final_pstar_residual,
                    float(
                        np.max(
                            np.abs(
                                probability[instance.ground_index, :]
                                - stored_rows.final_pstar.to_numpy(float)
                            )
                        )
                    ),
                )

            exact_initial = np.vstack(
                [
                    state_file[
                        f"{graph_name}__exact_natural__target_biased__seed{seed}__initial"
                    ]
                    for seed in PARAMETER_SEEDS
                ]
            )
            exact_final = np.vstack(
                [
                    state_file[
                        f"{graph_name}__exact_natural__target_biased__seed{seed}__final"
                    ]
                    for seed in PARAMETER_SEEDS
                ]
            )
            expected_exact = np.vstack(
                [
                    representation.coefficients
                    + canonical_noise(seed, representation.edges)
                    for seed in PARAMETER_SEEDS
                ]
            )
            maximum_initial_residual = max(
                maximum_initial_residual,
                float(np.max(np.abs(exact_initial - expected_exact))),
            )
            energy, _gradient, probability, _fishers, _moments = batch_state(
                exact_final, problem, representation
            )
            stored_rows = (
                instance_summary[
                    (instance_summary.graph == graph_name)
                    & (instance_summary.method == "exact_natural")
                ]
                .set_index("seed")
                .loc[list(PARAMETER_SEEDS)]
            )
            gap = energy - instance.ground_energy
            maximum_final_gap_residual = max(
                maximum_final_gap_residual,
                float(np.max(np.abs(gap - stored_rows.final_gap.to_numpy(float)))),
            )
            maximum_final_pstar_residual = max(
                maximum_final_pstar_residual,
                float(
                    np.max(
                        np.abs(
                            probability[instance.ground_index, :]
                            - stored_rows.final_pstar.to_numpy(float)
                        )
                    )
                ),
            )

    if maximum_initial_residual > 1e-14:
        raise AssertionError(f"initialization residual {maximum_initial_residual}")
    if maximum_final_gap_residual > 2e-10 or maximum_final_pstar_residual > 2e-11:
        raise AssertionError("final-state residual too large")
    checks.append(
        {
            "name": "all stored initial and final states",
            "passed": True,
            "maximum_initial_residual": maximum_initial_residual,
            "maximum_final_gap_residual": maximum_final_gap_residual,
            "maximum_final_pstar_residual": maximum_final_pstar_residual,
        }
    )

    success_rows: list[pd.Series] = []
    monotonic_violations = 0
    maximum_monotonic_increase = 0.0
    for path in log_paths:
        frame = pd.read_csv(path)
        exact = frame[frame.method == "exact_natural"]
        for _key, trajectory in exact.groupby(["graph", "seed"]):
            difference = np.diff(trajectory.sort_values("step").gap.to_numpy(float))
            if difference.size:
                maximum_monotonic_increase = max(
                    maximum_monotonic_increase, float(difference.max())
                )
                monotonic_violations += int(np.sum(difference > 2e-10))
        for _key, trajectory in frame.groupby(
            ["method", "initialization", "graph", "seed"]
        ):
            trajectory = trajectory.sort_values("step")
            qualifying = trajectory[trajectory.normalized_gap <= 0.1]
            if qualifying.empty:
                continue
            first = qualifying.iloc[0]
            if float(first.pstar) < 0.9 - 2e-10:
                raise AssertionError("spectral success certificate violated")
            previous = trajectory[trajectory.step < first.step]
            if not previous.empty and float(previous.iloc[-1].normalized_gap) <= 0.1:
                raise AssertionError("first-success convention violated")
            success_rows.append(first)
    if monotonic_violations:
        raise AssertionError(f"exact-natural monotonicity violations: {monotonic_violations}")
    checks.append(
        {
            "name": "success certificate and exact-natural monotonicity",
            "passed": True,
            "successful_trajectories": len(success_rows),
            "minimum_success_pstar": float(min(row.pstar for row in success_rows)),
            "maximum_exact_natural_gap_increase": maximum_monotonic_increase,
        }
    )

    canonical_trajectories = pd.read_csv(OUTPUT / "trajectory_summary.csv")
    if canonical_trajectories.shape != summaries.shape:
        raise AssertionError("canonical trajectory-summary shape mismatch")
    pd.testing.assert_frame_equal(
        canonical_trajectories.reset_index(drop=True),
        summaries.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    aggregate = pd.read_csv(OUTPUT / "aggregate.csv")
    effects = pd.read_csv(OUTPUT / "primary_effects.csv")
    hypotheses = json.loads((OUTPUT / "hypothesis_results.json").read_text(encoding="utf-8"))
    if hypotheses["status"] != "PASS" or not effects.hypothesis_pass.astype(bool).all():
        raise AssertionError("one or more prespecified comparisons failed")
    if not hypotheses["preparation_criterion_pass"]:
        raise AssertionError("preparation criterion failed")
    expected_successes = {
        ("adam", "random", "chain"): 1,
        ("adam", "random", "random_tree"): 1,
        ("adam", "random", "problem_tree"): 1,
        ("adam", "random", "full"): 14,
        ("adam", "target_biased", "chain"): 4,
        ("adam", "target_biased", "random_tree"): 21,
        ("adam", "target_biased", "problem_tree"): 43,
        ("adam", "target_biased", "full"): 84,
        ("exact_natural", "target_biased", "chain"): 35,
        ("exact_natural", "target_biased", "random_tree"): 69,
        ("exact_natural", "target_biased", "problem_tree"): 97,
        ("exact_natural", "target_biased", "full"): 120,
    }
    observed = {
        (row.method, row.initialization, row.graph): int(row.successes)
        for row in aggregate.itertuples(index=False)
    }
    if observed != expected_successes:
        raise AssertionError("aggregate success counts changed")
    checks.append({"name": "prespecified paired effects and aggregate counts", "passed": True})

    resources = pd.read_csv(OUTPUT / "preparation_resources.csv")
    if len(resources) != 96:
        raise AssertionError("resource row count")
    trees = resources[resources.graph.isin(["chain", "random_tree", "problem_tree"])]
    if not ((trees.width == 1).all() and (trees.conditional_angle_entries == 31).all()):
        raise AssertionError("tree preparation resources")
    full = resources[resources.graph == "full"]
    if not full.width.between(3, 5).all():
        raise AssertionError("full-graph width range")
    checks.append({"name": "preparation resource accounting", "passed": True})

    validation = {
        "status": "PASS",
        "checks": checks,
        "manifest_payload_sha256": manifest_payload["manifest_sha256"],
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "protocol_lock_sha256": lock["lock_sha256"],
        "seed_commitment_sha256": seed_payload["canonical_json_sha256"],
        "trajectories": len(summaries),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
