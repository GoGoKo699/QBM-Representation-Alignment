#!/usr/bin/env python3
"""Regenerate the canonical confirmatory summary tables from raw trajectories."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
RAW = EXPERIMENT / "results" / "raw"
OUTPUT = REPOSITORY / "results" / "confirmatory"
OUTPUT.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260825
RESOURCE_RATIO_THRESHOLD = 0.5

PRIMARY = (
    {
        "id": "H1",
        "label": "target-biased Adam: problem tree minus chain",
        "method": "adam",
        "initialization": "target_biased",
        "treatment": "problem_tree",
        "control": "chain",
        "minimum_effect": 0.15,
    },
    {
        "id": "H2",
        "label": "target-biased Adam: problem tree minus random problem tree",
        "method": "adam",
        "initialization": "target_biased",
        "treatment": "problem_tree",
        "control": "random_tree",
        "minimum_effect": 0.10,
    },
    {
        "id": "H3",
        "label": "target-biased exact natural: problem tree minus chain",
        "method": "exact_natural",
        "initialization": "target_biased",
        "treatment": "problem_tree",
        "control": "chain",
        "minimum_effect": 0.25,
    },
)

SECONDARY = (
    ("Adam biased full - problem tree", "adam", "target_biased", "full", "problem_tree"),
    ("Exact natural full - problem tree", "exact_natural", "target_biased", "full", "problem_tree"),
    ("Adam biased random tree - chain", "adam", "target_biased", "random_tree", "chain"),
)


def _holm_adjusted(pvalues: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * pvalues[key]))
        adjusted[key] = running
    return adjusted, ordered


def _paired_effect(
    trajectories: pd.DataFrame,
    *,
    method: str,
    initialization: str,
    treatment: str,
    control: str,
    rng: np.random.Generator,
) -> tuple[dict[str, object], np.ndarray]:
    subset = trajectories[
        (trajectories.method == method) & (trajectories.initialization == initialization)
    ]
    table = subset.pivot_table(
        index=["instance_id", "seed"],
        columns="graph",
        values="success",
        aggfunc="first",
    )[[treatment, control]].dropna()
    instance_difference = (table[treatment] - table[control]).groupby(level=0).mean()
    values = instance_difference.to_numpy(dtype=float)
    draws = values[
        rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    ].mean(axis=1)
    raw_low, raw_high = np.quantile(draws, [0.025, 0.975])
    left = (np.sum(draws <= 0.0) + 1) / (BOOTSTRAP_REPLICATES + 1)
    right = (np.sum(draws >= 0.0) + 1) / (BOOTSTRAP_REPLICATES + 1)
    pvalue = min(1.0, 2.0 * min(left, right))
    result: dict[str, object] = {
        "point_difference": float(values.mean()),
        "raw_ci_low": float(raw_low),
        "raw_ci_high": float(raw_high),
        "raw_bootstrap_pvalue": float(pvalue),
        "a_only": int(((table[treatment] == 1) & (table[control] == 0)).sum()),
        "b_only": int(((table[treatment] == 0) & (table[control] == 1)).sum()),
        "both": int(((table[treatment] == 1) & (table[control] == 1)).sum()),
        "neither": int(((table[treatment] == 0) & (table[control] == 0)).sum()),
        "instance_differences": " ".join(f"{value:.3f}" for value in values),
    }
    return result, draws


def _query(method: str, initialization: str, graph: str) -> str:
    return (
        f"method == '{method}' and initialization == '{initialization}' "
        f"and graph == '{graph}'"
    )


def main() -> int:
    raw_paths = sorted(RAW.glob("*_summary.csv"))
    if len(raw_paths) != 24:
        raise FileNotFoundError(f"expected 24 raw summary files under {RAW}, found {len(raw_paths)}")
    trajectories = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True)
    trajectories.to_csv(OUTPUT / "trajectory_summary.csv", index=False)

    aggregate = (
        trajectories.groupby(["method", "initialization", "graph"], as_index=False)
        .agg(
            trajectories=("success", "size"),
            successes=("success", "sum"),
            success_rate=("success", "mean"),
            mean_first_success_step=("first_success_step", "mean"),
            median_minimum_normalized_gap=("minimum_normalized_gap", "median"),
            mean_final_pstar=("final_pstar", "mean"),
            boundary_traps=("boundary_trap", "sum"),
        )
        .sort_values(["method", "initialization", "graph"])
    )
    aggregate.to_csv(OUTPUT / "aggregate.csv", index=False)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary_rows: list[dict[str, object]] = []
    primary_draws: dict[str, np.ndarray] = {}
    raw_pvalues: dict[str, float] = {}
    for hypothesis in PRIMARY:
        result, draws = _paired_effect(
            trajectories,
            method=str(hypothesis["method"]),
            initialization=str(hypothesis["initialization"]),
            treatment=str(hypothesis["treatment"]),
            control=str(hypothesis["control"]),
            rng=rng,
        )
        identifier = str(hypothesis["id"])
        result = {
            "id": identifier,
            "label": hypothesis["label"],
            "a_query": _query(
                str(hypothesis["method"]),
                str(hypothesis["initialization"]),
                str(hypothesis["treatment"]),
            ),
            "b_query": _query(
                str(hypothesis["method"]),
                str(hypothesis["initialization"]),
                str(hypothesis["control"]),
            ),
            "minimum_effect": hypothesis["minimum_effect"],
            **result,
        }
        primary_rows.append(result)
        primary_draws[identifier] = draws
        raw_pvalues[identifier] = float(result["raw_bootstrap_pvalue"])

    adjusted, ordered = _holm_adjusted(raw_pvalues)
    rank = {identifier: index + 1 for index, identifier in enumerate(ordered)}
    count = len(primary_rows)
    for row in primary_rows:
        identifier = str(row["id"])
        index = rank[identifier] - 1
        alpha = 0.05 / (count - index)
        low, high = np.quantile(primary_draws[identifier], [alpha / 2.0, 1.0 - alpha / 2.0])
        row["holm_rank"] = rank[identifier]
        row["holm_adjusted_pvalue"] = adjusted[identifier]
        row["holm_interval_confidence"] = 1.0 - alpha
        row["holm_ci_low"] = float(low)
        row["holm_ci_high"] = float(high)
        row["effect_threshold_pass"] = int(
            float(row["point_difference"]) >= float(row["minimum_effect"])
        )
        row["adjusted_interval_excludes_zero"] = int(float(low) > 0.0)
        row["hypothesis_pass"] = int(
            bool(row["effect_threshold_pass"])
            and bool(row["adjusted_interval_excludes_zero"])
        )
    primary = pd.DataFrame(primary_rows).sort_values("id")
    primary.to_csv(OUTPUT / "primary_effects.csv", index=False)

    secondary_rows: list[dict[str, object]] = []
    secondary_rng = np.random.default_rng(BOOTSTRAP_SEED)
    for label, method, initialization, treatment, control in SECONDARY:
        result, _draws = _paired_effect(
            trajectories,
            method=method,
            initialization=initialization,
            treatment=treatment,
            control=control,
            rng=secondary_rng,
        )
        secondary_rows.append(
            {
                "comparison": label,
                "point_difference": result["point_difference"],
                "ci_low": result["raw_ci_low"],
                "ci_high": result["raw_ci_high"],
                "a_only": result["a_only"],
                "b_only": result["b_only"],
                "both": result["both"],
                "neither": result["neither"],
            }
        )
    pd.DataFrame(secondary_rows).to_csv(OUTPUT / "secondary_effects.csv", index=False)

    resources = pd.read_csv(OUTPUT / "preparation_resources.csv")
    problem_tree = resources[resources.graph == "problem_tree"]
    full = resources[resources.graph == "full"]
    resource_pairs = problem_tree.merge(full, on="instance_id", suffixes=("_problem_tree", "_full"))
    resource_pairs["angle_ratio"] = (
        resource_pairs.conditional_angle_entries_problem_tree
        / resource_pairs.conditional_angle_entries_full
    )
    resource_pairs["cnot_ratio"] = (
        resource_pairs.gray_code_cnot_upper_bound_problem_tree
        / resource_pairs.gray_code_cnot_upper_bound_full
    )
    resource_pairs.to_csv(OUTPUT / "resource_pairs.csv", index=False)
    preparation_pass = bool(
        (resource_pairs.width_problem_tree == 1).all()
        and (resource_pairs.width_full >= 2).all()
        and resource_pairs.angle_ratio.median() <= RESOURCE_RATIO_THRESHOLD
        and resource_pairs.cnot_ratio.median() <= RESOURCE_RATIO_THRESHOLD
    )
    hypothesis_results = {
        "status": "PASS" if primary.hypothesis_pass.astype(bool).all() and preparation_pass else "FAIL",
        "hypotheses": {
            row.id: bool(row.hypothesis_pass) for row in primary.itertuples(index=False)
        },
        "preparation_criterion_pass": preparation_pass,
        "material_resource_ratio_threshold": RESOURCE_RATIO_THRESHOLD,
        "median_angle_ratio_problem_tree_to_full": float(resource_pairs.angle_ratio.median()),
        "median_cnot_ratio_problem_tree_to_full": float(resource_pairs.cnot_ratio.median()),
        "full_target_biased_adam_boundary_traps": int(
            trajectories[
                (trajectories.graph == "full")
                & (trajectories.method == "adam")
                & (trajectories.initialization == "target_biased")
            ].boundary_trap.sum()
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (OUTPUT / "hypothesis_results.json").write_text(
        json.dumps(hypothesis_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(primary.to_string(index=False))
    print(json.dumps(hypothesis_results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
