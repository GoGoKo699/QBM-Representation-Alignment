#!/usr/bin/env python3
"""Regenerate preparation-resource tables from stored elimination-order proofs.

Use ``--recompute-exact-orders`` to rerun the expensive exact treewidth search
for every full graph.  The default verifies and reuses the packaged exact
orders, making routine reproduction much faster.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "src"))

from qbm_alignment.sparse_ising import (
    N,
    build_problem,
    exact_min_width_order,
    induced_width,
    load_manifest,
)


def tree_resources() -> dict[str, object]:
    return {
        "width": 1,
        "conditional_angle_entries": 2 * N - 1,
        "gray_code_cnot_upper_bound": 2 * N - 2,
        "order": "tree_leaf_order",
        "elimination_degrees": "1 " * (N - 1) + "0",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute-exact-orders",
        action="store_true",
        help="rerun exact subset-DP treewidth searches for all full graphs",
    )
    args = parser.parse_args()

    output = REPOSITORY / "results" / "confirmatory" / "preparation_resources.csv"
    proof_table = pd.read_csv(output).set_index(["instance_id", "graph"]) if output.is_file() else None
    instances = load_manifest(EXPERIMENT / "instances" / "manifest.json")
    rows: list[dict[str, object]] = []
    for instance in instances:
        problem = build_problem(instance, compute_resources=False)
        for graph, representation in problem.representations.items():
            if graph != "full":
                resources = tree_resources()
            else:
                if args.recompute_exact_orders or proof_table is None:
                    order = exact_min_width_order(N, representation.edges)
                else:
                    stored = str(proof_table.loc[(instance.instance_id, graph), "order"])
                    order = tuple(int(value) for value in stored.split())
                width, degrees = induced_width(N, representation.edges, order)
                angles = int(sum(1 << degree for degree in degrees))
                resources = {
                    "width": width,
                    "conditional_angle_entries": angles,
                    "gray_code_cnot_upper_bound": angles - 1,
                    "order": " ".join(map(str, order)),
                    "elimination_degrees": " ".join(map(str, degrees)),
                }
            rows.append(
                {
                    "instance_id": instance.instance_id,
                    "graph": graph,
                    "edge_count": len(representation.edges),
                    "parameter_count": representation.features.shape[1],
                    **resources,
                }
            )
        print(f"resources {instance.instance_id}", flush=True)
    table = pd.DataFrame(rows).sort_values(["instance_id", "graph"])
    table.to_csv(output, index=False)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
