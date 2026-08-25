from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS = REPOSITORY / "results" / "partial_alignment_geometry"
INSTANCES = REPOSITORY / "data" / "certificate_tight_instances"
GRAPHS = STUDY / "graphs"
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from qbm_alignment.certificate_family import all_bits, formula_edges, induced_width  # noqa: E402

EXPECTED_INSTANCES = tuple(f"ct_w{width}_i{index}" for width in (3, 4, 5, 6) for index in range(1, 6))
HELDOUT = tuple(instance for instance in EXPECTED_INSTANCES if not instance.endswith("_i1"))
CALIBRATION = tuple(instance for instance in EXPECTED_INSTANCES if instance.endswith("_i1"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_instance(path: Path) -> tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n, m, planted_weight = (int(value) for value in lines[0].split())
    planted = tuple(int(value) for value in lines[1].split())
    clauses = tuple(tuple(int(value) - 1 for value in line.split()) for line in lines[2:])
    require(len(planted) == n, f"{path}: planted length mismatch")
    require(sum(planted) == planted_weight, f"{path}: planted weight mismatch")
    require(len(clauses) == m, f"{path}: clause count mismatch")
    return planted, clauses


def validate_instances_and_graphs() -> None:
    manifest = json.loads((INSTANCES / "manifest.json").read_text(encoding="utf-8"))
    require(len(manifest) == 20, "expected twenty certificate-tight instances")
    require(tuple(row["instance_id"] for row in manifest) == EXPECTED_INSTANCES, "instance ordering mismatch")
    bits = all_bits(16)
    manifest_by_id = {row["instance_id"]: row for row in manifest}
    for instance_id in EXPECTED_INSTANCES:
        planted, clauses = read_instance(INSTANCES / f"{instance_id}.txt")
        row = manifest_by_id[instance_id]
        require(tuple(row["planted"]) == planted, f"{instance_id}: manifest planted mismatch")
        require(tuple(tuple(clause) for clause in row["clauses"]) == clauses, f"{instance_id}: clause mismatch")
        clause_array = np.asarray(clauses, dtype=np.int64)
        satisfying = np.all(bits[:, clause_array].sum(axis=2) == 1, axis=1)
        solutions = np.flatnonzero(satisfying)
        planted_index = int("".join(map(str, planted)), 2)
        require(solutions.tolist() == [planted_index], f"{instance_id}: not uniquely planted")
        edges = formula_edges(clauses)
        width, _ = induced_width(16, edges, tuple(row["order"]))
        require(width == int(row["width"]), f"{instance_id}: exact order width mismatch")
        require(len(row["witnesses"]) == len(clauses), f"{instance_id}: witness count mismatch")
        for removed, witness in enumerate(row["witnesses"]):
            assignment = np.asarray(witness, dtype=np.int8)
            require(not np.array_equal(assignment, np.asarray(planted)), f"{instance_id}: planted witness")
            for clause_index, clause in enumerate(clauses):
                occupancy = int(assignment[list(clause)].sum())
                if clause_index == removed:
                    require(occupancy != 1, f"{instance_id}: witness satisfies removed clause {removed}")
                else:
                    require(occupancy == 1, f"{instance_id}: witness violates retained clause {clause_index}")

    graph_data = json.loads((GRAPHS / "partial_graphs.json").read_text(encoding="utf-8"))
    require(set(graph_data) == set(EXPECTED_INSTANCES), "partial graph instance set mismatch")
    expected_widths = {"chain": 1, "problem_tree": 1, "width2": 2, "width3": 3}
    for instance_id, graphs in graph_data.items():
        full_edges = {tuple(edge) for edge in graphs["full"]["edges"]}
        for graph, expected_width in expected_widths.items():
            entry = graphs[graph]
            edges = tuple(tuple(edge) for edge in entry["edges"])
            order = tuple(int(value) for value in entry["order"])
            width, _ = induced_width(16, edges, order)
            require(width == expected_width, f"{instance_id}/{graph}: width mismatch")
            require(int(entry["compiled_width"]) == expected_width, f"{instance_id}/{graph}: stored width mismatch")
            if graph != "chain":
                require(set(edges).issubset(full_edges), f"{instance_id}/{graph}: edge outside target graph")
        require(len(graphs["problem_tree"]["edges"]) == 15, f"{instance_id}: tree is not spanning")


def validate_primary_results() -> None:
    trajectories = pd.read_csv(RESULTS / "partial_alignment_trajectories.csv")
    require(len(trajectories) == 3200, "partial trajectory row count mismatch")
    require(set(trajectories.instance_id) == set(EXPECTED_INSTANCES), "partial trajectory instance mismatch")
    require(set(trajectories.split) == {"calibration", "evaluation"}, "split values mismatch")
    require(set(trajectories.graph) == {"chain", "problem_tree", "width2", "width3"}, "graph values mismatch")
    require((trajectories.success.isin([0, 1])).all(), "invalid success flags")
    require((trajectories.minimum_gap >= -1e-10).all(), "negative minimum gap")
    require((trajectories.total_samples >= 0).all(), "negative sample count")

    primary = pd.read_csv(RESULTS / "primary_summary.csv")
    expected = {
        ("chain", "exact_natural_oracle"): (80, 1),
        ("chain", "sampled_full_fisher"): (80, 0),
        ("problem_tree", "exact_natural_oracle"): (80, 76),
        ("problem_tree", "sampled_adam"): (80, 61),
        ("problem_tree", "sampled_full_fisher"): (80, 75),
        ("problem_tree", "sampled_two_block_fisher"): (80, 73),
        ("problem_tree", "sampled_bag_fisher"): (80, 73),
        ("width2", "exact_natural_oracle"): (80, 71),
        ("width2", "sampled_adam"): (80, 32),
        ("width2", "sampled_full_fisher"): (80, 57),
        ("width2", "protected_ray_star"): (80, 46),
        ("width3", "exact_natural_oracle"): (80, 80),
        ("width3", "sampled_adam"): (80, 31),
        ("width3", "sampled_full_fisher"): (80, 66),
        ("width3", "protected_ray_star"): (80, 71),
        ("width3", "sampled_bag_fisher"): (80, 45),
    }
    indexed = primary.set_index(["graph", "method"])
    for key, (trials, successes) in expected.items():
        require(key in indexed.index, f"missing primary summary cell {key}")
        row = indexed.loc[key]
        require(int(row.trajectories) == trials and int(row.successes) == successes, f"primary result mismatch {key}")
        require(abs(float(row.success_rate) - successes / trials) < 1e-12, f"rate mismatch {key}")

    oracle = pd.read_csv(RESULTS / "exact_natural_oracle.csv")
    require(len(oracle) == 400, "exact oracle row count mismatch")
    held = oracle[oracle.split == "evaluation"]
    oracle_success = held.groupby("graph").success.sum().to_dict()
    require(oracle_success == {"chain": 1, "problem_tree": 76, "width2": 71, "width3": 80}, "oracle success mismatch")

    chain = pd.read_csv(RESULTS / "chain_representability_control.csv")
    require(len(chain) == 20, "chain representability row count mismatch")
    require(float(np.max(np.abs(chain.full_gap_check - 0.1))) < 3e-12, "chain gap control mismatch")
    require(float(chain.planted_probability.min()) >= 0.952, "chain planted probability too low")
    require(float(chain.probability_residual.abs().max()) < 5e-12, "chain analytic probability residual")


def validate_geometry_and_sampling_controls() -> None:
    metrics = pd.read_csv(RESULTS / "partial_graph_metrics_enriched.csv")
    require(len(metrics) == 100, "graph metrics row count mismatch")
    full = metrics[metrics.graph == "full"]
    require(np.allclose(full.explained_variance_fraction, 1.0, atol=3e-12, rtol=0), "full explained variance mismatch")
    require(float(full.projected_identity_relative_residual.abs().max()) < 2e-12, "full identity residual")
    graph_means = metrics.groupby("graph").agg(
        pair=("pair_weight_fraction", "mean"),
        explained=("explained_variance_fraction", "mean"),
        cosine=("exact_ng_projected_target_cosine", "mean"),
    )
    require(float(graph_means.loc["chain", "cosine"]) < -0.7, "chain target direction should be anti-aligned")
    require(float(graph_means.loc["width3", "cosine"]) > 0.98, "width3 target alignment too low")

    first = pd.read_csv(RESULTS / "first_step_diagnostics.csv")
    require(len(first) == 2400, "first-step diagnostic row count mismatch")
    exact_pre = pd.read_csv(RESULTS / "exact_preconditioner_diagnostics.csv")
    require(len(exact_pre) == 2400, "exact preconditioner diagnostic row count mismatch")

    independent = pd.read_csv(RESULTS / "independent_full_fisher.csv")
    equal = pd.read_csv(RESULTS / "independent_equal_full_fisher.csv")
    require(len(independent) == 400, "independent split row count mismatch")
    require(len(equal) == 300, "independent equal row count mismatch")
    controls = pd.read_csv(RESULTS / "full_fisher_batch_controls.csv")
    lookup = controls.set_index(["graph", "nominal_budget", "estimator"])
    expected = {
        ("problem_tree", 256, "same_batch"): 0.9375,
        ("problem_tree", 256, "independent_split_total"): 0.8875,
        ("problem_tree", 256, "independent_equal_batches"): 0.9125,
        ("width3", 64, "same_batch"): 0.675,
        ("width3", 64, "independent_split_total"): 0.225,
        ("width3", 64, "independent_equal_batches"): 0.5625,
        ("width3", 1024, "same_batch"): 0.875,
        ("width3", 1024, "independent_split_total"): 0.875,
    }
    for key, value in expected.items():
        require(key in lookup.index, f"missing batching control {key}")
        require(abs(float(lookup.loc[key].success_rate) - value) < 1e-12, f"batching control mismatch {key}")

    bag = pd.read_csv(RESULTS / "sampled_bag_fisher.csv")
    require(len(bag) == 400, "bag Fisher trajectory count mismatch")
    held_bag = bag[bag.split == "evaluation"].groupby("graph").success.sum().to_dict()
    require(held_bag == {"chain": 0, "problem_tree": 73, "width2": 37, "width3": 45}, "bag Fisher success mismatch")
    bag_diag = pd.read_csv(RESULTS / "bag_fisher_diagnostic.csv")
    require(len(bag_diag) == 400, "bag diagnostic row count mismatch")


def validate_documents_and_figures() -> None:
    required_docs = [
        "README.md",
        "protocol.md",
        "partial_alignment_identity.md",
        "bag_fisher_note.md",
    ]
    for name in required_docs:
        path = STUDY / name
        require(path.is_file() and path.stat().st_size > 100, f"missing document {name}")
    selected_figures = [
        "success_by_graph",
        "oracle_and_sampled_ceiling",
        "same_vs_independent_partial",
        "graph_alignment_metrics",
        "width3_sample_scaling",
        "storage_success_frontier",
        "chain_representability_vs_optimization",
        "first_direction_alignment",
    ]
    for stem in selected_figures:
        for suffix in (".png", ".pdf"):
            path = STUDY / "figures" / f"{stem}{suffix}"
            require(path.is_file() and path.stat().st_size > 1000, f"missing figure {path.name}")


def main() -> int:
    validate_instances_and_graphs()
    validate_primary_results()
    validate_geometry_and_sampling_controls()
    validate_documents_and_figures()
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    require(summary["heldout_instances"] == 16, "heldout summary mismatch")
    require(summary["parameter_seeds"] == 5, "seed summary mismatch")
    report = {
        "status": "PASS",
        "certificate_tight_instances": 20,
        "sampled_trajectories": 3200,
        "exact_natural_oracle_trajectories": 400,
        "independent_batch_controls": 700,
        "bag_fisher_trajectories": 400,
        "heldout_instances": 16,
    }
    (RESULTS / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Partial-alignment study validation passed.")
    print("  certificate-tight instances: 20 exact unique")
    print("  sampled trajectories: 3,200")
    print("  exact natural oracle trajectories: 400")
    print("  independent-batch controls: 700")
    print("  bag-Fisher trajectories: 400")
    print("  held-out instances: 16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
