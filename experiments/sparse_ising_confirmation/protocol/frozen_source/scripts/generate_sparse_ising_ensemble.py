#!/usr/bin/env python3
"""Generate the preregistered weighted sparse-Ising ensemble.

This script implements only instance generation and exact spectral checks. It
contains no QBM optimizer and cannot select instances by optimization outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np

N = 16
DEGREE = 3
MIN_GAP = 0.05
COUPLING_MIN = 0.5
COUPLING_MAX = 1.5
FIELD_ABS_MAX = 0.35
MAX_ATTEMPTS = 10000


@dataclass(frozen=True)
class IsingInstance:
    instance_id: str
    base_seed: int
    accepted_attempt: int
    n: int
    degree: int
    edges: list[list[int]]
    fields: list[float]
    couplings: list[float]
    ground_index: int
    ground_spins: list[int]
    ground_energy: float
    spectral_gap: float
    coefficient_rms_before_normalization: float
    coefficient_rms_after_normalization: float
    graph_seed: int
    coefficient_seed: int
    relabel_seed: int
    random_tree_seed: int


def seed_stream(base_seed: int, attempt: int) -> tuple[int, int, int, int]:
    sequence = np.random.SeedSequence([int(base_seed), int(attempt), 20260825])
    children = sequence.spawn(4)
    return tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)  # type: ignore[return-value]


def exact_spectrum(n: int, fields: np.ndarray, edges: list[tuple[int, int]], couplings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    integers = np.arange(1 << n, dtype=np.uint64)[:, None]
    shifts = np.arange(n - 1, -1, -1, dtype=np.uint64)
    bits = ((integers >> shifts) & 1).astype(np.int8)
    spins = 1 - 2 * bits
    energy = spins @ fields
    for (left, right), coupling in zip(edges, couplings, strict=True):
        energy = energy + coupling * spins[:, left] * spins[:, right]
    return spins, np.asarray(energy, dtype=np.float64)


def make_instance(base_seed: int, instance_id: str) -> IsingInstance:
    for attempt in range(MAX_ATTEMPTS):
        graph_seed, coefficient_seed, relabel_seed, random_tree_seed = seed_stream(base_seed, attempt)
        graph = nx.random_regular_graph(DEGREE, N, seed=graph_seed)
        if not nx.is_connected(graph):
            continue

        relabel_rng = np.random.default_rng(relabel_seed)
        permutation = relabel_rng.permutation(N)
        mapping = {old: int(permutation[old]) for old in range(N)}
        graph = nx.relabel_nodes(graph, mapping, copy=True)
        edges = sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in graph.edges())

        rng = np.random.default_rng(coefficient_seed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(edges))
        magnitudes = rng.uniform(COUPLING_MIN, COUPLING_MAX, size=len(edges))
        couplings = signs * magnitudes
        fields = rng.uniform(-FIELD_ABS_MAX, FIELD_ABS_MAX, size=N)

        coefficients = np.concatenate([fields, couplings])
        rms_before = float(np.sqrt(np.mean(coefficients * coefficients)))
        fields = fields / rms_before
        couplings = couplings / rms_before

        spins, energy = exact_spectrum(N, fields, edges, couplings)
        order = np.argsort(energy, kind="stable")
        ground_index = int(order[0])
        ground_energy = float(energy[ground_index])
        degeneracy = int(np.sum(np.isclose(energy, ground_energy, atol=1e-12, rtol=0.0)))
        if degeneracy != 1:
            continue
        spectral_gap = float(energy[int(order[1])] - ground_energy)
        if spectral_gap < MIN_GAP:
            continue

        rms_after = float(np.sqrt(np.mean(np.concatenate([fields, couplings]) ** 2)))
        return IsingInstance(
            instance_id=instance_id,
            base_seed=int(base_seed),
            accepted_attempt=attempt,
            n=N,
            degree=DEGREE,
            edges=[[int(a), int(b)] for a, b in edges],
            fields=[float(value) for value in fields],
            couplings=[float(value) for value in couplings],
            ground_index=ground_index,
            ground_spins=[int(value) for value in spins[ground_index]],
            ground_energy=ground_energy,
            spectral_gap=spectral_gap,
            coefficient_rms_before_normalization=rms_before,
            coefficient_rms_after_normalization=rms_after,
            graph_seed=graph_seed,
            coefficient_seed=coefficient_seed,
            relabel_seed=relabel_seed,
            random_tree_seed=random_tree_seed,
        )
    raise RuntimeError(f"no accepted instance for seed {base_seed} after {MAX_ATTEMPTS} attempts")


def canonical_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commitment", type=Path, default=Path(__file__).resolve().parents[1] / "CONFIRMATORY_SEED_COMMITMENT.json")
    parser.add_argument("--mode", choices=("engineering", "confirmatory"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    commitment = json.loads(args.commitment.read_text(encoding="utf-8"))
    key = "engineering_instance_seeds" if args.mode == "engineering" else "confirmatory_instance_seeds"
    seeds = [int(value) for value in commitment[key]]
    records = [make_instance(seed, f"ising_{args.mode}_{index + 1:02d}") for index, seed in enumerate(seeds)]
    payload = {
        "protocol": "weighted connected 3-regular Ising, n=16",
        "mode": args.mode,
        "acceptance": {"unique_ground_state": True, "minimum_gap": MIN_GAP, "coefficient_rms": 1.0},
        "instances": [asdict(record) for record in records],
    }
    payload["manifest_sha256"] = canonical_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Instances: {len(records)}")
    print(f"Manifest SHA-256: {payload['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
