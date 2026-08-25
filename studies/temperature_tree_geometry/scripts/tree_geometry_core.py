#!/usr/bin/env python3
"""Core exact-enumeration utilities for temperature-dependent tree geometry."""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

RCOND = 1.0e-12
Q_TIE_REL = 1.0e-10
Q_TIE_ABS = 1.0e-12
KL_TIE_REL = 1.0e-10
KL_TIE_ABS = 1.0e-12
NEAR_OPT_FRACTION = 0.01
S_GRID = np.round(np.arange(0.0, 1.5000001, 0.025), 12)
EXPECTED_TOTAL_TREES = 20_812

@dataclass(frozen=True)
class Instance:
    name: str
    n: int
    m: int
    planted: tuple[int, ...]
    clauses: tuple[tuple[int, int, int], ...]
    bits: np.ndarray
    z: np.ndarray
    cost: np.ndarray
    ground_energy: float
    spectral_gap: float
    h: np.ndarray
    edges: tuple[tuple[int, int], ...]
    couplings: np.ndarray
    features: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class TreeFamily:
    trees: tuple[tuple[tuple[int, int], ...], ...]
    edge_indices: np.ndarray
    incidence: np.ndarray
    feature_indices: np.ndarray
    edge_strings: tuple[str, ...]
    tree_hashes: tuple[str, ...]
    hot_total_score: np.ndarray
    hot_mask: np.ndarray
    maxj_lex_index: int


def parse_instance(path: Path) -> Instance:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Malformed instance: {path}")
    n, m, _declared_weight = map(int, lines[0].split())
    planted = tuple(map(int, lines[1].split()))
    clauses = tuple(tuple(int(token) - 1 for token in line.split()) for line in lines[2:])
    if len(planted) != n:
        raise ValueError(f"Planted assignment length mismatch in {path}")
    if len(clauses) != m:
        raise ValueError(f"Clause count mismatch in {path}")
    if any(len(clause) != 3 or len(set(clause)) != 3 for clause in clauses):
        raise ValueError(f"Clauses must contain three distinct variables in {path}")

    states = np.arange(2**n, dtype=np.uint16)
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint16)
    bits = ((states[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
    z = (1 - 2 * bits).astype(np.int8)

    cost = np.zeros(2**n, dtype=np.float64)
    h = np.zeros(n, dtype=np.float64)
    coupling_map: dict[tuple[int, int], float] = {}
    for clause in clauses:
        occupancy = bits[:, clause].sum(axis=1)
        cost += (occupancy - 1) ** 2 - 1
        for i in clause:
            h[i] -= 0.5
        for edge in itertools.combinations(clause, 2):
            normalized = tuple(sorted(edge))
            coupling_map[normalized] = coupling_map.get(normalized, 0.0) + 0.5

    edges = tuple(sorted(coupling_map))
    couplings = np.asarray([coupling_map[edge] for edge in edges], dtype=np.float64)
    pair_features = np.column_stack([z[:, i] * z[:, j] for i, j in edges]).astype(np.float64)
    features = np.column_stack([z.astype(np.float64), pair_features])
    coefficients = np.concatenate([h, couplings])
    residual = float(np.max(np.abs(features @ coefficients - cost)))
    if residual > 1.0e-12:
        raise AssertionError(f"Ising conversion residual {residual:g} in {path}")

    unique_energies = np.unique(cost)
    ground_energy = float(unique_energies[0])
    if unique_energies.size < 2:
        raise AssertionError(f"No excited level in {path}")
    spectral_gap = float(unique_energies[1] - unique_energies[0])

    return Instance(
        name=path.stem,
        n=n,
        m=m,
        planted=planted,
        clauses=clauses,
        bits=bits,
        z=z,
        cost=cost,
        ground_energy=ground_energy,
        spectral_gap=spectral_gap,
        h=h,
        edges=edges,
        couplings=couplings,
        features=features,
        coefficients=coefficients,
    )


def _is_tree(n: int, edges: Sequence[tuple[int, int]]) -> bool:
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            return False
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1
    return len(edges) == n - 1


def enumerate_tree_family(instance: Instance) -> TreeFamily:
    trees = tuple(
        tuple(combination)
        for combination in itertools.combinations(instance.edges, instance.n - 1)
        if _is_tree(instance.n, combination)
    )
    if not trees:
        raise AssertionError(f"Target graph is disconnected for {instance.name}")

    edge_to_index = {edge: index for index, edge in enumerate(instance.edges)}
    edge_indices = np.empty((len(trees), instance.n - 1), dtype=np.int16)
    incidence = np.zeros((len(trees), len(instance.edges)), dtype=np.float64)
    feature_indices = np.empty((len(trees), 2 * instance.n - 1), dtype=np.int16)
    edge_strings: list[str] = []
    tree_hashes: list[str] = []

    for tree_index, tree in enumerate(trees):
        indices = np.asarray([edge_to_index[edge] for edge in tree], dtype=np.int16)
        edge_indices[tree_index] = indices
        incidence[tree_index, indices] = 1.0
        feature_indices[tree_index] = np.concatenate(
            [np.arange(instance.n, dtype=np.int16), instance.n + indices]
        )
        edge_string = ";".join(f"{i}-{j}" for i, j in tree)
        edge_strings.append(edge_string)
        digest = hashlib.sha256(f"{instance.name}|{edge_string}".encode("utf-8")).hexdigest()[:16]
        tree_hashes.append(digest)

    hot_total_score = float(instance.h @ instance.h) + incidence @ (instance.couplings**2)
    hot_max = float(np.max(hot_total_score))
    hot_mask = hot_total_score >= hot_max - Q_TIE_ABS
    maxj_lex_index = int(np.flatnonzero(hot_mask)[0])

    return TreeFamily(
        trees=trees,
        edge_indices=edge_indices,
        incidence=incidence,
        feature_indices=feature_indices,
        edge_strings=tuple(edge_strings),
        tree_hashes=tuple(tree_hashes),
        hot_total_score=hot_total_score,
        hot_mask=hot_mask,
        maxj_lex_index=maxj_lex_index,
    )


def gibbs_probability(cost: np.ndarray, beta: float) -> np.ndarray:
    logits = -float(beta) * np.asarray(cost, dtype=np.float64)
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    return weights / float(np.sum(weights))


def find_beta_cert(instance: Instance) -> float:
    def residual(beta: float) -> float:
        probability = gibbs_probability(instance.cost, beta)
        return float(probability @ instance.cost - instance.ground_energy - 0.1)

    if residual(0.0) <= 0.0:
        return 0.0
    upper = 1.0
    while residual(upper) > 0.0:
        upper *= 2.0
        if upper > 1024.0:
            raise RuntimeError(f"Could not bracket beta_cert for {instance.name}")
    return float(brentq(residual, 0.0, upper, xtol=1.0e-13, rtol=1.0e-13))


def batch_pinv_quadratic(
    matrices: np.ndarray,
    vectors: np.ndarray,
    *,
    rcond: float = RCOND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return v^T M^+ v, smallest retained eigenvalue, rank, condition number."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    largest = np.maximum(eigenvalues[:, -1], 0.0)
    cutoff = rcond * largest
    mask = eigenvalues > cutoff[:, None]
    coefficients = np.einsum("tik,ti->tk", eigenvectors, vectors, optimize=True)
    inverse = np.zeros_like(eigenvalues)
    np.divide(1.0, eigenvalues, out=inverse, where=mask)
    quadratic = np.sum(coefficients * coefficients * inverse, axis=1)
    retained = np.where(mask, eigenvalues, np.inf)
    smallest = np.min(retained, axis=1)
    smallest[~np.isfinite(smallest)] = np.nan
    rank = np.sum(mask, axis=1).astype(np.int16)
    condition = np.divide(
        largest,
        smallest,
        out=np.full_like(largest, np.nan),
        where=np.isfinite(smallest) & (smallest > 0.0),
    )
    return quadratic, smallest, rank, condition


def tree_projection_batch(
    probability: np.ndarray,
    instance: Instance,
    family: TreeFamily,
) -> tuple[np.ndarray, np.ndarray, float]:
    bits = instance.bits
    state_count = bits.shape[0]
    node_marginals = np.empty((instance.n, 2), dtype=np.float64)
    for i in range(instance.n):
        node_marginals[i] = np.bincount(bits[:, i], weights=probability, minlength=2)

    base = np.zeros(state_count, dtype=np.float64)
    for i in range(instance.n):
        base += np.log(node_marginals[i, bits[:, i]])

    edge_adjustment = np.empty((len(instance.edges), state_count), dtype=np.float64)
    for edge_index, (i, j) in enumerate(instance.edges):
        code = 2 * bits[:, i] + bits[:, j]
        pair_marginal = np.bincount(code, weights=probability, minlength=4).reshape(2, 2)
        edge_adjustment[edge_index] = (
            np.log(pair_marginal[bits[:, i], bits[:, j]])
            - np.log(node_marginals[i, bits[:, i]])
            - np.log(node_marginals[j, bits[:, j]])
        )

    raw_log_probability = base[None, :] + family.incidence @ edge_adjustment
    raw_log_normalization = logsumexp(raw_log_probability, axis=1)
    log_probability = raw_log_probability - raw_log_normalization[:, None]
    projected = np.exp(log_probability)
    return log_probability, projected, float(np.max(np.abs(raw_log_normalization)))


def exact_target_geometry(
    probability: np.ndarray,
    instance: Instance,
    family: TreeFamily,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    feature_mean = probability @ instance.features
    second_moment = (instance.features * probability[:, None]).T @ instance.features
    full_covariance = second_moment - np.outer(feature_mean, feature_mean)
    target_energy = float(probability @ instance.cost)
    cost_feature_mean = (instance.features * (probability * instance.cost)[:, None]).sum(axis=0)
    full_b = cost_feature_mean - feature_mean * target_energy

    indices = family.feature_indices
    tree_covariance = full_covariance[indices[:, :, None], indices[:, None, :]]
    tree_b = full_b[indices]
    retained_power, _smallest, _rank, _condition = batch_pinv_quadratic(tree_covariance, tree_b)
    full_power = float(probability @ (instance.cost**2) - target_energy**2)
    return retained_power, tree_b, full_power, feature_mean, full_covariance


def longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
