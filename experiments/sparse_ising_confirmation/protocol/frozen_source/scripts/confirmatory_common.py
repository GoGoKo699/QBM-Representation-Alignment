from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numba import njit

N = 16
STEPS = 200
PARAMETER_SEEDS = (0, 19, 42, 50, 101)
LR = 0.02
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8
STEP_CAP = 0.5
ARMIJO = 1e-4
STAGNATION_WINDOW = 15
STAGNATION_TOL = 1e-10
EIGEN_CUT = 1e-12
PSEUDOINVERSE_CUT = 1e-10

ALL_PAIRS = tuple(itertools.combinations(range(N), 2))
PAIR_INDEX = {edge: index for index, edge in enumerate(ALL_PAIRS)}
FIELD_MASKS = np.asarray([1 << (N - 1 - index) for index in range(N)], dtype=np.int64)
PAIR_MASKS = {edge: FIELD_MASKS[edge[0]] ^ FIELD_MASKS[edge[1]] for edge in ALL_PAIRS}


def all_bits_and_spins(n: int = N) -> tuple[np.ndarray, np.ndarray]:
    integers = np.arange(1 << n, dtype=np.uint64)[:, None]
    shifts = np.arange(n - 1, -1, -1, dtype=np.uint64)
    bits = ((integers >> shifts) & 1).astype(np.int8)
    spins = (1 - 2 * bits).astype(np.float64)
    return bits, spins


BITS, SPINS = all_bits_and_spins()


@njit(cache=True)
def _fwht_inplace_numba(output: np.ndarray) -> None:
    count = output.shape[0]
    batch = output.shape[1]
    width = 1
    while width < count:
        block = 2 * width
        for begin in range(0, count, block):
            for offset in range(width):
                left_index = begin + offset
                right_index = left_index + width
                for column in range(batch):
                    left = output[left_index, column]
                    right = output[right_index, column]
                    output[left_index, column] = left + right
                    output[right_index, column] = left - right
        width *= 2


def fwht_batch(values: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform along axis 0."""
    output = np.asarray(values, dtype=np.float64).copy()
    if output.ndim == 1:
        output = output[:, None]
    _fwht_inplace_numba(output)
    return output


def feature_masks(edges: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.concatenate(
        [FIELD_MASKS, np.asarray([PAIR_MASKS[tuple(edge)] for edge in edges], dtype=np.int64)]
    )


def feature_matrix(edges: Sequence[tuple[int, int]]) -> np.ndarray:
    columns = [SPINS]
    if edges:
        columns.append(
            np.column_stack([SPINS[:, left] * SPINS[:, right] for left, right in edges])
        )
    return np.column_stack(columns).astype(np.float64, copy=False)


def canonical_noise(seed: int, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    full = 0.3 * rng.standard_normal(N + len(ALL_PAIRS))
    active = np.asarray(
        list(range(N)) + [N + PAIR_INDEX[tuple(edge)] for edge in edges], dtype=np.int64
    )
    return full[active]


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True


def kruskal_tree(
    edges: Sequence[tuple[int, int]],
    keys: Sequence[float],
    *,
    descending: bool,
) -> tuple[tuple[int, int], ...]:
    if len(edges) != len(keys):
        raise ValueError("edge/key length mismatch")
    if descending:
        order = sorted(range(len(edges)), key=lambda i: (-float(keys[i]), tuple(edges[i])))
    else:
        order = sorted(range(len(edges)), key=lambda i: (float(keys[i]), tuple(edges[i])))
    dsu = DisjointSet(N)
    selected: list[tuple[int, int]] = []
    for index in order:
        edge = tuple(int(value) for value in edges[index])
        if dsu.union(*edge):
            selected.append(edge)
            if len(selected) == N - 1:
                break
    if len(selected) != N - 1:
        raise RuntimeError("target graph is not connected")
    return tuple(sorted(selected))


def random_problem_tree(
    edges: Sequence[tuple[int, int]], random_tree_seed: int
) -> tuple[tuple[int, int], ...]:
    rng = np.random.default_rng(int(random_tree_seed))
    priorities = rng.random(len(edges))
    return kruskal_tree(edges, priorities, descending=False)


def maximum_weight_problem_tree(
    edges: Sequence[tuple[int, int]], couplings: Sequence[float]
) -> tuple[tuple[int, int], ...]:
    return kruskal_tree(edges, np.abs(np.asarray(couplings, dtype=np.float64)), descending=True)


def min_fill_order(n: int, edges: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    adjacency = {index: set() for index in range(n)}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(range(n))
    order: list[int] = []
    while remaining:
        options = []
        for vertex in remaining:
            neighbors = sorted(adjacency[vertex] & remaining)
            fill = sum(
                right not in adjacency[left]
                for i, left in enumerate(neighbors)
                for right in neighbors[i + 1 :]
            )
            options.append((fill, len(neighbors), vertex))
        _fill, _degree, vertex = min(options)
        neighbors = list(adjacency[vertex] & remaining)
        for i, left in enumerate(neighbors):
            for right in neighbors[i + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        remaining.remove(vertex)
        order.append(vertex)
    return tuple(order)


def induced_width(
    n: int, edges: Sequence[tuple[int, int]], order: Sequence[int]
) -> tuple[int, tuple[int, ...]]:
    adjacency = {index: set() for index in range(n)}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(range(n))
    width = 0
    degrees: list[int] = []
    for vertex in order:
        neighbors = list(adjacency[vertex] & remaining)
        width = max(width, len(neighbors))
        degrees.append(len(neighbors))
        for i, left in enumerate(neighbors):
            for right in neighbors[i + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        remaining.remove(vertex)
    return width, tuple(degrees)


def exact_min_width_order(
    n: int, edges: Sequence[tuple[int, int]]
) -> tuple[int, ...]:
    adjacency = [0] * n
    for left, right in edges:
        adjacency[left] |= 1 << right
        adjacency[right] |= 1 << left
    full = (1 << n) - 1
    size = 1 << n
    best_width = np.full(size, n + 1, dtype=np.int16)
    best_cost = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
    parent_vertex = np.full(size, -1, dtype=np.int16)
    parent_subset = np.full(size, -1, dtype=np.int32)
    best_width[0] = 0
    best_cost[0] = 0
    cache: dict[tuple[int, int], int] = {}

    def degree(subset: int, vertex: int) -> int:
        key = (subset, vertex)
        if key in cache:
            return cache[key]
        remaining = full ^ subset
        reachable = adjacency[vertex] & subset
        frontier = reachable
        while frontier:
            bit = frontier & -frontier
            frontier -= bit
            node = bit.bit_length() - 1
            new = adjacency[node] & subset & ~reachable
            reachable |= new
            frontier |= new
        neighbors = adjacency[vertex] & remaining
        current = reachable
        while current:
            bit = current & -current
            current -= bit
            node = bit.bit_length() - 1
            neighbors |= adjacency[node] & remaining
        neighbors &= ~(1 << vertex)
        value = neighbors.bit_count()
        cache[key] = value
        return value

    for subset in range(size):
        if best_width[subset] > n:
            continue
        remaining = full ^ subset
        current = remaining
        while current:
            bit = current & -current
            current -= bit
            vertex = bit.bit_length() - 1
            local_degree = degree(subset, vertex)
            target = subset | bit
            width = max(int(best_width[subset]), local_degree)
            cost = int(best_cost[subset]) + (1 << (local_degree + 1))
            if width < best_width[target] or (
                width == best_width[target] and cost < best_cost[target]
            ):
                best_width[target] = width
                best_cost[target] = cost
                parent_vertex[target] = vertex
                parent_subset[target] = subset
    reverse: list[int] = []
    subset = full
    while subset:
        vertex = int(parent_vertex[subset])
        reverse.append(vertex)
        subset = int(parent_subset[subset])
    return tuple(reversed(reverse))


def preparation_resources(edges: Sequence[tuple[int, int]]) -> dict[str, object]:
    if len(edges) == N - 1:
        # Any tree has a width-one leaf elimination order; exact compiler counts are fixed.
        return {
            "width": 1,
            "conditional_angle_entries": 2 * N - 1,
            "gray_code_cnot_upper_bound": 2 * N - 2,
            "order": "tree_leaf_order",
            "elimination_degrees": "1 " * (N - 1) + "0",
        }
    order = exact_min_width_order(N, edges)
    width, degrees = induced_width(N, edges, order)
    angles = int(sum(1 << degree for degree in degrees))
    return {
        "width": width,
        "conditional_angle_entries": angles,
        "gray_code_cnot_upper_bound": angles - 1,
        "order": " ".join(map(str, order)),
        "elimination_degrees": " ".join(map(str, degrees)),
    }


@dataclass(frozen=True)
class InstanceData:
    instance_id: str
    fields: np.ndarray
    target_edges: tuple[tuple[int, int], ...]
    target_couplings: np.ndarray
    ground_index: int
    ground_spins: np.ndarray
    ground_energy: float
    spectral_gap: float
    random_tree_seed: int


@dataclass(frozen=True)
class Representation:
    name: str
    edges: tuple[tuple[int, int], ...]
    features: np.ndarray
    masks: np.ndarray
    coefficients: np.ndarray
    cross_active_full: np.ndarray
    cross_active_active: np.ndarray
    resources: dict[str, object]


@dataclass(frozen=True)
class Problem:
    instance: InstanceData
    full_masks: np.ndarray
    full_coefficients: np.ndarray
    cost: np.ndarray
    local_minimum: np.ndarray
    representations: dict[str, Representation]


def load_manifest(path: Path) -> list[InstanceData]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for item in payload["instances"]:
        records.append(
            InstanceData(
                instance_id=str(item["instance_id"]),
                fields=np.asarray(item["fields"], dtype=np.float64),
                target_edges=tuple(tuple(int(v) for v in edge) for edge in item["edges"]),
                target_couplings=np.asarray(item["couplings"], dtype=np.float64),
                ground_index=int(item["ground_index"]),
                ground_spins=np.asarray(item["ground_spins"], dtype=np.int8),
                ground_energy=float(item["ground_energy"]),
                spectral_gap=float(item["spectral_gap"]),
                random_tree_seed=int(item["random_tree_seed"]),
            )
        )
    return records


def build_problem(instance: InstanceData, *, compute_resources: bool = False) -> Problem:
    coupling_map = {
        edge: float(value)
        for edge, value in zip(instance.target_edges, instance.target_couplings, strict=True)
    }
    random_tree = random_problem_tree(instance.target_edges, instance.random_tree_seed)
    maximum_tree = maximum_weight_problem_tree(instance.target_edges, instance.target_couplings)
    graph_edges = {
        "chain": tuple((index, index + 1) for index in range(N - 1)),
        "random_tree": random_tree,
        "problem_tree": maximum_tree,
        "full": instance.target_edges,
    }
    full_masks = feature_masks(instance.target_edges)
    full_coefficients = np.concatenate([instance.fields, instance.target_couplings])
    full_features = feature_matrix(instance.target_edges)
    cost = full_features @ full_coefficients
    if abs(float(cost[instance.ground_index]) - instance.ground_energy) > 2e-10:
        raise RuntimeError(f"ground energy mismatch for {instance.instance_id}")
    local = np.ones(1 << N, dtype=bool)
    for state in range(1 << N):
        energy = cost[state]
        for variable in range(N):
            neighbor = state ^ (1 << (N - 1 - variable))
            if cost[neighbor] < energy - 1e-12:
                local[state] = False
                break
    representations: dict[str, Representation] = {}
    for name, edges in graph_edges.items():
        masks = feature_masks(edges)
        coefficients = np.concatenate(
            [instance.fields, np.asarray([coupling_map.get(edge, 0.0) for edge in edges])]
        )
        representations[name] = Representation(
            name=name,
            edges=edges,
            features=feature_matrix(edges),
            masks=masks,
            coefficients=coefficients,
            cross_active_full=np.bitwise_xor(masks[:, None], full_masks[None, :]),
            cross_active_active=np.bitwise_xor(masks[:, None], masks[None, :]),
            resources=(preparation_resources(edges) if compute_resources else {"width": 1 if len(edges) == N - 1 else None}),
        )
    return Problem(instance, full_masks, full_coefficients, cost, local, representations)


def probabilities(theta: np.ndarray, representation: Representation) -> np.ndarray:
    score = representation.features @ theta
    logits = -score
    logits -= float(np.max(logits))
    weight = np.exp(np.clip(logits, -745.0, 0.0))
    return weight / weight.sum()


def exact_state(
    theta: np.ndarray,
    problem: Problem,
    representation: Representation,
    *,
    want_fisher: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    probability = probabilities(theta, representation)
    moments = fwht_batch(probability)[:, 0]
    active_mean = moments[representation.masks]
    full_mean = moments[problem.full_masks]
    cross = moments[representation.cross_active_full]
    covariance_cost = cross - active_mean[:, None] * full_mean[None, :]
    gradient = -(covariance_cost @ problem.full_coefficients)
    energy = float(problem.full_coefficients @ full_mean)
    fisher = None
    if want_fisher:
        aa = moments[representation.cross_active_active]
        fisher = aa - active_mean[:, None] * active_mean[None, :]
        fisher = 0.5 * (fisher + fisher.T)
    return energy, gradient, probability, fisher, moments


def batch_state(
    theta: np.ndarray,
    problem: Problem,
    representation: Representation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    # theta: batch x dimension
    scores = representation.features @ theta.T
    logits = -scores
    logits -= logits.max(axis=0, keepdims=True)
    weights = np.exp(np.clip(logits, -745.0, 0.0))
    probability = weights / weights.sum(axis=0, keepdims=True)
    moments = fwht_batch(probability)
    active_mean = moments[representation.masks, :].T
    full_mean = moments[problem.full_masks, :].T
    cross = moments[representation.cross_active_full, :]
    covariance_cost = cross.transpose(2, 0, 1) - active_mean[:, :, None] * full_mean[:, None, :]
    gradient = -np.einsum("bij,j->bi", covariance_cost, problem.full_coefficients)
    energy = full_mean @ problem.full_coefficients
    fishers: list[np.ndarray] = []
    aa = moments[representation.cross_active_active, :]
    for batch_index in range(theta.shape[0]):
        fisher = aa[:, :, batch_index] - np.outer(active_mean[batch_index], active_mean[batch_index])
        fishers.append(0.5 * (fisher + fisher.T))
    return energy, gradient, probability, fishers, moments


def effective_spectrum(fisher: np.ndarray, cut: float = EIGEN_CUT) -> tuple[float, int, float, float]:
    values = np.linalg.eigvalsh(0.5 * (fisher + fisher.T))
    maximum = max(float(values[-1]), 0.0)
    retained = values[values > cut * maximum]
    if maximum <= 0.0 or retained.size == 0:
        return math.inf, 0, 0.0, maximum
    return float(maximum / retained[0]), int(retained.size), float(retained[0]), maximum


def pseudoinverse_direction(fisher: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, int]:
    values, vectors = np.linalg.eigh(0.5 * (fisher + fisher.T))
    maximum = max(float(values[-1]), 0.0)
    keep = values > PSEUDOINVERSE_CUT * maximum
    if not np.any(keep):
        return np.zeros_like(gradient), 0
    direction = -(vectors[:, keep] @ ((vectors[:, keep].T @ gradient) / values[keep]))
    return direction, int(np.sum(keep))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0.0 else math.nan


def geometry_row(
    theta: np.ndarray,
    energy: float,
    gradient: np.ndarray,
    probability: np.ndarray,
    fisher: np.ndarray,
    problem: Problem,
    representation: Representation,
    initial_gradient_rms: float,
    update: np.ndarray | None,
) -> dict[str, object]:
    condition, rank, minimum, maximum = effective_spectrum(fisher)
    dominant = int(np.argmax(probability))
    gradient_rms = float(np.sqrt(np.mean(gradient * gradient)))
    c = representation.coefficients
    c2 = float(c @ c)
    beta = float(theta @ c / c2) if c2 > 0 else math.nan
    transverse = theta - beta * c if c2 > 0 else theta.copy()
    result: dict[str, object] = {
        "gap": float(energy - problem.instance.ground_energy),
        "normalized_gap": float((energy - problem.instance.ground_energy) / problem.instance.spectral_gap),
        "pstar": float(probability[problem.instance.ground_index]),
        "dominant_index": dominant,
        "dominant_probability": float(probability[dominant]),
        "dominant_energy": float(problem.cost[dominant]),
        "dominant_gap": float(problem.cost[dominant] - problem.instance.ground_energy),
        "dominant_hamming": int(np.sum(SPINS[dominant] != problem.instance.ground_spins)),
        "dominant_local_min": int(problem.local_minimum[dominant]),
        "gradient_rms": gradient_rms,
        "gradient_ratio": gradient_rms / max(initial_gradient_rms, 1e-300),
        "target_cosine": cosine(theta, c),
        "target_beta": beta,
        "transverse_norm": float(np.linalg.norm(transverse)),
        "transverse_ratio": float(
            np.linalg.norm(transverse) / max(abs(beta) * np.linalg.norm(c), 1e-300)
        ) if c2 > 0 else math.nan,
        "fisher_condition": condition,
        "fisher_rank": rank,
        "fisher_min_kept": minimum,
        "fisher_max": maximum,
    }
    if update is None or float(np.linalg.norm(update)) == 0.0:
        result["update_norm"] = 0.0
        result["update_target_cosine"] = math.nan
    else:
        result["update_norm"] = float(np.linalg.norm(update))
        result["update_target_cosine"] = cosine(update, c)
    return result


class Adam:
    def __init__(self, dimension: int) -> None:
        self.m = np.zeros(dimension)
        self.v = np.zeros(dimension)
        self.t = 0

    def step(self, gradient: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = BETA1 * self.m + (1.0 - BETA1) * gradient
        self.v = BETA2 * self.v + (1.0 - BETA2) * gradient * gradient
        mhat = self.m / (1.0 - BETA1**self.t)
        vhat = self.v / (1.0 - BETA2**self.t)
        return -LR * mhat / (np.sqrt(vhat) + ADAM_EPS)


def is_boundary_trap(row: dict[str, object], trap_conditions: dict[str, object], ground_energy: float) -> bool:
    return bool(
        float(row["gap"]) > float(trap_conditions["energy_gap_strictly_greater_than"])
        and float(row["dominant_energy"]) > ground_energy + 1e-12
        and float(row["dominant_probability"]) >= float(trap_conditions["dominant_state_probability_at_least"])
        and float(row["pstar"]) <= float(trap_conditions["planted_state_probability_at_most"])
        and float(row["fisher_condition"]) >= float(trap_conditions["effective_fisher_condition_number_at_least"])
        and float(row["gradient_ratio"]) <= float(trap_conditions["gradient_rms_at_most_fraction_of_initial"])
    )
