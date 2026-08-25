from __future__ import annotations

import hashlib
import itertools
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS = REPOSITORY / 'results' / 'partial_alignment_geometry'
INSTANCES = REPOSITORY / 'data' / 'certificate_tight_instances'
GRAPHS = STUDY / 'graphs'
sys.path.insert(0, str(REPOSITORY / 'src'))
sys.path.insert(0, str(STUDY / 'scripts'))

from qbm_alignment.certificate_family import generate_family, state_features, costs, coefficients, evaluate, eff_condition, local_min

SEEDS = (0, 19, 42, 50, 101)
STEPS = 200
TOLERANCE = 0.1
MAIN_BUDGET = 256
SCALING_BUDGETS = (64, 1024)
GRAPH_NAMES = ('chain', 'problem_tree', 'width2', 'width3')
MAIN_METHODS = (
    'sampled_adam',
    'sampled_diagonal_fisher',
    'sampled_two_block_fisher',
    'sampled_star_fisher',
    'sampled_full_fisher',
    'protected_ray_star',
)
SCALING_METHODS = (
    'sampled_adam',
    'sampled_star_fisher',
    'sampled_full_fisher',
    'protected_ray_star',
)
LOG_STEPS = {0, 1, 2, 5, 10, 25, 50, 100, 150, 199}

LR = 0.02
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8
SHRINK = 0.1
RIDGE = 1e-3
FULL_SHRINK = 0.05
STEP_CAP = 0.5
RAY_CAP = 0.5
TRANSVERSE_KEEP = 0.95
RESIDUAL_CAP = 0.1


@dataclass(frozen=True)
class GraphProblem:
    instance_id: str
    instance_width: int
    split: str
    graph: str
    n: int
    planted: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    bits: np.ndarray
    F: np.ndarray
    C: np.ndarray
    c: np.ndarray
    ground: float
    pidx: int
    star_blocks: tuple[tuple[int, ...], ...]
    multiplicity: np.ndarray


class Adam:
    def __init__(self, dimension: int):
        self.m = np.zeros(dimension)
        self.v = np.zeros(dimension)
        self.t = 0

    def step(self, gradient: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = BETA1 * self.m + (1 - BETA1) * gradient
        self.v = BETA2 * self.v + (1 - BETA2) * gradient * gradient
        mhat = self.m / (1 - BETA1**self.t)
        vhat = self.v / (1 - BETA2**self.t)
        return -LR * mhat / (np.sqrt(vhat) + ADAM_EPS)


@dataclass(frozen=True)
class Moments:
    gradient: np.ndarray
    diagonal: np.ndarray
    centered: np.ndarray
    fisher: np.ndarray | None
    sample_rank: int
    alignment_residual: float


def stream_seed(*items: object) -> int:
    text = '|'.join(map(str, items)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'little')


def canonical_biased_theta(problem: GraphProblem, seed: int) -> np.ndarray:
    pairs = tuple(itertools.combinations(range(problem.n), 2))
    pair_index = {edge: index for index, edge in enumerate(pairs)}
    rng = np.random.default_rng(seed)
    full_noise = 0.3 * rng.standard_normal(problem.n + len(pairs))
    active = np.asarray(
        list(range(problem.n)) + [problem.n + pair_index[edge] for edge in problem.edges],
        dtype=np.int64,
    )
    return problem.c + full_noise[active]


def target_coefficients(instance, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    full_edges = tuple(instance.edges)
    full = coefficients(instance.n, instance.clauses, full_edges)
    h = full[: instance.n]
    mapping = {edge: float(value) for edge, value in zip(full_edges, full[instance.n:], strict=True)}
    return np.concatenate([h, np.asarray([mapping.get(edge, 0.0) for edge in edges])])


def build_problem(instance, graph: str, graph_payload: dict) -> GraphProblem:
    edges = tuple(tuple(edge) for edge in graph_payload[instance.instance_id][graph]['edges'])
    bits, F = state_features(instance.n, edges)
    C = costs(bits, instance.clauses)
    c = target_coefficients(instance, edges)
    pidx = int(''.join(map(str, instance.planted)), 2)
    edge_feature = {edge: instance.n + index for index, edge in enumerate(edges)}
    blocks = []
    multiplicity = np.zeros(F.shape[1], dtype=np.float64)
    for vertex in range(instance.n):
        block = [vertex]
        block.extend(edge_feature[edge] for edge in edges if vertex in edge)
        block_tuple = tuple(sorted(set(block)))
        blocks.append(block_tuple)
        multiplicity[list(block_tuple)] += 1.0
    if np.any(multiplicity == 0):
        raise RuntimeError('uncovered feature in star blocks')
    return GraphProblem(
        instance_id=instance.instance_id,
        instance_width=instance.width,
        split='calibration' if instance.instance_id.endswith('_i1') else 'evaluation',
        graph=graph,
        n=instance.n,
        planted=tuple(instance.planted),
        edges=edges,
        bits=bits,
        F=F,
        C=C,
        c=c,
        ground=float(C.min()),
        pidx=pidx,
        star_blocks=tuple(blocks),
        multiplicity=multiplicity,
    )


def exact_batch(problem: GraphProblem, theta_matrix: np.ndarray):
    scores = problem.F @ theta_matrix
    logits = -scores
    logits -= logits.max(axis=0, keepdims=True)
    weights = np.exp(np.clip(logits, -745.0, 0.0))
    probability = weights / weights.sum(axis=0, keepdims=True)
    energy = problem.C @ probability
    mean_features = probability.T @ problem.F
    mean_cost_features = (probability * problem.C[:, None]).T @ problem.F
    gradient = -(mean_cost_features - energy[:, None] * mean_features)
    return energy, probability, gradient


def sample_moments(
    problem: GraphProblem,
    probability: np.ndarray,
    rng: np.random.Generator,
    sample_count: int,
    need_full: bool,
    diagnostics: bool,
) -> Moments:
    indices = rng.choice(probability.size, size=sample_count, p=probability)
    X = problem.F[indices]
    y = problem.C[indices]
    centered = X - X.mean(axis=0)
    centered_y = y - y.mean()
    gradient = -(centered.T @ centered_y) / (sample_count - 1)
    diagonal = np.sum(centered * centered, axis=0) / (sample_count - 1)
    fisher = centered.T @ centered / (sample_count - 1) if need_full else None
    if diagnostics:
        # In the partial family this is the omitted-cost covariance, not a numerical residual.
        Ic = centered.T @ (centered @ problem.c) / (sample_count - 1)
        alignment_residual = float(np.linalg.norm(gradient + Ic) / max(np.linalg.norm(gradient), 1e-15))
        sample_rank = int(np.linalg.matrix_rank(centered, tol=1e-10))
    else:
        alignment_residual = math.nan
        sample_rank = -1
    return Moments(gradient, diagonal, centered, fisher, sample_rank, alignment_residual)


def cap(direction: np.ndarray, c: np.ndarray, fraction: float) -> np.ndarray:
    norm = float(np.linalg.norm(direction))
    limit = fraction * max(float(np.linalg.norm(c)), 1e-12)
    return direction * (limit / norm) if norm > limit else direction


def regularized_solve(covariance: np.ndarray, gradient: np.ndarray, shrink: float) -> np.ndarray:
    diagonal = np.diag(covariance)
    scale = max(float(diagonal.mean()), 1e-12)
    regularized = (1 - shrink) * covariance + shrink * np.diag(diagonal) + RIDGE * scale * np.eye(covariance.shape[0])
    try:
        return -np.linalg.solve(regularized, gradient)
    except np.linalg.LinAlgError:
        return -np.linalg.lstsq(regularized, gradient, rcond=1e-10)[0]


def two_block_direction(problem: GraphProblem, moments: Moments, gradient: np.ndarray) -> np.ndarray:
    direction = np.zeros_like(gradient)
    for selection in (slice(0, problem.n), slice(problem.n, gradient.size)):
        X = moments.centered[:, selection]
        covariance = X.T @ X / (X.shape[0] - 1)
        direction[selection] = regularized_solve(covariance, gradient[selection], SHRINK)
    return direction


def star_direction(problem: GraphProblem, moments: Moments, gradient: np.ndarray) -> np.ndarray:
    direction = np.zeros_like(gradient)
    for block in problem.star_blocks:
        index = np.asarray(block, dtype=np.int64)
        X = moments.centered[:, index]
        covariance = X.T @ X / (X.shape[0] - 1)
        local = regularized_solve(covariance, gradient[index], SHRINK)
        direction[index] += local
    return direction / problem.multiplicity


def full_direction(moments: Moments, gradient: np.ndarray) -> np.ndarray:
    if moments.fisher is None:
        raise RuntimeError('full covariance missing')
    return regularized_solve(moments.fisher, gradient, FULL_SHRINK)


def exact_natural_direction(problem: GraphProblem, theta: np.ndarray) -> np.ndarray:
    _E, gradient, _p, fisher = evaluate(theta, problem.F, problem.C, True)
    return -np.linalg.pinv(fisher, rcond=1e-10) @ gradient


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0 else math.nan


def sampled_update(
    problem: GraphProblem,
    theta: np.ndarray,
    method: str,
    probability: np.ndarray,
    rng: np.random.Generator,
    sample_count: int,
    adam: Adam | None,
    diagnostics: bool,
):
    need_full = method == 'sampled_full_fisher'
    moments = sample_moments(problem, probability, rng, sample_count, need_full, diagnostics)
    gradient = moments.gradient
    if method == 'sampled_adam':
        if adam is None:
            raise RuntimeError('Adam state missing')
        direction = adam.step(gradient)
    elif method == 'sampled_diagonal_fisher':
        ridge = RIDGE * max(float(moments.diagonal.max()), 1e-12)
        direction = cap(-gradient / (moments.diagonal + ridge), problem.c, STEP_CAP)
    elif method == 'sampled_two_block_fisher':
        direction = cap(two_block_direction(problem, moments, gradient), problem.c, STEP_CAP)
    elif method == 'sampled_star_fisher':
        direction = cap(star_direction(problem, moments, gradient), problem.c, STEP_CAP)
    elif method == 'sampled_full_fisher':
        direction = cap(full_direction(moments, gradient), problem.c, STEP_CAP)
    elif method == 'protected_ray_star':
        c2 = float(problem.c @ problem.c)
        beta = float(theta @ problem.c / c2)
        transverse = theta - beta * problem.c
        projected_sample = moments.centered @ problem.c
        scalar_fisher = float(projected_sample @ projected_sample / (sample_count - 1))
        scalar_gradient = float(gradient @ problem.c)
        scale = max(float(moments.diagonal.mean()), 1e-12)
        delta_beta = -scalar_gradient / (scalar_fisher + RIDGE * scale * c2)
        delta_beta = float(np.clip(delta_beta, -RAY_CAP, RAY_CAP))
        residual_gradient = gradient - (scalar_gradient / c2) * problem.c
        residual = star_direction(problem, moments, residual_gradient)
        residual -= float(residual @ problem.c / c2) * problem.c
        residual = cap(residual, problem.c, RESIDUAL_CAP)
        new_theta = (beta + delta_beta) * problem.c + TRANSVERSE_KEEP * transverse + residual
        direction = new_theta - theta
    else:
        raise ValueError(method)
    return direction, moments


def storage_entries(problem: GraphProblem, method: str) -> int:
    dimension = problem.F.shape[1]
    pairs = dimension - problem.n
    if method == 'sampled_adam':
        return 2 * dimension
    if method == 'sampled_diagonal_fisher':
        return dimension
    if method == 'sampled_two_block_fisher':
        return problem.n * (problem.n + 1) // 2 + pairs * (pairs + 1) // 2
    if method in ('sampled_star_fisher', 'protected_ray_star'):
        return sum(len(block) * (len(block) + 1) // 2 for block in problem.star_blocks)
    if method == 'sampled_full_fisher':
        return dimension * (dimension + 1) // 2
    raise ValueError(method)


def final_geometry(problem: GraphProblem, theta: np.ndarray, initial_gradient_rms: float):
    energy, gradient, probability, fisher = evaluate(theta, problem.F, problem.C, True)
    dominant = int(np.argmax(probability))
    gradient_rms = float(np.sqrt(np.mean(gradient * gradient)))
    condition, rank, minimum, maximum = eff_condition(fisher)
    return {
        'final_gap': float(energy - problem.ground),
        'final_pstar': float(probability[problem.pidx]),
        'final_dominant_probability': float(probability[dominant]),
        'final_dominant_gap': float(problem.C[dominant] - problem.ground),
        'final_dominant_hamming': int(np.sum(problem.bits[dominant] != np.asarray(problem.planted))),
        'final_dominant_local_min': int(local_min(problem.bits, problem.C, dominant)),
        'final_gradient_rms': gradient_rms,
        'final_gradient_ratio': gradient_rms / max(initial_gradient_rms, 1e-300),
        'final_fisher_condition': condition,
        'final_fisher_rank': rank,
        'final_fisher_min_kept': minimum,
        'final_fisher_max': maximum,
    }


def run_group(problem: GraphProblem, sample_budget: int, methods: tuple[str, ...]):
    specs = []
    for seed in SEEDS:
        base_stream = stream_seed('partial', problem.instance_id, problem.graph, seed, sample_budget)
        theta0 = canonical_biased_theta(problem, seed)
        for method in methods:
            specs.append({'seed': seed, 'method': method, 'stream_seed': base_stream, 'theta0': theta0.copy()})
    count = len(specs)
    dimension = problem.F.shape[1]
    theta = np.column_stack([spec['theta0'] for spec in specs])
    active = np.ones(count, dtype=bool)
    first_success = np.full(count, -1, dtype=np.int64)
    minimum_gap = np.full(count, np.inf)
    initial_gradient_rms = np.full(count, np.nan)
    sample_batches = np.zeros(count, dtype=np.int64)
    first_direction_cosine = np.full(count, np.nan)
    first_target_cosine = np.full(count, np.nan)
    first_sample_rank = np.full(count, -1, dtype=np.int64)
    first_alignment_residual = np.full(count, np.nan)
    adams = [Adam(dimension) if spec['method'] == 'sampled_adam' else None for spec in specs]
    rngs = [np.random.default_rng(int(spec['stream_seed'])) for spec in specs]
    logs = []

    for step in range(STEPS):
        energy, probability, exact_gradient = exact_batch(problem, theta)
        gap = energy - problem.ground
        pstar = probability[problem.pidx]
        if step == 0:
            initial_gradient_rms = np.sqrt(np.mean(exact_gradient * exact_gradient, axis=1))
        minimum_gap = np.minimum(minimum_gap, gap)
        newly = active & (gap <= TOLERANCE)
        first_success[newly] = step
        active[newly] = False
        if step in LOG_STEPS or np.any(newly):
            for index, spec in enumerate(specs):
                if step in LOG_STEPS or newly[index]:
                    beta = float(theta[:, index] @ problem.c / max(problem.c @ problem.c, 1e-15))
                    transverse = theta[:, index] - beta * problem.c
                    logs.append({
                        'instance_id': problem.instance_id,
                        'instance_width': problem.instance_width,
                        'split': problem.split,
                        'graph': problem.graph,
                        'sample_budget': sample_budget,
                        'seed': spec['seed'],
                        'method': spec['method'],
                        'step': step,
                        'gap': float(gap[index]),
                        'pstar': float(pstar[index]),
                        'target_beta': beta,
                        'transverse_norm': float(np.linalg.norm(transverse)),
                    })
        if not np.any(active) or step == STEPS - 1:
            break
        for index in np.flatnonzero(active):
            spec = specs[index]
            diagnostics = step == 0
            direction, moments = sampled_update(
                problem,
                theta[:, index],
                str(spec['method']),
                probability[:, index],
                rngs[index],
                sample_budget,
                adams[index],
                diagnostics,
            )
            theta[:, index] += direction
            sample_batches[index] += 1
            if diagnostics:
                first_direction_cosine[index] = math.nan
                first_target_cosine[index] = cosine(direction, problem.c)
                first_sample_rank[index] = moments.sample_rank
                first_alignment_residual[index] = moments.alignment_residual

    final_energy, final_probability, final_gradient = exact_batch(problem, theta)
    final_gap = final_energy - problem.ground
    final_pstar = final_probability[problem.pidx]
    final_dominant = np.argmax(final_probability, axis=0)
    final_dominant_probability = final_probability[final_dominant, np.arange(count)]
    final_gradient_rms = np.sqrt(np.mean(final_gradient * final_gradient, axis=1))
    final_gradient_ratio = final_gradient_rms / np.maximum(initial_gradient_rms, 1e-300)
    rows = []
    for index, spec in enumerate(specs):
        dominant = int(final_dominant[index])
        candidate = bool(
            final_gap[index] > TOLERANCE
            and problem.C[dominant] > problem.ground + 1e-12
            and final_dominant_probability[index] >= 0.9
            and final_pstar[index] <= 0.1
            and final_gradient_ratio[index] <= 0.1
        )
        condition = math.nan
        rank = -1
        minimum = math.nan
        maximum = math.nan
        if candidate:
            _E, _g, _p, fisher = evaluate(theta[:, index], problem.F, problem.C, True)
            condition, rank, minimum, maximum = eff_condition(fisher)
        rows.append({
            'instance_id': problem.instance_id,
            'instance_width': problem.instance_width,
            'split': problem.split,
            'graph': problem.graph,
            'sample_budget': sample_budget,
            'seed': spec['seed'],
            'method': spec['method'],
            'parameter_dimension': dimension,
            'success': int(first_success[index] >= 0),
            'first_success_step': int(first_success[index]) if first_success[index] >= 0 else math.nan,
            'minimum_gap': float(minimum_gap[index]),
            'initial_gradient_rms': float(initial_gradient_rms[index]),
            'sample_batches': int(sample_batches[index]),
            'total_samples': int(sample_batches[index] * sample_budget),
            'stored_metric_entries': storage_entries(problem, str(spec['method'])),
            'first_direction_cosine_exact_ng': float(first_direction_cosine[index]),
            'first_direction_cosine_projected_target': float(first_target_cosine[index]),
            'first_sample_rank': int(first_sample_rank[index]),
            'first_sample_alignment_residual': float(first_alignment_residual[index]),
            'final_gap': float(final_gap[index]),
            'final_pstar': float(final_pstar[index]),
            'final_dominant_probability': float(final_dominant_probability[index]),
            'final_dominant_gap': float(problem.C[dominant] - problem.ground),
            'final_dominant_hamming': int(np.sum(problem.bits[dominant] != np.asarray(problem.planted))),
            'final_dominant_local_min': int(local_min(problem.bits, problem.C, dominant)),
            'final_gradient_rms': float(final_gradient_rms[index]),
            'final_gradient_ratio': float(final_gradient_ratio[index]),
            'final_fisher_condition': float(condition),
            'final_fisher_rank': int(rank),
            'final_fisher_min_kept': float(minimum),
            'final_fisher_max': float(maximum),
            'final_trap_candidate': int(candidate),
        })
    return rows, logs


def worker(payload):
    instance, graph, sample_budget, methods, graph_payload = payload
    problem = build_problem(instance, graph, graph_payload)
    return run_group(problem, sample_budget, tuple(methods))


def main():
    start = time.time()
    instances = generate_family(INSTANCES)
    graph_payload = json.loads((GRAPHS / 'partial_graphs.json').read_text())
    payloads = []
    for instance in instances:
        for graph in GRAPH_NAMES:
            payloads.append((instance, graph, MAIN_BUDGET, MAIN_METHODS, graph_payload))
        for budget in SCALING_BUDGETS:
            payloads.append((instance, 'width3', budget, SCALING_METHODS, graph_payload))
    outputs = []
    context = mp.get_context('fork')
    workers = min(4, max(1, mp.cpu_count() - 1))
    with context.Pool(workers, maxtasksperchild=2) as pool:
        for index, output in enumerate(pool.imap_unordered(worker, payloads), start=1):
            outputs.append(output)
            print(f'group {index}/{len(payloads)}', flush=True)
    trajectories = pd.DataFrame([row for output in outputs for row in output[0]])
    logs = pd.DataFrame([row for output in outputs for row in output[1]])
    trajectories.to_csv(RESULTS / 'partial_alignment_trajectories.csv', index=False)
    logs.to_csv(RESULTS / 'partial_alignment_logs.csv.gz', index=False, compression='gzip')
    metadata = {
        'main_budget': MAIN_BUDGET,
        'scaling_budgets': SCALING_BUDGETS,
        'parameter_seeds': SEEDS,
        'graphs': GRAPH_NAMES,
        'main_methods': MAIN_METHODS,
        'scaling_methods': SCALING_METHODS,
        'steps': STEPS,
        'calibration_instances': [f'ct_w{width}_i1' for width in (3,4,5,6)],
        'evaluation_instances': [f'ct_w{width}_i{index}' for width in (3,4,5,6) for index in range(2,6)],
        'elapsed_seconds': time.time() - start,
        'hyperparameters': {
            'shrink': SHRINK,
            'ridge': RIDGE,
            'full_shrink': FULL_SHRINK,
            'step_cap': STEP_CAP,
            'ray_cap': RAY_CAP,
            'transverse_keep': TRANSVERSE_KEEP,
            'residual_cap': RESIDUAL_CAP,
        },
    }
    (RESULTS / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('wrote', len(trajectories), 'trajectories in', time.time() - start, 'seconds')

if __name__ == '__main__':
    main()
