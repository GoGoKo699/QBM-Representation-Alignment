from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS = REPOSITORY / 'results' / 'partial_alignment_geometry'
INSTANCES = REPOSITORY / 'data' / 'certificate_tight_instances'
GRAPHS = STUDY / 'graphs'
sys.path.insert(0, str(REPOSITORY / 'src'))
sys.path.insert(0, str(STUDY / 'scripts'))

from qbm_alignment.certificate_family import generate_family, formula_edges, min_fill_order, induced_width, exact_min_width_order, state_features, costs, coefficients, evaluate

GRAPH_NAMES = ('chain', 'problem_tree', 'width2', 'width3', 'full')
RANDOM_RESTARTS = 64


def target_maps(instance):
    edges = tuple(instance.edges)
    coeff = coefficients(instance.n, instance.clauses, edges)
    h = coeff[:instance.n]
    J = {edge: float(value) for edge, value in zip(edges, coeff[instance.n:], strict=True)}
    return h, J


def deterministic_problem_tree(n: int, J: dict[tuple[int,int], float]) -> tuple[tuple[int,int], ...]:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for edge, value in J.items():
        # lexical epsilon gives deterministic tie-breaking without affecting scientific weight.
        lexical = 1e-9 * (1.0 - (edge[0] * n + edge[1]) / (n*n))
        graph.add_edge(*edge, weight=abs(value) + lexical)
    if not nx.is_connected(graph):
        raise RuntimeError('problem graph is disconnected')
    tree = nx.maximum_spanning_tree(graph, weight='weight', algorithm='kruskal')
    return tuple(sorted((min(i,j), max(i,j)) for i,j in tree.edges()))


def greedy_bounded_graph(instance_id: str, n: int, full_edges: tuple[tuple[int,int], ...], J: dict[tuple[int,int],float], target_width: int, tree_edges: tuple[tuple[int,int], ...]) -> tuple[tuple[int,int], ...]:
    seed = int.from_bytes(hashlib.sha256(f'{instance_id}|w{target_width}'.encode()).digest()[:8], 'little')
    rng = random.Random(seed)
    best = None
    remaining = [edge for edge in full_edges if edge not in set(tree_edges)]
    for restart in range(RANDOM_RESTARTS):
        graph = set(tree_edges)
        # Primary key is descending |J|; random jitter only breaks equal-weight ties.
        jitter = {edge: rng.random() for edge in remaining}
        ordered = sorted(remaining, key=lambda e: (-abs(J[e]), jitter[e], e))
        for edge in ordered:
            trial = tuple(sorted(graph | {edge}))
            order = min_fill_order(n, trial)
            width, _ = induced_width(n, trial, order)
            if width <= target_width:
                graph.add(edge)
        result = tuple(sorted(graph))
        order = min_fill_order(n, result)
        width, _ = induced_width(n, result, order)
        weight = sum(J[e] ** 2 for e in result)
        key = (weight, len(result), tuple(result))
        if best is None or key > best[0]:
            best = (key, result, width, order)
    assert best is not None
    result = best[1]
    order = min_fill_order(n, result)
    certified_width, _ = induced_width(n, result, order)
    if certified_width > target_width:
        raise RuntimeError('constructed graph violates width budget')
    return result


def exact_alignment_metrics(instance, edges, h, J):
    bits, F = state_features(instance.n, edges)
    C = costs(bits, instance.clauses)
    c = np.concatenate([h, np.asarray([J.get(edge, 0.0) for edge in edges], dtype=np.float64)])
    theta = c.copy()
    E, g, p, I = evaluate(theta, F, C, True)
    varC = float(p @ ((C - E) ** 2))
    if varC <= 1e-15:
        explained = 1.0
    else:
        # Cov(F,C) = -g.  Regression-explained cost variance in retained features.
        explained = float(g @ np.linalg.pinv(I, rcond=1e-12) @ g / varC)
    projected_identity_residual = float(np.linalg.norm(g + I @ c) / max(np.linalg.norm(g), 1e-15))
    exact_ng = -np.linalg.pinv(I, rcond=1e-12) @ g
    cosine = float(exact_ng @ c / max(np.linalg.norm(exact_ng)*np.linalg.norm(c), 1e-15))
    return {
        'energy_at_projected_target': float(E),
        'cost_variance': varC,
        'explained_variance_fraction': explained,
        'projected_identity_relative_residual': projected_identity_residual,
        'exact_ng_projected_target_cosine': cosine,
        'parameter_dimension': len(c),
    }


def main():
    instances = generate_family(INSTANCES)
    rows = []
    graph_payload = {}
    for instance in instances:
        h, J = target_maps(instance)
        full = tuple(instance.edges)
        tree = deterministic_problem_tree(instance.n, J)
        graphs = {
            'chain': tuple((i, i+1) for i in range(instance.n-1)),
            'problem_tree': tree,
            'width2': greedy_bounded_graph(instance.instance_id, instance.n, full, J, 2, tree),
            'width3': greedy_bounded_graph(instance.instance_id, instance.n, full, J, 3, tree),
            'full': full,
        }
        full_norm2 = float(np.sum(h*h) + sum(v*v for v in J.values()))
        full_pair_norm2 = float(sum(v*v for v in J.values()))
        graph_payload[instance.instance_id] = {}
        for name, edges in graphs.items():
            if name == 'full':
                order = tuple(instance.order)
                width = int(instance.width)
                _, degrees = induced_width(instance.n, edges, order)
            else:
                order = min_fill_order(instance.n, edges)
                width, degrees = induced_width(instance.n, edges, order)
            retained_pair_norm2 = float(sum(J.get(edge, 0.0)**2 for edge in edges))
            retained_target_norm2 = float(np.sum(h*h) + retained_pair_norm2)
            target_edge_count = sum(edge in J for edge in edges)
            metrics = exact_alignment_metrics(instance, edges, h, J)
            row = {
                'instance_id': instance.instance_id,
                'instance_width': instance.width,
                'graph': name,
                'compiled_graph_width': width,
                'edge_count': len(edges),
                'target_edge_count': target_edge_count,
                'pair_weight_fraction': retained_pair_norm2 / full_pair_norm2,
                'target_norm_fraction': retained_target_norm2 / full_norm2,
                'elimination_order': ' '.join(map(str, order)),
                'elimination_degrees': ' '.join(map(str, degrees)),
                'edges': ';'.join(f'{i}-{j}' for i,j in edges),
                **metrics,
            }
            rows.append(row)
            graph_payload[instance.instance_id][name] = {
                'edges': edges,
                'compiled_width': width,
                'order': order,
            }
        print('built', instance.instance_id, flush=True)
    with (GRAPHS / 'partial_graphs.json').open('w', encoding='utf-8') as handle:
        json.dump(graph_payload, handle, indent=2)
    with (RESULTS / 'partial_graph_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('wrote graph definitions and metrics')

if __name__ == '__main__':
    main()
