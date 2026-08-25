from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from confirmatory_common import (
    Adam,
    ARMIJO,
    PARAMETER_SEEDS,
    PSEUDOINVERSE_CUT,
    STAGNATION_TOL,
    STAGNATION_WINDOW,
    STEP_CAP,
    STEPS,
    build_problem,
    batch_state,
    canonical_noise,
    cosine,
    effective_spectrum,
    exact_state,
    geometry_row,
    is_boundary_trap,
    load_manifest,
    pseudoinverse_direction,
)

MANIFEST = ROOT / 'instances' / 'confirmatory_sparse_ising_manifest.json'
RESULTS = ROOT / 'results'
RAW = RESULTS / 'raw'
STATES = RESULTS / 'states'
RAW.mkdir(parents=True, exist_ok=True)
STATES.mkdir(parents=True, exist_ok=True)
TRAP = json.loads((ROOT / 'TRAP_DEFINITION.json').read_text())['conditions']
GRAPH_NAMES = ('chain', 'random_tree', 'problem_tree', 'full')
TOLERANCE = 0.1


def run_adam(problem, representation, initialization: str):
    theta = np.vstack([canonical_noise(seed, representation.edges) for seed in PARAMETER_SEEDS])
    if initialization == 'target_biased':
        theta = theta + representation.coefficients[None, :]
    initial_theta = theta.copy()
    optimizers = [Adam(theta.shape[1]) for _ in PARAMETER_SEEDS]
    active = np.ones(len(PARAMETER_SEEDS), dtype=bool)
    first_success = np.full(len(PARAMETER_SEEDS), -1, dtype=np.int64)
    minimum_gap = np.full(len(PARAMETER_SEEDS), np.inf)
    initial_gradient_rms = np.full(len(PARAMETER_SEEDS), np.nan)
    rows: list[dict[str, object]] = []

    for step in range(STEPS):
        energy, gradient, probability, fishers, _moments = batch_state(theta, problem, representation)
        normalized_gap = (energy - problem.instance.ground_energy) / problem.instance.spectral_gap
        if step == 0:
            initial_gradient_rms = np.sqrt(np.mean(gradient * gradient, axis=1))
        minimum_gap = np.minimum(minimum_gap, normalized_gap)
        newly = active & (normalized_gap <= TOLERANCE)
        first_success[newly] = step

        updates = np.zeros_like(theta)
        if step < STEPS - 1:
            for index in np.flatnonzero(active & ~newly):
                updates[index] = optimizers[index].step(gradient[index])

        for index, seed in enumerate(PARAMETER_SEEDS):
            if not active[index]:
                continue
            geometry = geometry_row(
                theta[index],
                float(energy[index]),
                gradient[index],
                probability[:, index],
                fishers[index],
                problem,
                representation,
                float(initial_gradient_rms[index]),
                updates[index] if step < STEPS - 1 and not newly[index] else None,
            )
            rows.append({
                'instance_id': problem.instance.instance_id,
                'graph': representation.name,
                'method': 'adam',
                'initialization': initialization,
                'seed': seed,
                'step': step,
                **geometry,
                'omitted_cost_covariance_norm': math.nan,
                'exact_natural_target_cosine': math.nan,
                'accepted_alpha': math.nan,
            })

        active[newly] = False
        if not np.any(active) or step == STEPS - 1:
            break
        theta[active] += updates[active]

    frame = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    final_theta = np.full_like(theta, np.nan)
    for index, seed in enumerate(PARAMETER_SEEDS):
        subset = frame[frame.seed == seed].sort_values('step')
        last = subset.iloc[-1]
        final_theta[index] = theta[index]
        trap = is_boundary_trap(last.to_dict(), TRAP, problem.instance.ground_energy)
        summaries.append({
            'instance_id': problem.instance.instance_id,
            'graph': representation.name,
            'method': 'adam',
            'initialization': initialization,
            'seed': seed,
            'success': int(first_success[index] >= 0),
            'first_success_step': int(first_success[index]) if first_success[index] >= 0 else math.nan,
            'minimum_normalized_gap': float(minimum_gap[index]),
            'final_step': int(last.step),
            'final_gap': float(last.gap),
            'final_normalized_gap': float(last.normalized_gap),
            'final_pstar': float(last.pstar),
            'final_dominant_probability': float(last.dominant_probability),
            'final_dominant_gap': float(last.dominant_gap),
            'final_gradient_rms': float(last.gradient_rms),
            'final_gradient_ratio': float(last.gradient_ratio),
            'final_fisher_condition': float(last.fisher_condition),
            'final_fisher_rank': int(last.fisher_rank),
            'final_target_cosine': float(last.target_cosine),
            'final_transverse_ratio': float(last.transverse_ratio),
            'boundary_trap': int(trap),
        })
    return rows, summaries, initial_theta, final_theta


def run_exact_natural(problem, representation, seed: int):
    theta = representation.coefficients + canonical_noise(seed, representation.edges)
    initial_theta = theta.copy()
    rows: list[dict[str, object]] = []
    history: list[float] = []
    accepted: list[float] = []
    first_success = -1
    minimum_gap = math.inf
    initial_gradient_rms = math.nan

    for step in range(STEPS):
        energy, gradient, probability, fisher, _moments = exact_state(theta, problem, representation, want_fisher=True)
        if fisher is None:
            raise RuntimeError('Fisher matrix missing')
        normalized_gap = (energy - problem.instance.ground_energy) / problem.instance.spectral_gap
        minimum_gap = min(minimum_gap, normalized_gap)
        history.append(normalized_gap)
        if step == 0:
            initial_gradient_rms = float(np.sqrt(np.mean(gradient * gradient)))

        direction, natural_rank = pseudoinverse_direction(fisher, gradient)
        limit = STEP_CAP * max(float(np.linalg.norm(representation.coefficients)), 1e-12)
        norm = float(np.linalg.norm(direction))
        if norm > limit:
            direction *= limit / norm

        success = normalized_gap <= TOLERANCE
        update = None
        accepted_alpha = math.nan
        if not success and step < STEPS - 1:
            directional_derivative = float(gradient @ direction)
            if np.isfinite(directional_derivative) and directional_derivative < -1e-15:
                alpha = 1.0
                for _ in range(25):
                    trial_energy = exact_state(
                        theta + alpha * direction,
                        problem,
                        representation,
                        want_fisher=False,
                    )[0]
                    if trial_energy <= energy + ARMIJO * alpha * directional_derivative:
                        accepted_alpha = alpha
                        update = alpha * direction
                        break
                    alpha *= 0.5

        geometry = geometry_row(
            theta,
            energy,
            gradient,
            probability,
            fisher,
            problem,
            representation,
            initial_gradient_rms,
            update,
        )
        residual = gradient + fisher @ representation.coefficients
        rows.append({
            'instance_id': problem.instance.instance_id,
            'graph': representation.name,
            'method': 'exact_natural',
            'initialization': 'target_biased',
            'seed': seed,
            'step': step,
            **geometry,
            'omitted_cost_covariance_norm': float(np.linalg.norm(residual)),
            'exact_natural_target_cosine': cosine(direction, representation.coefficients),
            'natural_rank': natural_rank,
            'accepted_alpha': accepted_alpha,
        })

        if success:
            first_success = step
            break
        if update is None:
            break
        accepted.append(accepted_alpha)
        theta = theta + update
        if len(history) >= STAGNATION_WINDOW and max(history[-STAGNATION_WINDOW:]) - min(history[-STAGNATION_WINDOW:]) < STAGNATION_TOL:
            break

    last = rows[-1]
    summary = {
        'instance_id': problem.instance.instance_id,
        'graph': representation.name,
        'method': 'exact_natural',
        'initialization': 'target_biased',
        'seed': seed,
        'success': int(first_success >= 0),
        'first_success_step': first_success if first_success >= 0 else math.nan,
        'minimum_normalized_gap': minimum_gap,
        'final_step': int(last['step']),
        'final_gap': float(last['gap']),
        'final_normalized_gap': float(last['normalized_gap']),
        'final_pstar': float(last['pstar']),
        'final_dominant_probability': float(last['dominant_probability']),
        'final_dominant_gap': float(last['dominant_gap']),
        'final_gradient_rms': float(last['gradient_rms']),
        'final_gradient_ratio': float(last['gradient_ratio']),
        'final_fisher_condition': float(last['fisher_condition']),
        'final_fisher_rank': int(last['fisher_rank']),
        'final_target_cosine': float(last['target_cosine']),
        'final_transverse_ratio': float(last['transverse_ratio']),
        'mean_accepted_alpha': float(np.mean(accepted)) if accepted else math.nan,
        'boundary_trap': 0,
    }
    return rows, summary, initial_theta, theta


def run_instance(instance):
    problem = build_problem(instance)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    state_payload: dict[str, np.ndarray] = {}
    resources: list[dict[str, object]] = []

    for graph_name, representation in problem.representations.items():
        resources.append({
            'instance_id': instance.instance_id,
            'graph': graph_name,
            'edge_count': len(representation.edges),
            'parameter_count': representation.features.shape[1],
            **representation.resources,
        })
        for initialization in ('random', 'target_biased'):
            rows, summary, initial, final = run_adam(problem, representation, initialization)
            all_rows.extend(rows)
            summaries.extend(summary)
            state_payload[f'{graph_name}__adam__{initialization}__initial'] = initial
            state_payload[f'{graph_name}__adam__{initialization}__final'] = final
        for seed in PARAMETER_SEEDS:
            rows, summary, initial, final = run_exact_natural(problem, representation, seed)
            all_rows.extend(rows)
            summaries.append(summary)
            state_payload[f'{graph_name}__exact_natural__target_biased__seed{seed}__initial'] = initial
            state_payload[f'{graph_name}__exact_natural__target_biased__seed{seed}__final'] = final

    pd.DataFrame(all_rows).to_csv(RAW / f'{instance.instance_id}_logs.csv.gz', index=False, compression='gzip')
    pd.DataFrame(summaries).to_csv(RAW / f'{instance.instance_id}_summary.csv', index=False)
    pd.DataFrame(resources).to_csv(RAW / f'{instance.instance_id}_resources.csv', index=False)
    np.savez_compressed(STATES / f'{instance.instance_id}_states.npz', **state_payload)
    return {'instance_id': instance.instance_id, 'rows': len(all_rows), 'summaries': len(summaries)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    instances = load_manifest(MANIFEST)
    started = time.time()
    context = mp.get_context('fork')
    completed = []
    with context.Pool(args.workers, maxtasksperchild=1) as pool:
        for index, result in enumerate(pool.imap_unordered(run_instance, instances), 1):
            completed.append(result)
            print('completed', index, '/', len(instances), result, flush=True)
    (RESULTS / 'run_metadata.json').write_text(json.dumps({
        'manifest_sha256': json.loads(MANIFEST.read_text())['manifest_sha256'],
        'workers': args.workers,
        'elapsed_seconds': time.time() - started,
        'graphs': list(GRAPH_NAMES) if 'GRAPH_NAMES' in globals() else ['chain','random_tree','problem_tree','full'],
        'steps': STEPS,
        'parameter_seeds': list(PARAMETER_SEEDS),
        'implementation_decisions_sha256': __import__('hashlib').sha256((ROOT/'IMPLEMENTATION_DECISIONS.md').read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    print('elapsed', time.time() - started)


if __name__ == '__main__':
    main()
