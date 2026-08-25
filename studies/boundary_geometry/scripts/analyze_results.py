from __future__ import annotations
import sys

import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / 'src'))
STUDY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / 'results' / 'boundary_geometry'
INSTANCES = REPOSITORY / 'data' / 'certificate_tight_instances'

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = STUDY
OUT = RESULTS
RNG = np.random.default_rng(20260824)
BOOTSTRAPS = 50000


def cluster_interval(frame: pd.DataFrame, value: str, instance: str = 'instance_id') -> tuple[float, float, float]:
    values = frame.groupby(instance)[value].mean().to_numpy(float)
    draws = values[RNG.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
    q = np.quantile(draws, [0.025, 0.5, 0.975])
    return float(q[0]), float(q[1]), float(q[2])


def paired_interval(frame: pd.DataFrame, method: str, baseline: str = 'adam') -> dict[str, object]:
    key = ['instance_id', 'seed'] if 'seed' in frame.columns else ['instance_id']
    a = frame[frame.optimizer == method].set_index(key)['success'].astype(int)
    b = frame[frame.optimizer == baseline].set_index(key)['success'].astype(int)
    pair = pd.concat([a.rename('method'), b.rename('baseline')], axis=1).dropna()
    pair['difference'] = pair.method - pair.baseline
    per_instance = pair.groupby(level='instance_id').difference.mean().to_numpy(float)
    draws = per_instance[RNG.integers(0, len(per_instance), size=(BOOTSTRAPS, len(per_instance)))].mean(axis=1)
    q = np.quantile(draws, [0.025, 0.5, 0.975])
    method_only = int(((pair.method == 1) & (pair.baseline == 0)).sum())
    baseline_only = int(((pair.method == 0) & (pair.baseline == 1)).sum())
    discordant = method_only + baseline_only
    pvalue = float(binomtest(min(method_only, baseline_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        'optimizer': method,
        'baseline': baseline,
        'paired_rows': int(len(pair)),
        'success_difference': float(pair.difference.mean()),
        'cluster_ci_low': float(q[0]),
        'cluster_ci_median': float(q[1]),
        'cluster_ci_high': float(q[2]),
        'method_only_success': method_only,
        'baseline_only_success': baseline_only,
        'exact_mcnemar_two_sided_p': pvalue,
    }


def trap_flag(frame: pd.DataFrame, definition: dict) -> pd.Series:
    c = definition['conditions']
    return (
        (frame.gap > c['energy_gap_strictly_greater_than'])
        & (frame.dominant_gap > 1e-12)
        & (frame.dominant_probability >= c['dominant_state_probability_at_least'])
        & (frame.pstar <= c['planted_state_probability_at_most'])
        & (frame.fisher_condition >= c['effective_fisher_condition_number_at_least'])
        & (frame.gradient_ratio <= c['gradient_rms_at_most_fraction_of_initial'])
    )


def main() -> None:
    baseline = pd.read_csv(OUT / 'baseline_summary.csv')
    baseline_geometry = pd.read_csv(OUT / 'baseline_geometry.csv')
    replay = pd.read_csv(OUT / 'trap_replay_summary.csv')
    replay_geometry = pd.read_csv(OUT / 'trap_replay_geometry.csv')
    suite = pd.read_csv(OUT / 'optimizer_suite_summary.csv')
    suite_geometry = pd.read_csv(OUT / 'optimizer_suite_geometry.csv')
    definition = json.loads((REPOSITORY / 'data' / 'trap_definition.json').read_text())

    baseline_geometry['trap_v1'] = trap_flag(baseline_geometry, definition)
    replay_geometry['trap_v1'] = trap_flag(replay_geometry, definition)
    suite_geometry['trap_v1'] = trap_flag(suite_geometry, definition)
    baseline_geometry.to_csv(OUT / 'baseline_geometry_classified.csv', index=False)
    replay_geometry.to_csv(OUT / 'trap_replay_geometry_classified.csv', index=False)
    suite_geometry.to_csv(OUT / 'optimizer_suite_geometry_classified.csv', index=False)

    # Replay table and aggregate rescue rate.
    replay_table = replay.sort_values(['replay_checkpoint', 'instance_id', 'optimizer']).copy()
    replay_table.to_csv(OUT / 'trap_replay_comparison.csv', index=False)
    replay_aggregate = (
        replay.groupby(['replay_checkpoint', 'optimizer'], as_index=False)
        .agg(
            cases=('success', 'size'),
            rescues=('success', 'sum'),
            rescue_rate=('success', 'mean'),
            mean_first_success=('first_success', lambda x: float(x[x >= 0].mean()) if np.any(x >= 0) else np.nan),
            median_minimum_gap=('minimum_gap', 'median'),
            mean_gradient_evaluations=('gradient_evaluations', 'mean'),
            median_projection_displacement=('projection_displacement', 'median'),
        )
    )
    replay_aggregate.to_csv(OUT / 'trap_replay_aggregate.csv', index=False)

    # Main aggregate with hierarchical uncertainty.
    rows = []
    for (initialization, optimizer), group in suite.groupby(['initialization', 'optimizer']):
        low, median, high = cluster_interval(group, 'success')
        successes = int(group.success.sum())
        rows.append({
            'initialization': initialization,
            'optimizer': optimizer,
            'trajectories': int(len(group)),
            'successes': successes,
            'success_rate': float(group.success.mean()),
            'cluster_ci_low': low,
            'cluster_ci_median': median,
            'cluster_ci_high': high,
            'mean_first_success': float(group.loc[group.success, 'first_success'].mean()) if successes else np.nan,
            'median_minimum_gap': float(group.minimum_gap.median()),
            'mean_gradient_evaluations': float(group.gradient_evaluations.mean()),
            'total_gradient_evaluations_per_success': float(group.gradient_evaluations.sum() / successes) if successes else np.inf,
        })
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(OUT / 'optimizer_suite_aggregate.csv', index=False)

    width_aggregate = (
        suite.groupby(['initialization', 'optimizer', 'width'], as_index=False)
        .agg(
            trajectories=('success', 'size'),
            successes=('success', 'sum'),
            success_rate=('success', 'mean'),
            mean_first_success=('first_success', lambda x: float(x[x >= 0].mean()) if np.any(x >= 0) else np.nan),
            median_minimum_gap=('minimum_gap', 'median'),
            mean_gradient_evaluations=('gradient_evaluations', 'mean'),
        )
    )
    width_aggregate.to_csv(OUT / 'optimizer_suite_by_width.csv', index=False)

    paired_rows = []
    for initialization, group in suite.groupby('initialization'):
        for optimizer in sorted(set(group.optimizer) - {'adam'}):
            result = paired_interval(group, optimizer)
            result['initialization'] = initialization
            paired_rows.append(result)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / 'optimizer_paired_effects_vs_adam.csv', index=False)

    # Trap occurrence at the last available log for each trajectory.
    key = ['initialization', 'instance_id', 'seed', 'optimizer']
    final_logs = suite_geometry.sort_values('step').groupby(key, as_index=False).tail(1)
    final_traps = (
        final_logs.groupby(['initialization', 'optimizer'], as_index=False)
        .agg(
            logged_trajectories=('trap_v1', 'size'),
            final_traps=('trap_v1', 'sum'),
            final_trap_rate=('trap_v1', 'mean'),
            median_final_fisher_condition=('fisher_condition', 'median'),
            median_final_transverse_ratio=('transverse_ratio', 'median'),
            median_final_pstar=('pstar', 'median'),
        )
    )
    final_traps.to_csv(OUT / 'optimizer_final_trap_rates.csv', index=False)

    summary = {
        'trap_definition': definition,
        'baseline_traps': baseline.to_dict('records'),
        'replay_rescues': replay_aggregate.to_dict('records'),
        'aggregate': aggregate.to_dict('records'),
        'paired_effects': paired.to_dict('records'),
    }
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')

    print('\nOPTIMIZER AGGREGATE')
    print(aggregate.to_string(index=False))
    print('\nPAIRED EFFECTS')
    print(paired.to_string(index=False))
    print('\nREPLAY')
    print(replay_aggregate.to_string(index=False))


if __name__ == '__main__':
    main()
