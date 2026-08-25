from __future__ import annotations
import sys

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / 'src'))
STUDY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / 'results' / 'boundary_geometry'
INSTANCES = REPOSITORY / 'data' / 'certificate_tight_instances'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = STUDY
OUT = RESULTS
FIG = STUDY / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f'{name}.png', dpi=220, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def baseline_figures() -> None:
    data = pd.read_csv(OUT / 'baseline_geometry_classified.csv')
    for variable, ylabel, scale, name in [
        ('gap', 'energy gap', 'log', 'baseline_gap'),
        ('dominant_probability', 'dominant-state probability', 'linear', 'baseline_dominant_probability'),
        ('pstar', 'planted-solution probability', 'log', 'baseline_pstar'),
        ('fisher_condition', 'effective Fisher condition number', 'log', 'baseline_fisher_condition'),
        ('theta_target_cosine', 'parameter/target cosine', 'linear', 'baseline_target_cosine'),
    ]:
        fig, ax = plt.subplots(figsize=(6.7, 4.3))
        for instance, group in data.groupby('instance_id'):
            ax.plot(group.step, group[variable], marker='o', label=instance)
        if scale == 'log':
            ax.set_yscale('log')
        if variable == 'gap':
            ax.axhline(0.1, linestyle='--', linewidth=1, label='success threshold')
        ax.set_xlabel('recorded Adam state')
        ax.set_ylabel(ylabel)
        ax.set_title('Exact-target Adam boundary trajectories')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        save(fig, name)


def replay_figures() -> None:
    data = pd.read_csv(OUT / 'trap_replay_geometry_classified.csv')
    for instance in sorted(data.instance_id.unique()):
        for checkpoint in sorted(data.replay_checkpoint.unique()):
            subset = data[(data.instance_id == instance) & (data.replay_checkpoint == checkpoint)]
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            for optimizer, group in subset.groupby('optimizer'):
                ax.plot(group.step, group.gap, marker='o', markersize=3, label=optimizer.replace('_', ' '))
            ax.set_yscale('log')
            ax.axhline(0.1, linestyle='--', linewidth=1, label='success threshold')
            ax.set_xlabel('replay state')
            ax.set_ylabel('energy gap')
            ax.set_title(f'{instance}: replay from Adam state {checkpoint}')
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, fontsize=8, ncol=2)
            save(fig, f'replay_gap_{instance}_from_{checkpoint}')

            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            for optimizer, group in subset.groupby('optimizer'):
                ax.plot(group.step, group.pstar, marker='o', markersize=3, label=optimizer.replace('_', ' '))
            ax.set_yscale('log')
            ax.axhline(0.9, linestyle='--', linewidth=1, label='certified target')
            ax.set_xlabel('replay state')
            ax.set_ylabel('planted-solution probability')
            ax.set_title(f'{instance}: planted probability from state {checkpoint}')
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, fontsize=8, ncol=2)
            save(fig, f'replay_pstar_{instance}_from_{checkpoint}')


def suite_figures() -> None:
    aggregate = pd.read_csv(OUT / 'optimizer_suite_aggregate.csv')
    methods = list(dict.fromkeys(aggregate.optimizer.tolist()))
    for initialization in aggregate.initialization.unique():
        data = aggregate[aggregate.initialization == initialization].set_index('optimizer').loc[methods]
        x = np.arange(len(data))
        y = data.success_rate.to_numpy()
        low = y - data.cluster_ci_low.to_numpy()
        high = data.cluster_ci_high.to_numpy() - y
        fig, ax = plt.subplots(figsize=(8.2, 4.7))
        ax.bar(x, y)
        ax.errorbar(x, y, yerr=np.vstack([low, high]), fmt='none', capsize=4)
        ax.set_xticks(x, [name.replace('_', '\n') for name in data.index])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('success fraction')
        ax.set_title(f'Optimizer success: {initialization.replace("_", " ")} initialization')
        ax.grid(True, axis='y', alpha=0.25)
        save(fig, f'suite_success_{initialization}')

    width = pd.read_csv(OUT / 'optimizer_suite_by_width.csv')
    for initialization in width.initialization.unique():
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        data = width[width.initialization == initialization]
        for optimizer, group in data.groupby('optimizer'):
            ax.plot(group.width, group.success_rate, marker='o', label=optimizer.replace('_', ' '))
        ax.set_xticks(sorted(data.width.unique()))
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel('exact uniqueness-support width')
        ax.set_ylabel('success fraction')
        ax.set_title(f'Success by width: {initialization.replace("_", " ")}')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        save(fig, f'suite_success_by_width_{initialization}')

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for initialization, group in aggregate.groupby('initialization'):
        ax.scatter(group.total_gradient_evaluations_per_success, group.success_rate, label=initialization.replace('_', ' '), s=55)
        for _, row in group.iterrows():
            if np.isfinite(row.total_gradient_evaluations_per_success):
                ax.annotate(row.optimizer.replace('_', ' '), (row.total_gradient_evaluations_per_success, row.success_rate), xytext=(4, 3), textcoords='offset points', fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel('total objective/gradient evaluations per success')
    ax.set_ylabel('success fraction')
    ax.set_title('Exact-moment success–evaluation frontier')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)
    save(fig, 'success_evaluation_frontier')


def main() -> None:
    baseline_figures()
    replay_figures()
    suite_figures()
    mechanism_figures()
    print(f'wrote figures to {FIG}')


# Additional mechanism figures are generated by this helper when called directly.
def mechanism_figures() -> None:
    distances = pd.read_csv(OUT / 'trap_target_distance.csv')
    labels = [f"{r.instance_id}\nstate {int(r.checkpoint)}" for r in distances.itertuples()]
    x = np.arange(len(distances))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(x - width, distances.minimum_target_increment_to_success, width, label='required without projection')
    ax.bar(x, distances.fixed_1000_state_increment, width, label='fixed-step budget')
    ax.bar(x + width, distances.minimum_increment_after_projection, width, label='required after projection')
    ax.set_xticks(x, labels)
    ax.set_ylabel('target-direction increment')
    ax.set_title('Transverse drift creates a finite cooling-distance debt')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False)
    save(fig, 'trap_target_distance')

    paired = pd.read_csv(OUT / 'optimizer_paired_effects_vs_adam.csv')
    data = paired[paired.initialization == 'target_biased'].sort_values('success_difference', ascending=False)
    x = np.arange(len(data))
    y = data.success_difference.to_numpy()
    low = y - data.cluster_ci_low.to_numpy()
    high = data.cluster_ci_high.to_numpy() - y
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(x, y)
    ax.errorbar(x, y, yerr=np.vstack([low, high]), fmt='none', capsize=4)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x, [value.replace('_', '\n') for value in data.optimizer])
    ax.set_ylabel('paired success difference from Adam')
    ax.set_title('Target-biased paired optimizer effects')
    ax.grid(True, axis='y', alpha=0.25)
    save(fig, 'paired_effects_target_biased')


if __name__ == '__main__':
    main()
