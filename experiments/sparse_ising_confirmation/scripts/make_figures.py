#!/usr/bin/env python3
"""Regenerate public and experiment-specific confirmatory figures."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1787616000")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS = REPOSITORY / "results" / "confirmatory"
PUBLIC_FIGURES = REPOSITORY / "figures"
STUDY_FIGURES = EXPERIMENT / "figures"
PUBLIC_FIGURES.mkdir(parents=True, exist_ok=True)
STUDY_FIGURES.mkdir(parents=True, exist_ok=True)

PDF_METADATA = {
    "Creator": "QBM Representation Alignment",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {"Software": "QBM Representation Alignment"}

GRAPHS = ["chain", "random_tree", "problem_tree", "full"]
GRAPH_LABELS = ["Native chain", "Random target tree", "Max-weight problem tree", "Full graph"]


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight", metadata=PNG_METADATA)
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)


def main() -> int:
    aggregate = pd.read_csv(RESULTS / "aggregate.csv")
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(GRAPHS))
    width = 0.24
    cells = [
        ("Adam, random", "adam", "random"),
        ("Adam, target biased", "adam", "target_biased"),
        ("Exact natural, target biased", "exact_natural", "target_biased"),
    ]
    for index, (label, method, initialization) in enumerate(cells):
        data = aggregate[
            (aggregate.method == method) & (aggregate.initialization == initialization)
        ].set_index("graph").loc[GRAPHS]
        axis.bar(x + (index - 1) * width, data.success_rate, width, label=label)
    axis.set_xticks(x, GRAPH_LABELS, rotation=12, ha="right")
    axis.set_ylim(0.0, 1.06)
    axis.set_ylabel("Success fraction")
    axis.set_title("Independent weighted sparse-Ising confirmation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    save(figure, PUBLIC_FIGURES, "success_by_representation")

    effects = pd.read_csv(RESULTS / "primary_effects.csv").sort_values("id")
    figure, axis = plt.subplots(figsize=(7.5, 4.3))
    y = np.arange(len(effects))
    point = 100.0 * effects.point_difference
    lower = 100.0 * (effects.point_difference - effects.holm_ci_low)
    upper = 100.0 * (effects.holm_ci_high - effects.point_difference)
    axis.errorbar(point, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=5)
    axis.axvline(0.0, linewidth=1.0)
    axis.set_yticks(y, [f"{row.id}: {row.label}" for row in effects.itertuples()])
    axis.set_xlabel("Paired success difference (percentage points)")
    axis.set_title("Prespecified paired effects with Holm-adjusted intervals")
    axis.grid(axis="x", alpha=0.25)
    save(figure, PUBLIC_FIGURES, "primary_effects")

    resources = pd.read_csv(RESULTS / "preparation_resources.csv")
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    data = [resources[resources.graph == graph].conditional_angle_entries.to_numpy() for graph in GRAPHS]
    axis.boxplot(data, tick_labels=GRAPH_LABELS, showmeans=True)
    axis.set_yscale("log")
    axis.set_ylabel("Conditional rotation-angle entries (log scale)")
    axis.set_title("Exact q-sample representation cost")
    axis.grid(axis="y", which="both", alpha=0.25)
    plt.setp(axis.get_xticklabels(), rotation=12, ha="right")
    save(figure, PUBLIC_FIGURES, "preparation_resources")

    trajectories = pd.read_csv(RESULTS / "trajectory_summary.csv")
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(GRAPHS))
    width = 0.32
    for index, (initialization, label) in enumerate(
        [("random", "Random"), ("target_biased", "Target biased")]
    ):
        data = (
            trajectories[
                (trajectories.method == "adam")
                & (trajectories.initialization == initialization)
            ]
            .groupby("graph")
            .success.mean()
            .reindex(GRAPHS)
        )
        axis.bar(x + (index - 0.5) * width, data, width, label=label)
    axis.set_xticks(x, GRAPH_LABELS, rotation=12, ha="right")
    axis.set_ylim(0.0, 0.78)
    axis.set_ylabel("Adam success fraction")
    axis.set_title("Initialization effect across representations")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    save(figure, STUDY_FIGURES, "initialization_effect")

    figure, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)
    specifications = [
        ("H1", "adam", "problem_tree", "chain"),
        ("H2", "adam", "problem_tree", "random_tree"),
        ("H3", "exact_natural", "problem_tree", "chain"),
    ]
    for axis, (identifier, method, treatment, control) in zip(axes, specifications, strict=True):
        data = trajectories[
            (trajectories.method == method)
            & (trajectories.initialization == "target_biased")
        ]
        treatment_rate = data[data.graph == treatment].groupby("instance_id").success.mean()
        control_rate = data[data.graph == control].groupby("instance_id").success.mean()
        difference = treatment_rate - control_rate
        axis.bar(np.arange(len(difference)), difference.to_numpy())
        axis.axhline(0.0, linewidth=1.0)
        axis.set_ylabel(f"{identifier} diff.")
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Confirmatory instance (frozen order)")
    figure.suptitle("Instance-level paired effects")
    save(figure, STUDY_FIGURES, "instance_effects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
