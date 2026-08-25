from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_boundary_geometry_evidence():
    results = ROOT / "results" / "boundary_geometry"
    assert len(pd.read_csv(results / "optimizer_suite_summary.csv")) == 840
    replay = pd.read_csv(results / "trap_replay_summary.csv")
    assert len(replay) == 32
    assert set(replay.replay_checkpoint) == {199, 999}
    validation = json.loads((results / "validation.json").read_text())
    assert validation["status"] == "PASS"
    assert validation["reproduced_traps"] == ["ct_w5_i1", "ct_w6_i1"]


def test_finite_sample_evidence():
    results = ROOT / "results" / "finite_sample_geometry"
    assert len(pd.read_csv(results / "finite_sample_broad_trajectories.csv")) == 2100
    assert len(pd.read_csv(results / "independent_full_fisher_trajectories.csv")) == 400
    replay = pd.read_csv(results / "finite_sample_trap_replays.csv")
    assert len(replay) == 244
    assert int(replay[replay.method == "ray_plus_residual"].success.sum()) == 80


def test_partial_alignment_evidence():
    results = ROOT / "results" / "partial_alignment_geometry"
    assert len(pd.read_csv(results / "partial_alignment_trajectories.csv")) == 3200
    assert len(pd.read_csv(results / "exact_natural_oracle.csv")) == 400
    assert len(pd.read_csv(results / "independent_full_fisher.csv")) == 400
    assert len(pd.read_csv(results / "independent_equal_full_fisher.csv")) == 300
    assert len(pd.read_csv(results / "sampled_bag_fisher.csv")) == 400
    summary = pd.read_csv(results / "primary_summary.csv")
    tree_full = summary[
        (summary.graph == "problem_tree") & (summary.method == "sampled_full_fisher")
    ]
    assert int(tree_full.successes.iloc[0]) == 75
