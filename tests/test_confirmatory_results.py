from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "confirmatory"
EXPERIMENT = ROOT / "experiments" / "sparse_ising_confirmation"


def test_confirmatory_manifest_and_counts():
    manifest_path = EXPERIMENT / "instances" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["instances"]) == 24
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "f7f5e5d578c82631cbaed2d7cc5bded2a7d46739120c686610456792247bbb71"
    )

    aggregate = pd.read_csv(RESULTS / "aggregate.csv")
    observed = {
        (row.method, row.initialization, row.graph): int(row.successes)
        for row in aggregate.itertuples(index=False)
    }
    expected = {
        ("adam", "random", "chain"): 1,
        ("adam", "random", "random_tree"): 1,
        ("adam", "random", "problem_tree"): 1,
        ("adam", "random", "full"): 14,
        ("adam", "target_biased", "chain"): 4,
        ("adam", "target_biased", "random_tree"): 21,
        ("adam", "target_biased", "problem_tree"): 43,
        ("adam", "target_biased", "full"): 84,
        ("exact_natural", "target_biased", "chain"): 35,
        ("exact_natural", "target_biased", "random_tree"): 69,
        ("exact_natural", "target_biased", "problem_tree"): 97,
        ("exact_natural", "target_biased", "full"): 120,
    }
    assert observed == expected


def test_primary_effects_are_preserved():
    effects = pd.read_csv(RESULTS / "primary_effects.csv").set_index("id")
    expected = {
        "H1": (0.325, 0.1583333333333333, 0.5083333333333333),
        "H2": (0.1833333333333333, 0.0666666666666666, 0.3166666666666666),
        "H3": (0.5166666666666666, 0.35, 0.6666666666666666),
    }
    for identifier, (point, low, high) in expected.items():
        row = effects.loc[identifier]
        assert abs(float(row.point_difference) - point) < 1e-14
        assert abs(float(row.holm_ci_low) - low) < 1e-14
        assert abs(float(row.holm_ci_high) - high) < 1e-14
        assert bool(row.hypothesis_pass)


def test_preparation_resources_match_representation_contract():
    resources = pd.read_csv(RESULTS / "preparation_resources.csv")
    assert len(resources) == 96
    trees = resources[resources.graph.isin(["chain", "random_tree", "problem_tree"])]
    assert set(trees.width) == {1}
    assert set(trees.conditional_angle_entries) == {31}
    assert set(trees.gray_code_cnot_upper_bound) == {30}
    full = resources[resources.graph == "full"]
    assert full.width.between(3, 5).all()
    assert int(full.conditional_angle_entries.min()) == 75
    assert int(full.conditional_angle_entries.max()) == 159
    assert float(full.conditional_angle_entries.median()) == 131.0
