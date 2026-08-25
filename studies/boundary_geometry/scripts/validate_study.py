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

from qbm_alignment.certificate_family import evaluate, generate_family, state_features, costs, coefficients

ROOT = STUDY
OUT = RESULTS
INST = ROOT / 'instances'


def main() -> None:
    family = generate_family(INSTANCES)
    assert len(family) == 20
    counts = pd.Series([instance.width for instance in family]).value_counts().to_dict()
    assert counts == {3: 5, 4: 5, 5: 5, 6: 5}

    baseline = pd.read_csv(OUT / 'baseline_summary.csv').set_index('instance_id')
    expected = {
        'ct_w5_i1': (1.00064256345102, 55),
        'ct_w6_i1': (1.0006606555118918, 89),
    }
    for instance_id, (gap, trap_step) in expected.items():
        row = baseline.loc[instance_id]
        assert not bool(row.success)
        np.testing.assert_allclose(row.final_gap, gap, atol=3e-12, rtol=0)
        assert int(row.first_trap) == trap_step

    geometry = pd.read_csv(OUT / 'baseline_geometry.csv')
    final = geometry[geometry.step == 999].set_index('instance_id')
    for instance_id in expected:
        row = final.loc[instance_id]
        assert row.dominant_probability > 0.995
        assert row.pstar < 1e-10
        assert row.dominant_gap == 1.0
        assert bool(row.dominant_local_min)
        assert row.fisher_condition > 4e8
        assert row.theta_target_cosine < 0.81

    # Independent aligned-family identity at one finite random point per width.
    # The stored full-family result is retained in validation.json; this compact
    # rerun keeps public CI practical while covering every width class.
    max_identity = 0.0
    representatives = [next(instance for instance in family if instance.width == width) for width in (3, 4, 5, 6)]
    for index, instance in enumerate(representatives):
        bits, features = state_features(instance.n, instance.edges)
        cost = costs(bits, instance.clauses)
        target = coefficients(instance.n, instance.clauses, instance.edges)
        theta = 0.3 * np.random.default_rng(20260824 + index).standard_normal(target.size)
        _, gradient, _, fisher = evaluate(theta, features, cost, True)
        max_identity = max(max_identity, float(np.max(np.abs(gradient + fisher @ target))))
    assert max_identity < 2e-12

    replay = pd.read_csv(OUT / 'trap_replay_summary.csv')
    assert len(replay) == 2 * 2 * 8
    assert set(replay.replay_checkpoint) == {199, 999}
    assert set(replay.optimizer) == {
        'adam', 'armijo_gd', 'target_direction', 'ray_projected',
        'projected_adam', 'exact_natural', 'damped_natural', 'diagonal_fisher'
    }

    suite = pd.read_csv(OUT / 'optimizer_suite_summary.csv')
    assert len(suite) == 20 * 6 * 7
    assert set(suite.initialization) == {'exact_target', 'target_biased'}
    assert set(suite.optimizer) == {
        'adam', 'armijo_gd', 'target_direction', 'ray_projected',
        'projected_adam', 'exact_natural', 'diagonal_fisher'
    }
    assert suite[['minimum_gap', 'final_gap', 'gradient_evaluations']].notna().all().all()
    assert (suite.gradient_evaluations > 0).all()

    ray = pd.read_csv(OUT / 'optimizer_suite_geometry_classified.csv')
    ray = ray[(ray.optimizer == 'ray_projected') & (ray.step == 0)]
    assert len(ray) == 20 * 6
    assert float(ray.transverse_norm.max()) < 1e-10

    validation = {
        'status': 'PASS',
        'family_instances': len(family),
        'width_counts': counts,
        'reproduced_traps': list(expected),
        'maximum_aligned_identity_residual': max_identity,
        'replay_rows': len(replay),
        'suite_rows': len(suite),
    }
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
