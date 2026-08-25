from __future__ import annotations

from pathlib import Path

import numpy as np

from qbm_alignment.sparse_ising import build_problem, exact_state, load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "sparse_ising_confirmation" / "instances" / "manifest.json"


def test_full_alignment_identity():
    instance = load_manifest(MANIFEST)[0]
    problem = build_problem(instance)
    representation = problem.representations["full"]
    theta = representation.coefficients + 0.1 * np.random.default_rng(7).standard_normal(
        representation.coefficients.size
    )
    _energy, gradient, _probability, fisher, _moments = exact_state(
        theta, problem, representation, want_fisher=True
    )
    assert fisher is not None
    np.testing.assert_allclose(
        gradient + fisher @ representation.coefficients,
        0.0,
        atol=2e-11,
        rtol=0.0,
    )


def test_partial_alignment_decomposition():
    instance = load_manifest(MANIFEST)[0]
    problem = build_problem(instance)
    representation = problem.representations["problem_tree"]
    theta = representation.coefficients + 0.1 * np.random.default_rng(11).standard_normal(
        representation.coefficients.size
    )
    _energy, gradient, probability, fisher, _moments = exact_state(
        theta, problem, representation, want_fisher=True
    )
    assert fisher is not None
    represented_cost = representation.features @ representation.coefficients
    residual_cost = problem.cost - represented_cost
    mean_features = probability @ representation.features
    mean_residual = float(probability @ residual_cost)
    residual_covariance = (
        (probability * residual_cost) @ representation.features
        - mean_residual * mean_features
    )
    np.testing.assert_allclose(
        gradient + fisher @ representation.coefficients + residual_covariance,
        0.0,
        atol=3e-11,
        rtol=0.0,
    )
