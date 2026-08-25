# Claim-to-evidence map

This page gives reviewers, reusers, and first-time readers the shortest path from each public claim to its protocol, data, code, and validation record.

## Primary claim

**Claim.** At fixed treewidth, interaction count, and parameter count, a target-supported spanning-tree Gibbs representation trains better than a generic chain; selecting the spanning tree by maximum absolute target-coupling weight improves further over a prespecified random target-supported tree.

| Evidence layer | Canonical location |
|---|---|
| Human-readable result | [`README.md`](../README.md#confirmed-result) |
| Exact claim wording and nonclaims | [`docs/scientific_claims.md`](scientific_claims.md) |
| Prespecified protocol | [`experiments/sparse_ising_confirmation/protocol/protocol.md`](../experiments/sparse_ising_confirmation/protocol/protocol.md) |
| Locked pre-generation source | [`experiments/sparse_ising_confirmation/protocol/frozen_source/`](../experiments/sparse_ising_confirmation/protocol/frozen_source/) |
| Aggregate outcomes | [`results/confirmatory/aggregate.csv`](../results/confirmatory/aggregate.csv) |
| Primary paired effects | [`results/confirmatory/primary_effects.csv`](../results/confirmatory/primary_effects.csv) |
| Statistical method | [`docs/statistical_analysis.md`](statistical_analysis.md) |
| Raw trajectory evidence | [`experiments/sparse_ising_confirmation/results/raw/`](../experiments/sparse_ising_confirmation/results/raw/) |
| Stored initial/final parameters | [`experiments/sparse_ising_confirmation/results/states/`](../experiments/sparse_ising_confirmation/results/states/) |
| Analysis implementation | [`experiments/sparse_ising_confirmation/scripts/analyze_results.py`](../experiments/sparse_ising_confirmation/scripts/analyze_results.py) |
| Scientific validator | [`experiments/sparse_ising_confirmation/scripts/validate_experiment.py`](../experiments/sparse_ising_confirmation/scripts/validate_experiment.py) |
| Validation record | [`results/confirmatory/validation.json`](../results/confirmatory/validation.json) |

## Geometry claim

**Claim.** Full alignment gives $\nabla E=-Ic$, while partial alignment adds the state-dependent omitted-cost covariance term $r_G$.

- Derivation: [`docs/theory.md`](theory.md)
- Maintained implementation: [`src/qbm_alignment/`](../src/qbm_alignment/)
- Supporting partial-alignment study: [`studies/partial_alignment_geometry/`](../studies/partial_alignment_geometry/)
- Supporting validation: [`results/partial_alignment_geometry/validation.json`](../results/partial_alignment_geometry/validation.json)

## Preparation-resource claim

**Claim.** The chain and tree representations have exact width-one factorization and 31 conditional angles, while full graphs require 75–159 angles in the confirmatory ensemble.

- Definitions and scope: [`docs/preparation.md`](preparation.md)
- Instance-level resources: [`results/confirmatory/preparation_resources.csv`](../results/confirmatory/preparation_resources.csv)
- Resource-pair analysis: [`results/confirmatory/resource_pairs.csv`](../results/confirmatory/resource_pairs.csv)
- Compiler and resource script: [`experiments/sparse_ising_confirmation/scripts/compute_preparation_resources.py`](../experiments/sparse_ising_confirmation/scripts/compute_preparation_resources.py)

## Success certificate

**Claim.** For a unique ground state with gap $\gamma$, the criterion $\Delta E/\gamma\le0.1$ certifies $p_\star\ge0.9$.

- Proof: [`docs/theory.md#energy-certificate`](theory.md#energy-certificate)
- Exact spectra and numerical check: [`results/confirmatory/validation_spectra.csv`](../results/confirmatory/validation_spectra.csv)
- Stored-state validation: [`results/confirmatory/validation.json`](../results/confirmatory/validation.json)

## Reproduction routes

Fast package and metadata checks:

```bash
python scripts/validate_repository.py
python -m pytest -q
```

Full independent-experiment validation:

```bash
python experiments/sparse_ising_confirmation/scripts/validate_experiment.py
```

Regenerate compact tables and figures from packaged raw trajectories:

```bash
bash scripts/refresh_analysis.sh
```

See [`docs/reproducibility.md`](reproducibility.md) for the complete workflow.

## Citation

Use [`CITATION.md`](../CITATION.md) for copy-ready APA-style and BibTeX entries. Machine-readable metadata are in [`CITATION.cff`](../CITATION.cff).
