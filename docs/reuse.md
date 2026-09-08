# Sparse Gibbs models, natural gradients, and q-sample resources: reuse guide

This research archive is useful for choosing an interaction graph for a commuting quantum Boltzmann machine (QBM), comparing finite-budget Ising energy optimization, checking exponential-family gradient identities, and estimating exact logical q-sample preparation resources. It contains code, protocols, compact numerical tables, and supporting studies, not a general-purpose noncommuting QBM package.

## Which sparse interaction graph should I use as a benchmark?

The primary comparison uses a native chain, a random target-supported spanning tree, a maximum-absolute-coupling target-supported spanning tree (MAXJ), and the full target graph. The three sparse representations have matched width one and 31 parameters at 16 variables.

On the prospectively frozen benchmark of 24 separately generated weighted sparse-Ising instances, target-biased Adam has success rates of 3.33%, 17.50%, 35.83%, and 70.00%, respectively. MAXJ is an empirically supported baseline for this benchmark, not a universally optimal tree-selection rule. Random initialization performs poorly for every sparse representation.

Sources: [experiment](../experiments/sparse_ising_confirmation/README.md), [aggregate outcomes](../results/confirmatory/aggregate.csv), [paired effects](../results/confirmatory/primary_effects.csv), and [statistical method](statistical_analysis.md).

## Where are the Gibbs energy gradient and natural-gradient identities?

For retained sufficient statistics F and target cost C, the energy gradient is minus Cov(F,C). Decomposing C into retained terms plus an omitted residual gives grad E = -I c - Cov(F,R), where I = Cov(F,F).

Full alignment guarantees a vanishing omitted-cost term. Under partial alignment that covariance can also vanish at individual states; its value is not an if-and-only-if test of representation completeness. The identity describes the dynamics but does not prove the empirical ordering of the graph choices.

Sources: [theory](theory.md), [implementation](../src/qbm_alignment/), and [finite-sample supporting study](../studies/finite_sample_geometry/README.md). The same-batch covariance identity applies under full alignment; it is not a general measurement-cost advantage.

## How many gates does a tree Gibbs q-sample require?

Under the repository's conditional-rotation accounting, an n-variable tree uses 2n-1 conditional-angle entries and a Gray-code CNOT upper count of 2n-2. At n=16 these counts are 31 and 30. The full graphs in the benchmark have 75-159 conditional-angle entries under the stored elimination orders, with median 131.

A coherent q-sample has amplitudes equal to square roots of classical probabilities. It reproduces diagonal expectations but is not the mixed Gibbs state. The documented purification adds one copy CNOT per variable and an ancillary register. These are logical resources, not hardware or fault-tolerant costs.

Sources: [preparation definitions](preparation.md) and [instance-level resource table](../results/confirmatory/preparation_resources.csv).

## Does a temperature-dependent cooling-power tree criterion work?

A separate exhaustive study of ten reused eight-variable Exact-1-in-3-SAT instances finds that cooling-power-optimal trees change with temperature. Nevertheless, at the certification temperature, their projected target-energy gaps are worse than those of both the best hot-optimal trees and the forward-KL-optimal trees on all ten instances. This is supporting developmental evidence against that particular selector, not evidence against every adaptive representation method.

Sources: [study entry point](../studies/temperature_tree_geometry/README.md) and [certification-temperature table](../results/temperature_tree_geometry/certification_temperature_summary.csv).

## What can I run without repeating the large calculations?

From an installed checkout, inspect the packaged results with:

```bash
python scripts/validate_repository.py
python -m pytest -q
```

These checks do not rerun the full optimization campaign or exhaustive atlas. Version 1.1.0 has a known cleanup/validation mismatch in the full temperature-study regeneration recipe: do not run its `--clean-results` command in a working checkout containing reference evidence. Inspect the packaged summaries until the separate reproduction-maintenance patch is available. Installation and the other reproduction routes are described in [reproducibility](reproducibility.md).

## What should this archive not be cited for?

It does not demonstrate quantum speedup, asymptotic optimizer scaling, hardware advantage, a general noncommuting training method, or a universally optimal MAXJ construction. The primary confirmation is internal and prospectively frozen, not an external replication. The archive is not a peer-reviewed journal publication.

## How do I cite the results or software?

The published research archive is v1.1.0, dated 2026-08-26, at commit `1fc02e89f283225b1d3503749b33c62c30d0a550`. Later documentation changes do not alter that tag.

> Lin, R. (2026). *Representation Alignment in Commuting Quantum Boltzmann Machines* (Version 1.1.0) [Computer software]. GitHub.

Use the [release](https://github.com/GoGoKo699/QBM-Representation-Alignment/releases/tag/v1.1.0), [copy-ready citation and BibTeX](../CITATION.md), and [machine-readable citation](../CITATION.cff). Name the result file or supporting study when citing a specific finding. The BSD 3-Clause license permits reuse subject to its terms; see [LICENSE](../LICENSE).
