# Scientific claims and evidence

## Primary confirmed claim

For commuting quantum Boltzmann machines used to minimize sparse Ising costs, target-informed sparse interaction graphs can improve finite-budget trainability without increasing treewidth or parameter count.

The independent weighted sparse-Ising experiment compares four representations:

1. a native chain;
2. a prespecified random target-supported spanning tree;
3. a maximum-absolute-coupling target-supported spanning tree;
4. the full target graph.

The chain and both tree models have width one, fifteen pair interactions, and thirty-one parameters. Under target-biased Adam, the maximum-weight target tree exceeds the chain by 32.5 percentage points and the random target tree by 18.33 points. Under exact natural-gradient optimization, it exceeds the chain by 51.67 points. The paired effects and multiplicity-adjusted intervals are stored in [`results/confirmatory/primary_effects.csv`](../results/confirmatory/primary_effects.csv).

## Preparation-aware interpretation

The full graph is more trainable, but its exact graphical-model factorization is larger. All tree representations use thirty-one conditional rotations, whereas full graphs use 75–159, with median 131.

The supported conclusion is a finite-size tradeoff:

$$
\text{greater target alignment and trainability}
\quad\leftrightarrow\quad
\text{larger exact preparation description}.
$$

The experiment does not establish an asymptotic resource separation or hardware advantage.

## Geometry claim

For a partially aligned representation,

$$
\nabla E=-I_Gc_G-r_G,
\qquad
r_G=\mathrm{Cov}(F_G,R_G).
$$

The omitted-cost covariance term is zero only when the target Hamiltonian lies completely in the ansatz feature span. This supplies a representation-level explanation for the empirical chain/tree/full ordering.

## Supporting mechanism studies

The supporting studies establish narrower results:

- coordinatewise Adam can develop transverse parameter drift and concentrate on a wrong excited boundary state in selected development instances;
- same-batch sampled Fisher estimates obey an exact finite-sample identity in a fully aligned commuting family;
- local Fisher blocks approximate full geometry well on a problem tree, but not on the tested width-2 and width-3 cyclic representations;
- a native chain can fail even under exact natural gradient, showing that optimizer quality cannot repair severe feature mismatch by itself.

These mechanism results are not presented as universal explanations of all failures.

## Explicit nonclaims

No result in this repository demonstrates quantum speedup, general NP-hard problem solving, a standard asymptotic barren plateau, or a universally optimal graph-selection rule.

## Research context

Established prior work and the boundary between known QBM methods and this repository's contribution are summarized in [research context](research_context.md). A direct map from every public claim to its protocol, raw evidence, analysis code, and validator is provided in the [claim-to-evidence map](evidence_map.md).
