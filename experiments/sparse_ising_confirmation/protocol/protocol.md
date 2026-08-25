# Weighted sparse-Ising confirmatory protocol

## Purpose

This experiment tests whether a problem-aligned spanning-tree QBM outperforms generic sparse controls of the same treewidth and pair-parameter count on an independent weighted sparse-Ising ensemble, and whether exact natural geometry preserves the same representation ordering.

The protocol, instance seeds, graph rules, optimizer settings, endpoint, and primary effect thresholds were fixed before the confirmatory instances were generated. Exact locked source files and hashes are preserved under [`frozen_source/`](frozen_source/).

## Independent problem family

Each target is

$$
C(z)=\sum_{i=1}^{16}h_i z_i+\sum_{(i,j)\in E}J_{ij}z_i z_j,
\qquad z_i\in\{-1,+1\}.
$$

The target graph is a connected random 3-regular graph on sixteen variables, followed by a deterministic random relabeling. Edge signs are uniform, edge magnitudes are drawn from `Uniform(0.5, 1.5)`, and local fields from `Uniform(-0.35, 0.35)`. The complete coefficient vector is RMS normalized.

An instance is accepted only when exact enumeration verifies:

1. graph connectivity;
2. one unique ground state;
3. spectral gap at least $0.05$ after normalization.

Acceptance does not depend on QBM optimization outcomes.

## Endpoint

Success is

$$
\frac{E-E_0}{\gamma}\le0.1.
$$

For a unique ground state this certifies

$$
p_\star\ge0.9.
$$

## Instance and parameter seeds

```text
confirmatory instances: 24
engineering-only instances: 4
parameter seeds: 0, 19, 42, 50, 101
```

The instance-seed commitment is stored in [`seed_commitment.json`](seed_commitment.json).

## Representation ladder

### Native chain

```text
(0,1), (1,2), ..., (14,15)
```

The random graph relabeling makes this a generic local control.

### Random target-supported spanning tree

Assign random priorities independent of target weights and take the resulting Kruskal spanning tree. This controls for width, interaction count, parameter count, and use of target-supported edges.

### Maximum-weight target-supported spanning tree

Take a maximum spanning tree with edge weights $|J_{ij}|$ and lexicographic tie-breaking.

### Full target graph

Retain all target interactions.

The chain and both trees have width one, fifteen pair terms, and thirty-one parameters.

## Initialization

One all-pairs Gaussian vector is generated for each parameter seed and restricted to the active coordinates of each representation.

Random:

$$
\theta^{(0)}=0.3\xi.
$$

Target biased:

$$
\theta^{(0)}=c_G+0.3\xi,
$$

where $c_G$ is the target coefficient vector projected onto the retained features.

## Optimizers

### Adam

Applied to all four representations and both initializations:

```text
learning rate 0.02
beta1 0.9
beta2 0.999
epsilon 1e-8
```

### Exact natural-gradient Armijo oracle

Applied to all four representations from target-biased initialization:

- Moore-Penrose Fisher inverse;
- direction norm cap $0.5\lVert c_G\rVert_2$;
- exact Armijo energy evaluation;
- fixed pseudoinverse and line-search conventions.

This is an exact geometric ceiling, not a practical sampling-cost claim.

## Budget

Each primary trajectory contains 200 recorded states. Optimizer hyperparameters are fixed across all confirmatory instances.

## Prespecified primary comparisons

All comparisons are paired by instance and parameter seed.

- **H1:** target-biased Adam, maximum-weight tree minus native chain; minimum practical effect `+0.15`.
- **H2:** target-biased Adam, maximum-weight tree minus random target tree; minimum practical effect `+0.10`.
- **H3:** target-biased exact natural gradient, maximum-weight tree minus native chain; minimum practical effect `+0.25`.

Instance-cluster bootstrap intervals use 20,000 resamples and Holm adjustment across H1-H3.

The preparation criterion requires the maximum-weight tree to retain exact width one and a materially smaller conditional-table and logical-CNOT description than the full graph.

## Secondary comparisons

- full graph versus maximum-weight tree under Adam and exact natural gradient;
- random versus target-biased initialization;
- random target tree versus native chain;
- preparation resources across all four representations.

## Boundary-state replay rule

Every full-graph, target-biased Adam failure satisfying the frozen boundary-trap definition at state 199 is replayed from the identical stored parameter state with:

1. restarted Adam;
2. exact-natural Armijo;
3. target-ray projection followed by scalar cooling.

No confirmatory state satisfied the complete trap definition, so the replay set is empty.

## Stored diagnostics

Every recorded state includes normalized energy gap, planted and dominant-state probabilities, gradient RMS, target alignment, transverse displacement, effective Fisher rank and condition, and the update cosine with the projected target direction.

## Analysis integrity

The primary table is closed. Later graph heuristics, optimizer variants, or initialization rules are not added to the prespecified comparisons. The exact original protocol and source hashes remain in `frozen_source/` for independent verification.
