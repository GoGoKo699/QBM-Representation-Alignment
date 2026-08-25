# Exploratory elimination-bag Fisher approximation

## Status

This method was designed after the specified primary partial-alignment analysis showed that the original graph-star blocks did not retain full-Fisher performance on width-2 and width-3 graphs. It is therefore exploratory and is not included in the primary comparison.

## Construction

Given a stored elimination order, simulate fill-in. At each eliminated variable \(v\), form the variable bag

\[
B_v=\{v\}\cup N_+(v),
\]

where \(N_+(v)\) is the set of remaining neighbors at elimination.

A feature belongs to the bag block when its variable support is contained in \(B_v\). Thus each block contains:

- fields \(Z_i\) for variables in the bag;
- retained pair features \(Z_iZ_j\) whose endpoints are both in the bag.

For each block, solve the same shrinkage-regularized local covariance system used by the graph-star method. Add overlapping directions and divide each coordinate by its block multiplicity. The final direction is capped at \(0.5\|c_G\|\).

## Exact first-step diagnostic

Mean cosine with the exact natural direction on held-out starts:

| Graph | Graph-star | Elimination bag | Regularized full Fisher |
|---|---:|---:|---:|
| Chain | 0.570 | **0.935** | 0.962 |
| Problem tree | **0.753** | 0.605 | 0.812 |
| Width 2 | 0.690 | **0.718** | 0.894 |
| Width 3 | 0.656 | **0.741** | 0.912 |

The bag construction improves local first-step fidelity on chain, width-2, and width-3 graphs, but not on the problem tree.

## Held-out 256-sample performance

| Graph | Bag Fisher | Graph-star | Two-block | Full Fisher | Median bag storage |
|---|---:|---:|---:|---:|---:|
| Chain | 0% | 0% | 0% | 0% | 91 |
| Problem tree | **91.25%** | 88.75% | 91.25% | 93.75% | 91 |
| Width 2 | 46.25% | 48.75% | 47.5% | 71.25% | 268.5 |
| Width 3 | 56.25% | 55.0% | 57.5% | 82.5% | 419.5 |

On a problem tree, the method retains nearly all full-Fisher success while using approximately 82% fewer stored metric entries.

On width-2 and width-3 graphs, the larger separator-aware blocks do not close the trajectory-level performance gap. Better first-step cosine is insufficient.

## Interpretation

The negative result is useful. It suggests that the missing information is not captured by any fixed additive collection of small local blocks. Cross-block covariance must be propagated rather than discarded.

The next method should therefore use the bag approximation as a preconditioner for a matrix-free global Fisher solve, rather than treating it as a standalone metric.
