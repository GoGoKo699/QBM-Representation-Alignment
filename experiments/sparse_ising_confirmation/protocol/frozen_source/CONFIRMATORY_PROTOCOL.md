# Preregistered confirmatory protocol

## Status

This protocol is frozen before confirmatory instances are generated or optimized. The four engineering seeds may be used only to verify that code executes and output schemas are correct. Their outcomes are excluded from scientific analysis.

## Confirmatory question

> Does a problem-aligned spanning-tree QBM outperform generic chain and random-tree controls of the same treewidth and pair-parameter count on an independent weighted sparse-Ising ensemble, and does exact natural geometry preserve the same representation ordering?

## Independent problem family

### Target Hamiltonian

For each instance,

\[
C(z)=\sum_{i=1}^{16}h_i z_i+\sum_{(i,j)\in E}J_{ij}z_i z_j,
\qquad z_i\in\{-1,+1\}.
\]

### Graph ensemble

- \(n=16\);
- connected random 3-regular graph;
- graph generated independently for each preregistered instance seed;
- a deterministic random relabeling is applied before representations are constructed.

### Coefficients

For every target edge,

\[
J_{ij}=s_{ij}a_{ij},
\]

where \(s_{ij}\in\{-1,+1\}\) is uniform and \(a_{ij}\sim\mathrm{Uniform}(0.5,1.5)\).

Local fields are

\[
h_i\sim\mathrm{Uniform}(-0.35,0.35).
\]

The complete coefficient vector is divided by its root-mean-square magnitude, making the typical coefficient scale equal to one.

### Acceptance rule

An instance is accepted when:

1. the graph is connected;
2. exact enumeration finds one unique ground state;
3. the first spectral gap satisfies \(\gamma\ge0.05\) after RMS normalization.

A failed attempt is replaced deterministically by advancing the attempt counter within the same preregistered instance seed. Acceptance depends only on graph/spectral properties, never on QBM optimization outcomes.

### Endpoint

Success is

\[
\frac{\Delta E}{\gamma}\le0.1.
\]

The spectral certificate then gives

\[
p_\star\ge0.9.
\]

## Frozen instance counts

- Engineering-only instances: 4.
- Confirmatory instances: 24.
- Parameter seeds per instance: 5.

Parameter seeds:

```text
0, 19, 42, 50, 101
```

Instance seeds are frozen in `CONFIRMATORY_SEED_COMMITMENT.json`.

## Frozen representation ladder

### G1 — Native chain

\[
(0,1),(1,2),\ldots,(14,15).
\]

The graph labeling is randomized during instance generation, so this is a generic hardware-local control rather than a hand-selected ordering.

### G2 — Random problem spanning tree

Assign independent random priorities to the target edges and take the resulting Kruskal spanning tree. Priorities are independent of \(|J_{ij}|\).

This controls for:

- graph width;
- edge count;
- parameter count;
- use of only target-supported edges.

### G3 — Maximum-weight problem spanning tree

Take a maximum spanning tree using edge weights

\[
|J_{ij}|.
\]

All tie-breaking is lexicographic.

### G4 — Full target graph

Retain all target interactions.

## Frozen initialization

One canonical all-pairs Gaussian vector is generated for each parameter seed and restricted to the active coordinates of every representation.

### Random

\[
\theta^{(0)}=0.3\xi.
\]

### Target biased

\[
\theta^{(0)}=c_G+0.3\xi,
\]

where \(c_G\) is the target coefficient vector projected onto the retained features.

## Frozen optimizer cells

### Adam

Run on all four representations and both initializations.

Hyperparameters:

```text
learning rate 0.02
beta1        0.9
beta2        0.999
epsilon      1e-8
```

### Exact natural-gradient Armijo oracle

Run on all four representations from target-biased initialization only.

- Moore–Penrose Fisher inverse;
- direction norm cap \(0.5\|c_G\|_2\);
- exact Armijo energy evaluation;
- same pseudoinverse and line-search conventions as checkpoint 7.

This is an oracle ceiling, not a practical quantum-cost claim.

## Budgets

- Primary budget: 200 recorded states.
- No optimizer hyperparameter may be changed after engineering smoke tests.
- Confirmatory failures are not extended to 1,000 states for the primary success table.

## Frozen primary comparisons

All comparisons are paired by instance and parameter seed.

### H1 — Problem tree versus native chain

Target-biased Adam success on G3 exceeds target-biased Adam success on G1.

### H2 — Problem tree versus random problem tree

Target-biased Adam success on G3 exceeds target-biased Adam success on G2.

This is the cleanest test that interaction selection by target weight matters beyond merely using a target-supported tree.

### H3 — Representation ordering under exact geometry

Target-biased exact-natural success on G3 exceeds exact-natural success on G1.

### Secondary comparisons

- G4 versus G3 under Adam and exact natural gradient;
- random versus target-biased initialization within each representation;
- G2 versus G1;
- preparation resource frontier across G1–G4.

## Causal replay rule

For every confirmatory G4 target-biased Adam failure that satisfies the frozen checkpoint-5 trap definition at record 199, replay the identical state using:

1. restarted Adam;
2. exact-natural Armijo;
3. projection onto the target ray followed by scalar cooling.

The replay budget is 1,000 recorded states. All qualifying states are replayed; none may be selected by visual appearance.

## Stored diagnostics

At every recorded state, store:

- energy gap and normalized gap;
- planted-state probability;
- dominant-state probability and energy;
- gradient RMS;
- target cosine;
- transverse ratio;
- effective Fisher condition number;
- effective Fisher rank;
- update cosine with the projected target direction.

For exact-natural cells, also store the omitted-cost covariance norm

\[
\|r_G\|_2
\]

and the cosine between \(c_G\) and the exact natural direction.

## Statistical analysis

- Unit of resampling: instance.
- All five parameter seeds remain together when an instance is resampled.
- 20,000 hierarchical bootstrap resamples.
- Report paired point differences and 95% intervals.
- Report method-only and control-only wins.
- Apply Holm correction across H1–H3.
- Report raw counts in every table.

## Confirmatory go/no-go thresholds

### Full GO

All conditions hold:

1. H1 point difference is at least +15 percentage points and its corrected interval excludes zero.
2. H2 point difference is at least +10 points and its corrected interval excludes zero.
3. H3 point difference is at least +25 points and its corrected interval excludes zero.
4. The preparation cost of G3 remains width one and materially below G4.

A fresh causal trap replay is desirable but not required for the representation claim.

### Narrow-paper outcome

H1 confirms, but H2 or H3 does not. The paper is narrowed to a QBM-specific mechanism and problem-supported sparse representations, without claiming that maximum-weight tree selection is generally superior.

### No-go / reframe

Any of the following occurs:

- G3 does not outperform G1;
- G3 does not outperform G2 and the difference is practically negligible;
- exact geometry does not separate the representations;
- results reverse direction on the fresh ensemble.

In that case, the broad representation claim is abandoned. Checkpoint 5 may still support a narrow causal case study of boundary geometry.

## Analysis lock

No PCG method, new Fisher block, new graph heuristic, or new initialization may be added before the confirmatory table is finalized. New methods may be studied only after the preregistered outcomes are reported unchanged.
