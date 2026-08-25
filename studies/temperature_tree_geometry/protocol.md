# Protocol — temperature-dependent tree geometry

**Frozen before inspecting the exhaustive tree-temperature optima.**  
**Date:** 25 August 2026  
**Status:** supporting development study performed after the primary weighted sparse-Ising confirmation. It does not modify the frozen primary comparisons or their interpretation.

## 1. Question

For each of the ten repository `n=8` positive Exact-1-in-3-SAT instances, does the width-one spanning tree that best retains the instantaneous Gibbs-cooling direction change as the target Gibbs state cools? Is any change large enough to improve the projected state's target energy? Is the cooling-optimal tree distinct from the forward-KL-optimal tree?

## 2. Frozen corpus and model class

- `instances/n8i1.txt` through `n8i10.txt`;
- every spanning tree of each target interaction graph;
- all eight local-field features and the seven pair features on the tree;
- no optimizer trajectory or previously observed tree-training outcome is used to select a tree.

The exhaustive count must equal the previously reported total of 20,812 tree instances.

## 3. Target Gibbs path

For target cost `H(x)` and ground energy `E0`,

```math
\tau_\beta(x)=\frac{e^{-\beta H(x)}}{Z_\beta}.
```

For each instance, `beta_cert` is the unique nonnegative solution of

```math
\mathbb E_{\tau_\beta}[H]-E_0=0.1.
```

The frozen normalized grid is

```text
s = beta / beta_cert = 0, 0.025, 0.050, ..., 1.500
```

with 61 points per instance.

## 4. Cooling geometry

For a tree `T`, let `F_T` contain the local spins and tree-edge products. At the target Gibbs state,

```math
K_T(\beta)=\operatorname{Cov}_{\tau_\beta}(F_T,F_T),
\qquad
b_T(\beta)=\operatorname{Cov}_{\tau_\beta}(F_T,H).
```

Using a symmetric Moore–Penrose pseudoinverse with relative eigenvalue cutoff `1e-12`, define

```math
Q_T=b_T^\mathsf{T}K_T^+b_T,
\qquad
Q_{\rm full}=\operatorname{Var}_{\tau_\beta}(H),
```

```math
A_T=Q_T/Q_{\rm full},
\qquad
R_T=Q_{\rm full}-Q_T.
```

`Q_T` is retained instantaneous cooling power, `A_T` its retained fraction, and `R_T` the cooling defect.

## 5. Forward-KL tree projection

For each tree, the exact forward-KL projection `sigma_{T,beta}=P_T tau_beta` is reconstructed from the target one-site and tree-edge marginals:

```math
\sigma_{T,\beta}(x)
=\prod_i\tau_\beta(x_i)
\prod_{(i,j)\in T}
\frac{\tau_\beta(x_i,x_j)}
{\tau_\beta(x_i)\tau_\beta(x_j)}.
```

The atlas records

```math
D_{\rm KL}(\tau_\beta\Vert\sigma_{T,\beta})
```

and the projected target-energy gap

```math
\mathbb E_{\sigma_{T,\beta}}[H]-E_0.
```

## 6. Target–model tracking mismatch

At `sigma_{T,beta}`, define

```math
K_T^\sigma=\operatorname{Cov}_{\sigma_{T,\beta}}(F_T,F_T),
```

```math
b_T^\sigma=\operatorname{Cov}_{\sigma_{T,\beta}}(F_T,H),
\qquad
b_T^\tau=\operatorname{Cov}_{\tau_\beta}(F_T,H),
```

and

```math
\Xi_T^2=(b_T^\sigma-b_T^\tau)^\mathsf{T}
(K_T^\sigma)^+
(b_T^\sigma-b_T^\tau).
```

The smallest retained eigenvalue of `K_T^sigma` is also recorded.

## 7. Tie handling

At `beta=0`, every tree maximizing `sum_{e in T} J_e^2` is hot-optimal. A deterministic `MAXJ-LEX` representative is the lexicographically first member of that complete set.

At each temperature:

- exact cooling optima satisfy `Q >= Q_max - max(1e-12, 1e-10 Q_full)`;
- the one-percentage-point near-optimal set satisfies `A >= A_max - 0.01`;
- exact KL optima satisfy `KL <= KL_min + max(1e-12, 1e-10 max(1, |KL_min|))`.

A change of one representative hash inside a tied set is not treated as structural evolution.

## 8. Best fixed tree

The best fixed tree in hindsight maximizes the arithmetic mean of `A_T` over all 61 normalized temperatures. It is an oracle reference, not an implementable selector.

## 9. Frozen gates

### Gate A — tie-aware finite-temperature evolution

An instance passes when there is a contiguous run of at least four nonzero grid points for which no hot-optimal tree lies within one percentage point of the cooling optimum. The project gate passes for at least five of ten instances.

### Gate B — useful cooling headroom

On `0.25 <= s <= 1.25`, an instance passes when both hold:

1. the maximum recovered fraction of the defect left by the best hot-optimal tree is at least `0.10`;
2. the mean cooling-power advantage over the best hot-optimal tree is at least `0.005 Q_full`.

The project gate additionally requires at least five passing instances and a median maximum defect recovery of at least `0.10`.

### Gate C — operational relevance

On `0.75 <= s <= 1.25`, compare the minimum projected-energy gap among exact cooling-optimal trees with the minimum gap among hot-optimal trees. An instance passes when the mean improvement is at least `0.01`. The project gate passes for at least five instances and requires a positive median improvement.

### Gate D — distinction from static KL compression

An instance passes when there is a contiguous run of at least four nonzero points where:

- the exact cooling-optimal and KL-optimal sets are disjoint; and
- the most cooling-effective KL-optimal tree loses at least `0.005 Q_full` relative to the cooling optimum.

The project gate passes for at least five instances.

## 10. Decision

- **A–D all pass:** a separate adaptive-tree experiment would be scientifically motivated, but it would require a new frozen protocol and independent confirmation.
- **A and B pass but C or D fails:** retain the geometry as a supporting boundary result; do not promote the cooling-power selector as an operational method.
- **A or B fails:** stop the cooling-power-driven selector branch.

This is developmental evidence on reused instances. Any new positive selector claim would require an independently frozen confirmation.
