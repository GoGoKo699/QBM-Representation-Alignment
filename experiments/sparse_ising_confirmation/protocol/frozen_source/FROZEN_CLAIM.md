# Frozen paper claim

**Status:** frozen before any fresh confirmatory ensemble is generated or optimized.  
**Date:** 25 August 2026.

## Working title

**Representation Alignment and Boundary Geometry in Commuting Quantum Boltzmann Machines**

Alternative titles retained only for later editorial choice:

1. **Problem-Aligned Sparse Gibbs Models for Trainable Quantum Boltzmann Optimization**
2. **Beyond Barren Plateaus: Representation Mismatch and Boundary Traps in Quantum Boltzmann Machines**
3. **Which Interactions Matter? Trainability and Exact Preparation in Commuting Quantum Boltzmann Machines**

## One-sentence primary claim

> In commuting quantum Boltzmann machines for discrete optimization, finite-budget trainability is governed primarily by the alignment between the ansatz feature span and the target Hamiltonian: full alignment turns natural-gradient descent into target-Hamiltonian cooling, partial alignment adds a state-dependent omitted-cost covariance term, and problem-aligned spanning-tree models can substantially improve optimization over generic chains of identical width and parameter count while retaining exact low-width state preparation; Euclidean adaptive updates may instead accumulate transverse drift and concentrate on excited probability-simplex boundaries.

## Three load-bearing subclaims

### Theoretical subclaim T1 — aligned and partially aligned geometry

For a commuting Gibbs family

\[
 p_\theta(z)=\frac{e^{-\theta^{\mathsf T}F_G(z)}}{Z(\theta)},
\]

with target decomposition

\[
 C(z)=c_0+c_G^{\mathsf T}F_G(z)+R_G(z),
\]

one has

\[
 \nabla E=-I_Gc_G-r_G,
 \qquad
 r_G=\operatorname{Cov}(F_G,R_G).
\]

Full alignment is the special case \(R_G=0\), yielding

\[
 \nabla E=-Ic,
 \qquad
 -I^{-1}\nabla E=c.
\]

The project does not claim this exponential-family identity as a standalone new theorem. Its scientific use is to separate representation error from optimizer error.

### Mechanistic subclaim M1 — excited-boundary failure

In the fully aligned family, Adam can leave the exact cooling manifold, build a transverse log-odds advantage for a gap-one excited state, and approach a nearly singular probability-simplex boundary. Same-state interventions show that restoring the natural direction or projecting back to the cooling ray rescues the diagnosed failures, while restarted Adam, line-search Euclidean descent, and diagonal Fisher scaling do not.

The paper will call this an **excited-boundary trap** under the frozen operational definition from checkpoint 5. It will not call it a barren plateau.

### Constructive subclaim C1 — sparse problem alignment

A maximum-weight problem spanning tree has the same treewidth and pair-parameter count as a generic chain, but retains problem interactions. On the current Exact-1-in-3-SAT evidence it substantially improves trainability while preserving exact width-one Gibbs inference and q-sample preparation.

The confirmatory experiment must test this claim on a fresh weighted sparse-Ising ensemble with a random-tree control.

## Supporting, noncentral subclaims

1. The spectral-gap certificate makes the optimization endpoint meaningful:
   \[
   p_\star\ge 1-\Delta E/\gamma.
   \]
2. Exact natural gradient supplies an oracle ceiling that distinguishes representation mismatch from estimator failure.
3. Same-batch Fisher estimation has special finite-sample structure under full alignment, but this is an appendix-level result.
4. Exact preparation cost is controlled by the width of the chosen Gibbs representation, not merely by system size.

## Explicit nonclaims

The manuscript will not claim:

- a demonstrated barren plateau;
- quantum advantage;
- polynomial-time solution of general Exact Cover or Ising optimization;
- universal superiority of natural gradient;
- generic hardness of cusQBM state preparation;
- monotonic trainability as a function of treewidth;
- novelty of the local-\(Y\) EQBM ansatz;
- that the current certificate-tight family is an average-case hardness ensemble;
- that every QBM failure is an excited-boundary trap;
- that the fresh sparse-Ising ensemble represents industrial problem distributions.

## Claim-freezing rule

The one-sentence primary claim may be shortened editorially after the confirmatory study, but its logical content may not be strengthened. Any result that fails confirmation must be removed or narrowed rather than explained away through new post-hoc methods.
