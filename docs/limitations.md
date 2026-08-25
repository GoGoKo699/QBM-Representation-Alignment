# Scope and limitations

## Commuting and classically tractable sector

All principal models are diagonal commuting Gibbs families. The exact calculations can be performed classically at the studied sizes. The repository studies representation and preparation principles for thermal quantum ansätze; it does not demonstrate computational speedup.

## Finite size and finite budget

The independent experiment fixes `n=16` to isolate representation effects. It supports finite-budget comparisons, not asymptotic scaling claims.

## Target-informed initialization

The strongest representation ordering appears under target-biased initialization. Random Adam performs poorly across all sparse representations. The supported design principle is joint use of target-informed representation and initialization.

## Full graph remains more trainable

The problem tree does not match the full graph. It is a preparation-friendly compromise that improves substantially over generic sparse controls.

## Graph-selection rule

The maximum-absolute-coupling spanning tree is empirically effective in the prespecified ensemble. No theorem establishes its universal optimality for trainability, Gibbs approximation, or preparation–optimization Pareto performance.

## Temperature-dependent tree geometry

A later exhaustive development study finds that the tree maximizing retained target-state cooling power changes substantially with temperature. That geometric optimum is not operationally superior: at the certification temperature it gives a worse projected target-energy gap than both the best hot-optimal tree and the forward-KL-optimal tree on all ten reused instances. The repository therefore does not claim that retained cooling power is a valid adaptive tree-selection objective. See [`studies/temperature_tree_geometry/`](../studies/temperature_tree_geometry/).

## Logical preparation resources

Conditional-angle and CNOT counts are exact logical accounting under the stated compiler. They do not imply hardware-efficient preparation or robustness to finite rotation precision and noise.

## Barren plateaus

The studies demonstrate finite-budget failures, representation mismatch, boundary concentration in selected cases, and ill-conditioned Fisher geometry. They do not establish a standard asymptotic barren plateau.

## Boundary traps

Strong excited-boundary traps were causally reproduced in development instances, but none qualified under the complete frozen definition in the independent sparse-Ising confirmation. The mechanism is therefore a case study, not a universal failure model.
