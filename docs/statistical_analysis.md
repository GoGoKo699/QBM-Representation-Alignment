# Statistical analysis of the confirmatory experiment

## Unit of analysis

The confirmatory experiment contains 24 independently generated Ising targets and five matched parameter seeds per target. The primary comparisons are paired by both instance and parameter seed.

For each hypothesis, the treatment-minus-control success difference is first calculated for every matched trajectory. The five seed-level differences are then averaged within each instance. The resulting 24 instance-level differences are the independent units used for uncertainty estimation.

This prevents the five optimization seeds for one target from being treated as five independent problem instances.

## Primary hypotheses

The three comparisons and their minimum practical effects were fixed before the confirmatory targets were generated:

| ID | Paired comparison | Minimum practical effect |
|---|---|---:|
| H1 | Max-weight target tree minus native chain, target-biased Adam | +0.15 |
| H2 | Max-weight target tree minus random target-supported tree, target-biased Adam | +0.10 |
| H3 | Max-weight target tree minus native chain, exact-natural oracle | +0.25 |

The exact locked protocol and source hashes are preserved under [`experiments/sparse_ising_confirmation/protocol/frozen_source/`](../experiments/sparse_ising_confirmation/protocol/frozen_source/).

## Instance-cluster bootstrap

For each hypothesis, 20,000 bootstrap samples are drawn by resampling the 24 instance-level differences with replacement. The point estimate is the mean paired difference across the original 24 instances.

The analysis reports:

- an unadjusted percentile interval;
- a two-sided bootstrap tail probability for a zero effect;
- Holm step-down adjusted probabilities across H1–H3;
- rank-specific step-down intervals used by the frozen decision rule.

The term **Holm-adjusted interval** in the result files refers to this prespecified step-down testing construction. It should not be interpreted as a single simultaneous confidence region for all possible post-hoc comparisons.

## Decision rule

A primary hypothesis passes only when:

1. its point estimate reaches the prespecified minimum practical effect; and
2. its Holm step-down interval excludes zero.

The practical threshold is applied to the point estimate, not to the lower interval endpoint. This distinction is intentional and was fixed before the confirmatory targets were generated.

## Results

| ID | Point difference | Holm step-down interval | Decision |
|---|---:|---:|---|
| H1 | +0.3250 | [+0.1583, +0.5083] | Pass |
| H2 | +0.1833 | [+0.0667, +0.3167] | Pass |
| H3 | +0.5167 | [+0.3500, +0.6667] | Pass |

Machine-readable results are in [`results/confirmatory/primary_effects.csv`](../results/confirmatory/primary_effects.csv). The maintained implementation is [`experiments/sparse_ising_confirmation/scripts/analyze_results.py`](../experiments/sparse_ising_confirmation/scripts/analyze_results.py).

## Interpretation boundary

The intervals quantify uncertainty over the generated instance ensemble represented by the 24 targets. They do not establish asymptotic scaling, universal superiority of the graph rule, or independence from the stated target distribution, initialization, optimizer, and finite evaluation budget.
