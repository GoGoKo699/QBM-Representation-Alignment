# Weighted sparse-Ising confirmation

This is the repository's primary independent experiment. It tests whether a target-informed sparse Gibbs representation improves finite-budget QBM optimization while retaining exact low-width preparation.

## Design

Twenty-four connected weighted 3-regular Ising targets at `n=16` are compared using four representations:

- a native chain;
- a random target-supported spanning tree;
- a maximum-absolute-coupling target spanning tree;
- the full target graph.

The chain and both trees have the same treewidth, pair-term count, and parameter count. Optimizers and initializations follow the prespecified protocol in [`protocol/protocol.md`](protocol/protocol.md).

Success is defined by

```text
(expected energy - ground energy) / spectral gap <= 0.1
```

which certifies planted-ground-state probability at least 0.9 for the unique-ground-state targets.

## Main confirmed results

Under target-biased Adam, the maximum-weight problem tree improves success by 32.5 percentage points over the chain and by 18.33 points over the random target tree. Under exact natural gradient, it improves by 51.67 points over the chain. All three prespecified paired effects exceed their practical thresholds and have multiplicity-adjusted intervals above zero.

The full graph remains the trainability ceiling, while the problem tree retains exact width-one inference and a 31-angle q-sample description.

## Reproduction

```bash
python scripts/validate_seed_commitment.py
python scripts/analyze_results.py
python scripts/make_figures.py
python scripts/validate_experiment.py
```

The raw trajectory logs and stored initial/final parameter arrays are under `results/`. Canonical compact tables are under the repository-level `results/confirmatory/` directory.
