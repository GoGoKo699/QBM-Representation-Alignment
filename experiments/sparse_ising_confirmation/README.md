# Weighted sparse-Ising confirmation

This is the repository's prospectively frozen primary experiment. Its protocol, seed commitments, graph rules, endpoints, and decision thresholds were fixed before a separately generated target ensemble was evaluated. It is an internal confirmation on unseen targets, not an external replication by another group.

## Design

Twenty-four connected weighted 3-regular Ising targets at `n=16` are compared using four representations:

- a native chain;
- a random target-supported spanning tree;
- a maximum-absolute-coupling target spanning tree;
- the full target graph.

The chain and both trees have the same treewidth, pair-term count, and parameter count. Optimizers and initializations follow the prespecified protocol in [`protocol/protocol.md`](protocol/protocol.md).

In source code and result tables, `problem_tree` denotes the deterministic maximum-absolute-coupling spanning tree, abbreviated `MAXJ` in the documentation.

Success is defined by

```text
(expected energy - ground energy) / spectral gap <= 0.1
```

which certifies ground-state probability at least 0.9 for the unique-ground-state targets.

## Main confirmed results

Under target-biased Adam, the maximum-weight problem tree improves success by 32.5 percentage points over the chain and by 18.33 points over the random target tree. Under the exact-natural oracle, it improves by 51.67 points over the chain. All three prespecified paired effects exceed their practical thresholds and have multiplicity-adjusted intervals above zero.

The exact-natural oracle uses the exact Fisher pseudoinverse and exact Armijo energy evaluation. It is a geometric ceiling, not a practical sampled-cost claim.

The full graph remains the trainability ceiling, while the problem tree retains exact width-one inference and a 31-angle q-sample description.

## Reproduction

From the repository root:

```bash
python experiments/sparse_ising_confirmation/scripts/validate_seed_commitment.py
python experiments/sparse_ising_confirmation/scripts/analyze_results.py
python experiments/sparse_ising_confirmation/scripts/make_figures.py
python experiments/sparse_ising_confirmation/scripts/validate_experiment.py
```

The raw trajectory logs and stored initial/final parameter arrays are under `results/`. Canonical compact tables are under the repository-level [`results/confirmatory/`](../../results/confirmatory/) directory.

## Citation

Cite the repository using [`CITATION.md`](../../CITATION.md). For the primary confirmed result, the preserved release is [`v1.0.0`](https://github.com/GoGoKo699/QBM-Representation-Alignment/releases/tag/v1.0.0); include a full commit SHA when citing a later repository state.
