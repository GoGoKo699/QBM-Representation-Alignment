# Data formats

## Confirmatory instance manifest

`experiments/sparse_ising_confirmation/instances/manifest.json` stores 24 weighted Ising targets. Each record contains:

- local fields;
- target edges and couplings;
- exact ground-state index and spin string;
- exact ground energy and spectral gap;
- the prespecified random-tree seed.

## Raw trajectory summaries

Each `*_summary.csv` file contains one row per trajectory with:

- instance, graph, optimizer, initialization, and parameter seed;
- success and first-success state;
- minimum normalized energy gap;
- final energy, planted probability, dominant-state probability, and gradient diagnostics;
- effective Fisher rank and condition number;
- target alignment and transverse displacement;
- the frozen boundary-trap indicator.

## Dense logs

Each `*_logs.csv.gz` file records every evaluated state needed for exact-natural monotonicity and first-success checks.

## Stored parameters

Each `*_states.npz` file stores initial and final parameter arrays for every graph, optimizer, initialization, and seed cell. The validator recomputes energies and planted probabilities from these arrays.

## Canonical compact tables

`results/confirmatory/` contains the public one-copy summaries:

- `aggregate.csv`;
- `trajectory_summary.csv`;
- `primary_effects.csv`;
- `secondary_effects.csv`;
- `preparation_resources.csv`;
- `alignment_resources.csv`;
- `resource_pairs.csv`;
- `validation_spectra.csv`;
- `validation.json`.

Supporting-study tables use the same principle: raw evidence and compact derived summaries are stored once under their topic directory in `results/`.
