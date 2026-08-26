# Reproducibility

## Environment

Install the repository in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The maintained dependencies are listed in `pyproject.toml` and `requirements.txt`.

## Record the exact version

The complete archive is published as GitHub release [`v1.1.0`](https://github.com/GoGoKo699/QBM-Representation-Alignment/releases/tag/v1.1.0). The primary confirmed `MAXJ` result in its original release state is preserved as [`v1.0.0`](https://github.com/GoGoKo699/QBM-Representation-Alignment/releases/tag/v1.0.0).

Before running a calculation from a later checkout, record its commit:

```bash
git rev-parse HEAD
```

Citation formats and version-selection guidance are in [`CITATION.md`](../CITATION.md). Release contents are summarized in [`CHANGELOG.md`](../CHANGELOG.md).

## Fast validation

```bash
python scripts/validate_repository.py
python -m pytest -q
```

The repository validator checks that all five packaged scientific validation records have PASS status and that local documentation links resolve. To recompute the primary scientific record from the stored trajectories and parameter states, run:

```bash
python experiments/sparse_ising_confirmation/scripts/validate_experiment.py
```

That experiment validator checks the protocol and seed commitments, exact target spectra, stored parameter arrays, the energy-to-probability success certificate, exact-natural monotonicity, aggregate success counts, paired effects, and preparation-resource accounting. Each supporting study provides its own `scripts/validate_study.py`.

## Regenerate analyses and figures

The default command regenerates only the primary confirmatory tables and figures:

```bash
bash scripts/refresh_analysis.sh
```

Supporting studies are intentionally separate because their analysis scripts can take longer:

```bash
bash scripts/refresh_analysis.sh boundary
bash scripts/refresh_analysis.sh finite
bash scripts/refresh_analysis.sh partial
bash scripts/refresh_analysis.sh temperature
```

Use `bash scripts/refresh_analysis.sh all` to run every maintained compact analysis in sequence. These commands read packaged data and do not rerun the expensive optimization trajectories or the exhaustive tree-temperature calculation.

## Re-run the primary experiment

```bash
python experiments/sparse_ising_confirmation/scripts/validate_seed_commitment.py
python experiments/sparse_ising_confirmation/scripts/run_confirmatory_cells.py --workers 8
python experiments/sparse_ising_confirmation/scripts/compute_preparation_resources.py
python experiments/sparse_ising_confirmation/scripts/analyze_results.py
python experiments/sparse_ising_confirmation/scripts/make_figures.py
python experiments/sparse_ising_confirmation/scripts/validate_experiment.py
```

The raw logs and compressed parameter-state arrays are written inside the experiment directory. Compact public results are written to `results/confirmatory/`.

## Re-run the exhaustive temperature-tree study

```bash
python studies/temperature_tree_geometry/scripts/run_exhaustive_study.py --clean-results
python studies/temperature_tree_geometry/scripts/analyze_results.py
python studies/temperature_tree_geometry/scripts/validate_study.py
```

The exhaustive run enumerates all target-supported spanning trees over 61 temperatures for ten $n=8$ development instances. It generates large intermediate arrays and a long-form table locally. These generated intermediates are excluded from ordinary Git history; the repository includes compact canonical summaries, the protocol, validation records, and the figure-generation script.

## Frozen evidence

`experiments/sparse_ising_confirmation/protocol/frozen_source/` preserves the exact locked protocol and generator source used before the separately generated confirmatory targets were created. These files intentionally retain their original wording and hashes. Maintained scripts outside that directory use the public repository layout.

The temperature-tree study separately preserves its protocol in [`studies/temperature_tree_geometry/protocol.md`](../studies/temperature_tree_geometry/protocol.md). It is a later development study and is not part of the primary frozen comparison.

## Determinism boundaries

Instance generation, parameter seeds, graph construction, exact objectives, and bootstrap seeds are fixed. Floating-point eigendecompositions and timestamps in newly generated external environments can vary at final-bit or file-byte level; scientific validation therefore uses explicit tolerances in addition to hashes.
