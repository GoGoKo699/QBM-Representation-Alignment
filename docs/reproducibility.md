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

## Fast validation

```bash
python scripts/validate_repository.py
python -m pytest -q
```

The repository validator checks that all four packaged scientific validation records have PASS status and that local documentation links resolve. To recompute the primary scientific record from the stored trajectories and parameter states, run:

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
```

Use `bash scripts/refresh_analysis.sh all` to run every maintained analysis in sequence. These commands read the packaged raw trajectories and do not rerun the expensive optimization trajectories.

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

## Frozen evidence

`experiments/sparse_ising_confirmation/protocol/frozen_source/` preserves the exact locked protocol and generator source used before the independent instances were generated. These files intentionally retain their original wording and hashes. Maintained scripts outside that directory use the public repository layout.

## Determinism boundaries

Instance generation, parameter seeds, graph construction, exact objectives, and bootstrap seeds are fixed. Floating-point eigendecompositions and timestamps in newly generated external environments can vary at final-bit or file-byte level; scientific validation therefore uses explicit tolerances in addition to hashes.
