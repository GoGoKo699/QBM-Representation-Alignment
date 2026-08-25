# How to cite this repository

If you use the software, data, figures, protocols, or derived results, cite the repository and identify the exact version or commit used.

## Recommended citation

> Lin, R. (2026). *Representation Alignment in Commuting Quantum Boltzmann Machines* (Version 1.1.0) [Computer software]. GitHub. https://github.com/GoGoKo699/QBM-Representation-Alignment

GitHub also displays a **Cite this repository** control generated from [`CITATION.cff`](CITATION.cff).

## BibTeX

```bibtex
@software{lin_2026_qbm_representation_alignment,
  author  = {Ruge Lin},
  title   = {Representation Alignment in Commuting Quantum Boltzmann Machines},
  year    = {2026},
  version = {1.1.0},
  url     = {https://github.com/GoGoKo699/QBM-Representation-Alignment},
  license = {BSD-3-Clause}
}
```

## Which version to cite

- **Primary confirmed `MAXJ` result:** cite the preserved [`v1.0.0`](https://github.com/GoGoKo699/QBM-Representation-Alignment/releases/tag/v1.0.0) release. It contains the prospectively frozen weighted sparse-Ising confirmation and its complete evidence chain.
- **Expanded archive:** cite Version 1.1.0 together with a matching release tag when available. Otherwise include the full commit SHA so the cited state is unambiguous.
- **Pre-boundary-study state:** the branch `archive/confirmed-result-2026-08-25` preserves the repository immediately before the temperature-dependent supporting study was added.

To record the exact commit of a local checkout, run:

```bash
git rev-parse HEAD
```

## Citing an exact computational result

- The primary confirmation is under [`experiments/sparse_ising_confirmation/`](experiments/sparse_ising_confirmation/), with canonical tables in [`results/confirmatory/`](results/confirmatory/).
- The later exhaustive graph-selection boundary study is under [`studies/temperature_tree_geometry/`](studies/temperature_tree_geometry/), with compact tables in [`results/temperature_tree_geometry/`](results/temperature_tree_geometry/).
- For a particular table or figure, cite the repository and name the file path, release tag, or full commit SHA used.

## Future paper citation

There is currently no preferred journal-paper citation. If a paper or archival DOI becomes available, `CITATION.cff` and this file will identify it explicitly. Until then, cite the versioned repository.
