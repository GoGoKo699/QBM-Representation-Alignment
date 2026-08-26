# How to cite this repository

If you use the software, data, figures, protocols, or derived results, cite the repository and identify the exact release or commit used.

## Recommended citation

> Lin, R. (2026). *Representation Alignment in Commuting Quantum Boltzmann Machines* (Version 1.1.0) [Computer software]. GitHub. https://github.com/GoGoKo699/QBM-Representation-Alignment/releases/tag/v1.1.0

GitHub also displays a **Cite this repository** control generated from [`CITATION.cff`](CITATION.cff).

## BibTeX

```bibtex
@software{lin_2026_qbm_representation_alignment,
  author  = {Ruge Lin},
  title   = {Representation Alignment in Commuting Quantum Boltzmann Machines},
  year    = {2026},
  version = {1.1.0},
  url     = {https://github.com/GoGoKo699/QBM-Representation-Alignment/releases/tag/v1.1.0},
  license = {BSD-3-Clause}
}
```

## Which version to cite

- **Complete research archive:** cite the published [`v1.1.0`](https://github.com/GoGoKo699/QBM-Representation-Alignment/releases/tag/v1.1.0) release. It includes the prospectively frozen weighted sparse-Ising confirmation, all supporting studies, the later temperature-dependent tree boundary study, and the reviewer-facing citation and navigation improvements.
- **Primary confirmed `MAXJ` result in its original release state:** cite [`v1.0.0`](https://github.com/GoGoKo699/QBM-Representation-Alignment/releases/tag/v1.0.0).
- **Pre-boundary-study state:** the branch `archive/confirmed-result-2026-08-25` preserves the repository immediately before the temperature-dependent supporting study was added.
- **A later unreleased state:** cite the most recent release and include the full commit SHA used.

To record the exact commit of a local checkout, run:

```bash
git rev-parse HEAD
```

## Citing an exact computational result

- The primary confirmation is under [`experiments/sparse_ising_confirmation/`](experiments/sparse_ising_confirmation/), with canonical tables in [`results/confirmatory/`](results/confirmatory/).
- The later exhaustive graph-selection boundary study is under [`studies/temperature_tree_geometry/`](studies/temperature_tree_geometry/), with compact tables in [`results/temperature_tree_geometry/`](results/temperature_tree_geometry/).
- For a particular table or figure, cite the release and name the file path. Include a full commit SHA only when the cited state differs from the tagged release.

## Release history

See [`CHANGELOG.md`](CHANGELOG.md) for the scientific and repository-level contents of each release.

## Future paper citation

There is currently no preferred journal-paper citation. If a paper or archival DOI becomes available, `CITATION.cff` and this file will identify it explicitly. Until then, cite the versioned repository.
