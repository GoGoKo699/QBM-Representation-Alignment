# Research context and novelty boundary

This repository studies a commuting, classically tractable sector of quantum Boltzmann machines. Its contribution is a preparation-aware representation study, not a new definition of a QBM, a new natural-gradient formalism, or a claim of quantum speedup.

## Established foundations

- Amin *et al.*, [“Quantum Boltzmann Machine,”](https://doi.org/10.1103/PhysRevX.8.021050) *Physical Review X* **8**, 021050 (2018), introduced the QBM framework and training considerations for noncommuting thermal models.
- Patel *et al.*, [“Quantum Boltzmann machine learning of ground-state energies,”](https://arxiv.org/abs/2410.12935) analyzed QBM energy optimization and thermal-state gradient estimation.
- Patel and Wilde, [“Natural gradient and parameter estimation for quantum Boltzmann machines,”](https://doi.org/10.1103/j8nb-by4l) *Physical Review A* **112**, 052421 (2025), developed thermal-state information matrices and natural-gradient estimation procedures.
- Minervini, Patel, and Wilde, [“Evolved quantum Boltzmann machines,”](https://doi.org/10.1103/k2hw-r25g) *Physical Review A* **113**, 032427 (2026), introduced the broader evolved-QBM ansatz and its information geometry.
- McClean *et al.*, [“Barren plateaus in quantum neural network training landscapes,”](https://doi.org/10.1038/s41467-018-07090-4) *Nature Communications* **9**, 4812 (2018), established the standard random-circuit barren-plateau phenomenon. This repository does not claim that phenomenon.
- Peyrard *et al.*, [“Exact and approximate inference in graphical models: variable elimination and beyond,”](https://arxiv.org/abs/1506.08544) reviews variable elimination and the role of treewidth in exact inference.
- Möttönen *et al.*, [“Transformation of quantum states using uniformly controlled rotations,”](https://arxiv.org/abs/quant-ph/0407010) provides the controlled-rotation machinery used in logical q-sample resource accounting.

## What this repository adds

The repository combines four elements:

1. an exact decomposition of fully and partially aligned commuting Gibbs optimization geometry;
2. matched sparse-representation controls at fixed treewidth, interaction count, and parameter count;
3. a prespecified independent weighted sparse-Ising confirmation comparing a native chain, a random target-supported tree, a maximum-weight target-supported tree, and the full target graph;
4. exact logical q-sample preparation accounting for every confirmatory representation.

The maximum-weight spanning-tree algorithm itself is classical and is not claimed as new. The supported design result is empirical: retaining stronger target interactions improves finite-budget trainability over both a generic chain and a prespecified random target-supported tree while preserving width-one exact inference and q-sample preparation.

## Scope boundary

The numerical calculations use exact classical enumeration at the studied sizes. The quantum relevance lies in thermal-ansatz design and coherent q-sample or purification preparation. See [scientific claims](scientific_claims.md) and [limitations](limitations.md) for the precise claim hierarchy.
