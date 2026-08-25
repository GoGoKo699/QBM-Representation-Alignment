# Locked-protocol implementation decisions

These decisions resolve implementation details that were implicit in the frozen protocol. They were written before any confirmatory optimization outcome was generated.

1. **Random problem spanning tree.** Independent `Uniform(0,1)` priorities are generated in the stored target-edge order from `random_tree_seed`. Standard Kruskal processing is ascending in priority, with the edge tuple as the deterministic secondary key.
2. **Maximum-weight problem tree.** Edges are processed by descending `abs(J_ij)`, with lexicographic edge order as the deterministic tie-breaker.
3. **Canonical parameter noise.** `numpy.random.default_rng(parameter_seed)` generates one length-136 vector (`16` fields followed by all `120` lexicographically ordered pairs), multiplied by `0.3`; each representation restricts this vector to its active coordinates.
4. **Effective Fisher spectrum.** Eigenvalues exceeding `1e-12` times the largest eigenvalue are retained for rank and condition-number diagnostics, matching checkpoint 5.
5. **Holm-adjusted intervals.** H1–H3 are ordered by their two-sided instance-bootstrap p-values. Step-down confidence levels use alpha divided by `3`, `2`, and `1` in that order. Holm-adjusted p-values are also reported.
6. **Random-tree preparation.** Every spanning tree uses a leaf-elimination order, giving exact width one and 31 conditional-angle entries for 16 variables under the compiler accounting used in checkpoints 2–3.
7. **Material preparation distinction.** G4 is operationally satisfied when every problem tree has exact width one, every full graph has exact width at least two, and the median problem-tree/full ratios for both conditional-angle entries and Gray-code CNOT upper bounds are at most 0.5. This threshold was written before confirmatory resource values or success aggregates were inspected.
