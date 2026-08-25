# Confirmatory implementation decisions

These details define the maintained implementation corresponding to the prespecified protocol.

1. **Spin convention.** Bit `0` maps to spin `+1`; bit `1` maps to spin `-1`.
2. **Feature order.** Local fields come first, followed by lexicographically ordered active pair features.
3. **Canonical random coordinates.** One Gaussian vector is generated in the complete field-plus-all-pairs coordinate system and restricted to each representation.
4. **Fisher spectrum.** Eigenvalues exceeding `1e-12` times the largest eigenvalue are retained for rank and condition diagnostics.
5. **Natural-gradient pseudoinverse.** Eigenvalues exceeding `1e-10` times the largest are retained in the Moore-Penrose solve.
6. **Tree preparation.** Every spanning tree uses a leaf-elimination order, giving exact width one and thirty-one conditional-angle entries for sixteen variables under the q-sample compiler accounting.
7. **Full-graph preparation.** A deterministic min-fill elimination order is used for exact table accounting.
8. **Random tree.** Independent pseudorandom edge priorities are generated from the instance-specific committed seed; Kruskal selection and lexicographic tie-breaking are deterministic.
9. **Maximum-weight tree.** Kruskal selection uses descending `abs(J_ij)` with lexicographic tie-breaking.
10. **Recorded states.** Parameters and diagnostics are evaluated at states `0` through `199`; no unrecorded final update is used for success.
