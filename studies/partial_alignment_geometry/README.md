# Partial-alignment geometry

This supporting study asks what remains of geometry-aware QBM optimization when the ansatz graph represents only part of the target Hamiltonian.

For retained features `F_G`, the cost decomposes as

```text
C = c0 + c_G^T F_G + R_G
```

and the energy gradient becomes

```text
grad E = -I_G c_G - Cov(F_G, R_G).
```

The omitted-cost covariance is state dependent, so the exact natural direction is no longer a fixed projected target vector.

## Main findings

- A native chain is nearly unsolvable even under exact natural gradient, despite being able to represent a successful field-only state.
- A problem-supported spanning tree of the same width and parameter count is highly trainable.
- Local Fisher blocks retain nearly all full-Fisher success on the problem tree.
- On width-2 and width-3 cyclic representations, independently solved local blocks remain materially below the globally coupled Fisher solve.
- Same-batch full Fisher is especially useful at low sample counts, although its aligned finite-sample cancellation no longer applies exactly.

## Reproduce compact results

```bash
python scripts/analyze_results.py
python scripts/make_figures.py
python scripts/validate_study.py
```

Graphs are stored under `graphs/`; canonical tables are under `../../results/partial_alignment_geometry/`. The twenty shared instances are not duplicated here.
