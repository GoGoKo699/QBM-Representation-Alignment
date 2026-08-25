# Excited-boundary optimizer geometry

This supporting study isolates a finite-budget optimizer failure in a fully aligned commuting Gibbs family. The aligned family has no finite suboptimal stationary point and its exact natural-gradient direction is the target Hamiltonian direction, yet coordinatewise Adam can drift transversely and concentrate on an excited boundary state.

## Main findings

- Two long-run Adam failures concentrate about 99.5% probability on a gap-one excited assignment.
- Restarted Adam, Armijo Euclidean gradient descent, projected Adam, and diagonal Fisher do not rescue the saved states.
- Exact target-direction Armijo and explicit projection back to the scalar cooling ray rescue all saved states.
- The additional target-ray distance needed for rescue quantitatively matches the observed replay time.
- The complete frozen trap definition is stronger than ordinary optimization failure and is not claimed to describe every unsuccessful trajectory.

## Reproduce compact results

From the repository root:

```bash
python studies/boundary_geometry/scripts/analyze_results.py
python studies/boundary_geometry/scripts/make_figures.py
python studies/boundary_geometry/scripts/validate_study.py
```

The twenty shared instances are stored once under [`data/certificate_tight_instances/`](../../data/certificate_tight_instances/). Raw and derived tables are stored under [`results/boundary_geometry/`](../../results/boundary_geometry/).

## Citation

Cite the repository using [`CITATION.md`](../../CITATION.md) and include the release tag or full commit SHA used. This is a supporting mechanism study, not a separate primary confirmation.
