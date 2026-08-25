# Finite-sample geometry in an aligned Gibbs family

This supporting study asks which geometry-aware updates remain effective when QBM gradients and covariance matrices are estimated from finite Gibbs samples.

## Main findings

For an aligned commuting model with `C = c0 + c^T F`, a gradient and Fisher matrix estimated from the same sample batch satisfy

```text
g_hat = -I_hat c
```

exactly up to floating-point error. Same-batch full Fisher therefore recovers the component of the known target direction visible in the empirical covariance range. It substantially outperforms independently estimated gradient/Fisher batches at low sample counts.

The identity is not a generic claim about Fisher estimation. At mature wrong-state boundaries the empirical covariance rank collapses, and sampled full Fisher cannot recover missing escape directions. A target-direction-plus-residual update remains robust there.

## Reproduce compact results

From the repository root:

```bash
python studies/finite_sample_geometry/scripts/analyze_results.py
python studies/finite_sample_geometry/scripts/make_figures.py
python studies/finite_sample_geometry/scripts/validate_study.py
```

Raw and canonical tables are under [`results/finite_sample_geometry/`](../../results/finite_sample_geometry/). Saved boundary states are shared with [`studies/boundary_geometry/`](../boundary_geometry/) rather than duplicated.

## Citation

Cite the repository using [`CITATION.md`](../../CITATION.md) and include the release tag or full commit SHA used. This is a supporting estimator study, not a separate primary confirmation.
