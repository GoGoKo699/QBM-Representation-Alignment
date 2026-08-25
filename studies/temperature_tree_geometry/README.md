# Temperature-Dependent Tree Geometry

This supporting study asks a narrow follow-up question to the repository's confirmed maximum-weight-tree result:

> If a width-one tree is selected by the fraction of the target Gibbs cooling direction that it retains, does the preferred tree change with temperature, and does that change improve the compressed Gibbs state?

The study exhaustively evaluates all target-supported spanning trees for ten reused $n=8$ positive Exact-1-in-3-SAT instances. It is developmental evidence, not an additional independent confirmation.

## Result

The temperature-dependent geometry is real, but the proposed selector is not operationally useful on this corpus.

- The cooling-power-optimal tree changes with temperature on all ten instances.
- The geometric headroom over the hot-start maximum-weight set is substantial: the median maximum recovered cooling defect is **74.2%**.
- The cooling-power and forward-KL optima are distinct on all ten instances at the certification temperature.
- Nevertheless, the cooling-power-optimal tree has a **worse projected target-energy gap on all ten instances** than both the best hot-optimal tree and the forward-KL-optimal tree.

At the certification temperature, the mean projected gaps are

```text
forward-KL-optimal tree : 0.2655
best hot-optimal tree   : 0.2870
cooling-power optimum   : 0.4744
```

The supported conclusion is therefore negative and specific:

> Retained target-state cooling power is a valid local geometric quantity, but it is not a reliable tree-selection objective for compressed Gibbs-state representation on the tested corpus.

This result does **not** weaken the repository's primary confirmed claim. The primary experiment shows that the deterministic maximum-absolute-coupling tree is an effective preparation-matched heuristic under its prospectively frozen optimization benchmark. This study asks a different question: whether another target-state geometric criterion can systematically improve upon that heuristic.


## Evidence map

- [Frozen protocol](protocol.md)
- [Scientific report](report.md)
- [Input provenance](provenance.md)
- [Machine-readable verdict](../../results/temperature_tree_geometry/pt1a_verdict.json)
- [Temperature-path summary](../../results/temperature_tree_geometry/temperature_path_summary.csv)
- [Certification-temperature summary](../../results/temperature_tree_geometry/certification_temperature_summary.csv)
- [Instance gate summary](../../results/temperature_tree_geometry/instance_gate_summary.csv)
- [Mechanism correlation summary](../../results/temperature_tree_geometry/mechanism_correlation_summary.csv)
- [Mechanism summary](../../results/temperature_tree_geometry/mechanism_summary.json)
- [Detailed validation](../../results/temperature_tree_geometry/detailed_validation.json)
- [Independent validation](../../results/temperature_tree_geometry/independent_validation.json)
- [Repository validation record](../../results/temperature_tree_geometry/validation.json)

## Regenerate the study

Install the repository normally, then run the exhaustive calculation:

```bash
python studies/temperature_tree_geometry/scripts/run_exhaustive_study.py --clean-results
python studies/temperature_tree_geometry/scripts/analyze_results.py
python studies/temperature_tree_geometry/scripts/validate_study.py
```

The exhaustive script generates large intermediate arrays and a long-form table locally. Those generated intermediates are intentionally not part of the ordinary repository checkout. The compact canonical tables and validation records needed to inspect the conclusion are included. Figures are regenerated from those tables by the compact analysis command.

To regenerate only the compact report and figures from the packaged summary tables:

```bash
bash scripts/refresh_analysis.sh temperature
```

## Interpretation boundary

- The instances were reused from earlier development work.
- No adaptive rewiring experiment is claimed.
- The post-hoc metric correlations are descriptive rather than frozen selection tests.
- A different on-manifold tree criterion would be a new hypothesis and would require independent confirmation.
