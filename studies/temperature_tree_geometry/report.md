# Temperature-dependent tree geometry: study report

## Verdict

The study completed successfully over **1,269,532 tree-temperature cells**: all 20,812 target-supported spanning trees for ten `n=8` instances at 61 normalized temperatures.

```text
GEOMETRY_ONLY_DO_NOT_USE_COOLING_POWER_AS_OPERATIONAL_SELECTOR
```

The temperature-dependent geometry is real and large, but the proposed cooling-power selector fails the operational gate. It should not be advanced as an operational tree-selection method on the basis of these data.

## Frozen gates

| Gate | Passing instances | Total | Project pass |
| --- | --- | --- | --- |
| A: finite-temperature evolution | 10 | 10 | True |
| B: useful cooling headroom | 10 | 10 | True |
| C: projected-energy relevance | 0 | 10 | False |
| D: distinction from KL optimum | 10 | 10 | True |

- Gate A passed on **10/10** instances.
- Gate B passed on **10/10**; the median maximum recovery of the defect left by the hot tree was **74.2%**.
- Gate C passed on **0/10**. Its median operational-interval energy improvement was **-0.1730**; negative values mean that the cooling-power-optimal tree produced a worse projected state.
- Gate D passed on **10/10**.

## Main findings

At `s=1`, the mean retained fraction rises from **0.961** for the best hot-optimal tree to **0.988** for the cooling-power optimum. This geometric improvement does not improve the represented state. The mean projected target-energy gaps are

```text
forward-KL-optimal tree : 0.2655
best hot-optimal tree   : 0.2870
cooling-power optimum   : 0.4744
```

The cooling-power optimum is worse than both alternatives on **10/10** and **10/10** instances, respectively.

The post-hoc median within-instance Spearman correlations with projected gap are

```text
retained cooling fraction : -0.035
tracking mismatch         : +0.799
forward KL                : +0.884
```

These correlations are descriptive rather than frozen selection tests.

## Conceptual diagnosis

The retained-cooling quantity is evaluated at the exact target state. Except when that state already belongs to the selected tree family, it is not the on-manifold cooling rate of the projected tree model. Large target-state tangent capture therefore need not imply small projection error or accurate self-contained thermal tracking.

## Certification-temperature table

| Instance | Hot A | Q-opt A | Hot gap | Q-opt gap | KL-opt gap | Hot Xi | Q-opt Xi | KL-opt Xi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n8i1 | 0.9129 | 0.9792 | 0.2290 | 0.3415 | 0.2290 | 0.2935 | 0.5325 | 0.3098 |
| n8i2 | 0.9612 | 0.9892 | 0.2907 | 0.6476 | 0.2422 | 0.3370 | 1.1049 | 0.4361 |
| n8i3 | 0.9696 | 0.9916 | 0.3087 | 0.3150 | 0.2718 | 0.3488 | 0.5728 | 0.4653 |
| n8i4 | 0.9594 | 0.9851 | 0.2670 | 0.5904 | 0.2670 | 0.4647 | 0.8136 | 0.5072 |
| n8i5 | 0.9757 | 0.9949 | 0.2005 | 0.2500 | 0.1695 | 0.2374 | 0.5236 | 0.2567 |
| n8i6 | 0.9661 | 0.9901 | 0.3236 | 0.6455 | 0.3110 | 0.5457 | 0.9203 | 0.6112 |
| n8i7 | 0.9410 | 0.9777 | 0.2498 | 0.6095 | 0.2491 | 0.4039 | 1.3757 | 0.4289 |
| n8i8 | 0.9891 | 0.9960 | 0.3236 | 0.4207 | 0.3321 | 0.4890 | 0.6345 | 0.5720 |
| n8i9 | 0.9745 | 0.9827 | 0.2386 | 0.3121 | 0.2386 | 0.3878 | 0.5733 | 0.4302 |
| n8i10 | 0.9607 | 0.9983 | 0.4387 | 0.6114 | 0.3449 | 0.5344 | 1.2126 | 0.5859 |

## Interpretation

Finite-temperature covariance geometry substantially reorders sparse trees, and its optimum differs from the forward-KL optimum. However, maximizing target-state retained cooling power systematically worsens the projected target energy and increases target-model tracking mismatch on this corpus.

This result does not weaken the repository's primary confirmed result. The `MAXJ` claim is an optimizer-and-preparation result under a frozen benchmark, not a claim that `MAXJ` remains geometrically optimal at every temperature.

## Reproducibility

From the repository root:

```bash
python studies/temperature_tree_geometry/scripts/run_exhaustive_study.py --clean-results
python studies/temperature_tree_geometry/scripts/analyze_results.py
python studies/temperature_tree_geometry/scripts/validate_study.py
```

The exhaustive run generates large intermediates locally. The ordinary repository checkout contains the compact summaries and validation records.
