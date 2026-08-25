#!/usr/bin/env python3
"""Regenerate compact summaries, figures, and the tree-geometry report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=[np.number]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{digits}f}")
    headers = [str(column) for column in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in formatted.itertuples(index=False, name=None):
        cells = [str(value).replace("|", r"\|") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def derive_compact_tables(results: Path) -> None:
    full_path = results / "tree_optima_by_temperature.csv"
    if not full_path.is_file():
        return
    optima = pd.read_csv(full_path)
    path = optima.groupby("s", as_index=False).agg(
        mean_a_qopt=("a_max", "mean"),
        mean_a_hot=("a_hot_best", "mean"),
        mean_a_klopt=("a_klopt_best", "mean"),
        mean_a_best_fixed=("a_best_fixed", "mean"),
        mean_a_maxj=("a_maxj", "mean"),
        median_hot_defect_recovery=("hot_defect_recovery", "median"),
        mean_gap_improvement_over_hot=("gap_improvement_over_hot", "mean"),
        median_gap_improvement_over_hot=("gap_improvement_over_hot", "median"),
        fraction_hot_absent_q_near=("hot_in_q_near_set", lambda values: 1.0 - float(np.mean(values))),
        fraction_q_kl_disjoint=("qopt_klopt_intersect", lambda values: 1.0 - float(np.mean(values))),
        mean_gap_qopt=("gap_qopt_best", "mean"),
        mean_gap_hot=("gap_hot_best", "mean"),
        mean_gap_klopt=("gap_klopt_best", "mean"),
        mean_gap_best_fixed=("gap_best_fixed", "mean"),
    )
    path.to_csv(results / "temperature_path_summary.csv", index=False, float_format="%.12g")

    cert = optima[np.isclose(optima.s, 1.0)][
        [
            "instance",
            "a_hot_best",
            "a_max",
            "gap_hot_best",
            "gap_qopt_best",
            "gap_klopt_best",
            "xi_hot_min",
            "xi_qopt_min",
            "xi_klopt_min",
            "qopt_klopt_intersect",
            "hot_defect_recovery",
        ]
    ].copy()
    cert.columns = [
        "instance",
        "a_hot",
        "a_qopt",
        "gap_hot",
        "gap_qopt",
        "gap_klopt",
        "xi_hot",
        "xi_qopt",
        "xi_klopt",
        "qopt_klopt_intersect",
        "hot_defect_recovery",
    ]
    cert.to_csv(results / "certification_temperature_summary.csv", index=False, float_format="%.12g")

    diagnostics_path = results / "mechanism_diagnostics.csv"
    if diagnostics_path.is_file():
        diagnostics = pd.read_csv(diagnostics_path)
        operational = diagnostics[(diagnostics.s >= 0.75) & (diagnostics.s <= 1.25)]
        correlations = pd.DataFrame(
            [
                {
                    "metric": "retained_cooling_fraction",
                    "median_spearman_with_projected_gap": operational.rho_retained_fraction_vs_gap.median(),
                },
                {
                    "metric": "tracking_mismatch",
                    "median_spearman_with_projected_gap": operational.rho_tracking_mismatch_vs_gap.median(),
                },
                {
                    "metric": "forward_kl",
                    "median_spearman_with_projected_gap": operational.rho_forward_kl_vs_gap.median(),
                },
            ]
        )
        correlations.to_csv(
            results / "mechanism_correlation_summary.csv", index=False, float_format="%.12g"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    repository = args.root.resolve()
    study = Path(__file__).resolve().parents[1]
    results = repository / "results" / "temperature_tree_geometry"
    figures = study / "figures"

    derive_compact_tables(results)

    path = pd.read_csv(results / "temperature_path_summary.csv")
    cert = pd.read_csv(results / "certification_temperature_summary.csv")
    correlations = pd.read_csv(results / "mechanism_correlation_summary.csv")
    verdict = json.loads((results / "pt1a_verdict.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots()
    ax.plot(path.s, path.mean_a_qopt, label="Cooling-power optimum")
    ax.plot(path.s, path.mean_a_hot, label="Best hot-optimal tree")
    ax.plot(path.s, path.mean_a_klopt, label="Forward-KL optimum")
    ax.plot(path.s, path.mean_a_best_fixed, label="Best fixed tree")
    ax.plot(path.s, path.mean_a_maxj, label="MAXJ-LEX")
    ax.set_xlabel(r"Normalized inverse temperature $s=\beta/\beta_{\mathrm{cert}}$")
    ax.set_ylabel("Mean retained cooling fraction")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    save_figure(fig, figures / "retained_cooling_fraction")

    fig, ax = plt.subplots()
    ax.plot(path.s, path.median_hot_defect_recovery)
    ax.set_xlabel(r"Normalized inverse temperature $s=\beta/\beta_{\mathrm{cert}}$")
    ax.set_ylabel("Median fraction of hot-tree cooling defect recovered")
    ax.set_ylim(-0.02, 1.02)
    save_figure(fig, figures / "oracle_defect_recovery")

    fig, ax = plt.subplots()
    ax.plot(path.s, path.mean_gap_improvement_over_hot, label="Mean")
    ax.plot(path.s, path.median_gap_improvement_over_hot, label="Median")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel(r"Normalized inverse temperature $s=\beta/\beta_{\mathrm{cert}}$")
    ax.set_ylabel("Projected-gap improvement over hot tree")
    ax.legend()
    save_figure(fig, figures / "projected_energy_consequence")

    fig, ax = plt.subplots()
    ax.plot(path.s, path.fraction_hot_absent_q_near, label="Hot set absent from Q near-optimum")
    ax.plot(path.s, path.fraction_q_kl_disjoint, label="Q and KL optima disjoint")
    ax.set_xlabel(r"Normalized inverse temperature $s=\beta/\beta_{\mathrm{cert}}$")
    ax.set_ylabel("Fraction of ten instances")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    save_figure(fig, figures / "optimal_set_turnover")

    fig, ax = plt.subplots()
    ax.bar(correlations.metric, correlations.median_spearman_with_projected_gap)
    ax.axhline(0.0, linestyle="--")
    ax.set_ylabel("Median Spearman correlation with projected gap")
    ax.tick_params(axis="x", rotation=15)
    save_figure(fig, figures / "metric_correlations_with_projected_gap")

    fig, ax = plt.subplots()
    ax.plot(path.s, path.mean_gap_qopt, label="Cooling-power optimum")
    ax.plot(path.s, path.mean_gap_hot, label="Best hot-optimal tree")
    ax.plot(path.s, path.mean_gap_klopt, label="Forward-KL optimum")
    ax.plot(path.s, path.mean_gap_best_fixed, label="Best fixed tree")
    ax.axhline(0.1, linestyle="--", label="Target Gibbs gap at s=1")
    ax.set_xlabel(r"Normalized inverse temperature $s=\beta/\beta_{\mathrm{cert}}$")
    ax.set_ylabel("Mean projected target-energy gap")
    ax.legend()
    save_figure(fig, figures / "projected_gap_paths")

    gate_table = pd.DataFrame(
        [
            ["A: finite-temperature evolution", verdict["gate_a"]["passing_instances"], 10, verdict["gate_a"]["pass"]],
            ["B: useful cooling headroom", verdict["gate_b"]["passing_instances"], 10, verdict["gate_b"]["pass"]],
            ["C: projected-energy relevance", verdict["gate_c"]["passing_instances"], 10, verdict["gate_c"]["pass"]],
            ["D: distinction from KL optimum", verdict["gate_d"]["passing_instances"], 10, verdict["gate_d"]["pass"]],
        ],
        columns=["Gate", "Passing instances", "Total", "Project pass"],
    )
    cert_table = cert.rename(
        columns={
            "instance": "Instance",
            "a_hot": "Hot A",
            "a_qopt": "Q-opt A",
            "gap_hot": "Hot gap",
            "gap_qopt": "Q-opt gap",
            "gap_klopt": "KL-opt gap",
            "xi_hot": "Hot Xi",
            "xi_qopt": "Q-opt Xi",
            "xi_klopt": "KL-opt Xi",
        }
    )[
        ["Instance", "Hot A", "Q-opt A", "Hot gap", "Q-opt gap", "KL-opt gap", "Hot Xi", "Q-opt Xi", "KL-opt Xi"]
    ]
    correlation_map = dict(zip(correlations.metric, correlations.median_spearman_with_projected_gap))

    report = f"""# Temperature-dependent tree geometry: study report

## Verdict

The study completed successfully over **{verdict['tree_temperature_cells']:,} tree-temperature cells**: all {verdict['total_tree_count']:,} target-supported spanning trees for ten `n=8` instances at 61 normalized temperatures.

```text
{verdict['decision']}
```

The temperature-dependent geometry is real and large, but the proposed cooling-power selector fails the operational gate. It should not be advanced as an operational tree-selection method on the basis of these data.

## Frozen gates

{markdown_table(gate_table, digits=0)}

- Gate A passed on **{verdict['gate_a']['passing_instances']}/10** instances.
- Gate B passed on **{verdict['gate_b']['passing_instances']}/10**; the median maximum recovery of the defect left by the hot tree was **{verdict['gate_b']['median_max_defect_recovery']:.1%}**.
- Gate C passed on **{verdict['gate_c']['passing_instances']}/10**. Its median operational-interval energy improvement was **{verdict['gate_c']['median_mean_gap_improvement']:.4f}**; negative values mean that the cooling-power-optimal tree produced a worse projected state.
- Gate D passed on **{verdict['gate_d']['passing_instances']}/10**.

## Main findings

At `s=1`, the mean retained fraction rises from **{cert.a_hot.mean():.3f}** for the best hot-optimal tree to **{cert.a_qopt.mean():.3f}** for the cooling-power optimum. This geometric improvement does not improve the represented state. The mean projected target-energy gaps are

```text
forward-KL-optimal tree : {cert.gap_klopt.mean():.4f}
best hot-optimal tree   : {cert.gap_hot.mean():.4f}
cooling-power optimum   : {cert.gap_qopt.mean():.4f}
```

The cooling-power optimum is worse than both alternatives on **{int((cert.gap_qopt > cert.gap_hot).sum())}/10** and **{int((cert.gap_qopt > cert.gap_klopt).sum())}/10** instances, respectively.

The post-hoc median within-instance Spearman correlations with projected gap are

```text
retained cooling fraction : {correlation_map['retained_cooling_fraction']:+.3f}
tracking mismatch         : {correlation_map['tracking_mismatch']:+.3f}
forward KL                : {correlation_map['forward_kl']:+.3f}
```

These correlations are descriptive rather than frozen selection tests.

## Conceptual diagnosis

The retained-cooling quantity is evaluated at the exact target state. Except when that state already belongs to the selected tree family, it is not the on-manifold cooling rate of the projected tree model. Large target-state tangent capture therefore need not imply small projection error or accurate self-contained thermal tracking.

## Certification-temperature table

{markdown_table(cert_table, digits=4)}

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
"""
    (study / "report.md").write_text(report, encoding="utf-8")
    print("Regenerated compact temperature-tree summaries, report, and figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
