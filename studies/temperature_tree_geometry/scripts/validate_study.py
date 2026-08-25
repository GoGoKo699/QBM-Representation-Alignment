#!/usr/bin/env python3
"""Validate the compact temperature-dependent tree-geometry archive."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS = REPOSITORY / "results" / "temperature_tree_geometry"
EXPECTED_INSTANCES = {f"n8i{index}" for index in range(1, 11)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    require(not missing, f"{label}: missing columns {sorted(missing)}")


def main() -> int:
    verdict = json.loads((RESULTS / "pt1a_verdict.json").read_text(encoding="utf-8"))
    detailed = json.loads((RESULTS / "detailed_validation.json").read_text(encoding="utf-8"))
    independent = json.loads((RESULTS / "independent_validation.json").read_text(encoding="utf-8"))
    mechanism = json.loads((RESULTS / "mechanism_summary.json").read_text(encoding="utf-8"))
    record = json.loads((RESULTS / "validation.json").read_text(encoding="utf-8"))

    require(bool(verdict["validation_pass"]), "reference validation did not pass")
    require(bool(detailed["overall_pass"]), "detailed validation did not pass")
    require(bool(independent["overall_pass"]), "independent validation did not pass")
    require(record.get("status") == "PASS", "repository validation record is not PASS")
    require(
        verdict["decision"] == "GEOMETRY_ONLY_DO_NOT_USE_COOLING_POWER_AS_OPERATIONAL_SELECTOR",
        "unexpected scientific decision",
    )
    require(record.get("scientific_outcome") == verdict["decision"], "validation outcome mismatch")
    require(int(verdict["instance_count"]) == 10, "instance count mismatch")
    require(int(verdict["total_tree_count"]) == 20_812, "tree count mismatch")
    require(int(verdict["temperature_points_per_instance"]) == 61, "temperature count mismatch")
    require(int(verdict["tree_temperature_cells"]) == 1_269_532, "reference cell count mismatch")
    require(verdict["gate_a"]["passing_instances"] == 10, "gate A mismatch")
    require(verdict["gate_b"]["passing_instances"] == 10, "gate B mismatch")
    require(verdict["gate_c"]["passing_instances"] == 0, "gate C mismatch")
    require(verdict["gate_d"]["passing_instances"] == 10, "gate D mismatch")
    require(bool(verdict["gate_a"]["pass"]), "gate A should pass")
    require(bool(verdict["gate_b"]["pass"]), "gate B should pass")
    require(not bool(verdict["gate_c"]["pass"]), "gate C should fail")
    require(bool(verdict["gate_d"]["pass"]), "gate D should pass")

    instances = sorted((STUDY / "instances").glob("n8i*.txt"), key=lambda path: int(path.stem[3:]))
    require(len(instances) == 10, "expected ten input instances")
    require({path.stem for path in instances} == EXPECTED_INSTANCES, "instance filenames mismatch")
    require(all(path.stat().st_size > 0 for path in instances), "empty instance file")

    gates = pd.read_csv(RESULTS / "instance_gate_summary.csv")
    require_columns(
        gates,
        {
            "instance",
            "tree_count",
            "gate_a_instance_pass",
            "gate_b_instance_pass",
            "gate_c_instance_pass",
            "gate_d_instance_pass",
            "maxj_is_hot_optimal",
        },
        "instance gate summary",
    )
    require(len(gates) == 10, "instance gate summary row count mismatch")
    require(set(gates.instance) == EXPECTED_INSTANCES, "instance gate labels mismatch")
    require(int(gates.tree_count.sum()) == 20_812, "summed tree count mismatch")
    require(gates.gate_a_instance_pass.astype(bool).all(), "gate A instance mismatch")
    require(gates.gate_b_instance_pass.astype(bool).all(), "gate B instance mismatch")
    require((~gates.gate_c_instance_pass.astype(bool)).all(), "gate C instance mismatch")
    require(gates.gate_d_instance_pass.astype(bool).all(), "gate D instance mismatch")
    require(gates.maxj_is_hot_optimal.astype(bool).all(), "MAXJ hot-optimal identity mismatch")

    path = pd.read_csv(RESULTS / "temperature_path_summary.csv")
    require_columns(
        path,
        {
            "s",
            "mean_a_qopt",
            "mean_a_hot",
            "mean_a_klopt",
            "mean_gap_qopt",
            "mean_gap_hot",
            "mean_gap_klopt",
        },
        "temperature path summary",
    )
    expected_s = np.round(np.arange(0.0, 1.5000001, 0.025), 12)
    require(len(path) == 61, "temperature path summary row count mismatch")
    require(np.allclose(path.s.to_numpy(), expected_s, atol=1e-12, rtol=0), "temperature grid mismatch")
    require(np.all(np.isfinite(path.select_dtypes(include=[np.number]).to_numpy())), "nonfinite path value")

    cert = pd.read_csv(RESULTS / "certification_temperature_summary.csv")
    require_columns(
        cert,
        {
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
        },
        "certification-temperature summary",
    )
    require(len(cert) == 10, "certification-temperature row count mismatch")
    require(set(cert.instance) == EXPECTED_INSTANCES, "certification labels mismatch")
    require((cert.gap_qopt > cert.gap_hot).all(), "Q optimum is not worse than hot tree on every instance")
    require((cert.gap_qopt > cert.gap_klopt).all(), "Q optimum is not worse than KL tree on every instance")
    require((~cert.qopt_klopt_intersect.astype(bool)).all(), "Q/KL optima are not disjoint at certification")
    require((cert.a_qopt >= cert.a_hot - 1e-12).all(), "cooling-power optimum underperforms hot tree geometrically")

    correlations = pd.read_csv(RESULTS / "mechanism_correlation_summary.csv")
    require_columns(
        correlations,
        {"metric", "median_spearman_with_projected_gap"},
        "mechanism correlation summary",
    )
    require(len(correlations) == 3, "mechanism correlation row count mismatch")
    expected_metrics = {"retained_cooling_fraction", "tracking_mismatch", "forward_kl"}
    require(set(correlations.metric) == expected_metrics, "mechanism metric set mismatch")
    correlation_map = correlations.set_index("metric").median_spearman_with_projected_gap.to_dict()

    require(abs(float(cert.gap_hot.mean()) - float(mechanism["s1_mean_gap_hot"])) < 1e-10, "hot gap summary mismatch")
    require(abs(float(cert.gap_qopt.mean()) - float(mechanism["s1_mean_gap_qopt"])) < 1e-10, "Q-opt gap summary mismatch")
    require(abs(float(cert.gap_klopt.mean()) - float(mechanism["s1_mean_gap_klopt"])) < 1e-10, "KL-opt gap summary mismatch")
    require(
        abs(correlation_map["retained_cooling_fraction"] - float(mechanism["median_spearman_retained_fraction_vs_gap"])) < 1e-10,
        "retained-fraction correlation mismatch",
    )
    require(
        abs(correlation_map["tracking_mismatch"] - float(mechanism["median_spearman_tracking_mismatch_vs_gap"])) < 1e-10,
        "tracking-mismatch correlation mismatch",
    )
    require(
        abs(correlation_map["forward_kl"] - float(mechanism["median_spearman_forward_kl_vs_gap"])) < 1e-10,
        "forward-KL correlation mismatch",
    )
    require(int(mechanism["s1_qopt_worse_than_hot_count"]) == 10, "Q-opt/hot count mismatch")
    require(int(mechanism["s1_qopt_worse_than_kl_count"]) == 10, "Q-opt/KL count mismatch")

    required_docs = ["README.md", "protocol.md", "report.md", "provenance.md"]
    for name in required_docs:
        document = STUDY / name
        require(document.is_file() and document.stat().st_size > 300, f"missing document {name}")

    require(not (RESULTS / "temperature_tree_geometry.csv.gz").exists(), "large atlas should not be packaged")
    require(not (RESULTS / "tree_catalog.csv.gz").exists(), "tree catalogue should not be packaged")
    require(not (RESULTS / "atlas").exists(), "per-instance atlas directory should not be packaged")

    print("Temperature-dependent tree-geometry validation passed.")
    print("  instances: 10")
    print("  spanning trees in reference run: 20,812")
    print("  compact temperature rows: 61")
    print("  operational selector gate: failed as recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
