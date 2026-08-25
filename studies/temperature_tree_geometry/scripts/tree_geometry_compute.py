#!/usr/bin/env python3
"""Per-instance exhaustive computation for the temperature-tree study."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tree_geometry_core import (
    KL_TIE_ABS,
    KL_TIE_REL,
    NEAR_OPT_FRACTION,
    Q_TIE_ABS,
    Q_TIE_REL,
    S_GRID,
    batch_pinv_quadratic,
    enumerate_tree_family,
    exact_target_geometry,
    find_beta_cert,
    gibbs_probability,
    longest_true_run,
    parse_instance,
    tree_projection_batch,
)

def run_instance(
    instance_path: Path,
    root: Path,
    full_csv_path: Path,
    write_header: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], bool]:
    started = time.perf_counter()
    instance = parse_instance(instance_path)
    family = enumerate_tree_family(instance)
    beta_cert = find_beta_cert(instance)
    beta_grid = S_GRID * beta_cert
    beta_count = len(beta_grid)
    tree_count = len(family.trees)
    dimension = 2 * instance.n - 1

    # Advanced indexing creates a compact tree-specific feature tensor once.
    selected_features = np.transpose(
        instance.features[:, family.feature_indices], (1, 0, 2)
    )

    retained = np.empty((beta_count, tree_count), dtype=np.float64)
    full_power = np.empty(beta_count, dtype=np.float64)
    retained_fraction = np.empty_like(retained)
    defect = np.empty_like(retained)
    forward_kl = np.empty_like(retained)
    projected_gap = np.empty_like(retained)
    tracking_mismatch = np.empty_like(retained)
    sigma_lambda_min = np.empty_like(retained)
    sigma_rank = np.empty((beta_count, tree_count), dtype=np.int16)
    sigma_condition = np.empty_like(retained)

    max_projection_normalization_residual = 0.0
    max_projection_moment_residual = 0.0
    min_kl = math.inf
    max_q_excess = -math.inf
    max_retained_fraction_excess = -math.inf

    for beta_index, beta in enumerate(beta_grid):
        probability = gibbs_probability(instance.cost, float(beta))
        q_values, b_tau, variance, target_feature_mean, _full_covariance = exact_target_geometry(
            probability, instance, family
        )
        retained[beta_index] = q_values
        full_power[beta_index] = variance
        retained_fraction[beta_index] = q_values / variance
        defect[beta_index] = variance - q_values

        log_projected, projected, norm_residual = tree_projection_batch(
            probability, instance, family
        )
        max_projection_normalization_residual = max(
            max_projection_normalization_residual, norm_residual
        )
        log_target = np.log(probability)
        forward_kl[beta_index] = float(probability @ log_target) - log_projected @ probability
        projected_energy = projected @ instance.cost
        projected_gap[beta_index] = projected_energy - instance.ground_energy

        feature_mean_sigma = np.einsum(
            "ts,tsi->ti", projected, selected_features, optimize=True
        )
        target_tree_mean = target_feature_mean[family.feature_indices]
        max_projection_moment_residual = max(
            max_projection_moment_residual,
            float(np.max(np.abs(feature_mean_sigma - target_tree_mean))),
        )
        second_sigma = np.einsum(
            "ts,tsi,tsj->tij",
            projected,
            selected_features,
            selected_features,
            optimize=True,
        )
        covariance_sigma = (
            second_sigma
            - feature_mean_sigma[:, :, None] * feature_mean_sigma[:, None, :]
        )
        cost_feature_sigma = np.einsum(
            "ts,tsi,s->ti",
            projected,
            selected_features,
            instance.cost,
            optimize=True,
        )
        b_sigma = cost_feature_sigma - feature_mean_sigma * projected_energy[:, None]
        difference = b_sigma - b_tau
        xi_squared, lambda_min, rank, condition = batch_pinv_quadratic(
            covariance_sigma, difference
        )
        tracking_mismatch[beta_index] = np.sqrt(np.maximum(xi_squared, 0.0))
        sigma_lambda_min[beta_index] = lambda_min
        sigma_rank[beta_index] = rank
        sigma_condition[beta_index] = condition

        min_kl = min(min_kl, float(np.min(forward_kl[beta_index])))
        max_q_excess = max(max_q_excess, float(np.max(q_values - variance)))
        max_retained_fraction_excess = max(
            max_retained_fraction_excess,
            float(np.max(retained_fraction[beta_index] - 1.0)),
        )

        if beta_index % 10 == 0 or beta_index == beta_count - 1:
            print(
                f"[{instance.name}] beta {beta_index + 1:02d}/{beta_count}: "
                f"s={S_GRID[beta_index]:.3f}",
                flush=True,
            )

    # Exact beta=0 identity: Pauli monomials are orthonormal.
    beta0_identity_residual = float(
        np.max(np.abs(retained[0] - family.hot_total_score))
    )

    # Best fixed tree in hindsight, frozen as mean retained fraction over the grid.
    mean_retained_fraction = np.mean(retained_fraction, axis=0)
    best_fixed_max = float(np.max(mean_retained_fraction))
    best_fixed_candidates = np.flatnonzero(
        mean_retained_fraction >= best_fixed_max - Q_TIE_ABS
    )
    best_fixed_index = int(best_fixed_candidates[0])

    optima_rows: list[dict[str, object]] = []
    for beta_index, beta in enumerate(beta_grid):
        q = retained[beta_index]
        a = retained_fraction[beta_index]
        kl = forward_kl[beta_index]
        gap = projected_gap[beta_index]
        xi = tracking_mismatch[beta_index]
        variance = full_power[beta_index]

        q_max = float(np.max(q))
        q_tolerance = max(Q_TIE_ABS, Q_TIE_REL * variance)
        q_exact = q >= q_max - q_tolerance
        a_max = float(np.max(a))
        q_near = a >= a_max - NEAR_OPT_FRACTION

        kl_minimum = float(np.min(kl))
        kl_tolerance = max(KL_TIE_ABS, KL_TIE_REL * max(1.0, abs(kl_minimum)))
        kl_exact = kl <= kl_minimum + kl_tolerance

        hot = family.hot_mask
        maxj = family.maxj_lex_index
        hot_best_q = float(np.max(q[hot]))
        hot_best_gap = float(np.min(gap[hot]))
        qopt_best_gap = float(np.min(gap[q_exact]))
        klopt_best_q = float(np.max(q[kl_exact]))
        klopt_best_gap = float(np.min(gap[kl_exact]))
        qopt_min_kl = float(np.min(kl[q_exact]))

        q_rep = int(np.flatnonzero(q_exact)[0])
        kl_rep = int(np.flatnonzero(kl_exact)[0])
        hot_q_candidates = np.flatnonzero(hot & (q >= hot_best_q - q_tolerance))
        hot_q_rep = int(hot_q_candidates[0])

        sets_intersect = bool(np.any(q_exact & kl_exact))
        hot_in_near = bool(np.any(hot & q_near))
        if bool(np.any(q_exact & hot)):
            max_hot_q_overlap = instance.n - 1
        else:
            max_hot_q_overlap = int(
                np.max(family.incidence[q_exact] @ family.incidence[hot].T)
            )
        min_swaps = (instance.n - 1) - max_hot_q_overlap

        hot_edge_frequency = np.mean(family.incidence[hot], axis=0)
        q_near_edge_frequency = np.mean(family.incidence[q_near], axis=0)
        edge_frequency_tv = float(
            np.sum(np.abs(hot_edge_frequency - q_near_edge_frequency))
            / (2.0 * (instance.n - 1))
        )

        power_advantage = q_max - hot_best_q
        remaining_hot_defect = variance - hot_best_q
        if remaining_hot_defect > 1.0e-15:
            defect_recovery = power_advantage / remaining_hot_defect
        else:
            defect_recovery = 0.0

        optima_rows.append(
            {
                "instance": instance.name,
                "beta_index": beta_index,
                "s": float(S_GRID[beta_index]),
                "beta": float(beta),
                "beta_cert": beta_cert,
                "q_full": variance,
                "q_max": q_max,
                "a_max": a_max,
                "q_exact_count": int(np.sum(q_exact)),
                "q_near_count": int(np.sum(q_near)),
                "kl_min": kl_minimum,
                "kl_exact_count": int(np.sum(kl_exact)),
                "hot_count": int(np.sum(hot)),
                "qopt_tree_id": q_rep,
                "qopt_tree_hash": family.tree_hashes[q_rep],
                "klopt_tree_id": kl_rep,
                "klopt_tree_hash": family.tree_hashes[kl_rep],
                "maxj_tree_id": maxj,
                "maxj_tree_hash": family.tree_hashes[maxj],
                "hot_best_q_tree_id": hot_q_rep,
                "hot_best_q_tree_hash": family.tree_hashes[hot_q_rep],
                "best_fixed_tree_id": best_fixed_index,
                "best_fixed_tree_hash": family.tree_hashes[best_fixed_index],
                "q_maxj": float(q[maxj]),
                "a_maxj": float(a[maxj]),
                "gap_maxj": float(gap[maxj]),
                "kl_maxj": float(kl[maxj]),
                "xi_maxj": float(xi[maxj]),
                "q_hot_best": hot_best_q,
                "a_hot_best": hot_best_q / variance,
                "gap_hot_best": hot_best_gap,
                "q_best_fixed": float(q[best_fixed_index]),
                "a_best_fixed": float(a[best_fixed_index]),
                "gap_best_fixed": float(gap[best_fixed_index]),
                "q_klopt_best": klopt_best_q,
                "a_klopt_best": klopt_best_q / variance,
                "gap_klopt_best": klopt_best_gap,
                "gap_qopt_best": qopt_best_gap,
                "kl_qopt_best": qopt_min_kl,
                "q_advantage_over_hot": power_advantage,
                "q_advantage_fraction_full": power_advantage / variance,
                "hot_defect_recovery": defect_recovery,
                "gap_improvement_over_hot": hot_best_gap - qopt_best_gap,
                "q_loss_klopt_fraction_full": (q_max - klopt_best_q) / variance,
                "kl_loss_qopt": qopt_min_kl - kl_minimum,
                "qopt_klopt_intersect": sets_intersect,
                "hot_in_q_near_set": hot_in_near,
                "min_edge_swaps_hot_to_qopt": min_swaps,
                "edge_frequency_tv_hot_vs_qnear": edge_frequency_tv,
                "xi_qopt_min": float(np.min(xi[q_exact])),
                "xi_qopt_median": float(np.median(xi[q_exact])),
                "xi_hot_min": float(np.min(xi[hot])),
                "xi_hot_median": float(np.median(xi[hot])),
                "xi_klopt_min": float(np.min(xi[kl_exact])),
                "xi_klopt_median": float(np.median(xi[kl_exact])),
                "lambda_sigma_qopt_min": float(np.min(sigma_lambda_min[beta_index, q_exact])),
                "lambda_sigma_hot_min": float(np.min(sigma_lambda_min[beta_index, hot])),
            }
        )

    optima = pd.DataFrame(optima_rows)

    gate_a_mask = (optima["beta_index"].to_numpy() > 0) & (
        ~optima["hot_in_q_near_set"].to_numpy(dtype=bool)
    )
    gate_a_longest_run = longest_true_run(gate_a_mask)
    gate_a_instance = gate_a_longest_run >= 4

    gate_b_interval = (optima["s"] >= 0.25) & (optima["s"] <= 1.25)
    max_defect_recovery = float(
        optima.loc[gate_b_interval, "hot_defect_recovery"].max()
    )
    mean_power_advantage_fraction = float(
        optima.loc[gate_b_interval, "q_advantage_fraction_full"].mean()
    )
    gate_b_instance = (
        max_defect_recovery >= 0.10
        and mean_power_advantage_fraction >= 0.005
    )

    gate_c_interval = (optima["s"] >= 0.75) & (optima["s"] <= 1.25)
    mean_gap_improvement = float(
        optima.loc[gate_c_interval, "gap_improvement_over_hot"].mean()
    )
    gate_c_instance = mean_gap_improvement >= 0.01

    gate_d_mask = (
        (optima["beta_index"].to_numpy() > 0)
        & (~optima["qopt_klopt_intersect"].to_numpy(dtype=bool))
        & (optima["q_loss_klopt_fraction_full"].to_numpy() >= 0.005)
    )
    gate_d_longest_run = longest_true_run(gate_d_mask)
    gate_d_instance = gate_d_longest_run >= 4

    ground_count = int(np.sum(instance.cost == instance.ground_energy))
    planted_index = int(
        np.flatnonzero(np.all(instance.bits == np.asarray(instance.planted)[None, :], axis=1))[0]
    )
    planted_is_ground = bool(instance.cost[planted_index] == instance.ground_energy)
    beta_cert_residual = float(
        gibbs_probability(instance.cost, beta_cert) @ instance.cost
        - instance.ground_energy
        - 0.1
    )

    validation = {
        "instance": instance.name,
        "n": instance.n,
        "m": instance.m,
        "target_edge_count": len(instance.edges),
        "tree_count": tree_count,
        "ground_energy": instance.ground_energy,
        "ground_count": ground_count,
        "planted_is_ground": planted_is_ground,
        "spectral_gap": instance.spectral_gap,
        "beta_cert": beta_cert,
        "beta_cert_residual": beta_cert_residual,
        "hamiltonian_max_residual": float(
            np.max(np.abs(instance.features @ instance.coefficients - instance.cost))
        ),
        "beta0_hot_identity_residual": beta0_identity_residual,
        "max_projection_normalization_residual": max_projection_normalization_residual,
        "max_projection_moment_residual": max_projection_moment_residual,
        "minimum_forward_kl": min_kl,
        "maximum_q_minus_full_variance": max_q_excess,
        "maximum_retained_fraction_minus_one": max_retained_fraction_excess,
        "minimum_sigma_lambda_plus": float(np.nanmin(sigma_lambda_min)),
        "minimum_sigma_rank": int(np.min(sigma_rank)),
        "maximum_sigma_condition": float(np.nanmax(sigma_condition)),
        "hot_optimal_count": int(np.sum(family.hot_mask)),
        "maxj_is_hot_optimal": bool(family.hot_mask[family.maxj_lex_index]),
        "best_fixed_tree_id": best_fixed_index,
        "best_fixed_tree_hash": family.tree_hashes[best_fixed_index],
        "gate_a_longest_run": gate_a_longest_run,
        "gate_a_instance_pass": gate_a_instance,
        "max_defect_recovery": max_defect_recovery,
        "mean_power_advantage_fraction": mean_power_advantage_fraction,
        "gate_b_instance_pass": gate_b_instance,
        "mean_gap_improvement": mean_gap_improvement,
        "gate_c_instance_pass": gate_c_instance,
        "gate_d_longest_run": gate_d_longest_run,
        "gate_d_instance_pass": gate_d_instance,
        "runtime_seconds": time.perf_counter() - started,
    }

    # Tree catalog rows.
    tree_catalog = pd.DataFrame(
        {
            "instance": instance.name,
            "tree_id": np.arange(tree_count, dtype=int),
            "tree_hash": family.tree_hashes,
            "edges": family.edge_strings,
            "edge_indices": [
                ";".join(map(str, row.tolist())) for row in family.edge_indices
            ],
            "hot_total_score": family.hot_total_score,
            "is_hot_optimal": family.hot_mask,
            "is_maxj_lex": np.arange(tree_count) == family.maxj_lex_index,
            "mean_retained_fraction": mean_retained_fraction,
            "is_best_fixed": np.arange(tree_count) == best_fixed_index,
        }
    )

    npz_path = root / "results" / "temperature_tree_geometry" / "atlas" / f"{instance.name}_temperature_atlas.npz"
    np.savez_compressed(
        npz_path,
        s=S_GRID,
        beta=beta_grid,
        beta_cert=np.asarray(beta_cert),
        tree_id=np.arange(tree_count, dtype=np.int32),
        tree_hash=np.asarray(family.tree_hashes),
        edge_indices=family.edge_indices,
        hot_mask=family.hot_mask,
        maxj_lex_index=np.asarray(family.maxj_lex_index),
        best_fixed_index=np.asarray(best_fixed_index),
        q_retained=retained,
        q_full=full_power,
        retained_fraction=retained_fraction,
        cooling_defect=defect,
        forward_kl=forward_kl,
        projected_gap=projected_gap,
        tracking_mismatch=tracking_mismatch,
        sigma_lambda_min_plus=sigma_lambda_min,
        sigma_rank=sigma_rank,
        sigma_condition=sigma_condition,
    )

    # Canonical long-form CSV.GZ.  Static tree metadata live in tree_catalog.csv.gz.
    rows = beta_count * tree_count
    long_frame = pd.DataFrame(
        {
            "instance": np.repeat(instance.name, rows),
            "beta_index": np.repeat(np.arange(beta_count, dtype=np.int16), tree_count),
            "s": np.repeat(S_GRID, tree_count),
            "beta": np.repeat(beta_grid, tree_count),
            "tree_id": np.tile(np.arange(tree_count, dtype=np.int32), beta_count),
            "q_retained": retained.reshape(-1),
            "q_full": np.repeat(full_power, tree_count),
            "retained_fraction": retained_fraction.reshape(-1),
            "cooling_defect": defect.reshape(-1),
            "forward_kl": forward_kl.reshape(-1),
            "projected_gap": projected_gap.reshape(-1),
            "tracking_mismatch": tracking_mismatch.reshape(-1),
            "sigma_lambda_min_plus": sigma_lambda_min.reshape(-1),
            "sigma_rank": sigma_rank.reshape(-1),
            "sigma_condition": sigma_condition.reshape(-1),
        }
    )
    long_frame.to_csv(
        full_csv_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        compression={"method": "gzip", "compresslevel": 6},
        float_format="%.12g",
    )

    print(
        f"[{instance.name}] completed: {tree_count} trees, "
        f"{time.perf_counter() - started:.1f} s",
        flush=True,
    )
    return optima, tree_catalog, validation, False
