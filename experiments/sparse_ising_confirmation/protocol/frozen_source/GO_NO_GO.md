# Publication go/no-go criteria

## Purpose

The project has accumulated enough development evidence that further flexibility would make confirmation meaningless. This document specifies what outcomes justify a manuscript, what outcomes require narrowing, and what outcomes stop the current claim.

## Full-go decision

Proceed to a full manuscript built around representation alignment and boundary geometry only if the preregistered sparse-Ising study satisfies all four conditions.

### G1 — Problem-tree effect against the generic chain

For target-biased Adam,

\[
\widehat p_{\mathrm{problem\ tree}}
-
\widehat p_{\mathrm{chain}}
\ge 0.15,
\]

and the Holm-adjusted instance-cluster 95% interval excludes zero.

### G2 — Problem-tree effect against a random target-supported tree

For target-biased Adam,

\[
\widehat p_{\mathrm{problem\ tree}}
-
\widehat p_{\mathrm{random\ tree}}
\ge 0.10,
\]

and the Holm-adjusted interval excludes zero.

This is necessary to claim that selecting important interactions matters beyond merely using any target-supported tree.

### G3 — Exact-geometry representation separation

For target-biased exact natural gradient,

\[
\widehat p_{\mathrm{problem\ tree}}
-
\widehat p_{\mathrm{chain}}
\ge 0.25,
\]

and the adjusted interval excludes zero.

This is necessary to separate representation mismatch from Adam-specific behavior.

### G4 — Constructive preparation distinction

The problem tree remains exact width one with a materially smaller conditional-table and logical-gate description than the full graph. Resource accounting must use the same compiler definitions as checkpoints 2–3.

## Strong-go additions

The paper becomes stronger, but these are not mandatory:

1. at least two fresh full-graph Adam failures satisfy the frozen excited-boundary trap definition;
2. every qualifying trap is rescued by exact natural geometry or target-ray restoration and not by restarted Adam;
3. the target-distance debt predicts replay time within a prespecified tolerance;
4. the problem tree approaches full-graph exact-natural success while retaining width-one preparation.

## Narrow-paper decision

Proceed with a narrower manuscript if:

- G1 passes;
- the direction of G2 or G3 is positive but its interval includes zero;
- no result reverses direction;
- the existing causal trap mechanism remains internally valid.

The narrowed claim would be:

> Problem-supported sparse representations can improve QBM trainability over a generic chain without increasing treewidth, while aligned QBM failures can arise through excited-boundary optimizer drift.

It would not claim that maximum-weight tree selection is broadly optimal.

## Mechanism-only paper

If the fresh representation effects fail but the boundary mechanism is reproduced on fresh instances, retain only a causal optimizer paper:

> Coordinatewise adaptive optimization can create excited-boundary concentration in aligned commuting Gibbs models, despite the absence of finite stationary points; restoring the information-geometric cooling manifold repairs the failure.

This paper would omit the problem-tree design claim.

## No-go / reframe decision

Do not submit the proposed paper if any of the following occurs:

1. the problem tree performs no better than the chain and random-tree controls;
2. exact natural gradient fails to separate the representations;
3. effect directions reverse on the fresh ensemble;
4. the same-state trap replays cannot be reproduced from stored parameters;
5. the preparation-resource comparison depends on incompatible state targets or compiler conventions;
6. results require post-hoc graph heuristics or optimizer retuning;
7. the confirmatory generator produces a pathological coefficient-scale or spectral-gap distribution that invalidates the frozen optimizer protocol.

In a no-go outcome, checkpoints 1–7 remain useful internal research and software, but the broad publication claim is abandoned.

## Evidence grading

| Grade | Meaning |
|---|---|
| A | Exact theorem or algebraic identity with validated implementation |
| B | Same-state causal intervention with frozen definitions |
| C | Paired held-out empirical effect on one family |
| D | Development-family or exploratory result |
| E | Qualitative observation or hypothesis |

A full-go manuscript requires:

- all main theoretical claims at grade A;
- boundary mechanism at grade B;
- representation claim at grade C on two independent problem families;
- no main-text claim supported only at grade D or E.

## Freeze after confirmation

After confirmatory results are produced:

- no new optimizer enters the main comparison;
- no new graph heuristic enters the main comparison;
- no instance is removed except for a preregistered generation/validation failure;
- no success threshold changes;
- no claim is strengthened because an appendix result looks favorable.
