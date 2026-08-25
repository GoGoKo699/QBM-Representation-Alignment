# Exact Gibbs and q-sample preparation accounting

## Q-sample

For a classical Gibbs distribution $p(x)$, the coherent q-sample is

$$
|q\rangle=\sum_x\sqrt{p(x)}|x\rangle.
$$

It reproduces computational-basis sampling probabilities and diagonal-observable expectations. It is not the mixed Gibbs state.

## Mixed-state purification

The exact diagonal Gibbs state

$$
\rho=\sum_xp(x)|x\rangle\!\langle x|
$$

can be obtained by preparing

$$
|\Psi\rangle=\sum_x\sqrt{p(x)}|x\rangle_A|x\rangle_B
$$

and tracing out register $B$. After q-sample preparation on $A$, one copy CNOT per variable produces this purification.

## Graphical factorization

Variable elimination gives exact conditional probability tables. Reversing an elimination order yields a sequence of multiplexed $R_y$ rotations. If the induced width is $w$, each conditional depends on at most $w$ parent variables.

The repository reports:

- induced width;
- total conditional-angle entries;
- a Gray-code CNOT upper count;
- the elimination order and induced degrees.

These are logical description resources. They are not fault-tolerant gate counts and do not include state-verification, compilation to restricted connectivity, rotation synthesis precision, or error correction.

## Confirmatory representations

For $n=16$:

| Representation | Width | Conditional angles | CNOT upper count |
|---|---:|---:|---:|
| Native chain | 1 | 31 | 30 |
| Random target tree | 1 | 31 | 30 |
| Max-weight target tree | 1 | 31 | 30 |
| Full graph | 3–5 | 75–159 | 74–158 |

The tree models are exact low-width representations of their own Gibbs families. They are approximations to the full target feature span, not approximations in numerical inference within the selected family.
