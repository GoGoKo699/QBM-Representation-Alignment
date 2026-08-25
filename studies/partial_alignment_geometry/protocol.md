# Protocol: finite-sample geometry under partial alignment

## Scientific question

When the target cost is not fully contained in the ansatz feature span, can a graph-local covariance approximation retain most of the trainability of a regularized full-Fisher update at substantially lower storage cost?

The fully aligned problem graph is not rerun in this study. Results from the aligned finite-sample study are retained as a reference control.

## Instances and split

The experiment uses the same twenty fixed-$n=16$, certificate-tight instances with exact full widths 3, 4, 5, and 6.

Calibration labels:

```text
ct_w3_i1
ct_w4_i1
ct_w5_i1
ct_w6_i1
```

Held-out evaluation labels are `i2` through `i5` in each width class.

No graph or optimizer parameter uses optimization outcomes from the held-out instances.

## Ansatz graphs

For every instance, five graph representations are defined from the same target Hamiltonian.

### Native chain

```text
(0,1), (1,2), ..., (14,15)
```

Target coefficients are zero on chain edges absent from the problem graph.

### Maximum-weight problem spanning tree

A maximum spanning tree of the problem graph, weighted by $|J_{ij}|$. This graph has exact width 1.

### Greedy width-2 problem subgraph

Start from the problem spanning tree. Consider omitted problem edges in descending $|J_{ij}|$ order, with deterministic random tie-breaking over 64 restarts. Retain an edge only when a stored min-fill elimination order certifies induced width at most 2. Choose the restart with maximum retained squared target-coupling weight.

### Greedy width-3 problem subgraph

The same construction with a width budget of 3.

### Full problem graph

Used only as the aligned reference from the aligned finite-sample study.

The width-2 and width-3 procedures are constructive heuristics, not proofs that the selected subgraphs maximize retained target weight under their width budgets.

## Partial-alignment identity

Write the full cost as

$$
C(z)=c_G^{\mathsf{T}}F_G(z)+R_G(z)+c_0,
$$

where $F_G$ contains the ansatz features and $R_G$ contains omitted interactions. Then

$$
\nabla E
=-I_G c_G-r_G,
\qquad
r_G=\mathrm{Cov}(F_G,R_G).
$$

The fully aligned identity is recovered only when $r_G=0$.

The experiment records:

- retained target coefficient norm;
- retained pair-coupling norm;
- exact cost-variance fraction explained by the retained features;
- the residual $\|\nabla E+I_Gc_G\|/\|\nabla E\|$;
- cosine between the exact natural direction and $c_G$.

## Initial parameters

Every sampled trajectory starts from

$$
\theta^{(0)}=c_G+0.3\xi,
$$

with seeds

```text
0, 19, 42, 50, 101
```

A single Gaussian vector is generated in the canonical all-pairs coordinate system and restricted to each ansatz. This matches local-field and common-edge noise across graphs.

## Sample access

Gibbs samples are exact categorical draws from the enumerated $n=16$ distribution. This is a controlled Monte Carlo experiment rather than a circuit-shot experiment.

All sampled covariances use the unbiased $1/(M-1)$ convention.

## Main sample budget

The held-out main comparison uses

```text
256 samples per update
```

for all four partial graphs and all methods.

## Sample-scaling supplement

For the width-3 graph only, the following budgets are also used:

```text
64 and 1024 samples per update
```

for sampled Adam, graph-star Fisher, regularized full Fisher, and protected ray-plus-residual.

The full aligned reference at 64, 256, and 1024 samples is imported from the aligned finite-sample study without alteration.

## Optimizers

All methods are capped at 200 recorded states. Exact enumeration is used only for retrospective evaluation.

### Sampled Adam

Repository hyperparameters:

```text
learning rate 0.02
beta1 0.9
beta2 0.999
epsilon 1e-8
```

### Sampled diagonal Fisher

$$
d_j=-\frac{\widehat g_j}
{\widehat I_{jj}+10^{-3}\max_k\widehat I_{kk}}.
$$

Step norm cap: $0.5\|c_G\|_2$.

### Sampled two-block Fisher

One covariance block for fields and one for pair couplings. Each block uses

$$
0.9\widehat I+0.1\mathrm{diag}(\widehat I)
+10^{-3}\bar I_{\mathrm{diag}}\,\mathbb I.
$$

Step norm cap: $0.5\|c_G\|_2$.

### Graph-star Fisher

For every variable $v$, form an overlapping block containing:

- the field feature $Z_v$;
- all pair features $Z_vZ_u$ incident to $v$ in the ansatz graph.

Solve the same regularized local covariance system in every star. Sum local directions and divide each coordinate by the number of blocks containing it.

Step norm cap: $0.5\|c_G\|_2$.

This is an additive overlapping local preconditioner introduced for this project. It is not claimed to be a standard QBM natural-gradient algorithm.

### Regularized full Fisher

Use the same sample batch for gradient and covariance, but there is no exact same-batch cancellation under partial alignment. Solve

$$
\left[
0.95\widehat I+0.05\mathrm{diag}(\widehat I)
+10^{-3}\bar I_{\mathrm{diag}}\,\mathbb I
\right]d=-\widehat g.
$$

Step norm cap: $0.5\|c_G\|_2$.

### Protected ray plus graph-star residual

Decompose

$$
\theta=\beta c_G+u,
\qquad u\perp c_G.
$$

For one sample batch:

1. estimate the scalar natural step along $c_G$,
   $$
   \Delta\beta
   =-\frac{\widehat g^{\mathsf{T}}c_G}
   {c_G^{\mathsf{T}}\widehat I c_G+10^{-3}\overline I_{\mathrm{diag}}\|c_G\|^2};
   $$
2. clip $\Delta\beta$ to $[-0.5,0.5]$;
3. retain 95% of the old transverse component;
4. add a graph-star residual update projected orthogonally to $c_G$ and capped at $0.1\|c_G\|_2$.

Unlike the method used in the fully aligned finite-sample study, the scalar step is not forced positive because the projected target direction need not be a descent direction.

## Success and diagnostics

Primary success:

$$
\Delta E\le0.1
$$

within 200 recorded states.

For every trajectory, record:

- first success and minimum gap;
- sample count;
- final planted and dominant-state probabilities;
- exact final Fisher condition and rank when the trajectory is a trap candidate;
- first-update direction cosine with the exact natural direction;
- sample covariance rank;
- sample partial-alignment residual.

## Primary comparisons

At 256 samples per update on held-out instances:

1. graph-star Fisher versus regularized full Fisher;
2. graph-star Fisher versus two-block Fisher;
3. protected ray-plus-residual versus sampled Adam;
4. performance as a function of graph representation and explained cost variance.

Paired effects use the same instance, parameter seed, graph, and sample stream. Uncertainty is obtained by resampling instances and then retaining all seeds within each selected instance.
