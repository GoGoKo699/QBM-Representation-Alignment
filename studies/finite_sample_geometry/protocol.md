# Frozen finite-sample protocol

## Scientific question

What is the cheapest geometry-aware update that preserves the exact-natural
rescue when gradients and Fisher information are estimated from Gibbs samples?

## Instances and split

The experiment uses the twenty fixed-$n=16$, certificate-tight instances
from the exact-moment boundary study, with exact widths 3, 4, 5, and 6.

Hyperparameters were fixed after inspecting only:

```text
ct_w3_i1
ct_w4_i1
ct_w5_i1
ct_w6_i1
```

The held-out evaluation set consists of the remaining sixteen instances
(`i2` through `i5` within each width class).

## Parameter initialization

Every broad trajectory starts from

$$
\theta^{(0)}=c+0.3\xi,
$$

using seeds

```text
0, 19, 42, 50, 101
```

and the exact-moment boundary study canonical all-pairs random-coordinate contract.

## Sample budgets

```text
64, 256, 1024, 4096 Gibbs samples per update
```

The samples are exact categorical draws from the enumerated $n=16$ Gibbs
distribution. This is a controlled Monte Carlo experiment, not a hardware or
circuit-shot experiment.

Methods sharing an instance, parameter seed, and sample budget begin from the
same pseudorandom sample stream. Their distributions diverge after the first
update, so later realized configurations are not identical.

## Sample estimators

For one batch,

$$
\widehat g=-\widehat{\mathrm{Cov}}(F,C),
\qquad
\widehat I=\widehat{\mathrm{Cov}}(F,F).
$$

The unbiased $1/(M-1)$ covariance convention is used.

## Optimizers

### Sampled Adam

The repository Adam update with

```text
learning rate 0.02
beta1 0.9
beta2 0.999
epsilon 1e-8
```

using the sampled gradient.

### Sampled diagonal Fisher

$$
d_j=-\frac{\widehat g_j}
{\widehat I_{jj}+10^{-3}\max_k\widehat I_{kk}}.
$$

The Euclidean step norm is capped at $0.5\|c\|_2$.

### Sampled two-block Fisher

Separate covariance blocks are formed for:

1. local-field parameters;
2. pair-coupling parameters.

Within each block,

$$
\widehat I_{\mathrm{reg}}
=0.9\widehat I
+0.1\mathrm{diag}(\widehat I)
+10^{-3}\bar I_{\mathrm{diag}}\,\mathbb I.
$$

The two solves are concatenated and capped at $0.5\|c\|_2$.

### Same-batch full Fisher

The gradient and full Fisher matrix are estimated from the same batch. A tiny
ridge of $10^{-10}$ times the mean covariance diagonal is used for numerical
solving. The step norm is capped at $0.5\|c\|_2$.

### Independent-batch full Fisher

The same total budget is split equally:

- one half estimates the gradient;
- one half independently estimates the Fisher matrix.

This is a control for the exact same-batch cancellation.

### Ray plus residual

Write

$$
\theta=\beta c+u,
\qquad
u\perp c.
$$

Each update:

1. increases $\beta$ by 0.5;
2. retains 95% of the old transverse component;
3. adds a sampled two-block-Fisher residual step, projected orthogonally to
   $c$ and capped at $0.1\|c\|_2$.

### Analytic target cooling

$$
\theta\leftarrow\theta+0.5c.
$$

No moment-estimation samples are used for the update. This is the aligned
oracle/control. The reported zero sample count excludes any practical stopping
or validation measurement.

## Success and monitoring

The protocol permits at most 200 recorded states. Success is

$$
\Delta E\le0.1.
$$

Exact enumeration is used only to evaluate trajectories and identify the first
successful state retrospectively. The sample-to-success values are therefore
scientific evaluation metrics, not implementable stopping costs.

## Trap replay

The two the exact-moment boundary study excited-boundary trajectories are replayed from records
199 and 999. For each state and sample budget, five independent sample streams
are used for:

- sampled Adam;
- same-batch full Fisher;
- ray plus residual.

Analytic target cooling is included once per saved state.
