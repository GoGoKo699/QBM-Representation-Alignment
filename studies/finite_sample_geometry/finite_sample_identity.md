# Same-batch Fisher identity in an aligned commuting Gibbs family

## Setting

Let

$$
p_\theta(z)=\frac{e^{-\theta^{\mathsf{T}}F(z)}}{Z(\theta)}
$$

be a finite commuting Gibbs family, and suppose the target cost belongs exactly
to the same feature span,

$$
C(z)=c_0+c^{\mathsf{T}}F(z).
$$

Draw a batch $z^{(1)},\ldots,z^{(M)}$ from $p_\theta$. Define

$$
\bar F=\frac1M\sum_s F(z^{(s)}),
\qquad
\bar C=\frac1M\sum_s C(z^{(s)}),
$$

and the unbiased sample estimators

$$
\widehat I
=
\frac1{M-1}\sum_s
\bigl(F_s-\bar F\bigr)
\bigl(F_s-\bar F\bigr)^{\mathsf{T}},
$$

$$
\widehat g
=
-\frac1{M-1}\sum_s
\bigl(F_s-\bar F\bigr)
\bigl(C_s-\bar C\bigr).
$$

## Exact finite-sample identity

Because

$$
C_s-\bar C
=
c^{\mathsf{T}}(F_s-\bar F),
$$

one has for every batch, not merely in expectation,

$$
\boxed{\widehat g=-\widehat I c.}
$$

Consequently the undamped Moore--Penrose full-Fisher direction is

$$
-\widehat I^{+}\widehat g
=
\widehat I^{+}\widehat I c
=
P_{\mathrm{range}(\widehat I)}c.
$$

If $\widehat I$ has full rank, the sampled direction is exactly $c$, even
for a finite batch. Sampling noise cancels algebraically because the same
observations are used to estimate both moments.

If $\widehat I$ is rank deficient, the method recovers only the projection of
$c$ onto the sampled covariance subspace.

## Independent estimates

If the gradient and Fisher matrix are estimated from independent batches,

$$
\widehat g_A=-\widehat I_Ac,
\qquad
\widehat I_B\ne\widehat I_A,
$$

and therefore

$$
-\widehat I_B^{+}\widehat g_A
=
\widehat I_B^{+}\widehat I_Ac,
$$

which need not point toward $c$. The independent-batch control splits a
fixed total sample budget equally between the two estimators. Its performance
is much worse at small and intermediate budgets.

## Boundary limitation

At an excited-state boundary, the Gibbs distribution is concentrated on very
few configurations. Increasing the nominal batch size then produces many
repeated observations rather than new linearly independent feature vectors.
The empirical covariance remains low rank.

This is exactly what occurs in the saved excited-boundary states. At record 999, even
4,096 samples give a mean empirical covariance rank of only 3.4 in models with
roughly fifty active parameters. Same-batch full Fisher therefore cannot
recover the missing target directions and fails every replay.

## Scope

The identity requires:

1. a commuting/classical feature representation;
2. exact target alignment $C=c_0+c^{\mathsf{T}}F$;
3. the same samples in both covariance estimates.

It does not automatically extend to partially aligned ansätze, independently
estimated moments, or general noncommuting quantum Boltzmann machines.
