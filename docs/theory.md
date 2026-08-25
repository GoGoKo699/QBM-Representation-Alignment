# Gibbs-family geometry

## Commuting representation

Let a representation graph select sufficient statistics

\[
F_G(z)=(F_1(z),\ldots,F_d(z))
\]

and define

\[
p_\theta(z)=\frac{\exp[-\theta^{\mathsf T}F_G(z)]}{Z(\theta)}.
\]

For target cost `C`, the expected objective is

\[
E(\theta)=\mathbb E_{p_\theta}[C].
\]

Differentiating the finite exponential family gives

\[
\frac{\partial E}{\partial\theta_j}
=-\operatorname{Cov}_{p_\theta}(F_j,C).
\]

## Full alignment

If

\[
C(z)=c_0+c^{\mathsf T}F(z),
\]

then

\[
\nabla E=-I(\theta)c,
\qquad
I(\theta)=\operatorname{Cov}(F,F).
\]

For a minimal finite family at a finite parameter point, `I` is positive definite on the active feature space. The Fisher natural-gradient direction is therefore

\[
-I^{-1}\nabla E=c.
\]

Along `theta -> theta + beta c`,

\[
\frac{dE}{d\beta}=-\operatorname{Var}(C)\le0.
\]

The target direction is an exact cooling direction in the fully aligned commuting model.

## Partial alignment

For a restricted representation, decompose

\[
C(z)=c_0+c_G^{\mathsf T}F_G(z)+R_G(z).
\]

Then

\[
\nabla E=-I_Gc_G-r_G,
\]

with

\[
I_G=\operatorname{Cov}(F_G,F_G),
\qquad
r_G=\operatorname{Cov}(F_G,R_G).
\]

The exact natural direction becomes

\[
-I_G^+\nabla E
=P_{\operatorname{range}(I_G)}c_G+I_G^+r_G.
\]

The residual is state dependent. A representation can therefore be preparation-friendly yet difficult to optimize even when a low-energy state exists inside its family.

## Energy certificate

For a unique ground state with energy `E0` and spectral gap `gamma`, every density operator satisfies

\[
\Delta E=\operatorname{Tr}(H\rho)-E_0
\ge\gamma(1-p_\star).
\]

Thus

\[
p_\star\ge1-\frac{\Delta E}{\gamma}.
\]

The confirmatory success criterion `Delta E / gamma <= 0.1` certifies `p_star >= 0.9`.

## Finite-sample aligned identity

When the target is fully represented and the gradient and Fisher matrix use the same sample batch,

\[
\widehat g=-\widehat I c.
\]

The pseudoinverse update recovers the component of `c` visible in the empirical covariance range. This exact cancellation does not survive partial alignment, where the sampled omitted-cost covariance remains.
