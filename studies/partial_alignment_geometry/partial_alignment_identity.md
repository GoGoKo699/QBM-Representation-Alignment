# Partial-alignment geometry for commuting Gibbs ansätze

## Setting

Let

$$
p_\theta(z)=\frac{\exp[-\theta^{\mathsf{T}}F_G(z)]}{Z(\theta)}
$$

be a finite commuting Gibbs family whose retained features are determined by an ansatz graph $G$. Decompose the full target cost as

$$
C(z)=c_0+c_G^{\mathsf{T}}F_G(z)+R_G(z),
$$

where $R_G$ contains the interactions omitted from the ansatz representation.

Define

$$
E(\theta)=\mathbb E_{p_\theta}[C],
\qquad
I_G(\theta)=\mathrm{Cov}_{p_\theta}(F_G,F_G).
$$

## Proposition 1: partial-alignment gradient decomposition

Let

$$
r_G(\theta)=\mathrm{Cov}_{p_\theta}(F_G,R_G).
$$

Then

$$
\boxed{
\nabla_\theta E(\theta)
=-I_G(\theta)c_G-r_G(\theta).
}
$$

### Proof

For every retained feature,

$$
\frac{\partial E}{\partial\theta_j}
=-\mathrm{Cov}(F_{G,j},C).
$$

Substitute the cost decomposition and use bilinearity of covariance:

$$
\mathrm{Cov}(F_G,C)
=I_Gc_G+\mathrm{Cov}(F_G,R_G).
$$

Negating gives the result. ∎

## Corollary 1: natural direction

Using the Moore--Penrose inverse,

$$
-I_G^+\nabla E
=P_{\mathrm{range}(I_G)}c_G+I_G^+r_G.
$$

For a minimal finite family, $I_G$ is nonsingular and

$$
-I_G^{-1}\nabla E=c_G+I_G^{-1}r_G.
$$

The projected target direction $c_G$ is exact only when the omitted-cost covariance vanishes.

This condition is weaker than literal full alignment: an omitted term may be present yet uncorrelated with every retained feature at one parameter point. In general, however, the correction is state dependent.

## Corollary 2: finite-sample decomposition

For one sample batch, let

$$
\widehat I_G
=\widehat{\mathrm{Cov}}(F_G,F_G),
\qquad
\widehat r_G
=\widehat{\mathrm{Cov}}(F_G,R_G).
$$

The same batch obeys the exact algebraic identity

$$
\boxed{
\widehat g=-\widehat I_Gc_G-\widehat r_G.
}
$$

Therefore

$$
-\widehat I_G^+\widehat g
=P_{\mathrm{range}(\widehat I_G)}c_G
+\widehat I_G^+\widehat r_G.
$$

The complete same-batch cancellation of the fully aligned finite-sample study is absent unless $\widehat r_G=0$. Full-Fisher success in the partial family therefore tests genuine covariance estimation rather than merely rediscovering a known coefficient vector.

## Explained cost variance

Let

$$
b_G=\mathrm{Cov}(F_G,C).
$$

The fraction of centered cost variance explained by the best linear predictor from retained features is

$$
\boxed{
\eta_G
=\frac{b_G^{\mathsf{T}}I_G^+b_G}
{\mathrm{Var}(C)}.
}
$$

This is the squared norm of the orthogonal projection of the centered cost onto the span of centered retained features in the Hilbert space $L^2(p_\theta)$, so

$$
0\le\eta_G\le1.
$$

For the full aligned representation, $\eta_G=1$. In the graph construction used for this study, the mean value at $\theta=c_G$ is approximately:

| Representation | Mean $\eta_G$ |
|---|---:|
| Native chain | 0.718 |
| Problem spanning tree | 0.740 |
| Greedy width-2 subgraph | 0.811 |
| Greedy width-3 subgraph | 0.915 |
| Full problem graph | 1.000 |

This quantity measures representational alignment at a declared state. It is not by itself a prediction of optimizer success.

## Implications for approximate geometry

A diagonal, field/pair block, or graph-local covariance approximation changes two things simultaneously:

1. it approximates $I_G^{-1}$;
2. it changes how the omitted-cost correction $r_G$ is propagated through parameter space.

The scientific question is not merely whether an approximation resembles the full Fisher matrix entrywise. It is whether the resulting update preserves the components of

$$
c_G+I_G^{-1}r_G
$$

that are necessary to reach the certified low-energy region.

## Scope

These identities concern the commuting diagonal models in this repository. General noncommuting quantum Boltzmann machines use quantum thermal information matrices rather than the classical covariance matrix alone. The project-specific graph-star preconditioner is an empirical approximation and is not asserted to possess a general natural-gradient invariance property.
