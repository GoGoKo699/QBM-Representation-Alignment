# Excited-boundary optimizer study: protocol

## Scope

This study examines the twenty fixed-\(n=16\), certificate-tight positive
Exact-1-in-3-SAT instances reconstructed from the the deterministic certificate-tight instance contract.  It uses exact state enumeration throughout.  No sampling
noise, circuit noise, or approximate covariance estimator is present.

The objective is the expected Exact-1-in-3-SAT Hamiltonian energy.  Success is

\[
\Delta E\le 0.1,
\]

which certifies planted-solution probability at least \(0.9\) because every
instance has a unique ground state and unit gap.

## Frozen trap definition

`TRAP_DEFINITION.json` was written before the aggregate optimizer results were
examined.  A recorded state is an excited-boundary trap when all of the
following hold:

1. \(\Delta E>0.1\);
2. the dominant state is not a ground state;
3. dominant-state probability is at least \(0.9\);
4. planted-state probability is at most \(0.1\);
5. effective Fisher condition number is at least \(10^6\);
6. gradient RMS is at most one tenth of its value at the trajectory's initial
   state.

This is intentionally stronger than ordinary optimization failure.

## Reproduced baselines

Vector Adam starts at the exactly aligned coefficients \(\theta=c\) and runs
for 1,000 recorded states with

\[
\eta=0.02,\qquad \beta_1=0.9,\qquad \beta_2=0.999,
\qquad \epsilon=10^{-8}.
\]

The two documented long-run numerical failure fingerprints are reproduced.  The
second fingerprint is labelled `ct_w6_i1` in the deterministic reconstruction;
the matching is based on the complete numerical fingerprint.

Parameters are saved at records 199 and 999.

## Same-state replay methods

Every replay begins from exactly the saved parameter vector.  Optimizer history
is reset unless stated otherwise.  Continued Adam is already represented by
the original 1,000-state baseline.

### Restarted Adam

Standard vector Adam with fresh first- and second-moment accumulators.

### Armijo Euclidean gradient descent

Direction

\[
d=-\nabla E,
\]

with backtracking until

\[
E(\theta+\alpha d)
\le E(\theta)+10^{-4}\alpha\nabla E^{\mathsf T}d.
\]

### Fixed target-direction cooling

The saved transverse component is retained and

\[
\theta\leftarrow\theta+0.02c.
\]

This tests whether merely supplying the correct direction is sufficient after
a trap has developed.

### Scalar-ray projection and cooling

First project onto

\[
\theta=\beta c,
\qquad
\beta=\frac{\theta^{\mathsf T}c}{c^{\mathsf T}c},
\]

then apply the same fixed target-direction cooling.  This tests whether removal
of the accumulated transverse component is causal.

### Projected Adam

Compute the ordinary Adam proposal \(d_{\rm A}\) and keep only its Euclidean
target component,

\[
d=\frac{d_{\rm A}^{\mathsf T}c}{c^{\mathsf T}c}c.
\]

The existing transverse parameter component is retained.

### Exact aligned natural direction

For the fully aligned commuting family,

\[
I(\theta)^{-1}\nabla E=-c.
\]

The descent direction is therefore \(c\); an Armijo line search selects its
step length.  This is an exact oracle control, not a practical covariance
estimator.

### Damped full-Fisher natural gradient

Solve

\[
\left(I+10^{-8}\lambda_{\max}(I)\mathbb I\right)d=-\nabla E
\]

and apply an Armijo line search.  This tests the numerically regularized full
geometry near a nearly singular boundary.

### Diagonal Fisher

Use

\[
d_j=-\frac{g_j}{I_{jj}+10^{-6}\max_k I_{kk}},
\]

followed by Armijo backtracking.

## Full-family suite

All twenty instances are evaluated under:

- exact-target initialization \(\theta=c\);
- five matched target-biased initializations
  \(\theta=c+0.3\xi\), with seeds `0, 19, 42, 50, 101`.

The automatic suite compares restarted Adam, Armijo GD, fixed target direction,
scalar-ray projection, projected Adam, exact aligned natural direction, and
diagonal Fisher for 200 recorded states.

## Statistical unit

Instances, not trajectories, are the primary resampling unit.  Confidence
intervals use an instance-cluster bootstrap.  Pairwise optimizer comparisons
use identical instance/seed starts and report both cluster-bootstrap intervals
and discordant paired outcomes.

## Interpretation limits

- Exact natural direction is an oracle in this aligned model.
- Exact moments remove estimator noise; finite-sample performance is studied separately.
- Projecting onto the scalar ray deliberately changes the parameter state and
  therefore diagnoses the role of transverse displacement rather than serving
  as a state-preserving optimizer comparison.
- Absence of rescue within a finite budget does not prove impossibility.
