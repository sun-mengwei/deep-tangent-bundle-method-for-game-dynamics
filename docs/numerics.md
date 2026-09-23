# Numerical methods

This page defines the state updates and diagnostics implemented by the package. The named experiments and their controls are listed in [Experiments](experiments.md).

## Tangent projection

Let $X_k\in\mathbb{R}^{N\times d}$ be the physical particles, $T_\theta$ the neural map, and $S_k$ the selected trainable parameter coordinates. The tangent matrix is

$$
J_k=D_{\theta_{S_k}}T_{\theta_k}(B_k)
\in\mathbb{R}^{Nd\times |S_k|}.
$$

Its input $B_k$ is either the current particles $X_k$ or immutable initial labels $z$. These choices define different experiments. The physical velocity target is evaluated at the current physical particles in both cases.

For target velocity $v_k\in\mathbb{R}^{N\times d}$, the solver forms $A=J_k/\sqrt N$ and $y=\mathrm{vec}(v_k)/\sqrt N$, computes an SVD, and retains singular values satisfying the strict threshold

$$
s_i>\tau s_0.
$$

It computes the truncated pseudoinverse solution $\alpha_k$ and projected particle velocity $u_k=J_k\alpha_k$. There is no ridge penalty. A matrix with no positive or retained singular value raises an error rather than silently returning a zero velocity. The normalization changes reported singular-value scales but not the least-squares minimizer in exact arithmetic.

Selecting a fixed coordinate subset keeps $S_k$ constant; it does not freeze $\theta_k$ or the Jacobian. Selecting the full trainable basis gives the same coordinate set even if the selection routine consumes a new random permutation each step.

## Three distinct state-update rules

### Accumulated particles with parameter Euler updates

The standard DTB step is

$$
X_{k+1}=X_k+h_kJ_k\alpha_k,
\qquad
\theta_{k+1}[S_k]=\theta_k[S_k]+h_k\alpha_k.
$$

Unselected parameters are unchanged. The updated neural map is not evaluated to replace the physical particles. For fixed-label experiments, the initial map must agree with the initial particles; an identity-initialized residual network satisfies this condition. Optional network tracking measures $T_{\theta_k}(z)$ separately from the accumulated state.

### Direct neural-map evolution

The direct-map comparison computes its target at $X_k=T_{\theta_k}(z)$, applies the same form of parameter Euler update, and then sets

$$
X_{k+1}=T_{\theta_{k+1}}(z).
$$

It does not maintain a separate accumulated physical state. The two updates agree to first order in the parameter increment, but nonlinear terms can produce different trajectories. Shared initial conditions and coordinate schedules do not make the two methods identical.

### Accumulated particles with periodic supervised refits

The refit method evaluates tangents at fixed labels. Ordinary steps use the configured coordinate subset, and scheduled refit steps use every trainable coordinate. Each step first advances physical particles by $h_kJ_k\alpha_k$. Between refits, parameters remain fixed. At a refit event, a warm-started Adam fit minimizes

$$
\frac1N\sum_{i=1}^N\|T_\theta(z_i)-X_{i,k+1}\|_2^2.
$$

Only trainable parameters are optimized. The refit stops at the iteration limit or when network-particle RMS falls below the larger of the absolute tolerance and the relative tolerance times particle displacement since the previous refit. A fresh optimizer is used for each event. There is no parameter-norm or parameter-change penalty. The final outer step is always a refit event. Fitting changes the basis for subsequent projections; it never overwrites the accumulated physical particles.

## Neural parameterizations

The package provides ordinary and residual MLPs and MMNNs. An MMNN layer has the form $A\,\sigma(Wx+b)+c$. Random features $W,b$ are frozen, while $A,c$ are trainable. Only trainable coordinates enter the tangent parameter vector.

For a two-dimensional residual MMNN with width/rank/depth 12/12/3, the trainable parameter count is

$$
2(12\cdot12+12)+(2\cdot12+2)=338.
$$

Zeroing the final trainable layer initializes the residual map as the identity. It can also make some tangent columns vanish initially, so the number of available parameter coordinates and the retained tangent rank are different quantities. A 338-coordinate basis does not imply numerical rank 338.

## Deterministic and stochastic targets

For deterministic dynamics, $v_k=b(X_k,t_k)$. The automatic reference integrator is explicit Euler. Deterministic experiments may instead request RK4; its configured reference step is the maximum substep within each outer interval.

For an SDE $dX=b(X,t)\,dt+\Sigma(X,t)\,dW_t$, define $a=\Sigma\Sigma^\top$ and the score $q=\nabla\log\rho$. The probability-flow target is

$$
v=b-\tfrac12\left(\nabla\cdot a+a q\right).
$$

The stochastic DTB runner transports the score with the projected field using an explicit Euler update of

$$
\dot q=-(D_xu)^\top q-\nabla(\nabla\cdot u).
$$

Its supported stochastic tangent mode uses current physical particles. The automatic stochastic reference is Euler–Maruyama. Noise amplitude and covariance are distinct: isotropic amplitude 0.1 means $\Sigma=0.1I$ and $a=0.01I$. A functional diffusion supplies both its noise matrix and covariance divergence.

Probability-flow particles and stochastic reference paths are different objects. A label-paired RMS can be recorded descriptively, but it is not a pathwise stochastic error measure. Stochastic sweep comparisons use sliced 2-Wasserstein distance for a distributional comparison.

## Diagnostics and time indexing

The per-step projection residual is

$$
r_k=\frac{\|J_k\alpha_k-v_k\|_F}{\|v_k\|_F},
$$

with a machine-tiny denominator floor. The absolute residual is the particle RMS of the vector error. Coefficient norm is $\|\alpha_k\|_2$. The reported condition number is the largest singular value divided by the smallest **retained** singular value, not a condition number of the full untruncated matrix.

Paired trajectory RMS against a deterministic reference is

$$
E(t)=\sqrt{\frac1N\sum_i\|X_i(t)-X_i^{\rm ref}(t)\|_2^2}.
$$

Projection histories used for evolution are evaluated before the state updates, at $t_0,\ldots,t_{K-1}$. The experiment runner performs a fresh projection at the final state $T=t_K$. When coordinates are resampled, this final diagnostic draws the next reproducible subset $S_T$. The periodic-refit runner uses the full basis for its final projection. Final-state quantities must not be labeled as the last update's projection at $T-h$.

In the oscillatory frequency comparison, the final full tangent basis is evaluated on independent labels and fitted directly to the oscillatory target on their RK4 images. Its coefficient norm and condition number belong to that separate validation-cloud fit. A low representation residual does not by itself establish accurate physical trajectories, stable coefficients, or generalization of training-fitted coefficients.

## Reproducibility conventions

Particle sampling, model initialization, coordinate selection, and reference noise have distinct roles. The standard experiment runner uses the configured seed for particles and model initialization, seed plus one for coordinate selection, and seed plus two for reference noise unless an explicit reference seed is supplied. Some experiments supply an independently constructed model; their model seed is part of that experiment's configuration.

Changing a seed, initial law, precision, model initialization order, tangent-input mode, coordinate schedule, SVD cutoff, or state-update rule changes the numerical experiment. The deterministic potential study intentionally consumes its initial uniform sample from the global generator before constructing the model; this initialization order is preserved.

An outer time grid must contain an integer number of steps, and requested snapshots must lie on that grid. Device selection and numerical precision are recorded with run settings. Reproducible seeds support matched comparisons; they do not promise bitwise-identical SVD results across devices and library versions.

The three main notebooks in `notebooks/` and the additional studies in `notebooks/supplementary/` retain their controls, intermediate analysis, and plotting code. They import the shared numerical package directly. Saved final neural parameters alone should not be treated as a replayable accumulated particle trajectory; replay requires the applicable state-update history and, for frozen-feature models, the fixed features as well.
