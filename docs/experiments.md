# Experiments

The three [main studies](../notebooks/README.md#main-studies) contain their controls, simulation calls, plotting code, and numerical exports. The settings below describe the supplied notebook defaults.

| Study | Particles | Time grid | Neural map | Tangent basis | Reference |
| --- | --- | --- | --- | --- | --- |
| [Five-player Cournot](../notebooks/cournot_5d_analysis.ipynb) | 5,000 | h=0.005, T=2 | Residual MLP, width 16, depth 2 | 64 resampled coordinates, SVD cutoff 1e-6 | Euler–Maruyama in the default stochastic mode; Euler in deterministic mode |
| [Oscillatory frequency comparison](../notebooks/oscillatory_direct_vs_periodic_refit.ipynb) | 10,000 training + 10,000 independent validation | h=0.005, T=1 | Residual MMNN, width/rank/depth 12/12/3 | Requested 512, capped at all 338 available trainable coordinates, SVD cutoff 1e-5 | RK4, maximum step 0.000125 |
| [Eight-dimensional potential dynamics](../notebooks/potential_game_8d_mmnn_dtb.ipynb) | 10,000 | h=0.001, T=2 | Residual MMNN, width/rank/depth 24/12/4 | Full trainable basis, SVD cutoff 1e-3 | RK4, maximum step 0.00025 |

All three main notebooks use tanh activations and float64 arithmetic. Source-code calculations and parameter settings are independent of the folder used to store a notebook.

The following figures are extracted from the report. Figure captions give their settings; the notebook-default table above describes a separate preset where those settings differ.

## Five-player Cournot

The game uses b=2 and mu=7/4, with smoothed-uniform initial particles and smoothing standard deviation 0.02. The default stochastic mode uses isotropic Brownian amplitude 0.1, corresponding to covariance 0.01 I. Set `RUN_STOCHASTIC=False` to use deterministic dynamics. `RUN_STEP_SIZE_SWEEP` selects an optional matched step-size sweep. Snapshot times are 0, 0.5, 1, and 2.

Stochastic probability-flow particles and Euler–Maruyama reference paths are different objects. Label-paired RMS is descriptive; distributional comparisons use sliced Wasserstein distance.

| Neural-DTB | Explicit Euler reference |
| :---: | :---: |
| ![Five-player deterministic Cournot particle clouds under Neural-DTB](images/cournot-dtb.png) | ![Five-player deterministic Cournot particle clouds under explicit Euler](images/cournot-euler.png) |

*Report Figures 4-5. Columns show t = 0, 0.5, 1, 2; rows show adjacent coordinate pairs from (x1, x2) through (x4, x5). The plotted report run is deterministic, with 10,000 particles and h = 0.001.*

<details>
<summary>Step-size diagnostics from the report</summary>

![Cournot relative tangent-projection error across step sizes](images/cournot-step-size.png)

*Report Figure 3. Last-step relative tangent-projection error for h = 0.02, 0.01, 0.005, and 0.0025.*

</details>

## Oscillatory frequency comparison

Only omega/pi is swept, through 1, 4, and 8. Both adaptation methods evolve accumulated physical particles. The direct method applies a parameter Euler update after projection. The periodic method holds parameters fixed between supervised refits and fits the neural map to accumulated particles every 10% of the outer time grid.

The methods share seed 2026 for particle sampling and seed 2126 for model initialization. Independent validation labels use seed 9026. Each final tangent basis is fitted afresh to the oscillatory target evaluated on the same RK4 validation cloud. The diagnostic measures final basis representation capacity; it is not a validation error for coefficients fitted on the training cloud.

Outputs include frequency diagnostics, final particle clouds for both methods and RK4, and the corresponding plots and archive.

![Final particle clouds across oscillation frequencies: direct parameter updates, periodic full-basis refitting, and RK4](images/oscillatory-clouds.png)

*Report Figure 7. Rows show omega/pi = 1, 4, 8; columns show direct parameter updates, periodic full-basis refitting, and RK4. Final time T = 1, seed 2026. The report run uses h = 0.001 and SVD cutoff 1e-8; panel labels retain its paired RMS values.*

<details>
<summary>Frequency-sweep diagnostics from the report</summary>

![Oscillatory frequency-sweep representation error, coefficient norm, captured energy, and final paired RMS](images/oscillatory-diagnostics.png)

*Report Figure 6. Representation error, coefficient norm, captured-energy fraction, and final paired RMS against RK4 for the two adaptation methods.*

</details>

## Eight-dimensional potential dynamics

The velocity is the gradient of the notebook's potential, with all-to-all coupling, lambda=0.5, gamma=0.2, epsilon=0.5, and omega=4 pi. The experiment starts from uniform labels on [-1,1]^8 and an identity-initialized residual MMNN. The physical state is accumulated through tangent increments evaluated on fixed labels.

The full tangent basis is used by default. The notebook retains the gradient check, optional NumPy SVD recovery, accumulated-map replay check, and complete per-step map history. Contraction is examined through particle-cloud evolution; the notebook does not prove a global contraction theorem.

Outputs include four coordinate-pair snapshot panels, projection diagnostics, mean potential, domain-exit fractions, raw arrays, and a checkpoint containing the accumulated-map terms.

| MMNN-DTB | RK4 reference |
| :---: | :---: |
| ![Eight-dimensional potential-game particle clouds under MMNN-DTB](images/potential-8d-dtb.png) | ![Eight-dimensional potential-game particle clouds under RK4](images/potential-8d-rk4.png) |

*Report Figures 9-10. Columns show t = 0, 0.5, 1, 2; rows show (x1, x2), (x3, x4), (x5, x6), and (x7, x8). Both methods start from the same 10,000-particle ensemble. The report uses h = 0.001 and a refined RK4 reference.*

## Supplementary studies

The [notebook catalogue](../notebooks/README.md#supplementary-studies) lists the step-size, sample-size, periodic-refit, and stochastic two-dimensional potential studies. Their controls are separate experiments; do not substitute them for the three main studies when comparing results.
