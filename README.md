# Participation-ratio diagnostic for sparsity-adaptive versus MEWMA charts

Code for the paper

> Imran, M., Zaidi, F. S., & He, Z. *A participation-ratio diagnostic, validated in simulation, for choosing between sparsity-adaptive and multivariate EWMA charts for mean shifts.*

The scripts reproduce the simulation study, the Tennessee Eastman and WaDi case studies, and the five figures of the paper. They contain the run-length engine for every chart compared (MEWMA, BAS-EWMA, G-BAS, MCUSUM, VS-MEWMA, LEWMA, top-r CUSUM, thresholded and mixture EWMA, LAR-EWMA and MSGL-EWMA) and `gain_curve.py`, which returns the gain of BAS-EWMA over MEWMA and the break-even participation ratio for a configuration you specify.

## Layout

```
code/      run-length engine, calibration, simulation and case-study scripts, figure script
data/      empty; put the downloaded data here (see "Data")
figures/   created by code/make_figures.py
```

Every script starts with a docstring stating what it does, its inputs and outputs, and the part of the paper it supports.

## Installation

Python 3.13 was used.

```
pip install -r requirements.txt
```

## Running the scripts

Run every script from `code/`. Scripts read and write their JSON result files in `code/`, and read data from `../data/`. Simulation scripts use fixed random seeds, so a rerun reproduces the stored numbers exactly. Many scripts build on the output of others: for example, `front.py` produces the reference limits used by most Section 4 scripts, and `make_figures.py` draws the figures from the JSON files.

The full study takes several days on one machine; individual simulation scripts take from seconds to about a day.

```
cd code
python front.py            # reference designs used by later scripts
python robust2.py
...
python make_figures.py     # writes ../figures/fig1-fig5 (PDF)
```

Gain curve for your own configuration:

```
python gain_curve.py 30 0.05 0.25     # p, lambda, w1
```

## Data

No data are included. Download them from their sources and place them under `data/`.

- **Tennessee Eastman, Braatz data set** (Chiang, Russell & Braatz, 2000): `d00.dat`, `d00_te.dat`, `d01_te.dat` ... `d21_te.dat` in `data/tep_braatz/`. Used for the variable groups (`tep.py`).
- **Tennessee Eastman, Rieth et al. (2017)**, Harvard Dataverse, https://doi.org/10.7910/DVN/6C3JR1. Download `TEP_FaultFree_Training.RData`, `TEP_FaultFree_Testing.RData` and `TEP_Faulty_Testing.RData`, then run
  ```
  python convert_rieth.py <folder with the .RData files>   # writes data/rieth2017/
  python te_cache.py                                          # writes data/te_cache/
  ```
- **WaDi testbed** (iTrust, Singapore University of Technology and Design; access on request at https://itrust.sutd.edu.sg/itrust-labs_datasets/). Then run
  ```
  python wadi_prep.py <zip containing WADI_14days_new.csv> <WADI_attackdataLABLE.csv>
  python wadi_base.py
  python wadi_eval.py
  ```

## Scripts and the results they produce

### Library modules

| Script | What it does | Paper |
|---|---|---|
| `make_figures.py` | Draws Figs | Library module |

### Section 2: charts and the diagnostic

| Script | What it does | Paper |
|---|---|---|
| `bas_ewma.py` | BAS-EWMA chart: balanced calibration of the largest-coordinate term and the MEWMA total | Sections 2.2 and 2.4 |
| `calib.py` | Control-limit calibration helpers | Section 2.4 (balanced calibration, Algorithm 1); used by most simulation scripts |
| `gbas.py` | G-BAS chart: chi-square-standardised largest group energy paired with the MEWMA total | Sections 2.3 and 2.4 |
| `order_check.py` | Dependence of the participation ratio of a physical shift, and of the predicted gain, on the order of the variables under regression adjustment; PR under the symmetric whitening Sigma^{-1/2} | Section 2.5; Supplement S2 and S20 |
| `ss_checks.py` | Steady-state checks: change point tau=50 against tau=5/lambda, steady-state in-control ARL of the compared designs, and the group-EWMA maximum alone in the group benchmark | Sections 2.4 and 3.1; Supplement S3 |

### Section 3: design and calibration checks

| Script | What it does | Paper |
|---|---|---|
| `ar1_check.py` | In-control ARL of MEWMA under AR(1) data with independent-data limits (phi = 0 and 0.3) | Section 3.1; Supplement S14 |
| `calib_err.py` | Confidence intervals for paired ARL differences that include the calibration error of each design's control limit (elasticity of the out-of-control ARL with respect to the in-control ARL) | Sections 3 and 4.3; Supplement S3 |
| `cond_arl0.py` | Conditional in-control ARL over Phase I samples after matching the unconditional ARL to 200 | Section 3.1; Supplement S4 |
| `engine.py` | Numba run-length engine for all multivariate mean-shift charts in the paper | All simulations in Sections 3-4 and the supplement |
| `phase1.py` | Effect of Phase I estimation on the gain of BAS-EWMA over MEWMA | Section 3.1 (estimated parameters) |
| `serial.py` | Effect of serial correlation on calibration and on the diagnostic | Section 3.1 (autocorrelation); Supplement S14 |

### Section 4: simulation results

| Script | What it does | Paper |
|---|---|---|
| `ablation_clean.py` | Ablation of the balancing step: the two BAS-EWMA terms with one limit on their chi-square- standardised scale (as in VS-MEWMA) against the balanced limits | Section 4.3 (ablation) |
| `arl500.py` | Recalibrates key designs to an in-control ARL of 500 and rechecks rankings and the break-even | Sections 4.2 and 4.3 (in-control ARL 500) |
| `be_theory.py` | Fixed-sample (one-look) power approximation of the break-even, treating each look as a test at a fixed level | Section 4.2; Supplement S13 |
| `be_wls.py` | Standard errors of the break-even locations (delta method on the local gain grid) and weighted least-squares fits of the break-even on ln p, sqrt(p), sqrt(p)/ln p, ln ln p and a power of p, with lack-of-fit tests | Fig. 1 and Section 4.2; Supplement S20 |
| `bench_screen.py` | Calibrated limits for the competing sparsity-adaptive charts at p = 30, used as the starting designs of the benchmark | Section 4.3 |
| `bench_sparse_charts.py` | Benchmark of BAS-EWMA against the charts designed for sparse shifts (MEWMA, MCUSUM, VS-MEWMA, LEWMA, top-r CUSUM) at p = 30, Sigma = I, steady state (tau = 50), in-control ARL 200 | Section 4.3 |
| `breakeven_by_dimension.py` | Gain of BAS-EWMA over MEWMA against the participation ratio at one dimension p (command-line argument) and its break-even | Fig. 1 and Section 4.2 (break-even from p=10 to 1000) |
| `breakeven_large_p.py` | Break-even at p = 200, 500 and 1000, used to compare logarithmic and power-law growth | Fig. 1 and Section 4.2 |
| `breakeven_law_and_ss_adjustment.py` | Fits of the break-even and of the sparse-term limit against ln p and ln p + ln ln p, and the adjustment of out-of-control ARLs to a common steady-state in-control ARL | Sections 4.2-4.3; Supplement S3 and S20 |
| `breakeven_refine.py` | Locates the break-even at p = 100, 200 and 500 on a dense PR grid, reusing the limits from breakeven_by_dimension.py | Fig. 1 and Section 4.2 |
| `collapse.py` | Gain of BAS-EWMA against the participation ratio for 54 structured shift directions (equal-loading anchors and one-variable / all-variable shifts under eight covariance structures, in regression-adjusted coordinates), with the cubic fit in log PR | Section 4.2; Supplement S1 |
| `cv_struct.py` | Leave-one-covariance-structure-out cross-validation of the cubic fit and its residual standard error (percentage points) | Section 4.2; Supplement S1 |
| `diverse_dirs.py` | Out-of-sample test of shape summaries on 48 random directions with diverse shapes | Section 4.2; Supplement S1 |
| `final.py` | Tuned designs of every chart family at p = 30 (lambda, k, w1, LEWMA penalties) and their steady-state ARL profiles over shift sizes and sparsity levels | Section 4.3, Fig. 2 and Table 2 |
| `final_extra_designs.py` | Adds MCUSUM (k = 0.2, 0.3) and BAS-EWMA (lambda = 0.02) designs to final.json | Section 4.3 |
| `final_extra_mcusum.py` | Adds MCUSUM designs with k = 1.5 and 2.0 to final.json | Section 4.3 (MCUSUM reference value range) |
| `front.py` | Tuned families of all charts at p = 30: limits and steady-state ARL profiles used as the common reference designs (MEWMA and BAS-EWMA limits are reused by later scripts) | Section 4.3, Fig. 2 |
| `gbas_ss.py` | Steady-state in-control ARL of MEWMA and the G-BAS designs (p = 30) | Section 4.4; Supplement S17 |
| `group_bench.py` | Benchmark with known groups: p = 30 in six groups of five, Sigma = I, steady state, in-control ARL 200; G-BAS, MEWMA and coordinate-sparse charts on in-group and cross-group shifts | Section 4.4, Fig. 3 |
| `group_mix.py` | Group-level mixture chart, sum_k ln{1 - p0 + p0 exp(Q_k / 2)}, on the group benchmark | Section 4.4; Supplement S17 |
| `group_slep.py` | MSGL-EWMA (reading L, SLEP penalty order) on the group benchmark (p = 30, six groups of five) | Section 4.4 |
| `group_w1.py` | Exchange rate of G-BAS: in-group gain against worst-case cost over the budget share w1 | Section 4.4, Fig. 3 |
| `heavy_tail.py` | Gain of BAS-EWMA (w1=0.25) over MEWMA under a multivariate t distribution with 5 degrees of freedom and under normality, with empirically calibrated limits | Section 4.2 (heavy tails) |
| `lam20_cal.py` | Calibration of BAS-EWMA and MEWMA at lambda = 0.20 for p = 10, 100 and 1000 | Section 4.2 (slope at lambda = 0.20) |
| `lam20_scan.py` | Break-even at lambda = 0.20 using the limits from lam20_cal.py | Section 4.2 (slope at lambda = 0.20) |
| `lar_mix.py` | LAR-EWMA (Capizzi & Masarotto, 2011; for Sigma = I the maximum of standardised sums of the k largest u_i) and a mixture EWMA in the spirit of Xie & Siegmund (2013), sum_i ln{1 - p0 + p0 exp(u_i / 2)} | Section 4.3, Fig. 2, Table 2; Supplement S12 |
| `lewma.py` | LEWMA (Zou & Qiu, 2009) designs over the penalty path for one lambda (command-line arguments), calibrated and profiled; results added to final.json | Section 4.3 |
| `mix_dim.py` | Break-even of the mixture EWMA (p0 = 0.3) against MEWMA over dimension | Section 4.2; Supplement S16 |
| `mix_dimk.py` | As mix_dim.py with p0 = 3/p (fixed expected number of shifted coordinates) | Section 4.2; Supplement S16 |
| `msgl_authors_rmi.py` | Relative mean index (RMI) and worst case of each chart family's best design in the setting of Kim et al. (2024) | Section 4.4 (RMI and worst case) |
| `msgl_authors_setting.py` | G-BAS, MEWMA and MSGL-EWMA in the setting of Kim et al. (2024): five equal groups, within-group correlation 0.75, steady state, in-control ARL 200 | Fig. 4 and Section 4.4 |
| `msgl_authors_setting_reading_A.py` | MSGL-EWMA reading A re-tuned over the full (c1, c2) range in the setting of Kim et al. (2024) | Section 4.4; Supplement S6 |
| `msgl_authors_setting_reading_L.py` | MSGL-EWMA reading L (SLEP penalty order, the version used in the paper) tuned over the full (c1, c2) range in the setting of Kim et al. (2024), p = 25 | Fig. 4 and Section 4.4 |
| `msgl_authors_setting_reading_R.py` | MSGL-EWMA reading R (likelihood ratio refitted on the selected variables) re-tuned over the full (c1, c2) range in the setting of Kim et al. (2024) | Section 4.4; Supplement S6 |
| `msgl_bench.py` | MSGL-EWMA (Kim et al., 2024) tuned over its penalty constants against G-BAS on the group benchmark (10 shift patterns, 2 shift sizes, in-control ARL 200) | Section 4.4 |
| `msgl_bench_extend.py` | Extends the MSGL-EWMA tuning of msgl_bench.py around its best region | Section 4.4 |
| `pr_dense.py` | Prediction errors of the shape summaries on 16 additional dense directions | Section 4.2 (dense directions); Supplement S14 |
| `pr_general.py` | Gain of four adaptive charts (BAS-EWMA, mixture EWMA, soft-thresholded EWMA, LAR-EWMA) over MEWMA on 8 equal-loading and 48 random directions, on common random numbers | Table 1 and Section 4.2; Supplement S9 |
| `pr_general_d1.py` | As pr_general.py at shift size delta=1.0 | Section 4.2 (second shift size); Supplement S19 |
| `pr_general_fit.py` | Cubic fits in log PR for the four charts of pr_general.py, adjusted to MEWMA's steady-state in-control ARL (elasticity 0.37), and their break-even points | Section 4.2 (chart-specific break-even); Supplement S9 |
| `pr_halves.py` | Monte Carlo noise floor of the prediction error, estimated by splitting the runs in half | Section 4.2 (noise floor); Supplement S17 |
| `robust2.py` | Steady-state ARLs of the reference designs over shift sizes and sparsity levels, reusing the limits in front.json | Fig. 2, Table 2 and Section 4.3 |
| `shape_and_design_sensitivity.py` | (a) Equal-PR directions with different shapes; (b) dependence of the gain curve and the break-even on delta and w1 | Section 4.2 (design dependence of the break-even); Supplement S1 |
| `shape_general.py` | Prediction errors of shape summaries (participation ratio, largest loading, Hill numbers) for the four charts of pr_general.py on the 48 random directions | Table 1; Supplement S14 |
| `shape_general_d1.py` | As shape_general.py at shift size delta=1.0 | Section 4.2; Supplement S19 |
| `shape_se.py` | Monte Carlo standard errors of the gains of equal-PR directions with different shapes (paired, delta in {0.75, 1.0}) | Section 4.2 (direct shape test); Supplement S1 |
| `shape_summaries.py` | Out-of-sample comparison of shape summaries: cubic trained on the 54 structured directions, tested on the 48 random directions | Section 4.2; Supplement S1 |
| `split_select.py` | Design selection on a separate tuning grid against selection on the evaluation grid | Section 4.1 (hold-out selection); Supplement S3 and S10 |
| `threshold_ewma.py` | Thresholded EWMA charts: hard threshold sum_i u_i 1{u_i > c} (Wu & Wu, 2025) and the soft-thresholded variant sum_i (u_i - c)_+, for c in {2, 4, 6}, with MEWMA and BAS-EWMA (w1 = 0.4) in the same framework | Section 4.3, Fig. 2, Table 2 |
| `topr_k.py` | Tuning of the top-r CUSUM reference value k in {0.25, 0.375, 0.5} for r in {1, 3, 5}; p = 30, delta = 0.75, in-control ARL 200, steady state, same seeds as robust2.py | Section 4.3, Fig. 2, Table 2 |
| `unequal_groups.py` | G-BAS with unequal correlated groups (sizes 2, 3, 5, 8, 12; within-group correlation 0.6): raw against chi-square-standardised group maximum | Section 4.4 (unequal groups); Supplement S11 |

### Section 5: case studies (Tennessee Eastman, WaDi)

| Script | What it does | Paper |
|---|---|---|
| `convert_rieth.py` | Converts the Rieth et al. (2017) Tennessee Eastman .RData files to .npy arrays | Section 5.1 (data) |
| `fig5_ci.py` | Per-fault 95% intervals for the delay ratios on raw observations (paired delta method) | Fig. 5 |
| `negctl.py` | Negative control: fault-free testing runs treated as a pseudo-fault, same delay measure | Section 5.2 (pseudo-fault); Supplement S5 |
| `rieth.py` | Tennessee Eastman, Rieth et al. (2017) data on raw observations: Phase I from the 500 fault-free training runs, standardisation, Ledoit-Wolf covariance, calibration of MEWMA, G-BAS and BAS-EWMA (lambda = 0.1) | Section 5.1 |
| `rieth_eval.py` | Charts on the faulty training runs, where each fault starts at sample 21 (zero-state effect) | Section 5.2 (faulty training runs) |
| `rieth_pr.py` | Participation ratio of the mean post-fault signal for each fault, on observations and on VAR(1) residuals, and its correlation with the gain | Section 5.2 (diagnostic fault by fault) |
| `rieth_res.py` | As rieth.py on the one-step residuals of a VAR(1) model fitted to the fault-free training runs | Section 5.2 (residuals) |
| `te_bootstrap_equal.py` | Bootstrap confidence intervals for the residual gains under the primary protocol (equal in-control ARL on the testing runs, all runs) | Section 5.2 (bootstrap intervals); Supplement S15 |
| `te_cache.py` | Caches the EWMA vectors (lambda = 0.1) of raw observations and VAR(1) residuals for all Tennessee Eastman run sets ('split': VAR and Phase I fitted on training runs 1-250 only) | Sections 5.1-5.2 |
| `te_cache_var.py` | Caches residual EWMA arrays for a VAR(k) filter fitted on all fault-free training runs | Section 5.2 (VAR(10) check); Supplement S15 |
| `te_equal.py` | Primary Tennessee Eastman protocol: every chart's limits set to an in-control ARL of exactly 200 on the fault-free testing runs; delay of the first alarm at or after the fault over all 500 runs; paired t-tests on log delays with Holm adjustment over 80 comparisons | Fig. 5, Table 3 and Section 5.2; Supplement S8 |
| `te_lib.py` | Shared helpers for the Tennessee Eastman analyses on the cached EWMA arrays | Sections 5.1-5.2; Supplement S8, S11, S15 |
| `te_random_groups_equal.py` | Random partitions of the variables under the primary protocol of te_equal.py, in batches | Section 5.2 (random partitions); Supplement S11 |
| `te_random_groups_equal_summary.py` | Summary of the random partitions of te_random_groups_equal.py (100 partitions): mean gain, 5-95% range, share without a significant loss, and the percentile of the engineering grouping | Section 5.2 (random partitions); Supplement S11 |
| `te_split.py` | Residual comparison with the VAR(1) filter and Phase I fitted on half of the training runs | Section 5.2 (half-sample filter); Supplement S15 |
| `tep.py` | Tennessee Eastman (Braatz data set): variable list, process-unit groups and loading | variable groups of Section 5.1 (assignment in Supplement S5) |
| `var_order.py` | VAR order selected by BIC on the fault-free training runs (orders 1-10) | Section 5.2 (VAR(10) check); Supplement S15 |
| `wadi_base.py` | WaDi, step 1: cleaning, Phase I (standardisation, VAR(1) filter, shrinkage whitening) on the normal days, control-loop groups and EWMA statistics | Section 5.3; Supplement S18 |
| `wadi_eval.py` | WaDi, step 2: charts calibrated to an in-control ARL of 200 samples on the attack-free periods, pseudo-attack false-alarm check, detection delay per attack and participation ratio of each attack's shift | Section 5.3; Supplement S18 |
| `wadi_prep.py` | Prepares the WaDi data: reads the normal-operation and attack files, averages to 10-second samples and keeps the continuous signals | Section 5.3 |

### Section 6: discussion and practitioner tool

| Script | What it does | Paper |
|---|---|---|
| `gain_curve.py` | Practitioner tool: gain of BAS-EWMA over MEWMA, 1 - ARL/ARL_MEWMA (positive = faster), as a function of the participation ratio for a user-specified configuration (p, lambda, w1, delta, in-control ARL), and the break-even participation ratio | Algorithm 2 (procedure for using the diagnostic) |
| `lamscale.py` | Calibrated limit of a pure maximum-EWMA term against p for several lambda, h1 = 2 ln p + c | Section 6 (limit growth approximately as ln p) |
| `lamscale_intercepts.py` | Intercepts c(lambda) of h1 = 2 ln p + c for a pure maximum-EWMA term at in-control ARL 200 | Section 6 (limit growth approximately as ln p) |

### Supplementary material only

| Script | What it does | Paper |
|---|---|---|
| `holm.py` | Holm adjustment over the paired Tennessee Eastman tests under the matched protocol | Supplement S5 |
| `holm_eval2.py` | Holm adjustment over the paired tests of te_eval2.py (raw and residuals, two charts, 20 faults) | Supplement S5 |
| `holm_eval2_pooled.py` | As holm_eval2.py for limits matched on the pooled fault-free runs | Supplement S5 |
| `msgl_repro_reading_A.py` | Replication check of MSGL-EWMA reading A against Tables 2-4 of Kim et al. (2024): delta as noncentrality, zero state, smoothing 0.2, in-control ARL 200 | Supplement S6 |
| `msgl_repro_reading_L.py` | Replication check of MSGL-EWMA reading L (SLEP penalty order) against Tables 2-4 of Kim et al. (2024) | Supplement S6 |
| `msgl_repro_reading_R.py` | Replication check of MSGL-EWMA reading R against Tables 2-4 of Kim et al. (2024) | Supplement S6 |
| `msgl_repro_table.py` | Builds the replication table of Supplement S6 from the three readings and the published values | Supplement S6 |
| `rieth_randgroups.py` | Sensitivity of G-BAS on VAR(1) residuals to the grouping: engineering groups against random partitions with the same group sizes, matched protocol | Supplement S5 |
| `rieth_res_matched.py` | False-alarm matching of the residual charts on the 500 fault-free training runs: common limit scale that matches MEWMA's mean censored first-alarm time | Supplement S5 and S8 |
| `rieth_res_test.py` | As rieth_test.py on VAR(1) residuals | Supplement S5 |
| `rieth_res_test_matched.py` | Faulty testing runs on VAR(1) residuals under the matched limits of rieth_res_matched.py | Supplement S5 |
| `rieth_test.py` | Faulty testing runs on raw observations (fault after sample 160) under limits matched on the fault-free training runs | Supplement S5 |
| `sdrl_table.py` | Run-length distribution (ARL, SDRL, median, 90th percentile) for the designs of Figs | Supplement S7 |
| `te_boot.py` | Bootstrap confidence intervals for the residual gains under matched limits, resampling the limit-setting, matching and faulty runs (raw group maximum) | Supplement S5 |
| `te_boot_std.py` | As te_boot.py with the chi-square-standardised group maximum | Supplement S5 |
| `te_eval2.py` | Tennessee Eastman comparison with limits matched on the training runs or on the pooled fault-free runs (conditional delays) | Supplement S5 and S8 |
| `te_random_groups.py` | G-BAS with the chi-square-standardised group maximum for the engineering groups and for 100 random partitions with the same group sizes (raw and standardised maxima), matched protocol | Supplement S5 |
| `te_random_groups_batch.py` | Random partitions of the variables under the matched protocol, evaluated in batches of seeds (raw and standardised maxima) | Supplement S5 |
| `te_random_groups_cont1.py` | Continues te_random_groups.py for the remaining random partitions, evaluating the raw and standardised maxima together | Supplement S5 |
| `te_random_groups_eval2.py` | Random partitions of the variables under the protocol of te_eval2.py, in batches | Supplement S11 |
| `te_random_groups_eval2_summary.py` | Summary of the random partitions of te_random_groups_eval2.py: mean gain, percentiles, Holm- significant gains and losses, and the percentile of the engineering grouping | Supplement S11 |
| `te_robust.py` | Robustness of the residual comparison under matched limits: all runs, log delays and the Wilcoxon signed-rank test, Holm over faults (raw group maximum) | Supplement S5 |
| `te_robust_std.py` | As te_robust.py with the chi-square-standardised group maximum | Supplement S5 |
