"""
Scientific audit, Stage F CORRECTED.

The original Stage F pooled all 4 jitter levels x 15 seeds = 60 observations
per device into a single OLS regression, treating all 60 points as
independent. This is WRONG: the same 15 seed values (2000-2014) are reused
at every jitter level, and because the decorrelation-jitter line is the
LAST random draw in the pipeline (after telemetry generation, which is
identical for a given seed regardless of jitter level), the 4 observations
for a given seed are strongly correlated -- this is repeated-measures /
clustered data, not 60 i.i.d. draws. Pooled OLS standard errors on such data
are anti-conservative (too small), inflating apparent significance.

CORRECTED APPROACH: fit a separate within-seed OLS slope for each of the 15
seeds (4 points each: J in {0,15,30,60}), giving 15 independent per-seed
slope estimates (independent because different seeds use independent random
draws). Then test whether the across-seed distribution of these 15 slopes
has a mean different from zero via a one-sample t-test and Wilcoxon
signed-rank test, treating the 15 SEEDS (not the 60 points) as the sampling
units. This is the standard fix for repeated-measures/clustered inference
and does not require assuming a parametric correlation structure.
"""
import numpy as np
from scipy import stats as sp_stats
import audit_common as ac
from scientific_audit_E_F import run_collinearity_once_local, EXTRAPOLATION_DEVICE

N_SEEDS_COLLIN = 15
jitter_levels = [0, 15, 30, 60]

print("=" * 78)
print("STAGE F CORRECTED: per-seed slope regression (repeated-measures fix)")
print("=" * 78)

# Re-collect the same raw data as before, organized per-seed instead of pooled
per_seed_data = {name: {seed: [] for seed in range(2000, 2000 + N_SEEDS_COLLIN)}
                  for name in EXTRAPOLATION_DEVICE}
for J in jitter_levels:
    for seed in range(2000, 2000 + N_SEEDS_COLLIN):
        r = run_collinearity_once_local(seed, J)
        for name in EXTRAPOLATION_DEVICE:
            per_seed_data[name][seed].append((J, r[name]))

for name, by_seed in per_seed_data.items():
    short = name.split(" (")[0]
    per_seed_slopes = []
    per_seed_intercepts = []
    for seed, points in by_seed.items():
        J_arr = np.array([p[0] for p in points], dtype=float)
        pct_arr = np.array([p[1] for p in points])
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(J_arr, pct_arr)
        per_seed_slopes.append(slope)
        per_seed_intercepts.append(intercept)
    per_seed_slopes = np.array(per_seed_slopes)

    mean_slope = per_seed_slopes.mean()
    median_slope = np.median(per_seed_slopes)
    std_slope = per_seed_slopes.std(ddof=1)
    se_slope = std_slope / np.sqrt(N_SEEDS_COLLIN)
    tcrit = sp_stats.t.ppf(0.975, df=N_SEEDS_COLLIN - 1)
    ci_lo, ci_hi = mean_slope - tcrit * se_slope, mean_slope + tcrit * se_slope
    t_stat, p_val = sp_stats.ttest_1samp(per_seed_slopes, 0.0)
    w_stat, w_p = sp_stats.wilcoxon(per_seed_slopes, zero_method="wilcox")
    n_positive = int((per_seed_slopes > 0).sum())

    print(f"\n--- {short} ---")
    print(f"  15 per-seed slopes (percentage-points per us of jitter):")
    print(f"    {np.array2string(per_seed_slopes, precision=4, separator=', ')}")
    print(f"  mean={mean_slope:.4f}, median={median_slope:.4f}, std={std_slope:.4f}")
    print(f"  95% CI on mean per-seed slope: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  One-sample t-test (H0: mean slope=0), df={N_SEEDS_COLLIN-1}: "
          f"t={t_stat:.3f}, p={p_val:.4f} "
          f"({'SIGNIFICANT' if p_val < 0.05 else 'not significant'})")
    print(f"  Wilcoxon signed-rank on per-seed slopes: W={w_stat:.3f}, p={w_p:.4f} "
          f"({'SIGNIFICANT' if w_p < 0.05 else 'not significant'})")
    print(f"  Positive-slope seeds: {n_positive}/{N_SEEDS_COLLIN}")

print("\n" + "=" * 78)
print("COMPARISON: naive pooled-OLS (WRONG, i.i.d. assumption violated) vs.")
print("corrected per-seed-slope test (treats seeds as the true sampling unit)")
print("=" * 78)
print("Naive pooled OLS (from original Stage F, reported in manuscript v10):")
print("  osprey:     beta1=0.0609, p=0.470")
print("  sherbrooke: beta1=0.1690, p=0.0027  <-- this specific p-value is suspect")
print("  (both computed treating 60 correlated points as 60 independent draws)")
