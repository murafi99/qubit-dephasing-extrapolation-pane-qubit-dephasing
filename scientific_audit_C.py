"""
Scientific audit, Stage C: strengthen the multi-seed statistical analysis.

Uses the SAME 20-seed pipeline already in the manuscript (imported verbatim
from noise_simulation.py via audit_common.py, not reimplemented), captures
the raw PAIRED per-seed MSE_PI and MSE_naive values, and reports:
  - Delta_i = MSE_PI,i - MSE_naive,i  (negative = PI better)
  - mean, median, std of Delta
  - 95% bootstrap CI (percentile method, documented resample count)
  - proportion of seeds where PI wins
  - paired Wilcoxon signed-rank test (nonparametric alternative to the
    manuscript's existing paired t-test on percent-change)
"""
import numpy as np
from scipy import stats as sp_stats
import audit_common as ac

N_SEEDS = 20
N_BOOTSTRAP = 10000

print("=" * 78)
print(f"STAGE C: Paired per-seed analysis, N={N_SEEDS} seeds "
      f"(reusing the manuscript's exact run_full_pipeline_once)")
print("=" * 78)

paired_data = {name: {"naive_mse": [], "pi_mse": []} for name in ac.EXTRAPOLATION_DEVICE}
for seed in range(1000, 1000 + N_SEEDS):
    r = ac.run_full_pipeline_once(seed)
    for name in ac.EXTRAPOLATION_DEVICE:
        paired_data[name]["naive_mse"].append(r[name]["naive_mse"])
        paired_data[name]["pi_mse"].append(r[name]["pi_mse"])

rng_boot = np.random.default_rng(2026)

for name, d in paired_data.items():
    naive = np.array(d["naive_mse"])
    pi = np.array(d["pi_mse"])
    delta = pi - naive  # negative = PI better (lower MSE)
    pct_change = (naive - pi) / naive * 100  # positive = PI better, matches manuscript convention

    mean_d, median_d, std_d = delta.mean(), np.median(delta), delta.std(ddof=1)
    n_pi_wins = int((delta < 0).sum())

    # 95% bootstrap CI on mean(Delta), percentile method
    boot_means = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng_boot.integers(0, N_SEEDS, N_SEEDS)
        boot_means[b] = delta[idx].mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # Same bootstrap on percent-change scale for direct comparability with
    # the manuscript's existing t-test numbers
    boot_pct = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng_boot.integers(0, N_SEEDS, N_SEEDS)
        boot_pct[b] = pct_change[idx].mean()
    ci_pct_lo, ci_pct_hi = np.percentile(boot_pct, [2.5, 97.5])

    # Paired Wilcoxon signed-rank test on Delta vs 0
    # (guard: wilcoxon requires no exact zeros ideally; check)
    n_zero = int((delta == 0).sum())
    try:
        wstat, wp = sp_stats.wilcoxon(delta, zero_method="wilcox")
        wilcoxon_valid = True
    except ValueError as e:
        wstat, wp = np.nan, np.nan
        wilcoxon_valid = False
        print(f"  Wilcoxon failed for {name}: {e}")

    # Existing paired t-test (reproduced here for direct side-by-side, not
    # replacing it -- matches manuscript Table 3 approach)
    t_stat, t_p = sp_stats.ttest_1samp(pct_change, 0.0)

    print(f"\n--- {name} ---")
    print(f"  Delta_i = MSE_PI - MSE_naive (n={N_SEEDS}):")
    print(f"    mean(Delta)   = {mean_d:.4e}")
    print(f"    median(Delta) = {median_d:.4e}")
    print(f"    std(Delta)    = {std_d:.4e}")
    print(f"    95% bootstrap CI on mean(Delta) ({N_BOOTSTRAP} resamples, percentile method): "
          f"[{ci_lo:.4e}, {ci_hi:.4e}]")
    print(f"    PI wins (Delta<0): {n_pi_wins}/{N_SEEDS} ({100*n_pi_wins/N_SEEDS:.0f}%)")
    print(f"    zero-difference seeds: {n_zero}")
    if wilcoxon_valid:
        print(f"    Wilcoxon signed-rank: W={wstat:.3f}, p={wp:.4f}")
    print(f"  Percent-change scale (for direct comparison to manuscript Table 3):")
    print(f"    mean%={pct_change.mean():+.1f}%, existing t-test: t({N_SEEDS-1})={t_stat:.3f}, p={t_p:.4f}")
    print(f"    95% bootstrap CI on mean%: [{ci_pct_lo:+.1f}%, {ci_pct_hi:+.1f}%]")
    sig_delta_ci = "excludes 0 (significant)" if (ci_lo > 0 or ci_hi < 0) else "includes 0 (not significant)"
    sig_wilcoxon = f"p={wp:.4f} ({'significant' if wilcoxon_valid and wp < 0.05 else 'not significant'})" if wilcoxon_valid else "N/A"
    print(f"    Bootstrap CI on Delta: {sig_delta_ci}")
    print(f"    Wilcoxon result: {sig_wilcoxon}")

    # Save raw arrays to disk for reproducibility / re-use in later stages
    np.savez(f"audit_paired_data_{name.split(' (')[0]}.npz",
              naive_mse=naive, pi_mse=pi, delta=delta, pct_change=pct_change)

print("\nRaw paired per-seed arrays saved to audit_paired_data_*.npz for "
      "independent re-verification.")
