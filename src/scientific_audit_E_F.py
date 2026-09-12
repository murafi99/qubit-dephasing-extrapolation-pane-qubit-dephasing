"""
Scientific audit, Stage E & F.

E. Rerun the in-distribution interpolation experiment (manuscript Section
   6.5 / Figure 5) across 20 independent seeds instead of one, at the
   largest training-set size used in the manuscript (n=149, the full pool),
   reporting mean/CI rather than a single point estimate.

F. Formal trend regression for the Section 7 collinearity diagnostic:
   fit Delta_MSE(%) = beta0 + beta1 * J (jitter level in us) by OLS across
   the per-seed data pooled over all four jitter levels, report beta1, its
   standard error, 95% CI, and p-value, for both held-out devices.
"""
import numpy as np
from scipy import stats as sp_stats
import audit_common as ac
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DEVICES = ac.DEVICES
EXTRAPOLATION_DEVICE = ac.EXTRAPOLATION_DEVICE
N_STEPS = ac.N_STEPS
WIN = ac.WIN

# ---------------------------------------------------------------------------
# E. Multi-seed interpolation (in-distribution) experiment
# ---------------------------------------------------------------------------
print("=" * 78)
print("E. MULTI-SEED INTERPOLATION EXPERIMENT (20 seeds, full training pool)")
print("=" * 78)


def run_interpolation_once(seed):
    """Builds telemetry + training pool exactly as in the manuscript's
    Section 6.5 in-distribution test, using the SAME telemetry-generation
    logic as run_full_pipeline_once (verified identical), then reports
    naive vs. PI MSE on a held-out IN-DISTRIBUTION test split (not the
    cross-generation extrapolation devices)."""
    local_rng = np.random.default_rng(seed)

    def _tls(n_steps, base_T2_us, rate_per_hour=0.15, dt_min=10,
              duration_range_min=(30, 240), strength_range=(0.15, 0.6)):
        trace = np.ones(n_steps)
        total_hours = n_steps * dt_min / 60.0
        n_events = local_rng.poisson(rate_per_hour * total_hours)
        for _ in range(n_events):
            start = local_rng.integers(0, n_steps)
            dur_steps = int(local_rng.uniform(*duration_range_min) / dt_min)
            strength = local_rng.uniform(*strength_range)
            end = min(n_steps, start + dur_steps)
            trace[start:end] *= (1 - strength)
        return base_T2_us * trace

    def _flux(n_steps, dt_min=10, alpha=1.0, sigma_frac=0.06):
        white = local_rng.normal(0, 1, n_steps)
        freqs = np.fft.rfftfreq(n_steps, d=1.0)
        freqs[0] = freqs[1]
        shaping = 1.0 / (freqs ** (alpha / 2.0))
        spectrum = np.fft.rfft(white) * shaping
        colored = np.fft.irfft(spectrum, n=n_steps)
        return colored / np.std(colored) * sigma_frac

    def _qp(n_steps, base_T1_us, rate_per_hour=0.08, dt_min=10,
            recovery_range_min=(1, 15), suppression_range=(0.3, 0.85)):
        trace = np.ones(n_steps)
        total_hours = n_steps * dt_min / 60.0
        n_events = local_rng.poisson(rate_per_hour * total_hours)
        for _ in range(n_events):
            start = local_rng.integers(0, n_steps)
            dur_steps = max(1, int(local_rng.uniform(*recovery_range_min) / dt_min))
            suppression = local_rng.uniform(*suppression_range)
            end = min(n_steps, start + dur_steps)
            recovery_curve = np.exp(-np.linspace(0, 4, end - start))
            trace[start:end] *= (1 - suppression * recovery_curve)
        return base_T1_us * trace

    def _telemetry(devices_dict):
        out = {}
        for name, cal in devices_dict.items():
            T2_tls = _tls(N_STEPS, cal["T2_us"])
            flux_pert = _flux(N_STEPS)
            T1_qp = _qp(N_STEPS, cal["T1_us"])
            phi_floor = ac.thermal_photon_floor(cal["T2_us"])
            inv_T2 = 1.0 / T2_tls
            inv_T2 = inv_T2 * (1 + flux_pert) + phi_floor
            T2_eff = np.minimum(1.0 / inv_T2, 2 * T1_qp)
            T1_meas = T1_qp * (1 + local_rng.normal(0, 0.03, N_STEPS))
            T2_meas = T2_eff * (1 + local_rng.normal(0, 0.03, N_STEPS))
            out[name] = dict(T1=T1_meas, T2=T2_meas, cal=cal)
        return out

    telem = _telemetry(DEVICES)
    rows = []
    for name, d in telem.items():
        T1l, T2l = d["T1"], d["T2"]
        gamma_true_l = 1.0 / T2l - 1.0 / (2 * T1l)
        for i in range(0, N_STEPS - WIN, WIN):
            t1w, t2w = T1l[i:i + WIN], T2l[i:i + WIN]
            t1n = t1w.mean() + local_rng.normal(0, t1w.mean() * 0.08)
            t2n = t2w.mean() + local_rng.normal(0, t2w.mean() * 0.08)
            rows.append((t1n, t2n, gamma_true_l[i:i + WIN].mean()))
    arr = np.array(rows)
    X_reg, y_reg = arr[:, :2], arr[:, 2]

    from sklearn.model_selection import train_test_split
    X_pool, X_test, y_pool, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=7)

    sc = StandardScaler().fit(X_pool)
    naive_m = Ridge(alpha=1.0).fit(sc.transform(X_pool), y_pool)
    naive_mse = np.mean((naive_m.predict(sc.transform(X_test)) - y_test) ** 2)

    base_pool = 1.0 / (2 * X_pool[:, 0])
    resid_pool = y_pool - base_pool
    pi_sc = StandardScaler().fit(X_pool)
    pi_m = Ridge(alpha=1.0).fit(pi_sc.transform(X_pool), resid_pool)
    base_test = 1.0 / (2 * X_test[:, 0])
    pi_pred = base_test + pi_m.predict(pi_sc.transform(X_test))
    pi_mse = np.mean((pi_pred - y_test) ** 2)

    return naive_mse, pi_mse


N_SEEDS_INTERP = 20
naive_list, pi_list, pct_list = [], [], []
for seed in range(7000, 7000 + N_SEEDS_INTERP):
    nm, pm = run_interpolation_once(seed)
    naive_list.append(nm)
    pi_list.append(pm)
    pct_list.append((nm - pm) / nm * 100)

naive_arr, pi_arr, pct_arr = np.array(naive_list), np.array(pi_list), np.array(pct_list)
t_stat, p_val = sp_stats.ttest_1samp(pct_arr, 0.0)
se = pct_arr.std(ddof=1) / np.sqrt(N_SEEDS_INTERP)
tcrit = sp_stats.t.ppf(0.975, df=N_SEEDS_INTERP - 1)
ci_lo, ci_hi = pct_arr.mean() - tcrit * se, pct_arr.mean() + tcrit * se
n_pi_wins = int((pct_arr > 0).sum())
w_stat, w_p = sp_stats.wilcoxon(pi_arr - naive_arr, zero_method="wilcox")

print(f"In-distribution (interpolation) test, full training pool, N={N_SEEDS_INTERP} seeds:")
print(f"  naive MSE: mean={naive_arr.mean():.4e}, std={naive_arr.std(ddof=1):.4e}")
print(f"  PI MSE:    mean={pi_arr.mean():.4e}, std={pi_arr.std(ddof=1):.4e}")
print(f"  %% change: mean={pct_arr.mean():+.2f}%, std={pct_arr.std(ddof=1):.2f}%")
print(f"  95% CI on mean %% change: [{ci_lo:+.2f}%, {ci_hi:+.2f}%]")
print(f"  t({N_SEEDS_INTERP-1})={t_stat:.3f}, p={p_val:.4f} ({'significant' if p_val<0.05 else 'NOT significant'})")
print(f"  PI wins: {n_pi_wins}/{N_SEEDS_INTERP}")
print(f"  Wilcoxon signed-rank: W={w_stat:.3f}, p={w_p:.4f}")

# ---------------------------------------------------------------------------
# F. Formal trend regression for the collinearity jitter sweep
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("F. FORMAL TREND REGRESSION: Delta_MSE(%%) ~ beta0 + beta1 * jitter_std")
print("=" * 78)


def run_collinearity_once_local(seed, decorr_std_us):
    local_rng = np.random.default_rng(seed)

    def _tls(n_steps, base_T2_us, rate_per_hour=0.15, dt_min=10,
             duration_range_min=(30, 240), strength_range=(0.15, 0.6)):
        trace = np.ones(n_steps)
        total_hours = n_steps * dt_min / 60.0
        n_events = local_rng.poisson(rate_per_hour * total_hours)
        for _ in range(n_events):
            start = local_rng.integers(0, n_steps)
            dur_steps = int(local_rng.uniform(*duration_range_min) / dt_min)
            strength = local_rng.uniform(*strength_range)
            end = min(n_steps, start + dur_steps)
            trace[start:end] *= (1 - strength)
        return base_T2_us * trace

    def _flux(n_steps, dt_min=10, alpha=1.0, sigma_frac=0.06):
        white = local_rng.normal(0, 1, n_steps)
        freqs = np.fft.rfftfreq(n_steps, d=1.0)
        freqs[0] = freqs[1]
        shaping = 1.0 / (freqs ** (alpha / 2.0))
        spectrum = np.fft.rfft(white) * shaping
        colored = np.fft.irfft(spectrum, n=n_steps)
        return colored / np.std(colored) * sigma_frac

    def _qp(n_steps, base_T1_us, rate_per_hour=0.08, dt_min=10,
            recovery_range_min=(1, 15), suppression_range=(0.3, 0.85)):
        trace = np.ones(n_steps)
        total_hours = n_steps * dt_min / 60.0
        n_events = local_rng.poisson(rate_per_hour * total_hours)
        for _ in range(n_events):
            start = local_rng.integers(0, n_steps)
            dur_steps = max(1, int(local_rng.uniform(*recovery_range_min) / dt_min))
            suppression = local_rng.uniform(*suppression_range)
            end = min(n_steps, start + dur_steps)
            recovery_curve = np.exp(-np.linspace(0, 4, end - start))
            trace[start:end] *= (1 - suppression * recovery_curve)
        return base_T1_us * trace

    def _telemetry(devices_dict):
        out = {}
        for name, cal in devices_dict.items():
            T2_tls = _tls(N_STEPS, cal["T2_us"])
            flux_pert = _flux(N_STEPS)
            T1_qp = _qp(N_STEPS, cal["T1_us"])
            phi_floor = ac.thermal_photon_floor(cal["T2_us"])
            inv_T2 = 1.0 / T2_tls
            inv_T2 = inv_T2 * (1 + flux_pert) + phi_floor
            T2_eff = np.minimum(1.0 / inv_T2, 2 * T1_qp)
            T1_meas = T1_qp * (1 + local_rng.normal(0, 0.03, N_STEPS))
            T2_meas = T2_eff * (1 + local_rng.normal(0, 0.03, N_STEPS))
            out[name] = dict(T1=T1_meas, T2=T2_meas, cal=cal)
        return out

    telem = _telemetry(DEVICES)
    extrap = _telemetry(EXTRAPOLATION_DEVICE)

    rows_local = []
    for name, d in telem.items():
        T1l, T2l = d["T1"], d["T2"]
        gamma_true_l = 1.0 / T2l - 1.0 / (2 * T1l)
        for i in range(0, N_STEPS - WIN, WIN):
            t1w, t2w = T1l[i:i + WIN], T2l[i:i + WIN]
            t1n = t1w.mean() + local_rng.normal(0, t1w.mean() * 0.08)
            t2n = t2w.mean() + local_rng.normal(0, t2w.mean() * 0.08)
            rows_local.append((t1n, t2n, gamma_true_l[i:i + WIN].mean()))
    arr_local = np.array(rows_local)
    X_pool_l, y_pool_l = arr_local[:, :2], arr_local[:, 2]
    X_pool_l = X_pool_l.copy()
    X_pool_l[:, 1] = X_pool_l[:, 1] + local_rng.normal(0, decorr_std_us, len(X_pool_l))

    out_pct = {}
    for name, d in extrap.items():
        T1l, T2l = d["T1"], d["T2"]
        gamma_true_l = 1.0 / T2l - 1.0 / (2 * T1l)
        rows_e = []
        for i in range(0, N_STEPS - WIN, WIN):
            t1w, t2w = T1l[i:i + WIN], T2l[i:i + WIN]
            t1n = t1w.mean() + local_rng.normal(0, t1w.mean() * 0.08)
            t2n = t2w.mean() + local_rng.normal(0, t2w.mean() * 0.08)
            rows_e.append((t1n, t2n, gamma_true_l[i:i + WIN].mean()))
        arr_e = np.array(rows_e)
        X_e, y_e = arr_e[:, :2], arr_e[:, 2]

        sc = StandardScaler().fit(X_pool_l)
        naive_m = Ridge(alpha=1.0).fit(sc.transform(X_pool_l), y_pool_l)
        naive_mse = np.mean((naive_m.predict(sc.transform(X_e)) - y_e) ** 2)

        base_pool_l = 1.0 / (2 * X_pool_l[:, 0])
        resid_pool_l = y_pool_l - base_pool_l
        pi_sc = StandardScaler().fit(X_pool_l)
        pi_m = Ridge(alpha=1.0).fit(pi_sc.transform(X_pool_l), resid_pool_l)
        base_e = 1.0 / (2 * X_e[:, 0])
        pi_pred = base_e + pi_m.predict(pi_sc.transform(X_e))
        pi_mse = np.mean((pi_pred - y_e) ** 2)
        out_pct[name] = (naive_mse - pi_mse) / naive_mse * 100
    return out_pct


N_SEEDS_COLLIN = 15
jitter_levels = [0, 15, 30, 60]
trend_data = {name: {"J": [], "pct": []} for name in EXTRAPOLATION_DEVICE}
for J in jitter_levels:
    for seed in range(2000, 2000 + N_SEEDS_COLLIN):
        r = run_collinearity_once_local(seed, J)
        for name in EXTRAPOLATION_DEVICE:
            trend_data[name]["J"].append(J)
            trend_data[name]["pct"].append(r[name])

for name, d in trend_data.items():
    J_arr = np.array(d["J"], dtype=float)
    pct_arr = np.array(d["pct"])
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(J_arr, pct_arr)
    n = len(J_arr)
    tcrit = sp_stats.t.ppf(0.975, df=n - 2)
    ci_lo, ci_hi = slope - tcrit * std_err, slope + tcrit * std_err
    short = name.split(" (")[0]
    print(f"\n{short}: Delta_MSE(%%) = {intercept:.3f} + {slope:.4f} * J  "
          f"(n={n} pooled seed-condition points)")
    print(f"  beta1 (slope) = {slope:.4f} percentage-points per us of jitter")
    print(f"  95% CI on beta1: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  p-value (H0: beta1=0): {p_value:.4f} "
          f"({'SIGNIFICANT TREND' if p_value < 0.05 else 'no significant linear trend'})")
    print(f"  R^2 = {r_value**2:.4f}")
