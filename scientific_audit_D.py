"""
Scientific audit, Stage D: sensitivity analysis over the synthetic-noise
parameters. Sweeps each of the six principal illustrative parameters
identified in the manuscript's Limitations section, one at a time (holding
others at their manuscript-default value), 20 seeds per condition, and
reports PI vs. naive MSE, relative change, and cross-seed spread at both
held-out devices. Parameter ranges are chosen as a symmetric low/default/high
grid around each manuscript default, NOT selected to favor either model.
"""
import numpy as np
from scipy import stats as sp_stats
import audit_common as ac

DEVICES = ac.DEVICES
EXTRAPOLATION_DEVICE = ac.EXTRAPOLATION_DEVICE
N_STEPS = ac.N_STEPS
WIN = ac.WIN
thermal_photon_floor_default = ac.thermal_photon_floor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def run_pipeline_param(seed, tls_rate=0.15, qp_rate=0.08, flux_sigma=0.06,
                        thermal_nbar=0.006, chi_over_kappa=0.3,
                        meas_noise_std=0.03, feature_noise_std=0.08):
    """Same structure as run_full_pipeline_once, but with the six principal
    synthetic-noise parameters exposed as arguments for sensitivity sweeps."""
    local_rng = np.random.default_rng(seed)

    def _tls(n_steps, base_T2_us, rate_per_hour=tls_rate, dt_min=10,
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

    def _flux(n_steps, dt_min=10, alpha=1.0, sigma_frac=flux_sigma):
        white = local_rng.normal(0, 1, n_steps)
        freqs = np.fft.rfftfreq(n_steps, d=1.0)
        freqs[0] = freqs[1]
        shaping = 1.0 / (freqs ** (alpha / 2.0))
        spectrum = np.fft.rfft(white) * shaping
        colored = np.fft.irfft(spectrum, n=n_steps)
        return colored / np.std(colored) * sigma_frac

    def _qp(n_steps, base_T1_us, rate_per_hour=qp_rate, dt_min=10,
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

    def _thermal_floor(base_T2_us):
        return chi_over_kappa * thermal_nbar / base_T2_us

    def _telemetry(devices_dict):
        out = {}
        for name, cal in devices_dict.items():
            T2_tls = _tls(N_STEPS, cal["T2_us"])
            flux_pert = _flux(N_STEPS)
            T1_qp = _qp(N_STEPS, cal["T1_us"])
            phi_floor = _thermal_floor(cal["T2_us"])
            inv_T2 = 1.0 / T2_tls
            inv_T2 = inv_T2 * (1 + flux_pert) + phi_floor
            T2_eff = np.minimum(1.0 / inv_T2, 2 * T1_qp)
            T1_meas = T1_qp * (1 + local_rng.normal(0, meas_noise_std, N_STEPS))
            T2_meas = T2_eff * (1 + local_rng.normal(0, meas_noise_std, N_STEPS))
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
            t1n = t1w.mean() + local_rng.normal(0, t1w.mean() * feature_noise_std)
            t2n = t2w.mean() + local_rng.normal(0, t2w.mean() * feature_noise_std)
            rows_local.append((t1n, t2n, gamma_true_l[i:i + WIN].mean()))
    arr_local = np.array(rows_local)
    X_pool_l, y_pool_l = arr_local[:, :2], arr_local[:, 2]

    results_local = {}
    for name, d in extrap.items():
        T1l, T2l = d["T1"], d["T2"]
        gamma_true_l = 1.0 / T2l - 1.0 / (2 * T1l)
        rows_e = []
        for i in range(0, N_STEPS - WIN, WIN):
            t1w, t2w = T1l[i:i + WIN], T2l[i:i + WIN]
            t1n = t1w.mean() + local_rng.normal(0, t1w.mean() * feature_noise_std)
            t2n = t2w.mean() + local_rng.normal(0, t2w.mean() * feature_noise_std)
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

        results_local[name] = dict(naive_mse=naive_mse, pi_mse=pi_mse,
                                    pct_change=(naive_mse - pi_mse) / naive_mse * 100)
    return results_local


DEFAULTS = dict(tls_rate=0.15, qp_rate=0.08, flux_sigma=0.06, thermal_nbar=0.006,
                 chi_over_kappa=0.3, meas_noise_std=0.03, feature_noise_std=0.08)

SWEEPS = {
    "tls_rate (events/hour)": [0.05, 0.15, 0.30],
    "qp_rate (events/hour)": [0.02, 0.08, 0.20],
    "flux_sigma (fractional)": [0.02, 0.06, 0.12],
    "thermal_nbar": [0.002, 0.006, 0.02],
    "chi_over_kappa": [0.1, 0.3, 0.6],
    "meas_noise_std (per-shot, 3pct default)": [0.01, 0.03, 0.06],
    "feature_noise_std (calibration-report, 8pct default)": [0.04, 0.08, 0.16],
}

N_SEEDS_SENS = 20
print("=" * 90)
print(f"STAGE D: SENSITIVITY ANALYSIS -- {N_SEEDS_SENS} seeds per condition, "
      f"one parameter varied at a time, others held at manuscript defaults")
print(f"Defaults: {DEFAULTS}")
print("=" * 90)

sensitivity_results = {}
for param_name, values in SWEEPS.items():
    key = param_name.split(" ")[0]
    print(f"\n--- Sweeping {param_name} ---")
    sensitivity_results[param_name] = {}
    for val in values:
        kwargs = dict(DEFAULTS)
        kwargs[key] = val
        per_device = {name: {"naive": [], "pi": [], "pct": []} for name in EXTRAPOLATION_DEVICE}
        for seed in range(5000, 5000 + N_SEEDS_SENS):
            r = run_pipeline_param(seed, **kwargs)
            for name in EXTRAPOLATION_DEVICE:
                per_device[name]["naive"].append(r[name]["naive_mse"])
                per_device[name]["pi"].append(r[name]["pi_mse"])
                per_device[name]["pct"].append(r[name]["pct_change"])
        sensitivity_results[param_name][val] = per_device
        for name, d in per_device.items():
            pct = np.array(d["pct"])
            short = name.split(" (")[0]
            t_stat, p_val = sp_stats.ttest_1samp(pct, 0.0)
            sig = "SIG" if p_val < 0.05 else "ns "
            print(f"  {key}={val:<8} {short:16s}  naive_MSE={np.mean(d['naive']):.3e}  "
                  f"PI_MSE={np.mean(d['pi']):.3e}  %change={pct.mean():+6.1f}%"
                  f" (std={pct.std(ddof=1):5.1f}, t={t_stat:6.2f}, p={p_val:.4f} [{sig}])")

# Summary: does the qualitative conclusion (PI wins at high-T1, not at low-T1)
# hold across the whole sensitivity grid, or does it flip anywhere?
print("\n" + "=" * 90)
print("SUMMARY: qualitative robustness of the main finding across the sensitivity grid")
print("=" * 90)
osprey_key = [k for k in EXTRAPOLATION_DEVICE if "osprey" in k][0]
sherb_key = [k for k in EXTRAPOLATION_DEVICE if "sherbrooke" in k][0]
for param_name, by_val in sensitivity_results.items():
    for val, per_device in by_val.items():
        o_pct = np.array(per_device[osprey_key]["pct"])
        s_pct = np.array(per_device[sherb_key]["pct"])
        o_t, o_p = sp_stats.ttest_1samp(o_pct, 0.0)
        s_t, s_p = sp_stats.ttest_1samp(s_pct, 0.0)
        o_sig = "significant" if o_p < 0.05 else "not significant"
        s_sig = "significant" if s_p < 0.05 else "not significant"
        flag = ""
        if o_p < 0.05:
            flag = "  <-- OSPREY BECAME SIGNIFICANT (differs from default finding)"
        if s_p >= 0.05:
            flag += "  <-- SHERBROOKE LOST SIGNIFICANCE (differs from default finding)"
        print(f"{param_name}={val}: osprey {o_sig} (mean={o_pct.mean():+.1f}%), "
              f"sherbrooke {s_sig} (mean={s_pct.mean():+.1f}%){flag}")
