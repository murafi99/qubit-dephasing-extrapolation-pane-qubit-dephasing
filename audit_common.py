import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats

DEVICES = {
    "ibm_torino (Heron r1, 2024-05)": dict(
        T1_us=167.73, T2_us=130.19, sx_err=3.15e-4, cz_err=4.56e-3, ro_err=1.95e-2
    ),
    "ibm_fez (Heron, arXiv:2410.00916 snapshot)": dict(
        T1_us=136.52, T2_us=78.58, sx_err=6.78e-3, cz_err=5.0e-3, ro_err=1.5e-2
    ),
    "ibm_marrakesh (Heron r2, arXiv:2504.15698 snapshot)": dict(
        T1_us=188.55, T2_us=113.97, sx_err=2.532e-4, cz_err=2.227e-3, ro_err=1.465e-2
    ),
}

EXTRAPOLATION_DEVICE = {
    "ibm_osprey (Osprey r1, held-out, LOW-T1 side)": dict(
        T1_us=88.35, T2_us=58.73, sx_err=6.256e-4, cz_err=2.155e-2, ro_err=4.91e-2
    ),
    "ibm_sherbrooke (Eagle r3, held-out, HIGH-T1 side)": dict(
        T1_us=262.69, T2_us=176.67, sx_err=2.411e-4, cz_err=7.571e-3, ro_err=1.350e-2
    ),
}

def thermal_photon_floor(base_T2_us, n_bar=0.006, chi_over_kappa=0.3):
    """
    Constant dephasing-rate floor from residual thermal-photon shot noise in the
    readout resonator (AC-Stark-induced dephasing). n_bar=0.006 is the thermal
    photon occupation reported by Gambetta-group-style photon-shot-noise
    spectroscopy work (e.g. Sears et al./Ganjam et al.-style resonator thermal
    photon measurements at ~30-50 mK stages; representative order-of-magnitude
    value used broadly in circuit QED dephasing budgets). Returns an additive
    dephasing-rate contribution (1/us) applied uniformly (non-fluctuating floor).
    """
    gamma_phi_floor = chi_over_kappa * n_bar / base_T2_us  # heuristic small floor term
    return gamma_phi_floor

DT_MIN = 10
HOURS = 72
N_STEPS = int(HOURS * 60 / DT_MIN)
WIN = 6

def run_full_pipeline_once(seed):
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
            phi_floor = thermal_photon_floor(cal["T2_us"])
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

    results_local = {}
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

        results_local[name] = dict(naive_mse=naive_mse, pi_mse=pi_mse,
                                    pct_change=(naive_mse - pi_mse) / naive_mse * 100)
    return results_local
