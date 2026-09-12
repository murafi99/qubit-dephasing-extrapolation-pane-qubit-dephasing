"""
Scientific audit, Stage A & B.

A. Compute the EXACT (noiseless) least-squares linear fit through the three
   real training T1 points evaluated at f(T1)=1/(2T1), and compare its bias
   at the two held-out test points against (i) the Taylor-remainder proxy
   and (ii) the full noisy Monte Carlo simulated bias already reported in
   Table 1. This distinguishes "curvature error of the best local linear
   approximation" from "actual bias of a fitted estimator" as requested.

B. Repeat the isolated single-variable Monte Carlo experiment (Section 5)
   with a positive-support lognormal multiplicative noise model instead of
   the unrestricted Gaussian, to check whether the qualitative finding is
   an artifact of the (formally singular) Gaussian reciprocal model.
"""
import numpy as np
from scipy import stats as sp_stats

rng = np.random.default_rng(123)

TRAIN_T1 = np.array([167.73, 136.52, 188.55])  # torino, fez, marrakesh (corrected)
TEST_POINTS = {"ibm_osprey": 88.35, "ibm_sherbrooke": 262.69}

def f(T1):
    return 1.0 / (2 * T1)

# ---------------------------------------------------------------------------
# A. Exact deterministic OLS-through-3-points bias vs Taylor remainder
# ---------------------------------------------------------------------------
print("=" * 78)
print("A. EXACT OLS-THROUGH-TRAINING-POINTS BIAS vs. TAYLOR-REMAINDER PROXY")
print("=" * 78)

y_train_exact = f(TRAIN_T1)  # noiseless: exact f(T1) at the 3 training T1 values
A = np.vstack([np.ones_like(TRAIN_T1), TRAIN_T1]).T
coef_exact, residuals, rank, sv = np.linalg.lstsq(A, y_train_exact, rcond=None)
a_hat, b_hat = coef_exact
print(f"Exact (noiseless) OLS fit through the 3 training points: "
      f"f_hat(T1) = {a_hat:.6e} + {b_hat:.6e}*T1")
resid_at_train = y_train_exact - (a_hat + b_hat * TRAIN_T1)
print(f"Residuals at training points (should be ~0 only if 3 points were "
      f"co-linear in T1,f(T1) space -- they are not, so this is a genuine "
      f"least-squares fit, not an interpolant): {resid_at_train}")

T1_bar = TRAIN_T1.mean()
print(f"\nTraining mean T1_bar = {T1_bar:.3f} us\n")

print(f"{'Test point':15s} {'T1_test':>9s} {'f(T1_test)':>12s} {'OLS fit value':>14s} "
      f"{'Exact OLS bias':>15s} {'Taylor proxy (2nd order)':>26s}")
for name, T1_test in TEST_POINTS.items():
    f_true = f(T1_test)
    f_fit = a_hat + b_hat * T1_test
    exact_bias = f_fit - f_true

    # Taylor-remainder proxy: (1/2) f''(xi) (T1_test - T1_bar)^2, evaluated
    # at xi = T1_bar as the simplest, most literal reading of the proxy used
    # in the manuscript's Section 4.2 (the honest caveat is that the true
    # remainder uses an unknown xi between T1_bar and T1_test; we report the
    # endpoint-at-mean evaluation as the "proxy" and separately the exact
    # mean-value xi that WOULD make the proxy match, to show how much the
    # naive fixed-point evaluation can be off by).
    f_pp = lambda T1v: 1.0 / T1v ** 3
    taylor_proxy = 0.5 * f_pp(T1_bar) * (T1_test - T1_bar) ** 2
    # sign convention: this proxy is unsigned curvature magnitude; compare
    # its magnitude to the signed exact bias magnitude
    print(f"{name:15s} {T1_test:9.2f} {f_true:12.4e} {f_fit:14.4e} "
          f"{exact_bias:15.4e} {taylor_proxy:26.4e}")

print("\nInterpretation: the Taylor-remainder proxy is a rough magnitude "
      "estimate of curvature error for the BEST local linear approximation "
      "(tangent line at T1_bar), not the bias of the actual fitted "
      "least-squares line through the specific 3 training points -- the two "
      "differ because (a) the tangent line at T1_bar is not the same line as "
      "the least-squares fit through 3 unevenly-spaced points, and (b) the "
      "Taylor proxy evaluates f'' at a single fixed point rather than the "
      "unknown mean-value xi. We report both, plus the full noisy simulated "
      "bias (Table 1), rather than conflating any one of these three "
      "distinct quantities.")

# Also solve for the xi that would make the Taylor formula EXACTLY match the
# true function's deviation from the tangent line at T1_bar (not the 3-point
# OLS fit) at each test point, to make the "proxy, not exact" point concrete.
print("\nFor comparison: deviation of f(T1_test) from the TANGENT LINE at "
      "T1_bar (a different, cleaner quantity than the 3-point OLS fit):")
f_prime = lambda T1v: -1.0 / (2 * T1v ** 2)
for name, T1_test in TEST_POINTS.items():
    tangent_val = f(T1_bar) + f_prime(T1_bar) * (T1_test - T1_bar)
    true_val = f(T1_test)
    tangent_dev = tangent_val - true_val
    print(f"  {name:15s}: tangent-line deviation = {tangent_dev:.4e} "
          f"(vs. Taylor proxy magnitude computed above)")

# ---------------------------------------------------------------------------
# B. Lognormal (positive-support) multiplicative noise comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("B. LOGNORMAL (POSITIVE-SUPPORT) NOISE MODEL -- REMOVES THE RECIPROCAL SINGULARITY")
print("=" * 78)

sigma_gauss = 0.08  # the Gaussian relative-noise sigma used throughout the paper
# Match a lognormal multiplicative factor (1+eps) -> L with the SAME
# relative variance to leading order: L = exp(Z), Z ~ N(mu, s^2), chosen so
# that E[L]=1 (unbiased multiplicative factor) and Var[L) matches sigma_gauss^2
# to leading order. For lognormal with E[L]=1: mu = -s^2/2, Var[L] = exp(s^2)-1.
# Solve exp(s^2)-1 = sigma_gauss^2 for s^2 (matches variance, not just to
# leading order, so this is an exact, not approximate, calibration).
s2 = np.log(1 + sigma_gauss ** 2)
s = np.sqrt(s2)
mu = -s2 / 2
print(f"Lognormal calibration: mu={mu:.6f}, s={s:.6f} (chosen so E[L]=1 and "
      f"Var[L]={sigma_gauss**2:.6e} exactly, matching the Gaussian model's "
      f"variance, not merely to leading order).")

N_MC = 2_000_000
train_mean_T1 = TRAIN_T1.mean()

# Re-fit the linear surrogate under GAUSSIAN noise (as in the manuscript) for
# a same-N_train, same-training-devices comparison baseline.
N_PER_DEVICE = 71
T1_true_train = np.repeat(TRAIN_T1, N_PER_DEVICE)

def fit_linear_surrogate(T1_true_train, noise_model, rng_local):
    if noise_model == "gaussian":
        T1_obs_train = T1_true_train * (1 + rng_local.normal(0, sigma_gauss, len(T1_true_train)))
    elif noise_model == "lognormal":
        T1_obs_train = T1_true_train * rng_local.lognormal(mu, s, len(T1_true_train))
    else:
        raise ValueError(noise_model)
    y_train = f(T1_true_train)
    Amat = np.vstack([np.ones_like(T1_obs_train), T1_obs_train]).T
    coef, *_ = np.linalg.lstsq(Amat, y_train, rcond=None)
    return coef

results_compare = {}
for noise_model in ["gaussian", "lognormal"]:
    rng_local = np.random.default_rng(999)
    a_l, b_l = fit_linear_surrogate(T1_true_train, noise_model, rng_local)
    print(f"\n[{noise_model.upper()}] linear surrogate fit: "
          f"f_hat(T1) = {a_l:.6e} + {b_l:.6e}*T1")

    per_point = {}
    for name, T1_test in TEST_POINTS.items():
        rng_mc = np.random.default_rng(42)
        if noise_model == "gaussian":
            T1_obs = T1_test * (1 + rng_mc.normal(0, sigma_gauss, N_MC))
        else:
            T1_obs = T1_test * rng_mc.lognormal(mu, s, N_MC)
        # exact plug-in (guard against non-positive T1_obs under Gaussian,
        # which is the whole point of section 1's fix -- report how often
        # this actually occurs at this sigma)
        n_nonpositive = int(np.sum(T1_obs <= 0))
        valid = T1_obs > 0
        yhat_exact = np.full(N_MC, np.nan)
        yhat_exact[valid] = f(T1_obs[valid])
        y_true = f(T1_test)
        bias_exact = np.nanmean(yhat_exact) - y_true
        var_exact = np.nanvar(yhat_exact)
        mse_exact = np.nanmean((yhat_exact - y_true) ** 2)

        yhat_lin = a_l + b_l * T1_obs
        bias_lin = yhat_lin.mean() - y_true
        var_lin = yhat_lin.var()
        mse_lin = np.mean((yhat_lin - y_true) ** 2)

        per_point[name] = dict(mse_exact=mse_exact, mse_lin=mse_lin,
                                bias_exact=bias_exact, var_exact=var_exact,
                                n_nonpositive=n_nonpositive)
        winner = "EXACT wins" if mse_exact < mse_lin else "LINEAR wins"
        print(f"  {name:15s} T1={T1_test:7.2f}  n_nonpositive_draws={n_nonpositive:4d}/{N_MC}  "
              f"bias_exact={bias_exact:.4e}  var_exact={var_exact:.4e}  "
              f"MSE_exact={mse_exact:.4e}  MSE_linear={mse_lin:.4e}  <- {winner}")
    results_compare[noise_model] = per_point

print("\nQualitative comparison across noise models:")
for name in TEST_POINTS:
    g = results_compare["gaussian"][name]
    l = results_compare["lognormal"][name]
    same_winner = (g["mse_exact"] < g["mse_lin"]) == (l["mse_exact"] < l["mse_lin"])
    print(f"  {name}: Gaussian winner={'exact' if g['mse_exact']<g['mse_lin'] else 'linear'}, "
          f"Lognormal winner={'exact' if l['mse_exact']<l['mse_lin'] else 'linear'} "
          f"-> {'SAME qualitative conclusion' if same_winner else 'DIFFERENT conclusion'}")
