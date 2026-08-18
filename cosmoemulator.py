"""
CosmoEmulator-NN
=================
A neural-network emulator that replaces expensive numerical integration of the
Friedmann equation when computing the Type-Ia supernova distance modulus mu(z)
for a CPL (Chevallier-Polanski-Linder) dark energy model:

    w(z) = w0 + wa * z / (1 + z)

    E(z)^2 = Om*(1+z)^3 + (1-Om)*(1+z)^(3*(1+w0+wa)) * exp(-3*wa*z/(1+z))

    D_C(z) = (c/H0) * Integral_0^z dz' / E(z')
    D_L(z) = (1+z) * D_C(z)
    mu(z)  = 5*log10(D_L[Mpc]) + 25

Why this matters
-----------------
Every likelihood evaluation inside an MCMC / nested-sampling run for dark-energy
parameter estimation requires re-integrating E(z) numerically for every trial
(Om, w0, wa). For chains of 10^5-10^6 steps this integration is the dominant
cost. This project trains a small feed-forward network to *emulate* mu(z; Om,
w0, wa) directly, and benchmarks the resulting speed-up + accuracy trade-off --
the same idea behind production tools like CosmoPower / emulators used in
Planck / DES / LSST-era analyses, just at a much smaller, reproducible scale.

Output
------
- results/emulator_accuracy.png   : held-out residuals & example Hubble curves
- results/speed_benchmark.png     : numerical integration vs emulator wall-time
- Printed summary statistics (RMS residual in mag, speed-up factor)
"""

import time
import numpy as np
from scipy.integrate import quad
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(42)
os.makedirs("results", exist_ok=True)

C_KM_S = 299792.458   # speed of light, km/s
H0 = 70.0             # km/s/Mpc, held fixed (degenerate with abs. mag in real SN analyses)

# ----------------------------------------------------------------------------
# 1. Physics: exact numerical solution (the "ground truth" / expensive branch)
# ----------------------------------------------------------------------------
def E_of_z(z, Om, w0, wa):
    de_term = (1 - Om) * (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    return np.sqrt(Om * (1 + z) ** 3 + de_term)


def mu_numerical(z, Om, w0, wa):
    """Distance modulus via direct numerical integration (the slow, exact path)."""
    integrand = lambda zp: 1.0 / E_of_z(zp, Om, w0, wa)
    Dc, _ = quad(integrand, 0.0, z, limit=200)
    Dc *= C_KM_S / H0
    Dl = (1 + z) * Dc
    return 5 * np.log10(Dl) + 25


# ----------------------------------------------------------------------------
# 2. Build a training set spanning the region of parameter space that matters
#    for real SN Ia cosmology fits (priors similar to Pantheon+-style analyses)
# ----------------------------------------------------------------------------
def sample_params(n):
    Om = rng.uniform(0.15, 0.45, n)
    w0 = rng.uniform(-1.6, -0.5, n)
    wa = rng.uniform(-1.5, 1.5, n)
    return Om, w0, wa


def build_dataset(n_models, z_grid):
    Om, w0, wa = sample_params(n_models)
    X, y = [], []
    for i in range(n_models):
        for z in z_grid:
            X.append([Om[i], w0[i], wa[i], z])
            y.append(mu_numerical(z, Om[i], w0[i], wa[i]))
    return np.array(X), np.array(y)


print("Building training data (numerically integrating the Friedmann equation)...")
z_grid = np.linspace(0.01, 2.3, 40)          # matches the observed SN redshift range
X_train_raw, y_train = build_dataset(n_models=1200, z_grid=z_grid)
X_test_raw, y_test = build_dataset(n_models=250, z_grid=z_grid)
print(f"  train set: {X_train_raw.shape[0]:,} (model, z) points from 1200 dark-energy models")
print(f"  test set:  {X_test_raw.shape[0]:,} (model, z) points from 250 UNSEEN models")

# ----------------------------------------------------------------------------
# 3. Train the emulator
# ----------------------------------------------------------------------------
scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

emulator = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),
    activation="tanh",
    solver="adam",
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=25,
    random_state=42,
)
print("\nTraining emulator (MLP, 64-64-32, tanh)...")
t0 = time.time()
emulator.fit(X_train, y_train)
train_time = time.time() - t0
print(f"  training time: {train_time:.1f}s over {emulator.n_iter_} iterations")

pred_test = emulator.predict(X_test)
resid = pred_test - y_test
rms = np.sqrt(np.mean(resid ** 2))
max_err = np.max(np.abs(resid))
print(f"\nHeld-out accuracy on 250 unseen dark-energy models:")
print(f"  RMS residual : {rms*1000:.2f} mmag")
print(f"  max |residual|: {max_err*1000:.2f} mmag")

# ----------------------------------------------------------------------------
# 4. Speed benchmark: numerical integration vs. trained emulator
# ----------------------------------------------------------------------------
n_bench = 3000  # representative of a short MCMC chain segment
bench_Om, bench_w0, bench_wa = sample_params(n_bench)
bench_z = rng.choice(z_grid, n_bench)

t0 = time.time()
for i in range(n_bench):
    _ = mu_numerical(bench_z[i], bench_Om[i], bench_w0[i], bench_wa[i])
t_numerical = time.time() - t0

X_bench = scaler.transform(np.column_stack([bench_Om, bench_w0, bench_wa, bench_z]))
t0 = time.time()
_ = emulator.predict(X_bench)
t_emulator = time.time() - t0

speedup = t_numerical / t_emulator
print(f"\nSpeed benchmark over {n_bench:,} likelihood-style evaluations:")
print(f"  numerical integration : {t_numerical:.3f}s")
print(f"  neural emulator        : {t_emulator:.4f}s")
print(f"  speed-up               : {speedup:,.0f}x")

# ----------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) example Hubble diagrams: exact vs emulated, for 3 unseen models
ax = axes[0]
example_idx = rng.choice(250, 3, replace=False)
for k, idx in enumerate(example_idx):
    Om_e, w0_e, wa_e = X_test_raw[idx * len(z_grid), 0], X_test_raw[idx * len(z_grid), 1], X_test_raw[idx * len(z_grid), 2]
    mask = np.isclose(X_test_raw[:, 0], Om_e) & np.isclose(X_test_raw[:, 1], w0_e) & np.isclose(X_test_raw[:, 2], wa_e)
    zz = X_test_raw[mask, 3]
    order = np.argsort(zz)
    ax.plot(zz[order], y_test[mask][order], "-", lw=2, alpha=0.8,
            label=f"exact  (Om={Om_e:.2f}, w0={w0_e:.2f}, wa={wa_e:.2f})" if k == 0 else None,
            color=f"C{k}")
    ax.plot(zz[order], pred_test[mask][order], "--", lw=1.5, color=f"C{k}")
ax.set_xlabel("redshift z")
ax.set_ylabel(r"distance modulus $\mu(z)$")
ax.set_title("Solid = numerical integration, dashed = emulator\n(3 unseen dark-energy models)")

# (b) residual histogram
ax = axes[1]
ax.hist(resid * 1000, bins=50, color="#4C72B0", alpha=0.85)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("emulator residual (mmag)")
ax.set_ylabel("count")
ax.set_title(f"Held-out residuals, RMS = {rms*1000:.2f} mmag\n(typical SN Ia measurement error ~ 100-150 mmag)")

plt.tight_layout()
plt.savefig("results/emulator_accuracy.png", dpi=150)
print("\nSaved results/emulator_accuracy.png")

fig2, ax = plt.subplots(figsize=(5.5, 4.5))
bars = ax.bar(["Numerical\nintegration", "Neural\nemulator"], [t_numerical, t_emulator],
              color=["#C44E52", "#55A868"])
ax.set_ylabel("wall-clock time (s)")
ax.set_title(f"{n_bench:,} likelihood evaluations\n{speedup:,.0f}x speed-up")
for b, v in zip(bars, [t_numerical, t_emulator]):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}s", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("results/speed_benchmark.png", dpi=150)
print("Saved results/speed_benchmark.png")

# ----------------------------------------------------------------------------
# 6. Save a small summary file (used by README / LinkedIn post)
# ----------------------------------------------------------------------------
with open("results/summary.txt", "w") as f:
    f.write("CosmoEmulator-NN results\n")
    f.write("========================\n")
    f.write(f"Training set   : {X_train_raw.shape[0]:,} points from 1200 CPL dark-energy models\n")
    f.write(f"Held-out RMS   : {rms*1000:.2f} mmag (max {max_err*1000:.2f} mmag)\n")
    f.write(f"Speed-up       : {speedup:,.0f}x over {n_bench:,} likelihood evaluations\n")
    f.write(f"Numerical time : {t_numerical:.3f}s | Emulator time: {t_emulator:.4f}s\n")

print("\nDone.")
