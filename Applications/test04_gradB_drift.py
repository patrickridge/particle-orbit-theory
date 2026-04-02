import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from orbit_ivp_core import simulate_orbit_ivp, q, m
from fields import E_zero, B_gradx_z

# Figures directory — resolved relative to this script, so the script runs correctly from any working directory.
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 4: Grad-B drift (gradient drift)
#
# A spatially varying B field causes the guiding centre to drift sideways.
# Gyroradius is larger where B is weaker, producing a net drift perpendicular
# Expected: slow +y drift; measured speed matches v_gradB ≈ m v_perp^2 eps / (2 q B0).
# =============================================================

B0  = 1.0
eps = 0.05    # small gradient so adiabatic theory holds

E_func = E_zero
B_func = B_gradx_z(B0=B0, eps=eps)

dt     = 0.01
T      = 80.0
nsteps = int(T / dt)

r0     = np.array([0.0, 0.0, 0.0])
v0     = np.array([1.0, 0.3, 0.0])    # mostly perpendicular to B
state0 = np.concatenate((r0, v0))

t, traj = simulate_orbit_ivp(state0, dt, nsteps,
                              q=q, m=m,
                              E_func=E_func, B_func=B_func)

x, y = traj[:, 0], traj[:, 1]

# ---- Gyro-averaging --------------------------------------------------
Omega0 = q * B0 / m
Tgyro  = 2.0 * np.pi / Omega0
W      = max(5, int(Tgyro / dt))
kernel = np.ones(W) / W

y_gc   = np.convolve(y, kernel, mode="same")
x_gc   = np.convolve(x, kernel, mode="same")

# ---- Analytic drift --------------------------------------------------
v_perp_sq = v0[0]**2 + v0[1]**2      # initial v_perp^2  (vz=0 here)
v_gradB_theory = (m * v_perp_sq * eps) / (2.0 * q * B0)
print(f"Analytic  grad-B drift speed (y): {v_gradB_theory:.6e}")

# ---- Numerical estimate (avoid edge effects from convolution) --------
cut         = W
v_drift_num = np.polyfit(t[cut:-cut], y_gc[cut:-cut], 1)[0]
print(f"Numerical grad-B drift speed (y): {v_drift_num:.6e}")
print(f"Relative error: {abs(v_drift_num - v_gradB_theory) / v_gradB_theory * 100:.2f}%")

# ======================================================================
# Plot 1: Raw orbit in x-y (shows varying gyroradius)
# ======================================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y, alpha=0.45, lw=0.7)
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Test 4: Grad-B — raw orbit (note varying gyroradius)")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test04_gradB_xy_raw.png"), dpi=300)
plt.show()

# ======================================================================
# Plot 2: Guiding-centre path (gyro-averaged x-y)
# ======================================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x_gc, y_gc, lw=1.5, label="Guiding centre (gyro-avg)")
ax.set_xlabel("x (gyro-avg)"); ax.set_ylabel("y (gyro-avg)")
ax.set_title("Test 4: Grad-B drift — guiding-centre trajectory")
ax.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test04_gradB_gc_xy.png"), dpi=300)
plt.show()

# ======================================================================
# Plot 3: y(t) — gyro-averaged with analytic overlay and residuals
# ======================================================================
y_theory = v_gradB_theory * t    # analytic guiding-centre y(t)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})

axes[0].plot(t, y,    alpha=0.2, lw=0.6, color="C0", label="y(t) raw")
axes[0].plot(t, y_gc, lw=1.3,   color="C0", label="y(t) gyro-avg")
axes[0].plot(t, y_theory, "k--", lw=1.1,
             label=fr"Theory: $v_{{\nabla B}}={v_gradB_theory:.4f}$")
axes[0].set_ylabel("y")
axes[0].set_title("Test 4: Grad-B drift — numerical vs analytic")
axes[0].legend(frameon=True, fontsize=8)

axes[1].plot(t, y_gc - y_theory, lw=1.0, color="C1")
axes[1].axhline(0, color="k", lw=0.6, ls="--")
axes[1].set_xlabel("t")
axes[1].set_ylabel("residual")
axes[1].set_title("Residual: gyro-avg − theory")

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test04_gradB_y_vs_t.png"), dpi=300)
plt.show()
