import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from orbit_ivp_core import simulate_orbit_ivp, q, m
from fields import E_zero, B_curved_z

# output directory
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# Test 9: curvature drift
# Check: GC drifts at v_curv = m*v_par^2 / (q*B0*R_c)

B0  = 1.0
R_c = 10000.0  # must be >> v_par*T so particle stays in linear-approx valid region
               # With R_c=10000: z/R_c ~ 0.1 and x_GC/R_c ~ 5e-5 after T=1000

E_func = E_zero
B_func = B_curved_z(B0=B0, R_c=R_c)

dt     = 0.01
T      = 1000.0   # long run so drift accumulates: Δy = v_curv*T = T/R_c = 0.1
nsteps = int(T / dt)

# mostly parallel IC so curvature drift dominates
r0     = np.array([0.0, 0.0, 0.0])
v_par0 = 1.0
v_perp0 = 0.15                          # small perp so grad-B drift is tiny
v0     = np.array([0.0, 0.0, v_par0])   # parallel to B (z-direction near origin)
# small perp component
v0[0]  = v_perp0
state0 = np.concatenate((r0, v0))

t, traj = simulate_orbit_ivp(state0, dt, nsteps,
                              q=q, m=m,
                              E_func=E_func, B_func=B_func)

x, y = traj[:, 0], traj[:, 1]

# gyro-averaging
Omega0 = q * B0 / m
Tgyro  = 2.0 * np.pi / Omega0
W      = max(5, int(Tgyro / dt))
kernel = np.ones(W) / W

y_gc = np.convolve(y, kernel, mode="same")
x_gc = np.convolve(x, kernel, mode="same")

# analytic predictions
v_curv_theory  = (m * v_par0**2) / (q * B0 * R_c)    # y-component
v_gradB_theory = (m * v_perp0**2) / (2.0 * q * B0 * R_c)  # same direction, smaller

# numerical estimate
cut         = W
v_drift_num = np.polyfit(t[cut:-cut], y_gc[cut:-cut], 1)[0]

print(f"Analytic curvature drift speed    (y): {v_curv_theory:.6e}")
print(f"Analytic grad-B    drift speed    (y): {v_gradB_theory:.6e}  (for reference)")
print(f"Total analytic drift              (y): {v_curv_theory + v_gradB_theory:.6e}")
print(f"Numerical drift speed             (y): {v_drift_num:.6e}")
print(f"Relative error vs curvature+gradB: "
      f"{abs(abs(v_drift_num) - (v_curv_theory + v_gradB_theory)) / (v_curv_theory + v_gradB_theory) * 100:.2f}%")

# analytic y(t)
y_theory = y_gc[cut] - (v_curv_theory + v_gradB_theory) * (t - t[cut])

# x-y GC path
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x_gc, y_gc, lw=1.4, label="Guiding centre (gyro-avg)")
ax.set_xlabel("x (gyro-avg)")
ax.set_ylabel("y (gyro-avg)")
ax.set_title("Test 9: Curvature drift — guiding-centre path")
ax.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test09_curvature_gc_xy.png"), dpi=300)
plt.show()

# y(t) with analytic overlay
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})

axes[0].plot(t, y,    alpha=0.2, lw=0.6, color="C0", label="y(t) raw")
axes[0].plot(t, y_gc, lw=1.3,   color="C0", label="y(t) gyro-avg")
axes[0].plot(t, y_theory, "k--", lw=1.1,
             label=fr"Theory: $v_{{curv}}+v_{{\nabla B}} = {v_curv_theory + v_gradB_theory:.4f}$")
axes[0].set_ylabel("y")
axes[0].set_title("Test 9: Curvature drift — numerical vs analytic")
axes[0].legend(frameon=True, fontsize=8)

axes[1].plot(t, y_gc - y_theory, lw=1.0, color="C1")
axes[1].axhline(0, color="k", lw=0.6, ls="--")
axes[1].set_xlabel("t")
axes[1].set_ylabel("residual")

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test09_curvature_y_vs_t.png"), dpi=300)
plt.show()