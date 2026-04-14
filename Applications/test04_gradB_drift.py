import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from orbit_ivp_core import simulate_orbit_ivp, extract_gc, q, m
from fields import E_zero, B_gradx_z

# Figures directory
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 4: Grad-B drift (gradient drift)
#
# Static figure inspired by animate04_gradb.py.
# Shows the full orbit (asymmetric loops) and GC drift in a
# non-uniform field with field-line spacing encoding |B|.
# =============================================================

B0  = 1.0
eps = 0.28          # clear gradient (same as animation — gives visible asymmetry)

B_func  = B_gradx_z(B0=B0, eps=eps)
Omega_c = abs(q) * B0 / m
T_gyro  = 2.0 * np.pi / Omega_c
r_L     = 1.0
v_perp  = r_L * Omega_c

r0     = np.array([r_L, 0.0, 0.0])
v0     = np.array([0.0, v_perp, 0.0])
state0 = np.concatenate([r0, v0])

N_gyro = 10
T_run  = N_gyro * T_gyro
dt     = T_gyro / 80.0
nsteps = int(T_run / dt)

print("Integrating ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_zero, B_func=B_func,
                              rtol=1e-9, atol=1e-9)
r  = traj[:, :3]
gc = extract_gc(traj, t, B_func, q=q, m=m)
print("Done.")

# ---- Analytic drift speed ----
v_drift_theory = -v_perp**2 * eps / (2.0 * Omega_c)
print(f"Theory grad-B drift speed (y): {v_drift_theory:.4f}")
n = len(t); i0, i1 = n//10, 9*n//10
v_drift_num = np.polyfit(t[i0:i1], gc[i0:i1, 1], 1)[0]
print(f"Numerical drift speed (y):     {v_drift_num:.4f}")
rel_err = abs(v_drift_num - v_drift_theory) / abs(v_drift_theory) * 100
print(f"Relative error: {rel_err:.1f}%")

# ---- Axis limits ----
pad   = 1.4
x_min = r[:, 0].min() - pad;  x_max = r[:, 0].max() + pad
y_min = r[:, 1].min() - pad;  y_max = r[:, 1].max() + pad + 0.5

# ---- Field lines: vertical lines spaced inversely with B(x) ----
N_lines = 20
F_of_x  = lambda x: B0 * (x + eps * x**2 / 2.0)
F_steps = np.linspace(F_of_x(x_min), F_of_x(x_max), N_lines)
line_x  = []
for F in F_steps:
    a = eps / 2.0
    b = 1.0
    c = -F / B0
    disc = b**2 - 4*a*c
    line_x.append((-b + np.sqrt(disc)) / (2*a))

# ======================================================================
# Single figure (inspired by animate04)
# ======================================================================
fig, ax = plt.subplots(figsize=(6, 7))

# Subtle background gradient showing field strength variation
grad = np.linspace(0, 1, 300).reshape(1, -1)
ax.imshow(grad, aspect="auto", cmap="Blues", alpha=0.12,
          extent=[x_min, x_max, y_min, y_max], origin="lower")

# Field lines as vertical segments (closer together = stronger B)
for lx in line_x:
    ax.plot([lx, lx], [y_min, y_max],
            color="steelblue", lw=0.8, alpha=0.4)

# B out-of-page symbol
odot_x = x_min + 0.50
odot_y = y_max - 0.60
odot_r = 0.25
ax.add_patch(mpatches.Circle((odot_x, odot_y), odot_r,
                              fill=False, edgecolor="steelblue", lw=1.6, zorder=6))
ax.plot(odot_x, odot_y, "o", color="steelblue", ms=3.5, zorder=7)
ax.text(odot_x + odot_r + 0.12, odot_y, r"$\mathbf{B}$",
        va="center", fontsize=11, color="steelblue", fontweight="bold")

# "increasing |B|" arrow across the top
ax.annotate("", xy=(x_max - 0.3, y_max - 0.60),
            xytext=(x_min + 1.8, y_max - 0.60),
            arrowprops=dict(arrowstyle="-|>", color="steelblue",
                            lw=1.6, alpha=0.65))
ax.text((x_min + x_max) / 2 + 0.3, y_max - 0.35,
        r"increasing $|\mathbf{B}|$", ha="center", fontsize=9,
        color="steelblue", alpha=0.75)

# Full orbit — thin, show asymmetric loops
ax.plot(r[:, 0], r[:, 1], lw=0.7, alpha=0.45, color="#4488CC",
        label="Particle orbit")

# Guiding centre — bold orange
ax.plot(gc[:, 0], gc[:, 1], lw=2.5, color="darkorange",
        label="Guiding centre", zorder=5)

# Start marker
ax.plot(r[0, 0], r[0, 1], "o", color="k", ms=6, zorder=9)
ax.text(r[0, 0] + 0.15, r[0, 1] + 0.15, "Start", fontsize=8, color="k")

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"Gradient-$B$ drift", fontsize=11)
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test04_gradB_drift.png"), dpi=300)
plt.show()
