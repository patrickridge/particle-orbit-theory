import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", context="paper")

from orbit_ivp_core import simulate_orbit_ivp, q, m
from fields import E_zero, B_uniform_z

# Figures directory — resolved relative to this script, so the script runs correctly from any working directory.
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

# =============================================================
# Test 2: Helical motion in a uniform magnetic field
#
# Adding a velocity component along B makes the particle trace a helix.
# The magnetic force does not act along B, so the parallel speed is constant.
# Expected: 3D corkscrew; x-y circle unchanged, z(t) increases linearly.
# =============================================================

B0    = 1.0
B_func = B_uniform_z(B0)
E_func = E_zero

dt     = 0.01
T      = 20.0
nsteps = int(T / dt)

r0     = np.array([0.0, 0.0, 0.0])
v0     = np.array([1.0, 0.0, 0.2])   # v_perp = 1.0, v_par = 0.2
state0 = np.concatenate((r0, v0))

t, traj = simulate_orbit_ivp(
    state0=state0, dt=dt, nsteps=nsteps,
    q=q, m=m, E_func=E_func, B_func=B_func,
)

x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

# ---- Analytic predictions -------------------------------------------
v_perp = np.sqrt(v0[0]**2 + v0[1]**2)
v_par  = v0[2]
Omega  = q * B0 / m
r_L    = m * v_perp / (q * B0)

z_theory = v_par * t          # linear z(t)
print(f"v_parallel (theory): {v_par}")
print(f"Max |z_numerical - z_theory|: {np.max(np.abs(z - z_theory)):.2e}")

# ======================================================================
# Only figure: 3D helix (clean, report-quality)
# ======================================================================
fig = plt.figure(figsize=(6, 7))
ax  = fig.add_subplot(111, projection="3d")

# Field lines — vertical lines parallel to B = B0 z-hat
# Use uniform, subtle lines — all same weight
z_fl = np.array([z.min() - 0.3, z.max() + 0.3])
fl_positions = [(-1.3, 0), (1.3, 0), (0, -1.3), (0, 1.3)]
for xf, yf in fl_positions:
    ax.plot([xf]*2, [yf]*2, z_fl, color="steelblue", lw=0.5, alpha=0.25)
    ax.quiver(xf, yf, z_fl[-1], 0, 0, 0.2,
              color="steelblue", lw=0.5, alpha=0.25, arrow_length_ratio=0.7)

# GC path (dashed) — at the actual GC location (0, -r_L)
z_gc = v_par * t
ax.plot([0]*len(t), [-r_L]*len(t), z_gc,
        ls="--", lw=1.2, color="C1", alpha=0.7, label="Guiding centre")

# Particle orbit
ax.plot(x, y, z, lw=1.2, color="C0", label="Particle orbit")

# Start marker
ax.plot([x[0]], [y[0]], [z[0]], "o", color="k", ms=6, zorder=10)
ax.text(x[0] + 0.15, y[0], z[0] - 0.15, "Start", fontsize=8, color="k")

# B-field label
ax.text2D(0.05, 0.92, r"$\mathbf{B} = B_0\,\hat{\mathbf{z}}$",
          transform=ax.transAxes, fontsize=10, color="steelblue")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Helical motion in uniform $B$", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.view_init(elev=18, azim=-65)
ax.set_box_aspect((1, 1, 2))
fig.subplots_adjust(bottom=0.05, top=0.93, left=0.05, right=0.95)
plt.savefig(os.path.join(_FIG, "test02_uniformB_helix_3D.png"), dpi=300)
plt.show()
