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
# Plot 1: z(t) — should be linear; overlay analytic line
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, z, lw=1.5, label="Numerical z(t)")
ax.plot(t, z_theory, "k--", lw=1.0,
        label=fr"Theory: $z = v_{{\parallel}} t = {v_par}\,t$")
ax.set_xlabel("t")
ax.set_ylabel("z")
ax.set_title("Test 2: Uniform B — linear motion along field line")
ax.legend(frameon=True, fontsize=8)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test02_uniformB_z.png"), dpi=300)
plt.show()

# ======================================================================
# Plot 2: x-y projection — should be circular (unchanged by v_par)
# ======================================================================
gc_x     = 0.0
gc_y     = -r_L
theta    = np.linspace(0, 2 * np.pi, 500)
x_circle = gc_x + r_L * np.cos(theta)
y_circle = gc_y + r_L * np.sin(theta)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(x, y, lw=1.5, label="Numerical orbit")
ax.plot(x_circle, y_circle, "k--", lw=1.0,
        label=fr"Theory ($r_L={r_L:.2f}$)")
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Test 2: Uniform B — x-y projection (gyromotion)")
ax.legend(frameon=True, fontsize=8)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test02_uniformB_xy.png"), dpi=300)
plt.show()

# ======================================================================
# Plot 3: 3D helix
# ======================================================================
fig = plt.figure(figsize=(6, 5))
ax  = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, lw=1.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Test 2: Helical motion in uniform B field")
ax.set_box_aspect((1, 1, 2))
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test02_uniformB_helix_3D.png"), dpi=300)
plt.show()