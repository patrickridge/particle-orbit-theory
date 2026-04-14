import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from orbit_ivp_core import simulate_orbit_ivp, q, m
from fields import E_zero, B_mirror_div_free

# Figures directory — resolved relative to this script, so the script runs correctly from any working directory.
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 5: Magnetic mirror — bounce motion

#   A charged particle bouncing back and forth along a magnetic field
#   that gets stronger towards the ends (like a bottle). The increasing
#   field reflects the particle before it escapes — this is the magnetic
#   mirror effect. The particle's z-coordinate oscillates between two
#   turning (mirror) points where its parallel velocity reverses.
#
# Field used: B_mirror_div_free — divergence-free mirror model
#   Bz(z) = B0 * (1 + alpha * z^2),  Bx and By adjusted so div(B) = 0
#   Field gets stronger away from z = 0 (the midplane).
#
# Expected result:
#   - z(t) oscillates between ±z_mirror  (bounce motion)
#   - Turning points occur where all kinetic energy is in v_perp
#   - Energy is conserved (E = 0, so |v| = const)
# =============================================================

# --- Mirror-like field (divergence-free local model) ---
B0 = 1.0
alpha = 0.5
B_func = B_mirror_div_free(B0=B0, alpha=alpha)
E_func = E_zero

# --- Time grid ---
dt = 0.01
T = 80.0
nsteps = int(T / dt)

# --- IC: mostly perpendicular, small parallel ---
r0 = np.array([0.0, 0.0, 0.0])
v0 = np.array([1.0, 0.0, 0.05])
state0 = np.concatenate((r0, v0))

t, traj = simulate_orbit_ivp(
    state0=state0, dt=dt, nsteps=nsteps,
    q=q, m=m, E_func=E_func, B_func=B_func,
)

r = traj[:, :3]
v = traj[:, 3:]

z = r[:, 2]
vz = v[:, 2]

# turning points: vz changes sign
turn = np.where(np.sign(vz[:-1]) != np.sign(vz[1:]))[0]

# energy diagnostic
K = 0.5 * m * np.sum(v**2, axis=1)
abs_rel_drift = np.abs((K - K[0]) / K[0])

print(f"Number of turning points: {len(turn)}")
print(f"Max |relative energy drift|: {abs_rel_drift.max():.3e}")

# --- Plot 1: z(t) with turning points ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, z, lw=1.2, label="z(t)")
if len(turn) > 0:
    ax.scatter(t[turn], z[turn], s=18, zorder=3, label="turning points")
ax.set_xlabel("t")
ax.set_ylabel("z")
ax.set_title("Mirror bounce in toy field", fontsize=11)
ax.legend(fontsize=8, frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test05_mirror_z_turning.png"), dpi=300)
plt.show()

# --- Plot 2: 3D mirror bounce (static version of animate05) ---
from orbit_ivp_core import extract_gc
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Re-run with ICs matching animate05 for cleaner 3D view
omega_gc  = abs(q) * B0 / m
T_gyro_gc = 2.0 * np.pi / omega_gc
r_L_gc    = 1.0
v_perp_gc = r_L_gc * omega_gc
v_par_gc  = 0.5 * v_perp_gc

r0_3d     = np.array([0.0, r_L_gc, 0.0])
v0_3d     = np.array([v_perp_gc, 0.0, v_par_gc])
state0_3d = np.concatenate([r0_3d, v0_3d])

v_tot_3d    = np.sqrt(v_perp_gc**2 + v_par_gc**2)
sin_a0_3d   = v_perp_gc / v_tot_3d
B_mirror_3d = B0 / sin_a0_3d**2
z_mirror_3d = np.sqrt((B_mirror_3d / B0 - 1) / alpha)

T_bounce_3d = 4 * z_mirror_3d / (v_par_gc * 0.7)
T_run_3d    = 3.0 * T_bounce_3d
dt_3d       = T_gyro_gc / 40
nsteps_3d   = int(T_run_3d / dt_3d)

t_3d, traj_3d = simulate_orbit_ivp(
    state0=state0_3d, dt=dt_3d, nsteps=nsteps_3d,
    q=q, m=m, E_func=E_func, B_func=B_func,
    rtol=1e-9, atol=1e-9)
r_3d = traj_3d[:, :3]
gc_3d = extract_gc(traj_3d, t_3d, B_func, q=q, m=m)

# Mirror points
vz_3d = traj_3d[:, 5]
mirrors_3d = np.where(np.diff(np.sign(vz_3d)))[0]

# Converging field lines
z_fl     = np.linspace(-z_mirror_3d * 1.3, z_mirror_3d * 1.3, 300)
r_starts = [0.5, 1.0, 1.5, 2.0]
phi_vals = np.linspace(0, 2 * np.pi, 8, endpoint=False)

fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")

# Field lines
for r0_fl in r_starts:
    for phi_f in phi_vals:
        scale = 1.0 / np.sqrt(1 + alpha * z_fl**2)
        xf = r0_fl * np.cos(phi_f) * scale
        yf = r0_fl * np.sin(phi_f) * scale
        ax.plot(xf, yf, z_fl, color="#888888", lw=0.5, alpha=0.2)

# Mirror plane rings
theta_r = np.linspace(0, 2 * np.pi, 80)
r_ring  = r_starts[-1] / np.sqrt(1 + alpha * z_mirror_3d**2) * 1.05
for zm in [z_mirror_3d, -z_mirror_3d]:
    ax.plot(r_ring * np.cos(theta_r), r_ring * np.sin(theta_r),
            [zm] * 80, "r--", lw=1.5, alpha=0.6)
ax.text(r_ring + 0.1, 0, z_mirror_3d + 0.05,
        "mirror", color="crimson", fontsize=7)
ax.text(r_ring + 0.1, 0, -z_mirror_3d + 0.05,
        "mirror", color="crimson", fontsize=7)

# Full orbit
ax.plot(r_3d[:, 0], r_3d[:, 1], r_3d[:, 2],
        lw=0.5, alpha=0.35, color="C0", label="Full orbit")

# GC path
ax.plot(gc_3d[:, 0], gc_3d[:, 1], gc_3d[:, 2],
        lw=1.8, color="C1", label="Guiding centre")

# Mirror point markers
if len(mirrors_3d) > 0:
    ax.scatter(gc_3d[mirrors_3d, 0], gc_3d[mirrors_3d, 1], gc_3d[mirrors_3d, 2],
               s=40, color="limegreen", zorder=10, label="Mirror points",
               edgecolor="white", linewidth=0.5)

lim = r_starts[-1] * 1.1
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_zlim(-z_mirror_3d * 1.5, z_mirror_3d * 1.5)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Magnetic mirror bounce", fontsize=11)
ax.view_init(elev=20, azim=-55)
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test05_mirror_3D.png"), dpi=300)
plt.show()
