import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib import cm

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, E_corotation, B_dipole_cartesian

# output directory
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# Test 15: corotation ExB drift
# Check: GC co-rotates at angular velocity Omega with corotation E field

q, m  = 1.0, 1.0
M     = 500.0
Omega = 0.02

B_func = B_dipole_cartesian(M=M, tilt_deg=0.0)
E_func = E_corotation(B_func, Omega=Omega)

r0    = np.array([3.0, 0.0, 0.0])
v_mag = 1.0
pitch = np.deg2rad(45.0)

B0_vec    = B_func(r0, 0.0)
bhat      = B0_vec / np.linalg.norm(B0_vec)
v0        = v_mag * (np.cos(pitch) * bhat
                   + np.sin(pitch) * np.array([0.0, 1.0, 0.0]))
state0    = np.concatenate([r0, v0])

Omega_gyro = abs(q) * np.linalg.norm(B0_vec) / m
T_gyro     = 2.0 * np.pi / Omega_gyro
T_b_est    = 4.0 * r0[0] / abs(np.dot(v0, bhat))
T_cor      = 2.0 * np.pi / Omega
T_run      = 0.5 * T_cor         # half corotation period
dt         = min(T_b_est / 500.0, 0.05 * T_gyro)
nsteps     = int(T_run / dt) + 1
skip       = max(1, int(round(T_gyro / dt)))

print(f"T_corotation={T_cor:.1f}, T_bounce={T_b_est:.2f}, "
      f"T_b/T_g={T_b_est/T_gyro:.1f}, nsteps={nsteps}")
print(f"Timescale hierarchy: T_gyro={T_gyro:.3f} << T_bounce={T_b_est:.1f} << T_corot={T_cor:.0f}")
print("Integrating (with E_corotation) ...")

t, traj = simulate_orbit_ivp(
    state0=state0, dt=dt, nsteps=nsteps,
    q=q, m=m, E_func=E_func, B_func=B_func,
    rtol=1e-9, atol=1e-9,
)
print("done.")

# no-E run for comparison
print("Integrating (no E field, gradient+curvature drift only) ...")
t_noE, traj_noE = simulate_orbit_ivp(
    state0=state0, dt=dt, nsteps=nsteps,
    q=q, m=m, E_func=E_zero, B_func=B_func,
    rtol=1e-9, atol=1e-9,
)
print("done.")

# GC positions
gc     = extract_gc(traj,    t,    B_func, q=q, m=m)
gc_noE = extract_gc(traj_noE, t_noE, B_func, q=q, m=m)

# decimate for plotting
x_gc = gc[::skip, 0];    y_gc = gc[::skip, 1];    z_gc = gc[::skip, 2];    t_gc = t[::skip]
x_gc_noE = gc_noE[::skip, 0];  y_gc_noE = gc_noE[::skip, 1];  t_gc_noE = t_noE[::skip]

# measure co-rotation rate
phi_gc     = np.unwrap(np.arctan2(y_gc, x_gc))
phi_gc_noE = np.unwrap(np.arctan2(y_gc_noE, x_gc_noE))
n          = len(phi_gc)
i0, i1     = n // 10, 9 * n // 10
Omega_meas    = np.polyfit(t_gc[i0:i1],    phi_gc[i0:i1],    1)[0]
Omega_meas_noE = np.polyfit(t_gc_noE[i0:i1], phi_gc_noE[i0:i1], 1)[0]
print(f"With E:    measured Ω = {Omega_meas:.5f}  (input Ω = {Omega:.5f},"
      f"  excess = {(Omega_meas-Omega)/Omega*100:.1f}%)")
print(f"Without E: measured Ω = {Omega_meas_noE:.5f}  (gradient+curvature drift only)")
print(f"E×B contribution: ΔΩ = {Omega_meas - Omega_meas_noE:.5f} rad/unit-time"
      f"  (expected ≈ {Omega:.5f})")

norm      = Normalize(vmin=t_gc.min(), vmax=t_gc.max())
cmap_used = cm.plasma

# x-y view
fig1, ax1 = plt.subplots(figsize=(7, 7))

# L-shell circle
theta = np.linspace(0, 2 * np.pi, 500)
ax1.plot(r0[0] * np.cos(theta), r0[0] * np.sin(theta),
         "k--", lw=0.8, alpha=0.4, zorder=1, label=f"$L = {r0[0]:.0f}$ reference circle")

# trajectory coloured by time
for i in range(len(x_gc) - 1):
    c = cmap_used(norm(0.5 * (t_gc[i] + t_gc[i + 1])))
    ax1.plot(x_gc[i:i+2], y_gc[i:i+2], color=c, lw=2.2, zorder=3)

# planet
theta_p = np.linspace(0, 2 * np.pi, 200)
ax1.fill(np.cos(theta_p), np.sin(theta_p), color="lightsteelblue", zorder=5)
ax1.plot(np.cos(theta_p), np.sin(theta_p), color="#666666", lw=0.8, zorder=6)

# start marker
ax1.plot(r0[0], 0, "o", color="k", ms=6, zorder=9)
ax1.text(r0[0] + 0.12, 0.18, "Start", fontsize=8, color="k")

# rotation arm arrow
phi_end = Omega * t_gc[-1]
ax1.annotate("",
             xy=(r0[0]*np.cos(phi_end), r0[0]*np.sin(phi_end)),
             xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.8),
             zorder=7)

# swept arc
R_arm = r0[0] - 0.6
phi_arm = np.linspace(0, phi_end, 300)
ax1.plot(R_arm*np.cos(phi_arm), R_arm*np.sin(phi_arm),
         color="crimson", lw=2.5, ls="-", alpha=0.7, zorder=4)
ax1.plot(R_arm*np.cos(phi_end), R_arm*np.sin(phi_end),
         "o", color="crimson", ms=5, zorder=4)

sm = cm.ScalarMappable(cmap=cmap_used, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax1, label="t (code units)", shrink=0.8)

# legend
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color="crimson", lw=1.8, linestyle="-",
           label=r"$\mathbf{E}\times\mathbf{B}$ drift ($\Omega$ arrow)"),
    Line2D([0], [0], color="C1", lw=2.2,
           label=r"Full drift: $\mathbf{E}\times\mathbf{B}$ + magnetic drifts"),
    Line2D([0], [0], color="k", lw=0.8, linestyle="--",
           label=f"$L = {r0[0]:.0f}$ reference circle"),
]
ax1.legend(handles=legend_handles, fontsize=8, loc="lower left", framealpha=0.9)

ax1.set_aspect("equal")
ax1.set_xlim(-4.0, 4.0); ax1.set_ylim(-4.0, 4.0)
ax1.set_xlabel("x (code units)")
ax1.set_ylabel("y (code units)")
ax1.set_title(f"Co-rotation drift — top-down view", fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test15_corotation_xy.png"), dpi=300)
plt.close()
print("Saved test15_corotation_xy.png")

# Plot 2: z(t) bounce under co-rotation
fig2, ax2 = plt.subplots(figsize=(9, 3.5))
ax2.plot(t_gc, z_gc, lw=0.9, color="C0")
ax2.axhline(0, color="gray", lw=0.6, ls="--")
ax2.set_xlabel("t (code units)")
ax2.set_ylabel("z (code units)")
ax2.set_title("Bounce motion persists under co-rotation")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test15_corotation_z_vs_t.png"), dpi=300)
plt.close()
print("Saved test15_corotation_z_vs_t.png")

print("\nAll test15 figures saved.")
