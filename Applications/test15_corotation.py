import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib import cm

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, E_corotation, B_dipole_cartesian

# Figures directory — resolved relative to this script, so the script runs correctly from any working directory.
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 15: Corotation E×B drift — aligned rotating dipole
#
# Adding the corotation electric field E = -(Omega x r) x B
# causes the guiding centre to co-rotate with the field line
# at angular velocity Omega.  Gyration and bounce are unchanged
# and sit on top of the azimuthal co-rotation drift.
#
# Omega = 0.02 gives one full corotation in ~314 time units,
# covering ~18 bounce periods — enough to see the drift clearly.
# =============================================================

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
T_run      = T_cor
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

# Run B: same parameters but no E field — isolates gradient+curvature drift
print("Integrating (no E field, gradient+curvature drift only) ...")
t_noE, traj_noE = simulate_orbit_ivp(
    state0=state0, dt=dt, nsteps=nsteps,
    q=q, m=m, E_func=E_zero, B_func=B_func,
    rtol=1e-9, atol=1e-9,
)
print("done.")

# Guiding-centre positions extracted analytically
gc     = extract_gc(traj,    t,    B_func, q=q, m=m)
gc_noE = extract_gc(traj_noE, t_noE, B_func, q=q, m=m)

# Decimate for plotting (one point per gyration keeps plots readable)
x_gc = gc[::skip, 0];    y_gc = gc[::skip, 1];    z_gc = gc[::skip, 2];    t_gc = t[::skip]
x_gc_noE = gc_noE[::skip, 0];  y_gc_noE = gc_noE[::skip, 1];  t_gc_noE = t_noE[::skip]

# Measure actual co-rotation rate from linear fit to phi(t)
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

# ======================================================================
# Plot 1: Top-down x-y view
# ======================================================================
fig1, ax1 = plt.subplots(figsize=(6, 6))

for i in range(len(x_gc) - 1):
    c = cmap_used(norm(0.5 * (t_gc[i] + t_gc[i + 1])))
    ax1.plot(x_gc[i:i+2], y_gc[i:i+2], color=c, lw=1.6)

theta = np.linspace(0, 2 * np.pi, 500)
ax1.plot(r0[0] * np.cos(theta), r0[0] * np.sin(theta),
         "k--", lw=1.0, label=f"Expected co-rotation (r={r0[0]})")

theta_p = np.linspace(0, 2 * np.pi, 200)
ax1.fill(np.cos(theta_p), np.sin(theta_p), color="lightgray", zorder=5)
ax1.plot(np.cos(theta_p), np.sin(theta_p), "k-", lw=0.8, zorder=6)

sm = cm.ScalarMappable(cmap=cmap_used, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax1, label="t (code units)", shrink=0.8)

ax1.set_aspect("equal")
ax1.set_xlabel("x (code units)")
ax1.set_ylabel("y (code units)")
ax1.set_title(f"Test 15: Co-rotation — top-down view (Ω={Omega})")
ax1.legend(fontsize=8, handles=[
    plt.Line2D([0],[0], color="gray", lw=1.5, label="GC trajectory (coloured by time)"),
    plt.Line2D([0],[0], color="k", lw=1.0, ls="--", label=f"Expected co-rotation (r={r0[0]:.1f})")
])
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test15_corotation_xy.png"), dpi=300)
plt.close()
print("Saved test15_corotation_xy.png")

# ======================================================================
# Plot 2: z(t) — bounce persists under co-rotation
# ======================================================================
fig2, ax2 = plt.subplots(figsize=(9, 3.5))
ax2.plot(t_gc, z_gc, lw=0.9, color="C0")
ax2.axhline(0, color="gray", lw=0.6, ls="--")
ax2.set_xlabel("t (code units)")
ax2.set_ylabel("z (code units)")
ax2.set_title("Test 15: z(t) — bounce persists under co-rotation")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test15_corotation_z_vs_t.png"), dpi=300)
plt.close()
print("Saved test15_corotation_z_vs_t.png")

# ======================================================================
# Plot 3: φ(t) — co-rotation rate verification + comparison with no-E
# ======================================================================
phi_theory = Omega * t_gc

fig3, ax_top = plt.subplots(figsize=(9, 4.5))

ax_top.plot(t_gc,     phi_gc,     lw=1.0, color="C0",
            label=f"With E (co-rotation): Ω_meas = {Omega_meas:.4f}")
ax_top.plot(t_gc_noE, phi_gc_noE, lw=1.0, color="C2", ls="-.",
            label=f"No E (grad+curv drift only): Ω_meas = {Omega_meas_noE:.4f}")
ax_top.plot(t_gc,     phi_theory, lw=1.2, color="k",  ls="--",
            label=f"Pure co-rotation: φ = Ωt  (Ω={Omega})")
ax_top.set_xlabel("t (code units)")
ax_top.set_ylabel("φ (rad)")
ax_top.set_title("Test 15: Azimuthal drift — co-rotation vs gradient/curvature drift\n"
                 "Gap between blue and dashed = gradient+curvature drift (physical, not error)")
ax_top.legend(fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test15_corotation_phi_vs_t.png"), dpi=300)
plt.close()
print("Saved test15_corotation_phi_vs_t.png")

print("\nAll test15 figures saved.")