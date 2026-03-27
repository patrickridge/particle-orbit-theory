"""
animate15_corotation.py
=======================
Animated comparison of GC drift with and without the corotation E field.

Left panel:  No E field — only slow gradient+curvature drift (Ω ≈ 0.008)
Right panel: Corotation E field — GC rotates at exactly Ω = 0.02 with the planet

A rotating "planet arm" in the right panel shows the nominal corotation rate.
The GC stays locked to it — demonstrating perfect co-rotation.

Saves: ../Figures/animate15_corotation.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, E_corotation, B_dipole_cartesian

# ---- Parameters (match test15) ----
q, m  = 1.0, 1.0
M     = 500.0
Omega = 0.02

B_func   = B_dipole_cartesian(M=M, tilt_deg=0.0)
E_func   = E_corotation(B_func, Omega=Omega)
E_func_0 = E_zero

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

dt     = min(T_b_est / 300.0, 0.05 * T_gyro)
nsteps = int(T_cor / dt) + 1
skip   = max(1, int(round(T_gyro / dt)))

print(f"T_gyro={T_gyro:.3f}, T_bounce={T_b_est:.2f}, T_cor={T_cor:.1f}")
print(f"dt={dt:.4f}, nsteps={nsteps}")

print("Integrating with E field ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_func, B_func=B_func,
                              rtol=1e-9, atol=1e-9)
print("Integrating without E field ...")
_, traj_noE = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                                  q=q, m=m, E_func=E_func_0, B_func=B_func,
                                  rtol=1e-9, atol=1e-9)
print("Done.")

gc     = extract_gc(traj,    t,    B_func, q=q, m=m)
gc_noE = extract_gc(traj_noE, t,   B_func, q=q, m=m)

# Decimate to one point per gyration
gc_s     = gc[::skip];    t_s = t[::skip]
gc_noE_s = gc_noE[::skip]

# ---- Figure — two side-by-side top-down views ----
fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(
    r"Corotation E×B drift — aligned dipole ($\Omega = 0.02$)"
    "\nTop-down view over one full planetary rotation",
    fontsize=11
)

L0 = np.linalg.norm(r0)
theta_c = np.linspace(0, 2 * np.pi, 300)
r_planet = 1.0

for ax, title in zip([ax_L, ax_R],
                     ["No E field\n(gradient+curvature drift only, slow)",
                      "Corotation E field\n(GC rotates with planet at Ω)"]):
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title, fontsize=9, pad=6)

    # L-shell reference circle
    ax.plot(L0 * np.cos(theta_c), L0 * np.sin(theta_c),
            "k--", lw=0.8, alpha=0.4, label=f"L={L0:.0f} shell")
    # Planet
    ax.fill(r_planet * np.cos(theta_c), r_planet * np.sin(theta_c),
            color="lightsteelblue", zorder=5)
    ax.plot(r_planet * np.cos(theta_c), r_planet * np.sin(theta_c),
            "k-", lw=0.8, zorder=6)

# ---- Animated artists ----
TRAIL = 60   # number of GC points in trail

# Left: no E
trail_L, = ax_L.plot([], [], lw=1.6, color="C0", alpha=0.85,
                      label="GC (no E)")
dot_L,   = ax_L.plot([], [], "o", color="C0", ms=8, zorder=10)
ax_L.legend(fontsize=8, loc="upper right")

# Right: with E + rotating planet arm
trail_R,  = ax_R.plot([], [], lw=1.8, color="C1", alpha=0.9,
                       label="GC (with E)")
dot_R,    = ax_R.plot([], [], "o", color="C1", ms=8, zorder=10)

# Rotating planet arm (shows nominal Ω rotation)
arm_line, = ax_R.plot([], [], "-", color="crimson", lw=2.0, alpha=0.8,
                       zorder=7, label="Planet reference (Ω)")
arm_dot,  = ax_R.plot([], [], "o", color="crimson", ms=9, zorder=8)
ax_R.legend(fontsize=8, loc="upper right")

# Time labels
txt_L = ax_L.text(0.03, 0.97, "", transform=ax_L.transAxes,
                   fontsize=8, va="top")
txt_R = ax_R.text(0.03, 0.97, "", transform=ax_R.transAxes,
                   fontsize=8, va="top")

# Measure actual drift rates for annotation
phi_noE = np.unwrap(np.arctan2(gc_noE_s[:, 1], gc_noE_s[:, 0]))
phi_E   = np.unwrap(np.arctan2(gc_s[:, 1], gc_s[:, 0]))
n  = len(t_s)
i0, i1 = n // 10, 9 * n // 10
Om_noE  = np.polyfit(t_s[i0:i1], phi_noE[i0:i1], 1)[0]
Om_E    = np.polyfit(t_s[i0:i1], phi_E[i0:i1],   1)[0]
ax_L.text(0.03, 0.06, f"Ω_drift ≈ {Om_noE:.4f}", transform=ax_L.transAxes,
          fontsize=8, color="C0")
ax_R.text(0.03, 0.06,
          f"GC Ω ≈ {Om_E:.4f}\nPlanet Ω = {Omega:.4f}",
          transform=ax_R.transAxes, fontsize=8, color="C1")

# ---- Animation ----
N_FRAMES  = 220
skip_anim = max(1, len(t_s) // N_FRAMES)

def update(frame):
    j  = frame * skip_anim
    j  = min(j, len(t_s) - 1)
    j0 = max(0, j - TRAIL)
    t_now = t_s[j]

    # Left panel — no E
    trail_L.set_data(gc_noE_s[j0:j, 0], gc_noE_s[j0:j, 1])
    dot_L.set_data([gc_noE_s[j, 0]], [gc_noE_s[j, 1]])
    txt_L.set_text(f"t = {t_now:.0f} / {T_cor:.0f}\n"
                   f"({100*t_now/T_cor:.0f}% of rotation)")

    # Right panel — with E
    trail_R.set_data(gc_s[j0:j, 0], gc_s[j0:j, 1])
    dot_R.set_data([gc_s[j, 0]], [gc_s[j, 1]])

    # Rotating planet arm at nominal Ω
    phi_planet = Omega * t_now
    px = L0 * np.cos(phi_planet)
    py = L0 * np.sin(phi_planet)
    arm_line.set_data([0, px], [0, py])
    arm_dot.set_data([px], [py])

    txt_R.set_text(f"t = {t_now:.0f} / {T_cor:.0f}\n"
                   f"({100*t_now/T_cor:.0f}% of rotation)")

    return trail_L, dot_L, trail_R, dot_R, arm_line, arm_dot, txt_L, txt_R

n_anim = min(N_FRAMES, len(t_s) // skip_anim)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=50, blit=False)

plt.tight_layout()

out = "../Figures/animate15_corotation.gif"
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=20, dpi=120)
print("Done.")
plt.show()
