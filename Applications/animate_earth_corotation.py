"""
animate_earth_corotation.py
===========================
Physics message (one figure, one idea):
  The corotation E×B drift would keep the guiding centre locked to the
  planet arm.  Magnetic (gradient + curvature) drifts add a small extra
  azimuthal velocity, so the real guiding centre runs slightly ahead.

Two elements:
  Crimson dot  — pure corotation: stays on the arm by construction
                 trail traces the perfect L = 3 circle
  Orange dot   — full drift (E×B + magnetic): drifts ahead of arm
                 trail shows the actual GC path

The angular gap that opens between the two dots IS the magnetic drift.

Simulation: code units (q=m=1, M=500, Ω=0.02, L=3).
Earth-inspired physical values are printed to the console.

Saves: ../Figures/animate_earth_corotation.gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_corotation, B_dipole_cartesian

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Earth reference (console only) -----------------------------------
Omega_E = 7.27e-5; R_E = 6.371e6; B_E = 3.11e-5; L = 3.0
B_eq  = B_E / L**3
v_rot = Omega_E * L * R_E
E_cor = v_rot * B_eq
print(f"Earth L=3 reference:  B={B_eq*1e6:.2f} μT  "
      f"E={E_cor*1e3:.2f} mV/m  v_ExB={v_rot/1e3:.2f} km/s")

# ---- Simulation -------------------------------------------------------
q, m = 1.0, 1.0
M    = 500.0
Omega = 0.02

B_func = B_dipole_cartesian(M=M, tilt_deg=0.0)
E_func = E_corotation(B_func, Omega=Omega)

r0     = np.array([L, 0.0, 0.0])
v_mag  = 1.0
pitch  = np.deg2rad(45.0)
B0_vec = B_func(r0, 0.0)
bhat   = B0_vec / np.linalg.norm(B0_vec)
v0     = v_mag * (np.cos(pitch) * bhat
                  + np.sin(pitch) * np.array([0.0, 1.0, 0.0]))
state0 = np.concatenate([r0, v0])

Omega_gyro = abs(q) * np.linalg.norm(B0_vec) / m
T_gyro     = 2.0 * np.pi / Omega_gyro
T_b_est    = 4.0 * r0[0] / abs(np.dot(v0, bhat))
T_cor      = 2.0 * np.pi / Omega
dt         = min(T_b_est / 300.0, 0.05 * T_gyro)
nsteps     = int(T_cor / dt) + 1
skip       = max(1, int(round(T_gyro / dt)))

print(f"T_gyro={T_gyro:.3f}  T_bounce={T_b_est:.2f}  T_cor={T_cor:.1f}")
print("Integrating ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_func, B_func=B_func,
                              rtol=1e-9, atol=1e-9)
print("Done.")

gc_E = extract_gc(traj, t, B_func, q=q, m=m)[::skip]
t_s  = t[::skip]

# Pure corotation: analytic, always on the arm
x_cor = L * np.cos(Omega * t_s)
y_cor = L * np.sin(Omega * t_s)

# Measured rates
phi_full = np.unwrap(np.arctan2(gc_E[:, 1], gc_E[:, 0]))
n = len(t_s); i0, i1 = n // 10, 9 * n // 10
Om_full = np.polyfit(t_s[i0:i1], phi_full[i0:i1], 1)[0]
print(f"Pure corotation Ω = {Omega:.4f}  |  Full drift Ω = {Om_full:.4f}  "
      f"|  ΔΩ = {Om_full - Omega:.4f}")

# ---- Figure -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

lim = 4.1
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_xticks([-3, 0, 3])
ax.set_yticks([-3, 0, 3])
ax.tick_params(labelsize=9, color="#bbbbbb", labelcolor="#888888")
for spine in ax.spines.values():
    spine.set_edgecolor("#dddddd")

theta_c = np.linspace(0, 2.0 * np.pi, 400)

# L = 3 reference circle — this IS the pure corotation path
ax.plot(L * np.cos(theta_c), L * np.sin(theta_c),
        "--", color="#cccccc", lw=1.4, zorder=1)
ax.text(L * np.cos(np.radians(38)) + 0.12,
        L * np.sin(np.radians(38)) + 0.15,
        r"$L = 3$", fontsize=9, color="#aaaaaa", zorder=1)

# Planet — simple, not dominant
ax.fill(np.cos(theta_c), np.sin(theta_c),
        color="#5599cc", alpha=0.70, zorder=5)

# Rotation symbol inside planet
ax.text(0, 0, "\u21ba", fontsize=19, color="white",
        ha="center", va="center", zorder=6, alpha=0.80)

# ---- Static labels — top left, plain text, no boxes ------------------
ax.text(0.04, 0.95, "\u25cf  Pure corotation (E\u00d7B only)",
        color="crimson", fontsize=11,
        transform=ax.transAxes, va="top")
ax.text(0.04, 0.87, "\u25cf  Full drift (E\u00d7B + magnetic)",
        color="darkorange", fontsize=11,
        transform=ax.transAxes, va="top")

# ---- Animated artists -------------------------------------------------
TRAIL = 45   # short trail — keeps figure clean

# Planet arm — thin grey reference line
arm, = ax.plot([], [], "-", color="#aaaaaa", lw=0.9, alpha=0.55, zorder=7)

# Pure corotation: arc along L-shell circle + dot at arm tip
trail_cor, = ax.plot([], [], lw=1.8, color="crimson", alpha=0.45, zorder=3)
dot_cor,   = ax.plot([], [], "o", color="white", ms=11, zorder=9,
                     markeredgecolor="crimson", markeredgewidth=2.2)

# Full drift: actual GC trail + dot
trail_full, = ax.plot([], [], lw=2.8, color="darkorange", alpha=0.95, zorder=4)
dot_full,   = ax.plot([], [], "o", color="darkorange", ms=11, zorder=10,
                      markeredgecolor="white", markeredgewidth=1.6)

ax.set_title("Corotation and magnetic drift",
             fontsize=13, fontweight="bold", pad=10)
fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.07)

# ---- Animation --------------------------------------------------------
N_FRAMES  = 200
skip_anim = max(1, len(t_s) // N_FRAMES)


def update(frame):
    j     = min(frame * skip_anim, len(t_s) - 1)
    j0    = max(0, j - TRAIL)
    t_now = t_s[j]

    phi_p  = Omega * t_now
    px, py = L * np.cos(phi_p), L * np.sin(phi_p)
    arm.set_data([0, px], [0, py])

    # Pure corotation arc traces the circle exactly
    phi_arc = np.linspace(Omega * t_s[j0], phi_p, 80)
    trail_cor.set_data(L * np.cos(phi_arc), L * np.sin(phi_arc))
    dot_cor.set_data([px], [py])

    # Full drift trail and dot
    trail_full.set_data(gc_E[j0:j, 0], gc_E[j0:j, 1])
    dot_full.set_data([gc_E[j, 0]], [gc_E[j, 1]])

    return arm, trail_cor, dot_cor, trail_full, dot_full


n_anim = min(N_FRAMES, len(t_s) // skip_anim)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=65, blit=False)

out = os.path.join(_DIR, "..", "Figures", "animate_earth_corotation.gif")
print(f"Saving → {out}")
anim.save(out, writer="pillow", fps=15, dpi=130)
print("Done.")
plt.show()
