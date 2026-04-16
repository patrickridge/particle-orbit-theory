"""Animate corotation E x B drift vs magnetic drift only."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, E_corotation, B_dipole_cartesian

_DIR = os.path.dirname(os.path.abspath(__file__))

# parameters
q, m  = 1.0, 1.0
M     = 500.0
Omega = 0.02          # planet rotation rate

B_func   = B_dipole_cartesian(M=M, tilt_deg=0.0)
E_func   = E_corotation(B_func, Omega=Omega)

r0    = np.array([3.0, 0.0, 0.0])
v_mag = 1.0
pitch = np.deg2rad(45.0)

B0_vec = B_func(r0, 0.0)
bhat   = B0_vec / np.linalg.norm(B0_vec)
v0     = v_mag * (np.cos(pitch) * bhat
                  + np.sin(pitch) * np.array([0.0, 1.0, 0.0]))
state0 = np.concatenate([r0, v0])

Omega_gyro = abs(q) * np.linalg.norm(B0_vec) / m
T_gyro     = 2.0 * np.pi / Omega_gyro
T_b_est    = 4.0 * r0[0] / abs(np.dot(v0, bhat))
T_cor      = 2.0 * np.pi / Omega          # one full planetary rotation
L0         = np.linalg.norm(r0)

dt     = min(T_b_est / 300.0, 0.05 * T_gyro)
nsteps = int(T_cor / dt) + 1
skip   = max(1, int(round(T_gyro / dt)))  # one point per gyration

print(f"T_gyro={T_gyro:.3f}  T_bounce={T_b_est:.2f}  T_cor={T_cor:.1f}")
print(f"dt={dt:.4f}  nsteps={nsteps}  skip={skip}")

print("Integrating — with E ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_func, B_func=B_func,
                              rtol=1e-9, atol=1e-9)
print("Integrating — no E ...")
_, traj_noE = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                                  q=q, m=m, E_func=E_zero, B_func=B_func,
                                  rtol=1e-9, atol=1e-9)
print("Done.")

gc_E   = extract_gc(traj,     t, B_func, q=q, m=m)[::skip]
gc_noE = extract_gc(traj_noE, t, B_func, q=q, m=m)[::skip]
t_s    = t[::skip]

# Measure actual angular drift rates for annotations
phi_E   = np.unwrap(np.arctan2(gc_E[:, 1],   gc_E[:, 0]))
phi_noE = np.unwrap(np.arctan2(gc_noE[:, 1], gc_noE[:, 0]))
n = len(t_s); i0, i1 = n // 10, 9 * n // 10
Om_E   = np.polyfit(t_s[i0:i1], phi_E[i0:i1],   1)[0]
Om_noE = np.polyfit(t_s[i0:i1], phi_noE[i0:i1], 1)[0]
print(f"Drift rates:  no-E = {Om_noE:.4f}   with-E = {Om_E:.4f}   planet = {Omega}")

# figure setup
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

lim = 4.2
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_xlabel("x", fontsize=10); ax.set_ylabel("y", fontsize=10)
ax.tick_params(labelsize=9)

# L-shell guide circle
theta_c = np.linspace(0, 2 * np.pi, 300)
ax.plot(L0 * np.cos(theta_c), L0 * np.sin(theta_c),
        "--", color="#cccccc", lw=1.0, zorder=1)

# Planet
r_planet = 1.0
ax.fill(r_planet * np.cos(theta_c), r_planet * np.sin(theta_c),
        color="lightsteelblue", zorder=5)
ax.plot(r_planet * np.cos(theta_c), r_planet * np.sin(theta_c),
        color="#666666", lw=0.8, zorder=6)

# faint full paths for reference
ax.plot(gc_noE[:, 0], gc_noE[:, 1],
        color="#bbbbbb", lw=0.8, alpha=0.40, zorder=2)
ax.plot(gc_E[:, 0], gc_E[:, 1],
        color="darkorange", lw=0.8, alpha=0.20, zorder=2)

# animated artists
TRAIL = 80

# Planet rotation arm
arm,     = ax.plot([], [], "-", color="crimson", lw=2.5, alpha=0.9,  zorder=7)
arm_dot, = ax.plot([], [], "o", color="crimson", ms=10,               zorder=8)

# no-E guiding centre
trail_noE, = ax.plot([], [], lw=1.8, color="#888888", alpha=0.70, zorder=3)
dot_noE,   = ax.plot([], [], "o", color="#888888", ms=9, zorder=9,
                     markeredgecolor="white", markeredgewidth=1.2)

# with-E guiding centre
trail_E, = ax.plot([], [], lw=3.2, color="darkorange", alpha=1.0, zorder=4)
dot_E,   = ax.plot([], [], "o", color="darkorange", ms=13, zorder=10,
                   markeredgecolor="white", markeredgewidth=1.6)

# legend
ax.plot([], [], "-",  color="#888888",  lw=1.8, label="Magnetic drift only")
ax.plot([], [], "-",  color="darkorange", lw=3.2, label=r"With corotation E×B drift")
ax.plot([], [], "-",  color="crimson",  lw=2.5, label="Planet rotation")
ax.legend(fontsize=10, loc="upper right", framealpha=0.93,
          edgecolor="#dddddd")

# title
ax.set_title("Corotation E×B drift", fontsize=14, fontweight="bold", pad=10)

fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.08)

# animation
N_FRAMES  = 220
skip_anim = max(1, len(t_s) // N_FRAMES)

def update(frame):
    j     = min(frame * skip_anim, len(t_s) - 1)
    j0    = max(0, j - TRAIL)
    t_now = t_s[j]

    # Planet arm rotates at Omega
    phi_p = Omega * t_now
    px = L0 * np.cos(phi_p);  py = L0 * np.sin(phi_p)
    arm.set_data([0, px], [0, py])
    arm_dot.set_data([px], [py])

    # No-E trail + dot
    trail_noE.set_data(gc_noE[j0:j, 0], gc_noE[j0:j, 1])
    dot_noE.set_data([gc_noE[j, 0]], [gc_noE[j, 1]])

    # With-E trail + dot
    trail_E.set_data(gc_E[j0:j, 0], gc_E[j0:j, 1])
    dot_E.set_data([gc_E[j, 0]], [gc_E[j, 1]])

    return arm, arm_dot, trail_noE, dot_noE, trail_E, dot_E

n_anim = min(N_FRAMES, len(t_s) // skip_anim)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=50, blit=False)

out = os.path.join(_DIR, "..", "Figures", "animate15_corotation.gif")
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=20, dpi=120)
print("Done.")
plt.show()
