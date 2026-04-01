"""
animate14a_aligned.py
=====================
GC bounce in a 0° (aligned) dipole.
Pair with animate14b_tilted.gif — same physics, same starting conditions
relative to the magnetic axis. In the aligned case the magnetic and geographic
axes coincide, so the drift orbit sits in the geographic equatorial plane.

Saves: ../Figures/animate14a_aligned.gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, B_dipole_cartesian

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Parameters ----
q, m  = 1.0, 1.0
M     = 500.0
L0    = 3.0
v_mag = 1.0
pitch = np.deg2rad(45.0)

# Start on the magnetic equatorial plane (same relative position as 14b)
r0     = np.array([L0, 0.0, 0.0])
B_func = B_dipole_cartesian(M=M, tilt_deg=0.0)
B0_vec = B_func(r0, 0.0)
bhat   = B0_vec / np.linalg.norm(B0_vec)
eperp  = np.array([0.0, 1.0, 0.0])
v0     = v_mag * (np.cos(pitch) * bhat + np.sin(pitch) * eperp)
state0 = np.concatenate([r0, v0])

Omega_g   = abs(q) * np.linalg.norm(B0_vec) / m
T_gyro    = 2.0 * np.pi / Omega_g
v_par_mag = abs(np.dot(v0, bhat))
T_b_est   = 4.0 * L0 / v_par_mag
dt        = min(T_b_est / 400.0, 0.05 * T_gyro)
T_run     = 5.0 * T_b_est
nsteps    = int(T_run / dt) + 1

print(f"Aligned: T_gyro={T_gyro:.3f}, T_b={T_b_est:.2f}, nsteps={nsteps}")
print("Integrating ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_zero, B_func=B_func,
                              rtol=1e-9, atol=1e-9)
gc = extract_gc(traj, t, B_func, q=q, m=m)
print("Done.")

# ---- Figure ----
fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#e0e0e0")
ax.grid(False)

# ---- Field lines — faint, context only ----
lam_fl = np.linspace(-1.25, 1.25, 300)
L_fl   = [2.0, 2.5, 3.0, 3.5]
phi_fl = np.linspace(0, 2 * np.pi, 8, endpoint=False)
for phi_f in phi_fl:
    for L_f in L_fl:
        r_fl = L_f * np.cos(lam_fl) ** 2
        xf   = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        yf   = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zf   = r_fl * np.sin(lam_fl)
        below = (xf**2 + yf**2 + zf**2) < 1.02**2
        xf[below] = np.nan; yf[below] = np.nan; zf[below] = np.nan
        ax.plot(xf, yf, zf, color="#909090", lw=0.5, alpha=0.28)

# ---- Geographic equatorial plane ring (z = 0) ----
phi_r = np.linspace(0, 2 * np.pi, 200)
R_eq  = 3.8
ax.plot(R_eq * np.cos(phi_r), R_eq * np.sin(phi_r), np.zeros(200),
        color="#999999", lw=1.2, alpha=0.55, linestyle="--")
ax.text(R_eq * np.cos(np.pi * 1.25),
        R_eq * np.sin(np.pi * 1.25) - 0.3, 0.15,
        "Geographic equatorial plane", fontsize=7, color="#888888")

# ---- Planet sphere ----
u_s = np.linspace(0, 2 * np.pi, 24)
v_s = np.linspace(0, np.pi, 16)
ax.plot_surface(
    np.outer(np.cos(u_s), np.sin(v_s)),
    np.outer(np.sin(u_s), np.sin(v_s)),
    np.outer(np.ones_like(u_s), np.cos(v_s)),
    color="lightsteelblue", alpha=0.55, zorder=0
)

# ---- Magnetic axis arrow — prominent, labelled ----
ax.quiver(0, 0, -2.2, 0, 0, 4.4,
          color="crimson", lw=3.0, arrow_length_ratio=0.12, zorder=5)
ax.text(0.15, 0, 2.45, "Magnetic axis", fontsize=8,
        color="crimson", fontweight="bold")

# ---- Static full GC path — bounce envelope visible from frame 1 ----
ax.plot(gc[:, 0], gc[:, 1], gc[:, 2], lw=1.2, alpha=0.30, color="C0")

# ---- Axes ----
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-3.5, 3.5)
ax.set_xlabel("x", fontsize=9); ax.set_ylabel("y", fontsize=9)
ax.set_zlabel("z", fontsize=9)
ax.tick_params(labelsize=7)
ax.set_title("Aligned dipole", fontsize=12, pad=8, fontweight="bold")

# ---- Animated artists ----
N_FRAMES = 200
N        = len(t)
skip     = max(1, N // N_FRAMES)
TRAIL    = max(50, N_FRAMES // 5)

trail, = ax.plot([], [], [], lw=2.8, color="C0", zorder=10)
dot,   = ax.plot([], [], [], "o", color="C0", ms=8, zorder=11,
                 markeredgecolor="white", markeredgewidth=1.4)

# Direct orbit label (replaces legend)
ax.text2D(0.04, 0.06, "Guiding-centre drift path",
          transform=ax.transAxes, fontsize=8, color="C0", fontweight="bold")

ax.view_init(elev=25, azim=-60)
fig.tight_layout()

def update(frame):
    i  = min(frame * skip, N - 1)
    i0 = max(0, i - TRAIL)

    trail.set_data(gc[i0:i, 0], gc[i0:i, 1])
    trail.set_3d_properties(gc[i0:i, 2])
    dot.set_data([gc[i, 0]], [gc[i, 1]])
    dot.set_3d_properties([gc[i, 2]])

    return trail, dot

anim = animation.FuncAnimation(fig, update, frames=N_FRAMES,
                                interval=55, blit=False)
out = os.path.join(_DIR, "..", "Figures", "animate14a_aligned.gif")
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()
