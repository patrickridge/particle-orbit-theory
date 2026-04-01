"""
animate14a_aligned.py
=====================
GC bounce in a 0° (aligned) dipole — symmetric about geographic equator.
Camera rotates slowly to show the 3D bounce shape.
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
tilt  = 0.0

r0     = np.array([L0, 0.0, 0.0])
B_func = B_dipole_cartesian(M=M, tilt_deg=tilt)
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

# ---- Field lines ----
lam_fl = np.linspace(-1.25, 1.25, 300)
L_fl   = [2.0, 2.5, 3.0, 3.5]
phi_fl = np.linspace(0, 2 * np.pi, 8, endpoint=False)

# ---- Figure ----
fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#dddddd")
ax.grid(False)

# Field lines
for phi_f in phi_fl:
    for L_f in L_fl:
        r_fl = L_f * np.cos(lam_fl) ** 2
        xf   = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        yf   = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zf   = r_fl * np.sin(lam_fl)
        below = (xf**2 + yf**2 + zf**2) < 1.02**2
        xf[below] = np.nan; yf[below] = np.nan; zf[below] = np.nan
        ax.plot(xf, yf, zf, color="#555555", lw=0.5, alpha=0.35)

# Geographic equatorial ring (z=0)
theta_eq = np.linspace(0, 2 * np.pi, 120)
r_eq     = 3.8
ax.plot(r_eq * np.cos(theta_eq), r_eq * np.sin(theta_eq), np.zeros(120),
        color="#aaaaaa", lw=1.0, alpha=0.6, linestyle="--")
ax.text(r_eq + 0.15, 0, 0, "z = 0", fontsize=7, color="#888888")

# Planet sphere
u_s = np.linspace(0, 2 * np.pi, 24)
v_s = np.linspace(0, np.pi, 16)
ax.plot_surface(
    np.outer(np.cos(u_s), np.sin(v_s)),
    np.outer(np.sin(u_s), np.sin(v_s)),
    np.outer(np.ones_like(u_s), np.cos(v_s)),
    color="lightsteelblue", alpha=0.55, zorder=0
)

# Magnetic axis arrow (aligned — points straight up)
ax.quiver(0, 0, 0, 0, 0, 1.5, color="crimson", lw=1.5, arrow_length_ratio=0.12)

# Static full GC path (shows bounce envelope from frame 1)
ax.plot(gc[:, 0], gc[:, 1], gc[:, 2], lw=0.7, alpha=0.20, color="C0")

ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-3.5, 3.5)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("0°  —  aligned dipole", fontsize=11, pad=6)

# ---- Animated artists ----
N_FRAMES = 200
N        = len(t)
skip     = max(1, N // N_FRAMES)
TRAIL    = max(50, N_FRAMES // 5)   # ~1 bounce worth of animated trail

trail, = ax.plot([], [], [], lw=2.0, color="C0", label="Guiding centre")
dot,   = ax.plot([], [], [], "o", color="C0", ms=7, zorder=10,
                 markeredgecolor="white", markeredgewidth=1.2)
time_txt = ax.text2D(0.03, 0.96, "", transform=ax.transAxes, fontsize=9,
                     color="#444444")

ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()

def update(frame):
    i  = min(frame * skip, N - 1)
    i0 = max(0, i - TRAIL)

    trail.set_data(gc[i0:i, 0], gc[i0:i, 1])
    trail.set_3d_properties(gc[i0:i, 2])
    dot.set_data([gc[i, 0]], [gc[i, 1]])
    dot.set_3d_properties([gc[i, 2]])

    # Slow camera rotation
    ax.view_init(elev=20, azim=-60 + frame * 0.6)

    time_txt.set_text(f"t = {t[i]:.1f}")
    return trail, dot, time_txt

anim = animation.FuncAnimation(fig, update, frames=N_FRAMES,
                                interval=55, blit=False)
out = os.path.join(_DIR, "..", "Figures", "animate14a_aligned.gif")
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()
