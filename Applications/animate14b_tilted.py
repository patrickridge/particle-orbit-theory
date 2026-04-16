"""GC bounce in a 59 deg tilted dipole."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, B_dipole_cartesian

_DIR = os.path.dirname(os.path.abspath(__file__))

# parameters
q, m  = 1.0, 1.0
M     = 500.0
L0    = 3.0
v_mag = 1.0
pitch = np.deg2rad(45.0)
tilt  = 59.0

theta_rad    = np.radians(tilt)
cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

# start on the magnetic equatorial plane (same as 14a)
r0     = np.array([L0 * cos_t, 0.0, -L0 * sin_t])
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

print(f"Tilted {tilt:.0f}°: T_gyro={T_gyro:.3f}, T_b={T_b_est:.2f}, nsteps={nsteps}")
print("Integrating ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_zero, B_func=B_func,
                              rtol=1e-9, atol=1e-9)
gc = extract_gc(traj, t, B_func, q=q, m=m)
print("Done.")

# figure setup
fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#e0e0e0")
ax.grid(False)

# tilted field lines
lam_fl = np.linspace(-1.25, 1.25, 300)
L_fl   = [2.0, 2.5, 3.0, 3.5]
phi_fl = np.linspace(0, 2 * np.pi, 8, endpoint=False)
for phi_f in phi_fl:
    for L_f in L_fl:
        r_fl = L_f * np.cos(lam_fl) ** 2
        xm   = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        ym   = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zm   = r_fl * np.sin(lam_fl)
        xg   =  xm * cos_t + zm * sin_t
        yg   =  ym.copy()
        zg   = -xm * sin_t + zm * cos_t
        below = (xg**2 + yg**2 + zg**2) < 1.02**2
        xg[below] = np.nan; yg[below] = np.nan; zg[below] = np.nan
        ax.plot(xg, yg, zg, color="#909090", lw=0.5, alpha=0.28)

# geographic equatorial plane
phi_r = np.linspace(0, 2 * np.pi, 200)
R_eq  = 3.8
ax.plot(R_eq * np.cos(phi_r), R_eq * np.sin(phi_r), np.zeros(200),
        color="#999999", lw=1.2, alpha=0.55, linestyle="--")
ax.text(R_eq * np.cos(np.pi * 1.25),
        R_eq * np.sin(np.pi * 1.25) - 0.3, 0.15,
        "Geographic equatorial plane", fontsize=7, color="#888888")

# magnetic equatorial plane
R_mag    = 3.2
phi_m    = np.linspace(0, 2 * np.pi, 200)
x_meq    = -R_mag * np.sin(phi_m) * cos_t
y_meq    =  R_mag * np.cos(phi_m)
z_meq    =  R_mag * np.sin(phi_m) * sin_t
ax.plot(x_meq, y_meq, z_meq,
        color="crimson", lw=1.4, alpha=0.50, linestyle="--")
lbl_idx = 40
ax.text(x_meq[lbl_idx] + 0.1, y_meq[lbl_idx] + 0.1, z_meq[lbl_idx] + 0.15,
        "Magnetic equatorial plane", fontsize=7,
        color="crimson", fontweight="bold")

# planet sphere
u_s = np.linspace(0, 2 * np.pi, 24)
v_s = np.linspace(0, np.pi, 16)
ax.plot_surface(
    np.outer(np.cos(u_s), np.sin(v_s)),
    np.outer(np.sin(u_s), np.sin(v_s)),
    np.outer(np.ones_like(u_s), np.cos(v_s)),
    color="lightsteelblue", alpha=0.55, zorder=0
)

# tilted magnetic axis arrow
ax.quiver(0, 0, 0,
          2.2 * sin_t, 0, 2.2 * cos_t,
          color="crimson", lw=3.0, arrow_length_ratio=0.12, zorder=5)
ax.quiver(0, 0, 0,
          -2.2 * sin_t, 0, -2.2 * cos_t,
          color="crimson", lw=1.5, alpha=0.35, arrow_length_ratio=0.0)
ax.text(2.2 * sin_t + 0.15, 0.1, 2.2 * cos_t + 0.1,
        "Tilted magnetic axis", fontsize=8,
        color="crimson", fontweight="bold")

# static full GC path
ax.plot(gc[:, 0], gc[:, 1], gc[:, 2], lw=1.2, alpha=0.30, color="C2")

# axes
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-3.5, 3.5)
ax.set_xlabel("x", fontsize=9); ax.set_ylabel("y", fontsize=9)
ax.set_zlabel("z", fontsize=9)
ax.tick_params(labelsize=7)
ax.set_title("Tilted dipole (59° tilt)", fontsize=12, pad=8, fontweight="bold")

# animated artists
N_FRAMES = 200
N        = len(t)
skip     = max(1, N // N_FRAMES)
TRAIL    = max(50, N_FRAMES // 5)

trail, = ax.plot([], [], [], lw=2.8, color="C2", zorder=10)
dot,   = ax.plot([], [], [], "o", color="C2", ms=8, zorder=11,
                 markeredgecolor="white", markeredgewidth=1.4)

# orbit label
ax.text2D(0.04, 0.06, "Guiding-centre drift path",
          transform=ax.transAxes, fontsize=8, color="C2", fontweight="bold")

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
out = os.path.join(_DIR, "..", "Figures", "animate14b_tilted.gif")
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()
