"""Side-by-side aligned vs tilted dipole GC bounce animation."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, B_dipole_cartesian

_DIR = os.path.dirname(os.path.abspath(__file__))

# shared parameters
q, m  = 1.0, 1.0
M     = 500.0
L0    = 3.0
v_mag = 1.0
pitch = np.deg2rad(45.0)
tilt  = 59.0

theta_rad    = np.radians(tilt)
cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

# simulation A - aligned dipole
r0_a     = np.array([L0, 0.0, 0.0])
B_func_a = B_dipole_cartesian(M=M, tilt_deg=0.0)
B0_a     = B_func_a(r0_a, 0.0)
bhat_a   = B0_a / np.linalg.norm(B0_a)
eperp    = np.array([0.0, 1.0, 0.0])
v0_a     = v_mag * (np.cos(pitch) * bhat_a + np.sin(pitch) * eperp)
state0_a = np.concatenate([r0_a, v0_a])

Omega_a   = abs(q) * np.linalg.norm(B0_a) / m
T_gyro_a  = 2.0 * np.pi / Omega_a
v_par_a   = abs(np.dot(v0_a, bhat_a))
T_b_a     = 4.0 * L0 / v_par_a
dt_a      = min(T_b_a / 400.0, 0.05 * T_gyro_a)
T_run_a   = 5.0 * T_b_a
nsteps_a  = int(T_run_a / dt_a) + 1

print("Aligned: integrating ...")
t_a, traj_a = simulate_orbit_ivp(state0=state0_a, dt=dt_a, nsteps=nsteps_a,
                                  q=q, m=m, E_func=E_zero, B_func=B_func_a,
                                  rtol=1e-9, atol=1e-9)
gc_a = extract_gc(traj_a, t_a, B_func_a, q=q, m=m)
print("Done.")

# simulation B - tilted dipole
r0_b     = np.array([L0 * cos_t, 0.0, -L0 * sin_t])
B_func_b = B_dipole_cartesian(M=M, tilt_deg=tilt)
B0_b     = B_func_b(r0_b, 0.0)
bhat_b   = B0_b / np.linalg.norm(B0_b)
v0_b     = v_mag * (np.cos(pitch) * bhat_b + np.sin(pitch) * eperp)
state0_b = np.concatenate([r0_b, v0_b])

Omega_b   = abs(q) * np.linalg.norm(B0_b) / m
T_gyro_b  = 2.0 * np.pi / Omega_b
v_par_b   = abs(np.dot(v0_b, bhat_b))
T_b_b     = 4.0 * L0 / v_par_b
dt_b      = min(T_b_b / 400.0, 0.05 * T_gyro_b)
T_run_b   = 5.0 * T_b_b
nsteps_b  = int(T_run_b / dt_b) + 1

print("Tilted: integrating ...")
t_b, traj_b = simulate_orbit_ivp(state0=state0_b, dt=dt_b, nsteps=nsteps_b,
                                  q=q, m=m, E_func=E_zero, B_func=B_func_b,
                                  rtol=1e-9, atol=1e-9)
gc_b = extract_gc(traj_b, t_b, B_func_b, q=q, m=m)
print("Done.")

# figure - two 3D subplots side by side
fig = plt.figure(figsize=(14, 7))

ax_a = fig.add_subplot(121, projection="3d")
ax_b = fig.add_subplot(122, projection="3d")

for ax in (ax_a, ax_b):
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#e0e0e0")
    ax.grid(False)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-3.5, 3.5)
    ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8)
    ax.set_zlabel("z", fontsize=8)
    ax.tick_params(labelsize=6)

# draw dipole field lines
lam_fl = np.linspace(-1.25, 1.25, 300)
L_fl   = [2.0, 2.5, 3.0, 3.5]
phi_fl = np.linspace(0, 2 * np.pi, 8, endpoint=False)

# aligned field lines (left panel)
for phi_f in phi_fl:
    for L_f in L_fl:
        r_fl = L_f * np.cos(lam_fl) ** 2
        xf = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        yf = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zf = r_fl * np.sin(lam_fl)
        below = (xf**2 + yf**2 + zf**2) < 1.02**2
        xf[below] = np.nan; yf[below] = np.nan; zf[below] = np.nan
        ax_a.plot(xf, yf, zf, color="#909090", lw=0.5, alpha=0.28)

# tilted field lines (right panel)
for phi_f in phi_fl:
    for L_f in L_fl:
        r_fl = L_f * np.cos(lam_fl) ** 2
        xm = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        ym = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zm = r_fl * np.sin(lam_fl)
        xg =  xm * cos_t + zm * sin_t
        yg =  ym.copy()
        zg = -xm * sin_t + zm * cos_t
        below = (xg**2 + yg**2 + zg**2) < 1.02**2
        xg[below] = np.nan; yg[below] = np.nan; zg[below] = np.nan
        ax_b.plot(xg, yg, zg, color="#909090", lw=0.5, alpha=0.28)

# geographic equatorial rings
phi_r = np.linspace(0, 2 * np.pi, 200)
R_eq  = 3.8
for ax in (ax_a, ax_b):
    ax.plot(R_eq * np.cos(phi_r), R_eq * np.sin(phi_r), np.zeros(200),
            color="#999999", lw=1.2, alpha=0.55, linestyle="--")
    ax.text(R_eq * np.cos(np.pi * 1.25),
            R_eq * np.sin(np.pi * 1.25) - 0.3, 0.15,
            "Geographic\nequatorial plane", fontsize=6, color="#888888")

# magnetic equatorial plane ring (right panel only)
R_mag = 3.2
phi_m = np.linspace(0, 2 * np.pi, 200)
x_meq = -R_mag * np.sin(phi_m) * cos_t
y_meq =  R_mag * np.cos(phi_m)
z_meq =  R_mag * np.sin(phi_m) * sin_t
ax_b.plot(x_meq, y_meq, z_meq,
          color="crimson", lw=1.4, alpha=0.50, linestyle="--")
lbl_idx = 40
ax_b.text(x_meq[lbl_idx] + 0.1, y_meq[lbl_idx] + 0.1, z_meq[lbl_idx] + 0.15,
          "Magnetic\nequatorial plane", fontsize=6,
          color="crimson", fontweight="bold")

# planet spheres
u_s = np.linspace(0, 2 * np.pi, 24)
v_s = np.linspace(0, np.pi, 16)
sphere_x = np.outer(np.cos(u_s), np.sin(v_s))
sphere_y = np.outer(np.sin(u_s), np.sin(v_s))
sphere_z = np.outer(np.ones_like(u_s), np.cos(v_s))
for ax in (ax_a, ax_b):
    ax.plot_surface(sphere_x, sphere_y, sphere_z,
                    color="lightsteelblue", alpha=0.55, zorder=0)

# magnetic axis arrows
# aligned: vertical
ax_a.quiver(0, 0, -2.2, 0, 0, 4.4,
            color="crimson", lw=3.0, arrow_length_ratio=0.12, zorder=5)
ax_a.text(0.15, 0, 2.45, "Magnetic axis", fontsize=7,
          color="crimson", fontweight="bold")

# tilted: 59 deg from vertical
ax_b.quiver(0, 0, 0,  2.2 * sin_t, 0,  2.2 * cos_t,
            color="crimson", lw=3.0, arrow_length_ratio=0.12, zorder=5)
ax_b.quiver(0, 0, 0, -2.2 * sin_t, 0, -2.2 * cos_t,
            color="crimson", lw=1.5, alpha=0.35, arrow_length_ratio=0.0)
ax_b.text(2.2 * sin_t + 0.15, 0.1, 2.2 * cos_t + 0.1,
          "Tilted magnetic\naxis (59°)", fontsize=7,
          color="crimson", fontweight="bold")

# static full GC paths
ax_a.plot(gc_a[:, 0], gc_a[:, 1], gc_a[:, 2], lw=1.2, alpha=0.30, color="C0")
ax_b.plot(gc_b[:, 0], gc_b[:, 1], gc_b[:, 2], lw=1.2, alpha=0.30, color="C2")

# titles
ax_a.set_title("Aligned dipole", fontsize=12, pad=8, fontweight="bold")
ax_b.set_title("Tilted dipole  (59°)", fontsize=12, pad=8, fontweight="bold")

# animated artists
N_FRAMES = 200
N_a = len(t_a);  skip_a = max(1, N_a // N_FRAMES)
N_b = len(t_b);  skip_b = max(1, N_b // N_FRAMES)
TRAIL = max(50, N_FRAMES // 5)

trail_a, = ax_a.plot([], [], [], lw=2.8, color="C0", zorder=10)
dot_a,   = ax_a.plot([], [], [], "o", color="C0", ms=8, zorder=11,
                     markeredgecolor="white", markeredgewidth=1.4)

trail_b, = ax_b.plot([], [], [], lw=2.8, color="C2", zorder=10)
dot_b,   = ax_b.plot([], [], [], "o", color="C2", ms=8, zorder=11,
                     markeredgecolor="white", markeredgewidth=1.4)

ax_a.text2D(0.04, 0.06, "Guiding-centre drift path",
            transform=ax_a.transAxes, fontsize=7, color="C0", fontweight="bold")
ax_b.text2D(0.04, 0.06, "Guiding-centre drift path",
            transform=ax_b.transAxes, fontsize=7, color="C2", fontweight="bold")

ax_a.view_init(elev=25, azim=-60)
ax_b.view_init(elev=25, azim=-60)
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.05)

def update(frame):
    ia  = min(frame * skip_a, N_a - 1);  ia0 = max(0, ia - TRAIL)
    ib  = min(frame * skip_b, N_b - 1);  ib0 = max(0, ib - TRAIL)

    trail_a.set_data(gc_a[ia0:ia, 0], gc_a[ia0:ia, 1])
    trail_a.set_3d_properties(gc_a[ia0:ia, 2])
    dot_a.set_data([gc_a[ia, 0]], [gc_a[ia, 1]])
    dot_a.set_3d_properties([gc_a[ia, 2]])

    trail_b.set_data(gc_b[ib0:ib, 0], gc_b[ib0:ib, 1])
    trail_b.set_3d_properties(gc_b[ib0:ib, 2])
    dot_b.set_data([gc_b[ib, 0]], [gc_b[ib, 1]])
    dot_b.set_3d_properties([gc_b[ib, 2]])

    return trail_a, dot_a, trail_b, dot_b

anim = animation.FuncAnimation(fig, update, frames=N_FRAMES,
                                interval=55, blit=False)
out = os.path.join(_DIR, "..", "Figures", "animate14_combined.gif")
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()
