"""
animate14_tilted.py
===================
Side-by-side animation comparing GC bounce in aligned (0°) vs
strongly tilted (59°, Uranus-like) dipole.

Left:  aligned dipole  — symmetric bounce about z=0
Right: 59° tilted dipole — asymmetric bounce, mirror points at
       different geographic latitudes north and south.

Saves: ../Figures/animate14_tilted.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, B_dipole_cartesian

# ---- Shared parameters ----
q, m  = 1.0, 1.0
M     = 500.0
L0    = 3.0
v_mag = 1.0
pitch = np.deg2rad(45.0)

tilts = [0.0, 59.0]
labels = ["0° — aligned dipole\n(symmetric bounce)", "59° — Uranus-like\n(asymmetric bounce)"]
colors = ["C0", "C2"]

results = {}

for tilt in tilts:
    theta_rad = np.radians(tilt)
    r0        = np.array([L0 * np.cos(theta_rad), 0.0, -L0 * np.sin(theta_rad)])
    B_func    = B_dipole_cartesian(M=M, tilt_deg=tilt)
    B0_vec    = B_func(r0, 0.0)
    bhat      = B0_vec / np.linalg.norm(B0_vec)
    eperp     = np.array([0.0, 1.0, 0.0])
    v0        = v_mag * (np.cos(pitch) * bhat + np.sin(pitch) * eperp)
    state0    = np.concatenate([r0, v0])

    Omega_g   = abs(q) * np.linalg.norm(B0_vec) / m
    T_gyro    = 2.0 * np.pi / Omega_g
    v_par_mag = abs(np.dot(v0, bhat))
    T_b_est   = 4.0 * L0 / v_par_mag
    dt        = min(T_b_est / 400.0, 0.05 * T_gyro)
    T_run     = 3.5 * T_b_est
    nsteps    = int(T_run / dt) + 1

    print(f"Tilt {tilt:.0f}°: T_gyro={T_gyro:.3f}, T_b={T_b_est:.2f}, nsteps={nsteps}")
    t, traj   = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                                    q=q, m=m, E_func=E_zero, B_func=B_func,
                                    rtol=1e-9, atol=1e-9)
    skip_gc   = max(1, int(round(T_gyro / dt)))
    gc        = extract_gc(traj, t, B_func, q=q, m=m)
    gc_s      = gc[::skip_gc]
    t_s       = t[::skip_gc]

    results[tilt] = dict(t=t_s, gc=gc_s, B_func=B_func, theta=theta_rad,
                         T_b=T_b_est)

print("Done integrating.")

# ---- Precompute field lines for each tilt ----
lam_fl   = np.linspace(-1.25, 1.25, 300)
L_fl     = [2.0, 2.5, 3.0, 3.5]
phi_fl   = np.linspace(0, 2 * np.pi, 8, endpoint=False)

def tilted_field_lines(theta_r):
    cos_t, sin_t = np.cos(theta_r), np.sin(theta_r)
    lines = []
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
            lines.append((xg, yg, zg))
    return lines

fl_left  = tilted_field_lines(results[0.0]["theta"])
fl_right = tilted_field_lines(results[59.0]["theta"])

# ---- Figure — two 3D subplots ----
fig = plt.figure(figsize=(12, 6))
ax_L = fig.add_subplot(121, projection="3d")
ax_R = fig.add_subplot(122, projection="3d")
fig.suptitle("Tilted dipole: effect of magnetic axis tilt on particle bounce",
             fontsize=11, y=1.01)

for ax, fl_lines, tilt, label, col in zip(
        [ax_L, ax_R], [fl_left, fl_right], tilts, labels, colors):

    for xg, yg, zg in fl_lines:
        ax.plot(xg, yg, zg, color="steelblue", lw=0.4, alpha=0.18)

    # Planet sphere
    u_s = np.linspace(0, 2*np.pi, 20)
    v_s = np.linspace(0, np.pi, 14)
    ax.plot_surface(
        np.outer(np.cos(u_s), np.sin(v_s)),
        np.outer(np.sin(u_s), np.sin(v_s)),
        np.outer(np.ones_like(u_s), np.cos(v_s)),
        color="lightsteelblue", alpha=0.5, zorder=0
    )

    # Magnetic axis arrow
    theta_r = np.radians(tilt)
    ax.quiver(0, 0, 0,
              1.2 * np.sin(theta_r), 0, 1.2 * np.cos(theta_r),
              color="crimson", lw=1.5, arrow_length_ratio=0.15)

    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-3.5, 3.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(label, fontsize=9, pad=2)
    ax.view_init(elev=20, azim=-60)

# Animated artists
TRAIL    = 80

trail_L, = ax_L.plot([], [], [], lw=1.8, color="C0")
dot_L,   = ax_L.plot([], [], [], "o", color="C0", ms=7, zorder=10)
trail_R, = ax_R.plot([], [], [], lw=1.8, color="C2")
dot_R,   = ax_R.plot([], [], [], "o", color="C2", ms=7, zorder=10)
time_txt = fig.text(0.5, 0.01, "", ha="center", fontsize=9)

# Align time arrays to same length
gc_L  = results[0.0]["gc"];   t_L  = results[0.0]["t"]
gc_R  = results[59.0]["gc"];  t_R  = results[59.0]["t"]
N     = min(len(t_L), len(t_R))
gc_L, t_L = gc_L[:N], t_L[:N]
gc_R, t_R = gc_R[:N], t_R[:N]

N_FRAMES = 200
skip     = max(1, N // N_FRAMES)

def update(frame):
    i  = frame * skip
    i  = min(i, N - 1)
    i0 = max(0, i - TRAIL)

    trail_L.set_data(gc_L[i0:i, 0], gc_L[i0:i, 1])
    trail_L.set_3d_properties(gc_L[i0:i, 2])
    dot_L.set_data([gc_L[i, 0]], [gc_L[i, 1]])
    dot_L.set_3d_properties([gc_L[i, 2]])

    trail_R.set_data(gc_R[i0:i, 0], gc_R[i0:i, 1])
    trail_R.set_3d_properties(gc_R[i0:i, 2])
    dot_R.set_data([gc_R[i, 0]], [gc_R[i, 1]])
    dot_R.set_3d_properties([gc_R[i, 2]])

    time_txt.set_text(f"t = {t_L[i]:.1f}")
    return trail_L, dot_L, trail_R, dot_R, time_txt

n_anim = min(N_FRAMES, N // skip)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=55, blit=False)

out = "../Figures/animate14_tilted.gif"
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()
