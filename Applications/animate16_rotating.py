"""
animate16_rotating.py
=====================
Animation of a charged particle guiding centre in a rotating tilted dipole
magnetosphere (Neptune-like: 47° tilt, Ω = 0.02).

Shows:
  - Tilted dipole field lines rotating with the planet
  - Magnetic axis arrow rotating about rotation axis (z)
  - GC position building up as a coloured trail (plasma colourmap: old→dark)
  - Large time label showing % of planetary rotation completed
  - Planet sphere and rotation axis

Saves:  ../Figures/animate16_rotating.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import B_dipole_rotating, E_corotation

# ---- Parameters (same as test16) ----
q, m      = 1.0, 1.0
M         = 500.0
Omega     = 0.02
tilt_deg  = 47.0

B_func = B_dipole_rotating(M=M, tilt_deg=tilt_deg, Omega=Omega)
E_func = E_corotation(B_func, Omega=Omega)

L0        = 3.0
theta_rad = np.radians(tilt_deg)
r0        = np.array([L0 * np.cos(theta_rad), 0.0, -L0 * np.sin(theta_rad)])
v_mag     = 1.0
pitch     = np.deg2rad(45.0)
B0_vec    = B_func(r0, 0.0)
bhat      = B0_vec / np.linalg.norm(B0_vec)
v0        = v_mag * (np.cos(pitch) * bhat + np.sin(pitch) * np.array([0.0, 1.0, 0.0]))
state0    = np.concatenate([r0, v0])

Omega_gyro = abs(q) * np.linalg.norm(B0_vec) / m
T_gyro     = 2.0 * np.pi / Omega_gyro
v_par_mag  = abs(np.dot(v0, bhat))
T_b_est    = 4.0 * L0 / v_par_mag
T_cor      = 2.0 * np.pi / Omega

dt_anim = min(T_b_est / 200.0, 0.1 * T_gyro)
nsteps  = int(T_cor / dt_anim) + 1

print(f"T_gyro={T_gyro:.3f}, T_bounce={T_b_est:.2f}, T_cor={T_cor:.1f}")
print(f"dt_anim={dt_anim:.4f}, nsteps={nsteps}")
print("Integrating ...")
t, traj = simulate_orbit_ivp(
    state0=state0, dt=dt_anim, nsteps=nsteps,
    q=q, m=m, E_func=E_func, B_func=B_func,
    rtol=1e-8, atol=1e-8,
)
skip_gc = max(1, int(round(T_gyro / dt_anim)))
gc      = extract_gc(traj, t, B_func, q=q, m=m)
gc_s    = gc[::skip_gc]
t_s     = t[::skip_gc]
print(f"Done.  GC points: {len(t_s)}")

# ---- Precompute tilted field lines at t=0 ----
cos_tilt = np.cos(theta_rad); sin_tilt = np.sin(theta_rad)
lam_fl   = np.linspace(-1.25, 1.25, 250)
L_fl     = [2.0, 2.5, 3.0, 3.5]
phi_fl   = np.linspace(0, 2*np.pi, 8, endpoint=False)

fl_ref = []  # (xg0, yg0, zg0) in geographic frame at t=0
for phi_f in phi_fl:
    for L_f in L_fl:
        r_fl = L_f * np.cos(lam_fl)**2
        xm   = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        ym   = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zm   = r_fl * np.sin(lam_fl)
        xg0  =  xm * cos_tilt + zm * sin_tilt
        yg0  =  ym.copy()
        zg0  = -xm * sin_tilt + zm * cos_tilt
        below = (xg0**2 + yg0**2 + zg0**2) < 1.02**2
        xg0[below] = np.nan; yg0[below] = np.nan; zg0[below] = np.nan
        fl_ref.append((xg0, yg0, zg0))

# Plasma colourmap for GC trail (index → colour)
cmap_trail = cm.plasma
norm_trail  = Normalize(vmin=0, vmax=len(t_s) - 1)

# ---- Figure ----
fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")

# Planet sphere (static)
u_s = np.linspace(0, 2*np.pi, 24)
v_s = np.linspace(0, np.pi, 16)
ax.plot_surface(
    np.outer(np.cos(u_s), np.sin(v_s)),
    np.outer(np.sin(u_s), np.sin(v_s)),
    np.outer(np.ones_like(u_s), np.cos(v_s)),
    color="lightgray", alpha=0.8, zorder=0
)

ax_len = 0.9 * L0
# Rotation axis (static)
ax.quiver(0, 0, 0, 0, 0, ax_len,
          color="dimgray", lw=2, arrow_length_ratio=0.15)
ax.text(0, 0, ax_len * 1.1, "Ω", fontsize=10, color="dimgray", ha="center")

# ---- Mutable artists ----
fl_artists = []
for _ in fl_ref:
    line, = ax.plot([], [], [], color="steelblue", lw=0.5, alpha=0.22)
    fl_artists.append(line)

# GC coloured trail — drawn as a single poly-segment updated each frame
gc_trail,  = ax.plot([], [], [], lw=1.8, color=cmap_trail(0.7), alpha=0.85)
gc_dot,    = ax.plot([], [], [], "o", color="white", ms=8, zorder=12,
                     markeredgecolor="C1", markeredgewidth=2)

mag_ax_q = [ax.quiver(0, 0, 0,
                       ax_len * np.sin(theta_rad), 0, ax_len * np.cos(theta_rad),
                       color="crimson", lw=2, arrow_length_ratio=0.18)]

# Large time display
time_txt  = ax.text2D(0.5, 0.97, "", transform=ax.transAxes,
                       fontsize=11, ha="center", va="top",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_zlim(-3.5, 3.5)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title(
    f"Rotating tilted dipole  ({tilt_deg:.0f}° tilt, Ω={Omega})\n"
    "GC orbit over one full planetary rotation",
    fontsize=10
)
ax.view_init(elev=25, azim=-55)

ax.legend(handles=[
    Line2D([0], [0], color="steelblue", lw=1, label="Field lines (rotating)"),
    Line2D([0], [0], color=cmap_trail(0.7), lw=2, label="Guiding centre"),
    Line2D([0], [0], color="crimson",   lw=2, label=f"Magnetic axis ({tilt_deg:.0f}°)"),
], fontsize=8, loc="upper left")

# ---- Helper ----
def rotate_z(x, y, angle):
    c, s = np.cos(angle), np.sin(angle)
    return x * c - y * s, x * s + y * c

# Subsampling
N_FRAMES  = 220
skip_anim = max(1, len(t_s) // N_FRAMES)

def update(frame):
    j     = frame * skip_anim
    j     = min(j, len(t_s) - 1)
    t_now = t_s[j]
    angle = Omega * t_now

    # Rotate field lines
    for artist, (xg0, yg0, zg0) in zip(fl_artists, fl_ref):
        xr, yr = rotate_z(xg0, yg0, angle)
        artist.set_data(xr, yr)
        artist.set_3d_properties(zg0)

    # Rotate magnetic axis arrow
    mag_ax_q[0].remove()
    mx = ax_len * np.sin(theta_rad) * np.cos(angle)
    my = ax_len * np.sin(theta_rad) * np.sin(angle)
    mz = ax_len * np.cos(theta_rad)
    mag_ax_q[0] = ax.quiver(0, 0, 0, mx, my, mz,
                             color="crimson", lw=2, arrow_length_ratio=0.18)

    # GC trail — colour changes from dark (old) to bright (new)
    col = cmap_trail(norm_trail(j))
    gc_trail.set_color(col)
    gc_trail.set_data(gc_s[:j+1, 0], gc_s[:j+1, 1])
    gc_trail.set_3d_properties(gc_s[:j+1, 2])

    gc_dot.set_data([gc_s[j, 0]], [gc_s[j, 1]])
    gc_dot.set_3d_properties([gc_s[j, 2]])

    pct = 100 * t_now / T_cor
    time_txt.set_text(
        f"t = {t_now:.0f} / {T_cor:.0f}  —  {pct:.0f}% of rotation"
    )
    return fl_artists + [gc_trail, gc_dot, time_txt]

n_anim = min(N_FRAMES, len(t_s) // skip_anim)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=50, blit=False)

out = "../Figures/animate16_rotating.gif"
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=20, dpi=120)
print("Done.")
plt.show()
