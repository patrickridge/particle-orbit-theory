import os
"""
animate16_rotating.py
=====================
Physics message (one figure, one idea):
  When the dipole is tilted AND rotating, the magnetic geometry changes
  with time in the lab frame.  The guiding-centre orbit is no longer a
  simple ring — it traces a complex asymmetric 3D path.

Visual priorities:
  1. GC trail colour varies with time (plasma: dark=early, bright=recent)
     so the audience can read the direction and history of the motion.
  2. Eight field lines (two L-shells, four azimuths) give enough 3D shape
     without dominating the figure.
  3. Two labels only: rotation axis, tilted magnetic axis.
  4. Camera angle chosen to make the tilt immediately obvious.

Saves: ../Figures/animate16_rotating.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import B_dipole_rotating, E_corotation

# ---- Parameters -------------------------------------------------------
q, m     = 1.0, 1.0
M        = 500.0
Omega    = 0.02
tilt_deg = 47.0

B_func = B_dipole_rotating(M=M, tilt_deg=tilt_deg, Omega=Omega)
E_func = E_corotation(B_func, Omega=Omega)

L0        = 3.0
theta_rad = np.radians(tilt_deg)
r0        = np.array([L0 * np.cos(theta_rad), 0.0, -L0 * np.sin(theta_rad)])
v_mag     = 1.0
pitch     = np.deg2rad(45.0)
B0_vec    = B_func(r0, 0.0)
bhat      = B0_vec / np.linalg.norm(B0_vec)
v0        = v_mag * (np.cos(pitch) * bhat
                     + np.sin(pitch) * np.array([0.0, 1.0, 0.0]))
state0    = np.concatenate([r0, v0])

Omega_gyro = abs(q) * np.linalg.norm(B0_vec) / m
T_gyro     = 2.0 * np.pi / Omega_gyro
v_par_mag  = abs(np.dot(v0, bhat))
T_b_est    = 4.0 * L0 / v_par_mag
T_cor      = 2.0 * np.pi / Omega

dt_anim = min(T_b_est / 200.0, 0.1 * T_gyro)
nsteps  = int(T_cor / dt_anim) + 1

print(f"T_gyro={T_gyro:.3f}, T_bounce={T_b_est:.2f}, T_cor={T_cor:.1f}")
print("Integrating ...")
t, traj = simulate_orbit_ivp(
    state0=state0, dt=dt_anim, nsteps=nsteps,
    q=q, m=m, E_func=E_func, B_func=B_func,
    rtol=1e-8, atol=1e-8,
)
skip_gc = max(1, int(round(T_gyro / dt_anim)))
gc_s    = extract_gc(traj, t, B_func, q=q, m=m)[::skip_gc]
t_s     = t[::skip_gc]
print(f"Done.  GC points: {len(t_s)}")

# ---- Field lines — two L-shells, four azimuths = 8 lines -------------
cos_tilt = np.cos(theta_rad)
sin_tilt = np.sin(theta_rad)
lam_fl   = np.linspace(-1.3, 1.3, 220)
L_fl     = [2.5, 3.5]
phi_fl   = np.linspace(0, 2 * np.pi, 4, endpoint=False)

fl_ref = []
for phi_f in phi_fl:
    for L_f in L_fl:
        r_fl = L_f * np.cos(lam_fl) ** 2
        xm   = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        ym   = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zm   = r_fl * np.sin(lam_fl)
        xg0  =  xm * cos_tilt + zm * sin_tilt
        yg0  =  ym.copy()
        zg0  = -xm * sin_tilt + zm * cos_tilt
        below = (xg0**2 + yg0**2 + zg0**2) < 1.02**2
        xg0[below] = np.nan; yg0[below] = np.nan; zg0[below] = np.nan
        fl_ref.append((xg0, yg0, zg0))

# ---- Trail colour map — plasma: dark purple (old) → bright (recent) --
cmap_trail = plt.cm.plasma


def make_segments(xs, ys, zs):
    """Pack x/y/z arrays into (N-1, 2, 3) segment array for Line3DCollection."""
    pts  = np.c_[xs, ys, zs].reshape(-1, 1, 3)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


# ---- Figure -----------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.02, 0.02, 0.96, 0.89], projection="3d")
ax.set_facecolor("white")

# Planet sphere
u_s = np.linspace(0, 2 * np.pi, 20)
v_s = np.linspace(0, np.pi, 14)
ax.plot_surface(
    np.outer(np.cos(u_s), np.sin(v_s)),
    np.outer(np.sin(u_s), np.sin(v_s)),
    np.outer(np.ones_like(u_s), np.cos(v_s)),
    color="#cccccc", alpha=0.55, zorder=0,
)

ax_len = 0.85 * L0

# Rotation axis — thin grey line + small arrow tip
ax.plot([0, 0], [0, 0], [-ax_len, ax_len * 1.15],
        color="#999999", lw=1.4, zorder=2)
ax.quiver(0, 0, ax_len * 0.95, 0, 0, ax_len * 0.25,
          color="#999999", lw=1.2, arrow_length_ratio=0.6, zorder=2)
ax.text(0.22, 0, ax_len * 1.22, "Rotation axis",
        fontsize=9, color="#777777", ha="left", va="bottom")

# ---- Mutable artists --------------------------------------------------

# Eight field lines (rotated each frame)
fl_artists = []
for _ in fl_ref:
    line, = ax.plot([], [], [],
                    color="#4477aa", lw=0.9, alpha=0.28, zorder=1)
    fl_artists.append(line)

# GC trail — replaced each frame with a fresh Line3DCollection
trail_coll = [None]   # mutable reference so update() can remove previous

# GC current-position dot
gc_dot, = ax.plot([], [], [], "o",
                  color=cmap_trail(0.92), ms=9, zorder=6,
                  markeredgecolor="white", markeredgewidth=1.5)

# Magnetic axis arrow (rotates with planet)
mag_ax_q = [ax.quiver(
    0, 0, 0,
    ax_len * np.sin(theta_rad), 0, ax_len * np.cos(theta_rad),
    color="crimson", lw=2.2, arrow_length_ratio=0.20, zorder=4,
)]

# ---- Axes — clean 3D frame, no numeric ticks -------------------------
ax.set_xlim(-4.2, 4.2)
ax.set_ylim(-4.2, 4.2)
ax.set_zlim(-3.2, 3.2)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("#eeeeee")
ax.yaxis.pane.set_edgecolor("#eeeeee")
ax.zaxis.pane.set_edgecolor("#eeeeee")

fig.suptitle("Guiding-centre motion in a rotating tilted dipole",
             fontsize=13, fontweight="bold", y=0.98)

ax.view_init(elev=20, azim=-70)

# ---- Two direct labels only ------------------------------------------
ax.text2D(0.78, 0.86, f"Tilted magnetic axis ({tilt_deg:.0f}°)",
          transform=ax.transAxes, fontsize=10, color="crimson",
          ha="right", va="top")
ax.text2D(0.78, 0.79, "Guiding-centre orbit",
          transform=ax.transAxes, fontsize=10, color=cmap_trail(0.75),
          ha="right", va="top")

# Rotation progress — tiny, bottom-right
time_txt = ax.text2D(0.97, 0.03, "0%",
                     transform=ax.transAxes, fontsize=9,
                     ha="right", va="bottom", color="#888888")

# Small colorbar showing trail = time (dark early, bright recent)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize as MplNorm

# Figures directory — resolved relative to this script.
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)
_sm = ScalarMappable(cmap=cmap_trail, norm=MplNorm(vmin=0, vmax=1))
_sm.set_array([])
cbar_ax = fig.add_axes([0.73, 0.91, 0.22, 0.018])
cbar = fig.colorbar(_sm, cax=cbar_ax, orientation="horizontal")
cbar.set_ticks([0.0, 1.0])
cbar.set_ticklabels([" ", " "])
cbar.ax.tick_params(labelsize=8, colors="#555555", length=0)
cbar.outline.set_edgecolor("#cccccc")
cbar.ax.set_title("time", fontsize=8, color="#555555", pad=3)


# ---- Helper -----------------------------------------------------------
def rotate_z(x, y, angle):
    c, s = np.cos(angle), np.sin(angle)
    return x * c - y * s, x * s + y * c


# ---- Animation --------------------------------------------------------
N_FRAMES  = 200
skip_anim = max(1, len(t_s) // N_FRAMES)


def update(frame):
    j     = min(frame * skip_anim, len(t_s) - 1)
    t_now = t_s[j]
    angle = Omega * t_now

    # Rotate eight field lines
    for artist, (xg0, yg0, zg0) in zip(fl_artists, fl_ref):
        xr, yr = rotate_z(xg0, yg0, angle)
        artist.set_data(xr, yr)
        artist.set_3d_properties(zg0)

    # Rotate magnetic axis
    mag_ax_q[0].remove()
    mx = ax_len * np.sin(theta_rad) * np.cos(angle)
    my = ax_len * np.sin(theta_rad) * np.sin(angle)
    mz = ax_len * np.cos(theta_rad)
    mag_ax_q[0] = ax.quiver(0, 0, 0, mx, my, mz,
                             color="crimson", lw=2.2,
                             arrow_length_ratio=0.20, zorder=4)

    # Remove previous trail collection and draw a fresh one
    if trail_coll[0] is not None:
        trail_coll[0].remove()
        trail_coll[0] = None

    if j > 1:
        xs = gc_s[:j + 1, 0]
        ys = gc_s[:j + 1, 1]
        zs = gc_s[:j + 1, 2]
        segs   = make_segments(xs, ys, zs)
        n_segs = len(segs)
        colors = cmap_trail(np.linspace(0.15, 0.92, n_segs))
        coll   = Line3DCollection(segs, colors=colors,
                                  linewidth=2.0, zorder=5)
        ax.add_collection3d(coll)
        trail_coll[0] = coll

    # Current position dot — matches bright end of colormap
    gc_dot.set_data([gc_s[j, 0]], [gc_s[j, 1]])
    gc_dot.set_3d_properties([gc_s[j, 2]])

    time_txt.set_text(f"{100.0 * t_now / T_cor:.0f}%")

    return fl_artists + [gc_dot, time_txt]


n_anim = min(N_FRAMES, len(t_s) // skip_anim)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=65, blit=False)

out = os.path.join(_FIG, "animate16_rotating.gif")
print(f"Saving → {out}")
anim.save(out, writer="pillow", fps=15, dpi=130)
print("Done.")
plt.show()
