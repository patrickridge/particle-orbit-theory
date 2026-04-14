import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from matplotlib.colors import Normalize
from matplotlib import cm

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, B_dipole_cartesian

# Figures directory — resolved relative to this script, so the script runs correctly from any working directory.
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

# =============================================================
# Test 14: Tilted dipole — planetary application
#
# Planetary reference tilts (magnetic vs rotation axis):
#   Earth:   ~11°  (nearly aligned — nearly symmetric bounce)
#   Neptune: ~47°  (strongly tilted — clearly asymmetric bounce)
#   Uranus:  ~59°  (most extreme — very asymmetric bounce)
#
# Each orbit starts at the MAGNETIC equatorial plane for its tilt:
#   r0 = L*(cosθ, 0, −sinθ)   (rotation of (3,0,0) around y by θ)
# This ensures equivalent initial conditions for all tilts —
# each particle starts at distance L=3 from the magnetic equator.
# The z(t) plot is in geographic coordinates, so the bounce appears
# asymmetric because the magnetic and geographic equators differ.
#
# Two separate simulation runs:
#   (1) All three tilts for 7 bounce periods — used for z(t) plot,
#       which shows long-time drift-shell modulation of the bounce.
#   (2) 47° only for 4 bounce periods — used for the 3D GC orbit
#       figure, which needs a short clean trajectory to be legible.
#
# Timestep strategy: dt = min(T_b/1000, safety*T_gyro) with
# safety=0.05 and rtol=atol=1e-9 to keep gyration resolved even
# at mirror points where B is stronger than at the start.
# =============================================================

q, m   = 1.0, 1.0
M      = 500.0
safety = 0.05
L0     = 3.0      # L-shell distance (code units)

v_mag = 1.0
pitch = np.deg2rad(45.0)

tilts_deg = [0.0, 47.0, 59.0]
labels    = ["0° (aligned)", "47° (Neptune-like)", "59° (Uranus-like)"]
colors    = ["C0", "C1", "C2"]


# ======================================================================
# Run 1: All three tilts, long duration (7 bounce periods) for z(t)
# ======================================================================
trajs      = {}
ts         = {}
step_skips = {}
B_funcs    = {}

for tilt in tilts_deg:
    # Starting position: magnetic equatorial plane for this tilt
    # r0 = L*(cosθ, 0, -sinθ) — rotation of (L,0,0) around y-axis by tilt angle
    theta_rad = np.radians(tilt)
    r0 = np.array([L0 * np.cos(theta_rad), 0.0, -L0 * np.sin(theta_rad)])

    B_func = B_dipole_cartesian(M=M, tilt_deg=tilt)
    B0_vec = B_func(r0, 0.0)
    bhat   = B0_vec / np.linalg.norm(B0_vec)
    eperp  = np.array([0.0, 1.0, 0.0])

    v0     = v_mag * (np.cos(pitch) * bhat + np.sin(pitch) * eperp)
    state0 = np.concatenate([r0, v0])

    v_par_mag = abs(np.dot(v0, bhat))
    Omega     = abs(q) * np.linalg.norm(B0_vec) / m
    T_gyro    = 2.0 * np.pi / Omega
    T_b_est   = 4.0 * L0 / v_par_mag
    T_run     = 7.0 * T_b_est
    dt        = min(T_b_est / 1000.0, safety * T_gyro)
    nsteps    = int(T_run / dt) + 1

    print(f"\n[Run 1] Tilt = {tilt:.0f}°,  r0 = ({r0[0]:.2f}, 0, {r0[2]:.2f})")
    print(f"  T_gyro={T_gyro:.3f}, T_b_est={T_b_est:.2f}, T_b/T_g={T_b_est/T_gyro:.1f}")
    print(f"  dt={dt:.5f}, nsteps={nsteps}")

    t, traj = simulate_orbit_ivp(
        state0=state0, dt=dt, nsteps=nsteps,
        q=q, m=m, E_func=E_zero, B_func=B_func,
        rtol=1e-10, atol=1e-10,
    )
    print("  done.")
    trajs[tilt]      = traj
    ts[tilt]         = t
    step_skips[tilt] = max(1, int(round(T_gyro / dt)))
    B_funcs[tilt]    = B_func


# ======================================================================
# Run 2: 59° (Uranus-like) — matches animate14b for 3D figure
# ======================================================================
tilt_show  = 59.0
theta_3d   = np.radians(tilt_show)
r0_3d      = np.array([L0 * np.cos(theta_3d), 0.0, -L0 * np.sin(theta_3d)])
B_func_3d  = B_dipole_cartesian(M=M, tilt_deg=tilt_show)
B0_vec_3d  = B_func_3d(r0_3d, 0.0)
bhat_3d    = B0_vec_3d / np.linalg.norm(B0_vec_3d)
eperp_3d   = np.array([0.0, 1.0, 0.0])
v0_3d      = v_mag * (np.cos(pitch) * bhat_3d + np.sin(pitch) * eperp_3d)
state0_3d  = np.concatenate([r0_3d, v0_3d])

v_par_3d  = abs(np.dot(v0_3d, bhat_3d))
Omega_3d  = abs(q) * np.linalg.norm(B0_vec_3d) / m
T_gyro_3d = 2.0 * np.pi / Omega_3d
T_b_3d    = 4.0 * L0 / v_par_3d
T_run_3d  = 5.0 * T_b_3d       # same as animate14b
dt_3d     = min(T_b_3d / 1000.0, safety * T_gyro_3d)
nsteps_3d = int(T_run_3d / dt_3d) + 1
skip_3d   = max(1, int(round(T_gyro_3d / dt_3d)))

print(f"\n[Run 2] Tilt = {tilt_show:.0f}° (short run for 3D figure)")
print(f"  dt={dt_3d:.5f}, nsteps={nsteps_3d}")

t_short, traj_short = simulate_orbit_ivp(
    state0=state0_3d, dt=dt_3d, nsteps=nsteps_3d,
    q=q, m=m, E_func=E_zero, B_func=B_func_3d,
    rtol=1e-10, atol=1e-10,
)
print("  done.")


# ======================================================================
# Plot 1: Field lines — aligned vs Neptune-like tilt
#
# No legend entries inside field_lines_xz — built from proxy artists
# outside to avoid duplicates.
# ======================================================================
L_shells = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

def field_lines_xz(tilt_deg, ax, title):
    """Draw field lines, planet, and axes.  No legend entries added here."""
    tilt_r = np.deg2rad(tilt_deg)
    ct, st = np.cos(tilt_r), np.sin(tilt_r)

    for L in L_shells:
        lam_E = np.arccos(np.sqrt(1.0 / L))
        lam   = np.linspace(-lam_E, lam_E, 600)
        r_m   = L * np.cos(lam)**2
        x_m   = r_m * np.cos(lam)
        z_m   = r_m * np.sin(lam)
        x_geo =  x_m * ct + z_m * st
        z_geo = -x_m * st + z_m * ct
        ax.plot(x_geo, z_geo, color="steelblue", lw=1.3)

        if L >= 3.0:
            ax.text(L * ct + 0.15, -L * st + 0.12,
                    f"L={int(L)}", fontsize=7,
                    color="steelblue", ha="left", va="bottom")

    # Planet
    theta_c = np.linspace(0, 2 * np.pi, 300)
    ax.fill(np.cos(theta_c), np.sin(theta_c), color="lightgray", zorder=5)
    ax.plot(np.cos(theta_c), np.sin(theta_c), "k-", lw=0.8, zorder=6)

    # Geographic rotation axis
    ax.axvline(0, color="gray", lw=0.8, ls=":")

    if tilt_deg != 0.0:
        # Magnetic axis
        ax_len = 6.0
        ax.annotate("", xy=(ax_len * st, ax_len * ct),
                    xytext=(-ax_len * st, -ax_len * ct),
                    arrowprops=dict(arrowstyle="-", color="crimson",
                                   lw=1.2, ls="--"))

        # Magnetic equatorial plane
        eq_len = 6.5
        ax.plot([-eq_len * ct,  eq_len * ct],
                [ eq_len * st, -eq_len * st],
                color="crimson", lw=0.9, ls=(0, (2, 3)), alpha=0.7)

        # Particle start at magnetic equatorial plane
        theta_start = np.radians(tilt_deg)
        r0_start = np.array([L0 * np.cos(theta_start), 0.0, -L0 * np.sin(theta_start)])
        ax.plot(r0_start[0], r0_start[2], marker="o", ms=5, color="darkorange", zorder=7)

    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-6.0, 6.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R")
    ax.set_ylabel("z / R")
    ax.set_title(title)


fig1, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5))
field_lines_xz(0.0,  ax_l, "Aligned dipole (0°)")
field_lines_xz(47.0, ax_r, "Neptune-like tilt (47°)")

ax_l.plot([], [], color="gray", lw=0.8, ls=":", label="Rotation axis")
ax_l.legend(fontsize=8, loc="upper right")

ax_r.plot([], [], color="gray",    lw=0.8, ls=":",        label="Rotation axis")
ax_r.plot([], [], color="crimson", lw=1.2, ls="--",       label="Magnetic axis")
ax_r.plot([], [], color="crimson", lw=0.9, ls=(0, (2, 3)),
          alpha=0.7,                                       label="Magnetic equator")
ax_r.plot([], [], marker="o", ms=5, color="darkorange",
          ls="none",                                       label="Particle start (L=3, mag. equator)")
ax_r.legend(fontsize=8, loc="upper right")

fig1.suptitle("Test 14: Dipole field lines — aligned vs tilted", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test14_tilted_field_lines.png"), dpi=300)
plt.close()
print("\nSaved test14_tilted_field_lines.png")


# ======================================================================
# Plot 2: z(t) bounce comparison — 7 bounce periods
#
# Decimated by step_skips to show guiding-centre z only, removing
# gyration residual from the raw trajectory.
# ======================================================================
fig2, ax2 = plt.subplots(figsize=(9, 4.5))

for tilt, lbl, col in zip(tilts_deg, labels, colors):
    skip  = step_skips[tilt]
    gc    = extract_gc(trajs[tilt], ts[tilt], B_funcs[tilt], q=q, m=m)
    t_dec = ts[tilt][::skip]
    z_dec = gc[::skip, 2]
    ax2.plot(t_dec, z_dec, lw=0.9, color=col, label=lbl, alpha=0.9)

ax2.axhline(0, color="gray", lw=0.6, ls="--", label="Geographic equator (z=0)")
ax2.set_xlabel("t (code units)")
ax2.set_ylabel("z (code units)")
ax2.set_title("Bounce motion — geographic $z(t)$ for different dipole tilts\n"
              "(each case starts at its magnetic equatorial plane)", fontsize=11)
ax2.legend(fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test14_z_vs_t_tilt_comparison.png"), dpi=300)
plt.close()
print("Saved test14_z_vs_t_tilt_comparison.png")


# ======================================================================
# Plot 2b: |r_xy|(t) sanity check — GC should stay near L=3
#
# If the guiding centre drifts outward (third adiabatic invariant
# violation) this plot will show r_xy growing away from L0=3.
# ======================================================================
fig_r, ax_r = plt.subplots(figsize=(9, 3.5))

for tilt, lbl, col in zip(tilts_deg, labels, colors):
    skip  = step_skips[tilt]
    gc    = extract_gc(trajs[tilt], ts[tilt], B_funcs[tilt], q=q, m=m)
    t_dec = ts[tilt][::skip]
    r_xy  = np.sqrt(gc[::skip, 0]**2 + gc[::skip, 1]**2)
    ax_r.plot(t_dec, r_xy, lw=0.9, color=col, label=lbl, alpha=0.9)

ax_r.axhline(L0, color="gray", lw=0.8, ls="--", label=f"L = {L0}")
ax_r.set_xlabel("t (code units)")
ax_r.set_ylabel(r"$\sqrt{x^2+y^2}$ (code units)")
ax_r.set_title("Test 14: Equatorial radius — GC confinement check")
ax_r.legend(fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test14_r_xy_sanity.png"), dpi=300)
plt.close()
print("Saved test14_r_xy_sanity.png")


# ======================================================================
# Plot 3: 3D GC orbit — static version of animate14b (59° tilt)
#
# Single colour GC path, two reference planes, tilted magnetic axis,
# same view angle and styling as the animation.
# ======================================================================
gc_3d  = extract_gc(traj_short, t_short, B_func_3d, q=q, m=m)
tilt_r = np.deg2rad(tilt_show)
cos_t  = np.cos(tilt_r); sin_t = np.sin(tilt_r)

fig3 = plt.figure(figsize=(7, 7))
ax3  = fig3.add_subplot(111, projection="3d")
ax3.set_facecolor("white")
for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#e0e0e0")
ax3.grid(False)

# --- Tilted field lines — faint context (from animate14b) ---
lam_fl = np.linspace(-1.25, 1.25, 300)
L_fl_3d = [2.0, 2.5, 3.0, 3.5]
phi_fl_3d = np.linspace(0, 2*np.pi, 8, endpoint=False)
for phi_f in phi_fl_3d:
    for L_f in L_fl_3d:
        r_fl = L_f * np.cos(lam_fl)**2
        xm   = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        ym   = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zm   = r_fl * np.sin(lam_fl)
        xg   =  xm * cos_t + zm * sin_t
        yg   =  ym.copy()
        zg   = -xm * sin_t + zm * cos_t
        below = (xg**2 + yg**2 + zg**2) < 1.02**2
        xg[below] = np.nan; yg[below] = np.nan; zg[below] = np.nan
        ax3.plot(xg, yg, zg, color="#909090", lw=0.5, alpha=0.28)

# --- Geographic equatorial plane (z = 0) — grey dashed ring ---
phi_r = np.linspace(0, 2*np.pi, 200)
R_eq  = 3.8
ax3.plot(R_eq * np.cos(phi_r), R_eq * np.sin(phi_r), np.zeros(200),
         color="#999999", lw=1.2, alpha=0.55, ls="--")
ax3.text(R_eq * np.cos(np.pi * 1.25),
         R_eq * np.sin(np.pi * 1.25) - 0.3, 0.15,
         "Geographic equatorial plane", fontsize=7, color="#888888")

# --- Magnetic equatorial plane — crimson dashed ring ---
R_mag = 3.2
phi_m = np.linspace(0, 2*np.pi, 200)
x_meq = -R_mag * np.sin(phi_m) * cos_t
y_meq =  R_mag * np.cos(phi_m)
z_meq =  R_mag * np.sin(phi_m) * sin_t
ax3.plot(x_meq, y_meq, z_meq,
         color="crimson", lw=1.4, alpha=0.50, ls="--")
lbl_idx = 40
ax3.text(x_meq[lbl_idx] + 0.1, y_meq[lbl_idx] + 0.1, z_meq[lbl_idx] + 0.15,
         "Magnetic equatorial plane", fontsize=7,
         color="crimson", fontweight="bold")

# --- Planet sphere ---
u_s = np.linspace(0, 2*np.pi, 24)
v_s = np.linspace(0, np.pi, 16)
ax3.plot_surface(
    np.outer(np.cos(u_s), np.sin(v_s)),
    np.outer(np.sin(u_s), np.sin(v_s)),
    np.outer(np.ones_like(u_s), np.cos(v_s)),
    color="lightsteelblue", alpha=0.55, zorder=0
)

# --- Tilted magnetic axis — prominent (from animate14b) ---
ax3.quiver(0, 0, 0,
           2.2 * sin_t, 0, 2.2 * cos_t,
           color="crimson", lw=3.0, arrow_length_ratio=0.12, zorder=5)
ax3.quiver(0, 0, 0,
           -2.2 * sin_t, 0, -2.2 * cos_t,
           color="crimson", lw=1.5, alpha=0.35, arrow_length_ratio=0.0)
ax3.text(2.2 * sin_t + 0.15, 0.1, 2.2 * cos_t + 0.1,
         "Tilted magnetic axis", fontsize=8,
         color="crimson", fontweight="bold")

# --- Full GC path — single colour (C2 = green, matching animate14b) ---
ax3.plot(gc_3d[:, 0], gc_3d[:, 1], gc_3d[:, 2],
         lw=2.0, alpha=0.55, color="C2")

# --- Start marker ---
ax3.plot([gc_3d[0, 0]], [gc_3d[0, 1]], [gc_3d[0, 2]],
         "o", color="k", ms=7, zorder=12)
ax3.text(gc_3d[0, 0] + 0.2, gc_3d[0, 1], gc_3d[0, 2] + 0.2,
         "Start", fontsize=8, color="k")

# --- Label ---
ax3.text2D(0.04, 0.06, "Guiding-centre drift path",
           transform=ax3.transAxes, fontsize=8, color="C2", fontweight="bold")

ax3.set_xlim(-4, 4); ax3.set_ylim(-4, 4); ax3.set_zlim(-3.5, 3.5)
ax3.set_xlabel("x", fontsize=9); ax3.set_ylabel("y", fontsize=9)
ax3.set_zlabel("z", fontsize=9)
ax3.tick_params(labelsize=7)
ax3.set_title(f"Tilted dipole ({tilt_show:.0f}° tilt)", fontsize=12,
              fontweight="bold")
ax3.view_init(elev=25, azim=-60)     # same as animate14b

plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test14_orbit_3D_tilted.png"), dpi=300)
plt.close()
print("Saved test14_orbit_3D_tilted.png")

print("\nAll test14 figures saved.")