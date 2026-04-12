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
# Run 2: 47° only, short duration (4 bounce periods) for 3D figure
# ======================================================================
tilt_show  = 47.0
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
T_run_3d  = 15.0 * T_b_3d
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
# Plot 3: 3D guiding-centre orbit — 47°, short run (4 bounce periods)
#
# Uses the dedicated short trajectory so the figure stays clean and
# uncluttered, independent of the longer z(t) run.
# Decimated to one point per gyration via skip_3d.
# ======================================================================
gc_3d  = extract_gc(traj_short, t_short, B_func_3d, q=q, m=m)
x_gc   = gc_3d[::skip_3d, 0]
y_gc   = gc_3d[::skip_3d, 1]
z_gc   = gc_3d[::skip_3d, 2]
t_gc   = t_short[::skip_3d]
tilt_r = np.deg2rad(tilt_show)

norm = Normalize(vmin=t_gc.min(), vmax=t_gc.max())
cmap = cm.plasma

fig3 = plt.figure(figsize=(8, 7))
ax3  = fig3.add_subplot(111, projection="3d")

# GC orbit coloured by time
for i in range(len(x_gc) - 1):
    c = cmap(norm(0.5 * (t_gc[i] + t_gc[i + 1])))
    ax3.plot(x_gc[i:i+2], y_gc[i:i+2], z_gc[i:i+2],
             color=c, lw=1.6, alpha=0.9)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig3.colorbar(sm, ax=ax3, pad=0.05, shrink=0.55, label="t (code units)")

# Planet sphere — radius 1 (code units) so it looks right relative to L=3 orbit
u_s = np.linspace(0, 2 * np.pi, 30)
v_s = np.linspace(0, np.pi, 20)
r_p = 1.0
ax3.plot_surface(
    r_p * np.outer(np.cos(u_s), np.sin(v_s)),
    r_p * np.outer(np.sin(u_s), np.sin(v_s)),
    r_p * np.outer(np.ones_like(u_s), np.cos(v_s)),
    color="lightgray", alpha=0.7, zorder=3, linewidth=0
)

# Magnetic equatorial plane — dashed circle at L=3 as spatial reference
# Plane normal: m_hat = (sin θ, 0, cos θ); two perpendicular vectors in the plane:
#   e1 = (cos θ, 0, -sin θ),  e2 = (0, 1, 0)
e1  = np.array([ np.cos(tilt_r), 0.0, -np.sin(tilt_r)])
e2  = np.array([0.0, 1.0, 0.0])
phi_eq = np.linspace(0, 2 * np.pi, 200)
eq_x = L0 * (np.cos(phi_eq) * e1[0] + np.sin(phi_eq) * e2[0])
eq_y = L0 * (np.cos(phi_eq) * e1[1] + np.sin(phi_eq) * e2[1])
eq_z = L0 * (np.cos(phi_eq) * e1[2] + np.sin(phi_eq) * e2[2])
ax3.plot(eq_x, eq_y, eq_z, color="crimson", lw=0.8, ls="--",
         alpha=0.5, label="Mag. equatorial plane (L=3)")

# Axis arrows — fixed length relative to orbit size
arrow_len = 1.8
ax3.quiver(0, 0, 0, 0, 0, arrow_len,
           color="dimgray", lw=1.5, arrow_length_ratio=0.12,
           label="Rotation axis (z)")
ax3.quiver(0, 0, 0,
           arrow_len * np.sin(tilt_r), 0, arrow_len * np.cos(tilt_r),
           color="crimson", lw=1.5, arrow_length_ratio=0.12,
           label=f"Magnetic axis ({tilt_show:.0f}° tilt)")

ax3.set_box_aspect([1, 1, 1])
ax3.view_init(elev=28, azim=-50)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("z")
ax3.set_title(f"GC orbit — {tilt_show:.0f}° tilted dipole (Neptune-like)",
              fontsize=11)
ax3.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test14_orbit_3D_tilted.png"), dpi=300)
plt.close()
print("Saved test14_orbit_3D_tilted.png")

print("\nAll test14 figures saved.")