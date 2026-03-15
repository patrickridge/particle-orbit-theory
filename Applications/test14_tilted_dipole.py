import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from orbit_ivp_core import simulate_orbit_ivp
from fields import E_zero, B_dipole_cartesian

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 14: Tilted dipole — planetary application
#
# Demonstrates how particle orbits change when the magnetic
# dipole axis is tilted relative to the planetary rotation axis.
#
# Planetary reference tilts (magnetic vs rotation axis):
#   Earth:   ~11°  (nearly aligned — nearly symmetric bounce)
#   Neptune: ~47°  (strongly tilted — clearly asymmetric bounce)
#   Uranus:  ~59°  (most extreme inner planet — very asymmetric)
#
# All orbits use code units (q=m=1, M=500) with r0 at the
# *geographic* equatorial plane (z=0, x-axis), so the particle
# starts displaced from the magnetic equatorial plane whenever
# tilt > 0.  This displacement is the origin of the north/south
# bounce asymmetry.
# =============================================================

q, m = 1.0, 1.0
M    = 500.0   # dipole strength: T_b/T_g ~ 50, r_gyro/r_eq ~ 0.04 (well adiabatic)

r0    = np.array([3.0, 0.0, 0.0])   # geographic equatorial plane, x-axis
v_mag = 1.0
pitch = np.deg2rad(45.0)            # 45° pitch angle

# Tilt angles to simulate
tilts_deg = [0.0, 47.0, 59.0]
labels    = ["0° (aligned)", "47° (Neptune-like)", "59° (Uranus-like)"]
colors    = ["C0", "C1", "C2"]

# ---- Run orbits for each tilt --------------------------------------------
trajs      = {}
ts         = {}
step_skips = {}   # gyration-period step size per tilt (for 3D decimation)

for tilt in tilts_deg:
    B_func = B_dipole_cartesian(M=M, tilt_deg=tilt)
    B0_vec = B_func(r0, 0.0)
    bhat   = B0_vec / np.linalg.norm(B0_vec)

    # e_perp: at r0=(L,0,0) B always lies in x-z plane (By=0),
    # so (0,1,0) is always perpendicular to bhat here.
    eperp = np.array([0.0, 1.0, 0.0])

    v0     = v_mag * (np.cos(pitch) * bhat + np.sin(pitch) * eperp)
    state0 = np.concatenate([r0, v0])

    # Timing: rough bounce period estimate (same formula as test08)
    v_par_mag = abs(np.dot(v0, bhat))
    Omega     = abs(q) * np.linalg.norm(B0_vec) / m
    T_gyro    = 2.0 * np.pi / Omega
    T_b_est   = 4.0 * r0[0] / v_par_mag
    T_run     = 4.0 * T_b_est
    dt        = T_b_est / 1000.0
    nsteps    = int(T_run / dt) + 1

    print(f"\nTilt = {tilt:.0f}°")
    print(f"  T_gyro={T_gyro:.3f}, T_b_est={T_b_est:.2f}, T_b/T_g={T_b_est/T_gyro:.1f}")
    print(f"  r_gyro/r_eq={v_mag*np.sin(pitch)/Omega/r0[0]:.4f}, nsteps={nsteps}")

    t, traj = simulate_orbit_ivp(
        state0=state0, dt=dt, nsteps=nsteps,
        q=q, m=m, E_func=E_zero, B_func=B_func,
    )
    print("  done.")
    trajs[tilt]      = traj
    ts[tilt]         = t
    step_skips[tilt] = max(1, int(round(T_gyro / dt)))

# ======================================================================
# Plot 1: Field lines — aligned vs Neptune-like tilt (side by side)
# ======================================================================
L_shells = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

def field_lines_xz(tilt_deg, ax, title):
    """Draw L-shell field lines in the x-z plane for a dipole tilted by tilt_deg."""
    tilt_r = np.deg2rad(tilt_deg)
    ct, st = np.cos(tilt_r), np.sin(tilt_r)

    for L in L_shells:
        lam_E = np.arccos(np.sqrt(1.0 / L))
        lam   = np.linspace(-lam_E, lam_E, 600)
        r_m   = L * np.cos(lam)**2

        # Field line in magnetic frame (meridional plane)
        x_m = r_m * np.cos(lam)
        z_m = r_m * np.sin(lam)

        # Rotate into geographic frame: mag-z is tilted by tilt_deg from geo-z
        x_geo =  x_m * ct + z_m * st
        z_geo = -x_m * st + z_m * ct

        ax.plot(x_geo, z_geo, color="steelblue", lw=1.3)
        # Label at equatorial crossing (lam=0)
        ax.text(L * ct + 0.08, -L * st + 0.05, f"L={L}", fontsize=7,
                color="steelblue", ha="left")

    # Planet circle
    theta_c = np.linspace(0, 2 * np.pi, 300)
    ax.fill(np.cos(theta_c), np.sin(theta_c), color="lightgray", zorder=5)
    ax.plot(np.cos(theta_c), np.sin(theta_c), "k-", lw=0.8, zorder=6)

    # Geographic rotation axis (z)
    ax.axvline(0, color="gray", lw=0.8, ls=":", label="Rotation axis")

    # Magnetic axis (tilted) — only draw when actually different from rotation axis
    ax_len = 6.0
    if tilt_deg != 0.0:
        ax.annotate("", xy=(ax_len * st, ax_len * ct),
                    xytext=(-ax_len * st, -ax_len * ct),
                    arrowprops=dict(arrowstyle="-", color="crimson", lw=1.2, ls="--"))

    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-6.0, 6.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R")
    ax.set_ylabel("z / R")
    ax.set_title(title)

fig1, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5))
field_lines_xz(0.0,  ax_l, "Aligned dipole (0°)")
field_lines_xz(47.0, ax_r, "Neptune-like tilt (47°)")

# Legends — rotation axis already labelled via axvline inside function;
# add magnetic axis proxy only for the tilted panel
ax_l.legend(fontsize=8, loc="upper right")
ax_r.plot([], [], color="crimson", lw=1.2, ls="--", label="Magnetic axis")
ax_r.legend(fontsize=8, loc="upper right")

fig1.suptitle("Test 14: Dipole field lines — aligned vs tilted", fontsize=11)
plt.tight_layout()
plt.savefig("Figures/test14_tilted_field_lines.png", dpi=300)
plt.close()
print("\nSaved test14_tilted_field_lines.png")

# ======================================================================
# Plot 2: z(t) bounce comparison — 0°, 47°, 59°
# ======================================================================
fig2, ax2 = plt.subplots(figsize=(9, 4))

for tilt, lbl, col in zip(tilts_deg, labels, colors):
    t   = ts[tilt]
    z   = trajs[tilt][:, 2]
    ax2.plot(t, z, lw=0.8, color=col, label=lbl, alpha=0.9)

ax2.axhline(0, color="gray", lw=0.6, ls="--", label="Geographic equator (z=0)")
ax2.set_xlabel("t (code units)")
ax2.set_ylabel("z (code units)")
ax2.set_title("Test 14: Bounce motion — geographic z(t) for different dipole tilts")
ax2.legend(fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test14_z_vs_t_tilt_comparison.png", dpi=300)
plt.close()
print("Saved test14_z_vs_t_tilt_comparison.png")

# ======================================================================
# Plot 3: 3D guiding-centre orbit — Neptune-like tilt (47°)
#
# Decimate to one point per gyration so the GC bounce+drift is clear.
# Colour by time so trajectory progression is immediately visible.
# ======================================================================
tilt_show  = 47.0
traj_3d    = trajs[tilt_show]
t_3d       = ts[tilt_show]
skip       = step_skips[tilt_show]          # ~1 point per gyration

x_gc = traj_3d[::skip, 0]
y_gc = traj_3d[::skip, 1]
z_gc = traj_3d[::skip, 2]
t_gc = t_3d[::skip]

# Arrow length: ~40% of orbit x-extent so arrows are guides, not dominant
x_range    = x_gc.max() - x_gc.min()
arrow_len  = max(0.4 * x_range, 1.0)
tilt_r     = np.deg2rad(tilt_show)

fig3 = plt.figure(figsize=(7, 6))
ax3  = fig3.add_subplot(111, projection="3d")

# Guiding-centre scatter coloured by time
sc = ax3.scatter(x_gc, y_gc, z_gc, c=t_gc, cmap="plasma",
                 s=8, depthshade=True, alpha=0.85, label="GC trajectory")
fig3.colorbar(sc, ax=ax3, pad=0.1, shrink=0.6, label="t (code units)")

# Planet sphere at origin
u_s = np.linspace(0, 2 * np.pi, 24)
v_s = np.linspace(0, np.pi, 16)
r_p = 0.35
xs  = r_p * np.outer(np.cos(u_s), np.sin(v_s))
ys  = r_p * np.outer(np.sin(u_s), np.sin(v_s))
zs  = r_p * np.outer(np.ones_like(u_s), np.cos(v_s))
ax3.plot_surface(xs, ys, zs, color="lightgray", alpha=0.9, zorder=5)

# Geographic rotation axis (short)
ax3.quiver(0, 0, 0, 0, 0, arrow_len, color="dimgray", lw=1.5,
           arrow_length_ratio=0.12, label="Rotation axis (z)")

# Magnetic axis (short, tilted)
ax3.quiver(0, 0, 0,
           arrow_len * np.sin(tilt_r), 0, arrow_len * np.cos(tilt_r),
           color="crimson", lw=1.5, arrow_length_ratio=0.12,
           label=f"Magnetic axis ({tilt_show:.0f}° tilt)")

ax3.set_box_aspect([1, 1, 1])
ax3.view_init(elev=20, azim=-50)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("z")
ax3.set_title(f"Test 14: GC orbit — {tilt_show:.0f}° tilted dipole (Neptune-like)",
              fontsize=10)
ax3.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig("Figures/test14_orbit_3D_tilted.png", dpi=300)
plt.close()
print("Saved test14_orbit_3D_tilted.png")

print("\nAll test14 figures saved.")
