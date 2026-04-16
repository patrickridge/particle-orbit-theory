import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from matplotlib.colors import Normalize
from matplotlib import cm

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import B_dipole_rotating, E_corotation

# output directory
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# Test 16: rotating tilted dipole (Neptune-like, 47 deg)
# time-dependent B with corotation E field

q, m      = 1.0, 1.0
M         = 500.0
Omega     = 0.02
tilt_deg  = 47.0

B_func = B_dipole_rotating(M=M, tilt_deg=tilt_deg, Omega=Omega)
E_func = E_corotation(B_func, Omega=Omega)

L0    = 3.0
theta_rad = np.radians(tilt_deg)
r0    = np.array([L0 * np.cos(theta_rad), 0.0, -L0 * np.sin(theta_rad)])
v_mag = 1.0
pitch = np.deg2rad(45.0)

# field at t=0
B0_vec = B_func(r0, 0.0)
bhat   = B0_vec / np.linalg.norm(B0_vec)
v0     = v_mag * (np.cos(pitch) * bhat
                + np.sin(pitch) * np.array([0.0, 1.0, 0.0]))
state0 = np.concatenate([r0, v0])

Omega_gyro = abs(q) * np.linalg.norm(B0_vec) / m
T_gyro     = 2.0 * np.pi / Omega_gyro
v_par_mag  = abs(np.dot(v0, bhat))
T_b_est    = 4.0 * L0 / v_par_mag
T_cor      = 2.0 * np.pi / Omega
T_run      = 0.75 * T_cor    # 3/4 rotation — less dense than full revolution
dt         = min(T_b_est / 500.0, 0.05 * T_gyro)
nsteps     = int(T_run / dt) + 1
skip       = max(1, int(round(T_gyro / dt)))

print(f"Tilt = {tilt_deg:.0f}°,  Omega = {Omega}")
print(f"T_corotation = {T_cor:.1f},  T_bounce_est = {T_b_est:.2f}")
print(f"T_b / T_gyro = {T_b_est/T_gyro:.1f},  nsteps = {nsteps}")
print("Integrating ...")

t, traj = simulate_orbit_ivp(
    state0=state0, dt=dt, nsteps=nsteps,
    q=q, m=m, E_func=E_func, B_func=B_func,
    rtol=1e-9, atol=1e-9,
)
print("done.")

# GC extraction and decimation
gc   = extract_gc(traj, t, B_func, q=q, m=m)
x_gc = gc[::skip, 0];  y_gc = gc[::skip, 1];  z_gc = gc[::skip, 2];  t_gc = t[::skip]
r_gc = np.sqrt(x_gc**2 + y_gc**2)

# azimuthal drift rate (tilted dipole gives Omega_d < Omega)
phi_gc    = np.unwrap(np.arctan2(y_gc, x_gc))
n         = len(phi_gc)
i0, i1    = n // 10, 9 * n // 10
Omega_meas = np.polyfit(t_gc[i0:i1], phi_gc[i0:i1], 1)[0]
print(f"Nominal rotation rate  Omega   = {Omega:.5f} rad/unit-time")
print(f"Measured drift rate    Omega_d = {Omega_meas:.5f} rad/unit-time")
print(f"(For a tilted dipole E×B drift < Omega because v_rot · B != 0)")

norm      = Normalize(vmin=t_gc.min(), vmax=t_gc.max())
cmap_used = cm.plasma

# Plot 1: 3D GC orbit
fig1 = plt.figure(figsize=(8, 7))
ax1  = fig1.add_subplot(111, projection="3d")

# tilted field lines at t=0
tilt_r   = np.deg2rad(tilt_deg)
cos_tilt = np.cos(tilt_r); sin_tilt = np.sin(tilt_r)
lam_fl   = np.linspace(-1.25, 1.25, 300)
L_fl     = [2.0, 2.5, 3.0, 3.5]
phi_fl   = np.linspace(0, 2*np.pi, 8, endpoint=False)
for phi_f in phi_fl:
    for L_f in L_fl:
        r_fl = L_f * np.cos(lam_fl)**2
        # magnetic frame
        xm = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        ym = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zm = r_fl * np.sin(lam_fl)
        # rotate to geographic frame
        xg = xm * cos_tilt + zm * sin_tilt
        yg = ym.copy()
        zg = -xm * sin_tilt + zm * cos_tilt
        below = (xg**2 + yg**2 + zg**2) < 1.02**2
        xg[below] = np.nan; yg[below] = np.nan; zg[below] = np.nan
        ax1.plot(xg, yg, zg, color="steelblue", lw=0.4, alpha=0.18)

# GC path coloured by time
for i in range(len(x_gc) - 1):
    c = cmap_used(norm(0.5 * (t_gc[i] + t_gc[i + 1])))
    ax1.plot(x_gc[i:i+2], y_gc[i:i+2], z_gc[i:i+2],
             color=c, lw=1.4, alpha=0.9)

sm = cm.ScalarMappable(cmap=cmap_used, norm=norm)
sm.set_array([])
fig1.colorbar(sm, ax=ax1, pad=0.08, shrink=0.55, label="t (code units)")

# planet
u_s = np.linspace(0, 2 * np.pi, 24)
v_s = np.linspace(0, np.pi, 16)
r_p = 1.0
ax1.plot_surface(
    r_p * np.outer(np.cos(u_s), np.sin(v_s)),
    r_p * np.outer(np.sin(u_s), np.sin(v_s)),
    r_p * np.outer(np.ones_like(u_s), np.cos(v_s)),
    color="lightgray", alpha=0.8, zorder=5
)

arrow_len = 0.35 * (x_gc.max() - x_gc.min())
ax1.quiver(0, 0, 0, 0, 0, arrow_len,
           color="dimgray", lw=1.5, arrow_length_ratio=0.15,
           label="Rotation axis (z)")
ax1.quiver(0, 0, 0,
           arrow_len * np.sin(tilt_r), 0, arrow_len * np.cos(tilt_r),
           color="crimson", lw=1.5, arrow_length_ratio=0.15,
           label=f"Magnetic axis ({tilt_deg:.0f}° tilt)")

# start marker
ax1.plot([x_gc[0]], [y_gc[0]], [z_gc[0]], "o", color="k", ms=6, zorder=12)
ax1.text(x_gc[0] + 0.2, y_gc[0], z_gc[0] + 0.15, "Start", fontsize=7, color="k")

ax1.set_box_aspect([1, 1, 1])
ax1.view_init(elev=22, azim=-55)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.set_title(
    f"GC orbit — rotating tilted dipole "
    f"({tilt_deg:.0f}° tilt, Ω={Omega})",
    fontsize=11
)
ax1.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test16_rotating_dipole_3D.png"), dpi=300)
plt.close()
print("Saved test16_rotating_dipole_3D.png")

# Plot 2: z(t) bounce
fig2, ax2 = plt.subplots(figsize=(9, 3.5))
ax2.plot(t_gc, z_gc, lw=0.9, color="C0")
ax2.axhline(0, color="gray", lw=0.6, ls="--")
ax2.set_xlabel("t (code units)")
ax2.set_ylabel("z (code units)")
ax2.set_title(
    f"Bounce in rotating tilted dipole "
    f"({tilt_deg:.0f}°, Ω={Omega})",
    fontsize=11
)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test16_rotating_dipole_z_vs_t.png"), dpi=300)
plt.close()
print("Saved test16_rotating_dipole_z_vs_t.png")

# Plot 3: phi(t) co-rotation check
phi_theory = Omega * t_gc

fig3, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(9, 5),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True
)

phi_linear = Omega_meas * (t_gc - t_gc[0]) + phi_gc[0]   # best-fit line

ax_top.plot(t_gc, phi_gc,      lw=1.0, color="C0", label="Numerical φ(t)")
ax_top.plot(t_gc, phi_theory,  lw=1.2, color="k",  ls="--",
            label=f"Nominal: φ = Ωt  (Ω={Omega})")
ax_top.plot(t_gc, phi_linear,  lw=1.0, color="C1", ls="-.",
            label=f"Best-fit: Ω_d = {Omega_meas:.4f}")
ax_top.set_ylabel("φ (rad)")
ax_top.set_title(
    f"Azimuthal drift — rotating tilted dipole "
    f"({tilt_deg:.0f}°, Ω={Omega})",
    fontsize=11
)
ax_top.legend(fontsize=9)

ax_bot.plot(t_gc, phi_gc - phi_linear, lw=0.8, color="C1")
ax_bot.axhline(0, color="gray", lw=0.6, ls="--")
ax_bot.set_xlabel("t (code units)")
ax_bot.set_ylabel("residual from\nbest-fit (rad)")

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test16_rotating_dipole_phi_vs_t.png"), dpi=300)
plt.close()
print("Saved test16_rotating_dipole_phi_vs_t.png")

print("\nAll test16 figures saved.")
