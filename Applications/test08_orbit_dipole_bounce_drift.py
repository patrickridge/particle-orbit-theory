import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, B_dipole_cartesian

# output directory
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# Test 8: orbit in dipole field
# check bounce, v_par reversal, energy and mu conservation

q = 1.0
m = 1.0
M = 50.0     # dipole strength

E_func = E_zero
B_func = B_dipole_cartesian(M=M)

# start on equator
r0     = np.array([3.0, 0.0, 0.0])
B0_vec = B_func(r0, 0.0)          # b_hat = (0,0,-1) here
bhat   = B0_vec / np.linalg.norm(B0_vec)

v_mag     = 1.0
pitch_deg = 60.0
pitch     = np.deg2rad(pitch_deg)

# v_perp along y (b_hat is -z)
eperp = np.array([0.0, 1.0, 0.0])

v0     = v_mag * (np.cos(pitch) * bhat + np.sin(pitch) * eperp)
state0 = np.concatenate((r0, v0))

# time setup
v_par_mag = v_mag * np.cos(pitch)
T_b_est   = 4.0 * r0[0] / v_par_mag   # rough T_b
T         = 4.0 * T_b_est
dt        = 0.001
nsteps    = int(T / dt)

Omega   = abs(q) * np.linalg.norm(B0_vec) / m
T_gyro  = 2.0 * np.pi / Omega
r_gyro  = v_mag * np.sin(pitch) / Omega
print(f"M={M}, T_gyro={T_gyro:.3f}, T_b_est={T_b_est:.2f}")
print(f"r_gyro/r_eq={r_gyro/r0[0]:.3f}, T_b/T_g={T_b_est/T_gyro:.1f}")
print(f"T_run={T:.1f}, nsteps={nsteps}")

# integrate
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m,
                              E_func=E_func, B_func=B_func)

r = traj[:, :3]
v = traj[:, 3:]

# diagnostics
vpar  = np.zeros_like(t)
vperp = np.zeros_like(t)
Bmag  = np.zeros_like(t)
mu    = np.zeros_like(t)     # mu = m v_perp^2 / (2B)

for i in range(len(t)):
    Bi       = B_func(r[i], t[i])
    Bmag[i]  = np.linalg.norm(Bi)
    bh       = Bi / Bmag[i]
    vpar[i]  = np.dot(v[i], bh)
    vperp[i] = np.sqrt(max(np.dot(v[i], v[i]) - vpar[i]**2, 0.0))
    mu[i]    = 0.5 * m * vperp[i]**2 / Bmag[i]

# energy
K         = 0.5 * m * np.sum(v**2, axis=1)
rel_drift = (K - K[0]) / K[0]

# bounce period from v_par sign changes
sign_changes = np.where(np.diff(np.sign(vpar)))[0]
if len(sign_changes) >= 2:
    # pair of crossings = half bounce
    half_periods = np.diff(t[sign_changes])
    n_pairs      = min(len(half_periods[::2]), len(half_periods[1::2]))
    full_periods = half_periods[:2*n_pairs:2] + half_periods[1:2*n_pairs:2] if n_pairs > 0 else half_periods * 2
    tau_b_num    = np.mean(full_periods) if len(full_periods) > 0 else np.nan
    print(f"Mirror points (v_∥=0) detected at t = {t[sign_changes]}")
    print(f"Measured bounce period: {tau_b_num:.4f}  (time units)")
else:
    tau_b_num = np.nan
    print("Not enough mirror points detected — extend T or adjust pitch angle.")

# guiding centre
r_gc_extracted = extract_gc(traj, t, B_func, q=q, m=m)

# z(t) plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, r[:, 2], lw=0.6, alpha=0.4, color="C0", label="Full orbit")
ax.plot(t, r_gc_extracted[:, 2], lw=1.4, color="C1", label="Guiding centre")
if len(sign_changes) >= 1:
    ax.scatter(t[sign_changes], r_gc_extracted[sign_changes, 2],
               s=25, zorder=4, color="C2", label="Mirror points")
ax.set_xlabel("t"); ax.set_ylabel("z")
ax.set_title(
    fr"Bounce motion  [$\alpha$={pitch_deg}°, M={M}, "
    fr"$r_g/r_{{eq}}$={r_gyro/r0[0]:.3f}, $T_b/T_g$={T_b_est/T_gyro:.0f}]",
    fontsize=11
)
ax.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test08_dipole_z_vs_t.png"), dpi=300)
plt.show()

# v_par plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, vpar, lw=1.2)
ax.axhline(0.0, color="k", ls="--", lw=0.8)
ax.set_xlabel("t"); ax.set_ylabel(r"$v_\parallel$")
ax.set_title(r"$v_\parallel$ reverses sign at mirror points", fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test08_dipole_vpar_vs_t.png"), dpi=300)
plt.show()

# mu conservation
mu_rel = (mu - mu[0]) / mu[0]

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                          gridspec_kw={"height_ratios": [3, 2]})

axes[0].plot(t, mu, lw=1.1)
axes[0].set_ylabel(r"$\mu = mv_\perp^2 / 2B$")
axes[0].set_title(r"Test 8: Magnetic moment $\mu$ (adiabatic invariant)")

axes[1].plot(t, np.abs(mu_rel), lw=1.0, color="C1")
axes[1].axhline(0.01, color="k", ls="--", lw=0.7, label="1% level")
axes[1].set_yscale("log")
axes[1].set_xlabel("t")
axes[1].set_ylabel(r"$|(\mu - \mu_0)/\mu_0|$")
axes[1].legend(frameon=True, fontsize=8)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test08_dipole_mu_conservation.png"), dpi=300)
plt.show()

# energy conservation
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t, np.abs(rel_drift), lw=1.0)
ax.axhline(1e-6, color="k", ls="--", lw=0.7, label=r"$10^{-6}$ threshold")
ax.set_yscale("log")
ax.set_xlabel("t"); ax.set_ylabel(r"$|(K - K_0)/K_0|$")
ax.set_title("Test 8: Kinetic energy conservation (E = 0)")
ax.legend(frameon=True, fontsize=8)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test08_dipole_energy_drift.png"), dpi=300)
plt.show()

# 3D plot (1.5 bounces)
n3d = int(1.5 * T_b_est / dt)

fig = plt.figure(figsize=(8, 7))
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")

# field lines
lam_fl  = np.linspace(-1.25, 1.25, 300)
L_fl    = [2.5, 3.0, 3.5]
phi_fl  = np.linspace(0, 2*np.pi, 6, endpoint=False)
for phi_f in phi_fl:
    for L_f in L_fl:
        r_fl = L_f * np.cos(lam_fl)**2
        xf   = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        yf   = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zf   = r_fl * np.sin(lam_fl)
        below = (xf**2 + yf**2 + zf**2) < 1.02**2
        xf[below] = np.nan; yf[below] = np.nan; zf[below] = np.nan
        lw_fl = 0.6 if L_f == 3.0 else 0.3
        al_fl = 0.30 if L_f == 3.0 else 0.15
        ax.plot(xf, yf, zf, color="steelblue", lw=lw_fl, alpha=al_fl)

# planet
u_s = np.linspace(0, 2*np.pi, 30)
v_s = np.linspace(0, np.pi, 20)
ax.plot_surface(
    np.outer(np.cos(u_s), np.sin(v_s)),
    np.outer(np.sin(u_s), np.sin(v_s)),
    np.outer(np.ones_like(u_s), np.cos(v_s)),
    color="lightsteelblue", alpha=0.55, zorder=0
)

# orbits
ax.plot(r[:n3d, 0], r[:n3d, 1], r[:n3d, 2],
        lw=0.5, alpha=0.25, color="C0", label="Full orbit")
ax.plot(r_gc_extracted[:n3d, 0], r_gc_extracted[:n3d, 1], r_gc_extracted[:n3d, 2],
        lw=2.0, color="C1", label="Guiding centre")

# mirror points
if len(sign_changes) >= 1:
    mp_idx = sign_changes[sign_changes < n3d]
    ax.scatter(r_gc_extracted[mp_idx, 0], r_gc_extracted[mp_idx, 1],
               r_gc_extracted[mp_idx, 2], s=30, color="limegreen",
               zorder=10, label="Mirror points", edgecolor="white", linewidth=0.5)

# start marker
ax.plot([r_gc_extracted[0, 0]], [r_gc_extracted[0, 1]], [r_gc_extracted[0, 2]],
        "o", color="k", ms=6, zorder=12)
ax.text(r_gc_extracted[0, 0] + 0.25, r_gc_extracted[0, 1],
        r_gc_extracted[0, 2] + 0.15, "Start", fontsize=7, color="k")

ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_zlim(-2.5, 2.5)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Particle orbit in dipole field", fontsize=11)
ax.view_init(elev=10, azim=60)

ax.text2D(0.02, 0.90,
          r"$T_{\rm gyro} \ll T_{\rm bounce} \ll T_{\rm drift}$",
          transform=ax.transAxes, fontsize=9,
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test08_dipole_orbit_3D.png"), dpi=300)
plt.show()