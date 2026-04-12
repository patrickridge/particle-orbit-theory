import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, B_dipole_cartesian

# Figures directory — resolved relative to this script, so the script runs correctly from any working directory.
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 8: Full particle orbit in a dipole field
#
# Shows: bounce motion, v_∥ reversal, energy and μ conservation.
# Expected: regular z(t) bounce; slow x-y drift; |ΔE/E| and |Δμ/μ| < 1%.
# =============================================================

q = 1.0
m = 1.0
M = 50.0     # dipole strength — M=50 gives T_b/T_g ~ 7 (visible gyration) and r_gyro/r_eq ~ 0.16

E_func = E_zero
B_func = B_dipole_cartesian(M=M)

# Start at equatorial plane (z=0) on x-axis — per meeting notes:
# "equatorial plane first so z=0, if on x axis v_perp in y direction"
r0     = np.array([3.0, 0.0, 0.0])
B0_vec = B_func(r0, 0.0)          # = (0, 0, -M/r_eq^3)  => b_hat = (0,0,-1)
bhat   = B0_vec / np.linalg.norm(B0_vec)

v_mag     = 1.0
pitch_deg = 60.0
pitch     = np.deg2rad(pitch_deg)

# b_hat = (0,0,-1) at this point, so perp directions are x and y.
# Choose v_perp in y direction: eperp = (0,1,0)
# => v0 = (0, v_perp, -v_par)
eperp = np.array([0.0, 1.0, 0.0])

v0     = v_mag * (np.cos(pitch) * bhat + np.sin(pitch) * eperp)
state0 = np.concatenate((r0, v0))

# ---- Time setup: run for 4 bounce periods ----------------------------
v_par_mag = v_mag * np.cos(pitch)
T_b_est   = 4.0 * r0[0] / v_par_mag   # rough bounce period estimate
T         = 4.0 * T_b_est
dt        = 0.001
nsteps    = int(T / dt)

Omega   = abs(q) * np.linalg.norm(B0_vec) / m
T_gyro  = 2.0 * np.pi / Omega
r_gyro  = v_mag * np.sin(pitch) / Omega
print(f"M={M}, T_gyro={T_gyro:.3f}, T_b_est={T_b_est:.2f}")
print(f"r_gyro/r_eq={r_gyro/r0[0]:.3f}, T_b/T_g={T_b_est/T_gyro:.1f}")
print(f"T_run={T:.1f}, nsteps={nsteps}")

# ---- Integrate -------------------------------------------------------
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m,
                              E_func=E_func, B_func=B_func)

r = traj[:, :3]
v = traj[:, 3:]

# ---- Diagnostics at every time step ---------------------------------
vpar  = np.zeros_like(t)
vperp = np.zeros_like(t)
Bmag  = np.zeros_like(t)
mu    = np.zeros_like(t)     # magnetic moment  μ = m v_perp^2 / (2 B)

for i in range(len(t)):
    Bi       = B_func(r[i], t[i])
    Bmag[i]  = np.linalg.norm(Bi)
    bh       = Bi / Bmag[i]
    vpar[i]  = np.dot(v[i], bh)
    vperp[i] = np.sqrt(max(np.dot(v[i], v[i]) - vpar[i]**2, 0.0))
    mu[i]    = 0.5 * m * vperp[i]**2 / Bmag[i]

# Kinetic energy and relative drift
K         = 0.5 * m * np.sum(v**2, axis=1)
rel_drift = (K - K[0]) / K[0]

# ---- Bounce-period measurement from v_∥ zero-crossings ---------------
sign_changes = np.where(np.diff(np.sign(vpar)))[0]
if len(sign_changes) >= 2:
    # Each pair of crossings is half a bounce period
    half_periods = np.diff(t[sign_changes])
    n_pairs      = min(len(half_periods[::2]), len(half_periods[1::2]))
    full_periods = half_periods[:2*n_pairs:2] + half_periods[1:2*n_pairs:2] if n_pairs > 0 else half_periods * 2
    tau_b_num    = np.mean(full_periods) if len(full_periods) > 0 else np.nan
    print(f"Mirror points (v_∥=0) detected at t = {t[sign_changes]}")
    print(f"Measured bounce period: {tau_b_num:.4f}  (time units)")
else:
    tau_b_num = np.nan
    print("Not enough mirror points detected — extend T or adjust pitch angle.")

# ---- Extract guiding centre ------------------------------------------
r_gc_extracted = extract_gc(traj, t, B_func, q=q, m=m)

# ======================================================================
# Plot 1: z(t) — shows bounce oscillation, full orbit + GC
# ======================================================================
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

# ======================================================================
# Plot 2: v_∥(t) — reverses sign at mirror points
# ======================================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, vpar, lw=1.2)
ax.axhline(0.0, color="k", ls="--", lw=0.8)
ax.set_xlabel("t"); ax.set_ylabel(r"$v_\parallel$")
ax.set_title(r"$v_\parallel$ reverses sign at mirror points", fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test08_dipole_vpar_vs_t.png"), dpi=300)
plt.show()

# ======================================================================
# Plot 3: Magnetic moment μ(t) — should be approximately conserved
# ======================================================================
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

# ======================================================================
# Plot 4: Energy conservation
# ======================================================================
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

# ======================================================================
# Plot 5: 3D orbit — show first 1.5 bounce periods for clarity
# (full run is 4 bounce periods but the 3D tube becomes very dense)
# ======================================================================
n3d = int(1.5 * T_b_est / dt)   # 1.5 bounce periods

fig = plt.figure(figsize=(7, 6))
ax  = fig.add_subplot(111, projection="3d")

# --- Dipole field lines for geometric context ---
lam_fl  = np.linspace(-1.25, 1.25, 300)
L_fl    = [2.0, 2.5, 3.0, 3.5, 4.0]
phi_fl  = np.linspace(0, 2*np.pi, 8, endpoint=False)
for phi_f in phi_fl:
    for L_f in L_fl:
        r_fl = L_f * np.cos(lam_fl)**2
        xf   = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        yf   = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zf   = r_fl * np.sin(lam_fl)
        # Blank out below planet surface
        below = (xf**2 + yf**2 + zf**2) < 1.02**2
        xf[below] = np.nan; yf[below] = np.nan; zf[below] = np.nan
        ax.plot(xf, yf, zf, color="steelblue", lw=0.4, alpha=0.18)

# --- Planet sphere ---
u_s = np.linspace(0, 2*np.pi, 30)
v_s = np.linspace(0, np.pi, 20)
ax.plot_surface(
    np.outer(np.cos(u_s), np.sin(v_s)),
    np.outer(np.sin(u_s), np.sin(v_s)),
    np.outer(np.ones_like(u_s), np.cos(v_s)),
    color="lightsteelblue", alpha=0.6, zorder=0
)

ax.plot(r[:n3d, 0], r[:n3d, 1], r[:n3d, 2],
        lw=0.4, alpha=0.4, color="C0", label="Full orbit")
ax.plot(r_gc_extracted[:n3d, 0], r_gc_extracted[:n3d, 1], r_gc_extracted[:n3d, 2],
        lw=1.6, color="C1", label="Guiding centre")

ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("3D orbit in dipole field", fontsize=11)
ax.view_init(elev=25, azim=-50)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test08_dipole_orbit_3D.png"), dpi=300)
plt.show()