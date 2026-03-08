import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import interp1d

from orbit_ivp_core import simulate_orbit_ivp
from guiding_centre import simulate_gc_orbit
from fields import E_zero, B_dipole_cartesian

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 11: Full Lorentz orbit vs Guiding-Centre approximation in dipole field
#
# Runs both integrators from the same starting point and overlays results.
# Uses M=50 (strong dipole) so the adiabatic condition holds (r_gyro << r_eq).
# Expected: GC trajectory matches the orbit average; separation stays ~r_gyro.
# =============================================================

# ---- Parameters -------------------------------------------------------
q   = 1.0
m   = 1.0
M   = 50.0      # strong dipole -> high Omega -> small gyroradius -> adiabatic

B_func = B_dipole_cartesian(M=M)
E_func = E_zero

r_eq = 3.0
r0   = np.array([r_eq, 0.0, 0.0])     # equatorial plane, x-axis, z=0

# ---- Field at starting position ---------------------------------------
B0_vec = B_func(r0, 0.0)
Bmag0  = np.linalg.norm(B0_vec)
bhat0  = B0_vec / Bmag0

print(f"B at r0 = {B0_vec}  =>  b_hat = {bhat0}")
print(f"|B_eq|  = {Bmag0:.4f}")

# ---- Initial velocity from pitch angle --------------------------------
v_mag     = 0.5
pitch_deg = 45.0
pitch_rad = np.deg2rad(pitch_deg)
v_par_mag = v_mag * np.cos(pitch_rad)
v_perp    = v_mag * np.sin(pitch_rad)

# b_hat = (0,0,-1) at r0, so v_par component is along -z.
# Per meeting notes: vx=0, vy=v_perp, vz=-v_par_mag
v0 = np.array([0.0, v_perp, -v_par_mag])

v_par_init = np.dot(v0, bhat0)     # = +v_par_mag

print(f"\nv0        = {v0}")
print(f"|v0|      = {np.linalg.norm(v0):.4f}  (expected {v_mag})")
print(f"v_par     = v0.b_hat = {v_par_init:.4f}  (expected {v_par_mag:.4f})")
print(f"pitch     = {pitch_deg}°")

# ---- Adiabaticity checks ----------------------------------------------
Omega   = abs(q) * Bmag0 / m
T_gyro  = 2.0 * np.pi / Omega
r_gyro  = v_perp / Omega
T_b_est = 4.0 * r_eq / v_par_mag    # rough estimate

print(f"\nGyrofrequency  Omega = {Omega:.4f}")
print(f"Gyroperiod     T_g   = {T_gyro:.4f}")
print(f"Gyroradius     r_g   = {r_gyro:.4f}")
print(f"r_g / r_eq           = {r_gyro/r_eq:.4f}  (adiabatic if << 1)")
print(f"Rough bounce period  = {T_b_est:.2f}")
print(f"T_bounce / T_gyro   ~ {T_b_est/T_gyro:.1f}  (adiabatic if >> 1)")

# ---- Magnetic moment --------------------------------------------------
mu = 0.5 * m * v_perp**2 / Bmag0
print(f"\nInitial mu = {mu:.6f}")

# ---- Time setup -------------------------------------------------------
T_run      = 4.0 * T_b_est
dt_full    = 0.001     # small: resolves gyromotion (~T_gyro / 3000 steps)
dt_gc      = 0.1       # large: GC has no gyromotion to resolve
nsteps_full = int(T_run / dt_full)
nsteps_gc   = int(T_run / dt_gc)

print(f"\nT_run = {T_run:.1f},  "
      f"nsteps_full = {nsteps_full},  nsteps_gc = {nsteps_gc}")

# ---- Full Lorentz orbit -----------------------------------------------
print("\nIntegrating full orbit ...")
state0_full = np.concatenate([r0, v0])
t_full, traj_full = simulate_orbit_ivp(
    state0=state0_full, dt=dt_full, nsteps=nsteps_full,
    q=q, m=m, E_func=E_func, B_func=B_func)

r_full = traj_full[:, :3]
v_full = traj_full[:, 3:]
print("  done.")

# ---- Guiding-centre orbit ---------------------------------------------
# GC starts at the same position as the particle.
# (Gyroradius offset is a higher-order correction discussed in the report.)
print("Integrating guiding-centre orbit ...")
state0_gc = np.array([r0[0], r0[1], r0[2], v_par_init])
t_gc, traj_gc = simulate_gc_orbit(
    state0_gc=state0_gc, mu=mu, dt=dt_gc, nsteps=nsteps_gc,
    q=q, m=m, E_func=E_func, B_func=B_func)

r_gc  = traj_gc[:, :3]
vp_gc = traj_gc[:, 3]
print("  done.")

# ---- Full-orbit diagnostics (vpar, mu, energy) ------------------------
print("Computing diagnostics ...")
vpar_full = np.zeros(nsteps_full)
mu_full   = np.zeros(nsteps_full)

for i in range(nsteps_full):
    Bi      = B_func(r_full[i], t_full[i])
    Bi_mag  = np.linalg.norm(Bi)
    bhi     = Bi / Bi_mag
    vp_i    = np.dot(v_full[i], bhi)
    vperp_i = np.sqrt(max(np.dot(v_full[i], v_full[i]) - vp_i**2, 0.0))
    vpar_full[i] = vp_i
    mu_full[i]   = 0.5 * m * vperp_i**2 / Bi_mag

K_full  = 0.5 * m * np.sum(v_full**2, axis=1)
K_rel   = (K_full - K_full[0]) / K_full[0]
mu_rel  = (mu_full - mu_full[0]) / mu_full[0]

# ---- Interpolate GC to full-orbit time grid ---------------------------
gc_x  = interp1d(t_gc, r_gc[:, 0], kind="cubic", fill_value="extrapolate")
gc_y  = interp1d(t_gc, r_gc[:, 1], kind="cubic", fill_value="extrapolate")
gc_z  = interp1d(t_gc, r_gc[:, 2], kind="cubic", fill_value="extrapolate")
gc_vp = interp1d(t_gc, vp_gc,      kind="cubic", fill_value="extrapolate")

r_gc_interp  = np.column_stack([gc_x(t_full), gc_y(t_full), gc_z(t_full)])
vp_gc_interp = gc_vp(t_full)

sep = np.linalg.norm(r_full - r_gc_interp, axis=1)

# ---- Summary ----------------------------------------------------------
print(f"\n--- Summary ---")
print(f"Gyroradius / r_eq              = {r_gyro/r_eq:.4f}")
print(f"Max separation |r - r_GC|      = {np.max(sep):.4f}")
print(f"Mean separation                = {np.mean(sep):.4f}")
print(f"Max |Delta mu / mu0|           = {np.max(np.abs(mu_rel)):.2e}")
print(f"Max |Delta K / K0|             = {np.max(np.abs(K_rel)):.2e}")

# ======================================================================
# Plot 1: z(t) — bounce motion, full orbit vs GC
# ======================================================================
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_full, r_full[:, 2], lw=0.6, alpha=0.7, color="C0", label="Full orbit")
ax.plot(t_gc,   r_gc[:, 2],   lw=1.8, ls="--",  color="C1", label="Guiding centre")
ax.axhline(0.0, color="k", lw=0.4, ls=":")
ax.set_xlabel("t (code units)")
ax.set_ylabel("z")
ax.set_title(f"Test 11: z(t) — Full orbit vs GC   "
             fr"[$\alpha$={pitch_deg}°, M={M}, $r_g/r_{{eq}}$={r_gyro/r_eq:.3f}]")
ax.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test11_gc_vs_full_z.png", dpi=300)
plt.show()

# ======================================================================
# Plot 2: x-y projection — azimuthal drift
# ======================================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(r_full[:, 0], r_full[:, 1], lw=0.4, alpha=0.6, color="C0",
        label="Full orbit")
ax.plot(r_gc[:, 0],   r_gc[:, 1],   lw=1.8, ls="--",  color="C1",
        label="Guiding centre")
ax.plot(*r0[:2], "ko", ms=7, zorder=5, label="Start")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Test 11: Equatorial (x-y) projection — azimuthal drift")
ax.legend(frameon=True)
ax.set_aspect("equal")
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test11_gc_vs_full_xy.png", dpi=300)
plt.show()

# ======================================================================
# Plot 3: Separation |r_full - r_GC| vs time
# Should oscillate near the gyroradius — not grow secularly
# ======================================================================
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(t_full, sep, lw=0.8, color="C2")
ax.axhline(r_gyro, color="k", ls="--", lw=0.9,
           label=fr"Gyroradius $r_g = {r_gyro:.3f}$")
ax.set_xlabel("t (code units)")
ax.set_ylabel(r"$|\mathbf{r}_{full} - \mathbf{r}_{GC}|$")
ax.set_title("Test 11: Full orbit – GC separation  (expect ~$r_g$ if adiabatic)")
ax.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test11_separation.png", dpi=300)
plt.show()

# ======================================================================
# Plot 4: v_parallel — full orbit vs GC
# ======================================================================
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_full, vpar_full,    lw=0.6, alpha=0.7, color="C0",
        label=r"Full orbit $v_\parallel$")
ax.plot(t_gc,   vp_gc,        lw=1.8, ls="--",  color="C1",
        label=r"GC $v_\parallel$")
ax.axhline(0.0, color="k", lw=0.4, ls=":")
ax.set_xlabel("t (code units)")
ax.set_ylabel(r"$v_\parallel$")
ax.set_title(r"Test 11: $v_\parallel$ — Full orbit vs GC")
ax.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test11_vpar_comparison.png", dpi=300)
plt.show()

# ======================================================================
# Plot 5: mu conservation (full orbit)
# ======================================================================
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(t_full, np.abs(mu_rel), lw=0.8, color="C3")
ax.axhline(0.01, color="k", ls="--", lw=0.7, label="1% level")
ax.set_yscale("log")
ax.set_xlabel("t (code units)")
ax.set_ylabel(r"$|(\mu - \mu_0)/\mu_0|$")
ax.set_title(r"Test 11: Adiabatic invariant $\mu$ conservation (full orbit)")
ax.legend(frameon=True, fontsize=8)
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test11_mu_conservation.png", dpi=300)
plt.show()
