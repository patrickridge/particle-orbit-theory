import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq

from orbit_ivp_core import simulate_orbit_ivp
from fields import E_zero, B_dipole_cartesian

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 13: Full 10 keV electron orbit in Earth's dipole field — SI units
#
# Simulates a 10 keV electron at L = 3 using physical units (metres, Tesla, kg).
# Computes gyroperiod (~0.03 s), gyroradius (~200 km), and bounce period (~1.8 s).
# Expected: numerical mirror points match analytic mirror latitude; μ conserved < 1%.
# =============================================================

# ---- Physical constants ------------------------------------------------
Re       = 6.371e6            # Earth radius (m)
m_e      = 9.10938e-31        # electron mass (kg)
q_e      = 1.60218e-19        # elementary charge magnitude (C)
c_light  = 2.99792e8          # speed of light (m/s)
B_s      = 3.0e-5             # Earth equatorial surface B field (T)

# ---- Earth dipole moment: |B|(Re, equatorial) = B_s -------------------
M = B_s * Re**3               # SI: T m^3

B_func = B_dipole_cartesian(M=M)
E_func = E_zero

# ---- L-shell and equatorial crossing ----------------------------------
L    = 3.0
r_eq = L * Re                 # equatorial crossing radius (m)
r0   = np.array([r_eq, 0.0, 0.0])

B0_vec  = B_func(r0, 0.0)
Bmag_eq = np.linalg.norm(B0_vec)
bhat0   = B0_vec / Bmag_eq   # = (0, 0, -1) at equatorial x-axis

print("=" * 55)
print("Test 13: 10 keV electron orbit in Earth's dipole (SI)")
print("=" * 55)
print(f"\nDipole moment:    M  = {M:.3e} T m^3")
print(f"L-shell:          L  = {L:.0f}")
print(f"r_eq:             {r_eq:.3e} m  =  {L:.0f} Re")
print(f"|B_eq| at r0:     {Bmag_eq*1e9:.1f} nT"
      f"  (expected {B_s/L**3*1e9:.1f} nT)")

# ---- Relativistic electron speed for 10 keV ---------------------------
E_keV = 10.0
E_J   = E_keV * 1.60218e-16        # J  (1 keV = 1.602e-16 J)
gamma = 1.0 + E_J / (m_e * c_light**2)
beta  = np.sqrt(1.0 - 1.0 / gamma**2)
v_mag = beta * c_light              # speed (m/s)

print(f"\nE_kin:            {E_keV} keV")
print(f"Lorentz gamma:    {gamma:.5f}  (barely relativistic)")
print(f"v:                {v_mag:.4e} m/s  ({100*beta:.2f}% of c)")

# ---- Initial velocity from pitch angle ---------------------------------
pitch_deg = 45.0
pitch_rad = np.deg2rad(pitch_deg)
v_par_mag = v_mag * np.cos(pitch_rad)
v_perp    = v_mag * np.sin(pitch_rad)

# b_hat = (0, 0, -1) at r0  =>  v0 = (0, v_perp, -v_par_mag)
# => v_par = v0 . bhat = (0, v_perp, -v_par_mag).(0, 0, -1) = +v_par_mag
# Particle moves initially in the -z direction (southward along B)
v0 = np.array([0.0, v_perp, -v_par_mag])
v_par_init = np.dot(v0, bhat0)

print(f"\nPitch angle:      alpha_eq = {pitch_deg}°")
print(f"v_par:            {v_par_mag:.4e} m/s")
print(f"v_perp:           {v_perp:.4e} m/s")
print(f"v_par check:      {v_par_init:.4e}  (expect +{v_par_mag:.4e})")

# ---- Electron: charge is negative -------------------------------------
q = -q_e                      # electron charge (C), q < 0

# ---- Adiabaticity parameters ------------------------------------------
Omega_eq = abs(q) * Bmag_eq / m_e    # cyclotron frequency (rad/s)
T_gyro   = 2.0 * np.pi / Omega_eq    # gyroperiod (s)
r_gyro   = v_perp / Omega_eq          # gyroradius (m)
T_b_est  = 4.0 * r_eq / v_par_mag    # rough bounce period (s)

print(f"\nCyclotron freq:   Omega = {Omega_eq:.2f} rad/s")
print(f"Gyroperiod:       T_g   = {T_gyro:.5f} s  ({1/T_gyro:.1f} Hz)")
print(f"Gyroradius:       r_g   = {r_gyro/Re:.5f} Re  ({r_gyro/1e3:.1f} km)")
print(f"r_g / r_eq:       {r_gyro/r_eq:.5f}  (adiabatic if << 1)")
print(f"Bounce T (rough): {T_b_est:.3f} s")
print(f"T_b / T_g:        {T_b_est/T_gyro:.1f}  (adiabatic if >> 1)")

# ---- Analytic mirror latitude ------------------------------------------
# B(lambda)/B_eq = sqrt(1 + 3*sin^2(lambda)) / cos^6(lambda)
# Mirror condition: B(lambda_m) / B_eq = 1 / sin^2(alpha_eq)

def B_over_Beq(lam):
    """B(lam)/B_eq for a dipole field line."""
    s = np.sin(lam)
    c = np.cos(lam)
    return np.sqrt(1.0 + 3.0 * s * s) / (c ** 6)

lam_m_rad = brentq(
    lambda lam: B_over_Beq(lam) - 1.0 / np.sin(pitch_rad)**2,
    0.0, 0.5 * np.pi - 1e-6
)
lam_m_deg = np.rad2deg(lam_m_rad)

# Mirror z on field line: z_m = L*Re * cos^2(lam_m) * sin(lam_m)
z_mirror = L * Re * np.cos(lam_m_rad)**2 * np.sin(lam_m_rad)

# Field-line foot on Earth surface: lambda_E = arccos(1/sqrt(L))
lam_E_deg = np.rad2deg(np.arccos(1.0 / np.sqrt(L)))

print(f"\nMirror latitude:  lambda_m = {lam_m_deg:.2f}°")
print(f"Field-line foot:  lambda_E = {lam_E_deg:.2f}°"
      f"  (lambda_m < lambda_E: mirrors safely above atmosphere)")
print(f"Mirror z:         +/-{z_mirror/Re:.4f} Re  "
      f"= +/-{z_mirror/1e3:.0f} km from equatorial plane")

# ---- Initial magnetic moment (adiabatic invariant) ---------------------
mu0 = 0.5 * m_e * v_perp**2 / Bmag_eq
print(f"Initial mu:       {mu0:.4e} J/T")

# ---- Time setup --------------------------------------------------------
T_run  = 4.0 * T_b_est             # 4 bounce periods
# Use bounce-period sampling, not gyro-period: gyroradius = 200 m = 1e-5 Re is invisible at plot scale.
# Using T_gyro/30 would give ~7 million output points and poor μ conservation from interpolation error.
dt     = T_b_est / 1000.0          # 1000 output points per bounce period
nsteps = int(T_run / dt) + 1

print(f"\nT_run:            {T_run:.3f} s  (~{T_run/T_b_est:.0f} bounce periods)")
print(f"dt:               {dt:.5f} s  ({T_b_est/dt:.0f} steps/bounce)")
print(f"nsteps:           {nsteps}")

# ---- Integrate full Lorentz orbit -------------------------------------
print("\nIntegrating full orbit ...")
t_start = time.perf_counter()
state0 = np.concatenate([r0, v0])
t, traj = simulate_orbit_ivp(
    state0=state0, dt=dt, nsteps=nsteps,
    q=q, m=m_e, E_func=E_func, B_func=B_func,
    rtol=1e-10, atol=1.0        # atol=1 m; suitable for SI (|r| ~ 1e7 m)
)
r_traj = traj[:, :3]
v_traj = traj[:, 3:]
print("  done.")
print(f"  Integration took {time.perf_counter() - t_start:.2f} s")

# ---- Diagnostics: v_par, mu, kinetic energy ---------------------------
print("Computing diagnostics ...")
vpar_arr = np.zeros(nsteps)
mu_arr   = np.zeros(nsteps)
KE_arr   = 0.5 * m_e * np.sum(v_traj**2, axis=1)

for i in range(nsteps):
    Bi     = B_func(r_traj[i], t[i])
    Bmag_i = np.linalg.norm(Bi)
    bhi    = Bi / Bmag_i
    vp_i   = np.dot(v_traj[i], bhi)
    vperp2 = max(np.dot(v_traj[i], v_traj[i]) - vp_i**2, 0.0)
    vpar_arr[i] = vp_i
    mu_arr[i]   = 0.5 * m_e * vperp2 / Bmag_i

mu_rel = (mu_arr - mu0) / mu0
KE_rel = (KE_arr - KE_arr[0]) / KE_arr[0]

# ---- Mirror crossings (v_par sign changes) ----------------------------
sign_ch  = np.where(np.diff(np.sign(vpar_arr)))[0]
mirror_t = t[sign_ch]
mirror_z = r_traj[sign_ch, 2]

print(f"Mirror crossings detected: {len(sign_ch)}")
if len(sign_ch) > 0:
    print(f"Numerical |z_mirror| / Re: {np.abs(mirror_z[:6]) / Re}")
print(f"Analytic  |z_mirror| / Re:  {z_mirror / Re:.4f}")
print(f"Max |Δμ/μ0|:  {np.max(np.abs(mu_rel)):.2e}")
print(f"Max |ΔKE/KE|: {np.max(np.abs(KE_rel)):.2e}")

# ======================================================================
# Plot 1: z(t) in Re — bounce motion with analytic mirror latitude
# ======================================================================
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, r_traj[:, 2] / Re, lw=0.8, color="C0", label="Full orbit  z(t)")
if len(sign_ch) > 0:
    ax.scatter(mirror_t, mirror_z / Re, s=20, color="C1", zorder=5,
               label="Mirror points (numerical)")
ax.axhline(+z_mirror / Re, color="k", ls="--", lw=1.0,
           label=(fr"Analytic $\lambda_m={lam_m_deg:.1f}°$  "
                  fr"($z_m=\pm{z_mirror/Re:.3f}\,R_E$)"))
ax.axhline(-z_mirror / Re, color="k", ls="--", lw=1.0)
ax.axhline(0.0, color="gray", lw=0.4, ls=":")
ax.set_xlabel("t (s)")
ax.set_ylabel(r"z  ($R_E$)")
ax.set_title(
    fr"Test 13 (SI): Bounce motion — 10 keV electron, "
    fr"L={L:.0f}, $\alpha_{{eq}}={pitch_deg:.0f}°$"
)
ax.legend(frameon=True, fontsize=8)
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test13_SI_z_vs_t.png", dpi=300)
plt.show()

# ======================================================================
# Plot 2: v_par(t) — parallel velocity oscillation (bounce motion)
# Note: azimuthal drift is ~900 m/s; drift period ~36 hr >> T_run = 7 s,
#       so x-y drift is invisible on this timescale.
# ======================================================================
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, vpar_arr / 1e6, lw=0.8, color="C2", label=r"$v_\parallel(t)$")
if len(sign_ch) > 0:
    ax.scatter(mirror_t, vpar_arr[sign_ch] / 1e6, s=20, color="C1", zorder=5,
               label="Mirror crossings ($v_\\parallel = 0$)")
ax.axhline(0.0, color="gray", lw=0.4, ls=":")
ax.set_xlabel("t (s)")
ax.set_ylabel(r"$v_\parallel$  (Mm/s)")
ax.set_title(
    fr"Test 13 (SI): Parallel velocity — 10 keV electron, "
    fr"L={L:.0f}, $\alpha_{{eq}}={pitch_deg:.0f}°$"
)
ax.legend(frameon=True, fontsize=8)
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test13_SI_vpar_vs_t.png", dpi=300)
plt.show()

# ======================================================================
# Plot 3: mu relative error — adiabatic invariant conservation
# ======================================================================
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(t, np.abs(mu_rel), lw=0.8, color="C3")
ax.axhline(0.01, color="k", ls="--", lw=0.7, label="1% level")
ax.set_yscale("log")
ax.set_xlabel("t (s)")
ax.set_ylabel(r"$|(\mu - \mu_0)/\mu_0|$")
ax.set_title(
    fr"Test 13 (SI): Adiabatic invariant $\mu$ conservation — "
    f"10 keV electron, L={L:.0f}"
)
ax.legend(frameon=True, fontsize=8)
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test13_SI_mu_error.png", dpi=300)
plt.show()
