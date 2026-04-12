import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
from scipy.optimize import brentq

from orbit_ivp_core import simulate_orbit_ivp
from fields import E_zero, B_dipole_cartesian

# Figures directory — resolved relative to this script, so the script runs correctly from any working directory.
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 10: Cross-check — analytic vs numerical bounce period
#
# Runs a full numerical orbit in the dipole field and detects mirror crossings,
# then compares the measured bounce period to the analytic Schulz & Lanzerotti
# formula. Confirms adiabatic theory matches the simulation.
# =============================================================

# ---- Dimensionless parameters matching orbit code -------------------
q   = 1.0
m   = 1.0
M   = 500.0   # dipole moment — matches test08/11 (M=5 was non-adiabatic: T_b/T_gyro < 1)

# Equatorial |B| at r = r_eq is |B_eq| = M/r_eq^3 (on-axis formula)
# For a particle starting at r_eq = 3 (code units), B_eq ≈ M/r_eq^3
r_eq = 3.0
B_eq = M / r_eq**3    # |B| at equatorial crossing (code units)

v_mag     = 1.0
pitch_deg = 60.0
pitch_rad = np.deg2rad(pitch_deg)

# ---- Analytic bounce period (dimensionless) --------------------------

def B_over_Beq_dipole(lam):
    """B(lambda)/B_eq for a dipole field along a field line."""
    s = np.sin(lam)
    c = np.cos(lam)
    return np.sqrt(1.0 + 3.0 * s * s) / (c ** 6)

def mirror_latitude(alpha_eq_rad):
    target = 1.0 / np.sin(alpha_eq_rad)**2
    f = lambda lam: B_over_Beq_dipole(lam) - target
    return brentq(f, 0.0, 0.5 * np.pi - 1e-6, maxiter=300)

def analytic_bounce_period(r_eq, alpha_eq_rad, v, M):
    """
    Bounce period in code units (dimensionless).
    The factor 4 r_eq / v replaces 4 L R_E / v.
    """
    lam_m  = mirror_latitude(alpha_eq_rad)
    sin2a  = np.sin(alpha_eq_rad)**2

    def integrand(lam):
        c    = np.cos(lam)
        root = np.sqrt(1.0 + 3.0 * np.sin(lam)**2)
        val  = max(1.0 - sin2a * B_over_Beq_dipole(lam), 0.0)
        return c * root / np.sqrt(val)

    I, _   = quad(integrand, 0.0, lam_m, points=[lam_m], limit=300)
    return 4.0 * r_eq * I / v

tau_b_analytic = analytic_bounce_period(r_eq, pitch_rad, v_mag, M)
lam_m_deg      = np.rad2deg(mirror_latitude(pitch_rad))
print(f"Pitch angle:           {pitch_deg:.1f}°")
print(f"Mirror latitude:       {lam_m_deg:.2f}°")
print(f"Analytic bounce period: {tau_b_analytic:.4f}  (code time units)")

# ---- Numerical orbit -------------------------------------------------
B_func = B_dipole_cartesian(M=M)
E_func = E_zero

r0   = np.array([r_eq, 0.0, 0.0])    # start exactly at equatorial plane
B0v  = B_func(r0, 0.0)
bhat = B0v / np.linalg.norm(B0v)

# Build perpendicular direction
a = np.array([0.0, 0.0, 1.0])
if abs(np.dot(a, bhat)) > 0.9:
    a = np.array([0.0, 1.0, 0.0])
eperp = a - np.dot(a, bhat) * bhat
eperp = eperp / np.linalg.norm(eperp)

v0     = v_mag * (np.cos(pitch_rad) * bhat + np.sin(pitch_rad) * eperp)
state0 = np.concatenate((r0, v0))

# Run for several bounce periods
T_run  = 6.0 * tau_b_analytic
dt     = 0.001
nsteps = int(T_run / dt)

t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m,
                              E_func=E_func, B_func=B_func)

r = traj[:, :3]
v = traj[:, 3:]

# ---- Compute v_∥ and detect zero-crossings --------------------------
vpar = np.zeros(len(t))
Bmag = np.zeros(len(t))

for i in range(len(t)):
    Bi      = B_func(r[i], t[i])
    Bmag[i] = np.linalg.norm(Bi)
    bh      = Bi / Bmag[i]
    vpar[i] = np.dot(v[i], bh)

sign_changes = np.where(np.diff(np.sign(vpar)))[0]
mirror_times = t[sign_changes]

if len(sign_changes) >= 4:
    # Full bounce = two consecutive zero crossings
    full_period_estimates = np.diff(mirror_times[::2])
    tau_b_num = np.mean(full_period_estimates)
    print(f"Numerical bounce period: {tau_b_num:.4f}  (mean over "
          f"{len(full_period_estimates)} full bounces)")
    print(f"Relative error: {abs(tau_b_num - tau_b_analytic) / tau_b_analytic * 100:.2f}%")
else:
    tau_b_num = np.nan
    print("Fewer than 4 mirror crossings — increase T_run.")

# ======================================================================
# Plot 1: z(t) with mirror points and period annotation
# ======================================================================
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, r[:, 2], lw=1.0, label="z(t)")
if len(sign_changes) >= 1:
    ax.scatter(mirror_times, r[sign_changes, 2],
               s=30, zorder=5, label="mirror points")
ax.axhline(0.0, color="k", lw=0.6, ls="--")
ax.set_xlabel("t (code units)")
ax.set_ylabel("z")
ax.set_title(f"Bounce motion — "
             fr"$\tau_b$ analytic={tau_b_analytic:.3f},  numerical={tau_b_num:.3f}",
             fontsize=11)
ax.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test10_bounce_period_check.png"), dpi=300)
plt.show()

# ======================================================================
# Plot 2: v_∥(t)
# ======================================================================
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(t, vpar, lw=0.9)
ax.axhline(0.0, color="k", lw=0.6, ls="--")
if len(sign_changes) >= 1:
    ax.scatter(mirror_times, np.zeros_like(mirror_times),
               s=30, zorder=5, color="C1")
ax.set_xlabel("t"); ax.set_ylabel(r"$v_\parallel$")
ax.set_title(r"Test 10: $v_\parallel$ reversal at mirror points")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test10_bounce_vpar.png"), dpi=300)
plt.show()

# ======================================================================
# Plot 3 (optional): Bounce period vs pitch angle — sweep
# ======================================================================
pitch_degs = np.array([20, 30, 40, 50, 60, 70, 80])
tau_analytic_arr = []
for adeg in pitch_degs:
    tau = analytic_bounce_period(r_eq, np.deg2rad(adeg), v_mag, M)
    tau_analytic_arr.append(tau)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(pitch_degs, tau_analytic_arr, "o-", lw=1.3, label="Analytic")
if not np.isnan(tau_b_num):
    ax.scatter([pitch_deg], [tau_b_num], s=80, marker="*", zorder=5,
               label=f"Numerical (α={pitch_deg}°)")
ax.set_xlabel(r"Equatorial pitch angle $\alpha_{eq}$ (deg)")
ax.set_ylabel(r"Bounce period $\tau_b$ (code units)")
ax.set_title("Bounce period vs pitch angle", fontsize=11)
ax.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test10_bounce_vs_pitch.png"), dpi=300)
plt.show()