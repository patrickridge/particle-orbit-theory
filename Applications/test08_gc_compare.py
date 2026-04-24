import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from guiding_centre import simulate_gc_orbit
from fields import E_zero, B_dipole_cartesian

_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
os.makedirs(_FIG, exist_ok=True)
os.makedirs(_RES, exist_ok=True)
sns.set_theme(style="ticks", context="paper")

# Test 08 GC-equations comparison.
# Compare full orbit, extracted GC, and GC equations at M = 50
# (marginal adiabaticity) and M = 500 (deep adiabatic regime).
# The M = 50 case should show disagreement between the three traces
# because the first-order extraction formula breaks down when
# rho/r_eq is no longer small.

q, m = 1.0, 1.0

E_func = E_zero
r0_vec = np.array([3.0, 0.0, 0.0])

v_mag     = 1.0
pitch_deg = 60.0
pitch     = np.deg2rad(pitch_deg)
eperp     = np.array([0.0, 1.0, 0.0])


def setup_initial_conditions(M):
    B_func = B_dipole_cartesian(M=M)
    B_r0   = B_func(r0_vec, 0.0)
    bhat_r0 = B_r0 / np.linalg.norm(B_r0)
    v0 = v_mag * (np.cos(pitch) * bhat_r0 + np.sin(pitch) * eperp)
    # extracted initial GC position
    R_GC0 = r0_vec + (m / (q * np.dot(B_r0, B_r0))) * np.cross(v0, B_r0)
    # use B at R_GC (not at r0) for mu0 — correct GC invariant
    B_GC0 = B_func(R_GC0, 0.0)
    Bmag_GC0 = np.linalg.norm(B_GC0)
    bhat_GC0 = B_GC0 / Bmag_GC0
    v_par0 = float(np.dot(v0, bhat_GC0))
    v_perp0_sq = max(np.dot(v0, v0) - v_par0**2, 0.0)
    mu0 = 0.5 * m * v_perp0_sq / Bmag_GC0
    return B_func, v0, R_GC0, v_par0, mu0, Bmag_GC0


def run_case(M, T_run, dt_full=0.0001, dt_gc=0.01):
    B_func, v0, R_GC0, v_par0, mu0, Bmag0 = setup_initial_conditions(M)

    nsteps_full = int(T_run / dt_full)
    nsteps_gc   = int(T_run / dt_gc)

    # full orbit
    state0 = np.concatenate([r0_vec, v0])
    t0 = time.perf_counter()
    t_full, traj_full = simulate_orbit_ivp(
        state0=state0, dt=dt_full, nsteps=nsteps_full,
        q=q, m=m, E_func=E_func, B_func=B_func,
        rtol=1e-9, atol=1e-12,
    )
    w_full = time.perf_counter() - t0
    r_full = traj_full[:, :3]
    v_full = traj_full[:, 3:]

    # extracted GC
    gc_ext = extract_gc(traj_full, t_full, B_func, q=q, m=m)

    # GC equations
    state0_gc = np.array([R_GC0[0], R_GC0[1], R_GC0[2], v_par0])
    t0 = time.perf_counter()
    t_gc, traj_gc = simulate_gc_orbit(
        state0_gc=state0_gc, mu=mu0, dt=dt_gc, nsteps=nsteps_gc,
        q=q, m=m, E_func=E_func, B_func=B_func,
        rtol=1e-9, atol=1e-12,
    )
    w_gc = time.perf_counter() - t0

    # v_par along orbit
    vpar = np.zeros_like(t_full)
    for i in range(len(t_full)):
        Bi = B_func(r_full[i], t_full[i])
        vpar[i] = np.dot(v_full[i], Bi / np.linalg.norm(Bi))
    sign_ch = np.where(np.diff(np.sign(vpar)))[0]

    # gyroradius and r_g/r_eq ratio
    v_perp = v_mag * np.sin(pitch)
    Omega  = abs(q) * Bmag0 / m
    r_gyro = v_perp / Omega
    r_gyro_over_req = r_gyro / r0_vec[0]

    return {
        "M": M, "t_full": t_full, "r_full": r_full,
        "t_gc": t_gc, "r_gc_ext": gc_ext, "r_gceq": traj_gc[:, :3],
        "sign_ch": sign_ch, "vpar": vpar,
        "r_gyro_over_req": r_gyro_over_req,
        "R_GC0": R_GC0, "mu0": mu0,
        "w_full": w_full, "w_gc": w_gc,
    }


# bounce-period estimate
T_b_est = 4.0 * r0_vec[0] / (v_mag * np.cos(pitch))
T_run   = 2.0 * T_b_est

print(f"T_b_est = {T_b_est:.1f}, T_run = {T_run:.1f}")
print()

case_50  = run_case(M=50.0,  T_run=T_run)
case_500 = run_case(M=500.0, T_run=T_run)

for c in (case_50, case_500):
    print(f"M = {c['M']:.0f}:  rho/r_eq = {c['r_gyro_over_req']:.4f}  "
          f"R_GC(0) = ({c['R_GC0'][0]:.3f},{c['R_GC0'][1]:.3f},{c['R_GC0'][2]:.3f})  "
          f"mu_0 = {c['mu0']:.4f}  "
          f"full {c['w_full']:.1f}s / GC {c['w_gc']:.1f}s")

# analytic mirror latitude for pitch 60 deg
from scipy.optimize import brentq
def B_over_Beq(lam):
    s, c = np.sin(lam), np.cos(lam)
    return np.sqrt(1.0 + 3.0 * s * s) / c**6

lam_m = brentq(lambda lam: B_over_Beq(lam) - 1.0 / np.sin(pitch)**2,
               0.0, 0.5 * np.pi - 1e-6)
z_mirror_analytic = r0_vec[0] * np.cos(lam_m)**2 * np.sin(lam_m)
print(f"\nAnalytic mirror latitude (pitch {pitch_deg} deg): "
      f"lambda_m = {np.rad2deg(lam_m):.2f} deg, "
      f"z_mirror = {z_mirror_analytic:.3f}  (on L = {r0_vec[0]:.0f} shell)")

# measure bounce amplitude for each representation in each case
def amp(z):
    return 0.5 * (np.max(z) - np.min(z))

for c in (case_50, case_500):
    A_full  = amp(c["r_full"][:, 2])
    A_ext   = amp(c["r_gc_ext"][:, 2])
    A_gceq  = amp(c["r_gceq"][:, 2])
    print(f"M = {c['M']:.0f}:  amplitudes (peak):  full = {A_full:.3f}  "
          f"extracted = {A_ext:.3f}  GC-eqns = {A_gceq:.3f}  "
          f"(analytic = {z_mirror_analytic:.3f})")

# verdict logic: at M = 500 all three should agree with analytic mirror;
# at M = 50 the extracted-GC amplitude should significantly exceed the
# analytic value because of residual Larmor contribution.
A_ext_50  = amp(case_50["r_gc_ext"][:, 2])
A_ext_500 = amp(case_500["r_gc_ext"][:, 2])
err_50  = abs(A_ext_50  - z_mirror_analytic) / z_mirror_analytic
err_500 = abs(A_ext_500 - z_mirror_analytic) / z_mirror_analytic

print()
print(f"Extracted-GC amplitude error vs analytic mirror z:")
print(f"  M = 50:  {100*err_50:.1f}%")
print(f"  M = 500: {100*err_500:.1f}%")
if err_50 > 0.2 and err_500 < 0.05:
    verdict = "HYPOTHESIS CONFIRMED"
    reason = (f"extraction error is {100*err_50:.0f}% at M=50 "
              f"(marginal adiabatic) but only {100*err_500:.1f}% at M=500 "
              f"(well adiabatic). The first-order extraction formula "
              f"breaks down when rho/r_eq is not small.")
else:
    verdict = "INCONCLUSIVE"
    reason = (f"extraction errors: M=50 -> {100*err_50:.1f}%, "
              f"M=500 -> {100*err_500:.1f}%")
print(f"\nVERDICT: {verdict}")
print(f"REASON : {reason}")

# figure: side-by-side panels for M = 50 and M = 500
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, c in zip(axes, (case_50, case_500)):
    t_f = c["t_full"]
    z_f = c["r_full"][:, 2]
    z_e = c["r_gc_ext"][:, 2]
    t_g = c["t_gc"]
    z_g = c["r_gceq"][:, 2]

    ax.plot(t_f, z_f, lw=0.5, alpha=0.35, color="C0", label="Full orbit")
    ax.plot(t_f, z_e, lw=1.3, color="C1", label="Extracted GC")
    ax.plot(t_g, z_g, lw=1.4, ls="--", color="C3", label="GC equations")
    ax.axhline(+z_mirror_analytic, color="k", ls=":", lw=0.7,
               label=fr"analytic $z_m = \pm{z_mirror_analytic:.3f}$")
    ax.axhline(-z_mirror_analytic, color="k", ls=":", lw=0.7)

    if len(c["sign_ch"]):
        ax.scatter(t_f[c["sign_ch"]], z_e[c["sign_ch"]],
                   s=22, color="green", zorder=6, label="mirror crossings")

    ax.set_xlabel("t")
    ax.set_ylabel("z")
    ax.set_title(
        fr"$M={c['M']:.0f}$, "
        fr"$\rho/r_{{eq}} = {c['r_gyro_over_req']:.3f}$",
        fontsize=11,
    )
    ax.legend(frameon=True, fontsize=7, loc="upper right")

fig.suptitle("Full orbit vs extracted GC vs GC equations — "
             "side-by-side M comparison", fontsize=12)

# insets zooming on first upper mirror
for ax, c in zip(axes, (case_50, case_500)):
    t_f = c["t_full"]; z_e = c["r_gc_ext"][:, 2]
    if len(c["sign_ch"]):
        upper = [i for i in c["sign_ch"] if z_e[i] > 0]
        if upper:
            t_peak = t_f[upper[0]]
            win = 1.5
            axins = ax.inset_axes([0.08, 0.08, 0.38, 0.32])
            mask_f = (t_f >= t_peak - win) & (t_f <= t_peak + win)
            mask_g = (c["t_gc"] >= t_peak - win) & (c["t_gc"] <= t_peak + win)
            axins.plot(t_f[mask_f], c["r_full"][mask_f, 2],
                       lw=0.4, alpha=0.4, color="C0")
            axins.plot(t_f[mask_f], z_e[mask_f], lw=1.3, color="C1")
            axins.plot(c["t_gc"][mask_g], c["r_gceq"][mask_g, 2],
                       lw=1.3, ls="--", color="C3")
            axins.axhline(z_mirror_analytic, color="k", ls=":", lw=0.6)
            axins.set_title(fr"zoom at $t={t_peak:.1f}$", fontsize=8)
            axins.tick_params(labelsize=7)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "fig_5_2_gc_equations.png"), dpi=200)
plt.show()

# CSV: M=50 case (primary) in 4 columns
z_gceq_on_full_50 = np.interp(case_50["t_full"], case_50["t_gc"],
                              case_50["r_gceq"][:, 2])
csv_path = os.path.join(_RES, "test08_gc_compare.csv")
np.savetxt(
    csv_path,
    np.column_stack([case_50["t_full"], case_50["r_full"][:, 2],
                     case_50["r_gc_ext"][:, 2], z_gceq_on_full_50]),
    header="t,z_full,z_GCextracted,z_GCeq",
    comments="", delimiter=",",
)
print(f"\nCSV (M=50 case) written: {csv_path}")
