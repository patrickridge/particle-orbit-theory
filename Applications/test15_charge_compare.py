import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib import cm

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, E_corotation, B_dipole_cartesian

_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
os.makedirs(_FIG, exist_ok=True)
os.makedirs(_RES, exist_ok=True)
sns.set_theme(style="ticks", context="paper")

# Test 15 charge comparison: corotation drift for q=+1 and q=-1.
# Verify: E×B is charge-independent, gradB+curv reverses sign.

m     = 1.0
M     = 500.0
Omega = 0.02

B_func = B_dipole_cartesian(M=M, tilt_deg=0.0)
E_cor  = E_corotation(B_func, Omega=Omega)

r0    = np.array([3.0, 0.0, 0.0])
v_mag = 1.0
pitch = np.deg2rad(45.0)

B0_vec = B_func(r0, 0.0)
bhat   = B0_vec / np.linalg.norm(B0_vec)

# initial velocity: same v_perp and magnitude for both charges
# (direction of v_par along -b_hat in both cases since that's the
# test15 convention; charge only enters through the Lorentz force)
v0 = v_mag * (np.cos(pitch) * bhat + np.sin(pitch) * np.array([0.0, 1.0, 0.0]))
state0 = np.concatenate([r0, v0])

Omega_gyro = abs(1.0) * np.linalg.norm(B0_vec) / m
T_gyro     = 2.0 * np.pi / Omega_gyro
T_b_est    = 4.0 * r0[0] / abs(np.dot(v0, bhat))
T_cor      = 2.0 * np.pi / Omega
T_run      = 0.5 * T_cor
dt         = min(T_b_est / 500.0, 0.05 * T_gyro)
nsteps     = int(T_run / dt) + 1

print(f"M={M}, Omega={Omega}, T_b={T_b_est:.2f}, T_cor={T_cor:.1f}, nsteps={nsteps}")

def run(q, E_func, label):
    t0 = time.perf_counter()
    t, traj = simulate_orbit_ivp(
        state0=state0, dt=dt, nsteps=nsteps,
        q=q, m=m, E_func=E_func, B_func=B_func,
        rtol=1e-9, atol=1e-9,
    )
    wall = time.perf_counter() - t0
    print(f"{label}: {wall:.1f} s")
    gc = extract_gc(traj, t, B_func, q=q, m=m)
    return t, traj, gc

t_pE, _, gc_pE = run(+1.0, E_cor,  "q=+1 with corotation")
t_pN, _, gc_pN = run(+1.0, E_zero, "q=+1 no E")
t_nE, _, gc_nE = run(-1.0, E_cor,  "q=-1 with corotation")
t_nN, _, gc_nN = run(-1.0, E_zero, "q=-1 no E")

def fit_omega(t, gc):
    phi = np.unwrap(np.arctan2(gc[:, 1], gc[:, 0]))
    n = len(phi)
    i0, i1 = n // 10, 9 * n // 10
    slope = np.polyfit(t[i0:i1], phi[i0:i1], 1)[0]
    return phi, slope

phi_pE, O_pE = fit_omega(t_pE, gc_pE)
phi_pN, O_pN = fit_omega(t_pN, gc_pN)
phi_nE, O_nE = fit_omega(t_nE, gc_nE)
phi_nN, O_nN = fit_omega(t_nN, gc_nN)

# inferred E×B
O_ExB_p = O_pE - O_pN
O_ExB_n = O_nE - O_nN

print()
print(f"{'Charge':<8} {'Omega(E=0)':>14} {'Omega(full)':>14} "
      f"{'Omega_ExB':>14} {'Input Omega':>14}")
print("-" * 70)
print(f"{'+1':<8} {O_pN:>14.6f} {O_pE:>14.6f} {O_ExB_p:>14.6f} {Omega:>14.6f}")
print(f"{'-1':<8} {O_nN:>14.6f} {O_nE:>14.6f} {O_ExB_n:>14.6f} {Omega:>14.6f}")
print()

# checks
exb_same_mag = abs(abs(O_ExB_p) - abs(O_ExB_n)) / abs(O_ExB_p) < 0.05
gradB_flipped = np.sign(O_pN) != np.sign(O_nN) and abs(O_pN) > 1e-6
pos_gt     = O_pE > Omega
neg_lt     = O_nE < Omega

print(f"E×B magnitudes match across charges? {exb_same_mag}  "
      f"(|+|={abs(O_ExB_p):.5f}, |-|={abs(O_ExB_n):.5f})")
print(f"GradB+curv reverses sign?            {gradB_flipped}  "
      f"(+: {O_pN:+.5f}, -: {O_nN:+.5f})")
print(f"Omega_full > Omega for q=+1?         {pos_gt}  ({O_pE:.5f} vs {Omega:.5f})")
print(f"Omega_full < Omega for q=-1?         {neg_lt}  ({O_nE:.5f} vs {Omega:.5f})")

# figure
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

# top-down GC views
skip = max(1, len(t_pE) // 2000)
for ax, tt, gc, title in [
    (ax1, t_pE, gc_pE, r"$q=+1$, corotation"),
    (ax2, t_nE, gc_nE, r"$q=-1$, corotation"),
]:
    norm = Normalize(vmin=tt.min(), vmax=tt.max())
    ax.scatter(gc[::skip, 0], gc[::skip, 1],
               c=tt[::skip], cmap="plasma", s=0.5)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(3.0*np.cos(theta), 3.0*np.sin(theta),
            "k--", lw=0.6, alpha=0.5)
    ax.plot(np.cos(theta), np.sin(theta), color="lightsteelblue", lw=1)
    ax.fill(np.cos(theta), np.sin(theta), "lightsteelblue", alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title, fontsize=10)

# phi(t)
ax3.plot(t_pE, phi_pE, lw=1.2, color="C0", label=fr"$q=+1$: $\Omega={O_pE:.4f}$")
ax3.plot(t_nE, phi_nE, lw=1.2, color="C3", label=fr"$q=-1$: $\Omega={O_nE:.4f}$")
# linear fit overlay
for tt, phi, slope, color in [
    (t_pE, phi_pE, O_pE, "C0"),
    (t_nE, phi_nE, O_nE, "C3"),
]:
    intercept = phi[len(phi)//2] - slope * tt[len(tt)//2]
    ax3.plot(tt, slope * tt + intercept, ls=":", lw=0.8, color=color, alpha=0.7)
ax3.axhline(0, color="k", lw=0.4, ls="--")
ax3.set_xlabel("t")
ax3.set_ylabel(r"$\phi(t)$ (unwrapped)")
ax3.set_title(fr"Azimuthal drift, input $\Omega={Omega}$", fontsize=10)
ax3.legend(frameon=True, fontsize=8, loc="best")

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "fig_5_6_charge_comparison.png"), dpi=200)
plt.show()

# CSV: time + phi for all four runs (interpolate to common grid if needed)
# all four use same dt/nsteps so grids match
ncommon = min(len(t_pE), len(t_pN), len(t_nE), len(t_nN))
csv_path = os.path.join(_RES, "test14_charge_comparison.csv")
np.savetxt(
    csv_path,
    np.column_stack([
        t_pE[:ncommon], phi_pE[:ncommon], phi_pN[:ncommon],
        phi_nE[:ncommon], phi_nN[:ncommon],
    ]),
    header="t,phi_q+1_cor,phi_q+1_noE,phi_q-1_cor,phi_q-1_noE",
    comments="", delimiter=",",
)
print(f"CSV written: {csv_path}")
