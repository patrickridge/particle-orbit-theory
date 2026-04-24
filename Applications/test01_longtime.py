import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", context="paper")

from orbit_ivp_core import simulate_orbit_ivp, q, m
from fields import E_zero, B_uniform_z

_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
os.makedirs(_FIG, exist_ok=True)
os.makedirs(_RES, exist_ok=True)

# Test 1 (long-time): gyromotion in uniform B over t in [0, 2000]
# Two diagnostics: radial deviation from GC and relative kinetic energy drift.

B0     = 1.0
B_func = B_uniform_z(B0)
E_func = E_zero

T       = 2000.0
nsteps  = 20000
dt      = T / nsteps     # uniform output grid

r0     = np.array([0.0, 0.0, 0.0])
v0     = np.array([1.0, 0.0, 0.0])
state0 = np.concatenate((r0, v0))

t, traj = simulate_orbit_ivp(
    state0=state0, dt=dt, nsteps=nsteps,
    q=q, m=m, E_func=E_func, B_func=B_func,
    rtol=1e-9, atol=1e-12,
)

x = traj[:, 0]; y = traj[:, 1]
v = traj[:, 3:]

# analytic predictions
v_perp = np.linalg.norm(v0)
Omega  = q * B0 / m
r_L    = m * v_perp / (q * B0)
T_gyro = 2.0 * np.pi / Omega

gc_x, gc_y = 0.0, -r_L

# diagnostics
radial_dev = np.abs(np.sqrt((x - gc_x)**2 + (y - gc_y)**2) - r_L)
K          = 0.5 * m * np.sum(v**2, axis=1)
energy_dev = np.abs(K - K[0]) / K[0]

max_rad = np.max(radial_dev)
max_dK  = np.max(energy_dev)

print(f"Larmor radius  (theory): {r_L:.6f}")
print(f"Gyroperiod     (theory): {T_gyro:.6f}")
print(f"Integration range:       t in [0, {T:.0f}]  ({T/T_gyro:.0f} gyroperiods)")
print(f"Output points:           {nsteps}")
print(f"Max radial deviation:    {max_rad:.3e}")
print(f"Max |dK/K0|:             {max_dK:.3e}")

# analytic circle for left panel
theta    = np.linspace(0, 2 * np.pi, 500)
x_circle = gc_x + r_L * np.cos(theta)
y_circle = gc_y + r_L * np.sin(theta)

# two-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# left: orbit
ax1.plot(x, y, lw=0.4, alpha=0.6, label="Numerical orbit")
ax1.plot(x_circle, y_circle, "k--", lw=1.0, label=fr"Theory ($r_L={r_L:.2f}$)")
ax1.set_aspect("equal", adjustable="box")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Circular gyromotion in uniform $B$", fontsize=11)
ax1.legend(frameon=True, fontsize=8)

# right: long-time diagnostics
ax2.semilogy(t, radial_dev, lw=0.9, color="C0",
             label=r"Radial deviation $||\mathbf{r}-\mathbf{r}_{GC}|-r_L|$")
ax2.semilogy(t, energy_dev, lw=0.9, color="C3",
             label=r"$|\delta K / K_0|$")
ax2.set_xlabel("t")
ax2.set_ylabel("Relative deviation")
ax2.set_title(fr"Long-time verification ($T={T:.0f}$, $\sim\!{T/T_gyro:.0f}$ gyroperiods)",
              fontsize=11)
ax2.legend(frameon=True, fontsize=8, loc="best")

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "fig_4_1_longtime.png"), dpi=300)
plt.show()

# CSV
csv_path = os.path.join(_RES, "test01_longtime.csv")
np.savetxt(
    csv_path,
    np.column_stack([t, radial_dev, energy_dev]),
    header="t,radial_deviation,energy_drift",
    comments="", delimiter=",",
)
print(f"CSV written: {csv_path}")
