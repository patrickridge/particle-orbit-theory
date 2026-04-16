import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from orbit_ivp_core import simulate_orbit_ivp, q, m
from fields import E_zero, E_const, B_uniform_z

# output directory
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# Test 3: ExB drift
# check GC drift speed = E0/B0

B0 = 1.0
E0 = 0.1

E_func_zero  = E_zero
E_func_const = E_const([E0, 0.0, 0.0])   # E along x-hat
B_func       = B_uniform_z(B0)

# theory v_ExB
Evec  = np.array([E0, 0.0, 0.0])
Bvec  = np.array([0.0, 0.0, B0])
v_exb = np.cross(Evec, Bvec) / (np.linalg.norm(Bvec) ** 2)
print(f"Theoretical ExB drift velocity: {v_exb}  (magnitude {np.linalg.norm(v_exb):.4f})")

# time grid
dt     = 0.01
T      = 50.0
nsteps = int(T / dt)

# initial condition
r0     = np.array([0.0, 0.0, 0.0])
v0     = np.array([1.0, 0.0, 0.2])
state0 = np.concatenate((r0, v0))

# integrate
t, traj_E0 = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                                  q=q, m=m,
                                  E_func=E_func_zero,
                                  B_func=B_func)

t, traj_E  = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                                  q=q, m=m,
                                  E_func=E_func_const,
                                  B_func=B_func)

# gyro-averaging
Omega0 = q * B0 / m
Tgyro  = 2.0 * np.pi / Omega0
W      = max(5, int(Tgyro / dt))          # window ≈ 1 gyroperiod
kernel = np.ones(W) / W

yE_smooth  = np.convolve(traj_E[:, 1],  kernel, mode="same")
yE0_smooth = np.convolve(traj_E0[:, 1], kernel, mode="same")

# numerical drift
cut         = W
slope, _    = np.polyfit(t[cut:-cut], yE_smooth[cut:-cut], 1), None
v_drift_num = np.polyfit(t[cut:-cut], yE_smooth[cut:-cut], 1)[0]
print(f"Numerical  ExB drift velocity (y-component): {v_drift_num:.4f}")
print(f"Theory     ExB drift velocity (y-component): {v_exb[1]:.4f}")
print(f"Relative error: {abs(v_drift_num - v_exb[1]) / abs(v_exb[1]) * 100:.2f}%")

# plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(traj_E0[:, 0], traj_E0[:, 1], alpha=0.6, lw=0.8, label="E = 0")
ax.plot(traj_E[:, 0],  traj_E[:, 1],  alpha=0.8, lw=0.8,
        label=r"E = $E_0\hat{x}$  (cycloid)")

# start marker
ax.plot(r0[0], r0[1], "o", color="k", ms=6, zorder=9)
ax.text(r0[0] + 0.12, r0[1] + 0.15, "Start", fontsize=8, color="k")

# field arrows
ax.annotate("", xy=(0.85, 0.95), xytext=(0.72, 0.95),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.8))
ax.text(0.87, 0.95, r"$\mathbf{E}$", color="crimson", fontsize=11,
        va="center", transform=ax.transAxes)
ax.text(0.72, 0.88, r"$\odot\;\mathbf{B}$", color="steelblue", fontsize=11,
        va="center", transform=ax.transAxes)

ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(r"$\mathbf{E}\times\mathbf{B}$ drift — $x$-$y$ projection", fontsize=11)
ax.set_aspect("equal")
ax.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test03_exb_xy_projection.png"), dpi=300)
plt.show()