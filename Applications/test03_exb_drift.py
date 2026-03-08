import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from orbit_ivp_core import simulate_orbit_ivp, q, m
from fields import E_zero, E_const, B_uniform_z

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 3: E × B drift
#
# Perpendicular E and B fields cause the guiding centre to drift sideways,
# perpendicular to both fields. The drift speed v_ExB = (E × B) / B^2
# Expected: guiding centre drifts in -y at v = E0/B0; numerical agrees.
# =============================================================

B0 = 1.0
E0 = 0.1

E_func_zero  = E_zero
E_func_const = E_const([E0, 0.0, 0.0])   # E along x-hat
B_func       = B_uniform_z(B0)

# Theoretical ExB drift velocity
Evec  = np.array([E0, 0.0, 0.0])
Bvec  = np.array([0.0, 0.0, B0])
v_exb = np.cross(Evec, Bvec) / (np.linalg.norm(Bvec) ** 2)
print(f"Theoretical ExB drift velocity: {v_exb}  (magnitude {np.linalg.norm(v_exb):.4f})")

# Time grid
dt     = 0.01
T      = 50.0
nsteps = int(T / dt)

# Initial condition (small z-velocity gives a nicer 3D look)
r0     = np.array([0.0, 0.0, 0.0])
v0     = np.array([1.0, 0.0, 0.2])
state0 = np.concatenate((r0, v0))

# ---- Integrate -------------------------------------------------------
t, traj_E0 = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                                  q=q, m=m,
                                  E_func=E_func_zero,
                                  B_func=B_func)

t, traj_E  = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                                  q=q, m=m,
                                  E_func=E_func_const,
                                  B_func=B_func)

# ---- Gyro-averaging to isolate guiding-centre motion -----------------
Omega0 = q * B0 / m
Tgyro  = 2.0 * np.pi / Omega0
W      = max(5, int(Tgyro / dt))          # window ≈ 1 gyroperiod
kernel = np.ones(W) / W

yE_smooth  = np.convolve(traj_E[:, 1],  kernel, mode="same")
yE0_smooth = np.convolve(traj_E0[:, 1], kernel, mode="same")

# ---- Numerical drift estimate (fit a line to middle 80% of run) ------
cut         = W
slope, _    = np.polyfit(t[cut:-cut], yE_smooth[cut:-cut], 1), None
v_drift_num = np.polyfit(t[cut:-cut], yE_smooth[cut:-cut], 1)[0]
print(f"Numerical  ExB drift velocity (y-component): {v_drift_num:.4f}")
print(f"Theory     ExB drift velocity (y-component): {v_exb[1]:.4f}")
print(f"Relative error: {abs(v_drift_num - v_exb[1]) / abs(v_exb[1]) * 100:.2f}%")

# ======================================================================
# Plot 1: 3D orbits — E=0 vs E=const
# ======================================================================
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection="3d")
ax.plot(traj_E0[:, 0], traj_E0[:, 1], traj_E0[:, 2],
        lw=1.0, label="E = 0 (no drift)")
ax.plot(traj_E[:, 0],  traj_E[:, 1],  traj_E[:, 2],
        lw=1.0, label=r"E = $E_0\hat{x}$ (ExB drift)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Test 3: E×B drift — 3D comparison")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("Figures/test03_exb_3D_compare.png", dpi=300)
plt.show()

# ======================================================================
# Plot 2: x-y plane — raw cycloid orbit (most intuitive picture)
# ======================================================================
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(traj_E0[:, 0], traj_E0[:, 1], alpha=0.6, lw=0.8, label="E = 0")
ax.plot(traj_E[:, 0],  traj_E[:, 1],  alpha=0.8, lw=0.8,
        label=r"E = $E_0\hat{x}$  (cycloid)")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Test 3: E×B drift — x-y projection")
ax.set_aspect("equal")
ax.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test03_exb_xy_projection.png", dpi=300)
plt.show()