import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from orbit_ivp_core import simulate_orbit_ivp, q, m
from fields import E_zero, B_mirror_div_free

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 5: Magnetic mirror — bounce motion

#   A charged particle bouncing back and forth along a magnetic field
#   that gets stronger towards the ends (like a bottle). The increasing
#   field reflects the particle before it escapes — this is the magnetic
#   mirror effect. The particle's z-coordinate oscillates between two
#   turning (mirror) points where its parallel velocity reverses.
#
# Field used: B_mirror_div_free — divergence-free mirror model
#   Bz(z) = B0 * (1 + alpha * z^2),  Bx and By adjusted so div(B) = 0
#   Field gets stronger away from z = 0 (the midplane).
#
# Expected result:
#   - z(t) oscillates between ±z_mirror  (bounce motion)
#   - Turning points occur where all kinetic energy is in v_perp
#   - Energy is conserved (E = 0, so |v| = const)
# =============================================================

# --- Mirror-like field (divergence-free local model) ---
B0 = 1.0
alpha = 0.5
B_func = B_mirror_div_free(B0=B0, alpha=alpha)
E_func = E_zero

# --- Time grid ---
dt = 0.01
T = 80.0
nsteps = int(T / dt)

# --- IC: mostly perpendicular, small parallel ---
r0 = np.array([0.0, 0.0, 0.0])
v0 = np.array([1.0, 0.0, 0.05])
state0 = np.concatenate((r0, v0))

t, traj = simulate_orbit_ivp(
    state0=state0, dt=dt, nsteps=nsteps,
    q=q, m=m, E_func=E_func, B_func=B_func,
)

r = traj[:, :3]
v = traj[:, 3:]

z = r[:, 2]
vz = v[:, 2]

# turning points: vz changes sign
turn = np.where(np.sign(vz[:-1]) != np.sign(vz[1:]))[0]

# energy diagnostic
K = 0.5 * m * np.sum(v**2, axis=1)
abs_rel_drift = np.abs((K - K[0]) / K[0])

print(f"Number of turning points: {len(turn)}")
print(f"Max |relative energy drift|: {abs_rel_drift.max():.3e}")

# --- Plot 1: z(t) with turning points ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, z, lw=1.2, label="z(t)")
if len(turn) > 0:
    ax.scatter(t[turn], z[turn], s=18, zorder=3, label="turning points")
ax.set_xlabel("t")
ax.set_ylabel("z")
ax.set_title("Test 5: Mirror-like field — bounce motion (turning points)")
ax.legend(fontsize=8, frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test05_mirror_z_turning.png", dpi=300)
plt.show()
'''
# --- Plot 2: energy conservation ---
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t, abs_rel_drift, lw=1.0)
ax.set_yscale("log")
ax.set_xlabel("t")
ax.set_ylabel(r"$|(K(t)-K(0))/K(0)|$")
ax.set_title("Test 5: Energy conservation (E=0)")
sns.despine()
plt.tight_layout()
plt.savefig("Figures/test05_mirror_energy_drift.png", dpi=300)
plt.show()
'''