import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", context="paper")

from orbit_ivp_core import simulate_orbit_ivp, q, m
from fields import E_zero, B_uniform_z

# output directory
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

# Test 1: gyromotion in uniform B
# check numerical radius = r_L, period = 2pi/Omega

B0    = 1.0
B_func = B_uniform_z(B0)
E_func = E_zero

dt     = 0.01
T      = 20.0
nsteps = int(T / dt)

r0     = np.array([0.0, 0.0, 0.0])
v0     = np.array([1.0, 0.0, 0.0])   # perp to B
state0 = np.concatenate((r0, v0))

t, traj = simulate_orbit_ivp(
    state0=state0, dt=dt, nsteps=nsteps,
    q=q, m=m, E_func=E_func, B_func=B_func,
)

x, y = traj[:, 0], traj[:, 1]

# analytic predictions
v_perp  = np.linalg.norm(v0)
Omega   = q * B0 / m
r_L     = m * v_perp / (q * B0)      # Larmor radius
T_gyro  = 2.0 * np.pi / Omega

print(f"Larmor radius  (theory): {r_L:.6f}")
print(f"Gyroperiod     (theory): {T_gyro:.6f}")

# numerical radius from GC at (0, -r_L)
gc_x = 0.0
gc_y = -r_L
r_numerical = np.sqrt((x - gc_x)**2 + (y - gc_y)**2)
print(f"Mean numerical radius:   {np.mean(r_numerical):.6f}")
print(f"Max radius error:        {np.max(np.abs(r_numerical - r_L)):.2e}")

# analytic circle
theta     = np.linspace(0, 2 * np.pi, 500)
x_circle  = gc_x + r_L * np.cos(theta)
y_circle  = gc_y + r_L * np.sin(theta)

# plot
fig, ax = plt.subplots(figsize=(5.5, 5.5))

ax.plot(x, y, lw=1.5, label="Numerical orbit")
ax.plot(x_circle, y_circle, "k--", lw=1.0, label=fr"Theory ($r_L={r_L:.2f}$)")

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Circular gyromotion in uniform $B$", fontsize=11)
ax.legend(frameon=True, fontsize=8)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test01_uniformB_xy.png"), dpi=300)
plt.show()