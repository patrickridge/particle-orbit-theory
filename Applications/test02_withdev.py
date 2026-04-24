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

# Test 2 (with deviation): helix in uniform B + parallel-motion deviation panel.

B0     = 1.0
B_func = B_uniform_z(B0)
E_func = E_zero

dt     = 0.01
T      = 20.0
nsteps = int(T / dt)

r0     = np.array([0.0, 0.0, 0.0])
v0     = np.array([1.0, 0.0, 0.2])
state0 = np.concatenate((r0, v0))

t, traj = simulate_orbit_ivp(
    state0=state0, dt=dt, nsteps=nsteps,
    q=q, m=m, E_func=E_func, B_func=B_func,
)

x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

v_perp = np.sqrt(v0[0]**2 + v0[1]**2)
v_par  = v0[2]
Omega  = q * B0 / m
r_L    = m * v_perp / (q * B0)

z_theory    = v_par * t
z_deviation = np.abs(z - z_theory)
max_zdev    = np.max(z_deviation)

print(f"v_parallel (theory):        {v_par}")
print(f"Max |z_num - v_par * t|:    {max_zdev:.3e}")

# two-panel figure
fig = plt.figure(figsize=(12, 7))
ax  = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)

# field lines
z_fl = np.array([z.min() - 0.3, z.max() + 0.3])
fl_positions = [(-1.3, 0), (1.3, 0), (0, -1.3), (0, 1.3)]
for xf, yf in fl_positions:
    ax.plot([xf]*2, [yf]*2, z_fl, color="steelblue", lw=0.5, alpha=0.25)
    ax.quiver(xf, yf, z_fl[-1], 0, 0, 0.2,
              color="steelblue", lw=0.5, alpha=0.25, arrow_length_ratio=0.7)

# GC path
z_gc = v_par * t
ax.plot([0]*len(t), [-r_L]*len(t), z_gc,
        ls="--", lw=1.2, color="C1", alpha=0.7, label="Guiding centre")

# particle orbit
ax.plot(x, y, z, lw=1.2, color="C0", label="Particle orbit")

ax.plot([x[0]], [y[0]], [z[0]], "o", color="k", ms=6, zorder=10)
ax.text(x[0] + 0.15, y[0], z[0] - 0.15, "Start", fontsize=8, color="k")

ax.text2D(0.05, 0.92, r"$\mathbf{B} = B_0\,\hat{\mathbf{z}}$",
          transform=ax.transAxes, fontsize=10, color="steelblue")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Helical motion in uniform $B$", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.view_init(elev=18, azim=-65)
ax.set_box_aspect((1, 1, 2))

# right: deviation plot
# absolute z-deviation is dominated by floating-point round-off, so it
# contains zeros and sub-eps values that look like vertical bars on a log
# plot.  Use a running maximum (cumulative) of the deviation: that gives
# a monotone non-decreasing envelope of the drift and reads cleanly.
z_dev_env = np.maximum.accumulate(z_deviation)
# avoid log(0) by clipping the envelope at the machine-precision floor
floor = 1e-17
z_dev_env = np.maximum(z_dev_env, floor)

ax2.semilogy(t, z_dev_env, lw=1.2, color="C0",
             label=r"running max $|z_{num}-v_\parallel t|$")
ax2.axhline(np.finfo(float).eps, color="gray", ls=":", lw=0.8,
            label=r"machine $\varepsilon \approx 2.2\times10^{-16}$")
ax2.set_xlabel("t")
ax2.set_ylabel(r"$|z_{num}(t) - v_\parallel t|$")
ax2.set_ylim(floor, 1e-13)
ax2.set_title("Parallel-motion deviation (double-precision floor)",
              fontsize=11)
ax2.legend(frameon=True, fontsize=8, loc="lower right")
sns.despine(ax=ax2)

fig.subplots_adjust(bottom=0.08, top=0.92, left=0.05, right=0.97, wspace=0.25)
plt.savefig(os.path.join(_FIG, "fig_4_2_helix.png"), dpi=300)
plt.show()

# CSV
csv_path = os.path.join(_RES, "test02_zdev.csv")
np.savetxt(
    csv_path,
    np.column_stack([t, z_deviation]),
    header="t,z_deviation",
    comments="", delimiter=",",
)
print(f"CSV written: {csv_path}")
