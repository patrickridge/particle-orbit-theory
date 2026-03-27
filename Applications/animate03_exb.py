"""
animate03_exb.py
================
Animated E×B drift in uniform perpendicular fields.
Shows: gyration without E (circle), cycloid orbit with E, guiding centre drift.
2D top-down view (xy plane), B pointing out of screen.
Saves: ../Figures/animate03_exb.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from orbit_ivp_core import simulate_orbit_ivp, q, m
from fields import E_zero, E_const, B_uniform_z

# ---- Parameters (match test03) ----
B0, E0 = 1.0, 0.2
E_func_zero  = E_zero
E_func_const = E_const([E0, 0.0, 0.0])   # E along x
B_func       = B_uniform_z(B0)

v_exb_mag = E0 / B0   # drift in -y direction
omega  = abs(q) * B0 / m
T_gyro = 2 * np.pi / omega
r_L    = 1.0           # gyroradius (v_perp = r_L * omega)

r0     = np.array([0.0, 0.0, 0.0])
v0     = np.array([r_L * omega, 0.0, 0.0])  # start moving in +x → gyrates in xy
state0 = np.concatenate([r0, v0])

T_run  = 6 * T_gyro
dt     = T_gyro / 60
nsteps = int(T_run / dt)

print("Integrating ...")
_, traj_E0 = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                                  q=q, m=m, E_func=E_func_zero, B_func=B_func)
t, traj_E  = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                                  q=q, m=m, E_func=E_func_const, B_func=B_func)
print("Done.")

# GC of the drifting particle (simple: subtract gyration → straight drift line)
# GC position: x_gc = 0, y_gc = -v_exb*t  (drift in -y)
xgc = np.zeros_like(t)
ygc = -v_exb_mag * t

# ---- Figure ----
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_facecolor("white")

# Static arrows for E and B
ax.annotate("", xy=(1.3, 5.2), xytext=(0, 5.2),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
ax.text(1.4, 5.2, r"$\mathbf{E}$", color="crimson", fontsize=13, va="center")

ax.text(3.8, 5.2, "⊙", fontsize=18, color="steelblue", va="center", ha="center")
ax.text(4.6, 5.2, r"$\mathbf{B}$ (out of page)", color="steelblue",
        fontsize=11, va="center")

# Full reference paths (faint)
ax.plot(traj_E0[:, 0], traj_E0[:, 1], lw=0.6, alpha=0.15, color="C0")
ax.plot(traj_E[:, 0],  traj_E[:, 1],  lw=0.6, alpha=0.15, color="C1")

# Animated artists — E=0 (circle)
TRAIL = 100
trail_no_E, = ax.plot([], [], lw=1.2, color="C0", alpha=0.8, label="E = 0 (circular orbit)")
dot_no_E,   = ax.plot([], [], "o", color="C0", ms=6, zorder=10)

# Animated artists — E≠0 (cycloid)
trail_E,    = ax.plot([], [], lw=1.2, color="C1", alpha=0.9, label=r"$E_0\hat{x}$: cycloid (E×B drift)")
dot_E,      = ax.plot([], [], "o", color="C1", ms=6, zorder=10)

# GC drift line
gc_line,    = ax.plot([], [], "--", color="black", lw=1.4, alpha=0.85,
                      label=f"Guiding centre  ($v_{{E×B}}$ = {v_exb_mag:.2f} in −y)")
gc_dot,     = ax.plot([], [], "D", color="black", ms=7, zorder=11)

drift_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                           arrowprops=dict(arrowstyle="->", color="black", lw=2))

time_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                   fontsize=9, va="top")

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-7.0, 5.8)
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(
    r"E×B drift — uniform fields  ($\mathbf{B} = B_0\hat{z}$, $\mathbf{E} = E_0\hat{x}$)"
    f"\n" + r"$v_{E\times B} = E\times B / B^2 = -E_0/B_0\,\hat{y}$",
    fontsize=10
)
ax.legend(fontsize=9, loc="lower right")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.axvline(0, color="gray", lw=0.5, ls=":")

N_FRAMES = 180
skip     = max(1, len(t) // N_FRAMES)

def update(frame):
    i  = frame * skip
    i  = min(i, len(t) - 1)
    i0 = max(0, i - TRAIL)

    trail_no_E.set_data(traj_E0[i0:i, 0], traj_E0[i0:i, 1])
    dot_no_E.set_data([traj_E0[i, 0]], [traj_E0[i, 1]])

    trail_E.set_data(traj_E[i0:i, 0], traj_E[i0:i, 1])
    dot_E.set_data([traj_E[i, 0]], [traj_E[i, 1]])

    gc_line.set_data(xgc[:i], ygc[:i])
    gc_dot.set_data([xgc[i]], [ygc[i]])

    time_txt.set_text(
        f"t = {t[i]:.1f}  ({t[i]/T_gyro:.1f} gyrations)\n"
        f"GC drifted {abs(ygc[i]):.2f} units in −y"
    )
    return trail_no_E, dot_no_E, trail_E, dot_E, gc_line, gc_dot, time_txt

n_anim = min(N_FRAMES, len(t) // skip)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=55, blit=False)

out = "../Figures/animate03_exb.gif"
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()
