"""
animate03_exb.py
================
Animated E×B drift in uniform perpendicular fields.
Shows: cycloid orbit with E, guiding centre drift.
2D top-down view (xy plane), B pointing out of screen.
Saves: ../Figures/animate03_exb.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from orbit_ivp_core import simulate_orbit_ivp, q, m
from fields import E_const, B_uniform_z

# ---- Parameters ----
B0, E0 = 1.0, 0.2
B_func = B_uniform_z(B0)
E_func = E_const([E0, 0.0, 0.0])   # E along x

v_exb_mag = E0 / B0
omega  = abs(q) * B0 / m
T_gyro = 2 * np.pi / omega
r_L    = 1.0           # gyroradius (v_perp = r_L * omega)

r0     = np.array([0.0, 0.0, 0.0])
v0     = np.array([r_L * omega, 0.0, 0.0])
state0 = np.concatenate([r0, v0])

T_run  = 6 * T_gyro
dt     = T_gyro / 60
nsteps = int(T_run / dt)

print("Integrating ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_func, B_func=B_func)
print(f"Done. {len(t)} points over {T_run:.1f} time units ({T_run/T_gyro:.0f} gyrations).")

# GC trajectory: starts at (0, -r_L), drifts in -y
# At t=0 the particle is r_L above the GC (force is initially in -y so GC is below)
xgc = np.zeros_like(t)
ygc = -r_L - v_exb_mag * t

# ---- Figure ----
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_facecolor("white")

# Field labels
ax.annotate("", xy=(1.3, 6.0), xytext=(0.0, 6.0),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
ax.text(1.5, 6.0, r"$\mathbf{E}$", color="crimson", fontsize=14, va="center")

ax.text(3.2, 6.0, "⊙", fontsize=20, color="steelblue", va="center", ha="center")
ax.text(3.9, 6.0, r"$\mathbf{B}$", color="steelblue", fontsize=14, va="center")

# Faint full path for reference
ax.plot(traj[:, 0], traj[:, 1], lw=0.5, alpha=0.12, color="C1")

# Animated — cycloid trail + dot
TRAIL = 120
trail, = ax.plot([], [], lw=1.8, color="C1", alpha=0.9, label="Particle (cycloid)")
dot,   = ax.plot([], [], "o", color="C1", ms=7, zorder=10)

# GC
gc_line, = ax.plot([], [], "--", color="royalblue", lw=1.6, alpha=0.85,
                   label=r"Guiding centre  ($\mathbf{v}_{E\times B}$)")
gc_dot,  = ax.plot([], [], "s", color="royalblue", ms=8, zorder=11)

gyration_txt = ax.text(0.03, 0.97, "", transform=ax.transAxes,
                       fontsize=10, va="top", color="gray")

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-8.5, 6.8)
ax.set_aspect("equal")
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title(r"E×B Drift" + "\n"
             + r"$\mathbf{v}_{E\times B} = \mathbf{E}\times\mathbf{B}\,/\,B^2$",
             fontsize=13, pad=10)
ax.legend(fontsize=10, loc="lower right", framealpha=0.8)
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.axvline(0, color="gray", lw=0.5, ls=":")

N_FRAMES = 180
skip = max(1, len(t) // N_FRAMES)

def update(frame):
    i  = min(frame * skip, len(t) - 1)
    i0 = max(0, i - TRAIL)

    trail.set_data(traj[i0:i, 0], traj[i0:i, 1])
    dot.set_data([traj[i, 0]], [traj[i, 1]])

    gc_line.set_data(xgc[:i], ygc[:i])
    gc_dot.set_data([xgc[i]], [ygc[i]])

    gyration_txt.set_text(f"{t[i]/T_gyro:.1f} gyrations")
    return trail, dot, gc_line, gc_dot, gyration_txt

n_anim = min(N_FRAMES, len(t) // skip)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=55, blit=False)

out = "../Figures/animate03_exb.gif"
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()
