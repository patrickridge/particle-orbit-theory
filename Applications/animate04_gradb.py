"""
animate04_gradb.py

Saves: ../Figures/animate04_gradb.gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, B_gradx_z

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Parameters ----
q, m  = 1.0, 1.0
B0    = 1.0
eps   = 0.28          # clear gradient: B varies ~±30% across the plot

B_func  = B_gradx_z(B0=B0, eps=eps)
Omega_c = abs(q) * B0 / m
T_gyro  = 2.0 * np.pi / Omega_c
r_L     = 1.0
v_perp  = r_L * Omega_c

r0     = np.array([r_L, 0.0, 0.0])
v0     = np.array([0.0, v_perp, 0.0])
state0 = np.concatenate([r0, v0])

N_gyro  = 10
T_run   = N_gyro * T_gyro
dt      = T_gyro / 80.0
nsteps  = int(T_run / dt)

print("Integrating ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_zero, B_func=B_func,
                              rtol=1e-9, atol=1e-9)
r  = traj[:, :3]
gc = extract_gc(traj, t, B_func, q=q, m=m)
print("Done.")

# ---- Axis limits ----
pad   = 1.6
x_min = r[:, 0].min() - pad;  x_max = r[:, 0].max() + pad
y_min = r[:, 1].min() - pad;  y_max = r[:, 1].max() + pad + 0.5

# ---- Field lines: vertical lines spaced by 1/B(x) ----
# Place N_lines lines so their x-positions are uniform in "flux" space:
# cumulative flux F(x) = integral_x0^x B(x') dx' = B0*(x + eps*x^2/2)
# Invert to get x given F.
N_lines = 22
x0 = x_min
F_of_x  = lambda x: B0 * (x + eps * x**2 / 2.0)
F_total = F_of_x(x_max) - F_of_x(x_min)
F_steps = np.linspace(F_of_x(x_min), F_of_x(x_max), N_lines)

# Invert F(x) = B0*(x + eps*x²/2): eps*x²/2 + x - F/B0 = 0
line_x = []
for F in F_steps:
    # quadratic: (eps/2)*x^2 + x - F/B0 = 0
    a = eps / 2.0
    b = 1.0
    c = -F / B0
    disc = b**2 - 4*a*c
    line_x.append((-b + np.sqrt(disc)) / (2*a))

# ---- Figure ----
fig, ax = plt.subplots(figsize=(8, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F5F8FF")   # very faint blue tint

# Background gradient (subtle, field lines carry the main visual)
grad = np.linspace(0, 1, 300).reshape(1, -1)
ax.imshow(grad, aspect="auto", cmap="Blues", alpha=0.20,
          extent=[x_min, x_max, y_min, y_max], origin="lower")

# Draw field lines as vertical segments
for lx in line_x:
    ax.plot([lx, lx], [y_min, y_max],
            color="steelblue", lw=1.0, alpha=0.55)

# ⊙ symbol in bottom-left: circle with central dot showing B out of page
import matplotlib.patches as mpatches
odot_x = x_min + 0.55
odot_y = y_min + 0.55
odot_r = 0.32
ax.add_patch(mpatches.Circle((odot_x, odot_y), odot_r,
                              fill=False, edgecolor="#1a4a8a", lw=2.0, zorder=6))
ax.plot(odot_x, odot_y, "o", color="#1a4a8a", ms=5, zorder=7)
ax.text(odot_x + odot_r + 0.15, odot_y, r"$\mathbf{B}$",
        va="center", fontsize=13, color="#1a4a8a", fontweight="bold")

# "increasing |B|" arrow across the top
ax.annotate("", xy=(x_max - 0.4, y_max - 0.85),
            xytext=(x_min + 0.3, y_max - 0.85),
            arrowprops=dict(arrowstyle="-|>", color="steelblue",
                            lw=2.2, alpha=0.85))
ax.text((x_min + x_max) / 2 - 0.3, y_max - 0.65,
        "increasing  |B|", ha="center", fontsize=12,
        color="steelblue", fontweight="bold", alpha=0.9)


ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal")
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title("Grad-B drift in a non-uniform magnetic field",
             fontsize=12, fontweight="bold", pad=10)
ax.tick_params(labelsize=10)

# Faint full-orbit ghost so the asymmetric loops are visible throughout
ax.plot(r[:, 0], r[:, 1], lw=0.5, alpha=0.12, color="C0")

# ---- Animated artists ----
TRAIL       = 220
trail_part, = ax.plot([], [], lw=1.0, alpha=0.5, color="#4488CC",
                      label="Particle orbit")
trail_gc,   = ax.plot([], [], lw=3.0, color="darkorange",
                      label="Guiding centre", zorder=5)
dot_part,   = ax.plot([], [], "o", color="#4488CC", ms=8, zorder=10)
dot_gc,     = ax.plot([], [], "o", color="darkorange", ms=12,
                      zorder=11, markeredgecolor="white", markeredgewidth=1.5)

ax.legend(fontsize=12, loc="lower right",
          framealpha=0.92, edgecolor="#cccccc")

fig.tight_layout()

N_FRAMES = 220
skip     = max(1, len(t) // N_FRAMES)

def update(frame):
    i  = frame * skip
    i  = min(i, len(t) - 1)
    i0 = max(0, i - TRAIL)

    trail_part.set_data(r[i0:i, 0], r[i0:i, 1])
    trail_gc.set_data(gc[i0:i, 0], gc[i0:i, 1])
    dot_part.set_data([r[i, 0]], [r[i, 1]])
    dot_gc.set_data([gc[i, 0]], [gc[i, 1]])

    return trail_part, trail_gc, dot_part, dot_gc

n_anim = min(N_FRAMES, len(t) // skip)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=55, blit=False)

out = os.path.join(_DIR, "..", "Figures", "animate04_gradb.gif")
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()