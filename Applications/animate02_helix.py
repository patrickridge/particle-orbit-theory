"""
animate02_helix.py
==================
Animated helix orbit in a uniform magnetic field.
Shows: gyration around field lines, parallel drift along B, resulting helix.
Camera slowly rotates for 3D effect.
Saves: ../Figures/animate02_helix.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from orbit_ivp_core import simulate_orbit_ivp
from fields import E_zero, B_uniform_z

# ---- Parameters ----
q, m = 1.0, 1.0
B0   = 1.0
B_func = B_uniform_z(B0)

omega  = q * B0 / m        # cyclotron frequency
T_gyro = 2 * np.pi / omega
r_L    = 1.0               # gyroradius
v_perp = r_L * omega
v_par  = 0.4               # parallel drift speed

r0     = np.array([r_L, 0.0, 0.0])
v0     = np.array([0.0, v_perp, v_par])
state0 = np.concatenate([r0, v0])

T_run  = 5 * T_gyro
dt     = T_gyro / 60       # 60 points per gyration
nsteps = int(T_run / dt)

print("Integrating ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_zero, B_func=B_func)
r = traj[:, :3]
print(f"Done. {len(t)} points over {T_run:.1f} time units ({T_run/T_gyro:.0f} gyrations).")

# ---- Figure ----
fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")

# Static field lines (vertical arrows along z)
z_fl = np.linspace(0, T_run * v_par * 1.05, 20)
for xf, yf in [(-2.5, -2.5), (-2.5, 2.5), (2.5, -2.5), (2.5, 2.5),
                (0, -2.5), (0, 2.5), (-2.5, 0), (2.5, 0)]:
    ax.plot([xf]*2, [yf]*2, [z_fl[0], z_fl[-1]],
            color="steelblue", lw=0.6, alpha=0.35)
    ax.quiver(xf, yf, z_fl[-1], 0, 0, 0.3,
              color="steelblue", lw=0.6, alpha=0.5, arrow_length_ratio=0.8)

# Full orbit (faint static reference)
ax.plot(r[:, 0], r[:, 1], r[:, 2], lw=0.5, alpha=0.12, color="gray")

# Animated artists
TRAIL = 80
trail, = ax.plot([], [], [], lw=1.6, color="C1", label="Particle orbit")
dot,   = ax.plot([], [], [], "o", color="C1", ms=7, zorder=10)
gc_ln, = ax.plot([], [], [], lw=1.2, color="C0", ls="--", alpha=0.7,
                 label="Guiding centre (field line)")
time_txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9)

ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
ax.set_zlim(0, T_run * v_par * 1.05)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Uniform B field  (B = Bẑ)\nGyration + parallel drift → helix", fontsize=10)
ax.legend(fontsize=8, loc="upper right")

# GC path (on-axis, z only — since no drift in x/y in uniform B+no E)
z_gc = v_par * t

N_FRAMES  = 180
skip      = max(1, len(t) // N_FRAMES)

def update(frame):
    i  = frame * skip
    i  = min(i, len(t) - 1)
    i0 = max(0, i - TRAIL)

    trail.set_data(r[i0:i, 0], r[i0:i, 1])
    trail.set_3d_properties(r[i0:i, 2])

    dot.set_data([r[i, 0]], [r[i, 1]])
    dot.set_3d_properties([r[i, 2]])

    gc_ln.set_data([0, 0], [0, 0])
    gc_ln.set_3d_properties([0, r[i, 2]])

    # Slowly rotate camera
    ax.view_init(elev=20, azim=-60 + frame * 1.2)

    time_txt.set_text(
        f"t = {t[i]:.1f}  ({t[i]/T_gyro:.1f} gyrations)"
    )
    return trail, dot, gc_ln, time_txt

n_anim = min(N_FRAMES, len(t) // skip)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=55, blit=False)

out = "../Figures/animate02_helix.gif"
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()
