"""
animate05_mirror.py
===================
Animated magnetic mirror bounce.
Shows: particle spiralling in a converging (mirror) field, reflecting at the
strong-field ends (mirror points), bouncing back and forth.
3D view with field lines that converge toward the ends.
Saves: ../Figures/animate05_mirror.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from orbit_ivp_core import simulate_orbit_ivp, extract_gc, q, m
from fields import E_zero, B_mirror_div_free

# ---- Parameters (match test05) ----
B0    = 1.0
alpha = 0.5      # controls how quickly B strengthens away from z=0
B_func = B_mirror_div_free(B0=B0, alpha=alpha)

omega  = abs(q) * B0 / m
T_gyro = 2 * np.pi / omega
r_L    = 1.0
v_perp = r_L * omega
v_par  = 0.5 * v_perp   # moderate pitch — gives clear bounce

# Choose ICs so GC starts on the z-axis (origin):
# At z=0, B=(0,0,B0). GC = r + (m/qB²)(v×B).
# With r0=(0,r_L,0), v0=(v_perp,0,v_par):
#   v×B = (v_perp,0,v_par)×(0,0,B0) = (0*B0-v_par*0, v_par*0-v_perp*B0, 0)
#        = (0, -v_perp*B0, 0)
#   (m/qB²)(v×B) = (0, -v_perp/B0, 0) = (0, -r_L, 0)
#   GC = (0,r_L,0) + (0,-r_L,0) = (0,0,0)  ✓ on axis
r0     = np.array([0.0, r_L, 0.0])
v0     = np.array([v_perp, 0.0, v_par])
state0 = np.concatenate([r0, v0])

# Mirror point estimate: B_mirror = B0/sin²(alpha_0)
# alpha_0 = pitch angle at z=0: sin(alpha_0) = v_perp/|v| = v_perp/sqrt(v_perp²+v_par²)
v_tot    = np.sqrt(v_perp**2 + v_par**2)
sin_a0   = v_perp / v_tot
B_mirror = B0 / sin_a0**2
z_mirror = np.sqrt((B_mirror / B0 - 1) / alpha)
print(f"Mirror field: B_mirror = {B_mirror:.3f},  z_mirror ≈ ±{z_mirror:.2f}")

# Run for several bounce periods
T_bounce_est = 4 * z_mirror / (v_par * 0.7)   # rough estimate
T_run  = 3.5 * T_bounce_est
dt     = T_gyro / 40
nsteps = int(T_run / dt)

print(f"T_gyro={T_gyro:.3f}, T_bounce_est={T_bounce_est:.2f}")
print(f"dt={dt:.4f}, nsteps={nsteps}")
print("Integrating ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_zero, B_func=B_func,
                              rtol=1e-9, atol=1e-9)
r  = traj[:, :3]
gc = extract_gc(traj, t, B_func, q=q, m=m)

# Detect mirror points from z sign changes of v_par
vz = traj[:, 5]
mirrors = np.where(np.diff(np.sign(vz)))[0]
print(f"Mirror points detected: {len(mirrors)} at z = {r[mirrors, 2]}")
print("Done.")

# ---- Build converging field lines ----
# Field line for B_mirror_div_free: x(z) = x0/sqrt(1 + alpha*z²)
z_fl    = np.linspace(-z_mirror * 1.4, z_mirror * 1.4, 300)
r_starts = [0.5, 1.0, 1.5, 2.0]   # starting radii at z=0
phi_vals = np.linspace(0, 2 * np.pi, 10, endpoint=False)

fl_data = []  # (x_arr, y_arr, z_arr) for each field line
for r0_fl in r_starts:
    for phi_f in phi_vals:
        scale = 1.0 / np.sqrt(1 + alpha * z_fl**2)
        xf    = r0_fl * np.cos(phi_f) * scale
        yf    = r0_fl * np.sin(phi_f) * scale
        fl_data.append((xf, yf, z_fl))

# ---- Figure ----
fig = plt.figure(figsize=(7, 8))
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")

# Static field lines
for xf, yf, zf in fl_data:
    ax.plot(xf, yf, zf, color="steelblue", lw=0.5, alpha=0.2)

# Mirror plane indicators (horizontal rings at ±z_mirror)
theta_r = np.linspace(0, 2 * np.pi, 80)
r_ring  = r_starts[-1] / np.sqrt(1 + alpha * z_mirror**2) * 1.1
for zm in [z_mirror, -z_mirror]:
    ax.plot(r_ring * np.cos(theta_r), r_ring * np.sin(theta_r),
            [zm] * 80, "r--", lw=1.0, alpha=0.5)
ax.text(r_ring + 0.1, 0, z_mirror, "mirror", color="crimson", fontsize=8)
ax.text(r_ring + 0.1, 0, -z_mirror, "mirror", color="crimson", fontsize=8)

# Full trajectory (faint)
ax.plot(r[:, 0], r[:, 1], r[:, 2], lw=0.4, alpha=0.12, color="gray")

# Animated artists
TRAIL    = 150
trail,   = ax.plot([], [], [], lw=0.6, alpha=0.35, color="C0", label="Full orbit")
gc_trail,= ax.plot([], [], [], lw=1.8, color="C1", label="Guiding centre")
dot,     = ax.plot([], [], [], "o", color="C0", ms=5, zorder=10)
gc_dot,  = ax.plot([], [], [], "s", color="C1", ms=6, zorder=10)

# Mirror point flash (green dot at GC position)
mirror_dot, = ax.plot([], [], [], "o", color="limegreen", ms=12,
                       zorder=12, alpha=0, label="Mirror point!")

time_txt = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=9)

lim = r_starts[-1] * 1.2   # GC on axis, particle gyrates with r_L around it
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_zlim(-z_mirror * 1.8, z_mirror * 1.8)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Magnetic mirror: particle bouncing\nbetween mirror points", fontsize=10)
ax.view_init(elev=20, azim=-55)
ax.legend(fontsize=8, loc="upper right")

N_FRAMES = 200
skip     = max(1, len(t) // N_FRAMES)

def update(frame):
    i  = frame * skip
    i  = min(i, len(t) - 1)
    i0 = max(0, i - TRAIL)

    trail.set_data(r[i0:i, 0], r[i0:i, 1])
    trail.set_3d_properties(r[i0:i, 2])

    gc_trail.set_data(gc[i0:i, 0], gc[i0:i, 1])
    gc_trail.set_3d_properties(gc[i0:i, 2])

    dot.set_data([r[i, 0]], [r[i, 1]])
    dot.set_3d_properties([r[i, 2]])
    gc_dot.set_data([gc[i, 0]], [gc[i, 1]])
    gc_dot.set_3d_properties([gc[i, 2]])

    # Flash mirror dot if near a turning point
    near_mirror = any(abs(i - m_idx) < 8 for m_idx in mirrors)
    if near_mirror:
        mirror_dot.set_data([gc[i, 0]], [gc[i, 1]])
        mirror_dot.set_3d_properties([gc[i, 2]])
        mirror_dot.set_alpha(0.9)
    else:
        mirror_dot.set_alpha(0)

    time_txt.set_text(f"t = {t[i]:.1f}")
    return trail, gc_trail, dot, gc_dot, mirror_dot, time_txt

n_anim = min(N_FRAMES, len(t) // skip)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=50, blit=False)

out = "../Figures/animate05_mirror.gif"
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=20, dpi=120)
print("Done.")
plt.show()
