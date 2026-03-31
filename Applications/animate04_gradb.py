"""
animate04_gradb.py
==================
Animated 3D view of curvature drift — fixed camera, trail grows over time.

A single well-chosen viewpoint shows both motions at once:
  - particle/GC follows the curved field line arc in x-z
  - GC drifts steadily in −y out of the field-line plane

Parameters are chosen so the GC approximation is well-satisfied:
  r_L / R_c < 0.01,   |z| / R_c < 0.35   throughout the run.

Saves: ../Figures/animate04_gradb.gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, B_curved_z

_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Parameters
# ============================================================
q, m  = 1.0, 1.0
B0    = 1.0
R_c   = 500.0   # large — keeps z/R_c and x/R_c << 1 throughout

v_par  = 5.0    # mostly parallel
v_perp = 0.3    # small gyration; r_L = 0.3, r_L/R_c = 0.0006

B_curv  = B_curved_z(B0=B0, R_c=R_c)
Omega_c = abs(q) * B0 / m
T_gyro  = 2.0 * np.pi / Omega_c
r_L     = m * v_perp / (abs(q) * B0)

v_curv_theory  = (m * v_par**2)  / (abs(q) * B0 * R_c)
v_gradB_theory = (m * v_perp**2) / (2.0 * abs(q) * B0 * R_c)

# ---- Diagnostics ----
print("=== Parameters (t = 0) ===")
print(f"  v_par  = {v_par:.3f},  v_perp = {v_perp:.3f}")
print(f"  r_L    = {r_L:.4f},   R_c    = {R_c:.1f}")
print(f"  r_L/R_c          = {r_L/R_c:.5f}   (need << 0.05)")
print(f"  v_curv  (theory) = {v_curv_theory:.5f}")
print(f"  v_gradB (theory) = {v_gradB_theory:.7f}  "
      f"({100*v_gradB_theory/v_curv_theory:.2f}% of curvature term)")

# ---- Initial conditions — start at θ=0.15 along arc, not at origin ----
# This places the GC in the middle of the drawn field-line arcs rather
# than at the very bottom corner of the diagram.
theta_start = 0.15
r0 = np.array([
    -R_c * (1.0 - np.cos(theta_start)),   # ≈ −5.6
    0.0,
    R_c * np.sin(theta_start),             # ≈ 74.9
])
B_at_r0 = B_curv(r0, 0.0)
bhat    = B_at_r0 / np.linalg.norm(B_at_r0)
eperp   = np.cross(bhat, np.array([0.0, 1.0, 0.0]))
eperp  /= np.linalg.norm(eperp)
v0      = v_par * bhat + v_perp * eperp
state0  = np.concatenate([r0, v0])

# Run 10 gyrations → y-drift ≈ v_curv × T = 0.05 × 62.8 ≈ 3.14
# so field lines at y = 0, −1, −2, −3 bracket the full trajectory.
N_gyro  = 10
T_run   = N_gyro * T_gyro
dt      = T_gyro / 80.0
nsteps  = int(T_run / dt)

print(f"\n  T_gyro = {T_gyro:.3f},  T_run = {T_run:.2f},  nsteps = {nsteps}")
print("Integrating ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_zero, B_func=B_curv,
                              rtol=1e-9, atol=1e-9)
gc = extract_gc(traj, t, B_curv, q=q, m=m)
print("Done.")

# ---- End-of-run diagnostics ----
v_end      = traj[-1, 3:]
B_end_vec  = B_curv(traj[-1, :3], t[-1])
bhat_end   = B_end_vec / np.linalg.norm(B_end_vec)
v_par_end  = abs(np.dot(v_end, bhat_end))
v_perp_end = np.sqrt(max(0.0, np.dot(v_end, v_end) - v_par_end**2))
r_L_end    = m * v_perp_end / (abs(q) * np.linalg.norm(B_end_vec))
z_final    = gc[-1, 2]

print(f"\n=== At t = {t[-1]:.2f} ===")
print(f"  v_par  = {v_par_end:.4f},  v_perp = {v_perp_end:.4f}")
print(f"  r_L/R_c = {r_L_end/R_c:.5f}")
print(f"  GC y = {gc[-1, 1]:.4f},  z = {z_final:.2f},  z/R_c = {z_final/R_c:.4f}")

# Measured drift speed from GC y(t) linear fit (middle 80%)
i0f, i1f = len(t) // 10, 9 * len(t) // 10
v_drift_meas = np.polyfit(t[i0f:i1f], gc[i0f:i1f, 1], 1)[0]
print(f"\n  Analytic drift speed : {-v_curv_theory:.5f}  (−y)")
print(f"  Measured drift speed : {v_drift_meas:.5f}")
print(f"  Error: {100*abs(v_drift_meas + v_curv_theory)/v_curv_theory:.2f}%")

# ============================================================
# Axis limits — zoomed out to show full arc curvature
# z starts at 0 so the bottom of the arc and its bend are visible
# ============================================================
x_lo = gc[:, 0].min() - 15.0;  x_hi = gc[:, 0].max() + 10.0
y_lo = gc[:, 1].min() - 1.0;   y_hi = 1.5
z_lo = 0.0;                     z_hi = gc[:, 2].max() + 20.0

# ============================================================
# Figure — single 3D panel, fixed camera
# ============================================================
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")

fig.suptitle("Curvature drift", fontsize=14, fontweight="bold")

# ---- Clean up box: remove filled panes and grid ----
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#dddddd")
ax.grid(False)

ax.set_xlabel("x", fontsize=10, labelpad=2)
ax.set_ylabel("y", fontsize=10, labelpad=2)
ax.set_zlabel("z", fontsize=10, labelpad=2)
ax.tick_params(labelsize=8)
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_zlim(z_lo, z_hi)

# Camera: elev=28 shows more of the x-z arc curvature from above;
# azim=-50 balances the y-drift direction with the arc shape.
ax.view_init(elev=28, azim=-50)

# ---- Static reference field lines ----
# Arcs anchored to the GC's actual starting and ending y-positions so the
# particle visually sits on the first arc and ends near the last one.
# Arc runs from theta=0 (bottom of the curve) to beyond the particle's
# highest point — this makes the bend of the field line clearly visible.
theta_max_fl = gc[:, 2].max() / R_c + 0.08
theta_fl = np.linspace(0.0, theta_max_fl, 500)
xa_fl = -R_c + R_c * np.cos(theta_fl)
za_fl =        R_c * np.sin(theta_fl)

y_start = gc[0, 1]    # GC y at t=0  (particle sits here)
y_end   = gc[-1, 1]   # GC y at end  (particle ends here)

# Four evenly-spaced arcs spanning start → end drift range
for k, y_ref in enumerate(np.linspace(y_start, y_end, 4)):
    is_first = (k == 0)
    ax.plot(xa_fl, np.full_like(xa_fl, y_ref), za_fl,
            color="#999999", lw=(1.2 if is_first else 0.7),
            alpha=(0.75 if is_first else 0.4), linestyle="--",
            label="Field lines  ($|B|=B_0$ along arc)" if is_first else None)

# ---- Static annotation — field equations only ----
ax.text2D(0.97, 0.97,
          r"$B_x = -B_0\,z/R_c$" + "\n"
          r"$B_z = B_0(1 - x/R_c)$" + "\n"
          r"$|B| = B_0$ along field lines",
          transform=ax.transAxes, fontsize=10, ha="right", va="top",
          color="#333333",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.82))

y_txt    = ax.text2D(0.03, 0.04, "", transform=ax.transAxes,
                     fontsize=10, color="steelblue",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.82))
time_txt = ax.text2D(0.03, 0.96, "", transform=ax.transAxes,
                     fontsize=10, color="#444444")

# ============================================================
# Animated artists — GC only, no particle trail
# ============================================================
TRAIL = 120   # full persistent trail so drift accumulation is visible

trail_gc, = ax.plot([], [], [], lw=2.5, color="darkorange",
                    label="Guiding centre")
dot_gc,   = ax.plot([], [], [], "o", color="darkorange", ms=9, zorder=11,
                    markeredgecolor="white", markeredgewidth=1.4)

ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
fig.tight_layout()

# ============================================================
# Animation — camera never moves
# ============================================================
N_FRAMES = 200
skip     = max(1, len(t) // N_FRAMES)

def update(frame):
    i  = min(frame * skip, len(t) - 1)
    i0 = max(0, i - TRAIL)

    # GC trail and dot
    trail_gc.set_data(gc[i0:i, 0], gc[i0:i, 1])
    trail_gc.set_3d_properties(gc[i0:i, 2])
    dot_gc.set_data([gc[i, 0]], [gc[i, 1]])
    dot_gc.set_3d_properties([gc[i, 2]])

    y_txt.set_text(f"$y_{{GC}}$ = {gc[i, 1]:.3f}")
    time_txt.set_text(f"t = {t[i]:.1f}  ({t[i]/T_gyro:.1f} gyrations)")

    return trail_gc, dot_gc, y_txt, time_txt

anim = animation.FuncAnimation(fig, update, frames=N_FRAMES,
                                interval=55, blit=False)

out = os.path.join(_DIR, "..", "Figures", "animate04_gradb.gif")
print(f"\nSaving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()
