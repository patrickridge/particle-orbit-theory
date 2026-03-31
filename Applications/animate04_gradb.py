"""
animate04_gradb.py
==================
Side-by-side 2D animation showing grad-B drift and curvature drift separately.

Left panel  (x-y view): B_gradx_z field, v_perp only, v_par = 0.
  → grad-B drift. Larmor radius larger where B is weaker. GC drifts in +y.
  → No curvature drift (field lines are straight).

Right panel (x-z view): B_curved_z field, v_par only, v_perp = 0.
  → curvature drift ONLY. Since v_perp = 0, μ = 0 and the grad-B term
    vanishes exactly. Particle slides along a curved field line in x-z and
    drifts in -y (shown as live annotation).
  → No grad-B drift (μ = 0).

Field line geometry (right panel):
  B_curved_z gives field lines that are circular arcs in the x-z plane,
  centred at (R_c, 0).  ∇·B = 0 exactly (Bx = -B0*z/R_c, Bz = B0*(1-x/R_c)
  do not depend on x and z respectively).
  Keep |x|, |z| << R_c for the approximation to hold well.

Saves: ../Figures/animate04_gradb.gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, B_gradx_z, B_curved_z

_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LEFT PANEL — grad-B drift
# ============================================================
q, m  = 1.0, 1.0
B0    = 1.0
eps   = 0.3

B_grad  = B_gradx_z(B0=B0, eps=eps)
Omega_c = abs(q) * B0 / m
T_gyro  = 2.0 * np.pi / Omega_c
r_L     = 1.0
v_perp  = r_L * Omega_c    # v_par = 0 → only grad-B drift

r0_L    = np.array([r_L, 0.0, 0.0])
v0_L    = np.array([0.0, v_perp, 0.0])
state0_L = np.concatenate([r0_L, v0_L])

T_run_L = 6.0 * T_gyro
dt_L    = T_gyro / 60.0
nsteps_L = int(T_run_L / dt_L)

v_gradB_theory = (m * v_perp**2 * eps) / (2.0 * q * B0)

print("Integrating grad-B case ...")
t_L, traj_L = simulate_orbit_ivp(state0=state0_L, dt=dt_L, nsteps=nsteps_L,
                                  q=q, m=m, E_func=E_zero, B_func=B_grad)
gc_L = extract_gc(traj_L, t_L, B_grad, q=q, m=m)
print("Done.")

# ============================================================
# RIGHT PANEL — curvature drift (v_perp = 0, so μ = 0)
# ============================================================
R_c   = 10.0
v_par = 2.5    # only parallel velocity — no gyration, no grad-B drift

B_curv  = B_curved_z(B0=B0, R_c=R_c)

# Keep arc length v_par*T_run_R << R_c*π/2 so |x|,|z| << R_c
T_run_R  = 5.0
dt_R     = 0.02
nsteps_R = int(T_run_R / dt_R)

r0_R    = np.array([0.0, 0.0, 0.0])
v0_R    = np.array([0.0, 0.0, v_par])   # along B at origin (B = B0 ẑ there)
state0_R = np.concatenate([r0_R, v0_R])

# Analytic curvature drift speed at origin: v_curv = m*v_par²/(q*B0*R_c)
v_curv_theory = (m * v_par**2) / (q * B0 * R_c)   # in -y direction

print("Integrating curvature case ...")
t_R, traj_R = simulate_orbit_ivp(state0=state0_R, dt=dt_R, nsteps=nsteps_R,
                                  q=q, m=m, E_func=E_zero, B_func=B_curv)
print("Done.")

# ============================================================
# Figure
# ============================================================
fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(13, 7))
fig.suptitle("Grad-B Drift  ·  Curvature Drift",
             fontsize=15, fontweight="bold", y=0.99)

# ---- Left panel — x-y view ----
ax_L.set_facecolor("#F5F8FF")
ax_L.set_xlabel("x", fontsize=12)
ax_L.set_ylabel("y", fontsize=12)
ax_L.tick_params(labelsize=10)
ax_L.axhline(0, color="#cccccc", lw=0.5, ls=":")
ax_L.axvline(0, color="#cccccc", lw=0.5, ls=":")

pad_L = 1.8
gc0_L = gc_L[0]
ax_L.set_xlim(gc0_L[0] - pad_L - 0.5, gc0_L[0] + pad_L)
ax_L.set_ylim(gc_L[:, 1].min() - 1.0, gc_L[:, 1].max() + 1.5)
ax_L.set_aspect("equal")
ax_L.set_title(r"Grad-B drift  —  $\mathbf{B} = B_0(1+\varepsilon x)\,\hat{z}$"
               + f"\n" + r"$v_\perp \neq 0,\; v_\parallel = 0$"
               + f"    (theory: {v_gradB_theory:.3f})",
               fontsize=11, pad=8)

# ∇B arrow
xl = ax_L.get_xlim()
yl = ax_L.get_ylim()
arr_y = yl[1] - 0.7
ax_L.annotate("", xy=(xl[1] - 0.3, arr_y),
              xytext=(xl[0] + 0.3, arr_y),
              arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=2.0))
ax_L.text((xl[0] + xl[1]) / 2, arr_y + 0.25,
          r"increasing $|\mathbf{B}|$", ha="center",
          fontsize=10, color="steelblue", fontweight="bold")

# ⊙ B out of page
cx_L, cy_L, cr_L = xl[0] + 0.45, yl[0] + 0.45, 0.22
ax_L.add_patch(mpatches.Circle((cx_L, cy_L), cr_L,
               fill=False, edgecolor="#1a4a8a", lw=1.8, zorder=6))
ax_L.plot(cx_L, cy_L, "o", color="#1a4a8a", ms=4, zorder=7)
ax_L.text(cx_L + cr_L + 0.12, cy_L, r"$\mathbf{B}$",
          va="center", fontsize=12, color="#1a4a8a", fontweight="bold")

# Ghost orbit
ax_L.plot(traj_L[:, 0], traj_L[:, 1], lw=0.4, alpha=0.08, color="C0")

# ---- Right panel — x-z view ----
ax_R.set_facecolor("#FFF8F0")   # warm tint to distinguish panels
ax_R.set_xlabel("x", fontsize=12)
ax_R.set_ylabel("z", fontsize=12)
ax_R.tick_params(labelsize=10)
ax_R.axhline(0, color="#cccccc", lw=0.5, ls=":")
ax_R.axvline(0, color="#cccccc", lw=0.5, ls=":")

pad_R = 1.5
x_r   = traj_R[:, 0];  z_r = traj_R[:, 2]
ax_R.set_xlim(x_r.min() - pad_R, x_r.max() + pad_R + 1.0)
ax_R.set_ylim(z_r.min() - pad_R, z_r.max() + pad_R + 1.5)
ax_R.set_aspect("equal")
ax_R.set_title(r"Curvature drift  —  $\mathbf{B}$ curves in $x$-$z$ plane  ($R_c=$"
               + f"{R_c:.0f})"
               + f"\n" + r"$v_\perp = 0,\; v_\parallel \neq 0$   ($\mu=0$, no grad-B term)"
               + f"    (theory: {v_curv_theory:.3f})",
               fontsize=11, pad=8)

# Draw field lines as circular arcs centred at (R_c, 0) in x-z
theta_arc = np.linspace(np.radians(100), np.radians(260), 200)
for r_arc in [4.0, 7.0, 10.0, 13.0, 16.0]:
    xa = R_c + r_arc * np.cos(theta_arc)
    za = r_arc * np.sin(theta_arc)
    lw = 2.0 if abs(r_arc - R_c) < 0.5 else 0.8
    al = 0.65 if abs(r_arc - R_c) < 0.5 else 0.30
    ax_R.plot(xa, za, color="#8B5E3C", lw=lw, alpha=al)

# Label the field line the particle follows
theta_label = np.radians(175)
x_fl_lbl = R_c + R_c * np.cos(theta_label) + 0.3
z_fl_lbl = R_c * np.sin(theta_label) + 0.3
ax_R.text(x_fl_lbl, z_fl_lbl, "field line", fontsize=9,
          color="#8B5E3C", rotation=-15, alpha=0.75)

# ⊙ B near-origin (B ≈ B0 ẑ at origin, i.e. out of x-z plane = into/out of page
# → show ⊙ indicating B approximately out of the x-z page at origin)
xl_R = ax_R.get_xlim();  yl_R = ax_R.get_ylim()
cx_R, cy_R, cr_R = xl_R[0] + 0.55, yl_R[0] + 0.55, 0.28
ax_R.add_patch(mpatches.Circle((cx_R, cy_R), cr_R,
               fill=False, edgecolor="#1a4a8a", lw=1.8, zorder=6))
ax_R.plot(cx_R, cy_R, "o", color="#1a4a8a", ms=4, zorder=7)
ax_R.text(cx_R + cr_R + 0.15, cy_R, r"$\mathbf{B}$ (at origin)",
          va="center", fontsize=10, color="#1a4a8a", fontweight="bold")

# Static arrow showing drift direction (in -y, annotated on plot)
xl_R2 = ax_R.get_xlim()
yl_R2 = ax_R.get_ylim()
ax_R.text(0.97, 0.97,
          r"GC drifts in $-\hat{y}$" + f"\n≈ {v_curv_theory:.3f} per unit time",
          transform=ax_R.transAxes, fontsize=11, ha="right", va="top",
          color="darkorange", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# ============================================================
# Animated artists
# ============================================================
TRAIL_L = 130
TRAIL_R = 120

# Left panel
trail_L, = ax_L.plot([], [], lw=1.0, alpha=0.5, color="C0",
                     label="Particle orbit")
dot_L,   = ax_L.plot([], [], "o", color="C0", ms=8, zorder=10)
gc_trail_L, = ax_L.plot([], [], lw=2.8, color="darkorange",
                         label=f"Guiding centre")
gc_dot_L,   = ax_L.plot([], [], "o", color="darkorange", ms=11, zorder=11,
                         markeredgecolor="white", markeredgewidth=1.5)
ax_L.legend(fontsize=10, loc="lower right", framealpha=0.9)

# Right panel — particle = GC (no gyration), show in x-z
trail_R, = ax_R.plot([], [], lw=2.5, color="darkorange",
                     label="Particle / GC  (x-z)")
dot_R,   = ax_R.plot([], [], "o", color="darkorange", ms=11, zorder=11,
                     markeredgecolor="white", markeredgewidth=1.5)
ax_R.legend(fontsize=10, loc="upper left", framealpha=0.9)

# Live y-position annotation (the drift we can't see in x-z view)
y_txt = ax_R.text(0.97, 0.03, "", transform=ax_R.transAxes,
                  fontsize=12, ha="right", va="bottom", color="steelblue",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# Gyration counter
gyration_txt = fig.text(0.27, 0.01, "", ha="center", fontsize=10, color="#444444")
time_txt_R   = fig.text(0.73, 0.01, "", ha="center", fontsize=10, color="#444444")

fig.tight_layout(rect=[0, 0.04, 1, 0.98])

# ============================================================
# Animation
# ============================================================
N_FRAMES = 180
skip_L   = max(1, len(t_L) // N_FRAMES)
skip_R   = max(1, len(t_R) // N_FRAMES)

def update(frame):
    # Left — grad-B
    i_L  = min(frame * skip_L, len(t_L) - 1)
    i0_L = max(0, i_L - TRAIL_L)

    trail_L.set_data(traj_L[i0_L:i_L, 0], traj_L[i0_L:i_L, 1])
    dot_L.set_data([traj_L[i_L, 0]], [traj_L[i_L, 1]])
    gc_trail_L.set_data(gc_L[:i_L, 0], gc_L[:i_L, 1])
    gc_dot_L.set_data([gc_L[i_L, 0]], [gc_L[i_L, 1]])
    gyration_txt.set_text(f"t = {t_L[i_L]:.1f}  ({t_L[i_L]/T_gyro:.1f} gyrations)")

    # Right — curvature (x-z view)
    i_R  = min(frame * skip_R, len(t_R) - 1)
    i0_R = max(0, i_R - TRAIL_R)

    trail_R.set_data(traj_R[i0_R:i_R, 0], traj_R[i0_R:i_R, 2])
    dot_R.set_data([traj_R[i_R, 0]], [traj_R[i_R, 2]])

    y_now = traj_R[i_R, 1]
    y_txt.set_text(f"y = {y_now:.3f}  (drift out of plane)")
    time_txt_R.set_text(f"t = {t_R[i_R]:.2f}")

    return trail_L, dot_L, gc_trail_L, gc_dot_L, trail_R, dot_R, y_txt, gyration_txt, time_txt_R

n_anim = N_FRAMES
anim   = animation.FuncAnimation(fig, update, frames=n_anim,
                                  interval=55, blit=False)

out = os.path.join(_DIR, "..", "Figures", "animate04_gradb.gif")
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()
