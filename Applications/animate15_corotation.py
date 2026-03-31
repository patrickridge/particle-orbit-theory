"""
animate15_corotation.py
=======================
Animated comparison of GC drift with and without the corotation E field.
Restructured to tell the physics story clearly:

Left panel:  No E field — gradient+curvature drift only (Ω ≈ 0.008).
             Planet arm rotates at Ω = 0.02. GC falls far behind → no co-rotation.

Right panel: Corotation E field — total drift (Ω ≈ 0.027).
             Planet arm at Ω = 0.02. GC tracks the arm closely (E×B ≈ Ω)
             but is slightly ahead (gradient drift adds a small extra contribution).

Story: E×B drift ≈ Ω (planet rate). Gradient drift adds a small correction on top.
Without E: particle barely moves. With E: particle approximately co-rotates.

Saves: ../Figures/animate15_corotation.gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from orbit_ivp_core import simulate_orbit_ivp, extract_gc

_DIR = os.path.dirname(os.path.abspath(__file__))
from fields import E_zero, E_corotation, B_dipole_cartesian

# ---- Parameters (match test15) ----
q, m  = 1.0, 1.0
M     = 500.0
Omega = 0.02

B_func   = B_dipole_cartesian(M=M, tilt_deg=0.0)
E_func   = E_corotation(B_func, Omega=Omega)
E_func_0 = E_zero

r0    = np.array([3.0, 0.0, 0.0])
v_mag = 1.0
pitch = np.deg2rad(45.0)

B0_vec    = B_func(r0, 0.0)
bhat      = B0_vec / np.linalg.norm(B0_vec)
v0        = v_mag * (np.cos(pitch) * bhat
                    + np.sin(pitch) * np.array([0.0, 1.0, 0.0]))
state0    = np.concatenate([r0, v0])

Omega_gyro = abs(q) * np.linalg.norm(B0_vec) / m
T_gyro     = 2.0 * np.pi / Omega_gyro
T_b_est    = 4.0 * r0[0] / abs(np.dot(v0, bhat))
T_cor      = 2.0 * np.pi / Omega
L0         = np.linalg.norm(r0)

dt     = min(T_b_est / 300.0, 0.05 * T_gyro)
nsteps = int(T_cor / dt) + 1
skip   = max(1, int(round(T_gyro / dt)))

print(f"T_gyro={T_gyro:.3f}, T_bounce={T_b_est:.2f}, T_cor={T_cor:.1f}")
print(f"dt={dt:.4f}, nsteps={nsteps}")

Omega_10x  = 10.0 * Omega   # 10× faster planetary rotation
E_func_10x = E_corotation(B_func, Omega=Omega_10x)

print("Integrating with E field ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                              q=q, m=m, E_func=E_func, B_func=B_func,
                              rtol=1e-9, atol=1e-9)
print("Integrating without E field ...")
_, traj_noE = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                                  q=q, m=m, E_func=E_func_0, B_func=B_func,
                                  rtol=1e-9, atol=1e-9)
print("Integrating with 10× E field ...")
_, traj_10x = simulate_orbit_ivp(state0=state0, dt=dt, nsteps=nsteps,
                                  q=q, m=m, E_func=E_func_10x, B_func=B_func,
                                  rtol=1e-9, atol=1e-9)
print("Done.")

gc     = extract_gc(traj,    t,    B_func, q=q, m=m)
gc_noE = extract_gc(traj_noE, t,   B_func, q=q, m=m)
gc_10x = extract_gc(traj_10x, t,   B_func, q=q, m=m)

gc_s     = gc[::skip];    t_s = t[::skip]
gc_noE_s = gc_noE[::skip]
gc_10x_s = gc_10x[::skip]

# Measure actual drift rates
phi_noE = np.unwrap(np.arctan2(gc_noE_s[:, 1], gc_noE_s[:, 0]))
phi_E   = np.unwrap(np.arctan2(gc_s[:, 1], gc_s[:, 0]))
phi_10x = np.unwrap(np.arctan2(gc_10x_s[:, 1], gc_10x_s[:, 0]))
n  = len(t_s)
i0, i1 = n // 10, 9 * n // 10
Om_noE  = np.polyfit(t_s[i0:i1], phi_noE[i0:i1], 1)[0]
Om_E    = np.polyfit(t_s[i0:i1], phi_E[i0:i1],   1)[0]
Om_10x  = np.polyfit(t_s[i0:i1], phi_10x[i0:i1], 1)[0]
print(f"No-E drift rate: Ω = {Om_noE:.4f}")
print(f"With-E drift rate: Ω = {Om_E:.4f}")
print(f"With-10×E drift rate: Ω = {Om_10x:.4f}")
print(f"E×B contribution: ΔΩ ≈ {Om_E - Om_noE:.4f}  (input Ω = {Omega})")

# ---- Figure ----
fig, (ax_L, ax_R, ax_3) = plt.subplots(1, 3, figsize=(19, 6.5))
fig.suptitle(
    r"Co-rotation E×B drift — aligned dipole"
    "\nTop-down view over one full planetary rotation",
    fontsize=14, fontweight="bold"
)

theta_c  = np.linspace(0, 2 * np.pi, 300)
r_planet = 1.0

for ax in [ax_L, ax_R, ax_3]:
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=12); ax.set_ylabel("y", fontsize=12)
    ax.tick_params(labelsize=11)
    # L-shell reference
    ax.plot(L0 * np.cos(theta_c), L0 * np.sin(theta_c),
            "k--", lw=0.8, alpha=0.35)
    # Planet
    ax.fill(r_planet * np.cos(theta_c), r_planet * np.sin(theta_c),
            color="lightsteelblue", zorder=5)
    ax.plot(r_planet * np.cos(theta_c), r_planet * np.sin(theta_c),
            "k-", lw=0.8, zorder=6)

ax_L.set_title(
    f"No E field  (Ω_planet = 0)\nGradient drift only  (Ω_GC ≈ {Om_noE:.3f})",
    fontsize=12, pad=8
)
ax_R.set_title(
    f"Corotation E field  (Ω = {Omega})\nE×B + gradient drift  (Ω_GC ≈ {Om_E:.3f})",
    fontsize=12, pad=8
)
ax_3.set_title(
    f"10× faster rotation  (10Ω = {Omega_10x})\nE×B + gradient drift  (Ω_GC ≈ {Om_10x:.3f})",
    fontsize=12, pad=8
)

# ---- Animated artists ----
TRAIL = 60

# Planet arm in ALL panels (rotates at respective Ω)
arm_L,     = ax_L.plot([], [], "-",  color="crimson", lw=3.5, alpha=0.85,
                        zorder=7, label=f"Planet arm (Ω = 0)")
arm_dot_L, = ax_L.plot([], [], "o",  color="crimson", ms=12, zorder=8)
arm_R,     = ax_R.plot([], [], "-",  color="crimson", lw=3.5, alpha=0.85,
                        zorder=7, label=f"Planet arm (Ω = {Omega})")
arm_dot_R, = ax_R.plot([], [], "o",  color="crimson", ms=12, zorder=8)
arm_3,     = ax_3.plot([], [], "-",  color="crimson", lw=3.5, alpha=0.85,
                        zorder=7, label=f"Planet arm (10Ω = {Omega_10x})")
arm_dot_3, = ax_3.plot([], [], "o",  color="crimson", ms=12, zorder=8)

# Left: no E
trail_L,   = ax_L.plot([], [], lw=1.8, color="C0", alpha=0.85,
                        label=f"GC  (Ω ≈ {Om_noE:.3f})")
dot_L,     = ax_L.plot([], [], "o",  color="C0", ms=9,  zorder=10)

# Right: with E
trail_R,   = ax_R.plot([], [], lw=1.8, color="C1", alpha=0.9,
                        label=f"GC  (Ω ≈ {Om_E:.3f})")
dot_R,     = ax_R.plot([], [], "o",  color="C1", ms=9,  zorder=10)

# Third: with 10× E
trail_3,   = ax_3.plot([], [], lw=1.8, color="C2", alpha=0.9,
                        label=f"GC  (Ω ≈ {Om_10x:.3f})")
dot_3,     = ax_3.plot([], [], "o",  color="C2", ms=9,  zorder=10)

for ax in [ax_L, ax_R, ax_3]:
    ax.legend(fontsize=11, loc="upper right")

# Physics annotation boxes
ax_L.text(0.03, 0.12,
          "No E field → no co-rotation",
          transform=ax_L.transAxes, fontsize=10, color="C0",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax_R.text(0.03, 0.12,
          f"E×B ≈ Ω → co-rotation",
          transform=ax_R.transAxes, fontsize=10, color="C1",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax_3.text(0.03, 0.12,
          f"E×B ≈ 10Ω → fast co-rotation",
          transform=ax_3.transAxes, fontsize=10, color="C2",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# Time label
txt = fig.text(0.5, 0.01, "", ha="center", fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

# ---- Animation ----
N_FRAMES  = 220
skip_anim = max(1, len(t_s) // N_FRAMES)

def update(frame):
    j     = frame * skip_anim
    j     = min(j, len(t_s) - 1)
    j0    = max(0, j - TRAIL)
    t_now = t_s[j]

    # Rotating planet arms — each at its own Ω
    # Left: reference arm at Ω (shows how far GC falls behind)
    phi_ref   = Omega * t_now
    px_ref    = L0 * np.cos(phi_ref);  py_ref  = L0 * np.sin(phi_ref)
    arm_L.set_data([0, px_ref], [0, py_ref])
    arm_dot_L.set_data([px_ref], [py_ref])

    # Middle: arm at Ω
    px_mid = px_ref;  py_mid = py_ref
    arm_R.set_data([0, px_mid], [0, py_mid])
    arm_dot_R.set_data([px_mid], [py_mid])

    # Right: arm at 10Ω
    phi_10x   = Omega_10x * t_now
    px_10     = L0 * np.cos(phi_10x);  py_10 = L0 * np.sin(phi_10x)
    arm_3.set_data([0, px_10], [0, py_10])
    arm_dot_3.set_data([px_10], [py_10])

    # Left — no E
    trail_L.set_data(gc_noE_s[j0:j, 0], gc_noE_s[j0:j, 1])
    dot_L.set_data([gc_noE_s[j, 0]], [gc_noE_s[j, 1]])

    # Middle — with E
    trail_R.set_data(gc_s[j0:j, 0], gc_s[j0:j, 1])
    dot_R.set_data([gc_s[j, 0]], [gc_s[j, 1]])

    # Right — with 10× E
    trail_3.set_data(gc_10x_s[j0:j, 0], gc_10x_s[j0:j, 1])
    dot_3.set_data([gc_10x_s[j, 0]], [gc_10x_s[j, 1]])

    pct = 100 * t_now / T_cor
    txt.set_text(f"t = {t_now:.0f} / {T_cor:.0f}  —  {pct:.0f}% of one planetary rotation")

    return (trail_L, dot_L, trail_R, dot_R, trail_3, dot_3,
            arm_L, arm_dot_L, arm_R, arm_dot_R, arm_3, arm_dot_3, txt)

n_anim = min(N_FRAMES, len(t_s) // skip_anim)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=50, blit=False)

plt.tight_layout(rect=[0, 0.04, 1, 1])

out = os.path.join(_DIR, "..", "Figures", "animate15_corotation.gif")
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=20, dpi=120)
print("Done.")
plt.show()
