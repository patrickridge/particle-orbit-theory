"""Animate particle bounce motion in a dipole field."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from orbit_ivp_core import simulate_orbit_ivp, extract_gc

_DIR = os.path.dirname(os.path.abspath(__file__))
from fields import E_zero, B_dipole_cartesian

# parameters (same as test08)
q, m, M = 1.0, 1.0, 50.0
B_func  = B_dipole_cartesian(M=M)

r0     = np.array([3.0, 0.0, 0.0])
B0_vec = B_func(r0, 0.0)
bhat   = B0_vec / np.linalg.norm(B0_vec)
eperp  = np.array([0.0, 1.0, 0.0])
v_mag, pitch = 1.0, np.deg2rad(60.0)
v0     = v_mag * (np.cos(pitch) * bhat + np.sin(pitch) * eperp)
state0 = np.concatenate([r0, v0])

Omega_gyro = abs(q) * np.linalg.norm(B0_vec) / m
T_gyro     = 2.0 * np.pi / Omega_gyro
v_par_mag  = v_mag * np.cos(pitch)
T_b_est    = 4.0 * r0[0] / v_par_mag

# azimuthal drift rate estimate
sin_p       = np.sin(pitch)
Omega_drift = 3.0 * v_mag**2 * (1.0 + sin_p**2 / 2.0) / (2.0 * Omega_gyro * r0[0]**2)
T_drift     = 2.0 * np.pi / Omega_drift
# Neptune rotation period in seconds
T_Neptune_s = 58000.0
ratio_drift = T_drift / T_Neptune_s

# ~30 points per gyration
dt_anim = T_gyro / 30.0
T_run   = 2.5 * T_b_est
nsteps  = int(T_run / dt_anim)

print(f"T_gyro={T_gyro:.3f}, T_bounce={T_b_est:.2f}")
print(f"dt_anim={dt_anim:.4f}, nsteps={nsteps}")
print("Integrating ...")
t, traj = simulate_orbit_ivp(state0=state0, dt=dt_anim, nsteps=nsteps,
                              q=q, m=m, E_func=E_zero, B_func=B_func,
                              rtol=1e-9, atol=1e-9)
r  = traj[:, :3]
gc = extract_gc(traj, t, B_func, q=q, m=m)

# Detect mirror points from v_par sign changes
vpar = np.array([np.dot(traj[i, 3:], B_func(r[i], t[i]) /
                         np.linalg.norm(B_func(r[i], t[i])))
                 for i in range(0, len(t), max(1, len(t)//500))])
t_sub = t[::max(1, len(t)//500)]
mirror_t = t_sub[np.where(np.diff(np.sign(vpar)))[0]]
print(f"Mirror points at t ≈ {mirror_t}")
print("Done.")

# figure setup
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")

# Static dipole field lines
lam_fl = np.linspace(-1.25, 1.25, 300)
L_fl   = [2.0, 2.5, 3.0, 3.5, 4.0]
phi_fl = np.linspace(0, 2 * np.pi, 8, endpoint=False)
for phi_f in phi_fl:
    for L_f in L_fl:
        r_fl = L_f * np.cos(lam_fl) ** 2
        xf   = r_fl * np.cos(lam_fl) * np.cos(phi_f)
        yf   = r_fl * np.cos(lam_fl) * np.sin(phi_f)
        zf   = r_fl * np.sin(lam_fl)
        below = (xf ** 2 + yf ** 2 + zf ** 2) < 1.02 ** 2
        xf[below] = np.nan; yf[below] = np.nan; zf[below] = np.nan
        ax.plot(xf, yf, zf, color="steelblue", lw=0.4, alpha=0.28)

# Planet sphere
u_s = np.linspace(0, 2 * np.pi, 30)
v_s = np.linspace(0, np.pi, 20)
ax.plot_surface(
    np.outer(np.cos(u_s), np.sin(v_s)),
    np.outer(np.sin(u_s), np.sin(v_s)),
    np.outer(np.ones_like(u_s), np.cos(v_s)),
    color="lightsteelblue", alpha=0.55, zorder=0
)

# Animated artists
TRAIL = 120
trail_full, = ax.plot([], [], [], lw=0.5, alpha=0.25, color="C0")
trail_gc,   = ax.plot([], [], [], lw=1.6, color="C1")
dot_part,   = ax.plot([], [], [], "o", color="C0", ms=5,  zorder=10, label="Particle")
dot_gc_pt,  = ax.plot([], [], [], "s", color="C1", ms=6,  zorder=10, label="Guiding centre")

time_txt = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=9)

ax.text2D(0.02, 0.78,
          r"$T_{\rm gyro} \ll T_{\rm bounce} \ll T_{\rm drift}$",
          transform=ax.transAxes, fontsize=10, verticalalignment="top",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_zlim(-3.0, 3.0)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title(f"Particle orbit in dipole field \n"
             "Gyration · Bounce · Azimuthal drift", fontsize=10)
ax.legend(fontsize=8, loc="upper right")

# subsample for animation
N_FRAMES = 200
skip     = max(1, len(t) // N_FRAMES)

def update(frame):
    i  = frame * skip
    i  = min(i, len(t) - 1)
    i0 = max(0, i - TRAIL)

    trail_full.set_data(r[i0:i, 0], r[i0:i, 1])
    trail_full.set_3d_properties(r[i0:i, 2])

    trail_gc.set_data(gc[i0:i, 0], gc[i0:i, 1])
    trail_gc.set_3d_properties(gc[i0:i, 2])

    dot_part.set_data([r[i, 0]], [r[i, 1]])
    dot_part.set_3d_properties([r[i, 2]])

    dot_gc_pt.set_data([gc[i, 0]], [gc[i, 1]])
    dot_gc_pt.set_3d_properties([gc[i, 2]])

    # Slowly rotate camera
    ax.view_init(elev=25, azim=-50 + frame * 0.8)

    time_txt.set_text(f"t = {t[i]:.1f}  /  T_bounce ≈ {T_b_est:.0f}")
    return trail_full, trail_gc, dot_part, dot_gc_pt, time_txt

n_anim = min(N_FRAMES, len(t) // skip)
anim = animation.FuncAnimation(fig, update, frames=n_anim,
                                interval=50, blit=False)

fig.tight_layout()
out = os.path.join(_DIR, "..", "Figures", "animate08_bounce.gif")
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=20, dpi=120)
print("Done.")
plt.show()
