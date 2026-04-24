import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Pedagogical schematic for Section 2.5: bounce motion and the loss cone
# in a dipole field. Style follows Baumjohann & Treumann Fig. 3.5.

_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)
sns.set_theme(style="ticks", context="paper")

fig, ax = plt.subplots(figsize=(8, 7))

# Earth
theta_e = np.linspace(0, 2 * np.pi, 300)
ax.fill(np.cos(theta_e), np.sin(theta_e),
        color="lightgray", zorder=3)
ax.plot(np.cos(theta_e), np.sin(theta_e), "k-", lw=1.2, zorder=4)

# field lines (L = 2, 3, 4 on both sides)
L_shells = [2.0, 3.0, 4.0]
for L in L_shells:
    lam_E = np.arccos(np.sqrt(1.0 / L))
    lam   = np.linspace(-lam_E, lam_E, 400)
    r = L * np.cos(lam)**2
    x = r * np.cos(lam)
    z = r * np.sin(lam)
    lw = 1.8 if np.isclose(L, 3.0) else 0.7
    alpha = 1.0 if np.isclose(L, 3.0) else 0.4
    ax.plot( x, z, "b-", lw=lw, alpha=alpha, zorder=2)
    ax.plot(-x, z, "b-", lw=lw, alpha=alpha, zorder=2)

# reference L = 3 field line, highlighted
L_ref = 3.0
lam_E_ref = np.arccos(np.sqrt(1.0 / L_ref))

# mirror latitude for equatorial pitch angle alpha_eq = 45 degrees
alpha_eq = np.deg2rad(45.0)
# solve sin^2(alpha_eq) = B_eq/B_m with B(lambda)/B_eq = sqrt(1+3 sin^2 lambda)/cos^6 lambda
from scipy.optimize import brentq
def B_ratio(lam):
    s, c = np.sin(lam), np.cos(lam)
    return np.sqrt(1 + 3 * s * s) / c**6
lam_m = brentq(lambda lam: B_ratio(lam) - 1.0 / np.sin(alpha_eq)**2,
               0.0, 0.5 * np.pi - 1e-6)

# bounce trajectory on L = 3, between -lam_m and +lam_m
lam_bounce = np.linspace(-lam_m, lam_m, 200)
r_b = L_ref * np.cos(lam_bounce)**2
x_b = r_b * np.cos(lam_bounce)
z_b = r_b * np.sin(lam_bounce)
# draw the bounce path (overlaid with small gyro-wiggles for realism)
wiggle = 0.05 * np.sin(25 * lam_bounce / lam_m)
x_wig = x_b + wiggle * np.sin(lam_bounce)
z_wig = z_b - wiggle * np.cos(lam_bounce)
ax.plot(x_wig, z_wig, color="C3", lw=1.3, zorder=5, label="bounce trajectory")

# mirror points on L = 3
mx = L_ref * np.cos(lam_m)**2 * np.cos(lam_m)
mz = L_ref * np.cos(lam_m)**2 * np.sin(lam_m)
for zsign in (+1, -1):
    ax.plot(mx, zsign * mz, "o", color="green", ms=8,
            zorder=8, markeredgecolor="white", markeredgewidth=0.5)
ax.text(mx + 0.2, +mz + 0.15, "mirror point", fontsize=9, color="green")
ax.text(mx + 0.2, -mz - 0.15, "mirror point", fontsize=9, color="green",
        va="top")

# equator label and alpha_eq arrow
ax.plot(L_ref, 0, "ko", ms=5, zorder=9)
ax.text(L_ref + 0.12, 0.15, r"$\alpha_\text{eq}$", fontsize=12)
# pitch-angle arc at the equator
arc_r = 0.35
arc_ang = np.linspace(0, alpha_eq, 30)
ax.plot(L_ref + arc_r * np.cos(arc_ang),
        arc_r * np.sin(arc_ang),
        color="k", lw=0.9)
# velocity arrow at the equator illustrating pitch angle
ax.annotate("", xy=(L_ref + 0.7 * np.cos(alpha_eq), 0.7 * np.sin(alpha_eq)),
            xytext=(L_ref, 0),
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1))

# LOSS CONE shading at Earth's surface
# field-line intersects Earth at latitude ±lam_E_ref for L = 3.
# loss cone at equator: sin²(α_lc) = B_eq / B_surface.
# For dipole, B_surface/B_eq = B(lam_E)/B_eq = sqrt(1+3 sin^2 lam_E)/cos^6 lam_E.
B_surf_ratio = B_ratio(lam_E_ref)
sin2_alpha_lc = 1.0 / B_surf_ratio
alpha_lc = np.arcsin(np.sqrt(sin2_alpha_lc))

# shade loss cone at the equator as a sector of angles
# around the field direction (which at the equator is -z for positive L)
# For visualisation we draw two small wedges above and below the equator.
for zsign in (+1, -1):
    # field-aligned direction at the foot of the field line (tangent to field line)
    lam_foot = zsign * lam_E_ref
    # tangent direction at foot (pointing into the planet from outside)
    # field line: r(lam) = L cos^2(lam), x = r cos(lam), z = r sin(lam)
    dr_dlam = -2 * L_ref * np.cos(lam_foot) * np.sin(lam_foot)
    dxdl = dr_dlam * np.cos(lam_foot) - L_ref * np.cos(lam_foot)**2 * np.sin(lam_foot)
    dzdl = dr_dlam * np.sin(lam_foot) + L_ref * np.cos(lam_foot)**2 * np.cos(lam_foot)
    norm = np.hypot(dxdl, dzdl)
    tx, tz = dxdl / norm, dzdl / norm
    # wedge centred at foot point
    fx = L_ref * np.cos(lam_foot)**2 * np.cos(lam_foot)
    fz = L_ref * np.cos(lam_foot)**2 * np.sin(lam_foot)
    # but we want the wedge at the surface; fx/fz already on the L=3 shell but not
    # quite at r=1. Take surface point along field line at r=1:
    # solve L cos^2(lam) = 1 -> cos^2(lam) = 1/L -> lam = lam_E_ref
    # which is what lam_foot already represents.
    # draw two rays at ±alpha_lc around the field direction
    for sign in (-1, +1):
        ang = sign * alpha_lc
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        # rotate tangent by ±alpha_lc
        rx = cos_a * tx - sin_a * tz
        rz = sin_a * tx + cos_a * tz
        wedge_len = 0.6
        ax.plot([fx, fx + wedge_len * rx],
                [fz, fz + wedge_len * rz],
                color="orange", lw=1.2, zorder=6)
    ax.plot(fx, fz, "s", color="orange", ms=5, zorder=7)

# loss cone label (one side)
ax.text(0.55, 1.45, "loss cone", fontsize=10, color="orange",
        ha="left")
ax.annotate("", xy=(0.95, 1.1), xytext=(0.8, 1.35),
            arrowprops=dict(arrowstyle="-", color="orange", lw=0.7))

# magnetic axis
ax.axhline(0.0, color="gray", lw=0.5, ls="--", zorder=1)
ax.axvline(0.0, color="gray", lw=0.5, ls=":",  zorder=1)

ax.set_xlim(-5.0, 5.0); ax.set_ylim(-3.0, 3.0)
ax.set_aspect("equal")
ax.set_xlabel(r"$x / R_E$", fontsize=11)
ax.set_ylabel(r"$z / R_E$", fontsize=11)
ax.set_title(
    "Bounce motion between mirror points and the loss cone\n"
    r"(dipole field, reference shell $L = 3$, $\alpha_\text{eq} = 45^\circ$)",
    fontsize=11,
)

# legend
earth_patch = mpatches.Patch(facecolor="lightgray", edgecolor="k",
                             label=r"Earth ($R_E = 1$)")
trajectory_line = plt.Line2D([], [], color="C3", lw=1.3,
                              label="bounce trajectory")
mirror_pt = plt.Line2D([], [], color="green", marker="o", lw=0,
                        markersize=8, label="mirror point")
loss_cone_line = plt.Line2D([], [], color="orange", lw=1.2,
                             label="loss-cone boundary")
ax.legend(handles=[earth_patch, trajectory_line, mirror_pt, loss_cone_line],
          frameon=True, loc="upper right", fontsize=9)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "fig_2_5_bounce_loss_cone.png"), dpi=200)
plt.show()
