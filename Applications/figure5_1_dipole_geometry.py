import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# Figure 5.1 (redrawn): dipole field lines with staggered L-shell labels,
# thicker L=3 reference shell, and the r_eq / lambda annotations moved
# to the left hemisphere so they no longer clash with the L labels.

L_shells   = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
L_ref      = 3.0
label_size = 12

fig, ax = plt.subplots(figsize=(8, 7.5))

# Earth
theta_e = np.linspace(0, 2 * np.pi, 300)
ax.fill(np.cos(theta_e), np.sin(theta_e),
        color="lightgray", zorder=3)
ax.plot(np.cos(theta_e), np.sin(theta_e), "k-", lw=1.2, zorder=4)

# field lines (right-half labels done here)
for L in L_shells:
    lam_E = np.arccos(np.sqrt(1.0 / L))
    lam   = np.linspace(-lam_E, lam_E, 600)

    r = L * np.cos(lam)**2
    x = r * np.cos(lam)
    z = r * np.sin(lam)

    lw = 2.5 if np.isclose(L, L_ref) else 1.0
    ax.plot( x, z, "b-", lw=lw, zorder=2)
    ax.plot(-x, z, "b-", lw=lw, zorder=2)

# staggered L-shell labels on the RIGHT hemisphere:
# odd index above (+dz), even index below (-dz), well clear of Earth
# and of the (now left-side) r_eq / lambda annotations.
dz = 1.35
for i, L in enumerate(L_shells):
    above = (i % 2 == 1)         # L=2.0, 4.0, 6.0 above
    z_lab = dz if above else -dz
    va    = "bottom" if above else "top"
    ax.annotate(
        f"$L={L}$",
        xy=(L, 0.0),
        xytext=(L, z_lab),
        fontsize=label_size,
        ha="center", va=va,
        color="navy",
        arrowprops=dict(arrowstyle="-", color="navy", lw=0.6),
        zorder=6,
    )

# equator and axis
ax.axhline(0.0, color="gray", lw=0.7, ls="--", zorder=1)
ax.axvline(0.0, color="gray", lw=0.7, ls=":",  zorder=1)

# r_eq annotation — moved to LEFT hemisphere
L_annot = 4.0
ax.annotate(
    "", xy=(-L_annot, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="<->", color="k", lw=1.0),
    zorder=5,
)
ax.text(-L_annot / 2, 0.30, r"$r_{eq}$", ha="center", fontsize=12)

# lambda annotation on the LEFT hemisphere, on the L=3 reference shell
L_lam     = 3.0
lam_annot = np.deg2rad(35)
r_annot   = L_lam * np.cos(lam_annot)**2
x_a       = -r_annot * np.cos(lam_annot)   # negative: left hemisphere
z_a       =  r_annot * np.sin(lam_annot)
ax.annotate(
    "", xy=(x_a, z_a), xytext=(0, 0),
    arrowprops=dict(arrowstyle="-", color="k", lw=0.8, linestyle="dashed"),
    zorder=5,
)
# lambda arc (on the left side, sweeping from the negative x-axis upward)
arc_r   = 0.8
lam_arc = np.linspace(np.pi - lam_annot, np.pi, 50)
ax.plot(arc_r * np.cos(lam_arc), arc_r * np.sin(lam_arc), "k-", lw=0.9)
ax.text(-0.55, 0.25, r"$\lambda$", ha="center", fontsize=12)
ax.text(x_a - 0.15, z_a / 2, r"$r$", ha="right", fontsize=12)

ax.set_xlabel(r"$x$ / $R_E$", fontsize=12)
ax.set_ylabel(r"$z$ / $R_E$", fontsize=12)
ax.set_title("Dipolar magnetic field lines\n"
             "(after Baumjohann & Treumann Fig. 3.2)", fontsize=12)
ax.set_xlim(-7.2, 7.2)
ax.set_ylim(-4.0, 4.0)
ax.set_aspect("equal")

earth_patch = mpatches.Patch(
    facecolor="lightgray", edgecolor="k", label=r"Earth ($R_E = 1$)"
)
ax.legend(handles=[earth_patch], frameon=True, loc="upper right", fontsize=10)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "fig_5_1_dipole_geometry.png"), dpi=200)
plt.show()
