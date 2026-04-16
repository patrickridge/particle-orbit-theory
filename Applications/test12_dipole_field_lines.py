import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# output directory
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# Test 12: dipole field line geometry (cf. B&T Fig 3.2)
# Check: field lines follow r = L cos^2(lambda)

# L-shells
L_shells = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

fig, ax = plt.subplots(figsize=(7, 8))

# Earth
theta_e = np.linspace(0, 2 * np.pi, 300)
ax.fill(np.cos(theta_e), np.sin(theta_e),
        color="lightgray", zorder=3, label="Earth ($R_E$)")
ax.plot(np.cos(theta_e), np.sin(theta_e), "k-", lw=1.2, zorder=4)

# field lines
for L in L_shells:
    # latitude range outside Earth's surface
    lam_E = np.arccos(np.sqrt(1.0 / L))   # latitude where field line hits Earth
    lam   = np.linspace(-lam_E, lam_E, 600)

    r = L * np.cos(lam)**2
    x = r * np.cos(lam)
    z = r * np.sin(lam)

    # both halves of meridional plane
    ax.plot( x, z, "b-", lw=1.0, zorder=2)
    ax.plot(-x, z, "b-", lw=1.0, zorder=2)

    # label
    ax.text(L + 0.08, 0.0, f"$L={L}$", fontsize=7,
            va="center", ha="left", color="navy")

# annotations
ax.axhline(0.0, color="gray", lw=0.7, ls="--", zorder=1)   # equatorial plane
ax.axvline(0.0, color="gray", lw=0.7, ls=":",  zorder=1)   # dipole axis

# r_eq annotation
L_annot = 4.0
ax.annotate(
    "", xy=(L_annot, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="<->", color="k", lw=1.0),
    zorder=5
)
ax.text(L_annot / 2, 0.25, r"$r_{eq}$", ha="center", fontsize=10)

# lambda annotation
L_lam = 3.0
lam_annot = np.deg2rad(35)
r_annot = L_lam * np.cos(lam_annot)**2
x_a = r_annot * np.cos(lam_annot)
z_a = r_annot * np.sin(lam_annot)
ax.annotate(
    "", xy=(x_a, z_a), xytext=(0, 0),
    arrowprops=dict(arrowstyle="-", color="k", lw=0.8, linestyle="dashed"),
    zorder=5
)
# arc for lambda
arc_r = 0.8
lam_arc = np.linspace(0, lam_annot, 50)
ax.plot(arc_r * np.cos(lam_arc), arc_r * np.sin(lam_arc), "k-", lw=0.9)
ax.text(0.55, 0.22, r"$\lambda$", ha="center", fontsize=11)

# r label
ax.text(x_a + 0.15, z_a / 2, r"$r$", ha="left", fontsize=11)

# formatting
ax.set_xlabel(r"$x$ / $R_E$", fontsize=11)
ax.set_ylabel(r"$z$ / $R_E$", fontsize=11)
ax.set_title("Dipolar magnetic field lines\n"
             "(after Baumjohann & Treumann Fig. 3.2)")
ax.set_xlim(-7.2, 7.2)
ax.set_ylim(-5.5, 5.5)
ax.set_aspect("equal")

earth_patch = mpatches.Patch(facecolor="lightgray", edgecolor="k", label=r"Earth ($R_E = 1$)")
ax.legend(handles=[earth_patch], frameon=True, loc="upper right", fontsize=8)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test12_dipole_field_lines.png"), dpi=300)
plt.show()
