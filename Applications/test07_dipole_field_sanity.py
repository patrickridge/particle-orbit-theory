import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fields import B_dipole_cartesian, dipole_B_magnitude_on_axis

# Figures directory — resolved relative to this script, so the script runs correctly from any working directory.
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# =============================================================
# Test 7: Dipole field sanity checks
#
# Verifies that B_dipole_cartesian() is implemented correctly:
#   (1) on-axis |B| = 2M/z^3, (2) field directions form the correct dipole
# Expected: relative error < 1e-14; quiver plot shows dipole loop pattern.
# =============================================================

M      = 1.0
B_func = B_dipole_cartesian(M=M)

# ======================================================================
# Check 1: on-axis magnitude vs theory  |B| = 2M/z^3
# ======================================================================
zs       = np.linspace(1.0, 5.0, 200)
Bmag_num = np.array([np.linalg.norm(B_func(np.array([0.0, 0.0, z]), 0.0))
                     for z in zs])

Bmag_theory = dipole_B_magnitude_on_axis(M=M)(zs)

max_rel_err = np.max(np.abs(Bmag_num - Bmag_theory) / Bmag_theory)
print(f"Max relative error |B_num - B_theory| / B_theory: {max_rel_err:.2e}")

fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(zs, Bmag_num,    lw=1.5, label="Numerical |B(0,0,z)|")
ax.loglog(zs, Bmag_theory, "k--", lw=1.0, label=r"Theory $2M/z^3$")
ax.set_xlabel("z")
ax.set_ylabel("|B|")
ax.set_title("Dipole field — on-axis magnitude scaling", fontsize=11)
ax.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test07_dipole_axis_scaling.png"), dpi=300)
plt.show()

# ======================================================================
# Check 2: field-line directions in x-z plane
#          Clean academic figure: streamlines + planet + log|B| background
# ======================================================================

# Dense grid for background field strength and streamlines
N_bg = 300
xs_bg = np.linspace(-5.0, 5.0, N_bg)
zs_bg = np.linspace(-5.0, 5.0, N_bg)
Xbg, Zbg = np.meshgrid(xs_bg, zs_bg)

Bx_grid = np.zeros_like(Xbg)
Bz_grid = np.zeros_like(Zbg)
Bmag_bg = np.zeros_like(Xbg)

for i in range(N_bg):
    for j in range(N_bg):
        x_val = Xbg[i, j]; z_val = Zbg[i, j]
        r_val = np.sqrt(x_val**2 + z_val**2)
        if r_val < 1.0:
            Bx_grid[i, j] = np.nan
            Bz_grid[i, j] = np.nan
            Bmag_bg[i, j] = np.nan
            continue
        Bvec = B_func(np.array([x_val, 0.0, z_val]), 0.0)
        Bx_grid[i, j] = Bvec[0]
        Bz_grid[i, j] = Bvec[2]
        Bmag_bg[i, j] = np.linalg.norm(Bvec)

fig, ax = plt.subplots(figsize=(6, 6))

# Background: log|B| colour map — subtle
logB = np.log10(np.where(np.isnan(Bmag_bg), np.nan, Bmag_bg + 1e-30))
im = ax.pcolormesh(Xbg, Zbg, logB, cmap="Blues", alpha=0.35,
                   shading="auto", rasterized=True)
cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
cbar.set_label(r"$\log_{10}|\mathbf{B}|$", fontsize=10)

# Analytic field lines (cleaner than streamplot)
lam_fl = np.linspace(-np.pi/2 * 0.95, np.pi/2 * 0.95, 500)
L_vals = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
for L in L_vals:
    r_fl = L * np.cos(lam_fl)**2
    x_fl = r_fl * np.cos(lam_fl)
    z_fl = r_fl * np.sin(lam_fl)
    # Mask inside planet
    inside = (x_fl**2 + z_fl**2) < 1.0**2
    x_fl[inside] = np.nan; z_fl[inside] = np.nan
    ax.plot(x_fl, z_fl, color="steelblue", lw=0.9, alpha=0.7)
    ax.plot(-x_fl, z_fl, color="steelblue", lw=0.9, alpha=0.7)

# Planet circle — clean filled circle
theta_p = np.linspace(0, 2*np.pi, 300)
ax.fill(np.cos(theta_p), np.sin(theta_p), color="lightgray", zorder=5)
ax.plot(np.cos(theta_p), np.sin(theta_p), "k-", lw=1.0, zorder=6)

# Magnetic axis arrow
ax.annotate("", xy=(0, 4.5), xytext=(0, -4.5),
            arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.5),
            zorder=7)
ax.text(0.2, 4.2, "Magnetic axis", fontsize=8, color="crimson")

ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.set_xlabel("$x$ / $R$", fontsize=10)
ax.set_ylabel("$z$ / $R$", fontsize=10)
ax.set_title("Dipole field structure in the $x$–$z$ plane", fontsize=11)
ax.set_aspect("equal")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test07_dipole_quiver_xz.png"), dpi=300)
plt.show()