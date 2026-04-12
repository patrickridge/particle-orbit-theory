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
# Check 2: field-line directions in x-z plane (quiver)
#          coloured by log|B| to show field strength variation
# ======================================================================
xs_q = np.linspace(-4.0, 4.0, 25)
zs_q = np.linspace(-4.0, 4.0, 25)
X, Z = np.meshgrid(xs_q, zs_q)

Ux   = np.zeros_like(X)
Uz   = np.zeros_like(Z)
Bstr = np.zeros_like(X)   # |B| at each grid point (for colouring)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = X[i, j]; z = Z[i, j]
        r = np.array([x, 0.0, z])
        if np.linalg.norm(r) < 0.75:      # exclude near-origin
            Ux[i, j] = np.nan
            Uz[i, j] = np.nan
            Bstr[i, j] = np.nan
            continue
        B    = B_func(r, 0.0)
        bx, bz = B[0], B[2]
        norm = np.sqrt(bx*bx + bz*bz)
        Ux[i, j]   = bx / norm
        Uz[i, j]   = bz / norm
        Bstr[i, j] = np.linalg.norm(B)

fig, ax = plt.subplots(figsize=(6, 6))

# colour the arrows by field strength
Bstr_flat = Bstr.ravel()
colours   = np.log10(np.where(np.isnan(Bstr_flat), np.nan, Bstr_flat + 1e-30))

q_plot = ax.quiver(X, Z, Ux, Uz,
                   colours,
                   pivot="mid", scale=30,
                   cmap="viridis")
cbar = plt.colorbar(q_plot, ax=ax, pad=0.02)
cbar.set_label(r"$\log_{10}|B|$", fontsize=9)

ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_title("Dipole field directions in $x$–$z$ plane ($y = 0$)", fontsize=11)
ax.set_aspect("equal")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test07_dipole_quiver_xz.png"), dpi=300)
plt.show()