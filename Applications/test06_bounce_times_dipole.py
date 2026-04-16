import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
from scipy.optimize import brentq
import pandas as pd

# output directory
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
os.makedirs(_FIG, exist_ok=True)

sns.set_theme(style="ticks", context="paper")

# Test 6: bounce periods in dipole field
# tau_b vs L-shell and pitch angle (Schulz & Lanzerotti)

# physical constants
e   = 1.602176634e-19       # C
m_e = 9.1093837015e-31      # kg
R_E = 6371e3                # m

# 10 keV electron
E_keV = 10.0
E_J   = E_keV * 1e3 * e
v     = np.sqrt(2.0 * E_J / m_e)
print(f"10 keV electron speed: {v:.4e} m/s  ({v/3e8*100:.2f}% of c)")

# helpers

def B_over_Beq(lam):
    """B(lambda)/B_eq along a dipole field line."""
    s, c = np.sin(lam), np.cos(lam)
    return np.sqrt(1.0 + 3.0 * s * s) / (c ** 6)

def mirror_latitude(alpha_eq):
    """Latitude of mirror point for equatorial pitch angle alpha_eq (rad)."""
    target = 1.0 / np.sin(alpha_eq)**2
    return brentq(lambda lam: B_over_Beq(lam) - target,
                  0.0, 0.5 * np.pi - 1e-6, maxiter=200)

def bounce_period(L, alpha_eq):
    """Bounce period in seconds for L-shell L and pitch angle alpha_eq (rad)."""
    lam_m = mirror_latitude(alpha_eq)
    s2    = np.sin(alpha_eq)**2

    def integrand(lam):
        c    = np.cos(lam)
        root = np.sqrt(1.0 + 3.0 * np.sin(lam)**2)
        val  = max(1.0 - s2 * B_over_Beq(lam), 0.0)
        return c * root / np.sqrt(val)

    I, _ = quad(integrand, 0.0, lam_m, points=[lam_m], limit=300)
    return 4.0 * L * R_E * I / v

# sweep L and pitch angle
Ls         = np.array([2, 3, 4, 5, 6], dtype=float)
alpha_degs = [30, 60, 80]

rows = []
for adeg in alpha_degs:
    a         = np.deg2rad(adeg)
    lam_m_deg = np.rad2deg(mirror_latitude(a))
    print(f"alpha_eq = {adeg:2d}°  →  mirror latitude = {lam_m_deg:.2f}°")
    for L in Ls:
        tau = bounce_period(L, a)
        rows.append({"L": L, "alpha_eq_deg": adeg, "tau_b_s": tau})

df = pd.DataFrame(rows)

# plot tau_b vs L
fig, ax = plt.subplots(figsize=(6, 4))

for adeg in alpha_degs:
    sub  = df[df["alpha_eq_deg"] == adeg]
    ax.plot(sub["L"], sub["tau_b_s"], marker="o", lw=1.3,
            label=fr"$\alpha_{{eq}}={adeg}^\circ$")

ax.set_xlabel("L-shell")
ax.set_ylabel(r"Bounce period $\tau_b$ (s)")
ax.set_title("10 keV Electron Bounce Periods — Earth's Dipole Field")
ax.legend(frameon=True, loc="upper left")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test06_bounce_times.png"), dpi=300)
plt.show()

# plot tau_b vs pitch angle
L_fixed    = 4.0
a_arr      = np.deg2rad(np.linspace(10, 85, 80))
tau_arr    = [bounce_period(L_fixed, a) for a in a_arr]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.rad2deg(a_arr), tau_arr, lw=1.3)
ax.set_xlabel(r"Equatorial pitch angle $\alpha_{eq}$ (deg)")
ax.set_ylabel(r"Bounce period $\tau_b$ (s)")
ax.set_title(fr"Bounce period vs pitch angle  ($L={L_fixed}$, 10 keV $e^-$)")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "test06_bounce_vs_pitch.png"), dpi=300)
plt.show()

# save table
df.to_csv("Results/test06_bounce_times.csv", index=False)
print("\nSaved Results/test06_bounce_times.csv")
print(df.to_string(index=False))