"""
Compute and visualize bounce periods for 10 keV electrons
in Earth's dipole magnetic field.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import csv

# --- Physical constants ---
Re = 6_371e3                     # Earth radius (m)
m_e = 9.10938356e-31             # electron mass (kg)
c = 2.99792458e8
E_keV = 10.0
E_J = E_keV * 1.602176634e-16    # kinetic energy (J)

# Relativistic speed (for 10 keV electrons)
gamma = 1 + E_J / (m_e * c**2)
beta = np.sqrt(1 - 1 / gamma**2)
v = beta * c


# --- Magnetic field ratio in dipole ---
def B_ratio(lam):
    """Return B(λ)/B₀ for dipole magnetic field."""
    return np.sqrt(1 + 3 * np.sin(lam)**2) / np.cos(lam)**6


# --- Mirror latitude solver ---
def mirror_lat(alpha_eq):
    """Find λ_m for given equatorial pitch angle (radians)."""
    f = lambda lam: np.sin(alpha_eq)**2 - np.cos(lam)**6 / np.sqrt(1 + 3*np.sin(lam)**2)
    return brentq(f, 0, np.pi/2 - 1e-4)


# --- Integrand of bounce integral ---
def integrand(lam, alpha_eq):
    """Integrand for the bounce time integral."""
    return (np.cos(lam) * np.sqrt(1 + 3*np.sin(lam)**2)) / np.sqrt(
        1 - (np.sin(alpha_eq)**2) * B_ratio(lam)
    )


# --- Bounce period calculator ---
def bounce_time(L, alpha_deg):
    """Compute bounce period τ_b (s) for given L-shell and pitch angle."""
    alpha_eq = np.radians(alpha_deg)
    lam_m = mirror_lat(alpha_eq)
    I, _ = quad(integrand, 0, lam_m, args=(alpha_eq,), limit=500)
    tau_b = 4 * L * Re * I / v
    return tau_b


# --- Plot τ_b vs L ---
def plot_bounce_times():
    """Plot τ_b vs L for several equatorial pitch angles."""
    L_values = np.arange(2, 6.1, 1)
    alphas = [30, 60, 80]
    plt.figure(figsize=(7, 5))

    for alpha in alphas:
        taus = [bounce_time(L, alpha) for L in L_values]
        plt.plot(L_values, taus, marker='o', label=f'α_eq = {alpha}°')

    plt.xlabel('L-shell (r₀ / R_E)')
    plt.ylabel('Bounce period τ_b (s)')
    plt.title('10 keV Electron Bounce Periods in Earth’s Dipole Field')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figures/g_bounce_times.png", dpi=300)
    plt.show()


# --- Save results to CSV ---
def save_results(filename="Results/g_bounce_times.csv"):
    """Save computed bounce times to CSV file."""
    L_values = [2, 3, 4, 5, 6]
    alphas = [30, 60, 80]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["L", "alpha_eq_deg", "tau_b_s"])
        for alpha in alphas:
            for L in L_values:
                tau = bounce_time(L, alpha)
                writer.writerow([L, alpha, tau])

    print(f"Results saved to '{filename}'")


# --- Main entry point ---
if __name__ == "__main__":
    # Print results to screen
    for alpha in [30, 60, 80]:
        for L in [2, 3, 4, 5, 6]:
            tau = bounce_time(L, alpha)
            print(f"L={L:>2}, α_eq={alpha:>2}° → τ_b = {tau:.3f} s")

    # Plot results
    plot_bounce_times()

    # Save to CSV
    save_results()