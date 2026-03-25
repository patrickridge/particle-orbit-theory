# Particle Orbit Theory — Numerical Simulations

A Python project numerically integrating the Lorentz force equation for charged particles in various magnetic field geometries. Written as a BSc project in plasma and magnetospheric physics.

Numerical results are compared against analytic predictions from guiding-centre (adiabatic) theory throughout.

---

## Physics Covered

| Phenomenon | Script |
|-----------|--------|
| Circular gyromotion | test01 |
| Helical motion along B | test02 |
| E×B drift | test03 |
| Grad-B (gradient) drift | test04 |
| Magnetic mirror bounce | test05 |
| Analytic bounce periods | test06 |
| Dipole field geometry | test07, test12 |
| Full orbit in dipole field | test08 |
| Curvature drift | test09 |
| Analytic vs numerical bounce period | test10 |
| Full orbit vs guiding-centre | test11 |
| 10 keV electron in Earth's dipole (SI units) | test13 |
| Tilted dipole — Neptune/Uranus geometry | test14 |
| Corotation E×B drift — aligned rotating dipole | test15 |
| Rotating tilted dipole — full planetary magnetosphere | test16 |

---

## Project Structure

```
Applications/           Core modules and test scripts (run from here)
  orbit_ivp_core.py     Lorentz force solver (scipy solve_ivp) + extract_gc()
  fields.py             Canonical field library (uniform, dipole, mirror, ...)
  guiding_centre.py     Guiding-centre equations of motion (all three drifts)
  test01_*.py  ...      Test scripts — see Applications/README.md for descriptions

Results/                Analytic scripts and output CSVs (run from project root)
  g_bounce_times.py     Analytic bounce periods for 10 keV electrons

Figures/                Auto-generated PNG output files (not tracked in git)
Notes/                  Reference textbooks (not tracked in git)
```

---

## Installation

```bash
pip install numpy scipy matplotlib seaborn
```

Tested with Python 3.10+.

---

## Running Scripts

All test scripts must be run from inside the `Applications/` directory:

```bash
cd Applications
python test01_uniformB_xy_orbit.py
python test11_gc_vs_full_orbit.py
python test13_dipole_physical_units.py
```

The analytic script in `Results/` is run from the project root:

```bash
python Results/g_bounce_times.py
```

Figures are saved to `Figures/` relative to the working directory.

---

## Key Conventions

- **Units:** Dimensionless code units (q = m = B₀ = 1) for tests 01–12, 14–16. Physical SI units (Re, me, keV) for test13 and `g_bounce_times.py`.
- **State vector:** `[x, y, z, vx, vy, vz]` (6 elements).
- **Pitch angle:** Equatorial pitch angle α_eq.
- **Adiabatic invariant:** μ = m v_⊥² / (2B); conservation quality is used as a solver accuracy diagnostic.
- **Bounce detection:** Zero-crossings of v‖ = v·B̂.
- **Guiding-centre extraction:** `R_gc = r + (m / q B²)(v × B)` — analytically removes the Larmor radius at every timestep. Implemented in `orbit_ivp_core.extract_gc`.
- **Plotting:** seaborn (`"ticks"` style, `"paper"` context) + matplotlib, 300 dpi.

---

## References

- Baumjohann & Treumann, *Basic Space Plasma Physics* (1996)
- Schulz & Lanzerotti, *Particle Diffusion in the Radiation Belts* (1974)
- Jackson, *Classical Electrodynamics* (3rd ed.)
