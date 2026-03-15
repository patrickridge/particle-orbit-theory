# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python simulation project for particle orbit theory in plasma/magnetospheric physics. It numerically integrates the Lorentz force equation for charged particles in various magnetic field geometries, comparing numerical results against analytic predictions from adiabatic theory.

## Running Scripts

All test scripts in `Applications/` must be run from within the `Applications/` directory (so that local imports resolve):

```bash
cd Applications
python test01_uniformB_xy_orbit.py
python test08_orbit_dipole_bounce_drift.py
# etc.
```

The analytic/results scripts in `Results/` are run from the project root:

```bash
python Results/g_bounce_times.py
```

Figures are saved to a `Figures/` path relative to the working directory. CSVs are saved to `Results/`.

## Architecture

### Core modules (in `Applications/`)

**`orbit_ivp_core.py`** — the single reusable solver. Wraps `scipy.integrate.solve_ivp` to integrate the Lorentz equation of motion. Key function:
```python
simulate_orbit_ivp(state0, dt, nsteps, q, m, E_func, B_func, method="RK45", rtol=1e-9, atol=1e-12)
# Returns (t, traj) where traj has shape (nsteps, 6) — columns [x,y,z,vx,vy,vz]
```

**`fields.py`** — canonical field library. All field functions have signature `f(r, t) -> np.ndarray(3,)`. Factory functions return closures:
- `E_zero`, `E_const(vec)` — electric fields
- `B_uniform_z(B0)`, `B_gradx_z(B0, eps)` — uniform and gradient fields
- `B_mirror_z(B0, alpha)` — toy mirror field (∇·B ≠ 0, for demonstration only)
- `B_mirror_div_free(B0, alpha)` — divergence-free mirror model
- `B_curved_z(B0, R_c)` — curved field line geometry for curvature drift
- `B_dipole_cartesian(M)` — full magnetic dipole (physically self-consistent)
- `dipole_B_magnitude_on_axis(M)` — analytic |B|(z) on dipole axis

**Key rule:** Never define field functions inline in test files — always import from `fields.py`.

### Test scripts (in `Applications/`)

Each `testNN_*.py` demonstrates one physical phenomenon:

| Script | Physics |
|--------|---------|
| `test01` | Circular gyromotion in uniform B |
| `test02` | Helical motion (uniform B + v‖) |
| `test03` | E×B drift |
| `test04` | Grad-B drift |
| `test05` | Magnetic mirror bounce |
| `test06` | Bounce period vs pitch angle (dipole) |
| `test07` | Dipole field geometry sanity check |
| `test08` | Full orbit in dipole: bounce, drift, μ conservation, energy conservation |
| `test09` | Curvature drift |
| `test10` | Cross-check analytic vs numerical bounce period |

Each test script: sets up parameters → integrates with `simulate_orbit_ivp` → computes diagnostics → saves figures. Companion `.md` files in `Applications/` describe the goal and expected behaviour for each test.

### Results scripts (in `Results/`)

**`g_bounce_times.py`** — analytic (not numerical) computation of 10 keV electron bounce periods in Earth's dipole field using `scipy.integrate.quad` and `scipy.optimize.brentq`. Outputs a figure and CSV.

## Key Conventions

- **Units:** Dimensionless code units (q = m = B0 = 1) for test scripts. Physical SI units (with Earth radius Re, electron mass m_e) only in `Results/g_bounce_times.py`.
- **State vector:** Always `[x, y, z, vx, vy, vz]` (6-element array).
- **Pitch angle** is equatorial pitch angle α_eq; mirror latitude λ_m is found via `scipy.optimize.brentq`.
- **Adiabatic invariant** μ = m v_⊥² / (2B); conservation quality is a diagnostic for solver accuracy.
- **Bounce period** measured numerically by detecting zero-crossings of v‖ = v·B̂.
- Plotting uses `seaborn` (style `"ticks"`, context `"paper"`) + `matplotlib`. Figures saved at 300 dpi.

## Dependencies

`numpy`, `scipy`, `matplotlib`, `seaborn`
