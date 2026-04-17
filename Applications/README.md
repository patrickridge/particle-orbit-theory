# Test Scripts — What Each One Shows

All scripts must be run from inside this `Applications/` directory:

```bash
cd Applications
python test01_uniformB_xy_orbit.py
```

---

## test01 — Circular gyromotion

A charged particle with velocity perpendicular to a uniform magnetic field moves in a perfect circle. This is the most basic motion in plasma physics. The script checks that the numerical orbit matches the analytic gyroradius and period.

**Key result:** Circular orbit in x-y plane. Numerical radius agrees with `r_L = m v_⊥ / (q B₀)`.

---

## test02 — Helical motion

Adding a velocity component parallel to **B** gives the particle a constant forward speed along the field while it still gyrates. The combined motion is a helix (corkscrew shape).

**Key result:** 3D helical trajectory. `z(t)` increases linearly while x-y gyration continues.

---

## test03 — E×B drift

With both an electric field **E** and magnetic field **B** present (at right angles), the particle's guiding centre drifts sideways — perpendicular to both fields. This drift is the same for all particle types regardless of charge or mass.

**Key result:** Guiding centre drifts at `v_ExB = (E × B) / B²`. Compared to analytic value.

---

## test04 — Grad-B drift

When the magnetic field strength varies in space, the gyration radius is slightly larger on the weak-field side than the strong-field side. This causes a slow sideways drift called the gradient drift.

**Key result:** Guiding centre drifts perpendicular to both **B** and ∇B. Speed matches `v_∇B ≈ (m v_⊥² ε) / (2 q B₀)`.

---

## test05 — Magnetic mirror bounce

A field that gets stronger toward the ends acts like a magnetic bottle — the particle bounces back and forth between two mirror points where its parallel velocity reverses. This is the mirror effect.

**Key result:** `z(t)` oscillates regularly. Turning points are detected and plotted. Energy is conserved.

---

## test06 — Analytic bounce periods (dipole)

Computes bounce periods analytically using the standard bounce-period integral — no simulation, just numerical quadrature. Shows how bounce period varies with L-shell and equatorial pitch angle for a 10 keV electron.

**Key result:** Bounce period increases with L-shell, decreases for larger pitch angles. Test 10 verifies these numbers.

---

## test07 — Dipole field sanity checks

Before using the dipole field in particle simulations, this script verifies the implementation is correct. Two checks: (1) the on-axis field strength follows `|B| = 2M/z³`, (2) the field directions in the x-z plane show the correct dipole pattern.

**Key result:** Numerical field agrees with analytic formula to machine precision. Quiver plot shows classic dipole loops.

---

## test08 — Full orbit in dipole field (code units)

A full particle orbit in a dipole field showing all three motions at once: gyration around field lines, bounce between mirror points, and slow azimuthal drift. Uses M = 50, giving T_b/T_g ≈ 7 (marginally adiabatic) — the gyration is visible in the raw orbit. The guiding centre is extracted analytically via `extract_gc` and plotted alongside the full orbit, showing how the smooth bounce envelope differs from the oscillating raw trajectory.

**Key result:** GC trace shows regular z(t) bounce; mirror points identified from v‖ = 0; `μ` conserved to a few percent (expected for marginal adiabaticity). Energy conserved to < 10⁻⁶.

---

## test09 — Curvature drift

When a particle moves along curved field lines (as in a dipole), the bending of the field causes a sideways drift proportional to `v_∥²`. This is tested using a curved-field geometry with mostly parallel initial velocity so curvature drift dominates.

**Key result:** Guiding centre drifts in y at a rate matching `v_curv ≈ (m v_∥²) / (q B₀ R_c)`.

---

## test10 — Analytic vs numerical bounce period

Runs a full numerical orbit in the dipole field, detects mirror crossings, and compares the measured bounce period directly to the analytic formula from test06. This cross-check validates both the orbit integrator and the analytic theory.

**Key result:** Numerical bounce period matches analytic value to < 1% for the given pitch angle and L-shell.

---

## test11 — Full orbit vs Guiding-Centre approximation

The central validation test of the project. Runs three simultaneous representations in the same dipole field (M = 500, well adiabatic: T_b/T_g ≈ 100):
1. Full Lorentz orbit
2. GC extracted analytically from the full orbit via `R_gc = r + (m/qB²)(v × B)`
3. GC equations of motion integrated directly

The z(t) plot overlays all three; the separation plot shows two metrics: `|r_full − r_GC_eqs|` (includes Larmor radius, ~r_gyro) and `|r_GC_extracted − r_GC_eqs|` (true GC approximation error, much smaller).

**Key result:** GC equations track the extracted GC closely; the true approximation error is far smaller than the Larmor radius and stays bounded over 4 bounce periods. `μ` conserved to < 0.1%.

---

## test12 — Dipole field lines (B&T Fig 3.2)

Draws the magnetic field lines for several L-shells in the meridional (x-z) plane, with Earth shown as a circle. Reproduces Fig 3.2 from Baumjohann & Treumann. Provides visual context for where particle orbits live relative to the Earth.

**Key result:** Field-line diagram showing L = 1.5 to 6, with foot-points on Earth's surface.

---

## test13 — 10 keV electron in Earth's dipole (SI units)

The physical application. Simulates a 10 keV electron starting at L = 3 in Earth's dipole field using real SI units (metres, Tesla, kg). Computes the gyroperiod (~32 µs), gyroradius (~0.2 km), and bounce period (~1.8 s). The z(t) plot shows the full orbit (faint) and the guiding centre extracted via `extract_gc` (solid), with mirror points marked on the GC track. Mirror latitudes are compared to the analytic formula.

**Key result:** Numerical mirror latitude agrees with analytic prediction to 0.02%; `μ` conserved to ~2 × 10⁻⁴ over 2 bounce periods (~115,000 gyrations).

---

## test14 — Tilted dipole (planetary application)

Extends the dipole field to include a tilt between the magnetic and rotation axes — as seen on Neptune (47°) and Uranus (59°). The starting position is placed on the magnetic equatorial plane for each tilt angle (r₀ = L·(cosθ, 0, −sinθ)), giving equivalent initial conditions across all cases.

Two runs: a long run comparing z(t) for three tilts (0°, 47°, 59°) to show how bounce becomes asymmetric as tilt increases, and a 15-bounce run for the 3D GC orbit. An |r_xy|(t) sanity plot confirms the GC stays near L = 3.

**Key result:** At 0° bounce is symmetric; at 47°/59° the bounce midpoint shifts and the amplitude modulates. The 3D plot shows the tilted drift shell. GC extracted via `extract_gc`.

---

## test15 — Corotation E×B drift (aligned rotating dipole)

Adds the corotation electric field E = −(Ω×r)×B to an aligned dipole. Two runs are made with the same initial conditions: one with the corotation E field, one without (E = 0). Comparing φ(t) for both isolates the pure E×B contribution from the gradient+curvature drift that is always present.

**Key result:** Without E: GC drifts azimuthally at Ω ≈ 0.008 (gradient+curvature only). With E: total drift Ω ≈ 0.027. Difference ≈ input Ω = 0.020, confirming the E×B co-rotation. Bounce in z persists throughout.

---

## test16 — Rotating tilted dipole (full planetary magnetosphere)

The full simulation combining test14 and test15: a Neptune-like tilted dipole (47°) whose moment rotates about z at Ω = 0.02, plus the corotation electric field E = −(Ω×r)×B(r,t). B is now time-varying (both tilt and rotation are required to induce a time-dependent B via Faraday's law). The integrator evaluates B at each timestep automatically.

Start on the magnetic equatorial plane (r₀ = L·(cosθ, 0, −sinθ)), same as test14. GC extracted via `extract_gc`. All three motions — gyration, bounce, co-rotation — present simultaneously. 3D plot includes tilted field lines at t=0.

**Key result:** GC co-rotates at a rate slightly above Ω (gradient+curvature drift adds to E×B); z(t) shows bounce amplitude modulated by the rotating asymmetric field; 3D orbit shows the tilted drift shell sweeping around the rotation axis over one corotation period.

---

## Animation Scripts

These produce animated GIFs saved to `../Figures/`. Same physics as the corresponding test scripts.

| Script | Description | Output |
|--------|------------|--------|
| animate02_helix | Helical orbit in 3D with rotating camera | `animate02_helix.gif` |
| animate03_exb | Top-down E×B cycloid vs stationary circle | `animate03_exb.gif` |
| animate04_gradb | Gradient-B drift with field line background | `animate04_gradb.gif` |
| animate05_mirror | Mirror bounce in converging field | `animate05_mirror.gif` |
| animate08_bounce | Full dipole orbit — gyration, bounce, drift | `animate08_bounce.gif` |
| animate14a_aligned | Aligned dipole bounce (0°) | `animate14a_aligned.gif` |
| animate14b_tilted | Tilted dipole bounce (59°, Uranus-like) | `animate14b_tilted.gif` |
| animate14_combined | Side-by-side 0° vs 59° comparison | `animate14_combined.gif` |
| animate15_corotation | Corotation: with vs without E field | `animate15_corotation.gif` |
| animate_earth_corotation | Corotation: analytic reference vs full GC | `animate_earth_corotation.gif` |
| animate_gc_gradient | Gradient drift with GC extraction | `animate_gc_gradient.gif` |
| animate16_rotating | Rotating tilted dipole over one rotation period | `animate16_rotating.gif` |
