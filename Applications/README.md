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

Computes bounce periods analytically using the Schulz & Lanzerotti formula — no simulation, just numerical integration of the bounce integral. Shows how bounce period varies with L-shell and equatorial pitch angle for a 10 keV electron.

**Key result:** Bounce period increases with L-shell, decreases for larger pitch angles. Test 10 verifies these numbers.

---

## test07 — Dipole field sanity checks

Before using the dipole field in particle simulations, this script verifies the implementation is correct. Two checks: (1) the on-axis field strength follows `|B| = 2M/z³`, (2) the field directions in the x-z plane show the correct dipole pattern.

**Key result:** Numerical field agrees with analytic formula to machine precision. Quiver plot shows classic dipole loops.

---

## test08 — Full orbit in dipole field (code units)

A full particle orbit in a dipole field showing all three motions at once: gyration around field lines, bounce motion between mirror points, and slow azimuthal drift around Earth. Also checks that energy and the magnetic moment μ are conserved.

**Key result:** `z(t)` bounces regularly; x-y projection shows slow drift; `μ` conserved to < 1%.

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

The central comparison of the project. Runs both a full Lorentz orbit and the guiding-centre (GC) equations simultaneously in the same dipole field, then overlays the results. The GC approximation is valid when the gyroradius is much smaller than the orbit scale.

**Key result:** GC trajectory tracks the full orbit closely; separation stays near the gyroradius and does not grow over time. `μ` conservation confirms the adiabatic regime.

---

## test12 — Dipole field lines (B&T Fig 3.2)

Draws the magnetic field lines for several L-shells in the meridional (x-z) plane, with Earth shown as a circle. Reproduces Fig 3.2 from Baumjohann & Treumann. Provides visual context for where particle orbits live relative to the Earth.

**Key result:** Field-line diagram showing L = 1.5 to 6, with foot-points on Earth's surface.

---

## test13 — 10 keV electron in Earth's dipole (SI units)

The physical application. Simulates a 10 keV electron starting at L = 3 in Earth's dipole field using real SI units (metres, Tesla, kg). Computes the gyroperiod, gyroradius, and bounce period in seconds. Compares the numerically detected mirror points to the analytic mirror latitude. Checks that the adiabatic condition holds.

**Key result:** Particle bounces at ±z_mirror matching the analytic prediction. `μ` conserved to < 1% over 4 bounce periods (~7 seconds of real time).

---

## test14 — Tilted dipole (planetary application)

Extends the dipole field to include a tilt between the magnetic and rotation axes — as seen on Neptune (47°) and Uranus (59°). A particle starting at the geographic equator (z = 0) is displaced from the magnetic equatorial plane, producing asymmetric bounce motion that differs from the aligned case.

Two runs: a long run for all three tilts (0°, 47°, 59°) to compare z(t), and a short 4-bounce run for the 3D guiding-centre orbit figure.

**Key result:** Bounce amplitude and midpoint shift with tilt; the asymmetry grows with tilt angle. 3D GC orbit shows the particle tracing a tilted drift shell.

---

## test15 — Corotation E×B drift (aligned rotating dipole)

Adds the corotation electric field E = −(Ω×r)×B to an aligned dipole. This is the inertial-frame electric field induced when the planet (and its magnetosphere) rotates at angular rate Ω. The resulting E×B drift equals Ω×r exactly — the guiding centre co-rotates with the field line.

**Key result:** GC traces a circle of radius ≈ 3 in the x-y plane over one corotation period. Angular rate measured from φ(t) agrees with Ω to < 1%. Bounce motion in z persists simultaneously.

---

## test16 — Rotating tilted dipole (full planetary magnetosphere)

The full simulation combining test14 and test15: a Neptune-like tilted dipole (47°) whose moment rotates about z at Ω = 0.02, plus the corotation electric field E = −(Ω×r)×B(r,t). B is now time-varying. The integrator evaluates B at each timestep automatically.

The particle simultaneously gyrates, bounces in the asymmetric tilted geometry, and co-rotates azimuthally — all three adiabatic motions present at once.

**Key result:** GC co-rotates at rate ≈ Ω; z(t) shows bounce modulated by the rotating asymmetric field; 3D orbit shows the tilted drift shell sweeping around the rotation axis.
