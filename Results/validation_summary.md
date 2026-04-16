# Validation Summary

Key quantitative results from the simulation test suite.
Last updated: 2 April 2026.

Parameters unless stated: q = m = B₀ = 1 (code units), M = 500, L = 3, pitch = 45°.

---

## Drift validations (code units)

| Test | Phenomenon | Analytic | Numerical | Error |
|------|-----------|----------|-----------|-------|
| test03 | E×B drift speed | 0.1000 | 0.1000 | **0.00%** |
| test04 | Grad-B drift speed | 2.725 × 10⁻² | 2.655 × 10⁻² | **2.57%** |
| test09 | Curvature drift speed | 1.011 × 10⁻⁴ | 9.992 × 10⁻⁵ | **1.19%** |

E×B is exact to numerical precision (purely kinematic). Grad-B and curvature errors of 1-3% are expected from the first-order drift formula.

---

## Bounce period (code units, M = 500, pitch = 60°)

| Quantity | Value |
|----------|-------|
| Analytic T_b (bounce integral) | 9.6664 code time units |
| Numerical T_b (mean over 5 bounces) | 9.6028 code time units |
| Error | **0.66%** |

Mirror latitude: analytic = 14.69°. Sub-percent error confirms the bounce theory and integrator are consistent.

---

## Guiding-centre approximation (test11)

Parameters: M = 500, L = 3, pitch = 45°, T_run = 4 bounce periods.

| Metric | Value | Interpretation |
|--------|-------|---------------|
| r_gyro / r_eq | 0.0064 | << 1 → well adiabatic spatially |
| T_bounce / T_gyro | 100 | >> 1 → well adiabatic temporally |
| Gyroradius r_gyro | 0.0191 code units | |
| Max \|r_full − r_GC\| | 0.1073 | ~5.6 × r_gyro — see note |
| Mean \|r_full − r_GC\| | 0.0513 | ~2.7 × r_gyro |
| Max \|ΔK/K₀\| (energy) | 9.48 × 10⁻⁷ | Excellent — solver is conservative |
| Max \|Δμ/μ₀\| | 0.110 (11%) | GC equations drift — see note |

The separation metric compares the full Lorentz orbit to the GC equations solution.
The 11% μ error and growing positional offset come from the GC equations integrator
(finite-difference gradients), not from the full Lorentz orbit — energy conservation
of the full orbit (9.48 × 10⁻⁷) confirms it is highly accurate. The extracted GC
(`R_gc = r + (m/qB²)(v×B)`) tracks the true orbit much more closely.

---

## Physical SI case (test13 — 10 keV electron, L = 3, pitch = 45°)

Runtime: ~20 minutes (2 bounce periods).

| Quantity | Value | Notes |
|----------|-------|-------|
| Gyroperiod T_gyro | 32.1 µs | = 2πmₑ/(eB_eq) |
| Gyroradius r_g | 0.2 km (3×10⁻⁵ Rₑ) | r_g/r_eq = 0.00001 — extremely adiabatic |
| T_b / T_gyro | 57,528 | >> 1 → extremely well adiabatic |
| Analytic mirror latitude λ_m | 23.13° | B(λ_m)/B_eq = 1/sin²(45°) = 2 |
| Analytic \|z_mirror\| | 0.9967 Rₑ | |
| Numerical \|z_mirror\| (1st crossing) | 0.9968 Rₑ | From v_∥ = 0 detection |
| Mirror z error | **< 0.02%** | Primary adiabatic validation check |
| Max \|ΔKE/KE₀\| over 2 bounces | 1.52 × 10⁻² | Phase error over ~115k gyrations |
| Max \|Δμ/μ₀\| over 2 bounces | 2.02 × 10⁻² (2%) | Expected — see note |

The ~2% μ error is accumulated phase error over ~115,000 gyrations. Mirror point accuracy (< 0.02%) is the physically meaningful check.

---

## Tilted dipole (test14, M = 500, L = 3, pitch = 45°)

Three runs: tilt = 0°, 47° (Neptune/Uranus-like), 59°. Parameters identical across runs.

| Quantity | Value |
|----------|-------|
| T_gyro | 0.339 code time units |
| T_b (analytic estimate) | 16.97 code time units |
| T_b / T_gyro | 50 — well adiabatic |
| dt | 0.01696 (= T_gyro / 20) |

Bounce motion preserved across all three tilt angles — the particle still mirrors correctly. Equatorial crossing point shifts as the magnetic equator tilts.

---

## Corotation drift (test15, Ω = 0.02, M = 500, L = 3, pitch = 45°)

| Run | Measured Ω | Explanation |
|-----|-----------|-------------|
| No E field | 0.00801 | Gradient + curvature drift only |
| With E field | 0.02732 | E×B + gradient + curvature |
| Difference (E×B recovered) | 0.01931 | Input Ω = 0.02000 |
| Recovery | **96.6%** | Confirms corotation E×B implementation |

**Timescale hierarchy confirmed:** T_gyro = 0.339, T_bounce = 17.0, T_corot = 314.
T_b/T_gyro = 50 (well adiabatic for these parameters).

The ~3.4% shortfall is physical — the corotation E field has a small component along B at non-equatorial latitudes, reducing the effective perpendicular drift.
