## orbit_ivp_core (solve_ivp solver)

Goal: provide one reusable solver for the Lorentz force equations that can be used for all test cases.

Setup:
- Uses SciPy `solve_ivp` to integrate Newton’s second law with the Lorentz force.
- State vector is [x, y, z, vx, vy, vz].
- Electric and magnetic fields are passed as functions E(r,t) and B(r,t).

Expected behaviour:
- The solver advances the particle motion in time automatically.
- Changing the physics only requires changing E_func or B_func, not the solver itself.