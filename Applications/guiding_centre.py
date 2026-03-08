import numpy as np
from scipy.integrate import solve_ivp

# =============================================================
# guiding_centre.py
# Guiding-centre (GC) equations of motion.
#
# State vector:  [X, Y, Z, v_par]   (4 elements)
# Conserved:     mu = m * v_perp^2 / (2 * |B|)   (adiabatic invariant)
#
# Equations of motion (Baumjohann & Treumann, Ch. 2):
#
#   dR/dt    = v_par * b_hat + v_ExB + v_gradB + v_curv
#
#   dv_par/dt = (q/m) * E_par  -  (mu/m) * (b_hat . grad|B|)
#                                            ^-- mirror force
#
# Drift velocities:
#   v_ExB   = (E x B) / B^2
#   v_gradB = (mu / q) * (b_hat x grad|B|) / |B|
#   v_curv  = -(m * v_par^2 / q) * (kappa x b_hat) / |B|
#             where  kappa = (b_hat . nabla) b_hat  (curvature vector,
#             pointing toward centre of curvature)
#
# All gradients computed by central finite differences.
#
# Sign conventions (verified for dipole + B_curved_z):
#   - Both v_gradB and v_curv are in the SAME azimuthal direction for
#     the same particle species (both add to produce the ring current).
#   - The NEGATIVE sign in v_curv is essential.
# =============================================================


def _grad_Bmag(B_func, r, t, h=1e-4):
    """
    Numerical gradient of |B| at position r using central differences.
    Returns the 3-vector grad|B|.
    """
    r = np.asarray(r, dtype=float)
    grad = np.zeros(3)
    for i in range(3):
        rp = r.copy(); rp[i] += h
        rm = r.copy(); rm[i] -= h
        grad[i] = (np.linalg.norm(B_func(rp, t)) -
                   np.linalg.norm(B_func(rm, t))) / (2.0 * h)
    return grad


def _curvature(B_func, r, t, h=1e-4):
    """
    Numerical field-line curvature vector kappa = (b_hat . nabla) b_hat
    evaluated at position r, using central differences along b_hat.

    kappa points TOWARD the centre of curvature.
    """
    r  = np.asarray(r, dtype=float)
    B0 = B_func(r, t)
    b0 = B0 / np.linalg.norm(B0)
    rp = r + h * b0
    rm = r - h * b0
    Bp = B_func(rp, t);  bp = Bp / np.linalg.norm(Bp)
    Bm = B_func(rm, t);  bm = Bm / np.linalg.norm(Bm)
    return (bp - bm) / (2.0 * h)


def gc_rhs(t, state, q, m, mu, E_func, B_func, h=1e-4):
    """
    RHS of the guiding-centre equations of motion.

    Parameters
    ----------
    t      : float  — current time
    state  : (4,)   — [X, Y, Z, v_par]
    q, m   : charge and mass
    mu     : magnetic moment (conserved adiabatic invariant)
    E_func : callable  E(r, t) -> (3,)
    B_func : callable  B(r, t) -> (3,)
    h      : finite-difference step for gradient calculations

    Returns
    -------
    dsdt : (4,)
    """
    r     = state[:3]
    v_par = state[3]

    B    = B_func(r, t)
    Bmag = np.linalg.norm(B)
    bhat = B / Bmag

    E = E_func(r, t)

    # ---- Precompute spatial derivatives --------------------------------
    gradB = _grad_Bmag(B_func, r, t, h)
    kappa = _curvature(B_func, r, t, h)

    # ---- Drift velocities ---------------------------------------------
    # E x B drift
    v_ExB = np.cross(E, B) / Bmag**2

    # Grad-B drift:  (mu / q) * (b_hat x grad|B|) / |B|
    v_gradB = (mu / q) * np.cross(bhat, gradB) / Bmag

    # Curvature drift:  -(m v_par^2 / q) * (kappa x b_hat) / |B|
    # NOTE: the negative sign is required for both v_gradB and v_curv
    # to be in the same azimuthal direction (verified analytically).
    v_curv = -(m * v_par**2 / q) * np.cross(kappa, bhat) / Bmag

    v_drift = v_ExB + v_gradB + v_curv

    # ---- Equations of motion ------------------------------------------
    drdt = v_par * bhat + v_drift

    E_par     = np.dot(E, bhat)
    dv_par_dt = (q / m) * E_par - (mu / m) * np.dot(bhat, gradB)

    return np.concatenate([drdt, [dv_par_dt]])


def simulate_gc_orbit(
    state0_gc,
    mu,
    dt,
    nsteps,
    q,
    m,
    E_func,
    B_func,
    method="RK45",
    rtol=1e-9,
    atol=1e-12,
    h_fd=1e-4,
):
    """
    Integrate the guiding-centre equations of motion.

    Parameters
    ----------
    state0_gc : (4,) — initial [X, Y, Z, v_par]
    mu        : float — conserved magnetic moment  m * v_perp^2 / (2|B|)
    dt        : float — output time step
    nsteps    : int   — number of output steps
    q, m      : charge and mass
    E_func, B_func : field callables
    method    : scipy solve_ivp method (default 'RK45')
    rtol, atol: solver tolerances
    h_fd      : finite-difference step for gradient calculations
                (a value ~1e-4 works well for the dipole field at r ~ 3)

    Returns
    -------
    t    : (nsteps,) array of times
    traj : (nsteps, 4) array  — columns [X, Y, Z, v_par]
    """
    t_eval = dt * np.arange(nsteps)
    t_span = (t_eval[0], t_eval[-1])

    sol = solve_ivp(
        fun=lambda t, y: gc_rhs(t, y, q, m, mu, E_func, B_func, h=h_fd),
        t_span=t_span,
        y0=np.array(state0_gc, dtype=float),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"GC integrator failed: {sol.message}")

    return sol.t, sol.y.T
