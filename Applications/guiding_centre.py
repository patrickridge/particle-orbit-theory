import numpy as np
from scipy.integrate import solve_ivp

# guiding_centre.py — GC equations of motion
# State: [X, Y, Z, v_par],  mu conserved
# dR/dt = v_par*bhat + v_ExB + v_gradB + v_curv
# dv_par/dt = (q/m)*E_par - (mu/m)*(bhat . grad|B|)
# NB: minus sign in v_curv is needed so grad-B and curvature drifts
#     go in the same direction (both contribute to ring current)


def _grad_Bmag(B_func, r, t, h=1e-4):
    """grad|B| by central differences."""
    r = np.asarray(r, dtype=float)
    grad = np.zeros(3)
    for i in range(3):
        rp = r.copy(); rp[i] += h
        rm = r.copy(); rm[i] -= h
        grad[i] = (np.linalg.norm(B_func(rp, t)) -
                   np.linalg.norm(B_func(rm, t))) / (2.0 * h)
    return grad


def _curvature(B_func, r, t, h=1e-4):
    """Curvature vector kappa = (bhat . nabla) bhat, by central differences."""
    r  = np.asarray(r, dtype=float)
    B0 = B_func(r, t)
    b0 = B0 / np.linalg.norm(B0)
    rp = r + h * b0
    rm = r - h * b0
    Bp = B_func(rp, t);  bp = Bp / np.linalg.norm(Bp)
    Bm = B_func(rm, t);  bm = Bm / np.linalg.norm(Bm)
    return (bp - bm) / (2.0 * h)


def gc_rhs(t, state, q, m, mu, E_func, B_func, h=1e-4):
    """RHS for the GC equations."""
    r     = state[:3]
    v_par = state[3]

    B    = B_func(r, t)
    Bmag = np.linalg.norm(B)
    bhat = B / Bmag

    E = E_func(r, t)

    gradB = _grad_Bmag(B_func, r, t, h)
    kappa = _curvature(B_func, r, t, h)

    # drift velocities
    v_ExB   = np.cross(E, B) / Bmag**2
    v_gradB = (mu / q) * np.cross(bhat, gradB) / Bmag
    v_curv  = -(m * v_par**2 / q) * np.cross(kappa, bhat) / Bmag

    v_drift = v_ExB + v_gradB + v_curv

    drdt = v_par * bhat + v_drift

    E_par     = np.dot(E, bhat)
    dv_par_dt = (q / m) * E_par - (mu / m) * np.dot(bhat, gradB)

    return np.concatenate([drdt, [dv_par_dt]])


def simulate_gc_orbit(
    state0_gc, mu, dt, nsteps,
    q, m, E_func, B_func,
    method="RK45", rtol=1e-9, atol=1e-12, h_fd=1e-4,
):
    """Integrate GC equations.  Returns (t, traj) with traj shape (nsteps, 4)."""
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
