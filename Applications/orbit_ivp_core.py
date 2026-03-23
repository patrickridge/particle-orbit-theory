import numpy as np
from scipy.integrate import solve_ivp

# Physical parameters (defaults)
q = 1.0
m = 1.0
B0 = 1.0

# Simple fields (for testing)
def E_zero(r, t):
    return np.array([0.0, 0.0, 0.0])

def B_uniform_z(r, t, B0=B0):
    return np.array([0.0, 0.0, B0])

# Lorentz RHS for solve_ivp
def lorentz_rhs(t, state, q, m, E_func, B_func):
    r = state[:3]
    v = state[3:]
    E = E_func(r, t)
    B = B_func(r, t)
    drdt = v
    dvdt = (q / m) * (E + np.cross(v, B))
    return np.concatenate((drdt, dvdt))

# Main simulator using solve_ivp
def simulate_orbit_ivp(
    state0,
    dt,
    nsteps,
    q=q,
    m=m,
    E_func=E_zero,
    B_func=B_uniform_z,
    method="RK45",
    rtol=1e-9,
    atol=1e-12,
):
    # sample times you want output at
    t_eval = dt * np.arange(nsteps)
    t_span = (t_eval[0], t_eval[-1])

    sol = solve_ivp(
        fun=lambda t, y: lorentz_rhs(t, y, q, m, E_func, B_func),
        t_span=t_span,
        y0=state0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    # return (t, traj) with traj shape (nsteps, 6)
    return sol.t, sol.y.T


def extract_gc(traj, t, B_func, q=1.0, m=1.0):
    """
    Extract true guiding-centre positions from a full Lorentz orbit by
    subtracting the Larmor radius vector analytically at every timestep:

        R_gc = r + (m / q·B²) · (v × B)

    This removes the gyration exactly regardless of phase, eliminating
    the decimation artefact (wiggles) seen when sampling every N-th point.
    Decimate the returned array afterwards for 3D plots if needed.

    Parameters
    ----------
    traj : (N, 6) array — output of simulate_orbit_ivp
    t    : (N,) time array
    B_func : callable B(r, t)
    q, m : charge and mass (default 1.0)

    Returns
    -------
    r_gc : (N, 3) array of guiding-centre positions
    """
    r_gc = np.empty((len(t), 3))
    for i in range(len(t)):
        r  = traj[i, :3]
        v  = traj[i, 3:]
        B  = B_func(r, float(t[i]))
        B2 = float(np.dot(B, B))
        r_gc[i] = r + (m / (q * B2)) * np.cross(v, B)
    return r_gc