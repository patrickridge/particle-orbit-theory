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