import numpy as np

# fields.py — E and B field definitions
# All functions have signature f(r, t) -> np.ndarray(3,)


# E fields

def E_zero(r, t):
    """Zero electric field."""
    return np.zeros(3)


def E_const(vec):
    """
    Constant electric field.
    Usage: E_func = E_const([Ex, Ey, Ez])
    """
    vec = np.array(vec, dtype=float)
    def _E(r, t):
        return vec.copy()
    return _E


def E_corotation(B_func, Omega):
    """Corotation E field: E = -(Omega x r) x B.  ExB drift = v_rot to lowest order."""
    def _E(r, t):
        B    = B_func(r, t)
        vrot = np.array([-Omega * r[1], Omega * r[0], 0.0])
        return -np.cross(vrot, B)
    return _E



# B fields

def B_uniform_z(B0=1.0):
    """Uniform field along z."""
    B0 = float(B0)
    def _B(r, t):
        return np.array([0.0, 0.0, B0])
    return _B


def B_gradx_z(B0=1.0, eps=0.05):
    """Bz = B0*(1 + eps*x).  Has div B = 0.  Keep |eps*x| << 1."""
    B0 = float(B0)
    eps = float(eps)
    def _B(r, t):
        x = float(r[0])
        Bz = B0 * (1.0 + eps * x)
        return np.array([0.0, 0.0, Bz])
    return _B


def B_mirror_z(B0=1.0, alpha=0.1):
    """Toy mirror: Bz = B0*(1 + alpha*z^2).  NB: div B != 0."""
    B0 = float(B0)
    alpha = float(alpha)
    def _B(r, t):
        z = float(r[2])
        Bz = B0 * (1.0 + alpha * z * z)
        return np.array([0.0, 0.0, Bz])
    return _B

def B_mirror_div_free(B0=1.0, alpha=0.5):
    """Divergence-free mirror: Bz = B0(1+alpha*z^2), Bx = -alpha*B0*x*z, etc."""
    def _B(r, t):
        x, y, z = r
        Bz = B0 * (1.0 + alpha * z**2)
        Bx = -alpha * B0 * x * z
        By = -alpha * B0 * y * z
        return np.array([Bx, By, Bz])
    return _B


def B_curved_z(B0=1.0, R_c=10.0):
    """Curved field lines in x-z plane, radius of curvature R_c.  div B = 0 to first order."""
    B0 = float(B0)
    R_c = float(R_c)
    def _B(r, t):
        x, y, z = float(r[0]), float(r[1]), float(r[2])
        Bx = -B0 * z / R_c
        Bz =  B0 * (1.0 - x / R_c)
        return np.array([Bx, 0.0, Bz])
    return _B



# Dipole fields

def B_dipole_cartesian(M=1.0, tilt_deg=0.0, eps=1e-12):
    """Magnetic dipole, moment M tilted by tilt_deg from +z in x-z plane."""
    M     = float(M)
    eps   = float(eps)
    theta = float(tilt_deg) * np.pi / 180.0
    mx    = M * np.sin(theta)
    mz    = M * np.cos(theta)

    def _B(r, t):
        x, y, z = float(r[0]), float(r[1]), float(r[2])
        r2 = x*x + y*y + z*z
        r2 = max(r2, eps)
        r5 = r2 ** 2.5

        m_dot_r = mx * x + mz * z
        Bx = (3.0 * m_dot_r * x - r2 * mx) / r5
        By = (3.0 * m_dot_r * y             ) / r5
        Bz = (3.0 * m_dot_r * z - r2 * mz) / r5
        return np.array([Bx, By, Bz])

    return _B


def B_dipole_rotating(M=1.0, tilt_deg=0.0, Omega=0.0, eps=1e-12):
    """Rotating tilted dipole: m(t) = M*(sin θ cos Ωt, sin θ sin Ωt, cos θ)."""
    M   = float(M)
    eps = float(eps)
    theta = float(tilt_deg) * np.pi / 180.0
    st, ct = np.sin(theta), np.cos(theta)

    def _B(r, t):
        t  = float(t)
        mx = M * st * np.cos(Omega * t)
        my = M * st * np.sin(Omega * t)
        mz = M * ct

        x, y, z = float(r[0]), float(r[1]), float(r[2])
        r2 = max(x*x + y*y + z*z, eps)
        r5 = r2 ** 2.5
        m_dot_r = mx*x + my*y + mz*z
        return np.array([
            (3.0*m_dot_r*x - r2*mx) / r5,
            (3.0*m_dot_r*y - r2*my) / r5,
            (3.0*m_dot_r*z - r2*mz) / r5,
        ])
    return _B


def dipole_B_magnitude_on_axis(M=1.0):
    """|B| on the dipole axis: 2M/|z|^3."""
    M = float(M)
    def _Bmag(z):
        z = np.asarray(z, dtype=float)
        return np.abs(2.0 * M / (z ** 3))
    return _Bmag
    