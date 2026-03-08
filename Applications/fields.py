import numpy as np

# =============================================================
# fields.py
# Canonical field definitions for particle orbit simulations.
# ALL test scripts should import from here — never redefine
# field functions inline in test files.
#
# Conventions
# -----------
# Every field function has signature  f(r, t) -> np.ndarray(3,)
# Factory functions (B_uniform_z, B_gradx_z, …) return such a
# function after capturing their parameters in a closure.
# =============================================================


# -------------------------------------------------------------
# E fields
# -------------------------------------------------------------

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


# -------------------------------------------------------------
# B fields — uniform / simple
# -------------------------------------------------------------

def B_uniform_z(B0=1.0):
    """
    Uniform field along z-hat: B = (0, 0, B0).
    Usage: B_func = B_uniform_z(B0=1.0)
    """
    B0 = float(B0)
    def _B(r, t):
        return np.array([0.0, 0.0, B0])
    return _B


def B_gradx_z(B0=1.0, eps=0.05):
    """
    Linearly-varying field: Bz(x) = B0 * (1 + eps * x), pointing z-hat.

    NOTE: This field has ∇·B = 0 only approximately (it strictly satisfies
    ∂Bz/∂z = 0 and ∂Bx/∂x = ∂By/∂y = 0, so ∇·B = 0 exactly for this
    z-only form).  Keep |eps * x_max| << 1 so B does not approach zero.

    Analytic grad-B drift (perpendicular to both B and ∇B):
        v_gradB = (m v_perp^2) / (2 q B^2)  *  (B × ∇B) / B
    For this field  ∇B = eps * B0 * x-hat,  B = Bz(x) z-hat, so
        v_gradB ≈ (m v_perp^2 * eps) / (2 q B0)  in the  -y  direction
                  (for positive charge, negative eps gradient sense).
    """
    B0 = float(B0)
    eps = float(eps)
    def _B(r, t):
        x = float(r[0])
        Bz = B0 * (1.0 + eps * x)
        return np.array([0.0, 0.0, Bz])
    return _B


def B_mirror_z(B0=1.0, alpha=0.1):
    """
    Toy mirror field: Bz(z) = B0 * (1 + alpha * z^2), pointing z-hat.

    WARNING: ∇·B ≠ 0 for this field — it is a convenient toy model used
    to demonstrate bounce motion but does NOT satisfy Maxwell's equations.
    Use B_dipole_cartesian for a physically self-consistent mirror geometry.

    Mirror (turning) point condition (adiabatic):
        sin^2(alpha_eq) / sin^2(alpha_mirror) = B_eq / B_mirror
    """
    B0 = float(B0)
    alpha = float(alpha)
    def _B(r, t):
        z = float(r[2])
        Bz = B0 * (1.0 + alpha * z * z)
        return np.array([0.0, 0.0, Bz])
    return _B

def B_mirror_div_free(B0=1.0, alpha=0.5):
    """
    Simple divergence-free 'mirror-like' field (local model).

    Bz(z) = B0 (1 + alpha z^2)
    Bx    = -alpha B0 x z
    By    = -alpha B0 y z

    This satisfies div B = 0 and introduces transverse components so that
    full-orbit Lorentz dynamics can produce mirror-like behaviour.
    """
    def _B(r, t):
        x, y, z = r
        Bz = B0 * (1.0 + alpha * z**2)
        Bx = -alpha * B0 * x * z
        By = -alpha * B0 * y * z
        return np.array([Bx, By, Bz])
    return _B


# -------------------------------------------------------------
# B fields — curvature (for test 09)
# -------------------------------------------------------------

def B_curved_z(B0=1.0, R_c=10.0):
    """
    Simple curved-field-line geometry: field lines curve in the x-z plane
    with radius of curvature R_c.  In the local region near the origin:

        Bx = -B0 * z / R_c
        Bz =  B0 * (1 - x / R_c)          (to first order in 1/R_c)

    This satisfies ∇·B = 0 to first order.

    Analytic curvature drift:
        v_curv = (m v_par^2) / (q B^2)  *  (R_c × B) / R_c^2
    For this geometry (curvature in -x direction for field along +z):
        v_curv ≈  (m v_par^2) / (q B0 R_c)  in the y direction.

    Keep |x / R_c| << 1 and |z / R_c| << 1 for the approximation to hold.
    """
    B0 = float(B0)
    R_c = float(R_c)
    def _B(r, t):
        x, y, z = float(r[0]), float(r[1]), float(r[2])
        Bx = -B0 * z / R_c
        Bz =  B0 * (1.0 - x / R_c)
        return np.array([Bx, 0.0, Bz])
    return _B


# -------------------------------------------------------------
# B fields — magnetic dipole (Cartesian, static)
# -------------------------------------------------------------

def B_dipole_cartesian(M=1.0, eps=1e-12):
    """
    Magnetic dipole field with dipole moment along +z-hat:

        B(r) = (M / r^5) * [3zx,  3zy,  3z^2 - r^2]

    This satisfies ∇·B = 0 and ∇×B = 0 everywhere except the origin.
    The eps guard prevents division by zero near r = 0.

    On-axis (x = y = 0):  |B| = 2M / z^3   (see dipole_B_magnitude_on_axis)
    """
    M = float(M)
    eps = float(eps)

    def _B(r, t):
        x, y, z = float(r[0]), float(r[1]), float(r[2])
        r2 = x*x + y*y + z*z
        r2 = max(r2, eps)
        r1 = np.sqrt(r2)
        r5 = r1 ** 5

        Bx = M * (3.0 * x * z) / r5
        By = M * (3.0 * y * z) / r5
        Bz = M * (3.0 * z * z - r2) / r5
        return np.array([Bx, By, Bz])

    return _B


def dipole_B_magnitude_on_axis(M=1.0):
    """
    Analytic |B| on the dipole axis (x = y = 0):
        |B(z)| = 2M / |z|^3
    Accepts scalar or numpy array z.
    """
    M = float(M)
    def _Bmag(z):
        z = np.asarray(z, dtype=float)
        return np.abs(2.0 * M / (z ** 3))
    return _Bmag
    