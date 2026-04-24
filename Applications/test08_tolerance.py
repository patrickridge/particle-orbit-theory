import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from orbit_ivp_core import simulate_orbit_ivp, extract_gc
from fields import E_zero, B_dipole_cartesian

_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
os.makedirs(_FIG, exist_ok=True)
os.makedirs(_RES, exist_ok=True)
sns.set_theme(style="ticks", context="paper")

# Test 08 tolerance check: are the "dips" in the extracted GC near mirror
# points numerical (tolerance-dependent) or physical?

q, m = 1.0, 1.0
M    = 50.0

B_func = B_dipole_cartesian(M=M)
E_func = E_zero

r0     = np.array([3.0, 0.0, 0.0])
B0_vec = B_func(r0, 0.0)
bhat   = B0_vec / np.linalg.norm(B0_vec)

v_mag     = 1.0
pitch_deg = 60.0
pitch     = np.deg2rad(pitch_deg)
eperp     = np.array([0.0, 1.0, 0.0])
v0        = v_mag * (np.cos(pitch) * bhat + np.sin(pitch) * eperp)
state0    = np.concatenate((r0, v0))

v_par_mag = v_mag * np.cos(pitch)
T_b_est   = 4.0 * r0[0] / v_par_mag
T         = 1.3 * T_b_est             # one bounce + a bit (covers both mirror points)
dt        = 0.0001
nsteps    = int(T / dt)

print(f"T_b_est = {T_b_est:.2f}, T_run = {T:.1f}, nsteps = {nsteps}")

def run(rtol, atol, label):
    t0 = time.perf_counter()
    t, traj = simulate_orbit_ivp(
        state0=state0, dt=dt, nsteps=nsteps,
        q=q, m=m, E_func=E_func, B_func=B_func,
        rtol=rtol, atol=atol,
    )
    wall = time.perf_counter() - t0
    print(f"{label}: rtol={rtol:.0e}, atol={atol:.0e}  in {wall:.1f} s")
    r_gc = extract_gc(traj, t, B_func, q=q, m=m)
    return t, traj, r_gc

t_L, traj_L, gc_L = run(1e-9,  1e-12, "loose")
t_T, traj_T, gc_T = run(1e-11, 1e-14, "tight")

z_L      = traj_L[:, 2];   z_T      = traj_T[:, 2]
zGC_L    = gc_L[:, 2];     zGC_T    = gc_T[:, 2]

# mirror points from v_par sign changes (loose run)
vpar_L = np.zeros_like(t_L)
for i in range(len(t_L)):
    Bi = B_func(traj_L[i, :3], t_L[i])
    vpar_L[i] = np.dot(traj_L[i, 3:], Bi / np.linalg.norm(Bi))
sign_ch = np.where(np.diff(np.sign(vpar_L)))[0]
mirror_t = t_L[sign_ch] if len(sign_ch) else np.array([])

# peak-to-peak of GC z in a small window around the first mirror (first +z peak)
# pick the first mirror point that is in the upper hemisphere
if len(sign_ch):
    # find first peak (first mirror where zGC > 0)
    upper = [i for i in sign_ch if zGC_L[i] > 0]
    first_mirror_idx = upper[0] if upper else sign_ch[0]
    t_mirror_peak = t_L[first_mirror_idx]
else:
    # fallback: use global z max
    first_mirror_idx = int(np.argmax(zGC_L))
    t_mirror_peak = t_L[first_mirror_idx]

# window ±0.5 time units around the peak for dip measurement
win = 0.5
def local_p2p(t, z, t0, w=win):
    m = (t >= t0 - w) & (t <= t0 + w)
    if not np.any(m):
        return np.nan
    return np.max(z[m]) - np.min(z[m])

p2p_L = local_p2p(t_L, zGC_L, t_mirror_peak)
p2p_T = local_p2p(t_T, zGC_T, t_mirror_peak)

# max difference between loose and tight z over full integration
n = min(len(t_L), len(t_T))
max_dz = np.max(np.abs(z_L[:n] - z_T[:n]))
max_dzGC = np.max(np.abs(zGC_L[:n] - zGC_T[:n]))

print()
print(f"First mirror point in upper hemisphere at t = {t_mirror_peak:.3f}")
print(f"Dip amplitude (peak-to-peak of z_GC in +/- {win} window):")
print(f"  loose  (rtol=1e-9,  atol=1e-12):  {p2p_L:.3e}")
print(f"  tight  (rtol=1e-11, atol=1e-14):  {p2p_T:.3e}")
print(f"  ratio tight/loose:                {p2p_T/p2p_L:.3f}")
print()
print(f"Max |z_loose - z_tight| over full run:    {max_dz:.3e}")
print(f"Max |zGC_loose - zGC_tight| over full:    {max_dzGC:.3e}")

# verdict
shrink = 1.0 - p2p_T / p2p_L
if shrink > 0.5:
    verdict = "DIPS ARE NUMERICAL"
    reason = f"amplitude shrinks by {100*shrink:.0f}% under tighter tolerance"
elif abs(p2p_T / p2p_L - 1.0) < 10.0:
    verdict = "DIPS ARE PHYSICAL"
    reason = f"amplitude changes by only {100*(p2p_T/p2p_L - 1):+.0f}% despite 100x tighter tolerance"
else:
    verdict = "INCONCLUSIVE"
    reason = f"ratio {p2p_T/p2p_L:.2f}"
print(f"\nVERDICT: {verdict} ({reason})")

# figure
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for ax, (tt, zz, gg, lbl) in zip(
    axes,
    [(t_L, z_L, zGC_L, f"Loose: rtol=1e-9, atol=1e-12"),
     (t_T, z_T, zGC_T, f"Tight: rtol=1e-11, atol=1e-14")],
):
    ax.plot(tt, zz, lw=0.5, alpha=0.4, color="C0", label="Full orbit z(t)")
    ax.plot(tt, gg, lw=1.2, color="C1", label="Extracted GC z(t)")
    if len(sign_ch):
        ax.scatter(mirror_t, zGC_L[sign_ch], s=25, color="green",
                   zorder=5, label="Mirror points")
    ax.set_ylabel("z")
    ax.set_title(lbl, fontsize=10)
    ax.legend(frameon=True, fontsize=8, loc="lower left")

axes[-1].set_xlabel("t")
fig.suptitle(r"Test 08 — tolerance check on the GC `dips' near mirror points",
             fontsize=12)

# inset zooms
for ax, tt, gg, zz in zip(axes, [t_L, t_T], [zGC_L, zGC_T], [z_L, z_T]):
    axins = ax.inset_axes([0.58, 0.10, 0.38, 0.55])
    mask = (tt >= t_mirror_peak - win) & (tt <= t_mirror_peak + win)
    axins.plot(tt[mask], zz[mask], lw=0.5, alpha=0.5, color="C0")
    axins.plot(tt[mask], gg[mask], lw=1.5, color="C1")
    axins.axvline(t_mirror_peak, color="green", lw=0.6, alpha=0.5)
    axins.set_title(fr"Zoom: mirror at $t={t_mirror_peak:.1f}$", fontsize=8)
    axins.tick_params(labelsize=7)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(_FIG, "fig_5_2_tolerance.png"), dpi=200)
plt.show()

# CSV
# interpolate tight to loose time grid (shared grid guaranteed since same dt/nsteps)
csv_path = os.path.join(_RES, "test08_tolerance_comparison.csv")
np.savetxt(
    csv_path,
    np.column_stack([t_L, z_L, z_T, zGC_L, zGC_T]),
    header="t,z_loose,z_tight,zGC_loose,zGC_tight",
    comments="", delimiter=",",
)
print(f"CSV written: {csv_path}")
