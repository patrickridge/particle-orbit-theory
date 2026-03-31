"""
animate_timescales.py
=====================
Visual comparison of the four timescales in a dipole magnetosphere,
using the same parameters as animate08_bounce.py (M=50, r0=3, pitch=60°).

Shows T_gyro, T_bounce, T_drift (azimuthal grad+curv) and T_Neptune on a
shared logarithmic axis.  Bars fill in one by one so each scale can be
introduced in sequence during a talk.

Physical note
-------------
The code uses dimensionless units where q = m = B0 = 1.  To convert to
Neptune-like physical units, one would need to fix a length scale (e.g.,
1 code-length = R_Neptune) and a field scale.  Here we instead show the
ratio T_drift / T_Neptune as a dimensionless number:

    T_Neptune (physical) ≈ 58 000 s  (ω ≈ 1.08 × 10⁻⁴ rad s⁻¹, 16.11 hr)

The ratio tells you how many Neptune rotations fit inside one drift period
— a physically meaningful comparison regardless of unit system.

Saves: ../Figures/animate_timescales.gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Reproduce animate08 parameters --------------------------------
q, m, M = 1.0, 1.0, 50.0

# B at equatorial r0 (dipole on z-axis): |B| = M / r^3
r0_mag    = 3.0
B_eq      = M / r0_mag**3            # ≈ 1.85 code units

Omega_gyro = abs(q) * B_eq / m
T_gyro     = 2.0 * np.pi / Omega_gyro

v_mag, pitch = 1.0, np.deg2rad(60.0)
v_par_mag    = v_mag * np.cos(pitch)
v_perp_mag   = v_mag * np.sin(pitch)
T_bounce_est = 4.0 * r0_mag / v_par_mag   # rough estimate

# Equatorial grad+curv drift (Baumjohann & Treumann §2.3)
sin_p        = np.sin(pitch)
Omega_drift  = 3.0 * v_mag**2 * (1.0 + sin_p**2 / 2.0) / (2.0 * Omega_gyro * r0_mag**2)
T_drift      = 2.0 * np.pi / Omega_drift

# Neptune rotation period (physical: comment-only; ratio is unit-free)
T_Neptune_s  = 58_000.0   # seconds  (16.11 hr, ω ≈ 1.08 × 10⁻⁴ rad s⁻¹)
ratio_drift  = T_drift / T_Neptune_s   # dimensionless

print(f"T_gyro     = {T_gyro:.3f}  code units")
print(f"T_bounce   ≈ {T_bounce_est:.2f}  code units")
print(f"T_drift    ≈ {T_drift:.1f}  code units")
print(f"T_Neptune  = {T_Neptune_s:.0f}  s  (physical)")
print(f"T_drift / T_Neptune  ≈ {ratio_drift:.3e}  (dimensionless ratio)")

# ---- Build figure -----------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F7F8FC")

# Log-scale x axis from just below T_gyro to just above T_drift
x_lo = T_gyro * 0.3
x_hi = T_drift * 3.5

ax.set_xscale("log")
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(-0.5, 3.5)
ax.set_yticks([])
ax.set_xlabel("Time  (code units)", fontsize=12)
ax.set_title(
    "Separation of timescales — dipole magnetosphere\n"
    r"$(M=50,\; r_0=3,\; \mathrm{pitch}=60°)$",
    fontsize=12)

# ---- Define the four bars ----------------------------------------
bars_info = [
    # (y, colour, label, value, value_str, note)
    (2.8, "#E63946", "Gyration",
     T_gyro,   f"T_gyro = {T_gyro:.2f}",
     "fastest — averages out in GC approx."),
    (1.9, "#2A9D8F", "Bounce",
     T_bounce_est, f"T_bounce ≈ {T_bounce_est:.1f}",
     "particle trapped between mirror points"),
    (1.0, "#457B9D", "Azimuthal drift",
     T_drift,  f"T_drift ≈ {T_drift:.0f}",
     "grad-B + curvature drift around planet"),
    (0.1, "#F4A261", "Neptune rotation",
     T_Neptune_s, f"T_Neptune = {T_Neptune_s:.0f} s",
     f"T_drift / T_Neptune ≈ {ratio_drift:.2e}"),
]

HEIGHT = 0.55   # bar height in y units

# Create artists (all starting at width zero)
bar_patches   = []
label_texts   = []
value_texts   = []
note_texts    = []

for y, colour, label, value, value_str, note in bars_info:
    patch = mpatches.FancyArrowPatch(
        (x_lo, y), (x_lo, y),  # will be replaced by Rectangle
        visible=False
    )
    # Use a Rectangle directly
    rect = plt.Rectangle((x_lo, y - HEIGHT/2), 0, HEIGHT,
                          facecolor=colour, alpha=0.85, zorder=3)
    ax.add_patch(rect)
    bar_patches.append((rect, value, colour))

    # Label on the left
    lt = ax.text(x_lo * 0.95, y, label, ha="right", va="center",
                 fontsize=11, fontweight="bold", color=colour)
    label_texts.append(lt)

    # Value text (appears when bar reaches full width)
    vt = ax.text(value * 1.05, y, value_str, ha="left", va="center",
                 fontsize=10, color=colour, alpha=0)
    value_texts.append(vt)

    # Note text below bar
    nt = ax.text(np.sqrt(x_lo * value), y - HEIGHT/2 - 0.12,
                 note, ha="center", va="top",
                 fontsize=8.5, color="#444444", style="italic", alpha=0)
    note_texts.append(nt)

# Vertical reference lines at each timescale
for _, _, _, value, _, _ in bars_info:
    ax.axvline(value, color="#CCCCCC", lw=0.8, ls="--", zorder=1)

# ---- Animation: bars fill in one by one --------------------------
# Each bar gets ~40 frames to fill; then value and note text fade in.
FRAMES_PER_BAR = 40
FADE_FRAMES    = 15
N_BARS         = len(bars_info)
N_FRAMES       = N_BARS * (FRAMES_PER_BAR + FADE_FRAMES) + 20

def bar_width_at_frame(bar_idx, frame):
    """Return current bar right edge given the overall frame number."""
    start  = bar_idx * (FRAMES_PER_BAR + FADE_FRAMES)
    if frame < start:
        return x_lo
    t_in = min(frame - start, FRAMES_PER_BAR)
    frac = t_in / FRAMES_PER_BAR
    # ease-in-out
    frac = frac * frac * (3 - 2 * frac)
    _, value, _ = bar_patches[bar_idx]
    return x_lo * (value / x_lo) ** frac   # log-linear interpolation

def fade_alpha(bar_idx, frame):
    start  = bar_idx * (FRAMES_PER_BAR + FADE_FRAMES) + FRAMES_PER_BAR
    if frame < start:
        return 0.0
    return min(1.0, (frame - start) / FADE_FRAMES)

def update(frame):
    artists = []
    for i, (rect, value, colour) in enumerate(bar_patches):
        right = bar_width_at_frame(i, frame)
        rect.set_width(right - x_lo)
        rect.set_visible(right > x_lo)
        artists.append(rect)

        alpha = fade_alpha(i, frame)
        value_texts[i].set_alpha(alpha)
        note_texts[i].set_alpha(alpha)
        artists += [value_texts[i], note_texts[i]]

    return artists

anim = animation.FuncAnimation(fig, update, frames=N_FRAMES,
                                interval=60, blit=False)

fig.tight_layout()

out = os.path.join(_DIR, "..", "Figures", "animate_timescales.gif")
print(f"Saving {out} ...")
anim.save(out, writer="pillow", fps=18, dpi=120)
print("Done.")
plt.show()
