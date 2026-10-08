r"""
Morse potential + Gaussian perturbation: how an extra force shifts the
equilibrium internuclear distance r0.

    U(r) = D * [(1 - exp(-a (r - r0)))^2 - 1]  +  A * exp(-((r - rc) / w)^2)
           \_________ Morse (fixed) _________/     \___ Gaussian (sliders) ___/

    A  - strength of the extra interaction (A > 0 repulsive bump, A < 0 attractive well)
    rc - distance where the extra interaction acts most strongly
    w  - range (width) of the extra interaction

Left panel : U(r), the bare Morse curve and the Gaussian term, with the new minimum.
Right panels: where the minimum lies as a function of A (at current rc, w)
              and as a function of w (at current A, rc).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ---- Fixed Morse parameters (roughly H2) ----
D = 4.75    # well depth [eV]
a = 1.94    # stiffness [1/Å]
R0 = 0.74   # equilibrium distance without perturbation [Å]

# ---- Initial Gaussian parameters ----
A0, RC0, W0 = 1.0, 0.9, 0.3

r = np.linspace(0.3, 4.0, 4000)


def morse(r):
    return D * ((1 - np.exp(-a * (r - R0))) ** 2 - 1)


def gauss(r, A, rc, w):
    return A * np.exp(-(((r - rc) / w) ** 2))


def total(r, A, rc, w):
    return morse(r) + gauss(r, A, rc, w)


def find_minimum(U):
    """Global minimum on the grid along the last axis, refined with a parabola fit."""
    i = np.argmin(U, axis=-1)
    i = np.clip(i, 1, len(r) - 2)
    idx = np.arange(U.shape[0]) if U.ndim == 2 else None
    if U.ndim == 2:
        y0, y1, y2 = U[idx, i - 1], U[idx, i], U[idx, i + 1]
    else:
        y0, y1, y2 = U[i - 1], U[i], U[i + 1]
    denom = y0 - 2 * y1 + y2
    shift = np.where(denom != 0, 0.5 * (y0 - y2) / np.where(denom == 0, 1, denom), 0)
    dr = r[1] - r[0]
    r_min = r[i] + shift * dr
    U_min = y1 - 0.25 * (y0 - y2) * shift
    return r_min, U_min


# ---- Figure layout ----
fig = plt.figure(figsize=(13, 7.5))
fig.canvas.manager.set_window_title("Morse + Gaussian potential")
ax = fig.add_axes([0.06, 0.32, 0.55, 0.62])
ax_A = fig.add_axes([0.70, 0.66, 0.27, 0.28])
ax_w = fig.add_axes([0.70, 0.32, 0.27, 0.25])

# Main plot
(l_morse,) = ax.plot(r, morse(r), "--", color="gray", lw=1.5, label="Morse only")
(l_gauss,) = ax.plot(r, gauss(r, A0, RC0, W0), ":", color="tab:orange", lw=1.5,
                     label="Gaussian term")
(l_total,) = ax.plot(r, total(r, A0, RC0, W0), color="tab:blue", lw=2.2,
                     label="Total U(r)")
ax.axvline(R0, color="gray", lw=0.8, ls="--")
ax.plot([R0], [-D], "o", color="gray", ms=7)
(p_min,) = ax.plot([], [], "o", color="tab:red", ms=9, label="New minimum r0'")
v_min = ax.axvline(R0, color="tab:red", lw=1)
v_rc = ax.axvline(RC0, color="tab:orange", lw=0.8, alpha=0.6)
info = ax.text(0.98, 0.04, "", transform=ax.transAxes, ha="right", va="bottom",
               fontsize=11, family="monospace",
               bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax.set_xlim(r[0], r[-1])
ax.set_ylim(-D - 2, 4)
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("Internuclear distance r [Å]")
ax.set_ylabel("Potential energy U [eV]")
ax.set_title("U(r) = D[(1 − e^(−a(r−r0)))² − 1] + A·exp[−((r−rc)/w)²]")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

# Sweep plots
A_sweep = np.linspace(-4, 4, 241)
w_sweep = np.linspace(0.05, 1.5, 241)
(l_rA,) = ax_A.plot([], [], color="tab:purple")
(p_rA,) = ax_A.plot([], [], "o", color="tab:red")
ax_A.axhline(R0, color="gray", ls="--", lw=0.8)
ax_A.set_xlim(A_sweep[0], A_sweep[-1])
ax_A.set_xlabel("Strength A [eV]")
ax_A.set_ylabel("r0' [Å]")
ax_A.set_title("Minimum vs strength (rc, w fixed)", fontsize=10)
ax_A.grid(alpha=0.3)

(l_rw,) = ax_w.plot([], [], color="tab:green")
(p_rw,) = ax_w.plot([], [], "o", color="tab:red")
ax_w.axhline(R0, color="gray", ls="--", lw=0.8)
ax_w.set_xlim(w_sweep[0], w_sweep[-1])
ax_w.set_xlabel("Range w [Å]")
ax_w.set_ylabel("r0' [Å]")
ax_w.set_title("Minimum vs range (A, rc fixed)", fontsize=10)
ax_w.grid(alpha=0.3)

# Sliders (only the Gaussian part is adjustable)
s_A = Slider(fig.add_axes([0.12, 0.18, 0.45, 0.03]), "A  strength [eV]",
             A_sweep[0], A_sweep[-1], valinit=A0)
s_rc = Slider(fig.add_axes([0.12, 0.13, 0.45, 0.03]), "rc  centre [Å]",
              0.3, 3.0, valinit=RC0)
s_w = Slider(fig.add_axes([0.12, 0.08, 0.45, 0.03]), "w  range [Å]",
             w_sweep[0], w_sweep[-1], valinit=W0)
b_reset = Button(fig.add_axes([0.12, 0.02, 0.08, 0.04]), "Reset")
fig.text(0.70, 0.13,
         f"Fixed Morse parameters:\n  D  = {D} eV\n  a  = {a} 1/Å\n  r0 = {R0} Å",
         fontsize=10, family="monospace", va="center")


def update(_=None):
    A, rc, w = s_A.val, s_rc.val, s_w.val

    U = total(r, A, rc, w)
    l_gauss.set_ydata(gauss(r, A, rc, w))
    l_total.set_ydata(U)
    v_rc.set_xdata([rc, rc])

    r_min, U_min = find_minimum(U)
    p_min.set_data([r_min], [U_min])
    v_min.set_xdata([r_min, r_min])
    info.set_text(f"r0'  = {r_min:6.3f} Å\n"
                  f"Δr0  = {r_min - R0:+6.3f} Å\n"
                  f"U(r0') = {U_min:6.3f} eV\n"
                  f"ΔU   = {U_min + D:+6.3f} eV")

    # r0' as a function of A (rc, w fixed)
    UA = morse(r)[None, :] + gauss(r[None, :], A_sweep[:, None], rc, w)
    rA, _ = find_minimum(UA)
    l_rA.set_data(A_sweep, rA)
    p_rA.set_data([A], [r_min])
    ax_A.set_ylim(min(rA.min(), R0) - 0.1, max(rA.max(), R0) + 0.1)

    # r0' as a function of w (A, rc fixed)
    Uw = morse(r)[None, :] + gauss(r[None, :], A, rc, w_sweep[:, None])
    rw, _ = find_minimum(Uw)
    l_rw.set_data(w_sweep, rw)
    p_rw.set_data([w], [r_min])
    ax_w.set_ylim(min(rw.min(), R0) - 0.1, max(rw.max(), R0) + 0.1)

    fig.canvas.draw_idle()


for s in (s_A, s_rc, s_w):
    s.on_changed(update)
b_reset.on_clicked(lambda _: [s.reset() for s in (s_A, s_rc, s_w)])

update()
plt.show()
