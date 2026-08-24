#!/usr/bin/env python3
"""
plot_fig1.py — Paper 2, Figure 1
================================

gamma(omega) spectrum and IDOS staircase of the Fibonacci chain, with the
twelve labelled gaps of Table I marked. Inset: gap-labelling residuals,
measured gaps against random controls, on a log axis — the falsifiability
argument of §III A in one picture.

    python plot_fig1.py            # ~2 min: N=60k sites, 6000 frequencies
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transfer_matrix import (fibonacci_masses, lyapunov, idos, find_gaps,
                             gap_label, A_NM, OMEGA_MAX)

N = 60_000
m = fibonacci_masses(N)
w = np.linspace(0.02, 0.995, 6000) * OMEGA_MAX
f = w / (2 * np.pi)
g = lyapunov(m, w)
n = idos(m, w)
# annotate exactly the twelve gaps of Table I (a threshold-based pick can
# swap the weakest entry, e.g. (-1,2) for (-5,9), and figure and table must
# agree gap for gap)
TABLE_I_THZ = [14.386, 17.213, 18.792, 19.693, 21.040, 22.222, 22.859,
               23.760, 24.222, 24.479, 24.826, 25.239]
gaps = [int(np.argmin(np.abs(f - ft))) for ft in TABLE_I_THZ]

fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True,
                       gridspec_kw=dict(height_ratios=[1.1, 1.0], hspace=0.07))

ax[0].semilogy(f, np.maximum(g, 1e-7), lw=0.6, color="#20456b")
for i in gaps:
    ax[0].axvline(f[i], color="#eb6834", lw=0.5, alpha=0.5, zorder=0)
ax[0].set_ylabel(r"$\gamma$ (site$^{-1}$)")
ax[0].set_ylim(1e-5, 3e-2)

ax[1].plot(f, n, lw=0.8, color="#20456b")
# Ladder annotation: the eight upper gaps crowd into IDOS 0.6-0.9, so in-place
# labels overlap. Stack the labels in the empty upper-left region, sorted by
# IDOS so the leader lines never cross.
vals = []
info = []
for i in gaps:
    p, q, err = gap_label(n[i])
    vals.append(n[i])
    info.append((n[i], f[i], p, q))
    ax[1].plot([f[i]], [n[i]], "o", ms=2.2, color="#eb6834", zorder=4)
info.sort()
y_lab = np.linspace(0.26, 0.97, len(info))
x_lab = 6.2
for (ni, fi, p, q), yl in zip(info, y_lab):
    ax[1].annotate(f"({p},{q})", xy=(fi, ni), xytext=(x_lab, yl),
                   fontsize=5.2, color="#eb6834", ha="right", va="center",
                   arrowprops=dict(arrowstyle="-", lw=0.35,
                                   color="#c9a08a", shrinkA=1, shrinkB=2))
ax[1].set_xlabel(r"$f$ (THz)")
ax[1].set_ylabel(r"IDOS $\mathcal{N}$")
ax[1].set_xlim(f[0], f[-1])

# inset: residual histogram, measured vs random controls
rng = np.random.default_rng(0)
ctrl = [gap_label(x)[2] for x in rng.random(3000)]
meas = [gap_label(v)[2] for v in vals]
ia = ax[0].inset_axes([0.13, 0.55, 0.42, 0.40])
bins = np.logspace(-7, -1, 25)
ia.hist(ctrl, bins=bins, color="#bbbbbb", label="random")
ia.hist(meas, bins=bins, color="#eb6834", label="gaps")
ia.set_xscale("log")
ia.set_xlabel(r"$|\mathcal{N}-(p{+}q\tau^{-1})|$", fontsize=5)
ia.tick_params(labelsize=5)
ia.legend(fontsize=4.5, frameon=False)

for a in ax:
    a.tick_params(labelsize=7)
fig.savefig("fig_gamma_idos.pdf", dpi=400, bbox_inches="tight")
print("figure -> fig_gamma_idos.pdf")
