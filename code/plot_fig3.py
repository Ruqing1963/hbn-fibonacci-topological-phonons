#!/usr/bin/env python3
"""
plot_fig3.py — Paper 2, Figure 3
================================

Topological survival phase diagram. Entirely analytic — the payoff of §II F:

    xi_edge = 2 W^2 / [q^2 S_w(2q)],  S_w = Dr^2 sqrt(2pi) Lc exp(-2 q^2 Lc^2)
    xi_gap  = C_xi * lambda,          C_xi = 54  (measured, (0,1) gap)
    margin  M(f, Dr) = xi_edge / (3 xi_gap),  W = 0.8 * v/(2f)  [widest single-mode]

Contours M = 1 (survival boundary), 3, 10, with the Ziman-based boundary
overlaid to show what the ray formula would have forbidden, and the working
point marked.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V, Lc, CXI = 20.0, 5.0, 54.0

f = np.linspace(0.05, 0.45, 400)                 # THz
Dr = np.linspace(0.05, 2.0, 400)                 # nm
F, D = np.meshgrid(f, Dr)
q = 2 * np.pi * F / V
W = 0.8 * V / (2 * F)                            # single-mode-saturated width
Sw = D ** 2 * np.sqrt(2 * np.pi) * Lc * np.exp(-2 * q ** 2 * Lc ** 2)
xi_edge = 2 * W ** 2 / (q ** 2 * Sw)
xi_gap = CXI * V / F
M = xi_edge / (3 * xi_gap)

xi_ziman = W / (2 * q ** 2 * D ** 2) * 2         # xi = 2 * mfp
Mz = xi_ziman / (3 * xi_gap)

fig, ax = plt.subplots(figsize=(3.5, 2.9))
cs = ax.contourf(F, D, np.log10(M), levels=np.linspace(-2, 3, 21),
                 cmap="RdYlBu")
c1 = ax.contour(F, D, M, levels=[1, 3, 10], colors="k",
                linewidths=[1.6, 0.9, 0.9], linestyles=["-", "--", ":"])
ax.clabel(c1, fmt={1: "M=1", 3: "3", 10: "10"}, fontsize=6)
cz = ax.contour(F, D, Mz, levels=[1], colors="#666666", linewidths=1.2,
                linestyles="-.")
ax.annotate("Ziman M=1\n(ray limit)", xy=(0.30, 0.16), fontsize=6,
            color="#555555")
ax.plot([0.25], [0.6], "*", ms=11, color="k", mec="w", mew=0.5)
ax.annotate("working point\n(21x)", xy=(0.25, 0.6), xytext=(0.29, 0.95),
            fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.6))
ax.plot([0.25], [1.0], "o", ms=4, color="k", mec="w", mew=0.5)
ax.annotate("EBL typical (7.4x)", xy=(0.25, 1.0), xytext=(0.06, 1.55),
            fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.6))
ax.set_xlabel(r"$f$ (THz)")
ax.set_ylabel(r"edge roughness $\Delta_r$ (nm rms)")
cb = fig.colorbar(cs, ax=ax, pad=0.02)
cb.set_label(r"$\log_{10}\,[\xi_{\rm dis}/3\xi_{\rm gap}]$", fontsize=7)
cb.ax.tick_params(labelsize=6)
ax.tick_params(labelsize=7)
fig.savefig("fig_survival.pdf", dpi=400, bbox_inches="tight")
print("figure -> fig_survival.pdf")
