#!/usr/bin/env python3
"""
plot_fig4.py — Paper 2, Figure 4 (SCHEMATIC)
============================================

Two panels: (a) the device — ribbon with Fibonacci isotope letters, Al
transducers, dimensions; (b) an ILLUSTRATIVE cavity-mode spectrum: Fibonacci
against the composition-matched random control, gap region emptied, one
phason edge mode inside the gap. Marked "schematic" because panel (b) is a
mock spectrum built from the calibrated parameters, not measured data.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

TAU = (1 + 5 ** 0.5) / 2

fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.0),
                       gridspec_kw=dict(height_ratios=[0.75, 1.0], hspace=0.42))

# ---- (a) device ----
a0 = ax[0]
a0.set_xlim(0, 10); a0.set_ylim(0, 3); a0.axis("off")
a0.add_patch(Rectangle((1.0, 1.2), 8.0, 0.7, fc="#dfe8f0", ec="k", lw=0.7))
j = np.arange(40)
s = np.floor((j + 2) / TAU) - np.floor((j + 1) / TAU)
x = 1.0
for k in range(32):
    wdt = 8.0 / 32
    if s[k]:
        a0.add_patch(Rectangle((x, 1.2), wdt, 0.7, fc="#8fb3d9", ec="none"))
    x += wdt
for xc in (0.55, 9.05):
    a0.add_patch(Rectangle((xc - 0.25, 1.05), 0.6, 1.0, fc="#c9c9c9",
                           ec="k", lw=0.7))
a0.text(0.85, 2.25, "Al", fontsize=6, ha="center")
a0.text(9.35, 2.25, "Al", fontsize=6, ha="center")
a0.annotate("", xy=(9.0, 0.85), xytext=(1.0, 0.85),
            arrowprops=dict(arrowstyle="<->", lw=0.6))
a0.text(5.0, 0.55, r"$L = 20\ \mu$m", fontsize=6, ha="center")
a0.annotate("", xy=(1.75, 2.15), xytext=(1.0, 2.15),
            arrowprops=dict(arrowstyle="<->", lw=0.6))
a0.text(1.4, 2.35, r"$d=24.8$ nm", fontsize=5.5, ha="center")
a0.text(9.75, 1.55, r"$W=30$ nm", fontsize=6, rotation=90, va="center")
a0.text(5.0, 1.55, r"$^{10}$B / $^{11}$B Fibonacci word", fontsize=6,
        ha="center")
a0.text(0.2, 2.85, "(a)", fontsize=8)

# ---- (b) mock spectrum ----
a1 = ax[1]
fsr = 0.5e-3                                      # THz
f0, gap_w = 0.250, 0.250 * 0.0067
modes = np.arange(0.238, 0.262, fsr)
freq = np.linspace(0.238, 0.262, 3000)
def comb(centers, width=0.12e-3):
    y = np.zeros_like(freq)
    for c in centers:
        y += np.exp(-0.5 * ((freq - c) / width) ** 2)
    return y
in_gap = np.abs(modes - f0) < gap_w / 2
y_rand = comb(modes)
y_fib = comb(modes[~in_gap])
phi_mode = f0 - 0.18 * gap_w
y_fib += 0.75 * np.exp(-0.5 * ((freq - phi_mode) / 0.12e-3) ** 2)
a1.plot(freq * 1e3, y_rand + 1.3, lw=0.7, color="#999999",
        label="random control")
a1.plot(freq * 1e3, y_fib, lw=0.7, color="#20456b", label="Fibonacci")
a1.axvspan((f0 - gap_w / 2) * 1e3, (f0 + gap_w / 2) * 1e3,
           color="#f0e2c8", zorder=0)
a1.annotate(r"$(0,1)$ gap, $1.7$ GHz", xy=(f0 * 1e3, 2.45), fontsize=6,
            ha="center")
a1.annotate(r"edge mode ($\phi$-dependent)",
            xy=(phi_mode * 1e3, 0.8), xytext=(240.5, 1.05), fontsize=5.5,
            arrowprops=dict(arrowstyle="-", lw=0.5))
a1.annotate("", xy=((phi_mode + 0.5 * gap_w) * 1e3, 0.88),
            xytext=(phi_mode * 1e3, 0.88),
            arrowprops=dict(arrowstyle="->", lw=0.7, color="#eb6834"))
a1.set_xlabel("frequency (GHz)", fontsize=7)
a1.set_ylabel("cavity response (arb.)", fontsize=7)
a1.set_yticks([])
a1.legend(fontsize=6, frameon=False, loc="upper left")
a1.tick_params(labelsize=6)
a1.text(0.015, 0.93, "(b)", fontsize=8, transform=a1.transAxes)
a1.text(0.985, 0.04, "SCHEMATIC", fontsize=6, color="#aaaaaa",
        ha="right", transform=a1.transAxes)
fig.savefig("fig_experiment.pdf", dpi=400, bbox_inches="tight")
print("figure -> fig_experiment.pdf")
