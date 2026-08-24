#!/usr/bin/env python3
"""
plot_phason.py — Paper 2, Figure 2
==================================

Frequency against phason angle over one full cycle, with the edge-bound branch
picked out by edge weight.

Design decisions worth stating, because they change what the figure claims:

1. Bulk states are drawn as a grey point cloud, not as connected lines. They
   are not tracked branch by branch, and joining them would assert a
   continuity that has not been verified.
2. The edge branch IS drawn as a connected line, because it was continued by
   eigenvector overlap (`phason_tracking.track`) rather than by frequency
   ordering. Line width and colour encode edge weight.
3. Left- and right-bound branches use different colours and are never merged.
   The absence of a left-bound partner is a result of this work (§III C), so
   the figure must be able to display one if it existed.
4. The gap band is shaded so a reader can see that the branch sweeps it rather
   than merely lying inside it.

Usage
-----
    python plot_phason.py                       # compute and plot
    python plot_phason.py --cache fig2.npz      # reuse a previous computation
    python plot_phason.py --n-sites 987 --n-phi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import phason_tracking as pt


def compute(n_sites: int, n_phi: int, f_centre: float, rel_width: float,
            pad: float = 6.0, frac: float = 0.06):
    """Bulk cloud plus overlap-tracked branches in a window around the gap."""
    fc = 2 * np.pi * f_centre
    half = 0.5 * rel_width * fc
    lo, hi = fc - pad * half, fc + pad * half

    phis = np.linspace(0.0, 1.0, n_phi, endpoint=False)
    cloud_phi, cloud_f, cloud_w = [], [], []
    for phi in phis:
        f, vec = pt.open_chain(pt.fibonacci_masses(n_sites, 1, phi))
        sel = (f > lo) & (f < hi)
        wl, wr = pt.edge_weights(vec[:, sel], frac)
        cloud_phi.append(np.full(sel.sum(), phi))
        cloud_f.append(f[sel])
        cloud_w.append(np.maximum(wl, wr))

    branches, gap = pt.track(n_sites, fc - half, fc + half, n_phi, pad=pad,
                             frac=frac)
    return dict(cloud_phi=np.concatenate(cloud_phi),
                cloud_f=np.concatenate(cloud_f),
                cloud_w=np.concatenate(cloud_w),
                branches=branches, gap=gap, f_centre=f_centre,
                window=(lo, hi), n_sites=n_sites, n_phi=n_phi)


def plot(d, out: Path, edge_cut: float = 0.7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    tp = 1.0 / (2 * np.pi)                      # rad/ps -> THz
    f_lo, f_hi = d["gap"]

    fig, ax = plt.subplots(figsize=(3.5, 2.9))

    ax.axhspan(f_lo * tp, f_hi * tp, color="#f0e2c8", zorder=0, lw=0)
    ax.text(0.985, (0.5 * (f_lo + f_hi)) * tp, "gap", ha="right", va="center",
            fontsize=7, color="#8a7550", transform=ax.get_yaxis_transform())

    ax.scatter(d["cloud_phi"], d["cloud_f"] * tp, s=0.6, c="#bdbdbd",
               linewidths=0, zorder=1, rasterized=True)

    # Segments are coloured by the INSTANTANEOUS dominant side, not by a
    # per-branch vote: the two counter-propagating edge modes hybridize where
    # they are degenerate inside the gap, and overlap continuation exchanges
    # identities there (§III C). Per-segment colouring makes the stitch
    # visible instead of hiding it.
    drawn = {"left": False, "right": False}
    for b in d["branches"]:
        w_inst = np.maximum(b["w_left"], b["w_right"])
        if w_inst.max() < edge_cut:
            continue
        side_inst = b["w_left"] > b["w_right"]          # True = left
        pts = np.column_stack([b["phi"], b["freq"] * tp]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cols = np.where(side_inst[:-1], "#2a78d6", "#eb6834")
        lc = LineCollection(segs, colors=cols,
                            linewidths=0.6 + 3.0 * w_inst[:-1], zorder=3,
                            capstyle="round")
        ax.add_collection(lc)
        for side, mask in (("left", side_inst), ("right", ~side_inst)):
            if mask.any() and not drawn[side]:
                wmax = w_inst[mask].max()
                colour = "#2a78d6" if side == "left" else "#eb6834"
                ax.plot([], [], color=colour, lw=2.0,
                        label=f"{side}-end (max $w_{{edge}}$ = {wmax:.2f})")
                drawn[side] = True

    ax.set_xlim(0, 1)
    ax.set_ylim(d["window"][0] * tp, d["window"][1] * tp)
    ax.set_xlabel(r"phason angle $\phi$")
    ax.set_ylabel(r"$f$ (THz)")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.tick_params(labelsize=7)
    ax.set_title(f"{d['n_sites']} sites, {d['n_phi']} phason steps",
                 fontsize=7, color="#666666")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400)
    print(f"figure -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sites", type=int, default=610)
    p.add_argument("--n-phi", type=int, default=400)
    p.add_argument("--f-centre", type=float, default=21.040)
    p.add_argument("--rel-width", type=float, default=0.0067)
    p.add_argument("--edge-cut", type=float, default=0.7)
    p.add_argument("--cache", type=Path, default=Path("fig2_phason.npz"))
    p.add_argument("--out", type=Path, default=Path("fig_phason.pdf"))
    p.add_argument("--recompute", action="store_true")
    a = p.parse_args()

    if a.cache.exists() and not a.recompute:
        z = np.load(a.cache, allow_pickle=True)
        d = {k: z[k] for k in ("cloud_phi", "cloud_f", "cloud_w")}
        d.update(branches=list(z["branches"]), gap=tuple(z["gap"]),
                 f_centre=float(z["f_centre"]), window=tuple(z["window"]),
                 n_sites=int(z["n_sites"]), n_phi=int(z["n_phi"]))
    else:
        d = compute(a.n_sites, a.n_phi, a.f_centre, a.rel_width)
        np.savez(a.cache, **{k: v for k, v in d.items() if k != "branches"},
                 branches=np.array(d["branches"], dtype=object))
        print(f"cache -> {a.cache}")

    plot(d, a.out, a.edge_cut)


if __name__ == "__main__":
    main()
