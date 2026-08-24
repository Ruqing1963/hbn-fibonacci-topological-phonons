#!/usr/bin/env python3
"""
phason_tracking.py — Paper 2, topological verdict
=================================================

Track in-gap branches of a finite open Fibonacci mass chain as the phason angle
phi sweeps one full cycle, using eigenvector-overlap continuation rather than
frequency ordering.

Why overlap tracking is necessary
---------------------------------
Sorting states by frequency at each phi silently swaps identities whenever a
branch crosses a bulk band or another branch. A naive "take the most
edge-localised state in the gap at each phi" then produces a zig-zag that looks
like a non-monotonic sweep even when the underlying branch is perfectly
monotonic. The fix is to match states between adjacent phi by maximising

    O_ij = |<u_i(phi) | u_j(phi + dphi)>|^2

over permutations, i.e. a linear assignment problem, solved here with the
Hungarian algorithm (scipy.optimize.linear_sum_assignment on -O).

A second, purely physical, source of apparent non-monotonicity: a chain with
two free ends carries TWO edge branches, one bound to each end, and they
generically sweep the gap in OPPOSITE directions. Left and right edge weights
are therefore reported separately and never merged.

Topological content
-------------------
The Fibonacci chain is a cut-and-project slice of a 2D parent lattice; phi is
the synthetic second dimension. For a gap with Bellissard label q, an edge
branch traverses the gap q times per phi cycle, and that winding number is the
Chern number of the parent. The verdict here is therefore:

    net signed traversals of the branch  ==  q  ->  topological
    branch present but no net traversal  ->  a trivial in-gap state
    no edge-localised branch at all      ->  a quasiperiodic gap, not topological

Usage
-----
    python phason_tracking.py
    python phason_tracking.py --n-sites 987 --n-phi 600 --gap-index 0
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import linear_sum_assignment

A_NM = 0.2504
M_A, M_B = 24.01, 25.01
V_NM_PS = 20.0
TAU = (1.0 + 5.0 ** 0.5) / 2.0
ITAU = 1.0 / TAU
MBAR = 0.5 * (M_A + M_B)
K_FC = MBAR * (V_NM_PS / A_NM) ** 2


def fibonacci_masses(n, block=1, phi=0.0):
    j = np.arange(n // block + 2)
    s = np.floor((j + 2) * ITAU + phi) - np.floor((j + 1) * ITAU + phi)
    return np.repeat(np.where(s == 1, M_A, M_B), block)[:n]


def open_chain(masses):
    """Free-ended (Neumann) chain: eigenfrequencies and eigenvectors."""
    n = masses.size
    s = 1.0 / np.sqrt(masses)
    diag = np.full(n, 2.0 * K_FC)
    diag[0] = diag[-1] = K_FC
    val, vec = eigh_tridiagonal(diag * s * s, -K_FC * s[:-1] * s[1:])
    return np.sqrt(np.maximum(val, 0.0)), vec


def edge_weights(vec, frac=0.06):
    """Fraction of |u|^2 within `frac` of the left end and of the right end."""
    n = vec.shape[0]
    k = max(int(frac * n), 2)
    w2 = vec ** 2
    return w2[:k].sum(0), w2[-k:].sum(0)


def track(n_sites, f_lo, f_hi, n_phi=400, block=1, pad=4.0, frac=0.06):
    """
    Continuation of every state whose frequency lies in a window around the gap.

    Returns a list of branches; each is a dict of arrays indexed by phi step.
    """
    fc = 0.5 * (f_lo + f_hi)
    half = 0.5 * (f_hi - f_lo)
    lo, hi = fc - pad * half, fc + pad * half

    phis = np.linspace(0.0, 1.0, n_phi, endpoint=False)
    prev_vec = prev_idx = None
    tracks = {}                    # branch id -> list of (step, f, wl, wr)
    next_id = 0
    active = {}                    # window position -> branch id

    for step, phi in enumerate(phis):
        f, vec = open_chain(fibonacci_masses(n_sites, block, phi))
        sel = np.flatnonzero((f > lo) & (f < hi))
        V = vec[:, sel]
        wl, wr = edge_weights(V, frac)

        if prev_vec is None:
            new_active = {}
            for j in range(sel.size):
                tracks[next_id] = [(step, f[sel[j]], wl[j], wr[j], phi)]
                new_active[j] = next_id
                next_id += 1
        else:
            O = (prev_vec.T @ V) ** 2                       # overlap matrix
            r, c = linear_sum_assignment(-O)
            new_active = {}
            matched_cols = set()
            for a, b in zip(r, c):
                if O[a, b] < 0.15:                          # identity lost
                    continue
                bid = active.get(a)
                if bid is None:
                    continue
                tracks[bid].append((step, f[sel[b]], wl[b], wr[b], phi))
                new_active[b] = bid
                matched_cols.add(b)
            for b in range(sel.size):                       # states entering
                if b not in matched_cols:
                    tracks[next_id] = [(step, f[sel[b]], wl[b], wr[b], phi)]
                    new_active[b] = next_id
                    next_id += 1

        prev_vec, active = V, new_active

    out = []
    for bid, rows in tracks.items():
        if len(rows) < 3:
            continue
        arr = np.array(rows, float)
        out.append(dict(id=bid, step=arr[:, 0], freq=arr[:, 1],
                        w_left=arr[:, 2], w_right=arr[:, 3], phi=arr[:, 4]))
    return out, (f_lo, f_hi)


def classify(branches, gap, min_edge=0.5):
    """Report, per branch, whether it is edge-bound and how far it sweeps."""
    f_lo, f_hi = gap
    width = f_hi - f_lo
    rows = []
    for b in branches:
        el, er = b["w_left"].max(), b["w_right"].max()
        side = "left" if el > er else "right"
        w = max(el, er)
        if w < min_edge:
            continue
        inside = (b["freq"] > f_lo) & (b["freq"] < f_hi)
        if inside.sum() < 3:
            continue
        fin = b["freq"][inside]
        swept = (fin.max() - fin.min()) / width
        d = np.diff(fin)
        d = d[np.abs(d) > 1e-6]
        flips = int((np.diff(np.sign(d)) != 0).sum()) if d.size > 1 else 0
        rows.append(dict(id=b["id"], side=side, edge_weight=w,
                         coverage=b["phi"].size, swept=swept, flips=flips,
                         net=np.sign(fin[-1] - fin[0]), n_in=int(inside.sum())))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sites", type=int, default=610)
    p.add_argument("--n-phi", type=int, default=400)
    p.add_argument("--f-centre", type=float, default=21.040)  # THz, q=1 gap
    p.add_argument("--rel-width", type=float, default=0.0067)
    args = p.parse_args()

    fc = 2 * np.pi * args.f_centre
    half = 0.5 * args.rel_width * fc
    branches, gap = track(args.n_sites, fc - half, fc + half, args.n_phi)
    rows = classify(branches, gap)

    print(f"chain {args.n_sites} sites, {args.n_phi} phason steps, "
          f"gap {args.f_centre:.3f} THz +/- {half/(2*np.pi)*1e3:.2f} GHz")
    print(f"branches tracked in window: {len(branches)}   "
          f"edge-bound branches in gap: {len(rows)}\n")
    print(f"{'id':>5}{'side':>7}{'edge w':>9}{'pts in gap':>12}"
          f"{'swept':>9}{'flips':>7}{'net':>6}")
    for r in sorted(rows, key=lambda r: -r["edge_weight"]):
        print(f"{r['id']:5d}{r['side']:>7}{r['edge_weight']:9.3f}"
              f"{r['n_in']:12d}{r['swept']:9.2f}{r['flips']:7d}{r['net']:+6.0f}")

    lefts = [r for r in rows if r["side"] == "left"]
    rights = [r for r in rows if r["side"] == "right"]
    print(f"\nleft-bound branches: {len(lefts)}   right-bound: {len(rights)}")
    if lefts and rights:
        nl = np.sign(sum(r["net"] for r in lefts))
        nr = np.sign(sum(r["net"] for r in rights))
        print(f"net sweep direction: left {nl:+.0f}, right {nr:+.0f}  "
              f"-> {'OPPOSITE (counter-propagating pair)' if nl*nr < 0 else 'same sign'}")
    mono = [r for r in rows if r["flips"] <= 1 and r["swept"] > 0.6]
    print(f"\nVERDICT: {len(mono)} branch(es) sweep >60% of the gap with <=1 "
          f"sign change -> {'clean traversal' if mono else 'no clean traversal'}")


if __name__ == "__main__":
    main()
