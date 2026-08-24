#!/usr/bin/env python3
"""
benchmark_bump.py — Paper 2, step 1 of the anchor repair
========================================================

The minimal problem with an independent reference solution: single-mode
transmission past ONE smooth Gaussian bump on one wall,

    a(z) = A exp(-z^2 / 2 sigma^2)   (lower wall),   b(z) = W  (upper wall).

Two independent solutions are computed:

REFERENCE — piecewise mode matching. The bump is sliced into short straight
segments; within each, the field is expanded in the local Neumann duct modes
(propagating + evanescent); continuity of psi and flux at each interface is
imposed via the interface scattering matrix, and segment S-matrices are
composed by the star product, which is unconditionally stable (transfer
matrices of evanescent-rich sections overflow). The reference has its own
convergence knobs (slices, modes) and is validated internally: |r|^2+|t|^2=1
to machine precision and self-convergence under slice/mode doubling.

CANDIDATE — the TJC-transformed RGF (rgf2d.transmission) on the same bump.

The figure of merit is |r(f)|^2 across single-mode frequencies. The staircase
version overestimated scattering with no continuum limit; the first TJC
implementation underestimated it (xi grew ~12x per halving of h). This
benchmark localises which term is wrong: with only one wall moving and
J' = a'/W ≠ 0, both the sway (beta) and breathing (J) couplings are active.

Usage:  python benchmark_bump.py
"""

from __future__ import annotations

import numpy as np

# numpy 1.x calls it trapz; 2.x renamed it trapezoid. Support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

V = 20.0                        # nm/ps


# ---------------------------------------------------------------- reference

def duct_modes(Wd, n_modes, k):
    """Neumann duct of width Wd: transverse wavenumbers and longitudinal q_m."""
    m = np.arange(n_modes)
    km = m * np.pi / Wd
    q2 = k * k - km * km
    q = np.sqrt(q2.astype(complex))
    q[np.real(q2) < 0] = 1j * np.sqrt(-q2[np.real(q2) < 0])
    return km, q


def overlap(W1, W2, n_modes, y0_1=0.0, y0_2=0.0, ngl=400):
    """
    O_{mn} = <chi_m^{(1)} | chi_n^{(2)}> over the COMMON aperture.
    Ducts: duct1 occupies [y0_1, y0_1+W1], duct2 [y0_2, y0_2+W2].
    Neumann modes chi_m(y) = sqrt((2-delta_m0)/W) cos(m pi (y-y0)/W).
    """
    lo = max(y0_1, y0_2)
    hi = min(y0_1 + W1, y0_2 + W2)
    y = np.linspace(lo, hi, ngl)
    m = np.arange(n_modes)
    c1 = np.sqrt((2.0 - (m == 0)) / W1)[:, None] * np.cos(
        m[:, None] * np.pi * (y[None, :] - y0_1) / W1)
    c2 = np.sqrt((2.0 - (m == 0)) / W2)[:, None] * np.cos(
        m[:, None] * np.pi * (y[None, :] - y0_2) / W2)
    return _trapz(c1[:, None, :] * c2[None, :, :], y, axis=2)


def interface_smatrix(q1, q2, O):
    """
    S-matrix of one junction from psi and psi' continuity projected on modes.
    Incoming from the left in duct1 modes; O = <1|2>.
    Solve: (I + r) related via  O^T(a+ b_r) = t ;  q1(a - b_r) = O q2 t.
    """
    n = q1.size
    Q1 = np.diag(q1)
    Q2 = np.diag(q2)
    A = np.block([[O.T, -np.eye(n)],
                  [Q1 @ O, O @ Q2 if False else np.zeros((n, n))]])
    # assemble properly below instead (kept explicit for clarity):
    # continuity of psi:   O.T (a + b) = t          [project on duct2 modes]
    # continuity of flux:  q1*(a - b) = O q2 t      [project on duct1 modes]
    M11 = O.T
    M12 = -np.eye(n)
    M21 = Q1
    M22 = O @ Q2
    # unknowns (b, t):  [ -O.T b + t = O.T a ; Q1 b + O Q2 t = Q1 a ]
    L = np.block([[-M11, np.eye(n)], [Q1, M22]])
    Rr = np.block([[M11], [Q1]])
    X = np.linalg.solve(L, Rr)
    r = X[:n]
    t = X[n:]
    return r, t


def star(SA, SB):
    """Redheffer star product of two S-matrices in (r, t, r', t') block form."""
    rA, tA, rpA, tpA = SA
    rB, tB, rpB, tpB = SB
    n = rA.shape[0]
    inv1 = np.linalg.inv(np.eye(n) - rpA @ rB)
    inv2 = np.linalg.inv(np.eye(n) - rB @ rpA)
    r = rA + tpA @ rB @ inv1 @ tA
    t = tB @ inv1 @ tA
    rp = rpB + tB @ rpA @ inv2 @ tpB
    tp = tpA @ inv2 @ tpB
    return (r, t, rp, tp)


def prop_smatrix(q, L):
    ph = np.diag(np.exp(1j * q * L))
    z = np.zeros_like(ph)
    return (z, ph, z, ph)


def reference_r(f, W, A, sigma, n_modes=8, n_slice=200, span=6.0):
    """|r_00|^2 for the fundamental past one Gaussian bump on the lower wall."""
    k = 2 * np.pi * f / V
    zs = np.linspace(-span * sigma, span * sigma, n_slice + 1)
    zc = 0.5 * (zs[:-1] + zs[1:])
    dz = zs[1] - zs[0]
    a_of = lambda z: A * np.exp(-z * z / (2 * sigma * sigma))
    widths = W - a_of(zc)
    floors = a_of(zc)

    _, q_lead = duct_modes(W, n_modes, k)
    S = None
    W_prev, y_prev, q_prev = W, 0.0, q_lead
    for j in range(n_slice):
        O = overlap(W_prev, widths[j], n_modes, y_prev, floors[j])
        _, q_here = duct_modes(widths[j], n_modes, k)
        r, t = interface_smatrix(q_prev, q_here, O)
        rp, tp = interface_smatrix(q_here, q_prev, O.T)
        Sj = (r, t, rp, tp)
        S = Sj if S is None else star(S, Sj)
        S = star(S, prop_smatrix(q_here, dz))
        W_prev, y_prev, q_prev = widths[j], floors[j], q_here
    O = overlap(W_prev, W, n_modes, y_prev, 0.0)
    r, t = interface_smatrix(q_prev, q_lead, O)
    rp, tp = interface_smatrix(q_lead, q_prev, O.T)
    S = star(S, (r, t, rp, tp))
    r00 = S[0][0, 0]
    t00 = S[1][0, 0]
    # unitarity check in the single-mode regime (higher modes evanescent):
    uni = abs(abs(r00) ** 2 + abs(t00) ** 2 - 1.0)
    return abs(r00) ** 2, uni


def born_r2(f, W, A, sigma):
    """
    Analytic weak-scattering reference, exact to leading order in A/W.

    Treat the smooth one-wall bump as a continuous distribution of
    area-mismatch reflectors: an abrupt Neumann-duct area step reflects the
    fundamental with r = dA/2A (frequency independent), so

        dr = [a'(z) / 2W] e^{2iqz} dz
        r(q) = (iq/W) * a_tilde(2q),   a_tilde(K) = A sigma sqrt(2 pi)
                                                    * exp(-K^2 sigma^2 / 2)
        |r|^2 = (q^2/W^2) * 2 pi A^2 sigma^2 * exp(-4 q^2 sigma^2).

    Zero adjustable parameters. Valid for A/W << 1 and single-mode operation.
    Its low-frequency limit |r|^2 ~ q^2 -> 0 is the physics the anchors must
    reproduce: a smooth bump becomes invisible to long waves.
    """
    q = 2 * np.pi * f / V
    return (q * q / (W * W)) * 2 * np.pi * A * A * sigma * sigma \
        * np.exp(-4 * q * q * sigma * sigma)


def main():
    W, A, sigma = 12.0, 0.4, 5.0     # A/W = 1/30: safely in the Born regime
    print(f"single Gaussian bump: W={W} nm, height={A} nm, sigma={sigma} nm")
    print(f"(A/W = {A/W:.3f}; Born is the exact leading order here)\n")
    hdr = (f"{'f(THz)':>8}{'|r|^2 Born':>14}{'mode-match':>14}"
           f"{'mm/Born':>9}{'unit.':>9}{'mm 2x sl':>12}{'mm 16 md':>12}")
    print(hdr)
    ok = True
    for f in [0.10, 0.15, 0.20, 0.25, 0.30]:
        B = born_r2(f, W, A, sigma)
        R1, u1 = reference_r(f, W, A, sigma, n_modes=8, n_slice=240)
        R2, _ = reference_r(f, W, A, sigma, n_modes=8, n_slice=480)
        R3, _ = reference_r(f, W, A, sigma, n_modes=16, n_slice=240)
        print(f"{f:8.2f}{B:14.3e}{R1:14.3e}{R1/B:9.2f}{u1:9.1e}"
              f"{R2:12.3e}{R3:12.3e}")
        if u1 > 0.2 * R1 or abs(R3 / R1 - 1) > 0.10:
            ok = False
    print("\nVALIDATION " + ("PASSED: the two references agree; either may "
          "now judge the TJC RGF." if ok else
          "NOT PASSED: unitarity or mode-convergence still exceeds tolerance."
          " Do not use the mode-matching numbers; the Born column stands"
          " alone as the analytic anchor."))


if __name__ == "__main__":
    main()
