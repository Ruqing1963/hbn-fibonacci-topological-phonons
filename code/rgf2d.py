#!/usr/bin/env python3
"""
rgf2d.py — Paper 2, 2D anchor, TJC boundary-conforming version
==============================================================

Frequency-domain RGF for the scalar acoustic Helmholtz equation in a ribbon
with smooth rough edges, in EXACT boundary-conforming coordinates
(Tesanovic-Jaklevic-Chaudhari). Replaces the staircase-mask version, whose
step corners produced a grid-DIVERGENT artificial potential (xi = 120/42/16 nm
at h = 2/1/0.5 on the same parameters — no continuum limit).

Transformation (exact, no truncation)
-------------------------------------
    eta = W (y - a(z)) / (b(z) - a(z)) in [0, W],   J(z) = (b - a)/W,
    u = psi / sqrt(J).

Interior equation (derived symbolically, every term kept):

  psi_zz - 2 beta psi_{eta z} + (beta^2 + 1/J^2) psi_{eta eta}
        + [2 (J'/J) beta - beta_z] psi_eta - (J'/J) psi_z
        + [k^2 (1+eps) - J''/(2J) + (3/4)(J'/J)^2] psi = 0,

  beta(eta, z) = (a' + eta J') / J.

Free-edge BC on a tilted wall is NOT psi_eta = 0. Exact condition at
eta = 0 (slope a') and eta = W (slope b'):

  (1 + w'^2)/J * psi_eta = w' psi_z - (w' J' / 2J) psi,   w' = a' or b'.

It couples the boundary row to psi_z and is implemented by ghost elimination,
which adds boundary-row terms to the inter-layer blocks. Pure sidewall sway
(a' = b', J' = 0) then scatters the fundamental mode, as it must physically;
with the naive psi_eta = 0 it would not.

Discretisation: cell-centred eta_i = (i + 1/2) h, z_m = m h; central
differences; generalized block-tridiagonal RGF
    R_m psi_{m-1} + Q_m psi_m + P_m psi_{m+1} = 0,
    g_m = [Q_m - R_m g_{m-1} P_{m-1}]^{-1},  G_1n <- G_1n (-P_{m-1}) g_m,
Fisher-Lee T = Tr[Gam_L G_1n Gam_R G_1n^dag]. Leads are straight (roughness
is tapered to zero over 3 L_c at both ends), so the lead self-energy is the
standard mode sum.

MANDATORY CONVERGENCE GATE
--------------------------
`convergence_gate()` runs a fixed test point at h = 2, 1, 0.5 nm and requires
(i) clean-limit T = N_prop at every h, and (ii) |xi(0.5) - xi(1)| / xi(0.5)
< 10%. The scan driver refuses to start unless the gate passes. This is the
lesson of the staircase failure, hard-coded.
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

V_NM_PS = 20.0
TAU = (1.0 + 5.0 ** 0.5) / 2.0
ITAU = 1.0 / TAU
EPS0 = 0.041


def rough_profile(nz, h, rms, Lc, rng, taper_cells):
    """Smooth Gaussian-correlated profile, tapered to 0 at both ends."""
    if rms <= 0:
        return np.zeros(nz)
    x = (np.arange(nz) - nz // 2) * h
    ker = np.exp(-x ** 2 / (2 * Lc ** 2))
    ker /= np.sqrt((ker ** 2).sum())
    g = np.convolve(rng.standard_normal(nz), ker, mode="same")
    g = rms * g / max(g.std(), 1e-12)
    t = np.ones(nz)
    n = min(taper_cells, nz // 4)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
    t[:n] = ramp
    t[-n:] = ramp[::-1]
    return g * t


def fib_eps(nz, h, d, phi=0.0):
    j = np.floor(np.arange(nz) * h / d).astype(int)
    s = np.floor((j + 2) * ITAU + phi) - np.floor((j + 1) * ITAU + phi)
    return np.where(s == 1, +EPS0 / 2, -EPS0 / 2)


def _dmats(ny, h):
    """Cell-centred Neumann-free transverse difference matrices (interior)."""
    D2 = np.zeros((ny, ny))
    i = np.arange(ny)
    D2[i, i] = -2.0
    D2[i[:-1], i[:-1] + 1] = 1.0
    D2[i[1:], i[1:] - 1] = 1.0
    # plain reflecting ghost (psi_{-1}=psi_0): corrected per layer by the BC
    D2[0, 0] = -1.0
    D2[-1, -1] = -1.0
    D2 /= h * h
    D1 = np.zeros((ny, ny))
    D1[i[:-1], i[:-1] + 1] = 0.5
    D1[i[1:], i[1:] - 1] = -0.5
    D1[0, 0] = -0.5; D1[0, 1] = 0.5
    D1[-1, -1] = 0.5; D1[-1, -2] = -0.5
    D1 /= h
    return D2, D1


def lead_self_energy(W, ny, h, k):
    """Straight Neumann lead: Sigma = U diag(exp(i q_m)) U^T (h^2-scaled units)."""
    D2, _ = _dmats(ny, h)
    lam, U = np.linalg.eigh(-D2)                      # transverse k_m^2
    Mm = 2.0 - h * h * (k * k - lam)
    q = np.arccos(np.clip(Mm / 2.0, -1.0, 1.0)).astype(complex)
    ev = Mm / 2.0 > 1.0
    q[ev] = 1j * np.arccosh(Mm[ev] / 2.0)
    ev2 = Mm / 2.0 < -1.0
    q[ev2] = np.pi + 1j * np.arccosh(-Mm[ev2] / 2.0)
    g = np.exp(1j * q)
    S = U @ np.diag(g) @ U.T
    Gam = 1j * (S - S.conj().T)
    prop = np.abs(Mm) < 2.0
    # group velocity of the discrete dispersion 2cos(q) = M:  v_m = sin(q_m)
    modes = dict(U=U, q=q, prop=prop, v=np.where(prop, np.sin(q.real), 0.0))
    return S, Gam, int(prop.sum()), modes


def master_profiles(L_nm, rms, Lc, seed, h_master=0.25):
    """One roughness realization on a fixed fine grid, downsampled per h."""
    nzm = int(round(L_nm / h_master))
    rng = np.random.default_rng(seed)
    taper = int(3 * Lc / h_master)
    ha = rough_profile(nzm, h_master, rms / np.sqrt(2), Lc, rng, taper)
    hb = rough_profile(nzm, h_master, rms / np.sqrt(2), Lc, rng, taper)
    z = np.arange(nzm) * h_master
    return z, ha, hb


def _half(x):
    """Mid-step values x_{m+1/2}; length nz-1."""
    return 0.5 * (x[1:] + x[:-1])


def _assemble_layers(nz, ny, h, k, eps, a, b):
    """
    Hermitian variational assembly. The quadratic form

        F = int dz d_eta [ (1+g^2)/J |u_eta|^2 + J |u_z|^2
                           - 2 g Re(u_z* u_eta) - k^2 (1+eps) J |u|^2 ],

    g(eta,z) = a' + eta J',  J = (b-a)/W, is discretized term by term on cell
    centres (eta) and layers (z). Symmetry of every block and the exact
    tilted-wall Neumann condition are then properties of the CONSTRUCTION,
    not of a hand-derived stencil: the free edge is the natural boundary
    condition of F, so no ghost elimination exists to get wrong. The mixed
    term is assembled per plaquette; its closed-form contribution is
    +-g/2 on the cross-layer off-diagonals and (g_{m-1/2}-g_{m+1/2})/2
    telescopes on the layer diagonal.

    Returns lists Q[0..nz-1], P[0..nz-2]; the recursion uses R_m = P_{m-1}^T.
    In a straight lead (g=0, J=1, eps=0) the blocks reduce exactly to the
    previous convention Q = 2I - h^2(D2 + k^2 I), P = -I, so lead_self_energy
    and the Fisher-Lee normalisation are unchanged.
    """
    W = ny * h
    J = (b - a) / W                                # per layer
    aph = np.diff(a) / h                           # a'_{m+1/2}, len nz-1
    bph = np.diff(b) / h
    Jph = (bph - aph) / W
    Jh = _half(J)                                  # J_{m+1/2}
    # centred layer derivatives from half-steps (ends: one-sided)
    apc = np.empty(nz); apc[1:-1] = 0.5 * (aph[1:] + aph[:-1])
    apc[0], apc[-1] = aph[0], aph[-1]
    Jpc = np.empty(nz); Jpc[1:-1] = 0.5 * (Jph[1:] + Jph[:-1])
    Jpc[0], Jpc[-1] = Jph[0], Jph[-1]

    eta_link = (np.arange(ny - 1) + 1.0) * h       # eta at links / plaquettes
    i_sup = np.arange(ny - 1)

    Qs, Ps = [], []
    for m in range(nz):
        # --- transverse stiffness (1+g^2)/J on eta-links ---
        gm = apc[m] + eta_link * Jpc[m]
        alpha = (1.0 + gm * gm) / J[m]
        Q = np.zeros((ny, ny))
        Q[i_sup, i_sup] += alpha
        Q[i_sup + 1, i_sup + 1] += alpha
        Q[i_sup, i_sup + 1] -= alpha
        Q[i_sup + 1, i_sup] -= alpha
        # --- longitudinal stiffness J_{m+-1/2} ---
        Jl = Jh[m - 1] if m > 0 else 1.0
        Jr = Jh[m] if m < nz - 1 else 1.0
        Q[np.arange(ny), np.arange(ny)] += Jl + Jr
        # --- mass ---
        Q[np.arange(ny), np.arange(ny)] -= h * h * k * k * (1.0 + eps[m]) * J[m]
        # --- mixed-term diagonal (telescoping g/2 pieces) ---
        gR = (aph[m] + eta_link * Jph[m]) if m < nz - 1 else np.zeros(ny - 1)
        gL = (aph[m - 1] + eta_link * Jph[m - 1]) if m > 0 else np.zeros(ny - 1)
        dg = 0.5 * (gL - gR)
        Q[i_sup, i_sup] += dg                      # site as plaquette corner s1/s3
        Q[i_sup + 1, i_sup + 1] -= dg              # site as corner s2/s4
        Qs.append(Q)
        # --- cross-layer block ---
        if m < nz - 1:
            P = np.zeros((ny, ny))
            P[np.arange(ny), np.arange(ny)] = -Jh[m]
            P[i_sup, i_sup + 1] += 0.5 * gR
            P[i_sup + 1, i_sup] -= 0.5 * gR
            Ps.append(P)
    return Qs, Ps


def transmission(f_thz, W, rms, Lc, nz, h=1.0, d=24.79, iso=True, seed=0,
                 master=None, return_tmatrix=False, profiles=None):
    """Landauer T through the Hermitian-assembled rough ribbon.

    `profiles=(a_array, b_array)` overrides the random roughness with an
    explicit deterministic geometry (used by the Born single-bump gate).
    """
    k = 2 * np.pi * f_thz / V_NM_PS
    ny = int(round(W / h))
    if profiles is not None:
        a, b = profiles
    else:
        if master is None:
            master = master_profiles(nz * h, rms, Lc, seed)
        zm, ham, hbm = master
        z = np.arange(nz) * h
        a = np.interp(z, zm, ham)
        b = W + np.interp(z, zm, hbm)
    eps = fib_eps(nz, h, d) if iso else np.zeros(nz)

    SL, GL, npr, mL = lead_self_energy(W, ny, h, k)
    SR, GR, _, mR = lead_self_energy(W, ny, h, k)

    Qs, Ps = _assemble_layers(nz, ny, h, k, eps, a, b)
    g = np.linalg.inv(Qs[0] - SL)
    G1n = g.copy()
    for m in range(1, nz):
        A = Qs[m] - Ps[m - 1].T @ g @ Ps[m - 1]
        if m == nz - 1:
            A = A - SR
        gnew = np.linalg.inv(A)
        G1n = G1n @ (-Ps[m - 1]) @ gnew
        g = gnew
    T = np.trace(GL @ G1n @ GR @ G1n.conj().T).real
    if return_tmatrix:
        pL, pR = mL["prop"], mR["prop"]
        DL = np.sqrt(2.0 * mL["v"][pL])
        DR = np.sqrt(2.0 * mR["v"][pR])
        t = (DR[:, None] * (mR["U"][:, pR].T @ G1n @ mL["U"][:, pL])
             * DL[None, :])
        tau = np.linalg.svd(t, compute_uv=False) ** 2
        return float(T), npr, tau
    return float(T), npr


def xi_fit(f, W, rms, Lc, lengths_nm, h, seeds, d=24.79, iso=False):
    y = []
    for L in lengths_nm:
        nz = int(round(L / h))
        v = []
        for s in range(seeds):
            m = master_profiles(nz * h, rms, Lc, s)
            v.append(max(transmission(f, W, rms, Lc, nz, h, d, iso, s,
                                      master=m)[0], 1e-290))
        y.append(float(np.mean(np.log(v))))
    p = np.polyfit(np.asarray(lengths_nm, float), np.asarray(y), 1)
    return -2.0 / p[0]


def convergence_gate(verbose=True):
    """
    Hard gate, redesigned. Two checks:

    (1) clean limit T = N_prop at h = 2, 1, 0.5;
    (2) xi grid-converged to <10% between h = 1 and h = 0.5, with the fit
        lengths chosen ADAPTIVELY from a pilot estimate of xi.

    The first version of this gate used fixed lengths [400, 1200, 2400] nm.
    When the true xi greatly exceeds the longest chain, the ln T slope is
    noise-dominated and the gate reports spurious divergence -- the same trap
    as the very first Ziman calibration attempt. Lengths now track xi.
    """
    rep=[]; ok=True
    for h in (2.0,1.0,0.5):
        T,npr=transmission(0.25,12.0,0.0,5.0,int(800/h),h,iso=False)
        rep.append(f"  clean h={h}: T={T:.6f} (N_prop={npr})")
        if abs(T-npr)>1e-3: ok=False
    xi_pilot=xi_fit(0.25,12.0,1.0,5.0,[1000,2500,5000],1.0,seeds=4)
    Ls=[max(int(0.5*xi_pilot),600),int(1.4*xi_pilot),
        min(int(2.5*xi_pilot),15000)]
    rep.append(f"  pilot xi(h=1) = {xi_pilot:.0f} nm -> lengths {Ls} nm")
    xis={}
    for h in (1.0,0.5):
        xis[h]=xi_fit(0.25,12.0,1.0,5.0,Ls,h,seeds=6)
        rep.append(f"  rough h={h}: xi = {xis[h]:.1f} nm")
    dev=abs(xis[0.5]-xis[1.0])/xis[0.5]
    rep.append(f"  |xi(0.5)-xi(1.0)|/xi(0.5) = {dev*100:.1f}%  (limit 10%)")
    ok=ok and (dev<0.10)
    rep.append("  GATE "+("PASSED" if ok else "FAILED — do not scan"))
    if verbose:
        print("convergence gate:"); print("\n".join(rep),flush=True)
    return ok,"\n".join(rep)


if __name__ == "__main__":
    convergence_gate()
