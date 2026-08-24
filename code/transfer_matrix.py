#!/usr/bin/env python3
"""
transfer_matrix.py — Paper 2 core
=================================

Coherent phonon transport through a one-dimensional Fibonacci mass chain:
Lyapunov exponents, integrated density of states, Bellissard gap labelling,
and phason pumping.

Migrated from Paper 1's transport.py. The force constant is isotope-independent
(Born-Oppenheimer), so a single primitive-cell value serves every arrangement;
only the diagonal mass matrix changes.

Model
-----
    -M_n w^2 u_n = K (u_{n+1} - 2 u_n + u_{n-1})
    (u_{n+1}, u_n)^T = T_n (u_n, u_{n-1})^T,
    T_n = [[2 - M_n w^2 / K, -1], [1, 0]]

    gamma(w) = lim (1/N) ln ||prod T_n||      inverse localisation length
    xi(w)    = a / gamma(w)
    T(L)     ~ exp(-2L/xi)                    inside a gap

Fibonacci chain by cut-and-project, with the phason angle phi as an explicit
argument:

    s_n(phi) = floor((n+2)/tau + phi) - floor((n+1)/tau + phi)

Sweeping phi over one period is the pump cycle; the number of times an edge
branch traverses a gap equals that gap's label q.

-------------------------------------------------------------------------
WHAT THIS MODULE CANNOT DO
-------------------------------------------------------------------------
It cannot test edge-roughness decoherence. Adding random on-site masses to a
strictly one-dimensional chain produces *additional localisation* — gamma rises
everywhere — whereas physical edge roughness in a ribbon scatters carriers
*between transverse subbands*, which destroys the single-channel coherence that
makes the gap observable. A 1D chain has no other channel to scatter into, so
the two mechanisms are not the same physics and the 1D result is not evidence
either way. Use `gap_contrast()` as a bulk-disorder robustness check only, and
see the quasi-1D multichannel extension for the decoherence question.
-------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

# h-BN parameters (Paper 1, Sec. IV E)
A_NM = 0.2504                    # boron sublattice spacing [nm]
M_A, M_B = 24.01, 25.01          # 10-BN / 11-BN formula unit [amu]
V_NM_PS = 20.0                   # in-plane LA velocity [nm/ps]

TAU = (1.0 + 5.0 ** 0.5) / 2.0
ITAU = 1.0 / TAU

MBAR = 0.5 * (M_A + M_B)
K_FC = MBAR * (V_NM_PS / A_NM) ** 2          # amu/ps^2
OMEGA_MAX = 2.0 * np.sqrt(K_FC / MBAR)
R_IFACE = 0.25 * (M_B - M_A) / MBAR          # single-interface reflection


# ---------------------------------------------------------------- chains

def fibonacci_letters(n: int, phi: float = 0.0) -> np.ndarray:
    """Cut-and-project Fibonacci word; 1 = A, 0 = B. `phi` is the phason angle."""
    j = np.arange(n)
    s = np.floor((j + 2) * ITAU + phi) - np.floor((j + 1) * ITAU + phi)
    return s.astype(int)


def fibonacci_masses(n_sites: int, block: int = 1, phi: float = 0.0) -> np.ndarray:
    """Fibonacci chain, each letter decorated into `block` identical sites."""
    letters = fibonacci_letters(n_sites // block + 2, phi)
    m = np.repeat(np.where(letters == 1, M_A, M_B), block)
    return m[:n_sites]


def periodic_masses(n_sites: int, period_sites: int) -> np.ndarray:
    idx = np.arange(n_sites) % period_sites
    return np.where(idx < period_sites // 2, M_A, M_B)


def random_masses(n_sites: int, c: float, rng) -> np.ndarray:
    return np.where(rng.random(n_sites) < c, M_A, M_B)


# ------------------------------------------------------- Lyapunov / IDOS

def lyapunov(masses: np.ndarray, omegas, K: float = K_FC,
             renorm: int = 32) -> np.ndarray:
    """
    gamma(omega) per lattice site, vectorised over omega.

    Renormalising every `renorm` steps prevents overflow; the accumulated log
    norms are the Lyapunov sum. Cost O(N * n_omega), no diagonalisation.
    """
    w2 = np.atleast_1d(np.asarray(omegas, float)) ** 2
    x = np.ones_like(w2)
    y = np.zeros_like(w2)
    acc = np.zeros_like(w2)
    for j, M in enumerate(masses):
        d = 2.0 - M * w2 / K
        x, y = d * x - y, x
        if j % renorm == 0:
            r = np.hypot(x, y)
            ok = r > 0
            acc[ok] += np.log(r[ok])
            x[ok] /= r[ok]
            y[ok] /= r[ok]
    r = np.hypot(x, y)
    ok = r > 0
    acc[ok] += np.log(r[ok])
    return acc / masses.size


def localisation_length(masses, omegas, K: float = K_FC) -> np.ndarray:
    """xi(omega) in nm."""
    return A_NM / np.maximum(lyapunov(masses, omegas, K), 1e-300)


def idos(masses: np.ndarray, omegas, K: float = K_FC,
         renorm: int = 64) -> np.ndarray:
    """
    Integrated density of states by Sturm node counting.

    For the Dirichlet problem u_0 = 0, u_1 = 1, the number of sign changes of
    u_n below frequency omega equals the number of eigenvalues below omega^2.
    O(N) and reuses the same recursion as the Lyapunov exponent.
    """
    w2 = np.atleast_1d(np.asarray(omegas, float)) ** 2
    x = np.ones_like(w2)
    y = np.zeros_like(w2)
    nodes = np.zeros(w2.size, dtype=np.int64)
    for j, M in enumerate(masses):
        d = 2.0 - M * w2 / K
        xn = d * x - y
        nodes += (xn * x < 0)
        y, x = x, xn
        if j % renorm == 0:
            r = np.hypot(x, y)
            ok = r > 0
            x[ok] /= r[ok]
            y[ok] /= r[ok]
    return nodes / masses.size


# ----------------------------------------------------------- gap labels

def gap_label(idos_value: float, qmax: int = 30) -> tuple[int, int, float]:
    """
    Bellissard gap labelling: in a gap the IDOS takes a value in
    Z + tau^-1 Z, i.e. N = (p + q/tau) mod 1 for integers p, q.

    Returns (p, q, residual).

    CAUTION on interpretation. Z + tau^-1 Z is dense in [0,1], so "the IDOS
    lies in Z + tau^-1 Z" is not by itself falsifiable — some (p,q) fits any
    number. The content of the test is quantitative: genuine gaps fit to
    ~1e-6 with small |q|, whereas random values in [0,1] fit only to ~4e-3.
    Always report the residual, and calibrate it against random controls
    (see `labelling_significance`).
    """
    best = (np.inf, 0, 0)
    for q in range(-qmax, qmax + 1):
        v = q * ITAU
        p = int(round(idos_value - v))
        err = abs(idos_value - (p + v))
        if err < best[0]:
            best = (err, p, q)
    return best[1], best[2], best[0]


def labelling_significance(values, n_control: int = 5000, seed: int = 0) -> dict:
    """Compare measured gap residuals against uniformly random IDOS values."""
    rng = np.random.default_rng(seed)
    em = np.array([gap_label(v)[2] for v in values])
    ec = np.array([gap_label(v)[2] for v in rng.random(n_control)])
    return dict(measured_median=float(np.median(em)),
                measured_max=float(em.max()),
                control_median=float(np.median(ec)),
                separation=float(np.median(ec) / np.median(em)),
                p_value=float(np.mean(ec <= em.max())))


def find_gaps(omegas, gamma, threshold=5e-4, window=3, n_max=12):
    """Local maxima of gamma above `threshold`, strongest first."""
    idx = [i for i in range(window, gamma.size - window)
           if gamma[i] > threshold and gamma[i] == gamma[i - window:i + window + 1].max()]
    return sorted(sorted(idx, key=lambda i: -gamma[i])[:n_max])


# -------------------------------------------------------- phason pumping

def open_chain_spectrum(masses: np.ndarray, K: float = K_FC):
    """Eigenfrequencies and eigenvectors of a finite free-ended chain."""
    n = masses.size
    s = 1.0 / np.sqrt(masses)
    diag = np.full(n, 2.0 * K)
    diag[0] = diag[-1] = K                       # free ends
    val, vec = eigh_tridiagonal(diag * s * s, -K * s[:-1] * s[1:])
    return np.sqrt(np.maximum(val, 0.0)), vec


def phason_sweep(n_sites: int, gap_centre: float, gap_halfwidth: float,
                 n_phi: int = 240, block: int = 1, edge_frac: float = 0.08):
    """
    Track in-gap states as the phason angle sweeps one full cycle.

    Returns a dict with the phi values at which an in-gap state exists, its
    frequency, and its edge weight (fraction of |u|^2 within `edge_frac` of
    either end). A gap with label q should show q traversals per cycle.
    """
    phis, freqs, edges = [], [], []
    for phi in np.linspace(0.0, 1.0, n_phi, endpoint=False):
        f, vec = open_chain_spectrum(fibonacci_masses(n_sites, block, phi))
        sel = (f > gap_centre - gap_halfwidth) & (f < gap_centre + gap_halfwidth)
        if not sel.any():
            continue
        w2 = vec[:, sel] ** 2
        mask = np.arange(n_sites) < edge_frac * n_sites
        ew = w2[mask].sum(0) + w2[::-1][mask].sum(0)
        k = int(np.argmax(ew))
        phis.append(phi); freqs.append(f[sel][k]); edges.append(ew[k])
    phis, freqs, edges = map(np.asarray, (phis, freqs, edges))
    swept = (freqs.max() - freqs.min()) / (2 * gap_halfwidth) if freqs.size else 0.0
    return dict(phi=phis, freq=freqs, edge_weight=edges,
                coverage=phis.size / n_phi, fraction_of_gap_swept=float(swept))


# ------------------------------------------------------- robustness check

def gap_contrast(masses: np.ndarray, omegas, gap_index: int,
                 eta: float = 0.0, n_rep: int = 3, seed: int = 0,
                 band_exclude: float = 0.06) -> dict:
    """
    Ratio gamma(gap) / median gamma(passband) under bulk mass disorder of rms
    fractional amplitude `eta`.

    This is a BULK robustness check, not a decoherence test — see the module
    docstring. A high contrast means the gap remains spectrally distinct
    against added 1D mass disorder; it says nothing about inter-subband
    scattering in a real ribbon.
    """
    omegas = np.asarray(omegas, float)
    w0 = omegas[gap_index]
    band = np.abs(omegas - w0) > band_exclude * w0
    gg = gb = 0.0
    for s in range(n_rep):
        rng = np.random.default_rng(seed + s)
        m = masses * (1.0 + eta * rng.standard_normal(masses.size)) if eta else masses
        g = lyapunov(m, omegas)
        gg += g[gap_index]
        gb += float(np.median(g[band]))
        if not eta:
            break
    n = n_rep if eta else 1
    gg, gb = gg / n, gb / n
    return dict(gamma_gap=gg, gamma_band=gb,
                contrast=gg / max(gb, 1e-300), xi_gap_nm=A_NM / max(gg, 1e-300))


# --------------------------------------------------------------- design

def coherence_criterion(f_thz: float, width_nm: float, roughness_nm: float,
                        xi_nm: float, v_nm_ps: float = V_NM_PS) -> dict:
    """
    Compare the coherent localisation length against the edge-decoherence mean
    free path and the single-transverse-mode condition.

    Lambda_edge = W / (4 q^2 Delta^2)     (Ziman specularity, small q*Delta)
    N_channels  ~ 2 W f / v

    A coherent gap requires BOTH Lambda_edge > 3 xi AND N_channels ~ 1:
    a wide ribbon defeats decoherence but reintroduces the transverse-channel
    averaging that smears Cantor gaps (Paper 1, Sec. IV D).
    """
    q = 2 * np.pi * f_thz / v_nm_ps
    lam = v_nm_ps / f_thz
    Lambda = width_nm / (4 * q * q * roughness_nm ** 2)
    n_ch = 2 * width_nm / lam
    return dict(wavelength_nm=lam, Lambda_edge_nm=Lambda, n_channels=n_ch,
                ratio=Lambda / xi_nm,
                coherent=bool(Lambda > 3 * xi_nm),
                single_mode=bool(n_ch <= 1.0),
                verdict="OK" if (Lambda > 3 * xi_nm and n_ch <= 1.0)
                        else ("decohered" if Lambda <= 3 * xi_nm
                              else "multimode: gaps smeared"))


if __name__ == "__main__":
    print(f"K = {K_FC:.0f} amu/ps^2   omega_max = {OMEGA_MAX/(2*np.pi):.2f} THz"
          f"   r_interface = {R_IFACE:.4f}")
    N = 60_000
    m = fibonacci_masses(N)
    w = np.linspace(0.02, 0.995, 3000) * OMEGA_MAX
    g, n = lyapunov(m, w), idos(m, w)
    gaps = find_gaps(w, g)
    print(f"\n{'f (THz)':>9}{'gamma':>10}{'xi (nm)':>10}{'IDOS':>10}"
          f"{'(p,q)':>10}{'resid':>10}")
    vals = []
    for i in gaps:
        p, q, e = gap_label(n[i])
        vals.append(n[i])
        print(f"{w[i]/(2*np.pi):9.3f}{g[i]:10.5f}{A_NM/g[i]:10.1f}"
              f"{n[i]:10.5f}{f'({p},{q})':>10}{e:10.2e}")
    s = labelling_significance(vals)
    print(f"\nlabelling: measured residual {s['measured_median']:.2e}, "
          f"random control {s['control_median']:.2e}, "
          f"separation {s['separation']:.0f}x, p = {s['p_value']:.1e}")
