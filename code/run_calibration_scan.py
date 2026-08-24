#!/usr/bin/env python3
"""
run_calibration_scan.py — Paper 2, production calibration
=========================================================

Multicore 2D recursive-Green's-function scan that measures the roughness
localisation length xi_2D(Delta_r, W, f) in a strictly single-mode ribbon, and
fits the three scaling exponents against the Ziman prediction (-2, +1, -2).

This is the anchor that replaces the Ziman formula as an assumption. It is also
what the earlier failed calibration attempt lacked: enough seeds to beat
mesoscopic fluctuations, and a parameter grid that never crosses a transverse
mode threshold.

-------------------------------------------------------------------------
THE TWO FAILURE MODES THIS SCRIPT IS BUILT TO AVOID
-------------------------------------------------------------------------
1. MODE-THRESHOLD CROSSING. A ribbon is single-mode when the first excited
   transverse mode is evanescent, k < k_1 = pi/W, i.e.

       f < v / (2 W)                    <-- one inequality, not two
                                            (W < v/2f is the same statement)

   A scan that varies W or f without enforcing this will silently cross into
   two-mode operation, where the transmission jumps by a whole channel. That
   is what produced exponents of +5.45 and -2.91 for d(ln xi)/d(ln W) in the
   preliminary scan: the sign was not even stable. We enforce
   f <= margin * v/(2W) with margin < 1 so no point sits on the threshold.

2. MESOSCOPIC UNDER-SAMPLING. ln T fluctuates by order unity between disorder
   realisations. Two seeds and three lengths cannot constrain a slope. We
   require >= 20 seeds and >= 5 lengths, and report the fit uncertainty so an
   under-sampled point is visible rather than silently wrong.
-------------------------------------------------------------------------

Acceptance criterion
--------------------
All THREE exponents must agree with Ziman to within `--tol` (default 0.15).
A single-point match on the absolute value is NOT sufficient: the retracted
claim that "Ziman is inapplicable at lambda/W ~ 2" came from exactly that.

Usage
-----
    python run_calibration_scan.py --workers 16 --out calib.json
    python run_calibration_scan.py --dry-run          # grid + cost, no compute
    python run_calibration_scan.py --resume calib.json
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rgf2d import transmission, convergence_gate    # noqa: E402

V_NM_PS = 20.0


# ------------------------------------------------------------------ geometry

def single_mode_ok(f_thz: float, W_nm: float, margin: float = 0.8) -> bool:
    """True when only the m = 0 transverse mode propagates, with headroom."""
    return f_thz <= margin * V_NM_PS / (2.0 * W_nm)


def ziman_xi(f_thz: float, W_nm: float, rms_nm: float) -> float:
    """Ziman single-channel localisation length [nm]; xi = 2 * mean free path.

    Kept as a reported comparison only. The 24-seed TJC scan of 2026-08 showed
    Ziman underestimates xi by a median 11.6x in the single-mode wave limit
    (lambda >> W); the acceptance reference is `born_law_xi` below."""
    q = 2 * np.pi * f_thz / V_NM_PS
    return W_nm / (2.0 * q * q * rms_nm * rms_nm)


def born_law_xi(f_thz: float, W_nm: float, rms_nm: float, Lc_nm: float) -> float:
    """
    Wave-limit Born-transparency law (zero parameters):

        xi = 2 W^2 / [q^2 S_w(2q)],   S_w = rms^2 sqrt(2pi) Lc exp(-2 q^2 Lc^2)

    Derivation: single-bump Born amplitude r(q) = (iq/W) a~(2q) [verified
    against the TJC RGF to <2% in its validity band], generalized to a random
    width profile, with the 1D localization relation xi = 2/gamma. Verified
    against the 15-point 24-seed scan: median meas/law = 1.001; exponents
    rms -1.90+/-0.09 (law -2), W +2.03+/-0.12 (law +2), f -1.40+/-0.14
    (law -2+4q^2Lc^2 = -1.38). Deviations appear only where second-order
    corrections are expected: rms/W > 0.05 or q Lc < 0.2.
    """
    q = 2 * np.pi * f_thz / V_NM_PS
    Sw = rms_nm ** 2 * np.sqrt(2 * np.pi) * Lc_nm * np.exp(-2 * q * q * Lc_nm ** 2)
    return 2.0 * W_nm ** 2 / (q * q * Sw)


def choose_lengths(xi_guess_nm: float, n_len: int, h: float,
                   span=(0.4, 3.0)) -> list[int]:
    """
    Lengths spanning ~0.4 to 3 localisation lengths.

    Too short and ln T has not started to fall; too long and it underflows.
    Both give a meaningless slope, so the window is tied to the Ziman estimate
    rather than fixed in absolute terms.
    """
    L = np.linspace(span[0] * xi_guess_nm, span[1] * xi_guess_nm, n_len)
    return [max(int(round(x / h)), 50) for x in L]


# ------------------------------------------------------------------- worker

@dataclass(frozen=True)
class Point:
    f: float
    W: float
    rms: float
    Lc: float
    h: float
    n_seed: int
    n_len: int

    def key(self) -> str:
        return f"f{self.f:.4f}_W{self.W:.2f}_r{self.rms:.4f}_Lc{self.Lc:.2f}"


def _one(args):
    """One (point, length, seed) transmission. Top level so Pool can pickle it."""
    pt, nz, seed = args
    try:
        T, npr = transmission(pt.f, pt.W, pt.rms, pt.Lc, nz, pt.h,
                              iso=False, seed=seed)
        return (nz, seed, float(T), int(npr), None)
    except Exception as exc:                            # keep the scan alive
        return (nz, seed, float("nan"), -1, repr(exc))


def fit_point(pt: Point, results) -> dict:
    """Fit ln T = c - 2 L / xi over lengths, averaging seeds."""
    by_len = {}
    modes = set()
    for nz, seed, T, npr, err in results:
        if err or not np.isfinite(T) or T <= 0:
            continue
        by_len.setdefault(nz, []).append(np.log(T))
        modes.add(npr)
    if len(by_len) < 3:
        return dict(ok=False, reason="too few usable lengths")
    nzs = np.array(sorted(by_len))
    L = nzs * pt.h
    y = np.array([np.mean(by_len[int(n)]) for n in nzs])
    s = np.array([np.std(by_len[int(n)]) / np.sqrt(len(by_len[int(n)]))
                  for n in nzs])
    wgt = 1.0 / np.maximum(s, 1e-6) ** 2
    Xd = np.vstack([L, np.ones_like(L)]).T
    Wm = np.diag(wgt)
    C = np.linalg.inv(Xd.T @ Wm @ Xd)
    b = C @ Xd.T @ Wm @ y
    slope, slope_err = b[0], np.sqrt(C[0, 0])
    if slope >= 0:
        return dict(ok=False, reason="non-decaying transmission")
    xi = -2.0 / slope
    resid = y - Xd @ b
    ss_res = float(np.sum(wgt * resid ** 2))
    ss_tot = float(np.sum(wgt * (y - np.average(y, weights=wgt)) ** 2))
    return dict(ok=True, xi_nm=float(xi),
                extrapolative=bool(max(L) < 0.5 * xi),
                xi_err_nm=float(2.0 * slope_err / slope ** 2),
                r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                n_prop=sorted(modes), lengths_nm=L.tolist(),
                lnT=y.tolist(), lnT_err=s.tolist(),
                xi_ziman_nm=ziman_xi(pt.f, pt.W, pt.rms))


# --------------------------------------------------------------------- grid

def build_grid(args) -> list[Point]:
    """
    Three one-at-a-time ladders through a common base point, so each exponent
    is measured along a line on which the other two variables are fixed.
    """
    base = dict(f=args.f0, W=args.W0, rms=args.rms0, Lc=args.Lc,
                h=args.h, n_seed=args.seeds, n_len=args.lengths)
    pts, seen = [], set()

    def add(**kw):
        p = Point(**{**base, **kw})
        if not single_mode_ok(p.f, p.W, args.margin):
            print(f"  skip (multimode): f={p.f:.3f} W={p.W:.1f} "
                  f"(limit f<{args.margin*V_NM_PS/(2*p.W):.3f})")
            return
        if p.key() not in seen:
            seen.add(p.key())
            pts.append(p)

    # roughness does not affect the mode count, so this ladder is unrestricted
    for r in np.geomspace(args.rms0 / 2.5, args.rms0 * 2.5, args.n_rms):
        add(rms=float(r))

    # width and frequency ladders are CLIPPED to the single-mode region rather
    # than generated and discarded, so the lever arm is used symmetrically
    W_max = args.margin * V_NM_PS / (2.0 * args.f0)
    if args.W0 > W_max:
        raise SystemExit(
            f"base point is not single-mode: W0={args.W0} nm exceeds "
            f"{W_max:.1f} nm at f0={args.f0} THz with margin {args.margin}. "
            f"Lower W0, lower f0, or raise --margin (not recommended).")
    for W in np.geomspace(args.W0 / 2.5, min(args.W0 * 2.5, 0.99 * W_max),
                          args.n_W):
        add(W=float(W))

    f_max = args.margin * V_NM_PS / (2.0 * args.W0)
    for f in np.geomspace(args.f0 / 2.5, min(args.f0 * 2.5, 0.99 * f_max),
                          args.n_f):
        add(f=float(f))
    return pts


def fit_exponent(points, values, errs, var: str) -> tuple[float, float, int]:
    """Weighted log-log slope of xi against one control variable."""
    x = np.log(np.array([getattr(p, var) for p in points], float))
    y = np.log(np.array(values, float))
    sy = np.array(errs, float) / np.array(values, float)
    w = 1.0 / np.maximum(sy, 1e-3) ** 2
    Xd = np.vstack([x, np.ones_like(x)]).T
    Wm = np.diag(w)
    C = np.linalg.inv(Xd.T @ Wm @ Xd)
    b = C @ Xd.T @ Wm @ y
    return float(b[0]), float(np.sqrt(C[0, 0])), len(x)


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f0", type=float, default=0.25, help="base frequency [THz]")
    p.add_argument("--W0", type=float, default=30.0, help="base width [nm]; must\n                   satisfy W0 <= margin*v/(2*f0) -- 35 nm FAILS at 0.25 THz")
    p.add_argument("--rms0", type=float, default=1.0, help="base roughness [nm]")
    p.add_argument("--Lc", type=float, default=5.0, help="roughness correlation length [nm]")
    p.add_argument("--h", type=float, default=1.0, help="grid spacing [nm]")
    p.add_argument("--n-rms", type=int, default=5)
    p.add_argument("--n-W", type=int, default=5)
    p.add_argument("--n-f", type=int, default=5)
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--lengths", type=int, default=6)
    p.add_argument("--margin", type=float, default=0.8,
                   help="single-mode headroom: f <= margin * v/(2W)")
    p.add_argument("--tol", type=float, default=0.15,
                   help="exponent agreement required to pass")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", type=Path, default=Path("calibration.json"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--len-cap", type=float, default=250_000.0,
                   help="hard cap on chain length [nm]; points whose 3*xi_law "
                        "exceeds it are flagged EXTRAPOLATIVE in the output")
    p.add_argument("--skip-gate", action="store_true",
                   help="bypass the grid-convergence gate (NOT recommended)")
    args = p.parse_args()

    if not (args.dry_run or args.skip_gate):
        ok, _ = convergence_gate(verbose=True)
        if not ok:
            raise SystemExit("\nABORT: convergence gate failed. The solver "
                             "is not grid-converged; scanning would produce "
                             "artifacts. Fix rgf2d.py first or (against "
                             "advice) pass --skip-gate.")
        print()

    if args.seeds < 20:
        print(f"WARNING: {args.seeds} seeds is below the recommended 20; "
              f"mesoscopic fluctuations in ln T are O(1).", file=sys.stderr)
    if args.lengths < 5:
        print(f"WARNING: {args.lengths} lengths is below the recommended 5.",
              file=sys.stderr)

    print("building grid (single-mode enforced)")
    pts = build_grid(args)
    print(f"  {len(pts)} parameter points\n")

    jobs = []
    for pt in pts:
        xg = min(born_law_xi(pt.f, pt.W, pt.rms, pt.Lc), args.len_cap / 3.0)
        for nz in choose_lengths(xg, pt.n_len, pt.h):
            for s in range(pt.n_seed):
                jobs.append((pt, nz, s))

    ny = int(np.ceil(max(p_.W for p_ in pts) / args.h)) + 6
    flops = sum(j[1] for j in jobs) * ny ** 3
    layers = sum(j[1] for j in jobs)
    # The solve is Python-loop bound, not flop bound: measure per-layer cost
    # rather than counting flops, which underestimated the runtime by >100x.
    per_layer_s = 2.5e-4
    est = layers * per_layer_s / max(args.workers, 1)
    print(f"  {len(jobs)} transmission solves, {layers/1e6:.1f}M layers, "
          f"ny_max = {ny}")
    print(f"  ~{flops/1e9:.0f} Gflop, but runtime is loop-bound: "
          f"~{per_layer_s*1e3:.2f} ms/layer measured")
    print(f"  estimate {est/60:.0f} min on {args.workers} workers "
          f"(single core: {layers*per_layer_s/60:.0f} min)\n")
    if args.dry_run:
        for pt in pts:
            xg = min(born_law_xi(pt.f, pt.W, pt.rms, pt.Lc), args.len_cap / 3.0)
            flag = "  [EXTRAPOLATIVE: 3*xi_law > len-cap]" \
                if born_law_xi(pt.f, pt.W, pt.rms, pt.Lc) > args.len_cap / 3.0 else ""
            print(f"  f={pt.f:.4f} W={pt.W:6.2f} rms={pt.rms:.4f}  "
                  f"xi_law={born_law_xi(pt.f,pt.W,pt.rms,pt.Lc)/1000:7.2f} um  "
                  f"L={[round(n*pt.h/1000,2) for n in choose_lengths(xg,pt.n_len,pt.h)]} um{flag}")
        return

    done = {}
    if args.resume and args.resume.exists():
        done = {k: v for k, v in json.loads(args.resume.read_text())["points"].items()}
        print(f"resuming: {len(done)} points already complete")

    # cheapest points first, so progress appears within seconds rather than
    # after the single slowest point in the grid
    pts = sorted(pts, key=lambda q: ziman_xi(q.f, q.W, q.rms) * q.W ** 2)
    todo = [pt for pt in pts if pt.key() not in done]
    print(f"running {len(todo)} points, cheapest first\n")
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, pt in enumerate(todo, 1):
            xg = min(born_law_xi(pt.f, pt.W, pt.rms, pt.Lc), args.len_cap / 3.0)
            sub = [(pt, nz, s) for nz in choose_lengths(xg, pt.n_len, pt.h)
                   for s in range(pt.n_seed)]
            res, t1 = [], time.time()
            for k, r in enumerate(pool.imap_unordered(_one, sub, chunksize=1), 1):
                res.append(r)
                if k % max(len(sub) // 4, 1) == 0 or k == len(sub):
                    print(f"    {pt.key()}  {k}/{len(sub)} solves "
                          f"({time.time()-t1:.0f}s)", flush=True)
            fit = fit_point(pt, res)
            done[pt.key()] = dict(**asdict(pt), **fit)
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            tag = (f"xi={fit['xi_nm']/1000:6.2f}+/-{fit['xi_err_nm']/1000:.2f} um"
                   f"  R2={fit['r2']:.3f}  N_prop={fit['n_prop']}"
                   if fit["ok"] else f"FAILED: {fit['reason']}")
            print(f"[{i}/{len(todo)}] {pt.key()}  {tag}   ETA {eta/60:.0f} min",
                  flush=True)
            args.out.write_text(json.dumps(dict(args=vars(args) | {"out": str(args.out),
                                                                  "resume": str(args.resume)},
                                                points=done), indent=1, default=str))

    # ------------------------------------------------------------ exponents
    print("\n" + "=" * 66)
    print("SCALING EXPONENTS (RGF anchor vs Ziman)")
    print("=" * 66)
    ok = {k: v for k, v in done.items() if v.get("ok")}
    verdict = {}
    q0 = 2 * np.pi * args.f0 / V_NM_PS
    f_target = -2.0 + 4.0 * q0 * q0 * args.Lc ** 2
    for var, base_attr, target in [("rms", "rms", -2.0), ("W", "W", +2.0),
                                   ("f", "f", f_target)]:
        sel = [v for v in ok.values()
               if all(abs(v[a] - getattr(args, a + "0")) < 1e-9
                      for a in ("f", "W", "rms") if a != base_attr)]
        if len(sel) < 3:
            print(f"  {var:>4}: only {len(sel)} points — cannot fit")
            verdict[var] = None
            continue
        P = [Point(v["f"], v["W"], v["rms"], v["Lc"], v["h"],
                   v["n_seed"], v["n_len"]) for v in sel]
        e, se, n = fit_exponent(P, [v["xi_nm"] for v in sel],
                                [v["xi_err_nm"] for v in sel], base_attr)
        dev = abs(e - target)
        verdict[var] = dev <= args.tol
        print(f"  d ln xi / d ln {var:<4} = {e:+.3f} +/- {se:.3f}   "
              f"(Born law {target:+.2f}, |dev| = {dev:.3f})  "
              f"{'PASS' if verdict[var] else 'FAIL'}   n={n}")

    ratios = [v["xi_nm"] / v["xi_ziman_nm"] for v in ok.values()]
    law = [v["xi_nm"] / born_law_xi(v["f"], v["W"], v["rms"], v["Lc"])
           for v in ok.values()]
    print(f"\n  xi_RGF / xi_Born-law: median {np.median(law):.3f}, "
          f"range {min(law):.3f}-{max(law):.3f}")
    print(f"  xi_RGF / xi_Ziman   : median {np.median(ratios):.3f}  "
          f"(ray formula; reported for contrast only)")
    passed = all(verdict.get(k) for k in ("rms", "W", "f"))
    print("\n  VERDICT: " + ("CALIBRATION PASSED — the Born-transparency law "
                             "is the roughness anchor for Sec. II."
                             if passed else
                             "CALIBRATION FAILED — do not write Sec. II.6."))
    args.out.write_text(json.dumps(dict(args=vars(args) | {"out": str(args.out),
                                                           "resume": str(args.resume)},
                                        points=done,
                                        exponents=verdict,
                                        ratio_median=float(np.median(ratios)),
                                        passed=bool(passed)),
                                   indent=1, default=str))
    print(f"\n  written -> {args.out}")


if __name__ == "__main__":
    main()
