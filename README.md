# Topological phonon gaps in Fibonacci isotope nanoribbons of h-BN

Theory, calibrated transport code, and an experimental proposal for
Chern-labeled phonon band gaps in single-mode hexagonal boron nitride
nanoribbons carrying a Fibonacci ¹⁰BN/¹¹BN isotope modulation.

**Companion to Paper 1** (hyperuniform isotope phononics):
[hbn-hyperuniform-isotope](https://github.com/Ruqing1963/hbn-hyperuniform-isotope)
· [Zenodo record](https://zenodo.org/records/21981511)

## Core results

1. **Born-transparency law for edge roughness.** In the single-mode wave
   limit (λ ≫ W) a rough nanoribbon edge is far more transparent than the
   Ziman ray formula predicts. The zero-parameter law

   ```
   ξ_edge = 2W² / [q² S_w(2q)],   S_w(2q) = Δr² √(2π) L_c exp(−2q²L_c²)
   ```

   reproduces a 15-point, 24-seed scan in all three scaling exponents
   (−1.91±0.06 / +1.87±0.08 / −1.384±0.095 vs the law's −2 / +2 / −1.38 —
   the frequency exponent to one part in a thousand) and in magnitude to
   3.5% (median ξ/law = 0.965, independently confirmed by a single-bump
   Born test predicting 0.964). Ziman underestimates coherence by a median
   10.8×,
   relaxing the edge-flatness requirement from atomic (≲0.13 nm) to
   lithographic (≲1 nm).

2. **Falsifiable gap labeling.** Twelve gaps labeled with median IDOS
   residual 3×10⁻⁶ against 4×10⁻³ for random controls — a 1385× separation,
   p = 1.4×10⁻³ (`data/gap_table.csv`).

3. **Full counter-propagating phason pumping.** Per-edge spectral flow ±q
   verified for all twelve gaps up to |q| = 11; crossing-moment
   classification data in `data/phason_crossings.csv`. Includes a cautionary
   result: Hungarian eigenvector-overlap tracking silently stitches
   counter-propagating edge branches at their in-gap avoided crossing.

4. **Working point** 0.250 THz, W = 30 nm, d = 24.8 nm, with the (1,−1) gap
   at 0.154 THz as a built-in opposite-Chern-number control. Topology costs
   2.20× in localization length vs a periodic superlattice at the same
   frequency.

## Repository layout

```
code/    transfer_matrix.py       1D Fibonacci chains: Lyapunov, IDOS, gap labels, phason
         rgf2d.py                 2D boundary-conforming RGF anchor (TJC coordinates)
         run_calibration_scan.py  parallel calibration scan (gate-protected)
         phason_tracking.py       overlap tracking + Hungarian assignment
         benchmark_bump.py        analytic Born anchor + mode-matching reference
         plot_fig1.py ... plot_fig4.py
data/    calibration.json         DEFINITIVE 15-point scan (Hermitian operator,
                                  law-based length windows, 24 seeds × 6 lengths)
         calibration_points.csv   per-point ξ vs Ziman vs Born law
         gap_table.csv            Table I (12 labeled gaps)
         phason_crossings.csv     crossing-moment edge classification
paper/   paper2.tex, refs.bib     REVTeX 4.2 source (single-column `preprint`;
                                  switch to `reprint` for the journal layout)
         paper2_preview.pdf       article-class structural preview (see below)
         paper2_zh.tex/.pdf       Chinese version (XeLaTeX + Noto CJK)
         figures/  fig_gamma_idos / fig_phason / fig_survival / fig_experiment .pdf
                                  (numberless names: figure NUMBERS are assigned by LaTeX
                                   at first-citation order and differ from the draft order)
```

## Reproducing the results

```bash
pip install numpy scipy matplotlib
cd code

python transfer_matrix.py        # Table I: gaps, labels, significance
python benchmark_bump.py         # Born anchor + reference self-checks
python rgf2d.py                  # convergence gate (must print GATE PASSED)
python run_calibration_scan.py --workers 16 --seeds 24 --out calibration.json
python plot_fig1.py && python plot_phason.py && python plot_fig3.py && python plot_fig4.py
```

**Two hard-won warnings, encoded in the scripts:**

- **MKL oversubscription.** Every script pins BLAS to one thread *before*
  importing numpy. Under conda/MKL, 16 workers × 16 threads deadlocks a
  16-core machine into apparent hangs. Do not remove the guard.
- **The convergence gate is not optional.** `run_calibration_scan.py` refuses
  to scan unless the solver passes a realization-matched grid-convergence
  test. Two prior implementations produced beautiful, wrong, grid-divergent
  numbers in opposite directions (staircase mask: spurious excess scattering;
  mis-scaled boundary term: spurious transparency). `--skip-gate` exists and
  should not be used.

## Building the paper

The container preview `paper2_preview.pdf` uses an article-class shim.
The official build requires REVTeX 4.2 (standard in TeX Live / MiKTeX):

```bash
cd paper
pdflatex paper2 && bibtex paper2 && pdflatex paper2 && pdflatex paper2   # English
xelatex paper2_zh && bibtex paper2_zh && xelatex paper2_zh && xelatex paper2_zh  # Chinese
```

Bibliography volume/page numbers follow canonical citations; verify DOIs
against publisher records before submission.

## License

MIT — see [LICENSE](LICENSE).

## Citation

R. Wu and R. Chen, *Topological phonon gaps in Fibonacci isotope nanoribbons
of hexagonal boron nitride* (2026). Code and data:
https://github.com/Ruqing1963/hbn-fibonacci-topological-phonons
