# Reproduction Code: Physics-Anchored vs. Data-Driven Extrapolation (Revision 4)

No network access or quantum-computing toolkit required for any script.

## Core pipeline
- `noise_simulation.py` — full applied pipeline (Sections 6-7, Appendix A).
- `isolated_mechanism_demo.py` — isolated single-variable Monte Carlo validation (Section 5.1).

## Audit scripts (Revision 3)
- `audit_common.py` — common definitions extracted verbatim from noise_simulation.py.
- `scientific_audit_A_B.py` — Section 4.2 exact-OLS-vs-Taylor; Section 5.2 lognormal check.
- `scientific_audit_C.py` — Section 6.4 paired bootstrap CI / Wilcoxon.
- `scientific_audit_D.py` — Section 6.6 21-condition sensitivity analysis.
- `scientific_audit_E_F.py` — Section 6.5 20-seed interpolation; Section 7 ORIGINAL
  (now superseded/invalid) pooled-OLS trend regression — kept for transparency about
  what changed, NOT as the current method.

## Audit scripts (Revision 4 — critical correction)
- `scientific_audit_F_corrected.py` — Section 7's CORRECTED per-seed-slope trend
  analysis, replacing the invalid pooled-OLS inference in scientific_audit_E_F.py.
  Run this, not the trend-regression portion of E_F, for the current Section 7 numbers.

## Known, disclosed issues from earlier revisions (see manuscript Appendix E)
- An earlier internal draft's Section 6.6 sensitivity table contained one fabricated
  data point (a claimed significant "exception" at feature-noise=16%) that does not
  reproduce; scientific_audit_D.py's actual output is the correct, verified source.
- An earlier internal draft's Section 6.5 interpolation figures were also transcribed
  incorrectly at one point; scientific_audit_E_F.py's actual output is correct.
Both are corrected in the current manuscript; this file exists so a skeptical reader
can re-run the scripts directly rather than trust any narrative description, including
this one.

## Running (in order)
```
python3 isolated_mechanism_demo.py      # Section 5.1, Table 1, Figure 1
python3 scientific_audit_A_B.py         # Section 4.2 Table 1a; Section 5.2 Table 1b
python3 noise_simulation.py             # Sections 6-7, Appendix A
python3 scientific_audit_C.py           # Section 6.4 Table 3b
python3 scientific_audit_D.py           # Section 6.6 Table 3d (21 conditions, verified)
python3 scientific_audit_E_F.py         # Section 6.5 Table 3c (interpolation)
python3 scientific_audit_F_corrected.py # Section 7 Table 4b (CORRECTED per-seed method)
```
All random seeds are fixed and stated in-line; every number in the manuscript is
reproducible by re-running the corresponding script above.
