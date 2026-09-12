# Physics-Anchored vs. Data-Driven Extrapolation of Superconducting-Qubit Dephasing Rates

Synthetic-data proof-of-concept study, Revision 4. Went through four revisions, three
triggered by catching real errors (two citation errors, one critical statistical bug
that had hidden a significant effect in Section 7).

## Structure
- `paper/` — the manuscript (docx)
- `src/` — all reproduction code (see src/CODE_README.md for run order)
- `figures/` — every figure in the paper
- `tables/` — key result tables as CSV
- `dashboard/index.html` — open in any browser, no server needed

## Headline results
| Test point | Effect | Significance |
|---|---|---|
| ibm_osprey (low-T1) | −2.5% ± 17.0% alone | not significant (p=0.514); **significant (p=0.021)** via corrected §7 test |
| ibm_sherbrooke (high-T1) | +19.7% ± 6.4% | significant, p<0.0001 |
| Sensitivity sweep | 21/21 conditions consistent | no exception |

See the manuscript's Appendix E for what did and didn't reproduce.
