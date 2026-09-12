# Changelog

All notable changes to this repository are documented here.

## [1.0.0] - 2026-09-12

### Added

- Professional repository structure:
  - `paper/` for manuscript files
  - `src/` for reproduction and audit code
  - `figures/` for generated and manuscript figures
  - `tables/` for CSV result tables
  - `dashboard/` for the interactive research dashboard
- `requirements.txt` with the Python dependencies used by the reproduction scripts.
- `LICENSE` with the MIT License.
- `CITATION.cff` for software citation.
- Reproduction instructions and repository-structure documentation.

### Changed

- Reorganized source code, figures, tables, and manuscript files into dedicated folders.
- Updated dashboard and documentation paths after repository reorganization.
- Identified `scientific_audit_F_corrected.py` as the current Section 7 analysis.
- Identified the pooled-OLS analysis in `scientific_audit_E_F.py` as superseded.

### Scientific status

- This repository contains a synthetic-data proof-of-concept study.
- The results are not a validation of live quantum hardware.
- The main reported results include the approximately −2.5% low-T1 Osprey effect and the approximately +19.7% high-T1 Sherbrooke effect.
- The corrected Section 7 per-seed-slope analysis supersedes the original pooled-OLS inference.

## [0.1.0] - 2026-09-12

### Added

- Initial manuscript, source code, figures, tables, and dashboard.
- Reproduction and scientific audit scripts.
- Multi-seed robustness analysis and sensitivity analysis.
