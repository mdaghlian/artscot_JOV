Stages of PRF scotoma analyses
===============================

Preprocessing
-------------
Raw BOLD data must first be processed to produce surface-projected .npy
time-series files in fsnative space. This step is external to this repo.


[s1] s1_psc.py  —  percent signal change
Note required already done for data provided - but included for completeness
  - Average time courses across runs (run-1, run-2)
  - Baseline to 0 using the first 19 TRs
  - Discard first 5 TRs (screen-change artefact) → 220 TRs retained


[s2] s2_G_fit.py  —  Gaussian pRF fit
  Fit an isotropic 2D Gaussian pRF model to the PSC time courses.
  Settings read from s0_prf_analysis.yml.


[s3] s3_N_fit.py  —  extended model fit (norm / css / dog)
  Fit an extended pRF model, initialised from the Gaussian fit.


Notebooks — analysis & figures
--------------------------------
See analysis/r0_reports/ and the top-level README for details.
  artscot_plots.ipynb        — main paper figures
  art_scot_simulations.ipynb — simulation validation
