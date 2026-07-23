# analysis/

This directory contains the `artscot_JOV` Python package, the fitting scripts, and the report notebooks.

See the top-level [README](../README.md) for setup instructions and a full project overview.

## Layout

```
analysis/
├── artscot_JOV/            ← installable Python package
│   ├── utils.py            ← scotoma geometry, time-series helpers
│   ├── load_saved_info.py  ← I/O: time courses, pRF fits, stimuli, ROIs
│   ├── prfpy_functions.py  ← Prf1T1M / PrfMulti / PrfDiff classes
│   ├── plot_functions.py   ← colour palettes and figure helpers
│   └── prfpy_ts_plotter.py ← time-series plotting utilities
├── r0_reports/
│   ├── artscot_plots.ipynb        ← main results notebook
│   └── art_scot_simulations.ipynb ← simulation validation notebook
└── s0_analysis_steps/
    ├── s0_prf_analysis.yml ← fitting hyperparameters
    ├── s1_psc.py           ← percent signal change
    ├── s2_G_fit.py         ← Gaussian pRF fit
    ├── s3_N_fit.py         ← Divisive Normalisation fit
    └── s{1,2,3}_qsub_*.py  ← SGE cluster submission wrappers
```

## Running the pipeline

Steps must run in order. The cluster wrappers (`s*_qsub_*.py`) submit jobs via `qsub` (Sun Grid Engine); the plain scripts can also be run directly.

```bash
# Step 1 — percent signal change
# python s0_analysis_steps/s1_psc.py --sub 01 --task AS0
# NOT NEEDED AS THE DATA IS ALREADY IN PSC

# Step 2 — Gaussian pRF fit
python s0_analysis_steps/s2_G_fit.py --sub 01 --task AS0

# Step 3 — Normalisation model fit
python s0_analysis_steps/s3_N_fit.py --sub 01 --task AS0 --model norm
```

Repeat for each subject (`01`–`N`) and each task (`AS0`, `AS1`, `AS2`).

## Notebooks

Open with `jupyter lab` from this directory after activating the `artscot` environment. Both notebooks save figures to `fig_output` (set at the top of each notebook).

- **`artscot_plots.ipynb`** — loads published fit files and produces all main paper figures (vector plots, bar charts, nonlinearity parameters, correlation plots).
- **`art_scot_simulations.ipynb`** — generates synthetic pRF grids under Gaussian and DN models, fits a 1-Gaussian model to both, and checks that the shift analysis can distinguish the two.
