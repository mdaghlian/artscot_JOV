# Artificial Scotoma — JOV

Code for the experiment and analyses accompanying the artificial scotoma paper.

Participants viewed standard drifting-bar retinotopic stimuli (AS0) with a mean-luminance patch — an artificial scotoma — overlaid at two sizes and locations (AS1, AS2). Population receptive field (pRF) positions are compared across conditions to measure cortical remapping, and two neural models (isotropic Gaussian and Divisive Normalisation) are evaluated against each other.

## Task conditions

| ID  | Description | Scotoma radius | Centre (visual degrees) |
|-----|-------------|---------------|------------------------|
| AS0 | No scotoma (baseline) | — | — |
| AS1 | Small scotoma, offset | ~0.83° | [0.83, 0.83] |
| AS2 | Large scotoma, centred | ~2.0° | [0, 0] |

## Data

Pre-processed data are archived on Zenodo: https://zenodo.org/records/19632556

Download the `prf_for_pub/` folder and place it at the repo root before running the notebooks.
Seven subjects (`sub-01` … `sub-07`) are included, each with the same file set.

### Directory layout

```
prf_for_pub/
  sub-01/
  │   sub-01_roi.npz           ← ROI masks (bool, 617 224 vertices = both hemispheres)
  │   sub-01_us_factor.npy     ← scalar: vol→surface upsampling factor (used for permutation tests)
  └── ses-1/
      ├── *_desc-avg_bold.npy      ← PSC time courses  (3 files, one per task)
      ├── *_mean_epi.npy           ← mean EPI signal per vertex  (3 files)
      ├── *_run_correlation.npy    ← inter-run correlation per vertex  (3 files)
      ├── *_stage-grid_*.pkl       ← grid-search pRF fit  (6 files: 2 models × 3 tasks)
      └── *_stage-iter_*.pkl       ← iterative pRF fit  (6 files: 2 models × 3 tasks)
```

### File types

#### `*_desc-avg_bold.npy`

Percent signal change (PSC) time courses for one task, averaged across the two runs.
Shape: **(617 224 vertices × 220 TRs)**. The first 5 TRs were discarded during preprocessing
(scanner settling / scotoma onset artefact), leaving 220 of the original 225 TRs.

#### `*_mean_epi.npy`

Mean BOLD signal per surface vertex before PSC conversion.

#### `*_run_correlation.npy`

Pearson correlation between the two runs for each vertex after PSC.

#### `*_stage-grid_desc-prf_params.pkl` and `*_stage-iter_desc-prf_params.pkl`

Python pickled dicts with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `pars` | ndarray | Fitted parameters |
| `settings` | dict | All fitting hyperparameters from `s0_prf_analysis.yml` plus job metadata |
| `preds` | ndarray | Predicted time courses, shape ; iter files only |
| `start_time` / `end_time` | str | Fitting timestamps (`YYYY-MM-DD_HH-MM`) |
| `prfpy_model` | object | Fitted prfpy model object; norm iter files only |

The `pars` array columns follow the `prfpy_params_dict()` order defined in
[`prfpy_functions.py`](analysis/artscot_JOV/prfpy_functions.py):

**Gaussian model** (`n_params = 8`):

| Col | Name | Notes |
|-----|------|-------|
| 0 | `x` | pRF centre, visual degrees |
| 1 | `y` | pRF centre, visual degrees |
| 2 | `size_1` | pRF sigma, visual degrees |
| 3 | `amp_1` | response amplitude |
| 4 | `bold_baseline` | fixed to 0 |
| 5 | `hrf_deriv` | fixed to 1 |
| 6 | `hrf_disp` | fixed to 0 |
| 7 | `rsq` | variance explained (R²) |

**Divisive Normalisation model** (`n_params = 12`):

| Col | Name | Notes |
|-----|------|-------|
| 0 | `x` | pRF centre, visual degrees |
| 1 | `y` | pRF centre, visual degrees |
| 2 | `size_1` | excitatory pRF sigma |
| 3 | `amp_1` | excitatory amplitude |
| 4 | `bold_baseline` | fixed to 0 |
| 5 | `amp_2` | suppressive amplitude |
| 6 | `size_2` | suppressive pRF sigma |
| 7 | `b_val` | neural baseline |
| 8 | `d_val` | suppressive baseline |
| 9 | `hrf_deriv` | fixed to 1 |
| 10 | `hrf_disp` | fixed to 0 |
| 11 | `rsq` | variance explained (R²) |

Grid files contain only `pars` and `settings` (no `preds`); parameter
ranges may include `-inf`/`nan` for vertices excluded from the grid search.
The iterative fit is what the notebooks and analysis scripts use.

#### `sub-XX_roi.npz`

Boolean surface masks for 9 ROIs, each a flat array of **617 224 vertices**
(left + right hemispheres concatenated):

| Key | ROI |
|-----|-----|
| `all` | every vertex (all True) |
| `v1custom` | V1 |
| `v2custom` | V2 |
| `v3custom` | V3 |
| `v3abcustom` | V3A/B |
| `v4custom` | V4 |
| `LOcustom` | Lateral Occipital |
| `TOcustom` | Temporal Occipital |
| `IPScustom` | Intraparietal Sulcus |

ROIs were drawn manually on each subject's FreeSurfer surface and saved as
boolean arrays aligned to the fsnative surface space.

#### `sub-XX_us_factor.npy`

Scalar float. The ratio of surface vertices to volumetric voxels for this
subject (≈ 5.4 for sub-01). Used to correct for the non-independence of
surface samples when computing cortical permutation tests in
`artscot_plots.ipynb`.

---

## Setup

### 1. Clone with submodules

```bash
git clone https://github.com/mdaghlian/artscot_JOV.git
cd artscot_JOV
```

### 2. Create the conda environment

A single script handles everything — it creates the `artscot` conda environment, pins the `dpu_mini` submodule to the tested commit, installs all three local packages in editable mode, and registers the Jupyter kernel.

```bash
bash create_env.sh
conda activate artscot
```

Dependencies installed:
- Python 3.10, numpy, scipy, matplotlib, seaborn, pandas, nibabel, joblib, statsmodels, pyyaml, jupyterlab
- `prf_packages/prfpy_csenf` — pRF modelling and fitting ([spinoza-centre/prfpy_csenf](https://github.com/spinoza-centre/prfpy_csenf))
- `prf_packages/dpu_mini` — general MRI/surface utilities ([mdaghlian/dpu_mini](https://github.com/mdaghlian/dpu_mini))
- `analysis/` — this repo's `artscot_JOV` package (editable)

### 3. Open a notebook

```bash
conda activate artscot
cd analysis/r0_reports
jupyter lab
```

Select the **artscot** kernel when prompted.

---

## Repository structure

```
artscot_JOV/
├── analysis/
│   ├── artscot_JOV/            ← Python package (importable as artscot_JOV)
│   │   ├── utils.py            ← scotoma geometry and time-series helpers
│   │   ├── load_saved_info.py  ← data loading: time courses, pRF fits, stimuli, ROIs
│   │   ├── prfpy_functions.py  ← Prf1T1M / PrfMulti / PrfDiff classes
│   │   ├── plot_functions.py   ← colour palettes and plotting helpers
│   │   └── prfpy_ts_plotter.py ← time-series plotting utilities
│   ├── r0_reports/
│   │   ├── artscot_plots.ipynb        ← main results notebook
│   │   └── art_scot_simulations.ipynb ← simulation validation notebook
│   └── s0_analysis_steps/
│       ├── s0_prf_analysis.yml ← fitting hyperparameters
│       ├── s1_psc.py           ← step 1: percent signal change
│       ├── s2_G_fit.py         ← step 2: Gaussian pRF fit
│       ├── s3_N_fit.py         ← step 3: Normalisation model fit
│       ├── s1_qsub_psc.py      ← cluster submission wrapper for s1
│       ├── s2_qsub_G.py        ← cluster submission wrapper for s2
│       └── s3_qsub_N.py        ← cluster submission wrapper for s3
├── prf_packages/
│   ├── dpu_mini/               ← git submodule (pinned)
│   └── prfpy_csenf/            ← git submodule
├── prf_for_pub/                ← downloaded data (not tracked)
├── experiment/                 ← stimulus delivery code
├── environment.yml
└── create_env.sh
```

---

## Analysis pipeline

The pipeline runs in three sequential steps, followed by notebook-based analysis.

### Preprocessing

Raw BOLD data are preprocessed using the [linescanning](https://github.com/gjheij/linescanning) pipeline (not included here). The output is a per-subject, per-task `.npy` file containing the surface-projected time series (vertices × TRs), stored in `prf_for_pub/<sub>/ses-1/`.

### Step 1 — Percent signal change (`s1_psc.py`)

Converts BOLD time courses to percent signal change. Baseline is estimated from the first 19 TRs of each run. The two runs per task are averaged, and the first 5 TRs (screen-change artefact) are discarded, leaving 220 TRs.

```bash
python s1_psc.py --sub 01 --task AS0 --prf_out /path/to/derivatives
```

Or via the cluster wrapper: `python s1_qsub_psc.py`

### Step 2 — Gaussian pRF fit (`s2_G_fit.py`)

Fits an isotropic 2D Gaussian pRF model to the PSC time courses using `prfpy_csenf`. Grid search followed by iterative optimisation (L-BFGS-B). Fitting settings are read from `s0_prf_analysis.yml`.

```bash
python s2_G_fit.py --sub 01 --task AS0 --roi_fit all
```

Key parameters from `s0_prf_analysis.yml`:

| Parameter | Value |
|-----------|-------|
| Screen size | 39.3 cm at 196 cm |
| Visual field extent | ±5° |
| TR | 1.5 s |
| Grid points | 10 |
| R² threshold | 0.1 |

### Step 3 — Normalisation model fit (`s3_N_fit.py`)

Fits the Divisive Normalisation (DN) model, initialised from the Gaussian fit of the same task. Supports `norm`, `css`, and `dog` models via `--model`.

```bash
python s3_N_fit.py --sub 01 --task AS0 --model norm
```

---

## Notebooks

Figures are saved as `.png` files to the output directory set at the top of each notebook (`fig_output`).

### `artscot_plots.ipynb` — main results

Loads the pre-computed pRF fit files from `prf_for_pub/` and produces all main paper figures.

| Section | What it does |
|---------|-------------|
| **Load** | Reads pRF parameters for all subjects and tasks into `prf_obj` (a dict of `PrfMulti` objects) |
| **Timeseries plots** | Example vertex time series overlaid with model predictions (`ts_plot_eg2.png`) |
| **Vector plots** | Arrow plots showing per-vertex pRF position shifts between AS0→AS1 and AS0→AS2, split by ROI and model (`vector_plot*.png`) |
| **Paired bar** | Mean pRF shift magnitude per ROI, Gaussian vs DN, with paired t-test overlays |
| **Stats** | Cortical permutation test comparing Gaussian and DN shift magnitudes across ROIs |
| **Nonlinearities** | DN suppression parameters (b, d, amplitude ratio) per ROI (`param_roi_plot.png`) |
| **Correlation** | Scatter of DN amplitude ratio vs Gaussian–DN shift difference (`amp_ratio_dshift_corr.png`) |

Set `fig_output` at the top of the notebook to control where figures are saved (default: `./figures`).

### `art_scot_simulations.ipynb` — simulation validation

Validates that the shift-magnitude analysis can distinguish Gaussian-generated from DN-generated data when both are fit with a 1-Gaussian model.

| Section | What it does |
|---------|-------------|
| **Simulate** | Generates synthetic pRF grids under both models (`sim_gridG`, `sim_gridN`) for each task; saves ground-truth and predicted time series to `notebook_timeline/` |
| **Fit** | Fits a 1-Gaussian model to both sets of synthetic time courses |
| **Arrow plots** | Visual field arrows for simulated Gaussian vs DN fits (`sim_arrows*.png`) |
| **Shift summary** | Bar chart of mean shift magnitude per model type (`mean_shift_sim.png`) |
| **Correlations** | Correlation matrices between simulation parameters |

Intermediate results (ground-truth grids, fit objects) are pickled to `notebook_timeline/` so the fitting step can be skipped on subsequent runs.

---

## Code reference

### `artscot_JOV/prfpy_functions.py`

Contains the three main data-container classes used throughout the notebooks:

- **`Prf1T1M`** — holds pRF parameters for one subject × one task × one model. Provides `return_vx_mask(th)` for threshold-based voxel selection, plus `hist`, `scatter`, `visual_field`, and `arrow` plot methods.
- **`PrfMulti`** — aggregates several `Prf1T1M` objects (e.g., all tasks and models for one subject). Keys are `'{task}f_{model}'` strings. Supports cross-condition comparisons via `return_diff_params` and `add_prf_diff`.
- **`PrfDiff`** — stores the parameter-wise difference between two `Prf1T1M` objects; used internally by `PrfMulti.add_prf_diff`.

Threshold dict convention (used throughout):
```python
th = {
    'AS0f_G-min-rsq': 0.3,   # R² > 0.3 for Gaussian fit of AS0
    'AS1f_G-max-ecc': 5.0,   # eccentricity < 5° for AS1 Gaussian fit
    'roi': boolean_array,     # optional surface ROI mask
}
```

### `artscot_JOV/load_saved_info.py`

Handles all file I/O. Key functions:

- `load_data_tc(sub, task_list)` — load PSC time courses
- `load_data_prf(sub, task_list, model_list)` — load pRF fit `.pkl` files
- `get_scotoma_info(sub)` — returns scotoma centre, radius, and pixel grid for each task
- `get_prfpy_stim(sub, task_list)` — construct `PRFStimulus2D` objects from design matrices
- `get_roi(sub, label)` — boolean surface mask for a named ROI
