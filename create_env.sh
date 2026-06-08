#!/usr/bin/env bash
# Creates the 'artscot' conda environment and installs all local packages.
# Run from the repo root: bash create_env.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DPU_MINI_COMMIT="d359609428b209ed62132b05766e4bebbda0de98"

echo "==> Initialising git submodules..."
git -C "$REPO_ROOT" submodule update --init --recursive
git -C "$REPO_ROOT/prf_packages/dpu_mini" checkout "$DPU_MINI_COMMIT"

echo "==> Creating conda environment from environment.yml..."
conda env create -f "$REPO_ROOT/environment.yml"

# Resolve the python executable inside the new env
CONDA_BASE=$(conda info --base)
PYTHON="$CONDA_BASE/envs/artscot/bin/python"

echo "==> Installing local packages in editable mode..."
"$PYTHON" -m pip install -e "$REPO_ROOT/prf_packages/dpu_mini"
"$PYTHON" -m pip install -e "$REPO_ROOT/prf_packages/prfpy_csenf"
"$PYTHON" -m pip install -e "$REPO_ROOT/analysis"

echo "==> Registering Jupyter kernel..."
"$PYTHON" -m ipykernel install --user --name artscot --display-name "artscot"

echo ""
echo "Done. Activate with:  conda activate artscot"
