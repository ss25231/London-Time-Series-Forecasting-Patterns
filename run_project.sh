#!/usr/bin/env bash
# =============================================================================
#  run_project.sh  —  one command: install dependencies, then run everything.
#
#  Usage (from the project root):
#      bash run_project.sh
#      bash run_project.sh --quick        # fast pass, small bootstrap
#
#  It will:
#    1. install the required Python packages
#    2. run the whole pipeline via run_all.py (all stages, in order)
#    3. print a timed summary at the end
# =============================================================================
set -e

# pick an interpreter: prefer python3, fall back to python
if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi

echo
echo "============================================================"
echo "  STEP 1 of 2 — installing Python dependencies"
echo "============================================================"
$PY -m pip install --upgrade pip
$PY -m pip install pandas numpy scikit-learn lightgbm matplotlib \
    pyarrow python-docx openpyxl python-pptx

echo
echo "============================================================"
echo "  STEP 2 of 2 - running the full pipeline (prep, model, forecast, validation, sensitivity, graphs)"
echo "============================================================"

# run_all.py finds scripts in ./scripts or alongside itself.
if [ -f "scripts/run_all.py" ]; then
    $PY scripts/run_all.py "$@"
else
    $PY run_all.py "$@"
fi

echo
echo "============================================================"
echo "  Done. See the summary above and pipeline_runtime.txt."
echo "============================================================"
