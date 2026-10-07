#!/usr/bin/env bash
# Convert lab sources (jupytext py:percent) into notebooks and execute them, so the committed
# .ipynb files contain the expected outputs.
#   scripts/build_labs.sh                               # all labs
#   scripts/build_labs.sh labs/03_attention/03_attention.py   # one lab
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/data/hf}"
files=("$@")
[ ${#files[@]} -eq 0 ] && files=(labs/*/[0-9][0-9]_*.py)
for src in "${files[@]}"; do
  nb="${src%.py}.ipynb"
  echo "== $src"
  uv run jupytext --quiet --to ipynb --output "$nb" "$src"
  if [[ "$src" == *14_smolvla* ]]; then          # Lab 14 runs in the separate lerobot environment
    [ -x data/venv-lerobot/bin/jupyter ] || scripts/setup_lerobot_env.sh
    data/venv-lerobot/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"
  else
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"
  fi
done
