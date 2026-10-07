#!/usr/bin/env bash
# A separate environment for Lab 14: lerobot pins older numpy/transformers than the course environment,
# so it gets its own virtualenv in data/venv-lerobot (git-ignored). torch and torchvision come first from
# the CUDA 12.8 index (Blackwell GPUs need cu128 builds; mixing PyPI and cu128 builds breaks torchvision).
#   scripts/setup_lerobot_env.sh
#   data/venv-lerobot/bin/jupyter lab labs/14_smolvla/14_smolvla.ipynb
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=data/venv-lerobot
PY=$VENV/bin/python
uv venv "$VENV" --python 3.12
uv pip install --python "$PY" "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" "lerobot[smolvla,dataset]==0.6.1" ipykernel jupyterlab matplotlib jupytext nbconvert -e .
"$PY" -c "import torch, torchvision, lerobot; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| CUDA', torch.cuda.is_available())"
