#!/usr/bin/env bash
# Optional: install PyTorch for ROCm (AMD GPU) into .venv and check the GPU is visible.
# A speed upgrade only: the CPU path already runs the full graph above Mario's 60 Hz.
# Targets AMD RDNA4 + ROCm; there is no CUDA, so CuPy-based code will not run.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Installing PyTorch for ROCm (~3 GB) ..."
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/rocm6.4 torch
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| hip", torch.version.hip)
print("GPU visible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
