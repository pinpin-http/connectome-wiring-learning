#!/usr/bin/env bash
# Server installer for SMB_Flywire_Speedrun on Linux + NVIDIA (Vast.ai, Runpod, any SSH host;
# a container is fine: no Docker-in-Docker, no systemd).
#   deployment/flydeploy install            (reads TORCH_VERSION / TORCH_CUDA from deploy.env)
#   TORCH_CUDA=cu126 deployment/install_remote.sh [--cpu-only]
# Relaunchable: every step checks before acting. It never touches the GPU driver, never runs
# the ROCm installer (scripts/install_gpu.sh belongs to the AMD workstation) and never copies
# a venv: it builds a fresh Python 3.11 venv in .venv.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p deployment/state
LOG=deployment/state/install.log
exec > >(tee -a "$LOG") 2>&1
echo "=== install $(date -u +%FT%TZ) in $(pwd)"

TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCH_CUDA="${TORCH_CUDA:-auto}"
CPU_ONLY=0
[ "${1:-}" = "--cpu-only" ] && CPU_ONLY=1
step() { echo; echo "== $*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

step "system"
[ "$(uname -s)" = Linux ] || fail "Linux only"
# getconf, not `ldd --version | head`: under pipefail, head closing the pipe appends the fallback
# value to the real version ("2.39" + "0.0") and rejects a valid system
glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
glibc="${glibc:-0.0}"
gmaj="${glibc%%.*}"; gmin="${glibc##*.}"
if [ "$gmaj" -lt 2 ] || { [ "$gmaj" -eq 2 ] && [ "$gmin" -lt 28 ]; }; then
  fail "glibc $glibc < 2.28: the torch and stable-retro wheels are manylinux_2_28 (use an Ubuntu >= 20.04 image)"
fi
echo "glibc $glibc, $(nproc) CPUs, $(awk '/MemTotal/ {printf "%.0f GB RAM", $2/1e6}' /proc/meminfo), \
$(df -h . | awk 'NR==2 {print $4 " free on " $6}')"

step "system packages (only the missing ones; nothing GPU-related)"
need=()
for b in rsync tmux git curl unzip; do command -v "$b" >/dev/null || need+=("$b"); done
if [ "${#need[@]}" -gt 0 ]; then
  SUDO=""
  if [ "$(id -u)" != 0 ]; then SUDO="sudo -n"; fi
  if command -v apt-get >/dev/null && { [ -z "$SUDO" ] || sudo -n true 2>/dev/null; }; then
    $SUDO apt-get update -qq
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends "${need[@]}" ca-certificates
  else
    echo "WARNING: missing ${need[*]} and no apt/root here: install them yourself (tmux is optional: runs then use setsid)"
  fi
else
  echo "rsync tmux git curl unzip present"
fi

step "NVIDIA driver (read only)"
if [ "$CPU_ONLY" = 0 ]; then
  command -v nvidia-smi >/dev/null || fail "nvidia-smi not found: not a GPU instance? (--cpu-only installs a CPU torch)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  dmaj="${drv%%.*}"
  [ "$dmaj" -ge 525 ] || fail "driver $drv < 525: CUDA 12 wheels need >= 525; the driver is the provider's, pick another host"
  if [ "$TORCH_CUDA" = auto ]; then
    if [ "$dmaj" -ge 570 ]; then TORCH_CUDA=cu128; else TORCH_CUDA=cu126; fi
  fi
  INDEX="https://download.pytorch.org/whl/$TORCH_CUDA"
  WANT="$TORCH_VERSION+$TORCH_CUDA"
else
  INDEX="https://download.pytorch.org/whl/cpu"
  WANT="$TORCH_VERSION+cpu"
fi
echo "torch target $WANT from $INDEX"

step "uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

step "Python 3.11 venv (.venv, built here)"
uv python install 3.11
if [ -e .venv ] && ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' 2>/dev/null; then
  fail ".venv exists but is not a working Python 3.11 venv (copied from another machine?): remove it and rerun"
fi
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
.venv/bin/python --version

step "PyTorch $WANT"
have="$(.venv/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
if [ "$have" = "$WANT" ]; then
  echo "already installed: torch $have"
else
  uv pip install --python .venv/bin/python --index-url "$INDEX" "torch==$TORCH_VERSION"
fi

step "dependencies (requirements.txt pins) and the project"
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
have="$(.venv/bin/python -c 'import torch; print(torch.__version__)')"
if [ "$have" != "$WANT" ]; then      # a dependency must not have replaced the CUDA build
  echo "torch became $have: reinstalling $WANT"
  uv pip install --python .venv/bin/python --reinstall --no-deps --index-url "$INDEX" "torch==$TORCH_VERSION"
fi

step "check"
SDL_VIDEODRIVER=dummy .venv/bin/python - <<'PY'
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy, scipy, sklearn, torch, gymnasium
import retro  # stable-retro
info = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "numpy": numpy.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "gymnasium": gymnasium.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
print(json.dumps(info, indent=1))
open("deployment/state/install.json", "w").write(json.dumps(info, indent=1))
PY
echo
echo "installed. Next: deployment/flydeploy verify ; deployment/flydeploy import-rom ; deployment/flydeploy doctor"
