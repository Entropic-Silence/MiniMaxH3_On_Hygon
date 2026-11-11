#!/usr/bin/env bash
# Offline setup; files are supplied by the workstation, never replace torch.
set -euo pipefail
ulimit -c 0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/setup_env.sh"
DEPS=${1:?usage: bootstrap.sh /persistent/path/to/dependencies}
if ! command -v xmake >/dev/null 2>&1; then
  sh "$DEPS/xmake-3.0.5.run"
fi
xmake global --network=private
python -m pip install --no-index --find-links "$DEPS" 'pybind11==2.13.6' 'triton==3.8.0'
python -c 'import torch, triton, pybind11; print(torch.__version__, triton.__version__, pybind11.__version__)'
echo BOOTSTRAP_Z200_OK
