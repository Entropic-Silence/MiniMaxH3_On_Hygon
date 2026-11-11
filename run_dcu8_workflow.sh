#!/usr/bin/env bash
set -euo pipefail
if [[ -e /etc/gitconfig && ! -r /etc/gitconfig ]]; then export GIT_CONFIG_NOSYSTEM=1; fi
BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-generate}; shift || true
ROOT="$BASE/implementation"
if [[ "$ACTION" == fetch || "$ACTION" == build || "$ACTION" == reproduce ]]; then
  python3 "$BASE/fetch_sources.py"
fi
if [[ "$ACTION" == fetch ]]; then exit 0; fi
set +u
source "${DTK_ENV:-/opt/dtk/env.sh}"
set -u
if [[ "$ACTION" == build || "$ACTION" == reproduce ]]; then
  python3 -m venv --without-pip --system-site-packages "$BASE/runtime/venv"
fi
if [[ -f "$BASE/runtime/venv/bin/activate" ]]; then source "$BASE/runtime/venv/bin/activate"; fi
export PYTHON=python3
mkdir -p "$BASE/runtime/cache" "$BASE/runtime/tmp"
export TRITON_CACHE_DIR="$BASE/runtime/cache" TMPDIR="$BASE/runtime/tmp"
export TRITON_CACHE_MANAGER=ifab_triton_cache:AfsCacheManager
export INFINI_ROOT=${INFINI_ROOT:-"$ROOT/.infini"}
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/InfiniLM/python:$ROOT/InfiniCore/python:$ROOT/third_party/diffusers/src:$ROOT/ninetoothed/src:$ROOT/tools:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=${H3_OUTPUT:-"$BASE/outputs"}; mkdir -p "$OUT"
if [[ "$ACTION" == reproduce ]]; then
  bash "$ROOT/scripts/build_all.sh" 2>&1 | tee "$OUT/build.log"
  bash "$BASE/run_dcu8_workflow.sh" generate
  bash "$BASE/run_dcu8_workflow.sh" reference
  bash "$BASE/run_dcu8_workflow.sh" compare
  exit
fi
if [[ "$ACTION" == build ]]; then
  bash "$ROOT/scripts/build_all.sh" "$@"; exit
fi
: "${H3_MODEL:?Set H3_MODEL to the local MiniMax-H3 model directory}"
if [[ "$ACTION" == compare ]]; then
  python3 "$ROOT/InfiniLM/examples/minimax_h3_compare.py" --tasks fl2va     --model-path "$H3_MODEL" --reference-fl2va-video "$OUT/reference.mp4"     --infinilm-fl2va-video "$OUT/optimized.mp4" --output-dir "$OUT/comparison" "$@" 2>&1 | tee "$OUT/comparison.log"
  exit
fi
EXTRA=(); NAME=optimized
if [[ "$ACTION" == reference ]]; then EXTRA=(--legacy --legacy-memory-fraction 0.4); NAME=reference
elif [[ "$ACTION" != generate ]]; then echo 'Use build, generate, reference, or compare' >&2; exit 2; fi
cd "$ROOT"
python3 -u InfiniLM/examples/minimax_h3.py fl2va --model-path "$H3_MODEL"   --text "${H3_TEXT:-A dog falls asleep}" --first-frame "${H3_IMAGE:-$BASE/assets/first_frame.png}"   --gpus 0,1,2,3,4,5,6,7 --steps 50 --seed 42 --num-frames 124 --height 768 --width 768   --output "$OUT/$NAME.mp4" "${EXTRA[@]}" "$@" 2>&1 | tee "$OUT/$NAME.log"
