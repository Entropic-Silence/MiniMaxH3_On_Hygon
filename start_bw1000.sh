#!/usr/bin/env bash
set -euo pipefail
BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-generate}; shift || true
ARCHIVE="$BASE/source/h3_bw1000_source.tar.gz"
echo "cfc760d64eeef6d3b47c16b1be36856a3ecd7e6448183dd7b3e8aa85d1457512  $ARCHIVE" | sha256sum -c -
ROOT=${H3_ROOT:-"$BASE/runtime/h3-extreme"}
if [[ ! -d "$ROOT" ]]; then
  mkdir -p "$BASE/runtime"
  tar -xzf "$ARCHIVE" -C "$BASE/runtime"
fi
set +u
source "${DTK_ENV:-/opt/dtk/env.sh}"
set -u
export INFINI_ROOT=${INFINI_ROOT:-"$ROOT/.infini"}
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/InfiniLM/python:$ROOT/InfiniCore/python:$ROOT/third_party/diffusers/src:$ROOT/ninetoothed/src:$ROOT/tools:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=${H3_OUTPUT:-"$BASE/outputs"}; mkdir -p "$OUT"
if [[ "$ACTION" == build ]]; then
  bash "$ROOT/scripts/build_all.sh" "$@"; exit
fi
: "${H3_MODEL:?Set H3_MODEL to the local MiniMax-H3 model directory}"
if [[ "$ACTION" == compare ]]; then
  exec python3 "$ROOT/InfiniLM/examples/minimax_h3_compare.py" --tasks fl2va     --model-path "$H3_MODEL" --reference-fl2va-video "$OUT/reference.mp4"     --infinilm-fl2va-video "$OUT/optimized.mp4" --output-dir "$OUT/comparison" "$@"
fi
EXTRA=(); NAME=optimized
if [[ "$ACTION" == reference ]]; then EXTRA=(--legacy --legacy-memory-fraction 0.4); NAME=reference
elif [[ "$ACTION" != generate ]]; then echo 'Use build, generate, reference, or compare' >&2; exit 2; fi
cd "$ROOT"
python3 -u InfiniLM/examples/minimax_h3.py fl2va --model-path "$H3_MODEL"   --text "${H3_TEXT:-A dog falls asleep}" --first-frame "${H3_IMAGE:-$BASE/assets/first_frame.png}"   --gpus 0,1,2,3,4,5,6,7 --steps 50 --seed 42 --num-frames 124 --height 768 --width 768   --output "$OUT/$NAME.mp4" "${EXTRA[@]}" "$@" 2>&1 | tee "$OUT/$NAME.log"
