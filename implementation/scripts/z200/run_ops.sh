#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/setup_env.sh"
RESULTS=${1:-$WORK_HOME/h3-z200/results}
mkdir -p "$RESULTS"
python -u "$HERE/run_suite.py" --output "$RESULTS"
echo Z200_OPERATOR_SUITE_OK
