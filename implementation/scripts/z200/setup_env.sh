#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../z100/setup_env_z100.sh"
# Triton 3.8's HIP property struct differs from DTK 26.04's ABI.
# Query metadata through a host-only helper built with DTK's own headers.
export H3_DTK_TRITON_COMPAT=1
export TRITON_LIBHIP_PATH=/opt/dtk/lib/libamdhip64.so
export H3_GFX906_PORTABLE_BF16=1
