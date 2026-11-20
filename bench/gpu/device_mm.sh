#!/usr/bin/env bash
# Native build + run of the device-buffer effect.
set -euo pipefail
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
cd "$(dirname "$0")"
bend device_mm.bend -o /tmp/devmm
/tmp/devmm
