#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
cd "$(dirname "$0")/.."
for lr in 0.05 0.1 0.5 1.0; do
  python3 - "$lr" <<'PY'
import sys
lr=sys.argv[1]
s=open("train/mlp_batch.bend").read().replace("+lr : F32 = 0.2","+lr : F32 = %s" % lr)
open("train/_mb.bend","w").write(s)
PY
  bend train/_mb.bend -o /tmp/mb >/dev/null 2>&1
  printf '%s: ' "$lr"
  /tmp/mb
done
rm -f train/_mb.bend
