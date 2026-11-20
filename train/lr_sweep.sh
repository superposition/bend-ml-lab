#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
cd "$(dirname "$0")/.."
for lr in 0.01 0.05 0.1 0.2; do
  python3 - "$lr" <<'PY'
import sys
lr=sys.argv[1]
s=open("train/linear_batch.bend").read().replace("+lr : F32 = 0.5","+lr : F32 = %s" % lr).replace("train.n(40n","train.n(5n")
open("train/_lr.bend","w").write(s)
PY
  bend train/_lr.bend -o /tmp/lr >/dev/null 2>&1
  printf '%s: ' "$lr"
  /tmp/lr
done
rm -f train/_lr.bend
