#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
cd "$(dirname "$0")/.."
for e in 256 512; do
  for lr in 1.0 2.0; do
    python3 - "$e" "$lr" <<'PY'
import sys
e,lr=sys.argv[1],sys.argv[2]
s=open("train/mlp_batch.bend").read().replace("train.n(64n","train.n(%sn" % e).replace("+lr : F32 = 0.2","+lr : F32 = %s" % lr)
open("train/_mb.bend","w").write(s)
PY
    bend train/_mb.bend -o /tmp/mb >/dev/null 2>&1
    printf '%s,%s: ' "$e" "$lr"; /tmp/mb
  done
done
rm -f train/_mb.bend
